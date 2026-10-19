//! From input points to dabs: smoothing, brush pose, path walking (spacing, airbrush build-up) and
//! the per-dab dynamics (shape, scattering, transfer, colour, texture depth).
//!
//! Everything is incremental (feed points in any chunking, get the same dabs) and deterministic:
//! jitter comes from [`crate::rng`] keyed by the stroke seed and the step/dab index.

use photocraft_geom::Point;

use crate::brush::{BrushSettings, Control, Dynamic, Smoothing};
use crate::rng::{rand_signed, rand01, stream};
use crate::{Dab, StrokePoint};

/// Maximum pulled-string length at 100 % smoothing, in screen pixels.
pub const MAX_STRING_PX: f64 = 100.0;
/// Stroke time over which exponential smoothing moves its fraction of the way to the pointer.
pub const SMOOTHING_STEP_MS: f64 = 16.0;

/// With Spacing unchecked, one dab per this many milliseconds of stroke time, so faster strokes
/// space their dabs further apart (Photoshop: "the speed of the cursor determines the spacing").
pub const SPEED_SPACING_MS: f64 = 8.0;

/// Longest pixel-grid segment (in pixels) before falling back to distance spacing.
const MAX_GRID_STEPS: f64 = 1e7;

/// Densest speed spacing, in pixels between dabs (a slow, long stroke never floods the buffer).
const MIN_SPEED_STEP: f64 = 0.5;
/// Maximum airbrush catch-up dabs for one input segment.
const MAX_AIRBRUSH_DABS_PER_SEGMENT: usize = 512;

/// The dab centre of the pixel-grid cell holding `v` (see [`PathWalker::pixel_grid`]): the
/// pixel's centre for `offset` 0.5, the nearest pixel corner for 0. A value within rounding
/// noise of a cell boundary counts as on it, so a line lands on the same pixels both ways.
fn grid_cell(v: f64, offset: f64) -> f64 {
    let w = v - offset + 0.5;
    let w = if (w - w.round()).abs() < 1e-7 { w.round() } else { w };
    w.floor() + offset
}

#[inline]
fn lerp_pt(a: &StrokePoint, b: &StrokePoint, f: f64) -> StrokePoint {
    let ff = f as f32;
    StrokePoint {
        x: a.x + (b.x - a.x) * f,
        y: a.y + (b.y - a.y) * f,
        pressure: a.pressure + (b.pressure - a.pressure) * ff,
        tilt_x: a.tilt_x + (b.tilt_x - a.tilt_x) * ff,
        tilt_y: a.tilt_y + (b.tilt_y - a.tilt_y) * ff,
        rotation: a.rotation + (b.rotation - a.rotation) * ff,
        wheel: a.wheel + (b.wheel - a.wheel) * ff,
        time: a.time + (b.time - a.time) * f,
    }
}

// ---------------------------------------------------------------------------
// Smoothing
// ---------------------------------------------------------------------------

/// Incremental stroke smoothing: exponential (with optional time-based catch-up) or pulled string.
#[derive(Clone, Debug)]
pub struct Smoother {
    cfg: Smoothing,
    zoom: f64,
    pos: Option<StrokePoint>,
    last_input: Option<StrokePoint>,
}

impl Smoother {
    /// `zoom` is the view zoom (only used with `adjust_for_zoom`).
    pub fn new(cfg: &Smoothing, zoom: f32) -> Self {
        Self { cfg: cfg.clone(), zoom: f64::from(zoom.max(0.01)), pos: None, last_input: None }
    }

    fn active(&self) -> bool {
        self.cfg.enabled && self.cfg.amount > 0.0
    }

    /// Feed one input point; returns the brush positions to paint through.
    pub fn push(&mut self, p: StrokePoint, out: &mut Vec<StrokePoint>) {
        let prev_input = self.last_input.replace(p);
        if !self.active() {
            out.push(p);
            return;
        }
        let Some(pos) = self.pos else {
            self.pos = Some(p);
            out.push(p);
            return;
        };
        let amount = f64::from(self.cfg.amount.clamp(0.0, 1.0));
        if self.cfg.pulled_string {
            let mut len = amount * MAX_STRING_PX;
            if self.cfg.adjust_for_zoom {
                len /= self.zoom;
            }
            let (dx, dy) = (p.x - pos.x, p.y - pos.y);
            let d = (dx * dx + dy * dy).sqrt();
            if d > len {
                let k = (d - len) / d;
                let np = StrokePoint { x: pos.x + dx * k, y: pos.y + dy * k, ..p };
                self.pos = Some(np);
                out.push(np);
            }
            return;
        }
        // Exponential: every SMOOTHING_STEP_MS of stroke time moves `a` of the way towards the
        // pointer, so the feel doesn't depend on how often the device reports. Input without
        // timestamps takes one step per point.
        let a = 1.0 - amount.min(0.95);
        let dt = prev_input.map_or(0.0, |q| p.time - q.time);
        let still = prev_input.is_some_and(|q| q.x == p.x && q.y == p.y);
        if still && !self.cfg.catch_up {
            // Without Stroke Catch-Up the brush stops while the pointer pauses (Photoshop).
            return;
        }
        let (iters, step) = if dt > 0.0 && dt.is_finite() {
            let steps = dt / SMOOTHING_STEP_MS;
            let iters = steps.ceil().clamp(1.0, 64.0);
            // Each of `iters` moves covers `steps / iters` steps' worth of pull.
            (iters as usize, 1.0 - (1.0 - a).powf(steps.min(64.0) / iters))
        } else {
            (1, a)
        };
        let mut cur = pos;
        for _ in 0..iters {
            cur = StrokePoint { x: cur.x + (p.x - cur.x) * step, y: cur.y + (p.y - cur.y) * step, ..p };
            out.push(cur);
        }
        self.pos = Some(cur);
    }

    /// Is Stroke Catch-Up still pulling the brush towards a pointer that has stopped? Then the
    /// stroke needs time to keep passing (repeats of the last point) while the pointer is held.
    pub fn lagging(&self) -> bool {
        match (self.active() && self.cfg.catch_up && !self.cfg.pulled_string, self.pos, self.last_input) {
            (true, Some(pos), Some(last)) => (pos.x - last.x).abs() + (pos.y - last.y).abs() > 0.01,
            _ => false,
        }
    }

    /// The string between the brush and the pointer: `(brush, pointer)` document positions while
    /// smoothing holds the brush back from the last input point, `None` when they coincide (or
    /// smoothing is off). The canvas draws it, as Photoshop does for Pulled String Mode.
    pub fn string(&self) -> Option<([f64; 2], [f64; 2])> {
        let (pos, last) = (self.pos?, self.last_input?);
        (self.active() && (pos.x != last.x || pos.y != last.y)).then_some(([pos.x, pos.y], [last.x, last.y]))
    }

    /// End of stroke: with Catch-Up On Stroke End the brush finishes at the last pointer position.
    pub fn finish(&mut self, out: &mut Vec<StrokePoint>) {
        if !self.active() || !self.cfg.catch_up_on_end {
            return;
        }
        if let (Some(pos), Some(last)) = (self.pos, self.last_input)
            && (pos.x != last.x || pos.y != last.y)
        {
            let (dx, dy) = (last.x - pos.x, last.y - pos.y);
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > 2.0 {
                let steps = (dist / 4.0).ceil().clamp(2.0, 16.0) as usize;
                for i in 1..steps {
                    let t = i as f64 / steps as f64;
                    out.push(StrokePoint {
                        x: pos.x + dx * t,
                        y: pos.y + dy * t,
                        pressure: pos.pressure + (last.pressure - pos.pressure) * t as f32,
                        tilt_x: pos.tilt_x + (last.tilt_x - pos.tilt_x) * t as f32,
                        tilt_y: pos.tilt_y + (last.tilt_y - pos.tilt_y) * t as f32,
                        rotation: pos.rotation + (last.rotation - pos.rotation) * t as f32,
                        wheel: pos.wheel + (last.wheel - pos.wheel) * t as f32,
                        time: pos.time + (last.time - pos.time) * t,
                    });
                }
            }
            self.pos = Some(last);
            out.push(last);
        }
    }
}

/// Smooth a whole point list with the brush's smoothing settings.
pub fn smooth_points(cfg: &Smoothing, zoom: f32, pts: &[StrokePoint]) -> Vec<StrokePoint> {
    let mut s = Smoother::new(cfg, zoom);
    let mut out = Vec::with_capacity(pts.len());
    for p in pts {
        s.push(*p, &mut out);
    }
    s.finish(&mut out);
    out
}

// ---------------------------------------------------------------------------
// Path walking
// ---------------------------------------------------------------------------

/// One spacing (or airbrush time) step along the path.
#[derive(Clone, Copy, Debug)]
pub struct StepInput {
    pub point: StrokePoint,
    /// Visual (counter-clockwise, y-up) direction of travel in radians.
    pub direction: f32,
    pub initial_direction: f32,
    /// Sequential step number (0 = first).
    pub step: u64,
}

/// Result of an adaptive stepping query along a path segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StepResult {
    /// Dab triggered at parameter `t` (`0.0..=1.0`) along the evaluated segment span.
    Dab(f64),
    /// No dab on this segment; distance was accumulated.
    Accumulated,
}

impl StepResult {
    #[inline]
    pub fn is_dab(&self) -> bool {
        matches!(self, StepResult::Dab(_))
    }

    #[inline]
    pub fn t(&self) -> Option<f64> {
        match *self {
            StepResult::Dab(t) => Some(t),
            StepResult::Accumulated => None,
        }
    }
}

/// Spacing parameters for adaptive stepping, supporting both isotropic and anisotropic brush tips.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpacingInfo {
    /// Spacing along major axis (x-axis in brush tip local coordinate frame).
    pub spacing_x: f64,
    /// Spacing along minor axis (y-axis in brush tip local coordinate frame).
    pub spacing_y: f64,
    /// Brush tip rotation angle in radians (counter-clockwise).
    pub rotation: f64,
}

impl SpacingInfo {
    #[inline]
    pub fn isotropic(spacing: f64) -> Self {
        let s = spacing.max(0.25);
        Self { spacing_x: s, spacing_y: s, rotation: 0.0 }
    }

    #[inline]
    pub fn anisotropic(spacing_x: f64, spacing_y: f64, rotation: f64) -> Self {
        Self {
            spacing_x: spacing_x.max(0.25),
            spacing_y: spacing_y.max(0.25),
            rotation,
        }
    }

    #[inline]
    pub fn is_isotropic(&self) -> bool {
        (self.spacing_x - self.spacing_y).abs() < 1e-4
    }

    #[inline]
    pub fn scalar_spacing(&self) -> f64 {
        self.spacing_x
    }
}

impl From<f64> for SpacingInfo {
    #[inline]
    fn from(s: f64) -> Self {
        Self::isotropic(s)
    }
}

/// Adaptive distance accumulator modeled after Krita's `KisDistanceInformation`.
///
/// Tracks accumulated path distance since the last dab. When required spacing collapses
/// (such as during stylus pressure drop on a stroke taper or flick) such that
/// `spacing - accumulated_distance <= 0.0`, it immediately triggers a dab at `t = 0.0`
/// and resets accumulators, eliminating dab starvation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceAccumulator {
    /// Accumulated scalar distance along the path since the last emitted dab.
    pub accumulated_distance: f64,
    /// Last applied spacing value.
    pub last_spacing: f64,
    /// Accumulated x-distance in the brush local coordinate frame.
    pub accum_x: f64,
    /// Accumulated y-distance in the brush local coordinate frame.
    pub accum_y: f64,
}

impl Default for DistanceAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl DistanceAccumulator {
    pub const fn new() -> Self {
        Self {
            accumulated_distance: 0.0,
            last_spacing: 0.0,
            accum_x: 0.0,
            accum_y: 0.0,
        }
    }

    /// Reset distance accumulators after a dab has been placed.
    #[inline]
    pub fn reset(&mut self) {
        self.accumulated_distance = 0.0;
        self.accum_x = 0.0;
        self.accum_y = 0.0;
    }

    /// Step along a linear segment of length `segment_len` with isotropic `current_spacing`.
    ///
    /// Implements Krita's `KisDistanceInformation::getNextPointPositionIsotropic`:
    /// When `next_point_distance = current_spacing - accumulated_distance <= 0.0`,
    /// triggers an immediate dab at `t = 0.0` and resets the accumulator.
    pub fn step(&mut self, segment_len: f64, current_spacing: f64) -> StepResult {
        let spacing = current_spacing.max(0.25);
        self.last_spacing = spacing;

        if segment_len <= 1e-9 {
            if spacing <= self.accumulated_distance {
                self.reset();
                return StepResult::Dab(0.0);
            }
            return StepResult::Accumulated;
        }

        let next_point_distance = spacing - self.accumulated_distance;
        if next_point_distance <= 0.0 {
            self.reset();
            StepResult::Dab(0.0)
        } else if next_point_distance <= segment_len + 1e-7 {
            let t = (next_point_distance / segment_len).clamp(0.0, 1.0);
            self.reset();
            StepResult::Dab(t)
        } else {
            self.accumulated_distance += segment_len;
            self.accum_x += segment_len;
            StepResult::Accumulated
        }
    }

    /// Step along a displacement `(dx, dy)` with anisotropic spacing and rotation.
    ///
    /// Implements Krita's `KisDistanceInformation::getNextPointPositionAnisotropic`:
    /// Solves the quadratic equation for the intersection of the segment with the
    /// spacing ellipse oriented at `spacing.rotation`. When the accumulator is already
    /// on or past the spacing ellipse (`gamma >= 0.0`), immediately triggers at `t = 0.0`.
    pub fn step_anisotropic(&mut self, dx: f64, dy: f64, spacing: &SpacingInfo) -> StepResult {
        if spacing.is_isotropic() {
            let seg_len = (dx * dx + dy * dy).sqrt();
            return self.step(seg_len, spacing.spacing_x);
        }

        let sx = spacing.spacing_x.max(0.25);
        let sy = spacing.spacing_y.max(0.25);
        self.last_spacing = sx;

        let a_rev = 1.0 / sx;
        let b_rev = 1.0 / sy;

        let x = self.accum_x;
        let y = self.accum_y;

        let gamma = (x * a_rev).powi(2) + (y * b_rev).powi(2) - 1.0;
        if gamma >= 0.0 {
            self.reset();
            return StepResult::Dab(0.0);
        }

        let seg_len = (dx * dx + dy * dy).sqrt();
        if seg_len <= 1e-9 {
            return StepResult::Accumulated;
        }

        // Rotate diff by brush rotation angle into the brush local frame
        let rot = spacing.rotation;
        let (rx, ry) = if rot.abs() > 2e-3 {
            let cos_r = rot.cos();
            let sin_r = rot.sin();
            (dx * cos_r - dy * sin_r, -dx * sin_r - dy * cos_r)
        } else {
            (dx, dy)
        };

        let abs_dx = rx.abs();
        let abs_dy = ry.abs();

        let alpha = (abs_dx * a_rev).powi(2) + (abs_dy * b_rev).powi(2);
        let beta = x * abs_dx * a_rev * a_rev + y * abs_dy * b_rev * b_rev;
        let d_4 = beta * beta - alpha * gamma;

        if d_4 >= 0.0 && alpha > 1e-12 {
            let k = (-beta + d_4.sqrt()) / alpha;
            if (0.0..=1.0 + 1e-6).contains(&k) {
                self.reset();
                return StepResult::Dab(k.min(1.0));
            }
        }

        self.accum_x += abs_dx;
        self.accum_y += abs_dy;
        self.accumulated_distance += seg_len;
        StepResult::Accumulated
    }
}

/// Walks a polyline and emits steps adaptively using a Krita-parity distance accumulator,
/// plus airbrush steps in time.
#[derive(Clone, Debug)]
pub struct PathWalker {
    last: Option<StrokePoint>,
    prev_pt: Option<StrokePoint>,
    /// Distance accumulator modeled after Krita's `KisDistanceInformation`.
    pub distance_acc: DistanceAccumulator,
    /// First point, emitted lazily once the initial direction is known.
    pending_first: Option<StrokePoint>,
    initial_dir: Option<f32>,
    dir: f32,
    step: u64,
    /// Airbrush interval in ms (None = off).
    interval: Option<f64>,
    time_acc: f64,
    /// Speed spacing (Spacing unchecked): one step per this many ms of stroke time.
    speed: Option<f64>,
    speed_acc: f64,
    /// Pixel-grid stepping for small aliased tips: the dab-centre offset inside a pixel (see
    /// [`PathWalker::pixel_grid`]).
    grid: Option<f64>,
    /// Whether to interpolate cubic Bézier curves through Catmull-Rom tangents (when smoothing is enabled).
    bezier: bool,
}

impl PathWalker {
    pub fn new(build_up_interval_ms: Option<f64>) -> Self {
        Self {
            last: None,
            prev_pt: None,
            distance_acc: DistanceAccumulator::new(),
            pending_first: None,
            initial_dir: None,
            dir: 0.0,
            step: 0,
            interval: build_up_interval_ms,
            time_acc: 0.0,
            speed: None,
            speed_acc: 0.0,
            grid: None,
            bezier: true,
        }
    }

    /// Enable or disable cubic Bézier curve fitting.
    pub fn bezier(mut self, enabled: bool) -> Self {
        self.bezier = enabled;
        self
    }

    /// Step pixel by pixel instead of by distance: one step per pixel along the segment's major
    /// axis, each at the dab centre (`offset` 0.5 for odd tip diameters, 0 for even ones, as
    /// `render::grid_center` snaps) of the pixel the segment's line crosses at that column (or
    /// row, when steep). A small aliased tip (the Pencil) then draws a clean 8-connected
    /// staircase: stepping by distance put two dabs in a row wherever the line crossed a pixel
    /// corner (#2139). The line is sampled where it really runs, not between the pixels holding
    /// the segment's ends, so a straight line split into pieces (by Smoothing, or the pointer
    /// samples of a ⇧-drag) paints the pixels of the whole line, whichever end it starts from
    /// (#6336).
    pub fn pixel_grid(mut self, offset: f64) -> Self {
        self.grid = Some(offset);
        self
    }

    /// Emits the pixel steps from `a` to `p` (see [`PathWalker::pixel_grid`]). False, emitting
    /// nothing, when the span is not finite or implausibly long: distance spacing takes over.
    fn grid_steps(&mut self, a: &StrokePoint, p: &StrokePoint, offset: f64, out: &mut impl FnMut(StepInput)) -> bool {
        // The pieces of a 45° line differ in rounding noise: keep them all on one major axis.
        let steep = (p.y - a.y).abs() > (p.x - a.x).abs() * (1.0 + 1e-9);
        let (am, an, pm, pn) = if steep { (a.y, a.x, p.y, p.x) } else { (a.x, a.y, p.x, p.y) };
        let (m0, m1) = (grid_cell(am, offset), grid_cell(pm, offset));
        let (dm, dn) = (pm - am, pn - an);
        let n = (m1 - m0).abs();
        if !(n.is_finite() && dn.is_finite()) || n > MAX_GRID_STEPS || dm == 0.0 {
            return false;
        }
        let dir = (m1 - m0).signum();
        for i in 1..=n as u64 {
            let c = m0 + dir * i as f64;
            // The last column's centre can sit up to half a pixel past `p`: the line goes on to
            // it, so the next piece of a straight line picks up where this one left off.
            let t = (c - am) / dm;
            let mut q = lerp_pt(a, p, t.clamp(0.0, 1.0));
            let cn = grid_cell(an + dn * t, offset);
            (q.x, q.y) = if steep { (cn, c) } else { (c, cn) };
            self.emit(q, out);
        }
        true
    }

    /// Space steps by stroke time instead of distance ([`SPEED_SPACING_MS`]): faster movement
    /// spreads them out. Segments without timestamps are spaced by distance.
    pub fn speed_spacing(mut self, interval_ms: f64) -> Self {
        self.speed = Some(interval_ms.max(0.1));
        self
    }

    fn emit(&mut self, point: StrokePoint, out: &mut impl FnMut(StepInput)) {
        let s = StepInput { point, direction: self.dir, initial_direction: self.initial_dir.unwrap_or(self.dir), step: self.step };
        self.step += 1;
        out(s);
    }

    /// Walks the segment between `a` and `p` using adaptive distance stepping.
    fn walk_distance(
        &mut self,
        a: &StrokePoint,
        p: &StrokePoint,
        spacing_fn: &impl Fn(&StrokePoint) -> SpacingInfo,
        out: &mut impl FnMut(StepInput),
    ) {
        let (dx, dy) = (p.x - a.x, p.y - a.y);
        let seg_len = (dx * dx + dy * dy).sqrt();
        if seg_len <= 1e-9 {
            return;
        }

        let mut u = 0.0;
        let mut cur_pt = *a;
        const MAX_STEPS: usize = 10_000;

        for _ in 0..MAX_STEPS {
            let rem_u = 1.0 - u;
            let rem_len = rem_u * seg_len;
            if rem_len <= 1e-9 {
                break;
            }

            let rem_dx = p.x - cur_pt.x;
            let rem_dy = p.y - cur_pt.y;
            let spacing = spacing_fn(&cur_pt);

            match self.distance_acc.step_anisotropic(rem_dx, rem_dy, &spacing) {
                StepResult::Dab(k) => {
                    let k = k.clamp(0.0, 1.0);
                    let new_u = (u + k * rem_u).clamp(0.0, 1.0);
                    let dab_pt = lerp_pt(a, p, new_u);
                    self.emit(dab_pt, out);
                    cur_pt = dab_pt;
                    u = new_u;
                }
                StepResult::Accumulated => {
                    break;
                }
            }
        }
    }

    /// Walks a curved cubic Bézier segment between `p1` and `p2`, using `p0` and `p3` for
    /// Catmull-Rom velocity tangents, with De Casteljau subdivision at ~0.5 px flatness.
    pub fn walk_bezier_curve(
        &mut self,
        p0: &StrokePoint,
        p1: &StrokePoint,
        p2: &StrokePoint,
        p3: &StrokePoint,
        spacing_fn: &impl Fn(&StrokePoint) -> SpacingInfo,
        out: &mut impl FnMut(StepInput),
    ) {
        let sub_pts = crate::curve::fit_cubic_bezier(p0, p1, p2, p3);
        if sub_pts.len() < 2 {
            return;
        }
        for w in sub_pts.windows(2) {
            let (sub_a, sub_b) = (&w[0], &w[1]);
            let (dx, dy) = (sub_b.x - sub_a.x, sub_b.y - sub_a.y);
            let len = (dx * dx + dy * dy).sqrt();
            if len > 1e-9 {
                self.dir = (-dy).atan2(dx) as f32;
                if self.initial_dir.is_none() {
                    self.initial_dir = Some(self.dir);
                }
            }
            self.walk_distance(sub_a, sub_b, spacing_fn, out);
        }
    }

    /// Push an input point using anisotropic spacing information.
    pub fn push_spacing(
        &mut self,
        p: StrokePoint,
        spacing_fn: &impl Fn(&StrokePoint) -> SpacingInfo,
        out: &mut impl FnMut(StepInput),
    ) {
        let Some(a) = self.last else {
            self.last = Some(p);
            self.prev_pt = None;
            self.pending_first = Some(p);
            self.distance_acc.reset();
            return;
        };
        let (dx, dy) = (p.x - a.x, p.y - a.y);
        let len = (dx * dx + dy * dy).sqrt();
        if len > 1e-9 {
            self.dir = (-dy).atan2(dx) as f32;
            if self.initial_dir.is_none() {
                self.initial_dir = Some(self.dir);
            }
        }
        if let Some(f) = self.pending_first.take() {
            self.emit(f, out);
            self.distance_acc.reset();
        }
        if len > 1e-9 {
            // Speed spacing needs the segment's duration; without timestamps (a straight ⇧-click
            // line, or input that carries no times) the segment is spaced by distance instead.
            let dt = p.time - a.time;
            match self.speed.filter(|_| dt > 0.0 && dt.is_finite()) {
                None => {
                    if !self.grid.is_some_and(|g| self.grid_steps(&a, &p, g, out)) {
                        if self.bezier {
                            let p0 = self.prev_pt.unwrap_or(a);
                            self.walk_bezier_curve(&p0, &a, &p, &p, spacing_fn, out);
                        } else {
                            self.walk_distance(&a, &p, spacing_fn, out);
                        }
                    } else {
                        self.distance_acc.reset();
                    }
                }
                Some(iv) => {
                    self.speed_acc += dt;
                    let max_n = (len / MIN_SPEED_STEP).ceil().max(1.0) as usize;
                    let mut n = 0;
                    while self.speed_acc >= iv && n < max_n {
                        self.speed_acc -= iv;
                        let f = ((dt - self.speed_acc) / dt).clamp(0.0, 1.0);
                        self.emit(lerp_pt(&a, &p, f), out);
                        n += 1;
                    }
                    if n == max_n {
                        self.speed_acc = self.speed_acc.rem_euclid(iv);
                    }
                }
            }
        }
        if let Some(iv) = self.interval {
            let dt = p.time - a.time;
            if dt > 0.0 && dt.is_finite() {
                if dt > iv * MAX_AIRBRUSH_DABS_PER_SEGMENT as f64 {
                    // A stalled or hostile timestamp must not replay an arbitrarily long backlog.
                    self.time_acc = 0.0;
                    self.emit(p, out);
                    self.prev_pt = self.last;
                    self.last = Some(p);
                    return;
                }
                self.time_acc += dt;
                let mut n = 0;
                while self.time_acc >= iv && n < MAX_AIRBRUSH_DABS_PER_SEGMENT {
                    self.time_acc -= iv;
                    let f = ((dt - self.time_acc) / dt).clamp(0.0, 1.0);
                    self.emit(lerp_pt(&a, &p, f), out);
                    n += 1;
                }
            }
        }
        self.prev_pt = self.last;
        self.last = Some(p);
    }

    pub fn push(&mut self, p: StrokePoint, step_len: &impl Fn(&StrokePoint) -> f64, out: &mut impl FnMut(StepInput)) {
        self.push_spacing(p, &|q| SpacingInfo::isotropic(step_len(q)), out);
    }

    pub fn finish(&mut self, out: &mut impl FnMut(StepInput)) {
        if let Some(f) = self.pending_first.take() {
            self.emit(f, out);
        }
        self.prev_pt = None;
        self.distance_acc.reset();
        self.time_acc = 0.0;
        self.speed_acc = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Dynamics
// ---------------------------------------------------------------------------

/// Scalar control value in 0..1.
pub fn control_value(c: Control, fade_steps: u32, s: &StepInput) -> f32 {
    let p = &s.point;
    match c {
        Control::Off | Control::InitialDirection | Control::Direction => 1.0,
        Control::Fade => 1.0 - (s.step as f32 / fade_steps.max(1) as f32).min(1.0),
        Control::PenPressure => p.pressure.clamp(0.0, 1.0),
        // Upright pen = 1, flat pen = 0 (a mouse reports no tilt, so paints at full value).
        Control::PenTilt => 1.0 - (p.tilt_x.hypot(p.tilt_y) / 90.0).clamp(0.0, 1.0),
        Control::StylusWheel => p.wheel.clamp(0.0, 1.0),
        Control::Rotation => (p.rotation.rem_euclid(360.0)) / 360.0,
    }
}

/// A dynamic's control value passed through its own response curve ([`Dynamic::curve`]). Controls
/// with no 0..1 input (Off, the direction controls) read 1 and skip the curve.
pub fn dynamic_value(d: &Dynamic, s: &StepInput) -> f32 {
    let v = control_value(d.control, d.fade_steps, s);
    match d.control {
        Control::Off | Control::InitialDirection | Control::Direction => v,
        Control::Fade | Control::PenPressure | Control::PenTilt | Control::StylusWheel | Control::Rotation => d.curve.eval(v),
    }
}

/// Angle contributed by a control, in degrees (counter-clockwise).
pub fn control_angle(c: Control, fade_steps: u32, s: &StepInput) -> f32 {
    let p = &s.point;
    match c {
        Control::Off => 0.0,
        Control::Fade => 360.0 * (s.step as f32 / fade_steps.max(1) as f32).min(1.0),
        Control::PenPressure => 360.0 * p.pressure.clamp(0.0, 1.0),
        Control::PenTilt => {
            if p.tilt_x == 0.0 && p.tilt_y == 0.0 {
                0.0
            } else {
                (-p.tilt_y).atan2(p.tilt_x).to_degrees()
            }
        }
        Control::StylusWheel => 360.0 * p.wheel.clamp(0.0, 1.0),
        Control::Rotation => p.rotation,
        Control::InitialDirection => s.initial_direction.to_degrees(),
        Control::Direction => s.direction.to_degrees(),
    }
}

/// A 0..1 multiplier from a dynamic: control lerps from `minimum` to 1, jitter reduces randomly,
/// never below `minimum`.
fn scalar(d: &Dynamic, s: &StepInput, r: f32) -> f32 {
    if !d.is_active() {
        return 1.0;
    }
    let min = d.minimum.clamp(0.0, 1.0);
    let cv = dynamic_value(d, s);
    let v = (min + (1.0 - min) * cv) * (1.0 - d.jitter.clamp(0.0, 1.0) * r);
    v.max(min)
}

fn rgb_to_hsv(c: [f32; 3]) -> [f32; 3] {
    let mx = c[0].max(c[1]).max(c[2]);
    let mn = c[0].min(c[1]).min(c[2]);
    let d = mx - mn;
    let h = if d <= 0.0 {
        0.0
    } else if mx == c[0] {
        ((c[1] - c[2]) / d).rem_euclid(6.0)
    } else if mx == c[1] {
        (c[2] - c[0]) / d + 2.0
    } else {
        (c[0] - c[1]) / d + 4.0
    } / 6.0;
    [h, if mx > 0.0 { d / mx } else { 0.0 }, mx]
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h6 = h.rem_euclid(1.0) * 6.0;
    let i = h6.floor();
    let f = h6 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    match i as i32 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// Colour Dynamics for one dab (or once per stroke, with `index = 0`).
pub fn dynamic_color(b: &BrushSettings, s: &StepInput, seed: u64, index: u64) -> [f32; 4] {
    let cd = &b.color_dynamics;
    let (fg, bg) = (b.color, b.background);
    let r = |st: u64| rand01(seed, index, st);
    let base = if cd.fg_bg.control != Control::Off { 1.0 - dynamic_value(&cd.fg_bg, s) } else { 0.0 };
    let t = (base + (1.0 - base) * cd.fg_bg.jitter.clamp(0.0, 1.0) * r(stream::FG_BG)).clamp(0.0, 1.0);
    let rgb = [fg[0] + (bg[0] - fg[0]) * t, fg[1] + (bg[1] - fg[1]) * t, fg[2] + (bg[2] - fg[2]) * t];
    let [mut h, mut sat, mut v] = rgb_to_hsv(rgb);
    h += rand_signed(seed, index, stream::HUE) * cd.hue_jitter.clamp(0.0, 1.0) * 0.5;
    sat = (sat + rand_signed(seed, index, stream::SAT) * cd.saturation_jitter.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    let pu = cd.purity.clamp(-1.0, 1.0);
    sat = if pu >= 0.0 { sat + (1.0 - sat) * pu } else { sat * (1.0 + pu) };
    v = (v + rand_signed(seed, index, stream::BRIGHT) * cd.brightness_jitter.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    let o = hsv_to_rgb(h, sat, v);
    [o[0], o[1], o[2], fg[3] + (bg[3] - fg[3]) * t]
}

/// Base (pre-jitter) diameter at a step: size × size control (pressure etc.). Drives spacing.
pub fn base_diameter(b: &BrushSettings, s: &StepInput) -> f32 {
    let sd = &b.shape_dynamics;
    let mut f = 1.0;
    if sd.enabled && sd.size.control != Control::Off {
        let min = sd.size.minimum.clamp(0.0, 1.0);
        f = min + (1.0 - min) * dynamic_value(&sd.size, s);
    } else if b.pressure_size {
        f = s.point.pressure.clamp(0.0, 1.0);
    }
    b.size * f
}

/// Builds the primary dabs of a step (Shape Dynamics, Scattering, Transfer, Color, Texture depth).
#[derive(Clone, Debug)]
pub struct DabBuilder {
    pub brush: BrushSettings,
    pub seed: u64,
    /// Per-stroke colour (Color Dynamics with per-tip off), computed at the first step.
    stroke_color: Option<[f32; 4]>,
    next_index: u64,
}

impl DabBuilder {
    pub fn new(brush: &BrushSettings) -> Self {
        Self { brush: brush.clone(), seed: brush.seed, stroke_color: None, next_index: 0 }
    }

    pub fn spacing_info(&self, p: &StrokePoint) -> SpacingInfo {
        let s = StepInput { point: *p, direction: 0.0, initial_direction: 0.0, step: 0 };
        let d = f64::from(base_diameter(&self.brush, &s).max(1.0));
        let sp = f64::from(self.brush.spacing.max(0.01));
        let major_spacing = (d * sp).max(0.25);

        let b = &self.brush;
        let mut roundness = f64::from(b.roundness.clamp(0.01, 1.0));
        let mut angle_deg = f64::from(b.angle);

        if b.shape_dynamics.enabled {
            let sd = &b.shape_dynamics;
            if sd.angle.control != Control::Off {
                angle_deg += f64::from(control_angle(sd.angle.control, sd.angle.fade_steps, &s));
            }
            if sd.roundness.control != Control::Off {
                let r_dyn = dynamic_value(&sd.roundness, &s);
                let min_r = f64::from(sd.roundness.minimum.clamp(0.0, 1.0));
                let factor = min_r + (1.0 - min_r) * f64::from(r_dyn);
                roundness = (roundness * factor).clamp(0.01, 1.0);
            }
        }

        if roundness >= 0.999 {
            SpacingInfo::isotropic(major_spacing)
        } else {
            let minor_spacing = (major_spacing * roundness).max(0.25);
            let rot_rad = angle_deg.to_radians();
            SpacingInfo::anisotropic(major_spacing, minor_spacing, rot_rad)
        }
    }

    pub fn step_len(&self, p: &StrokePoint) -> f64 {
        self.spacing_info(p).spacing_x
    }

    pub fn build(&mut self, s: &StepInput, out: &mut Vec<Dab>) {
        let b = &self.brush;
        let seed = self.seed;
        let sd = &b.shape_dynamics;
        let sc = &b.scattering;
        let count = if sc.enabled {
            let n = sc.count.clamp(1, 16) as f32;
            let k = scalar(&sc.count_jitter, s, rand01(seed, s.step, stream::COUNT));
            ((n * k).round() as u32).clamp(1, 16)
        } else {
            1
        };
        let base_d = base_diameter(b, s);
        for _ in 0..count {
            let i = self.next_index;
            self.next_index += 1;
            let r = |st: u64| rand01(seed, i, st);
            // Shape.
            let mut diameter = base_d;
            let mut angle = b.angle;
            let mut roundness = b.roundness.clamp(0.01, 1.0);
            let (mut fx, mut fy) = (b.flip_x, b.flip_y);
            if sd.enabled {
                if sd.size.jitter > 0.0 {
                    let min = sd.size.minimum.clamp(0.0, 1.0);
                    let f = (1.0 - sd.size.jitter.clamp(0.0, 1.0) * r(stream::SIZE)).max(min);
                    diameter *= f;
                }
                angle +=
                    control_angle(sd.angle.control, sd.angle.fade_steps, s) + rand_signed(seed, i, stream::ANGLE) * sd.angle.jitter.clamp(0.0, 1.0) * 180.0;
                roundness = (roundness * scalar(&sd.roundness, s, r(stream::ROUNDNESS))).clamp(0.01, 1.0);
                if sd.flip_x_jitter && r(stream::FLIP_X) < 0.5 {
                    fx = !fx;
                }
                if sd.flip_y_jitter && r(stream::FLIP_Y) < 0.5 {
                    fy = !fy;
                }
            }
            // Tilt Scale and Brush Projection (Shape Dynamics).
            let (mut proj_angle, mut proj_scale) = (0.0, 1.0);
            if sd.enabled {
                let tilt = s.point.tilt_x.hypot(s.point.tilt_y).clamp(0.0, 90.0);
                if sd.size.control == Control::PenTilt && sd.tilt_scale > 0.0 {
                    let k = 1.0 - sd.tilt_scale.clamp(0.0, 2.0) * 0.5 * tilt / 90.0;
                    roundness = (roundness * k).clamp(0.01, 1.0);
                }
                if sd.brush_projection {
                    angle += s.point.rotation;
                    if tilt > 0.0 {
                        proj_angle = (-s.point.tilt_y).atan2(s.point.tilt_x);
                        proj_scale = tilt.to_radians().cos().max(0.05);
                    }
                }
            }
            let radius = (diameter / 2.0).max(0.5);
            // Scatter.
            let mut c = Point::new(s.point.x, s.point.y);
            if sc.enabled && sc.scatter.jitter > 0.0 {
                let cv = dynamic_value(&sc.scatter, s);
                let amt = f64::from(sc.scatter.jitter * cv * radius);
                let (ox, oy) = (rand_signed(seed, i, stream::SCATTER_X) as f64 * amt, rand_signed(seed, i, stream::SCATTER_Y) as f64 * amt);
                if sc.both_axes {
                    c.x += ox;
                    c.y += oy;
                } else {
                    // Perpendicular to the direction of travel (y-down image space).
                    let (dx, dy) = (f64::from(s.direction.cos()), -f64::from(s.direction.sin()));
                    c.x += -dy * ox;
                    c.y += dx * ox;
                }
            }
            // Transfer.
            let tr = &b.transfer;
            let (mut opacity, mut flow) = (1.0, b.flow);
            if tr.enabled {
                opacity = scalar(&tr.opacity, s, r(stream::OPACITY));
                flow *= scalar(&tr.flow, s, r(stream::FLOW));
            }
            let (wet, mix) = if tr.enabled { (scalar(&tr.wetness, s, r(stream::WET)), scalar(&tr.mix, s, r(stream::MIX))) } else { (1.0, 1.0) };
            if b.pressure_opacity && !(tr.enabled && tr.opacity.control != Control::Off) {
                opacity *= s.point.pressure.clamp(0.0, 1.0);
            }
            // Colour.
            let color = if b.color_dynamics.enabled {
                if b.color_dynamics.per_tip { dynamic_color(b, s, seed, i) } else { *self.stroke_color.get_or_insert_with(|| dynamic_color(b, s, seed, 0)) }
            } else {
                b.color
            };
            // Texture depth (per tip).
            let tx = &b.texture;
            let depth =
                if tx.enabled && tx.each_tip { tx.depth.clamp(0.0, 1.0) * scalar(&tx.depth_jitter, s, r(stream::DEPTH)) } else { tx.depth.clamp(0.0, 1.0) };
            out.push(Dab {
                center: c,
                radius,
                alpha: flow.clamp(0.0, 1.0),
                angle: angle.to_radians(),
                roundness,
                flip_x: fx,
                flip_y: fy,
                opacity: opacity.clamp(0.0, 1.0),
                color,
                depth,
                index: i,
                proj_angle,
                proj_scale,
                wet,
                mix,
            });
        }
    }
}

/// Builds Dual Brush dabs (the second tip: size, spacing, scatter, count, flip).
#[derive(Clone, Debug)]
pub struct DualBuilder {
    pub brush: BrushSettings,
    next_index: u64,
}

impl DualBuilder {
    pub fn new(brush: &BrushSettings) -> Self {
        Self { brush: brush.clone(), next_index: 0 }
    }
    pub fn spacing_info(&self, _p: &StrokePoint) -> SpacingInfo {
        let d = &self.brush.dual_brush;
        let major = f64::from(d.size.max(1.0) * d.spacing.max(0.01)).max(0.25);
        let roundness = f64::from(d.roundness.clamp(0.01, 1.0));
        if roundness >= 0.999 {
            SpacingInfo::isotropic(major)
        } else {
            let minor = (major * roundness).max(0.25);
            let rot = f64::from(d.angle).to_radians();
            SpacingInfo::anisotropic(major, minor, rot)
        }
    }
    pub fn step_len(&self, p: &StrokePoint) -> f64 {
        self.spacing_info(p).spacing_x
    }
    pub fn build(&mut self, s: &StepInput, out: &mut Vec<Dab>) {
        let d = &self.brush.dual_brush;
        let seed = self.brush.seed ^ 0xD0A1;
        for _ in 0..d.count.clamp(1, 16) {
            let i = self.next_index;
            self.next_index += 1;
            let radius = (d.size / 2.0).max(0.5);
            let amt = f64::from(d.scatter * radius);
            let (ox, oy) = (rand_signed(seed, i, stream::DUAL_X) as f64 * amt, rand_signed(seed, i, stream::DUAL_Y) as f64 * amt);
            let mut c = Point::new(s.point.x, s.point.y);
            if d.both_axes {
                c.x += ox;
                c.y += oy;
            } else {
                let (dx, dy) = (f64::from(s.direction.cos()), -f64::from(s.direction.sin()));
                c.x += -dy * ox;
                c.y += dx * ox;
            }
            let flip = d.flip && rand01(seed, i, stream::DUAL_FLIP) < 0.5;
            out.push(Dab {
                center: c,
                radius,
                alpha: 1.0,
                angle: d.angle.to_radians(),
                roundness: d.roundness.clamp(0.01, 1.0),
                flip_x: flip,
                flip_y: false,
                opacity: 1.0,
                color: [0.0, 0.0, 0.0, 1.0],
                depth: 1.0,
                index: i,
                proj_angle: 0.0,
                proj_scale: 1.0,
                wet: 1.0,
                mix: 1.0,
            });
        }
    }
}

/// Apply Brush Pose to an input point.
pub fn apply_pose(b: &BrushSettings, mut p: StrokePoint) -> StrokePoint {
    let pose = &b.pose;
    if pose.enabled {
        if pose.override_tilt {
            p.tilt_x = pose.tilt_x;
            p.tilt_y = pose.tilt_y;
        }
        if pose.override_rotation {
            p.rotation = pose.rotation;
        }
        if pose.override_pressure {
            p.pressure = pose.pressure;
        }
    }
    p
}

/// Incremental dab generation for a whole brush (primary and optional dual dabs).
#[derive(Clone, Debug)]
pub struct DabGenerator {
    smoother: Smoother,
    walker: PathWalker,
    dual_walker: Option<PathWalker>,
    builder: DabBuilder,
    dual: Option<DualBuilder>,
    scratch: Vec<StrokePoint>,
}

impl DabGenerator {
    /// Does the stroke change with time while the pointer is held still (airbrush Build-up, or
    /// smoothing still catching up)? The canvas then keeps feeding it the held point.
    pub fn wants_time(&self) -> bool {
        self.walker.interval.is_some() || self.smoother.lagging()
    }

    /// See [`Smoother::string`].
    pub fn smoothing_string(&self) -> Option<([f64; 2], [f64; 2])> {
        self.smoother.string()
    }

    pub fn new(brush: &BrushSettings, zoom: f32) -> Self {
        let brush = brush.bounded_for_render();
        let interval = brush.build_up.then(|| 1000.0 / f64::from(brush.build_up_rate.clamp(0.1, 1000.0)));
        let dual_on = brush.dual_brush.enabled;
        let mut walker = if brush.spacing_enabled { PathWalker::new(interval) } else { PathWalker::new(interval).speed_spacing(SPEED_SPACING_MS) };
        if !brush.smoothing.enabled || brush.smoothing.amount <= 0.0 {
            walker = walker.bezier(false);
        }
        // A small aliased tip whose dabs are at most a pixel apart steps pixel by pixel (#2139).
        if brush.aliased && brush.size * brush.spacing.max(0.01) <= 1.0 {
            walker = walker.pixel_grid(f64::from(crate::render::grid_center(0.0, 0.0, brush.size).0));
        }
        Self {
            smoother: Smoother::new(&brush.smoothing, zoom),
            walker,
            dual_walker: dual_on.then(|| PathWalker::new(None)),
            builder: DabBuilder::new(&brush),
            dual: dual_on.then(|| DualBuilder::new(&brush)),
            scratch: Vec::new(),
        }
    }

    fn walk(&mut self, pts: &[StrokePoint], dabs: &mut Vec<Dab>, dual: &mut Vec<Dab>) {
        for p in pts {
            let p = apply_pose(&self.builder.brush, *p);
            let b = &self.builder;
            let spacing = |q: &StrokePoint| b.spacing_info(q);
            let mut steps = Vec::new();
            self.walker.push_spacing(p, &spacing, &mut |s| steps.push(s));
            for s in &steps {
                self.builder.build(s, dabs);
            }
            if let (Some(w), Some(db)) = (self.dual_walker.as_mut(), self.dual.as_mut()) {
                let mut steps = Vec::new();
                let spacing = |q: &StrokePoint| db.spacing_info(q);
                w.push_spacing(p, &spacing, &mut |s| steps.push(s));
                for s in &steps {
                    db.build(s, dual);
                }
            }
        }
    }

    /// Feed input points; appends the new primary and dual dabs.
    pub fn push(&mut self, pts: &[StrokePoint], dabs: &mut Vec<Dab>, dual: &mut Vec<Dab>) {
        let mut sm = std::mem::take(&mut self.scratch);
        sm.clear();
        for p in pts {
            self.smoother.push(*p, &mut sm);
        }
        self.walk(&sm, dabs, dual);
        self.scratch = sm;
    }

    /// End of stroke.
    pub fn finish(&mut self, dabs: &mut Vec<Dab>, dual: &mut Vec<Dab>) {
        let mut sm = Vec::new();
        self.smoother.finish(&mut sm);
        if sm.len() >= 2 {
            crate::curve::apply_tail_aggressiveness(&mut sm, 1.0);
        }
        self.walk(&sm, dabs, dual);

        let mut steps = Vec::new();
        self.walker.finish(&mut |s| steps.push(s));
        for s in &steps {
            self.builder.build(s, dabs);
        }
        if let (Some(w), Some(db)) = (self.dual_walker.as_mut(), self.dual.as_mut()) {
            let mut steps = Vec::new();
            w.finish(&mut |s| steps.push(s));
            for s in &steps {
                db.build(s, dual);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_accumulator_isotropic_stepping() {
        let mut acc = DistanceAccumulator::new();
        assert_eq!(acc.accumulated_distance, 0.0);

        // Step 1: segment length 10.0, spacing 4.0 -> first dab at t = 4.0 / 10.0 = 0.4
        let res = acc.step(10.0, 4.0);
        assert_eq!(res, StepResult::Dab(0.4));
        assert_eq!(acc.accumulated_distance, 0.0);

        // Step 2: on remaining segment of length 6.0, spacing 4.0 -> dab at t = 4.0 / 6.0
        let res = acc.step(6.0, 4.0);
        match res {
            StepResult::Dab(t) => assert!((t - 4.0 / 6.0).abs() < 1e-6),
            _ => panic!("expected dab"),
        }
        assert_eq!(acc.accumulated_distance, 0.0);

        // Step 3: on remaining segment of length 2.0, spacing 4.0 -> too short, accumulate
        let res = acc.step(2.0, 4.0);
        assert_eq!(res, StepResult::Accumulated);
        assert!((acc.accumulated_distance - 2.0).abs() < 1e-6);

        // Step 4: next segment of length 5.0, spacing 4.0 -> remaining needed distance = 4.0 - 2.0 = 2.0
        // dab at t = 2.0 / 5.0 = 0.4
        let res = acc.step(5.0, 4.0);
        match res {
            StepResult::Dab(t) => assert!((t - 0.4).abs() < 1e-6),
            _ => panic!("expected dab"),
        }
        assert_eq!(acc.accumulated_distance, 0.0);
    }

    #[test]
    fn test_distance_accumulator_immediate_trigger_on_spacing_shrinkage() {
        let mut acc = DistanceAccumulator::new();
        // Simulate having traveled 15.0 px under thick brush (spacing was 20.0 px)
        acc.accumulated_distance = 15.0;

        // Stylus pressure suddenly collapses, reducing required spacing to 5.0 px.
        // nextPointDistance = 5.0 - 15.0 = -10.0 <= 0.0 -> must immediately trigger at t = 0.0!
        let res = acc.step(10.0, 5.0);
        assert_eq!(res, StepResult::Dab(0.0));
        assert_eq!(acc.accumulated_distance, 0.0);

        // Now with accumulators reset, next step on remaining length 10.0 produces dab at t = 5.0 / 10.0 = 0.5
        let res = acc.step(10.0, 5.0);
        assert_eq!(res, StepResult::Dab(0.5));
        assert_eq!(acc.accumulated_distance, 0.0);
    }

    #[test]
    fn test_distance_accumulator_anisotropic_rotation() {
        let mut acc = DistanceAccumulator::new();
        // Chisel brush tip: major spacing 20.0 along X, minor spacing 5.0 along Y, unrotated
        let spacing_unrotated = SpacingInfo::anisotropic(20.0, 5.0, 0.0);
        assert!(!spacing_unrotated.is_isotropic());

        // Moving along major axis (X): segment (20.0, 0.0) -> dab at 1.0
        let res = acc.step_anisotropic(20.0, 0.0, &spacing_unrotated);
        match res {
            StepResult::Dab(k) => assert!((k - 1.0).abs() < 1e-6),
            _ => panic!("expected dab along major axis"),
        }
        assert_eq!(acc.accum_x, 0.0);
        assert_eq!(acc.accum_y, 0.0);

        // Moving along minor axis (Y): segment (0.0, 5.0) -> dab at 1.0
        let res = acc.step_anisotropic(0.0, 5.0, &spacing_unrotated);
        match res {
            StepResult::Dab(k) => assert!((k - 1.0).abs() < 1e-6),
            _ => panic!("expected dab along minor axis"),
        }

        // Now rotate brush by 90 degrees counter-clockwise (PI / 2)
        // Now moving along X corresponds to the brush's minor axis (needs 5.0 px)
        // Moving along Y corresponds to the brush's major axis (needs 20.0 px)
        let spacing_rotated = SpacingInfo::anisotropic(20.0, 5.0, std::f64::consts::FRAC_PI_2);

        // Moving 5.0 along X (minor axis now) -> dab triggered!
        let res = acc.step_anisotropic(5.0, 0.0, &spacing_rotated);
        match res {
            StepResult::Dab(k) => assert!((k - 1.0).abs() < 1e-4),
            _ => panic!("expected dab along rotated minor axis"),
        }

        // Moving 20.0 along Y (major axis now) -> dab triggered!
        let res = acc.step_anisotropic(0.0, 20.0, &spacing_rotated);
        match res {
            StepResult::Dab(k) => assert!((k - 1.0).abs() < 1e-4),
            _ => panic!("expected dab along rotated major axis"),
        }

        // Anisotropic collapse: if accum_x = 10.0, and spacing shrinks to sx = 5.0 -> immediate trigger at 0.0
        acc.accum_x = 10.0;
        let small_spacing = SpacingInfo::anisotropic(5.0, 2.0, 0.0);
        let res = acc.step_anisotropic(10.0, 0.0, &small_spacing);
        assert_eq!(res, StepResult::Dab(0.0));
        assert_eq!(acc.accum_x, 0.0);
        assert_eq!(acc.accum_y, 0.0);
    }

    #[test]
    fn test_tapering_stroke_generates_dabs_to_the_tip_without_starvation() {
        // Brush configured with pressure-driven size: 60px diameter at pressure 1.0 down to tiny at 0.01
        let brush = BrushSettings {
            size: 60.0,
            spacing: 0.25,
            pressure_size: true,
            ..Default::default()
        };

        // Construct a stroke that starts thick and then rapidly tapers across small segments to the tip
        let mut pts = Vec::new();
        pts.push(StrokePoint::new(0.0, 0.0, 1.0));
        pts.push(StrokePoint::new(10.0, 0.0, 1.0)); // 10px segment (spacing is 15px, so no dab yet on this segment)

        // Rapid taper: 12 trailing segments of 1.5 px each (x from 10.0 to 28.0), pressure dropping from 1.0 to 0.01
        for i in 1..=12 {
            let frac = i as f64 / 12.0;
            let x = 10.0 + i as f64 * 1.5;
            let pressure = (1.0 - frac * 0.99).max(0.01) as f32;
            pts.push(StrokePoint::new(x, 0.0, pressure));
        }

        let mut dabs = Vec::new();
        let mut dual = Vec::new();
        let mut generator = DabGenerator::new(&brush, 1.0);
        generator.push(&pts, &mut dabs, &mut dual);
        generator.finish(&mut dabs, &mut dual);

        assert!(!dabs.is_empty(), "stroke must emit dabs");

        // The stroke end is at x = 28.0.
        // In the broken linear next_at implementation, the large initial next_at skipped all trailing segments,
        // leaving the stroke terminated prematurely at x = 0.
        // In our adaptive implementation, dabs are emitted all the way along the taper to the tip!
        let last_dab = dabs.last().expect("last dab exists");
        assert!(
            last_dab.center.x >= 26.5,
            "dabs must reach the end of the stroke tip, got last dab at x = {}",
            last_dab.center.x
        );

        // Check that trailing dabs have continuously shrinking radii down to needle tip
        assert!(last_dab.radius < 5.0, "tip dab must be thin, got radius {}", last_dab.radius);

        // Check that multiple dabs exist in the tapering region (x > 10.0)
        let taper_dabs = dabs.iter().filter(|d| d.center.x > 10.0).count();
        assert!(taper_dabs >= 5, "taper region must have multiple fine dabs, got {}", taper_dabs);
    }

    #[test]
    fn test_high_speed_flick_zero_dab_starvation() {
        // High speed flick: pen accelerates rapidly while pressure plummets
        let brush = BrushSettings {
            size: 40.0,
            spacing: 0.25,
            pressure_size: true,
            ..Default::default()
        };

        let mut pts = Vec::new();
        pts.push(StrokePoint::new(0.0, 0.0, 1.0));
        // High speed flick coordinates with increasing segment distances and dropping pressure
        let flick_data = [
            (10.0, 0.9),
            (25.0, 0.75),
            (50.0, 0.5),
            (85.0, 0.3),
            (130.0, 0.15),
            (180.0, 0.08),
            (225.0, 0.03),
            (250.0, 0.01),
        ];
        for (x, p) in flick_data {
            pts.push(StrokePoint::new(x, 0.0, p));
        }

        let mut dabs = Vec::new();
        let mut dual = Vec::new();
        let mut generator = DabGenerator::new(&brush, 1.0);
        generator.push(&pts, &mut dabs, &mut dual);
        generator.finish(&mut dabs, &mut dual);

        // Verify that dabs are emitted throughout the flick with zero starvation gaps
        assert!(dabs.len() >= 20, "expected abundant dabs across flick, got {}", dabs.len());

        // Check consecutive dabs: gap between dabs must be bounded by brush spacing at each location
        for window in dabs.windows(2) {
            let gap = window[1].center.x - window[0].center.x;
            assert!(gap > 0.0, "dabs must advance monotonically along flick");
            // The gap should never exceed the maximum brush spacing (40.0 * 0.25 = 10.0 px)
            assert!(gap <= 10.5, "starvation gap detected: gap {} px exceeds allowed spacing", gap);
        }

        // Final dab reaches near the tip of the flick
        let last_dab = dabs.last().expect("last dab");
        assert!(last_dab.center.x >= 240.0, "flick dabs must reach the end, got x = {}", last_dab.center.x);
    }

    #[test]
    fn test_spacing_scales_smoothly_and_monotonically_with_pressure() {
        let brush = BrushSettings {
            size: 50.0,
            spacing: 0.25,
            pressure_size: true,
            ..Default::default()
        };
        let builder = DabBuilder::new(&brush);

        let mut prev_spacing = 0.0;
        for step in 1..=100 {
            let pressure = step as f32 / 100.0;
            let pt = StrokePoint::new(0.0, 0.0, pressure);
            let spacing = builder.step_len(&pt);
            assert!(spacing >= prev_spacing, "spacing must be monotonically non-decreasing: at pressure {} spacing {} < prev {}", pressure, spacing, prev_spacing);
            prev_spacing = spacing;
        }

        // Check boundary values
        let low_pt = StrokePoint::new(0.0, 0.0, 0.01);
        let low_sp = builder.step_len(&low_pt);
        let high_pt = StrokePoint::new(0.0, 0.0, 1.0);
        let high_sp = builder.step_len(&high_pt);
        assert!(low_sp < high_sp, "low pressure spacing {} must be smaller than high pressure spacing {}", low_sp, high_sp);
        assert!((high_sp - 50.0 * 0.25).abs() < 1e-4, "high pressure spacing matches diameter * spacing");
    }

    #[test]
    fn test_path_walker_finish_clean_handling() {
        let mut walker = PathWalker::new(None);

        // 1. Single click (1 point): finish should emit pending_first
        let pt = StrokePoint::new(10.0, 20.0, 1.0);
        let mut steps = Vec::new();
        walker.push(pt, &|_| 5.0, &mut |s| steps.push(s));
        assert!(steps.is_empty(), "first point is held pending");

        walker.finish(&mut |s| steps.push(s));
        assert_eq!(steps.len(), 1, "finish must emit pending first point");
        assert_eq!(steps[0].point.x, 10.0);
        assert_eq!(steps[0].point.y, 20.0);

        // 2. Calling finish again is clean and idempotent (no extra dabs)
        steps.clear();
        walker.finish(&mut |s| steps.push(s));
        assert!(steps.is_empty(), "subsequent finish calls emit nothing");

        // 3. Multi-point stroke: accumulators are cleanly reset on finish
        let p1 = StrokePoint::new(0.0, 0.0, 1.0);
        let p2 = StrokePoint::new(10.0, 0.0, 1.0);
        walker.push(p1, &|_| 5.0, &mut |s| steps.push(s));
        walker.push(p2, &|_| 5.0, &mut |s| steps.push(s));
        steps.clear();
        walker.finish(&mut |s| steps.push(s));
        assert!(steps.is_empty(), "finish after full walk emits nothing extra");
        assert_eq!(walker.distance_acc.accumulated_distance, 0.0);
    }

    #[test]
    fn test_decreasing_pressure_monotonically_tapers_diameter() {
        let brush = BrushSettings {
            size: 60.0,
            pressure_size: true,
            spacing: 0.1,
            ..Default::default()
        };

        let pts = vec![
            StrokePoint { x: 0.0, y: 0.0, pressure: 1.0, time: 0.0, ..Default::default() },
            StrokePoint { x: 40.0, y: 5.0, pressure: 0.8, time: 10.0, ..Default::default() },
            StrokePoint { x: 80.0, y: 15.0, pressure: 0.6, time: 20.0, ..Default::default() },
            StrokePoint { x: 120.0, y: 25.0, pressure: 0.3, time: 30.0, ..Default::default() },
            StrokePoint { x: 150.0, y: 32.0, pressure: 0.1, time: 40.0, ..Default::default() },
            StrokePoint { x: 170.0, y: 35.0, pressure: 0.02, time: 50.0, ..Default::default() },
        ];

        let mut generator = DabGenerator::new(&brush, 1.0);
        let mut dabs = Vec::new();
        let mut dual = Vec::new();
        generator.push(&pts, &mut dabs, &mut dual);
        generator.finish(&mut dabs, &mut dual);

        assert!(dabs.len() >= 5, "must generate dabs along stroke");

        // Verify diameters monotonically decrease along the tapering stroke
        let mut prev_radius = dabs[0].radius;
        for (i, dab) in dabs.iter().enumerate().skip(1) {
            assert!(
                dab.radius <= prev_radius + 1e-4,
                "dab {} radius {} must not exceed previous radius {}",
                i,
                dab.radius,
                prev_radius
            );
            prev_radius = dab.radius;
        }

        // Verify needle-sharp taper at the tip
        let last_dab = dabs.last().copied().unwrap_or(dabs[0]);
        assert!(last_dab.radius <= 3.0, "final dab radius {} must be needle sharp", last_dab.radius);
        assert!(last_dab.center.x >= 150.0, "dabs must reach to the tapered tip");
    }

    #[test]
    fn test_path_walker_walk_bezier_curve() {
        let mut walker = PathWalker::new(None);
        let p0 = StrokePoint::new(0.0, 0.0, 1.0);
        let p1 = StrokePoint::new(0.0, 0.0, 1.0);
        let p2 = StrokePoint::new(50.0, 50.0, 0.8);
        let p3 = StrokePoint::new(100.0, 50.0, 0.6);

        let mut steps = Vec::new();
        walker.walk_bezier_curve(&p0, &p1, &p2, &p3, &|_q| SpacingInfo::isotropic(10.0), &mut |s| steps.push(s));

        assert!(!steps.is_empty(), "walk_bezier_curve must emit steps along the curve");
        for s in &steps {
            assert!(s.point.x >= 0.0 && s.point.x <= 55.0);
            assert!(s.point.y >= 0.0 && s.point.y <= 55.0);
        }
    }

    #[test]
    fn test_short_fast_flick_velocity_tapering() {
        let brush = BrushSettings {
            size: 40.0,
            pressure_size: true,
            spacing: 0.1,
            ..Default::default()
        };

        // 4-point fast flick with natural lift-off tapering to zero pressure
        let pts = vec![
            StrokePoint { x: 0.0, y: 0.0, pressure: 0.9, time: 0.0, ..Default::default() },
            StrokePoint { x: 30.0, y: 10.0, pressure: 0.6, time: 8.0, ..Default::default() },
            StrokePoint { x: 70.0, y: 20.0, pressure: 0.25, time: 16.0, ..Default::default() },
            StrokePoint { x: 80.0, y: 22.0, pressure: 0.02, time: 20.0, ..Default::default() },
        ];

        let mut generator = DabGenerator::new(&brush, 1.0);
        let mut dabs = Vec::new();
        let mut dual = Vec::new();
        generator.push(&pts, &mut dabs, &mut dual);

        assert!(!dabs.is_empty(), "dabs must be generated along stroke");

        generator.finish(&mut dabs, &mut dual);

        // Verify the final dab is needle-sharp (<= 2.0 px radius) and strictly stays on path
        let final_dab = dabs.last().unwrap();
        assert!(final_dab.radius <= 2.0, "final dab radius {} must be needle sharp", final_dab.radius);
        assert!(final_dab.center.x <= 80.5, "finish must never project artificial spikes beyond the stroke trajectory");
    }
}
