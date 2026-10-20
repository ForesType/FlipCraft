//! Stylus input: real pen pressure (and tilt / barrel rotation where the platform reports them)
//! for the painting tools. A mouse always paints at pressure 1 with no tilt.
//!
//! Sources, by platform (eframe 0.36.2, egui-winit 0.36.2, winit 0.30.13):
//!
//! - **Windows**: winit turns `WM_POINTER` pen and touch input into `WindowEvent::Touch` with a
//!   normalised force (`POINTER_PEN_INFO::pressure / 1024`); egui-winit forwards it as
//!   `egui::Event::Touch { force }` alongside the emulated pointer. [`Stylus::update`] reads it.
//!   winit drops the pen's tilt and rotation, so those stay 0, and reports pressure 0 (a tap
//!   lighter than the pen's first pressure level) as no force, which [`Stylus::update`] reads as
//!   pressure 0 once it knows a pen is in use.
//! - **Web**: eframe forwards touch force but not pen pointer events, so the web runner listens
//!   to `pointerdown`/`pointermove` itself and writes `pressure`, `tiltX`, `tiltY`, `twist` and
//!   the eraser button of `pointerType == "pen"` events into the [`StylusFeed`].
//! - **macOS**: winit 0.30 drops `NSEvent` tablet data, so the desktop app installs an AppKit
//!   local event monitor (the `photocraft-tablet` crate) that writes pressure, tilt, rotation and
//!   the eraser end into the [`StylusFeed`] before winit handles each event.
//! - **Linux X11**: the desktop app reads XInput2 raw valuator events on its own X connection
//!   (`photocraft-tablet`, x11rb) and writes them into the [`StylusFeed`].
//! - **Linux Wayland**: compositors give a pen only to clients that bind `zwp_tablet_v2`, which
//!   winit doesn't (it would have to share winit's connection), so with a pen attached the
//!   desktop app opens its window through Xwayland, which reports tablet valuators to the X11
//!   reader. `PHOTOCRAFT_NATIVE_WAYLAND=1` keeps native Wayland, where the pen has no input.
//!
//! Preferences › Tools › Use Tablet Pressure off makes a pen paint like a mouse. Flipping the pen
//! to its eraser end selects the Eraser tool and flipping back restores the previous tool, as in
//! Photoshop.
//!
//! Automation simulates a pen with `ui.pointer` events carrying `pressure`, `tiltX`, `tiltY`
//! and `rotation`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};

/// One stylus reading. Tilt is in degrees (-90..90, W3C Pointer Events convention), rotation in
/// degrees 0..360.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PenSample {
    pub pressure: f32,
    pub tilt_x: f32,
    pub tilt_y: f32,
    pub rotation: f32,
    /// The pen's eraser end is in use.
    pub eraser: bool,
}

impl Default for PenSample {
    fn default() -> Self {
        Self { pressure: 1.0, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, eraser: false }
    }
}

impl PenSample {
    /// Clamp to the documented ranges (and replace non-finite values).
    pub fn sanitized(self) -> Self {
        let f = |v: f32, lo: f32, hi: f32, d: f32| if v.is_finite() { v.clamp(lo, hi) } else { d };
        Self {
            pressure: f(self.pressure, 0.0, 1.0, 1.0),
            tilt_x: f(self.tilt_x, -90.0, 90.0, 0.0),
            tilt_y: f(self.tilt_y, -90.0, 90.0, 0.0),
            // `rem_euclid` rounds tiny negative angles up to exactly 360.
            rotation: if self.rotation.is_finite() { self.rotation.rem_euclid(360.0) % 360.0 } else { 0.0 },
            eraser: self.eraser,
        }
    }
}

/// Most samples a [`StylusFeed`] keeps between two frames (a stalled UI drops the oldest).
const FEED_QUEUE: usize = 1024;

#[derive(Debug, Default)]
struct FeedState {
    /// The current sample (`None` = no pen down).
    latest: Option<PenSample>,
    /// Every sample set since the canvas last drained them, oldest first.
    queue: VecDeque<PenSample>,
}

/// Shared state a platform backend writes pen samples into. Cloning shares it. Each sample both
/// becomes the current one and is queued, so a frame sees every sample the pen reported, in
/// order, not just the last one (the pressure of each pointer move is interpolated from them).
#[derive(Clone, Debug, Default)]
pub struct StylusFeed(Arc<Mutex<FeedState>>);

impl StylusFeed {
    /// The current sample (`None` = no pen down); a sample is also queued for [`Self::drain`].
    pub fn set(&self, s: Option<PenSample>) {
        let mut g = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let s = s.map(PenSample::sanitized);
        g.latest = s;
        if let Some(s) = s {
            if g.queue.len() >= FEED_QUEUE {
                g.queue.pop_front();
            }
            g.queue.push_back(s);
        }
    }

    pub fn get(&self) -> Option<PenSample> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).latest
    }

    /// The samples set since the last drain, oldest first.
    pub fn drain(&self) -> Vec<PenSample> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).queue.drain(..).collect()
    }
}

/// Per-app stylus state.
#[derive(Clone, Debug)]
pub struct Stylus {
    /// Pen samples pushed by the platform (desktop tablet monitor, web runner) or by automation.
    pub feed: StylusFeed,
    /// Preferences › Tools › Use Tablet Pressure: off, a pen paints like a mouse.
    pub use_pressure: bool,
    /// Preferences › Tools › Pressure Curve (`None`: linear), with the points it was built from.
    pressure_curve: Option<photocraft_engine::prefs::PressureCurve>,
    curve_points: Vec<[f32; 2]>,
    /// The pen end last seen (`Some(true)` = eraser), for the eraser tool switch.
    end: Option<bool>,
    /// The tool to restore when the pen tip comes back after the eraser end switched tools.
    pub(crate) tool_before_eraser: Option<crate::state::Tool>,
    /// Force of the touch/pen contact currently down (egui `Event::Touch`).
    touch: Option<f32>,
    /// The touch contacts currently down, so a touch move outside them is a hovering pen.
    contacts: Vec<egui::TouchId>,
    /// A pen reported through touch events (it hovered, or a contact carried a force): its
    /// contacts without a force are at pressure 0, not a finger's full pressure.
    touch_pen: bool,
    /// The contact ended this frame: keep its force for this frame's last tool events, clear next frame.
    lifted: bool,
    /// This frame's pen samples, oldest first (the feed's queue and the touch forces).
    frame: Vec<PenSample>,
    /// The last pen sample of an earlier frame: where this frame's interpolation starts.
    prev: Option<PenSample>,
    /// The sample [`Self::select`] interpolated for the pointer move being processed.
    current: Option<PenSample>,
    /// The highest pen pressure (before the curve) since the drag began, for a tap's one dab.
    peak: f32,
    /// The pen's touch contacts in the raw input, so a held pen is no press-and-hold right click.
    pub(crate) pen_contacts: PenContacts,
    /// The egui frame [`Self::update_for_frame`] last ran in.
    updated_frame: Option<u64>,
    /// Tilt X, tilt Y, rotation of each point of the current drag (parallel to its points).
    pub(crate) stroke: Vec<[f32; 3]>,
    /// Time in milliseconds of each point of the current drag (parallel to its points), from
    /// [`Stylus::clock_ms`]. Speed spacing, airbrush build-up and smoothing catch-up use it.
    pub(crate) times: Vec<f64>,
    /// Time of the tool event being processed, in milliseconds. The canvas spreads a frame's
    /// samples between the previous and the current frame time (egui's events carry none).
    pub(crate) clock_ms: f64,
    /// Rolling window of recent hover coordinates `(doc_xy, time_ms)` tracked while pen is in air.
    hover_history: VecDeque<([f64; 2], f64)>,
}

impl Default for Stylus {
    fn default() -> Self {
        Self {
            feed: StylusFeed::default(),
            use_pressure: true,
            pressure_curve: None,
            curve_points: Vec::new(),
            end: None,
            tool_before_eraser: None,
            touch: None,
            contacts: Vec::new(),
            touch_pen: false,
            lifted: false,
            frame: Vec::new(),
            prev: None,
            current: None,
            peak: 0.0,
            pen_contacts: PenContacts::default(),
            updated_frame: None,
            stroke: Vec::new(),
            times: Vec::new(),
            clock_ms: 0.0,
            hover_history: VecDeque::new(),
        }
    }
}

impl Stylus {
    /// [`Self::update`] once per egui frame: the canvas runs once per view, and egui can run a
    /// frame in several passes; a second update would drain an empty queue and lose the frame's
    /// pen samples.
    pub fn update_for_frame(&mut self, frame: u64, events: &[egui::Event]) {
        if self.updated_frame.replace(frame) != Some(frame) {
            self.update(events);
        }
    }

    /// Track touch/pen contacts from this frame's input events.
    pub fn update(&mut self, events: &[egui::Event]) {
        if std::mem::take(&mut self.lifted) {
            self.touch = None;
        }
        if let Some(last) = self.frame.last() {
            self.prev = Some(*last);
        }
        self.frame = self.feed.drain();
        let has_feed = !self.frame.is_empty();
        self.current = None;
        for e in events {
            if let egui::Event::Touch { id, phase, force, .. } = e {
                let force = force.filter(|f| f.is_finite()).map(|f| f.clamp(0.0, 1.0));
                match phase {
                    egui::TouchPhase::Start | egui::TouchPhase::Move => {
                        let start = *phase == egui::TouchPhase::Start;
                        if start {
                            // A new contact keeps its force into the next frame, even when an
                            // earlier contact lifted in this one.
                            self.lifted = false;
                            if !self.contacts.contains(id) {
                                if self.contacts.len() >= 16 {
                                    self.contacts.remove(0);
                                }
                                self.contacts.push(*id);
                            }
                        }
                        let down = start || self.contacts.contains(id);
                        // Only a pen moves without touching, and only a pen (or a force-sensing
                        // screen) reports a force.
                        if !down || force.is_some() {
                            self.touch_pen = true;
                        }
                        // Windows reports a pen pressed lighter than its first pressure level with
                        // no force at all (winit maps pressure 0 to `None`): a light tap is
                        // pressure 0, never a mouse click's full pressure. A finger on a screen
                        // without force sensing stays at full pressure.
                        let force = force.or((down && self.touch_pen && (start || self.touch.is_none())).then_some(0.0));
                        if let Some(f) = force {
                            self.touch = Some(f);
                            if !has_feed {
                                self.frame.push(PenSample { pressure: f, ..Default::default() });
                            }
                        }
                    }
                    egui::TouchPhase::End | egui::TouchPhase::Cancel => {
                        self.lifted = true;
                        self.contacts.retain(|c| c != id);
                    }
                }
            }
        }
        // A pen reports pressure 0 while it hovers and as it lifts (X11, macOS): that is no
        // contact, so a frame that also touched keeps only its contact samples. A stroke's last
        // moves then keep the pressure the pen left with, and a tap its own (#1798).
        if self.frame.iter().any(|s| s.pressure > 0.0) {
            self.frame.retain(|s| s.pressure > 0.0);
        }
        self.peak = self.frame.iter().fold(self.peak, |m, s| m.max(s.pressure));
    }

    /// The pen's current sample. When the pen touched during this frame but has lifted since
    /// (the feed is cleared or back at pressure 0 by the time the UI runs, as for a tap shorter
    /// than a frame), the frame's last contact sample: never full pressure as for a mouse, and
    /// never the lift's 0.
    fn latest(&self) -> Option<PenSample> {
        let feed = self.feed.get();
        match self.frame.last() {
            Some(s) if s.pressure > 0.0 && feed.is_none_or(|f| f.pressure <= 0.0) => Some(*s),
            _ => feed,
        }
    }

    /// The current pen sample, `None` for a mouse (and for any pen while Use Tablet Pressure is
    /// off).
    pub fn sample(&self) -> Option<PenSample> {
        if !self.use_pressure {
            return None;
        }
        let s = self.current.or_else(|| self.latest()).or(self.touch.map(|pressure| PenSample { pressure, ..Default::default() }))?;
        // The pen's pressure through Preferences › Tools › Pressure Curve, before any brush sees it.
        Some(match &self.pressure_curve {
            Some(c) => PenSample { pressure: c.eval(s.pressure), ..s },
            None => s,
        })
    }

    /// Preferences › Tools › Pressure Curve (rebuilt only when the points change).
    pub fn set_pressure_curve(&mut self, points: &[[f32; 2]]) {
        if self.curve_points != points {
            self.curve_points = points.to_vec();
            let c = photocraft_engine::prefs::PressureCurve::new(points);
            self.pressure_curve = (!c.is_linear()).then_some(c);
        }
    }

    /// Choose the pen sample for move `k` of the `n` pointer moves this frame delivered: the pen
    /// reports at its own rate, so each move takes the samples interpolated at its share of the
    /// frame (from the previous frame's last sample), instead of all sharing the latest one. A
    /// mouse, or a frame without pen samples, keeps the current sample.
    pub(crate) fn select(&mut self, k: usize, n: usize) {
        self.current = None;
        if !self.use_pressure || n == 0 || self.frame.is_empty() || self.sample().is_none() {
            return;
        }
        let m = self.frame.len();
        let start = self.prev.or(self.frame.first().copied()).unwrap_or_default();
        let at = |i: usize| if i == 0 { start } else { self.frame.get(i - 1).copied().unwrap_or(start) };
        let pos = (k + 1).min(n) as f32 / n as f32 * m as f32;
        let i = pos.floor() as usize;
        let (a, b, t) = (at(i), at((i + 1).min(m)), pos - pos.floor());
        let lerp = |x: f32, y: f32| x + (y - x) * t;
        // Rotation takes the shorter way round.
        let dr = (b.rotation - a.rotation + 540.0).rem_euclid(360.0) - 180.0;
        self.current = Some(
            PenSample {
                pressure: lerp(a.pressure, b.pressure),
                tilt_x: lerp(a.tilt_x, b.tilt_x),
                tilt_y: lerp(a.tilt_y, b.tilt_y),
                rotation: a.rotation + dr * t,
                eraser: b.eraser,
            }
            .sanitized(),
        );
    }

    /// Back to the current sample after a frame's moves.
    pub(crate) fn clear_selection(&mut self) {
        self.current = None;
    }

    /// Did the pen just flip to its eraser end (`Some(true)`) or back to its tip (`Some(false)`)?
    /// Each flip is reported once; a mouse in between changes nothing.
    pub(crate) fn take_end_flip(&mut self) -> Option<bool> {
        let eraser = self.feed.get()?.eraser;
        let flipped = self.end.map_or(eraser, |e| e != eraser);
        self.end = Some(eraser);
        flipped.then_some(eraser)
    }

    /// Switch to the Eraser when the pen's eraser end comes in, and back to the previous tool when
    /// the tip does (Photoshop). Not during a drag. Returns whether the tool changed.
    pub fn sync_eraser_tool(app: &mut crate::PhotocraftApp) -> bool {
        use crate::state::Tool;
        if app.drag.is_some() {
            return false;
        }
        match app.stylus.take_end_flip() {
            Some(true) if app.ui.tool != Tool::Eraser => {
                app.stylus.tool_before_eraser = Some(app.ui.tool);
                app.ui.tool = Tool::Eraser;
                true
            }
            Some(false) => match app.stylus.tool_before_eraser.take() {
                Some(t) if app.ui.tool == Tool::Eraser => {
                    app.ui.tool = t;
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Pressure for the next tool event (1 for a mouse).
    pub fn pressure(&self) -> f32 {
        self.sample().map_or(1.0, |s| s.pressure)
    }

    /// Pressure for a tap, a drag released where it was pressed: the highest the pen pressed
    /// since the drag began (through the pressure curve), as the first sample of a contact is
    /// often far lighter than the tap. 1 for a mouse.
    pub(crate) fn tap_pressure(&self) -> f32 {
        let Some(s) = self.sample() else { return 1.0 };
        let peak = match &self.pressure_curve {
            Some(c) => c.eval(self.peak),
            None => self.peak,
        };
        if self.peak > 0.0 { peak.max(s.pressure) } else { s.pressure }
    }

    /// Start recording a drag's tilt/rotation.
    pub(crate) fn begin_stroke(&mut self) {
        // A new stroke's first moves interpolate from this frame's samples, never from a sample
        // left over from an earlier stroke.
        self.prev = None;
        self.peak = self.frame.iter().fold(0.0, |m, s| m.max(s.pressure));
        self.stroke.clear();
        self.times.clear();
        self.record_point();
    }

    /// Record the tilt/rotation and time of a point just added to the drag.
    pub(crate) fn record_point(&mut self) {
        let s = self.sample().unwrap_or_default();
        self.stroke.push([s.tilt_x, s.tilt_y, s.rotation]);
        self.times.push(self.clock_ms);
    }

    /// Time of the drag's last recorded point.
    pub(crate) fn last_point_ms(&self) -> Option<f64> {
        self.times.last().copied()
    }

    /// Time of drag point `i` (the last known time past the end).
    pub(crate) fn point_ms(&self, i: usize) -> f64 {
        self.times.get(i).or(self.times.last()).copied().unwrap_or(0.0)
    }

    /// Stroke points for `paint.stroke`: `[x, y, pressure]`, extended with
    /// `tiltX, tiltY, rotation` when the drag carried any tilt or rotation, and with `timeMs`
    /// (from the drag's first point) when its points carry times.
    pub fn stroke_points(&self, points: &[[f64; 3]]) -> Vec<Vec<f64>> {
        let has_pose = self.stroke.iter().any(|t| t.iter().any(|v| *v != 0.0));
        let t0 = self.point_ms(0);
        let has_time = self.times.iter().any(|t| *t != t0);
        points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut v = vec![p[0], p[1], p[2]];
                if has_pose || has_time {
                    let t = self.stroke.get(i).or(self.stroke.last()).copied().unwrap_or_default();
                    v.extend(t.map(f64::from));
                }
                if has_time {
                    v.push(self.point_ms(i) - t0);
                }
                v
            })
            .collect()
    }

    /// Record a hover position while pen is in air.
    pub fn record_hover(&mut self, pos: [f64; 2], time_ms: f64) {
        if self.hover_history.len() >= 32 {
            self.hover_history.pop_front();
        }
        self.hover_history.push_back((pos, time_ms));
    }

    /// Compute the physical continuation point on pen lift: uses the actual hover coordinate
    /// if the pen moved continuously into the air during a stroke flick.
    pub fn hover_lift_point(&self, last_down: [f64; 2], hover_now: Option<[f64; 2]>) -> [f64; 2] {
        if let Some(h) = hover_now {
            let (dx, dy) = (h[0] - last_down[0], h[1] - last_down[1]);
            let dist = dx.hypot(dy);
            if dist >= 0.5 && dist <= 120.0 {
                return h;
            }
        }
        last_down
    }

    /// Record the tilt/rotation and time of a continuation point just added to the drag.
    pub(crate) fn record_continuation_point(&mut self, time_ms: f64) {
        let s = self.sample().unwrap_or_default();
        self.stroke.push([s.tilt_x, s.tilt_y, s.rotation]);
        self.times.push(time_ms);
    }
}

/// Parameters describing a flick/lift continuation taper.
#[derive(Clone, Copy, Debug)]
pub struct FlickTaperParams {
    pub target_dist: f64,
    pub m0: f32,
    pub exit_dir: [f64; 2],
    pub speed: f64,
    pub v_last: f64,
    pub p_last: f64,
    pub p_retained: f64,
    pub kappa: f64,
    pub is_fast_flick: bool,
    pub speed_t: f64,
}

impl Stylus {
    /// Finds the index of the visible anchor sample at the end of the stroke,
    /// pruning trailing samples that represent in-place nib decompression
    /// (samples with dropping pressure clustered within the footprint of a larger preceding sample).
    pub fn find_visible_anchor(points: &[[f64; 3]], _brush_radius: f64) -> usize {
        let n = points.len();
        if n <= 1 {
            return 0;
        }

        // Inspect trailing window of up to 6 samples
        let k = n.saturating_sub(6);
        let mut best_idx = n - 1;
        let mut max_p = points[n - 1][2];

        for i in (k..n - 1).rev() {
            if points[i][2] > max_p {
                max_p = points[i][2];
                best_idx = i;
            }
        }

        if best_idx == n - 1 {
            return n - 1;
        }

        // Check if the samples after best_idx are clustered in-place (nib decompression while stationary).
        // For stationary decompression:
        // 1. Each consecutive step between trailing samples is small (nib jitter in place: <= 2.0px).
        // 2. The total net distance from the anchor to the final sample is within a tight cluster radius (<= 3.0px).
        let anchor_pos = [points[best_idx][0], points[best_idx][1]];
        let net_dx = points[n - 1][0] - anchor_pos[0];
        let net_dy = points[n - 1][1] - anchor_pos[1];
        let net_dist = net_dx.hypot(net_dy);

        // If the pen traveled more than 3.0px away from the anchor, it was moving forward in space,
        // not lifting stationary in place.
        if net_dist > 3.0 {
            return n - 1;
        }

        let all_trailing_clustered = (best_idx + 1..n).all(|i| {
            let step_dx = points[i][0] - points[i - 1][0];
            let step_dy = points[i][1] - points[i - 1][1];
            step_dx.hypot(step_dy) <= 2.0 && points[i][2] <= max_p
        });

        if all_trailing_clustered { best_idx } else { n - 1 }
    }

    /// Evaluates trailing stroke points to determine if the pen lifted during an active stroke or flick
    /// that calls for continuation tapering, or if it is a blunt/steady/stopped stroke that must end as-is.
    ///
    /// Rules:
    /// 1. Continuous speed factor `speed_t`:
    ///    - Minimal bullet/dome tapering (`slow`) applies on a small slow velocity range (v <= 0.10 px/ms).
    ///    - Average and fast strokes transition smoothly into elongated dome and sleek calligraphy tapering
    ///      up to v >= 0.50 px/ms, ensuring strokes cutting off midway always taper.
    /// 2. Speed-dependent curvature:
    ///    - Slow: convex Gothic arch bullet dome `(1 - u^2)` maintaining body width before tapering to a point.
    ///    - Fast: Hermite cubic C^1 natural decay matching incoming stroke slope.
    /// 3. Standstill & pause rejection:
    ///    - Strokes that dwelled (> 45ms), completely stopped (v < 0.05), or braked sharply to a halt (v_last < 0.04
    ///      and v_last < speed * 0.15) reject tapering and commit as-is.
    pub fn evaluate_flick_taper(&self, raw_points: &[[f64; 3]], brush_radius: f64, hover_target: Option<[f64; 2]>) -> Option<FlickTaperParams> {
        let anchor_idx = Self::find_visible_anchor(raw_points, brush_radius);
        if anchor_idx < raw_points.len() - 1 {
            // Trailing samples are clustered in-place inside the anchor sample's footprint
            // with falling pressure. The pen came to a halt and lifted in-place; the bigger
            // brush sample at that area is the true end of the stroke, so tapering is ZERO.
            return None;
        }
        let points = raw_points;
        let n = points.len();
        if n < 2 {
            return None;
        }

        // Tap or dot press check: if total distance traversed across all points is < 2.5px, it's a dot
        let mut total_dist = 0.0f64;
        for i in 1..n {
            let dx = points[i][0] - points[i - 1][0];
            let dy = points[i][1] - points[i - 1][1];
            total_dist += dx.hypot(dy);
        }
        if total_dist < 2.5 {
            return None;
        }

        let p_last = points[n - 1][2];
        if p_last <= 0.02 {
            return None;
        }

        let t_end = self.times.get(n - 1).copied().unwrap_or(self.clock_ms);
        // Freshness/dwell check: if the pen paused or dwelled motionless before lifting (> 45ms),
        // velocity dropped to 0; this is an intentional end of stroke, not a lift or flick.
        if self.clock_ms > t_end && self.clock_ms - t_end > 45.0 {
            return None;
        }

        // Inspect 3 to 5 points before the anchor contact point
        let k = n.saturating_sub(5);
        let pt_start = points[k];
        let pt_end = points[n - 1];
        let t_start = self.times.get(k).copied().unwrap_or(t_end);

        let dt = (t_end - t_start).max(0.001);

        // Calculate spacing (arc distance) across the trailing 3-5 points
        let mut s_trail = 0.0f64;
        for i in k + 1..n {
            let dx = points[i][0] - points[i - 1][0];
            let dy = points[i][1] - points[i - 1][1];
            s_trail += dx.hypot(dy);
        }

        if s_trail < 0.5 {
            return None;
        }

        let speed = s_trail / dt; // px / ms

        // Instantaneous exit velocity of final segment
        let pt_prev = points[n - 2];
        let t_prev = self.times.get(n - 2).copied().unwrap_or(t_end);
        let dt_last = (t_end - t_prev).max(0.001);
        let dx_last = pt_end[0] - pt_prev[0];
        let dy_last = pt_end[1] - pt_prev[1];
        let dist_last = dx_last.hypot(dy_last);
        let v_last = if dt_last >= 1.0 { dist_last / dt_last } else { speed };

        // Standstill & sharp braking check:
        // - Pen came to a physical standstill before lifting (speed < 0.05 and v_last < 0.05)
        // - Pen braked sharply into a stop at the end: (v_last < 0.15 and v_last < speed * 0.40)
        if (speed < 0.05 && v_last < 0.05) || (v_last < 0.15 && v_last < speed * 0.40) {
            return None;
        }

        // Pen pressure must not have bounced upward abruptly on the final point
        if p_last > pt_prev[2] + 0.10 {
            return None;
        }

        let p_trail_max = points[k..n].iter().map(|p| p[2]).fold(0.0f64, f64::max).max(p_last);
        let p_retained = (p_last / p_trail_max).clamp(0.0, 1.0);

        // Examine samples before the final liftoff decompression (User's Point 2):
        // If the stroke body was steady (artist drew a solid line without intent to fade)
        // and maintained drawing pressure right up to pre-liftoff,
        // a sudden liftoff without slowing down must NOT sprout a needle taper from the cap.
        let is_steady_stroke = if n >= 5 {
            let p_pre = points[n - 2][2];
            let p_pre2 = points[n - 3][2];
            let k_pre = n.saturating_sub(6);
            let p_body_max = points[k_pre..n - 2].iter().map(|p| p[2]).fold(0.0f64, f64::max);
            let p_body_min = points[k_pre..n - 2].iter().map(|p| p[2]).fold(1.0f64, f64::min);
            let body_drop = p_body_max - p_body_min;

            // Case 1: single liftoff decompression point at n-1 (p_pre is still steady drawing pressure):
            let single_lift_steady = p_pre >= 0.30 && (p_body_max - p_pre) < 0.15 && body_drop < 0.15;
            // Case 2: two liftoff decompression points at n-2 and n-1 (p_pre2 is still steady drawing pressure):
            let double_lift_steady = n >= 6 && p_pre2 >= 0.30 && (p_body_max - p_pre2) < 0.15 && body_drop < 0.15;

            single_lift_steady || double_lift_steady
        } else {
            let p_all_min = points.iter().map(|p| p[2]).fold(1.0f64, f64::min);
            let p_all_max = points.iter().map(|p| p[2]).fold(0.0f64, f64::max);
            p_all_max - p_all_min < 0.10 && p_all_min >= 0.30
        };

        if is_steady_stroke {
            return None;
        }

        // Effective exit velocity:
        let v_exit = if speed < 0.40 {
            v_last.min(speed * 1.15).max(0.04)
        } else if v_last < speed * 0.50 {
            v_last.max(0.04)
        } else {
            v_last.max(speed * 0.65).max(0.04)
        };

        // Continuous speed factor speed_t in [0.0, 1.0]:
        let speed_t = ((v_exit - 0.10) / (0.50 - 0.10)).clamp(0.0, 1.0);

        let is_fast_flick = speed_t >= 0.85 && speed >= 0.60 && v_last >= 0.50 && p_retained <= 0.85;

        // Taper length scaling:
        // - As pen velocity drops, the tapering scales down directly with velocity into a short bullet line
        // - Moderate, proportional taper length scaling to avoid excessive 70-240px comets
        let r = brush_radius.max(1.0);
        let cutoff_width = 2.0 * r * p_last;
        let v_factor = (v_exit / 0.35).clamp(0.10, 1.0);
        let l_slow = (cutoff_width * 0.60 * v_factor).clamp(3.0, 14.0);
        let l_fast = (cutoff_width * (1.1 + 0.4 * (speed - 0.50).max(0.0).min(2.0))).max(v_last * 25.0 * p_last).clamp(8.0, 45.0);
        let mut target_dist = (1.0 - speed_t) * l_slow + speed_t * l_fast;

        let exit_dir = if dist_last > 1e-4 {
            [dx_last / dist_last, dy_last / dist_last]
        } else {
            let dx = pt_end[0] - pt_start[0];
            let dy = pt_end[1] - pt_start[1];
            let d = dx.hypot(dy);
            if d > 1e-4 { [dx / d, dy / d] } else { [1.0, 0.0] }
        };

        let theta_exit = exit_dir[1].atan2(exit_dir[0]);

        // Curvature detection:
        let mut kappa = 0.0f64;
        if n >= 3 {
            let p0 = points[n - 3];
            let p1 = points[n - 2];
            let p2 = points[n - 1];
            let v0x = p1[0] - p0[0];
            let v0y = p1[1] - p0[1];
            let d0 = v0x.hypot(v0y);
            let v1x = p2[0] - p1[0];
            let v1y = p2[1] - p1[1];
            let d1 = v1x.hypot(v1y);

            if d0 > 0.5 && d1 > 0.5 {
                let th0 = v0y.atan2(v0x);
                let th1 = v1y.atan2(v1x);
                let dth = (th1 - th0 + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI;
                kappa = (dth / d1).clamp(-0.035, 0.035);
            }
        }

        // In-air hover cursor travel and bounding (User's Point 3):
        // When the pen lifts, look how far the pen travels while hovering in the stroke direction.
        // The tapering should NEVER surpass the cursor!
        if let Some(hover) = hover_target {
            let h_dx = hover[0] - pt_end[0];
            let h_dy = hover[1] - pt_end[1];
            let hover_forward = h_dx * exit_dir[0] + h_dy * exit_dir[1];
            let hover_lateral = (-h_dx * exit_dir[1] + h_dy * exit_dir[0]).abs();

            if hover_forward <= 1.5 {
                // Pen cursor did not travel forward along stroke direction (stationary lift or lift-and-hover in place).
                // Never fabricate a continuation that shoots past the cursor!
                return None;
            }

            if hover_lateral > hover_forward * 1.5 {
                // Cursor moved sideways away from stroke direction -> zero tapering.
                return None;
            }

            // Tapering can never surpass the cursor's forward progress:
            target_dist = target_dist.min(hover_forward);

            if speed_t >= 0.70 {
                let h_dist = h_dx.hypot(h_dy);
                if h_dist >= 2.0 && h_dist <= 150.0 {
                    let h_th = h_dy.atan2(h_dx);
                    let diff_th = (h_th - theta_exit + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI;
                    if diff_th.abs() < 1.0 {
                        let target_kappa = (2.0 * diff_th / h_dist).clamp(-0.035, 0.035);
                        kappa = 0.4 * kappa + 0.6 * target_kappa;
                    }
                }
            }
        }

        // Incoming pressure slope matching the natural curve of the stroke
        let dp = pt_end[2] - pt_start[2];
        let incoming_slope = if s_trail > 1e-4 { (dp / s_trail).min(0.0) } else { 0.0 };
        let m0 = (incoming_slope * target_dist).clamp(-1.8 * p_last, 0.0) as f32;

        Some(FlickTaperParams { target_dist, m0, exit_dir, speed, v_last, p_last, p_retained, kappa, is_fast_flick, speed_t })
    }

    /// Computes velocity-aware taper continuation points for a stroke release or flick.
    ///
    /// Extrapolates along the stroke's arc curvature (and in-air hover target if available)
    /// and smoothly decays pressure to 0.0 without sudden drops in brush thickness or blunt spikes.
    ///
    /// - For slow strokes: produces a compact convex Gothic arch bullet dome.
    /// - For medium strokes: produces an elongated bullet dome.
    /// - For fast strokes: produces a sleek calligraphy flick taper.
    ///
    /// Returns `None` if the stroke stopped, dwelled (> 45ms), or was already tapered to <= 0.02.
    pub fn flick_continuation(&self, raw_points: &mut [[f64; 3]], brush_radius: f64, hover_target: Option<[f64; 2]>) -> Option<Vec<([f64; 3], f64)>> {
        let anchor_idx = Self::find_visible_anchor(raw_points, brush_radius);
        if anchor_idx < raw_points.len() - 1 {
            return None;
        }
        let params = self.evaluate_flick_taper(raw_points, brush_radius, hover_target)?;
        let n = raw_points.len();
        if n < 2 {
            return None;
        }
        let pt_end = raw_points[n - 1];
        let t_end = self.times.get(n - 1).copied().unwrap_or(self.clock_ms);
        let l = params.target_dist;
        if l < 0.5 {
            return None;
        }
        let theta_exit = params.exit_dir[1].atan2(params.exit_dir[0]);
        let kappa = params.kappa;
        let speed = params.speed.max(0.1);
        let _p_last = pt_end[2];

        // Identify lead-in transition zone on raw_points before liftoff to normalize the tip
        // seamlessly without any sudden slope change, kink, or shoulder (Clip Studio Paint style):
        let mut total_stroke_len = 0.0f64;
        for i in 1..n {
            let dx = raw_points[i][0] - raw_points[i - 1][0];
            let dy = raw_points[i][1] - raw_points[i - 1][1];
            total_stroke_len += dx.hypot(dy);
        }

        let brush_width = 2.0 * brush_radius;
        let max_lead = total_stroke_len * 0.45;
        let target_lead = (brush_width * 2.5).clamp(30.0, 160.0).min(max_lead);

        let mut lead_dist = 0.0f64;
        let mut k = n - 1;
        let min_k = n.saturating_sub(40);
        for i in (1..n).rev() {
            let dx = raw_points[i][0] - raw_points[i - 1][0];
            let dy = raw_points[i][1] - raw_points[i - 1][1];
            lead_dist += dx.hypot(dy);
            k = i - 1;
            if lead_dist >= target_lead || k <= min_k {
                break;
            }
        }

        let p_anchor = raw_points[k][2];
        let total_l = lead_dist + l;

        // Compute incoming body slope before the anchor point:
        let body_slope = if k >= 2 {
            let dx = raw_points[k][0] - raw_points[k - 2][0];
            let dy = raw_points[k][1] - raw_points[k - 2][1];
            let d = dx.hypot(dy);
            if d > 0.1 { (raw_points[k][2] - raw_points[k - 2][2]) / d } else { 0.0 }
        } else if k == 1 {
            let dx = raw_points[1][0] - raw_points[0][0];
            let dy = raw_points[1][1] - raw_points[0][1];
            let d = dx.hypot(dy);
            if d > 0.1 { (raw_points[1][2] - raw_points[0][2]) / d } else { 0.0 }
        } else {
            0.0
        };

        let m0 = (body_slope * total_l).clamp(-2.5 * p_anchor, 0.0);

        // Normalize trailing physical points leading up to liftoff:
        // Re-evaluates thickness so the transition begins smoothly from the body without a kink
        let mut s_acc = 0.0f64;
        for i in k + 1..n {
            let dx = raw_points[i][0] - raw_points[i - 1][0];
            let dy = raw_points[i][1] - raw_points[i - 1][1];
            s_acc += dx.hypot(dy);
            let u = (s_acc / total_l).clamp(0.0, 1.0);
            let p_norm = (1.0 - u) * (1.0 - u) * ((2.0 * u + 1.0) * p_anchor + u * m0);
            raw_points[i][2] = p_norm.clamp(0.0, raw_points[i - 1][2]);
        }

        let p_exit = raw_points[n - 1][2];

        // Generate 8 smooth continuation knots along the exact same natural curve:
        let num_knots = 8;
        let mut out = Vec::with_capacity(num_knots);
        let mut curr_x = pt_end[0];
        let mut curr_y = pt_end[1];
        let mut prev_s = 0.0;

        for step in 1..=num_knots {
            let u_knot = step as f64 / num_knots as f64;
            let s_cont = u_knot * l;
            let ds = s_cont - prev_s;
            let mid_s = (prev_s + s_cont) * 0.5;
            let angle = theta_exit + kappa * mid_s * (1.0 - mid_s / (2.0 * l));
            curr_x += ds * angle.cos();
            curr_y += ds * angle.sin();
            prev_s = s_cont;

            let s_total = lead_dist + s_cont;
            let u_total = (s_total / total_l).clamp(0.0, 1.0);

            let p_u = if (u_knot - 1.0).abs() < 1e-6 {
                0.0
            } else {
                let p_norm = (1.0 - u_total) * (1.0 - u_total) * ((2.0 * u_total + 1.0) * p_anchor + u_total * m0);
                p_norm.clamp(0.0, p_exit)
            };

            let pt_time = t_end + (s_cont / speed);
            out.push(([curr_x, curr_y, p_u], pt_time));
        }

        Some(out)
    }
}

/// The touch contacts a pen makes, read from the raw input. egui takes every touch contact held
/// still past `max_click_duration` (0.8 s) for a touch screen's press-and-hold: a secondary click
/// (a context menu) that also drops the drag. On Windows a pen arrives as touch events, so a pen
/// held still opened a context menu. A pen held still is just a pen, as in Photoshop.
#[derive(Clone, Debug, Default)]
pub(crate) struct PenContacts {
    /// Touch ids that moved without touching: only a pen hovers.
    hovering: Vec<egui::TouchId>,
    /// The contacts down, and whether each is a pen.
    down: Vec<(egui::TouchId, bool)>,
}

/// Most touch ids [`PenContacts`] remembers (the oldest go first).
const CONTACTS: usize = 16;

impl PenContacts {
    /// Follow a frame's touch events. True when a pen contact was down at any point of the frame.
    pub(crate) fn observe(&mut self, events: &[egui::Event]) -> bool {
        let pressed = |f: &Option<f32>| f.is_some_and(|f| f > 0.0);
        let mut pen = self.down.iter().any(|(_, p)| *p);
        for e in events {
            let egui::Event::Touch { id, phase, force, .. } = e else { continue };
            match phase {
                egui::TouchPhase::Start => {
                    let is_pen = self.hovering.contains(id) || pressed(force);
                    self.down.retain(|(c, _)| c != id);
                    if self.down.len() >= CONTACTS {
                        self.down.remove(0);
                    }
                    self.down.push((*id, is_pen));
                    pen |= is_pen;
                }
                egui::TouchPhase::Move => match self.down.iter_mut().find(|(c, _)| c == id) {
                    Some(c) if pressed(force) => {
                        c.1 = true;
                        pen = true;
                    }
                    Some(_) => {}
                    None if !self.hovering.contains(id) => {
                        if self.hovering.len() >= CONTACTS {
                            self.hovering.remove(0);
                        }
                        self.hovering.push(*id);
                    }
                    None => {}
                },
                egui::TouchPhase::End | egui::TouchPhase::Cancel => self.down.retain(|(c, _)| c != id),
            }
        }
        pen
    }
}

/// Turn egui's touch press-and-hold off while a pen contact is down (and through the frame it
/// lifts in, so the release is the click a quick tap would be), back on otherwise for fingers.
pub(crate) fn pen_hold_is_no_right_click(ctx: &egui::Context, pen_down: bool) {
    let want = if pen_down { f64::INFINITY } else { egui::InputOptions::default().max_click_duration };
    if ctx.options(|o| o.input_options.max_click_duration) != want {
        ctx.options_mut(|o| o.input_options.max_click_duration = want);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hold a touch contact still for 1.2 s over a click-sensing area, with the press-and-hold
    /// handling the app does in `raw_input_hook`. Did the area see a secondary click?
    fn held_contact_right_clicks(hover_first: bool, force: Option<f32>) -> bool {
        let ctx = egui::Context::default();
        let mut contacts = PenContacts::default();
        let mut right_clicked = false;
        let ev = |phase, force| egui::Event::Touch { device_id: egui::TouchDeviceId(1), id: egui::TouchId(3), phase, pos: egui::pos2(50.0, 50.0), force };
        // A first frame lays the area out, so the press hits it.
        let mut frames = vec![(0.0, vec![])];
        if hover_first {
            frames.push((0.05, vec![ev(egui::TouchPhase::Move, None), egui::Event::PointerMoved(egui::pos2(50.0, 50.0))]));
        }
        frames.push((
            0.1,
            vec![
                ev(egui::TouchPhase::Start, force),
                egui::Event::PointerMoved(egui::pos2(50.0, 50.0)),
                egui::Event::PointerButton {
                    pos: egui::pos2(50.0, 50.0),
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        ));
        for t in [0.4, 0.7, 1.0, 1.3] {
            frames.push((t, vec![]));
        }
        for (time, mut events) in frames {
            // egui-winit emits the touch and its emulated pointer events together.
            let mut raw = egui::RawInput {
                time: Some(time),
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200.0, 200.0))),
                ..Default::default()
            };
            raw.events.append(&mut events);
            let pen = contacts.observe(&raw.events);
            pen_hold_is_no_right_click(&ctx, pen);
            let mut out = ctx.run_ui(raw, |ui| {
                let r = ui.allocate_rect(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200.0, 200.0)), egui::Sense::click_and_drag());
                right_clicked |= r.secondary_clicked();
            });
            out.textures_delta.clear();
        }
        right_clicked
    }

    #[test]
    fn a_held_pen_is_not_a_right_click_but_a_held_finger_still_is() {
        assert!(held_contact_right_clicks(false, None), "a finger's press-and-hold is egui's context menu");
        assert!(!held_contact_right_clicks(true, None), "a pen that hovered first");
        assert!(!held_contact_right_clicks(false, Some(0.4)), "a contact that reports pressure");
    }

    #[test]
    fn pen_contacts_follow_hover_force_and_lift() {
        use egui::TouchPhase::{Cancel, End, Move, Start};
        let t = |id, phase, force| egui::Event::Touch { device_id: egui::TouchDeviceId(1), id: egui::TouchId(id), phase, pos: egui::pos2(1.0, 1.0), force };
        let mut c = PenContacts::default();
        assert!(!c.observe(&[t(1, Move, None)]), "hovering is no contact");
        assert!(c.observe(&[t(1, Start, None)]));
        assert!(c.observe(&[]), "still down");
        assert!(c.observe(&[t(1, End, None)]), "the lift frame still counts");
        assert!(!c.observe(&[]));
        // A finger, until it reports a force.
        assert!(!c.observe(&[t(9, Start, None), t(9, Move, None)]));
        assert!(c.observe(&[t(9, Move, Some(0.5))]));
        assert!(c.observe(&[t(9, Cancel, None)]));
        assert!(!c.observe(&[]));
    }

    fn touch(phase: egui::TouchPhase, force: Option<f32>) -> egui::Event {
        egui::Event::Touch { device_id: egui::TouchDeviceId(1), id: egui::TouchId(0), phase, pos: egui::pos2(1.0, 1.0), force }
    }

    fn pen(pressure: f32) -> PenSample {
        PenSample { pressure, ..Default::default() }
    }

    #[test]
    fn the_feed_queues_every_sample_in_order_and_keeps_the_latest() {
        let feed = StylusFeed::default();
        for p in [0.1, 0.2, 0.3] {
            feed.set(Some(pen(p)));
        }
        assert_eq!(feed.get().map(|s| s.pressure), Some(0.3));
        assert_eq!(feed.drain().iter().map(|s| s.pressure).collect::<Vec<_>>(), vec![0.1, 0.2, 0.3]);
        assert!(feed.drain().is_empty(), "drained once");
        feed.set(None);
        assert_eq!(feed.get(), None, "a lifted pen is not queued");
        assert!(feed.drain().is_empty());
        // A UI that stops draining keeps only the newest samples.
        for i in 0..FEED_QUEUE + 10 {
            feed.set(Some(pen((i % 100) as f32 / 100.0)));
        }
        assert_eq!(feed.drain().len(), FEED_QUEUE);
    }

    #[test]
    fn each_move_takes_its_share_of_the_frames_pen_samples() {
        let mut s = Stylus::default();
        s.feed.set(Some(pen(0.2)));
        s.update(&[]);
        s.begin_stroke();
        // Next frame: the pen reported 0.4, 0.6 and 0.8 while six pointer moves arrived.
        for p in [0.4, 0.6, 0.8] {
            s.feed.set(Some(pen(p)));
        }
        s.update(&[]);
        let got: Vec<f32> = (0..6)
            .map(|k| {
                s.select(k, 6);
                s.pressure()
            })
            .collect();
        let want = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-6), "{got:?}");
        s.clear_selection();
        assert_eq!(s.pressure(), 0.8, "back to the current sample");
    }

    #[test]
    fn the_pressure_curve_reshapes_pen_pressure_but_not_the_mouse() {
        let mut s = Stylus::default();
        s.set_pressure_curve(&[[0.0, 0.0], [0.3, 0.6], [1.0, 1.0]]);
        s.feed.set(Some(pen(0.3)));
        s.update(&[]);
        assert!((s.pressure() - 0.6).abs() < 1e-5, "{}", s.pressure());
        s.set_pressure_curve(&[[0.0, 0.0], [1.0, 1.0]]);
        assert!((s.pressure() - 0.3).abs() < 1e-6, "linear again");
        // A mouse is full pressure whatever the curve.
        let mut m = Stylus::default();
        m.set_pressure_curve(&[[0.0, 0.0], [1.0, 0.2]]);
        m.update(&[egui::Event::PointerMoved(egui::pos2(1.0, 1.0))]);
        assert_eq!(m.pressure(), 1.0);
    }

    #[test]
    fn a_second_update_in_the_same_frame_keeps_its_samples() {
        // The canvas runs once per view, and egui may run a frame in several passes.
        let mut s = Stylus::default();
        s.feed.set(Some(pen(0.4)));
        s.feed.set(Some(pen(0.8)));
        s.update_for_frame(7, &[]);
        s.update_for_frame(7, &[]);
        s.select(0, 2);
        assert!((s.pressure() - 0.4).abs() < 1e-6, "{}", s.pressure());
    }

    #[test]
    fn a_new_stroke_never_starts_from_an_old_strokes_pressure() {
        let mut s = Stylus::default();
        s.feed.set(Some(pen(0.9)));
        s.update(&[]);
        s.feed.set(Some(pen(0.1)));
        s.update(&[]);
        s.begin_stroke();
        s.select(0, 2);
        assert!((s.pressure() - 0.1).abs() < 1e-6, "{}", s.pressure());
    }

    #[test]
    fn a_mouse_or_use_pressure_off_paints_at_full_pressure() {
        let mut s = Stylus::default();
        s.update(&[egui::Event::PointerMoved(egui::pos2(1.0, 1.0))]);
        s.select(0, 1);
        assert_eq!(s.pressure(), 1.0);
        s.feed.set(Some(pen(0.3)));
        s.use_pressure = false;
        s.update(&[]);
        s.select(0, 1);
        assert_eq!(s.pressure(), 1.0);
    }

    #[test]
    fn rotation_between_samples_takes_the_short_way_round() {
        let mut s = Stylus::default();
        s.feed.set(Some(PenSample { rotation: 350.0, ..pen(1.0) }));
        s.update(&[]);
        s.feed.set(Some(PenSample { rotation: 10.0, ..pen(1.0) }));
        s.update(&[]);
        s.select(0, 2);
        let r = s.sample().map_or(-1.0, |p| p.rotation);
        assert!(r.abs() < 1e-3 || (r - 360.0).abs() < 1e-3, "halfway from 350° to 10° is 0°, got {r}");
    }

    #[test]
    fn mouse_is_full_pressure() {
        let mut s = Stylus::default();
        s.update(&[egui::Event::PointerMoved(egui::pos2(3.0, 4.0))]);
        assert_eq!(s.pressure(), 1.0);
        assert_eq!(s.sample(), None);
    }

    #[test]
    fn touch_force_drives_pressure_until_lift() {
        let mut s = Stylus::default();
        s.update(&[touch(egui::TouchPhase::Start, Some(0.25))]);
        assert_eq!(s.pressure(), 0.25);
        s.update(&[touch(egui::TouchPhase::Move, Some(0.8)), touch(egui::TouchPhase::Move, Some(1.7))]);
        assert_eq!(s.pressure(), 1.0, "clamped");
        s.update(&[touch(egui::TouchPhase::Move, Some(f32::NAN))]);
        assert_eq!(s.pressure(), 1.0);
        s.update(&[touch(egui::TouchPhase::Move, Some(0.4))]);
        assert_eq!(s.pressure(), 0.4);
        // The lift frame still paints at the last force; the next frame is back to the mouse.
        s.update(&[touch(egui::TouchPhase::End, None)]);
        assert_eq!(s.pressure(), 0.4);
        s.update(&[]);
        assert_eq!(s.pressure(), 1.0);
    }

    fn touch_id(id: u64, phase: egui::TouchPhase, force: Option<f32>) -> egui::Event {
        egui::Event::Touch { device_id: egui::TouchDeviceId(1), id: egui::TouchId(id), phase, pos: egui::pos2(1.0, 1.0), force }
    }

    #[test]
    fn a_light_windows_pen_tap_is_not_full_pressure() {
        use egui::TouchPhase::{End, Move, Start};
        // winit on Windows reports a pen pressed below its first pressure level with no force
        // (pressure 0 maps to `None`). The pen hovered first, so the tap is pressure 0, not a
        // mouse click's 1.
        let mut s = Stylus::default();
        s.update(&[touch_id(5, Move, None)]);
        assert_eq!(s.pressure(), 1.0, "hovering is no contact");
        s.update(&[touch_id(5, Start, None), touch_id(5, End, None)]);
        assert_eq!(s.pressure(), 0.0);
        assert_eq!(s.tap_pressure(), 0.0);
        s.update(&[]);
        assert_eq!(s.pressure(), 1.0, "lifted");
        // A pen known from an earlier force: a stroke pressed before its first level starts at 0
        // and then follows the pen.
        let mut s = Stylus::default();
        s.update(&[touch_id(7, Start, Some(0.5)), touch_id(7, End, None)]);
        s.update(&[]);
        s.update(&[touch_id(8, Start, None)]);
        assert_eq!(s.pressure(), 0.0);
        s.update(&[touch_id(8, Move, Some(0.3))]);
        assert_eq!(s.pressure(), 0.3);
        // A force-less move mid-stroke keeps the last force.
        s.update(&[touch_id(8, Move, None)]);
        assert_eq!(s.pressure(), 0.3);
    }

    #[test]
    fn a_finger_without_force_still_paints_at_full_pressure() {
        use egui::TouchPhase::{End, Move, Start};
        let mut s = Stylus::default();
        s.update(&[touch_id(1, Start, None), touch_id(1, Move, None)]);
        assert_eq!(s.pressure(), 1.0);
        s.update(&[touch_id(1, End, None)]);
        s.update(&[]);
        assert_eq!(s.pressure(), 1.0);
    }

    #[test]
    fn a_new_tap_in_the_frame_an_old_one_lifted_keeps_its_force() {
        use egui::TouchPhase::{End, Start};
        let mut s = Stylus::default();
        s.update(&[touch_id(1, Start, Some(0.6)), touch_id(1, End, None), touch_id(2, Start, Some(0.2))]);
        assert_eq!(s.pressure(), 0.2);
        s.update(&[]);
        assert_eq!(s.pressure(), 0.2, "the second contact is still down");
    }

    #[test]
    fn feed_wins_and_is_sanitized() {
        let mut s = Stylus::default();
        s.update(&[touch(egui::TouchPhase::Start, Some(0.3))]);
        s.feed.set(Some(PenSample { pressure: 0.6, tilt_x: 120.0, tilt_y: -30.0, rotation: -90.0, eraser: false }));
        assert_eq!(s.sample(), Some(PenSample { pressure: 0.6, tilt_x: 90.0, tilt_y: -30.0, rotation: 270.0, eraser: false }));
        s.feed.set(None);
        assert_eq!(s.pressure(), 0.3);
    }

    #[test]
    fn pen_samples_reach_the_brush_stroke() {
        use crate::canvas::{ToolEvent, tool_event};
        use serde_json::json;
        let mut app = crate::PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.session.execute("file.new", json!({"width": 120, "height": 80})).unwrap();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.ui.tool = crate::state::Tool::Brush;
        let m = egui::Modifiers::NONE;
        app.stylus.feed.set(Some(PenSample { pressure: 0.2, tilt_x: 40.0, tilt_y: 0.0, rotation: 30.0, eraser: false }));
        let pr = app.stylus.pressure();
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 40.0, pressure: pr }, m);
        app.stylus.feed.set(Some(PenSample { pressure: 0.9, tilt_x: 10.0, tilt_y: -5.0, rotation: 60.0, eraser: false }));
        let pr = app.stylus.pressure();
        tool_event(&mut app, ToolEvent::Move { x: 100.0, y: 40.0, pressure: pr }, m);
        tool_event(&mut app, ToolEvent::Up { x: 100.0, y: 40.0 }, m);
        let (id, p) = app.session.journal.iter().rev().find(|(id, _)| id == "paint.stroke").cloned().unwrap();
        assert_eq!(id, "paint.stroke");
        assert_eq!(p["points"][0], json!([10.0, 40.0, 0.2f32 as f64, 40.0, 0.0, 30.0]));
        assert_eq!(p["points"][1], json!([100.0, 40.0, 0.9f32 as f64, 10.0, -5.0, 60.0]));
        // A mouse stroke stays at full pressure with plain [x, y, p] points.
        app.stylus.feed.set(None);
        let pr = app.stylus.pressure();
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 60.0, pressure: pr }, m);
        tool_event(&mut app, ToolEvent::Up { x: 90.0, y: 60.0 }, m);
        let (_, p) = app.session.journal.iter().rev().find(|(id, _)| id == "paint.stroke").cloned().unwrap();
        assert_eq!(p["points"][0], json!([10.0, 60.0, 1.0]));
    }

    #[test]
    fn stroke_points_carry_tilt_only_when_present() {
        let mut s = Stylus::default();
        let pts = [[0.0, 0.0, 0.5], [10.0, 0.0, 0.7], [20.0, 0.0, 0.7]];
        s.begin_stroke();
        s.record_point();
        assert_eq!(s.stroke_points(&pts)[1], vec![10.0, 0.0, 0.7]);
        s.feed.set(Some(PenSample { pressure: 0.7, tilt_x: 30.0, tilt_y: 10.0, rotation: 45.0, eraser: false }));
        s.record_point();
        let out = s.stroke_points(&pts);
        assert_eq!(out[0], vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.0]);
        assert_eq!(out[2], vec![20.0, 0.0, 0.7, 30.0, 10.0, 45.0]);
    }

    #[test]
    fn use_pressure_off_ignores_pen_and_touch() {
        let mut s = Stylus { use_pressure: false, ..Default::default() };
        s.update(&[touch(egui::TouchPhase::Start, Some(0.3))]);
        s.feed.set(Some(PenSample { pressure: 0.2, tilt_x: 40.0, ..Default::default() }));
        assert_eq!((s.sample(), s.pressure()), (None, 1.0));
        s.begin_stroke();
        assert_eq!(s.stroke_points(&[[1.0, 2.0, 1.0]]), vec![vec![1.0, 2.0, 1.0]], "no tilt either");
    }

    #[test]
    fn eraser_end_switches_tools_once_and_never_mid_drag() {
        use crate::canvas::{ToolEvent, tool_event};
        use crate::state::Tool;
        let mut app = crate::PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.session.execute("file.new", serde_json::json!({"width": 60, "height": 60})).unwrap();
        app.ui.tool = Tool::Brush;
        let pen = |eraser| Some(PenSample { pressure: 0.5, eraser, ..Default::default() });
        // A mouse never switches.
        assert!(!Stylus::sync_eraser_tool(&mut app));
        app.stylus.feed.set(pen(true));
        assert!(Stylus::sync_eraser_tool(&mut app));
        assert_eq!(app.ui.tool, Tool::Eraser);
        assert!(!Stylus::sync_eraser_tool(&mut app), "reported once");
        // Flipping back during a drag waits for the drag to end.
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 0.5 }, egui::Modifiers::NONE);
        app.stylus.feed.set(pen(false));
        assert!(!Stylus::sync_eraser_tool(&mut app));
        assert_eq!(app.ui.tool, Tool::Eraser);
        tool_event(&mut app, ToolEvent::Up { x: 20.0, y: 10.0 }, egui::Modifiers::NONE);
        assert!(Stylus::sync_eraser_tool(&mut app));
        assert_eq!(app.ui.tool, Tool::Brush, "the tip restores the previous tool");
        // The eraser end while the Eraser is already chosen remembers nothing to restore.
        app.ui.tool = Tool::Eraser;
        app.stylus.feed.set(pen(true));
        assert!(!Stylus::sync_eraser_tool(&mut app));
        app.stylus.feed.set(pen(false));
        assert!(!Stylus::sync_eraser_tool(&mut app));
        assert_eq!(app.ui.tool, Tool::Eraser);
    }

    #[test]
    fn hover_movement_and_lift_tapering() {
        use crate::canvas::{ToolEvent, tool_event};
        use crate::state::Tool;

        let mut app = crate::PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.session.execute("file.new", serde_json::json!({"width": 100, "height": 100})).unwrap();
        app.ui.tool = Tool::Brush;

        // 1. Hover recording
        app.stylus.record_hover([10.0, 10.0], 100.0);
        app.stylus.record_hover([15.0, 12.0], 116.0);
        assert_eq!(app.stylus.hover_history.len(), 2);

        // 2. Hover lift continuation
        let lift = app.stylus.hover_lift_point([50.0, 50.0], Some([55.0, 52.0]));
        assert_eq!(lift, [55.0, 52.0]);
        // Discontinuous hover jump (> 120px) is ignored
        let lift_jump = app.stylus.hover_lift_point([50.0, 50.0], Some([300.0, 300.0]));
        assert_eq!(lift_jump, [50.0, 50.0]);

        // 3. Live stylus stroke releases with 0 pressure at hover point
        app.stylus.feed.set(Some(PenSample { pressure: 0.8, ..Default::default() }));
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 0.8 }, egui::Modifiers::NONE);
        tool_event(&mut app, ToolEvent::Move { x: 30.0, y: 15.0, pressure: 0.5 }, egui::Modifiers::NONE);
        // Stylus lifted and moved to [38.0, 18.0] in the air
        app.hover_doc = Some([38.0, 18.0]);
        tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 15.0 }, egui::Modifiers::NONE);
        // The stroke committed cleanly with needle-sharp taper to pressure 0
        assert!(app.drag.is_none());
    }

    #[test]
    fn flick_continuation_preserves_c1_pressure_slope_and_needle_taper() {
        let mut s = Stylus::default();
        s.clock_ms = 0.0;
        s.times = vec![0.0, 16.0, 32.0];
        let mut pts = [[10.0, 10.0, 0.7], [35.0, 15.0, 0.5], [65.0, 20.0, 0.35]];

        // 40px brush -> radius 20px
        let continuation = s.flick_continuation(&mut pts, 20.0, None);
        assert!(continuation.is_some(), "fast flick with high remaining pressure must produce continuation");
        let continuation = continuation.unwrap();
        assert_eq!(continuation.len(), 8, "generates 8 Hermite-interpolated knots");

        // Coordinates must extend along the stroke exit direction (positive X)
        let mut prev_x = pts[2][0];
        let mut prev_p = pts[2][2];
        for (pt, time) in &continuation {
            assert!(pt[0] > prev_x, "continuation point {} must project forward past {}", pt[0], prev_x);
            assert!(pt[2] <= prev_p, "pressure {} must monotonically decrease from {}", pt[2], prev_p);
            assert!(*time > 32.0, "timestamp {} must advance along continuation", time);
            prev_x = pt[0];
            prev_p = pt[2];
        }

        // Terminal point must be exactly 0.0 pressure
        assert_eq!(continuation.last().unwrap().0[2], 0.0, "final knot must reach zero pressure");

        // Verify C1 smoothness: incoming slope into exit point pts[2]
        let dx_in = 65.0f64 - 35.0f64;
        let dy_in = 20.0f64 - 15.0f64;
        let dist_in = dx_in.hypot(dy_in);
        let in_slope = (pts[2][2] - pts[1][2]) / dist_in;

        // First continuation knot slope from exit point pts[2] to knot 0
        let dx_out = continuation[0].0[0] - 65.0f64;
        let dy_out = continuation[0].0[1] - 20.0f64;
        let dist_out = dx_out.hypot(dy_out);
        let out_slope = (continuation[0].0[2] - pts[2][2]) / dist_out;

        // Slopes should be smoothly matched across the boundary
        assert!((out_slope - in_slope).abs() < 0.03, "out_slope {out_slope} must smoothly match in_slope {in_slope}");
    }

    #[test]
    fn flick_continuation_preserves_arc_curvature_on_curved_strokes() {
        let mut s = Stylus::default();
        s.times = vec![0.0, 16.0, 32.0];
        // Curving stroke turning upward (+Y): theta increases from ~0.46 to ~0.93 rad
        let mut curved_pts = [[10.0, 10.0, 0.7], [30.0, 20.0, 0.5], [45.0, 40.0, 0.35]];

        let continuation = s.flick_continuation(&mut curved_pts, 20.0, None).unwrap();
        // Tangent if straight: dx=15, dy=20 -> ratio dy/dx = 1.33
        // With positive curvature (turning upward), the continuation must curve upward even further
        let last_pt = continuation.last().unwrap().0;
        let ext_dx = last_pt[0] - 45.0;
        let ext_dy = last_pt[1] - 40.0;
        assert!(ext_dx > 0.0 && ext_dy > 0.0, "continuation must project forward");
        assert!(ext_dy / ext_dx > 1.33, "curvature must bend trajectory along the turning arc");
    }

    #[test]
    fn flick_continuation_skips_slow_strokes_and_already_tapered_strokes() {
        let mut s = Stylus::default();
        // 1. Stationary/standstill stroke: 0.5px in 50ms (< 1.0 px movement threshold)
        s.times = vec![0.0, 50.0];
        let mut slow_pts = [[10.0, 10.0, 0.8], [10.5, 10.0, 0.4]];
        assert!(s.flick_continuation(&mut slow_pts, 20.0, None).is_none(), "stationary stroke must not extrapolate");

        // 2. Already tapered stroke: remaining pressure <= 0.02
        s.times = vec![0.0, 16.0];
        let mut tapered_pts = [[10.0, 10.0, 0.5], [60.0, 10.0, 0.015]];
        assert!(s.flick_continuation(&mut tapered_pts, 20.0, None).is_none(), "already tapered stroke must not extrapolate");

        // 3. Steady stroke without prior pressure reduction: 0.85 -> 0.85 -> 0.84 at 1.87 px/ms.
        // User request: steady strokes must end with a clean rounded cap, not sprout needle tapers!
        s.times = vec![0.0, 16.0, 32.0];
        let mut cutoff_midway_pts = [[10.0, 10.0, 0.85], [40.0, 10.0, 0.85], [70.0, 10.0, 0.84]];
        let cont = s.flick_continuation(&mut cutoff_midway_pts, 20.0, None);
        assert!(cont.is_none(), "steady stroke without prior pressure reduction must reject tapering and end capped");

        // 4. Fast flick cutting off midway while rapidly decreasing in pressure:
        // 0.80 -> 0.60 -> 0.35 at speed 1.87 px/ms. MUST extrapolate and taper to 0.0!
        s.times = vec![0.0, 16.0, 32.0];
        let mut cutoff_pts = [[10.0, 10.0, 0.80], [40.0, 10.0, 0.60], [70.0, 10.0, 0.35]];
        let cont = s.flick_continuation(&mut cutoff_pts, 20.0, None);
        assert!(cont.is_some(), "midway cutoff during rapid pressure drop must extrapolate and taper");
        let knots = cont.unwrap();
        assert_eq!(knots.last().unwrap().0[2], 0.0, "final knot must reach zero pressure");
    }

    #[test]
    fn fast_long_stroke_taper_is_not_excessive() {
        let mut s = Stylus::default();
        s.times = vec![0.0, 16.0, 32.0];
        // Fast flick: 80px in 16ms = 5.0 px/ms!
        // Pressure rapidly dropping: 0.80 -> 0.55 -> 0.35 (retained = 43.75%)
        let mut fast_pts = [[10.0, 10.0, 0.80], [90.0, 10.0, 0.55], [170.0, 10.0, 0.35]];
        let cont = s.flick_continuation(&mut fast_pts, 20.0, None).unwrap();
        let first_ext_x = fast_pts.last().unwrap()[0];
        let final_ext_x = cont.last().unwrap().0[0];
        let total_taper_dist = final_ext_x - first_ext_x;
        // Taper distance scales with fast speed and width up to 240px:
        assert!(total_taper_dist <= 240.0, "taper distance {total_taper_dist} must be bounded by 240px");
        assert!(total_taper_dist >= 16.0, "taper distance {total_taper_dist} must have sufficient length to taper smoothly");
    }

    #[test]
    fn live_fast_flick_end_to_end_tapers_in_journal() {
        use crate::canvas::{ToolEvent, tool_event};
        use crate::state::Tool;

        let mut app = crate::PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.session.execute("file.new", serde_json::json!({"width": 200, "height": 100})).unwrap();
        app.session.execute("layer.new.layer", serde_json::json!({})).unwrap();
        app.session.execute("tools.setBrush", serde_json::json!({"brush": {"size": 30}})).unwrap();
        app.ui.tool = Tool::Brush;
        let m = egui::Modifiers::NONE;

        // Down at t = 0ms, pressure 0.8
        app.stylus.clock_ms = 0.0;
        app.stylus.feed.set(Some(PenSample { pressure: 0.8, ..Default::default() }));
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 40.0, pressure: 0.8 }, m);

        // Move 1 at t = 16ms, x = 40.0, pressure 0.55
        app.stylus.clock_ms = 16.0;
        app.stylus.feed.set(Some(PenSample { pressure: 0.55, ..Default::default() }));
        tool_event(&mut app, ToolEvent::Move { x: 40.0, y: 41.0, pressure: 0.55 }, m);

        // Move 2 at t = 32ms, x = 80.0, pressure 0.35 (Fast flick! 40px in 16ms = 2.5 px/ms)
        app.stylus.clock_ms = 32.0;
        app.stylus.feed.set(Some(PenSample { pressure: 0.35, ..Default::default() }));
        tool_event(&mut app, ToolEvent::Move { x: 80.0, y: 42.0, pressure: 0.35 }, m);

        // Lift at t = 32ms
        tool_event(&mut app, ToolEvent::Up { x: 80.0, y: 42.0 }, m);

        let (id, p) = app.session.journal.iter().rev().find(|(id, _)| id == "paint.stroke").cloned().unwrap();
        assert_eq!(id, "paint.stroke");

        let pts = p["points"].as_array().unwrap();
        // 3 physical points + 8 continuation knots = 11 total points
        assert_eq!(pts.len(), 11, "stroke should contain physical points plus Hermite continuation knots: {pts:?}");

        // Points must project past the lift point (x > 80.0)
        let last_pt = pts.last().unwrap().as_array().unwrap();
        assert!(last_pt[0].as_f64().unwrap() > 80.0, "continuation must project past flick lift coordinate");

        // Last point pressure must be exactly 0.0
        assert_eq!(last_pt[2].as_f64().unwrap(), 0.0, "stroke must taper all the way down to pressure 0.0");
    }

    #[test]
    fn slow_brush_stroke_minimal_tapering_uses_smooth_hermite_curve() {
        let mut s = Stylus::default();
        // Slow stroke: 10px in 100ms = 0.10 px/ms (speed_t = 0.0)
        s.times = vec![0.0, 50.0, 100.0];
        let mut slow_pts = [[10.0, 10.0, 0.80], [15.0, 10.0, 0.60], [20.0, 10.0, 0.40]];
        let cont = s.flick_continuation(&mut slow_pts, 20.0, None).expect("slow stroke must taper with bullet dome");

        assert_eq!(cont.len(), 8);
        let first_x = slow_pts.last().unwrap()[0];
        let last_x = cont.last().unwrap().0[0];
        let taper_len = last_x - first_x;
        // Compact bullet dome (length ~ 0.60 * cutoff_width = 0.60 * 16 = 9.6 px)
        assert!(taper_len <= 20.0, "slow stroke taper length {taper_len} must be compact (<= 20px)");
        assert!(taper_len >= 2.0, "slow stroke taper length {taper_len} must be at least 2px");

        // Check smooth monotonic Hermite cubic decay
        for i in 1..cont.len() {
            assert!(cont[i].0[2] <= cont[i - 1].0[2], "knot {i} pressure must be monotonically decreasing");
        }
        // Final knot reaches exactly 0.0
        assert_eq!(cont.last().unwrap().0[2], 0.0, "final knot must reach exactly 0.0 thickness");
    }

    #[test]
    fn decelerating_stroke_dropping_velocity_at_end_rejects_tapering() {
        let mut s = Stylus::default();
        // Stroke starts at 20px / 20ms = 1.0 px/ms, then brakes sharply to stop at 0.5px / 20ms = 0.025 px/ms
        s.times = vec![0.0, 20.0, 40.0];
        let mut pts = [[10.0, 10.0, 0.80], [30.0, 10.0, 0.60], [30.5, 10.0, 0.40]];
        assert!(s.flick_continuation(&mut pts, 20.0, None).is_none(), "decelerating stroke must produce NO tapering");
    }

    #[test]
    fn stroke_with_dropping_pressure_at_drawing_speed_tapers_compactly_to_sharp_tip() {
        let mut s = Stylus::default();
        // Moving stroke at 0.20 px/ms where pressure drops as artist lifts off:
        // 0.85 -> 0.85 -> 0.75 -> 0.45 -> 0.15
        s.times = vec![0.0, 35.0, 70.0, 105.0, 140.0];
        let mut pts = [[10.0, 10.0, 0.85], [18.0, 10.0, 0.85], [25.0, 10.0, 0.75], [33.0, 10.0, 0.45], [40.0, 10.0, 0.15]];
        let cont = s.flick_continuation(&mut pts, 20.0, None).expect("moving stroke with dropping pressure must taper cutoff cleanly to 0.0");
        assert_eq!(cont.len(), 8);
        let first_x = pts.last().unwrap()[0];
        let last_x = cont.last().unwrap().0[0];
        let taper_len = last_x - first_x;
        // Compact bullet dome (length ~ 3-14 px, not a 70px comet)
        assert!(taper_len <= 14.0, "taper length {taper_len} must be compact (<= 14px)");
        assert!(taper_len >= 2.0, "taper length {taper_len} must be at least 2px");
        assert_eq!(cont.last().unwrap().0[2], 0.0, "final knot must reach zero pressure");
    }

    #[test]
    fn slow_stroke_with_tightly_spaced_lift_points_rejects_tapering() {
        let mut s = Stylus::default();
        // User slows down pen at the end and lifts in-place:
        // Trailing points are spaced less than 1px apart while pressure drops from 0.85 to 0.25
        s.times = vec![0.0, 30.0, 60.0, 90.0];
        let mut pts = [
            [100.0, 100.0, 0.85],
            [120.0, 100.0, 0.85], // big sample at x = 120, R = 17px
            [120.8, 100.0, 0.55], // 0.8px away
            [121.2, 100.0, 0.25], // 0.4px away
        ];
        // Must reject tapering: the big sample at x = 120 is the real end of the stroke!
        assert!(s.flick_continuation(&mut pts, 20.0, None).is_none(), "tightly spaced samples during in-place lift must NOT add extra tapering");
    }

    #[test]
    fn hover_cursor_bounding_never_surpasses_cursor_and_rejects_stationary_hover() {
        let mut s = Stylus::default();
        s.times = vec![0.0, 16.0, 32.0];
        let mut pts = [[10.0, 10.0, 0.8], [40.0, 10.0, 0.6], [70.0, 10.0, 0.4]];

        // 1. Stationary hover: cursor is only 1.0px ahead of the stroke end (at 71.0)
        // Must reject continuation: taper cannot shoot past stationary cursor!
        let cont_stationary = s.flick_continuation(&mut pts, 20.0, Some([71.0, 10.0]));
        assert!(cont_stationary.is_none(), "stationary cursor must reject continuation");

        // 2. Sideways hover: cursor moved up into toolbar (+Y by 30px) instead of stroke direction (+X)
        let cont_sideways = s.flick_continuation(&mut pts, 20.0, Some([72.0, 40.0]));
        assert!(cont_sideways.is_none(), "sideways cursor movement must reject continuation");

        // 3. Forward hover travel: cursor flew 25px forward along the stroke to 95.0
        let cont_forward = s.flick_continuation(&mut pts, 20.0, Some([95.0, 10.0])).expect("forward hover travel must taper");
        assert_eq!(cont_forward.len(), 8);
        let final_pt = cont_forward.last().unwrap().0;
        assert!(final_pt[0] <= 95.01, "tapering point {} must NEVER surpass the cursor at 95.0", final_pt[0]);
        assert_eq!(final_pt[2], 0.0, "final knot must reach zero pressure");
    }

    #[test]
    fn short_fast_flick_with_few_samples_tapers_smoothly() {
        let mut s = Stylus::default();
        // Short fast flick: only 2 contact samples drawn by pen, but moving fast (30px in 15ms = 2.0 px/ms)
        // Pressure drops drastically from 1.0 to 0.4:
        s.times = vec![0.0, 15.0];
        let mut pts = [[10.0, 10.0, 1.0], [40.0, 10.0, 0.4]];
        // Cursor flew 30px forward in the air to 70.0:
        let cont = s.flick_continuation(&mut pts, 20.0, Some([70.0, 10.0])).expect("short fast flick must taper");
        let final_pt = cont.last().unwrap().0;
        assert!(final_pt[0] <= 70.01, "short flick must be bounded by cursor");
        assert_eq!(final_pt[2], 0.0, "final knot must reach 0.0");
    }

    #[test]
    fn steady_body_stroke_ending_with_velocity_drop_rejects_tapering() {
        let mut s = Stylus::default();
        // Steady line drawn at 0.20 px/ms, then slowing down at end:
        // Pressure is constant 0.85 across the body (mid=0.85, end=0.85)
        s.times = vec![0.0, 50.0, 100.0, 150.0, 210.0];
        let mut pts = [
            [10.0, 10.0, 0.85],
            [20.0, 10.0, 0.85],
            [30.0, 10.0, 0.85],
            [40.0, 10.0, 0.85],
            [44.0, 10.0, 0.85], // slowed down to 4px in 60ms = 0.067 px/ms
        ];
        assert!(s.flick_continuation(&mut pts, 20.0, None).is_none(), "steady body with velocity drop must reject tapering");
    }

    #[test]
    fn capped_stroke_with_sudden_lift_at_drawing_speed_rejects_tapering() {
        let mut s = Stylus::default();
        // Capped stroke drawn with steady pressure at 0.50 px/ms, lifted suddenly off the glass:
        // Points leading up to liftoff maintain steady 0.85 pressure.
        // On the final contact sample, the nib separates from the glass and pressure drops to 0.20.
        s.times = vec![0.0, 30.0, 60.0, 90.0, 120.0, 140.0];
        let mut pts = [
            [10.0, 10.0, 0.85],
            [25.0, 10.0, 0.85],
            [40.0, 10.0, 0.85],
            [55.0, 10.0, 0.84],
            [70.0, 10.0, 0.83],
            [80.0, 10.0, 0.20], // Sudden liftoff: 10px in 20ms = 0.50 px/ms
        ];
        // Must reject tapering: the stroke is a solid line intended to end with a rounded cap,
        // NOT sprout a needle taper from the cap!
        assert!(s.flick_continuation(&mut pts, 20.0, None).is_none(), "steady capped stroke with sudden liftoff must produce NO tapering");
    }

    #[test]
    fn capped_stroke_with_two_liftoff_decompression_samples_rejects_tapering() {
        let mut s = Stylus::default();
        // Hardware stylus reporting two rapid decompression samples at liftoff:
        // 0.85 -> 0.85 -> 0.85 -> 0.84 -> 0.83 -> 0.45 (lifting) -> 0.12 (breaking contact)
        s.times = vec![0.0, 30.0, 60.0, 90.0, 120.0, 135.0, 145.0];
        let mut pts =
            [[10.0, 10.0, 0.85], [25.0, 10.0, 0.85], [40.0, 10.0, 0.85], [55.0, 10.0, 0.84], [70.0, 10.0, 0.83], [80.0, 10.0, 0.45], [88.0, 10.0, 0.12]];
        assert!(s.flick_continuation(&mut pts, 20.0, None).is_none(), "capped stroke with two liftoff decompression samples must reject tapering");
    }

    #[test]
    fn tapered_stroke_normalizes_liftoff_cliff_for_seamless_c1_transition() {
        let mut s = Stylus::default();
        // Intentional tapering stroke where artist eases pressure:
        // 0.80 -> 0.65 -> 0.50 -> 0.35 -> cliff to 0.08 on liftoff
        s.times = vec![0.0, 20.0, 40.0, 60.0, 80.0];
        let mut pts = [
            [10.0, 10.0, 0.80],
            [25.0, 10.0, 0.65],
            [40.0, 10.0, 0.50],
            [55.0, 10.0, 0.35],
            [65.0, 10.0, 0.08], // Sudden mechanical drop from 0.35 to 0.08 in 10px (cliff!)
        ];
        let cont = s.flick_continuation(&mut pts, 20.0, None).expect("tapered stroke must produce continuation");

        // Continuation knots must decay strictly monotonically to 0.0 without inflection notches
        let mut prev_p = pts[4][2];
        for (knot, _) in &cont {
            assert!(knot[2] <= prev_p, "knot pressure {} must decrease monotonically from {}", knot[2], prev_p);
            prev_p = knot[2];
        }
        assert_eq!(cont.last().unwrap().0[2], 0.0, "final knot must reach zero");
    }
}
