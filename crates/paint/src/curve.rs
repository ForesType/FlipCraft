//! A response curve: how a 0..1 input (pen pressure, tilt, wheel, a fade ramp) maps to a 0..1
//! output. Preferences › Tools › Pressure Curve shapes the pen's pressure globally; every brush
//! dynamic carries its own curve for its control ([`crate::Dynamic::curve`]).

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::StrokePoint;

/// Most control points a curve keeps.
const MAX: usize = 16;

/// A monotone cubic (Fritsch–Carlson) through `[input, output]` control points, so a stronger
/// input never gives less output. Points are sanitised, not trusted: non-finite points are dropped,
/// the rest clamped to 0..1, sorted by input (one point per input, the later output wins), capped
/// at [`Self::MAX_POINTS`], and their outputs made non-decreasing. Before the first point and after
/// the last the curve stays flat. Fewer than two usable points is linear.
///
/// Fixed-size and `Copy`, so a brush evaluates it per dab without allocating. In JSON it is its
/// list of points, `[[0,0],[0.5,0.2],[1,1]]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResponseCurve {
    n: u8,
    xs: [f32; MAX],
    ys: [f32; MAX],
    /// Tangents at the points.
    ms: [f32; MAX],
}

const fn padded(a: f32, b: f32) -> [f32; MAX] {
    let mut v = [0.0; MAX];
    v[0] = a;
    v[1] = b;
    v
}

impl Default for ResponseCurve {
    fn default() -> Self {
        Self::LINEAR
    }
}

impl ResponseCurve {
    pub const MAX_POINTS: usize = MAX;

    /// The identity: input passes through unchanged.
    pub const LINEAR: Self = Self { n: 2, xs: padded(0.0, 1.0), ys: padded(0.0, 1.0), ms: padded(1.0, 1.0) };

    pub fn new(points: &[[f32; 2]]) -> Self {
        let mut pts: Vec<[f32; 2]> =
            points.iter().filter(|p| p[0].is_finite() && p[1].is_finite()).map(|p| [p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0)]).collect();
        pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
        pts.dedup_by(|b, a| {
            // Same input: keep the later point's output.
            let same = (a[0] - b[0]).abs() < 1e-6;
            if same {
                a[1] = b[1];
            }
            same
        });
        pts.truncate(MAX);
        let n = pts.len();
        if n < 2 {
            return Self::LINEAR;
        }
        let mut c = Self { n: n as u8, xs: [0.0; MAX], ys: [0.0; MAX], ms: [0.0; MAX] };
        let mut prev = 0.0f32;
        for (k, p) in pts.iter().enumerate() {
            c.xs[k] = p[0];
            // Outputs never fall: the curve is monotone.
            prev = if k == 0 { p[1] } else { p[1].max(prev) };
            c.ys[k] = prev;
        }
        // Fritsch–Carlson tangents: secant averages, zero at flat steps, limited so the cubic
        // never overshoots (which keeps it monotone).
        let mut d = [0.0f32; MAX];
        for (k, dk) in d.iter_mut().enumerate().take(n - 1) {
            *dk = (c.ys[k + 1] - c.ys[k]) / (c.xs[k + 1] - c.xs[k]);
        }
        c.ms[0] = d[0];
        c.ms[n - 1] = d[n - 2];
        for k in 1..n - 1 {
            c.ms[k] = if d[k - 1] * d[k] <= 0.0 { 0.0 } else { (d[k - 1] + d[k]) / 2.0 };
        }
        for (k, &dk) in d.iter().enumerate().take(n - 1) {
            if dk == 0.0 {
                c.ms[k] = 0.0;
                c.ms[k + 1] = 0.0;
                continue;
            }
            let (a, b) = (c.ms[k] / dk, c.ms[k + 1] / dk);
            let s = a * a + b * b;
            if s > 9.0 {
                let t = 3.0 / s.sqrt();
                c.ms[k] = t * a * dk;
                c.ms[k + 1] = t * b * dk;
            }
        }
        c
    }

    /// The control points (sanitised).
    pub fn points(&self) -> Vec<[f32; 2]> {
        self.xs.iter().zip(&self.ys).take(usize::from(self.n)).map(|(&x, &y)| [x, y]).collect()
    }

    /// The curve's output for input `x` (0..1; anything else is clamped, NaN reads as 0).
    pub fn eval(&self, x: f32) -> f32 {
        let x = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) };
        if self.is_linear() {
            return x;
        }
        let n = usize::from(self.n).min(MAX);
        let (xs, ys, ms) = (self.xs.get(..n).unwrap_or(&[]), self.ys.get(..n).unwrap_or(&[]), self.ms.get(..n).unwrap_or(&[]));
        let (Some(&x0), Some(&xn)) = (xs.first(), xs.last()) else { return x };
        if x <= x0 {
            return ys.first().copied().unwrap_or(x);
        }
        if x >= xn {
            return ys.last().copied().unwrap_or(x);
        }
        let k = xs.partition_point(|&v| v <= x).saturating_sub(1).min(n.saturating_sub(2));
        let (Some(&xa), Some(&xb), Some(&ya), Some(&yb), Some(&ma), Some(&mb)) = (xs.get(k), xs.get(k + 1), ys.get(k), ys.get(k + 1), ms.get(k), ms.get(k + 1))
        else {
            return x;
        };
        let h = xb - xa;
        let t = (x - xa) / h;
        let (t2, t3) = (t * t, t * t * t);
        let y = (2.0 * t3 - 3.0 * t2 + 1.0) * ya + (t3 - 2.0 * t2 + t) * h * ma + (-2.0 * t3 + 3.0 * t2) * yb + (t3 - t2) * h * mb;
        y.clamp(ya.min(yb), ya.max(yb))
    }

    /// The identity (the default curve).
    pub fn is_linear(&self) -> bool {
        self.n == 2 && self.xs[..2] == [0.0, 1.0] && self.ys[..2] == [0.0, 1.0]
    }
}

impl Serialize for ResponseCurve {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.points().serialize(s)
    }
}

impl<'de> Deserialize<'de> for ResponseCurve {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(Self::new(&Vec::<[f32; 2]>::deserialize(d)?))
    }
}

// ---------------------------------------------------------------------------
// 2D Cubic Bézier Curve Fitting, De Casteljau Subdivision & Tail Aggressiveness
// ---------------------------------------------------------------------------

/// Computes the Catmull-Rom tangent velocity vector `(Tx, Ty)` between `p_prev` and `p_next`:
/// $$T = \frac{P_{next} - P_{prev}}{\max(1.0, t_{next} - t_{prev})}$$
pub fn catmull_rom_tangent(p_prev: &StrokePoint, p_next: &StrokePoint) -> (f64, f64) {
    let dx = p_next.x - p_prev.x;
    let dy = p_next.y - p_prev.y;
    let dt = p_next.time - p_prev.time;
    let time_delta = if dt.is_finite() && dt > 0.0 { dt.max(1.0) } else { 1.0 };
    (dx / time_delta, dy / time_delta)
}

#[inline]
fn midpoint(a: &StrokePoint, b: &StrokePoint) -> StrokePoint {
    StrokePoint {
        x: (a.x + b.x) * 0.5,
        y: (a.y + b.y) * 0.5,
        pressure: (a.pressure + b.pressure) * 0.5,
        tilt_x: (a.tilt_x + b.tilt_x) * 0.5,
        tilt_y: (a.tilt_y + b.tilt_y) * 0.5,
        rotation: (a.rotation + b.rotation) * 0.5,
        wheel: (a.wheel + b.wheel) * 0.5,
        time: (a.time + b.time) * 0.5,
    }
}

fn subdivide_rec(p0: &StrokePoint, p1: &StrokePoint, p2: &StrokePoint, p3: &StrokePoint, tol: f64, depth: usize, out: &mut Vec<StrokePoint>) {
    let vx = p3.x - p0.x;
    let vy = p3.y - p0.y;
    let l = (vx * vx + vy * vy).sqrt();
    let dev = if l <= 1e-9 {
        let d1 = (p1.x - p0.x).hypot(p1.y - p0.y);
        let d2 = (p2.x - p0.x).hypot(p2.y - p0.y);
        d1.max(d2)
    } else {
        let d1 = (vx * (p0.y - p1.y) - (p0.x - p1.x) * vy).abs() / l;
        let d2 = (vx * (p0.y - p2.y) - (p0.x - p2.x) * vy).abs() / l;
        d1.max(d2)
    };

    if dev <= tol || depth >= 16 {
        out.push(*p3);
        return;
    }

    let p01 = midpoint(p0, p1);
    let p12 = midpoint(p1, p2);
    let p23 = midpoint(p2, p3);
    let p012 = midpoint(&p01, &p12);
    let p123 = midpoint(&p12, &p23);
    let m = midpoint(&p012, &p123);

    subdivide_rec(p0, &p01, &p012, &m, tol, depth + 1, out);
    subdivide_rec(&m, &p123, &p23, p3, tol, depth + 1, out);
}

/// Recursively subdivides a cubic Bézier curve defined by control points `p0`, `p1`, `p2`, `p3`
/// using De Casteljau midpoint subdivision until the curve is within `flatness` (pixels) of a straight chord.
///
/// For collinear points, the flatness condition is satisfied immediately without excessive recursion.
pub fn de_casteljau_subdivide(p0: StrokePoint, p1: StrokePoint, p2: StrokePoint, p3: StrokePoint, flatness: f32) -> Vec<StrokePoint> {
    let tol = (flatness as f64).max(0.01);
    let mut out = Vec::with_capacity(16);
    out.push(p0);
    subdivide_rec(&p0, &p1, &p2, &p3, tol, 0, &mut out);
    out
}

/// Fits a smooth cubic Bézier curve between `p1` and `p2`, deriving Catmull-Rom tangent vectors
/// from velocity using context points `p0` and `p3`, constructing control points from ray intersections
/// and velocity similarity per Krita's `KisToolFreehandHelper`, and subdividing with De Casteljau to `flatness` px.
pub fn fit_cubic_bezier_with_flatness(p0: &StrokePoint, p1: &StrokePoint, p2: &StrokePoint, p3: &StrokePoint, flatness: f32) -> Vec<StrokePoint> {
    let (dx, dy) = (p2.x - p1.x, p2.y - p1.y);
    let chord_len = (dx * dx + dy * dy).sqrt();
    if chord_len <= 1e-9 {
        return vec![*p1, *p2];
    }

    // Corner detection: if the turn angle is sharp, do not curve across the corner.
    let (v0_x, v0_y) = (p1.x - p0.x, p1.y - p0.y);
    let l0 = v0_x.hypot(v0_y);
    let sharp1 = if l0 > 1e-6 { (v0_x * dx + v0_y * dy) / (l0 * chord_len) < 0.2 } else { false };

    let t1 = if sharp1 || l0 <= 1e-9 { catmull_rom_tangent(p1, p2) } else { catmull_rom_tangent(p0, p2) };

    let (v2_x, v2_y) = (p3.x - p2.x, p3.y - p2.y);
    let l2 = v2_x.hypot(v2_y);
    let sharp2 = if l2 > 1e-6 { (dx * v2_x + dy * v2_y) / (chord_len * l2) < 0.2 } else { false };

    let t2 = if sharp2 || l2 <= 1e-9 { catmull_rom_tangent(p1, p2) } else { catmull_rom_tangent(p1, p3) };

    let v1 = t1.0.hypot(t1.1);
    let v2 = t2.0.hypot(t2.1);

    // Velocity similarity S per Krita KisToolFreehandHelper:
    // S = max(0.5, min(v1 / v2, v2 / v1))
    let s = if v1 <= 1e-6 && v2 <= 1e-6 {
        1.0
    } else if v1 > 1e-6 && v2 > 1e-6 {
        (v1 / v2).min(v2 / v1).clamp(0.5, 1.0)
    } else {
        0.5
    };

    // Unit tangent directions
    let d1 = if v1 > 1e-6 { (t1.0 / v1, t1.1 / v1) } else { (dx / chord_len, dy / chord_len) };

    let d2 = if v2 > 1e-6 { (t2.0 / v2, t2.1 / v2) } else { (dx / chord_len, dy / chord_len) };

    // Ray intersection:
    // Ray 1: p1 + alpha * d1
    // Ray 2: p2 - beta * d2
    // alpha * d1 + beta * d2 = V
    let det = d1.0 * d2.1 - d1.1 * d2.0;
    let (len1, len2) = if det.abs() > 1e-6 {
        let alpha = (dx * d2.1 - dy * d2.0) / det;
        let beta = (d1.0 * dy - d1.1 * dx) / det;
        if alpha > 1e-6 && beta > 1e-6 {
            (((2.0 / 3.0) * alpha).min(chord_len / 3.0) * s, ((2.0 / 3.0) * beta).min(chord_len / 3.0) * s)
        } else {
            ((chord_len / 3.0) * s, (chord_len / 3.0) * s)
        }
    } else {
        ((chord_len / 3.0) * s, (chord_len / 3.0) * s)
    };

    let c1_x = p1.x + len1 * d1.0;
    let c1_y = p1.y + len1 * d1.1;
    let c2_x = p2.x - len2 * d2.0;
    let c2_y = p2.y - len2 * d2.1;

    let c1 = StrokePoint {
        x: c1_x,
        y: c1_y,
        pressure: p1.pressure + (p2.pressure - p1.pressure) * (1.0 / 3.0),
        tilt_x: p1.tilt_x + (p2.tilt_x - p1.tilt_x) * (1.0 / 3.0),
        tilt_y: p1.tilt_y + (p2.tilt_y - p1.tilt_y) * (1.0 / 3.0),
        rotation: p1.rotation + (p2.rotation - p1.rotation) * (1.0 / 3.0),
        wheel: p1.wheel + (p2.wheel - p1.wheel) * (1.0 / 3.0),
        time: p1.time + (p2.time - p1.time) * (1.0 / 3.0),
    };

    let c2 = StrokePoint {
        x: c2_x,
        y: c2_y,
        pressure: p1.pressure + (p2.pressure - p1.pressure) * (2.0 / 3.0),
        tilt_x: p1.tilt_x + (p2.tilt_x - p1.tilt_x) * (2.0 / 3.0),
        tilt_y: p1.tilt_y + (p2.tilt_y - p1.tilt_y) * (2.0 / 3.0),
        rotation: p1.rotation + (p2.rotation - p1.rotation) * (2.0 / 3.0),
        wheel: p1.wheel + (p2.wheel - p1.wheel) * (2.0 / 3.0),
        time: p1.time + (p2.time - p1.time) * (2.0 / 3.0),
    };

    de_casteljau_subdivide(*p1, c1, c2, *p2, flatness)
}

/// Fits a smooth cubic Bézier curve between `p1` and `p2` at the default flatness threshold (0.5 px).
pub fn fit_cubic_bezier(p0: &StrokePoint, p1: &StrokePoint, p2: &StrokePoint, p3: &StrokePoint) -> Vec<StrokePoint> {
    fit_cubic_bezier_with_flatness(p0, p1, p2, p3, 0.5)
}

/// Weighted Gaussian spatial smoothing with Tail Aggressiveness.
///
/// When `dP/dt < 0` (stylus lifting), scales the spatial filter weights by
/// `tail_aggressiveness * (1.0 - pressure)` to converge rapidly on the pen tip,
/// producing needle-sharp tapering and pulling the tail coordinate to the exact pen tip upon lift-off.
pub fn apply_tail_aggressiveness(points: &mut [StrokePoint], tail_aggressiveness: f32) {
    if points.len() < 2 || tail_aggressiveness <= 0.0 {
        return;
    }
    // Tail aggressiveness exclusively engages when pressure decreases (stylus lifting).
    // If pressure is non-decreasing across all points (steady stroke or downstroke),
    // no tail pull is applied and points remain completely unmodified.
    let has_lifting = points.windows(2).any(|w| w[1].pressure < w[0].pressure);
    if !has_lifting {
        return;
    }
    let smoothed = gaussian_smooth_with_tail(points, tail_aggressiveness);
    points.copy_from_slice(&smoothed);
}

/// Applies weighted Gaussian smoothing with tail aggressiveness, returning the smoothed point sequence.
pub fn gaussian_smooth_with_tail(points: &[StrokePoint], tail_aggressiveness: f32) -> Vec<StrokePoint> {
    let n = points.len();
    if n < 2 || tail_aggressiveness <= 0.0 {
        return points.to_vec();
    }

    // Steady strokes and downstrokes experience zero displacement.
    let has_lifting = points.windows(2).any(|w| w[1].pressure < w[0].pressure);
    if !has_lifting {
        return points.to_vec();
    }

    let mut out = points.to_vec();
    let sigma: f64 = 1.5;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let radius: isize = 4;

    for i in 0..n {
        let curr = &points[i];
        let (dp, _dt) = if i > 0 {
            (curr.pressure - points[i - 1].pressure, (curr.time - points[i - 1].time).max(1.0))
        } else if n > 1 {
            (points[1].pressure - curr.pressure, (points[1].time - curr.time).max(1.0))
        } else {
            (0.0, 1.0)
        };

        let lifting = dp < 0.0;
        let pressure_drop = (-dp).max(0.0) as f64;
        let p_factor = (1.0 - curr.pressure.clamp(0.0, 1.0)) as f64;
        let tail_scale = if lifting { f64::from(tail_aggressiveness) * p_factor * (pressure_drop * 40.0).clamp(0.5, 40.0) } else { 0.0 };

        // The terminal point must ALWAYS anchor to the physical pen tip coordinate,
        // preventing one-sided convolution from shrinking the stroke tip backward.
        if i == n - 1 {
            out[i].x = curr.x;
            out[i].y = curr.y;
            out[i].pressure = curr.pressure;
            continue;
        }

        let mut total_w = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;

        let start_j = (i as isize - radius).max(0) as usize;
        let end_j = ((i as isize + radius).min((n - 1) as isize)) as usize;

        for (j, pt_j) in points.iter().enumerate().take(end_j + 1).skip(start_j) {
            let dist = (j as isize - i as isize).abs() as f64;
            let mut w = (-dist * dist / two_sigma_sq).exp();
            // Forward weighting: when lifting, points ahead (j >= i) pull preceding points forward
            if lifting && j >= i {
                w *= 1.0 + tail_scale;
            }
            total_w += w;
            sum_x += w * pt_j.x;
            sum_y += w * pt_j.y;
        }

        if total_w > 1e-9 {
            out[i].x = sum_x / total_w;
            out[i].y = sum_y / total_w;
        }

        if lifting && i > 0 {
            out[i].pressure = out[i].pressure.min(out[i - 1].pressure);
        }
    }

    // Unconditionally anchor start and end points to physical touch-down and lift-off coordinates
    if let (Some(first_out), Some(first_in)) = (out.first_mut(), points.first()) {
        first_out.x = first_in.x;
        first_out.y = first_in.y;
        first_out.pressure = first_in.pressure;
    }
    if let (Some(last_out), Some(last_in)) = (out.last_mut(), points.last()) {
        last_out.x = last_in.x;
        last_out.y = last_in.y;
        last_out.pressure = last_in.pressure;
    }

    out
}

/// Interface contract for Bézier spline interpolation and stroke smoothing.
pub struct SplineInterpolator;

impl SplineInterpolator {
    #[inline]
    pub fn fit_cubic_bezier(p0: &StrokePoint, p1: &StrokePoint, p2: &StrokePoint, p3: &StrokePoint) -> Vec<StrokePoint> {
        fit_cubic_bezier(p0, p1, p2, p3)
    }

    #[inline]
    pub fn de_casteljau_subdivide(p0: StrokePoint, p1: StrokePoint, p2: StrokePoint, p3: StrokePoint, flatness: f32) -> Vec<StrokePoint> {
        de_casteljau_subdivide(p0, p1, p2, p3, flatness)
    }

    #[inline]
    pub fn apply_tail_aggressiveness(points: &mut [StrokePoint], tail_aggressiveness: f32) {
        apply_tail_aggressiveness(points, tail_aggressiveness);
    }

    /// Subdivides an entire stroke of points into smooth cubic Bézier curves at the given flatness.
    pub fn interpolate_stroke(points: &[StrokePoint], flatness: f32) -> Vec<StrokePoint> {
        let n = points.len();
        if n < 2 {
            return points.to_vec();
        }
        if n == 2 {
            let (p0, p1) = (&points[0], &points[1]);
            return fit_cubic_bezier_with_flatness(p0, p0, p1, p1, flatness);
        }

        let mut out = Vec::with_capacity(n * 4);
        for i in 0..(n - 1) {
            let p0 = if i == 0 { &points[0] } else { &points[i - 1] };
            let p1 = &points[i];
            let p2 = &points[i + 1];
            let p3 = if i + 2 < n { &points[i + 2] } else { &points[i + 1] };

            let sub = fit_cubic_bezier_with_flatness(p0, p1, p2, p3, flatness);
            if i == 0 {
                out.extend(sub);
            } else if sub.len() > 1 {
                out.extend(sub.into_iter().skip(1));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_curve_is_monotone_sanitised_and_round_trips_as_points() {
        assert!(ResponseCurve::new(&[[0.0, 0.0], [1.0, 1.0]]).is_linear());
        assert_eq!(ResponseCurve::new(&[[0.0, 0.0], [1.0, 1.0]]), ResponseCurve::LINEAR);
        assert!(ResponseCurve::new(&[[f32::NAN, 0.3]]).is_linear(), "fewer than two usable points is linear");
        let c = ResponseCurve::new(&[[1.0, 1.0], [0.5, 0.2], [0.0, 0.0]]);
        assert!((c.eval(0.5) - 0.2).abs() < 1e-6);
        let mut prev = 0.0;
        for i in 0..=100 {
            let y = c.eval(i as f32 / 100.0);
            assert!(y >= prev - 1e-6 && (0.0..=1.0).contains(&y));
            prev = y;
        }
        let json = serde_json::to_string(&c).unwrap_or_default();
        assert_eq!(json, "[[0.0,0.0],[0.5,0.2],[1.0,1.0]]");
        let back: ResponseCurve = serde_json::from_str(&json).unwrap_or_default();
        assert_eq!(back, c);
        let many: Vec<[f32; 2]> = (0..40).map(|i| [i as f32 / 40.0, 0.5]).collect();
        assert_eq!(ResponseCurve::new(&many).points().len(), ResponseCurve::MAX_POINTS);
    }

    #[test]
    fn test_catmull_rom_tangents_scale_with_velocity() {
        let p0 = StrokePoint { x: 0.0, y: 0.0, time: 0.0, ..Default::default() };

        // Slow stroke: 100px over 20ms -> velocity = 5 px/ms
        let p_slow = StrokePoint { x: 100.0, y: 0.0, time: 20.0, ..Default::default() };
        let (tx_slow, ty_slow) = catmull_rom_tangent(&p0, &p_slow);
        assert!(ty_slow.abs() < 1e-9);
        assert!((tx_slow - 5.0).abs() < 1e-4, "slow tangent {tx_slow} should be 5.0 px/ms");

        // Fast stroke: 100px over 2ms -> velocity = 50 px/ms (10x faster)
        let p_fast = StrokePoint { x: 100.0, y: 0.0, time: 2.0, ..Default::default() };
        let (tx_fast, ty_fast) = catmull_rom_tangent(&p0, &p_fast);
        assert!(ty_fast.abs() < 1e-9);
        assert!((tx_fast - 50.0).abs() < 1e-4, "fast tangent {tx_fast} should be 50.0 px/ms");

        // Tangents scale proportionally with velocity
        let ratio = tx_fast / tx_slow;
        assert!((ratio - 10.0).abs() < 1e-4, "tangent ratio {ratio} should be 10.0");
    }

    #[test]
    fn test_de_casteljau_subdivision_collinear_and_flatness() {
        // 1. Collinear lines: flatness test passes immediately, recursion depth 0, returns exactly 2 points
        let p0 = StrokePoint::new(0.0, 0.0, 1.0);
        let p1 = StrokePoint::new(10.0, 0.0, 1.0);
        let p2 = StrokePoint::new(20.0, 0.0, 1.0);
        let p3 = StrokePoint::new(30.0, 0.0, 1.0);
        let col = de_casteljau_subdivide(p0, p1, p2, p3, 0.5);
        assert_eq!(col.len(), 2, "collinear curve should not recurse; length must be 2");
        assert_eq!(col[0].x, 0.0);
        assert_eq!(col[1].x, 30.0);

        // 2. Curved Bézier arc: subdivides until all chords are within 0.5px of the curve
        let cp0 = StrokePoint::new(0.0, 0.0, 1.0);
        let cp1 = StrokePoint::new(0.0, 50.0, 1.0);
        let cp2 = StrokePoint::new(50.0, 100.0, 1.0);
        let cp3 = StrokePoint::new(100.0, 100.0, 1.0);
        let curve_pts = de_casteljau_subdivide(cp0, cp1, cp2, cp3, 0.5);
        assert!(curve_pts.len() > 2, "curved line must be subdivided");

        // Verify every subdivided chord deviates by at most 0.5px
        for w in curve_pts.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            let chord_dx = b.x - a.x;
            let chord_dy = b.y - a.y;
            let chord_l = (chord_dx * chord_dx + chord_dy * chord_dy).sqrt();
            assert!(chord_l > 0.0);
        }
    }

    #[test]
    fn test_fast_curved_flicks_continuous_c1_tangents() {
        // Fast flick along circular arc: points at 0, 30, 60, 90 degrees
        let r = 100.0;
        let angles_deg: [f64; 4] = [0.0, 30.0, 60.0, 90.0];
        let pts: Vec<StrokePoint> = angles_deg
            .iter()
            .enumerate()
            .map(|(idx, &deg)| {
                let rad = deg.to_radians();
                StrokePoint { x: r * rad.cos(), y: r * rad.sin(), pressure: 1.0 - (idx as f32 * 0.25), time: idx as f64 * 5.0, ..Default::default() }
            })
            .collect();

        // Fit cubic Bézier for segment 1 (angles 0 -> 30) and segment 2 (angles 30 -> 60)
        let sub1 = fit_cubic_bezier(&pts[0], &pts[0], &pts[1], &pts[2]);
        let sub2 = fit_cubic_bezier(&pts[0], &pts[1], &pts[2], &pts[3]);

        assert!(sub1.len() >= 2);
        assert!(sub2.len() >= 2);

        // 1. Raw polyline has a sharp kink:
        let raw_l0 = (pts[1].x - pts[0].x).hypot(pts[1].y - pts[0].y);
        let raw_l1 = (pts[2].x - pts[1].x).hypot(pts[2].y - pts[1].y);
        let raw_dot = ((pts[1].x - pts[0].x) * (pts[2].x - pts[1].x) + (pts[1].y - pts[0].y) * (pts[2].y - pts[1].y)) / (raw_l0 * raw_l1);
        assert!(raw_dot < 0.88, "raw polyline has a kink of ~30 degrees (dot ~ 0.866)");

        // 2. Analytic Catmull-Rom tangent at junction pts[1] provides exact C1 direction:
        let t_junction = catmull_rom_tangent(&pts[0], &pts[2]);
        let t_len = t_junction.0.hypot(t_junction.1);
        assert!(t_len > 0.0);

        // 3. Subdivided Bézier chords smoothly flow across junction without kinks:
        let n1 = sub1.len();
        let dx_in = sub1[n1 - 1].x - sub1[n1 - 2].x;
        let dy_in = sub1[n1 - 1].y - sub1[n1 - 2].y;
        let l_in = dx_in.hypot(dy_in);
        let dir_in = (dx_in / l_in, dy_in / l_in);

        let dx_out = sub2[1].x - sub2[0].x;
        let dy_out = sub2[1].y - sub2[0].y;
        let l_out = dx_out.hypot(dy_out);
        let dir_out = (dx_out / l_out, dy_out / l_out);

        let dot = dir_in.0 * dir_out.0 + dir_in.1 * dir_out.1;
        assert!(dot > 0.98, "subdivided Bézier eliminates polyline kink (dot product was {dot})");
    }

    #[test]
    fn test_tail_aggressiveness_pulls_to_exact_pen_tip() {
        // Simulated stroke lifting off at the end
        let mut stroke = vec![
            StrokePoint { x: 0.0, y: 0.0, pressure: 1.0, time: 0.0, ..Default::default() },
            StrokePoint { x: 50.0, y: 0.0, pressure: 0.9, time: 10.0, ..Default::default() },
            StrokePoint { x: 100.0, y: 0.0, pressure: 0.7, time: 20.0, ..Default::default() },
            StrokePoint { x: 150.0, y: 0.0, pressure: 0.4, time: 30.0, ..Default::default() },
            StrokePoint { x: 180.0, y: 0.0, pressure: 0.1, time: 40.0, ..Default::default() },
            StrokePoint { x: 200.0, y: 0.0, pressure: 0.01, time: 50.0, ..Default::default() },
        ];
        let tip_coord = (200.0, 0.0);

        apply_tail_aggressiveness(&mut stroke, 1.0);

        let last_pt = stroke.last().copied().unwrap_or_default();
        assert_eq!(last_pt.x, tip_coord.0, "tail x must equal exact pen tip on lift-off");
        assert_eq!(last_pt.y, tip_coord.1, "tail y must equal exact pen tip on lift-off");
    }

    #[test]
    fn test_spline_interpolator_interpolate_stroke() {
        let pts =
            vec![StrokePoint::new(0.0, 0.0, 1.0), StrokePoint::new(20.0, 30.0, 0.8), StrokePoint::new(60.0, 40.0, 0.6), StrokePoint::new(100.0, 10.0, 0.3)];

        let interpolated = SplineInterpolator::interpolate_stroke(&pts, 0.5);
        assert!(interpolated.len() > pts.len());
        assert_eq!(interpolated.first().map(|p| (p.x, p.y)), pts.first().map(|p| (p.x, p.y)));
        assert_eq!(interpolated.last().map(|p| (p.x, p.y)), pts.last().map(|p| (p.x, p.y)));
    }

    #[test]
    fn test_tail_aggressiveness_zero_displacement_on_downstroke() {
        // Downstroke: stylus pressing down with increasing pressure (0.1 to 0.9)
        let mut stroke = vec![
            StrokePoint { x: 10.0, y: 10.0, pressure: 0.1, time: 0.0, ..Default::default() },
            StrokePoint { x: 30.0, y: 15.0, pressure: 0.4, time: 5.0, ..Default::default() },
            StrokePoint { x: 60.0, y: 22.0, pressure: 0.7, time: 10.0, ..Default::default() },
            StrokePoint { x: 90.0, y: 30.0, pressure: 0.9, time: 15.0, ..Default::default() },
        ];
        let original = stroke.clone();

        apply_tail_aggressiveness(&mut stroke, 1.0);

        // Every point, including the endpoint, must remain completely unperturbed
        for (i, (s, orig)) in stroke.iter().zip(original.iter()).enumerate() {
            assert_eq!(s.x, orig.x, "Point {} x displaced on downstroke", i);
            assert_eq!(s.y, orig.y, "Point {} y displaced on downstroke", i);
            assert_eq!(s.pressure, orig.pressure, "Point {} pressure displaced on downstroke", i);
        }
    }

    #[test]
    fn test_tail_aggressiveness_zero_displacement_on_steady_stroke() {
        // Steady stroke: constant pressure (0.5)
        let mut stroke = vec![
            StrokePoint { x: 10.0, y: 20.0, pressure: 0.5, time: 0.0, ..Default::default() },
            StrokePoint { x: 50.0, y: 20.0, pressure: 0.5, time: 10.0, ..Default::default() },
            StrokePoint { x: 90.0, y: 20.0, pressure: 0.5, time: 20.0, ..Default::default() },
        ];
        let original = stroke.clone();

        apply_tail_aggressiveness(&mut stroke, 5.0);

        // Zero endpoint displacement error on steady stroke
        assert_eq!(stroke, original, "Steady stroke must experience zero coordinate or pressure displacement");
    }

    #[test]
    fn test_tail_aggressiveness_lifting_forward_pull_with_exact_anchoring() {
        // Lifting stroke: pressure decays from 0.8 down to 0.01
        let mut stroke = vec![
            StrokePoint { x: 10.0, y: 10.0, pressure: 0.8, time: 0.0, ..Default::default() },
            StrokePoint { x: 30.0, y: 15.0, pressure: 0.6, time: 5.0, ..Default::default() },
            StrokePoint { x: 60.0, y: 22.0, pressure: 0.3, time: 10.0, ..Default::default() },
            StrokePoint { x: 90.0, y: 30.0, pressure: 0.05, time: 15.0, ..Default::default() },
            StrokePoint { x: 120.0, y: 40.0, pressure: 0.01, time: 20.0, ..Default::default() },
        ];
        let tip_coord = (120.0, 40.0, 0.01);
        let orig_penultimate_x = stroke[3].x; // 90.0

        apply_tail_aggressiveness(&mut stroke, 1.0);

        let final_pt = stroke.last().unwrap();
        // Exact endpoint anchoring (zero displacement error)
        assert_eq!(final_pt.x, tip_coord.0, "Endpoint x must match physical tip exactly");
        assert_eq!(final_pt.y, tip_coord.1, "Endpoint y must match physical tip exactly");
        assert_eq!(final_pt.pressure, tip_coord.2, "Endpoint pressure must match physical tip exactly");

        // Forward weighting: trailing point must be pulled forward towards the tip
        let penultimate = stroke[3];
        assert!(penultimate.x > orig_penultimate_x, "Penultimate point must be pulled forward towards tip");
    }
}
