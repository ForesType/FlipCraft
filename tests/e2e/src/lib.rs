//! PhotoCraft End-to-End Test Infrastructure & Authoritative Oracles
//!
//! Provides opaque-box test harnesses, synthetic digitizer generators, mathematical
//! reference oracles, and invariant checkers for the PhotoCraft brush engine and
//! Windows tablet driver overhaul.

pub use photocraft_geom::Point;
pub use photocraft_paint::brush::BrushSettings;
pub use photocraft_paint::dynamics::{DabBuilder, DabGenerator, PathWalker, StepInput};
pub use photocraft_paint::{Dab, StrokePoint};
pub use photocraft_tablet::ink;
pub use photocraft_tablet::ink::{DeviceBounds, RawPointerEvent, State as InkState};
pub use photocraft_tablet::{Sample, Update};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ============================================================================
// 1. Synthetic Stroke Generator & Builders
// ============================================================================

/// Builder for synthetic stylus stroke trajectories.
#[derive(Clone, Debug)]
pub struct StrokeTrajectoryBuilder {
    points: Vec<StrokePoint>,
    current_time: f64,
}

impl Default for StrokeTrajectoryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl StrokeTrajectoryBuilder {
    pub fn new() -> Self {
        Self { points: Vec::new(), current_time: 0.0 }
    }

    /// Appends a point with explicit coordinates and pressure.
    pub fn add_point(mut self, x: f64, y: f64, pressure: f32) -> Self {
        self.points.push(StrokePoint { x, y, pressure, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: self.current_time });
        self.current_time += 16.666; // default 60Hz tick
        self
    }

    /// Appends a point with full telemetry (pressure, tilt, rotation, time).
    #[allow(clippy::too_many_arguments)]
    pub fn add_full(mut self, x: f64, y: f64, pressure: f32, tilt_x: f32, tilt_y: f32, rotation: f32, time: f64) -> Self {
        self.points.push(StrokePoint { x, y, pressure, tilt_x, tilt_y, rotation, wheel: 0.0, time });
        self.current_time = time;
        self
    }

    /// Generates a fast tapering flick: downstroke with heavy pressure tapering
    /// rapidly to zero pressure over a short distance with accelerating velocity.
    pub fn create_calligraphy_flick(start_x: f64, start_y: f64, length: f64) -> Vec<StrokePoint> {
        let mut pts = Vec::new();
        let steps = 40;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            // Nonlinear spacing: accelerates at the tail
            let dist = length * (t * t);
            // Pressure decays from 0.95 to 0.0
            let pressure = if t < 0.3 { 0.95 } else { (0.95 * (1.0 - (t - 0.3) / 0.7)).max(0.0) as f32 };
            pts.push(StrokePoint {
                x: start_x + dist,
                y: start_y + dist * 0.25,
                pressure,
                tilt_x: 0.0,
                tilt_y: 0.0,
                rotation: 0.0,
                wheel: 0.0,
                time: (i as f64) * 4.0, // 250Hz sample rate
            });
        }
        pts
    }

    /// Generates rapid hatch shading: multiple rapid bidirectional strokes.
    pub fn create_hatch_shading(origin_x: f64, origin_y: f64, stroke_length: f64, stroke_count: usize) -> Vec<StrokePoint> {
        let mut pts = Vec::new();
        let mut t_ms = 0.0;
        for s in 0..stroke_count {
            let y_off = (s as f64) * 8.0;
            let forward = s % 2 == 0;
            let n = 15;
            for i in 0..=n {
                let frac = i as f64 / n as f64;
                let x = if forward { origin_x + frac * stroke_length } else { origin_x + (1.0 - frac) * stroke_length };
                let p = (0.3 + 0.5 * (1.0 - (2.0 * frac - 1.0).powi(2))) as f32;
                pts.push(StrokePoint { x, y: origin_y + y_off, pressure: p, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: t_ms });
                t_ms += 2.0; // 500Hz
            }
        }
        pts
    }

    /// Generates a smooth curved arc (e.g. 90-degree bend).
    pub fn create_curved_arc(center_x: f64, center_y: f64, radius: f64, steps: usize) -> Vec<StrokePoint> {
        let mut pts = Vec::new();
        for i in 0..=steps {
            let theta = (i as f64 / steps as f64) * std::f64::consts::FRAC_PI_2;
            let x = center_x + radius * theta.cos();
            let y = center_y + radius * theta.sin();
            let pressure = 0.5 + 0.3 * (theta.sin() as f32);
            pts.push(StrokePoint { x, y, pressure, tilt_x: 0.0, tilt_y: 0.0, rotation: (theta.to_degrees() as f32), wheel: 0.0, time: (i as f64) * 8.0 });
        }
        pts
    }

    pub fn build(self) -> Vec<StrokePoint> {
        self.points
    }
}

// ============================================================================
// 2. Adaptive Distance Stepping Reference Oracle (Krita KisDistanceInformation Parity)
// ============================================================================

/// Result of an adaptive distance step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AdaptiveStepResult {
    /// Immediate dab triggered due to spacing shrinkage ($t = 0.0$).
    ImmediateShrinkageDab,
    /// Dab triggered along segment at parametric position $t \in [0.0, 1.0]$.
    SegmentDab(f64),
    /// No dab emitted; distance accumulated into internal state.
    Accumulated(f64),
}

/// Authoritative clean-room reference implementation of Krita's `KisDistanceInformation`.
///
/// Invariant:
/// - When pressure drops and target spacing shrinks below currently accumulated distance
///   ($\Delta_{\text{next}} \le 0.0$), a dab is immediately triggered at $t = 0.0$ and the
///   accumulator is reset, guaranteeing zero dab starvation during rapid flick tapering.
#[derive(Clone, Debug, PartialEq)]
pub struct KisDistanceOracle {
    pub accum_distance: f64,
    pub last_spacing: f64,
}

impl Default for KisDistanceOracle {
    fn default() -> Self {
        Self::new()
    }
}

impl KisDistanceOracle {
    pub fn new() -> Self {
        Self { accum_distance: 0.0, last_spacing: 0.0 }
    }

    /// Reset distance accumulator (Krita `resetAccumulators()`).
    pub fn reset_accumulators(&mut self) {
        self.accum_distance = 0.0;
    }

    /// Steps distance along a segment of length `segment_len` with target `current_spacing`.
    ///
    /// Implements Krita's exact logic from `kis_distance_information.cpp:471-487`:
    /// ```cpp
    /// qreal nextPointDistance = spacing - accumDistance;
    /// if (nextPointDistance <= 0.0) {
    ///     resetAccumulators();
    ///     t = 0.0;
    /// } else if (nextPointDistance <= dragVecLength) {
    ///     t = nextPointDistance / dragVecLength;
    ///     resetAccumulators();
    /// } else {
    ///     t = -1;
    ///     accumDistance += dragVecLength;
    /// }
    /// ```
    pub fn step(&mut self, segment_len: f64, current_spacing: f64) -> (Vec<f64>, AdaptiveStepResult) {
        let current_spacing = current_spacing.max(0.1);
        let mut dabs = Vec::new();

        let next_point_distance = current_spacing - self.accum_distance;
        if next_point_distance <= 0.0 {
            // Spacing shrank below accumulated distance -> immediate dab at t=0.0!
            self.reset_accumulators();
            self.last_spacing = current_spacing;
            dabs.push(0.0);
        }

        if segment_len > 1e-9 {
            let mut remaining = segment_len;
            let mut traveled = 0.0;
            while current_spacing - self.accum_distance <= remaining {
                let step_needed = (current_spacing - self.accum_distance).max(0.0);
                traveled += step_needed;
                let t = (traveled / segment_len).clamp(0.0, 1.0);
                dabs.push(t);
                remaining -= step_needed;
                self.reset_accumulators();
                self.last_spacing = current_spacing;
                if step_needed < 1e-9 {
                    break;
                }
            }
            self.accum_distance += remaining;
        }

        let res = if !dabs.is_empty() {
            if dabs.contains(&0.0) && next_point_distance <= 0.0 {
                AdaptiveStepResult::ImmediateShrinkageDab
            } else {
                AdaptiveStepResult::SegmentDab(*dabs.last().unwrap_or(&0.0))
            }
        } else {
            AdaptiveStepResult::Accumulated(self.accum_distance)
        };

        (dabs, res)
    }

    /// Convenience runner for a sequence of (segment_len, current_spacing) steps.
    pub fn run_sequence(&mut self, steps: &[(f64, f64)]) -> Vec<(Vec<f64>, AdaptiveStepResult)> {
        steps.iter().map(|&(len, sp)| self.step(len, sp)).collect()
    }
}

// ============================================================================
// 3. Cubic Bézier & De Casteljau Reference Oracle
// ============================================================================

/// Evaluates cubic Bézier curve at parameter $t \in [0.0, 1.0]$.
pub fn eval_cubic_bezier_pt(p0: (f64, f64), p1: (f64, f64), p2: (f64, f64), p3: (f64, f64), t: f64) -> (f64, f64) {
    let u = 1.0 - t;
    let tt = t * t;
    let uu = u * u;
    let uuu = uu * u;
    let ttt = tt * t;

    let x = uuu * p0.0 + 3.0 * uu * t * p1.0 + 3.0 * u * tt * p2.0 + ttt * p3.0;
    let y = uuu * p0.1 + 3.0 * uu * t * p1.1 + 3.0 * u * tt * p2.1 + ttt * p3.1;
    (x, y)
}

/// Measures the maximum deviation (flatness) of control points from the chord connecting endpoints.
pub fn bezier_flatness(p0: (f64, f64), p1: (f64, f64), p2: (f64, f64), p3: (f64, f64)) -> f64 {
    let (dx, dy) = (p3.0 - p0.0, p3.1 - p0.1);
    let chord_len = (dx * dx + dy * dy).sqrt();
    if chord_len < 1e-9 {
        let d1 = ((p1.0 - p0.0).powi(2) + (p1.1 - p0.1).powi(2)).sqrt();
        let d2 = ((p2.0 - p0.0).powi(2) + (p2.1 - p0.1).powi(2)).sqrt();
        return d1.max(d2);
    }
    // Perpendicular distance from P to line P0->P3
    let dist = |p: (f64, f64)| -> f64 { ((dy * p.0 - dx * p.1 + p3.0 * p0.1 - p3.1 * p0.0).abs()) / chord_len };
    dist(p1).max(dist(p2))
}

/// Recursive De Casteljau subdivision producing polylines guaranteed to meet the flatness threshold.
pub fn de_casteljau_subdivide(
    p0: (f64, f64),
    p1: (f64, f64),
    p2: (f64, f64),
    p3: (f64, f64),
    flatness_threshold: f64,
    depth: usize,
    out: &mut Vec<(f64, f64)>,
) {
    let flatness_threshold = if flatness_threshold <= 0.0 { 0.5 } else { flatness_threshold };
    if depth > 16 || bezier_flatness(p0, p1, p2, p3) <= flatness_threshold {
        out.push(p3);
        return;
    }

    // Midpoint calculations (Krita kis_paintop.cc:103-120)
    let mid = |a: (f64, f64), b: (f64, f64)| ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);

    let l1 = p0;
    let l2 = mid(p0, p1);
    let h = mid(p1, p2);
    let r3 = mid(p2, p3);
    let r4 = p3;

    let l3 = mid(l2, h);
    let r2 = mid(h, r3);
    let l4 = mid(l3, r2);
    let r1 = l4;

    de_casteljau_subdivide(l1, l2, l3, l4, flatness_threshold, depth + 1, out);
    de_casteljau_subdivide(r1, r2, r3, r4, flatness_threshold, depth + 1, out);
}

/// Computes Catmull-Rom velocity tangent for intermediate point $P_i$.
pub fn catmull_rom_tangent(prev: &StrokePoint, _curr: &StrokePoint, next: &StrokePoint) -> (f64, f64) {
    let dt = (next.time - prev.time).abs().max(1.0);
    let dx = (next.x - prev.x) / dt;
    let dy = (next.y - prev.y) / dt;
    (dx, dy)
}

// ============================================================================
// 4. Tail Aggressiveness Smoothing Oracle
// ============================================================================

/// Applies tail aggressiveness smoothing to a sequence of points.
///
/// Invariant:
/// - When $dP/dt < 0$ (pressure falling / stylus lifting), spatial filter weights
///   scale with $dP/dt \times \text{aggressiveness} \times (1.0 - P)$, rapidly pulling the stroke
///   toward the pen tip coordinate and eliminating trailing blunt cutoffs.
pub fn apply_tail_aggressiveness(points: &mut [StrokePoint], tail_aggressiveness: f32) {
    if points.len() < 2 || tail_aggressiveness <= 0.0 {
        return;
    }
    let n = points.len();
    for i in 1..n {
        let prev_p = points[i - 1].pressure;
        let curr_p = points[i].pressure;
        let dp = prev_p - curr_p; // positive when pressure is falling

        if dp > 0.0 && curr_p.is_finite() && prev_p.is_finite() {
            // Tail pull factor: proportional to pressure gradient and inverted pressure
            let pull = (dp * tail_aggressiveness * (1.0 - curr_p)).clamp(0.0, 1.0) as f64;
            // Pull coordinates towards current pen position
            let target_x = points[i].x;
            let target_y = points[i].y;
            points[i - 1].x = points[i - 1].x * (1.0 - pull) + target_x * pull;
            points[i - 1].y = points[i - 1].y * (1.0 - pull) + target_y * pull;
        }
    }
}

// ============================================================================
// 5. Quantized Dab Cache Reference Oracle (`DabCache`)
// ============================================================================

/// Discretized key for dab coverage mask caching.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DabKey {
    pub radius_quant: u16,  // e.g. radius * 4 (0.25px resolution)
    pub hardness_quant: u8, // 0..100
    pub softness_quant: u8, // 0..100
    pub aspect_quant: u8,   // 0..100
    pub angle_quant: u16,   // 0..360
}

impl DabKey {
    pub fn new(radius: f32, hardness: f32, softness: f32, aspect: f32, angle_deg: f32) -> Self {
        let r = radius.max(0.0);
        let radius_quant = if r < 32.0 {
            (r * 4.0).round() as u16 // 0.25px steps for small radii
        } else {
            ((r / 32.0).ln_1p() * 100.0).round() as u16 + 128 // logarithmic/2% quantization
        };
        Self {
            radius_quant,
            hardness_quant: (hardness.clamp(0.0, 1.0) * 100.0).round() as u8,
            softness_quant: (softness.clamp(0.0, 1.0) * 100.0).round() as u8,
            aspect_quant: (aspect.clamp(0.01, 1.0) * 100.0).round() as u8,
            angle_quant: (angle_deg.rem_euclid(360.0)).round() as u16,
        }
    }
}

/// Pre-rendered float coverage mask.
#[derive(Clone, Debug, PartialEq)]
pub struct DabMask {
    pub width: u32,
    pub height: u32,
    pub coverage: Arc<[f32]>,
}

/// Thread-safe bounded LRU DabCache reference oracle.
#[allow(clippy::type_complexity)]
pub struct DabCacheOracle {
    capacity: usize,
    cache: Mutex<HashMap<DabKey, (Arc<[f32]>, u64)>>,
    counter: Mutex<u64>,
}

impl DabCacheOracle {
    pub fn new(capacity: usize) -> Self {
        Self { capacity: capacity.max(1), cache: Mutex::new(HashMap::new()), counter: Mutex::new(0) }
    }

    pub fn len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieves or computes dab coverage mask.
    pub fn get_or_create(&self, key: &DabKey, generator: impl FnOnce() -> Arc<[f32]>) -> (Arc<[f32]>, bool) {
        let mut cache = self.cache.lock().unwrap();
        let mut counter = self.counter.lock().unwrap();
        *counter += 1;
        let now = *counter;

        if let Some((mask, timestamp)) = cache.get_mut(key) {
            *timestamp = now;
            return (Arc::clone(mask), true); // Cache HIT
        }

        // Cache MISS: generate mask
        let mask = generator();

        // Evict LRU if capacity exceeded
        if cache.len() >= self.capacity
            && let Some((&oldest_key, _)) = cache.iter().min_by_key(|(_, (_, ts))| *ts)
        {
            cache.remove(&oldest_key);
        }

        cache.insert(*key, (Arc::clone(&mask), now));
        (mask, false)
    }
}

/// Generates an analytical circular float coverage mask for testing accuracy.
pub fn generate_analytical_mask(radius: f32, hardness: f32) -> Arc<[f32]> {
    let r = radius.max(0.5);
    let size = (r.ceil() as usize * 2 + 3).max(3);
    let center = size as f32 / 2.0;
    let mut data = vec![0.0f32; size * size];

    for y in 0..size {
        let dy = (y as f32 + 0.5) - center;
        for x in 0..size {
            let dx = (x as f32 + 0.5) - center;
            let d = (dx * dx + dy * dy).sqrt();
            let norm_d = d / r;
            if norm_d <= hardness {
                data[y * size + x] = 1.0;
            } else if norm_d <= 1.0 {
                let t = (1.0 - norm_d) / (1.0 - hardness).max(1e-4);
                data[y * size + x] = (t * t * (3.0 - 2.0 * t)).clamp(0.0, 1.0);
            }
        }
    }
    data.into()
}

// ============================================================================
// 6. SIMD / Tile Rasterization Reference Oracle (COV_TILE = 64)
// ============================================================================

pub const COV_TILE_SIZE: usize = 64;

/// Represents a 64x64 float coverage tile.
#[derive(Clone, Debug)]
pub struct CoverageTile {
    pub tx: i32,
    pub ty: i32,
    pub data: [f32; COV_TILE_SIZE * COV_TILE_SIZE],
}

impl CoverageTile {
    pub fn new(tx: i32, ty: i32) -> Self {
        Self { tx, ty, data: [0.0; COV_TILE_SIZE * COV_TILE_SIZE] }
    }

    /// Blits coverage with saturation clamp at 1.0.
    pub fn blit_pixel(&mut self, lx: usize, ly: usize, alpha: f32) {
        if lx < COV_TILE_SIZE && ly < COV_TILE_SIZE {
            let idx = ly * COV_TILE_SIZE + lx;
            let current = self.data[idx];
            self.data[idx] = (current + alpha * (1.0 - current)).clamp(0.0, 1.0);
        }
    }
}

/// Blits a dab into a tiled canvas, returning touched tile coordinates.
pub fn blit_dab_into_tiles(tiles: &mut HashMap<(i32, i32), CoverageTile>, center_x: f64, center_y: f64, radius: f64, alpha: f32) -> Vec<(i32, i32)> {
    if alpha <= 0.0 || radius <= 0.0 || !center_x.is_finite() || !center_y.is_finite() {
        return Vec::new();
    }

    let min_x = (center_x - radius).floor() as i32;
    let max_x = (center_x + radius).ceil() as i32;
    let min_y = (center_y - radius).floor() as i32;
    let max_y = (center_y + radius).ceil() as i32;

    let t_min_x = min_x.div_euclid(COV_TILE_SIZE as i32);
    let t_max_x = max_x.div_euclid(COV_TILE_SIZE as i32);
    let t_min_y = min_y.div_euclid(COV_TILE_SIZE as i32);
    let t_max_y = max_y.div_euclid(COV_TILE_SIZE as i32);

    let mut touched = Vec::new();

    for ty in t_min_y..=t_max_y {
        for tx in t_min_x..=t_max_x {
            touched.push((tx, ty));
            let tile = tiles.entry((tx, ty)).or_insert_with(|| CoverageTile::new(tx, ty));

            let tile_x0 = tx * (COV_TILE_SIZE as i32);
            let tile_y0 = ty * (COV_TILE_SIZE as i32);

            for ly in 0..COV_TILE_SIZE {
                let py = (tile_y0 + ly as i32) as f64 + 0.5;
                let dy = py - center_y;
                for lx in 0..COV_TILE_SIZE {
                    let px = (tile_x0 + lx as i32) as f64 + 0.5;
                    let dx = px - center_x;
                    let dist2 = dx * dx + dy * dy;
                    if dist2 <= radius * radius {
                        let d = dist2.sqrt();
                        let falloff = ((radius - d) / radius).clamp(0.0, 1.0) as f32;
                        tile.blit_pixel(lx, ly, falloff * alpha);
                    }
                }
            }
        }
    }
    touched
}

// ============================================================================
// 7. Windows Ink & Rate Compression Reference Oracle
// ============================================================================

/// Simulates native Win32 `WM_POINTER` events and tests event rate compression.
pub struct WindowsInkOracle {
    pub state: InkState,
}

impl Default for WindowsInkOracle {
    fn default() -> Self {
        Self::new()
    }
}

impl WindowsInkOracle {
    pub fn new() -> Self {
        Self { state: InkState::new() }
    }

    pub fn with_bounds(bounds: DeviceBounds) -> Self {
        Self { state: InkState::new().with_bounds(bounds) }
    }

    /// Feeds a synthetic pointer packet and receives resulting sample update.
    pub fn feed(&mut self, event: &RawPointerEvent) -> Update {
        self.state.handle(event)
    }

    /// Evaluates rate compression on a high-frequency packet stream.
    ///
    /// Returns the number of events that passed through vs compressed.
    pub fn filter_stream(&mut self, stream: &[RawPointerEvent]) -> (Vec<Sample>, usize) {
        let mut samples = Vec::new();
        let mut dropped = 0;

        for event in stream {
            match self.feed(event) {
                Update::Set(Some(sample)) => samples.push(sample),
                Update::Set(None) => {}
                Update::Keep => dropped += 1,
            }
        }
        (samples, dropped)
    }
}

// ============================================================================
// 8. Invariant & Assertion Helpers
// ============================================================================

/// Asserts that a sequence of numbers is strictly monotonically non-decreasing.
pub fn assert_monotonically_increasing(values: &[f32], tolerance: f32) {
    for i in 1..values.len() {
        assert!(values[i] >= values[i - 1] - tolerance, "Monotonicity violation at index {}: {} < {}", i, values[i], values[i - 1]);
    }
}

/// Asserts that a sequence of numbers is strictly monotonically non-increasing.
pub fn assert_monotonically_decreasing(values: &[f32], tolerance: f32) {
    for i in 1..values.len() {
        assert!(values[i] <= values[i - 1] + tolerance, "Decreasing monotonicity violation at index {}: {} > {}", i, values[i], values[i - 1]);
    }
}
