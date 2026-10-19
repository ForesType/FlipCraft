//! Tier 4: Real-World Workload Scenarios Suite (Artist Stroke Archetypes)
//!
//! Validates realistic artist workflows including calligraphy flicks, rapid hatch shading,
//! large watercolor washes, airbrush tapers, and smooth inking lines (10 tests total).

use photocraft_e2e_tests::*;
use std::collections::HashMap;

// ============================================================================
// 1. Scenario: Calligraphy Flick
// ============================================================================

#[test]
fn test_t4_calligraphy_flick() {
    // Artist performs a swift calligraphy stroke: heavy downstroke tapering to a needle tip
    let mut flick = StrokeTrajectoryBuilder::create_calligraphy_flick(100.0, 100.0, 350.0);
    apply_tail_aggressiveness(&mut flick, 12.0);

    let mut oracle = KisDistanceOracle::new();
    let mut dabs = Vec::new();

    for i in 1..flick.len() {
        let p0 = &flick[i - 1];
        let p1 = &flick[i];
        let seg_len = ((p1.x - p0.x).powi(2) + (p1.y - p0.y).powi(2)).sqrt();
        let target_spacing = (p1.pressure * 25.0).max(1.0) as f64;

        let (step_dabs, _) = oracle.step(seg_len, target_spacing);
        for t in step_dabs {
            let x = p0.x + (p1.x - p0.x) * t;
            let y = p0.y + (p1.y - p0.y) * t;
            let p = p0.pressure + (p1.pressure - p0.pressure) * (t as f32);
            let radius = (p * 20.0).max(0.5);
            dabs.push((x, y, radius));
        }
    }

    assert!(dabs.len() >= 25, "Calligraphy flick must emit abundant continuous dabs (got {})", dabs.len());

    // Verify needle-thin tip: final dab radius must be under 1.0px
    let (_, _, last_radius) = dabs.last().copied().expect("last dab");
    assert!(last_radius <= 1.0, "Calligraphy flick tip must taper to razor needle point (got radius {})", last_radius);

    // Verify absence of dab starvation gaps
    for w in dabs.windows(2) {
        let dist = ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt();
        assert!(dist <= 25.0, "Starvation gap detected in flick: distance between dabs was {} px", dist);
    }
}

// ============================================================================
// 2. Scenario: Rapid Hatch Shading
// ============================================================================

#[test]
fn test_t4_rapid_hatch_shading() {
    // Artist shades an area with 20 rapid back-and-forth strokes in 200ms
    let shading_strokes = StrokeTrajectoryBuilder::create_hatch_shading(50.0, 50.0, 120.0, 20);
    assert_eq!(shading_strokes.len(), 20 * 16);

    let mut oracle = KisDistanceOracle::new();
    let mut tiles = HashMap::new();
    let mut total_dabs = 0;

    for i in 1..shading_strokes.len() {
        let p0 = &shading_strokes[i - 1];
        let p1 = &shading_strokes[i];
        let len = ((p1.x - p0.x).powi(2) + (p1.y - p0.y).powi(2)).sqrt();
        let spacing = (p1.pressure * 10.0).max(1.0) as f64;

        let (dabs, _) = oracle.step(len, spacing);
        for t in dabs {
            total_dabs += 1;
            let x = p0.x + (p1.x - p0.x) * t;
            let y = p0.y + (p1.y - p0.y) * t;
            blit_dab_into_tiles(&mut tiles, x, y, 6.0, p1.pressure);
        }
    }

    assert!(total_dabs >= 100, "Rapid hatch shading must emit dabs across all 20 passes (got {})", total_dabs);
    assert!(!tiles.is_empty());
}

// ============================================================================
// 3. Scenario: Large Watercolor Wash
// ============================================================================

#[test]
fn test_t4_large_watercolor_wash() {
    // Artist paints a broad watercolor wash with a 450px-500px diameter soft brush
    let mut tiles = HashMap::new();
    let cache = DabCacheOracle::new(32);
    let radius = 225.0; // 450px diameter

    let key = DabKey::new(radius, 0.3, 0.8, 1.0, 0.0);
    let (mask, _) = cache.get_or_create(&key, || generate_analytical_mask(radius, 0.3));
    assert!(!mask.is_empty());

    // Sweep across a 1500x1500px canvas area
    for step in 0..15 {
        let center_x = (step as f64) * 80.0 + 250.0;
        let center_y = 300.0 + ((step as f64) * 0.5).sin() * 100.0;
        blit_dab_into_tiles(&mut tiles, center_x, center_y, radius as f64, 0.4);
    }

    assert!(tiles.len() >= 60, "450px wash must touch many tiles across canvas (touched: {})", tiles.len());
    // Verify DabCache memory remains strictly bounded
    assert!(cache.len() <= 32);
}

// ============================================================================
// 4. Scenario: Airbrush Taper
// ============================================================================

#[test]
fn test_t4_airbrush_taper() {
    // Artist holds pen in place for 300ms (buildup), then sweeps outward with decaying pressure
    let mut pts = Vec::new();

    // 1. Stationary buildup: same coordinate, time advancing
    for t in 0..15 {
        pts.push(StrokePoint {
            x: 100.0,
            y: 100.0,
            pressure: 0.8,
            time: (t as f64) * 20.0, // 300ms total
            ..Default::default()
        });
    }

    // 2. Accelerating taper away from center
    for i in 1..=20 {
        let dist = (i as f64) * 10.0;
        let p = (0.8 * (1.0 - i as f32 / 20.0)).max(0.01);
        pts.push(StrokePoint {
            x: 100.0 + dist,
            y: 100.0,
            pressure: p,
            time: 300.0 + (i as f64) * 8.0,
            ..Default::default()
        });
    }

    let mut oracle = KisDistanceOracle::new();
    let mut dabs = 0;

    for i in 1..pts.len() {
        let len = ((pts[i].x - pts[i - 1].x).powi(2) + (pts[i].y - pts[i - 1].y).powi(2)).sqrt();
        let spacing = (pts[i].pressure * 15.0).max(1.0) as f64;
        let (step_dabs, _) = oracle.step(len, spacing);
        dabs += step_dabs.len();
    }

    assert!(dabs > 0, "Airbrush stroke must successfully emit dabs during transition from hold to sweep");
}

// ============================================================================
// 5. Scenario: Smooth Inking Line
// ============================================================================

#[test]
fn test_t4_smooth_inking_line() {
    // Artist draws a slow, deliberate curved inking line (S-curve)
    let p0 = (50.0, 100.0);
    let p1 = (150.0, 30.0);
    let p2 = (250.0, 250.0);
    let p3 = (350.0, 150.0);

    let mut curve = vec![p0];
    de_casteljau_subdivide(p0, p1, p2, p3, 0.5, 0, &mut curve);

    assert!(curve.len() >= 10, "S-curve inking line must be finely subdivided for smoothness");

    // Verify curvature smoothness: angle change between consecutive segments must be continuous
    let mut prev_angle: Option<f64> = None;
    for i in 1..curve.len() {
        let dx = curve[i].0 - curve[i - 1].0;
        let dy = curve[i].1 - curve[i - 1].1;
        let angle = dy.atan2(dx);

        if let Some(pa) = prev_angle {
            let diff = (angle - pa).abs();
            // In a smooth C1 spline with 0.5px flatness, angular change between consecutive sub-segments is small
            assert!(diff < std::f64::consts::FRAC_PI_2, "Inking line must have smooth C1 continuity without abrupt kinks");
        }
        prev_angle = Some(angle);
    }
}

// ============================================================================
// 6. Scenario: Dynamic Calligraphy Ribbon
// ============================================================================

#[test]
fn test_t4_dynamic_calligraphy_ribbon() {
    // Artist uses a chisel tip brush following an S-curve while rotating barrel
    let mut oracle = KisDistanceOracle::new();
    let arc = StrokeTrajectoryBuilder::create_curved_arc(200.0, 200.0, 150.0, 30);
    let mut ribbon_dabs = 0;

    for i in 1..arc.len() {
        let p0 = &arc[i - 1];
        let p1 = &arc[i];
        let len = ((p1.x - p0.x).powi(2) + (p1.y - p0.y).powi(2)).sqrt();

        // Chisel aspect ratio (0.2) modulated by rotation
        let angle_rad = (p1.rotation as f64).to_radians();
        let effective_aspect = (0.2 + 0.8 * angle_rad.sin().abs()).clamp(0.2, 1.0);
        let spacing = 10.0 * effective_aspect;

        let (dabs, _) = oracle.step(len, spacing);
        ribbon_dabs += dabs.len();
    }

    assert!(ribbon_dabs >= 15, "Dynamic calligraphy ribbon must generate smooth dab stream");
}

// ============================================================================
// 7. Scenario: Expressive Manga Speedlines
// ============================================================================

#[test]
fn test_t4_manga_speedlines() {
    // Artist draws 10 radial tapering speedlines radiating outward from (200, 200)
    let center = (200.0, 200.0);
    let speedline_count = 10;
    let length = 180.0;

    for s in 0..speedline_count {
        let theta = (s as f64 / speedline_count as f64) * std::f64::consts::TAU;
        let mut speedline = Vec::new();

        for i in 0..=15 {
            let frac = i as f64 / 15.0;
            let r = frac * length;
            let pressure = (1.0 - frac as f32).max(0.01);
            speedline.push(StrokePoint {
                x: center.0 + r * theta.cos(),
                y: center.1 + r * theta.sin(),
                pressure,
                time: (i as f64) * 4.0,
                ..Default::default()
            });
        }

        apply_tail_aggressiveness(&mut speedline, 15.0);
        assert_eq!(speedline.len(), 16);
        // Tip must reach near max radius
        let last = speedline.last().unwrap();
        let actual_r = ((last.x - center.0).powi(2) + (last.y - center.1).powi(2)).sqrt();
        assert!((actual_r - length).abs() < 5.0, "Speedline must reach its intended outer radius");
    }
}

// ============================================================================
// 8. Scenario: Stippling Pointillism
// ============================================================================

#[test]
fn test_t4_stippling_pointillism() {
    // Artist performs 200 discrete micro-taps with varying pressure
    let bounds = DeviceBounds::LEVELS_8K;
    let mut oracle = WindowsInkOracle::with_bounds(bounds);
    let mut recorded_taps = 0;

    for i in 0..200 {
        let raw_p = ((i * 41) % 8192) as u32;
        let tap = RawPointerEvent {
            message: ink::message::WM_POINTERDOWN,
            pointer_id: (i + 1) as u32,
            raw_pressure: raw_p,
            timestamp: (i * 20) as u32,
            ..Default::default()
        };

        if let Update::Set(Some(s)) = oracle.feed(&tap) {
            let expected_p = ink::normalize_pressure(raw_p, true, Some(bounds));
            assert!((s.pressure - expected_p).abs() < 1e-4);
            recorded_taps += 1;
        }

        let release = RawPointerEvent {
            message: ink::message::WM_POINTERUP,
            pointer_id: (i + 1) as u32,
            raw_pressure: 0,
            timestamp: (i * 20 + 5) as u32,
            ..Default::default()
        };
        let _ = oracle.feed(&release);
    }

    assert_eq!(recorded_taps, 200, "All 200 stippling micro-taps must be recorded without omissions");
}

// ============================================================================
// 9. Scenario: Wet-on-Wet Blending Wash
// ============================================================================

#[test]
fn test_t4_wet_on_wet_blending_wash() {
    // Multiple overlapping brush strokes building saturated coverage
    let mut tiles = HashMap::new();

    // Pass 1: broad wash at 50% opacity
    for x in 0..10 {
        blit_dab_into_tiles(&mut tiles, (x as f64) * 20.0 + 50.0, 50.0, 30.0, 0.5);
    }

    // Pass 2: overlapping perpendicular wash at 50% opacity
    for y in 0..10 {
        blit_dab_into_tiles(&mut tiles, 100.0, (y as f64) * 20.0 + 50.0, 30.0, 0.5);
    }

    // Intersection tile at (100, 50) must have saturated coverage higher than individual pass
    let intersection_tile = tiles.get(&(1, 0)).expect("intersection tile exists");
    let max_alpha = intersection_tile.data.iter().copied().fold(0.0f32, f32::max);
    assert!(max_alpha > 0.65, "Overlapping wash passes must accumulate coverage (> 0.65, got {})", max_alpha);
    assert!(max_alpha <= 1.0);
}

// ============================================================================
// 10. Scenario: Sketch-Erase-Resume Cycle
// ============================================================================

#[test]
fn test_t4_sketch_erase_resume_cycle() {
    let mut oracle = WindowsInkOracle::new();
    let mut tiles = HashMap::new();

    // 1. Sketch stroke with pen tip
    let stroke1 = RawPointerEvent {
        message: ink::message::WM_POINTERDOWN,
        pointer_id: 1,
        pen_flags: ink::pen_flag::NONE,
        raw_pressure: 800,
        x: 50,
        y: 50,
        ..Default::default()
    };
    if let Update::Set(Some(s)) = oracle.feed(&stroke1) {
        assert!(!s.eraser);
        blit_dab_into_tiles(&mut tiles, 50.0, 50.0, 15.0, s.pressure);
    }

    // Release stroke 1
    let _ = oracle.feed(&RawPointerEvent {
        message: ink::message::WM_POINTERUP,
        pointer_id: 1,
        ..Default::default()
    });

    // 2. Flip stylus to physical eraser to correct a section
    let erase_stroke = RawPointerEvent {
        message: ink::message::WM_POINTERDOWN,
        pointer_id: 1,
        pen_flags: ink::pen_flag::INVERTED,
        raw_pressure: 600,
        x: 55,
        y: 50,
        ..Default::default()
    };
    if let Update::Set(Some(s)) = oracle.feed(&erase_stroke) {
        assert!(s.eraser, "Stylus flip must switch to eraser tool");
    }

    // Release eraser
    let _ = oracle.feed(&RawPointerEvent {
        message: ink::message::WM_POINTERUP,
        pointer_id: 1,
        pen_flags: ink::pen_flag::INVERTED,
        ..Default::default()
    });

    // 3. Flip back to pen tip to continue sketch
    let resume_stroke = RawPointerEvent {
        message: ink::message::WM_POINTERDOWN,
        pointer_id: 1,
        pen_flags: ink::pen_flag::NONE,
        raw_pressure: 900,
        x: 70,
        y: 50,
        ..Default::default()
    };
    if let Update::Set(Some(s)) = oracle.feed(&resume_stroke) {
        assert!(!s.eraser, "Flipping back must restore drawing tool");
        blit_dab_into_tiles(&mut tiles, 70.0, 50.0, 15.0, s.pressure);
    }

    assert!(!tiles.is_empty(), "Sketch-erase-resume cycle must maintain canvas integrity");
}
