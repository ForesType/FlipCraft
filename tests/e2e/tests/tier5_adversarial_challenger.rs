//! Tier 5: Adversarial Empirical Challenge & Stress Test Suite
//!
//! Executed by Challenger 1 (Adversarial Stroke & Tablet Input Verifier).
//! Stress-tests extreme stroke dynamics, hairpin reversals, tail aggressiveness,
//! driver packet flooding (10k packets in 2ms), 16k/8k pressure scaling,
//! and tilt/rotation boundaries.

use photocraft_e2e_tests::*;
use photocraft_paint::brush::{BrushSettings, Control};
use photocraft_paint::curve::{apply_tail_aggressiveness, catmull_rom_tangent, fit_cubic_bezier};
use photocraft_paint::dynamics::{DabGenerator, DistanceAccumulator, StepResult};
use photocraft_tablet::Update;
use photocraft_tablet::ink::{self, DeviceBounds, RawPointerEvent, State as InkState, compress_events, is_eraser, map_rotation, map_tilt, normalize_pressure};
use std::time::Instant;

// ============================================================================
// 1. Extreme Stroke Dynamics: Ultra-Fast Flicks & Pressure Collapse
// ============================================================================

#[test]
fn test_adversarial_ultra_fast_flick_50000_px_s_taper() {
    // 50,000 px/s velocity: in 10ms, moves 500 px.
    // Pressure instantaneously collapses from 1.0 to 0.001.
    // Must achieve needle-sharp tapering with zero dab starvation.
    let mut brush = BrushSettings { size: 100.0, spacing: 0.25, spacing_enabled: true, ..Default::default() };
    brush.shape_dynamics.enabled = true;
    brush.shape_dynamics.size.control = Control::PenPressure;
    brush.shape_dynamics.size.minimum = 0.01; // minimum size = 1px

    let mut dab_generator = DabGenerator::new(&brush, 1.0);
    let mut dabs = Vec::new();
    let mut dual = Vec::new();

    let steps = 50;
    let mut pts = Vec::new();
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        let x = t * 500.0;
        let y = 0.0;
        // Instantaneous pressure collapse: starts at 1.0, plunges to 0.001
        let pressure = if i == 0 { 1.0 } else { (1.0 - t).powi(4).max(0.001) as f32 };
        let time = t * 10.0; // 10ms total = 50,000 px/s
        pts.push(StrokePoint { x, y, pressure, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time });
    }

    dab_generator.push(&pts, &mut dabs, &mut dual);
    dab_generator.finish(&mut dabs, &mut dual);

    assert!(!dabs.is_empty(), "50,000 px/s flick must generate dabs without starvation");

    // Verify needle-sharp tapering: diameter of final dabs must reach sub-pixel / micro-scale
    let first_dabs_radius = dabs[0].radius;
    let last_dabs_radius = dabs.last().unwrap().radius;
    assert!(first_dabs_radius > 30.0, "Initial dab radius should reflect high pressure: was {}", first_dabs_radius);
    assert!(last_dabs_radius <= 2.0, "Terminal dab radius must achieve needle-sharp taper: was {}", last_dabs_radius);

    // Verify continuous spacing: no massive gaps between consecutive dabs
    for w in dabs.windows(2) {
        let d = ((w[1].center.x - w[0].center.x).powi(2) + (w[1].center.y - w[0].center.y).powi(2)).sqrt();
        // At 100px brush with 25% spacing, spacing at start is ~25px and shrinks to <1px at end
        assert!(d <= 35.0, "Dab starvation gap detected: distance {} between dabs exceeds upper bound", d);
    }
}

#[test]
fn test_adversarial_distance_accumulator_immediate_trigger_verification() {
    let mut acc = DistanceAccumulator::new();
    // Start with large spacing (20px) under heavy pressure, accumulate 15px along segment
    let res1 = acc.step(15.0, 20.0);
    assert_eq!(res1, StepResult::Accumulated);
    assert_eq!(acc.accumulated_distance, 15.0);

    // Pressure collapses: required spacing drops to 5px.
    // 5.0 - 15.0 <= 0.0 -> must trigger immediate dab at t = 0.0 and reset accumulator
    let res2 = acc.step(2.0, 5.0);
    assert_eq!(res2, StepResult::Dab(0.0), "Spacing shrinkage must trigger immediate dab at t = 0.0");
    assert_eq!(acc.accumulated_distance, 0.0, "Accumulator must reset cleanly after immediate trigger");
}

// ============================================================================
// 2. 180-Degree Hairpin Turns & Curvature Continuity (C1 Tangents)
// ============================================================================

#[test]
fn test_adversarial_hairpin_180_degree_turn_continuity() {
    // 180° acute reversal: moving (0,0) -> (100,0) -> (0,0) -> (-100,0)
    let p0 = StrokePoint { x: 0.0, y: 0.0, pressure: 0.5, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: 0.0 };
    let p1 = StrokePoint { x: 100.0, y: 0.0, pressure: 0.5, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: 10.0 };
    let p2 = StrokePoint { x: 0.0, y: 0.0, pressure: 0.5, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: 20.0 };
    let p3 = StrokePoint { x: -100.0, y: 0.0, pressure: 0.5, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: 30.0 };

    let curve_pts = fit_cubic_bezier(&p0, &p1, &p2, &p3);
    assert!(curve_pts.len() >= 2, "Hairpin turn must subdivide into valid points");

    // All evaluated coordinates must be strictly finite (no NaN / Inf)
    for p in &curve_pts {
        assert!(p.x.is_finite() && p.y.is_finite(), "Curve point coordinates must be finite");
    }

    // Ensure endpoint anchors are strictly preserved
    assert_eq!(curve_pts.first().unwrap().x, p1.x);
    assert_eq!(curve_pts.first().unwrap().y, p1.y);
    assert_eq!(curve_pts.last().unwrap().x, p2.x);
    assert_eq!(curve_pts.last().unwrap().y, p2.y);
}

#[test]
fn test_adversarial_high_curvature_arc_c1_continuity() {
    // Sharp circular arc with radius 10px swept over 180 degrees
    let center_x = 50.0;
    let center_y = 50.0;
    let radius = 10.0;
    let n = 20;
    let mut arc_pts = Vec::new();

    for i in 0..=n {
        let theta = (i as f64 / n as f64) * std::f64::consts::PI;
        arc_pts.push(StrokePoint {
            x: center_x + radius * theta.cos(),
            y: center_y + radius * theta.sin(),
            pressure: 0.5,
            tilt_x: 0.0,
            tilt_y: 0.0,
            rotation: 0.0,
            wheel: 0.0,
            time: i as f64 * 5.0,
        });
    }

    // Check Catmull-Rom tangents along the arc
    for i in 1..arc_pts.len() - 1 {
        let t = catmull_rom_tangent(&arc_pts[i - 1], &arc_pts[i + 1]);
        assert!(t.0.is_finite() && t.1.is_finite());
        let speed = (t.0 * t.0 + t.1 * t.1).sqrt();
        assert!(speed > 0.0, "Tangents on arc must be non-zero");
    }
}

// ============================================================================
// 3. Tail Aggressiveness Lift-off Convergence
// ============================================================================

#[test]
fn test_adversarial_tail_aggressiveness_lift_off_convergence() {
    // Stylus lifting rapidly: pressure decays from 0.8 down to 0.02
    let mut stroke = vec![
        StrokePoint { x: 10.0, y: 10.0, pressure: 0.8, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: 0.0 },
        StrokePoint { x: 30.0, y: 15.0, pressure: 0.6, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: 5.0 },
        StrokePoint { x: 60.0, y: 22.0, pressure: 0.3, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: 10.0 },
        StrokePoint { x: 90.0, y: 30.0, pressure: 0.05, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: 15.0 },
        StrokePoint { x: 120.0, y: 40.0, pressure: 0.01, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: 20.0 },
    ];

    let physical_tip = *stroke.last().unwrap();
    apply_tail_aggressiveness(&mut stroke, 1.0);

    // Terminal coordinate must converge tightly onto physical pen tip
    let final_point = *stroke.last().unwrap();
    let dx = final_point.x - physical_tip.x;
    let dy = final_point.y - physical_tip.y;
    let dist_to_tip = (dx * dx + dy * dy).sqrt();
    assert!(dist_to_tip < 1e-4, "Terminal point must converge exactly onto physical pen tip at lift-off: was {}", dist_to_tip);

    // The preceding point should also be pulled significantly toward the physical tip
    let penultimate = stroke[stroke.len() - 2];
    assert!(penultimate.x > 80.0, "Tail aggressiveness should pull trailing points towards lift-off tip");
}

#[test]
fn test_adversarial_tail_aggressiveness_no_pull_on_downstroke() {
    // Stylus pressing DOWN: pressure increasing from 0.1 to 0.9
    let mut stroke = vec![
        StrokePoint { x: 10.0, y: 10.0, pressure: 0.1, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: 0.0 },
        StrokePoint { x: 30.0, y: 15.0, pressure: 0.4, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: 5.0 },
        StrokePoint { x: 60.0, y: 22.0, pressure: 0.7, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: 10.0 },
        StrokePoint { x: 90.0, y: 30.0, pressure: 0.9, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: 15.0 },
    ];

    let orig = stroke.clone();
    apply_tail_aggressiveness(&mut stroke, 1.0);

    // On downstroke, tail aggressiveness must have ZERO displacement error (0.000px)
    for i in 0..stroke.len() {
        let err_x = (stroke[i].x - orig[i].x).abs();
        let err_y = (stroke[i].y - orig[i].y).abs();
        assert_eq!(err_x, 0.0, "Downstroke coordinate x must have exactly 0.000px displacement error");
        assert_eq!(err_y, 0.0, "Downstroke coordinate y must have exactly 0.000px displacement error");
    }
}

#[test]
fn test_adversarial_tail_aggressiveness_zero_displacement_on_steady_stroke() {
    let mut stroke = vec![
        StrokePoint { x: 10.0, y: 20.0, pressure: 0.5, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: 0.0 },
        StrokePoint { x: 50.0, y: 20.0, pressure: 0.5, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: 10.0 },
        StrokePoint { x: 90.0, y: 20.0, pressure: 0.5, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, wheel: 0.0, time: 20.0 },
    ];
    let orig = stroke.clone();
    apply_tail_aggressiveness(&mut stroke, 5.0);

    for i in 0..stroke.len() {
        let err_x = (stroke[i].x - orig[i].x).abs();
        let err_y = (stroke[i].y - orig[i].y).abs();
        assert_eq!(err_x, 0.0, "Steady stroke coordinate x must have exactly 0.000px displacement error");
        assert_eq!(err_y, 0.0, "Steady stroke coordinate y must have exactly 0.000px displacement error");
    }
    assert_eq!(stroke, orig, "Steady stroke must experience exactly 0.000px displacement error");
}

// ============================================================================
// 4. Tablet Driver Resilience: 10,000 Packet Flood in 2ms
// ============================================================================

#[test]
fn test_adversarial_10000_packet_flood_within_2ms() {
    let mut state = InkState::new();
    let mut stream = Vec::with_capacity(10_000);

    // 10,000 packets delivered within 2ms (microsecond intervals)
    for i in 0..10_000 {
        let frac = i as f64 / 10_000.0;
        stream.push(RawPointerEvent {
            message: if i == 0 {
                ink::message::WM_POINTERDOWN
            } else if i == 9_999 {
                ink::message::WM_POINTERUP
            } else {
                ink::message::WM_POINTERUPDATE
            },
            pointer_id: 1,
            pointer_flags: if i == 9_999 { ink::pointer_flag::UP } else { ink::pointer_flag::IN_RANGE | ink::pointer_flag::IN_CONTACT },
            x: (100.0 + frac * 200.0) as i32,
            y: (100.0 + (frac * 10.0).sin() * 50.0) as i32,
            raw_pressure: (500.0 + 300.0 * (frac * 5.0).sin()) as u32,
            timestamp: (frac * 2.0) as u32, // all within 2ms!
            ..Default::default()
        });
    }

    let start = Instant::now();
    let mut delivered = 0;
    let mut _dropped = 0;

    for ev in &stream {
        match state.handle(ev) {
            Update::Set(Some(_)) => delivered += 1,
            Update::Set(None) => {}
            Update::Keep => _dropped += 1,
        }
    }
    let elapsed = start.elapsed();

    assert!(elapsed.as_millis() < 500, "10,000 packet flood must process within 500ms: took {:?}", elapsed);
    assert!(delivered > 0, "At least start and end boundary packets must be delivered");

    // Also test batch compress_events function on 10,000 packets
    let start_comp = Instant::now();
    let compressed = compress_events(&stream);
    let comp_elapsed = start_comp.elapsed();

    assert!(comp_elapsed.as_millis() < 500, "compress_events on 10,000 packets must complete quickly: took {:?}", comp_elapsed);
    assert!(!compressed.is_empty(), "Compressed stream must not be empty");
    assert!(compressed.len() <= stream.len(), "Compressed stream cannot exceed raw stream length");
}

// ============================================================================
// 5. Linear 16k & 8k Digitizer Pressure Scaling (No premature 1024 saturation)
// ============================================================================

#[test]
fn test_adversarial_pressure_scaling_16k_8k_no_premature_saturation() {
    let b8k = DeviceBounds::LEVELS_8K; // 0..8191
    let b16k = DeviceBounds::LEVELS_16K; // 0..16383

    // Test raw 1024 on 8K: must be ~0.125, NOT 1.0!
    let p_1024_8k = normalize_pressure(1024, true, Some(b8k));
    assert!((p_1024_8k - 1024.0 / 8191.0).abs() < 1e-4, "8K digitizer at raw 1024 must normalize to ~0.125: was {}", p_1024_8k);
    assert!(p_1024_8k < 0.2, "8K digitizer must NOT saturate at 1024!");

    // Test raw 1024 on 16K: must be ~0.0625, NOT 1.0!
    let p_1024_16k = normalize_pressure(1024, true, Some(b16k));
    assert!((p_1024_16k - 1024.0 / 16383.0).abs() < 1e-4, "16K digitizer at raw 1024 must normalize to ~0.0625: was {}", p_1024_16k);
    assert!(p_1024_16k < 0.1, "16K digitizer must NOT saturate at 1024!");

    // Test micro-pressure sensitivity at level 1
    let p_micro_8k = normalize_pressure(1, true, Some(b8k));
    assert!(p_micro_8k > 0.0 && p_micro_8k < 0.001, "Micro pressure level 1 must be registered");
    let p_micro_16k = normalize_pressure(1, true, Some(b16k));
    assert!(p_micro_16k > 0.0 && p_micro_16k < 0.0005, "Micro pressure level 1 must be registered on 16k");

    // Full scale saturation: only at 8191 and 16383
    let p_full_8k = normalize_pressure(8191, true, Some(b8k));
    assert_eq!(p_full_8k, 1.0, "8191 on 8K must reach 1.0");
    let p_full_16k = normalize_pressure(16383, true, Some(b16k));
    assert_eq!(p_full_16k, 1.0, "16383 on 16K must reach 1.0");

    // Values beyond logicalMax must clamp safely to 1.0
    let p_over_8k = normalize_pressure(9000, true, Some(b8k));
    assert_eq!(p_over_8k, 1.0, "Over-range raw pressure must clamp to 1.0");

    // Monotonicity check across full 8k and 16k ranges
    let mut prev = 0.0f32;
    for raw in (0..=8191).step_by(128) {
        let p = normalize_pressure(raw, true, Some(b8k));
        assert!(p >= prev, "8K pressure must be strictly monotonic");
        prev = p;
    }
}

// ============================================================================
// 6. Inverted Eraser Flag, Tilt & Rotation Boundary Extremes
// ============================================================================

#[test]
fn test_adversarial_eraser_inversion_and_flags() {
    assert!(!is_eraser(ink::pen_flag::NONE));
    assert!(!is_eraser(ink::pen_flag::BARREL));
    assert!(is_eraser(ink::pen_flag::INVERTED));
    assert!(is_eraser(ink::pen_flag::ERASER));
    assert!(is_eraser(ink::pen_flag::INVERTED | ink::pen_flag::ERASER));

    // Mid-stroke flip test
    let mut state = InkState::new();
    let down = RawPointerEvent { message: ink::message::WM_POINTERDOWN, pen_flags: ink::pen_flag::NONE, raw_pressure: 500, ..Default::default() };
    if let Update::Set(Some(s)) = state.handle(&down) {
        assert!(!s.eraser, "Pen tip should not be eraser");
    }

    let mid_flip =
        RawPointerEvent { message: ink::message::WM_POINTERUPDATE, pen_flags: ink::pen_flag::INVERTED, raw_pressure: 500, timestamp: 50, ..Default::default() };
    if let Update::Set(Some(s)) = state.handle(&mid_flip) {
        assert!(s.eraser, "Inverted pen flag mid-stroke must set eraser true");
    }

    let flip_back =
        RawPointerEvent { message: ink::message::WM_POINTERUPDATE, pen_flags: ink::pen_flag::NONE, raw_pressure: 500, timestamp: 100, ..Default::default() };
    if let Update::Set(Some(s)) = state.handle(&flip_back) {
        assert!(!s.eraser, "Flipping back to tip must set eraser false");
    }
}

#[test]
fn test_adversarial_tilt_and_rotation_boundary_extremes() {
    // Tilt boundary extremes: -90..+90
    assert_eq!(map_tilt(0, true), 0.0);
    assert_eq!(map_tilt(90, true), 90.0);
    assert_eq!(map_tilt(-90, true), -90.0);
    assert_eq!(map_tilt(120, true), 90.0, "Tilt > 90 must clamp to 90");
    assert_eq!(map_tilt(-120, true), -90.0, "Tilt < -90 must clamp to -90");
    assert_eq!(map_tilt(45, false), 0.0, "Invalid tilt mask must return 0.0");

    // Rotation boundary extremes: 0..359
    assert_eq!(map_rotation(0, true), 0.0);
    assert_eq!(map_rotation(180, true), 180.0);
    assert_eq!(map_rotation(359, true), 359.0);
    assert_eq!(map_rotation(360, true), 0.0, "360 must wrap cleanly to 0");
    assert_eq!(map_rotation(720, true), 0.0, "720 must wrap cleanly to 0");
    assert_eq!(map_rotation(450, true), 90.0, "450 must wrap to 90");
    assert_eq!(map_rotation(180, false), 0.0, "Invalid rotation mask must return 0.0");
}

// ============================================================================
// 7. Empirical Reproduction of Upstream Failure Modes
// ============================================================================

#[test]
fn test_empirical_investigation_rate_compression_behavior() {
    let mut state = InkState::new();

    let mut stream = Vec::new();
    for i in 0i32..50 {
        stream.push(RawPointerEvent {
            message: ink::message::WM_POINTERUPDATE,
            x: 250 + i * 4,
            y: 250,
            raw_pressure: 1024,
            timestamp: (i * 4) as u32,
            ..Default::default()
        });
    }

    let mut _dropped = 0;
    let mut delivered = 0;
    for ev in &stream {
        match state.handle(ev) {
            Update::Set(Some(_)) => delivered += 1,
            Update::Set(None) => {}
            Update::Keep => _dropped += 1,
        }
    }

    println!("[EMPIRICAL FINDING] 240Hz moving packets (dt=4, dx=4): delivered={}, dropped={}", delivered, _dropped);
}

#[test]
fn test_empirical_investigation_8k_auto_detect_pressure_bias() {
    let mut state = InkState::new();
    let raw_p = 41u32;
    let tap = RawPointerEvent { message: ink::message::WM_POINTERDOWN, raw_pressure: raw_p, ..Default::default() };

    if let Update::Set(Some(s)) = state.handle(&tap) {
        let expected_with_8k = normalize_pressure(raw_p, true, Some(DeviceBounds::LEVELS_8K));
        println!("[EMPIRICAL FINDING] Raw pressure 41 on unconfigured state: actual={}, expected_8k={}", s.pressure, expected_with_8k);
    }
}

#[test]
fn test_adversarial_240hz_rate_compression_preservation_invariants() {
    let mut state = InkState::new();

    // 1. WM_POINTERDOWN boundary must ALWAYS be delivered
    let down = RawPointerEvent {
        message: ink::message::WM_POINTERDOWN,
        pointer_flags: ink::pointer_flag::IN_CONTACT | ink::pointer_flag::IN_RANGE,
        x: 100,
        y: 100,
        raw_pressure: 512,
        timestamp: 0,
        ..Default::default()
    };
    let res = state.handle(&down);
    assert!(matches!(res, Update::Set(Some(_))), "WM_POINTERDOWN boundary must be delivered immediately");

    // 2. Continuous 240Hz linear movement (dt=4ms < 8ms, dx=4, dy=0)
    // Intermediate packets within 8ms must be throttled to 120Hz (every other packet dropped)
    let p1 = RawPointerEvent {
        message: ink::message::WM_POINTERUPDATE,
        pointer_flags: ink::pointer_flag::IN_CONTACT | ink::pointer_flag::IN_RANGE,
        x: 104,
        y: 100,
        raw_pressure: 512,
        timestamp: 4,
        ..Default::default()
    };
    let res1 = state.handle(&p1);
    assert_eq!(res1, Update::Keep, "Intermediate 240Hz packet at dt=4ms must be throttled (dropped)");

    let p2 = RawPointerEvent {
        message: ink::message::WM_POINTERUPDATE,
        pointer_flags: ink::pointer_flag::IN_CONTACT | ink::pointer_flag::IN_RANGE,
        x: 108,
        y: 100,
        raw_pressure: 512,
        timestamp: 8,
        ..Default::default()
    };
    let res2 = state.handle(&p2);
    assert!(matches!(res2, Update::Set(Some(_))), "Intermediate packet at dt=8ms (120Hz) must be delivered");

    // 3. Eraser flip mid-stroke (even if dt=4ms < 8ms) must NEVER be dropped
    let p_eraser = RawPointerEvent {
        message: ink::message::WM_POINTERUPDATE,
        pointer_flags: ink::pointer_flag::IN_CONTACT | ink::pointer_flag::IN_RANGE,
        pen_flags: ink::pen_flag::INVERTED,
        x: 112,
        y: 100,
        raw_pressure: 512,
        timestamp: 12,
        ..Default::default()
    };
    let res_eraser = state.handle(&p_eraser);
    if let Update::Set(Some(s)) = res_eraser {
        assert!(s.eraser, "Eraser flip at dt=4ms must be delivered and have eraser=true");
    } else {
        panic!("Eraser flip packet must never be dropped by rate compression!");
    }

    // 4. Pressure spike (|ΔP| >= 0.05) even at dt=4ms must NEVER be dropped
    // Current normalized pressure: 512 / 1024 = 0.5. Change to 600 / 1024 = 0.5859 (|ΔP| = 0.0859 >= 0.05)
    let p_spike = RawPointerEvent {
        message: ink::message::WM_POINTERUPDATE,
        pointer_flags: ink::pointer_flag::IN_CONTACT | ink::pointer_flag::IN_RANGE,
        pen_flags: ink::pen_flag::INVERTED,
        x: 116,
        y: 100,
        raw_pressure: 600,
        timestamp: 16,
        ..Default::default()
    };
    let res_spike = state.handle(&p_spike);
    assert!(matches!(res_spike, Update::Set(Some(_))), "Pressure spike (|ΔP| >= 0.05) must never be dropped");

    // 5. Trajectory inflection point (> 45 degree turn) even at dt=4ms must NEVER be dropped
    // Prior movement was along x-axis (dx=4, dy=0).
    // Now make a 90-degree turn: x remains 116, y changes to 106 (dx=0, dy=6).
    let p_inflect = RawPointerEvent {
        message: ink::message::WM_POINTERUPDATE,
        pointer_flags: ink::pointer_flag::IN_CONTACT | ink::pointer_flag::IN_RANGE,
        pen_flags: ink::pen_flag::INVERTED,
        x: 116,
        y: 106,
        raw_pressure: 600,
        timestamp: 20,
        ..Default::default()
    };
    let res_inflect = state.handle(&p_inflect);
    assert!(matches!(res_inflect, Update::Set(Some(_))), "Sharp corner inflection angle (>45 deg) must never be dropped");

    // 6. WM_POINTERUP boundary must ALWAYS be delivered
    let up = RawPointerEvent {
        message: ink::message::WM_POINTERUP,
        pointer_flags: ink::pointer_flag::UP,
        x: 116,
        y: 106,
        raw_pressure: 0,
        timestamp: 24,
        ..Default::default()
    };
    let res_up = state.handle(&up);
    assert!(matches!(res_up, Update::Set(Some(_))), "WM_POINTERUP boundary must be delivered immediately");
}
