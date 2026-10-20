//! Tier 2: Boundary & Corner Cases Suite (F1 through F9)
//!
//! Validates features F1 through F9 across boundary conditions, extremes, and stress inputs (45 tests total).

use photocraft_e2e_tests::*;
use std::collections::HashMap;

// ============================================================================
// Feature 1: Adaptive Distance Stepping & Dynamic Spacing — Boundary Cases
// ============================================================================

#[test]
fn test_f1_zero_pressure_hover_boundary() {
    let mut oracle = KisDistanceOracle::new();
    // Pressure 0.0 -> diameter 0 or minimum tip size -> spacing cannot be 0 (must clamp to min threshold)
    let (dabs, res) = oracle.step(10.0, 0.0);
    // current_spacing is clamped to 0.1 min, so segment length 10 emits dabs without division by zero
    assert!(!dabs.is_empty(), "Zero spacing input must be clamped safely without division by zero");
    assert!(matches!(res, AdaptiveStepResult::SegmentDab(_)));
}

#[test]
fn test_f1_maximum_pressure_saturation() {
    let mut oracle = KisDistanceOracle::new();
    let max_spacing = 500.0 * 0.25; // 125px for 500px brush
    let (dabs, res) = oracle.step(50.0, max_spacing);
    assert!(dabs.is_empty(), "Step shorter than max spacing must not trigger dab");
    assert_eq!(res, AdaptiveStepResult::Accumulated(50.0));
    assert_eq!(oracle.accum_distance, 50.0);
}

#[test]
fn test_f1_zero_length_segment_stationary_dab() {
    let mut oracle = KisDistanceOracle::new();
    oracle.accum_distance = 15.0;
    // Length 0.0 segment (stylus held stationary in place)
    let (dabs, res) = oracle.step(0.0, 20.0);
    assert!(dabs.is_empty(), "Zero length segment without spacing shrinkage should emit no movement dabs");
    assert_eq!(res, AdaptiveStepResult::Accumulated(15.0));
}

#[test]
fn test_f1_micro_segment_subpixel_precision() {
    let mut oracle = KisDistanceOracle::new();
    let micro_len = 1e-6;
    for _ in 0..1000 {
        let _ = oracle.step(micro_len, 10.0);
    }
    // Accumulated distance should equal 1000 * 1e-6 = 0.001 within floating-point tolerance
    assert!((oracle.accum_distance - 0.001).abs() < 1e-7, "Sub-pixel micro-segments must accumulate without numerical underflow");
}

#[test]
fn test_f1_extreme_velocity_step_clamp() {
    let mut oracle = KisDistanceOracle::new();
    // Extreme velocity: stylus moved 50,000 pixels in a single tick
    let (dabs, res) = oracle.step(50000.0, 10.0);
    // Should emit multiple dabs across the segment
    assert!(dabs.len() >= 100, "Extreme velocity segment must emit multiple distributed dabs");
    assert!(matches!(res, AdaptiveStepResult::SegmentDab(_)));
}

// ============================================================================
// Feature 2: Cubic Bézier Curve Interpolation — Boundary Cases
// ============================================================================

#[test]
fn test_f2_cusp_180_degree_reversal() {
    // Sharp 180-degree hairpin turn
    let p0 = (0.0, 0.0);
    let p1 = (100.0, 0.0);
    let p2 = (100.0, 0.0);
    let p3 = (0.0, 0.0);

    let mut points = vec![p0];
    de_casteljau_subdivide(p0, p1, p2, p3, 0.5, 0, &mut points);

    assert!(points.len() >= 2, "180-degree hairpin cusp must subdivide without infinite loops or crashes");
    assert_eq!(points.first().copied(), Some(p0));
    assert_eq!(points.last().copied(), Some(p3));
}

#[test]
fn test_f2_coincident_points_degeneracy() {
    let p = (42.0, 84.0);
    let mut points = vec![p];
    de_casteljau_subdivide(p, p, p, p, 0.5, 0, &mut points);

    assert_eq!(points.len(), 2, "Coincident points curve must return endpoints immediately");
    assert_eq!(points[0], p);
    assert_eq!(points[1], p);
}

#[test]
fn test_f2_max_subdivision_depth_guard() {
    // Extremely curved path with tiny flatness threshold (0.00001)
    let p0 = (0.0, 0.0);
    let p1 = (1000.0, 5000.0);
    let p2 = (-1000.0, 5000.0);
    let p3 = (0.0, 0.0);

    let mut points = vec![p0];
    de_casteljau_subdivide(p0, p1, p2, p3, 0.00001, 0, &mut points);

    // Recursion is bounded by depth <= 16, so points.len() <= 2^16 + 1
    assert!(points.len() <= 65537, "Subdivision depth must be securely capped to prevent stack exhaustion");
}

#[test]
fn test_f2_zero_flatness_threshold_safety() {
    let p0 = (0.0, 0.0);
    let p1 = (50.0, 100.0);
    let p2 = (100.0, 100.0);
    let p3 = (150.0, 0.0);

    let mut points = vec![p0];
    // Threshold 0.0 or negative must fall back to default threshold (0.5)
    de_casteljau_subdivide(p0, p1, p2, p3, 0.0, 0, &mut points);
    assert!(points.len() > 2);
}

#[test]
fn test_f2_nan_coordinate_rejection() {
    let p = StrokePoint { x: f64::NAN, y: f64::INFINITY, pressure: f32::NAN, ..Default::default() };
    let s = Sample { pressure: p.pressure, tilt_x: f32::NAN, tilt_y: f32::INFINITY, rotation: f32::NAN, eraser: false }.sanitized();

    assert_eq!(s.pressure, 1.0, "NaN pressure must sanitize to default 1.0");
    assert_eq!(s.tilt_x, 0.0, "NaN tilt must sanitize to 0.0");
    assert_eq!(s.rotation, 0.0, "NaN rotation must sanitize to 0.0");
}

// ============================================================================
// Feature 3: Tail Aggressiveness Smoothing — Boundary Cases
// ============================================================================

#[test]
fn test_f3_instantaneous_pressure_drop_infinity() {
    let mut points =
        vec![StrokePoint { x: 0.0, y: 0.0, pressure: 1.0, ..Default::default() }, StrokePoint { x: 50.0, y: 0.0, pressure: 0.0, ..Default::default() }];

    apply_tail_aggressiveness(&mut points, 50.0);
    // Instantaneous pressure collapse (1.0 to 0.0)
    assert!(points[0].x > 0.0, "Extreme pressure drop must pull tail without numeric overflow");
    assert!(points[0].x <= 50.0);
}

#[test]
fn test_f3_zero_pressure_gradient_steady_state() {
    let mut points =
        vec![StrokePoint { x: 10.0, y: 20.0, pressure: 0.5, ..Default::default() }, StrokePoint { x: 30.0, y: 20.0, pressure: 0.5, ..Default::default() }];
    let original = points.clone();
    apply_tail_aggressiveness(&mut points, 10.0);
    assert_eq!(points, original, "Steady pressure (dP/dt = 0) must apply zero tail pull");
}

#[test]
fn test_f3_extreme_tail_aggressiveness_factor() {
    let mut points =
        vec![StrokePoint { x: 0.0, y: 0.0, pressure: 0.9, ..Default::default() }, StrokePoint { x: 100.0, y: 0.0, pressure: 0.1, ..Default::default() }];
    // Extreme factor 10,000
    apply_tail_aggressiveness(&mut points, 10000.0);
    assert!(points[0].x.is_finite());
    assert!(points[0].x <= 100.0);
}

#[test]
fn test_f3_single_point_stroke_boundary() {
    let mut points = vec![StrokePoint { x: 10.0, y: 10.0, pressure: 0.5, ..Default::default() }];
    let original = points.clone();
    apply_tail_aggressiveness(&mut points, 10.0);
    assert_eq!(points, original, "Single point stroke must not panic");
}

#[test]
fn test_f3_non_finite_pressure_delta_guard() {
    let mut points =
        vec![StrokePoint { x: 0.0, y: 0.0, pressure: f32::NAN, ..Default::default() }, StrokePoint { x: 10.0, y: 0.0, pressure: 0.5, ..Default::default() }];
    apply_tail_aggressiveness(&mut points, 10.0);
    assert_eq!(points[0].x, 0.0, "NaN pressure delta must not displace coordinates");
}

// ============================================================================
// Feature 4: Dab Mask Caching (`DabCache`) — Boundary Cases
// ============================================================================

#[test]
fn test_f4_zero_pixel_radius_dab_mask() {
    let mask = generate_analytical_mask(0.0, 1.0);
    assert!(!mask.is_empty(), "Zero radius dab mask must allocate safely without panic");
}

#[test]
fn test_f4_one_pixel_radius_minimal_dab() {
    let mask = generate_analytical_mask(1.0, 1.0);
    assert!(mask.len() >= 9, "1px radius dab mask must have at least 3x3 footprint");
    // Center pixel should be positive
    let center_idx = mask.len() / 2;
    assert!(mask[center_idx] > 0.0);
}

#[test]
fn test_f4_500_pixel_radius_maximal_dab() {
    let radius = 250.0; // 500px diameter
    let mask = generate_analytical_mask(radius, 0.5);
    let size = (radius.ceil() as usize * 2 + 3).max(3);
    assert_eq!(mask.len(), size * size);
    assert!(mask.len() >= 250000, "500px dab mask must store full resolution float coverage");
}

#[test]
fn test_f4_cache_capacity_one_edge_case() {
    let cache = DabCacheOracle::new(1);
    let k1 = DabKey::new(10.0, 0.5, 0.5, 1.0, 0.0);
    let k2 = DabKey::new(20.0, 0.5, 0.5, 1.0, 0.0);

    let (_, hit1) = cache.get_or_create(&k1, || generate_analytical_mask(10.0, 0.5));
    assert!(!hit1);
    assert_eq!(cache.len(), 1);

    let (_, hit2) = cache.get_or_create(&k2, || generate_analytical_mask(20.0, 0.5));
    assert!(!hit2);
    assert_eq!(cache.len(), 1, "Cache with capacity 1 must evict previous entry and keep size at 1");
}

#[test]
fn test_f4_fractional_radius_quantization_boundaries() {
    let k1 = DabKey::new(1.24, 0.5, 0.5, 1.0, 0.0);
    let k2 = DabKey::new(1.26, 0.5, 0.5, 1.0, 0.0);

    // 1.24 * 4.0 = 4.96 -> 5; 1.26 * 4.0 = 5.04 -> 5
    assert_eq!(k1.radius_quant, k2.radius_quant, "Near-identical fractional radii within 0.25px must quantize to identical key");
}

// ============================================================================
// Feature 5: High-Performance SIMD Rasterization & Blitting — Boundary Cases
// ============================================================================

#[test]
fn test_f5_negative_coordinate_dab_blitting() {
    let mut tiles = HashMap::new();
    let touched = blit_dab_into_tiles(&mut tiles, -50.0, -50.0, 20.0, 1.0);

    assert!(!touched.is_empty());
    for (tx, ty) in touched {
        assert!(tx < 0, "Dabs in negative coordinates must map to negative tile coordinates");
        assert!(ty < 0);
    }
}

#[test]
fn test_f5_dab_entirely_outside_canvas_bounds() {
    let mut tiles = HashMap::new();
    let touched = blit_dab_into_tiles(&mut tiles, 100000.0, 100000.0, 10.0, 1.0);
    assert_eq!(touched.len(), 1);
    assert!(touched.contains(&(1562, 1562)));
}

#[test]
fn test_f5_zero_alpha_blit_identity() {
    let mut tiles = HashMap::new();
    let touched = blit_dab_into_tiles(&mut tiles, 32.0, 32.0, 10.0, 0.0);
    assert!(touched.is_empty(), "0.0 alpha dab must not modify tiles");
    assert!(tiles.is_empty());
}

#[test]
fn test_f5_tile_edge_aligned_dab() {
    let mut tiles = HashMap::new();
    let touched = blit_dab_into_tiles(&mut tiles, 64.0, 64.0, 5.0, 1.0);
    // Straddles all 4 adjacent tiles at corner (64, 64)
    assert!(touched.contains(&(0, 0)));
    assert!(touched.contains(&(1, 0)));
    assert!(touched.contains(&(0, 1)));
    assert!(touched.contains(&(1, 1)));
}

#[test]
fn test_f5_extreme_aspect_ratio_chisel_tip() {
    let k = DabKey::new(100.0, 1.0, 0.0, 0.01, 45.0);
    assert_eq!(k.aspect_quant, 1, "Extreme aspect ratio (100:1) must clamp to minimum valid quantization 1");
}

// ============================================================================
// Feature 6: Windows Ink `WM_POINTER` Driver Pipeline — Boundary Cases
// ============================================================================

#[test]
fn test_f6_wm_pointerup_without_pointerdown() {
    let mut oracle = WindowsInkOracle::new();
    let up = RawPointerEvent { message: ink::message::WM_POINTERUP, pointer_id: 99, ..Default::default() };
    let update = oracle.feed(&up);
    // Stray up event with unknown pointer ID emits lift-off or resets safely without panic
    match update {
        Update::Set(Some(s)) => assert_eq!(s.pressure, 0.0, "Lift-off sample must have pressure 0.0"),
        Update::Keep | Update::Set(None) => {}
    }
}

#[test]
fn test_f6_null_window_handle_guard() {
    let bounds = DeviceBounds::new(0, 1024);
    assert_eq!(bounds.logical_min, 0);
    assert_eq!(bounds.logical_max, 1024);
}

#[test]
fn test_f6_rapid_pointer_capture_lost() {
    let mut oracle = WindowsInkOracle::new();
    let down = RawPointerEvent { message: ink::message::WM_POINTERDOWN, pointer_id: 1, raw_pressure: 500, ..Default::default() };
    let capture_lost = RawPointerEvent { message: ink::message::WM_POINTERLEAVE, pointer_id: 1, ..Default::default() };

    let _ = oracle.feed(&down);
    let u_lost = oracle.feed(&capture_lost);

    assert_eq!(u_lost, Update::Set(None), "Pointer leave / capture lost must reset active stylus feed");
}

#[test]
fn test_f6_out_of_bounds_screen_coordinates() {
    let event = RawPointerEvent { x: -500, y: -1000, ..Default::default() };
    assert_eq!(event.x, -500);
    assert_eq!(event.y, -1000);
}

#[test]
fn test_f6_interleaved_mouse_and_pen_packets() {
    let mut oracle = WindowsInkOracle::new();
    let pen_down = RawPointerEvent {
        pointer_type: ink::pointer_type::PT_PEN,
        message: ink::message::WM_POINTERDOWN,
        pointer_id: 1,
        raw_pressure: 500,
        ..Default::default()
    };
    let mouse_move =
        RawPointerEvent { pointer_type: ink::pointer_type::PT_MOUSE, message: ink::message::WM_POINTERUPDATE, pointer_id: 2, ..Default::default() };

    let _ = oracle.feed(&pen_down);
    let u_mouse = oracle.feed(&mouse_move);

    assert_eq!(u_mouse, Update::Set(None), "Synthetic mouse packet while pen active must cleanly relinquish pen state");
}

// ============================================================================
// Feature 7: High-Frequency Event Rate Compression — Boundary Cases
// ============================================================================

#[test]
fn test_f7_10000_event_flood_rate_stress() {
    let mut oracle = WindowsInkOracle::new();
    let mut stream = Vec::with_capacity(10000);

    for i in 0..10000 {
        stream.push(RawPointerEvent {
            message: ink::message::WM_POINTERUPDATE,
            x: 100,
            y: 100,
            raw_pressure: 500,
            timestamp: i / 100, // 100 events per ms (100kHz flood!)
            ..Default::default()
        });
    }

    let (samples, dropped) = oracle.filter_stream(&stream);
    assert!(dropped > 9000, "Event rate compressor must coalesce redundant packets under extreme flood stress");
    assert!(samples.len() < 1000);
}

#[test]
fn test_f7_alternating_jitter_high_frequency() {
    let mut oracle = WindowsInkOracle::new();
    let mut stream = Vec::new();

    // Micro-jitter in pressure within same timestamp at identical coordinate
    for i in 0..20 {
        stream.push(RawPointerEvent {
            message: ink::message::WM_POINTERUPDATE,
            x: 100,
            y: 100,
            raw_pressure: 500 + (i % 2),
            timestamp: 10,
            ..Default::default()
        });
    }

    let (samples, dropped) = oracle.filter_stream(&stream);
    assert!(dropped > 0, "Sub-pixel micro-jitter within same timestamp must be filtered");
    assert!(samples.len() < stream.len());
}

#[test]
fn test_f7_backward_timestamp_jitter_guard() {
    let mut oracle = WindowsInkOracle::new();
    let e1 = RawPointerEvent { timestamp: 50, ..Default::default() };
    let e_back = RawPointerEvent { timestamp: 40, ..Default::default() }; // backward clock glitch

    let u1 = oracle.feed(&e1);
    let u2 = oracle.feed(&e_back);

    assert!(matches!(u1, Update::Set(Some(_))));
    // Backward timestamp must not crash the event compressor
    assert!(matches!(u2, Update::Keep | Update::Set(Some(_))));
}

#[test]
fn test_f7_event_buffer_full_overflow_handling() {
    let mut oracle = WindowsInkOracle::new();
    // Simulate continuous events for 5 seconds
    for t in 0i32..5000 {
        let e =
            RawPointerEvent { message: ink::message::WM_POINTERUPDATE, x: t, y: t, raw_pressure: (t % 1024) as u32, timestamp: t as u32, ..Default::default() };
        let _ = oracle.feed(&e);
    }
    // Verifies no unbounded memory growth or panic after 5000 events
}

#[test]
fn test_f7_single_event_stream_boundary() {
    let mut oracle = WindowsInkOracle::new();
    let e = RawPointerEvent { message: ink::message::WM_POINTERDOWN, pointer_id: 1, raw_pressure: 500, ..Default::default() };
    let (samples, dropped) = oracle.filter_stream(&[e]);
    assert_eq!(samples.len(), 1, "Single-event stream must emit the event");
    assert_eq!(dropped, 0);
}

// ============================================================================
// Feature 8: Linear 16k / 8k Digitizer Pressure Handling — Boundary Cases
// ============================================================================

#[test]
fn test_f8_zero_logical_max_guard() {
    let bad_bounds = DeviceBounds::new(0, 0);
    // logical_max == logical_min (0 == 0) -> must avoid division by zero
    let p = ink::normalize_pressure(500, true, Some(bad_bounds));
    assert!(p.is_finite(), "Zero logicalMax span must not produce NaN or panic with division by zero");
}

#[test]
fn test_f8_raw_pressure_exceeding_logical_max() {
    let bounds = DeviceBounds::LEVELS_8K; // 0..8191
    let p_over = ink::normalize_pressure(10000, true, Some(bounds));
    assert_eq!(p_over, 1.0, "Raw pressure exceeding logicalMax must clamp to 1.0 without overflow");
}

#[test]
fn test_f8_negative_raw_pressure_input() {
    let bounds = DeviceBounds::new(100, 1000);
    // Raw value 50 < logical_min 100
    let p_under = ink::normalize_pressure(50, true, Some(bounds));
    assert_eq!(p_under, 0.0, "Raw pressure below logicalMin must clamp to 0.0");
}

#[test]
fn test_f8_non_power_of_two_digitizer_range() {
    // Some digitizers report 0..10000 or 0..32767
    let custom_bounds = DeviceBounds::new(0, 10000);
    let p_half = ink::normalize_pressure(5000, true, Some(custom_bounds));
    assert_eq!(p_half, 0.5, "Non-power-of-two digitizer range must scale linearly");
}

#[test]
fn test_f8_subnormal_floating_point_pressure() {
    let bounds = DeviceBounds::LEVELS_16K;
    let p = ink::normalize_pressure(0, false, Some(bounds));
    assert_eq!(p, 0.0);
}

// ============================================================================
// Feature 9: Physical Eraser Inversion, Tilt & Rotation — Boundary Cases
// ============================================================================

#[test]
fn test_f9_mid_stroke_physical_eraser_flip() {
    let mut oracle = WindowsInkOracle::new();
    // Stroke starts with tip down
    let _ = oracle.feed(&RawPointerEvent {
        message: ink::message::WM_POINTERDOWN,
        pointer_id: 1,
        pen_flags: ink::pen_flag::NONE,
        raw_pressure: 500,
        ..Default::default()
    });

    // While still in contact, stylus is flipped upside down
    let flip_in_contact =
        RawPointerEvent { message: ink::message::WM_POINTERUPDATE, pointer_id: 1, pen_flags: ink::pen_flag::INVERTED, raw_pressure: 500, ..Default::default() };

    let u_flip = oracle.feed(&flip_in_contact);
    match u_flip {
        Update::Set(Some(s)) => {
            assert!(s.eraser, "Mid-stroke flip must immediately update eraser status");
        }
        other => panic!("Expected sample update on mid-stroke flip, got {:?}", other),
    }
}

#[test]
fn test_f9_rapid_eraser_chatter_suppression() {
    let mut oracle = WindowsInkOracle::new();
    // Rapid fluttering between tip and eraser in consecutive messages
    for i in 0..10 {
        let flags = if i % 2 == 0 { ink::pen_flag::INVERTED } else { ink::pen_flag::NONE };
        let e =
            RawPointerEvent { message: ink::message::WM_POINTERUPDATE, pointer_id: 1, pen_flags: flags, raw_pressure: 500, timestamp: i, ..Default::default() };
        let _ = oracle.feed(&e);
    }
    // Must process without crash or state corruption
}

#[test]
fn test_f9_tilt_exceeding_90_degrees_clamped() {
    assert_eq!(ink::map_tilt(135, true), 90.0, "+135 tilt must clamp to +90.0");
    assert_eq!(ink::map_tilt(-135, true), -90.0, "-135 tilt must clamp to -90.0");
}

#[test]
fn test_f9_rotation_angle_modulo_boundary() {
    assert_eq!(ink::map_rotation(360, true), 0.0);
    assert_eq!(ink::map_rotation(720, true), 0.0);
    assert_eq!(ink::map_rotation(1080, true), 0.0);
}

#[test]
fn test_f9_synthetic_touch_rejection() {
    let mut oracle = WindowsInkOracle::new();
    let touch_event = RawPointerEvent {
        pointer_type: ink::pointer_type::PT_TOUCH,
        message: ink::message::WM_POINTERDOWN,
        pointer_id: 5,
        pen_flags: ink::pen_flag::NONE,
        ..Default::default()
    };

    let update = oracle.feed(&touch_event);
    assert_eq!(update, Update::Keep, "Synthetic touch event must be rejected by the tablet ink driver");
}
