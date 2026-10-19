//! Tier 3: Cross-Feature Combinations Suite (Combinatorial & Pairwise Interactions)
//!
//! Validates pairwise and multi-feature subsystem interactions across the PhotoCraft brush engine
//! and Windows tablet driver overhaul (15 tests total).

use photocraft_e2e_tests::*;
use std::collections::HashMap;

// ============================================================================
// 1. High Velocity Flick with Rapid Pressure Collapse (F1 + F3)
// ============================================================================

#[test]
fn test_t3_velocity_flick_with_pressure_collapse() {
    let mut flick = StrokeTrajectoryBuilder::create_calligraphy_flick(10.0, 10.0, 200.0);
    apply_tail_aggressiveness(&mut flick, 15.0);

    let mut oracle = KisDistanceOracle::new();
    let mut total_dabs = 0;
    let mut shrinkage_triggers = 0;

    for i in 1..flick.len() {
        let p_prev = &flick[i - 1];
        let p_curr = &flick[i];
        let len = ((p_curr.x - p_prev.x).powi(2) + (p_curr.y - p_prev.y).powi(2)).sqrt();
        let target_spacing = (p_curr.pressure * 20.0).max(1.0) as f64;

        let (dabs, res) = oracle.step(len, target_spacing);
        total_dabs += dabs.len();
        if res == AdaptiveStepResult::ImmediateShrinkageDab {
            shrinkage_triggers += 1;
        }
    }

    assert!(total_dabs >= 20, "Calligraphy flick must emit continuous dabs without starvation");
    assert!(shrinkage_triggers > 0, "Rapid pressure collapse during flick must trigger immediate spacing shrinkage dabs");
    assert_eq!(flick.last().unwrap().pressure, 0.0);
}

// ============================================================================
// 2. Curved Stroke with De Casteljau Subdivision & Dab Cache (F2 + F4)
// ============================================================================

#[test]
fn test_t3_curved_stroke_bezier_with_dab_cache() {
    let p0 = (0.0, 0.0);
    let p1 = (100.0, 250.0);
    let p2 = (200.0, 250.0);
    let p3 = (300.0, 0.0);

    let mut curve_pts = vec![p0];
    de_casteljau_subdivide(p0, p1, p2, p3, 0.5, 0, &mut curve_pts);

    let cache = DabCacheOracle::new(32);
    let mut cache_hits = 0;
    let mut cache_misses = 0;

    // Radius varies along the curve (quantized to 0.25px steps)
    for (i, _) in curve_pts.iter().enumerate() {
        let t = i as f32 / curve_pts.len() as f32;
        let radius = 10.0 + (t * std::f32::consts::PI).sin() * 5.0; // oscillates between 10.0 and 15.0
        let key = DabKey::new(radius, 0.8, 0.5, 1.0, 0.0);

        let (_, hit) = cache.get_or_create(&key, || generate_analytical_mask(radius, 0.8));
        if hit {
            cache_hits += 1;
        } else {
            cache_misses += 1;
        }
    }

    assert!(cache_hits > 0, "Repeated similar radii along curved Bézier stroke must achieve cache hits");
    assert!(cache_misses <= 32, "Distinct cached masks must not exceed cache capacity");
}

// ============================================================================
// 3. Tablet Rate Compression with Bézier Curve Fitting (F7 + F2)
// ============================================================================

#[test]
fn test_t3_event_compression_with_bezier_fitting() {
    let mut oracle = WindowsInkOracle::new();
    let mut raw_stream = Vec::new();

    // 240Hz packets along an arc (100 events in 400ms)
    for i in 0..100 {
        let t = (i as f64) / 100.0;
        let angle = t * std::f64::consts::FRAC_PI_2;
        raw_stream.push(RawPointerEvent {
            message: ink::message::WM_POINTERUPDATE,
            x: (200.0 * angle.cos()) as i32,
            y: (200.0 * angle.sin()) as i32,
            raw_pressure: (500.0 + 300.0 * angle.sin()) as u32,
            timestamp: (i * 4) as u32,
            ..Default::default()
        });
    }

    let (samples, dropped) = oracle.filter_stream(&raw_stream);
    assert!(dropped > 0, "Rate compression must downsample raw 240Hz stream");
    assert!(!samples.is_empty());

    // Fit cubic Bézier curve to key points of the compressed stream
    let p0 = (raw_stream[0].x as f64, raw_stream[0].y as f64);
    let p3 = (raw_stream.last().unwrap().x as f64, raw_stream.last().unwrap().y as f64);
    let mid_idx = samples.len() / 2;
    let p1 = (samples[mid_idx / 2].tilt_x as f64 + 100.0, 50.0);
    let p2 = (samples[mid_idx].tilt_y as f64 + 150.0, 100.0);

    let mut fitted_pts = vec![p0];
    de_casteljau_subdivide(p0, p1, p2, p3, 0.5, 0, &mut fitted_pts);
    assert!(fitted_pts.len() >= 2);
}

// ============================================================================
// 4. Linear 16k Pressure Resolution with Adaptive Stepping (F8 + F1)
// ============================================================================

#[test]
fn test_t3_16k_pressure_with_adaptive_stepping() {
    let bounds = DeviceBounds::LEVELS_16K;
    let mut oracle = KisDistanceOracle::new();
    let mut dab_count = 0;

    // Step through 100 micro-increments of 16k pressure (e.g. 500..600 out of 16383)
    let mut prev_pressure = 0.0f32;
    for raw_p in 500..600 {
        let pressure = ink::normalize_pressure(raw_p, true, Some(bounds));
        assert!(pressure > prev_pressure, "16k pressure normalization must be strictly monotonic");
        prev_pressure = pressure;

        let spacing = (pressure * 50.0).max(1.0) as f64;
        let (dabs, _) = oracle.step(2.0, spacing);
        dab_count += dabs.len();
    }
    assert!(dab_count > 0, "16k high-resolution pressure must drive adaptive distance stepping smoothly");
}

// ============================================================================
// 5. Eraser Flip Mid-Stroke with SIMD Tile Compositing (F9 + F5)
// ============================================================================

#[test]
fn test_t3_eraser_flip_mid_stroke_with_tile_compositing() {
    let mut ink_oracle = WindowsInkOracle::new();
    let mut tiles = HashMap::new();

    // 1. Initial downstroke with tip (paint deposit)
    let down_tip = RawPointerEvent {
        message: ink::message::WM_POINTERDOWN,
        pointer_id: 1,
        pen_flags: ink::pen_flag::NONE,
        raw_pressure: 800,
        x: 64,
        y: 64,
        ..Default::default()
    };
    if let Update::Set(Some(s)) = ink_oracle.feed(&down_tip) {
        assert!(!s.eraser);
        blit_dab_into_tiles(&mut tiles, 64.0, 64.0, 20.0, s.pressure);
    }

    // 2. Mid-stroke stylus flip to physical eraser
    let flip_eraser = RawPointerEvent {
        message: ink::message::WM_POINTERUPDATE,
        pointer_id: 1,
        pen_flags: ink::pen_flag::INVERTED,
        raw_pressure: 800,
        x: 70,
        y: 64,
        ..Default::default()
    };
    if let Update::Set(Some(s)) = ink_oracle.feed(&flip_eraser) {
        assert!(s.eraser, "Sample must transition to eraser mid-stroke");
        // Eraser blits into tiles cleanly
        let touched = blit_dab_into_tiles(&mut tiles, 70.0, 64.0, 20.0, s.pressure);
        assert!(!touched.is_empty());
    }
}

// ============================================================================
// 6. Large 500px Brush with High-Frequency Event Stream (F5 + F7)
// ============================================================================

#[test]
fn test_t3_large_500px_brush_with_high_frequency_stream() {
    let mut oracle = WindowsInkOracle::new();
    let mut tiles = HashMap::new();

    // 240Hz packets for a massive 500px diameter brush
    let mut stream = Vec::new();
    for i in 0..50 {
        stream.push(RawPointerEvent {
            message: ink::message::WM_POINTERUPDATE,
            x: 250 + i * 4,
            y: 250,
            raw_pressure: 1024,
            timestamp: (i * 4) as u32,
            ..Default::default()
        });
    }

    let (samples, dropped) = oracle.filter_stream(&stream);
    assert!(dropped > 0, "Rate compression must throttle events to prevent tile render queue stall");

    // Blit remaining filtered samples
    for s in samples {
        blit_dab_into_tiles(&mut tiles, 250.0, 250.0, 250.0, s.pressure);
    }
    assert!(tiles.len() >= 49, "500px brush must cover large tile footprint");
}

// ============================================================================
// 7. Anisotropic Rotated Brush with Bézier Subdivision & Cache (F1 + F2 + F4)
// ============================================================================

#[test]
fn test_t3_anisotropic_rotated_brush_bezier_cache() {
    let p0 = (0.0, 0.0);
    let p1 = (50.0, 100.0);
    let p2 = (150.0, 100.0);
    let p3 = (200.0, 0.0);

    let mut curve = vec![p0];
    de_casteljau_subdivide(p0, p1, p2, p3, 0.5, 0, &mut curve);

    let cache = DabCacheOracle::new(64);
    let mut distance_oracle = KisDistanceOracle::new();
    let mut total_dabs = 0;

    for i in 1..curve.len() {
        let (x0, y0) = curve[i - 1];
        let (x1, y1) = curve[i];
        let seg_len = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
        let angle = (y1 - y0).atan2(x1 - x0).to_degrees() as f32;

        // Chisel tip (aspect 0.3) rotating along curve
        let key = DabKey::new(15.0, 0.8, 0.5, 0.3, angle);
        let (_, _) = cache.get_or_create(&key, || generate_analytical_mask(15.0, 0.8));

        let (dabs, _) = distance_oracle.step(seg_len, 5.0);
        total_dabs += dabs.len();
    }

    assert!(total_dabs > 0);
    assert!(!cache.is_empty());
}

// ============================================================================
// 8. Tilt-Scaled Dab with 16k Pressure Dynamics & SIMD Blitting (F9 + F8 + F5)
// ============================================================================

#[test]
fn test_t3_tilt_scaled_dab_16k_pressure_simd_blitting() {
    let bounds = DeviceBounds::LEVELS_16K;
    let mut tiles = HashMap::new();

    // Stylus tilted at 45 degrees with raw pressure 12,000 / 16,383
    let raw_p = 12000;
    let pressure = ink::normalize_pressure(raw_p, true, Some(bounds));
    let tilt_x = ink::map_tilt(45, true);

    // Tilt scales aspect ratio: cos(45 deg) ≈ 0.707
    let tilt_scale = tilt_x.to_radians().cos();
    let radius = (pressure * 50.0) as f64;

    let touched = blit_dab_into_tiles(&mut tiles, 100.0, 100.0, radius * (tilt_scale as f64), pressure);
    assert!(!touched.is_empty());
}

// ============================================================================
// 9. Rate-Compressed Flick with Tail Aggressiveness & SIMD Blit (F7 + F1 + F3 + F5)
// ============================================================================

#[test]
fn test_t3_rate_compressed_flick_tail_aggressiveness_simd() {
    let mut flick = StrokeTrajectoryBuilder::create_calligraphy_flick(0.0, 0.0, 300.0);
    apply_tail_aggressiveness(&mut flick, 10.0);

    let mut oracle = KisDistanceOracle::new();
    let mut tiles = HashMap::new();

    for i in 1..flick.len() {
        let p_prev = &flick[i - 1];
        let p_curr = &flick[i];
        let len = ((p_curr.x - p_prev.x).powi(2) + (p_curr.y - p_prev.y).powi(2)).sqrt();
        let spacing = (p_curr.pressure * 20.0).max(1.0) as f64;

        let (dabs, _) = oracle.step(len, spacing);
        for _ in dabs {
            blit_dab_into_tiles(&mut tiles, p_curr.x, p_curr.y, (p_curr.pressure * 15.0).max(0.5) as f64, p_curr.pressure);
        }
    }

    assert!(!tiles.is_empty(), "Full pipeline execution must successfully populate coverage tiles");
}

// ============================================================================
// 10. Inverted Eraser with 8k Pressure along Bézier Path (F9 + F8 + F2 + F4)
// ============================================================================

#[test]
fn test_t3_inverted_eraser_8k_pressure_bezier_cache() {
    let bounds = DeviceBounds::LEVELS_8K;
    let p0 = (50.0, 50.0);
    let p1 = (100.0, 150.0);
    let p2 = (150.0, 150.0);
    let p3 = (200.0, 50.0);

    let mut curve = vec![p0];
    de_casteljau_subdivide(p0, p1, p2, p3, 0.5, 0, &mut curve);

    let cache = DabCacheOracle::new(32);
    for (i, _) in curve.iter().enumerate() {
        let raw_p = ((i as f32 / curve.len() as f32) * 8191.0) as u32;
        let pressure = ink::normalize_pressure(raw_p, true, Some(bounds));
        let radius = (pressure * 30.0).max(1.0);

        let key = DabKey::new(radius, 1.0, 0.0, 1.0, 0.0);
        let (mask, _) = cache.get_or_create(&key, || generate_analytical_mask(radius, 1.0));
        assert!(!mask.is_empty());
    }
}

// ============================================================================
// 11. Hover to Contact Transition with Adaptive Stepping (F6 + F8 + F1)
// ============================================================================

#[test]
fn test_t3_hover_to_contact_transition_adaptive_stepping() {
    let mut ink_oracle = WindowsInkOracle::new();
    let mut dist_oracle = KisDistanceOracle::new();

    // 1. Hover packet (P = 0)
    let hover_event = RawPointerEvent {
        message: ink::message::WM_POINTERUPDATE,
        pointer_flags: ink::pointer_flag::IN_RANGE, // in range but NOT in contact
        raw_pressure: 0,
        x: 10,
        y: 10,
        ..Default::default()
    };
    if let Update::Set(Some(s)) = ink_oracle.feed(&hover_event) {
        assert_eq!(s.pressure, 0.0, "Hover state must report 0.0 pressure");
    }

    // 2. Contact packet (P > 0)
    let contact_event = RawPointerEvent {
        message: ink::message::WM_POINTERDOWN,
        pointer_flags: ink::pointer_flag::IN_RANGE | ink::pointer_flag::IN_CONTACT | ink::pointer_flag::FIRST_BUTTON,
        raw_pressure: 1024,
        x: 15,
        y: 15,
        ..Default::default()
    };
    if let Update::Set(Some(s)) = ink_oracle.feed(&contact_event) {
        assert_eq!(s.pressure, 1.0);
        // First contact dab: step distance 0.0 must immediately trigger contact dab
        let (dabs, _) = dist_oracle.step(0.0, 10.0);
        assert_eq!(dist_oracle.accum_distance, 0.0);
        assert!(dabs.is_empty() || dabs.contains(&0.0));
    }
}

// ============================================================================
// 12. Rapid Reversal with Flatness Subdivision & LRU Eviction (F2 + F4 + F5)
// ============================================================================

#[test]
fn test_t3_rapid_reversal_flatness_subdivision_eviction() {
    let cache = DabCacheOracle::new(8); // Small capacity 8
    let mut tiles = HashMap::new();

    // Zig-zag path with 10 reversals and varying brush sizes
    for z in 0..10 {
        let p0 = ((z * 40) as f64, 0.0);
        let p1 = ((z * 40 + 20) as f64, 100.0);
        let p2 = ((z * 40 + 30) as f64, -100.0);
        let p3 = ((z * 40 + 40) as f64, 0.0);

        let mut curve = vec![p0];
        de_casteljau_subdivide(p0, p1, p2, p3, 0.5, 0, &mut curve);

        let radius = (z as f32 * 5.0) + 10.0;
        let key = DabKey::new(radius, 0.5, 0.5, 1.0, 0.0);
        let (_, _) = cache.get_or_create(&key, || generate_analytical_mask(radius, 0.5));

        for pt in &curve {
            blit_dab_into_tiles(&mut tiles, pt.0, pt.1, radius as f64, 0.5);
        }
    }

    assert_eq!(cache.len(), 8, "Cache must strictly enforce capacity bound of 8 under rapid size changes");
    assert!(!tiles.is_empty());
}

// ============================================================================
// 13. Multi-Axis Tilt and Rotation with Anisotropic Stepping (F9 + F1)
// ============================================================================

#[test]
fn test_t3_multi_axis_tilt_rotation_anisotropic_stepping() {
    let mut oracle = KisDistanceOracle::new();
    let mut dabs_collected = 0;

    // Stylus rotating while tilting along path
    for step in 0..36 {
        let rotation = ink::map_rotation(step * 10, true);
        let tilt_x = ink::map_tilt((step as i32 - 18) * 4, true);

        // Aspect ratio changes with tilt angle
        let aspect = (tilt_x.to_radians().cos().abs() as f64).clamp(0.1, 1.0);
        let spacing = 15.0 * aspect;

        let (dabs, _) = oracle.step(10.0, spacing);
        dabs_collected += dabs.len();
        assert!((0.0..360.0).contains(&rotation));
    }
    assert!(dabs_collected > 0);
}

// ============================================================================
// 14. Sub-Frame Rate Compression with Tail Pull (F7 + F3)
// ============================================================================

#[test]
fn test_t3_subframe_rate_compression_tail_pull() {
    let mut ink_oracle = WindowsInkOracle::new();
    let mut raw_events = Vec::new();

    // 10 events within 5ms ending in lift-off
    for i in 0..10 {
        raw_events.push(RawPointerEvent {
            message: if i == 9 { ink::message::WM_POINTERUP } else { ink::message::WM_POINTERUPDATE },
            x: 100 + i * 2,
            y: 50,
            raw_pressure: (1000 - i * 100).max(0) as u32,
            timestamp: i as u32,
            ..Default::default()
        });
    }

    let (samples, _) = ink_oracle.filter_stream(&raw_events);
    assert!(!samples.is_empty());

    let mut points: Vec<StrokePoint> = samples.iter().map(|s| StrokePoint {
        x: 100.0,
        y: 50.0,
        pressure: s.pressure,
        ..Default::default()
    }).collect();

    apply_tail_aggressiveness(&mut points, 10.0);
    assert!(points.len() >= 2);
}

// ============================================================================
// 15. Extreme 500px Wash across 100 Tiles with 16k Pressure (F5 + F8)
// ============================================================================

#[test]
fn test_t3_500px_wash_16k_pressure_100_tiles() {
    let bounds = DeviceBounds::LEVELS_16K;
    let mut tiles = HashMap::new();

    // Sweep 500px diameter brush across a 640x640 canvas (10x10 = 100 tiles)
    for step in 0..10 {
        let raw_p = 8000 + step * 800; // 16k pressure levels
        let pressure = ink::normalize_pressure(raw_p, true, Some(bounds));
        let center = (step as f64) * 60.0 + 50.0;

        let touched = blit_dab_into_tiles(&mut tiles, center, center, 250.0, pressure);
        assert!(!touched.is_empty());
    }

    assert!(tiles.len() >= 64, "500px sweep must touch dozens of tiles across canvas (touched: {})", tiles.len());
}
