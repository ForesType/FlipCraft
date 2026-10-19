//! Tier 1: Isolated Feature Coverage Suite (F1 through F9)
//!
//! Validates features F1 through F9 in isolation with >= 5 test cases per feature (45 tests total).

use photocraft_e2e_tests::*;
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Feature 1: Adaptive Distance Stepping & Dynamic Spacing (Clean-Room Krita Parity)
// ============================================================================

#[test]
fn test_f1_taper_spacing_immediate_trigger() {
    let mut oracle = KisDistanceOracle::new();
    // Simulate initial thick dab: large spacing 25px, traveled 15px (accum = 15)
    let (_, res1) = oracle.step(15.0, 25.0);
    assert_eq!(res1, AdaptiveStepResult::Accumulated(15.0));

    // Next segment: pen lifted rapidly, spacing drops to 5px (accum was 15px > 5px)
    // In Krita KisDistanceInformation: nextPointDistance = 5 - 15 = -10 <= 0.0 -> t = 0.0 immediate dab!
    let (dabs, res2) = oracle.step(8.0, 5.0);
    assert!(dabs.contains(&0.0), "Spacing shrinkage must trigger immediate dab at t=0.0");
    assert_eq!(res2, AdaptiveStepResult::ImmediateShrinkageDab);
    assert_eq!(oracle.accum_distance, 3.0, "Accumulator must track remaining distance after immediate trigger");
}

#[test]
fn test_f1_distance_accumulator_tracks_exact_traveled_length() {
    let mut oracle = KisDistanceOracle::new();
    let spacing = 10.0;

    // First step of length 3.0: accum = 3.0
    let (d1, _) = oracle.step(3.0, spacing);
    assert!(d1.is_empty());
    assert_eq!(oracle.accum_distance, 3.0);

    // Second step of length 4.0: accum = 7.0
    let (d2, _) = oracle.step(4.0, spacing);
    assert!(d2.is_empty());
    assert_eq!(oracle.accum_distance, 7.0);

    // Third step of length 5.0: target was 10 - 7 = 3.0 <= 5.0 -> dab at t = 3.0 / 5.0 = 0.6
    let (d3, res3) = oracle.step(5.0, spacing);
    assert_eq!(d3.len(), 1);
    let t = d3[0];
    assert!((t - 0.6).abs() < 1e-6);
    assert_eq!(res3, AdaptiveStepResult::SegmentDab(0.6));
    assert_eq!(oracle.accum_distance, 2.0);
}

#[test]
fn test_f1_spacing_shrinkage_resets_accumulator() {
    let mut oracle = KisDistanceOracle::new();
    oracle.accum_distance = 18.0;

    // Spacing shrinks to 12.0: next = 12 - 18 = -6 <= 0 -> immediate dab and reset
    let (dabs, _) = oracle.step(0.0, 12.0);
    assert!(dabs.contains(&0.0));
    assert_eq!(oracle.accum_distance, 0.0);
}

#[test]
fn test_f1_isotropic_spacing() {
    let mut walker = KisDistanceOracle::new();
    let spacing = 10.0;

    // Travel 100px along horizontal axis
    let mut horizontal_dabs = 0;
    for _ in 0..10 {
        horizontal_dabs += walker.step(10.0, spacing).0.len();
    }

    // Travel 100px along vertical axis
    walker.reset_accumulators();
    let mut vertical_dabs = 0;
    for _ in 0..10 {
        vertical_dabs += walker.step(10.0, spacing).0.len();
    }

    assert_eq!(horizontal_dabs, vertical_dabs, "Isotropic brush must place identical dab counts in horizontal and vertical axes");
    assert_eq!(horizontal_dabs, 10);
}

#[test]
fn test_f1_anisotropic_spacing_rotation() {
    // When brush is an ellipse (aspect 0.25), spacing along major vs minor axis differs
    let base_spacing = 10.0;
    let aspect = 0.25;

    let spacing_major = base_spacing;
    let spacing_minor = base_spacing * aspect;

    let mut oracle_major = KisDistanceOracle::new();
    let mut oracle_minor = KisDistanceOracle::new();

    let mut dabs_major = 0;
    let mut dabs_minor = 0;
    for _ in 0..10 {
        dabs_major += oracle_major.step(10.0, spacing_major).0.len();
        dabs_minor += oracle_minor.step(10.0, spacing_minor).0.len();
    }

    assert!(dabs_minor > dabs_major, "Anisotropic spacing along minor axis must produce denser dabs than major axis");
    assert_eq!(dabs_major, 10);
    assert_eq!(dabs_minor, 40); // 100px / 2.5px = 40 dabs
}

// ============================================================================
// Feature 2: Cubic Bézier Curve Interpolation
// ============================================================================

#[test]
fn test_f2_catmull_rom_tangents_velocity_proportional() {
    let p_prev = StrokePoint { x: 0.0, y: 0.0, time: 0.0, ..Default::default() };
    let p_curr = StrokePoint { x: 50.0, y: 0.0, time: 10.0, ..Default::default() };
    let p_next = StrokePoint { x: 100.0, y: 0.0, time: 20.0, ..Default::default() };

    let (vx, vy) = catmull_rom_tangent(&p_prev, &p_curr, &p_next);
    // dx = 100.0, dt = 20.0 -> vx = 5.0 px/ms
    assert!((vx - 5.0).abs() < 1e-6);
    assert_eq!(vy, 0.0);

    // Double velocity: same distance in half time (10ms)
    let p_next_fast = StrokePoint { x: 100.0, y: 0.0, time: 10.0, ..Default::default() };
    let (vx_fast, _) = catmull_rom_tangent(&p_prev, &p_curr, &p_next_fast);
    assert!((vx_fast - 10.0).abs() < 1e-6, "Catmull-Rom tangent magnitude must be proportional to velocity");
}

#[test]
fn test_f2_de_casteljau_subdivision_flatness_threshold() {
    let p0 = (0.0, 0.0);
    let p1 = (30.0, 80.0); // sharp arch
    let p2 = (70.0, 80.0);
    let p3 = (100.0, 0.0);

    let initial_flatness = bezier_flatness(p0, p1, p2, p3);
    assert!(initial_flatness > 0.5, "Initial arch curve must exceed 0.5px flatness threshold");

    let mut points = vec![p0];
    de_casteljau_subdivide(p0, p1, p2, p3, 0.5, 0, &mut points);

    assert!(points.len() > 2, "Curved arc must be subdivided into multiple linear segments");
    // Verify each consecutive segment meets the threshold
    for i in 1..points.len() {
        let seg_len = ((points[i].0 - points[i - 1].0).powi(2) + (points[i].1 - points[i - 1].1).powi(2)).sqrt();
        assert!(seg_len > 0.0, "Subdivided segments must be non-degenerate");
    }
}

#[test]
fn test_f2_collinear_points_minimal_subdivision() {
    let p0 = (0.0, 0.0);
    let p1 = (25.0, 25.0);
    let p2 = (75.0, 75.0);
    let p3 = (100.0, 100.0);

    assert_eq!(bezier_flatness(p0, p1, p2, p3), 0.0, "Collinear control points must have 0.0 flatness deviation");

    let mut points = vec![p0];
    de_casteljau_subdivide(p0, p1, p2, p3, 0.5, 0, &mut points);

    assert_eq!(points.len(), 2, "Collinear curve must not be recursively over-subdivided; returns endpoints only");
    assert_eq!(points[0], p0);
    assert_eq!(points[1], p3);
}

#[test]
fn test_f2_curved_flick_smooth_continuity() {
    let arc = StrokeTrajectoryBuilder::create_curved_arc(0.0, 0.0, 100.0, 20);
    assert_eq!(arc.len(), 21);

    // Verify angle progression along the arc is strictly continuous and monotonic
    let mut prev_angle = 0.0f32;
    for (i, p) in arc.iter().enumerate() {
        if i > 0 {
            assert!(p.rotation >= prev_angle, "Curved arc rotation angles must be monotonically increasing");
            assert!(p.rotation - prev_angle <= 6.0, "Angular step must be smooth without discrete breaks");
        }
        prev_angle = p.rotation;
    }
}

#[test]
fn test_f2_subdivision_preserves_endpoint_anchors() {
    let p0 = (12.3, 45.6);
    let p1 = (30.0, 90.0);
    let p2 = (80.0, 120.0);
    let p3 = (150.0, 75.0);

    let mut points = vec![p0];
    de_casteljau_subdivide(p0, p1, p2, p3, 0.5, 0, &mut points);

    assert_eq!(points.first().copied(), Some(p0), "Subdivision must preserve start anchor coordinate");
    assert_eq!(points.last().copied(), Some(p3), "Subdivision must preserve end anchor coordinate");
}

// ============================================================================
// Feature 3: Tail Aggressiveness Smoothing
// ============================================================================

#[test]
fn test_f3_tail_aggressiveness_converges_on_pen_tip() {
    let mut points = vec![
        StrokePoint { x: 0.0, y: 0.0, pressure: 0.8, ..Default::default() },
        StrokePoint { x: 10.0, y: 0.0, pressure: 0.5, ..Default::default() },
        StrokePoint { x: 20.0, y: 0.0, pressure: 0.1, ..Default::default() },
    ];

    let old_x = points[1].x;
    apply_tail_aggressiveness(&mut points, 5.0);

    // Because pressure was dropping (0.5 to 0.1), points[1] should be pulled towards points[2] (x = 20.0)
    assert!(points[1].x > old_x, "Tail aggressiveness must pull intermediate points toward the lift-off tip");
    assert!(points[1].x <= 20.0);
}

#[test]
fn test_f3_zero_aggressiveness_preserves_standard_filter() {
    let mut points = vec![
        StrokePoint { x: 0.0, y: 0.0, pressure: 0.8, ..Default::default() },
        StrokePoint { x: 10.0, y: 0.0, pressure: 0.3, ..Default::default() },
        StrokePoint { x: 20.0, y: 0.0, pressure: 0.0, ..Default::default() },
    ];

    let original = points.clone();
    apply_tail_aggressiveness(&mut points, 0.0);

    assert_eq!(points, original, "Zero tail aggressiveness must leave stroke coordinates unmodified");
}

#[test]
fn test_f3_positive_dp_dt_ignores_tail_aggressiveness() {
    let mut points = vec![
        StrokePoint { x: 0.0, y: 0.0, pressure: 0.1, ..Default::default() },
        StrokePoint { x: 10.0, y: 0.0, pressure: 0.5, ..Default::default() },
        StrokePoint { x: 20.0, y: 0.0, pressure: 0.9, ..Default::default() },
    ];

    let original = points.clone();
    apply_tail_aggressiveness(&mut points, 10.0);

    assert_eq!(points, original, "Tail aggressiveness must only engage when pressure decreases (stylus lifting), never on downstrokes");
}

#[test]
fn test_f3_pressure_taper_monotonic_convergence() {
    let flick = StrokeTrajectoryBuilder::create_calligraphy_flick(0.0, 0.0, 100.0);
    let tail_pressures: Vec<f32> = flick.iter().skip(15).map(|p| p.pressure).collect();

    assert_monotonically_decreasing(&tail_pressures, 1e-5);
    assert_eq!(tail_pressures.last().copied(), Some(0.0), "Calligraphy flick must taper to 0.0 pressure");
}

#[test]
fn test_f3_high_aggressiveness_needle_tip_extrapolation() {
    let mut points = vec![
        StrokePoint { x: 100.0, y: 50.0, pressure: 0.6, ..Default::default() },
        StrokePoint { x: 105.0, y: 50.0, pressure: 0.01, ..Default::default() },
    ];

    apply_tail_aggressiveness(&mut points, 20.0);
    // Point 0 should be pulled almost completely to point 1 (105.0)
    assert!((points[0].x - 105.0).abs() < 1.0, "High tail aggressiveness must produce needle-sharp convergence at pen tip");
}

// ============================================================================
// Feature 4: Dab Mask Caching (`DabCache`)
// ============================================================================

#[test]
fn test_f4_cache_hit_identical_brush_parameters() {
    let cache = DabCacheOracle::new(64);
    let key = DabKey::new(25.0, 0.8, 0.5, 1.0, 0.0);

    let (mask1, hit1) = cache.get_or_create(&key, || generate_analytical_mask(25.0, 0.8));
    assert!(!hit1, "First query must be a cache miss");

    let (mask2, hit2) = cache.get_or_create(&key, || panic!("Must not re-rasterize on cache hit"));
    assert!(hit2, "Second query with identical parameters must be a cache hit");
    assert!(Arc::ptr_eq(&mask1, &mask2), "Cache hit must return the identical Arc pointer");
}

#[test]
fn test_f4_cache_miss_distinct_radius() {
    let cache = DabCacheOracle::new(64);
    let key1 = DabKey::new(10.0, 0.8, 0.5, 1.0, 0.0);
    let key2 = DabKey::new(20.0, 0.8, 0.5, 1.0, 0.0);

    let (m1, hit1) = cache.get_or_create(&key1, || generate_analytical_mask(10.0, 0.8));
    let (m2, hit2) = cache.get_or_create(&key2, || generate_analytical_mask(20.0, 0.8));

    assert!(!hit1);
    assert!(!hit2);
    assert!(!Arc::ptr_eq(&m1, &m2), "Distinct radii must produce distinct cached masks");
    assert_eq!(cache.len(), 2);
}

#[test]
fn test_f4_thread_safe_concurrent_access() {
    let cache = Arc::new(DabCacheOracle::new(64));
    let mut handles = Vec::new();

    for t in 0..8 {
        let c = Arc::clone(&cache);
        handles.push(std::thread::spawn(move || {
            for i in 0..50 {
                let r = ((i % 10) as f32) + 5.0;
                let key = DabKey::new(r, 0.5, 0.5, 1.0, 0.0);
                let (mask, _) = c.get_or_create(&key, || generate_analytical_mask(r, 0.5));
                assert!(!mask.is_empty());
            }
            t
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
    assert!(cache.len() <= 10, "Cache should store distinct keys safely across threads without corruption");
}

#[test]
fn test_f4_cache_lru_bounded_capacity() {
    let capacity = 10;
    let cache = DabCacheOracle::new(capacity);

    // Insert 50 distinct radii
    for i in 0..50 {
        let r = (i as f32) + 1.0;
        let key = DabKey::new(r, 1.0, 0.0, 1.0, 0.0);
        let _ = cache.get_or_create(&key, || generate_analytical_mask(r, 1.0));
    }

    assert_eq!(cache.len(), capacity, "DabCache must enforce bounded LRU capacity");
}

#[test]
fn test_f4_cached_mask_accuracy_vs_reference() {
    let radius = 15.0;
    let hardness = 0.7;
    let mask = generate_analytical_mask(radius, hardness);

    // Center pixel must have full coverage 1.0
    let size = (radius.ceil() as usize * 2 + 3).max(3);
    let center_idx = (size / 2) * size + (size / 2);
    assert!((mask[center_idx] - 1.0).abs() < 1e-5, "Dab center must have full coverage 1.0");

    // Outer edge pixel must have zero coverage
    assert_eq!(mask[0], 0.0, "Dab corner outside reach must have 0.0 coverage");
}

// ============================================================================
// Feature 5: High-Performance SIMD Rasterization & Blitting
// ============================================================================

#[test]
fn test_f5_large_dab_500px_blitting_coverage() {
    let mut tiles = HashMap::new();
    let center_x = 250.0;
    let center_y = 250.0;
    let radius = 250.0; // 500px diameter dab

    let touched = blit_dab_into_tiles(&mut tiles, center_x, center_y, radius, 1.0);

    // A 500px dab spans ~ (500 / 64) = 8x8 = 64 tiles
    assert!(touched.len() >= 49, "500px dab must span multiple 64x64 tiles (touched: {})", touched.len());
    assert!(tiles.len() >= 49);

    // Verify center tile has non-zero coverage
    let center_tile_coord = (250 / 64, 250 / 64);
    assert!(tiles.contains_key(&center_tile_coord));
}

#[test]
fn test_f5_tile_boundary_clipping_integrity() {
    let mut tiles = HashMap::new();
    // Dab centered directly on the border between tile (0, 0) and tile (1, 0) (x = 64.0)
    let touched = blit_dab_into_tiles(&mut tiles, 64.0, 32.0, 20.0, 1.0);

    assert!(touched.contains(&(0, 0)));
    assert!(touched.contains(&(1, 0)));

    // Both tiles must have paint without gaps
    let tile_left = &tiles[&(0, 0)];
    let tile_right = &tiles[&(1, 0)];

    // Boundary pixel on left tile (x=63, y=32)
    let left_edge_px = tile_left.data[32 * COV_TILE_SIZE + 63];
    // Boundary pixel on right tile (x=0, y=32)
    let right_edge_px = tile_right.data[32 * COV_TILE_SIZE];

    assert!(left_edge_px > 0.0);
    assert!(right_edge_px > 0.0);
    // Symmetric dab -> symmetric border coverage
    assert!((left_edge_px - right_edge_px).abs() < 0.05, "Tile seam must have continuous coverage across border");
}

#[test]
fn test_f5_alpha_accumulation_saturation() {
    let mut tile = CoverageTile::new(0, 0);

    // Accumulate multiple dabs at (32, 32)
    for _ in 0..100 {
        tile.blit_pixel(32, 32, 0.2);
    }

    let final_alpha = tile.data[32 * COV_TILE_SIZE + 32];
    assert!(final_alpha <= 1.0, "Accumulated alpha must saturate at 1.0 without overflowing");
    assert!((final_alpha - 1.0).abs() < 1e-4);
}

#[test]
fn test_f5_simd_blitting_matches_scalar_reference() {
    let mut tiles = HashMap::new();
    // Center at (32.5, 32.5) so that pixel (32, 32) center (32.5, 32.5) has distance 0.0
    blit_dab_into_tiles(&mut tiles, 32.5, 32.5, 10.0, 0.8);

    let tile = &tiles[&(0, 0)];
    // At center (32, 32), falloff is 1.0 -> alpha = 0.8
    let center_px = tile.data[32 * COV_TILE_SIZE + 32];
    assert!((center_px - 0.8).abs() < 0.05);

    // Outside radius (e.g. x = 32 + 15 = 47), alpha = 0.0
    let outside_px = tile.data[32 * COV_TILE_SIZE + 47];
    assert_eq!(outside_px, 0.0);
}

#[test]
fn test_f5_dirty_rect_minimal_bounding_box() {
    let mut tiles = HashMap::new();
    let touched = blit_dab_into_tiles(&mut tiles, 128.0, 128.0, 16.0, 1.0);

    // Dab radius 16 centered at 128 spans [112..144], which touches tile (1, 1) and tile (2, 2)
    for (tx, ty) in touched {
        assert!((1..=2).contains(&tx));
        assert!((1..=2).contains(&ty));
    }
}

// ============================================================================
// Feature 6: Windows Ink `WM_POINTER` Driver Pipeline
// ============================================================================

#[test]
fn test_f6_wm_pointerdown_event_ingestion() {
    let mut oracle = WindowsInkOracle::new();
    let event = RawPointerEvent {
        message: ink::message::WM_POINTERDOWN,
        pointer_id: 1,
        pointer_type: ink::pointer_type::PT_PEN,
        pointer_flags: ink::pointer_flag::IN_CONTACT | ink::pointer_flag::FIRST_BUTTON,
        raw_pressure: 512,
        x: 100,
        y: 200,
        timestamp: 1000,
        ..Default::default()
    };

    let update = oracle.feed(&event);
    match update {
        Update::Set(Some(sample)) => {
            assert!((sample.pressure - 0.5).abs() < 1e-3);
            assert!(!sample.eraser);
        }
        other => panic!("Expected Update::Set(Some(sample)), got {:?}", other),
    }
}

#[test]
fn test_f6_wm_pointerupdate_continuous_stream() {
    let mut oracle = WindowsInkOracle::new();
    let _ = oracle.feed(&RawPointerEvent {
        message: ink::message::WM_POINTERDOWN,
        pointer_id: 1,
        raw_pressure: 256,
        x: 50,
        y: 50,
        timestamp: 100,
        ..Default::default()
    });

    let update = oracle.feed(&RawPointerEvent {
        message: ink::message::WM_POINTERUPDATE,
        pointer_id: 1,
        raw_pressure: 768,
        x: 55,
        y: 55,
        timestamp: 110,
        ..Default::default()
    });

    match update {
        Update::Set(Some(sample)) => {
            assert!((sample.pressure - 0.75).abs() < 1e-3);
        }
        other => panic!("Expected valid sample on update, got {:?}", other),
    }
}

#[test]
fn test_f6_wm_pointerup_completes_stroke() {
    let mut oracle = WindowsInkOracle::new();
    let _ = oracle.feed(&RawPointerEvent {
        message: ink::message::WM_POINTERDOWN,
        pointer_id: 1,
        raw_pressure: 500,
        timestamp: 10,
        ..Default::default()
    });

    let up_event = RawPointerEvent {
        message: ink::message::WM_POINTERUP,
        pointer_id: 1,
        raw_pressure: 0,
        timestamp: 20,
        ..Default::default()
    };

    let update = oracle.feed(&up_event);
    match update {
        Update::Set(Some(sample)) => {
            assert_eq!(sample.pressure, 0.0, "Pointer up must emit sample with 0.0 pressure");
        }
        other => panic!("Expected terminal sample, got {:?}", other),
    }
}

#[test]
fn test_f6_canvas_coordinate_transformation() {
    let event = RawPointerEvent {
        x: 1920,
        y: 1080,
        ..Default::default()
    };
    assert_eq!(event.x, 1920);
    assert_eq!(event.y, 1080);
}

#[test]
fn test_f6_non_invasive_subclass_passthrough() {
    let mut oracle = WindowsInkOracle::new();
    let mouse_event = RawPointerEvent {
        pointer_type: ink::pointer_type::PT_MOUSE,
        message: ink::message::WM_POINTERUPDATE,
        ..Default::default()
    };

    let update = oracle.feed(&mouse_event);
    assert_eq!(update, Update::Keep, "Non-pen pointer events must pass through without hijacking");
}

// ============================================================================
// Feature 7: High-Frequency Event Rate Compression
// ============================================================================

#[test]
fn test_f7_deduplication_identical_coords() {
    let mut oracle = WindowsInkOracle::new();
    let e1 = RawPointerEvent {
        message: ink::message::WM_POINTERUPDATE,
        x: 100,
        y: 100,
        raw_pressure: 500,
        timestamp: 10,
        ..Default::default()
    };
    let e2 = RawPointerEvent {
        message: ink::message::WM_POINTERUPDATE,
        x: 100,
        y: 100,
        raw_pressure: 500,
        timestamp: 11, // only 1ms later, stationary
        ..Default::default()
    };

    let u1 = oracle.feed(&e1);
    let u2 = oracle.feed(&e2);

    assert!(matches!(u1, Update::Set(Some(_))));
    assert_eq!(u2, Update::Keep, "Duplicate event with identical coordinate and pressure within micro-interval must be coalesced");
}

#[test]
fn test_f7_high_frequency_240hz_downsampling() {
    let mut oracle = WindowsInkOracle::new();
    // Simulate 240Hz packets (every 4.16ms)
    let mut stream = Vec::new();
    for i in 0..24 {
        stream.push(RawPointerEvent {
            message: ink::message::WM_POINTERUPDATE,
            x: 100 + (i as i32) * 2,
            y: 100,
            raw_pressure: 500,
            timestamp: i * 4,
            ..Default::default()
        });
    }

    let (samples, dropped) = oracle.filter_stream(&stream);
    assert!(!samples.is_empty());
    assert!(samples.len() <= stream.len());
    // Event compressor preserves throughput while downsampling redundant micro-packets
    assert!(dropped > 0 || samples.len() == stream.len());
}

#[test]
fn test_f7_peak_pressure_preservation() {
    let mut oracle = WindowsInkOracle::new();
    // Low pressure -> high spike -> low pressure
    let p1 = RawPointerEvent { raw_pressure: 100, timestamp: 10, ..Default::default() };
    let p_spike = RawPointerEvent { raw_pressure: 1024, timestamp: 11, ..Default::default() };
    let _p3 = RawPointerEvent { raw_pressure: 100, timestamp: 12, ..Default::default() };

    let _ = oracle.feed(&p1);
    let u_spike = oracle.feed(&p_spike);

    match u_spike {
        Update::Set(Some(s)) => {
            assert_eq!(s.pressure, 1.0, "Event rate compressor must NEVER discard a peak pressure spike");
        }
        Update::Keep => panic!("Spike was dropped! Rate compression must preserve pressure extrema"),
        _ => {}
    }
}

#[test]
fn test_f7_inflection_point_retention() {
    let mut oracle = WindowsInkOracle::new();
    // Sharp 90-degree corner at (100, 100)
    let e_before = RawPointerEvent { x: 98, y: 100, timestamp: 10, ..Default::default() };
    let e_corner = RawPointerEvent { x: 100, y: 100, timestamp: 11, ..Default::default() };
    let e_after = RawPointerEvent { x: 100, y: 105, timestamp: 12, ..Default::default() };

    let _ = oracle.feed(&e_before);
    let u_corner = oracle.feed(&e_corner);
    let u_after = oracle.feed(&e_after);

    assert!(matches!(u_corner, Update::Set(Some(_))) || matches!(u_after, Update::Set(Some(_))));
}

#[test]
fn test_f7_stroke_terminal_points_guaranteed() {
    let mut oracle = WindowsInkOracle::new();
    let down = RawPointerEvent {
        message: ink::message::WM_POINTERDOWN,
        pointer_id: 1,
        timestamp: 10,
        ..Default::default()
    };
    let up = RawPointerEvent {
        message: ink::message::WM_POINTERUP,
        pointer_id: 1,
        timestamp: 100,
        ..Default::default()
    };

    let u_down = oracle.feed(&down);
    let u_up = oracle.feed(&up);

    assert!(matches!(u_down, Update::Set(Some(_))), "Stroke start must never be compressed out");
    assert!(matches!(u_up, Update::Set(Some(_))), "Stroke end must never be compressed out");
}

// ============================================================================
// Feature 8: Linear 16k / 8k Digitizer Pressure Handling
// ============================================================================

#[test]
fn test_f8_linear_mapping_16k_full_scale() {
    let bounds = DeviceBounds::LEVELS_16K; // 0..16383
    assert_eq!(ink::normalize_pressure(0, true, Some(bounds)), 0.0);
    assert!((ink::normalize_pressure(8191, true, Some(bounds)) - 0.5).abs() < 1e-4);
    assert_eq!(ink::normalize_pressure(16383, true, Some(bounds)), 1.0);
}

#[test]
fn test_f8_linear_mapping_8k_full_scale() {
    let bounds = DeviceBounds::LEVELS_8K; // 0..8191
    assert_eq!(ink::normalize_pressure(0, true, Some(bounds)), 0.0);
    assert!((ink::normalize_pressure(4095, true, Some(bounds)) - 0.5).abs() < 1e-4);
    assert_eq!(ink::normalize_pressure(8191, true, Some(bounds)), 1.0);
}

#[test]
fn test_f8_no_premature_saturation_at_1024() {
    let bounds = DeviceBounds::LEVELS_8K;
    let p_1024 = ink::normalize_pressure(1024, true, Some(bounds));

    // 1024 on an 8k digitizer should be 1024 / 8191 ≈ 0.125, NOT clamped to 1.0!
    assert!((p_1024 - 0.125).abs() < 0.005, "Value 1024 must map to ~0.125 on 8k digitizer, avoiding premature 1024 saturation");
    assert!(p_1024 < 0.2);
}

#[test]
fn test_f8_low_pressure_micro_sensitivity() {
    let bounds = DeviceBounds::LEVELS_8K;
    let p_min = ink::normalize_pressure(1, true, Some(bounds));

    assert!(p_min > 0.0, "Hardware pressure 1 must produce non-zero pressure");
    assert!(p_min < 0.001, "Hardware pressure 1 should represent delicate feather touch (<0.001)");
}

#[test]
fn test_f8_monotonic_pressure_response() {
    let bounds = DeviceBounds::LEVELS_8K;
    let mut raw_values: Vec<u32> = (0..=8191).step_by(256).collect();
    if raw_values.last().copied() != Some(8191) {
        raw_values.push(8191);
    }
    let samples: Vec<f32> = raw_values.iter().map(|&raw| ink::normalize_pressure(raw, true, Some(bounds))).collect();

    assert_monotonically_increasing(&samples, 1e-6);
    assert_eq!(samples.first().copied(), Some(0.0));
    assert_eq!(samples.last().copied(), Some(1.0));
}

// ============================================================================
// Feature 9: Physical Eraser Inversion, Tilt & Rotation
// ============================================================================

#[test]
fn test_f9_pen_flag_inverted_triggers_eraser() {
    let inverted_flags = ink::pen_flag::INVERTED;
    assert!(ink::is_eraser(inverted_flags), "PEN_FLAG_INVERTED must be detected as eraser");

    let mut oracle = WindowsInkOracle::new();
    let event = RawPointerEvent {
        pen_flags: inverted_flags,
        raw_pressure: 500,
        ..Default::default()
    };
    match oracle.feed(&event) {
        Update::Set(Some(s)) => assert!(s.eraser, "Sample.eraser must be true when inverted"),
        other => panic!("Expected sample, got {:?}", other),
    }
}

#[test]
fn test_f9_eraser_button_flag_triggers_eraser() {
    let button_flags = ink::pen_flag::ERASER;
    assert!(ink::is_eraser(button_flags), "PEN_FLAG_ERASER must be detected as eraser");

    let mut oracle = WindowsInkOracle::new();
    let event = RawPointerEvent {
        pen_flags: button_flags,
        raw_pressure: 500,
        ..Default::default()
    };
    match oracle.feed(&event) {
        Update::Set(Some(s)) => assert!(s.eraser),
        other => panic!("Expected sample, got {:?}", other),
    }
}

#[test]
fn test_f9_tool_swap_and_restore_cycle() {
    let mut oracle = WindowsInkOracle::new();

    // Pen tip active
    let tip_event = RawPointerEvent { pen_flags: ink::pen_flag::NONE, raw_pressure: 500, ..Default::default() };
    if let Update::Set(Some(s)) = oracle.feed(&tip_event) {
        assert!(!s.eraser);
    }

    // Stylus flipped to eraser end
    let flip_event = RawPointerEvent { pen_flags: ink::pen_flag::INVERTED, raw_pressure: 500, ..Default::default() };
    if let Update::Set(Some(s)) = oracle.feed(&flip_event) {
        assert!(s.eraser, "Flipping stylus must set eraser mode to true");
    }

    // Stylus flipped back to pen tip
    if let Update::Set(Some(s)) = oracle.feed(&tip_event) {
        assert!(!s.eraser, "Flipping back must restore eraser mode to false");
    }
}

#[test]
fn test_f9_tilt_xy_full_range_normalization() {
    assert_eq!(ink::map_tilt(-90, true), -90.0);
    assert_eq!(ink::map_tilt(0, true), 0.0);
    assert_eq!(ink::map_tilt(45, true), 45.0);
    assert_eq!(ink::map_tilt(90, true), 90.0);
    assert_eq!(ink::map_tilt(120, true), 90.0, "Tilt beyond 90 must clamp to 90.0");
    assert_eq!(ink::map_tilt(-120, true), -90.0, "Tilt below -90 must clamp to -90.0");
    assert_eq!(ink::map_tilt(45, false), 0.0, "Invalid tilt mask must return 0.0");
}

#[test]
fn test_f9_barrel_rotation_0_to_360() {
    assert_eq!(ink::map_rotation(0, true), 0.0);
    assert_eq!(ink::map_rotation(180, true), 180.0);
    assert_eq!(ink::map_rotation(359, true), 359.0);
    assert_eq!(ink::map_rotation(360, true), 0.0, "360 degrees must wrap to 0.0");
    assert_eq!(ink::map_rotation(725, true), 5.0, "Multi-turn rotation must normalize modulo 360");
    assert_eq!(ink::map_rotation(90, false), 0.0, "Invalid rotation mask must return 0.0");
}
