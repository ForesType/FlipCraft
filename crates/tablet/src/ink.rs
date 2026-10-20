//! Pure safe Windows Ink (`WM_POINTER`) event mapping and normalization.
//!
//! Maps raw pointer and pen data to [`Sample`]s with:
//! - Full 0..=1.0 linear pressure normalization for 1024, 8,192, and 16,384 levels
//!   via [`DeviceBounds`] (`logicalMax`), eliminating premature 1024 clamping.
//! - Tilt X and Tilt Y degrees (-90°..+90°).
//! - Barrel rotation degrees (0°..359°).
//! - Physical eraser flip detection (`PEN_FLAG_INVERTED` and `PEN_FLAG_ERASER`).
//! - High-frequency event rate compression and deduplication (per Krita's `Qt::AA_CompressTabletEvents`).
//!
//! Pure Rust without `unsafe`, thoroughly tested across all platforms.

use crate::{Sample, Update};

/// Pointer window messages (Win32).
pub mod message {
    pub const WM_POINTERUPDATE: u32 = 0x0245;
    pub const WM_POINTERDOWN: u32 = 0x0246;
    pub const WM_POINTERUP: u32 = 0x0247;
    pub const WM_POINTERENTER: u32 = 0x0249;
    pub const WM_POINTERLEAVE: u32 = 0x024A;
    pub const WM_POINTERACTIVATE: u32 = 0x024B;
    pub const WM_POINTERCAPTURECHANGED: u32 = 0x024C;
}

/// Pointer input types (`POINTER_INPUT_TYPE`).
pub mod pointer_type {
    pub const PT_POINTER: u32 = 1;
    pub const PT_TOUCH: u32 = 2;
    pub const PT_PEN: u32 = 3;
    pub const PT_MOUSE: u32 = 4;
    pub const PT_TOUCHPAD: u32 = 5;
}

/// Pointer flags (`POINTER_FLAGS`).
pub mod pointer_flag {
    pub const NONE: u32 = 0x00000000;
    pub const NEW: u32 = 0x00000001;
    pub const IN_RANGE: u32 = 0x00000002;
    pub const IN_CONTACT: u32 = 0x00000004;
    pub const FIRST_BUTTON: u32 = 0x00000010;
    pub const SECOND_BUTTON: u32 = 0x00000020;
    pub const PRIMARY: u32 = 0x00002000;
    pub const CONFIDENCE: u32 = 0x00004000;
    pub const CANCELLED: u32 = 0x00008000;
    pub const DOWN: u32 = 0x00010000;
    pub const UPDATE: u32 = 0x00020000;
    pub const UP: u32 = 0x00040000;
}

/// Pen flags (`PEN_FLAGS`).
pub mod pen_flag {
    pub const NONE: u32 = 0x00000000;
    pub const BARREL: u32 = 0x00000001;
    pub const INVERTED: u32 = 0x00000002;
    pub const ERASER: u32 = 0x00000004;
}

/// Pen mask (`PEN_MASK`).
pub mod pen_mask {
    pub const NONE: u32 = 0x00000000;
    pub const PRESSURE: u32 = 0x00000001;
    pub const ROTATION: u32 = 0x00000002;
    pub const TILT_X: u32 = 0x00000004;
    pub const TILT_Y: u32 = 0x00000008;
}

/// Hardware digitizer property range for tip pressure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceBounds {
    pub logical_min: i32,
    pub logical_max: i32,
}

impl DeviceBounds {
    /// Standard Windows 1024-level default.
    pub const DEFAULT: Self = Self { logical_min: 0, logical_max: 1024 };
    /// 8,192 physical hardware pressure levels (e.g. Wacom Pro Pen 2, Huion PW517).
    pub const LEVELS_8K: Self = Self { logical_min: 0, logical_max: 8191 };
    /// 16,384 physical hardware pressure levels (e.g. XP-Pen X3 Pro).
    pub const LEVELS_16K: Self = Self { logical_min: 0, logical_max: 16383 };

    /// Creates bounds with given minimum and maximum logical values.
    pub const fn new(logical_min: i32, logical_max: i32) -> Self {
        Self { logical_min, logical_max }
    }
}

/// Raw pointer and pen event fields, read as plain values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawPointerEvent {
    /// Window message ID (e.g., `WM_POINTERDOWN`, `WM_POINTERUPDATE`, `WM_POINTERUP`, `WM_POINTERLEAVE`).
    pub message: u32,
    /// System pointer ID.
    pub pointer_id: u32,
    /// Pointer input type (`PT_PEN`, `PT_TOUCH`, etc.).
    pub pointer_type: u32,
    /// Pointer flags (`POINTER_FLAGS`).
    pub pointer_flags: u32,
    /// Pen flags (`PEN_FLAGS`, e.g. `PEN_FLAG_INVERTED`, `PEN_FLAG_ERASER`).
    pub pen_flags: u32,
    /// Pen mask indicating which fields are valid (`PEN_MASK`).
    pub pen_mask: u32,
    /// Raw un-normalized tip pressure reported by digitizer.
    pub raw_pressure: u32,
    /// Tilt X angle in degrees (-90..=90).
    pub tilt_x: i32,
    /// Tilt Y angle in degrees (-90..=90).
    pub tilt_y: i32,
    /// Rotation in degrees (0..=359).
    pub rotation: u32,
    /// Pixel X location.
    pub x: i32,
    /// Pixel Y location.
    pub y: i32,
    /// Message timestamp in milliseconds (`dwTime`).
    pub timestamp: u32,
}

impl Default for RawPointerEvent {
    fn default() -> Self {
        Self {
            message: message::WM_POINTERUPDATE,
            pointer_id: 1,
            pointer_type: pointer_type::PT_PEN,
            pointer_flags: pointer_flag::IN_RANGE | pointer_flag::IN_CONTACT | pointer_flag::UPDATE,
            pen_flags: pen_flag::NONE,
            pen_mask: pen_mask::PRESSURE | pen_mask::TILT_X | pen_mask::TILT_Y | pen_mask::ROTATION,
            raw_pressure: 0,
            tilt_x: 0,
            tilt_y: 0,
            rotation: 0,
            x: 0,
            y: 0,
            timestamp: 0,
        }
    }
}

/// Linearly normalizes raw digitizer pressure to 0.0..=1.0.
///
/// If `bounds` specifies `logical_max > logical_min`, pressure is scaled precisely against
/// that hardware range. When bounds are omitted, it detects whether raw values exceed 1024
/// (e.g. 4096, 8192, 16384) and normalizes without premature saturation.
pub fn normalize_pressure(raw: u32, in_contact: bool, bounds: Option<DeviceBounds>) -> f32 {
    if !in_contact && raw == 0 {
        return 0.0;
    }

    if let Some(b) = bounds
        && b.logical_max > b.logical_min
    {
        let span = (b.logical_max - b.logical_min) as f32;
        let val = (raw as f32) - (b.logical_min as f32);
        return (val / span).clamp(0.0, 1.0);
    }

    // Auto-detect digitizer scale if no explicit bounds provided.
    if raw <= 1024 {
        (raw as f32 / 1024.0).clamp(0.0, 1.0)
    } else if raw <= 2048 {
        (raw as f32 / 2048.0).clamp(0.0, 1.0)
    } else if raw <= 4096 {
        (raw as f32 / 4096.0).clamp(0.0, 1.0)
    } else if raw <= 8192 {
        (raw as f32 / 8192.0).clamp(0.0, 1.0)
    } else if raw <= 16384 {
        (raw as f32 / 16384.0).clamp(0.0, 1.0)
    } else {
        (raw as f32 / 65535.0).clamp(0.0, 1.0)
    }
}

/// Maps raw tilt angle to -90.0..=90.0 degrees.
pub fn map_tilt(raw: i32, valid: bool) -> f32 {
    if valid { (raw as f32).clamp(-90.0, 90.0) } else { 0.0 }
}

/// Maps raw barrel rotation to 0.0..360.0 degrees.
pub fn map_rotation(raw: u32, valid: bool) -> f32 {
    if valid {
        let deg = (raw as f32).rem_euclid(360.0);
        if deg >= 360.0 { 0.0 } else { deg }
    } else {
        0.0
    }
}

/// Detects whether the pen is physically inverted or its eraser button is active.
pub fn is_eraser(pen_flags: u32) -> bool {
    (pen_flags & (pen_flag::INVERTED | pen_flag::ERASER)) != 0
}

/// Tracking state across Windows Ink pointer messages.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    /// True when the active pen end is inverted or eraser.
    pub eraser: bool,
    /// Active pointer ID currently tracked.
    pub active_pointer_id: Option<u32>,
    /// Optional default device bounds for pressure scaling.
    pub default_bounds: Option<DeviceBounds>,
    /// Whether high-frequency event compression is enabled.
    pub compression_enabled: bool,
    /// Minimum timestamp delta (ms) required between non-essential updates.
    pub min_interval_ms: u32,
    /// Distance threshold (px) for micro-jitter filtering during compression.
    pub jitter_threshold_px: i32,
    /// Maximum raw pressure value observed so far.
    pub max_seen_pressure: u32,
    /// Last processed raw event.
    last_event: Option<RawPointerEvent>,
    /// Last emitted sample.
    last_sample: Option<Sample>,
    /// Timestamp of last emitted sample.
    last_emit_time: u32,
    /// Previous trajectory displacement X for inflection detection.
    last_dx: i32,
    /// Previous trajectory displacement Y for inflection detection.
    last_dy: i32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            eraser: false,
            active_pointer_id: None,
            default_bounds: None,
            compression_enabled: true,
            min_interval_ms: 8, // Target 120Hz frame interval (8ms downsampling)
            jitter_threshold_px: 1,
            max_seen_pressure: 0,
            last_event: None,
            last_sample: None,
            last_emit_time: 0,
            last_dx: 0,
            last_dy: 0,
        }
    }
}

impl State {
    /// Creates a new state with default compression enabled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets explicit hardware bounds for digitizer pressure scaling.
    pub fn with_bounds(mut self, bounds: DeviceBounds) -> Self {
        self.default_bounds = Some(bounds);
        self
    }

    /// Toggles high-frequency event rate compression.
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression_enabled = enabled;
        self
    }

    /// Sets target rate compression interval in milliseconds (e.g. 8 for 120Hz, 16 for 60Hz).
    pub fn with_min_interval(mut self, interval_ms: u32) -> Self {
        self.min_interval_ms = interval_ms;
        self
    }

    /// Reset tracking state (e.g. on window focus lost).
    pub fn reset(&mut self) {
        self.eraser = false;
        self.active_pointer_id = None;
        self.last_event = None;
        self.last_sample = None;
        self.last_emit_time = 0;
        self.last_dx = 0;
        self.last_dy = 0;
    }

    /// Map a raw Windows Ink event to an [`Update`].
    pub fn handle(&mut self, e: &RawPointerEvent) -> Update {
        use message::*;
        use pointer_type::*;

        // Ignore non-pen pointer events (mouse/touch), except when they tell us pen has left.
        if e.pointer_type != PT_PEN {
            if self.active_pointer_id.is_some() && (e.pointer_type == PT_MOUSE || e.pointer_type == PT_TOUCH) {
                self.reset();
                return Update::Set(None);
            }
            return Update::Keep;
        }

        match e.message {
            WM_POINTERLEAVE => {
                if self.active_pointer_id == Some(e.pointer_id) || self.active_pointer_id.is_none() {
                    self.reset();
                    return Update::Set(None);
                }
                Update::Keep
            }
            WM_POINTERENTER => {
                self.active_pointer_id = Some(e.pointer_id);
                self.eraser = is_eraser(e.pen_flags);
                let sample = Sample {
                    pressure: 0.0,
                    tilt_x: map_tilt(e.tilt_x, (e.pen_mask & pen_mask::TILT_X) != 0),
                    tilt_y: map_tilt(e.tilt_y, (e.pen_mask & pen_mask::TILT_Y) != 0),
                    rotation: map_rotation(e.rotation, (e.pen_mask & pen_mask::ROTATION) != 0),
                    eraser: self.eraser,
                };
                self.last_sample = Some(sample);
                self.last_event = Some(*e);
                self.last_emit_time = e.timestamp;
                Update::Set(Some(sample))
            }
            WM_POINTERDOWN | WM_POINTERUPDATE | WM_POINTERUP => {
                self.active_pointer_id = Some(e.pointer_id);
                let eraser = is_eraser(e.pen_flags);
                self.eraser = eraser;

                if e.raw_pressure > self.max_seen_pressure {
                    self.max_seen_pressure = e.raw_pressure;
                    if self.default_bounds.is_none() && self.max_seen_pressure > 1024 {
                        let max_logical = if self.max_seen_pressure <= 2048 {
                            2048
                        } else if self.max_seen_pressure <= 4096 {
                            4096
                        } else if self.max_seen_pressure <= 8192 {
                            8191
                        } else if self.max_seen_pressure <= 16384 {
                            16383
                        } else {
                            65535
                        };
                        self.default_bounds = Some(DeviceBounds::new(0, max_logical));
                    }
                }

                let in_contact = (e.pointer_flags & pointer_flag::IN_CONTACT) != 0;
                let pressure = if e.message == WM_POINTERUP {
                    // Lifting pen reports 0 pressure.
                    0.0
                } else {
                    normalize_pressure(e.raw_pressure, in_contact, self.default_bounds)
                };

                let tilt_x = map_tilt(e.tilt_x, (e.pen_mask & pen_mask::TILT_X) != 0);
                let tilt_y = map_tilt(e.tilt_y, (e.pen_mask & pen_mask::TILT_Y) != 0);
                let rotation = map_rotation(e.rotation, (e.pen_mask & pen_mask::ROTATION) != 0);

                let sample = Sample { pressure, tilt_x, tilt_y, rotation, eraser };

                // Apply deduplication and rate compression strictly to intermediate updates.
                // DOWN and UP events MUST ALWAYS be delivered immediately.
                if e.message == WM_POINTERUPDATE {
                    // 1. Deduplication: drop exact duplicate packets.
                    if let Some(ref last) = self.last_event
                        && last.x == e.x
                        && last.y == e.y
                        && last.raw_pressure == e.raw_pressure
                        && last.tilt_x == e.tilt_x
                        && last.tilt_y == e.tilt_y
                        && last.rotation == e.rotation
                        && last.pen_flags == e.pen_flags
                    {
                        return Update::Keep;
                    }

                    // 2. Rate compression: downsample high-frequency streams (120Hz-240Hz)
                    // while strictly preserving endpoints, tool switches, pressure spikes,
                    // and trajectory inflection points.
                    if self.compression_enabled
                        && let (Some(last_s), Some(last_e)) = (self.last_sample, self.last_event)
                    {
                        let dt = e.timestamp.saturating_sub(self.last_emit_time);
                        let curr_dx = e.x - last_e.x;
                        let curr_dy = e.y - last_e.y;
                        let dp = (sample.pressure - last_s.pressure).abs();

                        // Eraser / tool flip must never be dropped.
                        let is_eraser_flip = sample.eraser != last_s.eraser;

                        // Pressure spike / extremum must never be dropped.
                        let is_pressure_spike = dp >= 0.05;

                        // Trajectory inflection point (>45 degree corner) must never be dropped.
                        let curr_len_sq = (curr_dx as i64).pow(2) + (curr_dy as i64).pow(2);
                        let prev_len_sq = (self.last_dx as i64).pow(2) + (self.last_dy as i64).pow(2);
                        let is_inflection = if curr_len_sq >= 4 && prev_len_sq >= 4 {
                            let dot = (self.last_dx as i64) * (curr_dx as i64) + (self.last_dy as i64) * (curr_dy as i64);
                            dot <= 0 || (dot * dot * 2 < prev_len_sq * curr_len_sq)
                        } else {
                            false
                        };

                        // Stationary micro-jitter clamp: identical or sub-pixel trembling (never clamp tool flips).
                        let is_micro_jitter =
                            !is_eraser_flip && curr_dx.abs() < self.jitter_threshold_px && curr_dy.abs() < self.jitter_threshold_px && dp < 0.005;

                        if !is_eraser_flip && (is_micro_jitter || (dt < self.min_interval_ms && !is_pressure_spike && !is_inflection)) {
                            if curr_dx * curr_dx + curr_dy * curr_dy >= 4 {
                                self.last_dx = curr_dx;
                                self.last_dy = curr_dy;
                            }
                            self.last_event = Some(*e);
                            return Update::Keep;
                        }
                    }
                }

                // Update trajectory displacement history for non-trivial movements (>= 2px).
                if let Some(last_e) = self.last_event {
                    let curr_dx = e.x - last_e.x;
                    let curr_dy = e.y - last_e.y;
                    if curr_dx * curr_dx + curr_dy * curr_dy >= 4 {
                        self.last_dx = curr_dx;
                        self.last_dy = curr_dy;
                    }
                }

                self.last_sample = Some(sample);
                self.last_event = Some(*e);
                self.last_emit_time = e.timestamp;
                Update::Set(Some(sample))
            }
            _ => Update::Keep,
        }
    }
}

/// Batch event compression and deduplication per Krita's `Qt::AA_CompressTabletEvents`.
///
/// Preserves boundary events (`DOWN`, `UP`, `LEAVE`) and compresses intermediate high-rate
/// updates, discarding duplicates and micro-jitter within consecutive packets.
pub fn compress_events(events: &[RawPointerEvent]) -> Vec<RawPointerEvent> {
    if events.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(events.len());
    let mut last_emitted_time = 0u32;
    let mut last_update: Option<RawPointerEvent> = None;

    for &ev in events {
        match ev.message {
            message::WM_POINTERDOWN | message::WM_POINTERUP | message::WM_POINTERLEAVE | message::WM_POINTERENTER => {
                if let Some(pending) = last_update.take() {
                    out.push(pending);
                }
                out.push(ev);
                last_emitted_time = ev.timestamp;
            }
            message::WM_POINTERUPDATE => {
                if let Some(prev) = last_update {
                    // Exact duplicate check
                    let duplicate = prev.x == ev.x
                        && prev.y == ev.y
                        && prev.raw_pressure == ev.raw_pressure
                        && prev.tilt_x == ev.tilt_x
                        && prev.tilt_y == ev.tilt_y
                        && prev.rotation == ev.rotation
                        && prev.pen_flags == ev.pen_flags;

                    if !duplicate {
                        let dt = ev.timestamp.saturating_sub(last_emitted_time);
                        let dp = (ev.raw_pressure as i32 - prev.raw_pressure as i32).abs();
                        let is_eraser_flip =
                            (ev.pen_flags & (pen_flag::INVERTED | pen_flag::ERASER)) != (prev.pen_flags & (pen_flag::INVERTED | pen_flag::ERASER));

                        // Coalesce intermediate updates within 8ms unless pressure spike or eraser flip
                        if dt < 8 && dp < 100 && !is_eraser_flip {
                            last_update = Some(ev);
                        } else {
                            out.push(prev);
                            last_emitted_time = prev.timestamp;
                            last_update = Some(ev);
                        }
                    }
                } else {
                    last_update = Some(ev);
                }
            }
            _ => out.push(ev),
        }
    }

    if let Some(pending) = last_update {
        out.push(pending);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_scaling_1024_linear() {
        let b = DeviceBounds::DEFAULT;
        assert_eq!(normalize_pressure(0, false, Some(b)), 0.0);
        assert_eq!(normalize_pressure(0, true, Some(b)), 0.0);
        assert!((normalize_pressure(512, true, Some(b)) - 0.5).abs() < 1e-4);
        assert!((normalize_pressure(1024, true, Some(b)) - 1.0).abs() < 1e-4);
        assert_eq!(normalize_pressure(1025, true, Some(b)), 1.0); // Clamped at 1.0
    }

    #[test]
    fn pressure_scaling_8192_linear_no_premature_saturation() {
        let b = DeviceBounds::LEVELS_8K;
        // On an 8k tablet, raw value 1024 is ~12.5% pressure, NEVER 100%!
        let at_1024 = normalize_pressure(1024, true, Some(b));
        assert!((at_1024 - (1024.0 / 8191.0)).abs() < 1e-4, "1024 on 8k must not saturate to 1.0; got {at_1024}");
        assert!(at_1024 < 0.15, "1024 on 8k must be around 0.125, got {at_1024}");

        let at_4096 = normalize_pressure(4096, true, Some(b));
        assert!((at_4096 - 0.5).abs() < 0.01);

        let at_8191 = normalize_pressure(8191, true, Some(b));
        assert!((at_8191 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn pressure_scaling_16384_linear_no_premature_saturation() {
        let b = DeviceBounds::LEVELS_16K;
        // On a 16k tablet, raw value 1024 is ~6.25% pressure!
        let at_1024 = normalize_pressure(1024, true, Some(b));
        assert!(at_1024 < 0.07, "1024 on 16k must be ~0.0625, got {at_1024}");

        let at_8192 = normalize_pressure(8192, true, Some(b));
        assert!((at_8192 - 0.5).abs() < 0.01);

        let at_16383 = normalize_pressure(16383, true, Some(b));
        assert!((at_16383 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn auto_detection_without_bounds_does_not_clamp_8k_or_16k() {
        // When device bounds are absent, raw values > 1024 auto-detect high-res digitizer.
        let at_4096 = normalize_pressure(4096, true, None);
        assert!((at_4096 - 1.0).abs() < 1e-4); // 4096 level full scale

        let at_8192 = normalize_pressure(8192, true, None);
        assert!((at_8192 - 1.0).abs() < 1e-4); // 8192 level full scale

        let at_16384 = normalize_pressure(16384, true, None);
        assert!((at_16384 - 1.0).abs() < 1e-4); // 16384 level full scale
    }

    #[test]
    fn state_auto_detects_digitizer_bounds_on_unbounded_devices() {
        let mut st = State::new(); // No explicit bounds
        // Digitizer sends a high value (e.g. 8191)
        let ev1 = RawPointerEvent { message: message::WM_POINTERDOWN, raw_pressure: 8191, ..Default::default() };
        if let Update::Set(Some(s)) = st.handle(&ev1) {
            assert!((s.pressure - 1.0).abs() < 1e-4);
        }
        // Next event sends 2048 (should be ~0.25 on 8k scale)
        let ev2 = RawPointerEvent { message: message::WM_POINTERUPDATE, raw_pressure: 2048, timestamp: 10, ..Default::default() };
        if let Update::Set(Some(s)) = st.handle(&ev2) {
            assert!((s.pressure - (2048.0 / 8191.0)).abs() < 1e-3, "pressure on 8k was {}", s.pressure);
        }
    }

    #[test]
    fn tilt_and_rotation_mapping() {
        assert_eq!(map_tilt(45, true), 45.0);
        assert_eq!(map_tilt(-60, true), -60.0);
        assert_eq!(map_tilt(120, true), 90.0, "clamped to +90");
        assert_eq!(map_tilt(-100, true), -90.0, "clamped to -90");
        assert_eq!(map_tilt(45, false), 0.0, "ignored if mask invalid");

        assert_eq!(map_rotation(180, true), 180.0);
        assert_eq!(map_rotation(359, true), 359.0);
        assert_eq!(map_rotation(360, true), 0.0);
        assert_eq!(map_rotation(725, true), 5.0);
        assert_eq!(map_rotation(180, false), 0.0, "ignored if mask invalid");
    }

    #[test]
    fn eraser_inversion_and_button_detection() {
        assert!(!is_eraser(pen_flag::NONE));
        assert!(!is_eraser(pen_flag::BARREL));
        assert!(is_eraser(pen_flag::INVERTED));
        assert!(is_eraser(pen_flag::ERASER));
        assert!(is_eraser(pen_flag::INVERTED | pen_flag::ERASER));
    }

    #[test]
    fn state_tracks_pen_stroke_and_eraser_flip() {
        let mut st = State::new().with_bounds(DeviceBounds::LEVELS_8K);

        // Hover enter
        let enter = RawPointerEvent { message: message::WM_POINTERENTER, pen_flags: pen_flag::NONE, ..Default::default() };
        assert_eq!(st.handle(&enter), Update::Set(Some(Sample { pressure: 0.0, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, eraser: false })));

        // Pen down with tip
        let down = RawPointerEvent {
            message: message::WM_POINTERDOWN,
            raw_pressure: 4096,
            tilt_x: 30,
            tilt_y: -15,
            rotation: 120,
            pen_flags: pen_flag::NONE,
            ..Default::default()
        };
        if let Update::Set(Some(s)) = st.handle(&down) {
            assert!((s.pressure - 0.5).abs() < 0.01);
            assert_eq!(s.tilt_x, 30.0);
            assert_eq!(s.tilt_y, -15.0);
            assert_eq!(s.rotation, 120.0);
            assert!(!s.eraser);
        } else {
            panic!("expected Update::Set");
        }

        // Pen up
        let up = RawPointerEvent { message: message::WM_POINTERUP, pen_flags: pen_flag::NONE, ..Default::default() };
        if let Update::Set(Some(s)) = st.handle(&up) {
            assert_eq!(s.pressure, 0.0);
            assert!(!s.eraser);
        } else {
            panic!("expected Update::Set");
        }

        // Invert pen to eraser and press down
        let eraser_down = RawPointerEvent { message: message::WM_POINTERDOWN, raw_pressure: 8191, pen_flags: pen_flag::INVERTED, ..Default::default() };
        if let Update::Set(Some(s)) = st.handle(&eraser_down) {
            assert_eq!(s.pressure, 1.0);
            assert!(s.eraser, "eraser must be detected on INVERTED");
        } else {
            panic!("expected Update::Set");
        }

        // Leave window
        let leave = RawPointerEvent { message: message::WM_POINTERLEAVE, ..Default::default() };
        assert_eq!(st.handle(&leave), Update::Set(None));
        assert!(!st.eraser);
    }

    #[test]
    fn deduplication_drops_identical_intermediate_updates() {
        let mut st = State::new();
        let ev1 = RawPointerEvent { message: message::WM_POINTERUPDATE, raw_pressure: 500, x: 100, y: 200, timestamp: 10, ..Default::default() };
        assert!(matches!(st.handle(&ev1), Update::Set(Some(_))));

        // Same exact event
        let ev2 = ev1;
        assert_eq!(st.handle(&ev2), Update::Keep, "exact duplicate update must be dropped");

        // Event with coordinate change is accepted after rate interval (>= 8ms)
        let ev3 = RawPointerEvent { x: 105, timestamp: 20, ..ev1 };
        assert!(matches!(st.handle(&ev3), Update::Set(Some(_))));
    }

    #[test]
    fn batch_event_compression_preserves_boundaries() {
        let ev_down = RawPointerEvent { message: message::WM_POINTERDOWN, x: 10, y: 10, timestamp: 100, ..Default::default() };
        let ev_u1 = RawPointerEvent { message: message::WM_POINTERUPDATE, x: 11, y: 10, timestamp: 101, ..Default::default() };
        let ev_u2 = RawPointerEvent {
            // Duplicate of u1
            message: message::WM_POINTERUPDATE,
            x: 11,
            y: 10,
            timestamp: 101,
            ..Default::default()
        };
        let ev_u3 = RawPointerEvent {
            // Sub-millisecond cluster move
            message: message::WM_POINTERUPDATE,
            x: 11,
            y: 11,
            timestamp: 101,
            ..Default::default()
        };
        let ev_u4 = RawPointerEvent {
            // Distinct move
            message: message::WM_POINTERUPDATE,
            x: 20,
            y: 20,
            timestamp: 105,
            ..Default::default()
        };
        let ev_up = RawPointerEvent { message: message::WM_POINTERUP, x: 20, y: 20, timestamp: 110, ..Default::default() };

        let raw = vec![ev_down, ev_u1, ev_u2, ev_u3, ev_u4, ev_up];
        let compressed = compress_events(&raw);

        // Duplicates and sub-millisecond cluster coalesced; DOWN and UP preserved.
        assert_eq!(compressed.first().map(|e| e.message), Some(message::WM_POINTERDOWN));
        assert_eq!(compressed.last().map(|e| e.message), Some(message::WM_POINTERUP));
        assert!(compressed.len() < raw.len(), "compressed length {} must be less than raw {}", compressed.len(), raw.len());
    }
}
