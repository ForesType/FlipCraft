//! Dab rasterisation and stroke compositing.
//!
//! A stroke paints into a sparse, tiled *stroke buffer* (coverage plus, with Color Dynamics, a
//! premultiplied colour), exactly like a private stroke layer: dabs build up with flow up to their
//! opacity ceiling, stroke-level masks (Dual Brush, Texture) apply when the buffer is composited,
//! and the whole buffer composites onto the pre-stroke pixels at the stroke opacity. Because the
//! buffer is sparse and compositing always starts from the pre-stroke pixels, strokes can be fed in
//! chunks and re-composited incrementally for interactive painting (see [`StrokeRenderer`]).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use photocraft_color::{BlendMode, ColorMode, PixelFormat, SampleType};
use photocraft_geom::Rect;
use photocraft_raster::{Surface, from_rgba_into, to_rgba};

use crate::brush::{BrushSettings, MaskMode, Pattern, TipShape};
use crate::dynamics::DabGenerator;
use crate::retouch::{Footprint, alpha_index, over_native};
use crate::rng::hash2;
use crate::tile::{Mips, PatternImage};
use crate::{Dab, StrokePoint, dab_coverage, tip_falloff, tip_reach};

/// Salt of the 8-bit dither's per-pixel noise.
const DITHER_SALT: u64 = 0x0000_6469_7468_6572;

/// Before a stroke's result is quantised to 8 bits, offset every channel of the pixel by the same
/// deterministic noise of under half a level, so fractional values round up or down in proportion
/// (a soft, low-flow stroke's fine gradient survives as grain instead of bands). A value already on
/// an 8-bit level still rounds back to itself, and because the noise depends only on the pixel's
/// position, the live preview, the commit and a replay all agree.
#[inline]
fn dither8(px: &mut [f32], x: i32, y: i32) {
    let d = (hash2(x, y, DITHER_SALT) - 0.5) / 255.0;
    for v in px.iter_mut() {
        *v += d;
    }
}

/// Edge of a stroke-buffer tile.
pub const COV_TILE: i32 = 64;
/// Computed tips with a smaller radius (in pixels) are rasterised by area, not at pixel centres.
const SMALL_TIP_RADIUS: f32 = 3.0;
/// Sub-pixel sample offsets (a 4×4 grid) for the area coverage of small tips.
const SUBPIXEL: [f32; 4] = [-0.375, -0.125, 0.125, 0.375];
const NOISE_SALT: u64 = 0x006E_6F69_7365;

#[inline]
/// Where an aliased (Pencil) dab of `diameter` pixels centred near (`x`, `y`) lands on the pixel
/// grid: the centre of the pixel holding the point for odd diameters, the nearest pixel corner
/// for even ones. Its footprint is then a whole-pixel block around that point, the same square
/// the Pencil cursor shows ([`grid_square`]).
pub fn grid_center(x: f64, y: f64, diameter: f32) -> (f32, f32) {
    let n = if diameter.is_finite() { diameter.round().max(1.0) as i64 } else { 1 };
    let snap = |v: f64| if n % 2 == 1 { v.floor() + 0.5 } else { v.round() };
    (snap(x) as f32, snap(y) as f32)
}

/// The whole-pixel square `[x0, y0, x1, y1]` an aliased dab of `diameter` pixels at (`x`, `y`)
/// can touch (see [`grid_center`]): the Pencil's cursor.
pub fn grid_square(x: f64, y: f64, diameter: f32) -> [f64; 4] {
    let n = if diameter.is_finite() { f64::from(diameter.round().clamp(1.0, 100_000.0)) } else { 1.0 };
    let (cx, cy) = grid_center(x, y, n as f32);
    let h = n / 2.0;
    [f64::from(cx) - h, f64::from(cy) - h, f64::from(cx) + h, f64::from(cy) + h]
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Wet Edges: how much paint stays at a pixel whose nearest dab reaches it at `rn` (0 at a dab's
/// centre, 1 at its rim). The interior keeps half the paint, the rim all of it.
#[inline]
fn wet_factor(rn: f32) -> f32 {
    0.5 + 0.5 * smoothstep(0.5, 1.0, rn)
}

/// Combine brush coverage `v` with a mask value `t` (texture/dual tip) at `depth`.
/// Every mode maps `v = 0` to 0 (a mask never adds paint where the brush has none).
pub fn mask_combine(mode: MaskMode, v: f32, t: f32, d: f32) -> f32 {
    if v <= 0.0 {
        return 0.0;
    }
    let lerp = |a: f32, b: f32| a + (b - a) * d;
    let r = match mode {
        MaskMode::Multiply => v * (1.0 - d + d * t),
        MaskMode::Subtract => (v - d * (1.0 - t)).max(0.0),
        MaskMode::Darken => v.min(1.0 - d + d * t),
        MaskMode::Overlay => {
            let o = if v < 0.5 { 2.0 * v * t } else { 1.0 - 2.0 * (1.0 - v) * (1.0 - t) };
            lerp(v, o)
        }
        MaskMode::ColorDodge => lerp(v, (v / (1.0 - t).max(1e-3)).min(1.0)),
        MaskMode::ColorBurn => lerp(v, 1.0 - ((1.0 - v) / t.max(1e-3)).min(1.0)),
        MaskMode::LinearBurn => lerp(v, (v + t - 1.0).max(0.0)),
        MaskMode::HardMix => lerp(v, if v + t > 1.0 { 1.0 } else { 0.0 }),
        MaskMode::LinearHeight => v * ((t - (1.0 - d * v)) * 4.0 + 0.5).clamp(0.0, 1.0),
        MaskMode::Height => {
            if t >= 1.0 - d * v {
                v
            } else {
                0.0
            }
        }
    };
    r.clamp(0.0, 1.0)
}

/// How far a Height-mode dab at full depth and coverage reaches into the pattern, in pattern
/// ranges (measured against Photoshop, #3340).
const HEIGHT_REACH: f32 = 12.0;

/// Combine brush coverage `v` with a Texture pattern value `t` at `depth` (Brush Settings ›
/// Texture). Where it differs from [`mask_combine`] (still used for the Dual Brush) it follows
/// Photoshop as measured with single dabs over a grey ramp (#3340):
/// - Subtract takes `depth × t` away, so a white pattern removes the most paint.
/// - Color Dodge and Color Burn move the pattern toward their neutral colour as depth falls.
/// - Height treats the pattern as a height field the brush presses into: the paint reaches
///   `12 × depth × v` and fills whatever lies below that, so a low depth paints only the dark
///   valleys, from about 17 % the whole pattern is under the paint, and a brighter pattern
///   takes paint away.
/// - Linear Height is Height, but where the reach stays below the top of the pattern the paint
///   fades with the pattern instead of being cut off.
pub fn texture_combine(mode: MaskMode, v: f32, t: f32, d: f32) -> f32 {
    if v <= 0.0 {
        return 0.0;
    }
    let r = match mode {
        MaskMode::Subtract => v - d * t,
        MaskMode::ColorDodge => v / (1.0 - d * t).max(1e-3),
        MaskMode::ColorBurn => 1.0 - (1.0 - v) / (1.0 - d + d * t).max(1e-3),
        // Photoshop steps the depth in 1/255ths here, which shows at the lowest depths.
        MaskMode::Height => HEIGHT_REACH * (d * 255.0).round() / 255.0 * v - t,
        MaskMode::LinearHeight => {
            let reach = HEIGHT_REACH * (d * 255.0).round() / 255.0 * v;
            if reach < 1.0 { reach * (1.0 - t) } else { reach - t }
        }
        _ => mask_combine(mode, v, t, d),
    };
    if r.is_finite() { r.clamp(0.0, 1.0) } else { 0.0 }
}

/// Quantized parameters uniquely identifying a dab's coverage mask in [`DabCache`].
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct DabKey {
    /// Radius quantized to 0.25px steps (`radius * 4.0`).
    pub radius_quant: u16,
    /// Hardness quantized to 0..=100.
    pub hardness_quant: u8,
    /// Softness quantized to 0..=100.
    pub softness_quant: u8,
    /// Aspect ratio / roundness quantized to 0..=100.
    pub aspect_quant: u8,
    /// Rotation angle quantized to 0..=360 degrees (0 for isotropic circular tips).
    pub angle_quant: u16,
}

impl DabKey {
    pub const fn new(
        radius_quant: u16,
        hardness_quant: u8,
        softness_quant: u8,
        aspect_quant: u8,
        angle_quant: u16,
    ) -> Self {
        Self {
            radius_quant,
            hardness_quant,
            softness_quant,
            aspect_quant,
            angle_quant,
        }
    }

    pub fn from_params(radius: f32, hardness: f32, softness: f32, roundness: f32, angle: f32) -> Self {
        let radius_quant = (radius.max(0.0) * 4.0).round().min(u16::MAX as f32) as u16;
        let hardness_quant = (hardness.clamp(0.0, 1.0) * 100.0).round() as u8;
        let softness_quant = (softness.clamp(0.0, 1.0) * 100.0).round() as u8;
        let aspect_quant = (roundness.clamp(0.0, 1.0) * 100.0).round() as u8;
        // When aspect ratio is 100% (round dab), rotation is invariant so angle_quant can be 0.
        let angle_quant = if aspect_quant >= 100 {
            0
        } else {
            let norm = angle.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU;
            (norm * 360.0).round() as u16 % 360
        };
        Self {
            radius_quant,
            hardness_quant,
            softness_quant,
            aspect_quant,
            angle_quant,
        }
    }

    pub fn from_dab(d: &Dab, hardness: f32) -> Self {
        Self::from_params(d.radius, hardness, 1.0 - hardness, d.roundness, d.angle)
    }

    pub fn radius(&self) -> f32 {
        self.radius_quant as f32 * 0.25
    }

    pub fn hardness(&self) -> f32 {
        self.hardness_quant as f32 / 100.0
    }

    pub fn softness(&self) -> f32 {
        self.softness_quant as f32 / 100.0
    }

    pub fn roundness(&self) -> f32 {
        self.aspect_quant as f32 / 100.0
    }

    pub fn angle(&self) -> f32 {
        (self.angle_quant as f32 / 360.0) * std::f32::consts::TAU
    }
}

/// A pre-rendered 2D float coverage mask stored in [`DabCache`].
#[derive(Clone, Debug)]
pub struct DabMask {
    /// Width and height of the square mask in pixels (`data.len() == (size * size) as usize`).
    pub size: u32,
    /// Pre-rendered float coverage values in row-major order.
    pub data: Arc<[f32]>,
}

impl DabMask {
    pub fn new(size: u32, data: Arc<[f32]>) -> Self {
        Self { size, data }
    }

    pub fn width(&self) -> u32 {
        self.size
    }

    pub fn height(&self) -> u32 {
        self.size
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Pre-render a dab mask for a quantized [`DabKey`].
pub fn rasterize_mask_for_key(key: &DabKey) -> DabMask {
    let radius = key.radius();
    let hardness = key.hardness();
    let roundness = key.roundness();
    let angle = key.angle();

    let ro = roundness.max(0.5 / radius.max(0.5)).min(1.0);
    let rm = (radius * ro).max(0.5);
    let reach = tip_reach(radius, hardness).max(radius);
    let rr = (reach.ceil() as i32).saturating_add(1);
    let size = (2 * rr + 1) as u32;
    let reach2 = (reach + 0.5) * (reach + 0.5);

    let (sn, cs) = angle.sin_cos();
    let total_pixels = (size * size) as usize;
    let mut data = Vec::with_capacity(total_pixels);

    if roundness >= 0.999 {
        for yy in 0..size {
            let dy = yy as f32 - rr as f32;
            let dy2 = dy * dy;
            for xx in 0..size {
                let dx = xx as f32 - rr as f32;
                let d2 = dx * dx + dy2;
                let val = if d2 >= reach2 { 0.0 } else { tip_falloff(d2.sqrt(), rm, hardness) };
                data.push(val.max(0.0));
            }
        }
    } else {
        for yy in 0..size {
            let dy = yy as f32 - rr as f32;
            for xx in 0..size {
                let dx = xx as f32 - rr as f32;
                let (u, v) = (dx * cs + dy * sn, -dx * sn + dy * cs);
                let d2 = (u * ro).powi(2) + v * v;
                let val = if d2 >= reach2 { 0.0 } else { tip_falloff(d2.sqrt(), rm, hardness) };
                data.push(val.max(0.0));
            }
        }
    }

    DabMask {
        size,
        data: Arc::from(data),
    }
}

/// A thread-safe, bounded LRU cache for pre-rendered float coverage masks.
///
/// Modeled after Krita's `KisDabCache`, it eliminates per-pixel CPU recalculation by reusing
/// identical dab shapes across stroke rendering passes.
#[derive(Debug)]
pub struct DabCache {
    capacity: usize,
    entries: Mutex<DabCacheInner>,
}

#[derive(Debug)]
struct DabCacheInner {
    map: HashMap<DabKey, DabCacheEntry>,
    clock: u64,
}

#[derive(Debug)]
struct DabCacheEntry {
    mask: DabMask,
    last_used: u64,
}

impl Default for DabCache {
    fn default() -> Self {
        Self::new(Self::DEFAULT_CAPACITY)
    }
}

impl DabCache {
    pub const DEFAULT_CAPACITY: usize = 256;

    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            capacity: cap,
            entries: Mutex::new(DabCacheInner {
                map: HashMap::with_capacity(cap.min(256)),
                clock: 0,
            }),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.entries.lock().unwrap_or_else(std::sync::PoisonError::into_inner).map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        let mut inner = self.entries.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.map.clear();
        inner.clock = 0;
    }

    pub fn get(&self, key: &DabKey) -> Option<Arc<[f32]>> {
        let mut inner = self.entries.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.clock = inner.clock.wrapping_add(1);
        let clock = inner.clock;
        if let Some(entry) = inner.map.get_mut(key) {
            entry.last_used = clock;
            Some(entry.mask.data.clone())
        } else {
            None
        }
    }

    pub fn get_mask(&self, key: &DabKey) -> Option<DabMask> {
        let mut inner = self.entries.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.clock = inner.clock.wrapping_add(1);
        let clock = inner.clock;
        if let Some(entry) = inner.map.get_mut(key) {
            entry.last_used = clock;
            Some(entry.mask.clone())
        } else {
            None
        }
    }

    pub fn get_or_create(&self, key: &DabKey, rasterizer: impl FnOnce() -> DabMask) -> Arc<[f32]> {
        self.get_mask_or_create(key, rasterizer).data
    }

    pub fn get_mask_or_create(&self, key: &DabKey, rasterizer: impl FnOnce() -> DabMask) -> DabMask {
        {
            let mut inner = self.entries.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            inner.clock = inner.clock.wrapping_add(1);
            let clock = inner.clock;
            if let Some(entry) = inner.map.get_mut(key) {
                entry.last_used = clock;
                return entry.mask.clone();
            }
        }

        let mask = rasterizer();

        let mut inner = self.entries.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.clock = inner.clock.wrapping_add(1);
        let clock = inner.clock;

        if let Some(entry) = inner.map.get_mut(key) {
            entry.last_used = clock;
            return entry.mask.clone();
        }

        if inner.map.len() >= self.capacity {
            let oldest_key = inner
                .map
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(k, _)| k.clone());
            if let Some(k) = oldest_key {
                inner.map.remove(&k);
            }
        }

        inner.map.insert(key.clone(), DabCacheEntry {
            mask: mask.clone(),
            last_used: clock,
        });

        mask
    }
}

/// A stroke-ready brush: the settings plus prepared tip mips, texture, and dab cache.
#[derive(Clone, Debug)]
pub struct BrushContext {
    pub brush: BrushSettings,
    tip: Option<Arc<Mips>>,
    dual_tip: Option<Arc<Mips>>,
    texture: Option<Arc<PatternImage>>,
    pub dab_cache: Arc<DabCache>,
}

fn prepare_pattern(b: &BrushSettings) -> Option<Arc<PatternImage>> {
    let t = &b.texture;
    if !t.enabled {
        return None;
    }
    let mut img = match &t.pattern {
        Pattern::Procedural { style, size, seed } => crate::procedural::pattern(*style, *size, *seed),
        p => match p.bitmap() {
            Some(g) if g.is_valid() => PatternImage { width: g.width as usize, height: g.height as usize, data: g.to_f32() },
            _ => return None,
        },
    };
    for v in &mut img.data {
        *v = t.level(*v);
    }
    Some(Arc::new(img))
}

/// The pixel rectangle `reach` (plus 1 px of anti-aliasing slack) around a dab's centre. The casts
/// saturate and the arithmetic saturates, so a far or non-finite centre or reach from a direct
/// caller gives a (possibly empty) rectangle instead of an `i32` overflow (#977).
pub(crate) fn rect_around(d: &Dab, reach: f32) -> Rect {
    let rr = (reach.ceil() as i32).saturating_add(1);
    let (cx, cy) = (d.center.x.floor() as i32, d.center.y.floor() as i32);
    Rect::new(cx.saturating_sub(rr), cy.saturating_sub(rr), cx.saturating_add(rr).saturating_add(1), cy.saturating_add(rr).saturating_add(1))
}

impl BrushContext {
    pub fn new(brush: &BrushSettings) -> Self {
        let brush = brush.bounded_for_render();
        let mips = |t: &TipShape| match t.bitmap() {
            Some(g) if g.is_valid() => Some(Arc::new(Mips::new(g))),
            _ => None,
        };
        Self {
            tip: mips(&brush.tip),
            dual_tip: if brush.dual_brush.enabled { mips(&brush.dual_brush.tip) } else { None },
            texture: prepare_pattern(&brush),
            dab_cache: Arc::new(DabCache::new(DabCache::DEFAULT_CAPACITY)),
            brush,
        }
    }

    /// Create context with a custom shared [`DabCache`].
    pub fn with_cache(brush: &BrushSettings, dab_cache: Arc<DabCache>) -> Self {
        let mut ctx = Self::new(brush);
        ctx.dab_cache = dab_cache;
        ctx
    }

    /// Reference to the dab cache.
    pub fn dab_cache(&self) -> &DabCache {
        &self.dab_cache
    }

    /// Rasterise a dab using [`DabCache`] if available.
    pub fn rasterize_cached(&self, d: &Dab, dual: bool, rect: Rect, out: &mut Vec<f32>) {
        let hardness = if dual { self.brush.dual_brush.hardness } else { self.brush.hardness };
        let key = DabKey::from_dab(d, hardness);
        let mask = self.dab_cache.get_mask_or_create(&key, || rasterize_mask_for_key(&key));
        let (w, h) = (rect.width() as usize, rect.height() as usize);
        out.clear();
        out.resize(w * h, 0.0);
        let mw = mask.size as usize;
        let rr = (mask.size as i32 - 1) / 2;
        let (cx, cy) = (d.center.x.floor() as i32, d.center.y.floor() as i32);
        let mx0 = cx - rr;
        let my0 = cy - rr;
        let ox0 = (mx0 - rect.x0).max(0);
        let oy0 = (my0 - rect.y0).max(0);
        let ox1 = (mx0 + mask.size as i32 - rect.x0).min(rect.width() as i32);
        let oy1 = (my0 + mask.size as i32 - rect.y0).min(rect.height() as i32);
        if ox1 > ox0 && oy1 > oy0 {
            for y in oy0..oy1 {
                let my = (rect.y0 + y - my0) as usize;
                let o_row = y as usize * w;
                let m_row = my * mw;
                for x in ox0..ox1 {
                    let mx = (rect.x0 + x - mx0) as usize;
                    out[o_row + x as usize] = mask.data[m_row + mx] * d.alpha;
                }
            }
        }
    }

    /// Texture value at a document pixel (bilinear, tiled).
    #[inline]
    pub fn texture_at(&self, x: i32, y: i32) -> f32 {
        let Some(p) = &self.texture else { return 1.0 };
        let s = self.brush.texture.scale.max(0.01);
        if s == 1.0 {
            // Pixel centres land exactly on texels at 100 %.
            return p.data[y.rem_euclid(p.height as i32) as usize * p.width + x.rem_euclid(p.width as i32) as usize];
        }
        let (fx, fy) = ((x as f32 + 0.5) / s - 0.5, (y as f32 + 0.5) / s - 0.5);
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let a = p.sample_wrap(x0, y0) + (p.sample_wrap(x0 + 1.0, y0) - p.sample_wrap(x0, y0)) * tx;
        let b = p.sample_wrap(x0, y0 + 1.0) + (p.sample_wrap(x0 + 1.0, y0 + 1.0) - p.sample_wrap(x0, y0 + 1.0)) * tx;
        a + (b - a) * ty
    }

    /// Bounding rectangle of a dab (rotated sampled tips need the corner reach).
    pub fn dab_rect(&self, d: &Dab, dual: bool) -> Rect {
        let sampled = if dual { self.dual_tip.is_some() } else { self.tip.is_some() };
        let hardness = if dual { self.brush.dual_brush.hardness } else { self.brush.hardness };
        // A computed soft tip paints past its nominal radius (see `tip_shape`).
        let aliased = self.brush.aliased && !dual;
        let reach = if sampled {
            d.radius * std::f32::consts::SQRT_2
        } else if aliased {
            d.radius
        } else {
            crate::tip_reach(d.radius, hardness).max(d.radius)
        };
        rect_around(d, reach)
    }

    /// Rasterise a dab over `rect` into `out` (tip shape × noise × per-tip texture × wet edges × flow).
    /// Wet Edges shapes this one dab; a stroke applies it to its accumulated coverage instead.
    pub fn rasterize(&self, d: &Dab, dual: bool, rect: Rect, out: &mut Vec<f32>) {
        self.rasterize_into(d, dual, rect, out, None);
    }

    /// [`rasterize`](Self::rasterize); with `depth`, Wet Edges is left out of `out` and each
    /// painted pixel's depth inside the dab (`1 − rn`) goes to `depth`, for the stroke to apply it.
    pub fn rasterize_into(&self, d: &Dab, dual: bool, rect: Rect, out: &mut Vec<f32>, mut depth: Option<&mut Vec<f32>>) {
        let (w, h) = (rect.width() as usize, rect.height() as usize);
        out.clear();
        out.resize(w * h, 0.0);
        if let Some(dp) = depth.as_deref_mut() {
            dp.clear();
            dp.resize(w * h, 0.0);
        }
        let b = &self.brush;
        let (hardness, mips) = if dual { (b.dual_brush.hardness, self.dual_tip.as_deref()) } else { (b.hardness, self.tip.as_deref()) };
        let aliased = b.aliased && !dual;
        let wet = b.wet_edges && !dual && depth.is_none();
        let noise = b.noise && !dual;
        let tex_tip = !dual && b.texture.enabled && b.texture.each_tip && self.texture.is_some();
        let (cx, cy) = if aliased { grid_center(d.center.x, d.center.y, 2.0 * d.radius) } else { (d.center.x as f32, d.center.y as f32) };
        let (sn, cs) = d.angle.sin_cos();
        let (fx, fy) = (if d.flip_x { -1.0 } else { 1.0 }, if d.flip_y { -1.0 } else { 1.0 });
        // Brush Projection: stretch the sampling coordinates along the tilt direction, which
        // foreshortens the tip there by `proj_scale`.
        let proj = (!dual && d.proj_scale < 0.999).then(|| {
            let (ps, pc) = d.proj_angle.sin_cos();
            (pc, ps, 1.0 / d.proj_scale.max(0.05) - 1.0)
        });
        let r = d.radius;
        let ro = d.roundness.max(0.5 / r).min(1.0);
        let rm = (r * ro).max(0.5);
        let reach = if aliased || mips.is_some() { rm } else { crate::tip_reach(rm, hardness).max(rm) };
        let reach2 = (reach + 0.5) * (reach + 0.5);
        // Sampled-tip mapping.
        let (level, inv_scale, tw, th) = match mips {
            Some(m) => {
                let (w0, h0, _) = &m.levels[0];
                let big = *w0.max(h0) as f32;
                (m.level_for(2.0 * r * ro.max(0.25)), big / (2.0 * r), *w0 as f32, *h0 as f32)
            }
            None => (0, 1.0, 1.0, 1.0),
        };
        // A pixel offset from the dab centre in tip space: to y-up, project, rotate by -angle, flip.
        let to_tip = |dx: f32, dy: f32| {
            let (mut ux, mut uy) = (dx, -dy);
            if let Some((ax, ay, k)) = proj {
                let t = (ux * ax + uy * ay) * k;
                ux += t * ax;
                uy += t * ay;
            }
            ((ux * cs + uy * sn) * fx, (-ux * sn + uy * cs) * fy)
        };
        // A tip a few pixels across takes each pixel's covered area (a 4×4 grid inside it):
        // sampling only the pixel centre made a tiny dab's total coverage swing with its sub-pixel
        // position, and thin lines beaded.
        let small = mips.is_none() && !aliased && rm < SMALL_TIP_RADIUS;
        let small_reach2 = (reach + 0.75) * (reach + 0.75);
        // Keep ordinary computed tips separate from the feature loop, so their dense rows
        // can be vectorised without the per-pixel sampled-mask and stroke-depth branches.
        if !(mips.is_some() || small || (aliased && b.square) || noise || tex_tip || wet)
            && depth.is_none()
            && proj.is_none()
            && d.angle == 0.0
            && d.radius.is_finite()
            && d.roundness.is_finite()
            && rm.is_finite()
            && cx.is_finite()
            && cy.is_finite()
            && hardness.is_finite()
            && d.alpha.is_finite()
            && w > 0
        {
            rasterize_round(d, rect, out, hardness, aliased);
            return;
        }
        for yy in 0..h {
            let y = rect.y0 + yy as i32;
            let dy = y as f32 + 0.5 - cy;
            for xx in 0..w {
                let x = rect.x0 + xx as i32;
                let dx = x as f32 + 0.5 - cx;
                let (u, v) = to_tip(dx, dy);
                let (mut val, rn) = match mips {
                    // Block: the whole-pixel square, hard (the Eraser's Block mode).
                    _ if aliased && b.square => {
                        if dx.abs() >= r || dy.abs() >= r {
                            continue;
                        }
                        (1.0, 0.0)
                    }
                    None if small => {
                        let d2 = (u * ro).powi(2) + v * v;
                        if d2 >= small_reach2 {
                            continue;
                        }
                        let mut sum = 0.0;
                        for sy in SUBPIXEL {
                            for sx in SUBPIXEL {
                                let (su, sv) = to_tip(dx + sx, dy + sy);
                                let sd = ((su * ro).powi(2) + sv * sv).sqrt();
                                // Each sample takes the tip's own edge, so the covered area
                                // changes smoothly with the dab's position.
                                sum += tip_falloff(sd, rm, hardness);
                            }
                        }
                        (sum / (SUBPIXEL.len() * SUBPIXEL.len()) as f32, d2.sqrt() / rm)
                    }
                    None => {
                        let d2 = (u * ro).powi(2) + v * v;
                        if d2 >= reach2 {
                            continue;
                        }
                        let dd = d2.sqrt();
                        let val = if aliased { if dd <= rm { 1.0 } else { 0.0 } } else { dab_coverage(dd, rm, hardness) };
                        (val, dd / rm)
                    }
                    Some(m) => {
                        let (px, py) = (u * inv_scale + tw / 2.0, -(v / ro) * inv_scale + th / 2.0);
                        if px < -1.0 || py < -1.0 || px > tw + 1.0 || py > th + 1.0 {
                            continue;
                        }
                        let s = m.sample_clamped(level, px / tw, py / th);
                        let val = if aliased { if s >= 0.5 { 1.0 } else { 0.0 } } else { s };
                        (val, 1.0 - s)
                    }
                };
                if val <= 0.0 {
                    continue;
                }
                if noise && val < 1.0 {
                    let n = hash2(x, y, NOISE_SALT);
                    let hard = if n < val { 1.0 } else { 0.0 };
                    val += (hard - val) * 0.7;
                }
                if tex_tip {
                    val = texture_combine(b.texture.mode, val, self.texture_at(x, y), d.depth);
                }
                if wet {
                    val *= wet_factor(rn);
                }
                if let Some(slot) = depth.as_deref_mut().and_then(|dp| dp.get_mut(yy * w + xx)) {
                    *slot = (1.0 - rn).clamp(0.0, 1.0);
                }
                out[yy * w + xx] = val * d.alpha;
            }
        }
    }

    /// One dab's dense footprint (for the sequential retouch tools).
    pub fn footprint(&self, d: &Dab, index: usize) -> Footprint {
        let rect = self.dab_rect(d, false);
        let mut cov = Vec::new();
        self.rasterize(d, false, rect, &mut cov);
        Footprint { dab: *d, index, rect, cov }
    }

    /// Stroke-level masks applied to accumulated coverage `c` at a pixel.
    #[inline]
    pub fn stroke_mask(&self, c: f32, dual: Option<f32>, x: i32, y: i32) -> f32 {
        self.stroke_mask_wet(c, None, dual, x, y)
    }

    /// [`stroke_mask`](Self::stroke_mask) with Wet Edges from the stroke's depth at the pixel
    /// (the deepest any dab reaches it, see [`CoverageMap::depth`]).
    #[inline]
    pub fn stroke_mask_wet(&self, c: f32, depth: Option<f32>, dual: Option<f32>, x: i32, y: i32) -> f32 {
        let b = &self.brush;
        let mut m = match depth {
            Some(dp) => c * wet_factor(1.0 - dp),
            None => c,
        };
        if let Some(dv) = dual {
            m = mask_combine(b.dual_brush.mode, m, dv, 1.0);
        }
        if self.texture.is_some() && !b.texture.each_tip {
            m = texture_combine(b.texture.mode, m, self.texture_at(x, y), b.texture.depth.clamp(0.0, 1.0));
        }
        m
    }
}

// ---------------------------------------------------------------------------
// Sparse stroke buffer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct CovTile {
    cov: Vec<f32>,
    /// Wet Edges: the deepest any dab reaches each pixel (`1 − rn` of the nearest dab), empty
    /// without Wet Edges. The stroke's interior is deep, its rim shallow, whatever the dabs' overlap.
    depth: Vec<f32>,
    /// Premultiplied native colour channels (`nc` per pixel), empty without per-dab colour.
    col: Vec<f32>,
}

/// Vectorized SIMD-friendly row accumulation with saturating alpha build-up towards ceiling.
#[inline]
fn accumulate_row_simd(dst: &mut [f32], src: &[f32], alpha: f32, ceil: f32) {
    let len = dst.len().min(src.len());
    let dst = &mut dst[..len];
    let src = &src[..len];

    let chunks = len / 8;
    for i in 0..chunks {
        let base = i * 8;
        let d = &mut dst[base..base + 8];
        let s = &src[base..base + 8];
        for j in 0..8 {
            let c = d[j];
            let v = s[j] * alpha;
            let delta = (v * (ceil - c)).max(0.0);
            d[j] = c + delta;
        }
    }
    for i in (chunks * 8)..len {
        let c = dst[i];
        let v = src[i] * alpha;
        let delta = (v * (ceil - c)).max(0.0);
        dst[i] = c + delta;
    }
}

/// A sparse tiled coverage (+ optional colour) buffer.
#[derive(Clone, Debug, Default)]
pub struct CoverageMap {
    tiles: HashMap<(i32, i32), CovTile>,
    nc: usize,
    bounds: Rect,
    dirty: HashSet<(i32, i32)>,
}

impl CoverageMap {
    pub fn new(color_channels: usize) -> Self {
        Self { nc: color_channels, ..Default::default() }
    }
    pub fn bounds(&self) -> Rect {
        self.bounds
    }
    #[inline]
    pub fn get(&self, x: i32, y: i32) -> f32 {
        let (tx, ty) = (x.div_euclid(COV_TILE), y.div_euclid(COV_TILE));
        self.tiles.get(&(tx, ty)).map_or(0.0, |t| t.cov[((y - ty * COV_TILE) * COV_TILE + (x - tx * COV_TILE)) as usize])
    }

    /// Wet Edges depth at a pixel (see [`CovTile::depth`]); `None` when the stroke has none.
    #[inline]
    pub fn depth(&self, x: i32, y: i32) -> Option<f32> {
        let (tx, ty) = (x.div_euclid(COV_TILE), y.div_euclid(COV_TILE));
        self.tiles.get(&(tx, ty)).and_then(|t| t.depth.get(((y - ty * COV_TILE) * COV_TILE + (x - tx * COV_TILE)) as usize).copied())
    }

    /// Number of active tiles stored in the coverage buffer.
    pub fn tiles_count(&self) -> usize {
        self.tiles.len()
    }

    /// Accumulate dab values over `rect` scaled by `alpha` with ceiling `ceil`.
    pub fn accumulate_dab(
        &mut self,
        rect: Rect,
        mask: &[f32],
        alpha: f32,
        ceil: f32,
        color: Option<&[f32]>,
        depth: Option<&[f32]>,
    ) {
        if rect.is_empty() || alpha <= 0.0 || ceil <= 0.0 {
            return;
        }
        self.bounds = self.bounds.union(&rect);
        let w = rect.width() as usize;
        let nc = self.nc;
        let (t0x, t0y) = (rect.x0.div_euclid(COV_TILE), rect.y0.div_euclid(COV_TILE));
        let (t1x, t1y) = ((rect.x1 - 1).div_euclid(COV_TILE), (rect.y1 - 1).div_euclid(COV_TILE));
        let want_col = color.is_some() && nc > 0;
        let rr = ((rect.width() as i32) - 1) / 2;
        let cx = rect.x0 + rr;
        let cy = rect.y0 + rr;
        let rr2 = (rr + 1) * (rr + 1);

        for ty in t0y..=t1y {
            for tx in t0x..=t1x {
                let tr = Rect::new(tx * COV_TILE, ty * COV_TILE, (tx + 1) * COV_TILE, (ty + 1) * COV_TILE).intersect(&rect);
                if tr.is_empty() {
                    continue;
                }
                // Rapidly cull tiles outside the dab's circular reach
                let close_x = cx.clamp(tr.x0, tr.x1 - 1);
                let close_y = cy.clamp(tr.y0, tr.y1 - 1);
                let dx = close_x - cx;
                let dy = close_y - cy;
                if dx * dx + dy * dy > rr2 {
                    continue;
                }

                let span_w = (tr.x1 - tr.x0) as usize;
                let tile = match self.tiles.get_mut(&(tx, ty)) {
                    Some(t) => {
                        self.dirty.insert((tx, ty));
                        t
                    }
                    None => {
                        let any = (tr.y0..tr.y1).any(|y| {
                            let row = (y - rect.y0) as usize * w + (tr.x0 - rect.x0) as usize;
                            mask[row..row + span_w].iter().any(|&v| v > 0.0)
                        });
                        if !any {
                            continue;
                        }
                        self.dirty.insert((tx, ty));
                        self.tiles.entry((tx, ty)).or_insert_with(|| CovTile {
                            cov: vec![0.0; (COV_TILE * COV_TILE) as usize],
                            depth: Vec::new(),
                            col: if want_col { vec![0.0; (COV_TILE * COV_TILE) as usize * nc] } else { Vec::new() },
                        })
                    }
                };
                if want_col && tile.col.is_empty() {
                    tile.col = vec![0.0; (COV_TILE * COV_TILE) as usize * nc];
                }
                if depth.is_some() && tile.depth.is_empty() {
                    tile.depth = vec![0.0; (COV_TILE * COV_TILE) as usize];
                }

                if depth.is_none() && !want_col {
                    // Vectorized SIMD accumulation directly into tile coverage
                    for y in tr.y0..tr.y1 {
                        let m_row = (y - rect.y0) as usize * w + (tr.x0 - rect.x0) as usize;
                        let t_row = ((y - ty * COV_TILE) * COV_TILE) as usize + (tr.x0 - tx * COV_TILE) as usize;
                        accumulate_row_simd(
                            &mut tile.cov[t_row..t_row + span_w],
                            &mask[m_row..m_row + span_w],
                            alpha,
                            ceil,
                        );
                    }
                } else {
                    for y in tr.y0..tr.y1 {
                        let row = (y - rect.y0) as usize * w;
                        let trow = ((y - ty * COV_TILE) * COV_TILE) as usize;
                        for x in tr.x0..tr.x1 {
                            let v = mask[row + (x - rect.x0) as usize] * alpha;
                            if v <= 0.0 {
                                continue;
                            }
                            let ti = trow + (x - tx * COV_TILE) as usize;
                            if let (Some(dv), Some(slot)) = (depth.and_then(|dp| dp.get(row + (x - rect.x0) as usize)), tile.depth.get_mut(ti)) {
                                *slot = slot.max(*dv);
                            }
                            let c = tile.cov[ti];
                            let nv = if ceil > c { c + v * (ceil - c) } else { c };
                            if nv <= c {
                                continue;
                            }
                            if let (true, Some(col)) = (want_col, color) {
                                let k = if c < 1.0 { (nv - c) / (1.0 - c) } else { 0.0 };
                                let p = &mut tile.col[ti * nc..(ti + 1) * nc];
                                for i in 0..nc {
                                    p[i] = col[i] * k + p[i] * (1.0 - k);
                                }
                            }
                            tile.cov[ti] = nv;
                        }
                    }
                }
            }
        }
    }

    /// Accumulate dab values over `rect`: flow builds up towards the dab's opacity ceiling
    /// (`c ← c + v·(ceil − c)`). `color` = native colour for per-dab colour. `depth` (Wet Edges, one
    /// value per pixel of `rect`) keeps the deepest any dab reaches each pixel.
    pub fn accumulate(&mut self, rect: Rect, vals: &[f32], ceil: f32, color: Option<&[f32]>, depth: Option<&[f32]>) {
        self.accumulate_dab(rect, vals, 1.0, ceil, color, depth);
    }

    fn take_dirty(&mut self) -> Vec<(i32, i32)> {
        let mut v: Vec<_> = self.dirty.drain().collect();
        v.sort_unstable();
        v
    }

    /// The finished coverage over the tiles `keys`: `tail`'s copy of a tile where finishing the
    /// stroke changes it (see [`StrokeRenderer::tail_preview`]), else this map's own.
    fn finished_subset(&self, tail: Option<&Self>, keys: &HashSet<(i32, i32)>) -> Self {
        let tile = |k: &(i32, i32)| tail.and_then(|t| t.tiles.get(k)).or_else(|| self.tiles.get(k)).map(|t| (*k, t.clone()));
        Self { tiles: keys.iter().filter_map(tile).collect(), nc: self.nc, bounds: tail.map_or(self.bounds, |t| t.bounds), dirty: HashSet::new() }
    }

    /// Union two passes of the same brush stroke without applying opacity twice where they
    /// overlap. The stronger coverage wins, including its per-dab colour when present.
    fn union_max(&mut self, other: &Self) {
        self.bounds = self.bounds.union(&other.bounds);
        for (&key, source) in &other.tiles {
            let Some(target) = self.tiles.get_mut(&key) else {
                self.tiles.insert(key, source.clone());
                self.dirty.insert(key);
                continue;
            };
            if target.depth.is_empty() {
                target.depth.clone_from(&source.depth);
            } else {
                for (dst, src) in target.depth.iter_mut().zip(&source.depth) {
                    *dst = dst.max(*src);
                }
            }
            if self.nc == 0 {
                for (dst, src) in target.cov.iter_mut().zip(&source.cov) {
                    *dst = (*dst).max(*src);
                }
            } else {
                for (index, (dst, src)) in target.cov.iter_mut().zip(&source.cov).enumerate() {
                    if *src > *dst {
                        *dst = *src;
                        let start = index.saturating_mul(self.nc);
                        let end = start.saturating_add(self.nc);
                        if let (Some(dst_color), Some(src_color)) = (target.col.get_mut(start..end), source.col.get(start..end)) {
                            dst_color.copy_from_slice(src_color);
                        }
                    }
                }
            }
            self.dirty.insert(key);
        }
    }
}

// ---------------------------------------------------------------------------
// Stroke renderer
// ---------------------------------------------------------------------------

/// What [`StrokeRenderer::composite_union_live`] drew last: the coverage tiles of the passes'
/// smoothing tails (redrawn without them next time) and the merged bounds.
#[derive(Clone, Debug, Default)]
pub struct UnionLive {
    tail_keys: HashSet<(i32, i32)>,
    bounds: Rect,
}

impl UnionLive {
    /// The merged bounds of the finished passes, as last drawn.
    pub fn bounds(&self) -> Rect {
        self.bounds
    }
}

/// Incremental stroke rendering: feed points, composite dirty tiles from the pre-stroke pixels.
#[derive(Clone, Debug)]
pub struct StrokeRenderer {
    pub ctx: BrushContext,
    generator: DabGenerator,
    cov: CoverageMap,
    dual: Option<CoverageMap>,
    fmt: Option<PixelFormat>,
    per_dab_color: bool,
    dabs_done: usize,
    scratch: Vec<f32>,
    /// Wet Edges depth of the dab being accumulated (see [`CovTile::depth`]).
    depth_scratch: Vec<f32>,
    dab_buf: Vec<Dab>,
    dual_buf: Vec<Dab>,
    all_dabs: Option<Vec<Dab>>,
}

impl StrokeRenderer {
    /// Composite the union of this stroke and a mirrored pass as one stroke. This keeps
    /// overlapping dabs on a symmetry axis under one opacity ceiling at every bit depth.
    pub fn composite_union(&self, other: &Self, pre: &Surface, target: &mut Surface, selection: Option<&Surface>, lock_transparency: bool) -> Rect {
        self.composite_union_many(std::iter::once(other), pre, target, selection, lock_transparency)
    }

    /// Combine all symmetry passes before applying brush opacity and selection once.
    pub fn composite_union_many<'a>(
        &self,
        others: impl IntoIterator<Item = &'a Self>,
        pre: &Surface,
        target: &mut Surface,
        selection: Option<&Surface>,
        lock_transparency: bool,
    ) -> Rect {
        let mut merged = self.clone();
        for other in others {
            merged.cov.union_max(&other.cov);
            if let (Some(to), Some(from)) = (&mut merged.dual, &other.dual) {
                to.union_max(from);
            }
        }
        merged.composite(pre, target, selection, lock_transparency, true)
    }

    /// Coverage tiles (stroke and Dual Brush) changed since the last call or composite, into
    /// `keys`; forgets them.
    fn drain_dirty_into(&mut self, keys: &mut HashSet<(i32, i32)>) {
        keys.extend(self.cov.dirty.drain());
        if let Some(d) = self.dual.as_mut() {
            keys.extend(d.dirty.drain());
        }
    }

    /// Every coverage tile (stroke and Dual Brush), into `keys`.
    fn keys_into(&self, keys: &mut HashSet<(i32, i32)>) {
        keys.extend(self.cov.tiles.keys());
        if let Some(d) = &self.dual {
            keys.extend(d.tiles.keys());
        }
    }

    /// Live preview of a stroke painted as several passes (the stroke, then its symmetry
    /// mirrors): leaves `target` as finishing every pass now and compositing them with
    /// [`composite_union_many`](Self::composite_union_many) over `pre` would, but redraws only the
    /// coverage tiles that changed since the previous call (new dabs, and the smoothing tail now
    /// and before). `full` redraws every tile: on the first call, or when a pass joins. Returns
    /// the damaged rectangle.
    pub fn composite_union_live(
        passes: &mut [&mut Self],
        live: &mut UnionLive,
        full: bool,
        pre: &Surface,
        target: &mut Surface,
        selection: Option<&Surface>,
        lock_transparency: bool,
    ) -> Rect {
        let mut keys = std::mem::take(&mut live.tail_keys);
        for p in passes.iter_mut() {
            p.drain_dirty_into(&mut keys);
            if full {
                p.keys_into(&mut keys);
            }
        }
        let tails: Vec<_> = passes.iter().map(|p| p.tail_preview()).collect();
        let mut tail_keys = HashSet::new();
        for t in tails.iter().flatten() {
            t.keys_into(&mut tail_keys);
        }
        keys.extend(&tail_keys);
        let mut merged: Option<Self> = None;
        for (p, t) in passes.iter().zip(&tails) {
            let cov = p.cov.finished_subset(t.as_ref().map(|t| &t.cov), &keys);
            let dual = p.dual.as_ref().map(|d| d.finished_subset(t.as_ref().and_then(|t| t.dual.as_ref()), &keys));
            match merged.as_mut() {
                None => {
                    merged = Some(Self {
                        ctx: p.ctx.clone(),
                        generator: p.generator.clone(),
                        cov,
                        dual,
                        fmt: p.fmt,
                        per_dab_color: p.per_dab_color,
                        dabs_done: p.dabs_done,
                        scratch: Vec::new(),
                        depth_scratch: Vec::new(),
                        dab_buf: Vec::new(),
                        dual_buf: Vec::new(),
                        all_dabs: None,
                    })
                }
                Some(m) => {
                    m.cov.union_max(&cov);
                    if let (Some(to), Some(from)) = (&mut m.dual, &dual) {
                        to.union_max(from);
                    }
                }
            }
        }
        let Some(mut merged) = merged else { return Rect::EMPTY };
        let (before, after) = (live.bounds, merged.bounds());
        // Back to the pre-stroke pixels where the previous call drew and nothing covers now.
        let mut keys: Vec<_> = keys.into_iter().collect();
        keys.sort_unstable();
        let mut dmg = Rect::EMPTY;
        let mut region = Vec::new();
        for &(tx, ty) in &keys {
            let tile = Rect::new(tx * COV_TILE, ty * COV_TILE, (tx + 1) * COV_TILE, (ty + 1) * COV_TILE);
            let old = tile.intersect(&before);
            if !old.is_empty() {
                pre.read_region_into(old, &mut region);
                target.write_region(old, &region);
                dmg = dmg.union(&old);
            }
            if merged.cov.tiles.contains_key(&(tx, ty)) {
                dmg = dmg.union(&tile.intersect(&after));
            }
        }
        merged.composite(pre, target, selection, lock_transparency, true);
        live.tail_keys = tail_keys;
        live.bounds = after;
        dmg
    }
    /// `fmt` = target pixel format (needed for per-dab colour); `zoom` for smoothing.
    pub fn new(brush: &BrushSettings, fmt: Option<PixelFormat>, zoom: f32) -> Self {
        let per_dab_color = brush.color_dynamics.enabled && !brush.erase && fmt.is_some();
        let nc = if per_dab_color { fmt.map_or(0, |f| f.mode.color_channels()) } else { 0 };
        Self {
            ctx: BrushContext::new(brush),
            generator: DabGenerator::new(brush, zoom),
            cov: CoverageMap::new(nc),
            dual: brush.dual_brush.enabled.then(|| CoverageMap::new(0)),
            fmt,
            per_dab_color,
            dabs_done: 0,
            scratch: Vec::new(),
            depth_scratch: Vec::new(),
            dab_buf: Vec::new(),
            dual_buf: Vec::new(),
            all_dabs: None,
        }
    }

    /// Keep a copy of every primary dab (for tests and sequential tools).
    pub fn record_dabs(mut self) -> Self {
        self.all_dabs = Some(Vec::new());
        self
    }
    pub fn dabs(&self) -> &[Dab] {
        self.all_dabs.as_deref().unwrap_or(&[])
    }
    pub fn dab_count(&self) -> usize {
        self.dabs_done
    }
    /// Union of the dab rectangles so far.
    pub fn bounds(&self) -> Rect {
        self.cov.bounds
    }

    /// Reference to the dab cache used by this renderer.
    pub fn dab_cache(&self) -> &DabCache {
        &self.ctx.dab_cache
    }

    fn raster_pending(&mut self) {
        let mut dabs = std::mem::take(&mut self.dab_buf);
        let mut duals = std::mem::take(&mut self.dual_buf);
        let wet = self.ctx.brush.wet_edges;
        let mut native = [0.0f32; 8];
        for d in &dabs {
            let col = match (self.per_dab_color, self.fmt) {
                (true, Some(f)) => {
                    from_rgba_into(&f, d.color, &mut native);
                    Some(&native[..f.mode.color_channels()])
                }
                _ => None,
            };

            let r = self.ctx.dab_rect(d, false);
            self.ctx.rasterize_into(d, false, r, &mut self.scratch, wet.then_some(&mut self.depth_scratch));
            self.cov.accumulate(r, &self.scratch, d.opacity, col, wet.then_some(&self.depth_scratch[..]));
            self.cov.bounds = self.cov.bounds.union(&r);
        }
        if let Some(dm) = self.dual.as_mut() {
            for d in &duals {
                let r = self.ctx.dab_rect(d, true);
                self.ctx.rasterize(d, true, r, &mut self.scratch);
                dm.accumulate(r, &self.scratch, 1.0, None, None);
                dm.bounds = dm.bounds.union(&r);
            }
        }
        self.dabs_done += dabs.len();
        if let Some(all) = self.all_dabs.as_mut() {
            all.extend_from_slice(&dabs);
        }
        dabs.clear();
        duals.clear();
        self.dab_buf = dabs;
        self.dual_buf = duals;
    }

    /// See [`DabGenerator::wants_time`].
    pub fn wants_time(&self) -> bool {
        self.generator.wants_time()
    }

    /// The string between the brush and the pointer while smoothing lags (see
    /// [`Smoother::string`](crate::dynamics::Smoother::string)).
    pub fn smoothing_string(&self) -> Option<([f64; 2], [f64; 2])> {
        self.generator.smoothing_string()
    }

    /// Feed input points (any chunking gives the same result).
    pub fn push(&mut self, pts: &[StrokePoint]) {
        self.generator.push(pts, &mut self.dab_buf, &mut self.dual_buf);
        self.raster_pending();
    }

    /// End of stroke (smoothing catch-up, a lone first point).
    pub fn finish(&mut self) {
        self.generator.finish(&mut self.dab_buf, &mut self.dual_buf);
        self.raster_pending();
    }

    /// The primary dabs finishing the stroke now would add, without finishing it: for live
    /// previews of sequential-dab tools, which run each dab themselves (see [`tail_preview`](Self::tail_preview)).
    pub fn tail_dabs(&self) -> Vec<Dab> {
        let mut generator = self.generator.clone();
        let (mut dabs, mut duals) = (Vec::new(), Vec::new());
        generator.finish(&mut dabs, &mut duals);
        dabs
    }

    /// What finishing the stroke now would add (the smoothing catch-up tail to the last point, or
    /// a lone first dab), for live previews: a renderer holding copies of the coverage tiles the
    /// tail touches with the tail rendered in, so its `composite` draws exactly those tiles as
    /// [`finish`](Self::finish) would leave them. `None` when finishing adds nothing.
    pub fn tail_preview(&self) -> Option<StrokeRenderer> {
        let mut generator = self.generator.clone();
        let (mut dabs, mut duals) = (Vec::new(), Vec::new());
        generator.finish(&mut dabs, &mut duals);
        if dabs.is_empty() && duals.is_empty() {
            return None;
        }
        let mut keys = HashSet::new();
        for r in dabs.iter().map(|d| self.ctx.dab_rect(d, false)).chain(duals.iter().map(|d| self.ctx.dab_rect(d, true))) {
            if r.is_empty() {
                continue;
            }
            for ty in r.y0.div_euclid(COV_TILE)..=(r.y1 - 1).div_euclid(COV_TILE) {
                for tx in r.x0.div_euclid(COV_TILE)..=(r.x1 - 1).div_euclid(COV_TILE) {
                    keys.insert((tx, ty));
                }
            }
        }
        let subset = |m: &CoverageMap| CoverageMap {
            tiles: keys.iter().filter_map(|k| m.tiles.get(k).map(|t| (*k, t.clone()))).collect(),
            nc: m.nc,
            bounds: m.bounds,
            dirty: HashSet::new(),
        };
        let mut t = StrokeRenderer {
            ctx: self.ctx.clone(),
            generator,
            cov: subset(&self.cov),
            dual: self.dual.as_ref().map(subset),
            fmt: self.fmt,
            per_dab_color: self.per_dab_color,
            dabs_done: self.dabs_done,
            scratch: Vec::new(),
            depth_scratch: Vec::new(),
            dab_buf: dabs,
            dual_buf: duals,
            all_dabs: None,
        };
        t.raster_pending();
        // Tiles whose coverage the tail doesn't change still composite the same: redraw them all.
        t.cov.dirty.extend(t.cov.tiles.keys().copied());
        Some(t)
    }

    /// Mark the coverage tiles over `r` for the next `composite` (e.g. to redraw over a preview).
    pub fn mark_dirty(&mut self, r: Rect) {
        if r.is_empty() {
            return;
        }
        for ty in r.y0.div_euclid(COV_TILE)..=(r.y1 - 1).div_euclid(COV_TILE) {
            for tx in r.x0.div_euclid(COV_TILE)..=(r.x1 - 1).div_euclid(COV_TILE) {
                if self.cov.tiles.contains_key(&(tx, ty)) {
                    self.cov.dirty.insert((tx, ty));
                }
            }
        }
    }

    /// The coverage tiles touched since the last call (or `composite`), as one rectangle clipped to
    /// the stroke bounds, and forget them: what a live preview that composites its own paint has
    /// to redraw.
    pub fn take_dirty_rect(&mut self) -> Rect {
        let r = self
            .cov
            .take_dirty()
            .into_iter()
            .fold(Rect::EMPTY, |acc, (tx, ty)| acc.union(&Rect::new(tx * COV_TILE, ty * COV_TILE, (tx + 1) * COV_TILE, (ty + 1) * COV_TILE)));
        if let Some(d) = self.dual.as_mut() {
            d.dirty.clear();
        }
        r.intersect(&self.bounds())
    }

    /// Final coverage over `r`, one value per pixel in rows (as [`dense_coverage`](Self::dense_coverage)
    /// over its bounds).
    pub fn coverage_in(&self, r: Rect) -> Vec<f32> {
        (r.y0..r.y1).flat_map(|y| (r.x0..r.x1).map(move |x| (x, y))).map(|(x, y)| self.coverage_at(x, y)).collect()
    }

    /// Final stroke coverage at a pixel (stroke-level masks applied, before opacity/selection).
    pub fn coverage_at(&self, x: i32, y: i32) -> f32 {
        let c = self.cov.get(x, y);
        if c <= 0.0 {
            return 0.0;
        }
        self.ctx.stroke_mask_wet(c, self.cov.depth(x, y), self.dual.as_ref().map(|d| d.get(x, y)), x, y)
    }

    /// Dense final coverage over the stroke bounds.
    pub fn dense_coverage(&self) -> (Rect, Vec<f32>) {
        let b = self.bounds();
        let w = b.width() as usize;
        let mut out = vec![0.0f32; w * b.height() as usize];
        for (&(tx, ty), t) in &self.cov.tiles {
            let tr = Rect::new(tx * COV_TILE, ty * COV_TILE, (tx + 1) * COV_TILE, (ty + 1) * COV_TILE).intersect(&b);
            for y in tr.y0..tr.y1 {
                for x in tr.x0..tr.x1 {
                    let ti = ((y - ty * COV_TILE) * COV_TILE + (x - tx * COV_TILE)) as usize;
                    let c = t.cov[ti];
                    if c > 0.0 {
                        out[(y - b.y0) as usize * w + (x - b.x0) as usize] =
                            self.ctx.stroke_mask_wet(c, t.depth.get(ti).copied(), self.dual.as_ref().map(|d| d.get(x, y)), x, y);
                    }
                }
            }
        }
        (b, out)
    }

    /// Composite the stroke buffer onto `target`, reading original pixels from `pre` (the
    /// pre-stroke surface; pass a clone of the target taken before the first `push`). With
    /// `all = false` only tiles touched since the last composite are processed (interactive use).
    /// Returns the damaged rectangle.
    pub fn composite(&mut self, pre: &Surface, target: &mut Surface, selection: Option<&Surface>, lock_transparency: bool, all: bool) -> Rect {
        let keys: Vec<(i32, i32)> = if all {
            self.cov.dirty.clear();
            let mut k: Vec<_> = self.cov.tiles.keys().copied().collect();
            k.sort_unstable();
            k
        } else {
            self.cov.take_dirty()
        };
        if let Some(d) = self.dual.as_mut() {
            d.dirty.clear();
        }
        let bounds = self.bounds();
        let fmt = target.format();
        let n = fmt.channels();
        let a_idx = alpha_index(&fmt);
        let nc = fmt.mode.color_channels();
        let b = &self.ctx.brush;
        let opacity = b.opacity.clamp(0.0, 1.0);
        let mut src = [0.0f32; 8];
        from_rgba_into(&fmt, b.color, &mut src);
        // A colour picked as CMYK paints its inks as they are in a CMYK target (#4344).
        if let (ColorMode::Cmyk, Some((rgb, inks))) = (fmt.mode, b.color_cmyk)
            && rgb == [b.color[0], b.color[1], b.color[2]]
        {
            src[..4].copy_from_slice(&inks);
        }
        if let Some(a) = a_idx {
            src[a] = b.color[3];
        }
        // 8-bit targets: the pixels the stroke changes are dithered as they are quantised, so
        // soft, low-flow strokes don't band (see `dither8`). Aliased strokes (the Pencil, the
        // Eraser's Pencil and Block modes) paint every pixel by the same amount, so they have no
        // gradient to band and are not dithered: a Pencil line at partial opacity is one flat
        // colour, as in Photoshop, not grain that a tolerance-0 Magic Wand or Paint Bucket trips on.
        let dither = fmt.sample == SampleType::U8 && !b.aliased;
        let mut dmg = Rect::EMPTY;
        let mut region = Vec::new();
        for (tx, ty) in keys {
            let Some(tile) = self.cov.tiles.get(&(tx, ty)) else { continue };
            let tr = Rect::new(tx * COV_TILE, ty * COV_TILE, (tx + 1) * COV_TILE, (ty + 1) * COV_TILE).intersect(&bounds);
            if tr.is_empty() {
                continue;
            }
            pre.read_region_into(tr, &mut region);
            let sel = selection.map(|s| (s.channels(), s.read_region(tr)));
            let w = tr.width() as usize;
            for y in tr.y0..tr.y1 {
                for x in tr.x0..tr.x1 {
                    let ti = ((y - ty * COV_TILE) * COV_TILE + (x - tx * COV_TILE)) as usize;
                    let c = tile.cov[ti];
                    if c <= 0.0 {
                        continue;
                    }
                    let i = (y - tr.y0) as usize * w + (x - tr.x0) as usize;
                    let m = self.ctx.stroke_mask_wet(c, tile.depth.get(ti).copied(), self.dual.as_ref().map(|d| d.get(x, y)), x, y);
                    let s = sel.as_ref().map_or(1.0, |(sc, v)| v[i * sc]);
                    let mut k = (m * opacity * s).min(1.0);
                    if k <= 0.0 {
                        continue;
                    }
                    if b.mode == BlendMode::Dissolve {
                        k = if hash2(x, y, b.seed) < k { 1.0 } else { 0.0 };
                        if k == 0.0 {
                            continue;
                        }
                    }
                    let px = &mut region[i * n..(i + 1) * n];
                    if b.erase {
                        if let (Some(a), false) = (a_idx, lock_transparency) {
                            px[a] *= 1.0 - k;
                        }
                        if dither {
                            dither8(px, x, y);
                        }
                        continue;
                    }
                    if lock_transparency && a_idx.is_some_and(|a| px[a] <= 0.0) {
                        continue;
                    }
                    if self.per_dab_color && !tile.col.is_empty() {
                        let p = &tile.col[ti * nc..(ti + 1) * nc];
                        for ch in 0..nc {
                            src[ch] = p[ch] / c;
                        }
                    }
                    if matches!(b.mode, BlendMode::Normal | BlendMode::Dissolve) {
                        over_native(&fmt, px, &src[..n], k, lock_transparency);
                    } else {
                        let d = to_rgba(&fmt, px);
                        let sr = to_rgba(&fmt, &src[..n]);
                        let mut o = photocraft_color::blend::composite(b.mode, d, sr, k);
                        if lock_transparency {
                            o[3] = d[3];
                        }
                        let mut enc = [0.0f32; 8];
                        from_rgba_into(&fmt, o, &mut enc);
                        px.copy_from_slice(&enc[..n]);
                    }
                    if dither {
                        dither8(px, x, y);
                    }
                }
            }
            target.write_region(tr, &region);
            dmg = dmg.union(&tr);
        }
        if all { bounds } else { dmg }
    }
}

/// Render a whole stroke onto `target` (one-shot). Returns the damaged rectangle.
pub fn render_stroke(
    target: &mut Surface,
    brush: &BrushSettings,
    points: &[StrokePoint],
    selection: Option<&Surface>,
    lock_transparency: bool,
    zoom: f32,
) -> Rect {
    if points.is_empty() {
        return Rect::EMPTY;
    }
    let pre = target.clone();
    let mut r = StrokeRenderer::new(brush, Some(target.format()), zoom);
    r.push(points);
    r.finish();
    r.composite(&pre, target, selection, lock_transparency, true)
}

// The zero-angle transform only changes signs; squaring gives the same radial distance.
#[inline(never)]
fn rasterize_round(d: &Dab, rect: Rect, out: &mut [f32], hardness: f32, aliased: bool) {
    let (cx, cy) = if aliased { grid_center(d.center.x, d.center.y, 2.0 * d.radius) } else { (d.center.x as f32, d.center.y as f32) };
    let ro = d.roundness.max(0.5 / d.radius).min(1.0);
    let rm = (d.radius * ro).max(0.5);
    let w = rect.width() as usize;
    let reach = if aliased { rm } else { tip_reach(rm, hardness).max(rm) };
    let reach2 = (reach + 0.5) * (reach + 0.5);
    for (yy, row) in out.chunks_exact_mut(w).enumerate() {
        let dy = (rect.y0 + yy as i32) as f32 + 0.5 - cy;
        if aliased {
            for (xx, dst) in row.iter_mut().enumerate() {
                let dx = (rect.x0 + xx as i32) as f32 + 0.5 - cx;
                let d2 = (dx * ro).powi(2) + dy * dy;
                *dst = if d2 < reach2 && d2.sqrt() <= rm { d.alpha } else { 0.0 };
            }
        } else {
            for (xx, dst) in row.iter_mut().enumerate() {
                let dx = (rect.x0 + xx as i32) as f32 + 0.5 - cx;
                let d2 = (dx * ro).powi(2) + dy * dy;
                let val = if d2 >= reach2 { 0.0 } else { tip_falloff(d2.sqrt(), rm, hardness) };
                *dst = if val <= 0.0 { 0.0 } else { val * d.alpha };
            }
        }
    }
}

#[cfg(test)]
#[path = "render_reference.rs"]
mod reference;

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_geom::Point;

    #[test]
    fn identical_quantized_parameters_hit_cache_and_return_identical_pointer() {
        let cache = DabCache::new(64);
        let key1 = DabKey::from_params(25.0, 0.5, 0.5, 1.0, 0.0);
        // 25.05 * 4 = 100.2 -> 100, hardness 0.501 -> 50, roundness 1.0 -> angle 0
        let key2 = DabKey::from_params(25.05, 0.501, 0.499, 1.0, 0.1);
        assert_eq!(key1, key2, "Parameters quantizing to same values must produce equal keys");

        let mut compute_count = 0;
        let mask1 = cache.get_or_create(&key1, || {
            compute_count += 1;
            rasterize_mask_for_key(&key1)
        });
        let mask2 = cache.get_or_create(&key2, || {
            compute_count += 1;
            rasterize_mask_for_key(&key2)
        });

        assert_eq!(compute_count, 1, "Rasterizer closure should only be called once on cache hit");
        assert!(Arc::ptr_eq(&mask1, &mask2), "Cache hit must return identical Arc pointer");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn different_radii_produce_cache_misses() {
        let cache = DabCache::new(64);
        let key1 = DabKey::from_params(25.0, 0.5, 0.5, 1.0, 0.0);
        let key2 = DabKey::from_params(30.0, 0.5, 0.5, 1.0, 0.0);
        assert_ne!(key1, key2, "Different radii must produce different keys");

        let mask1 = cache.get_or_create(&key1, || rasterize_mask_for_key(&key1));
        let mask2 = cache.get_or_create(&key2, || rasterize_mask_for_key(&key2));

        assert!(!Arc::ptr_eq(&mask1, &mask2), "Different keys must return distinct mask buffers");
        assert_eq!(cache.len(), 2);
        assert_ne!(mask1.len(), mask2.len(), "Different radii produce different mask dimensions");
    }

    #[test]
    fn bounded_lru_capacity_evicts_properly_without_memory_leaks() {
        let capacity = 5;
        let cache = DabCache::new(capacity);
        let mut keys = Vec::new();

        // Insert 10 different keys into a cache with capacity 5
        for i in 0..10 {
            let key = DabKey::new(i as u16 * 10 + 10, 50, 50, 100, 0);
            cache.get_or_create(&key, || rasterize_mask_for_key(&key));
            keys.push(key);
        }

        // Capacity bound must be strictly enforced
        assert_eq!(cache.len(), capacity, "Cache size must be strictly bounded to capacity");

        // First 5 entries (0..5) should have been evicted
        for key in &keys[0..5] {
            assert!(cache.get(key).is_none(), "Oldest key should have been evicted by LRU policy");
        }

        // Remaining 5 entries (5..10) should be present
        for key in &keys[5..10] {
            assert!(cache.get(key).is_some(), "Recent key should remain in cache");
        }

        // Access key 5 to refresh its recency
        let _ = cache.get(&keys[5]);

        // Insert an 11th key
        let new_key = DabKey::new(999, 50, 50, 100, 0);
        cache.get_or_create(&new_key, || rasterize_mask_for_key(&new_key));
        assert_eq!(cache.len(), capacity);

        // Key 5 was refreshed, so key 6 should have been evicted instead
        assert!(cache.get(&keys[5]).is_some(), "Refreshed key must not be evicted");
        assert!(cache.get(&keys[6]).is_none(), "Least recently used key must be evicted");

        // Clear cache and verify it empties properly
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn dab_500px_diameter_blits_cleanly_across_tiles_without_seams() {
        // 500px diameter -> radius 250px
        let radius = 250.0;
        let hardness = 0.5;
        let brush = BrushSettings {
            size: 500.0,
            hardness,
            spacing: 1.0,
            ..Default::default()
        };
        // Position straddling multiple 64x64 tile boundaries
        let center = Point::new(320.5, 320.5);
        let dab = Dab::round(center, radius, 1.0);

        let ctx = BrushContext::new(&brush);
        let rect = ctx.dab_rect(&dab, false);

        let mut cov_map = CoverageMap::new(0);
        let key = DabKey::from_dab(&dab, hardness);
        let mask = ctx.dab_cache.get_mask_or_create(&key, || rasterize_mask_for_key(&key));

        let t0x = rect.x0.div_euclid(COV_TILE);
        let t1x = (rect.x1 - 1).div_euclid(COV_TILE);
        let t0y = rect.y0.div_euclid(COV_TILE);
        let t1y = (rect.y1 - 1).div_euclid(COV_TILE);
        assert!(t1x - t0x >= 7, "500px dab must span across at least 8 tiles horizontally");
        assert!(t1y - t0y >= 7, "500px dab must span across at least 8 tiles vertically");

        cov_map.accumulate_dab(rect, &mask.data, 1.0, 1.0, None, None);
        assert!(cov_map.tiles_count() >= 40, "500px dab should occupy dozens of 64x64 tiles");

        // Verify seam-free boundary continuity across tile edges
        for y in (rect.y0 + 50..rect.y1 - 50).step_by(32) {
            for seam_x in [127, 191, 255, 319, 383, 447] {
                if seam_x > rect.x0 && seam_x + 1 < rect.x1 - 1 {
                    let c_left = cov_map.get(seam_x, y);
                    let c_right = cov_map.get(seam_x + 1, y);
                    if c_left > 0.05 || c_right > 0.05 {
                        let diff = (c_left - c_right).abs();
                        assert!(
                            diff < 0.1,
                            "Tile boundary seam artifact detected at ({seam_x}, {y}): left={c_left}, right={c_right}, diff={diff}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn vectorized_blitting_output_matches_reference_falloff() {
        let radius = 60.0;
        let hardness = 0.5;
        let key = DabKey::from_params(radius, hardness, 1.0 - hardness, 1.0, 0.0);
        let mask = rasterize_mask_for_key(&key);

        let center_x = 128;
        let center_y = 128;
        let dab = Dab::round(Point::new(center_x as f64, center_y as f64), radius, 0.75);
        let rr = (mask.size as i32 - 1) / 2;
        let rect = Rect::new(center_x - rr, center_y - rr, center_x + rr + 1, center_y + rr + 1);

        let mut cov_map = CoverageMap::new(0);
        cov_map.accumulate_dab(rect, &mask.data, dab.alpha, dab.opacity, None, None);

        // Check falloff at multiple radial distances from center
        for r in [0, 5, 15, 25, 35, 45, 55, 60, 65] {
            let x = center_x + r;
            let y = center_y;
            let actual = cov_map.get(x, y);
            let expected_val = tip_falloff(r as f32, radius, hardness) * dab.alpha;
            let expected = expected_val.min(dab.opacity);
            let diff = (actual - expected).abs();
            assert!(
                diff < 0.02,
                "Falloff mismatch at radius {r}: actual={actual}, expected={expected}, diff={diff}"
            );
        }
    }
}
