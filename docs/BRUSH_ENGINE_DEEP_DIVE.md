# FlipCraft Brush Engine & Digitizer Architecture — Deep Dive

This document details the internal mathematics, spline interpolation, and hardware digitizer pipelines implemented in FlipCraft.

---

## 1. Adaptive Distance Stepping & Dynamic Spacing
Traditional digital painting engines suffer from dab starvation: when stylus pressure drops rapidly during quick flick strokes or line-art tapering, fixed distance accumulation skips segments, producing jagged, blunt cutoffs.

FlipCraft incorporates an adaptive distance accumulator modeled after Krita's `KisDistanceInformation`:
- As stylus pressure falls and required dab spacing shrinks, the accumulator triggers dabs instantaneously ($t = 0.0$) rather than skipping forward.
- Ensures seamless, dense, and continuous stroke continuity during lightning-fast pen flicks.
- Eliminates polygon stair-stepping and empty gaps across rapid hand motions.

---

## 2. Catmull-Rom Bézier Interpolation & Needle-Sharp Tapering
- **Curvature Continuity**: Stylus pointer events are fitted with smooth cubic Bézier splines using Catmull-Rom velocity tangents, subdivided recursively with De Casteljau midpoints (~0.5 px flatness). Fast flicks follow clean, flowing curves rather than polyline line segments.
- **Tail Aggressiveness**: A specialized pen-lift convergence filter dynamically scales spatial filter weights as stylus pressure decreases ($dP/dt < 0$), pulling the stroke rapidly and precisely into the liftoff point. Every stroke resolves to a razor-sharp, needle-thin finish.

---

## 3. Native 16k & 8k Windows Ink Digitizer Pipeline
- Direct Win32 `WM_POINTER` integration bypassing legacy WinTab abstractions.
- Full hardware dynamic range utilization for **8,192** and **16,384** pressure-level digitizers (Wacom Pro Pen 2/3, Huion, XPPen, Surface Pen) with true linear tracking and zero premature saturation.
- Real-time hardware physical eraser tip flip detection, tilt X/Y ($-90^\circ \dots +90^\circ$), and barrel rotation ($0^\circ \dots 359^\circ$) without interference from synthetic touch gestures.
- Event compression and deduplication preventing 240Hz+ high-frequency digitizer packet flooding.

---

## 4. High-Performance GPU Compositing & Bounded Dab Caching
- **Thread-safe Dab Cache**: Pre-rendered floating-point coverage masks indexed by discretized radius, softness, and hardness.
- Avoids costly per-pixel radial equations on the CPU, blitting directly into tile buffers with SIMD/vectorized coverage blending (`COV_TILE = 64`).
- Rock-solid 60+ FPS responsiveness even on ultra-large brushes (up to 500px diameter).
