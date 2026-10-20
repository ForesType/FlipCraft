<div align="center">

<img src="docs/assets/logo.svg" alt="FlipCraft Logo" width="112" height="112">

# FlipCraft

**PhotoCraft but refined for digital artists.**

[![Version](https://img.shields.io/badge/version-0.15.0-blue.svg)](https://github.com/ForesType/FlipCraft/releases)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Pure Rust](https://img.shields.io/badge/rust-100%25%20safe%20%7C%20wgpu-orange.svg)](https://www.rust-lang.org/)
[![No Electron](https://img.shields.io/badge/bloat-zero%20(No%20Electron%2FTauri)-success.svg)](#features)

<p align="center">
  <b>FlipCraft</b> is a fast, distraction-free painting app built in pure Rust.<br>
  It takes the lightweight foundation of PhotoCraft and elevates it specifically for artists — featuring an upgraded, natural brush engine, razor-sharp tapering, and native 16k/8k tablet support.
</p>

</div>

---

## 🎨 The Brush Difference

<div align="center">

<table>
  <thead>
    <tr>
      <th align="center" width="50%">Before (Original PhotoCraft)</th>
      <th align="center" width="50%">After (FlipCraft)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">
        <img src="docs/assets/before_brush.png" alt="Before Brush Engine Overhaul" width="440" />
      </td>
      <td align="center">
        <img src="docs/assets/after_brush.png" alt="After Brush Engine Overhaul" width="440" />
      </td>
    </tr>
    <tr>
      <td align="center"><i>Blunt ends, stair-stepping & choppy flick endings</i></td>
      <td align="center"><i>Natural, needle-sharp tapering & smooth curves</i></td>
    </tr>
  </tbody>
</table>

</div>

---

## ✨ Why Artists Choose FlipCraft

* 🖋️ **Silky-Smooth Strokes** — Advanced curve dynamics eliminate blunt stroke cutoffs and dab gaps during quick pen flicks.
* ✍️ **Native 16k & 8k Tablet Support** — Direct Windows Ink integration with true linear pressure tracking (Wacom, Huion, XPPen, Surface), tilt, rotation, and physical eraser tip flip.
* ⚡ **Pure Rust Performance** — Zero Electron or Chromium bloat. Launches instantly and maintains 60+ FPS on large brushes.
* 🎯 **Photoshop-Familiar Canvas** — Clutter-free UI with standard Photoshop navigation, blending modes, and keybindings so you can jump right in.
* 🔧 **Sensible Defaults** — Catch-up on stroke end is turned off by default for natural, responsive line work.

### 🧰 Comprehensive Toolset

<table>
  <tr>
    <td valign="top">
      <h4>🧰 67 tools</h4>
Move · Artboard · Rectangular and Elliptical Marquee · Single Row Marquee · Single Column Marquee · Lasso · Polygonal Lasso · Magnetic Lasso · Object Selection · Quick Selection · Magic Wand · Crop · Perspective Crop · Slice · Slice Select · Frame · Eyedropper · Color Sampler · Ruler · Note · Count · Remove · Spot Healing Brush · Healing Brush · Patch · Content-Aware Move · Red Eye · Brush · Pencil · Mixer Brush · Color Replacement · Clone Stamp · Pattern Stamp · History Brush · Eraser · Background Eraser · Magic Eraser · Gradient · Paint Bucket · Blur · Sharpen · Smudge · Dodge · Burn · Sponge · Pen · Freeform Pen · Curvature Pen · Add Anchor Point · Delete Anchor Point · Convert Point · Horizontal and Vertical Type · Horizontal and Vertical Type Mask · Path Selection · Direct Selection · Rectangle · Ellipse · Triangle · Polygon · Line · Custom Shape · Hand · Rotate View · Zoom
    </td>
  </tr>
</table>

*(Looking for the deep technical brush math? See the [Brush Engine Deep Dive](docs/BRUSH_ENGINE_DEEP_DIVE.md).)*

---

## 📥 Downloads

Ready-to-use installable and portable packages are available on our **[Releases Page](https://github.com/ForesType/FlipCraft/releases)**:

* 🪟 **Windows**: Portable `.zip` & `.msi` Installer (x64 & ARM64)
* 🐧 **Linux**: Standalone `.AppImage`, `.deb`, & `.tar.gz`
* 🍎 **macOS**: Universal `.dmg` Disk Image

---

## 💖 Support Development & Priority Requests

FlipCraft is independent open source. Donations directly fund GPU compute for AI tools and active engine development:

* **USDT (TRC-20 / Tron)**:
  `TFYNjg8Uob2mz84Q1yAV2RpHkhjJXafGq5`

* **USDT (BEP-20 / BNB Chain)**:
  `0x9428E0499C830A2d26A4A229Bc5b6e88184Fa960`

> *Disclaimer: Donations fund compute and ongoing development. Sponsored feature proposals receive top priority in development queues while maintaining FlipCraft's vision and quality.*

👉 **[Submit a Priority Feature Request](.github/ISSUE_TEMPLATE/priority_request.yml)** *(include your TxID)*

---

## 📄 License & Credits

FlipCraft is dual-licensed under **MIT OR Apache-2.0**.

FlipCraft is an independent, artist-focused fork of [**PhotoCraft**](https://github.com/storytold/photocraft) by Storyteller / Storytold. We gratefully acknowledge PhotoCraft for the core foundation. See [ATTRIBUTION.md](ATTRIBUTION.md) and [NOTICE](NOTICE) for full details.
