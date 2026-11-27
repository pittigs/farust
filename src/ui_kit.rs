//! Phase 20 front-end UI kit shared by the main menu, options, replay browser and lobbies.
//!
//! - [`Fonts`]: glyph atlas baked from the FA install's TrueType fonts (`fonts/ARIAL*.TTF`,
//!   `fonts/zeroes_3.ttf`) with a fallback to the HUD 3x5 glyph font.
//! - [`ImageStore`]: named RGBA images (FA menu textures from `textures.scd`, map previews).
//! - [`Canvas`]: immediate-mode 2D draw list (rects, text, images, widgets) in window pixels,
//!   batched per texture in painter's order.
//! - [`UiRenderer`]: one textured-quad pipeline drawing a canvas to any view; used by the
//!   window ([`WindowGpu`]) and offscreen PNG snapshots ([`render_png`]).

use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;

use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use crate::dds_decode::{RgbaImage, decode_dds_rgba8};
use crate::hud::draw_text;
use crate::ttf::{AtlasFace, Font, build_atlas};
use crate::viewer::HudVertex;

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.w && py >= self.y && py <= self.y + self.h
    }

    pub fn right(&self) -> f32 {
        self.x + self.w
    }

    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }

    pub fn inset(&self, d: f32) -> Self {
        Self::new(self.x + d, self.y + d, (self.w - 2.0 * d).max(0.0), (self.h - 2.0 * d).max(0.0))
    }

    #[allow(dead_code)]
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w * 0.5, self.y + self.h * 0.5)
    }
}

/// First widget whose rect contains the point (later widgets win ties, like painting).
#[allow(dead_code)]
pub fn hit_test<A: Copy>(widgets: &[(Rect, A)], x: f32, y: f32) -> Option<A> {
    widgets.iter().rev().find(|(r, _)| r.contains(x, y)).map(|(_, a)| *a)
}

// ---------------------------------------------------------------------------
// Theme (faction-neutral FA look: gunmetal panels, cyan energy accents, gold titles)
// ---------------------------------------------------------------------------

pub mod theme {
    pub const TEXT: [f32; 4] = [0.88, 0.91, 0.93, 1.0];
    pub const DIM: [f32; 4] = [0.56, 0.63, 0.69, 1.0];
    pub const FAINT: [f32; 4] = [0.36, 0.41, 0.46, 1.0];
    pub const ACCENT: [f32; 4] = [0.47, 0.86, 0.96, 1.0];
    pub const GOLD: [f32; 4] = [0.95, 0.82, 0.48, 1.0];
    pub const GOOD: [f32; 4] = [0.42, 0.86, 0.42, 1.0];
    pub const BAD: [f32; 4] = [0.95, 0.38, 0.30, 1.0];
    pub const PANEL: [f32; 4] = [0.035, 0.05, 0.07, 0.90];
    pub const PANEL_HI: [f32; 4] = [0.07, 0.10, 0.13, 0.92];
    pub const BORDER: [f32; 4] = [0.27, 0.34, 0.40, 1.0];
    pub const ROW_SEL: [f32; 4] = [0.10, 0.30, 0.38, 0.95];
    pub const ROW_HOVER: [f32; 4] = [0.09, 0.15, 0.19, 0.95];
    pub const BTN: [f32; 4] = [0.08, 0.11, 0.14, 0.96];
    pub const BTN_HOVER: [f32; 4] = [0.12, 0.22, 0.27, 0.98];
    pub const BTN_PRIMARY: [f32; 4] = [0.08, 0.32, 0.18, 0.97];
    pub const FIELD: [f32; 4] = [0.015, 0.02, 0.03, 0.95];
    pub const SLOT_COLORS: [[f32; 4]; 8] = [
        [0.22, 0.56, 0.98, 1.0],
        [0.94, 0.24, 0.22, 1.0],
        [0.20, 0.84, 0.44, 1.0],
        [0.96, 0.80, 0.22, 1.0],
        [0.80, 0.40, 0.95, 1.0],
        [0.98, 0.55, 0.15, 1.0],
        [0.30, 0.90, 0.92, 1.0],
        [0.90, 0.90, 0.90, 1.0],
    ];
}

// ---------------------------------------------------------------------------
// Phase 27 Classic FA Faction Palettes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassicPalette {
    pub name: &'static str,
    pub panel_top: [f32; 4],
    pub panel_bot: [f32; 4],
    pub header_bg: [f32; 4],
    pub border_outer: [f32; 4],
    pub border_inner: [f32; 4],
    pub bracket: [f32; 4],
    pub btn: [f32; 4],
    pub btn_hover: [f32; 4],
    pub btn_active: [f32; 4],
    pub accent: [f32; 4],
    pub title_gold: [f32; 4],
}

impl ClassicPalette {
    pub fn for_faction_key(key: &str) -> Self {
        match key.trim().to_ascii_lowercase().as_str() {
            "aeon" => Self {
                name: "AEON",
                panel_top: [0.05, 0.15, 0.10, 0.94],
                panel_bot: [0.02, 0.07, 0.05, 0.95],
                header_bg: [0.08, 0.25, 0.16, 0.96],
                border_outer: [0.38, 0.84, 0.56, 1.0],
                border_inner: [0.62, 0.56, 0.24, 0.85],
                bracket: [0.94, 0.84, 0.44, 0.95],
                btn: [0.06, 0.18, 0.12, 0.96],
                btn_hover: [0.11, 0.30, 0.20, 0.98],
                btn_active: [0.16, 0.46, 0.30, 0.98],
                accent: [0.52, 0.95, 0.68, 1.0],
                title_gold: [0.98, 0.90, 0.54, 1.0],
            },
            "cybran" => Self {
                name: "CYBRAN",
                panel_top: [0.15, 0.05, 0.05, 0.94],
                panel_bot: [0.07, 0.02, 0.02, 0.95],
                header_bg: [0.28, 0.07, 0.07, 0.96],
                border_outer: [0.90, 0.24, 0.18, 1.0],
                border_inner: [0.48, 0.12, 0.10, 0.85],
                bracket: [1.0, 0.42, 0.32, 0.95],
                btn: [0.18, 0.06, 0.06, 0.96],
                btn_hover: [0.32, 0.10, 0.10, 0.98],
                btn_active: [0.52, 0.14, 0.12, 0.98],
                accent: [1.0, 0.38, 0.28, 1.0],
                title_gold: [1.0, 0.78, 0.36, 1.0],
            },
            "seraphim" => Self {
                name: "SERAPHIM",
                panel_top: [0.15, 0.11, 0.04, 0.94],
                panel_bot: [0.07, 0.05, 0.02, 0.95],
                header_bg: [0.28, 0.20, 0.06, 0.96],
                border_outer: [0.96, 0.74, 0.22, 1.0],
                border_inner: [0.54, 0.38, 0.12, 0.85],
                bracket: [1.0, 0.86, 0.42, 0.95],
                btn: [0.18, 0.13, 0.05, 0.96],
                btn_hover: [0.32, 0.23, 0.08, 0.98],
                btn_active: [0.50, 0.36, 0.10, 0.98],
                accent: [1.0, 0.82, 0.32, 1.0],
                title_gold: [1.0, 0.90, 0.50, 1.0],
            },
            _ => Self {
                name: "UEF",
                panel_top: [0.07, 0.13, 0.22, 0.94],
                panel_bot: [0.03, 0.06, 0.11, 0.95],
                header_bg: [0.10, 0.22, 0.38, 0.96],
                border_outer: [0.28, 0.62, 0.94, 1.0],
                border_inner: [0.14, 0.32, 0.52, 0.85],
                bracket: [0.52, 0.82, 1.0, 0.95],
                btn: [0.08, 0.16, 0.26, 0.96],
                btn_hover: [0.14, 0.28, 0.46, 0.98],
                btn_active: [0.18, 0.44, 0.72, 0.98],
                accent: [0.45, 0.82, 1.0, 1.0],
                title_gold: [0.98, 0.86, 0.44, 1.0],
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Fonts
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontKind {
    Body,
    Bold,
    Title,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextStyle {
    pub font: FontKind,
    pub px: f32,
}

pub const fn body(px: f32) -> TextStyle {
    TextStyle { font: FontKind::Body, px }
}

pub const fn bold(px: f32) -> TextStyle {
    TextStyle { font: FontKind::Bold, px }
}

pub const fn title(px: f32) -> TextStyle {
    TextStyle { font: FontKind::Title, px }
}

pub struct Fonts {
    pub atlas: RgbaImage,
    faces: Vec<(FontKind, AtlasFace)>,
    white_uv: [f32; 2],
}

impl Fonts {
    /// HUD glyph font only (no TrueType faces).
    pub fn fallback() -> Self {
        Self {
            atlas: RgbaImage { width: 4, height: 4, pixels: vec![255; 64] },
            faces: Vec::new(),
            white_uv: [0.5, 0.5],
        }
    }

    /// Bakes the FA install's fonts; falls back to the glyph font when they are missing.
    pub fn load(game_dir: &Path) -> Self {
        let dir = game_dir.join("fonts");
        let files: Vec<_> = std::fs::read_dir(&dir)
            .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).collect())
            .unwrap_or_default();
        let load = |name: &str| -> Option<Font> {
            let p = files.iter().find(|p| {
                p.file_name().is_some_and(|f| f.to_string_lossy().eq_ignore_ascii_case(name))
            })?;
            Font::parse(std::fs::read(p).ok()?).ok()
        };
        let Some(body_f) = load("ARIAL.TTF") else {
            return Self::fallback();
        };
        let bold_f = load("ARIALBD.TTF");
        let title_f = load("zeroes_3.ttf");
        let bold_ref = bold_f.as_ref().unwrap_or(&body_f);
        let title_ref = title_f.as_ref().unwrap_or(bold_ref);
        let mut specs: Vec<(FontKind, &Font, f32)> = Vec::new();
        for px in [11.0, 13.0, 15.0, 17.0, 20.0] {
            specs.push((FontKind::Body, &body_f, px));
        }
        for px in [12.0, 14.0, 16.0, 19.0, 22.0, 26.0] {
            specs.push((FontKind::Bold, bold_ref, px));
        }
        for px in [16.0, 20.0, 26.0, 34.0, 46.0] {
            specs.push((FontKind::Title, title_ref, px));
        }
        let atlas_specs: Vec<(&Font, f32)> = specs.iter().map(|(_, f, px)| (*f, *px)).collect();
        let (atlas, faces, white_uv) = build_atlas(&atlas_specs, 1024);
        Self {
            atlas,
            faces: specs.iter().map(|(k, _, _)| *k).zip(faces).collect(),
            white_uv,
        }
    }

    #[allow(dead_code)]
    pub fn has_faces(&self) -> bool {
        !self.faces.is_empty()
    }

    fn face(&self, kind: FontKind, px: f32) -> Option<&AtlasFace> {
        let pick = |want: Option<FontKind>| {
            self.faces
                .iter()
                .filter(|(k, _)| want.is_none_or(|w| *k == w))
                .min_by(|a, b| (a.1.px - px).abs().total_cmp(&(b.1.px - px).abs()))
                .map(|(_, f)| f)
        };
        pick(Some(kind)).or_else(|| pick(None))
    }

    fn fallback_scale(px: f32) -> f32 {
        (px / 7.0).round().max(1.0)
    }

    pub fn measure(&self, st: TextStyle, s: &str) -> f32 {
        match self.face(st.font, st.px) {
            Some(face) => {
                let k = st.px / face.px;
                s.chars().map(|c| face.glyph(c).advance * k).sum()
            }
            None => s.len() as f32 * 4.0 * Self::fallback_scale(st.px),
        }
    }

    /// Visual line height (ascent + descent) at `st`.
    pub fn line_height(&self, st: TextStyle) -> f32 {
        match self.face(st.font, st.px) {
            Some(face) => (face.ascent + face.descent) * st.px / face.px,
            None => 5.0 * Self::fallback_scale(st.px),
        }
    }

    /// Truncates `s` with "..." so it fits `max_w` pixels.
    pub fn fit(&self, st: TextStyle, s: &str, max_w: f32) -> String {
        if self.measure(st, s) <= max_w {
            return s.to_string();
        }
        let mut out = String::new();
        for c in s.chars() {
            out.push(c);
            if self.measure(st, &format!("{out}...")) > max_w {
                out.pop();
                break;
            }
        }
        out.push_str("...");
        out
    }
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct ImageStore {
    images: HashMap<String, Arc<RgbaImage>>,
}

impl ImageStore {
    pub fn insert(&mut self, name: &str, img: RgbaImage) {
        self.images.insert(name.to_string(), Arc::new(img));
    }

    pub fn get(&self, name: &str) -> Option<&Arc<RgbaImage>> {
        self.images.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.images.contains_key(name)
    }

    /// Loads DDS textures from `gamedata/textures.scd` in one archive pass:
    /// `wanted` maps store names to inner paths (case-insensitive). Returns how many loaded.
    pub fn load_scd_textures(&mut self, game_dir: &Path, wanted: &[(&str, &str)]) -> usize {
        let scd = crate::scd::gamedata_dir(game_dir).join("textures.scd");
        let lower: Vec<String> = wanted.iter().map(|(_, p)| p.to_ascii_lowercase()).collect();
        let Ok(files) = crate::scd::read_scd_matching(&scd, |n| lower.iter().any(|w| n.eq_ignore_ascii_case(w))) else {
            return 0;
        };
        let mut n = 0;
        for (path, bytes) in files {
            let Ok(img) = decode_dds_rgba8(&bytes) else {
                continue;
            };
            let img = Arc::new(img);
            for (name, inner) in wanted {
                if inner.eq_ignore_ascii_case(&path) {
                    self.images.insert(name.to_string(), img.clone());
                    n += 1;
                }
            }
        }
        n
    }
}

// ---------------------------------------------------------------------------
// Canvas
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tex {
    Atlas,
    Image(String),
}

pub struct Batch {
    pub tex: Tex,
    pub verts: Vec<HudVertex>,
    pub idx: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Center,
    Right,
}

/// Visual state of a clickable widget.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WState {
    pub hover: bool,
    pub active: bool,
    pub disabled: bool,
    pub primary: bool,
}

pub struct Canvas<'a> {
    pub w: f32,
    pub h: f32,
    pub fonts: &'a Fonts,
    pub images: &'a ImageStore,
    pub hover: Option<(f32, f32)>,
    pub classic: Option<ClassicPalette>,
    pub batches: Vec<Batch>,
}

impl<'a> Canvas<'a> {
    pub fn new(w: f32, h: f32, fonts: &'a Fonts, images: &'a ImageStore, hover: Option<(f32, f32)>) -> Self {
        let classic = if crate::settings::ui_theme() == crate::settings::UiTheme::Classic {
            Some(ClassicPalette::for_faction_key(crate::settings::input_settings().faction_key()))
        } else {
            None
        };
        Self { w, h, fonts, images, hover, classic, batches: Vec::new() }
    }

    pub fn set_classic(&mut self, pal: Option<ClassicPalette>) {
        self.classic = pal;
    }

    pub fn is_hover(&self, r: Rect) -> bool {
        self.hover.is_some_and(|(x, y)| r.contains(x, y))
    }

    fn batch(&mut self, tex: Tex) -> &mut Batch {
        if self.batches.last().is_none_or(|b| b.tex != tex) {
            self.batches.push(Batch { tex, verts: Vec::new(), idx: Vec::new() });
        }
        self.batches.last_mut().expect("batch")
    }

    fn ndc(&self, x: f32, y: f32) -> [f32; 2] {
        [2.0 * x / self.w - 1.0, 1.0 - 2.0 * y / self.h]
    }

    /// Quad with per-corner colors (top-left, top-right, bottom-right, bottom-left).
    pub fn quad4(&mut self, tex: Tex, r: Rect, uv: [f32; 4], c: [[f32; 4]; 4]) {
        let p = [
            self.ndc(r.x, r.y),
            self.ndc(r.x + r.w, r.y),
            self.ndc(r.x + r.w, r.y + r.h),
            self.ndc(r.x, r.y + r.h),
        ];
        let b = self.batch(tex);
        let base = b.verts.len() as u32;
        b.verts.extend_from_slice(&[
            HudVertex { pos: p[0], uv: [uv[0], uv[1]], color: c[0] },
            HudVertex { pos: p[1], uv: [uv[2], uv[1]], color: c[1] },
            HudVertex { pos: p[2], uv: [uv[2], uv[3]], color: c[2] },
            HudVertex { pos: p[3], uv: [uv[0], uv[3]], color: c[3] },
        ]);
        b.idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    fn white_uv(&self) -> [f32; 4] {
        let [u, v] = self.fonts.white_uv;
        [u, v, u, v]
    }

    pub fn rect(&mut self, r: Rect, color: [f32; 4]) {
        let uv = self.white_uv();
        self.quad4(Tex::Atlas, r, uv, [color; 4]);
    }

    pub fn gradient_v(&mut self, r: Rect, top: [f32; 4], bottom: [f32; 4]) {
        let uv = self.white_uv();
        self.quad4(Tex::Atlas, r, uv, [top, top, bottom, bottom]);
    }

    pub fn gradient_h(&mut self, r: Rect, left: [f32; 4], right: [f32; 4]) {
        let uv = self.white_uv();
        self.quad4(Tex::Atlas, r, uv, [left, right, right, left]);
    }

    pub fn outline(&mut self, r: Rect, t: f32, color: [f32; 4]) {
        self.rect(Rect::new(r.x, r.y, r.w, t), color);
        self.rect(Rect::new(r.x, r.bottom() - t, r.w, t), color);
        self.rect(Rect::new(r.x, r.y + t, t, r.h - 2.0 * t), color);
        self.rect(Rect::new(r.right() - t, r.y + t, t, r.h - 2.0 * t), color);
    }

    /// FA-style corner brackets.
    pub fn brackets(&mut self, r: Rect, len: f32, t: f32, color: [f32; 4]) {
        let l = len.min(r.w * 0.5).min(r.h * 0.5);
        for (cx, cy, sx, sy) in [
            (r.x, r.y, 1.0, 1.0),
            (r.right(), r.y, -1.0, 1.0),
            (r.right(), r.bottom(), -1.0, -1.0),
            (r.x, r.bottom(), 1.0, -1.0),
        ] {
            let hx = if sx > 0.0 { cx } else { cx - l };
            let vy = if sy > 0.0 { cy } else { cy - l };
            let hy = if sy > 0.0 { cy } else { cy - t };
            let vx = if sx > 0.0 { cx } else { cx - t };
            self.rect(Rect::new(hx, hy, l, t), color);
            self.rect(Rect::new(vx, vy, t, l), color);
        }
    }

    pub fn has_image(&self, name: &str) -> bool {
        self.images.contains(name)
    }

    pub fn image_size(&self, name: &str) -> Option<(f32, f32)> {
        self.images.get(name).map(|i| (i.width as f32, i.height as f32))
    }

    /// Draws a stored image; returns `false` (drawing nothing) when it is missing.
    pub fn image(&mut self, name: &str, r: Rect, color: [f32; 4]) -> bool {
        self.image_uv(name, r, [0.0, 0.0, 1.0, 1.0], color)
    }

    pub fn image_uv(&mut self, name: &str, r: Rect, uv: [f32; 4], color: [f32; 4]) -> bool {
        if !self.images.contains(name) {
            return false;
        }
        self.quad4(Tex::Image(name.to_string()), r, uv, [color; 4]);
        true
    }

    /// Image scaled to cover `r` (cropped, aspect preserved).
    pub fn image_cover(&mut self, name: &str, r: Rect, color: [f32; 4]) -> bool {
        let Some((iw, ih)) = self.image_size(name) else {
            return false;
        };
        let (ra, ia) = (r.w / r.h.max(1.0), iw / ih.max(1.0));
        let uv = if ia > ra {
            let f = ra / ia;
            [(1.0 - f) * 0.5, 0.0, (1.0 + f) * 0.5, 1.0]
        } else {
            let f = ia / ra;
            [0.0, (1.0 - f) * 0.5, 1.0, (1.0 + f) * 0.5]
        };
        self.image_uv(name, r, uv, color)
    }

    pub fn measure(&self, st: TextStyle, s: &str) -> f32 {
        self.fonts.measure(st, s)
    }

    /// Draws `s` with its top-left at `(x, y)`; returns the advance width.
    pub fn text(&mut self, x: f32, y: f32, st: TextStyle, s: &str, color: [f32; 4]) -> f32 {
        let fonts = self.fonts;
        if let Some(face) = fonts.face(st.font, st.px) {
            let k = st.px / face.px;
            let base = (y + face.ascent * k).round();
            let mut pen = x;
            for ch in s.chars() {
                let g = *face.glyph(ch);
                if g.width > 0.0 {
                    let r = Rect::new((pen + g.left * k).round(), base + g.top * k, g.width * k, g.height * k);
                    self.quad4(Tex::Atlas, r, g.uv, [color; 4]);
                }
                pen += g.advance * k;
            }
            pen - x
        } else {
            let scale = Fonts::fallback_scale(st.px);
            let (mut v, mut i) = (Vec::new(), Vec::new());
            draw_text(&mut v, &mut i, x.round(), y.round(), scale, s, color, self.w, self.h);
            let uv = self.fonts.white_uv;
            let b = self.batch(Tex::Atlas);
            let base = b.verts.len() as u32;
            b.verts.extend(v.into_iter().map(|mut vx| {
                vx.uv = uv;
                vx
            }));
            b.idx.extend(i.into_iter().map(|ix| ix + base));
            s.len() as f32 * 4.0 * scale
        }
    }

    /// Text vertically centered in `r` with horizontal alignment.
    pub fn text_in(&mut self, r: Rect, st: TextStyle, s: &str, color: [f32; 4], align: Align) {
        let tw = self.measure(st, s);
        let lh = self.fonts.line_height(st);
        let x = match align {
            Align::Left => r.x,
            Align::Center => r.x + (r.w - tw) * 0.5,
            Align::Right => r.right() - tw,
        };
        self.text(x, r.y + (r.h - lh) * 0.5, st, s, color);
    }

    // ---- widgets -------------------------------------------------------------

    /// Dark translucent panel with border, top highlight and corner brackets.
    pub fn panel(&mut self, r: Rect) {
        if let Some(pal) = self.classic {
            self.gradient_v(r, pal.panel_top, pal.panel_bot);
            self.outline(r, 2.0, pal.border_outer);
            self.outline(r.inset(3.0), 1.0, pal.border_inner);
            self.rect(Rect::new(r.x + 2.0, r.y + 2.0, r.w - 4.0, 1.0), pal.accent);
            self.brackets(r.inset(-2.0), 18.0, 2.5, pal.bracket);
            return;
        }
        self.gradient_v(r, theme::PANEL_HI, theme::PANEL);
        self.outline(r, 1.0, theme::BORDER);
        self.rect(Rect::new(r.x + 1.0, r.y + 1.0, r.w - 2.0, 1.0), [0.45, 0.55, 0.62, 0.35]);
        self.brackets(r.inset(-3.0), 14.0, 2.0, [0.50, 0.62, 0.70, 0.9]);
    }

    /// Panel with a header strip and a gold title.
    pub fn titled_panel(&mut self, r: Rect, title_text: &str) -> Rect {
        self.panel(r);
        let hdr = Rect::new(r.x + 1.0, r.y + 1.0, r.w - 2.0, 30.0);
        if let Some(pal) = self.classic {
            let chdr = Rect::new(r.x + 2.0, r.y + 2.0, r.w - 4.0, 30.0);
            self.gradient_h(chdr, pal.header_bg, pal.panel_bot);
            self.rect(Rect::new(r.x + 2.0, chdr.bottom(), r.w - 4.0, 1.5), pal.border_outer);
            self.text_in(Rect::new(r.x + 14.0, chdr.y, hdr.w - 20.0, chdr.h), bold(16.0), title_text, pal.title_gold, Align::Left);
            return Rect::new(r.x + 12.0, chdr.bottom() + 10.0, r.w - 24.0, r.h - chdr.h - 22.0);
        }
        self.gradient_h(hdr, [0.10, 0.17, 0.21, 0.95], [0.04, 0.06, 0.08, 0.0]);
        self.rect(Rect::new(r.x + 1.0, hdr.bottom(), r.w - 2.0, 1.0), [0.47, 0.86, 0.96, 0.35]);
        self.text_in(Rect::new(r.x + 14.0, hdr.y, hdr.w - 20.0, hdr.h), bold(16.0), title_text, theme::GOLD, Align::Left);
        Rect::new(r.x + 12.0, hdr.bottom() + 10.0, r.w - 24.0, r.h - hdr.h - 22.0)
    }

    pub fn button(&mut self, r: Rect, label: &str, st: WState) {
        if let Some(pal) = self.classic {
            let bg = if st.disabled {
                [0.05, 0.06, 0.07, 0.9]
            } else if st.primary {
                if st.hover { pal.btn_active } else { pal.header_bg }
            } else if st.hover || st.active {
                pal.btn_hover
            } else {
                pal.btn
            };
            let top = [bg[0] * 1.45 + 0.03, bg[1] * 1.45 + 0.03, bg[2] * 1.45 + 0.03, bg[3]];
            self.gradient_v(r, top, bg);
            let border = if st.disabled {
                [0.18, 0.20, 0.22, 1.0]
            } else if st.hover || st.active {
                pal.bracket
            } else {
                pal.border_outer
            };
            self.outline(r, 1.5, border);
            self.outline(r.inset(2.0), 1.0, pal.border_inner);
            if !st.disabled && (st.hover || st.active || st.primary) {
                self.rect(Rect::new(r.x + 3.0, r.y + 2.0, r.w - 6.0, 1.0), pal.accent);
            }
            let color = if st.disabled { theme::FAINT } else if st.primary { pal.title_gold } else { theme::TEXT };
            let px = (r.h * 0.48).clamp(12.0, 22.0);
            let label = self.fonts.fit(bold(px), label, r.w - 12.0);
            self.text_in(r, bold(px), &label, color, Align::Center);
            return;
        }
        let bg = if st.disabled {
            [0.05, 0.06, 0.07, 0.9]
        } else if st.primary {
            if st.hover { [0.12, 0.45, 0.25, 0.98] } else { theme::BTN_PRIMARY }
        } else if st.hover || st.active {
            theme::BTN_HOVER
        } else {
            theme::BTN
        };
        let top = [bg[0] * 1.6 + 0.02, bg[1] * 1.6 + 0.02, bg[2] * 1.6 + 0.02, bg[3]];
        self.gradient_v(r, top, bg);
        let border = if st.disabled {
            [0.18, 0.20, 0.22, 1.0]
        } else if st.hover || st.active {
            theme::ACCENT
        } else {
            theme::BORDER
        };
        self.outline(r, 1.0, border);
        if !st.disabled && (st.hover || st.active) {
            self.rect(Rect::new(r.x + 2.0, r.bottom() - 3.0, r.w - 4.0, 1.0), [0.47, 0.86, 0.96, 0.6]);
        }
        let color = if st.disabled { theme::FAINT } else { theme::TEXT };
        let px = (r.h * 0.48).clamp(12.0, 22.0);
        let label = self.fonts.fit(bold(px), label, r.w - 10.0);
        self.text_in(r, bold(px), &label, color, Align::Center);
    }

    /// `< VALUE >` selector.
    pub fn cycler(&mut self, r: Rect, value: &str, st: WState) {
        if let Some(pal) = self.classic {
            self.rect(r, if st.hover && !st.disabled { pal.btn_hover } else { pal.btn });
            self.outline(r, 1.5, if st.hover && !st.disabled { pal.bracket } else { pal.border_outer });
            let arrow = if st.disabled { theme::FAINT } else { pal.title_gold };
            let px = (r.h * 0.55).clamp(11.0, 17.0);
            self.text_in(Rect::new(r.x + 6.0, r.y, 12.0, r.h), bold(px), "<", arrow, Align::Left);
            self.text_in(Rect::new(r.right() - 18.0, r.y, 12.0, r.h), bold(px), ">", arrow, Align::Right);
            let v = self.fonts.fit(body(px), value, r.w - 36.0);
            let col = if st.disabled { theme::DIM } else { theme::TEXT };
            self.text_in(r, body(px), &v, col, Align::Center);
            return;
        }
        self.rect(r, if st.hover && !st.disabled { theme::BTN_HOVER } else { theme::FIELD });
        self.outline(r, 1.0, if st.hover && !st.disabled { theme::ACCENT } else { theme::BORDER });
        let arrow = if st.disabled { theme::FAINT } else { theme::ACCENT };
        let px = (r.h * 0.55).clamp(11.0, 17.0);
        self.text_in(Rect::new(r.x + 6.0, r.y, 12.0, r.h), bold(px), "<", arrow, Align::Left);
        self.text_in(Rect::new(r.right() - 18.0, r.y, 12.0, r.h), bold(px), ">", arrow, Align::Right);
        let v = self.fonts.fit(body(px), value, r.w - 36.0);
        let col = if st.disabled { theme::DIM } else { theme::TEXT };
        self.text_in(r, body(px), &v, col, Align::Center);
    }

    pub fn checkbox(&mut self, r: Rect, on: bool, st: WState) {
        let b = Rect::new(r.x, r.y + (r.h - 18.0) * 0.5, 18.0, 18.0);
        let (acc, brd) = match self.classic {
            Some(pal) => (pal.accent, if st.hover { pal.bracket } else { pal.border_outer }),
            None => (theme::ACCENT, if st.hover { theme::ACCENT } else { theme::BORDER }),
        };
        self.rect(b, theme::FIELD);
        self.outline(b, 1.0, brd);
        if on {
            self.rect(b.inset(4.0), acc);
        }
        let label = if on { "ON" } else { "OFF" };
        self.text_in(Rect::new(b.right() + 10.0, r.y, 60.0, r.h), body(15.0), label, if on { theme::TEXT } else { theme::DIM }, Align::Left);
    }

    /// Horizontal slider, `frac` in `0..=1`, with a value label at the right.
    pub fn slider(&mut self, r: Rect, frac: f32, value: &str, st: WState) {
        let track = Rect::new(r.x, r.y + r.h * 0.5 - 3.0, r.w - 56.0, 6.0);
        self.rect(track, theme::FIELD);
        let (acc, brd, fill_lo) = match self.classic {
            Some(pal) => (pal.accent, pal.border_outer, pal.header_bg),
            None => (theme::ACCENT, theme::BORDER, [0.15, 0.45, 0.55, 1.0]),
        };
        self.outline(track, 1.0, brd);
        let f = frac.clamp(0.0, 1.0);
        self.gradient_h(
            Rect::new(track.x + 1.0, track.y + 1.0, (track.w - 2.0) * f, track.h - 2.0),
            fill_lo,
            acc,
        );
        let kx = track.x + track.w * f;
        let knob = Rect::new(kx - 5.0, r.y + r.h * 0.5 - 9.0, 10.0, 18.0);
        self.rect(knob, if st.hover { theme::TEXT } else { [0.70, 0.78, 0.84, 1.0] });
        self.outline(knob, 1.0, [0.05, 0.08, 0.1, 1.0]);
        self.text_in(Rect::new(track.right() + 8.0, r.y, 48.0, r.h), body(15.0), value, theme::TEXT, Align::Right);
    }

    /// Single-line text field with caret when focused.
    pub fn text_field(&mut self, r: Rect, text: &str, focused: bool, placeholder: &str, st: WState) {
        self.rect(r, theme::FIELD);
        let (acc, base_brd, hov_brd) = match self.classic {
            Some(pal) => (pal.accent, pal.border_outer, pal.bracket),
            None => (theme::ACCENT, theme::BORDER, [0.40, 0.50, 0.58, 1.0]),
        };
        let border = if focused { acc } else if st.hover { hov_brd } else { base_brd };
        self.outline(r, 1.0, border);
        let px = (r.h * 0.55).clamp(11.0, 17.0);
        let inner = Rect::new(r.x + 8.0, r.y, r.w - 16.0, r.h);
        if text.is_empty() && !focused {
            self.text_in(inner, body(px), placeholder, theme::FAINT, Align::Left);
            return;
        }
        // Keep the tail (caret end) visible.
        let mut shown = text.to_string();
        while !shown.is_empty() && self.measure(body(px), &shown) > inner.w - 6.0 {
            shown.remove(0);
        }
        self.text_in(inner, body(px), &shown, theme::TEXT, Align::Left);
        if focused {
            let cx = inner.x + self.measure(body(px), &shown) + 1.0;
            let lh = self.fonts.line_height(body(px));
            self.rect(Rect::new(cx, r.y + (r.h - lh) * 0.5, 2.0, lh), acc);
        }
    }

    /// Selection / hover highlight behind a list row.
    pub fn list_row(&mut self, r: Rect, selected: bool, hovered: bool) {
        if let Some(pal) = self.classic {
            if selected {
                self.gradient_h(r, pal.btn_active, pal.panel_bot);
                self.rect(Rect::new(r.x, r.y, 3.0, r.h), pal.bracket);
            } else if hovered {
                self.rect(r, pal.btn_hover);
            }
            return;
        }
        if selected {
            self.gradient_h(r, theme::ROW_SEL, [0.06, 0.18, 0.22, 0.85]);
            self.rect(Rect::new(r.x, r.y, 2.0, r.h), theme::ACCENT);
        } else if hovered {
            self.rect(r, theme::ROW_HOVER);
        }
    }

    /// Small `label` (dim) left of a control area; returns the control rect.
    #[allow(dead_code)]
    pub fn labeled_row(&mut self, r: Rect, label: &str, label_w: f32) -> Rect {
        self.text_in(Rect::new(r.x, r.y, label_w, r.h), bold(14.0), label, theme::DIM, Align::Left);
        Rect::new(r.x + label_w, r.y, r.w - label_w, r.h)
    }
}

// ---------------------------------------------------------------------------
// Screen frame (header / content / footer) shared by every sub-screen
// ---------------------------------------------------------------------------

pub const HEADER_H: f32 = 66.0;
pub const FOOTER_H: f32 = 64.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    pub header: Rect,
    pub content: Rect,
    pub footer: Rect,
}

impl Frame {
    pub fn new(w: f32, h: f32) -> Self {
        let cw = (w - 48.0).min(1500.0).max(200.0);
        let cx = ((w - cw) * 0.5).round();
        Self {
            header: Rect::new(0.0, 0.0, w, HEADER_H),
            content: Rect::new(cx, HEADER_H + 16.0, cw, (h - HEADER_H - FOOTER_H - 30.0).max(100.0)),
            footer: Rect::new(0.0, h - FOOTER_H, w, FOOTER_H),
        }
    }

    /// Footer button slot: `i`-th from the left (`right = false`) or right edge.
    pub fn footer_button(&self, i: usize, right: bool, width: f32) -> Rect {
        let y = self.footer.y + (FOOTER_H - 40.0) * 0.5;
        let step = width + 12.0;
        if right {
            Rect::new(self.content.right() - width - i as f32 * step, y, width, 40.0)
        } else {
            Rect::new(self.content.x + i as f32 * step, y, width, 40.0)
        }
    }
}

impl Canvas<'_> {
    /// Full-window FA backdrop: the menu background art (darkened) plus vignette.
    pub fn backdrop(&mut self, dim: f32) {
        let full = Rect::new(0.0, 0.0, self.w, self.h);
        if let Some(pal) = self.classic {
            let shade = [dim * 0.96, dim * 1.0, dim * 1.08, 1.0];
            if !self.image_cover("bg_classic", full, shade) && !self.image_cover("bg", full, shade) {
                self.gradient_v(full, pal.panel_top, [0.01, 0.015, 0.025, 1.0]);
            }
            let edge = [0.0, 0.0, 0.0, 0.78];
            let clear = [0.0, 0.0, 0.0, 0.0];
            let vw = (self.w * 0.20).min(280.0);
            self.gradient_h(Rect::new(0.0, 0.0, vw, self.h), edge, clear);
            self.gradient_h(Rect::new(self.w - vw, 0.0, vw, self.h), clear, edge);
            self.gradient_v(Rect::new(0.0, self.h - 160.0, self.w, 160.0), clear, [0.0, 0.0, 0.0, 0.65]);
            return;
        }
        let shade = [dim, dim * 1.04, dim * 1.1, 1.0];
        if !self.image_cover("bg", full, shade) {
            self.gradient_v(full, [0.04, 0.06, 0.09, 1.0], [0.01, 0.012, 0.02, 1.0]);
        }
        let edge = [0.0, 0.0, 0.0, 0.75];
        let clear = [0.0, 0.0, 0.0, 0.0];
        let vw = (self.w * 0.18).min(260.0);
        self.gradient_h(Rect::new(0.0, 0.0, vw, self.h), edge, clear);
        self.gradient_h(Rect::new(self.w - vw, 0.0, vw, self.h), clear, edge);
        self.gradient_v(Rect::new(0.0, self.h - 160.0, self.w, 160.0), clear, [0.0, 0.0, 0.0, 0.6]);
    }

    /// Sub-screen chrome: header strip with title / subtitle and the footer bar.
    pub fn chrome(&mut self, f: &Frame, title_text: &str, subtitle: &str) {
        if let Some(pal) = self.classic {
            let full = Rect::new(0.0, 0.0, self.w, self.h);
            let shade = [0.46, 0.48, 0.52, 1.0];
            if !self.image_cover("bg_classic_sub", full, shade)
                && !self.image_cover("bg_classic", full, shade)
                && !self.image_cover("bg", full, shade)
            {
                self.gradient_v(full, pal.panel_top, pal.panel_bot);
            }
            self.gradient_v(f.header, pal.header_bg, pal.panel_bot);
            self.rect(Rect::new(0.0, f.header.bottom() - 3.0, self.w, 2.0), pal.border_outer);
            self.rect(Rect::new(0.0, f.header.bottom() - 1.0, self.w, 1.0), pal.border_inner);
            let tx = f.content.x;
            self.rect(Rect::new(tx, 16.0, 5.0, 34.0), pal.bracket);
            self.text_in(Rect::new(tx + 16.0, 8.0, 900.0, 50.0), title(34.0), title_text, pal.title_gold, Align::Left);
            if !subtitle.is_empty() {
                let r = Rect::new(f.content.right() - 620.0, 8.0, 620.0, 50.0);
                self.text_in(r, body(15.0), subtitle, pal.accent, Align::Right);
            }
            self.gradient_v(f.footer, pal.panel_top, pal.panel_bot);
            self.rect(Rect::new(0.0, f.footer.y, self.w, 2.0), pal.border_outer);
            return;
        }
        self.backdrop(0.42);
        self.gradient_v(f.header, [0.02, 0.03, 0.04, 0.96], [0.04, 0.06, 0.08, 0.92]);
        self.rect(Rect::new(0.0, f.header.bottom() - 2.0, self.w, 1.0), [0.47, 0.86, 0.96, 0.55]);
        self.gradient_v(Rect::new(0.0, f.header.bottom() - 1.0, self.w, 8.0), [0.47, 0.86, 0.96, 0.18], [0.47, 0.86, 0.96, 0.0]);
        let tx = f.content.x;
        self.rect(Rect::new(tx, 18.0, 4.0, 30.0), theme::ACCENT);
        self.text_in(Rect::new(tx + 16.0, 8.0, 900.0, 50.0), title(34.0), title_text, theme::TEXT, Align::Left);
        if !subtitle.is_empty() {
            let r = Rect::new(f.content.right() - 600.0, 8.0, 600.0, 50.0);
            self.text_in(r, body(15.0), subtitle, theme::DIM, Align::Right);
        }
        self.gradient_v(f.footer, [0.03, 0.04, 0.05, 0.90], [0.01, 0.015, 0.02, 0.97]);
        self.rect(Rect::new(0.0, f.footer.y, self.w, 1.0), [0.27, 0.34, 0.40, 1.0]);
    }

    /// One-line status text in the footer, centered between the footer buttons.
    pub fn footer_status(&mut self, f: &Frame, text: &str, color: [f32; 4]) {
        let r = Rect::new(f.content.x + 190.0, f.footer.y, (f.content.w - 420.0).max(50.0), f.footer.h);
        let t = self.fonts.fit(body(15.0), text, r.w);
        self.text_in(r, body(15.0), &t, color, Align::Center);
    }

    /// Full-screen "loading" overlay.
    #[allow(dead_code)]
    pub fn loading_overlay(&mut self, headline: &str, detail: &str) {
        self.rect(Rect::new(0.0, 0.0, self.w, self.h), [0.0, 0.0, 0.0, 0.72]);
        let r = Rect::new(self.w * 0.5 - 260.0, self.h * 0.5 - 70.0, 520.0, 140.0);
        self.panel(r);
        self.text_in(Rect::new(r.x, r.y + 18.0, r.w, 50.0), title(34.0), headline, theme::ACCENT, Align::Center);
        self.text_in(Rect::new(r.x, r.y + 80.0, r.w, 30.0), body(15.0), detail, theme::DIM, Align::Center);
    }
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

const UI_SHADER: &str = r#"
struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};
@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>, @location(2) color: vec4<f32>) -> VOut {
    var o: VOut;
    o.pos = vec4<f32>(pos, 0.0, 1.0);
    o.uv = uv;
    o.color = color;
    return o;
}
@group(0) @binding(0) var t_img: texture_2d<f32>;
@group(0) @binding(1) var s_img: sampler;
@fragment
fn fs_main(i: VOut) -> @location(0) vec4<f32> {
    return textureSample(t_img, s_img, i.uv) * i.color;
}
"#;

pub struct UiRenderer {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    atlas: wgpu::BindGroup,
    images: HashMap<String, wgpu::BindGroup>,
}

impl UiRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat, fonts: &Fonts) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui_shader"),
            source: wgpu::ShaderSource::Wgsl(UI_SHADER.into()),
        });
        let tex_entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty,
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui_bgl"),
            entries: &[
                tex_entry(
                    0,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                tex_entry(1, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui_pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let vlayout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<HudVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4],
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vlayout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ui_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let atlas = Self::bind_image(device, queue, &layout, &sampler, &fonts.atlas);
        Self { pipeline, layout, sampler, atlas, images: HashMap::new() }
    }

    fn bind_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        img: &RgbaImage,
    ) -> wgpu::BindGroup {
        let size = wgpu::Extent3d { width: img.width.max(1), height: img.height.max(1), depth_or_array_layers: 1 };
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ui_tex"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &img.pixels,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * size.width), rows_per_image: Some(size.height) },
            size,
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        })
    }

    /// Drops a cached image texture (call when a store image under `name` changes).
    #[allow(dead_code)]
    pub fn forget(&mut self, name: &str) {
        self.images.remove(name);
    }

    pub fn draw(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::TextureView, canvas: &Canvas, clear: [f64; 3]) {
        for b in &canvas.batches {
            if let Tex::Image(name) = &b.tex
                && !self.images.contains_key(name)
                && let Some(img) = canvas.images.get(name)
            {
                let bg = Self::bind_image(device, queue, &self.layout, &self.sampler, img);
                self.images.insert(name.clone(), bg);
            }
        }
        let mut verts: Vec<HudVertex> = Vec::new();
        let mut idx: Vec<u32> = Vec::new();
        let mut ranges = Vec::with_capacity(canvas.batches.len());
        for b in &canvas.batches {
            let base = verts.len() as u32;
            let start = idx.len() as u32;
            verts.extend_from_slice(&b.verts);
            idx.extend(b.idx.iter().map(|i| i + base));
            ranges.push((start, idx.len() as u32));
        }
        let buffers = (!idx.is_empty()).then(|| {
            (
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ui_vb"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ui_ib"),
                    contents: bytemuck::cast_slice(&idx),
                    usage: wgpu::BufferUsages::INDEX,
                }),
            )
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ui_enc") });
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: clear[0], g: clear[1], b: clear[2], a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if let Some((vb, ib)) = &buffers {
                rp.set_pipeline(&self.pipeline);
                rp.set_vertex_buffer(0, vb.slice(..));
                rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                for (b, (s, e)) in canvas.batches.iter().zip(ranges) {
                    let group = match &b.tex {
                        Tex::Atlas => &self.atlas,
                        Tex::Image(name) => match self.images.get(name) {
                            Some(g) => g,
                            None => continue,
                        },
                    };
                    if e > s {
                        rp.set_bind_group(0, group, &[]);
                        rp.draw_indexed(s..e, 0, 0..1);
                    }
                }
            }
        }
        queue.submit(Some(enc.finish()));
    }
}

pub const CLEAR: [f64; 3] = [0.008, 0.011, 0.016];

/// Renders a canvas offscreen and writes it as PNG; returns the adapter description.
pub fn render_png(
    width: u32,
    height: u32,
    out: &Path,
    fonts: &Fonts,
    images: &ImageStore,
    build: impl FnOnce(&mut Canvas),
) -> Result<String, String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| "no GPU adapter for UI snapshot".to_string())?;
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        .map_err(|e| format!("failed to request GPU device: {e}"))?;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let extent = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ui_snap"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let mut r = UiRenderer::new(&device, &queue, format, fonts);
    let mut canvas = Canvas::new(width as f32, height as f32, fonts, images, None);
    build(&mut canvas);
    r.draw(&device, &queue, &view, &canvas, CLEAR);

    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let bpr = width * 4;
    let padded = bpr.div_ceil(align) * align;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ui_readback"),
        size: u64::from(padded) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(height) },
        },
        extent,
    );
    queue.submit(Some(enc.finish()));
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| format!("readback channel failed: {e}"))?
        .map_err(|e| format!("GPU buffer map failed: {e}"))?;
    let mapped = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((bpr * height) as usize);
    for row in 0..height as usize {
        let s = row * padded as usize;
        pixels.extend_from_slice(&mapped[s..s + bpr as usize]);
    }
    drop(mapped);
    buf.unmap();
    save_png(&RgbaImage { width, height, pixels }, out)?;
    Ok(format!("{} ({:?})", info.name, info.backend))
}

pub fn save_png(img: &RgbaImage, out: &Path) -> Result<(), String> {
    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    let file = File::create(out).map_err(|e| format!("failed to create {}: {e}", out.display()))?;
    let mut enc = png::Encoder::new(BufWriter::new(file), img.width, img.height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .and_then(|mut w| w.write_image_data(&img.pixels))
        .map_err(|e| format!("png write error: {e}"))
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------

pub struct WindowGpu {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    renderer: UiRenderer,
}

impl WindowGpu {
    /// Opens a window honoring the active settings (fullscreen, vsync).
    pub fn create(event_loop: &ActiveEventLoop, title: &str, size: (u32, u32), fonts: &Fonts) -> Result<Self, String> {
        let attrs = Window::default_attributes()
            .with_title(title)
            .with_inner_size(PhysicalSize::new(size.0, size.1))
            .with_min_inner_size(PhysicalSize::new(800u32, 600u32));
        let attrs = crate::settings::apply_window_attrs(attrs);
        let window = Arc::new(event_loop.create_window(attrs).map_err(|e| format!("failed to create window: {e}"))?);
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("failed to create wgpu surface: {e}"))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| "no compatible GPU adapter for the menu window".to_string())?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .map_err(|e| format!("failed to request GPU device: {e}"))?;
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let inner = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: inner.width.max(1),
            height: inner.height.max(1),
            present_mode: crate::settings::present_mode(),
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let renderer = UiRenderer::new(&device, &queue, format, fonts);
        Ok(Self { window, surface, device, queue, config, renderer })
    }

    pub fn size(&self) -> (f32, f32) {
        (self.config.width as f32, self.config.height as f32)
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w > 0 && h > 0 {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Applies vsync changes from the active settings.
    pub fn reconfigure(&mut self) {
        self.config.present_mode = crate::settings::present_mode();
        self.surface.configure(&self.device, &self.config);
    }

    pub fn render(&mut self, canvas: &Canvas) {
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.draw(&self.device, &self.queue, &view, canvas, CLEAR);
        frame.present();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_test_prefers_topmost_widget() {
        let ws = [(Rect::new(0.0, 0.0, 100.0, 100.0), 1), (Rect::new(10.0, 10.0, 20.0, 20.0), 2)];
        assert_eq!(hit_test(&ws, 15.0, 15.0), Some(2));
        assert_eq!(hit_test(&ws, 50.0, 50.0), Some(1));
        assert_eq!(hit_test(&ws, 150.0, 50.0), None);
    }

    #[test]
    fn canvas_batches_by_texture_in_paint_order() {
        let fonts = Fonts::fallback();
        let mut store = ImageStore::default();
        store.insert("img", RgbaImage { width: 1, height: 1, pixels: vec![255; 4] });
        let mut c = Canvas::new(800.0, 600.0, &fonts, &store, Some((5.0, 5.0)));
        c.rect(Rect::new(0.0, 0.0, 10.0, 10.0), theme::TEXT);
        c.text(0.0, 0.0, body(14.0), "HI", theme::TEXT);
        assert!(c.image("img", Rect::new(0.0, 0.0, 4.0, 4.0), theme::TEXT));
        assert!(!c.image("missing", Rect::new(0.0, 0.0, 4.0, 4.0), theme::TEXT));
        c.button(Rect::new(0.0, 0.0, 100.0, 30.0), "OK", WState { hover: c.is_hover(Rect::new(0.0, 0.0, 10.0, 10.0)), ..Default::default() });
        let kinds: Vec<_> = c.batches.iter().map(|b| b.tex.clone()).collect();
        assert_eq!(kinds, vec![Tex::Atlas, Tex::Image("img".into()), Tex::Atlas]);
        assert!(c.batches.iter().all(|b| b.idx.len() % 6 == 0 && b.verts.iter().all(|v| v.pos.iter().all(|p| p.is_finite()))));
        assert_eq!(fonts.measure(body(14.0), "ABC"), 3.0 * 4.0 * 2.0);
    }
}
