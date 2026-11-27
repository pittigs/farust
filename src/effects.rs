//! Phase 40 — Forged Alliance effect blueprints (`effects.scd`), `EffectTemplates.lua`
//! data-driven weapon/event mapping, curve evaluator, and VFX texture array loader.
//!
//! Purely render-side: never alters simulation state or determinism.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use crate::blueprint::{self, Value};
use crate::dds_decode::{RgbaImage, decode_dds_rgba8, resize_rgba_bilinear};
use crate::jobs::{default_thread_count, par_map_ordered};
use crate::scd;

/// Maximum sprite layers packed into the GPU `texture_2d_array` (wgpu default limit = 256).
pub const MAX_VFX_TEX_LAYERS: usize = 256;
/// Uniform tile edge size for each layer in the GPU VFX texture array.
pub const VFX_TEX_SIZE: u32 = 128;
/// Moho engine simulation rate used by emitter blueprint tick units (`Lifetime`, `LifetimeCurve`, etc.).
pub const MOHO_TICK_HZ: f32 = 10.0;

/// Sanitize a parsed float so NaN / Inf never propagate into curves or particles.
#[inline]
fn finite_or(v: f64, fallback: f32) -> f32 {
    if v.is_finite() { v as f32 } else { fallback }
}

/// Normalize an FA virtual asset path to lowercase forward-slash without a leading `/`.
pub fn norm_asset_path(p: &str) -> String {
    p.trim()
        .replace('\\', "/")
        .trim_start_matches('/')
        .to_ascii_lowercase()
}

// ---------------------------------------------------------------------------
// Effect blueprint data model & curves
// ---------------------------------------------------------------------------

/// One keyframe `{ x = time, y = mean, z = spread }` in an FA emitter curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveKey {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Piecewise-linear FA emitter curve (`XRange`, `Keys = { {x, y, z}, ... }`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Curve {
    pub x_range: f32,
    pub keys: Vec<CurveKey>,
}

impl Curve {
    /// Evaluates `(mean_y, spread_z)` at `x` (clamped at endpoints, linear between keys).
    pub fn eval(&self, x: f32) -> (f32, f32) {
        if self.keys.is_empty() {
            return (0.0, 0.0);
        }
        let x = if x.is_finite() { x } else { 0.0 };
        if self.keys.len() == 1 || x <= self.keys[0].x {
            let k = self.keys[0];
            return (k.y, k.z);
        }
        let last = self.keys[self.keys.len() - 1];
        if x >= last.x {
            return (last.y, last.z);
        }
        for pair in self.keys.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if x <= b.x {
                let dx = b.x - a.x;
                let t = if dx.abs() > 1e-6 {
                    ((x - a.x) / dx).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                return (a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);
            }
        }
        (last.y, last.z)
    }

    /// Samples a random value at `x` using `rand_signed ∈ [-1.0, 1.0]`: `y + 0.5 * rand_signed * z`.
    #[inline]
    pub fn sample(&self, x: f32, rand_signed: f32) -> f32 {
        let (y, z) = self.eval(x);
        let v = y + 0.5 * rand_signed.clamp(-1.0, 1.0) * z;
        if v.is_finite() { v } else { 0.0 }
    }

    /// Mean value at the midpoint of `x_range` (or first key).
    pub fn mid_y(&self) -> f32 {
        let x = if self.x_range > 0.0 { self.x_range * 0.5 } else { 0.0 };
        self.eval(x).0
    }

    /// Maximum `|y|` across all keys.
    pub fn max_abs_y(&self) -> f32 {
        self.keys.iter().fold(0.0f32, |m, k| m.max(k.y.abs()))
    }
}

/// Moho particle blend mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Blendmode 0 / 1: standard alpha blending (smoke, clouds, debris).
    Alpha,
    /// Blendmode 2 / 3: additive glow (muzzle flashes, fire, lasers, plasma).
    Additive,
    /// Blendmode 4: pre-modulated alpha.
    PreMod,
    /// Blendmode 5: distortion/refraction ring (rendered as soft additive shimmer).
    Refract,
}

impl BlendMode {
    pub fn from_code(code: f32) -> Self {
        match code.round() as i32 {
            2 | 3 => BlendMode::Additive,
            4 => BlendMode::PreMod,
            5 => BlendMode::Refract,
            _ => BlendMode::Alpha,
        }
    }

    #[inline]
    pub fn is_additive(self) -> bool {
        matches!(self, BlendMode::Additive | BlendMode::Refract)
    }
}

/// Compact 4-stop RGBA color ramp sampled from an FA `RampTexture` `.dds`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorRamp {
    pub stops: [[f32; 4]; 4],
}

impl Default for ColorRamp {
    fn default() -> Self {
        Self {
            stops: [
                [1.0, 0.96, 0.85, 0.95],
                [1.0, 0.82, 0.50, 0.90],
                [0.95, 0.48, 0.18, 0.48],
                [0.60, 0.20, 0.08, 0.0],
            ],
        }
    }
}

impl ColorRamp {
    /// Builds a 4-stop ramp from a decoded `RampTexture` at vertical strip `row_frac ∈ [0, 1)`.
    pub fn from_rgba_image(img: &RgbaImage, row_frac: f32) -> Self {
        if img.width == 0 || img.height == 0 || img.pixels.len() < 4 {
            return Self::default();
        }
        let w = img.width as usize;
        let h = img.height as usize;
        let y = ((row_frac.clamp(0.0, 0.999) * h as f32) as usize).min(h - 1);
        // Sample at u = 0.08, 0.30, 0.62, 0.94 so fast ramp fade-ins at pixel 0 don't hide short flashes.
        let sample_at = |u: f32| -> [f32; 4] {
            let x = ((u.clamp(0.0, 1.0) * (w.saturating_sub(1)) as f32).round() as usize).min(w - 1);
            let i = (y * w + x) * 4;
            if i + 3 < img.pixels.len() {
                [
                    img.pixels[i] as f32 / 255.0,
                    img.pixels[i + 1] as f32 / 255.0,
                    img.pixels[i + 2] as f32 / 255.0,
                    img.pixels[i + 3] as f32 / 255.0,
                ]
            } else {
                [1.0, 1.0, 1.0, 1.0]
            }
        };
        let mut stops = [
            sample_at(0.08),
            sample_at(0.30),
            sample_at(0.62),
            sample_at(0.95),
        ];
        // Some FA additive ramps store intensity in RGB with A = 0 or 255 across the strip.
        let max_a = stops.iter().fold(0.0f32, |m, c| m.max(c[3]));
        let max_rgb = stops
            .iter()
            .fold(0.0f32, |m, c| m.max(c[0]).max(c[1]).max(c[2]));
        if max_a < 0.05 && max_rgb > 0.05 {
            for c in &mut stops {
                c[3] = c[0].max(c[1]).max(c[2]);
            }
        }
        // Ensure smooth fade-out at end of life.
        stops[3][3] = stops[3][3].min(0.05);
        Self { stops }
    }

    /// Evaluates RGBA at normalized age `t ∈ [0.0, 1.0]`.
    #[inline]
    pub fn sample(&self, t: f32) -> [f32; 4] {
        if t <= 0.0 {
            return self.stops[0];
        }
        if t >= 1.0 {
            return self.stops[3];
        }
        let s = t * 3.0;
        let idx = (s.floor() as usize).min(2);
        let frac = s - idx as f32;
        let a = self.stops[idx];
        let b = self.stops[idx + 1];
        [
            a[0] + (b[0] - a[0]) * frac,
            a[1] + (b[1] - a[1]) * frac,
            a[2] + (b[2] - a[2]) * frac,
            a[3] + (b[3] - a[3]) * frac,
        ]
    }
}

/// Parsed `EmitterBlueprint { ... }` from `effects.scd`.
#[derive(Debug, Clone, PartialEq)]
pub struct EmitterBp {
    pub id: String,
    pub lifetime: f32,
    pub repeat_time: f32,
    pub blend_mode: BlendMode,
    pub flat: bool,
    pub local_velocity: bool,
    pub gravity: bool,
    pub align_rotation: bool,
    pub lod_cutoff: f32,
    pub texture_frame_count: u32,
    pub texture_strip_count: u32,
    pub texture: String,
    pub ramp_texture: String,
    pub x_dir: Curve,
    pub y_dir: Curve,
    pub z_dir: Curve,
    pub emit_rate: Curve,
    pub lifetime_curve: Curve,
    pub velocity: Curve,
    pub x_accel: Curve,
    pub y_accel: Curve,
    pub z_accel: Curve,
    pub resistance: Curve,
    pub size: Curve,
    pub start_size: Curve,
    pub end_size: Curve,
    pub x_pos: Curve,
    pub y_pos: Curve,
    pub z_pos: Curve,
    pub init_rot: Curve,
    pub rot_rate: Curve,
    pub frame_rate: Curve,
    pub tex_select: Curve,
    pub ramp_select: Curve,
}

/// Parsed `BeamBlueprint { ... }` from `effects.scd`.
#[derive(Debug, Clone, PartialEq)]
pub struct BeamBp {
    pub id: String,
    pub lifetime: f32,
    pub texture: String,
    pub thickness: f32,
    pub start_color: [f32; 4],
    pub end_color: [f32; 4],
    pub blend_mode: BlendMode,
    pub repeat_rate: f32,
    pub length: f32,
    pub u_shift: f32,
    pub v_shift: f32,
    pub lod_cutoff: f32,
}

/// Parsed `TrailEmitterBlueprint { ... }` from `effects.scd`.
#[derive(Debug, Clone, PartialEq)]
pub struct TrailBp {
    pub id: String,
    pub lifetime: f32,
    pub trail_length: f32,
    pub size: f32,
    pub blend_mode: BlendMode,
    pub texture_repeat_rate: f32,
    pub lod_cutoff: f32,
    pub repeat_texture: String,
    pub ramp_texture: String,
    pub u_shift: f32,
    pub v_shift: f32,
}

/// Any blueprint found in `effects.scd`.
#[derive(Debug, Clone, PartialEq)]
pub enum EffectBp {
    Emitter(Box<EmitterBp>),
    Beam(BeamBp),
    Trail(TrailBp),
    Other { constructor: String },
}

fn parse_curve(root: &Value, key: &str) -> Curve {
    let Some(tbl) = root.get(key) else {
        return Curve::default();
    };
    let x_range = tbl.get_f64("XRange").map(|v| finite_or(v, 1.0)).unwrap_or(1.0);
    let mut keys = Vec::new();
    if let Some(ks) = tbl.get("Keys") {
        for kt in ks.table_list() {
            let x = kt.get_f64("x").map(|v| finite_or(v, 0.0)).unwrap_or(0.0);
            let y = kt.get_f64("y").map(|v| finite_or(v, 0.0)).unwrap_or(0.0);
            let z = kt.get_f64("z").map(|v| finite_or(v, 0.0)).unwrap_or(0.0);
            keys.push(CurveKey { x, y, z });
        }
    }
    keys.sort_by(|a, b| a.x.total_cmp(&b.x));
    Curve { x_range, keys }
}

fn parse_color4(root: &Value, key: &str, default: [f32; 4]) -> [f32; 4] {
    let Some(tbl) = root.get(key) else {
        return default;
    };
    if let (Some(x), Some(y), Some(z)) = (tbl.get_f64("x"), tbl.get_f64("y"), tbl.get_f64("z")) {
        let w = tbl.get_f64("w").unwrap_or(1.0);
        return [
            finite_or(x, default[0]).clamp(0.0, 2.0),
            finite_or(y, default[1]).clamp(0.0, 2.0),
            finite_or(z, default[2]).clamp(0.0, 2.0),
            finite_or(w, default[3]).clamp(0.0, 2.0),
        ];
    }
    // Positional fallback `{r, g, b, a}`.
    if let Value::Table { entries, .. } = tbl {
        let nums: Vec<f32> = entries
            .iter()
            .filter_map(|e| match e.value {
                Value::Number(n) => Some(finite_or(n, 1.0).clamp(0.0, 2.0)),
                _ => None,
            })
            .collect();
        if nums.len() >= 3 {
            return [nums[0], nums[1], nums[2], nums.get(3).copied().unwrap_or(1.0)];
        }
    }
    default
}

/// Parses an FA effect blueprint (`EmitterBlueprint`, `BeamBlueprint`, `TrailEmitterBlueprint`,
/// or `ProjectileBlueprint`/`MeshBlueprint` found in `effects.scd`). Never panics.
pub fn parse_effect_blueprint(text: &str) -> Result<EffectBp, String> {
    let (ctor, root) = blueprint::parse_any_blueprint(text)?;
    let ctor_name = ctor.ok_or_else(|| "missing blueprint root constructor".to_string())?;
    let f = |k: &str, d: f32| root.get_f64(k).map(|v| finite_or(v, d)).unwrap_or(d);
    let b = |k: &str, d: bool| root.get_bool(k).unwrap_or(d);
    let s = |k: &str| root.get_str(k).map(norm_asset_path).unwrap_or_default();

    match ctor_name.as_str() {
        "EmitterBlueprint" => Ok(EffectBp::Emitter(Box::new(EmitterBp {
            id: root.get_str("BlueprintId").unwrap_or_default().to_string(),
            lifetime: f("Lifetime", 1.0),
            repeat_time: f("Repeattime", 1.0).max(0.0),
            blend_mode: BlendMode::from_code(
                root.get_f64("Blendmode")
                    .or_else(|| root.get_f64("BlendMode"))
                    .map(|v| finite_or(v, 3.0))
                    .unwrap_or(3.0),
            ),
            flat: b("Flat", false),
            local_velocity: b("LocalVelocity", false),
            gravity: b("Gravity", false),
            align_rotation: b("AlignRotation", false),
            lod_cutoff: f("LODCutoff", 150.0),
            texture_frame_count: (f("TextureFramecount", 1.0).round() as i32).clamp(1, 64) as u32,
            texture_strip_count: (f("TextureStripcount", 1.0).round() as i32).clamp(1, 64) as u32,
            texture: s("Texture"),
            ramp_texture: s("RampTexture"),
            x_dir: parse_curve(&root, "XDirectionCurve"),
            y_dir: parse_curve(&root, "YDirectionCurve"),
            z_dir: parse_curve(&root, "ZDirectionCurve"),
            emit_rate: parse_curve(&root, "EmitRateCurve"),
            lifetime_curve: parse_curve(&root, "LifetimeCurve"),
            velocity: parse_curve(&root, "VelocityCurve"),
            x_accel: parse_curve(&root, "XAccelCurve"),
            y_accel: parse_curve(&root, "YAccelCurve"),
            z_accel: parse_curve(&root, "ZAccelCurve"),
            resistance: parse_curve(&root, "ResistanceCurve"),
            size: parse_curve(&root, "SizeCurve"),
            start_size: parse_curve(&root, "StartSizeCurve"),
            end_size: parse_curve(&root, "EndSizeCurve"),
            x_pos: parse_curve(&root, "XPosCurve"),
            y_pos: parse_curve(&root, "YPosCurve"),
            z_pos: parse_curve(&root, "ZPosCurve"),
            init_rot: parse_curve(&root, "InitialRotationCurve"),
            rot_rate: parse_curve(&root, "RotationRateCurve"),
            frame_rate: parse_curve(&root, "FrameRateCurve"),
            tex_select: parse_curve(&root, "TextureSelectionCurve"),
            ramp_select: parse_curve(&root, "RampSelectionCurve"),
        }))),
        "BeamBlueprint" => Ok(EffectBp::Beam(BeamBp {
            id: root.get_str("BlueprintId").unwrap_or_default().to_string(),
            lifetime: f("Lifetime", 1.0),
            texture: s("TextureName"),
            thickness: f("Thickness", 0.1).abs().clamp(0.01, 20.0),
            start_color: parse_color4(&root, "StartColor", [1.0, 1.0, 1.0, 1.0]),
            end_color: parse_color4(&root, "EndColor", [1.0, 1.0, 1.0, 1.0]),
            blend_mode: BlendMode::from_code(
                root.get_f64("Blendmode")
                    .or_else(|| root.get_f64("BlendMode"))
                    .map(|v| finite_or(v, 3.0))
                    .unwrap_or(3.0),
            ),
            repeat_rate: f("RepeatRate", 1.0).abs().clamp(0.05, 64.0),
            length: f("Length", 8.0).abs().clamp(0.1, 1000.0),
            u_shift: f("UShift", 0.0),
            v_shift: f("VShift", 0.0),
            lod_cutoff: f("LODCutoff", 200.0),
        })),
        "TrailEmitterBlueprint" => Ok(EffectBp::Trail(TrailBp {
            id: root.get_str("BlueprintId").unwrap_or_default().to_string(),
            lifetime: f("Lifetime", -1.0),
            trail_length: f("TrailLength", 5.0).abs().clamp(0.1, 100.0),
            size: f("Size", 0.2).abs().clamp(0.01, 20.0),
            blend_mode: BlendMode::from_code(
                root.get_f64("BlendMode")
                    .or_else(|| root.get_f64("Blendmode"))
                    .map(|v| finite_or(v, 0.0))
                    .unwrap_or(0.0),
            ),
            texture_repeat_rate: f("TextureRepeatRate", 1.0).abs().clamp(0.05, 64.0),
            lod_cutoff: f("LODCutoff", 160.0),
            repeat_texture: s("RepeatTexture"),
            ramp_texture: s("RampTexture"),
            u_shift: f("UShift", 0.0),
            v_shift: f("VShift", 0.0),
        })),
        "ProjectileBlueprint" | "MeshBlueprint" => Ok(EffectBp::Other {
            constructor: ctor_name,
        }),
        other => Err(format!("unsupported effect blueprint constructor '{other}'")),
    }
}

// ---------------------------------------------------------------------------
// Data-driven Lua parsers for EffectTemplates.lua & weapon/projectile links
// ---------------------------------------------------------------------------

/// Strips `#` and `--` comments from Moho Lua source while preserving quoted strings.
fn strip_lua_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\'' || c == b'"' {
            let q = c;
            out.push(c as char);
            i += 1;
            while i < bytes.len() {
                let b = bytes[i];
                out.push(b as char);
                i += 1;
                if b == b'\\' && i < bytes.len() {
                    out.push(bytes[i] as char);
                    i += 1;
                } else if b == q {
                    break;
                }
            }
        } else if bytes[i..].starts_with(b"--[[") {
            i += 4;
            while i + 1 < bytes.len() && !(bytes[i] == b']' && bytes[i + 1] == b']') {
                if bytes[i] == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
            i = (i + 2).min(bytes.len());
        } else if bytes[i..].starts_with(b"--") || c == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else {
            out.push(c as char);
            i += 1;
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
enum LuaTok {
    Ident(String),
    Str(String),
    Eq,
    LBrace,
    RBrace,
    LParen,
    RParen,
    Comma,
    Concat,
    Other(char),
}

fn tokenize_lua(src: &str) -> Vec<LuaTok> {
    let clean = strip_lua_comments(src);
    let b = clean.as_bytes();
    let mut toks = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c == b'\'' || c == b'"' {
            let q = c;
            i += 1;
            let mut s = String::new();
            while i < b.len() {
                let ch = b[i];
                i += 1;
                if ch == q {
                    break;
                }
                if ch == b'\\' && i < b.len() {
                    s.push(b[i] as char);
                    i += 1;
                } else {
                    s.push(ch as char);
                }
            }
            toks.push(LuaTok::Str(s));
        } else if c == b'[' && b.get(i + 1) == Some(&b'[') {
            i += 2;
            let start = i;
            while i + 1 < b.len() && !(b[i] == b']' && b[i + 1] == b']') {
                i += 1;
            }
            toks.push(LuaTok::Str(clean[start..i].to_string()));
            i = (i + 2).min(b.len());
        } else if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            toks.push(LuaTok::Ident(clean[start..i].to_string()));
        } else if c == b'.' && b.get(i + 1) == Some(&b'.') {
            toks.push(LuaTok::Concat);
            i += 2;
        } else {
            toks.push(match c {
                b'=' => LuaTok::Eq,
                b'{' => LuaTok::LBrace,
                b'}' => LuaTok::RBrace,
                b'(' => LuaTok::LParen,
                b')' => LuaTok::RParen,
                b',' => LuaTok::Comma,
                other => LuaTok::Other(other as char),
            });
            i += 1;
        }
    }
    toks
}

/// Evaluates a string expression such as `EmtBpPath .. 'flash_01_emit.bp'` or a literal.
fn eval_lua_string_expr(
    toks: &[LuaTok],
    pos: &mut usize,
    vars: &HashMap<String, String>,
) -> Option<String> {
    let mut acc = match toks.get(*pos)? {
        LuaTok::Str(s) => {
            *pos += 1;
            s.clone()
        }
        LuaTok::Ident(id) => {
            let v = vars.get(id)?.clone();
            *pos += 1;
            v
        }
        _ => return None,
    };
    while toks.get(*pos) == Some(&LuaTok::Concat) {
        *pos += 1;
        match toks.get(*pos) {
            Some(LuaTok::Str(s)) => {
                acc.push_str(s);
                *pos += 1;
            }
            Some(LuaTok::Ident(id)) => {
                if let Some(v) = vars.get(id) {
                    acc.push_str(v);
                }
                *pos += 1;
            }
            _ => break,
        }
    }
    Some(acc)
}

fn parse_lua_string_list(
    toks: &[LuaTok],
    pos: &mut usize,
    vars: &HashMap<String, String>,
    templates: &BTreeMap<String, Vec<String>>,
) -> Vec<String> {
    let mut out = Vec::new();
    if toks.get(*pos) != Some(&LuaTok::LBrace) {
        return out;
    }
    *pos += 1;
    let mut depth = 1usize;
    while *pos < toks.len() && depth > 0 {
        match &toks[*pos] {
            LuaTok::RBrace => {
                depth -= 1;
                *pos += 1;
            }
            LuaTok::LBrace => {
                depth += 1;
                *pos += 1;
            }
            LuaTok::Comma => {
                *pos += 1;
            }
            LuaTok::Str(_) => {
                if let Some(s) = eval_lua_string_expr(toks, pos, vars) {
                    let n = norm_asset_path(&s);
                    if !n.is_empty() {
                        out.push(n);
                    }
                } else {
                    *pos += 1;
                }
            }
            LuaTok::Ident(id) => {
                if vars.contains_key(id) {
                    if let Some(s) = eval_lua_string_expr(toks, pos, vars) {
                        let n = norm_asset_path(&s);
                        if !n.is_empty() {
                            out.push(n);
                        }
                    } else {
                        *pos += 1;
                    }
                } else if let Some(prev) = templates.get(id) {
                    out.extend(prev.iter().cloned());
                    *pos += 1;
                } else {
                    *pos += 1;
                }
            }
            _ => {
                *pos += 1;
            }
        }
    }
    out
}

/// Parses `lua/EffectTemplates.lua` into `TemplateName -> Vec<normalized_emitter_bp_path>`.
/// Supports string prefix variables (`EmtBpPath .. '...'`), table literals, single strings,
/// template aliases, and `TableCat(A, B, { ... })`.
pub fn parse_effect_templates(lua_text: &str) -> BTreeMap<String, Vec<String>> {
    let toks = tokenize_lua(lua_text);
    let mut vars: HashMap<String, String> = HashMap::new();
    let mut templates: BTreeMap<String, Vec<String>> = BTreeMap::new();

    let mut i = 0usize;
    while i < toks.len() {
        // Look for `Ident = ...` at top level (skip `local`).
        if let LuaTok::Ident(ref kw) = toks[i]
            && kw == "local"
        {
            i += 1;
            continue;
        }
        let LuaTok::Ident(name) = toks[i].clone() else {
            i += 1;
            continue;
        };
        if toks.get(i + 1) != Some(&LuaTok::Eq) {
            i += 1;
            continue;
        }
        i += 2;
        match toks.get(i) {
            Some(LuaTok::LBrace) => {
                let list = parse_lua_string_list(&toks, &mut i, &vars, &templates);
                templates.insert(name, list);
            }
            Some(LuaTok::Ident(callee)) if callee == "TableCat" && toks.get(i + 1) == Some(&LuaTok::LParen) => {
                i += 2;
                let mut combined = Vec::new();
                let mut depth = 1usize;
                while i < toks.len() && depth > 0 {
                    match &toks[i] {
                        LuaTok::LParen => {
                            depth += 1;
                            i += 1;
                        }
                        LuaTok::RParen => {
                            depth -= 1;
                            i += 1;
                        }
                        LuaTok::LBrace => {
                            combined.extend(parse_lua_string_list(&toks, &mut i, &vars, &templates));
                        }
                        LuaTok::Ident(arg) => {
                            if let Some(prev) = templates.get(arg) {
                                combined.extend(prev.iter().cloned());
                            } else if vars.contains_key(arg)
                                && let Some(s) = eval_lua_string_expr(&toks, &mut i, &vars)
                            {
                                let n = norm_asset_path(&s);
                                if !n.is_empty() {
                                    combined.push(n);
                                }
                                continue;
                            }
                            i += 1;
                        }
                        LuaTok::Str(_) => {
                            if let Some(s) = eval_lua_string_expr(&toks, &mut i, &vars) {
                                let n = norm_asset_path(&s);
                                if !n.is_empty() {
                                    combined.push(n);
                                }
                            } else {
                                i += 1;
                            }
                        }
                        _ => i += 1,
                    }
                }
                templates.insert(name, combined);
            }
            Some(LuaTok::Str(_)) => {
                if let Some(s) = eval_lua_string_expr(&toks, &mut i, &vars) {
                    if s.ends_with('/') {
                        vars.insert(name, s);
                    } else {
                        let n = norm_asset_path(&s);
                        if n.ends_with(".bp") {
                            templates.insert(name, vec![n]);
                        } else {
                            vars.insert(name, s);
                        }
                    }
                }
            }
            Some(LuaTok::Ident(rhs)) => {
                if vars.contains_key(rhs) {
                    if let Some(s) = eval_lua_string_expr(&toks, &mut i, &vars) {
                        let n = norm_asset_path(&s);
                        if n.ends_with(".bp") {
                            templates.insert(name, vec![n]);
                        } else {
                            vars.insert(name, s);
                        }
                    }
                } else if let Some(prev) = templates.get(rhs).cloned() {
                    templates.insert(name, prev);
                    i += 1;
                } else {
                    i += 1;
                }
            }
            _ => {}
        }
    }
    templates
}

/// Extracted effect template names from a Lua projectile or weapon class definition.
#[derive(Debug, Clone, Default)]
struct LuaClassFx {
    parent: Option<String>,
    muzzle: Vec<String>,
    impact: Vec<String>,
    trails: Vec<String>,
    poly_trail: Vec<String>,
    beam_type: Option<String>,
    fx_beam: Vec<String>,
    direct_bp_muzzle: Vec<String>,
    direct_bp_impact: Vec<String>,
    direct_bp_trails: Vec<String>,
}

/// Parses `ClassName = Class(Parent) { ... }` blocks from FA weapon / projectile / beam Lua files.
fn parse_lua_class_fx(lua_text: &str, out: &mut HashMap<String, LuaClassFx>) {
    let clean = strip_lua_comments(lua_text);
    let lines: Vec<&str> = clean.lines().collect();
    let mut cur_name: Option<String> = None;
    let mut cur_fx = LuaClassFx::default();
    let mut brace_depth: i32 = 0;

    let flush = |name: &mut Option<String>, fx: &mut LuaClassFx, out: &mut HashMap<String, LuaClassFx>| {
        if let Some(n) = name.take() {
            out.insert(n, std::mem::take(fx));
        }
    };

    for raw_line in &lines {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        // Detect top-level `ClassName = Class(Parent) {`
        if brace_depth <= 0
            && let Some((lhs, rhs)) = line.split_once('=')
        {
            let lhs_t = lhs.trim().trim_start_matches("local ").trim();
            let rhs_t = rhs.trim();
            if is_ident(lhs_t) && rhs_t.starts_with("Class") && rhs_t.contains('(') {
                flush(&mut cur_name, &mut cur_fx, out);
                let parent = extract_class_parent(rhs_t);
                cur_name = Some(lhs_t.to_string());
                cur_fx = LuaClassFx {
                    parent,
                    ..LuaClassFx::default()
                };
                brace_depth = count_braces(rhs_t);
                extract_fx_fields_from_line(rhs_t, &mut cur_fx);
                continue;
            }
        }
        if cur_name.is_some() {
            extract_fx_fields_from_line(line, &mut cur_fx);
            brace_depth += count_braces(line);
            if brace_depth <= 0 {
                flush(&mut cur_name, &mut cur_fx, out);
                brace_depth = 0;
            }
        }
    }
    flush(&mut cur_name, &mut cur_fx, out);
}

fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && s.as_bytes()
            .first()
            .is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_')
        && s.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
}

fn count_braces(s: &str) -> i32 {
    let mut d = 0i32;
    for b in s.bytes() {
        if b == b'{' {
            d += 1;
        } else if b == b'}' {
            d -= 1;
        }
    }
    d
}

fn extract_class_parent(s: &str) -> Option<String> {
    let lp = s.find('(')?;
    let rp = s[lp + 1..].find(')')?;
    let inner = s[lp + 1..lp + 1 + rp].trim();
    // Handle `import('/lua/aeonprojectiles.lua').AProjectile` or bare `Parent`.
    let tail = inner.rsplit('.').next()?.trim();
    is_ident(tail).then(|| tail.to_string())
}

fn extract_quoted_bps(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\'' || b[i] == b'"' {
            let q = b[i];
            i += 1;
            let start = i;
            while i < b.len() && b[i] != q {
                i += 1;
            }
            let val = &s[start..i];
            if val.to_ascii_lowercase().ends_with(".bp") {
                out.push(norm_asset_path(val));
            }
            if i < b.len() {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    out
}

fn extract_et_name(rhs: &str) -> Option<String> {
    let idx = rhs.find("EffectTemplate.")?;
    let rest = &rhs[idx + "EffectTemplate.".len()..];
    let end = rest
        .bytes()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == b'_')
        .count();
    (end > 0).then(|| rest[..end].to_string())
}

fn extract_fx_fields_from_line(line: &str, fx: &mut LuaClassFx) {
    let Some((lhs, rhs)) = line.split_once('=') else {
        // Multi-line table continuation for FxBeam / FxImpact / etc.
        let bps = extract_quoted_bps(line);
        for bp in bps {
            if bp.contains("beam") {
                fx.fx_beam.push(bp);
            } else if bp.contains("trail") {
                fx.direct_bp_trails.push(bp);
            } else if bp.contains("muzzle") || bp.contains("flash") {
                fx.direct_bp_muzzle.push(bp);
            } else {
                fx.direct_bp_impact.push(bp);
            }
        }
        return;
    };
    let key = lhs.trim();
    let rhs = rhs.trim();
    match key {
        "FxMuzzleFlash" | "FxChargeMuzzleFlash" => {
            if let Some(et) = extract_et_name(rhs) {
                fx.muzzle.push(et);
            }
            fx.direct_bp_muzzle.extend(extract_quoted_bps(rhs));
        }
        "FxImpactUnit" | "FxImpactLand" | "FxImpactAirUnit" | "FxImpactWater"
        | "FxImpactProp" | "FxImpactNone" => {
            if let Some(et) = extract_et_name(rhs)
                && !fx.impact.contains(&et)
            {
                fx.impact.push(et);
            }
            fx.direct_bp_impact.extend(extract_quoted_bps(rhs));
        }
        "FxTrails" => {
            if let Some(et) = extract_et_name(rhs) {
                fx.trails.push(et);
            }
            fx.direct_bp_trails.extend(extract_quoted_bps(rhs));
        }
        "PolyTrail" | "PolyTrails" => {
            if let Some(et) = extract_et_name(rhs) {
                fx.poly_trail.push(et);
            }
            fx.direct_bp_trails.extend(extract_quoted_bps(rhs));
        }
        "BeamType" => {
            if let Some(tail) = rhs.split('.').next_back() {
                let name: String = tail
                    .bytes()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == b'_')
                    .map(|c| c as char)
                    .collect();
                if !name.is_empty() {
                    fx.beam_type = Some(name);
                }
            }
        }
        "FxBeam" | "FxBeamStartPoint" | "FxBeamEndPoint" => {
            let bps = extract_quoted_bps(rhs);
            if key == "FxBeam" {
                fx.fx_beam.extend(bps);
            } else if key == "FxBeamEndPoint" {
                fx.direct_bp_impact.extend(bps);
            } else {
                fx.direct_bp_muzzle.extend(bps);
            }
        }
        _ => {}
    }
}

/// Parses a `projectiles/*/*_script.lua` to extract its base projectile class and any direct FX.
fn parse_projectile_script(lua_text: &str) -> LuaClassFx {
    let clean = strip_lua_comments(lua_text);
    let mut fx = LuaClassFx::default();
    for line in clean.lines() {
        let t = line.trim();
        if fx.parent.is_none() && t.contains("Class(") {
            fx.parent = extract_class_parent(t);
        }
        extract_fx_fields_from_line(t, &mut fx);
    }
    fx
}

/// Parses a `units/*/*_script.lua` to map `weapon_label_lowercase -> WeaponClassName`.
fn parse_unit_script_weapons(lua_text: &str) -> HashMap<String, String> {
    let clean = strip_lua_comments(lua_text);
    let mut out = HashMap::new();
    for line in clean.lines() {
        let t = line.trim();
        if let Some((lhs, rhs)) = t.split_once('=') {
            let label = lhs.trim();
            let rhs_t = rhs.trim();
            if is_ident(label)
                && rhs_t.starts_with("Class")
                && rhs_t.contains('(')
                && let Some(parent) = extract_class_parent(rhs_t)
            {
                out.insert(label.to_ascii_lowercase(), parent);
            }
        }
    }
    out
}

/// Resolves inherited `LuaClassFx` fields up the class hierarchy (bounded depth 16).
fn resolve_class_fx(
    start: &str,
    classes: &HashMap<String, LuaClassFx>,
) -> LuaClassFx {
    let mut acc = LuaClassFx::default();
    let mut cur = Some(start);
    let mut depth = 0usize;
    while let Some(name) = cur {
        if depth >= 16 {
            break;
        }
        depth += 1;
        let Some(c) = classes.get(name) else {
            break;
        };
        if acc.muzzle.is_empty() {
            acc.muzzle = c.muzzle.clone();
        }
        if acc.impact.is_empty() {
            acc.impact = c.impact.clone();
        }
        if acc.trails.is_empty() {
            acc.trails = c.trails.clone();
        }
        if acc.poly_trail.is_empty() {
            acc.poly_trail = c.poly_trail.clone();
        }
        if acc.beam_type.is_none() {
            acc.beam_type = c.beam_type.clone();
        }
        if acc.fx_beam.is_empty() {
            acc.fx_beam = c.fx_beam.clone();
        }
        if acc.direct_bp_muzzle.is_empty() {
            acc.direct_bp_muzzle = c.direct_bp_muzzle.clone();
        }
        if acc.direct_bp_impact.is_empty() {
            acc.direct_bp_impact = c.direct_bp_impact.clone();
        }
        if acc.direct_bp_trails.is_empty() {
            acc.direct_bp_trails = c.direct_bp_trails.clone();
        }
        cur = c.parent.as_deref();
    }
    acc
}

// ---------------------------------------------------------------------------
// Weapon -> FA effect mapping & coverage reporting
// ---------------------------------------------------------------------------

/// Faction index (`0 = UEF`, `1 = Aeon`, `2 = Cybran`, `3 = Seraphim`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Faction {
    Uef = 0,
    Aeon = 1,
    Cybran = 2,
    Seraphim = 3,
}

impl Faction {
    pub fn from_unit_id(unit_id: &str) -> Self {
        let up = unit_id.to_ascii_uppercase();
        if up.starts_with("UA") || up.starts_with("XA") || up.starts_with("DA") {
            Faction::Aeon
        } else if up.starts_with("UR") || up.starts_with("XR") || up.starts_with("DR") {
            Faction::Cybran
        } else if up.starts_with("XS") || up.starts_with("US") || up.starts_with("DS") {
            Faction::Seraphim
        } else {
            Faction::Uef
        }
    }

    pub fn prefix_char(self) -> char {
        match self {
            Faction::Uef => 'T',
            Faction::Aeon => 'A',
            Faction::Cybran => 'C',
            Faction::Seraphim => 'S',
        }
    }
}

/// How a weapon was mapped to FA effect blueprints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingSource {
    /// Mapped directly via unit/projectile Lua scripts -> `EffectTemplates.lua`.
    ExactLua,
    /// Mapped by matching weapon label / projectile name tokens to `EffectTemplates.lua`.
    NameMatch,
    /// Mapped via faction + `WeaponCategory` / `Label` generic FA template.
    FactionCategoryFallback,
    /// No valid FA effect blueprint found.
    Unmapped,
}

/// Resolved FA blueprint paths for one weapon.
#[derive(Debug, Clone)]
pub struct WeaponEffectPaths {
    pub is_death: bool,
    pub muzzle: Vec<String>,
    pub impact: Vec<String>,
    pub trails: Vec<String>,
    pub beams: Vec<String>,
    pub source: MappingSource,
}

impl WeaponEffectPaths {
    pub fn has_any_fa_effect(&self) -> bool {
        !self.muzzle.is_empty()
            || !self.impact.is_empty()
            || !self.trails.is_empty()
            || !self.beams.is_empty()
    }
}

/// Weapon mapping coverage summary returned by `farust effects --coverage`.
#[derive(Debug, Clone, Default)]
pub struct CoverageReport {
    pub total_weapons: usize,
    pub mapped_weapons: usize,
    pub exact_lua: usize,
    pub name_match: usize,
    pub faction_fallback: usize,
    pub unmapped: usize,
    pub combat_total: usize,
    pub combat_mapped: usize,
    pub templates_loaded: usize,
    pub valid_effect_bps: usize,
}

impl CoverageReport {
    pub fn pct(&self) -> f64 {
        if self.total_weapons == 0 {
            0.0
        } else {
            self.mapped_weapons as f64 * 100.0 / self.total_weapons as f64
        }
    }

    pub fn combat_pct(&self) -> f64 {
        if self.combat_total == 0 {
            0.0
        } else {
            self.combat_mapped as f64 * 100.0 / self.combat_total as f64
        }
    }
}

/// Metadata extracted from a unit blueprint's `Weapon = { ... }` for effect mapping.
#[derive(Debug, Clone)]
pub struct WeaponSpec {
    pub unit_id: String,
    pub faction: Faction,
    pub label: String,
    pub category: String,
    pub projectile_id: Option<String>,
    pub dummy: bool,
    pub damage: f32,
}

pub fn extract_weapon_specs(unit_id: &str, bp: &Value) -> Vec<WeaponSpec> {
    let faction = Faction::from_unit_id(unit_id);
    let Some(wlist) = bp.get("Weapon") else {
        return Vec::new();
    };
    wlist
        .table_list()
        .into_iter()
        .map(|w| {
            let label = w.get_str("Label").unwrap_or("Weapon").to_string();
            let category = w.get_str("WeaponCategory").unwrap_or("").to_string();
            let projectile_id = w
                .get_str("ProjectileId")
                .map(norm_asset_path)
                .filter(|s| !s.is_empty());
            let dummy = w.get_bool("DummyWeapon").unwrap_or(false);
            let damage = w.get_f64("Damage").map(|v| finite_or(v, 0.0)).unwrap_or(0.0);
            WeaponSpec {
                unit_id: unit_id.to_string(),
                faction,
                label,
                category,
                projectile_id,
                dummy,
                damage,
            }
        })
        .collect()
}

/// Resolves a list of template names & direct `.bp` paths against `templates` and `valid_bps`.
fn resolve_bp_list(
    template_names: &[String],
    direct_bps: &[String],
    templates: &BTreeMap<String, Vec<String>>,
    valid_bps: &BTreeSet<String>,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut push_valid = |p: &str| {
        let n = norm_asset_path(p);
        if valid_bps.contains(&n) && !out.contains(&n) {
            out.push(n);
        }
    };
    for name in template_names {
        if let Some(list) = templates.get(name) {
            for p in list {
                push_valid(p);
            }
        }
    }
    for p in direct_bps {
        push_valid(p);
    }
    out
}

/// Faction + category fallback template candidates (all real `EffectTemplates.lua` names).
fn fallback_templates(faction: Faction, category: &str, label: &str) -> (&'static [&'static str], &'static [&'static str]) {
    let cat = category.to_ascii_lowercase();
    let lbl = label.to_ascii_lowercase();
    if cat.contains("death") || lbl.contains("death") {
        if lbl.contains("nuke") || lbl.contains("commander") {
            return (&["TNukeBaseEffects01"], &["TNukeHeadEffects01", "TNukeRings01", "ExplosionEffectsLrg01"]);
        }
        return (&["FlashSml01"], &["ExplosionEffectsMed01", "DefaultHitExplosion01"]);
    }
    if cat.contains("missile") || lbl.contains("missile") || lbl.contains("nuke") {
        return match faction {
            Faction::Uef => (&["TPlasmaCannonLightMuzzleFlash", "FlashSml01"], &["TMissileHit01", "DefaultMissileHit01"]),
            Faction::Aeon => (&["AMissileMuzzleFlash", "ADisruptorCannonMuzzle01", "FlashSml01"], &["AMissileHit01", "DefaultMissileHit01"]),
            Faction::Cybran => (&["CLaserMuzzleFlash01", "FlashSml01"], &["CMissileHit01", "DefaultMissileHit01"]),
            Faction::Seraphim => (&["SIFInainoLaunch01", "SDFAireauWeaponMuzzleFlash", "FlashSml01"], &["SMissileHit01", "DefaultMissileHit01"]),
        };
    }
    if cat.contains("artillery") || cat.contains("bomb") || lbl.contains("artillery") || lbl.contains("mortar") || lbl.contains("bomb") {
        return match faction {
            Faction::Uef => (&["TPlasmaCannonHeavyMuzzleFlash", "FlashSml01"], &["TAntiMatterShellHit01", "DefaultHitExplosion01"]),
            Faction::Aeon => (&["AIFBallisticMortarFlash02", "AQuantumCannonMuzzle01", "FlashSml01"], &["AQuarkBombHitUnit01", "DefaultHitExplosion01"]),
            Faction::Cybran => (&["CHvyProtonCannonMuzzleflash", "CLaserMuzzleFlash01", "FlashSml01"], &["CArtilleryHit01", "DefaultHitExplosion01"]),
            Faction::Seraphim => (&["SDFAireauWeaponMuzzleFlash", "FlashSml01"], &["SThunthoArtilleryShellHit01", "DefaultHitExplosion01"]),
        };
    }
    if cat.contains("anti air") || lbl.contains("aa") || lbl.contains("flak") {
        return match faction {
            Faction::Uef => (&["TPlasmaGatlingCannonMuzzleFlash", "TPlasmaCannonLightMuzzleFlash", "FlashSml01"], &["TFragmentationShell01", "DefaultHitExplosion02"]),
            Faction::Aeon => (&["AGravitonBolterMuzzleFlash01", "ADisruptorCannonMuzzle01", "FlashSml01"], &["AAAZealotMissileHit01", "DefaultHitExplosion02"]),
            Faction::Cybran => (&["CElectronBolterMuzzleFlash01", "CLaserMuzzleFlash01", "FlashSml01"], &["CAANanoDartHit01", "DefaultHitExplosion02"]),
            Faction::Seraphim => (&["SDFAireauWeaponMuzzleFlash", "FlashSml01"], &["SAALosaareAutoCannonHit", "DefaultHitExplosion02"]),
        };
    }
    if cat.contains("anti navy") || lbl.contains("torpedo") || lbl.contains("depth") {
        return match faction {
            Faction::Uef => (&["TPlasmaCannonLightMuzzleFlash", "FlashSml01"], &["TTorpedoHitUnit01", "DefaultHitExplosion02"]),
            Faction::Aeon => (&["ADisruptorCannonMuzzle01", "FlashSml01"], &["ATorpedoHit01", "DefaultHitExplosion02"]),
            Faction::Cybran => (&["CElectronBolterMuzzleFlash01", "FlashSml01"], &["CTorpedoHit01", "DefaultHitExplosion02"]),
            Faction::Seraphim => (&["SAnaitTorpedoMuzzleFlash", "FlashSml01"], &["STorpedoHit01", "DefaultHitExplosion02"]),
        };
    }
    // Direct Fire / Defense / Experimental / default
    match faction {
        Faction::Uef => (&["TPlasmaCannonLightMuzzleFlash", "TPlasmaCannonHeavyMuzzleFlash", "FlashSml01"], &["TGaussCannonHitUnit01", "DefaultHitExplosion01"]),
        Faction::Aeon => (&["ADisruptorCannonMuzzle01", "AQuantumCannonMuzzle01", "FlashSml01"], &["ALaserHit01", "DefaultHitExplosion01"]),
        Faction::Cybran => (&["CLaserMuzzleFlash01", "CElectronBolterMuzzleFlash01", "FlashSml01"], &["CLaserHit01", "DefaultHitExplosion01"]),
        Faction::Seraphim => (&["SDFAireauWeaponMuzzleFlash", "SDFSinnutheWeaponMuzzleFlash", "FlashSml01"], &["SPhasonLaserHit01", "DefaultHitExplosion01"]),
    }
}

/// Attempts a token match between a projectile stem or weapon label and `EffectTemplates` keys.
fn match_template_by_name(
    faction: Faction,
    proj_id: Option<&str>,
    label: &str,
    templates: &BTreeMap<String, Vec<String>>,
    valid_bps: &BTreeSet<String>,
) -> (Vec<String>, Vec<String>) {
    let mut tokens: Vec<String> = Vec::new();
    if let Some(pid) = proj_id
        && let Some(file) = pid.rsplit('/').next()
    {
        let stem = file.trim_end_matches("_proj.bp").trim_end_matches(".bp");
        if stem.len() >= 4 {
            tokens.push(stem.to_ascii_lowercase());
            // Strip leading 3-letter FA prefix (e.g. `tdf`, `adf`, `cdf`, `sdf`, `taa`)
            if stem.len() >= 7 {
                tokens.push(stem[3..].to_ascii_lowercase());
            }
        }
    }
    let l = label.to_ascii_lowercase();
    if l.len() >= 5 {
        tokens.push(l);
    }
    if tokens.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let fprefix = faction.prefix_char().to_ascii_lowercase();
    let mut muzzle = Vec::new();
    let mut impact = Vec::new();
    for (tname, bps) in templates {
        let tl = tname.to_ascii_lowercase();
        if !tl.starts_with(fprefix) {
            continue;
        }
        let matched = tokens.iter().any(|tok| tok.len() >= 5 && tl.contains(tok.as_str()));
        if !matched {
            continue;
        }
        let valid: Vec<String> = bps
            .iter()
            .filter(|p| valid_bps.contains(p.as_str()))
            .cloned()
            .collect();
        if valid.is_empty() {
            continue;
        }
        if tl.contains("muzzle") || tl.contains("flash") {
            if muzzle.is_empty() {
                muzzle = valid;
            }
        } else if (tl.contains("hit") || tl.contains("impact") || tl.contains("explosion"))
            && impact.is_empty()
        {
            impact = valid;
        }
        if !muzzle.is_empty() && !impact.is_empty() {
            break;
        }
    }
    (muzzle, impact)
}

/// Maps a single weapon spec using Lua class links, name matching, and faction/category fallback.
#[allow(clippy::too_many_arguments)]
fn map_weapon(
    spec: &WeaponSpec,
    unit_script_weapons: &HashMap<(String, String), String>,
    weapon_classes: &HashMap<String, LuaClassFx>,
    beam_classes: &HashMap<String, LuaClassFx>,
    proj_scripts: &HashMap<String, LuaClassFx>,
    proj_classes: &HashMap<String, LuaClassFx>,
    templates: &BTreeMap<String, Vec<String>>,
    valid_bps: &BTreeSet<String>,
) -> WeaponEffectPaths {
    let uid_l = spec.unit_id.to_ascii_lowercase();
    let lbl_l = spec.label.to_ascii_lowercase();
    let is_death = spec.category.eq_ignore_ascii_case("Death") || lbl_l.contains("death");

    let mut muzzle = Vec::new();
    let mut impact = Vec::new();
    let mut trails = Vec::new();
    let mut beams = Vec::new();
    let mut exact_hit = false;

    // 1a. Unit script -> WeaponClass -> FxMuzzleFlash / BeamType
    if let Some(wclass) = unit_script_weapons.get(&(uid_l, lbl_l)) {
        let wfx = resolve_class_fx(wclass, weapon_classes);
        muzzle.extend(resolve_bp_list(&wfx.muzzle, &wfx.direct_bp_muzzle, templates, valid_bps));
        if let Some(ref bt) = wfx.beam_type {
            let bfx = resolve_class_fx(bt, beam_classes);
            for bp in &bfx.fx_beam {
                let n = norm_asset_path(bp);
                if valid_bps.contains(&n) && !beams.contains(&n) {
                    beams.push(n);
                }
            }
            impact.extend(resolve_bp_list(&bfx.impact, &bfx.direct_bp_impact, templates, valid_bps));
            muzzle.extend(resolve_bp_list(&bfx.muzzle, &bfx.direct_bp_muzzle, templates, valid_bps));
        }
        if !muzzle.is_empty() || !beams.is_empty() || !impact.is_empty() {
            exact_hit = true;
        }
    }

    // 1b. ProjectileId -> projectiles/*_script.lua -> ProjectileClass -> FxImpact* / FxTrails / PolyTrail
    if let Some(ref pid) = spec.projectile_id
        && let Some(pscript) = proj_scripts.get(pid)
    {
        impact.extend(resolve_bp_list(&pscript.impact, &pscript.direct_bp_impact, templates, valid_bps));
        trails.extend(resolve_bp_list(&pscript.trails, &pscript.direct_bp_trails, templates, valid_bps));
        trails.extend(resolve_bp_list(&pscript.poly_trail, &[], templates, valid_bps));
        if let Some(ref parent) = pscript.parent {
            let pfx = resolve_class_fx(parent, proj_classes);
            if impact.is_empty() {
                impact.extend(resolve_bp_list(&pfx.impact, &pfx.direct_bp_impact, templates, valid_bps));
            }
            if trails.is_empty() {
                trails.extend(resolve_bp_list(&pfx.trails, &pfx.direct_bp_trails, templates, valid_bps));
                trails.extend(resolve_bp_list(&pfx.poly_trail, &[], templates, valid_bps));
            }
        }
        if !impact.is_empty() || !trails.is_empty() {
            exact_hit = true;
        }
    }

    let mut source = if exact_hit {
        MappingSource::ExactLua
    } else {
        MappingSource::Unmapped
    };

    // 2. Token/stem match against EffectTemplates keys when exact Lua missed muzzle or impact
    if muzzle.is_empty() || impact.is_empty() {
        let (nm, ni) = match_template_by_name(
            spec.faction,
            spec.projectile_id.as_deref(),
            &spec.label,
            templates,
            valid_bps,
        );
        if muzzle.is_empty() && !nm.is_empty() {
            muzzle = nm;
            if source == MappingSource::Unmapped {
                source = MappingSource::NameMatch;
            }
        }
        if impact.is_empty() && !ni.is_empty() {
            impact = ni;
            if source == MappingSource::Unmapped {
                source = MappingSource::NameMatch;
            }
        }
    }

    // 3. Faction + WeaponCategory authentic FA template fallback
    let (fb_muzzle, fb_impact) = fallback_templates(spec.faction, &spec.category, &spec.label);
    if muzzle.is_empty() {
        for cand in fb_muzzle {
            let resolved = resolve_bp_list(&[(*cand).to_string()], &[], templates, valid_bps);
            if !resolved.is_empty() {
                muzzle = resolved;
                if source == MappingSource::Unmapped {
                    source = MappingSource::FactionCategoryFallback;
                }
                break;
            }
        }
    }
    if impact.is_empty() {
        for cand in fb_impact {
            let resolved = resolve_bp_list(&[(*cand).to_string()], &[], templates, valid_bps);
            if !resolved.is_empty() {
                impact = resolved;
                if source == MappingSource::Unmapped {
                    source = MappingSource::FactionCategoryFallback;
                }
                break;
            }
        }
    }

    WeaponEffectPaths {
        is_death,
        muzzle,
        impact,
        trails,
        beams,
        source,
    }
}

// ---------------------------------------------------------------------------
// Compiled runtime effect library + GPU VFX texture array
// ---------------------------------------------------------------------------

/// Ready-to-spawn compiled emitter backed by an FA `EmitterBlueprint` and GPU texture array layer.
#[derive(Debug, Clone)]
pub struct CompiledEmitter {
    pub bp: EmitterBp,
    pub tex_layer: u32,
    pub has_texture: bool,
    pub ramp: ColorRamp,
    /// Precomputed burst particle count per single trigger (clamped for particle budget).
    pub burst_count: usize,
}

/// Ready-to-draw compiled FA `BeamBlueprint`.
#[derive(Debug, Clone)]
pub struct CompiledBeam {
    pub bp: BeamBp,
    pub tex_layer: u32,
    pub has_texture: bool,
}

/// Ready-to-draw compiled FA `TrailEmitterBlueprint`.
#[derive(Debug, Clone)]
pub struct CompiledTrail {
    pub bp: TrailBp,
    pub tex_layer: u32,
    pub has_texture: bool,
    pub ramp: ColorRamp,
}

/// All compiled weapon effects for a unit type (primary combat weapon).
#[derive(Debug, Clone, Default)]
pub struct UnitWeaponFx {
    pub muzzle: Vec<CompiledEmitter>,
    pub impact: Vec<CompiledEmitter>,
    pub trail_emitters: Vec<CompiledEmitter>,
    pub poly_trail: Option<CompiledTrail>,
    pub beam: Option<CompiledBeam>,
}

/// Packed uniform-size RGBA8 layers for the GPU `texture_2d_array`.
#[derive(Debug, Clone)]
pub struct VfxTextureArray {
    pub width: u32,
    pub height: u32,
    pub layers: Vec<RgbaImage>,
}

/// Complete render-only FA effect library attached to `LivingBattlefield`.
#[derive(Debug, Clone, Default)]
pub struct EffectLibrary {
    pub by_unit: HashMap<String, UnitWeaponFx>,
    pub shield_hit: [Vec<CompiledEmitter>; 4],
    pub build_beam: [Option<CompiledBeam>; 4],
    pub build_sparks: [Vec<CompiledEmitter>; 4],
    pub explosion_small: Vec<CompiledEmitter>,
    pub explosion_med: Vec<CompiledEmitter>,
    pub explosion_large: Vec<CompiledEmitter>,
    pub nuke: Vec<CompiledEmitter>,
    pub texture_array: Option<VfxTextureArray>,
}

impl EffectLibrary {
    pub fn is_loaded(&self) -> bool {
        self.texture_array.is_some() && !self.by_unit.is_empty()
    }

    pub fn unit_weapon_fx(&self, unit_id: &str) -> Option<&UnitWeaponFx> {
        self.by_unit.get(unit_id)
    }

    pub fn shield_hit_fx(&self, faction: Faction) -> &[CompiledEmitter] {
        &self.shield_hit[faction as usize]
    }

    pub fn build_beam_fx(&self, faction: Faction) -> Option<&CompiledBeam> {
        self.build_beam[faction as usize].as_ref()
    }

    pub fn build_sparks_fx(&self, faction: Faction) -> &[CompiledEmitter] {
        &self.build_sparks[faction as usize]
    }

    pub fn explosion_fx(&self, radius: f32) -> &[CompiledEmitter] {
        if radius >= 4.5 && !self.explosion_large.is_empty() {
            &self.explosion_large
        } else if radius >= 2.0 && !self.explosion_med.is_empty() {
            &self.explosion_med
        } else if !self.explosion_small.is_empty() {
            &self.explosion_small
        } else {
            &self.explosion_med
        }
    }
}

/// Computes a bounded burst count from an emitter's `EmitRateCurve` and `Lifetime`.
pub fn emitter_burst_count(bp: &EmitterBp) -> usize {
    let rate = bp.emit_rate.mid_y().max(bp.emit_rate.max_abs_y() * 0.5).max(0.0);
    let ticks = if bp.lifetime > 0.0 {
        bp.lifetime.min(2.0)
    } else {
        1.0
    };
    let raw = (rate * ticks).round() as usize;
    raw.clamp(1, 5)
}

/// Soft radial white fallback image for texture array layer 0.
fn make_fallback_sprite(size: u32) -> RgbaImage {
    let mut pixels = vec![0u8; (size * size * 4) as usize];
    let half = size as f32 * 0.5;
    for y in 0..size {
        for x in 0..size {
            let dx = (x as f32 + 0.5 - half) / half;
            let dy = (y as f32 + 0.5 - half) / half;
            let d = (dx * dx + dy * dy).sqrt();
            let a = (1.0 - d).clamp(0.0, 1.0).powf(1.6);
            let u = (a * 255.0).round() as u8;
            let i = ((y * size + x) * 4) as usize;
            pixels[i] = 255;
            pixels[i + 1] = 255;
            pixels[i + 2] = 255;
            pixels[i + 3] = u;
        }
    }
    RgbaImage { width: size, height: size, pixels }
}

/// Loads Lua templates & class links from `lua.scd`, `projectiles.scd`, and `units.scd`.
struct LuaEffectIndex {
    templates: BTreeMap<String, Vec<String>>,
    weapon_classes: HashMap<String, LuaClassFx>,
    beam_classes: HashMap<String, LuaClassFx>,
    proj_classes: HashMap<String, LuaClassFx>,
    proj_scripts: HashMap<String, LuaClassFx>,
    unit_script_weapons: HashMap<(String, String), String>,
}

fn load_lua_effect_index(game_dir: &Path) -> LuaEffectIndex {
    let gamedata = scd::gamedata_dir(game_dir);
    let lua_scd = gamedata.join("lua.scd");
    let proj_scd = gamedata.join("projectiles.scd");
    let units_scd = gamedata.join("units.scd");

    let (lua_files, proj_files, unit_scripts) = std::thread::scope(|s| {
        let j1 = s.spawn(|| {
            scd::read_scd_matching(&lua_scd, |n| {
                let l = norm_asset_path(n);
                matches!(
                    l.as_str(),
                    "lua/effecttemplates.lua"
                        | "lua/terranweapons.lua"
                        | "lua/aeonweapons.lua"
                        | "lua/cybranweapons.lua"
                        | "lua/seraphimweapons.lua"
                        | "lua/sim/defaultweapons.lua"
                        | "lua/defaultcollisionbeams.lua"
                        | "lua/terranprojectiles.lua"
                        | "lua/aeonprojectiles.lua"
                        | "lua/cybranprojectiles.lua"
                        | "lua/seraphimprojectiles.lua"
                        | "lua/sim/defaultprojectiles.lua"
                )
            })
            .unwrap_or_default()
        });
        let j2 = s.spawn(|| {
            scd::read_scd_matching(&proj_scd, |n| n.to_ascii_lowercase().ends_with("_script.lua"))
                .unwrap_or_default()
        });
        let j3 = s.spawn(|| {
            scd::read_scd_matching(&units_scd, |n| n.to_ascii_lowercase().ends_with("_script.lua"))
                .unwrap_or_default()
        });
        (
            j1.join().unwrap_or_default(),
            j2.join().unwrap_or_default(),
            j3.join().unwrap_or_default(),
        )
    });

    let mut templates = BTreeMap::new();
    let mut weapon_classes = HashMap::new();
    let mut beam_classes = HashMap::new();
    let mut proj_classes = HashMap::new();

    for (name, bytes) in lua_files {
        let l = norm_asset_path(&name);
        let text = String::from_utf8_lossy(&bytes);
        if l == "lua/effecttemplates.lua" {
            templates = parse_effect_templates(&text);
        } else if l == "lua/defaultcollisionbeams.lua" {
            parse_lua_class_fx(&text, &mut beam_classes);
        } else if l.contains("weapon") {
            parse_lua_class_fx(&text, &mut weapon_classes);
        } else if l.contains("projectile") {
            parse_lua_class_fx(&text, &mut proj_classes);
        }
    }

    let mut proj_scripts = HashMap::new();
    for (name, bytes) in proj_files {
        let n = norm_asset_path(&name);
        if let Some(stem) = n.strip_suffix("_script.lua") {
            let proj_bp = format!("{stem}_proj.bp");
            let text = String::from_utf8_lossy(&bytes);
            proj_scripts.insert(proj_bp, parse_projectile_script(&text));
        }
    }

    let mut unit_script_weapons = HashMap::new();
    for (name, bytes) in unit_scripts {
        let file = norm_asset_path(&name);
        let fname = file.rsplit('/').next().unwrap_or(&file);
        if let Some(uid) = fname.strip_suffix("_script.lua") {
            let text = String::from_utf8_lossy(&bytes);
            for (lbl, cls) in parse_unit_script_weapons(&text) {
                unit_script_weapons.insert((uid.to_string(), lbl), cls);
            }
        }
    }

    LuaEffectIndex {
        templates,
        weapon_classes,
        beam_classes,
        proj_classes,
        proj_scripts,
        unit_script_weapons,
    }
}

/// Loads and compiles the FA `EffectLibrary` for every unit in `unit_ids` (plus global
/// shield/build/explosion/nuke effects) in parallel.
pub fn load_effect_library(game_dir: &Path, unit_ids: &[String]) -> EffectLibrary {
    let gamedata = scd::gamedata_dir(game_dir);
    let effects_scd = gamedata.join("effects.scd");
    let units_scd = gamedata.join("units.scd");
    let textures_scd = gamedata.join("textures.scd");
    if !effects_scd.is_file() {
        return EffectLibrary::default();
    }

    let wanted_units: BTreeSet<String> = unit_ids
        .iter()
        .map(|id| {
            let l = id.to_ascii_lowercase();
            format!("units/{l}/{l}_unit.bp")
        })
        .collect();

    // Sweep archives in parallel: Lua index + unit blueprints + effects.scd blueprint list.
    let (lua_idx, unit_bps_raw, effect_bps_raw) = std::thread::scope(|s| {
        let j_lua = s.spawn(|| load_lua_effect_index(game_dir));
        let j_units = s.spawn(|| {
            scd::read_scd_matching(&units_scd, |n| wanted_units.contains(&norm_asset_path(n)))
                .unwrap_or_default()
        });
        let j_eff = s.spawn(|| {
            scd::read_scd_matching(&effects_scd, |n| n.to_ascii_lowercase().ends_with(".bp"))
                .unwrap_or_default()
        });
        (
            j_lua.join().unwrap_or_else(|_| LuaEffectIndex {
                templates: BTreeMap::new(),
                weapon_classes: HashMap::new(),
                beam_classes: HashMap::new(),
                proj_classes: HashMap::new(),
                proj_scripts: HashMap::new(),
                unit_script_weapons: HashMap::new(),
            }),
            j_units.join().unwrap_or_default(),
            j_eff.join().unwrap_or_default(),
        )
    });

    let effect_raw_map: HashMap<String, Vec<u8>> = effect_bps_raw
        .into_iter()
        .map(|(k, v)| (norm_asset_path(&k), v))
        .collect();
    let valid_bps: BTreeSet<String> = effect_raw_map.keys().cloned().collect();

    // Map primary combat weapon for each catalog unit.
    let threads = default_thread_count();
    let unit_mappings: Vec<(String, WeaponEffectPaths)> = par_map_ordered(&unit_bps_raw, threads, |_, (path, bytes)| {
        let uid = blueprint::unit_id_from_path(path);
        let bp = blueprint::parse_blueprint(&String::from_utf8_lossy(bytes)).ok()?;
        let specs = extract_weapon_specs(&uid, &bp);
        // Pick primary non-death, non-dummy combat weapon with highest damage (or first weapon).
        let primary = specs
            .iter()
            .filter(|w| !w.dummy && !w.category.eq_ignore_ascii_case("Death") && !w.label.to_ascii_lowercase().contains("death"))
            .max_by(|a, b| a.damage.total_cmp(&b.damage))
            .or_else(|| specs.first())?;
        let mapped = map_weapon(
            primary,
            &lua_idx.unit_script_weapons,
            &lua_idx.weapon_classes,
            &lua_idx.beam_classes,
            &lua_idx.proj_scripts,
            &lua_idx.proj_classes,
            &lua_idx.templates,
            &valid_bps,
        );
        Some((uid, mapped))
    })
    .into_iter()
    .flatten()
    .collect();

    // Global template paths (shields, build beams/sparks, explosions, nukes).
    let tpl_paths = |names: &[&str]| -> Vec<String> {
        let mut out = Vec::new();
        for &n in names {
            if let Some(list) = lua_idx.templates.get(n) {
                for p in list {
                    if valid_bps.contains(p) && !out.contains(p) {
                        out.push(p.clone());
                    }
                }
            }
        }
        out
    };
    let shield_paths: [Vec<String>; 4] = [
        tpl_paths(&["UEFShieldHit01"]),
        tpl_paths(&["AeonShieldHit01"]),
        tpl_paths(&["CybranShieldHit01"]),
        tpl_paths(&["SeraphimShieldHit01"]),
    ];
    let build_beam_paths: [Option<String>; 4] = [
        valid_bps.contains("effects/emitters/build_beam_01_emit.bp").then(|| "effects/emitters/build_beam_01_emit.bp".to_string()),
        tpl_paths(&["AeonBuildBeams01", "AeonBuildBeams02"]).into_iter().next()
            .or_else(|| valid_bps.contains("effects/emitters/build_beam_01_emit.bp").then(|| "effects/emitters/build_beam_01_emit.bp".to_string())),
        valid_bps.contains("effects/emitters/build_beam_02_emit.bp").then(|| "effects/emitters/build_beam_02_emit.bp".to_string()),
        tpl_paths(&["SeraphimBuildBeams01"]).into_iter().next()
            .or_else(|| valid_bps.contains("effects/emitters/build_beam_01_emit.bp").then(|| "effects/emitters/build_beam_01_emit.bp".to_string())),
    ];
    let build_spark_paths: [Vec<String>; 4] = [
        tpl_paths(&["DefaultBuildUnit01", "CybranBuildSparks01"]),
        tpl_paths(&["DefaultBuildUnit01"]),
        tpl_paths(&["CybranBuildSparks01", "CybranBuildFlash01"]),
        tpl_paths(&["DefaultBuildUnit01"]),
    ];
    let exp_small_paths = tpl_paths(&["ExplosionEffectsSml01", "DefaultHitExplosion02"]);
    let exp_med_paths = tpl_paths(&["ExplosionEffectsMed01", "DefaultHitExplosion01"]);
    let exp_large_paths = tpl_paths(&["ExplosionEffectsLrg01", "ConcussionRingLrg01"]);
    let nuke_paths = tpl_paths(&["TNukeRings01", "TNukeHeadEffects01", "TNukeBaseEffects01", "TNukeFlavorPlume01"]);

    // Collect every actually-referenced effect blueprint path so we only parse & decode what's used.
    let mut used_bps: BTreeSet<String> = BTreeSet::new();
    for (_, m) in &unit_mappings {
        for p in m.muzzle.iter().take(3) {
            used_bps.insert(p.clone());
        }
        for p in m.impact.iter().take(4) {
            used_bps.insert(p.clone());
        }
        for p in m.trails.iter().take(2) {
            used_bps.insert(p.clone());
        }
        for p in m.beams.iter().take(2) {
            used_bps.insert(p.clone());
        }
    }
    for list in shield_paths
        .iter()
        .chain(build_spark_paths.iter())
        .chain([&exp_small_paths, &exp_med_paths, &exp_large_paths, &nuke_paths])
    {
        for p in list.iter().take(5) {
            used_bps.insert(p.clone());
        }
    }
    for p in build_beam_paths.iter().flatten() {
        used_bps.insert(p.clone());
    }

    // Parse all used effect blueprints in parallel.
    let used_list: Vec<String> = used_bps.into_iter().collect();
    let parsed_bps: HashMap<String, EffectBp> = par_map_ordered(&used_list, threads, |_, path| {
        let bytes = effect_raw_map.get(path)?;
        let bp = parse_effect_blueprint(&String::from_utf8_lossy(bytes)).ok()?;
        Some((path.clone(), bp))
    })
    .into_iter()
    .flatten()
    .collect();

    // Tally sprite textures by frequency + collect ramp textures.
    let mut sprite_freq: HashMap<String, usize> = HashMap::new();
    let mut ramp_paths: BTreeSet<String> = BTreeSet::new();
    for bp in parsed_bps.values() {
        match bp {
            EffectBp::Emitter(e) => {
                if !e.texture.is_empty() {
                    *sprite_freq.entry(e.texture.clone()).or_insert(0) += 1;
                }
                if !e.ramp_texture.is_empty() {
                    ramp_paths.insert(e.ramp_texture.clone());
                }
            }
            EffectBp::Beam(b) => {
                if !b.texture.is_empty() {
                    *sprite_freq.entry(b.texture.clone()).or_insert(0) += 10; // prioritize beams
                }
            }
            EffectBp::Trail(t) => {
                if !t.repeat_texture.is_empty() {
                    *sprite_freq.entry(t.repeat_texture.clone()).or_insert(0) += 2;
                }
                if !t.ramp_texture.is_empty() {
                    ramp_paths.insert(t.ramp_texture.clone());
                }
            }
            EffectBp::Other { .. } => {}
        }
    }

    // Select up to MAX_VFX_TEX_LAYERS - 1 most-used sprite textures (layer 0 = fallback).
    let mut sprite_ranked: Vec<(String, usize)> = sprite_freq.into_iter().collect();
    sprite_ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    sprite_ranked.truncate(MAX_VFX_TEX_LAYERS.saturating_sub(1));
    let wanted_tex: BTreeSet<String> = sprite_ranked
        .iter()
        .map(|(p, _)| p.clone())
        .chain(ramp_paths.iter().cloned())
        .collect();

    // Load DDS bytes from textures.scd (and effects.scd fallback).
    let mut raw_dds: HashMap<String, Vec<u8>> = HashMap::new();
    if textures_scd.is_file()
        && let Ok(found) = scd::read_scd_matching(&textures_scd, |n| wanted_tex.contains(&norm_asset_path(n)))
    {
        for (k, v) in found {
            raw_dds.insert(norm_asset_path(&k), v);
        }
    }
    let missing_tex: BTreeSet<String> = wanted_tex
        .iter()
        .filter(|p| !raw_dds.contains_key(*p))
        .cloned()
        .collect();
    if !missing_tex.is_empty()
        && let Ok(found) = scd::read_scd_matching(&effects_scd, |n| missing_tex.contains(&norm_asset_path(n)))
    {
        for (k, v) in found {
            raw_dds.entry(norm_asset_path(&k)).or_insert(v);
        }
    }

    // Decode sprite layers in parallel and resize to uniform VFX_TEX_SIZE x VFX_TEX_SIZE.
    let sprite_names: Vec<String> = sprite_ranked.into_iter().map(|(p, _)| p).collect();
    let decoded_sprites: Vec<Option<(String, RgbaImage)>> = par_map_ordered(&sprite_names, threads, |_, path| {
        let bytes = raw_dds.get(path)?;
        let img = decode_dds_rgba8(bytes).ok()?;
        let resized = resize_rgba_bilinear(&img, VFX_TEX_SIZE, VFX_TEX_SIZE);
        Some((path.clone(), resized))
    });
    let mut layers: Vec<RgbaImage> = Vec::with_capacity(decoded_sprites.len() + 1);
    layers.push(make_fallback_sprite(VFX_TEX_SIZE));
    let mut tex_layer_map: HashMap<String, u32> = HashMap::new();
    for item in decoded_sprites.into_iter().flatten() {
        if layers.len() >= MAX_VFX_TEX_LAYERS {
            break;
        }
        let idx = layers.len() as u32;
        tex_layer_map.insert(item.0, idx);
        layers.push(item.1);
    }

    // Decode ramp DDS images in parallel.
    let ramp_list: Vec<String> = ramp_paths.into_iter().collect();
    let ramp_images: HashMap<String, RgbaImage> = par_map_ordered(&ramp_list, threads, |_, path| {
        let bytes = raw_dds.get(path)?;
        let img = decode_dds_rgba8(bytes).ok()?;
        Some((path.clone(), img))
    })
    .into_iter()
    .flatten()
    .collect();

    let compile_emitter = |path: &str| -> Option<CompiledEmitter> {
        let EffectBp::Emitter(bp) = parsed_bps.get(path)? else {
            return None;
        };
        let (tex_layer, has_texture) = match tex_layer_map.get(&bp.texture) {
            Some(&l) => (l, true),
            None => (0, false),
        };
        let ramp = if let Some(img) = ramp_images.get(&bp.ramp_texture) {
            let row = if img.height > 1 {
                (bp.ramp_select.mid_y().abs() / img.height as f32).fract()
            } else {
                0.0
            };
            ColorRamp::from_rgba_image(img, row)
        } else {
            ColorRamp::default()
        };
        let burst_count = emitter_burst_count(bp);
        Some(CompiledEmitter {
            bp: (**bp).clone(),
            tex_layer,
            has_texture,
            ramp,
            burst_count,
        })
    };
    let compile_emitters = |paths: &[String], cap: usize| -> Vec<CompiledEmitter> {
        paths
            .iter()
            .filter_map(|p| compile_emitter(p))
            .take(cap)
            .collect()
    };
    let compile_beam = |path: &str| -> Option<CompiledBeam> {
        let EffectBp::Beam(bp) = parsed_bps.get(path)? else {
            return None;
        };
        let (tex_layer, has_texture) = match tex_layer_map.get(&bp.texture) {
            Some(&l) => (l, true),
            None => (0, false),
        };
        Some(CompiledBeam {
            bp: bp.clone(),
            tex_layer,
            has_texture,
        })
    };
    let compile_trail = |path: &str| -> Option<CompiledTrail> {
        let EffectBp::Trail(bp) = parsed_bps.get(path)? else {
            return None;
        };
        let (tex_layer, has_texture) = match tex_layer_map.get(&bp.repeat_texture) {
            Some(&l) => (l, true),
            None => (0, false),
        };
        let ramp = ramp_images
            .get(&bp.ramp_texture)
            .map(|img| ColorRamp::from_rgba_image(img, 0.0))
            .unwrap_or_default();
        Some(CompiledTrail {
            bp: bp.clone(),
            tex_layer,
            has_texture,
            ramp,
        })
    };

    let mut by_unit = HashMap::with_capacity(unit_mappings.len());
    for (uid, m) in unit_mappings {
        let muzzle = compile_emitters(&m.muzzle, 3);
        let impact = compile_emitters(&m.impact, 4);
        let trail_emitters = compile_emitters(&m.trails, 2);
        let poly_trail = m.trails.iter().find_map(|p| compile_trail(p));
        let beam = m.beams.iter().find_map(|p| compile_beam(p));
        by_unit.insert(
            uid,
            UnitWeaponFx {
                muzzle,
                impact,
                trail_emitters,
                poly_trail,
                beam,
            },
        );
    }

    let shield_hit = std::array::from_fn(|i| compile_emitters(&shield_paths[i], 3));
    let build_beam = std::array::from_fn(|i| build_beam_paths[i].as_deref().and_then(compile_beam));
    let build_sparks = std::array::from_fn(|i| compile_emitters(&build_spark_paths[i], 2));

    EffectLibrary {
        by_unit,
        shield_hit,
        build_beam,
        build_sparks,
        explosion_small: compile_emitters(&exp_small_paths, 4),
        explosion_med: compile_emitters(&exp_med_paths, 4),
        explosion_large: compile_emitters(&exp_large_paths, 5),
        nuke: compile_emitters(&nuke_paths, 5),
        texture_array: Some(VfxTextureArray {
            width: VFX_TEX_SIZE,
            height: VFX_TEX_SIZE,
            layers,
        }),
    }
}

// ---------------------------------------------------------------------------
// CLI: `farust effects [--check] [--coverage] [GAME_DIR]`
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct CheckReport {
    pub total_files: usize,
    pub emitters: usize,
    pub beams: usize,
    pub trails: usize,
    pub other: usize,
    pub failures: Vec<(String, String)>,
}

/// Parses every `.bp` file in `effects.scd` and returns counts + failures. Never panics.
pub fn check_all_effect_blueprints(game_dir: &Path) -> Result<CheckReport, String> {
    let scd_path = scd::gamedata_dir(game_dir).join("effects.scd");
    let entries = scd::read_scd_matching(&scd_path, |n| n.to_ascii_lowercase().ends_with(".bp"))
        .map_err(|e| format!("{}: {e}", scd_path.display()))?;
    let threads = default_thread_count();
    let results = par_map_ordered(&entries, threads, |_, (name, bytes)| {
        let text = String::from_utf8_lossy(bytes);
        (name.clone(), parse_effect_blueprint(&text))
    });
    let mut rep = CheckReport {
        total_files: results.len(),
        ..CheckReport::default()
    };
    for (name, res) in results {
        match res {
            Ok(EffectBp::Emitter(_)) => rep.emitters += 1,
            Ok(EffectBp::Beam(_)) => rep.beams += 1,
            Ok(EffectBp::Trail(_)) => rep.trails += 1,
            Ok(EffectBp::Other { .. }) => rep.other += 1,
            Err(e) => rep.failures.push((name, e)),
        }
    }
    Ok(rep)
}

/// Measures FA emitter mapping coverage across all unit weapons in `units.scd`.
pub fn compute_weapon_coverage(game_dir: &Path) -> Result<(CoverageReport, Vec<WeaponEffectPaths>), String> {
    let gamedata = scd::gamedata_dir(game_dir);
    let effects_scd = gamedata.join("effects.scd");
    let units_scd = gamedata.join("units.scd");

    let (lua_idx, eff_entries, unit_entries) = std::thread::scope(|s| {
        let j1 = s.spawn(|| load_lua_effect_index(game_dir));
        let j2 = s.spawn(|| {
            scd::read_scd_matching(&effects_scd, |n| n.to_ascii_lowercase().ends_with(".bp"))
        });
        let j3 = s.spawn(|| {
            scd::read_scd_matching(&units_scd, |n| n.to_ascii_lowercase().ends_with("_unit.bp"))
        });
        (
            j1.join().map_err(|_| "lua index thread panicked".to_string()),
            j2.join().map_err(|_| "effects.scd thread panicked".to_string()),
            j3.join().map_err(|_| "units.scd thread panicked".to_string()),
        )
    });
    let lua_idx = lua_idx?;
    let eff_entries = eff_entries?.map_err(|e| format!("{}: {e}", effects_scd.display()))?;
    let unit_entries = unit_entries?.map_err(|e| format!("{}: {e}", units_scd.display()))?;

    let threads = default_thread_count();
    let valid_bps: BTreeSet<String> = par_map_ordered(&eff_entries, threads, |_, (name, bytes)| {
        let text = String::from_utf8_lossy(bytes);
        matches!(
            parse_effect_blueprint(&text),
            Ok(EffectBp::Emitter(_) | EffectBp::Beam(_) | EffectBp::Trail(_))
        )
        .then(|| norm_asset_path(name))
    })
    .into_iter()
    .flatten()
    .collect();

    let all_specs: Vec<WeaponSpec> = par_map_ordered(&unit_entries, threads, |_, (path, bytes)| {
        let uid = blueprint::unit_id_from_path(path);
        let bp = blueprint::parse_blueprint(&String::from_utf8_lossy(bytes)).ok()?;
        Some(extract_weapon_specs(&uid, &bp))
    })
    .into_iter()
    .flatten()
    .flatten()
    .collect();

    let mut report = CoverageReport {
        templates_loaded: lua_idx.templates.len(),
        valid_effect_bps: valid_bps.len(),
        ..CoverageReport::default()
    };
    let mut mapped_list = Vec::with_capacity(all_specs.len());
    for spec in &all_specs {
        if spec.dummy {
            continue;
        }
        let m = map_weapon(
            spec,
            &lua_idx.unit_script_weapons,
            &lua_idx.weapon_classes,
            &lua_idx.beam_classes,
            &lua_idx.proj_scripts,
            &lua_idx.proj_classes,
            &lua_idx.templates,
            &valid_bps,
        );
        report.total_weapons += 1;
        if !m.is_death {
            report.combat_total += 1;
        }
        if m.has_any_fa_effect() {
            report.mapped_weapons += 1;
            if !m.is_death {
                report.combat_mapped += 1;
            }
        }
        match m.source {
            MappingSource::ExactLua => report.exact_lua += 1,
            MappingSource::NameMatch => report.name_match += 1,
            MappingSource::FactionCategoryFallback => report.faction_fallback += 1,
            MappingSource::Unmapped => report.unmapped += 1,
        }
        mapped_list.push(m);
    }
    Ok((report, mapped_list))
}

/// CLI handler for `farust effects [--check] [--coverage] [GAME_DIR]`.
pub fn cmd_effects(args: &[String]) -> Result<(), String> {
    let do_check = args.iter().any(|a| a == "--check");
    let do_coverage = args.iter().any(|a| a == "--coverage");
    let both = !do_check && !do_coverage;
    let game_dir = args
        .iter()
        .skip(2)
        .find(|a| !a.starts_with("--"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(scd::default_game_dir);

    if do_check || both {
        let rep = check_all_effect_blueprints(&game_dir)?;
        println!(
            "effects.scd: {} blueprints parsed ({} emitter, {} beam, {} trail, {} other), {} failures",
            rep.total_files,
            rep.emitters,
            rep.beams,
            rep.trails,
            rep.other,
            rep.failures.len()
        );
        for (name, err) in rep.failures.iter().take(10) {
            println!("  FAIL {name}: {err}");
        }
        if !rep.failures.is_empty() {
            return Err(format!("{} effect blueprint(s) failed to parse", rep.failures.len()));
        }
    }

    if do_coverage || both {
        let (cov, _) = compute_weapon_coverage(&game_dir)?;
        println!(
            "weapon effects coverage: {}/{} ({:.1}%) weapons mapped to FA emitters (combat: {}/{} = {:.1}%)",
            cov.mapped_weapons,
            cov.total_weapons,
            cov.pct(),
            cov.combat_mapped,
            cov.combat_total,
            cov.combat_pct(),
        );
        println!(
            "  sources: exact_lua={} name_match={} faction_category_fallback={} unmapped={} (templates={}, valid_bps={})",
            cov.exact_lua,
            cov.name_match,
            cov.faction_fallback,
            cov.unmapped,
            cov.templates_loaded,
            cov.valid_effect_bps,
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Game-data-independent unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_EMITTER: &str = r#"
EmitterBlueprint {
    BlueprintId = 'test_muzzle_01',
    Lifetime = 1.00,
    Repeattime = 1.00,
    TextureFramecount = 4.00,
    TextureStripcount = 2.00,
    Blendmode = 3.00,
    LocalVelocity = true,
    Gravity = false,
    Flat = false,
    LODCutoff = 180.00,
    Texture = [[/textures/particles/glow_02.dds]],
    RampTexture = [[/textures/particles/ramp_fire_02.dds]],
    EmitRateCurve = {
        XRange = 1.00,
        Keys = {
            { x=0.000, y=2.000, z=0.000 },
            { x=1.000, y=6.000, z=2.000 },
        },
    },
    LifetimeCurve = {
        XRange = 1.00,
        Keys = {
            { x=0.500, y=4.000, z=1.000 },
        },
    },
    StartSizeCurve = {
        XRange = 1.00,
        Keys = {
            { x=0.500, y=1.500, z=0.500 },
        },
    },
    EndSizeCurve = {
        XRange = 1.00,
        Keys = {
            { x=0.500, y=0.200, z=0.000 },
        },
    },
    VelocityCurve = {
        XRange = 1.00,
        Keys = {
            { x=0.500, y=1.200, z=0.400 },
        },
    },
}
"#;

    const SAMPLE_BEAM: &str = r#"
BeamBlueprint {
    Lifetime = 2,
    TextureName = '/textures/particles/build_beam_01.dds',
    Thickness = 0.05,
    StartColor = {x=0.2, y=0.5, z=0.8, w=0.4}, # R,G,B,A
    EndColor = {x=0.1, y=0.3, z=0.9, w=0.2},
    Blendmode = 3,
    RepeatRate = 0.4,
    Length = 8,
    UShift = 0.0,
    VShift = -0.15,
}
"#;

    const SAMPLE_TRAIL: &str = r#"
TrailEmitterBlueprint {
    BlueprintId = 'aeon_cannon_trail',
    Lifetime = -1.00,
    TrailLength = 10,
    Size = 0.12,
    SortOrder = 0,
    BlendMode = 0,
    TextureRepeatRate = 1,
    LODCutoff = 160,
    RepeatTexture = [[/textures/particles/trail_white_01.dds]],
    RampTexture = [[/textures/particles/ramp_trail_01.dds]],
}
"#;

    #[test]
    fn parses_emitter_beam_and_trail_blueprints() {
        let EffectBp::Emitter(e) = parse_effect_blueprint(SAMPLE_EMITTER).unwrap() else {
            panic!("expected EmitterBlueprint");
        };
        assert_eq!(e.id, "test_muzzle_01");
        assert_eq!(e.blend_mode, BlendMode::Additive);
        assert_eq!(e.texture, "textures/particles/glow_02.dds");
        assert_eq!(e.ramp_texture, "textures/particles/ramp_fire_02.dds");
        assert_eq!(e.texture_frame_count, 4);
        assert_eq!(e.texture_strip_count, 2);
        assert_eq!(e.emit_rate.keys.len(), 2);
        assert!((e.start_size.mid_y() - 1.5).abs() < 1e-5);

        let EffectBp::Beam(b) = parse_effect_blueprint(SAMPLE_BEAM).unwrap() else {
            panic!("expected BeamBlueprint");
        };
        assert_eq!(b.texture, "textures/particles/build_beam_01.dds");
        assert!((b.thickness - 0.05).abs() < 1e-5);
        assert!((b.start_color[2] - 0.8).abs() < 1e-5);
        assert_eq!(b.blend_mode, BlendMode::Additive);

        let EffectBp::Trail(t) = parse_effect_blueprint(SAMPLE_TRAIL).unwrap() else {
            panic!("expected TrailEmitterBlueprint");
        };
        assert_eq!(t.id, "aeon_cannon_trail");
        assert_eq!(t.blend_mode, BlendMode::Alpha);
        assert_eq!(t.repeat_texture, "textures/particles/trail_white_01.dds");
        assert!((t.trail_length - 10.0).abs() < 1e-5);
    }

    #[test]
    fn curve_evaluation_lerp_clamp_and_variance() {
        let c = Curve {
            x_range: 10.0,
            keys: vec![
                CurveKey { x: 2.0, y: 10.0, z: 4.0 },
                CurveKey { x: 6.0, y: 30.0, z: 8.0 },
                CurveKey { x: 6.0, y: 50.0, z: 0.0 }, // duplicate x must not divide by zero
            ],
        };
        assert_eq!(c.eval(0.0), (10.0, 4.0));
        let (y, z) = c.eval(4.0);
        assert!((y - 20.0).abs() < 1e-5 && (z - 6.0).abs() < 1e-5);
        assert_eq!(c.eval(99.0), (50.0, 0.0));
        assert!((c.sample(4.0, 1.0) - 23.0).abs() < 1e-5);
        assert!((c.sample(4.0, -1.0) - 17.0).abs() < 1e-5);
        assert_eq!(c.eval(f32::NAN), (10.0, 4.0));
        assert_eq!(Curve::default().eval(1.0), (0.0, 0.0));
    }

    #[test]
    fn parses_effect_templates_lua_tables_aliases_and_tablecat() {
        let lua = r#"
# Header comment
TableCat = import('utilities.lua').TableCat
EmtBpPath = '/effects/emitters/'
EmitterTempEmtBpPath = '/effects/emitters/temp/'

FlashSml01 = { EmtBpPath .. 'flash_01_emit.bp', }
SmokeSml01 = { EmtBpPath .. 'smoke_01_emit.bp', '/effects/emitters/smoke_02_emit.bp' }
SingleRef = EmtBpPath .. 'single_emit.bp' # trailing comment
AliasRef = FlashSml01
Combined = TableCat(FlashSml01, SmokeSml01, { EmitterTempEmtBpPath .. 'extra_emit.bp' })
"#;
        let t = parse_effect_templates(lua);
        assert_eq!(t.get("FlashSml01").unwrap(), &vec!["effects/emitters/flash_01_emit.bp"]);
        assert_eq!(
            t.get("SmokeSml01").unwrap(),
            &vec!["effects/emitters/smoke_01_emit.bp", "effects/emitters/smoke_02_emit.bp"]
        );
        assert_eq!(t.get("SingleRef").unwrap(), &vec!["effects/emitters/single_emit.bp"]);
        assert_eq!(t.get("AliasRef").unwrap(), &vec!["effects/emitters/flash_01_emit.bp"]);
        assert_eq!(
            t.get("Combined").unwrap(),
            &vec![
                "effects/emitters/flash_01_emit.bp",
                "effects/emitters/smoke_01_emit.bp",
                "effects/emitters/smoke_02_emit.bp",
                "effects/emitters/temp/extra_emit.bp",
            ]
        );
    }

    #[test]
    fn maps_weapon_via_lua_inheritance_and_fallback() {
        let templates = parse_effect_templates(
            r#"
EmtBpPath = '/effects/emitters/'
TPlasmaCannonLightMuzzleFlash = { EmtBpPath .. 't_muzzle_emit.bp' }
TGaussCannonHitUnit01 = { EmtBpPath .. 't_hit_emit.bp' }
CLaserMuzzleFlash01 = { EmtBpPath .. 'c_muzzle_emit.bp' }
CLaserHit01 = { EmtBpPath .. 'c_hit_emit.bp' }
"#,
        );
        let valid_bps: BTreeSet<String> = [
            "effects/emitters/t_muzzle_emit.bp",
            "effects/emitters/t_hit_emit.bp",
            "effects/emitters/c_muzzle_emit.bp",
            "effects/emitters/c_hit_emit.bp",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();

        let mut weapon_classes = HashMap::new();
        parse_lua_class_fx(
            r#"
BaseGauss = Class(DefaultProjectileWeapon) {
    FxMuzzleFlash = EffectTemplate.TPlasmaCannonLightMuzzleFlash,
}
TankGauss = Class(BaseGauss) {
}
"#,
            &mut weapon_classes,
        );
        let mut proj_classes = HashMap::new();
        parse_lua_class_fx(
            r#"
TDFGaussProj = Class(EmitterProjectile) {
    FxImpactUnit = EffectTemplate.TGaussCannonHitUnit01,
}
"#,
            &mut proj_classes,
        );
        let mut proj_scripts = HashMap::new();
        proj_scripts.insert(
            "projectiles/tdfgauss01/tdfgauss01_proj.bp".to_string(),
            parse_projectile_script("TDFGauss01 = Class(TDFGaussProj) {}"),
        );
        let mut unit_weapons = HashMap::new();
        for (lbl, cls) in parse_unit_script_weapons("Weapons = { MainGun = Class(TankGauss) {}, }") {
            unit_weapons.insert(("uel0201".to_string(), lbl), cls);
        }

        let exact_spec = WeaponSpec {
            unit_id: "UEL0201".into(),
            faction: Faction::Uef,
            label: "MainGun".into(),
            category: "Direct Fire".into(),
            projectile_id: Some("projectiles/tdfgauss01/tdfgauss01_proj.bp".into()),
            dummy: false,
            damage: 24.0,
        };
        let mapped = map_weapon(
            &exact_spec,
            &unit_weapons,
            &weapon_classes,
            &HashMap::new(),
            &proj_scripts,
            &proj_classes,
            &templates,
            &valid_bps,
        );
        assert_eq!(mapped.source, MappingSource::ExactLua);
        assert_eq!(mapped.muzzle, vec!["effects/emitters/t_muzzle_emit.bp"]);
        assert_eq!(mapped.impact, vec!["effects/emitters/t_hit_emit.bp"]);

        // Cybran weapon without script entry falls back to faction + category FA templates.
        let fb_spec = WeaponSpec {
            unit_id: "URL0107".into(),
            faction: Faction::Cybran,
            label: "LaserArms".into(),
            category: "Direct Fire".into(),
            projectile_id: None,
            dummy: false,
            damage: 12.0,
        };
        let mapped_fb = map_weapon(
            &fb_spec,
            &unit_weapons,
            &weapon_classes,
            &HashMap::new(),
            &proj_scripts,
            &proj_classes,
            &templates,
            &valid_bps,
        );
        assert_eq!(mapped_fb.source, MappingSource::FactionCategoryFallback);
        assert_eq!(mapped_fb.muzzle, vec!["effects/emitters/c_muzzle_emit.bp"]);
        assert_eq!(mapped_fb.impact, vec!["effects/emitters/c_hit_emit.bp"]);
    }

    #[test]
    fn samples_color_ramp_from_synthetic_rgba() {
        let mut pixels = vec![0u8; 16 * 2 * 4];
        for x in 0..16 {
            let v = (x * 16).min(255) as u8;
            let i = x * 4;
            pixels[i] = 255;
            pixels[i + 1] = v;
            pixels[i + 2] = 0;
            pixels[i + 3] = 255 - v;
        }
        let img = RgbaImage { width: 16, height: 2, pixels };
        let ramp = ColorRamp::from_rgba_image(&img, 0.0);
        let c0 = ramp.sample(0.0);
        let c_mid = ramp.sample(0.5);
        let c1 = ramp.sample(1.0);
        assert!(c0[0] > 0.9 && c0[3] > c_mid[3]);
        assert!(c1[3] <= 0.05);
    }
}
