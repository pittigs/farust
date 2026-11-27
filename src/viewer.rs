//! GPU (`wgpu`) scene renderer supporting both interactive windows (`winit`) and
//! headless offscreen PNG snapshots.
//!
//! Renders real Forged Alliance assets:
//! - 6-face skybox cubemap (`SkyCube_*.dds`) + atmospheric fog
//! - Per-pixel multi-stratum terrain splatting (10 albedo + 9 tangent-normal strata + tm0/tm1)
//! - Soft 3x3 PCF directional shadow map (Depth32Float, size from `ShadowQuality`)
//! - Hardware-instanced `.scm` props (alpha-cutout pines, rocks, logs) and `_LOD0` units/wrecks
//! - Depth-faded transparent water surface with animated dual-layer wave normals + sky reflections
//! - Layered 2D HUD (minimap, unit icon atlas, bitmap font labels)

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::camera::OrbitCamera;
use crate::dds_decode::{RgbaImage, SpecGlowMap};
use crate::scmap::ScMapLighting;
use crate::scene::{GpuVertex, LoadedScene, TerrainSplatData, WaterGpuData};
use crate::settings::{AaMode, ShadowQuality};
use crate::vfx::VfxVertex;

pub(crate) const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub(crate) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub(crate) const BLOOM_MIPS: usize = 5;

/// Returns whether the adapter supports 4x MSAA + resolve on the `Rgba16Float` HDR target.
pub(crate) fn hdr_msaa4_supported(adapter: &wgpu::Adapter) -> bool {
    let flags = adapter.get_texture_format_features(HDR_FORMAT).flags;
    flags.contains(
        wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4
            | wgpu::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE,
    )
}

/// Resolves the requested [`AaMode`] against device capabilities (`Msaa4` falls back to `Fxaa`
/// when `Rgba16Float` 4x MSAA is unsupported; `Fxaa` and `Off` always work).
pub fn resolve_aa_mode(requested: AaMode, msaa4_supported: bool) -> AaMode {
    match requested {
        AaMode::Msaa4 if !msaa4_supported => AaMode::Fxaa,
        other => other,
    }
}

/// Computes downsampled bloom mip dimensions starting at half-resolution (`width/2 x height/2`),
/// halving until `max_mips` levels or `1x1` is reached.
pub fn bloom_mip_sizes(width: u32, height: u32, max_mips: usize) -> Vec<(u32, u32)> {
    if max_mips == 0 {
        return Vec::new();
    }
    let mut w = (width.max(1) / 2).max(1);
    let mut h = (height.max(1) / 2).max(1);
    let mut out = Vec::with_capacity(max_mips);
    for _ in 0..max_mips {
        out.push((w, h));
        if w == 1 && h == 1 {
            break;
        }
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    out
}

/// Maps `.scmap` `LightingMultiplier` (typically `1.0..1.8`, default `1.4`) to linear HDR exposure
/// calibrated for the Narkowicz ACES filmic curve.
pub fn exposure_from_lighting(lighting_multiplier: f32) -> f32 {
    let m = if lighting_multiplier.is_finite() && lighting_multiplier > 0.1 {
        lighting_multiplier
    } else {
        1.4
    };
    (0.95 * (m / 1.4).powf(0.45)).clamp(0.60, 1.45)
}

/// Maps `.scmap` `Bloom` (typically `0.0..0.12`, default `0.08`) to bloom composite weight.
pub fn bloom_strength_from_map(map_bloom: f32, enabled: bool) -> f32 {
    if !enabled || !map_bloom.is_finite() || map_bloom <= 0.0 {
        return 0.0;
    }
    (map_bloom * 3.5).clamp(0.0, 0.70)
}

/// Narkowicz ACES filmic tonemap operator (matches `aces_tonemap` in `TONEMAP_SHADER_WGSL`).
pub fn aces_tonemap_scalar(x: f32) -> f32 {
    let v = if x.is_finite() { x.max(0.0) } else { 0.0 };
    ((v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14)).clamp(0.0, 1.0)
}

/// CPU mirror of the tonemap + relative gamma curve (`gamma` in `0.8..1.4`, `1.0` = neutral).
#[allow(dead_code)]
pub fn tonemap_rgb(hdr: [f32; 3], exposure: f32, gamma: f32) -> [f32; 3] {
    let exp = if exposure.is_finite() { exposure.max(0.0) } else { 1.0 };
    let inv_gamma = 1.0 / gamma.clamp(0.8, 1.4);
    [
        aces_tonemap_scalar(hdr[0] * exp).powf(inv_gamma),
        aces_tonemap_scalar(hdr[1] * exp).powf(inv_gamma),
        aces_tonemap_scalar(hdr[2] * exp).powf(inv_gamma),
    ]
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PostUniforms {
    /// `[exposure, bloom_strength, inv_gamma, direct_encode_srgb]`
    params0: [f32; 4],
    /// `[bloom_threshold, bloom_knee, fxaa_decode_srgb, 0.0]`
    params1: [f32; 4],
}

fn make_post_uniforms(
    lighting: &ScMapLighting,
    bloom_enabled: bool,
    gamma: f32,
    target_is_srgb: bool,
) -> PostUniforms {
    let exposure = exposure_from_lighting(lighting.lighting_multiplier);
    let bloom_strength = bloom_strength_from_map(lighting.bloom, bloom_enabled);
    let inv_gamma = 1.0 / gamma.clamp(0.8, 1.4);
    PostUniforms {
        params0: [
            exposure,
            bloom_strength,
            inv_gamma,
            if target_is_srgb { 0.0 } else { 1.0 },
        ],
        params1: [0.95, 0.35, if target_is_srgb { 1.0 } else { 0.0 }, 0.0],
    }
}

/// `[texel, enabled, 0, 0]` for [`Uniforms::shadow_params`]: PCF tap size in UV
/// space plus a kill-switch so `Off` renders fully lit without a shadow pass.
fn shadow_uniform_params(quality: ShadowQuality) -> [f32; 4] {
    let edge = quality.map_size().max(1) as f32;
    let enabled = if quality.map_size() > 0 { 1.0 } else { 0.0 };
    [1.0 / edge, enabled, 0.0, 0.0]
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    shadow_view_proj: [[f32; 4]; 4],
    eye_pos: [f32; 4],       // xyz + w=time_sec
    sun_dir: [f32; 4],       // xyz + w=lighting_multiplier
    sun_color: [f32; 4],     // rgb + w=water_elevation
    ambient_color: [f32; 4], // rgb + w=map_width
    fog_color: [f32; 4],     // rgb + w=map_height
    fog_params: [f32; 4],    // fog_start, fog_end, water_deep, water_abyss
    shadow_params: [f32; 4], // PCF texel, shadows-enabled flag, unused, unused
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TerrainParamsUniform {
    alb_scales_0: [f32; 4],
    alb_scales_1: [f32; 4],
    alb_scales_2: [f32; 4], // 8, 9, unused, unused
    norm_scales_0: [f32; 4],
    norm_scales_1: [f32; 4],
    norm_scales_2: [f32; 4], // 8, unused, unused, unused
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct WaterParamsUniform {
    surface_color: [f32; 4],
    fresnel: [f32; 4], // bias, power, lerp0, lerp1
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuUnitInstance {
    pub row0: [f32; 4],       // [m00, m01, m02, tx]
    pub row1: [f32; 4],       // [m10, m11, m12, ty]
    pub row2: [f32; 4],       // [m20, m21, m22, tz]
    pub team_color: [f32; 4], // rgb = team/tint color, a = spec strength
    pub flags: [f32; 4],      // x = use_tex, y = alpha_cutout, z = wreck_char, w = team_mask_strength
    pub extra: [f32; 4],      // x = bone_base+1 (0=static), y = lod_fade, z = vet_glow, w = build_progress
}

impl GpuUnitInstance {
    pub fn from_pos_yaw_scale(
        pos: [f32; 3],
        yaw: f32,
        pitch: f32,
        scale: f32,
        team_color: [f32; 4],
        flags: [f32; 4],
    ) -> Self {
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        Self {
            row0: [scale * cy, scale * sy * sp, scale * sy * cp, pos[0]],
            row1: [0.0, scale * cp, -scale * sp, pos[1]],
            row2: [-scale * sy, scale * cy * sp, scale * cy * cp, pos[2]],
            team_color,
            flags,
            extra: [0.0; 4],
        }
    }

    pub fn from_basis(
        pos: [f32; 3],
        rx: [f32; 3],
        ry: [f32; 3],
        rz: [f32; 3],
        scale: [f32; 3],
        team_color: [f32; 4],
        flags: [f32; 4],
    ) -> Self {
        Self {
            row0: [rx[0] * scale[0], ry[0] * scale[1], rz[0] * scale[2], pos[0]],
            row1: [rx[1] * scale[0], ry[1] * scale[1], rz[1] * scale[2], pos[1]],
            row2: [rx[2] * scale[0], ry[2] * scale[1], rz[2] * scale[2], pos[2]],
            team_color,
            flags,
            extra: [0.0; 4],
        }
    }

    #[allow(dead_code)]
    pub fn with_extra(mut self, extra: [f32; 4]) -> Self {
        self.extra = extra;
        self
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct HudVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

pub(crate) struct GpuMeshTemplate {
    pub vertex_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    pub index_count: u32,
    pub bind_group: wgpu::BindGroup,
}

const SHADER_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    shadow_view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var t_albedo: texture_2d<f32>;
@group(0) @binding(2) var s_albedo: sampler;
@group(0) @binding(3) var t_spec_glow: texture_2d<f32>;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.normal = in.normal;
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    let tex = textureSample(t_albedo, s_albedo, in.uv);
    let sg = textureSample(t_spec_glow, s_albedo, in.uv);
    let base_rgb = tex.rgb * in.color.rgb;
    let spec_strength = tex.a * in.color.a;

    var n = normalize(in.normal);
    if (!front_facing) {
        n = -n;
    }

    let l = normalize(u.sun_dir.xyz);
    let v = normalize(u.eye_pos.xyz - in.world_pos);
    let h = normalize(l + v);

    let hemi_t = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    let ambient = mix(u.ambient_color.rgb * 0.75, u.ambient_color.rgb * 1.45 + vec3<f32>(0.10, 0.13, 0.18), hemi_t);

    let ndotl = max(dot(n, l), 0.0);
    let sun = u.sun_color.rgb * (u.sun_dir.w * 0.65) * ndotl;

    let spec = pow(max(dot(n, h), 0.0), 36.0) * spec_strength * (0.20 + 0.80 * ndotl);
    let emissive = base_rgb * (sg.g * 3.2);
    var lit = base_rgb * (ambient + sun) + vec3<f32>(spec) + emissive;

    let dist = length(u.eye_pos.xyz - in.world_pos);
    let fog_start = max(u.fog_params.x, max(u.eye_pos.y, 40.0) * 1.35 + 280.0);
    let fog_end = max(u.fog_params.y + 800.0, fog_start + 1800.0);
    let fog_t = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 0.35);
    lit = mix(lit, u.fog_color.rgb, fog_t);
    return vec4<f32>(lit, 1.0);
}
"#;

const SHADOW_SHADER_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    shadow_view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var<storage, read> bone_mats: array<mat4x4<f32>>;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) row0: vec4<f32>,
    @location(5) row1: vec4<f32>,
    @location(6) row2: vec4<f32>,
    @location(7) team_color: vec4<f32>,
    @location(8) flags: vec4<f32>,
    @location(9) extra: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> @builtin(position) vec4<f32> {
    var lpos = in.position;
    if (in.extra.x >= 0.5) {
        let base_i = u32(in.extra.x - 0.75);
        let bone_off = u32(max(in.color.w, 0.0) + 0.5);
        let bm = bone_mats[base_i + bone_off];
        lpos = (bm * vec4<f32>(in.position, 1.0)).xyz;
    }
    if (in.extra.w > 0.01 && in.extra.w < 0.99) {
        lpos.y = min(lpos.y, in.extra.w * max(in.extra.y, 0.25));
    }
    let p = vec4<f32>(lpos, 1.0);
    let world_p = vec3<f32>(dot(in.row0, p), dot(in.row1, p), dot(in.row2, p));
    return u.shadow_view_proj * vec4<f32>(world_p, 1.0);
}
"#;

const SKY_SHADER_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    shadow_view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var t_sky: texture_cube<f32>;
@group(1) @binding(1) var s_sky: sampler;

struct VsIn {
    @location(0) dir: vec3<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) dir: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let world_p = u.eye_pos.xyz + in.dir * 400.0;
    let clip = u.view_proj * vec4<f32>(world_p, 1.0);
    out.clip_pos = vec4<f32>(clip.x, clip.y, clip.w * 0.99995, clip.w);
    out.dir = in.dir;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = normalize(in.dir);
    let sky = textureSample(t_sky, s_sky, d).rgb;
    let horizon = smoothstep(-0.04, 0.20, d.y);
    let col = mix(u.fog_color.rgb, sky, horizon);
    return vec4<f32>(col, 1.0);
}
"#;

const TERRAIN_SHADER_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    shadow_view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
    shadow_params: vec4<f32>,
};

struct TerrainParams {
    alb_scales_0: vec4<f32>,
    alb_scales_1: vec4<f32>,
    alb_scales_2: vec4<f32>,
    norm_scales_0: vec4<f32>,
    norm_scales_1: vec4<f32>,
    norm_scales_2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<uniform> tp: TerrainParams;
@group(0) @binding(2) var t_tm0: texture_2d<f32>;
@group(0) @binding(3) var t_tm1: texture_2d<f32>;
@group(0) @binding(4) var t_alb: texture_2d_array<f32>;
@group(0) @binding(5) var t_norm: texture_2d_array<f32>;
@group(0) @binding(6) var s_clamp: sampler;
@group(0) @binding(7) var s_repeat: sampler;
@group(0) @binding(8) var t_fow: texture_2d<f32>;

@group(1) @binding(0) var t_shadow: texture_depth_2d;
@group(1) @binding(1) var s_shadow: sampler_comparison;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.normal = in.normal;
    out.uv = in.uv;
    return out;
}

fn sample_shadow_pcf(world_pos: vec3<f32>) -> f32 {
    let sc = u.shadow_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = sc.xyz / sc.w;
    let suv = vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
    let depth_ref = clamp(ndc.z - 0.0018, 0.0, 1.0);
    let step = u.shadow_params.x;

    var sum = 0.0;
    if (u.shadow_params.y < 0.5) {
        return 1.0;
    }
    sum += textureSampleCompare(t_shadow, s_shadow, suv + vec2<f32>(-step, -step), depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, suv + vec2<f32>(0.0, -step), depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, suv + vec2<f32>(step, -step), depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, suv + vec2<f32>(-step, 0.0), depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, suv, depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, suv + vec2<f32>(step, 0.0), depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, suv + vec2<f32>(-step, step), depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, suv + vec2<f32>(0.0, step), depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, suv + vec2<f32>(step, step), depth_ref);
    let pcf = sum / 9.0;

    let inside = suv.x >= 0.01 && suv.x <= 0.99 && suv.y >= 0.01 && suv.y <= 0.99 && ndc.z > 0.0 && ndc.z < 1.0;
    return select(1.0, pcf, inside);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let b0 = textureSample(t_tm0, s_clamp, in.uv);
    let b1 = textureSample(t_tm1, s_clamp, in.uv);

    let s0 = max(tp.alb_scales_0.x, 1.0) * 2.4;
    var rgb = textureSample(t_alb, s_repeat, in.world_pos.xz / s0, 0).rgb;
    let ns0 = max(tp.norm_scales_0.x, 1.0) * 2.4;
    var tnorm = textureSample(t_norm, s_repeat, in.world_pos.xz / ns0, 0).xyz * 2.0 - vec3<f32>(1.0);

    // Layer 1 (tm0.r)
    rgb = mix(rgb, textureSample(t_alb, s_repeat, in.world_pos.xz / (max(tp.alb_scales_0.y, 1.0) * 2.4), 1).rgb, b0.x * select(0.0, 1.0, tp.alb_scales_0.y > 0.0));
    tnorm = mix(tnorm, textureSample(t_norm, s_repeat, in.world_pos.xz / (max(tp.norm_scales_0.y, 1.0) * 2.4), 1).xyz * 2.0 - vec3<f32>(1.0), b0.x * select(0.0, 1.0, tp.norm_scales_0.y > 0.0));

    // Layer 2 (tm0.g)
    rgb = mix(rgb, textureSample(t_alb, s_repeat, in.world_pos.xz / (max(tp.alb_scales_0.z, 1.0) * 2.4), 2).rgb, b0.y * select(0.0, 1.0, tp.alb_scales_0.z > 0.0));
    tnorm = mix(tnorm, textureSample(t_norm, s_repeat, in.world_pos.xz / (max(tp.norm_scales_0.z, 1.0) * 2.4), 2).xyz * 2.0 - vec3<f32>(1.0), b0.y * select(0.0, 1.0, tp.norm_scales_0.z > 0.0));

    // Layer 3 (tm0.b)
    rgb = mix(rgb, textureSample(t_alb, s_repeat, in.world_pos.xz / (max(tp.alb_scales_0.w, 1.0) * 2.4), 3).rgb, b0.z * select(0.0, 1.0, tp.alb_scales_0.w > 0.0));
    tnorm = mix(tnorm, textureSample(t_norm, s_repeat, in.world_pos.xz / (max(tp.norm_scales_0.w, 1.0) * 2.4), 3).xyz * 2.0 - vec3<f32>(1.0), b0.z * select(0.0, 1.0, tp.norm_scales_0.w > 0.0));

    // Layer 4 (tm0.a)
    rgb = mix(rgb, textureSample(t_alb, s_repeat, in.world_pos.xz / (max(tp.alb_scales_1.x, 1.0) * 2.4), 4).rgb, b0.w * select(0.0, 1.0, tp.alb_scales_1.x > 0.0));
    tnorm = mix(tnorm, textureSample(t_norm, s_repeat, in.world_pos.xz / (max(tp.norm_scales_1.x, 1.0) * 2.4), 4).xyz * 2.0 - vec3<f32>(1.0), b0.w * select(0.0, 1.0, tp.norm_scales_1.x > 0.0));

    // Layer 5 (tm1.r)
    rgb = mix(rgb, textureSample(t_alb, s_repeat, in.world_pos.xz / (max(tp.alb_scales_1.y, 1.0) * 2.4), 5).rgb, b1.x * select(0.0, 1.0, tp.alb_scales_1.y > 0.0));
    tnorm = mix(tnorm, textureSample(t_norm, s_repeat, in.world_pos.xz / (max(tp.norm_scales_1.y, 1.0) * 2.4), 5).xyz * 2.0 - vec3<f32>(1.0), b1.x * select(0.0, 1.0, tp.norm_scales_1.y > 0.0));

    // Layer 6 (tm1.g)
    rgb = mix(rgb, textureSample(t_alb, s_repeat, in.world_pos.xz / (max(tp.alb_scales_1.z, 1.0) * 2.4), 6).rgb, b1.y * select(0.0, 1.0, tp.alb_scales_1.z > 0.0));
    tnorm = mix(tnorm, textureSample(t_norm, s_repeat, in.world_pos.xz / (max(tp.norm_scales_1.z, 1.0) * 2.4), 6).xyz * 2.0 - vec3<f32>(1.0), b1.y * select(0.0, 1.0, tp.norm_scales_1.z > 0.0));

    // Layer 7 (tm1.b)
    rgb = mix(rgb, textureSample(t_alb, s_repeat, in.world_pos.xz / (max(tp.alb_scales_1.w, 1.0) * 2.4), 7).rgb, b1.z * select(0.0, 1.0, tp.alb_scales_1.w > 0.0));
    tnorm = mix(tnorm, textureSample(t_norm, s_repeat, in.world_pos.xz / (max(tp.norm_scales_1.w, 1.0) * 2.4), 7).xyz * 2.0 - vec3<f32>(1.0), b1.z * select(0.0, 1.0, tp.norm_scales_1.w > 0.0));

    // Layer 8 (tm1.a)
    rgb = mix(rgb, textureSample(t_alb, s_repeat, in.world_pos.xz / (max(tp.alb_scales_2.x, 1.0) * 2.4), 8).rgb, b1.w * select(0.0, 1.0, tp.alb_scales_2.x > 0.0));
    tnorm = mix(tnorm, textureSample(t_norm, s_repeat, in.world_pos.xz / (max(tp.norm_scales_2.x, 1.0) * 2.4), 8).xyz * 2.0 - vec3<f32>(1.0), b1.w * select(0.0, 1.0, tp.norm_scales_2.x > 0.0));

    let macro_col = textureSample(t_alb, s_repeat, in.uv * 1.5, 9).rgb;
    let macro_mod = select(vec3<f32>(1.0), vec3<f32>(0.68) + 0.64 * macro_col, tp.alb_scales_2.y > 0.0);
    rgb = rgb * macro_mod;

    // Perturb geometric normal with blended tangent-space stratum normal.
    let g_n = normalize(in.normal);
    let t = normalize(vec3<f32>(1.0, 0.0, 0.0) - g_n * g_n.x);
    let b = normalize(cross(g_n, t));
    let n = normalize(g_n * max(tnorm.z, 0.25) + t * (tnorm.x * 0.55) + b * (tnorm.y * 0.55));

    let l = normalize(u.sun_dir.xyz);
    let ndotl = max(dot(n, l), 0.0);
    let shadow = sample_shadow_pcf(in.world_pos);
    let hemi = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    let ambient = mix(u.ambient_color.rgb * 0.55 + vec3<f32>(0.12, 0.13, 0.16), u.ambient_color.rgb + vec3<f32>(0.24, 0.27, 0.34), hemi);
    let sun = u.sun_color.rgb * (u.sun_dir.w * 0.65) * ndotl * shadow;

    var lit = rgb * (ambient + sun);

    // Subtle darkened seabed attenuation under water elevation
    let depth_under = max(u.sun_color.w - in.world_pos.y, 0.0);
    let wet = smoothstep(0.0, 0.9, depth_under);
    lit = mix(lit, lit * vec3<f32>(0.48, 0.58, 0.65), wet);

    let fow = textureSample(t_fow, s_clamp, in.uv).r;
    lit = lit * mix(0.32, 1.0, fow);

    let dist = length(u.eye_pos.xyz - in.world_pos);
    let fog_start = max(u.fog_params.x, max(u.eye_pos.y, 40.0) * 1.35 + 280.0);
    let fog_end = max(u.fog_params.y + 800.0, fog_start + 1800.0);
    let fog_t = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 0.35);
    lit = mix(lit, u.fog_color.rgb, fog_t);
    return vec4<f32>(lit, 1.0);
}
"#;

const WATER_SHADER_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    shadow_view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};

struct WaterParams {
    surface_color: vec4<f32>,
    fresnel: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<uniform> wp: WaterParams;
@group(0) @binding(2) var t_waves: texture_2d<f32>;
@group(0) @binding(3) var t_ramp: texture_2d<f32>;
@group(0) @binding(4) var s_repeat: sampler;
@group(0) @binding(5) var s_clamp: sampler;

@group(1) @binding(0) var t_sky: texture_cube<f32>;
@group(1) @binding(1) var s_sky: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>, // color.x = seabed depth in meters
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) depth_m: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.depth_m = in.color.x;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let time = u.eye_pos.w;
    let n1 = textureSample(t_waves, s_repeat, in.world_pos.xz * 0.018 + vec2<f32>(time * 0.012, time * 0.007)).xyz * 2.0 - vec3<f32>(1.0);
    let n2 = textureSample(t_waves, s_repeat, in.world_pos.xz * 0.032 - vec2<f32>(time * 0.009, -time * 0.011)).xyz * 2.0 - vec3<f32>(1.0);
    let n_wave = normalize(vec3<f32>((n1.x + n2.x) * 0.075, 1.0, (n1.y + n2.y) * 0.075));

    let v = normalize(u.eye_pos.xyz - in.world_pos);
    let refl = reflect(-v, n_wave);
    let sky_refl = textureSample(t_sky, s_sky, refl).rgb;

    let max_depth = max(u.sun_color.w - u.fog_params.w, 6.0);
    let ramp_u = clamp(in.depth_m / max_depth, 0.0, 1.0);
    let ramp_col = textureSample(t_ramp, s_clamp, vec2<f32>(ramp_u, 0.5)).rgb;
    let body_col = mix(ramp_col * 0.72, wp.surface_color.rgb + vec3<f32>(0.01, 0.06, 0.12), 0.45);

    let ndotv = max(dot(n_wave, v), 0.0);
    let fresnel = clamp(wp.fresnel.x * 0.7 + pow(1.0 - ndotv, max(wp.fresnel.y, 1.8)) * 0.65, 0.06, 0.78);
    var col = mix(body_col, sky_refl, fresnel);

    let l = normalize(u.sun_dir.xyz);
    let h = normalize(l + v);
    let spec = pow(max(dot(n_wave, h), 0.0), 180.0) * 0.85;
    col += u.sun_color.rgb * spec;

    // Subtle animated shoreline foam where water is very shallow
    let shore_mask = (1.0 - smoothstep(0.05, 0.90, in.depth_m)) * clamp((n1.x + n2.y) * 0.30 + 0.60, 0.0, 1.0);
    col = mix(col, vec3<f32>(0.72, 0.80, 0.88), shore_mask * 0.25);

    let dist = length(u.eye_pos.xyz - in.world_pos);
    let fog_start = max(u.fog_params.x, max(u.eye_pos.y, 40.0) * 1.35 + 280.0);
    let fog_end = max(u.fog_params.y + 800.0, fog_start + 1800.0);
    let fog_t = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 0.35);
    col = mix(col, u.fog_color.rgb, fog_t);

    let alpha = clamp(smoothstep(0.0, 1.35, in.depth_m) * 0.84 + fresnel * 0.14, 0.0, 0.92);
    return vec4<f32>(col, alpha);
}
"#;

const INSTANCED_SHADER_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    shadow_view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
    shadow_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var t_albedo: texture_2d<f32>;
@group(0) @binding(2) var s_albedo: sampler;
@group(0) @binding(3) var t_spec_glow: texture_2d<f32>;

@group(1) @binding(0) var t_shadow: texture_depth_2d;
@group(1) @binding(1) var s_shadow: sampler_comparison;
@group(2) @binding(0) var<storage, read> bone_mats: array<mat4x4<f32>>;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) row0: vec4<f32>,
    @location(5) row1: vec4<f32>,
    @location(6) row2: vec4<f32>,
    @location(7) team_color: vec4<f32>,
    @location(8) flags: vec4<f32>,
    @location(9) extra: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) team_color: vec4<f32>,
    @location(4) flags: vec4<f32>,
    @location(5) extra: vec4<f32>,
    @location(6) local_y: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var lpos = in.position;
    var lnrm = in.normal;
    if (in.extra.x >= 0.5) {
        let base_i = u32(in.extra.x - 0.75);
        let bone_off = u32(max(in.color.w, 0.0) + 0.5);
        let bm = bone_mats[base_i + bone_off];
        lpos = (bm * vec4<f32>(in.position, 1.0)).xyz;
        lnrm = (bm * vec4<f32>(in.normal, 0.0)).xyz;
    }
    let p = vec4<f32>(lpos, 1.0);
    let world_p = vec3<f32>(dot(in.row0, p), dot(in.row1, p), dot(in.row2, p));
    let world_n = normalize(vec3<f32>(
        dot(in.row0.xyz, lnrm),
        dot(in.row1.xyz, lnrm),
        dot(in.row2.xyz, lnrm)
    ));
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(world_p, 1.0);
    out.world_pos = world_p;
    out.normal = world_n;
    out.uv = in.uv;
    out.team_color = in.team_color;
    out.flags = in.flags;
    out.extra = in.extra;
    out.local_y = lpos.y;
    return out;
}

fn sample_shadow_pcf(world_pos: vec3<f32>) -> f32 {
    let sc = u.shadow_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = sc.xyz / sc.w;
    let suv = vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
    let depth_ref = clamp(ndc.z - 0.0018, 0.0, 1.0);
    let step = u.shadow_params.x;

    var sum = 0.0;
    if (u.shadow_params.y < 0.5) {
        return 1.0;
    }
    sum += textureSampleCompare(t_shadow, s_shadow, suv + vec2<f32>(-step, 0.0), depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, suv + vec2<f32>(step, 0.0), depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, suv, depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, suv + vec2<f32>(0.0, -step), depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, suv + vec2<f32>(0.0, step), depth_ref);
    let pcf = sum / 5.0;
    let inside = suv.x >= 0.01 && suv.x <= 0.99 && suv.y >= 0.01 && suv.y <= 0.99 && ndc.z > 0.0 && ndc.z < 1.0;
    return select(1.0, pcf, inside);
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    let tex = textureSample(t_albedo, s_albedo, in.uv);
    let sg = textureSample(t_spec_glow, s_albedo, in.uv);
    let shadow = sample_shadow_pcf(in.world_pos);
    if (in.flags.y > 0.5 && tex.a < 0.35) {
        discard;
    }
    let use_tex = clamp(in.flags.x, 0.0, 1.0);
    let tex_rgb = mix(tex.rgb, in.team_color.rgb, clamp(tex.a * in.flags.w, 0.0, 1.0));
    var base_rgb = mix(in.team_color.rgb, tex_rgb, use_tex);

    if (in.flags.z > 0.5) {
        let luma = dot(base_rgb, vec3<f32>(0.299, 0.587, 0.114));
        base_rgb = vec3<f32>(luma * 0.28, luma * 0.25, luma * 0.24);
    }

    var n = normalize(in.normal);
    if (!front_facing) {
        n = -n;
    }
    let l = normalize(u.sun_dir.xyz);
    let v = normalize(u.eye_pos.xyz - in.world_pos);
    let h = normalize(l + v);
    let hemi_t = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    let ambient = mix(u.ambient_color.rgb * 0.75 + vec3<f32>(0.18, 0.20, 0.24), u.ambient_color.rgb + vec3<f32>(0.36, 0.40, 0.48), hemi_t);
    let ndotl = max(dot(n, l), 0.0);
    let sun = u.sun_color.rgb * (u.sun_dir.w * 0.72) * ndotl * mix(0.40, 1.0, shadow);
    let spec_mod = mix(1.0, 0.35 + 1.15 * sg.r, use_tex);
    let spec = pow(max(dot(n, h), 0.0), 32.0) * in.team_color.a * spec_mod * (0.25 + 0.75 * ndotl) * shadow;
    let glow_mask = sg.g * use_tex * select(1.0, 0.0, in.flags.z > 0.5);
    let emissive = base_rgb * (glow_mask * 3.2);
    var lit = base_rgb * (ambient + sun) + vec3<f32>(spec) + emissive;

    if (in.extra.z > 0.01) {
        let rim = pow(clamp(1.0 - max(dot(n, v), 0.0), 0.0, 1.0), 2.2);
        lit = lit + vec3<f32>(1.0, 0.84, 0.28) * (rim * in.extra.z * 0.75);
    }
    if (in.extra.w > 0.01 && in.extra.w < 0.99) {
        // Construction build-up: geometry above `progress * height` (template units) is cut away,
        // a bright scan seam rides the cut and the built part keeps a cyan nano-lattice tint.
        let h = max(in.extra.y, 0.25);
        let cut = in.extra.w * h;
        if (in.local_y > cut) {
            discard;
        }
        let seam = 1.0 - smoothstep(0.0, 0.06 * h, cut - in.local_y);
        let lattice = 0.5 + 0.5 * sin(in.local_y * 40.0 / h);
        lit = mix(lit, vec3<f32>(0.25, 0.85, 1.0), 0.30 + 0.10 * lattice) + vec3<f32>(0.55, 0.95, 1.0) * seam * 1.4;
    }

    let dist = length(u.eye_pos.xyz - in.world_pos);
    let fog_start = max(u.fog_params.x, max(u.eye_pos.y, 40.0) * 1.35 + 280.0);
    let fog_end = max(u.fog_params.y + 800.0, fog_start + 1800.0);
    let fog_t = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 0.35);
    lit = mix(lit, u.fog_color.rgb, fog_t);
    return vec4<f32>(lit, 1.0);
}
"#;

const HUD_SHADER_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    shadow_view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> _u: Uniforms;
@group(0) @binding(1) var t_hud: texture_2d<f32>;
@group(0) @binding(2) var s_hud: sampler;

struct VsIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = vec4<f32>(in.pos, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(t_hud, s_hud, in.uv);
    return tex * in.color;
}
"#;

const DECAL_SHADER_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    shadow_view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var t_albedo: texture_2d<f32>;
@group(0) @binding(2) var s_albedo: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.normal = in.normal;
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(t_albedo, s_albedo, in.uv);
    if (tex.a < 0.04) {
        discard;
    }
    let n = normalize(in.normal);
    let l = normalize(u.sun_dir.xyz);
    let hemi = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    let ambient = mix(u.ambient_color.rgb * 0.55 + vec3<f32>(0.12, 0.13, 0.16), u.ambient_color.rgb + vec3<f32>(0.24, 0.27, 0.34), hemi);
    let ndotl = max(dot(n, l), 0.0);
    let sun = u.sun_color.rgb * (u.sun_dir.w * 0.55) * ndotl;
    var lit = tex.rgb * in.color.rgb * (ambient + sun);

    let dist = length(u.eye_pos.xyz - in.world_pos);
    let fog_start = max(u.fog_params.x, max(u.eye_pos.y, 40.0) * 1.35 + 280.0);
    let fog_end = max(u.fog_params.y + 800.0, fog_start + 1800.0);
    let fog_t = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 0.35);
    lit = mix(lit, u.fog_color.rgb, fog_t);
    return vec4<f32>(lit, tex.a * 0.90 * in.color.a);
}
"#;

const SHIELD_SHADER_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    shadow_view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) row0: vec4<f32>,
    @location(5) row1: vec4<f32>,
    @location(6) row2: vec4<f32>,
    @location(7) team_color: vec4<f32>,
    @location(8) flags: vec4<f32>,
    @location(9) extra: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) team_color: vec4<f32>,
    @location(3) local_pos: vec3<f32>,
    @location(4) flags: vec4<f32>,
    @location(5) extra: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let p = vec4<f32>(in.position, 1.0);
    let world_p = vec3<f32>(dot(in.row0, p), dot(in.row1, p), dot(in.row2, p));
    let world_n = normalize(vec3<f32>(
        dot(in.row0.xyz, in.normal),
        dot(in.row1.xyz, in.normal),
        dot(in.row2.xyz, in.normal)
    ));
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(world_p, 1.0);
    out.world_pos = world_p;
    out.normal = world_n;
    out.team_color = in.team_color;
    out.local_pos = in.position;
    out.flags = in.flags;
    out.extra = in.extra;
    return out;
}

// Instance: team_color = faction tint (a = overall opacity), flags = (hp ratio, hit intensity,
// personal, _), extra = (impact direction xyz in world axes, seconds since impact).
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let v = normalize(u.eye_pos.xyz - in.world_pos);
    let fresnel = pow(clamp(1.0 - abs(dot(n, v)), 0.0, 1.0), 2.4);
    let lp = normalize(in.local_pos);
    let t = u.eye_pos.w;
    let bands = 0.5 + 0.5 * sin(lp.y * 22.0 - t * 2.4) * sin(atan2(lp.z, lp.x) * 11.0 + t * 0.6);
    let hp = clamp(in.flags.x, 0.0, 1.0);
    var col = in.team_color.rgb * (0.55 + 1.15 * fresnel) * (0.84 + 0.16 * bands);
    var alpha = 0.05 + 0.48 * fresnel + 0.05 * bands * mix(1.6, 1.0, hp);
    let k = in.flags.y;
    if (k > 0.001) {
        let ang = acos(clamp(dot(lp, normalize(in.extra.xyz)), -1.0, 1.0));
        let front = in.extra.w * 2.6;
        let ring = exp(-pow((ang - front) * 8.0, 2.0)) * k;
        let spot = exp(-ang * ang * 16.0) * k;
        col = col + vec3<f32>(0.85, 0.95, 1.0) * (ring * 1.3 + spot * 1.6);
        alpha = alpha + ring * 0.5 + spot * 0.55 + k * 0.10;
    }
    alpha = alpha * in.team_color.a;
    return vec4<f32>(col, clamp(alpha, 0.0, 0.92));
}
"#;

const VFX_SHADER_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    shadow_view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var t_vfx: texture_2d_array<f32>;
@group(0) @binding(2) var s_vfx_clamp: sampler;
@group(0) @binding(3) var s_vfx_repeat: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) mode: f32,
    @location(4) tex_layer: u32,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) mode: f32,
    @location(3) @interpolate(flat) tex_layer: u32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.mode = in.mode;
    out.tex_layer = in.tex_layer;
    return out;
}

// Uses premultiplied alpha blending (ONE, ONE_MINUS_SRC_ALPHA):
// - Alpha-blended particles return vec4(rgb * a, a)
// - Additive particles return vec4(rgb * a, 0.0)
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if (in.mode >= 3.5) {
        let layer = i32(in.tex_layer);
        var tex: vec4<f32>;
        if (in.mode >= 5.5) {
            tex = textureSampleLevel(t_vfx, s_vfx_repeat, in.uv, layer, 0.0);
        } else {
            tex = textureSampleLevel(t_vfx, s_vfx_clamp, in.uv, layer, 0.0);
        }
        let is_add = (in.mode >= 4.5 && in.mode < 5.5) || (in.mode >= 6.5);
        let rgb = tex.rgb * in.color.rgb;
        if (is_add) {
            let a = max(tex.a, max(tex.r, max(tex.g, tex.b))) * in.color.a;
            let out_rgb = rgb * a * 1.85;
            if (max(out_rgb.r, max(out_rgb.g, out_rgb.b)) < 0.004) {
                discard;
            }
            return vec4<f32>(out_rgb, 0.0);
        } else {
            let a = clamp(tex.a * in.color.a, 0.0, 1.0);
            if (a < 0.01) {
                discard;
            }
            return vec4<f32>(rgb * a, a);
        }
    }

    let centered = in.uv * 2.0 - vec2<f32>(1.0, 1.0);
    let d = length(centered);
    var mask = 0.0;
    var is_add = false;
    if (in.mode >= 2.5) {
        mask = (1.0 - smoothstep(0.55, 1.0, d)) * (0.45 + 0.55 * smoothstep(0.15, 0.8, d));
    } else if (in.mode >= 1.5) {
        let ring = exp(-pow((d - 0.78) * 7.5, 2.0));
        mask = ring * (1.0 - smoothstep(0.90, 1.0, d));
        is_add = true;
    } else if (in.mode >= 0.5) {
        mask = exp(-d * d * 3.0) * (1.0 - smoothstep(0.65, 1.0, d));
    } else {
        mask = pow(clamp(1.0 - d, 0.0, 1.0), 1.5);
        is_add = true;
    }
    let a = mask * in.color.a;
    if (a < 0.01) {
        discard;
    }
    if (is_add) {
        return vec4<f32>(in.color.rgb * a * 1.55, 0.0);
    }
    return vec4<f32>(in.color.rgb * a, a);
}
"#;

const BLOOM_SHADER_WGSL: &str = r#"
struct PostUniforms {
    params0: vec4<f32>,
    params1: vec4<f32>,
};

@group(0) @binding(0) var<uniform> p: PostUniforms;
@group(0) @binding(1) var t_src: texture_2d<f32>;
@group(0) @binding(2) var s_lin: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32((vid >> 1u) & 1u) * 4 - 1);
    var out: VsOut;
    out.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

fn karis_weight(c: vec3<f32>) -> f32 {
    let luma = dot(max(c, vec3<f32>(0.0)), vec3<f32>(0.2126, 0.7152, 0.0722));
    return 1.0 / (1.0 + luma);
}

fn bright_extract(c: vec3<f32>, threshold: f32, knee: f32) -> vec3<f32> {
    let clamped = min(max(c, vec3<f32>(0.0)), vec3<f32>(12.0));
    let br = max(clamped.r, max(clamped.g, clamped.b));
    let soft = clamp(br - threshold + knee, 0.0, 2.0 * knee);
    let soft_term = (soft * soft) / max(4.0 * knee, 1e-5);
    let contrib = max(soft_term, br - threshold) / max(br, 1e-5);
    return clamped * max(contrib, 0.0);
}

@fragment
fn fs_bright(in: VsOut) -> @location(0) vec4<f32> {
    let dims = max(vec2<f32>(textureDimensions(t_src)), vec2<f32>(1.0));
    let tx = 1.0 / dims;
    let c0 = textureSample(t_src, s_lin, in.uv + vec2<f32>(-0.5, -0.5) * tx).rgb;
    let c1 = textureSample(t_src, s_lin, in.uv + vec2<f32>( 0.5, -0.5) * tx).rgb;
    let c2 = textureSample(t_src, s_lin, in.uv + vec2<f32>(-0.5,  0.5) * tx).rgb;
    let c3 = textureSample(t_src, s_lin, in.uv + vec2<f32>( 0.5,  0.5) * tx).rgb;
    let w0 = karis_weight(c0);
    let w1 = karis_weight(c1);
    let w2 = karis_weight(c2);
    let w3 = karis_weight(c3);
    let avg = (c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3) / max(w0 + w1 + w2 + w3, 1e-5);
    return vec4<f32>(bright_extract(avg, p.params1.x, p.params1.y), 1.0);
}

// 13-tap Jimenez downsample filter (eliminates fireflies & ringing across mip steps).
@fragment
fn fs_downsample(in: VsOut) -> @location(0) vec4<f32> {
    let dims = max(vec2<f32>(textureDimensions(t_src)), vec2<f32>(1.0));
    let tx = 1.0 / dims;
    let uv = in.uv;
    let a = textureSample(t_src, s_lin, uv + vec2<f32>(-2.0, -2.0) * tx).rgb;
    let b = textureSample(t_src, s_lin, uv + vec2<f32>( 0.0, -2.0) * tx).rgb;
    let c = textureSample(t_src, s_lin, uv + vec2<f32>( 2.0, -2.0) * tx).rgb;
    let d = textureSample(t_src, s_lin, uv + vec2<f32>(-1.0, -1.0) * tx).rgb;
    let e = textureSample(t_src, s_lin, uv + vec2<f32>( 1.0, -1.0) * tx).rgb;
    let f = textureSample(t_src, s_lin, uv + vec2<f32>(-2.0,  0.0) * tx).rgb;
    let g = textureSample(t_src, s_lin, uv).rgb;
    let h = textureSample(t_src, s_lin, uv + vec2<f32>( 2.0,  0.0) * tx).rgb;
    let i = textureSample(t_src, s_lin, uv + vec2<f32>(-1.0,  1.0) * tx).rgb;
    let j = textureSample(t_src, s_lin, uv + vec2<f32>( 1.0,  1.0) * tx).rgb;
    let k = textureSample(t_src, s_lin, uv + vec2<f32>(-2.0,  2.0) * tx).rgb;
    let l = textureSample(t_src, s_lin, uv + vec2<f32>( 0.0,  2.0) * tx).rgb;
    let m = textureSample(t_src, s_lin, uv + vec2<f32>( 2.0,  2.0) * tx).rgb;

    var out_c = g * 0.125;
    out_c += (a + c + k + m) * 0.03125;
    out_c += (b + f + h + l) * 0.0625;
    out_c += (d + e + i + j) * 0.125;
    return vec4<f32>(max(out_c, vec3<f32>(0.0)), 1.0);
}

// 9-tap 3x3 tent filter upsample (additively blended onto the previous mip level).
@fragment
fn fs_upsample(in: VsOut) -> @location(0) vec4<f32> {
    let dims = max(vec2<f32>(textureDimensions(t_src)), vec2<f32>(1.0));
    let tx = 1.0 / dims;
    let uv = in.uv;
    let a = textureSample(t_src, s_lin, uv + vec2<f32>(-1.0, -1.0) * tx).rgb;
    let b = textureSample(t_src, s_lin, uv + vec2<f32>( 0.0, -1.0) * tx).rgb;
    let c = textureSample(t_src, s_lin, uv + vec2<f32>( 1.0, -1.0) * tx).rgb;
    let d = textureSample(t_src, s_lin, uv + vec2<f32>(-1.0,  0.0) * tx).rgb;
    let e = textureSample(t_src, s_lin, uv).rgb;
    let f = textureSample(t_src, s_lin, uv + vec2<f32>( 1.0,  0.0) * tx).rgb;
    let g = textureSample(t_src, s_lin, uv + vec2<f32>(-1.0,  1.0) * tx).rgb;
    let h = textureSample(t_src, s_lin, uv + vec2<f32>( 0.0,  1.0) * tx).rgb;
    let i = textureSample(t_src, s_lin, uv + vec2<f32>( 1.0,  1.0) * tx).rgb;
    let out_c = (a + c + g + i + (b + d + f + h) * 2.0 + e * 4.0) * (1.0 / 16.0);
    return vec4<f32>(max(out_c, vec3<f32>(0.0)), 1.0);
}
"#;

const TONEMAP_SHADER_WGSL: &str = r#"
struct PostUniforms {
    params0: vec4<f32>, // exposure, bloom_strength, inv_gamma, direct_encode_srgb
    params1: vec4<f32>, // threshold, knee, fxaa_decode_srgb, 0
};

@group(0) @binding(0) var<uniform> p: PostUniforms;
@group(0) @binding(1) var t_hdr: texture_2d<f32>;
@group(0) @binding(2) var t_bloom: texture_2d<f32>;
@group(0) @binding(3) var s_lin: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32((vid >> 1u) & 1u) * 4 - 1);
    var out: VsOut;
    out.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb_ch(c: f32) -> f32 {
    if (c <= 0.0031308) {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(linear_to_srgb_ch(v.r), linear_to_srgb_ch(v.g), linear_to_srgb_ch(v.b));
}

fn tonemap_core(uv: vec2<f32>) -> vec3<f32> {
    var hdr = max(textureSample(t_hdr, s_lin, uv).rgb, vec3<f32>(0.0));
    if (p.params0.y > 0.0) {
        hdr += max(textureSample(t_bloom, s_lin, uv).rgb, vec3<f32>(0.0)) * p.params0.y;
    }
    let mapped = aces_tonemap(hdr * p.params0.x);
    return pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(p.params0.z));
}

@fragment
fn fs_tonemap_direct(in: VsOut) -> @location(0) vec4<f32> {
    var col = tonemap_core(in.uv);
    if (p.params0.w > 0.5) {
        col = linear_to_srgb(col);
    }
    return vec4<f32>(col, 1.0);
}

@fragment
fn fs_tonemap_fxaa_prep(in: VsOut) -> @location(0) vec4<f32> {
    let srgb = linear_to_srgb(tonemap_core(in.uv));
    let luma = dot(srgb, vec3<f32>(0.299, 0.587, 0.114));
    return vec4<f32>(srgb, luma);
}
"#;

const FXAA_SHADER_WGSL: &str = r#"
struct PostUniforms {
    params0: vec4<f32>,
    params1: vec4<f32>, // z = 1.0 when target_format is sRGB (convert sRGB -> linear)
};

@group(0) @binding(0) var<uniform> p: PostUniforms;
@group(0) @binding(1) var t_ldr: texture_2d<f32>;
@group(0) @binding(2) var s_lin: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32((vid >> 1u) & 1u) * 4 - 1);
    var out: VsOut;
    out.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

fn srgb_to_linear_ch(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(srgb_to_linear_ch(v.r), srgb_to_linear_ch(v.g), srgb_to_linear_ch(v.b));
}

@fragment
fn fs_fxaa(in: VsOut) -> @location(0) vec4<f32> {
    let dims = max(vec2<f32>(textureDimensions(t_ldr)), vec2<f32>(1.0));
    let tx = 1.0 / dims;
    let uv = in.uv;

    let c_m = textureSampleLevel(t_ldr, s_lin, uv, 0.0);
    let l_m = c_m.a;
    let l_n = textureSampleLevel(t_ldr, s_lin, uv + vec2<f32>( 0.0, -tx.y), 0.0).a;
    let l_s = textureSampleLevel(t_ldr, s_lin, uv + vec2<f32>( 0.0,  tx.y), 0.0).a;
    let l_w = textureSampleLevel(t_ldr, s_lin, uv + vec2<f32>(-tx.x,  0.0), 0.0).a;
    let l_e = textureSampleLevel(t_ldr, s_lin, uv + vec2<f32>( tx.x,  0.0), 0.0).a;

    let l_min = min(l_m, min(min(l_n, l_s), min(l_w, l_e)));
    let l_max = max(l_m, max(max(l_n, l_s), max(l_w, l_e)));
    let l_range = l_max - l_min;

    var out_srgb = c_m.rgb;
    if (l_range >= max(0.0312, l_max * 0.125)) {
        let l_nw = textureSampleLevel(t_ldr, s_lin, uv + vec2<f32>(-tx.x, -tx.y), 0.0).a;
        let l_ne = textureSampleLevel(t_ldr, s_lin, uv + vec2<f32>( tx.x, -tx.y), 0.0).a;
        let l_sw = textureSampleLevel(t_ldr, s_lin, uv + vec2<f32>(-tx.x,  tx.y), 0.0).a;
        let l_se = textureSampleLevel(t_ldr, s_lin, uv + vec2<f32>( tx.x,  tx.y), 0.0).a;

        let l_ns = l_n + l_s;
        let l_we = l_w + l_e;
        let sub_avg = (2.0 * (l_ns + l_we) + (l_nw + l_ne + l_sw + l_se)) * (1.0 / 12.0);
        let sub_c = clamp(abs(sub_avg - l_m) / l_range, 0.0, 1.0);
        let sub_s = (-2.0 * sub_c + 3.0) * sub_c * sub_c;
        let sub_blend = sub_s * sub_s * 0.75;

        let edge_h = abs(-2.0 * l_w + l_nw + l_sw)
            + abs(-2.0 * l_m + l_n + l_s) * 2.0
            + abs(-2.0 * l_e + l_ne + l_se);
        let edge_v = abs(-2.0 * l_n + l_nw + l_ne)
            + abs(-2.0 * l_m + l_w + l_e) * 2.0
            + abs(-2.0 * l_s + l_sw + l_se);
        let is_horiz = edge_h >= edge_v;

        let l1 = select(l_w, l_n, is_horiz);
        let l2 = select(l_e, l_s, is_horiz);
        let g1 = abs(l1 - l_m);
        let g2 = abs(l2 - l_m);
        let step_len = select(tx.x, tx.y, is_horiz);
        let neg_side = g1 >= g2;
        let signed_step = select(step_len, -step_len, neg_side);
        let l_edge = 0.5 * (select(l2, l1, neg_side) + l_m);
        let grad_thresh = 0.25 * max(g1, g2);

        var cur_uv = uv;
        if (is_horiz) {
            cur_uv.y += signed_step * 0.5;
        } else {
            cur_uv.x += signed_step * 0.5;
        }
        let step_tan = select(vec2<f32>(0.0, tx.y), vec2<f32>(tx.x, 0.0), is_horiz);

        var uv1 = cur_uv - step_tan;
        var uv2 = cur_uv + step_tan;
        var d1 = textureSampleLevel(t_ldr, s_lin, uv1, 0.0).a - l_edge;
        var d2 = textureSampleLevel(t_ldr, s_lin, uv2, 0.0).a - l_edge;
        var r1 = abs(d1) >= grad_thresh;
        var r2 = abs(d2) >= grad_thresh;
        for (var i = 0; i < 8; i = i + 1) {
            if (r1 && r2) {
                break;
            }
            if (!r1) {
                uv1 -= step_tan * 1.5;
                d1 = textureSampleLevel(t_ldr, s_lin, uv1, 0.0).a - l_edge;
                r1 = abs(d1) >= grad_thresh;
            }
            if (!r2) {
                uv2 += step_tan * 1.5;
                d2 = textureSampleLevel(t_ldr, s_lin, uv2, 0.0).a - l_edge;
                r2 = abs(d2) >= grad_thresh;
            }
        }
        let dist1 = select(uv.y - uv1.y, uv.x - uv1.x, is_horiz);
        let dist2 = select(uv2.y - uv.y, uv2.x - uv.x, is_horiz);
        let near1 = dist1 < dist2;
        let dist_near = min(dist1, dist2);
        let span = max(dist1 + dist2, 1e-5);
        let l_center_lt = l_m < l_edge;
        let good_span = select((d2 < 0.0) != l_center_lt, (d1 < 0.0) != l_center_lt, near1);
        let edge_blend = select(0.0, 0.5 - dist_near / span, good_span);
        let final_blend = max(sub_blend, edge_blend);

        var sample_uv = uv;
        if (is_horiz) {
            sample_uv.y += signed_step * final_blend;
        } else {
            sample_uv.x += signed_step * final_blend;
        }
        out_srgb = textureSampleLevel(t_ldr, s_lin, sample_uv, 0.0).rgb;
    }

    let out_col = select(out_srgb, srgb_to_linear(out_srgb), p.params1.z > 0.5);
    return vec4<f32>(out_col, 1.0);
}
"#;

/// Owned offscreen HDR, depth, bloom mip chain, and optional FXAA LDR render targets.
struct PostTargets {
    width: u32,
    height: u32,
    hdr_msaa_view: Option<wgpu::TextureView>,
    hdr_resolve_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    bloom_views: Vec<wgpu::TextureView>,
    /// Bind group 0 = `hdr_resolve` (for bright pass), `1..` = `bloom_views[i - 1]` (for down/upsample).
    bloom_src_bgs: Vec<wgpu::BindGroup>,
    tonemap_bg: wgpu::BindGroup,
    /// Present when `active_aa == AaMode::Fxaa`: `(ldr_srgb_luma_view, fxaa_sample_bg)`.
    fxaa: Option<(wgpu::TextureView, wgpu::BindGroup)>,
}

/// 3D HDR scene pipelines whose `multisample.count` depends on `AaMode` (`1` or `4`).
struct ScenePipelines {
    pipeline: wgpu::RenderPipeline,
    decal_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    terrain_pipeline: wgpu::RenderPipeline,
    water_pipeline: wgpu::RenderPipeline,
    instanced_pipeline: wgpu::RenderPipeline,
    shield_pipeline: wgpu::RenderPipeline,
    vfx_pipeline: wgpu::RenderPipeline,
}

pub(crate) struct GpuSceneResources {
    scene_pipes: ScenePipelines,
    shadow_pipeline: wgpu::RenderPipeline,
    hud_pipeline: wgpu::RenderPipeline,
    bright_pipeline: wgpu::RenderPipeline,
    downsample_pipeline: wgpu::RenderPipeline,
    upsample_pipeline: wgpu::RenderPipeline,
    tonemap_direct_pipeline: wgpu::RenderPipeline,
    tonemap_fxaa_prep_pipeline: wgpu::RenderPipeline,
    fxaa_pipeline: wgpu::RenderPipeline,

    /// Saved layouts/shaders so switching `AaMode` (`1` <-> `4` MSAA) rebuilds only 3D pipelines.
    mat_pl: wgpu::PipelineLayout,
    decal_pl: wgpu::PipelineLayout,
    sky_pl: wgpu::PipelineLayout,
    terrain_pl: wgpu::PipelineLayout,
    water_pl: wgpu::PipelineLayout,
    inst_pl: wgpu::PipelineLayout,
    uniform_only_pl: wgpu::PipelineLayout,
    vfx_pl: wgpu::PipelineLayout,
    scene_shader: wgpu::ShaderModule,
    decal_shader: wgpu::ShaderModule,
    sky_shader: wgpu::ShaderModule,
    terrain_shader: wgpu::ShaderModule,
    water_shader: wgpu::ShaderModule,
    inst_shader: wgpu::ShaderModule,
    shield_shader: wgpu::ShaderModule,
    vfx_shader: wgpu::ShaderModule,

    /// 3-entry layout (uniform, albedo, sampler) for decals & HUD.
    bind_group_layout: wgpu::BindGroupLayout,
    /// 4-entry layout (uniform, albedo, sampler, linear Rg8Unorm spec+glow) for units/props/scene.
    mat_bgl: wgpu::BindGroupLayout,
    default_spec_glow_view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    overlay_mat_bind_group: wgpu::BindGroup,
    overlay_bind_group: wgpu::BindGroup,
    minimap_bind_group: wgpu::BindGroup,
    hud_icons_bind_group: wgpu::BindGroup,
    hud_font_bind_group: wgpu::BindGroup,
    strat_icons_bind_group: wgpu::BindGroup,
    uniform_only_bg: wgpu::BindGroup,
    vfx_bgl: wgpu::BindGroupLayout,
    vfx_bind_group: wgpu::BindGroup,

    post_single_bgl: wgpu::BindGroupLayout,
    tonemap_bgl: wgpu::BindGroupLayout,
    post_uniform_buf: wgpu::Buffer,
    post: PostTargets,
    target_is_srgb: bool,
    msaa4_supported: bool,
    bloom_enabled: bool,
    active_aa: AaMode,
    gamma: f32,
    bloom_strength: f32,

    bone_bgl: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    bone_buf: wgpu::Buffer,
    bone_buf_cap: usize,
    bone_bind_group: wgpu::BindGroup,

    fow_tex: wgpu::Texture,

    shadow_view: wgpu::TextureView,
    shadow_bind_group: wgpu::BindGroup,
    shadow_bgl: wgpu::BindGroupLayout,
    /// Actual shadow-map edge length in texels (`1` = placeholder when `Off`).
    shadow_size: u32,
    /// `false` skips the shadow pass and forces PCF to return fully lit.
    shadows_enabled: bool,
    /// Cached [`shadow_uniform_params`] output, written into uniforms each frame.
    shadow_params: [f32; 4],
    sky_vb: wgpu::Buffer,
    sky_bind_group: Option<wgpu::BindGroup>,
    sky_fallback_bg: wgpu::BindGroup,

    sampler_clamp: wgpu::Sampler,
    sampler_repeat: wgpu::Sampler,
    uniform_buf: wgpu::Buffer,

    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    terrain_bind_group: Option<wgpu::BindGroup>,

    water_draw: Option<(wgpu::Buffer, wgpu::Buffer, u32, wgpu::BindGroup)>,
    decal_batches: Vec<(wgpu::Buffer, wgpu::Buffer, u32, wgpu::BindGroup)>,
    /// Phase 41: dynamic decal textures (bind groups) and geometry per kind, plus the
    /// tracker revision the geometry was built from.
    dyn_decal_bgs: Vec<Option<wgpu::BindGroup>>,
    dyn_decal_geo: Vec<Option<(wgpu::Buffer, wgpu::Buffer, u32)>>,
    pub dyn_decal_rev: u64,
    static_overlay: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    prop_batches: Vec<(
        wgpu::Buffer,
        wgpu::Buffer,
        u32,
        wgpu::BindGroup,
        wgpu::Buffer,
        u32,
        [f32; 3],
        [f32; 3],
    )>,
    unit_batches: Vec<(wgpu::Buffer, wgpu::Buffer, u32, wgpu::BindGroup, wgpu::Buffer, u32)>,
    shield_mesh: (wgpu::Buffer, wgpu::Buffer, u32),
    shield_instances: Option<(wgpu::Buffer, u32)>,
    vfx_draw: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    dynamic: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,

    hud_strat_icons: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    hud_minimap: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    hud_bg: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    hud_icons: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    hud_text: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    hud_ttf: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,

    lighting: ScMapLighting,
    map_size: [f32; 2],
    water_levels: [f32; 3],
    last_view_proj: [[f32; 4]; 4],
    last_shadow_view_proj: [[f32; 4]; 4],
    start_time: Instant,
    queue: wgpu::Queue,
    /// Counters of the last [`GpuSceneResources::encode_draw`] call.
    draw_stats: std::cell::Cell<DrawStats>,
    /// Optional render-pass timestamp queries (needs `Features::TIMESTAMP_QUERY`).
    gpu_timer: Option<GpuTimer>,
}

/// Per-frame draw counters (for `bench-render` and the F10 overlay).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct DrawStats {
    pub draw_calls: u32,
    pub instances: u64,
    pub triangles: u64,
    /// Unit/wreck/projectile instanced batches in the main pass.
    pub unit_batches: u32,
    pub unit_instances: u64,
}

impl DrawStats {
    #[inline]
    fn add(&mut self, index_count: u32, instances: u32) {
        self.draw_calls += 1;
        self.instances += u64::from(instances);
        self.triangles += u64::from(index_count / 3) * u64::from(instances);
    }
}

/// Render-pass GPU timing via timestamp queries:
/// `[shadow begin, shadow end, scene begin, scene end, post begin, post end]`.
/// One readback is in flight at a time; frames encoded while it is pending are not timed.
pub(crate) struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f32,
    /// 0 = idle, 1 = written this frame (awaiting map request), 2 = map requested, 3 = mapped.
    state: Arc<std::sync::atomic::AtomicU8>,
    shadow_written: std::cell::Cell<bool>,
    last_ms: std::cell::Cell<Option<(f64, f64)>>,
}

impl GpuTimer {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_timer_qs"),
            ty: wgpu::QueryType::Timestamp,
            count: 6,
        });
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_timer_resolve"),
            size: 48,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_timer_readback"),
            size: 48,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Self {
            query_set,
            resolve_buf,
            readback,
            period_ns: queue.get_timestamp_period(),
            state: Arc::new(std::sync::atomic::AtomicU8::new(0)),
            shadow_written: std::cell::Cell::new(false),
            last_ms: std::cell::Cell::new(None),
        })
    }

    fn idle(&self) -> bool {
        self.state.load(std::sync::atomic::Ordering::Acquire) == 0
    }
}

fn upload_spec_glow_2d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sg: &SpecGlowMap,
    label: &str,
) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width: sg.width.max(1),
        height: sg.height.max(1),
        depth_or_array_layers: 1,
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg8Unorm,
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
        &sg.pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(2 * size.width),
            rows_per_image: Some(size.height),
        },
        size,
    );
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn upload_texture_2d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img: &RgbaImage,
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width: img.width.max(1),
        height: img.height.max(1),
        depth_or_array_layers: 1,
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
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
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * size.width),
            rows_per_image: Some(size.height),
        },
        size,
    );
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn upload_texture_2d_array(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layers: &[RgbaImage],
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::TextureView {
    let w = layers.first().map(|i| i.width).unwrap_or(1).max(1);
    let h = layers.first().map(|i| i.height).unwrap_or(1).max(1);
    let count = layers.len().max(1) as u32;
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: count,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (idx, img) in layers.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: idx as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &img.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }
    tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        array_layer_count: Some(count),
        ..Default::default()
    })
}

fn upload_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    faces: &[RgbaImage; 6],
    label: &str,
) -> wgpu::TextureView {
    let w = faces[0].width.max(1);
    let h = faces[0].height.max(1);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (face_idx, img) in faces.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: face_idx as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &img.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * img.width.max(1)),
                rows_per_image: Some(img.height.max(1)),
            },
            wgpu::Extent3d {
                width: img.width.max(1),
                height: img.height.max(1),
                depth_or_array_layers: 1,
            },
        );
    }
    tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        array_layer_count: Some(6),
        ..Default::default()
    })
}

fn create_rgba_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    uniform_buf: &wgpu::Buffer,
    sampler: &wgpu::Sampler,
    img: &RgbaImage,
    label: &str,
) -> wgpu::BindGroup {
    let view = upload_texture_2d(device, queue, img, wgpu::TextureFormat::Rgba8UnormSrgb, label);
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn create_material_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    uniform_buf: &wgpu::Buffer,
    sampler: &wgpu::Sampler,
    img: &RgbaImage,
    spec_glow: Option<&SpecGlowMap>,
    default_spec_glow_view: &wgpu::TextureView,
    label: &str,
) -> wgpu::BindGroup {
    let view = upload_texture_2d(device, queue, img, wgpu::TextureFormat::Rgba8UnormSrgb, label);
    let owned_sg = spec_glow.map(|sg| upload_spec_glow_2d(device, queue, sg, label));
    let sg_view = owned_sg.as_ref().unwrap_or(default_spec_glow_view);
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(sg_view),
            },
        ],
    })
}

fn create_post_single_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    post_buf: &wgpu::Buffer,
    src_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: post_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(src_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn create_post_targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    aa: AaMode,
    post_single_bgl: &wgpu::BindGroupLayout,
    tonemap_bgl: &wgpu::BindGroupLayout,
    post_buf: &wgpu::Buffer,
    sampler_clamp: &wgpu::Sampler,
) -> PostTargets {
    let w = width.max(1);
    let h = height.max(1);
    let sample_count = aa.sample_count();

    let hdr_msaa_view = (sample_count > 1).then(|| {
        create_attachment(
            device,
            w,
            h,
            HDR_FORMAT,
            sample_count,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            "hdr_msaa",
        )
        .1
    });
    let (_, hdr_resolve_view) = create_attachment(
        device,
        w,
        h,
        HDR_FORMAT,
        1,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        "hdr_resolve",
    );
    let (_, depth_view) = create_attachment(
        device,
        w,
        h,
        DEPTH_FORMAT,
        sample_count,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
        "hdr_depth",
    );

    let mip_sizes = bloom_mip_sizes(w, h, BLOOM_MIPS);
    let mut bloom_views = Vec::with_capacity(mip_sizes.len());
    for (idx, &(mw, mh)) in mip_sizes.iter().enumerate() {
        let (_, v) = create_attachment(
            device,
            mw,
            mh,
            HDR_FORMAT,
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            &format!("bloom_mip_{idx}"),
        );
        bloom_views.push(v);
    }

    let mut bloom_src_bgs = Vec::with_capacity(bloom_views.len() + 1);
    bloom_src_bgs.push(create_post_single_bg(
        device,
        post_single_bgl,
        post_buf,
        &hdr_resolve_view,
        sampler_clamp,
        "bloom_src_hdr",
    ));
    for (idx, v) in bloom_views.iter().enumerate() {
        bloom_src_bgs.push(create_post_single_bg(
            device,
            post_single_bgl,
            post_buf,
            v,
            sampler_clamp,
            &format!("bloom_src_mip_{idx}"),
        ));
    }

    let tonemap_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tonemap_bg"),
        layout: tonemap_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: post_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&hdr_resolve_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&bloom_views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler_clamp),
            },
        ],
    });

    let fxaa = (aa == AaMode::Fxaa).then(|| {
        let (_, ldr_view) = create_attachment(
            device,
            w,
            h,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            "ldr_fxaa",
        );
        let bg = create_post_single_bg(
            device,
            post_single_bgl,
            post_buf,
            &ldr_view,
            sampler_clamp,
            "fxaa_bg",
        );
        (ldr_view, bg)
    });

    PostTargets {
        width: w,
        height: h,
        hdr_msaa_view,
        hdr_resolve_view,
        depth_view,
        bloom_views,
        bloom_src_bgs,
        tonemap_bg,
        fxaa,
    }
}

#[allow(clippy::too_many_arguments)]
fn create_vfx_array_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    uniform_buf: &wgpu::Buffer,
    sampler_clamp: &wgpu::Sampler,
    sampler_repeat: &wgpu::Sampler,
    width: u32,
    height: u32,
    layers: &[RgbaImage],
) -> wgpu::BindGroup {
    let w = width.max(1);
    let h = height.max(1);
    let layer_count = layers.len().clamp(1, 256) as u32;
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vfx_tex_array"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: layer_count,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let expected = (w as usize) * (h as usize) * 4;
    let fallback_pixels = vec![255u8; expected];
    for (i, img) in layers.iter().take(layer_count as usize).enumerate() {
        let data = if img.width == w && img.height == h && img.pixels.len() == expected {
            img.pixels.as_slice()
        } else {
            fallback_pixels.as_slice()
        };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: i as u32 },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vfx_array_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler_clamp),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler_repeat),
            },
        ],
    })
}

fn sky_cube_vertices() -> Vec<[f32; 3]> {
    let p = 1.0_f32;
    vec![
        // +Z
        [-p, -p, p], [p, -p, p], [p, p, p], [-p, -p, p], [p, p, p], [-p, p, p],
        // -Z
        [p, -p, -p], [-p, -p, -p], [-p, p, -p], [p, -p, -p], [-p, p, -p], [p, p, -p],
        // +X
        [p, -p, p], [p, -p, -p], [p, p, -p], [p, -p, p], [p, p, -p], [p, p, p],
        // -X
        [-p, -p, -p], [-p, -p, p], [-p, p, p], [-p, -p, -p], [-p, p, p], [-p, p, -p],
        // +Y
        [-p, p, p], [p, p, p], [p, p, -p], [-p, p, p], [p, p, -p], [-p, p, -p],
        // -Y
        [-p, -p, -p], [p, -p, -p], [p, -p, p], [-p, -p, -p], [p, -p, p], [-p, -p, p],
    ]
}

/// Builds a shadow-map render target plus its sampling bind group.
/// `size` is clamped to at least 1 so `Off` still yields a valid (unsampled) binding.
fn create_shadow_target(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    size: u32,
) -> (wgpu::TextureView, wgpu::BindGroup) {
    let edge = size.max(1);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow_map_tex"),
        size: wgpu::Extent3d {
            width: edge,
            height: edge,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("shadow_cmp_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shadow_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    (view, bg)
}

fn gpu_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: &[wgpu::VertexAttribute] = &[
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
            offset: 12,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
            offset: 24,
            shader_location: 2,
            format: wgpu::VertexFormat::Float32x2,
        },
        wgpu::VertexAttribute {
            offset: 32,
            shader_location: 3,
            format: wgpu::VertexFormat::Float32x4,
        },
    ];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<GpuVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: ATTRS,
    }
}

fn gpu_instance_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: &[wgpu::VertexAttribute] = &[
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 4,
            format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
            offset: 16,
            shader_location: 5,
            format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
            offset: 32,
            shader_location: 6,
            format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
            offset: 48,
            shader_location: 7,
            format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
            offset: 64,
            shader_location: 8,
            format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
            offset: 80,
            shader_location: 9,
            format: wgpu::VertexFormat::Float32x4,
        },
    ];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<GpuUnitInstance>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: ATTRS,
    }
}

fn vfx_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: &[wgpu::VertexAttribute] = &[
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
            offset: 12,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32x2,
        },
        wgpu::VertexAttribute {
            offset: 20,
            shader_location: 2,
            format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
            offset: 36,
            shader_location: 3,
            format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
            offset: 40,
            shader_location: 4,
            format: wgpu::VertexFormat::Uint32,
        },
    ];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<VfxVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: ATTRS,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_scene_pipelines(
    device: &wgpu::Device,
    sample_count: u32,
    mat_pl: &wgpu::PipelineLayout,
    decal_pl: &wgpu::PipelineLayout,
    sky_pl: &wgpu::PipelineLayout,
    terrain_pl: &wgpu::PipelineLayout,
    water_pl: &wgpu::PipelineLayout,
    inst_pl: &wgpu::PipelineLayout,
    uniform_only_pl: &wgpu::PipelineLayout,
    vfx_pl: &wgpu::PipelineLayout,
    scene_shader: &wgpu::ShaderModule,
    decal_shader: &wgpu::ShaderModule,
    sky_shader: &wgpu::ShaderModule,
    terrain_shader: &wgpu::ShaderModule,
    water_shader: &wgpu::ShaderModule,
    inst_shader: &wgpu::ShaderModule,
    shield_shader: &wgpu::ShaderModule,
    vfx_shader: &wgpu::ShaderModule,
) -> ScenePipelines {
    let vertex_layout = gpu_vertex_layout();
    let instance_layout = gpu_instance_layout();
    let vfx_layout = vfx_vertex_layout();
    let msaa = wgpu::MultisampleState {
        count: sample_count,
        mask: !0,
        alpha_to_coverage_enabled: false,
    };

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("scene_pipeline"),
        layout: Some(mat_pl),
        vertex: wgpu::VertexState {
            module: scene_shader,
            entry_point: Some("vs_main"),
            buffers: std::slice::from_ref(&vertex_layout),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: scene_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: msaa,
        multiview: None,
        cache: None,
    });

    let decal_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("decal_pipeline"),
        layout: Some(decal_pl),
        vertex: wgpu::VertexState {
            module: decal_shader,
            entry_point: Some("vs_main"),
            buffers: std::slice::from_ref(&vertex_layout),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: decal_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: -2,
                slope_scale: -1.0,
                clamp: 0.0,
            },
        }),
        multisample: msaa,
        multiview: None,
        cache: None,
    });

    let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sky_pipeline"),
        layout: Some(sky_pl),
        vertex: wgpu::VertexState {
            module: sky_shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 12,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            }],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: sky_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: msaa,
        multiview: None,
        cache: None,
    });

    let terrain_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("terrain_pipeline"),
        layout: Some(terrain_pl),
        vertex: wgpu::VertexState {
            module: terrain_shader,
            entry_point: Some("vs_main"),
            buffers: std::slice::from_ref(&vertex_layout),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: terrain_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: msaa,
        multiview: None,
        cache: None,
    });

    let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("water_pipeline"),
        layout: Some(water_pl),
        vertex: wgpu::VertexState {
            module: water_shader,
            entry_point: Some("vs_main"),
            buffers: std::slice::from_ref(&vertex_layout),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: water_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: msaa,
        multiview: None,
        cache: None,
    });

    let instanced_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("instanced_pipeline"),
        layout: Some(inst_pl),
        vertex: wgpu::VertexState {
            module: inst_shader,
            entry_point: Some("vs_main"),
            buffers: &[vertex_layout.clone(), instance_layout.clone()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: inst_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: msaa,
        multiview: None,
        cache: None,
    });

    let shield_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shield_pipeline"),
        layout: Some(uniform_only_pl),
        vertex: wgpu::VertexState {
            module: shield_shader,
            entry_point: Some("vs_main"),
            buffers: &[vertex_layout, instance_layout],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shield_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: msaa,
        multiview: None,
        cache: None,
    });

    let vfx_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("vfx_pipeline"),
        layout: Some(vfx_pl),
        vertex: wgpu::VertexState {
            module: vfx_shader,
            entry_point: Some("vs_main"),
            buffers: &[vfx_layout],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: vfx_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: msaa,
        multiview: None,
        cache: None,
    });

    ScenePipelines {
        pipeline,
        decal_pipeline,
        sky_pipeline,
        terrain_pipeline,
        water_pipeline,
        instanced_pipeline,
        shield_pipeline,
        vfx_pipeline,
    }
}

fn create_fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    frag_entry: &str,
    format: wgpu::TextureFormat,
    blend: wgpu::BlendState,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_fullscreen"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(frag_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

impl GpuSceneResources {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &LoadedScene,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        msaa4_supported: bool,
        camera: &OrbitCamera,
    ) -> Self {
        let lighting = scene
            .terrain_splat
            .as_ref()
            .map(|t| t.lighting.clone())
            .unwrap_or_default();
        let map_size = scene
            .terrain_splat
            .as_ref()
            .map(|t| t.map_size)
            .unwrap_or([1024.0, 1024.0]);
        let water_levels = scene
            .terrain_splat
            .as_ref()
            .map(|t| [t.water_elevation, t.water_deep, t.water_abyss])
            .unwrap_or([-1000.0, -1000.0, -1000.0]);
        let start_time = Instant::now();
        let shadow_quality = crate::settings::shadow_quality();
        let shadow_params = shadow_uniform_params(shadow_quality);
        let (bloom_enabled, req_aa, gamma) = crate::settings::render_post();
        let active_aa = resolve_aa_mode(req_aa, msaa4_supported);
        let sample_count = active_aa.sample_count();
        let target_is_srgb = target_format.is_srgb();
        let post_uniforms = make_post_uniforms(&lighting, bloom_enabled, gamma, target_is_srgb);
        let bloom_strength = post_uniforms.params0[1];
        let post_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("post_uniforms"),
            contents: bytemuck::bytes_of(&post_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = make_uniforms(camera, &lighting, map_size, water_levels, 0.0, shadow_params);
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler_clamp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scene_sampler_clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let sampler_repeat = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scene_sampler_repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let mat_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mat_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
            ],
        });

        let default_spec_glow_view = upload_spec_glow_2d(
            device,
            queue,
            &SpecGlowMap {
                width: 1,
                height: 1,
                pixels: vec![128, 0],
            },
            "default_spec_glow",
        );

        let uniform_only_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform_only_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform_only_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform_only_bg"),
            layout: &uniform_only_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // Shadow map (Depth32Float + comparison sampler, sized from settings).
        let shadow_size = shadow_quality.map_size().max(1);
        let shadows_enabled = shadow_quality.map_size() > 0;
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let (shadow_view, shadow_bind_group) =
            create_shadow_target(device, &shadow_bgl, shadow_size);

        let white_img = RgbaImage {
            width: 1,
            height: 1,
            pixels: vec![255, 255, 255, 255],
        };
        let bind_group = create_material_bind_group(
            device,
            queue,
            &mat_bgl,
            &uniform_buf,
            &sampler_clamp,
            &scene.albedo,
            scene.spec_glow.as_ref(),
            &default_spec_glow_view,
            "scene_bg",
        );
        let overlay_mat_bind_group = create_material_bind_group(
            device,
            queue,
            &mat_bgl,
            &uniform_buf,
            &sampler_clamp,
            &white_img,
            None,
            &default_spec_glow_view,
            "overlay_mat_bg",
        );
        let overlay_bind_group = create_rgba_bind_group(
            device,
            queue,
            &bind_group_layout,
            &uniform_buf,
            &sampler_clamp,
            &white_img,
            "overlay_bg",
        );
        let minimap_bind_group = create_rgba_bind_group(
            device,
            queue,
            &bind_group_layout,
            &uniform_buf,
            &sampler_clamp,
            &white_img,
            "minimap_bg",
        );
        let hud_icons_bind_group = create_rgba_bind_group(
            device,
            queue,
            &bind_group_layout,
            &uniform_buf,
            &sampler_clamp,
            &white_img,
            "hud_icons_bg",
        );
        let hud_font_bind_group = create_rgba_bind_group(
            device,
            queue,
            &bind_group_layout,
            &uniform_buf,
            &sampler_clamp,
            &white_img,
            "hud_font_bg",
        );
        let strat_icons_bind_group = create_rgba_bind_group(
            device,
            queue,
            &bind_group_layout,
            &uniform_buf,
            &sampler_clamp,
            &white_img,
            "strat_icons_bg",
        );

        let bone_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bone_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let identity_mat: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let bone_buf_cap = 256usize;
        let init_bones = vec![identity_mat; bone_buf_cap];
        let bone_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bone_palette_buf"),
            contents: bytemuck::cast_slice(&init_bones),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let bone_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bone_bg"),
            layout: &bone_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: bone_buf.as_entire_binding(),
            }],
        });

        let fow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fow_mask_tex"),
            size: wgpu::Extent3d {
                width: 128,
                height: 128,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let full_vis = vec![255u8; 128 * 128 * 4];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &fow_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &full_vis,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(128 * 4),
                rows_per_image: Some(128),
            },
            wgpu::Extent3d {
                width: 128,
                height: 128,
                depth_or_array_layers: 1,
            },
        );
        let fow_view = fow_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let vertex_layout = gpu_vertex_layout();
        let instance_layout = gpu_instance_layout();
        let hud_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<HudVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let mat_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mat_pl"),
            bind_group_layouts: &[&mat_bgl],
            push_constant_ranges: &[],
        });
        let decal_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("decal_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let scene_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("farust_scene_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_WGSL.into()),
        });
        let decal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("farust_decal_shader"),
            source: wgpu::ShaderSource::Wgsl(DECAL_SHADER_WGSL.into()),
        });

        // Shadow pipeline (depth-only single-sample with bone palette skinning)
        let shadow_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_pl"),
            bind_group_layouts: &[&uniform_only_bgl, &bone_bgl],
            push_constant_ranges: &[],
        });
        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("farust_shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER_WGSL.into()),
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipeline"),
            layout: Some(&shadow_pl),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout, instance_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Skybox resources
        let sky_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sky_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky_pl"),
            bind_group_layouts: &[&uniform_only_bgl, &sky_bgl],
            push_constant_ranges: &[],
        });
        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("farust_sky_shader"),
            source: wgpu::ShaderSource::Wgsl(SKY_SHADER_WGSL.into()),
        });
        let sky_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sky_vb"),
            contents: bytemuck::cast_slice(&sky_cube_vertices()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let fallback_face = RgbaImage {
            width: 1,
            height: 1,
            pixels: vec![
                (lighting.fog_color[0].clamp(0.0, 1.0) * 255.0) as u8,
                (lighting.fog_color[1].clamp(0.0, 1.0) * 255.0) as u8,
                (lighting.fog_color[2].clamp(0.0, 1.0) * 255.0) as u8,
                255,
            ],
        };
        let fallback_sky_view = upload_cubemap(
            device,
            queue,
            &[
                fallback_face.clone(),
                fallback_face.clone(),
                fallback_face.clone(),
                fallback_face.clone(),
                fallback_face.clone(),
                fallback_face,
            ],
            "sky_fallback",
        );
        let sky_fallback_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky_fallback_bg"),
            layout: &sky_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&fallback_sky_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler_clamp),
                },
            ],
        });
        let sky_bind_group = scene.sky_cube.as_ref().map(|faces| {
            let sky_view = upload_cubemap(device, queue, faces, "sky_cubemap");
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sky_cubemap_bg"),
                layout: &sky_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&sky_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler_clamp),
                    },
                ],
            })
        });

        // Terrain splat pipeline
        let terrain_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
            ],
        });
        let terrain_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_pl"),
            bind_group_layouts: &[&terrain_bgl, &shadow_bgl],
            push_constant_ranges: &[],
        });
        let terrain_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("farust_terrain_shader"),
            source: wgpu::ShaderSource::Wgsl(TERRAIN_SHADER_WGSL.into()),
        });
        let terrain_bind_group = scene.terrain_splat.as_ref().map(|ts| {
            upload_terrain_bind_group(
                device,
                queue,
                &terrain_bgl,
                &uniform_buf,
                &sampler_clamp,
                &sampler_repeat,
                &fow_view,
                ts,
            )
        });

        // Water pipeline
        let water_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let water_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water_pl"),
            bind_group_layouts: &[&water_bgl, &sky_bgl],
            push_constant_ranges: &[],
        });
        let water_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("farust_water_shader"),
            source: wgpu::ShaderSource::Wgsl(WATER_SHADER_WGSL.into()),
        });
        let water_draw = scene.water.as_ref().and_then(|wd| {
            if wd.vertices.is_empty() || wd.indices.is_empty() {
                return None;
            }
            Some(upload_water_draw(
                device,
                queue,
                &water_bgl,
                &uniform_buf,
                &sampler_repeat,
                &sampler_clamp,
                wd,
            ))
        });

        // Instanced props & units layouts/shaders
        let inst_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("inst_pl"),
            bind_group_layouts: &[&mat_bgl, &shadow_bgl, &bone_bgl],
            push_constant_ranges: &[],
        });
        let inst_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("farust_instanced_shader"),
            source: wgpu::ShaderSource::Wgsl(INSTANCED_SHADER_WGSL.into()),
        });

        // Translucent bubble-shield layout/shader
        let uniform_only_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("uniform_only_pl"),
            bind_group_layouts: &[&uniform_only_bgl],
            push_constant_ranges: &[],
        });
        let shield_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("farust_shield_shader"),
            source: wgpu::ShaderSource::Wgsl(SHIELD_SHADER_WGSL.into()),
        });

        // GPU billboard particle VFX resources (supports procedural + FA texture_2d_array sprites/beams)
        let vfx_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vfx_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let vfx_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vfx_pl"),
            bind_group_layouts: &[&vfx_bgl],
            push_constant_ranges: &[],
        });
        let vfx_placeholder = RgbaImage {
            width: 2,
            height: 2,
            pixels: vec![255u8; 2 * 2 * 4],
        };
        let vfx_bind_group = create_vfx_array_bind_group(
            device,
            queue,
            &vfx_bgl,
            &uniform_buf,
            &sampler_clamp,
            &sampler_repeat,
            2,
            2,
            std::slice::from_ref(&vfx_placeholder),
        );
        let vfx_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("farust_vfx_shader"),
            source: wgpu::ShaderSource::Wgsl(VFX_SHADER_WGSL.into()),
        });

        let scene_pipes = build_scene_pipelines(
            device,
            sample_count,
            &mat_pl,
            &decal_pl,
            &sky_pl,
            &terrain_pl,
            &water_pl,
            &inst_pl,
            &uniform_only_pl,
            &vfx_pl,
            &scene_shader,
            &decal_shader,
            &sky_shader,
            &terrain_shader,
            &water_shader,
            &inst_shader,
            &shield_shader,
            &vfx_shader,
        );

        // Post-processing pipelines: bright-pass -> bloom mip chain -> ACES tonemap (+ gamma) -> optional FXAA
        let post_single_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post_single_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let tonemap_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let post_single_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post_single_pl"),
            bind_group_layouts: &[&post_single_bgl],
            push_constant_ranges: &[],
        });
        let tonemap_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tonemap_pl"),
            bind_group_layouts: &[&tonemap_bgl],
            push_constant_ranges: &[],
        });
        let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("farust_bloom_shader"),
            source: wgpu::ShaderSource::Wgsl(BLOOM_SHADER_WGSL.into()),
        });
        let tonemap_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("farust_tonemap_shader"),
            source: wgpu::ShaderSource::Wgsl(TONEMAP_SHADER_WGSL.into()),
        });
        let fxaa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("farust_fxaa_shader"),
            source: wgpu::ShaderSource::Wgsl(FXAA_SHADER_WGSL.into()),
        });

        let additive_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        };
        let bright_pipeline = create_fullscreen_pipeline(
            device,
            "bloom_bright_pipeline",
            &post_single_pl,
            &bloom_shader,
            "fs_bright",
            HDR_FORMAT,
            wgpu::BlendState::REPLACE,
        );
        let downsample_pipeline = create_fullscreen_pipeline(
            device,
            "bloom_downsample_pipeline",
            &post_single_pl,
            &bloom_shader,
            "fs_downsample",
            HDR_FORMAT,
            wgpu::BlendState::REPLACE,
        );
        let upsample_pipeline = create_fullscreen_pipeline(
            device,
            "bloom_upsample_pipeline",
            &post_single_pl,
            &bloom_shader,
            "fs_upsample",
            HDR_FORMAT,
            additive_blend,
        );
        let tonemap_direct_pipeline = create_fullscreen_pipeline(
            device,
            "tonemap_direct_pipeline",
            &tonemap_pl,
            &tonemap_shader,
            "fs_tonemap_direct",
            target_format,
            wgpu::BlendState::REPLACE,
        );
        let tonemap_fxaa_prep_pipeline = create_fullscreen_pipeline(
            device,
            "tonemap_fxaa_prep_pipeline",
            &tonemap_pl,
            &tonemap_shader,
            "fs_tonemap_fxaa_prep",
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::BlendState::REPLACE,
        );
        let fxaa_pipeline = create_fullscreen_pipeline(
            device,
            "fxaa_pipeline",
            &post_single_pl,
            &fxaa_shader,
            "fs_fxaa",
            target_format,
            wgpu::BlendState::REPLACE,
        );
        let post = create_post_targets(
            device,
            width,
            height,
            active_aa,
            &post_single_bgl,
            &tonemap_bgl,
            &post_uniform_buf,
            &sampler_clamp,
        );

        // LDR 2D HUD pipeline (composited on top of tonemapped/anti-aliased output, 1x, no depth)
        let hud_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("farust_hud_shader"),
            source: wgpu::ShaderSource::Wgsl(HUD_SHADER_WGSL.into()),
        });
        let hud_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud_pipeline"),
            layout: Some(&decal_pl),
            vertex: wgpu::VertexState {
                module: &hud_shader,
                entry_point: Some("vs_main"),
                buffers: &[hud_vertex_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &hud_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
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

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buf"),
            contents: bytemuck::cast_slice(&scene.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index_buf"),
            contents: bytemuck::cast_slice(&scene.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let static_overlay = if !scene.overlay_vertices.is_empty() && !scene.overlay_indices.is_empty()
        {
            let ov_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("static_overlay_vb"),
                contents: bytemuck::cast_slice(&scene.overlay_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ov_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("static_overlay_ib"),
                contents: bytemuck::cast_slice(&scene.overlay_indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            Some((ov_vb, ov_ib, scene.overlay_indices.len() as u32))
        } else {
            None
        };

        // Upload terrain decals
        let mut decal_batches = Vec::with_capacity(scene.decal_batches.len());
        for db in &scene.decal_batches {
            if db.vertices.is_empty() || db.indices.is_empty() {
                continue;
            }
            let Some(ref img) = db.albedo else {
                continue;
            };
            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&db.label),
                contents: bytemuck::cast_slice(&db.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&db.label),
                contents: bytemuck::cast_slice(&db.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            let bg = create_rgba_bind_group(
                device,
                queue,
                &bind_group_layout,
                &uniform_buf,
                &sampler_clamp,
                img,
                &db.label,
            );
            decal_batches.push((vb, ib, db.indices.len() as u32, bg));
        }

        // Upload static environment prop batches (with 8x8 XZ chunk AABBs for frustum culling)
        let mut prop_batches = Vec::with_capacity(scene.prop_batches.len());
        for pb in &scene.prop_batches {
            if pb.instances.is_empty() || pb.indices.is_empty() {
                continue;
            }
            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&pb.label),
                contents: bytemuck::cast_slice(&pb.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&pb.label),
                contents: bytemuck::cast_slice(&pb.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            let bg = match pb.albedo {
                Some(ref img) => create_material_bind_group(
                    device,
                    queue,
                    &mat_bgl,
                    &uniform_buf,
                    &sampler_repeat,
                    img,
                    None,
                    &default_spec_glow_view,
                    &pb.label,
                ),
                None => overlay_mat_bind_group.clone(),
            };
            let inst_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&pb.label),
                contents: bytemuck::cast_slice(&pb.instances),
                usage: wgpu::BufferUsages::VERTEX,
            });
            prop_batches.push((
                vb,
                ib,
                pb.indices.len() as u32,
                bg,
                inst_b,
                pb.instances.len() as u32,
                pb.bounds_min,
                pb.bounds_max,
            ));
        }

        let (sh_verts, sh_indices) = build_shield_hemisphere_mesh();
        let sh_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shield_dome_vb"),
            contents: bytemuck::cast_slice(&sh_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sh_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shield_dome_ib"),
            contents: bytemuck::cast_slice(&sh_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let shield_mesh = (sh_vb, sh_ib, sh_indices.len() as u32);

        Self {
            scene_pipes,
            shadow_pipeline,
            hud_pipeline,
            bright_pipeline,
            downsample_pipeline,
            upsample_pipeline,
            tonemap_direct_pipeline,
            tonemap_fxaa_prep_pipeline,
            fxaa_pipeline,
            mat_pl,
            decal_pl,
            sky_pl,
            terrain_pl,
            water_pl,
            inst_pl,
            uniform_only_pl,
            vfx_pl,
            scene_shader,
            decal_shader,
            sky_shader,
            terrain_shader,
            water_shader,
            inst_shader,
            shield_shader,
            vfx_shader,
            bind_group_layout,
            mat_bgl,
            default_spec_glow_view,
            bind_group,
            overlay_mat_bind_group,
            overlay_bind_group,
            minimap_bind_group,
            hud_icons_bind_group,
            hud_font_bind_group,
            strat_icons_bind_group,
            uniform_only_bg,
            vfx_bgl,
            vfx_bind_group,
            post_single_bgl,
            tonemap_bgl,
            post_uniform_buf,
            post,
            target_is_srgb,
            msaa4_supported,
            bloom_enabled,
            active_aa,
            gamma,
            bloom_strength,
            bone_bgl,
            bone_buf,
            bone_buf_cap,
            bone_bind_group,
            fow_tex,
            shadow_view,
            shadow_bind_group,
            shadow_bgl,
            shadow_size,
            shadows_enabled,
            shadow_params,
            sky_vb,
            sky_bind_group,
            sky_fallback_bg,
            sampler_clamp,
            sampler_repeat,
            uniform_buf,
            vertex_buf,
            index_buf,
            index_count: scene.indices.len() as u32,
            terrain_bind_group,
            water_draw,
            decal_batches,
            dyn_decal_bgs: Vec::new(),
            dyn_decal_geo: Vec::new(),
            dyn_decal_rev: u64::MAX,
            static_overlay,
            prop_batches,
            unit_batches: Vec::new(),
            shield_mesh,
            shield_instances: None,
            vfx_draw: None,
            dynamic: None,
            hud_strat_icons: None,
            hud_minimap: None,
            hud_bg: None,
            hud_icons: None,
            hud_text: None,
            hud_ttf: None,
            lighting,
            map_size,
            water_levels,
            last_view_proj: uniforms.view_proj,
            last_shadow_view_proj: uniforms.shadow_view_proj,
            start_time,
            queue: queue.clone(),
            draw_stats: std::cell::Cell::new(DrawStats::default()),
            gpu_timer: None,
        }
    }

    /// Counters recorded by the last [`Self::encode_draw`].
    pub(crate) fn draw_stats(&self) -> DrawStats {
        self.draw_stats.get()
    }

    /// Enables render-pass timestamp queries when the device has `TIMESTAMP_QUERY`.
    /// Returns whether GPU timing is active.
    pub(crate) fn enable_gpu_timer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        if self.gpu_timer.is_none() {
            self.gpu_timer = GpuTimer::new(device, queue);
        }
        self.gpu_timer.is_some()
    }

    /// Call right after `queue.submit` of a frame encoded with [`Self::encode_draw`]: requests
    /// the timestamp readback when this frame wrote queries.
    pub(crate) fn gpu_timer_after_submit(&self) {
        let Some(t) = &self.gpu_timer else {
            return;
        };
        use std::sync::atomic::Ordering;
        if t.state.compare_exchange(1, 2, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            let st = Arc::clone(&t.state);
            t.readback.slice(..).map_async(wgpu::MapMode::Read, move |res| {
                st.store(if res.is_ok() { 3 } else { 0 }, Ordering::Release);
            });
        }
    }

    /// Collects a finished timestamp readback (after a `device.poll`). Returns
    /// `(shadow_ms, main_ms)` of the most recent timed frame and feeds the profiler.
    pub(crate) fn poll_gpu_timer(&self) -> Option<(f64, f64)> {
        let t = self.gpu_timer.as_ref()?;
        use std::sync::atomic::Ordering;
        if t.state.load(Ordering::Acquire) == 3 {
            let ts: [u64; 6] = {
                let data = t.readback.slice(..).get_mapped_range();
                let mut ts = [0u64; 6];
                for (i, &c) in data.as_chunks::<8>().0.iter().take(6).enumerate() {
                    ts[i] = u64::from_le_bytes(c);
                }
                ts
            };
            t.readback.unmap();
            let ms = |a: u64, b: u64| b.saturating_sub(a) as f64 * f64::from(t.period_ns) / 1e6;
            let shadow = if t.shadow_written.get() { ms(ts[0], ts[1]) } else { 0.0 };
            let scene = ms(ts[2], ts[3]);
            let post = ms(ts[4], ts[5]);
            t.last_ms.set(Some((shadow, scene + post)));
            crate::profiler::record_ms("gpu.shadow_pass", shadow);
            crate::profiler::record_ms("gpu.main_pass", scene);
            crate::profiler::record_ms("gpu.post_pass", post);
            t.state.store(0, Ordering::Release);
        }
        t.last_ms.get()
    }

    /// Cloned handle of the queue these resources were created with (render-glue uploads).
    pub(crate) fn queue(&self) -> wgpu::Queue {
        self.queue.clone()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_mesh_template(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[GpuVertex],
        indices: &[u32],
        albedo: Option<&RgbaImage>,
        spec_glow: Option<&SpecGlowMap>,
        label: &str,
    ) -> GpuMeshTemplate {
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let bind_group = match albedo {
            Some(img) => create_material_bind_group(
                device,
                queue,
                &self.mat_bgl,
                &self.uniform_buf,
                &self.sampler_repeat,
                img,
                spec_glow,
                &self.default_spec_glow_view,
                label,
            ),
            None => self.overlay_mat_bind_group.clone(),
        };
        GpuMeshTemplate {
            vertex_buf,
            index_buf,
            index_count: indices.len() as u32,
            bind_group,
        }
    }

    pub(crate) fn set_unit_batches(
        &mut self,
        device: &wgpu::Device,
        batches: Vec<(&GpuMeshTemplate, &[GpuUnitInstance])>,
    ) {
        // Recycle last frame's instance buffers (largest first) instead of allocating new
        // ones every frame; contents go through `queue.write_buffer`.
        let mut free: Vec<wgpu::Buffer> = self.unit_batches.drain(..).map(|b| b.4).collect();
        free.sort_by_key(|b| std::cmp::Reverse(b.size()));
        for (tpl, instances) in batches {
            if instances.is_empty() || tpl.index_count == 0 {
                continue;
            }
            let bytes: &[u8] = bytemuck::cast_slice(instances);
            let need = bytes.len() as u64;
            let inst_buf = match free.iter().rposition(|b| b.size() >= need) {
                Some(i) => {
                    let b = free.swap_remove(i);
                    free.sort_by_key(|b| std::cmp::Reverse(b.size()));
                    self.queue.write_buffer(&b, 0, bytes);
                    b
                }
                None => {
                    let cap = need.next_power_of_two().max(4096);
                    let b = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("unit_inst_buf"),
                        size: cap,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    self.queue.write_buffer(&b, 0, bytes);
                    b
                }
            };
            self.unit_batches.push((
                tpl.vertex_buf.clone(),
                tpl.index_buf.clone(),
                tpl.index_count,
                tpl.bind_group.clone(),
                inst_buf,
                instances.len() as u32,
            ));
        }
    }

    pub(crate) fn set_minimap_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &RgbaImage,
    ) {
        self.minimap_bind_group = create_rgba_bind_group(
            device,
            queue,
            &self.bind_group_layout,
            &self.uniform_buf,
            &self.sampler_clamp,
            img,
            "minimap_tex",
        );
    }

    pub(crate) fn set_hud_icons_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &RgbaImage,
    ) {
        self.hud_icons_bind_group = create_rgba_bind_group(
            device,
            queue,
            &self.bind_group_layout,
            &self.uniform_buf,
            &self.sampler_clamp,
            img,
            "hud_icons_tex",
        );
    }

    pub(crate) fn set_hud_font_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &RgbaImage,
    ) {
        self.hud_font_bind_group = create_rgba_bind_group(
            device,
            queue,
            &self.bind_group_layout,
            &self.uniform_buf,
            &self.sampler_clamp,
            img,
            "hud_font_tex",
        );
    }

    pub(crate) fn set_hud_ttf(
        &mut self,
        device: &wgpu::Device,
        verts: &[HudVertex],
        indices: &[u32],
    ) {
        self.hud_ttf = if verts.is_empty() || indices.is_empty() {
            None
        } else {
            Some((
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("hud_ttf"),
                    contents: bytemuck::cast_slice(verts),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("hud_ttf"),
                    contents: bytemuck::cast_slice(indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
                indices.len() as u32,
            ))
        };
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_hud(
        &mut self,
        device: &wgpu::Device,
        minimap_v: &[HudVertex],
        minimap_i: &[u32],
        bg_v: &[HudVertex],
        bg_i: &[u32],
        icons_v: &[HudVertex],
        icons_i: &[u32],
        text_v: &[HudVertex],
        text_i: &[u32],
    ) {
        let make = |v: &[HudVertex], i: &[u32], lbl: &str| {
            if v.is_empty() || i.is_empty() {
                None
            } else {
                Some((
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(lbl),
                        contents: bytemuck::cast_slice(v),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(lbl),
                        contents: bytemuck::cast_slice(i),
                        usage: wgpu::BufferUsages::INDEX,
                    }),
                    i.len() as u32,
                ))
            }
        };
        self.hud_minimap = make(minimap_v, minimap_i, "hud_minimap");
        self.hud_bg = make(bg_v, bg_i, "hud_bg");
        self.hud_icons = make(icons_v, icons_i, "hud_icons");
        self.hud_text = make(text_v, text_i, "hud_text");
    }

    /// Replaces the main terrain vertex/index buffers after live heightmap edits.
    pub(crate) fn update_terrain_mesh(
        &mut self,
        device: &wgpu::Device,
        vertices: &[GpuVertex],
        indices: &[u32],
    ) {
        if vertices.is_empty() || indices.is_empty() {
            self.index_count = 0;
            return;
        }
        self.vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene_vb_edited"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene_ib_edited"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.index_count = indices.len() as u32;
    }

    /// Replaces the static 3D overlay (marker beacons / legacy water quad).
    pub(crate) fn update_static_overlay(
        &mut self,
        device: &wgpu::Device,
        vertices: &[GpuVertex],
        indices: &[u32],
    ) {
        if vertices.is_empty() || indices.is_empty() {
            self.static_overlay = None;
            return;
        }
        let ov_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("static_overlay_vb_edited"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ov_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("static_overlay_ib_edited"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.static_overlay = Some((ov_vb, ov_ib, indices.len() as u32));
    }

    /// Replaces the static prop batches after placing or deleting props in the map editor.
    /// Phase 41: textures for the dynamic decal kinds (missing ones are never drawn).
    pub(crate) fn set_dynamic_decal_textures(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, images: &[Option<RgbaImage>]) {
        self.dyn_decal_bgs = images
            .iter()
            .enumerate()
            .map(|(k, img)| {
                img.as_ref().map(|img| {
                    create_rgba_bind_group(device, queue, &self.bind_group_layout, &self.uniform_buf, &self.sampler_clamp, img, &format!("dyn_decal_{k}"))
                })
            })
            .collect();
        self.dyn_decal_geo = (0..images.len()).map(|_| None).collect();
        self.dyn_decal_rev = u64::MAX;
    }

    /// Phase 41: replaces the geometry of dynamic decal `kind`.
    pub(crate) fn set_dynamic_decals(&mut self, device: &wgpu::Device, kind: usize, vertices: &[crate::scene::GpuVertex], indices: &[u32]) {
        let Some(slot) = self.dyn_decal_geo.get_mut(kind) else { return };
        *slot = (!indices.is_empty()).then(|| {
            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dyn_decal_vb"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dyn_decal_ib"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            (vb, ib, indices.len() as u32)
        });
    }

    pub(crate) fn update_prop_batches(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prop_batches: &[crate::scene::PropMeshBatch],
    ) {
        let mut uploaded = Vec::with_capacity(prop_batches.len());
        for pb in prop_batches {
            if pb.instances.is_empty() || pb.indices.is_empty() {
                continue;
            }
            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&pb.label),
                contents: bytemuck::cast_slice(&pb.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&pb.label),
                contents: bytemuck::cast_slice(&pb.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            let bg = match pb.albedo {
                Some(ref img) => create_material_bind_group(
                    device,
                    queue,
                    &self.mat_bgl,
                    &self.uniform_buf,
                    &self.sampler_repeat,
                    img,
                    None,
                    &self.default_spec_glow_view,
                    &pb.label,
                ),
                None => self.overlay_mat_bind_group.clone(),
            };
            let inst_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&pb.label),
                contents: bytemuck::cast_slice(&pb.instances),
                usage: wgpu::BufferUsages::VERTEX,
            });
            uploaded.push((
                vb,
                ib,
                pb.indices.len() as u32,
                bg,
                inst_b,
                pb.instances.len() as u32,
                pb.bounds_min,
                pb.bounds_max,
            ));
        }
        self.prop_batches = uploaded;
    }

    /// Replace the per-frame dynamic geometry (empty input clears it).
    pub(crate) fn set_dynamic(&mut self, device: &wgpu::Device, vertices: &[GpuVertex], indices: &[u32]) {
        if vertices.is_empty() || indices.is_empty() {
            self.dynamic = None;
            return;
        }
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dyn_vertex_buf"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dyn_index_buf"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.dynamic = Some((vb, ib, indices.len() as u32));
    }

    pub(crate) fn set_bone_palette(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mats: &[[[f32; 4]; 4]],
    ) {
        if mats.is_empty() {
            return;
        }
        if mats.len() > self.bone_buf_cap {
            let new_cap = mats.len().next_power_of_two().max(256);
            let mut padded = Vec::with_capacity(new_cap);
            padded.extend_from_slice(mats);
            let id_m: [[f32; 4]; 4] = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            padded.resize(new_cap, id_m);
            let new_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bone_palette_buf"),
                contents: bytemuck::cast_slice(&padded),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            self.bone_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bone_bg"),
                layout: &self.bone_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: new_buf.as_entire_binding(),
                }],
            });
            self.bone_buf = new_buf;
            self.bone_buf_cap = new_cap;
        } else {
            queue.write_buffer(&self.bone_buf, 0, bytemuck::cast_slice(mats));
        }
    }

    pub(crate) fn update_fow_texture(&self, queue: &wgpu::Queue, rgba_128x128: &[u8]) {
        if rgba_128x128.len() != 128 * 128 * 4 {
            return;
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.fow_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_128x128,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(128 * 4),
                rows_per_image: Some(128),
            },
            wgpu::Extent3d {
                width: 128,
                height: 128,
                depth_or_array_layers: 1,
            },
        );
    }

    pub(crate) fn set_shield_instances(
        &mut self,
        device: &wgpu::Device,
        instances: &[GpuUnitInstance],
    ) {
        if instances.is_empty() {
            self.shield_instances = None;
            return;
        }
        let inst_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shield_inst_buf"),
            contents: bytemuck::cast_slice(instances),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.shield_instances = Some((inst_buf, instances.len() as u32));
    }

    pub(crate) fn set_vfx_texture_array(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        layers: &[RgbaImage],
    ) {
        if layers.is_empty() || width == 0 || height == 0 {
            return;
        }
        self.vfx_bind_group = create_vfx_array_bind_group(
            device,
            queue,
            &self.vfx_bgl,
            &self.uniform_buf,
            &self.sampler_clamp,
            &self.sampler_repeat,
            width,
            height,
            layers,
        );
    }

    pub(crate) fn set_vfx(
        &mut self,
        device: &wgpu::Device,
        vertices: &[VfxVertex],
        indices: &[u32],
    ) {
        if vertices.is_empty() || indices.is_empty() {
            self.vfx_draw = None;
            return;
        }
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vfx_vb"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vfx_ib"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.vfx_draw = Some((vb, ib, indices.len() as u32));
    }

    pub(crate) fn set_strategic_icons_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &RgbaImage,
    ) {
        self.strat_icons_bind_group = create_rgba_bind_group(
            device,
            queue,
            &self.bind_group_layout,
            &self.uniform_buf,
            &self.sampler_clamp,
            img,
            "strat_icons_tex",
        );
    }

    pub(crate) fn set_hud_strategic_icons(
        &mut self,
        device: &wgpu::Device,
        verts: &[HudVertex],
        indices: &[u32],
    ) {
        if verts.is_empty() || indices.is_empty() {
            self.hud_strat_icons = None;
            return;
        }
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hud_strat_icons_vb"),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hud_strat_icons_ib"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.hud_strat_icons = Some((vb, ib, indices.len() as u32));
    }

    /// Applies a new shadow quality, rebuilding the shadow target when the size
    /// or enabled state changes. Returns `true` when anything was rebuilt; call
    /// [`Self::update_camera`] afterwards so the new PCF params reach the GPU.
    pub(crate) fn set_shadow_quality(
        &mut self,
        device: &wgpu::Device,
        quality: ShadowQuality,
    ) -> bool {
        let size = quality.map_size().max(1);
        let enabled = quality.map_size() > 0;
        if size == self.shadow_size && enabled == self.shadows_enabled {
            return false;
        }
        let (view, bg) = create_shadow_target(device, &self.shadow_bgl, size);
        self.shadow_view = view;
        self.shadow_bind_group = bg;
        self.shadow_size = size;
        self.shadows_enabled = enabled;
        self.shadow_params = shadow_uniform_params(quality);
        true
    }

    /// Resizes the internal HDR, depth, bloom mip chain, and optional FXAA targets.
    pub(crate) fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.post.width == w && self.post.height == h {
            return;
        }
        self.post = create_post_targets(
            device,
            w,
            h,
            self.active_aa,
            &self.post_single_bgl,
            &self.tonemap_bgl,
            &self.post_uniform_buf,
            &self.sampler_clamp,
        );
    }

    /// Applies any changed Phase 42 render settings (`bloom`, `aa`, `gamma`) from the lock-free
    /// atomic cache. Rebuilds 3D scene pipelines only when MSAA sample count changes (`1` <-> `4`).
    pub(crate) fn sync_render_settings(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (bloom_enabled, req_aa, gamma) = crate::settings::render_post();
        let next_aa = resolve_aa_mode(req_aa, self.msaa4_supported);

        if bloom_enabled != self.bloom_enabled || (gamma - self.gamma).abs() > 1e-4 {
            self.bloom_enabled = bloom_enabled;
            self.gamma = gamma;
            let pu = make_post_uniforms(&self.lighting, bloom_enabled, gamma, self.target_is_srgb);
            self.bloom_strength = pu.params0[1];
            queue.write_buffer(&self.post_uniform_buf, 0, bytemuck::bytes_of(&pu));
        }

        if next_aa != self.active_aa {
            let old_samples = self.active_aa.sample_count();
            let new_samples = next_aa.sample_count();
            self.active_aa = next_aa;
            if old_samples != new_samples {
                self.scene_pipes = build_scene_pipelines(
                    device,
                    new_samples,
                    &self.mat_pl,
                    &self.decal_pl,
                    &self.sky_pl,
                    &self.terrain_pl,
                    &self.water_pl,
                    &self.inst_pl,
                    &self.uniform_only_pl,
                    &self.vfx_pl,
                    &self.scene_shader,
                    &self.decal_shader,
                    &self.sky_shader,
                    &self.terrain_shader,
                    &self.water_shader,
                    &self.inst_shader,
                    &self.shield_shader,
                    &self.vfx_shader,
                );
            }
            self.post = create_post_targets(
                device,
                self.post.width,
                self.post.height,
                next_aa,
                &self.post_single_bgl,
                &self.tonemap_bgl,
                &self.post_uniform_buf,
                &self.sampler_clamp,
            );
        }
    }

    pub(crate) fn update_camera(&mut self, queue: &wgpu::Queue, camera: &OrbitCamera) {
        let t = self.start_time.elapsed().as_secs_f32();
        let uniforms = make_uniforms(
            camera,
            &self.lighting,
            self.map_size,
            self.water_levels,
            t,
            self.shadow_params,
        );
        self.last_view_proj = uniforms.view_proj;
        self.last_shadow_view_proj = uniforms.shadow_view_proj;
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    pub(crate) fn encode_draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        clear_color: [f64; 4],
    ) {
        let mut st = DrawStats::default();
        let timer = self.gpu_timer.as_ref().filter(|t| t.idle());
        let shadow_ts = timer.map(|t| wgpu::RenderPassTimestampWrites {
            query_set: &t.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        });
        let main_ts = timer.map(|t| wgpu::RenderPassTimestampWrites {
            query_set: &t.query_set,
            beginning_of_pass_write_index: Some(2),
            end_of_pass_write_index: Some(3),
        });
        if let Some(t) = timer {
            t.shadow_written.set(false);
        }

        // Pass 1: Directional Shadow Map (props + units), skipped when shadows are off.
        if self.shadows_enabled
            && (!self.prop_batches.is_empty() || !self.unit_batches.is_empty())
        {
            let mut spass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: shadow_ts,
                occlusion_query_set: None,
            });
            if let Some(t) = timer {
                t.shadow_written.set(true);
            }
            spass.set_pipeline(&self.shadow_pipeline);
            spass.set_bind_group(0, &self.uniform_only_bg, &[]);
            spass.set_bind_group(1, &self.bone_bind_group, &[]);
            if !self.prop_batches.is_empty() {
                let shadow_planes = extract_frustum_planes(&self.last_shadow_view_proj);
                for (vb, ib, idx_n, _, inst_b, inst_n, bmin, bmax) in &self.prop_batches {
                    if !aabb_in_frustum(&shadow_planes, *bmin, *bmax) {
                        continue;
                    }
                    spass.set_vertex_buffer(0, vb.slice(..));
                    spass.set_vertex_buffer(1, inst_b.slice(..));
                    spass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    { st.add(*idx_n, *inst_n); spass.draw_indexed(0..*idx_n, 0, 0..*inst_n); }
                }
            }
            for (vb, ib, idx_n, _, inst_b, inst_n) in &self.unit_batches {
                spass.set_vertex_buffer(0, vb.slice(..));
                spass.set_vertex_buffer(1, inst_b.slice(..));
                spass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                { st.add(*idx_n, *inst_n); spass.draw_indexed(0..*idx_n, 0, 0..*inst_n); }
            }
        }

        // Pass 2: Main 3D HDR Scene Pass (Rgba16Float, MSAA4 or 1x)
        let (hdr_color_view, hdr_resolve_target) = match &self.post.hdr_msaa_view {
            Some(msaa_v) => (msaa_v, Some(&self.post.hdr_resolve_view)),
            None => (&self.post.hdr_resolve_view, None),
        };
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hdr_scene_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: hdr_color_view,
                resolve_target: hdr_resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: clear_color[0],
                        g: clear_color[1],
                        b: clear_color[2],
                        a: clear_color[3],
                    }),
                    store: if hdr_resolve_target.is_some() {
                        wgpu::StoreOp::Discard
                    } else {
                        wgpu::StoreOp::Store
                    },
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.post.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: main_ts,
            occlusion_query_set: None,
        });

        // Skybox cube (only when loaded)
        if let Some(ref sky_bg) = self.sky_bind_group {
            rpass.set_pipeline(&self.scene_pipes.sky_pipeline);
            rpass.set_bind_group(0, &self.uniform_only_bg, &[]);
            rpass.set_bind_group(1, sky_bg, &[]);
            rpass.set_vertex_buffer(0, self.sky_vb.slice(..));
            st.add(36, 1);
            rpass.draw(0..36, 0..1);
        }

        // Terrain (multi-stratum splatting + FoW + shadow map, or single-texture fallback)
        if self.index_count > 0 {
            if let Some(ref t_bg) = self.terrain_bind_group {
                rpass.set_pipeline(&self.scene_pipes.terrain_pipeline);
                rpass.set_bind_group(0, t_bg, &[]);
                rpass.set_bind_group(1, &self.shadow_bind_group, &[]);
            } else {
                rpass.set_pipeline(&self.scene_pipes.pipeline);
                rpass.set_bind_group(0, &self.bind_group, &[]);
            }
            rpass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            rpass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            { st.add(self.index_count, 1); rpass.draw_indexed(0..self.index_count, 0, 0..1); }
        }

        // Terrain Decals (draped albedo craters, scorch marks, roads)
        if !self.decal_batches.is_empty() {
            rpass.set_pipeline(&self.scene_pipes.decal_pipeline);
            for (vb, ib, idx_n, bg) in &self.decal_batches {
                rpass.set_bind_group(0, bg, &[]);
                rpass.set_vertex_buffer(0, vb.slice(..));
                rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                { st.add(*idx_n, 1); rpass.draw_indexed(0..*idx_n, 0, 0..1); }
            }
        }
        // Phase 41: dynamic marks (footprints, treads, craters, scorch) over the map decals.
        if self.dyn_decal_geo.iter().any(Option::is_some) {
            rpass.set_pipeline(&self.scene_pipes.decal_pipeline);
            for k in [3usize, 2, 1, 0] {
                if let (Some(Some((vb, ib, n))), Some(Some(bg))) = (self.dyn_decal_geo.get(k), self.dyn_decal_bgs.get(k)) {
                    rpass.set_bind_group(0, bg, &[]);
                    rpass.set_vertex_buffer(0, vb.slice(..));
                    rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    { st.add(*n, 1); rpass.draw_indexed(0..*n, 0, 0..1); }
                }
            }
        }

        // Instanced Props (frustum-culled 8x8 XZ chunks) + Units & Wrecks (GPU skinned + SpecGlow)
        if !self.prop_batches.is_empty() || !self.unit_batches.is_empty() {
            let planes = extract_frustum_planes(&self.last_view_proj);
            rpass.set_pipeline(&self.scene_pipes.instanced_pipeline);
            rpass.set_bind_group(1, &self.shadow_bind_group, &[]);
            rpass.set_bind_group(2, &self.bone_bind_group, &[]);
            for (vb, ib, idx_n, bg, inst_b, inst_n, bmin, bmax) in &self.prop_batches {
                if !aabb_in_frustum(&planes, *bmin, *bmax) {
                    continue;
                }
                rpass.set_bind_group(0, bg, &[]);
                rpass.set_vertex_buffer(0, vb.slice(..));
                rpass.set_vertex_buffer(1, inst_b.slice(..));
                rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                { st.add(*idx_n, *inst_n); rpass.draw_indexed(0..*idx_n, 0, 0..*inst_n); }
            }
            for (vb, ib, idx_n, bg, inst_b, inst_n) in &self.unit_batches {
                st.unit_batches += 1;
                st.unit_instances += u64::from(*inst_n);
                rpass.set_bind_group(0, bg, &[]);
                rpass.set_vertex_buffer(0, vb.slice(..));
                rpass.set_vertex_buffer(1, inst_b.slice(..));
                rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                { st.add(*idx_n, *inst_n); rpass.draw_indexed(0..*idx_n, 0, 0..*inst_n); }
            }
        }

        // Transparent Water Surface with Wave Normals & Sky Reflections
        if let Some((wvb, wib, wn, wbg)) = &self.water_draw {
            let sky_bg = self.sky_bind_group.as_ref().unwrap_or(&self.sky_fallback_bg);
            rpass.set_pipeline(&self.scene_pipes.water_pipeline);
            rpass.set_bind_group(0, wbg, &[]);
            rpass.set_bind_group(1, sky_bg, &[]);
            rpass.set_vertex_buffer(0, wvb.slice(..));
            rpass.set_index_buffer(wib.slice(..), wgpu::IndexFormat::Uint32);
            { st.add(*wn, 1); rpass.draw_indexed(0..*wn, 0, 0..1); }
        }

        // Static & Dynamic 3D Overlays (bars, mass pads, selection rings, tracers)
        if self.static_overlay.is_some() || self.dynamic.is_some() {
            rpass.set_pipeline(&self.scene_pipes.pipeline);
            if let Some((vb, ib, n)) = &self.static_overlay {
                rpass.set_bind_group(0, &self.overlay_mat_bind_group, &[]);
                rpass.set_vertex_buffer(0, vb.slice(..));
                rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                { st.add(*n, 1); rpass.draw_indexed(0..*n, 0, 0..1); }
            }
            if let Some((vb, ib, n)) = &self.dynamic {
                rpass.set_bind_group(0, &self.overlay_mat_bind_group, &[]);
                rpass.set_vertex_buffer(0, vb.slice(..));
                rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                { st.add(*n, 1); rpass.draw_indexed(0..*n, 0, 0..1); }
            }
        }

        // Translucent Bubble Shield Domes
        if let Some((inst_b, inst_n)) = &self.shield_instances {
            rpass.set_pipeline(&self.scene_pipes.shield_pipeline);
            rpass.set_bind_group(0, &self.uniform_only_bg, &[]);
            rpass.set_vertex_buffer(0, self.shield_mesh.0.slice(..));
            rpass.set_vertex_buffer(1, inst_b.slice(..));
            rpass.set_index_buffer(self.shield_mesh.1.slice(..), wgpu::IndexFormat::Uint32);
            { st.add(self.shield_mesh.2, *inst_n); rpass.draw_indexed(0..self.shield_mesh.2, 0, 0..*inst_n); }
        }

        // GPU Billboard Particle VFX
        if let Some((vb, ib, n)) = &self.vfx_draw {
            rpass.set_pipeline(&self.scene_pipes.vfx_pipeline);
            rpass.set_bind_group(0, &self.vfx_bind_group, &[]);
            rpass.set_vertex_buffer(0, vb.slice(..));
            rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            { st.add(*n, 1); rpass.draw_indexed(0..*n, 0, 0..1); }
        }
        drop(rpass);

        // Pass 3: Bloom bright-pass -> downsampled mip chain -> additive tent upsample
        let run_bloom = self.bloom_strength > 0.0 && !self.post.bloom_views.is_empty();
        let has_hud = self.hud_strat_icons.is_some()
            || self.hud_minimap.is_some()
            || self.hud_bg.is_some()
            || self.hud_icons.is_some()
            || self.hud_text.is_some()
            || self.hud_ttf.is_some();
        let mut post_begin_written = false;
        let post_ts = |is_begin: bool, is_end: bool| -> Option<wgpu::RenderPassTimestampWrites<'_>> {
            if !is_begin && !is_end {
                return None;
            }
            timer.map(|t| wgpu::RenderPassTimestampWrites {
                query_set: &t.query_set,
                beginning_of_pass_write_index: is_begin.then_some(4),
                end_of_pass_write_index: is_end.then_some(5),
            })
        };

        if run_bloom {
            let mips = self.post.bloom_views.len();
            // 3a: Bright-pass + Karis downsample into mip 0
            {
                let mut bpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom_bright_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.post.bloom_views[0],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: post_ts(true, false),
                    occlusion_query_set: None,
                });
                post_begin_written = true;
                bpass.set_pipeline(&self.bright_pipeline);
                bpass.set_bind_group(0, &self.post.bloom_src_bgs[0], &[]);
                bpass.draw(0..3, 0..1);
            }
            // 3b: Downsample mip 0 -> 1 -> ... -> N-1
            for i in 1..mips {
                let mut dpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom_downsample_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.post.bloom_views[i],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                dpass.set_pipeline(&self.downsample_pipeline);
                dpass.set_bind_group(0, &self.post.bloom_src_bgs[i], &[]);
                dpass.draw(0..3, 0..1);
            }
            // 3c: Additive 3x3 tent upsample N-1 -> ... -> 0
            for i in (1..mips).rev() {
                let mut upass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom_upsample_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.post.bloom_views[i - 1],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                upass.set_pipeline(&self.upsample_pipeline);
                upass.set_bind_group(0, &self.post.bloom_src_bgs[i + 1], &[]);
                upass.draw(0..3, 0..1);
            }
        }

        // Pass 4: Filmic/ACES Tonemap + Bloom Composite + Gamma (+ optional FXAA)
        if self.active_aa == AaMode::Fxaa
            && let Some((ldr_view, fxaa_bg)) = &self.post.fxaa
        {
            {
                let mut tpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("tonemap_fxaa_prep_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: ldr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: post_ts(!post_begin_written, false),
                    occlusion_query_set: None,
                });
                tpass.set_pipeline(&self.tonemap_fxaa_prep_pipeline);
                tpass.set_bind_group(0, &self.post.tonemap_bg, &[]);
                tpass.draw(0..3, 0..1);
            }
            {
                let mut fpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("fxaa_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: post_ts(false, !has_hud),
                    occlusion_query_set: None,
                });
                fpass.set_pipeline(&self.fxaa_pipeline);
                fpass.set_bind_group(0, fxaa_bg, &[]);
                fpass.draw(0..3, 0..1);
            }
        } else {
            let mut tpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tonemap_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: post_ts(!post_begin_written, !has_hud),
                occlusion_query_set: None,
            });
            tpass.set_pipeline(&self.tonemap_direct_pipeline);
            tpass.set_bind_group(0, &self.post.tonemap_bg, &[]);
            tpass.draw(0..3, 0..1);
        }

        // Pass 5: LDR 2D HUD Pass (composited crisp on top after tonemapping & FXAA)
        if has_hud {
            let mut hpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ldr_hud_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: post_ts(false, true),
                occlusion_query_set: None,
            });
            hpass.set_pipeline(&self.hud_pipeline);
            if let Some((vb, ib, n)) = &self.hud_strat_icons {
                hpass.set_bind_group(0, &self.strat_icons_bind_group, &[]);
                hpass.set_vertex_buffer(0, vb.slice(..));
                hpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                { st.add(*n, 1); hpass.draw_indexed(0..*n, 0, 0..1); }
            }
            if let Some((vb, ib, n)) = &self.hud_minimap {
                hpass.set_bind_group(0, &self.minimap_bind_group, &[]);
                hpass.set_vertex_buffer(0, vb.slice(..));
                hpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                { st.add(*n, 1); hpass.draw_indexed(0..*n, 0, 0..1); }
            }
            if let Some((vb, ib, n)) = &self.hud_bg {
                hpass.set_bind_group(0, &self.overlay_bind_group, &[]);
                hpass.set_vertex_buffer(0, vb.slice(..));
                hpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                { st.add(*n, 1); hpass.draw_indexed(0..*n, 0, 0..1); }
            }
            if let Some((vb, ib, n)) = &self.hud_icons {
                hpass.set_bind_group(0, &self.hud_icons_bind_group, &[]);
                hpass.set_vertex_buffer(0, vb.slice(..));
                hpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                { st.add(*n, 1); hpass.draw_indexed(0..*n, 0, 0..1); }
            }
            if let Some((vb, ib, n)) = &self.hud_text {
                hpass.set_bind_group(0, &self.overlay_bind_group, &[]);
                hpass.set_vertex_buffer(0, vb.slice(..));
                hpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                { st.add(*n, 1); hpass.draw_indexed(0..*n, 0, 0..1); }
            }
            if let Some((vb, ib, n)) = &self.hud_ttf {
                hpass.set_bind_group(0, &self.hud_font_bind_group, &[]);
                hpass.set_vertex_buffer(0, vb.slice(..));
                hpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                { st.add(*n, 1); hpass.draw_indexed(0..*n, 0, 0..1); }
            }
        }

        if let Some(t) = timer {
            encoder.resolve_query_set(&t.query_set, 0..6, &t.resolve_buf, 0);
            encoder.copy_buffer_to_buffer(&t.resolve_buf, 0, &t.readback, 0, 48);
            t.state.store(1, std::sync::atomic::Ordering::Release);
        }
        self.draw_stats.set(st);
    }
}

fn upload_terrain_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    uniform_buf: &wgpu::Buffer,
    sampler_clamp: &wgpu::Sampler,
    sampler_repeat: &wgpu::Sampler,
    fow_view: &wgpu::TextureView,
    ts: &TerrainSplatData,
) -> wgpu::BindGroup {
    let params = TerrainParamsUniform {
        alb_scales_0: [
            ts.albedo_scales[0],
            ts.albedo_scales[1],
            ts.albedo_scales[2],
            ts.albedo_scales[3],
        ],
        alb_scales_1: [
            ts.albedo_scales[4],
            ts.albedo_scales[5],
            ts.albedo_scales[6],
            ts.albedo_scales[7],
        ],
        alb_scales_2: [ts.albedo_scales[8], ts.albedo_scales[9], 0.0, 0.0],
        norm_scales_0: [
            ts.normal_scales[0],
            ts.normal_scales[1],
            ts.normal_scales[2],
            ts.normal_scales[3],
        ],
        norm_scales_1: [
            ts.normal_scales[4],
            ts.normal_scales[5],
            ts.normal_scales[6],
            ts.normal_scales[7],
        ],
        norm_scales_2: [ts.normal_scales[8], 0.0, 0.0, 0.0],
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("terrain_params_buf"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let tm0_view = upload_texture_2d(device, queue, &ts.tm0, wgpu::TextureFormat::Rgba8Unorm, "tm0");
    let tm1_view = upload_texture_2d(device, queue, &ts.tm1, wgpu::TextureFormat::Rgba8Unorm, "tm1");
    let alb_view = upload_texture_2d_array(
        device,
        queue,
        &ts.strata_albedo,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        "strata_albedo_array",
    );
    let norm_view = upload_texture_2d_array(
        device,
        queue,
        &ts.strata_normal,
        wgpu::TextureFormat::Rgba8Unorm,
        "strata_normal_array",
    );

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("terrain_splat_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&tm0_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&tm1_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&alb_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&norm_view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(sampler_clamp),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::Sampler(sampler_repeat),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(fow_view),
            },
        ],
    })
}

/// Unit shield sphere (full sphere: bubble domes intersect the terrain naturally).
fn build_shield_hemisphere_mesh() -> (Vec<GpuVertex>, Vec<u32>) {
    const LAT: usize = 24;
    const LON: usize = 32;
    let mut verts = Vec::with_capacity((LAT + 1) * (LON + 1));
    let mut indices = Vec::with_capacity(LAT * LON * 6);
    for iy in 0..=LAT {
        let phi = (iy as f32 / LAT as f32) * std::f32::consts::PI - std::f32::consts::FRAC_PI_2;
        let (sin_p, cos_p) = phi.sin_cos();
        for ix in 0..=LON {
            let theta = (ix as f32 / LON as f32) * std::f32::consts::TAU;
            let (sin_t, cos_t) = theta.sin_cos();
            let nx = cos_p * cos_t;
            let ny = sin_p;
            let nz = cos_p * sin_t;
            verts.push(GpuVertex {
                position: [nx, ny, nz],
                normal: [nx, ny, nz],
                uv: [ix as f32 / LON as f32, iy as f32 / LAT as f32],
                color: [1.0, 1.0, 1.0, 0.0],
            });
        }
    }
    let stride = (LON + 1) as u32;
    for iy in 0..LAT as u32 {
        for ix in 0..LON as u32 {
            let a = iy * stride + ix;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    (verts, indices)
}

fn extract_frustum_planes(vp: &[[f32; 4]; 4]) -> [[f32; 4]; 6] {
    let row = |r: usize| [vp[0][r], vp[1][r], vp[2][r], vp[3][r]];
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);
    let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
    let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
    [
        add(r3, r0), // Left
        sub(r3, r0), // Right
        add(r3, r1), // Bottom
        sub(r3, r1), // Top
        r2,          // Near (WebGPU [0, 1] depth)
        sub(r3, r2), // Far
    ]
}

/// View frustum with normalized planes for CPU sphere culling of unit instances.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Frustum {
    planes: [[f32; 4]; 6],
}

impl Frustum {
    pub(crate) fn from_view_proj(vp: &[[f32; 4]; 4]) -> Self {
        let mut planes = extract_frustum_planes(vp);
        for p in &mut planes {
            let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            if l > 1e-12 {
                for c in p.iter_mut() {
                    *c /= l;
                }
            }
        }
        Self { planes }
    }

    /// `true` unless the sphere lies completely outside one plane.
    #[inline]
    pub(crate) fn sphere_visible(&self, c: [f32; 3], r: f32) -> bool {
        self.planes
            .iter()
            .all(|p| p[0] * c[0] + p[1] * c[1] + p[2] * c[2] + p[3] >= -r)
    }
}

fn aabb_in_frustum(planes: &[[f32; 4]; 6], bmin: [f32; 3], bmax: [f32; 3]) -> bool {
    for p in planes {
        let px = if p[0] >= 0.0 { bmax[0] } else { bmin[0] };
        let py = if p[1] >= 0.0 { bmax[1] } else { bmin[1] };
        let pz = if p[2] >= 0.0 { bmax[2] } else { bmin[2] };
        if p[0] * px + p[1] * py + p[2] * pz + p[3] < 0.0 {
            return false;
        }
    }
    true
}

fn upload_water_draw(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    uniform_buf: &wgpu::Buffer,
    sampler_repeat: &wgpu::Sampler,
    sampler_clamp: &wgpu::Sampler,
    wd: &WaterGpuData,
) -> (wgpu::Buffer, wgpu::Buffer, u32, wgpu::BindGroup) {
    let wp = WaterParamsUniform {
        surface_color: [wd.surface_color[0], wd.surface_color[1], wd.surface_color[2], 1.0],
        fresnel: [wd.fresnel_bias, wd.fresnel_power, 0.25, 0.05],
    };
    let wp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("water_params_buf"),
        contents: bytemuck::bytes_of(&wp),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let waves_view = upload_texture_2d(
        device,
        queue,
        &wd.waves_normal,
        wgpu::TextureFormat::Rgba8Unorm,
        "water_waves",
    );
    let ramp_view = upload_texture_2d(
        device,
        queue,
        &wd.water_ramp,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        "water_ramp",
    );
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("water_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wp_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&waves_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&ramp_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(sampler_repeat),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(sampler_clamp),
            },
        ],
    });
    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("water_vb"),
        contents: bytemuck::cast_slice(&wd.vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("water_ib"),
        contents: bytemuck::cast_slice(&wd.indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vb, ib, wd.indices.len() as u32, bg)
}

fn norm3_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let l2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if l2.is_finite() && l2 > 1e-8 {
        let inv = 1.0 / l2.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        fallback
    }
}

fn cross3_v(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3_v(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn mul_m4(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0_f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] =
                a[0][r] * b[c][0] + a[1][r] * b[c][1] + a[2][r] * b[c][2] + a[3][r] * b[c][3];
        }
    }
    out
}

fn make_shadow_view_proj(target: [f32; 3], sun_dir: [f32; 3], distance: f32) -> [[f32; 4]; 4] {
    let sd = norm3_or(sun_dir, [0.46, 0.71, 0.53]);
    let eye = [
        target[0] + sd[0] * 350.0,
        target[1] + sd[1] * 350.0,
        target[2] + sd[2] * 350.0,
    ];
    let f = norm3_or(
        [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]],
        [0.0, -1.0, 0.0],
    );
    let up_hint = if f[1].abs() > 0.96 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let s = norm3_or(cross3_v(f, up_hint), [1.0, 0.0, 0.0]);
    let u = cross3_v(s, f);
    let view = [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot3_v(s, eye), -dot3_v(u, eye), dot3_v(f, eye), 1.0],
    ];
    let half = (distance * 1.15).clamp(55.0, 260.0);
    let near = 10.0_f32;
    let far = 800.0_f32;
    let inv_h = 1.0 / half;
    let depth_inv = 1.0 / (near - far);
    let proj = [
        [inv_h, 0.0, 0.0, 0.0],
        [0.0, inv_h, 0.0, 0.0],
        [0.0, 0.0, depth_inv, 0.0],
        [0.0, 0.0, near * depth_inv, 1.0],
    ];
    mul_m4(proj, view)
}

fn make_uniforms(
    camera: &OrbitCamera,
    lighting: &ScMapLighting,
    map_size: [f32; 2],
    water_levels: [f32; 3],
    time_sec: f32,
    shadow_params: [f32; 4],
) -> Uniforms {
    let eye = camera.eye_position();
    let sd = norm3_or(lighting.sun_direction, [0.46, 0.71, 0.53]);
    let mult = if lighting.lighting_multiplier.is_finite() && lighting.lighting_multiplier > 0.1 {
        lighting.lighting_multiplier
    } else {
        1.4
    };
    Uniforms {
        view_proj: camera.view_proj_matrix(),
        shadow_view_proj: make_shadow_view_proj(camera.target, sd, camera.distance),
        eye_pos: [eye[0], eye[1], eye[2], time_sec],
        sun_dir: [sd[0], sd[1], sd[2], mult],
        sun_color: [
            lighting.sun_color[0],
            lighting.sun_color[1],
            lighting.sun_color[2],
            water_levels[0],
        ],
        ambient_color: [
            lighting.shadow_fill_color[0],
            lighting.shadow_fill_color[1],
            lighting.shadow_fill_color[2],
            map_size[0],
        ],
        fog_color: [
            lighting.fog_color[0],
            lighting.fog_color[1],
            lighting.fog_color[2],
            map_size[1],
        ],
        fog_params: [
            lighting.fog_start,
            lighting.fog_end.max(lighting.fog_start + 300.0),
            water_levels[1],
            water_levels[2],
        ],
        shadow_params,
    }
}

pub(crate) fn create_attachment(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    sample_count: u32,
    usage: wgpu::TextureUsages,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// Renders `scene` offscreen on the GPU and writes an RGBA8 PNG to `out_path`.
pub fn render_scene_to_png(
    scene: &LoadedScene,
    width: u32,
    height: u32,
    out_path: &Path,
) -> Result<String, String> {
    if width == 0 || height == 0 {
        return Err("render dimensions must be non-zero".to_string());
    }

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| "no compatible Vulkan/GPU adapter found".to_string())?;

    let adapter_info = adapter.get_info();
    let (device, queue) = pollster::block_on(
        adapter.request_device(&wgpu::DeviceDescriptor::default(), None),
    )
    .map_err(|e| format!("failed to create GPU device: {e}"))?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let msaa4_ok = hdr_msaa4_supported(&adapter);
    let mut camera = scene.initial_camera.clone();
    camera.set_aspect(width, height);

    let resources =
        GpuSceneResources::new(&device, &queue, scene, format, width, height, msaa4_ok, &camera);

    let (resolve_tex, resolve_view) = create_attachment(
        &device,
        width,
        height,
        format,
        1,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        "offscreen_resolve",
    );

    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let unpadded_bpr = width * 4;
    let padded_bpr = unpadded_bpr.div_ceil(align) * align;
    let readback_size = u64::from(padded_bpr) * u64::from(height);
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen_readback"),
        size: readback_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("offscreen_encoder"),
    });
    resources.encode_draw(&mut encoder, &resolve_view, scene.clear_color);
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &resolve_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| format!("readback channel failed: {e}"))?
        .map_err(|e| format!("GPU buffer map failed: {e}"))?;

    let mapped = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((unpadded_bpr * height) as usize);
    for row in 0..height as usize {
        let start = row * padded_bpr as usize;
        pixels.extend_from_slice(&mapped[start..start + unpadded_bpr as usize]);
    }
    drop(mapped);
    readback_buf.unmap();

    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    let file = File::create(out_path)
        .map_err(|e| format!("failed to create {}: {e}", out_path.display()))?;
    let mut png_enc = png::Encoder::new(BufWriter::new(file), width, height);
    png_enc.set_color(png::ColorType::Rgba);
    png_enc.set_depth(png::BitDepth::Eight);
    let mut writer = png_enc
        .write_header()
        .map_err(|e| format!("png header error: {e}"))?;
    writer
        .write_image_data(&pixels)
        .map_err(|e| format!("png write error: {e}"))?;

    Ok(format!("{} ({:?})", adapter_info.name, adapter_info.backend))
}

/// Opens an interactive `winit` + `wgpu` viewer window with orbit, pan, and strategic zoom.
pub fn run_window(scene: LoadedScene, width: u32, height: u32) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| format!("failed to create event loop: {e}"))?;
    let mut app = ViewerApp {
        scene,
        init_size: (width.max(320), height.max(240)),
        state: None,
        init_err: None,
    };
    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("event loop error: {e}"))?;
    if let Some(err) = app.init_err {
        return Err(err);
    }
    Ok(())
}

struct WindowGpuState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    resources: GpuSceneResources,
    camera: OrbitCamera,
    left_drag: bool,
    right_drag: bool,
    last_cursor: Option<(f64, f64)>,
}

struct ViewerApp {
    scene: LoadedScene,
    init_size: (u32, u32),
    state: Option<WindowGpuState>,
    init_err: Option<String>,
}

impl ViewerApp {
    fn init_window(&mut self, event_loop: &ActiveEventLoop) -> Result<WindowGpuState, String> {
        let attrs = Window::default_attributes()
            .with_title(format!("FARust Viewer — {}", self.scene.title))
            .with_inner_size(PhysicalSize::new(self.init_size.0, self.init_size.1));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .map_err(|e| format!("failed to create window: {e}"))?,
        );

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("failed to create wgpu surface: {e}"))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| "no compatible GPU adapter for window surface".to_string())?;

        let (device, queue) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor::default(), None),
        )
        .map_err(|e| format!("failed to request GPU device: {e}"))?;

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let mut camera = self.scene.initial_camera.clone();
        camera.set_aspect(width, height);

        let msaa4_ok = hdr_msaa4_supported(&adapter);
        let resources = GpuSceneResources::new(
            &device,
            &queue,
            &self.scene,
            format,
            width,
            height,
            msaa4_ok,
            &camera,
        );

        Ok(WindowGpuState {
            window,
            surface,
            device,
            queue,
            config,
            resources,
            camera,
            left_drag: false,
            right_drag: false,
            last_cursor: None,
        })
    }
}

impl ApplicationHandler for ViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match self.init_window(event_loop) {
            Ok(st) => {
                st.window.request_redraw();
                self.state = Some(st);
            }
            Err(err) => {
                self.init_err = Some(err);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(st) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                let w = new_size.width.max(1);
                let h = new_size.height.max(1);
                st.config.width = w;
                st.config.height = h;
                st.surface.configure(&st.device, &st.config);
                st.resources.resize(&st.device, w, h);
                st.camera.set_aspect(w, h);
                // Window re-config: apply active shadow & HDR post settings live.
                st.resources
                    .set_shadow_quality(&st.device, crate::settings::shadow_quality());
                st.resources.sync_render_settings(&st.device, &st.queue);
                st.resources.update_camera(&st.queue, &st.camera);
                st.window.request_redraw();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => st.left_drag = pressed,
                    MouseButton::Right | MouseButton::Middle => st.right_drag = pressed,
                    _ => {}
                }
                if !st.left_drag && !st.right_drag {
                    st.last_cursor = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let cur = (position.x, position.y);
                if let Some((px, py)) = st.last_cursor {
                    let dx = (cur.0 - px) as f32;
                    let dy = (cur.1 - py) as f32;
                    if st.left_drag {
                        st.camera.orbit(-dx * 0.006, dy * 0.006);
                        st.resources.update_camera(&st.queue, &st.camera);
                        st.window.request_redraw();
                    } else if st.right_drag {
                        st.camera.pan(-dx * 0.0018, dy * 0.0018);
                        st.resources.update_camera(&st.queue, &st.camera);
                        st.window.request_redraw();
                    }
                }
                if st.left_drag || st.right_drag {
                    st.last_cursor = Some(cur);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y / 40.0) as f32,
                };
                if steps.abs() > 0.0 {
                    st.camera.zoom(steps);
                    st.resources.update_camera(&st.queue, &st.camera);
                    st.window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                let mut changed = true;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => {
                        event_loop.exit();
                        return;
                    }
                    PhysicalKey::Code(KeyCode::KeyR) => {
                        let aspect = st.camera.aspect;
                        st.camera = self.scene.initial_camera.clone();
                        st.camera.aspect = aspect;
                    }
                    PhysicalKey::Code(KeyCode::KeyW | KeyCode::ArrowUp) => st.camera.pan(0.0, 0.06),
                    PhysicalKey::Code(KeyCode::KeyS | KeyCode::ArrowDown) => {
                        st.camera.pan(0.0, -0.06)
                    }
                    PhysicalKey::Code(KeyCode::KeyA | KeyCode::ArrowLeft) => {
                        st.camera.pan(-0.06, 0.0)
                    }
                    PhysicalKey::Code(KeyCode::KeyD | KeyCode::ArrowRight) => {
                        st.camera.pan(0.06, 0.0)
                    }
                    PhysicalKey::Code(KeyCode::KeyQ) => st.camera.orbit(-0.10, 0.0),
                    PhysicalKey::Code(KeyCode::KeyE) => st.camera.orbit(0.10, 0.0),
                    _ => changed = false,
                }
                if changed {
                    st.resources.update_camera(&st.queue, &st.camera);
                    st.window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                let frame = match st.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                        st.surface.configure(&st.device, &st.config);
                        return;
                    }
                    Err(_) => return,
                };
                let resolve_view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                    st.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("win_encoder"),
                        });
                st.resources
                    .encode_draw(&mut encoder, &resolve_view, self.scene.clear_color);
                st.queue.submit(Some(encoder.finish()));
                frame.present();
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_unit_instance_is_96_bytes_and_shield_mesh_valid() {
        assert_eq!(std::mem::size_of::<GpuUnitInstance>(), 96);
        let (verts, indices) = build_shield_hemisphere_mesh();
        assert!(!verts.is_empty() && !indices.is_empty());
        assert!(indices.iter().all(|&i| (i as usize) < verts.len()));
    }

    #[test]
    fn shadow_uniform_params_follow_quality() {
        // Stock behavior is Medium = 2048px (single source of truth in settings).
        assert_eq!(ShadowQuality::Medium.map_size(), 2048);
        let off = shadow_uniform_params(ShadowQuality::Off);
        assert_eq!(off[0], 1.0);
        assert_eq!(off[1], 0.0);
        for (q, edge) in [
            (ShadowQuality::Low, 1024.0),
            (ShadowQuality::Medium, 2048.0),
            (ShadowQuality::High, 4096.0),
        ] {
            let p = shadow_uniform_params(q);
            assert!((p[0] - 1.0 / edge).abs() < 1e-9, "{q:?}");
            assert_eq!(p[1], 1.0);
            assert_eq!(&p[2..], &[0.0, 0.0]);
        }
    }

    #[test]
    fn prop_aabb_frustum_culling_handles_visible_and_behind_camera() {
        let mut cam = OrbitCamera::for_map(512.0, 512.0, 10.0, 1.6);
        cam.target = [256.0, 10.0, 256.0];
        cam.distance = 100.0;
        let planes = extract_frustum_planes(&cam.view_proj_matrix());

        assert!(aabb_in_frustum(&planes, [250.0, 5.0, 250.0], [262.0, 25.0, 262.0]));
        assert!(!aabb_in_frustum(
            &planes,
            [250.0, 5.0, 900.0],
            [262.0, 25.0, 920.0]
        ));
    }

    #[test]
    fn shadow_prop_frustum_culling_handles_visible_and_outside() {
        let shadow_vp = make_shadow_view_proj([256.0, 10.0, 256.0], [0.46, 0.71, 0.53], 100.0);
        let planes = extract_frustum_planes(&shadow_vp);

        // Inside the shadow box centered around camera target (half = 115.0).
        assert!(aabb_in_frustum(
            &planes,
            [240.0, 0.0, 240.0],
            [270.0, 30.0, 270.0]
        ));
        // Far outside the shadow box on the map.
        assert!(!aabb_in_frustum(
            &planes,
            [600.0, 0.0, 600.0],
            [620.0, 30.0, 620.0]
        ));
    }

    #[test]
    fn bloom_mip_sizes_halves_cleanly_to_1x1() {
        assert_eq!(
            bloom_mip_sizes(1920, 1080, 5),
            vec![(960, 540), (480, 270), (240, 135), (120, 67), (60, 33)]
        );
        assert_eq!(
            bloom_mip_sizes(1280, 800, 5),
            vec![(640, 400), (320, 200), (160, 100), (80, 50), (40, 25)]
        );
        assert_eq!(bloom_mip_sizes(4, 2, 5), vec![(2, 1), (1, 1)]);
        assert_eq!(bloom_mip_sizes(1, 1, 5), vec![(1, 1)]);
        assert_eq!(bloom_mip_sizes(0, 0, 5), vec![(1, 1)]);
        assert!(bloom_mip_sizes(1920, 1080, 0).is_empty());
    }

    #[test]
    fn aces_tonemap_curve_is_monotonic_bounded_and_gamma_responsive() {
        assert_eq!(aces_tonemap_scalar(0.0), 0.0);
        assert_eq!(aces_tonemap_scalar(-5.0), 0.0);
        assert_eq!(aces_tonemap_scalar(f32::NAN), 0.0);

        let mut prev = 0.0;
        for &x in &[0.05, 0.18, 0.5, 1.0, 2.0, 5.0] {
            let y = aces_tonemap_scalar(x);
            assert!(y > prev && y <= 1.0, "not strictly increasing at {x}: {prev} -> {y}");
            prev = y;
        }
        for &x in &[20.0, 100.0] {
            let y = aces_tonemap_scalar(x);
            assert!(y >= prev && y <= 1.0, "not bounded/monotonic at {x}: {prev} -> {y}");
            prev = y;
        }
        // Highlight rolloff preserves headroom above 1.0 instead of hard clipping.
        assert!(aces_tonemap_scalar(1.0) < 0.85);
        assert!(aces_tonemap_scalar(5.0) > 0.95);

        // Relative gamma: 1.0 neutral, >1.0 brightens midtones, <1.0 darkens midtones.
        let mid_dark = tonemap_rgb([0.3, 0.3, 0.3], 0.85, 0.8)[0];
        let mid_neut = tonemap_rgb([0.3, 0.3, 0.3], 0.85, 1.0)[0];
        let mid_bright = tonemap_rgb([0.3, 0.3, 0.3], 0.85, 1.4)[0];
        assert!(mid_dark < mid_neut && mid_neut < mid_bright);

        // Exposure from .scmap LightingMultiplier increases monotonically.
        assert!(exposure_from_lighting(1.0) < exposure_from_lighting(1.4));
        assert!(exposure_from_lighting(1.4) < exposure_from_lighting(1.8));
        assert!((exposure_from_lighting(f32::NAN) - exposure_from_lighting(1.4)).abs() < 1e-6);
    }

    #[test]
    fn bloom_strength_and_aa_resolution() {
        assert_eq!(bloom_strength_from_map(0.08, false), 0.0);
        assert_eq!(bloom_strength_from_map(0.0, true), 0.0);
        assert_eq!(bloom_strength_from_map(-0.1, true), 0.0);
        let s = bloom_strength_from_map(0.08, true);
        assert!(s > 0.20 && s < 0.40, "unexpected SCMP_001 bloom strength {s}");
        assert!(bloom_strength_from_map(10.0, true) <= 0.70);

        assert_eq!(resolve_aa_mode(AaMode::Msaa4, true), AaMode::Msaa4);
        assert_eq!(resolve_aa_mode(AaMode::Msaa4, false), AaMode::Fxaa);
        assert_eq!(resolve_aa_mode(AaMode::Fxaa, false), AaMode::Fxaa);
        assert_eq!(resolve_aa_mode(AaMode::Off, false), AaMode::Off);
    }
}
