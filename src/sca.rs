//! Phase 11 clean-room Supreme Commander `.sca` (v5) skeletal animation parser,
//! keyframe SLERP sampler, and hierarchical GPU skinning matrix builder.

use crate::scm::ScmMesh;

const MAGIC: &[u8; 4] = b"ANIM";
const HEADER_SIZE: usize = 36;
const BONE_POSE_SIZE: usize = 28;

/// Local bone translation and orientation quaternion `[w, x, y, z]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaBonePose {
    pub position: [f32; 3],
    pub rotation_wxyz: [f32; 4],
}

impl Default for ScaBonePose {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation_wxyz: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

/// Single keyframe in an `.sca` animation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ScaFrame {
    pub time: f32,
    pub flags: u32,
    pub bones: Vec<ScaBonePose>,
}

/// Parsed `.sca` v5 skeletal animation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ScaAnimation {
    pub version: u32,
    pub duration: f32,
    pub bone_names: Vec<String>,
    pub parent_links: Vec<i32>,
    pub root_delta: ScaBonePose,
    pub frames: Vec<ScaFrame>,
}

/// Borrowed skeleton bone descriptor decoupled from mesh normalization so `.sca`
/// poses can be retargeted onto any skeleton.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct SkinBoneRef<'a> {
    pub name: &'a str,
    pub parent_index: i32,
    pub position: [f32; 3],
    pub rotation_wxyz: [f32; 4],
    pub inv_bind_pose: [f32; 16],
}

#[inline]
fn read_u32(data: &[u8], off: usize) -> Result<u32, String> {
    let bytes: [u8; 4] = data
        .get(off..off + 4)
        .ok_or_else(|| format!("SCA truncated reading u32 at {off}"))?
        .try_into()
        .map_err(|_| format!("SCA slice error at {off}"))?;
    Ok(u32::from_le_bytes(bytes))
}

#[inline]
fn read_i32(data: &[u8], off: usize) -> Result<i32, String> {
    let bytes: [u8; 4] = data
        .get(off..off + 4)
        .ok_or_else(|| format!("SCA truncated reading i32 at {off}"))?
        .try_into()
        .map_err(|_| format!("SCA slice error at {off}"))?;
    Ok(i32::from_le_bytes(bytes))
}

#[inline]
fn read_f32(data: &[u8], off: usize) -> Result<f32, String> {
    Ok(f32::from_bits(read_u32(data, off)?))
}

fn read_pose(data: &[u8], off: usize) -> Result<ScaBonePose, String> {
    Ok(ScaBonePose {
        position: [
            read_f32(data, off)?,
            read_f32(data, off + 4)?,
            read_f32(data, off + 8)?,
        ],
        rotation_wxyz: [
            read_f32(data, off + 12)?,
            read_f32(data, off + 16)?,
            read_f32(data, off + 20)?,
            read_f32(data, off + 24)?,
        ],
    })
}

impl ScaAnimation {
    /// Parses a Supreme Commander `.sca` skeletal animation binary payload.
    pub fn parse(data: &[u8]) -> Result<Self, String> {
        if data.len() < HEADER_SIZE {
            return Err(format!(
                "SCA too short: expected >= {HEADER_SIZE} bytes, got {}",
                data.len()
            ));
        }
        if &data[0..4] != MAGIC {
            return Err(format!("invalid SCA magic: {:?}", &data[0..4]));
        }
        let version = read_u32(data, 4)?;
        if version != 5 {
            return Err(format!("unsupported SCA version {version} (expected 5)"));
        }
        let num_frames = read_u32(data, 8)? as usize;
        // Corrupt durations (NaN/inf/negative) would make `clamp(0.0, duration)` panic downstream.
        let duration = read_f32(data, 12)?;
        let duration = if duration.is_finite() && duration >= 0.0 { duration } else { 0.0 };
        let num_bones = read_u32(data, 16)? as usize;
        let names_off = read_u32(data, 20)? as usize;
        let links_off = read_u32(data, 24)? as usize;
        let anim_off = read_u32(data, 28)? as usize;
        let frame_size = read_u32(data, 32)? as usize;

        if names_off > links_off || links_off > data.len() || anim_off > data.len() {
            return Err("invalid SCA section offsets".to_string());
        }
        let min_frame_size = 8usize
            .checked_add(
                num_bones
                    .checked_mul(BONE_POSE_SIZE)
                    .ok_or_else(|| "SCA bone count overflow".to_string())?,
            )
            .ok_or_else(|| "SCA frame size overflow".to_string())?;
        if frame_size < min_frame_size {
            return Err(format!(
                "invalid SCA frame_size {frame_size} < {min_frame_size}"
            ));
        }

        // Every bone needs a 4-byte parent link inside the file: bounds `num_bones` by the input
        // size before anything is allocated per bone.
        if num_bones
            .checked_mul(4)
            .and_then(|b| b.checked_add(links_off))
            .is_none_or(|end| end > data.len())
        {
            return Err("SCA links section truncated".to_string());
        }

        // Parse NUL-terminated bone names in names_off..links_off
        let names_slice = &data[names_off..links_off];
        let mut bone_names = Vec::with_capacity(num_bones);
        let mut cursor = 0usize;
        for _ in 0..num_bones {
            if cursor >= names_slice.len() {
                bone_names.push(String::new());
                continue;
            }
            let rel_end = names_slice[cursor..]
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(names_slice.len() - cursor);
            let s: String = names_slice[cursor..cursor + rel_end]
                .iter()
                .map(|&b| char::from(b))
                .collect();
            bone_names.push(s);
            cursor += rel_end + 1;
        }

        let links_bytes = num_bones
            .checked_mul(4)
            .ok_or_else(|| "SCA links overflow".to_string())?;
        if links_off + links_bytes > data.len() {
            return Err("SCA links section truncated".to_string());
        }
        let mut parent_links = Vec::with_capacity(num_bones);
        for i in 0..num_bones {
            parent_links.push(read_i32(data, links_off + i * 4)?);
        }

        if anim_off + BONE_POSE_SIZE > data.len() {
            return Err("SCA root delta truncated".to_string());
        }
        let root_delta = read_pose(data, anim_off)?;

        let frames_start = anim_off + BONE_POSE_SIZE;
        let total_frame_bytes = num_frames
            .checked_mul(frame_size)
            .ok_or_else(|| "SCA frames size overflow".to_string())?;
        if frames_start + total_frame_bytes > data.len() {
            return Err("SCA frames data truncated".to_string());
        }

        let mut frames = Vec::with_capacity(num_frames);
        for fi in 0..num_frames {
            let base = frames_start + fi * frame_size;
            let time = read_f32(data, base)?;
            let flags = read_u32(data, base + 4)?;
            let mut bones = Vec::with_capacity(num_bones);
            for bi in 0..num_bones {
                bones.push(read_pose(data, base + 8 + bi * BONE_POSE_SIZE)?);
            }
            frames.push(ScaFrame { time, flags, bones });
        }

        Ok(Self {
            version,
            duration,
            bone_names,
            parent_links,
            root_delta,
            frames,
        })
    }

    #[allow(dead_code)]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    #[allow(dead_code)]
    pub fn bone_count(&self) -> usize {
        self.bone_names.len()
    }

    /// Samples local bone poses at `time_sec` using linear position interpolation and shortest-arc SLERP.
    #[allow(dead_code)]
    pub fn sample_local(&self, time_sec: f32, looping: bool) -> Vec<ScaBonePose> {
        let nbones = self.bone_names.len();
        if self.frames.is_empty() {
            return vec![ScaBonePose::default(); nbones];
        }
        if self.frames.len() == 1 || self.duration <= 1e-6 {
            return self.frames[0].bones.clone();
        }

        let t = if looping {
            let m = time_sec.rem_euclid(self.duration);
            if m.is_finite() { m } else { 0.0 }
        } else {
            time_sec.clamp(0.0, self.duration)
        };

        let last = self.frames.len() - 1;
        let (f0_idx, f1_idx, alpha) = if t >= self.frames[last].time {
            (last, last, 0.0)
        } else {
            let mut idx0 = 0usize;
            for i in 0..last {
                if self.frames[i + 1].time >= t {
                    idx0 = i;
                    break;
                }
            }
            let idx1 = (idx0 + 1).min(last);
            let t0 = self.frames[idx0].time;
            let t1 = self.frames[idx1].time;
            let span = (t1 - t0).max(1e-6);
            (idx0, idx1, ((t - t0) / span).clamp(0.0, 1.0))
        };

        let f0 = &self.frames[f0_idx].bones;
        let f1 = &self.frames[f1_idx].bones;
        (0..nbones)
            .map(|bi| {
                let p0 = f0.get(bi).copied().unwrap_or_default();
                let p1 = f1.get(bi).copied().unwrap_or(p0);
                ScaBonePose {
                    position: [
                        p0.position[0] + (p1.position[0] - p0.position[0]) * alpha,
                        p0.position[1] + (p1.position[1] - p0.position[1]) * alpha,
                        p0.position[2] + (p1.position[2] - p0.position[2]) * alpha,
                    ],
                    rotation_wxyz: slerp_wxyz(p0.rotation_wxyz, p1.rotation_wxyz, alpha),
                }
            })
            .collect()
    }

    /// Maps each target skeleton bone name to an optional index in this `.sca`'s bone table.
    #[allow(dead_code)]
    pub fn build_bone_remap(&self, target_bone_names: &[&str]) -> Vec<Option<usize>> {
        target_bone_names
            .iter()
            .map(|tname| {
                self.bone_names
                    .iter()
                    .position(|sname| sname.eq_ignore_ascii_case(tname))
            })
            .collect()
    }
}

#[inline]
pub fn normalize_wxyz(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n > 1e-6 {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

/// Shortest-arc spherical linear interpolation for quaternions ordered `[w, x, y, z]`.
pub fn slerp_wxyz(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let qa = normalize_wxyz(a);
    let mut qb = normalize_wxyz(b);
    let mut dot = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3];
    if dot < 0.0 {
        dot = -dot;
        qb = [-qb[0], -qb[1], -qb[2], -qb[3]];
    }
    if dot > 0.9995 {
        return normalize_wxyz([
            qa[0] + (qb[0] - qa[0]) * t,
            qa[1] + (qb[1] - qa[1]) * t,
            qa[2] + (qb[2] - qa[2]) * t,
            qa[3] + (qb[3] - qa[3]) * t,
        ]);
    }
    let theta_0 = dot.clamp(-1.0, 1.0).acos();
    let theta = theta_0 * t;
    let sin_theta = theta.sin();
    let sin_theta_0 = theta_0.sin();
    let s0 = theta.cos() - dot * sin_theta / sin_theta_0;
    let s1 = sin_theta / sin_theta_0;
    [
        s0 * qa[0] + s1 * qb[0],
        s0 * qa[1] + s1 * qb[1],
        s0 * qa[2] + s1 * qb[2],
        s0 * qa[3] + s1 * qb[3],
    ]
}

#[inline]
pub fn mat4_identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Column-major 4x4 matrix multiply (`a * b`).
#[inline]
pub fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = a[0][r] * b[c][0]
                + a[1][r] * b[c][1]
                + a[2][r] * b[c][2]
                + a[3][r] * b[c][3];
        }
    }
    out
}

/// Column-major 4x4 matrix for local translation `pos` and quaternion `[w, x, y, z]`.
pub fn mat4_from_trs_wxyz(pos: [f32; 3], rot_wxyz: [f32; 4]) -> [[f32; 4]; 4] {
    let [w, x, y, z] = normalize_wxyz(rot_wxyz);
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy), 0.0],
        [2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx), 0.0],
        [2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy), 0.0],
        [pos[0], pos[1], pos[2], 1.0],
    ]
}

#[inline]
pub fn mat4_rot_y(rad: f32) -> [[f32; 4]; 4] {
    let (s, c) = rad.sin_cos();
    [
        [c, 0.0, -s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

#[inline]
pub fn mat4_rot_x(rad: f32) -> [[f32; 4]; 4] {
    let (s, c) = rad.sin_cos();
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, s, 0.0],
        [0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

#[inline]
fn inv_bind_to_col_mat4(raw: &[f32; 16]) -> [[f32; 4]; 4] {
    // `.scm` stores inverse bind pose as 4 DX rows [r0, r1, r2, r3=(tx,ty,tz,1)].
    // Viewed as 4 WGSL columns, `v_col' = M_col * v_col` matches `v_row' = v_row * M_row`.
    [
        [raw[0], raw[1], raw[2], raw[3]],
        [raw[4], raw[5], raw[6], raw[7]],
        [raw[8], raw[9], raw[10], raw[11]],
        [raw[12], raw[13], raw[14], raw[15]],
    ]
}

/// Convenience wrapper building skinning matrices directly from an [`ScmMesh`].
#[allow(dead_code)]
pub fn compute_scm_skin_matrices(
    mesh: &ScmMesh,
    anim: Option<(&ScaAnimation, &[Option<usize>], f32)>,
    turret_yaw: Option<(&str, f32)>,
    turret_pitch: Option<(&str, f32)>,
    norm_offset: [f32; 3],
    norm_scale: f32,
) -> Vec<[[f32; 4]; 4]> {
    let refs: Vec<SkinBoneRef<'_>> = mesh
        .bones
        .iter()
        .map(|b| SkinBoneRef {
            name: &b.name,
            parent_index: b.parent_index,
            position: b.position,
            rotation_wxyz: b.rotation,
            inv_bind_pose: b.inv_bind_pose,
        })
        .collect();
    compute_skin_matrices(&refs, anim, turret_yaw, turret_pitch, norm_offset, norm_scale)
}

/// Computes GPU-ready column-major skinning matrices in normalized mesh vertex space:
/// `N * GlobalPose[b] * InvBindPose[b] * N^{-1}`, where `N` maps raw SCM space to normalized space
/// via `p_norm = (p_raw - norm_offset) * norm_scale`.
#[allow(dead_code)]
pub fn compute_skin_matrices(
    bones: &[SkinBoneRef<'_>],
    anim: Option<(&ScaAnimation, &[Option<usize>], f32)>,
    turret_yaw: Option<(&str, f32)>,
    turret_pitch: Option<(&str, f32)>,
    norm_offset: [f32; 3],
    norm_scale: f32,
) -> Vec<[[f32; 4]; 4]> {
    let n = bones.len();
    if n == 0 {
        return Vec::new();
    }

    let sampled = anim.map(|(a, _, t)| a.sample_local(t, true));
    let mut local_mats = Vec::with_capacity(n);

    for (i, b) in bones.iter().enumerate() {
        let bind = ScaBonePose {
            position: b.position,
            rotation_wxyz: b.rotation_wxyz,
        };
        let pose = match (anim, sampled.as_ref()) {
            (Some((_, remap, _)), Some(poses)) => match remap.get(i).copied() {
                Some(Some(sca_bi)) => poses.get(sca_bi).copied().unwrap_or(bind),
                _ => bind,
            },
            _ => bind,
        };

        let mut m = mat4_from_trs_wxyz(pose.position, pose.rotation_wxyz);
        if let Some((yaw_bone, rad)) = turret_yaw
            && rad.abs() > 1e-4
            && b.name.eq_ignore_ascii_case(yaw_bone)
        {
            let r = mat4_rot_y(rad);
            m = mat4_mul(&m, &r);
        }
        if let Some((pitch_bone, rad)) = turret_pitch
            && rad.abs() > 1e-4
            && b.name.eq_ignore_ascii_case(pitch_bone)
        {
            let r = mat4_rot_x(rad);
            m = mat4_mul(&m, &r);
        }
        local_mats.push(m);
    }
    skin_from_local_mats(bones, &local_mats, norm_offset, norm_scale)
}

/// Resolves the bone hierarchy over per-bone local matrices and returns normalized-space skinning
/// matrices `N * Global[b] * InvBind[b] * N^{-1}`.
fn skin_from_local_mats(
    bones: &[SkinBoneRef<'_>],
    local_mats: &[[[f32; 4]; 4]],
    norm_offset: [f32; 3],
    norm_scale: f32,
) -> Vec<[[f32; 4]; 4]> {
    let n = bones.len().min(local_mats.len());
    let mut globals = vec![mat4_identity(); n];
    let mut visited = vec![0u8; n];
    fn resolve(
        idx: usize,
        bones: &[SkinBoneRef<'_>],
        locals: &[[[f32; 4]; 4]],
        globals: &mut [[[f32; 4]; 4]],
        visited: &mut [u8],
    ) {
        if visited[idx] == 2 {
            return;
        }
        if visited[idx] == 1 {
            // Cycle guard
            globals[idx] = locals[idx];
            visited[idx] = 2;
            return;
        }
        visited[idx] = 1;
        let p = bones[idx].parent_index;
        if p >= 0 && (p as usize) < bones.len() && (p as usize) != idx {
            let pi = p as usize;
            resolve(pi, bones, locals, globals, visited);
            globals[idx] = mat4_mul(&globals[pi], &locals[idx]);
        } else {
            globals[idx] = locals[idx];
        }
        visited[idx] = 2;
    }

    let bones = &bones[..n];
    for i in 0..n {
        resolve(i, bones, local_mats, &mut globals, &mut visited);
    }

    let s = if norm_scale.is_finite() && norm_scale.abs() > 1e-6 {
        norm_scale
    } else {
        1.0
    };
    let inv_s = 1.0 / s;
    // N: p_norm = (p_raw - c) * s
    let norm_m = [
        [s, 0.0, 0.0, 0.0],
        [0.0, s, 0.0, 0.0],
        [0.0, 0.0, s, 0.0],
        [-norm_offset[0] * s, -norm_offset[1] * s, -norm_offset[2] * s, 1.0],
    ];
    // N^{-1}: p_raw = p_norm / s + c
    let norm_inv = [
        [inv_s, 0.0, 0.0, 0.0],
        [0.0, inv_s, 0.0, 0.0],
        [0.0, 0.0, inv_s, 0.0],
        [norm_offset[0], norm_offset[1], norm_offset[2], 1.0],
    ];

    (0..n)
        .map(|i| {
            let inv_b = inv_bind_to_col_mat4(&bones[i].inv_bind_pose);
            let skin_raw = mat4_mul(&globals[i], &inv_b);
            let tmp = mat4_mul(&norm_m, &skin_raw);
            mat4_mul(&tmp, &norm_inv)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Phase 11 in-game unit rig: clip selection, time wrapping, bone retargeting
// ---------------------------------------------------------------------------

/// Which skeletal clip a unit plays this frame (purely visual; derived from read-only sim state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimMode {
    /// Bind pose (no clip, or no matching clip).
    Rest,
    /// Looping `Display.AnimationIdle` clip.
    Idle,
    /// Looping `Display.AnimationWalk` clip.
    Walk,
}

/// Picks the clip for a unit: walk while moving (if present), idle otherwise (if present), else rest.
pub fn select_anim_mode(moving: bool, has_walk: bool, has_idle: bool) -> AnimMode {
    if moving && has_walk {
        AnimMode::Walk
    } else if has_idle {
        AnimMode::Idle
    } else {
        AnimMode::Rest
    }
}

/// Wraps `t` into `[0, duration)`; non-finite inputs or degenerate durations map to `0`.
pub fn wrap_anim_time(t: f32, duration: f32) -> f32 {
    if !t.is_finite() || !duration.is_finite() || duration <= 1e-6 {
        return 0.0;
    }
    let m = t.rem_euclid(duration);
    if m.is_finite() && m < duration { m } else { 0.0 }
}

/// Maps every `target` bone name to the index of the same-named (case-insensitive) `source` bone.
pub fn remap_bones_by_name(target: &[&str], source: &[&str]) -> Vec<Option<usize>> {
    target
        .iter()
        .map(|t| source.iter().position(|s| s.eq_ignore_ascii_case(t)))
        .collect()
}

/// Wraps an angle in radians into `(-PI, PI]`.
pub fn wrap_angle_pi(a: f32) -> f32 {
    if !a.is_finite() {
        return 0.0;
    }
    let tau = std::f32::consts::TAU;
    let mut r = (a + std::f32::consts::PI).rem_euclid(tau) - std::f32::consts::PI;
    if r <= -std::f32::consts::PI {
        r += tau;
    }
    r
}

/// A looping `.sca` clip retargeted onto a rig skeleton (`remap[rig_bone] = sca_bone`).
#[derive(Debug, Clone)]
pub struct AnimClip {
    pub anim: ScaAnimation,
    pub remap: Vec<Option<usize>>,
}

impl AnimClip {
    /// Builds a clip for `rig_bone_names`; returns `None` when no bone matches by name.
    pub fn new(anim: ScaAnimation, rig_bone_names: &[&str]) -> Option<Self> {
        let src: Vec<&str> = anim.bone_names.iter().map(String::as_str).collect();
        let remap = remap_bones_by_name(rig_bone_names, &src);
        if remap.iter().all(Option::is_none) || anim.frames.is_empty() {
            return None;
        }
        Some(Self { anim, remap })
    }

    pub fn duration(&self) -> f32 {
        self.anim.duration
    }
}

/// Procedural weapon pose layered over the sampled clip (Phase 17).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AimPose {
    /// Turret yaw relative to the hull, radians (positive turns +Z toward +X).
    pub yaw: f32,
    /// Barrel elevation, radians (positive = up).
    pub pitch: f32,
    /// Recoil kick `0..=1`, scaling each rack's `RackRecoilDistance`.
    pub recoil: f32,
}

impl AimPose {
    pub fn is_neutral(&self) -> bool {
        self.yaw.abs() < 1e-3 && self.pitch.abs() < 1e-3 && self.recoil.abs() < 1e-3
    }
}

/// Weapon bones resolved to rig bone indices once per unit type.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RigBoneControls {
    pub yaw_bones: Vec<usize>,
    pub pitch_bones: Vec<usize>,
    /// `(rack bone, RackRecoilDistance)` in raw mesh units along the bone's local Z.
    pub recoil_bones: Vec<(usize, f32)>,
}

/// Smoothstep crossfade weight of the *new* clip `elapsed` seconds into a `duration` fade.
pub fn crossfade_weight(elapsed: f32, duration: f32) -> f32 {
    if !elapsed.is_finite() || duration <= 1e-6 {
        return 1.0;
    }
    let x = (elapsed / duration).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Per-bone blend `a -> b` (`w = 0` keeps `a`): lerped translation, shortest-arc SLERP rotation.
pub fn blend_poses(a: &[ScaBonePose], b: &[ScaBonePose], w: f32) -> Vec<ScaBonePose> {
    let w = if w.is_finite() { w.clamp(0.0, 1.0) } else { 1.0 };
    a.iter()
        .zip(b.iter())
        .map(|(pa, pb)| ScaBonePose {
            position: [
                pa.position[0] + (pb.position[0] - pa.position[0]) * w,
                pa.position[1] + (pb.position[1] - pa.position[1]) * w,
                pa.position[2] + (pb.position[2] - pa.position[2]) * w,
            ],
            rotation_wxyz: slerp_wxyz(pa.rotation_wxyz, pb.rotation_wxyz, w),
        })
        .collect()
}

/// Barrel recoil curve `t` seconds after a shot: instant kick to `1`, quadratic ease back to rest
/// over `return_secs`, `0` before the shot or once settled.
pub fn recoil_curve(t: f32, return_secs: f32) -> f32 {
    if !t.is_finite() || t < 0.0 || return_secs <= 1e-6 || t >= return_secs {
        return 0.0;
    }
    let r = 1.0 - t / return_secs;
    r * r
}

/// Moves `current` toward `target` by at most `max_step` (angles wrapped into `(-PI, PI]`).
pub fn slew_angle(current: f32, target: f32, max_step: f32) -> f32 {
    let d = wrap_angle_pi(target - current);
    let step = max_step.max(0.0);
    if d.abs() <= step {
        wrap_angle_pi(target)
    } else {
        wrap_angle_pi(current + step * d.signum())
    }
}

/// Barrel elevation (radians, positive up) to hit a point `dy` above the muzzle at horizontal
/// distance `dist`. Indirect-fire weapons use the launch slope of the render arc
/// `y = 4 t (1 - t) * arc_height` (`dy/dx = 4 * arc_height / dist + dy / dist` at `t = 0`).
pub fn aim_pitch(dist: f32, dy: f32, arc_height: f32, min_pitch: f32, max_pitch: f32) -> f32 {
    let d = dist.max(1e-3);
    let slope = if arc_height > 0.0 { 4.0 * arc_height / d + dy / d } else { dy / d };
    let p = slope.atan();
    if p.is_finite() { p.clamp(min_pitch.min(max_pitch), max_pitch) } else { 0.0 }
}

/// A one-shot clip (`AnimationDeath[]` entry) with its playback rate.
#[derive(Debug, Clone)]
pub struct OneShotClip {
    pub clip: AnimClip,
    pub rate: f32,
}

impl OneShotClip {
    /// Real seconds the clip takes at its rate.
    pub fn play_secs(&self) -> f32 {
        self.clip.duration() / self.rate.max(1e-3)
    }
}

/// Owned skeleton + normalization + clips for one unit type, producing GPU bone palettes.
#[derive(Debug, Clone)]
pub struct UnitRig {
    pub bones: Vec<crate::scm::ScmBone>,
    /// `p_norm = (p_raw - norm_offset) * norm_scale` (must match the uploaded template vertices).
    pub norm_offset: [f32; 3],
    pub norm_scale: f32,
    pub walk: Option<AnimClip>,
    pub idle: Option<AnimClip>,
    /// Clip-seconds advanced per real second while walking (`MaxSpeed / AnimationWalkRate`).
    pub walk_rate: f32,
    /// Turret yaw/pitch & rack recoil bones.
    pub controls: RigBoneControls,
    /// `Display.AnimationDeath[]` clips.
    pub death: Vec<OneShotClip>,
    /// Structure work clip: `AnimationBuild` (looping) or `AnimationOpen` (`looping = false`: opens,
    /// holds while working, closes in reverse).
    pub work: Option<(AnimClip, bool)>,
}

impl UnitRig {
    /// A rig without clips or weapon controls.
    pub fn new(bones: Vec<crate::scm::ScmBone>, norm_offset: [f32; 3], norm_scale: f32) -> Self {
        Self {
            bones,
            norm_offset,
            norm_scale,
            walk: None,
            idle: None,
            walk_rate: 1.0,
            controls: RigBoneControls::default(),
            death: Vec::new(),
            work: None,
        }
    }

    /// Resolves weapon bone names (case-insensitive, deduplicated) onto rig bone indices.
    /// Recoil distances are clamped to 15% of `span_raw` (the mesh footprint in raw units).
    pub fn set_weapon_bones(
        &mut self,
        yaw: &[&str],
        pitch: &[&str],
        recoil: &[(&str, f32)],
        span_raw: f32,
    ) {
        let find = |n: &str| self.bones.iter().position(|b| b.name.eq_ignore_ascii_case(n));
        let mut c = RigBoneControls::default();
        for n in yaw {
            if let Some(i) = find(n)
                && !c.yaw_bones.contains(&i)
            {
                c.yaw_bones.push(i);
            }
        }
        for n in pitch {
            if let Some(i) = find(n)
                && !c.pitch_bones.contains(&i)
                && !c.yaw_bones.contains(&i)
            {
                c.pitch_bones.push(i);
            }
        }
        let lim = (span_raw.abs() * 0.15).max(1e-3);
        for &(n, d) in recoil {
            if let Some(i) = find(n)
                && d.abs() > 1e-4
                && !c.recoil_bones.iter().any(|(b, _)| *b == i)
            {
                c.recoil_bones.push((i, d.clamp(-lim, lim)));
            }
        }
        self.controls = c;
    }

    pub fn has_turret(&self) -> bool {
        !self.controls.yaw_bones.is_empty() || !self.controls.pitch_bones.is_empty()
    }

    pub fn is_animated(&self) -> bool {
        self.walk.is_some()
            || self.idle.is_some()
            || self.has_turret()
            || !self.controls.recoil_bones.is_empty()
            || !self.death.is_empty()
            || self.work.is_some()
    }

    pub fn mode_for(&self, moving: bool) -> AnimMode {
        select_anim_mode(moving, self.walk.is_some(), self.idle.is_some())
    }

    pub fn clip_for(&self, mode: AnimMode) -> Option<&AnimClip> {
        match mode {
            AnimMode::Walk => self.walk.as_ref(),
            AnimMode::Idle => self.idle.as_ref(),
            AnimMode::Rest => None,
        }
    }

    /// Bind-pose local transforms of every rig bone.
    pub fn bind_pose(&self) -> Vec<ScaBonePose> {
        self.bones
            .iter()
            .map(|b| ScaBonePose { position: b.position, rotation_wxyz: b.rotation })
            .collect()
    }

    /// Local pose of every rig bone for `clip` at clip time `t` (bones the clip does not drive keep
    /// their bind pose). `looping = false` clamps `t` into the clip (one-shot death/open clips).
    pub fn sample_pose(&self, clip: Option<&AnimClip>, t: f32, looping: bool) -> Vec<ScaBonePose> {
        let mut pose = self.bind_pose();
        if let Some(c) = clip {
            let sampled = c.anim.sample_local(t, looping);
            for (i, p) in pose.iter_mut().enumerate() {
                if let Some(Some(si)) = c.remap.get(i).copied()
                    && let Some(s) = sampled.get(si)
                {
                    *p = *s;
                }
            }
        }
        pose
    }

    /// Skinning palette for a local `pose` with weapon `aim` layered on top: yaw about the yaw
    /// bones' local Y, elevation about the pitch bones' local X, recoil along the racks' local Z.
    pub fn palette_from_pose(&self, pose: &[ScaBonePose], aim: AimPose) -> Vec<[[f32; 4]; 4]> {
        let refs: Vec<SkinBoneRef<'_>> = self
            .bones
            .iter()
            .map(|b| SkinBoneRef {
                name: &b.name,
                parent_index: b.parent_index,
                position: b.position,
                rotation_wxyz: b.rotation,
                inv_bind_pose: b.inv_bind_pose,
            })
            .collect();
        let yaw = wrap_angle_pi(aim.yaw);
        let locals: Vec<[[f32; 4]; 4]> = (0..self.bones.len())
            .map(|i| {
                let p = pose.get(i).copied().unwrap_or(ScaBonePose {
                    position: self.bones[i].position,
                    rotation_wxyz: self.bones[i].rotation,
                });
                let mut m = mat4_from_trs_wxyz(p.position, p.rotation_wxyz);
                if yaw.abs() > 1e-4 && self.controls.yaw_bones.contains(&i) {
                    m = mat4_mul(&m, &mat4_rot_y(yaw));
                }
                if aim.pitch.abs() > 1e-4 && self.controls.pitch_bones.contains(&i) {
                    // `mat4_rot_x(+a)` dips local +Z; elevation is the opposite rotation.
                    m = mat4_mul(&m, &mat4_rot_x(-aim.pitch));
                }
                if aim.recoil.abs() > 1e-4
                    && let Some((_, d)) = self.controls.recoil_bones.iter().find(|(b, _)| *b == i)
                {
                    let mut t = mat4_identity();
                    t[3][2] = d * aim.recoil;
                    m = mat4_mul(&m, &t);
                }
                m
            })
            .collect();
        skin_from_local_mats(&refs, &locals, self.norm_offset, self.norm_scale)
    }

    /// Clip time for `mode` at render time `t_sec` with a per-unit `phase` in `[0, 1)`.
    pub fn clip_time(&self, mode: AnimMode, t_sec: f32, phase: f32) -> f32 {
        let (clip, rate) = match mode {
            AnimMode::Walk => (self.walk.as_ref(), self.walk_rate),
            AnimMode::Idle => (self.idle.as_ref(), 1.0),
            AnimMode::Rest => (None, 0.0),
        };
        let Some(c) = clip else {
            return 0.0;
        };
        let d = c.duration();
        wrap_anim_time(t_sec * rate + phase.clamp(0.0, 1.0) * d, d)
    }

    /// Skinning palette (one matrix per rig bone) for `mode` at clip time `clip_t`
    /// plus a relative turret yaw in radians.
    #[allow(dead_code)]
    pub fn palette(&self, mode: AnimMode, clip_t: f32, turret_yaw: f32) -> Vec<[[f32; 4]; 4]> {
        let pose = self.sample_pose(self.clip_for(mode), clip_t, true);
        self.palette_from_pose(&pose, AimPose { yaw: turret_yaw, ..AimPose::default() })
    }
}

/// CPU rigid skinning (`.scm` vertices use their first bone index) of a single position.
pub fn skin_point(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

/// CPU rigid skinning of a direction vector (ignores translation).
pub fn skin_dir(m: &[[f32; 4]; 4], d: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * d[0] + m[1][0] * d[1] + m[2][0] * d[2],
        m[0][1] * d[0] + m[1][1] * d[1] + m[2][1] * d[2],
        m[0][2] * d[0] + m[1][2] * d[1] + m[2][2] * d[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_synthetic_sca() -> Vec<u8> {
        let mut out = Vec::new();
        let names = b"Root\0Turret\0";
        let names_off = 36u32;
        let links_off = names_off + names.len() as u32;
        let anim_off = links_off + 8;
        let frame_size = 8 + 2 * 28u32;

        out.extend_from_slice(b"ANIM");
        out.extend_from_slice(&5u32.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes()); // frames
        out.extend_from_slice(&1.0f32.to_le_bytes()); // duration
        out.extend_from_slice(&2u32.to_le_bytes()); // bones
        out.extend_from_slice(&names_off.to_le_bytes());
        out.extend_from_slice(&links_off.to_le_bytes());
        out.extend_from_slice(&anim_off.to_le_bytes());
        out.extend_from_slice(&frame_size.to_le_bytes());

        out.extend_from_slice(names);
        out.extend_from_slice(&(-1i32).to_le_bytes());
        out.extend_from_slice(&0i32.to_le_bytes());

        // Root delta
        for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] {
            out.extend_from_slice(&v.to_le_bytes());
        }

        // Frame 0 at t=0.0
        out.extend_from_slice(&0.0f32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for v in [
            0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // Root
            0.0f32, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, // Turret
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }

        // Frame 1 at t=1.0: Turret moves +2m in X and rotates 90 deg around Y (w=c=0.7071, y=0.7071)
        let s = std::f32::consts::FRAC_1_SQRT_2;
        out.extend_from_slice(&1.0f32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for v in [
            0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // Root
            2.0f32, 2.0, 0.0, s, 0.0, s, 0.0, // Turret
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn parses_synthetic_sca_and_interpolates_slerp() {
        let bytes = build_synthetic_sca();
        let sca = ScaAnimation::parse(&bytes).expect("parse synthetic sca");
        assert_eq!(sca.version, 5);
        assert_eq!(sca.bone_names, vec!["Root", "Turret"]);
        assert_eq!(sca.parent_links, vec![-1, 0]);
        assert_eq!(sca.frame_count(), 2);

        let mid = sca.sample_local(0.5, false);
        assert!((mid[1].position[0] - 1.0).abs() < 1e-4);
        assert!((mid[1].position[1] - 2.0).abs() < 1e-4);
        let q = mid[1].rotation_wxyz;
        // 45 deg around Y: cos(22.5)=0.9239, sin(22.5)=0.3827
        assert!((q[0] - 0.9239).abs() < 1e-3);
        assert!((q[2] - 0.3827).abs() < 1e-3);
    }

    #[test]
    fn rejects_invalid_or_truncated_sca() {
        assert!(ScaAnimation::parse(b"NOPE0000").is_err());
        let mut bad = build_synthetic_sca();
        bad.truncate(40);
        assert!(ScaAnimation::parse(&bad).is_err());
    }

    #[test]
    fn skin_matrices_rest_is_identity_and_turret_propagates_to_child() {
        // Bone 0 at (0, 1, 0); Bone 1 (Turret) child at +(0, 1, 0) -> bind global (0, 2, 0);
        // Bone 2 (Muzzle) child of Turret at +(0, 0, 1) -> bind global (0, 2, 1).
        let bones = vec![
            SkinBoneRef {
                name: "Hull",
                parent_index: -1,
                position: [0.0, 1.0, 0.0],
                rotation_wxyz: [1.0, 0.0, 0.0, 0.0],
                inv_bind_pose: [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, -1.0, 0.0, 1.0,
                ],
            },
            SkinBoneRef {
                name: "Turret",
                parent_index: 0,
                position: [0.0, 1.0, 0.0],
                rotation_wxyz: [1.0, 0.0, 0.0, 0.0],
                inv_bind_pose: [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, -2.0, 0.0, 1.0,
                ],
            },
            SkinBoneRef {
                name: "Muzzle",
                parent_index: 1,
                position: [0.0, 0.0, 1.0],
                rotation_wxyz: [1.0, 0.0, 0.0, 0.0],
                inv_bind_pose: [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, -2.0, -1.0, 1.0,
                ],
            },
        ];

        // Rest pose must be identity even with normalization offset & scale
        let rest = compute_skin_matrices(&bones, None, None, None, [1.0, 0.5, -2.0], 0.5);
        for (bi, m) in rest.iter().enumerate() {
            for c in 0..4 {
                for r in 0..4 {
                    let expected = if c == r { 1.0 } else { 0.0 };
                    assert!(
                        (m[c][r] - expected).abs() < 1e-4,
                        "bone {bi} [{c}][{r}] = {} != {expected}",
                        m[c][r]
                    );
                }
            }
        }

        // Rotate Turret +90 deg around Y (unit scale at origin): Muzzle at bind (0, 2, 1) -> (1, 2, 0)
        let rot = compute_skin_matrices(
            &bones,
            None,
            Some(("turret", std::f32::consts::FRAC_PI_2)),
            None,
            [0.0, 0.0, 0.0],
            1.0,
        );
        let m = &rot[2];
        let vx = m[0][0] * 0.0 + m[1][0] * 2.0 + m[2][0] * 1.0 + m[3][0];
        let vy = m[0][1] * 0.0 + m[1][1] * 2.0 + m[2][1] * 1.0 + m[3][1];
        let vz = m[0][2] * 0.0 + m[1][2] * 2.0 + m[2][2] * 1.0 + m[3][2];
        assert!((vx - 1.0).abs() < 1e-4 && (vy - 2.0).abs() < 1e-4 && vz.abs() < 1e-4);
    }

    #[test]
    fn anim_mode_selection_and_time_wrapping() {
        assert_eq!(select_anim_mode(true, true, true), AnimMode::Walk);
        assert_eq!(select_anim_mode(true, false, true), AnimMode::Idle);
        assert_eq!(select_anim_mode(false, true, true), AnimMode::Idle);
        assert_eq!(select_anim_mode(false, true, false), AnimMode::Rest);
        assert_eq!(select_anim_mode(true, false, false), AnimMode::Rest);

        assert!((wrap_anim_time(2.5, 1.0) - 0.5).abs() < 1e-5);
        assert!((wrap_anim_time(-0.25, 1.0) - 0.75).abs() < 1e-5);
        assert_eq!(wrap_anim_time(3.0, 0.0), 0.0);
        assert_eq!(wrap_anim_time(f32::NAN, 1.0), 0.0);
        assert_eq!(wrap_anim_time(f32::INFINITY, 2.0), 0.0);
        let w = wrap_anim_time(1e9, 2.53);
        assert!((0.0..2.53).contains(&w));

        let pi = std::f32::consts::PI;
        assert!((wrap_angle_pi(3.0 * pi) - pi).abs() < 1e-4);
        assert!((wrap_angle_pi(-pi / 2.0) + pi / 2.0).abs() < 1e-5);
        assert!((wrap_angle_pi(2.0 * pi + 0.1) - 0.1).abs() < 1e-4);
    }

    #[test]
    fn bone_remap_by_name_is_case_insensitive() {
        let target = ["UEL0001", "Torso", "Head", "Missing"];
        let source = ["torso", "uel0001", "HEAD"];
        assert_eq!(
            remap_bones_by_name(&target, &source),
            vec![Some(1), Some(0), Some(2), None]
        );
        let sca = ScaAnimation::parse(&build_synthetic_sca()).unwrap();
        assert!(AnimClip::new(sca.clone(), &["Nope"]).is_none());
        let clip = AnimClip::new(sca, &["turret", "root"]).unwrap();
        assert_eq!(clip.remap, vec![Some(1), Some(0)]);
    }

    #[test]
    fn unit_rig_walk_palette_differs_from_rest_and_loops() {
        let id = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mk = |name: &str, parent: i32, pos: [f32; 3], inv_t: [f32; 3]| crate::scm::ScmBone {
            inv_bind_pose: [
                id[0], id[1], id[2], id[3], id[4], id[5], id[6], id[7], id[8], id[9], id[10], id[11],
                inv_t[0], inv_t[1], inv_t[2], 1.0,
            ],
            name: name.to_string(),
            parent_index: parent,
            position: pos,
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        let bones = vec![
            mk("Root", -1, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            mk("Turret", 0, [0.0, 2.0, 0.0], [0.0, -2.0, 0.0]),
        ];
        let sca = ScaAnimation::parse(&build_synthetic_sca()).unwrap();
        let clip = AnimClip::new(sca, &["Root", "Turret"]).unwrap();
        let mut rig = UnitRig::new(bones, [0.0; 3], 1.0);
        rig.walk = Some(clip);
        rig.walk_rate = 2.0;
        assert!(rig.is_animated());
        assert_eq!(rig.mode_for(true), AnimMode::Walk);
        assert_eq!(rig.mode_for(false), AnimMode::Rest);

        // walk_rate 2 => 0.25 s render time = 0.5 s clip time; phase 0.5 adds half a cycle.
        assert!((rig.clip_time(AnimMode::Walk, 0.25, 0.0) - 0.5).abs() < 1e-5);
        assert!((rig.clip_time(AnimMode::Walk, 0.25, 0.5) - 0.0).abs() < 1e-5);
        assert_eq!(rig.clip_time(AnimMode::Rest, 5.0, 0.3), 0.0);

        let rest = rig.palette(AnimMode::Rest, 0.0, 0.0);
        let walk = rig.palette(AnimMode::Walk, 0.5, 0.0);
        // Bind-pose vertex on the turret bone at (0, 2, 0): rest keeps it, mid-walk moves +1 in X.
        let r = skin_point(&rest[1], [0.0, 2.0, 0.0]);
        let w = skin_point(&walk[1], [0.0, 2.0, 0.0]);
        assert!(r[0].abs() < 1e-4 && (r[1] - 2.0).abs() < 1e-4);
        assert!((w[0] - 1.0).abs() < 1e-3 && (w[1] - 2.0).abs() < 1e-3);
        // Clip time 0 equals the first keyframe (bind-equivalent here).
        let start = rig.palette(AnimMode::Walk, 0.0, 0.0);
        let s0 = skin_point(&start[1], [0.0, 2.0, 1.0]);
        assert!(s0[0].abs() < 1e-4 && (s0[2] - 1.0).abs() < 1e-4);
        // Normals ignore translation.
        let n = skin_dir(&walk[1], [0.0, 0.0, 1.0]);
        assert!((n[0] * n[0] + n[1] * n[1] + n[2] * n[2] - 1.0).abs() < 1e-3);
    }

    /// Hull (root) -> Turret (yaw, at y=2) -> Barrel (pitch + recoil rack, at +z 1 from turret).
    fn turret_rig() -> UnitRig {
        let mk = |name: &str, parent: i32, pos: [f32; 3], g: [f32; 3]| crate::scm::ScmBone {
            inv_bind_pose: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -g[0], -g[1], -g[2], 1.0,
            ],
            name: name.to_string(),
            parent_index: parent,
            position: pos,
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        let bones = vec![
            mk("Hull", -1, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            mk("Turret", 0, [0.0, 2.0, 0.0], [0.0, 2.0, 0.0]),
            mk("Turret_Barrel", 1, [0.0, 0.0, 1.0], [0.0, 2.0, 1.0]),
        ];
        let mut rig = UnitRig::new(bones, [0.0; 3], 1.0);
        rig.set_weapon_bones(&["turret", "Missing"], &["TURRET_BARREL", "Turret"], &[("Turret_Barrel", -2.0)], 10.0);
        rig
    }

    #[test]
    fn weapon_bones_resolve_and_recoil_clamps() {
        let rig = turret_rig();
        assert_eq!(rig.controls.yaw_bones, vec![1]);
        // `Turret` is already a yaw bone, so it is not also pitched.
        assert_eq!(rig.controls.pitch_bones, vec![2]);
        // -2 exceeds 15% of a 10-unit span and is clamped to -1.5.
        assert_eq!(rig.controls.recoil_bones, vec![(2, -1.5)]);
        assert!(rig.has_turret() && rig.is_animated());
    }

    #[test]
    fn turret_pitch_raises_barrel_and_recoil_slides_back() {
        let rig = turret_rig();
        let rest = rig.palette_from_pose(&rig.bind_pose(), AimPose::default());
        let tip = [0.0, 2.0, 3.0]; // barrel vertex 2 units in front of the barrel pivot (0, 2, 1)
        let r = skin_point(&rest[2], tip);
        assert!((r[1] - 2.0).abs() < 1e-4 && (r[2] - 3.0).abs() < 1e-4);

        // +30 deg elevation lifts the muzzle: y = 2 + 2 sin 30 = 3, z = 1 + 2 cos 30.
        let up = rig.palette_from_pose(&rig.bind_pose(), AimPose { pitch: 30f32.to_radians(), ..AimPose::default() });
        let p = skin_point(&up[2], tip);
        assert!((p[1] - 3.0).abs() < 1e-3, "pitched tip {p:?}");
        assert!((p[2] - (1.0 + 2.0 * 30f32.to_radians().cos())).abs() < 1e-3);
        // Pitch never moves the turret ring itself.
        let ring = skin_point(&up[1], [0.5, 2.0, 0.0]);
        assert!((ring[1] - 2.0).abs() < 1e-4);

        // Full recoil slides the barrel 1.5 units back along its local Z.
        let kick = rig.palette_from_pose(&rig.bind_pose(), AimPose { recoil: 1.0, ..AimPose::default() });
        let k = skin_point(&kick[2], tip);
        assert!((k[2] - 1.5).abs() < 1e-4 && (k[1] - 2.0).abs() < 1e-4);

        // Yaw +90 deg swings the barrel tip from +Z to +X.
        let yawed = rig.palette_from_pose(&rig.bind_pose(), AimPose { yaw: std::f32::consts::FRAC_PI_2, ..AimPose::default() });
        let y = skin_point(&yawed[2], tip);
        assert!((y[0] - 3.0).abs() < 1e-3 && y[2].abs() < 1e-3);
    }

    #[test]
    fn aim_pitch_direct_and_ballistic() {
        // Direct fire at a target 10 up over 10 across = 45 deg.
        let p = aim_pitch(10.0, 10.0, 0.0, -0.5, 1.5);
        assert!((p - std::f32::consts::FRAC_PI_4).abs() < 1e-4);
        // Downhill shots dip, but not below the minimum.
        assert!((aim_pitch(10.0, -100.0, 0.0, -0.3, 1.0) + 0.3).abs() < 1e-6);
        // Ballistic arc of height d/4 launches at 45 deg on flat ground.
        let b = aim_pitch(40.0, 0.0, 10.0, -0.3, 1.5);
        assert!((b - std::f32::consts::FRAC_PI_4).abs() < 1e-4);
        // Clamped by the turret's pitch range.
        assert!((aim_pitch(1.0, 50.0, 0.0, -0.3, 0.5) - 0.5).abs() < 1e-6);
        assert_eq!(aim_pitch(0.0, 0.0, 0.0, -0.3, 0.5), 0.0);
    }

    #[test]
    fn recoil_curve_kicks_and_settles() {
        assert_eq!(recoil_curve(-0.01, 0.4), 0.0);
        assert_eq!(recoil_curve(0.0, 0.4), 1.0);
        assert!((recoil_curve(0.2, 0.4) - 0.25).abs() < 1e-6);
        assert_eq!(recoil_curve(0.4, 0.4), 0.0);
        assert_eq!(recoil_curve(f32::NAN, 0.4), 0.0);
        // Monotonically decreasing after the kick.
        let mut last = 1.0;
        for i in 1..40 {
            let v = recoil_curve(i as f32 * 0.01, 0.4);
            assert!(v <= last);
            last = v;
        }
    }

    #[test]
    fn crossfade_and_pose_blend() {
        assert_eq!(crossfade_weight(0.0, 0.2), 0.0);
        assert!((crossfade_weight(0.1, 0.2) - 0.5).abs() < 1e-6);
        assert_eq!(crossfade_weight(0.5, 0.2), 1.0);
        assert_eq!(crossfade_weight(0.1, 0.0), 1.0);
        assert!(crossfade_weight(0.05, 0.2) < 0.5);

        let a = vec![ScaBonePose { position: [0.0, 0.0, 0.0], rotation_wxyz: [1.0, 0.0, 0.0, 0.0] }];
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let b = vec![ScaBonePose { position: [2.0, 4.0, 0.0], rotation_wxyz: [s, 0.0, s, 0.0] }];
        let mid = blend_poses(&a, &b, 0.5);
        assert!((mid[0].position[0] - 1.0).abs() < 1e-6 && (mid[0].position[1] - 2.0).abs() < 1e-6);
        // Halfway between identity and 90 deg about Y = 45 deg.
        assert!((mid[0].rotation_wxyz[0] - 0.9239).abs() < 1e-3);
        assert_eq!(blend_poses(&a, &b, 0.0)[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(blend_poses(&a, &b, 1.0)[0].position, [2.0, 4.0, 0.0]);
    }

    #[test]
    fn slew_angle_limits_rate_and_wraps() {
        assert!((slew_angle(0.0, 1.0, 0.25) - 0.25).abs() < 1e-6);
        assert!((slew_angle(0.0, 0.1, 0.25) - 0.1).abs() < 1e-6);
        // Shortest way round across +-PI.
        let pi = std::f32::consts::PI;
        let s = slew_angle(pi - 0.1, -pi + 0.1, 0.05);
        assert!((s - (pi - 0.05)).abs() < 1e-4);
    }

    #[test]
    fn one_shot_clip_clamps_and_reports_duration() {
        let sca = ScaAnimation::parse(&build_synthetic_sca()).unwrap();
        let clip = AnimClip::new(sca, &["Turret"]).unwrap();
        let death = OneShotClip { clip: clip.clone(), rate: 2.0 };
        assert!((death.play_secs() - 0.5).abs() < 1e-6);
        let mut rig = UnitRig::new(
            vec![crate::scm::ScmBone {
                inv_bind_pose: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                name: "Turret".into(),
                parent_index: -1,
                position: [0.0, 2.0, 0.0],
                rotation: [1.0, 0.0, 0.0, 0.0],
            }],
            [0.0; 3],
            1.0,
        );
        rig.death.push(death);
        // Past the end a one-shot holds its last frame (x = 2) instead of wrapping to 0.
        let held = rig.sample_pose(Some(&clip), 5.0, false);
        assert!((held[0].position[0] - 2.0).abs() < 1e-4);
        let wrapped = rig.sample_pose(Some(&clip), 5.0, true);
        assert!(wrapped[0].position[0].abs() < 1e-4);
    }
}
