//! Phase 11 GPU billboard particle VFX system (muzzle flashes, explosions, nuclear shockwaves,
//! smoke trails, build sparks, and shield impacts).

use bytemuck::{Pod, Zeroable};

/// Maximum concurrent live particles across the scene.
pub const MAX_PARTICLES: usize = 4096;

/// GPU vertex for billboarded & ground-ring VFX particles.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct VfxVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    /// `0.0` = additive fire/spark glow, `1.0` = alpha smoke puff, `2.0` = expanding shockwave ring.
    pub mode: f32,
}

#[derive(Debug, Clone)]
pub struct Particle {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub gravity: f32,
    pub size_start: f32,
    pub size_end: f32,
    pub color_start: [f32; 4],
    pub color_end: [f32; 4],
    pub age: f32,
    pub lifetime: f32,
    pub mode: f32,
}

/// Purely visual billboard particle emitter pool (never alters simulation determinism).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct VfxSystem {
    particles: Vec<Particle>,
    rng: u64,
}

impl Default for VfxSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl VfxSystem {
    pub fn new() -> Self {
        Self {
            particles: Vec::with_capacity(512),
            rng: 0x517C_C1B7_2722_0A95,
        }
    }

    pub fn active_count(&self) -> usize {
        self.particles.len()
    }

    pub fn clear(&mut self) {
        self.particles.clear();
    }

    #[inline]
    fn rand01(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        ((self.rng >> 40) as f32) / ((1u32 << 24) as f32)
    }

    #[inline]
    fn rand_signed(&mut self) -> f32 {
        self.rand01() * 2.0 - 1.0
    }

    fn push_particle(&mut self, p: Particle, high_priority: bool) {
        if self.particles.len() >= MAX_PARTICLES {
            if !high_priority {
                return;
            }
            // Evict oldest low/normal particle so explosions & rings still spawn cleanly.
            self.particles.swap_remove(0);
        }
        self.particles.push(p);
    }

    /// Emits a sharp bright muzzle flash + tiny puff along `dir`.
    pub fn spawn_muzzle_flash(&mut self, pos: [f32; 3], dir: [f32; 3], scale: f32) {
        let s = scale.clamp(0.4, 4.0);
        let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
            .sqrt()
            .max(1e-4);
        let dx = dir[0] / norm;
        let dy = dir[1] / norm;
        let dz = dir[2] / norm;

        self.push_particle(
            Particle {
                pos: [pos[0] + dx * 0.4 * s, pos[1] + dy * 0.4 * s, pos[2] + dz * 0.4 * s],
                vel: [dx * 2.5 * s, dy * 2.5 * s + 0.5, dz * 2.5 * s],
                gravity: 0.0,
                size_start: 0.85 * s,
                size_end: 0.15 * s,
                color_start: [1.0, 0.90, 0.48, 0.95],
                color_end: [1.0, 0.42, 0.08, 0.0],
                age: 0.0,
                lifetime: 0.16,
                mode: 0.0,
            },
            false,
        );
    }

    /// Emits a multi-layered explosion: flash core, radial hot sparks, smoke plume, and ground ring.
    pub fn spawn_explosion(&mut self, pos: [f32; 3], radius: f32) {
        let r = radius.clamp(1.0, 18.0);
        self.push_particle(
            Particle {
                pos,
                vel: [0.0, 0.6, 0.0],
                gravity: 0.0,
                size_start: r * 1.4,
                size_end: r * 2.6,
                color_start: [1.0, 0.88, 0.45, 1.0],
                color_end: [1.0, 0.25, 0.04, 0.0],
                age: 0.0,
                lifetime: 0.35,
                mode: 0.0,
            },
            true,
        );

        // Expanding shockwave ring on the ground
        self.push_particle(
            Particle {
                pos: [pos[0], pos[1] + 0.22, pos[2]],
                vel: [0.0, 0.0, 0.0],
                gravity: 0.0,
                size_start: r * 0.6,
                size_end: r * 3.8,
                color_start: [1.0, 0.72, 0.32, 0.85],
                color_end: [1.0, 0.45, 0.15, 0.0],
                age: 0.0,
                lifetime: 0.55,
                mode: 2.0,
            },
            true,
        );

        let spark_count = ((r * 2.5) as usize).clamp(4, 14);
        for _ in 0..spark_count {
            let vx = self.rand_signed() * r * 3.2;
            let vy = (0.4 + self.rand01() * 1.1) * r * 2.4;
            let vz = self.rand_signed() * r * 3.2;
            let life = 0.45 + self.rand01() * 0.4;
            self.push_particle(
                Particle {
                    pos,
                    vel: [vx, vy, vz],
                    gravity: -9.8,
                    size_start: 0.28 + r * 0.08,
                    size_end: 0.06,
                    color_start: [1.0, 0.80, 0.28, 0.98],
                    color_end: [0.95, 0.22, 0.02, 0.0],
                    age: 0.0,
                    lifetime: life,
                    mode: 0.0,
                },
                false,
            );
        }

        let smoke_count = ((r * 1.4) as usize).clamp(3, 8);
        for _ in 0..smoke_count {
            let ox = self.rand_signed() * r * 0.45;
            let oz = self.rand_signed() * r * 0.45;
            let vy = 1.4 + self.rand01() * 1.6;
            let life = 1.1 + self.rand01() * 0.6;
            self.push_particle(
                Particle {
                    pos: [pos[0] + ox, pos[1] + 0.3, pos[2] + oz],
                    vel: [ox * 0.5, vy, oz * 0.5],
                    gravity: 0.4,
                    size_start: r * 0.9,
                    size_end: r * 2.4,
                    color_start: [0.22, 0.20, 0.19, 0.68],
                    color_end: [0.12, 0.12, 0.13, 0.0],
                    age: 0.0,
                    lifetime: life,
                    mode: 1.0,
                },
                false,
            );
        }
    }

    /// Emits a massive Forged Alliance style nuclear / ACU detonation effect.
    pub fn spawn_acu_nuke(&mut self, pos: [f32; 3]) {
        self.push_particle(
            Particle {
                pos: [pos[0], pos[1] + 1.5, pos[2]],
                vel: [0.0, 2.0, 0.0],
                gravity: 0.0,
                size_start: 6.0,
                size_end: 26.0,
                color_start: [1.0, 0.98, 0.82, 1.0],
                color_end: [1.0, 0.40, 0.06, 0.0],
                age: 0.0,
                lifetime: 1.35,
                mode: 0.0,
            },
            true,
        );

        // Dual concentric ground shockwaves
        for (size_end, life, col) in [
            (48.0, 1.8, [1.0, 0.88, 0.58, 0.95]),
            (32.0, 1.2, [0.85, 0.94, 1.0, 0.78]),
        ] {
            self.push_particle(
                Particle {
                    pos: [pos[0], pos[1] + 0.35, pos[2]],
                    vel: [0.0, 0.0, 0.0],
                    gravity: 0.0,
                    size_start: 2.0,
                    size_end,
                    color_start: col,
                    color_end: [col[0], col[1], col[2], 0.0],
                    age: 0.0,
                    lifetime: life,
                    mode: 2.0,
                },
                true,
            );
        }

        // Rising fireball stem & mushroom cap smoke
        for i in 0..24 {
            let frac = i as f32 / 24.0;
            let angle = frac * std::f32::consts::TAU;
            let (s, c) = angle.sin_cos();
            let is_fire = i % 3 == 0;
            let v_out_1 = 2.5 + self.rand01() * 3.0;
            let v_up = 4.5 + self.rand01() * 4.5;
            let v_out_2 = 2.5 + self.rand01() * 3.0;
            let life = if is_fire { 1.8 } else { 2.8 + self.rand01() * 0.9 };
            self.push_particle(
                Particle {
                    pos: [pos[0] + c * 1.8, pos[1] + 1.0 + frac * 6.0, pos[2] + s * 1.8],
                    vel: [c * v_out_1, v_up, s * v_out_2],
                    gravity: 0.3,
                    size_start: if is_fire { 4.2 } else { 5.0 },
                    size_end: if is_fire { 10.5 } else { 15.0 },
                    color_start: if is_fire {
                        [1.0, 0.65, 0.16, 0.92]
                    } else {
                        [0.24, 0.21, 0.19, 0.78]
                    },
                    color_end: [0.10, 0.09, 0.09, 0.0],
                    age: 0.0,
                    lifetime: life,
                    mode: if is_fire { 0.0 } else { 1.0 },
                },
                true,
            );
        }
    }

    /// Emits engineering build / repair welding sparks tinted slightly toward team color.
    pub fn spawn_build_spark(&mut self, pos: [f32; 3], team_rgb: [f32; 3]) {
        for _ in 0..2 {
            let vx = self.rand_signed() * 2.2;
            let vy = 1.0 + self.rand01() * 2.8;
            let vz = self.rand_signed() * 2.2;
            self.push_particle(
                Particle {
                    pos,
                    vel: [vx, vy, vz],
                    gravity: -6.5,
                    size_start: 0.45,
                    size_end: 0.10,
                    color_start: [
                        0.45 + team_rgb[0] * 0.55,
                        0.72 + team_rgb[1] * 0.28,
                        0.90 + team_rgb[2] * 0.10,
                        0.90,
                    ],
                    color_end: [team_rgb[0], team_rgb[1], 1.0, 0.0],
                    age: 0.0,
                    lifetime: 0.32,
                    mode: 0.0,
                },
                false,
            );
        }
    }

    /// Emits an electric shield impact shimmer.
    pub fn spawn_shield_hit(&mut self, pos: [f32; 3], team_rgb: [f32; 3]) {
        self.push_particle(
            Particle {
                pos,
                vel: [0.0, 0.3, 0.0],
                gravity: 0.0,
                size_start: 2.2,
                size_end: 4.5,
                color_start: [
                    (team_rgb[0] + 0.4).min(1.0),
                    (team_rgb[1] + 0.6).min(1.0),
                    1.0,
                    0.92,
                ],
                color_end: [0.35, 0.78, 1.0, 0.0],
                age: 0.0,
                lifetime: 0.28,
                mode: 0.0,
            },
            false,
        );
    }

    /// Emits a dense missile exhaust puff.
    pub fn spawn_missile_trail(&mut self, pos: [f32; 3]) {
        let vx = self.rand_signed() * 0.4;
        let vz = self.rand_signed() * 0.4;
        self.push_particle(
            Particle {
                pos,
                vel: [vx, 0.3, vz],
                gravity: 0.1,
                size_start: 0.65,
                size_end: 1.9,
                color_start: [0.85, 0.86, 0.88, 0.62],
                color_end: [0.45, 0.47, 0.50, 0.0],
                age: 0.0,
                lifetime: 0.75,
                mode: 1.0,
            },
            false,
        );
    }

    /// Emits a thin, long-lived condensation puff behind an aircraft wingtip.
    pub fn spawn_contrail(&mut self, pos: [f32; 3], size: f32) {
        let s = size.clamp(0.2, 3.0);
        let vx = self.rand_signed() * 0.08;
        let vz = self.rand_signed() * 0.08;
        self.push_particle(
            Particle {
                pos,
                vel: [vx, 0.05, vz],
                gravity: 0.0,
                size_start: 0.45 * s,
                size_end: 1.6 * s,
                color_start: [0.94, 0.95, 0.98, 0.42],
                color_end: [0.86, 0.88, 0.92, 0.0],
                age: 0.0,
                lifetime: 1.9,
                mode: 1.0,
            },
            false,
        );
    }

    /// Emits a flat foam patch on the water surface (`mode 3`) drifting with `vel`.
    pub fn spawn_wake(&mut self, pos: [f32; 3], vel: [f32; 3], size: f32) {
        let s = size.clamp(0.2, 8.0);
        let jx = self.rand_signed() * 0.15;
        let jz = self.rand_signed() * 0.15;
        self.push_particle(
            Particle {
                pos,
                vel: [vel[0] + jx, 0.0, vel[2] + jz],
                gravity: 0.0,
                size_start: 0.7 * s,
                size_end: 2.6 * s,
                color_start: [0.90, 0.95, 1.0, 0.55],
                color_end: [0.80, 0.90, 0.96, 0.0],
                age: 0.0,
                lifetime: 2.2,
                mode: 3.0,
            },
            false,
        );
    }

    /// Emits a few rising bubbles/spray puffs above a sinking wreck.
    pub fn spawn_bubbles(&mut self, pos: [f32; 3]) {
        for _ in 0..2 {
            let ox = self.rand_signed() * 0.8;
            let oz = self.rand_signed() * 0.8;
            let vy = 0.6 + self.rand01() * 0.9;
            self.push_particle(
                Particle {
                    pos: [pos[0] + ox, pos[1], pos[2] + oz],
                    vel: [ox * 0.2, vy, oz * 0.2],
                    gravity: -0.4,
                    size_start: 0.35,
                    size_end: 1.1,
                    color_start: [0.82, 0.90, 0.96, 0.55],
                    color_end: [0.70, 0.80, 0.88, 0.0],
                    age: 0.0,
                    lifetime: 0.9,
                    mode: 1.0,
                },
                false,
            );
        }
    }

    /// Advances all live particles by `dt_sec` seconds and removes expired ones.
    pub fn step(&mut self, dt_sec: f32) {
        if dt_sec <= 0.0 {
            return;
        }
        let mut i = 0;
        while i < self.particles.len() {
            let p = &mut self.particles[i];
            p.age += dt_sec;
            if p.age >= p.lifetime {
                self.particles.swap_remove(i);
                continue;
            }
            p.vel[1] += p.gravity * dt_sec;
            p.pos[0] += p.vel[0] * dt_sec;
            p.pos[1] += p.vel[1] * dt_sec;
            p.pos[2] += p.vel[2] * dt_sec;
            i += 1;
        }
    }

    /// Builds camera-facing billboard quads (`mode 0/1`) and horizontal ground shockwave quads
    /// (`mode 2`) ready for the GPU `vfx_pipeline`.
    pub fn build_billboards(
        &self,
        cam_right: [f32; 3],
        cam_up: [f32; 3],
    ) -> (Vec<VfxVertex>, Vec<u32>) {
        let mut verts = Vec::with_capacity(self.particles.len() * 4);
        let mut indices = Vec::with_capacity(self.particles.len() * 6);

        for p in &self.particles {
            let t = if p.lifetime > 1e-5 {
                (p.age / p.lifetime).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let half = 0.5 * (p.size_start + (p.size_end - p.size_start) * t);
            let col = [
                p.color_start[0] + (p.color_end[0] - p.color_start[0]) * t,
                p.color_start[1] + (p.color_end[1] - p.color_start[1]) * t,
                p.color_start[2] + (p.color_end[2] - p.color_start[2]) * t,
                p.color_start[3] + (p.color_end[3] - p.color_start[3]) * t,
            ];

            let (rx, ux) = if p.mode >= 1.5 {
                ([half, 0.0, 0.0], [0.0, 0.0, half])
            } else {
                (
                    [cam_right[0] * half, cam_right[1] * half, cam_right[2] * half],
                    [cam_up[0] * half, cam_up[1] * half, cam_up[2] * half],
                )
            };

            let base = verts.len() as u32;
            let c = p.pos;
            verts.extend_from_slice(&[
                VfxVertex {
                    position: [c[0] - rx[0] - ux[0], c[1] - rx[1] - ux[1], c[2] - rx[2] - ux[2]],
                    uv: [0.0, 0.0],
                    color: col,
                    mode: p.mode,
                },
                VfxVertex {
                    position: [c[0] + rx[0] - ux[0], c[1] + rx[1] - ux[1], c[2] + rx[2] - ux[2]],
                    uv: [1.0, 0.0],
                    color: col,
                    mode: p.mode,
                },
                VfxVertex {
                    position: [c[0] + rx[0] + ux[0], c[1] + rx[1] + ux[1], c[2] + rx[2] + ux[2]],
                    uv: [1.0, 1.0],
                    color: col,
                    mode: p.mode,
                },
                VfxVertex {
                    position: [c[0] - rx[0] + ux[0], c[1] - rx[1] + ux[1], c[2] - rx[2] + ux[2]],
                    uv: [0.0, 1.0],
                    color: col,
                    mode: p.mode,
                },
            ]);
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        (verts, indices)
    }
}

/// Appends one camera-facing glow billboard (`mode 0`) that lives for this frame only.
pub fn append_glow(
    verts: &mut Vec<VfxVertex>,
    indices: &mut Vec<u32>,
    pos: [f32; 3],
    size: f32,
    color: [f32; 4],
    cam_right: [f32; 3],
    cam_up: [f32; 3],
) {
    let h = size * 0.5;
    let r = [cam_right[0] * h, cam_right[1] * h, cam_right[2] * h];
    let u = [cam_up[0] * h, cam_up[1] * h, cam_up[2] * h];
    let base = verts.len() as u32;
    for (sx, sy, uv) in [(-1.0, -1.0, [0.0, 0.0]), (1.0, -1.0, [1.0, 0.0]), (1.0, 1.0, [1.0, 1.0]), (-1.0, 1.0, [0.0, 1.0])] {
        verts.push(VfxVertex {
            position: [
                pos[0] + r[0] * sx + u[0] * sy,
                pos[1] + r[1] * sx + u[1] * sy,
                pos[2] + r[2] * sx + u[2] * sy,
            ],
            uv,
            color,
            mode: 0.0,
        });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// Appends a view-aligned glowing streak quad from `a` to `b` (`width` across), e.g. a shell
/// tracer. The radial glow mask in UV space stretches into an elongated ellipse along the quad.
pub fn append_streak(
    verts: &mut Vec<VfxVertex>,
    indices: &mut Vec<u32>,
    a: [f32; 3],
    b: [f32; 3],
    width: f32,
    color: [f32; 4],
    eye: [f32; 3],
) {
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
    let v = [eye[0] - mid[0], eye[1] - mid[1], eye[2] - mid[2]];
    let mut s = [d[1] * v[2] - d[2] * v[1], d[2] * v[0] - d[0] * v[2], d[0] * v[1] - d[1] * v[0]];
    let sl = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
    if sl < 1e-6 {
        return;
    }
    let hw = width * 0.5 / sl;
    s = [s[0] * hw, s[1] * hw, s[2] * hw];
    let base = verts.len() as u32;
    for (p, side, uv) in [(a, -1.0, [0.0, 0.0]), (b, -1.0, [1.0, 0.0]), (b, 1.0, [1.0, 1.0]), (a, 1.0, [0.0, 1.0])] {
        verts.push(VfxVertex {
            position: [p[0] + s[0] * side, p[1] + s[1] * side, p[2] + s[2] * side],
            uv,
            color,
            mode: 0.0,
        });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn particles_spawn_advance_and_expire() {
        let mut vfx = VfxSystem::new();
        vfx.spawn_explosion([10.0, 5.0, 20.0], 3.0);
        assert!(vfx.active_count() >= 6);
        for _ in 0..10 {
            vfx.step(0.05);
        }
        assert!(vfx.active_count() > 0);
        for _ in 0..80 {
            vfx.step(0.05);
        }
        assert_eq!(vfx.active_count(), 0);
    }

    #[test]
    fn enforces_max_particle_cap() {
        let mut vfx = VfxSystem::new();
        for i in 0..6000 {
            vfx.spawn_build_spark([i as f32 * 0.1, 1.0, 0.0], [0.2, 0.6, 1.0]);
        }
        assert_eq!(vfx.active_count(), MAX_PARTICLES);
    }

    #[test]
    fn builds_billboard_and_shockwave_geometry() {
        let mut vfx = VfxSystem::new();
        vfx.spawn_acu_nuke([50.0, 2.0, 50.0]);
        let count = vfx.active_count();
        assert!(count > 10);

        let (verts, idx) = vfx.build_billboards([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert_eq!(verts.len(), count * 4);
        assert_eq!(idx.len(), count * 6);

        let mut has_glow = false;
        let mut has_smoke = false;
        let mut has_ring = false;
        for chunk in verts.chunks_exact(4) {
            match chunk[0].mode as i32 {
                0 => has_glow = true,
                1 => has_smoke = true,
                2 => {
                    has_ring = true;
                    // Ground shockwave quads keep constant Y across corners.
                    assert!((chunk[0].position[1] - chunk[2].position[1]).abs() < 1e-4);
                }
                other => panic!("unexpected vfx mode {other}"),
            }
        }
        assert!(has_glow && has_smoke && has_ring);
    }

    #[test]
    fn wakes_are_flat_and_streaks_span_endpoints() {
        let mut vfx = VfxSystem::new();
        vfx.spawn_wake([5.0, 1.0, 5.0], [1.0, 0.0, 0.0], 1.0);
        vfx.spawn_contrail([0.0, 20.0, 0.0], 1.0);
        vfx.spawn_bubbles([0.0, 1.0, 0.0]);
        assert_eq!(vfx.active_count(), 4);
        vfx.step(0.5);
        let (verts, _) = vfx.build_billboards([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let foam: Vec<_> = verts.chunks_exact(4).filter(|c| c[0].mode == 3.0).collect();
        assert_eq!(foam.len(), 1);
        // Foam lies flat on the water and drifted downstream (+X).
        assert!(foam[0].iter().all(|v| (v.position[1] - 1.0).abs() < 1e-4));
        assert!(foam[0].iter().map(|v| v.position[0]).sum::<f32>() / 4.0 > 5.2);

        let (mut v, mut i) = (Vec::new(), Vec::new());
        append_streak(&mut v, &mut i, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], 1.0, [1.0; 4], [5.0, 0.0, 20.0]);
        assert_eq!((v.len(), i.len()), (4, 6));
        // Ends sit on the endpoints, offset half the width sideways (here: along Y).
        assert!((v[0].position[0]).abs() < 1e-5 && (v[1].position[0] - 10.0).abs() < 1e-5);
        assert!((v[2].position[1] - v[1].position[1]).abs() - 1.0 < 1e-4);
        // Degenerate streak looking straight down its own axis is skipped.
        append_streak(&mut v, &mut i, [0.0; 3], [0.0, 0.0, 1.0], 1.0, [1.0; 4], [0.0, 0.0, 5.0]);
        assert_eq!(v.len(), 4);
        append_glow(&mut v, &mut i, [1.0, 2.0, 3.0], 2.0, [1.0; 4], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert_eq!(v.len(), 8);
        assert!((v[4].position[0] - 0.0).abs() < 1e-6 && (v[6].position[1] - 3.0).abs() < 1e-6);
    }
}
