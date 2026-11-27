//! Phase 11 & Phase 40 GPU billboard particle VFX system (muzzle flashes, impacts, explosions,
//! nuclear shockwaves, smoke/projectile trails, build beams & sparks, and shield impacts),
//! driven by Forged Alliance `EmitterBlueprint` / `BeamBlueprint` / `TrailEmitterBlueprint`
//! curves and textures with procedural fallbacks.

use bytemuck::{Pod, Zeroable};

use crate::effects::{ColorRamp, CompiledBeam, CompiledEmitter, CompiledTrail, MOHO_TICK_HZ};

/// Maximum concurrent live particles across the scene.
pub const MAX_PARTICLES: usize = 4096;

/// GPU vertex for billboarded, ground-ring, and textured FA VFX particles & beams.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct VfxVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    /// `0.0` = procedural additive glow, `1.0` = procedural alpha smoke puff,
    /// `2.0` = procedural expanding shockwave ring, `3.0` = procedural water foam wake,
    /// `4.0` = FA textured alpha billboard, `5.0` = FA textured additive billboard,
    /// `6.0` = FA textured beam/trail alpha (repeat UV), `7.0` = FA textured beam/trail additive (repeat UV).
    pub mode: f32,
    /// Layer index into the GPU VFX `texture_2d_array` (`0` for procedural / fallback sprite).
    pub tex_layer: u32,
}

#[derive(Debug, Clone)]
pub struct Particle {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub accel: [f32; 3],
    pub size_start: f32,
    pub size_end: f32,
    /// 4-stop RGBA color ramp evaluated over `[0.0, 1.0]` normalized age.
    pub ramp: ColorRamp,
    pub age: f32,
    pub lifetime: f32,
    pub rotation: f32,
    pub rot_rate: f32,
    pub mode: f32,
    pub tex_layer: u32,
    /// Sub-UV rectangle `[u0, v0, u1, v1]` for sprite-sheet frames/strips.
    pub uv_rect: [f32; 4],
    /// Horizontal ground-aligned quad (`Flat = true` in FA emitter or shockwave/wake).
    pub flat: bool,
}

impl Particle {
    #[allow(clippy::too_many_arguments)]
    fn from_two_colors(
        pos: [f32; 3],
        vel: [f32; 3],
        gravity: f32,
        size_start: f32,
        size_end: f32,
        color_start: [f32; 4],
        color_end: [f32; 4],
        lifetime: f32,
        mode: f32,
    ) -> Self {
        let lerp = |t: f32| {
            [
                color_start[0] + (color_end[0] - color_start[0]) * t,
                color_start[1] + (color_end[1] - color_start[1]) * t,
                color_start[2] + (color_end[2] - color_start[2]) * t,
                color_start[3] + (color_end[3] - color_start[3]) * t,
            ]
        };
        Self {
            pos,
            vel,
            accel: [0.0, gravity, 0.0],
            size_start,
            size_end,
            ramp: ColorRamp {
                stops: [color_start, lerp(1.0 / 3.0), lerp(2.0 / 3.0), color_end],
            },
            age: 0.0,
            lifetime,
            rotation: 0.0,
            rot_rate: 0.0,
            mode,
            tex_layer: 0,
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            flat: mode >= 1.5,
        }
    }
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

    /// Evaluates one FA `CompiledEmitter`'s curves and pushes its burst of textured particles
    /// into the GPU pool, respecting `MAX_PARTICLES`.
    pub fn spawn_emitter(
        &mut self,
        em: &CompiledEmitter,
        pos: [f32; 3],
        dir: [f32; 3],
        scale: f32,
        tint: Option<[f32; 3]>,
        high_priority: bool,
    ) {
        let bp = &em.bp;
        let s = scale.clamp(0.25, 8.0);
        let dlen = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        let fwd = if dlen > 1e-4 {
            [dir[0] / dlen, dir[1] / dlen, dir[2] / dlen]
        } else {
            [0.0, 0.0, 1.0]
        };
        // Local tangent basis `(right, up, fwd)` when `LocalVelocity` is enabled.
        let up_ref = if fwd[1].abs() < 0.95 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
        let rx = up_ref[1] * fwd[2] - up_ref[2] * fwd[1];
        let ry = up_ref[2] * fwd[0] - up_ref[0] * fwd[2];
        let rz = up_ref[0] * fwd[1] - up_ref[1] * fwd[0];
        let rlen = (rx * rx + ry * ry + rz * rz).sqrt().max(1e-4);
        let right = [rx / rlen, ry / rlen, rz / rlen];
        let up = [
            fwd[1] * right[2] - fwd[2] * right[1],
            fwd[2] * right[0] - fwd[0] * right[2],
            fwd[0] * right[1] - fwd[1] * right[0],
        ];

        let count = em.burst_count.clamp(1, 5);
        let x_span = bp.repeat_time.max(bp.lifetime.max(0.1));
        let strips = bp.texture_strip_count.max(1);
        let frames = bp.texture_frame_count.max(1);

        let mode = if em.has_texture {
            if bp.blend_mode.is_additive() { 5.0 } else { 4.0 }
        } else if bp.flat {
            2.0
        } else if bp.blend_mode.is_additive() {
            0.0
        } else {
            1.0
        };

        for idx in 0..count {
            let sample_x = if count == 1 {
                x_span * 0.5
            } else {
                x_span * (idx as f32 / (count - 1) as f32)
            };

            let life_ticks = bp
                .lifetime_curve
                .sample(sample_x, self.rand_signed())
                .abs();
            let lifetime = (life_ticks / MOHO_TICK_HZ).clamp(0.18, 3.2);

            let speed_ticks = bp.velocity.sample(sample_x, self.rand_signed());
            let speed = (speed_ticks * MOHO_TICK_HZ * 0.5).clamp(-40.0, 40.0);

            let dx = bp.x_dir.sample(sample_x, self.rand_signed());
            let dy = bp.y_dir.sample(sample_x, self.rand_signed());
            let dz = bp.z_dir.sample(sample_x, self.rand_signed());
            let vel = if bp.local_velocity {
                [
                    (right[0] * dx + up[0] * dy + fwd[0] * dz) * speed,
                    (right[1] * dx + up[1] * dy + fwd[1] * dz) * speed,
                    (right[2] * dx + up[2] * dy + fwd[2] * dz) * speed,
                ]
            } else {
                [dx * speed, dy * speed, dz * speed]
            };

            let ox = bp.x_pos.sample(sample_x, self.rand_signed()) * s;
            let oy = bp.y_pos.sample(sample_x, self.rand_signed()) * s;
            let oz = bp.z_pos.sample(sample_x, self.rand_signed()) * s;
            let spawn_pos = [
                pos[0] + ox.clamp(-6.0 * s, 6.0 * s),
                pos[1] + oy.clamp(-2.0 * s, 6.0 * s) + if bp.flat { 0.18 } else { 0.0 },
                pos[2] + oz.clamp(-6.0 * s, 6.0 * s),
            ];

            let ax = bp.x_accel.sample(sample_x, self.rand_signed()) * MOHO_TICK_HZ;
            let mut ay = bp.y_accel.sample(sample_x, self.rand_signed()) * MOHO_TICK_HZ;
            let az = bp.z_accel.sample(sample_x, self.rand_signed()) * MOHO_TICK_HZ;
            if bp.gravity {
                ay -= 9.8;
            }

            let raw_start = bp.start_size.sample(sample_x, self.rand_signed()).abs();
            let raw_end = bp.end_size.sample(sample_x, self.rand_signed()).abs();
            let raw_size = bp.size.sample(sample_x, self.rand_signed()).abs();
            let (sz0, sz1) = if raw_start > 1e-4 || raw_end > 1e-4 {
                let s0 = raw_start.max(0.22);
                let s1 = if raw_end > 1e-4 {
                    raw_end
                } else if raw_size > 1e-4 {
                    (s0 + raw_size * 0.35).min(s0 * 3.0)
                } else {
                    s0 * 0.3
                };
                (s0, s1)
            } else if raw_size > 1e-4 {
                let base = (raw_size * 0.25).clamp(0.35, 6.0);
                (base, base * 1.4)
            } else {
                (0.7, 0.25)
            };
            // FA units are ~0.7m wide while FARust renders normalized units ~2.2x larger.
            let size_start = (sz0 * s * 2.4).clamp(0.45 * s, 24.0);
            let size_end = (sz1 * s * 2.4).clamp(0.15 * s, 38.0);

            let init_rot = bp
                .init_rot
                .sample(sample_x, self.rand_signed())
                .to_radians();
            let rot_rate = (bp.rot_rate.sample(sample_x, self.rand_signed()) * MOHO_TICK_HZ * 0.25)
                .to_radians();

            // Select strip / frame sub-UV if the sprite texture is a sheet.
            let uv_rect = if strips > 1 || frames > 1 {
                let sel = bp
                    .tex_select
                    .sample(sample_x, self.rand_signed())
                    .abs()
                    .floor() as u32;
                let strip_idx = sel % strips;
                let frame_idx = (sel / strips) % frames;
                let du = 1.0 / frames as f32;
                let dv = 1.0 / strips as f32;
                [
                    frame_idx as f32 * du,
                    strip_idx as f32 * dv,
                    (frame_idx + 1) as f32 * du,
                    (strip_idx + 1) as f32 * dv,
                ]
            } else {
                [0.0, 0.0, 1.0, 1.0]
            };

            let mut ramp = em.ramp;
            if let Some(t) = tint {
                for stop in &mut ramp.stops {
                    stop[0] = (stop[0] * (0.55 + 0.45 * t[0])).clamp(0.0, 1.5);
                    stop[1] = (stop[1] * (0.55 + 0.45 * t[1])).clamp(0.0, 1.5);
                    stop[2] = (stop[2] * (0.55 + 0.45 * t[2])).clamp(0.0, 1.5);
                }
            }
            if bp.blend_mode == crate::effects::BlendMode::Refract {
                for stop in &mut ramp.stops {
                    stop[3] *= 0.45;
                }
            }

            self.push_particle(
                Particle {
                    pos: spawn_pos,
                    vel,
                    accel: [ax.clamp(-25.0, 25.0), ay.clamp(-25.0, 25.0), az.clamp(-25.0, 25.0)],
                    size_start,
                    size_end,
                    ramp,
                    age: 0.0,
                    lifetime,
                    rotation: init_rot,
                    rot_rate,
                    mode,
                    tex_layer: if em.has_texture { em.tex_layer } else { 0 },
                    uv_rect,
                    flat: bp.flat,
                },
                high_priority,
            );
        }
    }

    /// Spawns a slice of FA emitters at `(pos, dir)` with a hard per-burst cap.
    pub fn spawn_emitters(
        &mut self,
        emitters: &[CompiledEmitter],
        pos: [f32; 3],
        dir: [f32; 3],
        scale: f32,
        tint: Option<[f32; 3]>,
        high_priority: bool,
    ) {
        for em in emitters.iter().take(4) {
            self.spawn_emitter(em, pos, dir, scale, tint, high_priority);
        }
    }

    /// Emits a sharp bright muzzle flash + tiny puff along `dir` (procedural fallback).
    pub fn spawn_muzzle_flash(&mut self, pos: [f32; 3], dir: [f32; 3], scale: f32) {
        let s = scale.clamp(0.4, 4.0);
        let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
            .sqrt()
            .max(1e-4);
        let dx = dir[0] / norm;
        let dy = dir[1] / norm;
        let dz = dir[2] / norm;

        self.push_particle(
            Particle::from_two_colors(
                [pos[0] + dx * 0.4 * s, pos[1] + dy * 0.4 * s, pos[2] + dz * 0.4 * s],
                [dx * 2.5 * s, dy * 2.5 * s + 0.5, dz * 2.5 * s],
                0.0,
                0.85 * s,
                0.15 * s,
                [1.0, 0.90, 0.48, 0.95],
                [1.0, 0.42, 0.08, 0.0],
                0.16,
                0.0,
            ),
            false,
        );
    }

    /// Emits a multi-layered explosion: flash core, radial hot sparks, smoke plume, and ground ring.
    pub fn spawn_explosion(&mut self, pos: [f32; 3], radius: f32) {
        let r = radius.clamp(1.0, 18.0);
        self.push_particle(
            Particle::from_two_colors(
                pos,
                [0.0, 0.6, 0.0],
                0.0,
                r * 1.4,
                r * 2.6,
                [1.0, 0.88, 0.45, 1.0],
                [1.0, 0.25, 0.04, 0.0],
                0.35,
                0.0,
            ),
            true,
        );

        // Expanding shockwave ring on the ground
        self.push_particle(
            Particle::from_two_colors(
                [pos[0], pos[1] + 0.22, pos[2]],
                [0.0, 0.0, 0.0],
                0.0,
                r * 0.6,
                r * 3.8,
                [1.0, 0.72, 0.32, 0.85],
                [1.0, 0.45, 0.15, 0.0],
                0.55,
                2.0,
            ),
            true,
        );

        let spark_count = ((r * 2.5) as usize).clamp(4, 14);
        for _ in 0..spark_count {
            let vx = self.rand_signed() * r * 3.2;
            let vy = (0.4 + self.rand01() * 1.1) * r * 2.4;
            let vz = self.rand_signed() * r * 3.2;
            let life = 0.45 + self.rand01() * 0.4;
            self.push_particle(
                Particle::from_two_colors(
                    pos,
                    [vx, vy, vz],
                    -9.8,
                    0.28 + r * 0.08,
                    0.06,
                    [1.0, 0.80, 0.28, 0.98],
                    [0.95, 0.22, 0.02, 0.0],
                    life,
                    0.0,
                ),
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
                Particle::from_two_colors(
                    [pos[0] + ox, pos[1] + 0.3, pos[2] + oz],
                    [ox * 0.5, vy, oz * 0.5],
                    0.4,
                    r * 0.9,
                    r * 2.4,
                    [0.22, 0.20, 0.19, 0.68],
                    [0.12, 0.12, 0.13, 0.0],
                    life,
                    1.0,
                ),
                false,
            );
        }
    }

    /// Emits a massive Forged Alliance style nuclear / ACU detonation effect.
    pub fn spawn_acu_nuke(&mut self, pos: [f32; 3]) {
        self.push_particle(
            Particle::from_two_colors(
                [pos[0], pos[1] + 1.5, pos[2]],
                [0.0, 2.0, 0.0],
                0.0,
                6.0,
                26.0,
                [1.0, 0.98, 0.82, 1.0],
                [1.0, 0.40, 0.06, 0.0],
                1.35,
                0.0,
            ),
            true,
        );

        // Dual concentric ground shockwaves
        for (size_end, life, col) in [
            (48.0, 1.8, [1.0, 0.88, 0.58, 0.95]),
            (32.0, 1.2, [0.85, 0.94, 1.0, 0.78]),
        ] {
            self.push_particle(
                Particle::from_two_colors(
                    [pos[0], pos[1] + 0.35, pos[2]],
                    [0.0, 0.0, 0.0],
                    0.0,
                    2.0,
                    size_end,
                    col,
                    [col[0], col[1], col[2], 0.0],
                    life,
                    2.0,
                ),
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
                Particle::from_two_colors(
                    [pos[0] + c * 1.8, pos[1] + 1.0 + frac * 6.0, pos[2] + s * 1.8],
                    [c * v_out_1, v_up, s * v_out_2],
                    0.3,
                    if is_fire { 4.2 } else { 5.0 },
                    if is_fire { 10.5 } else { 15.0 },
                    if is_fire {
                        [1.0, 0.65, 0.16, 0.92]
                    } else {
                        [0.24, 0.21, 0.19, 0.78]
                    },
                    [0.10, 0.09, 0.09, 0.0],
                    life,
                    if is_fire { 0.0 } else { 1.0 },
                ),
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
                Particle::from_two_colors(
                    pos,
                    [vx, vy, vz],
                    -6.5,
                    0.45,
                    0.10,
                    [
                        0.45 + team_rgb[0] * 0.55,
                        0.72 + team_rgb[1] * 0.28,
                        0.90 + team_rgb[2] * 0.10,
                        0.90,
                    ],
                    [team_rgb[0], team_rgb[1], 1.0, 0.0],
                    0.32,
                    0.0,
                ),
                false,
            );
        }
    }

    /// Emits an electric shield impact shimmer.
    pub fn spawn_shield_hit(&mut self, pos: [f32; 3], team_rgb: [f32; 3]) {
        self.push_particle(
            Particle::from_two_colors(
                pos,
                [0.0, 0.3, 0.0],
                0.0,
                2.2,
                4.5,
                [
                    (team_rgb[0] + 0.4).min(1.0),
                    (team_rgb[1] + 0.6).min(1.0),
                    1.0,
                    0.92,
                ],
                [0.35, 0.78, 1.0, 0.0],
                0.28,
                0.0,
            ),
            false,
        );
    }

    /// Emits a dense missile exhaust puff.
    pub fn spawn_missile_trail(&mut self, pos: [f32; 3]) {
        let vx = self.rand_signed() * 0.4;
        let vz = self.rand_signed() * 0.4;
        self.push_particle(
            Particle::from_two_colors(
                pos,
                [vx, 0.3, vz],
                0.1,
                0.65,
                1.9,
                [0.85, 0.86, 0.88, 0.62],
                [0.45, 0.47, 0.50, 0.0],
                0.75,
                1.0,
            ),
            false,
        );
    }

    /// Emits a thin, long-lived condensation puff behind an aircraft wingtip.
    pub fn spawn_contrail(&mut self, pos: [f32; 3], size: f32) {
        let s = size.clamp(0.2, 3.0);
        let vx = self.rand_signed() * 0.08;
        let vz = self.rand_signed() * 0.08;
        self.push_particle(
            Particle::from_two_colors(
                pos,
                [vx, 0.05, vz],
                0.0,
                0.45 * s,
                1.6 * s,
                [0.94, 0.95, 0.98, 0.42],
                [0.86, 0.88, 0.92, 0.0],
                1.9,
                1.0,
            ),
            false,
        );
    }

    /// Emits a flat foam patch on the water surface (`mode 3`) drifting with `vel`.
    pub fn spawn_wake(&mut self, pos: [f32; 3], vel: [f32; 3], size: f32) {
        let s = size.clamp(0.2, 8.0);
        let jx = self.rand_signed() * 0.15;
        let jz = self.rand_signed() * 0.15;
        self.push_particle(
            Particle::from_two_colors(
                pos,
                [vel[0] + jx, 0.0, vel[2] + jz],
                0.0,
                0.7 * s,
                2.6 * s,
                [0.90, 0.95, 1.0, 0.55],
                [0.80, 0.90, 0.96, 0.0],
                2.2,
                3.0,
            ),
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
                Particle::from_two_colors(
                    [pos[0] + ox, pos[1], pos[2] + oz],
                    [ox * 0.2, vy, oz * 0.2],
                    -0.4,
                    0.35,
                    1.1,
                    [0.82, 0.90, 0.96, 0.55],
                    [0.70, 0.80, 0.88, 0.0],
                    0.9,
                    1.0,
                ),
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
            p.vel[0] += p.accel[0] * dt_sec;
            p.vel[1] += p.accel[1] * dt_sec;
            p.vel[2] += p.accel[2] * dt_sec;
            p.pos[0] += p.vel[0] * dt_sec;
            p.pos[1] += p.vel[1] * dt_sec;
            p.pos[2] += p.vel[2] * dt_sec;
            p.rotation += p.rot_rate * dt_sec;
            i += 1;
        }
    }

    /// Builds camera-facing billboard quads and horizontal ground quads ready for `vfx_pipeline`.
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
            let col = p.ramp.sample(t);

            let (base_r, base_u) = if p.flat {
                ([half, 0.0, 0.0], [0.0, 0.0, half])
            } else {
                (
                    [cam_right[0] * half, cam_right[1] * half, cam_right[2] * half],
                    [cam_up[0] * half, cam_up[1] * half, cam_up[2] * half],
                )
            };

            let (rx, ux) = if p.rotation.abs() > 1e-4 {
                let (sr, cr) = p.rotation.sin_cos();
                (
                    [
                        base_r[0] * cr + base_u[0] * sr,
                        base_r[1] * cr + base_u[1] * sr,
                        base_r[2] * cr + base_u[2] * sr,
                    ],
                    [
                        -base_r[0] * sr + base_u[0] * cr,
                        -base_r[1] * sr + base_u[1] * cr,
                        -base_r[2] * sr + base_u[2] * cr,
                    ],
                )
            } else {
                (base_r, base_u)
            };

            let [u0, v0, u1, v1] = p.uv_rect;
            let base = verts.len() as u32;
            let c = p.pos;
            verts.extend_from_slice(&[
                VfxVertex {
                    position: [c[0] - rx[0] - ux[0], c[1] - rx[1] - ux[1], c[2] - rx[2] - ux[2]],
                    uv: [u0, v0],
                    color: col,
                    mode: p.mode,
                    tex_layer: p.tex_layer,
                },
                VfxVertex {
                    position: [c[0] + rx[0] - ux[0], c[1] + rx[1] - ux[1], c[2] + rx[2] - ux[2]],
                    uv: [u1, v0],
                    color: col,
                    mode: p.mode,
                    tex_layer: p.tex_layer,
                },
                VfxVertex {
                    position: [c[0] + rx[0] + ux[0], c[1] + rx[1] + ux[1], c[2] + rx[2] + ux[2]],
                    uv: [u1, v1],
                    color: col,
                    mode: p.mode,
                    tex_layer: p.tex_layer,
                },
                VfxVertex {
                    position: [c[0] - rx[0] + ux[0], c[1] - rx[1] + ux[1], c[2] - rx[2] + ux[2]],
                    uv: [u0, v1],
                    color: col,
                    mode: p.mode,
                    tex_layer: p.tex_layer,
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
            tex_layer: 0,
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
            tex_layer: 0,
        });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// Appends a view-aligned FA `BeamBlueprint` textured ribbon from `a` to `b`, scrolling UVs with `time_sec`.
#[allow(clippy::too_many_arguments)]
pub fn append_fa_beam(
    verts: &mut Vec<VfxVertex>,
    indices: &mut Vec<u32>,
    a: [f32; 3],
    b: [f32; 3],
    beam: &CompiledBeam,
    width_scale: f32,
    time_sec: f32,
    eye: [f32; 3],
    tint: Option<[f32; 3]>,
) {
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let seg_len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if seg_len < 1e-4 {
        return;
    }
    let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
    let v = [eye[0] - mid[0], eye[1] - mid[1], eye[2] - mid[2]];
    let mut s = [d[1] * v[2] - d[2] * v[1], d[2] * v[0] - d[0] * v[2], d[0] * v[1] - d[1] * v[0]];
    let sl = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
    if sl < 1e-6 {
        return;
    }
    let width = (beam.bp.thickness * 6.0 * width_scale).clamp(0.18, 3.0);
    let hw = width * 0.5 / sl;
    s = [s[0] * hw, s[1] * hw, s[2] * hw];

    let apply_tint = |c: [f32; 4]| -> [f32; 4] {
        let mut out = [c[0], c[1], c[2], c[3].max(0.55)];
        if let Some(t) = tint {
            out[0] = (out[0] * 0.65 + t[0] * 0.35).clamp(0.0, 1.5);
            out[1] = (out[1] * 0.65 + t[1] * 0.35).clamp(0.0, 1.5);
            out[2] = (out[2] * 0.65 + t[2] * 0.35).clamp(0.0, 1.5);
        }
        out
    };
    let c0 = apply_tint(beam.bp.start_color);
    let c1 = apply_tint(beam.bp.end_color);

    let v_rep = (seg_len * beam.bp.repeat_rate * 0.25).clamp(1.0, 16.0);
    let u_off = beam.bp.u_shift * time_sec * 4.0;
    let v_off = beam.bp.v_shift * time_sec * 4.0;
    let mode = if beam.has_texture {
        if beam.bp.blend_mode.is_additive() { 7.0 } else { 6.0 }
    } else {
        0.0
    };
    let tex_layer = if beam.has_texture { beam.tex_layer } else { 0 };

    let base = verts.len() as u32;
    verts.extend_from_slice(&[
        VfxVertex {
            position: [a[0] - s[0], a[1] - s[1], a[2] - s[2]],
            uv: [u_off, v_off],
            color: c0,
            mode,
            tex_layer,
        },
        VfxVertex {
            position: [a[0] + s[0], a[1] + s[1], a[2] + s[2]],
            uv: [1.0 + u_off, v_off],
            color: c0,
            mode,
            tex_layer,
        },
        VfxVertex {
            position: [b[0] + s[0], b[1] + s[1], b[2] + s[2]],
            uv: [1.0 + u_off, v_rep + v_off],
            color: c1,
            mode,
            tex_layer,
        },
        VfxVertex {
            position: [b[0] - s[0], b[1] - s[1], b[2] - s[2]],
            uv: [u_off, v_rep + v_off],
            color: c1,
            mode,
            tex_layer,
        },
    ]);
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// Appends a view-aligned FA `TrailEmitterBlueprint` ribbon from `tail` to `head`.
#[allow(clippy::too_many_arguments)]
pub fn append_fa_trail(
    verts: &mut Vec<VfxVertex>,
    indices: &mut Vec<u32>,
    tail: [f32; 3],
    head: [f32; 3],
    trail: &CompiledTrail,
    width_scale: f32,
    time_sec: f32,
    eye: [f32; 3],
) {
    let d = [head[0] - tail[0], head[1] - tail[1], head[2] - tail[2]];
    let seg_len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if seg_len < 1e-4 {
        return;
    }
    let mid = [(tail[0] + head[0]) * 0.5, (tail[1] + head[1]) * 0.5, (tail[2] + head[2]) * 0.5];
    let v = [eye[0] - mid[0], eye[1] - mid[1], eye[2] - mid[2]];
    let mut s = [d[1] * v[2] - d[2] * v[1], d[2] * v[0] - d[0] * v[2], d[0] * v[1] - d[1] * v[0]];
    let sl = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
    if sl < 1e-6 {
        return;
    }
    let width = (trail.bp.size * 2.5 * width_scale).clamp(0.16, 2.5);
    let hw = width * 0.5 / sl;
    s = [s[0] * hw, s[1] * hw, s[2] * hw];

    let c_head = trail.ramp.sample(0.15);
    let c_tail = trail.ramp.sample(0.95);
    let v_rep = (seg_len * trail.bp.texture_repeat_rate * 0.25).clamp(1.0, 12.0);
    let u_off = trail.bp.u_shift * time_sec * 3.0;
    let v_off = trail.bp.v_shift * time_sec * 3.0;
    let mode = if trail.has_texture {
        if trail.bp.blend_mode.is_additive() { 7.0 } else { 6.0 }
    } else {
        0.0
    };
    let tex_layer = if trail.has_texture { trail.tex_layer } else { 0 };

    let base = verts.len() as u32;
    verts.extend_from_slice(&[
        VfxVertex {
            position: [tail[0] - s[0], tail[1] - s[1], tail[2] - s[2]],
            uv: [u_off, v_off],
            color: c_tail,
            mode,
            tex_layer,
        },
        VfxVertex {
            position: [tail[0] + s[0], tail[1] + s[1], tail[2] + s[2]],
            uv: [1.0 + u_off, v_off],
            color: c_tail,
            mode,
            tex_layer,
        },
        VfxVertex {
            position: [head[0] + s[0], head[1] + s[1], head[2] + s[2]],
            uv: [1.0 + u_off, v_rep + v_off],
            color: c_head,
            mode,
            tex_layer,
        },
        VfxVertex {
            position: [head[0] - s[0], head[1] - s[1], head[2] - s[2]],
            uv: [u_off, v_rep + v_off],
            color: c_head,
            mode,
            tex_layer,
        },
    ]);
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

    #[test]
    fn spawns_fa_emitter_with_curves_texture_layer_and_beam() {
        let bp_text = r#"
EmitterBlueprint {
    BlueprintId = 'unit_test_emit',
    Lifetime = 1.0,
    Repeattime = 1.0,
    Blendmode = 3.0,
    LocalVelocity = true,
    Flat = false,
    TextureFramecount = 2.0,
    TextureStripcount = 2.0,
    Texture = [[/textures/particles/glow_02.dds]],
    RampTexture = [[/textures/particles/ramp_white_01.dds]],
    EmitRateCurve = { XRange = 1.0, Keys = { { x=0.5, y=3.0, z=0.0 } } },
    LifetimeCurve = { XRange = 1.0, Keys = { { x=0.5, y=5.0, z=0.0 } } },
    StartSizeCurve = { XRange = 1.0, Keys = { { x=0.5, y=1.2, z=0.0 } } },
    EndSizeCurve = { XRange = 1.0, Keys = { { x=0.5, y=2.4, z=0.0 } } },
    VelocityCurve = { XRange = 1.0, Keys = { { x=0.5, y=0.5, z=0.0 } } },
    ZDirectionCurve = { XRange = 1.0, Keys = { { x=0.5, y=1.0, z=0.0 } } },
}
"#;
        let crate::effects::EffectBp::Emitter(bp) = crate::effects::parse_effect_blueprint(bp_text).unwrap() else {
            panic!("expected emitter");
        };
        let compiled = CompiledEmitter {
            burst_count: crate::effects::emitter_burst_count(&bp),
            bp: *bp,
            tex_layer: 7,
            has_texture: true,
            ramp: ColorRamp::default(),
        };
        let mut vfx = VfxSystem::new();
        vfx.spawn_emitter(&compiled, [1.0, 2.0, 3.0], [1.0, 0.0, 0.0], 1.0, None, false);
        assert_eq!(vfx.active_count(), 3);
        let (verts, idx) = vfx.build_billboards([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert_eq!(verts.len(), 12);
        assert_eq!(idx.len(), 18);
        assert_eq!(verts[0].mode, 5.0);
        assert_eq!(verts[0].tex_layer, 7);

        let beam_bp_text = r#"
BeamBlueprint {
    Lifetime = 2,
    TextureName = '/textures/particles/build_beam_01.dds',
    Thickness = 0.05,
    StartColor = {x=0.2, y=0.4, z=1.0, w=0.8},
    EndColor = {x=0.2, y=0.4, z=1.0, w=0.4},
    Blendmode = 3,
    RepeatRate = 0.5,
    Length = 8,
    VShift = -0.5,
}
"#;
        let crate::effects::EffectBp::Beam(bbp) = crate::effects::parse_effect_blueprint(beam_bp_text).unwrap() else {
            panic!("expected beam");
        };
        let cbeam = CompiledBeam { bp: bbp, tex_layer: 4, has_texture: true };
        let (mut bv, mut bi) = (Vec::new(), Vec::new());
        append_fa_beam(&mut bv, &mut bi, [0.0, 1.0, 0.0], [10.0, 1.0, 0.0], &cbeam, 1.0, 0.5, [5.0, 20.0, 20.0], None);
        assert_eq!((bv.len(), bi.len()), (4, 6));
        assert_eq!(bv[0].mode, 7.0);
        assert_eq!(bv[0].tex_layer, 4);
    }
}
