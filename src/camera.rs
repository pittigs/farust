/// Orbit and strategic-zoom camera producing column-major matrices for WebGPU (`[0, 1]` depth).
#[derive(Debug, Clone)]
pub struct OrbitCamera {
    pub target: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub min_distance: f32,
    pub max_distance: f32,
    pub fov_y_radians: f32,
    pub aspect: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl OrbitCamera {
    /// Frames an axis-aligned bounding box from a 3/4 hero angle (used for unit meshes).
    pub fn for_bounds(min: [f32; 3], max: [f32; 3], aspect: f32) -> Self {
        let center = [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ];
        let half_extents = [
            (max[0] - min[0]) * 0.5,
            (max[1] - min[1]) * 0.5,
            (max[2] - min[2]) * 0.5,
        ];
        let radius = half_extents[0]
            .hypot(half_extents[1])
            .hypot(half_extents[2])
            .max(0.25);

        Self {
            target: center,
            yaw: 0.60,
            pitch: 0.38,
            distance: radius * 2.3,
            min_distance: (radius * 0.15).max(0.05),
            max_distance: radius * 25.0,
            fov_y_radians: 45.0_f32.to_radians(),
            aspect: aspect.max(0.1),
            z_near: (radius * 0.02).max(0.01),
            z_far: (radius * 60.0).max(100.0),
        }
    }

    /// Frames a map of size `(map_width, map_height)` at strategic zoom around `focus_y`.
    pub fn for_map(map_width: f32, map_height: f32, focus_y: f32, aspect: f32) -> Self {
        let span = map_width.max(map_height).max(64.0);

        Self {
            target: [map_width * 0.5, focus_y, map_height * 0.5],
            yaw: 0.0,
            pitch: 1.12,
            distance: span * 1.25,
            min_distance: 4.0,
            max_distance: span * 3.5,
            fov_y_radians: 45.0_f32.to_radians(),
            aspect: aspect.max(0.1),
            z_near: 1.0,
            z_far: span * 6.0,
        }
    }

    /// Updates the viewport aspect ratio when both dimensions are non-zero.
    pub fn set_aspect(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.aspect = width as f32 / height as f32;
        }
    }

    /// Rotates the camera around `target` by `(delta_yaw, delta_pitch)` in radians,
    /// clamping `pitch` to `[-1.52, 1.52]` to avoid pole singularities.
    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-1.52, 1.52);
    }

    /// Translates `target` along the horizontal XZ camera basis derived from `yaw`.
    ///
    /// Both `right_delta` and `forward_delta` are expressed as fractions of the
    /// current `distance` (for example, `0.05` pans by 5% of `self.distance`).
    pub fn pan(&mut self, right_delta: f32, forward_delta: f32) {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let right = [cos_yaw, 0.0, -sin_yaw];
        let forward = [-sin_yaw, 0.0, -cos_yaw];
        let scale = self.distance;

        self.target[0] += (right[0] * right_delta + forward[0] * forward_delta) * scale;
        self.target[2] += (right[2] * right_delta + forward[2] * forward_delta) * scale;
    }

    /// Adjusts `distance` exponentially by `wheel_steps` (positive zooms in, negative zooms out),
    /// clamping to `[min_distance, max_distance]`.
    pub fn zoom(&mut self, wheel_steps: f32) {
        self.distance =
            (self.distance * 0.85_f32.powf(wheel_steps)).clamp(self.min_distance, self.max_distance);
    }

    /// Computes the world-space camera position (`+Y` up, right-handed).
    pub fn eye_position(&self) -> [f32; 3] {
        let cp = self.pitch.cos();
        [
            self.target[0] + self.distance * cp * self.yaw.sin(),
            self.target[1] + self.distance * self.pitch.sin(),
            self.target[2] + self.distance * cp * self.yaw.cos(),
        ]
    }

    /// Returns the column-major right-handed view matrix (`m[col][row]`).
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        look_at_rh(self.eye_position(), self.target, [0.0, 1.0, 0.0])
    }

    /// Returns the column-major right-handed WebGPU perspective matrix with `[0, 1]` depth.
    pub fn proj_matrix(&self) -> [[f32; 4]; 4] {
        let f = 1.0 / (0.5 * self.fov_y_radians).tan();
        let aspect = self.aspect.max(1e-6);
        let depth_inv = 1.0 / (self.z_near - self.z_far);

        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, self.z_far * depth_inv, -1.0],
            [0.0, 0.0, (self.z_near * self.z_far) * depth_inv, 0.0],
        ]
    }

    /// Returns `proj_matrix() * view_matrix()` in column-major layout (`m[col][row]`).
    pub fn view_proj_matrix(&self) -> [[f32; 4]; 4] {
        mul_mat4(self.proj_matrix(), self.view_matrix())
    }

    /// Casts a world-space ray `(origin, unit_dir)` from normalized device
    /// coordinates `(ndc_x, ndc_y)` in `[-1, 1]^2` (`+Y` up, `+X` right).
    pub fn screen_ray(&self, ndc_x: f32, ndc_y: f32) -> ([f32; 3], [f32; 3]) {
        let eye = self.eye_position();
        let forward = normalize3_or(sub3(self.target, eye), [0.0, 0.0, -1.0]);
        let right_raw = cross3(forward, [0.0, 1.0, 0.0]);
        let right = if dot3(right_raw, right_raw) > 1e-12 {
            normalize3_or(right_raw, [1.0, 0.0, 0.0])
        } else {
            normalize3_or(cross3(forward, [0.0, 0.0, -1.0]), [1.0, 0.0, 0.0])
        };
        let up = cross3(right, forward);
        let tan_half_y = (0.5 * self.fov_y_radians).tan();
        let tan_half_x = tan_half_y * self.aspect.max(1e-6);
        let sx = ndc_x * tan_half_x;
        let sy = ndc_y * tan_half_y;
        let dir = normalize3_or(
            [
                forward[0] + right[0] * sx + up[0] * sy,
                forward[1] + right[1] * sx + up[1] * sy,
                forward[2] + right[2] * sx + up[2] * sy,
            ],
            forward,
        );
        (eye, dir)
    }

    /// Intersects the ray through `(ndc_x, ndc_y)` with the horizontal plane
    /// `y = plane_y`, returning the hit point when in front of the camera.
    pub fn ray_plane_y(&self, ndc_x: f32, ndc_y: f32, plane_y: f32) -> Option<[f32; 3]> {
        let (eye, dir) = self.screen_ray(ndc_x, ndc_y);
        if dir[1].abs() <= 1e-6 {
            return None;
        }
        let t = (plane_y - eye[1]) / dir[1];
        if t <= 0.0 {
            return None;
        }
        Some([
            eye[0] + dir[0] * t,
            plane_y,
            eye[2] + dir[2] * t,
        ])
    }
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len_sq = dot3(v, v);
    if len_sq > 1e-12 {
        let inv = 1.0 / len_sq.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        fallback
    }
}

fn look_at_rh(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize3_or(sub3(center, eye), [0.0, 0.0, -1.0]);
    let s_raw = cross3(f, up);
    let s = if dot3(s_raw, s_raw) > 1e-12 {
        normalize3_or(s_raw, [1.0, 0.0, 0.0])
    } else {
        normalize3_or(cross3(f, [0.0, 0.0, -1.0]), [1.0, 0.0, 0.0])
    };
    let u = cross3(s, f);

    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot3(s, eye), -dot3(u, eye), dot3(f, eye), 1.0],
    ]
}

fn mul_mat4(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0_f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transform_point_ndc(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
        let mut clip = [0.0_f32; 4];
        let v = [p[0], p[1], p[2], 1.0];
        for row in 0..4 {
            clip[row] = m[0][row] * v[0]
                + m[1][row] * v[1]
                + m[2][row] * v[2]
                + m[3][row] * v[3];
        }
        assert!(clip[3] > 0.0, "expected positive clip.w, got {}", clip[3]);
        [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]]
    }

    #[test]
    fn target_projects_to_ndc_center_with_valid_webgpu_depth() {
        let cam = OrbitCamera::for_bounds([-2.0, 0.0, -1.5], [4.0, 3.0, 2.5], 16.0 / 9.0);
        let vp = cam.view_proj_matrix();
        let ndc = transform_point_ndc(vp, cam.target);

        assert!(ndc[0].abs() < 1e-5, "expected x_ndc ~ 0, got {}", ndc[0]);
        assert!(ndc[1].abs() < 1e-5, "expected y_ndc ~ 0, got {}", ndc[1]);
        assert!(
            ndc[2] > 0.0 && ndc[2] < 1.0,
            "expected z_ndc in (0, 1), got {}",
            ndc[2]
        );
    }

    #[test]
    fn closer_point_has_smaller_ndc_depth_than_target() {
        let cam = OrbitCamera::for_map(512.0, 512.0, 20.0, 1.6);
        let vp = cam.view_proj_matrix();
        let eye = cam.eye_position();
        let midpoint = [
            (eye[0] + cam.target[0]) * 0.5,
            (eye[1] + cam.target[1]) * 0.5,
            (eye[2] + cam.target[2]) * 0.5,
        ];

        let target_ndc = transform_point_ndc(vp, cam.target);
        let closer_ndc = transform_point_ndc(vp, midpoint);

        assert!(closer_ndc[0].abs() < 1e-5);
        assert!(closer_ndc[1].abs() < 1e-5);
        assert!(closer_ndc[2] > 0.0 && closer_ndc[2] < 1.0);
        assert!(
            closer_ndc[2] < target_ndc[2],
            "expected closer z_ndc ({}) < target z_ndc ({})",
            closer_ndc[2],
            target_ndc[2]
        );
    }

    #[test]
    fn near_and_far_planes_map_to_zero_and_one_in_ndc() {
        let cam = OrbitCamera::for_bounds([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 1.5);
        let eye = cam.eye_position();
        let forward = normalize3_or(sub3(cam.target, eye), [0.0, 0.0, -1.0]);
        let near_pt = [
            eye[0] + forward[0] * cam.z_near,
            eye[1] + forward[1] * cam.z_near,
            eye[2] + forward[2] * cam.z_near,
        ];
        let far_pt = [
            eye[0] + forward[0] * cam.z_far,
            eye[1] + forward[1] * cam.z_far,
            eye[2] + forward[2] * cam.z_far,
        ];

        let vp = cam.view_proj_matrix();
        let near_ndc = transform_point_ndc(vp, near_pt);
        let far_ndc = transform_point_ndc(vp, far_pt);

        assert!(near_ndc[2].abs() < 1e-5, "expected near z_ndc ~ 0, got {}", near_ndc[2]);
        assert!((far_ndc[2] - 1.0).abs() < 1e-5, "expected far z_ndc ~ 1, got {}", far_ndc[2]);
    }

    #[test]
    fn orbit_clamps_pitch_and_accumulates_yaw() {
        let mut cam = OrbitCamera::for_bounds([-1.0; 3], [1.0; 3], 1.0);
        let initial_yaw = cam.yaw;

        cam.orbit(0.75, 10.0);
        assert!((cam.yaw - (initial_yaw + 0.75)).abs() < 1e-6);
        assert!((cam.pitch - 1.52).abs() < 1e-6);

        cam.orbit(-1.25, -20.0);
        assert!((cam.yaw - (initial_yaw - 0.5)).abs() < 1e-6);
        assert!((cam.pitch - (-1.52)).abs() < 1e-6);
    }

    #[test]
    fn zoom_scales_exponentially_and_clamps_to_limits() {
        let mut cam = OrbitCamera::for_map(256.0, 256.0, 10.0, 1.0);
        let start_dist = cam.distance;

        cam.zoom(1.0);
        assert!((cam.distance - start_dist * 0.85).abs() < 1e-4);

        cam.zoom(1000.0);
        assert!((cam.distance - cam.min_distance).abs() < 1e-6);

        cam.zoom(-1000.0);
        assert!((cam.distance - cam.max_distance).abs() < 1e-6);
    }

    #[test]
    fn pan_moves_target_in_horizontal_yaw_basis_scaled_by_distance() {
        let mut cam = OrbitCamera::for_map(100.0, 200.0, 15.0, 1.0);
        cam.yaw = 0.0;
        cam.distance = 100.0;
        let orig = cam.target;

        // At yaw = 0, camera is at +Z looking toward -Z: right is +X, forward is -Z.
        cam.pan(0.10, 0.25);
        assert!((cam.target[0] - (orig[0] + 10.0)).abs() < 1e-5);
        assert!((cam.target[1] - orig[1]).abs() < 1e-6);
        assert!((cam.target[2] - (orig[2] - 25.0)).abs() < 1e-5);

        // Rotate yaw by +pi/2: camera is at +X looking toward -X: right is -Z, forward is -X.
        cam.target = orig;
        cam.yaw = core::f32::consts::FRAC_PI_2;
        cam.pan(0.10, 0.20);
        assert!((cam.target[0] - (orig[0] - 20.0)).abs() < 1e-4);
        assert!((cam.target[1] - orig[1]).abs() < 1e-6);
        assert!((cam.target[2] - (orig[2] - 10.0)).abs() < 1e-4);
    }

    #[test]
    fn for_bounds_and_for_map_and_set_aspect_follow_specifications() {
        let unit_cam = OrbitCamera::for_bounds([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0);
        assert_eq!(unit_cam.target, [0.0, 0.0, 0.0]);
        assert!((unit_cam.distance - 0.25 * 2.3).abs() < 1e-6);
        assert!((unit_cam.min_distance - 0.05).abs() < 1e-6);
        assert!((unit_cam.max_distance - 0.25 * 25.0).abs() < 1e-6);
        assert!((unit_cam.aspect - 0.1).abs() < 1e-6);
        assert!((unit_cam.z_near - 0.01).abs() < 1e-6);
        assert!((unit_cam.z_far - 100.0).abs() < 1e-6);

        let mut map_cam = OrbitCamera::for_map(32.0, 48.0, 12.5, 0.02);
        // span clamps up to 64.0
        assert_eq!(map_cam.target, [16.0, 12.5, 24.0]);
        assert!((map_cam.distance - 64.0 * 1.25).abs() < 1e-5);
        assert!((map_cam.min_distance - 4.0).abs() < 1e-6);
        assert!((map_cam.max_distance - 64.0 * 3.5).abs() < 1e-5);
        assert!((map_cam.pitch - 1.12).abs() < 1e-6);
        assert!((map_cam.z_near - 1.0).abs() < 1e-6);
        assert!((map_cam.z_far - 64.0 * 6.0).abs() < 1e-5);
        assert!((map_cam.aspect - 0.1).abs() < 1e-6);

        map_cam.set_aspect(0, 1080);
        assert!((map_cam.aspect - 0.1).abs() < 1e-6);
        map_cam.set_aspect(1920, 1080);
        assert!((map_cam.aspect - (1920.0 / 1080.0)).abs() < 1e-6);
    }

    #[test]
    fn screen_ray_inverts_view_proj_across_corners() {
        let cam = OrbitCamera::for_map(1024.0, 1024.0, 22.0, 16.0 / 9.0);
        let vp = cam.view_proj_matrix();
        for (nx, ny) in [(0.0, 0.0), (-0.6, 0.4), (0.75, -0.5)] {
            let (eye, dir) = cam.screen_ray(nx, ny);
            let p = [
                eye[0] + dir[0] * 120.0,
                eye[1] + dir[1] * 120.0,
                eye[2] + dir[2] * 120.0,
            ];
            let ndc = transform_point_ndc(vp, p);
            assert!(
                (ndc[0] - nx).abs() < 1e-4 && (ndc[1] - ny).abs() < 1e-4,
                "expected ({nx}, {ny}), got ({}, {})",
                ndc[0],
                ndc[1]
            );
        }
    }

    #[test]
    fn ray_plane_y_hits_target_plane_at_center() {
        let mut cam = OrbitCamera::for_map(512.0, 512.0, 26.0, 1.5);
        cam.yaw = 0.4;
        cam.pitch = 0.9;
        let hit = cam
            .ray_plane_y(0.0, 0.0, cam.target[1])
            .expect("expected plane hit");
        assert!((hit[0] - cam.target[0]).abs() < 1e-3);
        assert!((hit[1] - cam.target[1]).abs() < 1e-3);
        assert!((hit[2] - cam.target[2]).abs() < 1e-3);
    }
}
