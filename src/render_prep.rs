//! Phase 5 parallel render-frame preparation from a read-only simulation snapshot.

use crate::jobs::par_map_ordered;
use crate::scmap::ScMap;
use crate::sim::SimUnitSnapshot;

const FNV_OFFSET_BASIS: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct RenderInstance {
    pub unit_id: u32,
    pub model_matrix: [[f32; 4]; 4],
    pub world_pos: [f32; 3],
    pub health_ratio: f32,
    pub lod: u8,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct PreparedFrame {
    pub instances: Vec<RenderInstance>,
    pub culled_count: usize,
    pub lod_counts: [usize; 3],
    pub digest_fnv1a: u64,
}

#[inline]
fn fnv1a_extend(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

pub fn prepare_render_frame(
    units: &[SimUnitSnapshot],
    scmap: Option<&ScMap>,
    view_proj: [[f32; 4]; 4],
    camera_eye: [f32; 3],
    threads: usize,
) -> PreparedFrame {
    let scale = crate::sim::SCALE as f32;

    let results = par_map_ordered(units, threads, |_, u| {
        let wx = u.x as f32 / scale;
        let wz = u.y as f32 / scale;
        let wy = if let Some(map) = scmap {
            let gx = (wx.round() as i64).clamp(0, map.heightmap_width as i64) as u32;
            let gz = (wz.round() as i64).clamp(0, map.heightmap_height as i64) as u32;
            let terrain_y = map.height_at(gx, gz).unwrap_or(0.0);
            if map.water.has_water {
                terrain_y.max(map.water.elevation)
            } else {
                terrain_y
            }
        } else {
            0.0
        };

        let clip_row = |r: usize| -> f32 {
            view_proj[0][r] * wx + view_proj[1][r] * wy + view_proj[2][r] * wz + view_proj[3][r]
        };

        let clip_x = clip_row(0);
        let clip_y = clip_row(1);
        let clip_z = clip_row(2);
        let clip_w = clip_row(3);

        let m = 4.0_f32;
        let visible = clip_w > 0.0
            && clip_x >= -clip_w - m
            && clip_x <= clip_w + m
            && clip_y >= -clip_w - m
            && clip_y <= clip_w + m
            && clip_z >= -m
            && clip_z <= clip_w + m;

        if !visible {
            return None;
        }

        let dx = wx - camera_eye[0];
        let dy = wy - camera_eye[1];
        let dz = wz - camera_eye[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;

        let lod = if dist_sq < 80.0 * 80.0 {
            0
        } else if dist_sq < 260.0 * 260.0 {
            1
        } else {
            2
        };

        let health_ratio = if u.max_hp > 0 {
            (u.hp as f32 / u.max_hp as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let model_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [wx, wy, wz, 1.0],
        ];

        Some(RenderInstance {
            unit_id: u.id,
            model_matrix,
            world_pos: [wx, wy, wz],
            health_ratio,
            lod,
        })
    });

    let mut instances = Vec::with_capacity(units.len());
    let mut culled_count = 0usize;
    let mut lod_counts = [0usize; 3];

    for opt in &results {
        if let Some(inst) = opt {
            lod_counts[inst.lod as usize] += 1;
            instances.push(inst.clone());
        } else {
            culled_count += 1;
        }
    }

    let mut hash = FNV_OFFSET_BASIS;
    hash = fnv1a_extend(hash, &(culled_count as u64).to_le_bytes());

    for (u, opt) in units.iter().zip(results.iter()) {
        if let Some(inst) = opt {
            hash = fnv1a_extend(hash, &inst.unit_id.to_le_bytes());
            hash = fnv1a_extend(hash, &[inst.lod]);
            hash = fnv1a_extend(hash, &u.x.to_le_bytes());
            hash = fnv1a_extend(hash, &u.y.to_le_bytes());
            hash = fnv1a_extend(hash, &u.hp.to_le_bytes());
        }
    }

    PreparedFrame {
        instances,
        culled_count,
        lod_counts,
        digest_fnv1a: hash,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::SCALE;

    fn make_unit(id: u32, wx: f32, wz: f32, hp: i64, max_hp: i64) -> SimUnitSnapshot {
        SimUnitSnapshot {
            id,
            blueprint_id: "UAL0001".to_string(),
            x: (wx * SCALE as f32).round() as i64,
            y: (wz * SCALE as f32).round() as i64,
            hp,
            max_hp,
        }
    }

    #[test]
    fn test_deterministic_parallel_vs_single_thread_512_units() {
        let view_proj = [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.2, 1.0],
            [0.0, 0.0, 10.0, 50.0],
        ];
        let camera_eye = [0.0, 20.0, -50.0];

        let units: Vec<SimUnitSnapshot> = (0..512u32)
            .map(|i| {
                let wx = ((i as i32 % 32) - 16) as f32 * 8.0;
                let wz = ((i as i32 / 32) - 4) as f32 * 25.0;
                make_unit(i + 1, wx, wz, 50 + (i as i64 % 50), 100)
            })
            .collect();

        let frame_1 = prepare_render_frame(&units, None, view_proj, camera_eye, 1);
        let frame_4 = prepare_render_frame(&units, None, view_proj, camera_eye, 4);

        assert_eq!(frame_1, frame_4);
        assert_eq!(frame_1.digest_fnv1a, frame_4.digest_fnv1a);
        assert!(!frame_1.instances.is_empty());
        assert!(frame_1.culled_count > 0);
        assert_eq!(
            frame_1.instances.len() + frame_1.culled_count,
            units.len()
        );
    }

    #[test]
    fn test_frustum_culling_behind_camera_and_visible() {
        let view_proj = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let camera_eye = [0.0, 0.0, 0.0];

        let units = vec![
            make_unit(1, 0.0, 20.0, 100, 100),
            make_unit(2, 0.0, -10.0, 100, 100),
            make_unit(3, 0.0, 0.0, 100, 100),
            make_unit(4, 100.0, 10.0, 100, 100),
        ];

        let frame = prepare_render_frame(&units, None, view_proj, camera_eye, 2);
        assert_eq!(frame.instances.len(), 1);
        assert_eq!(frame.culled_count, 3);
        assert_eq!(frame.instances[0].unit_id, 1);
    }

    #[test]
    fn test_lod_assignment_by_distance() {
        let view_proj = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1000.0],
        ];
        let camera_eye = [0.0, 0.0, 0.0];

        let units = vec![
            make_unit(10, 50.0, 0.0, 80, 100),
            make_unit(20, 150.0, 0.0, 50, 100),
            make_unit(30, 300.0, 0.0, 25, 100),
        ];

        let frame = prepare_render_frame(&units, None, view_proj, camera_eye, 1);
        assert_eq!(frame.culled_count, 0);
        assert_eq!(frame.instances.len(), 3);
        assert_eq!(frame.instances[0].lod, 0);
        assert_eq!(frame.instances[1].lod, 1);
        assert_eq!(frame.instances[2].lod, 2);
        assert_eq!(frame.lod_counts, [1, 1, 1]);
        assert!((frame.instances[0].health_ratio - 0.8).abs() < 1e-6);
    }
}
