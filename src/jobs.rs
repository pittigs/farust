//! Phase 5 scoped worker pool (`std::thread::scope`) and parallel asset batch decoding.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use crate::blueprint;
use crate::dds_decode::decode_dds_rgba8;
use crate::scm::ScmMesh;

const FNV1A_OFFSET_BASIS: u64 = 0xCBF2_9CE4_8422_2325;
const FNV1A_PRIME: u64 = 0x0000_0100_0000_01B3;

#[inline]
fn fnv1a_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV1A_PRIME);
    }
    hash
}

/// Returns available hardware parallelism (minimum 1).
pub fn default_thread_count() -> usize {
    thread::available_parallelism()
        .map(|n| n.get().max(1))
        .unwrap_or(1)
}

/// Executes `f(index, &item)` across `threads.max(1)` scoped worker threads
/// and returns results in exact input order `0..items.len()`.
/// When `threads <= 1` or `items.len() <= 1`, runs inline on the current thread.
pub fn par_map_ordered<T, R, F>(items: &[T], threads: usize, f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(usize, &T) -> R + Sync,
{
    if items.is_empty() {
        return Vec::new();
    }

    let worker_count = threads.max(1).min(items.len());
    if worker_count == 1 {
        return items
            .iter()
            .enumerate()
            .map(|(i, item)| f(i, item))
            .collect();
    }

    let next = AtomicUsize::new(0);
    let len = items.len();
    let batch_size = (len / (worker_count * 4)).clamp(1, 64);
    let f_ref = &f;
    let next_ref = &next;

    let worker_batches: Vec<Vec<(usize, Vec<R>)>> = thread::scope(|s| {
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            handles.push(s.spawn(move || {
                let mut local = Vec::new();
                loop {
                    let start = next_ref.fetch_add(batch_size, Ordering::Relaxed);
                    if start >= len {
                        break;
                    }
                    let end = (start + batch_size).min(len);
                    let mut chunk_out = Vec::with_capacity(end - start);
                    for idx in start..end {
                        chunk_out.push(f_ref(idx, &items[idx]));
                    }
                    local.push((start, chunk_out));
                }
                local
            }));
        }

        handles
            .into_iter()
            .map(|h| h.join().expect("worker thread panicked"))
            .collect()
    });

    let mut slots: Vec<Option<R>> = (0..len).map(|_| None).collect();
    for batch in worker_batches {
        for (start, chunk_out) in batch {
            for (offset, val) in chunk_out.into_iter().enumerate() {
                slots[start + offset] = Some(val);
            }
        }
    }

    slots
        .into_iter()
        .map(|slot| slot.expect("all slots populated"))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct DecodedAssetInfo {
    pub path: String,
    pub kind: &'static str, // "blueprint" | "scm" | "dds"
    pub primary_count: usize,
    pub secondary_count: usize,
    pub content_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct AssetBatchReport {
    pub threads: usize,
    pub total_items: usize,
    pub blueprints_parsed: usize,
    pub meshes_parsed: usize,
    pub textures_decoded: usize,
    pub total_vertices: usize,
    pub total_texels: usize,
    pub digest_fnv1a: u64,
}

pub fn decode_asset_entry(path: &str, bytes: &[u8]) -> Result<DecodedAssetInfo, String> {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with("_unit.bp") || lower.ends_with(".bp") {
        let text = String::from_utf8_lossy(bytes);
        let root = blueprint::parse_blueprint(&text)?;
        let unit_id = blueprint::unit_id_from_path(path);
        let summary = blueprint::summarize(&unit_id, path, &root);

        let primary_count = summary.categories.len() + summary.weapons.len();
        let health_val = summary.health.unwrap_or(0.0);
        let secondary_count = health_val as usize;

        let mut h = FNV1A_OFFSET_BASIS;
        h = fnv1a_update(h, summary.id.as_bytes());
        h = fnv1a_update(h, &(summary.categories.len() as u64).to_le_bytes());
        h = fnv1a_update(h, &(summary.weapons.len() as u64).to_le_bytes());
        h = fnv1a_update(h, &health_val.to_bits().to_le_bytes());

        Ok(DecodedAssetInfo {
            path: path.to_string(),
            kind: "blueprint",
            primary_count,
            secondary_count,
            content_hash: h,
        })
    } else if lower.ends_with(".scm") {
        let mesh = ScmMesh::parse(bytes)?;
        let primary_count = mesh.vertices.len();
        let secondary_count = mesh.triangle_count();

        let mut h = FNV1A_OFFSET_BASIS;
        h = fnv1a_update(h, &(mesh.version as u64).to_le_bytes());
        h = fnv1a_update(h, &(mesh.vertices.len() as u64).to_le_bytes());
        h = fnv1a_update(h, &(mesh.indices.len() as u64).to_le_bytes());
        h = fnv1a_update(h, &(mesh.bones.len() as u64).to_le_bytes());

        Ok(DecodedAssetInfo {
            path: path.to_string(),
            kind: "scm",
            primary_count,
            secondary_count,
            content_hash: h,
        })
    } else if lower.ends_with(".dds") {
        let img = decode_dds_rgba8(bytes)?;
        let primary_count = (img.width as usize) * (img.height as usize);
        let secondary_count = img.pixels.len();

        let mut h = FNV1A_OFFSET_BASIS;
        h = fnv1a_update(h, &img.width.to_le_bytes());
        h = fnv1a_update(h, &img.height.to_le_bytes());
        h = fnv1a_update(h, &(img.pixels.len() as u64).to_le_bytes());
        for &b in img.pixels.iter().step_by(64) {
            h ^= u64::from(b);
            h = h.wrapping_mul(FNV1A_PRIME);
        }

        Ok(DecodedAssetInfo {
            path: path.to_string(),
            kind: "dds",
            primary_count,
            secondary_count,
            content_hash: h,
        })
    } else {
        Err(format!("unsupported asset extension: {path}"))
    }
}

pub fn decode_assets_batch(
    entries: &[(String, Vec<u8>)],
    threads: usize,
) -> Result<AssetBatchReport, String> {
    let eff_threads = threads.max(1);
    let results = par_map_ordered(entries, eff_threads, |_, (path, bytes)| {
        decode_asset_entry(path, bytes)
    });

    let mut blueprints_parsed = 0usize;
    let mut meshes_parsed = 0usize;
    let mut textures_decoded = 0usize;
    let mut total_vertices = 0usize;
    let mut total_texels = 0usize;
    let mut digest = FNV1A_OFFSET_BASIS;

    for res in results {
        let info = res?;
        match info.kind {
            "blueprint" => {
                blueprints_parsed += 1;
            }
            "scm" => {
                meshes_parsed += 1;
                total_vertices += info.primary_count;
            }
            "dds" => {
                textures_decoded += 1;
                total_texels += info.primary_count;
            }
            _ => {}
        }

        digest = fnv1a_update(digest, info.path.as_bytes());
        digest = fnv1a_update(digest, info.kind.as_bytes());
        digest = fnv1a_update(digest, &(info.primary_count as u64).to_le_bytes());
        digest = fnv1a_update(digest, &(info.secondary_count as u64).to_le_bytes());
        digest = fnv1a_update(digest, &info.content_hash.to_le_bytes());
    }

    Ok(AssetBatchReport {
        threads: eff_threads,
        total_items: entries.len(),
        blueprints_parsed,
        meshes_parsed,
        textures_decoded,
        total_vertices,
        total_texels,
        digest_fnv1a: digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_synthetic_scm() -> Vec<u8> {
        let mut buf = Vec::with_capacity(512);
        let push_u32 = |b: &mut Vec<u8>, v: u32| b.extend_from_slice(&v.to_le_bytes());
        let push_f32 = |b: &mut Vec<u8>, v: f32| b.extend_from_slice(&v.to_le_bytes());

        // Header: 4B magic + 11 u32s = 48 bytes
        // 1 bone at offset 48 (108 bytes -> ends at 156)
        // 3 vertices at offset 156 (3 * 68 = 204 bytes -> ends at 360)
        // 3 indices at offset 360 (3 * 2 = 6 bytes -> ends at 366)
        // Bone name at offset 366 ("root\0")
        buf.extend_from_slice(b"MODL");
        push_u32(&mut buf, 5); // version
        push_u32(&mut buf, 48); // bone_offset
        push_u32(&mut buf, 1); // weighted_bone_count
        push_u32(&mut buf, 156); // vertex_offset
        push_u32(&mut buf, 0); // extra_vertex_offset
        push_u32(&mut buf, 3); // vertex_count
        push_u32(&mut buf, 360); // index_offset
        push_u32(&mut buf, 3); // index_count
        push_u32(&mut buf, 0); // info_offset
        push_u32(&mut buf, 0); // info_count
        push_u32(&mut buf, 1); // total_bone_count

        // 1 bone (108 bytes): 16f inv_bind (64B) + 3f pos (12B) + 4f rot (16B) + name_off (4B) + parent_idx (4B) + 8B reserved
        for i in 0..16 {
            push_f32(&mut buf, if i % 5 == 0 { 1.0 } else { 0.0 });
        }
        for _ in 0..3 {
            push_f32(&mut buf, 0.0);
        }
        for i in 0..4 {
            push_f32(&mut buf, if i == 3 { 1.0 } else { 0.0 });
        }
        push_u32(&mut buf, 366); // name_offset
        buf.extend_from_slice(&(-1i32).to_le_bytes()); // parent_index
        buf.extend_from_slice(&[0u8; 8]); // reserved

        // 3 vertices (68 bytes each)
        for v in 0..3 {
            push_f32(&mut buf, v as f32);
            push_f32(&mut buf, 1.0);
            push_f32(&mut buf, 0.0);
            for _ in 0..13 {
                push_f32(&mut buf, 0.0);
            }
            buf.extend_from_slice(&[0u8, 0, 0, 0]);
        }

        // 3 indices (6 bytes)
        for idx in [0u16, 1, 2] {
            buf.extend_from_slice(&idx.to_le_bytes());
        }

        // Bone name at offset 366
        buf.extend_from_slice(b"root\0");
        buf
    }

    fn make_synthetic_dds_dxt1_4x4() -> Vec<u8> {
        let mut buf = vec![0u8; 128];
        buf[0..4].copy_from_slice(b"DDS ");
        buf[4..8].copy_from_slice(&124u32.to_le_bytes());
        buf[8..12].copy_from_slice(&0x1007u32.to_le_bytes());
        buf[12..16].copy_from_slice(&4u32.to_le_bytes());
        buf[16..20].copy_from_slice(&4u32.to_le_bytes());
        buf[76..80].copy_from_slice(&32u32.to_le_bytes());
        buf[80..84].copy_from_slice(&0x4u32.to_le_bytes());
        buf[84..88].copy_from_slice(b"DXT1");
        buf.extend_from_slice(&0xF800u16.to_le_bytes());
        buf.extend_from_slice(&0x07E0u16.to_le_bytes());
        buf.extend_from_slice(&0x0000_0000u32.to_le_bytes());
        buf
    }

    #[test]
    fn test_par_map_ordered_preserves_order() {
        let empty: Vec<u32> = Vec::new();
        let res_empty = par_map_ordered(&empty, 4, |i, &x| (i, x));
        assert!(res_empty.is_empty());

        let items: Vec<usize> = (0..256).collect();
        for threads in [0, 1, 2, 4, 8] {
            let mapped = par_map_ordered(&items, threads, |idx, &val| {
                assert_eq!(idx, val);
                val * 3 + 7
            });
            assert_eq!(mapped.len(), items.len());
            for (i, &got) in mapped.iter().enumerate() {
                assert_eq!(got, i * 3 + 7, "mismatch at index {i} with threads={threads}");
            }
        }
    }

    #[test]
    fn test_decode_assets_batch_deterministic() {
        let bp_bytes = br#"UnitBlueprint {
            Description = " Armored Tank ",
            Defense = { MaxHealth = 450, },
            Categories = { "MOBILE", "LAND", "DIRECTFIRE" },
            Weapon = {
                { Label = "MainGun", MaxRadius = 30 },
            },
        }"#
        .to_vec();
        let scm_bytes = make_synthetic_scm();
        let dds_bytes = make_synthetic_dds_dxt1_4x4();

        let mut entries = Vec::new();
        for i in 0..8 {
            entries.push((format!("/units/uel020{i}/uel020{i}_unit.bp"), bp_bytes.clone()));
            entries.push((format!("/units/uel020{i}/uel020{i}_lod0.scm"), scm_bytes.clone()));
            entries.push((format!("/units/uel020{i}/uel020{i}_albedo.dds"), dds_bytes.clone()));
        }

        let report_1t = decode_assets_batch(&entries, 1).expect("1-thread batch should succeed");
        let report_4t = decode_assets_batch(&entries, 4).expect("4-thread batch should succeed");

        assert_eq!(report_1t.total_items, 24);
        assert_eq!(report_1t.blueprints_parsed, 8);
        assert_eq!(report_1t.meshes_parsed, 8);
        assert_eq!(report_1t.textures_decoded, 8);
        assert_eq!(report_1t.total_vertices, 8 * 3);
        assert_eq!(report_1t.total_texels, 8 * 16);

        assert_eq!(report_1t.total_items, report_4t.total_items);
        assert_eq!(report_1t.blueprints_parsed, report_4t.blueprints_parsed);
        assert_eq!(report_1t.meshes_parsed, report_4t.meshes_parsed);
        assert_eq!(report_1t.textures_decoded, report_4t.textures_decoded);
        assert_eq!(report_1t.total_vertices, report_4t.total_vertices);
        assert_eq!(report_1t.total_texels, report_4t.total_texels);
        assert_eq!(report_1t.digest_fnv1a, report_4t.digest_fnv1a);
    }
}
