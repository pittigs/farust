//! Parser for Supreme Commander `.scm` (MODL) mesh files (versions 3..=5).

const MAGIC: &[u8; 4] = b"MODL";
const HEADER_SIZE: usize = 48;
const BONE_SIZE: usize = 108;
const VERTEX_SIZE: usize = 68;
const INDEX_SIZE: usize = 2;

/// A skeleton bone decoded from an `.scm` file.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ScmBone {
    /// Inverse bind pose 4x4 matrix (row-major DX layout `[r0, r1, r2, r3]`).
    pub inv_bind_pose: [f32; 16],
    /// Bone name decoded from the file's NUL-terminated ASCII/Latin-1 string pool.
    pub name: String,
    /// Index of the parent bone, or `-1` for a root bone.
    pub parent_index: i32,
    /// Local bone translation `[x, y, z]`.
    pub position: [f32; 3],
    /// Local bone rotation quaternion `[w, x, y, z]`.
    pub rotation: [f32; 4],
}

/// A mesh vertex decoded from an `.scm` file.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ScmVertex {
    /// Vertex position `[x, y, z]`.
    pub position: [f32; 3],
    /// Vertex normal `[x, y, z]`.
    pub normal: [f32; 3],
    /// Vertex tangent `[x, y, z]`.
    pub tangent: [f32; 3],
    /// Vertex binormal `[x, y, z]`.
    pub binormal: [f32; 3],
    /// Primary texture coordinates `[u, v]`.
    pub uv0: [f32; 2],
    /// Secondary texture coordinates `[u, v]`.
    pub uv1: [f32; 2],
    /// Up to four bone indices influencing this vertex.
    pub bone_indices: [u8; 4],
}

/// Axis-aligned bounding box across mesh vertex positions.
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    /// Minimum `[x, y, z]` coordinates.
    pub min: [f32; 3],
    /// Maximum `[x, y, z]` coordinates.
    pub max: [f32; 3],
}

/// Parsed `.scm` mesh data.
#[derive(Debug, Clone)]
pub struct ScmMesh {
    /// File format version (`3..=5`).
    pub version: u32,
    /// Number of weighted bones reported in the header.
    pub weighted_bone_count: u32,
    /// All skeleton bones (`total_bone_count`).
    pub bones: Vec<ScmBone>,
    /// Mesh vertices.
    pub vertices: Vec<ScmVertex>,
    /// Triangle index buffer.
    pub indices: Vec<u16>,
    /// Axis-aligned bounds over `vertices`, or `None` when `vertices` is empty.
    pub bounds: Option<Bounds>,
    /// Non-empty metadata strings decoded from the info section.
    pub info: Vec<String>,
}

impl ScmMesh {
    /// Parses a Supreme Commander `.scm` mesh from raw little-endian bytes.
    pub fn parse(data: &[u8]) -> Result<ScmMesh, String> {
        let header = parse_header(data)?;
        let bones = parse_bones(data, header.bone_offset, header.total_bone_count)?;
        let vertices = parse_vertices(data, header.vertex_offset, header.vertex_count)?;
        let indices = parse_indices(data, header.index_offset, header.index_count)?;
        let bounds = compute_bounds(&vertices);
        let info = parse_info(data, header.info_offset, header.info_count)?;

        Ok(ScmMesh {
            version: header.version,
            weighted_bone_count: header.weighted_bone_count,
            bones,
            vertices,
            indices,
            bounds,
            info,
        })
    }

    /// Returns the number of triangles represented by `indices`.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Case-insensitive skeleton bone lookup by name.
    #[allow(dead_code)]
    pub fn bone_index_ci(&self, name: &str) -> Option<usize> {
        self.bones
            .iter()
            .position(|b| b.name.eq_ignore_ascii_case(name))
    }

    /// Identifies primary yaw and pitch turret bone indices if present.
    #[allow(dead_code)]
    pub fn find_turret_bones(&self) -> (Option<usize>, Option<usize>) {
        let yaw_names = [
            "Turret",
            "Turret_Yaw",
            "Main_Turret",
            "Center_Turret",
            "Torso",
        ];
        let pitch_names = [
            "Turret_Barrel",
            "Turret_Pitch",
            "Main_Turret_Barrel",
            "Center_Turret_Barrel",
            "Barrel",
        ];
        let yaw = yaw_names
            .iter()
            .find_map(|n| self.bone_index_ci(n))
            .or_else(|| {
                self.bones
                    .iter()
                    .position(|b| b.name.to_ascii_lowercase().contains("turret"))
            });
        let pitch = pitch_names
            .iter()
            .find_map(|n| self.bone_index_ci(n))
            .or_else(|| {
                self.bones
                    .iter()
                    .position(|b| b.name.to_ascii_lowercase().contains("barrel"))
            });
        (yaw, pitch)
    }
}

struct ScmHeader {
    version: u32,
    bone_offset: u32,
    weighted_bone_count: u32,
    vertex_offset: u32,
    // Field at 20..24 is `extra_vertex_offset` (not needed for base mesh parsing).
    vertex_count: u32,
    index_offset: u32,
    index_count: u32,
    info_offset: u32,
    info_count: u32,
    total_bone_count: u32,
}

fn parse_header(data: &[u8]) -> Result<ScmHeader, String> {
    if data.len() < HEADER_SIZE {
        return Err(format!(
            "SCM data too short: expected at least {HEADER_SIZE} bytes, got {}",
            data.len()
        ));
    }
    if data.get(0..4) != Some(MAGIC.as_slice()) {
        return Err(format!(
            "invalid SCM magic: expected b\"MODL\", got {:?}",
            &data[0..4]
        ));
    }

    let version = read_u32(data, 4)?;
    if !(3..=5).contains(&version) {
        return Err(format!(
            "unsupported SCM version {version}: expected 3..=5"
        ));
    }

    Ok(ScmHeader {
        version,
        bone_offset: read_u32(data, 8)?,
        weighted_bone_count: read_u32(data, 12)?,
        vertex_offset: read_u32(data, 16)?,
        vertex_count: read_u32(data, 24)?,
        index_offset: read_u32(data, 28)?,
        index_count: read_u32(data, 32)?,
        info_offset: read_u32(data, 36)?,
        info_count: read_u32(data, 40)?,
        total_bone_count: read_u32(data, 44)?,
    })
}

fn parse_bones(data: &[u8], offset: u32, count: u32) -> Result<Vec<ScmBone>, String> {
    let bytes = checked_slice(data, offset, count, BONE_SIZE, "bones")?;
    let mut bones = Vec::with_capacity(bytes.len() / BONE_SIZE);
    for chunk in bytes.chunks_exact(BONE_SIZE) {
        // 0..64 holds the 16-f32 inverse bind pose matrix; 100..108 is reserved.
        let inv_bind_pose = read_f32_array::<16>(chunk, 0)?;
        let position = read_f32_array::<3>(chunk, 64)?;
        let rotation = read_f32_array::<4>(chunk, 76)?;
        let name_offset = read_f32_name_off_or_u32(chunk)?;
        let parent_index = read_i32(chunk, 96)?;
        let name = read_latin1_cstring(data, name_offset)?;

        bones.push(ScmBone {
            inv_bind_pose,
            name,
            parent_index,
            position,
            rotation,
        });
    }
    Ok(bones)
}

#[inline]
fn read_f32_name_off_or_u32(chunk: &[u8]) -> Result<u32, String> {
    read_u32(chunk, 92)
}

fn parse_vertices(data: &[u8], offset: u32, count: u32) -> Result<Vec<ScmVertex>, String> {
    let bytes = checked_slice(data, offset, count, VERTEX_SIZE, "vertices")?;
    let mut vertices = Vec::with_capacity(bytes.len() / VERTEX_SIZE);
    for chunk in bytes.chunks_exact(VERTEX_SIZE) {
        vertices.push(ScmVertex {
            position: read_f32_array::<3>(chunk, 0)?,
            normal: read_f32_array::<3>(chunk, 12)?,
            tangent: read_f32_array::<3>(chunk, 24)?,
            binormal: read_f32_array::<3>(chunk, 36)?,
            uv0: read_f32_array::<2>(chunk, 48)?,
            uv1: read_f32_array::<2>(chunk, 56)?,
            bone_indices: read_array::<4>(chunk, 64)?,
        });
    }
    Ok(vertices)
}

fn parse_indices(data: &[u8], offset: u32, count: u32) -> Result<Vec<u16>, String> {
    let bytes = checked_slice(data, offset, count, INDEX_SIZE, "indices")?;
    let mut indices = Vec::with_capacity(bytes.len() / INDEX_SIZE);
    for chunk in bytes.chunks_exact(INDEX_SIZE) {
        indices.push(read_u16(chunk, 0)?);
    }
    Ok(indices)
}

fn parse_info(data: &[u8], offset: u32, count: u32) -> Result<Vec<String>, String> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let bytes = checked_slice(data, offset, count, 1, "info")?;
    let info = bytes
        .split(|&b| b == 0)
        .filter(|slice| !slice.is_empty())
        .map(|slice| String::from_utf8_lossy(slice).into_owned())
        .collect();
    Ok(info)
}

fn compute_bounds(vertices: &[ScmVertex]) -> Option<Bounds> {
    let [first, rest @ ..] = vertices else {
        return None;
    };
    let mut min = first.position;
    let mut max = first.position;
    for vertex in rest {
        min = [
            min[0].min(vertex.position[0]),
            min[1].min(vertex.position[1]),
            min[2].min(vertex.position[2]),
        ];
        max = [
            max[0].max(vertex.position[0]),
            max[1].max(vertex.position[1]),
            max[2].max(vertex.position[2]),
        ];
    }
    Some(Bounds { min, max })
}

fn read_latin1_cstring(data: &[u8], offset: u32) -> Result<String, String> {
    let start = usize::try_from(offset)
        .map_err(|_| format!("bone name offset {offset} exceeds usize"))?;
    let tail = data
        .get(start..)
        .ok_or_else(|| format!("bone name offset {start} out of bounds (len {})", data.len()))?;
    let nul_pos = tail
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| format!("unterminated bone name string at offset {start}"))?;
    Ok(tail[..nul_pos].iter().copied().map(char::from).collect())
}

fn checked_slice<'a>(
    data: &'a [u8],
    offset: u32,
    count: u32,
    elem_size: usize,
    section: &str,
) -> Result<&'a [u8], String> {
    let start = usize::try_from(offset)
        .map_err(|_| format!("{section} offset {offset} exceeds usize"))?;
    let count_usize = usize::try_from(count)
        .map_err(|_| format!("{section} count {count} exceeds usize"))?;
    let byte_len = count_usize
        .checked_mul(elem_size)
        .ok_or_else(|| format!("{section} byte size overflow ({count} * {elem_size})"))?;
    let end = start
        .checked_add(byte_len)
        .ok_or_else(|| format!("{section} range overflow (offset {offset}, len {byte_len})"))?;
    data.get(start..end).ok_or_else(|| {
        format!(
            "{section} slice {start}..{end} out of bounds for input length {}",
            data.len()
        )
    })
}

fn read_array<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N], String> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| format!("offset overflow reading {N} bytes at {offset}"))?;
    bytes
        .get(offset..end)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| format!("unexpected EOF reading {N} bytes at {offset}"))
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, String> {
    Ok(u16::from_le_bytes(read_array::<2>(bytes, offset)?))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    Ok(u32::from_le_bytes(read_array::<4>(bytes, offset)?))
}

fn read_i32(bytes: &[u8], offset: usize) -> Result<i32, String> {
    Ok(i32::from_le_bytes(read_array::<4>(bytes, offset)?))
}

fn read_f32_array<const N: usize>(bytes: &[u8], offset: usize) -> Result<[f32; N], String> {
    let mut out = [0.0f32; N];
    for (i, slot) in out.iter_mut().enumerate() {
        let elem_offset = i
            .checked_mul(4)
            .and_then(|rel| offset.checked_add(rel))
            .ok_or_else(|| format!("offset overflow reading f32 array at {offset}"))?;
        *slot = f32::from_le_bytes(read_array::<4>(bytes, elem_offset)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestBone<'a> {
        name: &'a [u8],
        parent_index: i32,
        position: [f32; 3],
        rotation: [f32; 4],
    }

    struct TestScm<'a> {
        magic: [u8; 4],
        version: u32,
        weighted_bone_count: u32,
        bones: Vec<TestBone<'a>>,
        vertices: Vec<ScmVertex>,
        indices: Vec<u16>,
        info: &'a [u8],
    }

    fn push_f32_slice(buf: &mut Vec<u8>, values: &[f32]) {
        for &v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn write_u32_at(buf: &mut [u8], offset: usize, value: u32) {
        buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn build_scm(spec: &TestScm<'_>) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&spec.magic);
        buf.resize(HEADER_SIZE, 0);

        let mut name_offsets = Vec::with_capacity(spec.bones.len());
        for bone in &spec.bones {
            name_offsets.push(u32::try_from(buf.len()).unwrap());
            buf.extend_from_slice(bone.name);
            buf.push(0);
        }

        let bone_offset = u32::try_from(buf.len()).unwrap();
        for (bone, &name_offset) in spec.bones.iter().zip(name_offsets.iter()) {
            let inv_bind = [
                1.0f32, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ];
            push_f32_slice(&mut buf, &inv_bind);
            push_f32_slice(&mut buf, &bone.position);
            push_f32_slice(&mut buf, &bone.rotation);
            buf.extend_from_slice(&name_offset.to_le_bytes());
            buf.extend_from_slice(&bone.parent_index.to_le_bytes());
            buf.extend_from_slice(&[0u8; 8]);
        }

        let vertex_offset = u32::try_from(buf.len()).unwrap();
        for v in &spec.vertices {
            push_f32_slice(&mut buf, &v.position);
            push_f32_slice(&mut buf, &v.normal);
            push_f32_slice(&mut buf, &v.tangent);
            push_f32_slice(&mut buf, &v.binormal);
            push_f32_slice(&mut buf, &v.uv0);
            push_f32_slice(&mut buf, &v.uv1);
            buf.extend_from_slice(&v.bone_indices);
        }

        let index_offset = u32::try_from(buf.len()).unwrap();
        for &idx in &spec.indices {
            buf.extend_from_slice(&idx.to_le_bytes());
        }

        let info_offset = u32::try_from(buf.len()).unwrap();
        buf.extend_from_slice(spec.info);

        write_u32_at(&mut buf, 4, spec.version);
        write_u32_at(&mut buf, 8, bone_offset);
        write_u32_at(&mut buf, 12, spec.weighted_bone_count);
        write_u32_at(&mut buf, 16, vertex_offset);
        write_u32_at(&mut buf, 20, 0);
        write_u32_at(&mut buf, 24, u32::try_from(spec.vertices.len()).unwrap());
        write_u32_at(&mut buf, 28, index_offset);
        write_u32_at(&mut buf, 32, u32::try_from(spec.indices.len()).unwrap());
        write_u32_at(&mut buf, 36, info_offset);
        write_u32_at(&mut buf, 40, u32::try_from(spec.info.len()).unwrap());
        write_u32_at(&mut buf, 44, u32::try_from(spec.bones.len()).unwrap());

        buf
    }

    fn assert_f32_slice_eq(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_eq!(a.to_bits(), e.to_bits(), "float mismatch: {a} != {e}");
        }
    }

    fn sample_vertices() -> Vec<ScmVertex> {
        vec![
            ScmVertex {
                position: [-1.0, 0.0, 2.5],
                normal: [0.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0],
                binormal: [0.0, 0.0, 1.0],
                uv0: [0.0, 0.0],
                uv1: [0.1, 0.2],
                bone_indices: [0, 1, 2, 3],
            },
            ScmVertex {
                position: [3.0, -2.0, 0.5],
                normal: [0.0, 0.0, 1.0],
                tangent: [0.0, 1.0, 0.0],
                binormal: [1.0, 0.0, 0.0],
                uv0: [1.0, 0.0],
                uv1: [0.3, 0.4],
                bone_indices: [0, 0, 0, 0],
            },
            ScmVertex {
                position: [1.0, 4.5, -1.5],
                normal: [1.0, 0.0, 0.0],
                tangent: [0.0, 0.0, 1.0],
                binormal: [0.0, 1.0, 0.0],
                uv0: [0.5, 1.0],
                uv1: [0.5, 0.6],
                bone_indices: [2, 0, 0, 0],
            },
        ]
    }

    #[test]
    fn parses_valid_v5_mesh() {
        let vertices = sample_vertices();
        let data = build_scm(&TestScm {
            magic: *MAGIC,
            version: 5,
            weighted_bone_count: 1,
            bones: vec![TestBone {
                name: b"pelvis",
                parent_index: -1,
                position: [0.25, 1.5, -0.75],
                rotation: [0.0, 0.0, 0.70710677, 0.70710677],
            }],
            vertices: vertices.clone(),
            indices: vec![0, 1, 2],
            info: b"exporter=test\0\0lod=0\0",
        });

        let mesh = ScmMesh::parse(&data).expect("valid v5 SCM should parse");
        assert_eq!(mesh.version, 5);
        assert_eq!(mesh.weighted_bone_count, 1);
        assert_eq!(mesh.bones.len(), 1);
        assert_eq!(mesh.bones[0].name, "pelvis");
        assert_eq!(mesh.bones[0].parent_index, -1);
        assert_f32_slice_eq(
            &mesh.bones[0].inv_bind_pose,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        );
        assert_f32_slice_eq(&mesh.bones[0].position, &[0.25, 1.5, -0.75]);
        assert_f32_slice_eq(&mesh.bones[0].rotation, &[0.0, 0.0, 0.70710677, 0.70710677]);

        assert_eq!(mesh.vertices.len(), 3);
        for (actual, expected) in mesh.vertices.iter().zip(vertices.iter()) {
            assert_f32_slice_eq(&actual.position, &expected.position);
            assert_f32_slice_eq(&actual.normal, &expected.normal);
            assert_f32_slice_eq(&actual.tangent, &expected.tangent);
            assert_f32_slice_eq(&actual.binormal, &expected.binormal);
            assert_f32_slice_eq(&actual.uv0, &expected.uv0);
            assert_f32_slice_eq(&actual.uv1, &expected.uv1);
            assert_eq!(actual.bone_indices, expected.bone_indices);
        }

        assert_eq!(mesh.indices, vec![0, 1, 2]);
        assert_eq!(mesh.triangle_count(), 1);

        let bounds = mesh.bounds.expect("non-empty vertices should produce bounds");
        assert_f32_slice_eq(&bounds.min, &[-1.0, -2.0, -1.5]);
        assert_f32_slice_eq(&bounds.max, &[3.0, 4.5, 2.5]);

        assert_eq!(
            mesh.info,
            vec!["exporter=test".to_string(), "lod=0".to_string()]
        );
    }

    #[test]
    fn accepts_version_3_and_4() {
        for version in [3, 4] {
            let data = build_scm(&TestScm {
                magic: *MAGIC,
                version,
                weighted_bone_count: 1,
                bones: vec![TestBone {
                    name: b"root",
                    parent_index: -1,
                    position: [0.0, 0.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                }],
                vertices: sample_vertices(),
                indices: vec![0, 1, 2],
                info: b"",
            });

            let mesh = ScmMesh::parse(&data).expect("versions 3 and 4 should be accepted");
            assert_eq!(mesh.version, version);
            assert_eq!(mesh.bones.len(), 1);
            assert_eq!(mesh.bones[0].name, "root");
            assert_eq!(mesh.triangle_count(), 1);
            assert!(mesh.info.is_empty());
        }
    }

    #[test]
    fn rejects_unsupported_versions() {
        for version in [0, 1, 2, 6, 99, u32::MAX] {
            let data = build_scm(&TestScm {
                magic: *MAGIC,
                version,
                weighted_bone_count: 0,
                bones: vec![],
                vertices: vec![],
                indices: vec![],
                info: b"",
            });
            assert!(
                ScmMesh::parse(&data).is_err(),
                "version {version} should be rejected"
            );
        }
    }

    #[test]
    fn rejects_bad_magic_and_truncated_header() {
        assert!(ScmMesh::parse(&[]).is_err());
        assert!(ScmMesh::parse(&[0u8; HEADER_SIZE - 1]).is_err());

        let bad_magic = build_scm(&TestScm {
            magic: *b"BAD!",
            version: 5,
            weighted_bone_count: 0,
            bones: vec![],
            vertices: vec![],
            indices: vec![],
            info: b"",
        });
        assert!(ScmMesh::parse(&bad_magic).is_err());
    }

    #[test]
    fn rejects_out_of_bounds_offsets_and_overflows() {
        let base = build_scm(&TestScm {
            magic: *MAGIC,
            version: 5,
            weighted_bone_count: 0,
            bones: vec![],
            vertices: vec![],
            indices: vec![],
            info: b"tag\0",
        });

        let cases: &[(usize, u32, usize, u32)] = &[
            (8, u32::MAX, 44, 1),          // bone offset overflow
            (8, 48, 44, u32::MAX),         // bone count overflow
            (8, 4096, 44, 0),              // bone offset past EOF even with 0 count
            (16, u32::MAX, 24, 1),         // vertex offset overflow
            (16, 48, 24, u32::MAX),        // vertex count overflow
            (28, u32::MAX, 32, 1),         // index offset overflow
            (28, 48, 32, u32::MAX),        // index count overflow
            (36, u32::MAX, 40, 1),         // info offset overflow
            (36, 48, 40, u32::MAX),        // info count overflow
            (36, base.len() as u32, 40, 1), // info slice past EOF
        ];

        for &(off_pos, off_val, cnt_pos, cnt_val) in cases {
            let mut corrupted = base.clone();
            write_u32_at(&mut corrupted, off_pos, off_val);
            write_u32_at(&mut corrupted, cnt_pos, cnt_val);
            assert!(
                ScmMesh::parse(&corrupted).is_err(),
                "expected error for ({off_pos}={off_val}, {cnt_pos}={cnt_val})"
            );
        }
    }

    #[test]
    fn rejects_invalid_bone_name_offset_and_missing_nul() {
        let base = build_scm(&TestScm {
            magic: *MAGIC,
            version: 5,
            weighted_bone_count: 1,
            bones: vec![TestBone {
                name: b"ok",
                parent_index: -1,
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            }],
            vertices: vec![],
            indices: vec![],
            info: b"",
        });

        let bone_offset = read_u32(&base, 8).unwrap() as usize;

        // Name offset exactly at EOF.
        let mut at_eof = base.clone();
        let eof_offset = at_eof.len() as u32;
        write_u32_at(&mut at_eof, bone_offset + 92, eof_offset);
        assert!(ScmMesh::parse(&at_eof).is_err());

        // Name offset past EOF.
        let mut past_eof = base.clone();
        write_u32_at(&mut past_eof, bone_offset + 92, u32::MAX);
        assert!(ScmMesh::parse(&past_eof).is_err());

        // Name offset pointing to trailing bytes without a NUL terminator.
        let mut missing_nul = base;
        let tail_offset = missing_nul.len() as u32;
        missing_nul.extend_from_slice(b"unterminated");
        write_u32_at(&mut missing_nul, bone_offset + 92, tail_offset);
        assert!(ScmMesh::parse(&missing_nul).is_err());
    }

    #[test]
    fn handles_empty_mesh_and_ignores_info_offset_when_count_is_zero() {
        let mut data = build_scm(&TestScm {
            magic: *MAGIC,
            version: 5,
            weighted_bone_count: 0,
            bones: vec![],
            vertices: vec![],
            indices: vec![],
            info: b"",
        });
        // Info section is only read when info_count > 0.
        write_u32_at(&mut data, 36, u32::MAX);

        let mesh = ScmMesh::parse(&data).expect("empty mesh with info_count == 0 should parse");
        assert!(mesh.bones.is_empty());
        assert!(mesh.vertices.is_empty());
        assert!(mesh.indices.is_empty());
        assert!(mesh.bounds.is_none());
        assert_eq!(mesh.triangle_count(), 0);
        assert!(mesh.info.is_empty());
    }

    #[test]
    fn decodes_latin1_bone_names_and_lossy_utf8_info() {
        let data = build_scm(&TestScm {
            magic: *MAGIC,
            version: 5,
            weighted_bone_count: 1,
            bones: vec![TestBone {
                name: b"b\xf6ne_\xe9",
                parent_index: 0,
                position: [1.0, 2.0, 3.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            }],
            vertices: vec![],
            indices: vec![],
            info: b"\0first\0bad\xffutf8\0",
        });

        let mesh = ScmMesh::parse(&data).expect("latin-1 bone name and lossy info should parse");
        assert_eq!(mesh.bones[0].name, "böne_é");
        assert_eq!(
            mesh.info,
            vec!["first".to_string(), "bad\u{FFFD}utf8".to_string()]
        );
    }
}
