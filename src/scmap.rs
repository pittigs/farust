use std::fs;
use std::path::Path;

/// Water elevation configuration parsed from a Supreme Commander `.scmap` file.
#[derive(Debug, Clone)]
pub struct WaterInfo {
    pub has_water: bool,
    pub elevation: f32,
    pub elevation_deep: f32,
    pub elevation_abyss: f32,
}

/// One terrain texture stratum (`path` + tile `scale`) from a `.scmap` file.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TerrainStratum {
    pub path: String,
    pub scale: f32,
}

/// Terrain strata descriptors and embedded blend-map DDS textures parsed from a `.scmap` file.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TerrainTextures {
    pub strata_albedo: Vec<TerrainStratum>,
    pub strata_normal: Vec<TerrainStratum>,
    pub texturemap0_dds: Vec<u8>,
    pub texturemap1_dds: Vec<u8>,
}

/// Parsed Supreme Commander `.scmap` map data (header, heightmap, environment paths, and water).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ScMap {
    pub version_major: i32,
    pub version_minor: i32,
    pub map_width: f32,
    pub map_height: f32,
    pub preview_dds: Vec<u8>,
    pub heightmap_width: u32,
    pub heightmap_height: u32,
    pub height_scale: f32,
    pub heightmap: Vec<u16>,
    pub terrain_shader: String,
    pub background_path: String,
    pub sky_cubemap_path: String,
    pub cubemaps: Vec<(String, String)>,
    pub water: WaterInfo,
    pub terrain_textures: Option<TerrainTextures>,
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(len).ok_or_else(|| {
            format!("offset overflow at byte {}: requested {} bytes", self.pos, len)
        })?;
        if end > self.data.len() {
            return Err(format!(
                "unexpected EOF at byte {}: needed {} bytes, {} remaining",
                self.pos,
                len,
                self.remaining()
            ));
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn skip(&mut self, len: usize) -> Result<(), String> {
        let _ = self.read_bytes(len)?;
        Ok(())
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let slice = self.read_bytes(N)?;
        let mut arr = [0u8; N];
        arr.copy_from_slice(slice);
        Ok(arr)
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.read_array()?))
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.read_array()?))
    }

    fn read_f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.read_array()?))
    }

    fn read_cstr(&mut self) -> Result<String, String> {
        let rest = &self.data[self.pos..];
        let rel_end = rest.iter().position(|&b| b == 0).ok_or_else(|| {
            format!("unterminated string starting at byte {}", self.pos)
        })?;
        let bytes = &rest[..rel_end];
        self.pos += rel_end + 1;
        Ok(decode_cstr_bytes(bytes))
    }
}

fn decode_cstr_bytes(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_owned(),
        Err(_) => bytes.iter().map(|&b| b as char).collect(),
    }
}

/// Parses the v2.60 post-water water/wave/contour/stratum/decal/blendmap sections.
/// Returns `None` on short synthetic test tails where full strata/DDS payloads are absent.
fn parse_terrain_textures(tail: &[u8]) -> Option<TerrainTextures> {
    if tail.len() < 128 {
        return None;
    }
    let mut r = Reader::new(tail);
    // 20 f32 water shading parameters
    r.skip(80).ok()?;
    let _water_cubemap = r.read_cstr().ok()?;
    let _water_ramp = r.read_cstr().ok()?;
    // 4 wave normal repeats
    r.skip(16).ok()?;
    // 4 wave layers (vec2 movement + texture path)
    for _ in 0..4 {
        r.skip(8).ok()?;
        let _ = r.read_cstr().ok()?;
    }
    let wave_gen_count = usize::try_from(r.read_u32().ok()?).ok()?;
    if wave_gen_count > r.remaining() / 70 {
        return None;
    }
    for _ in 0..wave_gen_count {
        let _ = r.read_cstr().ok()?;
        let _ = r.read_cstr().ok()?;
        r.skip(68).ok()?;
    }
    // Contour colors & padding
    r.skip(28).ok()?;

    let mut strata_albedo = Vec::with_capacity(10);
    for _ in 0..10 {
        let path = r.read_cstr().ok()?;
        let scale = r.read_f32().ok()?;
        strata_albedo.push(TerrainStratum { path, scale });
    }
    let mut strata_normal = Vec::with_capacity(9);
    for _ in 0..9 {
        let path = r.read_cstr().ok()?;
        let scale = r.read_f32().ok()?;
        strata_normal.push(TerrainStratum { path, scale });
    }

    r.skip(8).ok()?;
    let decal_count = usize::try_from(r.read_u32().ok()?).ok()?;
    if decal_count > r.remaining() / 60 {
        return None;
    }
    for _ in 0..decal_count {
        r.skip(8).ok()?;
        let ntex = usize::try_from(r.read_u32().ok()?).ok()?;
        if ntex > 16 {
            return None;
        }
        for _ in 0..ntex {
            let tlen = usize::try_from(r.read_u32().ok()?).ok()?;
            r.skip(tlen).ok()?;
        }
        r.skip(48).ok()?;
    }

    let group_count = usize::try_from(r.read_u32().ok()?).ok()?;
    if group_count > r.remaining() / 8 {
        return None;
    }
    for _ in 0..group_count {
        r.skip(4).ok()?;
        let _ = r.read_cstr().ok()?;
        let nmem = usize::try_from(r.read_u32().ok()?).ok()?;
        let byte_len = nmem.checked_mul(4)?;
        r.skip(byte_len).ok()?;
    }

    r.skip(8).ok()?;
    let norm_count = usize::try_from(r.read_u32().ok()?).ok()?;
    if norm_count > 16 {
        return None;
    }
    for _ in 0..norm_count {
        let len = usize::try_from(r.read_u32().ok()?).ok()?;
        r.skip(len).ok()?;
    }

    let tm0_len = usize::try_from(r.read_u32().ok()?).ok()?;
    let texturemap0_dds = r.read_bytes(tm0_len).ok()?.to_vec();
    if !texturemap0_dds.starts_with(b"DDS ") {
        return None;
    }

    let texturemap1_dds = if r.remaining() >= 4 {
        let tm1_len = usize::try_from(r.read_u32().ok()?).ok()?;
        let dds = r.read_bytes(tm1_len).ok()?.to_vec();
        if !dds.starts_with(b"DDS ") {
            return None;
        }
        dds
    } else {
        Vec::new()
    };

    Some(TerrainTextures {
        strata_albedo,
        strata_normal,
        texturemap0_dds,
        texturemap1_dds,
    })
}

impl ScMap {
    /// Parses a `.scmap` binary buffer in memory.
    pub fn parse(data: &[u8]) -> Result<ScMap, String> {
        let mut r = Reader::new(data);

        let magic = r.read_bytes(4)?;
        if magic != b"Map\x1a" {
            return Err(format!("invalid scmap magic: {:?}", magic));
        }

        let version_major = r.read_i32()?;
        if version_major != 2 {
            return Err(format!("unsupported scmap major version: {}", version_major));
        }

        let _unknown1 = r.read_u32()?;
        let _unknown2 = r.read_u32()?;
        let map_width = r.read_f32()?;
        let map_height = r.read_f32()?;
        let _unknown3 = r.read_u32()?;
        let _unknown4 = r.read_u16()?;

        let preview_dds_len = usize::try_from(r.read_u32()?)
            .map_err(|_| "preview DDS length exceeds addressable range".to_string())?;
        let preview_dds = r.read_bytes(preview_dds_len)?.to_vec();

        let version_minor = r.read_i32()?;
        let hm_w = r.read_i32()?;
        if hm_w <= 0 {
            return Err(format!("invalid heightmap width: {}", hm_w));
        }
        let hm_h = r.read_i32()?;
        if hm_h <= 0 {
            return Err(format!("invalid heightmap height: {}", hm_h));
        }
        let height_scale = r.read_f32()?;

        let sample_w = (hm_w as usize)
            .checked_add(1)
            .ok_or_else(|| format!("heightmap width overflow: {}", hm_w))?;
        let sample_h = (hm_h as usize)
            .checked_add(1)
            .ok_or_else(|| format!("heightmap height overflow: {}", hm_h))?;
        let sample_count = sample_w
            .checked_mul(sample_h)
            .ok_or_else(|| format!("heightmap dimensions overflow: {}x{}", hm_w, hm_h))?;
        let hm_byte_len = sample_count
            .checked_mul(2)
            .ok_or_else(|| format!("heightmap byte size overflow: {} samples", sample_count))?;

        let hm_bytes = r.read_bytes(hm_byte_len)?;
        let mut heightmap = Vec::with_capacity(sample_count);
        for chunk in hm_bytes.chunks_exact(2) {
            heightmap.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }

        if version_minor >= 56 {
            let _ = r.read_u8()?;
        }

        let terrain_shader = r.read_cstr()?;
        let background_path = r.read_cstr()?;
        let sky_cubemap_path = r.read_cstr()?;

        let cubemaps = if version_minor >= 60 {
            let count = usize::try_from(r.read_u32()?)
                .map_err(|_| "cubemap count exceeds addressable range".to_string())?;
            // Cap preallocation by remaining bytes (each pair needs at least two null bytes) to avoid OOM on corrupt headers.
            let mut pairs = Vec::with_capacity(count.min(r.remaining() / 2));
            for _ in 0..count {
                let name = r.read_cstr()?;
                let path = r.read_cstr()?;
                pairs.push((name, path));
            }
            pairs
        } else {
            Vec::new()
        };

        // 23 f32 lighting/fog parameters.
        r.skip(92)?;

        let has_water = r.read_u8()? != 0;
        let elevation = r.read_f32()?;
        let elevation_deep = r.read_f32()?;
        let elevation_abyss = r.read_f32()?;

        let terrain_textures = parse_terrain_textures(&data[r.pos..]);

        Ok(ScMap {
            version_major,
            version_minor,
            map_width,
            map_height,
            preview_dds,
            heightmap_width: hm_w as u32,
            heightmap_height: hm_h as u32,
            height_scale,
            heightmap,
            terrain_shader,
            background_path,
            sky_cubemap_path,
            cubemaps,
            water: WaterInfo {
                has_water,
                elevation,
                elevation_deep,
                elevation_abyss,
            },
            terrain_textures,
        })
    }

    /// Reads and parses a `.scmap` file from disk.
    pub fn parse_file(path: &Path) -> Result<ScMap, String> {
        let data = fs::read(path)
            .map_err(|e| format!("failed to read '{}': {}", path.display(), e))?;
        Self::parse(&data)
    }

    /// Returns the world-space terrain height at heightmap grid coordinate `(x, z)`,
    /// or `None` if `x > heightmap_width` or `z > heightmap_height`.
    #[allow(dead_code)]
    pub fn height_at(&self, x: u32, z: u32) -> Option<f32> {
        if x > self.heightmap_width || z > self.heightmap_height {
            return None;
        }
        let stride = (self.heightmap_width as usize).checked_add(1)?;
        let idx = (z as usize)
            .checked_mul(stride)?
            .checked_add(x as usize)?;
        self.heightmap
            .get(idx)
            .map(|&sample| sample as f32 * self.height_scale)
    }

    /// Returns `(min, max, avg)` height in world units, or `(0.0, 0.0, 0.0)` if empty.
    pub fn height_stats(&self) -> (f32, f32, f32) {
        if self.heightmap.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let first = self.heightmap[0] as f32 * self.height_scale;
        let mut min_h = first;
        let mut max_h = first;
        let mut sum: u64 = 0;
        for &sample in &self.heightmap {
            let h = sample as f32 * self.height_scale;
            min_h = min_h.min(h);
            max_h = max_h.max(h);
            sum += u64::from(sample);
        }
        let avg_h =
            ((sum as f64 / self.heightmap.len() as f64) * self.height_scale as f64) as f32;
        (min_h, max_h, avg_h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestScMapSpec<'a> {
        magic: &'a [u8; 4],
        version_major: i32,
        version_minor: i32,
        map_width: f32,
        map_height: f32,
        preview_dds: &'a [u8],
        hm_w: i32,
        hm_h: i32,
        height_scale: f32,
        samples: &'a [u16],
        terrain_shader: &'a [u8],
        background_path: &'a [u8],
        sky_cubemap_path: &'a [u8],
        cubemaps: &'a [(&'a [u8], &'a [u8])],
        water: WaterInfo,
    }

    fn build_synthetic_scmap(spec: &TestScMapSpec<'_>) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(spec.magic);
        buf.extend_from_slice(&spec.version_major.to_le_bytes());
        buf.extend_from_slice(&0x1122_3344u32.to_le_bytes());
        buf.extend_from_slice(&0x5566_7788u32.to_le_bytes());
        buf.extend_from_slice(&spec.map_width.to_le_bytes());
        buf.extend_from_slice(&spec.map_height.to_le_bytes());
        buf.extend_from_slice(&0x99AA_BBCCu32.to_le_bytes());
        buf.extend_from_slice(&0xDDEEu16.to_le_bytes());
        let preview_len = u32::try_from(spec.preview_dds.len()).expect("preview fits u32");
        buf.extend_from_slice(&preview_len.to_le_bytes());
        buf.extend_from_slice(spec.preview_dds);

        buf.extend_from_slice(&spec.version_minor.to_le_bytes());
        buf.extend_from_slice(&spec.hm_w.to_le_bytes());
        buf.extend_from_slice(&spec.hm_h.to_le_bytes());
        buf.extend_from_slice(&spec.height_scale.to_le_bytes());

        for &s in spec.samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }

        if spec.version_minor >= 56 {
            buf.push(0x01);
        }

        buf.extend_from_slice(spec.terrain_shader);
        buf.push(0);
        buf.extend_from_slice(spec.background_path);
        buf.push(0);
        buf.extend_from_slice(spec.sky_cubemap_path);
        buf.push(0);

        if spec.version_minor >= 60 {
            let count = u32::try_from(spec.cubemaps.len()).expect("cubemap count fits u32");
            buf.extend_from_slice(&count.to_le_bytes());
            for &(name, path) in spec.cubemaps {
                buf.extend_from_slice(name);
                buf.push(0);
                buf.extend_from_slice(path);
                buf.push(0);
            }
        }

        // 23 f32 lighting/fog values = 92 bytes.
        for i in 0..23u32 {
            buf.extend_from_slice(&(i as f32 * 0.25).to_le_bytes());
        }

        buf.push(u8::from(spec.water.has_water));
        buf.extend_from_slice(&spec.water.elevation.to_le_bytes());
        buf.extend_from_slice(&spec.water.elevation_deep.to_le_bytes());
        buf.extend_from_slice(&spec.water.elevation_abyss.to_le_bytes());
        buf
    }

    fn valid_v2_60_bytes() -> Vec<u8> {
        let samples = [10u16, 20, 30, 40, 50, 60, 70, 80, 90];
        let spec = TestScMapSpec {
            magic: b"Map\x1a",
            version_major: 2,
            version_minor: 60,
            map_width: 512.0,
            map_height: 1024.0,
            preview_dds: b"DDS test preview payload",
            hm_w: 2,
            hm_h: 2,
            height_scale: 0.5,
            samples: &samples,
            terrain_shader: b"TTerrain",
            background_path: b"/textures/environment/default_background.dds",
            sky_cubemap_path: b"/textures/environment/default_sky.dds",
            cubemaps: &[
                (b"default", b"/env/default.dds"),
                (b"water", b"/env/water.dds"),
            ],
            water: WaterInfo {
                has_water: true,
                elevation: 17.5,
                elevation_deep: 15.0,
                elevation_abyss: 5.0,
            },
        };
        build_synthetic_scmap(&spec)
    }

    #[test]
    fn parses_valid_v2_60_scmap() {
        let mut bytes = valid_v2_60_bytes();
        // Real .scmap files contain additional sections after water info.
        bytes.extend_from_slice(b"TRAILING_SECTION_BYTES");

        let map = ScMap::parse(&bytes).expect("valid v2.60 scmap should parse");
        assert_eq!(map.version_major, 2);
        assert_eq!(map.version_minor, 60);
        assert_eq!(map.map_width, 512.0);
        assert_eq!(map.map_height, 1024.0);
        assert_eq!(map.preview_dds, b"DDS test preview payload");
        assert_eq!(map.heightmap_width, 2);
        assert_eq!(map.heightmap_height, 2);
        assert_eq!(map.height_scale, 0.5);
        assert_eq!(map.heightmap, vec![10, 20, 30, 40, 50, 60, 70, 80, 90]);

        assert_eq!(map.height_at(0, 0), Some(5.0));
        assert_eq!(map.height_at(2, 0), Some(15.0));
        assert_eq!(map.height_at(1, 1), Some(25.0));
        assert_eq!(map.height_at(2, 2), Some(45.0));
        assert_eq!(map.height_at(3, 0), None);
        assert_eq!(map.height_at(0, 3), None);

        assert_eq!(map.height_stats(), (5.0, 45.0, 25.0));

        assert_eq!(map.terrain_shader, "TTerrain");
        assert_eq!(
            map.background_path,
            "/textures/environment/default_background.dds"
        );
        assert_eq!(
            map.sky_cubemap_path,
            "/textures/environment/default_sky.dds"
        );
        assert_eq!(
            map.cubemaps,
            vec![
                ("default".to_string(), "/env/default.dds".to_string()),
                ("water".to_string(), "/env/water.dds".to_string()),
            ]
        );

        assert!(map.water.has_water);
        assert_eq!(map.water.elevation, 17.5);
        assert_eq!(map.water.elevation_deep, 15.0);
        assert_eq!(map.water.elevation_abyss, 5.0);
    }

    #[test]
    fn parses_older_minor_versions() {
        for minor in [55, 56] {
            let samples = [100u16, 200, 300, 400];
            let spec = TestScMapSpec {
                magic: b"Map\x1a",
                version_major: 2,
                version_minor: minor,
                map_width: 256.0,
                map_height: 256.0,
                preview_dds: b"DDS",
                hm_w: 1,
                hm_h: 1,
                height_scale: 0.25,
                samples: &samples,
                terrain_shader: b"TTerrainXP",
                background_path: b"/env/bg.dds",
                sky_cubemap_path: b"/env/sky.dds",
                cubemaps: &[],
                water: WaterInfo {
                    has_water: false,
                    elevation: 0.0,
                    elevation_deep: 0.0,
                    elevation_abyss: 0.0,
                },
            };
            let bytes = build_synthetic_scmap(&spec);
            let map = ScMap::parse(&bytes).expect("older minor version should parse");
            assert_eq!(map.version_minor, minor);
            assert_eq!(map.terrain_shader, "TTerrainXP");
            assert_eq!(map.background_path, "/env/bg.dds");
            assert_eq!(map.sky_cubemap_path, "/env/sky.dds");
            assert!(map.cubemaps.is_empty());
            assert!(!map.water.has_water);
            assert_eq!(map.height_stats(), (25.0, 100.0, 62.5));
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = valid_v2_60_bytes();
        bytes[..4].copy_from_slice(b"BAD!");
        let err = ScMap::parse(&bytes).expect_err("bad magic must fail");
        assert!(err.contains("magic"));
    }

    #[test]
    fn rejects_all_truncated_prefixes() {
        let bytes = valid_v2_60_bytes();
        for len in 0..bytes.len() {
            assert!(
                ScMap::parse(&bytes[..len]).is_err(),
                "truncated buffer of length {} unexpectedly succeeded",
                len
            );
        }
    }

    #[test]
    fn rejects_invalid_version_and_dimensions() {
        let mut bad_ver = valid_v2_60_bytes();
        bad_ver[4..8].copy_from_slice(&3i32.to_le_bytes());
        assert!(ScMap::parse(&bad_ver).is_err());

        let samples = [0u16; 2];
        let zero_w = build_synthetic_scmap(&TestScMapSpec {
            magic: b"Map\x1a",
            version_major: 2,
            version_minor: 60,
            map_width: 256.0,
            map_height: 256.0,
            preview_dds: &[],
            hm_w: 0,
            hm_h: 1,
            height_scale: 1.0,
            samples: &samples,
            terrain_shader: b"S",
            background_path: b"B",
            sky_cubemap_path: b"K",
            cubemaps: &[],
            water: WaterInfo {
                has_water: false,
                elevation: 0.0,
                elevation_deep: 0.0,
                elevation_abyss: 0.0,
            },
        });
        assert!(ScMap::parse(&zero_w).is_err());
    }

    #[test]
    fn rejects_oversized_counts_without_panicking() {
        // Huge preview_dds_len on a small buffer.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"Map\x1a");
        buf.extend_from_slice(&2i32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 22]);
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(ScMap::parse(&buf).is_err());

        // Huge heightmap dimensions on a small buffer.
        let mut hm_buf = Vec::new();
        hm_buf.extend_from_slice(b"Map\x1a");
        hm_buf.extend_from_slice(&2i32.to_le_bytes());
        hm_buf.extend_from_slice(&[0u8; 26]); // unknown1..4 + preview_len=0
        hm_buf.extend_from_slice(&60i32.to_le_bytes());
        hm_buf.extend_from_slice(&i32::MAX.to_le_bytes());
        hm_buf.extend_from_slice(&i32::MAX.to_le_bytes());
        hm_buf.extend_from_slice(&1.0f32.to_le_bytes());
        assert!(ScMap::parse(&hm_buf).is_err());
    }

    #[test]
    fn decodes_latin1_fallback_strings() {
        let samples = [0u16; 4];
        let spec = TestScMapSpec {
            magic: b"Map\x1a",
            version_major: 2,
            version_minor: 60,
            map_width: 128.0,
            map_height: 128.0,
            preview_dds: &[],
            hm_w: 1,
            hm_h: 1,
            height_scale: 1.0,
            samples: &samples,
            terrain_shader: b"caf\xe9",
            background_path: b"bg",
            sky_cubemap_path: b"sky",
            cubemaps: &[],
            water: WaterInfo {
                has_water: false,
                elevation: 0.0,
                elevation_deep: 0.0,
                elevation_abyss: 0.0,
            },
        };
        let bytes = build_synthetic_scmap(&spec);
        let map = ScMap::parse(&bytes).expect("latin1 string should decode");
        assert_eq!(map.terrain_shader, "café");
    }

    #[test]
    fn height_stats_empty_returns_zeroes() {
        let map = ScMap {
            version_major: 2,
            version_minor: 60,
            map_width: 0.0,
            map_height: 0.0,
            preview_dds: Vec::new(),
            heightmap_width: 0,
            heightmap_height: 0,
            height_scale: 0.5,
            heightmap: Vec::new(),
            terrain_shader: String::new(),
            background_path: String::new(),
            sky_cubemap_path: String::new(),
            cubemaps: Vec::new(),
            water: WaterInfo {
                has_water: false,
                elevation: 0.0,
                elevation_deep: 0.0,
                elevation_abyss: 0.0,
            },
            terrain_textures: None,
        };
        assert_eq!(map.height_stats(), (0.0, 0.0, 0.0));
        assert_eq!(map.height_at(0, 0), None);
    }

    #[test]
    fn parse_file_reads_and_validates() {
        let mut path = std::env::temp_dir();
        path.push(format!("farust_test_{}.scmap", std::process::id()));

        let bytes = valid_v2_60_bytes();
        fs::write(&path, &bytes).expect("write temp scmap");
        let parsed = ScMap::parse_file(&path);
        let _ = fs::remove_file(&path);

        let map = parsed.expect("parse_file should succeed");
        assert_eq!(map.terrain_shader, "TTerrain");
        assert_eq!(map.height_stats(), (5.0, 45.0, 25.0));

        assert!(ScMap::parse_file(&path).is_err());
    }
}
