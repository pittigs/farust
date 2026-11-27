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

/// Sun, ambient, shadow, and atmospheric fog parameters parsed from `.scmap`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ScMapLighting {
    pub lighting_multiplier: f32,
    pub sun_direction: [f32; 3],
    pub sun_ambience: [f32; 3],
    pub sun_color: [f32; 3],
    pub shadow_fill_color: [f32; 3],
    pub specular_color: [f32; 4],
    pub bloom: f32,
    pub fog_color: [f32; 3],
    pub fog_start: f32,
    pub fog_end: f32,
}

impl Default for ScMapLighting {
    fn default() -> Self {
        Self {
            lighting_multiplier: 1.4,
            sun_direction: [0.46, 0.71, 0.53],
            sun_ambience: [0.18, 0.20, 0.24],
            sun_color: [1.35, 1.28, 1.18],
            shadow_fill_color: [0.35, 0.38, 0.48],
            specular_color: [0.5, 0.5, 0.5, 0.5],
            bloom: 0.08,
            fog_color: [0.58, 0.72, 0.85],
            fog_start: 100.0,
            fog_end: 1000.0,
        }
    }
}

/// Water shader parameters parsed from `.scmap`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ScMapWaterParams {
    pub surface_color: [f32; 3],
    pub color_lerp: [f32; 2],
    pub fresnel_bias: f32,
    pub fresnel_power: f32,
    pub water_cubemap: String,
    pub water_ramp: String,
    pub wave_textures: Vec<String>,
}

impl Default for ScMapWaterParams {
    fn default() -> Self {
        Self {
            surface_color: [0.06, 0.09, 0.12],
            color_lerp: [0.25, 0.05],
            fresnel_bias: 0.06,
            fresnel_power: 1.8,
            water_cubemap: String::new(),
            water_ramp: String::new(),
            wave_textures: Vec::new(),
        }
    }
}

/// One placed prop instance (tree, rock, log, boulder) in `.scmap`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ScMapProp {
    pub blueprint_path: String,
    pub position: [f32; 3],
    pub rot_x: [f32; 3],
    pub rot_y: [f32; 3],
    pub rot_z: [f32; 3],
    pub scale: [f32; 3],
}

/// One terrain decal placement parsed from `.scmap`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ScMapDecal {
    pub id: u32,
    pub decal_type: u32,
    pub textures: Vec<String>,
    pub scale: [f32; 3],
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub cutoff_lod: f32,
    pub near_cutoff_lod: f32,
}

/// Terrain strata descriptors, water params, embedded blend maps, decals, and props parsed from `.scmap`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TerrainTextures {
    pub strata_albedo: Vec<TerrainStratum>,
    pub strata_normal: Vec<TerrainStratum>,
    pub decals: Vec<ScMapDecal>,
    pub texturemap0_dds: Vec<u8>,
    pub texturemap1_dds: Vec<u8>,
    pub normal_map_dds: Vec<u8>,
    pub water_params: ScMapWaterParams,
    pub props: Vec<ScMapProp>,
}

/// Parsed Supreme Commander `.scmap` map data (header, heightmap, environment paths, lighting, and water).
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
    pub lighting: ScMapLighting,
    pub water: WaterInfo,
    pub terrain_textures: Option<TerrainTextures>,
    /// Header fields the parser does not interpret, retained for the writer.
    pub unknown1: u32,
    pub unknown2: u32,
    pub unknown3: u32,
    pub unknown4: u16,
    /// Byte after the heightmap (present when `version_minor >= 56`).
    pub post_heightmap_byte: u8,
    /// Original bytes of `terrain_shader` (may be non-UTF-8); empty = use the string.
    pub shader_raw: Vec<u8>,
    pub background_raw: Vec<u8>,
    pub sky_raw: Vec<u8>,
    pub cubemaps_raw: Vec<(Vec<u8>, Vec<u8>)>,
    /// Everything after the water section, verbatim (strata/decals/props/…).
    pub tail_bytes: Vec<u8>,
}

/// Heightmap brush mode for [`ScMap::apply_brush`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrushMode {
    Raise,
    Lower,
    Smooth,
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
        Ok(self.read_cstr_raw()?.0)
    }

    fn read_cstr_raw(&mut self) -> Result<(String, Vec<u8>), String> {
        let rest = &self.data[self.pos..];
        let rel_end = rest.iter().position(|&b| b == 0).ok_or_else(|| {
            format!("unterminated string starting at byte {}", self.pos)
        })?;
        let bytes = &rest[..rel_end];
        self.pos += rel_end + 1;
        Ok((decode_cstr_bytes(bytes), bytes.to_vec()))
    }
}

fn decode_cstr_bytes(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_owned(),
        Err(_) => bytes.iter().map(|&b| b as char).collect(),
    }
}

/// Parses the v2.60 post-water water/wave/contour/stratum/decal/blendmap/skybox/props sections.
/// Returns `None` on short synthetic test tails where full strata/DDS payloads are absent.
fn parse_terrain_textures(tail: &[u8], hm_w: u32, hm_h: u32) -> Option<TerrainTextures> {
    if tail.len() < 128 {
        return None;
    }
    let mut r = Reader::new(tail);
    // 20 f32 water shading parameters
    let mut wf = [0.0_f32; 20];
    for slot in &mut wf {
        *slot = r.read_f32().ok()?;
    }
    let water_cubemap = r.read_cstr().ok()?;
    let water_ramp = r.read_cstr().ok()?;
    // 4 wave normal repeats
    r.skip(16).ok()?;
    // 4 wave layers (vec2 movement + texture path)
    let mut wave_textures = Vec::with_capacity(4);
    for _ in 0..4 {
        r.skip(8).ok()?;
        wave_textures.push(r.read_cstr().ok()?);
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
    let mut decals = Vec::with_capacity(decal_count);
    for _ in 0..decal_count {
        let id = r.read_u32().ok()?;
        let decal_type = r.read_u32().ok()?;
        let ntex = usize::try_from(r.read_u32().ok()?).ok()?;
        if ntex > 16 {
            return None;
        }
        let mut textures = Vec::with_capacity(ntex);
        for _ in 0..ntex {
            let tlen = usize::try_from(r.read_u32().ok()?).ok()?;
            let tb = r.read_bytes(tlen).ok()?;
            textures.push(tb.iter().map(|&b| char::from(b)).collect());
        }
        let mut vals = [0.0_f32; 12];
        for v in &mut vals {
            *v = r.read_f32().ok()?;
        }
        decals.push(ScMapDecal {
            id,
            decal_type,
            textures,
            scale: [vals[0], vals[1], vals[2]],
            position: [vals[3], vals[4], vals[5]],
            rotation: [vals[6], vals[7], vals[8]],
            cutoff_lod: vals[9],
            near_cutoff_lod: vals[10],
        });
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
    let mut normal_map_dds = Vec::new();
    for i in 0..norm_count {
        let len = usize::try_from(r.read_u32().ok()?).ok()?;
        let bytes = r.read_bytes(len).ok()?;
        if i == 0 && bytes.starts_with(b"DDS ") {
            normal_map_dds = bytes.to_vec();
        }
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

    let props = parse_props_tail(&mut r, hm_w, hm_h).unwrap_or_default();

    let water_params = ScMapWaterParams {
        surface_color: [wf[0], wf[1], wf[2]],
        color_lerp: [wf[3], wf[4]],
        fresnel_bias: if wf[6].is_finite() { wf[6] } else { 0.06 },
        fresnel_power: if wf[7].is_finite() && wf[7] > 0.0 { wf[7] } else { 1.8 },
        water_cubemap,
        water_ramp,
        wave_textures,
    };

    Some(TerrainTextures {
        strata_albedo,
        strata_normal,
        decals,
        texturemap0_dds,
        texturemap1_dds,
        normal_map_dds,
        water_params,
        props,
    })
}

fn parse_props_tail(r: &mut Reader<'_>, hm_w: u32, hm_h: u32) -> Option<Vec<ScMapProp>> {
    // Unknown v60 u32 (usually 1) + watermap DDS length/payload
    r.skip(4).ok()?;
    let wmap_len = usize::try_from(r.read_u32().ok()?).ok()?;
    r.skip(wmap_len).ok()?;

    // Half-res water foam/flatness/depthbias + full-res terrain-type bytes
    let half = ((hm_w as usize) / 2).checked_mul((hm_h as usize) / 2)?;
    let terrain_masks_len = half
        .checked_mul(3)?
        .checked_add((hm_w as usize).checked_mul(hm_h as usize)?)?;
    r.skip(terrain_masks_len).ok()?;

    // SkyBox + Planets + Cirrus layers (v2.60)
    r.skip(64).ok()?;
    let _sky_albedo = r.read_cstr().ok()?;
    let _sky_glow = r.read_cstr().ok()?;
    let planet_count = usize::try_from(r.read_u32().ok()?).ok()?;
    let planets_bytes = planet_count.checked_mul(40)?;
    r.skip(planets_bytes).ok()?;
    r.skip(19).ok()?;
    let _cirrus_tex = r.read_cstr().ok()?;
    let cirrus_count = usize::try_from(r.read_u32().ok()?).ok()?;
    let cirrus_bytes = cirrus_count.checked_mul(20)?;
    r.skip(cirrus_bytes).ok()?;
    r.skip(4).ok()?;

    let prop_count = usize::try_from(r.read_u32().ok()?).ok()?;
    if prop_count > r.remaining() / 61 {
        return None;
    }
    let mut props = Vec::with_capacity(prop_count);
    for _ in 0..prop_count {
        let blueprint_path = r.read_cstr().ok()?;
        let mut vals = [0.0_f32; 15];
        for v in &mut vals {
            *v = r.read_f32().ok()?;
        }
        props.push(ScMapProp {
            blueprint_path,
            position: [vals[0], vals[1], vals[2]],
            rot_x: [vals[3], vals[4], vals[5]],
            rot_y: [vals[6], vals[7], vals[8]],
            rot_z: [vals[9], vals[10], vals[11]],
            scale: [vals[12], vals[13], vals[14]],
        });
    }
    Some(props)
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

        let unknown1 = r.read_u32()?;
        let unknown2 = r.read_u32()?;
        let map_width = r.read_f32()?;
        let map_height = r.read_f32()?;
        let unknown3 = r.read_u32()?;
        let unknown4 = r.read_u16()?;

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

        let post_heightmap_byte = if version_minor >= 56 { r.read_u8()? } else { 0 };

        let (terrain_shader, shader_raw) = r.read_cstr_raw()?;
        let (background_path, background_raw) = r.read_cstr_raw()?;
        let (sky_cubemap_path, sky_raw) = r.read_cstr_raw()?;

        let mut cubemaps_raw = Vec::new();
        let cubemaps = if version_minor >= 60 {
            let count = usize::try_from(r.read_u32()?)
                .map_err(|_| "cubemap count exceeds addressable range".to_string())?;
            // Cap preallocation by remaining bytes (each pair needs at least two null bytes) to avoid OOM on corrupt headers.
            let mut pairs = Vec::with_capacity(count.min(r.remaining() / 2));
            for _ in 0..count {
                let (name, name_raw) = r.read_cstr_raw()?;
                let (path, path_raw) = r.read_cstr_raw()?;
                cubemaps_raw.push((name_raw, path_raw));
                pairs.push((name, path));
            }
            pairs
        } else {
            Vec::new()
        };

        // 23 f32 lighting/fog parameters.
        let mut lf = [0.0_f32; 23];
        for slot in &mut lf {
            *slot = r.read_f32()?;
        }
        let lighting = ScMapLighting {
            lighting_multiplier: lf[0],
            sun_direction: [lf[1], lf[2], lf[3]],
            sun_ambience: [lf[4], lf[5], lf[6]],
            sun_color: [lf[7], lf[8], lf[9]],
            shadow_fill_color: [lf[10], lf[11], lf[12]],
            specular_color: [lf[13], lf[14], lf[15], lf[16]],
            bloom: lf[17],
            fog_color: [lf[18], lf[19], lf[20]],
            fog_start: lf[21],
            fog_end: lf[22],
        };

        let has_water = r.read_u8()? != 0;
        let elevation = r.read_f32()?;
        let elevation_deep = r.read_f32()?;
        let elevation_abyss = r.read_f32()?;

        let tail_bytes = data[r.pos..].to_vec();
        let terrain_textures = parse_terrain_textures(&data[r.pos..], hm_w as u32, hm_h as u32);

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
            lighting,
            water: WaterInfo {
                has_water,
                elevation,
                elevation_deep,
                elevation_abyss,
            },
            terrain_textures,
            unknown1,
            unknown2,
            unknown3,
            unknown4,
            post_heightmap_byte,
            shader_raw,
            background_raw,
            sky_raw,
            cubemaps_raw,
            tail_bytes,
        })
    }

    /// Serializes back to `.scmap` bytes. The structured header is rewritten
    /// from the parsed fields; the post-water tail is emitted verbatim from
    /// `tail_bytes`, so `write(parse(bytes)) == bytes` for stock maps.
    pub fn write(&self) -> Vec<u8> {
        fn cstr(out: &mut Vec<u8>, raw: &[u8], text: &str) {
            if raw.is_empty() {
                out.extend_from_slice(text.as_bytes());
            } else {
                out.extend_from_slice(raw);
            }
            out.push(0);
        }
        let mut out = Vec::with_capacity(64 + self.heightmap.len() * 2 + self.tail_bytes.len());
        out.extend_from_slice(b"Map\x1a");
        out.extend_from_slice(&self.version_major.to_le_bytes());
        out.extend_from_slice(&self.unknown1.to_le_bytes());
        out.extend_from_slice(&self.unknown2.to_le_bytes());
        out.extend_from_slice(&self.map_width.to_le_bytes());
        out.extend_from_slice(&self.map_height.to_le_bytes());
        out.extend_from_slice(&self.unknown3.to_le_bytes());
        out.extend_from_slice(&self.unknown4.to_le_bytes());
        out.extend_from_slice(&(self.preview_dds.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.preview_dds);
        out.extend_from_slice(&self.version_minor.to_le_bytes());
        out.extend_from_slice(&(self.heightmap_width as i32).to_le_bytes());
        out.extend_from_slice(&(self.heightmap_height as i32).to_le_bytes());
        out.extend_from_slice(&self.height_scale.to_le_bytes());
        for &s in &self.heightmap {
            out.extend_from_slice(&s.to_le_bytes());
        }
        if self.version_minor >= 56 {
            out.push(self.post_heightmap_byte);
        }
        cstr(&mut out, &self.shader_raw, &self.terrain_shader);
        cstr(&mut out, &self.background_raw, &self.background_path);
        cstr(&mut out, &self.sky_raw, &self.sky_cubemap_path);
        if self.version_minor >= 60 {
            out.extend_from_slice(&(self.cubemaps.len() as u32).to_le_bytes());
            for (i, (name, path)) in self.cubemaps.iter().enumerate() {
                let (nr, pr) = self
                    .cubemaps_raw
                    .get(i)
                    .map(|(a, b)| (a.as_slice(), b.as_slice()))
                    .unwrap_or((&[], &[]));
                cstr(&mut out, nr, name);
                cstr(&mut out, pr, path);
            }
        }
        let l = &self.lighting;
        let lf = [
            l.lighting_multiplier,
            l.sun_direction[0], l.sun_direction[1], l.sun_direction[2],
            l.sun_ambience[0], l.sun_ambience[1], l.sun_ambience[2],
            l.sun_color[0], l.sun_color[1], l.sun_color[2],
            l.shadow_fill_color[0], l.shadow_fill_color[1], l.shadow_fill_color[2],
            l.specular_color[0], l.specular_color[1], l.specular_color[2], l.specular_color[3],
            l.bloom,
            l.fog_color[0], l.fog_color[1], l.fog_color[2],
            l.fog_start, l.fog_end,
        ];
        for v in lf {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.push(u8::from(self.water.has_water));
        out.extend_from_slice(&self.water.elevation.to_le_bytes());
        out.extend_from_slice(&self.water.elevation_deep.to_le_bytes());
        out.extend_from_slice(&self.water.elevation_abyss.to_le_bytes());
        out.extend_from_slice(&self.tail_bytes);
        out
    }

    /// Writes [`Self::write`] to disk.
    pub fn write_file(&self, path: &Path) -> Result<(), String> {
        fs::write(path, self.write()).map_err(|e| format!("failed to write '{}': {e}", path.display()))
    }

    /// Applies a heightmap brush in world coordinates (`x`, `z` on the map,
    /// `radius` in world units, `strength` in world height units for
    /// raise/lower). Returns the number of samples changed.
    pub fn apply_brush(&mut self, x: f32, z: f32, radius: f32, strength: f32, mode: BrushMode) -> usize {
        if self.heightmap_width == 0 || self.heightmap_height == 0 || radius <= 0.0 || self.height_scale == 0.0 {
            return 0;
        }
        let stride = self.heightmap_width as usize + 1;
        let gx = x / self.map_width.max(1.0) * self.heightmap_width as f32;
        let gz = z / self.map_height.max(1.0) * self.heightmap_height as f32;
        let gr = radius / self.map_width.max(1.0) * self.heightmap_width as f32;
        let x0 = (gx - gr).floor().max(0.0) as usize;
        let z0 = (gz - gr).floor().max(0.0) as usize;
        let x1 = ((gx + gr).ceil() as usize).min(self.heightmap_width as usize);
        let z1 = ((gz + gr).ceil() as usize).min(self.heightmap_height as usize);
        let mut changed = 0usize;
        // Snapshot for smooth mode so the pass is order-independent.
        let orig = (mode == BrushMode::Smooth).then(|| self.heightmap.clone());
        for gz_i in z0..=z1 {
            for gx_i in x0..=x1 {
                let d = ((gx_i as f32 - gx).powi(2) + (gz_i as f32 - gz).powi(2)).sqrt() / gr.max(0.001);
                if d > 1.0 {
                    continue;
                }
                let fall = 0.5 + 0.5 * (d * std::f32::consts::PI).cos();
                let idx = gz_i * stride + gx_i;
                let Some(slot) = self.heightmap.get_mut(idx) else { continue };
                let before = *slot;
                match mode {
                    BrushMode::Raise | BrushMode::Lower => {
                        let sign = if mode == BrushMode::Raise { 1.0 } else { -1.0 };
                        let delta = (sign * strength * fall / self.height_scale).round() as i32;
                        *slot = (*slot as i32 + delta).clamp(0, u16::MAX as i32) as u16;
                    }
                    BrushMode::Smooth => {
                        let src = orig.as_ref().unwrap();
                        let mut sum = 0u32;
                        let mut n = 0u32;
                        for dz in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let nx = gx_i as i32 + dx;
                                let nz = gz_i as i32 + dz;
                                if nx >= 0 && nz >= 0 && (nx as usize) <= self.heightmap_width as usize && (nz as usize) <= self.heightmap_height as usize {
                                    if let Some(&v) = src.get(nz as usize * stride + nx as usize) {
                                        sum += v as u32;
                                        n += 1;
                                    }
                                }
                            }
                        }
                        if n > 0 {
                            let avg = (sum / n) as i32;
                            let cur = *slot as i32;
                            *slot = (cur + ((avg - cur) as f32 * fall).round() as i32).clamp(0, u16::MAX as i32) as u16;
                        }
                    }
                }
                if *slot != before {
                    changed += 1;
                }
            }
        }
        changed
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
            lighting: ScMapLighting::default(),
            water: WaterInfo {
                has_water: false,
                elevation: 0.0,
                elevation_deep: 0.0,
                elevation_abyss: 0.0,
            },
            unknown1: 0, unknown2: 0, unknown3: 0, unknown4: 0, post_heightmap_byte: 0,
            shader_raw: Vec::new(), background_raw: Vec::new(), sky_raw: Vec::new(), cubemaps_raw: Vec::new(), tail_bytes: Vec::new(),
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

    #[test]
    fn writer_roundtrips_synthetic_byte_for_byte() {
        let mut bytes = valid_v2_60_bytes();
        bytes.extend_from_slice(b"TRAILING_SECTION_BYTES");
        let map = ScMap::parse(&bytes).expect("parse");
        assert_eq!(map.write(), bytes, "write(parse(bytes)) must be byte-identical");
        let reparsed = ScMap::parse(&map.write()).expect("reparse");
        assert_eq!(reparsed.heightmap, map.heightmap);
        assert_eq!(reparsed.tail_bytes, map.tail_bytes);
    }

    #[test]
    fn writer_roundtrips_edited_heightmap() {
        let bytes = valid_v2_60_bytes();
        let mut map = ScMap::parse(&bytes).expect("parse");
        map.heightmap[4] = 999;
        let written = map.write();
        assert_ne!(written, bytes);
        let reparsed = ScMap::parse(&written).expect("reparse edited");
        assert_eq!(reparsed.heightmap[4], 999);
        assert_eq!(reparsed.write(), written);
    }

    #[test]
    fn brush_raise_changes_center_only() {
        let bytes = valid_v2_60_bytes();
        let mut map = ScMap::parse(&bytes).expect("parse");
        let before = map.heightmap.clone();
        let changed = map.apply_brush(256.0, 512.0, 100.0, 10.0, BrushMode::Raise);
        assert!(changed > 0);
        assert!(map.heightmap.iter().zip(&before).any(|(a, b)| a > b));
        // Far corner (grid 0,0 is at world 0,0) untouched by a small brush at centre-ish.
        assert_eq!(map.heightmap[0], before[0]);
        let n = map.apply_brush(256.0, 512.0, 100.0, 10.0, BrushMode::Lower);
        assert!(n > 0);
    }

    #[test]
    fn parses_lighting_parameters() {
        let bytes = valid_v2_60_bytes();
        let map = ScMap::parse(&bytes).expect("parse synthetic");
        assert!((map.lighting.lighting_multiplier - 0.0).abs() < 1e-6);
        assert!((map.lighting.sun_direction[0] - 0.25).abs() < 1e-6);
        assert!((map.lighting.fog_end - 5.5).abs() < 1e-6);
    }
}
