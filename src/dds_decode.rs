//! Pure-Rust RGBA8 mip-0 decoder for Forged Alliance `.dds` textures.

use crate::dds::{DdsFormat, DdsHeader, parse_dds};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Two-channel linear `(specular, glow)` map extracted from a Forged Alliance `*_SpecTeam.dds`
/// (`G` = Phong specular mask, `B` = emissive/glow mask per FA's `effects/mesh.fx`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecGlowMap {
    pub width: u32,
    pub height: u32,
    /// Interleaved `[spec, glow]` pairs (`width * height * 2` bytes), uploaded as `Rg8Unorm`.
    pub pixels: Vec<u8>,
}

impl SpecGlowMap {
    /// Extracts `(spec = G, glow = B)` from a decoded RGBA8 `*_SpecTeam.dds` image.
    pub fn from_spec_team(spec: &RgbaImage) -> Self {
        let count = (spec.width as usize * spec.height as usize).min(spec.pixels.len() / 4);
        let mut pixels = Vec::with_capacity(count * 2);
        for &[_, g, b, _] in spec.pixels.as_chunks::<4>().0.iter().take(count) {
            pixels.push(g); // G = specular intensity
            pixels.push(b); // B = emissive/glow mask
        }
        Self {
            width: spec.width,
            height: spec.height,
            pixels,
        }
    }
}

pub fn decode_dds_rgba8(data: &[u8]) -> Result<RgbaImage, String> {
    let header = parse_dds(data)?;
    let payload = data
        .get(header.data_offset..)
        .ok_or_else(|| "DDS data offset out of bounds".to_string())?;
    decode_payload_rgba8(&header, payload)
}

pub fn decode_payload_rgba8(header: &DdsHeader, payload: &[u8]) -> Result<RgbaImage, String> {
    let width = header.width;
    let height = header.height;
    if width == 0 || height == 0 {
        return Err("DDS image dimensions must be non-zero".to_string());
    }

    let total_bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| "image dimensions overflow".to_string())?;
    // Validate the payload covers the image *before* allocating the output, so corrupt
    // dimensions can never trigger a huge allocation.
    let required = dds_mip_byte_len(&header.format, width, height)
        .map_err(|_| format!("unsupported DDS format for RGBA8 decode: {}", header.format))?;
    if payload.len() < required {
        return Err(format!(
            "truncated DDS payload: expected at least {required} bytes for {width}x{height} {}, got {}",
            header.format,
            payload.len()
        ));
    }
    let mut pixels = vec![0u8; total_bytes];

    match header.format {
        DdsFormat::Dxt1 => decode_dxt1(width, height, payload, &mut pixels)?,
        DdsFormat::Dxt3 => decode_dxt3(width, height, payload, &mut pixels)?,
        DdsFormat::Dxt5 => decode_dxt5(width, height, payload, &mut pixels)?,
        DdsFormat::Bgra8 => {
            let stride =
                resolve_row_stride(header.pitch_or_linear_size, width, height, 4, payload.len())?;
            let w = width as usize;
            for y in 0..height as usize {
                let src_row = &payload[y * stride..y * stride + w * 4];
                let dst_row = &mut pixels[y * w * 4..(y + 1) * w * 4];
                for x in 0..w {
                    let s = x * 4;
                    dst_row[s] = src_row[s + 2];
                    dst_row[s + 1] = src_row[s + 1];
                    dst_row[s + 2] = src_row[s];
                    dst_row[s + 3] = src_row[s + 3];
                }
            }
        }
        DdsFormat::Rgba8 => {
            let stride =
                resolve_row_stride(header.pitch_or_linear_size, width, height, 4, payload.len())?;
            let w = width as usize;
            for y in 0..height as usize {
                let src_row = &payload[y * stride..y * stride + w * 4];
                let dst_row = &mut pixels[y * w * 4..(y + 1) * w * 4];
                dst_row.copy_from_slice(src_row);
            }
        }
        DdsFormat::Bgr8 => {
            let stride =
                resolve_row_stride(header.pitch_or_linear_size, width, height, 3, payload.len())?;
            let w = width as usize;
            for y in 0..height as usize {
                let src_row = &payload[y * stride..y * stride + w * 3];
                let dst_row = &mut pixels[y * w * 4..(y + 1) * w * 4];
                for x in 0..w {
                    let s = x * 3;
                    let d = x * 4;
                    dst_row[d] = src_row[s + 2];
                    dst_row[d + 1] = src_row[s + 1];
                    dst_row[d + 2] = src_row[s];
                    dst_row[d + 3] = 255;
                }
            }
        }
        DdsFormat::A8 => {
            let stride =
                resolve_row_stride(header.pitch_or_linear_size, width, height, 1, payload.len())?;
            let w = width as usize;
            for y in 0..height as usize {
                let src_row = &payload[y * stride..y * stride + w];
                let dst_row = &mut pixels[y * w * 4..(y + 1) * w * 4];
                for (x, &a) in src_row.iter().enumerate() {
                    let d = x * 4;
                    dst_row[d] = 255;
                    dst_row[d + 1] = 255;
                    dst_row[d + 2] = 255;
                    dst_row[d + 3] = a;
                }
            }
        }
        DdsFormat::L8 => {
            let stride =
                resolve_row_stride(header.pitch_or_linear_size, width, height, 1, payload.len())?;
            let w = width as usize;
            for y in 0..height as usize {
                let src_row = &payload[y * stride..y * stride + w];
                let dst_row = &mut pixels[y * w * 4..(y + 1) * w * 4];
                for (x, &l) in src_row.iter().enumerate() {
                    let d = x * 4;
                    dst_row[d] = l;
                    dst_row[d + 1] = l;
                    dst_row[d + 2] = l;
                    dst_row[d + 3] = 255;
                }
            }
        }
        _ => {
            return Err(format!(
                "unsupported DDS format for RGBA8 decode: {}",
                header.format
            ));
        }
    }

    Ok(RgbaImage {
        width,
        height,
        pixels,
    })
}

fn dds_mip_byte_len(format: &DdsFormat, w: u32, h: u32) -> Result<usize, String> {
    let (bw, bh, unit) = match format {
        DdsFormat::Dxt1 => (w.div_ceil(4) as usize, h.div_ceil(4) as usize, 8),
        DdsFormat::Dxt3 | DdsFormat::Dxt5 => (w.div_ceil(4) as usize, h.div_ceil(4) as usize, 16),
        DdsFormat::Bgra8 | DdsFormat::Rgba8 => (w as usize, h as usize, 4),
        DdsFormat::Bgr8 => (w as usize, h as usize, 3),
        DdsFormat::A8 | DdsFormat::L8 => (w as usize, h as usize, 1),
        _ => return Err(format!("unsupported cubemap DDS format: {format}")),
    };
    bw.checked_mul(bh)
        .and_then(|n| n.checked_mul(unit))
        .ok_or_else(|| "DDS mip byte size overflow".to_string())
}

/// Decodes all 6 cubemap faces (`+X, -X, +Y, -Y, +Z, -Z` at mip 0) from a DDS cubemap file.
pub fn decode_dds_cubemap_rgba8(data: &[u8]) -> Result<[RgbaImage; 6], String> {
    let header = parse_dds(data)?;
    if !header.is_cubemap {
        return Err("DDS image is not a cubemap".to_string());
    }
    let payload = data
        .get(header.data_offset..)
        .ok_or_else(|| "DDS cubemap data offset out of bounds".to_string())?;
    // A u32 mip chain is at most 32 levels deep; corrupt counts must not loop for billions.
    let mips = header.mip_count.clamp(1, 32);
    let mut face_bytes: usize = 0;
    for m in 0..mips {
        let mw = header.width.checked_shr(m).unwrap_or(0).max(1);
        let mh = header.height.checked_shr(m).unwrap_or(0).max(1);
        face_bytes = face_bytes
            .checked_add(dds_mip_byte_len(&header.format, mw, mh)?)
            .ok_or_else(|| "cubemap face byte size overflow".to_string())?;
    }
    let single_face_payload = |face_idx: usize| -> Result<&[u8], String> {
        let truncated = || format!("truncated cubemap face {face_idx}");
        let start = face_idx.checked_mul(face_bytes).ok_or_else(truncated)?;
        let mip0 = dds_mip_byte_len(&header.format, header.width, header.height)?;
        let end = start.checked_add(mip0).ok_or_else(truncated)?;
        payload.get(start..end).ok_or_else(truncated)
    };
    let mut faces = Vec::with_capacity(6);
    for f in 0..6 {
        faces.push(decode_payload_rgba8(&header, single_face_payload(f)?)?);
    }
    faces
        .try_into()
        .map_err(|_| "cubemap decode failed".to_string())
}

/// Resizes an [`RgbaImage`] with repeating bilinear filtering (safe for tiled terrain textures).
pub fn resize_rgba_bilinear(img: &RgbaImage, dst_w: u32, dst_h: u32) -> RgbaImage {
    let dst_w = dst_w.max(1);
    let dst_h = dst_h.max(1);
    if img.width == dst_w && img.height == dst_h {
        return img.clone();
    }
    let sw = img.width.max(1) as usize;
    let sh = img.height.max(1) as usize;
    let mut out = vec![0u8; (dst_w as usize) * (dst_h as usize) * 4];
    for y in 0..dst_h as usize {
        let fy = (y as f32 + 0.5) / dst_h as f32 * sh as f32 - 0.5;
        let y0 = (fy.floor() as isize).rem_euclid(sh as isize) as usize;
        let y1 = (y0 + 1) % sh;
        let ty = fy - fy.floor();
        for x in 0..dst_w as usize {
            let fx = (x as f32 + 0.5) / dst_w as f32 * sw as f32 - 0.5;
            let x0 = (fx.floor() as isize).rem_euclid(sw as isize) as usize;
            let x1 = (x0 + 1) % sw;
            let tx = fx - fx.floor();
            let dst = (y * dst_w as usize + x) * 4;
            for c in 0..4 {
                let c00 = img.pixels[(y0 * sw + x0) * 4 + c] as f32;
                let c10 = img.pixels[(y0 * sw + x1) * 4 + c] as f32;
                let c01 = img.pixels[(y1 * sw + x0) * 4 + c] as f32;
                let c11 = img.pixels[(y1 * sw + x1) * 4 + c] as f32;
                let a = c00 + (c10 - c00) * tx;
                let b = c01 + (c11 - c01) * tx;
                out[dst + c] = (a + (b - a) * ty).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    RgbaImage {
        width: dst_w,
        height: dst_h,
        pixels: out,
    }
}

fn resolve_row_stride(
    pitch_or_linear_size: u32,
    width: u32,
    height: u32,
    bytes_per_pixel: usize,
    payload_len: usize,
) -> Result<usize, String> {
    let min_stride = (width as usize)
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| "row stride overflow".to_string())?;
    let tight_size = min_stride
        .checked_mul(height as usize)
        .ok_or_else(|| "image size overflow".to_string())?;

    let candidate_stride = if (pitch_or_linear_size as usize) >= min_stride {
        pitch_or_linear_size as usize
    } else {
        min_stride
    };

    match candidate_stride.checked_mul(height as usize) {
        Some(pitched_size) if pitched_size <= payload_len => Ok(candidate_stride),
        _ if tight_size <= payload_len => Ok(min_stride),
        _ => Err(format!(
            "truncated DDS payload: expected at least {} bytes, got {}",
            tight_size, payload_len
        )),
    }
}

fn check_block_grid(
    width: u32,
    height: u32,
    block_bytes: usize,
    payload_len: usize,
) -> Result<(usize, usize), String> {
    let blocks_x = width.div_ceil(4) as usize;
    let blocks_y = height.div_ceil(4) as usize;
    let required = blocks_x
        .checked_mul(blocks_y)
        .and_then(|n| n.checked_mul(block_bytes))
        .ok_or_else(|| "DDS block size overflow".to_string())?;
    if payload_len < required {
        return Err(format!(
            "truncated DDS block payload: expected at least {} bytes, got {}",
            required, payload_len
        ));
    }
    Ok((blocks_x, blocks_y))
}

#[inline]
fn unpack_rgb565(c: u16) -> [u8; 3] {
    let r5 = ((c >> 11) & 0x1F) as u8;
    let g6 = ((c >> 5) & 0x3F) as u8;
    let b5 = (c & 0x1F) as u8;
    [
        (r5 << 3) | (r5 >> 2),
        (g6 << 2) | (g6 >> 4),
        (b5 << 3) | (b5 >> 2),
    ]
}

#[inline]
fn decode_bc1_color_block(block: &[u8], force_four_color: bool) -> ([[u8; 4]; 4], u32) {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let lut = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);

    let [r0, g0, b0] = unpack_rgb565(c0);
    let [r1, g1, b1] = unpack_rgb565(c1);

    let palette = if force_four_color || c0 > c1 {
        [
            [r0, g0, b0, 255],
            [r1, g1, b1, 255],
            [
                ((2 * r0 as u16 + r1 as u16 + 1) / 3) as u8,
                ((2 * g0 as u16 + g1 as u16 + 1) / 3) as u8,
                ((2 * b0 as u16 + b1 as u16 + 1) / 3) as u8,
                255,
            ],
            [
                ((r0 as u16 + 2 * r1 as u16 + 1) / 3) as u8,
                ((g0 as u16 + 2 * g1 as u16 + 1) / 3) as u8,
                ((b0 as u16 + 2 * b1 as u16 + 1) / 3) as u8,
                255,
            ],
        ]
    } else {
        [
            [r0, g0, b0, 255],
            [r1, g1, b1, 255],
            [
                ((r0 as u16 + r1 as u16) / 2) as u8,
                ((g0 as u16 + g1 as u16) / 2) as u8,
                ((b0 as u16 + b1 as u16) / 2) as u8,
                255,
            ],
            [0, 0, 0, 0],
        ]
    };

    (palette, lut)
}

fn decode_dxt1(width: u32, height: u32, payload: &[u8], pixels: &mut [u8]) -> Result<(), String> {
    let (blocks_x, blocks_y) = check_block_grid(width, height, 8, payload.len())?;
    let w = width as usize;
    let h = height as usize;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block_offset = (by * blocks_x + bx) * 8;
            let block = &payload[block_offset..block_offset + 8];
            let (palette, lut) = decode_bc1_color_block(block, false);

            for py in 0..4usize {
                let y = by * 4 + py;
                if y >= h {
                    continue;
                }
                for px in 0..4usize {
                    let x = bx * 4 + px;
                    if x >= w {
                        continue;
                    }
                    let i = py * 4 + px;
                    let idx = ((lut >> (2 * i)) & 0x3) as usize;
                    let dst = (y * w + x) * 4;
                    pixels[dst..dst + 4].copy_from_slice(&palette[idx]);
                }
            }
        }
    }

    Ok(())
}

fn decode_dxt3(width: u32, height: u32, payload: &[u8], pixels: &mut [u8]) -> Result<(), String> {
    let (blocks_x, blocks_y) = check_block_grid(width, height, 16, payload.len())?;
    let w = width as usize;
    let h = height as usize;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block_offset = (by * blocks_x + bx) * 16;
            let block = &payload[block_offset..block_offset + 16];
            let alpha_bits = u64::from_le_bytes([
                block[0], block[1], block[2], block[3], block[4], block[5], block[6], block[7],
            ]);
            let (palette, lut) = decode_bc1_color_block(&block[8..16], true);

            for py in 0..4usize {
                let y = by * 4 + py;
                if y >= h {
                    continue;
                }
                for px in 0..4usize {
                    let x = bx * 4 + px;
                    if x >= w {
                        continue;
                    }
                    let i = py * 4 + px;
                    let color_idx = ((lut >> (2 * i)) & 0x3) as usize;
                    let alpha = (((alpha_bits >> (4 * i)) & 0xF) as u8) * 17;
                    let mut rgba = palette[color_idx];
                    rgba[3] = alpha;
                    let dst = (y * w + x) * 4;
                    pixels[dst..dst + 4].copy_from_slice(&rgba);
                }
            }
        }
    }

    Ok(())
}

fn decode_dxt5(width: u32, height: u32, payload: &[u8], pixels: &mut [u8]) -> Result<(), String> {
    let (blocks_x, blocks_y) = check_block_grid(width, height, 16, payload.len())?;
    let w = width as usize;
    let h = height as usize;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block_offset = (by * blocks_x + bx) * 16;
            let block = &payload[block_offset..block_offset + 16];

            let a0 = block[0];
            let a1 = block[1];
            let mut alpha_lut = [0u8; 8];
            alpha_lut[0] = a0;
            alpha_lut[1] = a1;
            if a0 > a1 {
                for j in 1..=6u16 {
                    alpha_lut[(j + 1) as usize] =
                        (((7 - j) * a0 as u16 + j * a1 as u16 + 3) / 7) as u8;
                }
            } else {
                for j in 1..=4u16 {
                    alpha_lut[(j + 1) as usize] =
                        (((5 - j) * a0 as u16 + j * a1 as u16 + 2) / 5) as u8;
                }
                alpha_lut[6] = 0;
                alpha_lut[7] = 255;
            }

            let alpha_indices = (block[2] as u64)
                | ((block[3] as u64) << 8)
                | ((block[4] as u64) << 16)
                | ((block[5] as u64) << 24)
                | ((block[6] as u64) << 32)
                | ((block[7] as u64) << 40);

            let (palette, lut) = decode_bc1_color_block(&block[8..16], true);

            for py in 0..4usize {
                let y = by * 4 + py;
                if y >= h {
                    continue;
                }
                for px in 0..4usize {
                    let x = bx * 4 + px;
                    if x >= w {
                        continue;
                    }
                    let i = py * 4 + px;
                    let color_idx = ((lut >> (2 * i)) & 0x3) as usize;
                    let alpha_idx = ((alpha_indices >> (3 * i)) & 0x7) as usize;
                    let mut rgba = palette[color_idx];
                    rgba[3] = alpha_lut[alpha_idx];
                    let dst = (y * w + x) * 4;
                    pixels[dst..dst + 4].copy_from_slice(&rgba);
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dxt1_dds_bytes(width: u32, height: u32, payload: &[u8]) -> Vec<u8> {
        let mut dds = vec![0u8; 128];
        dds[0..4].copy_from_slice(b"DDS ");
        dds[4..8].copy_from_slice(&124u32.to_le_bytes());
        dds[8..12].copy_from_slice(&0x0000_1007u32.to_le_bytes());
        dds[12..16].copy_from_slice(&height.to_le_bytes());
        dds[16..20].copy_from_slice(&width.to_le_bytes());
        dds[76..80].copy_from_slice(&32u32.to_le_bytes());
        dds[80..84].copy_from_slice(&0x4u32.to_le_bytes()); // DDPF_FOURCC
        dds[84..88].copy_from_slice(b"DXT1");
        dds[108..112].copy_from_slice(&0x1000u32.to_le_bytes());
        dds.extend_from_slice(payload);
        dds
    }

    fn make_header(
        width: u32,
        height: u32,
        pitch_or_linear_size: u32,
        format: DdsFormat,
    ) -> DdsHeader {
        let dds = make_dxt1_dds_bytes(4, 4, &[0u8; 8]);
        let mut header = parse_dds(&dds).expect("valid minimal DXT1 header");
        header.width = width;
        header.height = height;
        header.pitch_or_linear_size = pitch_or_linear_size;
        header.format = format;
        header
    }

    #[test]
    fn test_dxt1_opaque_and_transparent() {
        let mut block_opaque = [0u8; 8];
        block_opaque[0..2].copy_from_slice(&0xF800u16.to_le_bytes());
        block_opaque[2..4].copy_from_slice(&0x001Fu16.to_le_bytes());
        block_opaque[4] = 0xE4;

        let dds_bytes = make_dxt1_dds_bytes(4, 1, &block_opaque);
        let img = decode_dds_rgba8(&dds_bytes).expect("decode DXT1 opaque");
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 1);
        assert_eq!(&img.pixels[0..4], &[255, 0, 0, 255]);
        assert_eq!(&img.pixels[4..8], &[0, 0, 255, 255]);
        assert_eq!(&img.pixels[8..12], &[170, 0, 85, 255]);
        assert_eq!(&img.pixels[12..16], &[85, 0, 170, 255]);

        let mut block_alpha = [0u8; 8];
        block_alpha[0..2].copy_from_slice(&0x001Fu16.to_le_bytes());
        block_alpha[2..4].copy_from_slice(&0xF800u16.to_le_bytes());
        block_alpha[4] = 0xE4;

        let hdr = make_header(4, 1, 0, DdsFormat::Dxt1);
        let img_alpha = decode_payload_rgba8(&hdr, &block_alpha).expect("decode DXT1 alpha");
        assert_eq!(&img_alpha.pixels[0..4], &[0, 0, 255, 255]);
        assert_eq!(&img_alpha.pixels[4..8], &[255, 0, 0, 255]);
        assert_eq!(&img_alpha.pixels[8..12], &[127, 0, 127, 255]);
        assert_eq!(&img_alpha.pixels[12..16], &[0, 0, 0, 0]);
    }

    #[test]
    fn test_dxt3_explicit_alpha_and_non_multiple_of_4() {
        let mut payload = [0u8; 32];
        let alpha0: u64 = (0x0u64) | (0xFu64 << 4) | (0x8u64 << 8) | (0x5u64 << (4 * 11));
        payload[0..8].copy_from_slice(&alpha0.to_le_bytes());
        payload[8..10].copy_from_slice(&0x001Fu16.to_le_bytes());
        payload[10..12].copy_from_slice(&0xF800u16.to_le_bytes());
        payload[12..16].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);

        let alpha1: u64 = (0xAu64) | (0x3u64 << (4 * 8));
        payload[16..24].copy_from_slice(&alpha1.to_le_bytes());
        payload[24..26].copy_from_slice(&0x07E0u16.to_le_bytes());

        let hdr = make_header(5, 3, 0, DdsFormat::Dxt3);
        let img = decode_payload_rgba8(&hdr, &payload).expect("decode DXT3 5x3");
        assert_eq!(img.pixels.len(), 5 * 3 * 4);

        let px = |x: usize, y: usize| -> &[u8] { &img.pixels[(y * 5 + x) * 4..(y * 5 + x + 1) * 4] };
        assert_eq!(px(0, 0), &[170, 0, 85, 0]);
        assert_eq!(px(1, 0), &[170, 0, 85, 255]);
        assert_eq!(px(2, 0), &[170, 0, 85, 136]);
        assert_eq!(px(3, 2), &[170, 0, 85, 85]);
        assert_eq!(px(4, 0), &[0, 255, 0, 170]);
        assert_eq!(px(4, 2), &[0, 255, 0, 51]);
    }

    #[test]
    fn test_dxt5_both_alpha_modes() {
        let pack_alpha_indices = |indices: &[u8; 16]| -> [u8; 6] {
            let mut bits = 0u64;
            for (i, &idx) in indices.iter().enumerate() {
                bits |= ((idx & 0x7) as u64) << (3 * i);
            }
            let le = bits.to_le_bytes();
            [le[0], le[1], le[2], le[3], le[4], le[5]]
        };

        let indices = [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0];

        let mut block_hi = [0u8; 16];
        block_hi[0] = 210;
        block_hi[1] = 0;
        block_hi[2..8].copy_from_slice(&pack_alpha_indices(&indices));
        block_hi[8..10].copy_from_slice(&0xFFFFu16.to_le_bytes());

        let hdr = make_header(4, 2, 0, DdsFormat::Dxt5);
        let img_hi = decode_payload_rgba8(&hdr, &block_hi).expect("decode DXT5 a0 > a1");
        let alphas_hi: Vec<u8> = img_hi.pixels.chunks_exact(4).map(|p| p[3]).collect();
        assert_eq!(alphas_hi, vec![210, 0, 180, 150, 120, 90, 60, 30]);

        let mut block_lo = [0u8; 16];
        block_lo[0] = 50;
        block_lo[1] = 200;
        block_lo[2..8].copy_from_slice(&pack_alpha_indices(&indices));
        block_lo[8..10].copy_from_slice(&0xFFFFu16.to_le_bytes());

        let img_lo = decode_payload_rgba8(&hdr, &block_lo).expect("decode DXT5 a0 <= a1");
        let alphas_lo: Vec<u8> = img_lo.pixels.chunks_exact(4).map(|p| p[3]).collect();
        assert_eq!(alphas_lo, vec![50, 200, 80, 110, 140, 170, 0, 255]);
    }

    #[test]
    fn test_uncompressed_bgra8_rgba8_bgr8_a8_l8() {
        let bgra_payload = [
            10, 20, 30, 40, 50, 60, 70, 80, 0xAA, 0xBB, 0xCC, 0xDD,
            1, 2, 3, 4, 5, 6, 7, 8, 0xEE, 0xFF, 0x11, 0x22,
        ];
        let hdr_bgra = make_header(2, 2, 12, DdsFormat::Bgra8);
        let img_bgra = decode_payload_rgba8(&hdr_bgra, &bgra_payload).expect("BGRA8");
        assert_eq!(
            img_bgra.pixels,
            vec![
                30, 20, 10, 40, 70, 60, 50, 80,
                3, 2, 1, 4, 7, 6, 5, 8,
            ]
        );

        let hdr_fallback = make_header(2, 2, 16, DdsFormat::Bgra8);
        let tight_bgra = [10u8, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, 5, 6, 7, 8];
        let img_fb = decode_payload_rgba8(&hdr_fallback, &tight_bgra).expect("BGRA8 fallback");
        assert_eq!(img_fb.pixels, img_bgra.pixels);

        let hdr_rgba = make_header(2, 1, 0, DdsFormat::Rgba8);
        let rgba_payload = [11u8, 22, 33, 44, 55, 66, 77, 88];
        let img_rgba = decode_payload_rgba8(&hdr_rgba, &rgba_payload).expect("RGBA8");
        assert_eq!(img_rgba.pixels, rgba_payload);

        let hdr_bgr = make_header(2, 1, 0, DdsFormat::Bgr8);
        let bgr_payload = [100u8, 150, 200, 10, 20, 30];
        let img_bgr = decode_payload_rgba8(&hdr_bgr, &bgr_payload).expect("BGR8");
        assert_eq!(img_bgr.pixels, vec![200, 150, 100, 255, 30, 20, 10, 255]);

        let hdr_a8 = make_header(2, 1, 0, DdsFormat::A8);
        let img_a8 = decode_payload_rgba8(&hdr_a8, &[0, 128]).expect("A8");
        assert_eq!(img_a8.pixels, vec![255, 255, 255, 0, 255, 255, 255, 128]);

        let hdr_l8 = make_header(2, 1, 0, DdsFormat::L8);
        let img_l8 = decode_payload_rgba8(&hdr_l8, &[42, 200]).expect("L8");
        assert_eq!(img_l8.pixels, vec![42, 42, 42, 255, 200, 200, 200, 255]);
    }

    #[test]
    fn test_truncated_payload_and_zero_dimensions_errors() {
        let hdr_zero = make_header(0, 4, 0, DdsFormat::Dxt1);
        assert!(decode_payload_rgba8(&hdr_zero, &[0u8; 8]).is_err());

        let hdr_dxt1 = make_header(4, 4, 0, DdsFormat::Dxt1);
        assert!(decode_payload_rgba8(&hdr_dxt1, &[0u8; 7]).is_err());

        let hdr_dxt5 = make_header(5, 5, 0, DdsFormat::Dxt5);
        assert!(decode_payload_rgba8(&hdr_dxt5, &[0u8; 63]).is_err());

        let hdr_bgra = make_header(2, 2, 0, DdsFormat::Bgra8);
        assert!(decode_payload_rgba8(&hdr_bgra, &[0u8; 15]).is_err());
    }

    #[test]
    fn test_resize_rgba_bilinear_and_cubemap_6_faces() {
        let small = RgbaImage {
            width: 2,
            height: 2,
            pixels: vec![0, 0, 0, 255, 200, 100, 50, 255, 0, 0, 0, 255, 200, 100, 50, 255],
        };
        let resized = resize_rgba_bilinear(&small, 4, 4);
        assert_eq!(resized.width, 4);
        assert_eq!(resized.height, 4);
        assert_eq!(resized.pixels.len(), 64);

        let mut dds = vec![0u8; 128];
        dds[0..4].copy_from_slice(b"DDS ");
        dds[4..8].copy_from_slice(&124u32.to_le_bytes());
        dds[8..12].copy_from_slice(&0x0000_1007u32.to_le_bytes());
        dds[12..16].copy_from_slice(&4u32.to_le_bytes());
        dds[16..20].copy_from_slice(&4u32.to_le_bytes());
        dds[76..80].copy_from_slice(&32u32.to_le_bytes());
        dds[80..84].copy_from_slice(&0x4u32.to_le_bytes());
        dds[84..88].copy_from_slice(b"DXT1");
        dds[108..112].copy_from_slice(&0x1008u32.to_le_bytes());
        dds[112..116].copy_from_slice(&0xFE00u32.to_le_bytes()); // cubemap all faces
        for f in 0..6u8 {
            let mut blk = [0u8; 8];
            blk[0] = f;
            dds.extend_from_slice(&blk);
        }
        let faces = decode_dds_cubemap_rgba8(&dds).expect("6-face DXT1 cubemap");
        assert_eq!(faces.len(), 6);
        for f in &faces {
            assert_eq!((f.width, f.height), (4, 4));
        }
    }
}
