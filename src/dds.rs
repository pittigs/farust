use std::fmt;

const DDS_MAGIC: &[u8; 4] = b"DDS ";
const DDS_HEADER_MIN_LEN: usize = 128;
const DDS_HEADER_SIZE: u32 = 124;
const DDS_PF_SIZE: u32 = 32;
const DDS_DX10_HEADER_LEN: usize = 148;

const DDPF_ALPHA: u32 = 0x2;
const DDPF_FOURCC: u32 = 0x4;
const DDPF_RGB: u32 = 0x40;
const DDPF_LUMINANCE: u32 = 0x20000;

const DDSCAPS2_CUBEMAP: u32 = 0x200;
const DDSCAPS2_VOLUME: u32 = 0x20_0000;

/// Pixel format decoded from a DDS header or DX10 extension header.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DdsFormat {
    Dxt1,
    Dxt3,
    Dxt5,
    Dx10(u32),
    Bgra8,
    Rgba8,
    Bgr8,
    A8,
    L8,
    FourCc([u8; 4]),
    Uncompressed {
        bits: u32,
        r_mask: u32,
        g_mask: u32,
        b_mask: u32,
        a_mask: u32,
    },
}

impl fmt::Display for DdsFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dxt1 => f.write_str("DXT1"),
            Self::Dxt3 => f.write_str("DXT3"),
            Self::Dxt5 => f.write_str("DXT5"),
            Self::Dx10(fmt_id) => write!(f, "DX10({fmt_id})"),
            Self::Bgra8 => f.write_str("BGRA8"),
            Self::Rgba8 => f.write_str("RGBA8"),
            Self::Bgr8 => f.write_str("BGR8"),
            Self::A8 => f.write_str("A8"),
            Self::L8 => f.write_str("L8"),
            Self::FourCc(cc) => f.write_str(&String::from_utf8_lossy(cc)),
            Self::Uncompressed { bits, .. } => write!(f, "RGB{bits}"),
        }
    }
}

/// Parsed metadata and payload location for a DDS image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DdsHeader {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_count: u32,
    pub pitch_or_linear_size: u32,
    pub format: DdsFormat,
    pub is_cubemap: bool,
    pub is_volume: bool,
    pub data_offset: usize,
    pub data_len: usize,
}

fn read_u32_le(data: &[u8], offset: usize) -> Result<u32, String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| format!("offset overflow reading u32 at {offset}"))?;
    let bytes: [u8; 4] = data
        .get(offset..end)
        .and_then(|slice| slice.try_into().ok())
        .ok_or_else(|| format!("unexpected end of DDS data at offset {offset}"))?;
    Ok(u32::from_le_bytes(bytes))
}

/// Parses a DDS file header (and optional DX10 extension header) from raw bytes.
pub fn parse_dds(data: &[u8]) -> Result<DdsHeader, String> {
    if data.len() < 4 {
        return Err(format!(
            "DDS data truncated: expected at least {DDS_HEADER_MIN_LEN} bytes, got {}",
            data.len()
        ));
    }
    if &data[0..4] != DDS_MAGIC {
        return Err("invalid DDS magic: expected \"DDS \"".to_string());
    }
    if data.len() < DDS_HEADER_MIN_LEN {
        return Err(format!(
            "DDS header truncated: expected at least {DDS_HEADER_MIN_LEN} bytes, got {}",
            data.len()
        ));
    }

    let header_size = read_u32_le(data, 4)?;
    if header_size != DDS_HEADER_SIZE {
        return Err(format!(
            "invalid DDS header size: expected {DDS_HEADER_SIZE}, got {header_size}"
        ));
    }

    let height = read_u32_le(data, 12)?;
    let width = read_u32_le(data, 16)?;
    let pitch_or_linear_size = read_u32_le(data, 20)?;
    let depth = read_u32_le(data, 24)?;
    let raw_mip_count = read_u32_le(data, 28)?;
    let mip_count = if raw_mip_count == 0 { 1 } else { raw_mip_count };

    let pf_size = read_u32_le(data, 76)?;
    if pf_size != DDS_PF_SIZE {
        return Err(format!(
            "invalid DDS pixel format size: expected {DDS_PF_SIZE}, got {pf_size}"
        ));
    }

    let pf_flags = read_u32_le(data, 80)?;
    let fourcc = [data[84], data[85], data[86], data[87]];
    let rgb_bits = read_u32_le(data, 88)?;
    let r_mask = read_u32_le(data, 92)?;
    let g_mask = read_u32_le(data, 96)?;
    let b_mask = read_u32_le(data, 100)?;
    let a_mask = read_u32_le(data, 104)?;

    let caps2 = read_u32_le(data, 112)?;
    let is_cubemap = (caps2 & DDSCAPS2_CUBEMAP) != 0;
    let is_volume = (caps2 & DDSCAPS2_VOLUME) != 0;

    let mut data_offset = DDS_HEADER_MIN_LEN;
    let format = if pf_flags & DDPF_FOURCC != 0 {
        match &fourcc {
            b"DXT1" => DdsFormat::Dxt1,
            b"DXT3" => DdsFormat::Dxt3,
            b"DXT5" => DdsFormat::Dxt5,
            b"DX10" => {
                if data.len() < DDS_DX10_HEADER_LEN {
                    return Err(format!(
                        "DX10 DDS header truncated: expected at least {DDS_DX10_HEADER_LEN} bytes, got {}",
                        data.len()
                    ));
                }
                let dxgi_format = read_u32_le(data, 128)?;
                data_offset = DDS_DX10_HEADER_LEN;
                DdsFormat::Dx10(dxgi_format)
            }
            _ => DdsFormat::FourCc(fourcc),
        }
    } else if pf_flags & DDPF_RGB != 0 {
        match (rgb_bits, r_mask, g_mask, b_mask, a_mask) {
            (32, 0x00ff_0000, 0x0000_ff00, 0x0000_00ff, 0xff00_0000) => DdsFormat::Bgra8,
            (32, 0x0000_00ff, 0x0000_ff00, 0x00ff_0000, 0xff00_0000) => DdsFormat::Rgba8,
            (24, 0x00ff_0000, 0x0000_ff00, 0x0000_00ff, 0) => DdsFormat::Bgr8,
            _ => DdsFormat::Uncompressed {
                bits: rgb_bits,
                r_mask,
                g_mask,
                b_mask,
                a_mask,
            },
        }
    } else if pf_flags & DDPF_ALPHA != 0 && rgb_bits == 8 {
        DdsFormat::A8
    } else if pf_flags & DDPF_LUMINANCE != 0 && rgb_bits == 8 {
        DdsFormat::L8
    } else {
        DdsFormat::Uncompressed {
            bits: rgb_bits,
            r_mask,
            g_mask,
            b_mask,
            a_mask,
        }
    };

    let data_len = data.len() - data_offset;

    Ok(DdsHeader {
        width,
        height,
        depth,
        mip_count,
        pitch_or_linear_size,
        format,
        is_cubemap,
        is_volume,
        data_offset,
        data_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_u32(buf: &mut [u8], offset: usize, value: u32) {
        buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn make_dds_base(width: u32, height: u32, mip_count: u32, payload_len: usize) -> Vec<u8> {
        let mut buf = vec![0u8; 128 + payload_len];
        buf[0..4].copy_from_slice(b"DDS ");
        write_u32(&mut buf, 4, 124);
        write_u32(&mut buf, 12, height);
        write_u32(&mut buf, 16, width);
        write_u32(&mut buf, 28, mip_count);
        write_u32(&mut buf, 76, 32);
        buf
    }

    #[test]
    fn parse_dxt1() {
        let mut buf = make_dds_base(128, 64, 4, 4096);
        write_u32(&mut buf, 20, 4096);
        write_u32(&mut buf, 80, DDPF_FOURCC);
        buf[84..88].copy_from_slice(b"DXT1");

        let header = parse_dds(&buf).expect("DXT1 should parse");
        assert_eq!(header.width, 128);
        assert_eq!(header.height, 64);
        assert_eq!(header.depth, 0);
        assert_eq!(header.mip_count, 4);
        assert_eq!(header.pitch_or_linear_size, 4096);
        assert_eq!(header.format, DdsFormat::Dxt1);
        assert_eq!(header.format.to_string(), "DXT1");
        assert!(!header.is_cubemap);
        assert!(!header.is_volume);
        assert_eq!(header.data_offset, 128);
        assert_eq!(header.data_len, 4096);
    }

    #[test]
    fn parse_dxt3_and_dxt5() {
        let mut dxt3 = make_dds_base(64, 64, 1, 512);
        write_u32(&mut dxt3, 80, DDPF_FOURCC);
        dxt3[84..88].copy_from_slice(b"DXT3");
        let h3 = parse_dds(&dxt3).expect("DXT3 should parse");
        assert_eq!(h3.format, DdsFormat::Dxt3);
        assert_eq!(h3.format.to_string(), "DXT3");

        let mut dxt5 = make_dds_base(512, 512, 10, 1024);
        write_u32(&mut dxt5, 20, 262_144);
        write_u32(&mut dxt5, 80, DDPF_FOURCC);
        dxt5[84..88].copy_from_slice(b"DXT5");
        let h5 = parse_dds(&dxt5).expect("DXT5 should parse");
        assert_eq!(h5.width, 512);
        assert_eq!(h5.height, 512);
        assert_eq!(h5.mip_count, 10);
        assert_eq!(h5.pitch_or_linear_size, 262_144);
        assert_eq!(h5.format, DdsFormat::Dxt5);
        assert_eq!(h5.format.to_string(), "DXT5");
        assert_eq!(h5.data_offset, 128);
        assert_eq!(h5.data_len, 1024);
    }

    #[test]
    fn parse_bgra8_256x256_mips_zero_to_one() {
        let payload_len = 256 * 256 * 4;
        let mut buf = make_dds_base(256, 256, 0, payload_len);
        write_u32(&mut buf, 20, 256 * 4);
        write_u32(&mut buf, 80, DDPF_RGB | 0x1);
        write_u32(&mut buf, 88, 32);
        write_u32(&mut buf, 92, 0x00ff_0000);
        write_u32(&mut buf, 96, 0x0000_ff00);
        write_u32(&mut buf, 100, 0x0000_00ff);
        write_u32(&mut buf, 104, 0xff00_0000);

        let header = parse_dds(&buf).expect("BGRA8 should parse");
        assert_eq!(header.width, 256);
        assert_eq!(header.height, 256);
        assert_eq!(header.mip_count, 1);
        assert_eq!(header.format, DdsFormat::Bgra8);
        assert_eq!(header.format.to_string(), "BGRA8");
        assert_eq!(header.data_offset, 128);
        assert_eq!(header.data_len, payload_len);
    }

    #[test]
    fn parse_rgba8_and_bgr8() {
        let mut rgba = make_dds_base(32, 32, 1, 32 * 32 * 4);
        write_u32(&mut rgba, 80, DDPF_RGB | 0x1);
        write_u32(&mut rgba, 88, 32);
        write_u32(&mut rgba, 92, 0x0000_00ff);
        write_u32(&mut rgba, 96, 0x0000_ff00);
        write_u32(&mut rgba, 100, 0x00ff_0000);
        write_u32(&mut rgba, 104, 0xff00_0000);
        let h_rgba = parse_dds(&rgba).expect("RGBA8 should parse");
        assert_eq!(h_rgba.format, DdsFormat::Rgba8);
        assert_eq!(h_rgba.format.to_string(), "RGBA8");

        let mut bgr = make_dds_base(64, 32, 1, 64 * 32 * 3);
        write_u32(&mut bgr, 20, 64 * 3);
        write_u32(&mut bgr, 80, DDPF_RGB);
        write_u32(&mut bgr, 88, 24);
        write_u32(&mut bgr, 92, 0x00ff_0000);
        write_u32(&mut bgr, 96, 0x0000_ff00);
        write_u32(&mut bgr, 100, 0x0000_00ff);
        write_u32(&mut bgr, 104, 0);
        let h_bgr = parse_dds(&bgr).expect("BGR8 should parse");
        assert_eq!(h_bgr.format, DdsFormat::Bgr8);
        assert_eq!(h_bgr.format.to_string(), "BGR8");
        assert_eq!(h_bgr.data_len, 64 * 32 * 3);
    }

    #[test]
    fn parse_a8_and_l8() {
        let mut a8 = make_dds_base(16, 16, 1, 256);
        write_u32(&mut a8, 80, DDPF_ALPHA);
        write_u32(&mut a8, 88, 8);
        write_u32(&mut a8, 104, 0xff);
        let h_a8 = parse_dds(&a8).expect("A8 should parse");
        assert_eq!(h_a8.format, DdsFormat::A8);
        assert_eq!(h_a8.format.to_string(), "A8");

        let mut l8 = make_dds_base(16, 16, 1, 256);
        write_u32(&mut l8, 80, DDPF_LUMINANCE);
        write_u32(&mut l8, 88, 8);
        write_u32(&mut l8, 92, 0xff);
        let h_l8 = parse_dds(&l8).expect("L8 should parse");
        assert_eq!(h_l8.format, DdsFormat::L8);
        assert_eq!(h_l8.format.to_string(), "L8");
    }

    #[test]
    fn parse_dx10_header() {
        let mut buf = make_dds_base(1024, 1024, 11, 20 + 512);
        write_u32(&mut buf, 80, DDPF_FOURCC);
        buf[84..88].copy_from_slice(b"DX10");
        write_u32(&mut buf, 128, 98); // DXGI_FORMAT_BC7_UNORM

        let header = parse_dds(&buf).expect("DX10 header should parse");
        assert_eq!(header.width, 1024);
        assert_eq!(header.height, 1024);
        assert_eq!(header.mip_count, 11);
        assert_eq!(header.format, DdsFormat::Dx10(98));
        assert_eq!(header.format.to_string(), "DX10(98)");
        assert_eq!(header.data_offset, 148);
        assert_eq!(header.data_len, 512);

        let exact_148 = &buf[..148];
        let header_empty_payload = parse_dds(exact_148).expect("148-byte DX10 header should parse");
        assert_eq!(header_empty_payload.data_offset, 148);
        assert_eq!(header_empty_payload.data_len, 0);
    }

    #[test]
    fn parse_cubemap_and_volume_flags() {
        let mut cubemap = make_dds_base(256, 256, 9, 0);
        write_u32(&mut cubemap, 80, DDPF_FOURCC);
        cubemap[84..88].copy_from_slice(b"DXT1");
        write_u32(&mut cubemap, 112, DDSCAPS2_CUBEMAP | 0xfe00);
        let h_cube = parse_dds(&cubemap).expect("cubemap should parse");
        assert!(h_cube.is_cubemap);
        assert!(!h_cube.is_volume);

        let mut volume = make_dds_base(64, 64, 1, 0);
        write_u32(&mut volume, 24, 32);
        write_u32(&mut volume, 80, DDPF_FOURCC);
        volume[84..88].copy_from_slice(b"DXT5");
        write_u32(&mut volume, 112, DDSCAPS2_VOLUME);
        let h_vol = parse_dds(&volume).expect("volume texture should parse");
        assert!(!h_vol.is_cubemap);
        assert!(h_vol.is_volume);
        assert_eq!(h_vol.depth, 32);
    }

    #[test]
    fn parse_custom_fourcc_and_uncompressed_fallback() {
        let mut ati2 = make_dds_base(128, 128, 1, 0);
        write_u32(&mut ati2, 80, DDPF_FOURCC);
        ati2[84..88].copy_from_slice(b"ATI2");
        let h_ati2 = parse_dds(&ati2).expect("ATI2 should parse");
        assert_eq!(h_ati2.format, DdsFormat::FourCc(*b"ATI2"));
        assert_eq!(h_ati2.format.to_string(), "ATI2");

        let lossy = DdsFormat::FourCc([0xff, b'B', b'C', b'5']);
        assert_eq!(lossy.to_string(), "\u{fffd}BC5");

        let mut rgb565 = make_dds_base(32, 32, 1, 2048);
        write_u32(&mut rgb565, 80, DDPF_RGB);
        write_u32(&mut rgb565, 88, 16);
        write_u32(&mut rgb565, 92, 0xf800);
        write_u32(&mut rgb565, 96, 0x07e0);
        write_u32(&mut rgb565, 100, 0x001f);
        let h_565 = parse_dds(&rgb565).expect("RGB565 should parse");
        assert_eq!(
            h_565.format,
            DdsFormat::Uncompressed {
                bits: 16,
                r_mask: 0xf800,
                g_mask: 0x07e0,
                b_mask: 0x001f,
                a_mask: 0,
            }
        );
        assert_eq!(h_565.format.to_string(), "RGB16");

        let mut l8a8 = make_dds_base(16, 16, 1, 512);
        write_u32(&mut l8a8, 80, DDPF_LUMINANCE | 0x1);
        write_u32(&mut l8a8, 88, 16);
        write_u32(&mut l8a8, 92, 0x00ff);
        write_u32(&mut l8a8, 104, 0xff00);
        let h_l8a8 = parse_dds(&l8a8).expect("16-bit L8A8 fallback should parse");
        assert_eq!(
            h_l8a8.format,
            DdsFormat::Uncompressed {
                bits: 16,
                r_mask: 0x00ff,
                g_mask: 0,
                b_mask: 0,
                a_mask: 0xff00,
            }
        );
        assert_eq!(h_l8a8.format.to_string(), "RGB16");
    }

    #[test]
    fn reject_truncated_and_invalid_headers() {
        assert!(parse_dds(&[]).is_err());
        assert!(parse_dds(b"DDS").is_err());
        assert!(parse_dds(b"BAD!").is_err());

        let truncated = make_dds_base(16, 16, 1, 0);
        assert!(parse_dds(&truncated[..127]).is_err());

        let mut bad_magic = make_dds_base(16, 16, 1, 0);
        bad_magic[0..4].copy_from_slice(b"PNG ");
        assert!(parse_dds(&bad_magic).is_err());

        let mut bad_header_size = make_dds_base(16, 16, 1, 0);
        write_u32(&mut bad_header_size, 4, 120);
        assert!(parse_dds(&bad_header_size).is_err());

        let mut bad_pf_size = make_dds_base(16, 16, 1, 0);
        write_u32(&mut bad_pf_size, 76, 28);
        assert!(parse_dds(&bad_pf_size).is_err());

        let mut truncated_dx10 = make_dds_base(16, 16, 1, 19);
        write_u32(&mut truncated_dx10, 80, DDPF_FOURCC);
        truncated_dx10[84..88].copy_from_slice(b"DX10");
        assert!(parse_dds(&truncated_dx10[..128]).is_err());
        assert!(parse_dds(&truncated_dx10).is_err());
    }
}
