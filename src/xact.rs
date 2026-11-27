//! Phase 14 clean-room Microsoft XACT `.xwb` (WaveBank) & `.xsb` (SoundBank) binary
//! parser, PCM / MS-ADPCM decoder, WAV exporter, cue library and a threaded
//! spatial stereo mixer built exclusively on Rust `std`.
//!
//! Output goes to a `pw-cat` / `pacat` / `aplay` subprocess as raw S16 stereo; when none
//! is available the engine silently stays mute. Audio never feeds back into the sim.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const WBND_MAGIC: &[u8; 4] = b"WBND";
const SDBK_MAGIC: &[u8; 4] = b"SDBK";
/// Mixer output rate (PipeWire native).
pub const OUTPUT_SAMPLE_RATE: u32 = 48_000;
/// 5 ms mix quantum for the realtime worker.
const MIX_CHUNK_FRAMES: usize = 240;
/// How far the realtime worker keeps ahead of the wall clock (plus sink latency ~20-30 ms).
const SINK_AHEAD_MS: u64 = 45;
/// Global cap on simultaneously playing one-shot voices (music/ambient streams excluded).
pub const MAX_VOICES: usize = 40;
/// Phase 24: looping unit sounds (construction / movement) allowed at once; they share
/// the one-shot pool above.
pub const MAX_LOOP_VOICES: usize = 8;
/// Max simultaneous music/ambient voices (music crossfade = 2 + dual map ambient = 2 + headroom).
const MAX_STREAM_VOICES: usize = 6;

/// XACT mini-format tags.
pub const FORMAT_PCM: u16 = 0;
pub const FORMAT_XMA: u16 = 1;
pub const FORMAT_ADPCM: u16 = 2;
pub const FORMAT_WMA: u16 = 3;

pub fn format_name(tag: u16) -> &'static str {
    match tag {
        FORMAT_PCM => "PCM",
        FORMAT_XMA => "XMA",
        FORMAT_ADPCM => "MS-ADPCM",
        FORMAT_WMA => "xWMA",
        _ => "unknown",
    }
}

/// Metadata for one audio wave stored inside an `.xwb` WaveBank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveEntryInfo {
    pub index: usize,
    pub name: Option<String>,
    pub format_tag: u16,
    pub channels: u16,
    pub sample_rate: u32,
    pub bits_per_sample: u16,
    pub block_align: u16,
    pub data_offset: usize,
    pub data_length: usize,
}

/// Parsed Microsoft XACT WaveBank (`.xwb`). `data` holds the whole file for
/// [`WaveBank::parse`], or is empty for a header-only [`WaveBank::open_lazy`] bank.
#[derive(Debug, Clone)]
pub struct WaveBank {
    pub name: String,
    pub version: u32,
    pub compact: bool,
    pub entries: Vec<WaveEntryInfo>,
    pub data: Vec<u8>,
}

/// Decoded audio: interleaved normalized `f32` frames.
#[derive(Debug, Clone, PartialEq)]
pub struct PcmClip {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl PcmClip {
    pub fn frames(&self) -> usize {
        self.samples.len() / usize::from(self.channels.max(1))
    }

    pub fn duration_secs(&self) -> f64 {
        self.frames() as f64 / f64::from(self.sample_rate.max(1))
    }
}

#[inline]
fn read_u16(data: &[u8], off: usize) -> Result<u16, String> {
    let b: [u8; 2] = data
        .get(off..off + 2)
        .ok_or_else(|| format!("XACT truncated u16 at {off}"))?
        .try_into()
        .map_err(|_| format!("XACT slice error at {off}"))?;
    Ok(u16::from_le_bytes(b))
}

#[inline]
fn read_u32(data: &[u8], off: usize) -> Result<u32, String> {
    let b: [u8; 4] = data
        .get(off..off + 4)
        .ok_or_else(|| format!("XACT truncated u32 at {off}"))?
        .try_into()
        .map_err(|_| format!("XACT slice error at {off}"))?;
    Ok(u32::from_le_bytes(b))
}

fn read_fixed_cstr(data: &[u8]) -> String {
    let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
    data[..end].iter().map(|&b| char::from(b)).collect()
}

#[inline]
fn decode_mini_format(fmt: u32) -> (u16, u16, u32, u16, u16) {
    let tag = (fmt & 0x3) as u16;
    let channels = ((fmt >> 2) & 0x7).max(1) as u16;
    let sample_rate = ((fmt >> 5) & 0x3_FFFF).max(8_000);
    let block_align = ((fmt >> 23) & 0xFF).max(1) as u16;
    let bits_per_sample = if ((fmt >> 31) & 1) == 1 { 16 } else { 8 };
    (tag, channels, sample_rate, block_align, bits_per_sample)
}

type BankLayout = (String, u32, bool, Vec<WaveEntryInfo>);

/// Parses the WaveBank header/entry tables from `bytes` (which must contain at least
/// segments 0..=3) for a file of `total_len` bytes.
fn parse_layout(bytes: &[u8], total_len: usize) -> Result<BankLayout, String> {
    if bytes.len() < 48 {
        return Err("XWB file too short".to_string());
    }
    if &bytes[0..4] != WBND_MAGIC {
        return Err(format!("invalid XWB magic: {:?}", &bytes[0..4]));
    }
    let version = read_u32(bytes, 4)?;
    let seg_off = if version >= 42 { 12 } else { 8 };
    if bytes.len() < seg_off + 5 * 8 {
        return Err("XWB header segments truncated".to_string());
    }
    let mut segs = [(0usize, 0usize); 5];
    for (i, slot) in segs.iter_mut().enumerate() {
        let off = read_u32(bytes, seg_off + i * 8)? as usize;
        let len = read_u32(bytes, seg_off + i * 8 + 4)? as usize;
        if len > 0 && off.saturating_add(len) > total_len {
            return Err(format!("XWB segment {i} ({off}+{len}) out of bounds"));
        }
        *slot = (off, len);
    }

    let (bd_off, bd_len) = segs[0];
    if bd_len < 88 || bd_off + 88 > bytes.len() {
        return Err("XWB BankData segment too small".to_string());
    }
    let flags = read_u32(bytes, bd_off)?;
    let entry_count = read_u32(bytes, bd_off + 4)? as usize;
    let name = read_fixed_cstr(&bytes[bd_off + 8..bd_off + 72]);
    let entry_meta_size = read_u32(bytes, bd_off + 72)? as usize;
    let entry_name_size = read_u32(bytes, bd_off + 76)? as usize;
    let alignment = read_u32(bytes, bd_off + 80)?.max(1) as usize;
    let compact_fmt = read_u32(bytes, bd_off + 84)?;
    let compact = (flags & 0x0002_0000) != 0;

    let (em_off, em_len) = segs[1];
    let (en_off, en_len) = segs[3];
    let (wd_off, wd_len) = segs[4];
    let names_ok = en_len > 0 && en_off + en_len <= bytes.len() && entry_name_size > 0;
    let entry_name = |i: usize| -> Option<String> {
        if names_ok && (i + 1) * entry_name_size <= en_len {
            Some(read_fixed_cstr(
                &bytes[en_off + i * entry_name_size..en_off + (i + 1) * entry_name_size],
            ))
        } else {
            None
        }
    };

    let mut entries = Vec::with_capacity(entry_count.min(4096));
    if compact {
        if entry_count * 4 > em_len {
            return Err("XWB compact entry table truncated".to_string());
        }
        let (tag, channels, sample_rate, block_align, bps) = decode_mini_format(compact_fmt);
        for i in 0..entry_count {
            let raw = read_u32(bytes, em_off + i * 4)?;
            let rel_off = ((raw & ((1 << 21) - 1)) as usize) * alignment;
            let next_off = if i + 1 < entry_count {
                let next_raw = read_u32(bytes, em_off + (i + 1) * 4)?;
                (((next_raw & ((1 << 21) - 1)) as usize) * alignment).min(wd_len)
            } else {
                wd_len
            };
            entries.push(WaveEntryInfo {
                index: i,
                name: entry_name(i),
                format_tag: tag,
                channels,
                sample_rate,
                bits_per_sample: bps,
                block_align,
                data_offset: wd_off + rel_off.min(wd_len),
                data_length: next_off.saturating_sub(rel_off.min(wd_len)),
            });
        }
    } else {
        let stride = entry_meta_size.max(16);
        if entry_count.saturating_mul(stride) > em_len {
            return Err("XWB entry metadata table truncated".to_string());
        }
        for i in 0..entry_count {
            let eo = em_off + i * stride;
            let fmt = read_u32(bytes, eo + 4)?;
            let play_off = read_u32(bytes, eo + 8)? as usize;
            let play_len = read_u32(bytes, eo + 12)? as usize;
            let (tag, channels, sample_rate, block_align, bps) = decode_mini_format(fmt);
            let abs_off = wd_off.saturating_add(play_off).min(total_len);
            entries.push(WaveEntryInfo {
                index: i,
                name: entry_name(i),
                format_tag: tag,
                channels,
                sample_rate,
                bits_per_sample: bps,
                block_align,
                data_offset: abs_off,
                data_length: play_len.min(total_len.saturating_sub(abs_off)),
            });
        }
    }
    Ok((name, version, compact, entries))
}

const ADPCM_COEF1: [i32; 7] = [256, 512, 0, 192, 240, 460, 392];
const ADPCM_COEF2: [i32; 7] = [0, -256, 0, 64, 0, -208, -232];
const ADPCM_ADAPT: [i32; 16] = [
    230, 230, 230, 230, 307, 409, 512, 614, 768, 614, 512, 409, 307, 230, 230, 230,
];

/// Decodes Microsoft ADPCM blocks (XACT stores `wBlockAlign - 22` per channel).
fn decode_ms_adpcm(raw: &[u8], channels: usize, block_bytes: usize) -> Result<Vec<f32>, String> {
    let header = 7 * channels;
    if channels == 0 || channels > 2 || block_bytes <= header {
        return Err(format!("bad ADPCM layout ch={channels} block={block_bytes}"));
    }
    let mut out = Vec::with_capacity(raw.len() * 2);
    for block in raw.chunks(block_bytes) {
        if block.len() < header {
            break;
        }
        let mut pred = [0usize; 2];
        let mut delta = [0i32; 2];
        let mut s1 = [0i32; 2];
        let mut s2 = [0i32; 2];
        for c in 0..channels {
            pred[c] = usize::from(block[c]).min(6);
            let rd = |k: usize| i32::from(i16::from_le_bytes([block[k], block[k + 1]]));
            delta[c] = rd(channels + 2 * c);
            s1[c] = rd(3 * channels + 2 * c);
            s2[c] = rd(5 * channels + 2 * c);
        }
        for c in 0..channels {
            out.push(s2[c] as f32 / 32768.0);
        }
        for c in 0..channels {
            out.push(s1[c] as f32 / 32768.0);
        }
        let mut c = 0usize;
        for &byte in &block[header..] {
            for nib in [byte >> 4, byte & 0x0F] {
                let signed = if nib >= 8 { i32::from(nib) - 16 } else { i32::from(nib) };
                let p = (s1[c] * ADPCM_COEF1[pred[c]] + s2[c] * ADPCM_COEF2[pred[c]]) >> 8;
                let s = (p + signed * delta[c]).clamp(-32768, 32767);
                s2[c] = s1[c];
                s1[c] = s;
                delta[c] = ((ADPCM_ADAPT[usize::from(nib)] * delta[c]) >> 8).max(16);
                out.push(s as f32 / 32768.0);
                c = (c + 1) % channels;
            }
        }
    }
    Ok(out)
}

/// Decodes one wave's raw bytes into an interleaved [`PcmClip`].
/// Supports PCM 8/16-bit and MS-ADPCM; XMA and xWMA are reported as unsupported.
pub fn decode_wave_data(e: &WaveEntryInfo, raw: &[u8]) -> Result<PcmClip, String> {
    let ch = usize::from(e.channels.max(1));
    let samples = match (e.format_tag, e.bits_per_sample) {
        (FORMAT_PCM, 16) => {
            let n = raw.len() / 2 / ch * ch;
            (0..n)
                .map(|i| f32::from(i16::from_le_bytes([raw[2 * i], raw[2 * i + 1]])) / 32768.0)
                .collect()
        }
        (FORMAT_PCM, 8) => {
            let n = raw.len() / ch * ch;
            raw[..n].iter().map(|&b| (f32::from(b) - 128.0) / 128.0).collect()
        }
        (FORMAT_ADPCM, _) => {
            decode_ms_adpcm(raw, ch, (usize::from(e.block_align) + 22) * ch)?
        }
        (tag, bits) => {
            return Err(format!("unsupported wave format {} ({bits}-bit)", format_name(tag)));
        }
    };
    Ok(PcmClip {
        sample_rate: e.sample_rate,
        channels: ch as u16,
        samples,
    })
}

/// Builds a 16-bit PCM RIFF/WAVE file from interleaved samples.
pub fn wav_bytes_i16(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let block = channels.max(1) * 2;
    let mut wav = Vec::with_capacity(44 + samples.len() * 2);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&36u32.saturating_add(data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * u32::from(block)).to_le_bytes());
    wav.extend_from_slice(&block.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        wav.extend_from_slice(&s.to_le_bytes());
    }
    wav
}

#[inline]
fn f32_to_i16(s: f32) -> i16 {
    (s.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

impl WaveBank {
    /// Parses a binary XACT `.xwb` (`WBND`) WaveBank held fully in memory.
    pub fn parse(bytes: Vec<u8>) -> Result<Self, String> {
        let (name, version, compact, entries) = parse_layout(&bytes, bytes.len())?;
        Ok(Self {
            name,
            version,
            compact,
            entries,
            data: bytes,
        })
    }

    /// Reads only the header/entry tables of a WaveBank file (wave data stays on disk;
    /// fetch it with [`WaveBank::read_entry`]). Keeps 250 MB music banks out of RAM.
    pub fn open_lazy(path: &Path) -> Result<Self, String> {
        let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let total_len = f
            .metadata()
            .map_err(|e| format!("stat {}: {e}", path.display()))?
            .len() as usize;
        let mut head = vec![0u8; 64.min(total_len)];
        f.read_exact(&mut head).map_err(|e| format!("read {}: {e}", path.display()))?;
        if head.len() < 52 || &head[0..4] != WBND_MAGIC {
            return Err(format!("{}: not an XWB wavebank", path.display()));
        }
        let version = read_u32(&head, 4)?;
        let seg_off = if version >= 42 { 12 } else { 8 };
        let mut header_end = 0usize;
        for i in 0..4 {
            let off = read_u32(&head, seg_off + i * 8)? as usize;
            let len = read_u32(&head, seg_off + i * 8 + 4)? as usize;
            if len > 0 {
                header_end = header_end.max(off.saturating_add(len));
            }
        }
        if header_end > total_len || header_end > 64 << 20 {
            return Err(format!("{}: XWB header tables out of bounds", path.display()));
        }
        let mut bytes = vec![0u8; header_end.max(head.len())];
        f.seek(SeekFrom::Start(0)).map_err(|e| format!("seek: {e}"))?;
        f.read_exact(&mut bytes).map_err(|e| format!("read {}: {e}", path.display()))?;
        let (name, version, compact, entries) = parse_layout(&bytes, total_len)?;
        Ok(Self {
            name,
            version,
            compact,
            entries,
            data: Vec::new(),
        })
    }

    /// Returns wave `entry_idx`'s raw bytes from memory, or from `path` for lazy banks.
    pub fn read_entry(&self, entry_idx: usize, path: Option<&Path>) -> Result<Vec<u8>, String> {
        let e = self
            .entries
            .get(entry_idx)
            .ok_or_else(|| format!("wave #{entry_idx} out of range"))?;
        if let Some(raw) = self.data.get(e.data_offset..e.data_offset + e.data_length)
            && !self.data.is_empty()
        {
            return Ok(raw.to_vec());
        }
        let path = path.ok_or_else(|| format!("wave #{entry_idx} not resident"))?;
        let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        f.seek(SeekFrom::Start(e.data_offset as u64))
            .map_err(|err| format!("seek: {err}"))?;
        let mut raw = vec![0u8; e.data_length];
        f.read_exact(&mut raw).map_err(|err| format!("read wave #{entry_idx}: {err}"))?;
        Ok(raw)
    }

    /// Decodes wave `entry_idx` of an in-memory bank.
    pub fn decode(&self, entry_idx: usize) -> Result<PcmClip, String> {
        let raw = self.read_entry(entry_idx, None)?;
        decode_wave_data(&self.entries[entry_idx], &raw)
    }

    /// Decodes wave `entry_idx` into mono normalized `f32` samples `(sample_rate, samples)`.
    pub fn decode_mono_f32(&self, entry_idx: usize) -> Option<(u32, Vec<f32>)> {
        let clip = self.decode(entry_idx).ok()?;
        let ch = usize::from(clip.channels.max(1));
        let mono = clip
            .samples
            .chunks_exact(ch)
            .map(|f| f.iter().sum::<f32>() / ch as f32)
            .collect();
        Some((clip.sample_rate, mono))
    }

    /// Exports wave `entry_idx` as a standard 16-bit PCM RIFF/WAVE file (ADPCM is decoded).
    pub fn to_wav_bytes(&self, entry_idx: usize) -> Option<Vec<u8>> {
        let e = self.entries.get(entry_idx)?;
        if e.format_tag == FORMAT_PCM && e.bits_per_sample == 16 {
            let raw = self.read_entry(entry_idx, None).ok()?;
            let pcm: Vec<i16> = raw
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]))
                .collect();
            return Some(wav_bytes_i16(e.sample_rate, e.channels, &pcm));
        }
        let clip = self.decode(entry_idx).ok()?;
        let pcm: Vec<i16> = clip.samples.iter().map(|&s| f32_to_i16(s)).collect();
        Some(wav_bytes_i16(clip.sample_rate, clip.channels, &pcm))
    }
}

/// One resolved cue in an XACT SoundBank linking cue name to `(wavebank_name, wave_idx)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoundCue {
    pub name: String,
    pub waves: Vec<(String, u16)>,
    /// Phase 24: XACT tracks (layers played together), each a list of wave variations.
    pub tracks: Vec<Vec<(String, u16)>>,
}

/// Parsed Microsoft XACT SoundBank (`.xsb`).
#[derive(Debug, Clone)]
pub struct SoundBank {
    pub name: String,
    pub wavebank_names: Vec<String>,
    pub cues: Vec<SoundCue>,
}

#[cfg(test)]
fn extract_sound_waves(data: &[u8], sound_off: usize, nwb: usize) -> Vec<(u8, u16)> {
    extract_sound_tracks(data, sound_off, nwb).into_iter().flatten().collect()
}

/// Waves of one sound grouped per play event: every `PlayWave` / `PlayWaveVariation`
/// event is one XACT track (layer, all play together); a variation event lists the
/// alternatives of that layer. The event type is the low 5 bits of the event header byte
/// (high bits carry timestamp / loop flags; looping `*_Loop` cues set them).
fn extract_sound_tracks(data: &[u8], sound_off: usize, nwb: usize) -> Vec<Vec<(u8, u16)>> {
    if sound_off + 9 > data.len() {
        return Vec::new();
    }
    let sflags = data[sound_off];
    let slen = match read_u16(data, sound_off + 7) {
        Ok(l) => l as usize,
        Err(_) => return Vec::new(),
    };
    if slen < 9 || sound_off + slen > data.len() {
        return Vec::new();
    }
    let chunk = &data[sound_off..sound_off + slen];
    if (sflags & 1) == 0 {
        if chunk.len() >= 12 {
            if let Ok(wi) = read_u16(chunk, 9) {
                let wbi = chunk[11];
                if usize::from(wbi) < nwb {
                    return vec![vec![(wbi, wi)]];
                }
            }
        }
        return Vec::new();
    }

    // Complex sound: locate PlayWave (type 1 / 4) and PlayWaveVariation (type 3 / 6) XACT v43 event blocks.
    const MARKER: &[u8; 5] = b"\x20\x00\x00\xff\x0c";
    let mut tracks: Vec<Vec<(u8, u16)>> = Vec::new();
    let mut i = 0usize;
    while i + MARKER.len() <= chunk.len() {
        if &chunk[i..i + 5] == MARKER {
            let etype = if i >= 3 { chunk[i - 3] & 0x1f } else { 0 };
            if etype == 1 || etype == 4 {
                if i + 8 <= chunk.len()
                    && let Ok(wi) = read_u16(chunk, i + 5)
                {
                    let wbi = chunk[i + 7];
                    if usize::from(wbi) < nwb {
                        tracks.push(vec![(wbi, wi)]);
                    }
                }
            } else if etype == 3 || etype == 6 {
                let mut waves = Vec::new();
                // PlayWave with variation table: type 3 (plain) and type 6 (pitch/volume
                // variation) differ only in the 7-byte parameter block before the count.
                let (cnt_off, list_off) = if etype == 6 { (17, 25) } else { (10, 18) };
                if i + list_off <= chunk.len()
                    && let Ok(cnt) = read_u16(chunk, i + cnt_off)
                {
                    for k in 0..usize::from(cnt).min(32) {
                        let off = i + list_off + k * 5;
                        if off + 3 <= chunk.len()
                            && let Ok(wi) = read_u16(chunk, off)
                        {
                            let wbi = chunk[off + 2];
                            if usize::from(wbi) < nwb {
                                waves.push((wbi, wi));
                            }
                        }
                    }
                }
                if !waves.is_empty() {
                    tracks.push(waves);
                }
            }
            i += 5;
        } else {
            i += 1;
        }
    }
    tracks
}

impl SoundBank {
    /// Parses a binary XACT `.xsb` (`SDBK`) SoundBank.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 138 {
            return Err("XSB header too short".to_string());
        }
        if &bytes[0..4] != SDBK_MAGIC {
            return Err(format!("invalid XSB magic: {:?}", &bytes[0..4]));
        }
        let num_simple = read_u16(bytes, 19)? as usize;
        let num_complex = read_u16(bytes, 21)? as usize;
        let num_total = read_u16(bytes, 25)? as usize;
        let num_wb = bytes[27] as usize;
        let cue_name_table_len = read_u16(bytes, 30)? as usize;

        let mut offs = [0u32; 10];
        for (i, slot) in offs.iter_mut().enumerate() {
            *slot = read_u32(bytes, 34 + i * 4)?;
        }
        let name = read_fixed_cstr(&bytes[74..138]);

        let wb_off = offs[6] as usize;
        let mut wavebank_names = Vec::with_capacity(num_wb);
        for i in 0..num_wb {
            let start = wb_off + i * 64;
            if start + 64 <= bytes.len() {
                wavebank_names.push(read_fixed_cstr(&bytes[start..start + 64]));
            }
        }

        // Read cue names sequentially from `offs[2]` (matches Cue order in XSB).
        let cn_off = offs[2] as usize;
        let mut cue_names = Vec::with_capacity(num_total);
        if cn_off < bytes.len() && cue_name_table_len > 0 {
            let cn_end = (cn_off + cue_name_table_len).min(bytes.len());
            let mut pos = cn_off;
            while pos < cn_end && cue_names.len() < num_total {
                let rel_end = bytes[pos..cn_end]
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(cn_end - pos);
                if rel_end > 0 {
                    let s: String = bytes[pos..pos + rel_end]
                        .iter()
                        .map(|&b| char::from(b))
                        .collect();
                    cue_names.push(s);
                }
                pos += rel_end + 1;
            }
        }

        let mut cues = Vec::with_capacity(num_simple + num_complex);
        let mut idx = 0usize;

        let sc_off = offs[0] as usize;
        if offs[0] != u32::MAX && sc_off < bytes.len() {
            for i in 0..num_simple {
                if sc_off + (i + 1) * 5 > bytes.len() {
                    break;
                }
                let sound_off = read_u32(bytes, sc_off + i * 5 + 1)? as usize;
                let tracks: Vec<Vec<(String, u16)>> = extract_sound_tracks(bytes, sound_off, wavebank_names.len())
                    .into_iter()
                    .map(|t| {
                        t.into_iter()
                            .filter_map(|(wbi, wi)| wavebank_names.get(usize::from(wbi)).map(|wb| (wb.clone(), wi)))
                            .collect::<Vec<_>>()
                    })
                    .filter(|t| !t.is_empty())
                    .collect();
                let resolved = tracks.iter().flatten().cloned().collect();
                let cname = cue_names
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("cue_{idx}"));
                cues.push(SoundCue {
                    name: cname,
                    waves: resolved,
                    tracks,
                });
                idx += 1;
            }
        }

        let cc_off = offs[1] as usize;
        if offs[1] != u32::MAX && cc_off < bytes.len() {
            for i in 0..num_complex {
                if cc_off + (i + 1) * 15 > bytes.len() {
                    break;
                }
                let sound_off = read_u32(bytes, cc_off + i * 15 + 1)? as usize;
                let tracks: Vec<Vec<(String, u16)>> = extract_sound_tracks(bytes, sound_off, wavebank_names.len())
                    .into_iter()
                    .map(|t| {
                        t.into_iter()
                            .filter_map(|(wbi, wi)| wavebank_names.get(usize::from(wbi)).map(|wb| (wb.clone(), wi)))
                            .collect::<Vec<_>>()
                    })
                    .filter(|t| !t.is_empty())
                    .collect();
                let resolved = tracks.iter().flatten().cloned().collect();
                let cname = cue_names
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("cue_{idx}"));
                cues.push(SoundCue {
                    name: cname,
                    waves: resolved,
                    tracks,
                });
                idx += 1;
            }
        }

        Ok(Self {
            name,
            wavebank_names,
            cues,
        })
    }
}


// ---------------------------------------------------------------------------
// Cue library: every `.xsb` cue name -> waves in lazily opened `.xwb` banks
// ---------------------------------------------------------------------------

/// One named cue from a SoundBank, resolved to `(wavebank_name, wave_index)` refs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CueDef {
    pub name: String,
    pub soundbank: String,
    pub waves: Vec<(String, u16)>,
    /// Phase 24: layers (XACT tracks) of variations; `waves` is their concatenation.
    pub tracks: Vec<Vec<(String, u16)>>,
}

#[cfg(test)]
const MEM_BANK: &str = "<memory>";

/// Index of all cues in a `sounds/` directory with on-demand wave decoding & caching.
pub struct CueLibrary {
    cues: BTreeMap<String, CueDef>,
    bank_files: HashMap<String, PathBuf>,
    banks: HashMap<String, Option<Arc<WaveBank>>>,
    clips: HashMap<(String, u16), Option<Arc<PcmClip>>>,
    pub decode_failures: usize,
}

/// Parses a direct wave-index reference `"<WaveBank>#<Index>"` (e.g. `"XGG#0"`), used
/// when a wave exists in an `.xwb` bank without an `.xsb` cue wrapper.
pub fn parse_wave_ref(spec: &str) -> Option<(&str, u16)> {
    let (bank, idx_s) = spec.split_once('#')?;
    let bank = bank.trim();
    if bank.is_empty() {
        return None;
    }
    let idx = idx_s.trim().parse::<u16>().ok()?;
    Some((bank, idx))
}

/// Collects files inside `sounds_dir`: top-level entries first (sorted), then
/// subdirectories recursively (sorted, e.g. `Voice/US/*.xsb`), so root banks keep priority
/// while localized voice banks are indexed automatically.
pub fn collect_sound_paths(sounds_dir: &Path) -> Vec<PathBuf> {
    fn walk_subdirs(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = fs::read_dir(dir) else {
            return;
        };
        let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        entries.sort();
        let (mut dirs, mut files): (Vec<PathBuf>, Vec<PathBuf>) =
            entries.into_iter().partition(|p| p.is_dir());
        files.sort();
        dirs.sort();
        out.extend(files);
        for d in dirs {
            walk_subdirs(&d, out);
        }
    }
    let mut out = Vec::new();
    walk_subdirs(sounds_dir, &mut out);
    out
}

impl CueLibrary {
    pub fn empty() -> Self {
        Self {
            cues: BTreeMap::new(),
            bank_files: HashMap::new(),
            banks: HashMap::new(),
            clips: HashMap::new(),
            decode_failures: 0,
        }
    }

    /// Parses every `.xsb` in `sounds_dir` recursively (incl. `Voice/<lang>/`) and records
    /// `.xwb` paths. Top-level banks win when two `.xsb` files define the same cue name.
    pub fn scan(sounds_dir: &Path) -> Self {
        let mut lib = Self::empty();
        for p in collect_sound_paths(sounds_dir) {
            let ext = p
                .extension()
                .and_then(|s| s.to_str())
                .map(str::to_ascii_lowercase)
                .unwrap_or_default();
            let stem = p
                .file_stem()
                .and_then(|s| s.to_str())
                .map(str::to_ascii_lowercase)
                .unwrap_or_default();
            if ext == "xwb" {
                lib.bank_files.entry(stem).or_insert(p);
            } else if ext == "xsb"
                && let Ok(bytes) = fs::read(&p)
                && let Ok(sb) = SoundBank::parse(&bytes)
            {
                for cue in sb.cues {
                    let key = cue.name.to_ascii_lowercase();
                    let replace = lib.cues.get(&key).is_none_or(|c| c.waves.is_empty());
                    if replace {
                        lib.cues.insert(
                            key,
                            CueDef {
                                name: cue.name,
                                soundbank: sb.name.clone(),
                                waves: cue.waves,
                                tracks: cue.tracks,
                            },
                        );
                    }
                }
            }
        }
        lib
    }

    pub fn len(&self) -> usize {
        self.cues.len()
    }

    /// Every indexed cue definition in case-insensitive alphabetical order.
    pub fn cues(&self) -> impl Iterator<Item = &CueDef> {
        self.cues.values()
    }

    /// Case-insensitive cue lookup.
    pub fn cue(&self, name: &str) -> Option<&CueDef> {
        self.cues.get(&name.to_ascii_lowercase())
    }

    /// Metadata of a direct `Bank#Index` wave if the bank and entry exist.
    pub fn wave_entry(&mut self, wb_name: &str, idx: u16) -> Option<(String, WaveEntryInfo)> {
        let (bank, _) = self.bank(wb_name)?;
        let entry = bank.entries.get(usize::from(idx))?.clone();
        Some((bank.name.clone(), entry))
    }

    /// True if `name` exists and references at least one wave.
    #[cfg(test)]
    pub fn has(&self, name: &str) -> bool {
        self.cue(name).is_some_and(|c| !c.waves.is_empty())
    }

    /// First (alphabetical) playable cue starting with `prefix` whose SoundBank name
    /// contains `bank_filter` and whose name ends with none of `exclude`.
    pub fn find_prefixed(&self, prefix: &str, bank_filter: &str, exclude: &[&str]) -> Option<String> {
        let p = prefix.to_ascii_lowercase();
        let bf = bank_filter.to_ascii_lowercase();
        self.cues
            .range(p.clone()..)
            .take_while(|(k, _)| k.starts_with(&p))
            .find(|(k, c)| {
                !c.waves.is_empty()
                    && c.soundbank.to_ascii_lowercase().contains(&bf)
                    && !exclude.iter().any(|x| k.ends_with(&x.to_ascii_lowercase()))
            })
            .map(|(_, c)| c.name.clone())
    }

    fn bank(&mut self, wb_name: &str) -> Option<(Arc<WaveBank>, PathBuf)> {
        let key = wb_name.to_ascii_lowercase();
        let path = self.bank_files.get(&key)?.clone();
        let bank = self
            .banks
            .entry(key)
            .or_insert_with(|| WaveBank::open_lazy(&path).ok().map(Arc::new))
            .clone()?;
        Some((bank, path))
    }

    /// Decodes (and caches) every distinct wave of cue `name` (or direct `"Bank#Index"`).
    pub fn clips(&mut self, name: &str) -> Vec<Arc<PcmClip>> {
        if let Some(def) = self.cue(name).cloned() {
            return self.decode_waves(&def.waves);
        }
        if let Some((wb, idx)) = parse_wave_ref(name) {
            return self.decode_waves(&[(wb.to_string(), idx)]);
        }
        Vec::new()
    }

    /// Phase 24: decoded clips per layer of cue `name` (layers without any decodable wave
    /// are dropped). A plain cue or `"Bank#Index"` wave ref yields one layer.
    pub fn layer_clips(&mut self, name: &str) -> Vec<Vec<Arc<PcmClip>>> {
        if let Some(def) = self.cue(name).cloned() {
            let tracks = if def.tracks.is_empty() { vec![def.waves.clone()] } else { def.tracks.clone() };
            return tracks
                .iter()
                .map(|t| self.decode_waves(t))
                .filter(|c| !c.is_empty())
                .collect();
        }
        if let Some((wb, idx)) = parse_wave_ref(name) {
            let decoded = self.decode_waves(&[(wb.to_string(), idx)]);
            if !decoded.is_empty() {
                return vec![decoded];
            }
        }
        Vec::new()
    }

    fn decode_waves(&mut self, waves: &[(String, u16)]) -> Vec<Arc<PcmClip>> {
        let mut out: Vec<Arc<PcmClip>> = Vec::new();
        let mut seen: Vec<(String, u16)> = Vec::new();
        for (wb, idx) in waves.iter().cloned() {
            let key = (wb.to_ascii_lowercase(), idx);
            if seen.contains(&key) {
                continue;
            }
            seen.push(key.clone());
            if !self.clips.contains_key(&key) {
                let decoded = self.bank(&wb).and_then(|(bank, path)| {
                    let raw = bank.read_entry(usize::from(idx), Some(&path)).ok()?;
                    decode_wave_data(bank.entries.get(usize::from(idx))?, &raw).ok()
                });
                if decoded.is_none() {
                    self.decode_failures += 1;
                }
                self.clips.insert(key.clone(), decoded.map(Arc::new));
            }
            if let Some(Some(c)) = self.clips.get(&key) {
                out.push(Arc::clone(c));
            }
        }
        out
    }

    /// Opens wave `variant % n` of cue `name` (or `"Bank#Index"`) as a disk stream (16-bit PCM only).
    pub fn stream(&mut self, name: &str, variant: usize) -> Option<PcmStream> {
        let (wb, idx) = if let Some(def) = self.cue(name).cloned() {
            if def.waves.is_empty() {
                return None;
            }
            def.waves[variant % def.waves.len()].clone()
        } else if let Some((wb, idx)) = parse_wave_ref(name) {
            (wb.to_string(), idx)
        } else {
            return None;
        };
        let (bank, path) = self.bank(&wb)?;
        let e = bank.entries.get(usize::from(idx))?;
        if e.format_tag != FORMAT_PCM || e.bits_per_sample != 16 {
            return None;
        }
        PcmStream::open(&path, e)
    }

    /// Registers in-memory clips under a cue name (tests).
    #[cfg(test)]
    pub fn insert_clips(&mut self, name: &str, clips: Vec<PcmClip>) {
        let mut waves = Vec::new();
        for (i, c) in clips.into_iter().enumerate() {
            let idx = (self.clips.len() + i) as u16;
            self.clips.insert((MEM_BANK.to_string(), idx), Some(Arc::new(c)));
            waves.push((MEM_BANK.to_string(), idx));
        }
        self.cues.insert(
            name.to_ascii_lowercase(),
            CueDef {
                name: name.to_string(),
                soundbank: MEM_BANK.to_string(),
                tracks: vec![waves.clone()],
                waves,
            },
        );
    }

    /// Registers in-memory clips as separate layers of one cue (tests).
    #[cfg(test)]
    pub fn insert_layers(&mut self, name: &str, layers: Vec<Vec<PcmClip>>) {
        let mut tracks = Vec::new();
        for layer in layers {
            let mut t = Vec::new();
            for c in layer {
                let idx = self.clips.len() as u16;
                self.clips.insert((MEM_BANK.to_string(), idx), Some(Arc::new(c)));
                t.push((MEM_BANK.to_string(), idx));
            }
            tracks.push(t);
        }
        self.cues.insert(
            name.to_ascii_lowercase(),
            CueDef {
                name: name.to_string(),
                soundbank: MEM_BANK.to_string(),
                waves: tracks.iter().flatten().cloned().collect(),
                tracks,
            },
        );
    }
}

const STREAM_CHUNK_FRAMES: usize = 16_384;

/// Sequential 16-bit PCM reader over one wave inside a (large) `.xwb` file.
pub struct PcmStream {
    file: File,
    data_offset: u64,
    total_frames: usize,
    channels: u16,
    sample_rate: u32,
    buf: Vec<f32>,
    buf_start: usize,
    buf_frames: usize,
}

impl PcmStream {
    pub fn open(path: &Path, e: &WaveEntryInfo) -> Option<Self> {
        let file = File::open(path).ok()?;
        let ch = e.channels.max(1);
        Some(Self {
            file,
            data_offset: e.data_offset as u64,
            total_frames: e.data_length / (2 * usize::from(ch)),
            channels: ch,
            sample_rate: e.sample_rate,
            buf: Vec::new(),
            buf_start: 0,
            buf_frames: 0,
        })
    }

    fn refill(&mut self, idx: usize) -> bool {
        let ch = usize::from(self.channels);
        let frames = STREAM_CHUNK_FRAMES.min(self.total_frames.saturating_sub(idx));
        let mut raw = vec![0u8; frames * ch * 2];
        let ok = self
            .file
            .seek(SeekFrom::Start(self.data_offset + (idx * ch * 2) as u64))
            .is_ok()
            && self.file.read_exact(&mut raw).is_ok();
        if !ok || frames == 0 {
            self.total_frames = idx;
            return false;
        }
        self.buf.clear();
        self.buf.extend(
            raw.chunks_exact(2)
                .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0),
        );
        self.buf_start = idx;
        self.buf_frames = frames;
        true
    }

    fn frame(&mut self, idx: usize) -> Option<(f32, f32)> {
        if idx >= self.total_frames {
            return None;
        }
        if (idx < self.buf_start || idx >= self.buf_start + self.buf_frames) && !self.refill(idx) {
            return None;
        }
        let ch = usize::from(self.channels);
        let base = (idx - self.buf_start) * ch;
        let l = self.buf[base];
        Some((l, if ch > 1 { self.buf[base + 1] } else { l }))
    }
}

// ---------------------------------------------------------------------------
// Mixer
// ---------------------------------------------------------------------------

/// Mixer bus; each has its own gain. Music/Ambient voices bypass the one-shot voice cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bus {
    Sfx,
    Ui,
    Voice,
    Music,
    Ambient,
}

impl Bus {
    fn idx(self) -> usize {
        self as usize
    }

    fn is_stream(self) -> bool {
        matches!(self, Bus::Music | Bus::Ambient)
    }
}

/// Playback policy for a registered cue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CueSpec {
    pub bus: Bus,
    pub volume: f32,
    /// Minimum seconds between two starts of this cue.
    pub cooldown_secs: f32,
    /// Max concurrently playing instances of this cue.
    pub max_instances: usize,
    /// Weight used when stealing voices (higher survives).
    pub priority: f32,
    /// New instance always replaces the oldest one (voice acknowledgements).
    pub replace_oldest: bool,
}

impl CueSpec {
    pub const fn sfx(volume: f32, cooldown_secs: f32, max_instances: usize, priority: f32) -> Self {
        Self {
            bus: Bus::Sfx,
            volume,
            cooldown_secs,
            max_instances,
            priority,
            replace_oldest: false,
        }
    }

    pub const fn ui(volume: f32, cooldown_secs: f32) -> Self {
        Self {
            bus: Bus::Ui,
            volume,
            cooldown_secs,
            max_instances: 2,
            priority: 4.0,
            replace_oldest: true,
        }
    }

    pub const fn voice(volume: f32, cooldown_secs: f32) -> Self {
        Self {
            bus: Bus::Voice,
            volume,
            cooldown_secs,
            max_instances: 1,
            priority: 8.0,
            replace_oldest: true,
        }
    }
}

pub type CueId = usize;

/// Camera used for spatialization: column-major `view_proj` (`m[col][row]`) and eye.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Listener {
    pub view_proj: [[f32; 4]; 4],
    pub eye: [f32; 3],
}

/// Distance (world units from the camera eye) inside which sounds play at full level.
const REF_DISTANCE: f32 = 60.0;
/// Inverse-distance rolloff factor past `REF_DISTANCE`.
const ROLLOFF: f32 = 1.0;

/// Stereo gains for a world-space source: equal-power pan from the source's
/// screen-space x, inverse-distance attenuation from the camera eye (so zooming out
/// quiets the battlefield) and extra attenuation for off-screen sources.
pub fn spatial_gains(l: &Listener, pos: [f32; 3], volume: f32) -> (f32, f32) {
    let m = &l.view_proj;
    let clip = |r: usize| m[0][r] * pos[0] + m[1][r] * pos[1] + m[2][r] * pos[2] + m[3][r];
    let (cx, cy, cw) = (clip(0), clip(1), clip(3));
    let (nx, ny) = if cw > 1e-4 { (cx / cw, cy / cw) } else { (0.0, 3.0) };
    let pan = nx.clamp(-1.0, 1.0);
    let off = (nx.abs() - 1.0).max(0.0) + (ny.abs() - 1.0).max(0.0);
    let offscreen = 1.0 / (1.0 + 1.5 * off.min(10.0));
    let d = ((pos[0] - l.eye[0]).powi(2) + (pos[1] - l.eye[1]).powi(2) + (pos[2] - l.eye[2]).powi(2)).sqrt();
    let dist_gain = REF_DISTANCE / (REF_DISTANCE + (d - REF_DISTANCE).max(0.0) * ROLLOFF);
    let g = volume.max(0.0) * dist_gain * offscreen;
    let theta = (pan + 1.0) * std::f32::consts::FRAC_PI_4;
    (g * theta.cos(), g * theta.sin())
}

/// Outcome of a cue trigger (also used by tests and render statistics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerOutcome {
    Played(u64),
    Muted,
    NoClip,
    Cooldown,
    Inaudible,
    VoiceLimit,
}

/// Running mixer counters.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MixStats {
    pub triggered: u64,
    pub dropped_cooldown: u64,
    pub dropped_voice_limit: u64,
    pub dropped_inaudible: u64,
    pub stolen: u64,
    pub peak_voices: usize,
    pub min_limiter_gain: f32,
}

enum Source {
    Clip(Arc<PcmClip>),
    Stream(Box<PcmStream>),
}

impl Source {
    fn rate(&self) -> u32 {
        match self {
            Source::Clip(c) => c.sample_rate,
            Source::Stream(s) => s.sample_rate,
        }
    }

    fn frames(&self) -> usize {
        match self {
            Source::Clip(c) => c.frames(),
            Source::Stream(s) => s.total_frames,
        }
    }

    #[inline]
    fn frame(&mut self, idx: usize) -> Option<(f32, f32)> {
        match self {
            Source::Clip(c) => {
                let ch = usize::from(c.channels.max(1));
                let base = idx * ch;
                let l = *c.samples.get(base)?;
                Some((l, if ch > 1 { c.samples[base + 1] } else { l }))
            }
            Source::Stream(s) => s.frame(idx),
        }
    }
}

struct Voice {
    id: u64,
    /// Id of the first voice of the trigger (layered cues start one voice per layer).
    group: u64,
    cue: Option<CueId>,
    bus: Bus,
    source: Source,
    pos: f64,
    step: f64,
    gain_l: f32,
    gain_r: f32,
    priority: f32,
    fade: f32,
    fade_target: f32,
    fade_delta: f32,
    looping: bool,
}

impl Voice {
    fn weight(&self) -> f32 {
        (self.gain_l + self.gain_r) * self.priority * self.fade.max(0.05)
    }

    /// Adds this voice into interleaved `out`; returns false once finished.
    fn render_into(&mut self, out: &mut [f32], bus_gain: f32) -> bool {
        let total = self.source.frames();
        if total == 0 {
            return false;
        }
        for f in 0..out.len() / 2 {
            if self.fade_delta != 0.0 {
                self.fade += self.fade_delta;
                if (self.fade_delta > 0.0 && self.fade >= self.fade_target)
                    || (self.fade_delta < 0.0 && self.fade <= self.fade_target)
                {
                    self.fade = self.fade_target;
                    self.fade_delta = 0.0;
                    if self.fade <= 0.0 {
                        return false;
                    }
                }
            }
            let mut idx = self.pos as usize;
            if idx >= total {
                if !self.looping {
                    return false;
                }
                self.pos %= total as f64;
                idx = self.pos as usize;
            }
            let frac = (self.pos - idx as f64) as f32;
            let Some(a) = self.source.frame(idx) else {
                return false;
            };
            let next = if idx + 1 < total {
                idx + 1
            } else if self.looping {
                0
            } else {
                idx
            };
            let b = self.source.frame(next).unwrap_or(a);
            let g = self.fade * bus_gain;
            out[2 * f] += (a.0 + (b.0 - a.0) * frac) * self.gain_l * g;
            out[2 * f + 1] += (a.1 + (b.1 - a.1) * frac) * self.gain_r * g;
            self.pos += self.step;
        }
        true
    }
}

struct CueSlot {
    /// Layers (XACT tracks) played together; each holds round-robin variations.
    layers: Vec<Vec<Arc<PcmClip>>>,
    spec: CueSpec,
    last_start: Option<u64>,
    next_variant: Vec<usize>,
}

const LIMITER_CEILING: f32 = 0.97;
/// Per-sample limiter release (~250 ms time constant at 48 kHz).
const LIMITER_RELEASE: f32 = 8.3e-5;

/// Voice mixer with buses, cooldowns, voice stealing and a peak limiter.
pub struct Mixer {
    muted: bool,
    master: f32,
    bus_gain: [f32; 5],
    listener: Option<Listener>,
    cues: Vec<CueSlot>,
    voices: Vec<Voice>,
    clock: u64,
    next_id: u64,
    limiter: f32,
    stats: MixStats,
}

impl Default for Mixer {
    fn default() -> Self {
        Self::new()
    }
}

impl Mixer {
    pub fn new() -> Self {
        Self {
            muted: false,
            master: 0.8,
            bus_gain: [1.0, 0.8, 1.0, 0.45, 0.35],
            listener: None,
            cues: Vec::new(),
            voices: Vec::with_capacity(MAX_VOICES + MAX_STREAM_VOICES),
            clock: 0,
            next_id: 1,
            limiter: 1.0,
            stats: MixStats {
                min_limiter_gain: 1.0,
                ..MixStats::default()
            },
        }
    }

    #[cfg(test)]
    pub fn register_cue(&mut self, clips: Vec<Arc<PcmClip>>, spec: CueSpec) -> CueId {
        self.register_layered_cue(vec![clips], spec)
    }

    /// Phase 24: registers a cue whose layers all play on every trigger (one variation
    /// per layer, round-robin per layer). Empty layers are dropped.
    pub fn register_layered_cue(&mut self, layers: Vec<Vec<Arc<PcmClip>>>, spec: CueSpec) -> CueId {
        let layers: Vec<Vec<Arc<PcmClip>>> = layers.into_iter().filter(|l| !l.is_empty()).collect();
        self.cues.push(CueSlot {
            next_variant: vec![0; layers.len()],
            layers,
            spec,
            last_start: None,
        });
        self.cues.len() - 1
    }

    pub fn set_listener(&mut self, l: Option<Listener>) {
        self.listener = l;
    }

    pub fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
        if muted {
            self.voices.clear();
        }
    }

    pub fn muted(&self) -> bool {
        self.muted
    }

    pub fn master(&self) -> f32 {
        self.master
    }

    pub fn set_master(&mut self, v: f32) {
        self.master = v.clamp(0.0, 1.5);
    }

    pub fn stats(&self) -> MixStats {
        self.stats
    }

    #[cfg(test)]
    pub fn active_voices(&self) -> usize {
        self.voices.len()
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Starts cue `cue` at world `pos` (`None` = non-spatial, centered). Layered cues
    /// start every layer (extra layers only while the voice pool has room).
    pub fn trigger(&mut self, cue: CueId, pos: Option<[f32; 3]>, volume: f32) -> TriggerOutcome {
        self.start(cue, pos, volume, false)
    }

    /// Phase 24: starts cue `cue` looping (unit construction / movement loops) until
    /// [`Mixer::fade_out`]. No cooldown; at most [`MAX_LOOP_VOICES`] loops play, the
    /// weakest loop is stolen for a stronger one.
    pub fn start_loop(&mut self, cue: CueId, pos: Option<[f32; 3]>, volume: f32) -> TriggerOutcome {
        self.start(cue, pos, volume, true)
    }

    fn gains(&self, spec: &CueSpec, pos: Option<[f32; 3]>, volume: f32) -> (f32, f32) {
        let vol = volume * spec.volume;
        match (pos, self.listener.as_ref()) {
            (Some(p), Some(l)) => spatial_gains(l, p, vol),
            _ => (vol * std::f32::consts::FRAC_1_SQRT_2, vol * std::f32::consts::FRAC_1_SQRT_2),
        }
    }

    fn remove_group(&mut self, group: u64) {
        self.voices.retain(|v| v.group != group);
    }

    /// Weight of a trigger group (its primary voice).
    fn group_weight(&self, group: u64) -> f32 {
        self.voices.iter().find(|v| v.id == group).map(Voice::weight).unwrap_or(0.0)
    }

    /// Steals the weakest group among `groups` if weaker than `weight`.
    fn steal_weakest(&mut self, groups: Vec<u64>, weight: f32, oldest: bool) -> bool {
        let victim = if oldest {
            groups.iter().copied().min()
        } else {
            groups
                .iter()
                .copied()
                .min_by(|&a, &b| self.group_weight(a).total_cmp(&self.group_weight(b)))
                .filter(|&g| self.group_weight(g) < weight)
        };
        match victim {
            Some(g) => {
                self.remove_group(g);
                self.stats.stolen += 1;
                true
            }
            None => {
                self.stats.dropped_voice_limit += 1;
                false
            }
        }
    }

    fn one_shot_count(&self) -> usize {
        self.voices.iter().filter(|v| !v.bus.is_stream()).count()
    }

    fn start(&mut self, cue: CueId, pos: Option<[f32; 3]>, volume: f32, looping: bool) -> TriggerOutcome {
        if self.muted {
            return TriggerOutcome::Muted;
        }
        let Some(slot) = self.cues.get(cue) else {
            return TriggerOutcome::NoClip;
        };
        if slot.layers.is_empty() {
            return TriggerOutcome::NoClip;
        }
        let spec = slot.spec;
        let cooldown = (spec.cooldown_secs.max(0.0) * OUTPUT_SAMPLE_RATE as f32) as u64;
        if !looping
            && let Some(last) = slot.last_start
            && self.clock.saturating_sub(last) < cooldown
        {
            self.stats.dropped_cooldown += 1;
            return TriggerOutcome::Cooldown;
        }
        let (gl, gr) = self.gains(&spec, pos, volume);
        if gl + gr < 0.01 {
            self.stats.dropped_inaudible += 1;
            return TriggerOutcome::Inaudible;
        }
        let weight = (gl + gr) * spec.priority;

        // Per-cue instance cap (loops: global loop cap), then the global one-shot cap.
        let same: Vec<u64> = self
            .voices
            .iter()
            .filter(|v| v.id == v.group && v.cue.is_some())
            .filter(|v| if looping { v.looping } else { v.cue == Some(cue) && !v.looping })
            .map(|v| v.id)
            .collect();
        let cap = if looping { MAX_LOOP_VOICES } else { spec.max_instances.max(1) };
        if same.len() >= cap && !self.steal_weakest(same, weight, spec.replace_oldest && !looping) {
            return TriggerOutcome::VoiceLimit;
        }
        if self.one_shot_count() >= MAX_VOICES {
            let groups: Vec<u64> = self
                .voices
                .iter()
                .filter(|v| !v.bus.is_stream() && v.id == v.group)
                .map(|v| v.id)
                .collect();
            if !self.steal_weakest(groups, weight, false) {
                return TriggerOutcome::VoiceLimit;
            }
        }

        let slot = &mut self.cues[cue];
        let picks: Vec<Arc<PcmClip>> = slot
            .layers
            .iter()
            .zip(slot.next_variant.iter_mut())
            .map(|(layer, nv)| {
                let c = Arc::clone(&layer[*nv % layer.len()]);
                *nv = nv.wrapping_add(1);
                c
            })
            .collect();
        slot.last_start = Some(self.clock);
        let group = self.next_id;
        for (li, clip) in picks.into_iter().enumerate() {
            if li > 0 && self.one_shot_count() >= MAX_VOICES {
                break;
            }
            let id = self.alloc_id();
            self.voices.push(Voice {
                id,
                group,
                cue: Some(cue),
                bus: spec.bus,
                step: f64::from(clip.sample_rate) / f64::from(OUTPUT_SAMPLE_RATE),
                source: Source::Clip(clip),
                pos: 0.0,
                gain_l: gl,
                gain_r: gr,
                priority: spec.priority,
                fade: 1.0,
                fade_target: 1.0,
                fade_delta: 0.0,
                looping,
            });
        }
        self.stats.triggered += 1;
        self.stats.peak_voices = self.stats.peak_voices.max(self.voices.len());
        TriggerOutcome::Played(group)
    }

    /// Phase 24: re-spatializes a playing cue (all its layers), e.g. a loop following its unit.
    pub fn set_voice_spatial(&mut self, id: u64, pos: Option<[f32; 3]>, volume: f32) {
        let Some(cue) = self.voices.iter().find(|v| v.id == id).and_then(|v| v.cue) else {
            return;
        };
        let Some(spec) = self.cues.get(cue).map(|c| c.spec) else {
            return;
        };
        let (gl, gr) = self.gains(&spec, pos, volume);
        for v in self.voices.iter_mut().filter(|v| v.group == id) {
            v.gain_l = gl;
            v.gain_r = gr;
        }
    }

    /// Number of looping cue instances playing (tests).
    #[cfg(test)]
    pub fn active_loops(&self) -> usize {
        self.voices.iter().filter(|v| v.looping && v.cue.is_some() && v.id == v.group).count()
    }

    fn play_source(&mut self, source: Source, bus: Bus, volume: f32, looping: bool, fade_in_secs: f32) -> Option<u64> {
        if self.muted {
            return None;
        }
        let streams: Vec<usize> = (0..self.voices.len())
            .filter(|&i| self.voices[i].bus.is_stream())
            .collect();
        if streams.len() >= MAX_STREAM_VOICES
            && let Some(oldest) = streams.into_iter().min_by_key(|&i| self.voices[i].id)
        {
            self.voices.swap_remove(oldest);
        }
        let fade_frames = fade_in_secs.max(0.0) * OUTPUT_SAMPLE_RATE as f32;
        let id = self.alloc_id();
        self.voices.push(Voice {
            id,
            group: id,
            cue: None,
            bus,
            step: f64::from(source.rate()) / f64::from(OUTPUT_SAMPLE_RATE),
            source,
            pos: 0.0,
            gain_l: volume,
            gain_r: volume,
            priority: 100.0,
            fade: if fade_frames > 1.0 { 0.0 } else { 1.0 },
            fade_target: 1.0,
            fade_delta: if fade_frames > 1.0 { 1.0 / fade_frames } else { 0.0 },
            looping,
        });
        Some(id)
    }

    /// Fades voice `id` out over `secs` and then stops it.
    pub fn fade_out(&mut self, id: u64, secs: f32) {
        for v in self.voices.iter_mut().filter(|v| v.group == id) {
            let frames = (secs.max(0.0) * OUTPUT_SAMPLE_RATE as f32).max(1.0);
            v.fade_target = 0.0;
            v.fade_delta = -(v.fade.max(1e-3)) / frames;
        }
    }

    pub fn set_voice_volume(&mut self, id: u64, volume: f32) {
        for v in self.voices.iter_mut().filter(|v| v.group == id) {
            v.gain_l = volume;
            v.gain_r = volume;
        }
    }

    pub fn voice_active(&self, id: u64) -> bool {
        self.voices.iter().any(|v| v.group == id)
    }

    /// Mixes `out.len() / 2` interleaved stereo frames (range `[-1, 1]`).
    pub fn mix(&mut self, out: &mut [f32]) {
        out.fill(0.0);
        let frames = out.len() / 2;
        if self.muted {
            self.voices.clear();
            self.clock += frames as u64;
            return;
        }
        let bus_gain = self.bus_gain;
        self.voices.retain_mut(|v| v.render_into(out, bus_gain[v.bus.idx()]));
        for f in 0..frames {
            let l = out[2 * f] * self.master;
            let r = out[2 * f + 1] * self.master;
            let peak = l.abs().max(r.abs());
            // Release toward unity, then clamp instantly (zero attack) to the ceiling.
            self.limiter += (1.0 - self.limiter) * LIMITER_RELEASE;
            if peak * self.limiter > LIMITER_CEILING {
                self.limiter = LIMITER_CEILING / peak;
                self.stats.min_limiter_gain = self.stats.min_limiter_gain.min(self.limiter);
            }
            out[2 * f] = (l * self.limiter).clamp(-1.0, 1.0);
            out[2 * f + 1] = (r * self.limiter).clamp(-1.0, 1.0);
        }
        self.clock += frames as u64;
    }
}

// ---------------------------------------------------------------------------
// Engine (realtime worker + cue resolution)
// ---------------------------------------------------------------------------

struct LibState {
    lib: CueLibrary,
    resolved: HashMap<String, Option<CueId>>,
}

struct EngineInner {
    mixer: Arc<Mutex<Mixer>>,
    library: Mutex<LibState>,
    stop: Arc<AtomicBool>,
    output_ok: Arc<AtomicBool>,
    worker: Mutex<Option<JoinHandle<()>>>,
    realtime: bool,
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Ok(mut w) = self.worker.lock()
            && let Some(h) = w.take()
        {
            let _ = h.join();
        }
    }
}

/// Cheaply clonable handle to the audio engine.
#[derive(Clone)]
pub struct AudioEngine {
    inner: Arc<EngineInner>,
}

impl std::fmt::Debug for AudioEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioEngine")
            .field("realtime", &self.inner.realtime)
            .field("muted", &self.is_muted())
            .finish()
    }
}

impl AudioEngine {
    /// Realtime engine over `game_dir/sounds`; spawns the output worker unless `muted`.
    pub fn new(game_dir: Option<&Path>, muted: bool) -> Self {
        let lib = game_dir
            .map(|d| CueLibrary::scan(&d.join("sounds")))
            .unwrap_or_else(CueLibrary::empty);
        Self::with_library(lib, muted, true)
    }

    /// Offline engine (no device; pull samples with [`AudioEngine::render`]).
    pub fn offline(game_dir: Option<&Path>) -> Self {
        let lib = game_dir
            .map(|d| CueLibrary::scan(&d.join("sounds")))
            .unwrap_or_else(CueLibrary::empty);
        Self::with_library(lib, false, false)
    }

    pub fn with_library(lib: CueLibrary, muted: bool, realtime: bool) -> Self {
        let mut mixer = Mixer::new();
        mixer.set_muted(muted);
        let engine = Self {
            inner: Arc::new(EngineInner {
                mixer: Arc::new(Mutex::new(mixer)),
                library: Mutex::new(LibState {
                    lib,
                    resolved: HashMap::new(),
                }),
                stop: Arc::new(AtomicBool::new(false)),
                output_ok: Arc::new(AtomicBool::new(true)),
                worker: Mutex::new(None),
                realtime,
            }),
        };
        if realtime && !muted {
            engine.ensure_worker();
        }
        engine
    }

    fn ensure_worker(&self) {
        let Ok(mut w) = self.inner.worker.lock() else {
            return;
        };
        if w.as_ref().is_some_and(|h| !h.is_finished()) {
            return;
        }
        if let Some(h) = w.take() {
            let _ = h.join();
        }
        self.inner.output_ok.store(true, Ordering::Relaxed);
        let mixer = Arc::clone(&self.inner.mixer);
        let stop = Arc::clone(&self.inner.stop);
        let ok = Arc::clone(&self.inner.output_ok);
        *w = thread::Builder::new()
            .name("farust-audio".into())
            .spawn(move || run_audio_worker(mixer, stop, ok))
            .ok();
    }

    fn mixer(&self) -> Option<std::sync::MutexGuard<'_, Mixer>> {
        self.inner.mixer.lock().ok()
    }

    /// False when a realtime engine found no working output sink.
    pub fn output_available(&self) -> bool {
        !self.inner.realtime || self.inner.output_ok.load(Ordering::Relaxed)
    }

    pub fn cue_count(&self) -> usize {
        self.inner.library.lock().map(|l| l.lib.len()).unwrap_or(0)
    }

    /// Runs `f` against the cue library (lookups such as [`CueLibrary::find_prefixed`]).
    pub fn with_library_ref<R>(&self, f: impl FnOnce(&CueLibrary) -> R) -> Option<R> {
        self.inner.library.lock().ok().map(|l| f(&l.lib))
    }

    /// Resolves (decoding on first use) and registers cue `name`; cached by name.
    pub fn resolve(&self, name: &str, spec: CueSpec) -> Option<CueId> {
        let key = name.to_ascii_lowercase();
        let mut st = self.inner.library.lock().ok()?;
        if let Some(r) = st.resolved.get(&key) {
            return *r;
        }
        let layers = st.lib.layer_clips(name);
        let id = if layers.is_empty() {
            None
        } else {
            self.mixer().map(|mut m| m.register_layered_cue(layers, spec))
        };
        st.resolved.insert(key, id);
        id
    }

    pub fn trigger(&self, cue: CueId, pos: Option<[f32; 3]>, volume: f32) -> TriggerOutcome {
        if !self.output_available() {
            return TriggerOutcome::Muted;
        }
        self.mixer()
            .map(|mut m| m.trigger(cue, pos, volume))
            .unwrap_or(TriggerOutcome::Muted)
    }

    /// Phase 24: starts `cue` looping; stop it with [`AudioEngine::fade_out`].
    pub fn start_loop(&self, cue: CueId, pos: Option<[f32; 3]>, volume: f32) -> TriggerOutcome {
        if !self.output_available() {
            return TriggerOutcome::Muted;
        }
        self.mixer()
            .map(|mut m| m.start_loop(cue, pos, volume))
            .unwrap_or(TriggerOutcome::Muted)
    }

    /// Phase 24: moves a playing (looping) cue to `pos`.
    pub fn set_voice_spatial(&self, id: u64, pos: Option<[f32; 3]>, volume: f32) {
        if let Some(mut m) = self.mixer() {
            m.set_voice_spatial(id, pos, volume);
        }
    }

    /// Plays wave `variant` of cue `name` non-spatially: streamed from disk when possible
    /// (music), otherwise decoded. Returns the voice id.
    pub fn play_stream(&self, name: &str, variant: usize, bus: Bus, volume: f32, looping: bool, fade_in_secs: f32) -> Option<u64> {
        if !self.output_available() || self.is_muted() {
            return None;
        }
        let source = {
            let mut st = self.inner.library.lock().ok()?;
            match st.lib.stream(name, variant) {
                Some(s) => Source::Stream(Box::new(s)),
                None => {
                    let clips = st.lib.clips(name);
                    if clips.is_empty() {
                        return None;
                    }
                    Source::Clip(Arc::clone(&clips[variant % clips.len()]))
                }
            }
        };
        self.mixer()?.play_source(source, bus, volume, looping, fade_in_secs)
    }

    pub fn fade_out(&self, id: u64, secs: f32) {
        if let Some(mut m) = self.mixer() {
            m.fade_out(id, secs);
        }
    }

    pub fn set_voice_volume(&self, id: u64, volume: f32) {
        if let Some(mut m) = self.mixer() {
            m.set_voice_volume(id, volume);
        }
    }

    pub fn voice_active(&self, id: u64) -> bool {
        self.mixer().is_some_and(|m| m.voice_active(id))
    }

    pub fn set_listener(&self, l: Listener) {
        if let Some(mut m) = self.mixer() {
            m.set_listener(Some(l));
        }
    }

    pub fn set_muted(&self, muted: bool) {
        if let Some(mut m) = self.mixer() {
            m.set_muted(muted);
        }
        if !muted && self.inner.realtime {
            self.ensure_worker();
        }
    }

    pub fn is_muted(&self) -> bool {
        self.mixer().is_none_or(|m| m.muted())
    }

    /// Adds `delta` to the master volume (clamped to `0..=1.5`), returning the new value.
    pub fn adjust_master(&self, delta: f32) -> f32 {
        self.mixer()
            .map(|mut m| {
                let v = ((m.master() + delta) * 20.0).round() / 20.0;
                m.set_master(v);
                m.master()
            })
            .unwrap_or(0.0)
    }

    pub fn stats(&self) -> MixStats {
        self.mixer().map(|m| m.stats()).unwrap_or_default()
    }

    /// Offline pull: mixes `frames` interleaved stereo frames.
    pub fn render(&self, frames: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; frames * 2];
        if let Some(mut m) = self.mixer() {
            m.mix(&mut out);
        }
        out
    }
}

/// Raw S16LE stereo sink candidates, tried in order (stderr silenced).
fn spawn_pcm_sink(skip: usize) -> Option<(usize, Child)> {
    let rate = OUTPUT_SAMPLE_RATE.to_string();
    let pa_rate = format!("--rate={rate}");
    let candidates: [(&str, Vec<&str>); 3] = [
        (
            "pw-cat",
            vec![
                "--playback", "--raw", "--rate", &rate, "--channels", "2", "--format", "s16",
                "--latency", "20ms", "--media-role", "Game", "-",
            ],
        ),
        (
            "pacat",
            vec![
                "--playback", "--raw", &pa_rate, "--channels=2", "--format=s16le",
                "--latency-msec=30", "--client-name=farust",
            ],
        ),
        (
            "aplay",
            vec!["-q", "-t", "raw", "-f", "S16_LE", "-r", &rate, "-c", "2", "--buffer-time=60000"],
        ),
    ];
    for (i, (cmd, args)) in candidates.iter().enumerate().skip(skip) {
        if let Ok(child) = Command::new(cmd)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            return Some((i, child));
        }
    }
    None
}

fn stop_child(mut c: Child) {
    drop(c.stdin.take());
    let _ = c.kill();
    let _ = c.wait();
}

/// Realtime loop: keeps the sink ~`SINK_AHEAD_MS` ahead of the wall clock in 5 ms
/// quanta (low latency without a native audio API). Falls back through the sink list
/// and exits quietly (output unavailable) when none works.
fn run_audio_worker(mixer: Arc<Mutex<Mixer>>, stop: Arc<AtomicBool>, output_ok: Arc<AtomicBool>) {
    let Some((mut which, mut child)) = spawn_pcm_sink(0) else {
        output_ok.store(false, Ordering::Relaxed);
        return;
    };
    let rate = u64::from(OUTPUT_SAMPLE_RATE);
    let ahead = rate * SINK_AHEAD_MS / 1000;
    let mut start = Instant::now();
    let mut written: u64 = 0;
    let mut mixbuf = vec![0.0f32; MIX_CHUNK_FRAMES * 2];
    let mut bytes = Vec::with_capacity(MIX_CHUNK_FRAMES * 4);
    while !stop.load(Ordering::Relaxed) {
        let mut failed = matches!(child.try_wait(), Ok(Some(_)) | Err(_));
        let now = (start.elapsed().as_secs_f64() * rate as f64) as u64;
        if written + rate / 4 < now {
            // Stalled (suspend / debugger): resync instead of bursting.
            start = Instant::now();
            written = 0;
        }
        let now = (start.elapsed().as_secs_f64() * rate as f64) as u64;
        while !failed && written < now + ahead {
            match mixer.lock() {
                Ok(mut m) => m.mix(&mut mixbuf),
                Err(_) => return,
            }
            bytes.clear();
            for &s in &mixbuf {
                bytes.extend_from_slice(&f32_to_i16(s).to_le_bytes());
            }
            failed = child
                .stdin
                .as_mut()
                .is_none_or(|stdin| stdin.write_all(&bytes).is_err());
            written += MIX_CHUNK_FRAMES as u64;
        }
        if failed {
            stop_child(child);
            match spawn_pcm_sink(which + 1) {
                Some((w, c)) => {
                    which = w;
                    child = c;
                    start = Instant::now();
                    written = 0;
                    continue;
                }
                None => {
                    output_ok.store(false, Ordering::Relaxed);
                    if let Ok(mut m) = mixer.lock() {
                        m.voices.clear();
                    }
                    return;
                }
            }
        }
        thread::sleep(Duration::from_millis(2));
    }
    stop_child(child);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_synthetic_xwb() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(WBND_MAGIC);
        buf.extend_from_slice(&43u32.to_le_bytes());
        buf.extend_from_slice(&42u32.to_le_bytes());
        let bank_data_off = 52u32;
        let bank_data_len = 96u32;
        let meta_off = bank_data_off + bank_data_len;
        let meta_len = 24u32;
        let names_off = meta_off + meta_len;
        let names_len = 64u32;
        let wave_off = names_off + names_len;
        let wave_samples: Vec<i16> = (0..160).map(|i| ((i * 512) % 16000) as i16).collect();
        let wave_len = (wave_samples.len() * 2) as u32;

        for (o, l) in [
            (bank_data_off, bank_data_len),
            (meta_off, meta_len),
            (0, 0),
            (names_off, names_len),
            (wave_off, wave_len),
        ] {
            buf.extend_from_slice(&o.to_le_bytes());
            buf.extend_from_slice(&l.to_le_bytes());
        }

        buf.extend_from_slice(&0u32.to_le_bytes()); // non-compact
        buf.extend_from_slice(&1u32.to_le_bytes()); // entry_count=1
        let mut bname = [0u8; 64];
        bname[..4].copy_from_slice(b"Test");
        buf.extend_from_slice(&bname);
        buf.extend_from_slice(&24u32.to_le_bytes()); // meta_size
        buf.extend_from_slice(&64u32.to_le_bytes()); // name_size
        buf.extend_from_slice(&4u32.to_le_bytes()); // alignment
        buf.extend_from_slice(&0u32.to_le_bytes()); // compact_fmt
        buf.extend_from_slice(&[0u8; 8]);

        // Entry metadata (24B): tag=0, ch=1, sr=32000, ba=2, bps16=1
        let mini_fmt: u32 = (1 << 2) | (32000 << 5) | (2 << 23) | (1 << 31);
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&mini_fmt.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&wave_len.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);

        let mut ename = [0u8; 64];
        ename[..5].copy_from_slice(b"boom0");
        buf.extend_from_slice(&ename);

        for s in wave_samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        buf
    }

    fn tone(frames: usize, amp: f32) -> PcmClip {
        PcmClip {
            sample_rate: OUTPUT_SAMPLE_RATE,
            channels: 1,
            samples: (0..frames)
                .map(|i| amp * (i as f32 * 0.07).sin())
                .collect(),
        }
    }

    /// Identity view-projection: ndc == world xy, w == 1.
    fn flat_listener() -> Listener {
        Listener {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            eye: [0.0, 0.0, 0.0],
        }
    }

    fn energy(pcm: &[f32], ch: usize) -> f32 {
        pcm.iter().skip(ch).step_by(2).map(|s| s * s).sum()
    }

    #[test]
    fn parses_xwb_decodes_pcm_and_emits_wav() {
        let bytes = build_synthetic_xwb();
        let wb = WaveBank::parse(bytes.clone()).expect("parse synthetic xwb");
        assert_eq!(wb.name, "Test");
        assert_eq!(wb.entries.len(), 1);
        assert_eq!(wb.entries[0].name.as_deref(), Some("boom0"));
        assert_eq!(wb.entries[0].sample_rate, 32_000);
        let clip = wb.decode(0).expect("decode synthetic wave entry");
        assert_eq!((clip.sample_rate, clip.channels, clip.frames()), (32_000, 1, 160));
        assert!((clip.samples[1] - 512.0 / 32768.0).abs() < 1e-6);
        let (sr, mono) = wb.decode_mono_f32(0).expect("decode mono");
        assert_eq!((sr, mono.len()), (32_000, 160));
        let wav = wb.to_wav_bytes(0).expect("wav bytes");
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(wav.len(), 44 + 320);

        // Lazy (header-only) open + on-disk read yields the identical clip.
        let path = std::env::temp_dir().join(format!("farust_xact_test_{}.xwb", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let lazy = WaveBank::open_lazy(&path).expect("lazy open");
        assert!(lazy.data.is_empty());
        let raw = lazy.read_entry(0, Some(&path)).unwrap();
        assert_eq!(decode_wave_data(&lazy.entries[0], &raw).unwrap(), clip);
        let mut stream = PcmStream::open(&path, &lazy.entries[0]).unwrap();
        assert_eq!(stream.frame(1).map(|f| f.0), Some(clip.samples[1]));
        assert_eq!(stream.frame(160), None);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn decodes_synthetic_ms_adpcm_block_and_rejects_xwma() {
        // Mono block: predictor 0, delta 16, sample1 100, sample2 50, then nibbles.
        let mut block = vec![0u8];
        block.extend_from_slice(&16i16.to_le_bytes());
        block.extend_from_slice(&100i16.to_le_bytes());
        block.extend_from_slice(&50i16.to_le_bytes());
        block.extend_from_slice(&[0x00, 0x10]); // nibbles 0,0,1,0
        // Block size passed directly (XACT would store `block_align - 22` per channel).
        let pcm = decode_ms_adpcm(&block, 1, block.len()).unwrap();
        let ints: Vec<i32> = pcm.iter().map(|s| (s * 32768.0).round() as i32).collect();
        // pred(coef 256,0)=s1; nibble 1 adds delta (16 adapted to max(16*230/256,16)=16).
        assert_eq!(ints, vec![50, 100, 100, 100, 116, 116]);

        let wma = WaveEntryInfo {
            index: 0,
            name: None,
            format_tag: FORMAT_WMA,
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 16,
            block_align: 1,
            data_offset: 0,
            data_length: 4,
        };
        assert!(decode_wave_data(&wma, &[0, 0, 0, 0]).is_err());
    }

    #[test]
    fn parses_xsb_simple_cue() {
        let mut xsb = vec![0u8; 256];
        xsb[0..4].copy_from_slice(SDBK_MAGIC);
        xsb[19..21].copy_from_slice(&1u16.to_le_bytes()); // simple cues = 1
        xsb[25..27].copy_from_slice(&1u16.to_le_bytes()); // total cues = 1
        xsb[27] = 1; // 1 wavebank
        let cn = b"Test_Cue\0";
        xsb[30..32].copy_from_slice(&(cn.len() as u16).to_le_bytes());
        let sc_off = 138u32;
        let cn_off = 148u32;
        let wb_off = 160u32;
        let snd_off = 224u32;
        let offs = [sc_off, u32::MAX, cn_off, u32::MAX, u32::MAX, u32::MAX, wb_off, u32::MAX, u32::MAX, snd_off];
        for (i, v) in offs.iter().enumerate() {
            xsb[34 + i * 4..38 + i * 4].copy_from_slice(&v.to_le_bytes());
        }
        xsb[74..78].copy_from_slice(b"Test");
        xsb[sc_off as usize + 1..sc_off as usize + 5].copy_from_slice(&snd_off.to_le_bytes());
        xsb[cn_off as usize..cn_off as usize + cn.len()].copy_from_slice(cn);
        xsb[wb_off as usize..wb_off as usize + 4].copy_from_slice(b"Test");
        xsb[snd_off as usize] = 0; // simple
        xsb[snd_off as usize + 7..snd_off as usize + 9].copy_from_slice(&12u16.to_le_bytes());
        xsb[snd_off as usize + 9..snd_off as usize + 11].copy_from_slice(&3u16.to_le_bytes());
        xsb[snd_off as usize + 11] = 0;

        let sb = SoundBank::parse(&xsb).expect("parse synthetic xsb");
        assert_eq!(sb.name, "Test");
        assert_eq!(sb.cues.len(), 1);
        assert_eq!(sb.cues[0].name, "Test_Cue");
        assert_eq!(sb.cues[0].waves, vec![("Test".to_string(), 3)]);
    }

    #[test]
    fn extracts_type3_variation_table() {
        // Complex sound (flags&1) with a type-3 PlayWave-variation event of 2 waves.
        let mut chunk = vec![0x01u8, 0, 0, 0, 0, 0, 0, 0, 0];
        chunk.extend_from_slice(&[0x01, 0x03, 0x00, 0x00]);
        chunk.extend_from_slice(b"\x20\x00\x00\xff\x0c");
        chunk.extend_from_slice(&[0xff, 0, 0, 0xa0, 0x8c]);
        chunk.extend_from_slice(&2u16.to_le_bytes());
        chunk.extend_from_slice(&[0x43, 0x00, 0xff, 0xff, 0xff, 0xff]);
        chunk.extend_from_slice(&[7, 0, 0, 0, 0xff, 1, 0, 0, 0, 0xff]);
        let len = chunk.len() as u16;
        chunk[7..9].copy_from_slice(&len.to_le_bytes());
        assert_eq!(extract_sound_waves(&chunk, 0, 1), vec![(0, 7), (0, 1)]);
    }

    #[test]
    fn groups_layers_and_reads_looping_play_events() {
        // Two tracks: a looping PlayWave (header byte 0x84 = type 4 + flags) and a
        // type-3 variation event with 2 waves.
        let mut chunk = vec![0x01u8, 0, 0, 0, 0, 0, 0, 0, 0, 2];
        chunk.extend_from_slice(&[0x01, 0x84, 0x0c, 0x00]);
        chunk.extend_from_slice(b"\x20\x00\x00\xff\x0c");
        chunk.extend_from_slice(&[5, 0, 0, 0xff]);
        chunk.extend_from_slice(&[0x01, 0x03, 0x00, 0x00]);
        chunk.extend_from_slice(b"\x20\x00\x00\xff\x0c");
        chunk.extend_from_slice(&[0xff, 0, 0, 0xa0, 0x8c]);
        chunk.extend_from_slice(&2u16.to_le_bytes());
        chunk.extend_from_slice(&[0x43, 0x00, 0xff, 0xff, 0xff, 0xff]);
        chunk.extend_from_slice(&[7, 0, 0, 0, 0xff, 1, 0, 0, 0, 0xff]);
        let len = chunk.len() as u16;
        chunk[7..9].copy_from_slice(&len.to_le_bytes());
        assert_eq!(extract_sound_tracks(&chunk, 0, 1), vec![vec![(0, 5)], vec![(0, 7), (0, 1)]]);
        assert_eq!(extract_sound_waves(&chunk, 0, 1), vec![(0, 5), (0, 7), (0, 1)]);
    }

    #[test]
    fn layered_cue_plays_all_layers_and_rotates_variations_per_layer() {
        let mut lib = CueLibrary::empty();
        lib.insert_layers(
            "Layered",
            vec![vec![tone(4000, 0.1)], vec![tone(4100, 0.2), tone(4200, 0.3)], vec![tone(4300, 0.4)]],
        );
        let layers = lib.layer_clips("Layered");
        assert_eq!(layers.iter().map(Vec::len).collect::<Vec<_>>(), vec![1, 2, 1]);
        let mut m = Mixer::new();
        let cue = m.register_layered_cue(layers, CueSpec::sfx(1.0, 0.0, 2, 1.0));
        let TriggerOutcome::Played(g1) = m.trigger(cue, None, 1.0) else { panic!() };
        assert_eq!(m.active_voices(), 3, "all three layers start together");
        let frames = |m: &Mixer| -> Vec<usize> { m.voices.iter().map(|v| v.source.frames()).collect() };
        assert_eq!(frames(&m), vec![4000, 4100, 4300]);
        let TriggerOutcome::Played(_) = m.trigger(cue, None, 1.0) else { panic!() };
        assert_eq!(frames(&m)[3..], [4000, 4200, 4300], "layer 2 advanced to its 2nd variation");
        // Instance cap counts triggers, not layers: an equally loud third trigger is
        // dropped, a louder one steals a whole group (3 voices).
        assert_eq!(m.trigger(cue, None, 1.0), TriggerOutcome::VoiceLimit);
        assert!(matches!(m.trigger(cue, None, 2.0), TriggerOutcome::Played(_)));
        assert_eq!(m.active_voices(), 6);
        m.fade_out(g1, 0.0);
        // A plain cue is a single layer of variations (round-robin unchanged).
        let plain = m.register_cue(vec![Arc::new(tone(100, 0.1)), Arc::new(tone(200, 0.1))], CueSpec::sfx(1.0, 0.0, 8, 1.0));
        let before = m.active_voices();
        m.trigger(plain, None, 1.0);
        assert_eq!(m.active_voices(), before + 1);
    }

    #[test]
    fn loops_repeat_until_faded_and_respect_loop_cap() {
        let mut m = Mixer::new();
        let cue = m.register_cue(vec![Arc::new(tone(480, 0.2))], CueSpec::sfx(1.0, 5.0, 8, 1.0));
        let mut ids = Vec::new();
        for _ in 0..MAX_LOOP_VOICES {
            match m.start_loop(cue, None, 1.0) {
                TriggerOutcome::Played(id) => ids.push(id),
                o => panic!("{o:?}"),
            }
        }
        assert_eq!(m.active_loops(), MAX_LOOP_VOICES, "loops ignore the cue cooldown");
        assert_eq!(m.start_loop(cue, None, 1.0), TriggerOutcome::VoiceLimit);
        let mut out = vec![0.0f32; 4800];
        m.mix(&mut out);
        m.mix(&mut out);
        assert_eq!(m.active_loops(), MAX_LOOP_VOICES, "loops outlive their 480-frame clip");
        m.set_voice_spatial(ids[0], Some([0.5, 0.0, 0.0]), 0.5);
        m.fade_out(ids[0], 0.001);
        m.mix(&mut out);
        assert!(!m.voice_active(ids[0]));
        assert_eq!(m.active_loops(), MAX_LOOP_VOICES - 1);
        assert!(matches!(m.start_loop(cue, None, 1.0), TriggerOutcome::Played(_)));
    }

    #[test]
    fn spatial_pan_and_attenuation() {
        let l = flat_listener();
        let (cl, cr) = spatial_gains(&l, [0.0, 0.0, 10.0], 1.0);
        assert!((cl - cr).abs() < 1e-5, "centered source must be balanced");
        assert!((cl * cl + cr * cr - 1.0).abs() < 1e-4, "equal-power pan");
        let (rl, rr) = spatial_gains(&l, [0.8, 0.0, 10.0], 1.0);
        assert!(rr > rl * 3.0, "screen-right source pans right: {rl} {rr}");
        let (ll, lr) = spatial_gains(&l, [-0.8, 0.0, 10.0], 1.0);
        assert!(ll > lr * 3.0);
        // Distance (zoom) attenuation: twice past the reference distance halves-ish.
        let (nl, _) = spatial_gains(&l, [0.0, 0.0, REF_DISTANCE], 1.0);
        let (fl, _) = spatial_gains(&l, [0.0, 0.0, REF_DISTANCE * 3.0], 1.0);
        assert!((fl / nl - 1.0 / 3.0).abs() < 1e-3, "inverse-distance rolloff {fl}/{nl}");
        // Off-screen sources are quieter than on-screen ones at the same distance.
        let (ol, or) = spatial_gains(&l, [3.0, 0.0, 0.0], 1.0);
        let (il, ir) = spatial_gains(&l, [0.99, 0.0, 0.0], 1.0);
        assert!(ol + or < 0.5 * (il + ir));
        // Behind the camera (w <= 0) is attenuated, not mirrored.
        let mut back = l;
        back.view_proj[3][3] = -1.0;
        let (bl, br) = spatial_gains(&back, [0.0, 0.0, 0.0], 1.0);
        assert!(bl + br < 0.3 * (cl + cr));

        // End-to-end: mixer output is louder in the right channel.
        let mut m = Mixer::new();
        m.set_listener(Some(l));
        let cue = m.register_cue(vec![Arc::new(tone(4000, 0.5))], CueSpec::sfx(1.0, 0.0, 4, 1.0));
        assert!(matches!(m.trigger(cue, Some([0.8, 0.0, 5.0]), 1.0), TriggerOutcome::Played(_)));
        let mut out = vec![0.0; 2048];
        m.mix(&mut out);
        assert!(energy(&out, 1) > energy(&out, 0) * 3.0);
    }

    #[test]
    fn voice_limit_steals_quietest_and_drops_quieter() {
        let mut m = Mixer::new();
        let cue = m.register_cue(vec![Arc::new(tone(48_000, 0.3))], CueSpec::sfx(1.0, 0.0, usize::MAX, 1.0));
        for _ in 0..MAX_VOICES {
            assert!(matches!(m.trigger(cue, None, 0.5), TriggerOutcome::Played(_)));
        }
        assert_eq!(m.active_voices(), MAX_VOICES);
        // Same loudness -> dropped; never exceeds the cap.
        for _ in 0..200 {
            assert_eq!(m.trigger(cue, None, 0.5), TriggerOutcome::VoiceLimit);
        }
        assert_eq!(m.active_voices(), MAX_VOICES);
        // Louder sound steals the quietest voice.
        assert!(matches!(m.trigger(cue, None, 1.0), TriggerOutcome::Played(_)));
        assert_eq!(m.active_voices(), MAX_VOICES);
        let st = m.stats();
        assert_eq!((st.dropped_voice_limit, st.stolen), (200, 1));
        // Streams (music) bypass the one-shot cap.
        let src = Source::Clip(Arc::new(tone(1000, 0.1)));
        assert!(m.play_source(src, Bus::Music, 1.0, true, 0.0).is_some());
        assert_eq!(m.active_voices(), MAX_VOICES + 1);

        // 200 simultaneous full-scale shots: limiter keeps output below the ceiling.
        let mut out = vec![0.0; 9600];
        m.mix(&mut out);
        let peak = out.iter().fold(0.0f32, |a, s| a.max(s.abs()));
        assert!(peak > 0.1 && peak <= LIMITER_CEILING + 1e-6, "peak {peak}");
    }

    #[test]
    fn per_cue_instance_cap_and_replace_oldest() {
        let mut m = Mixer::new();
        let gun = m.register_cue(vec![Arc::new(tone(48_000, 0.3))], CueSpec::sfx(1.0, 0.0, 3, 1.0));
        for _ in 0..3 {
            assert!(matches!(m.trigger(gun, None, 1.0), TriggerOutcome::Played(_)));
        }
        assert_eq!(m.trigger(gun, None, 1.0), TriggerOutcome::VoiceLimit);
        assert_eq!(m.active_voices(), 3);
        let sel = m.register_cue(vec![Arc::new(tone(48_000, 0.3))], CueSpec::voice(1.0, 0.0));
        let TriggerOutcome::Played(a) = m.trigger(sel, None, 1.0) else { panic!() };
        let TriggerOutcome::Played(b) = m.trigger(sel, None, 1.0) else { panic!() };
        assert!(!m.voice_active(a) && m.voice_active(b), "acknowledgement replaces previous");
    }

    #[test]
    fn cue_cooldown_uses_mixer_clock() {
        let mut m = Mixer::new();
        let cue = m.register_cue(vec![Arc::new(tone(480, 0.3))], CueSpec::sfx(1.0, 0.1, 8, 1.0));
        assert!(matches!(m.trigger(cue, None, 1.0), TriggerOutcome::Played(_)));
        assert_eq!(m.trigger(cue, None, 1.0), TriggerOutcome::Cooldown);
        let mut out = vec![0.0; 2 * 4_000]; // 83 ms
        m.mix(&mut out);
        assert_eq!(m.trigger(cue, None, 1.0), TriggerOutcome::Cooldown);
        let mut out = vec![0.0; 2 * 1_000]; // total 104 ms
        m.mix(&mut out);
        assert!(matches!(m.trigger(cue, None, 1.0), TriggerOutcome::Played(_)));
        assert_eq!(m.stats().dropped_cooldown, 2);
    }

    #[test]
    fn muted_and_inaudible_and_engine_offline_render() {
        let mut lib = CueLibrary::empty();
        lib.insert_clips("Test_Boom", vec![tone(9600, 0.8), tone(4800, 0.4)]);
        assert!(lib.has("test_boom"));
        assert_eq!(lib.find_prefixed("TEST_", MEM_BANK, &["_Loop"]).as_deref(), Some("Test_Boom"));
        let eng = AudioEngine::with_library(lib, false, false);
        let cue = eng.resolve("TEST_BOOM", CueSpec::sfx(1.0, 0.0, 4, 1.0)).expect("resolve");
        assert_eq!(eng.resolve("test_boom", CueSpec::ui(1.0, 0.0)), Some(cue), "cached by name");
        assert!(eng.resolve("missing", CueSpec::ui(1.0, 0.0)).is_none());
        eng.set_listener(flat_listener());
        // Far away -> inaudible, not played.
        assert_eq!(eng.trigger(cue, Some([0.0, 0.0, 1.0e6]), 1.0), TriggerOutcome::Inaudible);
        assert!(matches!(eng.trigger(cue, None, 1.0), TriggerOutcome::Played(_)));
        let pcm = eng.render(4800);
        assert!(pcm.iter().any(|s| s.abs() > 0.05));
        assert!((eng.adjust_master(0.1) - 0.9).abs() < 1e-6);
        eng.set_muted(true);
        assert_eq!(eng.trigger(cue, None, 1.0), TriggerOutcome::Muted);
        assert!(eng.render(480).iter().all(|&s| s == 0.0));
        assert!(eng.play_stream("Test_Boom", 0, Bus::Music, 1.0, true, 0.5).is_none());
        eng.set_muted(false);
        let id = eng.play_stream("Test_Boom", 1, Bus::Music, 1.0, true, 0.0).expect("loop");
        let _ = eng.render(48_000); // loops past the 4800-frame clip
        assert!(eng.voice_active(id));
        eng.fade_out(id, 0.05);
        let _ = eng.render(4_800);
        assert!(!eng.voice_active(id));
    }

    #[test]
    fn recursive_scan_and_direct_wave_index_playback() {
        assert_eq!(parse_wave_ref("XGG#12"), Some(("XGG", 12)));
        assert_eq!(parse_wave_ref(" #0 "), None);
        assert_eq!(parse_wave_ref("XGG#nope"), None);

        let root = std::env::temp_dir().join(format!("farust_xact_scan_{}", std::process::id()));
        let voice_dir = root.join("Voice").join("US");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&voice_dir).unwrap();

        let xwb = build_synthetic_xwb();
        std::fs::write(voice_dir.join("Test.xwb"), &xwb).unwrap();

        let mut xsb = vec![0u8; 256];
        xsb[0..4].copy_from_slice(SDBK_MAGIC);
        xsb[19..21].copy_from_slice(&1u16.to_le_bytes());
        xsb[25..27].copy_from_slice(&1u16.to_le_bytes());
        xsb[27] = 1;
        let cn = b"Voice_Alert\0";
        xsb[30..32].copy_from_slice(&(cn.len() as u16).to_le_bytes());
        let (sc_off, cn_off, wb_off, snd_off) = (138u32, 148u32, 164u32, 228u32);
        let offs = [sc_off, u32::MAX, cn_off, u32::MAX, u32::MAX, u32::MAX, wb_off, u32::MAX, u32::MAX, snd_off];
        for (i, v) in offs.iter().enumerate() {
            xsb[34 + i * 4..38 + i * 4].copy_from_slice(&v.to_le_bytes());
        }
        xsb[74..78].copy_from_slice(b"Test");
        xsb[sc_off as usize + 1..sc_off as usize + 5].copy_from_slice(&snd_off.to_le_bytes());
        xsb[cn_off as usize..cn_off as usize + cn.len()].copy_from_slice(cn);
        xsb[wb_off as usize..wb_off as usize + 4].copy_from_slice(b"Test");
        xsb[snd_off as usize] = 0;
        xsb[snd_off as usize + 7..snd_off as usize + 9].copy_from_slice(&12u16.to_le_bytes());
        xsb[snd_off as usize + 9..snd_off as usize + 11].copy_from_slice(&0u16.to_le_bytes());
        xsb[snd_off as usize + 11] = 0;
        std::fs::write(voice_dir.join("Test.xsb"), &xsb).unwrap();

        let lib = CueLibrary::scan(&root);
        assert!(lib.has("Voice_Alert"), "nested Voice/US SoundBank must be indexed");
        let eng = AudioEngine::with_library(lib, false, false);
        // Play by cue name from nested Voice/US/Test.xsb
        let by_cue = eng.resolve("Voice_Alert", CueSpec::voice(1.0, 0.0)).expect("resolve nested cue");
        assert!(matches!(eng.trigger(by_cue, None, 1.0), TriggerOutcome::Played(_)));
        // Play directly by wave index ("Test#0") without needing a cue wrapper
        let by_idx = eng.resolve("Test#0", CueSpec::voice(1.0, 0.0)).expect("resolve wave index");
        assert!(matches!(eng.trigger(by_idx, None, 1.0), TriggerOutcome::Played(_)));
        assert!(eng.resolve("Test#99", CueSpec::voice(1.0, 0.0)).is_none());
        let pcm = eng.render(480);
        assert!(pcm.iter().any(|s| s.abs() > 0.01));

        let _ = std::fs::remove_dir_all(&root);
    }
}
