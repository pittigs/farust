//! Phase 25 deterministic mutation fuzzing of the binary asset parsers
//! (`.scm`, `.sca`, `.scmap`, `.dds`).
//!
//! Seeds come from small synthetic files (always) plus a few real files from the game install
//! (when present). Each iteration applies seeded mutations — byte/bit flips, truncation,
//! length-field inflation, chunk insert/delete — and runs the parser plus its main consumers
//! (skeleton rig + clip sampling, DDS decode, heightmap queries) under `catch_unwind`.
//! In `cargo test` a guarding allocator additionally flags any single allocation that is out
//! of proportion to the input size (unbounded allocation from corrupt length fields).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Mutex;

/// Parser families exercised by the fuzzer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Scm,
    Sca,
    ScMap,
    Dds,
    Loc,
}

impl Kind {
    pub const ALL: [Kind; 5] = [Kind::Scm, Kind::Sca, Kind::ScMap, Kind::Dds, Kind::Loc];

    pub fn name(self) -> &'static str {
        match self {
            Kind::Scm => "scm",
            Kind::Sca => "sca",
            Kind::ScMap => "scmap",
            Kind::Dds => "dds",
            Kind::Loc => "loc",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Seed {
    pub kind: Kind,
    pub label: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct KindStats {
    pub runs: u64,
    pub ok: u64,
    pub err: u64,
    pub panics: u64,
}

#[derive(Debug, Clone)]
pub struct Finding {
    pub kind: Kind,
    pub seed_label: String,
    pub iteration: u64,
    pub mutation: String,
    pub message: String,
}

#[derive(Debug, Clone, Default)]
pub struct Report {
    #[allow(dead_code)]
    pub iterations: u64,
    pub stats: Vec<(Kind, KindStats)>,
    pub findings: Vec<Finding>,
}

impl Report {
    pub fn panics(&self) -> u64 {
        self.stats.iter().map(|(_, s)| s.panics).sum()
    }
}

/// Small deterministic xorshift64* generator.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next() % n as u64) as usize }
    }
}

/// Applies 1..=3 seeded mutations, returning a short description.
fn mutate(rng: &mut Rng, data: &mut Vec<u8>) -> String {
    let mut desc = Vec::new();
    let n = 1 + rng.below(3);
    for _ in 0..n {
        let len = data.len();
        match rng.below(8) {
            0 => {
                let k = 1 + rng.below(8);
                for _ in 0..k {
                    if !data.is_empty() {
                        let i = rng.below(data.len());
                        data[i] = rng.next() as u8;
                    }
                }
                desc.push(format!("bytes{k}"));
            }
            1 => {
                if !data.is_empty() {
                    let i = rng.below(data.len());
                    data[i] ^= 1 << rng.below(8);
                }
                desc.push("bit".to_string());
            }
            2 => {
                let at = rng.below(len + 1);
                data.truncate(at);
                desc.push(format!("trunc@{at}"));
            }
            3 | 4 => {
                // Length/count/offset field inflation: 4-byte aligned, biased to the header.
                if len >= 4 {
                    let region = if rng.below(3) == 0 { len } else { len.min(256) };
                    let at = (rng.below(region.saturating_sub(3).max(1)) / 4) * 4;
                    let at = at.min(len - 4);
                    let v: u32 = match rng.below(7) {
                        0 => u32::MAX,
                        1 => 0x7FFF_FFFF,
                        2 => 0x8000_0000,
                        3 => (len as u32).saturating_mul(2).saturating_add(1),
                        4 => 0x0100_0000,
                        5 => 0,
                        _ => rng.next() as u32,
                    };
                    data[at..at + 4].copy_from_slice(&v.to_le_bytes());
                    desc.push(format!("u32@{at}={v:#x}"));
                }
            }
            5 => {
                if len >= 2 {
                    let at = rng.below(len - 1);
                    let v = [0xFFu8, 0x7F, 0x00, 0x80][rng.below(4)];
                    data[at] = v;
                    data[at + 1] = v;
                    desc.push(format!("u16@{at}"));
                }
            }
            6 => {
                let at = rng.below(len + 1);
                let k = 1 + rng.below(64);
                let fill = rng.next() as u8;
                data.splice(at..at, std::iter::repeat_n(fill, k));
                desc.push(format!("ins{k}@{at}"));
            }
            _ => {
                if len > 1 {
                    let at = rng.below(len);
                    let k = (1 + rng.below(64)).min(len - at);
                    data.drain(at..at + k);
                    desc.push(format!("del{k}@{at}"));
                }
            }
        }
    }
    desc.join("+")
}

/// Runs one parser (and its main consumers) over `data`.
/// Returns `true` when the parse itself succeeded.
pub fn exercise(kind: Kind, data: &[u8]) -> bool {
    match kind {
        Kind::Scm => match crate::scm::ScmMesh::parse(data) {
            Ok(m) => {
                let _ = m.triangle_count();
                let _ = m.find_turret_bones();
                let rig = crate::sca::UnitRig::new(m.bones.clone(), [0.0; 3], 1.0);
                let pose = rig.bind_pose();
                let _ = rig.palette_from_pose(&pose, crate::sca::AimPose { yaw: 0.5, pitch: 0.2, recoil: 0.3 });
                let _ = crate::sca::compute_scm_skin_matrices(&m, None, Some(("Turret", 0.4)), Some(("Barrel", 0.1)), [0.0; 3], 1.0);
                true
            }
            Err(_) => false,
        },
        Kind::Sca => match crate::sca::ScaAnimation::parse(data) {
            Ok(a) => {
                for t in [0.0f32, 0.37, a.duration, -1.0, f32::NAN, 1e9] {
                    let _ = a.sample_local(t, true);
                    let _ = a.sample_local(t, false);
                }
                let names: Vec<String> = a.bone_names.clone();
                let refs: Vec<&str> = names.iter().map(String::as_str).collect();
                let _ = a.build_bone_remap(&refs);
                let bones: Vec<crate::scm::ScmBone> = names
                    .iter()
                    .enumerate()
                    .take(256)
                    .map(|(i, n)| crate::scm::ScmBone {
                        inv_bind_pose: crate::sca::mat4_identity().concat().try_into().unwrap_or([0.0; 16]),
                        name: n.clone(),
                        parent_index: a.parent_links.get(i).copied().unwrap_or(-1),
                        position: [0.0; 3],
                        rotation: [1.0, 0.0, 0.0, 0.0],
                    })
                    .collect();
                if let Some(clip) = crate::sca::AnimClip::new(a, &refs) {
                    let mut rig = crate::sca::UnitRig::new(bones, [0.0; 3], 1.0);
                    let d = clip.duration();
                    rig.walk = Some(clip);
                    for t in [0.0f32, d * 0.5, d * 3.3, f32::NAN] {
                        let mode = rig.mode_for(true);
                        let ct = rig.clip_time(mode, t, 0.25);
                        let pose = rig.sample_pose(rig.clip_for(mode), ct, true);
                        let _ = rig.palette_from_pose(&pose, crate::sca::AimPose::default());
                    }
                }
                true
            }
            Err(_) => false,
        },
        Kind::ScMap => match crate::scmap::ScMap::parse(data) {
            Ok(m) => {
                let _ = m.height_at(0, 0);
                let _ = m.height_at(m.heightmap_width, m.heightmap_height);
                let _ = m.height_at(u32::MAX, 3);
                let _ = m.height_stats();
                if let Some(tt) = &m.terrain_textures {
                    let _ = crate::dds::parse_dds(&tt.texturemap0_dds);
                }
                true
            }
            Err(_) => false,
        },
        Kind::Dds => {
            let header_ok = crate::dds::parse_dds(data).is_ok();
            if let Ok(img) = crate::dds_decode::decode_dds_rgba8(data)
                && img.width <= 4096
                && img.height <= 4096
            {
                let _ = crate::dds_decode::resize_rgba_bilinear(&img, 7, 5);
            }
            let _ = crate::dds_decode::decode_dds_cubemap_rgba8(data);
            header_ok
        }
        Kind::Loc => {
            let ok_db = crate::loc::parse_strings_db(data).is_ok();
            let _ = crate::loc::parse_strings_db_lossy(data);
            let ok_tt = crate::loc::parse_tooltips(data).is_ok();
            let decoded = crate::loc::decode_text_bytes(data);
            let _ = crate::loc::split_loc_tag(&decoded);
            ok_db || ok_tt
        }
    }
}

static LAST_PANIC: Mutex<Option<String>> = Mutex::new(None);

/// Runs `iters` mutation iterations over `seeds` (round-robin), starting at `seed`.
/// `check` is invoked after every successful-or-failing run to enforce extra invariants
/// (the test allocator guard); returning `Some(msg)` records a finding.
pub fn run(seeds: &[Seed], iters: u64, seed: u64, check: &mut dyn FnMut(Kind, &[u8]) -> Option<String>) -> Report {
    let mut report = Report {
        iterations: iters,
        stats: Kind::ALL.iter().map(|&k| (k, KindStats::default())).collect(),
        findings: Vec::new(),
    };
    if seeds.is_empty() {
        return report;
    }
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|info| {
        let loc = info.location().map(|l| format!(" at {}:{}", l.file(), l.line())).unwrap_or_default();
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "panic".to_string());
        if let Ok(mut g) = LAST_PANIC.lock() {
            *g = Some(format!("{msg}{loc}"));
        }
    }));
    let trace = std::env::var_os("FARUST_FUZZ_TRACE").is_some();
    let mut rng = Rng::new(seed);
    // Unmutated seeds first.
    let mut plan: Vec<(usize, bool)> = (0..seeds.len()).map(|i| (i, false)).collect();
    plan.extend((0..iters).map(|i| ((i as usize) % seeds.len(), true)));
    for (it, (si, do_mutate)) in plan.into_iter().enumerate() {
        let s = &seeds[si];
        let mut data = s.bytes.clone();
        let mutation = if do_mutate { mutate(&mut rng, &mut data) } else { "none".to_string() };
        if trace {
            eprintln!("fuzz {it}: {} [{}] {mutation}", s.kind.name(), s.label);
        }
        let res = catch_unwind(AssertUnwindSafe(|| exercise(s.kind, &data)));
        let st = &mut report.stats.iter_mut().find(|(k, _)| *k == s.kind).expect("kind").1;
        st.runs += 1;
        match res {
            Ok(true) => st.ok += 1,
            Ok(false) => st.err += 1,
            Err(_) => {
                st.panics += 1;
                let message = LAST_PANIC.lock().ok().and_then(|mut g| g.take()).unwrap_or_default();
                if report.findings.len() < 64 {
                    report.findings.push(Finding {
                        kind: s.kind,
                        seed_label: s.label.clone(),
                        iteration: it as u64,
                        mutation: mutation.clone(),
                        message,
                    });
                }
            }
        }
        if let Some(msg) = check(s.kind, &data)
            && report.findings.len() < 64
        {
            report.findings.push(Finding {
                kind: s.kind,
                seed_label: s.label.clone(),
                iteration: it as u64,
                mutation,
                message: msg,
            });
        }
    }
    std::panic::set_hook(prev_hook);
    report
}

// ---------------------------------------------------------------------------
// Seed corpus
// ---------------------------------------------------------------------------

fn put_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn put_f32(v: &mut Vec<u8>, x: f32) {
    v.extend_from_slice(&x.to_le_bytes());
}

/// Minimal valid `.scm`: 2 bones, 3 vertices, 1 triangle, 1 info string.
pub fn synthetic_scm() -> Vec<u8> {
    let names = b"Root\0Turret\0";
    let bone_off = 48u32;
    let names_off = bone_off + 2 * 108;
    let vert_off = names_off + names.len() as u32;
    let idx_off = vert_off + 3 * 68;
    let info_off = idx_off + 6;
    let info = b"exporter=fuzz\0lod=0\0";
    let mut v = Vec::new();
    v.extend_from_slice(b"MODL");
    for x in [5, bone_off, 2, vert_off, 0, 3, idx_off, 3, info_off, info.len() as u32, 2] {
        put_u32(&mut v, x);
    }
    for (bi, parent) in [(0u32, -1i32), (1, 0)] {
        let m = crate::sca::mat4_identity();
        for row in m {
            for c in row {
                put_f32(&mut v, c);
            }
        }
        for c in [0.0f32, bi as f32, 0.0, 1.0, 0.0, 0.0, 0.0] {
            put_f32(&mut v, c);
        }
        put_u32(&mut v, names_off + if bi == 0 { 0 } else { 5 });
        v.extend_from_slice(&parent.to_le_bytes());
        v.extend_from_slice(&[0u8; 8]);
    }
    v.extend_from_slice(names);
    for i in 0..3u32 {
        for c in [i as f32, 0.0, (i % 2) as f32, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0, 0.0] {
            put_f32(&mut v, c);
        }
        v.extend_from_slice(&[(i % 2) as u8, 0, 0, 0]);
    }
    for i in [0u16, 1, 2] {
        v.extend_from_slice(&i.to_le_bytes());
    }
    v.extend_from_slice(info);
    v
}

/// Minimal valid `.sca`: 2 bones, 2 frames.
pub fn synthetic_sca() -> Vec<u8> {
    let names = b"Root\0Turret\0";
    let names_off = 36u32;
    let links_off = names_off + names.len() as u32;
    let anim_off = links_off + 8;
    let frame_size = 8 + 2 * 28u32;
    let mut v = Vec::new();
    v.extend_from_slice(b"ANIM");
    for x in [5u32, 2, 1.0f32.to_bits(), 2, names_off, links_off, anim_off, frame_size] {
        put_u32(&mut v, x);
    }
    v.extend_from_slice(names);
    v.extend_from_slice(&(-1i32).to_le_bytes());
    v.extend_from_slice(&0i32.to_le_bytes());
    for c in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] {
        put_f32(&mut v, c);
    }
    for (t, x) in [(0.0f32, 0.0f32), (1.0, 2.0)] {
        put_f32(&mut v, t);
        put_u32(&mut v, 0);
        for c in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, x, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0] {
            put_f32(&mut v, c);
        }
    }
    v
}

/// DDS header (`fourcc` or uncompressed RGBA masks) + payload.
fn dds_bytes(w: u32, h: u32, fourcc: Option<&[u8; 4]>, bits: u32, cubemap: bool, payload: &[u8]) -> Vec<u8> {
    let mut v = vec![0u8; 128];
    v[0..4].copy_from_slice(b"DDS ");
    let mut w32 = |off: usize, x: u32| v[off..off + 4].copy_from_slice(&x.to_le_bytes());
    w32(4, 124);
    w32(8, 0x1007);
    w32(12, h);
    w32(16, w);
    w32(20, if fourcc.is_some() { payload.len() as u32 } else { w * bits / 8 });
    w32(28, 1);
    w32(76, 32);
    match fourcc {
        Some(_) => w32(80, 0x4),
        None => {
            w32(80, 0x41);
            w32(88, bits);
            w32(92, 0x00ff_0000);
            w32(96, 0x0000_ff00);
            w32(100, 0x0000_00ff);
            w32(104, 0xff00_0000);
        }
    }
    w32(108, 0x1000);
    if cubemap {
        w32(112, 0xFE00);
    }
    if let Some(cc) = fourcc {
        v[84..88].copy_from_slice(cc);
    }
    v.extend_from_slice(payload);
    v
}

/// Small valid DDS images: DXT1/3/5, BGRA8 and a 6-face DXT1 cubemap.
pub fn synthetic_dds() -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let dxt1: Vec<u8> = (0..4 * 8).map(|i| (i * 37 + 11) as u8).collect();
    let dxt5: Vec<u8> = (0..4 * 16).map(|i| (i * 53 + 7) as u8).collect();
    let bgra: Vec<u8> = (0..8 * 4 * 4).map(|i| (i * 13) as u8).collect();
    out.push(("dxt1_8x8".into(), dds_bytes(8, 8, Some(b"DXT1"), 0, false, &dxt1)));
    out.push(("dxt3_8x8".into(), dds_bytes(8, 8, Some(b"DXT3"), 0, false, &dxt5)));
    out.push(("dxt5_8x8".into(), dds_bytes(8, 8, Some(b"DXT5"), 0, false, &dxt5)));
    out.push(("bgra_8x4".into(), dds_bytes(8, 4, None, 32, false, &bgra)));
    let cube: Vec<u8> = (0..6).flat_map(|_| dxt1.iter().copied()).collect();
    out.push(("dxt1_cube".into(), dds_bytes(8, 8, Some(b"DXT1"), 0, true, &cube)));
    out
}

/// Minimal valid v2.60 `.scmap` (4x4 heightmap) with a short texture tail.
pub fn synthetic_scmap() -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(b"Map\x1a");
    v.extend_from_slice(&2i32.to_le_bytes());
    put_u32(&mut v, 0xEDFE_EFBE);
    put_u32(&mut v, 2);
    put_f32(&mut v, 4.0);
    put_f32(&mut v, 4.0);
    put_u32(&mut v, 0);
    v.extend_from_slice(&0u16.to_le_bytes());
    let preview = dds_bytes(4, 4, Some(b"DXT1"), 0, false, &[0x55; 8]);
    put_u32(&mut v, preview.len() as u32);
    v.extend_from_slice(&preview);
    v.extend_from_slice(&60i32.to_le_bytes());
    v.extend_from_slice(&4i32.to_le_bytes());
    v.extend_from_slice(&4i32.to_le_bytes());
    put_f32(&mut v, 1.0 / 128.0);
    for i in 0..25u16 {
        v.extend_from_slice(&(i * 100).to_le_bytes());
    }
    v.push(0);
    v.extend_from_slice(b"TTerrain\0/textures/environment/defaultbackground.dds\0/textures/environment/sky.dds\0");
    put_u32(&mut v, 1);
    v.extend_from_slice(b"<default>\0/textures/environment/cube.dds\0");
    for i in 0..23 {
        put_f32(&mut v, i as f32 * 0.1);
    }
    v.push(1);
    for x in [2.0f32, 1.5, 0.5] {
        put_f32(&mut v, x);
    }
    // Texture tail (parsed opportunistically; mutations exercise its bounds checks).
    for i in 0..20 {
        put_f32(&mut v, i as f32);
    }
    v.extend_from_slice(b"/water/cube.dds\0/water/ramp.dds\0");
    v.extend_from_slice(&[0u8; 16]);
    for _ in 0..4 {
        v.extend_from_slice(&[0u8; 8]);
        v.extend_from_slice(b"/water/wave.dds\0");
    }
    put_u32(&mut v, 0);
    v.extend_from_slice(&[0u8; 28]);
    for _ in 0..19 {
        v.extend_from_slice(b"/env/s.dds\0");
        put_f32(&mut v, 4.0);
    }
    v.extend_from_slice(&[0u8; 8]);
    put_u32(&mut v, 1);
    put_u32(&mut v, 7);
    put_u32(&mut v, 1);
    put_u32(&mut v, 1);
    put_u32(&mut v, 6);
    v.extend_from_slice(b"/d.dds");
    for i in 0..12 {
        put_f32(&mut v, i as f32);
    }
    put_u32(&mut v, 0);
    v.extend_from_slice(&[0u8; 8]);
    put_u32(&mut v, 0);
    let tm0 = dds_bytes(4, 4, Some(b"DXT1"), 0, false, &[0x11; 8]);
    put_u32(&mut v, tm0.len() as u32);
    v.extend_from_slice(&tm0);
    v
}

/// Minimal valid loc string table + tooltips seed.
pub fn synthetic_loc() -> Vec<(String, Vec<u8>)> {
    vec![
        (
            "synthetic_strings_db.lua".into(),
            "\u{FEFF}# synthetic loc\nuel0001_desc=\"Armored Command Unit\"\nuel0201_desc=\"Mittlerer Panzer (äöüß)\"\n['enh_01']=\"Tech-2-Paket\\nZeile 2\"\n"
                .as_bytes()
                .to_vec(),
        ),
        (
            "synthetic_tooltips.lua".into(),
            b"Tooltips = {\n move = { title = \"<LOC tooltipui0000>Move\", description = \"<LOC d>Go\", keyID = \"move\" },\n}\n"
                .to_vec(),
        ),
    ]
}

/// Synthetic seeds for every parser family.
pub fn synthetic_seeds() -> Vec<Seed> {
    let mut v = vec![
        Seed { kind: Kind::Scm, label: "synthetic.scm".into(), bytes: synthetic_scm() },
        Seed { kind: Kind::Sca, label: "synthetic.sca".into(), bytes: synthetic_sca() },
        Seed { kind: Kind::ScMap, label: "synthetic.scmap".into(), bytes: synthetic_scmap() },
    ];
    for (label, bytes) in synthetic_dds() {
        v.push(Seed { kind: Kind::Dds, label, bytes });
    }
    for (label, bytes) in synthetic_loc() {
        v.push(Seed { kind: Kind::Loc, label, bytes });
    }
    v
}

/// A few small real files from the install (`units.scd` meshes/clips/textures and the
/// smallest skirmish map), or nothing when the game dir is missing.
pub fn real_seeds(game_dir: &Path, max_per_kind: usize, include_maps: bool) -> Vec<Seed> {
    let mut out = Vec::new();
    let units = crate::scd::gamedata_dir(game_dir).join("units.scd");
    if units.is_file() {
        let wanted = ["uel0201/uel0201_lod0.scm", "uel0201/uel0201_lod1.scm", "url0107/url0107_lod0.scm",
            "uel0201/uel0201_lod1_albedo.dds", "url0107/url0107_lod1_albedo.dds", "uel0201/uel0201_lod1_specteam.dds",
            "url0107/url0107_awalk01.sca", "url0106/url0106_awalk01.sca", "url0107/url0107_adeath01.sca"];
        if let Ok(files) = crate::scd::read_scd_matching(&units, |n| {
            let l = n.replace('\\', "/").to_ascii_lowercase();
            wanted.iter().any(|w| l.ends_with(w))
        }) {
            let mut counts = [0usize; 4];
            for (name, bytes) in files {
                let l = name.to_ascii_lowercase();
                let kind = if l.ends_with(".scm") {
                    Kind::Scm
                } else if l.ends_with(".sca") {
                    Kind::Sca
                } else {
                    Kind::Dds
                };
                let c = &mut counts[kind as usize];
                if *c < max_per_kind && bytes.len() <= 1 << 20 {
                    *c += 1;
                    out.push(Seed { kind, label: name, bytes });
                }
            }
        }
    }
    if include_maps
        && let Ok(maps) = std::fs::read_dir(game_dir.join("maps"))
    {
        let mut best: Option<(u64, std::path::PathBuf)> = None;
        for e in maps.flatten() {
            if let Ok(inner) = std::fs::read_dir(e.path()) {
                for f in inner.flatten() {
                    let p = f.path();
                    if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("scmap"))
                        && let Ok(md) = f.metadata()
                        && best.as_ref().is_none_or(|(s, _)| md.len() < *s)
                    {
                        best = Some((md.len(), p));
                    }
                }
            }
        }
        if let Some((_, p)) = best
            && let Ok(bytes) = std::fs::read(&p)
        {
            out.push(Seed { kind: Kind::ScMap, label: p.display().to_string(), bytes });
        }
    }
    out
}

/// Text table of a report.
pub fn format_report(r: &Report) -> String {
    let mut s = format!("{:<7} {:>9} {:>9} {:>9} {:>7}\n", "parser", "runs", "ok", "err", "panics");
    for (k, st) in &r.stats {
        s.push_str(&format!("{:<7} {:>9} {:>9} {:>9} {:>7}\n", k.name(), st.runs, st.ok, st.err, st.panics));
    }
    for f in &r.findings {
        s.push_str(&format!(
            "FINDING {} [{}] iter {} mutation {}: {}\n",
            f.kind.name(),
            f.seed_label,
            f.iteration,
            f.mutation,
            f.message
        ));
    }
    s
}

#[cfg(test)]
pub(crate) mod alloc_guard {
    //! Test-only allocator that records the largest single allocation made by the current
    //! thread while armed, so the fuzz test can flag allocations driven by corrupt counts.
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static ARMED: Cell<bool> = const { Cell::new(false) };
        static PEAK: Cell<usize> = const { Cell::new(0) };
    }

    pub struct Guard;

    #[inline]
    fn note(size: usize) {
        let _ = ARMED.try_with(|a| {
            if a.get() {
                let _ = PEAK.try_with(|p| p.set(p.get().max(size)));
            }
        });
    }

    unsafe impl GlobalAlloc for Guard {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            note(layout.size());
            unsafe { System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            note(layout.size());
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            note(new_size);
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static GLOBAL: Guard = Guard;

    pub fn arm() {
        ARMED.with(|a| a.set(true));
        PEAK.with(|p| p.set(0));
    }

    pub fn disarm() -> usize {
        ARMED.with(|a| a.set(false));
        PEAK.with(|p| p.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Largest legitimate single allocation for an input of `len` bytes: decoded images expand
    /// DXT1 8x (+ RGBA), so allow 64x the input plus a fixed 32 MiB floor.
    fn alloc_budget(len: usize) -> usize {
        (32 << 20) + len.saturating_mul(64)
    }

    #[test]
    fn synthetic_seeds_parse_cleanly() {
        for s in synthetic_seeds() {
            assert!(exercise(s.kind, &s.bytes), "seed {} must parse", s.label);
        }
        let m = crate::scmap::ScMap::parse(&synthetic_scmap()).unwrap();
        assert_eq!(m.heightmap.len(), 25);
        assert!(m.terrain_textures.is_some(), "synthetic texture tail parses");
    }

    #[test]
    fn mutation_fuzz_parsers_never_panic_or_overallocate() {
        let mut seeds = synthetic_seeds();
        let game_dir = crate::scd::default_game_dir();
        let real = real_seeds(&game_dir, 2, false);
        let have_real = !real.is_empty();
        seeds.extend(real);
        let iters = if have_real { 60_000 } else { 100_000 };
        // Re-run the parser with the allocation guard armed (outside `run`'s own call so the
        // panic counting stays independent of the allocator check).
        let mut check = |kind: Kind, data: &[u8]| {
            alloc_guard::arm();
            let _ = catch_unwind(AssertUnwindSafe(|| exercise(kind, data)));
            let peak = alloc_guard::disarm();
            (peak > alloc_budget(data.len())).then(|| format!("allocation of {peak} bytes for {}-byte input", data.len()))
        };
        let report = run(&seeds, iters, 0x5EED_F00D, &mut check);
        let text = format_report(&report);
        assert!(report.findings.is_empty(), "fuzz findings:\n{text}");
        // Mutations must reach both outcomes for every parser family.
        for (k, st) in &report.stats {
            assert!(st.runs > 0 && st.err > 0, "{} never failed:\n{text}", k.name());
        }
    }

    /// Regressions for the findings fixed in Phase 25.
    #[test]
    fn fixed_findings_stay_fixed() {
        // DDS: 65535x65535 DXT1 header with a 40-byte payload used to allocate 17 GB up front.
        let mut dds = dds_bytes(8, 8, Some(b"DXT1"), 0, false, &[0u8; 32]);
        dds[12..16].copy_from_slice(&0xFFFFu32.to_le_bytes());
        dds[16..20].copy_from_slice(&0xFFFFu32.to_le_bytes());
        alloc_guard::arm();
        assert!(crate::dds_decode::decode_dds_rgba8(&dds).is_err());
        assert!(alloc_guard::disarm() < 1 << 20);
        // DDS cubemap: absurd mip count must not loop/overflow.
        let mut cube = synthetic_dds().pop().unwrap().1;
        cube[28..32].copy_from_slice(&u32::MAX.to_le_bytes());
        let _ = crate::dds_decode::decode_dds_cubemap_rgba8(&cube);
        // SCA: NaN duration used to panic in `clamp(0.0, duration)`.
        let mut sca = synthetic_sca();
        sca[12..16].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        let a = crate::sca::ScaAnimation::parse(&sca).unwrap();
        assert_eq!(a.duration, 0.0);
        assert!(exercise(Kind::Sca, &sca));
        // SCA: huge bone count with zero frames and a huge frame size must fail before
        // allocating per-bone storage.
        let mut sca = synthetic_sca();
        sca[8..12].copy_from_slice(&0u32.to_le_bytes());
        sca[16..20].copy_from_slice(&100_000_000u32.to_le_bytes());
        sca[32..36].copy_from_slice(&u32::MAX.to_le_bytes());
        alloc_guard::arm();
        assert!(crate::sca::ScaAnimation::parse(&sca).is_err());
        assert!(alloc_guard::disarm() < 1 << 20);
    }

    #[test]
    fn mutate_is_deterministic() {
        let base = synthetic_scm();
        let a: Vec<String> = {
            let mut r = Rng::new(9);
            (0..50).map(|_| mutate(&mut r, &mut base.clone())).collect()
        };
        let b: Vec<String> = {
            let mut r = Rng::new(9);
            (0..50).map(|_| mutate(&mut r, &mut base.clone())).collect()
        };
        assert_eq!(a, b);
    }
}
