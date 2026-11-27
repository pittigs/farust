//! Phase 47 replay vault index, map+start thumbnails, search/filtering, and event bookmarks.
//!
//! Stores a compact binary index (`index.farust-idx`) and cached PNG thumbnails (`thumbs/*.png`)
//! alongside recorded `.farust-replay` files in the replay vault directory (`$FARUST_VAULT_DIR`,
//! else `$FARUST_DATA_DIR/replays`, `fa-unpack/replays`, or `$XDG_DATA_HOME/farust/replays`).

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::dds_decode::RgbaImage;
use crate::lobby;
use crate::mods::ModIdentity;
use crate::replay::{
    Replay, read_i64, read_str, read_u8, read_u32, read_u64, write_str,
};
use crate::savegame;
use crate::scenario;
use crate::sim::{SCALE, TICK_HZ};
use crate::skirmish::{
    ArmyId, ArmySetup, Catalog, GameResult, MapContext, RULES_VERSION, Role, SimEvent, Skirmish,
    UnitId, VictoryCondition,
};
use crate::ui_kit::theme;

pub const INDEX_MAGIC: &[u8; 8] = b"FARVX1\0\0";
pub const INDEX_VERSION: u32 = 1;
pub const INDEX_FILE: &str = "index.farust-idx";
pub const THUMBS_DIR: &str = "thumbs";
pub const THUMB_SIZE: u32 = 160;
pub const MAX_BOOKMARKS: usize = 64;

/// Kill-burst detection window (10 seconds = 100 ticks) and cooldown (30 seconds = 300 ticks).
const KILL_BURST_WINDOW_TICKS: u64 = 10 * TICK_HZ;
const KILL_BURST_MIN_KILLS: u32 = 8;
const KILL_BURST_COOLDOWN_TICKS: u64 = 30 * TICK_HZ;
const MAX_KILL_BURST_BOOKMARKS: usize = 6;

/// Human-readable faction label for a sim [`crate::skirmish::FACTION_TABLE`] index
/// (`0=UEF`, `1=Cybran`, `2=Aeon`, `3=Seraphim`).
pub fn sim_faction_label(faction: usize) -> &'static str {
    match crate::skirmish::FACTION_TABLE
        .get(faction)
        .map(|(k, _)| *k)
    {
        Some("uef") => "UEF",
        Some("cybran") => "Cybran",
        Some("aeon") => "Aeon",
        Some("seraphim") => "Seraphim",
        _ => "Unknown",
    }
}

// ---------------------------------------------------------------------------
// Vault path & hashing helpers
// ---------------------------------------------------------------------------

/// Directory where replays, `index.farust-idx`, and `thumbs/` live.
/// Override with `FARUST_VAULT_DIR`, else falls back to `savegame::data_dir("replays")`.
pub fn default_vault_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("FARUST_VAULT_DIR")
        && !d.is_empty()
    {
        return PathBuf::from(d);
    }
    savegame::data_dir("replays")
}

/// Deterministic FNV-1a64 content hash of raw replay bytes.
pub fn content_hash_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Days since 1970-01-01 → proleptic Gregorian `(year, month, day)`.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe as i64 + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

/// Formats Unix seconds as `YYYY-MM-DD` (UTC).
pub fn format_utc_date(secs: u64) -> String {
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Formats Unix seconds as `YYYY-MM-DD HH:MM` (UTC).
pub fn format_utc_datetime(secs: u64) -> String {
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    let rem = secs % 86_400;
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}", rem / 3600, (rem / 60) % 60)
}

/// Extracts `mission "NAME"` from a mission script header if present.
pub fn parse_mission_title(src: &str) -> String {
    for line in src.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("mission ") {
            let trimmed = rest.trim().trim_matches('"').trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }
    "Custom Mission".to_string()
}

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BookmarkKind {
    AcuDeath,
    Experimental,
    NukeLaunch,
    KillBurst,
}

impl BookmarkKind {
    pub fn code(self) -> u8 {
        match self {
            Self::AcuDeath => 0,
            Self::Experimental => 1,
            Self::NukeLaunch => 2,
            Self::KillBurst => 3,
        }
    }

    pub fn from_code(c: u8) -> Option<Self> {
        match c {
            0 => Some(Self::AcuDeath),
            1 => Some(Self::Experimental),
            2 => Some(Self::NukeLaunch),
            3 => Some(Self::KillBurst),
            _ => None,
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            Self::AcuDeath => "ACU",
            Self::Experimental => "EXP",
            Self::NukeLaunch => "NUKE",
            Self::KillBurst => "CLASH",
        }
    }

    pub fn color(self) -> [f32; 4] {
        match self {
            Self::AcuDeath => [1.0, 0.28, 0.24, 1.0],
            Self::Experimental => [0.84, 0.42, 1.0, 1.0],
            Self::NukeLaunch => [1.0, 0.62, 0.14, 1.0],
            Self::KillBurst => [0.98, 0.86, 0.28, 1.0],
        }
    }

    fn priority(self) -> u8 {
        match self {
            Self::AcuDeath => 0,
            Self::NukeLaunch => 1,
            Self::Experimental => 2,
            Self::KillBurst => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bookmark {
    pub tick: u64,
    pub kind: BookmarkKind,
    pub army: Option<ArmyId>,
    /// World coordinates in milli `(x, z)` so jumping to a bookmark can focus the camera.
    pub pos: (i64, i64),
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VaultArmy {
    pub name: String,
    pub faction: usize,
    pub team: u8,
    pub start: (i64, i64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VaultWinner {
    Army {
        id: ArmyId,
        name: String,
        faction: usize,
        team: u8,
    },
    Draw,
}

impl VaultWinner {
    pub fn label(&self) -> String {
        match self {
            Self::Army { name, faction, team, .. } => {
                format!("{name} ({}, T{})", sim_faction_label(*faction), team + 1)
            }
            Self::Draw => "Draw".to_string(),
        }
    }

    pub fn short_label(&self) -> String {
        match self {
            Self::Army { name, .. } => name.clone(),
            Self::Draw => "Draw".to_string(),
        }
    }

    pub fn army_id(&self) -> Option<ArmyId> {
        match self {
            Self::Army { id, .. } => Some(*id),
            Self::Draw => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VaultEntry {
    pub file_name: String,
    pub file_size: u64,
    pub mtime_secs: u64,
    pub mtime_nanos: u32,
    pub content_hash: u64,
    pub map_id: String,
    pub map_name: String,
    pub date_ymd: String,
    pub duration_ticks: u64,
    pub seed: u64,
    pub victory: VictoryCondition,
    pub unit_cap: u32,
    pub rules: u32,
    pub armies: Vec<VaultArmy>,
    pub winner: Option<VaultWinner>,
    pub mods: Vec<ModIdentity>,
    pub mission_name: Option<String>,
    pub commands: u32,
    pub checkpoints: u32,
    pub final_hash: u64,
    pub bookmarks: Vec<Bookmark>,
    pub sim_indexed: bool,
    pub thumb_file: Option<String>,
    pub share_code: Option<String>,
}

impl VaultEntry {
    pub fn needs_older_build(&self) -> bool {
        self.rules != RULES_VERSION
    }

    pub fn rules_status(&self) -> String {
        if self.rules == RULES_VERSION {
            format!("v{} (compatible)", self.rules)
        } else {
            format!("v{} (needs older build; current v{RULES_VERSION})", self.rules)
        }
    }

    pub fn path_in(&self, vault_dir: &Path) -> PathBuf {
        vault_dir.join(&self.file_name)
    }

    pub fn thumb_path_in(&self, vault_dir: &Path) -> Option<PathBuf> {
        self.thumb_file.as_ref().map(|rel| vault_dir.join(rel))
    }

    pub fn winner_label(&self) -> String {
        match &self.winner {
            Some(w) => w.label(),
            None if self.needs_older_build() => "Unknown (older rules)".to_string(),
            None if !self.sim_indexed => "Unindexed".to_string(),
            None => "Unfinished".to_string(),
        }
    }

    pub fn duration_mmss(&self) -> String {
        let s = self.duration_ticks / TICK_HZ;
        format!("{:02}:{:02}", s / 60, s % 60)
    }

    pub fn armies_summary(&self) -> String {
        self.armies
            .iter()
            .map(|a| format!("{}({})", a.name, sim_faction_label(a.faction)))
            .collect::<Vec<_>>()
            .join(" vs ")
    }

    pub fn setups(&self) -> Vec<ArmySetup> {
        self.armies
            .iter()
            .map(|a| ArmySetup {
                name: a.name.clone(),
                faction: a.faction,
                start: a.start,
                team: a.team,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Search & filter
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VaultQuery {
    pub text: Vec<String>,
    pub map: Option<String>,
    pub player: Option<String>,
    pub faction: Option<String>,
    pub winner: Option<String>,
    pub date_from: Option<String>,
    pub date_to: Option<String>,
}

impl VaultQuery {
    /// Parses a search string supporting both free-text terms and inline qualifiers:
    /// `map:<X>`, `player:<X>`, `faction:<X>`, `winner:<X>`, `from:YYYY-MM-DD`, `to:YYYY-MM-DD`.
    pub fn parse(raw: &str) -> Self {
        let mut q = Self::default();
        for tok in raw.split_whitespace() {
            if let Some((k, v)) = tok.split_once(':') {
                let v = v.trim();
                if v.is_empty() {
                    continue;
                }
                match k.to_ascii_lowercase().as_str() {
                    "map" => q.map = Some(v.to_string()),
                    "player" | "army" => q.player = Some(v.to_string()),
                    "faction" | "fac" => q.faction = Some(v.to_string()),
                    "winner" | "win" => q.winner = Some(v.to_string()),
                    "from" | "after" | "since" => q.date_from = Some(v.to_string()),
                    "to" | "before" | "until" => q.date_to = Some(v.to_string()),
                    _ => q.text.push(tok.to_ascii_lowercase()),
                }
            } else {
                q.text.push(tok.to_ascii_lowercase());
            }
        }
        q
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
            && self.map.is_none()
            && self.player.is_none()
            && self.faction.is_none()
            && self.winner.is_none()
            && self.date_from.is_none()
            && self.date_to.is_none()
    }

    pub fn matches(&self, e: &VaultEntry) -> bool {
        if let Some(ref m) = self.map {
            let ml = m.to_ascii_lowercase();
            if !e.map_id.to_ascii_lowercase().contains(&ml)
                && !e.map_name.to_ascii_lowercase().contains(&ml)
            {
                return false;
            }
        }
        if let Some(ref p) = self.player {
            let pl = p.to_ascii_lowercase();
            if !e.armies.iter().any(|a| a.name.to_ascii_lowercase().contains(&pl)) {
                return false;
            }
        }
        if let Some(ref f) = self.faction {
            let fl = f.to_ascii_lowercase();
            if !e
                .armies
                .iter()
                .any(|a| sim_faction_label(a.faction).to_ascii_lowercase().contains(&fl))
            {
                return false;
            }
        }
        if let Some(ref w) = self.winner {
            let wl = w.to_ascii_lowercase();
            let matched = match &e.winner {
                Some(VaultWinner::Draw) => "draw".contains(&wl),
                Some(VaultWinner::Army { name, faction, team, .. }) => {
                    name.to_ascii_lowercase().contains(&wl)
                        || sim_faction_label(*faction).to_ascii_lowercase().contains(&wl)
                        || format!("team{}", team + 1).contains(&wl)
                        || format!("t{}", team + 1) == wl
                }
                None => false,
            };
            if !matched {
                return false;
            }
        }
        if let Some(ref from) = self.date_from
            && e.date_ymd.as_str() < from.as_str()
        {
            return false;
        }
        if let Some(ref to) = self.date_to
            && e.date_ymd.as_str() > to.as_str()
        {
            return false;
        }
        for term in &self.text {
            let hit = e.file_name.to_ascii_lowercase().contains(term)
                || e.map_id.to_ascii_lowercase().contains(term)
                || e.map_name.to_ascii_lowercase().contains(term)
                || e.date_ymd.contains(term)
                || e.armies.iter().any(|a| {
                    a.name.to_ascii_lowercase().contains(term)
                        || sim_faction_label(a.faction).to_ascii_lowercase().contains(term)
                })
                || e.winner
                    .as_ref()
                    .is_some_and(|w| w.label().to_ascii_lowercase().contains(term))
                || e.mods.iter().any(|m| m.name.to_ascii_lowercase().contains(term))
                || e.mission_name
                    .as_ref()
                    .is_some_and(|m| m.to_ascii_lowercase().contains(term))
                || e.share_code
                    .as_ref()
                    .is_some_and(|c| c.to_ascii_lowercase().contains(term))
                || e.bookmarks
                    .iter()
                    .any(|b| b.label.to_ascii_lowercase().contains(term));
            if !hit {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Headless event-bookmark & winner collector
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ReplayAnalysis {
    pub winner: Option<VaultWinner>,
    pub bookmarks: Vec<Bookmark>,
    #[allow(dead_code)]
    pub verified_hash: u64,
}

/// Runs a fast headless re-sim of `replay` to extract the match outcome and notable event
/// bookmarks (ACU deaths, experimental completions, strategic nuke launches, kill bursts).
pub fn collect_replay_analysis(
    replay: &Replay,
    catalog: Catalog,
    map: &MapContext,
) -> Result<ReplayAnalysis, String> {
    if !map.map_id.eq_ignore_ascii_case(&replay.header.map_id) {
        return Err(format!(
            "replay map '{}' does not match loaded map '{}'",
            replay.header.map_id, map.map_id
        ));
    }
    crate::mods::check_match(&replay.header.mods, &catalog.mods, "this replay")?;
    replay.header.check_rules("this replay")?;

    let mut world = Skirmish::new(
        catalog,
        map.grid.clone(),
        map.size,
        map.mass_points.clone(),
        &replay.header.setups,
        replay.header.seed,
    )?;
    world.attach_map(map);
    replay.header.apply_mission(&mut world, map)?;
    world.set_options(replay.header.options);

    let army_info: Vec<(String, usize, u8)> = replay
        .header
        .setups
        .iter()
        .map(|s| (s.name.clone(), s.faction, s.team))
        .collect();

    let mut bookmarks: Vec<Bookmark> = Vec::new();
    let mut kill_bursts: Vec<(u32, Bookmark)> = Vec::new();
    let mut seen_experimentals: HashSet<UnitId> = HashSet::new();
    for u in world.units() {
        if u.role == Role::Experimental || u.tier >= 4 {
            seen_experimentals.insert(u.id);
        }
    }

    let mut prev_total_kills: u32 = 0;
    let mut recent_kills: VecDeque<(u64, u32, (i64, i64))> = VecDeque::new();
    let mut last_burst_tick: Option<u64> = None;

    let mut cmd_idx = 0usize;
    while world.tick() < replay.final_tick && world.result().is_none() {
        let cur_tick = world.tick();
        while cmd_idx < replay.commands.len() && replay.commands[cmd_idx].tick == cur_tick {
            let rc = &replay.commands[cmd_idx];
            world.issue(rc.army, rc.command.clone());
            cmd_idx += 1;
        }
        world.step();
        let tick = world.tick();

        let mut last_combat_pos: Option<(i64, i64)> = None;
        for ev in world.drain_events() {
            match ev {
                SimEvent::AcuDied { pos, army } => {
                    let who = army_info
                        .get(army as usize)
                        .map(|(n, f, _)| format!("{n} ({})", sim_faction_label(*f)))
                        .unwrap_or_else(|| format!("Army {army}"));
                    bookmarks.push(Bookmark {
                        tick,
                        kind: BookmarkKind::AcuDeath,
                        army: Some(army),
                        pos: (pos.0, pos.1),
                        label: format!("{who} ACU destroyed"),
                    });
                    last_combat_pos = Some((pos.0, pos.1));
                }
                SimEvent::MissileLaunched { pos, is_nuke: true } => {
                    bookmarks.push(Bookmark {
                        tick,
                        kind: BookmarkKind::NukeLaunch,
                        army: None,
                        pos: (pos.0, pos.1),
                        label: "Strategic nuke launched".to_string(),
                    });
                }
                SimEvent::Explosion { pos, .. } => {
                    last_combat_pos = Some((pos.0, pos.1));
                }
                _ => {}
            }
        }

        // Detect newly completed experimental units.
        for u in world.units() {
            if (u.role == Role::Experimental || u.tier >= 4) && seen_experimentals.insert(u.id) {
                let who = army_info
                    .get(u.army as usize)
                    .map(|(n, _, _)| n.as_str())
                    .unwrap_or("Army");
                let unit_name = world
                    .catalog()
                    .get(&u.type_id)
                    .map(|ut| ut.name.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or(u.type_id.as_str());
                bookmarks.push(Bookmark {
                    tick,
                    kind: BookmarkKind::Experimental,
                    army: Some(u.army),
                    pos: (u.pos.x, u.pos.y),
                    label: format!("{who} built {unit_name}"),
                });
            }
        }

        // Sliding-window kill-burst detection.
        let total_kills: u32 = world.armies().iter().map(|a| a.kills).sum();
        if total_kills > prev_total_kills {
            let delta = total_kills - prev_total_kills;
            prev_total_kills = total_kills;
            let pos = last_combat_pos.unwrap_or((map.size.0 / 2, map.size.1 / 2));
            recent_kills.push_back((tick, delta, pos));
        }
        while recent_kills
            .front()
            .is_some_and(|&(t, _, _)| tick.saturating_sub(t) > KILL_BURST_WINDOW_TICKS)
        {
            recent_kills.pop_front();
        }
        let window_kills: u32 = recent_kills.iter().map(|&(_, d, _)| d).sum();
        if window_kills >= KILL_BURST_MIN_KILLS
            && last_burst_tick.is_none_or(|lt| tick.saturating_sub(lt) >= KILL_BURST_COOLDOWN_TICKS)
        {
            let mut sx: i128 = 0;
            let mut sz: i128 = 0;
            let mut sw: i128 = 0;
            for &(_, d, (px, pz)) in &recent_kills {
                let w = d as i128;
                sx += px as i128 * w;
                sz += pz as i128 * w;
                sw += w;
            }
            let center = if sw > 0 {
                ((sx / sw) as i64, (sz / sw) as i64)
            } else {
                (map.size.0 / 2, map.size.1 / 2)
            };
            kill_bursts.push((
                window_kills,
                Bookmark {
                    tick,
                    kind: BookmarkKind::KillBurst,
                    army: None,
                    pos: center,
                    label: format!("Major clash ({window_kills} kills)"),
                },
            ));
            last_burst_tick = Some(tick);
            recent_kills.clear();
        }
    }

    // Keep the top deadliest kill bursts so major events (ACU, Nukes, Experimentals) stay prominent.
    kill_bursts.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.tick.cmp(&b.1.tick)));
    kill_bursts.truncate(MAX_KILL_BURST_BOOKMARKS);
    bookmarks.extend(kill_bursts.into_iter().map(|(_, bm)| bm));

    // Cap bookmarks while preserving high-priority events (ACU > Nuke > Experimental > Clash).
    if bookmarks.len() > MAX_BOOKMARKS {
        bookmarks.sort_by_key(|b| (b.kind.priority(), b.tick));
        bookmarks.truncate(MAX_BOOKMARKS);
    }
    bookmarks.sort_by_key(|b| (b.tick, b.kind.priority()));

    let winner = match world.result() {
        Some(GameResult::Victory(id)) => {
            let (name, faction, team) = army_info
                .get(id as usize)
                .cloned()
                .unwrap_or_else(|| (format!("Army {id}"), 0, id));
            Some(VaultWinner::Army { id, name, faction, team })
        }
        Some(GameResult::Draw) => Some(VaultWinner::Draw),
        None => None,
    };

    Ok(ReplayAnalysis {
        winner,
        bookmarks,
        verified_hash: world.state_hash(),
    })
}

// ---------------------------------------------------------------------------
// Thumbnail PNG rendering & I/O
// ---------------------------------------------------------------------------

/// Generates a `THUMB_SIZE x THUMB_SIZE` RGBA thumbnail combining the map preview (from
/// `.scmap` DDS or heightmap fallback) with colored army start markers.
pub fn render_thumbnail(
    game_dir: Option<&Path>,
    map_id: &str,
    setups: &[ArmySetup],
    winner_army: Option<ArmyId>,
    map_size_milli: Option<(i64, i64)>,
) -> RgbaImage {
    let mut base = game_dir
        .and_then(|gd| {
            let pv = lobby::load_preview(gd, map_id);
            pv.image
        })
        .map(|img| crate::dds_decode::resize_rgba_bilinear(&img, THUMB_SIZE, THUMB_SIZE))
        .unwrap_or_else(|| fallback_grid_thumbnail(THUMB_SIZE));

    // Determine map dimensions in world meters so army start coords normalize to 0..1.
    let (mw_m, mh_m) = if let Some((sx, sy)) = map_size_milli {
        ((sx as f32 / SCALE as f32).max(1.0), (sy as f32 / SCALE as f32).max(1.0))
    } else if let Some(gd) = game_dir
        && let Ok(files) = scenario::find_map(gd, map_id)
        && let Ok(meta) = scenario::parse_scenario_file(&files.scenario_path, &files.id)
    {
        ((meta.size.0 as f32).max(1.0), (meta.size.1 as f32).max(1.0))
    } else {
        let mut mx = 512.0f32;
        let mut my = 512.0f32;
        for s in setups {
            mx = mx.max((s.start.0 as f32 / SCALE as f32) * 1.15);
            my = my.max((s.start.1 as f32 / SCALE as f32) * 1.15);
        }
        (mx, my)
    };

    // Draw a subtle dark border frame around the thumbnail.
    draw_border_rgba(&mut base, [32, 46, 62, 255]);

    // Overlay start spots for each army.
    for (i, s) in setups.iter().enumerate() {
        let u = ((s.start.0 as f32 / SCALE as f32) / mw_m).clamp(0.04, 0.96);
        let v = ((s.start.1 as f32 / SCALE as f32) / mh_m).clamp(0.04, 0.96);
        let cx = (u * (base.width as f32 - 1.0)).round() as i32;
        let cy = (v * (base.height as f32 - 1.0)).round() as i32;
        let col = theme::SLOT_COLORS[i % theme::SLOT_COLORS.len()];
        let rgb = [
            (col[0] * 255.0).round() as u8,
            (col[1] * 255.0).round() as u8,
            (col[2] * 255.0).round() as u8,
            255,
        ];
        let is_winner = winner_army == Some(i as ArmyId);
        if is_winner {
            draw_disc_rgba(&mut base, cx, cy, 8, [250, 212, 72, 255]);
        }
        draw_disc_rgba(&mut base, cx, cy, 6, [12, 16, 22, 255]);
        draw_disc_rgba(&mut base, cx, cy, 4, rgb);
    }

    base
}

fn fallback_grid_thumbnail(size: u32) -> RgbaImage {
    let mut pixels = vec![0u8; (size * size * 4) as usize];
    for y in 0..size {
        for x in 0..size {
            let grid = x % 20 == 0 || y % 20 == 0;
            let idx = ((y * size + x) * 4) as usize;
            let (r, g, b) = if grid {
                (34, 52, 68)
            } else {
                let grad = ((x + y) * 18 / (size * 2).max(1)) as u8;
                (18 + grad, 28 + grad, 38 + grad)
            };
            pixels[idx] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
            pixels[idx + 3] = 255;
        }
    }
    RgbaImage { width: size, height: size, pixels }
}

fn draw_border_rgba(img: &mut RgbaImage, rgba: [u8; 4]) {
    let (w, h) = (img.width, img.height);
    for x in 0..w {
        for y in [0, h.saturating_sub(1)] {
            let idx = ((y * w + x) * 4) as usize;
            img.pixels[idx..idx + 4].copy_from_slice(&rgba);
        }
    }
    for y in 0..h {
        for x in [0, w.saturating_sub(1)] {
            let idx = ((y * w + x) * 4) as usize;
            img.pixels[idx..idx + 4].copy_from_slice(&rgba);
        }
    }
}

fn draw_disc_rgba(img: &mut RgbaImage, cx: i32, cy: i32, r: i32, rgba: [u8; 4]) {
    let r2 = r * r;
    for dy in -r..=r {
        let py = cy + dy;
        if py < 0 || py >= img.height as i32 {
            continue;
        }
        for dx in -r..=r {
            if dx * dx + dy * dy > r2 {
                continue;
            }
            let px = cx + dx;
            if px < 0 || px >= img.width as i32 {
                continue;
            }
            let idx = ((py as u32 * img.width + px as u32) * 4) as usize;
            img.pixels[idx..idx + 4].copy_from_slice(&rgba);
        }
    }
}

pub fn save_rgba_png(path: &Path, img: &RgbaImage) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    let tmp = path.with_extension("png.tmp");
    {
        let file = File::create(&tmp)
            .map_err(|e| format!("failed to create {}: {e}", tmp.display()))?;
        let mut enc = png::Encoder::new(BufWriter::new(file), img.width, img.height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc
            .write_header()
            .map_err(|e| format!("png header error: {e}"))?;
        writer
            .write_image_data(&img.pixels)
            .map_err(|e| format!("png write error: {e}"))?;
    }
    fs::rename(&tmp, path).map_err(|e| format!("failed to rename thumbnail: {e}"))
}

pub fn load_rgba_png(path: &Path) -> Result<RgbaImage, String> {
    let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let mut decoder = png::Decoder::new(file);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::ALPHA);
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("png read info {}: {e}", path.display()))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("png decode {}: {e}", path.display()))?;
    let raw = &buf[..info.buffer_size()];
    let pixels = match info.color_type {
        png::ColorType::Rgba => raw.to_vec(),
        png::ColorType::Rgb => {
            let mut out = Vec::with_capacity((info.width * info.height * 4) as usize);
            for &[r, g, b] in raw.as_chunks::<3>().0 {
                out.extend_from_slice(&[r, g, b, 255]);
            }
            out
        }
        png::ColorType::GrayscaleAlpha => {
            let mut out = Vec::with_capacity((info.width * info.height * 4) as usize);
            for &[g, a] in raw.as_chunks::<2>().0 {
                out.extend_from_slice(&[g, g, g, a]);
            }
            out
        }
        png::ColorType::Grayscale => {
            let mut out = Vec::with_capacity((info.width * info.height * 4) as usize);
            for &g in raw {
                out.extend_from_slice(&[g, g, g, 255]);
            }
            out
        }
        other => return Err(format!("unsupported png color type {other:?}")),
    };
    Ok(RgbaImage {
        width: info.width,
        height: info.height,
        pixels,
    })
}

// ---------------------------------------------------------------------------
// VaultIndex binary codec & incremental sync
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct SyncOptions {
    /// Run headless re-sim on unindexed compatible replays to extract winner & bookmarks.
    pub run_sim: bool,
    /// Generate PNG thumbnails in `<vault_dir>/thumbs/`.
    pub generate_thumbs: bool,
    /// Ignore cached mtime/size and re-index every replay file.
    pub force_rebuild: bool,
}

impl Default for SyncOptions {
    fn default() -> Self {
        Self {
            run_sim: true,
            generate_thumbs: true,
            force_rebuild: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SyncReport {
    pub total: usize,
    pub cached: usize,
    pub indexed: usize,
    pub sim_analyzed: usize,
    pub removed: usize,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VaultIndex {
    pub entries: Vec<VaultEntry>,
}

impl VaultIndex {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.entries.len() * 256 + 16);
        out.extend_from_slice(INDEX_MAGIC);
        out.extend_from_slice(&INDEX_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for e in &self.entries {
            write_str(&mut out, &e.file_name);
            out.extend_from_slice(&e.file_size.to_le_bytes());
            out.extend_from_slice(&e.mtime_secs.to_le_bytes());
            out.extend_from_slice(&e.mtime_nanos.to_le_bytes());
            out.extend_from_slice(&e.content_hash.to_le_bytes());
            write_str(&mut out, &e.map_id);
            write_str(&mut out, &e.map_name);
            write_str(&mut out, &e.date_ymd);
            out.extend_from_slice(&e.duration_ticks.to_le_bytes());
            out.extend_from_slice(&e.seed.to_le_bytes());
            out.push(e.victory.code());
            out.extend_from_slice(&e.unit_cap.to_le_bytes());
            out.extend_from_slice(&e.rules.to_le_bytes());

            out.extend_from_slice(&(e.armies.len() as u32).to_le_bytes());
            for a in &e.armies {
                write_str(&mut out, &a.name);
                out.extend_from_slice(&(a.faction as u32).to_le_bytes());
                out.push(a.team);
                out.extend_from_slice(&a.start.0.to_le_bytes());
                out.extend_from_slice(&a.start.1.to_le_bytes());
            }

            match &e.winner {
                None => out.push(0),
                Some(VaultWinner::Draw) => out.push(1),
                Some(VaultWinner::Army { id, name, faction, team }) => {
                    out.push(2);
                    out.push(*id);
                    write_str(&mut out, name);
                    out.extend_from_slice(&(*faction as u32).to_le_bytes());
                    out.push(*team);
                }
            }

            crate::mods::encode_identities(&mut out, &e.mods);
            match &e.mission_name {
                Some(m) => {
                    out.push(1);
                    write_str(&mut out, m);
                }
                None => out.push(0),
            }
            out.extend_from_slice(&e.commands.to_le_bytes());
            out.extend_from_slice(&e.checkpoints.to_le_bytes());
            out.extend_from_slice(&e.final_hash.to_le_bytes());

            out.extend_from_slice(&(e.bookmarks.len() as u32).to_le_bytes());
            for bm in &e.bookmarks {
                out.extend_from_slice(&bm.tick.to_le_bytes());
                out.push(bm.kind.code());
                match bm.army {
                    Some(a) => {
                        out.push(1);
                        out.push(a);
                    }
                    None => out.push(0),
                }
                out.extend_from_slice(&bm.pos.0.to_le_bytes());
                out.extend_from_slice(&bm.pos.1.to_le_bytes());
                write_str(&mut out, &bm.label);
            }

            out.push(u8::from(e.sim_indexed));
            match &e.thumb_file {
                Some(t) => {
                    out.push(1);
                    write_str(&mut out, t);
                }
                None => out.push(0),
            }
            match &e.share_code {
                Some(c) => {
                    out.push(1);
                    write_str(&mut out, c);
                }
                None => out.push(0),
            }
        }
        out
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 16 || &data[0..8] != INDEX_MAGIC {
            return Err("invalid vault index magic (expected FARVX1)".to_string());
        }
        let mut pos = 8;
        let ver = read_u32(data, &mut pos)?;
        if ver != INDEX_VERSION {
            return Err(format!("unsupported vault index version {ver}"));
        }
        let count = read_u32(data, &mut pos)? as usize;
        let mut entries = Vec::with_capacity(count.min(4096));
        for _ in 0..count {
            let file_name = read_str(data, &mut pos)?;
            let file_size = read_u64(data, &mut pos)?;
            let mtime_secs = read_u64(data, &mut pos)?;
            let mtime_nanos = read_u32(data, &mut pos)?;
            let content_hash = read_u64(data, &mut pos)?;
            let map_id = read_str(data, &mut pos)?;
            let map_name = read_str(data, &mut pos)?;
            let date_ymd = read_str(data, &mut pos)?;
            let duration_ticks = read_u64(data, &mut pos)?;
            let seed = read_u64(data, &mut pos)?;
            let vcode = read_u8(data, &mut pos)?;
            let victory = VictoryCondition::from_code(vcode)
                .ok_or_else(|| format!("unknown victory condition {vcode} in index"))?;
            let unit_cap = read_u32(data, &mut pos)?;
            let rules = read_u32(data, &mut pos)?;

            let n_armies = read_u32(data, &mut pos)? as usize;
            let mut armies = Vec::with_capacity(n_armies.min(16));
            for _ in 0..n_armies {
                armies.push(VaultArmy {
                    name: read_str(data, &mut pos)?,
                    faction: read_u32(data, &mut pos)? as usize,
                    team: read_u8(data, &mut pos)?,
                    start: (read_i64(data, &mut pos)?, read_i64(data, &mut pos)?),
                });
            }

            let winner = match read_u8(data, &mut pos)? {
                0 => None,
                1 => Some(VaultWinner::Draw),
                2 => Some(VaultWinner::Army {
                    id: read_u8(data, &mut pos)?,
                    name: read_str(data, &mut pos)?,
                    faction: read_u32(data, &mut pos)? as usize,
                    team: read_u8(data, &mut pos)?,
                }),
                other => return Err(format!("invalid winner tag {other} in index")),
            };

            let mods = crate::mods::decode_identities(data, &mut pos)?;
            let mission_name = if read_u8(data, &mut pos)? != 0 {
                Some(read_str(data, &mut pos)?)
            } else {
                None
            };
            let commands = read_u32(data, &mut pos)?;
            let checkpoints = read_u32(data, &mut pos)?;
            let final_hash = read_u64(data, &mut pos)?;

            let n_bms = read_u32(data, &mut pos)? as usize;
            let mut bookmarks = Vec::with_capacity(n_bms.min(MAX_BOOKMARKS));
            for _ in 0..n_bms {
                let tick = read_u64(data, &mut pos)?;
                let kc = read_u8(data, &mut pos)?;
                let kind = BookmarkKind::from_code(kc)
                    .ok_or_else(|| format!("invalid bookmark kind {kc} in index"))?;
                let army = if read_u8(data, &mut pos)? != 0 {
                    Some(read_u8(data, &mut pos)?)
                } else {
                    None
                };
                let px = read_i64(data, &mut pos)?;
                let pz = read_i64(data, &mut pos)?;
                let label = read_str(data, &mut pos)?;
                bookmarks.push(Bookmark {
                    tick,
                    kind,
                    army,
                    pos: (px, pz),
                    label,
                });
            }

            let sim_indexed = read_u8(data, &mut pos)? != 0;
            let thumb_file = if read_u8(data, &mut pos)? != 0 {
                Some(read_str(data, &mut pos)?)
            } else {
                None
            };
            let share_code = if read_u8(data, &mut pos)? != 0 {
                Some(read_str(data, &mut pos)?)
            } else {
                None
            };

            entries.push(VaultEntry {
                file_name,
                file_size,
                mtime_secs,
                mtime_nanos,
                content_hash,
                map_id,
                map_name,
                date_ymd,
                duration_ticks,
                seed,
                victory,
                unit_cap,
                rules,
                armies,
                winner,
                mods,
                mission_name,
                commands,
                checkpoints,
                final_hash,
                bookmarks,
                sim_indexed,
                thumb_file,
                share_code,
            });
        }
        Ok(Self { entries })
    }

    pub fn load(vault_dir: &Path) -> Result<Self, String> {
        let p = vault_dir.join(INDEX_FILE);
        let bytes = fs::read(&p).map_err(|e| format!("failed to read {}: {e}", p.display()))?;
        Self::from_bytes(&bytes)
    }

    pub fn load_or_empty(vault_dir: &Path) -> Self {
        Self::load(vault_dir).unwrap_or_default()
    }

    pub fn save(&self, vault_dir: &Path) -> Result<(), String> {
        fs::create_dir_all(vault_dir)
            .map_err(|e| format!("failed to create {}: {e}", vault_dir.display()))?;
        let target = vault_dir.join(INDEX_FILE);
        let tmp = vault_dir.join(format!("{INDEX_FILE}.tmp"));
        fs::write(&tmp, self.to_bytes())
            .map_err(|e| format!("failed to write {}: {e}", tmp.display()))?;
        fs::rename(&tmp, &target)
            .map_err(|e| format!("failed to move index to {}: {e}", target.display()))
    }

    /// Incrementally updates the index for all `.farust-replay` files in `vault_dir`.
    pub fn sync(
        &mut self,
        vault_dir: &Path,
        game_dir: Option<&Path>,
        opts: SyncOptions,
    ) -> Result<SyncReport, String> {
        let mut report = SyncReport::default();
        if !vault_dir.is_dir() {
            report.removed = self.entries.len();
            self.entries.clear();
            return Ok(report);
        }

        let mut existing: BTreeMap<String, VaultEntry> = self
            .entries
            .drain(..)
            .map(|e| (e.file_name.clone(), e))
            .collect();

        let mut files: Vec<(String, PathBuf, u64, u64, u32)> = Vec::new();
        let rd = fs::read_dir(vault_dir)
            .map_err(|e| format!("failed to read {}: {e}", vault_dir.display()))?;
        for ent in rd.filter_map(Result::ok) {
            let path = ent.path();
            if path.extension().and_then(|e| e.to_str()) != Some(savegame::REPLAY_EXT) {
                continue;
            }
            let Some(file_name) = path.file_name().map(|s| s.to_string_lossy().to_string()) else {
                continue;
            };
            let Ok(meta) = fs::metadata(&path) else {
                continue;
            };
            let file_size = meta.len();
            let (mtime_secs, mtime_nanos) = meta
                .modified()
                .ok()
                .and_then(|m| m.duration_since(UNIX_EPOCH).ok())
                .map(|d| (d.as_secs(), d.subsec_nanos()))
                .unwrap_or((0, 0));
            files.push((file_name, path, file_size, mtime_secs, mtime_nanos));
        }

        let mut catalog_cache: Option<Catalog> = None;
        let mut map_cache: HashMap<String, MapContext> = HashMap::new();
        let mut map_name_cache: HashMap<String, String> = HashMap::new();
        let mut next_entries: Vec<VaultEntry> = Vec::with_capacity(files.len());

        for (file_name, path, file_size, mtime_secs, mtime_nanos) in files {
            let mut prev = existing.remove(&file_name);
            let unchanged = prev.as_ref().is_some_and(|p| {
                !opts.force_rebuild
                    && p.file_size == file_size
                    && p.mtime_secs == mtime_secs
                    && p.mtime_nanos == mtime_nanos
            });

            if unchanged {
                let mut entry = prev.take().unwrap();
                let need_sim = opts.run_sim
                    && !entry.sim_indexed
                    && !entry.needs_older_build()
                    && game_dir.is_some();
                let need_thumb = opts.generate_thumbs
                    && entry
                        .thumb_path_in(vault_dir)
                        .is_none_or(|tp| !tp.is_file());

                if !need_sim && !need_thumb {
                    report.cached += 1;
                    next_entries.push(entry);
                    continue;
                }

                // Upgrade cached entry with sim analysis and/or thumbnail without losing share_code.
                if let Ok(bytes) = fs::read(&path)
                    && let Ok(rep) = Replay::from_bytes(&bytes)
                {
                    if need_sim
                        && let Some(gd) = game_dir
                        && let Some(analysis) =
                            try_analyze_replay(&rep, gd, &mut catalog_cache, &mut map_cache)
                    {
                        entry.winner = analysis.winner;
                        entry.bookmarks = analysis.bookmarks;
                        entry.sim_indexed = true;
                        report.sim_analyzed += 1;
                    }
                    if need_thumb {
                        let rel = format!("{THUMBS_DIR}/{:016x}.png", entry.content_hash);
                        let map_sz = map_cache.get(&entry.map_id).map(|m| m.size);
                        let img = render_thumbnail(
                            game_dir,
                            &entry.map_id,
                            &rep.header.setups,
                            entry.winner.as_ref().and_then(VaultWinner::army_id),
                            map_sz,
                        );
                        if save_rgba_png(&vault_dir.join(&rel), &img).is_ok() {
                            entry.thumb_file = Some(rel);
                        }
                    }
                    report.indexed += 1;
                    next_entries.push(entry);
                    continue;
                }
            }

            let raw = match fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    report.errors.push(format!("{file_name}: {e}"));
                    continue;
                }
            };
            let rep = match Replay::from_bytes(&raw) {
                Ok(r) => r,
                Err(e) => {
                    report.errors.push(format!("{file_name}: {e}"));
                    continue;
                }
            };

            let content_hash = content_hash_bytes(&raw);
            let map_name = resolve_map_name(game_dir, &rep.header.map_id, &mut map_name_cache);
            let date_ymd = format_utc_date(mtime_secs);
            let mission_name = rep.header.mission.as_deref().map(parse_mission_title);

            let (mut winner, mut bookmarks, mut sim_indexed) = (None, Vec::new(), false);
            if opts.run_sim
                && rep.header.rules == RULES_VERSION
                && let Some(gd) = game_dir
                && let Some(analysis) =
                    try_analyze_replay(&rep, gd, &mut catalog_cache, &mut map_cache)
            {
                winner = analysis.winner;
                bookmarks = analysis.bookmarks;
                sim_indexed = true;
                report.sim_analyzed += 1;
            }

            let thumb_file = if opts.generate_thumbs {
                let rel = format!("{THUMBS_DIR}/{content_hash:016x}.png");
                let map_sz = map_cache.get(&rep.header.map_id).map(|m| m.size);
                let img = render_thumbnail(
                    game_dir,
                    &rep.header.map_id,
                    &rep.header.setups,
                    winner.as_ref().and_then(VaultWinner::army_id),
                    map_sz,
                );
                if save_rgba_png(&vault_dir.join(&rel), &img).is_ok() {
                    Some(rel)
                } else {
                    None
                }
            } else {
                prev.as_ref().and_then(|p| p.thumb_file.clone())
            };

            let share_code = prev
                .as_ref()
                .filter(|p| p.content_hash == content_hash)
                .and_then(|p| p.share_code.clone());

            let armies = rep
                .header
                .setups
                .iter()
                .map(|s| VaultArmy {
                    name: s.name.clone(),
                    faction: s.faction,
                    team: s.team,
                    start: s.start,
                })
                .collect();

            next_entries.push(VaultEntry {
                file_name,
                file_size,
                mtime_secs,
                mtime_nanos,
                content_hash,
                map_id: rep.header.map_id.clone(),
                map_name,
                date_ymd,
                duration_ticks: rep.final_tick,
                seed: rep.header.seed,
                victory: rep.header.options.victory,
                unit_cap: rep.header.options.unit_cap,
                rules: rep.header.rules,
                armies,
                winner,
                mods: rep.header.mods.clone(),
                mission_name,
                commands: rep.commands.len() as u32,
                checkpoints: rep.checkpoints.len() as u32,
                final_hash: rep.final_hash,
                bookmarks,
                sim_indexed,
                thumb_file,
                share_code,
            });
            report.indexed += 1;
        }

        report.removed = existing.len();
        next_entries.sort_by(|a, b| {
            b.mtime_secs
                .cmp(&a.mtime_secs)
                .then_with(|| b.file_name.cmp(&a.file_name))
        });
        report.total = next_entries.len();
        self.entries = next_entries;
        self.save(vault_dir)?;
        Ok(report)
    }

    pub fn search(&self, query: &VaultQuery) -> Vec<&VaultEntry> {
        self.entries.iter().filter(|e| query.matches(e)).collect()
    }

    pub fn set_share_code(&mut self, file_name: &str, code: &str) -> bool {
        if let Some(e) = self.entries.iter_mut().find(|e| e.file_name == file_name) {
            e.share_code = Some(code.to_string());
            true
        } else {
            false
        }
    }

    /// Looks up cached bookmarks for a replay file from the vault index (if indexed).
    pub fn bookmarks_for_path(vault_dir: &Path, replay_path: &Path) -> Vec<Bookmark> {
        let Some(fname) = replay_path.file_name().map(|s| s.to_string_lossy().to_string()) else {
            return Vec::new();
        };
        let idx = Self::load_or_empty(vault_dir);
        if let Some(e) = idx.entries.iter().find(|e| e.file_name == fname) {
            return e.bookmarks.clone();
        }
        // Fallback: match by content hash if file was loaded from another path.
        if let Ok(bytes) = fs::read(replay_path) {
            let h = content_hash_bytes(&bytes);
            if let Some(e) = idx.entries.iter().find(|e| e.content_hash == h) {
                return e.bookmarks.clone();
            }
        }
        Vec::new()
    }
}

fn resolve_map_name(
    game_dir: Option<&Path>,
    map_id: &str,
    cache: &mut HashMap<String, String>,
) -> String {
    if let Some(name) = cache.get(map_id) {
        return name.clone();
    }
    let name = game_dir
        .and_then(|gd| scenario::find_map(gd, map_id).ok())
        .and_then(|files| scenario::parse_scenario_file(&files.scenario_path, &files.id).ok())
        .map(|m| m.name)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| map_id.to_string());
    cache.insert(map_id.to_string(), name.clone());
    name
}

fn try_analyze_replay(
    rep: &Replay,
    game_dir: &Path,
    catalog_cache: &mut Option<Catalog>,
    map_cache: &mut HashMap<String, MapContext>,
) -> Option<ReplayAnalysis> {
    if catalog_cache.is_none() {
        *catalog_cache = Catalog::load(game_dir).ok();
    }
    let catalog = catalog_cache.as_ref()?.clone();
    if !map_cache.contains_key(&rep.header.map_id) {
        let map = MapContext::load(game_dir, &rep.header.map_id).ok()?;
        map_cache.insert(rep.header.map_id.clone(), map);
    }
    let map = map_cache.get(&rep.header.map_id)?;
    collect_replay_analysis(rep, catalog, map).ok()
}

// ---------------------------------------------------------------------------
// Tests (game-data-independent)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pathfinding::PathGrid;
    use crate::replay::ReplayRecorder;
    use crate::scmap::{ScMap, WaterInfo};
    use crate::skirmish::{Command, GameOptions, UnitType, WeaponType};
    use crate::skirmish::tests::test_catalog;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TMP_SEQ: AtomicU64 = AtomicU64::new(1);

    fn temp_dir(label: &str) -> PathBuf {
        let n = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "farust_vault_test_{}_{}_{}",
            std::process::id(),
            label,
            n
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn dummy_map(id: &str) -> MapContext {
        MapContext {
            map_id: id.to_string(),
            map_name: format!("{id} Arena"),
            scmap: ScMap {
                version_major: 2,
                version_minor: 60,
                map_width: 256.0,
                map_height: 256.0,
                preview_dds: vec![],
                heightmap_width: 8,
                heightmap_height: 8,
                height_scale: 0.01,
                heightmap: vec![1000; 81],
                terrain_shader: String::new(),
                background_path: String::new(),
                sky_cubemap_path: String::new(),
                cubemaps: vec![],
                lighting: crate::scmap::ScMapLighting::default(),
                water: WaterInfo {
                    has_water: false,
                    elevation: 0.0,
                    elevation_deep: 0.0,
                    elevation_abyss: 0.0,
                },
                unknown1: 0,
                unknown2: 0,
                unknown3: 0,
                unknown4: 0,
                post_heightmap_byte: 0,
                shader_raw: Vec::new(),
                background_raw: Vec::new(),
                sky_raw: Vec::new(),
                cubemaps_raw: Vec::new(),
                tail_bytes: Vec::new(),
                terrain_textures: None,
            },
            grid: PathGrid::new(64, 64, 4 * SCALE),
            naval_grid: PathGrid::new(64, 64, 4 * SCALE),
            amphib_grid: PathGrid::new(64, 64, 4 * SCALE),
            size: (256 * SCALE, 256 * SCALE),
            starts: vec![
                ("ARMY_1".into(), (40 * SCALE, 40 * SCALE)),
                ("ARMY_2".into(), (200 * SCALE, 200 * SCALE)),
            ],
            mass_points: vec![(50 * SCALE, 40 * SCALE), (190 * SCALE, 200 * SCALE)],
            hydro_points: vec![],
            has_water: false,
            props: Vec::new(),
            markers: Default::default(),
            areas: Default::default(),
        }
    }

    fn sample_replay(map_id: &str, seed: u64, rules: u32, mission: Option<&str>) -> Replay {
        let setups = vec![
            ArmySetup {
                name: "Alice".into(),
                faction: 0,
                start: (40 * SCALE, 40 * SCALE),
                team: 0,
            },
            ArmySetup {
                name: "Bob".into(),
                faction: 2,
                start: (200 * SCALE, 200 * SCALE),
                team: 1,
            },
        ];
        let mut rec = ReplayRecorder::new(map_id, seed, setups);
        rec.set_options(GameOptions {
            victory: VictoryCondition::Assassination,
            unit_cap: 500,
        });
        rec.set_mission(mission.map(str::to_string));
        let mut r = rec.finish(1200, 0x1234_5678_9abc_def0);
        r.header.rules = rules;
        r
    }

    #[test]
    fn index_binary_roundtrip_preserves_all_fields() {
        let entry = VaultEntry {
            file_name: "match1.farust-replay".into(),
            file_size: 4096,
            mtime_secs: 1_760_000_000,
            mtime_nanos: 123_456,
            content_hash: 0xdead_beef_cafe_0001,
            map_id: "SCMP_009".into(),
            map_name: "Seton's Clutch".into(),
            date_ymd: "2025-10-09".into(),
            duration_ticks: 3600,
            seed: 42,
            victory: VictoryCondition::Assassination,
            unit_cap: 500,
            rules: RULES_VERSION,
            armies: vec![
                VaultArmy {
                    name: "Alice".into(),
                    faction: 0,
                    team: 0,
                    start: (40 * SCALE, 40 * SCALE),
                },
                VaultArmy {
                    name: "Bob".into(),
                    faction: 2,
                    team: 1,
                    start: (200 * SCALE, 200 * SCALE),
                },
            ],
            winner: Some(VaultWinner::Army {
                id: 0,
                name: "Alice".into(),
                faction: 0,
                team: 0,
            }),
            mods: vec![ModIdentity {
                uid: "mod-1".into(),
                name: "BalanceMod".into(),
                version: "1.2".into(),
                hash: 99,
            }],
            mission_name: Some("Hold the Line".into()),
            commands: 321,
            checkpoints: 12,
            final_hash: 0xfeed_face_0000_0002,
            bookmarks: vec![
                Bookmark {
                    tick: 1500,
                    kind: BookmarkKind::Experimental,
                    army: Some(0),
                    pos: (80 * SCALE, 90 * SCALE),
                    label: "Alice built Fatboy".into(),
                },
                Bookmark {
                    tick: 3590,
                    kind: BookmarkKind::AcuDeath,
                    army: Some(1),
                    pos: (200 * SCALE, 200 * SCALE),
                    label: "Bob (Cybran) ACU destroyed".into(),
                },
            ],
            sim_indexed: true,
            thumb_file: Some("thumbs/deadbeefcafe0001.png".into()),
            share_code: Some("ABCD-2345".into()),
        };

        let idx = VaultIndex { entries: vec![entry] };
        let bytes = idx.to_bytes();
        let back = VaultIndex::from_bytes(&bytes).unwrap();
        assert_eq!(back, idx);
        assert!(!back.entries[0].needs_older_build());
    }

    #[test]
    fn incremental_sync_caches_unchanged_and_reindexes_changes() {
        let dir = temp_dir("inc_sync");
        let r1 = sample_replay("SCMP_001", 1, RULES_VERSION, None);
        let r2 = sample_replay("SCMP_009", 2, 1, Some("mission \"Hold the Line\"\nmap SCMP_009\n"));
        r1.save(&dir.join("a.farust-replay")).unwrap();
        r2.save(&dir.join("b.farust-replay")).unwrap();

        let opts = SyncOptions {
            run_sim: false,
            generate_thumbs: true,
            force_rebuild: false,
        };
        let mut idx = VaultIndex::default();
        let rep1 = idx.sync(&dir, None, opts).unwrap();
        assert_eq!(rep1.total, 2);
        assert_eq!(rep1.indexed, 2);
        assert_eq!(rep1.cached, 0);

        // Older rules version flagged cleanly; mission title extracted; thumbs written.
        let eb = idx.entries.iter().find(|e| e.file_name == "b.farust-replay").unwrap();
        assert!(eb.needs_older_build());
        assert_eq!(eb.mission_name.as_deref(), Some("Hold the Line"));
        let thumb_p = eb.thumb_path_in(&dir).unwrap();
        assert!(thumb_p.is_file());
        let decoded = load_rgba_png(&thumb_p).unwrap();
        assert_eq!((decoded.width, decoded.height), (THUMB_SIZE, THUMB_SIZE));

        // Second sync without changes must hit cache for both entries.
        let mut idx2 = VaultIndex::load(&dir).unwrap();
        let rep2 = idx2.sync(&dir, None, opts).unwrap();
        assert_eq!(rep2.cached, 2);
        assert_eq!(rep2.indexed, 0);

        // Removing one file prunes it on sync.
        fs::remove_file(dir.join("a.farust-replay")).unwrap();
        let rep3 = idx2.sync(&dir, None, opts).unwrap();
        assert_eq!(rep3.total, 1);
        assert_eq!(rep3.removed, 1);
        assert_eq!(rep3.cached, 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn search_filters_map_player_faction_winner_and_date_range() {
        let mk = |name: &str, map: &str, date: &str, winner: Option<VaultWinner>, fac1: usize| VaultEntry {
            file_name: name.into(),
            file_size: 100,
            mtime_secs: 0,
            mtime_nanos: 0,
            content_hash: 1,
            map_id: map.into(),
            map_name: format!("{map} Arena"),
            date_ymd: date.into(),
            duration_ticks: 600,
            seed: 1,
            victory: VictoryCondition::Assassination,
            unit_cap: 500,
            rules: RULES_VERSION,
            armies: vec![
                VaultArmy { name: "CommanderA".into(), faction: 0, team: 0, start: (0, 0) },
                VaultArmy { name: "BotB".into(), faction: fac1, team: 1, start: (0, 0) },
            ],
            winner,
            mods: vec![],
            mission_name: None,
            commands: 10,
            checkpoints: 1,
            final_hash: 0,
            bookmarks: vec![],
            sim_indexed: true,
            thumb_file: None,
            share_code: None,
        };

        let idx = VaultIndex {
            entries: vec![
                mk(
                    "r1.farust-replay",
                    "SCMP_001",
                    "2026-01-15",
                    Some(VaultWinner::Army { id: 0, name: "CommanderA".into(), faction: 0, team: 0 }),
                    1,
                ),
                mk(
                    "r2.farust-replay",
                    "SCMP_009",
                    "2026-05-20",
                    Some(VaultWinner::Army { id: 1, name: "BotB".into(), faction: 2, team: 1 }),
                    2,
                ),
                mk("r3.farust-replay", "SCMP_009", "2026-11-01", Some(VaultWinner::Draw), 3),
            ],
        };

        let q = VaultQuery::parse("");
        assert!(q.is_empty());
        assert_eq!(idx.search(&VaultQuery::parse("map:scmp_009")).len(), 2);
        assert_eq!(idx.search(&VaultQuery::parse("faction:aeon")).len(), 1);
        assert_eq!(idx.search(&VaultQuery::parse("player:commandera winner:uef")).len(), 1);
        assert_eq!(idx.search(&VaultQuery::parse("winner:draw")).len(), 1);
        assert_eq!(
            idx.search(&VaultQuery::parse("from:2026-02-01 to:2026-10-01"))
                .iter()
                .map(|e| e.file_name.as_str())
                .collect::<Vec<_>>(),
            vec!["r2.farust-replay"]
        );
        assert_eq!(idx.search(&VaultQuery::parse("seraphim")).len(), 1);
    }

    #[test]
    fn headless_analysis_collects_acu_death_and_winner() {
        let mut cat = test_catalog();
        // Make commanders lethal and close so a recorded duel finishes quickly.
        let mut rc = cat.get("RC").unwrap().clone();
        rc.max_hp = 500 * SCALE;
        rc.weapons = vec![WeaponType {
            label: "Laser".into(),
            damage: 250 * SCALE,
            range: 40 * SCALE,
            reset: 5,
            splash_radius: 0,
            muzzle_velocity: 0,
            anti_air: false,
            anti_surface: true,
            turret_yaw_speed: 0,
            firing_tolerance: 0,
            beam: true,
            lead_target: false,
            emp: false,
            kamikaze: false,
            stun: None,
            turret_yaw_bone: None,
            turret_pitch_bone: None,
        }];
        cat.types.insert("RC".into(), rc.clone());
        let mut bc = rc;
        bc.id = "BC".into();
        bc.max_hp = 200 * SCALE;
        cat.types.insert("BC".into(), bc);

        // Also ensure an experimental type exists for completeness.
        let mut exp: UnitType = cat.get("RT").unwrap().clone();
        exp.id = "REXP".into();
        exp.name = "Fatboy".into();
        exp.role = Role::Experimental;
        exp.tier = 4;
        cat.types.insert("REXP".into(), exp);

        let map = dummy_map("SYNTH_01");
        let setups = vec![
            ArmySetup { name: "Red".into(), faction: 0, start: (100 * SCALE, 100 * SCALE), team: 0 },
            ArmySetup { name: "Blue".into(), faction: 1, start: (110 * SCALE, 100 * SCALE), team: 1 },
        ];
        let mut world = Skirmish::new(cat.clone(), map.grid.clone(), map.size, map.mass_points.clone(), &setups, 7).unwrap();
        world.attach_map(&map);
        let mut rec = ReplayRecorder::new("SYNTH_01", 7, setups);
        // Issue attack order from army 0's ACU onto army 1's ACU.
        let u0 = world.units().find(|u| u.army == 0).unwrap().id;
        let u1 = world.units().find(|u| u.army == 1).unwrap().id;
        let cmd = Command::Attack { units: vec![u0], target: u1, queue: false };
        rec.record(0, 0, &cmd);
        world.issue(0, cmd);
        for _ in 0..40 {
            if world.result().is_some() {
                break;
            }
            world.step();
            rec.maybe_checkpoint(world.tick(), world.state_hash());
        }
        let replay = rec.finish(world.tick(), world.state_hash());
        assert!(world.result().is_some(), "duel should finish");

        let analysis = collect_replay_analysis(&replay, cat, &map).unwrap();
        assert_eq!(analysis.verified_hash, replay.final_hash);
        assert!(
            matches!(analysis.winner, Some(VaultWinner::Army { id: 0, .. })),
            "expected army 0 victory, got {:?}",
            analysis.winner
        );
        assert!(
            analysis.bookmarks.iter().any(|b| b.kind == BookmarkKind::AcuDeath && b.army == Some(1)),
            "expected ACU death bookmark, got {:?}",
            analysis.bookmarks
        );
    }
}
