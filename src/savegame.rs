//! Phase 21 match persistence: auto-recorded replays, save games and fast re-simulating loads.
//!
//! A *save* is a replay of the match so far plus a little session metadata:
//!
//! ```text
//! .farust-save  = "FARSAV1\0" | u32 version | u8 human_army | u8 flags | u32 speed
//!                 | u32 n_ai_events × (u64 tick, u8 army, u8 enabled) | str label
//!                 | u32 replay_len | FARPL1 replay (final_tick = save tick, final_hash = state hash)
//! ```
//!
//! The embedded replay goes through the regular `replay.rs` codec, so every [`Command`]
//! variant (including ones added later) round-trips. Loading rebuilds the tick-0 world from
//! the replay header and re-simulates the recorded command stream in time slices
//! ([`LoadJob`], same idea as the replay seek), verifies the state hash and then hands control
//! back to the player. AI opponents are *resumed*, not restarted: every AI that was active in
//! the original match (`ai_events`) runs as a shadow during the re-simulation — its decisions
//! are discarded because the recorded stream is authoritative, but its internal state evolves
//! exactly as it did originally, so the continued match is identical to never having saved.
//!
//! Plain `.farust-replay` files load too (every army was AI from tick 0, e.g. `skirmish
//! --record`); the player takes over army 0 at the replay's final tick.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::play::PlaySession;
use crate::replay::{
    CHECKPOINT_INTERVAL, REPLAY_MAGIC, Replay, ReplayRecorder, read_str, read_u32, read_u64, read_u8, write_str,
};
use crate::skirmish::{ArmyId, ArmySetup, Catalog, Command, MapContext, Skirmish};
use crate::skirmish_ai::ScriptedAi;

pub const SAVE_MAGIC: &[u8; 8] = b"FARSAV1\0";
pub const SAVE_VERSION: u32 = 1;
pub const SAVE_EXT: &str = "farust-save";
pub const REPLAY_EXT: &str = "farust-replay";
pub const QUICKSAVE_FILE: &str = "quicksave.farust-save";

const FLAG_SPECTATE: u8 = 1;
const FLAG_FOG: u8 = 2;
const FLAG_FROM_NET: u8 = 4;

// ---------------------------------------------------------------------------
// Output locations
// ---------------------------------------------------------------------------

/// Directory for `kind` (`saves`, `replays`, `desyncs`): `$FARUST_DATA_DIR/<kind>` if set,
/// else `fa-unpack/<kind>` when `./fa-unpack` exists, else `$XDG_DATA_HOME/farust/<kind>`
/// (`~/.local/share/farust/<kind>`).
pub fn data_dir(kind: &str) -> PathBuf {
    if let Some(d) = std::env::var_os("FARUST_DATA_DIR") {
        return PathBuf::from(d).join(kind);
    }
    let local = Path::new("fa-unpack");
    if local.is_dir() {
        return local.join(kind);
    }
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("farust").join(kind)
}

/// `--no-record` on the command line (or `FARUST_NO_RECORD` in the environment) disables
/// writing the automatic match replay. Saves still work (they use the in-memory record).
pub fn recording_disabled() -> bool {
    std::env::var_os("FARUST_NO_RECORD").is_some() || std::env::args().any(|a| a == "--no-record")
}

/// UTC wall-clock stamp `YYYYMMDD-HHMMSS` for file names.
pub fn utc_stamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    let rem = secs % 86_400;
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}", rem / 3600, rem / 60 % 60, rem % 60)
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

fn fmt_clock(tick: u64) -> String {
    let s = tick / crate::sim::TICK_HZ;
    format!("{:02}M{:02}S", s / 60, s % 60)
}

/// Setup list of `world` as it goes into replay headers.
pub fn setups_of(world: &Skirmish) -> Vec<ArmySetup> {
    world
        .armies()
        .iter()
        .map(|a| ArmySetup {
            name: a.name.clone(),
            faction: a.faction,
            start: a.start,
            team: a.team,
        })
        .collect()
}

/// Tick-0 world for a replay header on `map` (the same construction `replay verify` uses).
pub fn origin_world(catalog: Catalog, map: &MapContext, replay: &Replay) -> Result<Skirmish, String> {
    if !map.map_id.eq_ignore_ascii_case(&replay.header.map_id) {
        return Err(format!(
            "save is for map {} but {} is loaded",
            replay.header.map_id, map.map_id
        ));
    }
    crate::mods::check_match(&replay.header.mods, &catalog.mods, "this save")?;
    let mut world = Skirmish::new(
        catalog,
        map.grid.clone(),
        map.size,
        map.mass_points.clone(),
        &replay.header.setups,
        replay.header.seed,
    )?;
    world.set_domain_grids(
        map.naval_grid.clone(),
        map.amphib_grid.clone(),
        map.hydro_points.clone(),
    );
    Ok(world)
}

// ---------------------------------------------------------------------------
// In-memory match record
// ---------------------------------------------------------------------------

/// Every command that entered the simulation (human, AI, network), keyed by the tick it was
/// applied on, plus periodic hash checkpoints and AI-control changes.
#[derive(Debug, Clone, Default)]
pub struct MatchRecorder {
    pub commands: Vec<(u64, ArmyId, Command)>,
    pub checkpoints: Vec<(u64, u64)>,
    /// `(tick, army, enabled)`: initial AIs at tick 0, `F1` toggles, network takeovers.
    pub ai_events: Vec<(u64, ArmyId, bool)>,
    /// Set for networked matches (other humans become AI when the save is loaded).
    pub from_net: bool,
    /// Where the automatic replay is written (fixed on the first write).
    pub replay_path: Option<PathBuf>,
    /// `(tick, commands)` at the last automatic replay write (avoids rewriting unchanged files).
    pub written: Option<(u64, usize)>,
}

impl MatchRecorder {
    /// Records everything queued for the next step (`world.pending_view()`, issue order) as
    /// applied on tick `world.tick()`. Returns the per-army command counts.
    pub fn record_pending(&mut self, world: &Skirmish) -> Vec<(ArmyId, u32)> {
        let tick = world.tick();
        let mut counts: Vec<(ArmyId, u32)> = Vec::new();
        for (army, cmd) in world.pending_view() {
            self.commands.push((tick, *army, cmd.clone()));
            bump(&mut counts, *army);
        }
        counts
    }

    /// Hash checkpoint every [`CHECKPOINT_INTERVAL`] ticks (call after each step).
    pub fn after_step(&mut self, world: &Skirmish) {
        let t = world.tick();
        if t > 0 && t % CHECKPOINT_INTERVAL == 0 && self.checkpoints.last().is_none_or(|c| c.0 < t) {
            self.checkpoints.push((t, world.state_hash()));
        }
    }

    pub fn ai_event(&mut self, tick: u64, army: ArmyId, enabled: bool) {
        self.ai_events.push((tick, army, enabled));
    }

    /// Replay of the match up to the current state of `world` (generic `replay.rs` codec).
    pub fn to_replay(&self, map_id: &str, world: &Skirmish) -> Replay {
        let mut rec = ReplayRecorder::new(map_id, world.seed(), setups_of(world));
        let now = world.tick();
        for (tick, army, cmd) in self.commands.iter().filter(|c| c.0 < now) {
            rec.record(*tick, *army, cmd);
        }
        for &(tick, hash) in self.checkpoints.iter().filter(|c| c.0 <= now) {
            rec.maybe_checkpoint(tick, hash);
        }
        rec.finish(now, world.state_hash())
    }
}

pub(crate) fn bump(counts: &mut Vec<(ArmyId, u32)>, army: ArmyId) {
    match counts.iter_mut().find(|(a, _)| *a == army) {
        Some((_, n)) => *n += 1,
        None => counts.push((army, 1)),
    }
}

// ---------------------------------------------------------------------------
// Save file
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SaveGame {
    /// Commands up to the save tick; `final_tick` = save tick, `final_hash` = state hash.
    pub replay: Replay,
    pub human_army: ArmyId,
    pub spectate: bool,
    pub fog_of_war: bool,
    pub from_net: bool,
    pub speed: u32,
    pub ai_events: Vec<(u64, ArmyId, bool)>,
    pub label: String,
}

impl SaveGame {
    pub fn tick(&self) -> u64 {
        self.replay.final_tick
    }

    pub fn hash(&self) -> u64 {
        self.replay.final_hash
    }

    pub fn map_id(&self) -> &str {
        &self.replay.header.map_id
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let rep = self.replay.to_bytes();
        let mut out = Vec::with_capacity(rep.len() + 64 + self.ai_events.len() * 10);
        out.extend_from_slice(SAVE_MAGIC);
        out.extend_from_slice(&SAVE_VERSION.to_le_bytes());
        out.push(self.human_army);
        let mut flags = 0u8;
        if self.spectate {
            flags |= FLAG_SPECTATE;
        }
        if self.fog_of_war {
            flags |= FLAG_FOG;
        }
        if self.from_net {
            flags |= FLAG_FROM_NET;
        }
        out.push(flags);
        out.extend_from_slice(&self.speed.to_le_bytes());
        out.extend_from_slice(&(self.ai_events.len() as u32).to_le_bytes());
        for &(tick, army, on) in &self.ai_events {
            out.extend_from_slice(&tick.to_le_bytes());
            out.push(army);
            out.push(u8::from(on));
        }
        write_str(&mut out, &self.label);
        out.extend_from_slice(&(rep.len() as u32).to_le_bytes());
        out.extend_from_slice(&rep);
        out
    }

    /// Decodes a `.farust-save`, or wraps a plain `FARPL1` replay as a save at its final tick.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() >= 8 && &data[0..8] == REPLAY_MAGIC {
            let replay = Replay::from_bytes(data)?;
            // `skirmish --record` style: every army was AI from tick 0; take over army 0.
            let ai_events = (0..replay.header.setups.len() as ArmyId).map(|a| (0, a, true)).collect();
            let label = format!("REPLAY {} {}", replay.header.map_id, fmt_clock(replay.final_tick));
            return Ok(Self {
                replay,
                human_army: 0,
                spectate: false,
                fog_of_war: false,
                from_net: false,
                speed: 1,
                ai_events,
                label,
            });
        }
        if data.len() < 12 || &data[0..8] != SAVE_MAGIC {
            return Err("not a FARust save (expected FARSAV1 or FARPL1 magic)".to_string());
        }
        let mut pos = 8;
        let ver = read_u32(data, &mut pos)?;
        if ver != SAVE_VERSION {
            return Err(format!("unsupported save version {ver}"));
        }
        let human_army = read_u8(data, &mut pos)?;
        let flags = read_u8(data, &mut pos)?;
        let speed = read_u32(data, &mut pos)?.clamp(1, 8);
        let n = read_u32(data, &mut pos)? as usize;
        if n > 100_000 {
            return Err(format!("save lists {n} AI events"));
        }
        let mut ai_events = Vec::with_capacity(n);
        for _ in 0..n {
            let tick = read_u64(data, &mut pos)?;
            let army = read_u8(data, &mut pos)?;
            let on = read_u8(data, &mut pos)? != 0;
            ai_events.push((tick, army, on));
        }
        let label = read_str(data, &mut pos)?;
        let len = read_u32(data, &mut pos)? as usize;
        let end = pos
            .checked_add(len)
            .filter(|&e| e == data.len())
            .ok_or_else(|| "save replay payload truncated or followed by garbage".to_string())?;
        let replay = Replay::from_bytes(&data[pos..end])?;
        if usize::from(human_army) >= replay.header.setups.len() {
            return Err(format!("save human army {human_army} out of range"));
        }
        Ok(Self {
            replay,
            human_army,
            spectate: flags & FLAG_SPECTATE != 0,
            fog_of_war: flags & FLAG_FOG != 0,
            from_net: flags & FLAG_FROM_NET != 0,
            speed,
            ai_events,
            label,
        })
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
        }
        std::fs::write(path, self.to_bytes()).map_err(|e| format!("failed to write {}: {e}", path.display()))
    }

    pub fn read(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    /// Builds the (still loading) session: tick-0 `origin` world + a [`LoadJob`] that
    /// re-simulates to the save tick. Pump it with [`PlaySession::pump_load`].
    pub fn into_session(self, origin: Skirmish) -> Result<PlaySession, String> {
        if origin.tick() != 0 {
            return Err("load origin must be a tick-0 world".to_string());
        }
        if origin.seed() != self.replay.header.seed || origin.armies().len() != self.replay.header.setups.len() {
            return Err("load origin does not match the save header".to_string());
        }
        let mut s = PlaySession::new(origin, self.human_army, self.spectate, self.speed);
        s.ais.clear();
        s.fog_of_war = self.fog_of_war;
        s.selected.clear();
        let mut rec = MatchRecorder {
            commands: self
                .replay
                .commands
                .iter()
                .filter(|c| c.tick < self.replay.final_tick)
                .map(|c| (c.tick, c.army, c.command.clone()))
                .collect(),
            checkpoints: self.replay.checkpoints.iter().map(|c| (c.tick, c.hash)).collect(),
            ai_events: self.ai_events.clone(),
            from_net: self.from_net,
            replay_path: None,
            written: None,
        };
        rec.checkpoints.retain(|c| c.0 <= self.replay.final_tick);
        s.status_banner = format!("LOADING {}", self.label);
        s.load = Some(Box::new(LoadJob::new(&self, rec.commands.clone())));
        s.recorder = Some(rec);
        Ok(s)
    }
}

/// One entry of the in-game "Load game" list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaveEntry {
    pub path: PathBuf,
    pub label: String,
    pub map_id: String,
    pub tick: u64,
    pub modified: u64,
}

/// Saves in [`data_dir`]`("saves")` for `map_id` (all maps when `None`), newest first.
pub fn list_saves(map_id: Option<&str>, limit: usize) -> Vec<SaveEntry> {
    let dir = data_dir("saves");
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<SaveEntry> = rd
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == SAVE_EXT))
        .filter_map(|path| {
            let modified = std::fs::metadata(&path)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let save = SaveGame::read(&path).ok()?;
            if map_id.is_some_and(|m| !m.eq_ignore_ascii_case(save.map_id())) {
                return None;
            }
            let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            Some(SaveEntry {
                label: format!("{} {}", name, fmt_clock(save.tick())),
                map_id: save.map_id().to_string(),
                tick: save.tick(),
                modified,
                path,
            })
        })
        .collect();
    out.sort_by(|a, b| b.modified.cmp(&a.modified).then_with(|| a.path.cmp(&b.path)));
    out.truncate(limit);
    out
}

// ---------------------------------------------------------------------------
// Load job (time-sliced deterministic re-simulation)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LoadJob {
    pub target_tick: u64,
    pub expected_hash: u64,
    commands: Vec<(u64, ArmyId, Command)>,
    cursor: usize,
    ai_events: Vec<(u64, ArmyId, bool)>,
    ai_cursor: usize,
    /// AIs re-living the match (decisions discarded; state resumes after the load).
    pub shadows: Vec<ScriptedAi>,
    pub human_army: ArmyId,
    pub spectate: bool,
    pub from_net: bool,
    pub started: Instant,
}

/// Outcome of a finished load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadResult {
    pub tick: u64,
    pub hash: u64,
    pub expected_hash: u64,
    pub millis: u64,
}

impl LoadResult {
    pub fn verified(&self) -> bool {
        self.hash == self.expected_hash
    }
}

impl LoadJob {
    fn new(save: &SaveGame, commands: Vec<(u64, ArmyId, Command)>) -> Self {
        let mut ai_events = save.ai_events.clone();
        // Stable by tick: same-tick toggles keep their recorded order.
        ai_events.sort_by_key(|e| e.0);
        Self {
            target_tick: save.tick(),
            expected_hash: save.hash(),
            commands,
            cursor: 0,
            ai_events,
            ai_cursor: 0,
            shadows: Vec::new(),
            human_army: save.human_army,
            spectate: save.spectate,
            from_net: save.from_net,
            started: Instant::now(),
        }
    }

    fn apply_ai_events(&mut self, upto: u64) {
        while let Some(&(tick, army, on)) = self.ai_events.get(self.ai_cursor) {
            if tick > upto {
                break;
            }
            self.ai_cursor += 1;
            if on {
                if !self.shadows.iter().any(|a| a.army == army) {
                    self.shadows.push(ScriptedAi::new(army));
                    self.shadows.sort_by_key(|a| a.army);
                }
            } else {
                self.shadows.retain(|a| a.army != army);
            }
        }
    }

    pub fn done(&self, world: &Skirmish) -> bool {
        world.tick() >= self.target_tick || world.result().is_some()
    }

    /// One re-simulated tick. Returns the per-army command counts (for the score sampler).
    pub fn step(&mut self, world: &mut Skirmish) -> Vec<(ArmyId, u32)> {
        let t = world.tick();
        self.apply_ai_events(t);
        for ai in &mut self.shadows {
            // State-only: the recorded stream below is authoritative.
            let _ = ai.decide(world);
        }
        while self.cursor < self.commands.len() && self.commands[self.cursor].0 < t {
            self.cursor += 1;
        }
        let mut counts = Vec::new();
        while self.cursor < self.commands.len() && self.commands[self.cursor].0 == t {
            let (_, army, ref cmd) = self.commands[self.cursor];
            world.issue(army, cmd.clone());
            bump(&mut counts, army);
            self.cursor += 1;
        }
        world.step();
        counts
    }

    /// The AI roster to resume with once the world reached the save tick.
    pub fn resumed_ais(&mut self, world: &Skirmish) -> Vec<ScriptedAi> {
        self.apply_ai_events(world.tick());
        let mut ais = std::mem::take(&mut self.shadows);
        if !self.spectate {
            ais.retain(|a| a.army != self.human_army);
        }
        if self.from_net {
            // Other humans of a networked match continue as fresh AI opponents.
            for a in world.armies() {
                if a.id != self.human_army && !a.defeated && !ais.iter().any(|x| x.army == a.id) {
                    ais.push(ScriptedAi::new(a.id));
                }
            }
        }
        ais.sort_by_key(|a| a.army);
        ais
    }
}

// ---------------------------------------------------------------------------
// PlaySession integration
// ---------------------------------------------------------------------------

impl PlaySession {
    pub fn loading(&self) -> bool {
        self.load.is_some()
    }

    /// `(current_tick, target_tick)` while a save is being loaded.
    pub fn load_progress(&self) -> Option<(u64, u64)> {
        self.load.as_ref().map(|j| (self.world.tick(), j.target_tick))
    }

    /// Advances a pending load within `budget`; returns `Some(result)` when it finished.
    pub fn pump_load(&mut self, budget: Duration) -> Option<LoadResult> {
        let mut job = self.load.take()?;
        let t0 = Instant::now();
        let mut n = 0u32;
        while !job.done(&self.world) {
            let counts = job.step(&mut self.world);
            self.stats.observe(&self.world, &counts);
            n += 1;
            if n % 32 == 0 && t0.elapsed() >= budget {
                break;
            }
        }
        let _ = self.world.drain_events();
        let _ = self.world.drain_audio_events();
        if !job.done(&self.world) {
            self.load = Some(job);
            return None;
        }
        self.ais = job.resumed_ais(&self.world);
        self.selected.clear();
        if !self.spectate
            && let Some(acu) = self.world.commander_of(self.human_army)
        {
            self.selected.push(acu.id);
        }
        let res = LoadResult {
            tick: self.world.tick(),
            hash: self.world.state_hash(),
            expected_hash: job.expected_hash,
            millis: job.started.elapsed().as_millis() as u64,
        };
        self.status_banner = if res.verified() {
            format!("LOADED {} - HASH OK", crate::controls::fmt_ticks(res.tick))
        } else {
            format!("LOADED {} - HASH MISMATCH {:016X}", crate::controls::fmt_ticks(res.tick), res.hash)
        };
        Some(res)
    }

    /// Runs a pending load to completion (headless / snapshot use).
    pub fn finish_load_blocking(&mut self) -> Option<LoadResult> {
        loop {
            self.load.as_ref()?;
            if let Some(r) = self.pump_load(Duration::from_secs(3600)) {
                return Some(r);
            }
        }
    }

    /// Snapshot of the match as a save (commands so far, current tick & hash).
    pub fn build_save(&self, map_id: &str, map_name: &str) -> Result<SaveGame, String> {
        if self.is_replay() {
            return Err("cannot save while watching a replay".to_string());
        }
        if self.loading() {
            return Err("cannot save while a load is in progress".to_string());
        }
        let rec = self
            .recorder
            .as_ref()
            .ok_or_else(|| "this session is not recorded".to_string())?;
        Ok(SaveGame {
            replay: rec.to_replay(map_id, &self.world),
            human_army: self.human_army,
            spectate: self.spectate,
            fog_of_war: self.fog_of_war,
            from_net: rec.from_net,
            speed: self.speed,
            ai_events: rec.ai_events.clone(),
            label: format!("{} {}", map_name.to_uppercase(), fmt_clock(self.world.tick())),
        })
    }

    /// Writes a save to `path`.
    pub fn save_game(&self, map_id: &str, map_name: &str, path: &Path) -> Result<SaveGame, String> {
        let save = self.build_save(map_id, map_name)?;
        save.write(path)?;
        Ok(save)
    }

    /// Writes (or refreshes) the automatic replay of this match. Returns the path when a
    /// file was written; `None` when recording is off, nothing happened, or it is unchanged.
    pub fn write_auto_replay(&mut self, map_id: &str) -> Result<Option<PathBuf>, String> {
        if !self.auto_record || self.is_replay() || self.loading() || self.world.tick() == 0 {
            return Ok(None);
        }
        let tick = self.world.tick();
        let Some(rec) = self.recorder.as_mut() else {
            return Ok(None);
        };
        if rec.written == Some((tick, rec.commands.len())) {
            return Ok(None);
        }
        let path = rec
            .replay_path
            .get_or_insert_with(|| {
                data_dir("replays").join(format!("{}-{}.{REPLAY_EXT}", utc_stamp(), map_id.to_lowercase()))
            })
            .clone();
        let replay = rec.to_replay(map_id, &self.world);
        replay.save(&path)?;
        rec.written = Some((tick, rec.commands.len()));
        Ok(Some(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skirmish::tests::test_world;

    #[test]
    fn civil_dates_and_stamp_shape() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        assert_eq!(civil_from_days(20_513), (2026, 3, 1));
        let s = utc_stamp();
        assert_eq!(s.len(), 15);
        assert_eq!(&s[8..9], "-");
    }

    fn run(s: &mut PlaySession, upto: u64) {
        while s.world.tick() < upto && s.world.result().is_none() {
            s.step_sim();
        }
    }

    /// Save at a mid-game tick, reload by re-simulation, verify the hash, then continue both
    /// the original and the loaded match: AIs resume with identical state, so hashes stay equal.
    #[test]
    fn save_load_reaches_identical_hash_and_resumes_ais() {
        let mut orig = PlaySession::new(test_world(4242), 0, true, 1);
        run(&mut orig, 420);
        // F1 toggles mid-game are part of the AI log.
        orig.toggle_spectate();
        run(&mut orig, 470);
        orig.toggle_spectate();
        run(&mut orig, 650);
        assert_eq!(orig.world.tick(), 650);
        let save = orig.build_save("TEST", "Test").unwrap();
        assert_eq!(save.tick(), 650);
        assert_eq!(save.hash(), orig.world.state_hash());
        assert!(!save.replay.commands.is_empty());

        let decoded = SaveGame::from_bytes(&save.to_bytes()).unwrap();
        assert_eq!(decoded.replay, save.replay);
        assert_eq!(decoded.ai_events, save.ai_events);
        assert!(decoded.spectate);

        let mut loaded = decoded.into_session(test_world(4242)).unwrap();
        assert!(loaded.loading());
        // Time-sliced like the GUI: tiny budgets still make progress.
        let mut slices = 0;
        let res = loop {
            slices += 1;
            if let Some(r) = loaded.pump_load(Duration::from_micros(1)) {
                break r;
            }
            assert!(loaded.load_progress().is_some_and(|(c, t)| c < t));
        };
        assert!(slices > 1, "load should take several slices");
        assert!(res.verified(), "{res:?}");
        assert_eq!(loaded.world.state_hash(), orig.world.state_hash());
        assert_eq!(loaded.ais, orig.ais, "AI state resumes exactly");
        assert_eq!(loaded.stats.totals.len(), orig.stats.totals.len());
        assert_eq!(loaded.stats.totals[0].mass_collected, orig.stats.totals[0].mass_collected);

        run(&mut orig, 1100);
        run(&mut loaded, 1100);
        assert_eq!(loaded.world.tick(), orig.world.tick());
        assert_eq!(loaded.world.state_hash(), orig.world.state_hash());

        // The re-saved match still verifies as a replay from tick 0.
        let again = loaded.build_save("TEST", "Test").unwrap();
        assert_eq!(again.hash(), orig.world.state_hash());
        assert_eq!(again.replay.commands.len(), orig.build_save("TEST", "Test").unwrap().replay.commands.len());
    }

    #[test]
    fn plain_replay_loads_and_hands_army_zero_to_the_player() {
        let mut orig = PlaySession::new(test_world(77), 0, true, 1);
        run(&mut orig, 300);
        let rep = orig.recorder.as_ref().unwrap().to_replay("TEST", &orig.world);
        let save = SaveGame::from_bytes(&rep.to_bytes()).unwrap();
        assert_eq!(save.tick(), 300);
        let mut s = save.into_session(test_world(77)).unwrap();
        let res = s.finish_load_blocking().unwrap();
        assert!(res.verified());
        assert!(!s.spectate);
        assert_eq!(s.ais.iter().map(|a| a.army).collect::<Vec<_>>(), vec![1]);
        assert_eq!(s.ais[0], orig.ais[1]);
    }

    #[test]
    fn corrupt_saves_are_rejected() {
        let mut orig = PlaySession::new(test_world(5), 0, true, 1);
        run(&mut orig, 40);
        let bytes = orig.build_save("TEST", "Test").unwrap().to_bytes();
        assert!(SaveGame::from_bytes(&bytes[..bytes.len() - 3]).is_err());
        let mut extra = bytes.clone();
        extra.push(1);
        assert!(SaveGame::from_bytes(&extra).is_err());
        assert!(SaveGame::from_bytes(b"NOTASAVE0000").is_err());
    }
}
