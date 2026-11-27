//! Phase 9 deterministic command-stream replays and checkpoint verification.
//!
//! Compact little-endian binary `.farust-replay` format storing initial setup,
//! every [`Command`] keyed by `(tick, army)`, periodic FNV-1a64 checkpoints,
//! and the final tick hash for bit-exact verification.
//!
//! Versions: v1 (no teams), v2 (+ team per army), v3 (Phase 18: + [`GameOptions`]
//! victory condition and unit cap after the army table), v4 (Phase 24: + enabled mod
//! identities after the options). All four are readable; v1-v3 mean "no mods".

use std::fs;
use std::path::Path;

use crate::mods::ModIdentity;
use crate::skirmish::{
    ArmyId, ArmySetup, Catalog, Command, GameOptions, MapContext, Skirmish, UnitId, VictoryCondition,
};

pub const REPLAY_MAGIC: &[u8; 8] = b"FARPL1\0\0";
pub const REPLAY_VERSION_V1: u32 = 1;
pub const REPLAY_VERSION_V2: u32 = 2;
pub const REPLAY_VERSION_V3: u32 = 3;
/// Phase 24: header carries the enabled mods' identities.
pub const REPLAY_VERSION: u32 = 4;
pub const CHECKPOINT_INTERVAL: u64 = 300;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayHeader {
    pub map_id: String,
    pub seed: u64,
    pub setups: Vec<ArmySetup>,
    /// Phase 18 game options (defaults when reading v1/v2 files).
    pub options: GameOptions,
    /// Phase 24: mods the match was played with (empty = stock content).
    pub mods: Vec<ModIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayCommand {
    pub tick: u64,
    pub army: ArmyId,
    pub command: Command,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashCheckpoint {
    pub tick: u64,
    pub hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replay {
    pub header: ReplayHeader,
    pub commands: Vec<ReplayCommand>,
    pub checkpoints: Vec<HashCheckpoint>,
    pub final_tick: u64,
    pub final_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyReport {
    pub map_id: String,
    pub ticks: u64,
    pub commands: usize,
    pub checkpoints_verified: usize,
    pub final_hash: u64,
}

pub struct ReplayRecorder {
    header: ReplayHeader,
    commands: Vec<ReplayCommand>,
    checkpoints: Vec<HashCheckpoint>,
}

impl ReplayRecorder {
    pub fn new(map_id: &str, seed: u64, setups: Vec<ArmySetup>) -> Self {
        Self {
            header: ReplayHeader {
                map_id: map_id.to_string(),
                seed,
                setups,
                options: GameOptions::default(),
                mods: crate::mods::active_identities(),
            },
            commands: Vec::new(),
            checkpoints: Vec::new(),
        }
    }

    /// Phase 18: record the game options the world was created with.
    pub fn set_options(&mut self, options: GameOptions) {
        self.header.options = options;
    }

    pub fn record(&mut self, tick: u64, army: ArmyId, command: &Command) {
        self.commands.push(ReplayCommand {
            tick,
            army,
            command: command.clone(),
        });
    }

    pub fn maybe_checkpoint(&mut self, tick: u64, hash: u64) {
        if tick > 0 && tick % CHECKPOINT_INTERVAL == 0 {
            self.checkpoints.push(HashCheckpoint { tick, hash });
        }
    }

    pub fn finish(self, final_tick: u64, final_hash: u64) -> Replay {
        Replay {
            header: self.header,
            commands: self.commands,
            checkpoints: self.checkpoints,
            final_tick,
            final_hash,
        }
    }
}

// ---------------------------------------------------------------------------
// Binary codec (shared with net.rs for lockstep Command serialization)
// ---------------------------------------------------------------------------

pub(crate) fn write_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

pub(crate) fn read_str(data: &[u8], pos: &mut usize) -> Result<String, String> {
    let len = read_u32(data, pos)? as usize;
    let end = pos.checked_add(len).filter(|&e| e <= data.len())
        .ok_or_else(|| "truncated string in replay/net stream".to_string())?;
    let s = std::str::from_utf8(&data[*pos..end])
        .map_err(|_| "invalid UTF-8 in replay/net stream".to_string())?
        .to_string();
    *pos = end;
    Ok(s)
}

pub(crate) fn read_u8(data: &[u8], pos: &mut usize) -> Result<u8, String> {
    let b = *data.get(*pos).ok_or_else(|| "unexpected EOF (u8)".to_string())?;
    *pos += 1;
    Ok(b)
}

pub(crate) fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    let end = pos.checked_add(4).filter(|&e| e <= data.len())
        .ok_or_else(|| "unexpected EOF (u32)".to_string())?;
    let v = u32::from_le_bytes(data[*pos..end].try_into().unwrap());
    *pos = end;
    Ok(v)
}

pub(crate) fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    let end = pos.checked_add(8).filter(|&e| e <= data.len())
        .ok_or_else(|| "unexpected EOF (u64)".to_string())?;
    let v = u64::from_le_bytes(data[*pos..end].try_into().unwrap());
    *pos = end;
    Ok(v)
}

pub(crate) fn read_i64(data: &[u8], pos: &mut usize) -> Result<i64, String> {
    let end = pos.checked_add(8).filter(|&e| e <= data.len())
        .ok_or_else(|| "unexpected EOF (i64)".to_string())?;
    let v = i64::from_le_bytes(data[*pos..end].try_into().unwrap());
    *pos = end;
    Ok(v)
}

fn write_ids(out: &mut Vec<u8>, ids: &[UnitId]) {
    out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    for &id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
}

fn read_ids(data: &[u8], pos: &mut usize) -> Result<Vec<UnitId>, String> {
    let n = read_u32(data, pos)? as usize;
    if *pos + n * 4 > data.len() {
        return Err("unit list truncated".to_string());
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(read_u32(data, pos)?);
    }
    Ok(out)
}

pub(crate) fn encode_command(out: &mut Vec<u8>, cmd: &Command) {
    match cmd {
        Command::Move { units, x, y, queue } => {
            out.push(0);
            write_ids(out, units);
            out.extend_from_slice(&x.to_le_bytes());
            out.extend_from_slice(&y.to_le_bytes());
            out.push(u8::from(*queue));
        }
        Command::AttackMove { units, x, y, queue } => {
            out.push(1);
            write_ids(out, units);
            out.extend_from_slice(&x.to_le_bytes());
            out.extend_from_slice(&y.to_le_bytes());
            out.push(u8::from(*queue));
        }
        Command::Attack { units, target, queue } => {
            out.push(2);
            write_ids(out, units);
            out.extend_from_slice(&target.to_le_bytes());
            out.push(u8::from(*queue));
        }
        Command::Stop { units } => {
            out.push(3);
            write_ids(out, units);
        }
        Command::Build { builder, unit_type, x, y, queue } => {
            out.push(4);
            out.extend_from_slice(&builder.to_le_bytes());
            write_str(out, unit_type);
            out.extend_from_slice(&x.to_le_bytes());
            out.extend_from_slice(&y.to_le_bytes());
            out.push(u8::from(*queue));
        }
        Command::Produce { factory, unit_type, count } => {
            out.push(5);
            out.extend_from_slice(&factory.to_le_bytes());
            write_str(out, unit_type);
            out.extend_from_slice(&count.to_le_bytes());
        }
        Command::Rally { factory, x, y } => {
            out.push(6);
            out.extend_from_slice(&factory.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
            out.extend_from_slice(&y.to_le_bytes());
        }
        Command::Reclaim { units, x, y, queue } => {
            out.push(7);
            write_ids(out, units);
            out.extend_from_slice(&x.to_le_bytes());
            out.extend_from_slice(&y.to_le_bytes());
            out.push(u8::from(*queue));
        }
        Command::Repair { units, target, queue } => {
            out.push(8);
            write_ids(out, units);
            out.extend_from_slice(&target.to_le_bytes());
            out.push(u8::from(*queue));
        }
        Command::Assist { units, target, queue } => {
            out.push(9);
            write_ids(out, units);
            out.extend_from_slice(&target.to_le_bytes());
            out.push(u8::from(*queue));
        }
        Command::Overcharge { unit, target } => {
            out.push(10);
            out.extend_from_slice(&unit.to_le_bytes());
            out.extend_from_slice(&target.to_le_bytes());
        }
        Command::Upgrade { unit, target_type } => {
            out.push(11);
            out.extend_from_slice(&unit.to_le_bytes());
            write_str(out, target_type);
        }
        Command::LaunchMissile { unit, x, y } => {
            out.push(12);
            out.extend_from_slice(&unit.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
            out.extend_from_slice(&y.to_le_bytes());
        }
        Command::FormationMove {
            units,
            x,
            y,
            facing_mrad,
            spacing,
            queue,
        } => {
            out.push(13);
            write_ids(out, units);
            out.extend_from_slice(&x.to_le_bytes());
            out.extend_from_slice(&y.to_le_bytes());
            out.extend_from_slice(&facing_mrad.to_le_bytes());
            out.extend_from_slice(&spacing.to_le_bytes());
            out.push(u8::from(*queue));
        }
        Command::Patrol { units, x, y, queue } => {
            out.push(14);
            write_ids(out, units);
            out.extend_from_slice(&x.to_le_bytes());
            out.extend_from_slice(&y.to_le_bytes());
            out.push(u8::from(*queue));
        }
        Command::ToggleShield { units, enabled } => {
            out.push(15);
            write_ids(out, units);
            out.push(u8::from(*enabled));
        }
        Command::Load { units, transport, queue } => {
            out.push(16);
            write_ids(out, units);
            out.extend_from_slice(&transport.to_le_bytes());
            out.push(u8::from(*queue));
        }
        Command::Unload { transports, x, y, queue } => {
            out.push(17);
            write_ids(out, transports);
            out.extend_from_slice(&x.to_le_bytes());
            out.extend_from_slice(&y.to_le_bytes());
            out.push(u8::from(*queue));
        }
        Command::Enhance { unit, name } => {
            out.push(18);
            out.extend_from_slice(&unit.to_le_bytes());
            write_str(out, name);
        }
        Command::SetRepeat { factories, enabled } => {
            out.push(19);
            write_ids(out, factories);
            out.push(u8::from(*enabled));
        }
        Command::CancelProduce { factory, unit_type, count } => {
            out.push(20);
            out.extend_from_slice(&factory.to_le_bytes());
            write_str(out, unit_type);
            out.extend_from_slice(&count.to_le_bytes());
        }
    }
}

pub(crate) fn decode_command(data: &[u8], pos: &mut usize) -> Result<Command, String> {
    match read_u8(data, pos)? {
        0 => Ok(Command::Move {
            units: read_ids(data, pos)?,
            x: read_i64(data, pos)?,
            y: read_i64(data, pos)?,
            queue: read_u8(data, pos)? != 0,
        }),
        1 => Ok(Command::AttackMove {
            units: read_ids(data, pos)?,
            x: read_i64(data, pos)?,
            y: read_i64(data, pos)?,
            queue: read_u8(data, pos)? != 0,
        }),
        2 => Ok(Command::Attack {
            units: read_ids(data, pos)?,
            target: read_u32(data, pos)?,
            queue: read_u8(data, pos)? != 0,
        }),
        3 => Ok(Command::Stop {
            units: read_ids(data, pos)?,
        }),
        4 => Ok(Command::Build {
            builder: read_u32(data, pos)?,
            unit_type: read_str(data, pos)?,
            x: read_i64(data, pos)?,
            y: read_i64(data, pos)?,
            queue: read_u8(data, pos)? != 0,
        }),
        5 => Ok(Command::Produce {
            factory: read_u32(data, pos)?,
            unit_type: read_str(data, pos)?,
            count: read_u32(data, pos)?,
        }),
        6 => Ok(Command::Rally {
            factory: read_u32(data, pos)?,
            x: read_i64(data, pos)?,
            y: read_i64(data, pos)?,
        }),
        7 => Ok(Command::Reclaim {
            units: read_ids(data, pos)?,
            x: read_i64(data, pos)?,
            y: read_i64(data, pos)?,
            queue: read_u8(data, pos)? != 0,
        }),
        8 => Ok(Command::Repair {
            units: read_ids(data, pos)?,
            target: read_u32(data, pos)?,
            queue: read_u8(data, pos)? != 0,
        }),
        9 => Ok(Command::Assist {
            units: read_ids(data, pos)?,
            target: read_u32(data, pos)?,
            queue: read_u8(data, pos)? != 0,
        }),
        10 => Ok(Command::Overcharge {
            unit: read_u32(data, pos)?,
            target: read_u32(data, pos)?,
        }),
        11 => Ok(Command::Upgrade {
            unit: read_u32(data, pos)?,
            target_type: read_str(data, pos)?,
        }),
        12 => Ok(Command::LaunchMissile {
            unit: read_u32(data, pos)?,
            x: read_i64(data, pos)?,
            y: read_i64(data, pos)?,
        }),
        13 => Ok(Command::FormationMove {
            units: read_ids(data, pos)?,
            x: read_i64(data, pos)?,
            y: read_i64(data, pos)?,
            facing_mrad: read_i64(data, pos)?,
            spacing: read_i64(data, pos)?,
            queue: read_u8(data, pos)? != 0,
        }),
        14 => Ok(Command::Patrol {
            units: read_ids(data, pos)?,
            x: read_i64(data, pos)?,
            y: read_i64(data, pos)?,
            queue: read_u8(data, pos)? != 0,
        }),
        15 => Ok(Command::ToggleShield {
            units: read_ids(data, pos)?,
            enabled: read_u8(data, pos)? != 0,
        }),
        16 => Ok(Command::Load {
            units: read_ids(data, pos)?,
            transport: read_u32(data, pos)?,
            queue: read_u8(data, pos)? != 0,
        }),
        17 => Ok(Command::Unload {
            transports: read_ids(data, pos)?,
            x: read_i64(data, pos)?,
            y: read_i64(data, pos)?,
            queue: read_u8(data, pos)? != 0,
        }),
        18 => Ok(Command::Enhance {
            unit: read_u32(data, pos)?,
            name: read_str(data, pos)?,
        }),
        19 => Ok(Command::SetRepeat {
            factories: read_ids(data, pos)?,
            enabled: read_u8(data, pos)? != 0,
        }),
        20 => Ok(Command::CancelProduce {
            factory: read_u32(data, pos)?,
            unit_type: read_str(data, pos)?,
            count: read_u32(data, pos)?,
        }),
        t => Err(format!("unknown command tag {t}")),
    }
}

impl Replay {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.commands.len() * 24 + 128);
        out.extend_from_slice(REPLAY_MAGIC);
        out.extend_from_slice(&REPLAY_VERSION.to_le_bytes());
        write_str(&mut out, &self.header.map_id);
        out.extend_from_slice(&self.header.seed.to_le_bytes());
        out.extend_from_slice(&(self.header.setups.len() as u32).to_le_bytes());
        for s in &self.header.setups {
            write_str(&mut out, &s.name);
            out.extend_from_slice(&(s.faction as u32).to_le_bytes());
            out.extend_from_slice(&s.start.0.to_le_bytes());
            out.extend_from_slice(&s.start.1.to_le_bytes());
            out.push(s.team);
        }
        out.push(self.header.options.victory.code());
        out.extend_from_slice(&self.header.options.unit_cap.to_le_bytes());
        crate::mods::encode_identities(&mut out, &self.header.mods);
        out.extend_from_slice(&(self.commands.len() as u32).to_le_bytes());
        for c in &self.commands {
            out.extend_from_slice(&c.tick.to_le_bytes());
            out.push(c.army);
            encode_command(&mut out, &c.command);
        }
        out.extend_from_slice(&(self.checkpoints.len() as u32).to_le_bytes());
        for cp in &self.checkpoints {
            out.extend_from_slice(&cp.tick.to_le_bytes());
            out.extend_from_slice(&cp.hash.to_le_bytes());
        }
        out.extend_from_slice(&self.final_tick.to_le_bytes());
        out.extend_from_slice(&self.final_hash.to_le_bytes());
        out
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 12 || &data[0..8] != REPLAY_MAGIC {
            return Err("invalid replay magic (expected FARPL1)".to_string());
        }
        let mut pos = 8;
        let ver = read_u32(data, &mut pos)?;
        if !(REPLAY_VERSION_V1..=REPLAY_VERSION).contains(&ver) {
            return Err(format!("unsupported replay version {ver}"));
        }
        let map_id = read_str(data, &mut pos)?;
        let seed = read_u64(data, &mut pos)?;
        let n_armies = read_u32(data, &mut pos)? as usize;
        if n_armies < 2 || n_armies > 8 {
            return Err(format!("invalid replay army count {n_armies}"));
        }
        let mut setups = Vec::with_capacity(n_armies);
        for i in 0..n_armies {
            let name = read_str(data, &mut pos)?;
            let faction = read_u32(data, &mut pos)? as usize;
            let sx = read_i64(data, &mut pos)?;
            let sy = read_i64(data, &mut pos)?;
            let team = if ver >= REPLAY_VERSION_V2 {
                read_u8(data, &mut pos)?
            } else {
                i as u8
            };
            setups.push(ArmySetup {
                name,
                faction,
                start: (sx, sy),
                team,
            });
        }
        let options = if ver >= REPLAY_VERSION_V3 {
            let code = read_u8(data, &mut pos)?;
            let victory = VictoryCondition::from_code(code)
                .ok_or_else(|| format!("unknown victory condition {code}"))?;
            GameOptions {
                victory,
                unit_cap: read_u32(data, &mut pos)?,
            }
        } else {
            GameOptions::default()
        };
        let mods = if ver >= REPLAY_VERSION {
            crate::mods::decode_identities(data, &mut pos)?
        } else {
            Vec::new()
        };
        let n_cmds = read_u32(data, &mut pos)? as usize;
        let mut commands = Vec::with_capacity(n_cmds.min(100_000));
        for _ in 0..n_cmds {
            let tick = read_u64(data, &mut pos)?;
            let army = read_u8(data, &mut pos)?;
            let command = decode_command(data, &mut pos)?;
            commands.push(ReplayCommand { tick, army, command });
        }
        let n_cps = read_u32(data, &mut pos)? as usize;
        let mut checkpoints = Vec::with_capacity(n_cps.min(10_000));
        for _ in 0..n_cps {
            let tick = read_u64(data, &mut pos)?;
            let hash = read_u64(data, &mut pos)?;
            checkpoints.push(HashCheckpoint { tick, hash });
        }
        let final_tick = read_u64(data, &mut pos)?;
        let final_hash = read_u64(data, &mut pos)?;
        Ok(Replay {
            header: ReplayHeader { map_id, seed, setups, options, mods },
            commands,
            checkpoints,
            final_tick,
            final_hash,
        })
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
        }
        fs::write(path, self.to_bytes())
            .map_err(|e| format!("failed to write {}: {e}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    /// Reconstructs `Skirmish` purely from header + command stream (NO AI!), steps to
    /// `final_tick`, and validates every hash checkpoint and the final state hash.
    pub fn verify(&self, catalog: Catalog, map: &MapContext) -> Result<VerifyReport, String> {
        if !map.map_id.eq_ignore_ascii_case(&self.header.map_id) {
            return Err(format!(
                "replay map '{}' does not match loaded map '{}'",
                self.header.map_id, map.map_id
            ));
        }
        crate::mods::check_match(&self.header.mods, &catalog.mods, "this replay")?;
        let mut world = Skirmish::new(
            catalog,
            map.grid.clone(),
            map.size,
            map.mass_points.clone(),
            &self.header.setups,
            self.header.seed,
        )?;
        world.set_domain_grids(
            map.naval_grid.clone(),
            map.amphib_grid.clone(),
            map.hydro_points.clone(),
        );
        world.set_options(self.header.options);

        let mut cmd_idx = 0usize;
        let mut cp_idx = 0usize;
        while world.tick() < self.final_tick && world.result().is_none() {
            let cur_tick = world.tick();
            while cmd_idx < self.commands.len() && self.commands[cmd_idx].tick == cur_tick {
                let rc = &self.commands[cmd_idx];
                world.issue(rc.army, rc.command.clone());
                cmd_idx += 1;
            }
            world.step();
            while cp_idx < self.checkpoints.len() && self.checkpoints[cp_idx].tick == world.tick() {
                let cp = &self.checkpoints[cp_idx];
                let got = world.state_hash();
                if got != cp.hash {
                    return Err(format!(
                        "replay checkpoint desync at tick {}: expected {:016x}, got {:016x}",
                        cp.tick, cp.hash, got
                    ));
                }
                cp_idx += 1;
            }
        }

        let got_final = world.state_hash();
        if world.tick() != self.final_tick || got_final != self.final_hash {
            return Err(format!(
                "replay final mismatch: expected tick={} hash={:016x}, got tick={} hash={:016x}",
                self.final_tick,
                self.final_hash,
                world.tick(),
                got_final
            ));
        }

        Ok(VerifyReport {
            map_id: self.header.map_id.clone(),
            ticks: world.tick(),
            commands: self.commands.len(),
            checkpoints_verified: cp_idx,
            final_hash: got_final,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pathfinding::PathGrid;
    use crate::scmap::{ScMap, WaterInfo};
    use crate::sim::SCALE;
    use crate::skirmish::tests::test_catalog;
    use crate::skirmish_ai::ScriptedAi;

    fn dummy_map_ctx() -> MapContext {
        MapContext {
            map_id: "TEST_MAP".into(),
            map_name: "Test".into(),
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
                    has_water: true,
                    elevation: 10.0,
                    elevation_deep: 8.0,
                    elevation_abyss: 2.0,
                },
                unknown1: 0, unknown2: 0, unknown3: 0, unknown4: 0, post_heightmap_byte: 0,
                shader_raw: Vec::new(), background_raw: Vec::new(), sky_raw: Vec::new(), cubemaps_raw: Vec::new(), tail_bytes: Vec::new(),
                terrain_textures: None,
            },
            grid: PathGrid::new(64, 64, 4 * SCALE),
            naval_grid: PathGrid::new(64, 64, 4 * SCALE),
            amphib_grid: PathGrid::new(64, 64, 4 * SCALE),
            size: (256 * SCALE, 256 * SCALE),
            starts: vec![
                ("ARMY_1".into(), (50 * SCALE, 50 * SCALE)),
                ("ARMY_2".into(), (200 * SCALE, 200 * SCALE)),
            ],
            mass_points: vec![(40 * SCALE, 40 * SCALE), (210 * SCALE, 210 * SCALE)],
            hydro_points: vec![],
            has_water: false,
        }
    }

    #[test]
    fn command_codec_roundtrips_all_variants() {
        let all = [
            Command::Move { units: vec![1, 2], x: 1000, y: 2000, queue: true },
            Command::AttackMove { units: vec![3], x: 3000, y: 4000, queue: false },
            Command::Attack { units: vec![4], target: 99, queue: false },
            Command::Stop { units: vec![5, 6] },
            Command::Build { builder: 1, unit_type: "UEB0101".into(), x: 5000, y: 6000, queue: true },
            Command::Produce { factory: 7, unit_type: "UEL0201".into(), count: 5 },
            Command::Rally { factory: 7, x: 7000, y: 8000 },
            Command::Reclaim { units: vec![8], x: 9000, y: 10000, queue: false },
            Command::Repair { units: vec![8], target: 1, queue: true },
            Command::Assist { units: vec![8], target: 7, queue: false },
            Command::Overcharge { unit: 1, target: 99 },
            Command::Upgrade { unit: 7, target_type: "UEB0201".into() },
            Command::LaunchMissile { unit: 9, x: 12000, y: 13000 },
            Command::FormationMove { units: vec![1, 2], x: 4000, y: 5000, facing_mrad: 785, spacing: 3000, queue: false },
            Command::Patrol { units: vec![3, 4], x: 6000, y: 7000, queue: true },
            Command::ToggleShield { units: vec![10], enabled: true },
            Command::CancelProduce { factory: 7, unit_type: "UEL0201".into(), count: 2 },
        ];
        let mut buf = Vec::new();
        for c in &all {
            encode_command(&mut buf, c);
        }
        let mut pos = 0;
        for c in &all {
            let dec = decode_command(&buf, &mut pos).expect("decode command");
            assert_eq!(&dec, c);
        }
        assert_eq!(pos, buf.len());
    }

    #[test]
    fn record_save_load_verify_and_detect_corruption() {
        let map = dummy_map_ctx();
        let setups = vec![
            ArmySetup { name: "ARMY_1".into(), faction: 0, start: map.starts[0].1, team: 0 },
            ArmySetup { name: "ARMY_2".into(), faction: 1, start: map.starts[1].1, team: 1 },
        ];
        let mut world = Skirmish::new(
            test_catalog(),
            map.grid.clone(),
            map.size,
            map.mass_points.clone(),
            &setups,
            42,
        )
        .unwrap();
        let mut ais = [ScriptedAi::new(0), ScriptedAi::new(1)];
        let mut rec = ReplayRecorder::new(&map.map_id, 42, setups);

        for _ in 0..950 {
            if world.result().is_some() {
                break;
            }
            for ai in &mut ais {
                for cmd in ai.decide(&world) {
                    rec.record(world.tick(), ai.army, &cmd);
                    world.issue(ai.army, cmd);
                }
            }
            world.step();
            rec.maybe_checkpoint(world.tick(), world.state_hash());
        }
        let replay = rec.finish(world.tick(), world.state_hash());
        let bytes = replay.to_bytes();
        let decoded = Replay::from_bytes(&bytes).expect("decode replay");

        let report = decoded.verify(test_catalog(), &map).expect("verify replay");
        assert_eq!(report.final_hash, replay.final_hash);
        assert!(report.checkpoints_verified >= 3);

        // Corrupt a checkpoint and confirm desync error is raised
        let mut bad = decoded.clone();
        bad.checkpoints[1].hash ^= 0xDEAD_BEEF;
        let err = bad.verify(test_catalog(), &map).expect_err("corrupt checkpoint must fail");
        assert!(err.contains("desync"));
    }
}
