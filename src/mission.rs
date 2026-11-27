//! Phase 48: scenario scripting — Rust-native, data-driven missions.
//!
//! A mission is a small line-based text file (`*.farust-mission`, `#` comments):
//!
//! ```text
//! mission "Hold the Line"
//! map SCMP_009
//! victory sandbox
//! army 0 faction uef human
//! army 1 faction cybran ai none
//! objective hold primary "Defend the base for 10 minutes"
//! when at 0:00 do say "Commander, hold this position."; camera 120,120 zoom 90
//! when every 1:30 from 1:00 until 8:30 do spawn 1 URL0106 x4 at 400,400 attack 150,150
//! when at 10:00 do objective hold complete; win 0
//! when dead commander 0 do objective hold fail; lose 0
//! ```
//!
//! Conditions: `at T`, `every P from T [until T]`, `dead commander A`, `no units A`,
//! `count A TYPE|any in X,Z rR OP N` (OP: `< <= > >= ==`), `objective ID complete|failed`.
//! Actions: `say "TEXT"`, `spawn A TYPE xN at X,Z [attack X,Z | move X,Z]`,
//! `objective ID complete|fail`, `win A`, `lose A`, `camera X,Z [zoom D]`.
//! Times are `m:ss`, positions map units. The script runs inside the deterministic sim
//! step (after cleanup, in file order) and its text is stored in replay headers.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::skirmish::{ArmyId, Command, GameResult, Role, Skirmish, UnitId, VictoryCondition};
use crate::sim::{SCALE, TICK_HZ};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Control {
    Human,
    Ai(String),
    None,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmyDecl {
    pub index: usize,
    pub faction: String,
    pub team: Option<u8>,
    pub control: Control,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
}

impl Cmp {
    fn test(self, a: i64, b: i64) -> bool {
        match self {
            Cmp::Lt => a < b,
            Cmp::Le => a <= b,
            Cmp::Gt => a > b,
            Cmp::Ge => a >= b,
            Cmp::Eq => a == b,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cond {
    At(u64),
    Every { period: u64, from: u64, until: Option<u64> },
    CommanderDead(ArmyId),
    NoUnits(ArmyId),
    Count { army: ArmyId, unit_type: Option<String>, at: (i64, i64), radius: i64, cmp: Cmp, n: i64 },
    Objective { id: String, failed: bool },
    /// Phase 39: every part holds (`COND and COND ...`).
    All(Vec<Cond>),
}

/// Phase 39: the built-in tutorial mission.
pub const TUTORIAL: &str = include_str!("../missions/tutorial.farust-mission");

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnitGoal {
    Attack((i64, i64)),
    Move((i64, i64)),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Say(String),
    Spawn { army: ArmyId, unit_type: String, count: u32, at: (i64, i64), goal: Option<UnitGoal> },
    Objective { id: String, failed: bool },
    Win(ArmyId),
    Lose(ArmyId),
    Camera { at: (i64, i64), zoom: Option<i64> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trigger {
    pub cond: Cond,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveState {
    Active,
    Complete,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectiveDecl {
    pub id: String,
    pub primary: bool,
    pub text: String,
}

/// A parsed mission script.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mission {
    pub name: String,
    pub map: String,
    pub victory: VictoryCondition,
    pub armies: Vec<ArmyDecl>,
    pub objectives: Vec<ObjectiveDecl>,
    pub triggers: Vec<Trigger>,
    /// The exact source text (stored in replays).
    pub source: String,
}

fn parse_time(s: &str) -> Result<u64, String> {
    let (m, sec) = s.split_once(':').ok_or_else(|| format!("time '{s}' must be m:ss"))?;
    let m: u64 = m.parse().map_err(|_| format!("bad minutes in '{s}'"))?;
    let sec: u64 = sec.parse().map_err(|_| format!("bad seconds in '{s}'"))?;
    if sec >= 60 {
        return Err(format!("seconds out of range in '{s}'"));
    }
    Ok((m * 60 + sec) * TICK_HZ)
}

fn parse_pos(s: &str) -> Result<(i64, i64), String> {
    let (x, z) = s.split_once(',').ok_or_else(|| format!("position '{s}' must be X,Z"))?;
    let f = |v: &str| v.trim().parse::<f64>().map(|v| (v * SCALE as f64).round() as i64).map_err(|_| format!("bad coordinate '{v}'"));
    Ok((f(x)?, f(z)?))
}

fn parse_army(s: &str) -> Result<ArmyId, String> {
    s.parse::<u8>().ok().filter(|a| *a < 8).ok_or_else(|| format!("bad army '{s}' (0..7)"))
}

/// Splits on whitespace, keeping `"quoted strings"` together (quotes stripped).
fn tokens(s: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    let mut was_quoted = false;
    for c in s.chars() {
        match c {
            '"' => {
                quoted = !quoted;
                was_quoted = true;
            }
            c if c.is_whitespace() && !quoted => {
                if !cur.is_empty() || was_quoted {
                    out.push(std::mem::take(&mut cur));
                }
                was_quoted = false;
            }
            c => cur.push(c),
        }
    }
    if quoted {
        return Err("unterminated quote".into());
    }
    if !cur.is_empty() || was_quoted {
        out.push(cur);
    }
    Ok(out)
}

fn parse_cond(t: &[String]) -> Result<Cond, String> {
    let parts: Vec<&[String]> = t.split(|x| x == "and").collect();
    if parts.len() > 1 {
        if parts.iter().any(|p| p.is_empty()) {
            return Err("empty side of 'and'".into());
        }
        return parts.into_iter().map(parse_cond).collect::<Result<Vec<_>, _>>().map(Cond::All);
    }
    let w = |i: usize| t.get(i).map(String::as_str).ok_or_else(|| "incomplete condition".to_string());
    match w(0)? {
        "at" => Ok(Cond::At(parse_time(w(1)?)?)),
        "every" => {
            let period = parse_time(w(1)?)?.max(1);
            let from = if w(2).ok() == Some("from") { parse_time(w(3)?)? } else { 0 };
            let until = match t.iter().position(|x| x == "until") {
                Some(i) => Some(parse_time(w(i + 1)?)?),
                None => None,
            };
            Ok(Cond::Every { period, from, until })
        }
        "dead" if w(1)? == "commander" => Ok(Cond::CommanderDead(parse_army(w(2)?)?)),
        "no" if w(1)? == "units" => Ok(Cond::NoUnits(parse_army(w(2)?)?)),
        "count" => {
            // count A TYPE in X,Z rR OP N
            let army = parse_army(w(1)?)?;
            let unit_type = match w(2)? {
                "any" => None,
                ty => Some(ty.to_ascii_uppercase()),
            };
            if w(3)? != "in" {
                return Err("count: expected 'in'".into());
            }
            let at = parse_pos(w(4)?)?;
            let radius = w(5)?
                .strip_prefix('r')
                .and_then(|r| r.parse::<f64>().ok())
                .map(|r| (r * SCALE as f64) as i64)
                .ok_or("count: radius must be rN")?;
            let cmp = match w(6)? {
                "<" => Cmp::Lt,
                "<=" => Cmp::Le,
                ">" => Cmp::Gt,
                ">=" => Cmp::Ge,
                "==" => Cmp::Eq,
                o => return Err(format!("count: unknown comparison '{o}'")),
            };
            let n = w(7)?.parse().map_err(|_| "count: bad number")?;
            Ok(Cond::Count { army, unit_type, at, radius, cmp, n })
        }
        "objective" => Ok(Cond::Objective {
            id: w(1)?.to_string(),
            failed: match w(2)? {
                "complete" => false,
                "failed" | "fail" => true,
                o => return Err(format!("objective state '{o}'")),
            },
        }),
        o => Err(format!("unknown condition '{o}'")),
    }
}

fn parse_action(t: &[String]) -> Result<Action, String> {
    let w = |i: usize| t.get(i).map(String::as_str).ok_or_else(|| "incomplete action".to_string());
    match w(0)? {
        "say" => Ok(Action::Say(w(1)?.to_string())),
        "spawn" => {
            let army = parse_army(w(1)?)?;
            let unit_type = w(2)?.to_ascii_uppercase();
            let count = w(3)?.strip_prefix('x').and_then(|n| n.parse().ok()).filter(|n| (1..=200).contains(n)).ok_or("spawn: count must be x1..x200")?;
            if w(4)? != "at" {
                return Err("spawn: expected 'at'".into());
            }
            let at = parse_pos(w(5)?)?;
            let goal = match t.get(6).map(String::as_str) {
                None => None,
                Some("attack") => Some(UnitGoal::Attack(parse_pos(w(7)?)?)),
                Some("move") => Some(UnitGoal::Move(parse_pos(w(7)?)?)),
                Some(o) => return Err(format!("spawn: unknown goal '{o}'")),
            };
            Ok(Action::Spawn { army, unit_type, count, at, goal })
        }
        "objective" => Ok(Action::Objective {
            id: w(1)?.to_string(),
            failed: match w(2)? {
                "complete" => false,
                "fail" | "failed" => true,
                o => return Err(format!("objective action '{o}'")),
            },
        }),
        "win" => Ok(Action::Win(parse_army(w(1)?)?)),
        "lose" => Ok(Action::Lose(parse_army(w(1)?)?)),
        "camera" => {
            let at = parse_pos(w(1)?)?;
            let zoom = (t.get(2).map(String::as_str) == Some("zoom")).then(|| w(3).ok().and_then(|z| z.parse::<i64>().ok())).flatten();
            Ok(Action::Camera { at, zoom })
        }
        o => Err(format!("unknown action '{o}'")),
    }
}

impl Mission {
    pub fn parse(source: &str) -> Result<Self, String> {
        let mut m = Mission {
            name: "Untitled".into(),
            map: String::new(),
            victory: VictoryCondition::Sandbox,
            armies: Vec::new(),
            objectives: Vec::new(),
            triggers: Vec::new(),
            source: source.to_string(),
        };
        for (ln, raw) in source.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let err = |e: String| format!("line {}: {e}", ln + 1);
            let t = tokens(line).map_err(err)?;
            match t[0].as_str() {
                "mission" => m.name = t.get(1).cloned().ok_or_else(|| err("mission needs a name".into()))?,
                "map" => m.map = t.get(1).cloned().ok_or_else(|| err("map needs an id".into()))?,
                "victory" => {
                    m.victory = t.get(1).and_then(|v| VictoryCondition::parse(v)).ok_or_else(|| err("unknown victory condition".into()))?
                }
                "army" => {
                    let index = t.get(1).and_then(|i| i.parse::<usize>().ok()).filter(|i| *i < 8).ok_or_else(|| err("army index 0..7".into()))?;
                    let mut d = ArmyDecl { index, faction: "uef".into(), team: None, control: Control::Ai("normal".into()) };
                    let mut i = 2;
                    while i < t.len() {
                        match t[i].as_str() {
                            "faction" => {
                                d.faction = t.get(i + 1).cloned().ok_or_else(|| err("faction name".into()))?.to_ascii_lowercase();
                                i += 1;
                            }
                            "team" => {
                                d.team = Some(t.get(i + 1).and_then(|v| v.parse().ok()).ok_or_else(|| err("team number".into()))?);
                                i += 1;
                            }
                            "human" => d.control = Control::Human,
                            "ai" => {
                                let v = t.get(i + 1).cloned().ok_or_else(|| err("ai difficulty or none".into()))?;
                                d.control = if v == "none" { Control::None } else { Control::Ai(v) };
                                i += 1;
                            }
                            o => return Err(err(format!("unknown army option '{o}'"))),
                        }
                        i += 1;
                    }
                    m.armies.push(d);
                }
                "objective" => {
                    let id = t.get(1).cloned().ok_or_else(|| err("objective id".into()))?;
                    let primary = match t.get(2).map(String::as_str) {
                        Some("primary") => true,
                        Some("secondary") => false,
                        _ => return Err(err("objective must be primary or secondary".into())),
                    };
                    let text = t.get(3).cloned().unwrap_or_else(|| id.clone());
                    m.objectives.push(ObjectiveDecl { id, primary, text });
                }
                "when" => {
                    let do_at = t.iter().position(|x| x == "do").ok_or_else(|| err("trigger needs 'do'".into()))?;
                    let cond = parse_cond(&t[1..do_at]).map_err(err)?;
                    // Actions are separated by ';' (possibly glued to the previous token).
                    let mut actions = Vec::new();
                    let mut cur: Vec<String> = Vec::new();
                    for tok in &t[do_at + 1..] {
                        let (body, ends) = match tok.strip_suffix(';') {
                            Some(b) => (b.to_string(), true),
                            None if tok == ";" => (String::new(), true),
                            None => (tok.clone(), false),
                        };
                        if !body.is_empty() {
                            cur.push(body);
                        }
                        if ends && !cur.is_empty() {
                            actions.push(parse_action(&cur).map_err(err)?);
                            cur.clear();
                        }
                    }
                    if !cur.is_empty() {
                        actions.push(parse_action(&cur).map_err(err)?);
                    }
                    if actions.is_empty() {
                        return Err(err("trigger has no actions".into()));
                    }
                    m.triggers.push(Trigger { cond, actions });
                }
                o => return Err(err(format!("unknown statement '{o}'"))),
            }
        }
        if m.map.is_empty() {
            return Err("mission has no 'map'".into());
        }
        if m.armies.len() < 2 {
            return Err("mission needs at least two 'army' lines".into());
        }
        let mut idx: Vec<usize> = m.armies.iter().map(|a| a.index).collect();
        idx.sort_unstable();
        if idx != (0..m.armies.len()).collect::<Vec<_>>() {
            return Err("army indices must be 0, 1, ... without gaps".into());
        }
        m.armies.sort_by_key(|a| a.index);
        let ids: Vec<&String> = m.objectives.iter().map(|o| &o.id).collect();
        for t in &m.triggers {
            let refs = t.actions.iter().filter_map(|a| match a {
                Action::Objective { id, .. } => Some(id),
                _ => None,
            });
            fn cond_refs<'a>(c: &'a Cond, out: &mut Vec<&'a String>) {
                match c {
                    Cond::Objective { id, .. } => out.push(id),
                    Cond::All(parts) => parts.iter().for_each(|p| cond_refs(p, out)),
                    _ => {}
                }
            }
            let mut cr = Vec::new();
            cond_refs(&t.cond, &mut cr);
            for id in refs.chain(cr) {
                if !ids.contains(&id) {
                    return Err(format!("trigger references unknown objective '{id}'"));
                }
            }
        }
        Ok(m)
    }

    /// Index of the human army (first `human`), if any.
    pub fn human_army(&self) -> Option<usize> {
        self.armies.iter().find(|a| a.control == Control::Human).map(|a| a.index)
    }
}

/// Reads and parses a mission file.
pub fn load_file(path: &std::path::Path) -> Result<Mission, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Mission::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// The world a mission starts in: army `i` takes map start spot `i` with its declared
/// faction and team, the mission's victory condition, and the script installed.
pub fn build_world(m: &Mission, catalog: crate::skirmish::Catalog, map: &crate::skirmish::MapContext, seed: u64) -> Result<Skirmish, String> {
    let setups = m
        .armies
        .iter()
        .map(|d| {
            let faction = catalog
                .factions
                .iter()
                .position(|f| f.key == d.faction)
                .ok_or_else(|| format!("unknown faction '{}'", d.faction))?;
            let (name, start) = map.starts.get(d.index).cloned().ok_or_else(|| format!("map has no start spot {}", d.index + 1))?;
            Ok(crate::skirmish::ArmySetup { name, faction, start, team: d.team.unwrap_or(d.index as u8) })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut w = Skirmish::new(catalog, map.grid.clone(), map.size, map.mass_points.clone(), &setups, seed)?;
    w.attach_map(map);
    w.set_options(crate::skirmish::GameOptions { victory: m.victory, ..crate::skirmish::GameOptions::default() });
    w.set_mission(m.clone());
    Ok(w)
}

/// AI-driven armies: `(army, difficulty)`; `human_as` also hands the human army to an AI
/// (headless runs).
pub fn ai_armies(m: &Mission, human_as: Option<&str>) -> Result<Vec<(ArmyId, crate::skirmish_ai::Difficulty)>, String> {
    m.armies
        .iter()
        .filter_map(|d| match &d.control {
            Control::Ai(diff) => Some((d.index, diff.as_str())),
            Control::Human => human_as.map(|h| (d.index, h)),
            Control::None => None,
        })
        .map(|(i, diff)| Ok((i as ArmyId, diff.parse()?)))
        .collect()
}

/// `farust mission FILE [GAME_DIR] [--seed S] [--max-ticks N] [--human-ai DIFF] [--record OUT]`:
/// runs a mission headless (the human army is idle unless `--human-ai` is given) and prints
/// its messages, objectives, result and final hash.
pub fn cmd_mission(args: &[String]) -> Result<(), String> {
    let file = args.get(2).filter(|a| !a.starts_with("--")).ok_or("usage: farust mission FILE [GAME_DIR] [--seed S] [--max-ticks N] [--human-ai DIFF] [--record OUT]")?;
    let flag = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let game_dir = args.get(3).filter(|a| !a.starts_with("--")).map(std::path::PathBuf::from).unwrap_or_else(crate::scd::default_game_dir);
    let seed = flag("--seed").map(|s| s.parse::<u64>().map_err(|_| "bad --seed")).transpose()?.unwrap_or(crate::sim::DEFAULT_SEED);
    let max_ticks = flag("--max-ticks").map(|s| s.parse::<u64>().map_err(|_| "bad --max-ticks")).transpose()?.unwrap_or(36_000);
    let m = load_file(std::path::Path::new(file))?;
    let catalog = crate::skirmish::Catalog::load(&game_dir)?;
    let map = crate::skirmish::MapContext::load(&game_dir, &m.map)?;
    let mut w = build_world(&m, catalog, &map, seed)?;
    let mut ais: Vec<crate::skirmish_ai::ScriptedAi> = ai_armies(&m, flag("--human-ai").as_deref())?
        .into_iter()
        .map(|(a, d)| crate::skirmish_ai::ScriptedAi::with_difficulty(a, d))
        .collect();
    let mut rec = flag("--record").map(|_| {
        let mut r = crate::replay::ReplayRecorder::new(&map.map_id, seed, crate::savegame::setups_of(&w));
        r.set_options(w.options());
        r.set_mission(w.mission_source());
        r
    });
    println!("mission:  {} on {} ({} trigger(s), {} objective(s))", m.name, map.map_name, m.triggers.len(), m.objectives.len());
    let mut said = 0;
    while w.result().is_none() && w.tick() < max_ticks {
        let tick = w.tick();
        for ai in &mut ais {
            for cmd in ai.decide(&w) {
                if let Some(r) = rec.as_mut() {
                    r.record(tick, ai.army, &cmd);
                }
                w.issue(ai.army, cmd);
            }
        }
        w.step();
        if let Some(r) = rec.as_mut() {
            r.maybe_checkpoint(w.tick(), w.state_hash());
        }
        if let Some(rt) = w.mission() {
            for (t, text) in &rt.log[said..] {
                println!("[{:02}:{:02}] {text}", t / TICK_HZ / 60, t / TICK_HZ % 60);
            }
            said = rt.log.len();
        }
    }
    if let Some(rt) = w.mission() {
        for o in &m.objectives {
            println!("objective {:<10} {:<9} {:?}  {}", o.id, if o.primary { "primary" } else { "secondary" }, rt.objectives[&o.id], o.text);
        }
    }
    println!("result:   {:?} after {} ticks", w.result(), w.tick());
    println!("hash:     {:016x}", w.state_hash());
    if let (Some(r), Some(out)) = (rec, flag("--record")) {
        let rep = r.finish(w.tick(), w.state_hash());
        rep.save(std::path::Path::new(&out))?;
        println!("replay:   {out}");
    }
    Ok(())
}

/// Per-world mission state (cloned with the world; part of its deterministic state).
#[derive(Debug, Clone)]
pub struct MissionRuntime {
    pub mission: Arc<Mission>,
    fired: Vec<bool>,
    pub objectives: BTreeMap<String, ObjectiveState>,
    /// `(tick, text)` of every `say` so far.
    pub log: Vec<(u64, String)>,
    /// Latest `camera` request: `(tick, position, zoom)`.
    pub camera: Option<(u64, (i64, i64), Option<i64>)>,
    pub outcome: Option<GameResult>,
}

impl MissionRuntime {
    pub fn new(mission: Mission) -> Self {
        let objectives = mission.objectives.iter().map(|o| (o.id.clone(), ObjectiveState::Active)).collect();
        Self { fired: vec![false; mission.triggers.len()], mission: Arc::new(mission), objectives, log: Vec::new(), camera: None, outcome: None }
    }

    /// Small digest of the script state for the world hash.
    pub fn digest(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |b: u8| {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        };
        for f in &self.fired {
            mix(u8::from(*f));
        }
        for s in self.objectives.values() {
            mix(*s as u8);
        }
        for b in (self.log.len() as u32).to_le_bytes() {
            mix(b);
        }
        h
    }

    fn cond_true(&self, c: &Cond, w: &Skirmish, tick: u64) -> bool {
        match c {
            Cond::At(t) => tick >= *t,
            Cond::Every { period, from, until } => tick >= *from && until.is_none_or(|u| tick <= u) && (tick - from).is_multiple_of(*period),
            Cond::CommanderDead(a) => w.armies().get(*a as usize).is_some() && w.commander_of(*a).is_none(),
            Cond::NoUnits(a) => !w.units().any(|u| u.alive && u.army == *a),
            Cond::Count { army, unit_type, at, radius, cmp, n } => {
                let lim = (*radius as i128).pow(2);
                let c = w
                    .units()
                    .filter(|u| {
                        u.alive
                            && u.army == *army
                            && unit_type.as_ref().is_none_or(|t| u.type_id.eq_ignore_ascii_case(t))
                            && ((u.pos.x - at.0) as i128).pow(2) + ((u.pos.y - at.1) as i128).pow(2) <= lim
                    })
                    .count() as i64;
                cmp.test(c, *n)
            }
            Cond::Objective { id, failed } => {
                self.objectives.get(id).is_some_and(|s| *s == if *failed { ObjectiveState::Failed } else { ObjectiveState::Complete })
            }
            Cond::All(parts) => parts.iter().all(|p| self.cond_true(p, w, tick)),
        }
    }

    /// Evaluates every trigger in file order against the post-cleanup world.
    pub fn step(&mut self, w: &mut Skirmish) {
        let tick = w.tick();
        for i in 0..self.mission.triggers.len() {
            let repeating = matches!(self.mission.triggers[i].cond, Cond::Every { .. });
            if (!repeating && self.fired[i]) || self.outcome.is_some() {
                continue;
            }
            if !self.cond_true(&self.mission.triggers[i].cond.clone(), w, tick) {
                continue;
            }
            self.fired[i] = true;
            let actions = self.mission.triggers[i].actions.clone();
            for a in actions {
                self.apply(a, w, tick);
            }
        }
    }

    fn apply(&mut self, a: Action, w: &mut Skirmish, tick: u64) {
        match a {
            Action::Say(t) => self.log.push((tick, t)),
            Action::Camera { at, zoom } => self.camera = Some((tick, at, zoom)),
            Action::Objective { id, failed } => {
                if let Some(s) = self.objectives.get_mut(&id)
                    && *s == ObjectiveState::Active
                {
                    *s = if failed { ObjectiveState::Failed } else { ObjectiveState::Complete };
                }
            }
            Action::Win(army) => self.outcome = Some(GameResult::Victory(army)),
            Action::Lose(army) => {
                let winner = w.armies().iter().find(|x| x.id != army && !w.are_allies(x.id, army)).map(|x| x.id);
                self.outcome = Some(winner.map_or(GameResult::Draw, GameResult::Victory));
            }
            Action::Spawn { army, unit_type, count, at, goal } => {
                if w.army(army).is_none() {
                    return;
                }
                let mut ids: Vec<UnitId> = Vec::new();
                for k in 0..count {
                    let slot = Skirmish::formation_slot(k as usize, count as usize, at.0, at.1);
                    if let Some(id) = w.spawn_unit(army, &unit_type, slot) {
                        ids.push(id);
                    }
                }
                let ids: Vec<UnitId> = ids.into_iter().filter(|id| w.unit(*id).is_some_and(|u| u.role != Role::Commander)).collect();
                match goal {
                    Some(UnitGoal::Attack((x, y))) if !ids.is_empty() => {
                        w.apply_command_direct(army, Command::AttackMove { units: ids, x, y, queue: false });
                    }
                    Some(UnitGoal::Move((x, y))) if !ids.is_empty() => {
                        w.apply_command_direct(army, Command::Move { units: ids, x, y, queue: false });
                    }
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOLD: &str = r#"
# test mission
mission "Hold the Line"
map TESTMAP
army 0 faction red human
army 1 faction blue ai none
objective hold primary "Defend for one minute"
when at 0:00 do say "Hold here."; camera 60,60 zoom 80
when every 0:10 from 0:05 until 0:25 do spawn 1 BT x2 at 200,200 attack 60,60
when at 1:00 do objective hold complete; win 0
when dead commander 0 do objective hold fail; lose 0
"#;

    #[test]
    fn runtime_spawns_waves_tracks_objectives_and_ends_the_game_deterministically() {
        let run = || {
            let mut w = crate::skirmish::tests::test_world(3);
            w.set_options(crate::skirmish::GameOptions { victory: VictoryCondition::Sandbox, ..Default::default() });
            w.set_mission(Mission::parse(HOLD).unwrap());
            w.run(300);
            let blue: Vec<_> = w.units().filter(|u| u.army == 1 && u.type_id == "BT").map(|u| u.id).collect();
            assert_eq!(blue.len(), 6, "three waves of two by 0:30 (0:05, 0:15, 0:25)");
            assert!(blue.iter().all(|id| matches!(w.unit(*id).unwrap().orders.front(), Some(crate::skirmish::UnitOrder::AttackMove { .. }))));
            let rt = w.mission().unwrap();
            assert_eq!(rt.log, vec![(1, "Hold here.".to_string())]);
            assert_eq!(rt.camera, Some((1, (60_000, 60_000), Some(80))));
            assert_eq!(rt.objectives["hold"], ObjectiveState::Active);
            w.run(400);
            (w.result(), w.mission().unwrap().objectives["hold"], w.tick(), w.state_hash())
        };
        let a = run();
        assert!(matches!(a.1, ObjectiveState::Complete | ObjectiveState::Failed), "objective resolved: {a:?}");
        assert!(a.0.is_some(), "the script ended the game");
        assert_eq!(run(), a, "mission runs are deterministic");
    }

    #[test]
    fn and_conditions_and_the_builtin_tutorial_parse() {
        let m = Mission::parse(TUTORIAL).unwrap();
        assert_eq!(m.objectives.len(), 6);
        assert!(matches!(&m.triggers[2].cond, Cond::All(p) if p.len() == 2));
        assert!(Mission::parse("map X\narmy 0\narmy 1\nwhen at 0:01 and do win 0").is_err());
    }

    #[test]
    fn parses_statements_and_rejects_errors() {
        let m = Mission::parse(HOLD).unwrap();
        assert_eq!(m.name, "Hold the Line");
        assert_eq!(m.human_army(), Some(0));
        assert_eq!(m.armies[1].control, Control::None);
        assert_eq!(m.triggers.len(), 4);
        assert_eq!(m.triggers[1].cond, Cond::Every { period: 100, from: 50, until: Some(250) });
        assert_eq!(m.triggers[0].actions[1], Action::Camera { at: (60_000, 60_000), zoom: Some(80) });
        for bad in [
            "map X\narmy 0\narmy 1\nwhen at 1:99 do win 0",
            "map X\narmy 0\narmy 1\nwhen at 0:10 do objective nope complete",
            "map X\narmy 0\narmy 1\nwhen at 0:10 do spawn 1 BT x0 at 1,1",
            "map X\narmy 0\nwhen at 0:10 do win 0",
            "army 0\narmy 1",
            "map X\narmy 0\narmy 1\nfrobnicate",
        ] {
            assert!(Mission::parse(bad).is_err(), "{bad}");
        }
    }
}
