//! Phase 48 & 49: scenario scripting — Rust-native, data-driven missions and campaigns.
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
//! Phase 49 extensions (backwards-compatible):
//! - Army marker/coord starts & ACU-less groups: `army N ... [at X,Z|MARKER] [no_acu]`
//! - Pre-placed starting units/structures: `start A TYPE xN at X,Z|MARKER [hp PCT%] [attack|move|patrol X,Z|MARKER]`
//! - Map areas & category filters: `count A TYPE|structures|experimental|any [in X,Z|MARKER rR | in area AREA] OP N`
//! - Primary completion gate: `all primary complete` condition, and `win A` only succeeds when
//!   every primary objective is `Complete`
//! - Stage army cleanup: `defeat A`
//! - Render-side voice cues: `say "TEXT" [voice CUE]` (voiced by `GameAudio`, never affects sim)

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use crate::sim::{SCALE, TICK_HZ};
use crate::skirmish::{
    ArmyId, Catalog, Command, GameOptions, GameResult, MapContext, Role, Skirmish, UnitId,
    VictoryCondition,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Control {
    Human,
    Ai(String),
    None,
}

/// Map location: quantized `(x, z)` milli-units or a named `_save.lua` marker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Place {
    Pos((i64, i64)),
    Marker(String),
}

impl Place {
    pub fn resolve(&self, map: &MapContext) -> Result<(i64, i64), String> {
        match self {
            Place::Pos(p) => Ok(*p),
            Place::Marker(name) => map
                .marker(name)
                .ok_or_else(|| format!("map '{}' has no marker '{name}'", map.map_id)),
        }
    }

    fn resolved_place(&self, map: &MapContext) -> Result<Place, String> {
        self.resolve(map).map(Place::Pos)
    }

    fn as_pos(&self) -> (i64, i64) {
        match self {
            Place::Pos(p) => *p,
            Place::Marker(_) => (0, 0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmyDecl {
    pub index: usize,
    pub faction: String,
    pub team: Option<u8>,
    pub control: Control,
    pub at: Option<Place>,
    pub no_acu: bool,
    /// Phase 53: AI play style (`personality rush`).
    pub personality: crate::skirmish_ai::Personality,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartUnit {
    pub army: ArmyId,
    pub unit_type: String,
    pub count: u32,
    pub at: Place,
    pub hp_pct: Option<u8>,
    pub goal: Option<UnitGoal>,
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
pub enum UnitFilter {
    Any,
    Structures,
    Experimental,
    Type(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Region {
    Map,
    Circle { at: Place, radius: i64 },
    Area(String),
    Rect((i64, i64, i64, i64)),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cond {
    At(u64),
    Every { period: u64, from: u64, until: Option<u64> },
    CommanderDead(ArmyId),
    NoUnits(ArmyId),
    Count { army: ArmyId, filter: UnitFilter, region: Region, cmp: Cmp, n: i64 },
    Objective { id: String, failed: bool },
    AllPrimaryComplete,
    /// Phase 39: every part holds (`COND and COND ...`).
    All(Vec<Cond>),
}

/// Phase 39: the built-in tutorial mission.
pub const TUTORIAL: &str = include_str!("../missions/tutorial.farust-mission");

/// Phase 49: Forged Alliance campaign Operation 1 (`X1CA_001` — Operation: Black Day).
pub const CAMPAIGN: &str = include_str!("../missions/x1ca_001.farust-mission");

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnitGoal {
    Attack(Place),
    Move(Place),
    Patrol(Place),
}

impl UnitGoal {
    fn resolve(&self, map: &MapContext) -> Result<Self, String> {
        match self {
            UnitGoal::Attack(p) => Ok(UnitGoal::Attack(p.resolved_place(map)?)),
            UnitGoal::Move(p) => Ok(UnitGoal::Move(p.resolved_place(map)?)),
            UnitGoal::Patrol(p) => Ok(UnitGoal::Patrol(p.resolved_place(map)?)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Say { text: String, voice: Option<String> },
    Spawn { army: ArmyId, unit_type: String, count: u32, at: Place, goal: Option<UnitGoal> },
    Objective { id: String, failed: bool },
    Defeat(ArmyId),
    Win(ArmyId),
    Lose(ArmyId),
    Camera { at: Place, zoom: Option<i64> },
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
    pub starts: Vec<StartUnit>,
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
    let f = |v: &str| {
        v.trim()
            .parse::<f64>()
            .map(|v| (v * SCALE as f64).round() as i64)
            .map_err(|_| format!("bad coordinate '{v}'"))
    };
    Ok((f(x)?, f(z)?))
}

fn parse_place(s: &str) -> Result<Place, String> {
    if s.contains(',') {
        return parse_pos(s).map(Place::Pos);
    }
    let trimmed = s.trim();
    if trimmed.is_empty()
        || !trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | ' '))
    {
        return Err(format!("bad location or marker '{s}'"));
    }
    Ok(Place::Marker(trimmed.to_string()))
}

fn parse_army(s: &str) -> Result<ArmyId, String> {
    s.parse::<u8>()
        .ok()
        .filter(|a| *a < 8)
        .ok_or_else(|| format!("bad army '{s}' (0..7)"))
}

fn parse_cmp(s: &str) -> Result<Cmp, String> {
    match s {
        "<" => Ok(Cmp::Lt),
        "<=" => Ok(Cmp::Le),
        ">" => Ok(Cmp::Gt),
        ">=" => Ok(Cmp::Ge),
        "==" => Ok(Cmp::Eq),
        o => Err(format!("count: unknown comparison '{o}'")),
    }
}

fn parse_goal(kind: &str, target: &str) -> Result<UnitGoal, String> {
    let p = parse_place(target)?;
    match kind {
        "attack" => Ok(UnitGoal::Attack(p)),
        "move" => Ok(UnitGoal::Move(p)),
        "patrol" => Ok(UnitGoal::Patrol(p)),
        o => Err(format!("unknown unit goal '{o}'")),
    }
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
        return parts
            .into_iter()
            .map(parse_cond)
            .collect::<Result<Vec<_>, _>>()
            .map(Cond::All);
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
        "all" if w(1)? == "primary" && w(2)? == "complete" && t.len() == 3 => {
            Ok(Cond::AllPrimaryComplete)
        }
        "count" => {
            // count A TYPE|structures|experimental|any [in PLACE rR | in area AREA] OP N
            let army = parse_army(w(1)?)?;
            let filter = match w(2)? {
                "any" => UnitFilter::Any,
                "structures" => UnitFilter::Structures,
                "experimental" => UnitFilter::Experimental,
                ty => UnitFilter::Type(ty.to_ascii_uppercase()),
            };
            let (region, op_idx) = if w(3)? == "in" {
                if w(4)? == "area" {
                    let area = w(5)?.trim();
                    if area.is_empty() {
                        return Err("count: empty area name".into());
                    }
                    (Region::Area(area.to_string()), 6)
                } else {
                    let at = parse_place(w(4)?)?;
                    let radius = w(5)?
                        .strip_prefix('r')
                        .and_then(|r| r.parse::<f64>().ok())
                        .filter(|r| *r >= 0.0)
                        .map(|r| (r * SCALE as f64) as i64)
                        .ok_or("count: radius must be rN")?;
                    (Region::Circle { at, radius }, 6)
                }
            } else {
                (Region::Map, 3)
            };
            let cmp = parse_cmp(w(op_idx)?)?;
            let n = w(op_idx + 1)?.parse().map_err(|_| "count: bad number")?;
            if t.len() != op_idx + 2 {
                return Err("count: trailing tokens".into());
            }
            Ok(Cond::Count { army, filter, region, cmp, n })
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
        "say" => {
            let text = w(1)?.to_string();
            let voice = match t.get(2).map(String::as_str) {
                None => None,
                Some("voice") => {
                    let cue = w(3)?.trim();
                    if cue.is_empty() || t.len() != 4 {
                        return Err("say: expected 'voice CUE'".into());
                    }
                    Some(cue.to_string())
                }
                Some(o) => return Err(format!("say: unknown option '{o}'")),
            };
            Ok(Action::Say { text, voice })
        }
        "spawn" => {
            let army = parse_army(w(1)?)?;
            let unit_type = w(2)?.to_ascii_uppercase();
            let count = w(3)?
                .strip_prefix('x')
                .and_then(|n| n.parse().ok())
                .filter(|n| (1..=200).contains(n))
                .ok_or("spawn: count must be x1..x200")?;
            if w(4)? != "at" {
                return Err("spawn: expected 'at'".into());
            }
            let at = parse_place(w(5)?)?;
            let goal = match t.get(6).map(String::as_str) {
                None => None,
                Some(g) => {
                    if t.len() != 8 {
                        return Err("spawn: goal needs a target location".into());
                    }
                    Some(parse_goal(g, w(7)?)?)
                }
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
        "defeat" => Ok(Action::Defeat(parse_army(w(1)?)?)),
        "win" => Ok(Action::Win(parse_army(w(1)?)?)),
        "lose" => Ok(Action::Lose(parse_army(w(1)?)?)),
        "camera" => {
            let at = parse_place(w(1)?)?;
            let zoom = match t.get(2).map(String::as_str) {
                None => None,
                Some("zoom") => Some(w(3)?.parse::<i64>().map_err(|_| "camera: bad zoom")?),
                Some(o) => return Err(format!("camera: unknown option '{o}'")),
            };
            Ok(Action::Camera { at, zoom })
        }
        o => Err(format!("unknown action '{o}'")),
    }
}

fn parse_start(t: &[String]) -> Result<StartUnit, String> {
    let w = |i: usize| t.get(i).map(String::as_str).ok_or_else(|| "incomplete start".to_string());
    let army = parse_army(w(1)?)?;
    let unit_type = w(2)?.to_ascii_uppercase();
    let count = w(3)?
        .strip_prefix('x')
        .and_then(|n| n.parse().ok())
        .filter(|n| (1..=200).contains(n))
        .ok_or("start: count must be x1..x200")?;
    if w(4)? != "at" {
        return Err("start: expected 'at'".into());
    }
    let at = parse_place(w(5)?)?;
    let mut hp_pct = None;
    let mut goal = None;
    let mut i = 6;
    while i < t.len() {
        match t[i].as_str() {
            "hp" => {
                let raw = w(i + 1)?;
                let num = raw.strip_suffix('%').unwrap_or(raw);
                let pct: u8 = num
                    .parse()
                    .ok()
                    .filter(|p| (1..=100).contains(p))
                    .ok_or("start: hp must be 1%..100%")?;
                hp_pct = Some(pct);
                i += 2;
            }
            g @ ("attack" | "move" | "patrol") => {
                goal = Some(parse_goal(g, w(i + 1)?)?);
                i += 2;
            }
            o => return Err(format!("start: unknown option '{o}'")),
        }
    }
    Ok(StartUnit { army, unit_type, count, at, hp_pct, goal })
}

fn spawn_batch(
    w: &mut Skirmish,
    army: ArmyId,
    unit_type: &str,
    count: u32,
    at: (i64, i64),
    hp_pct: Option<u8>,
    goal: Option<&UnitGoal>,
) {
    if w.army(army).is_none_or(|a| a.defeated) {
        return;
    }
    let mut ids: Vec<UnitId> = Vec::new();
    for k in 0..count {
        let slot = if count == 1 {
            at
        } else {
            Skirmish::formation_slot(k as usize, count as usize, at.0, at.1)
        };
        if let Some(id) = w.spawn_unit(army, unit_type, slot) {
            if let Some(pct) = hp_pct {
                w.set_unit_hp_pct(id, pct);
            }
            ids.push(id);
        }
    }
    let ids: Vec<UnitId> = ids
        .into_iter()
        .filter(|id| w.unit(*id).is_some_and(|u| u.role != Role::Commander))
        .collect();
    if ids.is_empty() {
        return;
    }
    match goal {
        Some(UnitGoal::Attack(p)) => {
            let (x, y) = p.as_pos();
            w.apply_command_direct(army, Command::AttackMove { units: ids, x, y, queue: false });
        }
        Some(UnitGoal::Move(p)) => {
            let (x, y) = p.as_pos();
            w.apply_command_direct(army, Command::Move { units: ids, x, y, queue: false });
        }
        Some(UnitGoal::Patrol(p)) => {
            let (x, y) = p.as_pos();
            w.apply_command_direct(army, Command::Patrol { units: ids, x, y, queue: false });
        }
        None => {}
    }
}

impl Mission {
    pub fn parse(source: &str) -> Result<Self, String> {
        let mut m = Mission {
            name: "Untitled".into(),
            map: String::new(),
            victory: VictoryCondition::Sandbox,
            armies: Vec::new(),
            starts: Vec::new(),
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
                "mission" => {
                    m.name = t.get(1).cloned().ok_or_else(|| err("mission needs a name".into()))?
                }
                "map" => m.map = t.get(1).cloned().ok_or_else(|| err("map needs an id".into()))?,
                "victory" => {
                    m.victory = t
                        .get(1)
                        .and_then(|v| VictoryCondition::parse(v))
                        .ok_or_else(|| err("unknown victory condition".into()))?
                }
                "army" => {
                    let index = t.get(1).and_then(|i| i.parse::<usize>().ok()).filter(|i| *i < 8).ok_or_else(|| err("army index 0..7".into()))?;
                    let mut d = ArmyDecl {
                        index,
                        faction: "uef".into(),
                        team: None,
                        control: Control::Ai("normal".into()),
                        at: None,
                        no_acu: false,
                        personality: crate::skirmish_ai::Personality::Balanced,
                    };
                    let mut i = 2;
                    while i < t.len() {
                        match t[i].as_str() {
                            "faction" => {
                                d.faction = t
                                    .get(i + 1)
                                    .cloned()
                                    .ok_or_else(|| err("faction name".into()))?
                                    .to_ascii_lowercase();
                                i += 1;
                            }
                            "team" => {
                                d.team = Some(
                                    t.get(i + 1)
                                        .and_then(|v| v.parse().ok())
                                        .ok_or_else(|| err("team number".into()))?,
                                );
                                i += 1;
                            }
                            "human" => d.control = Control::Human,
                            "personality" => {
                                d.personality = crate::skirmish_ai::Personality::parse(t.get(i + 1).map(String::as_str).unwrap_or("")).map_err(err)?;
                                i += 1;
                            }
                            "ai" => {
                                let v = t
                                    .get(i + 1)
                                    .cloned()
                                    .ok_or_else(|| err("ai difficulty or none".into()))?;
                                d.control = if v == "none" { Control::None } else { Control::Ai(v) };
                                i += 1;
                            }
                            "at" => {
                                let loc = t
                                    .get(i + 1)
                                    .ok_or_else(|| err("army 'at' needs a marker or X,Z".into()))?;
                                d.at = Some(parse_place(loc).map_err(err)?);
                                i += 1;
                            }
                            "no_acu" => d.no_acu = true,
                            o => return Err(err(format!("unknown army option '{o}'"))),
                        }
                        i += 1;
                    }
                    m.armies.push(d);
                }
                "start" => m.starts.push(parse_start(&t).map_err(err)?),
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
                    let do_at = t
                        .iter()
                        .position(|x| x == "do")
                        .ok_or_else(|| err("trigger needs 'do'".into()))?;
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
        let num_armies = m.armies.len();
        for st in &m.starts {
            if usize::from(st.army) >= num_armies {
                return Err(format!("start references undeclared army {}", st.army));
            }
        }
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

    /// Every explicit blueprint id referenced by `start`, `spawn`, or `count` statements.
    pub fn referenced_types(&self) -> Vec<String> {
        let mut set = BTreeSet::new();
        for s in &self.starts {
            set.insert(s.unit_type.clone());
        }
        fn collect_cond(c: &Cond, set: &mut BTreeSet<String>) {
            match c {
                Cond::Count { filter: UnitFilter::Type(ty), .. } => {
                    set.insert(ty.clone());
                }
                Cond::All(parts) => parts.iter().for_each(|p| collect_cond(p, set)),
                _ => {}
            }
        }
        for t in &self.triggers {
            collect_cond(&t.cond, &mut set);
            for a in &t.actions {
                if let Action::Spawn { unit_type, .. } = a {
                    set.insert(unit_type.clone());
                }
            }
        }
        set.into_iter().collect()
    }

    /// Ensures any campaign/civilian blueprints referenced by this mission are in `catalog`.
    pub fn ensure_catalog_types(&self, catalog: &mut Catalog, game_dir: &Path) -> Result<(), String> {
        catalog.ensure_mission_types(game_dir, &self.referenced_types())
    }

    /// Resolves every named marker and area against `map` into concrete integer coordinates.
    pub fn resolve(&self, map: &MapContext) -> Result<Self, String> {
        fn resolve_cond(c: &Cond, map: &MapContext) -> Result<Cond, String> {
            match c {
                Cond::Count { army, filter, region, cmp, n } => {
                    let region = match region {
                        Region::Map => Region::Map,
                        Region::Circle { at, radius } => {
                            Region::Circle { at: at.resolved_place(map)?, radius: *radius }
                        }
                        Region::Area(name) => {
                            let r = map
                                .area(name)
                                .ok_or_else(|| format!("map '{}' has no area '{name}'", map.map_id))?;
                            Region::Rect(r)
                        }
                        Region::Rect(r) => Region::Rect(*r),
                    };
                    Ok(Cond::Count { army: *army, filter: filter.clone(), region, cmp: *cmp, n: *n })
                }
                Cond::All(parts) => parts
                    .iter()
                    .map(|p| resolve_cond(p, map))
                    .collect::<Result<Vec<_>, _>>()
                    .map(Cond::All),
                other => Ok(other.clone()),
            }
        }

        let mut out = self.clone();
        for a in &mut out.armies {
            if let Some(at) = &a.at {
                a.at = Some(at.resolved_place(map)?);
            }
        }
        for s in &mut out.starts {
            s.at = s.at.resolved_place(map)?;
            if let Some(g) = &s.goal {
                s.goal = Some(g.resolve(map)?);
            }
        }
        for tr in &mut out.triggers {
            tr.cond = resolve_cond(&tr.cond, map)?;
            for act in &mut tr.actions {
                match act {
                    Action::Spawn { at, goal, .. } => {
                        *at = at.resolved_place(map)?;
                        if let Some(g) = goal {
                            *goal = Some(g.resolve(map)?);
                        }
                    }
                    Action::Camera { at, .. } => {
                        *at = at.resolved_place(map)?;
                    }
                    _ => {}
                }
            }
        }
        Ok(out)
    }

    /// Installs the mission into a fresh tick-0 world (applying `no_acu` and `start` placements
    /// identically during live play, save loading, and replay verification).
    pub fn install(&self, w: &mut Skirmish, map: &MapContext) -> Result<(), String> {
        let resolved = self.resolve(map)?;
        let has_custom_start =
            resolved.armies.iter().any(|a| a.no_acu) || !resolved.starts.is_empty();
        if has_custom_start {
            for a in &resolved.armies {
                if a.no_acu {
                    w.remove_initial_commander(a.index as ArmyId);
                }
            }
            for st in &resolved.starts {
                spawn_batch(
                    w,
                    st.army,
                    &st.unit_type,
                    st.count,
                    st.at.as_pos(),
                    st.hp_pct,
                    st.goal.as_ref(),
                );
            }
            w.finalize_mission_start();
        }
        w.set_mission(resolved);
        Ok(())
    }
}

/// Ensures any mission-specific blueprints referenced in `source` are loaded in `catalog`.
pub fn ensure_mission_types_src(
    source: &str,
    catalog: &mut Catalog,
    game_dir: &Path,
) -> Result<(), String> {
    Mission::parse(source)?.ensure_catalog_types(catalog, game_dir)
}

/// Reads and parses a mission file.
pub fn load_file(path: &Path) -> Result<Mission, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Mission::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// The world a mission starts in: army `i` takes its declared `at` marker/position or map
/// start spot `i`, with its declared faction, team, victory condition, starting units and script.
pub fn build_world(
    m: &Mission,
    catalog: Catalog,
    map: &MapContext,
    seed: u64,
) -> Result<Skirmish, String> {
    let setups = m
        .armies
        .iter()
        .map(|d| {
            let faction = catalog
                .factions
                .iter()
                .position(|f| f.key == d.faction)
                .ok_or_else(|| format!("unknown faction '{}'", d.faction))?;
            let (name, start) = match &d.at {
                Some(place) => {
                    let pos = place.resolve(map)?;
                    let label = match place {
                        Place::Marker(mk) => mk.clone(),
                        Place::Pos(_) => format!("ARMY_{}", d.index + 1),
                    };
                    (label, pos)
                }
                None => map
                    .starts
                    .get(d.index)
                    .cloned()
                    .ok_or_else(|| format!("map has no start spot {}", d.index + 1))?,
            };
            Ok(crate::skirmish::ArmySetup {
                name,
                faction,
                start,
                team: d.team.unwrap_or(d.index as u8),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut w = Skirmish::new(catalog, map.grid.clone(), map.size, map.mass_points.clone(), &setups, seed)?;
    w.attach_map(map);
    w.set_options(GameOptions { victory: m.victory, ..GameOptions::default() });
    m.install(&mut w, map)?;
    Ok(w)
}

/// AI-driven armies: `(army, difficulty)`; `human_as` also hands the human army to an AI
/// (headless runs).
pub fn ai_armies(m: &Mission, human_as: Option<&str>) -> Result<Vec<crate::skirmish_ai::ScriptedAi>, String> {
    m.armies
        .iter()
        .filter_map(|d| match &d.control {
            Control::Ai(diff) => Some((d, diff.as_str())),
            Control::Human => human_as.map(|h| (d, h)),
            Control::None => None,
        })
        .map(|(d, diff)| Ok(crate::skirmish_ai::ScriptedAi::with_personality(d.index as ArmyId, diff.parse()?, d.personality)))
        .collect()
}

/// `farust mission FILE [GAME_DIR] [--seed S] [--max-ticks N] [--human-ai DIFF] [--record OUT] [--trace MIN]`:
/// runs a mission headless (the human army is idle unless `--human-ai` is given) and prints
/// its messages, objectives, result and final hash.
pub fn cmd_mission(args: &[String]) -> Result<(), String> {
    let file = args
        .get(2)
        .filter(|a| !a.starts_with("--"))
        .ok_or("usage: farust mission FILE [GAME_DIR] [--seed S] [--max-ticks N] [--human-ai DIFF] [--record OUT]")?;
    let flag = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let game_dir = args
        .get(3)
        .filter(|a| !a.starts_with("--"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(crate::scd::default_game_dir);
    let seed = flag("--seed")
        .map(|s| s.parse::<u64>().map_err(|_| "bad --seed"))
        .transpose()?
        .unwrap_or(crate::sim::DEFAULT_SEED);
    let max_ticks = flag("--max-ticks")
        .map(|s| s.parse::<u64>().map_err(|_| "bad --max-ticks"))
        .transpose()?
        .unwrap_or(36_000);
    let m = load_file(Path::new(file))?;
    let mut catalog = Catalog::load(&game_dir)?;
    m.ensure_catalog_types(&mut catalog, &game_dir)?;
    let map = MapContext::load(&game_dir, &m.map)?;
    let mut w = build_world(&m, catalog, &map, seed)?;
    let mut ais: Vec<crate::skirmish_ai::ScriptedAi> = ai_armies(&m, flag("--human-ai").as_deref())?;
    let mut rec = flag("--record").map(|_| {
        let mut r =
            crate::replay::ReplayRecorder::new(&map.map_id, seed, crate::savegame::setups_of(&w));
        r.set_options(w.options());
        r.set_mission(w.mission_source());
        r
    });
    println!(
        "mission:  {} on {} ({} trigger(s), {} objective(s))",
        m.name,
        map.map_name,
        m.triggers.len(),
        m.objectives.len()
    );
    // `--trace MIN`: per-army status every MIN minutes (mission tuning).
    let trace = flag("--trace").and_then(|s| s.parse::<u64>().ok()).filter(|m| *m > 0).map(|m| m * 60 * TICK_HZ);
    let mut said = 0;
    let mut issued: BTreeMap<ArmyId, u64> = BTreeMap::new();
    while w.result().is_none() && w.tick() < max_ticks {
        if let Some(every) = trace
            && w.tick() % every == 0
        {
            let mut line = format!("[{:02}:00] trace", w.tick() / TICK_HZ / 60);
            for a in w.armies() {
                let (mobile, structs, value) = w.army_summary(a.id);
                let acu = if w.commander_of(a.id).is_some() { "+acu" } else { "" };
                line.push_str(&format!(
                    " | A{} m{:.0} e{:.0}/{:.0} u{mobile} s{structs} v{value}{acu}{}",
                    a.id,
                    a.income_mass as f64 * 10.0 / crate::sim::SCALE as f64,
                    a.income_energy as f64 * 10.0 / crate::sim::SCALE as f64,
                    a.energy as f64 / crate::sim::SCALE as f64,
                    if a.defeated { " DEAD" } else { "" }
                ));
            }
            println!("{line}");
            let cmds: Vec<String> = issued.iter().map(|(a, n)| format!("A{a}:{n}")).collect();
            println!("          AI commands issued since the last trace: {}", cmds.join(" "));
            issued.clear();
        }
        let tick = w.tick();
        for ai in &mut ais {
            for cmd in ai.decide(&w) {
                if let Some(r) = rec.as_mut() {
                    r.record(tick, ai.army, &cmd);
                }
                if trace.is_some() {
                    *issued.entry(ai.army).or_default() += 1;
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
            println!(
                "objective {:<10} {:<9} {:?}  {}",
                o.id,
                if o.primary { "primary" } else { "secondary" },
                rt.objectives[&o.id],
                o.text
            );
        }
    }
    println!("result:   {:?} after {} ticks", w.result(), w.tick());
    println!("hash:     {:016x}", w.state_hash());
    if let (Some(r), Some(out)) = (rec, flag("--record")) {
        let rep = r.finish(w.tick(), w.state_hash());
        rep.save(Path::new(&out))?;
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
    /// Phase 49: `(tick, cue_name)` of every `say ... voice CUE` (read render-side by audio).
    pub voices: Vec<(u64, String)>,
    /// Latest `camera` request: `(tick, position, zoom)`.
    pub camera: Option<(u64, (i64, i64), Option<i64>)>,
    pub outcome: Option<GameResult>,
}

impl MissionRuntime {
    pub fn new(mission: Mission) -> Self {
        let objectives = mission
            .objectives
            .iter()
            .map(|o| (o.id.clone(), ObjectiveState::Active))
            .collect();
        Self {
            fired: vec![false; mission.triggers.len()],
            mission: Arc::new(mission),
            objectives,
            log: Vec::new(),
            voices: Vec::new(),
            camera: None,
            outcome: None,
        }
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

    /// True when every declared primary objective has reached `ObjectiveState::Complete`.
    pub fn all_primary_complete(&self) -> bool {
        self.mission
            .objectives
            .iter()
            .filter(|o| o.primary)
            .all(|o| self.objectives.get(&o.id) == Some(&ObjectiveState::Complete))
    }

    fn cond_true(&self, c: &Cond, w: &Skirmish, tick: u64) -> bool {
        match c {
            Cond::At(t) => tick >= *t,
            Cond::Every { period, from, until } => {
                tick >= *from
                    && until.is_none_or(|u| tick <= u)
                    && (tick - from).is_multiple_of(*period)
            }
            Cond::CommanderDead(a) => {
                w.armies().get(*a as usize).is_some() && w.commander_of(*a).is_none()
            }
            Cond::NoUnits(a) => !w.units().any(|u| u.alive && u.army == *a),
            Cond::Count { army, filter, region, cmp, n } => {
                let c = w
                    .units()
                    .filter(|u| {
                        if !u.alive || u.army != *army {
                            return false;
                        }
                        let ok_filter = match filter {
                            UnitFilter::Any => true,
                            UnitFilter::Structures => {
                                u.role.is_structure() && u.role != Role::Wall
                            }
                            UnitFilter::Experimental => {
                                u.role == Role::Experimental || u.tier >= 4
                            }
                            UnitFilter::Type(t) => u.type_id.eq_ignore_ascii_case(t),
                        };
                        if !ok_filter {
                            return false;
                        }
                        match region {
                            Region::Map => true,
                            Region::Circle { at, radius } => {
                                let p = at.as_pos();
                                let lim = (*radius as i128).pow(2);
                                ((u.pos.x - p.0) as i128).pow(2)
                                    + ((u.pos.y - p.1) as i128).pow(2)
                                    <= lim
                            }
                            Region::Rect((x0, z0, x1, z1)) => {
                                u.pos.x >= *x0
                                    && u.pos.x <= *x1
                                    && u.pos.y >= *z0
                                    && u.pos.y <= *z1
                            }
                            Region::Area(_) => false,
                        }
                    })
                    .count() as i64;
                cmp.test(c, *n)
            }
            Cond::Objective { id, failed } => self.objectives.get(id).is_some_and(|s| {
                *s == if *failed { ObjectiveState::Failed } else { ObjectiveState::Complete }
            }),
            Cond::AllPrimaryComplete => self.all_primary_complete(),
            Cond::All(parts) => parts.iter().all(|p| self.cond_true(p, w, tick)),
        }
    }

    /// Evaluates every trigger in file order against the post-cleanup world.
    pub fn step(&mut self, w: &mut Skirmish) {
        fn is_repeating(c: &Cond) -> bool {
            match c {
                Cond::Every { .. } => true,
                Cond::All(parts) => parts.iter().any(is_repeating),
                _ => false,
            }
        }
        let tick = w.tick();
        for i in 0..self.mission.triggers.len() {
            let repeating = is_repeating(&self.mission.triggers[i].cond);
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
            Action::Say { text, voice } => {
                self.log.push((tick, text));
                if let Some(cue) = voice {
                    self.voices.push((tick, cue));
                }
            }
            Action::Camera { at, zoom } => self.camera = Some((tick, at.as_pos(), zoom)),
            Action::Objective { id, failed } => {
                if let Some(s) = self.objectives.get_mut(&id)
                    && *s == ObjectiveState::Active
                {
                    *s = if failed { ObjectiveState::Failed } else { ObjectiveState::Complete };
                }
            }
            Action::Defeat(army) => w.defeat_army(army),
            Action::Win(army) => {
                if self.all_primary_complete() {
                    self.outcome = Some(GameResult::Victory(army));
                }
            }
            Action::Lose(army) => {
                let winner = w
                    .armies()
                    .iter()
                    .find(|x| x.id != army && !w.are_allies(x.id, army))
                    .map(|x| x.id);
                self.outcome = Some(winner.map_or(GameResult::Draw, GameResult::Victory));
            }
            Action::Spawn { army, unit_type, count, at, goal } => {
                spawn_batch(w, army, &unit_type, count, at.as_pos(), None, goal.as_ref());
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
            w.set_options(GameOptions { victory: VictoryCondition::Sandbox, ..Default::default() });
            w.set_mission(Mission::parse(HOLD).unwrap());
            w.run(300);
            let blue: Vec<_> = w
                .units()
                .filter(|u| u.army == 1 && u.type_id == "BT")
                .map(|u| u.id)
                .collect();
            assert_eq!(blue.len(), 6, "three waves of two by 0:30 (0:05, 0:15, 0:25)");
            assert!(blue.iter().all(|id| matches!(
                w.unit(*id).unwrap().orders.front(),
                Some(crate::skirmish::UnitOrder::AttackMove { .. })
            )));
            let rt = w.mission().unwrap();
            assert_eq!(rt.log, vec![(1, "Hold here.".to_string())]);
            assert_eq!(rt.camera, Some((1, (60_000, 60_000), Some(80))));
            assert_eq!(rt.objectives["hold"], ObjectiveState::Active);
            w.run(400);
            (w.result(), w.mission().unwrap().objectives["hold"], w.tick(), w.state_hash())
        };
        let a = run();
        assert!(
            matches!(a.1, ObjectiveState::Complete | ObjectiveState::Failed),
            "objective resolved: {a:?}"
        );
        assert!(a.0.is_some(), "the script ended the game");
        assert_eq!(run(), a, "mission runs are deterministic");
    }

    #[test]
    fn and_conditions_and_builtin_missions_parse() {
        let m = Mission::parse(TUTORIAL).unwrap();
        assert_eq!(m.objectives.len(), 6);
        assert!(matches!(&m.triggers[2].cond, Cond::All(p) if p.len() == 2));
        let camp = Mission::parse(CAMPAIGN).unwrap();
        assert_eq!(camp.map, "X1CA_001");
        assert!(camp.objectives.iter().any(|o| o.primary));
        assert!(Mission::parse("map X\narmy 0\narmy 1\nwhen at 0:01 and do win 0").is_err());
    }

    #[test]
    fn parses_statements_and_rejects_errors() {
        let m = Mission::parse(HOLD).unwrap();
        assert_eq!(m.name, "Hold the Line");
        assert_eq!(m.human_army(), Some(0));
        assert_eq!(m.armies[1].control, Control::None);
        assert_eq!(m.triggers.len(), 4);
        assert_eq!(
            m.triggers[1].cond,
            Cond::Every { period: 100, from: 50, until: Some(250) }
        );
        assert_eq!(
            m.triggers[0].actions[1],
            Action::Camera { at: Place::Pos((60_000, 60_000)), zoom: Some(80) }
        );
        for bad in [
            "map X\narmy 0\narmy 1\nwhen at 1:99 do win 0",
            "map X\narmy 0\narmy 1\nwhen at 0:10 do objective nope complete",
            "map X\narmy 0\narmy 1\nwhen at 0:10 do spawn 1 BT x0 at 1,1",
            "map X\narmy 0\nwhen at 0:10 do win 0",
            "army 0\narmy 1",
            "map X\narmy 0\narmy 1\nfrobnicate",
            "map X\narmy 0\narmy 1\nstart 2 BT x1 at 10,10",
            "map X\narmy 0\narmy 1\nstart 0 BT x1 at 10,10 hp 0%",
            "map X\narmy 0\narmy 1\nwhen at 0:01 do say \"hi\" bogus X",
            "map X\narmy 0\narmy 1\nwhen count 1 structures in area do win 0",
        ] {
            assert!(Mission::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn campaign_extensions_markers_areas_voice_defeat_and_primary_win_gate() {
        let src = r#"
mission "Campaign Test"
map TEST_MAP
victory sandbox
army 0 faction red team 0 human at PlayerSpawn
army 1 faction red team 0 ai none at AllySpawn no_acu
army 2 faction blue team 1 ai none at Outpost no_acu
army 3 faction blue team 1 ai none at BossSpawn
start 1 RF x1 at AllySpawn hp 75%
start 2 BF x1 at 150,150
start 2 BT x2 at Outpost patrol 140,140

objective clear_outpost primary "Clear outpost"
objective kill_boss primary "Defeat boss"
objective bonus secondary "Optional"

# Premature win before primaries complete must be ignored.
when at 0:01 do say "Briefing" voice X01_HQ_M01_02881; win 0
when count 2 structures in area OutpostArea == 0 do objective clear_outpost complete; defeat 2
when objective clear_outpost complete and count 3 any >= 1 do objective kill_boss complete
when all primary complete do say "Victory!" voice X01_Hall_M03_04258; win 0
"#;
        let m = Mission::parse(src).unwrap();
        assert_eq!(m.armies[0].at, Some(Place::Marker("PlayerSpawn".into())));
        assert!(m.armies[1].no_acu);
        assert_eq!(m.starts.len(), 3);
        assert_eq!(m.starts[0].hp_pct, Some(75));
        assert_eq!(m.referenced_types(), vec!["BF".to_string(), "BT".to_string(), "RF".to_string()]);

        let mut map = crate::skirmish::MapContext {
            map_id: "TEST_MAP".into(),
            map_name: "Test".into(),
            scmap: crate::scmap::ScMap {
                version_major: 2,
                version_minor: 60,
                map_width: 256.0,
                map_height: 256.0,
                preview_dds: vec![],
                heightmap_width: 8,
                heightmap_height: 8,
                height_scale: 0.01,
                heightmap: vec![2000; 81],
                terrain_shader: String::new(),
                background_path: String::new(),
                sky_cubemap_path: String::new(),
                cubemaps: vec![],
                lighting: crate::scmap::ScMapLighting::default(),
                water: crate::scmap::WaterInfo {
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
            grid: crate::pathfinding::PathGrid::new(64, 64, 4 * SCALE),
            naval_grid: crate::pathfinding::PathGrid::new(64, 64, 4 * SCALE),
            amphib_grid: crate::pathfinding::PathGrid::new(64, 64, 4 * SCALE),
            size: (256 * SCALE, 256 * SCALE),
            starts: vec![],
            mass_points: vec![],
            hydro_points: vec![],
            has_water: false,
            props: vec![],
            markers: BTreeMap::from([
                ("PlayerSpawn".into(), (40 * SCALE, 40 * SCALE)),
                ("AllySpawn".into(), (60 * SCALE, 40 * SCALE)),
                ("Outpost".into(), (150 * SCALE, 150 * SCALE)),
                ("BossSpawn".into(), (210 * SCALE, 210 * SCALE)),
            ]),
            areas: BTreeMap::from([(
                "OutpostArea".into(),
                (130 * SCALE, 130 * SCALE, 170 * SCALE, 170 * SCALE),
            )]),
        };

        // Missing marker / area must error cleanly.
        let mut missing_area = map.clone();
        missing_area.areas.clear();
        assert!(m.resolve(&missing_area).is_err());

        let cat = crate::skirmish::tests::test_catalog();
        let mut w = build_world(&m, cat, &map, 7).unwrap();
        assert!(w.commander_of(0).is_some());
        assert!(w.commander_of(1).is_none(), "no_acu removed army 1 ACU");
        assert!(w.commander_of(2).is_none(), "no_acu removed army 2 ACU");
        let rf = w.units().find(|u| u.army == 1 && u.type_id == "RF").unwrap();
        assert_eq!(rf.hp, rf.max_hp * 75 / 100);

        // At 0:01 (tick 10), `win 0` fires while primaries are Active -> game MUST continue!
        w.run(15);
        assert!(w.result().is_none(), "win blocked until all primaries complete");
        assert_eq!(
            w.mission().unwrap().voices,
            vec![(10, "X01_HQ_M01_02881".to_string())]
        );

        // Destroy the BF structure in OutpostArea -> clears outpost, defeats army 2, chains to boss & win!
        w.defeat_army(2);
        w.step();
        assert!(w.army(2).unwrap().defeated);
        assert_eq!(w.result(), Some(GameResult::Victory(0)));
        let rt = w.mission().unwrap();
        assert!(rt.all_primary_complete());
        assert_eq!(rt.objectives["bonus"], ObjectiveState::Active);
        assert_eq!(rt.voices.len(), 2);
        let _ = &mut map;
    }
}
