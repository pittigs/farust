//! Phase 24 data-driven unit roster.
//!
//! Derives who can build what straight from blueprint data, the way the FA engine does:
//! every `Economy.BuildableCategory` string is an AND of categories (a bare unit id is a
//! category too), the list is an OR, Commander enhancements add `BuildableCategoryAdds`,
//! and `General.UpgradesTo` chains structures. Starting from each faction's ACU the
//! reachable set is computed twice: once over the full FA tech tree (every unit with
//! build categories counts as a builder) and once over the mechanics FARust simulates
//! (engineers/ACUs/SACUs place structures and experimentals, land/air/naval factories and
//! quantum gates produce mobile units, structures upgrade). A unit is *playable* when it
//! is in the second set; units in the first set but not the second are reported with the
//! missing mechanic.

use std::collections::{BTreeMap, BTreeSet};

use crate::blueprint::Value;
use crate::flowfield::NavDomain;
use crate::skirmish::Role;

/// Faction keys in [`crate::skirmish::FACTION_TABLE`] order with their category names.
pub const FACTION_CATS: [(&str, &str); 4] = [("uef", "UEF"), ("cybran", "CYBRAN"), ("aeon", "AEON"), ("seraphim", "SERAPHIM")];

/// Blueprint facts the roster derivation needs (all strings upper-case).
#[derive(Debug, Clone, PartialEq)]
pub struct BpInfo {
    pub id: String,
    /// `Categories` plus the blueprint id itself (FA treats ids as categories).
    pub cats: BTreeSet<String>,
    /// `Economy.BuildableCategory` expressions.
    pub buildable: Vec<String>,
    /// `Enhancements.*.BuildableCategoryAdds` expressions (ACU / SACU engineering suites).
    pub enh_adds: Vec<String>,
    pub upgrades_to: Option<String>,
    pub domain: NavDomain,
    pub name: String,
}

impl BpInfo {
    pub fn from_blueprint(id: &str, root: &Value) -> Self {
        let id = id.to_ascii_uppercase();
        let mut cats: BTreeSet<String> = root
            .get("Categories")
            .map(Value::string_list)
            .unwrap_or_default()
            .into_iter()
            .map(|c| c.to_ascii_uppercase())
            .collect();
        cats.insert(id.clone());
        let buildable = root
            .get("Economy")
            .and_then(|e| e.get("BuildableCategory"))
            .map(Value::string_list)
            .unwrap_or_default()
            .into_iter()
            .map(|s| s.to_ascii_uppercase())
            .collect();
        let enh_adds = root
            .get("Enhancements")
            .map(|e| {
                e.keyed_tables()
                    .into_iter()
                    .filter_map(|(_, t)| t.get_str("BuildableCategoryAdds"))
                    .map(str::to_ascii_uppercase)
                    .collect()
            })
            .unwrap_or_default();
        let upgrades_to = root
            .get("General")
            .and_then(|g| g.get_str("UpgradesTo"))
            .filter(|s| !s.is_empty())
            .map(str::to_ascii_uppercase);
        let motion = root.get("Physics").and_then(|p| p.get_str("MotionType")).unwrap_or("");
        let mut domain = NavDomain::from_motion_type(motion);
        if cats.contains("AIR") && !cats.contains("STRUCTURE") {
            domain = NavDomain::Air;
        }
        let name = root
            .get_str("Description")
            .map(|d| match d.find('>') {
                Some(i) if d.starts_with("<LOC") => d[i + 1..].to_string(),
                _ => d.to_string(),
            })
            .unwrap_or_else(|| id.clone());
        Self { id, cats, buildable, enh_adds, upgrades_to, domain, name }
    }

    pub fn has(&self, c: &str) -> bool {
        self.cats.contains(c)
    }

    /// Faction index into [`FACTION_CATS`].
    pub fn faction(&self) -> Option<usize> {
        FACTION_CATS.iter().position(|(_, c)| self.has(c))
    }

    /// Tech level 1..=4 (4 = experimental).
    pub fn tier(&self) -> u8 {
        if self.has("EXPERIMENTAL") {
            4
        } else if self.has("TECH3") {
            3
        } else if self.has("TECH2") {
            2
        } else {
            1
        }
    }

    fn is_mobile_kind(&self) -> bool {
        self.has("MOBILE") && !self.has("STRUCTURE") && !self.has("MOBILESONAR")
    }
}

/// True if every whitespace-separated category of `expr` is in `cats`.
pub fn expr_matches(expr: &str, cats: &BTreeSet<String>) -> bool {
    let mut any = false;
    for w in expr.split_whitespace() {
        any = true;
        if !cats.contains(w) {
            return false;
        }
    }
    any
}

/// Maps a blueprint onto a simulated [`Role`], or names the missing mechanic.
pub fn infer_role(b: &BpInfo) -> Result<Role, &'static str> {
    let h = |c: &str| b.has(c);
    if (h("COMMAND") || h("COMMANDER")) && !h("SUBCOMMANDER") {
        return Ok(Role::Commander);
    }
    if h("SATELLITE") || h("ORBITALSYSTEM") {
        return Err("orbital satellite");
    }
    if h("POD") || (h("ENGINEERSTATION") || (h("PODSTAGINGPLATFORM") && h("STRUCTURE"))) {
        return Err("engineering drones / pod staging");
    }
    if h("MOBILESONAR") {
        return Ok(Role::Radar);
    }
    if b.is_mobile_kind() {
        if h("EXPERIMENTAL") {
            return Ok(Role::Experimental);
        }
        if h("ENGINEER") || (h("CONSTRUCTION") && !h("FACTORY")) {
            return Ok(Role::Engineer);
        }
        return Ok(Role::Mobile);
    }
    if !h("STRUCTURE") {
        return Err("neither mobile nor structure");
    }
    if h("WALL") {
        return Ok(Role::Wall);
    }
    if h("AIRSTAGINGPLATFORM") {
        return Err("air staging / refuelling");
    }
    if h("FACTORY") {
        if h("GATE") {
            return Ok(Role::Factory);
        }
        if h("AIR") {
            return Ok(Role::AirFactory);
        }
        if h("NAVAL") {
            return Ok(Role::NavalFactory);
        }
        return Ok(Role::Factory);
    }
    if h("MASSEXTRACTION") {
        return Ok(Role::Extractor);
    }
    if h("HYDROCARBON") {
        return Ok(Role::Hydro);
    }
    if h("MASSFABRICATION") || h("ENERGYPRODUCTION") || h("MASSPRODUCTION") {
        return Ok(Role::Generator);
    }
    if h("MASSSTORAGE") {
        return Ok(Role::MassStorage);
    }
    if h("ENERGYSTORAGE") {
        return Ok(Role::EnergyStorage);
    }
    if h("ANTIMISSILE") {
        return Ok(Role::MissileDefense);
    }
    if h("NUKE") || h("TACTICALMISSILEPLATFORM") || h("SILO") {
        return Ok(Role::MissileSilo);
    }
    if h("SHIELD") {
        return Ok(Role::Shield);
    }
    if h("ANTIAIR") && !h("DIRECTFIRE") && !h("INDIRECTFIRE") {
        return Ok(Role::AntiAir);
    }
    if h("DIRECTFIRE") || h("INDIRECTFIRE") || h("ANTINAVY") || h("ARTILLERY") {
        return Ok(Role::PointDefense);
    }
    if h("INTELLIGENCE") || h("RADAR") || h("SONAR") || h("OMNI") || h("COUNTERINTELLIGENCE") {
        return Ok(Role::Radar);
    }
    Err("structure behaviour not modelled")
}

/// How a build option is issued: `Build` (builder places it, `Command::Build`), `Produce`
/// (factory queue, `Command::Produce`) or `Upgrade` (`Command::Upgrade` on the builder).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BuildKind {
    Build,
    Produce,
    Upgrade,
}

/// Build-card tab of an option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BuildGroup {
    Economy,
    Factory,
    Defense,
    Intel,
    Strategic,
    Experimental,
    Engineer,
    Land,
    Air,
    Naval,
}

impl BuildGroup {
    pub fn label(self) -> &'static str {
        match self {
            BuildGroup::Economy => "economy",
            BuildGroup::Factory => "factory",
            BuildGroup::Defense => "defense",
            BuildGroup::Intel => "intel",
            BuildGroup::Strategic => "strategic",
            BuildGroup::Experimental => "experimental",
            BuildGroup::Engineer => "engineer",
            BuildGroup::Land => "land",
            BuildGroup::Air => "air",
            BuildGroup::Naval => "naval",
        }
    }

    pub fn of(role: Role, b: &BpInfo) -> Self {
        if b.tier() == 4 {
            return BuildGroup::Experimental;
        }
        match role {
            Role::Factory | Role::AirFactory | Role::NavalFactory => BuildGroup::Factory,
            Role::Extractor | Role::Generator | Role::Hydro | Role::MassStorage | Role::EnergyStorage => BuildGroup::Economy,
            Role::PointDefense | Role::AntiAir | Role::Shield | Role::MissileDefense | Role::Wall => BuildGroup::Defense,
            Role::Radar => BuildGroup::Intel,
            Role::MissileSilo => BuildGroup::Strategic,
            Role::Experimental => BuildGroup::Experimental,
            Role::Engineer | Role::Commander => BuildGroup::Engineer,
            Role::Mobile => {
                if b.has("AIR") {
                    BuildGroup::Air
                } else if b.has("NAVAL") {
                    BuildGroup::Naval
                } else {
                    BuildGroup::Land
                }
            }
        }
    }
}

/// One entry of a builder's build card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildOption {
    pub id: String,
    /// Tech level 1..=4 of the target (ACUs may only build up to their engineering suite).
    pub tier: u8,
    pub kind: BuildKind,
    pub group: BuildGroup,
}

/// Roster status of one blueprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// Buildable by someone in a FARust match.
    Playable,
    /// In the FA skirmish tech tree but needs a mechanic FARust does not simulate.
    Excluded(&'static str),
    /// Not reachable from any ACU in FA either (campaign / civilian / drones / debug).
    NotInTree,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RosterEntry {
    pub id: String,
    pub name: String,
    pub faction: Option<usize>,
    pub tier: u8,
    pub role: Option<Role>,
    /// Build-card group (economy, defense, land, air, ...).
    pub group: Option<BuildGroup>,
    /// Upper-case blueprint categories (including the id itself).
    pub categories: BTreeSet<String>,
    pub status: Status,
}

/// Derived roster: per-builder options and per-faction playable sets.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Roster {
    pub entries: BTreeMap<String, RosterEntry>,
    /// Builder blueprint id -> options, sorted by (kind, group, tier, id).
    pub options: BTreeMap<String, Vec<BuildOption>>,
    /// Faction index ([`FACTION_CATS`]) -> playable ids.
    pub playable: Vec<BTreeSet<String>>,
}

impl Roster {
    pub fn options_of(&self, builder: &str) -> &[BuildOption] {
        self.options.get(builder).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn can(&self, builder: &str, target: &str, kind: BuildKind) -> bool {
        self.options_of(builder).iter().any(|o| o.kind == kind && o.id == target)
    }
}

fn mobile_domain_group(b: &BpInfo) -> u8 {
    if b.has("AIR") {
        1
    } else if b.has("NAVAL") {
        2
    } else {
        0
    }
}

/// Options of `builder` under simulated mechanics.
fn sim_options(
    builder: &BpInfo,
    role: Role,
    bps: &BTreeMap<String, BpInfo>,
    roles: &BTreeMap<String, Result<Role, &'static str>>,
) -> Vec<BuildOption> {
    let mut out: Vec<BuildOption> = Vec::new();
    let faction = builder.faction();
    let exprs: Vec<&String> = builder.buildable.iter().chain(builder.enh_adds.iter()).collect();
    let lists = |c: &BpInfo| exprs.iter().any(|e| expr_matches(e, &c.cats));
    if role.is_builder() {
        for (id, c) in bps {
            if *id == builder.id || c.faction() != faction || !lists(c) {
                continue;
            }
            let Some(Ok(r)) = roles.get(id) else {
                continue;
            };
            if r.is_structure() || *r == Role::Experimental {
                out.push(BuildOption { id: id.clone(), tier: c.tier(), kind: BuildKind::Build, group: BuildGroup::of(*r, c) });
            }
        }
    } else if role.is_factory() {
        let want = match role {
            Role::AirFactory => 1,
            Role::NavalFactory => 2,
            _ => 0,
        };
        for (id, c) in bps {
            if *id == builder.id || c.faction() != faction || !lists(c) {
                continue;
            }
            let Some(Ok(r)) = roles.get(id) else {
                continue;
            };
            if matches!(r, Role::Mobile | Role::Engineer) && mobile_domain_group(c) == want {
                out.push(BuildOption { id: id.clone(), tier: c.tier(), kind: BuildKind::Produce, group: BuildGroup::of(*r, c) });
            }
        }
    }
    if let Some(up) = &builder.upgrades_to
        && let (Some(c), Some(Ok(r))) = (bps.get(up), roles.get(up))
        && c.faction() == faction
    {
        out.push(BuildOption { id: up.clone(), tier: c.tier(), kind: BuildKind::Upgrade, group: BuildGroup::of(*r, c) });
    }
    out.sort_by(|a, b| (a.kind, a.group, a.tier, &a.id).cmp(&(b.kind, b.group, b.tier, &b.id)));
    out
}

/// Every blueprint `b` lists (FA engine semantics, any unit with build categories).
fn tree_children<'a>(b: &BpInfo, bps: &'a BTreeMap<String, BpInfo>) -> Vec<&'a str> {
    let mut out: Vec<&str> = bps
        .values()
        .filter(|c| c.id != b.id && b.buildable.iter().chain(b.enh_adds.iter()).any(|e| expr_matches(e, &c.cats)))
        .map(|c| c.id.as_str())
        .collect();
    if let Some(up) = &b.upgrades_to
        && let Some((k, _)) = bps.get_key_value(up)
    {
        out.push(k.as_str());
    }
    out
}

/// Derives the roster. `fixed_roles` pins the role of blueprints whose role was chosen
/// explicitly (the legacy hand-made faction tables), so their behaviour never changes.
pub fn derive(bps: &BTreeMap<String, BpInfo>, fixed_roles: &BTreeMap<String, Role>) -> Roster {
    let roles: BTreeMap<String, Result<Role, &'static str>> = bps
        .iter()
        .map(|(id, b)| (id.clone(), fixed_roles.get(id).copied().map(Ok).unwrap_or_else(|| infer_role(b))))
        .collect();
    let commanders: Vec<&String> = bps
        .iter()
        .filter(|(id, b)| roles.get(*id) == Some(&Ok(Role::Commander)) && b.faction().is_some() && !b.buildable.is_empty())
        .map(|(id, _)| id)
        .collect();

    // Full FA tree.
    let mut tree: BTreeSet<String> = BTreeSet::new();
    let mut stack: Vec<String> = commanders.iter().map(|s| (*s).clone()).collect();
    while let Some(id) = stack.pop() {
        if !tree.insert(id.clone()) {
            continue;
        }
        if let Some(b) = bps.get(&id) {
            stack.extend(tree_children(b, bps).into_iter().filter(|c| !tree.contains(*c)).map(str::to_string));
        }
    }

    // Simulated tree.
    let mut options: BTreeMap<String, Vec<BuildOption>> = BTreeMap::new();
    for (id, b) in bps {
        if let Some(Ok(r)) = roles.get(id) {
            let o = sim_options(b, *r, bps, &roles);
            if !o.is_empty() {
                options.insert(id.clone(), o);
            }
        }
    }
    let mut playable: Vec<BTreeSet<String>> = vec![BTreeSet::new(); FACTION_CATS.len()];
    let mut reached: BTreeSet<String> = BTreeSet::new();
    for c in &commanders {
        let Some(fi) = bps.get(*c).and_then(BpInfo::faction) else {
            continue;
        };
        let mut stack = vec![(*c).clone()];
        while let Some(id) = stack.pop() {
            if !playable[fi].insert(id.clone()) {
                continue;
            }
            reached.insert(id.clone());
            for o in options.get(&id).map(Vec::as_slice).unwrap_or(&[]) {
                if !playable[fi].contains(&o.id) {
                    stack.push(o.id.clone());
                }
            }
        }
    }

    let entries = bps
        .iter()
        .map(|(id, b)| {
            let role = roles.get(id).and_then(|r| r.ok());
            let status = if reached.contains(id) {
                Status::Playable
            } else if tree.contains(id) {
                match roles.get(id) {
                    Some(Err(reason)) => Status::Excluded(reason),
                    _ => Status::Excluded("only built by a unit whose production is not modelled (mobile factory / carrier / egg)"),
                }
            } else {
                Status::NotInTree
            };
            (
                id.clone(),
                RosterEntry {
                    id: id.clone(),
                    name: b.name.clone(),
                    faction: b.faction(),
                    tier: b.tier(),
                    role,
                    group: role.map(|r| BuildGroup::of(r, b)),
                    categories: b.cats.clone(),
                    status,
                },
            )
        })
        .collect();
    // Only keep options of builders that are themselves playable.
    options.retain(|id, _| reached.contains(id));
    Roster { entries, options, playable }
}

/// Coverage counts `[faction][tier-1] = (playable, in FA tree)`, plus totals.
pub fn coverage(r: &Roster) -> Vec<[(usize, usize); 4]> {
    let mut out = vec![[(0usize, 0usize); 4]; FACTION_CATS.len()];
    for e in r.entries.values() {
        let Some(f) = e.faction else {
            continue;
        };
        let t = (e.tier.clamp(1, 4) - 1) as usize;
        match e.status {
            Status::Playable => {
                out[f][t].0 += 1;
                out[f][t].1 += 1;
            }
            Status::Excluded(_) => out[f][t].1 += 1,
            Status::NotInTree => {}
        }
    }
    out
}

fn json_esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

/// `farust roster [GAME_DIR] [--faction F] [--json]`: coverage of the data-driven roster
/// (playable / in FA tree, and what the pre-Phase-24 hand-made tables had), the excluded
/// units with their missing mechanic, and (with `--faction`) every playable unit with
/// its builders.
pub fn cmd_roster(args: &[String]) -> Result<(), String> {
    use crate::skirmish::Catalog;
    if args.iter().any(|a| a == "--verify") {
        return cmd_roster_verify(args);
    }
    let mut game_dir = None;
    let mut only: Option<String> = None;
    let mut json = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--faction" => {
                only = args.get(i + 1).map(|s| s.to_ascii_lowercase());
                i += 1;
            }
            "--json" => json = true,
            a if !a.starts_with("--") => game_dir = Some(std::path::PathBuf::from(a)),
            _ => {}
        }
        i += 1;
    }
    let game_dir = game_dir.unwrap_or_else(crate::scd::default_game_dir);
    let cat = Catalog::load(&game_dir)?;
    let r = &cat.roster;
    let fsel: Option<usize> = match &only {
        Some(f) => Some(
            FACTION_CATS
                .iter()
                .position(|(k, _)| k.starts_with(f.as_str()))
                .ok_or_else(|| format!("unknown faction '{f}' (uef, cybran, aeon, seraphim)"))?,
        ),
        None => None,
    };
    let legacy = |fi: usize, id: &str| {
        cat.factions.get(fi).is_some_and(|f| f.commander == id) || cat.faction_can_build(fi, id) || cat.faction_can_produce(fi, id)
    };
    let mut builders: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (b, opts) in &r.options {
        for o in opts {
            builders.entry(o.id.as_str()).or_default().push(b.as_str());
        }
    }
    let cov = coverage(r);
    let mut before = vec![[0usize; 4]; FACTION_CATS.len()];
    for e in r.entries.values() {
        if let Some(f) = e.faction
            && legacy(f, &e.id)
        {
            before[f][(e.tier.clamp(1, 4) - 1) as usize] += 1;
        }
    }
    let total = r.entries.len();
    let tree = r.entries.values().filter(|e| e.status != Status::NotInTree).count();
    let playable = r.entries.values().filter(|e| e.status == Status::Playable).count();
    let before_total: usize = before.iter().flatten().sum();
    let mods: Vec<String> = cat.mods.iter().map(ToString::to_string).collect();
    let in_sel = |f: Option<usize>| fsel.is_none() || f == fsel;
    let excluded: Vec<&RosterEntry> = r
        .entries
        .values()
        .filter(|e| matches!(e.status, Status::Excluded(_)) && in_sel(e.faction))
        .collect();
    let units: Vec<&RosterEntry> = fsel
        .map(|fi| cat.faction_roster(fi).iter().filter_map(|t| r.entries.get(&t.id)).collect())
        .unwrap_or_default();

    if json {
        let mut o = format!(
            "{{\"blueprints\":{total},\"tree\":{tree},\"playable\":{playable},\"before\":{before_total},\"mods\":[{}],\"factions\":{{",
            mods.iter().map(|m| format!("\"{}\"", json_esc(m))).collect::<Vec<_>>().join(",")
        );
        let fs: Vec<String> = FACTION_CATS
            .iter()
            .enumerate()
            .filter(|(fi, _)| in_sel(Some(*fi)))
            .map(|(fi, (k, _))| {
                let tiers: Vec<String> = (0..4)
                    .map(|t| format!("{{\"tier\":{},\"playable\":{},\"tree\":{},\"before\":{}}}", t + 1, cov[fi][t].0, cov[fi][t].1, before[fi][t]))
                    .collect();
                format!("\"{k}\":[{}]", tiers.join(","))
            })
            .collect();
        o.push_str(&fs.join(","));
        o.push_str("},\"excluded\":[");
        let ex: Vec<String> = excluded
            .iter()
            .map(|e| {
                let reason = match &e.status {
                    Status::Excluded(m) => *m,
                    _ => "",
                };
                format!(
                    "{{\"id\":\"{}\",\"faction\":\"{}\",\"tier\":{},\"name\":\"{}\",\"reason\":\"{}\"}}",
                    e.id,
                    e.faction.map(|f| FACTION_CATS[f].0).unwrap_or("-"),
                    e.tier,
                    json_esc(&e.name),
                    json_esc(reason)
                )
            })
            .collect();
        o.push_str(&ex.join(","));
        o.push_str("],\"units\":[");
        let us: Vec<String> = units
            .iter()
            .map(|e| {
                format!(
                    "{{\"id\":\"{}\",\"faction\":\"{}\",\"tier\":{},\"role\":\"{}\",\"group\":\"{}\",\"name\":\"{}\",\"builders\":[{}]}}",
                    e.id,
                    e.faction.map(|f| FACTION_CATS[f].0).unwrap_or("-"),
                    e.tier,
                    cat.get(&e.id).map(|t| format!("{:?}", t.role)).unwrap_or_default(),
                    cat.build_group(&e.id).map(BuildGroup::label).unwrap_or("-"),
                    json_esc(&e.name),
                    builders.get(e.id.as_str()).map(|b| b.iter().map(|x| format!("\"{x}\"")).collect::<Vec<_>>().join(",")).unwrap_or_default()
                )
            })
            .collect();
        o.push_str(&us.join(","));
        o.push_str("]}");
        println!("{o}");
        return Ok(());
    }

    println!(
        "roster:   {total} unit blueprints | FA skirmish tree {tree} | playable {playable} (before Phase 24: {before_total}) | mods: {}",
        if mods.is_empty() { "none".to_string() } else { mods.join(", ") }
    );
    println!("coverage: playable/in-FA-tree (before Phase 24)");
    println!("  {:<9} {:>14} {:>14} {:>14} {:>14} {:>14}", "faction", "T1", "T2", "T3", "T4", "total");
    let mut col = [(0usize, 0usize, 0usize); 5];
    for (fi, (k, _)) in FACTION_CATS.iter().enumerate() {
        if !in_sel(Some(fi)) {
            continue;
        }
        let mut line = format!("  {k:<9}");
        let (mut p, mut t, mut b) = (0, 0, 0);
        for tier in 0..4 {
            let (cp, ct) = cov[fi][tier];
            let cb = before[fi][tier];
            line.push_str(&format!(" {:>14}", format!("{cp}/{ct} ({cb})")));
            p += cp;
            t += ct;
            b += cb;
            col[tier].0 += cp;
            col[tier].1 += ct;
            col[tier].2 += cb;
        }
        col[4].0 += p;
        col[4].1 += t;
        col[4].2 += b;
        line.push_str(&format!(" {:>14}", format!("{p}/{t} ({b})")));
        println!("{line}");
    }
    if fsel.is_none() {
        let mut line = format!("  {:<9}", "all");
        for c in col {
            line.push_str(&format!(" {:>14}", format!("{}/{} ({})", c.0, c.1, c.2)));
        }
        println!("{line}");
    }
    println!("excluded: {} unit(s) in the FA tree need mechanics FARust does not simulate", excluded.len());
    for e in &excluded {
        if let Status::Excluded(m) = &e.status {
            println!(
                "  {:<8} {:<8} T{} {:<34} {}",
                e.id,
                e.faction.map(|f| FACTION_CATS[f].0).unwrap_or("-"),
                e.tier,
                e.name.chars().take(34).collect::<String>(),
                m
            );
        }
    }
    if !units.is_empty() {
        println!("playable: {} unit(s)", units.len());
        for e in &units {
            let t = cat.get(&e.id);
            println!(
                "  T{} {:<8} {:<12} {:<14} {:<34} built by {}",
                e.tier,
                e.id,
                cat.build_group(&e.id).map(BuildGroup::label).unwrap_or("-"),
                t.map(|t| format!("{:?}", t.role)).unwrap_or_default(),
                e.name.chars().take(34).collect::<String>(),
                builders.get(e.id.as_str()).map(|b| b.join(",")).unwrap_or_else(|| "(start unit)".into())
            );
        }
    }
    Ok(())
}

/// `farust roster --verify [MAP] [--faction F] [--ticks N]`: in a sandbox world per
/// faction (resource-boosted ACU, no AI) spawns one builder per playable unit, issues the
/// data-driven Build / Produce / Upgrade command through the normal command validation,
/// runs the sim and reports which units were accepted and completed.
pub fn cmd_roster_verify(args: &[String]) -> Result<(), String> {
    use crate::sim::{SCALE, TICK_HZ};
    use crate::skirmish::{Catalog, Command, MapContext, Skirmish, UnitId};
    let flag = |f: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1)).cloned();
    let map_name = (2..args.len())
        .find(|&i| !args[i].starts_with("--") && !matches!(args[i - 1].as_str(), "--faction" | "--ticks"))
        .map(|i| args[i].clone())
        .unwrap_or_else(|| "SCMP_009".to_string());
    let only = flag("--faction").map(|s| s.to_ascii_lowercase());
    let max_ticks: u64 = flag("--ticks").and_then(|s| s.parse().ok()).unwrap_or(9000);
    let game_dir = crate::scd::default_game_dir();
    let base = Catalog::load(&game_dir)?;
    let map = MapContext::load(&game_dir, &map_name)?;
    let r = base.roster.clone();
    let mut builders: BTreeMap<&str, Vec<&BuildOption>> = BTreeMap::new();
    let mut opt_of: BTreeMap<(&str, &str), &BuildOption> = BTreeMap::new();
    for (b, opts) in &r.options {
        for o in opts {
            builders.entry(o.id.as_str()).or_default().push(o);
            opt_of.insert((b.as_str(), o.id.as_str()), o);
        }
    }
    let mut grand = (0usize, 0usize, 0usize);
    for (fi, f) in base.factions.iter().enumerate() {
        if only.as_ref().is_some_and(|o| !f.key.starts_with(o.as_str())) {
            continue;
        }
        let Some(ri) = FACTION_CATS.iter().position(|(k, _)| *k == f.key) else {
            continue;
        };
        let mut cat = base.clone();
        if let Some(t) = cat.types.get_mut(&f.commander) {
            t.storage_mass = 50_000_000 * SCALE;
            t.storage_energy = 500_000_000 * SCALE;
            t.mass_per_tick = 200_000 * SCALE / TICK_HZ as i64;
            t.energy_per_tick = 2_000_000 * SCALE / TICK_HZ as i64;
        }
        let other = (fi + 1) % base.factions.len();
        let far = (1..map.starts.len())
            .max_by_key(|&i| {
                let (a, b) = (map.starts[0].1, map.starts[i].1);
                (a.0 - b.0).abs() + (a.1 - b.1).abs()
            })
            .unwrap_or(1);
        let mut w = Skirmish::on_map(cat, &map, &[(0, fi), (far, other)], 1)?.with_options(crate::skirmish::GameOptions {
            victory: crate::skirmish::VictoryCondition::Sandbox,
            ..Default::default()
        });
        let start = w.armies()[0].start;
        // Candidate land / water spots on a lattice around the start, nearest first.
        let (mut land, mut water): (Vec<(i64, i64)>, Vec<(i64, i64)>) = (Vec::new(), Vec::new());
        let step = 26 * SCALE;
        for ring in 1..40i64 {
            for dx in -ring..=ring {
                for dy in -ring..=ring {
                    if dx.abs().max(dy.abs()) != ring {
                        continue;
                    }
                    let p = (start.0 + dx * step, start.1 + dy * step);
                    if p.0 < 20 * SCALE || p.1 < 20 * SCALE || p.0 > map.size.0 - 20 * SCALE || p.1 > map.size.1 - 20 * SCALE {
                        continue;
                    }
                    let g = &map.grid;
                    let (cx, cy) = g.world_to_cell(p.0, p.1);
                    let ng = &map.naval_grid;
                    let (nx, ny) = ng.world_to_cell(p.0, p.1);
                    if !g.is_blocked(cx, cy) {
                        land.push(p);
                    } else if !ng.is_blocked(nx, ny) {
                        water.push(p);
                    }
                }
            }
        }
        land.reverse();
        water.reverse();
        let mut mass = map.mass_points.clone();
        mass.sort_by_key(|m| (m.0 - start.0).abs() + (m.1 - start.1).abs());
        mass.reverse();
        let mut hydro = map.hydro_points.clone();
        hydro.sort_by_key(|m| (m.0 - start.0).abs() + (m.1 - start.1).abs());
        hydro.reverse();

        struct Job {
            target: String,
            builder: UnitId,
            kind: BuildKind,
            accepted: bool,
            note: &'static str,
        }
        let mut jobs: Vec<Job> = Vec::new();
        let mut reserved: Vec<((i64, i64), i64)> = Vec::new();
        let playable: Vec<String> = r.playable.get(ri).map(|s| s.iter().cloned().collect()).unwrap_or_default();
        for x in &playable {
            if *x == f.commander {
                continue;
            }
            let Some(opts) = builders.get(x.as_str()) else {
                continue;
            };
            // Pick a builder of this faction, preferring non-ACUs.
            let mut cands: Vec<(&str, &BuildOption)> = r
                .options
                .iter()
                .filter(|(b, _)| r.playable[ri].contains(*b))
                .filter_map(|(b, _)| opt_of.get(&(b.as_str(), x.as_str())).map(|o| (b.as_str(), *o)))
                .collect();
            cands.sort_by_key(|(b, _)| (*b == f.commander, base.get(b).map(|t| t.tier).unwrap_or(9), *b));
            let Some(&(b, o)) = cands.first() else {
                let _ = opts;
                continue;
            };
            let tgt = base.get(x);
            let btype = base.get(b);
            let needs_water = tgt.is_some_and(|t| t.ext.water_only || (t.domain == NavDomain::Naval && o.kind == BuildKind::Build))
                || btype.is_some_and(|t| t.ext.water_only || t.role == Role::NavalFactory);
            let tgt_role = tgt.map(|t| t.role);
            let spot = if o.kind == BuildKind::Upgrade && matches!(btype.map(|t| t.role), Some(Role::Extractor)) {
                mass.pop()
            } else if o.kind == BuildKind::Upgrade && matches!(btype.map(|t| t.role), Some(Role::Hydro)) {
                hydro.pop()
            } else if o.kind == BuildKind::Build && tgt_role == Some(Role::Extractor) {
                mass.pop().map(|m| (m.0 + 3 * SCALE, m.1))
            } else if o.kind == BuildKind::Build && tgt_role == Some(Role::Hydro) {
                hydro.pop().map(|m| (m.0 + 4 * SCALE, m.1))
            } else if needs_water {
                water.pop()
            } else {
                land.pop()
            };
            let Some(p) = spot else {
                jobs.push(Job { target: x.clone(), builder: 0, kind: o.kind, accepted: false, note: "no free spot on this map" });
                continue;
            };
            let Some(bid) = w.spawn_unit(0, b, p) else {
                jobs.push(Job { target: x.clone(), builder: 0, kind: o.kind, accepted: false, note: "builder spawn failed" });
                continue;
            };
            let cmd = match o.kind {
                BuildKind::Build => {
                    let (tx, ty) = match tgt_role {
                        Some(Role::Extractor) | Some(Role::Hydro) => (p.0 - 3 * SCALE, p.1),
                        _ => {
                            // First free site on a small spiral around the builder that no
                            // other queued job reserved.
                            let r = tgt.map(|t| t.radius).unwrap_or(SCALE);
                            let mut site = (p.0, p.1 + 10 * SCALE);
                            'search: for ring in 0..8i64 {
                                for (dx, dy) in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)] {
                                    let c = (p.0 + dx * (6 + 4 * ring) * SCALE, p.1 + dy * (6 + 4 * ring) * SCALE);
                                    let clear = reserved.iter().all(|&(q, qr): &((i64, i64), i64)| {
                                        let d = (r + qr + 2 * SCALE) as i128;
                                        let (ex, ey) = ((q.0 - c.0) as i128, (q.1 - c.1) as i128);
                                        ex * ex + ey * ey >= d * d
                                    });
                                    if clear && w.site_is_free(x, c.0, c.1) {
                                        site = c;
                                        break 'search;
                                    }
                                }
                            }
                            reserved.push((site, r));
                            site
                        }
                    };
                    Command::Build { builder: bid, unit_type: x.clone(), x: tx, y: ty, queue: false }
                }
                BuildKind::Produce => Command::Produce { factory: bid, unit_type: x.clone(), count: 1 },
                BuildKind::Upgrade => Command::Upgrade { unit: bid, target_type: x.clone() },
            };
            let accepted = w.apply_command_direct(0, cmd);
            if accepted && (o.tier >= 4 || (o.tier >= 3 && o.kind == BuildKind::Build)) {
                let helpers: Vec<UnitId> = (0..8).filter_map(|k| w.spawn_unit(0, b, (p.0 + (k - 4) * 2 * SCALE, p.1 - 6 * SCALE))).collect();
                w.apply_command_direct(0, Command::Assist { units: helpers, target: bid, queue: false });
            }
            jobs.push(Job { target: x.clone(), builder: bid, kind: o.kind, accepted, note: if accepted { "" } else { "command rejected" } });
        }
        let initial: std::collections::BTreeSet<UnitId> = w.units().map(|u| u.id).collect();
        let done_of = |w: &Skirmish, j: &Job| -> bool {
            match j.kind {
                BuildKind::Upgrade => w.unit(j.builder).is_some_and(|u| u.type_id.eq_ignore_ascii_case(&j.target)),
                _ => w.units().any(|u| u.army == 0 && u.type_id == j.target && !initial.contains(&u.id)),
            }
        };
        let mut done = vec![false; jobs.len()];
        let t0 = std::time::Instant::now();
        while w.tick() < max_ticks {
            if w.result().is_some() {
                break;
            }
            w.run(100);
            for (k, j) in jobs.iter().enumerate() {
                if !done[k] && j.accepted && done_of(&w, j) {
                    done[k] = true;
                }
            }
            if jobs.iter().zip(&done).all(|(j, d)| *d || !j.accepted) {
                break;
            }
        }
        let acc = jobs.iter().filter(|j| j.accepted).count();
        let comp = done.iter().filter(|d| **d).count();
        println!(
            "verify {:<8} on {}: {} units | accepted {}/{} | completed {}/{} in {} ticks ({:.1}s) | alive units {}",
            f.key,
            map.map_id,
            jobs.len(),
            acc,
            jobs.len(),
            comp,
            jobs.len(),
            w.tick(),
            t0.elapsed().as_secs_f32(),
            w.units().filter(|u| u.alive && u.army == 0).count()
        );
        for (k, j) in jobs.iter().enumerate() {
            if !done[k] {
                let t = base.get(&j.target);
                println!(
                    "    not completed: {:<8} T{} {:?} ({}){}",
                    j.target,
                    t.map(|t| t.tier).unwrap_or(0),
                    j.kind,
                    t.map(|t| t.name.as_str()).unwrap_or("-"),
                    if j.note.is_empty() { String::new() } else { format!(" - {}", j.note) }
                );
            }
        }
        grand.0 += jobs.len();
        grand.1 += acc;
        grand.2 += comp;
    }
    println!("verify total: {} units | accepted {} | completed {}", grand.0, grand.1, grand.2);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::parse_blueprint;

    fn bp(id: &str, src: &str) -> (String, BpInfo) {
        let root = parse_blueprint(src).unwrap();
        (id.to_string(), BpInfo::from_blueprint(id, &root))
    }

    fn synthetic() -> BTreeMap<String, BpInfo> {
        [
            bp("TSL0001", "UnitBlueprint { Categories = {'UEF','MOBILE','LAND','COMMANDER','ENGINEER'}, Economy = { BuildableCategory = {'BUILTBYCOMMANDER UEF'} }, Enhancements = { T2 = { BuildableCategoryAdds = 'BUILTBYTIER2COMMANDER UEF' } } }"),
            bp("TSB0101", "UnitBlueprint { Categories = {'UEF','STRUCTURE','LAND','FACTORY','TECH1','BUILTBYCOMMANDER'}, Economy = { BuildableCategory = {'BUILTBYTIER1FACTORY UEF MOBILE'} }, General = { UpgradesTo = 'tsb0201' } }"),
            bp("TSB0201", "UnitBlueprint { Categories = {'UEF','STRUCTURE','LAND','FACTORY','TECH2'}, Economy = { BuildableCategory = {'BUILTBYTIER2FACTORY UEF MOBILE'} } }"),
            bp("TSL0105", "UnitBlueprint { Categories = {'UEF','MOBILE','LAND','ENGINEER','CONSTRUCTION','TECH1','BUILTBYTIER1FACTORY','BUILTBYTIER2FACTORY'}, Economy = { BuildableCategory = {'BUILTBYTIER1ENGINEER UEF'} } }"),
            bp("TSL0201", "UnitBlueprint { Categories = {'UEF','MOBILE','LAND','DIRECTFIRE','TECH1','BUILTBYTIER1FACTORY','BUILTBYTIER2FACTORY'} }"),
            bp("TSA0101", "UnitBlueprint { Categories = {'UEF','MOBILE','AIR','TECH1','BUILTBYTIER1FACTORY'}, Physics = { MotionType = 'RULEUMT_Air' } }"),
            bp("TSL0202", "UnitBlueprint { Categories = {'UEF','MOBILE','LAND','DIRECTFIRE','TECH2','BUILTBYTIER2FACTORY'} }"),
            bp("TSB1101", "UnitBlueprint { Categories = {'UEF','STRUCTURE','ENERGYPRODUCTION','TECH1','BUILTBYTIER1ENGINEER','BUILTBYCOMMANDER'} }"),
            bp("TSB5101", "UnitBlueprint { Categories = {'UEF','STRUCTURE','WALL','TECH1','BUILTBYTIER1ENGINEER'} }"),
            bp("TSB2301", "UnitBlueprint { Categories = {'UEF','STRUCTURE','DIRECTFIRE','TECH2','BUILTBYTIER2COMMANDER'} }"),
            bp("TSL0401", "UnitBlueprint { Categories = {'UEF','MOBILE','LAND','EXPERIMENTAL','FACTORY','BUILTBYTIER2COMMANDER'}, Economy = { BuildableCategory = {'tsl0999'} } }"),
            bp("TSL0999", "UnitBlueprint { Categories = {'UEF','MOBILE','LAND','TECH1'} }"),
            bp("TSC0001", "UnitBlueprint { Categories = {'UEF','STRUCTURE','CIVILIAN'} }"),
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn matches_category_expressions() {
        let cats: BTreeSet<String> = ["UEF", "MOBILE", "LAND"].iter().map(|s| s.to_string()).collect();
        assert!(expr_matches("UEF MOBILE", &cats));
        assert!(!expr_matches("UEF MOBILE AIR", &cats));
        assert!(!expr_matches("  ", &cats));
    }

    #[test]
    fn derives_synthetic_roster() {
        let r = derive(&synthetic(), &BTreeMap::new());
        let st = |id: &str| r.entries[id].status.clone();
        for id in ["TSL0001", "TSB0101", "TSB0201", "TSL0105", "TSL0201", "TSL0202", "TSB1101", "TSB2301", "TSL0401"] {
            assert_eq!(st(id), Status::Playable, "{id}");
        }
        // Air unit is listed by the land factory in FA but needs an air factory here.
        assert!(matches!(st("TSA0101"), Status::Excluded(_)));
        assert_eq!(st("TSB5101"), Status::Playable, "Phase 28: walls are playable");
        assert!(matches!(st("TSL0999"), Status::Excluded(m) if m.contains("mobile factory")));
        assert_eq!(st("TSC0001"), Status::NotInTree);

        let fac = r.options_of("TSB0101");
        assert!(fac.iter().any(|o| o.id == "TSL0105" && o.kind == BuildKind::Produce && o.group == BuildGroup::Engineer));
        assert!(fac.iter().any(|o| o.id == "TSB0201" && o.kind == BuildKind::Upgrade && o.tier == 2));
        assert!(!fac.iter().any(|o| o.id == "TSL0202"), "T2 unit needs the T2 factory");
        assert!(r.can("TSB0201", "TSL0202", BuildKind::Produce));
        let acu = r.options_of("TSL0001");
        assert!(acu.iter().any(|o| o.id == "TSB2301" && o.tier == 2 && o.group == BuildGroup::Defense));
        assert!(acu.iter().any(|o| o.id == "TSL0401" && o.group == BuildGroup::Experimental));
        assert!(!r.can("TSL0105", "TSB2301", BuildKind::Build));
        assert_eq!(r.playable[0].len(), 10);
        let cov = coverage(&r);
        assert_eq!(cov[0][0], (6, 8));
        assert_eq!(cov[0][3], (1, 1));
    }

    #[test]
    fn fixed_roles_override_inference() {
        let mut fixed = BTreeMap::new();
        fixed.insert("TSB5101".to_string(), Role::MassStorage);
        let r = derive(&synthetic(), &fixed);
        assert_eq!(r.entries["TSB5101"].status, Status::Playable);
        assert_eq!(r.entries["TSB5101"].role, Some(Role::MassStorage));
    }
}
