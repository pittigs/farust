//! Phase 23 FA command interface: blueprint-driven command card, factory queue panel,
//! upgrade / enhancement panels, order buttons with targeting cursor modes, camera
//! bookmarks and the HUD unit-icon atlas.
//!
//! Everything is derived from the [`Catalog`] (roles, tiers, domains, costs, upgrade
//! chains, enhancements) — no unit ids are hard-coded — and every action reaches the
//! simulation only through ordinary [`Command`]s (deterministic replays / lockstep).

use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

use crate::camera::OrbitCamera;
use crate::dds_decode::{RgbaImage, decode_dds_rgba8, resize_rgba_bilinear};
use crate::flowfield::NavDomain;
use crate::keymap::CARD_SLOTS;
use crate::play::PlaySession;
use crate::sim::{SCALE, TICK_HZ};
use crate::skirmish::{Catalog, Command, EnhancementSpec, Role, UnitId, UnitOrder, UnitType};

// ---------------------------------------------------------------------------
// Card model (pure)
// ---------------------------------------------------------------------------

/// Category group of a card entry (FA construction / factory tabs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CardGroup {
    Factories,
    Economy,
    Defense,
    Intel,
    Shields,
    Missiles,
    Experimental,
    Engineers,
    Land,
    Air,
    Naval,
}

impl CardGroup {
    pub fn label(self) -> &'static str {
        match self {
            CardGroup::Factories => "FAC",
            CardGroup::Economy => "ECO",
            CardGroup::Defense => "DEF",
            CardGroup::Intel => "INT",
            CardGroup::Shields => "SHD",
            CardGroup::Missiles => "MSL",
            CardGroup::Experimental => "EXP",
            CardGroup::Engineers => "ENG",
            CardGroup::Land => "LAND",
            CardGroup::Air => "AIR",
            CardGroup::Naval => "NAVY",
        }
    }

    pub fn tooltip(self) -> (String, String) {
        let (key, fb_t, fb_d) = match self {
            CardGroup::Factories => ("factories", "Factories", "Filter production factories."),
            CardGroup::Economy => ("economy", "Economy", "Filter mass/energy economy structures."),
            CardGroup::Defense => ("defense", "Defense", "Filter defensive structures and walls."),
            CardGroup::Intel => ("intel", "Intelligence", "Filter radar, omni, and stealth structures."),
            CardGroup::Shields => ("shields", "Shields", "Filter shield generators."),
            CardGroup::Missiles => ("missiles", "Missiles", "Filter missile launchers and defenses."),
            CardGroup::Experimental => ("experimental", "Experimental", "Filter Tech 4 experimental units."),
            CardGroup::Engineers => ("engineers", "Engineers", "Filter engineer units."),
            CardGroup::Land => ("land", "Land Units", "Filter ground combat units."),
            CardGroup::Air => ("air", "Air Units", "Filter aircraft and transports."),
            CardGroup::Naval => ("naval", "Naval Units", "Filter naval vessels."),
        };
        (
            crate::loc::tr(&format!("group.{key}.title"), fb_t),
            crate::loc::tr(&format!("group.{key}.desc"), fb_d),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardSource {
    /// Engineers / commanders: structures to place.
    Construction,
    /// Factories: units to queue.
    Production,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardItem {
    pub unit_type: String,
    /// 1..=3, 4 = experimental.
    pub tier: u8,
    pub group: CardGroup,
}

/// Construction category of a structure (or experimental) blueprint.
pub fn structure_group(t: &UnitType) -> CardGroup {
    if t.tier >= 4 || t.role == Role::Experimental {
        return CardGroup::Experimental;
    }
    match t.role {
        Role::Factory | Role::AirFactory | Role::NavalFactory => CardGroup::Factories,
        Role::Extractor | Role::Generator | Role::Hydro | Role::MassStorage | Role::EnergyStorage => CardGroup::Economy,
        Role::Radar => CardGroup::Intel,
        Role::Shield => CardGroup::Shields,
        Role::MissileSilo | Role::MissileDefense => CardGroup::Missiles,
        _ if t.intercept_radius > 0 || t.missile_range > 0 => CardGroup::Missiles,
        _ if t.radar_radius > 0 || t.ext.intel.omni_radius > 0 || t.ext.intel.stealth_field_radius > 0 => CardGroup::Intel,
        _ if t.shield_max_hp > 0 => CardGroup::Shields,
        _ => CardGroup::Defense,
    }
}

/// Factory tab of a mobile unit blueprint.
pub fn production_group(t: &UnitType) -> CardGroup {
    if t.role.is_builder() {
        return CardGroup::Engineers;
    }
    match t.domain {
        NavDomain::Air => CardGroup::Air,
        NavDomain::Naval => CardGroup::Naval,
        NavDomain::Land | NavDomain::Amphibious => CardGroup::Land,
    }
}

/// Whether a factory of `role`/`tier` can produce `t` (domain and tech level).
pub fn factory_accepts(role: Role, tier: u8, t: &UnitType) -> bool {
    if t.tier > tier.max(1) || t.tier >= 4 || t.role.is_structure() {
        return false;
    }
    match role {
        Role::Factory => matches!(t.domain, NavDomain::Land | NavDomain::Amphibious),
        Role::AirFactory => t.domain == NavDomain::Air,
        Role::NavalFactory => t.domain == NavDomain::Naval,
        _ => false,
    }
}

/// Whether a builder with engineering tier `max_tier` may build tier `tier`
/// (experimentals need T3 engineering).
pub fn builder_accepts(max_tier: u8, tier: u8) -> bool {
    tier <= max_tier.max(1) || (tier >= 4 && max_tier >= 3)
}

fn sort_items(cat: &Catalog, items: &mut [CardItem]) {
    items.sort_by(|a, b| {
        let ta = cat.get(&a.unit_type);
        let tb = cat.get(&b.unit_type);
        let key = |t: Option<&UnitType>, it: &CardItem| {
            (
                it.group,
                it.tier,
                t.map(|t| t.role.code()).unwrap_or(u8::MAX),
                t.map(|t| t.cost_mass).unwrap_or(i64::MAX),
            )
        };
        key(ta, a).cmp(&key(tb, b)).then_with(|| a.unit_type.cmp(&b.unit_type))
    });
}

/// Whether `builder` (given `faction`) may place structure / experimental `t`
/// — matches [`crate::skirmish::Skirmish`] `Command::Build` validation.
pub fn unit_can_build(cat: &Catalog, faction: usize, builder: &crate::skirmish::Unit, t: &UnitType) -> bool {
    if builder.role == Role::Commander && t.tier > builder_tier(builder) {
        return false;
    }
    if !cat.roster.options.is_empty() && cat.roster.options.contains_key(&builder.type_id) {
        cat.builder_can_build(&builder.type_id, &t.id)
            || (cat.faction_can_build(faction, &t.id) && builder_accepts(builder_tier(builder), t.tier))
    } else {
        cat.faction_can_build(faction, &t.id) && builder_accepts(builder_tier(builder), t.tier)
    }
}

/// Whether `factory` (given `faction`) may queue mobile unit `t`
/// — matches [`crate::skirmish::Skirmish`] `Command::Produce` validation (including
/// Quantum Gates and mod-added units when `roster.options` is populated).
pub fn unit_can_produce(cat: &Catalog, faction: usize, factory: &crate::skirmish::Unit, t: &UnitType) -> bool {
    if cat.roster.options.contains_key(&factory.type_id) {
        cat.builder_can_produce(&factory.type_id, &t.id)
    } else {
        cat.faction_can_produce(faction, &t.id) && factory_accepts(factory.role, factory.tier, t)
    }
}

/// Structures faction `faction` can build with engineering tier `max_tier`.
#[allow(dead_code)]
pub fn construction_options(cat: &Catalog, faction: usize, max_tier: u8) -> Vec<CardItem> {
    let mut out: Vec<CardItem> = cat
        .types
        .values()
        .filter(|t| cat.faction_can_build(faction, &t.id) && builder_accepts(max_tier, t.tier))
        .map(|t| CardItem { unit_type: t.id.clone(), tier: t.tier.clamp(1, 4), group: structure_group(t) })
        .collect();
    sort_items(cat, &mut out);
    out
}

/// Structures the given live builders of `faction` can place (union across selected builders).
pub fn construction_options_for_units(
    cat: &Catalog,
    faction: usize,
    builders: &[&crate::skirmish::Unit],
) -> Vec<CardItem> {
    let mut out: Vec<CardItem> = cat
        .types
        .values()
        .filter(|t| builders.iter().any(|b| unit_can_build(cat, faction, b, t)))
        .map(|t| CardItem { unit_type: t.id.clone(), tier: t.tier.clamp(1, 4), group: structure_group(t) })
        .collect();
    sort_items(cat, &mut out);
    out
}

/// Units the given factories (`(role, tier)`) of `faction` can produce (union).
#[allow(dead_code)]
pub fn production_options(cat: &Catalog, faction: usize, factories: &[(Role, u8)]) -> Vec<CardItem> {
    let mut out: Vec<CardItem> = cat
        .types
        .values()
        .filter(|t| {
            cat.faction_can_produce(faction, &t.id) && factories.iter().any(|&(r, tier)| factory_accepts(r, tier, t))
        })
        .map(|t| CardItem { unit_type: t.id.clone(), tier: t.tier.clamp(1, 3), group: production_group(t) })
        .collect();
    sort_items(cat, &mut out);
    out
}

/// Units the given live factories (or Quantum Gates) of `faction` can produce (union).
pub fn production_options_for_units(
    cat: &Catalog,
    faction: usize,
    factories: &[&crate::skirmish::Unit],
) -> Vec<CardItem> {
    let mut out: Vec<CardItem> = cat
        .types
        .values()
        .filter(|t| factories.iter().any(|f| unit_can_produce(cat, faction, f, t)))
        .map(|t| CardItem { unit_type: t.id.clone(), tier: t.tier.clamp(1, 4), group: production_group(t) })
        .collect();
    sort_items(cat, &mut out);
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardModel {
    pub source: Option<CardSource>,
    /// Entries per tech tab (T1, T2, T3, EXP).
    pub tab_counts: [usize; 4],
    /// Effective tab `1..=4`.
    pub tab: u8,
    /// Groups present in the current tab (display order).
    pub groups: Vec<CardGroup>,
    /// Active group filter (`None` = all).
    pub group: Option<CardGroup>,
    /// Entries of the tab after the group filter (all pages).
    pub items: Vec<CardItem>,
    pub page: usize,
    pub pages: usize,
}

impl CardModel {
    pub fn empty() -> Self {
        Self { source: None, tab_counts: [0; 4], tab: 1, groups: Vec::new(), group: None, items: Vec::new(), page: 0, pages: 1 }
    }

    /// Builds the visible card from all entries and the requested tab / group / page; an
    /// empty requested tab falls back to the first non-empty one, a missing group to "all".
    pub fn build(all: Vec<CardItem>, source: CardSource, req_tab: u8, req_group: Option<CardGroup>, req_page: usize) -> Self {
        let mut tab_counts = [0usize; 4];
        for it in &all {
            tab_counts[(it.tier.clamp(1, 4) - 1) as usize] += 1;
        }
        let tab = if (1..=4).contains(&req_tab) && tab_counts[(req_tab - 1) as usize] > 0 {
            req_tab
        } else {
            tab_counts.iter().position(|&c| c > 0).map(|i| i as u8 + 1).unwrap_or(1)
        };
        let in_tab: Vec<CardItem> = all.into_iter().filter(|it| it.tier.clamp(1, 4) == tab).collect();
        let mut groups: Vec<CardGroup> = in_tab.iter().map(|it| it.group).collect();
        groups.sort();
        groups.dedup();
        let group = req_group.filter(|g| groups.contains(g));
        let items: Vec<CardItem> = in_tab.into_iter().filter(|it| group.is_none_or(|g| it.group == g)).collect();
        let pages = items.len().div_ceil(CARD_SLOTS).max(1);
        Self { source: Some(source), tab_counts, tab, groups, group, items, page: req_page.min(pages - 1), pages }
    }

    /// Entries on the current page (at most [`CARD_SLOTS`]).
    pub fn page_items(&self) -> &[CardItem] {
        let start = (self.page * CARD_SLOTS).min(self.items.len());
        let end = (start + CARD_SLOTS).min(self.items.len());
        &self.items[start..end]
    }

    /// Next non-empty tab after the current one (wrapping).
    pub fn next_tab(&self) -> u8 {
        for k in 1..=4u8 {
            let t = (self.tab - 1 + k) % 4 + 1;
            if self.tab_counts[(t - 1) as usize] > 0 {
                return t;
            }
        }
        self.tab
    }
}

pub fn tab_label(tab: u8) -> &'static str {
    match tab {
        1 => "T1",
        2 => "T2",
        3 => "T3",
        _ => "EXP",
    }
}

/// Tooltip lines for a card entry built at `build_rate` (work per tick, 0 = unknown).
/// Localized via [`crate::loc`] (unit name, FA description, cost/build time, and stats).
pub fn item_tooltip(cat: &Catalog, unit_type: &str, build_rate: i64) -> Vec<String> {
    let Some(t) = cat.get(unit_type) else {
        return vec![unit_type.to_string()];
    };
    let role_name = crate::loc::unit_display_name(&t.id, &t.name);
    let display = match crate::loc::unit_proper_name(&t.id) {
        Some(proper) if !proper.eq_ignore_ascii_case(&role_name) => format!("{proper} - {role_name}"),
        _ => role_name,
    };
    let tier = if t.tier >= 4 { "EXP".to_string() } else { format!("T{}", t.tier.max(1)) };
    let mut out = vec![
        format!("{display} ({tier})"),
        format!("{}  {:?}", t.id, t.role),
    ];
    let l_mass = crate::loc::tr("tt.mass", "Mass");
    let l_energy = crate::loc::tr("tt.energy", "Energy");
    let l_time = crate::loc::tr("tt.time", "Time");
    let mut cost = format!("{l_mass} {}  {l_energy} {}", t.cost_mass / SCALE, t.cost_energy / SCALE);
    if build_rate > 0 {
        let secs = t.build_work as f64 / build_rate as f64 / TICK_HZ as f64;
        cost.push_str(&format!("  {l_time} {secs:.0}s"));
    }
    out.push(cost);
    let dps: f64 = t
        .weapons
        .iter()
        .map(|w| (w.damage as f64 / SCALE as f64) * TICK_HZ as f64 / w.reset.max(1) as f64)
        .sum();
    let l_hp = crate::loc::tr("tt.hp", "HP");
    if dps > 0.0 {
        let l_dps = crate::loc::tr("tt.dps", "DPS");
        let l_rng = crate::loc::tr("tt.range", "Range");
        out.push(format!("{l_hp} {}  {l_dps} {:.0}  {l_rng} {}", t.max_hp / SCALE, dps, t.max_range() / SCALE));
    } else {
        out.push(format!("{l_hp} {}", t.max_hp / SCALE));
    }
    let mut extra = Vec::new();
    let per_s = |v: i64| v as f64 * TICK_HZ as f64 / SCALE as f64;
    if t.mass_per_tick > 0 {
        extra.push(format!("+{:.1} {l_mass}/s", per_s(t.mass_per_tick)));
    }
    if t.energy_per_tick > 0 {
        extra.push(format!("+{:.0} {l_energy}/s", per_s(t.energy_per_tick)));
    }
    if t.shield_max_hp > 0 {
        let l_shd = crate::loc::tr("tt.shield", "Shield");
        extra.push(format!("{l_shd} {} R{}", t.shield_max_hp / SCALE, t.shield_radius / SCALE));
    }
    if t.radar_radius > 0 {
        let l_rad = crate::loc::tr("tt.radar", "Radar");
        extra.push(format!("{l_rad} {}", t.radar_radius / SCALE));
    }
    if t.build_rate > 0 && !t.role.is_factory() {
        let l_br = crate::loc::tr("tt.buildrate", "Buildrate");
        extra.push(format!("{l_br} {}", t.build_rate));
    }
    if let Some(tr_spec) = &t.ext.transport {
        let l_cargo = crate::loc::tr("tt.cargo", "Cargo");
        extra.push(format!("{l_cargo} {}", tr_spec.slots()));
    }
    if t.missile_range > 0 {
        let l_msl = crate::loc::tr("tt.missile_range", "Missile Range");
        extra.push(format!("{l_msl} {}", t.missile_range / SCALE));
    }
    if !extra.is_empty() {
        out.push(extra.join("  "));
    }
    if let Some(desc) = crate::loc::unit_description(&t.id) {
        for line in crate::loc::word_wrap(&desc, 48).into_iter().take(4) {
            out.push(line);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Factory queue / enhancement models (pure)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueEntry {
    pub unit_type: String,
    pub count: u32,
    /// Per-mille progress of the in-progress unit (first entry only).
    pub progress: Option<i64>,
}

/// Groups a factory queue into runs; the in-progress job (if any) leads and absorbs a
/// following run of the same type.
pub fn queue_entries<'a>(active: Option<(&str, i64)>, production: impl IntoIterator<Item = &'a String>) -> Vec<QueueEntry> {
    let mut out: Vec<QueueEntry> = Vec::new();
    if let Some((t, p)) = active {
        out.push(QueueEntry { unit_type: t.to_string(), count: 1, progress: Some(p.clamp(0, 1000)) });
    }
    for t in production {
        match out.last_mut() {
            Some(e) if e.unit_type == *t => e.count += 1,
            _ => out.push(QueueEntry { unit_type: t.clone(), count: 1, progress: None }),
        }
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnhState {
    Installed,
    Available,
    /// Needs its prerequisite (or a free slot).
    Locked,
    /// Being installed (per-mille).
    InProgress(i64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnhRow {
    pub name: String,
    pub display: String,
    pub slot: String,
    pub prerequisite: Option<String>,
    pub cost_mass: i64,
    pub cost_energy: i64,
    pub build_work: i64,
    pub state: EnhState,
}

fn slot_rank(slot: &str) -> u8 {
    match slot.to_ascii_uppercase().as_str() {
        "LCH" => 0,
        "RCH" => 1,
        "BACK" => 2,
        _ => 3,
    }
}

pub fn slot_label(slot: &str) -> &'static str {
    match slot_rank(slot) {
        0 => "LEFT ARM",
        1 => "RIGHT ARM",
        2 => "BACK",
        _ => "OTHER",
    }
}

pub fn localized_slot_label(slot: &str) -> String {
    match slot_rank(slot) {
        0 => crate::loc::tr("tt.slot_lch", "Left Arm"),
        1 => crate::loc::tr("tt.slot_rch", "Right Arm"),
        2 => crate::loc::tr("tt.slot_back", "Back"),
        _ => crate::loc::tr("tt.slot_other", "Other"),
    }
}

/// Enhancement rows (removals excluded), ordered by slot (LCH, RCH, Back, …), then by
/// prerequisite depth, then name.
pub fn enhancement_rows(specs: &[EnhancementSpec], installed: &[(String, String)], active: Option<(&str, i64)>) -> Vec<EnhRow> {
    let prereq = |n: &str| specs.iter().find(|s| s.name == n).and_then(|s| s.prerequisite.clone());
    let depth = |n: &str| {
        let mut d = 0;
        let mut cur = prereq(n);
        while let Some(p) = cur {
            d += 1;
            if d > 16 {
                break;
            }
            cur = prereq(&p);
        }
        d
    };
    let mut rows: Vec<EnhRow> = specs
        .iter()
        .filter(|s| !s.is_removal())
        .map(|s| {
            let cur = installed.iter().find(|(sl, _)| *sl == s.slot).map(|(_, n)| n.as_str());
            // Everything on the installed chain counts as installed.
            let mut chain = Vec::new();
            let mut c = cur.map(str::to_string);
            while let Some(n) = c {
                if chain.len() > 16 {
                    break;
                }
                c = prereq(&n);
                chain.push(n);
            }
            let state = if let Some((an, p)) = active.filter(|(an, _)| *an == s.name) {
                let _ = an;
                EnhState::InProgress(p.clamp(0, 1000))
            } else if chain.iter().any(|n| *n == s.name) {
                EnhState::Installed
            } else {
                match (cur, s.prerequisite.as_deref()) {
                    (None, None) => EnhState::Available,
                    (Some(c), Some(p)) if c == p => EnhState::Available,
                    _ => EnhState::Locked,
                }
            };
            EnhRow {
                name: s.name.clone(),
                display: if s.display.trim().is_empty() { s.name.clone() } else { s.display.trim().to_string() },
                slot: s.slot.clone(),
                prerequisite: s.prerequisite.clone(),
                cost_mass: s.cost_mass,
                cost_energy: s.cost_energy,
                build_work: s.build_work,
                state,
            }
        })
        .collect();
    rows.sort_by(|a, b| {
        (slot_rank(&a.slot), a.slot.as_str(), depth(&a.name), a.name.as_str()).cmp(&(
            slot_rank(&b.slot),
            b.slot.as_str(),
            depth(&b.name),
            b.name.as_str(),
        ))
    });
    rows
}

// ---------------------------------------------------------------------------
// Orders & camera bookmarks (pure)
// ---------------------------------------------------------------------------

/// Order / system buttons of the orders panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderKind {
    Move,
    Attack,
    Patrol,
    Stop,
    Assist,
    Guard,
    Repair,
    Reclaim,
    Load,
    Unload,
    Overcharge,
    LaunchMissile,
    ToggleShield,
    ToggleRepeat,
    Pause,
    Fog,
    Ai,
    IdleEngineer,
    /// Phase 35: fire at a map position.
    AttackGround,
    /// Phase 35: cycle fire at will → return fire → hold fire.
    FireState,
    /// Phase 35: pause / resume construction & production.
    PauseBuild,
}

impl OrderKind {
    pub fn label(self) -> &'static str {
        match self {
            OrderKind::Move => "MOVE",
            OrderKind::Attack => "ATTACK",
            OrderKind::Patrol => "PATROL",
            OrderKind::Stop => "STOP",
            OrderKind::Assist => "ASSIST",
            OrderKind::Guard => "GUARD",
            OrderKind::Repair => "REPAIR",
            OrderKind::Reclaim => "RECLAIM",
            OrderKind::Load => "LOAD",
            OrderKind::Unload => "UNLOAD",
            OrderKind::Overcharge => "OVERCHG",
            OrderKind::LaunchMissile => "MISSILE",
            OrderKind::ToggleShield => "SHIELD",
            OrderKind::ToggleRepeat => "REPEAT",
            OrderKind::Pause => "PAUSE",
            OrderKind::Fog => "FOG",
            OrderKind::Ai => "AI",
            OrderKind::IdleEngineer => "IDLE ENG",
            OrderKind::AttackGround => "GROUND",
            OrderKind::FireState => "FIRE",
            OrderKind::PauseBuild => "HOLD BLD",
        }
    }

    /// Tooltip ID in FA's `lua/ui/help/tooltips.lua` (and FARust's `loc` fallback table).
    pub fn tooltip_id(self) -> &'static str {
        match self {
            OrderKind::Move => "move",
            OrderKind::Attack => "attack",
            OrderKind::Patrol => "patrol",
            OrderKind::Stop => "stop",
            OrderKind::Assist => "assist",
            OrderKind::Guard => "guard",
            OrderKind::Repair => "repair",
            OrderKind::Reclaim => "reclaim",
            OrderKind::Load => "load",
            OrderKind::Unload => "transport",
            OrderKind::Overcharge => "overcharge",
            OrderKind::LaunchMissile => "fire_tactical",
            OrderKind::ToggleShield => "toggle_shield",
            OrderKind::ToggleRepeat => "toggle_repeat",
            OrderKind::Pause => "pause",
            OrderKind::Fog => "fog",
            OrderKind::Ai => "ai",
            OrderKind::IdleEngineer => "idle_engineer",
            OrderKind::AttackGround => "attack_ground",
            OrderKind::FireState => "fire_state",
            OrderKind::PauseBuild => "pause_construction",
        }
    }

    /// Localized `(title, help_text)` for hover tooltips.
    pub fn tooltip(self) -> (String, String) {
        crate::loc::order_tooltip(self.tooltip_id(), self.label(), "")
    }

    /// Compact localized label for the 66px orders-panel button.
    pub fn short_label(self) -> String {
        let key = format!("order.{}.short", self.tooltip_id());
        let short = crate::loc::tr(&key, "");
        if !short.is_empty() {
            return short;
        }
        if crate::loc::active_language() == "us" {
            return self.label().to_string();
        }
        let (title, _) = self.tooltip();
        let trimmed = title.trim();
        if trimmed.is_empty() {
            return self.label().to_string();
        }
        trimmed.chars().take(8).collect::<String>().to_uppercase()
    }

    /// Whether the order waits for a map click (cursor mode).
    pub fn is_targeted(self) -> bool {
        matches!(
            self,
            OrderKind::Move
                | OrderKind::Attack
                | OrderKind::Patrol
                | OrderKind::Assist
                | OrderKind::Guard
                | OrderKind::Repair
                | OrderKind::Reclaim
                | OrderKind::Load
                | OrderKind::Unload
                | OrderKind::Overcharge
                | OrderKind::LaunchMissile
                | OrderKind::AttackGround
        )
    }

    /// Key binding shown on the button.
    pub fn key_action(self) -> Option<crate::keymap::Action> {
        use crate::keymap::Action as A;
        Some(match self {
            OrderKind::Move => A::OrderMove,
            OrderKind::Attack => A::AttackMove,
            OrderKind::Patrol => A::OrderPatrol,
            OrderKind::Stop => A::Stop,
            OrderKind::Assist => A::OrderAssist,
            OrderKind::Guard => A::OrderGuard,
            OrderKind::Repair => A::OrderRepair,
            OrderKind::Reclaim => A::OrderReclaim,
            OrderKind::Load => A::OrderLoad,
            OrderKind::Unload => A::UnloadMode,
            OrderKind::Overcharge => A::OrderOvercharge,
            OrderKind::LaunchMissile => A::OrderLaunchMissile,
            OrderKind::ToggleShield => A::ToggleShield,
            OrderKind::ToggleRepeat => A::ToggleRepeat,
            OrderKind::Pause => A::Pause,
            OrderKind::Fog => A::ToggleFog,
            OrderKind::Ai => A::ToggleAi,
            OrderKind::IdleEngineer => A::IdleEngineer,
            OrderKind::AttackGround | OrderKind::FireState | OrderKind::PauseBuild => return None,
        })
    }

    pub fn rgb(self) -> [f32; 3] {
        match self {
            OrderKind::Attack | OrderKind::Overcharge | OrderKind::LaunchMissile | OrderKind::AttackGround => [1.0, 0.35, 0.28],
            OrderKind::Move | OrderKind::Patrol | OrderKind::Load | OrderKind::Unload => [0.40, 0.85, 1.0],
            OrderKind::Assist | OrderKind::Guard | OrderKind::Repair => [0.45, 1.0, 0.55],
            OrderKind::Reclaim => [1.0, 0.85, 0.35],
            _ => [0.9, 0.92, 0.96],
        }
    }
}

/// Saved camera framing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraBookmark {
    pub target: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Bookmarks {
    slots: [Option<CameraBookmark>; crate::keymap::BOOKMARKS],
}

impl Bookmarks {
    pub fn save(&mut self, slot: usize, cam: &OrbitCamera) -> bool {
        match self.slots.get_mut(slot) {
            Some(s) => {
                *s = Some(CameraBookmark { target: cam.target, yaw: cam.yaw, pitch: cam.pitch, distance: cam.distance });
                true
            }
            None => false,
        }
    }

    /// Applies bookmark `slot` to `cam` (aspect / limits untouched).
    pub fn recall(&self, slot: usize, cam: &mut OrbitCamera) -> bool {
        match self.slots.get(slot).copied().flatten() {
            Some(b) => {
                cam.target = b.target;
                cam.yaw = b.yaw;
                cam.pitch = b.pitch;
                cam.distance = b.distance.clamp(cam.min_distance, cam.max_distance);
                true
            }
            None => false,
        }
    }

    #[cfg(test)]
    pub fn get(&self, slot: usize) -> Option<CameraBookmark> {
        self.slots.get(slot).copied().flatten()
    }
}

/// Edge-scroll pan direction `(right, forward)` in `[-1, 1]` for a cursor at `(x, y)` in a
/// `w × h` window (`margin` pixels from an edge).
pub fn edge_scroll_dir(x: f32, y: f32, w: f32, h: f32, margin: f32) -> (f32, f32) {
    if x < 0.0 || y < 0.0 || x >= w || y >= h {
        return (0.0, 0.0);
    }
    let dx = if x <= margin { -1.0 } else if x >= w - 1.0 - margin { 1.0 } else { 0.0 };
    let dy = if y <= margin { 1.0 } else if y >= h - 1.0 - margin { -1.0 } else { 0.0 };
    (dx, dy)
}

// ---------------------------------------------------------------------------
// HUD unit-icon atlas (grid of 64x64 FA `_icon.dds` images)
// ---------------------------------------------------------------------------

pub const ICON_CELL: usize = 64;
pub const ICON_COLS: usize = 16;
pub const MAX_ICONS: usize = 512;

#[derive(Debug, Clone, Default)]
struct AtlasIds {
    ids: Vec<String>,
    rows: usize,
}

static ATLAS: RwLock<Option<AtlasIds>> = RwLock::new(None);

/// Unit ids packed into the icon atlas (sorted, capped at [`MAX_ICONS`]).
pub fn atlas_icon_ids(cat: &Catalog) -> Vec<String> {
    cat.types.keys().take(MAX_ICONS).cloned().collect()
}

/// Builds the icon atlas from `textures.scd` in one archive pass. Returns the packed atlas
/// and the ordered list of ids whose `_icon.dds` actually decoded (`None` if none were found).
pub fn load_icon_atlas(game_dir: &Path, ids: &[String]) -> Option<(RgbaImage, Vec<String>)> {
    if ids.is_empty() {
        return None;
    }
    let ids: Vec<&String> = ids.iter().take(MAX_ICONS).collect();
    let wanted: std::collections::HashSet<String> = ids.iter().map(|s| s.to_ascii_lowercase()).collect();
    let prefix = "textures/ui/common/icons/units/";
    let scd = crate::scd::gamedata_dir(game_dir).join("textures.scd");
    let files = crate::scd::read_scd_matching(&scd, |name| {
        let l = name.to_ascii_lowercase();
        l.strip_prefix(prefix)
            .and_then(|r| r.strip_suffix("_icon.dds"))
            .is_some_and(|id| wanted.contains(id))
    })
    .ok()?;
    let mut decoded: HashMap<String, RgbaImage> = HashMap::with_capacity(files.len());
    for (name, bytes) in files {
        let l = name.to_ascii_lowercase();
        let Some(id) = l.strip_prefix(prefix).and_then(|r| r.strip_suffix("_icon.dds")) else {
            continue;
        };
        if decoded.contains_key(id) {
            continue;
        }
        let Ok(img) = decode_dds_rgba8(&bytes) else {
            continue;
        };
        decoded.insert(id.to_string(), resize_rgba_bilinear(&img, ICON_CELL as u32, ICON_CELL as u32));
    }
    let mut found: Vec<(String, RgbaImage)> = Vec::with_capacity(decoded.len());
    for orig in &ids {
        if let Some(icon) = decoded.remove(&orig.to_ascii_lowercase()) {
            found.push(((*orig).clone(), icon));
        }
    }
    if found.is_empty() {
        return None;
    }
    let rows = found.len().div_ceil(ICON_COLS).max(1);
    let tw = ICON_COLS * ICON_CELL;
    let th = rows * ICON_CELL;
    let mut pixels = vec![0u8; tw * th * 4];
    let mut found_ids = Vec::with_capacity(found.len());
    for (slot, (id, icon)) in found.into_iter().enumerate() {
        let (cx, cy) = (slot % ICON_COLS, slot / ICON_COLS);
        for y in 0..ICON_CELL {
            let src = &icon.pixels[y * ICON_CELL * 4..(y + 1) * ICON_CELL * 4];
            let dst = ((cy * ICON_CELL + y) * tw + cx * ICON_CELL) * 4;
            pixels[dst..dst + ICON_CELL * 4].copy_from_slice(src);
        }
        found_ids.push(id);
    }
    Some((RgbaImage { width: tw as u32, height: th as u32, pixels }, found_ids))
}

/// Publishes the id list of the atlas currently bound for the HUD (`None` = no atlas).
pub fn set_atlas_ids(ids: Option<Vec<String>>) {
    if let Ok(mut g) = ATLAS.write() {
        *g = ids.map(|ids| {
            let rows = ids.len().div_ceil(ICON_COLS).max(1);
            AtlasIds { ids, rows }
        });
    }
}

/// UV rect `[u0, v0, u1, v1]` of `unit_type` in the bound atlas.
pub fn icon_uv(unit_type: &str) -> Option<[f32; 4]> {
    let g = ATLAS.read().ok()?;
    let a = g.as_ref()?;
    let slot = a.ids.iter().position(|i| i.eq_ignore_ascii_case(unit_type))?;
    let (cx, cy) = ((slot % ICON_COLS) as f32, (slot / ICON_COLS) as f32);
    let (cols, rows) = (ICON_COLS as f32, a.rows as f32);
    let inset = 0.5 / ICON_CELL as f32;
    Some([(cx + inset) / cols, (cy + inset) / rows, (cx + 1.0 - inset) / cols, (cy + 1.0 - inset) / rows])
}

// ---------------------------------------------------------------------------
// PlaySession glue
// ---------------------------------------------------------------------------

/// Engineering tech level of a builder unit (commanders: enhancement suite).
fn builder_tier(u: &crate::skirmish::Unit) -> u8 {
    if u.role == Role::Commander { u.enh.engineering_tier.max(1) } else { u.tier.clamp(1, 3) }
}

impl PlaySession {
    fn human_faction(&self) -> Option<usize> {
        self.world.army(self.human_army).map(|a| a.faction)
    }

    /// Selected own builders (commanders / engineers) not aboard a transport.
    pub fn selected_builders(&self) -> Vec<UnitId> {
        self.selected
            .iter()
            .copied()
            .filter(|id| {
                self.world
                    .unit(*id)
                    .is_some_and(|u| u.army == self.human_army && u.role.is_builder() && u.carried_by.is_none())
            })
            .collect()
    }

    /// The command card for the current selection.
    pub fn card_model(&self) -> CardModel {
        let Some(fi) = self.human_faction() else {
            return CardModel::empty();
        };
        let cat = self.world.catalog();
        let builders: Vec<&crate::skirmish::Unit> = self
            .selected_builders()
            .into_iter()
            .filter_map(|id| self.world.unit(id))
            .collect();
        let (all, source) = if !builders.is_empty() {
            (construction_options_for_units(cat, fi, &builders), CardSource::Construction)
        } else {
            let facs: Vec<&crate::skirmish::Unit> = self
                .selected_factories()
                .into_iter()
                .filter_map(|id| self.world.unit(id))
                .collect();
            if facs.is_empty() {
                return CardModel::empty();
            }
            (production_options_for_units(cat, fi, &facs), CardSource::Production)
        };
        CardModel::build(all, source, self.ui.card_tab, self.ui.card_group, self.ui.card_page)
    }

    /// Build rate used for tooltips (best selected builder / factory).
    pub fn card_build_rate(&self) -> i64 {
        self.selected
            .iter()
            .filter_map(|id| self.world.unit(*id))
            .filter(|u| u.army == self.human_army && (self.world.is_constructor(u) || self.world.is_producer(u)))
            .filter_map(|u| {
                let base = self.world.catalog().get(&u.type_id).map(|t| t.build_rate).unwrap_or(0);
                Some(if u.enh.build_rate > 0 { u.enh.build_rate } else { base })
            })
            .max()
            .unwrap_or(0)
    }

    /// Queued + in-progress count of `unit_type` over the selected factories.
    pub fn queued_count(&self, unit_type: &str) -> u32 {
        self.selected_factories()
            .iter()
            .filter_map(|id| self.world.unit(*id))
            .map(|u| {
                u.production.iter().filter(|t| *t == unit_type).count() as u32
                    + u32::from(u.build.as_ref().is_some_and(|j| j.unit_type == unit_type))
            })
            .sum()
    }

    /// Whether the card is populated (card grid hotkeys apply).
    pub fn card_active(&self) -> bool {
        !self.is_replay() && self.card_model().source.is_some()
    }

    pub fn set_card_tab(&mut self, tab: u8) {
        self.ui.card_tab = tab.clamp(1, 4);
        self.ui.card_group = None;
        self.ui.card_page = 0;
    }

    pub fn cycle_card_tab(&mut self) {
        let m = self.card_model();
        if m.source.is_some() {
            self.set_card_tab(m.next_tab());
            self.status_banner = format!("Command card: {}", tab_label(self.ui.card_tab));
        }
    }

    pub fn set_card_group(&mut self, idx: u8) {
        let m = self.card_model();
        self.ui.card_group = m.groups.get(idx as usize).copied();
        // Keep the effective tab so a fallback tab does not jump back.
        self.ui.card_tab = m.tab;
        self.ui.card_page = 0;
    }

    pub fn next_card_page(&mut self) {
        let m = self.card_model();
        self.ui.card_tab = m.tab;
        self.ui.card_group = m.group;
        self.ui.card_page = (m.page + 1) % m.pages.max(1);
    }

    /// Activates card entry `idx` of the current page. Construction: enter placement mode.
    /// Production: queue (`shift` = ×5) or, with `remove`, dequeue one (`shift` = 5).
    pub fn trigger_card_item(&mut self, idx: usize, shift: bool, remove: bool) {
        let m = self.card_model();
        let Some(item) = m.page_items().get(idx).cloned() else {
            return;
        };
        match m.source {
            Some(CardSource::Construction) => {
                if remove {
                    if self.build_mode.as_deref() == Some(item.unit_type.as_str()) {
                        self.build_mode = None;
                    }
                    return;
                }
                self.enter_build_mode(&item.unit_type);
            }
            Some(CardSource::Production) => {
                let count = if shift { 5 } else { 1 };
                if remove {
                    self.cancel_production(&item.unit_type, count);
                } else {
                    self.queue_production(&item.unit_type, count);
                }
            }
            None => {}
        }
    }

    /// Placement mode for structure `unit_type` (existing ghost / drag-line flow).
    pub fn enter_build_mode(&mut self, unit_type: &str) {
        self.attack_move_mode = false;
        self.ui.unload_mode = false;
        self.ui.order_mode = None;
        self.build_mode = Some(unit_type.to_string());
        let name = self
            .world
            .catalog()
            .get(unit_type)
            .map(|t| t.name.trim().to_string())
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| unit_type.to_string());
        self.status_banner = format!("Place {name} (LMB place, drag = line, Shift queue, RMB cancel)");
    }

    /// Queues `count` × `unit_type` on every selected factory able to build it.
    pub fn queue_production(&mut self, unit_type: &str, count: u32) {
        let Some(fi) = self.human_faction() else {
            return;
        };
        let cat = self.world.catalog();
        let Some(t) = cat.get(unit_type).cloned() else {
            return;
        };
        let facs: Vec<UnitId> = self
            .selected_factories()
            .into_iter()
            .filter(|id| self.world.unit(*id).is_some_and(|u| unit_can_produce(cat, fi, u, &t)))
            .collect();
        if facs.is_empty() {
            self.status_banner = format!("No selected factory can build {unit_type}");
            return;
        }
        for f in &facs {
            self.world.issue(self.human_army, Command::Produce { factory: *f, unit_type: unit_type.to_string(), count });
        }
        self.status_banner = format!("Queued {count}x {} on {} factory(s)", t.name.trim(), facs.len());
    }

    /// Removes `count` × `unit_type` from every selected factory queue holding it.
    pub fn cancel_production(&mut self, unit_type: &str, count: u32) {
        let facs: Vec<UnitId> = self
            .selected_factories()
            .into_iter()
            .filter(|id| {
                self.world.unit(*id).is_some_and(|u| {
                    u.production.iter().any(|t| t == unit_type) || u.build.as_ref().is_some_and(|j| j.unit_type == unit_type)
                })
            })
            .collect();
        for f in &facs {
            self.world.issue(self.human_army, Command::CancelProduce { factory: *f, unit_type: unit_type.to_string(), count });
        }
        if !facs.is_empty() {
            self.status_banner = format!("Removed {count}x {unit_type} from queue");
        }
    }

    /// Primary selected factory (first with a queue, else first) for the queue panel.
    pub fn queue_panel_unit(&self) -> Option<UnitId> {
        let facs = self.selected_factories();
        facs.iter()
            .copied()
            .find(|id| self.world.unit(*id).is_some_and(|u| u.build.is_some() || !u.production.is_empty()))
            .or_else(|| facs.first().copied())
    }

    /// Queue entries of the queue-panel factory.
    pub fn queue_panel_entries(&self) -> Vec<QueueEntry> {
        let Some(u) = self.queue_panel_unit().and_then(|id| self.world.unit(id)) else {
            return Vec::new();
        };
        let upgrading = matches!(u.orders.front(), Some(UnitOrder::Upgrade(_)));
        let active = u
            .build
            .as_ref()
            .filter(|_| !upgrading)
            .map(|j| (j.unit_type.as_str(), j.progress_permille()));
        queue_entries(active, u.production.iter())
    }

    /// Click on queue entry `idx`: remove one (`shift` = 5) of that type.
    pub fn trigger_queue_cancel(&mut self, idx: usize, shift: bool) {
        let Some(e) = self.queue_panel_entries().get(idx).cloned() else {
            return;
        };
        let Some(f) = self.queue_panel_unit() else {
            return;
        };
        let count = if shift { 5 } else { 1 };
        self.world
            .issue(self.human_army, Command::CancelProduce { factory: f, unit_type: e.unit_type.clone(), count });
        self.status_banner = format!("Cancelled {count}x {}", e.unit_type);
    }

    /// `(target type, units)` of the upgrade button: selected own units sharing the first
    /// upgradable type (mex, factories, …), not already upgrading.
    pub fn upgrade_target(&self) -> Option<(String, Vec<UnitId>)> {
        let cat = self.world.catalog();
        let first = self.selected.iter().filter_map(|id| self.world.unit(*id)).find(|u| {
            u.army == self.human_army && cat.get(&u.type_id).and_then(|t| t.upgrades_to.as_ref()).is_some_and(|x| cat.get(x).is_some())
        })?;
        let target = cat.get(&first.type_id)?.upgrades_to.clone()?;
        let target = cat.types.keys().find(|k| k.eq_ignore_ascii_case(&target)).cloned().unwrap_or(target);
        let units: Vec<UnitId> = self
            .selected
            .iter()
            .filter_map(|id| self.world.unit(*id))
            .filter(|u| u.army == self.human_army && u.type_id == first.type_id && u.carried_by.is_none())
            .map(|u| u.id)
            .collect();
        Some((target, units))
    }

    /// Upgrade progress (per-mille) of the first selected unit currently upgrading.
    pub fn upgrade_progress(&self) -> Option<(String, i64)> {
        self.selected.iter().filter_map(|id| self.world.unit(*id)).find_map(|u| match u.orders.front() {
            Some(UnitOrder::Upgrade(t)) if u.army == self.human_army => {
                Some((t.clone(), u.build.as_ref().map(|j| j.progress_permille()).unwrap_or(0)))
            }
            _ => None,
        })
    }

    pub fn trigger_upgrade(&mut self) {
        let Some((target, units)) = self.upgrade_target() else {
            self.status_banner = "Nothing upgradable selected".to_string();
            return;
        };
        let n = units.len();
        for u in units {
            self.world.issue(self.human_army, Command::Upgrade { unit: u, target_type: target.clone() });
        }
        self.status_banner = format!("Upgrading {n} unit(s) to {target}");
    }

    /// `(unit, rows)` of the enhancement panel: first selected own unit with enhancements.
    pub fn enhancement_panel(&self) -> Option<(UnitId, Vec<EnhRow>)> {
        let cat = self.world.catalog();
        let u = self.selected.iter().filter_map(|id| self.world.unit(*id)).find(|u| {
            u.army == self.human_army && cat.get(&u.type_id).is_some_and(|t| t.ext.enhancements.iter().any(|e| !e.is_removal()))
        })?;
        let specs = &cat.get(&u.type_id)?.ext.enhancements;
        let active = match (u.orders.front(), u.build.as_ref()) {
            (Some(UnitOrder::Enhance(n)), Some(j)) if j.unit_type == *n => Some((n.as_str(), j.progress_permille())),
            (Some(UnitOrder::Enhance(n)), None) => Some((n.as_str(), 0)),
            _ => None,
        };
        Some((u.id, enhancement_rows(specs, &u.enh.installed, active)))
    }

    pub fn trigger_enhance_row(&mut self, idx: usize) {
        let Some((unit, rows)) = self.enhancement_panel() else {
            return;
        };
        let Some(row) = rows.get(idx) else {
            return;
        };
        match row.state {
            EnhState::Available => {
                self.world.issue(self.human_army, Command::Enhance { unit, name: row.name.clone() });
                self.status_banner = format!("Installing {} (M{} E{})", row.display, row.cost_mass / SCALE, row.cost_energy / SCALE);
            }
            EnhState::Installed => self.status_banner = format!("{} already installed", row.display),
            EnhState::InProgress(p) => self.status_banner = format!("{} in progress ({}%)", row.display, p / 10),
            EnhState::Locked => {
                self.status_banner = match &row.prerequisite {
                    Some(p) => format!("{} requires {p}", row.display),
                    None => format!("{} slot is occupied", slot_label(&row.slot)),
                }
            }
        }
    }

    // ----- orders -------------------------------------------------------------

    /// The active targeting mode (derived from the legacy attack-move / unload flags).
    pub fn current_order_mode(&self) -> Option<OrderKind> {
        if self.attack_move_mode {
            Some(OrderKind::Attack)
        } else if self.ui.unload_mode {
            Some(OrderKind::Unload)
        } else {
            self.ui.order_mode
        }
    }

    pub fn clear_order_mode(&mut self) {
        self.attack_move_mode = false;
        self.ui.unload_mode = false;
        self.ui.order_mode = None;
    }

    fn selected_with(&self, pred: impl Fn(&crate::skirmish::Unit) -> bool) -> Vec<UnitId> {
        self.selected
            .iter()
            .copied()
            .filter(|id| self.world.unit(*id).is_some_and(|u| u.army == self.human_army && u.carried_by.is_none() && pred(u)))
            .collect()
    }

    fn selected_silos(&self) -> Vec<UnitId> {
        self.selected_with(|u| u.role == Role::MissileSilo && u.silo_missiles > 0)
    }

    fn selected_passengers(&self) -> Vec<UnitId> {
        let cat = self.world.catalog();
        self.selected_with(|u| u.speed > 0 && cat.get(&u.type_id).is_some_and(|t| t.ext.transport_class > 0))
    }

    /// Order buttons for the selection with their highlighted state.
    pub fn order_buttons(&self) -> Vec<(OrderKind, bool)> {
        let mode = self.current_order_mode();
        let mut out = Vec::new();
        if !self.is_replay() {
            let mobiles = !self.selected_with(|u| u.speed > 0).is_empty();
            let builders = !self.selected_builders().is_empty();
            let mut push = |k: OrderKind, show: bool| {
                if show {
                    out.push((k, mode == Some(k)));
                }
            };
            push(OrderKind::Move, mobiles);
            push(OrderKind::Attack, mobiles);
            push(OrderKind::Patrol, mobiles);
            push(OrderKind::Stop, !self.selected.is_empty());
            push(OrderKind::Guard, mobiles);
            push(OrderKind::Assist, builders);
            push(OrderKind::Repair, builders);
            push(OrderKind::Reclaim, builders);
            push(OrderKind::Load, !self.selected_passengers().is_empty());
            push(OrderKind::Unload, !self.selected_transports().is_empty());
            push(OrderKind::Overcharge, !self.selected_with(|u| u.role == Role::Commander).is_empty());
            push(OrderKind::LaunchMissile, !self.selected_with(|u| u.role == Role::MissileSilo).is_empty());
            // Phase 35: ground fire, fire state (lit unless firing at will) and build pause.
            let armed = self.selected_with(|u| u.can_fire_surface);
            push(OrderKind::AttackGround, !armed.is_empty());
            let gunners = self.selected_with(|u| !u.weapons.is_empty());
            if !gunners.is_empty() {
                let holding = gunners
                    .iter()
                    .any(|id| self.world.unit(*id).is_some_and(|u| u.fire_state != crate::skirmish::FireState::FireAtWill));
                out.push((OrderKind::FireState, holding));
            }
            let workers = self.selected_with(|u| u.build.is_some() || !u.production.is_empty() || u.build_paused);
            if !workers.is_empty() {
                let paused = workers.iter().any(|id| self.world.unit(*id).is_some_and(|u| u.build_paused));
                out.push((OrderKind::PauseBuild, paused));
            }
            let shields = self.selected_shield_units();
            if !shields.is_empty() {
                let on = shields.iter().any(|id| self.world.unit(*id).is_some_and(|u| u.shield_on));
                out.push((OrderKind::ToggleShield, on));
            }
            let facs = self.selected_factories();
            if !facs.is_empty() {
                let on = facs.iter().any(|id| self.world.unit(*id).is_some_and(|u| u.repeat_queue));
                out.push((OrderKind::ToggleRepeat, on));
            }
        }
        out.push((OrderKind::Pause, self.paused));
        out.push((OrderKind::Fog, self.fog_of_war));
        if !self.is_replay() {
            out.push((OrderKind::Ai, self.spectate));
            out.push((OrderKind::IdleEngineer, false));
        }
        out
    }

    /// Order button / hotkey: toggles a targeting mode or runs an instant order.
    pub fn trigger_order(&mut self, kind: OrderKind) {
        if kind.is_targeted() {
            let was = self.current_order_mode() == Some(kind);
            self.build_mode = None;
            self.clear_order_mode();
            if was {
                self.status_banner = format!("Cancelled {}", kind.label());
                return;
            }
            match kind {
                OrderKind::Attack => self.attack_move_mode = true,
                OrderKind::Unload => {
                    self.trigger_unload_mode();
                    return;
                }
                _ => self.ui.order_mode = Some(kind),
            }
            self.status_banner = format!("{}: LMB target (Shift queue, RMB/Esc cancel)", kind.label());
            return;
        }
        match kind {
            OrderKind::Stop => self.trigger_stop(),
            OrderKind::ToggleShield => self.trigger_toggle_shield(),
            OrderKind::ToggleRepeat => self.trigger_toggle_repeat(),
            OrderKind::Pause => self.toggle_pause(),
            OrderKind::Fog => self.trigger_hud_action(crate::hud::HudAction::ToggleFog, false),
            OrderKind::Ai => self.toggle_spectate(),
            OrderKind::IdleEngineer => self.trigger_hud_action(crate::hud::HudAction::IdleEngineer, false),
            OrderKind::FireState => {
                let units = self.selected_with(|u| !u.weapons.is_empty());
                let cur = units.first().and_then(|id| self.world.unit(*id)).map(|u| u.fire_state).unwrap_or_default();
                let state = cur.next();
                if !units.is_empty() {
                    self.world.issue(self.human_army, Command::SetFireState { units, state });
                    self.status_banner = format!("Fire state: {state:?}");
                }
            }
            OrderKind::PauseBuild => {
                let units = self.selected_with(|u| u.build.is_some() || !u.production.is_empty() || u.build_paused);
                let paused = !units.iter().any(|id| self.world.unit(*id).is_some_and(|u| u.build_paused));
                if !units.is_empty() {
                    self.world.issue(self.human_army, Command::SetBuildPaused { units, paused });
                    self.status_banner = if paused { "Construction paused".into() } else { "Construction resumed".into() };
                }
            }
            _ => {}
        }
    }

    /// Resolves a map LMB click in a targeting mode (Move … LaunchMissile, except the legacy
    /// attack-move / unload flags). Returns `true` if consumed.
    pub fn handle_left_click_order(&mut self, p: (i64, i64), queue: bool, pick_radius: i64) -> bool {
        let Some(kind) = self.ui.order_mode else {
            return false;
        };
        let army = self.human_army;
        let mobiles = self.selected_with(|u| u.speed > 0);
        let builders = self.selected_builders();
        let ally_at = self.world.unit_at(p, pick_radius, Some(army), None);
        let enemy_at = self.world.unit_at(p, pick_radius, None, Some(army));
        let (cmd, banner): (Option<Command>, String) = match kind {
            OrderKind::Move => (
                (!mobiles.is_empty()).then(|| Command::Move { units: mobiles, x: p.0, y: p.1, queue }),
                "Move ordered".into(),
            ),
            OrderKind::Patrol => (
                (!mobiles.is_empty()).then(|| Command::Patrol { units: mobiles, x: p.0, y: p.1, queue }),
                "Patrol ordered".into(),
            ),
            OrderKind::AttackGround => {
                let armed = self.selected_with(|u| u.can_fire_surface);
                ((!armed.is_empty()).then_some(Command::AttackGround { units: armed, x: p.0, y: p.1, queue }), "Ground fire".into())
            }
            OrderKind::Assist => match ally_at {
                Some(t) if !builders.is_empty() => {
                    (Some(Command::Assist { units: builders, target: t, queue }), format!("Assisting #{t}"))
                }
                _ => (None, "Assist: click a friendly factory or builder".into()),
            },
            OrderKind::Guard => match ally_at.and_then(|t| self.world.unit(t)).map(|u| (u.id, u.role, (u.pos.x, u.pos.y))) {
                Some((t, role, _)) if (role.is_factory() || role.is_builder()) && !builders.is_empty() => {
                    (Some(Command::Assist { units: builders, target: t, queue }), format!("Guarding #{t}"))
                }
                Some((t, _, _)) if !mobiles.is_empty() => (
                    Some(Command::Guard { units: mobiles.into_iter().filter(|&m| m != t).collect(), target: t, queue }),
                    format!("Guarding #{t}"),
                ),
                _ => (None, "Guard: click a friendly unit".into()),
            },
            OrderKind::Repair => match ally_at {
                Some(t) if !builders.is_empty() => {
                    (Some(Command::Repair { units: builders, target: t, queue }), format!("Repairing #{t}"))
                }
                _ => (None, "Repair: click a friendly unit".into()),
            },
            OrderKind::Reclaim => {
                let at = self.world.wreck_at(p, pick_radius + 4 * SCALE).map(|(_, wp)| wp).unwrap_or(p);
                (
                    (!builders.is_empty()).then(|| Command::Reclaim { units: builders, x: at.0, y: at.1, queue }),
                    "Reclaiming".into(),
                )
            }
            OrderKind::Load => {
                let cat = self.world.catalog();
                let transport = ally_at.filter(|t| {
                    self.world.unit(*t).is_some_and(|u| cat.get(&u.type_id).is_some_and(|x| x.ext.transport.is_some()))
                });
                match transport {
                    Some(t) => {
                        let pax: Vec<UnitId> = self.selected_passengers().into_iter().filter(|&id| id != t).collect();
                        ((!pax.is_empty()).then(|| Command::Load { units: pax, transport: t, queue }), format!("Boarding #{t}"))
                    }
                    None => (None, "Load: click a friendly air transport".into()),
                }
            }
            OrderKind::Overcharge => {
                let acu = self.selected_with(|u| u.role == Role::Commander).first().copied();
                match (acu, enemy_at) {
                    (Some(unit), Some(target)) => (Some(Command::Overcharge { unit, target }), format!("OVERCHARGE -> #{target}")),
                    _ => (None, "Overcharge: click an enemy unit".into()),
                }
            }
            OrderKind::LaunchMissile => match self.selected_silos().first().copied() {
                Some(unit) => (Some(Command::LaunchMissile { unit, x: p.0, y: p.1 }), "Missile launched".into()),
                None => (None, "No loaded missile silo selected".into()),
            },
            _ => (None, String::new()),
        };
        if let Some(c) = cmd {
            self.world.issue(army, c);
        }
        if !banner.is_empty() {
            self.status_banner = banner;
        }
        if !queue {
            self.ui.order_mode = None;
        }
        true
    }

    /// Picks the builder for a structure order: the first selected builder able to build
    /// `unit_type`, else any selected builder, else the commander.
    pub fn builder_for(&self, unit_type: &str) -> Option<UnitId> {
        let cat = self.world.catalog();
        let t = cat.get(unit_type);
        let fi = self.human_faction().unwrap_or(usize::MAX);
        let builders = self.selected_builders();
        builders
            .iter()
            .copied()
            .find(|id| {
                t.is_some_and(|ut| self.world.unit(*id).is_some_and(|u| unit_can_build(cat, fi, u, ut)))
            })
            .or_else(|| builders.first().copied())
            .or_else(|| self.world.commander_of(self.human_army).map(|u| u.id))
    }

    /// Hover text for a camera bookmark recall.
    pub fn apply_bookmark(&mut self, slot: usize, save: bool, cam: &mut OrbitCamera) -> bool {
        if save {
            self.ui.bookmarks.save(slot, cam);
            self.status_banner = format!("Camera bookmark {} saved", slot + 1);
            false
        } else if self.ui.bookmarks.recall(slot, cam) {
            self.status_banner = format!("Camera bookmark {}", slot + 1);
            true
        } else {
            self.status_banner = format!("Camera bookmark {} is empty (Ctrl+key saves)", slot + 1);
            false
        }
    }
}

/// Snapshot hook (`FARUST_SNAPSHOT_SELECT=acu|engineer|factory|enhance|card`): selects one
/// unit class, and for `factory` / `enhance` starts a short queue / enhancement on the
/// (throw-away) session clone so the panels show progress.
pub fn snapshot_select(s: &mut PlaySession, what: &str) -> bool {
    let army = s.human_army;
    let pick = |s: &PlaySession, f: &dyn Fn(&crate::skirmish::Unit) -> bool| {
        let mut v: Vec<&crate::skirmish::Unit> = s.world.units().filter(|u| u.army == army && f(u)).collect();
        v.sort_by_key(|u| (std::cmp::Reverse(u.tier), u.id));
        v.first().map(|u| u.id)
    };
    let id = match what {
        "acu" | "enhance" => pick(s, &|u| u.role == Role::Commander),
        "engineer" => pick(s, &|u| u.role == Role::Engineer).or_else(|| pick(s, &|u| u.role == Role::Commander)),
        "factory" => pick(s, &|u| u.role.is_factory()),
        _ => return false,
    };
    let Some(id) = id else {
        return false;
    };
    s.selected = vec![id];
    s.ui.card_tab = 1;
    match what {
        "factory" => {
            let m = s.card_model();
            let types: Vec<String> = m.items.iter().map(|i| i.unit_type.clone()).take(3).collect();
            for (k, t) in types.iter().enumerate() {
                s.world.issue(army, Command::Produce { factory: id, unit_type: t.clone(), count: [2, 5, 1][k % 3] });
            }
            if let Some(f) = s.world.unit(id)
                && f.production.is_empty()
            {
                for _ in 0..30 {
                    s.world.step();
                }
            }
            s.world.step();
            s.world.issue(army, Command::SetRepeat { factories: vec![id], enabled: true });
            for _ in 0..20 {
                s.world.step();
            }
        }
        "enhance" => {
            if let Some((_, rows)) = s.enhancement_panel()
                && let Some(r) = rows.iter().find(|r| r.state == EnhState::Available)
            {
                s.world.issue(army, Command::Enhance { unit: id, name: r.name.clone() });
                for _ in 0..40 {
                    s.world.step();
                }
            }
        }
        _ => {}
    }
    s.selected.retain(|u| s.world.unit(*u).is_some());
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skirmish::tests::{test_catalog, test_world};

    fn ids(v: &[CardItem]) -> Vec<&str> {
        v.iter().map(|i| i.unit_type.as_str()).collect()
    }

    #[test]
    fn construction_card_is_catalog_driven_with_tabs_and_groups() {
        let mut cat = test_catalog();
        // A T2 radar and an experimental make the T2 / EXP tabs non-empty.
        let mut rad2 = cat.get("RRAD").unwrap().clone();
        rad2.id = "RRAD2".into();
        rad2.tier = 2;
        cat.types.insert("RRAD2".into(), rad2);
        let mut exp = cat.get("RT").unwrap().clone();
        exp.id = "REXP".into();
        exp.role = Role::Experimental;
        exp.tier = 4;
        cat.types.insert("REXP".into(), exp);
        cat.factions[0].t2_pgen = "RRAD2".into(); // listed as buildable by the faction
        cat.factions[0].experimental = "REXP".into();

        let t1 = construction_options(&cat, 0, 1);
        // Factory first, then economy (mex before pgen by role code), defense, intel, shields.
        assert_eq!(ids(&t1), vec!["RF", "RM", "RP", "RPD", "RRAD", "RSH"]);
        assert_eq!(t1[0].group, CardGroup::Factories);
        assert_eq!(t1[1].group, CardGroup::Economy);
        assert_eq!(t1[3].group, CardGroup::Defense);
        assert_eq!(t1[4].group, CardGroup::Intel);
        assert_eq!(t1[5].group, CardGroup::Shields);
        // Upgrade-only T2 factory is not buildable; T2 / EXP need the tech level.
        assert!(!ids(&t1).contains(&"RF2"));
        let t3 = construction_options(&cat, 0, 3);
        assert!(ids(&t3).contains(&"RRAD2") && ids(&t3).contains(&"REXP"));
        assert!(!ids(&construction_options(&cat, 0, 2)).contains(&"REXP"));
        // The other faction never sees red's ids.
        assert!(construction_options(&cat, 1, 3).iter().all(|i| i.unit_type.starts_with('B')));

        let m = CardModel::build(t3.clone(), CardSource::Construction, 1, None, 0);
        assert_eq!(m.tab_counts, [6, 1, 0, 1]);
        assert_eq!(m.tab, 1);
        assert_eq!(m.groups, vec![CardGroup::Factories, CardGroup::Economy, CardGroup::Defense, CardGroup::Intel, CardGroup::Shields]);
        assert_eq!(m.page_items().len(), 6);
        assert_eq!(m.next_tab(), 2);
        // Empty T3 tab falls back to T1; EXP tab holds the experimental.
        assert_eq!(CardModel::build(t3.clone(), CardSource::Construction, 3, None, 0).tab, 1);
        let e = CardModel::build(t3.clone(), CardSource::Construction, 4, None, 0);
        assert_eq!(ids(e.page_items()), vec!["REXP"]);
        assert_eq!(e.next_tab(), 1);
        // Group filter.
        let g = CardModel::build(t3.clone(), CardSource::Construction, 1, Some(CardGroup::Economy), 0);
        assert_eq!(ids(g.page_items()), vec!["RM", "RP"]);
        // A group absent from the tab is ignored.
        assert_eq!(CardModel::build(t3, CardSource::Construction, 1, Some(CardGroup::Air), 0).group, None);
    }

    #[test]
    fn production_card_filters_by_factory_domain_and_tier_and_pages() {
        let cat = test_catalog();
        let land1 = production_options(&cat, 0, &[(Role::Factory, 1)]);
        assert_eq!(ids(&land1), vec!["RENG", "RA", "RT"], "engineers first, then by tier / cost / id");
        assert_eq!(land1[0].group, CardGroup::Engineers);
        assert!(land1[1..].iter().all(|i| i.group == CardGroup::Land));
        assert!(!ids(&land1).contains(&"RT2"), "T2 needs a T2 factory");
        assert!(!ids(&land1).contains(&"RAIR"), "air units need an air factory");
        let land2 = production_options(&cat, 0, &[(Role::Factory, 2)]);
        assert!(ids(&land2).contains(&"RT2"));
        let air = production_options(&cat, 0, &[(Role::AirFactory, 1)]);
        assert_eq!(ids(&air), vec!["RAIR"]);
        assert_eq!(air[0].group, CardGroup::Air);
        let both = production_options(&cat, 0, &[(Role::Factory, 1), (Role::AirFactory, 1)]);
        assert_eq!(both.len(), land1.len() + 1);

        // Paging: 17 items → 2 pages of 15.
        let many: Vec<CardItem> = (0..17)
            .map(|i| CardItem { unit_type: format!("U{i:02}"), tier: 1, group: CardGroup::Land })
            .collect();
        let p0 = CardModel::build(many.clone(), CardSource::Production, 1, None, 0);
        assert_eq!((p0.pages, p0.page_items().len()), (2, 15));
        let p1 = CardModel::build(many.clone(), CardSource::Production, 1, None, 1);
        assert_eq!(ids(p1.page_items()), vec!["U15", "U16"]);
        assert_eq!(CardModel::build(many, CardSource::Production, 1, None, 9).page, 1);
    }

    #[test]
    fn queue_panel_model_groups_runs_and_leads_with_active_job() {
        let q: Vec<String> = ["T", "T", "A", "T", "T", "T"].iter().map(|s| s.to_string()).collect();
        let e = queue_entries(Some(("T", 420)), q.iter());
        assert_eq!(
            e,
            vec![
                QueueEntry { unit_type: "T".into(), count: 3, progress: Some(420) },
                QueueEntry { unit_type: "A".into(), count: 1, progress: None },
                QueueEntry { unit_type: "T".into(), count: 3, progress: None },
            ]
        );
        let e = queue_entries(Some(("A", 2000)), q.iter());
        assert_eq!(e[0], QueueEntry { unit_type: "A".into(), count: 1, progress: Some(1000) });
        assert_eq!(e[1].count, 2);
        assert!(queue_entries(None, std::iter::empty()).is_empty());
    }

    #[test]
    fn enhancement_rows_track_slots_prerequisites_and_progress() {
        let spec = |name: &str, slot: &str, pre: Option<&str>| EnhancementSpec {
            name: name.into(),
            display: format!("{name} Suite"),
            slot: slot.into(),
            prerequisite: pre.map(str::to_string),
            cost_mass: 1000 * SCALE,
            cost_energy: 20_000 * SCALE,
            build_work: 500,
            ..Default::default()
        };
        let mut rm = spec("AdvEngRemove", "LCH", Some("AdvEng"));
        rm.removes = vec!["AdvEng".into()];
        let specs = vec![
            spec("T3Eng", "LCH", Some("AdvEng")),
            spec("AdvEng", "LCH", None),
            rm,
            spec("Shield", "Back", None),
            spec("Gun", "RCH", None),
        ];
        let rows = enhancement_rows(&specs, &[], None);
        let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["AdvEng", "T3Eng", "Gun", "Shield"], "slot order, then prerequisite depth; removals hidden");
        assert_eq!(rows[0].state, EnhState::Available);
        assert_eq!(rows[1].state, EnhState::Locked);
        assert_eq!(rows[0].display, "AdvEng Suite");

        let inst = vec![("LCH".to_string(), "AdvEng".to_string())];
        let rows = enhancement_rows(&specs, &inst, Some(("Gun", 250)));
        let st = |n: &str| rows.iter().find(|r| r.name == n).unwrap().state;
        assert_eq!(st("AdvEng"), EnhState::Installed);
        assert_eq!(st("T3Eng"), EnhState::Available);
        assert_eq!(st("Gun"), EnhState::InProgress(250));
        assert_eq!(st("Shield"), EnhState::Available);
        let inst = vec![("LCH".to_string(), "T3Eng".to_string())];
        let rows = enhancement_rows(&specs, &inst, None);
        assert!(rows.iter().filter(|r| r.slot == "LCH").all(|r| r.state == EnhState::Installed), "whole chain installed");
    }

    #[test]
    fn bookmarks_save_and_recall_camera() {
        let mut cam = OrbitCamera::for_map(256.0, 256.0, 10.0, 1.6);
        let mut b = Bookmarks::default();
        let mut probe = cam.clone();
        assert!(!b.recall(0, &mut probe));
        cam.target = [10.0, 2.0, 20.0];
        cam.yaw = 0.7;
        cam.distance = 80.0;
        assert!(b.save(1, &cam));
        assert!(!b.save(crate::keymap::BOOKMARKS, &cam), "out of range slot");
        cam.target = [0.0; 3];
        cam.yaw = 0.0;
        assert!(b.recall(1, &mut cam));
        assert_eq!(cam.target, [10.0, 2.0, 20.0]);
        assert_eq!(cam.yaw, 0.7);
        assert_eq!(b.get(1).unwrap().distance, 80.0);
        assert!(b.get(0).is_none());
        assert_eq!(edge_scroll_dir(1.0, 300.0, 800.0, 600.0, 4.0), (-1.0, 0.0));
        assert_eq!(edge_scroll_dir(799.0, 0.0, 800.0, 600.0, 4.0), (1.0, 1.0));
        assert_eq!(edge_scroll_dir(400.0, 599.0, 800.0, 600.0, 4.0), (0.0, -1.0));
        assert_eq!(edge_scroll_dir(400.0, 300.0, 800.0, 600.0, 4.0), (0.0, 0.0));
        assert_eq!(edge_scroll_dir(-5.0, 300.0, 800.0, 600.0, 4.0), (0.0, 0.0));
    }

    #[test]
    fn session_card_queue_cancel_and_upgrade_go_through_commands() {
        let mut w = test_world(7);
        let acu = w.commander_of(0).unwrap().id;
        let fac = w.spawn_unit(0, "RF", (70 * SCALE, 50 * SCALE)).unwrap();
        w.run(2);
        let mut s = PlaySession::new(w, 0, false, 1);
        s.selected = vec![acu];
        let m = s.card_model();
        assert_eq!(m.source, Some(CardSource::Construction));
        assert_eq!(ids(m.page_items())[0], "RF");
        s.trigger_card_item(1, false, false);
        assert_eq!(s.build_mode.as_deref(), Some("RM"));
        assert_eq!(s.builder_for("RM"), Some(acu));

        s.build_mode = None;
        s.selected = vec![fac];
        let m = s.card_model();
        assert_eq!(m.source, Some(CardSource::Production));
        let tank = m.page_items().iter().position(|i| i.unit_type == "RT").unwrap();
        s.trigger_card_item(tank, true, false); // Shift: x5
        s.step_sim();
        s.step_sim();
        assert_eq!(s.queued_count("RT"), 5);
        let entries = s.queue_panel_entries();
        assert_eq!(entries[0].unit_type, "RT");
        assert!(entries[0].progress.is_some());
        assert_eq!(entries.iter().map(|e| e.count).sum::<u32>(), 5);
        // Right-click on the card removes one; a queue click removes another.
        s.trigger_card_item(tank, false, true);
        s.step_sim();
        assert_eq!(s.queued_count("RT"), 4);
        s.trigger_queue_cancel(0, true);
        s.step_sim();
        assert_eq!(s.queued_count("RT"), 0, "shift-cancel removes the rest incl. the active job");
        assert!(s.world.unit(fac).unwrap().build.is_none());

        // Upgrade button: RF -> RF2.
        let (target, units) = s.upgrade_target().unwrap();
        assert_eq!((target.as_str(), units), ("RF2", vec![fac]));
        s.trigger_upgrade();
        s.step_sim();
        assert!(s.upgrade_progress().is_some_and(|(t, _)| t == "RF2"));

        // Orders: targeted modes toggle; Move clicks issue commands.
        s.selected = vec![acu];
        let kinds: Vec<OrderKind> = s.order_buttons().iter().map(|b| b.0).collect();
        assert!(kinds.contains(&OrderKind::Repair) && kinds.contains(&OrderKind::Overcharge));
        assert!(!kinds.contains(&OrderKind::ToggleRepeat));
        s.trigger_order(OrderKind::Move);
        assert_eq!(s.current_order_mode(), Some(OrderKind::Move));
        assert!(s.handle_left_click_order((90 * SCALE, 90 * SCALE), false, 2 * SCALE));
        assert_eq!(s.current_order_mode(), None);
        s.step_sim();
        assert!(matches!(s.world.unit(acu).unwrap().orders.front(), Some(UnitOrder::Move { .. })));
        s.trigger_order(OrderKind::Attack);
        assert!(s.attack_move_mode);
        s.trigger_order(OrderKind::Attack);
        assert!(!s.attack_move_mode, "second press cancels");
    }

    #[test]
    fn cancel_produce_command_is_replay_safe_and_owner_checked() {
        let mut w = test_world(8);
        let fac = w.spawn_unit(0, "RF", (70 * SCALE, 50 * SCALE)).unwrap();
        w.issue(0, Command::Produce { factory: fac, unit_type: "RT".into(), count: 3 });
        w.issue(0, Command::Produce { factory: fac, unit_type: "RA".into(), count: 2 });
        w.run(2);
        let u = w.unit(fac).unwrap();
        assert_eq!(u.build.as_ref().unwrap().unit_type, "RT");
        assert_eq!(u.production.len(), 4);
        // Wrong army is rejected.
        assert!(!w.apply_command_direct(1, Command::CancelProduce { factory: fac, unit_type: "RT".into(), count: 1 }));
        assert!(w.apply_command_direct(0, Command::CancelProduce { factory: fac, unit_type: "RA".into(), count: 1 }));
        let q: Vec<&str> = w.unit(fac).unwrap().production.iter().map(String::as_str).collect();
        assert_eq!(q, vec!["RT", "RT", "RA"]);
        assert!(w.apply_command_direct(0, Command::CancelProduce { factory: fac, unit_type: "RT".into(), count: 3 }));
        let u = w.unit(fac).unwrap();
        assert!(u.build.is_none());
        assert_eq!(u.production.iter().map(String::as_str).collect::<Vec<_>>(), vec!["RA"]);
        assert!(!w.apply_command_direct(0, Command::CancelProduce { factory: fac, unit_type: "RT".into(), count: 1 }));
    }

    #[test]
    fn tooltip_lists_cost_time_and_combat_stats() {
        let cat = test_catalog();
        let t = item_tooltip(&cat, "RT", 20);
        assert!(t[0].starts_with("RT (T1)"));
        assert!(t[2].contains("Mass 50") && t[2].contains("Energy 250") && t[2].contains("Time 1s"), "{t:?}");
        assert!(t[3].contains("DPS 25") && t[3].contains("Range 16"), "{t:?}");
        assert_eq!(item_tooltip(&cat, "NOPE", 1), vec!["NOPE".to_string()]);
    }

    #[test]
    fn roster_backed_card_exposes_roster_and_mod_units() {
        use crate::pathfinding::PathGrid;
        use crate::roster::{BuildGroup, BuildKind, BuildOption, Roster};
        use crate::skirmish::{ArmySetup, Skirmish};
        use std::sync::Arc;

        let mut cat = test_catalog();
        // Add a roster-only T1 defense unit and a mod-added T2 unit not listed in Faction.
        let mut rmod_bld = cat.get("RPD").unwrap().clone();
        rmod_bld.id = "RMOD_PD".into();
        cat.types.insert("RMOD_PD".into(), rmod_bld);
        let mut rmod_tank = cat.get("RT").unwrap().clone();
        rmod_tank.id = "RMOD_TANK".into();
        cat.types.insert("RMOD_TANK".into(), rmod_tank);

        let mut roster = Roster::default();
        roster.options.insert(
            "RENG".into(),
            vec![
                BuildOption { id: "RF".into(), tier: 1, kind: BuildKind::Build, group: BuildGroup::Factory },
                BuildOption { id: "RMOD_PD".into(), tier: 1, kind: BuildKind::Build, group: BuildGroup::Defense },
            ],
        );
        roster.options.insert(
            "RF".into(),
            vec![
                BuildOption { id: "RT".into(), tier: 1, kind: BuildKind::Produce, group: BuildGroup::Land },
                BuildOption { id: "RMOD_TANK".into(), tier: 1, kind: BuildKind::Produce, group: BuildGroup::Land },
            ],
        );
        cat.roster = Arc::new(roster);

        let mut w = Skirmish::new(
            cat,
            PathGrid::new(32, 32, 4 * SCALE),
            (128 * SCALE, 128 * SCALE),
            vec![(20 * SCALE, 20 * SCALE)],
            &[
                ArmySetup { name: "A".into(), faction: 0, start: (24 * SCALE, 24 * SCALE), team: 0 },
                ArmySetup { name: "B".into(), faction: 1, start: (100 * SCALE, 100 * SCALE), team: 1 },
            ],
            1,
        )
        .unwrap();
        let eng = w.spawn_unit(0, "RENG", (28 * SCALE, 28 * SCALE)).unwrap();
        let fac = w.spawn_unit(0, "RF", (32 * SCALE, 28 * SCALE)).unwrap();
        let mut s = PlaySession::new(w, 0, false, 1);

        s.selected = vec![eng];
        let m = s.card_model();
        assert_eq!(m.source, Some(CardSource::Construction));
        assert!(m.items.iter().any(|i| i.unit_type == "RMOD_PD"), "mod/roster structure should appear on card");

        s.selected = vec![fac];
        let m = s.card_model();
        assert_eq!(m.source, Some(CardSource::Production));
        assert!(m.items.iter().any(|i| i.unit_type == "RMOD_TANK"), "mod/roster unit should appear on factory card");
    }

    #[test]
    fn quantum_gate_only_shows_sacus_and_routes_production_correctly() {
        use crate::pathfinding::PathGrid;
        use crate::roster::{BuildGroup, BuildKind, BuildOption, Roster};
        use crate::skirmish::{ArmySetup, Skirmish};
        use std::sync::Arc;

        let mut cat = test_catalog();
        let mut gate = cat.get("RF").unwrap().clone();
        gate.id = "RGATE".into();
        gate.tier = 3;
        cat.types.insert("RGATE".into(), gate);
        let mut sacu = cat.get("RENG").unwrap().clone();
        sacu.id = "RSACU".into();
        sacu.tier = 3;
        cat.types.insert("RSACU".into(), sacu);

        let mut roster = Roster::default();
        roster.options.insert(
            "RF".into(),
            vec![BuildOption { id: "RT".into(), tier: 1, kind: BuildKind::Produce, group: BuildGroup::Land }],
        );
        roster.options.insert(
            "RGATE".into(),
            vec![BuildOption { id: "RSACU".into(), tier: 3, kind: BuildKind::Produce, group: BuildGroup::Engineer }],
        );
        cat.roster = Arc::new(roster);

        let mut w = Skirmish::new(
            cat,
            PathGrid::new(32, 32, 4 * SCALE),
            (128 * SCALE, 128 * SCALE),
            vec![(20 * SCALE, 20 * SCALE)],
            &[
                ArmySetup { name: "A".into(), faction: 0, start: (24 * SCALE, 24 * SCALE), team: 0 },
                ArmySetup { name: "B".into(), faction: 1, start: (100 * SCALE, 100 * SCALE), team: 1 },
            ],
            2,
        )
        .unwrap();
        let fac = w.spawn_unit(0, "RF", (32 * SCALE, 28 * SCALE)).unwrap();
        let gate_id = w.spawn_unit(0, "RGATE", (40 * SCALE, 28 * SCALE)).unwrap();
        let mut s = PlaySession::new(w, 0, false, 1);

        // Selecting only the Quantum Gate exposes only RSACU, never land tanks.
        s.selected = vec![gate_id];
        let m = s.card_model();
        assert_eq!(ids(&m.items), vec!["RSACU"]);
        assert_eq!(m.tab, 3);

        // Selecting both RF and RGATE and queuing RT routes RT only to RF, not RGATE.
        s.selected = vec![fac, gate_id];
        s.queue_production("RT", 2);
        s.step_sim();
        assert!(
            s.world.unit(fac).unwrap().build.is_some() || !s.world.unit(fac).unwrap().production.is_empty(),
            "land factory should receive RT order"
        );
        assert!(
            s.world.unit(gate_id).unwrap().build.is_none() && s.world.unit(gate_id).unwrap().production.is_empty(),
            "quantum gate must NOT receive RT order"
        );
    }

    #[test]
    fn commander_engineering_tier_caps_roster_builds() {
        use crate::pathfinding::PathGrid;
        use crate::roster::{BuildGroup, BuildKind, BuildOption, Roster};
        use crate::skirmish::{ArmySetup, EnhancementSpec, Skirmish};
        use std::sync::Arc;

        let mut cat = test_catalog();
        if let Some(rc) = cat.types.get_mut("RC") {
            rc.ext.enhancements.push(EnhancementSpec {
                name: "T2Eng".into(),
                display: "T2 Engineering".into(),
                slot: "LCH".into(),
                engineering_tier: 2,
                cost_mass: 10 * SCALE,
                cost_energy: 50 * SCALE,
                build_work: 5,
                ..Default::default()
            });
        }
        let mut rrad2 = cat.get("RRAD").unwrap().clone();
        rrad2.id = "RRAD2".into();
        rrad2.tier = 2;
        cat.types.insert("RRAD2".into(), rrad2);
        let mut rrad3 = cat.get("RRAD").unwrap().clone();
        rrad3.id = "RRAD3".into();
        rrad3.tier = 3;
        cat.types.insert("RRAD3".into(), rrad3);

        let mut roster = Roster::default();
        roster.options.insert(
            "RC".into(),
            vec![
                BuildOption { id: "RRAD".into(), tier: 1, kind: BuildKind::Build, group: BuildGroup::Intel },
                BuildOption { id: "RRAD2".into(), tier: 2, kind: BuildKind::Build, group: BuildGroup::Intel },
                BuildOption { id: "RRAD3".into(), tier: 3, kind: BuildKind::Build, group: BuildGroup::Intel },
            ],
        );
        cat.roster = Arc::new(roster);

        let w = Skirmish::new(
            cat,
            PathGrid::new(32, 32, 4 * SCALE),
            (128 * SCALE, 128 * SCALE),
            vec![(20 * SCALE, 20 * SCALE)],
            &[
                ArmySetup { name: "A".into(), faction: 0, start: (24 * SCALE, 24 * SCALE), team: 0 },
                ArmySetup { name: "B".into(), faction: 1, start: (100 * SCALE, 100 * SCALE), team: 1 },
            ],
            3,
        )
        .unwrap();
        let acu = w.commander_of(0).unwrap().id;
        let mut s = PlaySession::new(w, 0, false, 1);
        s.selected = vec![acu];

        // Un-enhanced commander (tier 1) must not see T2/T3 builds even if present in roster.options.
        let m1 = s.card_model();
        assert_eq!(m1.tab_counts[1], 0, "T2 tab should be empty for T1 ACU");
        assert_eq!(m1.tab_counts[2], 0, "T3 tab should be empty for T1 ACU");
        assert!(m1.items.iter().any(|i| i.unit_type == "RRAD"));

        // Upgrade commander engineering tier to T2 -> T2 visible, T3 still hidden.
        s.world.issue(0, Command::Enhance { unit: acu, name: "T2Eng".into() });
        s.world.run(20);
        assert_eq!(s.world.unit(acu).unwrap().enh.engineering_tier, 2);
        s.set_card_tab(2);
        let m2 = s.card_model();
        assert_eq!(ids(&m2.items), vec!["RRAD2"]);
        assert_eq!(m2.tab_counts[2], 0);
    }
}
