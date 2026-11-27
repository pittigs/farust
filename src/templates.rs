//! Phase 36: build templates — named structure layouts (blueprint id + offset from the
//! anchor structure) saved to `$XDG_CONFIG_HOME/farust/templates.txt` and placed again as
//! ordinary queued `Command::Build`s, so a placed template replays deterministically.

use std::path::PathBuf;

use crate::skirmish::{Skirmish, UnitId};

/// One saved layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildTemplate {
    pub name: String,
    /// `(blueprint id, dx, dz)` in milli-units from the anchor, in build order.
    pub items: Vec<(String, i64, i64)>,
}

impl BuildTemplate {
    /// Layout of the selected structures `ids` of `world`; the lowest id is the anchor.
    pub fn from_structures(world: &Skirmish, ids: &[UnitId], name: &str) -> Option<Self> {
        let mut units: Vec<_> = ids
            .iter()
            .filter_map(|id| world.unit(*id))
            .filter(|u| u.alive && u.role.is_structure())
            .collect();
        units.sort_by_key(|u| u.id);
        let anchor = units.first()?.pos;
        Some(Self {
            name: name.to_string(),
            items: units.iter().map(|u| (u.type_id.clone(), u.pos.x - anchor.x, u.pos.y - anchor.y)).collect(),
        })
    }

    /// Absolute build sites with the anchor placed at `at`.
    pub fn sites(&self, at: (i64, i64)) -> Vec<(String, (i64, i64))> {
        self.items.iter().map(|(t, dx, dz)| (t.clone(), (at.0 + dx, at.1 + dz))).collect()
    }

    fn to_line(&self) -> String {
        let items: Vec<String> = self.items.iter().map(|(t, dx, dz)| format!("{t},{dx},{dz}")).collect();
        format!("{}\t{}", self.name.replace(['\t', '\n'], " "), items.join(";"))
    }

    fn from_line(line: &str) -> Option<Self> {
        let (name, rest) = line.split_once('\t')?;
        let items = rest
            .split(';')
            .filter(|s| !s.trim().is_empty())
            .map(|it| {
                let mut p = it.split(',');
                let t = p.next()?.trim();
                let dx = p.next()?.trim().parse().ok()?;
                let dz = p.next()?.trim().parse().ok()?;
                (!t.is_empty() && t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')).then(|| (t.to_string(), dx, dz))
            })
            .collect::<Option<Vec<_>>>()?;
        (!items.is_empty()).then(|| Self { name: name.trim().to_string(), items })
    }
}

/// Parses a templates file; malformed lines are skipped.
pub fn parse(text: &str) -> Vec<BuildTemplate> {
    text.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).filter_map(BuildTemplate::from_line).collect()
}

pub fn serialize(list: &[BuildTemplate]) -> String {
    let mut s = String::from("# FARust build templates: name<TAB>BLUEPRINT,dx,dz;... (milli-units)\n");
    for t in list {
        s.push_str(&t.to_line());
        s.push('\n');
    }
    s
}

fn file() -> Option<PathBuf> {
    crate::settings::config_dir().map(|d| d.join("templates.txt"))
}

pub fn load_all() -> Vec<BuildTemplate> {
    file().and_then(|f| std::fs::read_to_string(f).ok()).map(|t| parse(&t)).unwrap_or_default()
}

pub fn save_all(list: &[BuildTemplate]) -> Result<(), String> {
    let f = file().ok_or("no config directory")?;
    if let Some(dir) = f.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(&f, serialize(list)).map_err(|e| format!("{}: {e}", f.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_round_trip_and_reject_garbage() {
        let t = BuildTemplate { name: "Base A".into(), items: vec![("UEB0101".into(), 0, 0), ("UEB1101".into(), 9_000, -4_500)] };
        let text = serialize(std::slice::from_ref(&t));
        assert_eq!(parse(&text), vec![t.clone()]);
        assert_eq!(t.sites((100, 200))[1], ("UEB1101".to_string(), (9_100, -4_300)));
        assert!(parse("bad line\nx\tUEB0101,1\nx\t../evil,0,0\n").is_empty());
    }
}
