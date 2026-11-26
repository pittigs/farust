//! Parser for Forged Alliance `*_unit.bp` files.
//!
//! Format is a Lua-like pure-data table:
//! `UnitBlueprint { Key = value, Nested = { 'a', 'b' }, Sound { Bank = 'x' } }`
//! No functions occur in unit blueprints (verified over all 568 files).

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Number(f64),
    Str(String),
    Table {
        constructor: Option<String>,
        entries: Vec<Entry>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub key: Option<String>,
    pub value: Value,
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Table { entries, .. } => entries
                .iter()
                .find(|e| e.key.as_deref() == Some(key))
                .map(|e| &e.value),
            _ => None,
        }
    }
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.get(key) {
            Some(Value::Str(s)) => Some(s),
            _ => None,
        }
    }
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        match self.get(key) {
            Some(Value::Number(n)) => Some(*n),
            _ => None,
        }
    }
    /// Boolean field, e.g. `Intel = { Cloak = true }`. Numbers count as truthy when non-zero.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.get(key) {
            Some(Value::Bool(b)) => Some(*b),
            Some(Value::Number(n)) => Some(*n != 0.0),
            _ => None,
        }
    }
    /// Keyed sub-tables in declaration order, e.g. `Enhancements = { Shield = {..}, .. }`.
    pub fn keyed_tables(&self) -> Vec<(&str, &Value)> {
        match self {
            Value::Table { entries, .. } => entries
                .iter()
                .filter_map(|e| match (&e.key, &e.value) {
                    (Some(k), v @ Value::Table { .. }) => Some((k.as_str(), v)),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }
    /// Positional string entries of a table, e.g. `Categories = { 'AIR', ... }`.
    pub fn string_list(&self) -> Vec<String> {
        match self {
            Value::Table { entries, .. } => entries
                .iter()
                .filter(|e| e.key.is_none())
                .filter_map(|e| match &e.value {
                    Value::Str(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }
    /// Positional sub-tables of a table, e.g. `Weapon = { {..}, {..} }`.
    pub fn table_list(&self) -> Vec<&Value> {
        match self {
            Value::Table { entries, .. } => entries
                .iter()
                .filter(|e| e.key.is_none())
                .map(|e| &e.value)
                .filter(|v| matches!(v, Value::Table { .. }))
                .collect(),
            _ => Vec::new(),
        }
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, msg: &str) -> String {
        format!("{msg} at byte {}", self.pos)
    }
    fn skip_ws(&mut self) {
        loop {
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            // Moho blueprints use both `--` and `#` line comments.
            if self.bytes[self.pos..].starts_with(b"--")
                || self.bytes.get(self.pos) == Some(&b'#')
            {
                while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.bytes.get(self.pos).copied()
    }
    fn expect(&mut self, b: u8) -> Result<(), String> {
        if self.peek() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.err(&format!("expected '{}'", b as char)))
        }
    }
    fn parse_ident(&mut self) -> Result<String, String> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos].is_ascii_alphanumeric() || self.bytes[self.pos] == b'_')
        {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(self.err("expected identifier"));
        }
        Ok(String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned())
    }
    fn parse_string(&mut self) -> Result<String, String> {
        let quote = self.peek().ok_or_else(|| self.err("unterminated string"))?;
        self.pos += 1;
        let mut out: Vec<u8> = Vec::new();
        while self.pos < self.bytes.len() {
            let c = self.bytes[self.pos];
            self.pos += 1;
            if c == quote {
                return Ok(String::from_utf8_lossy(&out).into_owned());
            }
            if c == b'\\' && self.pos < self.bytes.len() {
                let e = self.bytes[self.pos];
                self.pos += 1;
                out.push(match e {
                    b'n' => b'\n',
                    b't' => b'\t',
                    b'r' => b'\r',
                    other => other,
                });
            } else {
                out.push(c);
            }
        }
        Err(self.err("unterminated string"))
    }
    fn parse_number(&mut self) -> Result<f64, String> {
        self.skip_ws();
        let start = self.pos;
        if self.bytes.get(self.pos) == Some(&b'-') || self.bytes.get(self.pos) == Some(&b'+') {
            self.pos += 1;
        }
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos].is_ascii_digit()
                || self.bytes[self.pos] == b'.'
                || self.bytes[self.pos] == b'e'
                || self.bytes[self.pos] == b'E'
                || ((self.bytes[self.pos] == b'-' || self.bytes[self.pos] == b'+')
                    && matches!(self.bytes.get(self.pos.wrapping_sub(1)), Some(b'e' | b'E'))))
        {
            self.pos += 1;
        }
        let text = String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned();
        text.parse::<f64>()
            .map_err(|_| self.err(&format!("bad number '{text}'")))
    }
    fn parse_table(&mut self, constructor: Option<String>) -> Result<Value, String> {
        self.expect(b'{')?;
        let mut entries = Vec::new();
        loop {
            match self.peek() {
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                None => return Err(self.err("unterminated table")),
                _ => {}
            }
            // Try `Key = value` first: save position, parse ident, check '='.
            let save = self.pos;
            let key = match self.parse_ident() {
                Ok(name) => {
                    if self.peek() == Some(b'=') {
                        self.pos += 1;
                        Some(name)
                    } else {
                        self.pos = save;
                        None
                    }
                }
                Err(_) => {
                    self.pos = save;
                    None
                }
            };
            let value = self.parse_value()?;
            entries.push(Entry { key, value });
            match self.peek() {
                Some(b',') | Some(b';') => {
                    self.pos += 1;
                }
                Some(b'}') => {}
                other => {
                    return Err(self.err(&format!(
                        "expected ',' or '}}', got {:?}",
                        other.map(|b| b as char)
                    )));
                }
            }
        }
        Ok(Value::Table {
            constructor,
            entries,
        })
    }
    fn parse_value(&mut self) -> Result<Value, String> {
        match self.peek() {
            Some(b'{') => self.parse_table(None),
            Some(b'\'') | Some(b'"') => Ok(Value::Str(self.parse_string()?)),
            Some(b'-') | Some(b'+') => Ok(Value::Number(self.parse_number()?)),
            Some(c) if c.is_ascii_digit() || c == b'.' => Ok(Value::Number(self.parse_number()?)),
            Some(c) if c.is_ascii_alphabetic() || c == b'_' => {
                let ident = self.parse_ident()?;
                match ident.as_str() {
                    "true" => Ok(Value::Bool(true)),
                    "false" => Ok(Value::Bool(false)),
                    "nil" => Ok(Value::Str(String::new())),
                    _ => {
                        if self.peek() == Some(b'{') {
                            self.parse_table(Some(ident))
                        } else {
                            // Bare identifier used as a value: keep as string.
                            Ok(Value::Str(ident))
                        }
                    }
                }
            }
            other => Err(self.err(&format!(
                "unexpected character {:?}",
                other.map(|b| b as char)
            ))),
        }
    }
}

/// Parse a whole blueprint file. Root must be `UnitBlueprint { ... }`.
pub fn parse_blueprint(text: &str) -> Result<Value, String> {
    parse_blueprint_kind(text, "UnitBlueprint")
}

/// Parse a prop blueprint file. Root must be `PropBlueprint { ... }`.
pub fn parse_prop_blueprint(text: &str) -> Result<Value, String> {
    parse_blueprint_kind(text, "PropBlueprint")
}

fn parse_blueprint_kind(text: &str, expected: &str) -> Result<Value, String> {
    let mut p = Parser {
        bytes: text.as_bytes(),
        pos: 0,
    };
    let value = p.parse_value()?;
    if let Value::Table {
        constructor: Some(name),
        ..
    } = &value
    {
        if name != expected {
            return Err(format!("root constructor is '{name}', expected '{expected}'"));
        }
    }
    Ok(value)
}

#[derive(Debug, Clone)]
pub struct WeaponSummary {
    pub label: Option<String>,
    pub damage: Option<f64>,
    pub max_radius: Option<f64>,
    pub rate_of_fire: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct UnitSummary {
    pub id: String,
    pub path: String,
    pub description: Option<String>,
    pub categories: Vec<String>,
    pub health: Option<f64>,
    pub build_cost_mass: Option<f64>,
    pub build_cost_energy: Option<f64>,
    pub build_time: Option<f64>,
    pub max_speed: Option<f64>,
    pub motion_type: Option<String>,
    pub weapons: Vec<WeaponSummary>,
}

pub fn summarize(id: &str, path: &str, root: &Value) -> UnitSummary {
    let defense = root.get("Defense");
    let economy = root.get("Economy");
    let physics = root.get("Physics");
    let weapons = root
        .get("Weapon")
        .map(|w| {
            w.table_list()
                .iter()
                .map(|t| WeaponSummary {
                    label: t.get_str("Label").map(str::to_string),
                    damage: t.get_f64("Damage"),
                    max_radius: t.get_f64("MaxRadius"),
                    rate_of_fire: t.get_f64("RateOfFire"),
                })
                .collect()
        })
        .unwrap_or_default();
    UnitSummary {
        id: id.to_string(),
        path: path.to_string(),
        description: root.get_str("Description").map(str::to_string),
        categories: root
            .get("Categories")
            .map(|c| c.string_list())
            .unwrap_or_default(),
        health: defense.and_then(|d| d.get_f64("MaxHealth").or_else(|| d.get_f64("Health"))),
        build_cost_mass: economy.and_then(|e| e.get_f64("BuildCostMass")),
        build_cost_energy: economy.and_then(|e| e.get_f64("BuildCostEnergy")),
        build_time: economy.and_then(|e| e.get_f64("BuildTime")),
        max_speed: physics.and_then(|p| p.get_f64("MaxSpeed")),
        motion_type: physics
            .and_then(|p| p.get_str("MotionType"))
            .map(str::to_string),
        weapons,
    }
}

/// Phase 24 mod merge: deep-merges `patch` into `base`.
///
/// * keyed entries: both tables -> recurse; otherwise the patch value replaces (or is
///   appended when the key is new);
/// * positional entries: if the patch has positional *tables* (e.g. `Weapon = { {..} }`)
///   they merge element-wise by index (extra ones are appended); positional scalars
///   (e.g. `Categories = { 'A', 'B' }`) replace the base's positional list as a whole;
/// * top-level `AddCategories` / `DelCategories` string lists edit `Categories` in place;
/// * the control keys `Merge`, `BlueprintId`, `AddCategories`, `DelCategories` are never
///   copied into the result.
pub fn merge_into(base: &mut Value, patch: &Value) {
    merge_table(base, patch, true);
}

const MERGE_CONTROL_KEYS: [&str; 4] = ["Merge", "BlueprintId", "AddCategories", "DelCategories"];

fn merge_table(base: &mut Value, patch: &Value, top: bool) {
    let Value::Table { entries: pe, .. } = patch else {
        *base = patch.clone();
        return;
    };
    if !matches!(base, Value::Table { .. }) {
        *base = Value::Table { constructor: None, entries: Vec::new() };
    }
    let Value::Table { entries: be, .. } = base else {
        return;
    };
    let patch_pos: Vec<&Value> = pe.iter().filter(|e| e.key.is_none()).map(|e| &e.value).collect();
    if !patch_pos.is_empty() {
        if patch_pos.iter().all(|v| matches!(v, Value::Table { .. })) {
            let mut idx = 0usize;
            for e in be.iter_mut().filter(|e| e.key.is_none()) {
                if let Some(p) = patch_pos.get(idx) {
                    merge_table(&mut e.value, p, false);
                }
                idx += 1;
            }
            for p in patch_pos.iter().skip(idx) {
                be.push(Entry { key: None, value: (*p).clone() });
            }
        } else {
            be.retain(|e| e.key.is_some());
            be.extend(patch_pos.iter().map(|v| Entry { key: None, value: (*v).clone() }));
        }
    }
    for e in pe.iter() {
        let Some(k) = e.key.as_deref() else {
            continue;
        };
        if top && MERGE_CONTROL_KEYS.contains(&k) {
            continue;
        }
        match be.iter_mut().find(|b| b.key.as_deref() == Some(k)) {
            Some(b) if matches!((&b.value, &e.value), (Value::Table { .. }, Value::Table { .. })) => {
                merge_table(&mut b.value, &e.value, false);
            }
            Some(b) => b.value = e.value.clone(),
            None => be.push(e.clone()),
        }
    }
    if top {
        let add = patch.get("AddCategories").map(Value::string_list).unwrap_or_default();
        let del = patch.get("DelCategories").map(Value::string_list).unwrap_or_default();
        if !add.is_empty() || !del.is_empty() {
            if !be.iter().any(|b| b.key.as_deref() == Some("Categories")) {
                be.push(Entry { key: Some("Categories".into()), value: Value::Table { constructor: None, entries: Vec::new() } });
            }
            if let Some(Entry { value: Value::Table { entries: ce, .. }, .. }) =
                be.iter_mut().find(|b| b.key.as_deref() == Some("Categories"))
            {
                ce.retain(|c| !matches!(&c.value, Value::Str(s) if del.iter().any(|d| d.eq_ignore_ascii_case(s))));
                for a in add {
                    if !ce.iter().any(|c| matches!(&c.value, Value::Str(s) if s.eq_ignore_ascii_case(&a))) {
                        ce.push(Entry { key: None, value: Value::Str(a) });
                    }
                }
            }
        }
    }
}

/// Removes mod control keys from a blueprint that is added (not merged).
pub fn strip_control_keys(v: &mut Value) {
    if let Value::Table { entries, .. } = v {
        entries.retain(|e| !e.key.as_deref().is_some_and(|k| MERGE_CONTROL_KEYS.contains(&k)));
    }
}

/// Unit id from `units/UAL0001/UAL0001_unit.bp` -> `UAL0001`.
pub fn unit_id_from_path(path: &str) -> String {
    let file = path.rsplit('/').next().unwrap_or(path);
    file.strip_suffix("_unit.bp")
        .or_else(|| file.strip_suffix(".bp"))
        .unwrap_or(file)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r"
UnitBlueprint {
    Categories = { 'AEON', 'MOBILE', 'AIR' },
    Defense = { Health = 10, MaxHealth = 10, ArmorType = 'Light' },
    Economy = { BuildCostMass = 100, BuildCostEnergy = -5.5 },
    Physics = { MaxSpeed = 1.7, MotionType = 'RULEUMT_Air' },
    Audio = { AmbientMove = Sound { Bank = 'UAA', Cue = 'Move' } },
    Weapon = {
        { Label = 'Gun', Damage = 12, MaxRadius = 20, RateOfFire = 2 },
    },
    Description = '<LOC x>Test',
}
";

    #[test]
    fn parses_nested_and_constructors() {
        let root = parse_blueprint(SAMPLE).unwrap();
        assert_eq!(root.get_str("Description"), Some("<LOC x>Test"));
        assert_eq!(
            root.get("Categories").unwrap().string_list(),
            vec!["AEON", "MOBILE", "AIR"]
        );
        let sound = root.get("Audio").unwrap().get("AmbientMove").unwrap();
        assert_eq!(sound.get_str("Bank"), Some("UAA"));
    }

    #[test]
    fn summarizes_fields() {
        let root = parse_blueprint(SAMPLE).unwrap();
        let s = summarize("TST0001", "units/TST0001/TST0001_unit.bp", &root);
        assert_eq!(s.health, Some(10.0));
        assert_eq!(s.build_cost_energy, Some(-5.5));
        assert_eq!(s.max_speed, Some(1.7));
        assert_eq!(s.weapons.len(), 1);
        assert_eq!(s.weapons[0].damage, Some(12.0));
    }

    #[test]
    fn skips_hash_comments() {
        let root = parse_blueprint(
            "UnitBlueprint {\n Categories = {\n 'AEON',\n # 'IGNORED',\n 'AIR',\n },\n}\n",
        )
        .unwrap();
        assert_eq!(
            root.get("Categories").unwrap().string_list(),
            vec!["AEON", "AIR"]
        );
    }

    #[test]
    fn reads_bools_and_keyed_tables() {
        let root = parse_blueprint(
            "UnitBlueprint { Intel = { Cloak = true, RadarStealth = false, JammerBlips = 5 },\n Enhancements = { Shield = { Slot = 'Back' }, Slots = { Back = { x = 1 } }, Odd = 3 } }",
        )
        .unwrap();
        let intel = root.get("Intel").unwrap();
        assert_eq!(intel.get_bool("Cloak"), Some(true));
        assert_eq!(intel.get_bool("RadarStealth"), Some(false));
        assert_eq!(intel.get_bool("JammerBlips"), Some(true));
        assert_eq!(intel.get_bool("Missing"), None);
        let names: Vec<&str> = root.get("Enhancements").unwrap().keyed_tables().iter().map(|(k, _)| *k).collect();
        assert_eq!(names, vec!["Shield", "Slots"]);
    }

    #[test]
    fn merge_semantics() {
        let mut base = parse_blueprint(
            "UnitBlueprint { Categories = {'UEF','LAND'}, Defense = { MaxHealth = 100, ArmorType = 'Normal' }, Weapon = { { Label = 'Gun', Damage = 10 }, { Label = 'AA', Damage = 3 } } }",
        )
        .unwrap();
        let patch = parse_blueprint(
            "UnitBlueprint { Merge = true, BlueprintId = 'x', AddCategories = {'NEWCAT'}, DelCategories = {'land'}, Defense = { MaxHealth = 250 }, Weapon = { { Damage = 42 } }, Intel = { VisionRadius = 30 } }",
        )
        .unwrap();
        merge_into(&mut base, &patch);
        assert_eq!(base.get("Categories").unwrap().string_list(), vec!["UEF", "NEWCAT"]);
        assert_eq!(base.get("Defense").unwrap().get_f64("MaxHealth"), Some(250.0));
        assert_eq!(base.get("Defense").unwrap().get_str("ArmorType"), Some("Normal"));
        let w = base.get("Weapon").unwrap().table_list();
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].get_f64("Damage"), Some(42.0));
        assert_eq!(w[0].get_str("Label"), Some("Gun"));
        assert_eq!(w[1].get_f64("Damage"), Some(3.0));
        assert_eq!(base.get("Intel").unwrap().get_f64("VisionRadius"), Some(30.0));
        assert!(base.get("Merge").is_none() && base.get("AddCategories").is_none());
        // Positional scalar lists replace as a whole.
        let p2 = parse_blueprint("UnitBlueprint { Categories = {'A'} }").unwrap();
        merge_into(&mut base, &p2);
        assert_eq!(base.get("Categories").unwrap().string_list(), vec!["A"]);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_blueprint("UnitBlueprint { oops ").is_err());
    }

    #[test]
    fn extracts_id() {
        assert_eq!(unit_id_from_path("units/UAL0001/UAL0001_unit.bp"), "UAL0001");
    }
}
