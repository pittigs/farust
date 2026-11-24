use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ScenarioMeta {
    pub id: String,
    pub name: String,
    pub description: String,
    pub scenario_type: String,
    pub size: (u32, u32),
    pub map_vfs_path: String,
    pub save_vfs_path: String,
    pub armies: Vec<String>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Marker {
    pub name: String,
    pub marker_type: String,
    pub position: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct SaveMarkers {
    pub start_spots: Vec<Marker>,
    pub mass_points: Vec<Marker>,
    pub hydro_points: Vec<Marker>,
    pub total_markers: usize,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MapFiles {
    pub id: String,
    pub dir: PathBuf,
    pub scenario_path: PathBuf,
    pub save_path: Option<PathBuf>,
    pub scmap_path: Option<PathBuf>,
}

/// Resolves a Forged Alliance virtual filesystem path (e.g. `/maps/SCMP_001/SCMP_001.scmap`)
/// case-insensitively on disk relative to `game_dir`.
pub fn resolve_vfs_path(game_dir: &Path, vfs_path: &str) -> Option<PathBuf> {
    let normalized = vfs_path.replace('\\', "/");
    let trimmed = normalized.trim_start_matches('/');
    let mut segments = trimmed.split('/').filter(|s| !s.is_empty()).peekable();
    segments.peek()?;

    let mut current = game_dir.to_path_buf();
    for seg in segments {
        let exact = current.join(seg);
        if exact.exists() {
            current = exact;
            continue;
        }
        let entries = std::fs::read_dir(&current).ok()?;
        let mut matched: Option<PathBuf> = None;
        for entry in entries.flatten() {
            let name = entry.file_name();
            if name.to_string_lossy().eq_ignore_ascii_case(seg) {
                matched = Some(entry.path());
                break;
            }
        }
        current = matched?;
    }

    Some(current)
}

/// Discovers all maps under `<game_dir>/maps` and resolves their scenario, save, and `.scmap` paths.
pub fn list_maps(game_dir: &Path) -> Result<Vec<MapFiles>, String> {
    let maps_dir = resolve_vfs_path(game_dir, "maps")
        .filter(|p| p.is_dir())
        .unwrap_or_else(|| game_dir.join("maps"));

    let entries = std::fs::read_dir(&maps_dir)
        .map_err(|e| format!("failed to read maps directory {}: {e}", maps_dir.display()))?;

    let mut subdirs = Vec::new();
    for entry in entries {
        let entry = entry
            .map_err(|e| format!("failed to read entry in {}: {e}", maps_dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            let id = entry.file_name().to_string_lossy().into_owned();
            subdirs.push((id, path));
        }
    }

    let mut maps = Vec::with_capacity(subdirs.len());
    for (id, dir) in subdirs {
        let files = list_dir_files_sorted(&dir)?;
        let Some(scenario_path) = find_file_by_suffix(&files, "_scenario.lua") else {
            continue;
        };

        let meta = parse_scenario_file(&scenario_path, &id)?;

        let scmap_path = resolve_vfs_path(game_dir, &meta.map_vfs_path)
            .filter(|p| p.is_file())
            .or_else(|| find_file_by_suffix(&files, ".scmap"));

        let save_path = resolve_vfs_path(game_dir, &meta.save_vfs_path)
            .filter(|p| p.is_file())
            .or_else(|| find_file_by_suffix(&files, "_save.lua"));

        maps.push(MapFiles {
            id,
            dir,
            scenario_path,
            save_path,
            scmap_path,
        });
    }

    maps.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(maps)
}

/// Finds a map by folder id (case-insensitive) under `<game_dir>/maps`.
pub fn find_map(game_dir: &Path, name: &str) -> Result<MapFiles, String> {
    let maps = list_maps(game_dir)?;
    maps.into_iter()
        .find(|m| m.id.eq_ignore_ascii_case(name))
        .ok_or_else(|| format!("map '{name}' not found in {}", game_dir.display()))
}

pub fn parse_scenario_file(path: &Path, id: &str) -> Result<ScenarioMeta, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read scenario file {}: {e}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes);
    parse_scenario(&text, id)
}

pub fn parse_scenario(text: &str, id: &str) -> Result<ScenarioMeta, String> {
    let mut parser = Parser::new(text)?;
    let root = parser.parse_top_level()?;
    let info = root
        .get("ScenarioInfo")
        .and_then(|v| v.as_table())
        .ok_or_else(|| format!("missing ScenarioInfo table for scenario '{id}'"))?;

    let name = info
        .get("name")
        .and_then(|v| v.as_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| id.to_string());

    let description = info
        .get("description")
        .and_then(|v| v.as_string())
        .unwrap_or_default();

    let scenario_type = info
        .get("type")
        .and_then(|v| v.as_string())
        .unwrap_or_default();

    let size = extract_size(info.get("size"));

    let map_vfs_path = info
        .get("map")
        .and_then(|v| v.as_string())
        .unwrap_or_default();

    let save_vfs_path = info
        .get("save")
        .and_then(|v| v.as_string())
        .unwrap_or_default();

    let armies = extract_armies(info);

    Ok(ScenarioMeta {
        id: id.to_string(),
        name,
        description,
        scenario_type,
        size,
        map_vfs_path,
        save_vfs_path,
        armies,
    })
}

pub fn parse_save_markers_file(path: &Path, armies: &[String]) -> Result<SaveMarkers, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read save file {}: {e}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes);
    parse_save_markers(&text, armies)
}

/// Extracts markers exclusively from `MasterChain -> ['_MASTERCHAIN_'] -> Markers`.
pub fn parse_save_markers(text: &str, armies: &[String]) -> Result<SaveMarkers, String> {
    let mut scanner = Lexer::new(text);
    let brace_pos = scanner
        .find_table_value_start("MasterChain")?
        .ok_or_else(|| "missing MasterChain table in save file".to_string())?;

    let mut parser = Parser::new(&scanner.src[brace_pos..])?;
    let master_chain = parser.parse_table()?;

    let master_inner = master_chain
        .get("_MASTERCHAIN_")
        .and_then(|v| v.as_table())
        .ok_or_else(|| "missing '_MASTERCHAIN_' table inside MasterChain".to_string())?;

    let markers_table = master_inner
        .get("Markers")
        .and_then(|v| v.as_table())
        .ok_or_else(|| "missing 'Markers' table inside '_MASTERCHAIN_'".to_string())?;

    let mut start_spots = Vec::new();
    let mut mass_points = Vec::new();
    let mut hydro_points = Vec::new();
    let mut total_markers = 0usize;

    for (key, val) in &markers_table.entries {
        let Some(marker_name) = key.as_ref() else {
            continue;
        };
        let Some(marker_table) = val.as_table() else {
            continue;
        };

        let marker_type = marker_table
            .get("type")
            .and_then(|v| v.as_string())
            .unwrap_or_default();

        let position = marker_table
            .get("position")
            .and_then(|v| v.as_vec3())
            .unwrap_or([0.0, 0.0, 0.0]);

        total_markers += 1;

        let is_start =
            marker_name.starts_with("ARMY_") || armies.iter().any(|a| a == marker_name);

        if is_start {
            start_spots.push(Marker {
                name: marker_name.clone(),
                marker_type,
                position,
            });
        } else if marker_type == "Mass" {
            mass_points.push(Marker {
                name: marker_name.clone(),
                marker_type,
                position,
            });
        } else if marker_type == "Hydrocarbon" {
            hydro_points.push(Marker {
                name: marker_name.clone(),
                marker_type,
                position,
            });
        }
    }

    start_spots.sort_by(|a, b| a.name.cmp(&b.name));
    mass_points.sort_by(|a, b| a.name.cmp(&b.name));
    hydro_points.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(SaveMarkers {
        start_spots,
        mass_points,
        hydro_points,
        total_markers,
    })
}

fn list_dir_files_sorted(dir: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("failed to read directory {}: {e}", dir.display()))?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry
            .map_err(|e| format!("failed to read entry in {}: {e}", dir.display()))?;
        let path = entry.path();
        if path.is_file() {
            let name = entry.file_name().to_string_lossy().into_owned();
            files.push((name, path));
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

fn find_file_by_suffix(files: &[(String, PathBuf)], suffix: &str) -> Option<PathBuf> {
    files.iter().find_map(|(name, path)| {
        if ends_with_ignore_ascii_case(name, suffix) {
            Some(path.clone())
        } else {
            None
        }
    })
}

fn ends_with_ignore_ascii_case(s: &str, suffix: &str) -> bool {
    let sb = s.as_bytes();
    let tb = suffix.as_bytes();
    sb.len() >= tb.len() && sb[sb.len() - tb.len()..].eq_ignore_ascii_case(tb)
}

fn extract_size(val: Option<&LuaVal>) -> (u32, u32) {
    let Some(table) = val.and_then(|v| v.as_table()) else {
        return (0, 0);
    };
    let mut dims = [0u32; 2];
    let mut count = 0;
    for (_, entry_val) in &table.entries {
        if let Some(n) = entry_val.as_f64() {
            dims[count] = n.max(0.0).round() as u32;
            count += 1;
            if count == 2 {
                return (dims[0], dims[1]);
            }
        }
    }
    (0, 0)
}

fn extract_armies(info: &LuaTable) -> Vec<String> {
    let mut armies = Vec::new();
    let mut seen = HashSet::new();

    let Some(teams) = info
        .get("Configurations")
        .and_then(|v| v.as_table())
        .and_then(|cfg| cfg.get("standard"))
        .and_then(|v| v.as_table())
        .and_then(|std_cfg| std_cfg.get("teams"))
        .and_then(|v| v.as_table())
    else {
        return armies;
    };

    for (_, team_val) in &teams.entries {
        let Some(team_table) = team_val.as_table() else {
            continue;
        };
        let Some(team_armies) = team_table.get("armies").and_then(|v| v.as_table()) else {
            continue;
        };
        for (_, army_val) in &team_armies.entries {
            if let Some(army_name) = army_val.as_string()
                && seen.insert(army_name.clone())
            {
                armies.push(army_name);
            }
        }
    }

    armies
}

#[derive(Debug, Clone)]
struct LuaTable {
    entries: Vec<(Option<String>, LuaVal)>,
}

impl LuaTable {
    fn get(&self, key: &str) -> Option<&LuaVal> {
        self.entries
            .iter()
            .rev()
            .find_map(|(k, v)| match k.as_deref() {
                Some(name) if name == key => Some(v),
                _ => None,
            })
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum LuaVal {
    Nil,
    Bool(bool),
    Num(f64),
    Str(String),
    Ident(String),
    Table(LuaTable),
    Call(String, Vec<LuaVal>),
}

impl LuaVal {
    fn as_string(&self) -> Option<String> {
        match self {
            LuaVal::Str(s) => Some(s.clone()),
            LuaVal::Call(name, args) if name.eq_ignore_ascii_case("STRING") => {
                args.first().and_then(|v| v.as_string())
            }
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            LuaVal::Num(n) => Some(*n),
            LuaVal::Call(name, args) if name.eq_ignore_ascii_case("FLOAT") => {
                args.first().and_then(|v| v.as_f64())
            }
            _ => None,
        }
    }

    fn as_table(&self) -> Option<&LuaTable> {
        match self {
            LuaVal::Table(t) => Some(t),
            LuaVal::Call(_, args) if args.len() == 1 => args[0].as_table(),
            _ => None,
        }
    }

    fn as_vec3(&self) -> Option<[f32; 3]> {
        match self {
            LuaVal::Call(name, args)
                if name.eq_ignore_ascii_case("VECTOR3") && args.len() >= 3 =>
            {
                Some([
                    args[0].as_f64()? as f32,
                    args[1].as_f64()? as f32,
                    args[2].as_f64()? as f32,
                ])
            }
            LuaVal::Table(t) => {
                let mut nums = Vec::with_capacity(3);
                for (_, v) in &t.entries {
                    if let Some(n) = v.as_f64() {
                        nums.push(n as f32);
                        if nums.len() == 3 {
                            return Some([nums[0], nums[1], nums[2]]);
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Str(String),
    Num(f64),
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    LParen,
    RParen,
    Eq,
    Comma,
    Semi,
    Dot,
    DotDot,
    Minus,
    Plus,
}

struct Lexer<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        let src = input.strip_prefix('\u{FEFF}').unwrap_or(input);
        Self {
            src,
            bytes: src.as_bytes(),
            pos: 0,
        }
    }

    fn parse_long_bracket(&self, start: usize) -> Option<(usize, usize)> {
        if *self.bytes.get(start)? != b'[' {
            return None;
        }
        let mut i = start + 1;
        while self.bytes.get(i) == Some(&b'=') {
            i += 1;
        }
        if self.bytes.get(i) == Some(&b'[') {
            Some((i - start - 1, i + 1))
        } else {
            None
        }
    }

    fn scan_long_bracket_end(&self, content_start: usize, eq_count: usize) -> Option<(usize, usize)> {
        let mut i = content_start;
        while i < self.bytes.len() {
            if self.bytes[i] == b']' {
                let mut j = i + 1;
                let mut eq = 0;
                while eq < eq_count && self.bytes.get(j) == Some(&b'=') {
                    j += 1;
                    eq += 1;
                }
                if eq == eq_count && self.bytes.get(j) == Some(&b']') {
                    return Some((i, j + 1));
                }
            }
            i += 1;
        }
        None
    }

    fn skip_whitespace_and_comments(&mut self) -> Result<(), String> {
        loop {
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.pos >= self.bytes.len() {
                return Ok(());
            }
            if self.bytes[self.pos] == b'#' {
                self.pos += 1;
                while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                    self.pos += 1;
                }
                continue;
            }
            if self.bytes[self.pos] == b'-' && self.bytes.get(self.pos + 1) == Some(&b'-') {
                self.pos += 2;
                if let Some((eq_count, content_start)) = self.parse_long_bracket(self.pos) {
                    let (_, after_close) = self
                        .scan_long_bracket_end(content_start, eq_count)
                        .ok_or_else(|| "unterminated Lua block comment".to_string())?;
                    self.pos = after_close;
                } else {
                    while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                        self.pos += 1;
                    }
                }
                continue;
            }
            return Ok(());
        }
    }

    fn skip_quoted_string(&mut self, quote: u8) -> Result<(), String> {
        self.pos += 1;
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == quote {
                self.pos += 1;
                return Ok(());
            }
            if b == b'\\' {
                self.pos += 1;
                let ch = self.src[self.pos..]
                    .chars()
                    .next()
                    .ok_or_else(|| "unterminated escape in Lua string".to_string())?;
                self.pos += ch.len_utf8();
                continue;
            }
            if b < 0x80 {
                self.pos += 1;
            } else if let Some(ch) = self.src[self.pos..].chars().next() {
                self.pos += ch.len_utf8();
            }
        }
        Err("unterminated Lua string".to_string())
    }

    fn read_quoted_string(&mut self, quote: u8) -> Result<String, String> {
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == quote {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.pos += 1;
                let escaped = self.src[self.pos..]
                    .chars()
                    .next()
                    .ok_or_else(|| "unterminated escape in Lua string".to_string())?;
                self.pos += escaped.len_utf8();
                match escaped {
                    '\\' => out.push('\\'),
                    '\'' => out.push('\''),
                    '"' => out.push('"'),
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'r' => out.push('\r'),
                    'a' => out.push('\x07'),
                    'b' => out.push('\x08'),
                    'f' => out.push('\x0C'),
                    'v' => out.push('\x0B'),
                    '\n' => out.push('\n'),
                    '\r' => {
                        if self.bytes.get(self.pos) == Some(&b'\n') {
                            self.pos += 1;
                        }
                        out.push('\n');
                    }
                    '0'..='9' => {
                        let mut val = (escaped as u8 - b'0') as u32;
                        for _ in 0..2 {
                            if let Some(&d) = self.bytes.get(self.pos)
                                && d.is_ascii_digit()
                            {
                                let next_val = val * 10 + (d - b'0') as u32;
                                if next_val <= 255 {
                                    val = next_val;
                                    self.pos += 1;
                                    continue;
                                }
                            }
                            break;
                        }
                        if let Some(c) = char::from_u32(val) {
                            out.push(c);
                        }
                    }
                    other => out.push(other),
                }
                continue;
            }
            if b < 0x80 {
                out.push(b as char);
                self.pos += 1;
            } else if let Some(ch) = self.src[self.pos..].chars().next() {
                out.push(ch);
                self.pos += ch.len_utf8();
            }
        }
        Err("unterminated Lua string".to_string())
    }

    fn read_long_string(&mut self, eq_count: usize, content_start: usize) -> Result<String, String> {
        let (content_end, after_close) = self
            .scan_long_bracket_end(content_start, eq_count)
            .ok_or_else(|| "unterminated Lua long string".to_string())?;
        let mut s = &self.src[content_start..content_end];
        if let Some(rest) = s.strip_prefix("\r\n") {
            s = rest;
        } else if let Some(rest) = s.strip_prefix('\n') {
            s = rest;
        }
        self.pos = after_close;
        Ok(s.to_string())
    }

    fn read_number(&mut self) -> Result<f64, String> {
        let start = self.pos;
        if self.bytes[self.pos] == b'0'
            && matches!(self.bytes.get(self.pos + 1), Some(b'x' | b'X'))
        {
            self.pos += 2;
            let hex_start = self.pos;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_hexdigit() {
                self.pos += 1;
            }
            if self.pos == hex_start {
                return Err("invalid hex literal in Lua".to_string());
            }
            let hex_str = &self.src[hex_start..self.pos];
            let val = u64::from_str_radix(hex_str, 16)
                .map_err(|e| format!("invalid hex literal '{hex_str}': {e}"))?;
            return Ok(val as f64);
        }

        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.bytes.get(self.pos) == Some(&b'.') && self.bytes.get(self.pos + 1) != Some(&b'.') {
            self.pos += 1;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        if matches!(self.bytes.get(self.pos), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.bytes.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let exp_start = self.pos;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
            if self.pos == exp_start {
                return Err("invalid exponent in Lua number".to_string());
            }
        }

        let num_str = &self.src[start..self.pos];
        num_str
            .parse::<f64>()
            .map_err(|e| format!("invalid number '{num_str}': {e}"))
    }

    fn next_token(&mut self) -> Result<Option<Token>, String> {
        self.skip_whitespace_and_comments()?;
        if self.pos >= self.bytes.len() {
            return Ok(None);
        }
        let b = self.bytes[self.pos];
        match b {
            b'{' => {
                self.pos += 1;
                Ok(Some(Token::LBrace))
            }
            b'}' => {
                self.pos += 1;
                Ok(Some(Token::RBrace))
            }
            b']' => {
                self.pos += 1;
                Ok(Some(Token::RBracket))
            }
            b'(' => {
                self.pos += 1;
                Ok(Some(Token::LParen))
            }
            b')' => {
                self.pos += 1;
                Ok(Some(Token::RParen))
            }
            b'=' => {
                self.pos += 1;
                Ok(Some(Token::Eq))
            }
            b',' => {
                self.pos += 1;
                Ok(Some(Token::Comma))
            }
            b';' => {
                self.pos += 1;
                Ok(Some(Token::Semi))
            }
            b'+' => {
                self.pos += 1;
                Ok(Some(Token::Plus))
            }
            b'-' => {
                self.pos += 1;
                Ok(Some(Token::Minus))
            }
            b'\'' | b'"' => Ok(Some(Token::Str(self.read_quoted_string(b)?))),
            b'[' => {
                if let Some((eq_count, content_start)) = self.parse_long_bracket(self.pos) {
                    Ok(Some(Token::Str(
                        self.read_long_string(eq_count, content_start)?,
                    )))
                } else {
                    self.pos += 1;
                    Ok(Some(Token::LBracket))
                }
            }
            b'.' => {
                if self.bytes.get(self.pos + 1) == Some(&b'.') {
                    self.pos += 2;
                    Ok(Some(Token::DotDot))
                } else if self.bytes.get(self.pos + 1).is_some_and(|c| c.is_ascii_digit()) {
                    Ok(Some(Token::Num(self.read_number()?)))
                } else {
                    self.pos += 1;
                    Ok(Some(Token::Dot))
                }
            }
            _ if b.is_ascii_digit() => Ok(Some(Token::Num(self.read_number()?))),
            _ if is_ident_start(b) => {
                let start = self.pos;
                self.pos += 1;
                while self.pos < self.bytes.len() && is_ident_continue(self.bytes[self.pos]) {
                    self.pos += 1;
                }
                Ok(Some(Token::Ident(self.src[start..self.pos].to_string())))
            }
            _ => {
                let ch = self.src[self.pos..].chars().next().unwrap_or('\0');
                Err(format!("unexpected character '{ch}' at byte {}", self.pos))
            }
        }
    }

    fn consume_eq_open_brace(&mut self) -> Result<Option<usize>, String> {
        self.skip_whitespace_and_comments()?;
        if self.bytes.get(self.pos) == Some(&b'=') {
            self.pos += 1;
            self.skip_whitespace_and_comments()?;
            if self.bytes.get(self.pos) == Some(&b'{') {
                return Ok(Some(self.pos));
            }
        }
        Ok(None)
    }

    fn find_table_value_start(&mut self, target_key: &str) -> Result<Option<usize>, String> {
        loop {
            self.skip_whitespace_and_comments()?;
            if self.pos >= self.bytes.len() {
                return Ok(None);
            }
            let b = self.bytes[self.pos];
            if b == b'\'' || b == b'"' {
                self.skip_quoted_string(b)?;
                continue;
            }
            if let Some((eq_count, content_start)) = self.parse_long_bracket(self.pos) {
                let (_, after_close) = self
                    .scan_long_bracket_end(content_start, eq_count)
                    .ok_or_else(|| "unterminated Lua long string".to_string())?;
                self.pos = after_close;
                continue;
            }
            if b == b'[' {
                self.pos += 1;
                self.skip_whitespace_and_comments()?;
                if self.pos < self.bytes.len()
                    && (self.bytes[self.pos] == b'\'' || self.bytes[self.pos] == b'"')
                {
                    let q = self.bytes[self.pos];
                    let s = self.read_quoted_string(q)?;
                    if s == target_key {
                        self.skip_whitespace_and_comments()?;
                        if self.bytes.get(self.pos) == Some(&b']') {
                            self.pos += 1;
                            if let Some(brace_pos) = self.consume_eq_open_brace()? {
                                return Ok(Some(brace_pos));
                            }
                        }
                    }
                }
                continue;
            }
            if is_ident_start(b) {
                let start = self.pos;
                self.pos += 1;
                while self.pos < self.bytes.len() && is_ident_continue(self.bytes[self.pos]) {
                    self.pos += 1;
                }
                if &self.src[start..self.pos] == target_key
                    && let Some(brace_pos) = self.consume_eq_open_brace()?
                {
                    return Ok(Some(brace_pos));
                }
                continue;
            }
            if b.is_ascii_digit()
                || (b == b'.' && self.bytes.get(self.pos + 1).is_some_and(|c| c.is_ascii_digit()))
            {
                let _ = self.read_number()?;
                continue;
            }
            if b < 0x80 {
                self.pos += 1;
            } else if let Some(ch) = self.src[self.pos..].chars().next() {
                self.pos += ch.len_utf8();
            }
        }
    }
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

struct Parser<'a> {
    lexer: Lexer<'a>,
    cur: Option<Token>,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Result<Self, String> {
        let mut lexer = Lexer::new(src);
        let cur = lexer.next_token()?;
        Ok(Self { lexer, cur })
    }

    fn advance(&mut self) -> Result<Option<Token>, String> {
        let next = self.lexer.next_token()?;
        Ok(std::mem::replace(&mut self.cur, next))
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        if self.cur.as_ref() == Some(expected) {
            self.advance()?;
            Ok(())
        } else {
            Err(format!("expected {expected:?}, found {:?}", self.cur))
        }
    }

    fn consume_dotted_ident(&mut self, name: &mut String) -> Result<(), String> {
        while self.cur == Some(Token::Dot) {
            self.advance()?;
            match self.cur.clone() {
                Some(Token::Ident(sub)) => {
                    self.advance()?;
                    name.push('.');
                    name.push_str(&sub);
                }
                other => {
                    return Err(format!("expected identifier after '.', found {other:?}"));
                }
            }
        }
        Ok(())
    }

    fn parse_top_level(&mut self) -> Result<LuaTable, String> {
        let mut entries = Vec::new();
        while let Some(tok) = self.cur.clone() {
            match tok {
                Token::Semi => {
                    self.advance()?;
                }
                Token::Ident(mut name) => {
                    self.advance()?;
                    if name == "local" {
                        if let Some(Token::Ident(local_name)) = self.cur.clone() {
                            self.advance()?;
                            if self.cur == Some(Token::Eq) {
                                self.advance()?;
                                let val = self.parse_expr()?;
                                entries.push((Some(local_name), val));
                            }
                        }
                        continue;
                    }
                    self.consume_dotted_ident(&mut name)?;
                    if self.cur == Some(Token::Eq) {
                        self.advance()?;
                        let val = self.parse_expr()?;
                        entries.push((Some(name), val));
                    } else if self.cur == Some(Token::LParen) || self.cur == Some(Token::LBrace) {
                        let _ = self.parse_ident_postfix(name)?;
                    } else {
                        return Err(format!(
                            "expected '=' after top-level identifier '{name}'"
                        ));
                    }
                }
                other => {
                    return Err(format!("unexpected top-level token: {other:?}"));
                }
            }
        }
        Ok(LuaTable { entries })
    }

    fn parse_table(&mut self) -> Result<LuaTable, String> {
        self.expect(&Token::LBrace)?;
        let mut entries = Vec::new();
        loop {
            if self.cur == Some(Token::RBrace) {
                self.advance()?;
                break;
            }
            if self.cur.is_none() {
                return Err("unterminated Lua table '{'".to_string());
            }

            let entry = match self.cur.clone() {
                Some(Token::LBracket) => {
                    self.advance()?;
                    let key = self.parse_bracket_key()?;
                    self.expect(&Token::RBracket)?;
                    self.expect(&Token::Eq)?;
                    let val = self.parse_expr()?;
                    (Some(key), val)
                }
                Some(Token::Ident(name)) if !matches!(name.as_str(), "true" | "false" | "nil") => {
                    self.advance()?;
                    if self.cur == Some(Token::Eq) {
                        self.advance()?;
                        let val = self.parse_expr()?;
                        (Some(name), val)
                    } else {
                        let primary = self.parse_ident_postfix(name)?;
                        let val = self.parse_expr_tail(primary)?;
                        (None, val)
                    }
                }
                _ => {
                    let val = self.parse_expr()?;
                    (None, val)
                }
            };
            entries.push(entry);

            match self.cur {
                Some(Token::Comma | Token::Semi) => {
                    self.advance()?;
                }
                Some(Token::RBrace) => {
                    self.advance()?;
                    break;
                }
                _ => {
                    return Err(format!(
                        "expected ',', ';', or '}}' in Lua table, found {:?}",
                        self.cur
                    ));
                }
            }
        }
        Ok(LuaTable { entries })
    }

    fn parse_bracket_key(&mut self) -> Result<String, String> {
        let val = self.parse_expr()?;
        match val {
            LuaVal::Str(s) => Ok(s),
            LuaVal::Num(n) => Ok(n.to_string()),
            LuaVal::Ident(s) => Ok(s),
            LuaVal::Bool(b) => Ok(b.to_string()),
            other => Err(format!("unsupported table bracket key: {other:?}")),
        }
    }

    fn parse_expr(&mut self) -> Result<LuaVal, String> {
        let primary = self.parse_primary()?;
        self.parse_expr_tail(primary)
    }

    fn parse_expr_tail(&mut self, mut lhs: LuaVal) -> Result<LuaVal, String> {
        while self.cur == Some(Token::DotDot) {
            self.advance()?;
            let rhs = self.parse_primary()?;
            lhs = match (lhs, rhs) {
                (LuaVal::Str(mut a), LuaVal::Str(b)) => {
                    a.push_str(&b);
                    LuaVal::Str(a)
                }
                (LuaVal::Str(mut a), LuaVal::Num(n)) => {
                    a.push_str(&n.to_string());
                    LuaVal::Str(a)
                }
                (LuaVal::Num(n), LuaVal::Str(b)) => LuaVal::Str(format!("{n}{b}")),
                (a, b) => match (a.as_string(), b.as_string()) {
                    (Some(mut sa), Some(sb)) => {
                        sa.push_str(&sb);
                        LuaVal::Str(sa)
                    }
                    _ => return Err("cannot concatenate non-string Lua values".to_string()),
                },
            };
        }
        Ok(lhs)
    }

    fn parse_primary(&mut self) -> Result<LuaVal, String> {
        match self.cur.clone() {
            Some(Token::Str(s)) => {
                self.advance()?;
                Ok(LuaVal::Str(s))
            }
            Some(Token::Num(n)) => {
                self.advance()?;
                Ok(LuaVal::Num(n))
            }
            Some(Token::Minus) => {
                self.advance()?;
                match self.parse_primary()? {
                    LuaVal::Num(n) => Ok(LuaVal::Num(-n)),
                    other => Err(format!("cannot apply unary '-' to {other:?}")),
                }
            }
            Some(Token::Plus) => {
                self.advance()?;
                match self.parse_primary()? {
                    LuaVal::Num(n) => Ok(LuaVal::Num(n)),
                    other => Err(format!("cannot apply unary '+' to {other:?}")),
                }
            }
            Some(Token::LBrace) => Ok(LuaVal::Table(self.parse_table()?)),
            Some(Token::LParen) => {
                self.advance()?;
                let val = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(val)
            }
            Some(Token::Ident(name)) => {
                self.advance()?;
                match name.as_str() {
                    "true" => Ok(LuaVal::Bool(true)),
                    "false" => Ok(LuaVal::Bool(false)),
                    "nil" => Ok(LuaVal::Nil),
                    _ => self.parse_ident_postfix(name),
                }
            }
            other => Err(format!("unexpected token in expression: {other:?}")),
        }
    }

    fn parse_ident_postfix(&mut self, mut name: String) -> Result<LuaVal, String> {
        self.consume_dotted_ident(&mut name)?;
        if self.cur == Some(Token::LParen) {
            self.advance()?;
            let mut args = Vec::new();
            if self.cur != Some(Token::RParen) {
                loop {
                    args.push(self.parse_expr()?);
                    if self.cur == Some(Token::Comma) {
                        self.advance()?;
                        if self.cur == Some(Token::RParen) {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }
            self.expect(&Token::RParen)?;
            Ok(LuaVal::Call(name, args))
        } else if self.cur == Some(Token::LBrace) {
            let table = self.parse_table()?;
            Ok(LuaVal::Call(name, vec![LuaVal::Table(table)]))
        } else {
            Ok(LuaVal::Ident(name))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TempDirGuard(PathBuf);

    impl TempDirGuard {
        fn new(prefix: &str) -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let dir = std::env::temp_dir().join(format!("farust_scenario_{prefix}_{pid}_{seq}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("create temp test directory");
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn parse_scenario_standard_and_helpers() {
        let lua = r#"
--[[                                                                           ]]--
--[[  File     :  /maps/SCMP_001/SCMP_001_scenario.lua                         ]]--
--[[                                                                           ]]--
# optional hash comment
version = 3 -- trailing line comment
ScenarioInfo = {
    name = STRING("Seton's Clutch"),
    description = "Line1\n\t\"Quoted\" \'Single\' \\Slash\r",
    preview = '',
    map_version = 1,
    type = 'skirmish',
    starts = true,
    size = {1024, 1024},
    reclaim = {0, 0},
    map = '/maps/SCMP_001/SCMP_001.scmap',
    save = '/maps/SCMP_001/SCMP_001_save.lua',
    Configurations = {
        ['standard'] = {
            teams = {
                {
                    name = 'FFA',
                    armies = {'ARMY_1', 'ARMY_2'},
                },
                {
                    name = 'EXTRA',
                    armies = {'ARMY_2', 'ARMY_3'},
                },
            },
        },
    },
}
"#;
        let meta = parse_scenario(lua, "SCMP_001").expect("parse scenario");
        assert_eq!(meta.id, "SCMP_001");
        assert_eq!(meta.name, "Seton's Clutch");
        assert_eq!(meta.description, "Line1\n\t\"Quoted\" 'Single' \\Slash\r");
        assert_eq!(meta.scenario_type, "skirmish");
        assert_eq!(meta.size, (1024, 1024));
        assert_eq!(meta.map_vfs_path, "/maps/SCMP_001/SCMP_001.scmap");
        assert_eq!(meta.save_vfs_path, "/maps/SCMP_001/SCMP_001_save.lua");
        assert_eq!(meta.armies, vec!["ARMY_1", "ARMY_2", "ARMY_3"]);
    }

    #[test]
    fn parse_scenario_name_fallback_and_lua_syntax() {
        let lua = r#"
version = 3;
ScenarioInfo = {
    ["description"] = [[Raw \n multiline
description]],
    type = 'campaign',
    size = { [1] = 256, [2] = 512 },
    map = '/maps/X1CA_001/X1CA_001.scmap',
    save = '/maps/X1CA_001/X1CA_001_save.lua',
    extra_rect = RECTANGLE(0, 0, 256, 512),
    extra_group = GROUP { count = FLOAT(3.5), enabled = BOOLEAN(false) },
    joined = 'PartA' .. 'PartB',
    Configurations = {
        ['standard'] = {
            teams = {
                [1] = {
                    name = 'FFA',
                    armies = { [1] = 'Player', [2] = 'Aeon' },
                },
            },
        },
    },
}
"#;
        let meta = parse_scenario(lua, "X1CA_001").expect("parse scenario");
        assert_eq!(meta.name, "X1CA_001");
        assert_eq!(meta.description, "Raw \\n multiline\ndescription");
        assert_eq!(meta.scenario_type, "campaign");
        assert_eq!(meta.size, (256, 512));
        assert_eq!(meta.armies, vec!["Player", "Aeon"]);
    }

    #[test]
    fn parse_save_markers_reads_only_masterchain() {
        let lua = r#"
--[[                                                                           ]]--
--[[  Dummy comment with fake MasterChain = { ['_MASTERCHAIN_'] = {          ]]--
--[[      Markers = { ['ARMY_99'] = { ['type'] = STRING('Blank Marker') } }   ]]--
--[[  } }                                                                      ]]--
Scenario = {
    next_area_id = '1',
    note = "MasterChain = { fake string }",
    Areas = {
        ['AREA_1'] = {
            ['rectangle'] = RECTANGLE( 0, 0, 512, 512 ),
        },
    },
    EarlyDummyChains = {
        ['FakeChain'] = {
            Markers = {
                ['ARMY_98'] = {
                    ['type'] = STRING( 'Blank Marker' ),
                    ['position'] = VECTOR3( 999.0, 0.0, 999.0 ),
                },
            },
        },
    },
    MasterChain = {
        ['_MASTERCHAIN_'] = {
            Markers = {
                ['ARMY_2'] = {
                    ['type'] = STRING( 'Blank Marker' ),
                    ['position'] = VECTOR3( 400.0, 20.0, 410.0 ),
                },
                ['ARMY_1'] = {
                    ['type'] = 'Blank Marker',
                    ['position'] = { 50.0, 20.0, 60.0 },
                },
                ['Player'] = {
                    ['type'] = STRING( 'Blank Marker' ),
                    ['position'] = VECTOR3( -12.5, 18.25, 34.5 ),
                },
                ['Mass 02'] = {
                    ['size'] = FLOAT( 1.000000 ),
                    ['resource'] = BOOLEAN( true ),
                    ['type'] = STRING( 'Mass' ),
                    ['orientation'] = VECTOR3( 0, -0, 0 ),
                    ['position'] = VECTOR3( 220.0, 15.0, 230.0 ),
                },
                ['Mass 01'] = {
                    ['type'] = 'Mass',
                    ['position'] = VECTOR3( 110.0, 15.0, 120.0 ),
                },
                ['Hydro 01'] = {
                    ['type'] = STRING( 'Hydrocarbon' ),
                    ['position'] = VECTOR3( 300.0, 16.0, 310.0 ),
                },
                ['Land Path Node 01'] = {
                    ['type'] = STRING( 'Land Path Node' ),
                    ['position'] = VECTOR3( 10.0, 10.0, 10.0 ),
                },
            },
        },
    },
    Chains = {
        ['Attack_Chain_1'] = {
            Markers = {
                ['ARMY_77'] = {
                    ['type'] = STRING( 'Blank Marker' ),
                    ['position'] = VECTOR3( 777.0, 0.0, 777.0 ),
                },
                ['Mass 99'] = {
                    ['type'] = STRING( 'Mass' ),
                    ['position'] = VECTOR3( 888.0, 0.0, 888.0 ),
                },
                ['Hydro 99'] = {
                    ['type'] = STRING( 'Hydrocarbon' ),
                    ['position'] = VECTOR3( 999.0, 0.0, 999.0 ),
                },
            },
        },
    },
    Armies = {
        ['ARMY_1'] = {
            Units = GROUP {
                ['UNIT_1'] = {
                    Position = { 1.0, 2.0, 3.0 },
                },
            },
        },
    },
}
"#;
        let armies = vec!["Player".to_string()];
        let markers = parse_save_markers(lua, &armies).expect("parse save markers");

        assert_eq!(markers.total_markers, 7);

        let start_names: Vec<&str> = markers.start_spots.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(start_names, vec!["ARMY_1", "ARMY_2", "Player"]);
        assert_eq!(markers.start_spots[0].position, [50.0, 20.0, 60.0]);
        assert_eq!(markers.start_spots[2].position, [-12.5, 18.25, 34.5]);

        let mass_names: Vec<&str> = markers.mass_points.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(mass_names, vec!["Mass 01", "Mass 02"]);
        assert_eq!(markers.mass_points[0].marker_type, "Mass");
        assert_eq!(markers.mass_points[0].position, [110.0, 15.0, 120.0]);

        let hydro_names: Vec<&str> = markers.hydro_points.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(hydro_names, vec!["Hydro 01"]);
        assert_eq!(markers.hydro_points[0].marker_type, "Hydrocarbon");
        assert_eq!(markers.hydro_points[0].position, [300.0, 16.0, 310.0]);
    }

    #[test]
    fn resolve_vfs_path_case_insensitive_in_temp_dir() {
        let guard = TempDirGuard::new("vfs");
        let map_dir = guard.path().join("Maps").join("SCMP_001");
        std::fs::create_dir_all(&map_dir).expect("create Maps/SCMP_001");
        let scmap = map_dir.join("ScMp_001.scmap");
        std::fs::write(&scmap, b"SCMAP").expect("write scmap");

        let resolved1 = resolve_vfs_path(guard.path(), "/maps/scmp_001/SCMP_001.SCMAP")
            .expect("resolve unix-style vfs path");
        assert_eq!(resolved1, scmap);

        let resolved2 = resolve_vfs_path(guard.path(), "\\MAPS\\\\ScMp_001\\scmp_001.scmap")
            .expect("resolve windows-style vfs path");
        assert_eq!(resolved2, scmap);

        assert!(resolve_vfs_path(guard.path(), "/maps/scmp_001/missing.scmap").is_none());
        assert!(resolve_vfs_path(guard.path(), "").is_none());
        assert!(resolve_vfs_path(guard.path(), "/").is_none());
    }

    #[test]
    fn list_maps_find_map_cp1252_and_fallbacks() {
        let guard = TempDirGuard::new("list");
        let maps_root = guard.path().join("maps");

        // Map 1: X1CA_001 with CP1252 bytes and stale VFS paths requiring suffix fallback
        let x1ca_dir = maps_root.join("X1CA_001");
        std::fs::create_dir_all(&x1ca_dir).expect("create X1CA_001");
        let mut cp1252_bytes = Vec::new();
        cp1252_bytes.extend_from_slice(b"version = 3\nScenarioInfo = {\n    name = 'Operation");
        cp1252_bytes.push(0x92); // CP1252 right single quote
        cp1252_bytes.extend_from_slice(b"s Dawn',\n    description = ");
        cp1252_bytes.push(b'"');
        cp1252_bytes.push(0x93);
        cp1252_bytes.extend_from_slice(b"CP1252");
        cp1252_bytes.push(0x94);
        cp1252_bytes.extend_from_slice(
            b"\",\n    type = 'campaign',\n    size = {512, 512},\n    map = '/maps/OLD_LOC/old.scmap',\n    save = '/maps/OLD_LOC/old_save.lua',\n}\n",
        );
        std::fs::write(x1ca_dir.join("x1ca_001_SCENARIO.LUA"), &cp1252_bytes)
            .expect("write cp1252 scenario");
        std::fs::write(x1ca_dir.join("X1CA_001.SCMAP"), b"scmap").expect("write fallback scmap");
        std::fs::write(
            x1ca_dir.join("X1CA_001_SAVE.LUA"),
            b"Scenario = { MasterChain = { ['_MASTERCHAIN_'] = { Markers = { ['ARMY_1'] = { ['type'] = 'Blank Marker', ['position'] = VECTOR3(1, 2, 3) } } } } }",
        )
        .expect("write fallback save");

        // Map 2: SCMP_002 with valid VFS paths
        let scmp_dir = maps_root.join("SCMP_002");
        std::fs::create_dir_all(&scmp_dir).expect("create SCMP_002");
        std::fs::write(
            scmp_dir.join("SCMP_002_scenario.lua"),
            b"ScenarioInfo = { name = 'Canis', type = 'skirmish', size = {256, 256}, map = '/maps/scmp_002/SCMP_002.scmap', save = '/maps/scmp_002/SCMP_002_save.lua' }",
        )
        .expect("write scmp_002 scenario");
        std::fs::write(scmp_dir.join("scmp_002.scmap"), b"scmap2").expect("write scmp_002 scmap");
        std::fs::write(scmp_dir.join("scmp_002_save.lua"), b"Scenario = {}")
            .expect("write scmp_002 save");

        // Empty non-map subdir should be skipped
        std::fs::create_dir_all(maps_root.join("unused_dir")).expect("create unused_dir");

        let maps = list_maps(guard.path()).expect("list_maps");
        assert_eq!(maps.len(), 2);
        assert_eq!(maps[0].id, "SCMP_002");
        assert_eq!(maps[1].id, "X1CA_001");
        assert!(maps[0].scmap_path.is_some());
        assert!(maps[0].save_path.is_some());
        assert_eq!(
            maps[1].scmap_path.as_ref().and_then(|p| p.file_name()).and_then(|n| n.to_str()),
            Some("X1CA_001.SCMAP")
        );
        assert_eq!(
            maps[1].save_path.as_ref().and_then(|p| p.file_name()).and_then(|n| n.to_str()),
            Some("X1CA_001_SAVE.LUA")
        );

        let found = find_map(guard.path(), "x1ca_001").expect("find_map case-insensitive");
        assert_eq!(found.id, "X1CA_001");
        let parsed_save = parse_save_markers_file(
            found.save_path.as_deref().expect("save path"),
            &[],
        )
        .expect("parse save file");
        assert_eq!(parsed_save.start_spots.len(), 1);
        assert_eq!(parsed_save.start_spots[0].name, "ARMY_1");
        assert_eq!(parsed_save.start_spots[0].position, [1.0, 2.0, 3.0]);
    }
}
