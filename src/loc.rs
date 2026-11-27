//! Phase 37 localization & tooltip text engine.
//!
//! Loads Forged Alliance `loc_<LANG>.scd` string tables (`loc/<LANG>/strings_db.lua`,
//! `usdb.lua`, loose `loc/<LANG>/*.lua`, and user overrides in `$XDG_DATA_HOME/farust/loc/<LANG>/`)
//! together with unit/enhancement descriptions (`lua/ui/help/unitdescription.lua`) and order
//! tooltips (`lua/ui/help/tooltips.lua`) from `lua.scd`.
//!
//! Lookup priority for a `<LOC key>Fallback` string in active language `L`:
//! 1. Loaded FA string table for `L` (e.g. `loc_DE.scd` when installed).
//! 2. Built-in FARust UI / core dictionary for `L` (clean-room translations so German works
//!    even on English-only Steam installs without `loc_DE.scd`).
//! 3. Loaded US English FA string table (`loc_US.scd`), optionally matched against the
//!    clean-room phrase fallback when `L == "de"` and no `loc_DE.scd` is present.
//! 4. Inline `Fallback` text from the `<LOC key>Fallback` tag.
//! 5. The key itself.
//!
//! Rendering/UI only: the deterministic simulation never reads localization state.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// Where a language's string table came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LangSource {
    Scd(PathBuf),
    Dir(PathBuf),
    Builtin,
}

impl LangSource {
    pub fn describe(&self) -> String {
        match self {
            LangSource::Scd(p) => format!("scd:{}", p.file_name().unwrap_or_default().to_string_lossy()),
            LangSource::Dir(p) => format!("dir:{}", p.display()),
            LangSource::Builtin => "built-in".to_string(),
        }
    }
}

/// Summary of one selectable language.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LangInfo {
    pub code: String,
    pub name: String,
    pub source: LangSource,
    pub entry_count: usize,
}

/// One tooltip entry parsed from `lua/ui/help/tooltips.lua`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TooltipEntry {
    pub title: String,
    pub description: String,
    pub key_id: Option<String>,
}

/// Loaded localization database (all discovered languages + FA help tables).
#[derive(Debug, Clone)]
pub struct LocDb {
    pub active_lang: String,
    pub languages: Vec<LangInfo>,
    /// `lang_code` -> (`lowercase_key` -> `value`).
    pub tables: HashMap<String, HashMap<String, String>>,
    /// `lowercase_id` (e.g. `uel0001`, `uel0001-tm`) -> raw `<LOC ...>...` from `unitdescription.lua`.
    pub unit_descriptions: HashMap<String, String>,
    /// `lowercase_id` (e.g. `move`, `toggle_shield`) -> entry from `tooltips.lua`.
    pub tooltips: HashMap<String, TooltipEntry>,
    /// `(unit_id_lower, enh_key_lower)` -> `(icon_lower, raw_name_loc)` from blueprint `Enhancements`.
    pub enh_meta: HashMap<(String, String), (String, String)>,
}

impl Default for LocDb {
    fn default() -> Self {
        Self::empty("us")
    }
}

/// Normalizes a language identifier (`en`, `en_US`, `DE`, `de-DE`, `german`, ...) to a
/// canonical lowercase FA language code (`us`, `de`, `fr`, ...).
pub fn normalize_lang(raw: &str) -> String {
    let t = raw.trim().to_ascii_lowercase();
    match t.as_str() {
        "" | "en" | "en_us" | "en-us" | "en_gb" | "en-gb" | "english" | "us" => "us".to_string(),
        "de" | "de_de" | "de-de" | "de_at" | "de_ch" | "german" | "deutsch" => "de".to_string(),
        "fr" | "fr_fr" | "fr-fr" | "french" | "français" | "francais" => "fr".to_string(),
        "es" | "es_es" | "es-es" | "spanish" | "español" | "espanol" => "es".to_string(),
        "it" | "it_it" | "it-it" | "italian" | "italiano" => "it".to_string(),
        "ru" | "ru_ru" | "ru-ru" | "russian" => "ru".to_string(),
        "pl" | "pl_pl" | "pl-pl" | "polish" => "pl".to_string(),
        "cz" | "cs" | "cs_cz" | "czech" => "cz".to_string(),
        "cn" | "zh" | "zh_cn" | "zh-cn" => "cn".to_string(),
        "tw" | "zh_tw" | "zh-tw" => "tw".to_string(),
        "kr" | "ko" | "ko_kr" => "kr".to_string(),
        "jp" | "ja" | "ja_jp" => "jp".to_string(),
        other => other.split(['_', '-']).next().unwrap_or(other).to_string(),
    }
}

/// Human-readable display name for a language code (rendered in Options).
pub fn language_display_name(code: &str) -> String {
    match normalize_lang(code).as_str() {
        "us" => "English (US)".to_string(),
        "de" => "Deutsch (DE)".to_string(),
        "fr" => "Français (FR)".to_string(),
        "es" => "Español (ES)".to_string(),
        "it" => "Italiano (IT)".to_string(),
        "ru" => "Русский (RU)".to_string(),
        "pl" => "Polski (PL)".to_string(),
        "cz" => "Čeština (CZ)".to_string(),
        "cn" => "中文 (CN)".to_string(),
        "tw" => "繁體中文 (TW)".to_string(),
        "kr" => "한국어 (KR)".to_string(),
        "jp" => "日本語 (JP)".to_string(),
        other => other.to_ascii_uppercase(),
    }
}

// ---------------------------------------------------------------------------
// Text decoding (never-panic, UTF-8 / UTF-16 BOM / Windows-1252 fallback)
// ---------------------------------------------------------------------------

/// Decodes raw bytes from FA Lua/loc files into UTF-8 text without panicking.
/// Handles UTF-8 (with or without BOM), UTF-16 LE/BE with BOM, and Windows-1252 / Latin-1.
pub fn decode_text_bytes(bytes: &[u8]) -> String {
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        let rest = &bytes[3..];
        return match std::str::from_utf8(rest) {
            Ok(s) => s.to_string(),
            Err(_) => decode_cp1252(rest),
        };
    }
    if bytes.starts_with(&[0xFF, 0xFE]) && bytes.len() >= 2 {
        let (chunks, _) = bytes[2..].as_chunks::<2>();
        let units = chunks.iter().map(|&c| u16::from_le_bytes(c));
        return char::decode_utf16(units).map(|r| r.unwrap_or('\u{FFFD}')).collect();
    }
    if bytes.starts_with(&[0xFE, 0xFF]) && bytes.len() >= 2 {
        let (chunks, _) = bytes[2..].as_chunks::<2>();
        let units = chunks.iter().map(|&c| u16::from_be_bytes(c));
        return char::decode_utf16(units).map(|r| r.unwrap_or('\u{FFFD}')).collect();
    }
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => decode_cp1252(bytes),
    }
}

fn decode_cp1252(bytes: &[u8]) -> String {
    const CP1252_HI: [char; 32] = [
        '\u{20AC}', '\u{FFFD}', '\u{201A}', '\u{0192}', '\u{201E}', '\u{2026}', '\u{2020}', '\u{2021}',
        '\u{02C6}', '\u{2030}', '\u{0160}', '\u{2039}', '\u{0152}', '\u{FFFD}', '\u{017D}', '\u{FFFD}',
        '\u{FFFD}', '\u{2018}', '\u{2019}', '\u{201C}', '\u{201D}', '\u{2022}', '\u{2013}', '\u{2014}',
        '\u{02DC}', '\u{2122}', '\u{0161}', '\u{203A}', '\u{0153}', '\u{FFFD}', '\u{017E}', '\u{0178}',
    ];
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        if !(0x80..0xA0).contains(&b) {
            out.push(char::from(b));
        } else {
            out.push(CP1252_HI[(b - 0x80) as usize]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Never-panic parsers
// ---------------------------------------------------------------------------

/// Splits a Forged Alliance `<LOC key>Fallback` string into `(Some(key), fallback)`.
/// Never panics on malformed input.
pub fn split_loc_tag(raw: &str) -> (Option<&str>, &str) {
    let trimmed = raw.trim_start();
    if trimmed.len() >= 5
        && trimmed.as_bytes()[..4].eq_ignore_ascii_case(b"<loc")
        && (trimmed.as_bytes()[4].is_ascii_whitespace() || trimmed.as_bytes()[4] == b'>')
        && let Some(gt) = trimmed.find('>')
    {
        let key = trimmed[4..gt].trim();
        let fallback = &trimmed[gt + 1..];
        if !key.is_empty() {
            return (Some(key), fallback);
        }
        return (None, fallback);
    }
    (None, raw)
}

/// Scans a Lua-quoted string starting at `bytes[pos]` (`"` or `'`), returning
/// `(unescaped_string, next_pos)`.
fn parse_lua_quoted(s: &str, start: usize) -> Option<(String, usize)> {
    let bytes = s.as_bytes();
    let quote = *bytes.get(start)?;
    if quote != b'"' && quote != b'\'' {
        return None;
    }
    let mut i = start + 1;
    let mut out = String::new();
    while i < bytes.len() {
        let b = bytes[i];
        if b == quote {
            return Some((out, i + 1));
        }
        if b == b'\\' {
            i += 1;
            let esc = *bytes.get(i)?;
            if esc < 0x80 {
                match esc {
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'"' => out.push('"'),
                    b'\'' => out.push('\''),
                    b'\\' => out.push('\\'),
                    other => out.push(char::from(other)),
                }
                i += 1;
            } else {
                let ch = s[i..].chars().next()?;
                out.push(ch);
                i += ch.len_utf8();
            }
        } else if b < 0x80 {
            out.push(char::from(b));
            i += 1;
        } else {
            let ch = s[i..].chars().next()?;
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    None
}

/// Scans a Lua `[[...]]` or `[=[...]=]` long string starting at `s.as_bytes()[start]`.
fn parse_lua_long_string(s: &str, start: usize) -> Option<(String, usize)> {
    let bytes = s.as_bytes();
    if bytes.get(start) != Some(&b'[') {
        return None;
    }
    let mut i = start + 1;
    let mut eqs = 0usize;
    while bytes.get(i) == Some(&b'=') {
        eqs += 1;
        i += 1;
    }
    if bytes.get(i) != Some(&b'[') {
        return None;
    }
    i += 1;
    if bytes.get(i) == Some(&b'\r') && bytes.get(i + 1) == Some(&b'\n') {
        i += 2;
    } else if bytes.get(i) == Some(&b'\n') {
        i += 1;
    }
    let content_start = i;
    while i < bytes.len() {
        if bytes[i] == b']' {
            let mut j = i + 1;
            let mut k = 0usize;
            while k < eqs && bytes.get(j) == Some(&b'=') {
                k += 1;
                j += 1;
            }
            if k == eqs && bytes.get(j) == Some(&b']') {
                return Some((s[content_start..i].to_string(), j + 1));
            }
        }
        i += 1;
    }
    None
}

fn skip_ws_and_comments(s: &str, mut pos: usize) -> usize {
    let bytes = s.as_bytes();
    while pos < bytes.len() {
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if bytes.get(pos) == Some(&b'#') {
            while pos < bytes.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
        } else if bytes[pos..].starts_with(b"--") {
            pos += 2;
            if let Some((_, next)) = parse_lua_long_string(s, pos) {
                pos = next;
            } else {
                while pos < bytes.len() && bytes[pos] != b'\n' {
                    pos += 1;
                }
            }
        } else {
            break;
        }
    }
    pos
}

fn is_key_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.'
}

/// Parses one `key = "value"` or `['key'] = "value"` assignment at `pos`.
fn parse_string_entry(s: &str, pos: usize) -> Option<(String, String, usize)> {
    let bytes = s.as_bytes();
    let mut i = pos;
    let key = if bytes.get(i) == Some(&b'[') {
        i = skip_ws_and_comments(s, i + 1);
        let (k, next) = parse_lua_quoted(s, i)?;
        i = skip_ws_and_comments(s, next);
        if bytes.get(i) != Some(&b']') {
            return None;
        }
        i += 1;
        k
    } else {
        let start = i;
        while i < bytes.len() && is_key_char(bytes[i]) {
            i += 1;
        }
        if i == start {
            return None;
        }
        s[start..i].to_string()
    };
    i = skip_ws_and_comments(s, i);
    if bytes.get(i) != Some(&b'=') {
        return None;
    }
    i = skip_ws_and_comments(s, i + 1);
    let (val, next) = match bytes.get(i) {
        Some(b'"' | b'\'') => parse_lua_quoted(s, i)?,
        Some(b'[') => parse_lua_long_string(s, i)?,
        _ => return None,
    };
    i = skip_ws_and_comments(s, next);
    if matches!(bytes.get(i), Some(b',' | b';')) {
        i += 1;
    }
    Some((key, val, i))
}

/// Strictly parses an FA `strings_db.lua` / `usdb.lua` byte slice.
/// Returns `Err` on NUL-corrupted binary data, syntax errors, or empty files (exercised by `fuzz.rs`).
pub fn parse_strings_db(bytes: &[u8]) -> Result<HashMap<String, String>, String> {
    if bytes.is_empty() {
        return Err("empty loc file".to_string());
    }
    // Reject obvious binary data unless it starts with a UTF-16 BOM.
    let utf16 = bytes.starts_with(&[0xFF, 0xFE]) || bytes.starts_with(&[0xFE, 0xFF]);
    if !utf16 && bytes.contains(&0) {
        return Err("NUL byte in loc string table".to_string());
    }
    let text = decode_text_bytes(bytes);
    let mut out = HashMap::new();
    let mut pos = 0usize;
    let b = text.as_bytes();
    while {
        pos = skip_ws_and_comments(&text, pos);
        pos < b.len()
    } {
        // Allow optional outer `return { ... }` or `ident = { ... }` table wrapper.
        if b[pos] == b'}' {
            pos += 1;
            continue;
        }
        if b[pos..].starts_with(b"return") && b.get(pos + 6).is_some_and(|c| c.is_ascii_whitespace() || *c == b'{') {
            pos = skip_ws_and_comments(&text, pos + 6);
            if b.get(pos) == Some(&b'{') {
                pos += 1;
            }
            continue;
        }
        if let Some((k, v, next)) = parse_string_entry(&text, pos) {
            out.insert(k.to_ascii_lowercase(), v);
            pos = next;
            continue;
        }
        // Optional `TableName = {` wrapper header.
        let mut j = pos;
        while j < b.len() && is_key_char(b[j]) {
            j += 1;
        }
        if j > pos {
            let eq = skip_ws_and_comments(&text, j);
            if b.get(eq) == Some(&b'=') {
                let brace = skip_ws_and_comments(&text, eq + 1);
                if b.get(brace) == Some(&b'{') {
                    pos = brace + 1;
                    continue;
                }
            }
        }
        return Err(format!("invalid loc entry at byte {pos}"));
    }
    if out.is_empty() {
        return Err("no loc entries found".to_string());
    }
    Ok(out)
}

/// Lossy variant for loading mod/user string tables: skips malformed lines instead of failing
/// the whole table.
pub fn parse_strings_db_lossy(bytes: &[u8]) -> HashMap<String, String> {
    if let Ok(strict) = parse_strings_db(bytes) {
        return strict;
    }
    let text = decode_text_bytes(bytes);
    let mut out = HashMap::new();
    for raw in text.lines() {
        let pos = skip_ws_and_comments(raw, 0);
        if pos >= raw.len() {
            continue;
        }
        if let Some((k, v, _)) = parse_string_entry(raw, pos) {
            out.insert(k.to_ascii_lowercase(), v);
        }
    }
    out
}

/// Parses `lua/ui/help/unitdescription.lua` (`Description = { ['uel0001'] = "<LOC ...>...", ... }`).
#[allow(dead_code)]
pub fn parse_unit_descriptions(bytes: &[u8]) -> Result<HashMap<String, String>, String> {
    parse_strings_db(bytes)
}

fn next_char_pos(s: &str, pos: usize) -> usize {
    s.get(pos..)
        .and_then(|rest| rest.chars().next())
        .map_or(s.len(), |c| pos + c.len_utf8())
}

/// Parses `lua/ui/help/tooltips.lua` (`Tooltips = { move = { title = "...", description = "...", keyID = "..." }, ... }`).
pub fn parse_tooltips(bytes: &[u8]) -> Result<HashMap<String, TooltipEntry>, String> {
    if bytes.is_empty() {
        return Err("empty tooltips file".to_string());
    }
    if bytes.contains(&0) {
        return Err("NUL byte in tooltips file".to_string());
    }
    let text = decode_text_bytes(bytes);
    let b = text.as_bytes();
    // Locate `Tooltips = {` (or first `{`).
    let mut pos = 0usize;
    let mut found_open = false;
    while {
        pos = skip_ws_and_comments(&text, pos);
        pos < b.len()
    } {
        if b[pos] == b'{' {
            pos += 1;
            found_open = true;
            break;
        }
        if b[pos] == b'"' || b[pos] == b'\'' {
            let Some((_, next)) = parse_lua_quoted(&text, pos) else {
                return Err(format!("unterminated string at byte {pos}"));
            };
            pos = next;
        } else {
            pos = next_char_pos(&text, pos);
        }
    }
    if !found_open {
        return Err("missing opening '{' in tooltips.lua".to_string());
    }

    let mut out = HashMap::new();
    while {
        pos = skip_ws_and_comments(&text, pos);
        pos < b.len()
    } {
        if b[pos] == b'}' {
            break;
        }
        // Entry key: `ident` or `['ident']`.
        let key = if b[pos] == b'[' {
            let kpos = skip_ws_and_comments(&text, pos + 1);
            let Some((k, next)) = parse_lua_quoted(&text, kpos) else {
                return Err(format!("bad tooltip bracket key at byte {pos}"));
            };
            let rb = skip_ws_and_comments(&text, next);
            if b.get(rb) != Some(&b']') {
                return Err(format!("missing ']' at byte {rb}"));
            }
            pos = rb + 1;
            k
        } else {
            let start = pos;
            while pos < b.len() && is_key_char(b[pos]) {
                pos += 1;
            }
            if pos == start {
                return Err(format!("expected tooltip key at byte {pos}"));
            }
            text[start..pos].to_string()
        };
        pos = skip_ws_and_comments(&text, pos);
        if b.get(pos) != Some(&b'=') {
            return Err(format!("expected '=' after '{key}' at byte {pos}"));
        }
        pos = skip_ws_and_comments(&text, pos + 1);
        if b.get(pos) != Some(&b'{') {
            // Skip non-table value safely (e.g. alias or string).
            if let Some((_, next)) = parse_lua_quoted(&text, pos) {
                pos = skip_ws_and_comments(&text, next);
                if matches!(b.get(pos), Some(b',' | b';')) {
                    pos += 1;
                }
                continue;
            }
            return Err(format!("expected '{{' for tooltip '{key}' at byte {pos}"));
        }
        pos += 1;
        let mut entry = TooltipEntry::default();
        let mut depth = 1usize;
        while {
            pos = skip_ws_and_comments(&text, pos);
            pos < b.len() && depth > 0
        } {
            if b[pos] == b'}' {
                depth -= 1;
                pos += 1;
                break;
            }
            if let Some((fk, fv, next)) = parse_string_entry(&text, pos) {
                match fk.to_ascii_lowercase().as_str() {
                    "title" => entry.title = fv,
                    "description" => entry.description = fv,
                    "keyid" => entry.key_id = Some(fv),
                    _ => {}
                }
                pos = next;
                continue;
            }
            // Skip unknown nested fields/values safely.
            let mut k_end = pos;
            while k_end < b.len() && is_key_char(b[k_end]) {
                k_end += 1;
            }
            if k_end > pos {
                let eq = skip_ws_and_comments(&text, k_end);
                if b.get(eq) == Some(&b'=') {
                    pos = skip_value(&text, eq + 1, 0)?;
                    pos = skip_ws_and_comments(&text, pos);
                    if matches!(b.get(pos), Some(b',' | b';')) {
                        pos += 1;
                    }
                    continue;
                }
            }
            return Err(format!("unexpected token in tooltip '{key}' at byte {pos}"));
        }
        if depth != 0 {
            return Err(format!("unterminated tooltip '{key}'"));
        }
        pos = skip_ws_and_comments(&text, pos);
        if matches!(b.get(pos), Some(b',' | b';')) {
            pos += 1;
        }
        out.insert(key.to_ascii_lowercase(), entry);
    }
    if out.is_empty() {
        return Err("no tooltips found".to_string());
    }
    Ok(out)
}

fn skip_value(s: &str, pos: usize, depth: usize) -> Result<usize, String> {
    if depth > 8 {
        return Err("tooltip nesting too deep".to_string());
    }
    let mut i = skip_ws_and_comments(s, pos);
    let b = s.as_bytes();
    match b.get(i) {
        Some(b'"' | b'\'') => parse_lua_quoted(s, i)
            .map(|(_, n)| n)
            .ok_or_else(|| format!("unterminated string at byte {i}")),
        Some(b'[') => parse_lua_long_string(s, i)
            .map(|(_, n)| n)
            .ok_or_else(|| format!("unterminated long string at byte {i}")),
        Some(b'{') => {
            i += 1;
            let mut level = 1usize;
            while level > 0 {
                i = skip_ws_and_comments(s, i);
                if i >= b.len() {
                    return Err("unterminated nested table".to_string());
                }
                match b[i] {
                    b'{' => {
                        if level + depth > 8 {
                            return Err("tooltip nesting too deep".to_string());
                        }
                        level += 1;
                        i += 1;
                    }
                    b'}' => {
                        level -= 1;
                        i += 1;
                    }
                    b'"' | b'\'' => {
                        let (_, next) = parse_lua_quoted(s, i)
                            .ok_or_else(|| format!("unterminated string at byte {i}"))?;
                        i = next;
                    }
                    _ => i = next_char_pos(s, i),
                }
            }
            Ok(i)
        }
        Some(_) => {
            while i < b.len() && !matches!(b[i], b',' | b';' | b'}' | b'\n') {
                i = next_char_pos(s, i);
            }
            Ok(i)
        }
        None => Err("unexpected EOF in value".to_string()),
    }
}

/// Extracts `(enh_key_lower, icon_lower, raw_name_loc)` entries from a unit blueprint.
fn extract_blueprint_enh_meta(bytes: &[u8]) -> Vec<(String, String, String)> {
    // Quick filter: only ~8 blueprints in units.scd define Enhancements.
    if !bytes.windows(12).any(|w| w == b"Enhancements") {
        return Vec::new();
    }
    let text = decode_text_bytes(bytes);
    let Ok(root) = crate::blueprint::parse_blueprint(&text) else {
        return Vec::new();
    };
    let Some(enhs) = root.get("Enhancements") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (k, v) in enhs.keyed_tables() {
        if k.eq_ignore_ascii_case("Slots") {
            continue;
        }
        let icon = v.get_str("Icon").unwrap_or("").trim().to_ascii_lowercase();
        let name = v.get_str("Name").unwrap_or("").trim().to_string();
        if !icon.is_empty() || !name.is_empty() {
            out.push((k.to_ascii_lowercase(), icon, name));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// LocDb loading & lookup
// ---------------------------------------------------------------------------

impl LocDb {
    pub fn empty(active_lang: &str) -> Self {
        let lang = normalize_lang(active_lang);
        let languages = vec![
            LangInfo {
                code: "us".into(),
                name: language_display_name("us"),
                source: LangSource::Builtin,
                entry_count: builtin_count("us"),
            },
            LangInfo {
                code: "de".into(),
                name: language_display_name("de"),
                source: LangSource::Builtin,
                entry_count: builtin_count("de"),
            },
        ];
        Self {
            active_lang: lang,
            languages,
            tables: HashMap::new(),
            unit_descriptions: HashMap::new(),
            tooltips: HashMap::new(),
            enh_meta: HashMap::new(),
        }
    }

    /// Loads every installed `loc_<LANG>.scd` (plus loose `loc/<LANG>/`, `$XDG_DATA_HOME/farust/loc/<LANG>/`,
    /// `lua.scd` help tables, and blueprint enhancement metadata) from `game_dir`. Never panics.
    pub fn load(game_dir: &Path) -> Self {
        let active = crate::settings::language();
        let mut db = Self {
            active_lang: active,
            languages: Vec::new(),
            tables: HashMap::new(),
            unit_descriptions: HashMap::new(),
            tooltips: HashMap::new(),
            enh_meta: HashMap::new(),
        };
        let mut sources: BTreeMap<String, LangSource> = BTreeMap::new();

        // 1. Scan gamedata/loc_*.scd
        let gamedata = crate::scd::gamedata_dir(game_dir);
        if let Ok(rd) = std::fs::read_dir(&gamedata) {
            let mut scds: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
            scds.sort();
            for path in scds {
                let Some(fname) = path.file_name().and_then(|f| f.to_str()) else {
                    continue;
                };
                let lower = fname.to_ascii_lowercase();
                let Some(code_raw) = lower.strip_prefix("loc_").and_then(|r| r.strip_suffix(".scd")) else {
                    continue;
                };
                if code_raw.is_empty() {
                    continue;
                }
                let code = normalize_lang(code_raw);
                if let Ok(files) = crate::scd::read_scd_matching(&path, |n| {
                    let nl = n.replace('\\', "/").to_ascii_lowercase();
                    nl.ends_with(".lua") && (nl.contains("strings_db") || nl.contains("usdb") || nl.starts_with("loc/"))
                }) {
                    let entry = db.tables.entry(code.clone()).or_default();
                    for (_, bytes) in files {
                        let parsed = parse_strings_db_lossy(&bytes);
                        entry.extend(parsed);
                    }
                    if !entry.is_empty() {
                        sources.insert(code, LangSource::Scd(path));
                    }
                }
            }
        }

        // 2. Loose loc/<LANG>/*.lua in game_dir and $XDG_DATA_HOME/farust/loc/<LANG>/*.lua
        let mut loose_roots = vec![game_dir.join("loc")];
        if let Some(dd) = crate::settings::data_dir() {
            loose_roots.push(dd.join("loc"));
        }
        for root in loose_roots {
            let Ok(rd) = std::fs::read_dir(&root) else {
                continue;
            };
            let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
            dirs.sort();
            for dir in dirs {
                let Some(code_raw) = dir.file_name().and_then(|f| f.to_str()) else {
                    continue;
                };
                let code = normalize_lang(code_raw);
                let Ok(files) = std::fs::read_dir(&dir) else {
                    continue;
                };
                let mut paths: Vec<PathBuf> = files
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("lua")))
                    .collect();
                paths.sort();
                let mut added = 0usize;
                for p in paths {
                    if let Ok(bytes) = std::fs::read(&p) {
                        let parsed = parse_strings_db_lossy(&bytes);
                        added += parsed.len();
                        db.tables.entry(code.clone()).or_default().extend(parsed);
                    }
                }
                if added > 0 {
                    sources.insert(code, LangSource::Dir(dir));
                }
            }
        }

        // 3. Load lua.scd help files (unitdescription.lua & tooltips.lua)
        let lua_scd = gamedata.join("lua.scd");
        if lua_scd.is_file() {
            if let Ok((_, bytes)) = crate::scd::read_scd_file_ci(&lua_scd, "lua/ui/help/unitdescription.lua") {
                db.unit_descriptions = parse_strings_db_lossy(&bytes);
            }
            if let Ok((_, bytes)) = crate::scd::read_scd_file_ci(&lua_scd, "lua/ui/help/tooltips.lua")
                && let Ok(parsed) = parse_tooltips(&bytes)
            {
                db.tooltips = parsed;
            }
        }

        // 4. Load enhancement icon + LOC name tags from units.scd (fast filter: only blueprints mentioning Enhancements)
        let units_scd = gamedata.join("units.scd");
        if units_scd.is_file()
            && let Ok(files) = crate::scd::read_scd_matching(&units_scd, |n| {
                let l = n.to_ascii_lowercase();
                l.ends_with("_unit.bp") && (l.contains("0001_unit.bp") || l.contains("0301_unit.bp"))
            })
        {
            for (path, bytes) in files {
                let uid = crate::blueprint::unit_id_from_path(&path).to_ascii_lowercase();
                for (enh_key, icon, raw_name) in extract_blueprint_enh_meta(&bytes) {
                    db.enh_meta.insert((uid.clone(), enh_key), (icon, raw_name));
                }
            }
        }

        // 5. Build language roster: `us`, `de`, then any additional discovered languages.
        let mut ordered: Vec<String> = vec!["us".into(), "de".into()];
        for k in sources.keys() {
            if !ordered.contains(k) {
                ordered.push(k.clone());
            }
        }
        for code in ordered {
            let source = sources.get(&code).cloned().unwrap_or(LangSource::Builtin);
            let loaded_count = db.tables.get(&code).map_or(0, HashMap::len);
            let entry_count = if loaded_count > 0 { loaded_count } else { builtin_count(&code) };
            db.languages.push(LangInfo {
                name: language_display_name(&code),
                code,
                source,
                entry_count,
            });
        }

        db
    }

    /// True when the active language has a loaded external FA string table (`loc_<LANG>.scd` or loose dir).
    pub fn has_external_table(&self, lang: &str) -> bool {
        let code = normalize_lang(lang);
        self.tables.get(&code).is_some_and(|t| !t.is_empty())
    }

    /// Looks up a localization key in `lang` with fallback to built-in `lang`, then US English.
    pub fn lookup_lang(&self, lang: &str, key: &str) -> Option<String> {
        let code = normalize_lang(lang);
        let k = key.trim().to_ascii_lowercase();
        if k.is_empty() {
            return None;
        }
        // 1. Loaded FA string table for `lang`
        if let Some(v) = self.tables.get(&code).and_then(|t| t.get(&k)) {
            return Some(v.clone());
        }
        // 2. Built-in FARust dictionary for `lang`
        if let Some(v) = builtin_lookup(&code, &k) {
            return Some(v.to_string());
        }
        // 3. When `de` has no external loc_DE.scd, try translating the US FA string via clean-room phrases
        if code == "de"
            && !self.has_external_table("de")
            && let Some(us_val) = self.tables.get("us").and_then(|t| t.get(&k))
            && let Some(de_val) = de_phrase_fallback(us_val)
        {
            return Some(de_val);
        }
        // 4. US English FA string table
        if code != "us"
            && let Some(v) = self.tables.get("us").and_then(|t| t.get(&k))
        {
            return Some(v.clone());
        }
        // 5. Built-in US English dictionary
        if code != "us"
            && let Some(v) = builtin_lookup("us", &k)
        {
            return Some(v.to_string());
        }
        None
    }

    /// Looks up `key` in the active language.
    #[allow(dead_code)]
    pub fn lookup(&self, key: &str) -> Option<String> {
        self.lookup_lang(&self.active_lang, key)
    }

    /// Resolves a `<LOC key>Fallback` string in `lang`.
    pub fn resolve_lang(&self, lang: &str, raw: &str) -> String {
        let code = normalize_lang(lang);
        let (key, fallback) = split_loc_tag(raw);
        if let Some(k) = key {
            if let Some(v) = self.lookup_lang(&code, k) {
                return v;
            }
            if !fallback.is_empty() {
                if code == "de" && !self.has_external_table("de")
                    && let Some(de_val) = de_phrase_fallback(fallback.trim())
                {
                    return de_val;
                }
                return fallback.trim().to_string();
            }
            return k.to_string();
        }
        let fb = fallback.trim();
        if code == "de" && !self.has_external_table("de")
            && let Some(de_val) = de_phrase_fallback(fb)
        {
            return de_val;
        }
        fb.to_string()
    }

    /// Resolves a `<LOC key>Fallback` string in the active language.
    #[allow(dead_code)]
    pub fn resolve(&self, raw: &str) -> String {
        self.resolve_lang(&self.active_lang, raw)
    }

    /// Localized unit class/description name (`<id>_desc`) with fallback to `fallback_desc`.
    pub fn unit_display_name_lang(&self, lang: &str, unit_id: &str, fallback_desc: &str) -> String {
        let key = format!("{}_desc", unit_id.to_ascii_lowercase());
        let code = normalize_lang(lang);
        if let Some(v) = self.lookup_lang(&code, &key)
            && !v.trim().is_empty()
        {
            return v.trim().to_string();
        }
        let fb = fallback_desc.trim();
        if !fb.is_empty() {
            if code == "de" && !self.has_external_table("de")
                && let Some(de_val) = de_phrase_fallback(fb)
            {
                return de_val;
            }
            return fb.to_string();
        }
        unit_id.to_string()
    }

    /// Localized proper unit name (`<id>_name`, e.g. `"MA12 Striker"`), if present.
    pub fn unit_proper_name_lang(&self, lang: &str, unit_id: &str) -> Option<String> {
        let key = format!("{}_name", unit_id.to_ascii_lowercase());
        self.lookup_lang(lang, &key).filter(|s| !s.trim().is_empty())
    }

    /// Localized unit description / rollover help text from `unitdescription.lua`.
    pub fn unit_description_lang(&self, lang: &str, unit_id: &str) -> Option<String> {
        let id = unit_id.to_ascii_lowercase();
        let code = normalize_lang(lang);
        if let Some(raw) = self.unit_descriptions.get(&id) {
            let (key, fallback) = split_loc_tag(raw);
            if let Some(k) = key {
                // Prefer loaded FA table for `code`
                if let Some(v) = self.tables.get(&code).and_then(|t| t.get(&k.to_ascii_lowercase())) {
                    return Some(v.trim().to_string());
                }
                // Built-in translation by loc key or unit id
                if let Some(v) = builtin_lookup(&code, &k.to_ascii_lowercase())
                    .or_else(|| builtin_unit_desc(&code, &id))
                {
                    return Some(v.to_string());
                }
                // Fallback to US FA table or inline fallback
                if let Some(v) = self.tables.get("us").and_then(|t| t.get(&k.to_ascii_lowercase())) {
                    return Some(v.trim().to_string());
                }
                if !fallback.trim().is_empty() {
                    return Some(fallback.trim().to_string());
                }
            } else if !raw.trim().is_empty() {
                return Some(raw.trim().to_string());
            }
        }
        builtin_unit_desc(&code, &id).map(str::to_string)
    }

    /// Localized enhancement display name.
    pub fn enhancement_name_lang(&self, lang: &str, unit_id: &str, enh_key: &str, fallback: &str) -> String {
        let code = normalize_lang(lang);
        let pair = (unit_id.to_ascii_lowercase(), enh_key.to_ascii_lowercase());
        if let Some((_, raw_name)) = self.enh_meta.get(&pair)
            && !raw_name.is_empty()
        {
            return self.resolve_lang(&code, raw_name);
        }
        let fb = fallback.trim();
        if code == "de" && !self.has_external_table("de")
            && let Some(de_val) = de_phrase_fallback(fb)
        {
            return de_val;
        }
        if fb.is_empty() { enh_key.to_string() } else { fb.to_string() }
    }

    /// Localized enhancement description from `unitdescription.lua` (`<uid>-<icon>`).
    pub fn enhancement_description_lang(&self, lang: &str, unit_id: &str, enh_key: &str) -> Option<String> {
        let pair = (unit_id.to_ascii_lowercase(), enh_key.to_ascii_lowercase());
        let (icon, _) = self.enh_meta.get(&pair)?;
        if icon.is_empty() {
            return None;
        }
        let ud_key = format!("{}-{icon}", unit_id.to_ascii_lowercase());
        self.unit_description_lang(lang, &ud_key)
    }

    /// Localized order title + help text from `tooltips.lua` + built-in fallback.
    pub fn order_tooltip_lang(&self, lang: &str, tooltip_id: &str, fallback_title: &str, fallback_help: &str) -> (String, String) {
        let code = normalize_lang(lang);
        let id = tooltip_id.to_ascii_lowercase();
        let mut title = None;
        let mut desc = None;
        if let Some(tt) = self.tooltips.get(&id) {
            if !tt.title.is_empty() {
                title = Some(self.resolve_lang(&code, &tt.title));
            }
            if !tt.description.is_empty() {
                desc = Some(self.resolve_lang(&code, &tt.description));
            }
        }
        let title_str = title
            .or_else(|| builtin_lookup(&code, &format!("order.{id}.title")).map(str::to_string))
            .unwrap_or_else(|| fallback_title.to_string());
        let desc_str = desc
            .filter(|s| !s.is_empty())
            .or_else(|| builtin_lookup(&code, &format!("order.{id}.help")).map(str::to_string))
            .or_else(|| builtin_lookup("us", &format!("order.{id}.help")).map(str::to_string))
            .unwrap_or_else(|| fallback_help.to_string());
        (title_str, desc_str)
    }
}

// ---------------------------------------------------------------------------
// Process-wide active LocDb
// ---------------------------------------------------------------------------

static ACTIVE: RwLock<Option<LocDb>> = RwLock::new(None);

/// Loads FA localization data from `game_dir` and activates it process-wide.
pub fn init(game_dir: &Path) {
    let db = LocDb::load(game_dir);
    if let Ok(mut g) = ACTIVE.write() {
        *g = Some(db);
    }
}

/// Switches the active language code at runtime (used when saving Options or in tests).
pub fn set_language(code: &str) {
    let norm = normalize_lang(code);
    if let Ok(mut g) = ACTIVE.write() {
        match g.as_mut() {
            Some(db) => db.active_lang = norm,
            None => *g = Some(LocDb::empty(&norm)),
        }
    }
}

/// Current effective language code (`FARUST_LANG` env override > active `LocDb` > `settings::language()`).
pub fn active_language() -> String {
    if let Ok(v) = std::env::var("FARUST_LANG")
        && let Ok(code) = crate::settings::validate_language(&v)
    {
        return code;
    }
    if let Ok(g) = ACTIVE.read()
        && let Some(db) = g.as_ref()
    {
        return db.active_lang.clone();
    }
    crate::settings::language()
}

/// Available languages as `(code, display_name)` pairs for the Options selector.
pub fn available_languages() -> Vec<(String, String)> {
    if let Ok(g) = ACTIVE.read()
        && let Some(db) = g.as_ref()
        && !db.languages.is_empty()
    {
        return db.languages.iter().map(|l| (l.code.clone(), l.name.clone())).collect();
    }
    vec![
        ("us".into(), language_display_name("us")),
        ("de".into(), language_display_name("de")),
    ]
}

fn with_db<R>(f: impl FnOnce(&LocDb, &str) -> R) -> R {
    let lang = active_language();
    if let Ok(g) = ACTIVE.read()
        && let Some(db) = g.as_ref()
    {
        return f(db, &lang);
    }
    let fallback = LocDb::empty(&lang);
    f(&fallback, &lang)
}

/// Looks up a localization key in the active language.
#[allow(dead_code)]
pub fn lookup(key: &str) -> Option<String> {
    with_db(|db, lang| db.lookup_lang(lang, key))
}

/// Resolves `<LOC key>Fallback` in the active language.
#[allow(dead_code)]
pub fn resolve(raw: &str) -> String {
    with_db(|db, lang| db.resolve_lang(lang, raw))
}

/// Translates a FARust UI key (`ui.*`, `hud.*`, `tt.*`, ...) in `lang`, falling back to `fallback`.
pub fn tr_lang(lang: &str, key: &str, fallback: &str) -> String {
    let code = normalize_lang(lang);
    if let Ok(g) = ACTIVE.read()
        && let Some(db) = g.as_ref()
        && let Some(v) = db.lookup_lang(&code, key)
    {
        return v;
    }
    if let Some(v) = builtin_lookup(&code, &key.to_ascii_lowercase()) {
        return v.to_string();
    }
    fallback.to_string()
}

/// Translates a FARust UI key in the active language, falling back to `fallback`.
pub fn tr(key: &str, fallback: &str) -> String {
    let lang = active_language();
    tr_lang(&lang, key, fallback)
}

pub fn unit_display_name(unit_id: &str, fallback_desc: &str) -> String {
    with_db(|db, lang| db.unit_display_name_lang(lang, unit_id, fallback_desc))
}

pub fn unit_proper_name(unit_id: &str) -> Option<String> {
    with_db(|db, lang| db.unit_proper_name_lang(lang, unit_id))
}

pub fn unit_description(unit_id: &str) -> Option<String> {
    with_db(|db, lang| db.unit_description_lang(lang, unit_id))
}

pub fn enhancement_name(unit_id: &str, enh_key: &str, fallback: &str) -> String {
    with_db(|db, lang| db.enhancement_name_lang(lang, unit_id, enh_key, fallback))
}

pub fn enhancement_description(unit_id: &str, enh_key: &str) -> Option<String> {
    with_db(|db, lang| db.enhancement_description_lang(lang, unit_id, enh_key))
}

pub fn order_tooltip(tooltip_id: &str, fallback_title: &str, fallback_help: &str) -> (String, String) {
    with_db(|db, lang| db.order_tooltip_lang(lang, tooltip_id, fallback_title, fallback_help))
}

/// Word-wraps `text` into lines of at most `max_chars` characters (preserving explicit `\n`).
pub fn word_wrap(text: &str, max_chars: usize) -> Vec<String> {
    let limit = max_chars.max(8);
    let mut out = Vec::new();
    for para in text.split('\n') {
        let mut line = String::new();
        let mut line_chars = 0usize;
        for word in para.split_whitespace() {
            let wlen = word.chars().count();
            if line_chars == 0 {
                if wlen <= limit {
                    line.push_str(word);
                    line_chars = wlen;
                } else {
                    for ch in word.chars() {
                        if line_chars >= limit {
                            out.push(std::mem::take(&mut line));
                            line_chars = 0;
                        }
                        line.push(ch);
                        line_chars += 1;
                    }
                }
            } else if line_chars + 1 + wlen <= limit {
                line.push(' ');
                line.push_str(word);
                line_chars += 1 + wlen;
            } else {
                out.push(std::mem::take(&mut line));
                line.push_str(word);
                line_chars = wlen;
            }
        }
        if !line.is_empty() {
            out.push(line);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Clean-room built-in FARust UI & German fallback dictionaries
// ---------------------------------------------------------------------------

fn builtin_count(lang: &str) -> usize {
    match lang {
        "us" => BUILTIN_US.len(),
        "de" => BUILTIN_DE.len() + DE_PHRASES.len() + DE_UNIT_DESCS.len(),
        _ => 0,
    }
}

fn builtin_lookup(lang: &str, key_lower: &str) -> Option<&'static str> {
    let table: &[(&str, &str)] = match lang {
        "us" => BUILTIN_US,
        "de" => BUILTIN_DE,
        _ => return None,
    };
    table.iter().find(|(k, _)| *k == key_lower).map(|(_, v)| *v)
}

fn builtin_unit_desc(lang: &str, unit_id_lower: &str) -> Option<&'static str> {
    if lang != "de" {
        return None;
    }
    DE_UNIT_DESCS.iter().find(|(k, _)| *k == unit_id_lower).map(|(_, v)| *v)
}

/// Clean-room German fallback for common FA unit/enhancement class names when `loc_DE.scd` is absent.
fn de_phrase_fallback(en: &str) -> Option<String> {
    let trimmed = en.trim();
    if let Some((_, de)) = DE_PHRASES.iter().find(|(k, _)| k.eq_ignore_ascii_case(trimmed)) {
        return Some((*de).to_string());
    }
    None
}

const BUILTIN_US: &[(&str, &str)] = &[
    // Main menu & options
    ("ui.skirmish", "SKIRMISH"),
    ("ui.multiplayer", "MULTIPLAYER"),
    ("ui.replays", "REPLAYS"),
    ("ui.options", "OPTIONS"),
    ("ui.quit_desktop", "QUIT TO DESKTOP"),
    ("ui.profile", "COMMANDER PROFILE"),
    ("ui.graphics_audio", "GRAPHICS & AUDIO"),
    ("ui.gameplay_controls", "GAMEPLAY & CONTROLS"),
    ("ui.resolution", "Resolution"),
    ("ui.fullscreen", "Fullscreen"),
    ("ui.vsync", "VSync"),
    ("ui.shadows", "Shadows"),
    ("ui.bloom", "Bloom"),
    ("ui.aa", "Anti-Aliasing"),
    ("ui.gamma", "Gamma"),
    ("ui.ui_theme", "UI Theme"),
    ("ui.language", "Language"),
    ("ui.master_volume", "Master Volume"),
    ("ui.music_volume", "Music Volume"),
    ("ui.sfx_volume", "SFX Volume"),
    ("ui.player_name", "Player Name"),
    ("ui.default_faction", "Default Faction"),
    ("ui.edge_scroll", "Edge Scroll"),
    ("ui.default_host", "Default Host"),
    ("ui.card_hotkeys", "Card Grid Keys"),
    ("ui.key_bindings", "Key Bindings"),
    ("ui.back", "BACK"),
    ("ui.defaults", "DEFAULTS"),
    ("ui.save_apply", "SAVE & APPLY"),
    ("ui.on", "ON"),
    ("ui.off", "OFF"),
    // Tooltips & HUD
    ("tt.mass", "Mass"),
    ("tt.energy", "Energy"),
    ("tt.time", "Time"),
    ("tt.hp", "HP"),
    ("tt.dps", "DPS"),
    ("tt.range", "Range"),
    ("tt.shield", "Shield"),
    ("tt.radar", "Radar"),
    ("tt.buildrate", "Buildrate"),
    ("tt.cargo", "Cargo"),
    ("tt.missile_range", "Missile Range"),
    ("tt.key", "Key"),
    ("tt.click_map", "Click on the map to target (Shift to queue)"),
    ("tt.hint_prod", "LMB: Queue  Shift: x5  RMB: Remove"),
    ("tt.hint_build", "LMB: Place  Drag: Line  Shift: Queue"),
    ("tt.hint_queue", "Click: Cancel 1  Shift: Cancel 5"),
    ("tt.upgrade_to", "Upgrade selected to"),
    ("tt.installed", "Installed"),
    ("tt.installing", "Installing"),
    ("tt.requires", "Requires"),
    ("tt.slot_occupied", "Slot occupied"),
    ("tt.click_install", "Click to install"),
    ("tt.entries", "entries"),
    ("tt.next_tab", "Next tab"),
    ("tt.page_hint", "Click to cycle command card page"),
    ("tt.filter_all", "All Categories"),
    ("tt.filter_all_desc", "Show every buildable unit and structure in this tech level."),
    ("tt.slot_lch", "Left Arm"),
    ("tt.slot_rch", "Right Arm"),
    ("tt.slot_back", "Back"),
    ("tt.slot_other", "Other"),
    // Category tooltips
    ("group.factories.title", "Factories"),
    ("group.factories.desc", "Filter structures that produce land, air, or naval units."),
    ("group.economy.title", "Economy"),
    ("group.economy.desc", "Filter mass extractors, power generators, hydrocarbons, and storage."),
    ("group.defense.title", "Defense"),
    ("group.defense.desc", "Filter point defense, anti-air, artillery, and walls."),
    ("group.intel.title", "Intelligence"),
    ("group.intel.desc", "Filter radar, omni sensors, and stealth field generators."),
    ("group.shields.title", "Shields"),
    ("group.shields.desc", "Filter protective shield generators."),
    ("group.missiles.title", "Missiles"),
    ("group.missiles.desc", "Filter tactical/strategic missile launchers and defenses."),
    ("group.experimental.title", "Experimental"),
    ("group.experimental.desc", "Filter Tech 4 experimental super-units."),
    ("group.engineers.title", "Engineers"),
    ("group.engineers.desc", "Filter construction, repair, and reclaim units."),
    ("group.land.title", "Land Units"),
    ("group.land.desc", "Filter mobile ground and amphibious combat units."),
    ("group.air.title", "Air Units"),
    ("group.air.desc", "Filter aircraft, bombers, fighters, gunships, and transports."),
    ("group.naval.title", "Naval Units"),
    ("group.naval.desc", "Filter surface warships and submarines."),
    // Order help text
    ("order.move.title", "Move"),
    ("order.move.help", "Order selected units to move to a target location."),
    ("order.attack.title", "Attack"),
    ("order.attack.help", "Order units to attack a target or attack-move along a path."),
    ("order.patrol.title", "Patrol"),
    ("order.patrol.help", "Patrol between waypoints, engaging enemies and repairing/reclaiming."),
    ("order.stop.title", "Stop"),
    ("order.stop.help", "Cancel all current and queued orders for selected units."),
    ("order.assist.title", "Assist"),
    ("order.assist.help", "Assist a friendly factory or engineer with construction or production."),
    ("order.guard.title", "Guard"),
    ("order.guard.help", "Escort and protect a friendly unit or assist a friendly structure."),
    ("order.repair.title", "Repair"),
    ("order.repair.help", "Repair a damaged friendly unit or structure."),
    ("order.reclaim.title", "Reclaim"),
    ("order.reclaim.help", "Reclaim wreckage to recover mass."),
    ("order.load.title", "Load Transport"),
    ("order.load.help", "Board selected land units onto a friendly air transport."),
    ("order.transport.title", "Unload Transport"),
    ("order.transport.help", "Drop carried units at the target drop zone."),
    ("order.overcharge.title", "Overcharge"),
    ("order.overcharge.help", "Fire a high-damage ACU Overcharge shot using stored energy."),
    ("order.fire_tactical.title", "Launch Missile"),
    ("order.fire_tactical.help", "Launch a loaded tactical or strategic missile at the target area."),
    ("order.toggle_shield.title", "Shield Toggle"),
    ("order.toggle_shield.help", "Turn the selected units' shields on or off to manage energy upkeep."),
    ("order.toggle_repeat.title", "Repeat Queue"),
    ("order.toggle_repeat.help", "Continuously repeat the selected factories' build queue."),
    ("order.pause.title", "Pause Simulation"),
    ("order.pause.help", "Pause or resume match simulation."),
    ("order.fog.title", "Fog of War"),
    ("order.fog.help", "Toggle Fog of War and radar intel visibility."),
    ("order.ai.title", "AI Control"),
    ("order.ai.help", "Hand control of your army to the AI or take manual control back."),
    ("order.idle_engineer.title", "Idle Engineer"),
    ("order.idle_engineer.help", "Select and cycle through idle engineers."),
    ("order.attack_ground.title", "Attack Ground"),
    ("order.attack_ground.help", "Fire at a map position until given another order."),
    ("order.fire_state.title", "Fire State"),
    ("order.fire_state.help", "Cycle fire at will, return fire and hold fire."),
    ("order.pause_construction.title", "Pause Construction"),
    ("order.pause_construction.help", "Pause or resume construction and production."),
    ("order.ferry.title", "Ferry Route"),
    ("order.ferry.help", "Click a pick-up point, then a drop-off point: transports shuttle idle units."),
];

const BUILTIN_DE: &[(&str, &str)] = &[
    // Main menu & options
    ("ui.skirmish", "GEFECHT"),
    ("ui.multiplayer", "MEHRSPIELER"),
    ("ui.replays", "WIEDERHOLUNGEN"),
    ("ui.options", "OPTIONEN"),
    ("ui.quit_desktop", "BEENDEN"),
    ("ui.profile", "KOMMANDANTEN-PROFIL"),
    ("ui.graphics_audio", "GRAFIK & AUDIO"),
    ("ui.gameplay_controls", "SPIEL & STEUERUNG"),
    ("ui.resolution", "Auflösung"),
    ("ui.fullscreen", "Vollbild"),
    ("ui.vsync", "VSync"),
    ("ui.shadows", "Schatten"),
    ("ui.bloom", "Bloom"),
    ("ui.aa", "Kantenglättung"),
    ("ui.gamma", "Gamma"),
    ("ui.ui_theme", "UI-Design"),
    ("ui.language", "Sprache"),
    ("ui.master_volume", "Gesamtlautstärke"),
    ("ui.music_volume", "Musiklautstärke"),
    ("ui.sfx_volume", "Effektlautstärke"),
    ("ui.player_name", "Spielername"),
    ("ui.default_faction", "Standard-Fraktion"),
    ("ui.edge_scroll", "Randscrollen"),
    ("ui.default_host", "Standard-Host"),
    ("ui.card_hotkeys", "Raster-Tasten"),
    ("ui.key_bindings", "Tastenbelegung"),
    ("ui.back", "ZURÜCK"),
    ("ui.defaults", "STANDARD"),
    ("ui.save_apply", "SPEICHERN"),
    ("ui.on", "EIN"),
    ("ui.off", "AUS"),
    // Tooltips & HUD
    ("tt.mass", "Masse"),
    ("tt.energy", "Energie"),
    ("tt.time", "Bauzeit"),
    ("tt.hp", "LP"),
    ("tt.dps", "SPS"),
    ("tt.range", "Reichweite"),
    ("tt.shield", "Schild"),
    ("tt.radar", "Radar"),
    ("tt.buildrate", "Baurate"),
    ("tt.cargo", "Fracht"),
    ("tt.missile_range", "Raketenreichweite"),
    ("tt.key", "Taste"),
    ("tt.click_map", "Auf die Karte klicken (Umschalt: Einreihen)"),
    ("tt.hint_prod", "LKB: Einreihen  Umschalt: x5  RKB: Entfernen"),
    ("tt.hint_build", "LKB: Platzieren  Ziehen: Reihe  Umschalt: Einreihen"),
    ("tt.hint_queue", "Klick: 1 abbrechen  Umschalt: 5 abbrechen"),
    ("tt.upgrade_to", "Auswahl aufwerten zu"),
    ("tt.installed", "Installiert"),
    ("tt.installing", "Wird installiert"),
    ("tt.requires", "Benötigt"),
    ("tt.slot_occupied", "Platz belegt"),
    ("tt.click_install", "Klicken zum Installieren"),
    ("tt.entries", "Einträge"),
    ("tt.next_tab", "Nächster Tab"),
    ("tt.page_hint", "Klicken für nächste Seite der Befehlskarte"),
    ("tt.filter_all", "Alle Kategorien"),
    ("tt.filter_all_desc", "Zeigt alle baubaren Einheiten und Gebäude dieser Tech-Stufe."),
    ("tt.slot_lch", "Linker Arm"),
    ("tt.slot_rch", "Rechter Arm"),
    ("tt.slot_back", "Rücken"),
    ("tt.slot_other", "Sonstige"),
    // Category tooltips
    ("group.factories.title", "Fabriken"),
    ("group.factories.desc", "Filtert Gebäude zur Produktion von Land-, Luft- und Marineeinheiten."),
    ("group.economy.title", "Wirtschaft"),
    ("group.economy.desc", "Filtert Masseextraktoren, Energiegeneratoren und Speicher."),
    ("group.defense.title", "Verteidigung"),
    ("group.defense.desc", "Filtert Punktverteidigung, Flugabwehr, Artillerie und Mauern."),
    ("group.intel.title", "Aufklärung"),
    ("group.intel.desc", "Filtert Radar-, Omni-Sensoren und Tarnfeldgeneratoren."),
    ("group.shields.title", "Schilde"),
    ("group.shields.desc", "Filtert schützende Schildgeneratoren."),
    ("group.missiles.title", "Raketen"),
    ("group.missiles.desc", "Filtert taktische/strategische Raketenwerfer und Raketenabwehr."),
    ("group.experimental.title", "Experimentell"),
    ("group.experimental.desc", "Filtert experimentelle Tech-4-Supereinheiten."),
    ("group.engineers.title", "Ingenieure"),
    ("group.engineers.desc", "Filtert Bau-, Reparatur- und Rückgewinnungseinheiten."),
    ("group.land.title", "Landeinheiten"),
    ("group.land.desc", "Filtert mobile Boden- und Amphibieneinheiten."),
    ("group.air.title", "Lufteinheiten"),
    ("group.air.desc", "Filtert Jäger, Bomber, Kampfhubschrauber und Transporter."),
    ("group.naval.title", "Marineeinheiten"),
    ("group.naval.desc", "Filtert Kriegsschiffe und U-Boote."),
    // FA tooltipui order titles (overridden by loc_DE.scd when installed)
    ("tooltipui0000", "Bewegen"),
    ("tooltipui0002", "Angreifen"),
    ("tooltipui0004", "Patrouillieren"),
    ("tooltipui0006", "Stopp"),
    ("tooltipui0008", "Unterstützen"),
    ("tooltipui0016", "Überladen"),
    ("tooltipui0018", "Ausladen"),
    ("tooltipui0020", "Strategische Rakete abfeuern"),
    ("tooltipui0022", "Rakete abfeuern"),
    ("tooltipui0032", "Zurückgewinnen"),
    ("tooltipui0036", "Reparieren"),
    ("tooltipui0038", "Pause"),
    ("tooltipui0040", "Schild umschalten"),
    ("tooltipui0481", "Schaltet die Schilde der ausgewählten Einheiten ein oder aus."),
    ("tooltipui0506", "Pausiert oder setzt die aktuelle Simulation fort."),
    // Order help text
    ("order.move.title", "Bewegen"),
    ("order.move.help", "Bewegt ausgewählte Einheiten zur Zielposition."),
    ("order.attack.title", "Angreifen"),
    ("order.attack.help", "Greift ein Ziel an oder rückt im Angriffsmodus vor."),
    ("order.patrol.title", "Patrouillieren"),
    ("order.patrol.help", "Patrouilliert zwischen Wegpunkten und bekämpft Feinde."),
    ("order.stop.title", "Stopp"),
    ("order.stop.help", "Bricht alle laufenden und eingereihten Befehle ab."),
    ("order.assist.title", "Unterstützen"),
    ("order.assist.help", "Unterstützt eine verbündete Fabrik oder einen Ingenieur."),
    ("order.guard.title", "Bewachen"),
    ("order.guard.help", "Eskortiert und schützt eine verbündete Einheit oder Struktur."),
    ("order.repair.title", "Reparieren"),
    ("order.repair.help", "Repariert beschädigte verbündete Einheiten oder Gebäude."),
    ("order.reclaim.title", "Zurückgewinnen"),
    ("order.reclaim.help", "Gewinnt Masse aus Wracks zurück."),
    ("order.load.title", "Einladen"),
    ("order.load.help", "Lädt ausgewählte Landeinheiten in einen Lufttransporter."),
    ("order.transport.title", "Ausladen"),
    ("order.transport.help", "Setzt transportierte Einheiten in der Zielzone ab."),
    ("order.overcharge.title", "Überladen"),
    ("order.overcharge.help", "Feuert einen hochenergetischen Überladungsschuss der ACU ab."),
    ("order.fire_tactical.title", "Rakete abfeuern"),
    ("order.fire_tactical.help", "Startet eine geladene taktische oder strategische Rakete."),
    ("order.toggle_shield.title", "Schild umschalten"),
    ("order.toggle_shield.help", "Schaltet die Schilde der ausgewählten Einheiten ein oder aus."),
    ("order.toggle_repeat.title", "Wiederholen"),
    ("order.toggle_repeat.help", "Wiederholt die Bauwarteschlange der Fabrik endlos."),
    ("order.pause.title", "Pause"),
    ("order.pause.help", "Pausiert oder setzt die Gefechtssimulation fort."),
    ("order.fog.title", "Kriegsnebel"),
    ("order.fog.help", "Schaltet Kriegsnebel und Radaraufklärung um."),
    ("order.ai.title", "KI-Steuerung"),
    ("order.ai.help", "Übergibt die Armeesteuerung an die KI oder übernimmt selbst."),
    ("order.idle_engineer.title", "Untätiger Ingenieur"),
    ("order.idle_engineer.help", "Wählt den nächsten untätigen Ingenieur aus."),
    ("order.attack_ground.title", "Bodenangriff"),
    ("order.attack_ground.help", "Feuert auf eine Kartenposition, bis ein neuer Befehl kommt."),
    ("order.fire_state.title", "Feuermodus"),
    ("order.fire_state.help", "Wechselt zwischen Feuer frei, Erwidern und Feuer einstellen."),
    ("order.pause_construction.title", "Bau pausieren"),
    ("order.pause_construction.help", "Pausiert oder setzt Bau und Produktion fort."),
    ("order.ferry.title", "Fährroute"),
    ("caption.commander_under_attack", "Kommandeur wird angegriffen"),
    ("caption.base_under_attack", "Basis wird angegriffen"),
    ("caption.structure_lost", "Gebäude verloren"),
    ("caption.shield_down", "Schild ausgefallen"),
    ("caption.enemy_nuke_inbound", "Atomrakete im Anflug"),
    ("caption.strategic_launch_detected", "Strategischer Start erkannt"),
    ("caption.strategic_missile_destroyed", "Strategische Rakete zerstört"),
    ("caption.experimental_detected", "Feindliche Experimentaleinheit gesichtet"),
    ("caption.energy_storage_empty", "Energie erschöpft"),
    ("caption.mass_storage_empty", "Masse erschöpft"),
    ("caption.energy_storage_low", "Wenig Energie"),
    ("caption.mass_storage_low", "Wenig Masse"),
    ("caption.unit_cap_reached", "Einheitenlimit erreicht"),
    ("caption.transport_full", "Transporter voll"),
    ("caption.mass_storage_full", "Massespeicher voll"),
    ("caption.energy_storage_full", "Energiespeicher voll"),
    ("ui.tutorial", "TUTORIAL"),
    ("ui.colorblind", "Farbenblind-Modus"),
    ("ui.mouse_swap", "Maustasten tauschen"),
    ("ui.subtitles", "Untertitel"),
    ("order.ferry.help", "Abholpunkt, dann Zielpunkt anklicken: Transporter pendeln untätige Einheiten."),
    // Compact order-panel button labels (<= 8 chars)
    ("order.move.short", "BEWEGEN"),
    ("order.attack.short", "ANGRIFF"),
    ("order.patrol.short", "PATROUI."),
    ("order.stop.short", "STOPP"),
    ("order.assist.short", "HILFE"),
    ("order.guard.short", "WACHE"),
    ("order.repair.short", "REPAR."),
    ("order.reclaim.short", "RÜCKGEW."),
    ("order.load.short", "LADEN"),
    ("order.transport.short", "AUSLADEN"),
    ("order.overcharge.short", "ÜBERLAD."),
    ("order.fire_tactical.short", "RAKETE"),
    ("order.toggle_shield.short", "SCHILD"),
    ("order.toggle_repeat.short", "WDH."),
    ("order.pause.short", "PAUSE"),
    ("order.fog.short", "NEBEL"),
    ("order.ai.short", "KI"),
    ("order.idle_engineer.short", "ING"),
    ("order.attack_ground.short", "BODEN"),
    ("order.fire_state.short", "FEUER"),
    ("order.pause_construction.short", "BAU-P"),
    ("order.ferry.short", "FÄHRE"),
];

const DE_PHRASES: &[(&str, &str)] = &[
    ("Armored Command Unit", "Gepanzerte Kommandoeinheit"),
    ("Support Armored Command Unit", "Unterstützungs-Kommandoeinheit"),
    ("Land Factory", "Landfabrik"),
    ("Land Factory HQ", "Landfabrik-Hauptquartier"),
    ("Air Factory", "Luftfabrik"),
    ("Air Factory HQ", "Luftfabrik-Hauptquartier"),
    ("Naval Factory", "Marinefabrik"),
    ("Naval Factory HQ", "Marinefabrik-Hauptquartier"),
    ("Mass Extractor", "Masseextraktor"),
    ("Power Generator", "Energiegenerator"),
    ("Hydrocarbon Power Plant", "Kohlenwasserstoff-Kraftwerk"),
    ("Mass Storage", "Massespeicher"),
    ("Energy Storage", "Energiespeicher"),
    ("Mass Fabricator", "Massefabrikator"),
    ("Point Defense", "Punktverteidigung"),
    ("Heavy Point Defense", "Schwere Punktverteidigung"),
    ("Anti-Air Turret", "Flugabwehrgeschütz"),
    ("Anti-Air Flak Artillery", "Flugabwehr-Flakartillerie"),
    ("Anti-Air SAM Launcher", "Flugabwehr-Raketenwerfer"),
    ("Radar System", "Radarsystem"),
    ("Omni Sensor Array", "Omni-Sensoranlage"),
    ("Sonar System", "Sonarsystem"),
    ("Sonar Platform", "Sonarplattform"),
    ("Stealth Field Generator", "Tarnfeldgenerator"),
    ("Shield Generator", "Schildgenerator"),
    ("Heavy Shield Generator", "Schwerer Schildgenerator"),
    ("Wall Section", "Mauerabschnitt"),
    ("Artillery Installation", "Artillerieanlage"),
    ("Heavy Artillery Installation", "Schwere Artillerieanlage"),
    ("Tactical Missile Launcher", "Taktischer Raketenwerfer"),
    ("Strategic Missile Launcher", "Strategischer Raketenwerfer"),
    ("Tactical Missile Defense", "Taktische Raketenabwehr"),
    ("Strategic Missile Defense", "Strategische Raketenabwehr"),
    ("Torpedo Launcher", "Torpedowerfer"),
    ("Torpedo Ambush System", "Torpedo-Hinterhaltsystem"),
    ("Air Staging Facility", "Luftversorgungsstation"),
    ("Quantum Gateway", "Quantentor"),
    ("Engineer", "Ingenieur"),
    ("Light Assault Bot", "Leichter Angriffsbot"),
    ("Light Tank", "Leichter Panzer"),
    ("Medium Tank", "Mittlerer Panzer"),
    ("Heavy Tank", "Schwerer Panzer"),
    ("Assault Tank", "Sturmpanzer"),
    ("Amphibious Tank", "Amphibienpanzer"),
    ("Hover Tank", "Schwebepanzer"),
    ("Mobile Light Artillery", "Mobile leichte Artillerie"),
    ("Mobile Missile Launcher", "Mobiler Raketenwerfer"),
    ("Mobile Anti-Air Gun", "Mobiles Flugabwehrgeschütz"),
    ("Mobile AA Flak Artillery", "Mobile Flakartillerie"),
    ("Mobile Shield Generator", "Mobiler Schildgenerator"),
    ("Mobile Stealth Field System", "Mobiles Tarnfeldsystem"),
    ("Land Scout", "Landaufklärer"),
    ("Air Scout", "Luftaufklärer"),
    ("Spy Plane", "Spionageflugzeug"),
    ("Interceptor", "Abfangjäger"),
    ("Air-Superiority Fighter", "Luftüberlegenheitsjäger"),
    ("Attack Bomber", "Angriffsbomber"),
    ("Strategic Bomber", "Strategischer Bomber"),
    ("Torpedo Bomber", "Torpedobomber"),
    ("Gunship", "Kanonenboot"),
    ("Heavy Gunship", "Schweres Kanonenboot"),
    ("Light Air Transport", "Leichter Lufttransporter"),
    ("Air Transport", "Lufttransporter"),
    ("Frigate", "Fregatte"),
    ("Destroyer", "Zerstörer"),
    ("Cruiser", "Kreuzer"),
    ("Battleship", "Schlachtschiff"),
    ("Aircraft Carrier", "Flugzeugträger"),
    ("Attack Submarine", "Angriffs-U-Boot"),
    ("Strategic Missile Submarine", "Strategisches Raketen-U-Boot"),
    ("Submarine Hunter", "U-Boot-Jäger"),
    ("Experimental Mobile Factory", "Experimentelle mobile Fabrik"),
    ("Experimental Assault Bot", "Experimenteller Angriffsbot"),
    ("Experimental Spiderbot", "Experimenteller Spinnenbot"),
    ("Experimental Gunship", "Experimentelles Kanonenboot"),
    ("Experimental Aircraft Carrier", "Experimenteller Flugzeugträger"),
    ("Experimental Artillery", "Experimentelle Artillerie"),
    ("Experimental Resource Generator", "Experimenteller Ressourcengenerator"),
    ("Tech 2 Engineering Suite", "Tech-2-Ingenieurpaket"),
    ("Tech 3 Engineering Suite", "Tech-3-Ingenieurpaket"),
    ("Personal Shield Generator", "Persönlicher Schildgenerator"),
    ("Shield Generator Field", "Schildgeneratorfeld"),
    ("Resource Allocation System", "Ressourcenzuteilungssystem"),
    ("Advanced Resource Allocation System", "Verbessertes Ressourcenzuteilungssystem"),
    ("Nano-Repair System", "Nano-Reparatursystem"),
    ("Personal Teleporter", "Persönlicher Teleporter"),
    ("Personal Stealth Generator", "Persönlicher Tarngenerator"),
    ("Personal Cloaking Generator", "Persönlicher Tarnkappengenerator"),
    ("Zephyr Amplifier", "Zephyr-Verstärker"),
    ("Heavy Anti-Matter Cannon", "Schwere Antimateriekanone"),
    ("C-D1 Shoulder Drone", "C-D1-Schulterdrohne"),
    ("C-D2 Shoulder Drone", "C-D2-Schulterdrohne"),
    ("Microwave Laser Generator", "Mikrowellenlaser-Generator"),
    ("Cooling Upgrade", "Kühlungs-Upgrade"),
];

const DE_UNIT_DESCS: &[(&str, &str)] = &[
    ("uel0001", "Beherbergt den Kommandanten. Kombiniert Kommandozentrale und Baueinheit mit Grundbauplänen für eine Armee."),
    ("ual0001", "Beherbergt den Kommandanten. Kombiniert Kommandozentrale und Baueinheit mit Grundbauplänen für eine Armee."),
    ("url0001", "Beherbergt den Kommandanten. Kombiniert Kommandozentrale und Baueinheit mit Grundbauplänen für eine Armee."),
    ("xsl0001", "Beherbergt den Kommandanten. Kombiniert Kommandozentrale und Baueinheit mit Grundbauplänen für eine Armee."),
    ("ueb0101", "Produziert Tech-1-Landeinheiten. Aufwertbar auf Tech 2 und Tech 3 für fortschrittlichere Einheiten."),
    ("uab0101", "Produziert Tech-1-Landeinheiten. Aufwertbar auf Tech 2 und Tech 3 für fortschrittlichere Einheiten."),
    ("urb0101", "Produziert Tech-1-Landeinheiten. Aufwertbar auf Tech 2 und Tech 3 für fortschrittlichere Einheiten."),
    ("xsb0101", "Produziert Tech-1-Landeinheiten. Aufwertbar auf Tech 2 und Tech 3 für fortschrittlichere Einheiten."),
    ("ueb0102", "Produziert Tech-1-Lufteinheiten. Aufwertbar auf Tech 2 und Tech 3."),
    ("uab0102", "Produziert Tech-1-Lufteinheiten. Aufwertbar auf Tech 2 und Tech 3."),
    ("urb0102", "Produziert Tech-1-Lufteinheiten. Aufwertbar auf Tech 2 und Tech 3."),
    ("xsb0102", "Produziert Tech-1-Lufteinheiten. Aufwertbar auf Tech 2 und Tech 3."),
    ("ueb0103", "Produziert Tech-1-Marineeinheiten. Muss im Wasser gebaut werden."),
    ("uab0103", "Produziert Tech-1-Marineeinheiten. Muss im Wasser gebaut werden."),
    ("urb0103", "Produziert Tech-1-Marineeinheiten. Muss im Wasser gebaut werden."),
    ("xsb0103", "Produziert Tech-1-Marineeinheiten. Muss im Wasser gebaut werden."),
    ("ueb1103", "Fördert Masse aus einer Masselagerstätte. Aufwertbar auf Tech 2 und Tech 3."),
    ("uab1103", "Fördert Masse aus einer Masselagerstätte. Aufwertbar auf Tech 2 und Tech 3."),
    ("urb1103", "Fördert Masse aus einer Masselagerstätte. Aufwertbar auf Tech 2 und Tech 3."),
    ("xsb1103", "Fördert Masse aus einer Masselagerstätte. Aufwertbar auf Tech 2 und Tech 3."),
    ("ueb1101", "Erzeugt elektrische Energie für Bau, Schilde und Waffensysteme."),
    ("uab1101", "Erzeugt elektrische Energie für Bau, Schilde und Waffensysteme."),
    ("urb1101", "Erzeugt elektrische Energie für Bau, Schilde und Waffensysteme."),
    ("xsb1101", "Erzeugt elektrische Energie für Bau, Schilde und Waffensysteme."),
    ("ueb1102", "Nutzt eine Kohlenwasserstoffquelle für hohe Energieproduktion."),
    ("ueb2101", "Leichter Verteidigungsturm gegen Bodenziele."),
    ("uab2101", "Leichter Verteidigungsturm gegen Bodenziele."),
    ("urb2101", "Leichter Verteidigungsturm gegen Bodenziele."),
    ("xsb2101", "Leichter Verteidigungsturm gegen Bodenziele."),
    ("ueb3101", "Überwacht das Umfeld und erkennt feindliche Einheiten auf große Entfernung."),
    ("uab3101", "Überwacht das Umfeld und erkennt feindliche Einheiten auf große Entfernung."),
    ("urb3101", "Überwacht das Umfeld und erkennt feindliche Einheiten auf große Entfernung."),
    ("xsb3101", "Überwacht das Umfeld und erkennt feindliche Einheiten auf große Entfernung."),
    ("ueb5101", "Gepanzerter Mauerabschnitt, der feindliche Bodeneinheiten blockiert."),
    ("uel0105", "Bau-, Reparatur- und Rückgewinnungseinheit der Stufe Tech 1."),
    ("uel0201", "Mittlerer Kampfpanzer für Frontangriffe und Basisverteidigung."),
    ("uel0001-aes", "Erweitert verfügbare Baupläne und erhöht Baurate sowie maximale Lebenspunkte der ACU."),
    ("uel0001-ees", "Ersetzt das Tech-2-Ingenieurpaket und schaltet Tech-3- sowie Experimental-Baupläne frei."),
    ("uel0001-psg", "Erzeugt einen persönlichen Energieschild um die ACU. Verbraucht Energie."),
    ("uel0001-sgf", "Vergrößert den Schildradius der ACU zum Schutz umliegender Einheiten."),
    ("uel0001-isb", "Erhöht die Masse- und Energieproduktion der ACU deutlich."),
    ("uel0001-tm", "Montiert einen taktischen Marschflugkörper-Werfer auf dem Rücken der ACU."),
    ("uel0001-dsu", "Verbessert die Selbstreparaturgeschwindigkeit der ACU deutlich."),
    ("uel0001-hamc", "Erhöht Schaden und Reichweite der Hauptkanone sowie der Überladung."),
];

// ---------------------------------------------------------------------------
// CLI subcommand: `farust loc [--check] [KEY] [--lang CODE] [GAME_DIR]`
// ---------------------------------------------------------------------------

pub fn cmd_loc(args: &[String]) -> Result<(), String> {
    let mut check = false;
    let mut lang: Option<String> = None;
    let mut unit_id: Option<String> = None;
    let mut pos_args: Vec<&str> = Vec::new();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--check" => check = true,
            "--lang" => {
                i += 1;
                let v = args.get(i).ok_or("--lang needs a language code (e.g. us, de)")?;
                lang = Some(normalize_lang(v));
            }
            "--unit" => {
                i += 1;
                let v = args.get(i).ok_or("--unit needs a unit id (e.g. UEL0201)")?;
                unit_id = Some(v.clone());
            }
            s if s.starts_with('-') => return Err(format!("unknown flag '{s}' for loc")),
            s => pos_args.push(s),
        }
        i += 1;
    }

    if check {
        let game_dir = pos_args
            .first()
            .map(PathBuf::from)
            .unwrap_or_else(crate::scd::default_game_dir);
        let db = LocDb::load(&game_dir);
        println!("game: {}", game_dir.display());
        println!("active language: {} ({})", db.active_lang, language_display_name(&db.active_lang));
        println!("languages ({}):", db.languages.len());
        for l in &db.languages {
            let ext = db.tables.get(&l.code).map_or(0, HashMap::len);
            println!(
                "  {:<4} {:<16} {:<20} fa_entries={:<6} effective={}",
                l.code,
                l.name,
                l.source.describe(),
                ext,
                l.entry_count
            );
        }
        println!(
            "help tables: unitdescription={} tooltips={} enhancements_meta={}",
            db.unit_descriptions.len(),
            db.tooltips.len(),
            db.enh_meta.len()
        );
        // Verify strict parsing of every installed loc_*.scd file.
        let gamedata = crate::scd::gamedata_dir(&game_dir);
        if let Ok(rd) = std::fs::read_dir(&gamedata) {
            let mut scds: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
            scds.sort();
            for p in scds {
                let lower = p.file_name().and_then(|f| f.to_str()).unwrap_or("").to_ascii_lowercase();
                if lower.starts_with("loc_") && lower.ends_with(".scd") {
                    let files = crate::scd::read_scd_matching(&p, |n| n.to_ascii_lowercase().ends_with(".lua"))
                        .map_err(|e| format!("{}: {e}", p.display()))?;
                    for (inner, bytes) in files {
                        let strict = parse_strings_db(&bytes)
                            .map_err(|e| format!("{}::{inner}: {e}", p.display()))?;
                        println!("  strict-check {}::{inner}: {} entries OK", p.file_name().unwrap_or_default().to_string_lossy(), strict.len());
                    }
                }
            }
        }
        // Spot check US vs DE output on ACU, Tank, and Move order.
        for code in ["us", "de"] {
            let acu = db.unit_display_name_lang(code, "UEL0001", "Armored Command Unit");
            let tank = db.unit_display_name_lang(code, "UEL0201", "Medium Tank");
            let (mv_t, mv_h) = db.order_tooltip_lang(code, "move", "Move", "");
            println!("  sample[{code}]: UEL0001='{acu}' | UEL0201='{tank}' | move='{mv_t}' ({mv_h})");
        }
        return Ok(());
    }

    if let Some(uid) = unit_id {
        let game_dir = pos_args
            .first()
            .map(PathBuf::from)
            .unwrap_or_else(crate::scd::default_game_dir);
        let db = LocDb::load(&game_dir);
        let code = lang.unwrap_or_else(|| db.active_lang.clone());
        let name = db.unit_display_name_lang(&code, &uid, &uid);
        let proper = db.unit_proper_name_lang(&code, &uid);
        let desc = db.unit_description_lang(&code, &uid);
        println!("lang: {code}");
        println!("unit: {}", uid.to_ascii_uppercase());
        if let Some(p) = proper {
            println!("name: {p}");
        }
        println!("role: {name}");
        if let Some(d) = desc {
            println!("desc: {d}");
        }
        return Ok(());
    }

    let key = pos_args
        .first()
        .copied()
        .ok_or_else(|| "usage: farust loc --check [GAME_DIR] | farust loc <KEY> [--lang CODE] [GAME_DIR] | farust loc --unit <ID> [--lang CODE]".to_string())?;
    let game_dir = pos_args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(crate::scd::default_game_dir);
    let db = LocDb::load(&game_dir);
    let code = lang.unwrap_or_else(|| db.active_lang.clone());
    let val = if key.trim_start().to_ascii_lowercase().starts_with("<loc") {
        db.resolve_lang(&code, key)
    } else {
        db.lookup_lang(&code, key)
            .ok_or_else(|| format!("key '{key}' not found in language '{code}'"))?
    };
    println!("{val}");
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests (game-data-independent + optional FA install smoke test)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_strings_db_with_bom_escapes_comments_and_quotes() {
        let sample = "\u{FEFF}# Generated by Localizer.py\r\n\
            -- Lua comment\n\
            uel0001_desc=\"Armored Command Unit\"\r\n\
            uel0201_desc = \"Mittlerer Panzer\"\n\
            enh_01 = 'Tech-2-Paket\\nZeile 2'\n\
            ['quoted_key'] = \"Escaped \\\"quote\\\" and \\\\ slash\",\n\
            long_val = [[Mehrzeilige\nBeschreibung mit Umlauten: äöüÄÖÜß]]\n";
        let m = parse_strings_db(sample.as_bytes()).unwrap();
        assert_eq!(m.get("uel0001_desc").map(String::as_str), Some("Armored Command Unit"));
        assert_eq!(m.get("uel0201_desc").map(String::as_str), Some("Mittlerer Panzer"));
        assert_eq!(m.get("enh_01").map(String::as_str), Some("Tech-2-Paket\nZeile 2"));
        assert_eq!(m.get("quoted_key").map(String::as_str), Some("Escaped \"quote\" and \\ slash"));
        assert_eq!(
            m.get("long_val").map(String::as_str),
            Some("Mehrzeilige\nBeschreibung mit Umlauten: äöüÄÖÜß")
        );
    }

    #[test]
    fn parses_unitdescription_and_tooltips_with_cp1252_copyright() {
        // Includes Windows-1252 byte 0xA9 (©) just like stock FA lua.scd files.
        let mut ud = b"# Copyright ".to_vec();
        ud.push(0xA9);
        ud.extend_from_slice(
            b" 2006 GPG\nDescription = {\n ['uel0001'] = \"<LOC Unit_Description_0303>Houses Commander.\",\n ['uel0001-tm'] = '<LOC ud_tm>Mounts missile.',\n}\n",
        );
        let parsed_ud = parse_unit_descriptions(&ud).unwrap();
        assert_eq!(
            parsed_ud.get("uel0001").map(String::as_str),
            Some("<LOC Unit_Description_0303>Houses Commander.")
        );
        assert_eq!(
            parsed_ud.get("uel0001-tm").map(String::as_str),
            Some("<LOC ud_tm>Mounts missile.")
        );

        let mut tt = b"# Copyright ".to_vec();
        tt.push(0xA9);
        tt.extend_from_slice(
            b" 2006 GPG\nTooltips = {\n move = {\n  title = \"<LOC tooltipui0000>Move\",\n  description = \"\",\n  keyID = \"move\",\n },\n toggle_shield = {\n  title = \"<LOC tooltipui0040>Shield Toggle\",\n  description = \"<LOC tooltipui0481>Turn shields on/off\",\n },\n}\n",
        );
        let parsed_tt = parse_tooltips(&tt).unwrap();
        assert_eq!(parsed_tt.get("move").unwrap().title, "<LOC tooltipui0000>Move");
        assert_eq!(parsed_tt.get("move").unwrap().key_id.as_deref(), Some("move"));
        assert_eq!(
            parsed_tt.get("toggle_shield").unwrap().description,
            "<LOC tooltipui0481>Turn shields on/off"
        );
    }

    #[test]
    fn split_loc_tag_and_fallback_priority_order() {
        assert_eq!(split_loc_tag("<LOC uel0201_desc>Medium Tank"), (Some("uel0201_desc"), "Medium Tank"));
        assert_eq!(split_loc_tag("<loc  FOO_1 >Bar"), (Some("FOO_1"), "Bar"));
        assert_eq!(split_loc_tag("<LOC only_key>"), (Some("only_key"), ""));
        assert_eq!(split_loc_tag("Plain text"), (None, "Plain text"));
        assert_eq!(split_loc_tag("<LOC broken"), (None, "<LOC broken"));

        let mut db = LocDb::empty("de");
        db.tables.insert(
            "us".into(),
            HashMap::from([
                ("only_us".into(), "US Only Text".into()),
                ("both_key".into(), "US Both".into()),
            ]),
        );
        db.tables.insert(
            "de".into(),
            HashMap::from([("both_key".into(), "Deutscher Text mit Ümlaut".into())]),
        );

        // 1. Active language (`de`) wins over `us` and inline fallback.
        assert_eq!(db.resolve("<LOC both_key>Inline"), "Deutscher Text mit Ümlaut");
        // 2. Missing in `de` falls back to `us`.
        assert_eq!(db.resolve("<LOC only_us>Inline"), "US Only Text");
        // 3. Missing in both `de` and `us` falls back to inline fallback.
        assert_eq!(db.resolve("<LOC unknown_key>Inline Fallback"), "Inline Fallback");
        // 4. Empty inline fallback returns key.
        assert_eq!(db.resolve("<LOC missing_no_fb>"), "missing_no_fb");
        // 5. Switching to US selects US text.
        db.active_lang = "us".into();
        assert_eq!(db.resolve("<LOC both_key>Inline"), "US Both");
    }

    #[test]
    fn loads_synthetic_loc_de_scd_archive_and_overrides_builtin() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("farust-loc-test-{}", std::process::id()));
        let gamedata = dir.join("gamedata");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&gamedata).unwrap();

        let make_scd = |name: &str, inner: &str, content: &str| {
            let f = std::fs::File::create(gamedata.join(name)).unwrap();
            let mut z = zip::ZipWriter::new(f);
            z.start_file(inner, zip::write::SimpleFileOptions::default()).unwrap();
            z.write_all(content.as_bytes()).unwrap();
            z.finish().unwrap();
        };
        make_scd("loc_US.scd", "loc/US/strings_db.lua", "uel0201_desc=\"Medium Tank\"\nonly_us=\"Hello\"\n");
        make_scd(
            "loc_DE.scd",
            "loc/DE/strings_db.lua",
            "\u{FEFF}uel0201_desc=\"Mittlerer Kampfpanzer (SCD)\"\nuel0001_desc=\"Gepanzerte Kommandoeinheit\"\n",
        );

        let db = LocDb::load(&dir);
        assert!(db.has_external_table("us"));
        assert!(db.has_external_table("de"));
        assert_eq!(
            db.unit_display_name_lang("de", "UEL0201", "Medium Tank"),
            "Mittlerer Kampfpanzer (SCD)"
        );
        assert_eq!(db.lookup_lang("de", "only_us").as_deref(), Some("Hello"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn german_builtin_fallback_works_without_loc_de_scd() {
        let db = LocDb::empty("de");
        assert_eq!(
            db.unit_display_name_lang("de", "UEL0001", "Armored Command Unit"),
            "Gepanzerte Kommandoeinheit"
        );
        assert_eq!(
            db.unit_display_name_lang("de", "UEB1103", "Mass Extractor"),
            "Masseextraktor"
        );
        let (title, help) = db.order_tooltip_lang("de", "overcharge", "Overcharge", "");
        assert_eq!(title, "Überladen");
        assert!(help.contains("Überladungsschuss"), "{help}");
        assert_eq!(tr_lang("de", "ui.resolution", "Resolution"), "Auflösung");
    }

    #[test]
    fn parsers_reject_corrupt_inputs_without_panicking() {
        for bad in [&b""[..], b"\0\0\0", b"not an assignment", b"k = \"unterminated", b"Tooltips = { broken }"] {
            assert!(parse_strings_db(bad).is_err());
            let _ = parse_strings_db_lossy(bad);
            let _ = parse_tooltips(bad);
        }
    }

    #[test]
    fn word_wrap_respects_char_limit_and_umlauts() {
        let wrapped = word_wrap("Schützt verbündete Einheiten vor Beschuss.", 20);
        assert!(wrapped.len() >= 2, "{wrapped:?}");
        assert!(wrapped.iter().all(|l| l.chars().count() <= 20), "{wrapped:?}");
    }

    #[test]
    fn real_fa_install_loc_smoke_when_present() {
        let dir = crate::scd::default_game_dir();
        if !crate::scd::gamedata_dir(&dir).join("loc_US.scd").is_file() {
            return;
        }
        let db = LocDb::load(&dir);
        assert!(db.tables.get("us").map_or(0, HashMap::len) > 5000);
        assert!(db.unit_descriptions.len() > 300);
        assert!(db.tooltips.len() > 100);
        assert_eq!(db.unit_display_name_lang("us", "UEL0001", ""), "Armored Command Unit");
        assert_eq!(db.unit_display_name_lang("de", "UEL0001", ""), "Gepanzerte Kommandoeinheit");
        assert!(db.unit_description_lang("us", "UEL0001").is_some());
        assert!(db.unit_description_lang("de", "UEL0001").is_some());
    }
}
