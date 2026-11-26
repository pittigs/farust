//! Phase 21 in-game chat: fading chat log, text input state and the wire codec.
//!
//! `Enter` opens the input (all players), `Shift+Enter` opens it allies-only; `Enter` sends,
//! `Esc` cancels. In networked matches lines travel as [`crate::net::NetControl::Chat`] in the
//! lockstep turn packets/bundles (allies-only lines are dropped by non-allied receivers); in
//! single player they are echoed locally. Chat never touches the simulation.

use std::time::Instant;

use crate::replay::{read_str, read_u8, write_str};
use crate::skirmish::ArmyId;

/// Longest chat line accepted (characters after sanitizing).
pub const MAX_CHAT_CHARS: usize = 96;
/// Lines kept in the log.
pub const MAX_LOG_LINES: usize = 64;
/// Seconds a line stays fully visible, then fades out over [`CHAT_FADE_SECS`].
pub const CHAT_VISIBLE_SECS: f64 = 10.0;
pub const CHAT_FADE_SECS: f64 = 3.0;
/// Lines shown at once.
pub const CHAT_SHOWN_LINES: usize = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct ChatLine {
    /// `None` for system notices (pause, disconnect, desync, saves).
    pub from: Option<ArmyId>,
    pub name: String,
    pub text: String,
    pub allies: bool,
    pub at: Instant,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatInput {
    pub text: String,
    pub allies: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ChatState {
    pub lines: Vec<ChatLine>,
    pub input: Option<ChatInput>,
    /// Chat lines waiting to be sent (drained by the network layer / local echo).
    pub outbox: Vec<(bool, String)>,
    /// Phase 23: active map pings (received like allies-only chat lines).
    pub pings: Vec<Ping>,
}

// ---------------------------------------------------------------------------
// Phase 23 map pings
// ---------------------------------------------------------------------------

/// Seconds a ping marker stays on the map / minimap.
pub const PING_SECS: f64 = 8.0;
/// Pings kept at once (oldest dropped).
pub const MAX_PINGS: usize = 16;
/// Wire prefix: a ping travels as an allies-only chat line `#PING <A|M|X> <x> <z>` (world
/// milli-units), so it reuses the lockstep chat channel and its allies filter unchanged.
pub const PING_PREFIX: &str = "#PING ";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PingKind {
    Alert,
    Move,
    Attack,
}

impl PingKind {
    fn code(self) -> char {
        match self {
            PingKind::Alert => 'A',
            PingKind::Move => 'M',
            PingKind::Attack => 'X',
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            PingKind::Alert => "ALERT",
            PingKind::Move => "MOVE",
            PingKind::Attack => "ATTACK",
        }
    }

    pub fn rgb(self) -> [f32; 3] {
        match self {
            PingKind::Alert => [1.0, 0.86, 0.22],
            PingKind::Move => [0.30, 0.80, 1.0],
            PingKind::Attack => [1.0, 0.28, 0.22],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ping {
    pub from: Option<ArmyId>,
    pub kind: PingKind,
    /// World position in milli-units.
    pub x: i64,
    pub z: i64,
    pub at: Instant,
}

pub fn encode_ping(kind: PingKind, x: i64, z: i64) -> String {
    format!("{PING_PREFIX}{} {x} {z}", kind.code())
}

pub fn decode_ping(text: &str) -> Option<(PingKind, i64, i64)> {
    let rest = text.strip_prefix(PING_PREFIX)?;
    let mut it = rest.split(' ');
    let kind = match it.next()? {
        "A" => PingKind::Alert,
        "M" => PingKind::Move,
        "X" => PingKind::Attack,
        _ => return None,
    };
    let x: i64 = it.next()?.parse().ok()?;
    let z: i64 = it.next()?.parse().ok()?;
    if it.next().is_some() || x.abs() > 1 << 40 || z.abs() > 1 << 40 {
        return None;
    }
    Some((kind, x, z))
}

/// Keeps printable ASCII (what the HUD font can draw), collapses whitespace, trims and caps
/// the length.
pub fn sanitize(text: &str) -> String {
    let mut out = String::with_capacity(text.len().min(MAX_CHAT_CHARS));
    let mut last_space = true;
    for c in text.chars() {
        let c = if c.is_whitespace() { ' ' } else { c };
        if !(c.is_ascii_graphic() || c == ' ') {
            continue;
        }
        if c == ' ' {
            if last_space {
                continue;
            }
            last_space = true;
        } else {
            last_space = false;
        }
        out.push(c);
        if out.len() >= MAX_CHAT_CHARS {
            break;
        }
    }
    out.trim_end().to_string()
}

/// Wire form of a chat line: `u8 allies | str text` (text sanitized on both ends).
pub fn encode_chat(out: &mut Vec<u8>, allies: bool, text: &str) {
    out.push(u8::from(allies));
    write_str(out, &sanitize(text));
}

pub fn decode_chat(data: &[u8], pos: &mut usize) -> Result<(bool, String), String> {
    let allies = match read_u8(data, pos)? {
        0 => false,
        1 => true,
        b => return Err(format!("invalid chat flag {b}")),
    };
    let raw = read_str(data, pos)?;
    if raw.len() > 4 * MAX_CHAT_CHARS {
        return Err(format!("chat line too long ({} bytes)", raw.len()));
    }
    let text = sanitize(&raw);
    if text != raw {
        return Err("chat line is not sanitized".to_string());
    }
    Ok((allies, text))
}

impl ChatState {
    pub fn push(&mut self, from: Option<ArmyId>, name: &str, text: &str, allies: bool) {
        let text = sanitize(text);
        if text.is_empty() {
            return;
        }
        if let Some((kind, x, z)) = decode_ping(&text) {
            self.pings.push(Ping { from, kind, x, z, at: Instant::now() });
            if self.pings.len() > MAX_PINGS {
                let drop = self.pings.len() - MAX_PINGS;
                self.pings.drain(..drop);
            }
            return;
        }
        self.lines.push(ChatLine {
            from,
            name: name.to_string(),
            text,
            allies,
            at: Instant::now(),
        });
        if self.lines.len() > MAX_LOG_LINES {
            let drop = self.lines.len() - MAX_LOG_LINES;
            self.lines.drain(..drop);
        }
    }

    pub fn system(&mut self, text: &str) {
        self.push(None, "", text, false);
    }

    pub fn open(&mut self, allies: bool) {
        self.input = Some(ChatInput { text: String::new(), allies });
    }

    pub fn is_typing(&self) -> bool {
        self.input.is_some()
    }

    pub fn type_text(&mut self, s: &str) {
        if let Some(inp) = &mut self.input {
            for c in s.chars() {
                if (c.is_ascii_graphic() || c == ' ') && inp.text.len() < MAX_CHAT_CHARS {
                    inp.text.push(c);
                }
            }
        }
    }

    pub fn backspace(&mut self) {
        if let Some(inp) = &mut self.input {
            inp.text.pop();
        }
    }

    /// Closes the input; a non-empty line is queued in [`ChatState::outbox`].
    pub fn submit(&mut self) -> bool {
        let Some(inp) = self.input.take() else {
            return false;
        };
        let text = sanitize(&inp.text);
        if text.is_empty() {
            return false;
        }
        self.outbox.push((inp.allies, text));
        true
    }

    /// Queues an allies-only map ping (sent like a chat line).
    pub fn send_ping(&mut self, kind: PingKind, x: i64, z: i64) {
        self.outbox.push((true, encode_ping(kind, x, z)));
    }

    /// Live pings with their age fraction in `[0, 1)`.
    pub fn active_pings(&self, now: Instant) -> Vec<(&Ping, f32)> {
        self.pings
            .iter()
            .filter_map(|p| {
                let age = now.saturating_duration_since(p.at).as_secs_f64() / PING_SECS;
                (age < 1.0).then_some((p, age as f32))
            })
            .collect()
    }

    /// Drops expired pings; returns whether any are still live.
    pub fn prune_pings(&mut self, now: Instant) -> bool {
        self.pings.retain(|p| now.saturating_duration_since(p.at).as_secs_f64() < PING_SECS);
        !self.pings.is_empty()
    }

    /// Visible lines (newest last) with their fade alpha in `[0, 1]`.
    pub fn visible(&self, now: Instant) -> Vec<(&ChatLine, f32)> {
        let typing = self.input.is_some();
        let mut v: Vec<(&ChatLine, f32)> = self
            .lines
            .iter()
            .filter_map(|l| {
                let age = now.saturating_duration_since(l.at).as_secs_f64();
                let a = if typing || age <= CHAT_VISIBLE_SECS {
                    1.0
                } else {
                    1.0 - ((age - CHAT_VISIBLE_SECS) / CHAT_FADE_SECS)
                };
                (a > 0.0).then_some((l, a.clamp(0.0, 1.0) as f32))
            })
            .collect();
        if v.len() > CHAT_SHOWN_LINES {
            v.drain(..v.len() - CHAT_SHOWN_LINES);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn chat_codec_round_trips_and_rejects_garbage() {
        for (allies, text) in [(false, "GG WP"), (true, "push mid at 5:00 - 3 tanks!"), (false, "")] {
            let mut buf = Vec::new();
            encode_chat(&mut buf, allies, text);
            let mut pos = 0;
            assert_eq!(decode_chat(&buf, &mut pos).unwrap(), (allies, text.to_string()));
            assert_eq!(pos, buf.len());
        }
        // Encoding sanitizes: control chars / non-ASCII dropped, whitespace collapsed, capped.
        let mut buf = Vec::new();
        encode_chat(&mut buf, false, "hi\tthere\n\u{1F600}  you");
        let mut pos = 0;
        assert_eq!(decode_chat(&buf, &mut pos).unwrap().1, "hi there you");
        let long: String = "x".repeat(500);
        let mut buf = Vec::new();
        encode_chat(&mut buf, true, &long);
        let mut pos = 0;
        assert_eq!(decode_chat(&buf, &mut pos).unwrap().1.len(), MAX_CHAT_CHARS);
        // Bad flag, unsanitized payload and truncation are errors.
        let mut bad = Vec::new();
        bad.push(7);
        write_str(&mut bad, "x");
        assert!(decode_chat(&bad, &mut 0).is_err());
        let mut raw = vec![0];
        write_str(&mut raw, "tab\there");
        assert!(decode_chat(&raw, &mut 0).is_err());
        let mut ok = Vec::new();
        encode_chat(&mut ok, false, "hello");
        assert!(decode_chat(&ok[..ok.len() - 1], &mut 0).is_err());
    }

    #[test]
    fn ping_codec_round_trips_through_chat() {
        for (k, x, z) in [(PingKind::Alert, 0, 0), (PingKind::Move, 123_456, -7), (PingKind::Attack, -1, 999_999_999)] {
            let t = encode_ping(k, x, z);
            assert_eq!(sanitize(&t), t, "ping text survives the chat sanitizer");
            assert_eq!(decode_ping(&t), Some((k, x, z)));
            // And through the binary chat codec used by the network layer.
            let mut buf = Vec::new();
            encode_chat(&mut buf, true, &t);
            let (allies, back) = decode_chat(&buf, &mut 0).unwrap();
            assert!(allies);
            assert_eq!(decode_ping(&back), Some((k, x, z)));
        }
        for bad in ["#PING Q 1 2", "#PING A 1", "#PING A 1 2 3", "#PING A x 2", "PING A 1 2", "#PING A 1 99999999999999999"] {
            assert_eq!(decode_ping(bad), None, "{bad}");
        }
        let mut c = ChatState::default();
        c.send_ping(PingKind::Attack, 5_000, 6_000);
        let (allies, text) = c.outbox.pop().unwrap();
        assert!(allies);
        c.push(Some(2), "Ally", &text, allies);
        assert!(c.lines.is_empty(), "pings never show as chat lines");
        assert_eq!(c.pings.len(), 1);
        assert_eq!((c.pings[0].from, c.pings[0].kind, c.pings[0].x, c.pings[0].z), (Some(2), PingKind::Attack, 5_000, 6_000));
        let now = Instant::now();
        assert_eq!(c.active_pings(now).len(), 1);
        assert!(c.active_pings(now + Duration::from_secs_f64(PING_SECS + 0.1)).is_empty());
        assert!(c.prune_pings(now));
        assert!(!c.prune_pings(now + Duration::from_secs_f64(PING_SECS + 0.1)));
        for i in 0..40 {
            c.push(Some(0), "Me", &encode_ping(PingKind::Alert, i, i), true);
        }
        assert_eq!(c.pings.len(), MAX_PINGS);
    }

    #[test]
    fn input_submit_and_fading_log() {
        let mut c = ChatState::default();
        c.open(true);
        c.type_text("attack ");
        c.type_text("now\u{7}");
        c.backspace();
        assert!(c.is_typing());
        assert!(c.submit());
        assert_eq!(c.outbox, vec![(true, "attack no".to_string())]);
        c.open(false);
        assert!(!c.submit(), "empty lines are not sent");
        for i in 0..12 {
            c.push(Some(1), "B", &format!("line {i}"), false);
        }
        let now = Instant::now();
        let vis = c.visible(now);
        assert_eq!(vis.len(), CHAT_SHOWN_LINES);
        assert_eq!(vis.last().unwrap().0.text, "line 11");
        assert!(vis.iter().all(|(_, a)| *a == 1.0));
        let later = now + Duration::from_secs_f64(CHAT_VISIBLE_SECS + CHAT_FADE_SECS * 0.5);
        let faded = c.visible(later);
        assert!(faded.iter().all(|(_, a)| *a > 0.3 && *a < 0.7), "{faded:?}");
        assert!(c.visible(now + Duration::from_secs(60)).is_empty());
        c.open(false);
        assert_eq!(c.visible(now + Duration::from_secs(60)).len(), CHAT_SHOWN_LINES, "log reappears while typing");
    }
}
