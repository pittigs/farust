//! Phases 9 & 15 deterministic 2–8 player TCP lockstep synchronization.
//!
//! Star topology: the host (army 0) accepts `players - 1` clients (armies `1..players`).
//! Every turn (`turn_ticks` sim ticks) each peer sends a [`TurnPacket`] holding its
//! locally issued [`Command`]s plus its `Skirmish::state_hash()`; the host gathers all
//! packets, canonicalizes them into a [`TurnBundle`] (ascending `ArmyId`, per-army issue
//! order preserved) and broadcasts it. Every peer executes the identical bundle.
//!
//! Latency hiding: commands collected while turn `n` runs are scheduled for turn
//! `n + input_delay`, so a bundle normally arrives before it is needed. Each packet's
//! hash is the state at the moment it was submitted (the same tick on every peer), and
//! every peer independently verifies all hashes of each received bundle ([`LockstepCore`]),
//! producing a desync report with tick, both hashes, state summary and recent commands.
//!
//! Armies with index `>= players` are AI slots: their [`ScriptedAi`] runs on every peer
//! inside [`LockstepCore::tick`], deterministically, and never goes over the wire.
//!
//! Users:
//! - `farust host/join --headless-ticks N` → [`run_headless`] (blocking transport)
//! - `farust host/join --play` → [`NetHandle`] driving [`crate::play::PlaySession`] through the
//!   non-blocking [`AsyncLockstepBridge`] worker thread.
//!
//! Phase 21 match session (protocol v3): packets and bundles also carry [`NetControl`]s,
//! applied by every peer at the start of the bundle's turn (same tick everywhere):
//! - **Synchronized pause** — any player may pause, at most [`MAX_PAUSES_PER_PLAYER`] times
//!   per match (further requests are denied deterministically); any player may resume. A
//!   paused turn is a *null turn*: turns keep flowing at the normal pace (so a resume can be
//!   delivered) but the simulation does not step.
//! - **Speed** — only the host (army [`HOST_ARMY`]) can change the game speed (1x–8x).
//! - **Chat** — delivered in bundles; allies-only lines are discarded by non-allied peers.
//! - **Disconnects** — when the host loses a client (EOF, socket error or read timeout) it
//!   stops waiting for it and adds `Dropped(army)` to that turn's bundle; every remaining peer
//!   hands the army to a deterministic [`ScriptedAi`] at that turn, so the match continues
//!   with identical state everywhere. Losing the host (the relay) ends the match.
//! - **Desync dumps** — on a hash mismatch peers stop simulating, exchange their state
//!   summaries for a few null turns and write `desyncs/<timestamp>-army<N>/` (replay of every
//!   executed command + per-peer summaries), see [`write_desync_dump`].

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::play::PlaySession;
use crate::replay::{
    decode_command, encode_command, read_i64, read_str, read_u32, read_u64, read_u8, write_str,
};
use crate::skirmish::{ArmyId, ArmySetup, Command, Skirmish};
use crate::skirmish_ai::ScriptedAi;
use crate::udp_transport::{NetemConfig, NetemStats, ROLE_HOST, UdpChannel, UdpMux};

pub const TURN_TICKS: u64 = 3;
/// Turns between collecting a command and executing it (latency hiding).
pub const DEFAULT_INPUT_DELAY: u64 = 2;
pub const MIN_INPUT_DELAY: u64 = 1;
pub const MAX_INPUT_DELAY: u64 = 8;
pub const ADAPTIVE_MIN_DELAY: u64 = 2;
pub const ADAPTIVE_MAX_DELAY: u64 = 8;
const ADAPTIVE_EVAL_INTERVAL: u64 = 8;
const ADAPTIVE_DOWN_HYSTERESIS: u8 = 2;
/// v3: menu-hosted lobbies greet with `Welcome` (`net_lobby.rs`); in-game turns carry [`NetControl`]s.
/// v4 (Phase 24): handshake carries the enabled mods' identities.
/// v5 (Phase 28): wall sections are buildable and block pathing (older peers would reject
/// wall `Build` commands and desync), so mixed v4/v5 games are refused at the handshake.
/// v6 (Phase 29): sim rules v2 (ballistics, turrets, structure footprints).
/// v7 (Phase 30): sim rules v3 (air fuel & staging).
/// v8 (Phase 31): sim rules v4 (engineering stations & pod drones).
/// v9 (Phase 32): sim rules v5 (mobile factories & staging, hatching eggs).
/// v10 (Phase 33): sim rules v6 (Novax, EMP stun, death weapons, kamikaze).
/// v11 (Phase 34): sim rules v7 (prop & live-unit reclaim).
/// v12 (Phase 35): sim rules v8 (fire states, ground fire, guard, build pause).
/// v13 (Phase 36): sim rules v9 (waypoint moves, ferries).
/// v14 (Phase 44): reliable UDP lockstep transport & synchronized adaptive input delay (`NetControl::InputDelay`).
/// v15 (Phase 53): sim rules v10 (pings as commands).
pub const PROTOCOL_VERSION: u32 = 15;
/// The host's army (star-topology relay); only it may change the game speed.
pub const HOST_ARMY: ArmyId = 0;
/// Sender id of host-generated control messages (`Dropped`).
pub const SYSTEM_SENDER: ArmyId = 255;
/// Synchronized pauses each player may request per match.
pub const MAX_PAUSES_PER_PLAYER: u8 = 3;
/// Null turns exchanged after a desync (beyond the input delay) to collect peer summaries.
const DESYNC_EXTRA_TURNS: u64 = 3;
const MAX_CONTROLS: usize = 1024;
const MAX_FRAME_BYTES: usize = 2 * 1024 * 1024;
const NET_TIMEOUT: Duration = Duration::from_secs(60);
const HISTORY_TURNS: usize = 4;
const MSG_HANDSHAKE: u8 = 1;
const MSG_TURN: u8 = 2;
const MSG_TURN_BUNDLE: u8 = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Handshake {
    pub map_id: String,
    pub seed: u64,
    /// All armies in the match; `setups[..player_count]` are humans/peers, the rest AI.
    pub setups: Vec<ArmySetup>,
    pub player_count: usize,
    pub turn_ticks: u64,
    pub input_delay: u64,
    /// Host-authoritative tick limit for headless matches (0 = unlimited).
    pub tick_limit: u64,
    pub assigned_army: ArmyId,
    /// Phase 38: live observers the host accepts after the players (host-side only; each
    /// gets `assigned_army` ≥ [`OBSERVER_ARMY_BASE`], which no army uses).
    pub observers: usize,
}

/// Phase 38: first army id given to live observers (they send empty turns, so their
/// hashes are still desync-checked but they never affect the sim).
pub const OBSERVER_ARMY_BASE: ArmyId = 200;

/// Phase 38: army id of accept slot `slot` with `players` players (then observers).
fn slot_army(slot: usize, players: usize) -> ArmyId {
    if slot < players { slot as ArmyId } else { OBSERVER_ARMY_BASE + (slot - players) as ArmyId }
}

/// Phase 38: whether `army` belongs to a live observer.
pub fn is_observer(army: ArmyId) -> bool {
    army >= OBSERVER_ARMY_BASE
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnPacket {
    pub turn: u64,
    pub hash: u64,
    pub commands: Vec<Command>,
    /// Phase 21 session control requests (pause, speed, chat, desync summary).
    pub controls: Vec<NetControl>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnBundle {
    pub turn: u64,
    pub hashes: Vec<(ArmyId, u64)>,
    pub batches: Vec<(ArmyId, Vec<Command>)>,
    /// `(sender, control)` in canonical order (ascending sender, per-sender issue order).
    pub controls: Vec<(ArmyId, NetControl)>,
}

/// In-game control messages carried next to the commands of a turn. They never enter the
/// simulation directly: pause/speed/drop are applied by [`LockstepCore`] at the start of the
/// bundle's turn on every peer, chat is UI-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetControl {
    /// Request a synchronized pause (`true`) or resume (`false`).
    Pause(bool),
    /// Request a game speed (1–8); honored from [`HOST_ARMY`] only.
    Speed(u32),
    /// Chat line; `allies` = allies only.
    Chat { allies: bool, text: String },
    /// Host-generated ([`SYSTEM_SENDER`]): `army` disconnected; AI takes over at this turn.
    Dropped(ArmyId),
    /// Post-desync diagnostics: the sender's state summary at the desync turn.
    DesyncSummary { turn: u64, tick: u64, hash: u64, text: String },
    /// Host-generated ([`SYSTEM_SENDER`]): all peers switch `input_delay` at this turn boundary.
    InputDelay(u8),
}

impl NetControl {
    /// Controls a peer may put in its own packet (the host strips everything else).
    pub fn peer_may_send(&self) -> bool {
        !matches!(self, NetControl::Dropped(_) | NetControl::InputDelay(_))
    }
}

fn encode_control(out: &mut Vec<u8>, c: &NetControl) {
    match c {
        NetControl::Pause(p) => {
            out.push(1);
            out.push(u8::from(*p));
        }
        NetControl::Speed(s) => {
            out.push(2);
            out.extend_from_slice(&s.to_le_bytes());
        }
        NetControl::Chat { allies, text } => {
            out.push(3);
            crate::chat::encode_chat(out, *allies, text);
        }
        NetControl::Dropped(a) => {
            out.push(4);
            out.push(*a);
        }
        NetControl::DesyncSummary { turn, tick, hash, text } => {
            out.push(5);
            out.extend_from_slice(&turn.to_le_bytes());
            out.extend_from_slice(&tick.to_le_bytes());
            out.extend_from_slice(&hash.to_le_bytes());
            let t: String = text.chars().take(4000).collect();
            write_str(out, &t);
        }
        NetControl::InputDelay(d) => {
            out.push(6);
            out.push(*d);
        }
    }
}

fn decode_control(data: &[u8], pos: &mut usize) -> Result<NetControl, String> {
    match read_u8(data, pos)? {
        1 => match read_u8(data, pos)? {
            0 => Ok(NetControl::Pause(false)),
            1 => Ok(NetControl::Pause(true)),
            b => Err(format!("invalid pause flag {b}")),
        },
        2 => Ok(NetControl::Speed(read_u32(data, pos)?)),
        3 => {
            let (allies, text) = crate::chat::decode_chat(data, pos)?;
            Ok(NetControl::Chat { allies, text })
        }
        4 => Ok(NetControl::Dropped(read_u8(data, pos)?)),
        5 => Ok(NetControl::DesyncSummary {
            turn: read_u64(data, pos)?,
            tick: read_u64(data, pos)?,
            hash: read_u64(data, pos)?,
            text: read_str(data, pos)?,
        }),
        6 => Ok(NetControl::InputDelay(read_u8(data, pos)?)),
        t => Err(format!("unknown net control tag {t}")),
    }
}

fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16, String> {
    let lo = read_u8(data, pos)?;
    let hi = read_u8(data, pos)?;
    Ok(u16::from_le_bytes([lo, hi]))
}

// ---------------------------------------------------------------------------
// Framing & codecs
// ---------------------------------------------------------------------------

fn write_frame(stream: &mut TcpStream, payload: &[u8]) -> Result<(), String> {
    let len = u32::try_from(payload.len()).map_err(|_| "frame too large".to_string())?;
    let mut buf = Vec::with_capacity(4 + payload.len());
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(payload);
    stream
        .write_all(&buf)
        .map_err(|e| format!("net write: {e}"))?;
    stream.flush().map_err(|e| format!("net flush: {e}"))
}

fn read_frame(stream: &mut TcpStream) -> Result<Vec<u8>, String> {
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .map_err(|e| format!("net read len: {e}"))?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len == 0 || len > MAX_FRAME_BYTES {
        return Err(format!("invalid net frame length {len}"));
    }
    let mut buf = vec![0u8; len];
    stream
        .read_exact(&mut buf)
        .map_err(|e| format!("net read payload: {e}"))?;
    Ok(buf)
}

fn encode_handshake(h: &Handshake) -> Vec<u8> {
    let mut out = Vec::with_capacity(160);
    out.push(MSG_HANDSHAKE);
    out.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    write_str(&mut out, &h.map_id);
    out.extend_from_slice(&h.seed.to_le_bytes());
    out.extend_from_slice(&h.turn_ticks.to_le_bytes());
    out.extend_from_slice(&h.input_delay.to_le_bytes());
    out.extend_from_slice(&h.tick_limit.to_le_bytes());
    out.push(h.player_count as u8);
    out.push(h.assigned_army);
    out.extend_from_slice(&(h.setups.len() as u32).to_le_bytes());
    for s in &h.setups {
        write_str(&mut out, &s.name);
        out.extend_from_slice(&(s.faction as u32).to_le_bytes());
        out.extend_from_slice(&s.start.0.to_le_bytes());
        out.extend_from_slice(&s.start.1.to_le_bytes());
        out.push(s.team);
    }
    // Protocol v4 (Phase 24): enabled mod identities; peers must match exactly.
    crate::mods::encode_identities(&mut out, &crate::mods::active_identities());
    out
}

fn decode_handshake(data: &[u8]) -> Result<Handshake, String> {
    let mut pos = 0;
    if read_u8(data, &mut pos)? != MSG_HANDSHAKE {
        return Err("expected handshake message".to_string());
    }
    let version = read_u32(data, &mut pos)?;
    if version != PROTOCOL_VERSION {
        return Err(format!(
            "lockstep protocol mismatch: host v{version}, local v{PROTOCOL_VERSION}"
        ));
    }
    let map_id = read_str(data, &mut pos)?;
    let seed = read_u64(data, &mut pos)?;
    let turn_ticks = read_u64(data, &mut pos)?.max(1);
    let input_delay = read_u64(data, &mut pos)?.max(1);
    let tick_limit = read_u64(data, &mut pos)?;
    let player_count = read_u8(data, &mut pos)? as usize;
    let assigned_army = read_u8(data, &mut pos)?;
    let n = read_u32(data, &mut pos)? as usize;
    if n > 16 {
        return Err(format!("handshake lists {n} armies (max 16)"));
    }
    let mut setups = Vec::with_capacity(n);
    for _ in 0..n {
        let name = read_str(data, &mut pos)?;
        let faction = read_u32(data, &mut pos)? as usize;
        let x = read_i64(data, &mut pos)?;
        let y = read_i64(data, &mut pos)?;
        let team = read_u8(data, &mut pos)?;
        setups.push(ArmySetup {
            name,
            faction,
            start: (x, y),
            team,
        });
    }
    let host_mods = crate::mods::decode_identities(data, &mut pos)?;
    crate::mods::check_match(&host_mods, &crate::mods::active_identities(), "the host")?;
    if player_count < 2 || player_count > setups.len() {
        return Err(format!(
            "handshake player_count {player_count} invalid for {} armies",
            setups.len()
        ));
    }
    Ok(Handshake {
        observers: 0,
        map_id,
        seed,
        setups,
        player_count,
        turn_ticks,
        input_delay,
        tick_limit,
        assigned_army,
    })
}

fn encode_turn(p: &TurnPacket) -> Vec<u8> {
    let mut out = Vec::with_capacity(64);
    out.push(MSG_TURN);
    out.extend_from_slice(&p.turn.to_le_bytes());
    out.extend_from_slice(&p.hash.to_le_bytes());
    out.extend_from_slice(&(p.commands.len() as u32).to_le_bytes());
    for cmd in &p.commands {
        encode_command(&mut out, cmd);
    }
    let n = p.controls.len().min(MAX_CONTROLS);
    out.extend_from_slice(&(n as u16).to_le_bytes());
    for c in p.controls.iter().take(n) {
        encode_control(&mut out, c);
    }
    out
}

fn decode_turn(data: &[u8]) -> Result<TurnPacket, String> {
    let mut pos = 0;
    if read_u8(data, &mut pos)? != MSG_TURN {
        return Err("expected turn packet".to_string());
    }
    let turn = read_u64(data, &mut pos)?;
    let hash = read_u64(data, &mut pos)?;
    let n = read_u32(data, &mut pos)? as usize;
    let mut commands = Vec::with_capacity(n.min(4096));
    for _ in 0..n {
        commands.push(decode_command(data, &mut pos)?);
    }
    let nc = read_u16(data, &mut pos)? as usize;
    if nc > MAX_CONTROLS {
        return Err(format!("turn packet lists {nc} controls"));
    }
    let mut controls = Vec::with_capacity(nc);
    for _ in 0..nc {
        controls.push(decode_control(data, &mut pos)?);
    }
    if pos != data.len() {
        return Err(format!("turn packet has {} trailing bytes", data.len() - pos));
    }
    Ok(TurnPacket { turn, hash, commands, controls })
}

fn encode_turn_bundle(b: &TurnBundle) -> Vec<u8> {
    let mut out = Vec::with_capacity(128);
    out.push(MSG_TURN_BUNDLE);
    out.extend_from_slice(&b.turn.to_le_bytes());
    out.push(b.hashes.len().min(255) as u8);
    for &(army, hash) in b.hashes.iter().take(255) {
        out.push(army);
        out.extend_from_slice(&hash.to_le_bytes());
    }
    out.push(b.batches.len().min(255) as u8);
    for (army, cmds) in b.batches.iter().take(255) {
        out.push(*army);
        out.extend_from_slice(&(cmds.len() as u32).to_le_bytes());
        for cmd in cmds {
            encode_command(&mut out, cmd);
        }
    }
    let n = b.controls.len().min(MAX_CONTROLS);
    out.extend_from_slice(&(n as u16).to_le_bytes());
    for (army, c) in b.controls.iter().take(n) {
        out.push(*army);
        encode_control(&mut out, c);
    }
    out
}

fn decode_turn_bundle(data: &[u8]) -> Result<TurnBundle, String> {
    let mut pos = 0;
    if read_u8(data, &mut pos)? != MSG_TURN_BUNDLE {
        return Err("expected turn bundle packet".to_string());
    }
    let turn = read_u64(data, &mut pos)?;
    let nh = read_u8(data, &mut pos)? as usize;
    let mut hashes = Vec::with_capacity(nh);
    for _ in 0..nh {
        let army = read_u8(data, &mut pos)?;
        let hash = read_u64(data, &mut pos)?;
        hashes.push((army, hash));
    }
    let nb = read_u8(data, &mut pos)? as usize;
    let mut batches = Vec::with_capacity(nb);
    for _ in 0..nb {
        let army = read_u8(data, &mut pos)?;
        let nc = read_u32(data, &mut pos)? as usize;
        let mut cmds = Vec::with_capacity(nc.min(4096));
        for _ in 0..nc {
            cmds.push(decode_command(data, &mut pos)?);
        }
        batches.push((army, cmds));
    }
    let nc = read_u16(data, &mut pos)? as usize;
    if nc > MAX_CONTROLS {
        return Err(format!("turn bundle lists {nc} controls"));
    }
    let mut controls = Vec::with_capacity(nc);
    for _ in 0..nc {
        let army = read_u8(data, &mut pos)?;
        controls.push((army, decode_control(data, &mut pos)?));
    }
    if pos != data.len() {
        return Err(format!("turn bundle has {} trailing bytes", data.len() - pos));
    }
    Ok(canonicalize_bundle(TurnBundle {
        turn,
        hashes,
        batches,
        controls,
    }))
}

/// Canonical execution order shared by all peers: hashes and batches sorted by ascending
/// `ArmyId`; duplicate army batches are merged (in arrival order) so per-army issue
/// order is preserved; empty batches are dropped. Controls are stably sorted by sender.
pub fn canonicalize_bundle(mut b: TurnBundle) -> TurnBundle {
    b.hashes.sort_by_key(|(a, _)| *a);
    b.hashes.dedup_by_key(|(a, _)| *a);
    b.controls.sort_by_key(|(a, _)| *a);
    // Stable sort keeps arrival order among equal armies.
    b.batches.sort_by_key(|(a, _)| *a);
    let mut merged: Vec<(ArmyId, Vec<Command>)> = Vec::with_capacity(b.batches.len());
    for (army, cmds) in b.batches {
        if cmds.is_empty() {
            continue;
        }
        match merged.last_mut() {
            Some((last, v)) if *last == army => v.extend(cmds),
            _ => merged.push((army, cmds)),
        }
    }
    b.batches = merged;
    b
}

/// Issues a bundle's commands into `world` in canonical order (for the next `step`).
pub fn issue_bundle(world: &mut Skirmish, bundle: &TurnBundle) {
    for (army, cmds) in &bundle.batches {
        for c in cmds {
            world.issue(*army, c.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// Transports: star-topology host & client (TCP + reliable-ordered UDP)
// ---------------------------------------------------------------------------

/// Blocking per-turn exchange: submit the local packet, get the canonical bundle back.
pub trait TurnTransport: Send {
    fn exchange(&mut self, pkt: TurnPacket) -> Result<TurnBundle, String>;
}

/// Transport abstraction so the lockstep core runs identically over TCP or reliable-ordered UDP.
pub enum PeerStream {
    Tcp(TcpStream),
    Udp(UdpChannel),
}

impl PeerStream {
    pub fn write_frame(&mut self, payload: &[u8]) -> Result<(), String> {
        match self {
            PeerStream::Tcp(s) => write_frame(s, payload),
            PeerStream::Udp(c) => c.write_frame(payload),
        }
    }

    pub fn read_frame(&mut self) -> Result<Vec<u8>, String> {
        match self {
            PeerStream::Tcp(s) => read_frame(s),
            PeerStream::Udp(c) => c.read_frame(),
        }
    }

    pub fn set_read_timeout(&mut self, timeout: Option<Duration>) {
        match self {
            PeerStream::Tcp(s) => {
                let _ = s.set_read_timeout(timeout);
            }
            PeerStream::Udp(c) => c.set_read_timeout(timeout),
        }
    }

    pub fn shutdown(&self) {
        match self {
            PeerStream::Tcp(s) => {
                let _ = s.shutdown(Shutdown::Both);
            }
            PeerStream::Udp(c) => c.shutdown(),
        }
    }

    pub fn rtt_stats(&self) -> Option<(f64, f64)> {
        match self {
            PeerStream::Tcp(_) => None,
            PeerStream::Udp(c) => c.rtt_stats(),
        }
    }

    pub fn netem_stats(&self) -> Option<NetemStats> {
        match self {
            PeerStream::Tcp(_) => None,
            PeerStream::Udp(c) => Some(c.netem_stats()),
        }
    }

    #[allow(dead_code)]
    pub fn is_udp(&self) -> bool {
        matches!(self, PeerStream::Udp(_))
    }
}

/// Computes the target lockstep `input_delay` (in turns) from measured `(srtt_ms, rttvar_ms)`.
///
/// Uses `srtt + 2 * rttvar` against half a nominal turn (`turn_ticks * 50 ms`, i.e. 150 ms for
/// `TURN_TICKS = 3`), clamped to `[ADAPTIVE_MIN_DELAY, ADAPTIVE_MAX_DELAY]` (`2..=8`).
/// Local/LAN/lossy-loopback links stay at `2` turns (`DEFAULT_INPUT_DELAY`); high-RTT links
/// automatically scale up to prevent stalls.
pub fn compute_adaptive_delay(srtt_ms: f64, rttvar_ms: f64, turn_ticks: u64) -> u64 {
    let effective_ms = srtt_ms.max(0.0) + 2.0 * rttvar_ms.max(0.0);
    let half_turn_ms = (turn_ticks.max(1) as f64) * 50.0;
    let raw = (effective_ms / half_turn_ms).ceil() as u64 + 1;
    raw.clamp(ADAPTIVE_MIN_DELAY, ADAPTIVE_MAX_DELAY)
}

/// Host-side adaptive input-delay controller with hysteresis to prevent oscillation.
#[derive(Debug, Clone)]
pub struct AdaptiveDelayController {
    pub enabled: bool,
    pub current_delay: u64,
    pub turn_ticks: u64,
    last_eval_turn: u64,
    low_streak: u8,
}

impl AdaptiveDelayController {
    pub fn new(enabled: bool, initial_delay: u64, turn_ticks: u64) -> Self {
        Self {
            enabled,
            current_delay: initial_delay.clamp(MIN_INPUT_DELAY, MAX_INPUT_DELAY),
            turn_ticks: turn_ticks.max(1),
            last_eval_turn: 0,
            low_streak: 0,
        }
    }

    /// Evaluates worst-peer `(srtt_ms, rttvar_ms)` at `turn`; returns `Some(new_delay)` only
    /// when a synchronized delay change should be emitted in this turn's bundle.
    pub fn maybe_adjust(&mut self, turn: u64, worst_srtt_ms: f64, worst_rttvar_ms: f64) -> Option<u64> {
        if !self.enabled || turn < 4 || turn.saturating_sub(self.last_eval_turn) < ADAPTIVE_EVAL_INTERVAL {
            return None;
        }
        self.last_eval_turn = turn;
        let desired = compute_adaptive_delay(worst_srtt_ms, worst_rttvar_ms, self.turn_ticks);
        if desired > self.current_delay {
            self.low_streak = 0;
            self.current_delay = desired;
            return Some(desired);
        }
        if desired < self.current_delay {
            let effective_ms = worst_srtt_ms.max(0.0) + 2.0 * worst_rttvar_ms.max(0.0);
            let half_turn_ms = (self.turn_ticks as f64) * 50.0;
            let down_threshold = ((self.current_delay.saturating_sub(2)) as f64) * half_turn_ms;
            if effective_ms + 30.0 <= down_threshold.max(half_turn_ms) {
                self.low_streak = self.low_streak.saturating_add(1);
                if self.low_streak >= ADAPTIVE_DOWN_HYSTERESIS {
                    self.low_streak = 0;
                    self.current_delay = self.current_delay.saturating_sub(1).max(ADAPTIVE_MIN_DELAY);
                    return Some(self.current_delay);
                }
            } else {
                self.low_streak = 0;
            }
        } else {
            self.low_streak = 0;
        }
        None
    }
}

pub struct LockstepHost {
    clients: Vec<(ArmyId, PeerStream)>,
    pub local_army: ArmyId,
    next_turn: u64,
    adaptive: AdaptiveDelayController,
}

impl LockstepHost {
    /// Enables or disables host-driven adaptive `input_delay` announcements.
    #[allow(dead_code)]
    pub fn set_adaptive_delay(&mut self, enabled: bool) {
        self.adaptive.enabled = enabled;
    }

    /// Returns the worst-peer `(srtt_ms, rttvar_ms)` across UDP peers, if available.
    #[allow(dead_code)]
    pub fn worst_rtt_stats(&self) -> Option<(f64, f64)> {
        let mut worst: Option<(f64, f64)> = None;
        for (_, stream) in &self.clients {
            if let Some((srtt, var)) = stream.rtt_stats() {
                let score = srtt + 2.0 * var;
                if worst.is_none_or(|(ws, wv)| score > ws + 2.0 * wv) {
                    worst = Some((srtt, var));
                }
            }
        }
        worst
    }

    /// Returns Netem impairment counters from the host's UDP mux, if running over UDP.
    pub fn netem_stats(&self) -> Option<NetemStats> {
        self.clients.iter().find_map(|(_, s)| s.netem_stats())
    }

    /// Binds `addr`, accepts `hs.player_count - 1` remote clients (`2..=8` players) and
    /// sends each its [`Handshake`] (with `assigned_army` filled in). `on_join` is called
    /// after every accepted client with `(army, peer_addr)`.
    pub fn bind_players(
        addr: &str,
        hs: &Handshake,
        mut on_join: impl FnMut(ArmyId, SocketAddr),
    ) -> Result<(Self, SocketAddr), String> {
        let listener =
            TcpListener::bind(addr).map_err(|e| format!("failed to bind {addr}: {e}"))?;
        let local_addr = listener
            .local_addr()
            .map_err(|e| format!("local_addr failed: {e}"))?;
        Self::accept_players(listener, hs, &mut on_join).map(|h| (h, local_addr))
    }

    /// Binds a UDP socket on `addr` (with optional `netem`), accepts `hs.player_count - 1`
    /// reliable-ordered UDP clients, and sends each its [`Handshake`].
    pub fn bind_players_udp(
        addr: &str,
        hs: &Handshake,
        netem: NetemConfig,
        mut on_join: impl FnMut(ArmyId, SocketAddr),
    ) -> Result<(Self, SocketAddr), String> {
        let mux = UdpMux::bind(addr, netem)?;
        let local_addr = mux.local_addr();
        Self::accept_players_udp(mux, hs, &mut on_join).map(|h| (h, local_addr))
    }

    /// Accepts `hs.player_count - 1` reliable-ordered UDP clients on an already-bound [`UdpMux`].
    pub fn accept_players_udp(
        mux: Arc<UdpMux>,
        hs: &Handshake,
        on_join: &mut dyn FnMut(ArmyId, SocketAddr),
    ) -> Result<Self, String> {
        let n = hs.player_count;
        if !(2..=8).contains(&n) {
            return Err(format!("player count {n} out of range 2..=8"));
        }
        if hs.setups.len() < n {
            return Err(format!(
                "{n} players need at least {n} armies, got {}",
                hs.setups.len()
            ));
        }
        mux.set_accepting(true);
        let mut clients = Vec::with_capacity(n - 1 + hs.observers);
        for slot in 1..n + hs.observers {
            let mut ch = mux
                .accept(None)
                .map_err(|e| format!("udp accept slot {slot} failed: {e}"))?;
            let peer = ch.peer_addr();
            ch.set_read_timeout(Some(NET_TIMEOUT));
            let army = slot_army(slot, n);
            let mut client_hs = hs.clone();
            client_hs.assigned_army = army;
            ch.write_frame(&encode_handshake(&client_hs))?;
            on_join(army, peer);
            clients.push((army, PeerStream::Udp(ch)));
        }
        mux.set_accepting(false);
        Ok(Self {
            clients,
            local_army: 0,
            next_turn: 0,
            adaptive: AdaptiveDelayController::new(true, hs.input_delay, hs.turn_ticks),
        })
    }

    /// Binds BOTH UDP (reliable-ordered + NAT hole punching) and TCP (direct + relay fallback)
    /// and registers an invite code on `rendezvous_addr`. Each client connects over punched UDP
    /// when possible, or falls back transparently to TCP direct / TCP relay.
    pub fn bind_players_udp_with_code(
        addr: &str,
        rendezvous_addr: &str,
        hs: &Handshake,
        netem: NetemConfig,
        mut on_join: impl FnMut(ArmyId, SocketAddr, crate::rendezvous::ConnectVia),
    ) -> Result<(Self, SocketAddr, String), String> {
        let n = hs.player_count;
        if !(2..=8).contains(&n) {
            return Err(format!("player count {n} out of range 2..=8"));
        }
        if hs.setups.len() < n {
            return Err(format!("{n} players need at least {n} armies, got {}", hs.setups.len()));
        }
        let listener =
            TcpListener::bind(addr).map_err(|e| format!("failed to bind {addr}: {e}"))?;
        let local_addr = listener
            .local_addr()
            .map_err(|e| format!("local_addr failed: {e}"))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| format!("set_nonblocking failed: {e}"))?;

        let udp_mux = match UdpMux::bind(&local_addr.to_string(), netem.clone()) {
            Ok(m) => m,
            Err(_) => {
                let fallback = if local_addr.is_ipv4() {
                    if local_addr.ip().is_loopback() { "127.0.0.1:0" } else { "0.0.0.0:0" }
                } else if local_addr.ip().is_loopback() {
                    "[::1]:0"
                } else {
                    "[::]:0"
                };
                UdpMux::bind(fallback, netem)?
            }
        };
        udp_mux.set_accepting(true);

        let (tx, rx) = mpsc::channel::<(PeerStream, SocketAddr, crate::rendezvous::ConnectVia)>();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let (atx, astop) = (tx.clone(), stop.clone());
        let tcp_thread = std::thread::spawn(move || {
            while !astop.load(std::sync::atomic::Ordering::Relaxed) {
                match listener.accept() {
                    Ok((s, p)) => {
                        let _ = s.set_nonblocking(false);
                        if atx
                            .send((PeerStream::Tcp(s), p, crate::rendezvous::ConnectVia::Direct(p)))
                            .is_err()
                        {
                            return;
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    Err(_) => return,
                }
            }
        });

        let (utx, ustop, umux) = (tx.clone(), stop.clone(), udp_mux.clone());
        let udp_thread = std::thread::spawn(move || {
            while !ustop.load(std::sync::atomic::Ordering::Relaxed) {
                if let Ok(ch) = umux.accept(Some(Duration::from_millis(50))) {
                    let p = ch.peer_addr();
                    if utx
                        .send((PeerStream::Udp(ch), p, crate::rendezvous::ConnectVia::UdpPunched(p)))
                        .is_err()
                    {
                        return;
                    }
                }
            }
        });

        let rtx = tx.clone();
        let punch_mux = udp_mux.clone();
        let reg = crate::rendezvous::HostRegistration::register_with_udp(
            rendezvous_addr,
            local_addr,
            udp_mux.clone(),
            "Host",
            &hs.map_id,
            Duration::from_secs(4),
            Box::new(move |s, p| {
                let _ = rtx.send((PeerStream::Tcp(s), p, crate::rendezvous::ConnectVia::Relayed(p)));
            }),
            Box::new(move |token, targets| {
                let _ = punch_mux.start_punch(token, ROLE_HOST, targets, Duration::from_secs(3));
            }),
        );
        let mut reg = match reg {
            Ok(r) => r,
            Err(e) => {
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
                udp_mux.set_accepting(false);
                let _ = tcp_thread.join();
                let _ = udp_thread.join();
                return Err(e);
            }
        };
        let code = reg.code.clone();
        println!("Invite code: {code} (rendezvous {rendezvous_addr}, UDP punch + TCP relay)");

        let mut clients = Vec::with_capacity(n - 1 + hs.observers);
        for slot in 1..n + hs.observers {
            let (mut stream, peer, via) = rx
                .recv()
                .map_err(|_| format!("accept slot {slot} failed: listener closed"))?;
            if let PeerStream::Tcp(s) = &stream {
                let _ = s.set_nodelay(true);
            }
            stream.set_read_timeout(Some(NET_TIMEOUT));
            let army = slot_army(slot, n);
            let mut client_hs = hs.clone();
            client_hs.assigned_army = army;
            stream.write_frame(&encode_handshake(&client_hs))?;
            on_join(army, peer, via);
            clients.push((army, stream));
        }

        reg.close();
        udp_mux.set_accepting(false);
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = tcp_thread.join();
        let _ = udp_thread.join();

        let has_udp = clients.iter().any(|(_, s)| s.is_udp());
        Ok((
            Self {
                clients,
                local_army: 0,
                next_turn: 0,
                adaptive: AdaptiveDelayController::new(has_udp, hs.input_delay, hs.turn_ticks),
            },
            local_addr,
            code,
        ))
    }

    /// Binds `addr` AND registers an invite code on `rendezvous_addr`, accepting `hs.player_count - 1`
    /// players from either direct TCP or relayed tunnels. `on_join(army, peer_addr, relayed)`.
    pub fn bind_players_with_code(
        addr: &str,
        rendezvous_addr: &str,
        hs: &Handshake,
        mut on_join: impl FnMut(ArmyId, SocketAddr, bool),
    ) -> Result<(Self, SocketAddr, String), String> {
        let n = hs.player_count;
        if !(2..=8).contains(&n) {
            return Err(format!("player count {n} out of range 2..=8"));
        }
        if hs.setups.len() < n {
            return Err(format!("{n} players need at least {n} armies, got {}", hs.setups.len()));
        }
        let listener =
            TcpListener::bind(addr).map_err(|e| format!("failed to bind {addr}: {e}"))?;
        let local_addr = listener
            .local_addr()
            .map_err(|e| format!("local_addr failed: {e}"))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| format!("set_nonblocking failed: {e}"))?;

        let (tx, rx) = mpsc::channel::<(TcpStream, SocketAddr, bool)>();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (atx, astop) = (tx.clone(), stop.clone());
        let accept_thread = std::thread::spawn(move || {
            while !astop.load(std::sync::atomic::Ordering::Relaxed) {
                match listener.accept() {
                    Ok((s, p)) => {
                        let _ = s.set_nonblocking(false);
                        if atx.send((s, p, false)).is_err() {
                            return;
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    Err(_) => return,
                }
            }
        });

        let rtx = tx.clone();
        let reg = crate::rendezvous::HostRegistration::register(
            rendezvous_addr,
            local_addr,
            "Host",
            &hs.map_id,
            Duration::from_secs(4),
            Box::new(move |s, p| {
                let _ = rtx.send((s, p, true));
            }),
        );
        let mut reg = match reg {
            Ok(r) => r,
            Err(e) => {
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
                let _ = accept_thread.join();
                return Err(e);
            }
        };
        let code = reg.code.clone();
        println!("Invite code: {code} (rendezvous {rendezvous_addr})");

        let mut clients = Vec::with_capacity(n - 1 + hs.observers);
        for slot in 1..n + hs.observers {
            let (mut stream, peer, relayed) = rx
                .recv()
                .map_err(|_| format!("accept slot {slot} failed: listener closed"))?;
            let _ = stream.set_nodelay(true);
            let _ = stream.set_read_timeout(Some(NET_TIMEOUT));
            let army = slot_army(slot, n);
            let mut client_hs = hs.clone();
            client_hs.assigned_army = army;
            write_frame(&mut stream, &encode_handshake(&client_hs))?;
            on_join(army, peer, relayed);
            clients.push((army, PeerStream::Tcp(stream)));
        }

        reg.close();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = accept_thread.join();

        Ok((
            Self {
                clients,
                local_army: 0,
                next_turn: 0,
                adaptive: AdaptiveDelayController::new(false, hs.input_delay, hs.turn_ticks),
            },
            local_addr,
            code,
        ))
    }

    pub fn accept_players(
        listener: TcpListener,
        hs: &Handshake,
        on_join: &mut dyn FnMut(ArmyId, SocketAddr),
    ) -> Result<Self, String> {
        let n = hs.player_count;
        if !(2..=8).contains(&n) {
            return Err(format!("player count {n} out of range 2..=8"));
        }
        if hs.setups.len() < n {
            return Err(format!(
                "{n} players need at least {n} armies, got {}",
                hs.setups.len()
            ));
        }
        let mut clients = Vec::with_capacity(n - 1 + hs.observers);
        for slot in 1..n + hs.observers {
            let (mut stream, peer) = listener
                .accept()
                .map_err(|e| format!("accept slot {slot} failed: {e}"))?;
            let _ = stream.set_nodelay(true);
            let _ = stream.set_read_timeout(Some(NET_TIMEOUT));
            let army = slot_army(slot, n);
            let mut client_hs = hs.clone();
            client_hs.assigned_army = army;
            write_frame(&mut stream, &encode_handshake(&client_hs))?;
            on_join(army, peer);
            clients.push((army, PeerStream::Tcp(stream)));
        }
        Ok(Self {
            clients,
            local_army: 0,
            next_turn: 0,
            adaptive: AdaptiveDelayController::new(false, hs.input_delay, hs.turn_ticks),
        })
    }
}

impl TurnTransport for LockstepHost {
    fn exchange(&mut self, pkt: TurnPacket) -> Result<TurnBundle, String> {
        let turn = self.next_turn;
        if pkt.turn != turn {
            return Err(format!("host submitted turn {} but expected {turn}", pkt.turn));
        }
        let mut hashes = Vec::with_capacity(self.clients.len() + 1);
        let mut batches = Vec::with_capacity(self.clients.len() + 1);
        let mut controls: Vec<(ArmyId, NetControl)> = Vec::new();
        hashes.push((self.local_army, pkt.hash));
        batches.push((self.local_army, pkt.commands));
        let local = self.local_army;
        controls.extend(pkt.controls.into_iter().filter(NetControl::peer_may_send).map(|c| (local, c)));
        // Phase 21: a client that fails (EOF, socket error, read timeout, garbage, wrong
        // turn) is dropped instead of ending the match; the bundle announces it.
        let mut dropped: Vec<ArmyId> = Vec::new();
        for (army, stream) in &mut self.clients {
            let p = match stream.read_frame().and_then(|raw| decode_turn(&raw)) {
                Ok(p) if p.turn == turn => p,
                Ok(p) => {
                    eprintln!("lockstep host: army {army} sent turn {} at turn {turn}; dropping it", p.turn);
                    dropped.push(*army);
                    continue;
                }
                Err(e) => {
                    eprintln!("lockstep host: army {army} disconnected at turn {turn} ({e}); AI takes over");
                    dropped.push(*army);
                    continue;
                }
            };
            hashes.push((*army, p.hash));
            batches.push((*army, p.commands));
            let a = *army;
            controls.extend(p.controls.into_iter().filter(NetControl::peer_may_send).map(|c| (a, c)));
        }
        if !dropped.is_empty() {
            self.clients.retain(|(a, _)| !dropped.contains(a));
            controls.extend(dropped.iter().map(|&a| (SYSTEM_SENDER, NetControl::Dropped(a))));
        }
        // Phase 44: adaptive input delay announced at agreed turn boundary from measured RTT + jitter.
        if let Some((srtt, rttvar)) = self.worst_rtt_stats()
            && let Some(new_delay) = self.adaptive.maybe_adjust(turn, srtt, rttvar)
        {
            controls.push((SYSTEM_SENDER, NetControl::InputDelay(new_delay as u8)));
        }
        let bundle = canonicalize_bundle(TurnBundle {
            turn,
            hashes,
            batches,
            controls,
        });
        let encoded = encode_turn_bundle(&bundle);
        for (army, stream) in &mut self.clients {
            if let Err(e) = stream.write_frame(&encoded) {
                // The next read fails too, so the drop is announced in the next bundle.
                eprintln!("lockstep host: send to army {army} failed: {e}");
                stream.shutdown();
            }
        }
        self.next_turn += 1;
        Ok(bundle)
    }
}

pub struct LockstepClient {
    stream: PeerStream,
    pub local_army: ArmyId,
    next_turn: u64,
}

impl LockstepClient {
    pub fn from_stream(mut stream: TcpStream) -> Result<(Self, Handshake), String> {
        let _ = stream.set_nodelay(true);
        let _ = stream.set_read_timeout(Some(NET_TIMEOUT));
        let raw = read_frame(&mut stream)?;
        let hs = if raw.first() == Some(&MSG_HANDSHAKE) {
            decode_handshake(&raw)?
        } else {
            // Phase 20: a menu-hosted lobby greets first; join it as an auto-ready peer.
            let hs = crate::net_lobby::auto_join(&mut stream, &raw)?;
            let _ = stream.set_read_timeout(Some(NET_TIMEOUT));
            hs
        };
        let client = Self {
            stream: PeerStream::Tcp(stream),
            local_army: hs.assigned_army,
            next_turn: 0,
        };
        Ok((client, hs))
    }

    pub fn from_udp(mut channel: UdpChannel) -> Result<(Self, Handshake), String> {
        channel.set_read_timeout(Some(NET_TIMEOUT));
        let raw = channel.read_frame()?;
        let hs = decode_handshake(&raw)?;
        let client = Self {
            stream: PeerStream::Udp(channel),
            local_army: hs.assigned_army,
            next_turn: 0,
        };
        Ok((client, hs))
    }

    pub fn connect(addr: &str) -> Result<(Self, Handshake), String> {
        let stream =
            TcpStream::connect(addr).map_err(|e| format!("failed to connect {addr}: {e}"))?;
        Self::from_stream(stream)
    }

    pub fn connect_udp(addr: &str, netem: NetemConfig, timeout: Duration) -> Result<(Self, Handshake), String> {
        let channel = UdpChannel::connect(addr, netem, timeout)?;
        Self::from_udp(channel)
    }

    pub fn connect_udp_retry(
        addr: &str,
        netem: NetemConfig,
        timeout: Duration,
    ) -> Result<(Self, Handshake), String> {
        let t0 = Instant::now();
        loop {
            let left = timeout.saturating_sub(t0.elapsed()).max(Duration::from_millis(200));
            match Self::connect_udp(addr, netem.clone(), left.min(Duration::from_secs(2))) {
                Ok(c) => return Ok(c),
                Err(e) if t0.elapsed() >= timeout => return Err(e),
                Err(_) => std::thread::sleep(Duration::from_millis(50)),
            }
        }
    }

    /// Connects via invite code trying UDP NAT hole punching first, then TCP direct / TCP relay.
    pub fn connect_code_udp_retry(
        rendezvous_addr: &str,
        code: &str,
        force_relay: bool,
        netem: NetemConfig,
        timeout: Duration,
    ) -> Result<(Self, Handshake, crate::rendezvous::ConnectVia), String> {
        let t0 = Instant::now();
        loop {
            match crate::rendezvous::connect_via_code_udp(
                rendezvous_addr,
                code,
                force_relay,
                netem.clone(),
                Duration::from_secs(3),
                Duration::from_millis(800),
            ) {
                Ok((crate::rendezvous::RendezvousStream::Udp(ch), via)) => match Self::from_udp(ch) {
                    Ok((c, hs)) => return Ok((c, hs, via)),
                    Err(e) if t0.elapsed() >= timeout => return Err(e),
                    Err(_) => std::thread::sleep(Duration::from_millis(80)),
                },
                Ok((crate::rendezvous::RendezvousStream::Tcp(s), via)) => match Self::from_stream(s) {
                    Ok((c, hs)) => return Ok((c, hs, via)),
                    Err(e) if t0.elapsed() >= timeout => return Err(e),
                    Err(_) => std::thread::sleep(Duration::from_millis(80)),
                },
                Err(e) if e.contains("invalid invite code") || t0.elapsed() >= timeout => {
                    return Err(e);
                }
                Err(_) => std::thread::sleep(Duration::from_millis(100)),
            }
        }
    }

    /// Connects to a host via `XXXX-XXXX` invite code (direct TCP first, automatic TCP relay fallback).
    pub fn connect_code_retry(
        rendezvous_addr: &str,
        code: &str,
        force_relay: bool,
        timeout: Duration,
    ) -> Result<(Self, Handshake, crate::rendezvous::ConnectVia), String> {
        let t0 = Instant::now();
        loop {
            match crate::rendezvous::connect_via_code(
                rendezvous_addr,
                code,
                force_relay,
                Duration::from_millis(800),
            ) {
                Ok((stream, via)) => match Self::from_stream(stream) {
                    Ok((c, hs)) => return Ok((c, hs, via)),
                    Err(e) if t0.elapsed() >= timeout => return Err(e),
                    Err(_) => std::thread::sleep(Duration::from_millis(80)),
                },
                Err(e) if e.contains("invalid invite code") || t0.elapsed() >= timeout => {
                    return Err(e);
                }
                Err(_) => std::thread::sleep(Duration::from_millis(100)),
            }
        }
    }

    /// Retries `connect` until it succeeds or `timeout` elapses (host may start later).
    pub fn connect_retry(addr: &str, timeout: Duration) -> Result<(Self, Handshake), String> {
        let t0 = Instant::now();
        loop {
            match Self::connect(addr) {
                Ok(c) => return Ok(c),
                Err(e) if t0.elapsed() >= timeout => return Err(e),
                Err(_) => std::thread::sleep(Duration::from_millis(50)),
            }
        }
    }

    #[allow(dead_code)]
    pub fn rtt_stats(&self) -> Option<(f64, f64)> {
        self.stream.rtt_stats()
    }

    pub fn netem_stats(&self) -> Option<NetemStats> {
        self.stream.netem_stats()
    }
}

impl TurnTransport for LockstepClient {
    fn exchange(&mut self, pkt: TurnPacket) -> Result<TurnBundle, String> {
        let turn = self.next_turn;
        if pkt.turn != turn {
            return Err(format!("client submitted turn {} but expected {turn}", pkt.turn));
        }
        self.stream.write_frame(&encode_turn(&pkt))?;
        let raw = self.stream.read_frame().map_err(|e| format!("host disconnected: {e}"))?;
        let bundle = decode_turn_bundle(&raw)?;
        if bundle.turn != turn {
            return Err(format!(
                "lockstep turn mismatch on army {}: expected {turn}, got {}",
                self.local_army, bundle.turn
            ));
        }
        self.next_turn += 1;
        Ok(bundle)
    }
}

// ---------------------------------------------------------------------------
// Phase 20 pre-game lobby hooks (frames + lobby → lockstep handover; see net_lobby.rs)
// ---------------------------------------------------------------------------

pub(crate) fn lobby_write_frame(stream: &mut TcpStream, payload: &[u8]) -> Result<(), String> {
    write_frame(stream, payload)
}

pub(crate) fn lobby_read_frame(stream: &mut TcpStream) -> Result<Vec<u8>, String> {
    read_frame(stream)
}

/// Decodes a frame if it is a lockstep [`Handshake`] (`None` for any other message).
pub(crate) fn lobby_try_handshake(data: &[u8]) -> Option<Result<Handshake, String>> {
    (data.first() == Some(&MSG_HANDSHAKE)).then(|| decode_handshake(data))
}

impl LockstepHost {
    /// Hands lobby connections over to lockstep: sends every peer its [`Handshake`]
    /// (`assigned_army` filled in) on the already-open stream.
    pub fn from_lobby(peers: Vec<(ArmyId, TcpStream)>, hs: &Handshake) -> Result<Self, String> {
        let n = hs.player_count;
        if !(2..=8).contains(&n) || peers.len() + 1 != n {
            return Err(format!("lobby start with {} peers for {n} players", peers.len()));
        }
        let mut clients = Vec::with_capacity(peers.len());
        for (army, mut stream) in peers {
            let _ = stream.set_nodelay(true);
            let _ = stream.set_read_timeout(Some(NET_TIMEOUT));
            let mut client_hs = hs.clone();
            client_hs.assigned_army = army;
            write_frame(&mut stream, &encode_handshake(&client_hs)).map_err(|e| format!("army {army}: {e}"))?;
            clients.push((army, PeerStream::Tcp(stream)));
        }
        clients.sort_by_key(|(a, _)| *a);
        Ok(Self {
            clients,
            local_army: 0,
            next_turn: 0,
            adaptive: AdaptiveDelayController::new(false, hs.input_delay, hs.turn_ticks),
        })
    }
}

impl LockstepClient {
    /// Lockstep client on a lobby connection whose [`Handshake`] was already received.
    pub fn from_lobby(stream: TcpStream, hs: &Handshake) -> Self {
        let _ = stream.set_nodelay(true);
        let _ = stream.set_read_timeout(Some(NET_TIMEOUT));
        Self {
            stream: PeerStream::Tcp(stream),
            local_army: hs.assigned_army,
            next_turn: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Transport-independent lockstep core (turn scheduling + desync verification)
// ---------------------------------------------------------------------------

struct SentTurn {
    turn: u64,
    hash: u64,
    tick: u64,
    summary: String,
}

/// Desync diagnostics: which turn, at which tick, which armies disagreed.
#[derive(Debug, Clone)]
pub struct DesyncReport {
    pub turn: u64,
    pub tick: u64,
    pub local_army: ArmyId,
    pub local_hash: u64,
    pub mismatches: Vec<(ArmyId, u64)>,
    /// Every hash carried by the offending bundle.
    pub peer_hashes: Vec<(ArmyId, u64)>,
    pub summary: String,
    pub recent: Vec<String>,
}

impl fmt::Display for DesyncReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LOCKSTEP DESYNC at turn {} (tick {}): local army {}={:016x}",
            self.turn, self.tick, self.local_army, self.local_hash
        )?;
        for (a, h) in &self.mismatches {
            write!(f, " vs army {a}={h:016x}")?;
        }
        write!(f, "\n  local state: {}", self.summary)?;
        if self.recent.is_empty() {
            write!(f, "\n  last commands: (none)")?;
        }
        for r in &self.recent {
            write!(f, "\n  last commands: {r}")?;
        }
        Ok(())
    }
}

/// Tick outcome of [`LockstepCore::tick`].
#[derive(Debug)]
pub enum TickOutcome {
    /// The next turn's bundle has not arrived; the sim did not advance.
    Stalled,
    /// One tick ran; contains 0, 1, or more turn packets when a turn completed (0 when shrinking
    /// adaptive `input_delay`, >1 when growing `input_delay`).
    Stepped(Vec<TurnPacket>),
    /// A synchronized pause is active: the turn advanced (null turn) but the simulation did
    /// not step; emitted packets follow the same rules as [`TickOutcome::Stepped`].
    Paused(Vec<TurnPacket>),
}

/// Session events for the UI ([`LockstepCore::drain_events`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetEvent {
    Chat { from: ArmyId, allies: bool, text: String },
    Paused { by: ArmyId, paused: bool, pauses_left: u8 },
    PauseDenied { by: ArmyId },
    Speed { by: ArmyId, speed: u32 },
    SpeedDenied { by: ArmyId },
    Dropped { army: ArmyId, tick: u64 },
    InputDelay { turn: u64, delay: u64 },
}

/// A pause/speed/drop control as applied at a turn start — identical on every peer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedControl {
    pub turn: u64,
    pub tick: u64,
    pub from: ArmyId,
    pub control: NetControl,
    pub accepted: bool,
}

pub struct LockstepCore {
    pub local_army: ArmyId,
    pub player_count: usize,
    pub turn_ticks: u64,
    pub input_delay: u64,
    next_submit: u64,
    next_recv: u64,
    next_exec: u64,
    tick_in_turn: u64,
    ready: VecDeque<TurnBundle>,
    sent: VecDeque<SentTurn>,
    history: VecDeque<TurnBundle>,
    outbox: Vec<Command>,
    shared_ais: Vec<ScriptedAi>,
    // Phase 21 match session state (all of it derived from the bundle stream, so equal on
    // every peer).
    outbox_controls: Vec<NetControl>,
    teams: Vec<u8>,
    dropped: Vec<ArmyId>,
    paused: bool,
    turn_paused: bool,
    speed: u32,
    pauses_used: BTreeMap<ArmyId, u8>,
    events: Vec<NetEvent>,
    applied: Vec<AppliedControl>,
    takeovers: Vec<(u64, ArmyId)>,
    /// Synchronized `(turn, new_input_delay)` transitions applied at turn boundaries.
    delay_changes: Vec<(u64, u64)>,
    /// Every command applied to the local world `(tick, army, command)` (all sources).
    recorded: Vec<(u64, ArmyId, Command)>,
    checkpoints: Vec<(u64, u64)>,
    /// Map id used for desync-dump replays.
    pub map_id: String,
    /// Desync dump base directory (default `savegame::data_dir("desyncs")`).
    pub dump_dir: Option<PathBuf>,
}

impl LockstepCore {
    pub fn new(
        world: &Skirmish,
        local_army: ArmyId,
        player_count: usize,
        turn_ticks: u64,
        input_delay: u64,
    ) -> Self {
        let shared_ais = world
            .armies()
            .iter()
            .filter(|a| a.id as usize >= player_count)
            .map(|a| ScriptedAi::new(a.id))
            .collect();
        Self {
            local_army,
            player_count,
            turn_ticks: turn_ticks.max(1),
            input_delay: input_delay.max(1),
            next_submit: 0,
            next_recv: 0,
            next_exec: 0,
            tick_in_turn: 0,
            ready: VecDeque::new(),
            sent: VecDeque::new(),
            history: VecDeque::new(),
            outbox: Vec::new(),
            shared_ais,
            outbox_controls: Vec::new(),
            teams: world.armies().iter().map(|a| a.team).collect(),
            dropped: Vec::new(),
            paused: false,
            turn_paused: false,
            speed: 1,
            pauses_used: BTreeMap::new(),
            events: Vec::new(),
            applied: Vec::new(),
            takeovers: Vec::new(),
            delay_changes: Vec::new(),
            recorded: Vec::new(),
            checkpoints: Vec::new(),
            map_id: String::from("UNKNOWN"),
            dump_dir: None,
        }
    }

    pub fn from_handshake(world: &Skirmish, hs: &Handshake) -> Self {
        let mut core = Self::new(world, hs.assigned_army, hs.player_count, hs.turn_ticks, hs.input_delay);
        core.map_id = hs.map_id.clone();
        core
    }

    /// AI-controlled army ids simulated identically on every peer.
    pub fn shared_ai_armies(&self) -> Vec<ArmyId> {
        self.shared_ais.iter().map(|a| a.army).collect()
    }

    fn make_packet(&mut self, world: &Skirmish) -> TurnPacket {
        let hash = world.state_hash();
        let turn = self.next_submit;
        self.next_submit += 1;
        self.sent.push_back(SentTurn {
            turn,
            hash,
            tick: world.tick(),
            summary: world.desync_summary(),
        });
        TurnPacket {
            turn,
            hash,
            commands: std::mem::take(&mut self.outbox),
            controls: std::mem::take(&mut self.outbox_controls),
        }
    }

    /// The first `input_delay` (empty) packets, all hashed at the initial state.
    pub fn prime(&mut self, world: &Skirmish) -> Vec<TurnPacket> {
        (0..self.input_delay).map(|_| self.make_packet(world)).collect()
    }

    /// Buffers locally issued commands for the next submitted turn.
    pub fn queue_local(&mut self, cmds: impl IntoIterator<Item = Command>) {
        self.outbox.extend(cmds);
    }

    /// Buffers a session control (pause / speed / chat) for the next submitted turn.
    pub fn queue_control(&mut self, c: NetControl) {
        if c.peer_may_send() && self.outbox_controls.len() < 64 {
            self.outbox_controls.push(c);
        }
    }

    /// Bundles received but not yet executed.
    pub fn buffered_turns(&self) -> usize {
        self.ready.len()
    }

    /// Turns submitted whose bundle has not arrived yet.
    pub fn in_flight_turns(&self) -> u64 {
        self.next_submit - self.next_recv
    }

    pub fn exec_turn(&self) -> u64 {
        self.next_exec
    }

    /// Next turn number this peer will submit.
    #[allow(dead_code)]
    pub fn submitted_turns(&self) -> u64 {
        self.next_submit
    }

    /// Next bundle turn this peer expects.
    pub fn received_turns(&self) -> u64 {
        self.next_recv
    }

    /// Synchronized pause state (as of the executing turn).
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Synchronized game speed (host controlled).
    #[allow(dead_code)]
    pub fn speed(&self) -> u32 {
        self.speed
    }

    pub fn pauses_left(&self, army: ArmyId) -> u8 {
        MAX_PAUSES_PER_PLAYER.saturating_sub(self.pauses_used.get(&army).copied().unwrap_or(0))
    }

    /// Armies announced as disconnected (receive order).
    #[allow(dead_code)]
    pub fn dropped_armies(&self) -> &[ArmyId] {
        &self.dropped
    }

    /// `(tick, army)` AI takeovers after disconnects.
    #[allow(dead_code)]
    pub fn takeovers(&self) -> &[(u64, ArmyId)] {
        &self.takeovers
    }

    /// Synchronized `(turn, new_input_delay)` transitions applied identically on all peers.
    #[allow(dead_code)]
    pub fn delay_changes(&self) -> &[(u64, u64)] {
        &self.delay_changes
    }

    /// Applied pause/speed/drop controls in execution order.
    #[allow(dead_code)]
    pub fn applied_controls(&self) -> &[AppliedControl] {
        &self.applied
    }

    /// Every command applied to the local world so far.
    #[allow(dead_code)]
    pub fn recorded_commands(&self) -> &[(u64, ArmyId, Command)] {
        &self.recorded
    }

    pub fn drain_events(&mut self) -> Vec<NetEvent> {
        std::mem::take(&mut self.events)
    }

    fn allied(&self, a: ArmyId, b: ArmyId) -> bool {
        a == b
            || matches!(
                (self.teams.get(a as usize), self.teams.get(b as usize)),
                (Some(x), Some(y)) if x == y
            )
    }

    /// Verifies a received bundle against the hash this peer submitted for that turn.
    pub fn receive(&mut self, bundle: TurnBundle) -> Result<(), DesyncReport> {
        // Disconnect notices first: a dropped army's hash is absent from this very bundle.
        for (from, c) in &bundle.controls {
            if let (SYSTEM_SENDER, NetControl::Dropped(a)) = (*from, c)
                && !self.dropped.contains(a)
            {
                self.dropped.push(*a);
            }
        }
        let expected = self.next_recv;
        let sent_idx = self.sent.iter().position(|s| s.turn == bundle.turn);
        let (local_hash, tick, summary) = sent_idx
            .map(|i| (self.sent[i].hash, self.sent[i].tick, self.sent[i].summary.clone()))
            .unwrap_or((0, 0, String::new()));
        let mut mismatches: Vec<(ArmyId, u64)> = bundle
            .hashes
            .iter()
            .filter(|(a, h)| *a != self.local_army && *h != local_hash)
            .copied()
            .collect();
        let own_ok = bundle
            .hashes
            .iter()
            .all(|(a, h)| *a != self.local_army || *h == local_hash);
        let present = bundle.hashes.len() + self.dropped.len() >= self.player_count;
        if bundle.turn != expected || sent_idx.is_none() || !own_ok || !present {
            mismatches.push((255, bundle.turn));
            let summary = format!(
                "protocol error: expected bundle turn {expected}, got {} with {} hashes{}",
                bundle.turn,
                bundle.hashes.len(),
                if own_ok { "" } else { " (own hash altered)" }
            );
            return Err(self.report(bundle.turn, tick, local_hash, mismatches, bundle.hashes.clone(), summary));
        }
        if !mismatches.is_empty() {
            return Err(self.report(bundle.turn, tick, local_hash, mismatches, bundle.hashes.clone(), summary));
        }
        // Drop hash records that are verified.
        while self.sent.front().is_some_and(|s| s.turn <= bundle.turn) {
            self.sent.pop_front();
        }
        // Chat is UI-only: surface it on arrival (allies-only lines for allies only).
        for (from, c) in &bundle.controls {
            if let NetControl::Chat { allies, text } = c
                && (*from as usize) < self.teams.len()
                && (!*allies || self.allied(*from, self.local_army))
            {
                self.events.push(NetEvent::Chat {
                    from: *from,
                    allies: *allies,
                    text: text.clone(),
                });
            }
        }
        self.next_recv += 1;
        self.ready.push_back(bundle);
        Ok(())
    }

    fn report(
        &self,
        turn: u64,
        tick: u64,
        local_hash: u64,
        mismatches: Vec<(ArmyId, u64)>,
        peer_hashes: Vec<(ArmyId, u64)>,
        summary: String,
    ) -> DesyncReport {
        let recent = self
            .history
            .iter()
            .rev()
            .map(|b| {
                let cmds: Vec<String> = b
                    .batches
                    .iter()
                    .map(|(a, c)| format!("A{a}:{c:?}"))
                    .collect();
                let mut line = format!("turn {}: ", b.turn);
                let joined = if cmds.is_empty() { "-".to_string() } else { cmds.join(" ; ") };
                line.push_str(&joined.chars().take(400).collect::<String>());
                line
            })
            .collect();
        DesyncReport {
            turn,
            tick,
            local_army: self.local_army,
            local_hash,
            mismatches,
            peer_hashes,
            summary,
            recent,
        }
    }

    /// Applies a bundle's pause / speed / disconnect controls at its turn start.
    fn apply_controls(&mut self, bundle: &TurnBundle, world: &Skirmish) {
        let tick = world.tick();
        for (from, c) in &bundle.controls {
            let from = *from;
            let accepted = match c {
                NetControl::Pause(true) => {
                    if self.paused {
                        continue;
                    }
                    let used = self.pauses_used.entry(from).or_insert(0);
                    if (from as usize) < self.player_count && *used < MAX_PAUSES_PER_PLAYER {
                        *used += 1;
                        self.paused = true;
                        let pauses_left = MAX_PAUSES_PER_PLAYER - *used;
                        self.events.push(NetEvent::Paused { by: from, paused: true, pauses_left });
                        true
                    } else {
                        self.events.push(NetEvent::PauseDenied { by: from });
                        false
                    }
                }
                NetControl::Pause(false) => {
                    if !self.paused {
                        continue;
                    }
                    self.paused = false;
                    let pauses_left = self.pauses_left(from);
                    self.events.push(NetEvent::Paused { by: from, paused: false, pauses_left });
                    true
                }
                NetControl::Speed(s) => {
                    if from == HOST_ARMY {
                        self.speed = (*s).clamp(1, 8);
                        self.events.push(NetEvent::Speed { by: from, speed: self.speed });
                        true
                    } else {
                        self.events.push(NetEvent::SpeedDenied { by: from });
                        false
                    }
                }
                NetControl::Dropped(a) if from == SYSTEM_SENDER => {
                    let a = *a;
                    let fresh = world.army(a).is_some() && !self.shared_ais.iter().any(|x| x.army == a);
                    if fresh {
                        self.shared_ais.push(ScriptedAi::new(a));
                        self.shared_ais.sort_by_key(|x| x.army);
                        self.takeovers.push((tick, a));
                    }
                    self.events.push(NetEvent::Dropped { army: a, tick });
                    fresh
                }
                NetControl::InputDelay(d) if from == SYSTEM_SENDER => {
                    let clamped = u64::from(*d).clamp(MIN_INPUT_DELAY, MAX_INPUT_DELAY);
                    if clamped != self.input_delay {
                        self.input_delay = clamped;
                        self.delay_changes.push((bundle.turn, clamped));
                        self.events.push(NetEvent::InputDelay {
                            turn: bundle.turn,
                            delay: clamped,
                        });
                    }
                    continue;
                }
                _ => continue,
            };
            self.applied.push(AppliedControl {
                turn: bundle.turn,
                tick,
                from,
                control: c.clone(),
                accepted,
            });
        }
    }

    /// Advances one sim tick if the current turn's bundle is available. At a turn start
    /// the bundle's controls are applied and its commands issued; shared AI slots think every
    /// tick; when a turn completes the next local packet (turn `exec + input_delay`) is
    /// returned. While a synchronized pause is active turns are *null turns*: they advance at
    /// the same pace but the world does not step.
    pub fn tick(&mut self, world: &mut Skirmish) -> TickOutcome {
        if self.tick_in_turn == 0 {
            let Some(bundle) = self.ready.pop_front() else {
                return TickOutcome::Stalled;
            };
            debug_assert_eq!(bundle.turn, self.next_exec);
            self.apply_controls(&bundle, world);
            issue_bundle(world, &bundle);
            self.turn_paused = self.paused;
            self.history.push_back(bundle);
            while self.history.len() > HISTORY_TURNS {
                self.history.pop_front();
            }
        }
        if !self.turn_paused && world.result().is_none() {
            for ai in &mut self.shared_ais {
                ai.think(world);
            }
            let t = world.tick();
            self.recorded
                .extend(world.pending_view().iter().map(|(a, c)| (t, *a, c.clone())));
            world.step();
            let t = world.tick();
            if t % crate::replay::CHECKPOINT_INTERVAL == 0 && self.checkpoints.last().is_none_or(|c| c.0 < t) {
                self.checkpoints.push((t, world.state_hash()));
            }
        }
        self.tick_in_turn += 1;
        let pkts = if self.tick_in_turn >= self.turn_ticks {
            self.tick_in_turn = 0;
            self.next_exec += 1;
            let target_submit = self.next_exec + self.input_delay;
            let mut out = Vec::new();
            while self.next_submit < target_submit {
                out.push(self.make_packet(world));
            }
            out
        } else {
            Vec::new()
        };
        if self.turn_paused {
            TickOutcome::Paused(pkts)
        } else {
            TickOutcome::Stepped(pkts)
        }
    }

    // ----- post-desync diagnostics exchange ----------------------------------------------

    /// Starts the desync exchange: our state summary goes out with the next packet and peers
    /// keep exchanging null turns up to (and including) the returned turn.
    pub fn begin_desync_exchange(&mut self, report: &DesyncReport) -> u64 {
        self.outbox.clear();
        self.outbox_controls.clear();
        self.outbox_controls.push(NetControl::DesyncSummary {
            turn: report.turn,
            tick: report.tick,
            hash: report.local_hash,
            text: report.summary.clone(),
        });
        self.next_recv = self.next_recv.max(report.turn + 1);
        report.turn + self.input_delay + DESYNC_EXTRA_TURNS
    }

    /// Null packet for the next turn during the desync exchange (`None` once `until` is sent).
    pub fn halted_packet(&mut self, world: &Skirmish, until: u64) -> Option<TurnPacket> {
        if self.next_submit > until {
            return None;
        }
        let turn = self.next_submit;
        self.next_submit += 1;
        Some(TurnPacket {
            turn,
            hash: world.state_hash(),
            commands: Vec::new(),
            controls: std::mem::take(&mut self.outbox_controls),
        })
    }

    /// Accepts a bundle without verification (after a desync); returns its summaries.
    pub fn receive_unverified(&mut self, bundle: &TurnBundle) -> Vec<(ArmyId, String)> {
        self.next_recv = self.next_recv.max(bundle.turn + 1);
        bundle
            .controls
            .iter()
            .filter_map(|(a, c)| match c {
                NetControl::DesyncSummary { text, .. } => Some((*a, text.clone())),
                _ => None,
            })
            .collect()
    }
}

/// Writes `desyncs/<timestamp>-army<N>/`: `commands.farust-replay` (every command this peer
/// executed, verifiable with `farust replay verify`), `report.txt` (report, all hashes of the
/// offending turn, per-peer summaries) and one `peer-army<N>.txt` per summary received.
pub fn write_desync_dump(
    core: &LockstepCore,
    world: &Skirmish,
    report: &DesyncReport,
    summaries: &BTreeMap<ArmyId, String>,
) -> Result<PathBuf, String> {
    let base = core
        .dump_dir
        .clone()
        .unwrap_or_else(|| crate::savegame::data_dir("desyncs"));
    let dir = unique_dir(&base, &format!("{}-army{}", crate::savegame::utc_stamp(), core.local_army));
    std::fs::create_dir_all(&dir).map_err(|e| format!("failed to create {}: {e}", dir.display()))?;
    let mut rec = crate::replay::ReplayRecorder::new(&core.map_id, world.seed(), crate::savegame::setups_of(world));
    for (tick, army, cmd) in core.recorded.iter().filter(|c| c.0 < world.tick()) {
        rec.record(*tick, *army, cmd);
    }
    for &(tick, hash) in &core.checkpoints {
        rec.maybe_checkpoint(tick, hash);
    }
    rec.finish(world.tick(), world.state_hash())
        .save(&dir.join("commands.farust-replay"))?;
    let mut txt = format!("{report}\n\nhashes of turn {} (tick {}):\n", report.turn, report.tick);
    for (a, h) in &report.peer_hashes {
        let tag = if *h == report.local_hash { "" } else { "   <-- differs from local" };
        txt.push_str(&format!("  army {a}: {h:016x}{tag}\n"));
    }
    txt.push_str(&format!(
        "\nlocal army {} world at dump: tick {} hash {:016x}\n\nper-peer state summaries at the desync turn:\n",
        core.local_army,
        world.tick(),
        world.state_hash()
    ));
    for a in 0..core.player_count as ArmyId {
        match summaries.get(&a) {
            Some(s) => {
                txt.push_str(&format!("  army {a}: {s}\n"));
                std::fs::write(dir.join(format!("peer-army{a}.txt")), format!("{s}\n"))
                    .map_err(|e| format!("failed to write peer summary: {e}"))?;
            }
            None if core.dropped.contains(&a) => txt.push_str(&format!("  army {a}: (disconnected)\n")),
            None => txt.push_str(&format!("  army {a}: (no summary received)\n")),
        }
    }
    std::fs::write(dir.join("report.txt"), txt).map_err(|e| format!("failed to write report: {e}"))?;
    Ok(dir)
}

fn unique_dir(base: &Path, name: &str) -> PathBuf {
    let mut dir = base.join(name);
    let mut i = 1;
    while dir.exists() {
        i += 1;
        dir = base.join(format!("{name}-{i}"));
    }
    dir
}

// ---------------------------------------------------------------------------
// Headless driver (AI on every slot)
// ---------------------------------------------------------------------------

/// Runs a complete headless lockstep match: the local army is driven by [`ScriptedAi`]
/// whose commands go over the wire; AI slots run inside the core. Returns the final hash.
pub fn run_headless(
    transport: &mut dyn TurnTransport,
    core: &mut LockstepCore,
    world: &mut Skirmish,
    ticks: u64,
) -> Result<u64, String> {
    run_headless_with(transport, core, world, ticks, &mut |_, _| true)
}

/// [`run_headless`] with a per-iteration `hook` (queue controls, inspect state); returning
/// `false` leaves the match immediately (the transport is dropped by the caller =
/// disconnect). On a desync the peer runs the summary exchange, writes the dump and errors.
pub fn run_headless_with(
    transport: &mut dyn TurnTransport,
    core: &mut LockstepCore,
    world: &mut Skirmish,
    ticks: u64,
    hook: &mut dyn FnMut(&mut LockstepCore, &Skirmish) -> bool,
) -> Result<u64, String> {
    // Round the limit up to a turn boundary so every peer stops after the same exchange.
    let limit = ticks.div_ceil(core.turn_ticks) * core.turn_ticks;
    let mut primed: VecDeque<TurnPacket> = core.prime(world).into();
    while let Some(pkt) = primed.pop_front() {
        let b = transport.exchange(pkt)?;
        receive_or_dump(transport, core, world, b, &mut primed)?;
    }
    let mut ai = ScriptedAi::new(core.local_army);
    let mut ai_tick = None;
    while world.tick() < limit && world.result().is_none() {
        if !hook(core, world) {
            return Ok(world.state_hash());
        }
        // Decide once per sim tick (paused null turns do not advance it).
        if ai_tick != Some(world.tick()) {
            ai_tick = Some(world.tick());
            let cmds = ai.decide(world);
            core.queue_local(cmds);
        }
        match core.tick(world) {
            TickOutcome::Stalled => {
                return Err(format!("headless lockstep stalled at tick {}", world.tick()));
            }
            TickOutcome::Stepped(pkts) | TickOutcome::Paused(pkts) => {
                if !pkts.is_empty() {
                    if world.tick() >= limit || world.result().is_some() {
                        break;
                    }
                    let mut pending: VecDeque<TurnPacket> = pkts.into();
                    while let Some(pkt) = pending.pop_front() {
                        let b = transport.exchange(pkt)?;
                        receive_or_dump(transport, core, world, b, &mut pending)?;
                    }
                }
            }
        }
    }
    Ok(world.state_hash())
}

/// Verifies `bundle`; on a desync exchanges summaries (after sending any still `unsent`
/// primed packets, keeping turn order), writes the dump and returns the error.
fn receive_or_dump(
    transport: &mut dyn TurnTransport,
    core: &mut LockstepCore,
    world: &Skirmish,
    bundle: TurnBundle,
    unsent: &mut VecDeque<TurnPacket>,
) -> Result<(), String> {
    let Err(report) = core.receive(bundle) else {
        return Ok(());
    };
    let until = core.begin_desync_exchange(&report);
    let mut summaries = BTreeMap::new();
    summaries.insert(core.local_army, report.summary.clone());
    while let Some(pkt) = unsent.pop_front().or_else(|| core.halted_packet(world, until)) {
        match transport.exchange(pkt) {
            Ok(b) => summaries.extend(core.receive_unverified(&b)),
            Err(_) => break,
        }
    }
    Err(match write_desync_dump(core, world, &report, &summaries) {
        Ok(dir) => format!("{report}\n  desync dump: {}", dir.display()),
        Err(e) => format!("{report}\n  desync dump failed: {e}"),
    })
}

// ---------------------------------------------------------------------------
// Non-blocking GUI bridge
// ---------------------------------------------------------------------------

pub struct AsyncLockstepBridge {
    tx: Option<Sender<TurnPacket>>,
    rx: Receiver<Result<(TurnBundle, Duration), String>>,
    worker: Option<JoinHandle<()>>,
}

impl AsyncLockstepBridge {
    /// Moves `transport` onto a worker thread; submitted packets are exchanged in order and
    /// each resulting bundle comes back with its submit→receive round-trip time.
    pub fn spawn(mut transport: Box<dyn TurnTransport>) -> Self {
        let (req_tx, req_rx) = mpsc::channel::<TurnPacket>();
        let (resp_tx, resp_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let mut submitted: VecDeque<Instant> = VecDeque::new();
            while let Ok(pkt) = req_rx.recv() {
                submitted.push_back(Instant::now());
                let res = transport.exchange(pkt);
                let sent_at = submitted.pop_front().unwrap_or_else(Instant::now);
                let err = res.is_err();
                let msg = res.map(|b| (b, sent_at.elapsed()));
                if resp_tx.send(msg).is_err() || err {
                    break;
                }
            }
        });
        Self {
            tx: Some(req_tx),
            rx: resp_rx,
            worker: Some(worker),
        }
    }

    pub fn submit(&self, pkt: TurnPacket) -> Result<(), String> {
        self.tx
            .as_ref()
            .ok_or_else(|| "lockstep bridge closed".to_string())?
            .send(pkt)
            .map_err(|_| "lockstep worker exited".to_string())
    }

    pub fn try_recv(&self) -> Option<Result<(TurnBundle, Duration), String>> {
        self.rx.try_recv().ok()
    }

    #[cfg(test)]
    pub fn recv_timeout(&self, timeout: Duration) -> Option<Result<(TurnBundle, Duration), String>> {
        self.rx.recv_timeout(timeout).ok()
    }
}

impl Drop for AsyncLockstepBridge {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(worker) = self.worker.take() {
            // The worker may be blocked in a socket read; never hang the UI on exit.
            if worker.is_finished() {
                let _ = worker.join();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Interactive play session hook
// ---------------------------------------------------------------------------

/// Post-desync state of an interactive peer (summary exchange, then dump).
struct DesyncState {
    report: DesyncReport,
    until: u64,
    summaries: BTreeMap<ArmyId, String>,
    started: Instant,
    dump: Option<PathBuf>,
    closed: bool,
}

pub struct NetPlay {
    core: LockstepCore,
    bridge: AsyncLockstepBridge,
    failure: Option<String>,
    rtt_ms: f64,
    stall_ticks: u64,
    stalled_now: bool,
    desync: Option<DesyncState>,
    /// Prefix of `core.recorded` already copied into the session recorder.
    recorded_seen: usize,
    /// Sim tick the `F1` assist AI last decided on (once per tick, not per stalled call).
    assist_tick: Option<u64>,
}

/// Shared handle stored on [`PlaySession::net`] (Clone + Debug like the session).
#[derive(Clone)]
pub struct NetHandle(Arc<Mutex<NetPlay>>);

impl fmt::Debug for NetHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NetHandle")
    }
}

fn army_label(world: &Skirmish, army: ArmyId) -> String {
    world
        .army(army)
        .map(|a| a.name.clone())
        .unwrap_or_else(|| format!("ARMY {army}"))
}

impl NetHandle {
    /// Primes the first `input_delay` turns and starts the worker.
    pub fn start(world: &Skirmish, hs: &Handshake, transport: Box<dyn TurnTransport>) -> Result<Self, String> {
        let mut core = LockstepCore::from_handshake(world, hs);
        let bridge = AsyncLockstepBridge::spawn(transport);
        for pkt in core.prime(world) {
            bridge.submit(pkt)?;
        }
        Ok(Self(Arc::new(Mutex::new(NetPlay {
            core,
            bridge,
            failure: None,
            rtt_ms: 0.0,
            stall_ticks: 0,
            stalled_now: false,
            desync: None,
            recorded_seen: 0,
            assist_tick: None,
        }))))
    }

    /// Turns a fresh single-player session into a networked one for `hs.assigned_army`.
    pub fn attach(self, session: &mut PlaySession) {
        let (army, shared) = self
            .0
            .lock()
            .map(|n| (n.core.local_army, n.core.shared_ai_armies()))
            .unwrap_or((0, Vec::new()));
        session.human_army = army;
        session.spectate = false;
        session.ais.clear();
        session.speed = 1;
        session.paused = false;
        session.selected.clear();
        if let Some(acu) = session.world.commander_of(army) {
            session.selected.push(acu.id);
        }
        session.status_banner = format!("LOCKSTEP: ARMY {army}");
        // Everything the lockstep core applies is recorded (saves, auto replay); AI slots
        // are logged so a loaded save resumes them exactly.
        let mut rec = crate::savegame::MatchRecorder {
            from_net: true,
            ..Default::default()
        };
        for a in shared {
            rec.ai_event(0, a, true);
        }
        session.recorder = Some(rec);
        session.net = Some(self);
    }

    /// Replaces `PlaySession::step_sim` while networked: routes locally issued commands
    /// into the next turn packet and only advances when the turn bundle is present.
    pub fn step_session(&self, s: &mut PlaySession) {
        let Ok(mut n) = self.0.lock() else {
            return;
        };
        let n = &mut *n;
        if n.desync.is_some() {
            pump_desync(n, s);
            return;
        }
        if n.failure.is_some() {
            return;
        }
        while let Some(msg) = n.bridge.try_recv() {
            match msg {
                Ok((bundle, rtt)) => {
                    let ms = rtt.as_secs_f64() * 1000.0;
                    n.rtt_ms = if n.rtt_ms == 0.0 { ms } else { n.rtt_ms * 0.85 + ms * 0.15 };
                    if let Err(d) = n.core.receive(bundle) {
                        let text = d.to_string();
                        eprintln!("{text}");
                        n.failure = Some(format!("DESYNC T{} TICK {}", d.turn, d.tick));
                        s.status_banner = "DESYNC - SIMULATION HALTED".to_string();
                        s.chat.system("DESYNC DETECTED - COLLECTING PEER STATE FOR THE DUMP");
                        let until = n.core.begin_desync_exchange(&d);
                        let mut summaries = BTreeMap::new();
                        summaries.insert(n.core.local_army, d.summary.clone());
                        n.desync = Some(DesyncState {
                            report: d,
                            until,
                            summaries,
                            started: Instant::now(),
                            dump: None,
                            closed: false,
                        });
                        pump_desync(n, s);
                        return;
                    }
                }
                Err(e) => {
                    eprintln!("lockstep network error: {e}");
                    n.failure = Some(format!("NET ERROR: {e}"));
                    s.status_banner = "NETWORK ERROR - SIMULATION HALTED".to_string();
                    s.chat.system("CONNECTION TO THE HOST LOST - MATCH ENDED");
                    return;
                }
            }
        }
        flush_events(n, s);
        for (allies, text) in s.chat.outbox.drain(..) {
            n.core.queue_control(NetControl::Chat { allies, text });
        }
        // Everything in `pending` was issued by local input since the last tick.
        let local = n.core.local_army;
        let drained = s.world.drain_pending();
        n.core
            .queue_local(drained.into_iter().filter(|(a, _)| *a == local).map(|(_, c)| c));
        // F1 assist AI for the local army also goes over the wire.
        if n.assist_tick != Some(s.world.tick()) {
            n.assist_tick = Some(s.world.tick());
            for ai in &mut s.ais {
                if ai.army == local {
                    let cmds = ai.decide(&s.world);
                    n.core.queue_local(cmds);
                }
            }
        }
        let before = s.world.tick();
        match n.core.tick(&mut s.world) {
            TickOutcome::Stalled => {
                n.stalled_now = true;
                n.stall_ticks += 1;
            }
            TickOutcome::Stepped(pkts) | TickOutcome::Paused(pkts) => {
                n.stalled_now = false;
                s.selected.retain(|id| s.world.unit(*id).is_some());
                for pkt in pkts {
                    if pkt.turn % 100 == 0 && std::env::var_os("FARUST_NET_LOG").is_some() {
                        println!(
                            "[net] army {} tick {} hash {:016x} | {}",
                            local,
                            s.world.tick(),
                            pkt.hash,
                            format_status(n)
                        );
                    }
                    if let Err(e) = n.bridge.submit(pkt) {
                        n.failure = Some(format!("NET ERROR: {e}"));
                        break;
                    }
                }
            }
        }
        if s.world.tick() != before {
            // Record what the core applied this step (bundles + shared AI) and sample stats.
            let new = &n.core.recorded[n.recorded_seen..];
            let mut counts = Vec::new();
            for (_, a, _) in new {
                crate::savegame::bump(&mut counts, *a);
            }
            if let Some(rec) = s.recorder.as_mut() {
                rec.commands.extend(new.iter().cloned());
                rec.after_step(&s.world);
            }
            n.recorded_seen = n.core.recorded.len();
            s.stats.observe(&s.world, &counts);
        }
        flush_events(n, s);
    }

    /// One-line HUD status: ping, buffered turns, stall time.
    pub fn hud_line(&self) -> String {
        self.0.lock().map(|n| format_status(&n)).unwrap_or_default()
    }

    /// Red banner text after a desync or a network failure.
    pub fn failure_banner(&self) -> Option<String> {
        self.0.lock().ok().and_then(|n| n.failure.clone())
    }

    #[allow(dead_code)]
    pub fn local_army(&self) -> ArmyId {
        self.0.lock().map(|n| n.core.local_army).unwrap_or(0)
    }

    #[allow(dead_code)]
    pub fn is_host(&self) -> bool {
        self.local_army() == HOST_ARMY
    }

    /// Synchronized pause currently in effect.
    pub fn is_paused(&self) -> bool {
        self.0.lock().is_ok_and(|n| n.core.is_paused())
    }

    /// Pauses the local player may still request.
    #[allow(dead_code)]
    pub fn pauses_left(&self) -> u8 {
        self.0
            .lock()
            .map(|n| n.core.pauses_left(n.core.local_army))
            .unwrap_or(0)
    }

    /// Requests a synchronized pause / resume (applied by every peer at the same turn).
    pub fn request_pause(&self, paused: bool) -> Result<(), String> {
        let Ok(mut n) = self.0.lock() else {
            return Err("network session unavailable".to_string());
        };
        if paused && !n.core.is_paused() && n.core.pauses_left(n.core.local_army) == 0 {
            return Err(format!("no pauses left ({MAX_PAUSES_PER_PLAYER} per match)"));
        }
        n.core.queue_control(NetControl::Pause(paused));
        Ok(())
    }

    /// Requests a synchronized speed change (host only).
    pub fn request_speed(&self, speed: u32) -> Result<(), String> {
        let Ok(mut n) = self.0.lock() else {
            return Err("network session unavailable".to_string());
        };
        if n.core.local_army != HOST_ARMY {
            return Err("only the host can change the game speed".to_string());
        }
        n.core.queue_control(NetControl::Speed(speed.clamp(1, 8)));
        Ok(())
    }

    /// Path of the desync dump once written.
    #[allow(dead_code)]
    pub fn desync_dump(&self) -> Option<PathBuf> {
        self.0
            .lock()
            .ok()
            .and_then(|n| n.desync.as_ref().and_then(|d| d.dump.clone()))
            .filter(|p| !p.as_os_str().is_empty())
    }
}

/// Turns core events into chat-log lines and session state (pause mirror, speed, AI log).
fn flush_events(n: &mut NetPlay, s: &mut PlaySession) {
    for ev in n.core.drain_events() {
        match ev {
            NetEvent::Chat { from, allies, text } => {
                let name = army_label(&s.world, from);
                s.chat.push(Some(from), &name, &text, allies);
            }
            NetEvent::Paused { by, paused, pauses_left } => {
                s.paused = paused;
                let who = army_label(&s.world, by);
                let line = if paused {
                    format!("{who} PAUSED THE GAME ({pauses_left} PAUSES LEFT) - SPACE TO RESUME")
                } else {
                    format!("{who} RESUMED THE GAME")
                };
                s.status_banner = line.clone();
                s.chat.system(&line);
            }
            NetEvent::PauseDenied { by } => {
                let who = army_label(&s.world, by);
                s.chat.system(&format!("{who} HAS NO PAUSES LEFT"));
            }
            NetEvent::Speed { by: _, speed } => {
                s.speed = speed;
                s.chat.system(&format!("GAME SPEED SET TO {speed}X BY THE HOST"));
            }
            NetEvent::SpeedDenied { by } => {
                if by == n.core.local_army {
                    s.chat.system("ONLY THE HOST CAN CHANGE THE GAME SPEED");
                }
            }
            NetEvent::Dropped { army, tick } => {
                let who = army_label(&s.world, army);
                s.chat.system(&format!("{who} DISCONNECTED - AI TAKES OVER"));
                s.status_banner = format!("{who} DISCONNECTED - AI TAKES OVER");
                if let Some(rec) = s.recorder.as_mut() {
                    rec.ai_event(tick, army, true);
                }
            }
            NetEvent::InputDelay { turn: _, delay } => {
                s.chat.system(&format!("ADAPTIVE INPUT DELAY SET TO {delay} TURNS"));
            }
        }
    }
}

/// Post-desync pump: collect summaries, keep the null-turn exchange going, write the dump.
fn pump_desync(n: &mut NetPlay, s: &mut PlaySession) {
    let Some(dx) = n.desync.as_mut() else {
        return;
    };
    while let Some(msg) = n.bridge.try_recv() {
        match msg {
            Ok((b, _)) => dx.summaries.extend(n.core.receive_unverified(&b)),
            Err(_) => {
                dx.closed = true;
                break;
            }
        }
    }
    if !dx.closed
        && let Some(pkt) = n.core.halted_packet(&s.world, dx.until)
        && n.bridge.submit(pkt).is_err()
    {
        dx.closed = true;
    }
    let live = n.core.player_count.saturating_sub(n.core.dropped.len());
    let complete = dx.summaries.len() >= live
        || n.core.received_turns() > dx.until
        || dx.closed
        || dx.started.elapsed() > Duration::from_secs(20);
    if dx.dump.is_none() && complete {
        match write_desync_dump(&n.core, &s.world, &dx.report, &dx.summaries) {
            Ok(dir) => {
                eprintln!("desync dump written to {}", dir.display());
                s.chat.system(&format!("DESYNC DUMP WRITTEN: {}", dir.display()));
                n.failure = Some(format!("DESYNC T{} - DUMP WRITTEN", dx.report.turn));
                dx.dump = Some(dir);
            }
            Err(e) => {
                eprintln!("desync dump failed: {e}");
                dx.dump = Some(PathBuf::new());
            }
        }
    }
}

fn format_status(n: &NetPlay) -> String {
    format!(
        "NET A{} P{} PING {:.0}MS BUF {}/{} LAG {}{}{}",
        n.core.local_army,
        n.core.player_count,
        n.rtt_ms,
        n.core.buffered_turns(),
        n.core.input_delay,
        n.core.in_flight_turns(),
        if n.stalled_now {
            format!(" WAIT {}", n.stall_ticks)
        } else {
            String::new()
        },
        if n.core.dropped.is_empty() {
            String::new()
        } else {
            format!(" DROPPED {}", n.core.dropped.len())
        }
    )
}

// ---------------------------------------------------------------------------
// Match setup helpers shared by `farust host/join`
// ---------------------------------------------------------------------------

/// Builds the deterministic world every peer simulates from the handshake.
pub fn world_from_handshake(
    catalog: crate::skirmish::Catalog,
    map: &crate::skirmish::MapContext,
    hs: &Handshake,
) -> Result<Skirmish, String> {
    if !map.map_id.eq_ignore_ascii_case(&hs.map_id) {
        return Err(format!("map mismatch: host {} local {}", hs.map_id, map.map_id));
    }
    let mut world = Skirmish::new(
        catalog,
        map.grid.clone(),
        map.size,
        map.mass_points.clone(),
        &hs.setups,
        hs.seed,
    )?;
    world.attach_map(map);
    Ok(world)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pathfinding::PathGrid;
    use crate::sim::SCALE;
    use crate::skirmish::tests::test_catalog;

    fn setups(n: usize) -> Vec<ArmySetup> {
        let spots = [
            (45, 45),
            (205, 205),
            (205, 45),
            (45, 205),
            (125, 30),
            (125, 220),
        ];
        (0..n)
            .map(|i| ArmySetup {
                name: format!("ARMY_{}", i + 1),
                faction: i % 2,
                start: (spots[i].0 * SCALE, spots[i].1 * SCALE),
                team: i as u8,
            })
            .collect()
    }

    fn world(setups: &[ArmySetup], seed: u64) -> Skirmish {
        let grid = PathGrid::new(64, 64, 4 * SCALE);
        let mass = setups
            .iter()
            .map(|s| (s.start.0 - 5 * SCALE, s.start.1 - 5 * SCALE))
            .collect();
        Skirmish::new(test_catalog(), grid, (256 * SCALE, 256 * SCALE), mass, setups, seed)
            .unwrap()
    }

    fn handshake(players: usize, armies: usize, seed: u64, ticks: u64) -> Handshake {
        Handshake {
            observers: 0,
            map_id: "LOOP".into(),
            seed,
            setups: setups(armies),
            player_count: players,
            turn_ticks: TURN_TICKS,
            input_delay: DEFAULT_INPUT_DELAY,
            tick_limit: ticks,
            assigned_army: 0,
        }
    }

    fn sample_commands() -> Vec<Command> {
        vec![
            Command::Move { units: vec![1, 2, 3], x: 10 * SCALE, y: -4, queue: true },
            Command::Produce { factory: 7, unit_type: "UEL0105".into(), count: 3 },
            Command::Stop { units: vec![] },
            Command::Build { builder: 1, unit_type: "UEB1103".into(), x: 5, y: 6, queue: false },
        ]
    }

    #[test]
    fn handshake_and_turn_codecs_round_trip() {
        let mut hs = handshake(3, 4, 99, 600);
        hs.assigned_army = 2;
        assert_eq!(decode_handshake(&encode_handshake(&hs)).unwrap(), hs);
        let pkt = TurnPacket {
            turn: 41,
            hash: 0xdead_beef_1234,
            commands: sample_commands(),
            controls: vec![
                NetControl::Pause(true),
                NetControl::Speed(4),
                NetControl::Chat { allies: true, text: "gl hf".into() },
                NetControl::DesyncSummary { turn: 3, tick: 9, hash: 7, text: "units=4".into() },
            ],
        };
        assert_eq!(decode_turn(&encode_turn(&pkt)).unwrap(), pkt);
    }

    #[test]
    fn turn_bundle_codec_round_trip() {
        let b = TurnBundle {
            turn: 1234,
            hashes: vec![(0, 1), (1, u64::MAX), (2, 77)],
            batches: vec![(0, sample_commands()), (2, vec![Command::Stop { units: vec![9] }])],
            controls: vec![(0, NetControl::Speed(2)), (1, NetControl::Chat { allies: false, text: "hi".into() }), (SYSTEM_SENDER, NetControl::Dropped(2))],
        };
        let enc = encode_turn_bundle(&b);
        assert_eq!(decode_turn_bundle(&enc).unwrap(), b);
        // Truncation and trailing garbage are rejected.
        assert!(decode_turn_bundle(&enc[..enc.len() - 1]).is_err());
        let mut extra = enc.clone();
        extra.push(0);
        assert!(decode_turn_bundle(&extra).is_err());
    }

    #[test]
    fn canonicalization_sorts_armies_and_preserves_issue_order() {
        let c = |u: u32| Command::Stop { units: vec![u] };
        let b = canonicalize_bundle(TurnBundle {
            turn: 5,
            hashes: vec![(2, 20), (0, 0), (1, 10), (2, 20)],
            batches: vec![
                (2, vec![c(1), c(2)]),
                (0, vec![]),
                (1, vec![c(3)]),
                (2, vec![c(4)]),
                (0, vec![c(5)]),
            ],
            controls: vec![(2, NetControl::Pause(true)), (0, NetControl::Speed(3)), (2, NetControl::Pause(false))],
        });
        assert_eq!(b.hashes, vec![(0, 0), (1, 10), (2, 20)]);
        assert_eq!(
            b.batches,
            vec![(0, vec![c(5)]), (1, vec![c(3)]), (2, vec![c(1), c(2), c(4)])]
        );
        // Permuted arrival yields the identical canonical bundle.
        let b2 = canonicalize_bundle(TurnBundle {
            turn: 5,
            hashes: vec![(1, 10), (2, 20), (0, 0)],
            batches: vec![(1, vec![c(3)]), (0, vec![c(5)]), (2, vec![c(1), c(2), c(4)])],
            controls: vec![(0, NetControl::Speed(3)), (2, NetControl::Pause(true)), (2, NetControl::Pause(false))],
        });
        assert_eq!(b, b2);
    }

    #[test]
    fn core_detects_desync_with_diagnostics() {
        let s = setups(2);
        let mut w = world(&s, 5);
        let mut core = LockstepCore::new(&w, 1, 2, TURN_TICKS, 1);
        let p0 = core.prime(&w).remove(0);
        let good = TurnBundle { turn: 0, hashes: vec![(0, p0.hash), (1, p0.hash)], batches: vec![], controls: vec![] };
        core.receive(good).unwrap();
        core.queue_local(vec![Command::Stop { units: vec![1] }]);
        let mut pkt = None;
        while pkt.is_none() {
            match core.tick(&mut w) {
                TickOutcome::Stepped(mut pkts) => pkt = pkts.pop(),
                TickOutcome::Stalled | TickOutcome::Paused(_) => panic!("bundle was buffered"),
            }
        }
        let pkt = pkt.unwrap();
        assert_eq!(pkt.turn, 1);
        assert_eq!(pkt.commands.len(), 1);
        assert!(matches!(core.tick(&mut w), TickOutcome::Stalled));
        let bad = TurnBundle {
            turn: 1,
            hashes: vec![(0, pkt.hash ^ 1), (1, pkt.hash)],
            batches: vec![(1, pkt.commands.clone())],
            controls: vec![],
        };
        let d = core.receive(bad).unwrap_err();
        assert_eq!(d.turn, 1);
        assert_eq!(d.tick, TURN_TICKS);
        assert_eq!(d.local_hash, pkt.hash);
        assert_eq!(d.mismatches, vec![(0, pkt.hash ^ 1)]);
        let text = d.to_string();
        assert!(text.contains("LOCKSTEP DESYNC at turn 1 (tick 3)"), "{text}");
        assert!(text.contains("last commands: turn 0"), "{text}");
        // Out-of-order bundles are protocol errors.
        let skip = TurnBundle { turn: 7, hashes: vec![], batches: vec![], controls: vec![] };
        assert!(core.receive(skip).is_err());
    }

    /// Runs host + clients as threads over 127.0.0.1 and returns every final hash.
    fn run_loopback(players: usize, armies: usize, ticks: u64) -> Vec<u64> {
        run_loopback_obs(players, armies, ticks, 0)
    }

    fn run_loopback_obs(players: usize, armies: usize, ticks: u64, observers: usize) -> Vec<u64> {
        let mut hs = handshake(players, armies, 4242, ticks);
        hs.observers = observers;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let hs_host = hs.clone();
        let host = std::thread::spawn(move || {
            let mut t = LockstepHost::accept_players(listener, &hs_host, &mut |_, _| {}).unwrap();
            let mut w = world(&hs_host.setups, hs_host.seed);
            let mut core = LockstepCore::from_handshake(&w, &hs_host);
            assert_eq!(core.shared_ai_armies().len(), armies - players);
            run_headless(&mut t, &mut core, &mut w, hs_host.tick_limit).unwrap()
        });
        let clients: Vec<_> = (1..players + observers)
            .map(|_| {
                let addr = addr.clone();
                std::thread::spawn(move || {
                    let (mut c, hs) = LockstepClient::connect_retry(&addr, Duration::from_secs(10)).unwrap();
                    let mut w = world(&hs.setups, hs.seed);
                    let mut core = LockstepCore::from_handshake(&w, &hs);
                    let h = run_headless(&mut c, &mut core, &mut w, hs.tick_limit).unwrap();
                    (hs.assigned_army, h, w.tick())
                })
            })
            .collect();
        let mut out = vec![host.join().unwrap()];
        let mut seen = Vec::new();
        for c in clients {
            let (army, h, tick) = c.join().unwrap();
            assert_eq!(tick, ticks.div_ceil(TURN_TICKS) * TURN_TICKS);
            seen.push(army);
            out.push(h);
        }
        seen.sort();
        let mut want: Vec<ArmyId> = (1..players as ArmyId).collect();
        want.extend((0..observers as ArmyId).map(|i| OBSERVER_ARMY_BASE + i));
        assert_eq!(seen, want);
        out
    }

    #[test]
    fn live_observer_follows_a_2_player_match_without_desync_or_influence() {
        let with_obs = run_loopback_obs(2, 2, 300, 1);
        assert_eq!(with_obs.len(), 3);
        assert!(with_obs.iter().all(|h| *h == with_obs[0]), "{with_obs:x?}");
        let without = run_loopback(2, 2, 300);
        assert_eq!(without[0], with_obs[0], "an observer never changes the match");
        assert!(is_observer(OBSERVER_ARMY_BASE) && !is_observer(7));
    }

    #[test]
    fn loopback_4_players_match_hashes() {
        let hashes = run_loopback(4, 4, 300);
        assert!(hashes.iter().all(|h| *h == hashes[0]), "{hashes:x?}");
    }

    #[test]
    fn loopback_3_players_plus_ai_slot_match_hashes() {
        let hashes = run_loopback(3, 4, 301);
        assert!(hashes.iter().all(|h| *h == hashes[0]), "{hashes:x?}");
    }

    #[test]
    fn async_bridge_drives_two_cores_in_lockstep() {
        let hs = handshake(2, 3, 42, 0);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let hs_host = hs.clone();
        let host_t = std::thread::spawn(move || {
            LockstepHost::accept_players(listener, &hs_host, &mut |_, _| {}).unwrap()
        });
        let (client, hs_c) = LockstepClient::connect_retry(&addr, Duration::from_secs(10)).unwrap();
        let host = host_t.join().unwrap();
        let mut peers = vec![
            (world(&hs.setups, hs.seed), AsyncLockstepBridge::spawn(Box::new(host)), {
                let w = world(&hs.setups, hs.seed);
                LockstepCore::from_handshake(&w, &hs)
            }),
            (world(&hs_c.setups, hs_c.seed), AsyncLockstepBridge::spawn(Box::new(client)), {
                let w = world(&hs_c.setups, hs_c.seed);
                LockstepCore::from_handshake(&w, &hs_c)
            }),
        ];
        for (w, b, core) in &mut peers {
            for p in core.prime(w) {
                b.submit(p).unwrap();
            }
        }
        for _ in 0..90 {
            for (w, b, core) in &mut peers {
                loop {
                    core.queue_local(vec![Command::Stop { units: vec![] }]);
                    match core.tick(w) {
                        TickOutcome::Stalled => {
                            let (bundle, _) = b
                                .recv_timeout(Duration::from_secs(5))
                                .expect("bundle")
                                .expect("ok");
                            core.receive(bundle).unwrap();
                        }
                        TickOutcome::Stepped(pkts) | TickOutcome::Paused(pkts) => {
                            for p in pkts {
                                b.submit(p).unwrap();
                            }
                            break;
                        }
                    }
                }
            }
        }
        assert_eq!(peers[0].0.tick(), 90);
        assert_eq!(peers[0].0.state_hash(), peers[1].0.state_hash());
    }

    // ----- Phase 21 match session -------------------------------------------------------

    type Hook = Box<dyn FnMut(&mut LockstepCore, &Skirmish) -> bool + Send>;

    struct PeerOut {
        army: ArmyId,
        result: Result<u64, String>,
        tick: u64,
        speed: u32,
        applied: Vec<AppliedControl>,
        events: Vec<NetEvent>,
        takeovers: Vec<(u64, ArmyId)>,
        shared: Vec<ArmyId>,
        recorded: Vec<(u64, ArmyId, Command)>,
    }

    fn run_peer(
        t: &mut dyn TurnTransport,
        hs: &Handshake,
        seed: u64,
        dump: Option<PathBuf>,
        make_hook: fn(ArmyId) -> Hook,
    ) -> PeerOut {
        let mut w = world(&hs.setups, seed);
        let mut core = LockstepCore::from_handshake(&w, hs);
        core.dump_dir = dump;
        let mut hook = make_hook(hs.assigned_army);
        let result = run_headless_with(t, &mut core, &mut w, hs.tick_limit, &mut *hook);
        PeerOut {
            army: hs.assigned_army,
            result,
            tick: w.tick(),
            speed: core.speed(),
            applied: core.applied_controls().to_vec(),
            events: core.drain_events(),
            takeovers: core.takeovers().to_vec(),
            shared: core.shared_ai_armies(),
            recorded: core.recorded_commands().to_vec(),
        }
    }

    /// Host + `players - 1` clients over 127.0.0.1 with per-army hooks / seeds; sorted by army.
    fn run_session(
        hs: Handshake,
        seed_of: fn(ArmyId, u64) -> u64,
        dump: Option<PathBuf>,
        make_hook: fn(ArmyId) -> Hook,
    ) -> Vec<PeerOut> {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let hs_host = hs.clone();
        let dump_h = dump.clone();
        let host = std::thread::spawn(move || {
            let mut t = LockstepHost::accept_players(listener, &hs_host, &mut |_, _| {}).unwrap();
            run_peer(&mut t, &hs_host, seed_of(0, hs_host.seed), dump_h, make_hook)
        });
        let clients: Vec<_> = (1..hs.player_count)
            .map(|_| {
                let addr = addr.clone();
                let dump = dump.clone();
                std::thread::spawn(move || {
                    let (mut c, hs) = LockstepClient::connect_retry(&addr, Duration::from_secs(10)).unwrap();
                    let seed = seed_of(hs.assigned_army, hs.seed);
                    // `c` is dropped when this returns: a hook that leaves early disconnects.
                    run_peer(&mut c, &hs, seed, dump, make_hook)
                })
            })
            .collect();
        let mut out = vec![host.join().unwrap()];
        out.extend(clients.into_iter().map(|c| c.join().unwrap()));
        out.sort_by_key(|p| p.army);
        out
    }

    fn same_seed(_: ArmyId, s: u64) -> u64 {
        s
    }

    fn control_plan(army: ArmyId) -> Hook {
        use NetControl::*;
        let plan: Vec<(u64, ArmyId, NetControl)> = vec![
            (8, 1, Pause(true)),
            (12, 0, Speed(4)),
            (14, 2, Speed(2)),
            (16, 2, Pause(false)),
            (20, 1, Chat { allies: false, text: "push now".into() }),
            (21, 2, Chat { allies: true, text: "team only".into() }),
            (24, 1, Pause(true)),
            (28, 0, Pause(false)),
            (32, 1, Pause(true)),
            (36, 0, Pause(false)),
            (40, 1, Pause(true)),
            (44, 2, Pause(true)),
            (48, 2, Pause(false)),
        ];
        let mut done = vec![false; plan.len()];
        Box::new(move |core, _w| {
            let turn = core.exec_turn();
            for (i, (t, a, c)) in plan.iter().enumerate() {
                if *a == army && turn >= *t && !done[i] {
                    done[i] = true;
                    core.queue_control(c.clone());
                }
            }
            true
        })
    }

    #[test]
    fn pause_speed_and_chat_controls_apply_identically_across_peers() {
        let ticks = 240;
        let peers = run_session(handshake(3, 3, 777, ticks), same_seed, None, control_plan);
        let limit = ticks.div_ceil(TURN_TICKS) * TURN_TICKS;
        let h0 = *peers[0].result.as_ref().unwrap();
        for p in &peers {
            assert_eq!(p.result.as_ref().unwrap(), &h0, "army {} hash", p.army);
            assert_eq!(p.tick, limit);
            assert_eq!(p.applied, peers[0].applied, "army {} applied a different control order", p.army);
            assert_eq!(p.speed, 4, "host speed change applies everywhere");
            assert_eq!(p.recorded, peers[0].recorded);
        }
        let log = &peers[0].applied;
        let find = |from: ArmyId, c: NetControl| log.iter().filter(move |a| a.from == from && a.control == c);
        // First pause by army 1 and its resume by army 2 happen at the same sim tick.
        let p1 = find(1, NetControl::Pause(true)).next().unwrap();
        let r2 = find(2, NetControl::Pause(false)).next().unwrap();
        assert!(p1.accepted && r2.accepted);
        assert!(r2.turn > p1.turn);
        assert_eq!(p1.tick, r2.tick, "no ticks run while paused");
        // Speed: host accepted, client denied.
        assert!(find(0, NetControl::Speed(4)).next().unwrap().accepted);
        assert!(!find(2, NetControl::Speed(2)).next().unwrap().accepted);
        // Army 1: three pauses accepted, the fourth denied; army 2 still has its own budget.
        let a1: Vec<bool> = find(1, NetControl::Pause(true)).map(|a| a.accepted).collect();
        assert_eq!(a1, vec![true, true, true, false]);
        assert!(find(2, NetControl::Pause(true)).next().unwrap().accepted);
        // Every pause/resume pair spans zero ticks; overall the sim still reached the limit.
        let paused_turns: u64 = log
            .windows(2)
            .filter(|w| w[0].control == NetControl::Pause(true) && w[0].accepted && w[1].control == NetControl::Pause(false))
            .map(|w| {
                assert_eq!(w[0].tick, w[1].tick);
                w[1].turn - w[0].turn
            })
            .sum();
        assert!(paused_turns >= 4 * 3, "{paused_turns}");
        // Chat: public line everywhere, allies-only line only for army 2 (teams differ).
        let chats = |p: &PeerOut| -> Vec<String> {
            p.events
                .iter()
                .filter_map(|e| match e {
                    NetEvent::Chat { text, .. } => Some(text.clone()),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(chats(&peers[0]), vec!["push now".to_string()]);
        assert_eq!(chats(&peers[1]), vec!["push now".to_string()]);
        assert_eq!(chats(&peers[2]), vec!["push now".to_string(), "team only".to_string()]);
    }

    fn leave_at_turn_40(army: ArmyId) -> Hook {
        Box::new(move |core, _| army != 2 || core.exec_turn() < 40)
    }

    #[test]
    fn disconnect_hands_army_to_ai_and_remaining_peers_agree() {
        let ticks = 420;
        let peers = run_session(handshake(3, 3, 991, ticks), same_seed, None, leave_at_turn_40);
        let limit = ticks.div_ceil(TURN_TICKS) * TURN_TICKS;
        let (host, c1, gone) = (&peers[0], &peers[1], &peers[2]);
        assert!(gone.tick < limit, "army 2 left early");
        let h = host.result.as_ref().unwrap();
        assert_eq!(c1.result.as_ref().unwrap(), h, "remaining peers agree");
        assert_eq!(host.tick, limit);
        assert_eq!(c1.tick, limit);
        assert_eq!(host.takeovers, c1.takeovers);
        assert_eq!(host.takeovers.len(), 1);
        let (takeover_tick, army) = host.takeovers[0];
        assert_eq!(army, 2);
        assert!(takeover_tick >= gone.tick.saturating_sub(3 * TURN_TICKS));
        assert_eq!(host.shared, vec![2]);
        assert_eq!(c1.shared, vec![2]);
        assert_eq!(host.recorded, c1.recorded);
        assert!(
            host.recorded.iter().any(|(t, a, _)| *a == 2 && *t >= takeover_tick),
            "the takeover AI commands army 2"
        );
        assert!(host.events.iter().any(|e| matches!(e, NetEvent::Dropped { army: 2, .. })));
    }

    fn desync_seed(army: ArmyId, s: u64) -> u64 {
        if army == 1 { s + 1 } else { s }
    }

    fn no_hook(_: ArmyId) -> Hook {
        Box::new(|_, _| true)
    }

    #[test]
    fn desync_exchanges_summaries_and_writes_dumps() {
        let base = std::env::temp_dir().join(format!(
            "farust-desync-test-{}-{}",
            std::process::id(),
            crate::savegame::utc_stamp()
        ));
        let peers = run_session(handshake(2, 2, 31337, 300), desync_seed, Some(base.clone()), no_hook);
        for p in &peers {
            let err = p.result.as_ref().unwrap_err();
            assert!(err.contains("LOCKSTEP DESYNC at turn 0"), "{err}");
            let dir = err.split("desync dump: ").nth(1).map(PathBuf::from).expect("dump path");
            assert!(dir.starts_with(&base));
            let report = std::fs::read_to_string(dir.join("report.txt")).unwrap();
            assert!(report.contains("army 0: tick=0 units="), "{report}");
            assert!(report.contains("army 1: tick=0 units="), "{report}");
            assert!(report.contains("<-- differs from local"), "{report}");
            assert!(dir.join("peer-army0.txt").is_file() && dir.join("peer-army1.txt").is_file());
            let rep = crate::replay::Replay::load(&dir.join("commands.farust-replay")).unwrap();
            assert_eq!(rep.header.map_id, "LOOP");
            assert_eq!(rep.header.seed, desync_seed(p.army, 31337));
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn control_codec_round_trips_in_packets_and_rejects_forged_drops() {
        let pkt = TurnPacket {
            turn: 2,
            hash: 3,
            commands: vec![],
            controls: vec![NetControl::Chat { allies: false, text: "x".into() }],
        };
        let enc = encode_turn(&pkt);
        assert_eq!(decode_turn(&enc).unwrap(), pkt);
        assert!(decode_turn(&enc[..enc.len() - 1]).is_err());
        let mut bad = enc.clone();
        bad.push(9);
        assert!(decode_turn(&bad).is_err());
        let s = setups(2);
        let w = world(&s, 1);
        let mut core = LockstepCore::new(&w, 1, 2, TURN_TICKS, 1);
        core.queue_control(NetControl::Dropped(0));
        core.queue_control(NetControl::InputDelay(5));
        core.queue_control(NetControl::Pause(true));
        assert_eq!(core.prime(&w)[0].controls, vec![NetControl::Pause(true)]);
    }

    // ----- Phase 44: UDP lockstep, lossy netem, adaptive input delay & hole punching ----

    struct UdpLoopbackRun {
        hashes: Vec<u64>,
        host_netem: NetemStats,
        client_netem: Vec<NetemStats>,
        delay_changes: Vec<Vec<(u64, u64)>>,
    }

    fn run_udp_loopback(
        players: usize,
        armies: usize,
        ticks: u64,
        host_netem: NetemConfig,
        client_netem_base: NetemConfig,
    ) -> UdpLoopbackRun {
        let hs = handshake(players, armies, 4242, ticks);
        let mux = UdpMux::bind("127.0.0.1:0", host_netem).unwrap();
        let addr = mux.local_addr().to_string();
        let hs_host = hs.clone();
        let host = std::thread::spawn(move || {
            let mut t = LockstepHost::accept_players_udp(mux, &hs_host, &mut |_, _| {}).unwrap();
            let mut w = world(&hs_host.setups, hs_host.seed);
            let mut core = LockstepCore::from_handshake(&w, &hs_host);
            let h = run_headless(&mut t, &mut core, &mut w, hs_host.tick_limit).unwrap();
            let st = t.netem_stats().unwrap_or_default();
            (h, st, core.delay_changes().to_vec())
        });
        let clients: Vec<_> = (1..players)
            .map(|idx| {
                let addr = addr.clone();
                let mut cn = client_netem_base.clone();
                cn.seed = cn.seed.wrapping_add((idx as u64) * 7919);
                std::thread::spawn(move || {
                    let (mut c, hs) =
                        LockstepClient::connect_udp_retry(&addr, cn, Duration::from_secs(15)).unwrap();
                    let mut w = world(&hs.setups, hs.seed);
                    let mut core = LockstepCore::from_handshake(&w, &hs);
                    let h = run_headless(&mut c, &mut core, &mut w, hs.tick_limit).unwrap();
                    let st = c.netem_stats().unwrap_or_default();
                    (hs.assigned_army, h, w.tick(), st, core.delay_changes().to_vec())
                })
            })
            .collect();
        let (h_host, host_st, host_dc) = host.join().unwrap();
        let mut hashes = vec![h_host];
        let mut client_sts = Vec::new();
        let mut dcs = vec![host_dc];
        let mut seen = Vec::new();
        for c in clients {
            let (army, h, tick, st, dc) = c.join().unwrap();
            assert_eq!(tick, ticks.div_ceil(TURN_TICKS) * TURN_TICKS);
            seen.push(army);
            hashes.push(h);
            client_sts.push(st);
            dcs.push(dc);
        }
        seen.sort();
        assert_eq!(seen, (1..players as ArmyId).collect::<Vec<_>>());
        UdpLoopbackRun {
            hashes,
            host_netem: host_st,
            client_netem: client_sts,
            delay_changes: dcs,
        }
    }

    #[test]
    fn udp_lossy_netem_2_and_3_peer_matches_tcp_hash() {
        let lossy = NetemConfig {
            loss_pct: 5.0,
            dup_pct: 2.0,
            reorder_window: 3,
            delay_ms: 0,
            jitter_ms: 0,
            seed: 44_001,
        };
        for (players, armies, ticks) in [(2, 2, 240u64), (3, 4, 240u64)] {
            let tcp_hashes = run_loopback(players, armies, ticks);
            assert!(tcp_hashes.iter().all(|h| *h == tcp_hashes[0]));
            let expected = tcp_hashes[0];

            let clean = run_udp_loopback(players, armies, ticks, NetemConfig::default(), NetemConfig::default());
            assert!(
                clean.hashes.iter().all(|h| *h == expected),
                "clean UDP ({players}p) mismatch: {:x?} vs tcp {expected:x}",
                clean.hashes
            );

            let lossy_run = run_udp_loopback(players, armies, ticks, lossy.clone(), lossy.clone());
            assert!(
                lossy_run.hashes.iter().all(|h| *h == expected),
                "lossy UDP ({players}p) mismatch: {:x?} vs tcp {expected:x}",
                lossy_run.hashes
            );
            let total_dropped: u64 =
                lossy_run.host_netem.dropped + lossy_run.client_netem.iter().map(|s| s.dropped).sum::<u64>();
            let total_duped: u64 = lossy_run.host_netem.duplicated
                + lossy_run.client_netem.iter().map(|s| s.duplicated).sum::<u64>();
            let total_reordered: u64 = lossy_run.host_netem.reordered
                + lossy_run.client_netem.iter().map(|s| s.reordered).sum::<u64>();
            assert!(total_dropped > 0, "expected simulated drops ({players}p)");
            assert!(total_duped > 0, "expected simulated duplicates ({players}p)");
            assert!(total_reordered > 0, "expected simulated reorders ({players}p)");
        }
    }

    #[test]
    fn adaptive_input_delay_curve_core_sync_and_high_rtt_udp() {
        // 1. Curve & clamping checks.
        assert_eq!(compute_adaptive_delay(1.0, 0.5, TURN_TICKS), 2);
        assert_eq!(compute_adaptive_delay(120.0, 10.0, TURN_TICKS), 2);
        assert_eq!(compute_adaptive_delay(160.0, 20.0, TURN_TICKS), 3);
        assert_eq!(compute_adaptive_delay(320.0, 40.0, TURN_TICKS), 4);
        assert_eq!(compute_adaptive_delay(2000.0, 500.0, TURN_TICKS), ADAPTIVE_MAX_DELAY);

        // 2. Controller step-up and hysteresis step-down.
        let mut ctrl = AdaptiveDelayController::new(true, DEFAULT_INPUT_DELAY, TURN_TICKS);
        assert_eq!(ctrl.maybe_adjust(2, 400.0, 50.0), None, "wait until turn 4");
        assert_eq!(ctrl.maybe_adjust(8, 350.0, 50.0), Some(4));
        assert_eq!(ctrl.maybe_adjust(16, 10.0, 2.0), None, "first low window builds hysteresis");
        assert_eq!(ctrl.maybe_adjust(24, 10.0, 2.0), Some(3), "steps down by 1 after hysteresis");
        assert_eq!(ctrl.maybe_adjust(32, 10.0, 2.0), None);
        assert_eq!(ctrl.maybe_adjust(40, 10.0, 2.0), Some(2));
        assert_eq!(ctrl.maybe_adjust(48, 10.0, 2.0), None, "clamped at ADAPTIVE_MIN_DELAY");

        // 3. Synchronized delay increase (2 -> 4) AND decrease (4 -> 2) across two cores.
        let s = setups(2);
        let mut w0 = world(&s, 101);
        let mut w1 = world(&s, 101);
        let mut c0 = LockstepCore::new(&w0, 0, 2, TURN_TICKS, 2);
        let mut c1 = LockstepCore::new(&w1, 1, 2, TURN_TICKS, 2);
        let mut q0: VecDeque<TurnPacket> = c0.prime(&w0).into();
        let mut q1: VecDeque<TurnPacket> = c1.prime(&w1).into();
        let exchange_pair = |c0: &mut LockstepCore, c1: &mut LockstepCore, p0: TurnPacket, p1: TurnPacket| {
            assert_eq!(p0.turn, p1.turn);
            let turn = p0.turn;
            let mut controls = Vec::new();
            if turn == 5 {
                controls.push((SYSTEM_SENDER, NetControl::InputDelay(4)));
            } else if turn == 14 {
                controls.push((SYSTEM_SENDER, NetControl::InputDelay(2)));
            }
            let b = canonicalize_bundle(TurnBundle {
                turn,
                hashes: vec![(0, p0.hash), (1, p1.hash)],
                batches: vec![(0, p0.commands), (1, p1.commands)],
                controls,
            });
            c0.receive(b.clone()).unwrap();
            c1.receive(b).unwrap();
        };
        while let (Some(p0), Some(p1)) = (q0.pop_front(), q1.pop_front()) {
            exchange_pair(&mut c0, &mut c1, p0, p1);
        }
        for _ in 0..90 {
            let o0 = c0.tick(&mut w0);
            let o1 = c1.tick(&mut w1);
            let (TickOutcome::Stepped(pkts0), TickOutcome::Stepped(pkts1)) = (o0, o1) else {
                panic!("expected both cores to step cleanly without stalling");
            };
            assert_eq!(pkts0.len(), pkts1.len());
            for (p0, p1) in pkts0.into_iter().zip(pkts1) {
                exchange_pair(&mut c0, &mut c1, p0, p1);
            }
        }
        assert_eq!(w0.state_hash(), w1.state_hash());
        assert_eq!(c0.delay_changes(), &[(5, 4), (14, 2)]);
        assert_eq!(c1.delay_changes(), &[(5, 4), (14, 2)]);
        assert_eq!(c0.input_delay, 2);
        assert_eq!(c1.input_delay, 2);

        // 4. Live high-RTT UDP loopback triggers host-driven adaptive delay and stays in sync.
        let high_rtt = NetemConfig {
            loss_pct: 2.0,
            dup_pct: 1.0,
            reorder_window: 0,
            delay_ms: 95,
            jitter_ms: 15,
            seed: 44_999,
        };
        let run = run_udp_loopback(2, 2, 48, high_rtt.clone(), high_rtt);
        assert_eq!(run.hashes[0], run.hashes[1], "high-RTT UDP peers desynced");
        assert!(
            !run.delay_changes[0].is_empty(),
            "expected host to increase input_delay under ~200ms RTT"
        );
        assert_eq!(
            run.delay_changes[0], run.delay_changes[1],
            "all peers must apply identical delay changes on the same turns"
        );
        assert!(
            run.delay_changes[0][0].1 > DEFAULT_INPUT_DELAY,
            "expected delay > 2, got {:?}",
            run.delay_changes[0]
        );
    }

    #[test]
    fn udp_rendezvous_hole_punch_lockstep_match() {
        let rv = crate::rendezvous::RendezvousServer::bind("127.0.0.1:0").unwrap();
        let rv_addr = rv.local_addr().to_string();
        let hs = handshake(2, 2, 7744, 150);
        let lossy = NetemConfig {
            loss_pct: 5.0,
            dup_pct: 2.0,
            reorder_window: 3,
            delay_ms: 0,
            jitter_ms: 0,
            seed: 44_777,
        };
        let (code_tx, code_rx) = mpsc::channel();
        let hs_host = hs.clone();
        let rv_h = rv_addr.clone();
        let netem_h = lossy.clone();
        let host_t = std::thread::spawn(move || {
            // Register via code with UDP hole punching enabled; send invite code once bound.
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let tcp_addr = listener.local_addr().unwrap();
            let udp_mux = UdpMux::bind("127.0.0.1:0", netem_h).unwrap();
            udp_mux.set_accepting(true);
            let punch_mux = udp_mux.clone();
            let mut reg = crate::rendezvous::HostRegistration::register_with_udp(
                &rv_h,
                tcp_addr,
                udp_mux.clone(),
                "Host",
                &hs_host.map_id,
                Duration::from_secs(3),
                Box::new(|_, _| {}),
                Box::new(move |tok, targets| {
                    let _ = punch_mux.start_punch(tok, ROLE_HOST, targets, Duration::from_secs(3));
                }),
            )
            .unwrap();
            let _ = code_tx.send(reg.code.clone());
            let mut t = LockstepHost::accept_players_udp(udp_mux, &hs_host, &mut |_, _| {}).unwrap();
            reg.close();
            let mut w = world(&hs_host.setups, hs_host.seed);
            let mut core = LockstepCore::from_handshake(&w, &hs_host);
            run_headless(&mut t, &mut core, &mut w, hs_host.tick_limit).unwrap()
        });

        let code = code_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let (mut client, c_hs, via) = LockstepClient::connect_code_udp_retry(
            &rv_addr,
            &code,
            false,
            lossy,
            Duration::from_secs(10),
        )
        .unwrap();
        assert!(via.is_udp(), "expected punched UDP connection, got {via:?}");
        let mut w_c = world(&c_hs.setups, c_hs.seed);
        let mut core_c = LockstepCore::from_handshake(&w_c, &c_hs);
        let h_client = run_headless(&mut client, &mut core_c, &mut w_c, c_hs.tick_limit).unwrap();
        let h_host = host_t.join().unwrap();
        assert_eq!(h_host, h_client, "punched UDP lockstep hashes must match");
    }
}
