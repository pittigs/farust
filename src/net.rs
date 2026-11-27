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

pub const TURN_TICKS: u64 = 3;
/// Turns between collecting a command and executing it (latency hiding).
pub const DEFAULT_INPUT_DELAY: u64 = 2;
/// v3: menu-hosted lobbies greet with `Welcome` (`net_lobby.rs`); in-game turns carry [`NetControl`]s.
/// v4 (Phase 24): handshake carries the enabled mods' identities.
/// v5 (Phase 28): wall sections are buildable and block pathing (older peers would reject
/// wall `Build` commands and desync), so mixed v4/v5 games are refused at the handshake.
/// v6 (Phase 29): sim rules v2 (ballistics, turrets, structure footprints).
/// v7 (Phase 30): sim rules v3 (air fuel & staging).
pub const PROTOCOL_VERSION: u32 = 7;
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
}

impl NetControl {
    /// Controls a peer may put in its own packet (the host strips everything else).
    pub fn peer_may_send(&self) -> bool {
        !matches!(self, NetControl::Dropped(_))
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
// Transports: star-topology host & client
// ---------------------------------------------------------------------------

/// Blocking per-turn exchange: submit the local packet, get the canonical bundle back.
pub trait TurnTransport: Send {
    fn exchange(&mut self, pkt: TurnPacket) -> Result<TurnBundle, String>;
}

pub struct LockstepHost {
    clients: Vec<(ArmyId, TcpStream)>,
    pub local_army: ArmyId,
    next_turn: u64,
}

impl LockstepHost {
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

        let mut clients = Vec::with_capacity(n - 1);
        for slot in 1..n {
            let (mut stream, peer, relayed) = rx
                .recv()
                .map_err(|_| format!("accept slot {slot} failed: listener closed"))?;
            let _ = stream.set_nodelay(true);
            let _ = stream.set_read_timeout(Some(NET_TIMEOUT));
            let army = slot as ArmyId;
            let mut client_hs = hs.clone();
            client_hs.assigned_army = army;
            write_frame(&mut stream, &encode_handshake(&client_hs))?;
            on_join(army, peer, relayed);
            clients.push((army, stream));
        }

        reg.close();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = accept_thread.join();

        Ok((
            Self {
                clients,
                local_army: 0,
                next_turn: 0,
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
        let mut clients = Vec::with_capacity(n - 1);
        for slot in 1..n {
            let (mut stream, peer) = listener
                .accept()
                .map_err(|e| format!("accept slot {slot} failed: {e}"))?;
            let _ = stream.set_nodelay(true);
            let _ = stream.set_read_timeout(Some(NET_TIMEOUT));
            let army = slot as ArmyId;
            let mut client_hs = hs.clone();
            client_hs.assigned_army = army;
            write_frame(&mut stream, &encode_handshake(&client_hs))?;
            on_join(army, peer);
            clients.push((army, stream));
        }
        Ok(Self {
            clients,
            local_army: 0,
            next_turn: 0,
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
            let p = match read_frame(stream).and_then(|raw| decode_turn(&raw)) {
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
        let bundle = canonicalize_bundle(TurnBundle {
            turn,
            hashes,
            batches,
            controls,
        });
        let encoded = encode_turn_bundle(&bundle);
        for (army, stream) in &mut self.clients {
            if let Err(e) = write_frame(stream, &encoded) {
                // The next read fails too, so the drop is announced in the next bundle.
                eprintln!("lockstep host: send to army {army} failed: {e}");
                let _ = stream.shutdown(Shutdown::Both);
            }
        }
        self.next_turn += 1;
        Ok(bundle)
    }
}

pub struct LockstepClient {
    stream: TcpStream,
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
            stream,
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
}

impl TurnTransport for LockstepClient {
    fn exchange(&mut self, pkt: TurnPacket) -> Result<TurnBundle, String> {
        let turn = self.next_turn;
        if pkt.turn != turn {
            return Err(format!("client submitted turn {} but expected {turn}", pkt.turn));
        }
        write_frame(&mut self.stream, &encode_turn(&pkt))?;
        let raw = read_frame(&mut self.stream).map_err(|e| format!("host disconnected: {e}"))?;
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
            clients.push((army, stream));
        }
        clients.sort_by_key(|(a, _)| *a);
        Ok(Self {
            clients,
            local_army: 0,
            next_turn: 0,
        })
    }
}

impl LockstepClient {
    /// Lockstep client on a lobby connection whose [`Handshake`] was already received.
    pub fn from_lobby(stream: TcpStream, hs: &Handshake) -> Self {
        let _ = stream.set_nodelay(true);
        let _ = stream.set_read_timeout(Some(NET_TIMEOUT));
        Self {
            stream,
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
    /// One tick ran; `Some(packet)` when a turn completed and its packet must be sent.
    Stepped(Option<TurnPacket>),
    /// A synchronized pause is active: the turn advanced (null turn) but the simulation did
    /// not step; `Some(packet)` exactly like [`TickOutcome::Stepped`].
    Paused(Option<TurnPacket>),
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
        let pkt = if self.tick_in_turn >= self.turn_ticks {
            self.tick_in_turn = 0;
            self.next_exec += 1;
            Some(self.make_packet(world))
        } else {
            None
        };
        if self.turn_paused {
            TickOutcome::Paused(pkt)
        } else {
            TickOutcome::Stepped(pkt)
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
            TickOutcome::Stepped(Some(pkt)) | TickOutcome::Paused(Some(pkt)) => {
                if world.tick() >= limit || world.result().is_some() {
                    break;
                }
                let b = transport.exchange(pkt)?;
                receive_or_dump(transport, core, world, b, &mut VecDeque::new())?;
            }
            TickOutcome::Stepped(None) | TickOutcome::Paused(None) => {}
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
            TickOutcome::Stepped(pkt) | TickOutcome::Paused(pkt) => {
                n.stalled_now = false;
                s.selected.retain(|id| s.world.unit(*id).is_some());
                if let Some(pkt) = pkt {
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
                TickOutcome::Stepped(p) => pkt = p,
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
        let hs = handshake(players, armies, 4242, ticks);
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
        let clients: Vec<_> = (1..players)
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
        assert_eq!(seen, (1..players as ArmyId).collect::<Vec<_>>());
        out
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
                        TickOutcome::Stepped(p) | TickOutcome::Paused(p) => {
                            if let Some(p) = p {
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
        core.queue_control(NetControl::Pause(true));
        assert_eq!(core.prime(&w)[0].controls, vec![NetControl::Pause(true)]);
    }
}
