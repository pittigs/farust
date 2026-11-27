//! Phase 20 network lobby: the pre-game phase of lockstep protocol v3.
//!
//! Same TCP star as the game ([`crate::net`]): the host listens, every client keeps one
//! connection that is handed over to [`LockstepHost`] / [`LockstepClient`] on start.
//!
//! ```text
//! client                          host
//!   connect  ───────────────────▶ accept
//!            ◀─────────────────── Welcome{version, peer_id, host_name}
//!   Hello{version, name} ───────▶ add peer to first open slot
//!            ◀─────────────────── State{map, seed, slots, peers} (after every change)
//!   Request{Ready|Faction|Chat} ▶            Chat{from, text} broadcast
//!            ◀─────────────────── Start                 (host pressed START)
//!   StartAck ───────────────────▶ (last lobby frame on this stream)
//!            ◀─────────────────── Handshake (lockstep, `assigned_army` per peer)
//!   … turn packets / bundles (net.rs) …
//! ```
//!
//! `StartAck` is a barrier: after it neither side sends lobby frames, so the stream
//! carries only lockstep traffic. A client meeting a CLI host (`farust host`) receives
//! the `Handshake` as its first frame and starts immediately; a CLI `farust join` meeting
//! a menu host answers `Welcome` via [`auto_join`] as an always-ready peer.
//!
//! Slots and lockstep armies: the host is army 0, other peers follow in slot order, AI
//! slots come last (armies `>= player_count`, simulated on every peer).

use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::net::{
    DEFAULT_INPUT_DELAY, Handshake, LockstepClient, LockstepHost, PROTOCOL_VERSION, TURN_TICKS,
    lobby_finish_handshake, lobby_is_handshake, lobby_read_frame, lobby_write_frame,
};
use crate::replay::{read_str, read_u8, read_u32, read_u64, write_str};
use crate::settings::FACTION_KEYS;
use crate::skirmish::{ArmyId, ArmySetup, FACTION_TABLE};

pub const MSG_WELCOME: u8 = 10;
pub const MSG_HELLO: u8 = 11;
pub const MSG_STATE: u8 = 12;
pub const MSG_REQUEST: u8 = 13;
pub const MSG_CHAT: u8 = 14;
pub const MSG_START: u8 = 15;
pub const MSG_START_ACK: u8 = 16;
pub const MSG_BYE: u8 = 17;

pub const MAX_SLOTS: usize = 8;
pub const HOST_ID: u8 = 0;
const MAX_CHAT: usize = 200;
const MAX_CHAT_LEN: usize = 160;
const START_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Lobby model (host-authoritative, mirrored on clients)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetSlotKind {
    Open,
    Closed,
    Ai,
    Peer(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetSlot {
    pub kind: NetSlotKind,
    /// Index into [`FACTION_KEYS`].
    pub faction: u8,
    pub team: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetPeer {
    pub id: u8,
    pub name: String,
    pub ready: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetLobbyState {
    pub map_id: String,
    pub map_name: String,
    pub seed: u64,
    /// Human players including the host (`2..=8`).
    pub max_players: u8,
    pub slots: Vec<NetSlot>,
    pub peers: Vec<NetPeer>,
    /// Phase 51: host-enabled mods advertised in the lobby (auto-synced at match start).
    pub mods: Vec<crate::mods::ModIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatLine {
    /// Empty = system message.
    pub from: String,
    pub text: String,
}

impl NetLobbyState {
    pub fn new(map_id: &str, map_name: &str, spots: usize, seed: u64, max_players: u8, host_name: &str, host_faction: u8) -> Self {
        let mut s = Self {
            map_id: map_id.to_string(),
            map_name: map_name.to_string(),
            seed,
            max_players: max_players.clamp(2, MAX_SLOTS as u8),
            slots: Vec::new(),
            peers: vec![NetPeer { id: HOST_ID, name: host_name.to_string(), ready: true }],
            mods: crate::mods::active_identities(),
        };
        s.resize_slots(spots);
        s.slots[0] = NetSlot { kind: NetSlotKind::Peer(HOST_ID), faction: host_faction % 4, team: 0 };
        s
    }

    fn resize_slots(&mut self, spots: usize) {
        let n = spots.clamp(2, MAX_SLOTS);
        while self.slots.len() < n {
            let i = self.slots.len();
            self.slots.push(NetSlot { kind: NetSlotKind::Open, faction: (i % 4) as u8, team: i as u8 });
        }
        // Peers in removed slots move to free ones (or lose their slot: kicked by caller).
        let removed: Vec<NetSlot> = self.slots.drain(n..).collect();
        for s in removed {
            if let NetSlotKind::Peer(id) = s.kind
                && let Some(free) = self.first_free()
            {
                self.slots[free] = NetSlot { kind: NetSlotKind::Peer(id), faction: s.faction, team: self.slots[free].team };
            }
        }
    }

    fn first_free(&self) -> Option<usize> {
        self.slots
            .iter()
            .position(|s| s.kind == NetSlotKind::Open)
            .or_else(|| self.slots.iter().position(|s| s.kind == NetSlotKind::Ai))
    }

    /// Switches maps; returns peers that no longer fit (they should be dropped).
    pub fn set_map(&mut self, map_id: &str, map_name: &str, spots: usize) -> Vec<u8> {
        self.map_id = map_id.to_string();
        self.map_name = map_name.to_string();
        self.resize_slots(spots);
        let seated: Vec<u8> = self.peers.iter().map(|p| p.id).filter(|id| self.slot_of(*id).is_some()).collect();
        self.peers.iter().map(|p| p.id).filter(|id| !seated.contains(id)).collect()
    }

    pub fn peer(&self, id: u8) -> Option<&NetPeer> {
        self.peers.iter().find(|p| p.id == id)
    }

    pub fn slot_of(&self, id: u8) -> Option<usize> {
        self.slots.iter().position(|s| s.kind == NetSlotKind::Peer(id))
    }

    /// Seats a new peer in the first open (else AI) slot.
    pub fn add_peer(&mut self, id: u8, name: &str) -> Result<usize, String> {
        if self.peers.len() >= self.max_players as usize {
            return Err(format!("lobby is full ({} players)", self.max_players));
        }
        let slot = self.first_free().ok_or("no free slot on this map")?;
        let mut unique = name.to_string();
        let mut n = 2;
        while self.peers.iter().any(|p| p.name.eq_ignore_ascii_case(&unique)) {
            unique = format!("{name} {n}");
            n += 1;
        }
        self.peers.push(NetPeer { id, name: unique, ready: false });
        self.slots[slot].kind = NetSlotKind::Peer(id);
        Ok(slot)
    }

    pub fn remove_peer(&mut self, id: u8) -> Option<NetPeer> {
        if let Some(i) = self.slot_of(id) {
            self.slots[i].kind = NetSlotKind::Open;
        }
        let idx = self.peers.iter().position(|p| p.id == id)?;
        Some(self.peers.remove(idx))
    }

    /// Host: Open → AI → Closed → Open (occupied slots are not changed).
    pub fn cycle_kind(&mut self, slot: usize) {
        if let Some(s) = self.slots.get_mut(slot) {
            s.kind = match s.kind {
                NetSlotKind::Open => NetSlotKind::Ai,
                NetSlotKind::Ai => NetSlotKind::Closed,
                NetSlotKind::Closed => NetSlotKind::Open,
                p @ NetSlotKind::Peer(_) => p,
            };
        }
    }

    pub fn cycle_faction(&mut self, slot: usize) {
        if let Some(s) = self.slots.get_mut(slot) {
            s.faction = (s.faction + 1) % FACTION_KEYS.len() as u8;
        }
    }

    pub fn cycle_team(&mut self, slot: usize) {
        let n = self.slots.len().max(2) as u8;
        if let Some(s) = self.slots.get_mut(slot) {
            s.team = (s.team + 1) % n;
        }
    }

    /// Host: moves the occupant of `slot` to the next non-occupied slot (swapping kinds,
    /// keeping each spot's team).
    pub fn move_slot(&mut self, slot: usize) {
        let n = self.slots.len();
        if slot >= n || !matches!(self.slots[slot].kind, NetSlotKind::Peer(_)) {
            return;
        }
        for k in 1..n {
            let to = (slot + k) % n;
            if !matches!(self.slots[to].kind, NetSlotKind::Peer(_)) {
                let (a, b) = (self.slots[slot], self.slots[to]);
                self.slots[slot] = NetSlot { kind: b.kind, faction: b.faction, team: a.team };
                self.slots[to] = NetSlot { kind: a.kind, faction: a.faction, team: b.team };
                return;
            }
        }
    }

    pub fn set_ready(&mut self, id: u8, ready: bool) {
        if let Some(p) = self.peers.iter_mut().find(|p| p.id == id) {
            p.ready = ready || id == HOST_ID;
        }
    }

    pub fn set_peer_faction(&mut self, id: u8, faction: u8) {
        if let Some(i) = self.slot_of(id) {
            self.slots[i].faction = faction % FACTION_KEYS.len() as u8;
        }
    }

    #[allow(dead_code)]
    pub fn army_count(&self) -> usize {
        self.slots.iter().filter(|s| matches!(s.kind, NetSlotKind::Peer(_) | NetSlotKind::Ai)).count()
    }

    /// Why START is not possible yet (`Ok` = all good).
    pub fn start_check(&self) -> Result<(), String> {
        if self.peers.len() < 2 {
            return Err("waiting for players to join".into());
        }
        if let Some(p) = self.peers.iter().find(|p| p.id != HOST_ID && !p.ready) {
            return Err(format!("{} is not ready", p.name));
        }
        if let Some(p) = self.peers.iter().find(|p| self.slot_of(p.id).is_none()) {
            return Err(format!("{} has no slot", p.name));
        }
        let teams: Vec<u8> = self
            .slots
            .iter()
            .filter(|s| matches!(s.kind, NetSlotKind::Peer(_) | NetSlotKind::Ai))
            .map(|s| s.team)
            .collect();
        if teams.len() < 2 {
            return Err("need at least 2 armies".into());
        }
        if teams.iter().all(|t| *t == teams[0]) {
            return Err("all armies are on one team".into());
        }
        Ok(())
    }

    /// Builds the lockstep handshake (host view, `assigned_army` 0) from the map's start
    /// spots `(ARMY_n name, sim position)`; returns `(peer id, army)` for every human.
    pub fn build_handshake(&self, starts: &[(String, (i64, i64))]) -> Result<(Handshake, Vec<(u8, ArmyId)>), String> {
        self.start_check()?;
        let mut order: Vec<usize> = Vec::new();
        if let Some(h) = self.slot_of(HOST_ID) {
            order.push(h);
        }
        for (i, s) in self.slots.iter().enumerate() {
            if matches!(s.kind, NetSlotKind::Peer(id) if id != HOST_ID) {
                order.push(i);
            }
        }
        let player_count = order.len();
        order.extend(self.slots.iter().enumerate().filter(|(_, s)| s.kind == NetSlotKind::Ai).map(|(i, _)| i));
        let mut setups = Vec::with_capacity(order.len());
        let mut peers = Vec::new();
        for (army, &slot) in order.iter().enumerate() {
            let s = self.slots[slot];
            let (name, start) = starts.get(slot).cloned().ok_or_else(|| format!("map has no start spot {}", slot + 1))?;
            let key = FACTION_KEYS[s.faction as usize % FACTION_KEYS.len()];
            let faction = FACTION_TABLE
                .iter()
                .position(|(k, _)| k.eq_ignore_ascii_case(key))
                .ok_or_else(|| format!("unknown faction {key}"))?;
            setups.push(ArmySetup { name, faction, start, team: s.team });
            if let NetSlotKind::Peer(id) = s.kind {
                peers.push((id, army as ArmyId));
            }
        }
        Ok((
            Handshake {
                observers: 0,
                map_id: self.map_id.clone(),
                seed: self.seed,
                setups,
                player_count,
                turn_ticks: TURN_TICKS,
                input_delay: DEFAULT_INPUT_DELAY,
                tick_limit: 0,
                assigned_army: 0,
                mods: self.mods.clone(),
            },
            peers,
        ))
    }
}

// ---------------------------------------------------------------------------
// Messages & codec
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientRequest {
    Ready(bool),
    Faction(u8),
    Chat(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LobbyMsg {
    Welcome { version: u32, peer_id: u8, host_name: String },
    Hello { version: u32, name: String },
    State(NetLobbyState),
    Request(ClientRequest),
    Chat(ChatLine),
    Start,
    StartAck,
    Bye { reason: String },
}

fn encode_state(out: &mut Vec<u8>, s: &NetLobbyState) {
    write_str(out, &s.map_id);
    write_str(out, &s.map_name);
    out.extend_from_slice(&s.seed.to_le_bytes());
    out.push(s.max_players);
    out.push(s.slots.len().min(MAX_SLOTS) as u8);
    for slot in s.slots.iter().take(MAX_SLOTS) {
        match slot.kind {
            NetSlotKind::Open => out.push(0),
            NetSlotKind::Closed => out.push(1),
            NetSlotKind::Ai => out.push(2),
            NetSlotKind::Peer(id) => out.extend_from_slice(&[3, id]),
        }
        out.push(slot.faction);
        out.push(slot.team);
    }
    out.push(s.peers.len().min(16) as u8);
    for p in s.peers.iter().take(16) {
        out.push(p.id);
        write_str(out, &p.name);
        out.push(p.ready as u8);
    }
    crate::mods::encode_identities(out, &s.mods);
}

fn decode_state(d: &[u8], pos: &mut usize) -> Result<NetLobbyState, String> {
    let map_id = read_str(d, pos)?;
    let map_name = read_str(d, pos)?;
    let seed = read_u64(d, pos)?;
    let max_players = read_u8(d, pos)?;
    let ns = read_u8(d, pos)? as usize;
    if ns > MAX_SLOTS {
        return Err(format!("lobby state lists {ns} slots"));
    }
    let mut slots = Vec::with_capacity(ns);
    for _ in 0..ns {
        let kind = match read_u8(d, pos)? {
            0 => NetSlotKind::Open,
            1 => NetSlotKind::Closed,
            2 => NetSlotKind::Ai,
            3 => NetSlotKind::Peer(read_u8(d, pos)?),
            k => return Err(format!("bad slot kind {k}")),
        };
        let faction = read_u8(d, pos)?;
        let team = read_u8(d, pos)?;
        slots.push(NetSlot { kind, faction, team });
    }
    let np = read_u8(d, pos)? as usize;
    let mut peers = Vec::with_capacity(np);
    for _ in 0..np {
        let id = read_u8(d, pos)?;
        let name = read_str(d, pos)?;
        let ready = read_u8(d, pos)? != 0;
        peers.push(NetPeer { id, name, ready });
    }
    let mods = crate::mods::decode_identities(d, pos)?;
    Ok(NetLobbyState { map_id, map_name, seed, max_players, slots, peers, mods })
}

pub fn encode_lobby_msg(m: &LobbyMsg) -> Vec<u8> {
    let mut out = Vec::with_capacity(64);
    match m {
        LobbyMsg::Welcome { version, peer_id, host_name } => {
            out.push(MSG_WELCOME);
            out.extend_from_slice(&version.to_le_bytes());
            out.push(*peer_id);
            write_str(&mut out, host_name);
        }
        LobbyMsg::Hello { version, name } => {
            out.push(MSG_HELLO);
            out.extend_from_slice(&version.to_le_bytes());
            write_str(&mut out, name);
        }
        LobbyMsg::State(s) => {
            out.push(MSG_STATE);
            encode_state(&mut out, s);
        }
        LobbyMsg::Request(r) => {
            out.push(MSG_REQUEST);
            match r {
                ClientRequest::Ready(b) => out.extend_from_slice(&[0, *b as u8]),
                ClientRequest::Faction(f) => out.extend_from_slice(&[1, *f]),
                ClientRequest::Chat(t) => {
                    out.push(2);
                    write_str(&mut out, t);
                }
            }
        }
        LobbyMsg::Chat(c) => {
            out.push(MSG_CHAT);
            write_str(&mut out, &c.from);
            write_str(&mut out, &c.text);
        }
        LobbyMsg::Start => out.push(MSG_START),
        LobbyMsg::StartAck => out.push(MSG_START_ACK),
        LobbyMsg::Bye { reason } => {
            out.push(MSG_BYE);
            write_str(&mut out, reason);
        }
    }
    out
}

pub fn decode_lobby_msg(d: &[u8]) -> Result<LobbyMsg, String> {
    let mut pos = 0;
    let m = match read_u8(d, &mut pos)? {
        MSG_WELCOME => LobbyMsg::Welcome {
            version: read_u32(d, &mut pos)?,
            peer_id: read_u8(d, &mut pos)?,
            host_name: read_str(d, &mut pos)?,
        },
        MSG_HELLO => LobbyMsg::Hello { version: read_u32(d, &mut pos)?, name: read_str(d, &mut pos)? },
        MSG_STATE => LobbyMsg::State(decode_state(d, &mut pos)?),
        MSG_REQUEST => LobbyMsg::Request(match read_u8(d, &mut pos)? {
            0 => ClientRequest::Ready(read_u8(d, &mut pos)? != 0),
            1 => ClientRequest::Faction(read_u8(d, &mut pos)?),
            2 => ClientRequest::Chat(read_str(d, &mut pos)?),
            k => return Err(format!("bad lobby request kind {k}")),
        }),
        MSG_CHAT => LobbyMsg::Chat(ChatLine { from: read_str(d, &mut pos)?, text: read_str(d, &mut pos)? }),
        MSG_START => LobbyMsg::Start,
        MSG_START_ACK => LobbyMsg::StartAck,
        MSG_BYE => LobbyMsg::Bye { reason: read_str(d, &mut pos)? },
        t => return Err(format!("unexpected lobby message type {t}")),
    };
    if pos != d.len() {
        return Err(format!("lobby message has {} trailing bytes", d.len() - pos));
    }
    Ok(m)
}

fn send(stream: &mut TcpStream, m: &LobbyMsg) -> Result<(), String> {
    lobby_write_frame(stream, &encode_lobby_msg(m))
}

fn clean_text(t: &str, max: usize) -> String {
    t.chars().filter(|c| !c.is_control()).take(max).collect::<String>().trim().to_string()
}

fn push_chat(log: &mut Vec<ChatLine>, line: ChatLine) {
    log.push(line);
    if log.len() > MAX_CHAT {
        log.drain(..log.len() - MAX_CHAT);
    }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

enum HostEvent {
    Accepted {
        stream: TcpStream,
        addr: SocketAddr,
        relayed: bool,
    },
    Frame(u8, LobbyMsg),
    Gone(u8, String),
    Acked(u8),
}

struct HostConn {
    id: u8,
    #[allow(dead_code)]
    addr: SocketAddr,
    relayed: bool,
    stream: TcpStream,
    joined: bool,
    acked: bool,
}

#[derive(Debug)]
pub enum HostPoll {
    Idle,
    Changed,
    /// Every peer acknowledged START: build the match with [`NetHostLaunch::into_transport`].
    Launch(NetHostLaunch),
    /// The start barrier failed; the lobby is closed.
    Failed(String),
}

#[derive(Debug)]
pub struct NetHostLaunch {
    pub state: NetLobbyState,
    peers: Vec<(u8, TcpStream)>,
}

impl NetHostLaunch {
    /// Sends every peer its handshake and returns the host handshake + lockstep transport.
    pub fn into_transport(self, starts: &[(String, (i64, i64))]) -> Result<(Handshake, LockstepHost), String> {
        let (hs, armies) = self.state.build_handshake(starts)?;
        let mut streams = Vec::new();
        let mut peers = self.peers;
        for (id, army) in armies {
            if id == HOST_ID {
                continue;
            }
            let i = peers.iter().position(|(p, _)| *p == id).ok_or_else(|| format!("peer {id} has no connection"))?;
            streams.push((army, peers.swap_remove(i).1));
        }
        let host = LockstepHost::from_lobby(streams, &hs)?;
        Ok((hs, host))
    }
}

pub struct NetLobbyHost {
    pub state: NetLobbyState,
    pub chat: Vec<ChatLine>,
    pub local_addr: SocketAddr,
    pub invite_code: Option<String>,
    conns: Vec<HostConn>,
    tx: Sender<HostEvent>,
    rx: Receiver<HostEvent>,
    stop: Arc<AtomicBool>,
    accept: Option<JoinHandle<()>>,
    registration: Option<crate::rendezvous::HostRegistration>,
    next_id: u8,
    starting: Option<Instant>,
    closed: bool,
}

impl NetLobbyHost {
    pub fn bind(addr: &str, state: NetLobbyState) -> Result<Self, String> {
        let listener = TcpListener::bind(addr).map_err(|e| format!("cannot listen on {addr}: {e}"))?;
        let local_addr = listener.local_addr().map_err(|e| format!("local_addr: {e}"))?;
        listener.set_nonblocking(true).map_err(|e| format!("set_nonblocking: {e}"))?;
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let (atx, astop) = (tx.clone(), stop.clone());
        let accept = std::thread::spawn(move || {
            while !astop.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((s, a)) => {
                        let _ = s.set_nonblocking(false);
                        if atx.send(HostEvent::Accepted { stream: s, addr: a, relayed: false }).is_err() {
                            return;
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(20)),
                    Err(_) => std::thread::sleep(Duration::from_millis(100)),
                }
            }
        });
        let mut chat = Vec::new();
        push_chat(&mut chat, ChatLine { from: String::new(), text: format!("Lobby open on {local_addr}") });
        Ok(Self {
            state,
            chat,
            local_addr,
            invite_code: None,
            conns: Vec::new(),
            tx,
            rx,
            stop,
            accept: Some(accept),
            registration: None,
            next_id: 1,
            starting: None,
            closed: false,
        })
    }

    /// Registers an invite code with `rendezvous_addr` so friends can join by code with
    /// automatic TCP relay fallback if direct UDP/TCP cannot pierce NAT.
    pub fn register_invite(&mut self, rendezvous_addr: &str, timeout: Duration) -> Result<String, String> {
        let host_name = self.state.peer(HOST_ID).map(|p| p.name.clone()).unwrap_or_default();
        let map_id = self.state.map_id.clone();
        let tx = self.tx.clone();
        let reg = crate::rendezvous::HostRegistration::register(
            rendezvous_addr,
            self.local_addr,
            &host_name,
            &map_id,
            timeout,
            Box::new(move |stream, addr| {
                let _ = tx.send(HostEvent::Accepted { stream, addr, relayed: true });
            }),
        )?;
        let code = reg.code.clone();
        self.invite_code = Some(code.clone());
        self.registration = Some(reg);
        push_chat(&mut self.chat, ChatLine { from: String::new(), text: format!("Invite code: {code}") });
        Ok(code)
    }

    #[allow(dead_code)]
    pub fn is_starting(&self) -> bool {
        self.starting.is_some()
    }

    /// Remote address per peer id (for the lobby peer list).
    #[allow(dead_code)]
    pub fn peer_addr(&self, id: u8) -> Option<SocketAddr> {
        self.conns.iter().find(|c| c.id == id).map(|c| c.addr)
    }

    pub fn peer_is_relayed(&self, id: u8) -> Option<bool> {
        self.conns.iter().find(|c| c.id == id).map(|c| c.relayed)
    }

    fn broadcast(&mut self, m: &LobbyMsg) {
        let frame = encode_lobby_msg(m);
        let mut dead = Vec::new();
        for c in self.conns.iter_mut().filter(|c| c.joined) {
            if let Err(e) = lobby_write_frame(&mut c.stream, &frame) {
                dead.push((c.id, e));
            }
        }
        for (id, e) in dead {
            let _ = self.tx.send(HostEvent::Gone(id, e));
        }
    }

    fn broadcast_state(&mut self) {
        let m = LobbyMsg::State(self.state.clone());
        self.broadcast(&m);
    }

    fn system(&mut self, text: String) {
        let line = ChatLine { from: String::new(), text };
        push_chat(&mut self.chat, line.clone());
        self.broadcast(&LobbyMsg::Chat(line));
    }

    fn drop_conn(&mut self, id: u8, reason: Option<&str>) {
        if let Some(i) = self.conns.iter().position(|c| c.id == id) {
            let mut c = self.conns.remove(i);
            if let Some(r) = reason {
                let _ = send(&mut c.stream, &LobbyMsg::Bye { reason: r.to_string() });
            }
            let _ = c.stream.shutdown(Shutdown::Both);
        }
    }

    /// Applies a host edit (slots, factions, teams, seed, map) and broadcasts it.
    pub fn edit(&mut self, f: impl FnOnce(&mut NetLobbyState) -> Vec<u8>) {
        if self.starting.is_some() || self.closed {
            return;
        }
        let evicted = f(&mut self.state);
        for id in evicted {
            if let Some(p) = self.state.remove_peer(id) {
                self.drop_conn(id, Some("no slot left on the new map"));
                self.system(format!("{} was removed (map too small)", p.name));
            }
        }
        self.broadcast_state();
    }

    pub fn send_chat(&mut self, text: &str) {
        let text = clean_text(text, MAX_CHAT_LEN);
        if text.is_empty() || self.starting.is_some() {
            return;
        }
        let from = self.state.peer(HOST_ID).map(|p| p.name.clone()).unwrap_or_default();
        let line = ChatLine { from, text };
        push_chat(&mut self.chat, line.clone());
        self.broadcast(&LobbyMsg::Chat(line));
    }

    pub fn kick(&mut self, id: u8) {
        if id == HOST_ID || self.starting.is_some() {
            return;
        }
        if let Some(p) = self.state.remove_peer(id) {
            self.drop_conn(id, Some("kicked by host"));
            self.system(format!("{} was kicked", p.name));
            self.broadcast_state();
        }
    }

    /// Sends `Start` to every peer; [`Self::poll`] returns `Launch` once all acknowledged.
    pub fn begin_start(&mut self) -> Result<(), String> {
        if self.starting.is_some() {
            return Ok(());
        }
        self.state.start_check()?;
        // Unjoined connections (no Hello yet) are turned away.
        let pending: Vec<u8> = self.conns.iter().filter(|c| !c.joined).map(|c| c.id).collect();
        for id in pending {
            self.drop_conn(id, Some("game is starting"));
        }
        self.system("Game starting...".into());
        self.broadcast(&LobbyMsg::Start);
        self.starting = Some(Instant::now());
        Ok(())
    }

    pub fn poll(&mut self) -> HostPoll {
        if self.closed {
            return HostPoll::Idle;
        }
        let mut changed = false;
        while let Ok(ev) = self.rx.try_recv() {
            changed = true;
            match ev {
                HostEvent::Accepted { stream: mut s, addr, relayed } => {
                    let full = self.conns.len() + 1 >= self.state.max_players as usize;
                    if self.starting.is_some() || full {
                        let reason = if full { "lobby is full" } else { "game is starting" };
                        let _ = send(&mut s, &LobbyMsg::Bye { reason: reason.into() });
                        let _ = s.shutdown(Shutdown::Both);
                        continue;
                    }
                    let id = self.next_id;
                    self.next_id = if self.next_id >= 250 { 1 } else { self.next_id + 1 };
                    let _ = s.set_nodelay(true);
                    let _ = s.set_write_timeout(Some(Duration::from_secs(3)));
                    let _ = s.set_read_timeout(None);
                    let host_name = self.state.peer(HOST_ID).map(|p| p.name.clone()).unwrap_or_default();
                    let welcome = LobbyMsg::Welcome { version: PROTOCOL_VERSION, peer_id: id, host_name };
                    let Ok(reader) = s.try_clone() else {
                        continue;
                    };
                    if send(&mut s, &welcome).is_err() {
                        continue;
                    }
                    let tx = self.tx.clone();
                    std::thread::spawn(move || host_reader(id, reader, tx));
                    self.conns.push(HostConn { id, addr, relayed, stream: s, joined: false, acked: false });
                }
                HostEvent::Frame(id, m) => self.on_frame(id, m),
                HostEvent::Gone(id, e) => {
                    let joined = self.conns.iter().any(|c| c.id == id && c.joined);
                    self.drop_conn(id, None);
                    if joined && let Some(p) = self.state.remove_peer(id) {
                        let why = if e.contains("bye") { String::new() } else { format!(" ({e})") };
                        self.system(format!("{} left{why}", p.name));
                        self.broadcast_state();
                    }
                }
                HostEvent::Acked(id) => {
                    if let Some(c) = self.conns.iter_mut().find(|c| c.id == id) {
                        c.acked = true;
                    }
                }
            }
        }
        if let Some(t0) = self.starting {
            let joined: Vec<&HostConn> = self.conns.iter().filter(|c| c.joined).collect();
            if joined.is_empty() {
                self.starting = None;
                self.system("Start cancelled: all players left".into());
                return HostPoll::Changed;
            }
            if joined.iter().all(|c| c.acked) {
                return HostPoll::Launch(self.take_launch());
            }
            if t0.elapsed() > START_TIMEOUT {
                let slow: Vec<String> = joined
                    .iter()
                    .filter(|c| !c.acked)
                    .filter_map(|c| self.state.peer(c.id).map(|p| p.name.clone()))
                    .collect();
                self.close("start timed out");
                return HostPoll::Failed(format!("no start acknowledgement from {}", slow.join(", ")));
            }
        }
        if changed { HostPoll::Changed } else { HostPoll::Idle }
    }

    fn on_frame(&mut self, id: u8, m: LobbyMsg) {
        let joined = self.conns.iter().any(|c| c.id == id && c.joined);
        match m {
            LobbyMsg::Hello { version, name } if !joined => {
                if version != PROTOCOL_VERSION {
                    self.drop_conn(id, Some(&format!("protocol mismatch: host v{PROTOCOL_VERSION}, you v{version}")));
                    return;
                }
                let name = crate::settings::validate_name(&clean_text(&name, 40)).unwrap_or_else(|_| format!("Player {id}"));
                match self.state.add_peer(id, &name) {
                    Ok(_) => {
                        if let Some(c) = self.conns.iter_mut().find(|c| c.id == id) {
                            c.joined = true;
                        }
                        let shown = self.state.peer(id).map(|p| p.name.clone()).unwrap_or(name);
                        self.broadcast_state();
                        // Catch the newcomer up on recent chat.
                        let recent: Vec<ChatLine> = self.chat.iter().rev().take(20).rev().cloned().collect();
                        if let Some(c) = self.conns.iter_mut().find(|c| c.id == id) {
                            for line in recent {
                                let _ = send(&mut c.stream, &LobbyMsg::Chat(line));
                            }
                        }
                        self.system(format!("{shown} joined"));
                    }
                    Err(e) => self.drop_conn(id, Some(&e)),
                }
            }
            LobbyMsg::Request(r) if joined && self.starting.is_none() => match r {
                ClientRequest::Ready(b) => {
                    self.state.set_ready(id, b);
                    self.broadcast_state();
                }
                ClientRequest::Faction(f) => {
                    self.state.set_peer_faction(id, f);
                    self.broadcast_state();
                }
                ClientRequest::Chat(t) => {
                    let text = clean_text(&t, MAX_CHAT_LEN);
                    if !text.is_empty() {
                        let from = self.state.peer(id).map(|p| p.name.clone()).unwrap_or_default();
                        let line = ChatLine { from, text };
                        push_chat(&mut self.chat, line.clone());
                        self.broadcast(&LobbyMsg::Chat(line));
                    }
                }
            },
            LobbyMsg::Bye { .. } => {
                let _ = self.tx.send(HostEvent::Gone(id, "bye".into()));
            }
            _ => {}
        }
    }

    fn take_launch(&mut self) -> NetHostLaunch {
        self.stop.store(true, Ordering::Relaxed);
        self.closed = true;
        if let Some(mut r) = self.registration.take() {
            r.close();
        }
        let peers = self.conns.drain(..).filter(|c| c.joined).map(|c| (c.id, c.stream)).collect();
        NetHostLaunch { state: self.state.clone(), peers }
    }

    /// Closes the lobby, telling every peer why.
    pub fn close(&mut self, reason: &str) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.stop.store(true, Ordering::Relaxed);
        if let Some(mut r) = self.registration.take() {
            r.close();
        }
        let ids: Vec<u8> = self.conns.iter().map(|c| c.id).collect();
        for id in ids {
            self.drop_conn(id, Some(reason));
        }
    }
}

impl Drop for NetLobbyHost {
    fn drop(&mut self) {
        self.close("host left the lobby");
        if let Some(h) = self.accept.take() {
            let _ = h.join();
        }
    }
}

fn host_reader(id: u8, mut stream: TcpStream, tx: Sender<HostEvent>) {
    loop {
        let ev = match lobby_read_frame(&mut stream).and_then(|raw| decode_lobby_msg(&raw)) {
            Ok(LobbyMsg::StartAck) => {
                let _ = tx.send(HostEvent::Acked(id));
                return;
            }
            Ok(m @ LobbyMsg::Bye { .. }) => {
                let _ = tx.send(HostEvent::Frame(id, m));
                return;
            }
            Ok(m) => HostEvent::Frame(id, m),
            Err(e) => {
                let _ = tx.send(HostEvent::Gone(id, e));
                return;
            }
        };
        if tx.send(ev).is_err() {
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

enum ClientEvent {
    Connected { relayed: bool },
    Welcome { peer_id: u8, host_name: String },
    Msg(LobbyMsg),
    Handshake(Handshake),
    Gone(String),
}

#[derive(Default)]
struct Writer {
    stream: Option<TcpStream>,
    /// Set (under the lock) when `StartAck` was sent: no more lobby frames after it.
    started: bool,
}

pub enum ClientPoll {
    Idle,
    Changed,
    Launch(Handshake, LockstepClient),
    Closed(String),
}

pub struct NetLobbyClient {
    pub addr: String,
    pub peer_id: Option<u8>,
    pub host_name: String,
    pub via_relay: Option<bool>,
    pub state: Option<NetLobbyState>,
    pub chat: Vec<ChatLine>,
    pub ready: bool,
    pub starting: bool,
    writer: Arc<Mutex<Writer>>,
    rx: Receiver<ClientEvent>,
    done: bool,
}

fn connect_any(addr: &str, timeout: Duration) -> Result<TcpStream, String> {
    let addrs: Vec<SocketAddr> = addr
        .to_socket_addrs()
        .map_err(|e| format!("cannot resolve {addr}: {e}"))?
        .collect();
    let mut last = format!("no address for {addr}");
    for a in addrs {
        match TcpStream::connect_timeout(&a, timeout) {
            Ok(s) => return Ok(s),
            Err(e) => last = format!("cannot connect to {a}: {e}"),
        }
    }
    Err(last)
}

impl NetLobbyClient {
    /// Connects in the background; progress arrives through [`Self::poll`].
    pub fn connect(addr: &str, name: &str) -> Self {
        let (tx, rx) = mpsc::channel();
        let writer = Arc::new(Mutex::new(Writer::default()));
        let (a, n, w) = (addr.to_string(), name.to_string(), writer.clone());
        std::thread::spawn(move || client_thread(a, n, w, tx));
        Self {
            addr: addr.to_string(),
            peer_id: None,
            host_name: String::new(),
            via_relay: None,
            state: None,
            chat: Vec::new(),
            ready: false,
            starting: false,
            writer,
            rx,
            done: false,
        }
    }

    /// Connects via a Phase 26 invite code (`XXXX-XXXX`), trying direct TCP first and
    /// falling back automatically to the rendezvous relay tunnel.
    pub fn connect_code(rendezvous_addr: &str, code: &str, force_relay: bool, name: &str) -> Self {
        let (tx, rx) = mpsc::channel();
        let writer = Arc::new(Mutex::new(Writer::default()));
        let rv = rendezvous_addr.to_string();
        let cd = code.to_string();
        let nm = name.to_string();
        let wr = writer.clone();
        std::thread::spawn(move || {
            match crate::rendezvous::connect_via_code(&rv, &cd, force_relay, Duration::from_millis(800)) {
                Ok((stream, via)) => {
                    let _ = tx.send(ClientEvent::Connected { relayed: via.is_relayed() });
                    run_client_stream(stream, nm, wr, tx);
                }
                Err(e) => {
                    let _ = tx.send(ClientEvent::Gone(e));
                }
            }
        });
        Self {
            addr: format!("code:{}", crate::rendezvous::normalize_code(code).unwrap_or_else(|| code.to_string())),
            peer_id: None,
            host_name: String::new(),
            via_relay: None,
            state: None,
            chat: Vec::new(),
            ready: false,
            starting: false,
            writer,
            rx,
            done: false,
        }
    }

    #[allow(dead_code)]
    pub fn connected(&self) -> bool {
        self.peer_id.is_some()
    }

    fn request(&mut self, r: ClientRequest) {
        if let Ok(mut w) = self.writer.lock()
            && !w.started
            && let Some(s) = w.stream.as_mut()
        {
            let _ = send(s, &LobbyMsg::Request(r));
        }
    }

    pub fn set_ready(&mut self, ready: bool) {
        self.ready = ready;
        self.request(ClientRequest::Ready(ready));
    }

    pub fn set_faction(&mut self, faction: u8) {
        self.request(ClientRequest::Faction(faction));
    }

    pub fn send_chat(&mut self, text: &str) {
        let t = clean_text(text, MAX_CHAT_LEN);
        if !t.is_empty() {
            self.request(ClientRequest::Chat(t));
        }
    }

    pub fn leave(&mut self) {
        if let Ok(mut w) = self.writer.lock()
            && let Some(mut s) = w.stream.take()
        {
            if !w.started {
                let _ = send(&mut s, &LobbyMsg::Bye { reason: "left".into() });
            }
            let _ = s.shutdown(Shutdown::Both);
        }
        self.done = true;
    }

    pub fn poll(&mut self) -> ClientPoll {
        if self.done {
            return ClientPoll::Idle;
        }
        let mut changed = false;
        while let Ok(ev) = self.rx.try_recv() {
            changed = true;
            match ev {
                ClientEvent::Connected { relayed } => self.via_relay = Some(relayed),
                ClientEvent::Welcome { peer_id, host_name } => {
                    self.peer_id = Some(peer_id);
                    self.host_name = host_name;
                    if self.via_relay.is_none() {
                        self.via_relay = Some(false);
                    }
                }
                ClientEvent::Msg(LobbyMsg::State(s)) => {
                    if let Some(me) = self.peer_id.and_then(|id| s.peer(id)) {
                        self.ready = me.ready;
                    }
                    self.state = Some(s);
                }
                ClientEvent::Msg(LobbyMsg::Chat(c)) => push_chat(&mut self.chat, c),
                ClientEvent::Msg(LobbyMsg::Start) => self.starting = true,
                ClientEvent::Msg(LobbyMsg::Bye { reason }) => {
                    self.leave();
                    return ClientPoll::Closed(reason);
                }
                ClientEvent::Msg(_) => {}
                ClientEvent::Gone(e) => {
                    self.leave();
                    return ClientPoll::Closed(e);
                }
                ClientEvent::Handshake(hs) => {
                    let stream = self.writer.lock().ok().and_then(|mut w| {
                        w.started = true;
                        w.stream.take()
                    });
                    self.done = true;
                    return match stream {
                        Some(s) => ClientPoll::Launch(hs.clone(), LockstepClient::from_lobby(s, &hs)),
                        None => ClientPoll::Closed("connection lost at start".into()),
                    };
                }
            }
        }
        if changed { ClientPoll::Changed } else { ClientPoll::Idle }
    }
}

impl Drop for NetLobbyClient {
    fn drop(&mut self) {
        if !self.done {
            self.leave();
        }
    }
}

fn client_thread(addr: String, name: String, writer: Arc<Mutex<Writer>>, tx: Sender<ClientEvent>) {
    let stream = match connect_any(&addr, Duration::from_secs(5)) {
        Ok(s) => s,
        Err(e) => {
            let _ = tx.send(ClientEvent::Gone(e));
            return;
        }
    };
    let _ = tx.send(ClientEvent::Connected { relayed: false });
    run_client_stream(stream, name, writer, tx);
}

fn run_client_stream(mut stream: TcpStream, name: String, writer: Arc<Mutex<Writer>>, tx: Sender<ClientEvent>) {
    let fail = |tx: &Sender<ClientEvent>, e: String| {
        let _ = tx.send(ClientEvent::Gone(e));
    };
    let _ = stream.set_nodelay(true);
    let _ = stream.set_write_timeout(Some(Duration::from_secs(3)));
    let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
    let first = match lobby_read_frame(&mut stream) {
        Ok(f) => f,
        Err(e) => return fail(&tx, format!("no greeting from host: {e}")),
    };
    let install = |s: &TcpStream| -> bool {
        match (s.try_clone(), writer.lock()) {
            (Ok(c), Ok(mut w)) => {
                w.stream = Some(c);
                true
            }
            _ => false,
        }
    };
    // CLI host (`farust host`): the handshake comes first, sync mods & start right away.
    if lobby_is_handshake(&first) {
        match lobby_finish_handshake(&mut stream, &first) {
            Ok(hs) if install(&stream) => {
                let _ = tx.send(ClientEvent::Handshake(hs));
            }
            Ok(_) => fail(&tx, "socket clone failed".into()),
            Err(e) => fail(&tx, e),
        }
        return;
    }
    match decode_lobby_msg(&first) {
        Ok(LobbyMsg::Welcome { version, peer_id, host_name }) => {
            if version != PROTOCOL_VERSION {
                return fail(&tx, format!("protocol mismatch: host v{version}, local v{PROTOCOL_VERSION}"));
            }
            if let Err(e) = send(&mut stream, &LobbyMsg::Hello { version: PROTOCOL_VERSION, name }) {
                return fail(&tx, e);
            }
            if !install(&stream) {
                return fail(&tx, "socket clone failed".into());
            }
            let _ = tx.send(ClientEvent::Welcome { peer_id, host_name });
        }
        Ok(LobbyMsg::Bye { reason }) => return fail(&tx, format!("host refused: {reason}")),
        Ok(other) => return fail(&tx, format!("unexpected greeting {other:?}")),
        Err(e) => return fail(&tx, e),
    }
    let _ = stream.set_read_timeout(None);
    loop {
        let raw = match lobby_read_frame(&mut stream) {
            Ok(r) => r,
            Err(e) => return fail(&tx, format!("connection to host lost: {e}")),
        };
        if lobby_is_handshake(&raw) {
            match lobby_finish_handshake(&mut stream, &raw) {
                Ok(hs) => {
                    let _ = tx.send(ClientEvent::Handshake(hs));
                }
                Err(e) => fail(&tx, e),
            }
            return;
        }
        match decode_lobby_msg(&raw) {
            Ok(LobbyMsg::Start) => {
                if let Ok(mut w) = writer.lock() {
                    w.started = true;
                    if let Some(s) = w.stream.as_mut()
                        && let Err(e) = send(s, &LobbyMsg::StartAck)
                    {
                        drop(w);
                        return fail(&tx, e);
                    }
                }
                let _ = tx.send(ClientEvent::Msg(LobbyMsg::Start));
            }
            Ok(m @ LobbyMsg::Bye { .. }) => {
                let _ = tx.send(ClientEvent::Msg(m));
                return;
            }
            Ok(m) => {
                if tx.send(ClientEvent::Msg(m)).is_err() {
                    return;
                }
            }
            Err(e) => return fail(&tx, e),
        }
    }
}

/// `farust join` against a menu-hosted lobby: greets as an always-ready peer, honors the
/// start barrier and returns the lockstep handshake. `first` is the already-read greeting.
pub fn auto_join(stream: &mut TcpStream, first: &[u8]) -> Result<Handshake, String> {
    match decode_lobby_msg(first)? {
        LobbyMsg::Welcome { version, .. } if version == PROTOCOL_VERSION => {}
        LobbyMsg::Welcome { version, .. } => {
            return Err(format!("lockstep protocol mismatch: host v{version}, local v{PROTOCOL_VERSION}"));
        }
        LobbyMsg::Bye { reason } => return Err(format!("host refused: {reason}")),
        other => return Err(format!("unexpected greeting {other:?}")),
    }
    let name = format!("CLI {}", std::process::id() % 1000);
    send(stream, &LobbyMsg::Hello { version: PROTOCOL_VERSION, name })?;
    send(stream, &LobbyMsg::Request(ClientRequest::Ready(true)))?;
    println!("joined lobby as a ready CLI peer; waiting for the host to start...");
    let _ = stream.set_read_timeout(None);
    loop {
        let raw = lobby_read_frame(stream)?;
        if lobby_is_handshake(&raw) {
            return lobby_finish_handshake(stream, &raw);
        }
        match decode_lobby_msg(&raw)? {
            LobbyMsg::Start => send(stream, &LobbyMsg::StartAck)?,
            LobbyMsg::Bye { reason } => return Err(format!("lobby closed: {reason}")),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::{TurnPacket, TurnTransport};

    fn starts(n: usize) -> Vec<(String, (i64, i64))> {
        (0..n).map(|i| (format!("ARMY_{}", i + 1), (1000 * i as i64, 500))).collect()
    }

    #[test]
    fn lobby_message_codec_round_trip() {
        let mut st = NetLobbyState::new("SCMP_001", "Burial Mounds", 4, 99, 4, "Host", 2);
        st.add_peer(7, "Alice").unwrap();
        st.cycle_kind(2);
        st.set_ready(7, true);
        let msgs = vec![
            LobbyMsg::Welcome { version: PROTOCOL_VERSION, peer_id: 3, host_name: "Host".into() },
            LobbyMsg::Hello { version: 3, name: "Bob".into() },
            LobbyMsg::State(st.clone()),
            LobbyMsg::Request(ClientRequest::Ready(true)),
            LobbyMsg::Request(ClientRequest::Faction(3)),
            LobbyMsg::Request(ClientRequest::Chat("gl hf".into())),
            LobbyMsg::Chat(ChatLine { from: String::new(), text: "Alice joined".into() }),
            LobbyMsg::Start,
            LobbyMsg::StartAck,
            LobbyMsg::Bye { reason: "kicked".into() },
        ];
        for m in msgs {
            let enc = encode_lobby_msg(&m);
            assert!(!lobby_is_handshake(&enc));
            assert_eq!(decode_lobby_msg(&enc).unwrap(), m);
            assert!(decode_lobby_msg(&enc[..enc.len() - 1]).is_err() || enc.len() == 1);
            let mut extra = enc.clone();
            extra.push(1);
            assert!(decode_lobby_msg(&extra).is_err());
        }
    }

    #[test]
    fn slot_rules_and_handshake_ordering() {
        let mut s = NetLobbyState::new("M", "Map", 4, 5, 3, "Host", 0);
        assert_eq!(s.start_check().unwrap_err(), "waiting for players to join");
        assert_eq!(s.add_peer(4, "Ann").unwrap(), 1);
        assert_eq!(s.add_peer(9, "ann").unwrap(), 2);
        assert_eq!(s.peer(9).unwrap().name, "ann 2");
        assert!(s.add_peer(11, "Late").is_err(), "max_players 3");
        assert!(s.start_check().unwrap_err().contains("not ready"));
        s.set_ready(4, true);
        s.set_ready(9, true);
        // Slot 4 → AI; move Ann (slot 1) to the next non-peer slot (slot 3 = AI, swapped).
        s.cycle_kind(3);
        assert_eq!(s.slots[3].kind, NetSlotKind::Ai);
        s.move_slot(1);
        assert_eq!(s.slots[1].kind, NetSlotKind::Ai);
        assert_eq!(s.slots[3].kind, NetSlotKind::Peer(4));
        s.cycle_faction(3);
        s.start_check().unwrap();
        let (hs, armies) = s.build_handshake(&starts(4)).unwrap();
        // Host army 0, then peers by slot (ann 2 @ slot 2, Ann @ slot 3), then the AI @ slot 1.
        assert_eq!(armies, vec![(HOST_ID, 0), (9, 1), (4, 2)]);
        assert_eq!(hs.player_count, 3);
        let names: Vec<&str> = hs.setups.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, vec!["ARMY_1", "ARMY_3", "ARMY_4", "ARMY_2"]);
        assert_eq!(hs.setups[0].team, 0);
        // Leaving frees the slot; shrinking the map evicts peers that no longer fit.
        assert!(s.remove_peer(9).is_some());
        assert_eq!(s.slots[2].kind, NetSlotKind::Open);
        assert!(s.set_map("S", "Small", 2).is_empty(), "Ann moves into a free slot");
        assert_eq!(s.slots[1].kind, NetSlotKind::Peer(4));
        // Everyone on one team is rejected.
        s.slots[1].team = 0;
        assert!(s.start_check().unwrap_err().contains("one team"));
    }

    fn pump(host: &mut NetLobbyHost, clients: &mut [&mut NetLobbyClient], mut done: impl FnMut(&NetLobbyHost, &[&mut NetLobbyClient]) -> bool) {
        let t0 = Instant::now();
        while !done(host, clients) {
            assert!(t0.elapsed() < Duration::from_secs(10), "lobby pump timed out");
            assert!(!matches!(host.poll(), HostPoll::Launch(_) | HostPoll::Failed(_)));
            for c in clients.iter_mut() {
                assert!(matches!(c.poll(), ClientPoll::Idle | ClientPoll::Changed));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn loopback_host_and_two_clients_ready_start_and_share_setup() {
        let st = NetLobbyState::new("LOOP", "Loop", 4, 4242, 4, "Host", 0);
        let mut host = NetLobbyHost::bind("127.0.0.1:0", st).unwrap();
        let addr = host.local_addr.to_string();
        let mut a = NetLobbyClient::connect(&addr, "Alice");
        let mut b = NetLobbyClient::connect(&addr, "Bob");
        pump(&mut host, &mut [&mut a, &mut b], |h, cs| {
            h.state.peers.len() == 3 && cs.iter().all(|c| c.state.as_ref().is_some_and(|s| s.peers.len() == 3))
        });
        let bob = b.peer_id.unwrap();
        assert!(host.state.start_check().is_err(), "nobody ready yet");
        a.set_ready(true);
        b.set_ready(true);
        b.set_faction(2);
        a.send_chat("gl hf");
        host.edit(|s| {
            s.cycle_kind(3); // open → AI
            s.seed = 777;
            Vec::new()
        });
        pump(&mut host, &mut [&mut a, &mut b], |h, cs| {
            h.state.start_check().is_ok()
                && h.state.slot_of(bob).is_some_and(|i| h.state.slots[i].faction == 2)
                && cs.iter().all(|c| {
                    c.state.as_ref() == Some(&h.state)
                        && c.chat.iter().any(|l| l.from == "Alice" && l.text == "gl hf")
                })
        });
        assert_eq!(a.state, b.state);
        assert_eq!(a.state.as_ref(), Some(&host.state));
        assert_eq!(host.state.slots[host.state.slot_of(bob).unwrap()].faction, 2);

        host.begin_start().unwrap();
        let t0 = Instant::now();
        let launch = loop {
            assert!(t0.elapsed() < Duration::from_secs(10));
            match host.poll() {
                HostPoll::Launch(l) => break l,
                HostPoll::Failed(e) => panic!("{e}"),
                _ => std::thread::sleep(Duration::from_millis(5)),
            }
        };
        assert!(a.poll().is_live() && b.poll().is_live());
        let (hs_host, mut t_host) = launch.into_transport(&starts(4)).unwrap();
        let mut got = Vec::new();
        for c in [&mut a, &mut b] {
            let t0 = Instant::now();
            loop {
                assert!(t0.elapsed() < Duration::from_secs(10));
                match c.poll() {
                    ClientPoll::Launch(hs, t) => {
                        got.push((hs, t));
                        break;
                    }
                    ClientPoll::Closed(e) => panic!("{e}"),
                    _ => std::thread::sleep(Duration::from_millis(5)),
                }
            }
        }
        assert_eq!(hs_host.player_count, 3);
        assert_eq!(hs_host.setups.len(), 4, "3 humans + 1 AI");
        assert_eq!(hs_host.seed, 777);
        let mut armies: Vec<ArmyId> = got.iter().map(|(hs, _)| hs.assigned_army).collect();
        armies.sort();
        assert_eq!(armies, vec![1, 2]);
        for (hs, _) in &got {
            let mut same = hs.clone();
            same.assigned_army = 0;
            assert_eq!(same, hs_host, "all peers receive the same match setup");
        }
        // The handed-over streams carry lockstep turns.
        let workers: Vec<_> = got
            .into_iter()
            .map(|(_, mut t)| std::thread::spawn(move || t.exchange(TurnPacket { turn: 0, hash: 9, commands: vec![], controls: vec![] }).unwrap()))
            .collect();
        let hb = t_host.exchange(TurnPacket { turn: 0, hash: 9, commands: vec![], controls: vec![] }).unwrap();
        assert_eq!(hb.hashes.len(), 3);
        for w in workers {
            assert_eq!(w.join().unwrap(), hb);
        }
    }

    #[test]
    fn cli_join_auto_joins_a_menu_lobby() {
        let st = NetLobbyState::new("LOOP", "Loop", 2, 1, 2, "Host", 1);
        let mut host = NetLobbyHost::bind("127.0.0.1:0", st).unwrap();
        let addr = host.local_addr.to_string();
        let cli = std::thread::spawn(move || LockstepClient::connect(&addr).map(|(_, hs)| hs));
        let t0 = Instant::now();
        while host.state.start_check().is_err() {
            assert!(t0.elapsed() < Duration::from_secs(10));
            host.poll();
            std::thread::sleep(Duration::from_millis(5));
        }
        host.begin_start().unwrap();
        let launch = loop {
            assert!(t0.elapsed() < Duration::from_secs(10));
            if let HostPoll::Launch(l) = host.poll() {
                break l;
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        let (hs, _t) = launch.into_transport(&starts(2)).unwrap();
        let cli_hs = cli.join().unwrap().unwrap();
        assert_eq!(cli_hs.assigned_army, 1);
        assert_eq!(cli_hs.setups, hs.setups);
    }

    impl ClientPoll {
        fn is_live(&self) -> bool {
            matches!(self, ClientPoll::Idle | ClientPoll::Changed)
        }
    }

    #[test]
    fn invite_code_lobby_direct_and_relayed_lockstep_match() {
        use crate::net::{LockstepCore, NetHandle, run_headless};
        use crate::pathfinding::PathGrid;
        use crate::sim::SCALE;
        use crate::skirmish::Skirmish;
        use crate::skirmish::tests::test_catalog;

        let rv = crate::rendezvous::RendezvousServer::bind("127.0.0.1:0").unwrap();
        let rv_addr = rv.local_addr().to_string();

        for (force_relay, seed) in [(false, 5150u64), (true, 6161u64)] {
            let mut st = NetLobbyState::new("LOOP", "Loop", 4, seed, 3, "Host", 0);
            // Map all 4 slots onto FACTION_TABLE indices < 2 (0=uef, 2=cybran) for test_catalog().
            st.slots[1].faction = 2;
            st.slots[2].faction = 0;
            st.slots[3].faction = 2;
            let mut host = NetLobbyHost::bind("127.0.0.1:0", st).unwrap();
            let code = host.register_invite(&rv_addr, Duration::from_secs(2)).unwrap();

            let mut a = NetLobbyClient::connect_code(&rv_addr, &code, force_relay, "Alice");
            let mut b = NetLobbyClient::connect_code(&rv_addr, &code, force_relay, "Bob");
            pump(&mut host, &mut [&mut a, &mut b], |h, cs| {
                h.state.peers.len() == 3 && cs.iter().all(|c| c.state.as_ref().is_some_and(|s| s.peers.len() == 3))
            });
            assert_eq!(a.via_relay, Some(force_relay));
            assert_eq!(b.via_relay, Some(force_relay));
            let aid = a.peer_id.unwrap();
            assert_eq!(host.peer_is_relayed(aid), Some(force_relay));

            a.set_ready(true);
            b.set_ready(true);
            pump(&mut host, &mut [&mut a, &mut b], |h, _| h.state.start_check().is_ok());

            host.begin_start().unwrap();
            let t0 = Instant::now();
            let launch = loop {
                assert!(t0.elapsed() < Duration::from_secs(10));
                match host.poll() {
                    HostPoll::Launch(l) => break l,
                    HostPoll::Failed(e) => panic!("{e}"),
                    _ => std::thread::sleep(Duration::from_millis(5)),
                }
            };
            let _ = a.poll();
            let _ = b.poll();
            let map_starts = vec![
                ("ARMY_1".into(), (60 * SCALE, 60 * SCALE)),
                ("ARMY_2".into(), (190 * SCALE, 190 * SCALE)),
                ("ARMY_3".into(), (60 * SCALE, 190 * SCALE)),
                ("ARMY_4".into(), (190 * SCALE, 60 * SCALE)),
            ];
            let (hs_host, mut t_host) = launch.into_transport(&map_starts).unwrap();
            let mut clients = Vec::new();
            for c in [&mut a, &mut b] {
                let t0 = Instant::now();
                loop {
                    assert!(t0.elapsed() < Duration::from_secs(10));
                    match c.poll() {
                        ClientPoll::Launch(hs, t) => {
                            clients.push((hs, t));
                            break;
                        }
                        ClientPoll::Closed(e) => panic!("{e}"),
                        _ => std::thread::sleep(Duration::from_millis(5)),
                    }
                }
            }
            // Registration closes cleanly at match launch so late lookups cannot rejoin.
            let lookup_after = crate::rendezvous::connect_via_code(&rv_addr, &code, true, Duration::from_millis(200));
            assert!(lookup_after.is_err());

            let make_world = |hs: &Handshake| {
                let grid = PathGrid::new(64, 64, 4 * SCALE);
                let mass = vec![(40 * SCALE, 40 * SCALE), (210 * SCALE, 210 * SCALE)];
                Skirmish::new(test_catalog(), grid, (256 * SCALE, 256 * SCALE), mass, &hs.setups, hs.seed).unwrap()
            };

            let workers: Vec<_> = clients
                .into_iter()
                .map(|(hs, mut t)| {
                    std::thread::spawn(move || {
                        let mut w = make_world(&hs);
                        let mut core = LockstepCore::from_handshake(&w, &hs);
                        run_headless(&mut t, &mut core, &mut w, 60).unwrap()
                    })
                })
                .collect();
            let mut w_host = make_world(&hs_host);
            let mut core_host = LockstepCore::from_handshake(&w_host, &hs_host);
            let h_host = run_headless(&mut t_host, &mut core_host, &mut w_host, 60).unwrap();
            let _ = NetHandle::start;
            for w in workers {
                assert_eq!(w.join().unwrap(), h_host, "relay={force_relay} lockstep hash mismatch");
            }
        }
    }
}
