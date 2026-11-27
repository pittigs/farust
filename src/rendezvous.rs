//! Phase 26 invite codes, lightweight TCP rendezvous, and zero-dependency TCP relay fallback.
//!
//! Peers behind strict NAT or firewalls do not need port-forwarding or IP sharing:
//! 1. The host opens a local listener (as usual) and holds one outbound control TCP stream to a
//!    `farust rendezvous` server under a random unambiguous 8-char code (`XXXX-XXXX`).
//! 2. Invited players look up the code: the client tries direct addresses first (short timeout)
//!    and automatically falls back to opening a relayed TCP stream through the rendezvous server.
//! 3. Once `RelayReady` / `TunnelGo` completes, the rendezvous server splices bytes
//!    bi-directionally (`std::io::copy`), so the existing v4 lobby and lockstep protocols run
//!    over the exact returned `TcpStream` unchanged.

use std::collections::HashMap;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::net::{lobby_read_frame, lobby_write_frame};
use crate::replay::{read_str, read_u8, read_u64, write_str};

pub const DEFAULT_ADDR: &str = "127.0.0.1:15050";

/// Unambiguous 32-character Crockford-like alphabet (no 0/O/1/I).
const ALPHABET: &[u8; 32] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

const MAGIC: u8 = b'R';
const VERSION: u8 = 1;

const MSG_REGISTER: u8 = 1;
const MSG_REGISTERED: u8 = 2;
const MSG_PING: u8 = 3;
const MSG_PONG: u8 = 4;
const MSG_UNREGISTER: u8 = 5;
const MSG_LOOKUP: u8 = 6;
const MSG_FOUND: u8 = 7;
const MSG_CLIENT_RELAY: u8 = 8;
const MSG_OPEN_TUNNEL: u8 = 9;
const MSG_HOST_TUNNEL: u8 = 10;
const MSG_TUNNEL_GO: u8 = 11;
const MSG_RELAY_READY: u8 = 12;
const MSG_ERROR: u8 = 13;

const CTRL_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);
const RELAY_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

static RNG_SEQ: AtomicU64 = AtomicU64::new(1);

fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn next_entropy() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let seq = RNG_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id() as u64;
    mix64(nanos ^ seq.wrapping_mul(0x517C_C1B7_2722_0A95) ^ (pid << 17))
}

/// Generates a random invite code formatted as `XXXX-XXXX`.
pub fn generate_code() -> String {
    let mut v = next_entropy();
    let mut out = String::with_capacity(9);
    for i in 0..8 {
        if i == 4 {
            out.push('-');
        }
        if i == 5 {
            v ^= next_entropy();
        }
        out.push(ALPHABET[(v & 31) as usize] as char);
        v >>= 5;
    }
    out
}

pub fn generate_token() -> u64 {
    next_entropy() | 1
}

/// Validates and canonicalizes a typed invite code (`xxxx xxxx` / `XXXX-XXXX` → `XXXX-XXXX`).
pub fn normalize_code(raw: &str) -> Option<String> {
    let clean: Vec<u8> = raw
        .trim()
        .bytes()
        .filter(|b| !b.is_ascii_whitespace() && *b != b'-')
        .map(|b| b.to_ascii_uppercase())
        .collect();
    if clean.len() != 8 || !clean.iter().all(|b| ALPHABET.contains(b)) {
        return None;
    }
    let mut out = String::with_capacity(9);
    for (i, &b) in clean.iter().enumerate() {
        if i == 4 {
            out.push('-');
        }
        out.push(b as char);
    }
    Some(out)
}

/// True when `raw` looks like an invite code rather than a `HOST:PORT` socket address.
pub fn looks_like_code(raw: &str) -> bool {
    let t = raw.trim();
    !t.contains(':') && normalize_code(t).is_some()
}

/// Returns candidate direct addresses for a bound game socket (LAN primary IP + localhost if loopback).
pub fn discover_local_addrs(bound: SocketAddr) -> Vec<SocketAddr> {
    let port = bound.port();
    let mut addrs = Vec::new();
    if !bound.ip().is_unspecified() {
        addrs.push(bound);
    } else {
        // Query kernel route table for default outbound interface IP (UDP connect sends zero packets).
        if let Ok(sock) = UdpSocket::bind((IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0))
            && sock.connect("192.0.2.1:80").is_ok()
            && let Ok(local) = sock.local_addr()
            && !local.ip().is_unspecified()
        {
            addrs.push(SocketAddr::new(local.ip(), port));
        }
        let lo = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
        if !addrs.contains(&lo) {
            addrs.push(lo);
        }
    }
    addrs
}

// ---------------------------------------------------------------------------
// Wire messages
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum RvMsg {
    Register {
        code: String,
        token: u64,
        listen_port: u16,
        local_addrs: Vec<SocketAddr>,
        host_name: String,
        map_id: String,
    },
    Registered {
        code: String,
        addrs: Vec<SocketAddr>,
    },
    Ping,
    Pong,
    Unregister {
        code: String,
        token: u64,
    },
    Lookup {
        code: String,
    },
    Found {
        code: String,
        addrs: Vec<SocketAddr>,
        host_name: String,
        map_id: String,
    },
    ClientRelay {
        code: String,
    },
    OpenTunnel {
        tunnel_id: u64,
        client_addr: SocketAddr,
    },
    HostTunnel {
        code: String,
        token: u64,
        tunnel_id: u64,
    },
    TunnelGo,
    RelayReady {
        host_addr: SocketAddr,
    },
    Error(String),
}

fn write_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16, String> {
    if *pos + 2 > data.len() {
        return Err("truncated u16".into());
    }
    let v = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    Ok(v)
}

fn write_addr(out: &mut Vec<u8>, addr: SocketAddr) {
    match addr {
        SocketAddr::V4(v4) => {
            out.push(4);
            out.extend_from_slice(&v4.ip().octets());
            write_u16(out, v4.port());
        }
        SocketAddr::V6(v6) => {
            out.push(6);
            out.extend_from_slice(&v6.ip().octets());
            write_u16(out, v6.port());
        }
    }
}

fn read_addr(data: &[u8], pos: &mut usize) -> Result<SocketAddr, String> {
    let kind = read_u8(data, pos)?;
    match kind {
        4 => {
            if *pos + 4 > data.len() {
                return Err("truncated ipv4".into());
            }
            let ip = Ipv4Addr::new(data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]);
            *pos += 4;
            let port = read_u16(data, pos)?;
            Ok(SocketAddr::new(IpAddr::V4(ip), port))
        }
        6 => {
            if *pos + 16 > data.len() {
                return Err("truncated ipv6".into());
            }
            let mut oct = [0u8; 16];
            oct.copy_from_slice(&data[*pos..*pos + 16]);
            *pos += 16;
            let port = read_u16(data, pos)?;
            Ok(SocketAddr::new(IpAddr::from(oct), port))
        }
        other => Err(format!("invalid address kind {other}")),
    }
}

fn write_addrs(out: &mut Vec<u8>, addrs: &[SocketAddr]) {
    let n = addrs.len().min(16);
    out.push(n as u8);
    for &a in &addrs[..n] {
        write_addr(out, a);
    }
}

fn read_addrs(data: &[u8], pos: &mut usize) -> Result<Vec<SocketAddr>, String> {
    let n = read_u8(data, pos)? as usize;
    let mut out = Vec::with_capacity(n.min(16));
    for _ in 0..n.min(16) {
        out.push(read_addr(data, pos)?);
    }
    Ok(out)
}

fn encode_msg(m: &RvMsg) -> Vec<u8> {
    let mut out = vec![MAGIC, VERSION];
    match m {
        RvMsg::Register { code, token, listen_port, local_addrs, host_name, map_id } => {
            out.push(MSG_REGISTER);
            write_str(&mut out, code);
            out.extend_from_slice(&token.to_le_bytes());
            write_u16(&mut out, *listen_port);
            write_addrs(&mut out, local_addrs);
            write_str(&mut out, host_name);
            write_str(&mut out, map_id);
        }
        RvMsg::Registered { code, addrs } => {
            out.push(MSG_REGISTERED);
            write_str(&mut out, code);
            write_addrs(&mut out, addrs);
        }
        RvMsg::Ping => out.push(MSG_PING),
        RvMsg::Pong => out.push(MSG_PONG),
        RvMsg::Unregister { code, token } => {
            out.push(MSG_UNREGISTER);
            write_str(&mut out, code);
            out.extend_from_slice(&token.to_le_bytes());
        }
        RvMsg::Lookup { code } => {
            out.push(MSG_LOOKUP);
            write_str(&mut out, code);
        }
        RvMsg::Found { code, addrs, host_name, map_id } => {
            out.push(MSG_FOUND);
            write_str(&mut out, code);
            write_addrs(&mut out, addrs);
            write_str(&mut out, host_name);
            write_str(&mut out, map_id);
        }
        RvMsg::ClientRelay { code } => {
            out.push(MSG_CLIENT_RELAY);
            write_str(&mut out, code);
        }
        RvMsg::OpenTunnel { tunnel_id, client_addr } => {
            out.push(MSG_OPEN_TUNNEL);
            out.extend_from_slice(&tunnel_id.to_le_bytes());
            write_addr(&mut out, *client_addr);
        }
        RvMsg::HostTunnel { code, token, tunnel_id } => {
            out.push(MSG_HOST_TUNNEL);
            write_str(&mut out, code);
            out.extend_from_slice(&token.to_le_bytes());
            out.extend_from_slice(&tunnel_id.to_le_bytes());
        }
        RvMsg::TunnelGo => out.push(MSG_TUNNEL_GO),
        RvMsg::RelayReady { host_addr } => {
            out.push(MSG_RELAY_READY);
            write_addr(&mut out, *host_addr);
        }
        RvMsg::Error(e) => {
            out.push(MSG_ERROR);
            write_str(&mut out, e);
        }
    }
    out
}

fn decode_msg(data: &[u8]) -> Result<RvMsg, String> {
    let mut pos = 0usize;
    let magic = read_u8(data, &mut pos)?;
    if magic != MAGIC {
        return Err(format!("bad rendezvous magic {magic}"));
    }
    let ver = read_u8(data, &mut pos)?;
    if ver != VERSION {
        return Err(format!("rendezvous protocol mismatch v{ver} (expected v{VERSION})"));
    }
    let msg = match read_u8(data, &mut pos)? {
        MSG_REGISTER => RvMsg::Register {
            code: read_str(data, &mut pos)?,
            token: read_u64(data, &mut pos)?,
            listen_port: read_u16(data, &mut pos)?,
            local_addrs: read_addrs(data, &mut pos)?,
            host_name: read_str(data, &mut pos)?,
            map_id: read_str(data, &mut pos)?,
        },
        MSG_REGISTERED => RvMsg::Registered {
            code: read_str(data, &mut pos)?,
            addrs: read_addrs(data, &mut pos)?,
        },
        MSG_PING => RvMsg::Ping,
        MSG_PONG => RvMsg::Pong,
        MSG_UNREGISTER => RvMsg::Unregister {
            code: read_str(data, &mut pos)?,
            token: read_u64(data, &mut pos)?,
        },
        MSG_LOOKUP => RvMsg::Lookup { code: read_str(data, &mut pos)? },
        MSG_FOUND => RvMsg::Found {
            code: read_str(data, &mut pos)?,
            addrs: read_addrs(data, &mut pos)?,
            host_name: read_str(data, &mut pos)?,
            map_id: read_str(data, &mut pos)?,
        },
        MSG_CLIENT_RELAY => RvMsg::ClientRelay { code: read_str(data, &mut pos)? },
        MSG_OPEN_TUNNEL => RvMsg::OpenTunnel {
            tunnel_id: read_u64(data, &mut pos)?,
            client_addr: read_addr(data, &mut pos)?,
        },
        MSG_HOST_TUNNEL => RvMsg::HostTunnel {
            code: read_str(data, &mut pos)?,
            token: read_u64(data, &mut pos)?,
            tunnel_id: read_u64(data, &mut pos)?,
        },
        MSG_TUNNEL_GO => RvMsg::TunnelGo,
        MSG_RELAY_READY => RvMsg::RelayReady { host_addr: read_addr(data, &mut pos)? },
        MSG_ERROR => RvMsg::Error(read_str(data, &mut pos)?),
        other => return Err(format!("unknown rendezvous message {other}")),
    };
    if pos != data.len() {
        return Err("trailing bytes in rendezvous message".into());
    }
    Ok(msg)
}

fn send_msg(stream: &mut TcpStream, m: &RvMsg) -> Result<(), String> {
    lobby_write_frame(stream, &encode_msg(m))
}

fn recv_msg(stream: &mut TcpStream) -> Result<RvMsg, String> {
    let raw = lobby_read_frame(stream)?;
    decode_msg(&raw)
}

fn connect_with_timeout(addr: &str, timeout: Duration) -> Result<(TcpStream, SocketAddr), String> {
    let addrs: Vec<SocketAddr> = addr
        .to_socket_addrs()
        .map_err(|e| format!("cannot resolve rendezvous {addr}: {e}"))?
        .collect();
    let mut last = format!("no address for rendezvous {addr}");
    for sa in addrs {
        match TcpStream::connect_timeout(&sa, timeout) {
            Ok(s) => return Ok((s, sa)),
            Err(e) => last = format!("rendezvous {sa}: {e}"),
        }
    }
    Err(last)
}

// ---------------------------------------------------------------------------
// Server (rendezvous directory + bi-directional TCP relay)
// ---------------------------------------------------------------------------

struct RoomEntry {
    token: u64,
    ctrl_writer: Arc<Mutex<TcpStream>>,
    addrs: Vec<SocketAddr>,
    host_name: String,
    map_id: String,
}

struct PendingTunnel {
    token: u64,
    tx: SyncSender<TcpStream>,
}

struct ServerState {
    rooms: Mutex<HashMap<String, RoomEntry>>,
    tunnels: Mutex<HashMap<(String, u64), PendingTunnel>>,
    next_tunnel: AtomicU64,
}

impl ServerState {
    fn new() -> Self {
        Self {
            rooms: Mutex::new(HashMap::new()),
            tunnels: Mutex::new(HashMap::new()),
            next_tunnel: AtomicU64::new(1),
        }
    }

    fn remove_room_if_token(&self, code: &str, token: u64) {
        if let Ok(mut rooms) = self.rooms.lock()
            && rooms.get(code).is_some_and(|r| r.token == token)
        {
            rooms.remove(code);
        }
        if let Ok(mut tunnels) = self.tunnels.lock() {
            tunnels.retain(|(c, _), p| c != code || p.token != token);
        }
    }
}

pub struct RendezvousServer {
    local_addr: SocketAddr,
    stop: Arc<AtomicBool>,
    accept_thread: Option<JoinHandle<()>>,
}

impl RendezvousServer {
    pub fn bind(addr: &str) -> Result<Self, String> {
        let listener = TcpListener::bind(addr)
            .map_err(|e| format!("cannot bind rendezvous server on {addr}: {e}"))?;
        let local_addr = listener.local_addr().map_err(|e| format!("local_addr: {e}"))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| format!("set_nonblocking: {e}"))?;
        let state = Arc::new(ServerState::new());
        let stop = Arc::new(AtomicBool::new(false));
        let (st, sp) = (state.clone(), stop.clone());
        let accept_thread = std::thread::spawn(move || {
            while !sp.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, peer)) => {
                        if sp.load(Ordering::Relaxed) {
                            let _ = stream.shutdown(Shutdown::Both);
                            break;
                        }
                        let st2 = st.clone();
                        std::thread::spawn(move || handle_server_conn(stream, peer, st2));
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    Err(_) => {
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }
            }
        });
        Ok(Self { local_addr, stop, accept_thread: Some(accept_thread) })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Blocks forever until [`Self::shutdown`] (used by `farust rendezvous`).
    pub fn run_forever(&self) {
        while !self.stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    pub fn shutdown(&mut self) {
        if !self.stop.swap(true, Ordering::Relaxed) {
            let wake_ip = match self.local_addr.ip() {
                ip if ip.is_unspecified() && self.local_addr.is_ipv4() => {
                    IpAddr::V4(Ipv4Addr::LOCALHOST)
                }
                ip if ip.is_unspecified() => IpAddr::V6(std::net::Ipv6Addr::LOCALHOST),
                ip => ip,
            };
            let _ = TcpStream::connect_timeout(
                &SocketAddr::new(wake_ip, self.local_addr.port()),
                Duration::from_millis(100),
            );
        }
        if let Some(h) = self.accept_thread.take() {
            let _ = h.join();
        }
    }
}

impl Drop for RendezvousServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn handle_server_conn(mut stream: TcpStream, peer: SocketAddr, state: Arc<ServerState>) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(RELAY_HANDSHAKE_TIMEOUT));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));

    let first = match recv_msg(&mut stream) {
        Ok(m) => m,
        Err(_) => return,
    };

    match first {
        RvMsg::Register { code, token, listen_port, local_addrs, host_name, map_id } => {
            let Some(norm) = normalize_code(&code) else {
                let _ = send_msg(&mut stream, &RvMsg::Error("invalid invite code".into()));
                return;
            };
            if listen_port == 0 {
                let _ = send_msg(&mut stream, &RvMsg::Error("invalid listen port 0".into()));
                return;
            }
            let mut addrs = Vec::new();
            // Prefer public/observed peer IP first unless both local and observed are loopback.
            if !peer.ip().is_unspecified() && !peer.ip().is_loopback() {
                addrs.push(SocketAddr::new(peer.ip(), listen_port));
            }
            for a in local_addrs {
                if !a.ip().is_unspecified() && a.port() > 0 && !addrs.contains(&a) {
                    addrs.push(a);
                }
            }
            if addrs.is_empty() && !peer.ip().is_unspecified() {
                addrs.push(SocketAddr::new(peer.ip(), listen_port));
            }
            let Ok(writer_clone) = stream.try_clone() else {
                return;
            };
            let ctrl_writer = Arc::new(Mutex::new(writer_clone));
            {
                let mut rooms = state.rooms.lock().unwrap();
                if rooms.contains_key(&norm) {
                    let _ = send_msg(&mut stream, &RvMsg::Error(format!("code {norm} already in use")));
                    return;
                }
                rooms.insert(
                    norm.clone(),
                    RoomEntry {
                        token,
                        ctrl_writer: ctrl_writer.clone(),
                        addrs: addrs.clone(),
                        host_name,
                        map_id,
                    },
                );
            }
            if let Ok(mut w) = ctrl_writer.lock() {
                if send_msg(&mut w, &RvMsg::Registered { code: norm.clone(), addrs }).is_err() {
                    state.remove_room_if_token(&norm, token);
                    return;
                }
            }
            // Control stream stays open until the host closes it or disconnects.
            let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
            loop {
                match recv_msg(&mut stream) {
                    Ok(RvMsg::Ping) => {
                        if let Ok(mut w) = ctrl_writer.lock()
                            && send_msg(&mut w, &RvMsg::Pong).is_err()
                        {
                            break;
                        }
                    }
                    Ok(RvMsg::Unregister { code: c, token: t }) => {
                        if let Some(nc) = normalize_code(&c) {
                            state.remove_room_if_token(&nc, t);
                        }
                        break;
                    }
                    Ok(_) => {}
                    Err(e) if e.contains("TimedOut") || e.contains("WouldBlock") => continue,
                    Err(_) => break,
                }
            }
            state.remove_room_if_token(&norm, token);
        }
        RvMsg::Lookup { code } => {
            let Some(norm) = normalize_code(&code) else {
                let _ = send_msg(&mut stream, &RvMsg::Error(format!("invalid invite code '{code}'")));
                return;
            };
            let reply = match state.rooms.lock().unwrap().get(&norm) {
                Some(r) => RvMsg::Found {
                    code: norm,
                    addrs: r.addrs.clone(),
                    host_name: r.host_name.clone(),
                    map_id: r.map_id.clone(),
                },
                None => RvMsg::Error(format!("unknown or expired invite code '{norm}'")),
            };
            let _ = send_msg(&mut stream, &reply);
        }
        RvMsg::ClientRelay { code } => {
            let Some(norm) = normalize_code(&code) else {
                let _ = send_msg(&mut stream, &RvMsg::Error(format!("invalid invite code '{code}'")));
                return;
            };
            let (token, ctrl_writer, host_addr) = {
                let rooms = state.rooms.lock().unwrap();
                let Some(r) = rooms.get(&norm) else {
                    drop(rooms);
                    let _ = send_msg(
                        &mut stream,
                        &RvMsg::Error(format!("unknown or expired invite code '{norm}'")),
                    );
                    return;
                };
                (
                    r.token,
                    r.ctrl_writer.clone(),
                    r.addrs.first().copied().unwrap_or(peer),
                )
            };
            let tunnel_id = state.next_tunnel.fetch_add(1, Ordering::Relaxed);
            let (tx, rx) = mpsc::sync_channel::<TcpStream>(1);
            state
                .tunnels
                .lock()
                .unwrap()
                .insert((norm.clone(), tunnel_id), PendingTunnel { token, tx });
            let ask_res = ctrl_writer
                .lock()
                .map_err(|_| "host control poisoned".to_string())
                .and_then(|mut w| send_msg(&mut w, &RvMsg::OpenTunnel { tunnel_id, client_addr: peer }));
            if ask_res.is_err() {
                state.tunnels.lock().unwrap().remove(&(norm, tunnel_id));
                let _ = send_msg(&mut stream, &RvMsg::Error("host control connection lost".into()));
                return;
            }
            let Ok(mut host_stream) = rx.recv_timeout(RELAY_HANDSHAKE_TIMEOUT) else {
                state.tunnels.lock().unwrap().remove(&(norm, tunnel_id));
                let _ = send_msg(&mut stream, &RvMsg::Error("host did not respond to relay request".into()));
                return;
            };
            // Prepare both sockets before sending Go/Ready so zero lockstep bytes overtake the control markers.
            let _ = host_stream.set_read_timeout(None);
            let _ = host_stream.set_write_timeout(None);
            let _ = host_stream.set_nodelay(true);
            let _ = stream.set_read_timeout(None);
            let _ = stream.set_write_timeout(None);
            let _ = stream.set_nodelay(true);
            if send_msg(&mut host_stream, &RvMsg::TunnelGo).is_err() {
                let _ = host_stream.shutdown(Shutdown::Both);
                let _ = stream.shutdown(Shutdown::Both);
                return;
            }
            if send_msg(&mut stream, &RvMsg::RelayReady { host_addr }).is_err() {
                let _ = host_stream.shutdown(Shutdown::Both);
                let _ = stream.shutdown(Shutdown::Both);
                return;
            }
            splice_bidirectional(stream, host_stream);
        }
        RvMsg::HostTunnel { code, token, tunnel_id } => {
            let Some(norm) = normalize_code(&code) else {
                return;
            };
            let tx = {
                let mut tunnels = state.tunnels.lock().unwrap();
                match tunnels.get(&(norm.clone(), tunnel_id)) {
                    Some(p) if p.token == token => tunnels.remove(&(norm, tunnel_id)).map(|p| p.tx),
                    _ => None,
                }
            };
            if let Some(tx) = tx {
                let _ = tx.send(stream);
            }
        }
        _ => {
            let _ = send_msg(&mut stream, &RvMsg::Error("unexpected initial rendezvous message".into()));
        }
    }
}

fn splice_bidirectional(a: TcpStream, b: TcpStream) {
    let (Ok(mut ar), Ok(mut bw)) = (a.try_clone(), b.try_clone()) else {
        let _ = a.shutdown(Shutdown::Both);
        let _ = b.shutdown(Shutdown::Both);
        return;
    };
    let (mut br, mut aw) = (b, a);
    let h = std::thread::spawn(move || {
        let _ = io::copy(&mut ar, &mut bw);
        let _ = ar.shutdown(Shutdown::Both);
        let _ = bw.shutdown(Shutdown::Both);
    });
    let _ = io::copy(&mut br, &mut aw);
    let _ = br.shutdown(Shutdown::Both);
    let _ = aw.shutdown(Shutdown::Both);
    let _ = h.join();
}

// ---------------------------------------------------------------------------
// Host registration
// ---------------------------------------------------------------------------

pub struct HostRegistration {
    pub code: String,
    #[allow(dead_code)]
    pub rendezvous_addr: SocketAddr,
    token: u64,
    stop: Arc<AtomicBool>,
    ctrl_writer: Arc<Mutex<TcpStream>>,
    thread: Option<JoinHandle<()>>,
}

impl HostRegistration {
    /// Registers a listening match/lobby at `rendezvous_addr` and spawns a control loop that
    /// opens outbound relay tunnels (`on_tunnel`) whenever direct TCP connections fail.
    pub fn register(
        rendezvous_addr: &str,
        bound: SocketAddr,
        host_name: &str,
        map_id: &str,
        connect_timeout: Duration,
        on_tunnel: Box<dyn Fn(TcpStream, SocketAddr) + Send + Sync + 'static>,
    ) -> Result<Self, String> {
        let on_tunnel: Arc<dyn Fn(TcpStream, SocketAddr) + Send + Sync + 'static> = Arc::from(on_tunnel);
        let local_addrs = discover_local_addrs(bound);
        let token = generate_token();
        let mut last_err = String::new();

        for _attempt in 0..3 {
            let code = generate_code();
            let (mut ctrl, rv_sa) = connect_with_timeout(rendezvous_addr, connect_timeout)?;
            let _ = ctrl.set_nodelay(true);
            let _ = ctrl.set_read_timeout(Some(Duration::from_secs(3)));
            let _ = ctrl.set_write_timeout(Some(Duration::from_secs(3)));

            send_msg(
                &mut ctrl,
                &RvMsg::Register {
                    code: code.clone(),
                    token,
                    listen_port: bound.port(),
                    local_addrs: local_addrs.clone(),
                    host_name: host_name.to_string(),
                    map_id: map_id.to_string(),
                },
            )?;
            match recv_msg(&mut ctrl)? {
                RvMsg::Registered { code: assigned, .. } => {
                    let ctrl_writer = Arc::new(Mutex::new(ctrl.try_clone().map_err(|e| format!("ctrl clone: {e}"))?));
                    let stop = Arc::new(AtomicBool::new(false));
                    let _ = ctrl.set_read_timeout(Some(Duration::from_millis(400)));

                    let (c_code, c_stop, c_writer) = (assigned.clone(), stop.clone(), ctrl_writer.clone());
                    let thread = std::thread::spawn(move || {
                        let mut last_ping = Instant::now();
                        while !c_stop.load(Ordering::Relaxed) {
                            match recv_msg(&mut ctrl) {
                                Ok(RvMsg::OpenTunnel { tunnel_id, client_addr }) => {
                                    let cb = on_tunnel.clone();
                                    let t_code = c_code.clone();
                                    std::thread::spawn(move || {
                                        if let Ok(mut s) = TcpStream::connect_timeout(&rv_sa, RELAY_HANDSHAKE_TIMEOUT) {
                                            let _ = s.set_nodelay(true);
                                            let _ = s.set_read_timeout(Some(RELAY_HANDSHAKE_TIMEOUT));
                                            let _ = s.set_write_timeout(Some(Duration::from_secs(5)));
                                            if send_msg(&mut s, &RvMsg::HostTunnel { code: t_code, token, tunnel_id }).is_ok()
                                                && matches!(recv_msg(&mut s), Ok(RvMsg::TunnelGo))
                                            {
                                                let _ = s.set_read_timeout(None);
                                                let _ = s.set_write_timeout(None);
                                                cb(s, client_addr);
                                            }
                                        }
                                    });
                                }
                                Ok(RvMsg::Pong) => {}
                                Ok(_) => {}
                                Err(e) if e.contains("TimedOut") || e.contains("WouldBlock") => {
                                    if last_ping.elapsed() >= CTRL_HEARTBEAT_INTERVAL {
                                        last_ping = Instant::now();
                                        if let Ok(mut w) = c_writer.lock()
                                            && send_msg(&mut w, &RvMsg::Ping).is_err()
                                        {
                                            break;
                                        }
                                    }
                                }
                                Err(_) => break,
                            }
                        }
                    });
                    return Ok(Self {
                        code: assigned,
                        rendezvous_addr: rv_sa,
                        token,
                        stop,
                        ctrl_writer,
                        thread: Some(thread),
                    });
                }
                RvMsg::Error(e) => last_err = e,
                other => return Err(format!("unexpected register reply: {other:?}")),
            }
        }
        Err(if last_err.is_empty() { "failed to register invite code".into() } else { last_err })
    }

    pub fn close(&mut self) {
        if !self.stop.swap(true, Ordering::Relaxed) {
            if let Ok(mut w) = self.ctrl_writer.lock() {
                let _ = send_msg(&mut w, &RvMsg::Unregister { code: self.code.clone(), token: self.token });
                let _ = w.shutdown(Shutdown::Both);
            }
        }
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}

impl Drop for HostRegistration {
    fn drop(&mut self) {
        self.close();
    }
}

// ---------------------------------------------------------------------------
// Client lookup + direct-first / relay-fallback connection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectVia {
    Direct(SocketAddr),
    Relayed(SocketAddr),
}

impl ConnectVia {
    pub fn is_relayed(self) -> bool {
        matches!(self, ConnectVia::Relayed(_))
    }

    pub fn addr(self) -> SocketAddr {
        match self {
            ConnectVia::Direct(a) | ConnectVia::Relayed(a) => a,
        }
    }
}

/// Resolves an invite code on `rendezvous_addr`, tries direct host addresses first (unless
/// `force_relay` or `FARUST_FORCE_RELAY=1`), and falls back seamlessly to a relayed TCP tunnel.
pub fn connect_via_code(
    rendezvous_addr: &str,
    code_raw: &str,
    force_relay: bool,
    direct_timeout: Duration,
) -> Result<(TcpStream, ConnectVia), String> {
    let code = normalize_code(code_raw)
        .ok_or_else(|| format!("invalid invite code '{}' (expected 8 characters like XXXX-XXXX)", code_raw.trim()))?;
    let (mut q, rv_sa) = connect_with_timeout(rendezvous_addr, Duration::from_secs(4))?;
    let _ = q.set_nodelay(true);
    let _ = q.set_read_timeout(Some(RELAY_HANDSHAKE_TIMEOUT));
    let _ = q.set_write_timeout(Some(Duration::from_secs(4)));
    send_msg(&mut q, &RvMsg::Lookup { code: code.clone() })?;
    let addrs = match recv_msg(&mut q)? {
        RvMsg::Found { addrs, .. } => addrs,
        RvMsg::Error(e) => return Err(e),
        other => return Err(format!("unexpected lookup response {other:?}")),
    };
    drop(q);

    let env_relay = std::env::var("FARUST_FORCE_RELAY").is_ok_and(|v| v == "1");
    if !force_relay && !env_relay {
        for addr in &addrs {
            if let Ok(s) = TcpStream::connect_timeout(addr, direct_timeout) {
                let _ = s.set_nodelay(true);
                return Ok((s, ConnectVia::Direct(*addr)));
            }
        }
    }

    let mut r = TcpStream::connect_timeout(&rv_sa, RELAY_HANDSHAKE_TIMEOUT)
        .map_err(|e| format!("cannot connect to relay {rv_sa}: {e}"))?;
    let _ = r.set_nodelay(true);
    let _ = r.set_read_timeout(Some(RELAY_HANDSHAKE_TIMEOUT));
    let _ = r.set_write_timeout(Some(Duration::from_secs(4)));
    send_msg(&mut r, &RvMsg::ClientRelay { code })?;
    match recv_msg(&mut r)? {
        RvMsg::RelayReady { host_addr } => {
            let _ = r.set_read_timeout(None);
            let _ = r.set_write_timeout(None);
            Ok((r, ConnectVia::Relayed(host_addr)))
        }
        RvMsg::Error(e) => Err(e),
        other => Err(format!("unexpected relay response {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_codec_validates_and_normalizes() {
        let code = generate_code();
        assert_eq!(code.len(), 9);
        assert_eq!(code.as_bytes()[4], b'-');
        assert!(looks_like_code(&code));

        let squeezed = code.replace('-', "").to_ascii_lowercase();
        assert_eq!(normalize_code(&format!("  {squeezed}  ")).as_deref(), Some(code.as_str()));
        // Ambiguous characters (0, O, 1, I) and host:port strings are rejected.
        assert_eq!(normalize_code("ABCD-0000"), None);
        assert_eq!(normalize_code("ABCD-IIII"), None);
        assert_eq!(normalize_code("ABC"), None);
        assert!(!looks_like_code("127.0.0.1:15000"));
        assert!(!looks_like_code("host:1234"));
    }

    #[test]
    fn rendezvous_direct_relay_and_expiry() {
        let rv = RendezvousServer::bind("127.0.0.1:0").unwrap();
        let rv_addr = rv.local_addr().to_string();

        // Unknown code errors cleanly.
        let err = connect_via_code(&rv_addr, "ABCD-2345", false, Duration::from_millis(300)).unwrap_err();
        assert!(err.contains("unknown or expired"), "{err}");

        let host_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let host_bound = host_listener.local_addr().unwrap();
        let (tunnel_tx, tunnel_rx) = mpsc::channel();
        let mut reg = HostRegistration::register(
            &rv_addr,
            host_bound,
            "HostOne",
            "SCMP_001",
            Duration::from_secs(2),
            Box::new(move |s, a| {
                let _ = tunnel_tx.send((s, a));
            }),
        )
        .unwrap();
        let code = reg.code.clone();

        // 1. Direct path connects directly to host_listener.
        let (mut cs_direct, via) =
            connect_via_code(&rv_addr, &code.to_ascii_lowercase(), false, Duration::from_secs(1)).unwrap();
        assert!(!via.is_relayed());
        let (mut hs_direct, _) = host_listener.accept().unwrap();
        lobby_write_frame(&mut cs_direct, b"ping-direct").unwrap();
        assert_eq!(lobby_read_frame(&mut hs_direct).unwrap(), b"ping-direct");

        // 2. Forced-relay path splices frames cleanly in both directions through the server.
        let (mut cs_relay, via_r) =
            connect_via_code(&rv_addr, &code, true, Duration::from_millis(200)).unwrap();
        assert!(via_r.is_relayed());
        let (mut hs_relay, _) = tunnel_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        lobby_write_frame(&mut cs_relay, b"ping-relay").unwrap();
        assert_eq!(lobby_read_frame(&mut hs_relay).unwrap(), b"ping-relay");
        lobby_write_frame(&mut hs_relay, b"pong-relay").unwrap();
        assert_eq!(lobby_read_frame(&mut cs_relay).unwrap(), b"pong-relay");

        // 3. Dropping/closing the registration immediately expires the code.
        reg.close();
        let gone = connect_via_code(&rv_addr, &code, true, Duration::from_millis(200)).unwrap_err();
        assert!(gone.contains("unknown or expired"), "{gone}");
    }
}
