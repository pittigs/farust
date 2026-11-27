//! Phase 44 reliable-ordered UDP transport, MTU-safe fragmentation, Karn/Jacobson RTT
//! estimation, NAT hole punching probes, and an in-process seeded lossy datagram shim.
//!
//! Zero external crates (`std::net::UdpSocket` only). A single [`UdpMux`] owns one
//! [`UdpSocket`] (wrapped by [`NetemSocket`]) and runs a lightweight background pump so:
//! - Multiple peers share one host UDP socket (critical for NAT hole punching, where the
//!   router's public mapping is tied to the exact bound UDP socket).
//! - Cumulative + 64-bit selective ACKs (SACK), fast retransmit, Karn/Jacobson RTO backoff,
//!   keepalives, and Netem delayed/reordered queues run continuously without blocking the
//!   simulation or other peers.
//! - Frames larger than [`MAX_UDP_PAYLOAD`] (e.g. lobby state or save snapshots) are split
//!   across sequenced chunks and reassembled strictly in order up to [`MAX_FRAME_BYTES`].

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const UDP_MAGIC: [u8; 2] = *b"FU";
pub const UDP_PROTO_VERSION: u8 = 1;
pub const HEADER_LEN: usize = 24;
/// MTU-safe payload per data datagram (`24 hdr + 1 frag + 1180 data = 1205 <= 1232`).
pub const MAX_UDP_PAYLOAD: usize = 1180;
pub const MAX_FRAME_BYTES: usize = 2 * 1024 * 1024;
const MAX_DATAGRAM_BYTES: usize = 2048;

const PKT_DATA: u8 = 1;
const PKT_ACK: u8 = 2;
const PKT_SYN: u8 = 3;
const PKT_SYN_ACK: u8 = 4;
const PKT_FIN: u8 = 5;
pub const PKT_PUNCH: u8 = 6;
pub const PKT_PUNCH_ACK: u8 = 7;
pub const PKT_RV_REG: u8 = 8;
pub const PKT_RV_ACK: u8 = 9;

pub const ROLE_CLIENT: u8 = 0;
pub const ROLE_HOST: u8 = 1;

const FRAG_START: u8 = 0x01;
const FRAG_END: u8 = 0x02;

const MAX_SEND_WINDOW: usize = 128;
const MAX_RECV_WINDOW: u32 = 256;
const INITIAL_RTO_MS: f64 = 80.0;
const MIN_RTO_MS: f64 = 25.0;
const MAX_RTO_MS: f64 = 2000.0;
const FAST_RETX_THRESHOLD: u8 = 3;
const KEEPALIVE_INTERVAL: Duration = Duration::from_millis(1000);
const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(60);
const SYN_RETRY_INTERVAL: Duration = Duration::from_millis(60);
const PUNCH_RETRY_INTERVAL: Duration = Duration::from_millis(40);
const PUNCH_LINGER: Duration = Duration::from_millis(600);
const RV_REG_RETRY_INTERVAL: Duration = Duration::from_millis(60);
const RV_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(10);

static CONN_ID_SEQ: AtomicU32 = AtomicU32::new(1);

fn next_conn_id() -> u32 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let seq = CONN_ID_SEQ.fetch_add(1, Ordering::Relaxed) as u64;
    let pid = std::process::id() as u64;
    let mut x = nanos ^ seq.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (pid << 13);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let id = (x ^ (x >> 27)) as u32;
    if id == 0 { 1 } else { id }
}

// ---------------------------------------------------------------------------
// Wire codecs (reliable channel + punch + rendezvous UDP registration)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChanHeader {
    kind: u8,
    conn_id: u32,
    seq: u32,
    ack: u32,
    ack_bits: u64,
}

fn encode_chan_packet(h: ChanHeader, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(&UDP_MAGIC);
    out.push(UDP_PROTO_VERSION);
    out.push(h.kind);
    out.extend_from_slice(&h.conn_id.to_le_bytes());
    out.extend_from_slice(&h.seq.to_le_bytes());
    out.extend_from_slice(&h.ack.to_le_bytes());
    out.extend_from_slice(&h.ack_bits.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

fn decode_chan_header(buf: &[u8]) -> Option<(ChanHeader, &[u8])> {
    if buf.len() < HEADER_LEN || buf[0..2] != UDP_MAGIC || buf[2] != UDP_PROTO_VERSION {
        return None;
    }
    let kind = buf[3];
    if !matches!(kind, PKT_DATA | PKT_ACK | PKT_SYN | PKT_SYN_ACK | PKT_FIN) {
        return None;
    }
    let conn_id = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let seq = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let ack = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
    let ack_bits = u64::from_le_bytes([
        buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23],
    ]);
    Some((
        ChanHeader {
            kind,
            conn_id,
            seq,
            ack,
            ack_bits,
        },
        &buf[HEADER_LEN..],
    ))
}

/// Encodes a NAT hole-punch probe or confirmation datagram.
pub fn encode_punch_packet(ack: bool, token: u64, role: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(13);
    out.extend_from_slice(&UDP_MAGIC);
    out.push(UDP_PROTO_VERSION);
    out.push(if ack { PKT_PUNCH_ACK } else { PKT_PUNCH });
    out.extend_from_slice(&token.to_le_bytes());
    out.push(role);
    out
}

/// Decodes a NAT hole-punch probe/ack: returns `(is_ack, token, role)`.
pub fn decode_punch_packet(buf: &[u8]) -> Option<(bool, u64, u8)> {
    if buf.len() != 13 || buf[0..2] != UDP_MAGIC || buf[2] != UDP_PROTO_VERSION {
        return None;
    }
    let ack = match buf[3] {
        PKT_PUNCH => false,
        PKT_PUNCH_ACK => true,
        _ => return None,
    };
    let token = u64::from_le_bytes([
        buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11],
    ]);
    Some((ack, token, buf[12]))
}

/// Encodes a UDP registration datagram sent to the rendezvous server so it observes our NAT mapping.
pub fn encode_rv_reg(role: u8, token: u64, code: &str) -> Vec<u8> {
    let cb = code.as_bytes();
    let clen = cb.len().min(32);
    let mut out = Vec::with_capacity(14 + clen);
    out.extend_from_slice(&UDP_MAGIC);
    out.push(UDP_PROTO_VERSION);
    out.push(PKT_RV_REG);
    out.push(role);
    out.extend_from_slice(&token.to_le_bytes());
    out.push(clen as u8);
    out.extend_from_slice(&cb[..clen]);
    out
}

/// Decodes a rendezvous UDP registration datagram: returns `(role, token, code)`.
pub fn decode_rv_reg(buf: &[u8]) -> Option<(u8, u64, String)> {
    if buf.len() < 14 || buf[0..2] != UDP_MAGIC || buf[2] != UDP_PROTO_VERSION || buf[3] != PKT_RV_REG {
        return None;
    }
    let role = buf[4];
    let token = u64::from_le_bytes([
        buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11], buf[12],
    ]);
    let clen = buf[13] as usize;
    if buf.len() != 14 + clen {
        return None;
    }
    let code = std::str::from_utf8(&buf[14..]).ok()?.to_string();
    Some((role, token, code))
}

/// Encodes the rendezvous server's UDP registration reply carrying the observed public endpoint.
pub fn encode_rv_ack(token: u64, observed: SocketAddr) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    out.extend_from_slice(&UDP_MAGIC);
    out.push(UDP_PROTO_VERSION);
    out.push(PKT_RV_ACK);
    out.extend_from_slice(&token.to_le_bytes());
    match observed {
        SocketAddr::V4(v4) => {
            out.push(4);
            out.extend_from_slice(&v4.ip().octets());
            out.extend_from_slice(&v4.port().to_le_bytes());
        }
        SocketAddr::V6(v6) => {
            out.push(6);
            out.extend_from_slice(&v6.ip().octets());
            out.extend_from_slice(&v6.port().to_le_bytes());
        }
    }
    out
}

/// Decodes a rendezvous UDP registration reply: returns `(token, observed_addr)`.
pub fn decode_rv_ack(buf: &[u8]) -> Option<(u64, SocketAddr)> {
    if buf.len() < 13 || buf[0..2] != UDP_MAGIC || buf[2] != UDP_PROTO_VERSION || buf[3] != PKT_RV_ACK {
        return None;
    }
    let token = u64::from_le_bytes([
        buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11],
    ]);
    match buf[12] {
        4 if buf.len() == 19 => {
            let ip = Ipv4Addr::new(buf[13], buf[14], buf[15], buf[16]);
            let port = u16::from_le_bytes([buf[17], buf[18]]);
            Some((token, SocketAddr::new(IpAddr::V4(ip), port)))
        }
        6 if buf.len() == 31 => {
            let mut oct = [0u8; 16];
            oct.copy_from_slice(&buf[13..29]);
            let port = u16::from_le_bytes([buf[29], buf[30]]);
            Some((token, SocketAddr::new(IpAddr::V6(Ipv6Addr::from(oct)), port)))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// In-process lossy datagram shim (Netem)
// ---------------------------------------------------------------------------

/// Simulated network impairment for UDP loopback testing and `--netem` CLI runs.
#[derive(Debug, Clone, PartialEq)]
pub struct NetemConfig {
    /// Packet drop percentage (`0.0..100.0`).
    pub loss_pct: f64,
    /// Packet duplicate percentage (`0.0..=100.0`).
    pub dup_pct: f64,
    /// Reordering intensity (`0` = off, `1..=32` = max reorder delay slots).
    pub reorder_window: usize,
    /// Base artificial one-way delay in milliseconds.
    pub delay_ms: u64,
    /// Random additional one-way jitter in milliseconds.
    pub jitter_ms: u64,
    /// Deterministic PRNG seed.
    pub seed: u64,
}

impl Default for NetemConfig {
    fn default() -> Self {
        Self {
            loss_pct: 0.0,
            dup_pct: 0.0,
            reorder_window: 0,
            delay_ms: 0,
            jitter_ms: 0,
            seed: 0x5EED_4400,
        }
    }
}

impl NetemConfig {
    pub fn is_active(&self) -> bool {
        self.loss_pct > 0.0
            || self.dup_pct > 0.0
            || self.reorder_window > 0
            || self.delay_ms > 0
            || self.jitter_ms > 0
    }

    /// Parses a comma-separated spec such as `loss=5,dup=2,reorder=3,delay=40,jitter=10,seed=1`.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let mut cfg = Self::default();
        for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let Some((k, v)) = part.split_once('=') else {
                return Err(format!("invalid --netem entry '{part}' (expected key=value)"));
            };
            let k = k.trim();
            let v = v.trim();
            match k {
                "loss" => {
                    let x: f64 = v.parse().map_err(|_| format!("invalid netem loss '{v}'"))?;
                    if !(0.0..100.0).contains(&x) {
                        return Err(format!("netem loss {x} must be in 0..100"));
                    }
                    cfg.loss_pct = x;
                }
                "dup" => {
                    let x: f64 = v.parse().map_err(|_| format!("invalid netem dup '{v}'"))?;
                    if !(0.0..=100.0).contains(&x) {
                        return Err(format!("netem dup {x} must be in 0..=100"));
                    }
                    cfg.dup_pct = x;
                }
                "reorder" => {
                    let x: usize = v.parse().map_err(|_| format!("invalid netem reorder '{v}'"))?;
                    if x > 32 {
                        return Err(format!("netem reorder {x} exceeds max 32"));
                    }
                    cfg.reorder_window = x;
                }
                "delay" => {
                    let x: u64 = v.parse().map_err(|_| format!("invalid netem delay '{v}'"))?;
                    if x > 5000 {
                        return Err(format!("netem delay {x}ms exceeds max 5000ms"));
                    }
                    cfg.delay_ms = x;
                }
                "jitter" => {
                    let x: u64 = v.parse().map_err(|_| format!("invalid netem jitter '{v}'"))?;
                    if x > 5000 {
                        return Err(format!("netem jitter {x}ms exceeds max 5000ms"));
                    }
                    cfg.jitter_ms = x;
                }
                "seed" => {
                    cfg.seed = v.parse().map_err(|_| format!("invalid netem seed '{v}'"))?;
                }
                other => {
                    return Err(format!(
                        "unknown --netem key '{other}' (expected loss, dup, reorder, delay, jitter, seed)"
                    ));
                }
            }
        }
        Ok(cfg)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NetemStats {
    pub sent: u64,
    pub dropped: u64,
    pub duplicated: u64,
    pub reordered: u64,
}

struct NetemSocket {
    sock: UdpSocket,
    cfg: NetemConfig,
    rng: u64,
    delayed: Vec<(Instant, SocketAddr, Vec<u8>)>,
    stats: Arc<Mutex<NetemStats>>,
}

impl NetemSocket {
    fn new(sock: UdpSocket, cfg: NetemConfig, stats: Arc<Mutex<NetemStats>>) -> Self {
        let rng = if cfg.seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { cfg.seed };
        Self {
            sock,
            cfg,
            rng,
            delayed: Vec::new(),
            stats,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_pct(&mut self) -> f64 {
        // Uniform in [0.0, 100.0)
        let bits = self.next_u64() >> 11;
        (bits as f64 / ((1u64 << 53) as f64)) * 100.0
    }

    fn flush_due(&mut self, now: Instant) {
        if self.delayed.is_empty() {
            return;
        }
        let mut i = 0;
        while i < self.delayed.len() {
            if self.delayed[i].0 <= now {
                let (_, addr, data) = self.delayed.swap_remove(i);
                let _ = self.sock.send_to(&data, addr);
            } else {
                i += 1;
            }
        }
    }

    fn send_to(&mut self, buf: &[u8], addr: SocketAddr) -> io::Result<usize> {
        let now = Instant::now();
        self.flush_due(now);
        if !self.cfg.is_active() {
            if let Ok(mut st) = self.stats.lock() {
                st.sent += 1;
            }
            return self.sock.send_to(buf, addr);
        }
        let drop_it = self.cfg.loss_pct > 0.0 && self.next_pct() < self.cfg.loss_pct;
        if drop_it {
            if let Ok(mut st) = self.stats.lock() {
                st.sent += 1;
                st.dropped += 1;
            }
            return Ok(buf.len());
        }
        let dup_it = self.cfg.dup_pct > 0.0 && self.next_pct() < self.cfg.dup_pct;
        let copies = if dup_it { 2 } else { 1 };
        let mut reordered_inc = 0u64;
        for c in 0..copies {
            let mut wait_ms = self.cfg.delay_ms;
            if self.cfg.jitter_ms > 0 {
                wait_ms += self.next_u64() % (self.cfg.jitter_ms + 1);
            }
            if self.cfg.reorder_window > 0 && self.next_u64() % 3 == 0 {
                let extra = 2 + (self.next_u64() % ((self.cfg.reorder_window as u64) * 4).max(1));
                wait_ms += extra;
                reordered_inc += 1;
            }
            if c == 1 && wait_ms == 0 {
                wait_ms = 1 + (self.next_u64() % 3);
            }
            if wait_ms == 0 {
                let _ = self.sock.send_to(buf, addr);
            } else {
                self.delayed
                    .push((now + Duration::from_millis(wait_ms), addr, buf.to_vec()));
            }
        }
        if let Ok(mut st) = self.stats.lock() {
            st.sent += 1;
            if dup_it {
                st.duplicated += 1;
            }
            st.reordered += reordered_inc;
        }
        Ok(buf.len())
    }

    fn recv_from(&mut self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.flush_due(Instant::now());
        self.sock.recv_from(buf)
    }
}

// ---------------------------------------------------------------------------
// Karn/Jacobson RTT estimator
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct RttEstimator {
    srtt_ms: Option<f64>,
    rttvar_ms: f64,
    rto_ms: f64,
}

impl Default for RttEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl RttEstimator {
    pub fn new() -> Self {
        Self {
            srtt_ms: None,
            rttvar_ms: 0.0,
            rto_ms: INITIAL_RTO_MS,
        }
    }

    /// Updates SRTT/RTTVAR/RTO with a fresh RTT sample (caller enforces Karn's rule:
    /// never call for retransmitted packets).
    pub fn observe(&mut self, sample_ms: f64) {
        let r = sample_ms.clamp(0.1, 10_000.0);
        match self.srtt_ms {
            None => {
                self.srtt_ms = Some(r);
                self.rttvar_ms = r * 0.5;
            }
            Some(srtt) => {
                self.rttvar_ms = 0.75 * self.rttvar_ms + 0.25 * (srtt - r).abs();
                self.srtt_ms = Some(0.875 * srtt + 0.125 * r);
            }
        }
        let srtt = self.srtt_ms.unwrap_or(r);
        self.rto_ms = (srtt + (4.0 * self.rttvar_ms).max(10.0)).clamp(MIN_RTO_MS, MAX_RTO_MS);
    }

    pub fn rto_ms(&self) -> f64 {
        self.rto_ms
    }

    pub fn srtt_ms(&self) -> Option<f64> {
        self.srtt_ms
    }

    pub fn rttvar_ms(&self) -> f64 {
        self.rttvar_ms
    }
}

/// Live per-channel RTT and reliability counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChannelStats {
    pub srtt_ms: f64,
    pub rttvar_ms: f64,
    pub rto_ms: f64,
    pub has_rtt: bool,
    pub packets_sent: u64,
    pub retransmits: u64,
    pub dupes_dropped: u64,
}

// ---------------------------------------------------------------------------
// Per-peer reliable-ordered channel state (owned by UdpMux thread)
// ---------------------------------------------------------------------------

struct SentSeg {
    frag_flags: u8,
    payload: Vec<u8>,
    first_sent: Instant,
    last_sent: Instant,
    rto_ms: f64,
    send_count: u32,
    fast_acks: u8,
}

enum OutCmd {
    Frame(Vec<u8>),
    Close,
}

struct PeerState {
    conn_id: u32,
    remote: SocketAddr,
    established: bool,
    connect_deadline: Option<Instant>,
    last_syn: Instant,
    connect_reply: Option<Sender<Result<(), String>>>,
    next_seq: u32,
    send_unacked: BTreeMap<u32, SentSeg>,
    pending_chunks: VecDeque<(u8, Vec<u8>)>,
    recv_ack: u32,
    recv_buf: BTreeMap<u32, (u8, Vec<u8>)>,
    reasm: Vec<u8>,
    reasm_active: bool,
    ack_dirty: bool,
    rtt: RttEstimator,
    stats: Arc<Mutex<ChannelStats>>,
    last_recv: Instant,
    last_send: Instant,
    idle_timeout: Arc<Mutex<Option<Duration>>>,
    out_rx: Receiver<OutCmd>,
    in_tx: Sender<Result<Vec<u8>, String>>,
    closed: bool,
}

impl PeerState {
    fn new(
        conn_id: u32,
        remote: SocketAddr,
        established: bool,
        connect_deadline: Option<Instant>,
        connect_reply: Option<Sender<Result<(), String>>>,
        out_rx: Receiver<OutCmd>,
        in_tx: Sender<Result<Vec<u8>, String>>,
        stats: Arc<Mutex<ChannelStats>>,
        idle_timeout: Arc<Mutex<Option<Duration>>>,
    ) -> Self {
        let now = Instant::now();
        Self {
            conn_id,
            remote,
            established,
            connect_deadline,
            last_syn: now - SYN_RETRY_INTERVAL,
            connect_reply,
            next_seq: 1,
            send_unacked: BTreeMap::new(),
            pending_chunks: VecDeque::new(),
            recv_ack: 0,
            recv_buf: BTreeMap::new(),
            reasm: Vec::new(),
            reasm_active: false,
            ack_dirty: false,
            rtt: RttEstimator::new(),
            stats,
            last_recv: now,
            last_send: now,
            idle_timeout,
            out_rx,
            in_tx,
            closed: false,
        }
    }

    fn compute_ack_bits(&self) -> u64 {
        let mut bits = 0u64;
        for i in 0..64u32 {
            let s = self.recv_ack.wrapping_add(1 + i);
            if self.recv_buf.contains_key(&s) {
                bits |= 1u64 << i;
            }
        }
        bits
    }

    fn sync_stats(&self, sent_inc: u64, retx_inc: u64, dup_inc: u64) {
        if let Ok(mut st) = self.stats.lock() {
            if let Some(srtt) = self.rtt.srtt_ms() {
                st.srtt_ms = srtt;
                st.rttvar_ms = self.rtt.rttvar_ms();
                st.has_rtt = true;
            }
            st.rto_ms = self.rtt.rto_ms();
            st.packets_sent += sent_inc;
            st.retransmits += retx_inc;
            st.dupes_dropped += dup_inc;
        }
    }

    fn enqueue_frame(&mut self, frame: Vec<u8>) {
        if frame.is_empty() || frame.len() > MAX_FRAME_BYTES {
            return;
        }
        if frame.len() <= MAX_UDP_PAYLOAD {
            self.pending_chunks.push_back((FRAG_START | FRAG_END, frame));
            return;
        }
        let chunks: Vec<&[u8]> = frame.chunks(MAX_UDP_PAYLOAD).collect();
        let last = chunks.len() - 1;
        for (idx, chunk) in chunks.into_iter().enumerate() {
            let mut flags = 0u8;
            if idx == 0 {
                flags |= FRAG_START;
            }
            if idx == last {
                flags |= FRAG_END;
            }
            self.pending_chunks.push_back((flags, chunk.to_vec()));
        }
    }

    fn send_data_seg(&mut self, sock: &mut NetemSocket, seq: u32, frag_flags: u8, payload: &[u8], now: Instant) {
        let mut body = Vec::with_capacity(1 + payload.len());
        body.push(frag_flags);
        body.extend_from_slice(payload);
        let pkt = encode_chan_packet(
            ChanHeader {
                kind: PKT_DATA,
                conn_id: self.conn_id,
                seq,
                ack: self.recv_ack,
                ack_bits: self.compute_ack_bits(),
            },
            &body,
        );
        let _ = sock.send_to(&pkt, self.remote);
        self.last_send = now;
        self.ack_dirty = false;
    }

    fn send_ack_only(&mut self, sock: &mut NetemSocket, now: Instant) {
        let pkt = encode_chan_packet(
            ChanHeader {
                kind: PKT_ACK,
                conn_id: self.conn_id,
                seq: 0,
                ack: self.recv_ack,
                ack_bits: self.compute_ack_bits(),
            },
            &[],
        );
        let _ = sock.send_to(&pkt, self.remote);
        self.last_send = now;
        self.ack_dirty = false;
    }

    fn send_fin(&mut self, sock: &mut NetemSocket, now: Instant) {
        let pkt = encode_chan_packet(
            ChanHeader {
                kind: PKT_FIN,
                conn_id: self.conn_id,
                seq: 0,
                ack: self.recv_ack,
                ack_bits: self.compute_ack_bits(),
            },
            &[],
        );
        // Send twice so a single dropped FIN does not force an idle timeout.
        let _ = sock.send_to(&pkt, self.remote);
        let _ = sock.send_to(&pkt, self.remote);
        self.last_send = now;
    }

    fn drain_send_window(&mut self, sock: &mut NetemSocket, now: Instant) {
        if !self.established || self.closed {
            return;
        }
        let mut sent_inc = 0u64;
        while self.send_unacked.len() < MAX_SEND_WINDOW {
            let Some((frag_flags, payload)) = self.pending_chunks.pop_front() else {
                break;
            };
            let seq = self.next_seq;
            self.next_seq = self.next_seq.wrapping_add(1).max(1);
            self.send_data_seg(sock, seq, frag_flags, &payload, now);
            self.send_unacked.insert(
                seq,
                SentSeg {
                    frag_flags,
                    payload,
                    first_sent: now,
                    last_sent: now,
                    rto_ms: self.rtt.rto_ms(),
                    send_count: 1,
                    fast_acks: 0,
                },
            );
            sent_inc += 1;
        }
        if sent_inc > 0 {
            self.sync_stats(sent_inc, 0, 0);
        }
    }

    fn process_acks(&mut self, ack: u32, ack_bits: u64, sock: &mut NetemSocket, now: Instant) {
        let mut highest_acked = ack;
        // Cumulative ACKs: seq <= ack
        let cum_seqs: Vec<u32> = self
            .send_unacked
            .keys()
            .copied()
            .filter(|&s| s <= ack)
            .collect();
        for seq in cum_seqs {
            if let Some(seg) = self.send_unacked.remove(&seq)
                && seg.send_count == 1
            {
                let sample_ms = now.duration_since(seg.first_sent).as_secs_f64() * 1000.0;
                self.rtt.observe(sample_ms);
            }
        }
        // Selective ACKs: bits 0..64 represent ack + 1 + i
        for i in 0..64u32 {
            if (ack_bits & (1u64 << i)) != 0 {
                let sseq = ack.wrapping_add(1 + i);
                if let Some(seg) = self.send_unacked.remove(&sseq) {
                    highest_acked = highest_acked.max(sseq);
                    if seg.send_count == 1 {
                        let sample_ms = now.duration_since(seg.first_sent).as_secs_f64() * 1000.0;
                        self.rtt.observe(sample_ms);
                    }
                }
            }
        }
        // Fast retransmit for segments below highest SACKed sequence
        let mut fast_retx: Vec<(u32, u8, Vec<u8>)> = Vec::new();
        for (&seq, seg) in &mut self.send_unacked {
            if seq < highest_acked {
                seg.fast_acks = seg.fast_acks.saturating_add(1);
                if seg.fast_acks >= FAST_RETX_THRESHOLD
                    && now.duration_since(seg.last_sent) >= Duration::from_millis(10)
                {
                    seg.last_sent = now;
                    seg.send_count += 1;
                    seg.fast_acks = 0;
                    fast_retx.push((seq, seg.frag_flags, seg.payload.clone()));
                }
            }
        }
        let retx_count = fast_retx.len() as u64;
        for (seq, flags, payload) in fast_retx {
            self.send_data_seg(sock, seq, flags, &payload, now);
        }
        self.sync_stats(retx_count, retx_count, 0);
        self.drain_send_window(sock, now);
    }

    fn deliver_in_order(&mut self) {
        while let Some((flags, chunk)) = self.recv_buf.remove(&(self.recv_ack + 1)) {
            self.recv_ack += 1;
            if (flags & FRAG_START) != 0 {
                self.reasm.clear();
                self.reasm_active = true;
            }
            if !self.reasm_active {
                continue;
            }
            if self.reasm.len() + chunk.len() > MAX_FRAME_BYTES {
                self.reasm.clear();
                self.reasm_active = false;
                let _ = self.in_tx.send(Err("udp frame exceeds MAX_FRAME_BYTES".into()));
                self.closed = true;
                return;
            }
            self.reasm.extend_from_slice(&chunk);
            if (flags & FRAG_END) != 0 {
                let frame = std::mem::take(&mut self.reasm);
                self.reasm_active = false;
                if self.in_tx.send(Ok(frame)).is_err() {
                    self.closed = true;
                    return;
                }
            }
        }
    }

    fn on_packet(&mut self, hdr: ChanHeader, payload: &[u8], src: SocketAddr, sock: &mut NetemSocket, now: Instant) {
        self.last_recv = now;
        self.remote = src;
        match hdr.kind {
            PKT_SYN_ACK => {
                if !self.established {
                    self.established = true;
                    self.connect_deadline = None;
                    if let Some(tx) = self.connect_reply.take() {
                        let _ = tx.send(Ok(()));
                    }
                    self.drain_send_window(sock, now);
                }
            }
            PKT_SYN => {
                // Duplicate SYN from client (our SYN_ACK was lost); re-ack.
                let reply = encode_chan_packet(
                    ChanHeader {
                        kind: PKT_SYN_ACK,
                        conn_id: self.conn_id,
                        seq: 0,
                        ack: self.recv_ack,
                        ack_bits: self.compute_ack_bits(),
                    },
                    &[],
                );
                let _ = sock.send_to(&reply, self.remote);
                self.last_send = now;
            }
            PKT_DATA => {
                if !self.established {
                    // Implicit SYN_ACK if SYN_ACK was dropped before server's first data frame.
                    self.established = true;
                    self.connect_deadline = None;
                    if let Some(tx) = self.connect_reply.take() {
                        let _ = tx.send(Ok(()));
                    }
                }
                self.process_acks(hdr.ack, hdr.ack_bits, sock, now);
                if payload.is_empty() {
                    return;
                }
                let frag_flags = payload[0];
                let body = &payload[1..];
                let seq = hdr.seq;
                self.ack_dirty = true;
                if seq == 0 || seq <= self.recv_ack || self.recv_buf.contains_key(&seq) {
                    self.sync_stats(0, 0, 1);
                } else if seq <= self.recv_ack.saturating_add(MAX_RECV_WINDOW) {
                    self.recv_buf.insert(seq, (frag_flags, body.to_vec()));
                    self.deliver_in_order();
                }
            }
            PKT_ACK => {
                if self.established {
                    self.process_acks(hdr.ack, hdr.ack_bits, sock, now);
                }
            }
            PKT_FIN => {
                self.closed = true;
                let _ = self.in_tx.send(Err("udp peer closed connection".into()));
            }
            _ => {}
        }
    }

    fn tick(&mut self, sock: &mut NetemSocket, now: Instant) {
        if self.closed {
            return;
        }
        // Drain outbound application commands.
        loop {
            match self.out_rx.try_recv() {
                Ok(OutCmd::Frame(f)) => self.enqueue_frame(f),
                Ok(OutCmd::Close) => {
                    self.send_fin(sock, now);
                    self.closed = true;
                    return;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.send_fin(sock, now);
                    self.closed = true;
                    return;
                }
            }
        }
        // Pending client SYN handshake.
        if !self.established {
            if let Some(dl) = self.connect_deadline
                && now >= dl
            {
                if let Some(tx) = self.connect_reply.take() {
                    let _ = tx.send(Err(format!("udp connect to {} timed out", self.remote)));
                }
                self.closed = true;
                return;
            }
            if now.duration_since(self.last_syn) >= SYN_RETRY_INTERVAL {
                let syn = encode_chan_packet(
                    ChanHeader {
                        kind: PKT_SYN,
                        conn_id: self.conn_id,
                        seq: 0,
                        ack: 0,
                        ack_bits: 0,
                    },
                    &[],
                );
                let _ = sock.send_to(&syn, self.remote);
                self.last_syn = now;
                self.last_send = now;
            }
            return;
        }

        // Idle timeout check.
        let timeout = self
            .idle_timeout
            .lock()
            .ok()
            .and_then(|g| *g)
            .unwrap_or(DEFAULT_IDLE_TIMEOUT);
        if now.duration_since(self.last_recv) >= timeout {
            let _ = self.in_tx.send(Err(format!(
                "udp peer {} timed out after {}ms",
                self.remote,
                timeout.as_millis()
            )));
            self.closed = true;
            return;
        }

        // Fill send window from queued chunks.
        self.drain_send_window(sock, now);

        // RTO retransmissions with Karn exponential backoff.
        let mut timed_out: Vec<(u32, u8, Vec<u8>)> = Vec::new();
        for (&seq, seg) in &mut self.send_unacked {
            let elapsed_ms = now.duration_since(seg.last_sent).as_secs_f64() * 1000.0;
            if elapsed_ms >= seg.rto_ms {
                seg.last_sent = now;
                seg.send_count += 1;
                seg.rto_ms = (seg.rto_ms * 2.0).min(MAX_RTO_MS);
                seg.fast_acks = 0;
                timed_out.push((seq, seg.frag_flags, seg.payload.clone()));
            }
        }
        let retx = timed_out.len() as u64;
        for (seq, flags, payload) in timed_out {
            self.send_data_seg(sock, seq, flags, &payload, now);
        }
        if retx > 0 {
            self.sync_stats(retx, retx, 0);
        }

        // Send immediate ACK if data arrived and wasn't piggybacked, or periodic keepalive.
        if self.ack_dirty || now.duration_since(self.last_send) >= KEEPALIVE_INTERVAL {
            self.send_ack_only(sock, now);
        }
    }
}

// ---------------------------------------------------------------------------
// Active punch / rendezvous registration trackers inside UdpMux
// ---------------------------------------------------------------------------

struct ActivePunch {
    token: u64,
    local_role: u8,
    targets: Vec<SocketAddr>,
    deadline: Instant,
    last_send: Instant,
    reply: Option<Sender<Result<SocketAddr, String>>>,
    linger_until: Option<Instant>,
}

struct PendingRvReg {
    role: u8,
    token: u64,
    code: String,
    rv_addr: SocketAddr,
    deadline: Instant,
    last_send: Instant,
    reply: Sender<Result<SocketAddr, String>>,
}

struct RvKeepalive {
    role: u8,
    token: u64,
    code: String,
    rv_addr: SocketAddr,
    last_send: Instant,
}

enum MuxCmd {
    Connect {
        conn_id: u32,
        remote: SocketAddr,
        timeout: Duration,
        out_rx: Receiver<OutCmd>,
        in_tx: Sender<Result<Vec<u8>, String>>,
        stats: Arc<Mutex<ChannelStats>>,
        idle_timeout: Arc<Mutex<Option<Duration>>>,
        reply: Sender<Result<(), String>>,
    },
    SetAccepting(bool),
    StartPunch {
        token: u64,
        role: u8,
        targets: Vec<SocketAddr>,
        timeout: Duration,
        reply: Sender<Result<SocketAddr, String>>,
    },
    RegisterRv {
        rv_addr: SocketAddr,
        role: u8,
        token: u64,
        code: String,
        timeout: Duration,
        keepalive: bool,
        reply: Sender<Result<SocketAddr, String>>,
    },
    ClearRvKeepalive,
}

// ---------------------------------------------------------------------------
// Public UdpMux + UdpChannel handles
// ---------------------------------------------------------------------------

/// Shared UDP multiplexer owning a single `UdpSocket` (with optional [`NetemConfig`]).
pub struct UdpMux {
    local_addr: SocketAddr,
    wake_addr: SocketAddr,
    wake_sock: Option<UdpSocket>,
    cmd_tx: Sender<MuxCmd>,
    accept_rx: Mutex<Receiver<UdpChannel>>,
    stop: Arc<AtomicBool>,
    netem_stats: Arc<Mutex<NetemStats>>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl UdpMux {
    /// Binds a UDP socket on `addr` with optional [`NetemConfig`] impairments.
    pub fn bind(addr: &str, netem: NetemConfig) -> Result<Arc<Self>, String> {
        let sock = UdpSocket::bind(addr).map_err(|e| format!("failed to bind UDP {addr}: {e}"))?;
        Self::from_socket(sock, netem)
    }

    pub fn from_socket(sock: UdpSocket, netem: NetemConfig) -> Result<Arc<Self>, String> {
        let local_addr = sock.local_addr().map_err(|e| format!("udp local_addr: {e}"))?;
        sock.set_read_timeout(Some(Duration::from_millis(4)))
            .map_err(|e| format!("udp set_read_timeout: {e}"))?;
        let wake_ip = match local_addr.ip() {
            ip if ip.is_unspecified() && local_addr.is_ipv4() => IpAddr::V4(Ipv4Addr::LOCALHOST),
            ip if ip.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
            ip => ip,
        };
        let wake_addr = SocketAddr::new(wake_ip, local_addr.port());
        let wake_sock = UdpSocket::bind(if wake_addr.is_ipv4() {
            "127.0.0.1:0"
        } else {
            "[::1]:0"
        })
        .ok();

        let (cmd_tx, cmd_rx) = mpsc::channel::<MuxCmd>();
        let (accept_tx, accept_rx) = mpsc::channel::<UdpChannel>();
        let stop = Arc::new(AtomicBool::new(false));
        let netem_stats = Arc::new(Mutex::new(NetemStats::default()));

        let mux = Arc::new(Self {
            local_addr,
            wake_addr,
            wake_sock,
            cmd_tx,
            accept_rx: Mutex::new(accept_rx),
            stop: stop.clone(),
            netem_stats: netem_stats.clone(),
            thread: Mutex::new(None),
        });
        let mux_weak = Arc::downgrade(&mux);
        let ns_sock = NetemSocket::new(sock, netem, netem_stats);
        let handle = std::thread::spawn(move || {
            mux_loop(ns_sock, cmd_rx, accept_tx, stop, mux_weak);
        });
        if let Ok(mut g) = mux.thread.lock() {
            *g = Some(handle);
        }
        Ok(mux)
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub fn netem_stats(&self) -> NetemStats {
        self.netem_stats.lock().map(|g| *g).unwrap_or_default()
    }

    fn wake(&self) {
        if let Some(s) = &self.wake_sock {
            let _ = s.send_to(&[0u8], self.wake_addr);
        }
    }

    /// Enables or disables accepting incoming `PKT_SYN` connections on this socket.
    pub fn set_accepting(&self, accepting: bool) {
        let _ = self.cmd_tx.send(MuxCmd::SetAccepting(accepting));
        self.wake();
    }

    /// Accepts the next incoming UDP channel (blocks up to `timeout`, or indefinitely if `None`).
    pub fn accept(&self, timeout: Option<Duration>) -> Result<UdpChannel, String> {
        let rx = self.accept_rx.lock().map_err(|_| "udp accept lock poisoned".to_string())?;
        match timeout {
            Some(t) => rx
                .recv_timeout(t)
                .map_err(|e| format!("udp accept timed out: {e}")),
            None => rx.recv().map_err(|_| "udp mux closed".to_string()),
        }
    }

    /// Connects a new reliable-ordered [`UdpChannel`] to `remote` over this mux's socket.
    pub fn connect(self: &Arc<Self>, remote: SocketAddr, timeout: Duration) -> Result<UdpChannel, String> {
        let conn_id = next_conn_id();
        let (out_tx, out_rx) = mpsc::channel::<OutCmd>();
        let (in_tx, in_rx) = mpsc::channel::<Result<Vec<u8>, String>>();
        let (rep_tx, rep_rx) = mpsc::channel::<Result<(), String>>();
        let stats = Arc::new(Mutex::new(ChannelStats::default()));
        let idle_timeout = Arc::new(Mutex::new(Some(DEFAULT_IDLE_TIMEOUT)));
        self.cmd_tx
            .send(MuxCmd::Connect {
                conn_id,
                remote,
                timeout,
                out_rx,
                in_tx,
                stats: stats.clone(),
                idle_timeout: idle_timeout.clone(),
                reply: rep_tx,
            })
            .map_err(|_| "udp mux closed".to_string())?;
        self.wake();
        match rep_rx.recv_timeout(timeout + Duration::from_millis(250)) {
            Ok(Ok(())) => Ok(UdpChannel {
                _mux: self.clone(),
                remote,
                out_tx,
                in_rx: Arc::new(Mutex::new(in_rx)),
                stats,
                idle_timeout,
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(format!("udp connect to {remote} timed out")),
        }
    }

    /// Initiates simultaneous NAT hole punching toward `targets`; resolves to the first
    /// remote address that responds with a matching punch token.
    pub fn start_punch(
        &self,
        token: u64,
        role: u8,
        targets: Vec<SocketAddr>,
        timeout: Duration,
    ) -> Receiver<Result<SocketAddr, String>> {
        let (tx, rx) = mpsc::channel();
        let _ = self.cmd_tx.send(MuxCmd::StartPunch {
            token,
            role,
            targets,
            timeout,
            reply: tx,
        });
        self.wake();
        rx
    }

    /// Blocking NAT hole punch helper returning the punched remote `SocketAddr`.
    pub fn punch(
        &self,
        token: u64,
        role: u8,
        targets: Vec<SocketAddr>,
        timeout: Duration,
    ) -> Result<SocketAddr, String> {
        let rx = self.start_punch(token, role, targets, timeout);
        match rx.recv_timeout(timeout + Duration::from_millis(250)) {
            Ok(res) => res,
            Err(_) => Err("udp hole punch timed out".into()),
        }
    }

    /// Sends UDP registration packets to the rendezvous server until it replies with our
    /// server-observed public UDP address. If `keepalive` is true, refreshes every 10s.
    pub fn register_rendezvous_udp(
        &self,
        rv_addr: SocketAddr,
        role: u8,
        token: u64,
        code: String,
        timeout: Duration,
        keepalive: bool,
    ) -> Result<SocketAddr, String> {
        let (tx, rx) = mpsc::channel();
        self.cmd_tx
            .send(MuxCmd::RegisterRv {
                rv_addr,
                role,
                token,
                code,
                timeout,
                keepalive,
                reply: tx,
            })
            .map_err(|_| "udp mux closed".to_string())?;
        self.wake();
        match rx.recv_timeout(timeout + Duration::from_millis(250)) {
            Ok(res) => res,
            Err(_) => Err(format!("rendezvous UDP registration to {rv_addr} timed out")),
        }
    }

    pub fn clear_rendezvous_keepalive(&self) {
        let _ = self.cmd_tx.send(MuxCmd::ClearRvKeepalive);
        self.wake();
    }
}

impl Drop for UdpMux {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.wake();
        if let Ok(mut g) = self.thread.lock()
            && let Some(h) = g.take()
        {
            let _ = h.join();
        }
    }
}

/// Reliable, ordered, MTU-fragmented frame channel over UDP.
pub struct UdpChannel {
    _mux: Arc<UdpMux>,
    remote: SocketAddr,
    out_tx: Sender<OutCmd>,
    in_rx: Arc<Mutex<Receiver<Result<Vec<u8>, String>>>>,
    stats: Arc<Mutex<ChannelStats>>,
    idle_timeout: Arc<Mutex<Option<Duration>>>,
}

impl UdpChannel {
    /// Connects directly to `addr` on a fresh ephemeral UDP socket (with optional `netem`).
    pub fn connect(addr: &str, netem: NetemConfig, timeout: Duration) -> Result<Self, String> {
        let addrs: Vec<SocketAddr> = addr
            .to_socket_addrs()
            .map_err(|e| format!("cannot resolve UDP {addr}: {e}"))?
            .collect();
        let mut last = format!("no address for UDP {addr}");
        for sa in addrs {
            let bind_local = match sa {
                SocketAddr::V4(v4) if v4.ip().is_loopback() => "127.0.0.1:0",
                SocketAddr::V4(_) => "0.0.0.0:0",
                SocketAddr::V6(v6) if v6.ip().is_loopback() => "[::1]:0",
                SocketAddr::V6(_) => "[::]:0",
            };
            let mux = UdpMux::bind(bind_local, netem.clone())?;
            match mux.connect(sa, timeout) {
                Ok(ch) => return Ok(ch),
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    pub fn peer_addr(&self) -> SocketAddr {
        self.remote
    }

    pub fn set_read_timeout(&mut self, timeout: Option<Duration>) {
        if let Ok(mut g) = self.idle_timeout.lock() {
            *g = timeout;
        }
    }

    pub fn write_frame(&mut self, payload: &[u8]) -> Result<(), String> {
        if payload.is_empty() || payload.len() > MAX_FRAME_BYTES {
            return Err(format!("invalid udp frame length {}", payload.len()));
        }
        self.out_tx
            .send(OutCmd::Frame(payload.to_vec()))
            .map_err(|_| "udp channel closed".to_string())?;
        self._mux.wake();
        Ok(())
    }

    pub fn read_frame(&mut self) -> Result<Vec<u8>, String> {
        let timeout = self.idle_timeout.lock().ok().and_then(|g| *g);
        let rx = self.in_rx.lock().map_err(|_| "udp recv lock poisoned".to_string())?;
        let res = match timeout {
            Some(t) => rx
                .recv_timeout(t)
                .map_err(|e| format!("udp read frame: {e}"))?,
            None => rx
                .recv()
                .map_err(|_| "udp channel closed".to_string())?,
        };
        res
    }

    pub fn shutdown(&self) {
        let _ = self.out_tx.send(OutCmd::Close);
        self._mux.wake();
    }

    pub fn stats(&self) -> ChannelStats {
        self.stats.lock().map(|g| *g).unwrap_or_default()
    }

    pub fn rtt_stats(&self) -> Option<(f64, f64)> {
        let st = self.stats();
        st.has_rtt.then_some((st.srtt_ms, st.rttvar_ms))
    }

    pub fn netem_stats(&self) -> NetemStats {
        self._mux.netem_stats()
    }
}

impl Drop for UdpChannel {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Multiplexer event loop
// ---------------------------------------------------------------------------

fn mux_loop(
    mut sock: NetemSocket,
    cmd_rx: Receiver<MuxCmd>,
    accept_tx: Sender<UdpChannel>,
    stop: Arc<AtomicBool>,
    mux_weak: std::sync::Weak<UdpMux>,
) {
    let mut peers: HashMap<u32, PeerState> = HashMap::new();
    let mut accepting = false;
    let mut punches: Vec<ActivePunch> = Vec::new();
    let mut rv_regs: Vec<PendingRvReg> = Vec::new();
    let mut rv_keepalive: Option<RvKeepalive> = None;
    let mut buf = [0u8; MAX_DATAGRAM_BYTES];

    while !stop.load(Ordering::Relaxed) {
        let now = Instant::now();
        // 1. Drain control commands.
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                MuxCmd::Connect {
                    conn_id,
                    remote,
                    timeout,
                    out_rx,
                    in_tx,
                    stats,
                    idle_timeout,
                    reply,
                } => {
                    let peer = PeerState::new(
                        conn_id,
                        remote,
                        false,
                        Some(now + timeout),
                        Some(reply),
                        out_rx,
                        in_tx,
                        stats,
                        idle_timeout,
                    );
                    peers.insert(conn_id, peer);
                }
                MuxCmd::SetAccepting(a) => accepting = a,
                MuxCmd::StartPunch {
                    token,
                    role,
                    targets,
                    timeout,
                    reply,
                } => {
                    punches.push(ActivePunch {
                        token,
                        local_role: role,
                        targets,
                        deadline: now + timeout,
                        last_send: now - PUNCH_RETRY_INTERVAL,
                        reply: Some(reply),
                        linger_until: None,
                    });
                }
                MuxCmd::RegisterRv {
                    rv_addr,
                    role,
                    token,
                    code,
                    timeout,
                    keepalive,
                    reply,
                } => {
                    if keepalive {
                        rv_keepalive = Some(RvKeepalive {
                            role,
                            token,
                            code: code.clone(),
                            rv_addr,
                            last_send: now,
                        });
                    }
                    rv_regs.push(PendingRvReg {
                        role,
                        token,
                        code,
                        rv_addr,
                        deadline: now + timeout,
                        last_send: now - RV_REG_RETRY_INTERVAL,
                        reply,
                    });
                }
                MuxCmd::ClearRvKeepalive => {
                    rv_keepalive = None;
                }
            }
        }

        // 2. Receive incoming datagrams (non-blocking burst after first blocking recv up to 4ms).
        let mut first = true;
        loop {
            if !first {
                let _ = sock.sock.set_nonblocking(true);
            }
            let recv_res = sock.recv_from(&mut buf);
            if !first {
                let _ = sock.sock.set_nonblocking(false);
            }
            first = false;
            let (n, src) = match recv_res {
                Ok(v) => v,
                Err(_) => break,
            };
            let now = Instant::now();
            let pkt = &buf[..n];
            if pkt.len() < 4 || pkt[0..2] != UDP_MAGIC || pkt[2] != UDP_PROTO_VERSION {
                continue;
            }
            match pkt[3] {
                PKT_PUNCH | PKT_PUNCH_ACK => {
                    if let Some((is_ack, token, remote_role)) = decode_punch_packet(pkt) {
                        for p in &mut punches {
                            if p.token == token && remote_role != p.local_role {
                                if !is_ack {
                                    let reply_pkt = encode_punch_packet(true, token, p.local_role);
                                    let _ = sock.send_to(&reply_pkt, src);
                                }
                                if let Some(tx) = p.reply.take() {
                                    let _ = tx.send(Ok(src));
                                    p.linger_until = Some(now + PUNCH_LINGER);
                                }
                            }
                        }
                    }
                }
                PKT_RV_ACK => {
                    if let Some((token, observed)) = decode_rv_ack(pkt) {
                        let mut i = 0;
                        while i < rv_regs.len() {
                            if rv_regs[i].token == token {
                                let reg = rv_regs.swap_remove(i);
                                let _ = reg.reply.send(Ok(observed));
                            } else {
                                i += 1;
                            }
                        }
                    }
                }
                PKT_DATA | PKT_ACK | PKT_SYN | PKT_SYN_ACK | PKT_FIN => {
                    if let Some((hdr, payload)) = decode_chan_header(pkt) {
                        if let Some(peer) = peers.get_mut(&hdr.conn_id) {
                            peer.on_packet(hdr, payload, src, &mut sock, now);
                        } else if hdr.kind == PKT_SYN && accepting {
                            if let Some(mux_arc) = mux_weak.upgrade() {
                                let (out_tx, out_rx) = mpsc::channel();
                                let (in_tx, in_rx) = mpsc::channel();
                                let stats = Arc::new(Mutex::new(ChannelStats::default()));
                                let idle_timeout = Arc::new(Mutex::new(Some(DEFAULT_IDLE_TIMEOUT)));
                                let mut peer = PeerState::new(
                                    hdr.conn_id,
                                    src,
                                    true,
                                    None,
                                    None,
                                    out_rx,
                                    in_tx,
                                    stats.clone(),
                                    idle_timeout.clone(),
                                );
                                // Reply SYN_ACK immediately.
                                let syn_ack = encode_chan_packet(
                                    ChanHeader {
                                        kind: PKT_SYN_ACK,
                                        conn_id: hdr.conn_id,
                                        seq: 0,
                                        ack: 0,
                                        ack_bits: 0,
                                    },
                                    &[],
                                );
                                let _ = sock.send_to(&syn_ack, src);
                                peer.last_send = now;
                                peers.insert(hdr.conn_id, peer);
                                let ch = UdpChannel {
                                    _mux: mux_arc,
                                    remote: src,
                                    out_tx,
                                    in_rx: Arc::new(Mutex::new(in_rx)),
                                    stats,
                                    idle_timeout,
                                };
                                let _ = accept_tx.send(ch);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // 3. Drive timers for active peers, punches, rendezvous registrations, and Netem queue.
        let now = Instant::now();
        sock.flush_due(now);
        for peer in peers.values_mut() {
            peer.tick(&mut sock, now);
        }
        peers.retain(|_, p| !p.closed);

        // Drive NAT hole punches.
        let mut i = 0;
        while i < punches.len() {
            let remove = if let Some(linger) = punches[i].linger_until {
                now >= linger
            } else if now >= punches[i].deadline {
                if let Some(tx) = punches[i].reply.take() {
                    let _ = tx.send(Err("udp hole punch timed out".into()));
                }
                true
            } else {
                if now.duration_since(punches[i].last_send) >= PUNCH_RETRY_INTERVAL {
                    let pkt = encode_punch_packet(false, punches[i].token, punches[i].local_role);
                    for &t in &punches[i].targets {
                        let _ = sock.send_to(&pkt, t);
                    }
                    punches[i].last_send = now;
                }
                false
            };
            if remove {
                punches.swap_remove(i);
            } else {
                i += 1;
            }
        }

        // Drive rendezvous UDP registrations & host keepalive.
        let mut j = 0;
        while j < rv_regs.len() {
            if now >= rv_regs[j].deadline {
                let reg = rv_regs.swap_remove(j);
                let _ = reg.reply.send(Err(format!(
                    "rendezvous UDP registration to {} timed out",
                    reg.rv_addr
                )));
            } else {
                if now.duration_since(rv_regs[j].last_send) >= RV_REG_RETRY_INTERVAL {
                    let pkt = encode_rv_reg(rv_regs[j].role, rv_regs[j].token, &rv_regs[j].code);
                    let _ = sock.send_to(&pkt, rv_regs[j].rv_addr);
                    rv_regs[j].last_send = now;
                }
                j += 1;
            }
        }
        if let Some(ka) = rv_keepalive.as_mut()
            && now.duration_since(ka.last_send) >= RV_KEEPALIVE_INTERVAL
        {
            let pkt = encode_rv_reg(ka.role, ka.token, &ka.code);
            let _ = sock.send_to(&pkt, ka.rv_addr);
            ka.last_send = now;
        }
    }
}

/// Generates a random non-zero 64-bit punch token.
pub fn next_punch_token() -> u64 {
    static SEQ: AtomicU64 = AtomicU64::new(1);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let s = SEQ.fetch_add(1, Ordering::Relaxed);
    let mut z = nanos ^ s.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((std::process::id() as u64) << 19);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) | 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_codecs_round_trip_and_reject_corrupt() {
        let h = ChanHeader {
            kind: PKT_DATA,
            conn_id: 0x1234_5678,
            seq: 42,
            ack: 40,
            ack_bits: 0x00FF_00AA_5500_0001,
        };
        let enc = encode_chan_packet(h, b"hello-udp");
        let (h2, p2) = decode_chan_header(&enc).unwrap();
        assert_eq!(h2, h);
        assert_eq!(p2, b"hello-udp");
        assert!(decode_chan_header(&enc[..HEADER_LEN - 1]).is_none());
        let mut bad = enc.clone();
        bad[0] = b'X';
        assert!(decode_chan_header(&bad).is_none());

        let p = encode_punch_packet(false, 0xCAFE_BABE, ROLE_CLIENT);
        assert_eq!(decode_punch_packet(&p), Some((false, 0xCAFE_BABE, ROLE_CLIENT)));
        let pa = encode_punch_packet(true, 0xCAFE_BABE, ROLE_HOST);
        assert_eq!(decode_punch_packet(&pa), Some((true, 0xCAFE_BABE, ROLE_HOST)));
        assert!(decode_punch_packet(&p[..12]).is_none());

        let reg = encode_rv_reg(ROLE_HOST, 98765, "ABCD-2345");
        assert_eq!(
            decode_rv_reg(&reg),
            Some((ROLE_HOST, 98765, "ABCD-2345".to_string()))
        );
        assert!(decode_rv_reg(&reg[..10]).is_none());

        let sa: SocketAddr = "203.0.113.7:41234".parse().unwrap();
        let ack = encode_rv_ack(98765, sa);
        assert_eq!(decode_rv_ack(&ack), Some((98765, sa)));
    }

    #[test]
    fn netem_config_parse_valid_and_invalid() {
        let c = NetemConfig::parse("loss=5,dup=2.5,reorder=3,delay=40,jitter=10,seed=99").unwrap();
        assert_eq!(c.loss_pct, 5.0);
        assert_eq!(c.dup_pct, 2.5);
        assert_eq!(c.reorder_window, 3);
        assert_eq!(c.delay_ms, 40);
        assert_eq!(c.jitter_ms, 10);
        assert_eq!(c.seed, 99);
        assert!(c.is_active());
        assert!(!NetemConfig::default().is_active());
        assert!(NetemConfig::parse("loss=100").is_err());
        assert!(NetemConfig::parse("bogus=1").is_err());
        assert!(NetemConfig::parse("noequals").is_err());
    }

    #[test]
    fn karn_jacobson_rtt_converges_and_clamps_rto() {
        let mut rtt = RttEstimator::new();
        assert!(rtt.srtt_ms().is_none());
        assert_eq!(rtt.rto_ms(), INITIAL_RTO_MS);
        rtt.observe(100.0);
        assert_eq!(rtt.srtt_ms(), Some(100.0));
        assert_eq!(rtt.rttvar_ms(), 50.0);
        assert_eq!(rtt.rto_ms(), 300.0);
        for _ in 0..40 {
            rtt.observe(100.0);
        }
        assert!((rtt.srtt_ms().unwrap() - 100.0).abs() < 1.0);
        assert!(rtt.rto_ms() >= 110.0 && rtt.rto_ms() < 120.0);
        let mut fast = RttEstimator::new();
        for _ in 0..20 {
            fast.observe(1.0);
        }
        assert!(fast.rto_ms() >= MIN_RTO_MS);
    }

    #[test]
    fn udp_reliable_ordered_fragmentation_survives_loss_dup_reorder() {
        let host_netem = NetemConfig {
            loss_pct: 10.0,
            dup_pct: 5.0,
            reorder_window: 4,
            delay_ms: 0,
            jitter_ms: 0,
            seed: 1001,
        };
        let client_netem = NetemConfig {
            loss_pct: 10.0,
            dup_pct: 5.0,
            reorder_window: 4,
            delay_ms: 0,
            jitter_ms: 0,
            seed: 2002,
        };
        let host_mux = UdpMux::bind("127.0.0.1:0", host_netem).unwrap();
        host_mux.set_accepting(true);
        let addr = host_mux.local_addr();

        // Large 96 KiB frame (84 fragments) + 60 small frames both directions.
        let big_frame: Vec<u8> = (0..96 * 1024).map(|i| ((i * 31 + 7) & 0xFF) as u8).collect();
        let big_expected = big_frame.clone();

        let h_mux = host_mux.clone();
        let server = std::thread::spawn(move || {
            let mut ch = h_mux.accept(Some(Duration::from_secs(5))).unwrap();
            let got_big = ch.read_frame().unwrap();
            assert_eq!(got_big.len(), big_expected.len());
            assert_eq!(got_big, big_expected);
            ch.write_frame(&got_big).unwrap();
            for i in 0..60u32 {
                let got = ch.read_frame().unwrap();
                assert_eq!(got, format!("msg-{i}").as_bytes());
                ch.write_frame(format!("ack-{i}").as_bytes()).unwrap();
            }
            (ch.stats(), ch.netem_stats())
        });

        let mut client = UdpChannel::connect(&addr.to_string(), client_netem, Duration::from_secs(5)).unwrap();
        client.write_frame(&big_frame).unwrap();
        let echoed = client.read_frame().unwrap();
        assert_eq!(echoed, big_frame);
        for i in 0..60u32 {
            client.write_frame(format!("msg-{i}").as_bytes()).unwrap();
            let reply = client.read_frame().unwrap();
            assert_eq!(reply, format!("ack-{i}").as_bytes());
        }
        let c_stats = client.stats();
        let c_netem = client.netem_stats();
        let (h_stats, h_netem) = server.join().unwrap();

        assert!(c_netem.dropped + h_netem.dropped > 0, "expected simulated drops");
        assert!(c_netem.duplicated + h_netem.duplicated > 0, "expected simulated duplicates");
        assert!(c_netem.reordered + h_netem.reordered > 0, "expected simulated reorders");
        assert!(c_stats.retransmits + h_stats.retransmits > 0, "expected RTO/fast retransmits");
        assert!(c_stats.has_rtt && h_stats.has_rtt, "expected RTT samples");
    }

    #[test]
    fn simultaneous_udp_hole_punch_and_channel_exchange() {
        let a = UdpMux::bind("127.0.0.1:0", NetemConfig::default()).unwrap();
        let b = UdpMux::bind("127.0.0.1:0", NetemConfig::default()).unwrap();
        a.set_accepting(true);
        let addr_a = a.local_addr();
        let addr_b = b.local_addr();
        let token = next_punch_token();

        let rx_a = a.start_punch(token, ROLE_HOST, vec![addr_b], Duration::from_secs(2));
        let punched_host = b
            .punch(token, ROLE_CLIENT, vec![addr_a], Duration::from_secs(2))
            .unwrap();
        let punched_client = rx_a.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
        assert_eq!(punched_host, addr_a);
        assert_eq!(punched_client, addr_b);

        let a_clone = a.clone();
        let accept_t = std::thread::spawn(move || {
            let mut ch = a_clone.accept(Some(Duration::from_secs(2))).unwrap();
            assert_eq!(ch.read_frame().unwrap(), b"punched-hello");
            ch.write_frame(b"punched-world").unwrap();
        });
        let mut ch_b = b.connect(punched_host, Duration::from_secs(2)).unwrap();
        ch_b.write_frame(b"punched-hello").unwrap();
        assert_eq!(ch_b.read_frame().unwrap(), b"punched-world");
        accept_t.join().unwrap();
    }
}
