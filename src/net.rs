//! Phase 9 deterministic 2-player TCP lockstep synchronization.
//!
//! Pure `std::net` length-prefixed framing exchanging per-turn [`Command`] batches
//! alongside `Skirmish::state_hash()` so any divergence triggers immediate desync
//! detection. Both peers execute ticks deterministically every [`TURN_TICKS`].

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use crate::replay::{decode_command, encode_command, read_i64, read_str, read_u32, read_u64, read_u8, write_str};
use crate::skirmish::{ArmyId, ArmySetup, Command, Skirmish};
use crate::skirmish_ai::ScriptedAi;

pub const TURN_TICKS: u64 = 3;
const MAX_FRAME_BYTES: usize = 1024 * 1024;
const MSG_HANDSHAKE: u8 = 1;
const MSG_TURN: u8 = 2;

#[derive(Debug, Clone)]
pub struct Handshake {
    pub map_id: String,
    pub seed: u64,
    pub setups: Vec<ArmySetup>,
    pub turn_ticks: u64,
    pub assigned_army: ArmyId,
}

#[derive(Debug, Clone)]
pub struct TurnPacket {
    pub turn: u64,
    pub hash: u64,
    pub commands: Vec<Command>,
}

fn write_frame(stream: &mut TcpStream, payload: &[u8]) -> Result<(), String> {
    let len = u32::try_from(payload.len()).map_err(|_| "frame too large".to_string())?;
    stream
        .write_all(&len.to_le_bytes())
        .map_err(|e| format!("net write len: {e}"))?;
    stream
        .write_all(payload)
        .map_err(|e| format!("net write payload: {e}"))?;
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
    let mut out = Vec::with_capacity(128);
    out.push(MSG_HANDSHAKE);
    write_str(&mut out, &h.map_id);
    out.extend_from_slice(&h.seed.to_le_bytes());
    out.extend_from_slice(&h.turn_ticks.to_le_bytes());
    out.push(h.assigned_army);
    out.extend_from_slice(&(h.setups.len() as u32).to_le_bytes());
    for s in &h.setups {
        write_str(&mut out, &s.name);
        out.extend_from_slice(&(s.faction as u32).to_le_bytes());
        out.extend_from_slice(&s.start.0.to_le_bytes());
        out.extend_from_slice(&s.start.1.to_le_bytes());
    }
    out
}

fn decode_handshake(data: &[u8]) -> Result<Handshake, String> {
    let mut pos = 0;
    if read_u8(data, &mut pos)? != MSG_HANDSHAKE {
        return Err("expected handshake message".to_string());
    }
    let map_id = read_str(data, &mut pos)?;
    let seed = read_u64(data, &mut pos)?;
    let turn_ticks = read_u64(data, &mut pos)?;
    let assigned_army = read_u8(data, &mut pos)?;
    let n = read_u32(data, &mut pos)? as usize;
    let mut setups = Vec::with_capacity(n.min(8));
    for _ in 0..n {
        let name = read_str(data, &mut pos)?;
        let faction = read_u32(data, &mut pos)? as usize;
        let x = read_i64(data, &mut pos)?;
        let y = read_i64(data, &mut pos)?;
        setups.push(ArmySetup { name, faction, start: (x, y) });
    }
    Ok(Handshake {
        map_id,
        seed,
        setups,
        turn_ticks,
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
    Ok(TurnPacket { turn, hash, commands })
}

pub struct LockstepPeer {
    stream: TcpStream,
    pub local_army: ArmyId,
    pub remote_army: ArmyId,
    pub turn_ticks: u64,
    pub turn: u64,
}

impl LockstepPeer {
    /// Binds `addr`, waits for one client to connect, and sends the initial [`Handshake`].
    pub fn host_bind(
        addr: &str,
        map_id: &str,
        seed: u64,
        setups: &[ArmySetup],
    ) -> Result<(Self, std::net::SocketAddr), String> {
        let listener =
            TcpListener::bind(addr).map_err(|e| format!("failed to bind {addr}: {e}"))?;
        let local_addr = listener
            .local_addr()
            .map_err(|e| format!("local_addr failed: {e}"))?;
        let (mut stream, _) = listener
            .accept()
            .map_err(|e| format!("accept failed: {e}"))?;
        let _ = stream.set_nodelay(true);
        let _ = stream.set_read_timeout(Some(Duration::from_secs(15)));

        let hs = Handshake {
            map_id: map_id.to_string(),
            seed,
            setups: setups.to_vec(),
            turn_ticks: TURN_TICKS,
            assigned_army: 1,
        };
        write_frame(&mut stream, &encode_handshake(&hs))?;

        Ok((
            Self {
                stream,
                local_army: 0,
                remote_army: 1,
                turn_ticks: TURN_TICKS,
                turn: 0,
            },
            local_addr,
        ))
    }

    /// Connects to a hosting peer and receives the [`Handshake`].
    pub fn connect(addr: &str) -> Result<(Self, Handshake), String> {
        let mut stream =
            TcpStream::connect(addr).map_err(|e| format!("failed to connect {addr}: {e}"))?;
        let _ = stream.set_nodelay(true);
        let _ = stream.set_read_timeout(Some(Duration::from_secs(15)));
        let raw = read_frame(&mut stream)?;
        let hs = decode_handshake(&raw)?;
        let peer = Self {
            stream,
            local_army: hs.assigned_army,
            remote_army: 0,
            turn_ticks: hs.turn_ticks,
            turn: 0,
        };
        Ok((peer, hs))
    }

    /// Exchanges one turn's command batch & state hash with the peer, verifies zero
    /// desyncs, applies commands in canonical ascending `ArmyId` order, and steps
    /// `self.turn_ticks`.
    pub fn step_turn(
        &mut self,
        world: &mut Skirmish,
        local_cmds: Vec<Command>,
    ) -> Result<(), String> {
        let cur_hash = world.state_hash();
        let out_pkt = TurnPacket {
            turn: self.turn,
            hash: cur_hash,
            commands: local_cmds.clone(),
        };
        write_frame(&mut self.stream, &encode_turn(&out_pkt))?;

        let raw = read_frame(&mut self.stream)?;
        let in_pkt = decode_turn(&raw)?;
        if in_pkt.turn != self.turn {
            return Err(format!(
                "lockstep turn mismatch: local {}, remote {}",
                self.turn, in_pkt.turn
            ));
        }
        if in_pkt.hash != cur_hash {
            return Err(format!(
                "LOCKSTEP DESYNC at turn {} (tick {}): local={cur_hash:016x} remote={:016x}",
                self.turn,
                world.tick(),
                in_pkt.hash
            ));
        }

        let mut batches = [
            (self.local_army, local_cmds),
            (self.remote_army, in_pkt.commands),
        ];
        batches.sort_by_key(|(army, _)| *army);
        for (army, cmds) in batches {
            for c in cmds {
                world.issue(army, c);
            }
        }
        world.run(self.turn_ticks);
        self.turn += 1;
        Ok(())
    }
}

/// Drives a complete headless AI-controlled lockstep match over TCP for `ticks` ticks.
pub fn run_headless_lockstep(
    peer: &mut LockstepPeer,
    world: &mut Skirmish,
    ticks: u64,
) -> Result<u64, String> {
    let mut ai = ScriptedAi::new(peer.local_army);
    while world.tick() < ticks && world.result().is_none() {
        let cmds = ai.decide(world);
        peer.step_turn(world, cmds)?;
    }
    Ok(world.state_hash())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pathfinding::PathGrid;
    use crate::sim::SCALE;
    use crate::skirmish::tests::test_catalog;

    #[test]
    fn tcp_lockstep_loopback_matches_hashes_for_600_ticks() {
        let setups = vec![
            ArmySetup { name: "ARMY_1".into(), faction: 0, start: (50 * SCALE, 50 * SCALE) },
            ArmySetup { name: "ARMY_2".into(), faction: 1, start: (200 * SCALE, 200 * SCALE) },
        ];
        let grid = PathGrid::new(64, 64, 4 * SCALE);
        let mass = vec![(40 * SCALE, 40 * SCALE), (210 * SCALE, 210 * SCALE)];
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
        let addr = listener.local_addr().unwrap();

        let setups_srv = setups.clone();
        let grid_srv = grid.clone();
        let mass_srv = mass.clone();
        let srv = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = stream.set_nodelay(true);
            let hs = Handshake {
                map_id: "LOOPBACK".into(),
                seed: 777,
                setups: setups_srv.clone(),
                turn_ticks: TURN_TICKS,
                assigned_army: 1,
            };
            write_frame(&mut stream, &encode_handshake(&hs)).unwrap();
            let mut peer = LockstepPeer {
                stream,
                local_army: 0,
                remote_army: 1,
                turn_ticks: TURN_TICKS,
                turn: 0,
            };
            let mut world = Skirmish::new(
                test_catalog(),
                grid_srv,
                (256 * SCALE, 256 * SCALE),
                mass_srv,
                &setups_srv,
                777,
            )
            .unwrap();
            run_headless_lockstep(&mut peer, &mut world, 600).unwrap()
        });

        let (mut client_peer, hs) = LockstepPeer::connect(&addr.to_string()).expect("connect");
        let mut client_world = Skirmish::new(
            test_catalog(),
            grid,
            (256 * SCALE, 256 * SCALE),
            mass,
            &hs.setups,
            hs.seed,
        )
        .unwrap();
        let h_client = run_headless_lockstep(&mut client_peer, &mut client_world, 600).unwrap();
        let h_server = srv.join().unwrap();
        assert_eq!(h_client, h_server);
    }
}
