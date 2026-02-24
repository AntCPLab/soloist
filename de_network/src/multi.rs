use lazy_static::lazy_static;
use log::debug;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Mutex;

use ark_std::{end_timer, start_timer};

use super::{DeNet, Stats};

lazy_static! {
    static ref CONNECTIONS: Mutex<Connections> = Mutex::new(Connections::default());
}

/// Macro for locmaster the FieldChannel singleton in the current scope.
macro_rules! get_ch {
    () => {
        CONNECTIONS.lock().expect("Poisoned FieldChannel")
    };
}

#[derive(Debug)]
#[allow(dead_code)]
struct Peer {
    id: usize,
    addr: SocketAddr,
    stream: Option<TcpStream>,
}

#[derive(Default, Debug)]
struct Connections {
    id: usize,
    peers: Vec<Peer>,
    stats: Stats,
}

impl std::default::Default for Peer {
    fn default() -> Self {
        Self {
            id: 0,
            addr: "127.0.0.1:8000".parse().unwrap(),
            stream: None,
        }
    }
}

impl Connections {
    /// Given a path and the `id` of oneself, initialize the structure
    fn init_from_path(&mut self, path: &str, id: usize) {
        let f = BufReader::new(File::open(path).expect("host configuration path"));
        let mut peer_id = 0;
        for line in f.lines() {
            let line = line.unwrap();
            let trimmed = line.trim();
            if trimmed.len() > 0 {
                let addr: SocketAddr = trimmed
                    .parse()
                    .unwrap_or_else(|e| panic!("bad socket address: {}:\n{}", trimmed, e));
                let peer = Peer {
                    id: peer_id,
                    addr,
                    stream: None,
                };
                self.peers.push(peer);
                peer_id += 1;
            }
        }
        assert!(id < self.peers.len());
        self.id = id;
    }
    fn connect_to_all(&mut self) {
        let timer = start_timer!(|| "Connecting");
        let n = self.peers.len();
        for from_id in 0..n {
            for to_id in (from_id + 1)..n {
                debug!("{} to {}", from_id, to_id);
                if self.id == from_id {
                    let to_addr = self.peers[to_id].addr;
                    debug!("Contacting {}", to_id);
                    let stream = loop {
                        let mut ms_waited = 0;
                        match TcpStream::connect(to_addr) {
                            Ok(s) => break s,
                            Err(e) => match e.kind() {
                                std::io::ErrorKind::ConnectionRefused
                                | std::io::ErrorKind::ConnectionReset => {
                                    ms_waited += 10;
                                    std::thread::sleep(std::time::Duration::from_millis(10));
                                    if ms_waited % 3_000 == 0 {
                                        debug!("Still waiting");
                                    } else if ms_waited > 30_000 {
                                        panic!("Could not find peer in 30s");
                                    }
                                }
                                _ => {
                                    panic!("Error during FieldChannel::new: {}", e);
                                }
                            },
                        }
                    };
                    stream.set_nodelay(true).unwrap();
                    self.peers[to_id].stream = Some(stream);
                } else if self.id == to_id {
                    debug!("Awaiting {}", from_id);
                    let listener = TcpListener::bind(self.peers[self.id].addr).unwrap();
                    let (stream, _addr) = listener.accept().unwrap();
                    stream.set_nodelay(true).unwrap();
                    self.peers[from_id].stream = Some(stream);
                }
            }
            // Sender for next round waits for note from this sender to prevent race on receipt.
            if from_id + 1 < n {
                if self.id == from_id {
                    self.peers[self.id + 1]
                        .stream
                        .as_mut()
                        .unwrap()
                        .write_all(&[0u8])
                        .unwrap();
                } else if self.id == from_id + 1 {
                    self.peers[self.id - 1]
                        .stream
                        .as_mut()
                        .unwrap()
                        .read_exact(&mut [0u8])
                        .unwrap();
                }
            }
        }
        // Do a round with the master, to be sure everyone is ready
        let from_all = self.send_to_master(&[self.id as u8]);
        self.recv_from_master(from_all);
        for id in 0..n {
            if id != self.id {
                assert!(self.peers[id].stream.is_some());
            }
        }
        println!("deNetwork ready!");
        end_timer!(timer);
    }
    fn am_master(&self) -> bool {
        self.id == 0
    }
    fn broadcast(&mut self, bytes_out: &[u8]) -> Vec<Vec<u8>> {
        let timer = start_timer!(|| format!("Broadcast {}", bytes_out.len()));
        let m = bytes_out.len();
        let own_id = self.id;
        self.stats.bytes_sent += (self.peers.len() - 1) * m;
        self.stats.bytes_recv += (self.peers.len() - 1) * m;
        self.stats.broadcasts += 1;
        let r = self
            .peers
            .par_iter_mut()
            .enumerate()
            .map(|(id, peer)| {
                let mut bytes_in = vec![0u8; m];
                if id < own_id {
                    let stream = peer.stream.as_mut().unwrap();
                    stream.read_exact(&mut bytes_in[..]).unwrap();
                    stream.write_all(bytes_out).unwrap();
                } else if id == own_id {
                    bytes_in.copy_from_slice(bytes_out);
                } else {
                    let stream = peer.stream.as_mut().unwrap();
                    stream.write_all(bytes_out).unwrap();
                    stream.read_exact(&mut bytes_in[..]).unwrap();
                };
                bytes_in
            })
            .collect();
        end_timer!(timer);
        r
    }
    fn send_to_master(&mut self, bytes_out: &[u8]) -> Option<Vec<Vec<u8>>> {
        let timer = start_timer!(|| format!("To master {}", bytes_out.len()));
        let m = bytes_out.len();
        // sub-party id
        let own_id = self.id;
        self.stats.to_master += 1;
        // The party is the master one
        let r = if self.am_master() {
            let party_data = // Iterate over the sub-parties
                self.peers
                    .par_iter_mut()
                    .enumerate()
                    .map(|(id, peer)| {
                        // Copy own data into bytes_in vector
                        if id == own_id {
                            let mut bytes_in = vec![0u8; m];
                            bytes_in.copy_from_slice(bytes_out);
                            bytes_in
                        } 
                        // Read from stream the current sub-party's data
                        else {
                            let stream = peer.stream.as_mut().unwrap();
                            let mut bytes_size = [0u8; 8];
                            stream.read_exact(&mut bytes_size).unwrap();
                            let m = u64::from_le_bytes(bytes_size) as usize;
                            let mut bytes_in = vec![0u8; m];
                            stream.read_exact(&mut bytes_in[..]).unwrap();
                            bytes_in
                        }
                    })
                    .collect::<Vec<_>>();
            self.stats.bytes_recv += party_data[1..]
                .iter()
                .map(|data| 8 + data.len())
                .sum::<usize>();

            Some(
                party_data
            )
        } 
        // The party is a sub-party
        else {
            // Just write its data to stream
            self.stats.bytes_sent += m + 8;
            let bytes_size = (m as u64).to_le_bytes();

            let stream = self.peers[0]
                .stream
                .as_mut()
                .unwrap();
            stream.write_all(&bytes_size).unwrap();
            stream.write_all(bytes_out).unwrap();
            None
        };
        end_timer!(timer);
        // Result: the master party gets all the sub-parties' data
        r
    }

    fn recv_from_master(&mut self, bytes_out: Option<Vec<Vec<u8>>>) -> Vec<u8> {
        let own_id = self.id;
        self.stats.from_master += 1;
        // The party is the master one
        if self.am_master() {
            let bytes_out = bytes_out.unwrap();
            let timer = start_timer!(|| format!("From master"));
            // Iterate over the sub-parties
            self.stats.bytes_sent += self.peers
                .par_iter_mut()
                .enumerate()
                .filter(|p| p.0 != own_id)
                .map(|(id, peer)| {
                    // Write each sub-party's data to its own stream
                    let stream = peer.stream.as_mut().unwrap();
                    let bytes_size = (bytes_out[id].len() as u64).to_le_bytes();
                    stream.write_all(&bytes_size).unwrap();
                    stream.write_all(&bytes_out[id]).unwrap();
                    bytes_out[id].len() + 8
                })
                .sum::<usize>();
            end_timer!(timer);
            // The master gets its own data
            bytes_out[own_id].clone()
        } 
        // The party is a sub-party
        else {
            // Just read from stream
            let stream = self.peers[0].stream.as_mut().unwrap();
            let mut bytes_size = [0u8; 8];
            stream.read_exact(&mut bytes_size).unwrap();
            let m = u64::from_le_bytes(bytes_size) as usize;
            self.stats.bytes_recv += m;
            let mut bytes_in = vec![0u8; m];
            stream.read_exact(&mut bytes_in).unwrap();
            bytes_in
        }
        // Result: all sub-parties gets its own data from the master
    }
    fn uninit(&mut self) {
        for p in &mut self.peers {
            p.stream = None;
        }
    }
}

pub struct DeMultiNet;

impl DeNet for DeMultiNet {
    #[inline]
    fn party_id() -> usize {
        get_ch!().id
    }

    #[inline]
    fn n_parties() -> usize {
        get_ch!().peers.len()
    }

    #[inline]
    fn init_from_file(path: &str, party_id: usize) {
        let mut ch = get_ch!();
        ch.init_from_path(path, party_id);
        ch.connect_to_all();
    }

    #[inline]
    fn is_init() -> bool {
        get_ch!()
            .peers
            .first()
            .map(|p| p.stream.is_some())
            .unwrap_or(false)
    }

    #[inline]
    fn deinit() {
        get_ch!().uninit()
    }

    #[inline]
    fn reset_stats() {
        get_ch!().stats = Stats::default();
    }

    #[inline]
    fn stats() -> crate::Stats {
        get_ch!().stats.clone()
    }

    #[inline]
    fn broadcast_bytes(bytes: &[u8]) -> Vec<Vec<u8>> {
        get_ch!().broadcast(bytes)
    }

    #[inline]
    fn send_bytes_to_master(bytes: &[u8]) -> Option<Vec<Vec<u8>>> {
        get_ch!().send_to_master(bytes)
    }

    #[inline]
    fn recv_bytes_from_master(bytes: Option<Vec<Vec<u8>>>) -> Vec<u8> {
        get_ch!().recv_from_master(bytes)
    }
}
