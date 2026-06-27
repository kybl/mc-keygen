use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use curve25519_dalek::constants::ED25519_BASEPOINT_TABLE;
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::EdwardsPoint;
use rand::rngs::OsRng;
use rand::RngCore;

use crate::types::{MeshCoreKeypair, SearchError, SearchResult, SearchStats};

const BATCH_SIZE: u64 = 1024;

/// Apply Ed25519 scalar clamp in place: zero low 3 bits of byte 0, zero
/// bit 7 of byte 31, set bit 6 of byte 31.
pub fn clamp_scalar(s: &mut [u8; 32]) {
    s[0] &= 248;
    s[31] &= 63;
    s[31] |= 64;
}

/// Add `delta` (a u64, treated as the low 8 bytes of a 256-bit value) to the
/// 32-byte little-endian scalar in place. Wraps mod 2^256.
pub fn advance_scalar(s: &mut [u8; 32], delta: u64) {
    let mut carry: u64 = delta;
    for byte in s.iter_mut() {
        let sum = (*byte as u64) + (carry & 0xFF);
        *byte = (sum & 0xFF) as u8;
        carry = (carry >> 8) + (sum >> 8);
        if carry == 0 {
            break;
        }
    }
}

/// Result from a single GPU batch dispatch.
pub struct GpuBatchResult {
    pub keys_checked: u64,
    pub keypair: Option<MeshCoreKeypair>,
}

/// Trait abstracting GPU vanity key search backends (Metal, CUDA, etc.).
/// Each implementor owns a single GPU device and is used from one thread.
pub trait GpuSearcher: Send {
    fn search_batch(
        &mut self,
        base_nonce: u64,
    ) -> Result<GpuBatchResult, Box<dyn std::error::Error + Send + Sync>>;

    /// Run one batch in counting mode: returns (keys_checked, matches_found)
    /// without extracting any keypair. Used by the bench harness so a single
    /// batch can validate observed-vs-expected match counts (Poisson check)
    /// without short-circuiting on first match.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    fn count_batch(
        &mut self,
    ) -> Result<(u64, u32), Box<dyn std::error::Error + Send + Sync>>;

    fn device_name(&self) -> &str;
}

/// Default CPU worker count for hybrid mode.
///
/// Reserves one SMT pair (or one core on non-SMT hardware) per GPU so the
/// GPU dispatch thread's host work — kernel launch, result copy — isn't
/// preempted by CPU workers. Without this, hybrid mode is slower than
/// pure-GPU because the dispatch thread loses cycles to oversubscription.
///
/// SMT-ness is detected from the logical:physical core ratio; on hybrid
/// CPUs (Intel P+E) `logical > physical` still implies SMT exists somewhere
/// and reserving 2 logical cores lands us on a P-core's full SMT pair.
pub fn default_hybrid_cpu_threads(num_gpus: usize) -> usize {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let physical = sysinfo::System::new().physical_core_count().unwrap_or(logical);
    let smt_factor = if physical > 0 && logical > physical { 2 } else { 1 };
    logical.saturating_sub(smt_factor * num_gpus).max(1)
}

/// Parsed prefix for fast nibble-level matching.
/// Avoids hex-encoding every public key in the hot loop.
pub struct PrefixMatcher {
    /// Full bytes to match (pairs of hex chars).
    pub(crate) full_bytes: Vec<u8>,
    /// If prefix has odd length, the high nibble of the trailing hex char.
    pub(crate) trailing_nibble: Option<u8>,
}

impl PrefixMatcher {
    /// Parse a hex prefix string into a matcher.
    /// Assumes input is already validated as uppercase hex.
    pub fn new(prefix: &str) -> Self {
        let prefix = prefix.to_ascii_uppercase();
        let mut full_bytes = Vec::new();
        let chars: Vec<u8> = prefix.bytes().collect();

        let mut i = 0;
        while i + 1 < chars.len() {
            let hi = nibble_from_ascii(chars[i]);
            let lo = nibble_from_ascii(chars[i + 1]);
            full_bytes.push((hi << 4) | lo);
            i += 2;
        }

        let trailing_nibble = if i < chars.len() {
            Some(nibble_from_ascii(chars[i]))
        } else {
            None
        };

        PrefixMatcher {
            full_bytes,
            trailing_nibble,
        }
    }

    /// Check if a 32-byte public key matches this prefix.
    #[inline(always)]
    pub fn matches(&self, public_key: &[u8; 32]) -> bool {
        // Check full bytes
        for (i, &expected) in self.full_bytes.iter().enumerate() {
            if public_key[i] != expected {
                return false;
            }
        }

        // Check trailing nibble (high nibble of next byte)
        if let Some(nibble) = self.trailing_nibble {
            let idx = self.full_bytes.len();
            if (public_key[idx] >> 4) != nibble {
                return false;
            }
        }

        true
    }

    /// `(mask, value)` for a first-byte prefilter: a public key can match this
    /// prefix only if `public_key[0] & mask == value`. Used to cheaply reject
    /// most candidates in SIMD before full encoding.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub fn first_byte_filter(&self) -> (u8, u8) {
        if let Some(&b) = self.full_bytes.first() {
            (0xFF, b)
        } else {
            // No full byte means the prefix is a single nibble.
            let n = self.trailing_nibble.expect("prefix has at least one nibble");
            (0xF0, n << 4)
        }
    }
}

fn nibble_from_ascii(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'A'..=b'F' => c - b'A' + 10,
        b'a'..=b'f' => c - b'a' + 10,
        _ => unreachable!("validated hex input"),
    }
}

/// Check if a public key should be skipped (starts with 00 or FF).
#[inline(always)]
fn should_skip(public_key: &[u8; 32]) -> bool {
    public_key[0] == 0x00 || public_key[0] == 0xFF
}

/// Internal result containing keypair + which prefix matched.
struct MatchResult {
    keypair: MeshCoreKeypair,
    matched_prefix: String,
}

/// Handle for a running vanity key search.
/// Exposes atomics so a TUI render loop can poll progress directly.
pub struct SearchHandle {
    found: Arc<AtomicBool>,
    attempts: Arc<AtomicU64>,
    result: Arc<Mutex<Option<MatchResult>>>,
    start: Instant,
    workers: Vec<JoinHandle<()>>,
}

impl SearchHandle {
    /// Start a vanity key search in background threads.
    pub fn start(prefixes: &[String], num_threads: usize) -> Self {
        let matchers: Arc<Vec<(String, PrefixMatcher)>> = Arc::new(
            prefixes
                .iter()
                .map(|p| (p.clone(), PrefixMatcher::new(p)))
                .collect(),
        );
        let found = Arc::new(AtomicBool::new(false));
        let attempts = Arc::new(AtomicU64::new(0));
        let result: Arc<Mutex<Option<MatchResult>>> = Arc::new(Mutex::new(None));

        let mut workers = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            workers.push(spawn_cpu_worker(
                Arc::clone(&matchers),
                Arc::clone(&found),
                Arc::clone(&attempts),
                Arc::clone(&result),
            ));
        }

        SearchHandle {
            found,
            attempts,
            result,
            start: Instant::now(),
            workers,
        }
    }

    /// Check if a match has been found.
    pub fn is_done(&self) -> bool {
        self.found.load(Ordering::Relaxed)
    }

    /// Get current search statistics.
    pub fn stats(&self, expected: u64) -> SearchStats {
        let attempts = self.attempts.load(Ordering::Relaxed);
        let elapsed = self.start.elapsed().as_secs_f64();
        SearchStats {
            attempts,
            expected_attempts: expected,
            elapsed_secs: elapsed,
            keys_per_sec: if elapsed > 0.0 {
                attempts as f64 / elapsed
            } else {
                0.0
            },
        }
    }

    /// Join all worker threads and return the result.
    /// Returns `Err` if all workers exited without finding a match (e.g., GPU errors).
    pub fn finish(self) -> Result<SearchResult, SearchError> {
        for h in self.workers {
            h.join().unwrap();
        }

        let elapsed = self.start.elapsed().as_secs_f64();
        let attempts = self.attempts.load(Ordering::Relaxed);

        match self.result.lock().unwrap().take() {
            Some(m) => Ok(SearchResult {
                public_key: hex::encode_upper(m.keypair.public_key),
                private_key: hex::encode_upper(m.keypair.private_key),
                matched_prefix: m.matched_prefix,
                attempts,
                elapsed_secs: elapsed,
            }),
            None => Err(SearchError {
                attempts,
                elapsed_secs: elapsed,
            }),
        }
    }

    /// Start a GPU-only vanity key search. Spawns one thread per GPU device.
    pub fn start_gpu(
        prefixes: &[String],
        gpu_searchers: Vec<Box<dyn GpuSearcher>>,
    ) -> Self {
        let matchers: Arc<Vec<(String, PrefixMatcher)>> = Arc::new(
            prefixes
                .iter()
                .map(|p| (p.clone(), PrefixMatcher::new(p)))
                .collect(),
        );
        let found = Arc::new(AtomicBool::new(false));
        let attempts = Arc::new(AtomicU64::new(0));
        let result: Arc<Mutex<Option<MatchResult>>> = Arc::new(Mutex::new(None));

        let mut workers = Vec::with_capacity(gpu_searchers.len());
        for searcher in gpu_searchers {
            workers.push(thread::spawn({
                let matchers = Arc::clone(&matchers);
                let found = Arc::clone(&found);
                let attempts = Arc::clone(&attempts);
                let result = Arc::clone(&result);
                move || gpu_dispatch_loop(searcher, matchers, found, attempts, result)
            }));
        }

        SearchHandle {
            found,
            attempts,
            result,
            start: Instant::now(),
            workers,
        }
    }

    /// Start a hybrid vanity key search: CPU threads + GPU devices concurrently.
    pub fn start_hybrid(
        prefixes: &[String],
        cpu_threads: usize,
        gpu_searchers: Vec<Box<dyn GpuSearcher>>,
    ) -> Self {
        let matchers: Arc<Vec<(String, PrefixMatcher)>> = Arc::new(
            prefixes
                .iter()
                .map(|p| (p.clone(), PrefixMatcher::new(p)))
                .collect(),
        );
        let found = Arc::new(AtomicBool::new(false));
        let attempts = Arc::new(AtomicU64::new(0));
        let result: Arc<Mutex<Option<MatchResult>>> = Arc::new(Mutex::new(None));

        let mut workers = Vec::with_capacity(cpu_threads + gpu_searchers.len());

        // Spawn CPU workers
        for _ in 0..cpu_threads {
            workers.push(spawn_cpu_worker(
                Arc::clone(&matchers),
                Arc::clone(&found),
                Arc::clone(&attempts),
                Arc::clone(&result),
            ));
        }

        // Spawn GPU dispatch threads
        for searcher in gpu_searchers {
            workers.push(thread::spawn({
                let matchers = Arc::clone(&matchers);
                let found = Arc::clone(&found);
                let attempts = Arc::clone(&attempts);
                let result = Arc::clone(&result);
                move || gpu_dispatch_loop(searcher, matchers, found, attempts, result)
            }));
        }

        SearchHandle {
            found,
            attempts,
            result,
            start: Instant::now(),
            workers,
        }
    }
}

/// Per worker: number of chained points compressed under one batched
/// inversion. The single field inversion in `compress_batch` (~265 field
/// muls via the pow22523 chain) is the dominant cost, so amortizing it over
/// more points is the biggest CPU lever: raising this from 16 to 256
/// measured ~1.57x throughput on a 4-core Xeon. Returns diminish past 256
/// (the per-point `+8B` add becomes the floor) while stack/cache footprint
/// keeps growing — the batch holds `CHAIN_BATCH` EdwardsPoints (~160 B each)
/// plus scratch — so 256 sits at the knee of the curve.
const CHAIN_BATCH: usize = 512;

/// Spawn one CPU worker thread that scans via a `+8B` chain with
/// Montgomery batched compression.
///
/// Each worker draws a single random clamped scalar, does ONE
/// `mul_base_clamped` to get its starting point, then in each iteration
/// chains `CHAIN_BATCH` points via cheap point-adds and compresses them
/// all with one batched inversion (`EdwardsPoint::compress_batch`). The
/// scalar accumulator tracks the base of each batch so a match at index
/// `i` emits `scalar + 8·i` as the private-key scalar.
///
/// This mirrors the GPU strategy: one heavy scalarmult per chain, then
/// `N` cheap chain steps amortizing one field inversion. Per-iter cost
/// goes from ~265 fe_muls (one full `compress`) to ~20 (batched).
fn spawn_cpu_worker(
    matchers: Arc<Vec<(String, PrefixMatcher)>>,
    found: Arc<AtomicBool>,
    attempts: Arc<AtomicU64>,
    result: Arc<Mutex<Option<MatchResult>>>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        // On x86-64 with AVX2, scan four independent keys per step in SIMD
        // lanes (radix-2^25.5 field, GPU-style). Falls back to the scalar
        // chain otherwise.
        #[cfg(target_arch = "x86_64")]
        {
            if CHAIN_BATCH % 8 == 0 && std::is_x86_feature_detected!("avx512f") {
                // SAFETY: guarded by the runtime avx512f check.
                unsafe { cpu_worker_simd512(&matchers, &found, &attempts, &result) };
                return;
            }
            if CHAIN_BATCH % 4 == 0 && std::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by the runtime avx2 check above.
                unsafe { cpu_worker_simd(&matchers, &found, &attempts, &result) };
                return;
            }
        }
        cpu_worker_scalar(&matchers, &found, &attempts, &result);
    })
}

/// Build the matched keypair (fresh random private-key prefix half), publish
/// it, and flip the `found` flag.
fn record_match(
    matched_prefix: String,
    public_key: [u8; 32],
    match_scalar: [u8; 32],
    found: &AtomicBool,
    result: &Mutex<Option<MatchResult>>,
) {
    let mut prefix_half = [0u8; 32];
    OsRng.fill_bytes(&mut prefix_half);
    let mut private_key = [0u8; 64];
    private_key[..32].copy_from_slice(&match_scalar);
    private_key[32..].copy_from_slice(&prefix_half);
    found.store(true, Ordering::Relaxed);
    *result.lock().unwrap() = Some(MatchResult {
        keypair: MeshCoreKeypair {
            public_key,
            private_key,
        },
        matched_prefix,
    });
}

/// Scalar `+8B` chained-compress worker (portable fallback).
fn cpu_worker_scalar(
    matchers: &[(String, PrefixMatcher)],
    found: &AtomicBool,
    attempts: &AtomicU64,
    result: &Mutex<Option<MatchResult>>,
) {
    let eight_b = ED25519_BASEPOINT_TABLE * &Scalar::from(8u64);

    let mut scalar = [0u8; 32];
    OsRng.fill_bytes(&mut scalar);
    clamp_scalar(&mut scalar);
    let mut point: EdwardsPoint = EdwardsPoint::mul_base_clamped(scalar);

    let mut local_count: u64 = 0;

    while !found.load(Ordering::Relaxed) {
        let (compressed, next_point) = point.chain_compress_y_only::<CHAIN_BATCH>(&eight_b);

        for (i, public_key) in compressed.iter().enumerate() {
            if should_skip(public_key) {
                continue;
            }
            if let Some(matched) = matchers.iter().find(|(_, m)| m.matches(public_key)) {
                let mut match_scalar = scalar;
                advance_scalar(&mut match_scalar, 8 * i as u64);
                // Sign bit was zeroed and full points weren't kept; recompute
                // point + i·8B and compress it properly for the hit.
                let mut hit_point = point;
                for _ in 0..i {
                    hit_point += eight_b;
                }
                let public_key = hit_point.compress().to_bytes();
                attempts.fetch_add(local_count + i as u64 + 1, Ordering::Relaxed);
                record_match(matched.0.clone(), public_key, match_scalar, found, result);
                return;
            }
        }

        local_count += CHAIN_BATCH as u64;
        if local_count >= BATCH_SIZE {
            attempts.fetch_add(local_count, Ordering::Relaxed);
            local_count = 0;
        }
        point = next_point;
        advance_scalar(&mut scalar, 8 * CHAIN_BATCH as u64);
    }

    attempts.fetch_add(local_count, Ordering::Relaxed);
}

/// AVX2 worker: four independent `+8B` chains advanced in SIMD lanes, scanning
/// `4·(CHAIN_BATCH/4)` keys per batch with one shared 4-wide batch inversion.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cpu_worker_simd(
    matchers: &[(String, PrefixMatcher)],
    found: &AtomicBool,
    attempts: &AtomicU64,
    result: &Mutex<Option<MatchResult>>,
) {
    use crate::simd4::{avx2, fe_frombytes};

    const K: usize = CHAIN_BATCH / 4;

    let eight_b = ED25519_BASEPOINT_TABLE * &Scalar::from(8u64);
    // Each batch advances every lane by K steps of 8B.
    let big_step = ED25519_BASEPOINT_TABLE * &Scalar::from((8 * K) as u64);
    let niels = avx2::niels4_from_bytes(&eight_b.niels_bytes());

    // Four independent random starting keys, one per lane.
    let mut scalars = [[0u8; 32]; 4];
    for l in 0..4 {
        OsRng.fill_bytes(&mut scalars[l]);
        clamp_scalar(&mut scalars[l]);
    }
    let mut starts: [EdwardsPoint; 4] =
        core::array::from_fn(|l| EdwardsPoint::mul_base_clamped(scalars[l]));
    let xyzt: [_; 4] = core::array::from_fn(|l| starts[l].xyzt_bytes());
    let mut p4 = avx2::point4_from_xyzt(
        &core::array::from_fn(|l| fe_frombytes(&xyzt[l][0])),
        &core::array::from_fn(|l| fe_frombytes(&xyzt[l][1])),
        &core::array::from_fn(|l| fe_frombytes(&xyzt[l][2])),
        &core::array::from_fn(|l| fe_frombytes(&xyzt[l][3])),
    );

    let mut out: Box<[[[u8; 32]; 4]; K]> = Box::new([[[0u8; 32]; 4]; K]);
    let mut local_count: u64 = 0;

    while !found.load(Ordering::Relaxed) {
        p4 = avx2::chain_y_only::<K>(p4, &niels, &mut out);

        for s in 0..K {
            for lane in 0..4 {
                let public_key = &out[s][lane];
                if should_skip(public_key) {
                    continue;
                }
                if let Some(matched) = matchers.iter().find(|(_, m)| m.matches(public_key)) {
                    let mut match_scalar = scalars[lane];
                    advance_scalar(&mut match_scalar, 8 * s as u64);
                    // Recompute the true signed pubkey for this lane/step from
                    // the dalek start point kept in lockstep.
                    let mut hit = starts[lane];
                    for _ in 0..s {
                        hit += eight_b;
                    }
                    let public_key = hit.compress().to_bytes();
                    attempts.fetch_add(
                        local_count + (s * 4 + lane) as u64 + 1,
                        Ordering::Relaxed,
                    );
                    record_match(matched.0.clone(), public_key, match_scalar, found, result);
                    return;
                }
            }
        }

        local_count += CHAIN_BATCH as u64;
        if local_count >= BATCH_SIZE {
            attempts.fetch_add(local_count, Ordering::Relaxed);
            local_count = 0;
        }
        // Advance scalars and dalek start points in lockstep with p4.
        for l in 0..4 {
            advance_scalar(&mut scalars[l], 8 * K as u64);
            starts[l] += big_step;
        }
    }

    attempts.fetch_add(local_count, Ordering::Relaxed);
}

/// AVX-512 worker: eight independent `+8B` chains in 8 SIMD lanes. Same shape
/// as [`cpu_worker_simd`], doubled lane count.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn cpu_worker_simd512(
    matchers: &[(String, PrefixMatcher)],
    found: &AtomicBool,
    attempts: &AtomicU64,
    result: &Mutex<Option<MatchResult>>,
) {
    use crate::simd4::{avx512, fe_frombytes};

    const K: usize = CHAIN_BATCH / 8;

    let eight_b = ED25519_BASEPOINT_TABLE * &Scalar::from(8u64);
    let big_step = ED25519_BASEPOINT_TABLE * &Scalar::from((8 * K) as u64);
    let niels = avx512::niels8_from_bytes(&eight_b.niels_bytes());

    let mut scalars = [[0u8; 32]; 8];
    for l in 0..8 {
        OsRng.fill_bytes(&mut scalars[l]);
        clamp_scalar(&mut scalars[l]);
    }
    let mut starts: [EdwardsPoint; 8] =
        core::array::from_fn(|l| EdwardsPoint::mul_base_clamped(scalars[l]));
    let xyzt: [_; 8] = core::array::from_fn(|l| starts[l].xyzt_bytes());
    let mut p8 = avx512::point8_from_xyzt(
        &core::array::from_fn(|l| fe_frombytes(&xyzt[l][0])),
        &core::array::from_fn(|l| fe_frombytes(&xyzt[l][1])),
        &core::array::from_fn(|l| fe_frombytes(&xyzt[l][2])),
        &core::array::from_fn(|l| fe_frombytes(&xyzt[l][3])),
    );

    let filter: Vec<(u8, u8)> = matchers.iter().map(|(_, m)| m.first_byte_filter()).collect();

    let mut out: Box<[[[u8; 32]; 8]; K]> = Box::new([[[0u8; 32]; 8]; K]);
    let mut local_count: u64 = 0;

    while !found.load(Ordering::Relaxed) {
        p8 = avx512::chain_y_only::<K>(p8, &niels, &filter, &mut out);

        for s in 0..K {
            for lane in 0..8 {
                let public_key = &out[s][lane];
                if should_skip(public_key) {
                    continue;
                }
                if let Some(matched) = matchers.iter().find(|(_, m)| m.matches(public_key)) {
                    let mut match_scalar = scalars[lane];
                    advance_scalar(&mut match_scalar, 8 * s as u64);
                    let mut hit = starts[lane];
                    for _ in 0..s {
                        hit += eight_b;
                    }
                    let public_key = hit.compress().to_bytes();
                    attempts.fetch_add(
                        local_count + (s * 8 + lane) as u64 + 1,
                        Ordering::Relaxed,
                    );
                    record_match(matched.0.clone(), public_key, match_scalar, found, result);
                    return;
                }
            }
        }

        local_count += CHAIN_BATCH as u64;
        if local_count >= BATCH_SIZE {
            attempts.fetch_add(local_count, Ordering::Relaxed);
            local_count = 0;
        }
        for l in 0..8 {
            advance_scalar(&mut scalars[l], 8 * K as u64);
            starts[l] += big_step;
        }
    }

    attempts.fetch_add(local_count, Ordering::Relaxed);
}

/// GPU dispatch loop shared by start_gpu and start_hybrid.
/// Runs search_batch in a loop until a match is found or another thread signals done.
fn gpu_dispatch_loop(
    mut searcher: Box<dyn GpuSearcher>,
    matchers: Arc<Vec<(String, PrefixMatcher)>>,
    found: Arc<AtomicBool>,
    attempts: Arc<AtomicU64>,
    result: Arc<Mutex<Option<MatchResult>>>,
) {
    let mut nonce_bytes = [0u8; 8];
    OsRng.fill_bytes(&mut nonce_bytes);
    let mut base_nonce: u64 = u64::from_le_bytes(nonce_bytes);

    while !found.load(Ordering::Relaxed) {
        match searcher.search_batch(base_nonce) {
            Ok(batch_result) => {
                attempts.fetch_add(batch_result.keys_checked, Ordering::Relaxed);
                if let Some(kp) = batch_result.keypair {
                    found.store(true, Ordering::Relaxed);
                    let matched_prefix = matchers
                        .iter()
                        .find(|(_, m)| m.matches(&kp.public_key))
                        .map(|(p, _)| p.clone())
                        .unwrap_or_else(|| matchers[0].0.clone());
                    *result.lock().unwrap() = Some(MatchResult {
                        keypair: kp,
                        matched_prefix,
                    });
                    return;
                }
                base_nonce = base_nonce.wrapping_add(batch_result.keys_checked);
            }
            Err(e) => {
                eprintln!("GPU error ({}): {}", searcher.device_name(), e);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_matcher_full_bytes() {
        let m = PrefixMatcher::new("AB");
        let mut key = [0u8; 32];
        key[0] = 0xAB;
        assert!(m.matches(&key));

        key[0] = 0xAC;
        assert!(!m.matches(&key));
    }

    #[test]
    fn prefix_matcher_odd_nibble() {
        let m = PrefixMatcher::new("A");
        let mut key = [0u8; 32];
        key[0] = 0xA0;
        assert!(m.matches(&key));

        key[0] = 0xAF;
        assert!(m.matches(&key));

        key[0] = 0xB0;
        assert!(!m.matches(&key));
    }

    #[test]
    fn prefix_matcher_multi_byte() {
        let m = PrefixMatcher::new("ABCD");
        let mut key = [0u8; 32];
        key[0] = 0xAB;
        key[1] = 0xCD;
        assert!(m.matches(&key));

        key[1] = 0xCE;
        assert!(!m.matches(&key));
    }

    #[test]
    fn prefix_matcher_three_nibbles() {
        let m = PrefixMatcher::new("ABC");
        let mut key = [0u8; 32];
        key[0] = 0xAB;
        key[1] = 0xC0;
        assert!(m.matches(&key));

        key[1] = 0xCF;
        assert!(m.matches(&key));

        key[1] = 0xD0;
        assert!(!m.matches(&key));
    }

    #[test]
    fn skip_00_prefix() {
        let mut key = [0u8; 32];
        key[0] = 0x00;
        assert!(should_skip(&key));
    }

    #[test]
    fn skip_ff_prefix() {
        let mut key = [0u8; 32];
        key[0] = 0xFF;
        assert!(should_skip(&key));
    }

    #[test]
    fn no_skip_normal_prefix() {
        let mut key = [0u8; 32];
        key[0] = 0xAB;
        assert!(!should_skip(&key));
    }

    #[test]
    fn prefix_matcher_case_insensitive() {
        let m = PrefixMatcher::new("ab");
        let mut key = [0u8; 32];
        key[0] = 0xAB;
        assert!(m.matches(&key));
    }

    #[test]
    fn search_handle_finds_single_char_prefix() {
        let handle = SearchHandle::start(&["A".to_string()], 2);
        let result = handle.finish().expect("search should find a match");
        assert!(
            result.public_key.starts_with('A'),
            "expected public key starting with A, got {}",
            result.public_key
        );
        assert_eq!(result.matched_prefix, "A");
    }

    #[test]
    fn search_handle_multiple_prefixes() {
        let handle = SearchHandle::start(&["A".to_string(), "B".to_string()], 2);
        let result = handle.finish().expect("search should find a match");
        assert!(
            result.public_key.starts_with('A') || result.public_key.starts_with('B'),
            "expected public key starting with A or B, got {}",
            result.public_key
        );
        assert!(
            result.matched_prefix == "A" || result.matched_prefix == "B",
            "expected matched_prefix A or B, got {}",
            result.matched_prefix
        );
    }

    /// The `+8B` chain advances point and scalar in lockstep; if they drift
    /// the prefix check still passes (pubkey would just have wrong prefix
    /// origin) but the returned scalar wouldn't reproduce the pubkey. Verify
    /// `scalar·B == returned_pubkey` via curve25519-dalek.
    #[test]
    fn search_handle_scalar_matches_pubkey() {
        let handle = SearchHandle::start(&["A".to_string()], 2);
        let result = handle.finish().expect("search should find a match");

        let priv_bytes = hex::decode(&result.private_key).unwrap();
        let mut scalar = [0u8; 32];
        scalar.copy_from_slice(&priv_bytes[..32]);

        let derived = EdwardsPoint::mul_base_clamped(scalar).compress().to_bytes();
        let expected = hex::decode(&result.public_key).unwrap();
        assert_eq!(
            derived[..],
            expected[..],
            "scalar·B != returned pubkey: scalar={} pubkey={}",
            hex::encode_upper(scalar),
            result.public_key
        );
    }

    /// The hot loop scans `compress_batch_y_only` (sign bit zeroed) but the
    /// prefix matcher only reads the low bytes, and the real pubkey is
    /// recompressed on a hit. Guard the fork invariant: y-only must equal the
    /// canonical compression on every byte except the sign bit of byte 31.
    #[test]
    fn compress_batch_y_only_matches_canonical() {
        let pts: [EdwardsPoint; 4] =
            core::array::from_fn(|i| EdwardsPoint::mul_base_clamped({
                let mut s = [0u8; 32];
                s[0] = (i as u8) * 40 + 9;
                s[8] = 0x5a;
                clamp_scalar(&mut s);
                s
            }));
        let y_only = EdwardsPoint::compress_batch_y_only::<4>(&pts);
        for (i, p) in pts.iter().enumerate() {
            let full = p.compress().to_bytes();
            assert_eq!(&y_only[i][..31], &full[..31], "low bytes differ at {i}");
            assert_eq!(y_only[i][31] & 0x80, 0, "sign bit not cleared at {i}");
            assert_eq!(
                y_only[i][31] & 0x7f,
                full[31] & 0x7f,
                "non-sign bits of byte 31 differ at {i}"
            );
        }
    }
}
