use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
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

/// Parsed prefix for fast nibble-level matching. Used by the GPU backends to
/// pack prefix data; the CPU path matches via [`Target`].
#[allow(dead_code)]
pub struct PrefixMatcher {
    /// Full bytes to match (pairs of hex chars).
    pub(crate) full_bytes: Vec<u8>,
    /// If prefix has odd length, the high nibble of the trailing hex char.
    pub(crate) trailing_nibble: Option<u8>,
}

#[allow(dead_code)]
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

/// The high nibble of byte 31 carries the Ed25519 sign bit, which the y-only
/// fast path leaves zeroed. Matching that touches this nibble is therefore done
/// leniently on the y-only encoding and confirmed on the recomputed real key.
const SIGN_NIBBLE: usize = 62;

/// Hex nibble at position `pos` (0..64) of a public key.
#[inline(always)]
fn nibble_at(pk: &[u8; 32], pos: usize) -> u8 {
    let b = pk[pos >> 1];
    if pos & 1 == 0 {
        b >> 4
    } else {
        b & 0x0F
    }
}

/// Where in the 64-hex-char key the pattern must appear.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Location {
    Prefix,
    Anywhere,
    Suffix,
}

/// What to look for.
enum What {
    /// Exact hex pattern(s), each stored as `(label, nibble values)`.
    Exact(Vec<(String, Vec<u8>)>),
    /// A run of at least N identical hex characters.
    Run(u32),
}

/// A search target: a location (where) combined with a criterion (what).
pub struct Target {
    location: Location,
    what: What,
    /// AVX2 detected at construction; selects the vectorized per-key matchers.
    simd: bool,
}

/// Runtime AVX2 detection for the per-key matchers, done once per Target.
fn detect_simd() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Interleave the bits of `x` into the even bit positions of a u64
/// (bit i of `x` moves to bit 2i). Classic Morton spread.
#[inline]
fn spread(x: u32) -> u64 {
    let mut v = x as u64;
    v = (v | (v << 16)) & 0x0000_FFFF_0000_FFFF;
    v = (v | (v << 8)) & 0x00FF_00FF_00FF_00FF;
    v = (v | (v << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    v = (v | (v << 2)) & 0x3333_3333_3333_3333;
    v = (v | (v << 1)) & 0x5555_5555_5555_5555;
    v
}

/// Nibble-adjacency mask of a key: bit `j` (j in 0..63) = "hex char j equals
/// hex char j+1". A run of `n` identical hex chars is exactly a run of `n-1`
/// consecutive set bits. Portable fallback for [`adj_mask`].
fn adj_mask_scalar(pk: &[u8; 32]) -> u64 {
    let mut a: u32 = 0; // bit i = hi nibble of byte i == lo nibble of byte i
    let mut b: u32 = 0; // bit i = lo nibble of byte i == hi nibble of byte i+1
    for i in 0..32 {
        a |= (((((pk[i] >> 4) ^ pk[i]) & 0xF) == 0) as u32) << i;
    }
    for i in 0..31 {
        b |= ((((pk[i] ^ (pk[i + 1] >> 4)) & 0xF) == 0) as u32) << i;
    }
    // String position 2i is byte i's hi nibble: within-byte adjacency lands on
    // even bits, cross-byte on odd bits.
    spread(a) | (spread(b) << 1)
}

/// AVX2 [`adj_mask_scalar`]: two 32-byte compares + movemask.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn adj_mask_avx2(pk: &[u8; 32]) -> u64 {
    use core::arch::x86_64::*;
    let v = _mm256_loadu_si256(pk.as_ptr() as *const __m256i);
    let nib = _mm256_set1_epi8(0x0F);
    let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), nib);
    let lo = _mm256_and_si256(v, nib);
    let a = _mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, lo)) as u32;
    // hi shifted down one byte across the full 256-bit register, so lane i
    // holds byte i+1's hi nibble.
    let t = _mm256_permute2x128_si256(hi, hi, 0x81);
    let hi_next = _mm256_alignr_epi8(t, hi, 1);
    let b = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, hi_next)) as u32) & 0x7FFF_FFFF;
    spread(a) | (spread(b) << 1)
}

/// Position mask of a key: bit `j` = "hex char j equals `val`".
/// Portable fallback for [`nib_match_mask`].
fn nib_match_mask_scalar(pk: &[u8; 32], val: u8) -> u64 {
    let mut a: u32 = 0;
    let mut b: u32 = 0;
    for i in 0..32 {
        a |= (((pk[i] >> 4) == val) as u32) << i;
        b |= (((pk[i] & 0xF) == val) as u32) << i;
    }
    spread(a) | (spread(b) << 1)
}

/// AVX2 [`nib_match_mask_scalar`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nib_match_mask_avx2(pk: &[u8; 32], val: u8) -> u64 {
    use core::arch::x86_64::*;
    let v = _mm256_loadu_si256(pk.as_ptr() as *const __m256i);
    let nib = _mm256_set1_epi8(0x0F);
    let w = _mm256_set1_epi8(val as i8);
    let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), nib);
    let lo = _mm256_and_si256(v, nib);
    let a = _mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, w)) as u32;
    let b = _mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, w)) as u32;
    spread(a) | (spread(b) << 1)
}

#[inline]
fn adj_mask(pk: &[u8; 32], simd: bool) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: `simd` is only true when AVX2 was detected at runtime.
        return unsafe { adj_mask_avx2(pk) };
    }
    let _ = simd;
    adj_mask_scalar(pk)
}

#[inline]
fn nib_match_mask(pk: &[u8; 32], val: u8, simd: bool) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: `simd` is only true when AVX2 was detected at runtime.
        return unsafe { nib_match_mask_avx2(pk, val) };
    }
    let _ = simd;
    nib_match_mask_scalar(pk, val)
}

/// True if `m` contains a run of at least `len` consecutive set bits,
/// in O(log len) shift-AND steps.
#[inline]
fn has_ones_run(mut m: u64, len: u32) -> bool {
    if len == 0 {
        return true;
    }
    let mut r = 1u32;
    while r < len {
        let s = (len - r).min(r);
        m &= m >> s;
        r += s;
    }
    m != 0
}

/// True if `pk` matches the exact nibble pattern at `loc`. In `candidate` mode
/// (y-only encoding) the sign nibble is treated as a wildcard.
fn exact_match(pk: &[u8; 32], nibs: &[u8], loc: Location, candidate: bool, simd: bool) -> bool {
    let l = nibs.len();
    let eq = |pos: usize, val: u8| (candidate && pos == SIGN_NIBBLE) || nibble_at(pk, pos) == val;
    match loc {
        Location::Prefix => (0..l).all(|j| eq(j, nibs[j])),
        Location::Suffix => (0..l).all(|j| eq(64 - l + j, nibs[j])),
        Location::Anywhere => {
            // Bit-parallel scan: m's bit j = "pattern[..=k] matches starting
            // at j". Shifted ANDs pull in zeros past bit 63, so starts whose
            // pattern would run off the key drop out automatically.
            let wc = if candidate { 1u64 << SIGN_NIBBLE } else { 0 };
            let mut m = nib_match_mask(pk, nibs[0], simd) | wc;
            for (k, &nb) in nibs.iter().enumerate().skip(1) {
                if m == 0 {
                    return false;
                }
                m &= (nib_match_mask(pk, nb, simd) | wc) >> k;
            }
            m != 0
        }
    }
}

/// True if `pk` has a run of `>= n` identical nibbles at `loc`.
fn run_match(pk: &[u8; 32], n: u32, loc: Location, candidate: bool, simd: bool) -> bool {
    let n = n as usize;
    let eq = |pos: usize, c: u8| (candidate && pos == SIGN_NIBBLE) || nibble_at(pk, pos) == c;
    match loc {
        Location::Prefix => {
            let c = nibble_at(pk, 0);
            n <= 64 && (0..n).all(|pos| eq(pos, c))
        }
        Location::Suffix => {
            let c = nibble_at(pk, 63);
            n <= 64 && (0..n).all(|j| eq(63 - j, c))
        }
        Location::Anywhere => {
            // A run of n chars = n-1 consecutive adjacency bits. The sign
            // nibble wildcard sets both adjacencies touching char 62.
            let mut adj = adj_mask(pk, simd);
            if candidate {
                adj |= 0b11 << (SIGN_NIBBLE - 1);
            }
            has_ones_run(adj, n.saturating_sub(1) as u32)
        }
    }
}

impl Target {
    pub fn exact(location: Location, patterns: &[String]) -> Self {
        let exact = patterns
            .iter()
            .map(|p| {
                let nibs = p.bytes().map(nibble_from_ascii).collect();
                (p.clone(), nibs)
            })
            .collect();
        Target {
            location,
            what: What::Exact(exact),
            simd: detect_simd(),
        }
    }

    pub fn run(location: Location, n: u32) -> Self {
        Target {
            location,
            what: What::Run(n),
            simd: detect_simd(),
        }
    }

    /// Sound over-approximation on the y-only encoding (sign bit zeroed): never
    /// a false negative. Candidates are confirmed by [`Target::authoritative`].
    #[inline]
    pub fn candidate(&self, pk: &[u8; 32]) -> bool {
        match &self.what {
            What::Exact(ts) => ts
                .iter()
                .any(|(_, nibs)| exact_match(pk, nibs, self.location, true, self.simd)),
            What::Run(n) => run_match(pk, *n, self.location, true, self.simd),
        }
    }

    /// Exact check on the true compressed key; returns the matched label.
    #[inline]
    pub fn authoritative(&self, pk: &[u8; 32]) -> Option<String> {
        match &self.what {
            What::Exact(ts) => ts
                .iter()
                .find(|(_, nibs)| exact_match(pk, nibs, self.location, false, self.simd))
                .map(|(l, _)| l.clone()),
            What::Run(n) => {
                run_match(pk, *n, self.location, false, self.simd).then(|| format!("{}+ identical", n))
            }
        }
    }

    /// Kernel prefilter for the SIMD fast paths. Sound (never rejects a real
    /// match); lanes that fail skip the full byte encoding entirely.
    ///
    /// - Prefix-exact: byte 0 must match the leading pattern nibbles.
    /// - Prefix-run (n >= 2): the first two nibbles are equal, so byte 0 is
    ///   `0xCC` for some run char C. 00/FF keys are skipped by MeshCore, so
    ///   those two values are excluded outright.
    /// - Suffix-exact: byte 31 must match the trailing nibbles — minus the
    ///   sign bit, which the y-only encoding zeroes (mask 0x7F).
    /// - Suffix-run (n >= 2): the last two nibbles are equal: byte 31 is
    ///   `((C & 7) << 4) | C` in the sign-less encoding.
    /// - Anywhere-run (n >= 5): a run of n nibbles always contains
    ///   `floor((n-1)/2)` consecutive equal double-nibble bytes; the kernel
    ///   checks that byte-level condition (capped at 3 bytes, byte 31
    ///   wildcarded). Below n = 7 the byte 31 wildcard makes the condition
    ///   too leaky to pay for itself (measured slower), so no filter —
    ///   which costs nothing, since short-run searches finish in well under
    ///   a second anyway.
    /// - Anywhere-exact: no cheap kernel constraint exists; `None`.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub fn kernel_filter(&self) -> Option<crate::simd4::KernelFilter> {
        use crate::simd4::{FilterByte, KernelFilter};
        match (self.location, &self.what) {
            (Location::Prefix, What::Exact(ts)) => Some(KernelFilter::Byte(
                FilterByte::First,
                ts.iter()
                    .map(|(_, nibs)| {
                        if nibs.len() >= 2 {
                            (0xFF, (nibs[0] << 4) | nibs[1])
                        } else {
                            (0xF0, nibs[0] << 4)
                        }
                    })
                    .collect(),
            )),
            (Location::Prefix, What::Run(n)) if *n >= 2 => Some(KernelFilter::Byte(
                FilterByte::First,
                (1..=14).map(|c| (0xFF, c * 0x11)).collect(),
            )),
            (Location::Suffix, What::Exact(ts)) => Some(KernelFilter::Byte(
                FilterByte::Last,
                ts.iter()
                    .map(|(_, nibs)| {
                        let l = nibs.len();
                        if l >= 2 {
                            (0x7F, ((nibs[l - 2] & 7) << 4) | nibs[l - 1])
                        } else {
                            (0x0F, nibs[l - 1])
                        }
                    })
                    .collect(),
            )),
            (Location::Suffix, What::Run(n)) if *n >= 2 => Some(KernelFilter::Byte(
                FilterByte::Last,
                (0..=15).map(|c| (0x7F, ((c & 7) << 4) | c)).collect(),
            )),
            (Location::Anywhere, What::Run(n)) if *n >= 7 => {
                Some(KernelFilter::RunBytes(((*n - 1) / 2).min(3) as u8))
            }
            _ => None,
        }
    }

    /// Prefix strings if this is a prefix-exact search (the only mode the GPU
    /// kernels support); `None` otherwise (CPU-only).
    pub fn gpu_prefixes(&self) -> Option<Vec<String>> {
        match (self.location, &self.what) {
            (Location::Prefix, What::Exact(ts)) => Some(ts.iter().map(|(l, _)| l.clone()).collect()),
            _ => None,
        }
    }

    /// Rough expected number of keys to try, for the progress display.
    pub fn expected_attempts(&self) -> u64 {
        let p: f64 = match &self.what {
            What::Exact(ts) => ts
                .iter()
                .map(|(_, nibs)| {
                    let l = nibs.len() as i32;
                    let positions = match self.location {
                        Location::Anywhere => (64 - l + 1).max(1) as f64,
                        _ => 1.0,
                    };
                    positions * 16f64.powi(-l)
                })
                .sum(),
            What::Run(n) => {
                let n = *n as i32;
                let positions = match self.location {
                    Location::Anywhere => (64 - n + 1).max(1) as f64,
                    _ => 1.0,
                };
                positions * 16f64.powi(-(n - 1))
            }
        };
        if p > 0.0 {
            (1.0 / p).min(u64::MAX as f64) as u64
        } else {
            u64::MAX
        }
    }
}

/// Internal result containing keypair + which prefix matched.
struct MatchResult {
    keypair: MeshCoreKeypair,
    matched_prefix: String,
}

/// Where a worker reports a match.
///
/// `OneShot` stores the first match and flips the stop flag (the classic
/// "find one key and exit" behaviour). `Stream` forwards every match over a
/// channel and lets the worker keep scanning, for the run-forever mode.
enum Sink {
    OneShot {
        stop: Arc<AtomicBool>,
        result: Arc<Mutex<Option<MatchResult>>>,
    },
    Stream {
        tx: mpsc::Sender<MatchResult>,
    },
}

impl Sink {
    /// Report a match. Returns `true` if the worker should stop scanning
    /// (first hit in one-shot mode, or the stream receiver has hung up).
    fn emit(&self, m: MatchResult) -> bool {
        match self {
            Sink::OneShot { stop, result } => {
                stop.store(true, Ordering::Relaxed);
                *result.lock().unwrap() = Some(m);
                true
            }
            Sink::Stream { tx } => tx.send(m).is_err(),
        }
    }
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
    pub fn start(target: Arc<Target>, num_threads: usize) -> Self {
        let found = Arc::new(AtomicBool::new(false));
        let attempts = Arc::new(AtomicU64::new(0));
        let result: Arc<Mutex<Option<MatchResult>>> = Arc::new(Mutex::new(None));

        let mut workers = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            workers.push(spawn_cpu_worker(
                Arc::clone(&target),
                Arc::clone(&found),
                Arc::clone(&attempts),
                Sink::OneShot {
                    stop: Arc::clone(&found),
                    result: Arc::clone(&result),
                },
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
    pub fn start_gpu(target: Arc<Target>, gpu_searchers: Vec<Box<dyn GpuSearcher>>) -> Self {
        let found = Arc::new(AtomicBool::new(false));
        let attempts = Arc::new(AtomicU64::new(0));
        let result: Arc<Mutex<Option<MatchResult>>> = Arc::new(Mutex::new(None));

        let mut workers = Vec::with_capacity(gpu_searchers.len());
        for searcher in gpu_searchers {
            workers.push(thread::spawn({
                let target = Arc::clone(&target);
                let found = Arc::clone(&found);
                let attempts = Arc::clone(&attempts);
                let result = Arc::clone(&result);
                move || gpu_dispatch_loop(searcher, target, found, attempts, result)
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
        target: Arc<Target>,
        cpu_threads: usize,
        gpu_searchers: Vec<Box<dyn GpuSearcher>>,
    ) -> Self {
        let found = Arc::new(AtomicBool::new(false));
        let attempts = Arc::new(AtomicU64::new(0));
        let result: Arc<Mutex<Option<MatchResult>>> = Arc::new(Mutex::new(None));

        let mut workers = Vec::with_capacity(cpu_threads + gpu_searchers.len());

        // Spawn CPU workers
        for _ in 0..cpu_threads {
            workers.push(spawn_cpu_worker(
                Arc::clone(&target),
                Arc::clone(&found),
                Arc::clone(&attempts),
                Sink::OneShot {
                    stop: Arc::clone(&found),
                    result: Arc::clone(&result),
                },
            ));
        }

        // Spawn GPU dispatch threads
        for searcher in gpu_searchers {
            workers.push(thread::spawn({
                let target = Arc::clone(&target);
                let found = Arc::clone(&found);
                let attempts = Arc::clone(&attempts);
                let result = Arc::clone(&result);
                move || gpu_dispatch_loop(searcher, target, found, attempts, result)
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
/// inversion. The single field inversion (~265 field muls via the pow22523
/// chain) is a fixed per-batch cost, so amortizing it over more points is a
/// big CPU lever. With the fully vectorized canon+pack path the per-point
/// floor dropped enough that 1024 measures ~10% faster than 512 on a 4-core
/// Xeon; 2048 is flat-to-negative while doubling the working set (the SIMD
/// workers hold 3 Fe8 arrays of K = CHAIN_BATCH/8 entries, ~240 KB at 1024),
/// so 1024 sits at the knee of the curve.
///
/// The same 1024 also holds for the 4-wide AVX2 worker despite its ~272 KB
/// working set exceeding the 256 KB per-core L2 of Haswell/Broadwell-class
/// parts: the Montgomery arrays are read as sequential streams, so hardware
/// prefetch hides the L3 round-trips behind the field arithmetic. Verified
/// empirically on a Xeon D-1541 — a 512 batch measured the same or slightly
/// slower there, while costing ~6% on large-L2 cores.
const CHAIN_BATCH: usize = 1024;

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
    target: Arc<Target>,
    stop: Arc<AtomicBool>,
    attempts: Arc<AtomicU64>,
    sink: Sink,
) -> JoinHandle<()> {
    thread::spawn(move || {
        // On x86-64 with AVX2, scan four independent keys per step in SIMD
        // lanes (radix-2^25.5 field, GPU-style). Falls back to the scalar
        // chain otherwise.
        #[cfg(target_arch = "x86_64")]
        {
            if CHAIN_BATCH % 8 == 0 && crate::simd4::avx512_ok() {
                // SAFETY: guarded by the runtime avx512f+bw check.
                unsafe { cpu_worker_simd512(&target, &stop, &attempts, &sink) };
                return;
            }
            if CHAIN_BATCH % 4 == 0 && std::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by the runtime avx2 check above.
                unsafe { cpu_worker_simd(&target, &stop, &attempts, &sink) };
                return;
            }
        }
        cpu_worker_scalar(&target, &stop, &attempts, &sink);
    })
}

/// Build the matched keypair with a fresh random private-key prefix half.
fn build_match(
    matched_prefix: String,
    public_key: [u8; 32],
    match_scalar: [u8; 32],
) -> MatchResult {
    let mut prefix_half = [0u8; 32];
    OsRng.fill_bytes(&mut prefix_half);
    let mut private_key = [0u8; 64];
    private_key[..32].copy_from_slice(&match_scalar);
    private_key[32..].copy_from_slice(&prefix_half);
    MatchResult {
        keypair: MeshCoreKeypair {
            public_key,
            private_key,
        },
        matched_prefix,
    }
}

/// Scalar `+8B` chained-compress worker (portable fallback).
fn cpu_worker_scalar(
    target: &Target,
    stop: &AtomicBool,
    attempts: &AtomicU64,
    sink: &Sink,
) {
    let eight_b = ED25519_BASEPOINT_TABLE * &Scalar::from(8u64);

    let mut scalar = [0u8; 32];
    OsRng.fill_bytes(&mut scalar);
    clamp_scalar(&mut scalar);
    let mut point: EdwardsPoint = EdwardsPoint::mul_base_clamped(scalar);

    let mut local_count: u64 = 0;

    while !stop.load(Ordering::Relaxed) {
        let (compressed, next_point) = point.chain_compress_y_only::<CHAIN_BATCH>(&eight_b);

        for (i, public_key) in compressed.iter().enumerate() {
            if should_skip(public_key) || !target.candidate(public_key) {
                continue;
            }
            // Candidate on the y-only encoding: recompute point + i·8B and
            // compress it properly (with sign) for the authoritative check.
            let mut hit_point = point;
            for _ in 0..i {
                hit_point += eight_b;
            }
            let real = hit_point.compress().to_bytes();
            if let Some(label) = target.authoritative(&real) {
                let mut match_scalar = scalar;
                advance_scalar(&mut match_scalar, 8 * i as u64);
                let m = build_match(label, real, match_scalar);
                // In stream mode the per-match count is rolled into the batch
                // accounting below; only one-shot needs an exact final tally.
                if sink.emit(m) {
                    attempts.fetch_add(local_count + i as u64 + 1, Ordering::Relaxed);
                    return;
                }
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
    target: &Target,
    stop: &AtomicBool,
    attempts: &AtomicU64,
    sink: &Sink,
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

    // Prefix and suffix searches supply a one-byte filter (prefilter fast
    // path); anywhere searches fully encode every lane.
    let filter = target.kernel_filter();

    let mut out: Box<[[[u8; 32]; 4]; K]> = Box::new([[[0u8; 32]; 4]; K]);
    let mut local_count: u64 = 0;

    while !stop.load(Ordering::Relaxed) {
        let f = filter.as_ref();
        p4 = avx2::chain_y_only::<K>(p4, &niels, f, &mut out);

        for s in 0..K {
            for lane in 0..4 {
                let public_key = &out[s][lane];
                if should_skip(public_key) || !target.candidate(public_key) {
                    continue;
                }
                let mut hit = starts[lane];
                for _ in 0..s {
                    hit += eight_b;
                }
                let real = hit.compress().to_bytes();
                if let Some(label) = target.authoritative(&real) {
                    let mut match_scalar = scalars[lane];
                    advance_scalar(&mut match_scalar, 8 * s as u64);
                    let m = build_match(label, real, match_scalar);
                    if sink.emit(m) {
                        attempts.fetch_add(
                            local_count + (s * 4 + lane) as u64 + 1,
                            Ordering::Relaxed,
                        );
                        return;
                    }
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
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn cpu_worker_simd512(
    target: &Target,
    stop: &AtomicBool,
    attempts: &AtomicU64,
    sink: &Sink,
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

    // Prefix and suffix searches supply a one-byte filter (prefilter fast
    // path); anywhere searches fully encode every lane.
    let filter = target.kernel_filter();

    let mut out: Box<[[[u8; 32]; 8]; K]> = Box::new([[[0u8; 32]; 8]; K]);
    let mut local_count: u64 = 0;

    while !stop.load(Ordering::Relaxed) {
        let f = filter.as_ref();
        p8 = avx512::chain_y_only::<K>(p8, &niels, f, &mut out);

        for s in 0..K {
            for lane in 0..8 {
                let public_key = &out[s][lane];
                if should_skip(public_key) || !target.candidate(public_key) {
                    continue;
                }
                let mut hit = starts[lane];
                for _ in 0..s {
                    hit += eight_b;
                }
                let real = hit.compress().to_bytes();
                if let Some(label) = target.authoritative(&real) {
                    let mut match_scalar = scalars[lane];
                    advance_scalar(&mut match_scalar, 8 * s as u64);
                    let m = build_match(label, real, match_scalar);
                    if sink.emit(m) {
                        attempts.fetch_add(
                            local_count + (s * 8 + lane) as u64 + 1,
                            Ordering::Relaxed,
                        );
                        return;
                    }
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

/// Handle for a run-forever streaming search.
///
/// Workers never stop on a match; instead every hit is pushed over a channel.
/// The caller drains [`StreamHandle::next_match`] and does whatever it likes
/// with each result (print it, append to a file, ...). The search runs until
/// [`StreamHandle::stop`] is called or the handle is dropped.
pub struct StreamHandle {
    stop: Arc<AtomicBool>,
    attempts: Arc<AtomicU64>,
    rx: mpsc::Receiver<MatchResult>,
    start: Instant,
    workers: Vec<JoinHandle<()>>,
}

impl StreamHandle {
    /// Start a streaming CPU search across `num_threads` worker threads.
    pub fn start(target: Arc<Target>, num_threads: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let attempts = Arc::new(AtomicU64::new(0));
        let (tx, rx) = mpsc::channel();

        let mut workers = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            workers.push(spawn_cpu_worker(
                Arc::clone(&target),
                Arc::clone(&stop),
                Arc::clone(&attempts),
                Sink::Stream { tx: tx.clone() },
            ));
        }
        // Drop our own sender so `rx` disconnects once every worker exits.
        drop(tx);

        StreamHandle {
            stop,
            attempts,
            rx,
            start: Instant::now(),
            workers,
        }
    }

    /// Block until the next match arrives. Returns `None` once all workers
    /// have stopped (e.g. after [`StreamHandle::stop`]).
    pub fn next_match(&self) -> Option<SearchResult> {
        let m = self.rx.recv().ok()?;
        let elapsed = self.start.elapsed().as_secs_f64();
        let attempts = self.attempts.load(Ordering::Relaxed);
        Some(SearchResult {
            public_key: hex::encode_upper(m.keypair.public_key),
            private_key: hex::encode_upper(m.keypair.private_key),
            matched_prefix: m.matched_prefix,
            attempts,
            elapsed_secs: elapsed,
        })
    }

    /// Current search statistics (keys checked, rate, elapsed).
    #[allow(dead_code)]
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

    /// Signal all workers to stop and join them.
    #[allow(dead_code)]
    pub fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        for h in self.workers {
            h.join().unwrap();
        }
    }
}

/// GPU dispatch loop shared by start_gpu and start_hybrid.
/// Runs search_batch in a loop until a match is found or another thread signals done.
fn gpu_dispatch_loop(
    mut searcher: Box<dyn GpuSearcher>,
    target: Arc<Target>,
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
                    // The GPU kernel emits the true compressed key (with sign).
                    let matched_prefix = target
                        .authoritative(&kp.public_key)
                        .unwrap_or_default();
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

    fn key_from_hex(h: &str) -> [u8; 32] {
        let bytes = hex::decode(h).unwrap();
        let mut k = [0u8; 32];
        k.copy_from_slice(&bytes);
        k
    }

    #[test]
    fn target_exact_locations() {
        // ...AB at start, CD anywhere, EF at end.
        let key = key_from_hex("AB00000000000000000000000000CD0000000000000000000000000000DD00EF");
        assert!(Target::exact(Location::Prefix, &["AB".into()]).authoritative(&key).is_some());
        assert!(Target::exact(Location::Prefix, &["CD".into()]).authoritative(&key).is_none());
        assert!(Target::exact(Location::Anywhere, &["CD".into()]).authoritative(&key).is_some());
        assert!(Target::exact(Location::Anywhere, &["DD00EF".into()]).authoritative(&key).is_some());
        assert!(Target::exact(Location::Suffix, &["EF".into()]).authoritative(&key).is_some());
        assert!(Target::exact(Location::Suffix, &["AB".into()]).authoritative(&key).is_none());
    }

    #[test]
    fn target_run_locations() {
        // 64 nibbles: CCC at the start, DDDDD in the middle, 777 at the end,
        // the rest a 1,2,3,4 cycle so there are no other runs.
        let mut nib = [0u8; 64];
        for (i, n) in nib.iter_mut().enumerate() {
            *n = 1 + (i % 4) as u8;
        }
        nib[0..3].fill(0xC);
        nib[30..35].fill(0xD);
        nib[61..64].fill(0x7);
        let mut key = [0u8; 32];
        for i in 0..32 {
            key[i] = (nib[2 * i] << 4) | nib[2 * i + 1];
        }
        assert!(Target::run(Location::Prefix, 3).authoritative(&key).is_some());
        assert!(Target::run(Location::Prefix, 4).authoritative(&key).is_none());
        assert!(Target::run(Location::Anywhere, 5).authoritative(&key).is_some());
        assert!(Target::run(Location::Anywhere, 6).authoritative(&key).is_none());
        assert!(Target::run(Location::Suffix, 3).authoritative(&key).is_some());
        assert!(Target::run(Location::Suffix, 4).authoritative(&key).is_none());
    }

    #[test]
    fn target_candidate_never_misses_authoritative() {
        // The candidate over-approximation must accept anything authoritative
        // accepts, for every nibble value at the sign position.
        for last_byte in 0u8..=255 {
            let mut key = key_from_hex("AB000000000000000000000000000000000000000000000000000000000000CD");
            key[31] = last_byte;
            for t in [
                Target::exact(Location::Suffix, &[format!("{:02X}", last_byte)]),
                Target::run(Location::Suffix, 2),
                Target::run(Location::Anywhere, 2),
            ] {
                if t.authoritative(&key).is_some() {
                    // y-only zeroes the sign bit: emulate that and check candidate.
                    let mut yonly = key;
                    yonly[31] &= 0x7f;
                    assert!(t.candidate(&yonly), "candidate missed a real match");
                }
            }
        }
    }

    /// Soundness property of every kernel prefilter table: a key that
    /// authoritatively matches (with either sign-bit value) must have its
    /// y-only filter byte accepted. A violation means the kernel silently
    /// drops real matches — the search never terminates. This is the test
    /// that catches wrong (mask, value) table entries (e.g. transposed
    /// nibbles or a missing run char) which end-to-end searches mask by
    /// simply finding a different key.
    #[test]
    fn kernel_filter_soundness() {
        use crate::simd4::{FilterByte, KernelFilter};
        fn filter_byte(true_key: &[u8; 32], byte: FilterByte) -> u8 {
            match byte {
                FilterByte::First => true_key[0],
                // The kernel tests the y-only encoding: sign bit zeroed.
                FilterByte::Last => true_key[31] & 0x7F,
            }
        }
        fn accepts(pairs: &[(u8, u8)], b: u8) -> bool {
            pairs.iter().any(|&(m, v)| b & m == v)
        }
        fn plant(key: &mut [u8; 32], pos: usize, nib: u8) {
            let by = &mut key[pos / 2];
            if pos % 2 == 0 {
                *by = (*by & 0x0F) | (nib << 4);
            } else {
                *by = (*by & 0xF0) | nib;
            }
        }
        let mut st = 0x5555_aaaa_1234_5678u64;
        let mut rnd = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (st >> 33) as u8
        };

        // Exact patterns, lengths 1..=5, planted at prefix and suffix.
        for len in 1..=5usize {
            for _ in 0..50 {
                let nibs: Vec<u8> = (0..len).map(|_| rnd() % 16).collect();
                let pat: String = nibs.iter().map(|n| format!("{:X}", n)).collect();
                for (loc, at) in [(Location::Prefix, 0usize), (Location::Suffix, 64 - len)] {
                    let t = Target::exact(loc, &[pat.clone()]);
                    let Some(KernelFilter::Byte(byte, pairs)) = t.kernel_filter() else {
                        continue;
                    };
                    for sign in [0u8, 0x80] {
                        let mut key = [0u8; 32];
                        for b in key.iter_mut() {
                            *b = rnd();
                        }
                        for (j, &nb) in nibs.iter().enumerate() {
                            plant(&mut key, at + j, nb);
                        }
                        key[31] = (key[31] & 0x7F) | sign;
                        // Forcing the sign bit may break the planted pattern
                        // at char 62; then the key is genuinely not a match.
                        if t.authoritative(&key).is_none() {
                            continue;
                        }
                        assert!(
                            accepts(&pairs, filter_byte(&key, byte)),
                            "filter rejected a real {loc:?} exact match: pat={pat} sign={sign:02X} key={}",
                            hex::encode_upper(key),
                        );
                    }
                }
            }
        }

        // Run targets: every run char, n in 2..=5, at prefix and suffix.
        for n in 2..=5u32 {
            for c in 0u8..=15 {
                for loc in [Location::Prefix, Location::Suffix] {
                    let t = Target::run(loc, n);
                    let Some(KernelFilter::Byte(byte, pairs)) = t.kernel_filter() else {
                        continue;
                    };
                    for sign in [0u8, 0x80] {
                        let mut key = [0u8; 32];
                        for b in key.iter_mut() {
                            *b = rnd();
                        }
                        let range = match loc {
                            Location::Prefix => 0..n as usize,
                            _ => (64 - n as usize)..64,
                        };
                        for pos in range {
                            plant(&mut key, pos, c);
                        }
                        key[31] = (key[31] & 0x7F) | sign;
                        if t.authoritative(&key).is_none() {
                            continue;
                        }
                        // Keys starting 00/FF are dropped by should_skip
                        // before the filter result matters.
                        if should_skip(&key) {
                            continue;
                        }
                        assert!(
                            accepts(&pairs, filter_byte(&key, byte)),
                            "filter rejected a real {loc:?} run: n={n} c={c:X} sign={sign:02X} key={}",
                            hex::encode_upper(key),
                        );
                    }
                }
            }
        }
    }

    /// Soundness of the anywhere-run kernel prefilter: every key whose true
    /// (sign-restored) encoding contains a run of >= n hex chars must pass
    /// the RunBytes byte-condition on its y-only encoding, for every run
    /// char, every position (including runs ending at char 63 and crossing
    /// the sign nibble), and both sign-bit values. A violation silently
    /// drops real matches in the kernel.
    /// Soundness of the anywhere-run kernel prefilter: every key whose true
    /// (sign-restored) encoding contains a run of >= n hex chars must pass
    /// the RunBytes byte-condition on its y-only encoding, for every run
    /// char, every position (including runs ending at char 63 and crossing
    /// the sign nibble), and both sign-bit values. A violation silently
    /// drops real matches in the kernel.
    #[test]
    fn kernel_run_filter_soundness() {
        use crate::simd4::{run_bytes_accept, KernelFilter};
        let mut st = 0x9a8b_7c6d_5e4f_3a2bu64;
        let mut rnd = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (st >> 33) as u8
        };
        for n in 7u32..=12 {
            let t = Target::run(Location::Anywhere, n);
            let Some(KernelFilter::RunBytes(m)) = t.kernel_filter() else {
                panic!("anywhere-run n={n} must produce a RunBytes filter");
            };
            for start in 0..=(64 - n as usize) {
                for c in 0u8..=15 {
                    for sign in [0u8, 0x80] {
                        let mut key = [0u8; 32];
                        for b in key.iter_mut() {
                            *b = rnd();
                        }
                        for pos in start..start + n as usize {
                            let by = &mut key[pos / 2];
                            if pos % 2 == 0 {
                                *by = (*by & 0x0F) | (c << 4);
                            } else {
                                *by = (*by & 0xF0) | c;
                            }
                        }
                        key[31] = (key[31] & 0x7F) | sign;
                        // The sign bit may have broken the planted run at
                        // char 62; then this key genuinely may not match.
                        if t.authoritative(&key).is_none() {
                            continue;
                        }
                        let mut yonly = key;
                        yonly[31] &= 0x7F;
                        assert!(
                            run_bytes_accept(&yonly, m),
                            "RunBytes({m}) rejected real run n={n} c={c:X} start={start} sign={sign:02X} key={}",
                            hex::encode_upper(key),
                        );
                    }
                }
            }
        }
    }

    /// The sign-nibble wildcard in the anywhere matchers is load-bearing:
    /// a true key whose qualifying run/pattern crosses char 62 changes there
    /// when the sign bit is zeroed, and candidate() must still accept the
    /// y-only encoding. Directed keys, both matcher backends.
    #[test]
    fn candidate_wildcard_at_sign_nibble() {
        let mut st = 0x1122_3344_5566_7788u64;
        let mut rnd = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (st >> 33) as u8
        };
        for _ in 0..100 {
            // Run chars with the high bit set: zeroing the sign bit changes
            // char 62 from c to c-8, so only the wildcard keeps the match.
            for c in 8u8..=15 {
                let mut key = [0u8; 32];
                for b in key.iter_mut() {
                    *b = rnd();
                }
                // Run of 5 at chars 59..=63 (byte29 lo, bytes 30-31 full).
                key[29] = (key[29] & 0xF0) | c;
                key[30] = (c << 4) | c;
                key[31] = (c << 4) | c;
                let mut yonly = key;
                yonly[31] &= 0x7F;
                for n in 2..=5u32 {
                    for simd in [false, true] {
                        if simd && !detect_simd() {
                            continue;
                        }
                        let mut t = Target::run(Location::Anywhere, n);
                        t.simd = simd;
                        if t.authoritative(&key).is_some() {
                            assert!(
                                t.candidate(&yonly),
                                "candidate missed anywhere-run n={n} c={c:X} simd={simd} key={}",
                                hex::encode_upper(key),
                            );
                        }
                    }
                }
                // Exact anywhere: patterns of length 1..=3 read from the true
                // key at starts covering char 62.
                for len in 1..=3usize {
                    for start in (62usize.saturating_sub(len - 1))..=(64 - len) {
                        let nibs: Vec<u8> =
                            (start..start + len).map(|p| nibble_at(&key, p)).collect();
                        let pat: String = nibs.iter().map(|n| format!("{:X}", n)).collect();
                        for simd in [false, true] {
                            if simd && !detect_simd() {
                                continue;
                            }
                            let mut t = Target::exact(Location::Anywhere, &[pat.clone()]);
                            t.simd = simd;
                            assert!(t.authoritative(&key).is_some());
                            assert!(
                                t.candidate(&yonly),
                                "candidate missed anywhere-exact pat={pat} start={start} simd={simd} key={}",
                                hex::encode_upper(key),
                            );
                        }
                    }
                }
            }
        }
    }

    /// The bit-parallel anywhere matchers must agree with a plain reference
    /// scan on random keys, for both the SIMD and scalar mask builders.
    #[test]
    fn swar_matchers_match_reference() {
        fn ref_longest_run(pk: &[u8; 32]) -> u32 {
            let mut max = 1u32;
            let mut run = 1u32;
            for pos in 1..64 {
                if nibble_at(pk, pos) == nibble_at(pk, pos - 1) {
                    run += 1;
                } else {
                    run = 1;
                }
                max = max.max(run);
            }
            max
        }
        fn ref_exact_anywhere(pk: &[u8; 32], nibs: &[u8], candidate: bool) -> bool {
            let l = nibs.len();
            let eq = |pos: usize, val: u8| {
                (candidate && pos == SIGN_NIBBLE) || nibble_at(pk, pos) == val
            };
            (0..=64 - l).any(|s| (0..l).all(|j| eq(s + j, nibs[j])))
        }

        let mut st = 0x0123_4567_89ab_cdefu64;
        let mut rnd = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (st >> 33) as u8
        };
        for iter in 0..4000 {
            let mut pk = [0u8; 32];
            for b in pk.iter_mut() {
                *b = rnd();
            }
            // Plant a run sometimes so long runs are exercised, not just the
            // random-tail distribution.
            if iter % 3 == 0 {
                let start = (rnd() % 28) as usize;
                let len = 2 + (rnd() % 12) as usize;
                let c = rnd() % 16;
                for j in start..(start + len).min(64) {
                    let byte = &mut pk[j / 2];
                    if j % 2 == 0 {
                        *byte = (*byte & 0x0F) | (c << 4);
                    } else {
                        *byte = (*byte & 0xF0) | c;
                    }
                }
            }

            for simd in [false, true] {
                if simd && !detect_simd() {
                    continue;
                }
                // Exact run matching (authoritative mode) vs reference.
                let lr = ref_longest_run(&pk);
                for n in 1..=16u32 {
                    assert_eq!(
                        run_match(&pk, n, Location::Anywhere, false, simd),
                        lr >= n,
                        "run n={n} simd={simd} pk={}",
                        hex::encode_upper(pk),
                    );
                }
                // Candidate mode must accept at least everything the exact
                // mode accepts (sound over-approximation).
                for n in 1..=16u32 {
                    if lr >= n {
                        assert!(run_match(&pk, n, Location::Anywhere, true, simd));
                    }
                }
                // Exact anywhere pattern matching, both modes, patterns taken
                // from the key (guaranteed hits) and random (mostly misses).
                for &(start, len) in &[(0usize, 3usize), (13, 5), (59, 5), (30, 8)] {
                    let nibs: Vec<u8> = (start..start + len).map(|p| nibble_at(&pk, p)).collect();
                    for cand in [false, true] {
                        assert_eq!(
                            exact_match(&pk, &nibs, Location::Anywhere, cand, simd),
                            ref_exact_anywhere(&pk, &nibs, cand),
                            "planted pattern start={start} len={len} cand={cand} simd={simd}",
                        );
                    }
                }
                let rand_nibs: Vec<u8> = (0..4).map(|_| rnd() % 16).collect();
                for cand in [false, true] {
                    assert_eq!(
                        exact_match(&pk, &rand_nibs, Location::Anywhere, cand, simd),
                        ref_exact_anywhere(&pk, &rand_nibs, cand),
                        "random pattern cand={cand} simd={simd}",
                    );
                }
            }
        }
    }

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
        let handle = SearchHandle::start(Arc::new(Target::exact(Location::Prefix, &["A".to_string()])), 2);
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
        let handle = SearchHandle::start(Arc::new(Target::exact(Location::Prefix, &["A".to_string(), "B".to_string()])), 2);
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
        let handle = SearchHandle::start(Arc::new(Target::exact(Location::Prefix, &["A".to_string()])), 2);
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

    /// End-to-end soundness for every search mode: run a real multi-threaded
    /// search with an easy target, then check both that the key satisfies the
    /// target and that the returned scalar reproduces the public key.
    #[test]
    fn search_all_modes_end_to_end() {
        let targets: Vec<(Target, Box<dyn Fn(&str) -> bool>)> = vec![
            (
                Target::exact(Location::Suffix, &["7".to_string()]),
                Box::new(|pk: &str| pk.ends_with('7')),
            ),
            // Two differing chars: exercises the two-nibble suffix kernel
            // filter arm (a transposed-nibble table would hang this search).
            (
                Target::exact(Location::Suffix, &["AB".to_string()]),
                Box::new(|pk: &str| pk.ends_with("AB")),
            ),
            (
                Target::exact(Location::Anywhere, &["ABC".to_string()]),
                Box::new(|pk: &str| pk.contains("ABC")),
            ),
            (
                Target::run(Location::Prefix, 2),
                Box::new(|pk: &str| {
                    let b = pk.as_bytes();
                    b[0] == b[1]
                }),
            ),
            (
                Target::run(Location::Suffix, 2),
                Box::new(|pk: &str| {
                    let b = pk.as_bytes();
                    b[62] == b[63]
                }),
            ),
            (
                Target::run(Location::Anywhere, 4),
                Box::new(|pk: &str| {
                    pk.as_bytes().windows(4).any(|w| w.iter().all(|&c| c == w[0]))
                }),
            ),
            // n = 7 turns on the RunBytes kernel prefilter — an unsound
            // filter would hang this search.
            (
                Target::run(Location::Anywhere, 7),
                Box::new(|pk: &str| {
                    pk.as_bytes().windows(7).any(|w| w.iter().all(|&c| c == w[0]))
                }),
            ),
        ];
        for (target, check) in targets {
            let handle = SearchHandle::start(Arc::new(target), 2);
            let result = handle.finish().expect("search should find a match");
            assert!(
                check(&result.public_key),
                "key {} does not satisfy its target",
                result.public_key
            );
            let priv_bytes = hex::decode(&result.private_key).unwrap();
            let mut scalar = [0u8; 32];
            scalar.copy_from_slice(&priv_bytes[..32]);
            let derived = EdwardsPoint::mul_base_clamped(scalar).compress().to_bytes();
            assert_eq!(
                hex::encode_upper(derived),
                result.public_key,
                "scalar does not reproduce pubkey"
            );
        }
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
