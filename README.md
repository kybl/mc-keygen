# mc-keygen

Vanity Ed25519 key generator for [MeshCore](https://github.com/ripplebiz/MeshCore). Find keys whose public key contains a chosen hex pattern — at the start, at the end, or anywhere — or a run of repeated hex characters.

> **CPU-optimized fork** of [samschlegel/mc-keygen](https://github.com/samschlegel/mc-keygen) by Sam Schlegel.

### What's new in this fork

- **~5× faster CPU search** (on an AVX-512 machine; ~3× on an AVX2-only CPU).
  The search loop was rewritten: a `+8B` point-addition chain instead of a
  full scalar-multiply per key, Montgomery batched inversion, y-only
  compression, SIMD-across-keys (runtime AVX-512 / AVX2 / scalar dispatch),
  and vectorized kernel prefilters. See [docs/benchmarks.md](docs/benchmarks.md).
- **`--where` — match position:** `prefix` (default), `anywhere`, or `suffix`.
- **`--run N` — repeated-character search:** find a run of at least N identical
  hex characters (combine with `--where`, e.g. `--run 12 --where anywhere`).
- **`--stream` — run forever:** print every match to stdout and keep going,
  for leaving a long search on a server and collecting all hits later.
- **Bug fix:** the TUI now restores the terminal cursor on exit.
- **Correctness:** every soundness-critical prefilter is covered by
  differential and mutation-verified property tests.

The original prefix search, JSON output, TUI, and CUDA/Metal GPU support are
all unchanged.

## Usage

```
mc-keygen <PATTERN>... [OPTIONS]
mc-keygen --run <N> [OPTIONS]
```

**Options:**
- `--where <prefix|anywhere|suffix>` — where in the 64-char hex key the pattern must appear (default: `prefix`)
- `--run <N>` — instead of an exact pattern, find a run of at least N identical hex characters (combine with `--where`)
- `-t, --threads <N>` — worker threads (default: all logical cores)
- `--json` — output result as JSON (no TUI, no color)
- `--stream` — run forever: print every match to stdout (one per line) and keep searching instead of stopping at the first hit. CPU-only, no TUI. Ideal for leaving a long search running on a server and collecting all results later (redirect stdout to a file).
- `--cpu-only` — force CPU-only search, skip GPU even if available (requires `cuda` or `metal` feature)
- `--gpu-only` — force GPU-only search, no CPU threads (requires `cuda` or `metal` feature; prefix search only)
- `--verify` — cross-check GPU keygen against CPU (requires `cuda` or `metal` feature)
- `--benchmark <SECS>` — run GPU search for N seconds, tally every match with a no-early-exit kernel, and compare observed vs expected count to validate the reported rate (requires `cuda` or `metal` feature)

When built with GPU support (`cuda` or `metal` feature), the default mode is **hybrid**: both CPU threads and GPU run concurrently, and the first match from either wins. If no GPU is detected at runtime, the tool falls back to CPU-only with a warning. The GPU kernels only handle prefix search; the other modes are CPU-only.

**Examples:**
```bash
mc-keygen AB             # find a key starting with AB (hybrid if GPU available)
mc-keygen AB CD EF       # find a key matching ANY of these prefixes
mc-keygen DEAD --where anywhere   # DEAD anywhere in the key
mc-keygen BEEF --where suffix     # key ending in BEEF
mc-keygen --run 10 --where anywhere   # >=10 identical chars in a row, anywhere
mc-keygen DEAD -t 4      # use 4 threads
mc-keygen AB --json      # machine-readable output
mc-keygen ABCDEF --gpu-only   # GPU only for longer prefixes
mc-keygen AB --cpu-only       # force CPU even when GPU is available
mc-keygen ABCD --stream > keys.txt &   # run for days, collect every match in a file
```

### Search modes

Two independent axes: **where** (`--where prefix|anywhere|suffix`) and **what**
(exact hex pattern(s), or `--run N` = at least N identical hex chars). Any
combination works, e.g. `--run 12 --where suffix` finds keys *ending* in 12+
identical characters. Prefix patterns are capped at 62 chars and may not start
with `00`/`FF` (MeshCore skips those keys); anywhere/suffix patterns may be
1-64 chars of any hex.

### Streaming mode (`--stream`)

By default the search stops at the first match. With `--stream` it never stops: every matching key is written to stdout the moment it's found and the search keeps running until you kill it (Ctrl+C). This is meant for "start it on a server, come back in a week, read all the results":

```bash
nohup mc-keygen ABCD --stream > keys.txt 2>log.txt &
# ... days later ...
cat keys.txt    # every key found so far
```

Output is one line per match, tab-separated:

```
<matched>	<public_key>	<private_key>
```

With `--stream --json` each line is a standalone JSON object (JSON Lines) instead. Each line is flushed immediately, so the file is always up to date even if the process is killed.

Multiple prefixes can be passed in a single invocation — every key is checked against all of them, so searching for N prefixes is ~N× more efficient than N separate runs. The JSON output includes a `matched_prefix` field.

### Search difficulty

Each hex character multiplies expected attempts by 16:

| Prefix | Expected attempts | CPU time¹ | GPU time¹ |
|--------|-------------------|-----------|-----------|
| 5 char | ~1M               | instant   | instant   |
| 6 char | ~16M              | <1s       | <1s       |
| 7 char | ~268M             | ~11s      | ~4s       |
| 8 char | ~4.3B             | ~3m       | ~63s      |
| 9 char | ~69B              | ~48m      | ~17m      |
| 10 char| ~1.1T             | ~13h      | ~4.5h     |

¹ CPU: Xeon D-1541, 8C/16T, AVX2 path (~24M keys/s). GPU: RTX 4090 (~68M keys/s). A 4-core AVX-512 cloud VM does ~34M keys/s. See [docs/gpu.md](docs/gpu.md) for GPU details.

An anywhere-run search (`--run N --where anywhere`) is much easier than a
prefix of the same length: a run of N chars has ~16× lower cost per extra
char *and* ~60 possible positions, e.g. `--run 12` ≈ a 9.5-char prefix.

## Building

```bash
cargo build --release                    # CPU only
cargo build --release --features cuda    # with NVIDIA GPU support
cargo build --release --features metal   # with Apple Metal GPU support
```

CUDA support requires the NVIDIA CUDA Toolkit. Metal support requires macOS with an Apple Silicon or AMD GPU. See [docs/gpu.md](docs/gpu.md) for details.

### PGO build (measure before trusting it)

`./scripts/pgo-build.sh` produces a profile-guided-optimized binary. The
effect is **microarchitecture-specific and can go either way**: measured
**+16%** on an Ice Lake-class AVX-512 machine but **−8%** on a Broadwell
Xeon D-1541 (AVX2 path) — code-layout changes that help one CPU hurt
another. Build it on the target machine, compare against the plain
`--release` binary with `bench -m cpu1`, and keep whichever wins there.
Requires `rustup component add llvm-tools-preview`.

### Thread count

The default is one worker per logical CPU. On homogeneous SMT machines the
sibling thread hides field-arithmetic latency and helps (~+22% on a Xeon
D-1541). On hybrid Intel P+E CPUs the optimum may differ — run
`mc-keygen bench -m all --machine-label <name>` on an otherwise idle machine
to compare the physical-core (`cpuP`) and all-logical (`cpuN`) rates, and
pass the winner with `-t` if it isn't the default.

## How it works

Each worker draws one random clamped Ed25519 scalar from the OS CSPRNG and
does a single full scalar-multiplication to get its starting point. From
there, candidates come cheaply:

1. **+8B chain** — successive candidate points are `P, P+8B, P+16B, …`
   (one 7-mul mixed point addition each, with a precomputed affine-niels
   addend), and the scalar advances by 8 in lockstep, so any hit's private
   key is immediately known. Stepping by 8 keeps the cofactor-clamped form.
2. **Batched compression** — points are compressed 1024 at a time with one
   shared Montgomery batch inversion (one field inversion per 1024 keys),
   encoding only the y-coordinate (the sign bit isn't needed to scan).
3. **SIMD across keys** — on x86-64 the whole pipeline runs 8 independent
   chains in AVX-512 lanes (4 in AVX2) using a radix-2^25.5 field, one key
   per 64-bit lane, with runtime dispatch (AVX-512 → AVX2 → scalar).
4. **Kernel prefilters** — before a candidate is even byte-encoded, a
   vectorized one-byte test (byte 0 for prefix, byte 31 for suffix, a
   repeated-byte condition for anywhere-runs) rejects the ~99%+ of keys
   that can't match; survivors get an exact match check, and a real hit is
   recomputed and verified on the true signed encoding.

Keys starting with `00` or `FF` are skipped (reserved by MeshCore). The GPU
path uses the same +8B chain strategy per GPU thread (see
[docs/gpu.md](docs/gpu.md)). All matching is exact: every soundness-critical
filter is covered by differential and mutation-verified property tests.

## Sources

- [MeshCore](https://github.com/ripplebiz/MeshCore) — the mesh networking firmware these keys are for
- [MeshCore mc-keygen web tool](https://gessaman.com/mc-keygen/) — reference implementation of the key generation algorithm
- [Ed25519 / RFC 8032](https://datatracker.ietf.org/doc/html/rfc8032) — the signature scheme spec
- [curve25519-dalek](https://github.com/dalek-cryptography/curve25519-dalek) — Rust Ed25519 elliptic curve library
- [ratatui](https://github.com/ratatui/ratatui) — TUI framework for the progress display
