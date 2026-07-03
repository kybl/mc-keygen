# mc-keygen

Vanity Ed25519 key generator for [MeshCore](https://github.com/ripplebiz/MeshCore). Find keys whose public key starts with a chosen hex prefix.

## Usage

```
mc-keygen <PREFIX>... [OPTIONS]
```

**Options:**
- `-t, --threads <N>` — worker threads (default: all cores)
- `--json` — output result as JSON (no TUI, no color)
- `--stream` — run forever: print every match to stdout (one per line) and keep searching instead of stopping at the first hit. CPU-only, no TUI. Ideal for leaving a long search running on a server and collecting all results later (redirect stdout to a file).
- `--cpu-only` — force CPU-only search, skip GPU even if available (requires `cuda` or `metal` feature)
- `--gpu-only` — force GPU-only search, no CPU threads (requires `cuda` or `metal` feature)
- `--verify` — cross-check GPU keygen against CPU (requires `cuda` or `metal` feature)
- `--benchmark <SECS>` — run GPU search for N seconds, tally every match with a no-early-exit kernel, and compare observed vs expected count to validate the reported rate (requires `cuda` or `metal` feature)

When built with GPU support (`cuda` or `metal` feature), the default mode is **hybrid**: both CPU threads and GPU run concurrently, and the first match from either wins. If no GPU is detected at runtime, the tool falls back to CPU-only with a warning.

**Examples:**
```bash
mc-keygen AB             # find a key starting with AB (hybrid if GPU available)
mc-keygen AB CD EF       # find a key matching ANY of these prefixes
mc-keygen DEAD -t 4      # use 4 threads
mc-keygen AB --json      # machine-readable output
mc-keygen ABCDEF --gpu-only   # GPU only for longer prefixes
mc-keygen AB --cpu-only       # force CPU even when GPU is available
mc-keygen ABCD --stream > keys.txt &   # run for days, collect every match in a file
```

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
| 1 char | 16                | instant   | instant   |
| 2 char | 256               | instant   | instant   |
| 3 char | 4,096             | instant   | instant   |
| 4 char | 65,536            | instant   | instant   |
| 5 char | ~1M               | <1s       | instant   |
| 6 char | ~16M              | ~12s      | <1s       |
| 7 char | ~268M             | ~3.5m     | ~4s       |
| 8 char | ~4.3B             | ~55m      | ~63s      |

¹ CPU: Ryzen 9 7950X3D, 32 threads (~1.3M keys/s). GPU: RTX 4090 (~68M keys/s). See [docs/gpu.md](docs/gpu.md) for details.

## Building

```bash
cargo build --release                    # CPU only
cargo build --release --features cuda    # with NVIDIA GPU support
cargo build --release --features metal   # with Apple Metal GPU support
```

CUDA support requires the NVIDIA CUDA Toolkit. Metal support requires macOS with an Apple Silicon or AMD GPU. See [docs/gpu.md](docs/gpu.md) for details.

### Faster build (PGO)

`./scripts/pgo-build.sh` produces a profile-guided-optimized binary, measured
**~16% faster** on the search loop than a plain `--release` build. It builds
an instrumented binary, records a profile from a short run, then rebuilds.
Run it **on the machine you'll search on** — the profile is specific to the
SIMD path your CPU takes (AVX-512 / AVX2 / scalar) and doesn't transfer across
microarchitectures. Requires `rustup component add llvm-tools-preview`.

### Thread count

By default the search uses one worker per logical CPU on homogeneous SMT
machines (the sibling thread hides field-arithmetic latency, ~+22% on a Xeon
D-1541), but one per **physical** core on hybrid Intel P+E CPUs, where the
P-core hyperthreads contend with the E-cores (~−5% on an i7-13700H at 20 vs 14
threads). Run `mc-keygen bench -m all --machine-label <name>` to find the best
count for your CPU and pass it with `-t` if the default isn't optimal.

## How it works

1. Draw 64 random bytes from the OS CSPRNG
2. Clamp the first 32 bytes to form an Ed25519 scalar
3. Multiply the base point by the scalar to get the public key
4. If the hex starts with the target prefix, return; otherwise repeat

Keys starting with `00` or `FF` are skipped (reserved by MeshCore). On GPU, only the starting scalar per thread is drawn from the CSPRNG — successive candidates come from repeated +8B point addition (see [docs/gpu.md](docs/gpu.md)).

## Sources

- [MeshCore](https://github.com/ripplebiz/MeshCore) — the mesh networking firmware these keys are for
- [MeshCore mc-keygen web tool](https://gessaman.com/mc-keygen/) — reference implementation of the key generation algorithm
- [Ed25519 / RFC 8032](https://datatracker.ietf.org/doc/html/rfc8032) — the signature scheme spec
- [curve25519-dalek](https://github.com/dalek-cryptography/curve25519-dalek) — Rust Ed25519 elliptic curve library
- [ratatui](https://github.com/ratatui/ratatui) — TUI framework for the progress display
