# GPU Acceleration

## Performance

Measured throughput on real hardware:

| | Throughput | Hardware |
|-|------------|----------|
| **CPU** | ~24M keys/sec | Xeon D-1541, 8C/16T, AVX2 path |
| **CPU** | ~34M keys/sec | 4-core Ice-Lake-class VM, AVX-512 path |
| **GPU** | ~68M keys/sec | NVIDIA RTX 4090 (24GB) |

The GPU speedup comes from running the full Ed25519 pipeline across thousands
of GPU threads in parallel. Note the CPU path has been heavily optimized
since the GPU numbers were taken (SIMD-across-keys, batched inversion, kernel
prefilters — see [benchmarks.md](benchmarks.md)), so the gap on this pairing
is now roughly 2–3× rather than the historical ~50×; the GPU kernel has not
received the same optimization pass.

## Building with CUDA

```bash
cargo build --release --features cuda
```

Requires the NVIDIA CUDA Toolkit to be installed. The CUDA kernel is compiled at runtime via NVRTC, so `nvrtc` and `cuda` shared libraries must be on PATH.

The `cudarc` dependency is configured for CUDA 13.1 by default. To target a different CUDA version, change the `cuda-13010` feature in `Cargo.toml` to match your installation (e.g. `cuda-12060` for CUDA 12.6).

## How GPU mode works

When `--gpu-only` (or hybrid) is selected, the entire keygen pipeline runs on the GPU. A single host thread loops: launch kernel batch, sync, check for match, repeat. The TUI progress display works identically for both CPU and GPU modes.

Each `GpuSearcher` draws a fresh 32-byte `start_scalar` from the OS CSPRNG and advances it by `8 * keys_checked` between batches. Per launch, every thread runs **one** `ge_scalarmult_base` on `start_scalar + 8 * tid` to derive its starting point `A`, then iterates `ITERS_PER_THREAD = 256` cheap point additions of `8B` (using `base[0][7] = 8B` as a `ge_precomp`) — each iteration produces a fresh candidate pubkey from `A` and checks the prefix. To amortize the expensive `fe_invert` needed for point compression, the kernel uses **Montgomery batch inversion** with `B = 16`: one inversion per 16 points instead of one per point. On every prefix hit the kernel writes back the scalar and pubkey, and the host runs a defensive `scalar·B == pubkey` check via curve25519-dalek before accepting the match.

The Ed25519 field arithmetic uses the ref10 implementation (radix-2^25.5, int32[10] limbs) from [solana-perf-libs](https://github.com/solana-labs/solana-perf-libs), which is a CUDA adaptation of the [SUPERCOP](https://bench.cr.yp.to/supercop.html) reference code. Scalar multiplication uses windowed lookup tables with 32 precomputed basepoint multiples.

## Verification

Use `--verify` to cross-check GPU-generated keys against the CPU implementation. This is useful for validating correctness after modifying the CUDA kernel.

Use `--benchmark <SECS>` to run a separate `vanity_count_matches` kernel that never exits early on a hit. It atomically tallies every match over the given duration and compares the observed count against the expected count implied by the reported keys/sec rate — a sanity check that the reported throughput isn't inflated by mid-launch early exits. Works on both the CUDA and Metal backends.

## Sources

- [solana-perf-libs](https://github.com/solana-labs/solana-perf-libs) — CUDA Ed25519 field arithmetic (ref10)
- [cudarc](https://github.com/coreylowman/cudarc) — safe Rust bindings for the CUDA driver API
