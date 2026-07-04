# Benchmarks

The authoritative benchmark is the built-in harness, which measures the real
search hot loop and validates the reported rate with a Poisson
observed-vs-expected match count:

```bash
mc-keygen bench -m all -d 15 --machine-label "my machine"
```

It reports single-thread (`cpu1`), physical-cores (`cpuP`) and all-logical
(`cpuN`) rates and emits one JSONL record per mode.

## Measured rates (current code)

| Machine | Path | 1 thread | All cores |
|---------|------|----------|-----------|
| Xeon D-1541 (8C/16T, 2.1 GHz) | AVX2 | ~3.0 MH/s | ~24 MH/s (16T) |
| 4-core Ice-Lake-class cloud VM (2.8 GHz) | AVX-512 | ~8.5–9.4 MH/s | ~34 MH/s |
| RTX 4090 (GPU path) | CUDA | — | ~68 MH/s |

Rates depend on the SIMD path the CPU takes (AVX-512 → AVX2 → scalar, chosen
at runtime). For historical context: the original implementation (one full
scalar multiplication per candidate, no SIMD, no batching) measured ~80K
keys/s per thread on a Ryzen 9 7950X3D — the +8B chain, Montgomery batch
inversion, SIMD-across-keys and kernel prefilters multiplied that by roughly
two orders of magnitude per thread.

## Criterion microbenches

`cargo bench` runs two microbenches (`prefix_match`, `keygen_search`) that
isolate the matcher and the keygen pipeline. They are useful for relative
comparisons when touching those code paths; the end-to-end numbers above are
what users actually see.

Multi-pattern matching adds no measurable overhead — searching for N patterns
at once costs the same as one, so batch your wishes into a single run.
