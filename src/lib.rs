// The `gpu` feature is an umbrella implied by `cuda` or `metal`; it should
// never be enabled on its own.
#[cfg(all(feature = "gpu", not(any(feature = "cuda", feature = "metal"))))]
compile_error!("feature `gpu` requires a backend; enable `cuda` or `metal`");

pub mod keygen;
pub mod search;
pub mod simd4;
pub mod types;

#[cfg(feature = "cuda")]
pub mod gpu;

#[cfg(feature = "metal")]
pub mod metal_gpu;
