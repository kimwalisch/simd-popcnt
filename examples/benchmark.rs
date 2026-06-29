//! Command-line benchmark for the `simd-popcnt` crate — a Rust port of
//! `libpopcnt/benchmark.cpp`. It repeatedly counts the 1 bits in an array and
//! reports throughput in GB/s, so you can measure how the crate performs on
//! your CPU for your array size.
//!
//! This is a developer tool, **not** part of the library or its test suite.
//! Files under `examples/` are not compiled when `simd-popcnt` is used as a
//! dependency, and `cargo build` does not build them either — they are built
//! only on demand (`cargo run --example`).
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --example benchmark [array_bytes] [iters]
//! ```
//!
//! Always use `--release`; a debug build measures unoptimized code. For the
//! fastest path, also pass `RUSTFLAGS="-C target-cpu=native"` so the best SIMD
//! kernel is selected at compile time. Defaults: 16 KiB array, 10,000,000 iters.

use simd_popcnt::popcnt;
use std::hint::black_box;
use std::io::{self, Write};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let bytes: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16 * 1024);
    let iters: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000)
        .max(1);

    if cfg!(debug_assertions) {
        eprintln!("warning: this is a debug build; rebuild with --release for meaningful numbers");
    }

    // Fill with deterministic pseudo-random data (xorshift). popcnt timing is
    // data-independent, so a fixed seed just keeps runs reproducible and avoids
    // pulling in a random-number dependency.
    let mut data = vec![0u8; bytes];
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    for b in data.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *b = state as u8;
    }

    println!("Iters: {iters}");
    print_array_size(bytes);
    println!("Algorithm: {}", algorithm(bytes));

    // Independent reference count (one pass) used to verify the benchmark.
    let expected: u64 = data.iter().map(|&b| b.count_ones() as u64).sum();

    let start = Instant::now();
    let mut total = 0u64;
    let step = (iters / 100).max(1);
    let mut next = 0usize;
    for i in 0..iters {
        if i >= next {
            print!("\rStatus: {}%", i * 100 / iters);
            io::stdout().flush().ok();
            next += step;
        }
        // `black_box(&data)` is essential: without it the optimizer would see
        // that `popcnt(&data)` is loop-invariant and hoist it out of the loop
        // (or fold all iterations into one), so the benchmark would time
        // nothing. The C version relies on the popcnt CPUID global + I/O to
        // inhibit this; in Rust we make the barrier explicit.
        total += popcnt(black_box(&data));
    }
    let seconds = start.elapsed().as_secs_f64();
    println!("\rStatus: 100%");

    println!("Seconds: {seconds:.2}");
    let total_bytes = bytes as f64 * iters as f64;
    let gbs = (total_bytes / 1e9) / seconds;
    println!("{gbs:.1} GB/s");

    // Each call returns `expected`, so the sum must be `expected * iters`.
    if total / iters as u64 != expected {
        eprintln!("simd-popcnt verification failed!");
        std::process::exit(1);
    }
}

fn print_array_size(bytes: usize) {
    if bytes < 1024 {
        println!("Array size: {bytes} bytes");
    } else if bytes < 1024 * 1024 {
        println!("Array size: {:.2} KB", bytes as f64 / 1024.0);
    } else {
        println!("Array size: {:.2} MB", bytes as f64 / (1024.0 * 1024.0));
    }
}

/// Report which kernel `popcnt()` will use for an array of `bytes` bytes,
/// mirroring the dispatch in the library (compile-time target features first,
/// then cached runtime detection) and the same size thresholds.
fn algorithm(bytes: usize) -> &'static str {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let avx512 = cfg!(target_feature = "avx512vpopcntdq")
            || (is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vpopcntdq"));
        let avx2 = cfg!(target_feature = "avx2") || is_x86_feature_detected!("avx2");
        let popcnt_hw = cfg!(target_feature = "popcnt") || is_x86_feature_detected!("popcnt");
        if avx512 && bytes >= 40 {
            "AVX512"
        } else if avx2 && bytes >= 512 {
            "AVX2"
        } else if popcnt_hw {
            "POPCNT"
        } else {
            "integer popcount"
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let _ = bytes;
        // NEON is mandatory; SVE is used when statically enabled, or when the
        // build.rs probe enabled SVE intrinsics and the CPU supports them.
        let sve = cfg!(target_feature = "sve")
            || (cfg!(simd_popcnt_have_sve) && std::arch::is_aarch64_feature_detected!("sve"));
        if sve { "ARM SVE" } else { "ARM NEON" }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = bytes;
        "integer popcount"
    }
}
