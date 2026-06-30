//! Command-line benchmark for `simd-popcnt`: repeatedly counts the 1 bits in an
//! array and reports throughput in GB/s.
//!
//! ```text
//! cargo run --release --example benchmark [array_bytes] [iters]
//! ```
//!
//! Defaults: 16 KiB array, 10,000,000 iterations.

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

    // Deterministic pseudo-random fill (popcount timing is data-independent).
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

    // Reference count, to verify the benchmark result.
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
        // `black_box` stops the optimizer from hoisting this loop-invariant
        // call out of the loop (which would time nothing).
        total += popcnt(black_box(&data));
    }
    let seconds = start.elapsed().as_secs_f64();
    println!("\rStatus: 100%");

    println!("Seconds: {seconds:.2}");
    let gbs = (bytes as f64 * iters as f64 / 1e9) / seconds;
    println!("{gbs:.1} GB/s");

    // Each call returns `expected`, so the total must be `expected * iters`.
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

/// Report which kernel `popcnt` will use for a `bytes`-byte array.
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
