# simd-popcnt

[![CI](https://github.com/kimwalisch/simd-popcnt/actions/workflows/ci.yml/badge.svg)](https://github.com/kimwalisch/simd-popcnt/actions/workflows/ci.yml)

`simd-popcnt` is a Rust library for counting the number of 1 bits (bit
population count, a.k.a. Hamming weight) in an array as quickly as possible
using specialized CPU instructions: POPCNT, AVX2, AVX512, ARM NEON and ARM SVE.

It is an AI-assisted Rust port of the [libpopcnt C/C++ library](https://github.com/kimwalisch/libpopcnt).

## API

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
simd-popcnt = "0.1"
```

The core function counts the one bits in a byte slice:

```rust
/// Returns the number of set 1-bits in a byte slice
pub fn popcnt(data: &[u8]) -> u64
```

The `PopcntExt` trait adds a `.popcnt()` method to slices, arrays and `Vec`s of
every built-in integer type (`u8`–`u128`, `i8`–`i128`, `usize`, `isize`), so you
can count the bits in any integer array directly, without converting it to bytes
by hand:

```rust
use simd_popcnt::{popcnt, PopcntExt};

// Byte slice — the core function:
assert_eq!(popcnt(&[0xFF, 0x0F]), 12);

// Wider integer types via the `PopcntExt::popcnt` method:
assert_eq!([u64::MAX, 0x0F0F_0F0F_0F0F_0F0F].popcnt(), 96); // [u64; 2]
assert_eq!([0xFFFFu16, 0x00FF, 0x0001].popcnt(), 25);       // [u16; 3]
assert_eq!(vec![1u32, 2, 3, 4].popcnt(), 5);                // Vec<u32>

let data: &[u64] = &[1, 2, 3];
assert_eq!(data.popcnt(), 4);                               // &[u64]

// Text: count the set bits in a string's UTF-8 bytes.
assert_eq!("hello".as_bytes().popcnt(), 21);
```

Population count is independent of byte order, so `.popcnt()` gives the same
result on little- and big-endian targets.

## Minimum supported Rust version

Rust **1.89** (the release that stabilized the AVX512 intrinsics in `std::arch`).
ARM SVE support additionally requires a nightly toolchain.

## CPU architectures

On the following CPU architectures `simd-popcnt` dispatches at runtime to the
fastest instruction set the CPU supports:

| Architecture    | Instruction sets     |
|-----------------|----------------------|
| x86 / x86-64    | POPCNT, AVX2, AVX512 |
| AArch64 (ARM64) | NEON, SVE            |

On every other architecture the count uses `u64::count_ones()`, which the
compiler lowers to the CPU's hardware popcount instruction where the target has
one (WebAssembly `i64.popcnt`, PowerPC `popcntd`, RISC-V `cpop` with Zbb), and to
a portable integer sequence otherwise.

## How it works

On x86 CPUs the supported instruction sets are detected at runtime (once, then
cached) and the fastest available algorithm is selected:

* If the CPU supports AVX512, the AVX512 VPOPCNTDQ algorithm is used.
* Else if the CPU supports AVX2, the AVX2 Harley-Seal algorithm is used.
* Else if the CPU supports POPCNT, the hardware POPCNT algorithm is used.
* For CPUs without POPCNT, a portable integer algorithm is used.

The library works on every architecture: it is portable by default, hardware
acceleration is enabled only when the CPU supports it, and it is thread-safe.

If you compile with `RUSTFLAGS="-C target-cpu=native"` on, say, an x86 CPU with
AVX512 support, all runtime feature checks are removed and the best algorithm is
selected at compile time.

## ARM SVE (Scalable Vector Extension)

ARM SVE is a vector instruction set for ARM CPUs, first widely available in
2020. It supports a variable vector length from 128 up to 2048 bits, so SVE
algorithms can be considerably faster than NEON algorithms, which are limited to
a 128-bit vector length.

On Linux and Windows, `simd-popcnt` detects at runtime whether the CPU supports
SVE and, if so, dispatches to the SVE popcount algorithm; otherwise it
transparently falls back to the portable NEON algorithm.

One Rust-specific caveat: the SVE intrinsics in `std::arch` are still
nightly-only (`feature(stdarch_aarch64_sve)`). A build script probes whether the
active compiler accepts them and enables the SVE code path only when it does. On
stable Rust the SVE path is compiled out and the NEON path is used; everything
else builds on stable.

## Development

Run the test suite with:

```bash
cargo test
```

The crate ships a `benchmark` example for measuring popcount throughput on your
CPU. Build it in release mode; the fastest algorithm is selected automatically
at runtime:

```bash
# Usage: cargo run --release --example benchmark [array_bytes] [iters]
cargo run --release --example benchmark
```

Below is a run on an Intel Core Ultra 5 245K CPU:

```
Iters: 10000000
Array size: 16.00 KB
Algorithm: AVX2
Status: 100%
Seconds: 1.91
85.6 GB/s
```

## Acknowledgments

Some of the algorithms used in ```simd-popcnt``` are described in the paper
[Faster Population Counts using AVX2 Instructions](https://arxiv.org/abs/1611.07612)
by Daniel Lemire, Nathan Kurz and Wojciech Mula (23 Nov 2016). The AVX2 Harley Seal
popcount algorithm used in ```simd-popcnt``` has been copied from Wojciech Muła's
[sse-popcount](https://github.com/WojciechMula/sse-popcount) GitHub repo.
