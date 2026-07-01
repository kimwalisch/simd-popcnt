# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-01

Initial release.

- `popcnt(&[u8]) -> u64`: counts the 1 bits in a byte slice, dispatching at
  runtime to the fastest instruction set the running CPU supports.
- `PopcntExt` trait: adds a `.popcnt()` method to slices, arrays and `Vec`s of
  every built-in integer type (`u8`–`u128`, `i8`–`i128`, `usize`, `isize`).
- SIMD acceleration: POPCNT / AVX2 / AVX512 on x86 and x86-64, NEON / SVE on
  AArch64, and a portable `u64::count_ones()` fallback on all other targets.

[0.1.0]: https://github.com/kimwalisch/simd-popcnt/releases/tag/v0.1.0
