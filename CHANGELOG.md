# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-07-01

No changes to the library code or public API — documentation and CI/testing
improvements only.

- Added README badges: crates.io version, docs.rs, CI status, and minimum
  supported Rust version.
- Documented that the crate has no dependencies beyond the Rust standard
  library.
- Added a note inviting a bug report if another crate is faster for your use
  case.
- Refreshed the crate-level (`lib.rs`) documentation to match the README: the
  portable `u64::count_ones()` fallback, thread-safety, zero dependencies, and
  a usage example covering `PopcntExt`.
- Added Valgrind memcheck CI jobs (x86-64 and AArch64) that run the test suite
  and benchmark under memcheck to check the SIMD code for memory-safety
  undefined behavior.

## [0.1.0] - 2026-07-01

Initial release.

- `popcnt(&[u8]) -> u64`: counts the 1 bits in a byte slice, dispatching at
  runtime to the fastest instruction set the running CPU supports.
- `PopcntExt` trait: adds a `.popcnt()` method to slices, arrays and `Vec`s of
  every built-in integer type (`u8`–`u128`, `i8`–`i128`, `usize`, `isize`).
- SIMD acceleration: POPCNT / AVX2 / AVX512 on x86 and x86-64, NEON / SVE on
  AArch64, and a portable `u64::count_ones()` fallback on all other targets.

[0.1.1]: https://github.com/kimwalisch/simd-popcnt/releases/tag/v0.1.1
[0.1.0]: https://github.com/kimwalisch/simd-popcnt/releases/tag/v0.1.0
