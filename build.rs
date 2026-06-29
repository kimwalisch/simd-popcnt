//! Build script: compile-time probe for ARM SVE intrinsic support.
//!
//! ARM SVE intrinsics in Rust's stdarch are nightly-gated under
//! `#![feature(stdarch_aarch64_sve)]`. Rather than forcing a hard `nightly`
//! Cargo feature, we probe whether the current `rustc` actually accepts SVE
//! intrinsics and emit `cargo:rustc-cfg=libpopcnt_have_sve` only when the
//! probe succeeds. On stable Rust (or nightly before SVE lands) the probe
//! fails silently and all SVE code is compiled out. This is the Rust
//! equivalent of autoconf's `AC_COMPILE_IFELSE`.

use std::{env, fs, process::Command};

fn main() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // Suppress the "unexpected cfg condition" lint (Rust 1.80+).
    println!("cargo:rustc-check-cfg=cfg(libpopcnt_have_sve)");

    // The probe only needs to run on aarch64; nothing else can use SVE.
    if target_arch == "aarch64" && probe_sve_intrinsics() {
        println!("cargo:rustc-cfg=libpopcnt_have_sve");
    }
}

fn probe_sve_intrinsics() -> bool {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let target = match env::var("TARGET") {
        Ok(t) => t,
        Err(_) => return false,
    };
    let out = match env::var("OUT_DIR") {
        Ok(o) => o,
        Err(_) => return false,
    };

    let src = format!("{out}/probe_sve.rs");
    let meta = format!("{out}/libprobe_sve.rmeta");

    let probe = r#"
#![no_std]
#![feature(stdarch_aarch64_sve)]
#[cfg(target_arch = "aarch64")]
pub fn probe() {
    unsafe { let _ = core::arch::aarch64::svptrue_b8(); }
}
"#;

    if fs::write(&src, probe).is_err() {
        return false;
    }

    Command::new(rustc)
        .args([
            "--edition=2024",
            "--crate-type=lib",
            "--emit=metadata",
            "--target",
            &target,
            "-o",
            &meta,
            &src,
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
