//! build.rs — Kuzo @extern("C") auto-compile + FFI generation integration
//!
//! Workflow:
//! 1. Scan the .kz files listed in EXTERN_KUZO_FILES (containing @extern("C") declarations).
//! 2. If the kuzo binary is available:
//!    a. For each .kz, invoke `kuzo emit-c` to emit a .c file into OUT_DIR (without polluting the source directory).
//!    b. Concatenate all .kz contents and invoke `kuzo emit-ffi -` via stdin to generate Rust FFI code.
//! 3. Compile all .c files into the static library kuzo_extern using the cc crate.
//! 4. After successful compilation, delete the .c intermediate artifacts from OUT_DIR (intermediates are not retained).
//! 5. The generated FFI code is written to $OUT_DIR/ffi_generated.rs, which is include!'d by Ffi.rs.

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// List of .kz files containing @extern("C") declarations.
///
/// reflect/Raw.kz is not in this list: its primitives are implemented on the Rust side as
/// `#[no_mangle] extern "C" fn`, so emit-c is not needed to extract a C body. The Raw.kz file
/// itself is loaded directly by Sema (builtin) for type checking.
const EXTERN_KUZO_FILES: &[&str] = &[
    "src/stdlib/builtin/io/Raw.kz",
    "src/stdlib/builtin/net/Raw.kz",
    "src/stdlib/builtin/time/Raw.kz",
    "src/stdlib/builtin/cast/Raw.kz",
    "src/stdlib/builtin/str/Raw.kz",
];

fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_extern_c)");

    let out_dir = env::var("OUT_DIR").unwrap();
    let ffi_path = Path::new(&out_dir).join("ffi_generated.rs");

    // Collect existing .kz files
    let kuzo_files: Vec<PathBuf> = EXTERN_KUZO_FILES
        .iter()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .collect();

    // 1. Extract Raw.c into OUT_DIR
    if !kuzo_files.is_empty() {
        try_auto_extract_c(&kuzo_files, &out_dir);
    }

    // 2. Generate FFI code
    let ffi_code = if !kuzo_files.is_empty() {
        try_generate_ffi(&kuzo_files)
    } else {
        None
    };
    let ffi_ok = ffi_code.is_some();
    fs::write(
        &ffi_path,
        ffi_code.unwrap_or_else(|| empty_ffi_module().to_string()),
    )
    .unwrap();
    println!("cargo::rerun-if-changed={}", ffi_path.display());

    // 3. Collect .c files (Raw.c in OUT_DIR)
    let mut c_files: Vec<PathBuf> = Vec::new();
    for kuzo_file in &kuzo_files {
        let c_name = kuzo_file_to_c_name(kuzo_file);
        let c_path = Path::new(&out_dir).join(&c_name);
        if c_path.exists() {
            c_files.push(c_path);
        }
    }

    if c_files.is_empty() {
        if !kuzo_files.is_empty() && !find_kuzo_bin().exists() {
            println!(
                "cargo:warning=Found @extern(\"C\") .kz but kuzo binary unavailable, FFI code not generated"
            );
            println!("cargo:warning=Please run cargo build again to generate automatically");
        }
        return;
    }

    // 4. Compile all .c files with cc::Build
    let mut build = cc::Build::new();
    build.flag("-Wno-unused-parameter");
    for c_file in &c_files {
        build.file(c_file);
    }
    for kuzo_file in &kuzo_files {
        println!("cargo::rerun-if-changed={}", kuzo_file.display());
    }
    match build.try_compile("kuzo_extern") {
        Ok(_) => {
            if ffi_ok {
                println!("cargo::rustc-cfg=has_extern_c");
            } else {
                println!(
                    "cargo:warning=C compilation succeeded but FFI generation failed, skipping has_extern_c cfg (wrapper module empty)"
                );
            }
            // After successful compilation, delete the .c intermediate artifacts from OUT_DIR (not retained)
            for c_file in &c_files {
                let _ = fs::remove_file(c_file);
            }
        }
        Err(e) => {
            println!("cargo:warning=C compilation failed, skipping has_extern_c cfg: {}", e);
        }
    }
}

/// Maps a kuzo file path to a unique .c file name in OUT_DIR.
fn kuzo_file_to_c_name(kuzo_file: &Path) -> String {
    let stem = kuzo_file
        .with_extension("")
        .to_string_lossy()
        .replace('/', "_");
    format!("{}.c", stem)
}

/// Empty FFI module (used when there are no @extern("C") functions).
fn empty_ffi_module() -> &'static str {
    r#"// Auto-generated: no @extern("C") functions
#[cfg(has_extern_c)]
pub mod bindings {
    extern "C" {}
}

pub mod wrapper {}
"#
}

/// Locates the built kuzo binary.
fn find_kuzo_bin() -> PathBuf {
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    PathBuf::from(env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string()))
        .join(profile)
        .join("kuzo")
}

/// Attempts to invoke `kuzo emit-c file.kz` to extract each .kz into OUT_DIR/xxx.c.
fn try_auto_extract_c(kuzo_files: &[PathBuf], out_dir: &str) {
    let kuzo_bin = find_kuzo_bin();
    if !kuzo_bin.exists() {
        return;
    }

    for kuzo_file in kuzo_files {
        let c_name = kuzo_file_to_c_name(kuzo_file);
        let c_path = Path::new(out_dir).join(&c_name);
        let output = Command::new(&kuzo_bin)
            .arg("debug")
            .arg("--stage")
            .arg("emit-c")
            .arg(kuzo_file)
            .output();

        match output {
            Ok(result) if result.status.success() => {
                if fs::write(&c_path, &result.stdout).is_ok() {
                    println!(
                        "cargo:warning=Extracted C: {} → {}",
                        kuzo_file.display(),
                        c_path.display()
                    );
                }
            }
            _ => {
                println!("cargo:warning=C extraction failed: {}", kuzo_file.display());
            }
        }
    }
}

/// Attempts to invoke `kuzo emit-ffi -` to generate FFI code (concatenating all .kz via stdin).
fn try_generate_ffi(kuzo_files: &[PathBuf]) -> Option<String> {
    let kuzo_bin = find_kuzo_bin();
    if !kuzo_bin.exists() {
        return None;
    }

    let mut combined = String::new();
    for kuzo_file in kuzo_files {
        match fs::read_to_string(kuzo_file) {
            Ok(content) => {
                combined.push_str(&content);
                combined.push('\n');
            }
            Err(_) => {
                println!("cargo:warning=Read failed: {}", kuzo_file.display());
                return None;
            }
        }
    }

    let mut child = match Command::new(&kuzo_bin)
        .arg("debug")
        .arg("--stage")
        .arg("emit-ffi")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => {
            println!("cargo:warning=Failed to start kuzo debug --stage emit-ffi");
            return None;
        }
    };

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(combined.as_bytes());
    }

    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(_) => {
            println!("cargo:warning=kuzo debug --stage emit-ffi execution failed");
            return None;
        }
    };

    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        println!("cargo:warning=kuzo debug --stage emit-ffi returned non-zero status");
        None
    }
}
