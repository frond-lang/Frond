//! build.rs — Kuzo `@extern("C")` auto-compile + FFI generation.
//!
//! ## Design
//!
//! This script extracts `@extern("C")` functions from `builtin/*/Raw.kz` files
//! via **direct text parsing** — no kuzo binary dependency. This solves the
//! bootstrap problem ("chicken-and-egg"): previously build.rs needed the kuzo
//! binary to emit C code, but the binary didn't exist on first build.
//!
//! The text extraction logic parses the fixed `.kz` syntax (`@extern("C")` +
//! `fun sig #{ body }#`), and shares the type-mapping + code-generation code
//! with the AST path via `include!("src/ffi/Gen.rs")`.
//!
//! ## Workflow
//!
//! 1. Scan `.kz` files listed in `EXTERN_KUZO_FILES` (containing `@extern("C")` declarations).
//! 2. Text-extract each into `ExternCFunc` records (shared struct from `Gen.rs`).
//! 3. Generate `.c` source for each file via `gen::generate_c_source`.
//! 4. Generate Rust FFI code (concatenated) via `gen::generate_rust_ffi`.
//! 5. Compile all `.c` files into the `kuzo_extern` static library using the `cc` crate.
//! 6. Delete intermediate `.c` artifacts from `OUT_DIR`.
//! 7. The generated FFI code is written to `$OUT_DIR/ffi_generated.rs`, `include!`d by `Ffi.rs`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

// Include the shared type-mapping + code-generation module.
// Same file is `#[path]`-included by `src/ffi/ExternC.rs`, ensuring AST path and
// text path produce identical output.
#[allow(dead_code)]
mod gen {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/Gen.rs"));
}

use gen::{is_str_return, kuzo_type_to_c_params, kuzo_type_to_c_return, ExternCFunc};

/// List of `.kz` files containing `@extern("C")` declarations.
///
/// `reflect/Raw.kz` is not in this list: its primitives are implemented on the Rust side as
/// `#[no_mangle] extern "C" fn`, so emit-c is not needed. The Raw.kz file itself
/// is loaded directly by Sema (builtin) for type checking.
const EXTERN_KUZO_FILES: &[&str] = &[
    "src/stdlib/builtin/io/Raw.kz",
    "src/stdlib/builtin/net/Raw.kz",
    "src/stdlib/builtin/time/Raw.kz",
    "src/stdlib/builtin/cast/Raw.kz",
    "src/stdlib/builtin/str/Raw.kz",
    "src/stdlib/builtin/terminal/Raw.kz",
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

    if kuzo_files.is_empty() {
        fs::write(&ffi_path, empty_ffi_module()).unwrap();
        println!("cargo::rerun-if-changed={}", ffi_path.display());
        return;
    }

    // 1. Text-extract each .kz → ExternCFunc list, then generate .c into OUT_DIR
    let mut all_funcs: Vec<ExternCFunc> = Vec::new();
    let mut c_files: Vec<PathBuf> = Vec::new();
    let mut any_error = false;

    for kuzo_file in &kuzo_files {
        let content = match fs::read_to_string(kuzo_file) {
            Ok(c) => c,
            Err(e) => {
                println!("cargo:warning=Read failed {}: {}", kuzo_file.display(), e);
                any_error = true;
                continue;
            }
        };

        let funcs = match parse_kz_extern_c(&content) {
            Ok(f) => f,
            Err(e) => {
                println!("cargo:warning=Parse failed {}: {}", kuzo_file.display(), e);
                any_error = true;
                continue;
            }
        };

        if funcs.is_empty() {
            continue;
        }

        // Generate .c file
        let c_source = match gen::generate_c_source(&funcs) {
            Ok(s) => s,
            Err(e) => {
                println!("cargo:warning=C gen failed {}: {}", kuzo_file.display(), e);
                any_error = true;
                continue;
            }
        };

        let c_name = kuzo_file_to_c_name(kuzo_file);
        let c_path = Path::new(&out_dir).join(&c_name);
        if fs::write(&c_path, &c_source).is_err() {
            println!("cargo:warning=Write .c failed: {}", c_path.display());
            any_error = true;
            continue;
        }
        c_files.push(c_path);

        // Accumulate funcs for FFI generation
        all_funcs.extend(funcs);

        println!("cargo::rerun-if-changed={}", kuzo_file.display());
    }

    // 2. Generate FFI code
    let ffi_code = if !any_error && !all_funcs.is_empty() {
        match gen::generate_rust_ffi(&all_funcs) {
            Ok(s) => Some(s),
            Err(e) => {
                println!("cargo:warning=FFI gen failed: {}", e);
                None
            }
        }
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

    if c_files.is_empty() {
        return;
    }

    // 3. Compile all .c files with cc::Build
    let mut build = cc::Build::new();
    build.flag("-Wno-unused-parameter");
    for c_file in &c_files {
        build.file(c_file);
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
            // After successful compilation, delete the .c intermediate artifacts from OUT_DIR
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

// ============ Text extraction ============
//
// Parses .kz source text and extracts @extern("C") function declarations.
// The .kz syntax for extern C is a fixed text pattern:
//
//   @c_include("header.h")      ← optional, zero or more
//   @extern("C")
//   fun name(params): ret_type #{
//       <C body>
//   }#
//
// This parser is intentionally lightweight — it does not need the full Kuzo
// parser. It only recognizes the subset of syntax that @extern("C") functions
// use. Unknown types produce a warning and the function is skipped (same
// behavior as the AST path in ExternC.rs).

/// Parse a .kz source string and extract all @extern("C") functions.
fn parse_kz_extern_c(source: &str) -> Result<Vec<ExternCFunc>, String> {
    let mut funcs = Vec::new();

    // Tokenize into lines while tracking byte positions for body extraction.
    let lines: Vec<&str> = source.lines().collect();
    let mut i = 0;

    // Accumulated attributes (reset after each function).
    let mut pending_includes: Vec<String> = Vec::new();
    let mut pending_extern_c = false;

    while i < lines.len() {
        let line = lines[i].trim();

        // Collect @c_include attributes
        if let Some(inc) = parse_c_include_attr(line) {
            if !pending_includes.contains(&inc) {
                pending_includes.push(inc);
            }
            i += 1;
            continue;
        }

        // Detect @extern("C")
        if line.starts_with("@extern") {
            if line.contains("\"C\"") {
                pending_extern_c = true;
            } else if line.contains("\"c\"") {
                eprintln!("warning: @extern(\"c\") must use uppercase 'C'; this attribute will be ignored");
            }
            i += 1;
            continue;
        }

        // Detect `fun` keyword — process if we have a pending @extern("C")
        if pending_extern_c && line.starts_with("fun ") {
            // Find the full function signature. It may span multiple lines but
            // in practice all Raw.kz files keep it on one line. We search for
            // `#{` to find the body start.
            //
            // Strategy: accumulate text from the `fun` line until we find `#{`,
            // then accumulate body until we find the matching `}#`.

            // Find `#{` starting from the current line
            let mut sig_and_body_start = i;
            let mut found_brace_hash = false;

            // Search forward for `#{`
            while sig_and_body_start < lines.len() {
                if lines[sig_and_body_start].contains("#{") {
                    found_brace_hash = true;
                    break;
                }
                sig_and_body_start += 1;
            }

            if !found_brace_hash {
                eprintln!(
                    "warning: @extern(\"C\") function near line {}: missing #{{ body, skipping",
                    i + 1
                );
                pending_includes.clear();
                pending_extern_c = false;
                i += 1;
                continue;
            }

            // Extract the signature: from the `fun` line to the `#{` line
            let sig_text: String = lines[i..=sig_and_body_start]
                .iter()
                .map(|l| {
                    // Stop at `#{` — take everything before it
                    if let Some(pos) = l.find("#{") {
                        &l[..pos]
                    } else {
                        l
                    }
                })
                .collect::<Vec<&str>>()
                .join(" ");

            // Extract the body: from after `#{` on sig_and_body_start line
            // until we find `}#`
            let mut body = String::new();
            let mut body_end_line = sig_and_body_start;
            let mut found_close = false;

            // Part of the start line after `#{`
            let first_body_line = &lines[sig_and_body_start];
            let after_brace = &first_body_line[first_body_line.find("#{").unwrap() + 2..];
            if after_brace.trim().contains("}#") {
                // Body is on the same line as #{ ... }#
                let close_pos = after_brace.find("}#").unwrap();
                body.push_str(&after_brace[..close_pos]);
                found_close = true;
            } else {
                body.push_str(after_brace);
                body.push('\n');

                // Search subsequent lines for `}#`
                let mut j = sig_and_body_start + 1;
                while j < lines.len() {
                    if let Some(close_pos) = lines[j].find("}#") {
                        body.push_str(&lines[j][..close_pos]);
                        body_end_line = j;
                        found_close = true;
                        break;
                    } else {
                        body.push_str(lines[j]);
                        body.push('\n');
                    }
                    j += 1;
                }
            }

            if !found_close {
                eprintln!(
                    "warning: @extern(\"C\") function near line {}: missing }}# body terminator, skipping",
                    i + 1
                );
                pending_includes.clear();
                pending_extern_c = false;
                i = body_end_line + 1;
                continue;
            }

            // Parse the signature text: `fun name(params): ret_type`
            match parse_fun_signature(&sig_text) {
                Ok((name, params, ret_type)) => {
                    let func = build_extern_c_func(&name, &params, &ret_type, &body, &pending_includes);
                    if let Some(f) = func {
                        funcs.push(f);
                    }
                }
                Err(e) => {
                    eprintln!("warning: @extern(\"C\") near line {}: {}", i + 1, e);
                }
            }

            // Reset attributes
            pending_includes.clear();
            pending_extern_c = false;
            i = body_end_line + 1;
            continue;
        }

        // Non-attribute, non-fun line: reset pending attributes only if it's
        // not blank or a comment (those don't break attribute grouping).
        if !line.is_empty() && !line.starts_with("//") {
            // A code line that's not an attribute or fun — clear pending state
            // unless it's another @ attribute.
            if !line.starts_with('@') {
                pending_includes.clear();
                pending_extern_c = false;
            }
        }
        i += 1;
    }

    Ok(funcs)
}

/// Parse a `@c_include("header.h")` attribute line, returning the header name.
fn parse_c_include_attr(line: &str) -> Option<String> {
    let line = line.trim();
    if !line.starts_with("@c_include") {
        return None;
    }
    // Extract the string inside the parentheses
    let start = line.find('(')?;
    let end = line.rfind(')')?;
    if start >= end {
        return None;
    }
    let inner = &line[start + 1..end];
    // Strip surrounding quotes
    let inner = inner.trim();
    if inner.len() >= 2 && inner.starts_with('"') && inner.ends_with('"') {
        Some(inner[1..inner.len() - 1].to_string())
    } else {
        None
    }
}

/// Parse `fun name(param: type, ...): ret_type` signature text.
///
/// Returns (name, params, return_type) where params is a Vec of (name, type).
fn parse_fun_signature(sig: &str) -> Result<(String, Vec<(String, String)>, String), String> {
    let sig = sig.trim();

    // Strip leading `fun `
    let sig = sig.strip_prefix("fun ").ok_or("not a fun declaration")?.trim();

    // Find the function name (up to `(`)
    let paren_open = sig.find('(').ok_or("missing '(' in fun signature")?;
    let name = sig[..paren_open].trim().to_string();
    if name.is_empty() {
        return Err("empty function name".to_string());
    }

    // Find matching `)`
    let paren_close = sig.rfind(')').ok_or("missing ')' in fun signature")?;
    if paren_close <= paren_open {
        return Err("malformed parentheses in fun signature".to_string());
    }

    let params_str = &sig[paren_open + 1..paren_close];

    // Parse parameters
    let mut params = Vec::new();
    for param in split_params(params_str) {
        let param = param.trim();
        if param.is_empty() {
            continue;
        }
        let colon = param.find(':').ok_or_else(|| format!("parameter '{}' missing type annotation", param))?;
        let p_name = param[..colon].trim().to_string();
        let p_type = param[colon + 1..].trim().to_string();
        params.push((p_name, p_type));
    }

    // Parse return type (after `)`)
    let after_paren = sig[paren_close + 1..].trim();
    let ret_type = if let Some(rest) = after_paren.strip_prefix(':') {
        rest.trim().to_string()
    } else {
        "void".to_string()
    };

    Ok((name, params, ret_type))
}

/// Split parameter list by commas, respecting nested brackets (for `u8[]` etc.).
fn split_params(params: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    for ch in params.chars() {
        match ch {
            '[' | '(' | '<' => {
                depth += 1;
                current.push(ch);
            }
            ']' | ')' | '>' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                result.push(current.clone());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        result.push(current);
    }
    result
}

/// Build an ExternCFunc from parsed text fields, applying type mapping.
///
/// Returns None (with a warning printed) if any type is unsupported.
fn build_extern_c_func(
    name: &str,
    params: &[(String, String)],
    ret_type: &str,
    body: &str,
    c_includes: &[String],
) -> Option<ExternCFunc> {
    let kuzo_return = ret_type.to_string();
    let is_str_ret = is_str_return(&kuzo_return);

    // Map return type
    let c_return = if is_str_ret {
        "void".to_string()
    } else {
        match kuzo_type_to_c_return(&kuzo_return) {
            Some(c) => c.to_string(),
            None => {
                eprintln!(
                    "warning: @extern(\"C\") function '{}': unsupported return type '{}', skipping",
                    name, ret_type
                );
                return None;
            }
        }
    };

    // Map parameters
    let mut c_params = Vec::new();
    let mut kuzo_params = Vec::new();
    for (p_name, p_type) in params {
        match kuzo_type_to_c_params(p_type, p_name) {
            Some(ps) => c_params.extend(ps),
            None => {
                eprintln!(
                    "warning: @extern(\"C\") function '{}': unsupported parameter type '{}' (parameter '{}'), skipping",
                    name, p_type, p_name
                );
                return None;
            }
        }
        kuzo_params.push(gen::KuzoParam {
            name: p_name.clone(),
            kuzo_type: p_type.clone(),
        });
    }

    // For `str` returns, append the out parameters
    if is_str_ret {
        c_params.push(gen::CParam {
            name: "out_data".to_string(),
            c_type: "const char**".to_string(),
        });
        c_params.push(gen::CParam {
            name: "out_len".to_string(),
            c_type: "size_t*".to_string(),
        });
    }

    Some(ExternCFunc {
        kuzo_name: name.to_string(),
        c_return,
        c_name: format!("kuzo_extern_{}", name),
        c_params,
        c_body: body.to_string(),
        c_includes: c_includes.to_vec(),
        kuzo_params,
        kuzo_return,
    })
}
