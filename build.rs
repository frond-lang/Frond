//! build.rs — Frond `@extern("C")` stdlib C compilation.
//!
//! ## Design
//!
//! This script extracts `@extern("C")` functions from `builtin/*/Raw.kz` files
//! via **direct text parsing** — no frond binary dependency. This solves the
//! bootstrap problem ("chicken-and-egg"): previously build.rs needed the frond
//! binary to emit C code, but the binary didn't exist on first build.
//!
//! The text extraction logic parses the fixed `.kz` syntax (`@extern("C")` +
//! `fun sig #{ body }#`), and shares the type-mapping + code-generation code
//! with the AST path via `include!("src/ffi/Gen.rs")`.
//!
//! ## Workflow
//!
//! 1. Scan `.kz` files listed in `EXTERN_FROND_FILES` (containing `@extern("C")` declarations).
//! 2. Text-extract each into `ExternCFunc` records (shared struct from `Gen.rs`).
//! 3. Generate `.c` source for each file via `gen::generate_c_source`.
//! 4. Compile all `.c` files into the `frond_extern` static library using the `cc` crate.
//! 5. Delete intermediate `.c` artifacts from `OUT_DIR`.
//!
//! Symbol resolution: no Rust FFI binding table (`bindings_addr`) is generated
//! anymore. At runtime, frond resolves in-process symbols via dlsym (GetProcAddress)
//! self-lookup (see `platform::ResolveSelfSymbol`). C functions are exported on Windows
//! via the `FROND_EXPORT` macro (`__declspec(dllexport)`); on Linux/macOS they are
//! exported by default and visible to dlsym.

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

use gen::{is_i128_return, is_str_return, type_to_c_params, type_to_c_return, ExternCFunc};

/// List of `.kz` files containing `@extern("C")` declarations.
///
/// `reflect/Raw.kz` is not in this list: its primitives are implemented on the Rust side as
/// `#[no_mangle] extern "C" fn`, so emit-c is not needed. The Raw.kz file itself
/// is loaded directly by Sema (builtin) for type checking.
const EXTERN_FROND_FILES: &[&str] = &[
    "src/stdlib/builtin/io/Raw.kz",
    "src/stdlib/builtin/net/Raw.kz",
    "src/stdlib/builtin/time/Raw.kz",
    "src/stdlib/builtin/cast/Raw.kz",
    "src/stdlib/builtin/str/Raw.kz",
    "src/stdlib/builtin/terminal/Raw.kz",
    "src/stdlib/builtin/os/Raw.kz",
];

fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_extern_c)");

    let out_dir = env::var("OUT_DIR").unwrap();

    // Collect existing .kz files
    let frond_files: Vec<PathBuf> = EXTERN_FROND_FILES
        .iter()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .collect();

    if frond_files.is_empty() {
        return;
    }

    // 1. Text-extract each .kz → ExternCFunc list, then generate .c into OUT_DIR
    let mut c_files: Vec<PathBuf> = Vec::new();

    for frond_file in &frond_files {
        let content = match fs::read_to_string(frond_file) {
            Ok(c) => c,
            Err(e) => {
                println!("cargo:warning=Read failed {}: {}", frond_file.display(), e);
                continue;
            }
        };

        let funcs = match parse_kz_extern_c(&content) {
            Ok(f) => f,
            Err(e) => {
                println!("cargo:warning=Parse failed {}: {}", frond_file.display(), e);
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
                println!("cargo:warning=C gen failed {}: {}", frond_file.display(), e);
                continue;
            }
        };

        let c_name = frond_file_to_c_name(frond_file);
        let c_path = Path::new(&out_dir).join(&c_name);
        if fs::write(&c_path, &c_source).is_err() {
            println!("cargo:warning=Write .c failed: {}", c_path.display());
            continue;
        }
        c_files.push(c_path);

        println!("cargo::rerun-if-changed={}", frond_file.display());
    }

    if c_files.is_empty() {
        return;
    }

    // 2. Compile all .c files with cc::Build into the frond_extern static library.
    //
    // Key point: use cargo_metadata(false) to suppress the rustc-link-lib that cc emits
    // automatically (a plain -l would drop unreferenced symbols). Instead we emit the
    // whole-archive link directives ourselves, forcing every symbol in the library into
    // the binary (otherwise dlsym could not find them — the main binary has no Rust-side
    // extern "C" references, so the linker would not extract these symbols from the
    // static library).
    let mut build = cc::Build::new();
    build.cargo_metadata(false);
    // Suppress the "unused parameter" warning. The flag spelling differs by compiler:
    //   - GCC/Clang: -Wno-unused-parameter
    //   - MSVC (cl.exe): /wd4100  (C4100 = unreferenced formal parameter)
    if build.is_flag_supported("-Wno-unused-parameter").unwrap_or(false) {
        build.flag("-Wno-unused-parameter");
    } else if build.is_flag_supported("/wd4100").unwrap_or(false) {
        build.flag("/wd4100");
    }
    for c_file in &c_files {
        build.file(c_file);
    }
    match build.try_compile("frond_extern") {
        Ok(_) => {
            // Emit link directives manually: whole-archive forces full linkage.
            // Output the library search path (try_compile already placed the .lib in
            // OUT_DIR, but cargo_metadata(false) does not emit link-search automatically).
            println!("cargo::rustc-link-search=native={}", out_dir);
            // Platform-specific whole-archive wrapping.
            let target = env::var("TARGET").unwrap_or_default();
            if target.contains("msvc") {
                // MSVC: /WHOLEARCHIVE:<libname>. libname has no `lib` prefix / `.lib` suffix.
                println!("cargo::rustc-link-arg-bin=frond=/WHOLEARCHIVE:frond_extern.lib");
            } else if target.contains("apple") || target.contains("darwin") {
                // macOS: -force_load <path> (requires the full path; -force_load wraps a single .a).
                let lib_path = Path::new(&out_dir).join("libfrond_extern.a");
                println!("cargo::rustc-link-arg-bin=frond=-force_load");
                println!("cargo::rustc-link-arg-bin=frond={}", lib_path.display());
            } else {
                // Linux / other ELF: -Wl,--whole-archive -lfrond_extern -Wl,--no-whole-archive
                println!("cargo::rustc-link-arg-bin=frond=-Wl,--whole-archive");
                println!("cargo::rustc-link-arg-bin=frond=-lfrond_extern");
                println!("cargo::rustc-link-arg-bin=frond=-Wl,--no-whole-archive");
            }
            println!("cargo::rustc-cfg=has_extern_c");
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

/// Maps a frond file path to a unique .c file name in OUT_DIR.
fn frond_file_to_c_name(frond_file: &Path) -> String {
    let stem = frond_file
        .with_extension("")
        .to_string_lossy()
        .replace('/', "_");
    format!("{}.c", stem)
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
// This parser is intentionally lightweight — it does not need the full Frond
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
    let return_ty = ret_type.to_string();
    let is_str_ret = is_str_return(&return_ty);
    let is_i128_ret = is_i128_return(&return_ty);

    // Map return type. Both `str` and 128-bit returns use the out-parameter
    // pattern (the C function becomes `void` and trailing out-pointers carry
    // the result), so the direct return slot is `void` for both.
    let c_return = if is_str_ret || is_i128_ret {
        "void".to_string()
    } else {
        match type_to_c_return(&return_ty) {
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
    let mut lang_params = Vec::new();
    for (p_name, p_type) in params {
        match type_to_c_params(p_type, p_name) {
            Some(ps) => c_params.extend(ps),
            None => {
                eprintln!(
                    "warning: @extern(\"C\") function '{}': unsupported parameter type '{}' (parameter '{}'), skipping",
                    name, p_type, p_name
                );
                return None;
            }
        }
        lang_params.push(gen::Param {
            name: p_name.clone(),
            type_name: p_type.clone(),
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

    // For i128/u128/f128 returns, append two uint64_t* out parameters (low/high).
    // MSVC x64 cannot return `__int128` by value (no conversion operators), so
    // we use the same out-parameter pattern as `str`. The C body must write
    // `*out_lo` / `*out_hi` instead of using a `return` statement.
    if is_i128_ret {
        c_params.push(gen::CParam {
            name: "out_lo".to_string(),
            c_type: "uint64_t*".to_string(),
        });
        c_params.push(gen::CParam {
            name: "out_hi".to_string(),
            c_type: "uint64_t*".to_string(),
        });
    }

    Some(ExternCFunc {
        name: name.to_string(),
        c_return,
        c_name: format!("frond_extern_{}", name),
        c_params,
        c_body: body.to_string(),
        c_includes: c_includes.to_vec(),
        params: lang_params,
        return_ty,
    })
}
