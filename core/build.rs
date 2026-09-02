//! build.rs — Frond `@extern("C")` stdlib C compilation.
//!
//! ## Design
//!
//! This script extracts `@extern("C")` functions from `builtin/*/Raw.frond` files
//! via **direct text parsing** — no frond binary dependency. This solves the
//! bootstrap problem ("chicken-and-egg"): previously build.rs needed the frond
//! binary to emit C code, but the binary didn't exist on first build.
//!
//! The text extraction logic parses the fixed `.frond` syntax (`@extern("C")` +
//! `fun sig #{ body }#`), and shares the type-mapping + code-generation code
//! with the AST path via `include!("src/ffi/Gen.rs")`.
//!
//! ## Workflow
//!
//! 1. Scan `.frond` files listed in `EXTERN_FROND_FILES` (containing `@extern("C")` declarations).
//! 2. Text-extract each into `ExternCFunc` records (shared struct from `Gen.rs`).
//! 3. Generate `.c` source for each file via `gen::generate_c_source`.
//! 4. Generate a symbol registry `.c` (function-pointer table) + matching Rust
//!    bindings in OUT_DIR.
//! 5. Compile all `.c` files into the `frond_extern` static library using the `cc` crate.
//! 6. Delete intermediate `.c` artifacts from `OUT_DIR`.
//!
//! ## Symbol resolution
//!
//! The Rust side references `frond_extern_registry` / `frond_extern_registry_names`
//! (generated in OUT_DIR, included by `src/ffi/Symbols.rs`). The table references
//! every primitive, which keeps them alive through linker dead-stripping —
//! macOS rustc links executables with `-Wl,-dead_strip`, Linux with `--gc-sections`,
//! MSVC release with `/OPT:REF`; a whole-archive link alone does NOT protect
//! unreferenced symbols from any of these. At runtime `ffi::Symbols` resolves
//! names by scanning the table; `dlsym`/`GetProcAddress` self-lookup remains as
//! a fallback (see `platform::ResolveSelfSymbol`). C functions are exported on
//! Windows via the `FROND_EXPORT` macro (`__declspec(dllexport)`); on Linux/macOS
//! they are exported by default.

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

/// List of `.frond` files containing `@extern("C")` declarations.
///
/// `reflect/Raw.frond` is not in this list: its primitives are implemented on the Rust side as
/// `#[no_mangle] extern "C" fn`, so emit-c is not needed. The Raw.frond file itself
/// is loaded directly by Sema (builtin) for type checking.
const EXTERN_FROND_FILES: &[&str] = &[
    "../std/builtin/io/Raw.frond",
    "../std/builtin/net/Raw.frond",
    "../std/builtin/time/Raw.frond",
    "../std/builtin/cast/Raw.frond",
    "../std/builtin/str/Raw.frond",
    "../std/builtin/os/Raw.frond",
    "../std/builtin/rand/Raw.frond",
    "../std/builtin/crypto/Raw.frond",
    "../std/builtin/mem/Raw.frond",
    "../std/builtin/sort/Raw.frond",
    "../std/builtin/encoding/Raw.frond",
];


// =========================================================================
// Order-preserving FFI trampoline table (generated Rust)
// =========================================================================

/// Maximum declared-argument count covered by the generated trampoline table.
const FFI_INVOKE_MAX_ARGS: usize = 12;
/// Argument counts up to this bound get FULL class-pattern enumeration; above
/// it only the all-integer shape is generated (see the coverage note in
/// `write_ffi_invoke_table`).
const FFI_INVOKE_FULL_ENUM_ARGS: usize = 8;

/// Renders one match arm body: a typed cast + call pulling arguments from
/// `ints`/`floats` in DECLARED order (bit i of `classes` = arg i class:
/// 0 = integer/pointer, 1 = float). `ret` selects the return form.
fn render_invoke_arm(classes: u32, nargs: usize, ret: &str) -> String {
    let mut params = Vec::with_capacity(nargs);
    let mut args = Vec::with_capacity(nargs);
    let (mut ii, mut ff) = (0usize, 0usize);
    for i in 0..nargs {
        if (classes >> i) & 1 == 1 {
            params.push("f64".to_string());
            args.push(format!("floats[{ff}]"));
            ff += 1;
        } else {
            params.push("u64".to_string());
            args.push(format!("ints[{ii}]"));
            ii += 1;
        }
    }
    let plist = params.join(", ");
    let alist = args.join(", ");
    if ret == "int" {
        format!(
            "        p => {{ let f: extern \"C\" fn({plist}) -> u64 = core::mem::transmute(fn_ptr); f({alist}) }}"
        ,)
    } else {
        format!(
            "        p => {{ let f: extern \"C\" fn({plist}) -> f64 = core::mem::transmute(fn_ptr); f({alist}).to_bits() }}"
        ,)
    }
}

/// Write the generated trampoline table into OUT_DIR. The table is keyed by
/// `pattern = (1 << nargs) | classes` (the sentinel bit removes the leading-
/// zero ambiguity), so the callee sees arguments in DECLARED order — the old
/// (u64 x I, f64 x F) split reordered int/float arguments, which SysV
/// tolerates (per-class register files) but Win64 (positional register
/// assignment) does not: an interleaved (double, i64) signature received both
/// values from the wrong registers.
fn write_ffi_invoke_table(out_dir: &str) {
    let mut src = String::with_capacity(1 << 21);
    src.push_str("// Auto-generated by build.rs — order-preserving FFI trampoline table. DO NOT EDIT.
");
    
    src.push_str("/// Maximum declared-argument count covered by the table.
");
    src.push_str("pub const MAX_FFI_ARGS: usize = ");
    src.push_str(&FFI_INVOKE_MAX_ARGS.to_string());
    src.push_str(";\n\n/// Argument counts up to this bound have every mixed int/float pattern generated;
/// above it only the all-integer shape exists (dispatch rejects mixed shapes).
pub const MAX_FFI_FULL_ENUM_ARGS: usize = ");
    src.push_str(&FFI_INVOKE_FULL_ENUM_ARGS.to_string());
    src.push_str(concat!(";", "\n", "\n"));
    for ret in ["int", "float"] {
        let (name, ret_ty, doc) = if ret == "int" {
            (
                "ffi_invoke_ret_int",
                "u64",
                "integer/pointer return form (raw RAX/x0)",
            )
        } else {
            (
                "ffi_invoke_ret_float",
                "u64",
                "float return form (f64::to_bits of XMM0/v0)",
            )
        };
        src.push_str(&format!(
            "/// Order-preserving dynamic call, {doc}.
/// `pattern = (1 << nargs) | classes`; bit i of `classes` = arg i class (0=int, 1=float).
#[inline(never)]
pub unsafe fn {name}(pattern: u32, fn_ptr: *mut core::ffi::c_void, ints: &[u64], floats: &[f64]) -> u64 {{
    match pattern {{
"
        ));
        // Coverage policy: full class-pattern enumeration for nargs <= 8
        // (511 patterns per return form — every mixed int/float shape up to
        // the x64/aarch64 register files); for nargs 9..=MAX only the
        // all-integer shape is generated (wide real-world signatures — e.g.
        // __dir_list_into's 9 slots — are pointer/length-only). A mixed
        // pattern beyond 8 args is rejected loudly by `dispatch` instead of
        // silently regrouping classes (the old table's Win64 hazard). This
        // keeps the generated match at ~1k arms; full enumeration to 12 args
        // was 8k+ arms and multiplied rustc MIR time beyond usability.
        for nargs in 0..=FFI_INVOKE_MAX_ARGS {
            let class_iter: Vec<u32> = if nargs <= FFI_INVOKE_FULL_ENUM_ARGS {
                (0u32..(1u32 << nargs)).collect()
            } else {
                vec![0u32]
            };
            for classes in class_iter {
                let pattern = (1u32 << nargs) | classes;
                src.push_str(&format!("        {pattern} => "));
                // render body without the duplicate `p =>` prefix
                let body = render_invoke_arm(classes, nargs, ret);
                let body = body.trim_start().replacen("p => ", "", 1);
                src.push_str(&body);
                src.push(char::from_u32(10).unwrap());
            }
        }
        src.push_str("        _ => unreachable!(\"ffi trampoline table: pattern {pattern:#x} exceeds coverage (MAX_FFI_ARGS)\", pattern = pattern),
    }
}

");
        let _ = ret_ty;
    }
    if let Err(e) = fs::write(Path::new(out_dir).join("ffi_invoke_table.rs"), src) {
        panic!("failed to write ffi_invoke_table.rs: {}", e);
    }
}

fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_extern_c)");

    // Static-musl link order: rustc places its own -lc BEFORE the cc-built
    // frond_extern archive; a static archive only satisfies symbols for
    // objects that precede it on the linker line, so the C kernels' libc
    // references (access/strtod/localtime_r/posix_spawn...) went unresolved.
    // Appending one more -lc at the very end closes the window.
    if let Ok(target) = env::var("TARGET") {
        if target.contains("musl") {
            println!("cargo:rustc-link-arg=-lc");
        }
    }

    let out_dir = env::var("OUT_DIR").unwrap();

    // The trampoline table is needed unconditionally (the lib includes it
    // whenever FFI dispatch exists), even when no @extern stdlib was found.
    write_ffi_invoke_table(&out_dir);

    // Collect existing .frond files
    let frond_files: Vec<PathBuf> = EXTERN_FROND_FILES
        .iter()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .collect();

    if frond_files.is_empty() {
        write_registry_stub(&out_dir);
        return;
    }

    // 1. Text-extract each .frond → ExternCFunc list, then generate .c into OUT_DIR
    let mut c_files: Vec<PathBuf> = Vec::new();
    // Every extracted function across all files — input for the symbol registry.
    let mut all_funcs: Vec<gen::ExternCFunc> = Vec::new();

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

        // Only functions whose .c was generated successfully enter the registry;
        // a registry entry for a skipped function would be an undefined symbol.
        for f in funcs {
            if !all_funcs.iter().any(|e| e.c_name == f.c_name) {
                all_funcs.push(f);
            }
        }

        println!("cargo::rerun-if-changed={}", frond_file.display());
    }

    if c_files.is_empty() {
        write_registry_stub(&out_dir);
        return;
    }

    // 1.5 Generate the symbol registry .c (function-pointer table). Rust-side
    // references to this table are what keeps every primitive alive through
    // linker dead-stripping; see the module doc comment.
    let registry_path = Path::new(&out_dir).join("frond_extern_registry.c");
    if let Err(e) = fs::write(&registry_path, generate_registry_c(&all_funcs)) {
        println!("cargo:warning=Write registry .c failed: {}", e);
        write_registry_stub(&out_dir);
        return;
    }
    c_files.push(registry_path);

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
            // Rust bindings for the registry table (real table this time).
            write_registry_rs(&out_dir, &all_funcs);
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
            // A frond binary without the builtin C primitives is functionally broken:
            // println/fs/net/time/os/rand all resolve to nothing at runtime and fail
            // silently (the Throw error is discarded when the call result is unused).
            // Fail the build loudly instead of shipping a degraded binary; the env
            // escape hatch exists for platforms without a C toolchain on purpose.
            if env::var("FROND_ALLOW_NO_EXTERN_C").ok().as_deref() == Some("1") {
                println!("cargo::warning=C compilation failed, building without builtin @extern(\"C\") primitives (FROND_ALLOW_NO_EXTERN_C=1): {}", e);
                write_registry_stub(&out_dir);
            } else {
                panic!(
                    "builtin @extern(\"C\") C compilation failed: {}\n  set FROND_ALLOW_NO_EXTERN_C=1 to build without builtin primitives (println/fs/net will not work)",
                    e
                );
            }
        }
    }
}

// ============ Symbol registry generation ============

/// Generate `frond_extern_registry.c`: a function-pointer table + parallel name
/// table + length. The Rust side (frond_extern_registry.rs) references these
/// statics, and the table references every primitive function — making them
/// reachable from Rust code and therefore immune to dead-strip/gc-sections.
fn generate_registry_c(funcs: &[gen::ExternCFunc]) -> String {
    let mut out = String::new();
    out.push_str("// Auto-generated symbol registry — DO NOT EDIT\n");
    out.push_str("#include <stdint.h>\n");
    out.push_str("#include <stddef.h>\n\n");
    for f in funcs {
        let params = if f.c_params.is_empty() {
            "void".to_string()
        } else {
            f.c_params
                .iter()
                .map(|p| format!("{} {}", p.c_type, p.name))
                .collect::<Vec<_>>()
                .join(", ")
        };
        out.push_str(&format!("extern {} {}({});\n", f.c_return, f.c_name, params));
    }
    out.push_str("\nvoid* const frond_extern_registry[] = {\n");
    for f in funcs {
        out.push_str(&format!("    (void*){},\n", f.c_name));
    }
    out.push_str("};\n\n");
    out.push_str("const char* const frond_extern_registry_names[] = {\n");
    for f in funcs {
        out.push_str(&format!("    \"{}\",\n", f.c_name));
    }
    out.push_str("};\n\n");
    out.push_str(&format!(
        "const uint64_t frond_extern_registry_len = {};\n",
        funcs.len()
    ));
    out
}

/// Write the real Rust registry bindings into OUT_DIR. Included by
/// `src/ffi/Symbols.rs` via `include!(concat!(env!("OUT_DIR"), ...))`.
fn write_registry_rs(out_dir: &str, funcs: &[gen::ExternCFunc]) {
    let n = funcs.len();
    let rs = format!(
        r#"// Auto-generated by build.rs — symbol registry bindings. DO NOT EDIT.

pub const FROND_EXTERN_REGISTRY_LEN: usize = {n};

extern "C" {{
    static frond_extern_registry: [*mut core::ffi::c_void; {n}];
    static frond_extern_registry_names: [*const core::ffi::c_char; {n}];
}}

/// Linear name → address lookup over the linked registry table. The references
/// to the extern statics are what keeps the table (and transitively every
/// @extern("C") primitive) alive through linker dead-stripping.
pub fn registry_lookup(symbol: &str) -> Option<*mut core::ffi::c_void> {{
    let bytes = symbol.as_bytes();
    unsafe {{
        for i in 0..FROND_EXTERN_REGISTRY_LEN {{
            let p = frond_extern_registry_names[i] as *const u8;
            if p.is_null() {{
                continue;
            }}
            let mut len = 0usize;
            while *p.add(len) != 0 {{
                len += 1;
            }}
            if core::slice::from_raw_parts(p, len) == bytes {{
                return Some(frond_extern_registry[i]);
            }}
        }}
    }}
    None
}}
"#,
        n = n
    );
    if let Err(e) = fs::write(Path::new(out_dir).join("frond_extern_registry.rs"), rs) {
        // The include! in Symbols.rs fails the build if the file is missing,
        // so surface this as a build error rather than continuing.
        panic!("failed to write frond_extern_registry.rs: {}", e);
    }
}

/// Write a stub registry (no table available). Keeps `include!` in Symbols.rs
/// valid on the FROND_ALLOW_NO_EXTERN_C=1 path.
fn write_registry_stub(out_dir: &str) {
    let rs = "// Auto-generated stub — builtin @extern(\"C\") C library unavailable.\n\
              pub fn registry_lookup(_symbol: &str) -> Option<*mut core::ffi::c_void> {\n    None\n}\n";
    if let Err(e) = fs::write(Path::new(out_dir).join("frond_extern_registry.rs"), rs) {
        panic!("failed to write frond_extern_registry.rs stub: {}", e);
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
// Parses .frond source text and extracts @extern("C") function declarations.
// The .frond syntax for extern C is a fixed text pattern:
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

/// Parse a .frond source string and extract all @extern("C") functions.
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
            // in practice all Raw.frond files keep it on one line. We search for
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
