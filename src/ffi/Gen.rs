// Shared type mapping + code generation for `@extern("C")` FFI.
//
// This file is included by TWO consumers via `#[path]` / `include!`:
// - `src/ffi/ExternC.rs` — AST extraction path (kuzo binary `emit-c` / `emit-ffi`)
// - `build.rs` — text extraction path (build-time, no kuzo binary needed)
//
// MUST NOT depend on `crate::` — only `std` + pure data. This keeps it
// compilable in both the lib crate and the build script context.

// Pull in the base Kuzo→C type table (lives in src/types/Ctype.rs, the single
// source of truth for "which C type does a Kuzo scalar map to"). Both Gen.rs
// and build.rs include Ctype.rs the same way, so the table is shared without a
// `crate::` dependency.
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/types/Ctype.rs"));

// ============ Data structures ============

/// C function parameter: name + type.
pub struct CParam {
    pub name: String,
    pub c_type: String,
}

/// Kuzo parameter (used in wrapper signatures).
pub struct KuzoParam {
    pub name: String,
    pub kuzo_type: String,
}

/// Extraction result: complete information for one `@extern("C")` function.
pub struct ExternCFunc {
    pub kuzo_name: String,
    pub c_return: String,
    pub c_name: String,
    pub c_params: Vec<CParam>,
    pub c_body: String,
    pub c_includes: Vec<String>,
    pub kuzo_params: Vec<KuzoParam>,
    pub kuzo_return: String,
}

/// Kuzo type → C/Rust mapping entry (single source of truth, eliminates scalar
/// duplication across 4 match sites).
///
/// - `rust_wrapper`: Rust wrapper parameter/return type
/// - `c_param_kind`: C parameter dispatch mode (scalars pass through directly,
///   `i128`/`u128`/`f128` split into `lo`/`hi`, `str`/`u8[]` split into `data`/`len`)
///
/// NOTE: the C type for a Kuzo scalar is NOT stored here — it lives in the base
/// table `KUZO_TO_C_TYPE` (`src/types/Ctype.rs`), queried via `kuzo_to_c_type`.
/// `KuzoTypeMapping` only carries FFI-strategy data (how to pass the parameter,
/// what Rust wrapper type to use).
pub struct KuzoTypeMapping {
    pub rust_wrapper: Option<&'static str>,
    pub c_param_kind: CParamKind,
}

/// C parameter construction mode.
#[derive(Clone, Copy)]
pub enum CParamKind {
    /// Single parameter, using the C type for the corresponding `kuzo_name`.
    Single(&'static str),
    /// Two parameters `lo`/`hi` (`i128`/`u128`/`f128`).
    LoHi,
    /// Two parameters `data`/`len` (`str`/`u8[]`).
    DataLen { c_data_type: &'static str },
}

/// Complete Kuzo FFI-strategy mapping table (scalars + `str`/`void`/pointers/arrays).
///
/// The C type for each Kuzo scalar is looked up via `kuzo_to_c_type`
/// (`KUZO_TO_C_TYPE` in `src/types/Ctype.rs`); this table only carries FFI
/// strategy data (Rust wrapper type + C parameter passing mode).
pub const KUZO_TYPE_MAP: &[(&str, KuzoTypeMapping)] = &[
    // Scalar integers
    ("i8",    KuzoTypeMapping { rust_wrapper: Some("i8"),    c_param_kind: CParamKind::Single("int8_t") }),
    ("i16",   KuzoTypeMapping { rust_wrapper: Some("i16"),   c_param_kind: CParamKind::Single("int16_t") }),
    ("i32",   KuzoTypeMapping { rust_wrapper: Some("i32"),   c_param_kind: CParamKind::Single("int32_t") }),
    ("i64",   KuzoTypeMapping { rust_wrapper: Some("i64"),   c_param_kind: CParamKind::Single("int64_t") }),
    ("i128",  KuzoTypeMapping { rust_wrapper: Some("i128"), c_param_kind: CParamKind::LoHi }),
    ("u8",    KuzoTypeMapping { rust_wrapper: Some("u8"),    c_param_kind: CParamKind::Single("uint8_t") }),
    ("u16",   KuzoTypeMapping { rust_wrapper: Some("u16"), c_param_kind: CParamKind::Single("uint16_t") }),
    ("u32",   KuzoTypeMapping { rust_wrapper: Some("u32"), c_param_kind: CParamKind::Single("uint32_t") }),
    ("u64",   KuzoTypeMapping { rust_wrapper: Some("u64"), c_param_kind: CParamKind::Single("uint64_t") }),
    ("u128",  KuzoTypeMapping { rust_wrapper: Some("u128"), c_param_kind: CParamKind::LoHi }),
    ("isize", KuzoTypeMapping { rust_wrapper: Some("isize"), c_param_kind: CParamKind::Single("ssize_t") }),
    ("usize", KuzoTypeMapping { rust_wrapper: Some("usize"), c_param_kind: CParamKind::Single("size_t") }),
    // Scalar floating-point
    ("f32",   KuzoTypeMapping { rust_wrapper: Some("f32"),   c_param_kind: CParamKind::Single("float") }),
    ("f64",   KuzoTypeMapping { rust_wrapper: Some("f64"),   c_param_kind: CParamKind::Single("double") }),
    ("f16",   KuzoTypeMapping { rust_wrapper: Some("u16"),  c_param_kind: CParamKind::Single("uint16_t") }),
    ("f128",  KuzoTypeMapping { rust_wrapper: Some("u128"), c_param_kind: CParamKind::LoHi }),
    // Non-arithmetic scalars
    ("bool",  KuzoTypeMapping { rust_wrapper: Some("bool"),  c_param_kind: CParamKind::Single("int") }),
    ("char",  KuzoTypeMapping { rust_wrapper: Some("char"), c_param_kind: CParamKind::Single("uint32_t") }),
    // Special types
    ("str",   KuzoTypeMapping { rust_wrapper: Some("&str"),  c_param_kind: CParamKind::DataLen { c_data_type: "const char*" } }),
    ("void",  KuzoTypeMapping { rust_wrapper: Some("()"),    c_param_kind: CParamKind::Single("void") }),
    ("u8[]",  KuzoTypeMapping { rust_wrapper: Some("&[u8]"), c_param_kind: CParamKind::DataLen { c_data_type: "uint8_t*" } }),
    // Pointer types
    ("*u8",   KuzoTypeMapping { rust_wrapper: Some("*mut u8"),  c_param_kind: CParamKind::Single("uint8_t*") }),
    ("*i8",   KuzoTypeMapping { rust_wrapper: Some("*mut i8"),  c_param_kind: CParamKind::Single("int8_t*") }),
    ("*u16",  KuzoTypeMapping { rust_wrapper: Some("*mut u16"), c_param_kind: CParamKind::Single("uint16_t*") }),
    ("*i16",  KuzoTypeMapping { rust_wrapper: Some("*mut i16"), c_param_kind: CParamKind::Single("int16_t*") }),
    ("*u32",  KuzoTypeMapping { rust_wrapper: Some("*mut u32"), c_param_kind: CParamKind::Single("uint32_t*") }),
    ("*i32",  KuzoTypeMapping { rust_wrapper: Some("*mut i32"), c_param_kind: CParamKind::Single("int32_t*") }),
    ("*u64",  KuzoTypeMapping { rust_wrapper: Some("*mut u64"), c_param_kind: CParamKind::Single("uint64_t*") }),
    ("*i64",  KuzoTypeMapping { rust_wrapper: Some("*mut i64"), c_param_kind: CParamKind::Single("int64_t*") }),
    ("*void", KuzoTypeMapping { rust_wrapper: Some("*mut core::ffi::c_void"), c_param_kind: CParamKind::Single("void*") }),
];

// ============ Type lookup ============

/// Look up the mapping table by Kuzo type name.
#[inline]
pub fn lookup_kuzo_type(kuzo_name: &str) -> Option<&'static KuzoTypeMapping> {
    KUZO_TYPE_MAP.iter().find(|(n, _)| *n == kuzo_name).map(|(_, m)| m)
}

/// Kuzo type name → C return type.
///
/// For most types this delegates to the base table `kuzo_to_c_type`
/// (see `src/types/Ctype.rs`). The exceptions are `str` and `u8[]`, which use
/// the out-parameter pattern and therefore return `None` here (the C function
/// becomes `void` and trailing out-pointers carry the result).
pub fn kuzo_type_to_c_return(kuzo_name: &str) -> Option<&'static str> {
    // str / u8[] returns are delivered via out-parameters, not a value return.
    if matches!(kuzo_name, "str" | "u8[]") {
        return None;
    }
    kuzo_to_c_type(kuzo_name)
}

/// Kuzo type name → Rust wrapper parameter/return type.
pub fn kuzo_type_to_rust_wrapper(kuzo_type: &str) -> Option<&'static str> {
    lookup_kuzo_type(kuzo_type).and_then(|m| m.rust_wrapper)
}

/// Kuzo type name → C parameter list (one Kuzo parameter may map to multiple C parameters).
pub fn kuzo_type_to_c_params(kuzo_name: &str, param_name: &str) -> Option<Vec<CParam>> {
    let m = lookup_kuzo_type(kuzo_name)?;
    Some(match m.c_param_kind {
        CParamKind::Single(c_type) => vec![CParam { name: param_name.to_string(), c_type: c_type.to_string() }],
        CParamKind::LoHi => vec![
            CParam { name: format!("{}_lo", param_name), c_type: "uint64_t".to_string() },
            CParam { name: format!("{}_hi", param_name), c_type: "uint64_t".to_string() },
        ],
        CParamKind::DataLen { c_data_type } => vec![
            CParam { name: format!("{}_data", param_name), c_type: c_data_type.to_string() },
            CParam { name: format!("{}_len", param_name), c_type: "size_t".to_string() },
        ],
    })
}

/// C type name → Rust binding type.
pub fn c_type_to_rust(c_type: &str) -> &'static str {
    match c_type {
        "int8_t" => "i8",
        "int16_t" => "i16",
        "int32_t" => "i32",
        "int64_t" => "i64",
        "uint8_t" => "u8",
        "uint16_t" => "u16",
        "uint32_t" => "u32",
        "uint64_t" => "u64",
        "__int128" => "i128",
        "unsigned __int128" => "u128",
        "ssize_t" => "isize",
        "size_t" => "usize",
        "int" => "core::ffi::c_int",
        "const char*" => "*const core::ffi::c_char",
        "const char**" => "*mut *const core::ffi::c_char",
        "size_t*" => "*mut usize",
        "const uint8_t*" => "*const u8",
        "uint8_t*" => "*mut u8",
        "int8_t*" => "*mut i8",
        "uint16_t*" => "*mut u16",
        "int16_t*" => "*mut i16",
        "uint32_t*" => "*mut u32",
        "int32_t*" => "*mut i32",
        "uint64_t*" => "*mut u64",
        "int64_t*" => "*mut i64",
        "void*" => "*mut core::ffi::c_void",
        "float" => "f32",
        "double" => "f64",
        "void" => "()",
        _ => "()",
    }
}

/// Returns true if the Kuzo return type uses the str out-parameter pattern.
#[inline]
pub fn is_str_return(kuzo_return: &str) -> bool {
    kuzo_return == "str"
}

/// Returns true if the Kuzo return type is a 128-bit integer/float that MSVC
/// cannot return by value (MSVC x64's `__int128` does not support conversion
/// operators, so even `return (__int128)x;` fails to compile).
///
/// Such returns use the same out-parameter pattern as `str`: the C function
/// becomes `void` and two trailing `uint64_t* out_lo, out_hi` parameters carry
/// the low/high 64 bits. GCC/Clang also accept this path uniformly, so the
/// generated C is portable across all three compilers.
#[inline]
pub fn is_i128_return(kuzo_return: &str) -> bool {
    matches!(kuzo_return, "i128" | "u128" | "f128")
}

// ============ C source generation ============

/// Headers that only exist on POSIX (Linux/macOS/BSD). On MSVC these files do
/// not exist, so an unconditional `#include` would fail the build even when the
/// surrounding `#if defined(_WIN32)` branch never calls into them. Such headers
/// must be wrapped in `#else` so the preprocessor only pulls them in off-Windows.
///
/// NOTE: `fcntl.h` and `sys/stat.h` are NOT here — MSVC ships both (with `_O_*`
/// and `_stat64` variants), so they are cross-platform and stay unconditional.
const POSIX_ONLY_HEADERS: &[&str] = &[
    "unistd.h",
    "sys/socket.h",
    "netinet/in.h",
    "arpa/inet.h",
    "netdb.h",
    "sys/types.h",
    "sys/ioctl.h",
    "sys/un.h",
    "termios.h",
    "dirent.h",
    "sys/mman.h",
    "sys/wait.h",
    "pthread.h",
];

/// Headers that only exist on Windows. Wrapped in `#if defined(_WIN32)` so
/// POSIX toolchains never try to resolve them.
const WINDOWS_ONLY_HEADERS: &[&str] = &[
    "windows.h",
    "winsock2.h",
    "ws2tcpip.h",
    "winbase.h",
    "io.h",
    "direct.h",
    "wincon.h",
    "winuser.h",
    "process.h",
];

/// Classify a header name as `"posix"`, `"windows"`, or `"common"`.
fn classify_header(h: &str) -> &'static str {
    if POSIX_ONLY_HEADERS.contains(&h) {
        "posix"
    } else if WINDOWS_ONLY_HEADERS.contains(&h) {
        "windows"
    } else {
        "common"
    }
}

/// Generate the complete `.c` file content from a list of extracted functions.
pub fn generate_c_source(funcs: &[ExternCFunc]) -> Result<String, String> {
    // Collect all header files and deduplicate them, preserving first-seen order.
    let mut all_includes: Vec<String> = Vec::new();
    for func in funcs {
        for inc in &func.c_includes {
            if !all_includes.contains(inc) {
                all_includes.push(inc.clone());
            }
        }
    }

    // Partition headers by platform so platform-exclusive includes are guarded
    // by `#if defined(_WIN32)` / `#else`. Without this, an MSVC build fails at
    // `#include <termios.h>` even when the C body's own `#if` never uses it.
    let mut common: Vec<&String> = Vec::new();
    let mut posix: Vec<&String> = Vec::new();
    let mut windows: Vec<&String> = Vec::new();
    for inc in &all_includes {
        match classify_header(inc) {
            "common" => common.push(inc),
            "posix" => posix.push(inc),
            "windows" => windows.push(inc),
            _ => common.push(inc),
        }
    }

    let mut out = String::new();
    out.push_str("// Auto-generated by kuzo-rs @extern(\"C\") extractor\n");
    out.push_str("// DO NOT EDIT — regenerate with: kuzo emit-c <file>\n");
    // Always-available freestanding headers.
    out.push_str("#include <stdint.h>\n");
    out.push_str("#include <stddef.h>\n");
    for inc in &common {
        out.push_str(&format!("#include <{}>\n", inc));
    }
    // Windows-exclusive headers.
    if !windows.is_empty() {
        out.push_str("#if defined(_WIN32) || defined(_WIN64)\n");
        for inc in &windows {
            out.push_str(&format!("#include <{}>\n", inc));
        }
        // POSIX-exclusive headers go in the #else (only pulled in off-Windows).
        if !posix.is_empty() {
            out.push_str("#else\n");
            for inc in &posix {
                out.push_str(&format!("#include <{}>\n", inc));
            }
        }
        out.push_str("#endif\n");
    } else if !posix.is_empty() {
        // No Windows headers requested — guard the POSIX headers directly.
        out.push_str("#if !defined(_WIN32) && !defined(_WIN64)\n");
        for inc in &posix {
            out.push_str(&format!("#include <{}>\n", inc));
        }
        out.push_str("#endif\n");
    }
    out.push('\n');

    for func in funcs {
        let params_str = if func.c_params.is_empty() {
            "void".to_string()
        } else {
            func.c_params
                .iter()
                .map(|p| format!("{} {}", p.c_type, p.name))
                .collect::<Vec<_>>()
                .join(", ")
        };
        out.push_str(&format!("{} {}({}) {{\n", func.c_return, func.c_name, params_str));

        // For i128/u128/f128 parameters, automatically insert lo/hi reconstruction
        // variables so the C body can use the original parameter name.
        // On MSVC x64, `__int128` is not a usable type (no conversion/arithmetic
        // operators), so the reconstruction is guarded out — the C body must use
        // the raw `{n}_lo` / `{n}_hi` variables directly on that platform.
        for p in &func.kuzo_params {
            match p.kuzo_type.as_str() {
                "i128" => {
                    out.push_str(&format!(
                        "#if !defined(_WIN32) && !defined(_WIN64)\n    __int128 {n} = (__int128)((unsigned __int128){n}_lo | ((unsigned __int128){n}_hi << 64));\n#endif\n",
                        n = p.name
                    ));
                }
                "u128" | "f128" => {
                    out.push_str(&format!(
                        "#if !defined(_WIN32) && !defined(_WIN64)\n    unsigned __int128 {n} = (unsigned __int128){n}_lo | ((unsigned __int128){n}_hi << 64);\n#endif\n",
                        n = p.name
                    ));
                }
                _ => {}
            }
        }

        let body = func.c_body.trim_matches('\n');
        out.push_str(body);
        out.push_str("\n}\n\n");
    }

    Ok(out)
}

// ============ Rust FFI generation ============

/// Generate Rust FFI code (bindings + wrapper) from a list of extracted functions.
pub fn generate_rust_ffi(funcs: &[ExternCFunc]) -> Result<String, String> {
    let mut out = String::new();

    out.push_str("// Auto-generated by kuzo-rs @extern(\"C\") FFI generator\n");
    out.push_str("// DO NOT EDIT — regenerate with: kuzo emit-ffi <file>\n\n");

    // === bindings module ===
    out.push_str("/// @extern(\"C\") bindings: symbols provided by the kuzo_extern static library compiled by build.rs\n");
    out.push_str("#[cfg(has_extern_c)]\n");
    out.push_str("pub mod bindings {\n");
    out.push_str("    extern \"C\" {\n");
    for func in funcs {
        let params = if func.c_params.is_empty() {
            String::new()
        } else {
            func.c_params
                .iter()
                .map(|p| format!("{}: {}", p.name, c_type_to_rust(&p.c_type)))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let ret = c_type_to_rust(&func.c_return);
        if ret == "()" {
            out.push_str(&format!("        pub fn {}({});\n", func.c_name, params));
        } else {
            out.push_str(&format!("        pub fn {}({}) -> {};\n", func.c_name, params, ret));
        }
    }
    out.push_str("    }\n");
    out.push_str("}\n\n");

    // === wrapper module ===
    out.push_str("/// Safe wrapper layer: Kuzo value → C ABI marshal\n");
    out.push_str("#[allow(clippy::missing_safety_doc)]\n");
    out.push_str("pub mod wrapper {\n");
    out.push_str("    /// Calls the underlying binding (requires cfg(has_extern_c))\n");
    out.push_str("    #[cfg(has_extern_c)]\n");
    out.push_str("    use super::bindings;\n\n");

    for func in funcs {
        out.push_str(&generate_wrapper_fn(func));
    }

    out.push_str("}\n");

    Ok(out)
}

/// Generate the wrapper function for a single function.
fn generate_wrapper_fn(func: &ExternCFunc) -> String {
    let mut out = String::new();
    let is_str_ret = is_str_return(&func.kuzo_return);
    let is_i128_ret = is_i128_return(&func.kuzo_return);

    // Doc comment.
    let kuzo_sig = format!(
        "fun {}({}): {}",
        func.kuzo_name,
        func.kuzo_params
            .iter()
            .map(|p| format!("{}: {}", p.name, p.kuzo_type))
            .collect::<Vec<_>>()
            .join(", "),
        func.kuzo_return
    );
    out.push_str(&format!("    /// Safe wrapper for @extern(\"C\") {}\n", func.kuzo_name));
    out.push_str(&format!("    /// Kuzo signature: {}\n", kuzo_sig));

    // Function signature.
    let rust_params: Vec<String> = func
        .kuzo_params
        .iter()
        .map(|p| {
            format!(
                "{}: {}",
                p.name,
                kuzo_type_to_rust_wrapper(&p.kuzo_type).unwrap_or("()")
            )
        })
        .collect();
    let rust_return = if is_str_ret {
        "&'static str".to_string()
    } else {
        kuzo_type_to_rust_wrapper(&func.kuzo_return).unwrap_or("()").to_string()
    };

    out.push_str("    #[cfg(has_extern_c)]\n");
    if rust_return == "()" {
        out.push_str(&format!(
            "    pub unsafe fn {}({}) {{\n",
            func.kuzo_name,
            rust_params.join(", ")
        ));
    } else {
        out.push_str(&format!(
            "    pub unsafe fn {}({}) -> {} {{\n",
            func.kuzo_name,
            rust_params.join(", "),
            rust_return
        ));
    }

    // For `str` returns: declare out variables for the C side to fill.
    if is_str_ret {
        out.push_str("        let mut out_data: *const core::ffi::c_char = core::ptr::null();\n");
        out.push_str("        let mut out_len: usize = 0;\n");
    }
    // For i128/u128/f128 returns: declare low/high out variables.
    if is_i128_ret {
        out.push_str("        let mut out_lo: u64 = 0;\n");
        out.push_str("        let mut out_hi: u64 = 0;\n");
    }

    // Marshal code: generate split variables for 1:N parameters.
    for p in &func.kuzo_params {
        match p.kuzo_type.as_str() {
            "str" => {
                out.push_str(&format!(
                    "        let {n}_data = {n}.as_ptr() as *const core::ffi::c_char;\n",
                    n = p.name
                ));
                out.push_str(&format!("        let {n}_len = {n}.len();\n", n = p.name));
            }
            "u8[]" => {
                out.push_str(&format!(
                    "        let {n}_data = {n}.as_ptr() as *mut u8;\n",
                    n = p.name
                ));
                out.push_str(&format!("        let {n}_len = {n}.len();\n", n = p.name));
            }
            "i128" | "u128" | "f128" => {
                out.push_str(&format!("        let {n}_lo = {n} as u64;\n", n = p.name));
                out.push_str(&format!("        let {n}_hi = ({n} >> 64) as u64;\n", n = p.name));
            }
            _ => {}
        }
    }

    // Generate call arguments: expand Kuzo parameters into C arguments in order.
    let mut call_args: Vec<String> = Vec::new();
    for p in &func.kuzo_params {
        match p.kuzo_type.as_str() {
            "str" | "u8[]" => {
                call_args.push(format!("{}_data", p.name));
                call_args.push(format!("{}_len", p.name));
            }
            "i128" | "u128" | "f128" => {
                call_args.push(format!("{}_lo", p.name));
                call_args.push(format!("{}_hi", p.name));
            }
            "bool" => {
                call_args.push(format!("{} as core::ffi::c_int", p.name));
            }
            "char" => {
                call_args.push(format!("{} as u32", p.name));
            }
            _ => {
                call_args.push(p.name.clone());
            }
        }
    }

    // For `str` returns, append the out parameters to the call list.
    if is_str_ret {
        call_args.push("&mut out_data".to_string());
        call_args.push("&mut out_len".to_string());
    }
    // For i128/u128/f128 returns, append the low/high out parameters.
    if is_i128_ret {
        call_args.push("&mut out_lo".to_string());
        call_args.push("&mut out_hi".to_string());
    }

    // Return value conversion.
    let call_expr = format!("bindings::{}({})", func.c_name, call_args.join(", "));

    if is_str_ret {
        out.push_str(&format!("        {};\n", call_expr));
        out.push_str("        if out_data.is_null() || out_len == 0 {\n");
        out.push_str("            \"\"\n");
        out.push_str("        } else {\n");
        out.push_str("            let bytes = core::slice::from_raw_parts(out_data as *const u8, out_len);\n");
        out.push_str("            core::str::from_utf8(bytes).unwrap_or(\"\")\n");
        out.push_str("        }\n");
    } else if is_i128_ret {
        // The C call is void; the result is in out_lo/out_hi (low 64 / high 64 bits).
        out.push_str(&format!("        {};\n", call_expr));
        out.push_str(&format!("        ((out_hi as {}) << 64) | (out_lo as {})\n", rust_return, rust_return));
    } else {
        let return_expr = if rust_return == "()" {
            call_expr
        } else {
            let c_ret = c_type_to_rust(&func.c_return);
            if c_ret == rust_return {
                call_expr
            } else {
                match func.kuzo_return.as_str() {
                    "bool" => format!("{} != 0", call_expr),
                    "char" => format!("{} as char", call_expr),
                    _ => format!("{} as {}", call_expr, rust_return),
                }
            }
        };
        if rust_return == "()" {
            out.push_str(&format!("        {};\n", return_expr));
        } else {
            out.push_str(&format!("        {}\n", return_expr));
        }
    }

    out.push_str("    }\n\n");

    out
}
