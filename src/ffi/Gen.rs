// Shared type mapping + code generation for `@extern("C")` FFI.
//
// This file is included by TWO consumers via `#[path]` / `include!`:
// - `src/ffi/ExternC.rs` — AST extraction path (frond binary `emit-c` / `emit-ffi`)
// - `build.rs` — text extraction path (build-time, no frond binary needed)
//
// MUST NOT depend on `crate::` — only `std` + pure data. This keeps it
// compilable in both the lib crate and the build script context.

// Pull in the base Frond→C type table (lives in src/types/Ctype.rs, the single
// source of truth for "which C type does a Frond scalar map to"). Both Gen.rs
// and build.rs include Ctype.rs the same way, so the table is shared without a
// `crate::` dependency.
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/types/Ctype.rs"));

// ============ Data structures ============

/// C function parameter: name + type.
pub struct CParam {
    pub name: String,
    pub c_type: String,
}

/// Frond parameter (used in wrapper signatures).
pub struct Param {
    pub name: String,
    pub type_name: String,
}

/// Extraction result: complete information for one `@extern("C")` function.
pub struct ExternCFunc {
    pub name: String,
    pub c_return: String,
    pub c_name: String,
    pub c_params: Vec<CParam>,
    pub c_body: String,
    pub c_includes: Vec<String>,
    pub params: Vec<Param>,
    pub return_ty: String,
}

/// Frond type → C/Rust mapping entry (single source of truth, eliminates scalar
/// duplication across 4 match sites).
///
/// - `rust_wrapper`: Rust wrapper parameter/return type
/// - `c_param_kind`: C parameter dispatch mode (scalars pass through directly,
///   `i128`/`u128`/`f128` split into `lo`/`hi`, `str`/`u8[]` split into `data`/`len`)
///
/// NOTE: the C type for a Frond scalar is NOT stored here — it lives in the base
/// table `TO_C_TYPE` (`src/types/Ctype.rs`), queried via `to_c_type`.
/// `TypeMapping` only carries FFI-strategy data (how to pass the parameter,
/// what Rust wrapper type to use).
pub struct TypeMapping {
    /// Rust wrapper parameter/return type. Rust FFI generation has been removed;
    /// this field is retained for potential future introspection/tooling use and
    /// currently has no consumer.
    #[allow(dead_code)]
    pub rust_wrapper: Option<&'static str>,
    pub c_param_kind: CParamKind,
}

/// C parameter construction mode.
#[derive(Clone, Copy)]
pub enum CParamKind {
    /// Single parameter, using the C type for the corresponding `name`.
    Single(&'static str),
    /// Two parameters `lo`/`hi` (`i128`/`u128`/`f128`).
    LoHi,
    /// Two parameters `data`/`len` (`str`/`u8[]`).
    DataLen { c_data_type: &'static str },
}

/// Complete Frond FFI-strategy mapping table (scalars + `str`/`void`/pointers/arrays).
///
/// The C type for each Frond scalar is looked up via `to_c_type`
/// (`TO_C_TYPE` in `src/types/Ctype.rs`); this table only carries FFI
/// strategy data (Rust wrapper type + C parameter passing mode).
pub const TYPE_MAP: &[(&str, TypeMapping)] = &[
    // Scalar integers
    ("i8",    TypeMapping { rust_wrapper: Some("i8"),    c_param_kind: CParamKind::Single("int8_t") }),
    ("i16",   TypeMapping { rust_wrapper: Some("i16"),   c_param_kind: CParamKind::Single("int16_t") }),
    ("i32",   TypeMapping { rust_wrapper: Some("i32"),   c_param_kind: CParamKind::Single("int32_t") }),
    ("i64",   TypeMapping { rust_wrapper: Some("i64"),   c_param_kind: CParamKind::Single("int64_t") }),
    ("i128",  TypeMapping { rust_wrapper: Some("i128"), c_param_kind: CParamKind::LoHi }),
    ("u8",    TypeMapping { rust_wrapper: Some("u8"),    c_param_kind: CParamKind::Single("uint8_t") }),
    ("u16",   TypeMapping { rust_wrapper: Some("u16"), c_param_kind: CParamKind::Single("uint16_t") }),
    ("u32",   TypeMapping { rust_wrapper: Some("u32"), c_param_kind: CParamKind::Single("uint32_t") }),
    ("u64",   TypeMapping { rust_wrapper: Some("u64"), c_param_kind: CParamKind::Single("uint64_t") }),
    ("u128",  TypeMapping { rust_wrapper: Some("u128"), c_param_kind: CParamKind::LoHi }),
    ("isize", TypeMapping { rust_wrapper: Some("isize"), c_param_kind: CParamKind::Single("ssize_t") }),
    ("usize", TypeMapping { rust_wrapper: Some("usize"), c_param_kind: CParamKind::Single("size_t") }),
    // Scalar floating-point
    ("f32",   TypeMapping { rust_wrapper: Some("f32"),   c_param_kind: CParamKind::Single("float") }),
    ("f64",   TypeMapping { rust_wrapper: Some("f64"),   c_param_kind: CParamKind::Single("double") }),
    ("f16",   TypeMapping { rust_wrapper: Some("u16"),  c_param_kind: CParamKind::Single("uint16_t") }),
    ("f128",  TypeMapping { rust_wrapper: Some("u128"), c_param_kind: CParamKind::LoHi }),
    // Non-arithmetic scalars
    ("bool",  TypeMapping { rust_wrapper: Some("bool"),  c_param_kind: CParamKind::Single("int") }),
    ("char",  TypeMapping { rust_wrapper: Some("char"), c_param_kind: CParamKind::Single("uint32_t") }),
    // Special types
    ("str",   TypeMapping { rust_wrapper: Some("&str"),  c_param_kind: CParamKind::DataLen { c_data_type: "const char*" } }),
    ("void",  TypeMapping { rust_wrapper: Some("()"),    c_param_kind: CParamKind::Single("void") }),
    ("u8[]",  TypeMapping { rust_wrapper: Some("&[u8]"), c_param_kind: CParamKind::DataLen { c_data_type: "uint8_t*" } }),
    // Pointer types
    ("*u8",   TypeMapping { rust_wrapper: Some("*mut u8"),  c_param_kind: CParamKind::Single("uint8_t*") }),
    ("*i8",   TypeMapping { rust_wrapper: Some("*mut i8"),  c_param_kind: CParamKind::Single("int8_t*") }),
    ("*u16",  TypeMapping { rust_wrapper: Some("*mut u16"), c_param_kind: CParamKind::Single("uint16_t*") }),
    ("*i16",  TypeMapping { rust_wrapper: Some("*mut i16"), c_param_kind: CParamKind::Single("int16_t*") }),
    ("*u32",  TypeMapping { rust_wrapper: Some("*mut u32"), c_param_kind: CParamKind::Single("uint32_t*") }),
    ("*i32",  TypeMapping { rust_wrapper: Some("*mut i32"), c_param_kind: CParamKind::Single("int32_t*") }),
    ("*u64",  TypeMapping { rust_wrapper: Some("*mut u64"), c_param_kind: CParamKind::Single("uint64_t*") }),
    ("*i64",  TypeMapping { rust_wrapper: Some("*mut i64"), c_param_kind: CParamKind::Single("int64_t*") }),
    ("*void", TypeMapping { rust_wrapper: Some("*mut core::ffi::c_void"), c_param_kind: CParamKind::Single("void*") }),
];

// ============ Type lookup ============

/// Look up the mapping table by Frond type name.
#[inline]
pub fn lookup_type(name: &str) -> Option<&'static TypeMapping> {
    TYPE_MAP.iter().find(|(n, _)| *n == name).map(|(_, m)| m)
}

/// Frond type name → C return type.
///
/// For most types this delegates to the base table `to_c_type`
/// (see `src/types/Ctype.rs`). The exceptions are `str` and `u8[]`, which use
/// the out-parameter pattern and therefore return `None` here (the C function
/// becomes `void` and trailing out-pointers carry the result).
pub fn type_to_c_return(name: &str) -> Option<&'static str> {
    // str / u8[] returns are delivered via out-parameters, not a value return.
    if matches!(name, "str" | "u8[]") {
        return None;
    }
    to_c_type(name)
}

/// Frond type name → C parameter list (one Frond parameter may map to multiple C parameters).
pub fn type_to_c_params(name: &str, param_name: &str) -> Option<Vec<CParam>> {
    let m = lookup_type(name)?;
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

/// Returns true if the Frond return type uses the str out-parameter pattern.
#[inline]
pub fn is_str_return(return_ty: &str) -> bool {
    return_ty == "str"
}

/// Returns true if the Frond return type is a 128-bit integer/float that MSVC
/// cannot return by value (MSVC x64's `__int128` does not support conversion
/// operators, so even `return (__int128)x;` fails to compile).
///
/// Such returns use the same out-parameter pattern as `str`: the C function
/// becomes `void` and two trailing `uint64_t* out_lo, out_hi` parameters carry
/// the low/high 64 bits. GCC/Clang also accept this path uniformly, so the
/// generated C is portable across all three compilers.
#[inline]
pub fn is_i128_return(return_ty: &str) -> bool {
    matches!(return_ty, "i128" | "u128" | "f128")
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
    "spawn.h",
    "sys/statvfs.h",
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
    out.push_str("// Auto-generated by frond-rs @extern(\"C\") extractor\n");
    out.push_str("// DO NOT EDIT — regenerate with: frond emit-c <file>\n");
    // Always-available freestanding headers.
    out.push_str("#include <stdint.h>\n");
    out.push_str("#include <stddef.h>\n");
    // Symbol export macro: Windows needs __declspec(dllexport) so the symbols
    // appear in the .exe export table and can be found by dlsym(GetProcAddress).
    // Linux/macOS export extern "C" globals by default — macro expands to nothing.
    out.push_str("#if defined(_WIN32) || defined(_WIN64)\n");
    out.push_str("  #define FROND_EXPORT __declspec(dllexport)\n");
    out.push_str("#else\n");
    out.push_str("  #define FROND_EXPORT\n");
    out.push_str("#endif\n");
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
        out.push_str(&format!("FROND_EXPORT {} {}({}) {{\n", func.c_return, func.c_name, params_str));

        // For i128/u128/f128 parameters, automatically insert lo/hi reconstruction
        // variables so the C body can use the original parameter name.
        // On MSVC x64, `__int128` is not a usable type (no conversion/arithmetic
        // operators), so the reconstruction is guarded out — the C body must use
        // the raw `{n}_lo` / `{n}_hi` variables directly on that platform.
        for p in &func.params {
            match p.type_name.as_str() {
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
