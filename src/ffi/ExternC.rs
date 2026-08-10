//! `@extern("C")` C code extractor + Rust FFI wrapper generator.
//!
//! Scans the parsed AST and extracts every `FunDecl` carrying the `@extern("C")`
//! attribute and an `extern_c_body`:
//! 1. Generates a C source file (function prototypes + function bodies + header file dependencies).
//! 2. Generates Rust FFI code (`extern "C"` bindings + safe wrappers).
//!
//! ## Header file management
//!
//! The function-level `@c_include("header.h")` attribute declares the system header
//! files that a C function body depends on. The extractor collects every function's
//! `@c_include` declarations, deduplicates them, and emits them at the top of the
//! generated `.c` file.
//!
//! ## Kuzo → C type mapping
//!
//! | Kuzo type | C parameter | C return | Rust wrapper type |
//! |-----------|-------------|----------|-------------------|
//! | i8/i16/i32/i64 | name | int8_t..int64_t | i8..i64 |
//! | u8/u16/u32/u64 | name | uint8_t..uint64_t | u8..u64 |
//! | i128 | name_lo, name_hi | __int128 | i128 |
//! | u128 | name_lo, name_hi | unsigned __int128 | u128 |
//! | isize/usize | name | ssize_t/size_t | isize/usize |
//! | bool | name | int | bool |
//! | char | name | uint32_t | char |
//! | str | name_data, name_len | out: out_data, out_len | &str |
//! | f32 | name | float | f32 |
//! | f64 | name | double | f64 |
//! | f16 | name | uint16_t (bit pattern) | u16 |
//! | f128 | name_lo, name_hi | unsigned __int128 (bit pattern) | u128 |
//! | void (return) | — | void | () |
//!
//! **`str` return**: C cannot return a fat pointer directly, so an out-parameter
//! pattern is used. Kuzo `fun foo(): str` → C
//! `void kuzo_foo(..., const char** out_data, size_t* out_len)`. The C body sets
//! `*out_data` and `*out_len`, and the Rust wrapper constructs a `&'static str`.
//!
//! **`i128`/`u128`/`f128` return**: The C side uses `__int128`/`unsigned __int128`
//! (GCC/Clang extensions). Rust `i128`/`u128` are ABI-compatible in `extern "C"`.
//! MSVC does not support `__int128`; GCC/Clang is required.
//!
//! **`f16`/`f128`**: Passed as bit patterns (`u16`/`u128`); the C body converts
//! internally via `union`/`memcpy`. `f128` parameters use the same layout as
//! `u128` (two `uint64_t` values, `lo`/`hi`); `f128` returns use
//! `unsigned __int128`.
//!
//! ## Rust wrapper marshal rules
//!
//! | Kuzo parameter | Rust wrapper type | marshal action |
//! |-----------|-------------------|-------------|
//! | str | &str | s.as_ptr() as *const c_char, s.len() |
//! | i128/u128 | i128/u128 | (n as u64), (n >> 64) as u64 |
//! | f128 | u128 | (n as u64), (n >> 64) as u64 |
//! | bool | bool | b as c_int |
//! | char | char | c as u32 |
//! | other scalars | same name | pass directly |

use crate::ast::Ast::{Attribute, Decl, Module, TypeNode};
use std::borrow::Cow;

// ============ Type mapping ============

/// C function parameter: name + type.
struct CParam {
    name: String,
    c_type: String,
}

/// Kuzo parameter (used in wrapper signatures).
struct KuzoParam {
    name: String,
    kuzo_type: String,
}

/// Extraction result: complete information for one `@extern("C")` function.
struct ExternCFunc {
    kuzo_name: String,
    c_return: String,
    c_name: String,
    c_params: Vec<CParam>,
    c_body: String,
    c_includes: Vec<String>,
    kuzo_params: Vec<KuzoParam>,
    kuzo_return: String,
}

/// Kuzo type → C/Rust mapping entry (single source of truth, eliminates scalar
/// duplication across 4 match sites).
///
/// - `c_return`: C return type (`str` returns `None`, handled by the out-parameter pattern)
/// - `rust_wrapper`: Rust wrapper parameter/return type
/// - `c_param_kind`: C parameter dispatch mode (scalars pass through directly,
///   `i128`/`u128`/`f128` split into `lo`/`hi`, `str`/`u8[]` split into `data`/`len`)
struct KuzoTypeMapping {
    c_return: Option<&'static str>,
    rust_wrapper: Option<&'static str>,
    c_param_kind: CParamKind,
}

/// C parameter construction mode.
#[derive(Clone, Copy)]
enum CParamKind {
    /// Single parameter, using the C type for the corresponding `kuzo_name`.
    Single(&'static str),
    /// Two parameters `lo`/`hi` (`i128`/`u128`/`f128`).
    LoHi,
    /// Two parameters `data`/`len` (`str`/`u8[]`).
    DataLen { c_data_type: &'static str },
}

/// Complete Kuzo type mapping table (scalars + `str`/`void`/pointers/arrays).
///
/// To add a new type, simply append a row here; `kuzo_type_to_c_return`,
/// `kuzo_type_to_rust_wrapper`, and `kuzo_type_to_c_params` are auto-derived.
const KUZO_TYPE_MAP: &[(&str, KuzoTypeMapping)] = &[
    // Scalar integers
    ("i8",    KuzoTypeMapping { c_return: Some("int8_t"),  rust_wrapper: Some("i8"),    c_param_kind: CParamKind::Single("int8_t") }),
    ("i16",   KuzoTypeMapping { c_return: Some("int16_t"), rust_wrapper: Some("i16"),   c_param_kind: CParamKind::Single("int16_t") }),
    ("i32",   KuzoTypeMapping { c_return: Some("int32_t"), rust_wrapper: Some("i32"),   c_param_kind: CParamKind::Single("int32_t") }),
    ("i64",   KuzoTypeMapping { c_return: Some("int64_t"), rust_wrapper: Some("i64"),   c_param_kind: CParamKind::Single("int64_t") }),
    ("i128",  KuzoTypeMapping { c_return: Some("__int128"), rust_wrapper: Some("i128"), c_param_kind: CParamKind::LoHi }),
    ("u8",    KuzoTypeMapping { c_return: Some("uint8_t"), rust_wrapper: Some("u8"),    c_param_kind: CParamKind::Single("uint8_t") }),
    ("u16",   KuzoTypeMapping { c_return: Some("uint16_t"), rust_wrapper: Some("u16"), c_param_kind: CParamKind::Single("uint16_t") }),
    ("u32",   KuzoTypeMapping { c_return: Some("uint32_t"), rust_wrapper: Some("u32"), c_param_kind: CParamKind::Single("uint32_t") }),
    ("u64",   KuzoTypeMapping { c_return: Some("uint64_t"), rust_wrapper: Some("u64"), c_param_kind: CParamKind::Single("uint64_t") }),
    ("u128",  KuzoTypeMapping { c_return: Some("unsigned __int128"), rust_wrapper: Some("u128"), c_param_kind: CParamKind::LoHi }),
    ("isize", KuzoTypeMapping { c_return: Some("ssize_t"), rust_wrapper: Some("isize"), c_param_kind: CParamKind::Single("ssize_t") }),
    ("usize", KuzoTypeMapping { c_return: Some("size_t"),  rust_wrapper: Some("usize"), c_param_kind: CParamKind::Single("size_t") }),
    // Scalar floating-point
    ("f32",   KuzoTypeMapping { c_return: Some("float"),   rust_wrapper: Some("f32"),   c_param_kind: CParamKind::Single("float") }),
    ("f64",   KuzoTypeMapping { c_return: Some("double"),  rust_wrapper: Some("f64"),   c_param_kind: CParamKind::Single("double") }),
    ("f16",   KuzoTypeMapping { c_return: Some("uint16_t"), rust_wrapper: Some("u16"),  c_param_kind: CParamKind::Single("uint16_t") }),
    ("f128",  KuzoTypeMapping { c_return: Some("unsigned __int128"), rust_wrapper: Some("u128"), c_param_kind: CParamKind::LoHi }),
    // Non-arithmetic scalars
    ("bool",  KuzoTypeMapping { c_return: Some("int"),     rust_wrapper: Some("bool"),  c_param_kind: CParamKind::Single("int") }),
    ("char",  KuzoTypeMapping { c_return: Some("uint32_t"), rust_wrapper: Some("char"), c_param_kind: CParamKind::Single("uint32_t") }),
    // Special types
    ("str",   KuzoTypeMapping { c_return: None,            rust_wrapper: Some("&str"),  c_param_kind: CParamKind::DataLen { c_data_type: "const char*" } }),
    ("void",  KuzoTypeMapping { c_return: Some("void"),    rust_wrapper: Some("()"),    c_param_kind: CParamKind::Single("void") }),
    ("u8[]",  KuzoTypeMapping { c_return: None,            rust_wrapper: Some("&[u8]"), c_param_kind: CParamKind::DataLen { c_data_type: "uint8_t*" } }),
    // Pointer types
    ("*u8",   KuzoTypeMapping { c_return: Some("uint8_t*"), rust_wrapper: Some("*mut u8"),  c_param_kind: CParamKind::Single("uint8_t*") }),
    ("*i8",   KuzoTypeMapping { c_return: Some("int8_t*"),  rust_wrapper: Some("*mut i8"),  c_param_kind: CParamKind::Single("int8_t*") }),
    ("*u16",  KuzoTypeMapping { c_return: Some("uint16_t*"), rust_wrapper: Some("*mut u16"), c_param_kind: CParamKind::Single("uint16_t*") }),
    ("*i16",  KuzoTypeMapping { c_return: Some("int16_t*"), rust_wrapper: Some("*mut i16"), c_param_kind: CParamKind::Single("int16_t*") }),
    ("*u32",  KuzoTypeMapping { c_return: Some("uint32_t*"), rust_wrapper: Some("*mut u32"), c_param_kind: CParamKind::Single("uint32_t*") }),
    ("*i32",  KuzoTypeMapping { c_return: Some("int32_t*"), rust_wrapper: Some("*mut i32"), c_param_kind: CParamKind::Single("int32_t*") }),
    ("*u64",  KuzoTypeMapping { c_return: Some("uint64_t*"), rust_wrapper: Some("*mut u64"), c_param_kind: CParamKind::Single("uint64_t*") }),
    ("*i64",  KuzoTypeMapping { c_return: Some("int64_t*"), rust_wrapper: Some("*mut i64"), c_param_kind: CParamKind::Single("int64_t*") }),
    ("*void", KuzoTypeMapping { c_return: Some("void*"),    rust_wrapper: Some("*mut core::ffi::c_void"), c_param_kind: CParamKind::Single("void*") }),
];

/// Look up the mapping table by Kuzo type name.
#[inline]
fn lookup_kuzo_type(kuzo_name: &str) -> Option<&'static KuzoTypeMapping> {
    KUZO_TYPE_MAP.iter().find(|(n, _)| *n == kuzo_name).map(|(_, m)| m)
}

/// Kuzo type name → C return type.
///
/// `str` returns `None`: C cannot return a fat pointer, so `extract_extern_c_funcs`
/// handles it via the out-parameter pattern.
/// `i128`/`u128`/`f128` return `__int128`/`unsigned __int128` (requires GCC/Clang).
/// `f16` returns `uint16_t` (bit pattern).
fn kuzo_type_to_c_return(kuzo_name: &str) -> Option<&'static str> {
    lookup_kuzo_type(kuzo_name).and_then(|m| m.c_return)
}

/// Kuzo type name → Rust wrapper parameter/return type.
///
/// `f16` → `u16`, `f128` → `u128`: passed as bit patterns; on the Kuzo side
/// `f16`/`f128` are internally `u16`/`u128`.
fn kuzo_type_to_rust_wrapper(kuzo_type: &str) -> Option<&'static str> {
    lookup_kuzo_type(kuzo_type).and_then(|m| m.rust_wrapper)
}

/// Kuzo type name → C parameter list (one Kuzo parameter may map to multiple C parameters).
fn kuzo_type_to_c_params(kuzo_name: &str, param_name: &str) -> Option<Vec<CParam>> {
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
fn c_type_to_rust(c_type: &str) -> &'static str {
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

// ============ Attribute recognition ============

/// Check whether the attribute list contains `@extern("C")`.
/// [E-3] Only recognizes uppercase "C" (project constraint); a lowercase "c"
/// emits a warning to avoid silently missing the attribute.
fn is_extern_c(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| {
        if a.name != super::ATTR_EXTERN {
            return false;
        }
        if a.args.contains(&"C") {
            return true;
        }
        if a.args.contains(&"c") {
            eprintln!("warning: @extern(\"c\") must use uppercase 'C' (i.e. @extern(\"C\")); this attribute will be ignored");
        }
        false
    })
}

/// Collect header file names from `@c_include("...")` attributes.
fn collect_c_includes(attrs: &[Attribute]) -> Vec<String> {
    let mut includes = Vec::new();
    for attr in attrs {
        if attr.name == super::ATTR_C_INCLUDE {
            for arg in &attr.args {
                let inc = arg.to_string();
                if !includes.contains(&inc) {
                    includes.push(inc);
                }
            }
        }
    }
    includes
}

// ============ AST extraction ============

/// Extract the type name string from a `TypeNode`.
///
/// Supported types:
/// - `Named { name }` → returns `name` (e.g. `"i32"`, `"str"`)
/// - `Array { element_type, size: None }` → returns `"elem[]"` (e.g. `"u8[]"`)
/// - `Array { element_type, size: Some(n) }` → returns `"elem[n]"` (`@extern("C")` not yet supported)
///
/// Other complex types (`Record`/`Function`, etc.) return `None`.
fn extract_type_name<'a>(
    ty: Option<crate::ast::Ast::TypeRef>,
    arena: &crate::ast::Ast::AstArena<'a>,
) -> Option<Cow<'a, str>> {
    let ty_ref = ty?;
    // [E-2] Safe indexing: a malformed AST (from parser error recovery) may produce
    // an invalid TypeRef; this avoids a panic.
    let node = arena.types.get(ty_ref.0 as usize)?;
    match &node.node {
        TypeNode::Named { name } => Some(Cow::Borrowed(*name)),
        TypeNode::RawPtr { inner } => {
            // *T → "*T" (e.g. *u8 → "*u8"), mapped to a C pointer by
            // kuzo_type_to_c_return / kuzo_type_to_c_params.
            let inner_ref = *inner;
            let inner_node = arena.types.get(inner_ref.0 as usize)?;
            if let TypeNode::Named { name: inner_name } = &inner_node.node {
                Some(Cow::Owned(format!("*{}", inner_name)))
            } else {
                None
            }
        }
        TypeNode::Array { element_type, size } => {
            let elem_ref = *element_type;
            let elem_node = arena.types.get(elem_ref.0 as usize)?;
            if let TypeNode::Named { name: elem_name } = &elem_node.node {
                if size.is_none() {
                    Some(Cow::Owned(format!("{}[]", elem_name)))
                } else {
                    // Fixed-size arrays are not yet supported for @extern("C").
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Extract information for all `@extern("C")` functions from a module.
fn extract_extern_c_funcs<'a>(module: &Module<'a>) -> Result<Vec<ExternCFunc>, String> {
    let arena = &module.arena;
    let mut funcs = Vec::new();
    let mut errors = Vec::new();

    for decl in &module.declarations {
        if let Decl::FunDecl {
            name,
            params,
            return_type,
            attributes,
            extern_c_body,
            ..
        } = &decl.node
        {
            // Only process functions with the @extern("C") attribute.
            if !is_extern_c(attributes) {
                continue;
            }

            // @extern("C") requires a C function body.
            let c_body = match extern_c_body {
                Some(body) => body.to_string(),
                None => {
                    errors.push(format!(
                        "@extern(\"C\") function '{}': missing C function body (expected #{{ ... }}# raw block)",
                        name
                    ));
                    continue;
                }
            };

            // Collect @c_include dependencies.
            let c_includes = collect_c_includes(attributes);

            // Map the return type.
            // `str` return uses the out-parameter pattern: the C function returns void
            // and `out_data`/`out_len` parameters are appended.
            // `i128`/`u128`/`f128` returns use `__int128`/`unsigned __int128` (requires GCC/Clang).
            let ret_name = extract_type_name(*return_type, arena);
            let kuzo_return = ret_name.as_deref().unwrap_or("void").to_string();
            let is_str_return = crate::value::ValueTag::from_name(&kuzo_return)
                .is_some_and(|t| t.family() == crate::types::TypeFamily::Str);
            let c_return = if is_str_return {
                "void".to_string()
            } else {
                match ret_name.as_deref().and_then(kuzo_type_to_c_return) {
                    Some(c) => c.to_string(),
                    None => {
                        let ty_str = ret_name.as_deref().unwrap_or("<unknown>");
                        errors.push(format!(
                            "@extern(\"C\") function '{}': unsupported return type '{}'",
                            name, ty_str
                        ));
                        continue;
                    }
                }
            };

            // Map parameters.
            let mut c_params = Vec::new();
            let mut kuzo_params = Vec::new();
            let mut param_ok = true;
            for param in params.iter() {
                let param_ty_name = extract_type_name(param.type_annotation, arena);
                let kuzo_type = param_ty_name.as_deref().unwrap_or("<unknown>").to_string();
                match param_ty_name.as_deref().and_then(|n| kuzo_type_to_c_params(n, param.name)) {
                    Some(ps) => c_params.extend(ps),
                    None => {
                        let ty_str = param_ty_name.as_deref().unwrap_or("<unknown>");
                        errors.push(format!(
                            "@extern(\"C\") function '{}': unsupported parameter type '{}' (parameter '{}')",
                            name, ty_str, param.name
                        ));
                        param_ok = false;
                        break;
                    }
                }
                kuzo_params.push(KuzoParam {
                    name: param.name.to_string(),
                    kuzo_type,
                });
            }
            if !param_ok {
                continue;
            }

            // For `str` returns, append the out parameters (the C side fills out_data/out_len).
            if is_str_return {
                c_params.push(CParam {
                    name: "out_data".to_string(),
                    c_type: "const char**".to_string(),
                });
                c_params.push(CParam {
                    name: "out_len".to_string(),
                    c_type: "size_t*".to_string(),
                });
            }

            funcs.push(ExternCFunc {
                kuzo_name: name.to_string(),
                c_return,
                c_name: format!("kuzo_extern_{}", name),
                c_params,
                c_body,
                c_includes,
                kuzo_params,
                kuzo_return,
            });
        }
    }

    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    Ok(funcs)
}

// ============ C code generation ============

/// Extract all `@extern("C")` functions from a module and generate the complete `.c` file content.
pub fn extract_c_from_module<'a>(module: &Module<'a>) -> Result<String, String> {
    let funcs = extract_extern_c_funcs(module)?;

    // Collect all header files and deduplicate them.
    let mut all_includes: Vec<String> = Vec::new();
    for func in &funcs {
        for inc in &func.c_includes {
            if !all_includes.contains(inc) {
                all_includes.push(inc.clone());
            }
        }
    }

    let mut out = String::new();
    out.push_str("// Auto-generated by kuzo-rs @extern(\"C\") extractor\n");
    out.push_str("// DO NOT EDIT — regenerate with: kuzo emit-c <file>\n");
    out.push_str("#include <stdint.h>\n");
    out.push_str("#include <stddef.h>\n");
    for inc in &all_includes {
        out.push_str(&format!("#include <{}>\n", inc));
    }
    out.push('\n');

    for func in &funcs {
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
        // variables so the C body can use the original parameter name (e.g. `x`)
        // without manually handling x_lo/x_hi.
        for p in &func.kuzo_params {
            match p.kuzo_type.as_str() {
                "i128" => {
                    out.push_str(&format!(
                        "    __int128 {n} = (__int128)((unsigned __int128){n}_lo | ((unsigned __int128){n}_hi << 64));\n",
                        n = p.name
                    ));
                }
                "u128" | "f128" => {
                    out.push_str(&format!(
                        "    unsigned __int128 {n} = (unsigned __int128){n}_lo | ((unsigned __int128){n}_hi << 64);\n",
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

/// Extract all `@extern("C")` functions from a module and generate Rust FFI code (bindings + wrapper).
pub fn extract_rust_ffi_from_module<'a>(module: &Module<'a>) -> Result<String, String> {
    let funcs = extract_extern_c_funcs(module)?;
    generate_rust_ffi(&funcs)
}

fn generate_rust_ffi(funcs: &[ExternCFunc]) -> Result<String, String> {
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
///
/// `str` return uses the out-parameter pattern: declare out variables → call the
/// binding with `&mut` → construct a `&'static str`.
/// `f128` parameters behave like `u128` (split into `lo`/`hi`); `f16` parameters
/// are passed directly as `u16`.
fn generate_wrapper_fn(func: &ExternCFunc) -> String {
    let mut out = String::new();
    let is_str_return = crate::value::ValueTag::from_name(&func.kuzo_return)
        .is_some_and(|t| t.family() == crate::types::TypeFamily::Str);

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
    // `str` returns use `&'static str` (filled by the C side's out parameters; the wrapper constructs the reference).
    let rust_return = if is_str_return {
        "&'static str"
    } else {
        kuzo_type_to_rust_wrapper(&func.kuzo_return).unwrap_or("()")
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
    if is_str_return {
        out.push_str("        let mut out_data: *const core::ffi::c_char = core::ptr::null();\n");
        out.push_str("        let mut out_len: usize = 0;\n");
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
            "str" => {
                call_args.push(format!("{}_data", p.name));
                call_args.push(format!("{}_len", p.name));
            }
            "u8[]" => {
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
    if is_str_return {
        call_args.push("&mut out_data".to_string());
        call_args.push("&mut out_len".to_string());
    }

    // Return value conversion.
    let call_expr = format!("bindings::{}({})", func.c_name, call_args.join(", "));

    if is_str_return {
        // `str` return: first call the binding to fill the out variables, then
        // construct a `&'static str` to return.
        // [E-1] Hardened: validate null + UTF-8, eliminating two sources of UB
        // (`from_utf8_unchecked` and null slicing).
        // The lifetime UB is eliminated by the C body contract: the C body must
        // write to 'static memory (a string literal pointer) and must not return
        // a stack/heap temporary buffer (otherwise the `&'static str` returned by
        // the wrapper would dangle after the C function returns).
        out.push_str(&format!("        {};\n", call_expr));
        out.push_str("        if out_data.is_null() || out_len == 0 {\n");
        out.push_str("            \"\"\n");
        out.push_str("        } else {\n");
        out.push_str("            let bytes = core::slice::from_raw_parts(out_data as *const u8, out_len);\n");
        out.push_str("            core::str::from_utf8(bytes).unwrap_or(\"\")\n");
        out.push_str("        }\n");
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
