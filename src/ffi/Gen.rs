// Shared type mapping + code generation for `@extern("C")` FFI.
//
// This file is included by TWO consumers via `#[path]` / `include!`:
// - `src/ffi/ExternC.rs` — AST extraction path (kuzo binary `emit-c` / `emit-ffi`)
// - `build.rs` — text extraction path (build-time, no kuzo binary needed)
//
// MUST NOT depend on `crate::` — only `std` + pure data. This keeps it
// compilable in both the lib crate and the build script context.

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
/// - `c_return`: C return type (`str` returns `None`, handled by the out-parameter pattern)
/// - `rust_wrapper`: Rust wrapper parameter/return type
/// - `c_param_kind`: C parameter dispatch mode (scalars pass through directly,
///   `i128`/`u128`/`f128` split into `lo`/`hi`, `str`/`u8[]` split into `data`/`len`)
pub struct KuzoTypeMapping {
    pub c_return: Option<&'static str>,
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

/// Complete Kuzo type mapping table (scalars + `str`/`void`/pointers/arrays).
///
/// To add a new type, simply append a row here; `kuzo_type_to_c_return`,
/// `kuzo_type_to_rust_wrapper`, and `kuzo_type_to_c_params` are auto-derived.
pub const KUZO_TYPE_MAP: &[(&str, KuzoTypeMapping)] = &[
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

// ============ Type lookup ============

/// Look up the mapping table by Kuzo type name.
#[inline]
pub fn lookup_kuzo_type(kuzo_name: &str) -> Option<&'static KuzoTypeMapping> {
    KUZO_TYPE_MAP.iter().find(|(n, _)| *n == kuzo_name).map(|(_, m)| m)
}

/// Kuzo type name → C return type.
pub fn kuzo_type_to_c_return(kuzo_name: &str) -> Option<&'static str> {
    lookup_kuzo_type(kuzo_name).and_then(|m| m.c_return)
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

// ============ C source generation ============

/// Generate the complete `.c` file content from a list of extracted functions.
pub fn generate_c_source(funcs: &[ExternCFunc]) -> Result<String, String> {
    // Collect all header files and deduplicate them.
    let mut all_includes: Vec<String> = Vec::new();
    for func in funcs {
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
    if is_str_ret {
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
