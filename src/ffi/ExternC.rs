//! `@extern("C")` C code extractor + Rust FFI wrapper generator (AST path).
//!
//! Scans the parsed AST and extracts every `FunDecl` carrying the `@extern("C")`
//! attribute and an `extern_c_body`, then delegates code generation to the shared
//! [`gen`] module (same module used by `build.rs` for the text-extraction path).
//!
//! ## Shared code generation
//!
//! Type mapping tables and C/Rust code generation live in [`gen`] (`Gen.rs`),
//! which is `#[path]`-included here and `include!`-ed in `build.rs`. This ensures
//! the AST path and the build-time text path produce identical output without
//! duplicating logic.
//!
//! ## Kuzo → C type mapping
//!
//! See `gen::KUZO_TYPE_MAP` for the full mapping table.
//!
//! ## `str` return (out-parameter pattern)
//!
//! C cannot return a fat pointer directly, so Kuzo `fun foo(): str` → C
//! `void kuzo_foo(..., const char** out_data, size_t* out_len)`. The C body sets
//! `*out_data` and `*out_len`; the Rust wrapper constructs a `&'static str`.

#[allow(dead_code)]
#[path = "Gen.rs"]
mod gen;

use crate::ast::Ast::{Attribute, Decl, Module, TypeNode};
use std::borrow::Cow;

use gen::{is_i128_return, is_str_return, kuzo_type_to_c_params, kuzo_type_to_c_return, ExternCFunc};

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
/// - `RawPtr { inner }` → returns `"*T"` (e.g. `"*u8"`)
/// - `Array { element_type, size: None }` → returns `"elem[]"` (e.g. `"u8[]"`)
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
                    eprintln!(
                        "warning: @extern(\"C\") function '{}': missing C function body (expected #{{ ... }}# raw block), skipping",
                        name
                    );
                    continue;
                }
            };

            // Collect @c_include dependencies.
            let c_includes = collect_c_includes(attributes);

            // Map the return type.
            // `str` and 128-bit returns both use the out-parameter pattern: the C
            // function returns void and trailing out-pointers carry the result.
            let ret_name = extract_type_name(*return_type, arena);
            let kuzo_return = ret_name.as_deref().unwrap_or("void").to_string();
            let is_str_ret = is_str_return(&kuzo_return);
            let is_i128_ret = is_i128_return(&kuzo_return);
            let c_return = if is_str_ret || is_i128_ret {
                "void".to_string()
            } else {
                match ret_name.as_deref().and_then(kuzo_type_to_c_return) {
                    Some(c) => c.to_string(),
                    None => {
                        let ty_str = ret_name.as_deref().unwrap_or("<unknown>");
                        eprintln!(
                            "warning: @extern(\"C\") function '{}': unsupported return type '{}', skipping",
                            name, ty_str
                        );
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
                        eprintln!(
                            "warning: @extern(\"C\") function '{}': unsupported parameter type '{}' (parameter '{}'), skipping",
                            name, ty_str, param.name
                        );
                        param_ok = false;
                        break;
                    }
                }
                kuzo_params.push(gen::KuzoParam {
                    name: param.name.to_string(),
                    kuzo_type,
                });
            }
            if !param_ok {
                continue;
            }

            // For `str` returns, append the out parameters (the C side fills out_data/out_len).
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

            // For i128/u128/f128 returns, append two uint64_t* out parameters
            // (low/high). MSVC x64 cannot return `__int128` by value, so we use
            // the same out-parameter pattern as `str`.
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

    Ok(funcs)
}

// ============ Public API ============

/// Extract all `@extern("C")` functions from a module and generate the complete `.c` file content.
pub fn extract_c_from_module<'a>(module: &Module<'a>) -> Result<String, String> {
    let funcs = extract_extern_c_funcs(module)?;
    gen::generate_c_source(&funcs)
}

/// Extract all `@extern("C")` functions from a module and generate Rust FFI code (bindings + wrapper).
pub fn extract_rust_ffi_from_module<'a>(module: &Module<'a>) -> Result<String, String> {
    let funcs = extract_extern_c_funcs(module)?;
    gen::generate_rust_ffi(&funcs)
}
