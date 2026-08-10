// =========================================================================
// Display — type display formatting (TypeDisplay).
// =========================================================================

use super::Tag::*;
use super::ty::*;
use super::Arena::*;
use std::fmt;

/// Formatting wrapper for `Ty`: a `type_var` follows the `bound` chain to display the
/// final type; an unbound variable is displayed as `'_<idx>`.
pub struct TypeDisplay<'a> {
    pub arena: &'a TypeArena,
    pub ty: TypeHandle,
}

impl fmt::Display for TypeDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let resolved = self.arena.resolve(self.ty);
        let t = self.arena.get(resolved);
        match t {
            Ty::TypeVar(idx) => write!(f, "'_{}", idx),
            Ty::Void => f.write_str("void"),
            Ty::Never => f.write_str("!"),
            Ty::Unknown => f.write_str("?"),
            // Builtin scalars + Str/Null: emit the static name directly.
            Ty::Bool | Ty::Char
            | Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::I128
            | Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64 | Ty::U128
            | Ty::Isize | Ty::Usize
            | Ty::F16 | Ty::F32 | Ty::F64 | Ty::F128
            | Ty::Str | Ty::Null => f.write_str(t.name()),
            Ty::Fn(_) => {
                let (params, return_type) = self.arena.fn_parts(resolved);
                f.write_str("(")?;
                for (i, &p) in params.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", self.arena.display(p))?;
                }
                f.write_str(") -> ")?;
                write!(f, "{}", self.arena.display(return_type))
            }
            Ty::Record(_) => {
                let fields = self.arena.record_fields(resolved);
                f.write_str("(")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    if let Some(name) = &field.name {
                        write!(f, "{}: ", name)?;
                    }
                    write!(f, "{}", self.arena.display(field.ty))?;
                }
                f.write_str(")")
            }
            Ty::Adt(_) => {
                let (name, type_args) = self.arena.adt_parts(resolved);
                f.write_str(name)?;
                fmt_type_args(f, self.arena, type_args)
            }
            Ty::Nullable(_) => {
                write!(f, "{}?", self.arena.display(self.arena.nullable_inner(resolved)))
            }
            Ty::Ref(_) => {
                let (inner, is_raw) = self.arena.ref_parts(resolved);
                f.write_str(if is_raw { "*" } else { "&" })?;
                write!(f, "{}", self.arena.display(inner))
            }
            Ty::Generic(_) => {
                let (name, args) = self.arena.generic_parts(resolved);
                f.write_str(name)?;
                fmt_type_args(f, self.arena, args)
            }
            Ty::Array(_) => {
                let (elem, size) = self.arena.array_parts(resolved);
                write!(f, "{}[", self.arena.display(elem))?;
                if let Some(s) = size {
                    write!(f, "{}", s)?;
                }
                f.write_str("]")
            }
            Ty::Throw(_) => {
                let (value_type, error_type) = self.arena.throw_parts(resolved);
                f.write_str("Throw<")?;
                write!(f, "{}", self.arena.display(value_type))?;
                f.write_str(", ")?;
                write!(f, "{}", self.arena.display(error_type))?;
                f.write_str(">")
            }
            Ty::Trait(_) => {
                let (name, type_args) = self.arena.trait_parts(resolved);
                f.write_str(name)?;
                fmt_type_args(f, self.arena, type_args)
            }
            Ty::TraitObject(_) => {
                let (trait_name, method_sigs) = self.arena.trait_object_parts(resolved);
                write!(f, "dyn {}", trait_name)?;
                if method_sigs.is_empty() {
                    return Ok(());
                }
                f.write_str(" { ")?;
                for (i, m) in method_sigs.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}(/{})", m.name, m.param_count)?;
                }
                f.write_str(" }")
            }
            // Builtin generic names: Channel/Async/Lazy/Atomic/Sender/Receiver.
            Ty::Channel(_)
            | Ty::Async(_)
            | Ty::Lazy(_)
            | Ty::Atomic(_)
            | Ty::Sender(_)
            | Ty::Receiver(_)
            | Ty::Timer(_) => f.write_str(t.name()),
            Ty::ModuleRef(_) => {
                let (path, _) = self.arena.module_ref_parts(resolved);
                write!(f, "module::{}", path)
            }
        }
    }
}

/// Format a type argument list `<T1, T2>`; returns nothing for an empty list.
fn fmt_type_args(
    f: &mut fmt::Formatter<'_>,
    arena: &TypeArena,
    args: &[TypeHandle],
) -> fmt::Result {
    if args.is_empty() {
        return Ok(());
    }
    f.write_str("<")?;
    for (i, &a) in args.iter().enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{}", arena.display(a))?;
    }
    f.write_str(">")
}
