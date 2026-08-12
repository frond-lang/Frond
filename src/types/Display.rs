// =========================================================================
// Display — type display formatting (TypeDisplay).
// =========================================================================

use super::Tag::*;
use super::Ty::*;
use super::Arena::*;
use std::fmt;

/// Formatting wrapper for `Type`: a `type_var` follows the `bound` chain to display the
/// final type; an unbound variable is displayed as `'_` (hiding the internal index).
pub struct TypeDisplay<'a> {
    pub arena: &'a TypeArena,
    pub ty: TypeHandle,
}

impl fmt::Display for TypeDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let resolved = self.arena.resolve(self.ty);
        let t = self.arena.get(resolved);
        match t {
            Type::TypeVar(_) => f.write_str("'_"),
            Type::Void => f.write_str("void"),
            Type::Never => f.write_str("!"),
            Type::Unknown => f.write_str("?"),
            // Builtin scalars + Str/Null: emit the static name directly.
            Type::Bool | Type::Char
            | Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::I128
            | Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::U128
            | Type::Isize | Type::Usize
            | Type::F16 | Type::F32 | Type::F64 | Type::F128
            | Type::Str | Type::Null => f.write_str(t.name()),
            Type::Fn(_) => {
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
            Type::Record(_) => {
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
            Type::Adt(_) => {
                let (name, type_args) = self.arena.adt_parts(resolved);
                f.write_str(name)?;
                fmt_type_args(f, self.arena, type_args)
            }
            Type::Nullable(_) => {
                write!(f, "{}?", self.arena.display(self.arena.nullable_inner(resolved)))
            }
            Type::Ref(_) => {
                let (inner, is_raw) = self.arena.ref_parts(resolved);
                f.write_str(if is_raw { "*" } else { "&" })?;
                write!(f, "{}", self.arena.display(inner))
            }
            Type::Generic(_) => {
                let (name, args) = self.arena.generic_parts(resolved);
                f.write_str(name)?;
                fmt_type_args(f, self.arena, args)
            }
            Type::Array(_) => {
                let (elem, size) = self.arena.array_parts(resolved);
                write!(f, "{}[", self.arena.display(elem))?;
                if let Some(s) = size {
                    write!(f, "{}", s)?;
                }
                f.write_str("]")
            }
            Type::Throw(_) => {
                let (value_type, error_type) = self.arena.throw_parts(resolved);
                f.write_str("Throw<")?;
                write!(f, "{}", self.arena.display(value_type))?;
                f.write_str(", ")?;
                write!(f, "{}", self.arena.display(error_type))?;
                f.write_str(">")
            }
            Type::Trait(_) => {
                let (name, type_args) = self.arena.trait_parts(resolved);
                f.write_str(name)?;
                fmt_type_args(f, self.arena, type_args)
            }
            Type::TraitObject(_) => {
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
            Type::Channel(_)
            | Type::Async(_)
            | Type::Lazy(_)
            | Type::Atomic(_)
            | Type::Sender(_)
            | Type::Receiver(_)
            | Type::Timer(_) => f.write_str(t.name()),
            Type::ModuleRef(_) => {
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
