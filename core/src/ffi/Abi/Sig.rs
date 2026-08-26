//! Sig — C ABI type and signature descriptions.
//!
//! These types describe the C ABI classification of FFI call parameters and
//! return values, letting the [`crate::ffi::Abi`] invoker dispatch each to the
//! appropriate trampoline for the platform.

/// C ABI type classification for parameters / return values.
///
/// v1 only covers scalars, floating-point, and pointers. `str`/`u8[]` are split
/// into `(ptr, len)` at the upper Marshal layer and then mapped to `Ptr` + `Int`
/// respectively. Passing `i128`/struct by value is deferred to a later stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AbiType {
    Void,
    /// Integer types (i8..i64, u8..u64, bool, char, isize/usize).
    /// `bits` is the bit width (8/16/32/64); `signed` distinguishes signedness.
    /// From the invoker's perspective, all integers are zero- or sign-extended to
    /// `u64` for transport.
    Int { bits: u8, signed: bool },
    Float32,
    Float64,
    /// Pointer (`*T` / opaque extern type), passed at the platform's pointer width.
    Ptr,
}

/// Complete FFI signature: parameter list + return type.
#[derive(Clone, Debug, Hash)]
pub struct AbiSig {
    pub params: Vec<AbiType>,
    pub ret: AbiType,
    /// Param indices declared with the runtime-only single-slot pointer atoms
    /// (`cstr` / `cbuf`): a lone C pointer slot in the system-C ABI shape.
    /// Unlike the `str`/array atoms these do NOT expand to `(ptr, len)`, so
    /// trailing args keep their positions (LLVM/Win32-style signatures with a
    /// pointer before other parameters). Empty for every compile-time
    /// `@extern` sig and any lookup sig without `cstr`/`cbuf`.
    pub raw_ptr_params: Vec<usize>,
}

impl AbiSig {
    pub fn new(params: Vec<AbiType>, ret: AbiType) -> Self {
        Self { params, ret, raw_ptr_params: Vec::new() }
    }
}

/// Map a Frond type name to an `AbiType` (single source of truth, shared by the
/// compile-time `@extern("C")` path and the runtime `Lib.lookup` sig parser).
/// `str` is handled separately by `push_abi_types_for_name` (two slots).
pub fn abi_type_from_name(ty_name: &str) -> AbiType {
    match ty_name {
        "void" => AbiType::Void,
        "i8" => AbiType::Int { bits: 8, signed: true },
        "i16" => AbiType::Int { bits: 16, signed: true },
        "i32" => AbiType::Int { bits: 32, signed: true },
        "i64" | "isize" => AbiType::Int { bits: 64, signed: true },
        "u8" | "bool" => AbiType::Int { bits: 8, signed: false },
        "char" => AbiType::Int { bits: 32, signed: false },
        "u16" => AbiType::Int { bits: 16, signed: false },
        "u32" => AbiType::Int { bits: 32, signed: false },
        "u64" | "usize" => AbiType::Int { bits: 64, signed: false },
        "f32" => AbiType::Float32,
        "f64" => AbiType::Float64,
        _ if ty_name.starts_with('*') => AbiType::Ptr,
        _ => AbiType::Int { bits: 64, signed: true }, // fallback
    }
}

/// Scalar array type names that expand to `(Ptr, Int)` like `str`/`u8[]` — the
/// scalar-intrinsic C fast path of std.core.mem. Keep in sync with the array
/// entries of `TYPE_MAP` (ffi/Gen.rs) and Marshal's scalar serialization set:
/// a name here without a TYPE_MAP entry cannot be declared in an `@extern("C")`
/// signature; one without a Marshal serializer would fail at call time.
/// Every scalar element width is covered, 128-bit elements included: they
/// marshal as native 16-byte blocks and their C pointers are `uint64_t*`
/// pair views (no portable 128-bit C type; fill VALUES cross FFI through a
/// single-element pattern array, not a scalar param).
pub const SCALAR_ARRAY_TYPES: &[&str] = &[
    "i8[]", "i16[]", "i32[]", "i64[]", "u16[]", "u32[]", "u64[]",
    "isize[]", "usize[]", "f32[]", "f64[]", "bool[]", "char[]",
    "i128[]", "u128[]", "f16[]", "f128[]",
];

/// Push `AbiType`(s) for a Frond type name. `str`, `u8[]`, and scalar arrays
/// expand to the `(Ptr, Int)` two slots, mirroring the DataLen C-side expansion
/// in ffi/Gen.rs (`{p}_data` / `{p}_len`). For arrays the Int slot carries the
/// ELEMENT count (u8[]: identical to its byte count); the C side knows its own
/// element width from the typed `{p}_data` pointer.
pub fn push_abi_types_for_name(ty_name: &str, out: &mut Vec<AbiType>) {
    if ty_name == "str" || ty_name == "u8[]" || SCALAR_ARRAY_TYPES.contains(&ty_name) {
        out.push(AbiType::Ptr);
        out.push(AbiType::Int { bits: 64, signed: false });
    } else {
        out.push(abi_type_from_name(ty_name));
    }
}

/// Parse a `Lib.lookup` argument-signature string (comma-separated Frond type
/// names, e.g. `"u64, u8[]"`; empty string = no arguments) into the parameter
/// list of an `AbiSig`. Unknown atoms are errors (unlike the compile-time path,
/// which falls back to i64 — a typo in a lookup sig should fail loudly).
/// `ret` is supplied by the caller (the static `ForeignFn[R]` type annotation).
pub fn parse_arg_sig(args_csv: &str, ret: AbiType) -> Result<AbiSig, String> {
    let mut params = Vec::new();
    let mut raw_ptr_params = Vec::new();
    let trimmed = args_csv.trim();
    if !trimmed.is_empty() {
        for atom in trimmed.split(',') {
            let atom = atom.trim();
            if atom.is_empty() {
                return Err(format!("empty type atom in signature '{}'", args_csv));
            }
            if !is_known_atom(atom) {
                return Err(format!(
                    "unknown type '{}' in signature '{}' (allowed: i8..i64/isize, u8..u64/usize, f32, f64, bool, char, str, cstr, cbuf, *T, scalar arrays)",
                    atom, args_csv
                ));
            }
            if atom == "cstr" || atom == "cbuf" {
                raw_ptr_params.push(params.len());
                params.push(AbiType::Ptr);
            } else {
                push_abi_types_for_name(atom, &mut params);
            }
        }
    }
    Ok(AbiSig { params, ret, raw_ptr_params })
}

/// Whether `atom` is a type name the sig parser accepts (no silent i64 fallback).
fn is_known_atom(atom: &str) -> bool {
    matches!(
        atom,
        "i8" | "i16" | "i32" | "i64" | "isize"
            | "u8" | "u16" | "u32" | "u64" | "usize"
            | "f32" | "f64" | "bool" | "char" | "str" | "cstr" | "cbuf" | "u8[]"
    ) || SCALAR_ARRAY_TYPES.contains(&atom)
        || atom.starts_with('*')
}

/// Runtime argument value (type-erased carrier), one-to-one with `AbiSig.params`.
///
/// The invoker dispatches each `AbiSlot` to an integer slot or a floating-point
/// slot according to the signature.
#[derive(Clone, Copy)]
pub enum AbiSlot {
    /// Integer / bool / char / pointer (zero-extended to `u64`).
    /// Pointers are converted via `as u64` (`*mut c_void as u64`).
    Int(u64),
    /// `f64` bit pattern (`f32` is also promoted to `f64` before being passed in;
    /// the invoker decides how to load the register based on the signature).
    Float(f64),
    /// Pointer. Equivalent to `Int(ptr as u64)` but carries clearer semantics,
    /// making it convenient for the upper Marshal layer to mark.
    Ptr(*mut core::ffi::c_void),
}

/// Runtime return value (type-erased carrier), assembled by the invoker according
/// to `AbiSig.ret`.
#[derive(Clone, Copy, Debug)]
pub enum RetSlot {
    Void,
    Int(u64),
    Float(f64),
    Ptr(*mut core::ffi::c_void),
}
