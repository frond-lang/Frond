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
}

impl AbiSig {
    pub fn new(params: Vec<AbiType>, ret: AbiType) -> Self {
        Self { params, ret }
    }
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
