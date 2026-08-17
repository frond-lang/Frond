// =========================================================================
// Arena — Bucket + ValueArena + ValueTrait + equality + deep clone
// =========================================================================

use std::cell::RefCell;
use rustc_hash::FxHashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use rayon::prelude::*;
use wide::{i32x4, i64x4, CmpEq};

pub use super::Tag::ValueTag;

use super::value::*;
use super::Ops::SIMD_LANES;
use super::Ops::PARALLEL_THRESHOLD;
use super::Ops::par_chunk_size;

// =========================================================================
// Part 3: Bucket<T> + ValueArena (fully bucketed SoA storage)
// =========================================================================

/// Type bucket: contiguous storage of same-type values + parallel reference counting + free-list recycling.
struct Bucket<T: Clone> {
    data: Vec<T>,
    refcounts: Vec<u32>,
    free_list: Vec<u32>,
}

impl<T: Clone> Bucket<T> {
    fn new() -> Self {
        Self { data: Vec::new(), refcounts: Vec::new(), free_list: Vec::new() }
    }

    fn alloc(&mut self, val: T) -> u32 {
        if let Some(idx) = self.free_list.pop() {
            self.data[idx as usize] = val;
            self.refcounts[idx as usize] = 1;
            idx
        } else {
            let idx = self.data.len() as u32;
            self.data.push(val);
            self.refcounts.push(1);
            idx
        }
    }

    #[inline]
    fn get(&self, idx: u32) -> &T {
        &self.data[idx as usize]
    }

    #[inline]
    fn get_mut(&mut self, idx: u32) -> &mut T {
        &mut self.data[idx as usize]
    }

    /// Current number of allocated slots (including free/unreclaimed), used for FFI handle validity checks
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    fn inc_ref(&mut self, idx: u32) {
        self.refcounts[idx as usize] += 1;
    }

    fn dec_ref(&mut self, idx: u32) -> bool {
        let rc = &mut self.refcounts[idx as usize];
        *rc = rc.saturating_sub(1);
        if *rc == 0 {
            self.free_list.push(idx);
            true
        } else {
            false
        }
    }

    fn _refcount(&self, idx: u32) -> u32 {
        self.refcounts[idx as usize]
    }

    /// Clears all data, refcounts, and free-list of this bucket, reclaiming memory.
    fn reset(&mut self) {
        self.data.clear();
        self.refcounts.clear();
        self.free_list.clear();
    }
}

// =========================================================================
// Macros for eliminating repetitive 20-arm ValueTag → bucket dispatch.
// Adding a new scalar ValueTag only requires updating these macros.
// =========================================================================

/// Dispatches a single-argument method call to the correct bucket by ValueTag.
/// `Null | Void | Bool` are singletons (no-op). Used by inc_ref / dec_ref.
macro_rules! dispatch_bucket_method {
    ($self:expr, $tag:expr, $idx:expr, $method:ident) => {
        match $tag {
            ValueTag::Null | ValueTag::Void | ValueTag::Bool => {}
            ValueTag::Char => { $self.char_bucket.$method($idx); }
            ValueTag::I8 => { $self.i8_bucket.$method($idx); }
            ValueTag::I16 => { $self.i16_bucket.$method($idx); }
            ValueTag::I32 => { $self.i32_bucket.$method($idx); }
            ValueTag::I64 => { $self.i64_bucket.$method($idx); }
            ValueTag::I128 => { $self.i128_bucket.$method($idx); }
            ValueTag::U8 => { $self.u8_bucket.$method($idx); }
            ValueTag::U16 => { $self.u16_bucket.$method($idx); }
            ValueTag::U32 => { $self.u32_bucket.$method($idx); }
            ValueTag::U64 => { $self.u64_bucket.$method($idx); }
            ValueTag::U128 => { $self.u128_bucket.$method($idx); }
            ValueTag::Isize => { $self.isz_bucket.$method($idx); }
            ValueTag::Usize => { $self.usz_bucket.$method($idx); }
            ValueTag::F16 => { $self.f16_bucket.$method($idx); }
            ValueTag::F32 => { $self.f32_bucket.$method($idx); }
            ValueTag::F64 => { $self.f64_bucket.$method($idx); }
            ValueTag::F128 => { $self.f128_bucket.$method($idx); }
            ValueTag::Ref => { $self.ref_bucket.$method($idx); }
        }
    };
}

/// Checks `idx < bucket.len()` for the correct bucket by ValueTag.
/// `Null | Void | Bool` are singletons (always valid).
macro_rules! check_bucket_len {
    ($self:expr, $tag:expr, $idx:expr) => {
        match $tag {
            ValueTag::Null | ValueTag::Void | ValueTag::Bool => true,
            ValueTag::Char => $idx < $self.char_bucket.len(),
            ValueTag::I8 => $idx < $self.i8_bucket.len(),
            ValueTag::I16 => $idx < $self.i16_bucket.len(),
            ValueTag::I32 => $idx < $self.i32_bucket.len(),
            ValueTag::I64 => $idx < $self.i64_bucket.len(),
            ValueTag::I128 => $idx < $self.i128_bucket.len(),
            ValueTag::U8 => $idx < $self.u8_bucket.len(),
            ValueTag::U16 => $idx < $self.u16_bucket.len(),
            ValueTag::U32 => $idx < $self.u32_bucket.len(),
            ValueTag::U64 => $idx < $self.u64_bucket.len(),
            ValueTag::U128 => $idx < $self.u128_bucket.len(),
            ValueTag::Isize => $idx < $self.isz_bucket.len(),
            ValueTag::Usize => $idx < $self.usz_bucket.len(),
            ValueTag::F16 => $idx < $self.f16_bucket.len(),
            ValueTag::F32 => $idx < $self.f32_bucket.len(),
            ValueTag::F64 => $idx < $self.f64_bucket.len(),
            ValueTag::F128 => $idx < $self.f128_bucket.len(),
            ValueTag::Ref => $idx < $self.ref_bucket.len(),
        }
    };
}

/// Calls `.reset()` on every bucket.
macro_rules! reset_all_buckets {
    ($self:expr) => {
        $self.char_bucket.reset();
        $self.i8_bucket.reset();
        $self.i16_bucket.reset();
        $self.i32_bucket.reset();
        $self.i64_bucket.reset();
        $self.u8_bucket.reset();
        $self.u16_bucket.reset();
        $self.u32_bucket.reset();
        $self.u64_bucket.reset();
        $self.isz_bucket.reset();
        $self.usz_bucket.reset();
        $self.i128_bucket.reset();
        $self.u128_bucket.reset();
        $self.f16_bucket.reset();
        $self.f32_bucket.reset();
        $self.f64_bucket.reset();
        $self.f128_bucket.reset();
        $self.ref_bucket.reset();
    };
}

/// Unified storage for Value: bucketed by type (SoA), each scalar type stored in independent contiguous storage.
/// Heap objects (HeapObj) use Arc and are stored in ref_bucket.
pub struct ValueArena {
    char_bucket: Bucket<u32>,
    i8_bucket: Bucket<i8>,
    i16_bucket: Bucket<i16>,
    i32_bucket: Bucket<i32>,
    i64_bucket: Bucket<i64>,
    u8_bucket: Bucket<u8>,
    u16_bucket: Bucket<u16>,
    u32_bucket: Bucket<u32>,
    u64_bucket: Bucket<u64>,
    isz_bucket: Bucket<isize>,
    usz_bucket: Bucket<usize>,
    i128_bucket: Bucket<[u64; 2]>,
    u128_bucket: Bucket<[u64; 2]>,
    f16_bucket: Bucket<u16>,
    f32_bucket: Bucket<f32>,
    f64_bucket: Bucket<f64>,
    f128_bucket: Bucket<[u64; 2]>,
    ref_bucket: Bucket<Arc<HeapObj>>,
}

macro_rules! impl_scalar_bucket_methods {
    ($($tag:ident => $alloc:ident / $get:ident : $ty:ty, $bucket:ident);* $(;)?) => {
        impl ValueArena {
            $(
                #[inline]
                pub fn $alloc(&mut self, v: $ty) -> ValueHandle {
                    let idx = self.$bucket.alloc(v);
                    ValueHandle::new(ValueTag::$tag, idx as usize)
                }
                #[inline]
                pub fn $get(&self, h: ValueHandle) -> $ty {
                    *self.$bucket.get(h.index() as u32)
                }
            )*
        }
    };
}

impl_scalar_bucket_methods! {
    Char => alloc_char / get_char : u32, char_bucket;
    I8 => alloc_i8 / get_i8 : i8, i8_bucket;
    I16 => alloc_i16 / get_i16 : i16, i16_bucket;
    I32 => alloc_i32 / get_i32 : i32, i32_bucket;
    I64 => alloc_i64 / get_i64 : i64, i64_bucket;
    U8 => alloc_u8 / get_u8 : u8, u8_bucket;
    U16 => alloc_u16 / get_u16 : u16, u16_bucket;
    U32 => alloc_u32 / get_u32 : u32, u32_bucket;
    U64 => alloc_u64 / get_u64 : u64, u64_bucket;
    Isize => alloc_isize / get_isize : isize, isz_bucket;
    Usize => alloc_usize / get_usize : usize, usz_bucket;
    F16 => alloc_f16 / get_f16 : u16, f16_bucket;
    F32 => alloc_f32 / get_f32 : f32, f32_bucket;
    F64 => alloc_f64 / get_f64 : f64, f64_bucket;
}

impl ValueArena {
    // ─── Global arena access (for extern "C" reflection primitives) ──────────────
    // Frond is a single-threaded compiler; thread_local is sufficient.
    thread_local! {
        static GLOBAL_ARENA: RefCell<ValueArena> = RefCell::new(ValueArena::new());
    }

    /// Global arena read-only access
    pub fn with_global<R>(f: impl FnOnce(&ValueArena) -> R) -> R {
        Self::GLOBAL_ARENA.with(|cell| f(&cell.borrow()))
    }

    /// Global arena mutable access
    pub fn with_global_mut<R>(f: impl FnOnce(&mut ValueArena) -> R) -> R {
        Self::GLOBAL_ARENA.with(|cell| f(&mut cell.borrow_mut()))
    }

    /// Looks up the HeapObj from the global arena via a ValueHandle (core path for reflection primitives).
    /// Returns an Arc clone to avoid lifetime issues from returning a thread_local borrow across functions.
    pub fn get_global_obj(handle: ValueHandle) -> Option<Arc<HeapObj>> {
        if handle.tag() != ValueTag::Ref {
            return None;
        }
        Some(Self::with_global(|arena| arena.get_ref(handle).clone()))
    }

    /// Validates whether a handle points to a legal slot in the arena (FFI boundary defense).
    /// Scalars check the index range in the corresponding bucket; Null/Void/Bool are singletons and always valid.
    /// Used at extern "C" reflection primitive entry points to prevent out-of-bounds panics from dirty C-side handles.
    pub fn is_valid_handle(handle: ValueHandle) -> bool {
        Self::with_global(|arena| arena.is_valid_handle_inner(handle))
    }

    pub(crate) fn is_valid_handle_inner(&self, h: ValueHandle) -> bool {
        let idx = h.index();
        check_bucket_len!(self, h.tag(), idx)
    }

    pub fn new() -> Self {
        Self {
            char_bucket: Bucket::new(),
            i8_bucket: Bucket::new(),
            i16_bucket: Bucket::new(),
            i32_bucket: Bucket::new(),
            i64_bucket: Bucket::new(),
            u8_bucket: Bucket::new(),
            u16_bucket: Bucket::new(),
            u32_bucket: Bucket::new(),
            u64_bucket: Bucket::new(),
            isz_bucket: Bucket::new(),
            usz_bucket: Bucket::new(),
            i128_bucket: Bucket::new(),
            u128_bucket: Bucket::new(),
            f16_bucket: Bucket::new(),
            f32_bucket: Bucket::new(),
            f64_bucket: Bucket::new(),
            f128_bucket: Bucket::new(),
            ref_bucket: Bucket::new(),
        }
    }

    /// Resets the arena, clearing all buckets and reclaiming memory.
    /// Used for batch cleanup after reflection operations, preventing memory buildup from reflection primitive allocs with no dec_ref.
    pub fn reset(&mut self) {
        reset_all_buckets!(self);
    }

    #[inline]
    pub fn alloc_i128(&mut self, v: i128) -> ValueHandle {
        let idx = self.i128_bucket.alloc([(v as u128 & 0xFFFF_FFFF_FFFF_FFFF) as u64, ((v as u128) >> 64) as u64]);
        ValueHandle::new(ValueTag::I128, idx as usize)
    }
    #[inline]
    pub fn get_i128(&self, h: ValueHandle) -> i128 {
        let [lo, hi] = *self.i128_bucket.get(h.index() as u32);
        ((hi as i128) << 64) | (lo as i128)
    }

    #[inline]
    pub fn alloc_u128(&mut self, v: u128) -> ValueHandle {
        let idx = self.u128_bucket.alloc([(v & 0xFFFF_FFFF_FFFF_FFFF) as u64, (v >> 64) as u64]);
        ValueHandle::new(ValueTag::U128, idx as usize)
    }
    #[inline]
    pub fn get_u128(&self, h: ValueHandle) -> u128 {
        let [lo, hi] = *self.u128_bucket.get(h.index() as u32);
        ((hi as u128) << 64) | (lo as u128)
    }

    #[inline]
    pub fn alloc_f128(&mut self, v: F128) -> ValueHandle {
        let bits = u128::from_le_bytes(v.0);
        let idx = self.f128_bucket.alloc([(bits & 0xFFFF_FFFF_FFFF_FFFF) as u64, (bits >> 64) as u64]);
        ValueHandle::new(ValueTag::F128, idx as usize)
    }
    #[inline]
    pub fn get_f128(&self, h: ValueHandle) -> F128 {
        let [lo, hi] = *self.f128_bucket.get(h.index() as u32);
        F128(((hi as u128) << 64 | lo as u128).to_le_bytes())
    }

    // ---- Heap object allocation ----
    #[inline]
    pub fn alloc_ref(&mut self, obj: HeapObj) -> ValueHandle {
        let idx = self.ref_bucket.alloc(Arc::new(obj));
        ValueHandle::new(ValueTag::Ref, idx as usize)
    }
    #[inline]
    pub fn alloc_ref_rc(&mut self, r: Arc<HeapObj>) -> ValueHandle {
        let idx = self.ref_bucket.alloc(r);
        ValueHandle::new(ValueTag::Ref, idx as usize)
    }
    #[inline]
    pub fn get_ref(&self, h: ValueHandle) -> &Arc<HeapObj> {
        self.ref_bucket.get(h.index() as u32)
    }

    // ---- Bool singleton ----
    #[inline]
    pub fn bool_val(v: bool) -> ValueHandle {
        if v { ValueHandle::TRUE } else { ValueHandle::FALSE }
    }
    #[inline]
    pub fn get_bool(&self, h: ValueHandle) -> bool {
        h.index() == 1
    }

    // ---- Null/Void singletons ----
    #[inline]
    pub fn null(&self) -> ValueHandle {
        ValueHandle::NULL
    }
    #[inline]
    pub fn void(&self) -> ValueHandle {
        ValueHandle::VOID
    }

    // ---- Value ↔ ValueHandle conversion (for reflection FFI boundary) ----
    // Reflection primitives receive u32 (ValueHandle raw), but HeapObj fields have been migrated to Value.
    // alloc_value converts Value fields back to ValueHandle for FFI return;
    // get_value converts the entry ValueHandle to Value for internal recursive processing.

    /// Converts a Value to a ValueHandle (reflection FFI boundary: Value field → ValueHandle raw u32).
    /// Scalars are bucketed by tag; Bool/Null/Void use singletons; Ref uses ref_bucket.
    pub fn alloc_value(&mut self, v: &Value) -> ValueHandle {
        match v {
            Value::Null => ValueHandle::NULL,
            Value::Void => ValueHandle::VOID,
            Value::Scalar(sv, tag) => unsafe {
                match tag {
                    ValueTag::Bool => if sv.bool_val { ValueHandle::TRUE } else { ValueHandle::FALSE },
                    ValueTag::Char => self.alloc_char(sv.char_val),
                    ValueTag::I8 => self.alloc_i8(sv.i8_val),
                    ValueTag::I16 => self.alloc_i16(sv.i16_val),
                    ValueTag::I32 => self.alloc_i32(sv.i32_val),
                    ValueTag::I64 => self.alloc_i64(sv.i64_val),
                    ValueTag::U8 => self.alloc_u8(sv.u8_val),
                    ValueTag::U16 => self.alloc_u16(sv.u16_val),
                    ValueTag::U32 => self.alloc_u32(sv.u32_val),
                    ValueTag::U64 => self.alloc_u64(sv.u64_val),
                    ValueTag::Isize => self.alloc_isize(sv.isize_val),
                    ValueTag::Usize => self.alloc_usize(sv.usize_val),
                    ValueTag::I128 => self.alloc_i128(i128::from_ne_bytes(std::mem::transmute(sv.i128_val))),
                    ValueTag::U128 => self.alloc_u128(u128::from_ne_bytes(std::mem::transmute(sv.u128_val))),
                    ValueTag::F16 => self.alloc_f16(sv.f16_val),
                    ValueTag::F32 => self.alloc_f32(sv.f32_val),
                    ValueTag::F64 => self.alloc_f64(sv.f64_val),
                    ValueTag::F128 => self.alloc_f128(F128(std::mem::transmute(sv.f128_val))),
                    _ => unreachable!("non-scalar tag {:?} in ScalarValue", tag),
                }
            },
            Value::Ref(r) => self.alloc_ref_rc(r.clone()),
        }
    }

    /// Converts a ValueHandle to a Value (reflection FFI boundary: entry handle → Value for recursive processing).
    pub fn get_value(&self, h: ValueHandle) -> Value {
        match h.tag() {
            ValueTag::Null => Value::Null,
            ValueTag::Void => Value::Void,
            ValueTag::Bool => Value::bool_val(self.get_bool(h)),
            ValueTag::Char => Value::char_val(unsafe { char::from_u32_unchecked(self.get_char(h)) }),
            ValueTag::I8 => Value::i8(self.get_i8(h)),
            ValueTag::I16 => Value::i16(self.get_i16(h)),
            ValueTag::I32 => Value::i32(self.get_i32(h)),
            ValueTag::I64 => Value::i64(self.get_i64(h)),
            ValueTag::U8 => Value::u8(self.get_u8(h)),
            ValueTag::U16 => Value::u16(self.get_u16(h)),
            ValueTag::U32 => Value::u32(self.get_u32(h)),
            ValueTag::U64 => Value::u64(self.get_u64(h)),
            ValueTag::Isize => Value::isize_val(self.get_isize(h)),
            ValueTag::Usize => Value::usize_val(self.get_usize(h)),
            ValueTag::I128 => Value::i128(self.get_i128(h)),
            ValueTag::U128 => Value::u128(self.get_u128(h)),
            ValueTag::F16 => Value::f16(F16(self.get_f16(h))),
            ValueTag::F32 => Value::f32(self.get_f32(h)),
            ValueTag::F64 => Value::f64(self.get_f64(h)),
            ValueTag::F128 => Value::f128(self.get_f128(h)),
            ValueTag::Ref => Value::Ref(self.get_ref(h).clone()),
        }
    }

    // ---- Heap object convenience constructors ----
    pub fn alloc_str(&mut self, s: impl Into<String>) -> ValueHandle {
        self.alloc_ref(HeapObj::Str(Str::new(s)))
    }
    pub fn alloc_str_from(&mut self, s: &str) -> ValueHandle {
        self.alloc_ref(HeapObj::Str(Str::from_rust_str(s)))
    }
    pub fn alloc_array(&mut self, arr: ArrayValue) -> ValueHandle {
        self.alloc_ref(HeapObj::Array(arr))
    }
    pub fn alloc_record(&mut self, r: RecordValue) -> ValueHandle {
        self.alloc_ref(HeapObj::Record(r))
    }

    /// In-place modification of a record field (via Arc::make_mut; zero-copy when refcount==1).
    /// `field_name` is the field name to look up in record.field_names.
    pub fn set_record_field_by_name(
        &mut self,
        handle: ValueHandle,
        field_name: &str,
        new_value: Value,
    ) {
        let rc = self.ref_bucket.get_mut(handle.index() as u32);
        if let HeapObj::Record(ref mut r) = Arc::make_mut(rc) {
            for (i, name) in r.field_names.iter().enumerate() {
                if name.as_deref() == Some(field_name) {
                    if i < r.fields.len() {
                        r.fields[i] = new_value;
                    }
                    return;
                }
            }
        }
    }
    pub fn alloc_adt(&mut self, a: AdtValue) -> ValueHandle {
        self.alloc_ref(HeapObj::Adt(a))
    }
    pub fn alloc_newtype(&mut self, type_name: impl Into<String>, inner: ValueHandle) -> ValueHandle {
        self.alloc_ref(HeapObj::Newtype(NewtypeValue { type_name: type_name.into(), inner }))
    }
    pub fn alloc_cell(&mut self, val: Value) -> ValueHandle {
        self.alloc_ref(HeapObj::Cell(Cell::new(val)))
    }
    pub fn alloc_range(&mut self, start: i64, end: i64, inclusive: bool) -> ValueHandle {
        self.alloc_ref(HeapObj::Range(Range::new(start, end, inclusive)))
    }
    pub fn alloc_closure(&mut self, c: Closure) -> ValueHandle {
        self.alloc_ref(HeapObj::Closure(c))
    }
    pub fn alloc_partial(&mut self, p: PartialApplication) -> ValueHandle {
        self.alloc_ref(HeapObj::Partial(p))
    }
    pub fn alloc_builtin(&mut self, fn_ptr: BuiltinFn, name: impl Into<String>) -> ValueHandle {
        self.alloc_ref(HeapObj::Builtin(Builtin { fn_ptr, name: name.into() }))
    }
    pub fn alloc_trait_val(&mut self, t: TraitValue) -> ValueHandle {
        self.alloc_ref(HeapObj::TraitVal(t))
    }
    pub fn alloc_lazy(&mut self, l: LazyValue) -> ValueHandle {
        self.alloc_ref(HeapObj::LazyVal(l))
    }
    pub fn alloc_error_val(&mut self, type_name: impl Into<String>, message: impl Into<String>, is_error_subtype: bool) -> ValueHandle {
        self.alloc_ref(HeapObj::ErrorVal(ErrorValue {
            type_name: type_name.into(),
            message: message.into(),
            is_error_subtype,
        }))
    }
    pub fn alloc_throw_ok(&mut self, val: Value) -> ValueHandle {
        self.alloc_ref(HeapObj::ThrowVal(ThrowValue { payload: ThrowPayload::Ok(val) }))
    }
    pub fn alloc_throw_err(&mut self, err_val: Value) -> ValueHandle {
        self.alloc_ref(HeapObj::ThrowVal(ThrowValue { payload: ThrowPayload::Err(err_val) }))
    }
    pub fn alloc_atomic(&mut self, val: Value) -> ValueHandle {
        self.alloc_ref(HeapObj::AtomicVal(AtomicValue::new(val)))
    }
    pub fn alloc_async_handle(&mut self) -> ValueHandle {
        self.alloc_ref(HeapObj::AsyncVal(AsyncHandle::new()))
    }
    pub fn alloc_channel(&mut self, capacity: usize) -> ValueHandle {
        self.alloc_ref(HeapObj::ChannelVal(Arc::new(ChannelValue::new(capacity))))
    }
    pub fn alloc_sender(&mut self, channel: Arc<ChannelValue>) -> ValueHandle {
        self.alloc_ref(HeapObj::SenderVal(SenderValue { channel }))
    }
    pub fn alloc_receiver(&mut self, channel: Arc<ChannelValue>) -> ValueHandle {
        self.alloc_ref(HeapObj::ReceiverVal(ReceiverValue { channel }))
    }

    /// Constructs the corresponding integer type from an i64, selected by tag
    pub fn int_from_i64(&mut self, tag: ValueTag, v: i64) -> ValueHandle {
        match tag {
            ValueTag::I8 => self.alloc_i8(v as i8),
            ValueTag::I16 => self.alloc_i16(v as i16),
            ValueTag::I32 => self.alloc_i32(v as i32),
            ValueTag::I64 => self.alloc_i64(v),
            ValueTag::I128 => self.alloc_i128(v as i128),
            ValueTag::U8 => self.alloc_u8(v as u8),
            ValueTag::U16 => self.alloc_u16(v as u16),
            ValueTag::U32 => self.alloc_u32(v as u32),
            ValueTag::U64 => self.alloc_u64(v as u64),
            ValueTag::U128 => self.alloc_u128(v as u128),
            ValueTag::Isize => self.alloc_isize(v as isize),
            ValueTag::Usize => self.alloc_usize(v as usize),
            _ => self.alloc_i64(v),
        }
    }

    // ---- Reference counting ----
    pub fn inc_ref(&mut self, h: ValueHandle) {
        dispatch_bucket_method!(self, h.tag(), h.index() as u32, inc_ref);
    }
    pub fn dec_ref(&mut self, h: ValueHandle) {
        dispatch_bucket_method!(self, h.tag(), h.index() as u32, dec_ref);
    }

    /// Fills the SoA fast path for an array (when elements are same-type scalars).
    /// Elements have been migrated to Value; the Value's built-in scalar accessors are used directly, without going through arena buckets.
    pub fn optimize_array_soa(&mut self, arr: &mut ArrayValue) {
        if arr.elements.is_empty() { return; }
        // Take the first element's scalar tag; all elements must have the same tag to enable SoA
        let tag = match arr.elements[0].scalar_tag() {
            Some(t) => t,
            None => return,
        };
        if !arr.elements.iter().all(|h| h.scalar_tag() == Some(tag)) {
            return;
        }
        arr.scalar_soa = Some(match tag {
            ValueTag::I8 => ScalarSoA::I8(arr.elements.iter().map(|h| h.as_i8()).collect()),
            ValueTag::I16 => ScalarSoA::I16(arr.elements.iter().map(|h| h.as_i16()).collect()),
            ValueTag::I32 => ScalarSoA::I32(arr.elements.iter().map(|h| h.as_i32()).collect()),
            ValueTag::I64 => ScalarSoA::I64(arr.elements.iter().map(|h| h.as_i64()).collect()),
            ValueTag::U8 => ScalarSoA::U8(arr.elements.iter().map(|h| h.as_u8()).collect()),
            ValueTag::U16 => ScalarSoA::U16(arr.elements.iter().map(|h| h.as_u16()).collect()),
            ValueTag::U32 => ScalarSoA::U32(arr.elements.iter().map(|h| h.as_u32()).collect()),
            ValueTag::U64 => ScalarSoA::U64(arr.elements.iter().map(|h| h.as_u64()).collect()),
            ValueTag::Bool => ScalarSoA::Bool(arr.elements.iter().map(|h| h.as_bool()).collect()),
            ValueTag::Char => ScalarSoA::Char(arr.elements.iter().map(|h| h.as_char() as u32).collect()),
            ValueTag::F32 => ScalarSoA::F32(arr.elements.iter().map(|h| h.as_f32()).collect()),
            ValueTag::F64 => ScalarSoA::F64(arr.elements.iter().map(|h| h.as_f64()).collect()),
            ValueTag::I128 => ScalarSoA::I128(arr.elements.iter().map(|h| h.as_i128()).collect()),
            ValueTag::U128 => ScalarSoA::U128(arr.elements.iter().map(|h| h.as_u128()).collect()),
            ValueTag::Isize => ScalarSoA::Isize(arr.elements.iter().map(|h| h.as_isize()).collect()),
            ValueTag::Usize => ScalarSoA::Usize(arr.elements.iter().map(|h| h.as_usize()).collect()),
            // F16/F128: read the union bit pattern directly, without f64 intermediate (preserves NaN bit exactness)
            ValueTag::F16 => ScalarSoA::F16(arr.elements.iter().map(|h| {
                let Value::Scalar(sv, _) = h else { unreachable!("tag checked above") };
                unsafe { sv.f16_val }
            }).collect()),
            ValueTag::F128 => ScalarSoA::F128(arr.elements.iter().map(|h| {
                let Value::Scalar(sv, _) = h else { unreachable!("tag checked above") };
                unsafe { F128(std::mem::transmute(sv.f128_val)) }
            }).collect()),
            _ => return,
        });
    }

    /// Formats a value as a string (Display semantics)
    pub fn format_value(&self, h: ValueHandle) -> String {
        self.display_value(h).to_string()
    }

    /// Returns a wrapper that implements Display
    pub fn display_value<'a>(&'a self, h: ValueHandle) -> ValueDisplay<'a> {
        ValueDisplay { arena: self, handle: h }
    }

    /// Returns a wrapper that implements Debug
    pub fn debug_value<'a>(&'a self, h: ValueHandle) -> ValueDebug<'a> {
        ValueDebug { arena: self, handle: h }
    }
}

impl Default for ValueArena {
    fn default() -> Self {
        Self::new()
    }
}

/// Display wrapper: formats a value via the arena
pub struct ValueDisplay<'a> {
    arena: &'a ValueArena,
    handle: ValueHandle,
}

impl<'a> fmt::Display for ValueDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        arena_display(self.arena, self.handle, f)
    }
}

/// Debug wrapper: formats a value via the arena
pub struct ValueDebug<'a> {
    arena: &'a ValueArena,
    handle: ValueHandle,
}

impl<'a> fmt::Debug for ValueDebug<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        arena_debug(self.arena, self.handle, f)
    }
}

fn arena_display(arena: &ValueArena, h: ValueHandle, f: &mut fmt::Formatter) -> fmt::Result {
    match h.tag() {
        ValueTag::Null => write!(f, "null"),
        ValueTag::Void => write!(f, "()"),
        ValueTag::Bool => write!(f, "{}", arena.get_bool(h)),
        ValueTag::Char => write!(f, "{}", Char::from_codepoint_unchecked(arena.get_char(h))),
        ValueTag::I8 => write!(f, "{}", arena.get_i8(h)),
        ValueTag::I16 => write!(f, "{}", arena.get_i16(h)),
        ValueTag::I32 => write!(f, "{}", arena.get_i32(h)),
        ValueTag::I64 => write!(f, "{}", arena.get_i64(h)),
        ValueTag::I128 => write!(f, "{}", arena.get_i128(h)),
        ValueTag::U8 => write!(f, "{}", arena.get_u8(h)),
        ValueTag::U16 => write!(f, "{}", arena.get_u16(h)),
        ValueTag::U32 => write!(f, "{}", arena.get_u32(h)),
        ValueTag::U64 => write!(f, "{}", arena.get_u64(h)),
        ValueTag::U128 => write!(f, "{}", arena.get_u128(h)),
        ValueTag::Isize => write!(f, "{}", arena.get_isize(h)),
        ValueTag::Usize => write!(f, "{}", arena.get_usize(h)),
        ValueTag::F16 => write!(f, "{}", F16(arena.get_f16(h)).to_f32()),
        ValueTag::F32 => write!(f, "{}", arena.get_f32(h)),
        ValueTag::F64 => write!(f, "{}", arena.get_f64(h)),
        ValueTag::F128 => write!(f, "{}", arena.get_f128(h).to_f64()),
        ValueTag::Ref => match arena.get_ref(h).as_ref() {
            HeapObj::Str(s) => write!(f, "{}", s),
            other => write!(f, "{:?}", other),
        },
    }
}

fn arena_debug(arena: &ValueArena, h: ValueHandle, f: &mut fmt::Formatter) -> fmt::Result {
    match h.tag() {
        ValueTag::Null => write!(f, "null"),
        ValueTag::Void => write!(f, "()"),
        ValueTag::Bool => write!(f, "{}", arena.get_bool(h)),
        ValueTag::Char => write!(f, "'{}'", Char::from_codepoint_unchecked(arena.get_char(h))),
        ValueTag::I8 => write!(f, "{}i8", arena.get_i8(h)),
        ValueTag::I16 => write!(f, "{}i16", arena.get_i16(h)),
        ValueTag::I32 => write!(f, "{}", arena.get_i32(h)),
        ValueTag::I64 => write!(f, "{}i64", arena.get_i64(h)),
        ValueTag::I128 => write!(f, "{}i128", arena.get_i128(h)),
        ValueTag::U8 => write!(f, "{}u8", arena.get_u8(h)),
        ValueTag::U16 => write!(f, "{}u16", arena.get_u16(h)),
        ValueTag::U32 => write!(f, "{}u32", arena.get_u32(h)),
        ValueTag::U64 => write!(f, "{}u64", arena.get_u64(h)),
        ValueTag::U128 => write!(f, "{}u128", arena.get_u128(h)),
        ValueTag::Isize => write!(f, "{}isize", arena.get_isize(h)),
        ValueTag::Usize => write!(f, "{}usize", arena.get_usize(h)),
        ValueTag::F16 => write!(f, "{:?}", F16(arena.get_f16(h))),
        ValueTag::F32 => write!(f, "{}f32", arena.get_f32(h)),
        ValueTag::F64 => write!(f, "{}", arena.get_f64(h)),
        ValueTag::F128 => write!(f, "{:?}", arena.get_f128(h)),
        ValueTag::Ref => write!(f, "{:?}", arena.get_ref(h).as_ref()),
    }
}

// =========================================================================
// ValueTrait — unified external interface (methods carry &ValueArena)
// =========================================================================

/// Frond unified value trait: the external interface for all value types.
pub trait ValueTrait: Sized + Clone + Copy + PartialEq + Eq + Hash {
    // ---- Predicates (tag only, no arena needed) ----
    fn is_null(&self) -> bool;
    fn is_void(&self) -> bool;
    fn is_bool(&self) -> bool;
    fn is_char(&self) -> bool;
    fn is_int(&self) -> bool;
    fn is_float(&self) -> bool;
    fn is_numeric(&self) -> bool;
    fn is_scalar(&self) -> bool;
    fn is_ref(&self) -> bool;
    fn requires_release(&self) -> bool;

    // ---- Heap predicates (need arena to dereference HeapObj) ----
    fn is_string(&self, arena: &ValueArena) -> bool;
    fn is_array(&self, arena: &ValueArena) -> bool;
    fn is_record(&self, arena: &ValueArena) -> bool;
    fn is_adt(&self, arena: &ValueArena) -> bool;
    fn is_closure(&self, arena: &ValueArena) -> bool;
    fn is_callable(&self, arena: &ValueArena) -> bool;

    // ---- Type info ----
    fn type_name(&self, arena: &ValueArena) -> &'static str;
    fn scalar_tag(&self) -> Option<ValueTag>;

    // ---- Scalar accessors (need arena to read values) ----
    fn as_bool(&self, arena: &ValueArena) -> Option<bool>;
    fn as_i8(&self, arena: &ValueArena) -> Option<i8>;
    fn as_i16(&self, arena: &ValueArena) -> Option<i16>;
    fn as_i32(&self, arena: &ValueArena) -> Option<i32>;
    fn as_i64(&self, arena: &ValueArena) -> Option<i64>;
    fn as_i128(&self, arena: &ValueArena) -> Option<i128>;
    fn as_u8(&self, arena: &ValueArena) -> Option<u8>;
    fn as_u16(&self, arena: &ValueArena) -> Option<u16>;
    fn as_u32(&self, arena: &ValueArena) -> Option<u32>;
    fn as_u64(&self, arena: &ValueArena) -> Option<u64>;
    fn as_u128(&self, arena: &ValueArena) -> Option<u128>;
    fn as_isize(&self, arena: &ValueArena) -> Option<isize>;
    fn as_usize(&self, arena: &ValueArena) -> Option<usize>;
    fn as_f32(&self, arena: &ValueArena) -> Option<f32>;
    fn as_f64(&self, arena: &ValueArena) -> Option<f64>;
    fn as_char(&self, arena: &ValueArena) -> Option<Char>;
    fn as_f16(&self, arena: &ValueArena) -> Option<F16>;
    fn as_f128(&self, arena: &ValueArena) -> Option<F128>;

    // ---- Heap accessors ----
    fn as_str<'a>(&self, arena: &'a ValueArena) -> Option<&'a Str>;
    fn as_array<'a>(&self, arena: &'a ValueArena) -> Option<&'a ArrayValue>;
    fn as_record<'a>(&self, arena: &'a ValueArena) -> Option<&'a RecordValue>;
    fn as_adt<'a>(&self, arena: &'a ValueArena) -> Option<&'a AdtValue>;
    fn as_newtype<'a>(&self, arena: &'a ValueArena) -> Option<&'a NewtypeValue>;
    fn as_cell<'a>(&self, arena: &'a ValueArena) -> Option<&'a Cell>;
    fn as_range<'a>(&self, arena: &'a ValueArena) -> Option<&'a Range>;
    fn as_closure<'a>(&self, arena: &'a ValueArena) -> Option<&'a Closure>;
    fn as_partial<'a>(&self, arena: &'a ValueArena) -> Option<&'a PartialApplication>;
    fn as_builtin<'a>(&self, arena: &'a ValueArena) -> Option<&'a Builtin>;
    fn as_trait_val<'a>(&self, arena: &'a ValueArena) -> Option<&'a TraitValue>;
    fn as_lazy<'a>(&self, arena: &'a ValueArena) -> Option<&'a LazyValue>;
    fn as_error_val<'a>(&self, arena: &'a ValueArena) -> Option<&'a ErrorValue>;
    fn as_throw_val<'a>(&self, arena: &'a ValueArena) -> Option<&'a ThrowValue>;
    fn as_atomic<'a>(&self, arena: &'a ValueArena) -> Option<&'a AtomicValue>;
    fn as_async_handle<'a>(&self, arena: &'a ValueArena) -> Option<&'a AsyncHandle>;
    fn as_channel<'a>(&self, arena: &'a ValueArena) -> Option<&'a ChannelValue>;
    fn as_sender<'a>(&self, arena: &'a ValueArena) -> Option<&'a SenderValue>;
    fn as_receiver<'a>(&self, arena: &'a ValueArena) -> Option<&'a ReceiverValue>;
    fn as_ref<'a>(&self, arena: &'a ValueArena) -> Option<&'a HeapRef>;
    fn ref_kind(&self, arena: &ValueArena) -> Option<RefKind>;

    // ---- Numeric promotion ----
    fn as_int_i64(&self, arena: &ValueArena) -> Option<i64>;
    fn as_int_i128(&self, arena: &ValueArena) -> Option<i128>;
    fn as_float_f64(&self, arena: &ValueArena) -> Option<f64>;

    // ---- Equality and deep clone ----
    fn equals(&self, other: &Self, arena: &ValueArena) -> bool;
    fn deep_clone(&self, arena: &mut ValueArena) -> Self;
}

// =========================================================================
// =========================================================================
// ValueHandle — ValueTrait implementation (accesses bucket data via ValueArena)
// =========================================================================

impl ValueTrait for ValueHandle {
    // ---- Predicates (tag only, no arena needed) ----
    #[inline]
    fn is_null(&self) -> bool {
        self.tag() == ValueTag::Null
    }
    #[inline]
    fn is_void(&self) -> bool {
        self.tag() == ValueTag::Void
    }
    #[inline]
    fn is_bool(&self) -> bool {
        self.tag() == ValueTag::Bool
    }
    #[inline]
    fn is_char(&self) -> bool {
        self.tag() == ValueTag::Char
    }
    #[inline]
    fn is_int(&self) -> bool {
        self.tag().is_int()
    }
    #[inline]
    fn is_float(&self) -> bool {
        self.tag().is_float()
    }
    #[inline]
    fn is_numeric(&self) -> bool {
        self.tag().is_numeric()
    }
    #[inline]
    fn is_scalar(&self) -> bool {
        self.tag().is_scalar()
    }
    #[inline]
    fn is_ref(&self) -> bool {
        self.tag() == ValueTag::Ref
    }
    #[inline]
    fn requires_release(&self) -> bool {
        !matches!(self.tag(), ValueTag::Null | ValueTag::Void | ValueTag::Bool)
    }

    // ---- Heap predicates (need arena to dereference HeapObj) ----
    #[inline]
    fn is_string(&self, arena: &ValueArena) -> bool {
        matches!(arena.heap_obj_opt(*self), Some(HeapObj::Str(_)))
    }
    #[inline]
    fn is_array(&self, arena: &ValueArena) -> bool {
        matches!(arena.heap_obj_opt(*self), Some(HeapObj::Array(_)))
    }
    #[inline]
    fn is_record(&self, arena: &ValueArena) -> bool {
        matches!(arena.heap_obj_opt(*self), Some(HeapObj::Record(_)))
    }
    #[inline]
    fn is_adt(&self, arena: &ValueArena) -> bool {
        matches!(arena.heap_obj_opt(*self), Some(HeapObj::Adt(_)))
    }
    #[inline]
    fn is_closure(&self, arena: &ValueArena) -> bool {
        matches!(arena.heap_obj_opt(*self), Some(HeapObj::Closure(_)))
    }
    #[inline]
    fn is_callable(&self, arena: &ValueArena) -> bool {
        matches!(
            arena.heap_obj_opt(*self),
            Some(HeapObj::Closure(_) | HeapObj::Builtin(_) | HeapObj::Partial(_))
        )
    }

    // ---- Type info ----
    fn type_name(&self, arena: &ValueArena) -> &'static str {
        match self.tag() {
            ValueTag::Null => "null",
            ValueTag::Void => "void",
            ValueTag::Ref => arena.get_ref(*self).type_name(),
            t => t.name(),
        }
    }
    #[inline]
    fn scalar_tag(&self) -> Option<ValueTag> {
        let t = self.tag();
        if t.is_scalar() {
            Some(t)
        } else {
            None
        }
    }

    // ---- Scalar accessors (need arena to read values) ----
    #[inline]
    fn as_bool(&self, arena: &ValueArena) -> Option<bool> {
        if self.tag() == ValueTag::Bool {
            Some(arena.get_bool(*self))
        } else {
            None
        }
    }
    #[inline]
    fn as_i8(&self, arena: &ValueArena) -> Option<i8> {
        if self.tag() == ValueTag::I8 {
            Some(arena.get_i8(*self))
        } else {
            None
        }
    }
    #[inline]
    fn as_i16(&self, arena: &ValueArena) -> Option<i16> {
        if self.tag() == ValueTag::I16 {
            Some(arena.get_i16(*self))
        } else {
            None
        }
    }
    #[inline]
    fn as_i32(&self, arena: &ValueArena) -> Option<i32> {
        if self.tag() == ValueTag::I32 {
            Some(arena.get_i32(*self))
        } else {
            None
        }
    }
    #[inline]
    fn as_i64(&self, arena: &ValueArena) -> Option<i64> {
        if self.tag() == ValueTag::I64 {
            Some(arena.get_i64(*self))
        } else {
            None
        }
    }
    #[inline]
    fn as_i128(&self, arena: &ValueArena) -> Option<i128> {
        if self.tag() == ValueTag::I128 {
            Some(arena.get_i128(*self))
        } else {
            None
        }
    }
    #[inline]
    fn as_u8(&self, arena: &ValueArena) -> Option<u8> {
        if self.tag() == ValueTag::U8 {
            Some(arena.get_u8(*self))
        } else {
            None
        }
    }
    #[inline]
    fn as_u16(&self, arena: &ValueArena) -> Option<u16> {
        if self.tag() == ValueTag::U16 {
            Some(arena.get_u16(*self))
        } else {
            None
        }
    }
    #[inline]
    fn as_u32(&self, arena: &ValueArena) -> Option<u32> {
        if self.tag() == ValueTag::U32 {
            Some(arena.get_u32(*self))
        } else {
            None
        }
    }
    #[inline]
    fn as_u64(&self, arena: &ValueArena) -> Option<u64> {
        if self.tag() == ValueTag::U64 {
            Some(arena.get_u64(*self))
        } else {
            None
        }
    }
    #[inline]
    fn as_u128(&self, arena: &ValueArena) -> Option<u128> {
        if self.tag() == ValueTag::U128 {
            Some(arena.get_u128(*self))
        } else {
            None
        }
    }
    #[inline]
    fn as_isize(&self, arena: &ValueArena) -> Option<isize> {
        if self.tag() == ValueTag::Isize {
            Some(arena.get_isize(*self))
        } else {
            None
        }
    }
    #[inline]
    fn as_usize(&self, arena: &ValueArena) -> Option<usize> {
        if self.tag() == ValueTag::Usize {
            Some(arena.get_usize(*self))
        } else {
            None
        }
    }
    #[inline]
    fn as_f32(&self, arena: &ValueArena) -> Option<f32> {
        if self.tag() == ValueTag::F32 {
            Some(arena.get_f32(*self))
        } else {
            None
        }
    }
    #[inline]
    fn as_f64(&self, arena: &ValueArena) -> Option<f64> {
        if self.tag() == ValueTag::F64 {
            Some(arena.get_f64(*self))
        } else {
            None
        }
    }
    #[inline]
    fn as_char(&self, arena: &ValueArena) -> Option<Char> {
        if self.tag() == ValueTag::Char {
            Some(Char::from_codepoint_unchecked(arena.get_char(*self)))
        } else {
            None
        }
    }
    #[inline]
    fn as_f16(&self, arena: &ValueArena) -> Option<F16> {
        if self.tag() == ValueTag::F16 {
            Some(F16(arena.get_f16(*self)))
        } else {
            None
        }
    }
    #[inline]
    fn as_f128(&self, arena: &ValueArena) -> Option<F128> {
        if self.tag() == ValueTag::F128 {
            Some(arena.get_f128(*self))
        } else {
            None
        }
    }

    // ---- Heap accessors ----
    #[inline]
    fn as_str<'a>(&self, arena: &'a ValueArena) -> Option<&'a Str> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::Str(s) => Some(s),
            _ => None,
        }
    }
    #[inline]
    fn as_array<'a>(&self, arena: &'a ValueArena) -> Option<&'a ArrayValue> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::Array(a) => Some(a),
            _ => None,
        }
    }
    #[inline]
    fn as_record<'a>(&self, arena: &'a ValueArena) -> Option<&'a RecordValue> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::Record(r) => Some(r),
            _ => None,
        }
    }
    #[inline]
    fn as_adt<'a>(&self, arena: &'a ValueArena) -> Option<&'a AdtValue> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::Adt(a) => Some(a),
            _ => None,
        }
    }
    #[inline]
    fn as_newtype<'a>(&self, arena: &'a ValueArena) -> Option<&'a NewtypeValue> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::Newtype(n) => Some(n),
            _ => None,
        }
    }
    #[inline]
    fn as_cell<'a>(&self, arena: &'a ValueArena) -> Option<&'a Cell> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::Cell(c) => Some(c),
            _ => None,
        }
    }
    #[inline]
    fn as_range<'a>(&self, arena: &'a ValueArena) -> Option<&'a Range> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::Range(r) => Some(r),
            _ => None,
        }
    }
    #[inline]
    fn as_closure<'a>(&self, arena: &'a ValueArena) -> Option<&'a Closure> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::Closure(c) => Some(c),
            _ => None,
        }
    }
    #[inline]
    fn as_partial<'a>(&self, arena: &'a ValueArena) -> Option<&'a PartialApplication> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::Partial(p) => Some(p),
            _ => None,
        }
    }
    #[inline]
    fn as_builtin<'a>(&self, arena: &'a ValueArena) -> Option<&'a Builtin> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::Builtin(b) => Some(b),
            _ => None,
        }
    }
    #[inline]
    fn as_trait_val<'a>(&self, arena: &'a ValueArena) -> Option<&'a TraitValue> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::TraitVal(t) => Some(t),
            _ => None,
        }
    }
    #[inline]
    fn as_lazy<'a>(&self, arena: &'a ValueArena) -> Option<&'a LazyValue> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::LazyVal(l) => Some(l),
            _ => None,
        }
    }
    #[inline]
    fn as_error_val<'a>(&self, arena: &'a ValueArena) -> Option<&'a ErrorValue> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::ErrorVal(e) => Some(e),
            _ => None,
        }
    }
    #[inline]
    fn as_throw_val<'a>(&self, arena: &'a ValueArena) -> Option<&'a ThrowValue> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::ThrowVal(t) => Some(t),
            _ => None,
        }
    }
    #[inline]
    fn as_atomic<'a>(&self, arena: &'a ValueArena) -> Option<&'a AtomicValue> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::AtomicVal(a) => Some(a),
            _ => None,
        }
    }
    #[inline]
    fn as_async_handle<'a>(&self, arena: &'a ValueArena) -> Option<&'a AsyncHandle> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::AsyncVal(a) => Some(a),
            _ => None,
        }
    }
    #[inline]
    fn as_channel<'a>(&self, arena: &'a ValueArena) -> Option<&'a ChannelValue> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::ChannelVal(c) => Some(c.as_ref()),
            _ => None,
        }
    }
    #[inline]
    fn as_sender<'a>(&self, arena: &'a ValueArena) -> Option<&'a SenderValue> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::SenderVal(s) => Some(s),
            _ => None,
        }
    }
    #[inline]
    fn as_receiver<'a>(&self, arena: &'a ValueArena) -> Option<&'a ReceiverValue> {
        match arena.heap_obj_opt(*self)? {
            HeapObj::ReceiverVal(r) => Some(r),
            _ => None,
        }
    }
    #[inline]
    fn as_ref<'a>(&self, arena: &'a ValueArena) -> Option<&'a HeapRef> {
        if self.tag() == ValueTag::Ref {
            Some(arena.get_ref(*self))
        } else {
            None
        }
    }
    #[inline]
    fn ref_kind(&self, arena: &ValueArena) -> Option<RefKind> {
        arena.heap_obj_opt(*self).map(|o| o.ref_kind())
    }

    // ---- Numeric promotion ----
    fn as_int_i64(&self, arena: &ValueArena) -> Option<i64> {
        match self.tag() {
            ValueTag::I8 => Some(arena.get_i8(*self) as i64),
            ValueTag::I16 => Some(arena.get_i16(*self) as i64),
            ValueTag::I32 => Some(arena.get_i32(*self) as i64),
            ValueTag::I64 => Some(arena.get_i64(*self)),
            ValueTag::I128 => Some(arena.get_i128(*self) as i64),
            ValueTag::U8 => Some(arena.get_u8(*self) as i64),
            ValueTag::U16 => Some(arena.get_u16(*self) as i64),
            ValueTag::U32 => Some(arena.get_u32(*self) as i64),
            ValueTag::U64 => Some(arena.get_u64(*self) as i64),
            ValueTag::U128 => Some(arena.get_u128(*self) as i64),
            ValueTag::Isize => Some(arena.get_isize(*self) as i64),
            ValueTag::Usize => Some(arena.get_usize(*self) as i64),
            _ => None,
        }
    }
    fn as_int_i128(&self, arena: &ValueArena) -> Option<i128> {
        match self.tag() {
            ValueTag::I8 => Some(arena.get_i8(*self) as i128),
            ValueTag::I16 => Some(arena.get_i16(*self) as i128),
            ValueTag::I32 => Some(arena.get_i32(*self) as i128),
            ValueTag::I64 => Some(arena.get_i64(*self) as i128),
            ValueTag::I128 => Some(arena.get_i128(*self)),
            ValueTag::U8 => Some(arena.get_u8(*self) as i128),
            ValueTag::U16 => Some(arena.get_u16(*self) as i128),
            ValueTag::U32 => Some(arena.get_u32(*self) as i128),
            ValueTag::U64 => Some(arena.get_u64(*self) as i128),
            ValueTag::U128 => Some(arena.get_u128(*self) as i128),
            ValueTag::Isize => Some(arena.get_isize(*self) as i128),
            ValueTag::Usize => Some(arena.get_usize(*self) as i128),
            _ => None,
        }
    }
    fn as_float_f64(&self, arena: &ValueArena) -> Option<f64> {
        match self.tag() {
            ValueTag::F16 => Some(F16(arena.get_f16(*self)).to_f64()),
            ValueTag::F32 => Some(arena.get_f32(*self) as f64),
            ValueTag::F64 => Some(arena.get_f64(*self)),
            ValueTag::F128 => Some(arena.get_f128(*self).to_f64()),
            _ => None,
        }
    }

    // ---- Equality and deep clone ----
    fn equals(&self, other: &Self, arena: &ValueArena) -> bool {
        if self.tag() != other.tag() {
            return false;
        }
        match self.tag() {
            ValueTag::Null | ValueTag::Void => true,
            ValueTag::Bool => arena.get_bool(*self) == arena.get_bool(*other),
            ValueTag::Char => arena.get_char(*self) == arena.get_char(*other),
            ValueTag::I8 => arena.get_i8(*self) == arena.get_i8(*other),
            ValueTag::I16 => arena.get_i16(*self) == arena.get_i16(*other),
            ValueTag::I32 => arena.get_i32(*self) == arena.get_i32(*other),
            ValueTag::I64 => arena.get_i64(*self) == arena.get_i64(*other),
            ValueTag::I128 => arena.get_i128(*self) == arena.get_i128(*other),
            ValueTag::U8 => arena.get_u8(*self) == arena.get_u8(*other),
            ValueTag::U16 => arena.get_u16(*self) == arena.get_u16(*other),
            ValueTag::U32 => arena.get_u32(*self) == arena.get_u32(*other),
            ValueTag::U64 => arena.get_u64(*self) == arena.get_u64(*other),
            ValueTag::U128 => arena.get_u128(*self) == arena.get_u128(*other),
            ValueTag::Isize => arena.get_isize(*self) == arena.get_isize(*other),
            ValueTag::Usize => arena.get_usize(*self) == arena.get_usize(*other),
            ValueTag::F16 => arena.get_f16(*self) == arena.get_f16(*other),
            ValueTag::F32 => {
                arena.get_f32(*self).to_bits() == arena.get_f32(*other).to_bits()
            }
            ValueTag::F64 => {
                arena.get_f64(*self).to_bits() == arena.get_f64(*other).to_bits()
            }
            ValueTag::F128 => arena.get_f128(*self) == arena.get_f128(*other),
            ValueTag::Ref => {
                let a = arena.get_ref(*self);
                let b = arena.get_ref(*other);
                Arc::ptr_eq(a, b) || heap_equals(a, b, arena)
            }
        }
    }

    fn deep_clone(&self, arena: &mut ValueArena) -> Self {
        let mut cache = DeepCloneCache { handle: FxHashMap::default(), value: FxHashMap::default() };
        deep_clone_handle(*self, arena, &mut cache)
    }
}

// =========================================================================
// Heap object deep equality and deep clone (with ptr_eq cache for subgraph sharing)
// =========================================================================

// -------------------- SoA SIMD fast path --------------------

/// Attempts a SIMD batch comparison of two SoA arrays.
/// Only takes effect when both sides share the same SoA type, returning `Some(bool)`.
/// Returns `None` on type mismatch, leaving the caller to fall back to the element-wise path.
fn try_simd_soa_equals(a: &ScalarSoA, b: &ScalarSoA) -> Option<bool> {
    match (a, b) {
        (ScalarSoA::I32(va), ScalarSoA::I32(vb)) => Some(simd_eq_i32(va, vb)),
        (ScalarSoA::I64(va), ScalarSoA::I64(vb)) => Some(simd_eq_i64(va, vb)),
        (ScalarSoA::F32(va), ScalarSoA::F32(vb)) => Some(simd_eq_f32_bits(va, vb)),
        (ScalarSoA::F64(va), ScalarSoA::F64(vb)) => Some(simd_eq_f64_bits(va, vb)),
        // Remaining types use plain slice comparison (Rust slice PartialEq is already optimized)
        (ScalarSoA::I8(va), ScalarSoA::I8(vb)) => Some(va == vb),
        (ScalarSoA::I16(va), ScalarSoA::I16(vb)) => Some(va == vb),
        (ScalarSoA::U8(va), ScalarSoA::U8(vb)) => Some(va == vb),
        (ScalarSoA::U16(va), ScalarSoA::U16(vb)) => Some(va == vb),
        (ScalarSoA::U32(va), ScalarSoA::U32(vb)) => Some(va == vb),
        (ScalarSoA::U64(va), ScalarSoA::U64(vb)) => Some(va == vb),
        (ScalarSoA::Bool(va), ScalarSoA::Bool(vb)) => Some(va == vb),
        (ScalarSoA::Char(va), ScalarSoA::Char(vb)) => Some(va == vb),
        (ScalarSoA::I128(va), ScalarSoA::I128(vb)) => Some(va == vb),
        (ScalarSoA::U128(va), ScalarSoA::U128(vb)) => Some(va == vb),
        (ScalarSoA::Isize(va), ScalarSoA::Isize(vb)) => Some(va == vb),
        (ScalarSoA::Usize(va), ScalarSoA::Usize(vb)) => Some(va == vb),
        // F16/F128 compared by bit pattern (consistent with F32/F64 to_bits() semantics; NaN == NaN is true)
        (ScalarSoA::F16(va), ScalarSoA::F16(vb)) => Some(va == vb),
        (ScalarSoA::F128(va), ScalarSoA::F128(vb)) => Some(va == vb),
        _ => None, // type mismatch, fall back
    }
}

#[inline]
fn simd_eq_i32(a: &[i32], b: &[i32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let n = a.len();
    if n == 0 {
        return true;
    }
    if n >= PARALLEL_THRESHOLD {
        let chunk = par_chunk_size(n);
        return a.par_chunks(chunk)
            .zip(b.par_chunks(chunk))
            .all(|(ca, cb)| simd_eq_i32_chunk(ca, cb));
    }
    simd_eq_i32_chunk(a, b)
}

#[inline]
fn simd_eq_i32_chunk(a: &[i32], b: &[i32]) -> bool {
    let n = a.len();
    let simd_len = (n / SIMD_LANES) * SIMD_LANES;
    for (ca, cb) in a[..simd_len]
        .chunks_exact(SIMD_LANES)
        .zip(b[..simd_len].chunks_exact(SIMD_LANES))
    {
        let va = i32x4::new(ca.try_into().unwrap());
        let vb = i32x4::new(cb.try_into().unwrap());
        let mask = va.cmp_eq(vb);
        let arr = mask.to_array();
        if arr.contains(&0) {
            return false;
        }
    }
    a[simd_len..]
        .iter()
        .zip(&b[simd_len..])
        .all(|(&x, &y)| x == y)
}

#[inline]
fn simd_eq_i64(a: &[i64], b: &[i64]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let n = a.len();
    if n == 0 {
        return true;
    }
    if n >= PARALLEL_THRESHOLD {
        let chunk = par_chunk_size(n);
        return a.par_chunks(chunk)
            .zip(b.par_chunks(chunk))
            .all(|(ca, cb)| simd_eq_i64_chunk(ca, cb));
    }
    simd_eq_i64_chunk(a, b)
}

#[inline]
fn simd_eq_i64_chunk(a: &[i64], b: &[i64]) -> bool {
    let n = a.len();
    let simd_len = (n / SIMD_LANES) * SIMD_LANES;
    for (ca, cb) in a[..simd_len]
        .chunks_exact(SIMD_LANES)
        .zip(b[..simd_len].chunks_exact(SIMD_LANES))
    {
        let va = i64x4::new(ca.try_into().unwrap());
        let vb = i64x4::new(cb.try_into().unwrap());
        let mask = va.cmp_eq(vb);
        let arr = mask.to_array();
        if arr.contains(&0) {
            return false;
        }
    }
    a[simd_len..]
        .iter()
        .zip(&b[simd_len..])
        .all(|(&x, &y)| x == y)
}

/// f32 bitwise comparison (avoids NaN inequality issues).
#[inline]
fn simd_eq_f32_bits(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let n = a.len();
    if n == 0 {
        return true;
    }
    if n >= PARALLEL_THRESHOLD {
        let chunk = par_chunk_size(n);
        return a.par_chunks(chunk)
            .zip(b.par_chunks(chunk))
            .all(|(ca, cb)| simd_eq_f32_bits_chunk(ca, cb));
    }
    simd_eq_f32_bits_chunk(a, b)
}

#[inline]
fn simd_eq_f32_bits_chunk(a: &[f32], b: &[f32]) -> bool {
    let n = a.len();
    let simd_len = (n / SIMD_LANES) * SIMD_LANES;
    for (ca, cb) in a[..simd_len]
        .chunks_exact(SIMD_LANES)
        .zip(b[..simd_len].chunks_exact(SIMD_LANES))
    {
        let va = i32x4::new([
            ca[0].to_bits() as i32,
            ca[1].to_bits() as i32,
            ca[2].to_bits() as i32,
            ca[3].to_bits() as i32,
        ]);
        let vb = i32x4::new([
            cb[0].to_bits() as i32,
            cb[1].to_bits() as i32,
            cb[2].to_bits() as i32,
            cb[3].to_bits() as i32,
        ]);
        let mask = va.cmp_eq(vb);
        let arr = mask.to_array();
        if arr.contains(&0) {
            return false;
        }
    }
    a[simd_len..]
        .iter()
        .zip(&b[simd_len..])
        .all(|(&x, &y)| x.to_bits() == y.to_bits())
}

/// f64 bitwise comparison (avoids NaN inequality issues).
#[inline]
fn simd_eq_f64_bits(a: &[f64], b: &[f64]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let n = a.len();
    if n == 0 {
        return true;
    }
    if n >= PARALLEL_THRESHOLD {
        let chunk = par_chunk_size(n);
        return a.par_chunks(chunk)
            .zip(b.par_chunks(chunk))
            .all(|(ca, cb)| simd_eq_f64_bits_chunk(ca, cb));
    }
    simd_eq_f64_bits_chunk(a, b)
}

#[inline]
fn simd_eq_f64_bits_chunk(a: &[f64], b: &[f64]) -> bool {
    let n = a.len();
    let simd_len = (n / SIMD_LANES) * SIMD_LANES;
    for (ca, cb) in a[..simd_len]
        .chunks_exact(SIMD_LANES)
        .zip(b[..simd_len].chunks_exact(SIMD_LANES))
    {
        let va = i64x4::new([
            ca[0].to_bits() as i64,
            ca[1].to_bits() as i64,
            ca[2].to_bits() as i64,
            ca[3].to_bits() as i64,
        ]);
        let vb = i64x4::new([
            cb[0].to_bits() as i64,
            cb[1].to_bits() as i64,
            cb[2].to_bits() as i64,
            cb[3].to_bits() as i64,
        ]);
        let mask = va.cmp_eq(vb);
        let arr = mask.to_array();
        if arr.contains(&0) {
            return false;
        }
    }
    a[simd_len..]
        .iter()
        .zip(&b[simd_len..])
        .all(|(&x, &y)| x.to_bits() == y.to_bits())
}

// -------------------- SoA SIMD batch hash --------------------

/// Hashes SoA data in SIMD batches.
/// I32/I64/F32/F64 take the SIMD accumulation path; remaining types fall back to per-element hashing.
pub fn simd_hash_soa<H: Hasher>(soa: &ScalarSoA, state: &mut H) {
    match soa {
        ScalarSoA::I32(v) => simd_hash_i32(v, state),
        ScalarSoA::I64(v) => simd_hash_i64(v, state),
        ScalarSoA::F32(v) => simd_hash_f32(v, state),
        ScalarSoA::F64(v) => simd_hash_f64(v, state),
        ScalarSoA::I8(v) => v.iter().for_each(|x| x.hash(state)),
        ScalarSoA::I16(v) => v.iter().for_each(|x| x.hash(state)),
        ScalarSoA::U8(v) => v.iter().for_each(|x| x.hash(state)),
        ScalarSoA::U16(v) => v.iter().for_each(|x| x.hash(state)),
        ScalarSoA::U32(v) => v.iter().for_each(|x| x.hash(state)),
        ScalarSoA::U64(v) => v.iter().for_each(|x| x.hash(state)),
        ScalarSoA::Bool(v) => v.iter().for_each(|x| x.hash(state)),
        ScalarSoA::Char(v) => v.iter().for_each(|x| x.hash(state)),
        ScalarSoA::I128(v) => v.iter().for_each(|x| x.hash(state)),
        ScalarSoA::U128(v) => v.iter().for_each(|x| x.hash(state)),
        ScalarSoA::Isize(v) => v.iter().for_each(|x| x.hash(state)),
        ScalarSoA::Usize(v) => v.iter().for_each(|x| x.hash(state)),
        ScalarSoA::F16(v) => v.iter().for_each(|x| x.hash(state)),
        ScalarSoA::F128(v) => v.iter().for_each(|x| x.hash(state)),
    }
}

fn simd_hash_i32<H: Hasher>(v: &[i32], state: &mut H) {
    let n = v.len();
    let simd_len = (n / SIMD_LANES) * SIMD_LANES;
    let mut acc = i32x4::splat(0);
    for chunk in v[..simd_len].chunks_exact(SIMD_LANES) {
        let c = i32x4::new(chunk.try_into().unwrap());
        acc = (acc << 1) ^ c;
    }
    for x in acc.to_array() {
        x.hash(state);
    }
    for &x in &v[simd_len..] {
        x.hash(state);
    }
}

fn simd_hash_i64<H: Hasher>(v: &[i64], state: &mut H) {
    let n = v.len();
    let simd_len = (n / SIMD_LANES) * SIMD_LANES;
    let mut acc = i64x4::splat(0);
    for chunk in v[..simd_len].chunks_exact(SIMD_LANES) {
        let c = i64x4::new(chunk.try_into().unwrap());
        acc = (acc << 1) ^ c;
    }
    for x in acc.to_array() {
        x.hash(state);
    }
    for &x in &v[simd_len..] {
        x.hash(state);
    }
}

fn simd_hash_f32<H: Hasher>(v: &[f32], state: &mut H) {
    let n = v.len();
    let simd_len = (n / SIMD_LANES) * SIMD_LANES;
    let mut acc = i32x4::splat(0);
    for chunk in v[..simd_len].chunks_exact(SIMD_LANES) {
        let c = i32x4::new([
            chunk[0].to_bits() as i32,
            chunk[1].to_bits() as i32,
            chunk[2].to_bits() as i32,
            chunk[3].to_bits() as i32,
        ]);
        acc = (acc << 1) ^ c;
    }
    for x in acc.to_array() {
        x.hash(state);
    }
    for &x in &v[simd_len..] {
        x.to_bits().hash(state);
    }
}

fn simd_hash_f64<H: Hasher>(v: &[f64], state: &mut H) {
    let n = v.len();
    let simd_len = (n / SIMD_LANES) * SIMD_LANES;
    let mut acc = i64x4::splat(0);
    for chunk in v[..simd_len].chunks_exact(SIMD_LANES) {
        let c = i64x4::new([
            chunk[0].to_bits() as i64,
            chunk[1].to_bits() as i64,
            chunk[2].to_bits() as i64,
            chunk[3].to_bits() as i64,
        ]);
        acc = (acc << 1) ^ c;
    }
    for x in acc.to_array() {
        x.hash(state);
    }
    for &x in &v[simd_len..] {
        x.to_bits().hash(state);
    }
}

// -------------------- SoA deep_clone fast path --------------------

/// SoA fast-path deep clone: scalars are `Copy`, so they are rebuilt inline via the `Value` constructors without going through arena buckets.
fn simd_soa_deep_clone(soa: &ScalarSoA) -> Vec<Value> {
    match soa {
        ScalarSoA::I32(v) => v.iter().map(|&x| Value::i32(x)).collect(),
        ScalarSoA::I64(v) => v.iter().map(|&x| Value::i64(x)).collect(),
        ScalarSoA::F32(v) => v.iter().map(|&x| Value::f32(x)).collect(),
        ScalarSoA::F64(v) => v.iter().map(|&x| Value::f64(x)).collect(),
        ScalarSoA::I8(v) => v.iter().map(|&x| Value::i8(x)).collect(),
        ScalarSoA::I16(v) => v.iter().map(|&x| Value::i16(x)).collect(),
        ScalarSoA::U8(v) => v.iter().map(|&x| Value::u8(x)).collect(),
        ScalarSoA::U16(v) => v.iter().map(|&x| Value::u16(x)).collect(),
        ScalarSoA::U32(v) => v.iter().map(|&x| Value::u32(x)).collect(),
        ScalarSoA::U64(v) => v.iter().map(|&x| Value::u64(x)).collect(),
        ScalarSoA::Bool(v) => v.iter().map(|&x| Value::bool_val(x)).collect(),
        ScalarSoA::Char(v) => v.iter().map(|&x| Value::char_val(char::from_u32(x).unwrap_or('\0'))).collect(),
        ScalarSoA::I128(v) => v.iter().map(|&x| Value::i128(x)).collect(),
        ScalarSoA::U128(v) => v.iter().map(|&x| Value::u128(x)).collect(),
        ScalarSoA::Isize(v) => v.iter().map(|&x| Value::isize_val(x)).collect(),
        ScalarSoA::Usize(v) => v.iter().map(|&x| Value::usize_val(x)).collect(),
        ScalarSoA::F16(v) => v.iter().map(|&x| Value::f16(F16(x))).collect(),
        ScalarSoA::F128(v) => v.iter().map(|&x| Value::f128(x)).collect(),
    }
}

pub fn heap_equals(a: &HeapObj, b: &HeapObj, arena: &ValueArena) -> bool {
    match (a, b) {
        (HeapObj::Str(x), HeapObj::Str(y)) => x.equals(y),
        (HeapObj::Array(x), HeapObj::Array(y)) => {
            if x.fixed_size != y.fixed_size || x.elements.len() != y.elements.len() {
                return false;
            }
            // SoA SIMD fast path: both sides have scalar_soa of the same type
            if let (Some(sa), Some(sb)) = (&x.scalar_soa, &y.scalar_soa) {
                if let Some(result) = try_simd_soa_equals(sa, sb) {
                    return result;
                }
            }
            // Fall back: element-wise comparison (elements are Values)
            x.elements
                .iter()
                .zip(&y.elements)
                .all(|(p, q)| value_equals_with_arena(p, q, arena))
        }
        (HeapObj::Record(x), HeapObj::Record(y)) => {
            x.type_name == y.type_name
                && x.field_names == y.field_names
                && x.fields.len() == y.fields.len()
                && x.fields.iter().zip(&y.fields).all(|(p, q)| value_equals_with_arena(p, q, arena))
        }
        (HeapObj::Adt(x), HeapObj::Adt(y)) => {
            x.type_name == y.type_name
                && x.constructor == y.constructor
                && x.fields.len() == y.fields.len()
                && x
                    .fields
                    .iter()
                    .zip(&y.fields)
                    .all(|(xf, yf)| value_equals_with_arena(&xf.value, &yf.value, arena))
        }
        (HeapObj::Newtype(x), HeapObj::Newtype(y)) => {
            x.type_name == y.type_name && x.inner.equals(&y.inner, arena)
        }
        (HeapObj::Cell(x), HeapObj::Cell(y)) => {
            let xb = x.inner.lock().clone();
            let yb = y.inner.lock().clone();
            value_equals_with_arena(&xb, &yb, arena)
        }
        (HeapObj::Range(x), HeapObj::Range(y)) => {
            x.start == y.start && x.end == y.end && x.inclusive == y.inclusive
        }
        (HeapObj::ErrorVal(x), HeapObj::ErrorVal(y)) => {
            x.type_name == y.type_name
                && x.message == y.message
                && x.is_error_subtype == y.is_error_subtype
        }
        (HeapObj::ThrowVal(x), HeapObj::ThrowVal(y)) => match (&x.payload, &y.payload) {
            (ThrowPayload::Ok(a), ThrowPayload::Ok(b)) => value_equals_with_arena(a, b, arena),
            (ThrowPayload::Err(a), ThrowPayload::Err(b)) => value_equals_with_arena(a, b, arena),
            _ => false,
        },
        (HeapObj::Closure(x), HeapObj::Closure(y)) => {
            x.func_id == y.func_id
                && x.arity == y.arity
                && x.upvalues.len() == y.upvalues.len()
                && x
                    .upvalues
                    .iter()
                    .zip(&y.upvalues)
                    .all(|(p, q)| value_equals_with_arena(p, q, arena))
        }
        (HeapObj::Builtin(x), HeapObj::Builtin(y)) => {
            (x.fn_ptr as usize) == (y.fn_ptr as usize) && x.name == y.name
        }
        (HeapObj::Partial(x), HeapObj::Partial(y)) => {
            x.func_id == y.func_id
                && x.remaining_arity == y.remaining_arity
                && x.upvalues.len() == y.upvalues.len()
                && x.bound_args.len() == y.bound_args.len()
                && x.upvalues.iter().zip(&y.upvalues).all(|(p, q)| value_equals_with_arena(p, q, arena))
                && x.bound_args.iter().zip(&y.bound_args).all(|(p, q)| value_equals_with_arena(p, q, arena))
        }
        (HeapObj::TraitVal(x), HeapObj::TraitVal(y)) => {
            x.trait_name == y.trait_name
                && x.method_names == y.method_names
                && x.method_values.len() == y.method_values.len()
                && x.method_values.iter().zip(&y.method_values).all(|(p, q)| value_equals_with_arena(p, q, arena))
                && match (&x.data, &y.data) {
                    (Some(a), Some(b)) => value_equals_with_arena(a, b, arena),
                    (None, None) => true,
                    _ => false,
                }
        }
        (HeapObj::LazyVal(x), HeapObj::LazyVal(y)) => {
            // Forced lazy values compare their cached results; unforced ones compare by thunk closure
            let xf = x.forced.load(std::sync::atomic::Ordering::Relaxed);
            let yf = y.forced.load(std::sync::atomic::Ordering::Relaxed);
            if xf && yf {
                let xc = x.cached.lock().unwrap_or_else(|e| e.into_inner());
                let yc = y.cached.lock().unwrap_or_else(|e| e.into_inner());
                match (&*xc, &*yc) {
                    (Some(a), Some(b)) => value_equals_with_arena(a, b, arena),
                    (None, None) => true,
                    _ => false,
                }
            } else {
                false
            }
        }
        (HeapObj::AtomicVal(x), HeapObj::AtomicVal(y)) => {
            let xv = x.load();
            let yv = y.load();
            value_equals_with_arena(&xv, &yv, arena)
        }
        // AsyncVal: each AsyncHandle represents an independent async operation; two distinct instances are never equal
        // (equality of the same instance is guaranteed by the upper-layer Arc::ptr_eq)
        (HeapObj::AsyncVal(_), HeapObj::AsyncVal(_)) => false,
        // Arc-wrapped shared resources: compared by pointer identity (semantically correct—only the same channel is equal)
        (HeapObj::ChannelVal(x), HeapObj::ChannelVal(y)) => std::sync::Arc::ptr_eq(x, y),
        (HeapObj::SenderVal(x), HeapObj::SenderVal(y)) => std::sync::Arc::ptr_eq(&x.channel, &y.channel),
        (HeapObj::ReceiverVal(x), HeapObj::ReceiverVal(y)) => std::sync::Arc::ptr_eq(&x.channel, &y.channel),
        (HeapObj::CoroutineFrame, HeapObj::CoroutineFrame) => false,
        // FFI opaque pointers: equal iff raw pointer value matches
        (HeapObj::OpaquePtr(x), HeapObj::OpaquePtr(y)) => x.ptr == y.ptr,
        // Lib/ForeignFn: identity via the shared handle (Arc ptr_eq), like ChannelVal
        (HeapObj::LibVal(x), HeapObj::LibVal(y)) => std::sync::Arc::ptr_eq(&x.shared, &y.shared),
        (HeapObj::ForeignFnVal(x), HeapObj::ForeignFnVal(y)) => {
            std::sync::Arc::ptr_eq(&x.shared, &y.shared) && x.addr == y.addr
        }
        // Different HeapObj variants are never equal
        _ => false,
    }
}

/// Value semantic equality (used for HeapObj field comparison).
/// Scalars compare by tag + bit; Ref recurses through `heap_equals`; Null/Void compare by discriminant.
pub fn value_equals(a: &Value, b: &Value) -> bool {
    value_equals_with_arena(a, b, &ValueArena::default())
}

/// Value semantic equality (with `ValueArena`, used for `ValueHandle` comparison).
pub fn value_equals_with_arena(a: &Value, b: &Value, arena: &ValueArena) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) | (Value::Void, Value::Void) => true,
        (Value::Scalar(av, at), Value::Scalar(bv, bt)) => {
            if at != bt {
                // Cross-tag scalar pair: `i64? == 42` routes through this
                // generic eq (the nullable dispatch in the IR builder sends
                // all nullable ==/!= here) while the bare literal keeps i32.
                // The dedicated scalar comparison paths promote numerically;
                // this generic path must compare by numeric value too, not
                // silently return false.
                return scalar_cross_tag_eq(av, *at, bv, *bt);
            }
            // Compare the union field bit pattern by tag.
            // Note: when a match arm body starts with `unsafe {}`, Rust parses it as an "expression block"
            // and treats it as the entire arm body; a subsequent `==` would be parsed as the next arm's
            // pattern. The comparison expression must be wrapped in parentheses.
            match at {
                ValueTag::Bool => (unsafe { av.bool_val } == unsafe { bv.bool_val }),
                ValueTag::Char => (unsafe { av.char_val } == unsafe { bv.char_val }),
                ValueTag::I8 => (unsafe { av.i8_val } == unsafe { bv.i8_val }),
                ValueTag::I16 => (unsafe { av.i16_val } == unsafe { bv.i16_val }),
                ValueTag::I32 => (unsafe { av.i32_val } == unsafe { bv.i32_val }),
                ValueTag::I64 => (unsafe { av.i64_val } == unsafe { bv.i64_val }),
                ValueTag::I128 => (unsafe { av.i128_val } == unsafe { bv.i128_val }),
                ValueTag::U8 => (unsafe { av.u8_val } == unsafe { bv.u8_val }),
                ValueTag::U16 => (unsafe { av.u16_val } == unsafe { bv.u16_val }),
                ValueTag::U32 => (unsafe { av.u32_val } == unsafe { bv.u32_val }),
                ValueTag::U64 => (unsafe { av.u64_val } == unsafe { bv.u64_val }),
                ValueTag::U128 => (unsafe { av.u128_val } == unsafe { bv.u128_val }),
                ValueTag::Isize => (unsafe { av.isize_val } == unsafe { bv.isize_val }),
                ValueTag::Usize => (unsafe { av.usize_val } == unsafe { bv.usize_val }),
                ValueTag::F16 => (unsafe { av.f16_val } == unsafe { bv.f16_val }),
                ValueTag::F32 => unsafe { av.f32_val }.to_bits() == unsafe { bv.f32_val }.to_bits(),
                ValueTag::F64 => unsafe { av.f64_val }.to_bits() == unsafe { bv.f64_val }.to_bits(),
                ValueTag::F128 => (unsafe { av.f128_val } == unsafe { bv.f128_val }),
                _ => unreachable!("non-scalar tag in ScalarValue"),
            }
        }
        (Value::Ref(ax), Value::Ref(bx)) => heap_equals(ax.as_ref(), bx.as_ref(), arena),
        _ => false,
    }
}

/// Numeric equality for a scalar pair whose tags differ. Mirrors the promotion
/// the dedicated scalar comparison paths apply: integers compare exactly
/// (signed/unsigned mixed by magnitude), floats compare by value (f128
/// involved → compare as f128 to keep full precision).
fn scalar_cross_tag_eq(av: &crate::value::ScalarValue, at: ValueTag, bv: &crate::value::ScalarValue, bt: ValueTag) -> bool {
    use crate::value::ValueTag;
    let unsigned = |t: ValueTag| {
        matches!(
            t,
            ValueTag::U8 | ValueTag::U16 | ValueTag::U32 | ValueTag::U64 | ValueTag::U128 | ValueTag::Usize
        )
    };
    let a = Value::Scalar(av.clone(), at);
    let b = Value::Scalar(bv.clone(), bt);
    match (at.is_int(), bt.is_int()) {
        (true, true) => match (unsigned(at), unsigned(bt)) {
            (false, false) => a.as_int_i128() == b.as_int_i128(),
            (true, true) => a.as_u128() == b.as_u128(),
            (true, false) => b.as_int_i128() >= 0 && a.as_u128() == (b.as_int_i128() as u128),
            (false, true) => a.as_int_i128() >= 0 && b.as_u128() == (a.as_int_i128() as u128),
        },
        _ => {
            // At least one float (mixed int/float pairs are rejected by sema;
            // this is a defensive numeric comparison): compare via f128 when
            // either side is f128, else via f64.
            if at == ValueTag::F128 || bt == ValueTag::F128 {
                a.as_f128() == b.as_f128()
            } else {
                a.as_float_f64() == b.as_float_f64()
            }
        }
    }
}

/// Deep-clone cache: the Value path and the ValueHandle path each maintain a ptr→result cache,
/// preventing infinite recursion from reference cycles (e.g. Cell). The two paths' caches are
/// independent because HeapObj fields are partially migrated (some are Values, some remain ValueHandles).
struct DeepCloneCache {
    handle: FxHashMap<*const HeapObj, ValueHandle>,
    value: FxHashMap<*const HeapObj, Value>,
}

/// Value-path deep clone: scalars/nulls clone directly (cheap); Ref recursively clones the HeapObj.
fn deep_clone_value(v: &Value, arena: &mut ValueArena, cache: &mut DeepCloneCache) -> Value {
    match v {
        Value::Null | Value::Void | Value::Scalar(_, _) => v.clone(),
        Value::Ref(rc) => {
            let key = Arc::as_ptr(rc);
            if let Some(cached) = cache.value.get(&key) {
                return cached.clone();
            }
            let new_obj = deep_clone_heap(rc.as_ref(), arena, cache);
            let new_v = Value::Ref(Arc::new(new_obj));
            cache.value.insert(key, new_v.clone());
            new_v
        }
    }
}

fn deep_clone_handle(
    h: ValueHandle,
    arena: &mut ValueArena,
    cache: &mut DeepCloneCache,
) -> ValueHandle {
    match h.tag() {
        ValueTag::Null => ValueHandle::NULL,
        ValueTag::Void => ValueHandle::VOID,
        ValueTag::Bool => ValueArena::bool_val(arena.get_bool(h)),
        ValueTag::Char => arena.alloc_char(arena.get_char(h)),
        ValueTag::I8 => arena.alloc_i8(arena.get_i8(h)),
        ValueTag::I16 => arena.alloc_i16(arena.get_i16(h)),
        ValueTag::I32 => arena.alloc_i32(arena.get_i32(h)),
        ValueTag::I64 => arena.alloc_i64(arena.get_i64(h)),
        ValueTag::I128 => arena.alloc_i128(arena.get_i128(h)),
        ValueTag::U8 => arena.alloc_u8(arena.get_u8(h)),
        ValueTag::U16 => arena.alloc_u16(arena.get_u16(h)),
        ValueTag::U32 => arena.alloc_u32(arena.get_u32(h)),
        ValueTag::U64 => arena.alloc_u64(arena.get_u64(h)),
        ValueTag::U128 => arena.alloc_u128(arena.get_u128(h)),
        ValueTag::Isize => arena.alloc_isize(arena.get_isize(h)),
        ValueTag::Usize => arena.alloc_usize(arena.get_usize(h)),
        ValueTag::F16 => arena.alloc_f16(arena.get_f16(h)),
        ValueTag::F32 => arena.alloc_f32(arena.get_f32(h)),
        ValueTag::F64 => arena.alloc_f64(arena.get_f64(h)),
        ValueTag::F128 => arena.alloc_f128(arena.get_f128(h)),
        ValueTag::Ref => {
            let rc = arena.get_ref(h).clone();
            let key = Arc::as_ptr(&rc);
            if let Some(&cached) = cache.handle.get(&key) {
                return cached;
            }
            let new_obj = deep_clone_heap(&rc, arena, cache);
            let new_h = arena.alloc_ref_rc(Arc::new(new_obj));
            cache.handle.insert(key, new_h);
            new_h
        }
    }
}

fn deep_clone_heap(
    obj: &HeapObj,
    arena: &mut ValueArena,
    cache: &mut DeepCloneCache,
) -> HeapObj {
    match obj {
        HeapObj::Str(s) => HeapObj::Str(s.clone()),
        HeapObj::Array(a) => {
            // SoA fast path: scalars are Copy; clone SoA directly and rebuild elements via Value
            if let Some(soa) = &a.scalar_soa {
                let elems: Vec<Value> = simd_soa_deep_clone(soa);
                return HeapObj::Array(ArrayValue {
                    elements: elems,
                    fixed_size: a.fixed_size,
                    elem_is_ref: a.elem_is_ref,
                    scalar_soa: Some(soa.clone()),
                });
            }
            // Fall back: element-wise deep_clone (elements are Values)
            let elems: Vec<Value> = a
                .elements
                .iter()
                .map(|e| deep_clone_value(e, arena, cache))
                .collect();
            HeapObj::Array(ArrayValue {
                elements: elems,
                fixed_size: a.fixed_size,
                elem_is_ref: a.elem_is_ref,
                scalar_soa: a.scalar_soa.clone(),
            })
        }
        HeapObj::Record(r) => {
            // fields have been migrated to Value
            let fields: Vec<Value> = r
                .fields
                .iter()
                .map(|e| deep_clone_value(e, arena, cache))
                .collect();
            HeapObj::Record(RecordValue {
                type_name: r.type_name.clone(),
                fields,
                field_names: r.field_names.clone(),
                field_ref_bits: r.field_ref_bits,
            })
        }
        HeapObj::Adt(a) => {
            // AdtField.value has been migrated to Value
            let fields: Vec<AdtField> = a
                .fields
                .iter()
                .map(|f| AdtField {
                    name: f.name.clone(),
                    value: deep_clone_value(&f.value, arena, cache),
                })
                .collect();
            HeapObj::Adt(AdtValue {
                type_name: a.type_name.clone(),
                constructor: a.constructor.clone(),
                fields,
                field_ref_bits: a.field_ref_bits,
            })
        }
        HeapObj::Newtype(n) => HeapObj::Newtype(NewtypeValue {
            type_name: n.type_name.clone(),
            // inner is still a ValueHandle
            inner: deep_clone_handle(n.inner, arena, cache),
        }),
        HeapObj::Cell(c) => {
            let inner = c.inner.lock().clone();
            HeapObj::Cell(Cell::new(deep_clone_value(&inner, arena, cache)))
        }
        HeapObj::Range(r) => HeapObj::Range(r.clone()),
        HeapObj::Closure(c) => {
            // upvalues have been migrated to Value; bound_args are still ValueHandles
            let upvalues: Vec<Value> = c
                .upvalues
                .iter()
                .map(|e| deep_clone_value(e, arena, cache))
                .collect();
            let bound_args: Vec<ValueHandle> = c
                .bound_args
                .iter()
                .map(|e| deep_clone_handle(*e, arena, cache))
                .collect();
            HeapObj::Closure(Closure {
                func_id: c.func_id,
                arity: c.arity,
                upvalues,
                bound_args,
                self_upvalue_idx: c.self_upvalue_idx,
                upvalue_ref_bits: c.upvalue_ref_bits,
                cell_upvalues: c.cell_upvalues,
            })
        }
        HeapObj::Partial(p) => {
            let upvalues: Vec<Value> = p
                .upvalues
                .iter()
                .map(|v| deep_clone_value(v, arena, cache))
                .collect();
            let bound_args: Vec<Value> = p
                .bound_args
                .iter()
                .map(|v| deep_clone_value(v, arena, cache))
                .collect();
            HeapObj::Partial(PartialApplication {
                func_id: p.func_id,
                upvalues,
                bound_args,
                remaining_arity: p.remaining_arity,
                self_upvalue_idx: p.self_upvalue_idx,
            })
        }
        HeapObj::ThrowVal(t) => match &t.payload {
            // ThrowPayload::Ok/Err both hold Values; deep-copy recursively
            ThrowPayload::Ok(v) => HeapObj::ThrowVal(ThrowValue {
                payload: ThrowPayload::Ok(deep_clone_value(v, arena, cache)),
            }),
            ThrowPayload::Err(v) => HeapObj::ThrowVal(ThrowValue {
                payload: ThrowPayload::Err(deep_clone_value(v, arena, cache)),
            }),
        },
        HeapObj::Builtin(b) => HeapObj::Builtin(b.clone()),
        HeapObj::TraitVal(t) => HeapObj::TraitVal(t.clone()),
        HeapObj::LazyVal(l) => HeapObj::LazyVal(l.clone()),
        HeapObj::ErrorVal(e) => HeapObj::ErrorVal(e.clone()),
        // AtomicValue.data is a Value; deep-copy recursively
        HeapObj::AtomicVal(a) => HeapObj::AtomicVal(AtomicValue::new(deep_clone_value(&a.load(), arena, cache))),
        HeapObj::AsyncVal(a) => HeapObj::AsyncVal(a.clone()),
        HeapObj::ChannelVal(c) => HeapObj::ChannelVal(c.clone()),
        HeapObj::SenderVal(s) => HeapObj::SenderVal(s.clone()),
        HeapObj::ReceiverVal(r) => HeapObj::ReceiverVal(r.clone()),
        HeapObj::CoroutineFrame => HeapObj::CoroutineFrame,
        HeapObj::OpaquePtr(op) => HeapObj::OpaquePtr(op.clone()),
        HeapObj::LibVal(l) => HeapObj::LibVal(l.clone()),
        HeapObj::ForeignFnVal(f) => HeapObj::ForeignFnVal(f.clone()),
    }
}

// =========================================================================
// ValueArena convenience constructors (mirroring the legacy ValueHandle constructor API) + formatting/hash helpers
// =========================================================================

impl ValueArena {
    /// If the handle is a Ref, returns the corresponding heap object reference; otherwise None.
    #[inline]
    pub fn heap_obj_opt(&self, h: ValueHandle) -> Option<&HeapObj> {
        if h.tag() == ValueTag::Ref {
            Some(self.get_ref(h).as_ref())
        } else {
            None
        }
    }

    // ---- Singleton convenience constructors (no allocation) ----
    // null()/void() are provided by the existing impl ValueArena (already changed to &self).
    #[inline]
    pub fn bool(&self, v: bool) -> ValueHandle {
        Self::bool_val(v)
    }

    // ---- Scalar allocation convenience aliases ----
    #[inline]
    pub fn i8(&mut self, v: i8) -> ValueHandle {
        self.alloc_i8(v)
    }
    #[inline]
    pub fn i16(&mut self, v: i16) -> ValueHandle {
        self.alloc_i16(v)
    }
    #[inline]
    pub fn i32(&mut self, v: i32) -> ValueHandle {
        self.alloc_i32(v)
    }
    #[inline]
    pub fn i64(&mut self, v: i64) -> ValueHandle {
        self.alloc_i64(v)
    }
    #[inline]
    pub fn i128(&mut self, v: i128) -> ValueHandle {
        self.alloc_i128(v)
    }
    #[inline]
    pub fn u8(&mut self, v: u8) -> ValueHandle {
        self.alloc_u8(v)
    }
    #[inline]
    pub fn u16(&mut self, v: u16) -> ValueHandle {
        self.alloc_u16(v)
    }
    #[inline]
    pub fn u32(&mut self, v: u32) -> ValueHandle {
        self.alloc_u32(v)
    }
    #[inline]
    pub fn u64(&mut self, v: u64) -> ValueHandle {
        self.alloc_u64(v)
    }
    #[inline]
    pub fn u128(&mut self, v: u128) -> ValueHandle {
        self.alloc_u128(v)
    }
    #[inline]
    pub fn isize(&mut self, v: isize) -> ValueHandle {
        self.alloc_isize(v)
    }
    #[inline]
    pub fn usize(&mut self, v: usize) -> ValueHandle {
        self.alloc_usize(v)
    }
    #[inline]
    pub fn f16(&mut self, v: F16) -> ValueHandle {
        self.alloc_f16(v.0)
    }
    #[inline]
    pub fn f32(&mut self, v: f32) -> ValueHandle {
        self.alloc_f32(v)
    }
    #[inline]
    pub fn f64(&mut self, v: f64) -> ValueHandle {
        self.alloc_f64(v)
    }
    #[inline]
    pub fn f128(&mut self, v: F128) -> ValueHandle {
        self.alloc_f128(v)
    }
    #[inline]
    pub fn char(&mut self, c: Char) -> ValueHandle {
        self.alloc_char(c.codepoint)
    }
    #[inline]
    pub fn from_rust_char(&mut self, c: char) -> ValueHandle {
        self.alloc_char(c as u32)
    }

    // ---- Heap object convenience constructors ----
    pub fn str(&mut self, s: impl Into<String>) -> ValueHandle {
        self.alloc_ref(HeapObj::Str(Str::new(s)))
    }
    pub fn str_from(&mut self, s: &str) -> ValueHandle {
        self.alloc_ref(HeapObj::Str(Str::from_rust_str(s)))
    }
    pub fn from_str(&mut self, s: Str) -> ValueHandle {
        self.alloc_ref(HeapObj::Str(s))
    }
    pub fn heap(&mut self, obj: HeapObj) -> ValueHandle {
        self.alloc_ref(obj)
    }
    pub fn from_ref(&mut self, r: HeapRef) -> ValueHandle {
        self.alloc_ref_rc(r)
    }
    pub fn array(&mut self, elements: Vec<Value>) -> ValueHandle {
        self.alloc_ref(HeapObj::Array(ArrayValue::new(elements)))
    }
    pub fn array_fixed(&mut self, elements: Vec<Value>, size: u64) -> ValueHandle {
        self.alloc_ref(HeapObj::Array(ArrayValue::new_fixed(elements, size)))
    }
    pub fn record(
        &mut self,
        type_name: impl Into<String>,
        fields: Vec<Value>,
        field_names: Vec<Option<String>>,
    ) -> ValueHandle {
        self.alloc_ref(HeapObj::Record(RecordValue::new(
            type_name.into(),
            fields,
            field_names,
        )))
    }
    pub fn adt(
        &mut self,
        type_name: impl Into<String>,
        constructor: impl Into<String>,
        fields: Vec<AdtField>,
    ) -> ValueHandle {
        self.alloc_ref(HeapObj::Adt(AdtValue::new(
            type_name.into(),
            constructor.into(),
            fields,
        )))
    }
    pub fn newtype(&mut self, type_name: impl Into<String>, inner: ValueHandle) -> ValueHandle {
        self.alloc_ref(HeapObj::Newtype(NewtypeValue {
            type_name: type_name.into(),
            inner,
        }))
    }
    pub fn cell(&mut self, val: Value) -> ValueHandle {
        self.alloc_ref(HeapObj::Cell(Cell::new(val)))
    }
    pub fn range(&mut self, start: i64, end: i64, inclusive: bool) -> ValueHandle {
        self.alloc_ref(HeapObj::Range(Range::new(start, end, inclusive)))
    }
    pub fn closure(&mut self, c: Closure) -> ValueHandle {
        self.alloc_ref(HeapObj::Closure(c))
    }
    pub fn partial(&mut self, p: PartialApplication) -> ValueHandle {
        self.alloc_ref(HeapObj::Partial(p))
    }
    pub fn builtin(&mut self, fn_ptr: BuiltinFn, name: impl Into<String>) -> ValueHandle {
        self.alloc_ref(HeapObj::Builtin(Builtin {
            fn_ptr,
            name: name.into(),
        }))
    }
    pub fn trait_val(&mut self, t: TraitValue) -> ValueHandle {
        self.alloc_ref(HeapObj::TraitVal(t))
    }
    pub fn lazy(&mut self, l: LazyValue) -> ValueHandle {
        self.alloc_ref(HeapObj::LazyVal(l))
    }
    pub fn error_val(
        &mut self,
        type_name: impl Into<String>,
        message: impl Into<String>,
        is_error_subtype: bool,
    ) -> ValueHandle {
        self.alloc_ref(HeapObj::ErrorVal(ErrorValue {
            type_name: type_name.into(),
            message: message.into(),
            is_error_subtype,
        }))
    }
    pub fn throw_ok(&mut self, val: Value) -> ValueHandle {
        self.alloc_ref(HeapObj::ThrowVal(ThrowValue {
            payload: ThrowPayload::Ok(val),
        }))
    }
    pub fn throw_err(&mut self, err_val: Value) -> ValueHandle {
        self.alloc_ref(HeapObj::ThrowVal(ThrowValue {
            payload: ThrowPayload::Err(err_val),
        }))
    }
    pub fn atomic(&mut self, val: Value) -> ValueHandle {
        self.alloc_ref(HeapObj::AtomicVal(AtomicValue::new(val)))
    }
    pub fn async_handle(&mut self) -> ValueHandle {
        self.alloc_ref(HeapObj::AsyncVal(AsyncHandle::new()))
    }
    pub fn channel(&mut self, capacity: usize) -> ValueHandle {
        self.alloc_ref(HeapObj::ChannelVal(Arc::new(ChannelValue::new(capacity))))
    }
    pub fn sender(&mut self, channel: Arc<ChannelValue>) -> ValueHandle {
        self.alloc_ref(HeapObj::SenderVal(SenderValue { channel }))
    }
    pub fn receiver(&mut self, channel: Arc<ChannelValue>) -> ValueHandle {
        self.alloc_ref(HeapObj::ReceiverVal(ReceiverValue { channel }))
    }

    // ---- Formatting wrappers ----
    pub fn display(&self, h: ValueHandle) -> ValueDisplay<'_> {
        ValueDisplay { arena: self, handle: h }
    }
    pub fn debug(&self, h: ValueHandle) -> ValueDebug<'_> {
        ValueDebug { arena: self, handle: h }
    }

    // ---- Hash by value ----
    pub fn hash_value<H: Hasher>(&self, h: ValueHandle, state: &mut H) {
        match h.tag() {
            ValueTag::Null => 0u8.hash(state),
            ValueTag::Void => 1u8.hash(state),
            ValueTag::Bool => {
                2u8.hash(state);
                self.get_bool(h).hash(state)
            }
            ValueTag::Char => {
                3u8.hash(state);
                self.get_char(h).hash(state)
            }
            ValueTag::I8 => {
                4u8.hash(state);
                self.get_i8(h).hash(state)
            }
            ValueTag::I16 => {
                5u8.hash(state);
                self.get_i16(h).hash(state)
            }
            ValueTag::I32 => {
                6u8.hash(state);
                self.get_i32(h).hash(state)
            }
            ValueTag::I64 => {
                7u8.hash(state);
                self.get_i64(h).hash(state)
            }
            ValueTag::I128 => {
                8u8.hash(state);
                self.get_i128(h).hash(state)
            }
            ValueTag::U8 => {
                9u8.hash(state);
                self.get_u8(h).hash(state)
            }
            ValueTag::U16 => {
                10u8.hash(state);
                self.get_u16(h).hash(state)
            }
            ValueTag::U32 => {
                11u8.hash(state);
                self.get_u32(h).hash(state)
            }
            ValueTag::U64 => {
                12u8.hash(state);
                self.get_u64(h).hash(state)
            }
            ValueTag::U128 => {
                13u8.hash(state);
                self.get_u128(h).hash(state)
            }
            ValueTag::Isize => {
                14u8.hash(state);
                self.get_isize(h).hash(state)
            }
            ValueTag::Usize => {
                15u8.hash(state);
                self.get_usize(h).hash(state)
            }
            ValueTag::F16 => {
                16u8.hash(state);
                self.get_f16(h).hash(state)
            }
            ValueTag::F32 => {
                17u8.hash(state);
                self.get_f32(h).to_bits().hash(state)
            }
            ValueTag::F64 => {
                18u8.hash(state);
                self.get_f64(h).to_bits().hash(state)
            }
            ValueTag::F128 => {
                19u8.hash(state);
                self.get_f128(h).hash(state)
            }
            ValueTag::Ref => {
                20u8.hash(state);
                self.get_ref(h).hash(state);
            }
        }
    }
}


