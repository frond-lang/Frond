//! Reflect.rs — 反射原语 extern "C" fn 实现
//!
//! 所有原语接收 ValueHandle (u32)，内部查全局 ValueArena 拿 HeapObj，
//! 直接 match 读取已携带的类型信息。无 type_table、无 type_id 注入、无查表。
//!
//! 职责边界：
//! - Value.rs：Value 系统自描述（RecordValue.type_name 等字段已自带类型信息）
//! - Reflect.rs：extern "C" 原语，match HeapObj 返回反射信息
//! - Raw.kz：@extern("C") 声明，C ABI 调用约定
//! - Reflect.kz：Reflect 内建类型 + wrapper 函数

use std::ffi::CString;

use super::arena::ValueArena;
use super::value::{F16, F128, HeapObj, RefKind, ValueTag, Value, ValueHandle};

// =========================================================================
// TypeKind 枚举（与 Kuzo 侧 kind 值一致，供用户判断类型分类）
// =========================================================================

pub const KIND_NULL: u8 = 0;
pub const KIND_VOID: u8 = 1;
pub const KIND_BOOL: u8 = 2;
pub const KIND_CHAR: u8 = 3;
pub const KIND_I8: u8 = 4;
pub const KIND_I16: u8 = 5;
pub const KIND_I32: u8 = 6;
pub const KIND_I64: u8 = 7;
pub const KIND_U8: u8 = 8;
pub const KIND_U16: u8 = 9;
pub const KIND_U32: u8 = 10;
pub const KIND_U64: u8 = 11;
pub const KIND_ISIZE: u8 = 12;
pub const KIND_USIZE: u8 = 13;
pub const KIND_I128: u8 = 14;
pub const KIND_U128: u8 = 15;
pub const KIND_F16: u8 = 16;
pub const KIND_F32: u8 = 17;
pub const KIND_F64: u8 = 18;
pub const KIND_F128: u8 = 19;
pub const KIND_STR: u8 = 20;
pub const KIND_REF: u8 = 21;
pub const KIND_RECORD: u8 = 22;
pub const KIND_ADT: u8 = 23;
pub const KIND_NEWTYPE: u8 = 24;
pub const KIND_ARRAY: u8 = 25;

/// ValueTag → TypeKind 映射（标量直接映射，堆对象走 ref_kind）
fn tag_to_kind(tag: ValueTag) -> u8 {
    match tag {
        ValueTag::Null => KIND_NULL,
        ValueTag::Void => KIND_VOID,
        ValueTag::Bool => KIND_BOOL,
        ValueTag::Char => KIND_CHAR,
        ValueTag::I8 => KIND_I8,
        ValueTag::I16 => KIND_I16,
        ValueTag::I32 => KIND_I32,
        ValueTag::I64 => KIND_I64,
        ValueTag::U8 => KIND_U8,
        ValueTag::U16 => KIND_U16,
        ValueTag::U32 => KIND_U32,
        ValueTag::U64 => KIND_U64,
        ValueTag::Isize => KIND_ISIZE,
        ValueTag::Usize => KIND_USIZE,
        ValueTag::I128 => KIND_I128,
        ValueTag::U128 => KIND_U128,
        ValueTag::F16 => KIND_F16,
        ValueTag::F32 => KIND_F32,
        ValueTag::F64 => KIND_F64,
        ValueTag::F128 => KIND_F128,
        ValueTag::Ref => KIND_REF,
    }
}

/// ref_kind → TypeKind
fn ref_kind_to_kind(rk: RefKind) -> u8 {
    match rk {
        RefKind::Str => KIND_STR,
        RefKind::Array => KIND_ARRAY,
        RefKind::Record => KIND_RECORD,
        RefKind::Adt => KIND_ADT,
        RefKind::Newtype => KIND_NEWTYPE,
        _ => KIND_REF,
    }
}

// =========================================================================
// str 返回辅助：thread_local 缓冲避免悬垂指针
//
// [R-1 契约] write_str_out / write_slice_out 写入 *out_data 的指针仅保证有效
// 到「同一线程下一次调用任意 reflect 原语之前」。C 侧必须立即消费（memcpy），
// 不得跨 reflect 调用持有。原因：缓冲为 thread_local 单槽，下次调用即替换。
// =========================================================================

thread_local! {
    static NAME_BUF: std::cell::RefCell<CString> = std::cell::RefCell::new(CString::new("").unwrap());
}

fn write_str_out(s: &str, out_data: *mut *const u8, out_len: *mut usize) {
    NAME_BUF.with(|buf| {
        let mut b = buf.borrow_mut();
        // [R-4] 含 NUL 的字符串视为非法（类型名不应含 NUL），显式标记而非静默空串
        *b = CString::new(s).unwrap_or_else(|_| CString::new("<invalid-name>").unwrap());
        unsafe {
            *out_data = b.as_ptr() as *const u8;
            *out_len = b.to_bytes().len();
        }
    });
}

/// 写入静态/借用字节切片指针到 out 参数（不经过 CString，零拷贝）。
/// 调用方须保证 `data` 在指针被消费前存活（静态切片或 thread_local 缓冲）。
fn write_slice_out(data: &[u8], out_data: *mut *const u8, out_len: *mut usize) {
    unsafe {
        *out_data = data.as_ptr();
        *out_len = data.len();
    }
}

thread_local! {
    static FORMAT_BUF: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

// =========================================================================
// 反射原语 — 全部 #[no_mangle] extern "C" fn，接收 u32 (ValueHandle raw)
// =========================================================================

/// 返回值的 TypeKind（标量直接映射，堆对象查 arena 取 ref_kind）
#[no_mangle]
pub extern "C" fn __reflect_kind(handle: u32) -> u8 {
    let h = ValueHandle::from_raw(handle);
    let tag = h.tag();
    if tag != ValueTag::Ref {
        return tag_to_kind(tag);
    }
    if let Some(obj) = ValueArena::get_global_obj(h) {
        ref_kind_to_kind(obj.ref_kind())
    } else {
        KIND_NULL
    }
}

/// 返回类型名（标量返回静态字符串，堆对象查 arena 读 type_name 字段）
///
/// 标量分支派生自 `Type::BUILTIN_TABLE`（单一真相源）：通过 `builtin_info_by_tag`
/// 查表获取 `&'static str` 类型名，消除原 21 个硬编码 `b"..."` 分支。
/// 不变量：外层 `tag != ValueTag::Ref` 保证 tag 必在 BUILTIN_TABLE 中（20 个非 Ref
/// tag 全部登记），`.expect` 为不变量违反时的 fail-fast，非回退。
#[no_mangle]
pub extern "C" fn __reflect_type_name(handle: u32, out_data: *mut *const u8, out_len: *mut usize) {
    let h = ValueHandle::from_raw(handle);
    let tag = h.tag();
    if tag != ValueTag::Ref {
        let info = crate::types::builtin_info_by_tag(tag)
            .expect("non-Ref ValueTag must be in BUILTIN_TABLE");
        // info.name 是 &'static str，指针 'static 有效，无悬垂风险
        write_slice_out(info.name.as_bytes(), out_data, out_len);
        return;
    }
    // 堆对象：查 arena 读用户类型名
    let name = if let Some(obj) = ValueArena::get_global_obj(h) {
        match &*obj {
            HeapObj::Record(r) => r.type_name.clone(),
            HeapObj::Adt(a) => a.type_name.clone(),
            HeapObj::Newtype(n) => n.type_name.clone(),
            HeapObj::Str(_) => "str".to_string(),
            HeapObj::Array(_) => "array".to_string(),
            _ => obj.ref_kind().as_str().to_string(),
        }
    } else {
        "<unknown>".to_string()
    };
    write_str_out(&name, out_data, out_len);
}

/// 返回值的字节大小（标量委托 `ValueTag::byte_width`，堆对象按 ref_kind 估算）
#[no_mangle]
pub extern "C" fn __reflect_size(handle: u32) -> u8 {
    let h = ValueHandle::from_raw(handle);
    let tag = h.tag();
    if tag != ValueTag::Ref {
        // 标量/Null/Void 统一委托 byte_width（与 Value.rs 单点同步）
        return tag.byte_width() as u8;
    }
    // 堆对象：str/array 估算为 16（data+len），其余无固定尺寸
    if let Some(obj) = ValueArena::get_global_obj(h) {
        match obj.ref_kind() {
            RefKind::Str => 16,
            RefKind::Array => 16,
            RefKind::Record | RefKind::Adt | RefKind::Newtype => 0,
            _ => 0,
        }
    } else { 0 }
}

/// 返回字段数（Record/Adt/Newtype/Array 的字段/元素数）
#[no_mangle]
pub extern "C" fn __reflect_field_count(handle: u32) -> u16 {
    let h = ValueHandle::from_raw(handle);
    if h.tag() != ValueTag::Ref {
        return 0;
    }
    if let Some(obj) = ValueArena::get_global_obj(h) {
        match &*obj {
            // [R-2] clamp 到 u16::MAX 而非 as 截断回绕，避免 >65535 字段静默变成错误小值
            HeapObj::Record(r) => r.fields.len().min(u16::MAX as usize) as u16,
            HeapObj::Adt(a) => a.fields.len().min(u16::MAX as usize) as u16,
            HeapObj::Newtype(_) => 1,
            HeapObj::Array(a) => a.elements.len().min(u16::MAX as usize) as u16,
            _ => 0,
        }
    } else {
        0
    }
}

/// 返回字段名（Record/Adt 的字段名，数组/元组返回空字符串）
#[no_mangle]
pub extern "C" fn __reflect_field_name(handle: u32, index: u16, out_data: *mut *const u8, out_len: *mut usize) {
    let h = ValueHandle::from_raw(handle);
    if h.tag() != ValueTag::Ref {
        write_str_out("", out_data, out_len);
        return;
    }
    let name = if let Some(obj) = ValueArena::get_global_obj(h) {
        match &*obj {
            HeapObj::Record(r) => r.field_names
                .get(index as usize)
                .and_then(|n| n.as_ref())
                .cloned()
                .unwrap_or_default(),
            HeapObj::Adt(a) => a.fields
                .get(index as usize)
                .and_then(|f| f.name.as_ref())
                .cloned()
                .unwrap_or_default(),
            _ => String::new(),
        }
    } else {
        String::new()
    };
    write_str_out(&name, out_data, out_len);
}

/// 返回字段值（子 ValueHandle，用于递归反射）。
/// HeapObj 字段已迁移为 Value，需通过 alloc_value 转回 ValueHandle 供 FFI 返回。
#[no_mangle]
pub extern "C" fn __reflect_field_value(handle: u32, index: u16) -> u32 {
    let h = ValueHandle::from_raw(handle);
    if h.tag() != ValueTag::Ref {
        return ValueHandle::NULL.to_raw();
    }
    if let Some(obj) = ValueArena::get_global_obj(h) {
        match &*obj {
            // Record/Adt/Array 字段为 Value：alloc_value 转回 ValueHandle
            HeapObj::Record(r) => r.fields.get(index as usize)
                .map(|f| ValueArena::with_global_mut(|a| a.alloc_value(f)).to_raw())
                .unwrap_or(ValueHandle::NULL.to_raw()),
            HeapObj::Adt(a) => a.fields.get(index as usize)
                .map(|f| ValueArena::with_global_mut(|a| a.alloc_value(&f.value)).to_raw())
                .unwrap_or(ValueHandle::NULL.to_raw()),
            // Newtype.inner 仍为 ValueHandle，直接返回
            HeapObj::Newtype(n) => if index == 0 { n.inner.to_raw() } else { ValueHandle::NULL.to_raw() },
            HeapObj::Array(a) => a.elements.get(index as usize)
                .map(|f| ValueArena::with_global_mut(|arena| arena.alloc_value(f)).to_raw())
                .unwrap_or(ValueHandle::NULL.to_raw()),
            _ => ValueHandle::NULL.to_raw(),
        }
    } else {
        ValueHandle::NULL.to_raw()
    }
}

/// 返回数组长度
#[no_mangle]
pub extern "C" fn __reflect_array_len(handle: u32) -> usize {
    let h = ValueHandle::from_raw(handle);
    if h.tag() != ValueTag::Ref {
        return 0;
    }
    if let Some(obj) = ValueArena::get_global_obj(h) {
        if let HeapObj::Array(a) = &*obj {
            return a.elements.len();
        }
    }
    0
}

/// 返回 ADT 构造器名
#[no_mangle]
pub extern "C" fn __reflect_adt_constructor(handle: u32, out_data: *mut *const u8, out_len: *mut usize) {
    let h = ValueHandle::from_raw(handle);
    let ctor = if h.tag() == ValueTag::Ref {
        if let Some(obj) = ValueArena::get_global_obj(h) {
            if let HeapObj::Adt(a) = &*obj {
                a.constructor.clone()
            } else {
                String::new()
            }
        } else { String::new() }
    } else { String::new() };
    write_str_out(&ctor, out_data, out_len);
}

/// 标量转字符串（按 ValueTag 分派格式化）
#[no_mangle]
pub extern "C" fn __reflect_scalar_to_str(handle: u32, out_data: *mut *const u8, out_len: *mut usize) {
    let h = ValueHandle::from_raw(handle);
    let tag = h.tag();
    FORMAT_BUF.with(|buf| {
        let mut s = buf.borrow_mut();
        s.clear();
        if tag == ValueTag::Ref {
            if let Some(obj) = ValueArena::get_global_obj(h) {
                match &*obj {
                    HeapObj::Str(kuzo_str) => s.push_str(kuzo_str.bytes()),
                    _ => s.push_str("<non-scalar>"),
                }
            } else {
                s.push_str("<unknown>");
            }
        } else {
            ValueArena::with_global(|a| {
                // [V-2] FFI 边界防御：脏 handle 的 index 可能越界，校验后再取值
                if !a.is_valid_handle_inner(h) {
                    s.push_str("<invalid>");
                    return;
                }
                match tag {
                    ValueTag::Null => s.push_str("null"),
                    ValueTag::Void => s.push_str("void"),
                    ValueTag::Bool => s.push_str(if h == ValueHandle::TRUE { "true" } else { "false" }),
                    ValueTag::Char => {
                        let c = a.get_char(h);
                        // 码点 → Unicode 标量值 → 字符（覆盖所有合法码点，包括非 ASCII）
                        if let Some(ch) = char::from_u32(c) {
                            s.push(ch);
                        } else {
                            s.push_str(&format!("U+{:04X}", c));
                        }
                    }
                    ValueTag::I8 => s.push_str(&a.get_i8(h).to_string()),
                    ValueTag::I16 => s.push_str(&a.get_i16(h).to_string()),
                    ValueTag::I32 => s.push_str(&a.get_i32(h).to_string()),
                    ValueTag::I64 => s.push_str(&a.get_i64(h).to_string()),
                    ValueTag::U8 => s.push_str(&a.get_u8(h).to_string()),
                    ValueTag::U16 => s.push_str(&a.get_u16(h).to_string()),
                    ValueTag::U32 => s.push_str(&a.get_u32(h).to_string()),
                    ValueTag::U64 => s.push_str(&a.get_u64(h).to_string()),
                    ValueTag::Isize => s.push_str(&a.get_isize(h).to_string()),
                    ValueTag::Usize => s.push_str(&a.get_usize(h).to_string()),
                    ValueTag::F32 => s.push_str(&a.get_f32(h).to_string()),
                    ValueTag::F64 => s.push_str(&a.get_f64(h).to_string()),
                    ValueTag::I128 => s.push_str(&a.get_i128(h).to_string()),
                    ValueTag::U128 => s.push_str(&a.get_u128(h).to_string()),
                    ValueTag::F16 => s.push_str(&F16(a.get_f16(h)).to_f32().to_string()),
                    ValueTag::F128 => s.push_str(&a.get_f128(h).to_f64().to_string()),
                    _ => s.push_str("<scalar>"),
                }
            });
        }
        // s 借用 FORMAT_BUF，指针有效至下次 reflect 调用（见 [R-1 契约]）
        write_slice_out(s.as_bytes(), out_data, out_len);
    });
}

/// 顶层格式化入口：递归 match HeapObj 生成字符串
#[no_mangle]
pub extern "C" fn __reflect_format(handle: u32, out_data: *mut *const u8, out_len: *mut usize) {
    // 入口 handle → Value，后续递归全部走 Value 路径
    let h = ValueHandle::from_raw(handle);
    let v = ValueArena::with_global(|arena| arena.get_value(h));
    let result = format_value(&v, 0);
    write_str_out(&result, out_data, out_len);
}

/// 递归格式化 Value 为 String（内部函数，非 extern "C"）。
/// [R-3] depth 限制递归深度，防止环引用或极深嵌套导致栈溢出。
const FORMAT_MAX_DEPTH: u32 = 64;
pub fn format_value(v: &Value, depth: u32) -> String {
    // 深度超限：截断为省略号，避免栈溢出（环/极深嵌套防御）
    if depth > FORMAT_MAX_DEPTH {
        return "...".to_string();
    }
    match v {
        Value::Null => "null".to_string(),
        Value::Void => "void".to_string(),
        Value::Scalar(sv, tag) => {
            // 标量格式化：直接从 ScalarValue 读取，不经 ValueArena
            unsafe {
                match tag {
                    ValueTag::Bool => (if sv.bool_val { "true" } else { "false" }).to_string(),
                    ValueTag::Char => {
                        let c = sv.char_val;
                        // 码点 → Unicode 标量值 → 字符（覆盖所有合法码点，包括非 ASCII）
                        char::from_u32(c).map(|ch| ch.to_string()).unwrap_or_else(|| format!("U+{:04X}", c))
                    }
                    ValueTag::I8 => sv.i8_val.to_string(),
                    ValueTag::I16 => sv.i16_val.to_string(),
                    ValueTag::I32 => sv.i32_val.to_string(),
                    ValueTag::I64 => sv.i64_val.to_string(),
                    ValueTag::U8 => sv.u8_val.to_string(),
                    ValueTag::U16 => sv.u16_val.to_string(),
                    ValueTag::U32 => sv.u32_val.to_string(),
                    ValueTag::U64 => sv.u64_val.to_string(),
                    ValueTag::Isize => sv.isize_val.to_string(),
                    ValueTag::Usize => sv.usize_val.to_string(),
                    ValueTag::I128 => i128::from_ne_bytes(std::mem::transmute(sv.i128_val)).to_string(),
                    ValueTag::U128 => u128::from_ne_bytes(std::mem::transmute(sv.u128_val)).to_string(),
                    ValueTag::F16 => format!("{:?}", F16(sv.f16_val)),
                    ValueTag::F32 => sv.f32_val.to_string(),
                    ValueTag::F64 => sv.f64_val.to_string(),
                    ValueTag::F128 => format!("{:?}", F128(std::mem::transmute(sv.f128_val))),
                    _ => unreachable!("non-scalar tag in ScalarValue"),
                }
            }
        }
        Value::Ref(r) => {
            // 堆对象：match HeapObj
            match &**r {
                HeapObj::Record(rec) => {
                    let mut out = format!("{}(", rec.type_name);
                    for (i, f) in rec.fields.iter().enumerate() {
                        if i > 0 { out.push_str(", "); }
                        if let Some(name) = rec.field_names.get(i).and_then(|n| n.as_ref()) {
                            out.push_str(name);
                            out.push_str(": ");
                        }
                        out.push_str(&format_value(f, depth + 1));
                    }
                    out.push(')');
                    out
                }
                HeapObj::Adt(a) => {
                    if a.fields.is_empty() {
                        a.constructor.clone()
                    } else {
                        let mut out = format!("{}(", a.constructor);
                        for (i, f) in a.fields.iter().enumerate() {
                            if i > 0 { out.push_str(", "); }
                            if let Some(name) = &f.name {
                                out.push_str(name);
                                out.push_str(": ");
                            }
                            out.push_str(&format_value(&f.value, depth + 1));
                        }
                        out.push(')');
                        out
                    }
                }
                HeapObj::Newtype(n) => {
                    // Newtype.inner 仍为 ValueHandle：转 Value 后递归
                    let inner_val = ValueArena::with_global(|arena| arena.get_value(n.inner));
                    format!("{}({})", n.type_name, format_value(&inner_val, depth + 1))
                }
                HeapObj::Array(a) => {
                    let mut out = String::from("[");
                    for (i, e) in a.elements.iter().enumerate() {
                        if i > 0 { out.push_str(", "); }
                        out.push_str(&format_value(e, depth + 1));
                    }
                    out.push(']');
                    out
                }
                HeapObj::Str(kuzo_str) => kuzo_str.bytes().to_string(),
                HeapObj::LazyVal(lazy) => {
                    // 已 forced 的 LazyValue：格式化缓存值
                    // 未 forced 的 LazyValue：由 Engine 的 force_lazy_value_sync 预先处理，
                    // 此处仅处理嵌套结构中残留的未 forced LazyValue（防御性兜底）
                    if lazy.forced.load(std::sync::atomic::Ordering::Relaxed) {
                        match &*lazy.cached.lock().unwrap() {
                            Some(v) => format_value(v, depth + 1),
                            None => "<lazy:empty>".to_string(),
                        }
                    } else {
                        "<lazy>".to_string()
                    }
                }
                _ => {
                    // 其他堆对象：用 ref_kind 名兜底
                    "<non-scalar>".to_string()
                }
            }
        }
    }
}

/// 返回类型种类字符串（"Primitive"/"Record"/"Adt"/"Newtype"/"Str"/"Array"/"Nullable"/"Ref"）
#[no_mangle]
pub extern "C" fn __reflect_kind_str(handle: u32, out_data: *mut *const u8, out_len: *mut usize) {
    let h = ValueHandle::from_raw(handle);
    let tag = h.tag();
    let kind: &[u8] = if tag != ValueTag::Ref {
        match tag {
            ValueTag::Null => b"Null",
            ValueTag::Void => b"Void",
            _ => b"Primitive",
        }
    } else if let Some(obj) = ValueArena::get_global_obj(h) {
        match obj.ref_kind() {
            RefKind::Str => b"Str",
            RefKind::Array => b"Array",
            RefKind::Record => b"Record",
            RefKind::Adt => b"Adt",
            RefKind::Newtype => b"Newtype",
            RefKind::Closure => b"Closure",
            RefKind::TraitVal => b"Trait",
            _ => b"Ref",
        }
    } else {
        b"Null"
    };
    write_slice_out(kind, out_data, out_len);
}

/// 返回值的布局大小（标量按 tag，堆对象按 ref_kind 估算字段总大小）
#[no_mangle]
pub extern "C" fn __reflect_layout_size(handle: u32) -> u32 {
    let h = ValueHandle::from_raw(handle);
    let tag = h.tag();
    match tag {
        ValueTag::Null | ValueTag::Void => 0,
        ValueTag::Bool => 1,
        ValueTag::Char => 4,
        ValueTag::I8 | ValueTag::U8 => 1,
        ValueTag::I16 | ValueTag::U16 | ValueTag::F16 => 2,
        ValueTag::I32 | ValueTag::U32 | ValueTag::F32 => 4,
        ValueTag::I64 | ValueTag::U64 | ValueTag::F64 | ValueTag::Isize | ValueTag::Usize => 8,
        ValueTag::I128 | ValueTag::U128 | ValueTag::F128 => 16,
        ValueTag::Ref => {
            if let Some(obj) = ValueArena::get_global_obj(h) {
                match &*obj {
                    HeapObj::Str(_) => 16,
                    HeapObj::Array(_) => {
                        // 16B fat pointer + 元素大小 * len（估算）
                        16
                    }
                    HeapObj::Record(r) => {
                        // 字段大小总和（粗略估算，不含 padding）
                        r.fields.iter().map(value_size).sum::<u32>()
                    }
                    HeapObj::Adt(a) => {
                        // 字段大小总和（当前构造器的字段，不含 tag）
                        a.fields.iter().map(|f| value_size(&f.value)).sum::<u32>()
                    }
                    HeapObj::Newtype(n) => {
                        // inner 值大小
                        ValueArena::with_global(|arena| value_size(&arena.get_value(n.inner)))
                    }
                    _ => 0,
                }
            } else { 0 }
        }
    }
}

/// 返回值的对齐（标量按大小，堆对象按最大字段对齐）
#[no_mangle]
pub extern "C" fn __reflect_layout_alignment(handle: u32) -> u32 {
    let h = ValueHandle::from_raw(handle);
    let tag = h.tag();
    match tag {
        ValueTag::Null | ValueTag::Void | ValueTag::Bool => 1,
        ValueTag::Char => 4,
        ValueTag::I8 | ValueTag::U8 => 1,
        ValueTag::I16 | ValueTag::U16 | ValueTag::F16 => 2,
        ValueTag::I32 | ValueTag::U32 | ValueTag::F32 => 4,
        ValueTag::I64 | ValueTag::U64 | ValueTag::F64 | ValueTag::Isize | ValueTag::Usize => 8,
        ValueTag::I128 | ValueTag::U128 | ValueTag::F128 => 16,
        ValueTag::Ref => {
            if let Some(obj) = ValueArena::get_global_obj(h) {
                match &*obj {
                    HeapObj::Str(_) => 8,
                    HeapObj::Array(_) => 8,
                    HeapObj::Record(r) => {
                        // 最大字段对齐
                        r.fields.iter().map(value_alignment).max().unwrap_or(1)
                    }
                    HeapObj::Adt(a) => {
                        // tag(1) 和最大字段对齐取最大
                        a.fields.iter().map(|f| value_alignment(&f.value)).max().unwrap_or(1).max(1)
                    }
                    HeapObj::Newtype(n) => {
                        ValueArena::with_global(|arena| value_alignment(&arena.get_value(n.inner)))
                    }
                    _ => 8,
                }
            } else { 8 }
        }
    }
}

/// 估算 Value 的字节大小（用于 Record/Adt/Newtype layout 估算）
fn value_size(v: &Value) -> u32 {
    match v {
        Value::Null | Value::Void => 0,
        Value::Scalar(_, tag) => {
            match tag {
                ValueTag::Bool => 1,
                ValueTag::Char => 4,
                ValueTag::I8 | ValueTag::U8 => 1,
                ValueTag::I16 | ValueTag::U16 | ValueTag::F16 => 2,
                ValueTag::I32 | ValueTag::U32 | ValueTag::F32 => 4,
                ValueTag::I64 | ValueTag::U64 | ValueTag::F64 | ValueTag::Isize | ValueTag::Usize => 8,
                ValueTag::I128 | ValueTag::U128 | ValueTag::F128 => 16,
                _ => unreachable!("non-scalar tag in ScalarValue"),
            }
        }
        Value::Ref(r) => {
            match &**r {
                HeapObj::Str(_) => 16,
                HeapObj::Array(_) => 16,
                HeapObj::Record(rec) => rec.fields.iter().map(value_size).sum(),
                // ADT：字段大小总和（当前构造器的字段，不含 tag）
                HeapObj::Adt(a) => a.fields.iter().map(|f| value_size(&f.value)).sum(),
                // Newtype：从全局 arena 查找 inner 值的大小
                HeapObj::Newtype(n) => {
                    ValueArena::with_global(|arena| value_size(&arena.get_value(n.inner)))
                }
                _ => 8,
            }
        }
    }
}

/// 估算 Value 的对齐
fn value_alignment(v: &Value) -> u32 {
    match v {
        Value::Null | Value::Void => 1,
        Value::Scalar(_, tag) => {
            match tag {
                ValueTag::Bool => 1,
                ValueTag::Char => 4,
                ValueTag::I8 | ValueTag::U8 => 1,
                ValueTag::I16 | ValueTag::U16 | ValueTag::F16 => 2,
                ValueTag::I32 | ValueTag::U32 | ValueTag::F32 => 4,
                ValueTag::I64 | ValueTag::U64 | ValueTag::F64 | ValueTag::Isize | ValueTag::Usize => 8,
                ValueTag::I128 | ValueTag::U128 | ValueTag::F128 => 16,
                _ => unreachable!("non-scalar tag in ScalarValue"),
            }
        }
        Value::Ref(r) => {
            match &**r {
                HeapObj::Str(_) | HeapObj::Array(_) => 8,
                HeapObj::Record(rec) => rec.fields.iter().map(value_alignment).max().unwrap_or(1),
                HeapObj::Adt(a) => a.fields.iter().map(|f| value_alignment(&f.value)).max().unwrap_or(1).max(1),
                // Newtype：从全局 arena 查找 inner 值的对齐
                HeapObj::Newtype(n) => {
                    ValueArena::with_global(|arena| value_alignment(&arena.get_value(n.inner)))
                }
                _ => 8,
            }
        }
    }
}

/// 公共 API：从 &Value 估算布局大小（供 Engine FFI 调用）
pub fn reflect_layout_size(v: &Value) -> u32 {
    value_size(v)
}

/// 公共 API：从 &Value 估算对齐（供 Engine FFI 调用）
pub fn reflect_layout_alignment(v: &Value) -> u32 {
    value_alignment(v)
}

// =========================================================================
// RefKind::as_str 辅助（用于 type_name 兜底）
// =========================================================================

impl RefKind {
    fn as_str(&self) -> &'static str {
        match self {
            RefKind::Str => "str",
            RefKind::Array => "array",
            RefKind::Record => "record",
            RefKind::Adt => "adt",
            RefKind::Newtype => "newtype",
            RefKind::Cell => "cell",
            RefKind::Range => "range",
            RefKind::Closure => "closure",
            RefKind::Partial => "partial",
            RefKind::Builtin => "builtin",
            RefKind::TraitVal => "trait",
            RefKind::LazyVal => "lazy",
            RefKind::ErrorVal => "error",
            RefKind::ThrowVal => "throw",
            RefKind::AtomicVal => "atomic",
            RefKind::AsyncVal => "async",
            RefKind::ChannelVal => "channel",
            RefKind::SenderVal => "sender",
            RefKind::ReceiverVal => "receiver",
            RefKind::CoroutineFrame => "coroutine",
        }
    }
}
