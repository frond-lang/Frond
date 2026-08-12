// Kuzo基础数据类型 ↔ C类型映射表(单一真相源)。
//
// 本文件是纯数据 + 查找函数,**不依赖 `crate::`**,因此可被两处复用:
//  - lib crate:`crate::types::Ctype`(通过 `pub mod Ctype`)
//  - build.rs:`include!` 内联(构建期文本提取路径)
//
// Gen.rs 的 `KUZO_TYPE_MAP` / `c_type_to_rust` 等FFI映射从本表派生,
// 避免Kuzo→C类型对应关系散落多处。

// ============ Kuzo类型 → C类型 ============

/// Kuzo基础数据类型 → C类型的映射表。
///
/// 涵盖:
///  - 标量整数:i8..i128, u8..u128, isize, usize
///  - 浮点数:f16, f32, f64, f128
///  - 布尔/字符:bool, char
///  - 空类型:void
///  - 字符串:str(C侧通过 data/len 传递,这里映射为 `const char*`)
///  - 字节数组:u8[](C侧通过 data/len 传递,这里映射为 `uint8_t*`)
///  - 原始指针:*u8, *i8, ..., *void
///
/// 注意:i128/u128/f128 在C返回值位置通过 out-param 模式传递(MSVC不支持__int128返回值),
/// 但在参数位置仍用 LoHi 拆分;此表给出的是"该类型在C中的对应类型名",
/// FFI层的具体传递策略由 Gen.rs 的 `CParamKind` 决定。
pub const KUZO_TO_C_TYPE: &[(&str, &str)] = &[
    // ── 标量整数 ──
    ("i8",    "int8_t"),
    ("i16",   "int16_t"),
    ("i32",   "int32_t"),
    ("i64",   "int64_t"),
    ("i128",  "__int128"),
    ("u8",    "uint8_t"),
    ("u16",   "uint16_t"),
    ("u32",   "uint32_t"),
    ("u64",   "uint64_t"),
    ("u128",  "unsigned __int128"),
    ("isize", "ssize_t"),
    ("usize", "size_t"),
    // ── 浮点数 ──
    ("f16",   "uint16_t"),    // IEEE 754 binary16 通过 uint16 传递
    ("f32",   "float"),
    ("f64",   "double"),
    ("f128",  "unsigned __int128"), // IEEE 754 binary128 位模式通过 __int128 传递
    // ── 布尔/字符/空 ──
    ("bool",  "int"),
    ("char",  "uint32_t"),
    ("void",  "void"),
    // ── 字符串/字节数组(实际通过 data/len 双参数传递,这里给出元素指针类型)──
    ("str",   "const char*"),
    ("u8[]",  "uint8_t*"),
    // ── 原始指针 ──
    ("*u8",   "uint8_t*"),
    ("*i8",   "int8_t*"),
    ("*u16",  "uint16_t*"),
    ("*i16",  "int16_t*"),
    ("*u32",  "uint32_t*"),
    ("*i32",  "int32_t*"),
    ("*u64",  "uint64_t*"),
    ("*i64",  "int64_t*"),
    ("*void", "void*"),
];

/// Kuzo类型名 → C类型名查找。
///
/// 返回 `None` 表示该类型没有直接的C对应(如用户自定义类型)。
#[inline]
pub fn kuzo_to_c_type(kuzo_name: &str) -> Option<&'static str> {
    KUZO_TO_C_TYPE
        .iter()
        .find(|(n, _)| *n == kuzo_name)
        .map(|(_, c)| *c)
}
