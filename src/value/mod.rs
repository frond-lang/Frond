#![allow(non_snake_case)]
//! value — Kuzo 统一值系统
//!
//! 拆分为四个子模块：
//! - `Value`：值表示层（标量基础类型 + 堆对象类型）
//! - `Arena`：存储与查询层（Bucket + ValueArena + ValueTrait + 相等性）
//! - `Ops`：操作层（Num/BitOps + cast + batch/SIMD + allocator + 纯算术核心）
//! - `Reflect`：反射层（extern "C" 原语 + format_value + layout 查询）

#[path = "Value.rs"]
mod value;

// #[macro_use] 使 Arena.rs 中定义的 read_int_as! / write_int_bytes! 宏
// 在后续的 Ops.rs 模块中可见。
#[macro_use]
#[path = "Arena.rs"]
mod arena;

#[path = "Ops.rs"]
mod ops;

#[path = "Reflect.rs"]
mod reflect;

pub use value::*;
pub use arena::*;
pub use ops::*;
pub use reflect::*;
