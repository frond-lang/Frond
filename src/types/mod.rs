#![allow(non_snake_case)]
//! Type — Kuzo 类型系统模块
//!
//! 承载所有内置类型的静态属性、统一语义层枚举 `Ty`（唯一类型来源，Copy）、
//! `TypeArena`（类型分配器 + unify/occurs/resolve）、`TypeOps` trait + ops 查找表、
//! `DynamicOpsRegistry`（替代 TypeDescriptorPool）、以及类型家族分类 `TypeFamily`。
//!
//! sema 和 IR 层都使用 `Ty`，不再有 `ConcreteType` / `TypeDescriptor`。
//!
//! 子模块组织：
//! - `Tag`：类型判别标签（ValueTag）+ 基础结构（TypeHandle / TypeFamily / FieldType / TraitMethodSig / EnvId）
//! - `ty`（文件 `Ty.rs`）：统一类型枚举 `Ty` + 类型变量 + BuiltinInfo/BUILTIN_TABLE + TypeDetail + SemKind + UnifyError
//! - `Arena`：类型分配器 `TypeArena` + unify/occurs/resolve + 快照
//! - `Display`：类型显示格式化（TypeDisplay）
//! - `Ops`：TypeOps trait + 标量/引用 ops 实现 + ops 查找表 + DynamicOpsRegistry

pub mod Tag;

// 注意：Ty.rs 中的核心类型 `Ty` 枚举与模块同名。
// 若用 `pub mod Ty;`，模块名会遮蔽 glob re-export 的 `Ty` 枚举，
// 导致 `crate::types::Ty` 指向模块而非枚举，破坏外部引用。
// 因此用 `#[path]` 将模块命名为 `ty`（私有），仅通过 glob re-export 暴露其 pub 项。
#[path = "Ty.rs"]
mod ty;

pub mod Arena;
pub mod Display;
pub mod Ops;

pub use Tag::*;
pub use ty::*;
pub use Arena::*;
pub use Display::*;
pub use Ops::*;
