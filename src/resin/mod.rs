//! resin — Kuzo 二进制可执行文件格式（.resin）
//!
//! 将 IR 阶段产出的 `DataFlowGraph` 持久化为跨平台二进制文件（`.resin`），
//! 支持 mmap zerocopy 加载，使 Kuzo 具备"源码编译 → 产物分发 → runtime 解释执行"的工作流。
//!
//! 模块组成：
//! - [`Spec`]: 格式规范层（常量、Header、Section、StringPool、CRC32、enum 映射）
//! - [`Format`]: 序列化/反序列化实现（serialize/load/inspect）
//! - [`Accessors`]: zerocopy 访问层（DataFlowGraph accessor 方法，mmap 切片读取）
//! - [`Migration`]: 跨版本迁移骨架（预留接口，当前不实现具体迁移）

#![allow(non_snake_case)]

pub mod Spec;
pub mod Format;
pub mod Accessors;
pub mod Migration;
