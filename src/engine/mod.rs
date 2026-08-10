#![allow(non_snake_case)]
//! Engine 模块 — 数据流就绪调度执行引擎（调度器）
//!
//! 基于 [`crate::ir::Ir::DataFlowGraph`]，实现：
//! - Frame 管理（HashMap + LockStrategy）
//! - 就绪调度（核心循环）
//! - 子图启动（start/complete）
//! - TimerRuntime / AsyncJoinRuntime 图外运行时
//! - 单线程（Single）/ 多线程（Multi）锁策略
//!
//! compute_fn 计算函数表已拆分至 Compute.rs，调度器通过 graph.compute_fns[idx]
//! 间接调用，不直接引用具体 compute_fn。
//!
//! 子模块：
//! - [`EngineCore`]: 核心类型定义 — `Engine<S>` 结构体 + `EngineRef` 工厂 + Send/Sync + 调度器常量 + env_flag
//! - [`AsyncRt`]: 异步运行时（TimerRuntime / AsyncJoinRuntime）+ 事件处理
//! - [`Schedule`]: 数据流调度核心（就绪调度、批量化、run_frame_nodes、process_frame）
//! - [`Frame`]: Frame 生命周期管理（分配、初始化、重置、帧链）
//! - [`Subgraph`]: 子图调用与返回（switch_subgraph、start_subgraph、complete_and_wake_caller）
//! - [`Strategy`]: 并发策略（LockStrategy / Single / Multi / QueueHandle + worker）
//!
//! 设计原则（见 docs/superpowers/specs/2026-07-31-dataflow-engine-design.md）：
//! - 无 dispatch：调度器只认"输入就绪"，节点自带 compute_fn
//! - sync/async 统一：子图有无挂起点
//! - 帧级回收 + 槽级 RC

pub mod EngineCore;
pub mod AsyncRt;
pub mod Schedule;
pub mod Frame;
pub mod Subgraph;
pub mod Strategy;

pub use EngineCore::{Engine, EngineRef};
pub use Schedule::{prepare_frame_nodes, notify_downstream, alloc_const_value};
pub use Subgraph::switch_subgraph;
pub use Strategy::{LockStrategy, Lockable, Single, Multi, QueueHandle};
pub use AsyncRt::{TimerRuntime, AsyncJoinRuntime};

// 调度器常量/辅助来自 EngineCore.rs；此处重新引入到 engine 命名空间，
// 使各子模块 `use super::*` 后可直接以裸名访问（PENDING_EXTERNAL / GOLDEN_RATIO_64 / env_flag）。
// 注意：`Engine` / `EngineRef` 类型已通过上方 `pub use` 重导出，不能在此重复 `use`，
// 否则与 `EngineCore` 模块名在类型命名空间冲突（E0255）。
use EngineCore::{PENDING_EXTERNAL, GOLDEN_RATIO_64, env_flag};
