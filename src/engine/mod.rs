#![allow(non_snake_case)]
//! Engine module — dataflow readiness-driven scheduling execution engine (scheduler).
//!
//! Built on [`crate::ir::Ir::DataFlowGraph`], it implements:
//! - Frame management (HashMap + LockStrategy)
//! - Readiness-driven scheduling (core loop)
//! - Subgraph launch (start/complete)
//! - TimerRuntime / AsyncJoinRuntime out-of-graph runtimes
//! - Single-threaded (Single) / Multi-threaded (Multi) lock strategies
//!
//! The compute_fn function table has been split into Compute.rs; the scheduler invokes
//! it indirectly via `graph.compute_fns[idx]` rather than referencing concrete compute_fns.
//!
//! Submodules:
//! - [`EngineCore`]: core type definitions — `Engine<S>` struct + `EngineRef` factory + Send/Sync + scheduler constants + env_flag
//! - [`AsyncRt`]: async runtime (TimerRuntime / AsyncJoinRuntime) + event handling
//! - [`Schedule`]: dataflow scheduling core (readiness scheduling, batching, run_frame_nodes, process_frame)
//! - [`Frame`]: frame lifecycle management (allocation, initialization, reset, frame chain)
//! - [`Subgraph`]: subgraph invocation and return (switch_subgraph, start_subgraph, complete_and_wake_caller)
//! - [`Strategy`]: concurrency strategies (LockStrategy / Single / Multi / QueueHandle + worker)
//!
//! Design principles (see docs/superpowers/specs/2026-07-31-dataflow-engine-design.md):
//! - No dispatch: the scheduler only recognizes "inputs ready"; nodes carry their own compute_fn
//! - Unified sync/async: subgraphs may or may not contain suspension points
//! - Frame-level reclamation + slot-level RC

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

// Scheduler constants/helpers originate from EngineCore.rs; they are re-imported here into the
// engine namespace so that submodules can use them by bare name after `use super::*`
// (PENDING_EXTERNAL / GOLDEN_RATIO_64 / env_flag).
// Note: the `Engine` / `EngineRef` types are already re-exported via the `pub use` above and must
// not be re-`use`d here, otherwise they conflict with the `EngineCore` module name in the type
// namespace (E0255).
use EngineCore::{PENDING_EXTERNAL, GOLDEN_RATIO_64, env_flag};
