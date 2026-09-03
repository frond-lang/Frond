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
//! - [`EngineCore`]: core type definitions — `Engine<S>` struct + `EngineRef` factory + Send/Sync + scheduler constants + env
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
pub mod Offload;
pub mod Schedule;
pub mod Frame;
pub mod Subgraph;
pub mod Strategy;

pub use EngineCore::{Engine, EngineRef};
pub use Schedule::{prepare_frame_nodes, notify_downstream, alloc_const_value};
pub use Subgraph::switch_subgraph;
pub use Strategy::{LockStrategy, Lockable, Single, Multi, QueueHandle};
pub use AsyncRt::{TimerRuntime, AsyncJoinRuntime};
pub use Frame::{prepare_defer_frame_sync, prepare_same_function_frame_sync};

// Scheduler constants/helpers originate from EngineCore.rs; they are re-imported here into the
// engine namespace so that submodules can use them by bare name after `use super::*`
// (PENDING_EXTERNAL).
// Note: the `Engine` / `EngineRef` types are already re-exported via the `pub use` above and must
// not be re-`use`d here, otherwise they conflict with the `EngineCore` module name in the type
// namespace (E0255).
use EngineCore::PENDING_EXTERNAL;

// =========================================================================
// Program argv — the engine-registered command-line view for std.os.Proc.args()
// =========================================================================
//
// The cli sets the trailing arguments (after `--`) before execution; the
// stdlib C primitive `__os_arg_count/__os_arg_get_into` reads them through the
// `#[no_mangle]` accessors below (direct symbol references from the linked
// frond_extern C objects — no dlsym needed). Never registered → argc 0 +
// args() returns only what was set by the cli (nothing embeds the host argv:
// a compiled .fndo run without `--` sees no arguments).
mod ProgramArgs {
    use std::sync::OnceLock;
    use parking_lot::Mutex;

    // Leaked boxed slices: the C accessors hand raw (ptr, len) pairs into the
    // linked frond_extern objects, so the backing bytes must live for the
    // process lifetime. Leaking one short buffer per argv entry per process
    // is bounded and intentional — the previous Mutex<Vec<Vec<u8>>> +
    // lifetime-transmuted return handed out slices that a second `set`
    // would dangle (and the lock is kept only to serialize re-sets).
    static ARGS: OnceLock<Mutex<Vec<&'static [u8]>>> = OnceLock::new();

    pub fn set(args: Vec<String>) {
        let slot = ARGS.get_or_init(|| Mutex::new(Vec::new()));
        let mut g = slot.lock();
        *g = args
            .into_iter()
            .map(|a| Box::leak(a.into_bytes().into_boxed_slice()) as &'static [u8])
            .collect();
    }

    pub fn count() -> i32 {
        ARGS.get().map(|m| m.lock().len() as i32).unwrap_or(0)
    }

    pub fn get(i: i32) -> Option<&'static [u8]> {
        let m = ARGS.get()?;
        let g = m.lock();
        g.get(i as usize).copied()
    }
}

/// Register the program arguments visible to `std.os.Proc.args()`.
pub fn set_program_args(args: Vec<String>) {
    ProgramArgs::set(args);
}

/// C accessor: argument count (0 when the cli passed no `--` arguments).
#[no_mangle]
pub extern "C" fn frond_runtime_argc() -> i32 {
    ProgramArgs::count()
}

/// C accessor: pointer to argument i's UTF-8 bytes (NULL when out of range).
#[no_mangle]
pub extern "C" fn frond_runtime_arg_ptr(i: i32) -> *const u8 {
    match ProgramArgs::get(i) {
        Some(b) => b.as_ptr(),
        None => core::ptr::null(),
    }
}

/// C accessor: byte length of argument i (0 when out of range).
#[no_mangle]
pub extern "C" fn frond_runtime_arg_len(i: i32) -> usize {
    ProgramArgs::get(i).map(|b| b.len()).unwrap_or(0)
}
