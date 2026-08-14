//! Versioning — W2 storage versioning (MemorySSA-lite over trailing inputs).
//!
//! Makes in-place heap writes (record field set / array store / deref write /
//! atomics) visible to the dataflow graph by threading a "storage version"
//! edge, exactly the way `current_effect` threads statement order:
//!
//! - Every aliased read (`CF_RECORD_FIELD_GET` / `CF_ARRAY_INDEX` /
//!   `CF_ARRAY_LEN` / `CF_PATTERN_ADT_FIELD_GET` / deref/atomic/reflect-value
//!   reads) whose storage root (always `inputs[0]`) has a recorded version, or
//!   that follows a potentially-mutating call/gate, gains trailing version
//!   inputs ordering it after the latest writer.
//! - Every in-place write gains trailing inputs ordering it after prior
//!   writers of the same root and after prior calls, and becomes the root's
//!   new version.
//! - Inside loop bodies, reads whose version input lies outside the loop are
//!   re-pointed at the loop's `cond_node`: the node range membership (not the
//!   runtime edge — outer inputs don't gate readiness) is what stops LICM
//!   from hoisting the read out of the loop (the general form of Bug #99),
//!   while per-iteration data freshness is already guaranteed by the
//!   frame-reset protocol.
//!
//! Implementation: a post-build linear scan per function-level subgraph
//! (node order ≈ source order at build time), recursing into nested child
//! subgraphs. State intentionally LEAKS from child scans back into the parent
//! (sibling references produce ignored outer-input edges — conservative, never
//! unsound), and versions pointing into an already-exited child range collapse
//! to the next epoch node (the branch Gate / loop launch Call), which is the
//! same-frame node that actually orders post-branch/post-loop reads.
//!
//! Kill switch: `KUZO_NO_VERSIONING=1`.

use super::Core::IrBuilder;
use crate::ir::Ir::*;
use rustc_hash::FxHashMap;

// Read/write predicates live in Ir.rs (`is_versioned_read_cf` /
// `is_versioned_write_cf`) — shared with the Verifier's V6 check.

/// Nodes that may transitively mutate ANY reachable storage (function calls
/// mutate through references; gates select mutating branches; async/ffi
/// arbitrary effects). They become the "epoch" for subsequent reads.
fn is_epoch_setter(node: Node) -> bool {
    is_launch_kind(node.kind)
        || matches!(effect_class(node.compute_fn), EffectClass::Ffi | EffectClass::Async | EffectClass::Runtime)
}

/// One pending loop-body fixup: an aliased read inside a LoopBody (or nested
/// within one) that must carry a loop-internal input so LICM cannot hoist it
/// out of a mutating loop.
struct BodyFixup {
    read: NodeId,
    /// Positions within the read's input vector that hold version inputs
    /// (empty when the read had no version candidates when scanned).
    positions: Vec<usize>,
    /// Range of the innermost subgraph containing the read.
    body_range: (u32, u32),
}

#[derive(Default)]
struct ScanState {
    /// Storage root node -> latest direct write node.
    versions: FxHashMap<u32, u32>,
    /// Latest epoch node (call/gate/launch) that may have mutated anything.
    epoch: Option<u32>,
    /// Max end of child subgraph ranges already exited at this level; versions
    /// older than this collapse at the next epoch setter.
    last_exited_end: u32,
}

impl IrBuilder<'_> {
    /// Entry: run storage versioning over the whole graph. Call after
    /// `compute_nested_ranges()` and before `compute_downstreams()`.
    pub(super) fn apply_storage_versioning(&mut self) {
        if std::env::var("KUZO_NO_VERSIONING").is_ok() {
            return;
        }
        // Child range -> child subgraph id lookup.
        let mut range_to_sg: FxHashMap<(u32, u32), SubGraphId> = FxHashMap::default();
        for sg in &self.graph.subgraphs {
            range_to_sg.insert((sg.node_range.0 .0, sg.node_range.1 .0), sg.id);
        }
        let fn_sgs: Vec<SubGraphId> = self
            .graph
            .subgraphs
            .iter()
            .filter(|sg| sg.function_id == sg.id.0 && sg.node_range.0 .0 < sg.node_range.1 .0)
            .map(|sg| sg.id)
            .collect();
        for sg_id in fn_sgs {
            let mut state = ScanState::default();
            let mut fixups: Vec<BodyFixup> = Vec::new();
            let in_loop: Option<((u32, u32), NodeId)> = None; // function level is never a LoopBody
            if std::env::var("KUZO_VERSIONING_DBG").is_ok() {
                eprintln!("[VER] scan fn sg {}", sg_id.0);
            }
            self.scan_subgraph(sg_id, &range_to_sg, &mut state, &mut fixups, in_loop);
        }
    }

    /// Linear scan over a subgraph's own nodes (skipping nested child ranges,
    /// which are recursed into). Returns whether any versioned write or epoch
    /// setter was seen (drives the parent's loop-body fixup decision).
    fn scan_subgraph(
        &mut self,
        sg_id: SubGraphId,
        range_to_sg: &FxHashMap<(u32, u32), SubGraphId>,
        state: &mut ScanState,
        fixups: &mut Vec<BodyFixup>,
        in_loop: Option<((u32, u32), NodeId)>,
    ) -> bool {
        let (start, end) = self.graph.subgraphs[sg_id.0 as usize].node_range;
        let mut children: Vec<(u32, u32)> = self.graph.subgraphs[sg_id.0 as usize]
            .nested_ranges
            .clone();
        children.sort_unstable();
        let mut saw_effect = false;
        let mut ci = 0usize; // next child index
        let mut pos = start.0;
        while pos < end.0 {
            // The nested_ranges tables list ALL descendant ranges (not just
            // direct children), overlapping ones included. Skip entries that
            // ended before pos; for one that started at/before pos, its
            // remaining interior belongs to the child — recurse and jump.
            while ci < children.len() && children[ci].1 <= pos {
                ci += 1;
            }
            if ci < children.len() && children[ci].0 <= pos {
                let (cs, ce) = children[ci];
                ci += 1;
                if let Some(&child_id) = range_to_sg.get(&(cs, ce)) {
                    let dbg = std::env::var("KUZO_VERSIONING_DBG").is_ok();
                    if dbg {
                        eprintln!(
                            "[VER] sg{} enter child sg{} range [{}..{}) kind={:?}",
                            sg_id.0, child_id.0, cs, ce,
                            self.graph.subgraphs[child_id.0 as usize].loop_kind
                        );
                    }
                    let child_kind = self.graph.subgraphs[child_id.0 as usize].loop_kind;
                    let child_in_loop = if child_kind == LoopKind::LoopBody {
                        self.graph.subgraphs[child_id.0 as usize]
                            .loop_parent_sg
                            .and_then(|p| {
                                let psg = &self.graph.subgraphs[p.0 as usize];
                                psg.cond_node.map(|c| (psg.node_range.0 .0, psg.node_range.1 .0, c))
                            })
                            .map(|(ls, le, cond)| ((ls, le), cond))
                    } else {
                        in_loop
                    };
                    let is_body = child_kind == LoopKind::LoopBody;
                    // Sibling isolation: the child's internal versions/epoch
                    // must NOT leak into this level's state — a leaked epoch
                    // would attach cross-sibling version edges (reads here
                    // ordering after a mutually-exclusive arm's nodes). Restore
                    // the pre-child state; ordering for code AFTER the child is
                    // provided by the collapse-at-epoch rule when this level's
                    // own Gate/launch-Call (an epoch setter) is scanned.
                    let saved_versions = state.versions.clone();
                    let saved_epoch = state.epoch;
                    let child_saw = self.scan_subgraph(child_id, range_to_sg, state, fixups, child_in_loop);
                    state.versions = saved_versions;
                    state.epoch = saved_epoch;
                    state.last_exited_end = state.last_exited_end.max(ce);
                    if is_body && child_saw {
                        // Loop exit fixup: every aliased read recorded inside
                        // this body must carry an input INSIDE the loop sg
                        // range (normally the cond_node), so LICM cannot hoist
                        // it out of a mutating loop. Runtime data freshness
                        // across iterations already comes from the frame-reset
                        // protocol.
                        let loop_info = self.graph.subgraphs[child_id.0 as usize]
                            .loop_parent_sg
                            .and_then(|p| {
                                let psg = &self.graph.subgraphs[p.0 as usize];
                                psg.cond_node.map(|c| (psg.node_range.0 .0, psg.node_range.1 .0, c))
                            });
                        if let Some((ls, le, cond)) = loop_info {
                            let mut i = 0;
                            while i < fixups.len() {
                                // Containment (not equality): reads nested in
                                // if-arms inside the body carry the arm's range.
                                let r = fixups[i].body_range;
                                if r.0 >= cs && r.1 <= ce {
                                    let f = fixups.remove(i);
                                    self.ensure_loop_input(f, (ls, le), cond);
                                } else {
                                    i += 1;
                                }
                            }
                        }
                    }
                }
                pos = ce;
                continue;
            }
            // Regular own node.
            let idx = pos as usize;
            let node = self.graph.nodes[idx];
            let cf = node.compute_fn;
            if is_versioned_read_cf(cf) {
                let root = self.graph.inputs_pool.get(node.inputs_offset, node.input_count)[0].0;
                let mut vers = state.versions.get(&root).copied();
                // Collapse a stale version (pointing into an exited child
                // range) up to the epoch if one exists.
                if vers.is_some_and(|v| v < state.last_exited_end) {
                    vers = state.epoch.or(vers);
                }
                let candidates: Vec<u32> = [vers, state.epoch]
                    .into_iter()
                    .flatten()
                    .filter(|&c| c != root)
                    .collect();
                let positions = if !candidates.is_empty() {
                    self.append_version_inputs(NodeId(pos), &candidates)
                } else {
                    Vec::new()
                };
                // Record EVERY aliased read inside a loop body (even with no
                // version candidates yet): a later write/call in the same body
                // still makes it iteration-variant, and the fixup will ensure
                // it carries a loop-internal input (cond_node) in that case.
                if in_loop.is_some() {
                    if std::env::var("KUZO_VERSIONING_DBG").is_ok() {
                        eprintln!("[VER] record read {} in body {:?}", pos, (start.0, end.0));
                    }
                    fixups.push(BodyFixup {
                        read: NodeId(pos),
                        positions,
                        body_range: (start.0, end.0),
                    });
                }
            } else if is_versioned_write_cf(cf) {
                let root = self.graph.inputs_pool.get(node.inputs_offset, node.input_count)[0].0;
                let mut vers = state.versions.get(&root).copied();
                if vers.is_some_and(|v| v < state.last_exited_end) {
                    vers = state.epoch.or(vers);
                }
                let candidates: Vec<u32> = [vers, state.epoch]
                    .into_iter()
                    .flatten()
                    .filter(|&c| c != root)
                    .collect();
                if !candidates.is_empty() {
                    self.append_version_inputs(NodeId(pos), &candidates);
                }
                state.versions.insert(root, pos);
                saw_effect = true;
            } else if is_epoch_setter(node) {
                // Any version left behind inside an exited child range (an
                // arm's write, a loop body's write) collapses to this node:
                // the same-frame Gate/Call that actually orders later reads.
                if state.last_exited_end > 0 {
                    let e = pos;
                    state
                        .versions
                        .values_mut()
                        .for_each(|v| {
                            if *v < state.last_exited_end {
                                *v = e;
                            }
                        });
                }
                state.epoch = Some(pos);
                saw_effect = true;
            }
            pos += 1;
        }
        saw_effect
    }

    /// Append version inputs to a node (deduplicated against existing inputs).
    /// Returns the positions of the appended entries.
    fn append_version_inputs(&mut self, node: NodeId, candidates: &[u32]) -> Vec<usize> {
        let idx = node.0 as usize;
        let n = self.graph.nodes[idx];
        let existing: Vec<NodeId> =
            self.graph.inputs_pool.get(n.inputs_offset, n.input_count).to_vec();
        let mut new_inputs = existing.clone();
        let mut positions = Vec::new();
        for &c in candidates {
            let cand = NodeId(c);
            if !new_inputs.contains(&cand) {
                positions.push(new_inputs.len());
                new_inputs.push(cand);
            }
        }
        if positions.is_empty() {
            return positions;
        }
        assert!(
            new_inputs.len() <= 255,
            "versioning: node {} input_count overflow",
            node.0
        );
        let off = self.graph.inputs_pool.push(&new_inputs);
        self.graph.nodes[idx].inputs_offset = off;
        self.graph.nodes[idx].input_count = new_inputs.len() as u8;
        positions
    }

    /// Enforce the loop-input invariant on one recorded read:
    /// 1. appended version inputs pointing outside the loop range are
    ///    re-pointed at the loop's cond_node;
    /// 2. if the read still has no input inside the loop range, cond_node is
    ///    appended (covers reads that had no version candidates when scanned).
    fn ensure_loop_input(&mut self, f: BodyFixup, loop_range: (u32, u32), cond: NodeId) {
        let idx = f.read.0 as usize;
        let n = self.graph.nodes[idx];
        for &p in &f.positions {
            let slot = (n.inputs_offset as usize) + p;
            if slot >= self.graph.inputs_pool.data.len() {
                return; // node inputs were rewritten after recording; skip
            }
            let cur = self.graph.inputs_pool.data[slot].0;
            if cur < loop_range.0 || cur >= loop_range.1 {
                self.graph.inputs_pool.data[slot] = cond;
            }
        }
        // Re-read (offsets may have changed in step 1? no — in-place rewrite
        // only; but be safe) and check for a loop-internal input.
        let n = self.graph.nodes[idx];
        let inputs = self.graph.inputs_pool.get(n.inputs_offset, n.input_count);
        let has_internal = inputs
            .iter()
            .any(|i| i.0 >= loop_range.0 && i.0 < loop_range.1);
        if !has_internal {
            let mut new_inputs = inputs.to_vec();
            if !new_inputs.contains(&cond) {
                new_inputs.push(cond);
                assert!(
                    new_inputs.len() <= 255,
                    "versioning: node {} input_count overflow",
                    f.read.0
                );
                let off = self.graph.inputs_pool.push(&new_inputs);
                self.graph.nodes[idx].inputs_offset = off;
                self.graph.nodes[idx].input_count = new_inputs.len() as u8;
            }
        }
    }
}
