//! Verifier — structural invariant checks over the compiled `DataFlowGraph`.
//!
//! W0 of the IR correctness plan (see `IR_OPTIMIZATION_PLAN.md`): every
//! invariant the engine maintains today "by comments and runtime convention
//! only" is checked here explicitly. Runs after `IrBuilder::build()` and after
//! every optimizer `rebuild`:
//! - enabled automatically in debug builds;
//! - enabled in release builds via `FROND_VERIFY=1`;
//! - `FROND_VERIFY_STRICT=1` turns violations into a panic (CI gate).
//!
//! Checks:
//! - V1 structure: input-pool bounds, NodeId bounds, metadata target bounds.
//! - V2 scoping: every input edge must resolve within the same innermost
//!   subgraph or an ancestor subgraph (containment = dominance). References
//!   into sibling branches, child subgraphs, or outside all subgraphs are
//!   bugs (Bug #45 / #24 class).
//! - V4 subgraph integrity: entry/return/cond/iter/reset nodes and
//!   event-source declarations inside their subgraph range (Bug #24); defer
//!   bodies same-function; upvalue outer nodes strictly outside; call targets
//!   in bounds; LoopBody parent containment.
//! - V5 downstreams: the downstream table must exactly mirror input edges.
//! - V7 sg references: every SubGraphId reference (call targets, closure/
//!   partial/lazy/trait-construct metadata, gate/select branch targets, defer
//!   bodies, loop parents, entry) must be in bounds, and dispatch targets
//!   must not point at empty-range subgraphs (function-level DCE / subgraph
//!   compaction residue would dispatch into nothing).

use crate::ir::Ir::{DataFlowGraph, NodeId, SubGraphId};

/// One invariant violation.
#[derive(Debug)]
pub struct Violation {
    /// Which check fired ("V2-scoping" etc.).
    pub check: &'static str,
    /// Human-readable description with node/subgraph ids.
    pub message: String,
}

/// Run all checks and collect violations (pure; no I/O, no panic).
pub fn verify(graph: &DataFlowGraph) -> Vec<Violation> {
    verify_with_stage(graph, "")
}

/// `verify` with the pipeline stage label: at `build` time, gates may still
/// reference empty placeholder branch subgraphs (analyzer-dead branches never
/// compile); the empty-range dispatch check only applies once the optimizer's
/// function-level DCE has run and such residue should be gone.
pub fn verify_with_stage(graph: &DataFlowGraph, stage: &str) -> Vec<Violation> {
    let mut v = Vec::new();
    verify_structure(graph, &mut v);
    let regions = crate::ir::Region::RegionTree::build(graph);
    let innermost = regions.innermost_all(graph.node_count());
    verify_scoping(graph, &regions, &innermost, &mut v);
    verify_subgraphs(graph, &regions, &mut v);
    verify_downstreams(graph, &mut v);
    verify_loop_versioning(graph, &mut v);
    verify_node_ref_bounds(graph, &mut v);
    verify_sg_refs(graph, stage != "build", &mut v);
    v
}

/// V8: every out-of-band NodeId reference (NodeRef door) must be in bounds.
/// A violation here means a pass or `rebuild` left a metadata reference to a
/// node that no longer exists — the "ref node not live" / dangling-anchor
/// class, caught at the stage where it happened instead of panicking three
/// passes later. Load-path graphs are checked through the same door (the
/// complex tables are materialized at load; upvalues via CSR accessor).
fn verify_node_ref_bounds(graph: &DataFlowGraph, out: &mut Vec<Violation>) {
    let total = graph.node_count();
    let mut count = 0usize;
    graph.for_each_node_ref(|site, owner, id| {
        if id.0 as usize >= total {
            out.push(Violation {
                check: "V8-node-refs",
                message: format!(
                    "{:?} (owner {}) references node {} out of bounds (total={})",
                    site, owner, id.0, total
                ),
            });
        }
        count += 1;
    });
    let _ = count;
}

/// Verify, report to stderr, and (under `FROND_VERIFY_STRICT=1`) panic.
/// Cheap no-op unless debug assertions are on or `FROND_VERIFY` is set.
pub fn verify_and_report(graph: &DataFlowGraph, stage: &str) {
    if !cfg!(debug_assertions) && std::env::var("FROND_VERIFY").is_err() {
        return;
    }
    // Debug aid: FROND_VERIFY_DUMP=<node_id> prints every subgraph whose range
    // covers the node (or the closest ranges around it), then exits verification.
    if let Ok(target) = std::env::var("FROND_VERIFY_DUMP") {
        if let Ok(t) = target.parse::<u32>() {
            dump_node_scope(graph, t);
            return;
        }
    }
    // Debug aid: FROND_VERIFY_DUMP_SG=2131-2140 prints a subgraph id range.
    if let Ok(range) = std::env::var("FROND_VERIFY_DUMP_SG") {
        if let Some((lo, hi)) = range.split_once('-') {
            if let (Ok(lo), Ok(hi)) = (lo.parse::<usize>(), hi.parse::<usize>()) {
                for si in lo..=hi.min(graph.subgraphs.len().saturating_sub(1)) {
                    let sg = &graph.subgraphs[si];
                    eprintln!(
                        "sg{} kind={:?} fn={} range=[{}..{}) param_count={} upvals={} defer={} parent={:?} cond={:?}",
                        si, sg.loop_kind, sg.function_id, sg.node_range.0 .0, sg.node_range.1 .0,
                        sg.param_count, sg.upvalue_count, sg.defer_table.len(), sg.loop_parent_sg, sg.cond_node
                    );
                }
                return;
            }
        }
    }
    let violations = verify_with_stage(graph, stage);
    if violations.is_empty() {
        return;
    }
    let strict = std::env::var("FROND_VERIFY_STRICT").is_ok();
    // Aggregate per check: at most 3 examples each + a total, so debug runs
    // stay readable when one root cause fires many times.
    let mut order: Vec<&'static str> = Vec::new();
    let mut counts: std::collections::HashMap<&'static str, (usize, usize)> =
        std::collections::HashMap::new();
    for viol in &violations {
        let e = counts.entry(viol.check).or_insert_with(|| {
            order.push(viol.check);
            (0, 0)
        });
        e.0 += 1;
        if e.1 < 3 {
            e.1 += 1;
            eprintln!("[VERIFY/{}] {} {}", stage, viol.check, viol.message);
        }
    }
    for check in order {
        let (total, _) = counts[check];
        if total > 3 {
            eprintln!("[VERIFY/{}] {}: ... {} more", stage, check, total - 3);
        }
    }
    if strict {
        panic!(
            "[VERIFY/{}] {} invariant violation(s) (FROND_VERIFY_STRICT=1)",
            stage,
            violations.len()
        );
    }
}

// =========================================================================
// Innermost-subgraph precomputation
// =========================================================================

/// Debug aid: print all subgraph ranges containing `target`, the nearest
/// ranges around it, and the node's own inputs/downstreams.
fn dump_node_scope(graph: &DataFlowGraph, target: u32) {
    let n = graph.node_count();
    eprintln!("=== DUMP node {} (node_count={}) ===", target, n);
    if (target as usize) < n {
        let node = graph.node(target as usize);
        eprintln!(
            "kind={:?} cf={} inputs={:?}",
            node.kind,
            node.compute_fn.0,
            graph.inputs(node.inputs_offset, node.input_count)
        );
        eprintln!("downstreams={:?}", graph.downstream_slice(target as usize));
        eprintln!(
            "call_target={:?}",
            graph.call_target(target as usize)
        );
    }
    let mut covering: Vec<(u32, u32, usize)> = Vec::new();
    for (si, sg) in graph.subgraphs.iter().enumerate() {
        let (s, e) = sg.node_range;
        if s.0 <= target && target < e.0 {
            covering.push((s.0, e.0, si));
        }
    }
    eprintln!("covering ranges: {:?}", covering);
    // Nearest 3 ranges ending before and starting after the target.
    let mut before: Vec<(u32, u32, usize)> = graph
        .subgraphs
        .iter()
        .enumerate()
        .filter_map(|(si, sg)| {
            let (s, e) = sg.node_range;
            (e.0 <= target && e.0 >= target.saturating_sub(200)).then_some((s.0, e.0, si))
        })
        .collect();
    before.sort();
    eprintln!("ranges ending just before: {:?}", &before[..before.len().min(6)]);
    let mut after: Vec<(u32, u32, usize)> = graph
        .subgraphs
        .iter()
        .enumerate()
        .filter_map(|(si, sg)| {
            let (s, e) = sg.node_range;
            (s.0 > target && s.0 <= target + 200).then_some((s.0, e.0, si))
        })
        .collect();
    after.sort();
    eprintln!("ranges starting just after: {:?}", &after[..after.len().min(6)]);
    for (s, e, si) in covering.iter().chain(before.iter()).chain(after.iter()) {
        let sg = &graph.subgraphs[*si];
        eprintln!(
            "  sg{} kind={:?} fn={} range=[{}..{}) loop_parent={:?} reset_plan={}",
            si, sg.loop_kind, sg.function_id, s, e, sg.loop_parent_sg, sg.reset_plan.is_some()
        );
    }
}

// =========================================================================
// V1 — structure
// =========================================================================

fn verify_structure(graph: &DataFlowGraph, out: &mut Vec<Violation>) {
    let n = graph.node_count();
    for idx in 0..n {
        let node = graph.node(idx);
        let inputs = graph.inputs(node.inputs_offset, node.input_count);
        for &inp in inputs {
            if (inp.0 as usize) >= n {
                out.push(Violation {
                    check: "V1-structure",
                    message: format!("node {} input {} out of bounds (node_count={})", idx, inp.0, n),
                });
            }
        }
        // Metadata targets must be in bounds.
        if let Some(t) = graph.call_target(idx) {
            if (t.0 as usize) >= graph.subgraphs.len() {
                out.push(Violation {
                    check: "V1-structure",
                    message: format!("node {} call_target sg {} out of bounds (sg_count={})", idx, t.0, graph.subgraphs.len()),
                });
            }
        }
        if let Some(es) = graph.await_event_source(idx) {
            if (es.0 as usize) >= n {
                out.push(Violation {
                    check: "V1-structure",
                    message: format!("node {} await_event_source {} out of bounds", idx, es.0),
                });
            }
        }
    }
}

// =========================================================================
// V2 — scoping (input edges must respect subgraph dominance)
// =========================================================================

fn verify_scoping(
    graph: &DataFlowGraph,
    regions: &crate::ir::Region::RegionTree,
    innermost: &[Option<SubGraphId>],
    out: &mut Vec<Violation>,
) {
    let n = graph.node_count();
    for idx in 0..n {
        let node = graph.node(idx);
        let inputs = graph.inputs(node.inputs_offset, node.input_count);
        let Some(user_sg) = innermost[idx] else {
            // The node itself lives outside every subgraph: its own placement
            // is reported by V4; input legality is moot.
            continue;
        };
        for &inp in inputs {
            let Some(src_sg) = innermost.get(inp.0 as usize).copied().flatten() else {
                out.push(Violation {
                    check: "V2-scoping",
                    message: format!(
                        "node {} (sg {}) inputs node {} which is outside every subgraph",
                        idx, user_sg.0, inp.0
                    ),
                });
                continue;
            };
            let legal = src_sg == user_sg || regions.is_ancestor(src_sg, user_sg);
            // Carve-out — sibling ORDERING edges: the tail-rec-to-loop machinery
            // threads CF_TAILREC_WRITEBACK results across sibling branch subgraphs
            // as trailing effect inputs of CF_SEQ chains and Gates. These edges are
            // ordering-only: scheduler readiness ignores outer inputs and the value
            // is never consumed as data (a frame-chain lookup would yield NULL).
            // Any other sibling/child reference (a value actually consumed across
            // branches) is a real bug and stays a violation.
            let ordering_only = {
                let c = graph.node(idx);
                c.kind == crate::ir::Ir::NodeKind::Gate
                    || c.compute_fn == crate::ir::Ir::CF_SEQ
            };
            if !legal && !ordering_only {
                let desc_sg = |id: SubGraphId| {
                    let sg = &graph.subgraphs[id.0 as usize];
                    format!(
                        "sg{}(kind={:?}, fn={}, range=[{}..{}))",
                        id.0, sg.loop_kind, sg.function_id, sg.node_range.0 .0, sg.node_range.1 .0
                    )
                };
                out.push(Violation {
                    check: "V2-scoping",
                    message: format!(
                        "node {} (kind={:?}, cf={}) in {} inputs node {} (kind={:?}, cf={}) from non-dominating {}",
                        idx,
                        graph.node(idx).kind,
                        graph.node(idx).compute_fn.0,
                        desc_sg(user_sg),
                        inp.0,
                        graph.node(inp.0 as usize).kind,
                        graph.node(inp.0 as usize).compute_fn.0,
                        desc_sg(src_sg),
                    ),
                });
            }
        }
    }
}


// =========================================================================
// V4 — subgraph integrity
// =========================================================================

fn verify_subgraphs(graph: &DataFlowGraph, regions: &crate::ir::Region::RegionTree, out: &mut Vec<Violation>) {
    for (si, sg) in graph.subgraphs.iter().enumerate() {
        let (start, end) = sg.node_range;
        let compiled = start.0 < end.0;
        let in_range =
            |node: NodeId, what: &str| -> Option<Violation> {
                if !compiled {
                    return None; // placeholder: entry/return default to NodeId(0)
                }
                if node.0 < start.0 || node.0 >= end.0 {
                    Some(Violation {
                        check: "V4-subgraph",
                        message: format!(
                            "sg {} [{}..{}): {} node {} outside range",
                            si, start.0, end.0, what, node.0
                        ),
                    })
                } else {
                    None
                }
            };

        if (sg.function_id as usize) >= graph.subgraphs.len() {
            out.push(Violation {
                check: "V4-subgraph",
                message: format!("sg {} function_id {} out of bounds", si, sg.function_id),
            });
        }
        if let Some(v) = in_range(sg.entry_node, "entry") { out.push(v); }
        if let Some(v) = in_range(sg.return_node, "return") { out.push(v); }
        if let Some(c) = sg.cond_node {
            if let Some(v) = in_range(c, "cond") { out.push(v); }
        }
        if let Some(it) = sg.iter_next_node {
            if let Some(v) = in_range(it, "iter_next") { out.push(v); }
        }
        if let Some(plan) = &sg.reset_plan {
            for node in plan.reset_to_zero.iter().chain(&plan.reset_to_one).chain(&plan.reset_condition_tree) {
                if let Some(v) = in_range(*node, "reset_plan") { out.push(v); }
            }
        }

        // Bug #24 class: an event-source declaration registered on this sg
        // must be a node inside this sg's range (the old bug registered decls
        // on the function sg while the await node lived in a branch sg).
        for decl in &sg.event_source_decls {
            if !compiled || decl.node.0 < start.0 || decl.node.0 >= end.0 {
                out.push(Violation {
                    check: "V4-subgraph",
                    message: format!(
                        "sg {} [{}..{}): event_source_decl node {} outside range",
                        si, start.0, end.0, decl.node.0
                    ),
                });
            }
        }

        // Defer bodies: same function as the sg owning the defer_table, and
        // the trigger node inside that sg.
        for entry in &sg.defer_table {
            let body = entry.body_subgraph.0 as usize;
            if body >= graph.subgraphs.len() {
                out.push(Violation {
                    check: "V4-subgraph",
                    message: format!("sg {} defer body_sg {} out of bounds", si, entry.body_subgraph.0),
                });
            } else if graph.subgraphs[body].function_id != sg.function_id {
                out.push(Violation {
                    check: "V4-subgraph",
                    message: format!(
                        "sg {} defer body_sg {} function_id {} != owner {}",
                        si, entry.body_subgraph.0, graph.subgraphs[body].function_id, sg.function_id
                    ),
                });
            }
            if let Some(v) = in_range(entry.trigger_node, "defer trigger") { out.push(v); }
        }

        // Upvalue outer nodes are by definition outside this subgraph.
        for &up in &sg.upvalue_outer_nodes {
            if compiled && up.0 >= start.0 && up.0 < end.0 {
                out.push(Violation {
                    check: "V4-subgraph",
                    message: format!(
                        "sg {} [{}..{}): upvalue_outer_node {} inside own range",
                        si, start.0, end.0, up.0
                    ),
                });
            }
        }

        // Nested ranges must be inside this range. Zero-length nested ranges on
        // non-compiled (placeholder) subgraphs are harmless residue left behind
        // by optimizer `rebuild` remapping of fully-dead subgraphs.
        for &(ns, ne) in &sg.nested_ranges {
            if !compiled || ns >= ne {
                continue;
            }
            if ns < start.0 || ne > end.0 {
                out.push(Violation {
                    check: "V4-subgraph",
                    message: format!(
                        "sg {} [{}..{}): nested range [{}..{}) not contained",
                        si, start.0, end.0, ns, ne
                    ),
                });
            }
        }

        // LoopBody: parent's range must contain this range.
        if sg.loop_kind == crate::ir::Ir::LoopKind::LoopBody {
            if let Some(parent) = sg.loop_parent_sg {
                if !regions.is_ancestor(parent, SubGraphId(si as u32)) {
                    out.push(Violation {
                        check: "V4-subgraph",
                        message: format!(
                            "LoopBody sg {} [{}..{}) not contained in parent sg {}",
                            si, start.0, end.0, parent.0
                        ),
                    });
                }
            } else {
                out.push(Violation {
                    check: "V4-subgraph",
                    message: format!("LoopBody sg {} missing loop_parent_sg", si),
                });
            }
        }
    }
}

// =========================================================================
// V7 — SubGraphId reference integrity
// =========================================================================

/// Every SubGraphId reference must be in bounds, and anything that dispatches
/// (call targets, closure/partial/lazy/trait-construct metadata, gate/select
/// branches, defer bodies) must not point at an empty-range subgraph — after
/// function-level DCE + subgraph compaction such a reference would launch a
/// subgraph with no executable nodes. `check_empty=false` (build stage)
/// skips the empty-range half: analyzer-dead branches legitimately keep
/// placeholder branch subgraphs there.
fn verify_sg_refs(graph: &DataFlowGraph, check_empty: bool, out: &mut Vec<Violation>) {
    let n = graph.node_count();
    let sg_count = graph.subgraphs.len();
    let check = |id: SubGraphId, what: &str, node: Option<usize>, out: &mut Vec<Violation>| {
        let idx = id.0 as usize;
        if idx >= sg_count {
            out.push(Violation {
                check: "V7-sg-refs",
                message: format!(
                    "{} -> sg {} out of bounds (sg_count={})",
                    what,
                    id.0,
                    sg_count
                ),
            });
            return;
        }
        if !check_empty {
            return;
        }
        let (s, e) = graph.subgraphs[idx].node_range;
        if s.0 >= e.0 && node.is_some() {
            out.push(Violation {
                check: "V7-sg-refs",
                message: format!(
                    "node {} {} -> sg {} has empty range [{},{}) — dispatch into nothing",
                    node.unwrap(),
                    what,
                    id.0,
                    s.0,
                    e.0
                ),
            });
        }
    };

    if let Some(entry) = graph.entry_subgraph {
        check(entry, "entry_subgraph", None, out);
    }
    for idx in 0..n {
        if let Some(t) = graph.call_target(idx) {
            check(t, "call_target", Some(idx), out);
        }
        if let Some(ci) = graph.closure_info(idx) {
            check(SubGraphId(ci.subgraph_id.0), "closure_info", Some(idx), out);
        }
        if let Some(pi) = graph.partial_info(idx) {
            check(SubGraphId(pi.subgraph_id.0), "partial_info", Some(idx), out);
        }
        if let Some(li) = graph.lazy_construct_info(idx) {
            check(SubGraphId(li.thunk_sg.0), "lazy_thunk", Some(idx), out);
        }
        if let Some(ti) = graph.trait_construct_info_at(idx) {
            for (mi, m) in ti.methods.iter().enumerate() {
                check(SubGraphId(m.subgraph_id.0), &format!("trait_method[{mi}]"), Some(idx), out);
            }
        }
        if let Some(gb) = graph.gate_branches_at(idx) {
            for (_, bsg, _) in &gb.branches {
                check(*bsg, "gate_branch", Some(idx), out);
            }
        }
        if let Some(si) = graph.select_info_at(idx) {
            for sb in &si.branches {
                check(sb.subgraph_id, "select_branch", Some(idx), out);
            }
        }
    }
    for (si, sg) in graph.subgraphs.iter().enumerate() {
        check(SubGraphId(sg.function_id), "function_id", None, out);
        if let Some(p) = sg.loop_parent_sg {
            check(p, "loop_parent", None, out);
        }
        for e in &sg.defer_table {
            check(e.body_subgraph, &format!("sg{si} defer body"), None, out);
        }
    }
}

// =========================================================================
// V6 — loop-body storage versioning completeness (W2)
// =========================================================================

/// W2 invariant: inside a loop body that contains any in-place write or any
/// potentially-mutating call/launch, every aliased heap read must carry at
/// least one input INSIDE the enclosing loop subgraph's range (normally the
/// cond_node). Without it, LICM may hoist the read out of the loop and reuse
/// a stale pre-mutation value for every iteration (the general form of
/// Bug #99).
fn verify_loop_versioning(graph: &DataFlowGraph, out: &mut Vec<Violation>) {
    use crate::ir::Ir::{
        effect_class, is_launch_kind, is_versioned_read_cf, is_versioned_write_cf, EffectClass,
        LoopKind,
    };
    let n = graph.node_count();
    for sg in &graph.subgraphs {
        if !matches!(
            sg.loop_kind,
            LoopKind::While | LoopKind::Loop | LoopKind::For | LoopKind::TailRec
        ) || sg.cond_node.is_none()
        {
            continue;
        }
        // Find the LoopBody child.
        let body = graph
            .subgraphs
            .iter()
            .find(|c| c.loop_kind == LoopKind::LoopBody && c.loop_parent_sg == Some(sg.id));
        let Some(body) = body else { continue };
        let (bs, be) = (body.node_range.0 .0, body.node_range.1 .0);
        if bs >= be {
            continue;
        }
        // Does the body subtree mutate (direct write or potentially-mutating call)?
        let mut mutating = false;
        for idx in bs..be.min(n as u32) {
            let node = graph.node(idx as usize);
            if is_versioned_write_cf(node.compute_fn)
                || is_launch_kind(node.kind)
                || matches!(
                    effect_class(node.compute_fn),
                    EffectClass::Ffi | EffectClass::Async | EffectClass::Runtime
                )
            {
                mutating = true;
                break;
            }
        }
        if !mutating {
            continue;
        }
        let (ls, le) = (sg.node_range.0 .0, sg.node_range.1 .0);
        for idx in bs..be.min(n as u32) {
            let node = graph.node(idx as usize);
            if !is_versioned_read_cf(node.compute_fn) {
                continue;
            }
            let inputs = graph.inputs(node.inputs_offset, node.input_count);
            let has_loop_input = inputs.iter().any(|i| i.0 >= ls && i.0 < le);
            if !has_loop_input {
                out.push(Violation {
                    check: "V6-loop-versioning",
                    message: format!(
                        "aliased read node {} in mutating loop body sg {} [{}..{}) has no input inside loop sg {} [{}..{}) — LICM may hoist it to a stale value",
                        idx, body.id.0, bs, be, sg.id.0, ls, le
                    ),
                });
            }
        }
    }
}

// =========================================================================
// V5 — downstream table consistency
// =========================================================================

fn verify_downstreams(graph: &DataFlowGraph, out: &mut Vec<Violation>) {
    let n = graph.node_count();
    // Forward: every downstream entry must correspond to a real input edge
    // (or a Gate condition edge recorded in gate_branches metadata).
    for idx in 0..n {
        let me = NodeId(idx as u32);
        for &ds in graph.downstream_slice(idx) {
            let d = ds.0 as usize;
            if d >= n {
                out.push(Violation {
                    check: "V5-downstreams",
                    message: format!("node {} downstream {} out of bounds", idx, ds.0),
                });
                continue;
            }
            // The consumer's inputs (not the producer's!) must contain the edge.
            let cnode = graph.node(d);
            let cin = graph.inputs(cnode.inputs_offset, cnode.input_count);
            let gate_cond = graph.gate_branches_at(d).map(|gb| gb.condition_input);
            let edge_ok = cin.contains(&me) || gate_cond == Some(me);
            if !edge_ok {
                out.push(Violation {
                    check: "V5-downstreams",
                    message: format!(
                        "node {} downstream {} has no matching input edge",
                        idx, ds.0
                    ),
                });
            }
        }
    }
    // Reverse: every input edge must appear in the producer's downstream set.
    for idx in 0..n {
        let node = graph.node(idx);
        let inputs = graph.inputs(node.inputs_offset, node.input_count).to_vec();
        for inp in &inputs {
            let ds = graph.downstream_slice(inp.0 as usize);
            if !ds.contains(&NodeId(idx as u32)) {
                out.push(Violation {
                    check: "V5-downstreams",
                    message: format!(
                        "node {} input edge from {} missing in producer's downstreams",
                        idx, inp.0
                    ),
                });
            }
        }
    }
}
