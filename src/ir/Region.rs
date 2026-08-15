//! Region — region tree + structural dominance (W3A).
//!
//! Subgraph ranges nest (a branch/loop subgraph's node range lies inside its
//! enclosing function subgraph's range), which yields a containment tree:
//! the REGION TREE. Structural dominance follows directly:
//!
//! - region A **dominates** region B iff A == B or A's range strictly
//!   contains B's range (A is an ancestor in the tree);
//! - sibling regions (two if-arms, a branch and a loop beside it) never
//!   dominate each other — they are mutually exclusive paths.
//!
//! This is the single authority for "may region B's node reference / merge
//! with / be redirected to region A's node": the Optimizer's CSE merge
//! legality and the Verifier's V2 scoping check both consult it.
//!
//! Caveat (F-3): subgraph ranges may OVERLAP without nesting in degenerate
//! builder output (e.g. a wrapper branch and an inner while loop sharing a
//! start node id). Containment dominance is unaffected — only strict
//! containment counts.

use crate::ir::Ir::{DataFlowGraph, SubGraphId};

/// Containment-based region tree over subgraph node ranges.
#[derive(Debug)]
pub struct RegionTree {
    /// parent[i] = smallest subgraph whose range strictly contains subgraph
    /// i's range (None = top level).
    parent: Vec<Option<SubGraphId>>,
    /// Cached [start, end) ranges for O(1) dominance tests.
    ranges: Vec<(u32, u32)>,
}

impl RegionTree {
    /// Build from the graph's subgraph node ranges.
    pub fn build(graph: &DataFlowGraph) -> Self {
        let ranges: Vec<(u32, u32)> = graph
            .subgraphs
            .iter()
            .map(|sg| (sg.node_range.0 .0, sg.node_range.1 .0))
            .collect();
        let mut parent: Vec<Option<SubGraphId>> = vec![None; ranges.len()];
        for (i, &(is_, ie)) in ranges.iter().enumerate() {
            if is_ >= ie {
                continue; // empty placeholder — never a child
            }
            let mut best: Option<(u32, SubGraphId)> = None; // (len, id)
            for (j, &(js, je)) in ranges.iter().enumerate() {
                if i == j || js >= je {
                    continue;
                }
                if js <= is_ && ie <= je && (js != is_ || je != ie) {
                    let len = je - js;
                    if best.is_none() || len < best.unwrap().0 {
                        best = Some((len, SubGraphId(j as u32)));
                    }
                }
            }
            parent[i] = best.map(|(_, id)| id);
        }
        RegionTree { parent, ranges }
    }

    /// Cached range of a subgraph.
    pub fn range(&self, sg: SubGraphId) -> (u32, u32) {
        self.ranges[sg.0 as usize]
    }

    /// Immediate (smallest strictly containing) parent region.
    pub fn parent_of(&self, sg: SubGraphId) -> Option<SubGraphId> {
        self.parent[sg.0 as usize]
    }

    /// A strictly contains B in the region tree.
    pub fn is_ancestor(&self, a: SubGraphId, b: SubGraphId) -> bool {
        if a == b {
            return false;
        }
        let (as_, ae) = self.ranges[a.0 as usize];
        let (bs, be) = self.ranges[b.0 as usize];
        as_ <= bs && be <= ae
    }

    /// Structural dominance: A dominates B iff A == B or A is an ancestor
    /// region of B. Sibling branches never dominate each other.
    pub fn dominates(&self, a: SubGraphId, b: SubGraphId) -> bool {
        a == b || self.is_ancestor(a, b)
    }

    /// For every node: the innermost (smallest containing) region, or None
    /// when the node lies outside every subgraph range.
    pub fn innermost_all(&self, node_count: usize) -> Vec<Option<SubGraphId>> {
        let mut best: Vec<Option<(u32, SubGraphId)>> = vec![None; node_count];
        for (si, &(start, end)) in self.ranges.iter().enumerate() {
            if start >= end {
                continue;
            }
            let len = end - start;
            for nid in start..end {
                let idx = nid as usize;
                if idx >= node_count {
                    break;
                }
                match &mut best[idx] {
                    None => best[idx] = Some((len, SubGraphId(si as u32))),
                    Some((blen, _)) if len < *blen => {
                        best[idx] = Some((len, SubGraphId(si as u32)));
                    }
                    _ => {}
                }
            }
        }
        best.into_iter().map(|e| e.map(|(_, id)| id)).collect()
    }
}
