//! DataFlowGraph zerocopy accessor layer.
//!
//! When `DataFlowGraph.mem = Some(GraphMemory)` (the `.fndo` loading path):
//! - `nodes` and `inputs` are read directly from the mmap'd byte slices via
//!   accessor methods (v2 packed 4B/8B node records; the inputs-offset column
//!   is either in-record or a load-time prefix table), with no copy into
//!   owned `Vec`s;
//! - the sparse per-Node scalar/composite/string tables (categories A/C/D)
//!   are scatter-materialized at load into owned `Vec<Option<T>>`s (v2 sparse
//!   sections hold only present entries — tiny), so their accessors read
//!   owned fields with no `mem` branching;
//! - the five variable-length complex tables (`gate_branches` /
//!   `record_lit_infos` / `select_infos` / `trait_construct_infos` /
//!   `record_extend_infos`) and `const_values` keep per-node byte-offset
//!   tables and parse single entries on demand from the mmap blob;
//! - `downstreams` were dropped from the v2 format and are re-derived at load
//!   into a flat CSR (`downstream_csr_offsets` / `downstream_csr_flat`).
//!
//! When `mem = None` (the build path), accessor methods read the owned `Vec`
//! fields.

#![allow(non_snake_case)]

use crate::ir::Ir::*;
use super::Spec::*;

// ==================== LE read helpers ====================

#[inline]
fn rd_u32(r: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([r[off], r[off + 1], r[off + 2], r[off + 3]])
}

#[inline]
fn rd_u16(r: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([r[off], r[off + 1]])
}

// ==================== Accessor methods ====================

impl DataFlowGraph {
    // ---- Counts ----

    /// Total node count (build path = `nodes.len()`; load path = `header.node_count`).
    #[inline]
    pub fn node_count(&self) -> usize {
        if let Some(ref mem) = self.mem {
            mem.header().node_count as usize
        } else {
            self.nodes.len()
        }
    }

    /// Whether the v2 Nodes section elides the per-node inputs_offset column
    /// (contiguous inputs pool; offsets come from the load-time prefix table).
    #[inline]
    fn node_inputs_elided(&self) -> bool {
        debug_assert!(self.mem.is_some() || self.node_input_offsets.is_empty());
        !self.node_input_offsets.is_empty()
    }

    // ---- Node ----

    /// Reads a node by index (load path: v2 packed record — 4B when offsets
    /// are elided, 8B otherwise — read from the mmap slice).
    #[inline]
    pub fn node(&self, idx: usize) -> Node {
        // Materialized fast path: .fndo graphs copy the packed Nodes
        // section into `nodes` once at EngineRef::new (per-call mmap
        // unpacking was the artifact 2-3x hot-loop regression);
        // build-path graphs live here natively.
        if !self.nodes.is_empty() {
            return self.nodes[idx];
        }
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::Nodes);
            let elided = self.node_inputs_elided();
            let off = if elided { idx * 4 } else { idx * 8 };
            Node {
                kind: u8_to_node_kind(r[off]),
                input_count: r[off + 1],
                inputs_offset: if elided {
                    self.node_input_offsets[idx]
                } else {
                    rd_u32(r, off + 4)
                },
                compute_fn: ComputeFnId(rd_u16(r, off + 2) as u32),
            }
        } else {
            self.nodes[idx]
        }
    }

    // ---- Inputs ----

    /// Reads the input slice for a node (zerocopy: transmuted from the mmap Inputs section into `&[NodeId]`).
    #[inline]
    pub fn inputs(&self, offset: u32, count: u8) -> &[NodeId] {
        // Materialized fast path (see node()); build-path graphs live
        // in the pool natively.
        if !self.inputs_pool.data.is_empty() {
            return self.inputs_pool.get(offset, count);
        }
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::Inputs);
            let start = offset as usize * 4;
            let n = count as usize;
            // SAFETY: NodeId is #[repr(transparent)] over u32 (4 bytes, 4-byte aligned).
            // The Inputs section starts 4-byte aligned (section align = 4), and start = offset*4 is a multiple of 4.
            // The slice [start..start+n*4] is within the section bounds (the IR guarantees offset+count <= total).
            unsafe {
                std::slice::from_raw_parts(r.as_ptr().add(start) as *const NodeId, n)
            }
        } else {
            self.inputs_pool.get(offset, count)
        }
    }

    // ---- Category A / C / D tables ----
    // v2: scatter-materialized into owned fields on BOTH the eager and
    // zerocopy load paths (sparse sections only carry present entries), so
    // these accessors need no `mem` branching.

    #[inline]
    pub fn call_target(&self, idx: usize) -> Option<SubGraphId> { self.call_targets[idx] }
    #[inline]
    pub fn field_access_info(&self, idx: usize) -> Option<u16> { self.field_access_infos[idx] }
    #[inline]
    pub fn vtable_call_method(&self, idx: usize) -> Option<u16> { self.vtable_call_methods[idx] }
    #[inline]
    pub fn await_event_source(&self, idx: usize) -> Option<NodeId> { self.await_event_sources[idx] }
    #[inline]
    pub fn global_load_slot(&self, idx: usize) -> Option<u32> { self.global_load_slots[idx] }
    #[inline]
    pub fn global_store_slot(&self, idx: usize) -> Option<u32> { self.global_store_slots[idx] }
    #[inline]
    pub fn pattern_field_index(&self, idx: usize) -> Option<u16> { self.pattern_field_indices[idx] }
    #[inline]
    pub fn closure_call_arg_count(&self, idx: usize) -> Option<u8> { self.closure_call_arg_counts[idx] }
    #[inline]
    pub fn lib_ret_kind(&self, idx: usize) -> Option<u8> { self.lib_ret_kinds.get(idx).copied().flatten() }
    #[inline]
    pub fn embed_info(&self, idx: usize) -> Option<u32> { self.embed_infos.get(idx).copied().flatten() }

    /// Lib.embed resource by index (original path, bytes). Both paths read the
    /// owned Vec (load path materializes from the CResources section).
    pub fn resource(&self, idx: usize) -> Option<(&str, &[u8])> {
        self.resources.get(idx).map(|(n, b)| (n.as_ref(), b.as_ref()))
    }

    // hoisted metadata: dropped from the v2 format — loaded graphs fill
    // sentinel values (no runtime consumer; post-rebuild hoisted nodes are
    // covered by their owning ranges).
    #[inline]
    pub fn hoisted_owner(&self, idx: usize) -> SubGraphId { self.hoisted_owners[idx] }
    #[inline]
    pub fn is_hoisted_node(&self, idx: usize) -> bool { self.hoisted_node[idx] }

    // ---- Category B: boolean tables (zerocopy, bitmap read) ----

    #[inline]
    pub fn tail_call_flag(&self, idx: usize) -> bool {
        // Materialized fast path (see node()): .fndo graphs fill the
        // Vec at EngineRef::new; per-call bitmap reads through the
        // mmap section cost measurably on hot loops.
        if !self.tail_call_flags.is_empty() {
            return self.tail_call_flags[idx];
        }
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::TailCallFlags);
            r[idx / 8] & (1 << (idx % 8)) != 0
        } else {
            self.tail_call_flags[idx]
        }
    }

    #[inline]
    pub fn safe_op_flag(&self, idx: usize) -> bool {
        // Materialized fast path (see node()): .fndo graphs fill the
        // Vec at EngineRef::new; per-call bitmap reads through the
        // mmap section cost measurably on hot loops.
        if !self.safe_op_flags.is_empty() {
            return self.safe_op_flags[idx];
        }
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::SafeOpFlags);
            r[idx / 8] & (1 << (idx % 8)) != 0
        } else {
            self.safe_op_flags[idx]
        }
    }

    #[inline]
    pub fn slice_inclusive(&self, idx: usize) -> bool {
        // Materialized fast path (see node()): .fndo graphs fill the
        // Vec at EngineRef::new; per-call bitmap reads through the
        // mmap section cost measurably on hot loops.
        if !self.slice_inclusive.is_empty() {
            return self.slice_inclusive[idx];
        }
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::SliceInclusive);
            r[idx / 8] & (1 << (idx % 8)) != 0
        } else {
            self.slice_inclusive[idx]
        }
    }

    // ---- Category C: tables with strings (materialized at load; owned read) ----

    #[inline]
    pub fn ffi_call_name(&self, idx: usize) -> Option<&str> { self.ffi_call_names[idx].as_deref() }
    #[inline]
    pub fn field_set_name(&self, idx: usize) -> Option<&str> { self.field_set_names[idx].as_deref() }
    #[inline]
    pub fn pattern_ctor_name(&self, idx: usize) -> Option<&str> { self.pattern_ctor_names[idx].as_deref() }
    #[inline]
    pub fn pattern_type_name(&self, idx: usize) -> Option<&str> { self.pattern_type_names[idx].as_deref() }
    #[inline]
    pub fn cast_target_type(&self, idx: usize) -> Option<&str> { self.cast_target_types[idx].as_deref() }

    // ---- Category D: fixed-width composite tables (materialized at load) ----

    #[inline]
    pub fn closure_info(&self, idx: usize) -> Option<ClosureInfo> {
        self.closure_infos[idx].clone()
    }

    /// stdlib @extern("C") #{ }# inline FFI call info (materialized at load;
    /// v2 serializes it — the v1 gap that panicked `frond run <file>.fndo` is
    /// closed).
    #[inline]
    pub fn dyn_ffi_info(&self, idx: usize) -> Option<DynFfiInfo> {
        self.dyn_ffi_infos[idx].clone()
    }

    #[inline]
    pub fn partial_info(&self, idx: usize) -> Option<PartialInfo> {
        self.partial_infos[idx].clone()
    }

    #[inline]
    pub fn lazy_construct_info(&self, idx: usize) -> Option<LazyConstructInfo> {
        self.lazy_construct_infos[idx].clone()
    }

    #[inline]
    pub fn memo_info(&self, idx: usize) -> Option<MemoInfo> {
        self.memo_infos[idx].clone()
    }

    #[inline]
    pub fn batch_info(&self, idx: usize) -> Option<BatchInfo> {
        self.batch_infos[idx].clone()
    }

    // ---- const_values (sparse fixed-stride blob; on-demand binary search) ----

    /// Parses a ConstValue on demand. Load path: the v2 section is
    /// `[count][ (idx u32, off u32) *count ][blob]` with a fixed 17B blob
    /// stride (tag + 16B payload), binary-searchable by node idx. Hot reads
    /// are covered by the engine's `const_cache` (populated once at start).
    #[inline]
    pub fn const_value(&self, idx: usize) -> Option<ConstValue> {
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::ConstValues);
            let count = rd_u32(r, 0) as usize;
            // Binary search the index region (entries sorted by idx).
            let (mut lo, mut hi) = (0usize, count);
            let mut found: Option<u32> = None;
            while lo < hi {
                let mid = (lo + hi) / 2;
                let key = rd_u32(r, 4 + mid * 8);
                if key == idx as u32 {
                    found = Some(rd_u32(r, 4 + mid * 8 + 4));
                    break;
                } else if key < idx as u32 {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let off = found? as usize;
            let blob_start = 4 + count * 8;
            let b = blob_start + off;
            let tag = r[b];
            if tag == 0 { return None; }
            // v3: variable-width payload.
            let w = super::Spec::const_payload_len(tag);
            Some(super::Format::parse_const_value(tag, &r[b + 1..b + 1 + w]))
        } else {
            self.const_values[idx].clone()
        }
    }

    // ---- Downstreams (load path: flat CSR derived at load) ----

    /// Returns the downstream-node slice for node `idx`.
    ///
    /// Load path: the v2 format no longer serializes Downstreams; the flat
    /// CSR table (`downstream_csr_offsets` / `downstream_csr_flat`) is derived
    /// once at load from inputs + gate condition edges.
    ///
    /// Build path: returns a reference to `self.downstreams[idx]`.
    #[inline]
    pub fn downstream_slice(&self, idx: usize) -> &[NodeId] {
        if self.mem.is_some() {
            let start = self.downstream_csr_offsets[idx] as usize;
            let end = self.downstream_csr_offsets[idx + 1] as usize;
            &self.downstream_csr_flat[start..end]
        } else {
            &self.downstreams[idx]
        }
    }

    /// E4 perf: flat per-node downstream consumer count (materialized once at engine start).
    /// Replaces the `downstream_slice(idx).len()` CSR arithmetic on every set_value.
    #[inline]
    pub fn downstream_count(&self, idx: usize) -> u16 {
        if !self.downstream_counts.is_empty() {
            self.downstream_counts[idx]
        } else {
            self.downstream_slice(idx).len() as u16
        }
    }

    // ---- String Pool (zerocopy: load path references mmap directly, avoiding .to_vec() copies) ----

    /// Returns the string pool byte slice.
    ///
    /// Load path (`mem = Some`): returns the `&[u8]` slice of the mmap StringPool
    /// section directly, with no heap allocation (eliminates the `.to_vec()` copy,
    /// typically saving several KB).
    ///
    /// Build path (`mem = None`): returns `&self.string_pool[..]`.
    #[inline]
    pub fn string_pool_slice(&self) -> &[u8] {
        if let Some(ref mem) = self.mem {
            mem.string_pool()
        } else {
            &self.string_pool[..]
        }
    }

    // ---- SubGraph variable-length fields (zerocopy CSR: eliminates per-subgraph Vec heap allocations) ----

    /// Returns the `upvalue_outer_nodes` slice for subgraph `sg_idx`.
    ///
    /// Load path: returns a `&[NodeId]` slice directly from the mmap
    /// SgUpvalueNodes section with no heap allocation.
    ///
    /// Build path: returns a reference to `self.subgraphs[sg_idx].upvalue_outer_nodes`.
    #[inline]
    pub fn sg_upvalue_outer_nodes(&self, sg_idx: usize) -> &[NodeId] {
        if let Some(ref mem) = self.mem {
            let (off, len) = self.sg_uv_offsets[sg_idx];
            let r = mem.section(SectionKind::SgUpvalueNodes);
            let byte_start = off as usize;
            let count = len as usize;
            // SAFETY: NodeId is #[repr(transparent)] over u32 (4 bytes, 4-byte aligned).
            // The SgUpvalueNodes section is 4-byte aligned, and offset was written during serialization (a multiple of 4).
            // The slice range is within the section bounds (serialization guarantees offset + count*4 <= section len).
            unsafe {
                std::slice::from_raw_parts(r.as_ptr().add(byte_start) as *const NodeId, count)
            }
        } else {
            &self.subgraphs[sg_idx].upvalue_outer_nodes
        }
    }

    /// Returns the `nested_ranges` slice for subgraph `sg_idx`.
    ///
    /// v3: nested_ranges are derived data (recomputed by
    /// `compute_nested_ranges` at build, after every optimizer rebuild, and
    /// at load) — both paths read the owned Vec.
    #[inline]
    pub fn sg_nested_ranges(&self, sg_idx: usize) -> &[(u32, u32)] {
        &self.subgraphs[sg_idx].nested_ranges
    }

    // ---- Five complex variable-length table on-demand accessors (zerocopy: eliminates Vec<Option<T>> arrays) ----

    /// Lightweight check whether node `idx` has a SelectInfo (hot path; constructs no owned data).
    #[inline]
    pub fn has_select_info(&self, idx: usize) -> bool {
        if self.mem.is_some() {
            self.select_info_offsets.get(idx).map_or(false, |&o| o != u32::MAX)
        } else {
            self.select_infos.get(idx).map_or(false, |v| v.is_some())
        }
    }

    /// Parses GateBranches on demand (load path parses a single entry from the mmap section).
    ///
    /// E0 perf note: returns a borrowed entry. The mmap (zerocopy) path relies on the one-time
    /// `materialize_gate_branches()` pass at load (see Format.rs) — every Gate execution and every
    /// call-completion capture check read the materialized entry without deep-cloning branch Vecs.
    pub fn gate_branches_at(&self, idx: usize) -> Option<&GateBranches> {
        self.gate_branches.get(idx).and_then(|g| g.as_ref())
    }

    /// One-time eager materialization of all GateBranches entries for zerocopy (mmap) graphs.
    /// Build-path graphs already own `gate_branches` (no-op when already populated).
    /// Must run while the graph is still owned (before Arc wrap), i.e. at the end of both loaders.
    pub fn materialize_gate_branches(&mut self) {
        if self.mem.is_none() || !self.gate_branches.is_empty() {
            return;
        }
        let offsets = std::mem::take(&mut self.gate_branch_offsets);
        let mut table: Vec<Option<GateBranches>> = Vec::with_capacity(offsets.len());
        for i in 0..offsets.len() {
            table.push(self.parse_gate_branches_entry(i, &offsets));
        }
        self.gate_branch_offsets = offsets;
        self.gate_branches = table;
    }

    /// Parses a single GateBranches entry from the mmap GateBranches section (by offsets table).
    fn parse_gate_branches_entry(
        &self,
        idx: usize,
        offsets: &[u32],
    ) -> Option<GateBranches> {
        let mem = self.mem.as_ref()?;
        let off = offsets.get(idx).copied()?;
        if off == u32::MAX {
            return None;
        }
        let section = mem.section(SectionKind::GateBranches);
        let mut r = &section[off as usize..];
        let valid = super::Spec::read_u8(&mut r);
        if valid == 0 {
            return None;
        }
        // Validity byte encodes the W4c capture flag: 2 = valid + capture.
        let capture = valid == 2;
        let condition_input = NodeId(super::Spec::read_u32(&mut r));
        let branch_count = super::Spec::read_u32(&mut r) as usize;
        let mut branches = Vec::with_capacity(branch_count);
        for _ in 0..branch_count {
            let cond = super::Spec::read_u8(&mut r) != 0;
            let sg = SubGraphId(super::Spec::read_u32(&mut r));
            let input_count = super::Spec::read_u32(&mut r) as usize;
            let mut inputs = Vec::with_capacity(input_count);
            for _ in 0..input_count {
                inputs.push(NodeId(super::Spec::read_u32(&mut r)));
            }
            branches.push((cond, sg, inputs));
        }
        Some(GateBranches { condition_input, branches, capture })
    }

    /// Parses RecordLitInfo on demand (load path parses from the mmap section + string_pool).
    pub fn record_lit_info_at(&self, idx: usize) -> Option<RecordLitInfo> {
        if let Some(ref mem) = self.mem {
            let off = self.record_lit_info_offsets.get(idx).copied()?;
            if off == u32::MAX { return None; }
            let section = mem.section(SectionKind::RecordLitInfos);
            let mut r = &section[off as usize..];
            let valid = super::Spec::read_u8(&mut r);
            if valid == 0 { return None; }
            let type_name = { let o = super::Spec::read_u32(&mut r); let l = super::Spec::read_u32(&mut r); mem.read_str(o, l) };
            let fn_count = super::Spec::read_u32(&mut r) as usize;
            let mut field_names = Vec::with_capacity(fn_count);
            for _ in 0..fn_count {
                let o = super::Spec::read_u32(&mut r); let l = super::Spec::read_u32(&mut r);
                field_names.push(if o == u32::MAX { None } else { Some(mem.read_str(o, l)) });
            }
            let constructor = { let o = super::Spec::read_u32(&mut r); let l = super::Spec::read_u32(&mut r); mem.read_str(o, l) };
            let kind = super::Spec::u8_to_record_lit_kind(super::Spec::read_u8(&mut r));
            Some(RecordLitInfo { type_name, field_names, constructor, kind })
        } else {
            self.record_lit_infos[idx].clone()
        }
    }

    /// Parses SelectInfo on demand (load path parses a single entry from the mmap section).
    pub fn select_info_at(&self, idx: usize) -> Option<SelectInfo> {
        if let Some(ref mem) = self.mem {
            let off = self.select_info_offsets.get(idx).copied()?;
            if off == u32::MAX { return None; }
            let section = mem.section(SectionKind::SelectInfos);
            let mut r = &section[off as usize..];
            let valid = super::Spec::read_u8(&mut r);
            if valid == 0 { return None; }
            let branch_count = super::Spec::read_u32(&mut r) as usize;
            let mut branches = Vec::with_capacity(branch_count);
            for _ in 0..branch_count {
                let subgraph_id = SubGraphId(super::Spec::read_u32(&mut r));
                let event_kind = super::Spec::u8_to_event_kind(super::Spec::read_u8(&mut r));
                let event_source_node = NodeId(super::Spec::read_u32(&mut r));
                branches.push(SelectBranch { subgraph_id, event_kind, event_source_node });
            }
            Some(SelectInfo { branches })
        } else {
            self.select_infos[idx].clone()
        }
    }

    /// Parses TraitConstructInfo on demand (load path parses from the mmap section + string_pool).
    pub fn trait_construct_info_at(&self, idx: usize) -> Option<TraitConstructInfo> {
        if let Some(ref mem) = self.mem {
            let off = self.trait_construct_info_offsets.get(idx).copied()?;
            if off == u32::MAX { return None; }
            let section = mem.section(SectionKind::TraitConstructInfos);
            let mut r = &section[off as usize..];
            let valid = super::Spec::read_u8(&mut r);
            if valid == 0 { return None; }
            let trait_name = { let o = super::Spec::read_u32(&mut r); let l = super::Spec::read_u32(&mut r); mem.read_str(o, l) };
            let mn_count = super::Spec::read_u32(&mut r) as usize;
            let mut method_names = Vec::with_capacity(mn_count);
            for _ in 0..mn_count {
                let o = super::Spec::read_u32(&mut r); let l = super::Spec::read_u32(&mut r);
                method_names.push(mem.read_str(o, l));
            }
            let m_count = super::Spec::read_u32(&mut r) as usize;
            let mut methods = Vec::with_capacity(m_count);
            for _ in 0..m_count {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(super::Spec::read_bytes(&mut r, 8));
                methods.push(super::Spec::bytes_to_trait_method_entry(&buf));
            }
            Some(TraitConstructInfo { trait_name, method_names, methods })
        } else {
            self.trait_construct_infos[idx].clone()
        }
    }

    /// Parses RecordExtendInfo on demand (load path parses from the mmap section + string_pool).
    pub fn record_extend_info_at(&self, idx: usize) -> Option<RecordExtendInfo> {
        if let Some(ref mem) = self.mem {
            let off = self.record_extend_info_offsets.get(idx).copied()?;
            if off == u32::MAX { return None; }
            let section = mem.section(SectionKind::RecordExtendInfos);
            let mut r = &section[off as usize..];
            let valid = super::Spec::read_u8(&mut r);
            if valid == 0 { return None; }
            let un_count = super::Spec::read_u32(&mut r) as usize;
            let mut update_names = Vec::with_capacity(un_count);
            for _ in 0..un_count {
                let o = super::Spec::read_u32(&mut r); let l = super::Spec::read_u32(&mut r);
                update_names.push(mem.read_str(o, l));
            }
            Some(RecordExtendInfo { update_names })
        } else {
            self.record_extend_infos[idx].clone()
        }
    }
}
