//! DataFlowGraph zerocopy accessor layer.
//!
//! When `DataFlowGraph.mem = Some(GraphMemory)` (the `.kzo` loading path), the
//! 24 per-Node scalar tables plus `nodes` and `inputs` are read directly from the
//! mmap'd byte slices via accessor methods, with no copy into owned `Vec`s.
//!
//! When `mem = None` (the build path), accessor methods fall back to owned `Vec`
//! field access.
//!
//! The five variable-length complex tables (`gate_branches` / `record_lit_infos` /
//! `select_infos` / `trait_construct_infos` / `record_extend_infos`) plus
//! `subgraphs` and `downstreams` stay owned on both paths (eager-loaded at load
//! time) and need no `mem` branching.

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

#[inline]
fn rd_i32(r: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([r[off], r[off + 1], r[off + 2], r[off + 3]])
}

#[inline]
fn rd_u8(r: &[u8], off: usize) -> u8 {
    r[off]
}

// ==================== Accessor generation macros ====================
//
// Category A (fixed-width scalar Option), B (boolean bitmap), and C (with strings)
// accessors are highly repetitive; the three macros below generate the method bodies.
// Category D (fixed-width composite) and the five on-demand variable-length tables are
// structurally heterogeneous and are not macro-generated.

/// Category A accessor: zerocopy reads a fixed-width scalar; the sentinel value means `None`.
///
/// `$read` is `rd_u8`/`rd_u16`/`rd_u32` (a type-annotated helper to work around closure
/// parameter inference issues). `$wrap` wraps the decoded integer into the target type
/// (`|v| v` means the value is already the integer).
macro_rules! accessor_opt {
    ($method:ident, $field:ident, $kind:ident, $read:ident, $width:expr, $sentinel:expr, $ret:ty, $wrap:expr) => {
        #[inline]
        pub fn $method(&self, idx: usize) -> Option<$ret> {
            if let Some(ref mem) = self.mem {
                let r = mem.section(SectionKind::$kind);
                let v = $read(r, idx * $width);
                if v == $sentinel {
                    None
                } else {
                    Some($wrap(v))
                }
            } else {
                self.$field[idx]
            }
        }
    };
}

/// Category B accessor: zerocopy reads a boolean bitmap bit.
macro_rules! accessor_bool {
    ($method:ident, $field:ident, $kind:ident) => {
        #[inline]
        pub fn $method(&self, idx: usize) -> bool {
            if let Some(ref mem) = self.mem {
                let r = mem.section(SectionKind::$kind);
                r[idx / 8] & (1 << (idx % 8)) != 0
            } else {
                self.$field[idx]
            }
        }
    };
}

/// Category C accessor: zerocopy reads a `StrRef` -> `&str` (from the StringPool section).
macro_rules! accessor_str {
    ($method:ident, $field:ident, $kind:ident) => {
        #[inline]
        pub fn $method(&self, idx: usize) -> Option<&str> {
            if let Some(ref mem) = self.mem {
                let r = mem.section(SectionKind::$kind);
                let off = idx * 8;
                let str_off = rd_u32(r, off);
                if str_off == u32::MAX {
                    None
                } else {
                    let str_len = rd_u32(r, off + 4);
                    let pool = mem.string_pool();
                    Some(std::str::from_utf8(
                        &pool[str_off as usize..(str_off + str_len) as usize],
                    ).unwrap())
                }
            } else {
                self.$field[idx].as_deref()
            }
        }
    };
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

    // ---- Node ----

    /// Reads a node by index (Copy; 14 bytes read from the mmap slice).
    #[inline]
    pub fn node(&self, idx: usize) -> Node {
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::Nodes);
            let off = idx * 14;
            Node {
                kind: u8_to_node_kind(r[off]),
                input_count: r[off + 1],
                inputs_offset: rd_u32(r, off + 2),
                compute_fn: ComputeFnId(rd_u32(r, off + 6)),
            }
        } else {
            self.nodes[idx]
        }
    }

    // ---- Inputs ----

    /// Reads the input slice for a node (zerocopy: transmuted from the mmap Inputs section into `&[NodeId]`).
    #[inline]
    pub fn inputs(&self, offset: u32, count: u8) -> &[NodeId] {
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

    // ---- Category A: fixed-width scalar tables (zerocopy, sentinel means None) ----

    accessor_opt!(call_target, call_targets, CallTargets, rd_u32, 4, u32::MAX, SubGraphId, |v| SubGraphId(v));
    accessor_opt!(field_access_info, field_access_infos, FieldAccessInfos, rd_u16, 2, u16::MAX, u16, |v| v);
    accessor_opt!(vtable_call_method, vtable_call_methods, VtableCallMethods, rd_u16, 2, u16::MAX, u16, |v| v);
    accessor_opt!(await_event_source, await_event_sources, AwaitEventSources, rd_u32, 4, u32::MAX, NodeId, |v| NodeId(v));
    accessor_opt!(writeback_target, writeback_targets, WritebackTargets, rd_u32, 4, u32::MAX, NodeId, |v| NodeId(v));

    // hoisted_owner: SubGraphId (no None, read directly)
    #[inline]
    pub fn hoisted_owner(&self, idx: usize) -> SubGraphId {
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::HoistedOwners);
            SubGraphId(rd_u32(r, idx * 4))
        } else {
            self.hoisted_owners[idx]
        }
    }

    accessor_opt!(global_load_slot, global_load_slots, GlobalLoadSlots, rd_u32, 4, u32::MAX, u32, |v| v);
    accessor_opt!(global_store_slot, global_store_slots, GlobalStoreSlots, rd_u32, 4, u32::MAX, u32, |v| v);
    accessor_opt!(pattern_field_index, pattern_field_indices, PatternFieldIndices, rd_u16, 2, u16::MAX, u16, |v| v);
    accessor_opt!(closure_call_arg_count, closure_call_arg_counts, ClosureCallArgCounts, rd_u8, 1, u8::MAX, u8, |v| v);

    // ---- Category B: boolean tables (zerocopy, bitmap read) ----

    accessor_bool!(tail_call_flag, tail_call_flags, TailCallFlags);
    accessor_bool!(safe_op_flag, safe_op_flags, SafeOpFlags);
    accessor_bool!(is_hoisted_node, hoisted_node, HoistedNode);
    accessor_bool!(slice_inclusive, slice_inclusive, SliceInclusive);

    // ---- Category C: tables with strings (zerocopy, StrRef -> &str from StringPool) ----

    accessor_str!(ffi_call_name, ffi_call_names, FfiCallNames);
    accessor_str!(field_set_name, field_set_names, FieldSetNames);
    accessor_str!(pattern_ctor_name, pattern_ctor_names, PatternCtorNames);
    accessor_str!(pattern_type_name, pattern_type_names, PatternTypeNames);
    accessor_str!(cast_target_type, cast_target_types, CastTargetTypes);

    // ---- Category D: fixed-width composite tables (zerocopy, validity byte + fields) ----

    #[inline]
    pub fn closure_info(&self, idx: usize) -> Option<ClosureInfo> {
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::ClosureInfos);
            let off = idx * 10; // valid(1) + subgraph_id(4) + arity(1) + self_upvalue_idx(4)
            if r[off] == 0 { None } else {
                Some(ClosureInfo {
                    subgraph_id: SubGraphId(rd_u32(r, off + 1)),
                    arity: r[off + 5],
                    self_upvalue_idx: rd_i32(r, off + 6),
                })
            }
        } else {
            self.closure_infos[idx].clone()
        }
    }

    /// stdlib @extern("C") #{ }# inline FFI call info.
    /// Build path: clone from Vec. Load path: v1 unsupported (returns None) — .kzo serialization
    /// of DynFfiInfo (containing String + Vec) is deferred to a later phase.
    #[inline]
    pub fn dyn_ffi_info(&self, idx: usize) -> Option<DynFfiInfo> {
        if self.mem.is_some() {
            // TODO: implement zerocopy load path for DynFfiInfo (symbol/sig/arg_count).
            None
        } else {
            self.dyn_ffi_infos[idx].clone()
        }
    }

    #[inline]
    pub fn partial_info(&self, idx: usize) -> Option<PartialInfo> {
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::PartialInfos);
            let off = idx * 6; // valid(1) + subgraph_id(4) + bound_count(1)
            if r[off] == 0 { None } else {
                Some(PartialInfo {
                    subgraph_id: SubGraphId(rd_u32(r, off + 1)),
                    bound_count: r[off + 5],
                })
            }
        } else {
            self.partial_infos[idx].clone()
        }
    }

    #[inline]
    pub fn lazy_construct_info(&self, idx: usize) -> Option<LazyConstructInfo> {
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::LazyConstructInfos);
            let off = idx * 5; // valid(1) + thunk_sg(4)
            if r[off] == 0 { None } else {
                Some(LazyConstructInfo {
                    thunk_sg: SubGraphId(rd_u32(r, off + 1)),
                })
            }
        } else {
            self.lazy_construct_infos[idx].clone()
        }
    }

    #[inline]
    pub fn memo_info(&self, idx: usize) -> Option<MemoInfo> {
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::MemoInfos);
            let off = idx * 6; // valid(1) + table_index(4) + param_count(1)
            if r[off] == 0 { None } else {
                Some(MemoInfo {
                    table_index: rd_u32(r, off + 1),
                    param_count: r[off + 5],
                })
            }
        } else {
            self.memo_infos[idx].clone()
        }
    }

    // ---- Fixed-width variable-length tables (zerocopy, tag + payload) ----

    #[inline]
    pub fn const_value(&self, idx: usize) -> Option<ConstValue> {
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::ConstValues);
            let off = idx * 17; // tag(1) + payload(16)
            let tag = r[off];
            if tag == 0 { return None; }
            let p = &r[off + 1..off + 17];
            Some(match tag {
                1 => ConstValue::I8(p[0] as i8),
                2 => ConstValue::I16(i16::from_le_bytes([p[0], p[1]])),
                3 => ConstValue::I32(i32::from_le_bytes([p[0], p[1], p[2], p[3]])),
                4 => ConstValue::I64(i64::from_le_bytes({ let mut b = [0u8; 8]; b.copy_from_slice(&p[0..8]); b })),
                5 => ConstValue::I128(i128::from_le_bytes({ let mut b = [0u8; 16]; b.copy_from_slice(p); b })),
                6 => ConstValue::U8(p[0]),
                7 => ConstValue::U16(u16::from_le_bytes([p[0], p[1]])),
                8 => ConstValue::U32(u32::from_le_bytes([p[0], p[1], p[2], p[3]])),
                9 => ConstValue::U64(u64::from_le_bytes({ let mut b = [0u8; 8]; b.copy_from_slice(&p[0..8]); b })),
                10 => ConstValue::U128(u128::from_le_bytes({ let mut b = [0u8; 16]; b.copy_from_slice(p); b })),
                11 => ConstValue::Isize(i64::from_le_bytes({ let mut b = [0u8; 8]; b.copy_from_slice(&p[0..8]); b }) as isize),
                12 => ConstValue::Usize(u64::from_le_bytes({ let mut b = [0u8; 8]; b.copy_from_slice(&p[0..8]); b }) as usize),
                13 => ConstValue::F32(f32::from_le_bytes([p[0], p[1], p[2], p[3]])),
                14 => ConstValue::F64(f64::from_le_bytes({ let mut b = [0u8; 8]; b.copy_from_slice(&p[0..8]); b })),
                15 => ConstValue::F16(u16::from_le_bytes([p[0], p[1]])),
                16 => ConstValue::F128({ let mut b = [0u8; 16]; b.copy_from_slice(p); b }),
                17 => ConstValue::Bool(p[0] != 0),
                18 => ConstValue::Char(u32::from_le_bytes([p[0], p[1], p[2], p[3]])),
                19 => {
                    let off = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
                    let len = u32::from_le_bytes([p[4], p[5], p[6], p[7]]);
                    ConstValue::Str { offset: off, len }
                }
                20 => ConstValue::Null,
                21 => ConstValue::Void,
                _ => panic!("invalid ConstTag: {}", tag),
            })
        } else {
            self.const_values[idx].clone()
        }
    }

    #[inline]
    pub fn batch_info(&self, idx: usize) -> Option<BatchInfo> {
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::BatchInfos);
            let off = idx * 5; // valid(1) + payload(4)
            let valid = r[off];
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&r[off + 1..off + 5]);
            if valid == 0 { None } else { Some(bytes_to_batch_info(&buf)) }
        } else {
            self.batch_infos[idx].clone()
        }
    }

    // ---- Downstreams (zerocopy CSR access) ----

    /// Returns the downstream-node slice for node `idx`.
    ///
    /// Load path: returns a `&[NodeId]` slice directly from the mmap Downstreams
    /// section with no heap allocation (eliminates the ~700KB memory blowup of
    /// `Vec<Vec<NodeId>>` and hot-path clones).
    ///
    /// Build path: returns a reference to `self.downstreams[idx]`.
    ///
    /// CSR layout: `[u32; N+1]` offsets (element-count index) followed by
    /// `[u32; M]` flat. `offsets[i]..offsets[i+1]` is the element range of node i's
    /// downstreams within the flat region.
    #[inline]
    pub fn downstream_slice(&self, idx: usize) -> &[NodeId] {
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::Downstreams);
            let n = mem.header().node_count as usize;
            // Offsets region: [u32; N+1], followed by the flat region.
            let offsets_start = 0;
            let flat_start = (n + 1) * 4;
            let start_elem = rd_u32(r, offsets_start + idx * 4) as usize;
            let end_elem = rd_u32(r, offsets_start + (idx + 1) * 4) as usize;
            let byte_start = flat_start + start_elem * 4;
            let count = end_elem - start_elem;
            // SAFETY: NodeId is #[repr(transparent)] over u32 (4 bytes, 4-byte aligned).
            // The Downstreams section is 4-byte aligned, flat_start = (N+1)*4 is a multiple of 4,
            // and byte_start = flat_start + start_elem*4 is also a multiple of 4.
            // The slice range is within the section bounds (serialization guarantees offsets[N] = M = flat length).
            unsafe {
                std::slice::from_raw_parts(r.as_ptr().add(byte_start) as *const NodeId, count)
            }
        } else {
            &self.downstreams[idx]
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
    /// SgUpvalueNodes section with no heap allocation (eliminates the per-subgraph
    /// `Vec<NodeId>` allocation, typically saving ~56B/subgraph).
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
    /// Load path: returns a `&[(u32, u32)]` slice directly from the mmap
    /// SgNestedRanges section with no heap allocation (eliminates the per-subgraph
    /// `Vec<(u32, u32)>` allocation).
    ///
    /// Build path: returns a reference to `self.subgraphs[sg_idx].nested_ranges`.
    #[inline]
    pub fn sg_nested_ranges(&self, sg_idx: usize) -> &[(u32, u32)] {
        if let Some(ref mem) = self.mem {
            let (off, len) = self.sg_nr_offsets[sg_idx];
            let r = mem.section(SectionKind::SgNestedRanges);
            let byte_start = off as usize;
            let count = len as usize;
            // SAFETY: (u32, u32) is 8 bytes under repr(Rust) (two consecutive u32s), 4-byte aligned.
            // The SgNestedRanges section is 4-byte aligned, and offset is a multiple of 4.
            // During serialization each element is written as two u32s (8 bytes), matching the (u32, u32) layout.
            unsafe {
                std::slice::from_raw_parts(r.as_ptr().add(byte_start) as *const (u32, u32), count)
            }
        } else {
            &self.subgraphs[sg_idx].nested_ranges
        }
    }

    // ---- The tables below stay owned on both paths; accessors index directly. ----
    //(gate_branches / record_lit_infos / select_infos /
    //  trait_construct_infos / record_extend_infos / subgraphs)
    // These tables need no mem branch; the execution path keeps using graph.field[idx] directly.

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
    pub fn gate_branches_at(&self, idx: usize) -> Option<GateBranches> {
        if let Some(ref mem) = self.mem {
            let off = self.gate_branch_offsets.get(idx).copied()?;
            if off == u32::MAX { return None; }
            let section = mem.section(SectionKind::GateBranches);
            let mut r = &section[off as usize..];
            let valid = super::Spec::read_u8(&mut r);
            if valid == 0 { return None; }
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
            Some(GateBranches { condition_input, branches })
        } else {
            self.gate_branches[idx].clone()
        }
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
