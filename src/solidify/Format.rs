//! Solidify binary format: serialization and deserialization (owned path, no mmap).
//!
//! Format layout:
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │ Header              (64B fixed)             │
//! ├─────────────────────────────────────────────┤
//! │ Section Index       (fixed-length array)    │
//! ├─────────────────────────────────────────────┤
//! │ Nodes / Inputs / SubGraphs / 29 tables / .. │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! Characteristics:
//! - The load path reads a `.kzo` back into owned `Vec`s to construct a `DataFlowGraph`
//!   (without modifying DFG fields).
//! - No mmap / zerocopy; pure `Vec<u8>` + manual LE read/write.
//! - Validates round-trip: serialize -> load -> fields match.

#![allow(non_snake_case)]

use std::io;
use std::sync::Arc;

use crate::ir::Ir::*;

use super::Spec::*;

// ==================== Serialization/deserialization helper macros ====================
//
// Category A (fixed-width scalar Option tables), B (boolean tables), and C (tables with
// strings) are highly repetitive; the four macros below eliminate the boilerplate.
// Category D (fixed-width composite) and E (variable-length fields) are structurally
// heterogeneous and are not macro-generated.

/// Category A serialization: `Vec<Option<T>>` -> fixed-width uN array, `None` = sentinel value.
///
/// The `$inner` closure takes `&T` and returns the raw integer to write (when `T` is
/// `Copy`, simply dereference).
macro_rules! ser_opt_table {
    ($sections:expr, $n:expr, $graph:expr, $field:ident, $kind:ident, $width:expr, $write:ident, $sentinel:expr, $inner:expr) => {{
        let mut buf = Vec::with_capacity($n * $width);
        for v in &$graph.$field {
            let encoded = match v {
                Some(x) => $inner(x),
                None => $sentinel,
            };
            $write(&mut buf, encoded);
        }
        $sections.push((SectionKind::$kind, buf));
    }};
}

/// Category B serialization: `Vec<bool>` -> bitmap (one bit per boolean, 8x compression).
macro_rules! ser_bool_table {
    ($sections:expr, $n:expr, $graph:expr, $field:ident, $kind:ident) => {{
        let mut buf = vec![0u8; ($n + 7) / 8];
        for (i, &v) in $graph.$field.iter().enumerate() {
            if v {
                buf[i / 8] |= 1 << (i % 8);
            }
        }
        $sections.push((SectionKind::$kind, buf));
    }};
}

/// Category C serialization: `Vec<Option<String>>` -> `(offset:u32, len:u32)` array, interned into the string pool.
macro_rules! ser_str_table {
    ($sections:expr, $n:expr, $graph:expr, $field:ident, $kind:ident, $pool:expr) => {{
        let mut buf = Vec::with_capacity($n * 8);
        for v in &$graph.$field {
            let (off, len) = $pool.add_opt(v);
            write_u32(&mut buf, off);
            write_u32(&mut buf, len);
        }
        $sections.push((SectionKind::$kind, buf));
    }};
}

/// Category A deserialization: fixed-width uN array -> `Vec<Option<T>>`, sentinel value = `None`.
///
/// Dispatches on the width literal 1/2/4 for decoding logic (`r` shares the macro's
/// syntactic context to work around hygiene issues). `$wrap` wraps the integer into
/// the target type (`|v| v` means the value is already the integer).
macro_rules! de_opt_table {
    ($mem:expr, $n:expr, $kind:ident, 1, $sentinel:expr, $wrap:expr) => {{
        let r = $mem.section(SectionKind::$kind);
        (0..$n)
            .map(|i| {
                let v = r[i];
                if v == $sentinel { None } else { Some($wrap(v)) }
            })
            .collect::<Vec<_>>()
    }};
    ($mem:expr, $n:expr, $kind:ident, 2, $sentinel:expr, $wrap:expr) => {{
        let r = $mem.section(SectionKind::$kind);
        (0..$n)
            .map(|i| {
                let base = i * 2;
                let v = u16::from_le_bytes([r[base], r[base + 1]]);
                if v == $sentinel { None } else { Some($wrap(v)) }
            })
            .collect::<Vec<_>>()
    }};
    ($mem:expr, $n:expr, $kind:ident, 4, $sentinel:expr, $wrap:expr) => {{
        let r = $mem.section(SectionKind::$kind);
        (0..$n)
            .map(|i| {
                let base = i * 4;
                let v = u32::from_le_bytes([r[base], r[base + 1], r[base + 2], r[base + 3]]);
                if v == $sentinel { None } else { Some($wrap(v)) }
            })
            .collect::<Vec<_>>()
    }};
}

// ==================== Serialization ====================

/// Serializes a `DataFlowGraph` into a `.kzo` byte stream.
pub fn serialize_solidify(graph: &DataFlowGraph) -> Vec<u8> {
    let n = graph.nodes.len();
    let mut string_pool = StringPool::new();

    // ---- Collect bytes for each section ----
    let mut sections: Vec<(SectionKind, Vec<u8>)> = Vec::new();

    // 1. Nodes: each Node is 16B
    {
        let mut buf = Vec::with_capacity(n * 16);
        for node in &graph.nodes {
            write_u8(&mut buf, node_kind_to_u8(node.kind));
            write_u8(&mut buf, node.input_count);
            write_u32(&mut buf, node.inputs_offset);
            write_u32(&mut buf, node.compute_fn.0);
            // Pad to 16B alignment (kind(1)+input_count(1)+pad(2)+inputs_offset(4)+compute_fn(4) = 12, pad 4)
            write_u32(&mut buf, 0); // 4B padding
        }
        sections.push((SectionKind::Nodes, buf));
    }

    // 2. Inputs: contiguous NodeId array
    {
        let mut buf = Vec::with_capacity(graph.inputs_pool.data.len() * 4);
        for nid in &graph.inputs_pool.data {
            write_u32(&mut buf, nid.0);
        }
        sections.push((SectionKind::Inputs, buf));
    }

    // 3. SubGraphs + variable-length regions
    {
        let mut sg_buf = Vec::new();
        let mut upvalue_nodes_buf = Vec::new();
        let mut nested_ranges_buf = Vec::new();
        let mut event_decls_buf = Vec::new();
        let mut defer_entries_buf = Vec::new();
        let mut defer_captured_buf = Vec::new();
        let mut reset_plan_buf = Vec::new();

        for sg in &graph.subgraphs {
            // Main structure fixed-width fields
            write_u32(&mut sg_buf, sg.id.0);
            write_u32(&mut sg_buf, sg.node_range.0.0);
            write_u32(&mut sg_buf, sg.node_range.1.0);
            write_u8(&mut sg_buf, sg.param_count);
            write_u32(&mut sg_buf, sg.entry_node.0);
            write_u32(&mut sg_buf, sg.return_node.0);
            write_u8(&mut sg_buf, sg.has_suspend as u8);
            write_u8(&mut sg_buf, loop_kind_to_u8(sg.loop_kind));
            write_u32(&mut sg_buf, sg.loop_parent_sg.map(|s| s.0).unwrap_or(u32::MAX));
            write_u32(&mut sg_buf, sg.cond_node.map(|s| s.0).unwrap_or(u32::MAX));
            write_u32(&mut sg_buf, sg.function_id);
            write_u32(&mut sg_buf, sg.iter_next_node.map(|s| s.0).unwrap_or(u32::MAX));
            write_u8(&mut sg_buf, sg.upvalue_count);

            // upvalue_outer_nodes
            let uv_off = upvalue_nodes_buf.len() as u32;
            write_u32(&mut sg_buf, uv_off);
            write_u32(&mut sg_buf, sg.upvalue_outer_nodes.len() as u32);
            for nid in &sg.upvalue_outer_nodes {
                write_u32(&mut upvalue_nodes_buf, nid.0);
            }

            // nested_ranges
            let nr_off = nested_ranges_buf.len() as u32;
            write_u32(&mut sg_buf, nr_off);
            write_u32(&mut sg_buf, sg.nested_ranges.len() as u32);
            for &(start, end) in &sg.nested_ranges {
                write_u32(&mut nested_ranges_buf, start);
                write_u32(&mut nested_ranges_buf, end);
            }

            // event_source_decls
            let ed_off = event_decls_buf.len() as u32;
            write_u32(&mut sg_buf, ed_off);
            write_u32(&mut sg_buf, sg.event_source_decls.len() as u32);
            for decl in &sg.event_source_decls {
                write_u32(&mut event_decls_buf, decl.node.0);
                write_u8(&mut event_decls_buf, event_kind_to_u8(decl.kind));
                write_u8(&mut event_decls_buf, 0); // padding
                write_u8(&mut event_decls_buf, 0);
                write_u8(&mut event_decls_buf, 0);
            }

            // defer_table
            let df_off = defer_entries_buf.len() as u32;
            write_u32(&mut sg_buf, df_off);
            write_u32(&mut sg_buf, sg.defer_table.len() as u32);
            for de in &sg.defer_table {
                let ci_off = defer_captured_buf.len() as u32;
                write_u32(&mut defer_entries_buf, de.trigger_node.0);
                write_u32(&mut defer_entries_buf, de.body_subgraph.0);
                write_u32(&mut defer_entries_buf, ci_off);
                write_u32(&mut defer_entries_buf, de.captured_inputs.len() as u32);
                // `registered` is runtime state and is not serialized.
                for nid in &de.captured_inputs {
                    write_u32(&mut defer_captured_buf, nid.0);
                }
            }

            // reset_plan
            write_u8(&mut sg_buf, sg.reset_plan.is_some() as u8);
            if let Some(rp) = &sg.reset_plan {
                let rp_off = reset_plan_buf.len() as u32;
                write_u32(&mut sg_buf, rp_off);
                // ResetPlan: 3 Vec<NodeId>
                write_u32(&mut reset_plan_buf, rp.reset_to_zero.len() as u32);
                for nid in &rp.reset_to_zero { write_u32(&mut reset_plan_buf, nid.0); }
                write_u32(&mut reset_plan_buf, rp.reset_to_one.len() as u32);
                for nid in &rp.reset_to_one { write_u32(&mut reset_plan_buf, nid.0); }
                write_u32(&mut reset_plan_buf, rp.reset_condition_tree.len() as u32);
                for nid in &rp.reset_condition_tree { write_u32(&mut reset_plan_buf, nid.0); }
            } else {
                write_u32(&mut sg_buf, 0); // placeholder
            }
        }

        sections.push((SectionKind::SubGraphs, sg_buf));
        sections.push((SectionKind::SgUpvalueNodes, upvalue_nodes_buf));
        sections.push((SectionKind::SgNestedRanges, nested_ranges_buf));
        sections.push((SectionKind::SgEventDecls, event_decls_buf));
        sections.push((SectionKind::SgDeferEntries, defer_entries_buf));
        sections.push((SectionKind::SgDeferCapturedInputs, defer_captured_buf));
        sections.push((SectionKind::SgResetPlan, reset_plan_buf));
    }

    // ---- per-Node fixed-width scalar tables (category A) ----
    ser_opt_table!(sections, n, graph, call_targets, CallTargets, 4, write_u32, u32::MAX, |s: &SubGraphId| s.0);
    ser_opt_table!(sections, n, graph, field_access_infos, FieldAccessInfos, 2, write_u16, u16::MAX, |x: &u16| *x);
    ser_opt_table!(sections, n, graph, vtable_call_methods, VtableCallMethods, 2, write_u16, u16::MAX, |x: &u16| *x);
    ser_opt_table!(sections, n, graph, await_event_sources, AwaitEventSources, 4, write_u32, u32::MAX, |s: &NodeId| s.0);
    ser_opt_table!(sections, n, graph, writeback_targets, WritebackTargets, 4, write_u32, u32::MAX, |s: &NodeId| s.0);
    // hoisted_owners: SubGraphId (no None, written directly)
    {
        let mut buf = Vec::with_capacity(n * 4);
        for v in &graph.hoisted_owners {
            write_u32(&mut buf, v.0);
        }
        sections.push((SectionKind::HoistedOwners, buf));
    }
    ser_opt_table!(sections, n, graph, global_load_slots, GlobalLoadSlots, 4, write_u32, u32::MAX, |x: &u32| *x);
    ser_opt_table!(sections, n, graph, global_store_slots, GlobalStoreSlots, 4, write_u32, u32::MAX, |x: &u32| *x);
    ser_opt_table!(sections, n, graph, pattern_field_indices, PatternFieldIndices, 2, write_u16, u16::MAX, |x: &u16| *x);
    ser_opt_table!(sections, n, graph, closure_call_arg_counts, ClosureCallArgCounts, 1, write_u8, u8::MAX, |x: &u8| *x);

    // ---- per-Node boolean tables (category B) ----
    ser_bool_table!(sections, n, graph, tail_call_flags, TailCallFlags);
    ser_bool_table!(sections, n, graph, safe_op_flags, SafeOpFlags);
    ser_bool_table!(sections, n, graph, hoisted_node, HoistedNode);
    ser_bool_table!(sections, n, graph, slice_inclusive, SliceInclusive);

    // ---- per-Node tables with strings (category C) ----
    // Each node writes (offset:u32, len:u32); None = (u32::MAX, 0), interned into the string pool.
    ser_str_table!(sections, n, graph, ffi_call_names, FfiCallNames, string_pool);
    ser_str_table!(sections, n, graph, field_set_names, FieldSetNames, string_pool);
    ser_str_table!(sections, n, graph, pattern_ctor_names, PatternCtorNames, string_pool);
    ser_str_table!(sections, n, graph, pattern_type_names, PatternTypeNames, string_pool);
    ser_str_table!(sections, n, graph, cast_target_types, CastTargetTypes, string_pool);

    // ---- per-Node fixed-width composite tables (category D) ----
    // ClosureInfo: each node writes a validity u8 + fixed-width data.
    {
        let mut buf = Vec::with_capacity(n * 13);
        for v in &graph.closure_infos {
            match v {
                None => { write_u8(&mut buf, 0); write_u32(&mut buf, 0); write_u8(&mut buf, 0); write_i32(&mut buf, 0); }
                Some(ci) => { write_u8(&mut buf, 1); write_u32(&mut buf, ci.subgraph_id.0); write_u8(&mut buf, ci.arity); write_i32(&mut buf, ci.self_upvalue_idx); }
            }
        }
        sections.push((SectionKind::ClosureInfos, buf));
    }
    // PartialInfo
    {
        let mut buf = Vec::with_capacity(n * 6);
        for v in &graph.partial_infos {
            match v {
                None => { write_u8(&mut buf, 0); write_u32(&mut buf, 0); write_u8(&mut buf, 0); }
                Some(pi) => { write_u8(&mut buf, 1); write_u32(&mut buf, pi.subgraph_id.0); write_u8(&mut buf, pi.bound_count); }
            }
        }
        sections.push((SectionKind::PartialInfos, buf));
    }
    // LazyConstructInfo
    {
        let mut buf = Vec::with_capacity(n * 5);
        for v in &graph.lazy_construct_infos {
            match v {
                None => { write_u8(&mut buf, 0); write_u32(&mut buf, 0); }
                Some(li) => { write_u8(&mut buf, 1); write_u32(&mut buf, li.thunk_sg.0); }
            }
        }
        sections.push((SectionKind::LazyConstructInfos, buf));
    }
    // MemoInfo
    {
        let mut buf = Vec::with_capacity(n * 6);
        for v in &graph.memo_infos {
            match v {
                None => { write_u8(&mut buf, 0); write_u32(&mut buf, 0); write_u8(&mut buf, 0); }
                Some(mi) => { write_u8(&mut buf, 1); write_u32(&mut buf, mi.table_index); write_u8(&mut buf, mi.param_count); }
            }
        }
        sections.push((SectionKind::MemoInfos, buf));
    }

    // ---- per-Node variable-length field tables (category E) ----
    // ConstValues: tag u8 + 16B payload per node (None -> tag=0).
    {
        let mut buf = Vec::with_capacity(n * 17);
        for v in &graph.const_values {
            match v {
                None => { write_u8(&mut buf, 0); write_bytes(&mut buf, &[0u8; 16]); }
                Some(cv) => {
                    let tag = const_tag_to_u8(cv);
                    write_u8(&mut buf, tag);
                    // 16B payload
                    let mut payload = [0u8; 16];
                    match cv {
                        ConstValue::I8(x) => payload[0] = *x as u8,
                        ConstValue::I16(x) => payload[0..2].copy_from_slice(&x.to_le_bytes()),
                        ConstValue::I32(x) => payload[0..4].copy_from_slice(&x.to_le_bytes()),
                        ConstValue::I64(x) => payload[0..8].copy_from_slice(&x.to_le_bytes()),
                        ConstValue::I128(x) => payload.copy_from_slice(&x.to_le_bytes()),
                        ConstValue::U8(x) => payload[0] = *x,
                        ConstValue::U16(x) => payload[0..2].copy_from_slice(&x.to_le_bytes()),
                        ConstValue::U32(x) => payload[0..4].copy_from_slice(&x.to_le_bytes()),
                        ConstValue::U64(x) => payload[0..8].copy_from_slice(&x.to_le_bytes()),
                        ConstValue::U128(x) => payload.copy_from_slice(&x.to_le_bytes()),
                        ConstValue::Isize(x) => payload[0..8].copy_from_slice(&(*x as i64).to_le_bytes()),
                        ConstValue::Usize(x) => payload[0..8].copy_from_slice(&(*x as u64).to_le_bytes()),
                        ConstValue::F32(x) => payload[0..4].copy_from_slice(&x.to_le_bytes()),
                        ConstValue::F64(x) => payload[0..8].copy_from_slice(&x.to_le_bytes()),
                        ConstValue::F16(x) => payload[0..2].copy_from_slice(&x.to_le_bytes()),
                        ConstValue::F128(x) => payload.copy_from_slice(x),
                        ConstValue::Bool(b) => payload[0] = *b as u8,
                        ConstValue::Char(c) => payload[0..4].copy_from_slice(&c.to_le_bytes()),
                        ConstValue::Str { offset, len } => {
                            // ConstValue holds an (offset, len) reference into graph.string_pool.
                            // On serialization, read the actual string and re-intern it into the
                            // serializer-side string_pool (the offset may differ).
                            let off = *offset as usize;
                            let end = off + *len as usize;
                            let s = std::str::from_utf8(&graph.string_pool[off..end]).unwrap_or("");
                            let (new_off, new_len) = string_pool.add(s);
                            payload[0..4].copy_from_slice(&new_off.to_le_bytes());
                            payload[4..8].copy_from_slice(&new_len.to_le_bytes());
                        }
                        ConstValue::Null | ConstValue::Void => {}
                    }
                    write_bytes(&mut buf, &payload);
                }
            }
        }
        sections.push((SectionKind::ConstValues, buf));
    }

    // GateBranches: per node validity u8 + condition_input u32 + branches_count u32 + branches data.
    // branches data: (bool u8, SubGraphId u32, inputs_count u32, inputs [u32]).
    {
        let mut buf = Vec::new();
        for v in &graph.gate_branches {
            match v {
                None => { write_u8(&mut buf, 0); write_u32(&mut buf, 0); write_u32(&mut buf, 0); }
                Some(gb) => {
                    write_u8(&mut buf, 1);
                    write_u32(&mut buf, gb.condition_input.0);
                    write_u32(&mut buf, gb.branches.len() as u32);
                    for (cond, sg, inputs) in &gb.branches {
                        write_u8(&mut buf, *cond as u8);
                        write_u32(&mut buf, sg.0);
                        write_u32(&mut buf, inputs.len() as u32);
                        for nid in inputs { write_u32(&mut buf, nid.0); }
                    }
                }
            }
        }
        sections.push((SectionKind::GateBranches, buf));
    }

    // RecordLitInfos: validity u8 + type_name StrRef + field_names_count u32 + field_names [StrRef] + constructor StrRef + kind u8
    {
        let mut buf = Vec::new();
        for v in &graph.record_lit_infos {
            match v {
                None => { write_u8(&mut buf, 0); write_u32(&mut buf, u32::MAX); write_u32(&mut buf, 0); write_u32(&mut buf, 0); write_u32(&mut buf, u32::MAX); write_u32(&mut buf, 0); write_u8(&mut buf, 0); }
                Some(ri) => {
                    write_u8(&mut buf, 1);
                    let (off, len) = string_pool.add(&ri.type_name);
                    write_u32(&mut buf, off); write_u32(&mut buf, len);
                    write_u32(&mut buf, ri.field_names.len() as u32);
                    for fn_opt in &ri.field_names {
                        let (fo, fl) = string_pool.add_opt(fn_opt);
                        write_u32(&mut buf, fo); write_u32(&mut buf, fl);
                    }
                    let (co, cl) = string_pool.add(&ri.constructor);
                    write_u32(&mut buf, co); write_u32(&mut buf, cl);
                    write_u8(&mut buf, record_lit_kind_to_u8(ri.kind));
                }
            }
        }
        sections.push((SectionKind::RecordLitInfos, buf));
    }

    // SelectInfos: validity u8 + branches_count u32 + branches [(SubGraphId u32, EventKind u8, EventSourceNode u32)]
    {
        let mut buf = Vec::new();
        for v in &graph.select_infos {
            match v {
                None => { write_u8(&mut buf, 0); write_u32(&mut buf, 0); }
                Some(si) => {
                    write_u8(&mut buf, 1);
                    write_u32(&mut buf, si.branches.len() as u32);
                    for b in &si.branches {
                        write_u32(&mut buf, b.subgraph_id.0);
                        write_u8(&mut buf, event_kind_to_u8(b.event_kind));
                        write_u32(&mut buf, b.event_source_node.0);
                    }
                }
            }
        }
        sections.push((SectionKind::SelectInfos, buf));
    }

    // TraitConstructInfos: validity u8 + trait_name StrRef + method_names_count u32 + method_names [StrRef] + methods_count u32 + methods [TraitMethodEntry 8B]
    {
        let mut buf = Vec::new();
        for v in &graph.trait_construct_infos {
            match v {
                None => { write_u8(&mut buf, 0); write_u32(&mut buf, u32::MAX); write_u32(&mut buf, 0); write_u32(&mut buf, 0); write_u32(&mut buf, 0); }
                Some(ti) => {
                    write_u8(&mut buf, 1);
                    let (off, len) = string_pool.add(&ti.trait_name);
                    write_u32(&mut buf, off); write_u32(&mut buf, len);
                    write_u32(&mut buf, ti.method_names.len() as u32);
                    for mn in &ti.method_names {
                        let (mo, ml) = string_pool.add(mn);
                        write_u32(&mut buf, mo); write_u32(&mut buf, ml);
                    }
                    write_u32(&mut buf, ti.methods.len() as u32);
                    for m in &ti.methods {
                        write_bytes(&mut buf, &trait_method_entry_to_bytes(m));
                    }
                }
            }
        }
        sections.push((SectionKind::TraitConstructInfos, buf));
    }

    // RecordExtendInfos: validity u8 + update_names_count u32 + update_names [StrRef]
    {
        let mut buf = Vec::new();
        for v in &graph.record_extend_infos {
            match v {
                None => { write_u8(&mut buf, 0); write_u32(&mut buf, 0); }
                Some(ri) => {
                    write_u8(&mut buf, 1);
                    write_u32(&mut buf, ri.update_names.len() as u32);
                    for un in &ri.update_names {
                        let (uo, ul) = string_pool.add(un);
                        write_u32(&mut buf, uo); write_u32(&mut buf, ul);
                    }
                }
            }
        }
        sections.push((SectionKind::RecordExtendInfos, buf));
    }

    // BatchInfos: validity u8 + 4B BatchInfo
    {
        let mut buf = Vec::with_capacity(n * 5);
        for v in &graph.batch_infos {
            match v {
                None => { write_u8(&mut buf, 0); write_bytes(&mut buf, &[0u8; 4]); }
                Some(bi) => { write_u8(&mut buf, 1); write_bytes(&mut buf, &batch_info_to_bytes(bi)); }
            }
        }
        sections.push((SectionKind::BatchInfos, buf));
    }

    // ---- String Pool ----
    {
        sections.push((SectionKind::StringPool, string_pool.data.clone()));
    }

    // ---- Downstreams (CSR) ----
    {
        let mut buf = Vec::new();
        // offsets: [u32; N+1]
        write_u32(&mut buf, 0);
        let mut cur = 0u32;
        for ds in &graph.downstreams {
            cur += ds.len() as u32;
            write_u32(&mut buf, cur);
        }
        // flat: [u32; M]
        for ds in &graph.downstreams {
            for nid in ds {
                write_u32(&mut buf, nid.0);
            }
        }
        sections.push((SectionKind::Downstreams, buf));
    }

    // ---- Compute runtime field counts ----
    let global_var_count = graph.global_var_storage.len() as u32;
    let memo_table_count = graph.memo_tables.len() as u32;

    // ---- Assemble the final byte stream ----
    let section_count = sections.len() as u16;
    let header_size = SolidifyHeader::SIZE as u32;
    let section_index_size = section_count as u32 * 9; // kind:u8 + offset:u32 + len:u32 = 9B

    // Compute offsets (each section is aligned to 4 bytes to avoid unaligned access).
    // Alignment value 4 rationale: all section data is u32/u16/u8 arrays; u32 is the widest natural alignment.
    // Alignment padding bytes are placed after the previous section and before the current section,
    // without changing the section's len.
    const SECTION_ALIGN: u32 = 4;
    let mut current_offset = header_size + section_index_size;
    // Record the number of padding bytes to insert before each section.
    let mut paddings: Vec<u8> = Vec::with_capacity(sections.len());
    let mut section_entries: Vec<SectionEntry> = Vec::new();
    for (kind, data) in &sections {
        let pad = (SECTION_ALIGN - (current_offset % SECTION_ALIGN)) % SECTION_ALIGN;
        current_offset += pad;
        paddings.push(pad as u8);
        section_entries.push(SectionEntry {
            kind: *kind as u8,
            offset: current_offset,
            len: data.len() as u32,
        });
        current_offset += data.len() as u32;
    }

    // Write the header.
    let mut header = SolidifyHeader {
        magic: SOLIDIFY_MAGIC,
        schema_version: SOLIDIFY_SCHEMA_VERSION,
        flags: 0,
        endianness: 1,
        pointer_width: 8,
        abi_version: SOLIDIFY_ABI_VERSION,
        node_count: n as u32,
        subgraph_count: graph.subgraphs.len() as u32,
        entry_subgraph: graph.entry_subgraph.map(|s| s.0).unwrap_or(u32::MAX),
        input_count: graph.inputs_pool.data.len() as u32,
        string_pool_len: string_pool.len(),
        global_var_count,
        memo_table_count,
        compute_fn_count: COMPUTE_FN_COUNT,
        crc32: 0, // Placeholder, back-filled later.
        section_count,
        _reserved: [0, 0],
        _padding: [0u8; 12],
    };

    // Write the body (everything after the header) for CRC computation.
    let mut body = Vec::new();
    // Section index
    for entry in &section_entries {
        body.push(entry.kind);
        body.extend_from_slice(&entry.offset.to_le_bytes());
        body.extend_from_slice(&entry.len.to_le_bytes());
    }
    // Section data (insert alignment padding before each section)
    for (i, (_, data)) in sections.iter().enumerate() {
        let pad = paddings[i] as usize;
        body.extend_from_slice(&vec![0u8; pad]);
        body.extend_from_slice(data);
    }

    // CRC32 (body)
    header.crc32 = crc32(&body);

    // Final output
    let mut output = Vec::with_capacity(header_size as usize + body.len());
    header.write_to(&mut output).unwrap();
    output.extend_from_slice(&body);

    output
}

// ==================== Deserialization ====================

/// Loads a `DataFlowGraph` from a `.kzo` byte stream (owned path, used for tests).
pub fn load_solidify_from_bytes(data: &[u8]) -> io::Result<DataFlowGraph> {
    let mem = GraphMemory::from_bytes(data)?;
    load_from_graph_memory(&mem)
}

/// Loads a `DataFlowGraph` from a `.kzo` file via mmap (zero-copy file reading).
///
/// The file is mapped directly into the address space, avoiding the kernel-to-userspace
/// copy of `fs::read`. `GraphMemory` owns the `Mmap` and automatically unmaps it when the
/// `DataFlowGraph` is dropped.
pub fn load_solidify_from_file(path: &str) -> io::Result<DataFlowGraph> {
    let mem = GraphMemory::from_file(path)?;
    load_zerocopy(mem)
}

/// Rebuilds an owned `DataFlowGraph` from a `GraphMemory` (shared parsing logic).
fn load_from_graph_memory(mem: &GraphMemory) -> io::Result<DataFlowGraph> {
    let n = mem.header().node_count as usize;

    // 1. Nodes
    let nodes = {
        let nr = mem.section(SectionKind::Nodes);
        let mut nodes = Vec::with_capacity(n);
        let mut r = nr;
        for _ in 0..n {
            let kind = u8_to_node_kind(read_u8(&mut r));
            let input_count = read_u8(&mut r);
            let inputs_offset = read_u32(&mut r);
            let compute_fn = ComputeFnId(read_u32(&mut r));
            let _pad = read_u32(&mut r); // padding
            nodes.push(Node { kind, input_count, inputs_offset, compute_fn });
        }
        nodes
    };

    // 2. Inputs
    let inputs_pool = {
        let ir = mem.section(SectionKind::Inputs);
        let mut data_vec = Vec::with_capacity(mem.header().input_count as usize);
        let mut r = ir;
        for _ in 0..mem.header().input_count {
            data_vec.push(NodeId(read_u32(&mut r)));
        }
        InputsPool { data: data_vec }
    };

    // 3. SubGraphs
    let subgraphs = {
        let sr = mem.section(SectionKind::SubGraphs);
        let upv = mem.section(SectionKind::SgUpvalueNodes);
        let nr2 = mem.section(SectionKind::SgNestedRanges);
        let ed = mem.section(SectionKind::SgEventDecls);
        let df = mem.section(SectionKind::SgDeferEntries);
        let dc = mem.section(SectionKind::SgDeferCapturedInputs);
        let rp = mem.section(SectionKind::SgResetPlan);

        let mut subgraphs = Vec::with_capacity(mem.header().subgraph_count as usize);
        let mut sr_r = sr;
        for _ in 0..mem.header().subgraph_count {
            let id = SubGraphId(read_u32(&mut sr_r));
            let node_range = (NodeId(read_u32(&mut sr_r)), NodeId(read_u32(&mut sr_r)));
            let param_count = read_u8(&mut sr_r);
            let entry_node = NodeId(read_u32(&mut sr_r));
            let return_node = NodeId(read_u32(&mut sr_r));
            let has_suspend = read_u8(&mut sr_r) != 0;
            let loop_kind = u8_to_loop_kind(read_u8(&mut sr_r));
            let loop_parent_sg = { let v = read_u32(&mut sr_r); if v == u32::MAX { None } else { Some(SubGraphId(v)) } };
            let cond_node = { let v = read_u32(&mut sr_r); if v == u32::MAX { None } else { Some(NodeId(v)) } };
            let function_id = read_u32(&mut sr_r);
            let iter_next_node = { let v = read_u32(&mut sr_r); if v == u32::MAX { None } else { Some(NodeId(v)) } };
            let upvalue_count = read_u8(&mut sr_r);

            // upvalue_outer_nodes
            let uv_off = read_u32(&mut sr_r) as usize;
            let uv_len = read_u32(&mut sr_r) as usize;
            let upvalue_outer_nodes: Vec<NodeId> = (0..uv_len)
                .map(|i| {
                    let base = uv_off + i * 4;
                    NodeId(u32::from_le_bytes([upv[base], upv[base+1], upv[base+2], upv[base+3]]))
                })
                .collect();

            // nested_ranges
            let nr_off = read_u32(&mut sr_r) as usize;
            let nr_len = read_u32(&mut sr_r) as usize;
            let nested_ranges: Vec<(u32, u32)> = (0..nr_len)
                .map(|i| {
                    let base = nr_off + i * 8;
                    (u32::from_le_bytes([nr2[base], nr2[base+1], nr2[base+2], nr2[base+3]]),
                     u32::from_le_bytes([nr2[base+4], nr2[base+5], nr2[base+6], nr2[base+7]]))
                })
                .collect();

            // event_source_decls
            let ed_off = read_u32(&mut sr_r) as usize;
            let ed_len = read_u32(&mut sr_r) as usize;
            let event_source_decls: Vec<EventSourceDecl> = (0..ed_len)
                .map(|i| {
                    let base = ed_off + i * 8;
                    EventSourceDecl {
                        node: NodeId(u32::from_le_bytes([ed[base], ed[base+1], ed[base+2], ed[base+3]])),
                        kind: u8_to_event_kind(ed[base+4]),
                    }
                })
                .collect();

            // defer_table
            let df_off = read_u32(&mut sr_r) as usize;
            let df_len = read_u32(&mut sr_r) as usize;
            let defer_table: Vec<DeferEntry> = (0..df_len)
                .map(|i| {
                    let base = df_off + i * 16; // 4*4 bytes per entry
                    let trigger_node = NodeId(u32::from_le_bytes([df[base], df[base+1], df[base+2], df[base+3]]));
                    let body_subgraph = SubGraphId(u32::from_le_bytes([df[base+4], df[base+5], df[base+6], df[base+7]]));
                    let ci_off = u32::from_le_bytes([df[base+8], df[base+9], df[base+10], df[base+11]]) as usize;
                    let ci_len = u32::from_le_bytes([df[base+12], df[base+13], df[base+14], df[base+15]]) as usize;
                    let captured_inputs: Vec<NodeId> = (0..ci_len)
                        .map(|j| NodeId(u32::from_le_bytes([dc[ci_off + j*4], dc[ci_off + j*4 + 1], dc[ci_off + j*4 + 2], dc[ci_off + j*4 + 3]])))
                        .collect();
                    DeferEntry { trigger_node, body_subgraph, captured_inputs, registered: false }
                })
                .collect();

            // reset_plan
            let has_rp = read_u8(&mut sr_r) != 0;
            let rp_off = read_u32(&mut sr_r) as usize;
            let reset_plan = if has_rp {
                let mut rp_r = &rp[rp_off..];
                let rz_len = read_u32(&mut rp_r) as usize;
                let reset_to_zero: Vec<NodeId> = (0..rz_len).map(|_| NodeId(read_u32(&mut rp_r))).collect();
                let ro_len = read_u32(&mut rp_r) as usize;
                let reset_to_one: Vec<NodeId> = (0..ro_len).map(|_| NodeId(read_u32(&mut rp_r))).collect();
                let rc_len = read_u32(&mut rp_r) as usize;
                let reset_condition_tree: Vec<NodeId> = (0..rc_len).map(|_| NodeId(read_u32(&mut rp_r))).collect();
                Some(ResetPlan { reset_to_zero, reset_to_one, reset_condition_tree })
            } else {
                None
            };

            subgraphs.push(SubGraph {
                id, node_range, param_count, entry_node, return_node, has_suspend,
                event_source_decls, defer_table, loop_kind, loop_parent_sg, cond_node,
                function_id, iter_next_node, upvalue_count, upvalue_outer_nodes,
                nested_ranges, reset_plan,
            });
        }
        subgraphs
    };

    // ---- per-Node fixed-width scalar tables (category A) ----
    let call_targets: Vec<Option<SubGraphId>> = de_opt_table!(mem, n, CallTargets, 4, u32::MAX, |v| SubGraphId(v));
    let field_access_infos: Vec<Option<u16>> = de_opt_table!(mem, n, FieldAccessInfos, 2, u16::MAX, |v| v);
    let vtable_call_methods: Vec<Option<u16>> = de_opt_table!(mem, n, VtableCallMethods, 2, u16::MAX, |v| v);
    let await_event_sources: Vec<Option<NodeId>> = de_opt_table!(mem, n, AwaitEventSources, 4, u32::MAX, |v| NodeId(v));
    let writeback_targets: Vec<Option<NodeId>> = de_opt_table!(mem, n, WritebackTargets, 4, u32::MAX, |v| NodeId(v));
    let hoisted_owners: Vec<SubGraphId> = {
        let r = mem.section(SectionKind::HoistedOwners);
        (0..n).map(|i| SubGraphId(u32::from_le_bytes([r[i*4], r[i*4+1], r[i*4+2], r[i*4+3]]))).collect()
    };
    let global_load_slots: Vec<Option<u32>> = de_opt_table!(mem, n, GlobalLoadSlots, 4, u32::MAX, |v| v);
    let global_store_slots: Vec<Option<u32>> = de_opt_table!(mem, n, GlobalStoreSlots, 4, u32::MAX, |v| v);
    let pattern_field_indices: Vec<Option<u16>> = de_opt_table!(mem, n, PatternFieldIndices, 2, u16::MAX, |v| v);
    let closure_call_arg_counts: Vec<Option<u8>> = de_opt_table!(mem, n, ClosureCallArgCounts, 1, u8::MAX, |v| v);

    // ---- per-Node boolean tables (category B) ----
    let read_bool_vec = |kind: SectionKind| -> Vec<bool> {
        let r = mem.section(kind);
        (0..n).map(|i| r[i / 8] & (1 << (i % 8)) != 0).collect()
    };
    let tail_call_flags = read_bool_vec(SectionKind::TailCallFlags);
    let safe_op_flags = read_bool_vec(SectionKind::SafeOpFlags);
    let hoisted_node = read_bool_vec(SectionKind::HoistedNode);
    let slice_inclusive = read_bool_vec(SectionKind::SliceInclusive);

    // ---- per-Node tables with strings (category C) ----
    let read_str_vec = |kind: SectionKind| -> Vec<Option<String>> {
        let r = mem.section(kind);
        (0..n).map(|i| {
            let off = u32::from_le_bytes([r[i*8], r[i*8+1], r[i*8+2], r[i*8+3]]);
            let len = u32::from_le_bytes([r[i*8+4], r[i*8+5], r[i*8+6], r[i*8+7]]);
            if off == u32::MAX { None } else { Some(mem.read_str(off, len)) }
        }).collect()
    };
    let ffi_call_names = read_str_vec(SectionKind::FfiCallNames);
    let field_set_names = read_str_vec(SectionKind::FieldSetNames);
    let pattern_ctor_names = read_str_vec(SectionKind::PatternCtorNames);
    let pattern_type_names = read_str_vec(SectionKind::PatternTypeNames);
    let cast_target_types = read_str_vec(SectionKind::CastTargetTypes);

    // ---- per-Node fixed-width composite tables (category D) ----
    let closure_infos: Vec<Option<ClosureInfo>> = {
        let r = mem.section(SectionKind::ClosureInfos);
        let mut r = r;
        (0..n).map(|_| {
            let valid = read_u8(&mut r);
            if valid == 0 {
                let _ = read_u32(&mut r); let _ = read_u8(&mut r); let _ = read_i32(&mut r);
                None
            } else {
                let subgraph_id = SubGraphId(read_u32(&mut r));
                let arity = read_u8(&mut r);
                let self_upvalue_idx = read_i32(&mut r);
                Some(ClosureInfo { subgraph_id, arity, self_upvalue_idx })
            }
        }).collect()
    };
    let partial_infos: Vec<Option<PartialInfo>> = {
        let r = mem.section(SectionKind::PartialInfos);
        let mut r = r;
        (0..n).map(|_| {
            let valid = read_u8(&mut r);
            if valid == 0 {
                let _ = read_u32(&mut r); let _ = read_u8(&mut r);
                None
            } else {
                let subgraph_id = SubGraphId(read_u32(&mut r));
                let bound_count = read_u8(&mut r);
                Some(PartialInfo { subgraph_id, bound_count })
            }
        }).collect()
    };
    let lazy_construct_infos: Vec<Option<LazyConstructInfo>> = {
        let r = mem.section(SectionKind::LazyConstructInfos);
        let mut r = r;
        (0..n).map(|_| {
            let valid = read_u8(&mut r);
            if valid == 0 {
                let _ = read_u32(&mut r);
                None
            } else {
                let thunk_sg = SubGraphId(read_u32(&mut r));
                Some(LazyConstructInfo { thunk_sg })
            }
        }).collect()
    };
    let memo_infos: Vec<Option<MemoInfo>> = {
        let r = mem.section(SectionKind::MemoInfos);
        let mut r = r;
        (0..n).map(|_| {
            let valid = read_u8(&mut r);
            if valid == 0 {
                let _ = read_u32(&mut r); let _ = read_u8(&mut r);
                None
            } else {
                let table_index = read_u32(&mut r);
                let param_count = read_u8(&mut r);
                Some(MemoInfo { table_index, param_count })
            }
        }).collect()
    };

    // ---- per-Node variable-length field tables (category E) ----
    let const_values: Vec<Option<ConstValue>> = {
        let r = mem.section(SectionKind::ConstValues);
        let mut r = r;
        (0..n).map(|_| {
            let tag = read_u8(&mut r);
            let payload = read_bytes(&mut r, 16);
            if tag == 0 {
                None
            } else {
                Some(match tag {
                    1 => ConstValue::I8(payload[0] as i8),
                    2 => ConstValue::I16(i16::from_le_bytes([payload[0], payload[1]])),
                    3 => ConstValue::I32(i32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]])),
                    4 => ConstValue::I64(i64::from_le_bytes({ let mut b = [0u8; 8]; b.copy_from_slice(&payload[0..8]); b })),
                    5 => ConstValue::I128(i128::from_le_bytes({ let mut b = [0u8; 16]; b.copy_from_slice(payload); b })),
                    6 => ConstValue::U8(payload[0]),
                    7 => ConstValue::U16(u16::from_le_bytes([payload[0], payload[1]])),
                    8 => ConstValue::U32(u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]])),
                    9 => ConstValue::U64(u64::from_le_bytes({ let mut b = [0u8; 8]; b.copy_from_slice(&payload[0..8]); b })),
                    10 => ConstValue::U128(u128::from_le_bytes({ let mut b = [0u8; 16]; b.copy_from_slice(payload); b })),
                    11 => ConstValue::Isize(i64::from_le_bytes({ let mut b = [0u8; 8]; b.copy_from_slice(&payload[0..8]); b }) as isize),
                    12 => ConstValue::Usize(u64::from_le_bytes({ let mut b = [0u8; 8]; b.copy_from_slice(&payload[0..8]); b }) as usize),
                    13 => ConstValue::F32(f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]])),
                    14 => ConstValue::F64(f64::from_le_bytes({ let mut b = [0u8; 8]; b.copy_from_slice(&payload[0..8]); b })),
                    15 => ConstValue::F16(u16::from_le_bytes([payload[0], payload[1]])),
                    16 => ConstValue::F128({ let mut b = [0u8; 16]; b.copy_from_slice(payload); b }),
                    17 => ConstValue::Bool(payload[0] != 0),
                    18 => ConstValue::Char(u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]])),
                    19 => {
                        // (offset, len) references the string_pool directly; no leak needed.
                        let off = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
                        let len = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
                        ConstValue::Str { offset: off, len }
                    }
                    20 => ConstValue::Null,
                    21 => ConstValue::Void,
                    _ => panic!("invalid ConstTag: {}", tag),
                })
            }
        }).collect()
    };

    // GateBranches
    let gate_branches: Vec<Option<GateBranches>> = {
        let r = mem.section(SectionKind::GateBranches);
        let mut r = r;
        (0..n).map(|_| {
            let valid = read_u8(&mut r);
            if valid == 0 {
                let _ = read_u32(&mut r); let _ = read_u32(&mut r);
                None
            } else {
                let condition_input = NodeId(read_u32(&mut r));
                let branch_count = read_u32(&mut r) as usize;
                let mut branches = Vec::with_capacity(branch_count);
                for _ in 0..branch_count {
                    let cond = read_u8(&mut r) != 0;
                    let sg = SubGraphId(read_u32(&mut r));
                    let input_count = read_u32(&mut r) as usize;
                    let mut inputs = Vec::with_capacity(input_count);
                    for _ in 0..input_count {
                        inputs.push(NodeId(read_u32(&mut r)));
                    }
                    branches.push((cond, sg, inputs));
                }
                Some(GateBranches { condition_input, branches })
            }
        }).collect()
    };

    // RecordLitInfos
    let record_lit_infos: Vec<Option<RecordLitInfo>> = {
        let r = mem.section(SectionKind::RecordLitInfos);
        let mut r = r;
        (0..n).map(|_| {
            let valid = read_u8(&mut r);
            if valid == 0 {
                let _ = read_u32(&mut r); let _ = read_u32(&mut r); let _ = read_u32(&mut r); let _ = read_u32(&mut r); let _ = read_u32(&mut r); let _ = read_u8(&mut r);
                None
            } else {
                let type_name = { let off = read_u32(&mut r); let len = read_u32(&mut r); mem.read_str(off, len) };
                let fn_count = read_u32(&mut r) as usize;
                let mut field_names = Vec::with_capacity(fn_count);
                for _ in 0..fn_count {
                    let off = read_u32(&mut r); let len = read_u32(&mut r);
                    field_names.push(if off == u32::MAX { None } else { Some(mem.read_str(off, len)) });
                }
                let constructor = { let off = read_u32(&mut r); let len = read_u32(&mut r); mem.read_str(off, len) };
                let kind = u8_to_record_lit_kind(read_u8(&mut r));
                Some(RecordLitInfo { type_name, field_names, constructor, kind })
            }
        }).collect()
    };

    // SelectInfos
    let select_infos: Vec<Option<SelectInfo>> = {
        let r = mem.section(SectionKind::SelectInfos);
        let mut r = r;
        (0..n).map(|_| {
            let valid = read_u8(&mut r);
            if valid == 0 {
                let _ = read_u32(&mut r);
                None
            } else {
                let branch_count = read_u32(&mut r) as usize;
                let mut branches = Vec::with_capacity(branch_count);
                for _ in 0..branch_count {
                    let subgraph_id = SubGraphId(read_u32(&mut r));
                    let event_kind = u8_to_event_kind(read_u8(&mut r));
                    let event_source_node = NodeId(read_u32(&mut r));
                    branches.push(SelectBranch { subgraph_id, event_kind, event_source_node });
                }
                Some(SelectInfo { branches })
            }
        }).collect()
    };

    // TraitConstructInfos
    let trait_construct_infos: Vec<Option<TraitConstructInfo>> = {
        let r = mem.section(SectionKind::TraitConstructInfos);
        let mut r = r;
        (0..n).map(|_| {
            let valid = read_u8(&mut r);
            if valid == 0 {
                let _ = read_u32(&mut r); let _ = read_u32(&mut r); let _ = read_u32(&mut r); let _ = read_u32(&mut r);
                None
            } else {
                let trait_name = { let off = read_u32(&mut r); let len = read_u32(&mut r); mem.read_str(off, len) };
                let mn_count = read_u32(&mut r) as usize;
                let mut method_names = Vec::with_capacity(mn_count);
                for _ in 0..mn_count {
                    let off = read_u32(&mut r); let len = read_u32(&mut r);
                    method_names.push(mem.read_str(off, len));
                }
                let m_count = read_u32(&mut r) as usize;
                let mut methods = Vec::with_capacity(m_count);
                for _ in 0..m_count {
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(read_bytes(&mut r, 8));
                    methods.push(bytes_to_trait_method_entry(&buf));
                }
                Some(TraitConstructInfo { trait_name, method_names, methods })
            }
        }).collect()
    };

    // RecordExtendInfos
    let record_extend_infos: Vec<Option<RecordExtendInfo>> = {
        let r = mem.section(SectionKind::RecordExtendInfos);
        let mut r = r;
        (0..n).map(|_| {
            let valid = read_u8(&mut r);
            if valid == 0 {
                let _ = read_u32(&mut r);
                None
            } else {
                let un_count = read_u32(&mut r) as usize;
                let mut update_names = Vec::with_capacity(un_count);
                for _ in 0..un_count {
                    let off = read_u32(&mut r); let len = read_u32(&mut r);
                    update_names.push(mem.read_str(off, len));
                }
                Some(RecordExtendInfo { update_names })
            }
        }).collect()
    };

    // BatchInfos
    let batch_infos: Vec<Option<BatchInfo>> = {
        let r = mem.section(SectionKind::BatchInfos);
        let mut r = r;
        (0..n).map(|_| {
            let valid = read_u8(&mut r);
            let mut buf = [0u8; 4];
            buf.copy_from_slice(read_bytes(&mut r, 4));
            if valid == 0 { None } else { Some(bytes_to_batch_info(&buf)) }
        }).collect()
    };

    // Downstreams (CSR)
    let downstreams: Vec<Vec<NodeId>> = {
        let r = mem.section(SectionKind::Downstreams);
        let mut r = r;
        let _first_offset = read_u32(&mut r); // offsets[0] = 0
        let mut offsets = vec![0u32];
        for _ in 0..n {
            offsets.push(read_u32(&mut r));
        }
        // flat data follows the offsets region
        let mut ds = Vec::with_capacity(n);
        for i in 0..n {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            let mut inner = Vec::with_capacity(end - start);
            for _ in start..end {
                inner.push(NodeId(read_u32(&mut r)));
            }
            ds.push(inner);
        }
        ds
    };

    // ---- Rebuild runtime fields ----
    let entry_subgraph = if mem.header().entry_subgraph == u32::MAX { None } else { Some(SubGraphId(mem.header().entry_subgraph)) };
    let compute_fns = build_compute_fn_table();
    let global_var_storage = std::sync::Arc::new(
        (0..mem.header().global_var_count)
            .map(|_| std::sync::Mutex::new(None))
            .collect::<Vec<_>>()
    );
    let memo_tables = std::sync::Arc::new(
        (0..mem.header().memo_table_count)
            .map(|_| std::sync::Mutex::new(rustc_hash::FxHashMap::default()))
            .collect::<Vec<_>>()
    );
    // Load the string pool bytes from the StringPool section; ConstValue::Str { offset, len } references this pool.
    let string_pool: Arc<[u8]> = Arc::from(mem.section(SectionKind::StringPool).to_vec());

    Ok(DataFlowGraph {
        nodes,
        inputs_pool,
        subgraphs,
        entry_subgraph,
        compute_fns,
        downstreams,
        const_values,
        call_targets,
        gate_branches,
        field_access_infos,
        record_lit_infos,
        ffi_call_names,
        dyn_ffi_infos: Vec::new(), // v1: DynFfiInfo .kzo serialization not yet supported
        field_set_names,
        vtable_call_methods,
        await_event_sources,
        closure_infos,
        partial_infos,
        closure_call_arg_counts,
        select_infos,
        writeback_targets,
        tail_call_flags,
        safe_op_flags,
        hoisted_node,
        hoisted_owners,
        batch_infos,
        ir_errors: Vec::new(),
        trait_construct_infos,
        lazy_construct_infos,
        record_extend_infos,
        slice_inclusive,
        global_var_storage,
        global_load_slots,
        global_store_slots,
        pattern_ctor_names,
        pattern_type_names,
        pattern_field_indices,
        cast_target_types,
        memo_infos,
        memo_tables,
        vtable_fallback_dispatch: rustc_hash::FxHashMap::default(),
        string_pool,
        mem: None,
        sg_uv_offsets: Vec::new(),
        sg_nr_offsets: Vec::new(),
        gate_branch_offsets: Vec::new(),
        record_lit_info_offsets: Vec::new(),
        select_info_offsets: Vec::new(),
        trait_construct_info_offsets: Vec::new(),
        record_extend_info_offsets: Vec::new(),
    })
}

/// Loads a `DataFlowGraph` from a `GraphMemory` via zerocopy (production path).
///
/// Only eager-loads the 5 complex variable-length tables + subgraphs + downstreams +
/// string_pool + runtime fields. The 24 per-Node scalar tables plus `nodes` and `inputs`
/// are read zerocopy from the mmap slices via accessor methods, without copying into owned
/// `Vec`s.
pub fn load_zerocopy(mem: GraphMemory) -> io::Result<DataFlowGraph> {
    let n = mem.header().node_count as usize;

    // Load inline C machine code (mmap executable)
    // SubGraphs (eager-load: includes variable-length fields upvalue_nodes/nested_ranges/event_decls/defer_table/reset_plan)
    // upvalue_outer_nodes / nested_ranges use zerocopy CSR accessors and are not parsed into owned Vecs.
    let (subgraphs, sg_uv_offsets, sg_nr_offsets) = {
        let sr = mem.section(SectionKind::SubGraphs);
        let ed = mem.section(SectionKind::SgEventDecls);
        let df = mem.section(SectionKind::SgDeferEntries);
        let dc = mem.section(SectionKind::SgDeferCapturedInputs);
        let rp = mem.section(SectionKind::SgResetPlan);

        let sg_count = mem.header().subgraph_count as usize;
        let mut subgraphs = Vec::with_capacity(sg_count);
        let mut sg_uv_offsets = Vec::with_capacity(sg_count);
        let mut sg_nr_offsets = Vec::with_capacity(sg_count);
        let mut sr_r = sr;
        for _ in 0..sg_count {
            let id = SubGraphId(read_u32(&mut sr_r));
            let node_range = (NodeId(read_u32(&mut sr_r)), NodeId(read_u32(&mut sr_r)));
            let param_count = read_u8(&mut sr_r);
            let entry_node = NodeId(read_u32(&mut sr_r));
            let return_node = NodeId(read_u32(&mut sr_r));
            let has_suspend = read_u8(&mut sr_r) != 0;
            let loop_kind = u8_to_loop_kind(read_u8(&mut sr_r));
            let loop_parent_sg = { let v = read_u32(&mut sr_r); if v == u32::MAX { None } else { Some(SubGraphId(v)) } };
            let cond_node = { let v = read_u32(&mut sr_r); if v == u32::MAX { None } else { Some(NodeId(v)) } };
            let function_id = read_u32(&mut sr_r);
            let iter_next_node = { let v = read_u32(&mut sr_r); if v == u32::MAX { None } else { Some(NodeId(v)) } };
            let upvalue_count = read_u8(&mut sr_r);

            // upvalue_outer_nodes: zerocopy CSR — store offset/len, do not parse into a Vec.
            let uv_off = read_u32(&mut sr_r);
            let uv_len = read_u32(&mut sr_r);
            sg_uv_offsets.push((uv_off, uv_len));
            let upvalue_outer_nodes: Vec<NodeId> = Vec::new();

            // nested_ranges: zerocopy CSR — store offset/len, do not parse into a Vec.
            let nr_off = read_u32(&mut sr_r);
            let nr_len = read_u32(&mut sr_r);
            sg_nr_offsets.push((nr_off, nr_len));
            let nested_ranges: Vec<(u32, u32)> = Vec::new();

            let ed_off = read_u32(&mut sr_r) as usize;
            let ed_len = read_u32(&mut sr_r) as usize;
            let event_source_decls: Vec<EventSourceDecl> = (0..ed_len)
                .map(|i| {
                    let base = ed_off + i * 8;
                    EventSourceDecl {
                        node: NodeId(u32::from_le_bytes([ed[base], ed[base+1], ed[base+2], ed[base+3]])),
                        kind: u8_to_event_kind(ed[base+4]),
                    }
                })
                .collect();

            let df_off = read_u32(&mut sr_r) as usize;
            let df_len = read_u32(&mut sr_r) as usize;
            let defer_table: Vec<DeferEntry> = (0..df_len)
                .map(|i| {
                    let base = df_off + i * 16;
                    let trigger_node = NodeId(u32::from_le_bytes([df[base], df[base+1], df[base+2], df[base+3]]));
                    let body_subgraph = SubGraphId(u32::from_le_bytes([df[base+4], df[base+5], df[base+6], df[base+7]]));
                    let ci_off = u32::from_le_bytes([df[base+8], df[base+9], df[base+10], df[base+11]]) as usize;
                    let ci_len = u32::from_le_bytes([df[base+12], df[base+13], df[base+14], df[base+15]]) as usize;
                    let captured_inputs: Vec<NodeId> = (0..ci_len)
                        .map(|j| {
                            let b2 = ci_off + j * 4;
                            NodeId(u32::from_le_bytes([dc[b2], dc[b2+1], dc[b2+2], dc[b2+3]]))
                        })
                        .collect();
                    DeferEntry { trigger_node, body_subgraph, captured_inputs, registered: false }
                })
                .collect();

            let has_reset_plan = read_u8(&mut sr_r) != 0;
            let reset_plan = if has_reset_plan {
                let rp_off = read_u32(&mut sr_r) as usize;
                let mut rp_r = &rp[rp_off..];
                let rz_len = read_u32(&mut rp_r) as usize;
                let reset_to_zero: Vec<NodeId> = (0..rz_len).map(|_| NodeId(read_u32(&mut rp_r))).collect();
                let ro_len = read_u32(&mut rp_r) as usize;
                let reset_to_one: Vec<NodeId> = (0..ro_len).map(|_| NodeId(read_u32(&mut rp_r))).collect();
                let rc_len = read_u32(&mut rp_r) as usize;
                let reset_condition_tree: Vec<NodeId> = (0..rc_len).map(|_| NodeId(read_u32(&mut rp_r))).collect();
                Some(ResetPlan { reset_to_zero, reset_to_one, reset_condition_tree })
            } else {
                let _ = read_u32(&mut sr_r); // skip placeholder
                None
            };

            subgraphs.push(SubGraph {
                id, node_range, param_count, entry_node, return_node, has_suspend,
                event_source_decls, defer_table, loop_kind, loop_parent_sg, cond_node,
                function_id, iter_next_node, upvalue_count, upvalue_outer_nodes,
                nested_ranges, reset_plan,
            });
        }
        (subgraphs, sg_uv_offsets, sg_nr_offsets)
    };

    // Five complex variable-length tables: zerocopy on-demand accessors.
    // Scan each section to record the byte offset of each entry (u32::MAX = None), without
    // constructing owned structures. Accessor methods parse individual entries on demand from
    // the mmap at execution time, eliminating the Vec<Option<T>> array memory.
    let (gate_branch_offsets, gate_branches): (Vec<u32>, Vec<Option<GateBranches>>) = {
        let r = mem.section(SectionKind::GateBranches);
        let mut r = r;
        let mut offsets = Vec::with_capacity(n);
        for _ in 0..n {
            let entry_off = (r.as_ptr() as usize - mem.section(SectionKind::GateBranches).as_ptr() as usize) as u32;
            let valid = read_u8(&mut r);
            if valid == 0 {
                let _ = read_u32(&mut r); let _ = read_u32(&mut r);
                offsets.push(u32::MAX);
            } else {
                let _ = read_u32(&mut r); // condition_input
                let branch_count = read_u32(&mut r) as usize;
                for _ in 0..branch_count {
                    let _ = read_u8(&mut r); // cond
                    let _ = read_u32(&mut r); // sg
                    let ic = read_u32(&mut r) as usize;
                    for _ in 0..ic { let _ = read_u32(&mut r); }
                }
                offsets.push(entry_off);
            }
        }
        (offsets, Vec::new())
    };

    let (record_lit_info_offsets, record_lit_infos): (Vec<u32>, Vec<Option<RecordLitInfo>>) = {
        let section = mem.section(SectionKind::RecordLitInfos);
        let mut r = section;
        let mut offsets = Vec::with_capacity(n);
        for _ in 0..n {
            let entry_off = (r.as_ptr() as usize - section.as_ptr() as usize) as u32;
            let valid = read_u8(&mut r);
            if valid == 0 {
                let _ = read_u32(&mut r); let _ = read_u32(&mut r); let _ = read_u32(&mut r);
                let _ = read_u32(&mut r); let _ = read_u32(&mut r); let _ = read_u8(&mut r);
                offsets.push(u32::MAX);
            } else {
                let _ = read_u32(&mut r); let _ = read_u32(&mut r); // type_name
                let fn_count = read_u32(&mut r) as usize;
                for _ in 0..fn_count { let _ = read_u32(&mut r); let _ = read_u32(&mut r); }
                let _ = read_u32(&mut r); let _ = read_u32(&mut r); // constructor
                let _ = read_u8(&mut r); // kind
                offsets.push(entry_off);
            }
        }
        (offsets, Vec::new())
    };

    let (select_info_offsets, select_infos): (Vec<u32>, Vec<Option<SelectInfo>>) = {
        let section = mem.section(SectionKind::SelectInfos);
        let mut r = section;
        let mut offsets = Vec::with_capacity(n);
        for _ in 0..n {
            let entry_off = (r.as_ptr() as usize - section.as_ptr() as usize) as u32;
            let valid = read_u8(&mut r);
            if valid == 0 {
                let _ = read_u32(&mut r);
                offsets.push(u32::MAX);
            } else {
                let branch_count = read_u32(&mut r) as usize;
                for _ in 0..branch_count {
                    let _ = read_u32(&mut r); // subgraph_id
                    let _ = read_u8(&mut r);  // event_kind
                    let _ = read_u32(&mut r); // event_source_node
                }
                offsets.push(entry_off);
            }
        }
        (offsets, Vec::new())
    };

    let (trait_construct_info_offsets, trait_construct_infos): (Vec<u32>, Vec<Option<TraitConstructInfo>>) = {
        let section = mem.section(SectionKind::TraitConstructInfos);
        let mut r = section;
        let mut offsets = Vec::with_capacity(n);
        for _ in 0..n {
            let entry_off = (r.as_ptr() as usize - section.as_ptr() as usize) as u32;
            let valid = read_u8(&mut r);
            if valid == 0 {
                let _ = read_u32(&mut r); let _ = read_u32(&mut r);
                let _ = read_u32(&mut r); let _ = read_u32(&mut r);
                offsets.push(u32::MAX);
            } else {
                let _ = read_u32(&mut r); let _ = read_u32(&mut r); // trait_name
                let mn_count = read_u32(&mut r) as usize;
                for _ in 0..mn_count { let _ = read_u32(&mut r); let _ = read_u32(&mut r); }
                let m_count = read_u32(&mut r) as usize;
                for _ in 0..m_count { let _ = read_bytes(&mut r, 8); }
                offsets.push(entry_off);
            }
        }
        (offsets, Vec::new())
    };

    let (record_extend_info_offsets, record_extend_infos): (Vec<u32>, Vec<Option<RecordExtendInfo>>) = {
        let section = mem.section(SectionKind::RecordExtendInfos);
        let mut r = section;
        let mut offsets = Vec::with_capacity(n);
        for _ in 0..n {
            let entry_off = (r.as_ptr() as usize - section.as_ptr() as usize) as u32;
            let valid = read_u8(&mut r);
            if valid == 0 {
                let _ = read_u32(&mut r);
                offsets.push(u32::MAX);
            } else {
                let un_count = read_u32(&mut r) as usize;
                for _ in 0..un_count { let _ = read_u32(&mut r); let _ = read_u32(&mut r); }
                offsets.push(entry_off);
            }
        }
        (offsets, Vec::new())
    };

    // Downstreams: no longer eager-loaded; accessed via downstream_slice(idx) zerocopy CSR.
    // The field stays an empty Vec (unused on the load path).

    // ---- Rebuild runtime fields ----
    let entry_subgraph = if mem.header().entry_subgraph == u32::MAX { None } else { Some(SubGraphId(mem.header().entry_subgraph)) };
    let compute_fns = build_compute_fn_table();
    let global_var_storage = std::sync::Arc::new(
        (0..mem.header().global_var_count)
            .map(|_| std::sync::Mutex::new(None))
            .collect::<Vec<_>>()
    );
    let memo_tables = std::sync::Arc::new(
        (0..mem.header().memo_table_count)
            .map(|_| std::sync::Mutex::new(rustc_hash::FxHashMap::default()))
            .collect::<Vec<_>>()
    );
    // string_pool: the zerocopy path does not copy; reads via the string_pool_slice() accessor from the mmap.
    // The field stays an empty Arc (unused on the load path).
    let string_pool: Arc<[u8]> = Arc::from(Vec::new());

    Ok(DataFlowGraph {
        nodes: Vec::new(),
        inputs_pool: InputsPool::new(),
        subgraphs,
        entry_subgraph,
        compute_fns,
        downstreams: Vec::new(),
        const_values: Vec::new(),
        call_targets: Vec::new(),
        gate_branches,
        field_access_infos: Vec::new(),
        record_lit_infos,
        ffi_call_names: Vec::new(),
        dyn_ffi_infos: Vec::new(), // v1: DynFfiInfo .kzo serialization not yet supported
        field_set_names: Vec::new(),
        vtable_call_methods: Vec::new(),
        await_event_sources: Vec::new(),
        closure_infos: Vec::new(),
        partial_infos: Vec::new(),
        closure_call_arg_counts: Vec::new(),
        select_infos,
        writeback_targets: Vec::new(),
        tail_call_flags: Vec::new(),
        safe_op_flags: Vec::new(),
        hoisted_node: Vec::new(),
        hoisted_owners: Vec::new(),
        batch_infos: Vec::new(),
        ir_errors: Vec::new(),
        trait_construct_infos,
        lazy_construct_infos: Vec::new(),
        record_extend_infos,
        slice_inclusive: Vec::new(),
        global_var_storage,
        global_load_slots: Vec::new(),
        global_store_slots: Vec::new(),
        pattern_ctor_names: Vec::new(),
        pattern_type_names: Vec::new(),
        pattern_field_indices: Vec::new(),
        cast_target_types: Vec::new(),
        memo_infos: Vec::new(),
        memo_tables,
        vtable_fallback_dispatch: rustc_hash::FxHashMap::default(),
        string_pool,
        mem: Some(mem),
        sg_uv_offsets,
        sg_nr_offsets,
        gate_branch_offsets,
        record_lit_info_offsets,
        select_info_offsets,
        trait_construct_info_offsets,
        record_extend_info_offsets,
    })
}

/// Loads zerocopy from a byte stream (for the serialize->load->run path; no file needed).
pub fn load_zerocopy_from_bytes(data: Vec<u8>) -> io::Result<DataFlowGraph> {
    let mem = GraphMemory::from_bytes(&data)?;
    load_zerocopy(mem)
}

// ==================== inspect ====================

/// `.kzo` file metadata (for the inspect command).
pub struct SolidifyInfo {
    pub schema_version: u16,
    pub abi_version: u16,
    pub node_count: u32,
    pub subgraph_count: u32,
    pub entry_subgraph: Option<u32>,
    pub input_count: u32,
    pub string_pool_len: u32,
    pub global_var_count: u32,
    pub memo_table_count: u32,
    pub compute_fn_count: u32,
    pub crc32: u32,
    pub section_count: u16,
    pub file_size: usize,
    /// Details of each section (kind_u8, offset, len); only populated when `inspect --verbose`.
    pub sections: Vec<(u8, u32, u32)>,
}

/// Reads `.kzo` file metadata (header only, does not load the full graph); mmap path.
pub fn inspect_solidify_from_file(path: &str) -> io::Result<SolidifyInfo> {
    let mem = GraphMemory::from_file(path)?;
    let h = mem.header();
    let file_size = std::fs::metadata(path).map(|m| m.len() as usize).unwrap_or(0);
    Ok(SolidifyInfo {
        schema_version: h.schema_version,
        abi_version: h.abi_version,
        node_count: h.node_count,
        subgraph_count: h.subgraph_count,
        entry_subgraph: if h.entry_subgraph == u32::MAX { None } else { Some(h.entry_subgraph) },
        input_count: h.input_count,
        string_pool_len: h.string_pool_len,
        global_var_count: h.global_var_count,
        memo_table_count: h.memo_table_count,
        compute_fn_count: h.compute_fn_count,
        crc32: h.crc32,
        section_count: h.section_count,
        file_size,
        sections: mem.sections_detail(),
    })
}

/// Reads `.kzo` byte-stream metadata (header only, does not load the full graph); owned path.
pub fn inspect_solidify(data: &[u8]) -> io::Result<SolidifyInfo> {
    let mem = GraphMemory::from_bytes(data)?;
    let h = mem.header();
    Ok(SolidifyInfo {
        schema_version: h.schema_version,
        abi_version: h.abi_version,
        node_count: h.node_count,
        subgraph_count: h.subgraph_count,
        entry_subgraph: if h.entry_subgraph == u32::MAX { None } else { Some(h.entry_subgraph) },
        input_count: h.input_count,
        string_pool_len: h.string_pool_len,
        global_var_count: h.global_var_count,
        memo_table_count: h.memo_table_count,
        compute_fn_count: h.compute_fn_count,
        crc32: h.crc32,
        section_count: h.section_count,
        file_size: data.len(),
        sections: mem.sections_detail(),
    })
}
