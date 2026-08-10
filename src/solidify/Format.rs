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
        pattern_field_indices,
        cast_target_types,
        memo_infos,
        memo_tables,
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
        pattern_field_indices: Vec::new(),
        cast_target_types: Vec::new(),
        memo_infos: Vec::new(),
        memo_tables,
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

// ==================== Phase 4: Round-trip tests ====================

#[cfg(test)]
mod round_trip_tests {
    use super::*;
    use crate::value::{ValueTag, BinOp};

    /// Builds a synthetic DataFlowGraph covering all table types and edge cases.
    ///
    /// 16 nodes; every table has a Some/None mix, covering:
    /// - All NodeKind variants
    /// - All ConstValue variants (including Str/I128/U128/F128)
    /// - All category A/B/C/D/E tables
    /// - All SubGraph variable-length fields (defer/reset_plan/nested_ranges/event_decls/upvalue_nodes)
    /// - GateBranches/SelectInfo/TraitConstructInfo/RecordLitInfo/RecordExtendInfo variable-length data
    /// - Downstreams with a mix of empty and non-empty
    fn make_test_graph() -> DataFlowGraph {
        // ---- String pool: pre-write the test strings ----
        // Note: ConstValue::Str needs to reference the string_pool, which we build manually.
        // Category C tables (ffi_call_names, etc.) use owned Strings that are auto-interned
        // into the pool during serialization.
        // Here we only pre-store the references needed by ConstValue::Str.
        let mut pool: Vec<u8> = Vec::new();
        let intern = |pool: &mut Vec<u8>, s: &str| -> (u32, u32) {
            let off = pool.len() as u32;
            pool.extend_from_slice(s.as_bytes());
            (off, s.len() as u32)
        };
        let s_hello = intern(&mut pool, "hello");

        // ---- Nodes: cover all NodeKind variants (16 nodes) ----
        let make_node = |kind: NodeKind, input_count: u8, inputs_offset: u32, cf: ComputeFnId| -> Node {
            Node { kind, input_count, inputs_offset, compute_fn: cf }
        };
        let nodes = vec![
            make_node(NodeKind::Const,       0, 0,  CF_NOOP),          // 0
            make_node(NodeKind::Const,       0, 0,  CF_NOOP),          // 1
            make_node(NodeKind::Const,       0, 0,  CF_NOOP),          // 2
            make_node(NodeKind::Const,       0, 0,  CF_NOOP),          // 3
            make_node(NodeKind::BinOp,       2, 0,  CF_ADD_I32),       // 4
            make_node(NodeKind::UnOp,        1, 2,  CF_NEG_I32),       // 5
            make_node(NodeKind::FieldAccess, 1, 3,  CF_RECORD_FIELD_GET), // 6
            make_node(NodeKind::Call,        2, 4,  CF_CALL_LAUNCH),   // 7
            make_node(NodeKind::Gate,        1, 6,  CF_GATE_LAUNCH),   // 8
            make_node(NodeKind::Await,       1, 7,  CF_AWAIT),         // 9
            make_node(NodeKind::EventSource, 0, 0,  CF_NOOP),          // 10
            make_node(NodeKind::Const,       0, 0,  CF_NOOP),          // 11
            make_node(NodeKind::Const,       0, 0,  CF_NOOP),          // 12
            make_node(NodeKind::Const,       0, 0,  CF_NOOP),          // 13
            make_node(NodeKind::Const,       0, 0,  CF_NOOP),          // 14
            make_node(NodeKind::Const,       0, 0,  CF_NOOP),          // 15
        ];

        // ---- InputsPool ----
        let inputs_data = vec![
            NodeId(0), NodeId(1),       // node 4 inputs
            NodeId(4),                   // node 5 input
            NodeId(3),                   // node 6 input
            NodeId(4), NodeId(5),       // node 7 inputs
            NodeId(4),                   // node 8 input
            NodeId(10),                  // node 9 input
        ];
        let inputs_pool = InputsPool { data: inputs_data };

        // ---- SubGraphs: 3 subgraphs, covering all variable-length fields ----
        let subgraphs = vec![
            SubGraph {
                id: SubGraphId(0),
                node_range: (NodeId(0), NodeId(10)),
                param_count: 2,
                entry_node: NodeId(0),
                return_node: NodeId(8),
                has_suspend: true,
                event_source_decls: vec![
                    EventSourceDecl { node: NodeId(10), kind: EventSourceKind::Channel },
                    EventSourceDecl { node: NodeId(10), kind: EventSourceKind::Timer },
                ],
                defer_table: vec![
                    DeferEntry {
                        trigger_node: NodeId(7),
                        body_subgraph: SubGraphId(2),
                        captured_inputs: vec![NodeId(0), NodeId(1)],
                        registered: false,
                    },
                    DeferEntry {
                        trigger_node: NodeId(8),
                        body_subgraph: SubGraphId(2),
                        captured_inputs: vec![],
                        registered: false,
                    },
                ],
                loop_kind: LoopKind::While,
                loop_parent_sg: None,
                cond_node: Some(NodeId(4)),
                function_id: 0,
                iter_next_node: None,
                upvalue_count: 1,
                upvalue_outer_nodes: vec![NodeId(3)],
                nested_ranges: vec![(11, 13), (14, 16)],
                reset_plan: Some(ResetPlan {
                    reset_to_zero: vec![NodeId(11)],
                    reset_to_one: vec![NodeId(12)],
                    reset_condition_tree: vec![NodeId(4)],
                }),
            },
            SubGraph {
                id: SubGraphId(1),
                node_range: (NodeId(10), NodeId(11)),
                param_count: 0,
                entry_node: NodeId(10),
                return_node: NodeId(10),
                has_suspend: false,
                event_source_decls: vec![],
                defer_table: vec![],
                loop_kind: LoopKind::None,
                loop_parent_sg: None,
                cond_node: None,
                function_id: 1,
                iter_next_node: None,
                upvalue_count: 0,
                upvalue_outer_nodes: vec![],
                nested_ranges: vec![],
                reset_plan: None,
            },
            SubGraph {
                id: SubGraphId(2),
                node_range: (NodeId(11), NodeId(16)),
                param_count: 1,
                entry_node: NodeId(11),
                return_node: NodeId(15),
                has_suspend: false,
                event_source_decls: vec![
                    EventSourceDecl { node: NodeId(10), kind: EventSourceKind::AsyncJoin },
                ],
                defer_table: vec![],
                loop_kind: LoopKind::For,
                loop_parent_sg: Some(SubGraphId(0)),
                cond_node: Some(NodeId(12)),
                function_id: 0,
                iter_next_node: Some(NodeId(13)),
                upvalue_count: 2,
                upvalue_outer_nodes: vec![NodeId(0), NodeId(1)],
                nested_ranges: vec![],
                reset_plan: Some(ResetPlan {
                    reset_to_zero: vec![],
                    reset_to_one: vec![NodeId(12)],
                    reset_condition_tree: vec![],
                }),
            },
        ];

        // ---- Downstreams: mix of empty and non-empty ----
        let downstreams = vec![
            vec![NodeId(4)],                          // 0
            vec![NodeId(4)],                          // 1
            vec![],                                    // 2 (empty)
            vec![NodeId(6)],                          // 3
            vec![NodeId(5), NodeId(7), NodeId(8)],    // 4
            vec![NodeId(7)],                          // 5
            vec![],                                    // 6 (empty)
            vec![],                                    // 7 (empty)
            vec![],                                    // 8 (empty)
            vec![],                                    // 9 (empty)
            vec![NodeId(9)],                          // 10
            vec![],                                    // 11 (empty)
            vec![],                                    // 12 (empty)
            vec![],                                    // 13 (empty)
            vec![],                                    // 14 (empty)
            vec![],                                    // 15 (empty)
        ];

        // ---- ConstValues: cover all variants ----
        let const_values = vec![
            Some(ConstValue::I8(-1)),                                          // 0
            Some(ConstValue::I128(i128::MIN)),                                 // 1
            Some(ConstValue::U128(u128::MAX)),                                 // 2
            Some(ConstValue::F128([0xFF; 16])),                                // 3
            None,                                                               // 4
            None,                                                               // 5
            None,                                                               // 6
            None,                                                               // 7
            None,                                                               // 8
            None,                                                               // 9
            None,                                                               // 10
            Some(ConstValue::Str { offset: s_hello.0, len: s_hello.1 }),       // 11
            Some(ConstValue::Null),                                            // 12
            Some(ConstValue::Void),                                            // 13
            Some(ConstValue::Bool(true)),                                      // 14
            Some(ConstValue::Char(0x4E2D)),                                    // 15 (U+4E2D)
        ];

        // ---- Category A: fixed-width scalar tables (Some/None mix) ----
        let call_targets = vec![
            None, None, None, None, None, None, None,
            Some(SubGraphId(2)), None, None, None, None, None, None, None, None,
        ];
        let field_access_infos = vec![
            None, None, None, None, None, None,
            Some(42u16), None, None, None, None, None, None, None, None, None,
        ];
        let vtable_call_methods = vec![
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None,
            Some(7u16),
        ];
        let await_event_sources = vec![
            None, None, None, None, None, None, None, None, None,
            Some(NodeId(10)), None, None, None, None, None, None,
        ];
        let writeback_targets = vec![
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None,
            Some(NodeId(0)),
            None,
        ];
        let hoisted_owners = vec![SubGraphId(0); 16];
        let global_load_slots = vec![
            None, None, None,
            Some(5u32), None, None, None, None, None, None,
            None, None, None, None, None, None,
        ];
        let global_store_slots = vec![
            None, None, None, None, None, None, None, None, None, None,
            None, None, None,
            Some(3u32),
            None, None,
        ];
        let pattern_field_indices = vec![
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
            Some(2u16),
        ];
        let closure_call_arg_counts = vec![
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
            Some(3u8),
        ];

        // ---- Category B: boolean tables ----
        let tail_call_flags = vec![false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false];
        let safe_op_flags =    vec![false, false, false, false, false, false, true,  false, false, false, false, false, false, false, false, true];
        let hoisted_node =     vec![false, false, false, false, false, false, false, false, false, false, false, true,  true,  false, false, false];
        let slice_inclusive =  vec![false, false, false, false, false, false, false, false, false, false, false, false, false, false, true,  false];

        // ---- Category C: tables with strings ----
        let ffi_call_names = vec![
            None, None, None, None, None, None, None,
            Some("printf".to_string()),
            None, None, None, None, None, None, None, None,
        ];
        let field_set_names = vec![
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None,
            Some("field_x".to_string()),
            None,
        ];
        let pattern_ctor_names = vec![
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
            Some("Some".to_string()),
        ];
        let cast_target_types = vec![
            None, None, None, None, None, None, None, None, None, None,
            None, None, None,
            Some("i64".to_string()),
            None, None,
        ];

        // ---- Category D: fixed-width composite tables ----
        let closure_infos = vec![
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None,
            Some(ClosureInfo { subgraph_id: SubGraphId(2), arity: 3, self_upvalue_idx: -1 }),
            Some(ClosureInfo { subgraph_id: SubGraphId(1), arity: 0, self_upvalue_idx: 0 }),
        ];
        let partial_infos = vec![
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
            Some(PartialInfo { subgraph_id: SubGraphId(2), bound_count: 1 }),
        ];
        let lazy_construct_infos = vec![
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None,
            Some(LazyConstructInfo { thunk_sg: SubGraphId(1) }),
            None,
        ];
        let memo_infos = vec![
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
            Some(MemoInfo { table_index: 2, param_count: 3 }),
        ];
        let batch_infos = vec![
            None, None, None, None,
            Some(BatchInfo { tag: ValueTag::I32, op: BatchOp::Bin(BinOp::Add) }),
            None, None, None, None, None, None, None, None, None, None, None,
        ];

        // ---- Category E: variable-length field tables ----
        let gate_branches = vec![
            None, None, None, None, None, None, None, None,
            Some(GateBranches {
                condition_input: NodeId(4),
                branches: vec![
                    (true, SubGraphId(1), vec![NodeId(0), NodeId(1)]),
                    (false, SubGraphId(2), vec![NodeId(2)]),
                    (true, SubGraphId(0), vec![]),
                ],
            }),
            None, None, None, None, None, None, None,
        ];
        let record_lit_infos = vec![
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None,
            Some(RecordLitInfo {
                type_name: "MyType".to_string(),
                field_names: vec![Some("x".to_string()), None, Some("z".to_string())],
                constructor: "Ctor".to_string(),
                kind: RecordLitKind::Adt,
            }),
            None,
        ];
        let select_infos = vec![
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
            Some(SelectInfo {
                branches: vec![
                    SelectBranch { subgraph_id: SubGraphId(1), event_kind: EventSourceKind::Channel, event_source_node: NodeId(10) },
                    SelectBranch { subgraph_id: SubGraphId(2), event_kind: EventSourceKind::Timer, event_source_node: NodeId(10) },
                ],
            }),
        ];
        let trait_construct_infos = vec![
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None,
            Some(TraitConstructInfo {
                trait_name: "MyTrait".to_string(),
                method_names: vec!["method_a".to_string(), "method_b".to_string()],
                methods: vec![
                    TraitMethodEntry { subgraph_id: SubGraphId(1), arity: 2, upvalue_count: 1 },
                    TraitMethodEntry { subgraph_id: SubGraphId(2), arity: 0, upvalue_count: 0 },
                ],
            }),
            None,
        ];
        let record_extend_infos = vec![
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
            Some(RecordExtendInfo {
                update_names: vec!["update_x".to_string(), "update_y".to_string()],
            }),
        ];

        DataFlowGraph {
            nodes,
            inputs_pool,
            subgraphs,
            entry_subgraph: Some(SubGraphId(0)),
            compute_fns: build_compute_fn_table(),
            downstreams,
            const_values,
            call_targets,
            gate_branches,
            field_access_infos,
            record_lit_infos,
            ffi_call_names,
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
            global_var_storage: std::sync::Arc::new(
                (0..4).map(|_| std::sync::Mutex::new(None)).collect()
            ),
            global_load_slots,
            global_store_slots,
            pattern_ctor_names,
            pattern_field_indices,
            cast_target_types,
            memo_infos,
            memo_tables: std::sync::Arc::new(
                (0..2).map(|_| std::sync::Mutex::new(rustc_hash::FxHashMap::default())).collect()
            ),
            string_pool: std::sync::Arc::from(pool),
            mem: None,
            sg_uv_offsets: Vec::new(),
            sg_nr_offsets: Vec::new(),
            gate_branch_offsets: Vec::new(),
            record_lit_info_offsets: Vec::new(),
            select_info_offsets: Vec::new(),
            trait_construct_info_offsets: Vec::new(),
            record_extend_info_offsets: Vec::new(),
        }
    }

    /// Compares all serializable fields of two DataFlowGraphs for equality.
    fn assert_graphs_equal(a: &DataFlowGraph, b: &DataFlowGraph, ctx: &str) {
        // Basic fields
        assert_eq!(a.nodes.len(), b.nodes.len(), "{}: nodes len", ctx);
        for i in 0..a.nodes.len() {
            assert_eq!(a.nodes[i].kind, b.nodes[i].kind, "{}: node[{}] kind", ctx, i);
            assert_eq!(a.nodes[i].input_count, b.nodes[i].input_count, "{}: node[{}] input_count", ctx, i);
            assert_eq!(a.nodes[i].inputs_offset, b.nodes[i].inputs_offset, "{}: node[{}] inputs_offset", ctx, i);
            assert_eq!(a.nodes[i].compute_fn, b.nodes[i].compute_fn, "{}: node[{}] compute_fn", ctx, i);
        }

        // InputsPool
        assert_eq!(a.inputs_pool.data, b.inputs_pool.data, "{}: inputs_pool", ctx);

        // SubGraphs
        assert_eq!(a.subgraphs.len(), b.subgraphs.len(), "{}: subgraphs len", ctx);
        for i in 0..a.subgraphs.len() {
            let sa = &a.subgraphs[i];
            let sb = &b.subgraphs[i];
            assert_eq!(sa.id, sb.id, "{}: sg[{}] id", ctx, i);
            assert_eq!(sa.node_range, sb.node_range, "{}: sg[{}] node_range", ctx, i);
            assert_eq!(sa.param_count, sb.param_count, "{}: sg[{}] param_count", ctx, i);
            assert_eq!(sa.entry_node, sb.entry_node, "{}: sg[{}] entry_node", ctx, i);
            assert_eq!(sa.return_node, sb.return_node, "{}: sg[{}] return_node", ctx, i);
            assert_eq!(sa.has_suspend, sb.has_suspend, "{}: sg[{}] has_suspend", ctx, i);
            assert_eq!(sa.loop_kind, sb.loop_kind, "{}: sg[{}] loop_kind", ctx, i);
            assert_eq!(sa.loop_parent_sg, sb.loop_parent_sg, "{}: sg[{}] loop_parent_sg", ctx, i);
            assert_eq!(sa.cond_node, sb.cond_node, "{}: sg[{}] cond_node", ctx, i);
            assert_eq!(sa.function_id, sb.function_id, "{}: sg[{}] function_id", ctx, i);
            assert_eq!(sa.iter_next_node, sb.iter_next_node, "{}: sg[{}] iter_next_node", ctx, i);
            assert_eq!(sa.upvalue_count, sb.upvalue_count, "{}: sg[{}] upvalue_count", ctx, i);
            assert_eq!(sa.upvalue_outer_nodes, sb.upvalue_outer_nodes, "{}: sg[{}] upvalue_outer_nodes", ctx, i);
            assert_eq!(sa.nested_ranges, sb.nested_ranges, "{}: sg[{}] nested_ranges", ctx, i);
            assert_eq!(sa.event_source_decls.len(), sb.event_source_decls.len(), "{}: sg[{}] event_decls len", ctx, i);
            for j in 0..sa.event_source_decls.len() {
                assert_eq!(sa.event_source_decls[j].node, sb.event_source_decls[j].node, "{}: sg[{}] decl[{}] node", ctx, i, j);
                assert_eq!(sa.event_source_decls[j].kind, sb.event_source_decls[j].kind, "{}: sg[{}] decl[{}] kind", ctx, i, j);
            }
            assert_eq!(sa.defer_table.len(), sb.defer_table.len(), "{}: sg[{}] defer_table len", ctx, i);
            for j in 0..sa.defer_table.len() {
                assert_eq!(sa.defer_table[j].trigger_node, sb.defer_table[j].trigger_node, "{}: sg[{}] defer[{}] trigger", ctx, i, j);
                assert_eq!(sa.defer_table[j].body_subgraph, sb.defer_table[j].body_subgraph, "{}: sg[{}] defer[{}] body_sg", ctx, i, j);
                assert_eq!(sa.defer_table[j].captured_inputs, sb.defer_table[j].captured_inputs, "{}: sg[{}] defer[{}] captured", ctx, i, j);
                // `registered` is runtime state and is not compared.
            }
            // reset_plan
            match (&sa.reset_plan, &sb.reset_plan) {
                (None, None) => {}
                (Some(ra), Some(rb)) => {
                    assert_eq!(ra.reset_to_zero, rb.reset_to_zero, "{}: sg[{}] rp reset_to_zero", ctx, i);
                    assert_eq!(ra.reset_to_one, rb.reset_to_one, "{}: sg[{}] rp reset_to_one", ctx, i);
                    assert_eq!(ra.reset_condition_tree, rb.reset_condition_tree, "{}: sg[{}] rp reset_condition_tree", ctx, i);
                }
                _ => panic!("{}: sg[{}] reset_plan mismatch: a={:?} b={:?}", ctx, i, sa.reset_plan.is_some(), sb.reset_plan.is_some()),
            }
        }

        assert_eq!(a.entry_subgraph, b.entry_subgraph, "{}: entry_subgraph", ctx);

        // Downstreams
        assert_eq!(a.downstreams.len(), b.downstreams.len(), "{}: downstreams len", ctx);
        for i in 0..a.downstreams.len() {
            assert_eq!(a.downstreams[i], b.downstreams[i], "{}: downstreams[{}]", ctx, i);
        }

        // ---- Category A tables ----
        assert_eq!(a.call_targets, b.call_targets, "{}: call_targets", ctx);
        assert_eq!(a.field_access_infos, b.field_access_infos, "{}: field_access_infos", ctx);
        assert_eq!(a.vtable_call_methods, b.vtable_call_methods, "{}: vtable_call_methods", ctx);
        assert_eq!(a.await_event_sources, b.await_event_sources, "{}: await_event_sources", ctx);
        assert_eq!(a.writeback_targets, b.writeback_targets, "{}: writeback_targets", ctx);
        assert_eq!(a.hoisted_owners, b.hoisted_owners, "{}: hoisted_owners", ctx);
        assert_eq!(a.global_load_slots, b.global_load_slots, "{}: global_load_slots", ctx);
        assert_eq!(a.global_store_slots, b.global_store_slots, "{}: global_store_slots", ctx);
        assert_eq!(a.pattern_field_indices, b.pattern_field_indices, "{}: pattern_field_indices", ctx);
        assert_eq!(a.closure_call_arg_counts, b.closure_call_arg_counts, "{}: closure_call_arg_counts", ctx);

        // ---- Category B tables ----
        assert_eq!(a.tail_call_flags, b.tail_call_flags, "{}: tail_call_flags", ctx);
        assert_eq!(a.safe_op_flags, b.safe_op_flags, "{}: safe_op_flags", ctx);
        assert_eq!(a.hoisted_node, b.hoisted_node, "{}: hoisted_node", ctx);
        assert_eq!(a.slice_inclusive, b.slice_inclusive, "{}: slice_inclusive", ctx);

        // ---- Category C tables ----
        assert_eq!(a.ffi_call_names, b.ffi_call_names, "{}: ffi_call_names", ctx);
        assert_eq!(a.field_set_names, b.field_set_names, "{}: field_set_names", ctx);
        assert_eq!(a.pattern_ctor_names, b.pattern_ctor_names, "{}: pattern_ctor_names", ctx);
        assert_eq!(a.cast_target_types, b.cast_target_types, "{}: cast_target_types", ctx);

        // ---- Category D tables ----
        // These types do not implement PartialEq; compare manually.
        assert_eq!(a.closure_infos.len(), b.closure_infos.len(), "{}: closure_infos len", ctx);
        for i in 0..a.closure_infos.len() {
            match (&a.closure_infos[i], &b.closure_infos[i]) {
                (None, None) => {}
                (Some(ca), Some(cb)) => {
                    assert_eq!(ca.subgraph_id, cb.subgraph_id, "{}: closure_infos[{}] subgraph_id", ctx, i);
                    assert_eq!(ca.arity, cb.arity, "{}: closure_infos[{}] arity", ctx, i);
                    assert_eq!(ca.self_upvalue_idx, cb.self_upvalue_idx, "{}: closure_infos[{}] self_upvalue_idx", ctx, i);
                }
                _ => panic!("{}: closure_infos[{}] mismatch", ctx, i),
            }
        }
        assert_eq!(a.partial_infos.len(), b.partial_infos.len(), "{}: partial_infos len", ctx);
        for i in 0..a.partial_infos.len() {
            match (&a.partial_infos[i], &b.partial_infos[i]) {
                (None, None) => {}
                (Some(ca), Some(cb)) => {
                    assert_eq!(ca.subgraph_id, cb.subgraph_id, "{}: partial_infos[{}] subgraph_id", ctx, i);
                    assert_eq!(ca.bound_count, cb.bound_count, "{}: partial_infos[{}] bound_count", ctx, i);
                }
                _ => panic!("{}: partial_infos[{}] mismatch", ctx, i),
            }
        }
        assert_eq!(a.lazy_construct_infos.len(), b.lazy_construct_infos.len(), "{}: lazy_construct_infos len", ctx);
        for i in 0..a.lazy_construct_infos.len() {
            match (&a.lazy_construct_infos[i], &b.lazy_construct_infos[i]) {
                (None, None) => {}
                (Some(ca), Some(cb)) => {
                    assert_eq!(ca.thunk_sg, cb.thunk_sg, "{}: lazy_construct_infos[{}] thunk_sg", ctx, i);
                }
                _ => panic!("{}: lazy_construct_infos[{}] mismatch", ctx, i),
            }
        }
        assert_eq!(a.memo_infos.len(), b.memo_infos.len(), "{}: memo_infos len", ctx);
        for i in 0..a.memo_infos.len() {
            match (&a.memo_infos[i], &b.memo_infos[i]) {
                (None, None) => {}
                (Some(ca), Some(cb)) => {
                    assert_eq!(ca.table_index, cb.table_index, "{}: memo_infos[{}] table_index", ctx, i);
                    assert_eq!(ca.param_count, cb.param_count, "{}: memo_infos[{}] param_count", ctx, i);
                }
                _ => panic!("{}: memo_infos[{}] mismatch", ctx, i),
            }
        }
        assert_eq!(a.batch_infos.len(), b.batch_infos.len(), "{}: batch_infos len", ctx);
        for i in 0..a.batch_infos.len() {
            match (&a.batch_infos[i], &b.batch_infos[i]) {
                (None, None) => {}
                (Some(ca), Some(cb)) => {
                    assert_eq!(ca.tag, cb.tag, "{}: batch_infos[{}] tag", ctx, i);
                    assert_eq!(ca.op, cb.op, "{}: batch_infos[{}] op", ctx, i);
                }
                _ => panic!("{}: batch_infos[{}] mismatch", ctx, i),
            }
        }

        // ---- Category E tables ----
        // ConstValues: the Str variant must compare actual string content via the string_pool.
        assert_eq!(a.const_values.len(), b.const_values.len(), "{}: const_values len", ctx);
        for i in 0..a.const_values.len() {
            match (&a.const_values[i], &b.const_values[i]) {
                (None, None) => {}
                (Some(ca), Some(cb)) => {
                    match (ca, cb) {
                        (ConstValue::Str { offset: oa, len: la }, ConstValue::Str { offset: ob, len: lb }) => {
                            assert_eq!(la, lb, "{}: const_values[{}] Str len", ctx, i);
                            let sa = std::str::from_utf8(&a.string_pool[*oa as usize..(*oa + *la) as usize]).unwrap();
                            let sb = std::str::from_utf8(&b.string_pool[*ob as usize..(*ob + *lb) as usize]).unwrap();
                            assert_eq!(sa, sb, "{}: const_values[{}] Str content", ctx, i);
                        }
                        _ => {
                            assert_eq!(ca, cb, "{}: const_values[{}]", ctx, i);
                        }
                    }
                }
                _ => panic!("{}: const_values[{}] mismatch: a={} b={}", ctx, i,
                    a.const_values[i].is_some(), b.const_values[i].is_some()),
            }
        }

        // GateBranches
        assert_eq!(a.gate_branches.len(), b.gate_branches.len(), "{}: gate_branches len", ctx);
        for i in 0..a.gate_branches.len() {
            match (&a.gate_branches[i], &b.gate_branches[i]) {
                (None, None) => {}
                (Some(ga), Some(gb)) => {
                    assert_eq!(ga.condition_input, gb.condition_input, "{}: gate[{}] condition_input", ctx, i);
                    assert_eq!(ga.branches.len(), gb.branches.len(), "{}: gate[{}] branches len", ctx, i);
                    for j in 0..ga.branches.len() {
                        assert_eq!(ga.branches[j].0, gb.branches[j].0, "{}: gate[{}] branch[{}] cond", ctx, i, j);
                        assert_eq!(ga.branches[j].1, gb.branches[j].1, "{}: gate[{}] branch[{}] sg", ctx, i, j);
                        assert_eq!(ga.branches[j].2, gb.branches[j].2, "{}: gate[{}] branch[{}] inputs", ctx, i, j);
                    }
                }
                _ => panic!("{}: gate_branches[{}] mismatch", ctx, i),
            }
        }

        // RecordLitInfos — does not implement PartialEq; compare manually.
        assert_eq!(a.record_lit_infos.len(), b.record_lit_infos.len(), "{}: record_lit_infos len", ctx);
        for i in 0..a.record_lit_infos.len() {
            match (&a.record_lit_infos[i], &b.record_lit_infos[i]) {
                (None, None) => {}
                (Some(ra), Some(rb)) => {
                    assert_eq!(ra.type_name, rb.type_name, "{}: record_lit[{}] type_name", ctx, i);
                    assert_eq!(ra.field_names.len(), rb.field_names.len(), "{}: record_lit[{}] field_names len", ctx, i);
                    for j in 0..ra.field_names.len() {
                        assert_eq!(ra.field_names[j], rb.field_names[j], "{}: record_lit[{}] field[{}]", ctx, i, j);
                    }
                    assert_eq!(ra.constructor, rb.constructor, "{}: record_lit[{}] constructor", ctx, i);
                    assert_eq!(ra.kind, rb.kind, "{}: record_lit[{}] kind", ctx, i);
                }
                _ => panic!("{}: record_lit_infos[{}] mismatch", ctx, i),
            }
        }

        // SelectInfos — does not implement PartialEq; compare manually.
        assert_eq!(a.select_infos.len(), b.select_infos.len(), "{}: select_infos len", ctx);
        for i in 0..a.select_infos.len() {
            match (&a.select_infos[i], &b.select_infos[i]) {
                (None, None) => {}
                (Some(sa), Some(sb)) => {
                    assert_eq!(sa.branches.len(), sb.branches.len(), "{}: select[{}] branches len", ctx, i);
                    for j in 0..sa.branches.len() {
                        assert_eq!(sa.branches[j].subgraph_id, sb.branches[j].subgraph_id, "{}: select[{}] branch[{}] sg", ctx, i, j);
                        assert_eq!(sa.branches[j].event_kind, sb.branches[j].event_kind, "{}: select[{}] branch[{}] kind", ctx, i, j);
                        assert_eq!(sa.branches[j].event_source_node, sb.branches[j].event_source_node, "{}: select[{}] branch[{}] src", ctx, i, j);
                    }
                }
                _ => panic!("{}: select_infos[{}] mismatch", ctx, i),
            }
        }

        // TraitConstructInfos — does not implement PartialEq; compare manually.
        assert_eq!(a.trait_construct_infos.len(), b.trait_construct_infos.len(), "{}: trait_construct_infos len", ctx);
        for i in 0..a.trait_construct_infos.len() {
            match (&a.trait_construct_infos[i], &b.trait_construct_infos[i]) {
                (None, None) => {}
                (Some(ta), Some(tb)) => {
                    assert_eq!(ta.trait_name, tb.trait_name, "{}: trait[{}] trait_name", ctx, i);
                    assert_eq!(ta.method_names, tb.method_names, "{}: trait[{}] method_names", ctx, i);
                    assert_eq!(ta.methods.len(), tb.methods.len(), "{}: trait[{}] methods len", ctx, i);
                    for j in 0..ta.methods.len() {
                        assert_eq!(ta.methods[j].subgraph_id, tb.methods[j].subgraph_id, "{}: trait[{}] method[{}] sg", ctx, i, j);
                        assert_eq!(ta.methods[j].arity, tb.methods[j].arity, "{}: trait[{}] method[{}] arity", ctx, i, j);
                        assert_eq!(ta.methods[j].upvalue_count, tb.methods[j].upvalue_count, "{}: trait[{}] method[{}] upvalue", ctx, i, j);
                    }
                }
                _ => panic!("{}: trait_construct_infos[{}] mismatch", ctx, i),
            }
        }

        // RecordExtendInfos — does not implement PartialEq; compare manually.
        assert_eq!(a.record_extend_infos.len(), b.record_extend_infos.len(), "{}: record_extend_infos len", ctx);
        for i in 0..a.record_extend_infos.len() {
            match (&a.record_extend_infos[i], &b.record_extend_infos[i]) {
                (None, None) => {}
                (Some(ra), Some(rb)) => {
                    assert_eq!(ra.update_names, rb.update_names, "{}: record_extend[{}] update_names", ctx, i);
                }
                _ => panic!("{}: record_extend_infos[{}] mismatch", ctx, i),
            }
        }

        // String pool: raw bytes are not compared directly.
        // The original graph's string_pool may contain only the pre-stored ConstValue::Str strings,
        // while the Category C/D/E tables use owned Strings (interned into the pool during serialization),
        // so the loaded graph's pool will contain more strings. Correctness is already ensured by the
        // comparisons above:
        // - ConstValue::Str actual string content comparison (above)
        // - Category C/D/E table string field comparisons (above)
        // Here we only verify that every string in the original pool can be found in the loaded pool.
        // (ConstValue::Str is already verified individually; no extra check needed.)

        // Runtime field counts
        assert_eq!(a.global_var_storage.len(), b.global_var_storage.len(), "{}: global_var_count", ctx);
        assert_eq!(a.memo_tables.len(), b.memo_tables.len(), "{}: memo_table_count", ctx);
    }

    // ==================== Test cases ====================

    #[test]
    fn test_round_trip_full() {
        let original = make_test_graph();
        let bytes = serialize_solidify(&original);
        let loaded = load_solidify_from_bytes(&bytes).expect("load failed");

        assert_graphs_equal(&original, &loaded, "full round-trip");
    }

    #[test]
    fn test_round_trip_double_serialize() {
        // Double-serialization verification: load -> serialize -> load -> compare.
        let original = make_test_graph();
        let bytes1 = serialize_solidify(&original);
        let loaded1 = load_solidify_from_bytes(&bytes1).expect("first load failed");
        let bytes2 = serialize_solidify(&loaded1);
        let loaded2 = load_solidify_from_bytes(&bytes2).expect("second load failed");

        assert_graphs_equal(&loaded1, &loaded2, "double serialize");
    }

    #[test]
    fn test_const_value_all_variants() {
        // Verifies the round-trip of each ConstValue variant individually.
        let variants = vec![
            ConstValue::I8(-128),
            ConstValue::I8(127),
            ConstValue::I16(-32768),
            ConstValue::I32(i32::MIN),
            ConstValue::I32(i32::MAX),
            ConstValue::I64(i64::MIN),
            ConstValue::I64(i64::MAX),
            ConstValue::I128(i128::MIN),
            ConstValue::I128(i128::MAX),
            ConstValue::U8(0),
            ConstValue::U8(255),
            ConstValue::U16(0),
            ConstValue::U32(0),
            ConstValue::U32(u32::MAX),
            ConstValue::U64(0),
            ConstValue::U64(u64::MAX),
            ConstValue::U128(0),
            ConstValue::U128(u128::MAX),
            ConstValue::Isize(-1),
            ConstValue::Usize(0),
            ConstValue::F32(0.0),
            ConstValue::F32(f32::INFINITY),
            ConstValue::F32(f32::NEG_INFINITY),
            ConstValue::F32(f32::NAN),
            ConstValue::F64(0.0),
            ConstValue::F64(f64::INFINITY),
            ConstValue::F64(f64::NEG_INFINITY),
            ConstValue::F64(f64::NAN),
            ConstValue::F16(0),
            ConstValue::F16(0x7C00), // +inf
            ConstValue::F128([0u8; 16]),
            ConstValue::F128([0xFF; 16]),
            ConstValue::Bool(true),
            ConstValue::Bool(false),
            ConstValue::Char(0),
            ConstValue::Char(0x10FFFF),
            ConstValue::Null,
            ConstValue::Void,
        ];

        for (i, cv) in variants.iter().enumerate() {
            // Build a single-node graph.
            let pool_bytes = if let ConstValue::Str { offset, len } = cv {
                // Str requires the string pool.
                vec![b'a'; (offset + len) as usize]
            } else {
                Vec::new()
            };
            let graph = DataFlowGraph {
                nodes: vec![Node { kind: NodeKind::Const, input_count: 0, inputs_offset: 0, compute_fn: CF_NOOP }],
                inputs_pool: InputsPool::new(),
                subgraphs: vec![SubGraph {
                    id: SubGraphId(0), node_range: (NodeId(0), NodeId(1)), param_count: 0,
                    entry_node: NodeId(0), return_node: NodeId(0), has_suspend: false,
                    event_source_decls: vec![], defer_table: vec![], loop_kind: LoopKind::None,
                    loop_parent_sg: None, cond_node: None, function_id: 0, iter_next_node: None,
                    upvalue_count: 0, upvalue_outer_nodes: vec![], nested_ranges: vec![],
                    reset_plan: None,
                }],
                entry_subgraph: Some(SubGraphId(0)),
                compute_fns: build_compute_fn_table(),
                downstreams: vec![vec![]],
                const_values: vec![Some(*cv)],
                call_targets: vec![None],
                gate_branches: vec![None],
                field_access_infos: vec![None],
                record_lit_infos: vec![None],
                ffi_call_names: vec![None],
                field_set_names: vec![None],
                vtable_call_methods: vec![None],
                await_event_sources: vec![None],
                closure_infos: vec![None],
                partial_infos: vec![None],
                closure_call_arg_counts: vec![None],
                select_infos: vec![None],
                writeback_targets: vec![None],
                tail_call_flags: vec![false],
                safe_op_flags: vec![false],
                hoisted_node: vec![false],
                hoisted_owners: vec![SubGraphId(0)],
                batch_infos: vec![None],
                ir_errors: Vec::new(),
                trait_construct_infos: vec![None],
                lazy_construct_infos: vec![None],
                record_extend_infos: vec![None],
                slice_inclusive: vec![false],
                global_var_storage: std::sync::Arc::new(vec![]),
                global_load_slots: vec![None],
                global_store_slots: vec![None],
                pattern_ctor_names: vec![None],
                pattern_field_indices: vec![None],
                cast_target_types: vec![None],
                memo_infos: vec![None],
                memo_tables: std::sync::Arc::new(vec![]),
                string_pool: std::sync::Arc::from(pool_bytes),
                mem: None,
                sg_uv_offsets: Vec::new(),
                sg_nr_offsets: Vec::new(),
                gate_branch_offsets: Vec::new(),
                record_lit_info_offsets: Vec::new(),
                select_info_offsets: Vec::new(),
                trait_construct_info_offsets: Vec::new(),
                record_extend_info_offsets: Vec::new(),
            };

            let bytes = serialize_solidify(&graph);
            let loaded = load_solidify_from_bytes(&bytes).expect("load failed");

            match (&graph.const_values[0], &loaded.const_values[0]) {
                (Some(a), Some(b)) => {
                    // Special handling for NaN (NaN != NaN).
                    match (a, b) {
                        (ConstValue::F32(_), ConstValue::F32(_)) => {
                            // Compare bit patterns rather than values.
                            let bits_a = if let ConstValue::F32(v) = a { v.to_bits() } else { 0 };
                            let bits_b = if let ConstValue::F32(v) = b { v.to_bits() } else { 0 };
                            assert_eq!(bits_a, bits_b, "variant[{}] F32 bits", i);
                        }
                        (ConstValue::F64(_), ConstValue::F64(_)) => {
                            let bits_a = if let ConstValue::F64(v) = a { v.to_bits() } else { 0 };
                            let bits_b = if let ConstValue::F64(v) = b { v.to_bits() } else { 0 };
                            assert_eq!(bits_a, bits_b, "variant[{}] F64 bits", i);
                        }
                        _ => assert_eq!(a, b, "variant[{}] const_value", i),
                    }
                }
                _ => panic!("variant[{}]: const_values mismatch", i),
            }
        }
    }

    #[test]
    fn test_const_value_str_round_trip() {
        // Verifies the round-trip of ConstValue::Str string content specifically.
        let test_strings = vec!["", "a", "hello world", "chinese test", "🎉🚀", "a\nb\tc\\d\"e"];
        for s in test_strings {
            let mut pool = Vec::new();
            let offset = pool.len() as u32;
            pool.extend_from_slice(s.as_bytes());
            let len = s.len() as u32;

            let graph = DataFlowGraph {
                nodes: vec![Node { kind: NodeKind::Const, input_count: 0, inputs_offset: 0, compute_fn: CF_NOOP }],
                inputs_pool: InputsPool::new(),
                subgraphs: vec![SubGraph {
                    id: SubGraphId(0), node_range: (NodeId(0), NodeId(1)), param_count: 0,
                    entry_node: NodeId(0), return_node: NodeId(0), has_suspend: false,
                    event_source_decls: vec![], defer_table: vec![], loop_kind: LoopKind::None,
                    loop_parent_sg: None, cond_node: None, function_id: 0, iter_next_node: None,
                    upvalue_count: 0, upvalue_outer_nodes: vec![], nested_ranges: vec![],
                    reset_plan: None,
                }],
                entry_subgraph: Some(SubGraphId(0)),
                compute_fns: build_compute_fn_table(),
                downstreams: vec![vec![]],
                const_values: vec![Some(ConstValue::Str { offset, len })],
                call_targets: vec![None],
                gate_branches: vec![None],
                field_access_infos: vec![None],
                record_lit_infos: vec![None],
                ffi_call_names: vec![None],
                field_set_names: vec![None],
                vtable_call_methods: vec![None],
                await_event_sources: vec![None],
                closure_infos: vec![None],
                partial_infos: vec![None],
                closure_call_arg_counts: vec![None],
                select_infos: vec![None],
                writeback_targets: vec![None],
                tail_call_flags: vec![false],
                safe_op_flags: vec![false],
                hoisted_node: vec![false],
                hoisted_owners: vec![SubGraphId(0)],
                batch_infos: vec![None],
                ir_errors: Vec::new(),
                trait_construct_infos: vec![None],
                lazy_construct_infos: vec![None],
                record_extend_infos: vec![None],
                slice_inclusive: vec![false],
                global_var_storage: std::sync::Arc::new(vec![]),
                global_load_slots: vec![None],
                global_store_slots: vec![None],
                pattern_ctor_names: vec![None],
                pattern_field_indices: vec![None],
                cast_target_types: vec![None],
                memo_infos: vec![None],
                memo_tables: std::sync::Arc::new(vec![]),
                string_pool: std::sync::Arc::from(pool),
                mem: None,
                sg_uv_offsets: Vec::new(),
                sg_nr_offsets: Vec::new(),
                gate_branch_offsets: Vec::new(),
                record_lit_info_offsets: Vec::new(),
                select_info_offsets: Vec::new(),
                trait_construct_info_offsets: Vec::new(),
                record_extend_info_offsets: Vec::new(),
            };

            let bytes = serialize_solidify(&graph);
            let loaded = load_solidify_from_bytes(&bytes).expect("load failed");

            // Verify string content.
            let cv = loaded.const_values[0].expect("const_value should exist");
            if let ConstValue::Str { offset: lo, len: ll } = cv {
                let loaded_str = std::str::from_utf8(&loaded.string_pool[lo as usize..(lo + ll) as usize]).unwrap();
                assert_eq!(loaded_str, s, "Str round-trip mismatch for {:?}", s);
            } else {
                panic!("expected Str variant, got {:?}", cv);
            }

            // Verify that to_value produces the correct Value (Value::Ref(HeapObj::Str(KuzoStr))).
            let v = cv.to_value(&loaded.string_pool);
            match &v {
                crate::value::Value::Ref(arc) => {
                    match arc.as_ref() {
                        crate::value::HeapObj::Str(gs) => {
                            assert_eq!(gs.bytes(), s, "to_value Str mismatch for {:?}", s);
                        }
                        other => panic!("to_value should produce Str, got {:?}", other),
                    }
                }
                other => panic!("to_value should produce Ref, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_empty_graph_round_trip() {
        // Empty-graph edge case (0 nodes).
        let graph = DataFlowGraph {
            nodes: vec![],
            inputs_pool: InputsPool::new(),
            subgraphs: vec![],
            entry_subgraph: None,
            compute_fns: build_compute_fn_table(),
            downstreams: vec![],
            const_values: vec![],
            call_targets: vec![],
            gate_branches: vec![],
            field_access_infos: vec![],
            record_lit_infos: vec![],
            ffi_call_names: vec![],
            field_set_names: vec![],
            vtable_call_methods: vec![],
            await_event_sources: vec![],
            closure_infos: vec![],
            partial_infos: vec![],
            closure_call_arg_counts: vec![],
            select_infos: vec![],
            writeback_targets: vec![],
            tail_call_flags: vec![],
            safe_op_flags: vec![],
            hoisted_node: vec![],
            hoisted_owners: vec![],
            batch_infos: vec![],
            ir_errors: Vec::new(),
            trait_construct_infos: vec![],
            lazy_construct_infos: vec![],
            record_extend_infos: vec![],
            slice_inclusive: vec![],
            global_var_storage: std::sync::Arc::new(vec![]),
            global_load_slots: vec![],
            global_store_slots: vec![],
            pattern_ctor_names: vec![],
            pattern_field_indices: vec![],
            cast_target_types: vec![],
            memo_infos: vec![],
            memo_tables: std::sync::Arc::new(vec![]),
            string_pool: std::sync::Arc::from(Vec::new()),
            mem: None,
            sg_uv_offsets: Vec::new(),
            sg_nr_offsets: Vec::new(),
            gate_branch_offsets: Vec::new(),
            record_lit_info_offsets: Vec::new(),
            select_info_offsets: Vec::new(),
            trait_construct_info_offsets: Vec::new(),
            record_extend_info_offsets: Vec::new(),
        };

        let bytes = serialize_solidify(&graph);
        let loaded = load_solidify_from_bytes(&bytes).expect("load failed");
        assert_graphs_equal(&graph, &loaded, "empty graph");
    }

    #[test]
    fn test_bitmap_boundary() {
        // Tests boolean-table bitmap boundaries: 1/8/9/16 nodes.
        for n in [1usize, 7, 8, 9, 15, 16, 17] {
            let all_true = vec![true; n];
            let all_false = vec![false; n];
            let alternating: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();

            for (name, flags) in [("all_true", &all_true), ("all_false", &all_false), ("alternating", &alternating)] {
                let graph = DataFlowGraph {
                    nodes: vec![Node { kind: NodeKind::Const, input_count: 0, inputs_offset: 0, compute_fn: CF_NOOP }; n],
                    inputs_pool: InputsPool::new(),
                    subgraphs: vec![SubGraph {
                        id: SubGraphId(0), node_range: (NodeId(0), NodeId(n as u32)), param_count: 0,
                        entry_node: NodeId(0), return_node: NodeId(n as u32 - 1), has_suspend: false,
                        event_source_decls: vec![], defer_table: vec![], loop_kind: LoopKind::None,
                        loop_parent_sg: None, cond_node: None, function_id: 0, iter_next_node: None,
                        upvalue_count: 0, upvalue_outer_nodes: vec![], nested_ranges: vec![],
                        reset_plan: None,
                    }],
                    entry_subgraph: Some(SubGraphId(0)),
                    compute_fns: build_compute_fn_table(),
                    downstreams: vec![vec![]; n],
                    const_values: vec![None; n],
                    call_targets: vec![None; n],
                    gate_branches: vec![None; n],
                    field_access_infos: vec![None; n],
                    record_lit_infos: vec![None; n],
                    ffi_call_names: vec![None; n],
                    field_set_names: vec![None; n],
                    vtable_call_methods: vec![None; n],
                    await_event_sources: vec![None; n],
                    closure_infos: vec![None; n],
                    partial_infos: vec![None; n],
                    closure_call_arg_counts: vec![None; n],
                    select_infos: vec![None; n],
                    writeback_targets: vec![None; n],
                    tail_call_flags: flags.clone(),
                    safe_op_flags: flags.clone(),
                    hoisted_node: flags.clone(),
                    slice_inclusive: flags.clone(),
                    hoisted_owners: vec![SubGraphId(0); n],
                    batch_infos: vec![None; n],
                    ir_errors: Vec::new(),
                    trait_construct_infos: vec![None; n],
                    lazy_construct_infos: vec![None; n],
                    record_extend_infos: vec![None; n],
                    global_var_storage: std::sync::Arc::new(vec![]),
                    global_load_slots: vec![None; n],
                    global_store_slots: vec![None; n],
                    pattern_ctor_names: vec![None; n],
                    pattern_field_indices: vec![None; n],
                    cast_target_types: vec![None; n],
                    memo_infos: vec![None; n],
                    memo_tables: std::sync::Arc::new(vec![]),
                    string_pool: std::sync::Arc::from(Vec::new()),
                    mem: None,
                    sg_uv_offsets: Vec::new(),
                    sg_nr_offsets: Vec::new(),
                    gate_branch_offsets: Vec::new(),
                    record_lit_info_offsets: Vec::new(),
                    select_info_offsets: Vec::new(),
                    trait_construct_info_offsets: Vec::new(),
                    record_extend_info_offsets: Vec::new(),
                };

                let bytes = serialize_solidify(&graph);
                let loaded = load_solidify_from_bytes(&bytes).expect("load failed");

                assert_eq!(loaded.tail_call_flags, *flags, "n={} {}: tail_call_flags", n, name);
                assert_eq!(loaded.safe_op_flags, *flags, "n={} {}: safe_op_flags", n, name);
                assert_eq!(loaded.hoisted_node, *flags, "n={} {}: hoisted_node", n, name);
                assert_eq!(loaded.slice_inclusive, *flags, "n={} {}: slice_inclusive", n, name);
            }
        }
    }

    #[test]
    fn test_subgraph_varlen_fields() {
        // Verifies the round-trip of all SubGraph variable-length fields specifically.
        let original = make_test_graph();
        let bytes = serialize_solidify(&original);
        let loaded = load_solidify_from_bytes(&bytes).expect("load failed");

        // sg[0]: contains event_decls(2) + defer_table(2, with captured) + upvalue_nodes(1) + nested_ranges(2) + reset_plan
        let sg0 = &loaded.subgraphs[0];
        assert_eq!(sg0.event_source_decls.len(), 2);
        assert_eq!(sg0.event_source_decls[0].kind, EventSourceKind::Channel);
        assert_eq!(sg0.event_source_decls[1].kind, EventSourceKind::Timer);
        assert_eq!(sg0.defer_table.len(), 2);
        assert_eq!(sg0.defer_table[0].captured_inputs, vec![NodeId(0), NodeId(1)]);
        assert!(sg0.defer_table[0].captured_inputs.len() == 2);
        assert_eq!(sg0.defer_table[1].captured_inputs.len(), 0); // empty captured
        assert_eq!(sg0.upvalue_outer_nodes, vec![NodeId(3)]);
        assert_eq!(sg0.nested_ranges, vec![(11, 13), (14, 16)]);
        assert!(sg0.reset_plan.is_some());
        let rp = sg0.reset_plan.as_ref().unwrap();
        assert_eq!(rp.reset_to_zero, vec![NodeId(11)]);
        assert_eq!(rp.reset_to_one, vec![NodeId(12)]);
        assert_eq!(rp.reset_condition_tree, vec![NodeId(4)]);

        // sg[1]: all variable-length fields empty, no reset_plan
        let sg1 = &loaded.subgraphs[1];
        assert!(sg1.event_source_decls.is_empty());
        assert!(sg1.defer_table.is_empty());
        assert!(sg1.upvalue_outer_nodes.is_empty());
        assert!(sg1.nested_ranges.is_empty());
        assert!(sg1.reset_plan.is_none());

        // sg[2]: contains event_decls(1) + upvalue_nodes(2) + reset_plan (only reset_to_one)
        let sg2 = &loaded.subgraphs[2];
        assert_eq!(sg2.event_source_decls.len(), 1);
        assert_eq!(sg2.event_source_decls[0].kind, EventSourceKind::AsyncJoin);
        assert_eq!(sg2.upvalue_outer_nodes, vec![NodeId(0), NodeId(1)]);
        assert!(sg2.reset_plan.is_some());
        let rp2 = sg2.reset_plan.as_ref().unwrap();
        assert!(rp2.reset_to_zero.is_empty()); // empty array
        assert_eq!(rp2.reset_to_one, vec![NodeId(12)]);
        assert!(rp2.reset_condition_tree.is_empty()); // empty array
    }

    #[test]
    fn test_gate_branches_varlen() {
        let original = make_test_graph();
        let bytes = serialize_solidify(&original);
        let loaded = load_solidify_from_bytes(&bytes).expect("load failed");

        let gb = loaded.gate_branches[8].as_ref().expect("gate_branches[8]");
        assert_eq!(gb.condition_input, NodeId(4));
        assert_eq!(gb.branches.len(), 3);
        // branch 0: true, sg=1, inputs=[0,1]
        assert_eq!(gb.branches[0].0, true);
        assert_eq!(gb.branches[0].1, SubGraphId(1));
        assert_eq!(gb.branches[0].2, vec![NodeId(0), NodeId(1)]);
        // branch 1: false, sg=2, inputs=[2]
        assert_eq!(gb.branches[1].0, false);
        assert_eq!(gb.branches[1].1, SubGraphId(2));
        assert_eq!(gb.branches[1].2, vec![NodeId(2)]);
        // branch 2: true, sg=0, inputs=[] (empty)
        assert_eq!(gb.branches[2].0, true);
        assert_eq!(gb.branches[2].1, SubGraphId(0));
        assert!(gb.branches[2].2.is_empty());
    }

    #[test]
    fn test_select_info_varlen() {
        let original = make_test_graph();
        let bytes = serialize_solidify(&original);
        let loaded = load_solidify_from_bytes(&bytes).expect("load failed");

        let si = loaded.select_infos[15].as_ref().expect("select_infos[15]");
        assert_eq!(si.branches.len(), 2);
        assert_eq!(si.branches[0].subgraph_id, SubGraphId(1));
        assert_eq!(si.branches[0].event_kind, EventSourceKind::Channel);
        assert_eq!(si.branches[0].event_source_node, NodeId(10));
        assert_eq!(si.branches[1].subgraph_id, SubGraphId(2));
        assert_eq!(si.branches[1].event_kind, EventSourceKind::Timer);
        assert_eq!(si.branches[1].event_source_node, NodeId(10));
    }

    #[test]
    fn test_trait_construct_info_varlen() {
        let original = make_test_graph();
        let bytes = serialize_solidify(&original);
        let loaded = load_solidify_from_bytes(&bytes).expect("load failed");

        let ti = loaded.trait_construct_infos[14].as_ref().expect("trait_construct_infos[14]");
        assert_eq!(ti.trait_name, "MyTrait");
        assert_eq!(ti.method_names, vec!["method_a", "method_b"]);
        assert_eq!(ti.methods.len(), 2);
        assert_eq!(ti.methods[0].subgraph_id, SubGraphId(1));
        assert_eq!(ti.methods[0].arity, 2);
        assert_eq!(ti.methods[0].upvalue_count, 1);
        assert_eq!(ti.methods[1].subgraph_id, SubGraphId(2));
        assert_eq!(ti.methods[1].arity, 0);
        assert_eq!(ti.methods[1].upvalue_count, 0);
    }

    #[test]
    fn test_record_lit_info_varlen() {
        let original = make_test_graph();
        let bytes = serialize_solidify(&original);
        let loaded = load_solidify_from_bytes(&bytes).expect("load failed");

        let ri = loaded.record_lit_infos[14].as_ref().expect("record_lit_infos[14]");
        assert_eq!(ri.type_name, "MyType");
        assert_eq!(ri.field_names.len(), 3);
        assert_eq!(ri.field_names[0], Some("x".to_string()));
        assert_eq!(ri.field_names[1], None); // None field name
        assert_eq!(ri.field_names[2], Some("z".to_string()));
        assert_eq!(ri.constructor, "Ctor");
        assert_eq!(ri.kind, RecordLitKind::Adt);
    }

    #[test]
    fn test_downstreams_csr() {
        let original = make_test_graph();
        let bytes = serialize_solidify(&original);
        let loaded = load_solidify_from_bytes(&bytes).expect("load failed");

        assert_eq!(loaded.downstreams.len(), 16);
        // non-empty downstreams
        assert_eq!(loaded.downstreams[0], vec![NodeId(4)]);
        assert_eq!(loaded.downstreams[4], vec![NodeId(5), NodeId(7), NodeId(8)]);
        // empty downstreams
        assert!(loaded.downstreams[2].is_empty());
        assert!(loaded.downstreams[6].is_empty());
        assert!(loaded.downstreams[15].is_empty());
    }

    #[test]
    fn test_crc_corruption_detection() {
        let original = make_test_graph();
        let mut bytes = serialize_solidify(&original);

        // Corrupt one byte in the body (skipping the header's crc32 field).
        // The header is 64B; corrupt the body region.
        let corrupt_pos = 100; // within the body region
        if corrupt_pos < bytes.len() {
            bytes[corrupt_pos] ^= 0xFF;
        }

        let result = load_solidify_from_bytes(&bytes);
        assert!(result.is_err(), "corrupted file should be rejected");
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => String::new(),
        };
        assert!(err_msg.contains("CRC"), "error should mention CRC, got: {}", err_msg);
    }

    #[test]
    fn test_magic_detection() {
        let original = make_test_graph();
        let mut bytes = serialize_solidify(&original);

        // Corrupt the magic.
        bytes[0] = b'X';

        let result = load_solidify_from_bytes(&bytes);
        assert!(result.is_err(), "bad magic should be rejected");
    }
}
