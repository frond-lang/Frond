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



// ==================== Serialization ====================

/// Serializes a `DataFlowGraph` into a `.kzo` byte stream.
pub fn serialize_solidify(graph: &DataFlowGraph) -> Vec<u8> {
    let n = graph.nodes.len();
    let mut string_pool = StringPool::new();

    // ---- Collect bytes for each section ----
    let mut sections: Vec<(SectionKind, Vec<u8>)> = Vec::new();

    // 1. Nodes (v2 packed): kind u8 + input_count u8 + compute_fn u16 = 4B.
    //    When the inputs pool is contiguous in node-id order (guaranteed after
    //    optimizer `rebuild`; builder layout is contiguous by construction),
    //    per-node inputs_offset is derived at load (header flag bit0) — else
    //    8B records carry it explicitly.
    let inputs_contiguous = {
        let mut expected = 0u32;
        graph.nodes.iter().all(|nd| {
            let ok = nd.inputs_offset == expected;
            expected += nd.input_count as u32;
            ok
        })
    };
    {
        let mut buf = Vec::with_capacity(n * 4);
        for node in &graph.nodes {
            write_u8(&mut buf, node_kind_to_u8(node.kind));
            write_u8(&mut buf, node.input_count);
            write_u16(&mut buf, node.compute_fn.0 as u16);
            if !inputs_contiguous {
                write_u32(&mut buf, node.inputs_offset);
            }
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

    // 3. SubGraphs + variable-length regions (v3 packed layout).
    //    Per-sg fixed record ~45B: `id` elided (== vector index); boolean
    //    fields merged into one flags byte; the reset-plan region uses
    //    explicit slice starts (plans can be large) while the other four
    //    regions append in sg order with per-sg u16 lengths only.
    //    nested_ranges are DERIVED at load (compute_nested_ranges) and are
    //    not serialized.
    {
        let mut sg_buf = Vec::new();
        let mut upvalue_nodes_buf = Vec::new();
        let mut event_decls_buf = Vec::new();
        let mut defer_entries_buf = Vec::new();
        let mut defer_captured_buf = Vec::new();
        let mut reset_plan_buf = Vec::new();

        for sg in &graph.subgraphs {
            let loop_kind = loop_kind_to_u8(sg.loop_kind);
            debug_assert!(loop_kind <= 0b1111);
            let mut flags: u8 = sg.has_suspend as u8;
            flags |= (sg.reset_plan.is_some() as u8) << 1;
            flags |= loop_kind << 4;

            write_u32(&mut sg_buf, sg.node_range.0.0);
            write_u32(&mut sg_buf, sg.node_range.1.0);
            write_u8(&mut sg_buf, sg.param_count);
            write_u32(&mut sg_buf, sg.entry_node.0);
            write_u32(&mut sg_buf, sg.return_node.0);
            write_u8(&mut sg_buf, flags);
            write_u8(&mut sg_buf, sg.upvalue_count);
            write_u32(&mut sg_buf, sg.loop_parent_sg.map(|s| s.0).unwrap_or(u32::MAX));
            write_u32(&mut sg_buf, sg.cond_node.map(|s| s.0).unwrap_or(u32::MAX));
            write_u32(&mut sg_buf, sg.function_id);
            write_u32(&mut sg_buf, sg.iter_next_node.map(|s| s.0).unwrap_or(u32::MAX));
            // Per-sg variable-region lengths (offsets implicit, append order).
            debug_assert!(sg.upvalue_outer_nodes.len() <= u16::MAX as usize);
            debug_assert!(sg.event_source_decls.len() <= u16::MAX as usize);
            debug_assert!(sg.defer_table.len() <= u16::MAX as usize);
            write_u16(&mut sg_buf, sg.upvalue_outer_nodes.len() as u16);
            write_u16(&mut sg_buf, sg.event_source_decls.len() as u16);
            write_u16(&mut sg_buf, sg.defer_table.len() as u16);
            write_u32(&mut sg_buf, reset_plan_buf.len() as u32); // reset-plan slice start

            // upvalue_outer_nodes
            for nid in &sg.upvalue_outer_nodes {
                write_u32(&mut upvalue_nodes_buf, nid.0);
            }

            // event_source_decls
            for decl in &sg.event_source_decls {
                write_u32(&mut event_decls_buf, decl.node.0);
                write_u8(&mut event_decls_buf, event_kind_to_u8(decl.kind));
                write_u8(&mut event_decls_buf, 0); // padding to 8B stride
                write_u8(&mut event_decls_buf, 0);
                write_u8(&mut event_decls_buf, 0);
            }

            // defer_table (10B/entry: trigger + body_sg + captured count u16;
            // captured ids are appended to defer_captured_buf in entry order)
            for de in &sg.defer_table {
                debug_assert!(de.captured_inputs.len() <= u16::MAX as usize);
                write_u32(&mut defer_entries_buf, de.trigger_node.0);
                write_u32(&mut defer_entries_buf, de.body_subgraph.0);
                write_u16(&mut defer_entries_buf, de.captured_inputs.len() as u16);
                // `registered` is runtime state and is not serialized.
                for nid in &de.captured_inputs {
                    write_u32(&mut defer_captured_buf, nid.0);
                }
            }

            // reset_plan region slice [start .. next sg's start)
            if let Some(rp) = &sg.reset_plan {
                write_u32(&mut reset_plan_buf, rp.reset_to_zero.len() as u32);
                for nid in &rp.reset_to_zero { write_u32(&mut reset_plan_buf, nid.0); }
                write_u32(&mut reset_plan_buf, rp.reset_to_one.len() as u32);
                for nid in &rp.reset_to_one { write_u32(&mut reset_plan_buf, nid.0); }
                write_u32(&mut reset_plan_buf, rp.reset_condition_tree.len() as u32);
                for nid in &rp.reset_condition_tree { write_u32(&mut reset_plan_buf, nid.0); }
            }
        }

        sections.push((SectionKind::SubGraphs, sg_buf));
        sections.push((SectionKind::SgUpvalueNodes, upvalue_nodes_buf));
        sections.push((SectionKind::SgEventDecls, event_decls_buf));
        sections.push((SectionKind::SgDeferEntries, defer_entries_buf));
        sections.push((SectionKind::SgDeferCapturedInputs, defer_captured_buf));
        sections.push((SectionKind::SgResetPlan, reset_plan_buf));
    }

    // ---- Sparse per-Node tables (v2: categories A/C/D/E) ----
    // Uniform layout: [count u32][ (node_idx u32, blob_off u32) * count ][blob].
    // Only present (Some) rows are stored; entries sorted by node_idx
    // (ascending, by construction). Dropped entirely: HoistedOwners /
    // HoistedNode (no runtime consumers) and Downstreams (derived at load
    // from inputs + gate condition edges).
    macro_rules! push_sparse_entry {
        ($index:expr, $blob:expr, $count:expr, $i:expr, $write_payload:expr) => {{
            write_u32(&mut $index, $i as u32);
            write_u32(&mut $index, $blob.len() as u32);
            $write_payload(&mut $blob);
            $count += 1;
        }};
    }
    macro_rules! finish_sparse {
        ($sections:expr, $kind:expr, $index:expr, $blob:expr, $count:expr) => {{
            let mut buf = Vec::with_capacity(4 + $index.len() + $blob.len());
            write_u32(&mut buf, $count);
            buf.extend_from_slice(&$index);
            buf.extend_from_slice(&$blob);
            $sections.push(($kind, buf));
        }};
    }

    // Category A: fixed-width scalar Option tables.
    macro_rules! ser_sparse_opt {
        ($sections:expr, $graph:expr, $field:ident, $kind:ident, |$b:ident, $v:ident| $write_payload:expr) => {{
            let mut index: Vec<u8> = Vec::new();
            let mut blob: Vec<u8> = Vec::new();
            let mut count = 0u32;
            for (i, opt) in $graph.$field.iter().enumerate() {
                if let Some($v) = opt {
                    write_u32(&mut index, i as u32);
                    write_u32(&mut index, blob.len() as u32);
                    let $b = &mut blob;
                    $write_payload;
                    count += 1;
                }
            }
            finish_sparse!($sections, SectionKind::$kind, index, blob, count);
        }};
    }
    ser_sparse_opt!(sections, graph, call_targets, CallTargets, |b, v| write_u32(b, v.0));
    ser_sparse_opt!(sections, graph, field_access_infos, FieldAccessInfos, |b, v| write_u16(b, *v));
    ser_sparse_opt!(sections, graph, vtable_call_methods, VtableCallMethods, |b, v| write_u16(b, *v));
    ser_sparse_opt!(sections, graph, await_event_sources, AwaitEventSources, |b, v| write_u32(b, v.0));
    ser_sparse_opt!(sections, graph, writeback_targets, WritebackTargets, |b, v| write_u32(b, v.0));
    ser_sparse_opt!(sections, graph, global_load_slots, GlobalLoadSlots, |b, v| write_u32(b, *v));
    ser_sparse_opt!(sections, graph, global_store_slots, GlobalStoreSlots, |b, v| write_u32(b, *v));
    ser_sparse_opt!(sections, graph, pattern_field_indices, PatternFieldIndices, |b, v| write_u16(b, *v));
    ser_sparse_opt!(sections, graph, closure_call_arg_counts, ClosureCallArgCounts, |b, v| write_u8(b, *v));
    ser_sparse_opt!(sections, graph, lib_ret_kinds, LibRetKinds, |b, v| write_u8(b, *v));
    ser_sparse_opt!(sections, graph, embed_infos, EmbedInfos, |b, v| write_u32(b, *v));

    // Lib.embed resources (v4): self-contained section
    // `[count u32]{ name_len u32, name bytes, data_len u32, data bytes }`.
    {
        let mut body: Vec<u8> = Vec::new();
        write_u32(&mut body, graph.resources.len() as u32);
        for (name, data) in &graph.resources {
            write_u32(&mut body, name.len() as u32);
            body.extend_from_slice(name.as_bytes());
            write_u32(&mut body, data.len() as u32);
            body.extend_from_slice(data);
        }
        sections.push((SectionKind::Resources, body));
    }

    // ---- per-Node boolean tables (category B; dense bitmaps stay) ----
    ser_bool_table!(sections, n, graph, tail_call_flags, TailCallFlags);
    ser_bool_table!(sections, n, graph, safe_op_flags, SafeOpFlags);
    ser_bool_table!(sections, n, graph, slice_inclusive, SliceInclusive);

    // Category C: string tables (payload = StrRef into the string pool).
    macro_rules! ser_sparse_str {
        ($sections:expr, $graph:expr, $field:ident, $kind:ident, $pool:expr) => {{
            let mut index: Vec<u8> = Vec::new();
            let mut blob: Vec<u8> = Vec::new();
            let mut count = 0u32;
            for (i, opt) in $graph.$field.iter().enumerate() {
                if let Some(s) = opt {
                    let (off, len) = $pool.add(s);
                    push_sparse_entry!(index, blob, count, i, |b: &mut Vec<u8>| {
                        write_u32(b, off);
                        write_u32(b, len);
                    });
                }
            }
            finish_sparse!($sections, SectionKind::$kind, index, blob, count);
        }};
    }
    ser_sparse_str!(sections, graph, ffi_call_names, FfiCallNames, string_pool);
    ser_sparse_str!(sections, graph, field_set_names, FieldSetNames, string_pool);
    ser_sparse_str!(sections, graph, pattern_ctor_names, PatternCtorNames, string_pool);
    ser_sparse_str!(sections, graph, pattern_type_names, PatternTypeNames, string_pool);
    ser_sparse_str!(sections, graph, cast_target_types, CastTargetTypes, string_pool);

    // Category D: fixed-width composites.
    {
        let mut index: Vec<u8> = Vec::new();
        let mut blob: Vec<u8> = Vec::new();
        let mut count = 0u32;
        for (i, opt) in graph.closure_infos.iter().enumerate() {
            if let Some(ci) = opt {
                push_sparse_entry!(index, blob, count, i, |b: &mut Vec<u8>| {
                    write_u32(b, ci.subgraph_id.0);
                    write_u8(b, ci.arity);
                    write_i32(b, ci.self_upvalue_idx);
                });
            }
        }
        finish_sparse!(sections, SectionKind::ClosureInfos, index, blob, count);
    }
    {
        let mut index: Vec<u8> = Vec::new();
        let mut blob: Vec<u8> = Vec::new();
        let mut count = 0u32;
        for (i, opt) in graph.partial_infos.iter().enumerate() {
            if let Some(pi) = opt {
                push_sparse_entry!(index, blob, count, i, |b: &mut Vec<u8>| {
                    write_u32(b, pi.subgraph_id.0);
                    write_u8(b, pi.bound_count);
                });
            }
        }
        finish_sparse!(sections, SectionKind::PartialInfos, index, blob, count);
    }
    {
        let mut index: Vec<u8> = Vec::new();
        let mut blob: Vec<u8> = Vec::new();
        let mut count = 0u32;
        for (i, opt) in graph.lazy_construct_infos.iter().enumerate() {
            if let Some(li) = opt {
                push_sparse_entry!(index, blob, count, i, |b: &mut Vec<u8>| {
                    write_u32(b, li.thunk_sg.0);
                });
            }
        }
        finish_sparse!(sections, SectionKind::LazyConstructInfos, index, blob, count);
    }
    {
        let mut index: Vec<u8> = Vec::new();
        let mut blob: Vec<u8> = Vec::new();
        let mut count = 0u32;
        for (i, opt) in graph.memo_infos.iter().enumerate() {
            if let Some(mi) = opt {
                push_sparse_entry!(index, blob, count, i, |b: &mut Vec<u8>| {
                    write_u32(b, mi.table_index);
                    write_u8(b, mi.param_count);
                });
            }
        }
        finish_sparse!(sections, SectionKind::MemoInfos, index, blob, count);
    }

    // Category E: variable-length tables. Blob entries keep the v1
    // Some-branch encoding (validity byte included — for GateBranches it
    // doubles as the W4c capture flag), so the on-demand parsers are
    // unchanged.
    {
        let mut index: Vec<u8> = Vec::new();
        let mut blob: Vec<u8> = Vec::new();
        let mut count = 0u32;
        for (i, opt) in graph.const_values.iter().enumerate() {
            if let Some(cv) = opt {
                push_sparse_entry!(index, blob, count, i, |b: &mut Vec<u8>| {
                    write_u8(b, const_tag_to_u8(cv));
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
                        ConstValue::Bool(v) => payload[0] = *v as u8,
                        ConstValue::Char(c) => payload[0..4].copy_from_slice(&c.to_le_bytes()),
                        ConstValue::Str { offset, len } => {
                            // Read the actual string and re-intern into the
                            // serializer-side pool (offsets may differ).
                            let off = *offset as usize;
                            let end = off + *len as usize;
                            let s = std::str::from_utf8(&graph.string_pool[off..end]).unwrap_or("");
                            let (new_off, new_len) = string_pool.add(s);
                            payload[0..4].copy_from_slice(&new_off.to_le_bytes());
                            payload[4..8].copy_from_slice(&new_len.to_le_bytes());
                        }
                        ConstValue::Null | ConstValue::Void => {}
                    }
                    // v3: variable-width payload (bool 1B, i32 4B ... i128 16B).
                    let w = const_payload_len(const_tag_to_u8(cv));
                    write_bytes(b, &payload[..w]);
                });
            }
        }
        finish_sparse!(sections, SectionKind::ConstValues, index, blob, count);
    }
    // GateBranches
    {
        let mut index: Vec<u8> = Vec::new();
        let mut blob: Vec<u8> = Vec::new();
        let mut count = 0u32;
        for (i, opt) in graph.gate_branches.iter().enumerate() {
            if let Some(gb) = opt {
                push_sparse_entry!(index, blob, count, i, |b: &mut Vec<u8>| {
                    // Validity byte doubles as the W4c capture flag carrier:
                    // 2 = valid + capture.
                    write_u8(b, if gb.capture { 2 } else { 1 });
                    write_u32(b, gb.condition_input.0);
                    write_u32(b, gb.branches.len() as u32);
                    for (cond, sg, inputs) in &gb.branches {
                        write_u8(b, *cond as u8);
                        write_u32(b, sg.0);
                        write_u32(b, inputs.len() as u32);
                        for nid in inputs { write_u32(b, nid.0); }
                    }
                });
            }
        }
        finish_sparse!(sections, SectionKind::GateBranches, index, blob, count);
    }
    // RecordLitInfos
    {
        let mut index: Vec<u8> = Vec::new();
        let mut blob: Vec<u8> = Vec::new();
        let mut count = 0u32;
        for (i, opt) in graph.record_lit_infos.iter().enumerate() {
            if let Some(ri) = opt {
                push_sparse_entry!(index, blob, count, i, |b: &mut Vec<u8>| {
                    write_u8(b, 1);
                    let (off, len) = string_pool.add(&ri.type_name);
                    write_u32(b, off); write_u32(b, len);
                    write_u32(b, ri.field_names.len() as u32);
                    for fn_opt in &ri.field_names {
                        let (fo, fl) = string_pool.add_opt(fn_opt);
                        write_u32(b, fo); write_u32(b, fl);
                    }
                    let (co, cl) = string_pool.add(&ri.constructor);
                    write_u32(b, co); write_u32(b, cl);
                    write_u8(b, record_lit_kind_to_u8(ri.kind));
                });
            }
        }
        finish_sparse!(sections, SectionKind::RecordLitInfos, index, blob, count);
    }
    // SelectInfos
    {
        let mut index: Vec<u8> = Vec::new();
        let mut blob: Vec<u8> = Vec::new();
        let mut count = 0u32;
        for (i, opt) in graph.select_infos.iter().enumerate() {
            if let Some(si) = opt {
                push_sparse_entry!(index, blob, count, i, |b: &mut Vec<u8>| {
                    write_u8(b, 1);
                    write_u32(b, si.branches.len() as u32);
                    for br in &si.branches {
                        write_u32(b, br.subgraph_id.0);
                        write_u8(b, event_kind_to_u8(br.event_kind));
                        write_u32(b, br.event_source_node.0);
                    }
                });
            }
        }
        finish_sparse!(sections, SectionKind::SelectInfos, index, blob, count);
    }
    // TraitConstructInfos
    {
        let mut index: Vec<u8> = Vec::new();
        let mut blob: Vec<u8> = Vec::new();
        let mut count = 0u32;
        for (i, opt) in graph.trait_construct_infos.iter().enumerate() {
            if let Some(ti) = opt {
                push_sparse_entry!(index, blob, count, i, |b: &mut Vec<u8>| {
                    write_u8(b, 1);
                    let (off, len) = string_pool.add(&ti.trait_name);
                    write_u32(b, off); write_u32(b, len);
                    write_u32(b, ti.method_names.len() as u32);
                    for mn in &ti.method_names {
                        let (mo, ml) = string_pool.add(mn);
                        write_u32(b, mo); write_u32(b, ml);
                    }
                    write_u32(b, ti.methods.len() as u32);
                    for m in &ti.methods {
                        write_bytes(b, &trait_method_entry_to_bytes(m));
                    }
                });
            }
        }
        finish_sparse!(sections, SectionKind::TraitConstructInfos, index, blob, count);
    }
    // RecordExtendInfos
    {
        let mut index: Vec<u8> = Vec::new();
        let mut blob: Vec<u8> = Vec::new();
        let mut count = 0u32;
        for (i, opt) in graph.record_extend_infos.iter().enumerate() {
            if let Some(ri) = opt {
                push_sparse_entry!(index, blob, count, i, |b: &mut Vec<u8>| {
                    write_u8(b, 1);
                    write_u32(b, ri.update_names.len() as u32);
                    for un in &ri.update_names {
                        let (uo, ul) = string_pool.add(un);
                        write_u32(b, uo); write_u32(b, ul);
                    }
                });
            }
        }
        finish_sparse!(sections, SectionKind::RecordExtendInfos, index, blob, count);
    }
    // BatchInfos
    {
        let mut index: Vec<u8> = Vec::new();
        let mut blob: Vec<u8> = Vec::new();
        let mut count = 0u32;
        for (i, opt) in graph.batch_infos.iter().enumerate() {
            if let Some(bi) = opt {
                push_sparse_entry!(index, blob, count, i, |b: &mut Vec<u8>| {
                    write_u8(b, 1);
                    write_bytes(b, &batch_info_to_bytes(bi));
                });
            }
        }
        finish_sparse!(sections, SectionKind::BatchInfos, index, blob, count);
    }

    // DynFfiInfos (v2: closes the v1 gap — `kuzo run <file>.kzo` previously
    // panicked with "no dyn_ffi_info" since these were never serialized).
    {
        let mut index: Vec<u8> = Vec::new();
        let mut blob: Vec<u8> = Vec::new();
        let mut count = 0u32;
        for (i, opt) in graph.dyn_ffi_infos.iter().enumerate() {
            if let Some(di) = opt {
                let (so, sl) = string_pool.add(&di.symbol);
                push_sparse_entry!(index, blob, count, i, |b: &mut Vec<u8>| {
                    write_u32(b, so);
                    write_u32(b, sl);
                    write_u8(b, di.arg_count);
                    write_u8(b, di.sig.params.len() as u8);
                    for p in &di.sig.params {
                        write_abi_type(b, p);
                    }
                    write_abi_type(b, &di.sig.ret);
                });
            }
        }
        finish_sparse!(sections, SectionKind::DynFfiInfos, index, blob, count);
    }

    // ---- String Pool ----
    {
        sections.push((SectionKind::StringPool, string_pool.data.clone()));
    }
    // v2: no Downstreams section — derived at load (compute_downstream_csr).

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
        flags: if inputs_contiguous { FLAG_NODE_INPUT_OFFSETS_ELIDED } else { 0 },
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

/// Parses one ConstValue payload (tag u8 + 16B payload; v1 dense and v2 sparse
/// blob share the encoding).
pub(crate) fn parse_const_value(tag: u8, payload: &[u8]) -> ConstValue {
    match tag {
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
            // (offset, len) references the string pool directly.
            let off = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
            let len = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
            ConstValue::Str { offset: off, len }
        }
        20 => ConstValue::Null,
        21 => ConstValue::Void,
        _ => panic!("invalid ConstTag: {}", tag),
    }
}

/// Parses one DynFfiInfo payload from a v2 sparse DynFfiInfos blob entry.
pub(crate) fn parse_dyn_ffi_info(r: &[u8], mem: &GraphMemory) -> DynFfiInfo {
    let mut r = r;
    let so = read_u32(&mut r);
    let sl = read_u32(&mut r);
    let arg_count = read_u8(&mut r);
    let param_count = read_u8(&mut r) as usize;
    let mut params = Vec::with_capacity(param_count);
    for _ in 0..param_count {
        params.push(read_abi_type(&mut r));
    }
    let ret = read_abi_type(&mut r);
    DynFfiInfo {
        symbol: mem.read_str(so, sl),
        sig: crate::ffi::Abi::AbiSig::new(params, ret),
        arg_count,
    }
}

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

/// Parses the v4 Resources section (self-contained
/// `[count u32]{ name_len u32, name bytes, data_len u32, data bytes }`).
/// Resources are small relative to the graph, so both load paths materialize
/// them into the owned Vec (no mmap accessor indirection).
fn parse_resources_section(mem: &GraphMemory) -> Vec<(Arc<str>, Arc<[u8]>)> {
    let r = mem.section(SectionKind::Resources);
    let count = u32::from_le_bytes([r[0], r[1], r[2], r[3]]) as usize;
    let mut out = Vec::with_capacity(count);
    let mut p = 4usize;
    for _ in 0..count {
        let nl = u32::from_le_bytes([r[p], r[p + 1], r[p + 2], r[p + 3]]) as usize;
        p += 4;
        let name = String::from_utf8_lossy(&r[p..p + nl]).into_owned();
        p += nl;
        let dl = u32::from_le_bytes([r[p], r[p + 1], r[p + 2], r[p + 3]]) as usize;
        p += 4;
        let data: Arc<[u8]> = Arc::from(r[p..p + dl].to_vec());
        p += dl;
        out.push((Arc::from(name), data));
    }
    out
}

/// Rebuilds an owned `DataFlowGraph` from a `GraphMemory` (shared parsing logic).
fn load_from_graph_memory(mem: &GraphMemory) -> io::Result<DataFlowGraph> {
    let n = mem.header().node_count as usize;
    let offsets_elided = mem.header().flags & FLAG_NODE_INPUT_OFFSETS_ELIDED != 0;
    let node_stride = if offsets_elided { 4 } else { 8 };

    // 1. Nodes (v2 packed)
    let nodes = {
        let nr = mem.section(SectionKind::Nodes);
        let mut nodes = Vec::with_capacity(n);
        let mut running_offset = 0u32;
        for i in 0..n {
            let base = i * node_stride;
            let kind = u8_to_node_kind(nr[base]);
            let input_count = nr[base + 1];
            let compute_fn = ComputeFnId(u16::from_le_bytes([nr[base + 2], nr[base + 3]]) as u32);
            let inputs_offset = if offsets_elided {
                let off = running_offset;
                running_offset += input_count as u32;
                off
            } else {
                u32::from_le_bytes([nr[base + 4], nr[base + 5], nr[base + 6], nr[base + 7]])
            };
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

    // 3. SubGraphs (v3 packed layout; nested_ranges derived after assembly)
    let subgraphs = {
        let sr = mem.section(SectionKind::SubGraphs);
        let upv = mem.section(SectionKind::SgUpvalueNodes);
        let ed = mem.section(SectionKind::SgEventDecls);
        let df = mem.section(SectionKind::SgDeferEntries);
        let dc = mem.section(SectionKind::SgDeferCapturedInputs);
        let rp = mem.section(SectionKind::SgResetPlan);

        let sg_count = mem.header().subgraph_count as usize;
        let mut subgraphs = Vec::with_capacity(sg_count);
        let mut sr_r = sr;
        // Implicit CSR cursors: the four small regions append in sg order;
        // reset plans use explicit slice starts (last sg's slice ends at the
        // region's total length, known after the loop).
        let mut uv_cur = 0usize;
        let mut ed_cur = 0usize;
        let mut df_cur = 0usize;
        let mut dc_cur = 0usize;
        let mut rp_starts: Vec<u32> = Vec::with_capacity(sg_count + 1);
        for i in 0..sg_count {
            let node_range = (NodeId(read_u32(&mut sr_r)), NodeId(read_u32(&mut sr_r)));
            let param_count = read_u8(&mut sr_r);
            let entry_node = NodeId(read_u32(&mut sr_r));
            let return_node = NodeId(read_u32(&mut sr_r));
            let flags = read_u8(&mut sr_r);
            let has_suspend = flags & 1 != 0;
            let has_rp = flags & 0b10 != 0;
            let loop_kind = u8_to_loop_kind(flags >> 4);
            let upvalue_count = read_u8(&mut sr_r);
            let loop_parent_sg = { let v = read_u32(&mut sr_r); if v == u32::MAX { None } else { Some(SubGraphId(v)) } };
            let cond_node = { let v = read_u32(&mut sr_r); if v == u32::MAX { None } else { Some(NodeId(v)) } };
            let function_id = read_u32(&mut sr_r);
            let iter_next_node = { let v = read_u32(&mut sr_r); if v == u32::MAX { None } else { Some(NodeId(v)) } };
            let uv_len = read_u16(&mut sr_r) as usize;
            let ed_len = read_u16(&mut sr_r) as usize;
            let df_len = read_u16(&mut sr_r) as usize;
            rp_starts.push(read_u32(&mut sr_r));

            let upvalue_outer_nodes: Vec<NodeId> = (0..uv_len)
                .map(|j| {
                    let base = (uv_cur + j) * 4;
                    NodeId(u32::from_le_bytes([upv[base], upv[base+1], upv[base+2], upv[base+3]]))
                })
                .collect();
            uv_cur += uv_len;

            let event_source_decls: Vec<EventSourceDecl> = (0..ed_len)
                .map(|j| {
                    let base = (ed_cur + j) * 8;
                    EventSourceDecl {
                        node: NodeId(u32::from_le_bytes([ed[base], ed[base+1], ed[base+2], ed[base+3]])),
                        kind: u8_to_event_kind(ed[base+4]),
                    }
                })
                .collect();
            ed_cur += ed_len;

            // defer_table: 10B/entry; captured ids continue the dc cursor.
            let mut defer_table = Vec::with_capacity(df_len);
            for _ in 0..df_len {
                let base = df_cur * 10;
                df_cur += 1;
                let trigger_node = NodeId(u32::from_le_bytes([df[base], df[base+1], df[base+2], df[base+3]]));
                let body_subgraph = SubGraphId(u32::from_le_bytes([df[base+4], df[base+5], df[base+6], df[base+7]]));
                let ci_len = u16::from_le_bytes([df[base+8], df[base+9]]) as usize;
                let captured_inputs: Vec<NodeId> = (0..ci_len)
                    .map(|j| {
                        let b2 = (dc_cur + j) * 4;
                        NodeId(u32::from_le_bytes([dc[b2], dc[b2+1], dc[b2+2], dc[b2+3]]))
                    })
                    .collect();
                dc_cur += ci_len;
                defer_table.push(DeferEntry { trigger_node, body_subgraph, captured_inputs, registered: false });
            }

            subgraphs.push(SubGraph {
                id: SubGraphId(i as u32),
                node_range, param_count, entry_node, return_node, has_suspend,
                event_source_decls, defer_table, loop_kind, loop_parent_sg, cond_node,
                function_id, iter_next_node, upvalue_count, upvalue_outer_nodes,
                nested_ranges: Vec::new(), // derived post-assembly (v3)
                reset_plan: None,           // filled below once slice ends are known
            });

            // Reset plans parse in the pass below (a plan's slice ends where
            // the next sg's slice starts); park a marker on the struct now.
            if has_rp {
                subgraphs[i].reset_plan = Some(ResetPlan {
                    reset_to_zero: Vec::new(), reset_to_one: Vec::new(),
                    reset_condition_tree: Vec::new(), condition_tree_plan: Vec::new(),
                });
            }
        }
        rp_starts.push(rp.len() as u32);
        // Fill reset plans with their slice [start_i, start_{i+1}).
        for (i, sg) in subgraphs.iter_mut().enumerate() {
            if sg.reset_plan.is_some() {
                let s = rp_starts[i] as usize;
                let e = rp_starts[i + 1] as usize;
                let mut rp_r = &rp[s..e];
                let rz_len = read_u32(&mut rp_r) as usize;
                let reset_to_zero: Vec<NodeId> = (0..rz_len).map(|_| NodeId(read_u32(&mut rp_r))).collect();
                let ro_len = read_u32(&mut rp_r) as usize;
                let reset_to_one: Vec<NodeId> = (0..ro_len).map(|_| NodeId(read_u32(&mut rp_r))).collect();
                let rc_len = read_u32(&mut rp_r) as usize;
                let reset_condition_tree: Vec<NodeId> = (0..rc_len).map(|_| NodeId(read_u32(&mut rp_r))).collect();
                sg.reset_plan = Some(ResetPlan { reset_to_zero, reset_to_one, reset_condition_tree, condition_tree_plan: Vec::new() });
            }
        }
        subgraphs
    };

    // ---- v2 sparse table helpers ----
    // Layout: [count u32][ (idx u32, blob_off u32) * count ][blob]
    // Returns (index_pairs, blob) where index_pairs[i] = (node_idx, blob_off).
    let split_sparse = |kind: SectionKind| -> (Vec<(u32, u32)>, &[u8]) {
        let r = mem.section(kind);
        let count = u32::from_le_bytes([r[0], r[1], r[2], r[3]]) as usize;
        let mut pairs = Vec::with_capacity(count);
        for i in 0..count {
            let base = 4 + i * 8;
            let idx = u32::from_le_bytes([r[base], r[base+1], r[base+2], r[base+3]]);
            let off = u32::from_le_bytes([r[base+4], r[base+5], r[base+6], r[base+7]]);
            pairs.push((idx, off));
        }
        let blob_start = 4 + count * 8;
        (pairs, &r[blob_start..])
    };

    // ---- category A (sparse scatter) ----
    let scatter_a_u32 = |kind: SectionKind| -> Vec<Option<u32>> {
        let (pairs, blob) = split_sparse(kind);
        let mut v: Vec<Option<u32>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            v[idx as usize] = Some(u32::from_le_bytes([blob[b], blob[b+1], blob[b+2], blob[b+3]]));
        }
        v
    };
    let scatter_a_u16 = |kind: SectionKind| -> Vec<Option<u16>> {
        let (pairs, blob) = split_sparse(kind);
        let mut v: Vec<Option<u16>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            v[idx as usize] = Some(u16::from_le_bytes([blob[b], blob[b+1]]));
        }
        v
    };
    let call_targets: Vec<Option<SubGraphId>> = scatter_a_u32(SectionKind::CallTargets)
        .into_iter().map(|o| o.map(SubGraphId)).collect();
    let await_event_sources: Vec<Option<NodeId>> = scatter_a_u32(SectionKind::AwaitEventSources)
        .into_iter().map(|o| o.map(NodeId)).collect();
    let writeback_targets: Vec<Option<NodeId>> = scatter_a_u32(SectionKind::WritebackTargets)
        .into_iter().map(|o| o.map(NodeId)).collect();
    let global_load_slots: Vec<Option<u32>> = scatter_a_u32(SectionKind::GlobalLoadSlots);
    let global_store_slots: Vec<Option<u32>> = scatter_a_u32(SectionKind::GlobalStoreSlots);
    let field_access_infos: Vec<Option<u16>> = scatter_a_u16(SectionKind::FieldAccessInfos);
    let vtable_call_methods: Vec<Option<u16>> = scatter_a_u16(SectionKind::VtableCallMethods);
    let pattern_field_indices: Vec<Option<u16>> = scatter_a_u16(SectionKind::PatternFieldIndices);
    let closure_call_arg_counts: Vec<Option<u8>> = {
        let (pairs, blob) = split_sparse(SectionKind::ClosureCallArgCounts);
        let mut v: Vec<Option<u8>> = vec![None; n];
        for (idx, off) in pairs { v[idx as usize] = Some(blob[off as usize]); }
        v
    };
    let lib_ret_kinds: Vec<Option<u8>> = {
        let (pairs, blob) = split_sparse(SectionKind::LibRetKinds);
        let mut v: Vec<Option<u8>> = vec![None; n];
        for (idx, off) in pairs { v[idx as usize] = Some(blob[off as usize]); }
        v
    };
    let embed_infos: Vec<Option<u32>> = {
        let (pairs, blob) = split_sparse(SectionKind::EmbedInfos);
        let mut v: Vec<Option<u32>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            v[idx as usize] = Some(u32::from_le_bytes([blob[b], blob[b+1], blob[b+2], blob[b+3]]));
        }
        v
    };
    let resources = parse_resources_section(mem);

    // ---- category B (dense bitmaps, unchanged) ----
    let read_bool_vec = |kind: SectionKind| -> Vec<bool> {
        let r = mem.section(kind);
        (0..n).map(|i| r[i / 8] & (1 << (i % 8)) != 0).collect()
    };
    let tail_call_flags = read_bool_vec(SectionKind::TailCallFlags);
    let safe_op_flags = read_bool_vec(SectionKind::SafeOpFlags);
    let slice_inclusive = read_bool_vec(SectionKind::SliceInclusive);
    // v2: HoistedNode/HoistedOwners sections dropped — no runtime consumers on
    // loaded graphs (post-rebuild, hoisted nodes are covered by their ranges).
    let hoisted_node: Vec<bool> = vec![false; n];
    let hoisted_owners: Vec<SubGraphId> = vec![SubGraphId(u32::MAX); n];

    // ---- category C (sparse strings) ----
    let scatter_str = |kind: SectionKind| -> Vec<Option<String>> {
        let (pairs, blob) = split_sparse(kind);
        let mut v: Vec<Option<String>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            let so = u32::from_le_bytes([blob[b], blob[b+1], blob[b+2], blob[b+3]]);
            let sl = u32::from_le_bytes([blob[b+4], blob[b+5], blob[b+6], blob[b+7]]);
            v[idx as usize] = Some(mem.read_str(so, sl));
        }
        v
    };
    let ffi_call_names = scatter_str(SectionKind::FfiCallNames);
    let field_set_names = scatter_str(SectionKind::FieldSetNames);
    let pattern_ctor_names = scatter_str(SectionKind::PatternCtorNames);
    let pattern_type_names = scatter_str(SectionKind::PatternTypeNames);
    let cast_target_types = scatter_str(SectionKind::CastTargetTypes);

    // ---- category D (sparse composites) ----
    let closure_infos: Vec<Option<ClosureInfo>> = {
        let (pairs, blob) = split_sparse(SectionKind::ClosureInfos);
        let mut v: Vec<Option<ClosureInfo>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            v[idx as usize] = Some(ClosureInfo {
                subgraph_id: SubGraphId(u32::from_le_bytes([blob[b], blob[b+1], blob[b+2], blob[b+3]])),
                arity: blob[b+4],
                self_upvalue_idx: i32::from_le_bytes([blob[b+5], blob[b+6], blob[b+7], blob[b+8]]),
            });
        }
        v
    };
    let partial_infos: Vec<Option<PartialInfo>> = {
        let (pairs, blob) = split_sparse(SectionKind::PartialInfos);
        let mut v: Vec<Option<PartialInfo>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            v[idx as usize] = Some(PartialInfo {
                subgraph_id: SubGraphId(u32::from_le_bytes([blob[b], blob[b+1], blob[b+2], blob[b+3]])),
                bound_count: blob[b+4],
            });
        }
        v
    };
    let lazy_construct_infos: Vec<Option<LazyConstructInfo>> = {
        let (pairs, blob) = split_sparse(SectionKind::LazyConstructInfos);
        let mut v: Vec<Option<LazyConstructInfo>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            v[idx as usize] = Some(LazyConstructInfo {
                thunk_sg: SubGraphId(u32::from_le_bytes([blob[b], blob[b+1], blob[b+2], blob[b+3]])),
            });
        }
        v
    };
    let memo_infos: Vec<Option<MemoInfo>> = {
        let (pairs, blob) = split_sparse(SectionKind::MemoInfos);
        let mut v: Vec<Option<MemoInfo>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            v[idx as usize] = Some(MemoInfo {
                table_index: u32::from_le_bytes([blob[b], blob[b+1], blob[b+2], blob[b+3]]),
                param_count: blob[b+4],
            });
        }
        v
    };

    // ---- category E (sparse: blob keeps the v1 Some-branch encoding) ----
    let const_values: Vec<Option<ConstValue>> = {
        let (pairs, blob) = split_sparse(SectionKind::ConstValues);
        let mut v: Vec<Option<ConstValue>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            let tag = blob[b];
            let w = const_payload_len(tag);
            let payload = &blob[b + 1..b + 1 + w];
            v[idx as usize] = Some(parse_const_value(tag, payload));
        }
        v
    };
    let gate_branches: Vec<Option<GateBranches>> = {
        let (pairs, blob) = split_sparse(SectionKind::GateBranches);
        let mut v: Vec<Option<GateBranches>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            let valid = blob[b];
            let capture = valid == 2;
            let mut r = &blob[b + 1..];
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
            v[idx as usize] = Some(GateBranches { condition_input, branches, capture });
        }
        v
    };
    let record_lit_infos: Vec<Option<RecordLitInfo>> = {
        let (pairs, blob) = split_sparse(SectionKind::RecordLitInfos);
        let mut v: Vec<Option<RecordLitInfo>> = vec![None; n];
        for (idx, off) in pairs {
            let mut r = &blob[off as usize..];
            let _valid = read_u8(&mut r);
            let type_name = { let o = read_u32(&mut r); let l = read_u32(&mut r); mem.read_str(o, l) };
            let fn_count = read_u32(&mut r) as usize;
            let mut field_names = Vec::with_capacity(fn_count);
            for _ in 0..fn_count {
                let o = read_u32(&mut r); let l = read_u32(&mut r);
                field_names.push(if o == u32::MAX { None } else { Some(mem.read_str(o, l)) });
            }
            let constructor = { let o = read_u32(&mut r); let l = read_u32(&mut r); mem.read_str(o, l) };
            let kind = u8_to_record_lit_kind(read_u8(&mut r));
            v[idx as usize] = Some(RecordLitInfo { type_name, field_names, constructor, kind });
        }
        v
    };
    let select_infos: Vec<Option<SelectInfo>> = {
        let (pairs, blob) = split_sparse(SectionKind::SelectInfos);
        let mut v: Vec<Option<SelectInfo>> = vec![None; n];
        for (idx, off) in pairs {
            let mut r = &blob[off as usize..];
            let _valid = read_u8(&mut r);
            let branch_count = read_u32(&mut r) as usize;
            let mut branches = Vec::with_capacity(branch_count);
            for _ in 0..branch_count {
                let subgraph_id = SubGraphId(read_u32(&mut r));
                let event_kind = u8_to_event_kind(read_u8(&mut r));
                let event_source_node = NodeId(read_u32(&mut r));
                branches.push(SelectBranch { subgraph_id, event_kind, event_source_node });
            }
            v[idx as usize] = Some(SelectInfo { branches });
        }
        v
    };
    let trait_construct_infos: Vec<Option<TraitConstructInfo>> = {
        let (pairs, blob) = split_sparse(SectionKind::TraitConstructInfos);
        let mut v: Vec<Option<TraitConstructInfo>> = vec![None; n];
        for (idx, off) in pairs {
            let mut r = &blob[off as usize..];
            let _valid = read_u8(&mut r);
            let trait_name = { let o = read_u32(&mut r); let l = read_u32(&mut r); mem.read_str(o, l) };
            let mn_count = read_u32(&mut r) as usize;
            let mut method_names = Vec::with_capacity(mn_count);
            for _ in 0..mn_count {
                let o = read_u32(&mut r); let l = read_u32(&mut r);
                method_names.push(mem.read_str(o, l));
            }
            let m_count = read_u32(&mut r) as usize;
            let mut methods = Vec::with_capacity(m_count);
            for _ in 0..m_count {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(read_bytes(&mut r, 8));
                methods.push(bytes_to_trait_method_entry(&buf));
            }
            v[idx as usize] = Some(TraitConstructInfo { trait_name, method_names, methods });
        }
        v
    };
    let record_extend_infos: Vec<Option<RecordExtendInfo>> = {
        let (pairs, blob) = split_sparse(SectionKind::RecordExtendInfos);
        let mut v: Vec<Option<RecordExtendInfo>> = vec![None; n];
        for (idx, off) in pairs {
            let mut r = &blob[off as usize..];
            let _valid = read_u8(&mut r);
            let un_count = read_u32(&mut r) as usize;
            let mut update_names = Vec::with_capacity(un_count);
            for _ in 0..un_count {
                let o = read_u32(&mut r); let l = read_u32(&mut r);
                update_names.push(mem.read_str(o, l));
            }
            v[idx as usize] = Some(RecordExtendInfo { update_names });
        }
        v
    };
    let batch_infos: Vec<Option<BatchInfo>> = {
        let (pairs, blob) = split_sparse(SectionKind::BatchInfos);
        let mut v: Vec<Option<BatchInfo>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&blob[b + 1..b + 5]);
            v[idx as usize] = Some(bytes_to_batch_info(&buf));
        }
        v
    };
    let dyn_ffi_infos: Vec<Option<DynFfiInfo>> = {
        let (pairs, blob) = split_sparse(SectionKind::DynFfiInfos);
        let mut v: Vec<Option<DynFfiInfo>> = vec![None; n];
        for (idx, off) in pairs {
            v[idx as usize] = Some(parse_dyn_ffi_info(&blob[off as usize..], mem));
        }
        v
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
    let string_pool: Arc<[u8]> = Arc::from(mem.string_pool().to_vec());

    let mut graph = DataFlowGraph {
        const_cache: Vec::new(),
        sg_initial_pending: Vec::new(),
        sg_initial_seed: Vec::new(),
        downstream_counts: Vec::new(),
        linear_plans: Vec::new(),
        nodes,
        inputs_pool,
        subgraphs,
        entry_subgraph,
        compute_fns,
        downstreams: Vec::new(),
        const_values,
        call_targets,
        gate_branches,
        field_access_infos,
        record_lit_infos,
        ffi_call_names,
        dyn_ffi_infos,
        field_set_names,
        vtable_call_methods,
        await_event_sources,
        closure_infos,
        partial_infos,
        closure_call_arg_counts,
        lib_ret_kinds,
        embed_infos,
        resources,
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
        sg_debug_names: Vec::new(),
        string_pool,
        mem: None,
        sg_uv_offsets: Vec::new(),
        gate_branch_offsets: Vec::new(),
        record_lit_info_offsets: Vec::new(),
        select_info_offsets: Vec::new(),
        trait_construct_info_offsets: Vec::new(),
        record_extend_info_offsets: Vec::new(),
        node_input_offsets: Vec::new(),
        downstream_csr_offsets: Vec::new(),
        downstream_csr_flat: Vec::new(),
    };
    // W5: rebuild the flattened condition-tree reset plans (kept out of the
    // .kzo format; recomputed at load).
    // v3: nested_ranges are derived (no longer serialized) — must run BEFORE
    // precompute_reset_plans, whose condition-tree walk reads them.
    graph.compute_nested_ranges();
    graph.precompute_reset_plans();
    // E0 perf: materialize GateBranches once (no-op on the owned path, which
    // already owns them).
    graph.materialize_gate_branches();
    // v2: downstreams are derived from inputs + gate condition edges.
    graph.compute_downstreams();
    Ok(graph)
}

/// Loads a `DataFlowGraph` from a `GraphMemory` via zerocopy (production path).
///
/// Only eager-loads the 5 complex variable-length tables + subgraphs + downstreams +
/// string_pool + runtime fields. The 24 per-Node scalar tables plus `nodes` and `inputs`
/// are read zerocopy from the mmap slices via accessor methods, without copying into owned
/// `Vec`s.
pub fn load_zerocopy(mem: GraphMemory) -> io::Result<DataFlowGraph> {
    let n = mem.header().node_count as usize;
    let offsets_elided = mem.header().flags & FLAG_NODE_INPUT_OFFSETS_ELIDED != 0;

    // SubGraphs eager-load (v3 packed layout — same scheme as the eager
    // loader; upvalue_outer_nodes stay zerocopy CSR, nested_ranges are
    // derived post-assembly, reset plans parse from explicit slice starts).
    let (subgraphs, sg_uv_offsets) = {
        let sr = mem.section(SectionKind::SubGraphs);
        let ed = mem.section(SectionKind::SgEventDecls);
        let df = mem.section(SectionKind::SgDeferEntries);
        let dc = mem.section(SectionKind::SgDeferCapturedInputs);
        let rp = mem.section(SectionKind::SgResetPlan);

        let sg_count = mem.header().subgraph_count as usize;
        let mut subgraphs = Vec::with_capacity(sg_count);
        let mut sg_uv_offsets = Vec::with_capacity(sg_count);
        let mut sr_r = sr;
        let mut uv_cur = 0usize;
        let mut ed_cur = 0usize;
        let mut df_cur = 0usize;
        let mut dc_cur = 0usize;
        let mut rp_starts: Vec<u32> = Vec::with_capacity(sg_count + 1);
        for i in 0..sg_count {
            let node_range = (NodeId(read_u32(&mut sr_r)), NodeId(read_u32(&mut sr_r)));
            let param_count = read_u8(&mut sr_r);
            let entry_node = NodeId(read_u32(&mut sr_r));
            let return_node = NodeId(read_u32(&mut sr_r));
            let flags = read_u8(&mut sr_r);
            let has_suspend = flags & 1 != 0;
            let has_rp = flags & 0b10 != 0;
            let loop_kind = u8_to_loop_kind(flags >> 4);
            let upvalue_count = read_u8(&mut sr_r);
            let loop_parent_sg = { let v = read_u32(&mut sr_r); if v == u32::MAX { None } else { Some(SubGraphId(v)) } };
            let cond_node = { let v = read_u32(&mut sr_r); if v == u32::MAX { None } else { Some(NodeId(v)) } };
            let function_id = read_u32(&mut sr_r);
            let iter_next_node = { let v = read_u32(&mut sr_r); if v == u32::MAX { None } else { Some(NodeId(v)) } };
            let uv_len = read_u16(&mut sr_r) as usize;
            let ed_len = read_u16(&mut sr_r) as usize;
            let df_len = read_u16(&mut sr_r) as usize;
            rp_starts.push(read_u32(&mut sr_r));

            // upvalue_outer_nodes: zerocopy CSR — store byte offset/len (u32 elements).
            sg_uv_offsets.push(((uv_cur * 4) as u32, uv_len as u32));
            uv_cur += uv_len;
            let upvalue_outer_nodes: Vec<NodeId> = Vec::new();

            let event_source_decls: Vec<EventSourceDecl> = (0..ed_len)
                .map(|j| {
                    let base = (ed_cur + j) * 8;
                    EventSourceDecl {
                        node: NodeId(u32::from_le_bytes([ed[base], ed[base+1], ed[base+2], ed[base+3]])),
                        kind: u8_to_event_kind(ed[base+4]),
                    }
                })
                .collect();
            ed_cur += ed_len;

            let mut defer_table = Vec::with_capacity(df_len);
            for _ in 0..df_len {
                let base = df_cur * 10;
                df_cur += 1;
                let trigger_node = NodeId(u32::from_le_bytes([df[base], df[base+1], df[base+2], df[base+3]]));
                let body_subgraph = SubGraphId(u32::from_le_bytes([df[base+4], df[base+5], df[base+6], df[base+7]]));
                let ci_len = u16::from_le_bytes([df[base+8], df[base+9]]) as usize;
                let captured_inputs: Vec<NodeId> = (0..ci_len)
                    .map(|j| {
                        let b2 = (dc_cur + j) * 4;
                        NodeId(u32::from_le_bytes([dc[b2], dc[b2+1], dc[b2+2], dc[b2+3]]))
                    })
                    .collect();
                dc_cur += ci_len;
                defer_table.push(DeferEntry { trigger_node, body_subgraph, captured_inputs, registered: false });
            }

            subgraphs.push(SubGraph {
                id: SubGraphId(i as u32),
                node_range, param_count, entry_node, return_node, has_suspend,
                event_source_decls, defer_table, loop_kind, loop_parent_sg, cond_node,
                function_id, iter_next_node, upvalue_count, upvalue_outer_nodes,
                nested_ranges: Vec::new(), // derived post-assembly (v3)
                reset_plan: None,
            });

            if has_rp {
                subgraphs[i].reset_plan = Some(ResetPlan {
                    reset_to_zero: Vec::new(), reset_to_one: Vec::new(),
                    reset_condition_tree: Vec::new(), condition_tree_plan: Vec::new(),
                });
            }
        }
        rp_starts.push(rp.len() as u32);
        for (i, sg) in subgraphs.iter_mut().enumerate() {
            if sg.reset_plan.is_some() {
                let s = rp_starts[i] as usize;
                let e = rp_starts[i + 1] as usize;
                let mut rp_r = &rp[s..e];
                let rz_len = read_u32(&mut rp_r) as usize;
                let reset_to_zero: Vec<NodeId> = (0..rz_len).map(|_| NodeId(read_u32(&mut rp_r))).collect();
                let ro_len = read_u32(&mut rp_r) as usize;
                let reset_to_one: Vec<NodeId> = (0..ro_len).map(|_| NodeId(read_u32(&mut rp_r))).collect();
                let rc_len = read_u32(&mut rp_r) as usize;
                let reset_condition_tree: Vec<NodeId> = (0..rc_len).map(|_| NodeId(read_u32(&mut rp_r))).collect();
                sg.reset_plan = Some(ResetPlan { reset_to_zero, reset_to_one, reset_condition_tree, condition_tree_plan: Vec::new() });
            }
        }
        (subgraphs, sg_uv_offsets)
    };

    // v2: node inputs offsets — when elided (contiguous pool), materialize the
    // prefix-sum table once so `node()` stays O(1) per access.
    let node_input_offsets: Vec<u32> = if offsets_elided {
        let r = mem.section(SectionKind::Nodes);
        let mut offsets = Vec::with_capacity(n);
        let mut acc = 0u32;
        for i in 0..n {
            offsets.push(acc);
            acc += r[i * 4 + 1] as u32; // input_count byte
        }
        offsets
    } else {
        Vec::new()
    };

    // ---- v2 sparse helpers (same layout as the eager path) ----
    let split_sparse = |kind: SectionKind| -> (Vec<(u32, u32)>, u32, &[u8]) {
        let r = mem.section(kind);
        let count = u32::from_le_bytes([r[0], r[1], r[2], r[3]]) as usize;
        let mut pairs = Vec::with_capacity(count);
        for i in 0..count {
            let base = 4 + i * 8;
            let idx = u32::from_le_bytes([r[base], r[base+1], r[base+2], r[base+3]]);
            let off = u32::from_le_bytes([r[base+4], r[base+5], r[base+6], r[base+7]]);
            pairs.push((idx, off));
        }
        let blob_start = (4 + count * 8) as u32;
        (pairs, blob_start, &r[blob_start as usize..])
    };
    // Offsets table for the on-demand E-class parsers: absolute byte offsets
    // into the section (blob_start + relative), u32::MAX = None.
    let build_offsets_table = |kind: SectionKind| -> Vec<u32> {
        let (pairs, blob_start, _) = split_sparse(kind);
        let mut offsets = vec![u32::MAX; n];
        for (idx, off) in pairs {
            offsets[idx as usize] = blob_start + off;
        }
        offsets
    };

    // ---- category A/C/D: scatter-materialize into owned Vecs ----
    // (v2 sparse tables are tiny; the hot accessors read owned fields — this
    // also removes the old fixed-stride mmap reads that no longer apply.)
    let scatter_a_u32 = |kind: SectionKind| -> Vec<Option<u32>> {
        let (pairs, _, blob) = split_sparse(kind);
        let mut v: Vec<Option<u32>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            v[idx as usize] = Some(u32::from_le_bytes([blob[b], blob[b+1], blob[b+2], blob[b+3]]));
        }
        v
    };
    let scatter_a_u16 = |kind: SectionKind| -> Vec<Option<u16>> {
        let (pairs, _, blob) = split_sparse(kind);
        let mut v: Vec<Option<u16>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            v[idx as usize] = Some(u16::from_le_bytes([blob[b], blob[b+1]]));
        }
        v
    };
    let call_targets: Vec<Option<SubGraphId>> = scatter_a_u32(SectionKind::CallTargets)
        .into_iter().map(|o| o.map(SubGraphId)).collect();
    let await_event_sources: Vec<Option<NodeId>> = scatter_a_u32(SectionKind::AwaitEventSources)
        .into_iter().map(|o| o.map(NodeId)).collect();
    let writeback_targets: Vec<Option<NodeId>> = scatter_a_u32(SectionKind::WritebackTargets)
        .into_iter().map(|o| o.map(NodeId)).collect();
    let global_load_slots: Vec<Option<u32>> = scatter_a_u32(SectionKind::GlobalLoadSlots);
    let global_store_slots: Vec<Option<u32>> = scatter_a_u32(SectionKind::GlobalStoreSlots);
    let field_access_infos: Vec<Option<u16>> = scatter_a_u16(SectionKind::FieldAccessInfos);
    let vtable_call_methods: Vec<Option<u16>> = scatter_a_u16(SectionKind::VtableCallMethods);
    let pattern_field_indices: Vec<Option<u16>> = scatter_a_u16(SectionKind::PatternFieldIndices);
    let closure_call_arg_counts: Vec<Option<u8>> = {
        let (pairs, _, blob) = split_sparse(SectionKind::ClosureCallArgCounts);
        let mut v: Vec<Option<u8>> = vec![None; n];
        for (idx, off) in pairs { v[idx as usize] = Some(blob[off as usize]); }
        v
    };
    let lib_ret_kinds: Vec<Option<u8>> = {
        let (pairs, _, blob) = split_sparse(SectionKind::LibRetKinds);
        let mut v: Vec<Option<u8>> = vec![None; n];
        for (idx, off) in pairs { v[idx as usize] = Some(blob[off as usize]); }
        v
    };
    let embed_infos: Vec<Option<u32>> = {
        let (pairs, _, blob) = split_sparse(SectionKind::EmbedInfos);
        let mut v: Vec<Option<u32>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            v[idx as usize] = Some(u32::from_le_bytes([blob[b], blob[b+1], blob[b+2], blob[b+3]]));
        }
        v
    };
    let resources = parse_resources_section(&mem);
    let scatter_str = |kind: SectionKind| -> Vec<Option<String>> {
        let (pairs, _, blob) = split_sparse(kind);
        let mut v: Vec<Option<String>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            let so = u32::from_le_bytes([blob[b], blob[b+1], blob[b+2], blob[b+3]]);
            let sl = u32::from_le_bytes([blob[b+4], blob[b+5], blob[b+6], blob[b+7]]);
            v[idx as usize] = Some(mem.read_str(so, sl));
        }
        v
    };
    let ffi_call_names = scatter_str(SectionKind::FfiCallNames);
    let field_set_names = scatter_str(SectionKind::FieldSetNames);
    let pattern_ctor_names = scatter_str(SectionKind::PatternCtorNames);
    let pattern_type_names = scatter_str(SectionKind::PatternTypeNames);
    let cast_target_types = scatter_str(SectionKind::CastTargetTypes);
    let closure_infos: Vec<Option<ClosureInfo>> = {
        let (pairs, _, blob) = split_sparse(SectionKind::ClosureInfos);
        let mut v: Vec<Option<ClosureInfo>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            v[idx as usize] = Some(ClosureInfo {
                subgraph_id: SubGraphId(u32::from_le_bytes([blob[b], blob[b+1], blob[b+2], blob[b+3]])),
                arity: blob[b+4],
                self_upvalue_idx: i32::from_le_bytes([blob[b+5], blob[b+6], blob[b+7], blob[b+8]]),
            });
        }
        v
    };
    let partial_infos: Vec<Option<PartialInfo>> = {
        let (pairs, _, blob) = split_sparse(SectionKind::PartialInfos);
        let mut v: Vec<Option<PartialInfo>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            v[idx as usize] = Some(PartialInfo {
                subgraph_id: SubGraphId(u32::from_le_bytes([blob[b], blob[b+1], blob[b+2], blob[b+3]])),
                bound_count: blob[b+4],
            });
        }
        v
    };
    let lazy_construct_infos: Vec<Option<LazyConstructInfo>> = {
        let (pairs, _, blob) = split_sparse(SectionKind::LazyConstructInfos);
        let mut v: Vec<Option<LazyConstructInfo>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            v[idx as usize] = Some(LazyConstructInfo {
                thunk_sg: SubGraphId(u32::from_le_bytes([blob[b], blob[b+1], blob[b+2], blob[b+3]])),
            });
        }
        v
    };
    let memo_infos: Vec<Option<MemoInfo>> = {
        let (pairs, _, blob) = split_sparse(SectionKind::MemoInfos);
        let mut v: Vec<Option<MemoInfo>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            v[idx as usize] = Some(MemoInfo {
                table_index: u32::from_le_bytes([blob[b], blob[b+1], blob[b+2], blob[b+3]]),
                param_count: blob[b+4],
            });
        }
        v
    };
    let batch_infos: Vec<Option<BatchInfo>> = {
        let (pairs, _, blob) = split_sparse(SectionKind::BatchInfos);
        let mut v: Vec<Option<BatchInfo>> = vec![None; n];
        for (idx, off) in pairs {
            let b = off as usize;
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&blob[b + 1..b + 5]);
            v[idx as usize] = Some(bytes_to_batch_info(&buf));
        }
        v
    };
    let dyn_ffi_infos: Vec<Option<DynFfiInfo>> = {
        let (pairs, _, blob) = split_sparse(SectionKind::DynFfiInfos);
        let mut v: Vec<Option<DynFfiInfo>> = vec![None; n];
        for (idx, off) in pairs {
            v[idx as usize] = Some(parse_dyn_ffi_info(&blob[off as usize..], &mem));
        }
        v
    };

    // ---- category E: on-demand parsers + per-node absolute offsets tables ----
    let gate_branch_offsets = build_offsets_table(SectionKind::GateBranches);
    let record_lit_info_offsets = build_offsets_table(SectionKind::RecordLitInfos);
    let select_info_offsets = build_offsets_table(SectionKind::SelectInfos);
    let trait_construct_info_offsets = build_offsets_table(SectionKind::TraitConstructInfos);
    let record_extend_info_offsets = build_offsets_table(SectionKind::RecordExtendInfos);

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

    let mut graph = DataFlowGraph {
        const_cache: Vec::new(),
        sg_initial_pending: Vec::new(),
        sg_initial_seed: Vec::new(),
        downstream_counts: Vec::new(),
        linear_plans: Vec::new(),
        nodes: Vec::new(),
        inputs_pool: InputsPool::new(),
        subgraphs,
        entry_subgraph,
        compute_fns,
        downstreams: Vec::new(),
        const_values: Vec::new(),
        call_targets,
        gate_branches: Vec::new(),
        field_access_infos,
        record_lit_infos: Vec::new(),
        ffi_call_names,
        dyn_ffi_infos,
        field_set_names,
        vtable_call_methods,
        await_event_sources,
        closure_infos,
        partial_infos,
        closure_call_arg_counts,
        lib_ret_kinds,
        embed_infos,
        resources,
        select_infos: Vec::new(),
        writeback_targets,
        tail_call_flags: Vec::new(),
        safe_op_flags: Vec::new(),
        hoisted_node: vec![false; n],
        hoisted_owners: vec![SubGraphId(u32::MAX); n],
        batch_infos,
        ir_errors: Vec::new(),
        trait_construct_infos: Vec::new(),
        lazy_construct_infos,
        record_extend_infos: Vec::new(),
        slice_inclusive: Vec::new(),
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
        sg_debug_names: Vec::new(),
        string_pool,
        mem: Some(mem),
        sg_uv_offsets,
        gate_branch_offsets,
        record_lit_info_offsets,
        select_info_offsets,
        trait_construct_info_offsets,
        record_extend_info_offsets,
        node_input_offsets,
        downstream_csr_offsets: Vec::new(),
        downstream_csr_flat: Vec::new(),
    };
    // W5: rebuild the flattened condition-tree reset plans (kept out of the
    // .kzo format; recomputed at load). Works through the mem-agnostic
    // accessors on the zerocopy backing.
    // v3: nested_ranges are derived (no longer serialized) — must run BEFORE
    // precompute_reset_plans, whose condition-tree walk reads them.
    graph.compute_nested_ranges();
    graph.precompute_reset_plans();
    // E0 perf: materialize GateBranches once (borrowed access on every Gate execution).
    graph.materialize_gate_branches();
    // v2: Downstreams section dropped — derive the flat CSR table from inputs
    // + gate condition edges (gate branches must be materialized first).
    graph.compute_downstream_csr();
    Ok(graph)
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

// ==================== Roundtrip tests (v2) ====================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Ir::*;

    /// Builds a small synthetic graph exercising every serialized surface:
    /// two function subgraphs (entry + callee), a cross-function Call node,
    /// a Gate with branches, Const values (incl. Str), closure/partial/lazy
    /// metadata, FFI call names, boolean flags, writeback/global slots and a
    /// defer entry.
    fn build_sample_graph() -> DataFlowGraph {
        let mut g = DataFlowGraph::new();

        // Callee function sg: [param, add, return] — nodes 0..3
        let callee_sg = g.add_subgraph(SubGraph {
            id: SubGraphId(0),
            node_range: (NodeId(0), NodeId(3)),
            param_count: 2,
            entry_node: NodeId(0),
            return_node: NodeId(2),
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: 0,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });

        // callee nodes: param placeholders + an add
        let off = g.inputs_pool.push(&[]);
        g.add_node(Node { kind: NodeKind::Const, input_count: 0, inputs_offset: off, compute_fn: CF_NOOP });
        let off = g.inputs_pool.push(&[]);
        g.add_node(Node { kind: NodeKind::Const, input_count: 0, inputs_offset: off, compute_fn: CF_NOOP });
        let off = g.inputs_pool.push(&[NodeId(0), NodeId(1)]);
        g.add_node(Node { kind: NodeKind::BinOp, input_count: 2, inputs_offset: off, compute_fn: ComputeFnId(1) });
        g.const_values[2] = Some(ConstValue::I32(7));

        // Entry function sg: [const "hi", const 42, const true, gate, call]
        let entry_start = g.nodes.len() as u32;
        g.add_subgraph(SubGraph {
            id: SubGraphId(1),
            node_range: (NodeId(entry_start), NodeId(entry_start + 5)),
            param_count: 0,
            entry_node: NodeId(entry_start),
            return_node: NodeId(entry_start + 4),
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: vec![DeferEntry {
                trigger_node: NodeId(entry_start + 4),
                body_subgraph: SubGraphId(0),
                captured_inputs: vec![NodeId(entry_start)],
                registered: false,
            }],
            loop_kind: LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: 1,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: Some(ResetPlan {
                reset_to_zero: vec![NodeId(entry_start)],
                reset_to_one: vec![],
                reset_condition_tree: vec![],
                condition_tree_plan: Vec::new(),
            }),
        });

        // string pool for the Str const
        let pool: Arc<[u8]> = Arc::from(b"hello world".to_vec());
        g.string_pool = pool;

        // const "hello" (Str), const 42 (I64), const true (Bool)
        let e0 = entry_start as usize;
        let off = g.inputs_pool.push(&[]);
        g.add_node(Node { kind: NodeKind::Const, input_count: 0, inputs_offset: off, compute_fn: CF_NOOP });
        g.const_values[e0] = Some(ConstValue::Str { offset: 0, len: 11 });
        let off = g.inputs_pool.push(&[]);
        g.add_node(Node { kind: NodeKind::Const, input_count: 0, inputs_offset: off, compute_fn: CF_NOOP });
        g.const_values[e0 + 1] = Some(ConstValue::I64(42));
        let off = g.inputs_pool.push(&[]);
        g.add_node(Node { kind: NodeKind::Const, input_count: 0, inputs_offset: off, compute_fn: CF_NOOP });
        g.const_values[e0 + 2] = Some(ConstValue::Bool(true));

        // Gate node (branch target = callee sg)
        let off = g.inputs_pool.push(&[NodeId((e0 + 2) as u32)]);
        let gate = g.add_node(Node { kind: NodeKind::Gate, input_count: 1, inputs_offset: off, compute_fn: ComputeFnId(47) });
        g.set_gate_branches(gate, GateBranches {
            condition_input: NodeId((e0 + 2) as u32),
            branches: vec![(true, SubGraphId(0), vec![NodeId(e0 as u32), NodeId((e0 + 1) as u32)])],
            capture: true,
        });

        // Call node → callee
        let off = g.inputs_pool.push(&[NodeId((e0 + 1) as u32), gate]);
        let call = g.add_node(Node { kind: NodeKind::Call, input_count: 2, inputs_offset: off, compute_fn: ComputeFnId(36) });
        g.set_call_target(call, SubGraphId(0));
        g.set_tail_call(call);

        // Metadata coverage (gate + call nodes as hosts)
        g.set_writeback_target(gate, NodeId((e0 + 1) as u32));
        g.set_global_load_slot(gate, 3);
        g.set_global_store_slot(call, 4);
        g.set_ffi_call_name(call, "kuzo_extern_test".to_string());
        g.set_cast_target_type(gate, "i32".to_string());
        g.set_closure_info(gate, ClosureInfo { subgraph_id: callee_sg, arity: 2, self_upvalue_idx: -1 });
        g.set_partial_info(gate, PartialInfo { subgraph_id: callee_sg, bound_count: 1 });
        g.set_lazy_construct_info(gate, LazyConstructInfo { thunk_sg: callee_sg });
        g.set_memo_info(gate, MemoInfo { table_index: 1, param_count: 2 });
        g.set_field_access_info(gate, 9);
        g.set_vtable_call(gate, 2);
        g.set_await_event_source(gate, NodeId(e0 as u32));
        g.set_closure_call_arg_count(call, 2);
        g.set_pattern_ctor_name(call, "Some".to_string());
        g.set_pattern_type_name(call, "Option".to_string());
        g.set_pattern_field_index(call, 1);
        g.set_safe_op(gate);
        g.set_slice_inclusive(call, true);
        g.set_dyn_ffi_info(call, DynFfiInfo {
            symbol: "kuzo_extern_test".to_string(),
            sig: crate::ffi::Abi::AbiSig::new(
                vec![crate::ffi::Abi::AbiType::Int { bits: 64, signed: false }],
                crate::ffi::Abi::AbiType::Void,
            ),
            arg_count: 1,
        });
        g.set_record_lit_info(gate, RecordLitInfo {
            type_name: "P".to_string(),
            field_names: vec![Some("x".to_string()), None],
            constructor: "P".to_string(),
            kind: RecordLitKind::Record,
        });
        g.set_select_info(gate, SelectInfo {
            branches: vec![SelectBranch { subgraph_id: SubGraphId(0), event_kind: EventSourceKind::SubgraphComplete, event_source_node: NodeId(e0 as u32) }],
        });
        g.set_trait_construct_info(gate, TraitConstructInfo {
            trait_name: "T".to_string(),
            method_names: vec!["m".to_string()],
            methods: vec![TraitMethodEntry { subgraph_id: SubGraphId(0), arity: 1, upvalue_count: 0 }],
        });
        g.set_record_extend_info(gate, RecordExtendInfo { update_names: vec!["y".to_string()] });
        g.set_batch_info(gate, BatchInfo { tag: crate::value::ValueTag::I32, op: BatchOp::Bin(crate::value::BinOp::Add) });

        g.set_entry_subgraph(SubGraphId(1));
        g.compute_downstreams();
        g
    }

    /// Field-by-field semantic comparison through the mem-agnostic accessors —
    /// the exact surfaces the engine reads.
    fn assert_graphs_equal(a: &DataFlowGraph, b: &DataFlowGraph) {
        assert_eq!(a.node_count(), b.node_count());
        let n = a.node_count();
        for i in 0..n {
            let na = a.node(i);
            let nb = b.node(i);
            assert_eq!(format!("{:?}", na.kind), format!("{:?}", nb.kind), "node {i} kind");
            assert_eq!(na.input_count, nb.input_count, "node {i} input_count");
            assert_eq!(na.compute_fn.0, nb.compute_fn.0, "node {i} compute_fn");
            let ia = a.inputs(na.inputs_offset, na.input_count);
            let ib = b.inputs(nb.inputs_offset, nb.input_count);
            assert_eq!(ia, ib, "node {i} inputs");
            // Str consts are re-interned per pool: compare resolved bytes, not offsets.
            let (ca, cb) = (a.const_value(i), b.const_value(i));
            match (&ca, &cb) {
                (Some(ConstValue::Str { offset: o1, len: l1 }), Some(ConstValue::Str { offset: o2, len: l2 })) => {
                    assert_eq!(l1, l2, "node {i} str len");
                    let s1 = &a.string_pool_slice()[*o1 as usize..(*o1 + *l1) as usize];
                    let s2 = &b.string_pool_slice()[*o2 as usize..(*o2 + *l2) as usize];
                    assert_eq!(s1, s2, "node {i} str bytes");
                }
                _ => assert_eq!(format!("{:?}", ca), format!("{:?}", cb), "node {i} const_value"),
            }
            assert_eq!(a.call_target(i), b.call_target(i), "node {i} call_target");
            assert_eq!(a.field_access_info(i), b.field_access_info(i));
            assert_eq!(a.vtable_call_method(i), b.vtable_call_method(i));
            assert_eq!(a.await_event_source(i), b.await_event_source(i));
            assert_eq!(a.writeback_target(i), b.writeback_target(i));
            assert_eq!(a.global_load_slot(i), b.global_load_slot(i));
            assert_eq!(a.global_store_slot(i), b.global_store_slot(i));
            assert_eq!(a.pattern_field_index(i), b.pattern_field_index(i));
            assert_eq!(a.closure_call_arg_count(i), b.closure_call_arg_count(i));
            assert_eq!(a.tail_call_flag(i), b.tail_call_flag(i));
            assert_eq!(a.safe_op_flag(i), b.safe_op_flag(i));
            assert_eq!(a.slice_inclusive(i), b.slice_inclusive(i));
            assert_eq!(a.ffi_call_name(i), b.ffi_call_name(i));
            assert_eq!(a.field_set_name(i), b.field_set_name(i));
            assert_eq!(a.pattern_ctor_name(i), b.pattern_ctor_name(i));
            assert_eq!(a.pattern_type_name(i), b.pattern_type_name(i));
            assert_eq!(a.cast_target_type(i), b.cast_target_type(i));
            assert_eq!(format!("{:?}", a.closure_info(i)), format!("{:?}", b.closure_info(i)));
            assert_eq!(format!("{:?}", a.partial_info(i)), format!("{:?}", b.partial_info(i)));
            assert_eq!(format!("{:?}", a.lazy_construct_info(i)), format!("{:?}", b.lazy_construct_info(i)));
            assert_eq!(format!("{:?}", a.memo_info(i)), format!("{:?}", b.memo_info(i)));
            assert_eq!(format!("{:?}", a.dyn_ffi_info(i)), format!("{:?}", b.dyn_ffi_info(i)));
            assert_eq!(format!("{:?}", a.batch_info(i)), format!("{:?}", b.batch_info(i)));
            assert_eq!(format!("{:?}", a.record_lit_info_at(i)), format!("{:?}", b.record_lit_info_at(i)));
            assert_eq!(format!("{:?}", a.select_info_at(i)), format!("{:?}", b.select_info_at(i)));
            assert_eq!(format!("{:?}", a.trait_construct_info_at(i)), format!("{:?}", b.trait_construct_info_at(i)));
            assert_eq!(format!("{:?}", a.record_extend_info_at(i)), format!("{:?}", b.record_extend_info_at(i)));
            let gba = format!("{:?}", a.gate_branches_at(i));
            let gbb = format!("{:?}", b.gate_branches_at(i));
            assert_eq!(gba, gbb, "node {i} gate_branches");
            assert_eq!(a.downstream_slice(i), b.downstream_slice(i), "node {i} downstreams");
        }
        assert_eq!(a.subgraphs.len(), b.subgraphs.len());
        for (sa, sb) in a.subgraphs.iter().zip(b.subgraphs.iter()) {
            assert_eq!(sa.id, sb.id);
            assert_eq!(sa.node_range, sb.node_range);
            assert_eq!(sa.param_count, sb.param_count);
            assert_eq!(sa.entry_node, sb.entry_node);
            assert_eq!(sa.return_node, sb.return_node);
            assert_eq!(sa.has_suspend, sb.has_suspend);
            assert_eq!(format!("{:?}", sa.loop_kind), format!("{:?}", sb.loop_kind));
            assert_eq!(sa.loop_parent_sg, sb.loop_parent_sg);
            assert_eq!(sa.cond_node, sb.cond_node);
            assert_eq!(sa.function_id, sb.function_id);
            assert_eq!(sa.iter_next_node, sb.iter_next_node);
            assert_eq!(sa.upvalue_count, sb.upvalue_count);
            assert_eq!(sa.defer_table.len(), sb.defer_table.len());
            for (da, db) in sa.defer_table.iter().zip(sb.defer_table.iter()) {
                assert_eq!(da.trigger_node, db.trigger_node);
                assert_eq!(da.body_subgraph, db.body_subgraph);
                assert_eq!(da.captured_inputs, db.captured_inputs);
            }
            assert_eq!(
                sa.reset_plan.as_ref().map(|p| &p.reset_to_zero),
                sb.reset_plan.as_ref().map(|p| &p.reset_to_zero),
            );
            assert_eq!(sa.nested_ranges, sb.nested_ranges, "sg {} nested_ranges", sa.id.0);
            assert_eq!(sa.reset_plan.as_ref().map(|p| &p.reset_condition_tree), sb.reset_plan.as_ref().map(|p| &p.reset_condition_tree));
        }
        assert_eq!(a.entry_subgraph, b.entry_subgraph);
    }

    #[test]
    fn v2_roundtrip_eager_and_zerocopy() {
        let g = build_sample_graph();
        let bytes = serialize_solidify(&g);
        assert!(bytes.len() < 2048, "sample artifact unexpectedly large: {}", bytes.len());

        // Eager (owned) path.
        let eager = load_solidify_from_bytes(&bytes).expect("eager load");
        assert_graphs_equal(&g, &eager);

        // Zerocopy (mmap-style) path.
        let zc = load_zerocopy_from_bytes(bytes.clone()).expect("zerocopy load");
        assert_graphs_equal(&g, &zc);

        // The two load paths must agree with each other too.
        assert_graphs_equal(&eager, &zc);
    }

    #[test]
    fn v2_rejects_corruption() {
        let g = build_sample_graph();
        let mut bytes = serialize_solidify(&g);
        // Corrupt one body byte -> CRC mismatch.
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert!(load_solidify_from_bytes(&bytes).is_err());
    }
}
