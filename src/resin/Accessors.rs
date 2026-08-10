//! DataFlowGraph zerocopy 访问层
//!
//! 当 `DataFlowGraph.mem = Some(GraphMemory)` 时（.resin 加载路径），
//! 24 个 per-Node 标量表 + nodes + inputs 通过 accessor 方法直接从
//! mmap'd 字节切片读取，无需拷贝到 owned Vec。
//!
//! 当 `mem = None` 时（构建路径），accessor 方法回退到 owned Vec 字段访问。
//!
//! 5 个变长复杂表（gate_branches / record_lit_infos / select_infos /
//! trait_construct_infos / record_extend_infos）+ subgraphs + downstreams
//! 在两条路径都保持 owned（加载期 eager-load），无需 mem 分支。

#![allow(non_snake_case)]

use crate::ir::Ir::*;
use super::Spec::*;

// ==================== LE 读取辅助 ====================

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

// ==================== accessor 生成宏 ====================
//
// 类别 A（定宽标量 Option）、B（布尔 bitmap）、C（含字符串）三类 accessor 高度重复，
// 以下 3 个宏生成方法体。类别 D（定宽复合）及 5 个 on-demand 变长表结构异构，不宏化。

/// 类别 A accessor：zerocopy 读取定宽标量，哨兵值 = None。
///
/// `$read` 为 rd_u8/rd_u16/rd_u32（带类型注解的 helper，规避闭包参数推断问题）。
/// `$wrap` 将解码整数包装为目标类型（`|v| v` 表示本身即整数）。
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

/// 类别 B accessor：zerocopy 读取 bitmap 布尔位。
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

/// 类别 C accessor：zerocopy 读取 StrRef → &str（从 StringPool section）。
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

// ==================== accessor 方法 ====================

impl DataFlowGraph {
    // ---- 计数 ----

    /// 节点总数（构建路径 = nodes.len()，加载路径 = header.node_count）
    #[inline]
    pub fn node_count(&self) -> usize {
        if let Some(ref mem) = self.mem {
            mem.header().node_count as usize
        } else {
            self.nodes.len()
        }
    }

    // ---- Node ----

    /// 按索引读取节点（Copy，14B 从 mmap 切片读取）
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

    /// 读取节点的输入切片（zerocopy：从 mmap Inputs section transmute 为 &[NodeId]）
    #[inline]
    pub fn inputs(&self, offset: u32, count: u8) -> &[NodeId] {
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::Inputs);
            let start = offset as usize * 4;
            let n = count as usize;
            // SAFETY: NodeId is #[repr(transparent)] over u32（4B, 4B 对齐）。
            // Inputs section 起始 4B 对齐（section align = 4），start = offset*4 也是 4 的倍数。
            // 切片 [start..start+n*4] 在 section 边界内（IR 保证 offset+count <= total）。
            unsafe {
                std::slice::from_raw_parts(r.as_ptr().add(start) as *const NodeId, n)
            }
        } else {
            self.inputs_pool.get(offset, count)
        }
    }

    // ---- 类别 A: 定宽标量表（zerocopy，sentinel 表示 None）----

    accessor_opt!(call_target, call_targets, CallTargets, rd_u32, 4, u32::MAX, SubGraphId, |v| SubGraphId(v));
    accessor_opt!(field_access_info, field_access_infos, FieldAccessInfos, rd_u16, 2, u16::MAX, u16, |v| v);
    accessor_opt!(vtable_call_method, vtable_call_methods, VtableCallMethods, rd_u16, 2, u16::MAX, u16, |v| v);
    accessor_opt!(await_event_source, await_event_sources, AwaitEventSources, rd_u32, 4, u32::MAX, NodeId, |v| NodeId(v));
    accessor_opt!(writeback_target, writeback_targets, WritebackTargets, rd_u32, 4, u32::MAX, NodeId, |v| NodeId(v));

    // hoisted_owner: SubGraphId（无 None，直接读）
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

    // ---- 类别 B: 布尔表（zerocopy，bitmap 读取）----

    accessor_bool!(tail_call_flag, tail_call_flags, TailCallFlags);
    accessor_bool!(safe_op_flag, safe_op_flags, SafeOpFlags);
    accessor_bool!(is_hoisted_node, hoisted_node, HoistedNode);
    accessor_bool!(slice_inclusive, slice_inclusive, SliceInclusive);

    // ---- 类别 C: 含字符串表（zerocopy，StrRef → &str from StringPool）----

    accessor_str!(ffi_call_name, ffi_call_names, FfiCallNames);
    accessor_str!(field_set_name, field_set_names, FieldSetNames);
    accessor_str!(pattern_ctor_name, pattern_ctor_names, PatternCtorNames);
    accessor_str!(cast_target_type, cast_target_types, CastTargetTypes);

    // ---- 类别 D: 定宽复合表（zerocopy，validity byte + 字段）----

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

    // ---- 定宽变长表（zerocopy，tag + payload）----

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

    // ---- Downstreams（zerocopy CSR 访问）----

    /// 返回节点 `idx` 的下游节点切片。
    ///
    /// 加载路径：从 mmap Downstreams section 直接返回 `&[NodeId]` 切片，
    /// 无堆分配（消除 `Vec<Vec<NodeId>>` 的 ~700KB 内存放大和热路径 clone）。
    ///
    /// 构建路径：返回 `self.downstreams[idx]` 的引用。
    ///
    /// CSR 布局：`[u32; N+1]` offsets（元素数索引）紧接 `[u32; M]` flat。
    /// offsets[i]..offsets[i+1] 是节点 i 的下游在 flat 中的元素范围。
    #[inline]
    pub fn downstream_slice(&self, idx: usize) -> &[NodeId] {
        if let Some(ref mem) = self.mem {
            let r = mem.section(SectionKind::Downstreams);
            let n = mem.header().node_count as usize;
            // offsets 区：[u32; N+1]，紧跟 flat 区
            let offsets_start = 0;
            let flat_start = (n + 1) * 4;
            let start_elem = rd_u32(r, offsets_start + idx * 4) as usize;
            let end_elem = rd_u32(r, offsets_start + (idx + 1) * 4) as usize;
            let byte_start = flat_start + start_elem * 4;
            let count = end_elem - start_elem;
            // SAFETY: NodeId 是 #[repr(transparent)] over u32（4B，4B 对齐）。
            // Downstreams section 4B 对齐，flat_start = (N+1)*4 是 4 的倍数，
            // byte_start = flat_start + start_elem*4 也是 4 的倍数。
            // 切片范围在 section 边界内（序列化保证 offsets[N] = M = flat 长度）。
            unsafe {
                std::slice::from_raw_parts(r.as_ptr().add(byte_start) as *const NodeId, count)
            }
        } else {
            &self.downstreams[idx]
        }
    }

    // ---- String Pool（zerocopy：加载路径直接引用 mmap，避免 .to_vec() 拷贝）----

    /// 返回字符串池字节切片。
    ///
    /// 加载路径（mem=Some）：直接返回 mmap StringPool section 的 `&[u8]` 切片，
    /// 无堆分配（消除 `.to_vec()` 拷贝，典型节省数 KB）。
    ///
    /// 构建路径（mem=None）：返回 `&self.string_pool[..]`。
    #[inline]
    pub fn string_pool_slice(&self) -> &[u8] {
        if let Some(ref mem) = self.mem {
            mem.string_pool()
        } else {
            &self.string_pool[..]
        }
    }

    // ---- SubGraph 变长字段（zerocopy CSR：消除 per-subgraph Vec 堆分配）----

    /// 返回子图 `sg_idx` 的 upvalue_outer_nodes 切片。
    ///
    /// 加载路径：从 mmap SgUpvalueNodes section 直接返回 `&[NodeId]` 切片，
    /// 无堆分配（消除每个子图的 `Vec<NodeId>` 分配，典型节省 ~56B/subgraph）。
    ///
    /// 构建路径：返回 `self.subgraphs[sg_idx].upvalue_outer_nodes` 的引用。
    #[inline]
    pub fn sg_upvalue_outer_nodes(&self, sg_idx: usize) -> &[NodeId] {
        if let Some(ref mem) = self.mem {
            let (off, len) = self.sg_uv_offsets[sg_idx];
            let r = mem.section(SectionKind::SgUpvalueNodes);
            let byte_start = off as usize;
            let count = len as usize;
            // SAFETY: NodeId is #[repr(transparent)] over u32（4B，4B 对齐）。
            // SgUpvalueNodes section 4B 对齐，offset 是序列化时写入的（4 的倍数）。
            // 切片范围在 section 边界内（序列化保证 offset + count*4 <= section len）。
            unsafe {
                std::slice::from_raw_parts(r.as_ptr().add(byte_start) as *const NodeId, count)
            }
        } else {
            &self.subgraphs[sg_idx].upvalue_outer_nodes
        }
    }

    /// 返回子图 `sg_idx` 的 nested_ranges 切片。
    ///
    /// 加载路径：从 mmap SgNestedRanges section 直接返回 `&[(u32, u32)]` 切片，
    /// 无堆分配（消除每个子图的 `Vec<(u32, u32)>` 分配）。
    ///
    /// 构建路径：返回 `self.subgraphs[sg_idx].nested_ranges` 的引用。
    #[inline]
    pub fn sg_nested_ranges(&self, sg_idx: usize) -> &[(u32, u32)] {
        if let Some(ref mem) = self.mem {
            let (off, len) = self.sg_nr_offsets[sg_idx];
            let r = mem.section(SectionKind::SgNestedRanges);
            let byte_start = off as usize;
            let count = len as usize;
            // SAFETY: (u32, u32) 在 repr(Rust) 下为 8B（两个连续 u32），4B 对齐。
            // SgNestedRanges section 4B 对齐，offset 是 4 的倍数。
            // 序列化时每个元素写入两个 u32（8B），与 (u32, u32) 布局一致。
            unsafe {
                std::slice::from_raw_parts(r.as_ptr().add(byte_start) as *const (u32, u32), count)
            }
        } else {
            &self.subgraphs[sg_idx].nested_ranges
        }
    }

    // ---- 以下表在两条路径都保持 owned，accessor 直接索引 ----
    //（gate_branches / record_lit_infos / select_infos /
    //  trait_construct_infos / record_extend_infos / subgraphs）
    // 这些表不需要 mem 分支，执行路径继续用 graph.field[idx] 直接访问。

    // ---- 5 个复杂变长表 on-demand accessor（zerocopy：消除 Vec<Option<T>> 数组）----

    /// 轻量检查节点 idx 是否有 SelectInfo（热路径，不构造 owned 数据）。
    #[inline]
    pub fn has_select_info(&self, idx: usize) -> bool {
        if self.mem.is_some() {
            self.select_info_offsets.get(idx).map_or(false, |&o| o != u32::MAX)
        } else {
            self.select_infos.get(idx).map_or(false, |v| v.is_some())
        }
    }

    /// 按需解析 GateBranches（加载路径从 mmap section 解析单条目）。
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

    /// 按需解析 RecordLitInfo（加载路径从 mmap section + string_pool 解析）。
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

    /// 按需解析 SelectInfo（加载路径从 mmap section 解析单条目）。
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

    /// 按需解析 TraitConstructInfo（加载路径从 mmap section + string_pool 解析）。
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

    /// 按需解析 RecordExtendInfo（加载路径从 mmap section + string_pool 解析）。
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
