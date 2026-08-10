//! Resin 二进制格式规范层
//!
//! 定义 .resin 文件格式的常量、Header、Section、字符串池、CRC32、
//! 以及 IR enum 与字节之间的映射函数。
//!
//! 本模块只描述"格式是什么"，不包含序列化/反序列化逻辑（见 Format.rs）。

#![allow(non_snake_case)]

use std::collections::HashMap;
use std::io::{self, Read, Write};

use crate::ir::Ir::*;

// ==================== 格式常量 ====================

/// magic number: b"RSN\x00" (Resin 缩写)
pub const RESIN_MAGIC: [u8; 4] = *b"RSN\x00";
/// 格式 schema 版本
pub const RESIN_SCHEMA_VERSION: u16 = 1;
/// runtime ABI 版本（compute_fn 表版本）
pub const RESIN_ABI_VERSION: u16 = 1;
/// compute_fn 数量（用于 ABI 校验）
pub const COMPUTE_FN_COUNT: u32 = 314;

// ==================== Header (64B) ====================

/// .resin 文件头，64 字节定长，little-endian 布局。
///
/// 不使用 `#[repr(C, packed)]`，而是手动 LE 读写，避免 unaligned access。
pub struct ResinHeader {
    pub magic: [u8; 4],
    pub schema_version: u16,
    pub flags: u16,
    pub endianness: u8,       // 1 = LE
    pub pointer_width: u8,    // 8 = 64-bit
    pub abi_version: u16,
    pub node_count: u32,
    pub subgraph_count: u32,
    pub entry_subgraph: u32,  // u32::MAX = None
    pub input_count: u32,
    pub string_pool_len: u32,
    pub global_var_count: u32,
    pub memo_table_count: u32,
    pub compute_fn_count: u32,
    pub crc32: u32,
    pub section_count: u16,
    pub _reserved: [u8; 2],
    pub _padding: [u8; 12],   // 填充至 64B（字段合计 52B + 12B padding = 64B）
}

impl ResinHeader {
    pub const SIZE: usize = 64;

    pub fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&self.magic)?;
        w.write_all(&self.schema_version.to_le_bytes())?;
        w.write_all(&self.flags.to_le_bytes())?;
        w.write_all(&[self.endianness])?;
        w.write_all(&[self.pointer_width])?;
        w.write_all(&self.abi_version.to_le_bytes())?;
        w.write_all(&self.node_count.to_le_bytes())?;
        w.write_all(&self.subgraph_count.to_le_bytes())?;
        w.write_all(&self.entry_subgraph.to_le_bytes())?;
        w.write_all(&self.input_count.to_le_bytes())?;
        w.write_all(&self.string_pool_len.to_le_bytes())?;
        w.write_all(&self.global_var_count.to_le_bytes())?;
        w.write_all(&self.memo_table_count.to_le_bytes())?;
        w.write_all(&self.compute_fn_count.to_le_bytes())?;
        w.write_all(&self.crc32.to_le_bytes())?;
        w.write_all(&self.section_count.to_le_bytes())?;
        w.write_all(&self._reserved)?;
        w.write_all(&self._padding)?;
        Ok(())
    }

    pub fn read_from(r: &mut impl Read) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != RESIN_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        let mut buf2 = [0u8; 2];
        let mut buf4 = [0u8; 4];
        let mut buf1 = [0u8; 1];

        r.read_exact(&mut buf2)?;
        let schema_version = u16::from_le_bytes(buf2);
        r.read_exact(&mut buf2)?;
        let flags = u16::from_le_bytes(buf2);
        r.read_exact(&mut buf1)?;
        let endianness = buf1[0];
        r.read_exact(&mut buf1)?;
        let pointer_width = buf1[0];
        r.read_exact(&mut buf2)?;
        let abi_version = u16::from_le_bytes(buf2);
        r.read_exact(&mut buf4)?;
        let node_count = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let subgraph_count = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let entry_subgraph = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let input_count = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let string_pool_len = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let global_var_count = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let memo_table_count = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let compute_fn_count = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let crc32 = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf2)?;
        let section_count = u16::from_le_bytes(buf2);
        let mut _reserved = [0u8; 2];
        r.read_exact(&mut _reserved)?;
        let mut _padding = [0u8; 12];
        r.read_exact(&mut _padding)?;

        Ok(Self {
            magic, schema_version, flags, endianness, pointer_width, abi_version,
            node_count, subgraph_count, entry_subgraph, input_count, string_pool_len,
            global_var_count, memo_table_count, compute_fn_count, crc32, section_count, _reserved,
            _padding,
        })
    }
}

// ==================== Section ====================

/// Section 种类枚举
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::IntoStaticStr, num_enum::TryFromPrimitive)]
pub enum SectionKind {
    Nodes = 0,
    Inputs = 1,
    SubGraphs = 2,
    // SubGraph 变长区
    SgUpvalueNodes = 3,
    SgNestedRanges = 4,
    SgEventDecls = 5,
    SgDeferEntries = 6,
    SgDeferCapturedInputs = 7,
    SgResetPlan = 8,
    // per-Node 定宽标量表 (类别 A)
    CallTargets = 10,
    FieldAccessInfos = 11,
    VtableCallMethods = 12,
    AwaitEventSources = 13,
    WritebackTargets = 14,
    HoistedOwners = 15,
    GlobalLoadSlots = 16,
    GlobalStoreSlots = 17,
    PatternFieldIndices = 18,
    ClosureCallArgCounts = 19,
    // per-Node 布尔表 (类别 B)
    TailCallFlags = 20,
    SafeOpFlags = 21,
    HoistedNode = 22,
    SliceInclusive = 23,
    // per-Node 含字符串表 (类别 C)
    FfiCallNames = 30,
    FieldSetNames = 31,
    PatternCtorNames = 32,
    CastTargetTypes = 33,
    // per-Node 含定宽复合表 (类别 D)
    ClosureInfos = 40,
    PartialInfos = 41,
    LazyConstructInfos = 42,
    MemoInfos = 43,
    // per-Node 含变长字段表 (类别 E)
    ConstValues = 50,
    GateBranches = 51,
    RecordLitInfos = 52,
    SelectInfos = 53,
    TraitConstructInfos = 54,
    RecordExtendInfos = 55,
    BatchInfos = 56,
    // 共享区
    StringPool = 60,
    Downstreams = 61,
}

impl SectionKind {
    /// 返回 section 的可读名字（strum::IntoStaticStr 自动生成）
    pub fn name(self) -> &'static str {
        self.into()
    }

    /// 从 u8 转 SectionKind（num_enum::TryFromPrimitive 自动生成）
    pub fn from_u8(v: u8) -> Option<Self> {
        <Self as num_enum::TryFromPrimitive>::try_from_primitive(v).ok()
    }
}

/// Section 索引条目
pub struct SectionEntry {
    pub kind: u8,
    pub offset: u32,
    pub len: u32,
}

// ==================== LE 读写辅助 ====================

pub fn write_u8(w: &mut Vec<u8>, v: u8) { w.push(v); }
pub fn write_u16(w: &mut Vec<u8>, v: u16) { w.extend_from_slice(&v.to_le_bytes()); }
pub fn write_u32(w: &mut Vec<u8>, v: u32) { w.extend_from_slice(&v.to_le_bytes()); }
pub fn write_i32(w: &mut Vec<u8>, v: i32) { w.extend_from_slice(&v.to_le_bytes()); }
pub fn write_bytes(w: &mut Vec<u8>, v: &[u8]) { w.extend_from_slice(v); }

pub fn read_u8(r: &mut &[u8]) -> u8 {
    let v = r[0]; *r = &r[1..]; v
}
pub fn read_u32(r: &mut &[u8]) -> u32 {
    let v = u32::from_le_bytes([r[0], r[1], r[2], r[3]]); *r = &r[4..]; v
}
pub fn read_i32(r: &mut &[u8]) -> i32 {
    let v = i32::from_le_bytes([r[0], r[1], r[2], r[3]]); *r = &r[4..]; v
}
pub fn read_bytes<'a>(r: &mut &'a [u8], n: usize) -> &'a [u8] {
    let v = &r[..n]; *r = &r[n..]; v
}

// ==================== GraphMemory ====================

/// backing 存储：owned Vec 或 mmap 映射
pub enum GraphBacking {
    /// owned 字节（fs::read 路径 / 测试用）
    Owned(Vec<u8>),
    /// mmap 映射（文件加载路径，零拷贝读取）
    Mapped(memmap2::Mmap),
}

impl std::ops::Deref for GraphBacking {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            GraphBacking::Owned(v) => v,
            GraphBacking::Mapped(m) => m,
        }
    }
}

/// .resin 文件解析后的内存表示，持有 backing 字节所有权。
///
/// 支持两种 backing：
/// - `Owned(Vec<u8>)`：fs::read 路径，用于测试或小文件
/// - `Mapped(Mmap)`：mmap 路径，文件直接映射到地址空间，零拷贝读取
///
/// 封装 header 解析 + CRC 校验 + section index，提供 `section(kind) -> &[u8]` 切片访问。
/// 两种 backing 的 `section()` 接口完全一致，调用方无感知。
pub struct GraphMemory {
    backing: GraphBacking,
    header: ResinHeader,
    /// kind → (offset, len)，offset 相对 backing 起始
    sections: HashMap<u8, (u32, u32)>,
}

impl GraphMemory {
    /// 从 owned 字节解析：校验 magic/version/CRC，解析 section index
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        Self::from_backing(GraphBacking::Owned(data.to_vec()))
    }

    /// 从文件 mmap 加载：映射文件 → 校验 → 解析 section index
    pub fn from_file(path: &str) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: memmap2 的 unsafe { Mmap::map(&file) } 映射文件到只读内存。
        // 安全前提：文件内容不会被外部进程修改（.resin 是编译产物，分发后不可变）。
        // 如果文件被修改，CRC 校验会拒绝加载。
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Self::from_backing(GraphBacking::Mapped(mmap))
    }

    /// 从 backing 字节解析（共用逻辑）
    fn from_backing(backing: GraphBacking) -> io::Result<Self> {
        let data: &[u8] = &backing;
        if data.len() < ResinHeader::SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
        }

        // 解析 Header
        let mut r = data;
        let header = ResinHeader::read_from(&mut r)?;

        if header.magic != RESIN_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        if header.schema_version != RESIN_SCHEMA_VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("schema version mismatch: file={}, expected={}", header.schema_version, RESIN_SCHEMA_VERSION)));
        }
        if header.abi_version != RESIN_ABI_VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("ABI version mismatch: file={}, expected={}", header.abi_version, RESIN_ABI_VERSION)));
        }
        if header.compute_fn_count != COMPUTE_FN_COUNT {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("compute_fn count mismatch: file={}, expected={}", header.compute_fn_count, COMPUTE_FN_COUNT)));
        }

        // CRC 校验（header 之后的所有字节）
        let body_start = ResinHeader::SIZE;
        let body = &data[body_start..];
        let computed_crc = crc32(body);
        if computed_crc != header.crc32 {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("CRC mismatch: file=0x{:08X}, computed=0x{:08X}", header.crc32, computed_crc)));
        }

        // 解析 Section Index
        let mut sections: HashMap<u8, (u32, u32)> = HashMap::new();
        let mut sec_r = &data[body_start..];
        for _ in 0..header.section_count {
            let kind = read_u8(&mut sec_r);
            let offset = read_u32(&mut sec_r);
            let len = read_u32(&mut sec_r);
            sections.insert(kind, (offset, len));
        }

        Ok(Self { backing, header, sections })
    }

    /// 获取 Header
    pub fn header(&self) -> &ResinHeader { &self.header }

    /// 获取指定 section 的字节切片
    pub fn section(&self, kind: SectionKind) -> &[u8] {
        let (off, len) = self.sections[&(kind as u8)];
        &self.backing[off as usize..(off + len) as usize]
    }

    /// 字符串池切片
    pub fn string_pool(&self) -> &[u8] {
        self.section(SectionKind::StringPool)
    }

    /// 从字符串池读取字符串（offset 相对于 StringPool section 起始）
    pub fn read_str(&self, offset: u32, len: u32) -> String {
        let pool = self.string_pool();
        read_str(pool, offset, len)
    }

    /// 返回所有 section 的详情（kind_u8, offset, len），按 kind 排序
    pub fn sections_detail(&self) -> Vec<(u8, u32, u32)> {
        let mut v: Vec<(u8, u32, u32)> = self.sections.iter()
            .map(|(&k, &(o, l))| (k, o, l))
            .collect();
        v.sort_by_key(|&(k, _, _)| k);
        v
    }
}

// ==================== String Pool ====================

/// 字符串池：收集所有 String 字段，连续存储，引用为 (offset, len)
pub struct StringPool {
    pub data: Vec<u8>,
    /// 去重映射：String → offset
    map: std::collections::HashMap<String, u32>,
}

impl StringPool {
    pub fn new() -> Self {
        Self { data: Vec::new(), map: std::collections::HashMap::new() }
    }

    /// 添加字符串，返回 (offset, len)
    pub fn add(&mut self, s: &str) -> (u32, u32) {
        if let Some(&off) = self.map.get(s) {
            return (off, s.len() as u32);
        }
        let offset = self.data.len() as u32;
        self.data.extend_from_slice(s.as_bytes());
        self.map.insert(s.to_string(), offset);
        (offset, s.len() as u32)
    }

    /// 添加 Option<String>，None 返回 (u32::MAX, 0)
    pub fn add_opt(&mut self, s: &Option<String>) -> (u32, u32) {
        match s {
            None => (u32::MAX, 0),
            Some(s) => self.add(s),
        }
    }

    pub fn len(&self) -> u32 {
        self.data.len() as u32
    }
}

/// 从 string pool 读取字符串
pub fn read_str(pool: &[u8], offset: u32, len: u32) -> String {
    if offset == u32::MAX {
        return String::new();
    }
    let off = offset as usize;
    let end = off + len as usize;
    String::from_utf8_lossy(&pool[off..end]).into_owned()
}

// ==================== enum 序列化辅助（num_enum 驱动，消除手写 match）====================

/// NodeKind → u8（#[repr(u8)] 直接 as u8）
pub fn node_kind_to_u8(k: NodeKind) -> u8 { k as u8 }
pub fn u8_to_node_kind(v: u8) -> NodeKind {
    <NodeKind as num_enum::TryFromPrimitive>::try_from_primitive(v)
        .unwrap_or_else(|_| panic!("invalid NodeKind: {}", v))
}

/// LoopKind → u8
pub fn loop_kind_to_u8(k: LoopKind) -> u8 { k as u8 }
pub fn u8_to_loop_kind(v: u8) -> LoopKind {
    <LoopKind as num_enum::TryFromPrimitive>::try_from_primitive(v)
        .unwrap_or_else(|_| panic!("invalid LoopKind: {}", v))
}

/// EventSourceKind → u8
pub fn event_kind_to_u8(k: EventSourceKind) -> u8 { k as u8 }
pub fn u8_to_event_kind(v: u8) -> EventSourceKind {
    <EventSourceKind as num_enum::TryFromPrimitive>::try_from_primitive(v)
        .unwrap_or_else(|_| panic!("invalid EventSourceKind: {}", v))
}

/// RecordLitKind → u8
pub fn record_lit_kind_to_u8(k: RecordLitKind) -> u8 { k as u8 }
pub fn u8_to_record_lit_kind(v: u8) -> RecordLitKind {
    <RecordLitKind as num_enum::TryFromPrimitive>::try_from_primitive(v)
        .unwrap_or_else(|_| panic!("invalid RecordLitKind: {}", v))
}

/// ConstValue → tag u8
pub fn const_tag_to_u8(c: &ConstValue) -> u8 {
    match c {
        ConstValue::I8(_) => 1,
        ConstValue::I16(_) => 2,
        ConstValue::I32(_) => 3,
        ConstValue::I64(_) => 4,
        ConstValue::I128(_) => 5,
        ConstValue::U8(_) => 6,
        ConstValue::U16(_) => 7,
        ConstValue::U32(_) => 8,
        ConstValue::U64(_) => 9,
        ConstValue::U128(_) => 10,
        ConstValue::Isize(_) => 11,
        ConstValue::Usize(_) => 12,
        ConstValue::F32(_) => 13,
        ConstValue::F64(_) => 14,
        ConstValue::F16(_) => 15,
        ConstValue::F128(_) => 16,
        ConstValue::Bool(_) => 17,
        ConstValue::Char(_) => 18,
        ConstValue::Str { .. } => 19,
        ConstValue::Null => 20,
        ConstValue::Void => 21,
    }
}

// ==================== BatchInfo 序列化 ====================

/// BatchInfo 含 ValueTag + BatchOp，需展平为字节
pub fn batch_info_to_bytes(bi: &BatchInfo) -> [u8; 4] {
    let tag = bi.tag as u8;
    let (op_kind, op_value) = match &bi.op {
        BatchOp::Bin(b) => (0u8, bin_op_to_u8(b)),
        BatchOp::Cmp(c) => (1u8, cmp_op_to_u8(c)),
        BatchOp::Unary(u) => (2u8, unary_op_to_u8(u)),
    };
    [tag, op_kind, op_value, 0]
}

pub fn bytes_to_batch_info(buf: &[u8; 4]) -> BatchInfo {
    let tag = u8_to_value_tag(buf[0]);
    let op = match buf[1] {
        0 => BatchOp::Bin(u8_to_bin_op(buf[2])),
        1 => BatchOp::Cmp(u8_to_cmp_op(buf[2])),
        2 => BatchOp::Unary(u8_to_unary_op(buf[2])),
        _ => panic!("invalid BatchOp kind: {}", buf[1]),
    };
    BatchInfo { tag, op }
}

pub fn bin_op_to_u8(op: &crate::value::BinOp) -> u8 { *op as u8 }
pub fn u8_to_bin_op(v: u8) -> crate::value::BinOp {
    <crate::value::BinOp as num_enum::TryFromPrimitive>::try_from_primitive(v)
        .unwrap_or_else(|_| panic!("invalid BinOp: {}", v))
}
pub fn cmp_op_to_u8(op: &crate::value::CmpOp) -> u8 { *op as u8 }
pub fn u8_to_cmp_op(v: u8) -> crate::value::CmpOp {
    <crate::value::CmpOp as num_enum::TryFromPrimitive>::try_from_primitive(v)
        .unwrap_or_else(|_| panic!("invalid CmpOp: {}", v))
}
pub fn unary_op_to_u8(op: &crate::value::UnaryOp) -> u8 { *op as u8 }
pub fn u8_to_unary_op(v: u8) -> crate::value::UnaryOp {
    <crate::value::UnaryOp as num_enum::TryFromPrimitive>::try_from_primitive(v)
        .unwrap_or_else(|_| panic!("invalid UnaryOp: {}", v))
}
pub fn u8_to_value_tag(v: u8) -> crate::value::ValueTag {
    <crate::value::ValueTag as num_enum::TryFromPrimitive>::try_from_primitive(v)
        .unwrap_or_else(|_| panic!("invalid ValueTag: {}", v))
}

// ==================== TraitMethodEntry 序列化 ====================

pub fn trait_method_entry_to_bytes(e: &TraitMethodEntry) -> [u8; 8] {
    let mut buf = [0u8; 8];
    buf[0..4].copy_from_slice(&e.subgraph_id.0.to_le_bytes());
    buf[4] = e.arity;
    buf[5] = e.upvalue_count;
    buf
}

pub fn bytes_to_trait_method_entry(buf: &[u8; 8]) -> TraitMethodEntry {
    let subgraph_id = SubGraphId(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]));
    let arity = buf[4];
    let upvalue_count = buf[5];
    TraitMethodEntry { subgraph_id, arity, upvalue_count }
}

// ==================== CRC32 ====================

/// 简单 CRC32 实现（IEEE 802.3 多项式）
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}
