//! Solidify binary format spec layer.
//!
//! Defines the constants, Header, Section, string pool, and CRC32 of the `.kzo`
//! file format, plus the mapping functions between IR enums and bytes.
//!
//! This module only describes "what the format is"; serialization/deserialization
//! logic lives in `Format.rs`.

#![allow(non_snake_case)]

use std::collections::HashMap;
use std::io::{self, Read, Write};

use crate::ir::Ir::*;

// ==================== Format constants ====================

/// Magic number: `b"KZO\x00"` (Kuzo abbreviation).
pub const SOLIDIFY_MAGIC: [u8; 4] = *b"KZO\x00";
/// Format schema version.
pub const SOLIDIFY_SCHEMA_VERSION: u16 = 1;
/// Runtime ABI version (compute_fn table version).
pub const SOLIDIFY_ABI_VERSION: u16 = 1;
/// Number of compute_fn entries (used for ABI validation).
///
/// Derived from the actual table length (`ir::Ir::COMPUTE_FN_TABLE_LEN`,
/// asserted at table build) — was previously a hand-maintained `314` that had
/// drifted 23 entries behind the real table, defeating the load-time check.
/// `.kzo` files written by binaries with a different count are rejected;
/// rebuilding the source regenerates them.
pub const COMPUTE_FN_COUNT: u32 = crate::ir::Ir::COMPUTE_FN_TABLE_LEN;

// ==================== Header (64B) ====================

/// `.kzo` file header: a fixed 64-byte, little-endian layout.
///
/// Does not use `#[repr(C, packed)]`; instead fields are read/written manually as
/// LE bytes to avoid unaligned access.
pub struct SolidifyHeader {
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
    pub _padding: [u8; 12],   // Padded to 64B (fields total 52B + 12B padding = 64B)
}

impl SolidifyHeader {
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
        if magic != SOLIDIFY_MAGIC {
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

/// Section kind enum.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::IntoStaticStr, num_enum::TryFromPrimitive)]
pub enum SectionKind {
    Nodes = 0,
    Inputs = 1,
    SubGraphs = 2,
    // SubGraph variable-length region
    SgUpvalueNodes = 3,
    SgNestedRanges = 4,
    SgEventDecls = 5,
    SgDeferEntries = 6,
    SgDeferCapturedInputs = 7,
    SgResetPlan = 8,
    // per-Node fixed-width scalar tables (category A)
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
    // per-Node boolean tables (category B)
    TailCallFlags = 20,
    SafeOpFlags = 21,
    HoistedNode = 22,
    SliceInclusive = 23,
    // per-Node tables with strings (category C)
    FfiCallNames = 30,
    FieldSetNames = 31,
    PatternCtorNames = 32,
    PatternTypeNames = 34,
    CastTargetTypes = 33,
    // per-Node fixed-width composite tables (category D)
    ClosureInfos = 40,
    PartialInfos = 41,
    LazyConstructInfos = 42,
    MemoInfos = 43,
    // per-Node variable-length field tables (category E)
    ConstValues = 50,
    GateBranches = 51,
    RecordLitInfos = 52,
    SelectInfos = 53,
    TraitConstructInfos = 54,
    RecordExtendInfos = 55,
    BatchInfos = 56,
    // Shared region
    StringPool = 60,
    Downstreams = 61,
    // Inline C FFI (compiled by kuzo build → cc → object extraction)
    CMachineCode = 70,
    CSymbols = 71,
}

impl SectionKind {
    /// Returns the human-readable name of the section (auto-generated by `strum::IntoStaticStr`).
    pub fn name(self) -> &'static str {
        self.into()
    }

    /// Converts a `u8` to a `SectionKind` (auto-generated by `num_enum::TryFromPrimitive`).
    pub fn from_u8(v: u8) -> Option<Self> {
        <Self as num_enum::TryFromPrimitive>::try_from_primitive(v).ok()
    }
}

/// Section index entry.
pub struct SectionEntry {
    pub kind: u8,
    pub offset: u32,
    pub len: u32,
}

// ==================== LE read/write helpers ====================

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

/// Backing storage: an owned `Vec` or an `mmap` mapping.
pub enum GraphBacking {
    /// Owned bytes (the `fs::read` path, used for tests).
    Owned(Vec<u8>),
    /// mmap mapping (the file-loading path, zero-copy reads).
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

/// In-memory representation of a parsed `.kzo` file, owning the backing bytes.
///
/// Supports two backing kinds:
/// - `Owned(Vec<u8>)`: the `fs::read` path, used for tests or small files.
/// - `Mapped(Mmap)`: the mmap path; the file is mapped directly into the address
///   space for zero-copy reads.
///
/// Wraps header parsing, CRC validation, and the section index, exposing
/// `section(kind) -> &[u8]` slice access. The `section()` interface is identical
/// for both backings, so callers are unaware of which is in use.
pub struct GraphMemory {
    backing: GraphBacking,
    header: SolidifyHeader,
    /// kind -> (offset, len); offset is relative to the start of the backing.
    sections: HashMap<u8, (u32, u32)>,
}

impl GraphMemory {
    /// Parses from owned bytes: validates magic/version/CRC and parses the section index.
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        Self::from_backing(GraphBacking::Owned(data.to_vec()))
    }

    /// Loads from a file via mmap: maps the file -> validates -> parses the section index.
    pub fn from_file(path: &str) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: memmap2's `unsafe { Mmap::map(&file) }` maps the file into read-only memory.
        // Safety precondition: the file contents will not be modified by an external process
        // (`.kzo` is a compiled artifact and immutable once distributed).
        // If the file is tampered with, the CRC check will reject the load.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Self::from_backing(GraphBacking::Mapped(mmap))
    }

    /// Parses from backing bytes (shared logic).
    fn from_backing(backing: GraphBacking) -> io::Result<Self> {
        let data: &[u8] = &backing;
        if data.len() < SolidifyHeader::SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
        }

        // Parse the header.
        let mut r = data;
        let header = SolidifyHeader::read_from(&mut r)?;

        if header.magic != SOLIDIFY_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        if header.schema_version != SOLIDIFY_SCHEMA_VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("schema version mismatch: file={}, expected={}", header.schema_version, SOLIDIFY_SCHEMA_VERSION)));
        }
        if header.abi_version != SOLIDIFY_ABI_VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("ABI version mismatch: file={}, expected={}", header.abi_version, SOLIDIFY_ABI_VERSION)));
        }
        if header.compute_fn_count != COMPUTE_FN_COUNT {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("compute_fn count mismatch: file={}, expected={}", header.compute_fn_count, COMPUTE_FN_COUNT)));
        }

        // CRC validation (all bytes after the header).
        let body_start = SolidifyHeader::SIZE;
        let body = &data[body_start..];
        let computed_crc = crc32(body);
        if computed_crc != header.crc32 {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("CRC mismatch: file=0x{:08X}, computed=0x{:08X}", header.crc32, computed_crc)));
        }

        // Parse the section index.
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

    /// Returns the header.
    pub fn header(&self) -> &SolidifyHeader { &self.header }

    /// Returns the byte slice of the specified section.
    pub fn section(&self, kind: SectionKind) -> &[u8] {
        let (off, len) = self.sections[&(kind as u8)];
        &self.backing[off as usize..(off + len) as usize]
    }

    /// Returns the string pool slice.
    pub fn string_pool(&self) -> &[u8] {
        self.section(SectionKind::StringPool)
    }

    /// Reads a string from the string pool (offset is relative to the start of the StringPool section).
    pub fn read_str(&self, offset: u32, len: u32) -> String {
        let pool = self.string_pool();
        read_str(pool, offset, len)
    }

    /// Returns details of all sections `(kind_u8, offset, len)`, sorted by kind.
    pub fn sections_detail(&self) -> Vec<(u8, u32, u32)> {
        let mut v: Vec<(u8, u32, u32)> = self.sections.iter()
            .map(|(&k, &(o, l))| (k, o, l))
            .collect();
        v.sort_by_key(|&(k, _, _)| k);
        v
    }
}

// ==================== String Pool ====================

/// String pool: collects all `String` fields into contiguous storage, referenced by `(offset, len)`.
pub struct StringPool {
    pub data: Vec<u8>,
    /// Deduplication map: String -> offset.
    map: std::collections::HashMap<String, u32>,
}

impl StringPool {
    pub fn new() -> Self {
        Self { data: Vec::new(), map: std::collections::HashMap::new() }
    }

    /// Adds a string and returns `(offset, len)`.
    pub fn add(&mut self, s: &str) -> (u32, u32) {
        if let Some(&off) = self.map.get(s) {
            return (off, s.len() as u32);
        }
        let offset = self.data.len() as u32;
        self.data.extend_from_slice(s.as_bytes());
        self.map.insert(s.to_string(), offset);
        (offset, s.len() as u32)
    }

    /// Adds an `Option<String>`; `None` returns `(u32::MAX, 0)`.
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

/// Reads a string from the string pool.
pub fn read_str(pool: &[u8], offset: u32, len: u32) -> String {
    if offset == u32::MAX {
        return String::new();
    }
    let off = offset as usize;
    let end = off + len as usize;
    String::from_utf8_lossy(&pool[off..end]).into_owned()
}

// ==================== Enum serialization helpers (num_enum-driven, no hand-written match) ====================

/// `NodeKind` -> u8 (via `#[repr(u8)]`, directly `as u8`).
pub fn node_kind_to_u8(k: NodeKind) -> u8 { k as u8 }
pub fn u8_to_node_kind(v: u8) -> NodeKind {
    <NodeKind as num_enum::TryFromPrimitive>::try_from_primitive(v)
        .unwrap_or_else(|_| panic!("invalid NodeKind: {}", v))
}

/// `LoopKind` -> u8.
pub fn loop_kind_to_u8(k: LoopKind) -> u8 { k as u8 }
pub fn u8_to_loop_kind(v: u8) -> LoopKind {
    <LoopKind as num_enum::TryFromPrimitive>::try_from_primitive(v)
        .unwrap_or_else(|_| panic!("invalid LoopKind: {}", v))
}

/// `EventSourceKind` -> u8.
pub fn event_kind_to_u8(k: EventSourceKind) -> u8 { k as u8 }
pub fn u8_to_event_kind(v: u8) -> EventSourceKind {
    <EventSourceKind as num_enum::TryFromPrimitive>::try_from_primitive(v)
        .unwrap_or_else(|_| panic!("invalid EventSourceKind: {}", v))
}

/// `RecordLitKind` -> u8.
pub fn record_lit_kind_to_u8(k: RecordLitKind) -> u8 { k as u8 }
pub fn u8_to_record_lit_kind(v: u8) -> RecordLitKind {
    <RecordLitKind as num_enum::TryFromPrimitive>::try_from_primitive(v)
        .unwrap_or_else(|_| panic!("invalid RecordLitKind: {}", v))
}

/// `ConstValue` -> tag u8.
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

// ==================== BatchInfo serialization ====================

/// `BatchInfo` contains a `ValueTag` + `BatchOp` and must be flattened into bytes.
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

// ==================== TraitMethodEntry serialization ====================

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

/// Simple CRC32 implementation (IEEE 802.3 polynomial).
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
