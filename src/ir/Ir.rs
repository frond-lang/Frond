//! Ir.rs — Core IR data structures for the dataflow-ready scheduling execution model.
//!
//! Produced from the `SemaResult` in Sema.rs, this module defines:
//! - `Node` (fixed 16B, stores topological references, not values)
//! - `InputsPool` (a standalone contiguous input pool)
//! - `ValueTable` / `Frame` (runtime value tables, SoA layout)
//! - `SubGraph` (function = subgraph)
//! - `EventSource` (declarations for channel/async/timer/subgraph-completion event sources)
//! - `DataFlowGraph` (the global graph container)
//! - `ComputeFn` (a compute function index bound at build time, eliminating dispatch)
//!
//! Design principles (see docs/superpowers/specs/2026-07-31-dataflow-engine-design.md):
//! - Nodes are fixed 16B and store only topological references; the output is implicitly the node's own id.
//! - `kind` has 9 variants, used solely for scheduler readiness checks, not for operation dispatch.
//! - `compute_fn` is a function index bound at build time via type specialization, looked up at runtime by array index.
//! - Value table slots use the `Value` enum from Value.rs (scalars and `Arc<HeapObj>` references).
//! - The standalone input pool stores all node inputs contiguously for cache friendliness.

use crate::value::Value;
use std::sync::Arc;

// =========================================================================
// Index newtypes — type-safe handles
// =========================================================================

/// Node id (globally contiguous; the value table is indexed by this).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// Subgraph id (function = subgraph).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubGraphId(pub u32);

/// Function id (one-to-one with `SubGraphId`; a semantic alias).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncId(pub u32);

/// Subgraph instance id (runtime; one instance per invocation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubgraphInstanceId(pub u32);

/// Frame id (runtime).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameId(pub u32);

/// Compute function index (points into `COMPUTE_FN_TABLE`).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeFnId(pub u32);

/// Macro generating named `ComputeFnId` constants, corresponding one-to-one with `build_compute_fn_table()` indices.
macro_rules! compute_fn_ids {
    ( $( $idx:literal => $name:ident ),* $(,)? ) => {
        $(
            pub const $name: ComputeFnId = ComputeFnId($idx);
        )*
    };
}

compute_fn_ids! {
    0 => CF_NOOP,
    1 => CF_ADD_I32,
    2 => CF_ADD_F64,
    3 => CF_MUL_I32,
    4 => CF_LE_I32,
    5 => CF_SUB_I32,
    6 => CF_DIV_I32,
    7 => CF_MOD_I32,
    8 => CF_EQ_I32,
    9 => CF_NE_I32,
    10 => CF_LT_I32,
    11 => CF_GT_I32,
    12 => CF_GE_I32,
    13 => CF_SUB_F64,
    14 => CF_MUL_F64,
    15 => CF_DIV_F64,
    16 => CF_EQ_F64,
    17 => CF_NE_F64,
    18 => CF_LT_F64,
    19 => CF_GT_F64,
    20 => CF_LE_F64,
    21 => CF_GE_F64,
    22 => CF_AND_BOOL,
    23 => CF_OR_BOOL,
    24 => CF_NOT_BOOL,
    25 => CF_NEG_I32,
    26 => CF_NEG_F64,
    27 => CF_EQ_BOOL,
    28 => CF_THROW_WRAP_ERR,
    29 => CF_RECORD_CONSTRUCT,
    30 => CF_RECORD_FIELD_GET,
    31 => CF_ARRAY_CONSTRUCT,
    32 => CF_ARRAY_INDEX,
    33 => CF_RECORD_FIELD_SET,
    34 => CF_IS_NULL,
    35 => CF_ARRAY_LEN,
    36 => CF_CALL_LAUNCH,
    37 => CF_GATE_LAUNCH,
    38 => CF_AWAIT,
    39 => CF_ASYNC_CALL_LAUNCH,
    40 => CF_CLOSURE_CONSTRUCT,
    41 => CF_CLOSURE_CALL,
    42 => CF_CANCEL_ASYNC_HANDLE,
    43 => CF_SELECT_GATE,
    44 => CF_THROW_OK,
    45 => CF_THROW_ERR,
    46 => CF_FFI_CALL,
    47 => CF_PROPAGATE,
    48 => CF_SEQ,
    49 => CF_WRITEBACK,
    // i64 arithmetic and comparison (50-63)
    50 => CF_ADD_I64,
    51 => CF_SUB_I64,
    52 => CF_MUL_I64,
    53 => CF_DIV_I64,
    54 => CF_MOD_I64,
    55 => CF_EQ_I64,
    56 => CF_NE_I64,
    57 => CF_LT_I64,
    58 => CF_GT_I64,
    59 => CF_LE_I64,
    60 => CF_GE_I64,
    61 => CF_NEG_I64,
    62 => CF_BITNOT_I32,
    63 => CF_BITNOT_I64,
    // i128 arithmetic and comparison (64-77)
    64 => CF_ADD_I128,
    65 => CF_SUB_I128,
    66 => CF_MUL_I128,
    67 => CF_DIV_I128,
    68 => CF_MOD_I128,
    69 => CF_EQ_I128,
    70 => CF_NE_I128,
    71 => CF_LT_I128,
    72 => CF_GT_I128,
    73 => CF_LE_I128,
    74 => CF_GE_I128,
    75 => CF_NEG_I128,
    76 => CF_BITNOT_I128,
    // Integer bitwise operations (77-91)
    77 => CF_BITAND_I32,
    78 => CF_BITOR_I32,
    79 => CF_BITXOR_I32,
    80 => CF_BITAND_I64,
    81 => CF_BITOR_I64,
    82 => CF_BITXOR_I64,
    83 => CF_BITAND_I128,
    84 => CF_BITOR_I128,
    85 => CF_BITXOR_I128,
    86 => CF_SHL_I32,
    87 => CF_SHR_I32,
    88 => CF_SHL_I64,
    89 => CF_SHR_I64,
    90 => CF_SHL_I128,
    91 => CF_SHR_I128,
    // compute_fn for all primitive types (92-259)
    // i8: 92-103
    92 => CF_ADD_I8,
    93 => CF_SUB_I8,
    94 => CF_MUL_I8,
    95 => CF_DIV_I8,
    96 => CF_MOD_I8,
    97 => CF_BITAND_I8,
    98 => CF_BITOR_I8,
    99 => CF_BITXOR_I8,
    100 => CF_SHL_I8,
    101 => CF_SHR_I8,
    102 => CF_NEG_I8,
    103 => CF_BITNOT_I8,
    // i16: 104-115
    104 => CF_ADD_I16,
    105 => CF_SUB_I16,
    106 => CF_MUL_I16,
    107 => CF_DIV_I16,
    108 => CF_MOD_I16,
    109 => CF_BITAND_I16,
    110 => CF_BITOR_I16,
    111 => CF_BITXOR_I16,
    112 => CF_SHL_I16,
    113 => CF_SHR_I16,
    114 => CF_NEG_I16,
    115 => CF_BITNOT_I16,
    // i32: 116-127
    116 => CF_ADD_I32_FULL,
    117 => CF_SUB_I32_FULL,
    118 => CF_MUL_I32_FULL,
    119 => CF_DIV_I32_FULL,
    120 => CF_MOD_I32_FULL,
    121 => CF_BITAND_I32_FULL,
    122 => CF_BITOR_I32_FULL,
    123 => CF_BITXOR_I32_FULL,
    124 => CF_SHL_I32_FULL,
    125 => CF_SHR_I32_FULL,
    126 => CF_NEG_I32_FULL,
    127 => CF_BITNOT_I32_FULL,
    // i64: 128-139
    128 => CF_ADD_I64_FULL,
    129 => CF_SUB_I64_FULL,
    130 => CF_MUL_I64_FULL,
    131 => CF_DIV_I64_FULL,
    132 => CF_MOD_I64_FULL,
    133 => CF_BITAND_I64_FULL,
    134 => CF_BITOR_I64_FULL,
    135 => CF_BITXOR_I64_FULL,
    136 => CF_SHL_I64_FULL,
    137 => CF_SHR_I64_FULL,
    138 => CF_NEG_I64_FULL,
    139 => CF_BITNOT_I64_FULL,
    // i128: 140-151
    140 => CF_ADD_I128_FULL,
    141 => CF_SUB_I128_FULL,
    142 => CF_MUL_I128_FULL,
    143 => CF_DIV_I128_FULL,
    144 => CF_MOD_I128_FULL,
    145 => CF_BITAND_I128_FULL,
    146 => CF_BITOR_I128_FULL,
    147 => CF_BITXOR_I128_FULL,
    148 => CF_SHL_I128_FULL,
    149 => CF_SHR_I128_FULL,
    150 => CF_NEG_I128_FULL,
    151 => CF_BITNOT_I128_FULL,
    // u8: 152-163
    152 => CF_ADD_U8,
    153 => CF_SUB_U8,
    154 => CF_MUL_U8,
    155 => CF_DIV_U8,
    156 => CF_MOD_U8,
    157 => CF_BITAND_U8,
    158 => CF_BITOR_U8,
    159 => CF_BITXOR_U8,
    160 => CF_SHL_U8,
    161 => CF_SHR_U8,
    162 => CF_NEG_U8,
    163 => CF_BITNOT_U8,
    // u16: 164-175
    164 => CF_ADD_U16,
    165 => CF_SUB_U16,
    166 => CF_MUL_U16,
    167 => CF_DIV_U16,
    168 => CF_MOD_U16,
    169 => CF_BITAND_U16,
    170 => CF_BITOR_U16,
    171 => CF_BITXOR_U16,
    172 => CF_SHL_U16,
    173 => CF_SHR_U16,
    174 => CF_NEG_U16,
    175 => CF_BITNOT_U16,
    // u32: 176-187
    176 => CF_ADD_U32,
    177 => CF_SUB_U32,
    178 => CF_MUL_U32,
    179 => CF_DIV_U32,
    180 => CF_MOD_U32,
    181 => CF_BITAND_U32,
    182 => CF_BITOR_U32,
    183 => CF_BITXOR_U32,
    184 => CF_SHL_U32,
    185 => CF_SHR_U32,
    186 => CF_NEG_U32,
    187 => CF_BITNOT_U32,
    // u64: 188-199
    188 => CF_ADD_U64,
    189 => CF_SUB_U64,
    190 => CF_MUL_U64,
    191 => CF_DIV_U64,
    192 => CF_MOD_U64,
    193 => CF_BITAND_U64,
    194 => CF_BITOR_U64,
    195 => CF_BITXOR_U64,
    196 => CF_SHL_U64,
    197 => CF_SHR_U64,
    198 => CF_NEG_U64,
    199 => CF_BITNOT_U64,
    // u128: 200-211
    200 => CF_ADD_U128,
    201 => CF_SUB_U128,
    202 => CF_MUL_U128,
    203 => CF_DIV_U128,
    204 => CF_MOD_U128,
    205 => CF_BITAND_U128,
    206 => CF_BITOR_U128,
    207 => CF_BITXOR_U128,
    208 => CF_SHL_U128,
    209 => CF_SHR_U128,
    210 => CF_NEG_U128,
    211 => CF_BITNOT_U128,
    // isize: 212-223
    212 => CF_ADD_ISIZE,
    213 => CF_SUB_ISIZE,
    214 => CF_MUL_ISIZE,
    215 => CF_DIV_ISIZE,
    216 => CF_MOD_ISIZE,
    217 => CF_BITAND_ISIZE,
    218 => CF_BITOR_ISIZE,
    219 => CF_BITXOR_ISIZE,
    220 => CF_SHL_ISIZE,
    221 => CF_SHR_ISIZE,
    222 => CF_NEG_ISIZE,
    223 => CF_BITNOT_ISIZE,
    // usize: 224-235
    224 => CF_ADD_USIZE,
    225 => CF_SUB_USIZE,
    226 => CF_MUL_USIZE,
    227 => CF_DIV_USIZE,
    228 => CF_MOD_USIZE,
    229 => CF_BITAND_USIZE,
    230 => CF_BITOR_USIZE,
    231 => CF_BITXOR_USIZE,
    232 => CF_SHL_USIZE,
    233 => CF_SHR_USIZE,
    234 => CF_NEG_USIZE,
    235 => CF_BITNOT_USIZE,
    // 4 floating-point types × 6 operations
    // f16: 236-241
    236 => CF_ADD_F16,
    237 => CF_SUB_F16,
    238 => CF_MUL_F16,
    239 => CF_DIV_F16,
    240 => CF_MOD_F16,
    241 => CF_NEG_F16,
    // f32: 242-247
    242 => CF_ADD_F32,
    243 => CF_SUB_F32,
    244 => CF_MUL_F32,
    245 => CF_DIV_F32,
    246 => CF_MOD_F32,
    247 => CF_NEG_F32,
    // f64: 248-253
    248 => CF_ADD_F64_FULL,
    249 => CF_SUB_F64_FULL,
    250 => CF_MUL_F64_FULL,
    251 => CF_DIV_F64_FULL,
    252 => CF_MOD_F64_FULL,
    253 => CF_NEG_F64_FULL,
    // f128: 254-259
    254 => CF_ADD_F128,
    255 => CF_SUB_F128,
    256 => CF_MUL_F128,
    257 => CF_DIV_F128,
    258 => CF_MOD_F128,
    259 => CF_NEG_F128,
    // Semantic operations (260-265)
    260 => CF_REF_EQ,
    261 => CF_REF_NEQ,
    262 => CF_CONCAT_LIST,
    263 => CF_RANGE,
    264 => CF_RANGE_INCLUSIVE,
    265 => CF_ELVIS,
    // inline_trait / lazy construction (266-267)
    266 => CF_TRAIT_CONSTRUCT,
    267 => CF_LAZY_CONSTRUCT,
    268 => CF_SLICE,
    269 => CF_STR_CONCAT,
    // Global variable read/write (270-271)
    270 => CF_GLOBAL_LOAD,
    271 => CF_GLOBAL_STORE,
    // Record extension / atomic construction (272-273)
    272 => CF_RECORD_EXTEND,
    273 => CF_ATOMIC_CONSTRUCT,
    // Pattern matching (274-276)
    274 => CF_PATTERN_CTOR_MATCH,
    275 => CF_PATTERN_ADT_FIELD_GET,
    276 => CF_PATTERN_STR_EQ,
    // General type conversion (277-278)
    277 => CF_CAST_TO_STR,
    278 => CF_CAST_SCALAR,
    // Reference semantics and non-null assertion (279-282)
    279 => CF_NON_NULL_ASSERT,
    280 => CF_REF_OF,
    281 => CF_DEREF_READ,
    282 => CF_DEREF_WRITE,
    // Channel operations (283-285)
    283 => CF_CHANNEL_CREATE,
    284 => CF_CHANNEL_SEND,
    285 => CF_CHANNEL_CLOSE,
    // Partial application construction (286)
    286 => CF_PARTIAL_CONSTRUCT,
    // str.bytes() → u8[] (287)
    287 => CF_STR_BYTES,
    // Stack-allocated construction (288-289)
    288 => CF_RECORD_CONSTRUCT_STACK,
    289 => CF_ARRAY_CONSTRUCT_STACK,
    // Standalone reflect compute_fn (290-291): split from compute_ffi_call
    // to decouple lazy force logic from FFI calls.
    290 => CF_REFLECT_FORMAT,
    291 => CF_REFLECT_SCALAR_TO_STR,
    // String comparison (292-297): lexicographic by Unicode code point sequence.
    // Does not use the i32 path (str has no as_i32 semantics; the i32 path would always yield 0, producing wrong results).
    292 => CF_EQ_STR,
    293 => CF_NE_STR,
    294 => CF_LT_STR,
    295 => CF_GT_STR,
    296 => CF_LE_STR,
    297 => CF_GE_STR,
    // Semantic equality/inequality for composite types (298-299): record/adt/newtype/array/closure/throw, etc.
    298 => CF_EQ_OBJ,
    299 => CF_NE_OBJ,
    // Boolean inequality (300): symmetric with CF_EQ_BOOL(27); as_i32 on bool is always 0, so CF_NE_I32 cannot be used.
    300 => CF_NE_BOOL,
    // Array index store (301): arr[i] = x, in-place mutation of the Array heap object.
    301 => CF_ARRAY_STORE,
    // f128 comparison (302-307): f128 via to_f64 loses 60 bits of precision, requiring dedicated bit-pattern comparison.
    302 => CF_EQ_F128,
    303 => CF_NE_F128,
    304 => CF_LT_F128,
    305 => CF_GT_F128,
    306 => CF_LE_F128,
    307 => CF_GE_F128,
    // Memoization cache (308-309): memo_check queries the cache and returns record(hit, value); memo_store writes the cache and passes the value through.
    308 => CF_MEMO_CHECK,
    309 => CF_MEMO_STORE,
    // Tail-recursion WriteBack (310): compute_writeback + sets the Continue signal.
    310 => CF_TAILREC_WRITEBACK,
    // Control-flow compute_fn (311-313): replaces the control_signal_nodes table;
    // compute_fn directly returns NodeResult::Return/Break/Continue.
    311 => CF_RETURN,
    312 => CF_BREAK,
    313 => CF_CONTINUE,
    314 => CF_MATCH_FALLBACK,
    // Atomic operations (315-318): load/store/swap/compare_exchange on Atomic<T>.
    315 => CF_ATOMIC_LOAD,
    316 => CF_ATOMIC_STORE,
    317 => CF_ATOMIC_SWAP,
    318 => CF_ATOMIC_COMPARE_EXCHANGE,
    319 => CF_STR_MULTI_CONCAT,
    320 => CF_STR_ARRAY_JOIN,
    // Array fill [value, ..count] (321): repeats value count times
    321 => CF_ARRAY_FILL,
    // Runtime defer registration/execution (322-323): table override (new signature)
    322 => CF_DEFER_REGISTER,
    323 => CF_DEFER_RUN,
    // Block-scoped defer registration (324): like CF_DEFER_REGISTER but input[0] is an effect dep.
    324 => CF_BLOCK_DEFER_REGISTER,
    // stdlib @extern("C") #{ }# inline FFI call (325): resolves a self-symbol (kuzo_extern_<name>)
    // via dlsym/GetProcAddress + Abi::call_dynamic. Inputs are the call arguments + a trailing
    // effect dep; dyn_ffi_info metadata carries (symbol, sig, arg_count).
    325 => CF_DYN_FFI_CALL,
    // ── reflect compute_fns (326-340): standalone reflect primitives.
    // Replaces the former @builtin + REFLECT_ENTRIES + CF_FFI_CALL dispatch path.
    // Each takes the receiver value as input[0] (+ optional index as input[1])
    // and calls the pure-Rust helper in value/Reflect.rs directly — no FFI.
    326 => CF_REFLECT_KIND,           // v.kind()        -> u8
    327 => CF_REFLECT_TYPE_NAME,      // v.type_name()   -> str
    328 => CF_REFLECT_KIND_STR,       // v.kind_str()    -> str
    329 => CF_REFLECT_SIZE,           // v.size()        -> u8  (scalar byte width)
    330 => CF_REFLECT_LAYOUT_SIZE,    // v.size()        -> u32 (aggregate layout size)
    331 => CF_REFLECT_LAYOUT_ALIGN,   // v.alignment()   -> u32
    332 => CF_REFLECT_FIELD_COUNT,    // v.field_count() -> u16
    333 => CF_REFLECT_FIELD_NAME,     // v.field_name(i) -> str
    334 => CF_REFLECT_FIELD_VALUE,    // v.field_value(i)-> Value
    335 => CF_REFLECT_ARRAY_LEN,      // v.array_len()   -> usize
    336 => CF_REFLECT_ADT_CTOR,       // v.adt_constructor() -> str
}

/// Number of entries in `build_compute_fn_table()`.
///
/// Single source of truth for the solidify header's `compute_fn_count`
/// compatibility check (`solidify::Spec::COMPUTE_FN_COUNT`): a `.kzo` written
/// by a binary with a different table length is rejected at load instead of
/// silently mis-dispatching node compute fns. `build_compute_fn_table()`
/// asserts equality so this constant cannot drift silently.
pub const COMPUTE_FN_TABLE_LEN: u32 = 337;

// =========================================================================
// NodeKind — node category (not an op; 9 variants for readiness checks)
// =========================================================================

/// Node category: used solely by the scheduler to determine readiness, not for operation dispatch.
///
/// Fundamentally different from a traditional IR op (100+ opcodes): `kind` does not participate in dispatch.
/// The actual operation (add, subtract, multiply, divide, etc.) is determined by the build-time-bound `compute_fn`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, num_enum::TryFromPrimitive)]
#[repr(u8)]
pub enum NodeKind {
    /// Pure computation: constant.
    Const = 0,
    /// Pure computation: binary operation (executes when inputs are ready).
    BinOp = 1,
    /// Pure computation: ternary operation (recv + 2 args, e.g. atomic compare_exchange).
    TriOp = 2,
    /// Pure computation: unary operation.
    UnOp = 3,
    /// Pure computation: field access.
    FieldAccess = 4,
    /// Function call: launches a subgraph + waits for a completion event.
    Call = 5,
    /// Event source consumption: waits for an event (channel/async/timer).
    Await = 6,
    /// Control flow: conditional selection; activates the chosen subgraph.
    Gate = 7,
    /// Event source declaration: performs no computation; declares an external event entry point.
    EventSource = 8,
}

// =========================================================================
// Node — fixed-size node (stores only topological references, not values)
// =========================================================================

/// Dataflow graph node: fixed-size, stores only topological references.
///
/// - `kind`: node category (for readiness checks only, not operation dispatch)
/// - `input_count`: number of inputs (arbitrary; actual inputs live in InputsPool)
/// - `inputs_offset`: start position within InputsPool.data
/// - `compute_fn`: compute function index (bound at build time, invoked by array index at runtime)
///
/// The output is implicitly the node's own NodeId (the value table is indexed by NodeId).
/// The actual operation is determined by `compute_fn`; the scheduler does not care.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct Node {
    pub kind: NodeKind,
    pub input_count: u8,
    pub inputs_offset: u32,
    pub compute_fn: ComputeFnId,
}
// Layout: kind(1) + input_count(1) + pad(2) + inputs_offset(4) + compute_fn(4) = 12B
// align(16) forces 16B alignment; size is rounded up to 16B (4-byte trailing pad).
const _: () = assert!(std::mem::size_of::<Node>() == 16);

// =========================================================================
// BatchInfo — compile-time SIMD/parallel batching marker (per-Node, modeled after tail_call_flags)
// =========================================================================

/// Batching operation type: maps to the SIMD/rayon batch functions in Value.rs.
///
/// Set at compile time by compile_binary/compile_unary. At runtime, run_ready_nodes
/// groups ready nodes by (ValueTag, BatchOp) and reuses Value.rs's batch_binop/batch_cmp/
/// batch_unaryop for SIMD vectorization + rayon parallel batching, avoiding per-node
/// compute_fn overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchOp {
    /// Binary arithmetic/bitwise operation (returns a scalar of the same type).
    Bin(crate::value::BinOp),
    /// Comparison operation (returns bool).
    Cmp(crate::value::CmpOp),
    /// Unary operation (returns a scalar of the same type).
    Unary(crate::value::UnaryOp),
}

/// Compile-time batching info (per-Node, indexed by NodeId).
///
/// Only BinOp/UnOp nodes with scalar types have BatchInfo; Call/Gate/Await/record/array/
/// field nodes are None (not batchable; they follow the existing sequential compute_fn path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchInfo {
    /// Input/output scalar type (determines SIMD lane width).
    pub tag: crate::value::ValueTag,
    /// Operation type.
    pub op: BatchOp,
}

// =========================================================================
// InputsPool — standalone contiguous input pool
// =========================================================================

/// Standalone input pool: stores all node input NodeIds contiguously.
///
/// Node N's inputs = `data[N.inputs_offset .. N.inputs_offset + N.input_count]`.
/// Contiguous storage ensures cache friendliness and enables batch SIMD scanning of readiness.
#[derive(Clone)]
pub struct InputsPool {
    pub data: Vec<NodeId>,
}

impl InputsPool {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Pushes a group of inputs, returns the starting offset.
    pub fn push(&mut self, inputs: &[NodeId]) -> u32 {
        let offset = self.data.len() as u32;
        self.data.extend_from_slice(inputs);
        offset
    }

    /// Reads the input slice at the given position.
    pub fn get(&self, offset: u32, count: u8) -> &[NodeId] {
        let start = offset as usize;
        let end = start + count as usize;
        &self.data[start..end]
    }

    /// Current pool length.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Default for InputsPool {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// ValueTable — value table (SoA layout, runtime, one per frame)
// =========================================================================

/// Value table (SoA layout): values / ready / refcounts stored separately.
///
/// Compared to AoS (Vec<ValueSlot>), SoA keeps Value contiguous (stride = sizeof(Value)),
/// eliminating bool/u16 interleaving and improving cache density and vectorization
/// efficiency for SIMD batch extraction.
///
/// - `values`: node output values (indexed by local NodeId within the frame)
/// - `ready`: whether the value has been produced
/// - `refcounts`: slot-level RC (remaining downstream consumers; 0 means reclaimable)
///
/// Slot-level RC: when a node produces a value, refcount is set to the downstream count;
/// each downstream consumer decrements it by 1; when it reaches zero the slot can be cleared.
/// Frame-level fallback: at frame end, all non-zero slots are reclaimed uniformly
/// (heap object Arc Drop auto-decrefs).
#[derive(Clone)]
pub struct ValueTable {
    pub values: Vec<Value>,
    /// Ready bitmap (each bit represents one node's readiness state).
    /// Replaces the original Vec<bool>, 8x compression (N nodes: N B → N/8 B).
    pub ready: Vec<u8>,
    pub refcounts: Vec<u16>,
}

impl ValueTable {
    /// Creates an empty table.
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            ready: Vec::new(),
            refcounts: Vec::new(),
        }
    }

    /// Creates a table with the given capacity, all slots unready.
    pub fn with_unready(n: usize) -> Self {
        Self {
            values: vec![Value::NULL; n],
            ready: vec![0u8; (n + 7) / 8],
            refcounts: vec![0; n],
        }
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Resizes the table (new slots are unready).
    pub fn resize(&mut self, n: usize) {
        self.values.resize(n, Value::NULL);
        self.ready.resize((n + 7) / 8, 0);
        self.refcounts.resize(n, 0);
    }

    /// Checks whether node idx is ready.
    #[inline]
    pub fn is_ready(&self, idx: usize) -> bool {
        self.ready[idx >> 3] & (1 << (idx & 7)) != 0
    }

    /// Marks node idx as ready.
    #[inline]
    pub fn set_ready(&mut self, idx: usize) {
        self.ready[idx >> 3] |= 1 << (idx & 7);
    }

    /// Marks node idx as not ready.
    #[inline]
    pub fn clear_ready(&mut self, idx: usize) {
        self.ready[idx >> 3] &= !(1 << (idx & 7));
    }

    /// Sets the output value and downstream consumer count (local index).
    pub fn set_value(&mut self, idx: usize, value: Value, consumer_count: u16) {
        self.values[idx] = value;
        self.set_ready(idx);
        self.refcounts[idx] = consumer_count;
    }

    /// Gets the output value (cloned).
    pub fn get_value(&self, idx: usize) -> Value {
        self.values[idx].clone()
    }

    /// Gets a mutable reference to the output value (for &self semantics to directly modify the underlying HeapObj).
    pub fn get_value_mut(&mut self, idx: usize) -> Option<&mut Value> {
        self.values.get_mut(idx)
    }

    /// Consumes once (downstream read). Returns true if refcount is still > 0 (not zeroed);
    /// returns false if it has reached zero and can be reclaimed.
    pub fn consume(&mut self, idx: usize) -> bool {
        if self.refcounts[idx] > 0 {
            self.refcounts[idx] -= 1;
        }
        self.refcounts[idx] > 0
    }

    /// Whether all consumers have finished consuming (refcount reached zero).
    pub fn is_consumed(&self, idx: usize) -> bool {
        self.is_ready(idx) && self.refcounts[idx] == 0
    }

    /// Resets a single slot to unready (heap object Arc Drop auto-decrefs).
    pub fn reset_slot(&mut self, idx: usize) {
        self.values[idx] = Value::NULL;
        self.clear_ready(idx);
        self.refcounts[idx] = 0;
    }

    /// Resets all slots to unready (heap object Arc Drop auto-decrefs).
    pub fn reset_all(&mut self) {
        self.values.fill(Value::NULL);
        self.ready.fill(0);
        self.refcounts.fill(0);
    }
}

impl Default for ValueTable {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ValueTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValueTable")
            .field("len", &self.values.len())
            .field("ready_count", &(0..self.values.len()).filter(|&i| self.is_ready(i)).count())
            .finish()
    }
}

// =========================================================================
// ConstValue — compile-time constant raw value (stored by IrBuilder; Engine allocates ValueHandle)
// =========================================================================

/// Compile-time constant raw value (stored by IrBuilder; Engine allocates ValueHandle).
///
/// When compiling a Const node, IrBuilder stores the raw value in graph.const_values[NodeId].
/// At frame initialization, Engine allocates a ValueHandle via ValueArena and pre-fills the value_table.
///
/// The `Str` variant stores an (offset, len) reference into `DataFlowGraph.string_pool`.
/// Access via `to_value(&pool)` passes the string pool slice to construct a Str on the fly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    Isize(isize),
    Usize(usize),
    F32(f32),
    F64(f64),
    F16(u16),
    F128([u8; 16]),
    Bool(bool),
    Char(u32),
    /// String reference: (offset, len) into DataFlowGraph.string_pool.
    Str { offset: u32, len: u32 },
    Null,
    Void,
}

impl ConstValue {
    /// Converts to a Value (used by the optimizer/Engine to read constant values).
    /// `pool` = byte slice of DataFlowGraph.string_pool; the Str variant reads the string from it.
    pub fn to_value(&self, pool: &[u8]) -> crate::value::Value {
        match self {
            ConstValue::I8(v) => crate::value::Value::i8(*v),
            ConstValue::I16(v) => crate::value::Value::i16(*v),
            ConstValue::I32(v) => crate::value::Value::i32(*v),
            ConstValue::I64(v) => crate::value::Value::i64(*v),
            ConstValue::I128(v) => crate::value::Value::i128(*v),
            ConstValue::U8(v) => crate::value::Value::u8(*v),
            ConstValue::U16(v) => crate::value::Value::u16(*v),
            ConstValue::U32(v) => crate::value::Value::u32(*v),
            ConstValue::U64(v) => crate::value::Value::u64(*v),
            ConstValue::U128(v) => crate::value::Value::u128(*v),
            ConstValue::Isize(v) => crate::value::Value::isize_val(*v),
            ConstValue::Usize(v) => crate::value::Value::usize_val(*v),
            ConstValue::F32(v) => crate::value::Value::f32(*v),
            ConstValue::F64(v) => crate::value::Value::f64(*v),
            ConstValue::F16(bits) => crate::value::Value::f16(crate::value::F16(*bits)),
            ConstValue::F128(bytes) => crate::value::Value::f128(crate::value::F128(*bytes)),
            ConstValue::Bool(v) => crate::value::Value::bool_val(*v),
            ConstValue::Char(cp) => crate::value::Value::char_val(
                char::from_u32(*cp).unwrap_or('\0'),
            ),
            ConstValue::Str { offset, len } => {
                let off = *offset as usize;
                let end = off + *len as usize;
                let s = std::str::from_utf8(&pool[off..end]).unwrap_or("");
                crate::value::Value::ref_val(
                    crate::value::HeapObj::Str(crate::value::Str::from_rust_str(s)),
                )
            }
            ConstValue::Null => crate::value::Value::NULL,
            ConstValue::Void => crate::value::Value::VOID,
        }
    }

    /// Constructs a ConstValue from a Value (used by ConstFold to generate new constants).
    pub fn from_value(v: &crate::value::Value) -> Option<ConstValue> {
        use crate::value::{ValueTag, Value};
        match v {
            Value::Scalar(sv, tag) => match tag {
                ValueTag::I8 => Some(ConstValue::I8(unsafe { sv.i8_val })),
                ValueTag::I16 => Some(ConstValue::I16(unsafe { sv.i16_val })),
                ValueTag::I32 => Some(ConstValue::I32(unsafe { sv.i32_val })),
                ValueTag::I64 => Some(ConstValue::I64(unsafe { sv.i64_val })),
                ValueTag::I128 => {
                    let bits = unsafe { (sv.i128_val[0] as u128) | ((sv.i128_val[1] as u128) << 64) };
                    Some(ConstValue::I128(bits as i128))
                }
                ValueTag::U8 => Some(ConstValue::U8(unsafe { sv.u8_val })),
                ValueTag::U16 => Some(ConstValue::U16(unsafe { sv.u16_val })),
                ValueTag::U32 => Some(ConstValue::U32(unsafe { sv.u32_val })),
                ValueTag::U64 => Some(ConstValue::U64(unsafe { sv.u64_val })),
                ValueTag::U128 => {
                    let bits = unsafe { (sv.u128_val[0] as u128) | ((sv.u128_val[1] as u128) << 64) };
                    Some(ConstValue::U128(bits))
                }
                ValueTag::Isize => Some(ConstValue::Isize(unsafe { sv.isize_val })),
                ValueTag::Usize => Some(ConstValue::Usize(unsafe { sv.usize_val })),
                ValueTag::F32 => Some(ConstValue::F32(unsafe { sv.f32_val })),
                ValueTag::F64 => Some(ConstValue::F64(unsafe { sv.f64_val })),
                ValueTag::F16 => Some(ConstValue::F16(unsafe { sv.f16_val })),
                ValueTag::F128 => {
                    let lo = unsafe { sv.f128_val[0] };
                    let hi = unsafe { sv.f128_val[1] };
                    let bits: u128 = (lo as u128) | ((hi as u128) << 64);
                    Some(ConstValue::F128(bits.to_le_bytes()))
                }
                ValueTag::Bool => Some(ConstValue::Bool(unsafe { sv.bool_val })),
                ValueTag::Char => Some(ConstValue::Char(unsafe { sv.u32_val })),
                _ => None,
            },
            Value::Null => Some(ConstValue::Null),
            Value::Void => Some(ConstValue::Void),
            _ => None,
        }
    }
}

/// Converts a `u32` codepoint to a `char`, falling back to U+0000 for invalid codepoints.
/// Used by ConstValue::Char conversion and compute_fn cast operations.
#[inline]
pub fn char_from_u32_or_nul(u: u32) -> char {
    char::from_u32(u).unwrap_or('\0')
}

/// Branch info for a Gate node.
///
/// A Gate node selects which branch subgraph to activate based on the condition value.
/// `condition_input` is the NodeId of the condition value (global).
/// `branches` is the list of branches, each carrying its own inputs (global NodeIds; values read from the parent frame).
/// Different branches may have different numbers of inputs (corresponding to subgraphs with different param_counts).
#[derive(Debug, Clone)]
pub struct GateBranches {
    /// Condition input node (global NodeId).
    pub condition_input: NodeId,
    /// Branch list: (condition value, subgraph id, parameter node list).
    pub branches: Vec<(bool, SubGraphId, Vec<NodeId>)>,
    /// W4c capture gate: the selected branch's Return signal is CAPTURED as
    /// the Gate's value instead of propagating to the caller frame. Used by
    /// inline expansion of bodies with early `return`/`?` — the captured
    /// value flows as data to the call site (exactly what a non-inlined call
    /// does), and the caller keeps executing. Serialized in the section's
    /// validity byte (2 = valid + capture).
    pub capture: bool,
}

/// select expression branch info (indexed by Gate node NodeId into select_infos).
#[derive(Debug, Clone)]
pub struct SelectInfo {
    /// Info for each Receive/Timeout branch.
    pub branches: Vec<SelectBranch>,
}

/// Info for a single branch of a select expression.
#[derive(Debug, Clone)]
pub struct SelectBranch {
    /// Branch subgraph id (executes the branch body).
    pub subgraph_id: SubGraphId,
    /// Event source type (Channel or Timer).
    pub event_kind: EventSourceKind,
    /// Event source value node (NodeId of the channel handle or timer handle, global).
    pub event_source_node: NodeId,
}

// =========================================================================
// ControlSignal — control signal (unified representation of non-local jumps)
// =========================================================================

/// Control signal: unified representation of non-local jumps.
///
/// run_ready_nodes checks this field each loop iteration; if not None, processing stops.
/// Triggered by control-flow compute_fn (CF_RETURN/CF_BREAK/CF_CONTINUE/CF_THROW_WRAP_ERR)
/// returning NodeResult::Return/Break/Continue.
#[derive(Debug, Clone, Default)]
pub enum ControlSignal {
    /// No signal; normal execution.
    #[default]
    None,
    /// Triggered by a return statement: the subgraph returns this value early.
    Return(Value),
    /// Triggered by a break statement: exits the loop.
    Break,
    /// Triggered by a continue statement: proceeds to the next loop iteration.
    Continue,
}

/// Checks whether a node is a control-flow node (Return/Break/Continue/Throw).
///
/// Replaces the old control_signal_nodes table check: control-flow semantics are now
/// expressed directly via compute_fn (CF_RETURN/CF_BREAK/CF_CONTINUE/CF_THROW_WRAP_ERR)
/// returning NodeResult::Return/Break/Continue.
///
/// NOTE: this is the narrow "produces a non-local exit signal" predicate, kept
/// exact for behavior preservation. The broader dispatch classification lives
/// in `effect_class` (whose ControlFlow class additionally contains Gate launch
/// and match fallback).
pub fn is_control_flow_compute_fn(cf: ComputeFnId) -> bool {
    cf == CF_RETURN || cf == CF_BREAK || cf == CF_CONTINUE || cf == CF_THROW_WRAP_ERR
}

/// Control-signal propagation matrix (single source of truth).
///
/// Decides whether a completing child subgraph's control signal propagates to
/// the caller frame on the normal (non-LoopBody) completion path. Shared by
/// the async engine (`engine/Subgraph.rs` `complete_and_wake_caller`) and the
/// sync interpreter (`ir/Compute.rs`); the `pending_completions` race path in
/// `engine/Schedule.rs` intentionally propagates more broadly (see the
/// comment there — a dropped signal cannot be recovered on that path).
///
/// - `Return`: propagates from Gate branches (if/match arm) and loop frames
///   (while/loop/for/tailrec sg); NOT from cross-function or lambda calls —
///   their return value has already been extracted as data (Bug #65/#97
///   class: propagating would make the caller exit prematurely).
/// - `Break`/`Continue`: propagate from Gate branches only — they must
///   penetrate to the enclosing LoopBody frame; a loop frame's own
///   Break/Continue has already been consumed by the loop.
/// - `None`: never propagates.
///
/// Both engines additionally gate propagation on `child.function_id ==
/// caller.function_id` (in-function only).
pub fn should_propagate_control_signal(
    child_signal: &ControlSignal,
    call_node_is_gate: bool,
    child_loop_kind: LoopKind,
) -> bool {
    match child_signal {
        ControlSignal::Return(_) => call_node_is_gate || child_loop_kind != LoopKind::None,
        ControlSignal::Break | ControlSignal::Continue => call_node_is_gate,
        ControlSignal::None => false,
    }
}

// =========================================================================
// FrameState — frame state
// =========================================================================

/// Frame state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameState {
    /// Ready to execute (has ready nodes).
    Ready,
    /// Currently executing.
    Running,
    /// Suspended waiting for an event (async; implemented in phase 5).
    Suspended,
    /// Being cancelled (implemented in phase 5).
    Cancelling,
    /// Completed.
    Completed,
    /// Failed.
    Failed,
}

// =========================================================================
// SuspendState — frame suspend state (deviation 2: unified call/await suspend model)
// =========================================================================

/// Frame suspend state.
///
/// A frame suspends when it reaches a call/await node, waiting for an event to resume:
/// - `NotSuspended`: running normally
/// - `WaitingSubgraph(FrameId)`: waiting for a subgraph frame to complete (used by sync call nodes)
/// - `WaitingEvent(NodeId)`: waiting for a channel/timer/async event (used by await nodes; NodeId is the await node)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuspendState {
    NotSuspended,
    WaitingSubgraph(FrameId),
    WaitingEvent(NodeId),
}

// =========================================================================
// RuntimeEvent — runtime event (subgraph completion, etc.)
// =========================================================================

/// Runtime event: drives suspended frames to resume execution.
///
/// Spec 4.4 on_event_arrived handles all event sources uniformly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeEvent {
    /// Subgraph execution completed (the event awaited by sync call nodes).
    SubgraphComplete(FrameId),
    /// Channel has data available for reading (the event awaited by channel.recv()).
    ChannelReady(ChannelId),
    /// Timer fired (the event awaited by timer.sleep()).
    TimerFired(TimerId),
    /// Async call completed (the event awaited by async_handle.await()).
    AsyncJoin(AsyncHandleId),
}

// =========================================================================
// PendingCall — pending subgraph call (constructed when a call node executes)
// =========================================================================

/// Pending subgraph call.
///
/// Constructed when a call node's compute_fn executes; the scheduler consumes it and launches a subgraph frame.
/// - `is_async=false`: sync call; the current frame suspends waiting for SubgraphComplete
/// - `is_async=true`: async call; the current frame does not suspend; the call node writes an AsyncHandle + notifies downstream
#[derive(Debug, Clone)]
pub struct PendingCall {
    /// Target subgraph id.
    pub target_sg: SubGraphId,
    /// Call arguments (value list).
    pub args: Vec<Value>,
    /// The node that initiated the call (local NodeId within the frame; the return value is written back after subgraph completion).
    pub call_node_local: NodeId,
    /// Async call flag: true = do not suspend the current frame; returns an AsyncHandle.
    pub is_async: bool,
    /// Stores the Closure value for escaping closure calls, used to write back upvalues to the Closure after the child frame completes.
    /// None = ordinary function call or same_function closure call (no writeback needed).
    pub closure_val: Option<Value>,
}

// =========================================================================
// PendingAwait — pending await suspension (constructed when an await node executes)
// =========================================================================

/// Pending await suspension.
///
/// Constructed when an await node's compute_fn executes; the core loop consumes it and
/// checks event source readiness: ready → inject the value and continue; not ready →
/// register event_waiters + suspend the frame.
#[derive(Debug, Clone)]
pub struct PendingAwait {
    /// The await node (local NodeId within the frame; the value is written back when the event arrives).
    pub await_node_local: NodeId,
    /// Event object value (the Value of an AsyncHandle/Channel/Timer).
    pub event_obj: Value,
    /// Event kind (determines how to check readiness and how to resolve the event source id).
    pub event_kind: EventSourceKind,
}

// =========================================================================
// Pending — unified suspend action enum
// =========================================================================

// The Pending enum has been removed: side effects are passed explicitly via NodeResult return values.
// The PendingCall/PendingAwait structs are retained for use by NodeResult::Call/Await.

// =========================================================================
// NodeResult — unified compute_fn return value (all side effects passed explicitly)
// =========================================================================

/// Unified return value of compute_fn.
///
/// All side effects are passed explicitly via the return value, eliminating the implicit
/// side effects of frame.pending. The engine hot loop dispatches on NodeResult via match.
#[derive(Debug, Clone)]
pub enum NodeResult {
    /// Normal value computation completed.
    Value(Value),
    /// Batch computation completed (multiple nodes produce values simultaneously).
    Batch(Vec<(NodeId, Value)>),
    /// Function call (sync/async/tail call, distinguished by PendingCall.is_async).
    Call(PendingCall),
    /// Await suspension (waiting for a channel/timer/async event).
    Await(PendingAwait),
    /// Channel notification (a Send operation triggers ChannelReady to wake waiting frames).
    ChannelNotify(ChannelId),
    /// Cancel an async operation.
    Cancel(AsyncHandleId),
    /// Select wait (suspends when no Gate branch is ready).
    SelectWait(NodeId),
    /// Control flow: return (the value serves as the function return value).
    Return(Value),
    /// Control flow: break.
    Break,
    /// Control flow: continue.
    Continue,
}

// =========================================================================
// EvalContext — compute_fn execution context (provides batching decision support)
// =========================================================================

/// compute_fn execution context.
///
/// Does not borrow frame data (to avoid conflicts with &mut Frame borrows).
/// collect_batch_candidates receives &Frame as a parameter to access ready_queue.
///
/// `graph` carries a shared borrow of the DataFlowGraph supplied by the driver (engine loop /
/// sync interpreter), so compute_fns never need `frame.graph.clone()` (a per-node atomic RMW
/// pair) to obtain graph access.
pub struct EvalContext<'a> {
    /// Subgraph node start offset (used for local NodeId → global NodeId conversion).
    pub node_start: u32,
    /// The executing graph (shared borrow, outlives the current node execution).
    pub graph: &'a DataFlowGraph,
}

impl<'a> EvalContext<'a> {
    /// Scans ready_queue after the current node, collecting nodes of the same type as the current node.
    ///
    /// compute_fn uses this method to decide whether to do SIMD batching.
    /// Returns a list of global NodeIds.
    pub fn collect_batch_candidates(
        &self,
        frame: &Frame,
        _current: NodeId,
        predicate: impl Fn(NodeId) -> bool,
    ) -> Vec<NodeId> {
        let mut result = Vec::new();
        for &local_nid in frame.ready_queue.iter() {
            let gid = NodeId(local_nid.0 + self.node_start);
            if predicate(gid) {
                result.push(gid);
            }
        }
        result
    }

    /// ready_queue length.
    pub fn queue_len(&self, frame: &Frame) -> usize {
        frame.ready_queue.len()
    }
}

// =========================================================================
// Frame — execution frame (runtime state of a single function call)
// =========================================================================

/// Execution frame: the runtime state of a single function call.
///
/// - `value_table`: value table indexed by NodeId (SoA layout, one slot per node)
/// - `pending_inputs`: remaining unready input count per node
/// - `ready_queue`: queue of nodes ready for execution
/// - `state`: frame state
/// - `subgraph_id`: owning subgraph
/// - `caller`: caller frame + call node (return value written back on subgraph completion)
///
/// Frame-level reclamation: at frame end the entire value_table is released; heap objects go through Arc Drop RC.
pub struct Frame {
    /// Dataflow graph (read-only shared; compute_fn accesses it via frame.graph).
    pub graph: std::sync::Arc<DataFlowGraph>,
    /// Value table (SoA layout, indexed by local NodeId within the frame, starting from 0).
    pub value_table: ValueTable,
    /// Remaining unready input count per node.
    pub pending_inputs: Vec<u16>,
    /// Queue of nodes ready for execution.
    pub ready_queue: std::collections::VecDeque<NodeId>,
    /// Frame state.
    pub state: FrameState,
    /// Owning subgraph id.
    pub subgraph_id: SubGraphId,
    /// Caller frame + call node (None = top-level frame).
    pub caller: Option<(FrameId, NodeId)>,
    /// Frame id.
    pub id: FrameId,
    /// Subgraph node start offset (global NodeId = local NodeId + node_offset).
    pub node_offset: u32,
    /// Control signal (triggered by return/break/continue).
    pub control_signal: ControlSignal,
    /// Suspend state (set when a call/await node suspends).
    pub suspend_state: SuspendState,
    /// Defer stack (runtime; executed LIFO on frame release).
    /// Stores dynamically-registered defers (e.g. defer-in-loop), each with captured values.
    pub defer_stack: Vec<RuntimeDefer>,
    /// Suspend event (subgraph completion, etc.; drives frame resumption).
    pub suspend_event: Option<RuntimeEvent>,
    /// Timers already started in select (branch_idx, timer_id); Timer branches are started on first check.
    pub select_timers: Vec<(usize, crate::ir::Ir::TimerId)>,
    /// Points to the function root frame. Inherited within the same function's subgraphs;
    /// set to null for cross-function calls and async child frames.
    /// Safety is ensured by Box<Frame> address stability + single-worker synchronous loop.
    pub root_frame_ptr: *mut Frame,
    /// Points to the direct caller frame. Used by get_value_by_global to traverse intermediate frames
    /// (e.g., variables declared in loop-body frames), compensating for root_frame_ptr only reaching the root frame directly.
    pub parent_frame_ptr: *mut Frame,
    /// Generic cached child frame ID (loop-body frame reuse: while_sg/loop_sg/for_sg/tailrec frames cache the body_sg child frame).
    pub cached_child_frame: Option<FrameId>,
    /// E2 loop hot path: the loop-body frame carried in hand across iterations. While set, the
    /// Gate drives the body directly on the current stack (no queue round-trip per iteration).
    /// The stash is dropped on frame reuse (acquire_frame); a suspending body is handed back to
    /// the frames map (fallback to the queue protocol) and the stash cleared.
    pub hot_body: Option<(FrameId, Box<Frame>)>,
    /// E3 perf: steady-state cache for same_function branch-frame preparation. Holds
    /// (parent-ready bitmap snapshot at derivation time, derived pending_inputs, seed list).
    /// When the copied-in ready bitmap matches the snapshot, the derivation is skipped and the
    /// cached pending/seed reused (memcpy). Cleared on frame reuse / subgraph switch; length
    /// mismatches (resize) also invalidate via the comparison.
    pub same_fn_prep_cache: Option<Box<(Vec<u8>, Vec<u16>, Vec<NodeId>)>>,
    /// E5: set by prepare paths when the frame starts fresh (all own slots unready except
    /// injected params). One-shot: the dispatch wrapper consumes it to enter run_linear once;
    /// a bail or resume falls back to the dataflow engine permanently for this frame.
    pub linear_fresh: bool,
    /// Stores the Closure value for escaping closure calls, used to write back upvalues to the Closure after the child frame completes.
    /// None = ordinary function call or same_function closure call.
    pub closure_val: Option<Value>,
}

impl Frame {
    /// Creates a new frame; value_table and pending_inputs are initialized to the subgraph's node count.
    pub fn new(id: FrameId, subgraph_id: SubGraphId, node_count: usize, graph: std::sync::Arc<DataFlowGraph>) -> Self {
        Self {
            graph,
            value_table: ValueTable::with_unready(node_count),
            pending_inputs: vec![0; node_count],
            ready_queue: std::collections::VecDeque::new(),
            state: FrameState::Ready,
            subgraph_id,
            caller: None,
            id,
            node_offset: 0,
            control_signal: ControlSignal::None,
            suspend_state: SuspendState::NotSuspended,
            defer_stack: Vec::new(),
            suspend_event: None,
            select_timers: Vec::new(),
            root_frame_ptr: std::ptr::null_mut(),
            parent_frame_ptr: std::ptr::null_mut(),
            cached_child_frame: None,
            hot_body: None,
            same_fn_prep_cache: None,
            linear_fresh: false,
            closure_val: None,
        }
    }

    /// Sets a node's output value (local NodeId).
    pub fn set_value(&mut self, node: NodeId, value: Value, consumer_count: u16) {
        self.value_table.set_value(node.0 as usize, value, consumer_count);
    }

    /// Gets a node's output value (local NodeId; returns a clone).
    pub fn get_value(&self, node: NodeId) -> Value {
        self.value_table.get_value(node.0 as usize)
    }

    /// Gets a node's output value (global NodeId; auto-converts to local index; returns a clone).
    /// compute_fn uses this method when reading inputs (inputs_pool stores global NodeIds).
    /// On out-of-bounds, traverses the call chain via parent_frame_ptr (intermediate-frame variables),
    /// then falls back to root_frame_ptr (function root frame).
    pub fn get_value_by_global(&self, global_node: NodeId) -> Value {
        let local = global_node.0.wrapping_sub(self.node_offset);
        if (local as usize) < self.value_table.len() {
            if self.value_table.is_ready(local as usize) {
                self.value_table.get_value(local as usize)
            } else if self.pending_inputs[local as usize] > 0 {
                // The node is within the current frame's range but will never become ready
                // (nested subgraph nodes have pending_inputs=MAX, or nodes depending on
                // nested nodes have pending_inputs>0 and will never reach zero).
                // Walk up to the parent frame to get the value.
                if !self.parent_frame_ptr.is_null() {
                    unsafe { (*self.parent_frame_ptr).get_value_by_global(global_node) }
                } else if !self.root_frame_ptr.is_null() {
                    unsafe { (*self.root_frame_ptr).get_value_by_global(global_node) }
                } else {
                    Value::NULL
                }
            } else {
                self.value_table.get_value(local as usize)
            }
        } else if !self.parent_frame_ptr.is_null() {
            unsafe { (*self.parent_frame_ptr).get_value_by_global(global_node) }
        } else if !self.root_frame_ptr.is_null() {
            unsafe { (*self.root_frame_ptr).get_value_by_global(global_node) }
        } else {
            Value::NULL
        }
    }

    /// Checks whether a node is ready (all inputs produced).
    pub fn is_node_ready(&self, node: NodeId) -> bool {
        self.pending_inputs[node.0 as usize] == 0
    }

    /// Enqueues a node into the ready queue.
    pub fn push_ready(&mut self, node: NodeId) {
        self.ready_queue.push_back(node);
    }

    /// Dequeues a ready node.
    pub fn pop_ready(&mut self) -> Option<NodeId> {
        self.ready_queue.pop_front()
    }
}

// =========================================================================
// EventSource — event source (runtime object external to the graph; its produced value is injected into await node input edges)
// =========================================================================

/// Channel id (runtime).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChannelId(pub u64);

/// Timer id (runtime).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimerId(pub u32);

/// Async handle id (runtime; async call completion event).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsyncHandleId(pub u32);

/// Event source: a runtime object external to the graph whose produced value is injected into an await node's input edge.
///
/// - One input edge of an await node points to an EventSource declaration node
/// - The EventSource declaration node is bound to a concrete EventSource instance at runtime
/// - When an event arrives, the event source writes the value to the value-table slot of the await node's corresponding input
///
/// call nodes wait for "subgraph completion events"; await nodes wait for "channel/timer/async events" —
/// the execution engine handles both uniformly, which is the unification of call and await.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventSource {
    /// Channel has data available for reading/writing.
    Channel(ChannelId),
    /// Async call completed.
    AsyncJoin(AsyncHandleId),
    /// Timer expired.
    Timer(TimerId),
    /// Subgraph execution completed (used by call nodes).
    SubgraphComplete(SubgraphInstanceId),
}

// =========================================================================
// DeferEntry — defer block (frame Drop semantics)
// =========================================================================

/// defer block definition: compiled as an independent subgraph, executed in LIFO order on frame release.
///
/// Addresses a Zig pain point: defer is attached to the frame and executes on any frame-release path
/// (normal return, error propagation, cancellation), uniformly with no special cases.
#[derive(Debug, Clone)]
pub struct DeferEntry {
    /// defer registration point (trigger node).
    pub trigger_node: NodeId,
    /// defer block body subgraph.
    pub body_subgraph: SubGraphId,
    /// Captured variables (NodeId list snapshotted at registration).
    pub captured_inputs: Vec<NodeId>,
    /// Whether it has been registered to the frame's defer_stack (runtime marker to prevent duplicate execution).
    pub registered: bool,
}

/// Runtime defer entry: a defer body subgraph + the captured values at registration time.
///
/// Used by the dynamic defer mechanism (CF_DEFER_REGISTER / CF_DEFER_RUN) to support
/// defer-in-loop: each loop iteration pushes a `RuntimeDefer` (with the current loop
/// variable values) onto `frame.defer_stack`; the loop-exit `CF_DEFER_RUN` node drains
/// the stack in LIFO order and executes each defer body with its captured values.
#[derive(Debug, Clone)]
pub struct RuntimeDefer {
    /// defer block body subgraph (same_function branch).
    pub body_subgraph: SubGraphId,
    /// Captured NodeIds (global) whose values were snapshotted at registration time.
    pub captured_nodes: Vec<NodeId>,
    /// Captured values snapshotted at registration time (injected into defer frame slots).
    pub captured_values: Vec<Value>,
}

// =========================================================================
// RecordLitInfo — record construction info (for RecordLit nodes)
// =========================================================================

/// Construction kind: distinguishes Record / ADT / Newtype, driving compute_record_construct to build different HeapObjs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, num_enum::TryFromPrimitive)]
#[repr(u8)]
pub enum RecordLitKind {
    Record = 0,
    Adt = 1,
    Newtype = 2,
}

/// Type field info (registered in type_scope_stack, indexed by constructor name or type name).
///
/// field_names: the constructor's field name list (empty for Newtype, which uses a separate path)
/// type_name: the owning type name (for multi-constructor ADTs, the constructor name != type name, so the type name must be stored)
/// kind: construction kind (Record / Adt / Newtype)
#[derive(Debug, Clone)]
pub struct TypeFieldInfo {
    pub field_names: Vec<String>,
    pub type_name: String,
    pub kind: RecordLitKind,
}

/// Record construction info (for RecordLit nodes).
///
/// RecordLit compiles to a construction node; compute_fn collects field values from inputs to construct a HeapObj.
/// Depending on kind, constructs RecordValue / AdtValue / NewtypeValue.
/// type_name stores the owning type name (not the constructor name); constructor stores the constructor name (for ADTs).
#[derive(Debug, Clone)]
pub struct RecordLitInfo {
    pub type_name: String,
    pub field_names: Vec<Option<String>>,
    pub constructor: String,
    pub kind: RecordLitKind,
}

/// Closure construction node info (indexed by NodeId; None for non-closure-construction nodes).
///
/// A closure construction node (compute_fn = 40) retrieves the subgraph id + arity from closure_infos at runtime,
/// merges inputs (captured values) to construct a Closure heap object.
#[derive(Debug, Clone, Copy)]
pub struct ClosureInfo {
    /// Closure subgraph id.
    pub subgraph_id: SubGraphId,
    /// Number of lambda parameters (excluding captured upvalues).
    pub arity: u8,
    /// Index of the self-reference upvalue (for recursive nested functions; -1 means no self-reference).
    pub self_upvalue_idx: i32,
}

/// Carries the FFI symbol info + ABI signature for `@extern("C")` stdlib C calls.
///
/// At runtime, `compute_dyn_ffi_call` reads this metadata, resolves the symbol address
/// via `ffi::Symbols` (dlsym self-lookup + cache), and invokes via
/// `Abi::CallDynamic::call_dynamic`.
#[derive(Debug, Clone, Hash)]
pub struct DynFfiInfo {
    /// C symbol name (e.g. "kuzo_extern___file_open_raw").
    pub symbol: String,
    /// ABI signature (parameter types + return type); str params are pre-expanded to (Ptr, Int).
    pub sig: crate::ffi::Abi::AbiSig,
    /// Number of Kuzo-level argument values (not counting the trailing effect dependency).
    /// Used by compute_dyn_ffi_call to separate args from the effect input.
    pub arg_count: u8,
}

/// Partial application construction node info (compute_fn = 286).
///
/// When compile_call detects that the number of actual arguments < the target function's parameter count,
/// it generates a partial_construct node. At runtime, it retrieves the subgraph id + bound_count from
/// partial_infos, merges inputs (bound argument values) to construct a HeapObj::Partial.
/// remaining_arity is derived from subgraph.param_count - bound_count.
#[derive(Debug, Clone, Copy)]
pub struct PartialInfo {
    /// Target function subgraph id.
    pub subgraph_id: SubGraphId,
    /// Number of bound parameters (= node input count).
    pub bound_count: u8,
}

/// inline_trait construction node info (indexed by NodeId).
///
/// compute_trait_construct (compute_fn=266) retrieves each method's subgraph id + arity + upvalue count
/// from this info at runtime, merges node inputs (each method's upvalues concatenated in order) to
/// construct multiple Closures, packed into a TraitValue heap object.
#[derive(Debug, Clone)]
pub struct TraitConstructInfo {
    /// Trait name (filled into TraitValue.trait_name at runtime).
    pub trait_name: String,
    /// Method name list (one-to-one with methods; filled into TraitValue.method_names).
    pub method_names: Vec<String>,
    /// Subgraph info for each method (one-to-one with method_names).
    pub methods: Vec<TraitMethodEntry>,
}

/// Subgraph info for a single inline_trait method.
#[derive(Debug, Clone, Copy)]
pub struct TraitMethodEntry {
    pub subgraph_id: SubGraphId,
    pub arity: u8,         // Number of method parameters (excluding upvalues)
    pub upvalue_count: u8, // Number of upvalues for this method (split from inputs in order)
}

/// Lazy construction node info (indexed by NodeId).
///
/// compute_lazy_construct (compute_fn=267) retrieves the thunk subgraph id from this info at runtime,
/// constructing a LazyValue heap object (the thunk is unevaluated; on first force, it launches the
/// subgraph computation and caches the result).
#[derive(Debug, Clone, Copy)]
pub struct LazyConstructInfo {
    /// Thunk subgraph id (no parameters; the return value is the lazy expression's value).
    pub thunk_sg: SubGraphId,
}

/// Record extension node info (indexed by NodeId).
///
/// compute_record_extend (compute_fn=272) retrieves the updated field name list from this info at runtime,
/// clones fields from the base RecordValue, replaces/appends per the updated field names, and constructs a new RecordValue.
/// inputs[0] = base record; inputs[1..] = updated field values (in order corresponding to update_names).
#[derive(Debug, Clone)]
pub struct RecordExtendInfo {
    /// Updated field name list (length = input_count - 1; corresponds to inputs[1..]).
    pub update_names: Vec<String>,
}

/// Memoization cache node metadata: shared by memo_check / memo_store.
/// memo_check: inputs[0..param_count] = argument values; table_index indexes the cache table
/// memo_store: inputs[0..param_count] = argument values, inputs[param_count] = result value
#[derive(Debug, Clone)]
pub struct MemoInfo {
    /// Cache table index (position in graph.memo_tables).
    pub table_index: u32,
    /// Number of parameters (the first param_count inputs are cache key components).
    pub param_count: u8,
}

// =========================================================================
// EventSourceDecl — event source declaration (static, compile time)
// =========================================================================

/// Event source declaration: declares an external event entry point within a subgraph.
///
/// An input edge of an await node points to an EventSource declaration node,
/// which is bound to a concrete EventSource instance at runtime.
#[derive(Debug, Clone)]
pub struct EventSourceDecl {
    /// The node where the declaration resides.
    pub node: NodeId,
    /// Event source kind (runtime-bound instance).
    pub kind: EventSourceKind,
}

/// Event source kind (static declaration; runtime-bound instance).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, num_enum::TryFromPrimitive)]
#[repr(u8)]
pub enum EventSourceKind {
    /// Channel event.
    Channel,
    /// Async join event.
    AsyncJoin,
    /// Timer event.
    Timer,
    /// Subgraph completion event.
    SubgraphComplete,
}

// =========================================================================
// SubGraph — function subgraph (static, generated at compile time)
// =========================================================================

/// Function subgraph: each function (including monomorphization instances) compiles to a SubGraph.
///
/// - `node_range`: node id range [start, end)
/// - `entry_node`: entry node (receives parameters)
/// - `return_node`: return node (produces the return value)
/// - `has_suspend`: whether there are suspend points (async=true)
///
/// A sync function = a subgraph with no suspend points; it runs to completion and immediately produces a completion event.
/// An async function = a subgraph with suspend points (await nodes connected to event sources).
/// The difference is only whether the subgraph has await nodes; the execution engine handles both uniformly.
#[derive(Debug, Clone)]
pub struct SubGraph {
    /// Subgraph id.
    pub id: SubGraphId,
    /// Node id range [start, end).
    pub node_range: (NodeId, NodeId),
    /// Number of parameters (input count of the entry node).
    pub param_count: u8,
    /// Entry node (receives parameters).
    pub entry_node: NodeId,
    /// Return node (produces the return value).
    pub return_node: NodeId,
    /// Whether there are suspend points (async=true).
    pub has_suspend: bool,
    /// Declared event sources (channel/timer, etc.).
    pub event_source_decls: Vec<EventSourceDecl>,
    /// Defer block subgraph definitions.
    pub defer_table: Vec<DeferEntry>,
    /// Loop kind (ordinary subgraph=None, while_sg=While, loop_sg=Loop, for_sg=For, body_sg=LoopBody).
    pub loop_kind: LoopKind,
    /// body_sg points to the parent loop subgraph (while_sg/loop_sg/for_sg).
    pub loop_parent_sg: Option<SubGraphId>,
    /// Loop condition node (used by While/For; must be reset on loop iteration reset).
    pub cond_node: Option<NodeId>,
    /// Owning function ID (top-level function subgraph = its own SubGraphId.0; loop/branch subgraph = parent function's function_id).
    pub function_id: u32,
    /// For-loop iterator advance node (reset on reset_loop_iteration).
    pub iter_next_node: Option<NodeId>,
    /// Upvalue count (number of lambda-captured variables, including self-recursive references).
    /// param_count = actual parameter count + upvalue_count
    pub upvalue_count: u8,
    /// Outer node ID for each upvalue (used to inject current parent-frame values during same_function calls).
    pub upvalue_outer_nodes: Vec<NodeId>,
    /// List of directly nested subgraph node_ranges (precomputed at build time; O(len) query at runtime instead of full-graph scan).
    /// Only includes directly nested subgraphs, not grandchild subgraphs (those are handled by recursive prepare logic).
    pub nested_ranges: Vec<(u32, u32)>,
    /// Frame-reuse reset plan (generated at compile time; replaces runtime LoopKind branch checks).
    /// Only loop subgraphs (while_sg/loop_sg/for_sg) have this plan; ordinary subgraphs are None.
    pub reset_plan: Option<ResetPlan>,
}

/// Reset plan for subgraph frame reuse (computed at compile time by Builder, stored in SubGraph).
///
/// Encodes the reset differences between For vs While/Loop as data, so the engine no longer branches on LoopKind.
#[derive(Debug, Clone, Default)]
pub struct ResetPlan {
    /// Nodes to reset to pending=0 and enqueue (For's iter_next_node).
    pub reset_to_zero: Vec<NodeId>,
    /// Nodes to reset to pending=1 (For's cond_node; input comes from iter_next).
    pub reset_to_one: Vec<NodeId>,
    /// Condition tree root nodes requiring recursive reset (While/Loop's cond_node).
    pub reset_condition_tree: Vec<NodeId>,
    /// W5: `reset_condition_tree` flattened ONCE at build/load time into
    /// (node, pending) pairs (topological order), so the engine applies the
    /// per-iteration reset mechanically instead of re-running the DFS (which
    /// also rescanned every subgraph for nested ranges) each iteration.
    /// Empty when there is nothing to precompute or before `precompute_reset_plans`.
    pub condition_tree_plan: Vec<(NodeId, u16)>,
}

/// Which side of the NodeRef door a reference came from: per-node metadata
/// (`owner` = node index) or a subgraph anchor (`owner` = subgraph index).
/// Diagnostic label for the door's consumers (see `for_each_node_ref` /
/// `map_node_refs` in the `rebuild` impl block).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRefSite {
    /// await source / writeback target / gate cond+params / select sources.
    NodeMeta,
    /// sg anchors / defer registration / event decls / upvalues / reset plan.
    SgAnchor,
}

/// Loop subgraph kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, num_enum::TryFromPrimitive)]
#[repr(u8)]
pub enum LoopKind {
    /// Ordinary subgraph.
    None,
    /// while_sg (contains cond + Gate).
    While,
    /// loop_sg (no cond; terminates via break).
    Loop,
    /// for_sg (contains iterator + cond).
    For,
    /// body_sg (loop body; no tail recursion).
    LoopBody,
    /// Tail-recursion-to-iteration loop (cond-based Gate + Continue signal exit mechanism).
    /// WriteBack sets Continue → loop continues; body_sg completes with no signal → base case hit → loop exits.
    TailRec,
}

// =========================================================================
// ComputeFn — compute function (bound at build time, eliminating dispatch)
// =========================================================================

/// Compute function signature: receives a frame + node id + execution context, returns a NodeResult.
///
/// The frame holds the graph (Arc<DataFlowGraph>); compute_fn accesses graph data via frame.graph.
/// Build-time-bound index (ComputeFnId); invoked at runtime via the compute function table index.
/// Each operation+type combination has a specialized function; at runtime there is no type checking or op table lookup.
/// All side effects are passed explicitly via the NodeResult return value.
pub type ComputeFn = fn(frame: &mut Frame, node: NodeId, ctx: &EvalContext) -> NodeResult;

/// Wrapper macro: wraps the old signature `fn(&mut Frame, NodeId) -> Value` into the new signature
/// `fn(&mut Frame, NodeId, &EvalContext) -> NodeResult`.
///
/// For nodes with BatchInfo (BinOp/UnOp/Cmp), uses EvalContext to check whether there are
/// same-type ready nodes in ready_queue. If there are ≥2 (including the current node), performs
/// SIMD batch computation and returns NodeResult::Batch; otherwise falls back to single-node computation.
/// For nodes without BatchInfo (Call/Gate/Await/record/array, etc.), follows the single-node path directly.
macro_rules! wrap_fn {
    ($f:expr) => {{
        fn wrapper(frame: &mut Frame, node: NodeId, ctx: &EvalContext) -> NodeResult {
            // safe_op short-circuit: nodes marked with ?. return Null when the receiver (inputs[0]) is null,
            // without executing subsequent computation (field access/method call/intrinsic, etc.).
            // A unified data-driven short-circuit logic, triggered by the compile-time set_safe_op flag.
            if frame.graph.safe_op_flag(node.0 as usize) {
                let n = ctx.graph.node(node.0 as usize);
                if n.input_count > 0 {
                    let inputs = ctx.graph.inputs(n.inputs_offset, n.input_count);
                    let recv = frame.get_value_by_global(inputs[0]);
                    if recv.is_null() {
                        return NodeResult::Value(Value::Null);
                    }
                }
            }
            // SIMD batch decision: check batch_infos; if there are same-type ready nodes, compute in batch
            if let Some(info) = ctx.graph.batch_info(node.0 as usize) {
                let graph = ctx.graph;
                let candidates = ctx.collect_batch_candidates(frame, node, |gid| {
                    graph.batch_info(gid.0 as usize) == Some(info)
                });
                if !candidates.is_empty() {
                    let mut all_locals = Vec::with_capacity(candidates.len() + 1);
                    all_locals.push(NodeId(node.0.wrapping_sub(ctx.node_start)));
                    for &gid in &candidates {
                        all_locals.push(NodeId(gid.0.wrapping_sub(ctx.node_start)));
                    }
                    if let Some(results) = super::Compute::do_simd_batch(
                        frame, &all_locals, info, ctx.node_start,
                    ) {
                        return NodeResult::Batch(results);
                    }
                }
            }
            NodeResult::Value($f(frame, node, ctx))
        }
        wrapper as ComputeFn
    }};
}

/// Compute function table registration macro.
///
/// Accepts a list of `idx => fn_path` pairs, expanding to a Vec construction with runtime index assertions.
/// After each push, immediately asserts `table.len() == idx + 1` to ensure the index matches the actual position.
/// If an entry is deleted but subsequent indices are not updated, the assertion fails immediately,
/// preventing ComputeFnId misalignment.
/// During the transition period, each entry is automatically wrapped with wrap_fn!.
macro_rules! compute_fn_table {
    ( $( $idx:literal => $f:expr ),* $(,)? ) => {{
        let mut table: Vec<ComputeFn> = Vec::new();
        $(
            table.push(wrap_fn!($f));
            assert_eq!(table.len(), ($idx as usize) + 1,
                concat!("compute_fn_table: index ", stringify!($idx), " mismatch"));
        )*
        table
    }};
}

/// Builds the real compute function table (references Engine module's compute_* functions).
///
/// Indices correspond one-to-one with ComputeFnId; IrBuilder::build() fills them into graph.compute_fns at the end.
/// Uses the `compute_fn_table!` macro: each `idx => fn_path` entry auto-generates a runtime assertion,
/// ensuring the index matches the actual position — if an entry is deleted but subsequent indices are not updated, the assertion fails immediately.
pub fn build_compute_fn_table() -> Vec<ComputeFn> {
    let mut table = compute_fn_table! {
        0   => super::Compute::noop_compute_real,
        1   => super::Compute::compute_add_i32,
        2   => super::Compute::compute_add_f64,
        3   => super::Compute::compute_mul_i32,
        4   => super::Compute::compute_le_i32,
        5   => super::Compute::compute_sub_i32,
        6   => super::Compute::compute_div_i32,
        7   => super::Compute::compute_mod_i32,
        8   => super::Compute::compute_eq_i32,
        9   => super::Compute::compute_ne_i32,
        10  => super::Compute::compute_lt_i32,
        11  => super::Compute::compute_gt_i32,
        12  => super::Compute::compute_ge_i32,
        13  => super::Compute::compute_sub_f64,
        14  => super::Compute::compute_mul_f64,
        15  => super::Compute::compute_div_f64,
        16  => super::Compute::compute_eq_f64,
        17  => super::Compute::compute_ne_f64,
        18  => super::Compute::compute_lt_f64,
        19  => super::Compute::compute_gt_f64,
        20  => super::Compute::compute_le_f64,
        21  => super::Compute::compute_ge_f64,
        22  => super::Compute::compute_and_bool,
        23  => super::Compute::compute_or_bool,
        24  => super::Compute::compute_not_bool,
        25  => super::Compute::compute_neg_i32,
        26  => super::Compute::compute_neg_f64,
        27  => super::Compute::compute_eq_bool,
        28  => super::Compute::noop_compute_real, // compute_throw_wrap_err — new signature, table override
        29  => super::Compute::compute_record_construct,
        30  => super::Compute::compute_record_field_get,
        31  => super::Compute::compute_array_construct,
        32  => super::Compute::compute_array_index,
        33  => super::Compute::compute_record_field_set,
        34  => super::Compute::compute_is_null,
        35  => super::Compute::compute_array_len,
        36  => super::Compute::noop_compute_real, // compute_call_launch — new signature, table override
        37  => super::Compute::noop_compute_real, // compute_gate_launch — new signature, table override
        38  => super::Compute::noop_compute_real, // compute_await — new signature, table override
        39  => super::Compute::noop_compute_real, // compute_call_launch alias — new signature, table override
        40  => super::Compute::compute_closure_construct,
        41  => super::Compute::noop_compute_real, // compute_closure_call — new signature, table override
        42  => super::Compute::noop_compute_real, // compute_cancel_async_handle — new signature, table override
        43  => super::Compute::noop_compute_real, // compute_select_gate — new signature, table override
        44  => super::Compute::compute_throw_ok,
        45  => super::Compute::compute_throw_err,
        46  => super::Compute::noop_compute_real, // CF_FFI_CALL is deprecated (wrapper table deleted); the slot is kept to avoid renumbering.
        47  => super::Compute::noop_compute_real, // compute_propagate — new signature, table override
        48  => super::Compute::compute_seq,
        49  => super::Compute::noop_compute_real, // compute_writeback — new signature, table override
        // i64 arithmetic and comparison (50-63)
        50  => super::Compute::compute_add_i64,
        51  => super::Compute::compute_sub_i64,
        52  => super::Compute::compute_mul_i64,
        53  => super::Compute::compute_div_i64,
        54  => super::Compute::compute_mod_i64,
        55  => super::Compute::compute_eq_i64,
        56  => super::Compute::compute_ne_i64,
        57  => super::Compute::compute_lt_i64,
        58  => super::Compute::compute_gt_i64,
        59  => super::Compute::compute_le_i64,
        60  => super::Compute::compute_ge_i64,
        61  => super::Compute::compute_neg_i64,
        62  => super::Compute::compute_bitnot_i32,
        63  => super::Compute::compute_bitnot_i64,
        // i128 arithmetic and comparison (64-77)
        64  => super::Compute::compute_add_i128,
        65  => super::Compute::compute_sub_i128,
        66  => super::Compute::compute_mul_i128,
        67  => super::Compute::compute_div_i128,
        68  => super::Compute::compute_mod_i128,
        69  => super::Compute::compute_eq_i128,
        70  => super::Compute::compute_ne_i128,
        71  => super::Compute::compute_lt_i128,
        72  => super::Compute::compute_gt_i128,
        73  => super::Compute::compute_le_i128,
        74  => super::Compute::compute_ge_i128,
        75  => super::Compute::compute_neg_i128,
        76  => super::Compute::compute_bitnot_i128,
        // Integer bitwise operations (77-92): BitAnd/BitOr/BitXor/Shl/Shr × i32/i64/i128
        77  => super::Compute::compute_bitand_i32,
        78  => super::Compute::compute_bitor_i32,
        79  => super::Compute::compute_bitxor_i32,
        80  => super::Compute::compute_bitand_i64,
        81  => super::Compute::compute_bitor_i64,
        82  => super::Compute::compute_bitxor_i64,
        83  => super::Compute::compute_bitand_i128,
        84  => super::Compute::compute_bitor_i128,
        85  => super::Compute::compute_bitxor_i128,
        86  => super::Compute::compute_shl_i32,
        87  => super::Compute::compute_shr_i32,
        88  => super::Compute::compute_shl_i64,
        89  => super::Compute::compute_shr_i64,
        90  => super::Compute::compute_shl_i128,
        91  => super::Compute::compute_shr_i128,
        // ---- compute_fn for all primitive types (92-259) ----
        // 12 integer types × 12 operations (add/sub/mul/div/mod/bitand/bitor/bitxor/shl/shr/neg/bitnot)
        // i8: 92-103
        92  => super::Compute::compute_add_i8,
        93  => super::Compute::compute_sub_i8,
        94  => super::Compute::compute_mul_i8,
        95  => super::Compute::compute_div_i8,
        96  => super::Compute::compute_mod_i8,
        97  => super::Compute::compute_bitand_i8,
        98  => super::Compute::compute_bitor_i8,
        99  => super::Compute::compute_bitxor_i8,
        100 => super::Compute::compute_shl_i8,
        101 => super::Compute::compute_shr_i8,
        102 => super::Compute::compute_neg_i8,
        103 => super::Compute::compute_bitnot_i8,
        // i16: 104-115
        104 => super::Compute::compute_add_i16,
        105 => super::Compute::compute_sub_i16,
        106 => super::Compute::compute_mul_i16,
        107 => super::Compute::compute_div_i16,
        108 => super::Compute::compute_mod_i16,
        109 => super::Compute::compute_bitand_i16,
        110 => super::Compute::compute_bitor_i16,
        111 => super::Compute::compute_bitxor_i16,
        112 => super::Compute::compute_shl_i16,
        113 => super::Compute::compute_shr_i16,
        114 => super::Compute::compute_neg_i16,
        115 => super::Compute::compute_bitnot_i16,
        // i32: 116-127
        116 => super::Compute::compute_add_i32,
        117 => super::Compute::compute_sub_i32,
        118 => super::Compute::compute_mul_i32,
        119 => super::Compute::compute_div_i32,
        120 => super::Compute::compute_mod_i32,
        121 => super::Compute::compute_bitand_i32,
        122 => super::Compute::compute_bitor_i32,
        123 => super::Compute::compute_bitxor_i32,
        124 => super::Compute::compute_shl_i32,
        125 => super::Compute::compute_shr_i32,
        126 => super::Compute::compute_neg_i32,
        127 => super::Compute::compute_bitnot_i32,
        // i64: 128-139
        128 => super::Compute::compute_add_i64,
        129 => super::Compute::compute_sub_i64,
        130 => super::Compute::compute_mul_i64,
        131 => super::Compute::compute_div_i64,
        132 => super::Compute::compute_mod_i64,
        133 => super::Compute::compute_bitand_i64,
        134 => super::Compute::compute_bitor_i64,
        135 => super::Compute::compute_bitxor_i64,
        136 => super::Compute::compute_shl_i64,
        137 => super::Compute::compute_shr_i64,
        138 => super::Compute::compute_neg_i64,
        139 => super::Compute::compute_bitnot_i64,
        // i128: 140-151
        140 => super::Compute::compute_add_i128,
        141 => super::Compute::compute_sub_i128,
        142 => super::Compute::compute_mul_i128,
        143 => super::Compute::compute_div_i128,
        144 => super::Compute::compute_mod_i128,
        145 => super::Compute::compute_bitand_i128,
        146 => super::Compute::compute_bitor_i128,
        147 => super::Compute::compute_bitxor_i128,
        148 => super::Compute::compute_shl_i128,
        149 => super::Compute::compute_shr_i128,
        150 => super::Compute::compute_neg_i128,
        151 => super::Compute::compute_bitnot_i128,
        // u8: 152-163
        152 => super::Compute::compute_add_u8,
        153 => super::Compute::compute_sub_u8,
        154 => super::Compute::compute_mul_u8,
        155 => super::Compute::compute_div_u8,
        156 => super::Compute::compute_mod_u8,
        157 => super::Compute::compute_bitand_u8,
        158 => super::Compute::compute_bitor_u8,
        159 => super::Compute::compute_bitxor_u8,
        160 => super::Compute::compute_shl_u8,
        161 => super::Compute::compute_shr_u8,
        162 => super::Compute::compute_neg_u8,
        163 => super::Compute::compute_bitnot_u8,
        // u16: 164-175
        164 => super::Compute::compute_add_u16,
        165 => super::Compute::compute_sub_u16,
        166 => super::Compute::compute_mul_u16,
        167 => super::Compute::compute_div_u16,
        168 => super::Compute::compute_mod_u16,
        169 => super::Compute::compute_bitand_u16,
        170 => super::Compute::compute_bitor_u16,
        171 => super::Compute::compute_bitxor_u16,
        172 => super::Compute::compute_shl_u16,
        173 => super::Compute::compute_shr_u16,
        174 => super::Compute::compute_neg_u16,
        175 => super::Compute::compute_bitnot_u16,
        // u32: 176-187
        176 => super::Compute::compute_add_u32,
        177 => super::Compute::compute_sub_u32,
        178 => super::Compute::compute_mul_u32,
        179 => super::Compute::compute_div_u32,
        180 => super::Compute::compute_mod_u32,
        181 => super::Compute::compute_bitand_u32,
        182 => super::Compute::compute_bitor_u32,
        183 => super::Compute::compute_bitxor_u32,
        184 => super::Compute::compute_shl_u32,
        185 => super::Compute::compute_shr_u32,
        186 => super::Compute::compute_neg_u32,
        187 => super::Compute::compute_bitnot_u32,
        // u64: 188-199
        188 => super::Compute::compute_add_u64,
        189 => super::Compute::compute_sub_u64,
        190 => super::Compute::compute_mul_u64,
        191 => super::Compute::compute_div_u64,
        192 => super::Compute::compute_mod_u64,
        193 => super::Compute::compute_bitand_u64,
        194 => super::Compute::compute_bitor_u64,
        195 => super::Compute::compute_bitxor_u64,
        196 => super::Compute::compute_shl_u64,
        197 => super::Compute::compute_shr_u64,
        198 => super::Compute::compute_neg_u64,
        199 => super::Compute::compute_bitnot_u64,
        // u128: 200-211
        200 => super::Compute::compute_add_u128,
        201 => super::Compute::compute_sub_u128,
        202 => super::Compute::compute_mul_u128,
        203 => super::Compute::compute_div_u128,
        204 => super::Compute::compute_mod_u128,
        205 => super::Compute::compute_bitand_u128,
        206 => super::Compute::compute_bitor_u128,
        207 => super::Compute::compute_bitxor_u128,
        208 => super::Compute::compute_shl_u128,
        209 => super::Compute::compute_shr_u128,
        210 => super::Compute::compute_neg_u128,
        211 => super::Compute::compute_bitnot_u128,
        // isize: 212-223
        212 => super::Compute::compute_add_isize,
        213 => super::Compute::compute_sub_isize,
        214 => super::Compute::compute_mul_isize,
        215 => super::Compute::compute_div_isize,
        216 => super::Compute::compute_mod_isize,
        217 => super::Compute::compute_bitand_isize,
        218 => super::Compute::compute_bitor_isize,
        219 => super::Compute::compute_bitxor_isize,
        220 => super::Compute::compute_shl_isize,
        221 => super::Compute::compute_shr_isize,
        222 => super::Compute::compute_neg_isize,
        223 => super::Compute::compute_bitnot_isize,
        // usize: 224-235
        224 => super::Compute::compute_add_usize,
        225 => super::Compute::compute_sub_usize,
        226 => super::Compute::compute_mul_usize,
        227 => super::Compute::compute_div_usize,
        228 => super::Compute::compute_mod_usize,
        229 => super::Compute::compute_bitand_usize,
        230 => super::Compute::compute_bitor_usize,
        231 => super::Compute::compute_bitxor_usize,
        232 => super::Compute::compute_shl_usize,
        233 => super::Compute::compute_shr_usize,
        234 => super::Compute::compute_neg_usize,
        235 => super::Compute::compute_bitnot_usize,
        // 4 floating-point types × 6 operations (add/sub/mul/div/mod/neg)
        // f16: 236-241
        236 => super::Compute::compute_add_f16,
        237 => super::Compute::compute_sub_f16,
        238 => super::Compute::compute_mul_f16,
        239 => super::Compute::compute_div_f16,
        240 => super::Compute::compute_mod_f16,
        241 => super::Compute::compute_neg_f16,
        // f32: 242-247
        242 => super::Compute::compute_add_f32,
        243 => super::Compute::compute_sub_f32,
        244 => super::Compute::compute_mul_f32,
        245 => super::Compute::compute_div_f32,
        246 => super::Compute::compute_mod_f32,
        247 => super::Compute::compute_neg_f32,
        // f64: 248-253
        248 => super::Compute::compute_add_f64,
        249 => super::Compute::compute_sub_f64,
        250 => super::Compute::compute_mul_f64,
        251 => super::Compute::compute_div_f64,
        252 => super::Compute::compute_mod_f64,
        253 => super::Compute::compute_neg_f64,
        // f128: 254-259
        254 => super::Compute::compute_add_f128,
        255 => super::Compute::compute_sub_f128,
        256 => super::Compute::compute_mul_f128,
        257 => super::Compute::compute_div_f128,
        258 => super::Compute::compute_mod_f128,
        259 => super::Compute::compute_neg_f128,
        // Semantic operations (260-265): RefEq/RefNeq/ConcatList/Range/RangeInclusive/Elvis
        260 => super::Compute::compute_ref_eq,
        261 => super::Compute::compute_ref_neq,
        262 => super::Compute::compute_concat_list,
        263 => super::Compute::compute_range,
        264 => super::Compute::compute_range_inclusive,
        265 => super::Compute::compute_elvis,
        // inline_trait / lazy construction (266-267)
        266 => super::Compute::compute_trait_construct,
        267 => super::Compute::compute_lazy_construct,
        268 => super::Compute::compute_slice,
        269 => super::Compute::compute_str_concat,
        // Global variable read/write (270-271)
        270 => super::Compute::compute_global_load,
        271 => super::Compute::compute_global_store,
        // Record extension / atomic construction (272-273)
        272 => super::Compute::compute_record_extend,
        273 => super::Compute::compute_atomic_construct,
        // Pattern matching (274-276)
        274 => super::Compute::compute_pattern_ctor_match,
        275 => super::Compute::compute_pattern_adt_field_get,
        276 => super::Compute::compute_pattern_str_eq,
        // General type conversion (277-278)
        277 => super::Compute::compute_cast_to_str,
        278 => super::Compute::compute_cast_scalar,
        // Reference semantics and non-null assertion (279-282)
        279 => super::Compute::compute_non_null_assert,
        280 => super::Compute::compute_ref_of,
        281 => super::Compute::compute_deref_read,
        282 => super::Compute::compute_deref_write,
        // Channel operations (283-285)
        283 => super::Compute::compute_channel_create,
        284 => super::Compute::noop_compute_real, // compute_channel_send — new signature, table override
        285 => super::Compute::compute_channel_close,
        // Partial application construction (286)
        286 => super::Compute::compute_partial_construct,
        // str.bytes() → u8[] (287)
        287 => super::Compute::compute_str_bytes,
        // Stack-allocated construction (288-289): uses non-escaping allocation points marked by the analyzer
        288 => super::Compute::compute_record_construct_stack,
        289 => super::Compute::compute_array_construct_stack,
        // Standalone reflect compute_fn (290-291): lazy force + Reflect::format_value
        290 => super::Compute::compute_reflect_format,
        291 => super::Compute::compute_reflect_scalar_to_str,
        // String comparison (292-297): lexicographic by Unicode code point sequence
        292 => super::Compute::compute_eq_str,
        293 => super::Compute::compute_ne_str,
        294 => super::Compute::compute_lt_str,
        295 => super::Compute::compute_gt_str,
        296 => super::Compute::compute_le_str,
        297 => super::Compute::compute_ge_str,
        // Semantic equality/inequality for composite types (298-299)
        298 => super::Compute::compute_eq_obj,
        299 => super::Compute::compute_ne_obj,
        // Boolean inequality (300)
        300 => super::Compute::compute_ne_bool,
        // Array index store (301)
        301 => super::Compute::compute_array_store,
        // f128 comparison (302-307)
        302 => super::Compute::compute_eq_f128,
        303 => super::Compute::compute_ne_f128,
        304 => super::Compute::compute_lt_f128,
        305 => super::Compute::compute_gt_f128,
        306 => super::Compute::compute_le_f128,
        307 => super::Compute::compute_ge_f128,
        // Memoization cache (308-309)
        308 => super::Compute::compute_memo_check,
        309 => super::Compute::compute_memo_store,
        // Tail-recursion WriteBack (310)
        310 => super::Compute::noop_compute_real, // compute_tailrec_writeback — new signature, table override
        // Control-flow compute_fn (311-314) — new signature, table override
        311 => super::Compute::noop_compute_real, // compute_return
        312 => super::Compute::noop_compute_real, // compute_break
        313 => super::Compute::noop_compute_real, // compute_continue
        314 => super::Compute::noop_compute_real, // compute_match_fallback
        // Atomic operations (315-318): load/store/swap/compare_exchange on Atomic<T>
        315 => super::Compute::compute_atomic_load,
        316 => super::Compute::compute_atomic_store,
        317 => super::Compute::compute_atomic_swap,
        318 => super::Compute::compute_atomic_compare_exchange,
        // Multi-input string concat (319): one-shot O(n) concat for string interpolation
        319 => super::Compute::compute_str_multi_concat,
        // Array join (320): str[] + sep → str, one-shot O(n) concat
        320 => super::Compute::compute_str_array_join,
        // Array fill (321): [value, ..count] — repeats value count times
        321 => super::Compute::compute_array_fill,
        // Runtime defer (322-323): new signature, table override
        322 => super::Compute::noop_compute_real, // compute_defer_register
        323 => super::Compute::noop_compute_real, // compute_defer_run
        324 => super::Compute::noop_compute_real, // compute_block_defer_register
        325 => super::Compute::compute_dyn_ffi_call,
        // reflect compute_fns (326-336): standalone reflect primitives.
        // Replaces @builtin + REFLECT_ENTRIES + CF_FFI_CALL dispatch.
        326 => super::Compute::compute_reflect_kind,
        327 => super::Compute::compute_reflect_type_name,
        328 => super::Compute::compute_reflect_kind_str,
        329 => super::Compute::compute_reflect_size,
        330 => super::Compute::compute_reflect_layout_size,
        331 => super::Compute::compute_reflect_layout_align,
        332 => super::Compute::compute_reflect_field_count,
        333 => super::Compute::compute_reflect_field_name,
        334 => super::Compute::compute_reflect_field_value,
        335 => super::Compute::compute_reflect_array_len,
        336 => super::Compute::compute_reflect_adt_ctor,
    };
    // Replace index 0 with compute_const (unwrapped, uses the new signature directly)
    // Const nodes use CF_NOOP(0); compute_const materializes the value from const_values
    table[0] = super::Compute::compute_const;
    // compute_fn entries migrated to the new signature (not wrapped via wrap_fn!, use the new signature directly)
    table[28] = super::Compute::compute_throw_wrap_err;
    table[36] = super::Compute::compute_call_launch;
    table[37] = super::Compute::compute_gate_launch;
    table[38] = super::Compute::compute_await;
    table[39] = super::Compute::compute_call_launch; // CF_ASYNC_CALL_LAUNCH alias
    table[41] = super::Compute::compute_closure_call;
    table[42] = super::Compute::compute_cancel_async_handle;
    table[43] = super::Compute::compute_select_gate;
    table[47] = super::Compute::compute_propagate;
    table[284] = super::Compute::compute_channel_send;
    table[310] = super::Compute::compute_tailrec_writeback;
    table[311] = super::Compute::compute_return;
    table[312] = super::Compute::compute_break;
    table[313] = super::Compute::compute_continue;
    table[314] = super::Compute::compute_match_fallback;
    table[49] = super::Compute::compute_writeback;
    table[322] = super::Compute::compute_defer_register;
    table[323] = super::Compute::compute_defer_run;
    table[324] = super::Compute::compute_block_defer_register;
    assert_eq!(
        table.len() as u32, COMPUTE_FN_TABLE_LEN,
        "build_compute_fn_table(): table length drifted from COMPUTE_FN_TABLE_LEN; update the constant"
    );
    table
}

/// Set of pure compute_fn (no side effects; eligible for CSE/DCE).
///
/// Derived from `effect_class` (W1); the derivation's equivalence with the
/// pre-W1 hand-written set is asserted by the unit test at the bottom of this
/// file. Aliasing heap reads are folded in by `aliasing_read_cfs`; their
/// movement safety is guaranteed by W2 storage-version edges (see
/// `graph_pure_set`), not by subtracting them from this set.
///
/// The four aliased heap reads (`is_versioned_read_cf` subset) folded into the
/// CSE/LICM pure set by `pure_compute_fn_set`. Their movement safety no longer
/// relies on purity subtraction (the pre-W2 Bug #99 stopgap, removed): the
/// storage-versioning pass (`Builder::Versioning`) attaches version edges that
/// make CSE/LICM see mutations through shared Arcs directly.
pub fn aliasing_read_cfs() -> [ComputeFnId; 4] {
    [CF_RECORD_FIELD_GET, CF_ARRAY_INDEX, CF_ARRAY_LEN, CF_PATTERN_ADT_FIELD_GET]
}

// =========================================================================
// EffectClass — single source of truth for per-CF effect classification (W1)
// =========================================================================

/// Effect classification of a compute function: the ONE place every pass asks
/// "what can be done with a node of this cf" (move / merge / duplicate /
/// delete / reorder). Adding a new CF means adding one classification here —
/// the pass layer must not keep its own lists.
///
/// `pure_compute_fn_set()` (CSE/LICM/DCE eligibility) is derived from this
/// classification; the derivation is equivalence-tested against the previous
/// hand-written set (see `effect_classification_tests`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectClass {
    /// No side effects, deterministic per input: reorder/merge/duplicate/delete
    /// freely.
    Pure,
    /// Pure value-metadata reads (reflect kind/type-name/size/...). Semantically
    /// pure, but kept OUT of the CSE/LICM set to preserve pre-W1 behavior until
    /// interning/metadata-hash stability is validated (W2).
    PureMeta,
    /// Allocates a distinct object per run (record/array/closure/range/slice/
    /// string builds): not CSE/LICM-able — sharing one instance across loop
    /// iterations would pollute state.
    Alloc,
    /// Reads mutable state that is NOT tracked as a dataflow input edge (heap
    /// field/array/deref/global/atomic/memo-cache reads). Pure only in graphs
    /// without writers to the same state (see `aliasing_read_cfs`).
    ReadMutable,
    /// In-place write to mutable state (field set / array store / deref write /
    /// atomic RMW / memo store).
    WriteMutable,
    /// Writes a graph location: WriteBack / TailRec WriteBack / global store.
    WriteLocal,
    /// Produces a non-local control signal or dispatches control (return/break/
    /// continue/throw/propagate/match-fallback, Gate launch).
    ControlFlow,
    /// Effect-chain ordering node (CF_SEQ): value-transparent sequencer.
    Seq,
    /// Interacts with the async runtime (await/async launch/select/channel
    /// send/close/cancel).
    Async,
    /// Launches a subgraph/closure; effect depends on the callee.
    Call,
    /// Foreign function call: assume ANY effect.
    Ffi,
    /// Engine/frame-level effect (defer register/run).
    Runtime,
}

/// Classify a compute function. Panics on unclassified ids so a newly added CF
/// cannot silently default to a wrong class — extend this match (and, if the
/// table grew, `COMPUTE_FN_TABLE_LEN`) in the same commit.
pub fn effect_class(cf: ComputeFnId) -> EffectClass {
    use EffectClass::*;
    match cf.0 {
        // ── Pure: legacy arithmetic/comparison ranges (equivalence-tested) ──
        1..=27 | 50..=91 | 92..=259 => Pure,
        34 | 260 | 261 | 265 | 274 | 276 | 278 | 279 | 287 => Pure, // reads/queries
        292..=300 | 302..=307 => Pure, // string/obj/bool/f128 comparison
        // ── Reads of mutable state (aliasing) ──
        30 | 32 | 35 | 275 => ReadMutable,      // field/array/pattern-ADT reads (aliasing_read_cfs)
        270 | 281 | 308 | 315 | 334 | 335 => ReadMutable, // global/deref/memo/atomic/reflect-value reads
        // ── In-place writes ──
        33 | 282 | 301 | 309 | 316 | 317 | 318 => WriteMutable,
        // ── Graph-location writes ──
        49 | 271 | 310 => WriteLocal, // writeback / global_store / tailrec writeback
        // ── Control flow / dispatch ──
        28 | 37 | 47 | 311 | 312 | 313 | 314 => ControlFlow,
        // ── Ordering ──
        48 => Seq, // CF_SEQ
        // ── Async runtime ──
        38 | 39 | 42 | 43 | 284 | 285 => Async,
        // ── Launches ──
        36 | 41 => Call, // call_launch / closure_call
        // ── FFI ──
        46 | 325 => Ffi,
        // ── Engine/frame effects ──
        322 | 323 | 324 => Runtime, // defer register/run/block-register
        // ── Allocation (distinct object per run) ──
        29 | 31 | 40 | 44 | 45 | 262 | 263 | 264 | 266 | 267 | 268 | 269 | 272 | 273
        | 277 | 280 | 283 | 286 | 288 | 289 | 290 | 291 | 319 | 320 | 321 => Alloc,
        // ── Pure value metadata (kept out of CSE/LICM pending W2 validation) ──
        326..=333 | 336 => PureMeta,
        // CF_NOOP: parameter placeholder passthrough.
        0 => Pure,
        other => panic!(
            "effect_class: unclassified compute fn {} — classify it in Ir::effect_class",
            other
        ),
    }
}

/// Aliased heap reads that participate in storage versioning (W2): the read's
/// storage root is always `inputs[0]`. Shared between the Builder versioning
/// pass and the Verifier's V6 completeness check.
pub fn is_versioned_read_cf(cf: ComputeFnId) -> bool {
    matches!(
        cf,
        CF_RECORD_FIELD_GET
            | CF_ARRAY_INDEX
            | CF_ARRAY_LEN
            | CF_PATTERN_ADT_FIELD_GET
            | CF_DEREF_READ
            | CF_ATOMIC_LOAD
            | CF_REFLECT_FIELD_VALUE
            | CF_REFLECT_ARRAY_LEN
    )
}

/// In-place heap writes that participate in storage versioning (W2): the
/// write's storage root is always `inputs[0]`; the write node becomes the
/// root's new version.
pub fn is_versioned_write_cf(cf: ComputeFnId) -> bool {
    matches!(
        cf,
        CF_RECORD_FIELD_SET
            | CF_ARRAY_STORE
            | CF_DEREF_WRITE
            | CF_ATOMIC_STORE
            | CF_ATOMIC_SWAP
            | CF_ATOMIC_COMPARE_EXCHANGE
    )
}

/// The CSE/LICM/DCE-pure set for a specific graph.
///
/// W2: aliased heap reads stay eligible — their correctness when moved is
/// guaranteed by storage-version edges attached at build time
/// (`Builder::Versioning`): reads inside mutating loop bodies carry a
/// loop-internal input (cond_node), so LICM cannot hoist them and CSE cannot
/// merge reads across a write (their version inputs differ). The pre-W2
/// blanket subtraction of aliasing reads (Bug #99 stopgap) is gone.
pub fn graph_pure_set(_graph: &DataFlowGraph) -> rustc_hash::FxHashSet<ComputeFnId> {
    pure_compute_fn_set()
}

/// Scheduler-visible node kinds that launch or gate runtime work. Passes must
/// not freely move/delete/merge these regardless of purity. The complement
/// (Const/BinOp/TriOp/UnOp/FieldAccess) is the "pure computation" kind set.
pub fn is_launch_kind(kind: NodeKind) -> bool {
    matches!(
        kind,
        NodeKind::Gate | NodeKind::Call | NodeKind::Await | NodeKind::EventSource
    )
}

pub fn pure_compute_fn_set() -> rustc_hash::FxHashSet<ComputeFnId> {
    // Derived from effect_class (W1). Equivalence with the pre-W1 hand-written
    // set is asserted by the unit test at the bottom of this file.
    // CF_NOOP is excluded: it is the parameter-placeholder passthrough, and
    // CSE-merging two 0-input NOOP nodes would merge distinct parameters.
    let mut s = rustc_hash::FxHashSet::default();
    for id in 0..COMPUTE_FN_TABLE_LEN {
        let cf = ComputeFnId(id);
        if cf == CF_NOOP {
            continue;
        }
        if effect_class(cf) == EffectClass::Pure {
            s.insert(cf);
        }
    }
    // Aliasing heap reads are pure ONLY in graphs without in-place mutators;
    // the mutator-aware subtraction lives in graph_pure_set.
    for cf in aliasing_read_cfs() {
        s.insert(cf);
    }
    s
}

// =========================================================================
// Node metadata macros: eliminate the 4-part declaration/new/add_node/setter boilerplate for NodeId-indexed fields
// =========================================================================
//
// The central definition macro `node_metadata!($callback)` lists all NodeId-indexed metadata fields,
// expanded via different callback macros into:
//   - add_node() push (metadata_push!)       ← auto-synced
//   - setter methods (metadata_setters!)      ← auto-synced
//   - struct field declarations               ← manual (Rust does not allow macros to expand here)
//   - new() initialization                    ← manual (same reason)
//
// Three field categories:
//   opt(field, Type, setter)   → Vec<Option<Type>>, set_setter(node, v: Type)
//   bool_flag(field, setter)   → Vec<bool>, set_setter(node) { = true }
//   bool_val(field, setter)    → Vec<bool>, set_setter(node, v: bool) { = v }
//
// To add a new metadata field: append a line to node_metadata! (push+setter auto-sync),
// then add a line to both the struct definition and new() (manual).

/// Central definition: all NodeId-indexed metadata fields.
macro_rules! node_metadata {
    ($callback:ident) => {
        $callback! {
            opt(call_targets, SubGraphId, set_call_target)
            opt(gate_branches, GateBranches, set_gate_branches)
            opt(field_access_infos, u16, set_field_access_info)
            opt(record_lit_infos, RecordLitInfo, set_record_lit_info)
            opt(ffi_call_names, String, set_ffi_call_name)
            opt(dyn_ffi_infos, DynFfiInfo, set_dyn_ffi_info)
            opt(field_set_names, String, set_field_set_name)
            opt(vtable_call_methods, u16, set_vtable_call)
            opt(await_event_sources, NodeId, set_await_event_source)
            opt(closure_infos, ClosureInfo, set_closure_info)
            opt(partial_infos, PartialInfo, set_partial_info)
            opt(closure_call_arg_counts, u8, set_closure_call_arg_count)
            opt(select_infos, SelectInfo, set_select_info)
            opt(writeback_targets, NodeId, set_writeback_target)
            opt(batch_infos, BatchInfo, set_batch_info)
            opt(trait_construct_infos, TraitConstructInfo, set_trait_construct_info)
            opt(lazy_construct_infos, LazyConstructInfo, set_lazy_construct_info)
            opt(record_extend_infos, RecordExtendInfo, set_record_extend_info)
            opt(global_load_slots, u32, set_global_load_slot)
            opt(global_store_slots, u32, set_global_store_slot)
            opt(pattern_ctor_names, String, set_pattern_ctor_name)
            opt(pattern_type_names, String, set_pattern_type_name)
            opt(pattern_field_indices, u16, set_pattern_field_index)
            opt(cast_target_types, String, set_cast_target_type)
            opt(memo_infos, MemoInfo, set_memo_info)
            ;
            bool_flag(tail_call_flags, set_tail_call)
            bool_flag(safe_op_flags, set_safe_op)
            bool_flag(hoisted_node, set_hoisted)
            ;
            bool_val(slice_inclusive, set_slice_inclusive)
        }
    };
    ($callback:ident, $self:ident $(, $extra:ident)*) => {
        $callback! {
            $self $(, $extra)* ;
            opt(call_targets, SubGraphId, set_call_target)
            opt(gate_branches, GateBranches, set_gate_branches)
            opt(field_access_infos, u16, set_field_access_info)
            opt(record_lit_infos, RecordLitInfo, set_record_lit_info)
            opt(ffi_call_names, String, set_ffi_call_name)
            opt(dyn_ffi_infos, DynFfiInfo, set_dyn_ffi_info)
            opt(field_set_names, String, set_field_set_name)
            opt(vtable_call_methods, u16, set_vtable_call)
            opt(await_event_sources, NodeId, set_await_event_source)
            opt(closure_infos, ClosureInfo, set_closure_info)
            opt(partial_infos, PartialInfo, set_partial_info)
            opt(closure_call_arg_counts, u8, set_closure_call_arg_count)
            opt(select_infos, SelectInfo, set_select_info)
            opt(writeback_targets, NodeId, set_writeback_target)
            opt(batch_infos, BatchInfo, set_batch_info)
            opt(trait_construct_infos, TraitConstructInfo, set_trait_construct_info)
            opt(lazy_construct_infos, LazyConstructInfo, set_lazy_construct_info)
            opt(record_extend_infos, RecordExtendInfo, set_record_extend_info)
            opt(global_load_slots, u32, set_global_load_slot)
            opt(global_store_slots, u32, set_global_store_slot)
            opt(pattern_ctor_names, String, set_pattern_ctor_name)
            opt(pattern_type_names, String, set_pattern_type_name)
            opt(pattern_field_indices, u16, set_pattern_field_index)
            opt(cast_target_types, String, set_cast_target_type)
            opt(memo_infos, MemoInfo, set_memo_info)
            ;
            bool_flag(tail_call_flags, set_tail_call)
            bool_flag(safe_op_flags, set_safe_op)
            bool_flag(hoisted_node, set_hoisted)
            ;
            bool_val(slice_inclusive, set_slice_inclusive)
        }
    };
}

// Note: struct field declarations and new() initializers cannot be macro-ified with macro_rules! —
// Rust does not allow macros to expand at struct field declaration positions or struct initializer field positions.
// These two places must be hand-written (see DataFlowGraph definition and new()). To add a new field:
//   1. Append a line to node_metadata! (auto-syncs push + setter)
//   2. Add a field line to the struct definition
//   3. Add a Vec::new() line to new()

/// Expands to push statements in add_node().
macro_rules! metadata_push {
    ( $self:ident ; $( opt($f:ident, $t:ty, $_s:ident) )* ; $( bool_flag($bf:ident, $_bs:ident) )* ; $( bool_val($vf:ident, $_vs:ident) )* ) => {
        $( $self.$f.push(None); )*
        $( $self.$bf.push(false); )*
        $( $self.$vf.push(false); )*
    };
}

/// Expands to setter methods in impl DataFlowGraph.
macro_rules! metadata_setters {
    ( $( opt($f:ident, $t:ty, $s:ident) )* ; $( bool_flag($bf:ident, $bs:ident) )* ; $( bool_val($vf:ident, $vs:ident) )* ) => {
        $(
            pub fn $s(&mut self, node: NodeId, v: $t) {
                self.$f[node.0 as usize] = Some(v);
            }
        )*
        $(
            pub fn $bs(&mut self, node: NodeId) {
                self.$bf[node.0 as usize] = true;
            }
        )*
        $(
            pub fn $vs(&mut self, node: NodeId, v: bool) {
                self.$vf[node.0 as usize] = v;
            }
        )*
    };
}

/// Expands to per-field clone statements in clone_node_metadata().
/// opt fields uniformly use `.clone()` (clone on Copy types is equivalent to a copy);
/// bool_flag/bool_val are Copy and assigned directly.
macro_rules! metadata_clone {
    ( $self:ident, $src:ident, $dst:ident ;
      $( opt($f:ident, $t:ty, $_s:ident) )* ;
      $( bool_flag($bf:ident, $_bs:ident) )* ;
      $( bool_val($vf:ident, $_vs:ident) )* ) => {
        $( $self.$f[$dst] = $self.$f[$src].clone(); )*
        $( $self.$bf[$dst] = $self.$bf[$src]; )*
        $( $self.$vf[$dst] = $self.$vf[$src]; )*
    };
}

// =========================================================================
// DataFlowGraph — global graph container
// =========================================================================

/// Dataflow graph: a global read-only container shared by all workers.
///
/// - `nodes`: all nodes (globally contiguous)
/// - `inputs_pool`: all inputs (contiguous storage)
/// - `subgraphs`: all function subgraphs
/// - `entry_subgraph`: program entry subgraph
/// - `downstreams`: downstream list per node (fan-out statistics, used for slot-level RC)
pub struct DataFlowGraph {
    /// All nodes (globally contiguous, indexed by NodeId).
    pub nodes: Vec<Node>,
    /// Standalone input pool.
    pub inputs_pool: InputsPool,
    /// All function subgraphs.
    pub subgraphs: Vec<SubGraph>,
    /// Program entry subgraph.
    pub entry_subgraph: Option<SubGraphId>,
    /// Compute function table (filled at build time; invoked by ComputeFnId index at runtime).
    pub compute_fns: Vec<ComputeFn>,
    /// Downstream list per node (fan-out statistics; downstreams[n] = list of downstream nodes that use node n as input).
    pub downstreams: Vec<Vec<NodeId>>,
    /// Raw values for constant nodes (indexed by NodeId; None for non-Const nodes).
    pub const_values: Vec<Option<ConstValue>>,
    /// E0 perf: one-time materialized Const values, parallel to `nodes` (empty until the engine
    /// populates it at startup — see EngineRef::new). Non-const slots hold `Value::VOID`.
    /// Derived data: never serialized; recomputed at engine start (after build/optimize/load),
    /// so optimizer `rebuild` renumbering cannot invalidate it.
    pub const_cache: Vec<crate::value::Value>,
    /// E3 perf: per-subgraph initial pending_inputs templates for cross-function frames
    /// (indexed by sg idx; includes PENDING_EXTERNAL sentinels for nested/EventSource nodes).
    /// Derived data: populated at EngineRef::new (after build/optimize/load) — never serialized.
    pub sg_initial_pending: Vec<Vec<u16>>,
    /// E3 perf: per-subgraph ready-queue seed lists (local NodeIds of 0-input non-Param
    /// non-nested nodes), parallel to sg_initial_pending.
    pub sg_initial_seed: Vec<Vec<NodeId>>,
    pub downstream_counts: Vec<u16>,
    /// E5 perf: per-subgraph linearized execution plans (topological order of sg-own
    /// nodes; None = not linearizable: EventSource present or cyclic). Populated once at
    /// EngineRef::new — derived data, never serialized (F-7 safe by construction).
    pub linear_plans: Vec<Option<Vec<NodeId>>>,
    /// Target subgraph for Call nodes (indexed by NodeId; None for non-Call nodes).
    pub call_targets: Vec<Option<SubGraphId>>,
    /// Branch info for Gate nodes (indexed by NodeId; None for non-Gate nodes).
    pub gate_branches: Vec<Option<GateBranches>>,
    /// Field access info (indexed by NodeId; stores field_idx).
    pub field_access_infos: Vec<Option<u16>>,
    /// Record construction info (indexed by NodeId).
    pub record_lit_infos: Vec<Option<RecordLitInfo>>,
    /// @extern("C") function name for FFI call nodes (used for compute_ffi_call dispatch).
    pub ffi_call_names: Vec<Option<String>>,
    /// stdlib @extern("C") #{ }# inline FFI call info (used by compute_dyn_ffi_call dispatch).
    pub dyn_ffi_infos: Vec<Option<DynFfiInfo>>,
    /// Field assignment info (indexed by NodeId; stores field name; used by compute_record_field_set).
    pub field_set_names: Vec<Option<String>>,
    /// Method idx for vtable dynamic dispatch Call nodes (indexed by NodeId; None = static call).
    /// method_idx = position of the method in TraitDefInfo.methods (consistent with TraitValue.method_values index).
    pub vtable_call_methods: Vec<Option<u16>>,
    /// EventSource declaration node for Await nodes (indexed by NodeId; None for non-Await nodes).
    pub await_event_sources: Vec<Option<NodeId>>,
    /// Closure construction node info (indexed by NodeId; None for non-closure-construction nodes).
    pub closure_infos: Vec<Option<ClosureInfo>>,
    /// Partial application construction node info (indexed by NodeId; None for non-partial_construct nodes).
    pub partial_infos: Vec<Option<PartialInfo>>,
    /// Argument count for closure call nodes (excluding closure value and effect; used for chained partial application detection).
    pub closure_call_arg_counts: Vec<Option<u8>>,
    /// select expression branch info (indexed by NodeId; None for non-select-gate nodes).
    pub select_infos: Vec<Option<SelectInfo>>,
    /// Target outer NodeId for WriteBack nodes (indexed by NodeId; None for non-WriteBack nodes).
    pub writeback_targets: Vec<Option<NodeId>>,
    /// Tail-call flag for Call nodes (indexed by NodeId; true = tail-call frame reuse).
    pub tail_call_flags: Vec<bool>,
    /// Safe-operation flag (indexed by NodeId; true = short-circuit return Null when inputs[0] is Null).
    /// Used for ?.field / ?.method() / cast(x).to(T)?
    pub safe_op_flags: Vec<bool>,
    /// Flag for nodes produced by hoisting/unrolling/inlining (indexed by NodeId).
    /// true = node appended by a pass layer; must be included in the owning function subgraph's frame initialization.
    pub hoisted_node: Vec<bool>,
    /// Owning function subgraph for hoisted nodes (indexed by NodeId; valid only when hoisted_node=true).
    /// During rebuild, when regrouping by function-level subgraphs, hoisted nodes are placed within the owner subgraph's range.
    pub hoisted_owners: Vec<SubGraphId>,
    /// Compile-time SIMD/parallel batching marker (indexed by NodeId; None = not batchable).
    pub batch_infos: Vec<Option<BatchInfo>>,
    /// IR compile-time errors (unimplemented features, missing functions, etc.); moved from IrBuilder.errors at the end of build().
    pub ir_errors: Vec<String>,
    /// inline_trait construction node info (indexed by NodeId; None for non-trait-construct nodes).
    pub trait_construct_infos: Vec<Option<TraitConstructInfo>>,
    /// lazy construction node info (indexed by NodeId; None for non-lazy-construct nodes).
    pub lazy_construct_infos: Vec<Option<LazyConstructInfo>>,
    /// Record extension node info (indexed by NodeId; None for non-record-extend nodes).
    pub record_extend_infos: Vec<Option<RecordExtendInfo>>,
    /// Inclusive flag for slice nodes (indexed by NodeId; true = `[start..=end]`, false = `[start..end]`).
    pub slice_inclusive: Vec<bool>,
    /// Global variable runtime storage (top-level var/val declarations; shared across functions, does not depend on frame chain).
    pub global_var_storage: Arc<Vec<std::sync::Mutex<Option<crate::value::Value>>>>,
    /// Slot index for global_load nodes (indexed by NodeId; None for non-global_load nodes).
    pub global_load_slots: Vec<Option<u32>>,
    /// Slot index for global_store nodes (indexed by NodeId; None for non-global_store nodes).
    pub global_store_slots: Vec<Option<u32>>,
    /// Pattern matching: constructor name stored by constructor-name discrimination nodes (indexed by NodeId).
    pub pattern_ctor_names: Vec<Option<String>>,
    /// Pattern matching: type name of the constructor's owning type (indexed by NodeId).
    /// Used together with `pattern_ctor_names` to disambiguate same-named constructors
    /// across different types (e.g. `FileKind.File` vs `File`).
    pub pattern_type_names: Vec<Option<String>>,
    /// Pattern matching: field index for ADT positional field extraction nodes (indexed by NodeId).
    pub pattern_field_indices: Vec<Option<u16>>,
    /// Target type name for general cast nodes (indexed by NodeId; None for non-cast nodes).
    pub cast_target_types: Vec<Option<String>>,
    /// Cache metadata for memo_check / memo_store nodes (indexed by NodeId; None = non-memo node).
    pub memo_infos: Vec<Option<MemoInfo>>,
    /// Memoization cache table runtime storage (one HashMap<u64, Value> per memoized function).
    pub memo_tables: Arc<Vec<std::sync::Mutex<rustc_hash::FxHashMap<u64, Value>>>>,
    /// Debug-only sg → qualified function name (e.g. "std.io.File.remove"),
    /// parallel to `subgraphs`. Filled by the builder from its registration
    /// table, remapped by `rebuild`'s sg compaction, NEVER serialized (the
    /// .kzo artifact carries no name table — loads leave this empty).
    /// Consumed by the execution-coverage instrumentation
    /// (`KUZO_EXEC_COVERAGE=1`): the "never-executed std path" detector.
    pub sg_debug_names: Vec<Option<Box<str>>>,
    /// Vtable fallback dispatch: (vtable_method_idx, type_name) → SubGraphId.
    /// When a vtable call receives a concrete record (not a TraitVal), the runtime looks up
    /// the method subgraph by the value's type_name here, enabling static dispatch on the
    /// concrete type without boxing into a TraitVal.
    pub vtable_fallback_dispatch: rustc_hash::FxHashMap<(u16, Box<str>), SubGraphId>,
    /// String pool: ConstValue::Str { offset, len } references this pool.
    /// Maintained by IrBuilder as intern during build; filled from the .kzo StringPool section during load.
    pub string_pool: Arc<[u8]>,
    /// GraphMemory (load path): binary backing of mmap or owned bytes.
    /// Build path is None (directly accesses owned Vec fields);
    /// Load path is Some(GraphMemory); zerocopy tables are read from this backing via accessor methods.
    pub mem: Option<crate::solidify::Spec::GraphMemory>,
    /// CSR offset table for SubGraph upvalue_outer_nodes (load path).
    /// Each element = (byte_offset_into_SgUpvalueNodes, count).
    /// Build path is empty (accessor falls back to SubGraph.upvalue_outer_nodes Vec).
    /// Load path (zerocopy) is filled; SubGraph.upvalue_outer_nodes Vec is set to empty.
    pub sg_uv_offsets: Vec<(u32, u32)>,
    /// Per-Node byte offset tables for the 5 complex variable-length tables (load path).
    /// u32::MAX = None (no data for this node in that table); other values = byte offset within the section.
    /// Build path is empty (accessor falls back to owned Vec fields).
    /// Load path (zerocopy) is filled; owned Vec fields are set to empty.
    pub gate_branch_offsets: Vec<u32>,
    pub record_lit_info_offsets: Vec<u32>,
    pub select_info_offsets: Vec<u32>,
    pub trait_construct_info_offsets: Vec<u32>,
    pub record_extend_info_offsets: Vec<u32>,
    /// Per-node inputs-pool offsets, materialized at load when the `.kzo` v2
    /// Nodes section elides them (inputs pool contiguous in node-id order).
    /// Build path / non-elided files: empty.
    pub node_input_offsets: Vec<u32>,
    /// Flat CSR downstream table, derived at load from inputs + gate condition
    /// edges (the `.kzo` v2 format no longer serializes Downstreams).
    /// `downstream_csr_offsets` has node_count+1 entries; slices of
    /// `downstream_csr_flat` are returned by `downstream_slice` on loaded graphs.
    /// Build path: empty (owned `downstreams` Vec is authoritative).
    pub downstream_csr_offsets: Vec<u32>,
    pub downstream_csr_flat: Vec<NodeId>,
}

impl Clone for DataFlowGraph {
    /// Deep clone with runtime-shared storage (global vars / memo tables stay
    /// shared via their Arcs). Used by the optimizer's per-round snapshot for
    /// the never-corrupt fallback policy: any optimization-round failure
    /// restores the snapshot instead of propagating a panic.
    ///
    /// Only build-path graphs (`mem = None`) are cloned; mmap-backed loaded
    /// graphs are never snapshotted (optimization never runs on them).
    fn clone(&self) -> Self {
        debug_assert!(
            self.mem.is_none(),
            "cloning a mmap-backed (loaded) graph is not supported"
        );
        Self {
            nodes: self.nodes.clone(),
            inputs_pool: self.inputs_pool.clone(),
            subgraphs: self.subgraphs.clone(),
            entry_subgraph: self.entry_subgraph,
            compute_fns: self.compute_fns.clone(),
            downstreams: self.downstreams.clone(),
            const_values: self.const_values.clone(),
            const_cache: self.const_cache.clone(),
            sg_initial_pending: self.sg_initial_pending.clone(),
            sg_initial_seed: self.sg_initial_seed.clone(),
            downstream_counts: self.downstream_counts.clone(),
            linear_plans: self.linear_plans.clone(),
            call_targets: self.call_targets.clone(),
            gate_branches: self.gate_branches.clone(),
            field_access_infos: self.field_access_infos.clone(),
            record_lit_infos: self.record_lit_infos.clone(),
            ffi_call_names: self.ffi_call_names.clone(),
            dyn_ffi_infos: self.dyn_ffi_infos.clone(),
            field_set_names: self.field_set_names.clone(),
            vtable_call_methods: self.vtable_call_methods.clone(),
            await_event_sources: self.await_event_sources.clone(),
            closure_infos: self.closure_infos.clone(),
            partial_infos: self.partial_infos.clone(),
            closure_call_arg_counts: self.closure_call_arg_counts.clone(),
            select_infos: self.select_infos.clone(),
            writeback_targets: self.writeback_targets.clone(),
            tail_call_flags: self.tail_call_flags.clone(),
            safe_op_flags: self.safe_op_flags.clone(),
            hoisted_node: self.hoisted_node.clone(),
            hoisted_owners: self.hoisted_owners.clone(),
            batch_infos: self.batch_infos.clone(),
            ir_errors: self.ir_errors.clone(),
            trait_construct_infos: self.trait_construct_infos.clone(),
            lazy_construct_infos: self.lazy_construct_infos.clone(),
            record_extend_infos: self.record_extend_infos.clone(),
            slice_inclusive: self.slice_inclusive.clone(),
            global_var_storage: self.global_var_storage.clone(),
            global_load_slots: self.global_load_slots.clone(),
            global_store_slots: self.global_store_slots.clone(),
            pattern_ctor_names: self.pattern_ctor_names.clone(),
            pattern_type_names: self.pattern_type_names.clone(),
            pattern_field_indices: self.pattern_field_indices.clone(),
            cast_target_types: self.cast_target_types.clone(),
            memo_infos: self.memo_infos.clone(),
            memo_tables: self.memo_tables.clone(),
            vtable_fallback_dispatch: self.vtable_fallback_dispatch.clone(),
            sg_debug_names: self.sg_debug_names.clone(),
            string_pool: self.string_pool.clone(),
            mem: None,
            sg_uv_offsets: self.sg_uv_offsets.clone(),
            gate_branch_offsets: self.gate_branch_offsets.clone(),
            record_lit_info_offsets: self.record_lit_info_offsets.clone(),
            select_info_offsets: self.select_info_offsets.clone(),
            trait_construct_info_offsets: self.trait_construct_info_offsets.clone(),
            record_extend_info_offsets: self.record_extend_info_offsets.clone(),
            node_input_offsets: self.node_input_offsets.clone(),
            downstream_csr_offsets: self.downstream_csr_offsets.clone(),
            downstream_csr_flat: self.downstream_csr_flat.clone(),
        }
    }
}

impl DataFlowGraph {
    /// E5: linearized execution plan for subgraph `sg_idx` (None = not linearizable or the
    /// engine has not materialized plans). Global NodeIds in topological order.
    pub fn linear_plan(&self, sg_idx: usize) -> Option<&[NodeId]> {
        self.linear_plans.get(sg_idx).and_then(|p| p.as_deref())
    }

    /// Creates an empty graph.
    pub fn new() -> Self {
        Self {
            const_cache: Vec::new(),
            sg_initial_pending: Vec::new(),
            sg_initial_seed: Vec::new(),
            downstream_counts: Vec::new(),
            linear_plans: Vec::new(),
            nodes: Vec::new(),
            inputs_pool: InputsPool::new(),
            subgraphs: Vec::new(),
            entry_subgraph: None,
            compute_fns: build_compute_fn_table(),
            downstreams: Vec::new(),
            const_values: Vec::new(),
            // Metadata field initialization (Rust does not allow macro expansion inside struct initializers, so hand-written)
            call_targets: Vec::new(),
            gate_branches: Vec::new(),
            field_access_infos: Vec::new(),
            record_lit_infos: Vec::new(),
            ffi_call_names: Vec::new(),
            dyn_ffi_infos: Vec::new(),
            field_set_names: Vec::new(),
            vtable_call_methods: Vec::new(),
            await_event_sources: Vec::new(),
            closure_infos: Vec::new(),
            partial_infos: Vec::new(),
            closure_call_arg_counts: Vec::new(),
            select_infos: Vec::new(),
            writeback_targets: Vec::new(),
            tail_call_flags: Vec::new(),
            safe_op_flags: Vec::new(),
            hoisted_node: Vec::new(),
            hoisted_owners: Vec::new(),
            batch_infos: Vec::new(),
            trait_construct_infos: Vec::new(),
            lazy_construct_infos: Vec::new(),
            record_extend_infos: Vec::new(),
            slice_inclusive: Vec::new(),
            global_load_slots: Vec::new(),
            global_store_slots: Vec::new(),
            pattern_ctor_names: Vec::new(),
            pattern_type_names: Vec::new(),
            pattern_field_indices: Vec::new(),
            cast_target_types: Vec::new(),
            ir_errors: Vec::new(),
            global_var_storage: Arc::new(Vec::new()),
            memo_infos: Vec::new(),
            memo_tables: Arc::new(Vec::new()),
            vtable_fallback_dispatch: rustc_hash::FxHashMap::default(),
            sg_debug_names: Vec::new(),
            string_pool: Arc::from(Vec::new()),
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
        }
    }

    /// Adds a node, returns its NodeId.
    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        self.downstreams.push(Vec::new());
        self.const_values.push(None);
        // Metadata field push (auto-generated by node_metadata! macro)
        node_metadata!(metadata_push, self);
        self.hoisted_owners.push(SubGraphId(u32::MAX));
        id
    }

    // ---- Node metadata setters (auto-generated by node_metadata! macro) ----
    node_metadata!(metadata_setters);

    /// Clones all metadata from the source node to the target node (used for pass-layer node cloning).
    pub fn clone_node_metadata(&mut self, src_idx: usize, dst_idx: usize) {
        self.const_values[dst_idx] = self.const_values[src_idx].clone();
        node_metadata!(metadata_clone, self, src_idx, dst_idx);
        self.hoisted_owners[dst_idx] = self.hoisted_owners[src_idx];
    }

    /// Directly adds a node (without going through Builder), used for pass-layer transformations.
    /// Auto-syncs metadata push (same as add_node).
    pub fn add_node_raw(
        &mut self,
        kind: NodeKind,
        inputs: &[NodeId],
        compute_fn: ComputeFnId,
    ) -> NodeId {
        let inputs_offset = self.inputs_pool.push(inputs);
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            kind,
            input_count: inputs.len() as u8,
            inputs_offset,
            compute_fn,
        });
        self.downstreams.push(Vec::new());
        self.const_values.push(None);
        node_metadata!(metadata_push, self);
        self.hoisted_owners.push(SubGraphId(u32::MAX));
        id
    }

    /// Finds the function subgraph containing the given node (outermost; loop_kind=None and loop_parent_sg=None).
    /// Used by pass layers to determine which subgraph newly appended nodes should belong to.
    pub fn find_function_sg_for_node(&self, node: NodeId) -> Option<SubGraphId> {
        let mut best_sg: Option<SubGraphId> = None;
        let mut best_range_size = u32::MAX;
        for (idx, sg) in self.subgraphs.iter().enumerate() {
            let (start, end) = sg.node_range;
            if node.0 >= start.0 && node.0 < end.0 {
                let size = end.0 - start.0;
                if size < best_range_size {
                    best_range_size = size;
                    best_sg = Some(SubGraphId(idx as u32));
                }
            }
        }
        // Walk up from the innermost subgraph to find the function subgraph
        if let Some(inner_sg_id) = best_sg {
            let mut cur = inner_sg_id;
            loop {
                let sg = &self.subgraphs[cur.0 as usize];
                if sg.loop_kind == LoopKind::None && sg.loop_parent_sg.is_none() {
                    return Some(cur);
                }
                // Find the outer subgraph containing cur
                let (cs, ce) = sg.node_range;
                let mut parent: Option<SubGraphId> = None;
                let mut parent_size = u32::MAX;
                for (idx, psg) in self.subgraphs.iter().enumerate() {
                    if idx == cur.0 as usize {
                        continue;
                    }
                    let (ps, pe) = psg.node_range;
                    if cs.0 >= ps.0 && ce.0 <= pe.0 {
                        let size = pe.0 - ps.0;
                        if size < parent_size {
                            parent_size = size;
                            parent = Some(SubGraphId(idx as u32));
                        }
                    }
                }
                match parent {
                    Some(p) => cur = p,
                    None => return Some(cur), // Already at the outermost level
                }
            }
        }
        None
    }

    /// Finds the innermost subgraph containing the given node.
    /// Used by pass layers to determine whether a node is directly in a function-level subgraph
    /// (rather than nested within a Gate branch or loop body).
    pub fn find_innermost_sg_for_node(&self, node: NodeId) -> Option<SubGraphId> {
        let mut best_sg: Option<SubGraphId> = None;
        let mut best_range_size = u32::MAX;
        for (idx, sg) in self.subgraphs.iter().enumerate() {
            let (start, end) = sg.node_range;
            if node.0 >= start.0 && node.0 < end.0 {
                let size = end.0 - start.0;
                if size < best_range_size {
                    best_range_size = size;
                    best_sg = Some(SubGraphId(idx as u32));
                }
            }
        }
        best_sg
    }

    /// Finds the smallest enclosing subgraph (immediate parent) of the given subgraph.
    /// Used by LICM: invariants should be hoisted to the loop_sg's immediate parent,
    /// not always to the function sg (for nested loops, the outer loop's body_sg is the correct target).
    pub fn find_immediate_parent_sg(&self, sg_id: SubGraphId) -> Option<SubGraphId> {
        let (cs, ce) = self.subgraphs[sg_id.0 as usize].node_range;
        let mut best: Option<SubGraphId> = None;
        let mut best_size = u32::MAX;
        for (idx, psg) in self.subgraphs.iter().enumerate() {
            if idx == sg_id.0 as usize {
                continue;
            }
            let (ps, pe) = psg.node_range;
            // Must strictly contain the sg's range
            if ps.0 <= cs.0 && pe.0 >= ce.0 && (ps.0 < cs.0 || pe.0 > ce.0) {
                let size = pe.0 - ps.0;
                if size < best_size {
                    best_size = size;
                    best = Some(SubGraphId(idx as u32));
                }
            }
        }
        best
    }

    /// Extends the subgraph's `node_range` to cover newly appended nodes, and
    /// recursively extends all ancestor subgraphs. Called after the pass layer
    /// appends nodes so that `rebuild` sees them within the subgraph range,
    /// while keeping the nesting structure consistent (ancestor ranges must
    /// contain descendant ranges).
    pub fn extend_function_sg_range(&mut self, sg_id: SubGraphId, new_node_end: NodeId) {
        // Extend the target subgraph.
        let sg = &mut self.subgraphs[sg_id.0 as usize];
        if sg.node_range.1 < new_node_end {
            sg.node_range.1 = new_node_end;
        }
        // Recursively extend every ancestor that contains the target subgraph.
        let (cs, ce) = self.subgraphs[sg_id.0 as usize].node_range;
        for (idx, psg) in self.subgraphs.iter_mut().enumerate() {
            if idx == sg_id.0 as usize {
                continue;
            }
            let (ps, pe) = psg.node_range;
            // If the ancestor contains the target subgraph's range and the new
            // node exceeds the ancestor range, extend the ancestor.
            if ps.0 <= cs.0 && pe.0 >= ce.0 && pe.0 < new_node_end.0 {
                psg.node_range.1 = new_node_end;
            }
        }
    }

    /// Adds a subgraph and returns its `SubGraphId`.
    pub fn add_subgraph(&mut self, sg: SubGraph) -> SubGraphId {
        let id = SubGraphId(self.subgraphs.len() as u32);
        self.subgraphs.push(sg);
        id
    }

    /// Sets the program entry subgraph.
    pub fn set_entry_subgraph(&mut self, id: SubGraphId) {
        self.entry_subgraph = Some(id);
    }

    /// Computes the metadata hash of a node, used as the CSE deduplication key.
    ///
    /// Generic method: hashes all per-node metadata fields into a `u64`.
    /// Two nodes with the same `(compute_fn, inputs)` but different metadata
    /// will not be merged by CSE. For example, `pattern_adt_field_get` nodes
    /// with different `field_index`, or `pattern_ctor_match` nodes with
    /// different `ctor_name`. Types that do not implement `Hash` are hashed
    /// via their `Debug` string.
    pub fn cse_metadata_hash(&self, idx: usize) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        let mut h = DefaultHasher::new();

        macro_rules! hash_opt {
            ($field:ident) => {
                match self.$field.get(idx).and_then(|o| o.as_ref()) {
                    None => 0u8.hash(&mut h),
                    Some(v) => { 1u8.hash(&mut h); format!("{:?}", v).hash(&mut h); }
                }
            };
        }

        hash_opt!(call_targets);
        hash_opt!(gate_branches);
        hash_opt!(field_access_infos);
        hash_opt!(record_lit_infos);
        hash_opt!(ffi_call_names);
        hash_opt!(dyn_ffi_infos);
        hash_opt!(field_set_names);
        hash_opt!(vtable_call_methods);
        hash_opt!(await_event_sources);
        hash_opt!(closure_infos);
        hash_opt!(partial_infos);
        hash_opt!(closure_call_arg_counts);
        hash_opt!(select_infos);
        hash_opt!(writeback_targets);
        hash_opt!(batch_infos);
        hash_opt!(trait_construct_infos);
        hash_opt!(lazy_construct_infos);
        hash_opt!(record_extend_infos);
        hash_opt!(global_load_slots);
        hash_opt!(global_store_slots);
        hash_opt!(pattern_ctor_names);
        hash_opt!(pattern_type_names);
        hash_opt!(pattern_field_indices);
        hash_opt!(cast_target_types);

        self.tail_call_flags.get(idx).copied().unwrap_or(false).hash(&mut h);
        self.safe_op_flags.get(idx).copied().unwrap_or(false).hash(&mut h);
        self.slice_inclusive.get(idx).copied().unwrap_or(false).hash(&mut h);

        h.finish()
    }

    /// Computes the downstream list (fan-out statistics) for all nodes.
    ///
    /// Iterates over each node's inputs and registers that node in the
    /// `downstreams` list of each of its input nodes. Used for slot-level RC:
    /// on node produce, `refcount = downstreams[n].len()`.
    pub fn compute_downstreams(&mut self) {
        // (Re)allocate to match the node count, then clear (callers that run
        // this on every iteration keep their buffers).
        if self.downstreams.len() != self.nodes.len() {
            self.downstreams = vec![Vec::new(); self.nodes.len()];
        } else {
            for ds in &mut self.downstreams {
                ds.clear();
            }
        }
        // Iterate over nodes and register downstream edges (input edges in inputs_pool).
        for nid in 0..self.nodes.len() {
            let node = self.nodes[nid];
            let inputs = self.inputs_pool.get(node.inputs_offset, node.input_count);
            for &input in inputs {
                self.downstreams[input.0 as usize].push(NodeId(nid as u32));
            }
        }
        // Gate node's condition_input → Gate edge (Gate readiness depends on the
        // condition value being computed). Avoid duplicates: if condition_input
        // is already in the Gate's inputs (input_count=1 for if/while/for/match
        // Gates), the first loop already registered the downstream edge, so skip.
        // Only register for Gates whose condition_input is not in inputs (e.g. select gate).
        for nid in 0..self.nodes.len() {
            if let Some(gb) = &self.gate_branches[nid] {
                let node = self.nodes[nid];
                let inputs = self.inputs_pool.get(node.inputs_offset, node.input_count);
                if !inputs.contains(&gb.condition_input) {
                    self.downstreams[gb.condition_input.0 as usize].push(NodeId(nid as u32));
                }
            }
        }
    }

    /// Computes the flat CSR downstream table for LOADED graphs (`.kzo` v2 no
    /// longer serializes Downstreams). Works through the mem-agnostic
    /// accessors so both the mmap and owned backends derive identically.
    /// Mirrors `compute_downstreams` + `rebuild` step 4: input edges plus the
    /// Gate condition_input edge when it is not already a regular input.
    /// Must run AFTER gate branches are materialized.
    pub fn compute_downstream_csr(&mut self) {
        let n = self.node_count();
        // Pass 1: degree per producer.
        let mut degrees = vec![0u32; n];
        for nid in 0..n {
            let node = self.node(nid);
            let inputs = self.inputs(node.inputs_offset, node.input_count);
            for &input in inputs {
                degrees[input.0 as usize] += 1;
            }
            if let Some(gb) = self.gate_branches.get(nid).and_then(|g| g.as_ref()) {
                if !inputs.contains(&gb.condition_input) {
                    degrees[gb.condition_input.0 as usize] += 1;
                }
            }
        }
        // Prefix sums → offsets.
        let mut offsets = Vec::with_capacity(n + 1);
        let mut acc = 0u32;
        offsets.push(0u32);
        for &d in &degrees {
            acc += d;
            offsets.push(acc);
        }
        // Pass 2: scatter with a moving cursor per producer.
        let mut flat: Vec<NodeId> = vec![NodeId(0); acc as usize];
        let mut cursor = offsets.clone();
        for nid in 0..n {
            let node = self.node(nid);
            let inputs = self.inputs(node.inputs_offset, node.input_count);
            for &input in inputs {
                let p = input.0 as usize;
                flat[cursor[p] as usize] = NodeId(nid as u32);
                cursor[p] += 1;
            }
            if let Some(gb) = self.gate_branches.get(nid).and_then(|g| g.as_ref()) {
                if !inputs.contains(&gb.condition_input) {
                    let p = gb.condition_input.0 as usize;
                    flat[cursor[p] as usize] = NodeId(nid as u32);
                    cursor[p] += 1;
                }
            }
        }
        self.downstream_csr_offsets = offsets;
        self.downstream_csr_flat = flat;
    }

    // ------------------------------------------------------------------
    // NodeRef door — the single enumeration of out-of-band NodeId refs
    // ------------------------------------------------------------------
    // Every NodeId the graph stores OUTSIDE `inputs_pool` is enumerated
    // through this door ONLY. Three consumers must see the identical list:
    // liveness seeding (Optimizer `compute_liveness`), preserve propagation
    // (`collect_refs`) and remapping (`rebuild` via `map_node_refs`). Before
    // the door existed each consumer hand-rolled its own list and they had
    // already diverged (`upvalue_outer_nodes` / `reset_plan` were remapped by
    // rebuild but never seeded as live → `remap_n` panics). Known non-door
    // consumer: `pass_dse`'s read/write ref sets intentionally enumerate a
    // subset (store analysis, not liveness).
    //
    // Adding a new metadata field that stores NodeIds: add it to
    // `each_node_ref` + `each_sg_anchor_ref` below (and only there).

    /// Per-node metadata NodeId refs of `idx`: await source, writeback target,
    /// gate condition + branch params, select event sources. Inputs are NOT
    /// included — they are structural edges every traversal already walks.
    /// Load-path safe (the complex tables are materialized at load).
    pub fn node_meta_refs(&self, idx: usize, out: &mut Vec<NodeId>) {
        self.each_node_ref(idx, |id| out.push(id));
    }

    #[inline]
    fn each_node_ref(&self, idx: usize, mut f: impl FnMut(NodeId)) {
        if let Some(src) = self.await_event_sources.get(idx).and_then(|o| *o) {
            f(src);
        }
        if let Some(t) = self.writeback_targets.get(idx).and_then(|o| *o) {
            f(t);
        }
        if let Some(gb) = self.gate_branches.get(idx).and_then(|o| o.as_ref()) {
            f(gb.condition_input);
            for (_, _, params) in &gb.branches {
                for &p in params {
                    f(p);
                }
            }
        }
        if let Some(si) = self.select_infos.get(idx).and_then(|o| o.as_ref()) {
            for sb in &si.branches {
                f(sb.event_source_node);
            }
        }
    }

    /// Anchor NodeId refs of subgraph `si`: everything that must stay live for
    /// the sg to remain structurally valid (anchors, defer registration,
    /// event declarations, upvalues, loop reset plan). Load-path safe via
    /// `sg_upvalue_outer_nodes`.
    pub fn sg_anchor_refs(&self, si: usize, out: &mut Vec<NodeId>) {
        self.each_sg_anchor_ref(si, |id| out.push(id));
    }

    #[inline]
    fn each_sg_anchor_ref(&self, si: usize, mut f: impl FnMut(NodeId)) {
        let sg = &self.subgraphs[si];
        f(sg.entry_node);
        f(sg.return_node);
        if let Some(c) = sg.cond_node {
            f(c);
        }
        if let Some(n) = sg.iter_next_node {
            f(n);
        }
        for decl in &sg.event_source_decls {
            f(decl.node);
        }
        for e in &sg.defer_table {
            f(e.trigger_node);
            for &c in &e.captured_inputs {
                f(c);
            }
        }
        for &u in self.sg_upvalue_outer_nodes(si) {
            f(u);
        }
        if let Some(plan) = &sg.reset_plan {
            for &n in &plan.reset_to_zero {
                f(n);
            }
            for &n in &plan.reset_to_one {
                f(n);
            }
            for &n in &plan.reset_condition_tree {
                f(n);
            }
            // plan.condition_tree_plan is deliberately NOT enumerated: rebuild
            // clears it (W5) and the pipeline recomputes it post-compaction.
        }
    }

    /// Read door over EVERY metadata NodeId in the graph (per-node tables +
    /// per-sg anchors). Used by liveness seeding and the Verifier.
    pub fn for_each_node_ref(&self, mut f: impl FnMut(NodeRefSite, u32, NodeId)) {
        for idx in 0..self.nodes.len() {
            self.each_node_ref(idx, |id| f(NodeRefSite::NodeMeta, idx as u32, id));
        }
        for si in 0..self.subgraphs.len() {
            self.each_sg_anchor_ref(si, |id| f(NodeRefSite::SgAnchor, si as u32, id));
        }
    }

    /// Write door: remaps EVERY metadata NodeId in place. Build path only —
    /// load-path graphs are never rebuilt (upvalues live in the CSR section).
    /// Replaces rebuild's scattered per-field remap blocks so the write list
    /// can never diverge from the read list above.
    pub fn map_node_refs(&mut self, mut f: impl FnMut(NodeRefSite, u32, NodeId) -> NodeId) {
        debug_assert!(self.mem.is_none() && self.sg_uv_offsets.is_empty());
        for idx in 0..self.nodes.len() {
            if let Some(src) = self.await_event_sources.get(idx).and_then(|o| *o) {
                self.await_event_sources[idx] =
                    Some(f(NodeRefSite::NodeMeta, idx as u32, src));
            }
            if let Some(t) = self.writeback_targets.get(idx).and_then(|o| *o) {
                self.writeback_targets[idx] =
                    Some(f(NodeRefSite::NodeMeta, idx as u32, t));
            }
            if let Some(gb) = self.gate_branches.get(idx).and_then(|o| o.as_ref()) {
                let new_cond = f(NodeRefSite::NodeMeta, idx as u32, gb.condition_input);
                let mut branches = gb.branches.clone();
                for (_, _, params) in &mut branches {
                    for p in params.iter_mut() {
                        *p = f(NodeRefSite::NodeMeta, idx as u32, *p);
                    }
                }
                let new_gb = GateBranches {
                    condition_input: new_cond,
                    branches,
                    capture: gb.capture,
                };
                self.gate_branches[idx] = Some(new_gb);
            }
            if let Some(si) = self.select_infos.get(idx).and_then(|o| o.as_ref()) {
                let mut branches = si.branches.clone();
                for sb in &mut branches {
                    sb.event_source_node =
                        f(NodeRefSite::NodeMeta, idx as u32, sb.event_source_node);
                }
                self.select_infos[idx] = Some(SelectInfo { branches });
            }
        }
        for si in 0..self.subgraphs.len() {
            let sg = &mut self.subgraphs[si];
            sg.entry_node = f(NodeRefSite::SgAnchor, si as u32, sg.entry_node);
            sg.return_node = f(NodeRefSite::SgAnchor, si as u32, sg.return_node);
            if let Some(c) = sg.cond_node {
                sg.cond_node = Some(f(NodeRefSite::SgAnchor, si as u32, c));
            }
            if let Some(n) = sg.iter_next_node {
                sg.iter_next_node = Some(f(NodeRefSite::SgAnchor, si as u32, n));
            }
            for decl in &mut sg.event_source_decls {
                decl.node = f(NodeRefSite::SgAnchor, si as u32, decl.node);
            }
            for e in &mut sg.defer_table {
                e.trigger_node = f(NodeRefSite::SgAnchor, si as u32, e.trigger_node);
                for c in e.captured_inputs.iter_mut() {
                    *c = f(NodeRefSite::SgAnchor, si as u32, *c);
                }
            }
            for u in sg.upvalue_outer_nodes.iter_mut() {
                *u = f(NodeRefSite::SgAnchor, si as u32, *u);
            }
            if let Some(plan) = &mut sg.reset_plan {
                for n in plan.reset_to_zero.iter_mut() {
                    *n = f(NodeRefSite::SgAnchor, si as u32, *n);
                }
                for n in plan.reset_to_one.iter_mut() {
                    *n = f(NodeRefSite::SgAnchor, si as u32, *n);
                }
                for n in plan.reset_condition_tree.iter_mut() {
                    *n = f(NodeRefSite::SgAnchor, si as u32, *n);
                }
                // condition_tree_plan: cleared by rebuild (W5), never remapped.
            }
        }
    }

    /// Late compaction rebuild: rebuilds the graph from the `dead` set,
    /// `redirect` map and `dead_sgs` set. Compacts `nodes`/`inputs_pool`,
    /// remaps all `NodeId` references and per-NodeId metadata vectors, removes
    /// dead/placeholder subgraphs and remaps every `SubGraphId` reference, then
    /// recomputes `nested_ranges` from the final ranges. After rebuild, all
    /// `NodeId`/`SubGraphId` references are updated to the new contiguous
    /// numbering. Returns the `old_to_new` map (so callers can sync
    /// `expr_node_map`, etc.).
    pub fn rebuild(
        &mut self,
        dead: &rustc_hash::FxHashSet<NodeId>,
        redirect: &rustc_hash::FxHashMap<NodeId, NodeId>,
        dead_sgs: &rustc_hash::FxHashSet<SubGraphId>,
    ) -> Vec<Option<NodeId>> {
        // Test hook for the optimizer stability policy (run_guarded snapshot
        // rollback): simulates an invariant violation inside rebuild.
        if std::env::var("KUZO_TEST_INJECT_REBUILD_FAIL").is_ok() {
            panic!("rebuild: injected invariant failure (KUZO_TEST_INJECT_REBUILD_FAIL)");
        }
        // ── Recursively resolve redirects ──
        let resolve = |id: NodeId| -> NodeId {
            let mut cur = id;
            while let Some(&next) = redirect.get(&cur) { cur = next; }
            cur
        };



        // ── 1. Arrange live nodes grouped by function-level subgraph ──
        // Hoisted nodes appended by the pass layer (LICM/inline) live at the end
        // of graph.nodes, outside the caller subgraph's node_range. If we compact
        // in 0..total order, the hoisted nodes' new_id ends up at the tail,
        // outside the caller's node_range → the caller frame never executes them
        // → the transformation is void.
        //
        // Instead, arrange by function-level subgraph grouping: each function-level
        // subgraph's native live nodes + the hoisted live nodes that belong to it,
        // so hoisted nodes' new_id directly follows the caller's native nodes,
        // keeping them contiguous.
        let total = self.nodes.len();
        let mut old_to_new: Vec<Option<NodeId>> = vec![None; total];
        let mut new_to_old: Vec<usize> = Vec::with_capacity(total);
        let mut new_nodes: Vec<Node> = Vec::with_capacity(total);

        // 1a. Compute the owning function-level subgraph for each node (keep a
        // copy for step 5, since step 3b compaction would otherwise clobber it).
        // Only true function-level subgraphs (id == function_id) set node_owner.
        // Branch subgraphs (if/match arms) have loop_kind=None, loop_parent=None
        // but id != function_id — their nodes are owned by the parent function subgraph.
        let mut node_owner: Vec<u32> = vec![u32::MAX; total];
        for sg in &self.subgraphs {
            if sg.loop_kind != LoopKind::None || sg.loop_parent_sg.is_some() {
                continue;
            }
            // Skip branch subgraphs: their nodes are owned by the function-level subgraph
            // whose node_range encompasses them. This prevents hoisted nodes (whose
            // hoisted_owners points to the function-level subgraph) from being
            // incorrectly attributed to a branch subgraph, which would extend the
            // branch subgraph's new node_range past the Gate node and cause infinite recursion.
            if sg.id.0 != sg.function_id {
                continue;
            }
            let start = sg.node_range.0.0 as usize;
            let end = (sg.node_range.1.0 as usize).min(total);
            for idx in start..end {
                node_owner[idx] = sg.id.0;
            }
        }
        // Ownership of hoisted nodes (not inside any node_range; determined via hoisted_owners).
        // IMPORTANT: hoisted_owners may point to a branch subgraph (id != function_id).
        // Hoisted nodes must be attributed to the function-level subgraph, otherwise
        // the branch subgraph's node_range (recomputed in step 5) would extend to
        // cover the hoisted nodes' new positions, accidentally encompassing the Gate
        // node that sits between the branch's native nodes and the hoisted nodes,
        // causing infinite recursion at runtime (Gate launches a subgraph containing itself).
        for idx in 0..total {
            if self.hoisted_node[idx] && node_owner[idx] == u32::MAX {
                let raw_owner = self.hoisted_owners[idx].0 as usize;
                let func_owner = if raw_owner < self.subgraphs.len() {
                    self.subgraphs[raw_owner].function_id
                } else {
                    self.hoisted_owners[idx].0
                };
                node_owner[idx] = func_owner;
            }
        }
        // Save a copy of node_owner indexed by old indices (step 5 still needs
        // old-index access after step 3b compaction).
        let node_owner_old = node_owner.clone();

        // 1b. Collect the function-level subgraph list (sorted by node_range.0
        // to preserve original order). Only true function-level subgraphs (id == function_id).
        let mut func_sgs: Vec<u32> = self
            .subgraphs
            .iter()
            .filter(|sg| sg.loop_kind == LoopKind::None && sg.loop_parent_sg.is_none() && sg.id.0 == sg.function_id)
            .map(|sg| sg.id.0)
            .collect();
        func_sgs.sort_by_key(|&sg_id| self.subgraphs[sg_id as usize].node_range.0);

        // 1c. Assign new_id in function-level subgraph order.
        for &sg_id in &func_sgs {
            let sg = &self.subgraphs[sg_id as usize];
            let start = sg.node_range.0.0 as usize;
            let end = (sg.node_range.1.0 as usize).min(total);

            // Native live nodes (including nested subgraph nodes; skip hoisted).
            for old_idx in start..end {
                if self.hoisted_node[old_idx] {
                    continue;
                }
                let old_id = NodeId(old_idx as u32);
                if dead.contains(&old_id) || redirect.contains_key(&old_id) {
                    continue;
                }
                // De-duplicate: if the node was already assigned in another
                // subgraph iteration (overlapping node_range), skip.
                if old_to_new[old_idx].is_some() {
                    continue;
                }
                let new_id = NodeId(new_nodes.len() as u32);
                old_to_new[old_idx] = Some(new_id);
                new_to_old.push(old_idx);
                new_nodes.push(self.nodes[old_idx]);
            }

            // Hoisted live nodes (owner == sg_id). Use node_owner (resolved to
            // function-level subgraph) instead of raw hoisted_owners, so hoisted
            // nodes are placed after the function-level subgraph's native nodes.
            for old_idx in 0..total {
                if !self.hoisted_node[old_idx] {
                    continue;
                }
                if node_owner[old_idx] != sg_id {
                    continue;
                }
                let old_id = NodeId(old_idx as u32);
                if dead.contains(&old_id) || redirect.contains_key(&old_id) {
                    continue;
                }
                if old_to_new[old_idx].is_some() {
                    continue;
                }
                let new_id = NodeId(new_nodes.len() as u32);
                old_to_new[old_idx] = Some(new_id);
                new_to_old.push(old_idx);
                new_nodes.push(self.nodes[old_idx]);
            }
        }

        // 1d. Unowned live nodes (should not exist; placed last for safety).
        for old_idx in 0..total {
            if old_to_new[old_idx].is_some() {
                continue;
            }
            let old_id = NodeId(old_idx as u32);
            if dead.contains(&old_id) || redirect.contains_key(&old_id) {
                continue;
            }
            let new_id = NodeId(new_nodes.len() as u32);
            old_to_new[old_idx] = Some(new_id);
            new_to_old.push(old_idx);
            new_nodes.push(self.nodes[old_idx]);
        }

        // ── 2. Rebuild inputs_pool (resolve + remap) ──
        let mut new_inputs: Vec<NodeId> = Vec::new();
        for node in &mut new_nodes {
            let old_inputs = self.inputs_pool.get(node.inputs_offset, node.input_count);
            let new_offset = new_inputs.len() as u32;
            for &old_in in old_inputs {
                let resolved = resolve(old_in);
                let new_in = old_to_new[resolved.0 as usize]
                    .expect("rebuild: input node not live");
                new_inputs.push(new_in);
            }
            node.inputs_offset = new_offset;
        }

        self.nodes = new_nodes;
        self.inputs_pool.data = new_inputs;

        // ── 3. Compact per-NodeId metadata vectors ──
        // Collect metadata in new-index order using the new_to_old map.
        // NodeId VALUES are not remapped here — every metadata NodeId remap
        // happens in one place after step 6, through `map_node_refs` (the
        // write door), so this list can never diverge from liveness seeding.

        // 3a. Compact Vec<Option<T: Clone>> (no interior NodeId).
        macro_rules! compress_opt {
            ($field:ident) => {{
                let mut v: Vec<_> = Vec::with_capacity(new_to_old.len());
                for &old_idx in &new_to_old {
                    v.push(self.$field[old_idx].clone());
                }
                self.$field = v;
            }};
        }
        compress_opt!(const_values);
        compress_opt!(call_targets);
        compress_opt!(field_access_infos);
        compress_opt!(record_lit_infos);
        compress_opt!(ffi_call_names);
        compress_opt!(dyn_ffi_infos);
        compress_opt!(field_set_names);
        compress_opt!(vtable_call_methods);
        compress_opt!(closure_infos);
        compress_opt!(partial_infos);
        compress_opt!(closure_call_arg_counts);
        compress_opt!(batch_infos);
        compress_opt!(trait_construct_infos);
        compress_opt!(lazy_construct_infos);
        compress_opt!(record_extend_infos);
        compress_opt!(global_load_slots);
        compress_opt!(global_store_slots);
        compress_opt!(pattern_ctor_names);
        compress_opt!(pattern_type_names);
        compress_opt!(pattern_field_indices);
        compress_opt!(cast_target_types);
        compress_opt!(memo_infos);

        // 3b. Compact Vec<bool>.
        macro_rules! compress_bool {
            ($field:ident) => {{
                let mut v: Vec<_> = Vec::with_capacity(new_to_old.len());
                for &old_idx in &new_to_old {
                    v.push(self.$field[old_idx]);
                }
                self.$field = v;
            }};
        }
        compress_bool!(tail_call_flags);
        compress_bool!(safe_op_flags);
        compress_bool!(hoisted_node);
        compress_bool!(slice_inclusive);

        // 3b2. Compact hoisted_owners: Vec<SubGraphId>.
        {
            let mut v: Vec<SubGraphId> = Vec::with_capacity(new_to_old.len());
            for &old_idx in &new_to_old {
                v.push(self.hoisted_owners[old_idx]);
            }
            self.hoisted_owners = v;
        }

        // 3c. Compact vectors containing NodeId (plain clone — remap happens
        // once, after step 6, via `map_node_refs`).
        // await_event_sources / writeback_targets: Vec<Option<NodeId>>
        {
            let mut v: Vec<Option<NodeId>> = Vec::with_capacity(new_to_old.len());
            for &old_idx in &new_to_old {
                v.push(self.await_event_sources[old_idx]);
            }
            self.await_event_sources = v;
        }
        {
            let mut v: Vec<Option<NodeId>> = Vec::with_capacity(new_to_old.len());
            for &old_idx in &new_to_old {
                v.push(self.writeback_targets[old_idx]);
            }
            self.writeback_targets = v;
        }
        // gate_branches / select_infos: interior NodeIds remapped by the door.
        {
            let mut v: Vec<Option<GateBranches>> = Vec::with_capacity(new_to_old.len());
            for &old_idx in &new_to_old {
                v.push(self.gate_branches[old_idx].clone());
            }
            self.gate_branches = v;
        }
        {
            let mut v: Vec<Option<SelectInfo>> = Vec::with_capacity(new_to_old.len());
            for &old_idx in &new_to_old {
                v.push(self.select_infos[old_idx].clone());
            }
            self.select_infos = v;
        }

        // ── 4. (moved) Rebuild downstreams AFTER the NodeId door remap below —
        // gate condition_input must already be new-indexed when this runs.

        // ── 5. Remap NodeId references inside subgraphs ──
        // node_range is recomputed by scanning live nodes within the old range
        // + hoisted nodes that belong to the subgraph. Step 1 already arranged
        // nodes by function-level subgraph grouping, with hoisted nodes directly
        // following the native nodes, so the new node_range is naturally
        // contiguous (native live node new_id + hoisted live node new_id).
        //
        // 5-pre. Subgraph removal candidates: (a) subgraphs explicitly killed
        // by function-level DCE (pass_func_dce marks the function sg plus all
        // nested branch/loop/defer-body subgraphs); (b) empty-range subgraphs —
        // never-compiled placeholders (analyzer-dead branches), or subgraphs
        // fully emptied by earlier rounds (e.g. LoopUnroll).
        //
        // A candidate still referenced by a live structure is vetoed BEFORE
        // step 5, so vetoed subgraphs go through normal range-recalc/remap
        // (their anchors stay consistent instead of going stale). The veto is
        // iterative: keeping a subgraph keeps its function/loop-parent/defer
        // bodies alive too (cascade), so a kept subgraph never references a
        // removed one and step 6's renumbering is always total.
        let mut remove_sg: Vec<bool> = self
            .subgraphs
            .iter()
            .map(|sg| dead_sgs.contains(&sg.id) || sg.node_range.0 == sg.node_range.1)
            .collect();
        if remove_sg.iter().any(|&r| r) {
            let mut referenced: rustc_hash::FxHashSet<SubGraphId> = rustc_hash::FxHashSet::default();
            if let Some(e) = self.entry_subgraph {
                referenced.insert(e);
            }
            for sg in self.vtable_fallback_dispatch.values() {
                referenced.insert(*sg);
            }
            // Live nodes' cross-sg references. NOTE: this runs AFTER step 3's
            // metadata compaction, so these tables are already new-indexed and
            // contain ONLY live nodes' entries (dead/redirect-key nodes' rows
            // were dropped) — no liveness filtering needed here.
            for ct in self.call_targets.iter().flatten() {
                referenced.insert(*ct);
            }
            for ci in self.closure_infos.iter().flatten() {
                referenced.insert(ci.subgraph_id);
            }
            for pi in self.partial_infos.iter().flatten() {
                referenced.insert(pi.subgraph_id);
            }
            for li in self.lazy_construct_infos.iter().flatten() {
                referenced.insert(li.thunk_sg);
            }
            for ti in self.trait_construct_infos.iter().flatten() {
                for m in &ti.methods {
                    referenced.insert(m.subgraph_id);
                }
            }
            for gb in self.gate_branches.iter().flatten() {
                for (_, bsg, _) in &gb.branches {
                    referenced.insert(*bsg);
                }
            }
            for si in self.select_infos.iter().flatten() {
                for sb in &si.branches {
                    referenced.insert(sb.subgraph_id);
                }
            }
            // Kept (non-candidate) subgraphs' sg-internal links.
            for (i, sg) in self.subgraphs.iter().enumerate() {
                if remove_sg[i] {
                    continue;
                }
                referenced.insert(SubGraphId(sg.function_id));
                if let Some(p) = sg.loop_parent_sg {
                    referenced.insert(p);
                }
                for e in &sg.defer_table {
                    referenced.insert(e.body_subgraph);
                }
            }
            // Iterative veto with cascade.
            loop {
                let mut changed = false;
                for i in 0..self.subgraphs.len() {
                    if !remove_sg[i] {
                        continue;
                    }
                    let sg = &self.subgraphs[i];
                    if referenced.contains(&sg.id) {
                        remove_sg[i] = false;
                        changed = true;
                        // Tripwire: a NON-EMPTY subgraph needing this veto
                        // means the liveness closure (compute_liveness, which
                        // seeds callee anchors from live call/closure edges)
                        // missed a cross-sg edge kind — the function should
                        // have stayed reachable in the first place. Empty
                        // placeholders (analyzer-dead branches) are the only
                        // legitimate veto saves.
                        debug_assert!(
                            sg.node_range.0 == sg.node_range.1,
                            "rebuild: non-empty sg {} (fn={} range=[{},{}) survived only via \
                             the removal veto — liveness closure missed an edge to it",
                            sg.id.0, sg.function_id, sg.node_range.0.0, sg.node_range.1.0
                        );
                        referenced.insert(SubGraphId(sg.function_id));
                        if let Some(p) = sg.loop_parent_sg {
                            referenced.insert(p);
                        }
                        for e in &sg.defer_table {
                            referenced.insert(e.body_subgraph);
                        }
                        if std::env::var("KUZO_DEBUG_REBUILD").is_ok() {
                            eprintln!(
                                "[REBUILD] sg={} removal vetoed: still referenced (fn={} kind={:?} range=[{},{})",
                                sg.id.0, sg.function_id, sg.loop_kind,
                                sg.node_range.0.0, sg.node_range.1.0
                            );
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        let dbg_rebuild = std::env::var("KUZO_DEBUG_REBUILD").is_ok();
        // Save old node_ranges for debugging.
        let old_ranges: Vec<(u32, u32)> = if dbg_rebuild {
            self.subgraphs.iter().map(|sg| (sg.node_range.0.0, sg.node_range.1.0)).collect()
        } else {
            Vec::new()
        };
        for (sg_idx, sg) in self.subgraphs.iter_mut().enumerate() {
            let old_start = sg.node_range.0.0 as usize;
            let old_end = (sg.node_range.1.0 as usize).min(total);
            let sg_id = sg.id;

            // Removed subgraphs: dropped in step 6 — nothing to remap (their
            // anchor nodes are dead; remap_n would panic on them).
            if remove_sg[sg_idx] {
                continue;
            }
            // NOTE: kept subgraphs — including still-referenced EMPTY
            // placeholders (analyzer-dead branches that live gates select at
            // runtime) — go through the same anchor remapping as before:
            // their entry/return point at live nodes of sibling regions, and
            // the engine resolves the branch result through them.

            let mut new_start: Option<u32> = None;
            let mut new_end: u32 = 0;

            // Helper: update new_start/new_end.
            let mut update_range = |nid: NodeId| {
                if new_start.is_none() {
                    new_start = Some(nid.0);
                }
                if nid.0 + 1 > new_end {
                    new_end = nid.0 + 1;
                }
            };

            // Live nodes within the native range.
            for old_idx in old_start..old_end {
                let old_id = NodeId(old_idx as u32);
                if dead.contains(&old_id) || redirect.contains_key(&old_id) {
                    continue;
                }
                update_range(old_to_new[old_idx].unwrap());
            }

            // Hoisted live nodes belonging to this subgraph (use node_owner_old,
            // because self.hoisted_owners was already compacted to the new array
            // in step 3b2 and can no longer be accessed by old index).
            for old_idx in 0..total {
                if node_owner_old[old_idx] != sg_id.0 {
                    continue;
                }
                // Skip nodes within the native range (already handled above).
                if old_idx >= old_start && old_idx < old_end {
                    continue;
                }
                let old_id = NodeId(old_idx as u32);
                if dead.contains(&old_id) || redirect.contains_key(&old_id) {
                    continue;
                }
                update_range(old_to_new[old_idx].unwrap());
            }

            // DEBUG: if this subgraph is a Gate branch and the new range contains the Gate node, print details.
            if dbg_rebuild {
                let (ns, ne) = match new_start {
                    Some(ns) => (ns, new_end),
                    None => (0u32, 0u32),
                };
                // Check if any Gate node's new_id falls in [ns, ne) for this subgraph's branches
                let sg_idx = sg_id.0 as usize;
                if sg_idx < old_ranges.len() {
                    let (o_s, o_e) = old_ranges[sg_idx];
                    // gate_branches was already compacted in step 3c, so gate_idx IS the new_id.
                    // To get the old_idx, use new_to_old[gate_idx].
                    for (gate_new_id, gb_opt) in self.gate_branches.iter().enumerate() {
                        if let Some(gb) = gb_opt {
                            // Check if this subgraph is a branch of this gate
                            let is_branch = gb.branches.iter().any(|(_, bsg, _)| bsg.0 == sg_id.0);
                            if is_branch {
                                let gnid = gate_new_id as u32;
                                if gnid >= ns && gnid < ne {
                                    let gate_old_idx = if gate_new_id < new_to_old.len() { new_to_old[gate_new_id] } else { usize::MAX };
                                    eprintln!("[REBUILD-DETAIL] sg={} old_range=[{},{}) new_range=[{},{}) gate_node old_idx={} new_id={} function_id={} loop_kind={:?} loop_parent={:?}",
                                        sg_id.0, o_s, o_e, ns, ne, gate_old_idx, gnid,
                                        sg.function_id, sg.loop_kind, sg.loop_parent_sg);
                                    // Print the old_to_new mapping for nodes in [old_start, old_end)
                                    eprint!("[REBUILD-DETAIL]   native mapping:");
                                    for oi in old_start..old_end {
                                        if let Some(nid) = old_to_new[oi] {
                                            let st = if dead.contains(&NodeId(oi as u32)) { "dead" }
                                                     else if redirect.contains_key(&NodeId(oi as u32)) { "redirect" }
                                                     else { "live" };
                                            eprint!(" {}→{}({})", oi, nid.0, st);
                                        }
                                    }
                                    eprintln!();
                                    // Print hoisted nodes for this sg
                                    eprint!("[REBUILD-DETAIL]   hoisted (owner={}):", sg_id.0);
                                    for oi in 0..total {
                                        if node_owner_old[oi] == sg_id.0 && (oi < old_start || oi >= old_end) {
                                            if let Some(nid) = old_to_new[oi] {
                                                eprint!(" {}→{}", oi, nid.0);
                                            }
                                        }
                                    }
                                    eprintln!();
                                }
                            }
                        }
                    }
                }
            }

            sg.node_range = match new_start {
                Some(ns) => (NodeId(ns), NodeId(new_end)),
                None => (NodeId(0), NodeId(0)), // All dead; range collapses.
            };
            // Anchor/metadata NodeId remapping for kept subgraphs no longer
            // happens here — it happens in ONE place after step 6, through
            // `map_node_refs` (the write door), for per-node tables and
            // per-sg anchors alike.
            // nested_ranges: NOT remapped here — step 6 recomputes every kept
            // subgraph's nested_ranges from the final ranges.
            // W5: the flattened condition-tree plan is stale after compaction
            // (node ids AND tree membership changed). Clear it; the engine
            // falls back to the runtime DFS until the pipeline recomputes it
            // post-optimization.
            if let Some(ref mut plan) = sg.reset_plan {
                plan.condition_tree_plan = Vec::new();
            }
        }

        // ── 6. Compact subgraphs: remove empty/dead subgraphs, remap SubGraphIds ──
        // Removals were finalized (with reference-veto cascade) in 5-pre, so
        // every kept SubGraphId reference maps to a kept subgraph.
        {
            let removed_count = remove_sg.iter().filter(|&&r| r).count();

            if removed_count > 0 {
                let mut sg_old_to_new: Vec<u32> = vec![u32::MAX; self.subgraphs.len()];
                let mut new_sgs: Vec<SubGraph> = Vec::with_capacity(self.subgraphs.len() - removed_count);
                for (i, sg) in std::mem::take(&mut self.subgraphs).into_iter().enumerate() {
                    if remove_sg[i] {
                        continue;
                    }
                    sg_old_to_new[i] = new_sgs.len() as u32;
                    new_sgs.push(sg);
                }
                // Vetoed references guarantee map_sg never sees a removed id.
                let map_sg = |id: SubGraphId| -> SubGraphId {
                    SubGraphId(sg_old_to_new[id.0 as usize])
                };
                for sg in &mut new_sgs {
                    sg.id = map_sg(sg.id);
                    sg.function_id = map_sg(SubGraphId(sg.function_id)).0;
                    sg.loop_parent_sg = sg.loop_parent_sg.map(map_sg);
                    for e in &mut sg.defer_table {
                        e.body_subgraph = map_sg(e.body_subgraph);
                    }
                }
                // Debug-only name sidecar follows the same renumbering
                // (entries are parallel to `subgraphs`).
                if !self.sg_debug_names.is_empty() {
                    let mut new_names: Vec<Option<Box<str>>> =
                        Vec::with_capacity(new_sgs.len());
                    for (i, keep) in remove_sg.iter().enumerate() {
                        if *keep {
                            continue;
                        }
                        new_names.push(self.sg_debug_names.get(i).cloned().flatten());
                    }
                    self.sg_debug_names = new_names;
                }
                self.subgraphs = new_sgs;
                if let Some(e) = self.entry_subgraph {
                    self.entry_subgraph = Some(map_sg(e));
                }
                for ct in self.call_targets.iter_mut() {
                    *ct = ct.map(map_sg);
                }
                for ci in self.closure_infos.iter_mut().flatten() {
                    ci.subgraph_id = map_sg(ci.subgraph_id);
                }
                for pi in self.partial_infos.iter_mut().flatten() {
                    pi.subgraph_id = map_sg(pi.subgraph_id);
                }
                for li in self.lazy_construct_infos.iter_mut().flatten() {
                    li.thunk_sg = map_sg(li.thunk_sg);
                }
                for ti in self.trait_construct_infos.iter_mut().flatten() {
                    for m in &mut ti.methods {
                        m.subgraph_id = map_sg(m.subgraph_id);
                    }
                }
                for gb in self.gate_branches.iter_mut().flatten() {
                    for (_, bsg, _) in &mut gb.branches {
                        let mapped = map_sg(*bsg);
                        // The 5-pre reference-veto guarantees no live gate
                        // branch points at a removed subgraph.
                        debug_assert!(mapped.0 != u32::MAX, "gate branch mapped to removed sg");
                        *bsg = mapped;
                    }
                }
                for si in self.select_infos.iter_mut().flatten() {
                    for sb in &mut si.branches {
                        sb.subgraph_id = map_sg(sb.subgraph_id);
                    }
                }
                for owner in self.hoisted_owners.iter_mut() {
                    // u32::MAX sentinel for non-hoisted nodes is preserved.
                    if owner.0 != u32::MAX {
                        *owner = map_sg(*owner);
                    }
                }
                let mut new_vtable: rustc_hash::FxHashMap<(u16, Box<str>), SubGraphId> =
                    rustc_hash::FxHashMap::default();
                for (k, v) in std::mem::take(&mut self.vtable_fallback_dispatch) {
                    new_vtable.insert(k, map_sg(v));
                }
                self.vtable_fallback_dispatch = new_vtable;

                // F-7: per-sg-index derived data is invalidated by renumbering.
                // EngineRef::new recomputes it after build/optimize/load.
                // (Load-path CSR offset tables only exist on loaded graphs,
                // which are never rebuilt.)
                debug_assert!(self.sg_uv_offsets.is_empty());
                self.sg_initial_pending = Vec::new();
                self.sg_initial_seed = Vec::new();
                self.linear_plans = Vec::new();

                if dbg_rebuild {
                    eprintln!(
                        "[REBUILD] compacted subgraphs: removed {} ({} remain)",
                        removed_count,
                        self.subgraphs.len()
                    );
                }
            }
        }

        // ── 6b. Remap every metadata NodeId through the single write door ──
        // Per-node tables were compacted in step 3 (new-indexed, values still
        // old ids) and removed subgraphs are gone after step 6, so the door
        // sees exactly the kept structures. Any metadata NodeId not in
        // old_to_new here means the liveness seeding (Optimizer's
        // compute_liveness, which seeds through the READ door) diverged from
        // this enumeration — an invariant violation, not a salvage case.
        {
            let remap_ref = |site: NodeRefSite, owner: u32, id: NodeId| -> NodeId {
                let r = resolve(id);
                old_to_new[r.0 as usize].unwrap_or_else(|| panic!(
                    "rebuild: node-ref door {:?} (owner {}) ref node {} not live — \
                     liveness seeding diverged from the door enumeration",
                    site, owner, r.0
                ))
            };
            self.map_node_refs(remap_ref);
        }

        // ── 4. Rebuild downstreams (including Gate condition_input edges) ──
        // Runs after the door remap so every NodeId (inputs pool since step 2,
        // gate condition_input since 6b) is new-indexed.
        {
            let n = self.nodes.len();
            self.downstreams = vec![Vec::new(); n];
            for node_idx in 0..n {
                let node = self.nodes[node_idx];
                let inputs = self.inputs_pool.get(node.inputs_offset, node.input_count);
                for &input in inputs {
                    self.downstreams[input.0 as usize].push(NodeId(node_idx as u32));
                }
            }
            // Gate condition_input → Gate edge (aligned with compute_downstreams).
            for nid in 0..n {
                if let Some(gb) = &self.gate_branches[nid] {
                    let node = self.nodes[nid];
                    let inputs = self.inputs_pool.get(node.inputs_offset, node.input_count);
                    if !inputs.contains(&gb.condition_input) {
                        self.downstreams[gb.condition_input.0 as usize].push(NodeId(nid as u32));
                    }
                }
            }
        }

        // Recompute nested_ranges for every kept subgraph from the final
        // ranges: the per-range remap above cannot express ranges that
        // collapsed this round, and it accumulated zero-length residue across
        // rounds (formerly tolerated by the Verifier).
        self.compute_nested_ranges();

        // Verify: check for dangling references.
        if std::env::var("KUZO_VERIFY_GRAPH").is_ok() {
            let total = self.nodes.len();
            for (idx, node) in self.nodes.iter().enumerate() {
                let inputs = self.inputs_pool.get(node.inputs_offset, node.input_count);
                for &inp in inputs {
                    if inp.0 as usize >= total {
                        eprintln!("[VERIFY] node={} has dangling input {} (total={})", idx, inp.0, total);
                    }
                }
            }
            for (sg_idx, sg) in self.subgraphs.iter().enumerate() {
                if sg.entry_node.0 as usize >= total {
                    eprintln!("[VERIFY] sg={} has dangling entry_node {} (total={})", sg_idx, sg.entry_node.0, total);
                }
                if sg.return_node.0 as usize >= total {
                    eprintln!("[VERIFY] sg={} has dangling return_node {} (total={})", sg_idx, sg.return_node.0, total);
                }
                let (s, e) = sg.node_range;
                if e.0 < s.0 || e.0 as usize > total {
                    eprintln!("[VERIFY] sg={} has invalid node_range [{},{}) (total={})", sg_idx, s.0, e.0, total);
                }
            }
        }

        // DEBUG: check if any Gate node is inside its branch subgraph's node_range.
        // This would cause infinite recursion (Gate launches a subgraph that contains itself).
        if std::env::var("KUZO_DEBUG_REBUILD").is_ok() {
            for (idx, gb_opt) in self.gate_branches.iter().enumerate() {
                if let Some(gb) = gb_opt {
                    let gate_node = NodeId(idx as u32);
                    for (cond, branch_sg, _) in &gb.branches {
                        let branch_sg_id = branch_sg.0 as usize;
                        if branch_sg_id < self.subgraphs.len() {
                            let (s, e) = self.subgraphs[branch_sg_id].node_range;
                            if gate_node.0 >= s.0 && gate_node.0 < e.0 {
                                eprintln!("[REBUILD-BUG] Gate node {} is INSIDE branch sg={} (cond={}) node_range [{},{}) function_id={}",
                                    gate_node.0, branch_sg_id, cond, s.0, e.0,
                                    self.subgraphs[branch_sg_id].function_id);
                            }
                        }
                    }
                }
            }
        }

        old_to_new
    }

/// Computes `nested_ranges` for all subgraphs: for each subgraph, the ranges
/// of subgraphs DIRECTLY nested within its `node_range`. Called at build
/// time, after every optimizer `rebuild`, and at `.kzo` load (v3: the section
/// is no longer serialized — this is the derived source of truth). Empty
/// ranges (collapsed or placeholder subgraphs) are never children.
///
/// O(SG.log SG): sort non-empty ranges by (start asc, end desc) and walk with
/// a stack of open ancestors — each range's direct parent is the stack top
/// after popping everything that has ended or does not contain it.
pub fn compute_nested_ranges(&mut self) {
    let mut ranges: Vec<(u32, u32, usize)> = self
        .subgraphs
        .iter()
        .enumerate()
        .filter(|(_, sg)| sg.node_range.0 .0 < sg.node_range.1 .0)
        .map(|(i, sg)| (sg.node_range.0 .0, sg.node_range.1 .0, i))
        .collect();
    ranges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));

    let mut children: Vec<Vec<(u32, u32)>> = vec![Vec::new(); self.subgraphs.len()];
    // Stack of (end, idx) of currently open ancestors (outermost first).
    let mut stack: Vec<(u32, usize)> = Vec::new();
    for &(start, end, idx) in &ranges {
        // Pop everything that cannot contain this range.
        while let Some(&(top_end, _)) = stack.last() {
            if top_end <= start || end > top_end {
                stack.pop();
            } else {
                break;
            }
        }
        if let Some(&(_, parent_idx)) = stack.last() {
            children[parent_idx].push((start, end));
        }
        stack.push((end, idx));
    }
    for (i, sg) in self.subgraphs.iter_mut().enumerate() {
        sg.nested_ranges = std::mem::take(&mut children[i]);
    }
}

    /// W5: flatten every loop subgraph's `reset_condition_tree` roots into the
    /// mechanical `condition_tree_plan` — the same DFS the engine's
    /// `reset_condition_tree` performs, done ONCE at build/load time instead
    /// of on every loop iteration. Uses the mem-agnostic accessors so both the
    /// built and the loaded (zerocopy) graphs can run it.
    pub fn precompute_reset_plans(&mut self) {
        let plans: Vec<(usize, Vec<(NodeId, u16)>)> = self
            .subgraphs
            .iter()
            .enumerate()
            .filter_map(|(i, sg)| {
                let plan = sg.reset_plan.as_ref()?;
                if plan.reset_condition_tree.is_empty() {
                    return None;
                }
                Some((i, self.compute_condition_tree_plan(i, sg, &plan.reset_condition_tree)))
            })
            .collect();
        for (i, computed) in plans {
            if let Some(plan) = self.subgraphs[i].reset_plan.as_mut() {
                plan.condition_tree_plan = computed;
            }
        }
    }

    /// The condition-tree flattening shared semantics with
    /// `Engine::reset_condition_tree`: DFS over the cond roots' inputs, keeping
    /// nodes inside the loop subgraph but outside every nested subgraph range
    /// (Gate nodes included — Bug #38); each node's pending count is the number
    /// of its inputs that are also inside the collected tree.
    fn compute_condition_tree_plan(
        &self,
        loop_sg_idx: usize,
        loop_sg: &SubGraph,
        roots: &[NodeId],
    ) -> Vec<(NodeId, u16)> {
        let (sg_start, sg_end) = loop_sg.node_range;
        // Mem-agnostic nested ranges (zerocopy graphs keep them in the mmap
        // CSR section rather than the owned Vec).
        let nested = self.sg_nested_ranges(loop_sg_idx);
        let is_nested = |gid: u32| nested.iter().any(|&(s, e)| gid >= s && gid < e);
        let is_in_sg = |gid: u32| gid >= sg_start.0 && gid < sg_end.0 && !is_nested(gid);

        let mut visited: rustc_hash::FxHashSet<u32> = rustc_hash::FxHashSet::default();
        let mut stack: Vec<NodeId> = roots.to_vec();
        let mut cond_nodes: Vec<NodeId> = Vec::new();
        while let Some(gid) = stack.pop() {
            if !visited.insert(gid.0) {
                continue;
            }
            if !is_in_sg(gid.0) {
                continue;
            }
            cond_nodes.push(gid);
            let node = self.node(gid.0 as usize);
            let inputs = self.inputs(node.inputs_offset, node.input_count);
            for &inp in inputs {
                stack.push(inp);
            }
        }

        cond_nodes
            .into_iter()
            .map(|gid| {
                let node = self.node(gid.0 as usize);
                let inputs = self.inputs(node.inputs_offset, node.input_count);
                let pending = inputs
                    .iter()
                    .filter(|&&inp| visited.contains(&inp.0) && is_in_sg(inp.0))
                    .count() as u16;
                (gid, pending)
            })
            .collect()
    }
}

impl Default for DataFlowGraph {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// expr_id_to_key — ExprId → expr_types key conversion
// =========================================================================

/// ExprId → expr_types key (`u64`).
///
/// Sema's expr_types key is the "AST Expr handle address"; investigation shows
/// `key = ExprId.0 as u64` (the index into `AstArena.exprs`).
#[inline]
pub fn expr_id_to_key(id: crate::ast::Ast::ExprId) -> u64 {
    id.0 as u64
}

// =========================================================================
// LoopContext — loop context (continue jump target + For-loop iterator node)
// =========================================================================

/// Loop context: pushed onto `loop_stack` for continue/break semantics.
///
/// - `sg`: recursive subgraph id (continue jump target).
/// - `iter_node`: iterator parameter node in the For-loop's `body_sg` (continue
///   must pass it to the tail recursion; `None` = While/Loop, param_count=0,
///   no argument needed).
/// - `body_node_start`: starting node id of the loop body subgraph; used to
///   determine whether a captured variable is defined inside the loop body.
///   Variables defined inside the loop body become inaccessible once the loop
///   body frame is destroyed, so closures capturing such variables must take
///   the Cell path.
#[derive(Debug, Clone, Copy)]
pub struct LoopContext {
    pub sg: SubGraphId,
    pub iter_node: Option<NodeId>,
    pub body_node_start: u32,
    /// For-loop's current-value param node (bound to the loop variable `name`).
    /// Used by defer-in-loop: CF_DEFER_REGISTER captures this node's value per-iteration
    /// so each defer body reads the iteration's `i` rather than the final value.
    /// None for while/loop (no loop variable).
    pub loop_var_node: Option<NodeId>,
}

// =========================================================================
// IrBuilder — builds a DataFlowGraph from SemaResult + Module
// =========================================================================

/// IR builder: walks the AST and emits Node + InputsPool + SubGraph.
///
/// Compiles subgraphs on a per-function basis:
/// 1. Register all functions as SubGraphs.

// =========================================================================
// SCALAR_META — scalar-type arithmetic metadata (keyed by ValueTag; name derived
//               single-source from Value.rs)
// =========================================================================
//
// Centralizes, for each scalar type:
//   - arith_base:  arithmetic compute_fn base index (matches compute_fn_table! indices).
//   - family:      comparison dispatch family ("i32"/"i64"/"i128"/"float"/"bool").
//   - is_float:    whether floating-point (governs bitwise-op availability, neg offset).
//
// The type-name ↔ ValueTag mapping is maintained single-source by
// `Value::ValueTag::from_name`/`type_name`; this table does not duplicate the
// name field. arith_base must strictly match the indices in compute_fn_table!.

/// Scalar-type arithmetic metadata.
///
/// `family` is the `TypeFamily` enum (a unified type family; callers combine
/// integer variants with `|` to dispatch by bit-width).
pub struct ScalarMeta {
    pub arith_base: u32,
    pub family: crate::types::TypeFamily,
    pub is_float: bool,
}

/// Looks up arithmetic metadata by ValueTag (const fn, evaluable at compile time).
///
/// `family` is derived from `ValueTag::family()` (single-source; no longer
/// hand-written 18-arm match).
pub const fn scalar_meta(tag: crate::value::ValueTag) -> Option<ScalarMeta> {
    use crate::value::ValueTag;
    // family is derived from ValueTag::family() (single source of truth).
    let family = tag.family();
    Some(match tag {
        // 12 integer types (arith_base starts at 92, 12 indices each).
        ValueTag::I8    => ScalarMeta { arith_base: 92,  family, is_float: false },
        ValueTag::I16   => ScalarMeta { arith_base: 104, family, is_float: false },
        ValueTag::I32   => ScalarMeta { arith_base: 116, family, is_float: false },
        ValueTag::I64   => ScalarMeta { arith_base: 128, family, is_float: false },
        ValueTag::I128  => ScalarMeta { arith_base: 140, family, is_float: false },
        ValueTag::U8    => ScalarMeta { arith_base: 152, family, is_float: false },
        ValueTag::U16   => ScalarMeta { arith_base: 164, family, is_float: false },
        ValueTag::U32   => ScalarMeta { arith_base: 176, family, is_float: false },
        ValueTag::U64   => ScalarMeta { arith_base: 188, family, is_float: false },
        ValueTag::U128  => ScalarMeta { arith_base: 200, family, is_float: false },
        ValueTag::Isize => ScalarMeta { arith_base: 212, family, is_float: false },
        ValueTag::Usize => ScalarMeta { arith_base: 224, family, is_float: false },
        // 4 floating-point types (arith_base starts at 236, 6 indices each).
        ValueTag::F16   => ScalarMeta { arith_base: 236, family, is_float: true },
        ValueTag::F32   => ScalarMeta { arith_base: 242, family, is_float: true },
        ValueTag::F64   => ScalarMeta { arith_base: 248, family, is_float: true },
        ValueTag::F128  => ScalarMeta { arith_base: 254, family, is_float: true },
        // Non-arithmetic scalar types (bool/char, no arith_base).
        ValueTag::Bool  => ScalarMeta { arith_base: 0,   family, is_float: false },
        ValueTag::Char  => ScalarMeta { arith_base: 0,   family, is_float: false },
        // Non-scalar tag: no arithmetic metadata.
        _ => return None,
    })
}

// =========================================================================
// Effect classification tests (W1)
// =========================================================================

#[cfg(test)]
mod effect_classification_tests {
    use super::*;

    /// Verbatim copy of the PRE-W1 hand-written pure set (the behavior W1 must
    /// preserve exactly). Keep frozen; if `pure_compute_fn_set()` drifts from
    /// this, the equivalence guarantee is broken.
    fn legacy_pure_set() -> rustc_hash::FxHashSet<ComputeFnId> {
        let mut s = rustc_hash::FxHashSet::default();
        for id in 1..=27u32 { s.insert(ComputeFnId(id)); }
        for id in 50..=91u32 { s.insert(ComputeFnId(id)); }
        for id in 92..=259u32 { s.insert(ComputeFnId(id)); }
        s.insert(CF_RECORD_FIELD_GET);
        s.insert(CF_ARRAY_INDEX);
        s.insert(CF_IS_NULL);
        s.insert(CF_ARRAY_LEN);
        s.insert(CF_REF_EQ);
        s.insert(CF_REF_NEQ);
        s.insert(CF_ELVIS);
        s.insert(CF_PATTERN_CTOR_MATCH);
        s.insert(CF_PATTERN_ADT_FIELD_GET);
        s.insert(CF_PATTERN_STR_EQ);
        s.insert(CF_CAST_SCALAR);
        s.insert(CF_NON_NULL_ASSERT);
        s.insert(CF_STR_BYTES);
        s.insert(CF_EQ_STR);
        s.insert(CF_NE_STR);
        s.insert(CF_LT_STR);
        s.insert(CF_GT_STR);
        s.insert(CF_LE_STR);
        s.insert(CF_GE_STR);
        s.insert(CF_EQ_OBJ);
        s.insert(CF_NE_OBJ);
        s.insert(CF_NE_BOOL);
        s.insert(CF_EQ_F128);
        s.insert(CF_NE_F128);
        s.insert(CF_LT_F128);
        s.insert(CF_GT_F128);
        s.insert(CF_LE_F128);
        s.insert(CF_GE_F128);
        s
    }

    /// The derived set must equal the legacy set member-for-member.
    #[test]
    fn pure_set_equivalence_with_legacy() {
        let new = pure_compute_fn_set();
        let old = legacy_pure_set();
        let missing: Vec<_> = old.difference(&new).collect();
        let added: Vec<_> = new.difference(&old).collect();
        assert!(
            missing.is_empty() && added.is_empty(),
            "pure set drift: lost {:?}, gained {:?}",
            missing,
            added
        );
    }

    /// Every compute fn id in the table must be classified (effect_class
    /// panics otherwise — this test turns that into a named failure).
    #[test]
    fn classification_covers_all_cfs() {
        for id in 0..COMPUTE_FN_TABLE_LEN {
            let _ = effect_class(ComputeFnId(id));
        }
    }

    /// Aliasing reads are classified ReadMutable; in-place mutators are
    /// WriteMutable (the graph_pure_set contract depends on this).
    #[test]
    fn aliasing_contract() {
        for cf in aliasing_read_cfs() {
            assert_eq!(effect_class(cf), EffectClass::ReadMutable, "cf {}", cf.0);
        }
        for cf in [CF_RECORD_FIELD_SET, CF_ARRAY_STORE, CF_DEREF_WRITE] {
            assert_eq!(effect_class(cf), EffectClass::WriteMutable, "cf {}", cf.0);
        }
    }
}
