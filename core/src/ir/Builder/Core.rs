//! Core — IrBuilder struct, core state/scope/lookup helpers, compile_expr and
//! build() entry points. Extracted from the former Builder.rs (no logic changes).

use super::*;

/// 2. Compile the body of each function.
/// 3. Compute fan-out (downstreams).
///
/// This stage implements compilation of the core Expr variants
/// (Const/BinOp/Call/FieldAccess/Ident/Block).
/// Control flow (If/Match/Loop) is deferred to stage 4.
/// `compute_fn` is placeholdered with `noop_compute`; the Engine stage replaces it
/// with a type-specialized function.
pub struct IrBuilder<'a> {
    pub sema: &'a crate::sema::Sema::SemaResult,
    pub type_arena: &'a crate::sema::Sema::TypeArena,
    pub module: &'a crate::ast::Ast::Module<'a>,
    /// List of builtin modules (precompiled; functions are registered into `func_subgraphs`).
    pub builtin_modules: Vec<&'a crate::ast::Ast::Module<'a>>,
    /// The builtin module currently being compiled (`None` = user module).
    pub compiling_builtin: Option<&'a crate::ast::Ast::Module<'a>>,
    /// Static analysis report (for the entry module).
    pub analysis: Option<&'a crate::pass::Analyzer::AnalysisReport>,
    /// Static analysis reports for builtin modules (indexed in parallel with `builtin_modules`).
    pub builtin_analyses: Vec<Option<&'a crate::pass::Analyzer::AnalysisReport>>,
    pub graph: DataFlowGraph,
    /// Function name -> subgraph id map (looked up when compiling `Call` to bind `call_target`).
    /// Key kinds: bare (builtin + user/dep only — std never registers bare),
    /// mangled (`std.io.File.remove`), short-qualified (`File.remove`),
    /// package (`std.math::ldexp_impl`), generic instance (`name#id`).
    /// ALL lookups go through `resolve_func` — the single resolution point.
    pub func_subgraphs: rustc_hash::FxHashMap<String, SubGraphId>,
    /// Short-qualified index (`Module.fn` by module TAIL name), SEPARATE from
    /// func_subgraphs: a root-level user module's own mangled key ("Parse.parse",
    /// no directory prefix) is STRING-IDENTICAL to a std module's short key
    /// (std.json.Parse → "Parse.parse"). Sharing one map made the preregistration
    /// reuse logic treat the two distinct functions as one — the later-compiled
    /// user body overwrote the std subgraph, so calls through the std full-path
    /// key silently executed the user function (BOOTSTRAP 1C root fix).
    pub func_short_index: rustc_hash::FxHashMap<String, SubGraphId>,
    /// Collision tripwire, all key families: key → qualified display names of
    /// the DISTINCT functions that competed for one slot during registration
    /// (bare, short-qualified, package, import alias). `resolve_func` turns
    /// any call resolving through a conflicted key into a hard compile error
    /// — never a silent first/last-writer-wins (the File.remove/Env.remove
    /// incident: identical signatures, wrong function, perfect fake success).
    pub name_conflicts: rustc_hash::FxHashMap<String, Vec<String>>,
    /// Global unique-name index across ALL modules (std included): bare name
    /// → sg. The sema layer predeclares every loaded module's functions into
    /// the root env, so a bare call to a GLOBALLY UNIQUE name type-checks
    /// from anywhere (e.g. `is_tty(0)` after `import
    /// std.os.Tty`); the IR must honor the same contract. Names
    /// contested by ≥2 distinct functions go to the tripwire instead —
    /// first registrant stays indexed, and any bare call through the
    /// contested name is a hard error.
    pub global_bare_index: rustc_hash::FxHashMap<String, SubGraphId>,
    /// Bare names of functions declared `@internal` (stdlib implementation
    /// primitives). `internal_access_blocked` is the single predicate guarding
    /// both resolution paths: `resolve_func` (subgraph targets) and the
    /// extern-dispatch branch in `compile_call` (CF_DYN_FFI_CALL — extern
    /// functions never enter `func_subgraphs`, so the resolve_func guard
    /// alone would miss them).
    pub internal_funcs: rustc_hash::FxHashSet<String>,
    /// sg → qualified display name, for collision diagnostics.
    pub sg_qualified_names: rustc_hash::FxHashMap<SubGraphId, String>,
    /// Type method subgraph table: (type_id, method_idx) -> SubGraphId.
    /// `type_id = FIRST_DYNAMIC_TYPE_ID + type_def_index`; `method_idx` is the position of the
    /// method within `TypeDefInfo.methods`. Replaces the `"TypeName.method"` string key
    /// previously stored in `func_subgraphs`.
    pub method_subgraphs: rustc_hash::FxHashMap<(u16, u16), SubGraphId>,
    /// Trait default-method subgraph table (monomorphized): (type_id, trait_def_idx, method_idx) -> SubGraphId.
    /// A specialized subgraph is generated for each type implementing the trait, so that `self`
    /// carries concrete type information inside the body.
    pub trait_default_subgraphs: rustc_hash::FxHashMap<(u16, u16, u16), SubGraphId>,
    /// Index into `sema.trait_default_instances` of the trait default-method specialization
    /// currently being compiled.
    /// `expr_type_name`/`expr_type_id` use this index to look up
    /// `TraitDefaultInstance.type_name` in sema and obtain the concrete type of `self`
    /// (consuming sema output; the IR does not hold semantic information).
    pub current_trait_default_idx: Option<usize>,
    /// Allocation counter for inheritance base-dispatch site indices (vtable
    /// keys). Starts at 0x8000 to stay clear of trait method indices.
    pub next_base_dispatch_idx: u16,
    /// The subgraph id of the function currently being compiled (used for `defer` registration).
    pub current_function_sg: Option<SubGraphId>,
    /// W3C region context: the INNERMOST branch/loop-body subgraph currently
    /// being compiled (None at function level). `build_await_node` registers
    /// EventSourceDecls here first — structurally correct scoping that made
    /// the post-hoc "drain decls from the function sg into the branch sg"
    /// migrations (Bug #24) unnecessary. Save/restore around nested bodies;
    /// cleared on function/lambda entry (their bodies are new function scopes).
    pub current_branch_sg: Option<SubGraphId>,
    /// Loop context stack: the top entry is the current loop's context
    /// (continue jump target + For iterator node).
    pub loop_stack: Vec<LoopContext>,
    /// Variable scope stack: variable name -> NodeId producing that variable's value.
    pub scope_stack: Vec<rustc_hash::FxHashMap<String, NodeId>>,
    /// Captured-variable scope stack: per lambda layer, the list of captured
    /// variables `(name, outer_node)`.
    /// Used by `Assignment` to decide whether a WriteBack is required: assignments to
    /// captured variables must write back to the outer node.
    pub captured_scopes: Vec<Vec<(String, NodeId)>>,
    /// Local variables captured by an inner lambda: variable name -> original node id at
    /// capture time.
    /// When an outer `Assignment` assigns to one of these variables it must emit a WriteBack
    /// to the original node, so the closure can read the latest value from the parent frame
    /// during a `same_function` call (capture-by-reference semantics).
    pub captured_vars: rustc_hash::FxHashMap<String, NodeId>,
    /// Place-model C1-①: names of THIS function whose address is taken
    /// (`&x` outside lambda/nested-fn bodies), collected by the pre-pass in
    /// `compile_function_body`. Their `val`/`var` declarations lower to a
    /// Cell allocation at the DECL SITE and every name read/write routes
    /// through it (`&x` then costs nothing — the cell already exists).
    pub fn_address_taken: rustc_hash::FxHashSet<String>,
    /// Names read inside lambda / nested-function bodies in this function.
    /// A binding both address-taken AND captured stays plain: the capture
    /// machinery snapshots the binding node, which must remain the raw value.
    pub fn_lambda_captured: rustc_hash::FxHashSet<String>,
    /// Scope-parallel stack of cell-backed bindings: name -> (cell node, owner
    /// `current_function_id`). Pushed/popped with `scope_stack`; the owner tag
    /// blocks cross-function leakage through the shared scope chain. A
    /// cell-backed name's `scope_stack` entry points at the cell node itself.
    pub cell_bound: Vec<rustc_hash::FxHashMap<String, (NodeId, u32)>>,
    /// Place-model all-vars (C1-③④): true while compiling a function whose
    /// scalar `var`s are ALL cell-backed (every non-transformed function).
    /// Drives decl-site backing and the compile-time store→load forwarding.
    pub fn_all_vars_slot: bool,
    /// Compile-time forwarding memory: cell node -> the node producing its
    /// CURRENT value. Reads of a tracked cell forward to that node directly
    /// (zero-cost SSA edge, the mem2reg equivalent); stores update it;
    /// barriers (loop bodies, branch exits, defer bodies) snapshot/clear it.
    /// Only non-escaped cells are tracked (`no_forward_cells`).
    pub cell_values: rustc_hash::FxHashMap<NodeId, NodeId>,
    /// Cells whose reference escaped (`&x`): calls or stored refs may write
    /// them, so forwarding would read stale compile-time values. Reads of
    /// these cells always emit CF_DEREF_READ loads.
    pub no_forward_cells: rustc_hash::FxHashSet<NodeId>,
    /// Entry-call argument nodes for the while sg most recently registered
    /// (the loop-carried cell params initial values). Consumed by the Stmt
    /// While branch when emitting the launch call.
    pub while_entry_args: Vec<NodeId>,
    /// Function subgraphs that contain a function-level defer (Bug #49). Historically this
    /// was `!defer_table.is_empty()`; now that defers register dynamically at runtime
    /// (CF_BLOCK_DEFER_REGISTER + frame.defer_stack), the table is always empty, so this
    /// builder-side set is the flag. Local reassignment inside such functions emits a
    /// WriteBack to the original node so the defer body reads the LATEST value
    /// (Reference-mode captures).
    pub function_defer_sgs: rustc_hash::FxHashSet<SubGraphId>,
    /// Type-field scope stack: constructor/type name -> field name list
    /// (managed in parallel with `scope_stack`).
    pub type_scope_stack: Vec<rustc_hash::FxHashMap<String, TypeFieldInfo>>,
    /// Per-module type-field scopes (module-scoped type resolution): module
    /// name -> (bare type/ctor name -> TypeFieldInfo with CANONICAL
    /// type_name). Own-module bindings shadow the base layer (`type_scope_stack[0]`),
    /// so a local `type List` wins inside its module without clobbering the
    /// std binding other modules compile against. Only user modules carry
    /// entries — std/builtin keep the historical flat base layer.
    pub module_type_scopes: rustc_hash::FxHashMap<String, rustc_hash::FxHashMap<String, TypeFieldInfo>>,
    /// `function_id` of the function currently being compiled (used for subgraph tagging and
    /// `root_frame_ptr` inheritance decisions).
    pub current_function_id: u32,
    /// Starting NodeId of the subgraph currently being compiled (used to determine whether a
    /// variable belongs to an outer scope).
    pub current_sg_start: u32,
    /// The previous effect node in the current statement block (subsequent effect nodes depend
    /// on it to preserve statement ordering).
    pub current_effect: Option<NodeId>,
    /// Whether the current position is a tail position (for tail-call analysis).
    /// Set to `true` at the `compile_function` entry and at `Return value`;
    /// inherited by `Block` trailing expressions and by `If`/`Match` branches;
    /// set to `false` for arguments, conditions, and assignment right-hand sides.
    pub in_tail_position: bool,
    /// Bug #97: whether the current function's return type is Throw (one Async layer
    /// unfolded). A tail-position `expr?` must produce the FUNCTION's return value:
    /// the propagate node yields the UNWRAPPED value (statement/value use), so the
    /// Propagate lowering re-wraps it with `compute_throw_ok` when this is set.
    pub fn_returns_throw: bool,
    /// Bug #100: canonical "home" slot per variable name (the first node the name was
    /// bound to — the var-decl node, which is also every WriteBack's target). A loop
    /// condition must read loop-modified variables through their HOME node (the slot
    /// WriteBacks keep current across iterations), not through the mid-chain node of a
    /// previous assignment — otherwise the condition re-evaluates against a stale
    /// snapshot every iteration and the loop never terminates.
    /// Bug #66: Whether the current `compile_block` call is the function body's top-level block.
    /// Set to `true` at `compile_function_body` entry; reset to `false` by the first `compile_block`
    /// call (the function body itself). Nested blocks see `false`, so only they extract
    /// block-scoped defers — function-level defers stay in `defer_table` for function-exit execution.
    pub in_function_top_block: bool,
    /// Depth of the scope_stack at function entry (the parameter scope). Used by defer body
    /// compilation to resolve external variables to parameter nodes (authoritative) rather than
    /// body-local rebindings (which carry stale values in copied defer frames).
    pub param_scope_depth: usize,
    /// Whether the current `compile_block` call is inside a loop body subgraph.
    /// Set to `true` by `compile_loop_body_subgraph`; reset to `false` by `compile_block`
    /// (so nested blocks within the loop body see `false`). When `true`, `defer` statements
    /// compile to `CF_DEFER_REGISTER` nodes (dynamic defer_stack registration) instead of
    /// being registered to the static `defer_table`; the loop-exit `CF_DEFER_RUN` node drains
    /// the stack in LIFO order, capturing per-iteration values.
    pub in_loop_body: bool,
    /// Tail-recursion-to-iteration context: when `Some`, `compile_call` intercepts self-calls
    /// as `WriteBack + Call(while_sg)`.
    /// `None` = not compiling a tail-recursion-to-iteration body.
    pub(crate) tail_rec_ctx: Option<TailRecCtx>,
    /// Non-tail-recursion-to-iteration context.
    pub(crate) non_tail_rec_ctx: Option<NonTailRecCtx>,
    /// Type-parameter mapping for the monomorphization instance currently being compiled
    /// (type-parameter name -> TypeHandle).
    /// Empty means we are not in a generic-instantiation context (a plain non-generic function).
    /// `compile_cast_call` consults this table to substitute concrete types when resolving target
    /// type parameters; `expr_type_name` falls back to the instance-local `expr_types` when sema's
    /// `expr_types` misses.
    pub current_type_args: Vec<(String, crate::sema::Sema::TypeHandle)>,
    /// ID of the monomorphization instance currently being compiled (`None` = non-generic function).
    /// Used as an index into `sema.monomorph_instances` to look up instance-local `expr_types`.
    pub current_instance_id: Option<u32>,
    /// The owning type (name, type_id) of the method currently being compiled.
    /// Used by `compile_method_call` to dispatch implicit-this method calls (where `recv`
    /// is the callee Ident, not the receiver — the receiver is `this`).
    pub current_method_type: Option<(Box<str>, u16)>,
    /// Static type name of the match scrutinee currently being pattern-compiled
    /// (`None` outside `compile_match`). Literal patterns route their equality
    /// compute_fn by this: the literal's own spelling (suffix/magnitude) cannot
    /// recover the scrutinee's width, and the eq_i32 default truncated both
    /// sides (i64 4294967296 falsely matched pattern 0).
    pub pattern_scrutinee_ty: Option<Box<str>>,
    /// Compile-time error list (unimplemented features, missing functions, etc.; inspectable
    /// after compilation).
    pub errors: Vec<String>,
    /// Global variable name -> slot index map (top-level `var`/`val` declarations, shared across
    /// functions).
    pub global_var_slots: rustc_hash::FxHashMap<String, u32>,
    /// Top-level `var`/`val` declaration statement list (initialization code is injected when
    /// compiling the entry function).
    /// Element: (module index, StmtId); `None` = entry module, `Some(i)` = `builtin_modules[i]`.
    pub top_level_var_decls: Vec<(Option<usize>, crate::ast::Ast::StmtId)>,
    /// Memoization cache table counter (each memoized function is allocated a `table_index`).
    pub memo_table_count: u32,
    /// String intern pool (written during construction; moved into `graph.string_pool` at the end
    /// of `build()`).
    pub string_pool: Vec<u8>,
    /// String intern dedup table: string content -> offset within `string_pool`.
    pub string_map: rustc_hash::FxHashMap<String, u32>,
    // Escape analysis is produced uniformly by the analyzer (analyze_escape); the IR consumes
    // it via `analysis.escape`. The old `escape_context_stack` has been removed.
}

// =========================================================================
// Builtin constructor / method / cast dispatch registry
// (data-driven, eliminates special-case branches on method/type names).
// =========================================================================

// (FFI_INTRINSIC_TABLE removed: reflect primitives now lower directly to
// CF_REFLECT_* compute_fns via reflect_top_level_cf / reflect_method_intrinsic,
// never reaching the @extern("C") dispatch path.)

// =========================================================================
// Escape analysis has been migrated to the analyzer
// (analyze_escape + analyze_lambda_escape). The IR consumes it via
// `analysis.escape`; there is no longer a parallel implementation.
// =========================================================================

/// Lowering strategy for builtin constructors.
///
/// Adding a builtin constructor only requires appending a row to `BUILTIN_CTORS`;
/// no new `if` branch is needed.
pub(super) enum BuiltinCtorLower {
    /// `Ok(val)`: single-node `compute_throw_ok` (idx 44).
    Ok,
    /// `Err(...)`: inner `record_construct` wrapped by outer `throw_err` (idx 45).
    Err,
    /// `channel(capacity)`: single-node `compute_channel_create` (idx 283).
    Channel,
}

/// Builtin constructor dispatch table: constructor name -> lowering strategy.
pub(super) const BUILTIN_CTORS: &[(&str, BuiltinCtorLower)] = &[
    ("Ok", BuiltinCtorLower::Ok),
    ("Err", BuiltinCtorLower::Err),
    ("channel", BuiltinCtorLower::Channel),
];

/// reflect top-level function → standalone compute_fn mapping.
///
/// `format(x)` / `type_name(x)` are the two reflect entry points called from
/// generic contexts (e.g. Console.frond `print<T>`). Lowering them directly to
/// `CF_REFLECT_*` keeps the hot path off the FFI dispatch table.
/// The remaining reflect primitives are only reachable as trait-style method
/// calls (`x.kind()`, `x.field_count()`, ...) and are dispatched via
/// `lookup_intrinsic` + `try_lower_intrinsic`.
pub(super) fn reflect_top_level_cf(name: &str) -> Option<ComputeFnId> {
    use crate::ir::Ir::*;
    match name {
        "repr" => Some(CF_REFLECT_FORMAT),
        "type_name" => Some(CF_REFLECT_TYPE_NAME),
        _ => None,
    }
}

/// reflect method-name → (IntrinsicKind, arg_count) mapping.
///
/// Used by `lookup_intrinsic` to give every value — regardless of its static
/// type — access to reflect trait methods (`x.kind()`, `x.repr()`, ...).
/// This is the "auto-impl" of `trait Type` / `trait Value`: rather than
/// synthesizing witness-table entries and method bodies for every type, the
/// Builder recognizes reflect method names structurally and lowers them
/// directly to the corresponding `CF_REFLECT_*` compute_fn.
pub(super) fn reflect_method_intrinsic(method: &str) -> Option<(crate::sema::Sema::IntrinsicKind, usize)> {
    use crate::sema::Sema::IntrinsicKind;
    // UnOp: receiver only, no extra args
    let un = |id: u32| Some((IntrinsicKind::UnOp(id), 0));
    // BinOp: receiver + one index arg
    let bin = |id: u32| Some((IntrinsicKind::BinOp(id), 1));
    match method {
        "kind" => un(325),              // CF_REFLECT_KIND_STR (kind() returns str)
        "type_name" => un(324),         // CF_REFLECT_TYPE_NAME
        "size" => un(327),              // CF_REFLECT_LAYOUT_SIZE (aggregate)
        "alignment" => un(328),         // CF_REFLECT_LAYOUT_ALIGN
        "field_count" => un(329),       // CF_REFLECT_FIELD_COUNT
        "repr" => un(288),               // CF_REFLECT_FORMAT (renamed from format 2026-08-17)
        "constructor" => un(333),       // CF_REFLECT_ADT_CTOR
        "clone" => un(348),             // CF_REFLECT_CLONE (deep copy, data domain)
        "field_name" => bin(330),       // CF_REFLECT_FIELD_NAME
        // field_value removed: its return type cannot be expressed without an "any"
        // type in Frond's type system. CF_REFLECT_FIELD_VALUE (331) remains implemented
        // in Compute.rs for potential future use (e.g. a typed field_value<T>(i): T).
        _ => None,
    }
}

/// Tail-recursion-to-iteration context: used by `compile_call` when intercepting self-calls.
/// `self_name` is the current function name; `param_cells` are the parameter register Cells
/// (B2): tail-call actuals are stored through CF_DEREF_WRITE and the re-evaluated condition
/// loads the current values — replacing the old WriteBack param-register machinery.
#[derive(Clone)]
pub(crate) struct TailRecCtx {
    pub(super) self_name: String,
    pub(super) param_cells: Vec<NodeId>,
    /// L3' slot transport flag: non-empty = params ride while_sg PARAM
    /// slots + ResetPlan carries (no Cells); the tail call lowers to a bare
    /// void node. The entries are the while_sg param Const nodes.
    pub(super) slot_params: Vec<NodeId>,
}

/// call_result_map marker: the mapped call's result lives in the nontail-rec
/// converter's result Cell — the consumer synthesizes a CF_DEREF_READ in its
/// own state subgraph (never a valid NodeId).
pub(crate) const RESULT_CELL_MARKER: NodeId = NodeId(u32::MAX);

/// Non-tail-recursion-to-iteration context: intercepts self-calls as `push + continue` while
/// compiling `body_sg`.
#[derive(Clone)]
pub(crate) struct NonTailRecCtx {
    /// The function's own name.
    pub self_name: String,
    /// Function parameter node list (updated to the current frame's `param_cur` node when
    /// compiling continuations).
    pub param_nodes: Vec<NodeId>,
    /// Work-stack array node (a local variable within the function subgraph).
    pub stack_node: NodeId,
    /// Stack-pointer Cell (B3): sp lives across iterations as engine state.
    pub sp_cell: NodeId,
    /// Result Cell (B3): the deepest completed state's result.
    pub result_cell: NodeId,
    /// Call-site ExprId -> node mapping.
    /// When compiling a continuation, encountering an ExprId in this map returns the
    /// corresponding node (a `saved` node, or [`RESULT_CELL_MARKER`] for the
    /// result Cell).
    pub call_result_map: rustc_hash::FxHashMap<crate::ast::Ast::ExprId, NodeId>,
    /// Truncation flag: set to `true` after intercepting the first self-call; subsequent
    /// self-calls generate a void constant.
    pub truncated: bool,
    /// Frame stride = param_count + 1 (state) + max_saved_count.
    pub stride: u32,
    /// Number of function parameters.
    pub param_count: usize,
    /// Maximum number of saved values = `call_sites.len() - 1`.
    pub max_saved: usize,
    /// The state number currently being compiled (0 = INIT).
    pub current_state: u32,
    /// Saved node list for the current frame (read from the frame during the `body_sg` pop
    /// phase).
    pub saved_nodes: Vec<NodeId>,
}

impl<'a> IrBuilder<'a> {
    /// Create a new builder.
    pub fn new(sema: &'a crate::sema::Sema::SemaResult, type_arena: &'a crate::sema::Sema::TypeArena, module: &'a crate::ast::Ast::Module<'a>) -> Self {
        Self {
            sema,
            type_arena,
            module,
            builtin_modules: Vec::new(),
            compiling_builtin: None,
            analysis: None,
            builtin_analyses: Vec::new(),
            graph: DataFlowGraph::new(),
            func_subgraphs: rustc_hash::FxHashMap::default(),
            func_short_index: rustc_hash::FxHashMap::default(),
            name_conflicts: rustc_hash::FxHashMap::default(),
            global_bare_index: rustc_hash::FxHashMap::default(),
            internal_funcs: rustc_hash::FxHashSet::default(),
            sg_qualified_names: rustc_hash::FxHashMap::default(),
            method_subgraphs: rustc_hash::FxHashMap::default(),
            trait_default_subgraphs: rustc_hash::FxHashMap::default(),
            current_trait_default_idx: None,
            next_base_dispatch_idx: 0x8000,
            current_function_sg: None,
            current_branch_sg: None,
            loop_stack: Vec::new(),
            scope_stack: Vec::new(),
            captured_scopes: Vec::new(),
            captured_vars: rustc_hash::FxHashMap::default(),
            fn_address_taken: rustc_hash::FxHashSet::default(),
            fn_lambda_captured: rustc_hash::FxHashSet::default(),
            cell_bound: Vec::new(),
            fn_all_vars_slot: false,
            cell_values: rustc_hash::FxHashMap::default(),
            no_forward_cells: rustc_hash::FxHashSet::default(),
            while_entry_args: Vec::new(),
            function_defer_sgs: rustc_hash::FxHashSet::default(),
            current_function_id: 0,
            current_sg_start: 0,
            current_effect: None,
            in_tail_position: false,
            fn_returns_throw: false,
            in_function_top_block: false,
            param_scope_depth: 0,
            in_loop_body: false,
            tail_rec_ctx: None,
            non_tail_rec_ctx: None,
            current_type_args: Vec::new(),
            current_instance_id: None,
            current_method_type: None,
            pattern_scrutinee_ty: None,
            errors: Vec::new(),
            global_var_slots: rustc_hash::FxHashMap::default(),
            top_level_var_decls: Vec::new(),
            type_scope_stack: Vec::new(),
            module_type_scopes: rustc_hash::FxHashMap::default(),
            memo_table_count: 0,
            string_pool: Vec::new(),
            string_map: rustc_hash::FxHashMap::default(),
        }
    }

    /// String interning: append the string content to `string_pool` and return `(offset, len)`.
    /// Identical strings are stored only once (deduplication via `string_map`).
    pub fn intern_str(&mut self, s: &str) -> (u32, u32) {
        let len = s.len() as u32;
        if len == 0 {
            return (0, 0);
        }
        if let Some(&off) = self.string_map.get(s) {
            return (off, len);
        }
        let off = self.string_pool.len() as u32;
        self.string_pool.extend_from_slice(s.as_bytes());
        self.string_map.insert(s.to_string(), off);
        (off, len)
    }

    /// Set the builtin module list (builder style, chainable).
    pub fn with_builtins(
        mut self,
        modules: Vec<&'a crate::ast::Ast::Module<'a>>,
    ) -> Self {
        self.builtin_modules = modules;
        self
    }

    /// Inject the static analysis report (builder style, chainable).
    /// The report is only valid for the entry module; when compiling the entry module the
    /// `IrBuilder` consults the report to skip dead code/dead functions and to apply inlining
    /// and stack-allocation annotations.
    pub fn with_analysis(
        mut self,
        analysis: &'a crate::pass::Analyzer::AnalysisReport,
    ) -> Self {
        self.analysis = Some(analysis);
        self
    }

    /// Inject static analysis reports for builtin modules (indexed in parallel with
    /// `builtin_modules`).
    pub fn with_builtin_analyses(
        mut self,
        analyses: Vec<Option<&'a crate::pass::Analyzer::AnalysisReport>>,
    ) -> Self {
        self.builtin_analyses = analyses;
        self
    }

    /// Return the static analysis report for the module currently being compiled (generic
    /// query entry point).
    /// Entry module -> `self.analysis`; builtin module -> the corresponding index in
    /// `builtin_analyses`.
    /// The `compiling_builtin` restriction is removed; all modules uniformly go through this
    /// entry to query memoize/inline/dead_code, etc.
    #[inline]
    pub(super) fn current_analysis(&self) -> Option<&'a crate::pass::Analyzer::AnalysisReport> {
        if let Some(builtin) = self.compiling_builtin {
            let idx = self.builtin_modules.iter()
                .position(|&m| std::ptr::eq(m, builtin))?;
            self.builtin_analyses.get(idx).copied().flatten()
        } else {
            self.analysis
        }
    }

    /// Query whether a statement is dead code.
    #[inline]
    pub(super) fn is_dead_stmt(&self, stmt_id: crate::ast::Ast::StmtId) -> bool {
        self.current_analysis().map_or(false, |r| r.dead_code.dead_stmts.contains(&stmt_id))
    }

    /// Query whether a function is dead.
    /// `FuncId` = index into the current module's `declarations`.
    #[inline]
    pub(super) fn is_dead_func(&self, decl_idx: usize) -> bool {
        self.current_analysis().map_or(false, |r| r.dead_func.dead.contains(&crate::pass::Analyzer::FuncId(decl_idx as u32)))
    }

    /// Query whether an expression is an inline-candidate call site.
    /// Returns the callee's `FuncId`; the `IrBuilder` should expand its body rather than launch
    /// a subgraph.
    #[inline]
    pub(super) fn inline_target(&self, expr_id: crate::ast::Ast::ExprId) -> Option<crate::pass::Analyzer::FuncId> {
        let report = self.current_analysis()?;
        report.inline.expansions.get(&expr_id).copied()
    }

    /// Query whether an expression is marked for stack allocation.
    #[inline]
    pub(super) fn should_stack_alloc(&self, expr_id: crate::ast::Ast::ExprId) -> bool {
        self.current_analysis().map_or(false, |r| r.stack_alloc.candidates.contains(&expr_id))
    }

    /// Return the module currently being compiled (builtin takes priority, otherwise the user
    /// module).
    pub(super) fn current_module(&self) -> &'a crate::ast::Ast::Module<'a> {
        self.compiling_builtin.unwrap_or(self.module)
    }

    /// Enter a new scope (variables and type fields are pushed together).
    pub(super) fn enter_scope(&mut self) {
        self.scope_stack.push(rustc_hash::FxHashMap::default());
        self.type_scope_stack.push(rustc_hash::FxHashMap::default());
        self.cell_bound.push(rustc_hash::FxHashMap::default());
    }

    /// Exit a scope (variables and type fields are popped together).
    pub(super) fn exit_scope(&mut self) {
        self.scope_stack.pop();
        self.type_scope_stack.pop();
        self.cell_bound.pop();
    }

    /// Register type field info in the current scope (constructor name / type name ->
    /// `TypeFieldInfo`).
    pub(super) fn bind_type_fields(&mut self, name: &str, info: TypeFieldInfo) {
        if let Some(scope) = self.type_scope_stack.last_mut() {
            scope.insert(name.to_string(), info);
        }
    }

    /// Register type field info into the BASE layer (index 0) without
    /// displacing an existing binding — std modules bind first, so user
    /// types only fill free bare keys (std keeps priority for cross-module
    /// bare references). Canonical keys are unique and always land.
    pub(super) fn bind_type_fields_base_first_wins(&mut self, name: &str, info: TypeFieldInfo) {
        if self.type_scope_stack.is_empty() {
            return;
        }
        self.type_scope_stack[0].entry(name.to_string()).or_insert(info);
    }

    /// Bind a user-module type/ctor binding into that module's dedicated
    /// scope (shadow layer above the base).
    pub(super) fn bind_module_type_fields(&mut self, module_name: &str, name: &str, info: TypeFieldInfo) {
        self.module_type_scopes
            .entry(module_name.to_string())
            .or_default()
            .insert(name.to_string(), info);
    }

    /// Look up type field info by walking the scope stack from inner to outer
    /// (constructor name or type name). The per-module scope of the module
    /// currently being compiled sits between the local (nested-type) layers
    /// and the base layer — own-module types shadow the flat base.
    pub(super) fn lookup_type_fields(&self, name: &str) -> Option<TypeFieldInfo> {
        // Local layers above the base (index 0), innermost first.
        if self.type_scope_stack.len() > 1 {
            for scope in self.type_scope_stack[1..].iter().rev() {
                if let Some(info) = scope.get(name) {
                    return Some(info.clone());
                }
            }
        }
        // The current module's own scope.
        if let Some(scope) = self.module_type_scopes.get(self.current_module().name) {
            if let Some(info) = scope.get(name) {
                return Some(info.clone());
            }
        }
        // Base layer.
        self.type_scope_stack.first()?.get(name).cloned()
    }

    /// Bind a variable name to a NodeId (in the current scope).
    pub(super) fn declare_var(&mut self, name: &str, node_id: NodeId) {
        self.bind_var(name, node_id);
    }

    pub(super) fn bind_var(&mut self, name: &str, node_id: NodeId) {
        if let Some(scope) = self.scope_stack.last_mut() {
            scope.insert(name.to_string(), node_id);
        }
    }

    /// Look up the NodeId bound to a variable (searching from inner to outer scope).
    pub(super) fn lookup_var(&self, name: &str) -> Option<NodeId> {
        for scope in self.scope_stack.iter().rev() {
            if let Some(&node_id) = scope.get(name) {
                return Some(node_id);
            }
        }
        // Global variables: return None; the caller handles them via is_global_var + global_var_slots.
        None
    }

    /// Look up the capture list recorded by Sema for a nested scope (lambda /
    /// defer / nested function). `scope_expr_id` is the entry expression's
    /// ExprId (the Lambda expr, the defer body expr, or the nested-fun body).
    /// Returns an empty slice when no captures are recorded (the scope captures
    /// nothing, or capture data is unavailable in instantiation mode).
    ///
    /// This is the single source of truth for captures, replacing the builder's
    /// own `collect_free_idents_expr` re-scan.
    pub(super) fn lookup_captures(&self, scope_expr_id: crate::ast::Ast::ExprId) -> &[crate::sema::Sema::CaptureInfo] {
        let module_name = self.current_module().name;
        let key = crate::sema::Sema::module_expr_key(module_name, scope_expr_id.0 as u64);
        self.sema.get_captures(key)
    }

    /// Check whether a name is a global variable and return its slot index.
    pub(super) fn lookup_global_var(&self, name: &str) -> Option<u32> {
        self.global_var_slots.get(name).copied()
    }

    /// Compile a global-variable load node (`compute_global_load`, idx 270).
    /// Takes no input; at runtime it reads from `global_var_storage[slot]`.
    pub(super) fn compile_global_load(&mut self, slot: u32) -> NodeId {
        // Append `current_effect` as an implicit dependency input to ensure the load executes
        // only after prior `global_store` operations complete.
        // `compute_global_load` does not read the input value; this input exists solely for
        // scheduler ordering.
        let (input_count, inputs_offset) = match self.current_effect {
            Some(eff) => (1, self.graph.inputs_pool.push(&[eff])),
            None => (0, self.graph.inputs_pool.push(&[])),
        };
        let node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count,
            inputs_offset,
            compute_fn: CF_GLOBAL_LOAD,
        });
        self.graph.set_global_load_slot(node, slot);
        node
    }

    /// Compile a global-variable store node (`compute_global_store`, idx 271).
    /// `inputs[0]` is the value-source node; at runtime it writes to
    /// `global_var_storage[slot]`.
    ///
    /// `current_effect` is appended as an implicit ordering input, mirroring
    /// `compile_global_load`: stores must not overtake prior effects, and
    /// global stores are chained serially. Without this edge the entry-injected
    /// module initializers are mutually unordered — a body load (which deps only
    /// the LAST init store) could fire while an earlier call-initialized store
    /// (e.g. a builtin C call, whose chain is 2+ hops) is still in flight and
    /// read the unwritten slot as NULL.
    pub(super) fn compile_global_store(&mut self, val_node: NodeId, slot: u32) -> NodeId {
        let (input_count, inputs_offset) = match self.current_effect {
            Some(eff) => (2, self.graph.inputs_pool.push(&[val_node, eff])),
            None => (1, self.graph.inputs_pool.push(&[val_node])),
        };
        let node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count,
            inputs_offset,
            compute_fn: CF_GLOBAL_STORE,
        });
        self.graph.set_global_store_slot(node, slot);
        node
    }

    /// Determine whether a NodeId falls within the current subgraph's range (i.e. is not an
    /// outer variable).
    pub(super) fn is_in_current_subgraph(&self, node: NodeId) -> bool {
        node.0 >= self.current_sg_start
    }

    /// Guard for cell-backed name traffic (reads/writes/`&name`): the site
    /// must not be inside a lambda body (`captured_scopes` non-empty —
    /// outer-name reads there route through the capture machinery) and the
    /// name must have a cell binding visible from the current scope, owned by
    /// the function being compiled (the owner tag blocks cross-function
    /// leakage through the shared scope chain).
    pub(super) fn lookup_cell_binding(&self, name: &str) -> Option<NodeId> {
        // NOTE: no captured_scopes guard — the binding-identity check below
        // already routes correctly inside lambda bodies: a cell-captured name
        // resolves to the lambda's UPVALUE node (registered in cell_bound by
        // compile_lambda), while a plain outer name resolves to its upvalue
        // and fails the identity check against the outer cell entry.
        let owner = self.current_function_id;
        let cell_node = self
            .cell_bound
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
            .filter(|(_, o)| *o == owner)
            .map(|(node, _)| *node)?;
        // Binding-identity guard: the name's scope_stack resolution must BE
        // the cell node. A plain rebind of the same name (inline-expansion
        // params binding formals via bind_var, pattern variables, for-vars)
        // shadows the cell WITHOUT touching cell_bound — routing by name
        // alone would read the wrong storage (observed: inlined callee param
        // colliding with a cell-backed caller param of the same name).
        if self.lookup_var(name) != Some(cell_node) {
            return None;
        }
        Some(cell_node)
    }

    /// Register a cell-backed binding in the current scope and rebind the
    /// name in `scope_stack` to the cell node.
    pub(super) fn bind_cell(&mut self, name: &str, cell_node: NodeId) {
        let owner = self.current_function_id;
        if let Some(scope) = self.cell_bound.last_mut() {
            scope.insert(name.to_string(), (cell_node, owner));
        }
        self.bind_var(name, cell_node);
    }

    /// Decl-site cell-backing eligibility (place model):
    /// - `var` decls in all-vars functions (C1-③④): EVERY binding (any value
    ///   type — scalars AND containers/records; `arr[i] = x` reads the cell
    ///   for the current Arc and mutates in place, unchanged), not captured
    ///   by a lambda / nested fn, not declared inside a lambda body.
    /// - `val` decls: only when address-taken (C1-① shape).
    pub(super) fn decl_cell_backing_eligible(
        &self,
        name: &str,
        _value_expr: crate::ast::Ast::ExprId,
        is_var: bool,
    ) -> bool {
        // ⑤ Lambda-body LOCAL vars are slot-backed too (④ removed the
        // captured_scopes routing block — the binding-identity guard routes
        // correctly inside lambda bodies). Nested-lambda captures ALSO stay
        // slot-backed: ④'s cell-capture chain propagates the shared cell
        // through upvalue levels (the outer lambda cell-captures, the inner
        // registers against the outer's upvalue).
        if is_var {
            self.fn_all_vars_slot || self.fn_address_taken.contains(name)
        } else {
            self.fn_address_taken.contains(name)
        }
    }

    /// Record the value a cell-backed binding holds right after its cell
    /// allocation, and mark address-taken cells non-forwardable.
    pub(super) fn track_cell_decl(&mut self, name: &str, cell_node: NodeId, value_node: NodeId) {
        // Cell-captured by a lambda: like address-taken, the cell escapes —
        // calls through the closure can mutate it invisibly to the compiler,
        // so reads must always LOAD (a forwarded read would pin the
        // compile-time value).
        if self.fn_address_taken.contains(name) || self.fn_lambda_captured.contains(name) {
            self.no_forward_cells.insert(cell_node);
        } else {
            self.cell_values.insert(cell_node, value_node);
        }
    }

    /// A store to a tracked cell updates the forwarding memory.
    pub(super) fn track_cell_store(&mut self, cell_node: NodeId, value_node: NodeId) {
        if !self.no_forward_cells.contains(&cell_node) {
            self.cell_values.insert(cell_node, value_node);
        }
    }

    /// Forwarding lookup for a read of `cell_node`: the remembered current
    /// value node, when the cell is tracked.
    pub(super) fn cell_forwarded_value(&self, cell_node: NodeId) -> Option<NodeId> {
        if self.no_forward_cells.contains(&cell_node) {
            return None;
        }
        self.cell_values.get(&cell_node).copied()
    }

    /// Barrier (subgraph body, branch- AND loop-like): snapshot the
    /// forwarding memory and clear it. Correctness rule: a body that
    /// re-executes (loop) or executes conditionally (branch) invalidates
    /// compile-time values BOTH ways — pre-entry values are stale after it,
    /// and stores made inside it must not leak to code after it. Reads
    /// inside and after the body emit real loads (stores still forward
    /// WITHIN one body execution).
    pub(super) fn cell_barrier_enter(&mut self) -> rustc_hash::FxHashMap<NodeId, NodeId> {
        std::mem::take(&mut self.cell_values)
    }

    /// Defer-body barrier exit: RESTORE the pre-entry snapshot. The defer
    /// body runs at exit time, so the enclosing function's subsequent code
    /// still sees pre-defer values (the defer's stores must not leak).
    pub(super) fn cell_barrier_exit_defer(
        &mut self,
        saved: rustc_hash::FxHashMap<NodeId, NodeId>,
    ) {
        self.cell_values = saved;
    }

    /// Subgraph-body barrier exit (branch- AND loop-like): clear EVERYTHING —
    /// neither pre-body values (the body may have overwritten the cell) nor
    /// the body's own stores (their nodes live inside the subgraph range the
    /// parent frame never executes — forwarding them leaves reads pending
    /// forever) may survive. Reads after the body emit real loads.
    pub(super) fn cell_barrier_exit(&mut self) {
        self.cell_values.clear();
    }

    /// True when `recv.field` is the `Type.Ctor` qualified-constructor form —
    /// constructors are rvalues, not places (defense in depth behind the sema
    /// place check).
    pub(super) fn is_qualified_ctor_place(
        &self,
        recv_id: crate::ast::Ast::ExprId,
        field: &str,
    ) -> bool {
        if let crate::ast::Ast::Expr::Ident(type_name) =
            &self.current_module().arena.expr(recv_id).node
        {
            return self.check_qualified_ctor_ir(type_name, field).is_some();
        }
        false
    }

    pub(super) fn compound_assign_op_to_compute_fn(
        &mut self,
        op: crate::ast::Ast::CompoundAssignOp,
        target_expr: crate::ast::Ast::ExprId,
    ) -> ComputeFnId {
        use crate::ast::Ast::CompoundAssignOp;
        let ty = self.expr_type_name_checked(target_expr, "compound_assign_op");
        let is_float = crate::value::ValueTag::from_name(ty).and_then(scalar_meta).map(|m| m.is_float).unwrap_or(false);
        let base = Self::arith_base(ty).unwrap_or(CF_ADD_I32_FULL.0); // fallback: i32
        // Integer offsets: add(0) sub(1) mul(2) div(3) mod(4) bitand(5) bitor(6) bitxor(7) shl(8) shr(9)
        // Float offsets:   add(0) sub(1) mul(2) div(3) mod(4) neg(5)
        let offset = match op {
            CompoundAssignOp::AddAssign => 0,
            CompoundAssignOp::SubAssign => 1,
            CompoundAssignOp::MulAssign => 2,
            CompoundAssignOp::DivAssign => 3,
            CompoundAssignOp::ModAssign => 4,
            // Bitwise ops are integers-only; a float reaching here means sema failed to intercept,
            // so fall back to the i32 path.
            CompoundAssignOp::BitAndAssign if !is_float => 5,
            CompoundAssignOp::BitOrAssign if !is_float => 6,
            CompoundAssignOp::BitXorAssign if !is_float => 7,
            CompoundAssignOp::ShlAssign if !is_float => 8,
            CompoundAssignOp::ShrAssign if !is_float => 9,
            // Floats must not perform bitwise ops; if one appears, fall back to noop.
            _ => return CF_NOOP,
        };
        ComputeFnId(base + offset)
    }

    /// Register a placeholder subgraph (node range is filled in after compilation).
    pub fn register_subgraph_placeholder(
        &mut self,
        _name: &str,
        param_count: u8,
        is_async: bool,
    ) -> SubGraphId {
        let id = SubGraphId(self.graph.subgraphs.len() as u32);
        let sg = SubGraph {
            converter_generated: false,
            id,
            node_range: (NodeId(0), NodeId(0)),
            param_count,
            entry_node: NodeId(0),
            return_node: NodeId(0),
            has_suspend: is_async,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: id.0,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        };
        self.graph.subgraphs.push(sg);
        id
    }

    /// Compile an expression into a Node and return its NodeId.
    pub fn compile_expr(&mut self, expr_id: crate::ast::Ast::ExprId) -> NodeId {
        let spanned = self.current_module().arena.expr(expr_id);
        let expr = &spanned.node;
        match expr {
            // Constants
            crate::ast::Ast::Expr::IntLit { .. }
            | crate::ast::Ast::Expr::FloatLit { .. }
            | crate::ast::Ast::Expr::BoolLit(_)
            | crate::ast::Ast::Expr::CharLit(_)
            | crate::ast::Ast::Expr::StrLit(_)
            | crate::ast::Ast::Expr::NullLit
            | crate::ast::Ast::Expr::VoidLit => self.compile_const_with_value(expr_id),

            // Variable reference
            crate::ast::Ast::Expr::Ident(name) => self.compile_ident(expr_id, name),

            // Binary operations
            crate::ast::Ast::Expr::Binary { op, lhs, rhs } => {
                self.compile_binary(*op, expr_id, *lhs, *rhs)
            }

            // Function call
            crate::ast::Ast::Expr::Call { callee, args, type_args: _ } => {
                // Implicit this: bare call resolved to an instance method by sema.
                // Sema marks the callee Ident with `implicit_this = Method(name)`; synthesize an
                // explicit `this.method(args)` dispatch using the already-bound `this` node.
                if let Some(access) = self.expr_implicit_this(*callee).cloned() {
                    if let crate::sema::Sema::ImplicitThisAccess::Method(method) = access {
                        let this_node = self
                            .lookup_var("this")
                            .expect("this binding must exist in method body");
                        return self.compile_method_call(expr_id, *callee, &method, args, Some(this_node));
                    }
                }
                self.compile_call(expr_id, *callee, args)
            }
            crate::ast::Ast::Expr::MethodCall { recv, method, args, .. } => {
                self.compile_method_call(expr_id, *recv, method, args, None)
            }

            // Field access
            crate::ast::Ast::Expr::FieldAccess { recv, field } => {
                self.compile_field_access(expr_id, *recv, field)
            }
            // Safe field access `recv?.field`: compiled as a normal field access + safe flag.
            crate::ast::Ast::Expr::SafeAccess { recv, field } => {
                let node = self.compile_field_access(expr_id, *recv, field);
                self.graph.set_safe_op(node);
                node
            }
            crate::ast::Ast::Expr::Index { recv, index } => self.compile_index(*recv, *index),

            // Block expression
            crate::ast::Ast::Expr::Block { stmts, trailing } => self.compile_block(stmts, trailing),

            // If expression -> Gate node + branch subgraphs
            crate::ast::Ast::Expr::If {
                cond,
                then_branch,
                else_branch,
            } => self.compile_if(*cond, *then_branch, *else_branch),

            // Match expression -> Gate chain
            crate::ast::Ast::Expr::Match { scrutinee, arms } => {
                self.compile_match(*scrutinee, arms)
            }

            // Record construction
            crate::ast::Ast::Expr::RecordLit(fields) => self.compile_record_lit(expr_id, fields),

            // Lambda expression -> closure subgraph + closure construct node
            crate::ast::Ast::Expr::Lambda { params, body, is_async, .. } => {
                let body_expr = match body {
                    crate::ast::Ast::LambdaBody::Block(e) | crate::ast::Ast::LambdaBody::Expression(e) => *e,
                };
                self.compile_lambda(params, body_expr, *is_async, None, Some(expr_id))
            }

            // Array construction
            crate::ast::Ast::Expr::ArrayLit { elements, fill } => {
                self.compile_array_lit(expr_id, elements, *fill)
            }

            // Assignment expression: `target = value`.
            // Used for assignments in expression contexts such as defer bodies.
            // Consistent with the `Ident` logic of `Stmt::Assignment`:
            //   captured variable -> WriteBack; outer variable -> WriteBack;
            //   global variable -> global_store; local -> bind_var.
            crate::ast::Ast::Expr::Assign { target, value } => {
                self.compile_assign(*target, *value)
            }

            // Compound assignment: `target op= value`
            crate::ast::Ast::Expr::CompoundAssign { op, target, value } => {
                self.compile_compound_assign(*op, *target, *value)
            }

            // select expression -> Gate node (compute_select_gate) + an independent subgraph per branch.
            crate::ast::Ast::Expr::Select(arms) => self.compile_select(arms),

            // `?` operator (Propagate): unwraps a Throw; on Err, returns early.
            crate::ast::Ast::Expr::Propagate(inner) => {
                let inner_node = self.compile_subexpr(*inner);
                let inputs_offset = self.graph.inputs_pool.push(&[inner_node]);
                let mut n = self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: 1,
                    inputs_offset,
                    compute_fn: CF_PROPAGATE, // compute_propagate
                });
                // Bug #97: at the tail of a function whose return type is Throw, the
                // propagate node's UNWRAPPED value would leak out as the function's
                // return value (callers matching Ok/Err then hit the fallback panic).
                // Re-wrap with Ok: the Err path has already exited via the Return
                // control signal, so only the Ok path flows through this node.
                if self.in_tail_position && self.fn_returns_throw {
                    let off = self.graph.inputs_pool.push(&[n]);
                    n = self.graph.add_node(Node {
                        kind: NodeKind::Call,
                        input_count: 1,
                        inputs_offset: off,
                        compute_fn: CF_THROW_OK, // compute_throw_ok
                    });
                }                n
            }

            // Unary operations: `!` (logical not), `-` (arithmetic negation), `~` (bitwise not).
            crate::ast::Ast::Expr::Unary { op, operand } => {
                let operand_node = self.compile_subexpr(*operand);
                let inputs_offset = self.graph.inputs_pool.push(&[operand_node]);
                let compute_fn = self.select_unary_compute_fn(*op, *operand);
                let node = self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: 1,
                    inputs_offset,
                    compute_fn,
                });
                // Compile-time SIMD batching tag: Neg/BitNot + scalar type.
                if let Some(info) = self.unary_batch_info(*op, *operand) {
                    self.graph.set_batch_info(node, info);
                }
                node
            }

            // Type cast `expr as T`: dispatch to the same codegen paths as the former cast syntax.
            crate::ast::Ast::Expr::As { expr, target } => {
                self.compile_as_cast(*expr, *target)
            }

            // String interpolation: `"text {expr} more {expr}"` -> chained `str_concat`.
            crate::ast::Ast::Expr::StrInterp(parts) => {
                self.compile_str_interp(parts)
            }

            // Take a reference `&expr` -> `compute_ref_of` (280): scalars are boxed into a Cell;
            // heap objects share an Arc.
            crate::ast::Ast::Expr::RefOf(inner) => {
                let inner_ast = &self.current_module().arena.expr(*inner).node;
                // ── Place references (place model B-stage) ──
                // `&arr[i]` → ArrayElemRef (live element location, SoA-aware).
                if let crate::ast::Ast::Expr::Index { recv, index } = inner_ast {
                    let recv_node = self.compile_subexpr(*recv);
                    let idx_node = self.compile_subexpr(*index);
                    let inputs_offset = self.graph.inputs_pool.push(&[recv_node, idx_node]);
                    return self.graph.add_node(Node {
                        kind: NodeKind::UnOp,
                        input_count: 2,
                        inputs_offset,
                        compute_fn: CF_REF_OF,
                    });
                }
                // `&rec.field` → RecordFieldRef (1 input + field name side
                // entry). Implicit-this `&field` inside methods resolves to
                // `&this.field`. Qualified constructors (`Type.Ctor`) are NOT
                // places — leave them to the normal path (sema rejects them).
                let field_place: Option<(crate::ast::Ast::ExprId, &str)> = match inner_ast {
                    crate::ast::Ast::Expr::FieldAccess { recv, field } => Some((*recv, field)),
                    _ => None,
                };
                if let Some((recv_id, field)) = field_place {
                    if !self.is_qualified_ctor_place(recv_id, field) {
                        let recv_node = self.compile_subexpr(recv_id);
                        let inputs_offset = self.graph.inputs_pool.push(&[recv_node]);
                        let ref_node = self.graph.add_node(Node {
                            kind: NodeKind::UnOp,
                            input_count: 1,
                            inputs_offset,
                            compute_fn: CF_REF_OF,
                        });
                        self.graph.set_field_set_name(ref_node, field.to_string());
                        return ref_node;
                    }
                }
                if let crate::ast::Ast::Expr::Ident(_) = inner_ast {
                    if let Some(crate::sema::Sema::ImplicitThisAccess::Field(field)) =
                        self.expr_implicit_this(*inner).cloned()
                    {
                        let field_name = field.to_string();
                        let this_node = self
                            .lookup_var("this")
                            .expect("this binding must exist in method body");
                        let inputs_offset = self.graph.inputs_pool.push(&[this_node]);
                        let ref_node = self.graph.add_node(Node {
                            kind: NodeKind::UnOp,
                            input_count: 1,
                            inputs_offset,
                            compute_fn: CF_REF_OF,
                        });
                        self.graph.set_field_set_name(ref_node, field_name);
                        return ref_node;
                    }
                }
                // `&global` → GlobalSlotRef (0 inputs + slot side entry) — a
                // LIVE reference to the global slot, not a snapshot. Only
                // when the name is NOT a local (locals take the Cell path).
                if let crate::ast::Ast::Expr::Ident(name) = inner_ast {
                    if self.lookup_var(name).is_none() {
                        if let Some(slot) = self.lookup_global_var(name) {
                            let inputs_offset = self.graph.inputs_pool.push(&[]);
                            let ref_node = self.graph.add_node(Node {
                                kind: NodeKind::UnOp,
                                input_count: 0,
                                inputs_offset,
                                compute_fn: CF_REF_OF,
                            });
                            self.graph.set_global_load_slot(ref_node, slot);
                            return ref_node;
                        }
                    }
                }
                // ── Binding references (locals/params; C1-① cell backing) ──
                // Cell-backed binding: the decl site (or a previous `&x`)
                // already allocated THE cell — the reference is that node
                // itself, zero cost, and all aliases share one storage.
                if let crate::ast::Ast::Expr::Ident(name) =
                    &self.current_module().arena.expr(*inner).node
                {
                    if let Some(cell_node) = self.lookup_cell_binding(name) {
                        return cell_node;
                    }
                }
                let inner_node = self.compile_subexpr(*inner);
                let inputs_offset = self.graph.inputs_pool.push(&[inner_node]);
                let ref_node = self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: 1,
                    inputs_offset,
                    compute_fn: CF_REF_OF,
                });
                // LAZY cell backing: the first `&x` of a binding that did not
                // get decl-site backing (a scalar PARAM — param slotting is
                // off — or a local the pre-pass skipped in a
                // strategy-transformed function). SCALARS ONLY: compute_ref_of
                // SHARES heap-object Arcs (`&rec` object semantics), so a
                // lazy "cell" for a container binding would be the Arc itself
                // and name stores through it would silently no-op.
                // Registers the cell so later name reads/writes route through
                // it; the name is NOT value-tracked (forwarding is disabled:
                // `val r = &x` binds a COPY of the cell node, so deref stores
                // through `r` cannot be mapped back to the cell at compile
                // time — a stale forward would be worse than a load).
                if self.captured_scopes.is_empty() {
                    if let crate::ast::Ast::Expr::Ident(name) =
                        &self.current_module().arena.expr(*inner).node
                    {
                        let eligible = self.lookup_var(name)
                            .map(|b| self.is_in_current_subgraph(b))
                            .unwrap_or(false)
                            && self.expr_type_is_scalar(*inner);
                        if eligible {
                            self.bind_cell(name, ref_node);
                            self.no_forward_cells.insert(ref_node);
                        }
                    }
                }
                ref_node
            }

            // Dereference read `*ref` -> `compute_deref_read` (281): returns the inner value for a
            // Cell; other Ref types pass through.
            crate::ast::Ast::Expr::Deref(inner) => {
                let inner_node = self.compile_subexpr(*inner);
                let inputs_offset = self.graph.inputs_pool.push(&[inner_node]);
                self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: 1,
                    inputs_offset,
                    compute_fn: CF_DEREF_READ,
                })
            }

            // Non-null assertion `expr!` -> `compute_non_null_assert` (279): panics on Null;
            // non-Null passes through.
            crate::ast::Ast::Expr::NonNullAssert(inner) => {
                let inner_node = self.compile_subexpr(*inner);
                let inputs_offset = self.graph.inputs_pool.push(&[inner_node]);
                self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: 1,
                    inputs_offset,
                    compute_fn: CF_NON_NULL_ASSERT,
                })
            }

            // Elvis: `lhs ?: rhs` -> `compute_elvis` (idx 265).
            crate::ast::Ast::Expr::Elvis { lhs, rhs } => {
                let lhs_node = self.compile_subexpr(*lhs);
                let rhs_node = self.compile_subexpr(*rhs);
                let inputs_offset = self.graph.inputs_pool.push(&[lhs_node, rhs_node]);
                self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset,
                    compute_fn: CF_ELVIS, // compute_elvis
                })
            }

            // Safe method call `recv?.method(args)`: compiled as a normal method call + safe flag.
            crate::ast::Ast::Expr::SafeMethodCall { recv, method, args, .. } => {
                let node = self.compile_method_call(expr_id, *recv, method, args, None);
                self.graph.set_safe_op(node);
                node
            }

            // Record extension `(...base, field: value, ...)` -> base + updates input nodes +
            // RecordExtendInfo.
            crate::ast::Ast::Expr::RecordExtend { base, updates } => {
                self.compile_record_extend(*base, updates)
            }

            // Atomic construction `atomic expr` -> single-input node wrapping into an AtomicValue.
            crate::ast::Ast::Expr::Atomic(operand) => self.compile_atomic(*operand),

            // inline_trait expression -> per-method compiled subgraph + TraitValue construct node.
            crate::ast::Ast::Expr::InlineTrait(methods) => self.compile_inline_trait(expr_id, methods),

            // lazy expression -> thunk subgraph + LazyValue construct node.
            crate::ast::Ast::Expr::Lazy(operand) => self.compile_lazy(expr_id, *operand),

            // Slice `recv[start..end]` / `recv[start..=end]` -> three-input node + inclusive flag.
            crate::ast::Ast::Expr::Slice { recv, start, end, inclusive } => {
                self.compile_slice(*recv, *start, *end, *inclusive)
            }
        }
    }

    /// Return the module name used for the expr_types composite key.
    ///
    /// In a monomorphization-instance context, the function body expressions belong to the module
    /// of the callee function; the expr_types key must use the instance's module_name (not the call-site
    /// module name), otherwise cross-module generic calls fail type lookup (e.g. when Math.abs calls
    /// cast(x).to(i32), source_ty resolves to void).
    /// Bug #97 helper: does this (possibly Async-wrapped) type denote a Throw return?
    /// Guards against invalid/placeholder handles (u32::MAX) seen on some sig entries.
    pub(super) fn handle_returns_throw(&self, t0: crate::sema::Sema::TypeHandle) -> bool {
        let mut t = t0;
        for _ in 0..3 {
            if (t.0 as usize) >= self.type_arena.len() {
                return false;
            }
            match self.type_arena.get(t) {
                crate::sema::Sema::Type::Fn(id) | crate::sema::Sema::Type::Async(id) => {
                    // Placeholder detail ids (see concretize_type's Generic branch
                    // note) can no longer occur for checked signatures; the guard
                    // stays as defense in depth for predeclared-but-never-checked
                    // shapes.
                    if (id.0 as usize) >= self.type_arena.details_len() {
                        return false;
                    }
                    t = match self.type_arena.get(t) {
                        crate::sema::Sema::Type::Fn(_) => self.type_arena.fn_parts(t).1,
                        _ => self.type_arena.async_value(t),
                    };
                }
                crate::sema::Sema::Type::Throw(_) => return true,
                _ => return false,
            }
        }
        false
    }

    pub(super) fn expr_key_module(&self) -> &'a str {
        if let Some(inst_id) = self.current_instance_id {
            if let Some(inst) = self.sema.monomorph_instances.get(inst_id as usize) {
                return &*inst.module_name;
            }
        }
        self.current_module().name
    }

    /// Look up an expression's type name (from Sema).
    ///
    /// Prefer ExprInfo.type_name (adt/generic scenarios); fall back to "unknown" when no record exists.
    /// When Sema has no record (self in a specialized trait default method version), look up
    /// sema's TraitDefaultInstance.type_name to get self's concrete implementation type.
    pub(super) fn expr_type_name(&self, expr_id: crate::ast::Ast::ExprId) -> Option<&str> {
        // self in a specialized trait default method version: consume sema's TraitDefaultInstance.type_name.
        // When sema infers a trait default method body, self is the abstract ThisType; the specialized
        // instance records the concrete implementation type name.
        // The IR indexes sema output via current_trait_default_idx; it does not hold the type-name string.
        if let Some(idx) = self.current_trait_default_idx {
            if let crate::ast::Ast::Expr::Ident(name) = &self.module.arena.expr(expr_id).node {
                if *name == "this" {
                    if let Some(inst) = self.sema.trait_default_instances.get(idx) {
                        return Some(inst.type_name.as_ref());
                    }
                }
            }
        }
        let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), expr_id.0 as u64);
        // Instance context: prefer the instance-local expr_types (type parameters replaced with concrete types)
        if let Some(inst_id) = self.current_instance_id {
            if let Some(inst) = self.sema.monomorph_instances.get(inst_id as usize) {
                if let Some(info) = inst.expr_types.get(&key) {
                    return Some(
                        info.type_name
                            .as_deref()
                            .map(|n| n)
                            .unwrap_or_else(|| self.type_arena.get(info.ty).source_name()),
                    );
                }
            }
        }
        // Global expr_types fallback
        if let Some(info) = self.sema.expr_types.get(&key) {
            return Some(
                info.type_name
                    .as_deref()
                    .map(|n| n)
                    .unwrap_or_else(|| self.type_arena.get(info.ty).source_name()),
            );
        }
        None
    }

    /// Look up an expression's inferred TypeHandle (from Sema).
    ///
    /// Companion of `expr_type_name` with the same instance-local → global
    /// fallback order; used where the concrete type structure matters (e.g.
    /// extracting `ForeignFn[R]`'s `R` for Lib.lookup lowering).
    pub(super) fn expr_type_handle(&self, expr_id: crate::ast::Ast::ExprId) -> Option<crate::types::TypeHandle> {
        let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), expr_id.0 as u64);
        if let Some(inst_id) = self.current_instance_id {
            if let Some(inst) = self.sema.monomorph_instances.get(inst_id as usize) {
                if let Some(info) = inst.expr_types.get(&key) {
                    return Some(info.ty);
                }
            }
        }
        self.sema.expr_types.get(&key).map(|info| info.ty)
    }

    /// Look up the implicit-this access kind for an expression (set by sema).
    ///
    /// Sema records on `ExprInfo.implicit_this` whether a bare identifier/call inside a method
    /// body resolved to an instance field or method (i.e. an implicit `this.` access). The IR
    /// builder consumes this marker to synthesize the explicit `this`-based access nodes.
    pub(super) fn expr_implicit_this(
        &self,
        expr_id: crate::ast::Ast::ExprId,
    ) -> Option<&crate::sema::Sema::ImplicitThisAccess> {
        let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), expr_id.0 as u64);
        if let Some(inst_id) = self.current_instance_id {
            if let Some(inst) = self.sema.monomorph_instances.get(inst_id as usize) {
                if let Some(info) = inst.expr_types.get(&key) {
                    return info.implicit_this.as_ref();
                }
            }
        }
        self.sema
            .expr_types
            .get(&key)
            .and_then(|info| info.implicit_this.as_ref())
    }

    /// Checked version of `expr_type_name`: the sema contract guarantees ExprInfo is registered.
    /// A missing entry indicates a sema inference omission -- report a compile error (not silent),
    /// and use "i32" as a placeholder to continue compiling and surface further errors.
    #[inline]
    pub(super) fn expr_type_name_checked(&mut self, expr_id: crate::ast::Ast::ExprId, context: &str) -> &str {
        let has_type = self.expr_type_name(expr_id).is_some();
        if has_type {
            return self.expr_type_name(expr_id).unwrap();
        }
        self.errors.push(format!(
            "internal: missing ExprInfo for expr {:?} in {}", expr_id, context));
        "i32"
    }

    /// Determine whether an expression is a nullable type (Type::Nullable).
    /// Nullable types' ==/!= need null-discriminant comparison: ?. short-circuit or null literals
    /// produce Value::Null, and dedicated comparison functions for str/i32 etc. do not handle Null,
    /// yielding wrong results.
    /// Dispatch to CF_EQ_OBJ/CF_NE_OBJ (value_equals_with_arena correctly handles Null).
    pub(super) fn expr_is_nullable(&self, expr_id: crate::ast::Ast::ExprId) -> bool {
        let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), expr_id.0 as u64);
        self.sema
            .expr_types
            .get(&key)
            .map(|info| matches!(self.type_arena.get(info.ty), crate::sema::Sema::Type::Nullable(_)))
            .unwrap_or(false)
    }

    /// Single-point function-name resolution: EVERY call-site binding goes
    /// through here (compile_call, module-qualified Path 0, partial
    /// application, make_call_by_name, free-function method fallback). One
    /// authority, one precedence order — lexical scope first, global bare
    /// last:
    ///
    ///   1. generic instance key `<name>#<inst_id>` when provided — Sema
    ///      already chose this monomorphization for the call site; it must
    ///      outrank the generic declaration's own keys (the unmonomorphized
    ///      body has unresolved type parameters and evaluates to void);
    ///   2. current-module mangled key `<cur_mod_path>.<name>` (std modules
    ///      have no bare slot, so their intra-module calls resolve here);
    ///   3. recv short-qualified key `<Recv>.<name>` — the `File.remove(...)`
    ///      shape. An EXPLICIT qualifier outranks ambient package visibility:
    ///      `Instant.now()` inside std.time must bind Instant.now, not the
    ///      package key `std.time::now` (contested by SystemTime.now — the
    ///      silent wrong-callee the tripwire exposed in Timer.frond);
    ///   4. package-scoped key `<cur_pkg>::<name>` (stdlib sibling files
    ///      calling each other bare within one package directory);
    ///   5. bare name — builtin (globally visible by design) and user/dep
    ///      modules only; std never registers bare.
    ///
    /// Orders 3–5 consult the collision tripwire: a key contested by two
    /// distinct functions is a HARD error listing the candidates — never
    /// first/last-writer-wins.
    ///
    /// Returns `Some(Ok(sg))` on success, `Some(Err(diagnostic))` on an
    /// ambiguous key, `None` when nothing matched.
    pub(super) fn resolve_func(
        &self,
        site: &str,
        name: &str,
        inst_mangled: Option<&str>,
        recv: Option<&str>,
    ) -> Option<Result<SubGraphId, String>> {
        // @internal guard — runs BEFORE any key lookup so the deny is uniform
        // across all five binding shapes (bare / mangled / recv-qualified /
        // package / generic instance).
        if self.internal_access_blocked(name) {
            return Some(Err(self.internal_access_diag(name)));
        }
        let cur_path = crate::sema::Sema::module_logical_path(self.current_module().name);
        // 1. generic instance (the sema-chosen monomorphization)
        if let Some(m) = inst_mangled {
            if let Some(&sg) = self.func_subgraphs.get(m) {
                self.log_call_bind(site, name, recv, m, sg);
                return Some(Ok(sg));
            }
        }
        // 2. recv short-qualified (`F64.parse`) — an EXPLICIT qualifier outranks
        //    every ambient key. This must run BEFORE the current-module mangled
        //    probe: while compiling std/core/F32.frond, a call to F64.parse(s)
        //    (name="parse", recv="F64") used to hit "std.core.F32.parse" first —
        //    binding the call to the CALLING function itself (infinite
        //    self-recursion, scheduler deadlock).
        if let Some(rn) = recv {
            let key = format!("{}.{}", rn, name);
            // func_subgraphs FIRST: it holds full mangled keys ("std.json.Parse.parse")
            // AND root-level user modules' own mangled keys ("Parse.parse" — a src/-
            // root module has no directory prefix), which is exactly what sema's
            // import-resolved qualifier names. The tail-key index is the fallback
            // for plain short qualifiers whose owner is a DIRECTORY-qualified module
            // ("File" → std.io.File.remove). This order keeps both call forms
            // binding to the module the import resolved to.
            let hit = self
                .func_subgraphs
                .get(key.as_str())
                .or_else(|| self.func_short_index.get(key.as_str()));
            if let Some(&sg) = hit {
                if let Some(diag) = self.conflict_diag(&key) {
                    return Some(Err(diag));
                }
                self.log_call_bind(site, name, recv, &key, sg);
                return Some(Ok(sg));
            }
        }
        // 3. current-module mangled
        if let Some(ref mp) = cur_path {
            let key = format!("{}.{}", mp, name);
            if let Some(&sg) = self.func_subgraphs.get(key.as_str()) {
                self.log_call_bind(site, name, recv, &key, sg);
                return Some(Ok(sg));
            }
        }
        // 4. package-scoped key
        if let Some(ref mp) = cur_path {
            if let Some(pos) = mp.rfind('.') {
                let key = format!("{}::{}", &mp[..pos], name);
                if let Some(&sg) = self.func_subgraphs.get(key.as_str()) {
                    if let Some(diag) = self.conflict_diag(&key) {
                        return Some(Err(diag));
                    }
                    self.log_call_bind(site, name, recv, &key, sg);
                    return Some(Ok(sg));
                }
            }
        }
        // 5. bare (builtin + user/dep), guarded by the collision tripwire
        if let Some(&sg) = self.func_subgraphs.get(name) {
            if let Some(diag) = self.conflict_diag(name) {
                return Some(Err(diag));
            }
            self.log_call_bind(site, name, recv, name, sg);
            return Some(Ok(sg));
        }
        // 5b. Global unique-name index (std included): the sema layer
        // predeclares every loaded module's functions into the root env, so
        // a bare call to a globally-UNIQUE name type-checks from anywhere —
        // honor the same contract (`is_tty(0)` after importing
        // std.os.Tty). Contested names were recorded by the tripwire
        // and error here.
        if let Some(&sg) = self.global_bare_index.get(name) {
            if let Some(diag) = self.conflict_diag(name) {
                return Some(Err(diag));
            }
            self.log_call_bind(site, name, recv, name, sg);
            return Some(Ok(sg));
        }
        None
    }

    /// Diagnostic for a call resolving through a tripwire-conflicted key:
    /// `None` when the key is unambiguous. Covers every registration key
    /// family (bare / short-qualified / package / import alias).
    fn conflict_diag(&self, key: &str) -> Option<String> {
        let cands = self.name_conflicts.get(key)?;
        if cands.len() < 2 {
            return None;
        }
        // Suggest the fully-qualified form of the first candidate: its head
        // is the qualifier, the key's tail is the method (`A.f` + key "f" →
        // "A.f(...)", not "f.f(...)").
        let example = match cands.first().and_then(|c| c.rsplit_once('.')) {
            Some((qual, _)) => format!("{}.{}(...)", qual, key.rsplit('.').next().unwrap_or(key)),
            None => format!("{}(...)", key),
        };
        Some(format!(
            "ambiguous call '{}': [{}] — qualify the call (e.g. {})",
            key,
            cands.join(", "),
            example,
        ))
    }

    /// Records a key-family collision into the tripwire registry: `key`'s
    /// slot is contested by two DISTINCT functions (`sg_id` and `other`).
    /// Any call that later RESOLVES through `key` becomes a hard error
    /// listing both candidates — recording alone changes nothing for calls
    /// that resolve through unambiguous keys.
    fn record_key_conflict(&mut self, key: &str, sg_id: SubGraphId, other: SubGraphId) {
        let entry = self.name_conflicts.entry(key.to_string()).or_default();
        for cand in [sg_id, other] {
            if let Some(q) = self.sg_qualified_names.get(&cand) {
                if !entry.contains(q) {
                    entry.push(q.clone());
                }
            }
        }
    }

    /// Provenance for every resolution: which site asked, which key won, and
    /// which sg it bound — "who did I actually call" in one glance
    /// (module-name diagnostic, removed 2026-09-03).
    pub(super) fn log_call_bind(&self, site: &str, name: &str, recv: Option<&str>, key: &str, sg: SubGraphId) {
    }

    /// @internal access enforcement. Functions marked `@internal` are stdlib
    /// implementation primitives: only `builtin/**` and `std/**` modules may
    /// bind them. Consulted from `resolve_func` (subgraph-target calls) and
    /// the extern-dispatch branch of `compile_call` (CF_DYN_FFI_CALL). A
    /// same-named declaration in the caller's own module shadows the internal
    /// one — the name is theirs, not the stdlib's.
    pub(super) fn internal_access_blocked(&self, name: &str) -> bool {
        if !self.internal_funcs.contains(name) {
            return false;
        }
        let caller = self.current_module().name;
        if caller.starts_with("builtin/") || caller.starts_with("std/") {
            return false;
        }
        !self.current_module().find_function(name).is_some()
    }

    /// Diagnostic for a blocked @internal reference (shared by both guard
    /// sites so the wording stays identical).
    pub(super) fn internal_access_diag(&self, name: &str) -> String {
        format!(
            "'{}' is @internal (a stdlib implementation primitive): call the public std wrapper instead — in module '{}'",
            name,
            self.current_module().name
        )
    }

    pub fn build(mut self) -> DataFlowGraph {
        // 0. Pre-register all functions (builtin + std + dep + user) into func_subgraphs to solve forward references:
        //    When function A calls function B, B may not yet be compiled (not registered in func_subgraphs),
        //    causing call_target to be unbound and compute_call_launch to silently return VOID.
        //    After pre-registration, all function names resolve to a SubGraphId; bodies are filled in a later pass.
        //    Also register mangled names (module_path.function_name) for selective import alias resolution.
        let all_modules: Vec<&crate::ast::Ast::Module<'_>> = self
            .builtin_modules
            .iter()
            .copied()
            .chain(std::iter::once(self.module))
            .collect();
        for m in &all_modules {
            let module_path = crate::sema::Sema::module_logical_path(m.name);
            for d in &m.declarations {
                if let crate::ast::Ast::Decl::FunDecl { name, params, is_async, attributes, visibility, .. } = &d.node {
                    // @internal registry: bare names of stdlib implementation
                    // primitives, consulted by `internal_access_blocked` in
                    // BOTH resolution paths (externs are skipped as subgraph
                    // targets below but still land here).
                    if attributes.iter().any(|a| a.name == crate::ffi::ATTR_INTERNAL)
                        // Phase 2 规则化(2026-09-05):@internal 的语义是
                        // "用户禁调"——std 层的非 pub 函数天然如此(模块私有,
                        // 选择性导入绕过可见性时的守卫),逐个打标是历史欠账;
                        // 现由引擎规则统一覆盖(std 非 pub 一律入表),builtin
                        // 的显式 @internal 保留(语义相同的声明位形态)。
                        || (m.name.starts_with("std/")
                            && !matches!(visibility, crate::ast::Ast::Visibility::Public))
                    {
                        self.internal_funcs.insert(name.to_string());
                    }
                    // Skip @extern("C") functions: they are only called via FFI and need no subgraph
                    if let crate::ast::Ast::Decl::FunDecl { extern_c_body, .. } = &d.node {
                        if extern_c_body.is_some() {
                            continue;
                        }
                    }
                    // One function = one placeholder. If the mangled key
                    // already has a placeholder (the module was reached twice
                    // — e.g. via the user's import AND the full-std preload),
                    // REUSE it: minting a second sg would leave every
                    // or_insert key (short-qualified / package) pointing at
                    // the never-compiled first placeholder.
                    let mangled_key = module_path
                        .as_ref()
                        .map(|mp| format!("{}.{}", mp, name));
                    let sg_id = match mangled_key.as_deref().and_then(|k| self.func_subgraphs.get(k)) {
                        Some(&existing) => existing,
                        None => self.register_subgraph_placeholder(name, params.len() as u8, *is_async),
                    };
                    // Qualified display name for collision diagnostics.
                    let qualified = mangled_key
                        .clone()
                        .unwrap_or_else(|| name.to_string());
                    self.sg_qualified_names.insert(sg_id, qualified.clone());
                    // Bare-name policy (user-approved): std modules
                    // (stdlib/std/**, logical path "std.*") register NO bare
                    // slot — their calls resolve through mangled / package /
                    // recv-qualified keys (see `resolve_func`). builtin
                    // (globally visible by design) and user/dep modules keep
                    // the bare slot, guarded by the collision tripwire: two
                    // distinct functions competing for one bare key are
                    // recorded and any bare call through that key is a hard
                    // error at the call site.
                    let is_std = module_path
                        .as_deref()
                        .map(|mp| mp.starts_with("std."))
                        .unwrap_or(false);
                    if !is_std {
                        if let Some(&prev_sg) = self.func_subgraphs.get(&**name) {
                            if prev_sg != sg_id {
                                self.record_key_conflict(name, sg_id, prev_sg);
                            }
                        }
                        self.func_subgraphs.insert(name.to_string(), sg_id);
                    }
                    // Global unique-name index (ALL modules, std included):
                    // mirrors sema's root-env predeclaration, so a bare call
                    // to a globally-unique name binds even without a bare
                    // slot. Contested names go to the tripwire; first
                    // registrant stays indexed.
                    match self.global_bare_index.get(&**name) {
                        Some(&prev_sg) => {
                            if prev_sg != sg_id {
                                self.record_key_conflict(name, sg_id, prev_sg);
                            }
                        }
                        None => {
                            self.global_bare_index.insert(name.to_string(), sg_id);
                        }
                    }
                    if let Some(ref mp) = module_path {
                        // Full mangled name (module_path.function_name)
                        let mangled = format!("{}.{}", mp, name);
                        self.func_subgraphs.insert(mangled, sg_id);
                        // Short qualified name (module tail segment + fn name): the call-site
                        // shape `File.remove(...)` resolves by the recv identifier.
                        // Lives in func_short_index (NOT func_subgraphs) so a root-level
                        // user module's own mangled key can never collide with — and be
                        // reuse-confused with — a std module's short key.
                        // or_insert (first-wins) keeps the slot stable, but a DIFFERENT
                        // function competing for it is recorded — resolving a call through
                        // a conflicted short key is a hard error at the call site.
                        if let Some(tail) = mp.rsplit('.').next() {
                            let short = format!("{}.{}", tail, name);
                            match self.func_short_index.get(&short) {
                                Some(&prev_sg) if prev_sg != sg_id => {
                                    self.record_key_conflict(&short, sg_id, prev_sg);
                                }
                                None => {
                                    self.func_short_index.insert(short, sg_id);
                                }
                                _ => {}
                            }
                        }
                        // Package-scoped key (`std.math::fn`): stdlib modules commonly call
                        // siblings in the same package directory bare (`ldexp_f64_impl`
                        // defined in Round.frond, called from Power.frond). This is package
                        // visibility, not a global bare name — first registrant wins inside
                        // the package; two same-named functions in ONE package are a real
                        // ambiguity and get the same tripwire treatment.
                        if let Some(pos) = mp.rfind('.') {
                            let pkg = &mp[..pos];
                            let pkg_key = format!("{}::{}", pkg, name);
                            match self.func_subgraphs.get(&pkg_key) {
                                Some(&prev_sg) if prev_sg != sg_id => {
                                    self.record_key_conflict(&pkg_key, sg_id, prev_sg);
                                }
                                None => {
                                    self.func_subgraphs.insert(pkg_key, sg_id);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        // 0a. Pre-register type method subgraphs into method_subgraphs: (type_id, method_idx) -> SubGraphId
        //     Also register the mangled name in func_subgraphs for selective import alias resolution
        //     type_id = dynamic_type_id(type_def_index); method_idx = the method's position in TypeDefInfo.methods
        for m in &all_modules {
            for d in &m.declarations {
                if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = &d.node {
                    let canonical = self.sema.resolve_type_key_in(m.name, name);
                    let type_id = self.sema.type_def_idx(&canonical).map(|idx| crate::types::dynamic_type_id(idx));
                    if let Some(tid) = type_id {
                        for (method_idx, method) in methods.iter().enumerate() {
                            if method.body.is_some() {
                                let mangled = format!("{}.{}", name, method.name);
                                let sg_id = self.register_subgraph_placeholder(
                                    &mangled,
                                    method.params.len() as u8,
                                    method.is_async,
                                );
                                self.method_subgraphs.insert((tid, method_idx as u16), sg_id);
                                self.func_subgraphs.insert(mangled, sg_id);
                            }
                        }
                    }
                }
            }
        }
        // 0a-local. Pre-register LOCAL type method subgraphs (types declared inside function
        // bodies via `Stmt::LocalDecl(TypeDecl)`). Sema registers these into type_def_index
        // during check_decl, but the loop above only scans top-level declarations. The collector
        // recursively walks function bodies to surface nested type declarations.
        {
            let local_type_methods = self.collect_local_type_methods();
            for (type_name, method_idx) in &local_type_methods {
                // Look up the TypeDecl (top-level or local) to get method info.
                let method_info = self.find_type_method(type_name, *method_idx);
                if let Some((method_name, params_count, is_async)) = method_info {
                    let canonical = self
                        .sema
                        .resolve_type_key_in(self.current_module().name, type_name.as_str());
                    let type_id = match self.sema.type_def_idx(&canonical) {
                        Some(idx) => crate::types::dynamic_type_id(idx),
                        None => continue,
                    };
                    // Skip if already registered (top-level scan covered it)
                    if self.method_subgraphs.contains_key(&(type_id, *method_idx as u16)) {
                        continue;
                    }
                    let mangled = format!("{}.{}", type_name, method_name);
                    let sg_id = self.register_subgraph_placeholder(&mangled, params_count, is_async);
                    self.method_subgraphs.insert((type_id, *method_idx as u16), sg_id);
                    self.func_subgraphs.insert(mangled, sg_id);
                }
            }
        }

        // 0a-inh-links. Record (child, base) pairs for runtime match
        // disambiguation (an ADT child inherits the base's ctor set verbatim,
        // so a value of the child type must match arms compiled for the base).
        for d in self.sema.type_defs.values() {
            for b in d.bases.iter() {
                self.graph
                    .inheritance_links
                    .push((d.name.clone(), b.clone()));
            }
        }

        // 0a-trait. Pre-register trait default method monomorphization subgraphs:
        //   (type_id, trait_def_idx, method_idx) -> SubGraphId
        //   Consumes trait_default_instances collected in the Sema post-phase; registers a dedicated subgraph for each specialization instance.
        //   Instance collection (binding-driven: only the bound trait, plus overrides that
        //   call super) is already done by Monomorph::collect_trait_default_instances.
        for inst in &self.sema.trait_default_instances {
            // Look up the AST info for the trait default method (method_name, params_count,
            // is_async). Search builtin modules AND the current module: a type may implement
            // a trait declared in an embedded stdlib module.
            let method_info = self
                .builtin_modules
                .iter()
                .copied()
                .chain(std::iter::once(self.module))
                .find_map(|m| {
                    m.declarations.iter().find_map(|d| {
                        if let crate::ast::Ast::Decl::TraitDecl { name, methods, .. } = &d.node {
                            // trait_name is a canonical key — bare tail match.
                            if inst.trait_name.as_ref().rsplit('.').next() == Some(name.as_ref()) {
                                if let Some(method) = methods.get(inst.method_idx as usize) {
                                    return Some((
                                        method.name.to_string(),
                                        method.params.len() as u8,
                                        method.is_async,
                                    ));
                                }
                            }
                        }
                        None
                    })
                });
            let (method_name, params_count, is_async) = match method_info {
                Some(info) => info,
                None => continue,
            };
            let mangled = format!("{}.{}.{}", inst.type_name, inst.trait_name, method_name);
            let sg_id = self.register_subgraph_placeholder(&mangled, params_count, is_async);
            self.trait_default_subgraphs
                .insert((inst.type_id, inst.trait_idx, inst.method_idx), sg_id);
        }

        // 0a-inh. Pre-register inherited-method subgraphs (inheritance):
        //   (child_type_id, child_method_idx) -> SubGraphId, mangled
        //   "Child.method" — consumed by compile_inherited_method (step 2d),
        //   which compiles the BASE's method body with the child as receiver.
        for inst in &self.sema.inherited_method_instances {
            let method_info = self
                .builtin_modules
                .iter()
                .copied()
                .chain(std::iter::once(self.module))
                .find_map(|m| {
                    m.declarations.iter().find_map(|d| {
                        if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = &d.node {
                            // base_type_name is a canonical key — match the
                            // AST declaration by bare (tail-segment) name.
                            if inst.base_type_name.as_ref().rsplit('.').next() == Some(name.as_ref()) {
                                if let Some(method) = methods.get(inst.base_method_idx as usize) {
                                    return Some((
                                        method.name.to_string(),
                                        method.params.len() as u8,
                                        method.is_async,
                                    ));
                                }
                            }
                        }
                        None
                    })
                });
            let (method_name, params_count, is_async) = match method_info {
                Some(info) => info,
                None => continue,
            };
            // The child's own methods (registered above by the top-level
            // scan) take precedence for the same (type_id, idx) key — they
            // never collide by construction (inherited idx = own len + k).
            if self.method_subgraphs.contains_key(&(inst.type_id, inst.method_idx)) {
                continue;
            }
            let mangled = format!("{}.{}", inst.type_name, method_name);
            let sg_id = self.register_subgraph_placeholder(&mangled, params_count, is_async);
            self.method_subgraphs.insert((inst.type_id, inst.method_idx), sg_id);
            self.func_subgraphs.insert(mangled, sg_id);
        }

        // 0b. Register selective import aliases in func_subgraphs:
        //     Iterate sema.import_aliases, mapping the alias name to the sg_id of the corresponding mangled name.
        //     The alias name (e.g. "area") goes via import_alias -> mangled name (e.g. "Math.Geometry.circle_area")
        //     -> func_subgraphs lookup for sg_id, registering the alias name to point to the same sg_id.
        let alias_entries: Vec<(String, String)> = self.sema.import_aliases.iter()
            .filter_map(|(alias, target)| match target {
                crate::sema::Sema::AliasTarget::Symbol(mangled) => Some((alias.clone(), mangled.to_string())),
                crate::sema::Sema::AliasTarget::Module(_) => None,
            })
            .collect();
        for (alias, mangled) in &alias_entries {
            if let Some(&sg_id) = self.func_subgraphs.get(mangled.as_str()) {
                // Alias collisions go through the same tripwire: two imports
                // binding the same alias name to different functions make any
                // bare call through that alias a hard error.
                if let Some(&prev_sg) = self.func_subgraphs.get(alias.as_str()) {
                    if prev_sg != sg_id {
                        self.record_key_conflict(alias, sg_id, prev_sg);
                    }
                }
                self.func_subgraphs.insert(alias.clone(), sg_id);
            }
        }

        // 0b. Collect top-level var/val declarations across all modules (entry + builtin/std/dep),
        //     allocating global slots. The entry function injects initialization code at compile time
        //     (switching arena per module).
        //     Global variables are stored in DataFlowGraph.global_var_storage, shared across functions,
        //     independent of the frame chain.
        //     Module index: None = entry module, Some(i) = builtin_modules[i]
        //     Also register mangled names (module_path.name) for selective import alias resolution.
        for d in &self.module.declarations {
            if let crate::ast::Ast::Decl::ExprDecl { stmt: Some(stmt_id), .. } = &d.node {
                let stmt = &self.module.arena.stmt(*stmt_id).node;
                if matches!(stmt, crate::ast::Ast::Stmt::VarDecl { .. } | crate::ast::Ast::Stmt::ValDecl { .. }) {
                    let name = match stmt {
                        crate::ast::Ast::Stmt::VarDecl { name, .. } => *name,
                        crate::ast::Ast::Stmt::ValDecl { name, .. } => *name,
                        _ => unreachable!(),
                    };
                    // Slot keying: the mangled name (module_path.name) is the
                    // primary key — every module's top-level val gets its OWN
                    // slot (std.core.I8.MAX vs std.core.I64.MAX share the bare
                    // name but are distinct variables). The bare name is only a
                    // first-wins alias for same-module lookups.
                    match crate::sema::Sema::module_logical_path(self.module.name) {
                        Some(mp) => {
                            let mangled = format!("{}.{}", mp, name);
                            if !self.global_var_slots.contains_key(&mangled) {
                                let slot = self.global_var_slots.len() as u32;
                                self.global_var_slots.insert(mangled, slot);
                                self.top_level_var_decls.push((None, *stmt_id));
                                self.global_var_slots
                                    .entry(name.to_string())
                                    .or_insert(slot);
                            }
                        }
                        None => {
                            if !self.global_var_slots.contains_key(name) {
                                let slot = self.global_var_slots.len() as u32;
                                self.global_var_slots.insert(name.to_string(), slot);
                                self.top_level_var_decls.push((None, *stmt_id));
                            }
                        }
                    }
                }
            }
        }
        for (i, m) in self.builtin_modules.iter().enumerate() {
            for d in &m.declarations {
                if let crate::ast::Ast::Decl::ExprDecl { stmt: Some(stmt_id), .. } = &d.node {
                    let stmt = &m.arena.stmt(*stmt_id).node;
                    if matches!(stmt, crate::ast::Ast::Stmt::VarDecl { .. } | crate::ast::Ast::Stmt::ValDecl { .. }) {
                        let name = match stmt {
                            crate::ast::Ast::Stmt::VarDecl { name, .. } => *name,
                            crate::ast::Ast::Stmt::ValDecl { name, .. } => *name,
                            _ => unreachable!(),
                        };
                        // Same mangled-primary / bare-alias keying as above.
                        match crate::sema::Sema::module_logical_path(m.name) {
                            Some(mp) => {
                                let mangled = format!("{}.{}", mp, name);
                                if !self.global_var_slots.contains_key(&mangled) {
                                    let slot = self.global_var_slots.len() as u32;
                                    self.global_var_slots.insert(mangled, slot);
                                    self.top_level_var_decls.push((Some(i), *stmt_id));
                                    self.global_var_slots
                                        .entry(name.to_string())
                                        .or_insert(slot);
                                }
                            }
                            None => {
                                if !self.global_var_slots.contains_key(name) {
                                    let slot = self.global_var_slots.len() as u32;
                                    self.global_var_slots.insert(name.to_string(), slot);
                                    self.top_level_var_decls.push((Some(i), *stmt_id));
                                }
                            }
                        }
                    }
                }
            }
        }

        // 0b-2. Register selective import aliases in global_var_slots:
        //       Iterate sema.import_aliases, mapping the alias to the slot of the corresponding mangled name.
        //       e.g. "phi" -> "Math.Algebra.GOLDEN_RATIO" -> slot
        for (alias, target) in &self.sema.import_aliases {
            if let crate::sema::Sema::AliasTarget::Symbol(mangled) = target {
                if let Some(&slot) = self.global_var_slots.get(mangled.as_ref()) {
                    self.global_var_slots.insert(alias.clone(), slot);
                }
            }
        }

        // 0c. Register all modules' top-level types into the base scope (unified with nested types via type_scope_stack lookup)
        //     ADT registers both the type name and each constructor name (constructor name maps to type name, for type_name reflection)
        //     Newtype registers the constructor name (== type name); kind=Newtype drives compute_record_construct to build a NewtypeValue
        //
        //     Module-scoped: the recorded `type_name` is the CANONICAL key
        //     (user modules module-qualified) — it flows into
        //     RecordLitInfo and the runtime AdtValue identity. std/builtin
        //     modules bind straight into the base layer (bare names, std
        //     first). USER modules bind into their per-module scope (own
        //     shadow layer) plus a first-wins base entry for cross-module
        //     bare references — std keeps priority — and an unconditional
        //     canonical base entry (unique per module).
        self.type_scope_stack.push(rustc_hash::FxHashMap::default());
        for m in &all_modules {
            let is_user_module = !(m.name.starts_with("std/") || m.name.starts_with("builtin/"));
            for d in &m.declarations {
                if let crate::ast::Ast::Decl::TypeDecl { name, def, base_types, .. } = &d.node {
                    let canonical: String = self.sema.resolve_type_key_in(m.name, name);
                    // One bind closure: module scope + base-layer policy.
                    let mut bind = |b: &mut Self, key: &str, info: TypeFieldInfo| {
                        if is_user_module {
                            b.bind_module_type_fields(m.name, key, info.clone());
                            b.bind_type_fields_base_first_wins(key, info.clone());
                            if key != canonical {
                                b.bind_type_fields_base_first_wins(&canonical, info);
                            }
                        } else {
                            b.bind_type_fields(key, info);
                        }
                    };
                    // Inheritance: children bind the MERGED field list (base
                    // fields + own) from sema — the AST ctor only shows own
                    // fields, and a positional (unnamed) construct breaks
                    // name-based field reads inside inherited/override methods.
                    let merged_names: Option<Vec<String>> = if base_types.is_empty() {
                        None
                    } else {
                        self.sema.type_def_idx(&canonical).and_then(|idx| {
                            let def = &self.sema.type_defs[&idx];
                            if def.bases.is_empty() { return None; }
                            def.constructors.first().map(|c| {
                                c.field_names.iter()
                                    .map(|n| n.as_deref().unwrap_or("_").to_string())
                                    .collect()
                            })
                        })
                    };
                    match def {
                        crate::ast::Ast::TypeDef::Record { fields } => {
                            let field_names: Vec<String> = merged_names.unwrap_or_else(|| {
                                fields.iter().map(|f| f.name.to_string()).collect()
                            });
                            bind(&mut self, name, TypeFieldInfo {
                                field_names,
                                type_name: canonical.clone(),
                                kind: RecordLitKind::Record,
            field_tags: Vec::new(),
                            });
                        }
                        crate::ast::Ast::TypeDef::Adt { constructors } => {
                            // Multi-ctor ADT children: the ctor set lives in
                            // sema (inherited verbatim) — bind EACH inherited
                            // ctor name to the child so expected-type-guided
                            // construction builds child values.
                            if merged_names.is_some() {
                                if let Some(idx) = self.sema.type_def_idx(&canonical) {
                                    let def = &self.sema.type_defs[&idx];
                                    if !def.bases.is_empty() && def.constructors.len() > 1 {
                                        for c in def.constructors.iter() {
                                            let field_names: Vec<String> = c
                                                .field_names
                                                .iter()
                                                .map(|n| n.as_deref().unwrap_or("_").to_string())
                                                .collect();
                                            bind(&mut self, c.name.as_ref(), TypeFieldInfo {
                                                field_names,
                                                type_name: canonical.clone(),
                                                kind: RecordLitKind::Adt,
            field_tags: Vec::new(),
                                            });
                                        }
                                        continue;
                                    }
                                }
                            }
                            if !is_user_module {
                                // std/builtin: original order — the type-name
                                // entry first, then each ctor binding OVERWRITES
                                // it (same key for single-ctor ADTs).
                                bind(&mut self, name, TypeFieldInfo {
                                    field_names: Vec::new(),
                                    type_name: canonical.clone(),
                                    kind: RecordLitKind::Adt,
            field_tags: Vec::new(),
                                });
                                for ctor in constructors {
                                    let field_names: Vec<String> = match &merged_names {
                                        Some(names) if constructors.len() == 1 => names.clone(),
                                        _ => ctor.fields.iter()
                                            .map(|f| f.name.unwrap_or("_").to_string())
                                            .collect(),
                                    };
                                    bind(&mut self, ctor.name, TypeFieldInfo {
                                        field_names,
                                        type_name: canonical.clone(),
                                        kind: RecordLitKind::Adt,
            field_tags: Vec::new(),
                                    });
                                }
                            } else {
                                // USER modules: bind ctor entries FIRST. For
                                // single-ctor ADTs (type name == ctor name,
                                // the stdlib container shape) the old order
                                // bound the empty-field TYPE entry first and
                                // the base layer is first-wins — a cross-module
                                // bare construct then looked up the empty entry,
                                // built an unnamed AdtValue, and every field
                                // read on it failed at runtime (FieldError).
                                // Ctor entries carry the real field list; the
                                // type-name entry fills only free keys.
                                for ctor in constructors {
                                    let field_names: Vec<String> = match &merged_names {
                                        Some(names) if constructors.len() == 1 => names.clone(),
                                        _ => ctor.fields.iter()
                                            .map(|f| f.name.unwrap_or("_").to_string())
                                            .collect(),
                                    };
                                    bind(&mut self, ctor.name, TypeFieldInfo {
                                        field_names,
                                        type_name: canonical.clone(),
                                        kind: RecordLitKind::Adt,
            field_tags: Vec::new(),
                                    });
                                }
                                let taken_module = self
                                    .module_type_scopes
                                    .get(m.name)
                                    .map(|s| s.contains_key(&name.to_string()))
                                    .unwrap_or(false);
                                if !taken_module {
                                    self.bind_module_type_fields(m.name, name, TypeFieldInfo {
                                        field_names: Vec::new(),
                                        type_name: canonical.clone(),
                                        kind: RecordLitKind::Adt,
            field_tags: Vec::new(),
                                    });
                                }
                                self.bind_type_fields_base_first_wins(name, TypeFieldInfo {
                                    field_names: Vec::new(),
                                    type_name: canonical.clone(),
                                    kind: RecordLitKind::Adt,
            field_tags: Vec::new(),
                                });
                            }
                        }
                        crate::ast::Ast::TypeDef::Newtype { name: nt_name, .. } => {
                            // Newtype: constructor name == type name, kind=Newtype
                            bind(&mut self, nt_name, TypeFieldInfo {
                                field_names: Vec::new(),
                                type_name: canonical.clone(),
                                kind: RecordLitKind::Newtype,
            field_tags: Vec::new(),
                            });
                        }
                        crate::ast::Ast::TypeDef::Alias { .. } => {}
                    }
                }
            }
        }

        // 0c. Pre-register monomorphization instance subgraph placeholders BEFORE any module is compiled.
        //     This ensures that cross-module calls to generic functions (e.g. EditorRenderer calling eprintln<T>)
        //     can find the mangled name (func_name#instance_id) during step 1 (builtin module compilation).
        //     Without this, call_target is never set -> compute_call_launch returns VOID at runtime.
        //     (Previously this ran as step 2d after step 1, causing cross-module generic calls to fail silently.)
        for inst in self.sema.monomorph_instances.iter().filter(|inst| !inst.type_args.is_empty()) {
            let mangled = format!("{}#{}", inst.func_name, inst.instance_id);
            if !self.func_subgraphs.contains_key(mangled.as_str()) {
                let param_count = self.sema.get_func_sig_in(&inst.module_name, &inst.func_name)
                    .map(|sig| sig.param_is_ref.len() as u8)
                    .unwrap_or(0);
                let sg_id = self.register_subgraph_placeholder(&mangled, param_count, inst.is_async);
                self.func_subgraphs.insert(mangled, sg_id);
            }
        }

        // 1. Compile builtin module functions first (register into func_subgraphs for user code to call)
        let builtin_fun_names: Vec<(Box<str>, usize)> = self
            .builtin_modules
            .iter()
            .enumerate()
            .flat_map(|(i, m)| {
                m.declarations.iter().filter_map(move |d| match &d.node {
                    crate::ast::Ast::Decl::FunDecl { name, extern_c_body, .. } => {
                        // Skip @extern("C") functions
                        if extern_c_body.is_some() { return None; }
                        Some((name.to_string().into_boxed_str(), i))
                    }
                    _ => None,
                })
            })
            .collect();
        // Compile by (module, name): builtin_fun_names already carries the
        // declaring module index — same-named functions in different modules
        // (File.chmod / Fs.chmod) each get THEIR body compiled. A bare-name
        // re-resolution here would compile the first match twice and leave
        // the other's qualified keys pointing at an empty placeholder.
        for (name, mod_idx) in &builtin_fun_names {
            self.compile_function_in(Some(*mod_idx), name);
        }

        // 1b. Compile TypeDecl methods in builtin modules (indexed by method_idx).
        // Module-qualified (same rationale as 1a above): a bare type name shared
        // by a std module and a user dep module (std.json's `Parser` vs a user
        // `Parser`) made the bare scan compile the std twin twice and leave the
        // user type's method subgraphs as empty placeholders (runtime panic).
        let builtin_methods: Vec<(usize, String, usize)> = self
            .builtin_modules
            .iter()
            .enumerate()
            .flat_map(|(mod_i, m)| {
                m.declarations.iter().flat_map(move |d| {
                    if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = &d.node {
                        methods
                            .iter()
                            .enumerate()
                            .filter(|(_, mt)| mt.body.is_some())
                            .map(move |(idx, _)| (mod_i, name.to_string(), idx))
                            .collect::<Vec<_>>()
                    } else {
                        Vec::new()
                    }
                })
            })
            .collect();
        for (mod_i, type_name, method_idx) in &builtin_methods {
            self.compile_builtin_method_in(Some(*mod_i), type_name, *method_idx);
        }

        // 2. Collect user module function names (skip @extern("C") functions + analyzer-flagged dead functions)
        //    Dead functions are not compiled into subgraphs: the analyzer has confirmed no call path reaches them (single-module analysis)
        let fun_names: Vec<Box<str>> = self
            .module
            .declarations
            .iter()
            .enumerate()
            .filter_map(|(idx, d)| match &d.node {
                crate::ast::Ast::Decl::FunDecl { name, extern_c_body, is_entry, .. } => {
                    if extern_c_body.is_some() { return None; }
                    // Entry functions are never eliminated (the analyzer already excludes them; this is a double check)
                    if *is_entry { return Some(name.to_string().into_boxed_str()); }
                    // Skip analyzer-flagged dead functions
                    if self.is_dead_func(idx) { return None; }
                    Some(name.to_string().into_boxed_str())
                }
                _ => None,
            })
            .collect();

        // 2b. Compile TypeDecl methods in the user module (indexed by method_idx; must complete before step 3)
        let user_methods: Vec<(String, usize)> = self
            .module
            .declarations
            .iter()
            .flat_map(|d| {
                if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = &d.node {
                    methods
                        .iter()
                        .enumerate()
                        .filter(|(_, mt)| mt.body.is_some())
                        .map(|(idx, _)| (name.to_string(), idx))
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                }
            })
            .collect();
        for (type_name, method_idx) in &user_methods {
            self.compile_user_method(type_name, *method_idx);
        }
        // 2b-local. Compile LOCAL type methods (types declared inside function bodies).
        // These were pre-registered in step 0a-local; compile their bodies now.
        let local_methods = self.collect_local_type_methods();
        for (type_name, method_idx) in &local_methods {
            self.compile_user_method(type_name, *method_idx);
        }

        // 2c. Compile the monomorphized specialization of trait default methods:
        //     Consumes trait_default_instances collected in the Sema post-phase; compiles a specialized subgraph for each instance.
        //     Entries in trait_default_subgraphs were pre-registered in step 0a-trait.
        for (inst_idx, inst) in self.sema.trait_default_instances.iter().enumerate() {
            self.compile_trait_default_method(
                inst.trait_name.as_ref(),
                inst.method_idx as usize,
                inst.type_name.as_ref(),
                inst_idx,
            );
        }

        // 2d. Compile inherited-method instances (inheritance): base method
        //     bodies compiled with the child as receiver. Entries in
        //     method_subgraphs were pre-registered in step 0a-inh.
        for inst_idx in 0..self.sema.inherited_method_instances.len() {
            self.compile_inherited_method(inst_idx);
        }

        // 3. Compile user module functions (declaring module = the entry module)
        for name in &fun_names {
            self.compile_function_in(None, name);
        }

        // 3a. Compile monomorphization instances: consumes Sema's monomorph_instances,
        //     generating a specialized subgraph for each generic function instance (registered under mangled name).
        //     Only handles instances with type_args (non-generic instances are covered by compile_function).
        let instances: Vec<crate::sema::Sema::MonomorphInstance> = self.sema.monomorph_instances
            .iter()
            .filter(|inst| !inst.type_args.is_empty())
            .cloned()
            .collect();
        for inst in &instances {
            self.compile_monomorph_instance(inst);
        }

        // DEBUG: dump all func_subgraphs whose node_range is (0,0) — these are uncompiled placeholders

        // W2: storage versioning needs nested_ranges first; downstreams must
        // reflect the version edges appended by the versioning pass.
        self.graph.compute_nested_ranges();
        self.apply_storage_versioning();

        // Compute fan-out
        self.graph.compute_downstreams();

        // Set the entry subgraph: look up the function name in func_subgraphs (compile_function may
        // generate multiple subgraphs per function; declaration index and subgraph index are not 1:1)
        for d in &self.module.declarations {
            if let crate::ast::Ast::Decl::FunDecl { name, is_entry: true, .. } = &d.node {
                if let Some(&sg) = self.func_subgraphs.get(*name) {
                    self.graph.entry_subgraph = Some(sg);
                }
                break;
            }
        }

        // Populate the compute-fn table at build time (indexed by ComputeFnId at runtime)
        self.graph.compute_fns = build_compute_fn_table();

        // Initialize the global variable storage area (pre-allocate Mutex slots by slot count)
        let global_var_count = self.global_var_slots.len();
        let storage: Vec<std::sync::Mutex<Option<crate::value::Value>>> = (0..global_var_count)
            .map(|_| std::sync::Mutex::new(None))
            .collect();
        self.graph.global_var_storage = Arc::new(storage);

        // Initialize the memoization cache tables (one HashMap<u64, Value> per memoized function)
        let memo_table_count = self.memo_table_count as usize;
        let memo_tables: Vec<std::sync::Mutex<rustc_hash::FxHashMap<u64, crate::value::Value>>> =
            (0..memo_table_count)
                .map(|_| std::sync::Mutex::new(rustc_hash::FxHashMap::default()))
                .collect();
        self.graph.memo_tables = Arc::new(memo_tables);

        // Move IR compile-time errors (unimplemented features, etc.) in for the caller to inspect
        self.graph.ir_errors = std::mem::take(&mut self.errors);

        // Pre-compute nested_ranges for all subgraphs; runtime O(len) lookup replaces full-graph scans
        self.graph.compute_nested_ranges();

        // W5: flatten loop condition-tree reset plans once, so the engine's
        // per-iteration reset is mechanical (no per-iteration DFS).
        self.graph.precompute_reset_plans();

        // Move the build-time string_pool into graph.string_pool (ConstValue::Str references this pool)
        let pool = std::mem::take(&mut self.string_pool);
        self.graph.string_pool = Arc::from(pool);

        self.fill_sg_names();
        self.graph
    }

    /// Debug aid: mirror func_subgraphs (mangled → sg_id) into a vec indexed
    /// by sg_id so --dump-ir can name the function owning each subgraph.
    fn fill_sg_names(&mut self) {
        let n = self.graph.subgraphs.len();
        self.graph.sg_names = vec![String::new(); n];
        let entries: Vec<(String, SubGraphId)> = self
            .func_subgraphs
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        for (name, sg) in entries {
            let idx = sg.0 as usize;
            if idx < n && self.graph.sg_names[idx].is_empty() {
                self.graph.sg_names[idx] = name;
            }
        }
    }

}

/// Bug #100 helper: collects the names assigned inside a loop body (Assignment /
/// CompoundAssignment with an Ident target). Over-approximation is safe (rebinding a
/// non-modified variable to its home is a no-op when home == current binding).
/// Recurses through Block/If/Match/While/Loop/For and expression blocks; skips lambda
/// bodies (their assignments are scoped to the nested function).
pub(super) fn collect_assigned_names(
    arena: &crate::ast::Ast::AstArena<'_>,
    expr: crate::ast::Ast::ExprId,
    out: &mut rustc_hash::FxHashSet<String>,
) {
    use crate::ast::Ast::Expr;
    match &arena.expr(expr).node {
        Expr::Block { stmts, trailing } => {
            for &st in stmts {
                collect_assigned_names_stmt(arena, st, out);
            }
            if let Some(t) = trailing {
                collect_assigned_names(arena, *t, out);
            }
        }
        Expr::If { cond, then_branch, else_branch } => {
            collect_assigned_names(arena, *cond, out);
            collect_assigned_names(arena, *then_branch, out);
            if let Some(e) = else_branch {
                collect_assigned_names(arena, *e, out);
            }
        }
        Expr::Match { scrutinee, arms } => {
            collect_assigned_names(arena, *scrutinee, out);
            for arm in arms {
                collect_assigned_names(arena, arm.body, out);
            }
        }
        // Skip nested functions/lambdas: their assignments bind their own scopes.
        Expr::Lambda { .. } => {}
        _ => {}
    }
}

/// Deep variant for the param-slotting pass (③): recurses into loop/branch
/// BODIES (the shallow variant deliberately does not — nested loops' own
/// registrations rebind at their own level, and recursing would collect outer
/// loop variables for inner conditions). Lambdas are still skipped: their
/// assignments bind their own scopes.
pub(super) fn collect_assigned_names_deep(
    arena: &crate::ast::Ast::AstArena<'_>,
    expr: crate::ast::Ast::ExprId,
    out: &mut rustc_hash::FxHashSet<String>,
) {
    use crate::ast::Ast::Expr;
    match &arena.expr(expr).node {
        Expr::Block { stmts, trailing } => {
            for &st in stmts {
                collect_assigned_names_deep_stmt(arena, st, out);
            }
            if let Some(t) = trailing {
                collect_assigned_names_deep(arena, *t, out);
            }
        }
        Expr::If { cond, then_branch, else_branch } => {
            collect_assigned_names_deep(arena, *cond, out);
            collect_assigned_names_deep(arena, *then_branch, out);
            if let Some(e) = else_branch {
                collect_assigned_names_deep(arena, *e, out);
            }
        }
        Expr::Match { scrutinee, arms } => {
            collect_assigned_names_deep(arena, *scrutinee, out);
            for arm in arms {
                collect_assigned_names_deep(arena, arm.body, out);
            }
        }
        Expr::Lambda { .. } => {}
        _ => {}
    }
}

fn collect_assigned_names_deep_stmt(
    arena: &crate::ast::Ast::AstArena<'_>,
    stmt: crate::ast::Ast::StmtId,
    out: &mut rustc_hash::FxHashSet<String>,
) {
    use crate::ast::Ast::Stmt;
    match &arena.stmt(stmt).node {
        Stmt::Assignment { target, .. } | Stmt::CompoundAssignment { target, .. } => {
            if let crate::ast::Ast::Expr::Ident(name) = &arena.expr(*target).node {
                out.insert(name.to_string());
            }
        }
        Stmt::Expression { expr } => collect_assigned_names_deep(arena, *expr, out),
        Stmt::While { condition, body } => {
            collect_assigned_names_deep(arena, *condition, out);
            collect_assigned_names_deep(arena, *body, out);
        }
        Stmt::Loop { body } => collect_assigned_names_deep(arena, *body, out),
        Stmt::For { iterable, body, .. } => {
            collect_assigned_names_deep(arena, *iterable, out);
            collect_assigned_names_deep(arena, *body, out);
        }
        Stmt::Return { value: Some(v) } => collect_assigned_names_deep(arena, *v, out),
        Stmt::Throw { expr } => collect_assigned_names_deep(arena, *expr, out),
        _ => {}
    }
}

fn collect_assigned_names_stmt(
    arena: &crate::ast::Ast::AstArena<'_>,
    stmt: crate::ast::Ast::StmtId,
    out: &mut rustc_hash::FxHashSet<String>,
) {
    use crate::ast::Ast::Stmt;
    match &arena.stmt(stmt).node {
        Stmt::Assignment { target, .. } | Stmt::CompoundAssignment { target, .. } => {
            if let crate::ast::Ast::Expr::Ident(name) = &arena.expr(*target).node {
                out.insert(name.to_string());
            }
        }
        Stmt::Expression { expr } => collect_assigned_names(arena, *expr, out),
        // Nested loops' own conditions rebind at their own registration; recursing
        // here would rebind OUTER loop variables in the INNER condition (e.g. `si`
        // in bubble sort's inner loop), which breaks nested-loop shapes.
        Stmt::While { condition, .. } => {
            collect_assigned_names(arena, *condition, out);
        }
        Stmt::Loop { .. } => {}
        Stmt::For { iterable, .. } => {
            collect_assigned_names(arena, *iterable, out);
        }
        Stmt::Return { value: Some(v) } => collect_assigned_names(arena, *v, out),
        Stmt::Throw { expr } => collect_assigned_names(arena, *expr, out),
        _ => {}
    }
}
