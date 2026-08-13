//! Builder.rs — IR builder (AST + SemaResult -> DataFlowGraph)
//!
//! Split from Ir.rs. Contains IrBuilder struct, all compile_* methods,
//! and build() entry point orchestrating the multi-pass compilation pipeline.
//! Depends on crate::ir::Ir (IR data structures).

use crate::ir::Ir::*;
use std::sync::Arc;


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
    pub func_subgraphs: rustc_hash::FxHashMap<String, SubGraphId>,
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
    /// The subgraph id of the function currently being compiled (used for `defer` registration).
    pub current_function_sg: Option<SubGraphId>,
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
    /// Type-field scope stack: constructor/type name -> field name list
    /// (managed in parallel with `scope_stack`).
    pub type_scope_stack: Vec<rustc_hash::FxHashMap<String, TypeFieldInfo>>,
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
enum BuiltinCtorLower {
    /// `Ok(val)`: single-node `compute_throw_ok` (idx 44).
    Ok,
    /// `Err(...)`: inner `record_construct` wrapped by outer `throw_err` (idx 45).
    Err,
    /// `channel(capacity)`: single-node `compute_channel_create` (idx 283).
    Channel,
}

/// Builtin constructor dispatch table: constructor name -> lowering strategy.
const BUILTIN_CTORS: &[(&str, BuiltinCtorLower)] = &[
    ("Ok", BuiltinCtorLower::Ok),
    ("Err", BuiltinCtorLower::Err),
    ("channel", BuiltinCtorLower::Channel),
];

/// reflect top-level function → standalone compute_fn mapping.
///
/// `format(x)` / `type_name(x)` are the two reflect entry points called from
/// generic contexts (e.g. Console.kz `print<T>`). Lowering them directly to
/// `CF_REFLECT_*` keeps the hot path off the FFI dispatch table.
/// The remaining reflect primitives are only reachable as trait-style method
/// calls (`x.kind()`, `x.field_count()`, ...) and are dispatched via
/// `lookup_intrinsic` + `try_lower_intrinsic`.
fn reflect_top_level_cf(name: &str) -> Option<ComputeFnId> {
    use crate::ir::Ir::*;
    match name {
        "format" => Some(CF_REFLECT_FORMAT),
        "type_name" => Some(CF_REFLECT_TYPE_NAME),
        _ => None,
    }
}

/// reflect method-name → (IntrinsicKind, arg_count) mapping.
///
/// Used by `lookup_intrinsic` to give every value — regardless of its static
/// type — access to reflect trait methods (`x.kind()`, `x.format()`, ...).
/// This is the "auto-impl" of `trait Type` / `trait Value`: rather than
/// synthesizing witness-table entries and method bodies for every type, the
/// Builder recognizes reflect method names structurally and lowers them
/// directly to the corresponding `CF_REFLECT_*` compute_fn.
fn reflect_method_intrinsic(method: &str) -> Option<(crate::sema::Sema::IntrinsicKind, usize)> {
    use crate::sema::Sema::IntrinsicKind;
    // UnOp: receiver only, no extra args
    let un = |id: u32| Some((IntrinsicKind::UnOp(id), 0));
    // BinOp: receiver + one index arg
    let bin = |id: u32| Some((IntrinsicKind::BinOp(id), 1));
    match method {
        "kind" => un(328),              // CF_REFLECT_KIND_STR (kind() returns str)
        "type_name" => un(327),         // CF_REFLECT_TYPE_NAME
        "size" => un(330),              // CF_REFLECT_LAYOUT_SIZE (aggregate)
        "alignment" => un(331),         // CF_REFLECT_LAYOUT_ALIGN
        "field_count" => un(332),       // CF_REFLECT_FIELD_COUNT
        "format" => un(290),            // CF_REFLECT_FORMAT
        "constructor" => un(336),       // CF_REFLECT_ADT_CTOR
        "field_name" => bin(333),       // CF_REFLECT_FIELD_NAME
        // field_value removed: its return type cannot be expressed without an "any"
        // type in Kuzo's type system. CF_REFLECT_FIELD_VALUE (334) remains implemented
        // in Compute.rs for potential future use (e.g. a typed field_value<T>(i): T).
        _ => None,
    }
}

/// Tail-recursion-to-iteration context: used by `compile_call` when intercepting self-calls.
/// `self_name` is the current function name; `param_nodes` is the parameter node list.
#[derive(Clone)]
pub(crate) struct TailRecCtx {
    self_name: String,
    param_nodes: Vec<NodeId>,
}

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
    /// Stack-pointer node (`sp`; a local variable within the function subgraph).
    pub sp_node: NodeId,
    /// Result variable node (`result`; a local variable within the function subgraph).
    pub result_node: NodeId,
    /// Call-site ExprId -> node mapping.
    /// When compiling a continuation, encountering an ExprId in this map returns the
    /// corresponding node (`result` or a `saved` node).
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
            method_subgraphs: rustc_hash::FxHashMap::default(),
            trait_default_subgraphs: rustc_hash::FxHashMap::default(),
            current_trait_default_idx: None,
            current_function_sg: None,
            loop_stack: Vec::new(),
            scope_stack: Vec::new(),
            captured_scopes: Vec::new(),
            captured_vars: rustc_hash::FxHashMap::default(),
            current_function_id: 0,
            current_sg_start: 0,
            current_effect: None,
            in_tail_position: false,
            in_function_top_block: false,
            param_scope_depth: 0,
            in_loop_body: false,
            tail_rec_ctx: None,
            non_tail_rec_ctx: None,
            current_type_args: Vec::new(),
            current_instance_id: None,
            current_method_type: None,
            errors: Vec::new(),
            global_var_slots: rustc_hash::FxHashMap::default(),
            top_level_var_decls: Vec::new(),
            type_scope_stack: Vec::new(),
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
    fn current_analysis(&self) -> Option<&'a crate::pass::Analyzer::AnalysisReport> {
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
    fn is_dead_stmt(&self, stmt_id: crate::ast::Ast::StmtId) -> bool {
        self.current_analysis().map_or(false, |r| r.dead_code.dead_stmts.contains(&stmt_id))
    }

    /// Query whether a function is dead.
    /// `FuncId` = index into the current module's `declarations`.
    #[inline]
    fn is_dead_func(&self, decl_idx: usize) -> bool {
        self.current_analysis().map_or(false, |r| r.dead_func.dead.contains(&crate::pass::Analyzer::FuncId(decl_idx as u32)))
    }

    /// Query whether an expression is an inline-candidate call site.
    /// Returns the callee's `FuncId`; the `IrBuilder` should expand its body rather than launch
    /// a subgraph.
    #[inline]
    fn inline_target(&self, expr_id: crate::ast::Ast::ExprId) -> Option<crate::pass::Analyzer::FuncId> {
        let report = self.current_analysis()?;
        report.inline.expansions.get(&expr_id).copied()
    }

    /// Query whether an expression is marked for stack allocation.
    #[inline]
    fn should_stack_alloc(&self, expr_id: crate::ast::Ast::ExprId) -> bool {
        self.current_analysis().map_or(false, |r| r.stack_alloc.candidates.contains(&expr_id))
    }

    /// Return the module currently being compiled (builtin takes priority, otherwise the user
    /// module).
    fn current_module(&self) -> &'a crate::ast::Ast::Module<'a> {
        self.compiling_builtin.unwrap_or(self.module)
    }

    /// Enter a new scope (variables and type fields are pushed together).
    fn enter_scope(&mut self) {
        self.scope_stack.push(rustc_hash::FxHashMap::default());
        self.type_scope_stack.push(rustc_hash::FxHashMap::default());
    }

    /// Exit a scope (variables and type fields are popped together).
    fn exit_scope(&mut self) {
        self.scope_stack.pop();
        self.type_scope_stack.pop();
    }

    /// Register type field info in the current scope (constructor name / type name ->
    /// `TypeFieldInfo`).
    fn bind_type_fields(&mut self, name: &str, info: TypeFieldInfo) {
        if let Some(scope) = self.type_scope_stack.last_mut() {
            scope.insert(name.to_string(), info);
        }
    }

    /// Look up type field info by walking the scope stack from inner to outer (constructor name
    /// or type name).
    fn lookup_type_fields(&self, name: &str) -> Option<TypeFieldInfo> {
        for scope in self.type_scope_stack.iter().rev() {
            if let Some(info) = scope.get(name) {
                return Some(info.clone());
            }
        }
        None
    }

    /// Bind a variable name to a NodeId (in the current scope).
    fn bind_var(&mut self, name: &str, node_id: NodeId) {
        if let Some(scope) = self.scope_stack.last_mut() {
            scope.insert(name.to_string(), node_id);
        }
    }

    /// Look up the NodeId bound to a variable (searching from inner to outer scope).
    fn lookup_var(&self, name: &str) -> Option<NodeId> {
        for scope in self.scope_stack.iter().rev() {
            if let Some(&node_id) = scope.get(name) {
                return Some(node_id);
            }
        }
        // Global variables: return None; the caller handles them via is_global_var + global_var_slots.
        None
    }

    /// Look up the outermost binding of a variable (the original declaration in the
    /// function-level scope). Used by `Assignment` to determine the correct WriteBack
    /// target: when a variable is assigned in a nested same_function subgraph (e.g.
    /// if branch inside a while body), WriteBack must target the outermost binding
    /// (the root-frame declaration), not an intermediate node in a branch subgraph.
    fn lookup_root_frame_var(&self, name: &str) -> Option<NodeId> {
        for scope in self.scope_stack.iter() {
            if let Some(&node_id) = scope.get(name) {
                return Some(node_id);
            }
        }
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
    fn lookup_captures(&self, scope_expr_id: crate::ast::Ast::ExprId) -> &[crate::sema::Sema::CaptureInfo] {
        let module_name = self.current_module().name;
        let key = crate::sema::Sema::module_expr_key(module_name, scope_expr_id.0 as u64);
        self.sema.get_captures(key)
    }

    /// Check whether a name is a global variable and return its slot index.
    fn lookup_global_var(&self, name: &str) -> Option<u32> {
        self.global_var_slots.get(name).copied()
    }

    /// Compile a global-variable load node (`compute_global_load`, idx 270).
    /// Takes no input; at runtime it reads from `global_var_storage[slot]`.
    fn compile_global_load(&mut self, slot: u32) -> NodeId {
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
    fn compile_global_store(&mut self, val_node: NodeId, slot: u32) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(&[val_node]);
        let node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 1,
            inputs_offset,
            compute_fn: CF_GLOBAL_STORE,
        });
        self.graph.set_global_store_slot(node, slot);
        node
    }

    /// Determine whether a NodeId falls within the current subgraph's range (i.e. is not an
    /// outer variable).
    fn is_in_current_subgraph(&self, node: NodeId) -> bool {
        node.0 >= self.current_sg_start
    }

    /// Bug #49: check whether the current function subgraph has registered a defer (after defer
    /// compilation, `defer_table` is non-empty).
    /// Used to decide whether a local variable reassignment needs a WriteBack to the original
    /// node.
    fn current_function_has_defer(&self) -> bool {
        if let Some(sg_id) = self.current_function_sg {
            if let Some(sg) = self.graph.subgraphs.get(sg_id.0 as usize) {
                return !sg.defer_table.is_empty();
            }
        }
        false
    }

    /// Compile a WriteBack node: assigns an outer variable, writing it back to the function's
    /// root frame via `root_frame_ptr`.
    /// Returns the NodeId of the WriteBack node.
    fn compile_writeback_node(&mut self, val_node: NodeId, target_outer: NodeId) -> NodeId {
        let wb_off = self.graph.inputs_pool.push(&[val_node]);
        let wb_node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 1,
            inputs_offset: wb_off,
            compute_fn: CF_WRITEBACK, // compute_writeback
        });
        self.graph.set_writeback_target(wb_node, target_outer);
        wb_node
    }

    /// Map a `CompoundAssignOp` to the `ComputeFnId` of the corresponding binary operation.
    ///
    /// Looks up `arith_base` by concrete type: arithmetic operations use offsets 0-4;
    /// bitwise operations use offsets 5-9 (integers only).
    fn compound_assign_op_to_compute_fn(
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
                let n = self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: 1,
                    inputs_offset,
                    compute_fn: CF_PROPAGATE, // compute_propagate
                });
                n
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
                let inner_node = self.compile_subexpr(*inner);
                let inputs_offset = self.graph.inputs_pool.push(&[inner_node]);
                self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: 1,
                    inputs_offset,
                    compute_fn: CF_REF_OF,
                })
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

    /// Compile an `Expr::Ident(name)` reference: local var, captured/outer var, implicit-this
    /// field access, global load, or nullary ADT/type constructor.
    fn compile_ident(
        &mut self,
        expr_id: crate::ast::Ast::ExprId,
        name: &str,
    ) -> NodeId {
        match self.lookup_var(name) {
            Some(node_id) => {
                // When `current_effect` exists, create a CF_SEQ dependency node to ensure the
                // variable read executes only after prior side effects complete.
                // This prevents an expression from reading a stale value before a WriteBack in
                // a while/loop subgraph updates it.
                // Consistent with the `current_effect` dependency mechanism in
                // `compile_global_load`.
                match self.current_effect {
                    Some(eff) => self.chain_effects(Some(eff), node_id),
                    None => node_id,
                }
            }
            None => {
                // Implicit this: bare identifier resolved to an instance field by sema.
                // Sema marks such accesses on `ExprInfo.implicit_this`; synthesize an explicit
                // `this.<field>` FieldAccess node. The method variant is handled in the Call
                // branch (it needs the argument list).
                if let Some(access) = self.expr_implicit_this(expr_id).cloned() {
                    if let crate::sema::Sema::ImplicitThisAccess::Field(field) = access {
                        let this_node = self
                            .lookup_var("this")
                            .expect("this binding must exist in method body");
                        return self.build_implicit_field_access(this_node, &field);
                    }
                    // Method variant handled in Call branch.
                }
                match self.lookup_global_var(name) {
                    Some(slot) => self.compile_global_load(slot),
                    None => {
                        // Nullary ADT / type constructor detection: when an Ident is neither a
                        // local nor a global variable, check whether it is a nullary constructor
                        // (e.g. `Lt`/`Leaf`/`Red`) and compile it as a nullary construct node.
                        // Parameterized constructors (non-empty `field_names`) are not handled
                        // here (they go through the `Call` path with arguments).
                        // A newtype always has an inner value, so it can never be nullary.
                        let tf_info = self.lookup_constructor_field_names(name)
                            .or_else(|| self.lookup_type_field_names(name));
                        match tf_info {
                            Some(info) if info.field_names.is_empty() && info.kind != RecordLitKind::Newtype => {
                                let inputs_offset = self.graph.inputs_pool.push(&[]);
                                let node = self.graph.add_node(Node {
                                    kind: NodeKind::BinOp,
                                    input_count: 0,
                                    inputs_offset,
                                    compute_fn: CF_RECORD_CONSTRUCT, // record_construct
                                });
                                self.graph.set_record_lit_info(node, RecordLitInfo {
                                    type_name: info.type_name.clone(),
                                    field_names: Vec::new(),
                                    constructor: name.to_string(),
                                    kind: info.kind,
                                });
                                node
                            }
                            _ => self.compile_const(),
                        }
                    }
                }
            }
        }
    }

    /// Compile an `Expr::Assign { target, value }` expression.
    ///
    /// Used for assignments in expression contexts such as defer bodies.
    /// Consistent with the `Ident` logic of `Stmt::Assignment`:
    ///   captured variable -> WriteBack; outer variable -> WriteBack;
    ///   global variable -> global_store; local -> bind_var.
    fn compile_assign(
        &mut self,
        target: crate::ast::Ast::ExprId,
        value: crate::ast::Ast::ExprId,
    ) -> NodeId {
        let raw_val = self.compile_subexpr(value);
        let val_node = self.chain_effects(self.current_effect, raw_val);
        let target_expr = &self.current_module().arena.expr(target).node;
        match target_expr {
            crate::ast::Ast::Expr::Ident(name) => {
                // Implicit-this field assignment: `field = value` inside a method body
                // resolves to `this.field = value`. Without this, the bare name would
                // create a local binding instead of mutating the instance field.
                if let Some(crate::sema::Sema::ImplicitThisAccess::Field(field)) = self.expr_implicit_this(target).cloned() {
                    let this_node = self
                        .lookup_var("this")
                        .expect("this binding must exist in method body");
                    let off = self.graph.inputs_pool.push(&[this_node, val_node]);
                    let set_node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: 2,
                        inputs_offset: off,
                        compute_fn: CF_RECORD_FIELD_SET,
                    });
                    self.graph.set_field_set_name(set_node, field.to_string());
                    return self.compile_void_const();
                }
                let captured_source = self.captured_scopes.iter().rev()
                    .find_map(|scope| scope.iter()
                        .find(|(n, _)| n.as_str() == *name)
                        .map(|(_, node)| *node));
                if let Some(source) = captured_source {
                    let wb_node = self.compile_writeback_node(val_node, source);
                    self.bind_var(name, val_node);
                    self.current_effect = Some(wb_node);
                } else if let Some(outer_node) = self.lookup_var(name) {
                    if !self.is_in_current_subgraph(outer_node) {
                        let wb_node = self.compile_writeback_node(val_node, outer_node);
                        self.bind_var(name, val_node);
                        self.current_effect = Some(wb_node);
                    } else if let Some(&captured_node) = self.captured_vars.get(*name) {
                        let wb_node = self.compile_writeback_node(val_node, captured_node);
                        self.bind_var(name, val_node);
                        self.current_effect = Some(wb_node);
                    } else {
                        self.bind_var(name, val_node);
                    }
                } else if let Some(slot) = self.lookup_global_var(name) {
                    let store_node = self.compile_global_store(val_node, slot);
                    self.current_effect = Some(store_node);
                } else {
                    self.bind_var(name, val_node);
                }
            }
            crate::ast::Ast::Expr::FieldAccess { recv: obj, field } => {
                let obj_node = self.compile_subexpr(*obj);
                let off = self.graph.inputs_pool.push(&[obj_node, val_node]);
                let set_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: off,
                    compute_fn: CF_RECORD_FIELD_SET, // record_field_set
                });
                self.graph.set_field_set_name(set_node, field.to_string());
                self.current_effect = Some(self.chain_effects(self.current_effect, set_node));
            }
            // `recv?.field = value`: skip the assignment when `obj` is null.
            crate::ast::Ast::Expr::SafeAccess { recv: obj, field } => {
                let obj_node = self.compile_subexpr(*obj);
                let off = self.graph.inputs_pool.push(&[obj_node, val_node]);
                let set_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: off,
                    compute_fn: CF_RECORD_FIELD_SET, // record_field_set
                });
                self.graph.set_field_set_name(set_node, field.to_string());
                self.graph.set_safe_op(set_node);
                self.current_effect = Some(self.chain_effects(self.current_effect, set_node));
            }
            // `*ref = value` → compute_deref_write(282)
            crate::ast::Ast::Expr::Deref(ref_inner) => {
                let ref_node = self.compile_subexpr(*ref_inner);
                let off = self.graph.inputs_pool.push(&[ref_node, val_node]);
                let write_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: off,
                    compute_fn: CF_DEREF_WRITE, // compute_deref_write
                });
                self.current_effect = Some(self.chain_effects(self.current_effect, write_node));
            }
            _ => {}
        }
        self.compile_void_const()
    }

    /// Compile an `Expr::CompoundAssign { op, target, value }` expression: `target op= value`.
    fn compile_compound_assign(
        &mut self,
        op: crate::ast::Ast::CompoundAssignOp,
        target: crate::ast::Ast::ExprId,
        value: crate::ast::Ast::ExprId,
    ) -> NodeId {
        let val_node = self.compile_subexpr(value);
        let target_expr = &self.current_module().arena.expr(target).node;
        let bin_compute = self.compound_assign_op_to_compute_fn(op, target);
        match target_expr {
            crate::ast::Ast::Expr::Ident(name) => {
                // Implicit-this field compound assignment: `field op= value` inside a
                // method body resolves to `this.field op= value`.
                if let Some(crate::sema::Sema::ImplicitThisAccess::Field(field)) = self.expr_implicit_this(target).cloned() {
                    let this_node = self
                        .lookup_var("this")
                        .expect("this binding must exist in method body");
                    // Read the current field value.
                    let get_off = self.graph.inputs_pool.push(&[this_node]);
                    let get_node = self.graph.add_node(Node {
                        kind: NodeKind::FieldAccess,
                        input_count: 1,
                        inputs_offset: get_off,
                        compute_fn: CF_RECORD_FIELD_GET,
                    });
                    // Operation.
                    let bin_off = self.graph.inputs_pool.push(&[get_node, val_node]);
                    let result_node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: 2,
                        inputs_offset: bin_off,
                        compute_fn: bin_compute,
                    });
                    // Write back.
                    let set_off = self.graph.inputs_pool.push(&[this_node, result_node]);
                    let set_node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: 2,
                        inputs_offset: set_off,
                        compute_fn: CF_RECORD_FIELD_SET,
                    });
                    self.graph.set_field_set_name(set_node, field.to_string());
                    return result_node;
                }
                let cur_node = self
                    .lookup_var(name)
                    .unwrap_or_else(|| self.compile_placeholder());
                let off = self.graph.inputs_pool.push(&[cur_node, val_node]);
                let result_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: off,
                    compute_fn: bin_compute,
                });
                self.bind_var(name, result_node);
                result_node
            }
            crate::ast::Ast::Expr::FieldAccess { recv: obj, field }
            | crate::ast::Ast::Expr::SafeAccess { recv: obj, field } => {
                let obj_node = self.compile_subexpr(*obj);
                // Read the current field value.
                let get_off = self.graph.inputs_pool.push(&[obj_node]);
                let get_node = self.graph.add_node(Node {
                    kind: NodeKind::FieldAccess,
                    input_count: 1,
                    inputs_offset: get_off,
                    compute_fn: CF_RECORD_FIELD_GET, // record_field_get
                });
                // The field_get node needs the field name metadata to know which field to extract.
                // compute_record_field_get reads the name via field_set_name (same metadata as field_set).
                self.graph.set_field_set_name(get_node, field.to_string());
                // Operation.
                let bin_off = self.graph.inputs_pool.push(&[get_node, val_node]);
                let result_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: bin_off,
                    compute_fn: bin_compute,
                });
                // Write back. The set_node MUST be chained into the effect graph,
                // otherwise DCE drops it and the field mutation never executes.
                let set_off = self.graph.inputs_pool.push(&[obj_node, result_node]);
                let set_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: set_off,
                    compute_fn: CF_RECORD_FIELD_SET, // record_field_set
                });
                self.graph.set_field_set_name(set_node, field.to_string());
                self.chain_effects(self.current_effect, set_node)
            }
            // `*ref op= value` -> read Cell + operation + write back to Cell.
            crate::ast::Ast::Expr::Deref(ref_inner) => {
                let ref_node = self.compile_subexpr(*ref_inner);
                // Read the current value: compute_deref_read (281).
                let read_off = self.graph.inputs_pool.push(&[ref_node]);
                let read_node = self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: 1,
                    inputs_offset: read_off,
                    compute_fn: CF_DEREF_READ,
                });
                // Operation.
                let bin_off = self.graph.inputs_pool.push(&[read_node, val_node]);
                let result_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: bin_off,
                    compute_fn: bin_compute,
                });
                // Write back to Cell: compute_deref_write (282).
                let write_off = self.graph.inputs_pool.push(&[ref_node, result_node]);
                let _write_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: write_off,
                    compute_fn: CF_DEREF_WRITE,
                });
                result_node
            }
            _ => self.compile_void_const(),
        }
    }

    /// Compile a sub-expression (not in tail position).
    ///
    /// The value of a sub-expression (operand, function argument, `if` condition, field-access
    /// base, etc.) is consumed by its parent expression rather than returned directly as the
    /// function result, so it is never in tail position: `in_tail_position` is turned off before
    /// compilation and restored afterwards. This prevents a `Call` inside a sub-expression from
    /// being mis-tagged as a tail call (otherwise `switch_subgraph` frame reuse would swap away
    /// the current frame, breaking the parent expression's execution of the remaining
    /// sub-expressions / operation nodes; e.g. in `fib(n-1)+fib(n-2)`, mis-tagging `fib(n-1)` as
    /// a tail call would cause `fib(n-2)` and the addition node to never execute).
    fn compile_subexpr(&mut self, expr_id: crate::ast::Ast::ExprId) -> NodeId {
        let prev_tail = self.in_tail_position;
        self.in_tail_position = false;
        let node = self.compile_expr(expr_id);
        self.in_tail_position = prev_tail;
        node
    }

    /// Compile a constant expression (no inputs).
    fn compile_const(&mut self) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(&[]);
        self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset,
            compute_fn: CF_NOOP,
        })
    }

    /// Compile a void constant node (used when `return`/`break`/`continue` has no value).
    fn compile_void_const(&mut self) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(&[]);
        let n = self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset,
            compute_fn: CF_NOOP,
        });
        self.graph.const_values[n.0 as usize] = Some(ConstValue::Void);
        n
    }

    /// Compile a constant expression carrying a raw value, populating `const_values`.
    fn compile_const_with_value(&mut self, expr_id: crate::ast::Ast::ExprId) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(&[]);
        let node_id = self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset,
            compute_fn: CF_NOOP,
        });
        match self.parse_const_value(expr_id) {
            Ok(cv) => {
                self.graph.const_values[node_id.0 as usize] = cv;
            }
            Err(msg) => {
                self.graph.const_values[node_id.0 as usize] = None;
                self.errors.push(msg);
            }
        }
        node_id
    }

    /// Parse a constant value from an AST expression.
    ///
    /// Return value semantics:
    /// - `Ok(Some(cv))`: a valid constant literal that passed type-range checks.
    /// - `Ok(None)`: a non-constant expression (e.g. a variable reference) that cannot be folded
    ///   into a constant.
    /// - `Err(msg)`: constant-literal parsing failed (syntax error or value out of target-type
    ///   range).
    fn parse_const_value(&mut self, expr_id: crate::ast::Ast::ExprId) -> Result<Option<ConstValue>, String> {
        let spanned = self.current_module().arena.expr(expr_id);
        let span = spanned.span;
        match &spanned.node {
            crate::ast::Ast::Expr::IntLit { raw, suffix } => {
                // Suffix takes priority; when absent, consult the type inferred by sema to pick
                // the corresponding integer ConstValue, ensuring the literal's runtime tag matches
                // the contextual type.
                let ty = suffix
                    .map(|s| s.to_string())
                    .or_else(|| self.expr_type_name(expr_id).map(|s| s.to_string()));
                let ty_name = match ty.as_deref() {
                    Some(t) => t,
                    None => return Err(format!(
                        "internal: missing ExprInfo for int literal expr {:?}", expr_id)),
                };

                // The u128 range (0..=2^128-1) exceeds i128, so parse directly with
                // `u128::from_str_radix`.
                // As with float-suffix dispatch, u128 is the only integer type whose range
                // exceeds i128; the dedicated parse path is mathematically necessary, not a
                // special-case judgement.
                if crate::value::ValueTag::from_name(ty_name) == Some(crate::value::ValueTag::U128) {
                    let v = parse_int_to_u128(raw, span)?;
                    return Ok(Some(ConstValue::U128(v)));
                }

                // Parse the integer: supports 0x/0o/0b prefixes and underscore separators.
                let v = parse_int_to_i128(raw, span)?;

                // Range check + type conversion (generic approach, unified for all integer types
                // via a macro).
                Ok(Some(check_int_range(v, ty_name, raw, span)?))
            }
            crate::ast::Ast::Expr::FloatLit { raw, suffix } => {
                // Strip underscore separators (Rust's `parse` does not accept underscores).
                let cleaned: String = raw.chars().filter(|c| *c != '_').collect();
                let is_hex = cleaned.starts_with("0x") || cleaned.starts_with("0X");
                let cv = match suffix {
                    None | Some("f64") => {
                        if is_hex { parse_hex_float_f64(&cleaned).map(ConstValue::F64) }
                        else { cleaned.parse::<f64>().ok().map(ConstValue::F64) }
                    }
                    Some("f32") => {
                        if is_hex { parse_hex_float_f32(&cleaned).map(ConstValue::F32) }
                        else { cleaned.parse::<f32>().ok().map(ConstValue::F32) }
                    }
                    Some("f16") => {
                        if is_hex { parse_hex_float_f16(&cleaned).map(ConstValue::F16) }
                        else {
                            cleaned.parse::<f64>()
                                .ok()
                                .map(|f| ConstValue::F16(crate::value::F16::from_f64(f).to_bits()))
                        }
                    }
                    Some("f128") => {
                        if is_hex { parse_hex_float_f128(&cleaned).map(ConstValue::F128) }
                        else { parse_decimal_f128(&cleaned).map(ConstValue::F128) }
                    }
                    _ => {
                        if is_hex { parse_hex_float_f64(&cleaned).map(ConstValue::F64) }
                        else { cleaned.parse::<f64>().ok().map(ConstValue::F64) }
                    }
                };
                Ok(cv)
            }
            crate::ast::Ast::Expr::BoolLit(b) => Ok(Some(ConstValue::Bool(*b))),
            crate::ast::Ast::Expr::CharLit(c) => Ok(Some(ConstValue::Char(*c))),
            crate::ast::Ast::Expr::StrLit(s) => {
                let (offset, len) = self.intern_str(s);
                Ok(Some(ConstValue::Str { offset, len }))
            }
            crate::ast::Ast::Expr::NullLit => Ok(Some(ConstValue::Null)),
            crate::ast::Ast::Expr::VoidLit => Ok(Some(ConstValue::Void)),
            _ => Ok(None),
        }
    }

    /// Compile a placeholder node (for Expr variants not yet implemented at this stage).
    fn compile_placeholder(&mut self) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(&[]);
        self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset,
            compute_fn: CF_NOOP,
        })
    }

    /// Compile an `If` expression into a Gate node + two branch subgraphs.
    ///
    /// `cond` is compiled into a condition node; `then`/`else` are each compiled into independent
    /// subgraphs.
    /// The Gate node's `condition_input` points to the cond node, and `branches` carries the
    /// branch subgraph ids.
    /// Branch subgraphs take no parameters (closure variable capture is deferred to a later
    /// stage).
    fn compile_if(
        &mut self,
        cond: crate::ast::Ast::ExprId,
        then_branch: crate::ast::Ast::ExprId,
        else_branch: Option<crate::ast::Ast::ExprId>,
    ) -> NodeId {
        // The condition is not in tail position: its value only selects a branch for the Gate
        // rather than being returned directly.
        // The branch result expressions inherit the current tail position (the value of the `if`
        // expression equals the value of the selected branch).
        let cond_node = self.compile_subexpr(cond);
        // Save `current_effect`: branch compilation (`compile_branch_subgraph`) does not restore
        // `current_effect`, so side effects in the else branch (e.g. a non-tail-recursion
        // interception barrier) would leak into the Gate's effect dependency, causing the Gate
        // to wait for the barrier to become ready and prevent completion along the base-case path.
        let prev_effect = self.current_effect;
        self.current_effect = None;
        let (then_sg, then_inputs) = self.compile_branch_subgraph(then_branch);
        let (else_sg, else_inputs) = match else_branch {
            Some(e) => self.compile_branch_subgraph(e),
            None => (self.compile_void_subgraph(), Vec::new()),
        };
        self.current_effect = prev_effect;
        // The Gate depends on `cond_node` (the condition value) and `current_effect` (the prior
        // side effects in the effect chain), ensuring the Gate executes only after prior
        // statements (e.g. `println`) complete.
        let gate_inputs: Vec<NodeId> = match self.current_effect {
            Some(eff) => vec![cond_node, eff],
            None => vec![cond_node],
        };
        let inputs_offset = self.graph.inputs_pool.push(&gate_inputs);
        let gate_node = self.graph.add_node(Node {
            kind: NodeKind::Gate,
            input_count: gate_inputs.len() as u8,
            inputs_offset,
            compute_fn: CF_GATE_LAUNCH,
        });
        self.graph.set_gate_branches(
            gate_node,
            GateBranches {
                condition_input: cond_node,
                branches: vec![
                    (true, then_sg, then_inputs),
                    (false, else_sg, else_inputs),
                ],
            },
        );
        if std::env::var("KUZO_DEBUG_COMPILE").is_ok() {
            let then_r = self.graph.subgraphs[then_sg.0 as usize].node_range;
            let else_r = self.graph.subgraphs[else_sg.0 as usize].node_range;
            eprintln!("[COMPILE_IF] cond_node={} then_sg={} then_range=[{},{}) else_sg={} else_range=[{},{}) gate_node={} cur_mod={:?}",
                cond_node.0, then_sg.0, then_r.0.0, then_r.1.0,
                else_sg.0, else_r.0.0, else_r.1.0,
                gate_node.0, self.current_module().name);
        }
        gate_node
    }

    /// Compile a branch expression into a subgraph (an `If` then/else branch, a Match arm body,
    /// or a defer body).
    ///
    /// A branch subgraph executes in an independent child frame and cannot directly access the
    /// parent frame's value table.
    /// Therefore, the free variables in the branch expression (identifiers referencing outer
    /// scopes) must be captured:
    /// 1. Collect all identifiers within the expression.
    /// 2. Filter out those already bound in the current scope stack (i.e. outer variables).
    /// 3. Create capture nodes at the start of the subgraph (Const placeholders); at runtime
    ///    the Gate/defer injects the values.
    /// 4. Bind the captured names to the capture nodes, so the compiled body references the
    ///    capture nodes rather than the outer nodes.
    ///
    /// Returns `(subgraph id, list of captured outer nodes)`.
    /// The caller passes the outer-node list as `GateBranches.branch_inputs`; the Gate node
    /// injects the captured values via `start_subgraph` when launching the subgraph.
    fn compile_branch_subgraph(&mut self, expr: crate::ast::Ast::ExprId) -> (SubGraphId, Vec<NodeId>) {
        let node_start = self.graph.nodes.len() as u32;

        // Frame-chain passthrough (`root_frame_ptr`) lets the branch subgraph directly reference
        // outer nodes without a capture mechanism (no local copy is created; assignments write
        // back to the root frame via WriteBack).
        // `branch_inputs` is empty: the Gate injects no arguments; nodes inside the branch read
        // outer variables via `get_value_by_global` frame-chain backtracking.
        self.enter_scope();
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;

        // Record the length of the function subgraph's `event_source_decls` before compilation.
        // While compiling the branch body, `build_await_node` registers EventSourceDecls into
        // `current_function_sg` (the function subgraph), but at runtime `compute_await` looks
        // them up using `frame.subgraph_id` (the branch subgraph) -- the branch subgraph's empty
        // `event_source_decls` causes a fallback to AsyncJoin, misclassifying `channel.recv` /
        // `timer.await` as async join (Bug #24). After compilation, the newly added decls are
        // migrated into the branch subgraph.
        // Nested branches are handled correctly: an inner branch drains its own decls first, so
        // by the time an outer branch drains, only its own remain.
        let prev_decl_count = self.current_function_sg
            .and_then(|sg_id| self.graph.subgraphs.get(sg_id.0 as usize))
            .map(|sg| sg.event_source_decls.len())
            .unwrap_or(0);

        let raw_return = self.compile_expr(expr);
        // Link any pending effect into the return node so side-effecting expressions
        // (e.g. `defer b.v = 77` compiles to an Assign with a field_set in current_effect)
        // are not orphaned. Without this, the set_node has no consumer and is dropped by DCE.
        // Only chain when there IS a pending effect (avoids spurious seq nodes for pure branches).
        let return_node = match self.current_effect {
            Some(eff) if eff != raw_return => self.chain_effects(Some(eff), raw_return),
            _ => raw_return,
        };
        self.current_sg_start = prev_sg_start;
        self.exit_scope();

        let node_end = self.graph.nodes.len() as u32;
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);

        // Migrate the event_source_decls newly added while compiling the branch body from the
        // function subgraph into the branch subgraph.
        let branch_decls: Vec<_> = if let Some(func_sg_id) = self.current_function_sg {
            if let Some(func_sg) = self.graph.subgraphs.get_mut(func_sg_id.0 as usize) {
                func_sg.event_source_decls.drain(prev_decl_count..).collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        self.graph.add_subgraph(SubGraph {
            id: sg_id,
            node_range: (NodeId(node_start), NodeId(node_end)),
            param_count: 0,
            entry_node: NodeId(node_start),
            return_node,
            has_suspend: false,
            event_source_decls: branch_decls,
            defer_table: Vec::new(),
            loop_kind: LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });
        (sg_id, Vec::new())
    }

    /// Compile a void subgraph (used when there is no `else` branch).
    fn compile_void_subgraph(&mut self) -> SubGraphId {
        let inputs_offset = self.graph.inputs_pool.push(&[]);
        let node = self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset,
            compute_fn: CF_NOOP,
        });
        self.graph.const_values[node.0 as usize] = Some(ConstValue::Void);
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            id: sg_id,
            node_range: (node, NodeId(node.0 + 1)),
            param_count: 0,
            entry_node: node,
            return_node: node,
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });
        sg_id
    }

    /// Compile a defer-run exit subgraph for loops.
    ///
    /// Contains a single CF_DEFER_RUN node that drains the loop frame's `defer_stack`
    /// (accumulated by CF_DEFER_REGISTER during each iteration) in LIFO order and
    /// executes each defer body with its captured loop-variable value.
    /// Used as the "exit" branch (void_sg replacement) for For/While loops.
    fn compile_defer_run_subgraph(&mut self) -> SubGraphId {
        let inputs_offset = self.graph.inputs_pool.push(&[]);
        let node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 0,
            inputs_offset,
            compute_fn: CF_DEFER_RUN,
        });
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            id: sg_id,
            node_range: (node, NodeId(node.0 + 1)),
            param_count: 0,
            entry_node: node,
            return_node: node,
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });
        sg_id
    }

    /// Compile a panic subgraph (used as match fallback when no arm matches).
    /// The single node uses CF_MATCH_FALLBACK which panics at runtime.
    fn compile_panic_subgraph(&mut self) -> SubGraphId {
        let inputs_offset = self.graph.inputs_pool.push(&[]);
        let node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 0,
            inputs_offset,
            compute_fn: CF_MATCH_FALLBACK,
        });
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            id: sg_id,
            node_range: (node, NodeId(node.0 + 1)),
            param_count: 0,
            entry_node: node,
            return_node: node,
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });
        sg_id
    }

    /// Compile a `select` expression.
    ///
    /// Each `SelectArm` is compiled into an independent subgraph (event-source check + body).
    /// The Gate node (`compute_select_gate`) selects the first ready branch: if a ready branch
    /// exists, it launches that branch's subgraph; if none is ready, the frame suspends,
    /// registers all event sources to wait, and wakes up to re-check when any event arrives.
    fn compile_select(&mut self, arms: &[crate::ast::Ast::SelectArm<'_>]) -> NodeId {
        let mut branches = Vec::with_capacity(arms.len());

        for arm in arms {
            let (event_kind, event_source_node, body_expr, binding) = match arm {
                crate::ast::Ast::SelectArm::Receive { channel_expr, body, binding } => {
                    // `channel_expr` is of the form `ch.recv()`: at compile time we must take the
                    // receiver of `recv` (the channel value), not the entire method call
                    // (`recv()` returns the received value, not the channel itself).
                    // Determined via the intrinsic flag in sema's `method_dispatches` (eliminates
                    // string guards).
                    let ch_node = match &self.current_module().arena.expr(*channel_expr).node {
                        crate::ast::Ast::Expr::MethodCall { recv, .. } => {
                            let key = crate::sema::Sema::module_expr_key(
                                self.expr_key_module(),
                                channel_expr.0 as u64,
                            );
                            let is_recv = self.sema.method_dispatches.get(&key)
                                .and_then(|d| d.intrinsic)
                                .is_some_and(|i| i == crate::sema::Sema::IntrinsicKind::ChannelAwait);
                            if is_recv {
                                self.compile_subexpr(*recv)
                            } else {
                                self.compile_subexpr(*channel_expr)
                            }
                        }
                        _ => self.compile_subexpr(*channel_expr),
                    };
                    (EventSourceKind::Channel, ch_node, *body, *binding)
                }
                crate::ast::Ast::SelectArm::Timeout { duration, body } => {
                    let dur_node = self.compile_subexpr(*duration);
                    (EventSourceKind::Timer, dur_node, *body, None)
                }
            };

            // Create a subgraph for each branch: register a placeholder first (node_range is
            // back-filled after compilation).
            let node_start = self.graph.nodes.len() as u32;
            let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
            self.graph.add_subgraph(SubGraph {
                id: sg_id,
                node_range: (NodeId(node_start), NodeId(node_start)),
                param_count: 0,
                entry_node: NodeId(node_start),
                return_node: NodeId(node_start),
                has_suspend: true,
                event_source_decls: Vec::new(),
                defer_table: Vec::new(),
            loop_kind: LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
            });

            let prev_sg = self.current_function_sg;
            self.current_function_sg = Some(sg_id);
            self.enter_scope();

            // When the arm has a binding (`ch.recv() => v => body`), the received value is bound
            // to `binding` inside the body. Emit a 0-input parameter node as the FIRST node of the
            // branch subgraph; the runtime injects the recv'd value into it (param_count=1) when the
            // branch is selected. The body then references `binding` via this node.
            let mut branch_param_count: u8 = 0;
            if let Some(name) = binding {
                let off = self.graph.inputs_pool.push(&[]);
                let param_node = self.graph.add_node(Node {
                    kind: NodeKind::Const,
                    input_count: 0,
                    inputs_offset: off,
                    compute_fn: CF_NOOP,
                });
                self.bind_var(name, param_node);
                branch_param_count = 1;
            }

            // Compile the body (variable bindings inside the body live in the subgraph scope).
            let result_node = self.compile_expr(body_expr);

            self.exit_scope();
            self.current_function_sg = prev_sg;

            let node_end = self.graph.nodes.len() as u32;
            let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
            sg.node_range = (NodeId(node_start), NodeId(node_end));
            sg.return_node = result_node;
            sg.param_count = branch_param_count;

            branches.push(SelectBranch {
                subgraph_id: sg_id,
                event_kind,
                event_source_node,
            });
        }

        // Create the Gate node (the core of `select`: choose the first ready branch).
        // The Gate depends on every branch's `event_source_node` + `current_effect`, ensuring
        // readiness is checked only after all event sources (channel/timer) are evaluated.
        let mut gate_inputs: Vec<NodeId> = branches.iter().map(|b| b.event_source_node).collect();
        if let Some(eff) = self.current_effect {
            gate_inputs.push(eff);
        }
        let gate_off = self.graph.inputs_pool.push(&gate_inputs);
        let gate_node = self.graph.add_node(Node {
            kind: NodeKind::Gate,
            input_count: gate_inputs.len() as u8,
            inputs_offset: gate_off,
            compute_fn: CF_SELECT_GATE, // compute_select_gate
        });
        self.graph.set_select_info(gate_node, SelectInfo { branches });

        gate_node
    }

    /// Compile a `Lambda` expression into a closure subgraph + closure construct node.
    ///
    /// Appended-parameter model: captured variables are appended to the end of the subgraph's
    /// parameter list.
    /// - Subgraph `param_count` = lambda parameter count + captured-variable count.
    /// - The first N nodes of the subgraph are lambda parameter nodes; the rest are captured
    ///   upvalue parameter nodes.
    /// - A closure construct node (`compute_fn` 40) is created in the current scope; its inputs
    ///   are the captured value nodes.
    /// - The value of the Lambda expression is the closure construct node (at runtime it produces
    ///   a Closure heap object).
    fn compile_lambda(
        &mut self,
        params: &[crate::ast::Ast::Param<'_>],
        body_expr: crate::ast::Ast::ExprId,
        is_async: bool,
        fn_name: Option<&str>,
        lambda_expr_id: Option<crate::ast::Ast::ExprId>,
    ) -> NodeId {
        // 1. Capture analysis (unified): consume Sema's authoritative capture
        //    table for this nested scope. This replaces the builder's own
        //    `collect_free_idents_expr` re-scan. Each CaptureInfo carries the
        //    variable name and a by-val (Snapshot) / by-ref (Reference) mode.
        //
        //    Node resolution still uses `lookup_var` + the `captured_scopes`
        //    chain (unchanged): the Sema table is the single source of truth for
        //    *which* variables are captured and *how*, but the NodeId is an IR
        //    concept that only the builder can resolve.
        let param_names: rustc_hash::FxHashSet<&str> =
            params.iter().map(|p| p.name).collect();
        // Lookup Sema's capture table. For Lambda expressions the key is the
        // Lambda's own ExprId; for nested function declarations (Stmt::LocalDecl)
        // there is no Lambda expr, so Sema records captures under the *body*
        // expression's ExprId — fall back to that.
        let sema_captures: Vec<crate::sema::Sema::CaptureInfo> = {
            let key_id = lambda_expr_id.unwrap_or(body_expr);
            self.lookup_captures(key_id).to_vec()
        };
        let mut captured: Vec<(String, NodeId)> = Vec::new();

        // Self-reference detection: a named function that references itself in its body becomes an
        // upvalue placeholder.
        // At runtime `compute_closure_call` injects the closure value into that slot, enabling
        // recursive calls.
        let self_upvalue_idx = if let Some(fname) = fn_name {
            if !param_names.contains(fname)
                && sema_captures.iter().any(|c| c.name.as_ref() == fname)
            {
                let void_node = self.compile_void_const();
                let idx = captured.len();
                captured.push((fname.to_string(), void_node));
                idx as i32
            } else {
                -1
            }
        } else {
            -1
        };

        for cap in &sema_captures {
            let ident = cap.name.as_ref();
            if param_names.contains(ident) {
                continue;
            }
            // Skip the self-reference name (already added as a placeholder upvalue).
            if Some(ident) == fn_name && self_upvalue_idx >= 0 {
                continue;
            }
            if let Some(node) = self.lookup_var(ident) {
                if !captured.iter().any(|(n, _)| n.as_str() == ident) {
                    // If the variable has already been captured by an outer lambda, use the outer
                    // original node.
                    // This ensures the WriteBack target points to the outermost defining node (root
                    // frame), not an intermediate lambda's upvalue parameter node (an
                    // intermediate-frame copy).
                    let outer_node = self.captured_scopes.iter().rev()
                        .find_map(|scope| scope.iter()
                            .find(|(n, _)| n.as_str() == ident)
                            .map(|(_, node)| *node))
                            .unwrap_or(node);
                    captured.push((ident.to_string(), outer_node));
                }
            }
        }

        // No Step 7: the declaring frame keeps its original variable bindings.
        // Cross-function upvalue visibility is handled at runtime: compute_closure_construct
        // wraps upvalues in Cells, compute_closure_call preserves Cell references, and
        // compute_writeback's Path 3 writes to Cell upvalues for escaped closures.
        // For defer bodies (same-function branch frames), WriteBack's frame-chain paths
        // (Path 0/1) propagate upvalue mutations within the function's frame chain.

        let param_count = (params.len() + captured.len()) as u8;

        // 2. Register a placeholder subgraph (node range is filled in after compilation).
        let sg_id = self.register_subgraph_placeholder("", param_count, is_async);
        let node_start = self.graph.nodes.len() as u32;

        // 3. Enter the lambda scope: create lambda parameter nodes first, then captured upvalue
        //    parameter nodes, and `bind_var` all of them (capture nodes shadow outer bindings of
        //    the same name within the lambda scope).
        self.enter_scope();
        for param in params {
            let inputs_offset = self.graph.inputs_pool.push(&[]);
            let param_node = self.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            });
            self.bind_var(param.name, param_node);
        }
        for (name, _outer_node) in &captured {
            let inputs_offset = self.graph.inputs_pool.push(&[]);
            let upvalue_node = self.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            });
            self.bind_var(name, upvalue_node);
        }

        // 4. Compile the body to obtain the return node.
        //    Set `current_sg_start = node_start` so `is_in_current_subgraph` correctly
        //    identifies nodes inside the lambda (including upvalue placeholder nodes), preventing
        //    captured-variable assignments from taking the local path by mistake.
        //    Reset `current_effect = None` to isolate the lambda body from the outer effect chain,
        //    ensuring that a block with no trailing expression returns the `void_const` created
        //    inside the lambda rather than an outer effect node.
        //    Push onto `captured_scopes` so that `Assignment` can recognize captured variables
        //    and create WriteBacks.

        // Escape analysis (Bug #41 + Bug #40 loop capture):
        // Consume the analyzer's unified escape table; the IR performs no parallel escape
        // analysis.
        // 1. Tail-position escape (Bug #41): a lambda at the tail position of its enclosing
        //    function -> the defining frame is destroyed after the function returns.
        // 2. Loop-body capture escape (Bug #40): a lambda captures a loop-body local variable
        //    -> after the loop-body frame is destroyed, the access reads null.
        // Both cases require allocating an independent `function_id` and taking the cross-function
        // Cell path to persist the upvalue.
        let escapes = lambda_expr_id.is_some_and(|id| {
            self.current_analysis()
                .map_or(false, |r| {
                    r.escape.lookup(id).is_some_and(|info| {
                        matches!(
                            info,
                            crate::pass::Analyzer::EscapeInfo::Escapes(
                                crate::pass::Analyzer::EscapeKind::Lambda { .. }
                            )
                        )
                    })
                })
        });

        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
        let prev_effect = self.current_effect;
        self.current_effect = None;
        // Set `current_function_sg` so defer statements can register into the lambda subgraph's
        // `defer_table`.
        // If left unset, defer bodies would be lost (`current_function_sg` would be `None` or point
        // to the outer function).
        let prev_func_sg = self.current_function_sg;
        self.current_function_sg = Some(sg_id);
        // Escaping lambdas use an independent `function_id` so that subgraphs inside the body
        // (if-else/match branches, etc.) inherit this id and are distinguished from the enclosing
        // function -> cross-function Cell path.
        let prev_func_id = self.current_function_id;
        if escapes {
            self.current_function_id = sg_id.0;
        }
        self.captured_scopes.push(captured.clone());

        // Unified entry: memoize/tail_rec/non_tail_rec apply equally to closures
        // (the lambda is not in the call_graph, so `lookup_memo_strategy` returns None -> the
        // default `compile_expr` path is taken).
        let lambda_name = fn_name.unwrap_or("");
        let return_node = self.compile_function_body(lambda_name, None, body_expr, params, false, is_async);

        self.current_sg_start = prev_sg_start;
        self.current_effect = prev_effect;
        self.current_function_sg = prev_func_sg;
        self.current_function_id = prev_func_id;
        self.captured_scopes.pop();
        self.exit_scope();

        // 5. Update the subgraph's node_range + return_node + function_id + upvalue metadata.
        // `function_id`: escaping lambdas use an independent id (`sg_id.0`); non-escaping lambdas
        // inherit the outer id.
        // - Escaping: `same_function=false` -> cross-function Cell path (defining frame destroyed;
        //   Cell persists the upvalue).
        // - Non-escaping: `same_function=true` -> frame-chain path (defining frame alive; shared
        //   state).
        // `upvalue_count` + `upvalue_outer_nodes` are used by `start_subgraph` to inject the
        // current parent-frame values on `same_function` calls (capture-by-reference semantics).
        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = return_node;
        sg.has_suspend = is_async;
        sg.function_id = if escapes { sg_id.0 } else { prev_func_id };
        sg.upvalue_count = captured.len() as u8;
        sg.upvalue_outer_nodes = captured.iter().map(|(_, n)| *n).collect();

        // Register captured variables into `captured_vars`: when an outer `Assignment` assigns to
        // one of these variables, it must emit a WriteBack to the original node, so a
        // `same_function` closure call reads the latest value.
        for (name, node) in &captured {
            self.captured_vars.entry(name.clone()).or_insert(*node);
        }

        // 6. Create the closure construct node in the current scope (inputs = captured outer
        //    nodes, `compute_fn` 40).
        let upvalue_nodes: Vec<NodeId> = captured.iter().map(|(_, n)| *n).collect();
        let inputs_offset = self.graph.inputs_pool.push(&upvalue_nodes);
        let construct_node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: upvalue_nodes.len() as u8,
            inputs_offset,
            compute_fn: CF_CLOSURE_CONSTRUCT, // compute_closure_construct
        });
        self.graph.set_closure_info(
            construct_node,
            ClosureInfo {
                subgraph_id: sg_id,
                arity: params.len() as u8,
                self_upvalue_idx,
            },
        );
        construct_node
    }

    /// Compile an `inline_trait` expression: each method is compiled into a subgraph (with upvalue
    /// captures), and a TraitValue construct node (`compute_fn=266`) is created that at runtime
    /// packs the closures together.
    ///
    /// Method-subgraph compilation mirrors `compile_lambda`: free-variable analysis ->
    /// placeholder subgraph -> enter scope (params + upvalues) -> compile body -> fill
    /// `node_range`.
    /// The upvalues of all methods are concatenated in order as the construct node's inputs.
    fn compile_inline_trait(&mut self, expr_id: crate::ast::Ast::ExprId, methods: &[crate::ast::Ast::MethodDecl<'_>]) -> NodeId {
        // Infer the trait name (from `sema.expr_types` as `Type::TraitObject`).
        let trait_name = self.expr_type_name(expr_id)
            .map(|s| s.to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        // Reorder the `inline_trait` methods by the declaration order in `trait_def.methods`,
        // ensuring `method_values[i]` corresponds to `trait_def.methods[i]`, so the vtable can
        // dispatch by `method_idx` positional index (Task 10).
        let method_order: Vec<usize> = if let Some(trait_def) = self.sema.get_trait_def(&trait_name) {
            let mut order: Vec<usize> = (0..methods.len()).collect();
            order.sort_by_key(|&i| {
                trait_def.methods.iter().position(|tm| tm.name.as_ref() == methods[i].name)
                    .unwrap_or(usize::MAX)
            });
            order
        } else {
            (0..methods.len()).collect()
        };

        let mut method_names: Vec<String> = Vec::with_capacity(methods.len());
        let mut method_entries: Vec<TraitMethodEntry> = Vec::with_capacity(methods.len());
        let mut all_upvalue_nodes: Vec<NodeId> = Vec::new();

        for &idx in &method_order {
            let m = &methods[idx];
            let body_expr = match m.body {
                Some(b) => b,
                None => {
                    self.errors.push(format!(
                        "compile_inline_trait: method {} has no body (inline_trait requires all methods to have bodies)",
                        m.name
                    ));
                    continue;
                }
            };

            // 1. Free-variable analysis: collect outer variables referenced in the body (excluding
            //    the method's own parameters).
            let param_names: rustc_hash::FxHashSet<&str> =
                m.params.iter().map(|p| p.name).collect();
            let mut ident_names: Vec<String> = Vec::new();
            self.collect_free_idents_expr(body_expr, &mut ident_names);
            let mut captured: Vec<(String, NodeId)> = Vec::new();
            for name in &ident_names {
                if param_names.contains(name.as_str()) {
                    continue;
                }
                if let Some(node) = self.lookup_var(name) {
                    if !captured.iter().any(|(n, _)| n == name) {
                        captured.push((name.clone(), node));
                    }
                }
            }

            let param_count = (m.params.len() + captured.len()) as u8;

            // 2. Register a placeholder subgraph.
            let sg_id = self.register_subgraph_placeholder("", param_count, m.is_async);
            let node_start = self.graph.nodes.len() as u32;

            // 3. Enter the method scope: parameter nodes + upvalue nodes.
            self.enter_scope();
            for param in &m.params {
                let inputs_offset = self.graph.inputs_pool.push(&[]);
                let param_node = self.graph.add_node(Node {
                    kind: NodeKind::Const,
                    input_count: 0,
                    inputs_offset,
                    compute_fn: CF_NOOP,
                });
                self.bind_var(param.name, param_node);
            }
            for (name, _outer_node) in &captured {
                let inputs_offset = self.graph.inputs_pool.push(&[]);
                let upvalue_node = self.graph.add_node(Node {
                    kind: NodeKind::Const,
                    input_count: 0,
                    inputs_offset,
                    compute_fn: CF_NOOP,
                });
                self.bind_var(name, upvalue_node);
            }

            // 4. Compile the method body.
            //    Set `current_sg_start` + reset `current_effect` + push `captured_scopes`
            //    (consistent with `compile_lambda`).
            let prev_sg_start = self.current_sg_start;
            self.current_sg_start = node_start;
            let prev_effect = self.current_effect;
            self.current_effect = None;
            self.captured_scopes.push(captured.clone());

            let return_node = self.compile_expr(body_expr);

            self.current_sg_start = prev_sg_start;
            self.current_effect = prev_effect;
            self.captured_scopes.pop();
            self.exit_scope();

            // 5. Fill in the subgraph's `node_range` + upvalue metadata.
            let node_end = self.graph.nodes.len() as u32;
            let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
            sg.node_range = (NodeId(node_start), NodeId(node_end));
            sg.entry_node = NodeId(node_start);
            sg.return_node = return_node;
            sg.has_suspend = m.is_async;
            sg.upvalue_count = captured.len() as u8;
            sg.upvalue_outer_nodes = captured.iter().map(|(_, n)| *n).collect();

            // 6. Collect upvalue nodes + record method info.
            let upvalue_count = captured.len() as u8;
            for (_, n) in &captured {
                all_upvalue_nodes.push(*n);
            }
            method_names.push(m.name.to_string());
            method_entries.push(TraitMethodEntry {
                subgraph_id: sg_id,
                arity: m.params.len() as u8,
                upvalue_count,
            });
        }

        // Construct the TraitValue construct node (inputs = the concatenated upvalues of all
        // methods).
        let inputs_offset = self.graph.inputs_pool.push(&all_upvalue_nodes);
        let construct_node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: all_upvalue_nodes.len() as u8,
            inputs_offset,
            compute_fn: CF_TRAIT_CONSTRUCT, // compute_trait_construct
        });
        self.graph.set_trait_construct_info(
            construct_node,
            TraitConstructInfo {
                trait_name,
                method_names,
                methods: method_entries,
            },
        );
        construct_node
    }

    /// Compile a `lazy` expression: the operand is compiled into a zero-parameter thunk
    /// subgraph, and a LazyValue construct node (`compute_fn=267`) is created that at runtime
    /// produces an unevaluated LazyValue.
    ///
    /// The thunk subgraph captures outer free variables (the same capture mechanism as for
    /// lambdas). On the first force it launches the subgraph to compute the value; the result is
    /// cached for reuse by subsequent forces.
    fn compile_lazy(&mut self, expr_id: crate::ast::Ast::ExprId, operand: crate::ast::Ast::ExprId) -> NodeId {
        let _ = expr_id; // trait_name inference is not yet needed; the parameter is retained for
                         // future force semantics.
        // 1. Free-variable analysis.
        let mut ident_names: Vec<String> = Vec::new();
        self.collect_free_idents_expr(operand, &mut ident_names);
        let mut captured: Vec<(String, NodeId)> = Vec::new();
        for name in &ident_names {
            if let Some(node) = self.lookup_var(name) {
                if !captured.iter().any(|(n, _)| n == name) {
                    captured.push((name.clone(), node));
                }
            }
        }

        let param_count = captured.len() as u8;

        // 2. Register a placeholder subgraph (thunk: no explicit parameters, only upvalues).
        let sg_id = self.register_subgraph_placeholder("", param_count, false);
        let node_start = self.graph.nodes.len() as u32;

        // 3. Enter the thunk scope: upvalue nodes.
        self.enter_scope();
        for (name, _outer_node) in &captured {
            let inputs_offset = self.graph.inputs_pool.push(&[]);
            let upvalue_node = self.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            });
            self.bind_var(name, upvalue_node);
        }

        // 4. Compile the operand to obtain the return node.
        let return_node = self.compile_expr(operand);
        self.exit_scope();

        // 5. Fill in the subgraph's `node_range`.
        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = return_node;
        sg.has_suspend = false;

        // 6. Construct the LazyValue construct node (inputs = upvalues).
        let upvalue_nodes: Vec<NodeId> = captured.iter().map(|(_, n)| *n).collect();
        let inputs_offset = self.graph.inputs_pool.push(&upvalue_nodes);
        let construct_node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: upvalue_nodes.len() as u8,
            inputs_offset,
            compute_fn: CF_LAZY_CONSTRUCT, // compute_lazy_construct
        });
        self.graph.set_lazy_construct_info(
            construct_node,
            LazyConstructInfo { thunk_sg: sg_id },
        );
        construct_node
    }

    /// Recursively collect all Ident names in an expression (deduplicated, preserving
    /// first-occurrence order).
    ///
    /// A simplified free-variable analysis: traverse common Expr variants collecting identifier
    /// references; the caller excludes lambda parameters and checks outer-scope bindings.
    fn collect_free_idents_expr(&self, expr_id: crate::ast::Ast::ExprId, names: &mut Vec<String>) {
        use crate::ast::Ast::LambdaBody;
        let spanned = self.current_module().arena.expr(expr_id);
        match &spanned.node {
            crate::ast::Ast::Expr::Ident(name) => {
                if !names.iter().any(|n| n == name) {
                    names.push((*name).to_string());
                }
            }
            crate::ast::Ast::Expr::Binary { lhs, rhs, .. } => {
                self.collect_free_idents_expr(*lhs, names);
                self.collect_free_idents_expr(*rhs, names);
            }
            crate::ast::Ast::Expr::Unary { operand, .. } => {
                self.collect_free_idents_expr(*operand, names);
            }
            crate::ast::Ast::Expr::As { expr, .. } => {
                self.collect_free_idents_expr(*expr, names);
            }
            crate::ast::Ast::Expr::Call { callee, args, .. } => {
                self.collect_free_idents_expr(*callee, names);
                for &a in args {
                    self.collect_free_idents_expr(a, names);
                }
            }
            crate::ast::Ast::Expr::MethodCall { recv, args, .. } => {
                self.collect_free_idents_expr(*recv, names);
                for &a in args {
                    self.collect_free_idents_expr(a, names);
                }
            }
            crate::ast::Ast::Expr::FieldAccess { recv, .. }
            | crate::ast::Ast::Expr::SafeAccess { recv, .. } => {
                self.collect_free_idents_expr(*recv, names);
            }
            crate::ast::Ast::Expr::Index { recv, index } => {
                self.collect_free_idents_expr(*recv, names);
                self.collect_free_idents_expr(*index, names);
            }
            crate::ast::Ast::Expr::Assign { target, value } => {
                self.collect_free_idents_expr(*target, names);
                self.collect_free_idents_expr(*value, names);
            }
            crate::ast::Ast::Expr::CompoundAssign { target, value, .. } => {
                self.collect_free_idents_expr(*target, names);
                self.collect_free_idents_expr(*value, names);
            }
            crate::ast::Ast::Expr::RecordLit(fields) => {
                for f in fields {
                    self.collect_free_idents_expr(f.value, names);
                }
            }
            crate::ast::Ast::Expr::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.collect_free_idents_expr(*cond, names);
                self.collect_free_idents_expr(*then_branch, names);
                if let Some(e) = else_branch {
                    self.collect_free_idents_expr(*e, names);
                }
            }
            crate::ast::Ast::Expr::Block { stmts, trailing } => {
                for &s in stmts {
                    self.collect_free_idents_stmt(s, names);
                }
                if let Some(t) = trailing {
                    self.collect_free_idents_expr(*t, names);
                }
            }
            crate::ast::Ast::Expr::Lambda { body, .. } => {
                let inner = match body {
                    LambdaBody::Block(e) | LambdaBody::Expression(e) => *e,
                };
                self.collect_free_idents_expr(inner, names);
            }
            crate::ast::Ast::Expr::Match { scrutinee, arms } => {
                self.collect_free_idents_expr(*scrutinee, names);
                for arm in arms {
                    if let Some(g) = arm.guard {
                        self.collect_free_idents_expr(g, names);
                    }
                    self.collect_free_idents_expr(arm.body, names);
                }
            }
            // Single-operand expressions: RefOf/Deref/Propagate/NonNullAssert/Atomic/Lazy.
            crate::ast::Ast::Expr::RefOf(inner)
            | crate::ast::Ast::Expr::Deref(inner)
            | crate::ast::Ast::Expr::Propagate(inner)
            | crate::ast::Ast::Expr::NonNullAssert(inner)
            | crate::ast::Ast::Expr::Atomic(inner)
            | crate::ast::Ast::Expr::Lazy(inner) => {
                self.collect_free_idents_expr(*inner, names);
            }
            // Elvis: `lhs ?: rhs`.
            crate::ast::Ast::Expr::Elvis { lhs, rhs } => {
                self.collect_free_idents_expr(*lhs, names);
                self.collect_free_idents_expr(*rhs, names);
            }
            // Slice: `recv[start..end]` (`inclusive` does not affect ident collection).
            crate::ast::Ast::Expr::Slice { recv, start, end, .. } => {
                self.collect_free_idents_expr(*recv, names);
                self.collect_free_idents_expr(*start, names);
                self.collect_free_idents_expr(*end, names);
            }
            // Safe method call: `recv?.method(args)`.
            crate::ast::Ast::Expr::SafeMethodCall { recv, args, .. } => {
                self.collect_free_idents_expr(*recv, names);
                for &a in args {
                    self.collect_free_idents_expr(a, names);
                }
            }
            // Record extension: `{ base with x: 1, ... }`.
            crate::ast::Ast::Expr::RecordExtend { base, updates } => {
                self.collect_free_idents_expr(*base, names);
                for f in updates {
                    self.collect_free_idents_expr(f.value, names);
                }
            }
            // Array literal `fill` clause: `[value, ..count]`.
            crate::ast::Ast::Expr::ArrayLit { elements, fill } => {
                for &e in elements {
                    self.collect_free_idents_expr(e, names);
                }
                if let Some((v, c)) = fill {
                    self.collect_free_idents_expr(*v, names);
                    self.collect_free_idents_expr(*c, names);
                }
            }
            // String interpolation: may contain `{expr}`.
            crate::ast::Ast::Expr::StrInterp(parts) => {
                for part in parts {
                    if let crate::ast::Ast::InterpolationPart::Expression(e) = part {
                        self.collect_free_idents_expr(*e, names);
                    }
                }
            }
            // select expression: each branch contains channel_expr/duration + body.
            crate::ast::Ast::Expr::Select(arms) => {
                for arm in arms {
                    match arm {
                        crate::ast::Ast::SelectArm::Receive { channel_expr, body, .. } => {
                            self.collect_free_idents_expr(*channel_expr, names);
                            self.collect_free_idents_expr(*body, names);
                        }
                        crate::ast::Ast::SelectArm::Timeout { duration, body } => {
                            self.collect_free_idents_expr(*duration, names);
                            self.collect_free_idents_expr(*body, names);
                        }
                    }
                }
            }
            // inline_trait: method bodies may reference outer variables.
            crate::ast::Ast::Expr::InlineTrait(methods) => {
                for m in methods {
                    if let Some(body_expr) = m.body {
                        self.collect_free_idents_expr(body_expr, names);
                    }
                }
            }
            // Constant / no-subexpression variants: IntLit/FloatLit/BoolLit/CharLit/StrLit/NullLit/VoidLit.
            _ => {}
        }
    }

    /// Recursively collect Ident names in a statement (statement version of
    /// `collect_free_idents_expr`).
    fn collect_free_idents_stmt(&self, stmt_id: crate::ast::Ast::StmtId, names: &mut Vec<String>) {
        let spanned = self.current_module().arena.stmt(stmt_id);
        match &spanned.node {
            crate::ast::Ast::Stmt::ValDecl { value, .. }
            | crate::ast::Ast::Stmt::VarDecl { value, .. } => {
                self.collect_free_idents_expr(*value, names);
            }
            crate::ast::Ast::Stmt::Expression { expr } => {
                self.collect_free_idents_expr(*expr, names);
            }
            crate::ast::Ast::Stmt::Assignment { target, value } => {
                self.collect_free_idents_expr(*target, names);
                self.collect_free_idents_expr(*value, names);
            }
            crate::ast::Ast::Stmt::FieldAssignment { object, value, .. } => {
                self.collect_free_idents_expr(*object, names);
                self.collect_free_idents_expr(*value, names);
            }
            crate::ast::Ast::Stmt::CompoundAssignment { target, value, .. } => {
                self.collect_free_idents_expr(*target, names);
                self.collect_free_idents_expr(*value, names);
            }
            crate::ast::Ast::Stmt::Return { value } => {
                if let Some(v) = value {
                    self.collect_free_idents_expr(*v, names);
                }
            }
            crate::ast::Ast::Stmt::Throw { expr } => {
                self.collect_free_idents_expr(*expr, names);
            }
            crate::ast::Ast::Stmt::For { iterable, body, .. } => {
                self.collect_free_idents_expr(*iterable, names);
                self.collect_free_idents_expr(*body, names);
            }
            crate::ast::Ast::Stmt::While { condition, body } => {
                self.collect_free_idents_expr(*condition, names);
                self.collect_free_idents_expr(*body, names);
            }
            crate::ast::Ast::Stmt::Loop { body } => {
                self.collect_free_idents_expr(*body, names);
            }
            crate::ast::Ast::Stmt::Defer { expr } => {
                self.collect_free_idents_expr(*expr, names);
            }
            crate::ast::Ast::Stmt::Break | crate::ast::Ast::Stmt::Continue => {}
            crate::ast::Ast::Stmt::LocalDecl { decl } => match decl.as_ref() {
                crate::ast::Ast::Decl::FunDecl { body, .. } => {
                    self.collect_free_idents_expr(*body, names);
                }
                _ => {}
            },
        }
    }

    /// Create a sequence node: after `prev_effect` completes, returns `current_node`'s value.
    ///
    /// Used for statement-order chaining: ensures nodes depending on `current_node` execute only
    /// after `prev_effect` completes.
    /// `compute_seq` (idx 48) takes all inputs and returns the value of the last input.
    fn chain_effects(&mut self, prev: Option<NodeId>, current: NodeId) -> NodeId {
        match prev {
            Some(prev_node) => {
                let off = self.graph.inputs_pool.push(&[prev_node, current]);
                self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: off,
                    compute_fn: CF_SEQ, // compute_seq
                })
            }
            None => current,
        }
    }

    /// Create a Call node targeting `target_sg` (no input dependency; immediately ready).
    ///
    /// Used for the initial call of a loop and for `continue` jumps.
    fn compile_recursive_call(&mut self, target_sg: SubGraphId) -> NodeId {
        // Append `current_effect` as an implicit dependency (consistent with `compile_call`),
        // ensuring the while/loop recursive Call executes only after prior statements (e.g. an
        // array literal) complete.
        // Otherwise the Call node has no input dependency and could launch the subgraph frame
        // before the `arr` value is ready, preventing the frame from copying `arr`'s Ref value
        // and causing `arr[0]` inside the loop body to return `<non-scalar>`.
        let mut inputs: Vec<NodeId> = Vec::new();
        if let Some(eff) = self.current_effect {
            inputs.push(eff);
        }
        let input_count = inputs.len() as u8;
        let inputs_offset = self.graph.inputs_pool.push(&inputs);
        let call_node = self.graph.add_node(Node {
            kind: NodeKind::Call,
            input_count,
            inputs_offset,
            compute_fn: CF_CALL_LAUNCH,
        });
        self.graph.set_call_target(call_node, target_sg);
        call_node
    }

    /// Create a Call node targeting `target_sg`, passing the given argument nodes.
    fn make_call(&mut self, target_sg: SubGraphId, arg_nodes: &[NodeId]) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(arg_nodes);
        let call_node = self.graph.add_node(Node {
            kind: NodeKind::Call,
            input_count: arg_nodes.len() as u8,
            inputs_offset,
            compute_fn: CF_CALL_LAUNCH,
        });
        self.graph.set_call_target(call_node, target_sg);
        call_node
    }

    /// Create a Call node targeting a function name (looked up via `func_subgraphs`).
    fn make_call_by_name(&mut self, name: &str, arg_nodes: &[NodeId]) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(arg_nodes);
        let call_node = self.graph.add_node(Node {
            kind: NodeKind::Call,
            input_count: arg_nodes.len() as u8,
            inputs_offset,
            compute_fn: CF_CALL_LAUNCH,
        });
        if let Some(&target_sg) = self.func_subgraphs.get(name) {
            self.graph.set_call_target(call_node, target_sg);
        }
        call_node
    }

    /// Create a vtable dynamic-dispatch Call node (method call on a trait value).
    ///
    /// Unlike `make_call_by_name`: the target subgraph id is looked up at runtime from the
    /// TraitVal's vtable rather than bound at compile time. Used when a For-loop iterable is a
    /// trait value (`Iterator<T>`).
    fn make_vtable_call(&mut self, recv_node: NodeId, trait_name: &str, method_name: &str) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(&[recv_node]);
        let call_node = self.graph.add_node(Node {
            kind: NodeKind::Call,
            input_count: 1,
            inputs_offset,
            compute_fn: CF_CALL_LAUNCH,
        });
        let method_idx = self.sema.get_trait_def(trait_name)
            .and_then(|td| td.methods.iter().position(|m| m.name.as_ref() == method_name))
            .map(|i| i as u16);
        match method_idx {
            Some(idx) => self.graph.set_vtable_call(call_node, idx),
            None => self.errors.push(format!(
                "internal: trait method '{}' not found in trait '{}' for vtable dispatch",
                method_name, trait_name)),
        }
        call_node
    }

    /// Look up an expression's type information from sema (used for For-loop dispatch
    /// decisions).
    /// Returns `(type name, is_trait_object)`:
    /// - `(Some("RangeIterator"), false)` -> static dispatch to `"RangeIterator.next"`.
    /// - `(Some("Iterator"), true)` -> vtable dynamic dispatch (an inline_trait value).
    /// - `(None, false)` -> type inference failed; fall back to vtable dispatch.
    fn lookup_expr_iter_info(&self, expr: crate::ast::Ast::ExprId) -> (Option<String>, bool) {
        let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), expr.0 as u64);
        if let Some(info) = self.sema.expr_types.get(&key) {
            return (info.type_name.as_deref().map(|s| s.to_string()), info.is_trait_object);
        }
        (None, false)
    }

    /// Register the For-loop subgraph (a recursive subgraph; `param_count=1` to receive the
    /// iterator).
    ///
    /// Structure:
    /// - `for_sg` (`param_count=1`): receives the iterator.
    ///   - `param_0` = iterator.
    ///   - `next_call` = `Call("Iterator.next", [param_0])` // returns `T?`.
    ///   - `is_null_node` = `UnOp(is_null, [next_call])`.
    ///   - `body_sg` (`param_count=2`): iterator + current value (bind name, compile body,
    ///     tail-recurses into `for_sg`).
    ///   - `void_sg` (`param_count=0`): exit.
    ///   - `gate` = `Gate(is_null_node)`: `true` -> `void_sg` (exit), `false` -> `body_sg`
    ///     (continue).
    ///
    /// Execution: `next()` returns non-null -> `body_sg` executes and then tail-recurses into
    /// `for_sg`; returns null -> `void_sg` exits.
    /// A Break signal terminates the `body_sg` frame -> the Gate completes -> `for_sg` ends.
    /// Continue is compiled as `Call(for_sg, [iter_param]) + Return` signal -> tail-recurses
    /// into the next iteration.
    fn register_for_subgraph(
        &mut self,
        name: &str,
        body: crate::ast::Ast::ExprId,
        iter_type_name: Option<&str>,
        is_trait_object: bool,
    ) -> SubGraphId {
        let node_start = self.graph.nodes.len() as u32;
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        // Placeholder registration (reserve the id first to allow recursive references).
        self.graph.add_subgraph(SubGraph {
            id: sg_id,
            node_range: (NodeId(node_start), NodeId(node_start)),
            param_count: 1,
            entry_node: NodeId(node_start),
            return_node: NodeId(node_start),
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });

        // param_0 = iterator.
        let iter_off = self.graph.inputs_pool.push(&[]);
        let iter_param = self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset: iter_off,
            compute_fn: CF_NOOP,
        });

        // next_call = Call("{iter_type_name}.next", [iter_param]) -> T?
        // Dynamic dispatch: when `is_trait_object=true`, go through the vtable (look up `next` at
        // runtime from the TraitVal).
        // Static dispatch: concrete types are bound by mangled type name (e.g. "ArrayIter.next").
        // Fallback: type inference failed (None) -> vtable dispatch.
        let next_call = if is_trait_object || iter_type_name.is_none() {
            let trait_name = iter_type_name.as_deref().unwrap_or("Iterator");
            self.make_vtable_call(iter_param, trait_name, "next")
        } else {
            let next_method = format!("{}.next", iter_type_name.unwrap());
            self.make_call_by_name(&next_method, &[iter_param])
        };

        // is_null_node = UnOp(is_null, [next_call]).
        let is_null_off = self.graph.inputs_pool.push(&[next_call]);
        let is_null_node = self.graph.add_node(Node {
            kind: NodeKind::UnOp,
            input_count: 1,
            inputs_offset: is_null_off,
            compute_fn: CF_IS_NULL, // is_null
        });

        // body_sg (param_count=2: iterator + current value).
        // Reset `current_effect = None` (same as `register_while_subgraph`, to avoid an
        // external effect dependency causing a deadlock after `reset_loop_iteration` of the
        // loop-body frame).
        let prev_effect = self.current_effect;
        self.current_effect = None;
        let body_sg = self.compile_for_body_subgraph(body, sg_id, name);

        // void_sg (exit): includes CF_DEFER_RUN to drain defer-in-loop entries at loop exit.
        let void_sg = self.compile_defer_run_subgraph();
        self.current_effect = prev_effect;

        // gate = Gate(is_null_node): true -> void_sg, false -> body_sg (inputs=[iter_param,
        // next_call]).
        let gate_off = self.graph.inputs_pool.push(&[is_null_node]);
        let gate_node = self.graph.add_node(Node {
            kind: NodeKind::Gate,
            input_count: 1,
            inputs_offset: gate_off,
            compute_fn: CF_GATE_LAUNCH,
        });
        self.graph.set_gate_branches(
            gate_node,
            GateBranches {
                condition_input: is_null_node,
                branches: vec![
                    (true, void_sg, vec![]),
                    (false, body_sg, vec![iter_param, next_call]),
                ],
            },
        );

        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = gate_node;
        sg.loop_kind = LoopKind::For;
        sg.cond_node = Some(is_null_node);
        sg.iter_next_node = Some(next_call);
        sg.reset_plan = Some(ResetPlan {
            reset_to_zero: vec![next_call],
            reset_to_one: vec![is_null_node],
            reset_condition_tree: vec![],
        });
        sg_id
    }

    /// Compile the For-loop body subgraph (`param_count=2`: iterator + current value).
    ///
    /// - `param_0` = iterator (for tail recursion).
    /// - `param_1` = current value (bound to the loop variable `name`).
    /// - Compiles `body`; at the end emits a tail-recursive `Call(for_sg, [param_0])` (depends
    ///   on `body_last` to preserve ordering).
    fn compile_for_body_subgraph(
        &mut self,
        body: crate::ast::Ast::ExprId,
        for_sg: SubGraphId,
        name: &str,
    ) -> SubGraphId {
        let node_start = self.graph.nodes.len() as u32;

        // param_0 = iterator (a node inside body_sg, injected by the Gate branch inputs).
        let iter_off = self.graph.inputs_pool.push(&[]);
        let iter_param = self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset: iter_off,
            compute_fn: CF_NOOP,
        });

        // param_1 = current value (bound to the loop variable `name`).
        let val_off = self.graph.inputs_pool.push(&[]);
        let val_param = self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset: val_off,
            compute_fn: CF_NOOP,
        });

        self.enter_scope();
        self.bind_var(name, val_param);
        self.loop_stack.push(LoopContext {
            sg: for_sg,
            iter_node: Some(iter_param),
            body_node_start: node_start,
            loop_var_node: Some(val_param),
        });

        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
        let prev_in_loop = self.in_loop_body;
        self.in_loop_body = true;
        let body_last = self.compile_expr(body);
        self.in_loop_body = prev_in_loop;
        self.current_sg_start = prev_sg_start;

        self.loop_stack.pop();
        self.exit_scope();

        // Tail-recursion eliminated: `return_node = body_last`; frame reuse is handled by the
        // Engine's `reset_loop_iteration`.
        let node_end = self.graph.nodes.len() as u32;
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            id: sg_id,
            node_range: (NodeId(node_start), NodeId(node_end)),
            param_count: 2,
            entry_node: NodeId(node_start),
            return_node: body_last,
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: LoopKind::LoopBody,
            loop_parent_sg: Some(for_sg),
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });
        sg_id
    }

    /// Register the While-loop subgraph (a recursive subgraph).
    ///
    /// Structure:
    /// - `cond_node` = `compile_expr(condition)`.
    /// - `gate_node` = `Gate(cond)`: `true` -> `body_sg` (tail recursion), `false` -> `void_sg`
    ///   (exit).
    /// - `body_sg`: compiles `body`; at the end emits a `Call` back to `while_sg` (depends on
    ///   the body's trailing node to preserve ordering).
    ///
    /// Execution: when `cond` is true, `body_sg` executes and then tail-recurses into
    /// `while_sg`; when false, `void_sg` exits.
    /// A Break signal terminates the `body_sg` frame -> `while_sg`'s Gate completes -> the loop
    /// ends.
    /// Continue is compiled as `Call(while_sg) + Return` signal -> tail-recurses into the next
    /// iteration (skipping the rest of the body).
    fn register_while_subgraph(
        &mut self,
        condition: crate::ast::Ast::ExprId,
        body: crate::ast::Ast::ExprId,
    ) -> SubGraphId {
        let node_start = self.graph.nodes.len() as u32;
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        // Placeholder registration (reserve the id first to allow recursive references).
        self.graph.add_subgraph(SubGraph {
            id: sg_id,
            node_range: (NodeId(node_start), NodeId(node_start)),
            param_count: 0,
            entry_node: NodeId(node_start),
            return_node: NodeId(node_start),
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });

        // Compile the condition.
        // Reset `current_effect = None` to avoid creating CF_SEQ nodes inside the loop subgraph
        // that depend on the external effect chain.
        // After `reset_loop_iteration`, the loop-body frame's value table is cleared and external
        // effect nodes are not re-copied; CF_SEQ nodes depending on them would stay pending
        // forever, causing a deadlock.
        let prev_effect = self.current_effect;
        self.current_effect = None;
        let cond_node = self.compile_subexpr(condition);
        // body subgraph (trailing tail-recursive call to while_sg).
        let body_sg = self.compile_loop_body_subgraph(body, sg_id);
        // void subgraph (false branch; loop ends): includes CF_DEFER_RUN for defer-in-loop.
        let void_sg = self.compile_defer_run_subgraph();
        self.current_effect = prev_effect;

        // Gate node: cond true -> body_sg, false -> void_sg.
        let gate_off = self.graph.inputs_pool.push(&[cond_node]);
        let gate_node = self.graph.add_node(Node {
            kind: NodeKind::Gate,
            input_count: 1,
            inputs_offset: gate_off,
            compute_fn: CF_GATE_LAUNCH,
        });
        self.graph.set_gate_branches(
            gate_node,
            GateBranches {
                condition_input: cond_node,
                branches: vec![
                    (true, body_sg, vec![]),
                    (false, void_sg, vec![]),
                ],
            },
        );

        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = gate_node;
        sg.loop_kind = LoopKind::While;
        sg.cond_node = Some(cond_node);
        sg.reset_plan = Some(ResetPlan {
            reset_to_zero: vec![],
            reset_to_one: vec![],
            reset_condition_tree: vec![cond_node],
        });
        sg_id
    }

    /// Look up the memo strategy for the current function/method (generic entry point, unified
    /// `FuncId` query).
    ///
    /// Obtains the `FuncId` by looking up the mangled name in `CallGraph.name_to_func`:
    /// - FunDecl / lambda / monomorph instance: `self_type = None`, mangled = name.
    /// - Method: `self_type = Some(type_name)`, mangled = `"{type_name}.{name}"`.
    ///
    /// The mangled-name format matches how methods are registered in `build_call_graph`
    /// (`"Type.method"`).
    /// `memo_pass` already makes the unique decision; a function has at most one strategy.
    fn lookup_memo_strategy(
        &self,
        name: &str,
        self_type: Option<&str>,
    ) -> Option<crate::pass::Analyzer::MemoStrategy> {
        let report = self.current_analysis()?;
        let mangled: String = match self_type {
            Some(t) => format!("{}.{}", t, name),
            None => name.to_string(),
        };
        let func_id = *report.call_graph.name_to_func.get(&mangled)?;
        report.memo.candidates.iter()
            .find(|c| c.func == func_id)
            .map(|c| c.strategy.clone())
    }

    /// Unified function-body compilation entry point: queries the memo strategy and dispatches
    /// to the corresponding compilation path.
    ///
    /// All function compilation entry points (`compile_function` /
    /// `compile_monomorph_instance` / `compile_builtin_method` / `compile_user_method` /
    /// `compile_lambda`) call this method, ensuring memoize / tail_rec / non_tail_rec
    /// optimizations apply uniformly to FunDecls, methods, lambdas, and monomorph instances.
    ///
    /// `self_type`: pass `Some(type_name)` for methods, `None` otherwise. Used to construct the
    /// mangled name for the `FuncId` lookup.
    /// Precondition: the caller has set `current_sg_start = node_start` (`compile_memoize`
    /// relies on this value to compute parameter node ids = `current_sg_start + param_index`).
    ///
    /// `is_async`: whether the enclosing function is declared `async`. When true and the body
    /// expression's inferred type is `Async<T>` (Bug #79), an implicit await node is inserted so
    /// the function returns the resolved `T` rather than the raw async handle. This implements
    /// transparent async forwarding — `async fun f(): Async<T> { g() }` where `g(): Async<T>`
    /// automatically awaits `g()` and returns its result.
    fn compile_function_body(
        &mut self,
        name: &str,
        self_type: Option<&str>,
        body_expr: crate::ast::Ast::ExprId,
        params: &[crate::ast::Ast::Param<'_>],
        is_void_fn: bool,
        is_async: bool,
    ) -> NodeId {
        let prev_tail = self.in_tail_position;
        self.in_tail_position = !is_void_fn;
        // Bug #66: Mark that the next compile_block call is the function body's top-level block.
        // compile_block reads and resets this flag so that only nested blocks extract
        // block-scoped defers; function-level defers stay in defer_table for function-exit execution.
        let prev_top_block = self.in_function_top_block;
        self.in_function_top_block = true;
        // Unified memo-strategy query (memo_pass already makes the unique decision; mutually
        // exclusive).
        let strategy = self.lookup_memo_strategy(name, self_type);
        let r = match strategy {
            Some(crate::pass::Analyzer::MemoStrategy::TailRecToLoop { info }) => {
                self.compile_tail_rec_to_loop(name, body_expr, params, &info)
            }
            Some(crate::pass::Analyzer::MemoStrategy::NonTailRecToLoop { info }) => {
                self.compile_non_tail_rec_to_loop(name, body_expr, params, &info)
            }
            Some(crate::pass::Analyzer::MemoStrategy::Memoize { cache_key, .. }) => {
                self.compile_memoize(name, body_expr, params, &cache_key)
            }
            _ => {
                let node = self.compile_expr(body_expr);
                // Bug #79: Auto-await forwarding. If this is an async function and the body
                // expression's inferred type is Async<T>, the body produces an async handle (a
                // raw i32 async_id) rather than the resolved value T. Insert an implicit await
                // node to resolve it, so the function returns T (which the runtime wraps back
                // into Async<T> for the caller). Sema already validates type compatibility
                // (unify_return_type handles Async<X> vs Async<Y> by unifying inner types).
                if is_async && self.expr_type_is_async(body_expr) {
                    self.build_await_node(body_expr, node)
                } else {
                    node
                }
            }
        };
        self.in_tail_position = prev_tail;
        self.in_function_top_block = prev_top_block;
        r
    }

    /// Memoization cache: consumes the Memoize strategy and inserts a cache-check Gate at the
    /// function entry and a cache-write after the body.
    ///
    /// Called by `compile_function` when it detects `MemoStrategy::Memoize`.
    /// The parameter nodes have already been created and `bind_var`'d by `compile_function`;
    /// this method constructs the cache structure.
    ///
    /// Structure:
    /// - `memo_check` node: inputs = the parameter nodes; returns `record(hit, value)`.
    /// - `field_get(hit)` -> `Gate(hit)` for branching.
    /// - `hit=true` branch: `field_get(value)` as the return value (passthrough subgraph).
    /// - `hit=false` branch: compile the function body normally + `memo_store(args, body_result)`.
    ///
    /// Recursive calls remain ordinary Calls (on cache hit they return directly without
    /// expanding).
    fn compile_memoize(
        &mut self,
        _name: &str,
        body_expr: crate::ast::Ast::ExprId,
        _params: &[crate::ast::Ast::Param<'_>],
        cache_key: &crate::pass::Analyzer::CacheKeySpec,
    ) -> NodeId {
        // Allocate a cache-table index.
        let table_index = self.memo_table_count;
        self.memo_table_count += 1;

        // Collect the parameter nodes that participate in the cache key (by
        // `cache_key.param_indices`).
        // The parameter nodes are the first `param_count` nodes of the subgraph
        // (`current_sg_start` is the start of the function subgraph).
        let param_nodes: Vec<NodeId> = cache_key.param_indices.iter()
            .map(|&idx| {
                let node_id = self.current_sg_start + idx;
                NodeId(node_id)
            })
            .collect();
        let memo_param_count = param_nodes.len() as u8;

        // 1. Create the memo_check node: inputs = the parameter nodes.
        let check_inputs = self.graph.inputs_pool.push(&param_nodes);
        let memo_check_node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: param_nodes.len() as u8,
            inputs_offset: check_inputs,
            compute_fn: CF_MEMO_CHECK,
        });
        self.graph.set_memo_info(memo_check_node, crate::ir::Ir::MemoInfo {
            table_index,
            param_count: memo_param_count,
        });

        // 2. Extract the `hit` field from the record returned by `memo_check` (used as the Gate
        //    condition).
        let hit_inputs = self.graph.inputs_pool.push(&[memo_check_node]);
        let hit_node = self.graph.add_node(Node {
            kind: NodeKind::FieldAccess,
            input_count: 1,
            inputs_offset: hit_inputs,
            compute_fn: CF_RECORD_FIELD_GET,
        });
        self.graph.set_field_set_name(hit_node, "hit".to_string());

        // 3. hit=true branch subgraph: extract the `value` field from the record (cache hit:
        //    return the cached value directly).
        //    Uses the `compile_branch_subgraph` pattern: an independent subgraph + frame-chain
        //    passthrough to access `memo_check_node`.
        let hit_sg = {
            let node_start = self.graph.nodes.len() as u32;
            self.enter_scope();
            let prev_sg_start = self.current_sg_start;
            self.current_sg_start = node_start;
            let v_inputs = self.graph.inputs_pool.push(&[memo_check_node]);
            let value_node = self.graph.add_node(Node {
                kind: NodeKind::FieldAccess,
                input_count: 1,
                inputs_offset: v_inputs,
                compute_fn: CF_RECORD_FIELD_GET,
            });
            self.graph.set_field_set_name(value_node, "value".to_string());
            self.current_sg_start = prev_sg_start;
            self.exit_scope();
            let node_end = self.graph.nodes.len() as u32;
            let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
            self.graph.add_subgraph(crate::ir::Ir::SubGraph {
                id: sg_id,
                node_range: (NodeId(node_start), NodeId(node_end)),
                param_count: 0,
                entry_node: NodeId(node_start),
                return_node: value_node,
                has_suspend: false,
                event_source_decls: Vec::new(),
                defer_table: Vec::new(),
                loop_kind: crate::ir::Ir::LoopKind::None,
                loop_parent_sg: None,
                cond_node: None,
                function_id: self.current_function_id,
                iter_next_node: None,
                upvalue_count: 0,
                upvalue_outer_nodes: Vec::new(),
                nested_ranges: Vec::new(),
            reset_plan: None,
            });
            sg_id
        };

        // 4. hit=false branch subgraph: compile the function body normally + memo_store (cache
        //    miss: compute and write to the cache).
        //    Uses the `compile_branch_subgraph` pattern: an independent subgraph + frame-chain
        //    passthrough to access parameters and recursive calls.
        let miss_sg = {
            let node_start = self.graph.nodes.len() as u32;
            self.enter_scope();
            let prev_sg_start = self.current_sg_start;
            self.current_sg_start = node_start;
            let prev_effect = self.current_effect;
            self.current_effect = None;
            // Recursive Calls inside miss_sg are NOT tagged as tail_calls: `switch_subgraph`
            // frame reuse for a tail_call would skip the Memoize Gate structure, losing the
            // return value (the recursive Call reuses the miss_sg frame to execute the callee
            // subgraph, causing a `value_table` index mismatch -> returns null).
            // Force non-tail position so recursive Calls go through the normal Call path and
            // create a new frame, correctly returning the result.
            let prev_tail = self.in_tail_position;
            self.in_tail_position = false;
            let body_node = self.compile_expr(body_expr);
            self.in_tail_position = prev_tail;
            // memo_store: inputs = the parameter nodes + body_node.
            let mut store_inputs = param_nodes.clone();
            store_inputs.push(body_node);
            let store_off = self.graph.inputs_pool.push(&store_inputs);
            let store_node = self.graph.add_node(Node {
                kind: NodeKind::BinOp,
                input_count: store_inputs.len() as u8,
                inputs_offset: store_off,
                compute_fn: CF_MEMO_STORE,
            });
            self.graph.set_memo_info(store_node, crate::ir::Ir::MemoInfo {
                table_index,
                param_count: memo_param_count,
            });
            self.current_effect = prev_effect;
            self.current_sg_start = prev_sg_start;
            self.exit_scope();
            let node_end = self.graph.nodes.len() as u32;
            let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
            self.graph.add_subgraph(crate::ir::Ir::SubGraph {
                id: sg_id,
                node_range: (NodeId(node_start), NodeId(node_end)),
                param_count: 0,
                entry_node: NodeId(node_start),
                return_node: store_node,
                has_suspend: false,
                event_source_decls: Vec::new(),
                defer_table: Vec::new(),
                loop_kind: crate::ir::Ir::LoopKind::None,
                loop_parent_sg: None,
                cond_node: None,
                function_id: self.current_function_id,
                iter_next_node: None,
                upvalue_count: 0,
                upvalue_outer_nodes: Vec::new(),
                nested_ranges: Vec::new(),
            reset_plan: None,
            });
            sg_id
        };

        // 5. Gate node: hit true -> hit_sg (return the cached value), false -> miss_sg (compute
        //    + write to the cache).
        let prev_effect = self.current_effect;
        self.current_effect = None;
        let gate_off = self.graph.inputs_pool.push(&[hit_node]);
        let gate_node = self.graph.add_node(Node {
            kind: NodeKind::Gate,
            input_count: 1,
            inputs_offset: gate_off,
            compute_fn: CF_GATE_LAUNCH,
        });
        self.graph.set_gate_branches(
            gate_node,
            crate::ir::Ir::GateBranches {
                condition_input: hit_node,
                branches: vec![
                    (true, hit_sg, vec![]),
                    (false, miss_sg, vec![]),
                ],
            },
        );
        self.current_effect = prev_effect;
        gate_node
    }

    /// Tail-recursion to iteration: consumes `TailRecInfo` to construct the `while_sg` IR.
    ///
    /// Called by `compile_function` when it detects `MemoStrategy::TailRecToLoop`.
    /// The parameter nodes have already been created and `bind_var`'d by `compile_function`;
    /// this method constructs the loop structure.
    ///
    /// Structure:
    /// - `while_sg`: `cond = NOT(base_case condition)`, `Gate(cond)` -> `body_sg` / `exit_sg`.
    /// - `body_sg`: compiles the original function body (`tail_rec_ctx` intercepts tail calls as
    ///   `WriteBack + Call(while_sg)`).
    /// - `exit_sg`: compiles the base-case return value.
    fn compile_tail_rec_to_loop(
        &mut self,
        name: &str,
        body_expr: crate::ast::Ast::ExprId,
        params: &[crate::ast::Ast::Param<'_>],
        info: &crate::pass::Analyzer::TailRecInfo,
    ) -> NodeId {
        // 1. Collect parameter nodes (already `bind_var`'d by `compile_function`).
        let param_nodes: Vec<NodeId> = params
            .iter()
            .filter_map(|p| self.lookup_var(p.name))
            .collect();

        // 2. Placeholder-register while_sg.
        let node_start = self.graph.nodes.len() as u32;
        let while_sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            id: while_sg_id,
            node_range: (NodeId(node_start), NodeId(node_start)),
            param_count: 0,
            entry_node: NodeId(node_start),
            return_node: NodeId(node_start),
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });

        // 3. Build the loop condition `cond_node` (within while_sg's node_range).
        let cond_node = self.build_tail_rec_cond(&info.base_cases, &info.rec_branches);

        // 4. Set `tail_rec_ctx` (`compile_call` intercepts self-calls as `WriteBack +
        //    Call(while_sg)`).
        self.tail_rec_ctx = Some(TailRecCtx {
            self_name: name.to_string(),
            param_nodes,
        });

        // 5. Compile body_sg: compiles the original function body (LoopBody; after completion
        //    `reset_loop_iteration` auto-jumps back).
        //    A tail call `self(args)` is intercepted by `compile_call` as a WriteBack (no Call,
        //    no tail_call).
        //    The base-case path is also compiled, but `cond` guarantees it never executes (DCE
        //    can eliminate it).
        //    Force `in_tail_position = true`: a void function's `in_tail_position` defaults to
        //    `false` (line 5208 of `compile_function`: `!is_void_fn`), but in the body_sg of a
        //    tail-recursion transform, self-calls must be intercepted in tail position as a
        //    WriteBack; otherwise a genuine recursive Call node would be generated, causing an
        //    infinite loop (the loop condition is based on the initial parameter values, which
        //    are never updated).
        let prev_effect = self.current_effect;
        let prev_tail = self.in_tail_position;
        self.current_effect = None;
        self.in_tail_position = true;
        let body_sg = self.compile_loop_body_subgraph(body_expr, while_sg_id);
        self.in_tail_position = prev_tail;
        self.current_effect = prev_effect;

        // 6. Clear `tail_rec_ctx`.
        self.tail_rec_ctx = None;

        // 7. Compile exit_sg: compiles the base-case return value.
        //    v1 supports a single base_case: directly compile the return-value expression.
        //    For multiple base_cases, take the first one with a condition (`cond` guarantees
        //    only one holds).
        let exit_expr = info.base_cases
            .iter()
            .find(|(c, _)| c.is_some())
            .or_else(|| info.base_cases.first())
            .map(|(_, ret)| *ret)
            .unwrap_or(body_expr);
        let (exit_sg, exit_inputs) = self.compile_branch_subgraph(exit_expr);

        // 8. Gate(cond): true -> body_sg, false -> exit_sg.
        let gate_inputs: Vec<NodeId> = match self.current_effect {
            Some(eff) => vec![cond_node, eff],
            None => vec![cond_node],
        };
        let gate_off = self.graph.inputs_pool.push(&gate_inputs);
        let gate_node = self.graph.add_node(Node {
            kind: NodeKind::Gate,
            input_count: gate_inputs.len() as u8,
            inputs_offset: gate_off,
            compute_fn: CF_GATE_LAUNCH,
        });
        self.graph.set_gate_branches(
            gate_node,
            GateBranches {
                condition_input: cond_node,
                branches: vec![
                    (true, body_sg, Vec::new()),
                    (false, exit_sg, exit_inputs),
                ],
            },
        );

        // 9. Fill in while_sg metadata.
        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[while_sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = gate_node;
        sg.loop_kind = LoopKind::TailRec;
        sg.cond_node = Some(cond_node);

        // 10. Create a Call node to launch while_sg (consistent with
        //     `register_while_subgraph` + `compile_recursive_call`).
        //     while_sg runs as a `same_function` subgraph frame; after body_sg completes,
        //     `reset_loop_iteration` reads while_sg's `loop_kind=While` + `cond_node` to correctly
        //     reset the loop.
        //     If `gate_node` were returned directly, while_sg's nodes would execute in the
        //     function's main subgraph frame, where `reset_loop_iteration` would read
        //     `loop_kind=None` -> loop reset would fail.
        let call_node = self.compile_recursive_call(while_sg_id);
        call_node
    }

    /// Build the loop condition for tail-recursion-to-iteration.
    /// Rules:
    /// - Has a base_case with `Some(cond)`: `cond = AND(NOT(base_cond_i))`.
    /// - No base_case with `Some(cond)`: `cond = OR(rec_cond_i)` (implemented via De Morgan).
    /// - Neither: `cond = Const(true)` (should not happen).
    fn build_tail_rec_cond(
        &mut self,
        base_cases: &[(Option<crate::ast::Ast::ExprId>, crate::ast::Ast::ExprId)],
        rec_branches: &[(Option<crate::ast::Ast::ExprId>, Vec<crate::ast::Ast::ExprId>)],
    ) -> NodeId {
        let base_conds: Vec<crate::ast::Ast::ExprId> = base_cases
            .iter()
            .filter_map(|(c, _)| *c)
            .collect();

        if !base_conds.is_empty() {
            // cond = AND(NOT(base_cond_i)).
            let mut negated_nodes: Vec<NodeId> = Vec::new();
            for c in &base_conds {
                let cond_node = self.compile_subexpr(*c);
                let not_off = self.graph.inputs_pool.push(&[cond_node]);
                negated_nodes.push(self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: 1,
                    inputs_offset: not_off,
                    compute_fn: CF_NOT_BOOL,
                }));
            }
            let mut result = negated_nodes[0];
            for n in &negated_nodes[1..] {
                let and_off = self.graph.inputs_pool.push(&[result, *n]);
                result = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: and_off,
                    compute_fn: CF_AND_BOOL,
                });
            }
            result
        } else {
            // No base_case with `Some(cond)`: `cond = OR(rec_cond_i) = NOT(AND(NOT(rec_cond_i)))`.
            // If a rec_branch has `cond=None` (a fallback branch, e.g. the `else` of an if-else),
            // then some rec path always executes, so `cond` should be `Const(true)`, and the
            // rec/base split is dispatched by body_sg's internal Gate + Continue signal.
            let has_none_rec = rec_branches.iter().any(|(c, _)| c.is_none());
            let rec_conds: Vec<crate::ast::Ast::ExprId> = rec_branches
                .iter()
                .filter_map(|(c, _)| *c)
                .collect();
            if rec_conds.is_empty() || has_none_rec {
                // No synthesizable ExprId condition, or a fallback rec branch exists
                // (match/if-else tail recursion).
                // `cond = Const(true)`; body_sg always executes, and the rec/base split is
                // distinguished by the Continue signal:
                //   A rec arm's WriteBack sets Continue -> the loop continues;
                //   a base arm has no WriteBack -> None -> the loop exits (returns body_sg's
                //   return value).
                let off = self.graph.inputs_pool.push(&[]);
                let true_node = self.graph.add_node(Node {
                    kind: NodeKind::Const,
                    input_count: 0,
                    inputs_offset: off,
                    compute_fn: CF_NOOP,
                });
                self.graph.const_values[true_node.0 as usize] = Some(ConstValue::Bool(true));
                true_node
            } else {
                let mut negated: Vec<NodeId> = Vec::new();
                for c in &rec_conds {
                    let cn = self.compile_subexpr(*c);
                    let not_off = self.graph.inputs_pool.push(&[cn]);
                    negated.push(self.graph.add_node(Node {
                        kind: NodeKind::UnOp,
                        input_count: 1,
                        inputs_offset: not_off,
                        compute_fn: CF_NOT_BOOL,
                    }));
                }
                let mut and_result = negated[0];
                for n in &negated[1..] {
                    let and_off = self.graph.inputs_pool.push(&[and_result, *n]);
                    and_result = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: 2,
                        inputs_offset: and_off,
                        compute_fn: CF_AND_BOOL,
                    });
                }
                let not_off = self.graph.inputs_pool.push(&[and_result]);
                self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: 1,
                    inputs_offset: not_off,
                    compute_fn: CF_NOT_BOOL,
                })
            }
        }
    }

    // ---- Non-tail-recursion-to-iteration helpers ----

    /// Create an i32 constant node.
    fn make_i32_const(&mut self, val: i32) -> NodeId {
        let n = self.compile_const();
        self.graph.const_values[n.0 as usize] = Some(ConstValue::I32(val));
        n
    }

    /// Create a binary-operation node.
    fn make_binop(&mut self, lhs: NodeId, rhs: NodeId, cf: ComputeFnId) -> NodeId {
        let off = self.graph.inputs_pool.push(&[lhs, rhs]);
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 2,
            inputs_offset: off,
            compute_fn: cf,
        })
    }

    /// Create an array-store node `arr[idx] = val`.
    fn make_array_store(&mut self, arr: NodeId, idx: NodeId, val: NodeId) -> NodeId {
        let off = self.graph.inputs_pool.push(&[arr, idx, val]);
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 3,
            inputs_offset: off,
            compute_fn: CF_ARRAY_STORE,
        })
    }

    /// Create an array-index node `arr[idx]`.
    fn make_array_index(&mut self, arr: NodeId, idx: NodeId) -> NodeId {
        let off = self.graph.inputs_pool.push(&[arr, idx]);
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 2,
            inputs_offset: off,
            compute_fn: CF_ARRAY_INDEX,
        })
    }

    /// Create a Continue-signal barrier node (depends on `dep`; triggers the Continue signal).
    fn make_continue_barrier(&mut self, dep: NodeId) -> NodeId {
        let off = self.graph.inputs_pool.push(&[dep]);
        let n = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 1,
            inputs_offset: off,
            compute_fn: CF_CONTINUE,
        });
        n
    }

    /// Non-tail-recursion to iteration: consumes `NonTailRecInfo` to construct the
    /// "work-stack + while-loop + state-machine" IR.
    ///
    /// Called by `compile_function` when it detects `MemoStrategy::NonTailRecToLoop`.
    /// The parameter nodes have already been created and `bind_var`'d by `compile_function`;
    /// this method constructs the loop structure.
    ///
    /// Structure:
    /// - Function subgraph: param nodes + local variables (stack, sp, result) + initial frame
    ///   push + `Call(while_sg)`.
    /// - `while_sg`: `cond = sp > 0`, `Gate(cond)` -> `body_sg` / `result_sg`.
    /// - `body_sg` (LoopBody): pop a frame -> read `param_cur`/state/saved -> Gate chain
    ///   dispatches by state.
    /// - `state_N_sg`: compile the function body (`non_tail_rec_ctx` intercepts self-calls as
    ///   `push + barrier(Continue)`).
    /// - `result_sg`: returns `result_node`.
    ///
    /// Frame layout (stride = param_count + 1 + max_saved):
    /// `[param_0, ..., param_{P-1}, state, saved_0, ..., saved_{max_saved-1}]`
    fn compile_non_tail_rec_to_loop(
        &mut self,
        name: &str,
        body_expr: crate::ast::Ast::ExprId,
        params: &[crate::ast::Ast::Param<'_>],
        info: &crate::pass::Analyzer::NonTailRecInfo,
    ) -> NodeId {
        let param_count = info.param_count;
        let call_sites: Vec<crate::ast::Ast::ExprId> = info.call_sites.clone();
        let num_call_sites = call_sites.len();
        let max_saved = num_call_sites.saturating_sub(1);
        let stride = (param_count + 1 + max_saved) as u32;

        // 1. Collect parameter nodes (compile_function has already bind_var'd them)
        let param_nodes: Vec<NodeId> = params
            .iter()
            .filter_map(|p| self.lookup_var(p.name))
            .collect();

        // 2. Create local variables: stack_node (empty array), sp_node (0), result_node (void)
        let stack_off = self.graph.inputs_pool.push(&[]);
        let stack_node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 0,
            inputs_offset: stack_off,
            compute_fn: CF_ARRAY_CONSTRUCT,
        });
        let sp_node = self.make_i32_const(0);
        let result_node = self.compile_void_const();

        // 3. Push the initial frame: stack[0..P] = params, stack[P] = 0 (INIT), stack[P+1..] = 0; sp = 1
        // All array_stores must be chained into the effect chain to ensure Call(while_sg) executes after the stack is filled.
        let zero_init = self.make_i32_const(0);
        let mut init_effect: Option<NodeId> = None;
        for i in 0..param_count {
            let idx = self.make_i32_const(i as i32);
            let store = self.make_array_store(stack_node, idx, param_nodes[i]);
            init_effect = Some(self.chain_effects(init_effect, store));
        }
        let state_zero_idx = self.make_i32_const(param_count as i32);
        let state_zero_store = self.make_array_store(stack_node, state_zero_idx, zero_init);
        init_effect = Some(self.chain_effects(init_effect, state_zero_store));
        for i in 0..max_saved {
            let idx = self.make_i32_const((param_count + 1 + i) as i32);
            let store = self.make_array_store(stack_node, idx, zero_init);
            init_effect = Some(self.chain_effects(init_effect, store));
        }
        let one_init = self.make_i32_const(1);
        let sp_init_wb = self.compile_writeback_node(one_init, sp_node);
        self.current_effect = Some(self.chain_effects(init_effect, sp_init_wb));

        // 4. Placeholder-register while_sg
        let while_node_start = self.graph.nodes.len() as u32;
        let while_sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            id: while_sg_id,
            node_range: (NodeId(while_node_start), NodeId(while_node_start)),
            param_count: 0,
            entry_node: NodeId(while_node_start),
            return_node: NodeId(while_node_start),
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });

        // 5. cond_node: sp > 0 (within while_sg node_range)
        let zero_cond = self.make_i32_const(0);
        let cond_node = self.make_binop(sp_node, zero_cond, CF_GT_I32);

        // Save the init effect chain (including sp=1 WriteBack); body_sg compilation will reset current_effect
        let init_effect_chain = self.current_effect;

        // 6. Compile body_sg (LoopBody: pop + read frame + state dispatch)
        let body_sg = self.compile_non_tail_rec_body_sg(
            body_expr,
            params,
            name,
            &call_sites,
            while_sg_id,
            stack_node,
            sp_node,
            result_node,
            param_count,
            max_saved,
            stride,
        );

        // Restore the init effect chain so Call(while_sg) depends on the init code (including sp=1 WriteBack)
        self.current_effect = init_effect_chain;

        // 7. Compile result_sg (false branch, returns result_node)
        let result_sg = {
            let rs_start = self.graph.nodes.len() as u32;
            let off = self.graph.inputs_pool.push(&[result_node]);
            let passthrough = self.graph.add_node(Node {
                kind: NodeKind::BinOp,
                input_count: 1,
                inputs_offset: off,
                compute_fn: CF_SEQ,
            });
            let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
            self.graph.add_subgraph(SubGraph {
                id: sg_id,
                node_range: (NodeId(rs_start), NodeId(rs_start + 1)),
                param_count: 0,
                entry_node: NodeId(rs_start),
                return_node: passthrough,
                has_suspend: false,
                event_source_decls: Vec::new(),
                defer_table: Vec::new(),
                loop_kind: LoopKind::None,
                loop_parent_sg: None,
                cond_node: None,
                function_id: self.current_function_id,
                iter_next_node: None,
                upvalue_count: 0,
                upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
            });
            sg_id
        };

        // 8. Gate(cond): true → body_sg, false → result_sg
        let gate_off = self.graph.inputs_pool.push(&[cond_node]);
        let gate_node = self.graph.add_node(Node {
            kind: NodeKind::Gate,
            input_count: 1,
            inputs_offset: gate_off,
            compute_fn: CF_GATE_LAUNCH,
        });
        self.graph.set_gate_branches(
            gate_node,
            GateBranches {
                condition_input: cond_node,
                branches: vec![
                    (true, body_sg, Vec::new()),
                    (false, result_sg, Vec::new()),
                ],
            },
        );

        // 9. Populate while_sg metadata
        let while_node_end = self.graph.nodes.len() as u32;
        let while_sg = &mut self.graph.subgraphs[while_sg_id.0 as usize];
        while_sg.node_range = (NodeId(while_node_start), NodeId(while_node_end));
        while_sg.entry_node = NodeId(while_node_start);
        while_sg.return_node = gate_node;
        while_sg.loop_kind = LoopKind::While;
        while_sg.cond_node = Some(cond_node);
        while_sg.reset_plan = Some(ResetPlan {
            reset_to_zero: vec![],
            reset_to_one: vec![],
            reset_condition_tree: vec![cond_node],
        });

        // 10. Create the Call node to launch while_sg
        let call_node = self.compile_recursive_call(while_sg_id);
        call_node
    }

    /// Compile the body_sg for non-tail-recursion-to-iteration (LoopBody subgraph).
    ///
    /// Structure:
    /// 1. Pop: sp = sp - 1 (WriteBack), read the stack frame
    /// 2. Read param_cur[i] = stack[sp * stride + i]
    /// 3. Read state = stack[sp * stride + P]
    /// 4. Read saved[i] = stack[sp * stride + P + 1 + i]
    /// 5. Gate chain dispatches by state to each state_N_sg
    fn compile_non_tail_rec_body_sg(
        &mut self,
        body_expr: crate::ast::Ast::ExprId,
        params: &[crate::ast::Ast::Param<'_>],
        self_name: &str,
        call_sites: &[crate::ast::Ast::ExprId],
        while_sg_id: SubGraphId,
        stack_node: NodeId,
        sp_node: NodeId,
        result_node: NodeId,
        param_count: usize,
        max_saved: usize,
        stride: u32,
    ) -> SubGraphId {
        let body_node_start = self.graph.nodes.len() as u32;
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = body_node_start;

        // Push the loop context (Continue signal jump-back target = while_sg)
        self.loop_stack.push(LoopContext {
            sg: while_sg_id,
            iter_node: None,
            body_node_start,
            loop_var_node: None,
        });

        // Record the function subgraph's event_source_decls length before compilation (same as compile_loop_body_subgraph)
        let prev_decl_count = self.current_function_sg
            .and_then(|sg_id| self.graph.subgraphs.get(sg_id.0 as usize))
            .map(|sg| sg.event_source_decls.len())
            .unwrap_or(0);

        // 1. Pop: sp = sp - 1 (WriteBack to sp_node)
        let one_pop = self.make_i32_const(1);
        let sp_minus_1 = self.make_binop(sp_node, one_pop, CF_SUB_I32);
        let pop_wb = self.compile_writeback_node(sp_minus_1, sp_node);
        self.current_effect = Some(pop_wb);

        // 2. Read the stack frame: frame_base = sp_minus_1 * stride
        let stride_node = self.make_i32_const(stride as i32);
        let frame_base = self.make_binop(sp_minus_1, stride_node, CF_MUL_I32);

        // param_cur[i] = stack[frame_base + i]
        let mut param_cur: Vec<NodeId> = Vec::with_capacity(param_count);
        for i in 0..param_count {
            let offset = self.make_i32_const(i as i32);
            let idx = self.make_binop(frame_base, offset, CF_ADD_I32);
            param_cur.push(self.make_array_index(stack_node, idx));
        }

        // state = stack[frame_base + param_count]
        let state_offset = self.make_i32_const(param_count as i32);
        let state_idx = self.make_binop(frame_base, state_offset, CF_ADD_I32);
        let state_node = self.make_array_index(stack_node, state_idx);

        // saved[i] = stack[frame_base + param_count + 1 + i]
        let mut saved_nodes: Vec<NodeId> = Vec::with_capacity(max_saved);
        for i in 0..max_saved {
            let offset = self.make_i32_const((param_count + 1 + i) as i32);
            let idx = self.make_binop(frame_base, offset, CF_ADD_I32);
            saved_nodes.push(self.make_array_index(stack_node, idx));
        }

        // Chain param_cur / state_node / saved_nodes into the effect chain,
        // ensuring they are ready before the dispatch Gate launches state_N_sg.
        // Otherwise the Gate only depends on cmp(state_node==i), which could launch
        // state_N_sg before param_cur is computed, causing the frame copy to receive void parameter values.
        for &pc in &param_cur {
            self.current_effect = Some(self.chain_effects(self.current_effect, pc));
        }
        self.current_effect = Some(self.chain_effects(self.current_effect, state_node));
        for &sn in &saved_nodes {
            self.current_effect = Some(self.chain_effects(self.current_effect, sn));
        }
        let frame_read_effect = self.current_effect;

        // 3. Compile each state_N_sg (each state compiles the function body; non_tail_rec_ctx intercepts self-calls)
        let num_states = call_sites.len() + 1;
        let mut state_sgs: Vec<SubGraphId> = Vec::with_capacity(num_states);

        for state_idx in 0..num_states {
            // Build call_result_map:
            // state 0: empty (all calls are fresh)
            // state N: call_sites[0..N-2] -> saved[0..N-2], call_sites[N-1] -> result_node
            let mut call_result_map: rustc_hash::FxHashMap<crate::ast::Ast::ExprId, NodeId> =
                rustc_hash::FxHashMap::default();
            for i in 0..state_idx {
                if i + 1 < state_idx {
                    call_result_map.insert(call_sites[i], saved_nodes[i]);
                } else {
                    // i == state_idx - 1: the most recently completed call result is in result_node
                    call_result_map.insert(call_sites[i], result_node);
                }
            }

            // Set up non_tail_rec_ctx
            self.non_tail_rec_ctx = Some(NonTailRecCtx {
                self_name: self_name.to_string(),
                param_nodes: param_cur.clone(),
                stack_node,
                sp_node,
                result_node,
                call_result_map,
                truncated: false,
                stride,
                param_count,
                max_saved,
                current_state: state_idx as u32,
                saved_nodes: saved_nodes.clone(),
            });

            // Compile state_N_sg
            let sg_node_start = self.graph.nodes.len() as u32;
            let prev_sg_start_inner = self.current_sg_start;
            self.current_sg_start = sg_node_start;

            self.enter_scope();
            // Bind parameter names to param_cur nodes (instead of the function's param nodes)
            for (i, param) in params.iter().enumerate() {
                if i < param_cur.len() {
                    self.bind_var(param.name, param_cur[i]);
                }
            }

            let prev_effect_inner = self.current_effect;
            let prev_tail = self.in_tail_position;
            self.current_effect = None;
            self.in_tail_position = false;
            let body_node = self.compile_expr(body_expr);
            self.in_tail_position = prev_tail;
            self.current_effect = prev_effect_inner;

            // Clear non_tail_rec_ctx
            self.non_tail_rec_ctx = None;
            self.exit_scope();
            self.current_sg_start = prev_sg_start_inner;

            // Always WriteBack the body result to result_node.
            // Recursion path: the barrier's Continue signal terminates state_sg before the WriteBack executes,
            //   so the WriteBack does not run.
            // Base case path: the body completes normally, and the WriteBack writes the result to result_node.
            let return_node = self.compile_writeback_node(body_node, result_node);

            let sg_node_end = self.graph.nodes.len() as u32;

            // Migrate event_source_decls (same as compile_branch_subgraph)
            let state_decls: Vec<_> = if let Some(func_sg_id) = self.current_function_sg {
                if let Some(func_sg) = self.graph.subgraphs.get_mut(func_sg_id.0 as usize) {
                    func_sg.event_source_decls.drain(prev_decl_count..).collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            let state_sg = SubGraphId(self.graph.subgraphs.len() as u32);
            self.graph.add_subgraph(SubGraph {
                id: state_sg,
                node_range: (NodeId(sg_node_start), NodeId(sg_node_end)),
                param_count: 0,
                entry_node: NodeId(sg_node_start),
                return_node,
                has_suspend: false,
                event_source_decls: state_decls,
                defer_table: Vec::new(),
                loop_kind: LoopKind::None,
                loop_parent_sg: None,
                cond_node: None,
                function_id: self.current_function_id,
                iter_next_node: None,
                upvalue_count: 0,
                upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
            });
            state_sgs.push(state_sg);
        }

        // 4. Build the Gate chain dispatching by state (build the else chain from back to front)
        let void_sg = self.compile_void_subgraph();
        let mut false_sg = void_sg;
        let mut dispatch_gate: NodeId = NodeId(u32::MAX); // sentinel value, always overwritten in the loop

        // Reset current_effect to prevent the Gate chain from depending on residual effects from state compilation
        self.current_effect = None;

        for i in (0..num_states).rev() {
            let wrap_start = self.graph.nodes.len() as u32;

            // cmp = state_node == i
            let state_const = self.make_i32_const(i as i32);
            let cmp = self.make_binop(state_node, state_const, CF_EQ_I32);
            // Make cmp depend on frame_read_effect to ensure param_cur / saved_nodes are ready
            // before launching state_N_sg (otherwise the frame copy receives void parameter values)
            let cmp_eff = self.chain_effects(frame_read_effect, cmp);

            // Gate(cmp_eff): true → state_sgs[i], false → false_sg
            let gate_inputs: Vec<NodeId> = vec![cmp_eff];
            let gate_off = self.graph.inputs_pool.push(&gate_inputs);
            let gate_node = self.graph.add_node(Node {
                kind: NodeKind::Gate,
                input_count: gate_inputs.len() as u8,
                inputs_offset: gate_off,
                compute_fn: CF_GATE_LAUNCH,
            });
            self.graph.set_gate_branches(
                gate_node,
                GateBranches {
                    condition_input: cmp_eff,
                    branches: vec![
                        (true, state_sgs[i], Vec::new()),
                        (false, false_sg, Vec::new()),
                    ],
                },
            );

            if i == 0 {
                // The first Gate stays in body_sg
                dispatch_gate = gate_node;
            } else {
                // Wrap as a subgraph, serving as the previous Gate's false branch
                let wrap_end = self.graph.nodes.len() as u32;
                let wrap_sg = SubGraphId(self.graph.subgraphs.len() as u32);
                self.graph.add_subgraph(SubGraph {
                    id: wrap_sg,
                    node_range: (NodeId(wrap_start), NodeId(wrap_end)),
                    param_count: 0,
                    entry_node: NodeId(wrap_start),
                    return_node: gate_node,
                    has_suspend: false,
                    event_source_decls: Vec::new(),
                    defer_table: Vec::new(),
                    loop_kind: LoopKind::None,
                    loop_parent_sg: None,
                    cond_node: None,
                    function_id: self.current_function_id,
                    iter_next_node: None,
                    upvalue_count: 0,
                    upvalue_outer_nodes: Vec::new(),
                nested_ranges: Vec::new(),
            reset_plan: None,
            });
                false_sg = wrap_sg;
            }
        }

        // 5. Pop the loop context and register body_sg
        self.loop_stack.pop();
        self.current_sg_start = prev_sg_start;

        let body_node_end = self.graph.nodes.len() as u32;

        // Migrate body_sg's own event_source_decls
        let body_decls: Vec<_> = if let Some(func_sg_id) = self.current_function_sg {
            if let Some(func_sg) = self.graph.subgraphs.get_mut(func_sg_id.0 as usize) {
                func_sg.event_source_decls.drain(prev_decl_count..).collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let body_sg = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            id: body_sg,
            node_range: (NodeId(body_node_start), NodeId(body_node_end)),
            param_count: 0,
            entry_node: NodeId(body_node_start),
            return_node: dispatch_gate,
            has_suspend: false,
            event_source_decls: body_decls,
            defer_table: Vec::new(),
            loop_kind: LoopKind::LoopBody,
            loop_parent_sg: Some(while_sg_id),
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });
        body_sg
    }

    /// Register the Loop subgraph (no condition; terminates via Break).
    ///
    /// Structure (unified with While, cond is always true):
    /// - cond_node = Const(true)
    /// - gate_node = Gate(cond): true -> body_sg, false -> void_sg (unreachable)
    /// - body_sg: compiles the body, not tail-recursive (frame reuse on the Engine side)
    ///
    /// Execution: after the body runs, the Engine's reset_loop_iteration resets the Gate to re-execute;
    /// the Break signal terminates body_sg -> Gate completes -> the loop ends.
    fn register_loop_subgraph(&mut self, body: crate::ast::Ast::ExprId) -> SubGraphId {
        let node_start = self.graph.nodes.len() as u32;
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            id: sg_id,
            node_range: (NodeId(node_start), NodeId(node_start)),
            param_count: 0,
            entry_node: NodeId(node_start),
            return_node: NodeId(node_start),
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });

        // cond_node = Const(true) (loop has no condition; always true)
        let cond_node = self.compile_bool_const(true);
        // Reset current_effect = None (same as register_while_subgraph, to avoid deadlock from
        // external effect dependencies after the loop body's reset_loop_iteration)
        let prev_effect = self.current_effect;
        self.current_effect = None;
        // body subgraph (not tail-recursive)
        let body_sg = self.compile_loop_body_subgraph(body, sg_id);
        // void subgraph (unreachable branch; used on break exit): includes CF_DEFER_RUN for defer-in-loop.
        let void_sg = self.compile_defer_run_subgraph();
        self.current_effect = prev_effect;

        // Gate node: cond(true) -> body_sg, false -> void_sg (unreachable)
        let gate_off = self.graph.inputs_pool.push(&[cond_node]);
        let gate_node = self.graph.add_node(Node {
            kind: NodeKind::Gate,
            input_count: 1,
            inputs_offset: gate_off,
            compute_fn: CF_GATE_LAUNCH,
        });
        self.graph.set_gate_branches(
            gate_node,
            GateBranches {
                condition_input: cond_node,
                branches: vec![
                    (true, body_sg, vec![]),
                    (false, void_sg, vec![]),
                ],
            },
        );

        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = gate_node;
        sg.loop_kind = LoopKind::Loop;
        sg.cond_node = Some(cond_node);
        sg.reset_plan = Some(ResetPlan {
            reset_to_zero: vec![],
            reset_to_one: vec![],
            reset_condition_tree: vec![cond_node],
        });
        sg_id
    }

    /// Compile the loop body subgraph: compiles the body, not tail-recursive (frame reuse handled by Engine-side reset_loop_iteration).
    ///
    /// `loop_sg` is the while_sg of a While or the loop_sg of a Loop.
    /// return_node = body_last (the body's last node); the Engine detects LoopBody completion and resets the loop.
    fn compile_loop_body_subgraph(
        &mut self,
        body: crate::ast::Ast::ExprId,
        loop_sg: SubGraphId,
    ) -> SubGraphId {
        let node_start = self.graph.nodes.len() as u32;
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
        // Push the loop context (continue jump target; While/Loop have no iterator parameter)
        self.loop_stack.push(LoopContext {
            sg: loop_sg,
            iter_node: None,
            body_node_start: node_start,
            loop_var_node: None,
        });

        // Enter a new scope so that bind_var inside the loop body does not overwrite
        // the function-level binding. This is critical for WriteBack target resolution:
        // when an if/match branch inside the loop body assigns to an outer variable,
        // lookup_root_frame_var must find the function-level declaration (not an
        // intermediate node produced by the loop body's bind_var).
        self.enter_scope();

        // Record the function subgraph's event_source_decls length before compilation (same as compile_branch_subgraph, Bug #24)
        let prev_decl_count = self.current_function_sg
            .and_then(|sg_id| self.graph.subgraphs.get(sg_id.0 as usize))
            .map(|sg| sg.event_source_decls.len())
            .unwrap_or(0);

        let prev_in_loop = self.in_loop_body;
        self.in_loop_body = true;
        let body_last = self.compile_expr(body);
        self.in_loop_body = prev_in_loop;
        self.loop_stack.pop();
        self.exit_scope();
        self.current_sg_start = prev_sg_start;
        let node_end = self.graph.nodes.len() as u32;
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);

        // Migrate the event_source_decls added during loop-body compilation from the function subgraph to the loop body subgraph
        let body_decls: Vec<_> = if let Some(func_sg_id) = self.current_function_sg {
            if let Some(func_sg) = self.graph.subgraphs.get_mut(func_sg_id.0 as usize) {
                func_sg.event_source_decls.drain(prev_decl_count..).collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        self.graph.add_subgraph(SubGraph {
            id: sg_id,
            node_range: (NodeId(node_start), NodeId(node_end)),
            param_count: 0,
            entry_node: NodeId(node_start),
            return_node: body_last,
            has_suspend: false,
            event_source_decls: body_decls,
            defer_table: Vec::new(),
            loop_kind: LoopKind::LoopBody,
            loop_parent_sg: Some(loop_sg),
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });
        sg_id
    }

    ///
    /// Each arm is compiled into a Gate:
    /// - discriminant node = pattern match result (bool), used as the Gate's condition_input
    /// - Gate(true) -> arm body subgraph
    /// - Gate(false) -> next arm's Gate subgraph (as the else branch)
    ///
    /// Chain structure (built from the last arm backwards): each non-first arm's Gate + pattern is wrapped as an independent
    /// subgraph (param_count=1, receiving the scrutinee as a parameter), serving as the previous arm's else branch.
    /// The first arm's Gate stays in the parent frame; return_node = that Gate.
    ///
    /// The scrutinee is injected layer by layer into the wrap subgraphs' param nodes via the Gate's branch inputs,
    /// so that pattern discrimination inside each wrap subgraph can access the scrutinee.
    ///
    /// Two-phase compilation:
    /// 1. Front to back: compile each arm's pattern discriminant + variable binding + body subgraph
    /// 2. Back to front: build the Gate else chain, wrapping the wrap subgraphs
    ///
    /// Pattern variables are bound to field-extraction nodes via bind_var; the body subgraph accesses them through the frame chain.
    fn compile_match(
        &mut self,
        scrutinee: crate::ast::Ast::ExprId,
        arms: &[crate::ast::Ast::MatchArm],
    ) -> NodeId {
        // Empty match: return a void constant node to avoid panic
        if arms.is_empty() {
            return self.compile_void_const();
        }

        let scrutinee_node = self.compile_subexpr(scrutinee);
        let n_arms = arms.len();

        // Phase 1: front to back, compile each arm's pattern + body
        struct ArmData {
            wrap_start: u32,
            scrutinee_in_frame: NodeId,
            cond_node: NodeId,
            body_sg: SubGraphId,
            body_inputs: Vec<NodeId>,
            // The current_effect before this arm is compiled; used for Gate construction.
            // compile_branch_subgraph does not isolate current_effect, so side effects in later arm bodies
            // (such as the Continue barrier intercepted by non_tail_rec) leak into the prior arm's Gate input,
            // causing the prior arm to never execute (Bug #56).
            effect_before: Option<NodeId>,
        }

        let mut arm_data: Vec<ArmData> = Vec::with_capacity(n_arms);

        for (i, arm) in arms.iter().enumerate() {
            let wrap_start = self.graph.nodes.len() as u32;

            // Save the current effect: this arm's Gate should only depend on side effects completed before it,
            // not on side effects produced by later arm body compilation.
            let effect_before = self.current_effect;

            // Scrutinee source: i==0 uses scrutinee_node directly in the parent frame; i>0 uses a param node
            let scrutinee_in_frame = if i == 0 {
                scrutinee_node
            } else {
                let off = self.graph.inputs_pool.push(&[]);
                self.graph.add_node(Node {
                    kind: NodeKind::Const,
                    input_count: 0,
                    inputs_offset: off,
                    compute_fn: CF_NOOP,
                })
            };

            // Enter a scope to bind pattern variables
            self.enter_scope();

            // Compile the pattern: produce the discriminant node + bind variables to field-extraction nodes
            let pattern_node = self.compile_pattern_match(scrutinee_in_frame, arm.pattern);

            // Guard condition: pattern_match && guard
            let cond_node = if let Some(guard) = arm.guard {
                let guard_node = self.compile_subexpr(guard);
                self.compile_bool_and(pattern_node, guard_node)
            } else {
                pattern_node
            };

            // Compile the body subgraph (pattern variables are look-up-able in the scope)
            let (body_sg, body_inputs) = self.compile_branch_subgraph(arm.body);

            self.exit_scope();

            arm_data.push(ArmData {
                wrap_start,
                scrutinee_in_frame,
                cond_node,
                body_sg,
                body_inputs,
                effect_before,
            });
        }

        // Phase 2: back to front, build the Gate else chain
        let mut pending_else_sg: Option<SubGraphId> = None;
        let mut result_gate: Option<NodeId> = None;

        for (i, ad) in arm_data.iter().enumerate().rev() {
            // All arms use cond_node as the discriminant.
            // This ensures the Gate depends on the field-extraction nodes (via cond_node's dependency chain),
            // so the variable-bound field-extraction nodes execute before the Gate.
            // If the last arm is an exhaustive match (e.g. _), cond_node is true with no extra cost.
            let pattern_node = ad.cond_node;

            // false branch: if there is a pending_else (from i+1), use it and pass in the current frame's scrutinee.
            // If there is no else (non-exhaustive match), use a panic subgraph as runtime safety net.
            let (false_sg, false_inputs) = match pending_else_sg {
                Some(else_sg) => (else_sg, vec![ad.scrutinee_in_frame]),
                None => (self.compile_panic_subgraph(), Vec::new()),
            };

            // The Gate depends on pattern_node (the condition value) and the effect before this arm was compiled (prior side effects).
            // Use the arm-level effect_before rather than the global current_effect:
            // compile_branch_subgraph does not isolate current_effect, so side effects in later arm bodies
            // (such as the Continue barrier intercepted by non_tail_rec) leak into the prior arm's Gate,
            // causing the prior arm to never execute (Bug #56).
            let gate_inputs: Vec<NodeId> = match ad.effect_before {
                Some(eff) => vec![pattern_node, eff],
                None => vec![pattern_node],
            };
            let gate_off = self.graph.inputs_pool.push(&gate_inputs);
            let gate_node = self.graph.add_node(Node {
                kind: NodeKind::Gate,
                input_count: gate_inputs.len() as u8,
                inputs_offset: gate_off,
                compute_fn: CF_GATE_LAUNCH,
            });
            self.graph.set_gate_branches(
                gate_node,
                GateBranches {
                    condition_input: pattern_node,
                    branches: vec![
                        (true, ad.body_sg, ad.body_inputs.clone()),
                        (false, false_sg, false_inputs),
                    ],
                },
            );

            if i == 0 {
                result_gate = Some(gate_node);
            } else {
                let wrap_end = self.graph.nodes.len() as u32;
                let wrap_sg = SubGraphId(self.graph.subgraphs.len() as u32);
                self.graph.add_subgraph(SubGraph {
                    id: wrap_sg,
                    node_range: (NodeId(ad.wrap_start), NodeId(wrap_end)),
                    param_count: 1,
                    entry_node: NodeId(ad.wrap_start),
                    return_node: gate_node,
                    has_suspend: false,
                    event_source_decls: Vec::new(),
                    defer_table: Vec::new(),
                    loop_kind: LoopKind::None,
                    loop_parent_sg: None,
                    cond_node: None,
                    function_id: self.current_function_id,
                    iter_next_node: None,
                    upvalue_count: 0,
                    upvalue_outer_nodes: Vec::new(),
                nested_ranges: Vec::new(),
            reset_plan: None,
            });
                pending_else_sg = Some(wrap_sg);
            }
        }

        result_gate.expect("match must have at least one arm")
    }

    /// Compile a bool constant node.
    fn compile_bool_const(&mut self, b: bool) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(&[]);
        let n = self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset,
            compute_fn: CF_NOOP,
        });
        self.graph.const_values[n.0 as usize] = Some(ConstValue::Bool(b));
        n
    }

    /// Compile a bool AND node (used for guard conditions pattern && guard).
    fn compile_bool_and(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let off = self.graph.inputs_pool.push(&[lhs, rhs]);
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 2,
            inputs_offset: off,
            compute_fn: CF_AND_BOOL, // and_bool
        })
    }

    /// Compile a bool OR node (used for or-patterns p1 | p2).
    fn compile_bool_or(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let off = self.graph.inputs_pool.push(&[lhs, rhs]);
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 2,
            inputs_offset: off,
            compute_fn: CF_OR_BOOL, // or_bool
        })
    }

    /// Compile a pattern-match discriminant node (returns bool), binding pattern variables to field-extraction nodes.
    ///
    /// Recursively handles all pattern types:
    /// - Wildcard/Variable -> const(true); Variable binds the variable to the scrutinee
    /// - Literal -> eq(scrutinee, lit), selecting compute_fn by type
    /// - Constructor -> constructor-name discriminant + recursive sub-patterns
    /// - Record -> field extraction + recursive sub-patterns
    /// - OrPattern -> left_match || right_match
    /// - Guard -> pattern_match && condition
    fn compile_pattern_match(
        &mut self,
        scrutinee_node: NodeId,
        pattern_id: crate::ast::Ast::PatternId,
    ) -> NodeId {
        let pattern = self.current_module().arena.pattern(pattern_id);
        let module_name = self.current_module().name.to_string();
        match &pattern.node {
            crate::ast::Ast::Pattern::Wildcard => self.compile_bool_const(true),
            crate::ast::Ast::Pattern::Variable { name } => {
                // Nullary ADT constructors (e.g. JNull, Nil) cannot be distinguished from variables at parse time;
                // disambiguate via sema's ctor_def_index: if it is a known constructor, compile as Constructor
                if self.sema.ctor_def_index.contains_key(*name) {
                    let type_name = self.sema.pattern_ctor_types
                        .get(&(module_name.clone(), pattern_id.0))
                        .map(|s| s.as_ref());
                    self.compile_pattern_constructor(scrutinee_node, name, &[], type_name)
                } else {
                    self.bind_var(name, scrutinee_node);
                    self.compile_bool_const(true)
                }
            }
            crate::ast::Ast::Pattern::Literal(pl) => {
                self.compile_pattern_literal_match(scrutinee_node, pl)
            }
            crate::ast::Ast::Pattern::Constructor { name, patterns } => {
                let type_name = self.sema.pattern_ctor_types
                    .get(&(module_name, pattern_id.0))
                    .map(|s| s.as_ref());
                self.compile_pattern_constructor(scrutinee_node, name, patterns, type_name)
            }
            crate::ast::Ast::Pattern::Record { fields } => {
                self.compile_pattern_record(scrutinee_node, fields)
            }
            crate::ast::Ast::Pattern::OrPattern { left, right } => {
                let left_match = self.compile_pattern_match(scrutinee_node, *left);
                let right_match = self.compile_pattern_match(scrutinee_node, *right);
                self.compile_bool_or(left_match, right_match)
            }
            crate::ast::Ast::Pattern::Guard { pattern, condition } => {
                let pattern_match = self.compile_pattern_match(scrutinee_node, *pattern);
                let cond_node = self.compile_subexpr(*condition);
                self.compile_bool_and(pattern_match, cond_node)
            }
        }
    }

    /// Compile a literal pattern discriminant node.
    fn compile_pattern_literal_match(
        &mut self,
        scrutinee_node: NodeId,
        pl: &crate::ast::Ast::PatternLiteral,
    ) -> NodeId {
        match pl {
            crate::ast::Ast::PatternLiteral::Null => {
                // null discriminant: compute_is_null (idx 34)
                let off = self.graph.inputs_pool.push(&[scrutinee_node]);
                self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 1,
                    inputs_offset: off,
                    compute_fn: CF_IS_NULL,
                })
            }
            crate::ast::Ast::PatternLiteral::String(s) => {
                // string discriminant: compute_pattern_str_eq (idx 276)
                let str_node = self.compile_str_const(s);
                let off = self.graph.inputs_pool.push(&[scrutinee_node, str_node]);
                self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: off,
                    compute_fn: CF_PATTERN_STR_EQ,
                })
            }
            crate::ast::Ast::PatternLiteral::Int(s) => {
                let lit_node = self.compile_pattern_literal(pl);
                let compute_fn = self.select_literal_eq_fn(s, false);
                let off = self.graph.inputs_pool.push(&[scrutinee_node, lit_node]);
                self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: off,
                    compute_fn,
                })
            }
            crate::ast::Ast::PatternLiteral::Float(s) => {
                let lit_node = self.compile_pattern_literal(pl);
                // f128/f32/f16 suffixes need CF_EQ_OBJ for exact comparison (avoiding f128->f64 precision loss)
                // f64 or no suffix uses CF_EQ_F64 (f32/f16->f64 is lossless)
                let cleaned: String = s.chars().filter(|c| *c != '_').collect();
                let (_, suffix) = detect_float_suffix(&cleaned);
                let compute_fn = match suffix {
                    Some("f128") | Some("f32") | Some("f16") => CF_EQ_OBJ,
                    _ => CF_EQ_F64,
                };
                let off = self.graph.inputs_pool.push(&[scrutinee_node, lit_node]);
                self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: off,
                    compute_fn,
                })
            }
            crate::ast::Ast::PatternLiteral::Bool(_) => {
                let lit_node = self.compile_pattern_literal(pl);
                let off = self.graph.inputs_pool.push(&[scrutinee_node, lit_node]);
                self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: off,
                    compute_fn: CF_EQ_BOOL, // eq_bool
                })
            }
            crate::ast::Ast::PatternLiteral::Char(_) => {
                let lit_node = self.compile_pattern_literal(pl);
                let off = self.graph.inputs_pool.push(&[scrutinee_node, lit_node]);
                self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: off,
                    compute_fn: CF_EQ_I32, // eq_i32 (char stored as i32)
                })
            }
        }
    }

    /// Select the compute_fn for integer literal equality discriminant.
    fn select_literal_eq_fn(&self, s: &str, _is_unsigned: bool) -> ComputeFnId {
        // Strip hex/binary/octal prefix so 'x'/'b'/'o' is not mistaken for a type suffix.
        let stripped = if s.starts_with("0x") || s.starts_with("0X")
            || s.starts_with("0b") || s.starts_with("0B")
            || s.starts_with("0o") || s.starts_with("0O") {
            &s[2..]
        } else {
            s
        };
        // Dispatch via ValueTag::from_name + TypeFamily to eliminate string comparison
        if let Some(suffix) = stripped.find(|c: char| c.is_ascii_alphabetic()) {
            let suffix_str = &stripped[suffix..];
            if let Some(tag) = crate::value::ValueTag::from_name(suffix_str) {
                let ty = crate::types::Type::from(tag);
                use crate::types::TypeFamily;
                return match ty.family() {
                    TypeFamily::SignedInt64 | TypeFamily::UnsignedInt64 => CF_EQ_I64,
                    TypeFamily::SignedInt128 | TypeFamily::UnsignedInt128 => CF_EQ_I128,
                    _ => CF_EQ_I32,
                };
            }
        }
        CF_EQ_I32 // eq_i32 default
    }

    /// Compile a constructor pattern: constructor-name discriminant + recursive sub-patterns.
    fn compile_pattern_constructor(
        &mut self,
        scrutinee_node: NodeId,
        name: &str,
        patterns: &[crate::ast::Ast::PatternRef],
        type_name: Option<&str>,
    ) -> NodeId {
        // Constructor-name discriminant node
        let ctor_match_off = self.graph.inputs_pool.push(&[scrutinee_node]);
        let ctor_match_node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 1,
            inputs_offset: ctor_match_off,
            compute_fn: CF_PATTERN_CTOR_MATCH, // pattern_ctor_match
        });
        self.graph.set_pattern_ctor_name(ctor_match_node, name.to_string());
        // Record the constructor's owning type name for runtime disambiguation
        // (same-named constructors across different types, e.g. FileKind.File vs File).
        // Prefer the sema-disambiguated type_name; fall back to get_ctor_def.
        if let Some(tn) = type_name {
            self.graph.set_pattern_type_name(ctor_match_node, tn.to_string());
        } else if let Some(ctor_def) = self.sema.get_ctor_def(name) {
            self.graph.set_pattern_type_name(ctor_match_node, ctor_def.type_name.to_string());
        }

        // Recursively process sub-patterns: extract fields + discriminate
        let mut result = ctor_match_node;
        for (i, &sub_pattern_id) in patterns.iter().enumerate() {
            // Field-extraction node (by position)
            let field_get_off = self.graph.inputs_pool.push(&[scrutinee_node]);
            let field_get_node = self.graph.add_node(Node {
                kind: NodeKind::BinOp,
                input_count: 1,
                inputs_offset: field_get_off,
                compute_fn: CF_PATTERN_ADT_FIELD_GET, // pattern_adt_field_get
            });
            self.graph.set_pattern_field_index(field_get_node, i as u16);

            // Recursively compile the sub-pattern (may bind variables)
            let sub_match = self.compile_pattern_match(field_get_node, sub_pattern_id);

            // result = result && sub_match
            // field_get_node is an extra dependency input: ensures the variable-bound field-extraction node
            // executes before the Gate triggers (compute_and_bool only reads inputs[0..2]; inputs[2] is for scheduling only)
            let and_off = self.graph.inputs_pool.push(&[result, sub_match, field_get_node]);
            result = self.graph.add_node(Node {
                kind: NodeKind::BinOp,
                input_count: 3,
                inputs_offset: and_off,
                compute_fn: CF_AND_BOOL, // and_bool (ignores inputs[2])
            });
        }

        result
    }

    /// Compile a record pattern: field extraction + recursive sub-patterns.
    fn compile_pattern_record(
        &mut self,
        scrutinee_node: NodeId,
        fields: &[crate::ast::Ast::PatternRecordField<'_>],
    ) -> NodeId {
        let mut result = self.compile_bool_const(true);

        for field in fields.iter() {
            // Field-extraction node (by name, reuses compute_record_field_get idx 30)
            let field_get_off = self.graph.inputs_pool.push(&[scrutinee_node]);
            let field_get_node = self.graph.add_node(Node {
                kind: NodeKind::FieldAccess,
                input_count: 1,
                inputs_offset: field_get_off,
                compute_fn: CF_RECORD_FIELD_GET, // record_field_get
            });
            self.graph.set_field_set_name(field_get_node, field.name.to_string());

            // Recursively compile the sub-pattern
            let sub_match = self.compile_pattern_match(field_get_node, field.pattern);

            // field_get_node is an extra dependency input (same as compile_pattern_constructor)
            let and_off = self.graph.inputs_pool.push(&[result, sub_match, field_get_node]);
            result = self.graph.add_node(Node {
                kind: NodeKind::BinOp,
                input_count: 3,
                inputs_offset: and_off,
                compute_fn: CF_AND_BOOL,
            });
        }

        result
    }

    /// Compile a literal pattern into a Const node.
    fn compile_pattern_literal(&mut self, pl: &crate::ast::Ast::PatternLiteral) -> NodeId {
        let const_val = match pl {
            crate::ast::Ast::PatternLiteral::Int(s) => {
                // Strip underscore separators
                let cleaned: String = s.chars().filter(|c| *c != '_').collect();
                // Detect radix prefix (0x/0b/0o) — the prefix letter must not be
                // mistaken for a type suffix when separating digits from suffix.
                let (radix, skip_prefix) = if cleaned.starts_with("0x") || cleaned.starts_with("0X") {
                    (16, 2)
                } else if cleaned.starts_with("0b") || cleaned.starts_with("0B") {
                    (2, 2)
                } else if cleaned.starts_with("0o") || cleaned.starts_with("0O") {
                    (8, 2)
                } else {
                    (10, 0)
                };
                // After the prefix, find the type suffix (first alphabetic char)
                let after_prefix = &cleaned[skip_prefix..];
                let suffix_pos = after_prefix.find(|c: char| c.is_ascii_alphabetic());
                let digits = if let Some(pos) = suffix_pos {
                    &after_prefix[..pos]
                } else {
                    after_prefix
                };
                i32::from_str_radix(digits, radix).ok().map(ConstValue::I32)
            }
            crate::ast::Ast::PatternLiteral::Float(s) => {
                // Strip underscore separators + type suffix (f64/f32/f16/f128)
                // Bug #42: the suffix in pattern-position `0.0f64` caused parse::<f64>() to fail,
                // leaving the Const node's value as None (not pre-populated), so CF_EQ_F64 waited forever for input -> match hang
                // f128 suffix must be stored exactly as F128 to avoid f128->f64 precision loss
                let cleaned: String = s.chars().filter(|c| *c != '_').collect();
                let (stripped, suffix) = detect_float_suffix(&cleaned);
                let is_hex = stripped.starts_with("0x") || stripped.starts_with("0X");
                match suffix {
                    Some("f128") => {
                        if is_hex { parse_hex_float_f128(stripped).map(ConstValue::F128) }
                        else { parse_decimal_f128(stripped).map(ConstValue::F128) }
                    }
                    Some("f32") => {
                        if is_hex { parse_hex_float_f32(stripped).map(ConstValue::F32) }
                        else { stripped.parse::<f32>().ok().map(ConstValue::F32) }
                    }
                    Some("f16") => {
                        if is_hex { parse_hex_float_f16(stripped).map(ConstValue::F16) }
                        else {
                            stripped.parse::<f64>()
                                .ok()
                                .map(|f| ConstValue::F16(crate::value::F16::from_f64(f).to_bits()))
                        }
                    }
                    // f64 or no suffix (default f64): f32/f16->f64 is lossless, can use CF_EQ_F64
                    _ => {
                        if is_hex { parse_hex_float_f64(stripped).map(ConstValue::F64) }
                        else { stripped.parse::<f64>().ok().map(ConstValue::F64) }
                    }
                }
            }
            crate::ast::Ast::PatternLiteral::Bool(b) => Some(ConstValue::Bool(*b)),
            crate::ast::Ast::PatternLiteral::String(_) => {
                Some(ConstValue::Bool(true)) // placeholder; actually uses compile_str_const
            }
            crate::ast::Ast::PatternLiteral::Char(c) => Some(ConstValue::I32(*c as i32)),
            crate::ast::Ast::PatternLiteral::Null => Some(ConstValue::Null),
        };
        let inputs_offset = self.graph.inputs_pool.push(&[]);
        let n = self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset,
            compute_fn: CF_NOOP,
        });
        self.graph.const_values[n.0 as usize] = const_val;
        n
    }

    /// Compile a string constant node (used for string literals in pattern matching).
    fn compile_str_const(&mut self, s: &str) -> NodeId {
        let (offset, len) = self.intern_str(s);
        let inputs_offset = self.graph.inputs_pool.push(&[]);
        let n = self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset,
            compute_fn: CF_NOOP,
        });
        self.graph.const_values[n.0 as usize] = Some(ConstValue::Str { offset, len });
        n
    }

    /// Compile string interpolation: lower `"text {expr} more {expr}"` into a chained str_concat.
    ///
    /// Each Literal part is compiled into a string constant node;
    /// each Expression part is converted to a string via `compute_reflect_format` (idx 290);
    /// all parts are chained together via `compute_str_concat` (idx 269).
    fn compile_str_interp(
        &mut self,
        parts: &[crate::ast::Ast::InterpolationPart<'_>],
    ) -> NodeId {
        if parts.is_empty() {
            return self.compile_str_const("");
        }

        // Collect all part nodes
        let mut nodes: Vec<NodeId> = Vec::with_capacity(parts.len());
        for part in parts {
            match part {
                crate::ast::Ast::InterpolationPart::Literal(text) => {
                    if !text.is_empty() {
                        nodes.push(self.compile_str_const(text));
                    }
                }
                crate::ast::Ast::InterpolationPart::Expression(expr_id) => {
                    let expr_node = self.compile_subexpr(*expr_id);
                    // Convert any value to a string via compute_reflect_format
                    // (a standalone compute_fn, not going through FFI dispatch, with built-in lazy force)
                    let inputs_offset = self.graph.inputs_pool.push(&[expr_node]);
                    let reflect_node = self.graph.add_node(Node {
                        kind: NodeKind::Call,
                        input_count: 1,
                        inputs_offset,
                        compute_fn: CF_REFLECT_FORMAT, // compute_reflect_format
                    });
                    nodes.push(reflect_node);
                }
            }
        }

        // Single part: return directly
        if nodes.len() == 1 {
            return nodes[0];
        }

        // Multi-input one-shot concat: O(n) one-shot concatenation, replacing chained O(n²) concat
        let inputs_offset = self.graph.inputs_pool.push(&nodes);
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: nodes.len() as u8,
            inputs_offset,
            compute_fn: CF_STR_MULTI_CONCAT,
        })
    }

    /// Convert any value node to a string node via compute_reflect_format (idx 290).
    /// Used to convert non-string operands to strings for `str + non-str` concatenation (same as string interpolation).
    fn make_reflect_format_node(&mut self, value_node: NodeId) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(&[value_node]);
        self.graph.add_node(Node {
            kind: NodeKind::Call,
            input_count: 1,
            inputs_offset,
            compute_fn: CF_REFLECT_FORMAT,
        })
    }

    /// Return the module name used for the expr_types composite key.
    ///
    /// In a monomorphization-instance context, the function body expressions belong to the module
    /// of the callee function; the expr_types key must use the instance's module_name (not the call-site
    /// module name), otherwise cross-module generic calls fail type lookup (e.g. when Math.abs calls
    /// cast(x).to(i32), source_ty resolves to void).
    fn expr_key_module(&self) -> &'a str {
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
    fn expr_type_name(&self, expr_id: crate::ast::Ast::ExprId) -> Option<&str> {
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
                            .unwrap_or_else(|| self.type_arena.get(info.ty).name()),
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
                    .unwrap_or_else(|| self.type_arena.get(info.ty).name()),
            );
        }
        None
    }

    /// Look up the implicit-this access kind for an expression (set by sema).
    ///
    /// Sema records on `ExprInfo.implicit_this` whether a bare identifier/call inside a method
    /// body resolved to an instance field or method (i.e. an implicit `this.` access). The IR
    /// builder consumes this marker to synthesize the explicit `this`-based access nodes.
    fn expr_implicit_this(
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
    fn expr_type_name_checked(&mut self, expr_id: crate::ast::Ast::ExprId, context: &str) -> &str {
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
    fn expr_is_nullable(&self, expr_id: crate::ast::Ast::ExprId) -> bool {
        let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), expr_id.0 as u64);
        self.sema
            .expr_types
            .get(&key)
            .map(|info| matches!(self.type_arena.get(info.ty), crate::sema::Sema::Type::Nullable(_)))
            .unwrap_or(false)
    }

    /// Type family: returns the `TypeFamily` (caller merges integer variants with `|` to dispatch by bit-width).
    /// i8/i16/u8/u16/u32/char -> SignedInt32/UnsignedInt32/Char; i64/u64/isize/usize -> SignedInt64/UnsignedInt64;
    /// i128/u128 -> SignedInt128/UnsignedInt128; bool -> Bool; floats -> Float.
    fn int_family(ty_name: &str) -> crate::types::TypeFamily {
        match crate::value::ValueTag::from_name(ty_name).and_then(scalar_meta) {
            Some(m) => m.family,
            None => crate::types::TypeFamily::SignedInt32, // unknown integer type falls back to the Int32 path
        }
    }

    /// Get TypeFamily from a type name (including non-scalar built-in types like Str).
    /// Unlike int_family, this method uses ValueTag::from_name + family() directly,
    /// bypassing scalar_meta, so for str it returns TypeFamily::Str instead of falling back to SignedInt32.
    fn type_family(ty_name: &str) -> crate::types::TypeFamily {
        match crate::value::ValueTag::from_name(ty_name) {
            Some(tag) => tag.family(),
            None => crate::types::TypeFamily::SignedInt32,
        }
    }

    /// Arithmetic/bitwise compute_fn table lookup: returns the arithmetic base by type name.
    /// Integer types: 12 consecutive indices each (add/sub/mul/div/mod/bitand/bitor/bitxor/shl/shr/neg/bitnot);
    /// float types: 6 consecutive indices each (add/sub/mul/div/mod/neg, no bitwise ops).
    /// Returns None when the type does not support arithmetic.
    /// The base comes from `scalar_meta`, kept in single-point sync with the compute_fn_table! indices.
    fn arith_base(ty_name: &str) -> Option<u32> {
        crate::value::ValueTag::from_name(ty_name).and_then(scalar_meta).map(|m| m.arith_base)
    }

    /// Select a compute_fn id by op + expression type.
    fn select_binary_compute_fn(
        &mut self,
        op: crate::ast::Ast::BinaryOp,
        binary_expr_id: crate::ast::Ast::ExprId,
        lhs_expr: crate::ast::Ast::ExprId,
        _rhs_expr: crate::ast::Ast::ExprId,
    ) -> ComputeFnId {
        // Consume the sema-promoted type: binary_expr_id's ExprInfo.type_name is the binary
        // operation's result type inferred by sema. For arithmetic, the result type is the promoted
        // operand type (i32+f64 -> f64); for comparisons, the result type is bool, so the operand type
        // must be used to select the compute_fn.
        // Check in two steps to avoid borrow conflicts: first check whether the lhs type exists and report errors, then get the type reference
        let has_lhs_ty = self.expr_type_name(lhs_expr).is_some();
        if !has_lhs_ty {
            self.errors.push(format!(
                "internal: missing ExprInfo for expr {:?} in binary_op", lhs_expr));
        }
        let lhs_ty = self.expr_type_name(lhs_expr).unwrap_or("i32");
        let ty_name = match self.expr_type_name(binary_expr_id) {
            Some(t) if Self::type_family(t) == crate::types::TypeFamily::Bool => lhs_ty,  // comparison: use operand type
            Some(t) => t,             // arithmetic: use promoted type
            None => lhs_ty,           // no sema record: fall back to lhs type
        };
        let ty_meta = crate::value::ValueTag::from_name(ty_name).and_then(scalar_meta);
        let is_float = ty_meta.as_ref().map(|m| m.is_float).unwrap_or(false);
        // f128 needs a dedicated comparison path: going through to_f64 drops 60 bits of precision, causing distinct f128 values to be misjudged as equal
        let is_f128 = crate::value::ValueTag::from_name(ty_name) == Some(crate::value::ValueTag::F128);
        // is_int: non-float and non-bool (reuses the TypeFamily enum, eliminating string comparison)
        let is_int = !is_float && Self::int_family(ty_name) != crate::types::TypeFamily::Bool;
        let base = Self::arith_base(ty_name);

        // Elvis (??) operator: returns rhs when lhs is null, otherwise lhs.
        // Does not depend on operand types; must be handled before the str/composite-type branches.
        if matches!(op, crate::ast::Ast::BinaryOp::Elvis) {
            return CF_ELVIS;
        }

        // ==/!= for nullable types: ?. short-circuit or null literals produce Value::Null,
        // and dedicated comparison functions for str/i32 etc. do not handle Null (heap_obj() returns None, always false).
        // Dispatch to CF_EQ_OBJ/CF_NE_OBJ (value_equals_with_arena correctly handles Null discrimination).
        if matches!(op, crate::ast::Ast::BinaryOp::Eq | crate::ast::Ast::BinaryOp::NotEq)
            && self.expr_is_nullable(lhs_expr)
        {
            return match op {
                crate::ast::Ast::BinaryOp::Eq => CF_EQ_OBJ,
                crate::ast::Ast::BinaryOp::NotEq => CF_NE_OBJ,
                _ => unreachable!(),
            };
        }

        // str + str -> string concatenation (compute_str_concat, 269)
        if Self::type_family(ty_name) == crate::types::TypeFamily::Str
            && matches!(op, crate::ast::Ast::BinaryOp::Add)
        {
            return CF_STR_CONCAT;
        }

        // str comparison -> dedicated str comparison compute_fn (292-297)
        // Does not go through the i32 path: str has no as_i32 semantics; the i32 path would always be 0, yielding wrong results
        if Self::type_family(ty_name) == crate::types::TypeFamily::Str {
            return match op {
                crate::ast::Ast::BinaryOp::Eq => CF_EQ_STR,
                crate::ast::Ast::BinaryOp::NotEq => CF_NE_STR,
                crate::ast::Ast::BinaryOp::Lt => CF_LT_STR,
                crate::ast::Ast::BinaryOp::Gt => CF_GT_STR,
                crate::ast::Ast::BinaryOp::LtEq => CF_LE_STR,
                crate::ast::Ast::BinaryOp::GtEq => CF_GE_STR,
                _ => CF_EQ_STR, // arithmetic etc. already handled above; unreachable here
            };
        }

        // ==/!= for composite types (record/adt/newtype/array/closure/throw etc.) ->
        // generic semantic comparison compute_fn (298-299). Going through the i32 path would make
        // as_i32() always 0, judging all composite types as equal.
        // Rationale: scalar_meta being None means a non-scalar type. At this point Str and Nullable
        // are already handled above; the remaining None cases are all composite types (Array/Ref/Fn/Adt/Record/...).
        // scalar_meta is the single source of truth for scalar types, so is_none() is the necessary and sufficient condition for composite types.
        if matches!(op, crate::ast::Ast::BinaryOp::Eq | crate::ast::Ast::BinaryOp::NotEq)
            && ty_meta.is_none()
        {
            return match op {
                crate::ast::Ast::BinaryOp::Eq => CF_EQ_OBJ,
                crate::ast::Ast::BinaryOp::NotEq => CF_NE_OBJ,
                _ => unreachable!(),
            };
        }

        // Arithmetic (add/sub/mul/div/mod): supported by both integers and floats; look up by concrete type
        // Integer index order: add(0) sub(1) mul(2) div(3) mod(4) bitand(5) bitor(6) bitxor(7) shl(8) shr(9) neg(10) bitnot(11)
        // Float index order: add(0) sub(1) mul(2) div(3) mod(4) neg(5)
        let arith_offset = |op: &crate::ast::Ast::BinaryOp| -> Option<u32> {
            match op {
                crate::ast::Ast::BinaryOp::Add => Some(0),
                crate::ast::Ast::BinaryOp::Sub => Some(1),
                crate::ast::Ast::BinaryOp::Mul => Some(2),
                crate::ast::Ast::BinaryOp::Div => Some(3),
                crate::ast::Ast::BinaryOp::Mod => Some(4),
                _ => None,
            }
        };
        if let Some(off) = arith_offset(&op) {
            if let Some(b) = base {
                return ComputeFnId(b + off);
            }
            // Unknown type falls back to the i32 path
            return ComputeFnId(116 + off);
        }

        // Bitwise (bitand/bitor/bitxor/shl/shr): only supported by integers
        if is_int {
            let bit_offset = match op {
                crate::ast::Ast::BinaryOp::BitAnd => Some(5),
                crate::ast::Ast::BinaryOp::BitOr => Some(6),
                crate::ast::Ast::BinaryOp::BitXor => Some(7),
                crate::ast::Ast::BinaryOp::Shl => Some(8),
                crate::ast::Ast::BinaryOp::Shr => Some(9),
                _ => None,
            };
            if let Some(off) = bit_offset {
                if let Some(b) = base {
                    return ComputeFnId(b + off);
                }
                return ComputeFnId(CF_ADD_I32_FULL.0 + off); // fall back to i32
            }
        }

        // Comparison: result is bool; input read by type family
        // fam is the TypeFamily enum; use | to merge signed/unsigned integer variants to dispatch by bit-width (compiler exhaustive check)
        let fam = Self::int_family(ty_name);
        use crate::types::TypeFamily;
        // The 6 comparison ops share an f128->float->(bool)->i128->i64->i32 cascade; a macro removes the repetition.
        // Eq/NotEq have a Bool branch; Lt/Gt/LtEq/GtEq have no Bool branch (bool cannot be ordered).
        // The macro only expands the cascade block (=> right side); match patterns stay explicit to preserve the compiler's exhaustive check.
        macro_rules! cmp_arm {
            ($f128:ident, $f64:ident, $bool:ident, $i128:ident, $i64:ident, $i32:ident) => {
                if is_f128 { $f128 }
                else if is_float { $f64 }
                else if fam == TypeFamily::Bool { $bool }
                else if matches!(fam, TypeFamily::SignedInt128 | TypeFamily::UnsignedInt128) { $i128 }
                else if matches!(fam, TypeFamily::SignedInt64 | TypeFamily::UnsignedInt64) { $i64 }
                else { $i32 }
            };
            ($f128:ident, $f64:ident, $i128:ident, $i64:ident, $i32:ident) => {
                if is_f128 { $f128 }
                else if is_float { $f64 }
                else if matches!(fam, TypeFamily::SignedInt128 | TypeFamily::UnsignedInt128) { $i128 }
                else if matches!(fam, TypeFamily::SignedInt64 | TypeFamily::UnsignedInt64) { $i64 }
                else { $i32 }
            };
        }
        match op {
            crate::ast::Ast::BinaryOp::Eq => cmp_arm!(CF_EQ_F128, CF_EQ_F64, CF_EQ_BOOL, CF_EQ_I128, CF_EQ_I64, CF_EQ_I32),
            crate::ast::Ast::BinaryOp::NotEq => cmp_arm!(CF_NE_F128, CF_NE_F64, CF_NE_BOOL, CF_NE_I128, CF_NE_I64, CF_NE_I32),
            crate::ast::Ast::BinaryOp::Lt => cmp_arm!(CF_LT_F128, CF_LT_F64, CF_LT_I128, CF_LT_I64, CF_LT_I32),
            crate::ast::Ast::BinaryOp::Gt => cmp_arm!(CF_GT_F128, CF_GT_F64, CF_GT_I128, CF_GT_I64, CF_GT_I32),
            crate::ast::Ast::BinaryOp::LtEq => cmp_arm!(CF_LE_F128, CF_LE_F64, CF_LE_I128, CF_LE_I64, CF_LE_I32),
            crate::ast::Ast::BinaryOp::GtEq => cmp_arm!(CF_GE_F128, CF_GE_F64, CF_GE_I128, CF_GE_I64, CF_GE_I32),
            crate::ast::Ast::BinaryOp::And => CF_AND_BOOL, // and_bool
            crate::ast::Ast::BinaryOp::Or => CF_OR_BOOL,  // or_bool
            crate::ast::Ast::BinaryOp::RefEq => CF_REF_EQ,          // ref_eq
            crate::ast::Ast::BinaryOp::RefNeq => CF_REF_NEQ,         // ref_neq
            crate::ast::Ast::BinaryOp::ConcatList => CF_CONCAT_LIST,     // concat_list
            crate::ast::Ast::BinaryOp::Range => CF_RANGE,          // range
            crate::ast::Ast::BinaryOp::RangeInclusive => CF_RANGE_INCLUSIVE, // range_inclusive
            crate::ast::Ast::BinaryOp::Elvis => CF_ELVIS,          // elvis
            _ => CF_NOOP,
        }
    }

    /// Select a unary operation compute_fn id by op + operand expression type.
    fn select_unary_compute_fn(
        &mut self,
        op: crate::ast::Ast::UnaryOp,
        operand_expr: crate::ast::Ast::ExprId,
    ) -> ComputeFnId {
        let ty_name = self.expr_type_name_checked(operand_expr, "unary_op");
        let is_float = crate::value::ValueTag::from_name(ty_name).and_then(scalar_meta).map(|m| m.is_float).unwrap_or(false);
        let base = Self::arith_base(ty_name);
        match op {
            crate::ast::Ast::UnaryOp::Not => CF_NOT_BOOL, // not_bool
            crate::ast::Ast::UnaryOp::Neg => {
                // integer neg is at base+10; float neg is at base+5
                if let Some(b) = base {
                    let off = if is_float { 5 } else { 10 };
                    return ComputeFnId(b + off);
                }
                CF_NEG_I32_FULL // fall back to neg_i32
            }
            crate::ast::Ast::UnaryOp::BitNot => {
                // integers only; bitnot is at base+11
                if let Some(b) = base {
                    return ComputeFnId(b + 11);
                }
                CF_BITNOT_I32_FULL // fall back to bitnot_i32
            }
        }
    }

    /// Compile a binary operation.
    fn compile_binary(
        &mut self,
        op: crate::ast::Ast::BinaryOp,
        binary_expr_id: crate::ast::Ast::ExprId,
        lhs: crate::ast::Ast::ExprId,
        rhs: crate::ast::Ast::ExprId,
    ) -> NodeId {
        // Range/RangeInclusive compiled as a range_iter(start, end, inclusive) function call
        // (Range itself is an iterator; the For loop statically dispatches via RangeIterator.next)
        match op {
            crate::ast::Ast::BinaryOp::Range | crate::ast::Ast::BinaryOp::RangeInclusive => {
                let lhs_node = self.compile_subexpr(lhs);
                let rhs_node = self.compile_subexpr(rhs);
                let inclusive = matches!(op, crate::ast::Ast::BinaryOp::RangeInclusive);
                let bool_node = self.compile_bool_const(inclusive);
                self.make_call_by_name("range_iter", &[lhs_node, rhs_node, bool_node])
            }
            // Bug #38: &&/|| short-circuit evaluation -- lowered to a Gate conditional branch, ensuring RHS is
            // evaluated only when LHS does not satisfy the short-circuit condition (same conditional dataflow as the if expression).
            //   lhs && rhs  =>  if lhs { rhs } else { false }
            //   lhs || rhs  =>  if lhs { true } else { rhs }
            crate::ast::Ast::BinaryOp::And | crate::ast::Ast::BinaryOp::Or => {
                self.compile_short_circuit(op, lhs, rhs)
            }
            _ => {
                // str + non-str / non-str + str -> convert the non-string operand to a string via
                // compute_reflect_format, then concat with str_concat
                // (same lowering path as string interpolation "{expr}")
                if matches!(op, crate::ast::Ast::BinaryOp::Add) {
                    let lhs_ty = self.expr_type_name(lhs).unwrap_or("");
                    let rhs_ty = self.expr_type_name(rhs).unwrap_or("");
                    let lhs_is_str = Self::type_family(lhs_ty) == crate::types::TypeFamily::Str;
                    let rhs_is_str = Self::type_family(rhs_ty) == crate::types::TypeFamily::Str;
                    if lhs_is_str || rhs_is_str {
                        let lhs_node = self.compile_subexpr(lhs);
                        let rhs_node = self.compile_subexpr(rhs);
                        let lhs_final = if lhs_is_str {
                            lhs_node
                        } else {
                            self.make_reflect_format_node(lhs_node)
                        };
                        let rhs_final = if rhs_is_str {
                            rhs_node
                        } else {
                            self.make_reflect_format_node(rhs_node)
                        };
                        let inputs_offset =
                            self.graph.inputs_pool.push(&[lhs_final, rhs_final]);
                        return self.graph.add_node(Node {
                            kind: NodeKind::BinOp,
                            input_count: 2,
                            inputs_offset,
                            compute_fn: CF_STR_CONCAT,
                        });
                    }
                }
                // Operands are not in tail position: their values are consumed by the operation node, not returned directly.
                let lhs_node = self.compile_subexpr(lhs);
                let rhs_node = self.compile_subexpr(rhs);
                let inputs_offset = self.graph.inputs_pool.push(&[lhs_node, rhs_node]);
                let compute_fn = self.select_binary_compute_fn(op, binary_expr_id, lhs, rhs);
                let node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset,
                    compute_fn,
                });
                // Compile-time SIMD batching marker: scalar type + op -> runtime batches by (tag, op) group
                if let Some(info) = self.binary_batch_info(op, lhs) {
                    self.graph.set_batch_info(node, info);
                }
                node
            }
        }
    }

    /// Bug #38: compile &&/|| short-circuit evaluation.
    ///
    /// Uses a Gate conditional branch to ensure RHS is evaluated only when LHS does not satisfy the short-circuit condition:
    ///   lhs && rhs  =>  if lhs { rhs } else { false }
    ///   lhs || rhs  =>  if lhs { true } else { rhs }
    ///
    /// Consistent with compile_if's Gate pattern: cond_node + then_sg + else_sg.
    /// The then/else branch bodies are Const nodes (short-circuit value) or the RHS expression (the branch needing evaluation).
    fn compile_short_circuit(
        &mut self,
        op: crate::ast::Ast::BinaryOp,
        lhs: crate::ast::Ast::ExprId,
        rhs: crate::ast::Ast::ExprId,
    ) -> NodeId {
        let cond_node = self.compile_subexpr(lhs);
        let is_and = matches!(op, crate::ast::Ast::BinaryOp::And);
        // && : lhs=true -> evaluate rhs ; lhs=false -> false (short-circuit)
        // || : lhs=true -> true (short-circuit)   ; lhs=false -> evaluate rhs
        let (then_sg, then_inputs) = if is_and {
            self.compile_branch_subgraph(rhs)
        } else {
            self.compile_bool_branch(true)
        };
        let (else_sg, else_inputs) = if is_and {
            self.compile_bool_branch(false)
        } else {
            self.compile_branch_subgraph(rhs)
        };
        let gate_inputs: Vec<NodeId> = match self.current_effect {
            Some(eff) => vec![cond_node, eff],
            None => vec![cond_node],
        };
        let inputs_offset = self.graph.inputs_pool.push(&gate_inputs);
        let gate_node = self.graph.add_node(Node {
            kind: NodeKind::Gate,
            input_count: gate_inputs.len() as u8,
            inputs_offset,
            compute_fn: CF_GATE_LAUNCH,
        });
        self.graph.set_gate_branches(
            gate_node,
            GateBranches {
                condition_input: cond_node,
                branches: vec![
                    (true, then_sg, then_inputs),
                    (false, else_sg, else_inputs),
                ],
            },
        );
        gate_node
    }

    /// Compile a constant bool branch (short-circuit value), used for &&'s false branch and ||'s true branch.
    fn compile_bool_branch(&mut self, value: bool) -> (SubGraphId, Vec<NodeId>) {
        let node_start = self.graph.nodes.len() as u32;
        self.enter_scope();
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
        let return_node = self.compile_bool_const(value);
        self.current_sg_start = prev_sg_start;
        self.exit_scope();
        let node_end = self.graph.nodes.len() as u32;
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            id: sg_id,
            node_range: (NodeId(node_start), NodeId(node_end)),
            param_count: 0,
            entry_node: NodeId(node_start),
            return_node,
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: crate::ir::Ir::LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: self.current_function_sg
                .map(|sg| sg.0)
                .unwrap_or(0),
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });
        (sg_id, Vec::new())
    }

    /// Map a Kuzo BinaryOp + type name to a BatchInfo (batchable op + scalar type combination).
    ///
    /// Returns None when the op is not SIMD-batchable (e.g. And/Or/RefEq/ConcatList/Range
    /// and other non-scalar arithmetic ops, or non-scalar types).
    fn binary_batch_info(
        &self,
        op: crate::ast::Ast::BinaryOp,
        lhs_expr: crate::ast::Ast::ExprId,
    ) -> Option<BatchInfo> {
        use crate::ast::Ast::BinaryOp;
        use crate::value::{BinOp as VBinOp, CmpOp as VCmpOp};

        let ty = self.expr_type_name(lhs_expr)?;
        let tag = Self::ty_name_to_scalar_tag(ty)?;
        let is_float = scalar_meta(tag).map(|m| m.is_float).unwrap_or(false);

        let batch_op = match op {
            BinaryOp::Add => BatchOp::Bin(VBinOp::Add),
            BinaryOp::Sub => BatchOp::Bin(VBinOp::Sub),
            BinaryOp::Mul => BatchOp::Bin(VBinOp::Mul),
            BinaryOp::Div => BatchOp::Bin(VBinOp::Div),
            BinaryOp::Mod => BatchOp::Bin(VBinOp::Mod),
            BinaryOp::BitAnd if !is_float => BatchOp::Bin(VBinOp::Band),
            BinaryOp::BitOr if !is_float => BatchOp::Bin(VBinOp::Bor),
            BinaryOp::BitXor if !is_float => BatchOp::Bin(VBinOp::Bxor),
            BinaryOp::Shl if !is_float => BatchOp::Bin(VBinOp::Shl),
            BinaryOp::Shr if !is_float => BatchOp::Bin(VBinOp::Shr),
            BinaryOp::Eq => BatchOp::Cmp(VCmpOp::Eq),
            BinaryOp::NotEq => BatchOp::Cmp(VCmpOp::Ne),
            BinaryOp::Lt => BatchOp::Cmp(VCmpOp::Lt),
            BinaryOp::Gt => BatchOp::Cmp(VCmpOp::Gt),
            BinaryOp::LtEq => BatchOp::Cmp(VCmpOp::Le),
            BinaryOp::GtEq => BatchOp::Cmp(VCmpOp::Ge),
            // And/Or/RefEq/RefNeq/ConcatList/Range/RangeInclusive/Elvis -> not batchable
            _ => return None,
        };
        Some(BatchInfo { tag, op: batch_op })
    }

    /// Map a Kuzo UnaryOp + type name to a BatchInfo.
    ///
    /// Neg (integer/float negation) and BitNot (integer bitwise not) are batchable;
    /// Not (bool logical not) does not go through SIMD batching.
    fn unary_batch_info(
        &self,
        op: crate::ast::Ast::UnaryOp,
        operand_expr: crate::ast::Ast::ExprId,
    ) -> Option<BatchInfo> {
        use crate::ast::Ast::UnaryOp;
        use crate::value::UnaryOp as VUnaryOp;

        let ty = self.expr_type_name(operand_expr)?;
        let tag = Self::ty_name_to_scalar_tag(ty)?;
        let is_float = scalar_meta(tag).map(|m| m.is_float).unwrap_or(false);

        let batch_op = match op {
            UnaryOp::Neg => BatchOp::Unary(VUnaryOp::Neg),
            UnaryOp::BitNot if !is_float => BatchOp::Unary(VUnaryOp::Bnot),
            // Not (bool logical not) -> not batchable
            _ => return None,
        };
        Some(BatchInfo { tag, op: batch_op })
    }

    /// Map a type name to a ValueTag (delegates to `ValueTag::from_name`, single-point sync with Value).
    fn ty_name_to_scalar_tag(ty: &str) -> Option<crate::value::ValueTag> {
        crate::value::ValueTag::from_name(ty)
    }

    /// Compile a type cast `expr as T`.
    ///
    /// Two codegen paths, both single-node:
    ///   - target is str: `compute_cast_to_str` (idx 277) — covers scalar/char/bool/array→str
    ///   - scalar→scalar: `compute_cast_scalar` (idx 278) — covers all int↔int/int↔float/char↔int
    fn compile_as_cast(
        &mut self,
        expr: crate::ast::Ast::ExprId,
        target: crate::ast::Ast::TypeRef,
    ) -> NodeId {
        // Get the target type name.
        // In a generic context, target may be a type-parameter name (e.g. "T"); look up
        // current_type_args to replace it with the concrete type name.
        let target_ty = {
            let spanned = &self.current_module().arena.types[target.0 as usize];
            match &spanned.node {
                crate::ast::Ast::TypeNode::Named { name } => {
                    let name = *name;
                    // Type-parameter replacement (monomorphization instance context)
                    if let Some((_, h)) = self.current_type_args.iter().find(|(n, _)| n == name) {
                        if let Some(resolved) = self.type_arena.type_name(*h) {
                            resolved.to_string()
                        } else {
                            name.to_string()
                        }
                    } else {
                        name.to_string()
                    }
                }
                _ => "i64".to_string(),
            }
        };

        // Get the source type name (from Sema expr_types)
        let source_ty = self.expr_type_name(expr).unwrap_or("i64").to_string();

        let input = self.compile_subexpr(expr);

        // Path 1: any type -> str
        if Self::type_family(&target_ty) == crate::types::TypeFamily::Str {
            let inputs_offset = self.graph.inputs_pool.push(&[input]);
            return self.graph.add_node(Node {
                kind: NodeKind::UnOp,
                input_count: 1,
                inputs_offset,
                compute_fn: CF_CAST_TO_STR, // compute_cast_to_str
            });
        }

        // Path 2: scalar -> scalar (int<->int, int<->float, float<->float, bool<->int, char<->int)
        if Self::ty_name_to_scalar_tag(&source_ty).is_some()
            && Self::ty_name_to_scalar_tag(&target_ty).is_some()
        {
            let inputs_offset = self.graph.inputs_pool.push(&[input]);
            let node = self.graph.add_node(Node {
                kind: NodeKind::UnOp,
                input_count: 1,
                inputs_offset,
                compute_fn: CF_CAST_SCALAR, // compute_cast_scalar
            });
            self.graph.set_cast_target_type(node, target_ty.clone());
            return node;
        }

        // Fallback: source or target is a generic type parameter whose concrete type is not yet
        // known (resolved later by monomorphization). Emit a scalar cast node optimistically;
        // `compute_cast_scalar` reads the concrete target at runtime via `cast_target_type`.
        let inputs_offset = self.graph.inputs_pool.push(&[input]);
        let node = self.graph.add_node(Node {
            kind: NodeKind::UnOp,
            input_count: 1,
            inputs_offset,
            compute_fn: CF_CAST_SCALAR, // compute_cast_scalar
        });
        self.graph.set_cast_target_type(node, target_ty.clone());
        node
    }

    /// Compile a function call.
    ///
    /// If the callee is a known function name -> Call node + set_call_target.
    /// If the callee is a type name (e.g. `Iterator(arr, 0)`) -> compile into a record-construction node.
    fn compile_call(
        &mut self,
        call_expr_id: crate::ast::Ast::ExprId,
        callee: crate::ast::Ast::ExprId,
        args: &[crate::ast::Ast::ExprId],
    ) -> NodeId {
        // Generic calls prefer the monomorphization-instance path: inline expansion does not handle
        // type-parameter substitution, so inlining a generic function would leave type parameter T
        // unresolved to a concrete type in the body.
        let call_inst_key = crate::sema::Sema::module_expr_key(
            self.expr_key_module(),
            call_expr_id.0 as u64,
        );
        let is_generic_call = self.sema.call_instantiations.contains_key(&call_inst_key);

        // -- Inline expansion: call sites flagged by the analyzer compile the callee body directly instead of launching a subgraph --
        // pure function + small body + non-recursive -> bind actuals to formals, compile body, avoid call overhead
        // generic calls skip inlining (type parameters need replacement via the monomorph instance subgraph)
        if !is_generic_call {
            if let Some(callee_func) = self.inline_target(call_expr_id) {
                if let crate::ast::Ast::Decl::FunDecl { params, body, .. } =
                    &self.current_module().declarations[callee_func.0 as usize].node
                {
                    return self.compile_inline_expansion(*body, params, args);
                }
            }
        }

        let callee_expr = self.current_module().arena.expr(callee);

        // ── reflect top-level function interception (safety net) ──
        // Historically `format(x)` / `type_name(x)` were top-level wrapper functions;
        // they have been removed in favor of direct method calls (`x.format()`).
        // This interception remains as a safety net: if a future top-level reflect
        // wrapper is reintroduced, it lowers directly to CF_REFLECT_* without going
        // through FFI dispatch. Today it is effectively dead code (no such functions
        // are declared, so Sema rejects bare `format(x)` before reaching here).
        if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
            if let Some(cf) = reflect_top_level_cf(name) {
                let mut inputs = Vec::with_capacity(args.len());
                for &arg in args {
                    inputs.push(self.compile_subexpr(arg));
                }
                let inputs_offset = self.graph.inputs_pool.push(&inputs);
                return self.graph.add_node(Node {
                    kind: NodeKind::Call,
                    input_count: inputs.len() as u8,
                    inputs_offset,
                    compute_fn: cf,
                });
            }
        }

        // Built-in constructor detection: Ok(val) / Err(record) / channel(capacity)
        // Lowered via the BUILTIN_CTORS registry lookup; missed error types fall through to the record construction path below
        if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
            if !self.func_subgraphs.contains_key(*name) {
                if let Some(lower) = BUILTIN_CTORS.iter().find_map(|(n, l)| (*n == *name).then_some(l)) {
                    return match lower {
                        // Ok(val) -> compute_throw_ok (idx 44), input = val
                        BuiltinCtorLower::Ok => {
                            let mut inputs = Vec::with_capacity(args.len());
                            for &arg in args {
                                inputs.push(self.compile_subexpr(arg));
                            }
                            let inputs_offset = self.graph.inputs_pool.push(&inputs);
                            self.graph.add_node(Node {
                                kind: NodeKind::Call,
                                input_count: inputs.len() as u8,
                                inputs_offset,
                                compute_fn: CF_THROW_OK, // throw_ok
                            })
                        }
                        // Err(...) -> first record_construct, then wrap with throw_err (idx 45)
                        BuiltinCtorLower::Err => {
                            let inner = self.compile_record_like(crate::ir::Compute::CTOR_ERR, args);
                            let inputs_offset = self.graph.inputs_pool.push(&[inner]);
                            self.graph.add_node(Node {
                                kind: NodeKind::Call,
                                input_count: 1,
                                inputs_offset,
                                compute_fn: CF_THROW_ERR, // throw_err
                            })
                        }
                        // channel(capacity) -> compute_channel_create (idx 283), input = args
                        BuiltinCtorLower::Channel => {
                            let mut inputs = Vec::with_capacity(args.len());
                            for &arg in args {
                                inputs.push(self.compile_subexpr(arg));
                            }
                            let inputs_offset = self.graph.inputs_pool.push(&inputs);
                            self.graph.add_node(Node {
                                kind: NodeKind::BinOp,
                                input_count: inputs.len() as u8,
                                inputs_offset,
                                compute_fn: CF_CHANNEL_CREATE, // compute_channel_create
                            })
                        }
                    };
                }
            }
        }

        // Type constructor / ADT / Newtype constructor detection: callee is an Ident and not a known function
        if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
            if !self.func_subgraphs.contains_key(*name) {
                // First look up the type name (Record or single-constructor ADT), then the constructor name of a multi-constructor ADT
                let tf_info = self.lookup_type_field_names(name)
                    .or_else(|| self.lookup_constructor_field_names(name));
                if let Some(info) = tf_info {
                    // Compile into a construction node (compute_record_construct = 29, dispatches to HeapObj by kind)
                    let mut inputs = Vec::with_capacity(args.len());
                    for &arg in args {
                        inputs.push(self.compile_subexpr(arg));
                    }
                    let inputs_offset = self.graph.inputs_pool.push(&inputs);
                    let node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: inputs.len() as u8,
                        inputs_offset,
                        compute_fn: CF_RECORD_CONSTRUCT, // record_construct
                    });
                    self.graph.set_record_lit_info(node, RecordLitInfo {
                        type_name: info.type_name.clone(),
                        field_names: info.field_names.into_iter().map(Some).collect(),
                        constructor: name.to_string(),
                        kind: info.kind,
                    });
                    return node;
                }
            }
        }

        // Closure call detection: callee is an Ident, not a known function, but bound in scope (variable holds a Closure/Partial)
        // -> use compute_closure_call (idx 41); inputs[0] = callable value node, inputs[1..1+arg_count] = call arguments
        // current_effect appended at the end as an implicit dependency (ensures Call executes only after prior effects complete)
        // arg_count metadata records the actual argument count (excluding closure value and effect), used for chained partial-application detection
        if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
            if !self.func_subgraphs.contains_key(*name) {
                if let Some(closure_node) = self.lookup_var(name) {
                    let mut inputs = Vec::with_capacity(args.len() + 2);
                    inputs.push(closure_node);
                    for &arg in args {
                        inputs.push(self.compile_subexpr(arg));
                    }
                    if let Some(eff) = self.current_effect {
                        inputs.push(eff);
                    }
                    let inputs_offset = self.graph.inputs_pool.push(&inputs);
                    let call_node = self.graph.add_node(Node {
                        kind: NodeKind::Call,
                        input_count: inputs.len() as u8,
                        inputs_offset,
                        compute_fn: CF_CLOSURE_CALL, // compute_closure_call
                    });
                    self.graph.set_closure_call_arg_count(call_node, args.len() as u8);
                    return call_node;
                }
            }
        }

        // @extern("C") / @builtin call detection: does not launch a sub-frame; calls Ffi::wrapper
        // (extern C) or the Rust #[no_mangle] fn (builtin, e.g. reflect) directly via FFI dispatch.
        // current_effect appended at the end as an implicit dependency (ensures the Call executes
        // only after prior effects complete).
        //
        // @extern("C") call detection: all stdlib `#{ }#` functions go through
        // CF_DYN_FFI_CALL (dlsym self-lookup + Abi::call_dynamic). There is no longer a
        // CF_FFI_CALL / wrapper table. The C symbol name is uniformly `kuzo_extern_<name>`
        // (generated by build.rs).
        if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
            if self.is_extern_c_func(name) {
                let sig = self.build_abi_sig(name);
                let c_symbol = format!("kuzo_extern_{name}");
                let mut inputs = Vec::with_capacity(args.len() + 1);
                for &arg in args {
                    inputs.push(self.compile_subexpr(arg));
                }
                if let Some(eff) = self.current_effect {
                    inputs.push(eff);
                }
                let inputs_offset = self.graph.inputs_pool.push(&inputs);
                let node = self.graph.add_node(Node {
                    kind: NodeKind::Call,
                    input_count: inputs.len() as u8,
                    inputs_offset,
                    compute_fn: CF_DYN_FFI_CALL,
                });
                self.graph.set_dyn_ffi_info(
                    node,
                    crate::ir::Ir::DynFfiInfo {
                        symbol: c_symbol,
                        sig,
                        arg_count: args.len() as u8,
                    },
                );
                return node;
            }
        }

        // Partial application detection: callee is a known function name, but actual count < target function formal count
        // -> generate a partial_construct node (idx 286), producing a HeapObj::Partial
        // bound_args = the already-supplied actuals (binding leading parameters in the original function's parameter order)
        if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
            if let Some(&target_sg) = self.func_subgraphs.get(*name) {
                if let Some(sg) = self.graph.subgraphs.get(target_sg.0 as usize) {
                    let param_count = sg.param_count as usize;
                    if args.len() < param_count {
                        let mut bound_inputs = Vec::with_capacity(args.len());
                        for &arg in args {
                            bound_inputs.push(self.compile_subexpr(arg));
                        }
                        let inputs_offset = self.graph.inputs_pool.push(&bound_inputs);
                        let partial_node = self.graph.add_node(Node {
                            kind: NodeKind::BinOp,
                            input_count: bound_inputs.len() as u8,
                            inputs_offset,
                            compute_fn: CF_PARTIAL_CONSTRUCT, // compute_partial_construct
                        });
                        self.graph.set_partial_info(partial_node, PartialInfo {
                            subgraph_id: target_sg,
                            bound_count: args.len() as u8,
                        });
                        return partial_node;
                    }
                }
            }
        }

        // Dynamic closure call: callee is a non-Ident expression (e.g. arr[i], field.access,
        // a closure literal invoked directly fun() {...}(), etc.), evaluated at runtime to a Closure/Partial.
        // Use compute_closure_call (idx 41) for dynamic invocation; inputs[0] = callable value node.
        // current_effect appended at the end as an implicit dependency (consistent with the Ident closure call path).
        if !matches!(&callee_expr.node, crate::ast::Ast::Expr::Ident(_)) {
            let callable_node = self.compile_subexpr(callee);
            let mut inputs = Vec::with_capacity(args.len() + 2);
            inputs.push(callable_node);
            for &arg in args {
                inputs.push(self.compile_subexpr(arg));
            }
            if let Some(eff) = self.current_effect {
                inputs.push(eff);
            }
            let inputs_offset = self.graph.inputs_pool.push(&inputs);
            let call_node = self.graph.add_node(Node {
                kind: NodeKind::Call,
                input_count: inputs.len() as u8,
                inputs_offset,
                compute_fn: CF_CLOSURE_CALL, // compute_closure_call
            });
            self.graph.set_closure_call_arg_count(call_node, args.len() as u8);
            return call_node;
        }

        // Tail-recursion-to-iteration interception: currently compiling in a TailRecToLoop body and callee is self_name,
        // and in tail position (to avoid recursive calls in arguments being mistakenly intercepted),
        // generate WriteBack(param, actual) instead of Call(self).
        // body_sg is a LoopBody; after it completes, reset_loop_iteration automatically jumps back to while_sg to re-evaluate cond.
        if self.in_tail_position && self.tail_rec_ctx.is_some() {
            // Tail-recursion interception: generate WriteBack(param, actual) instead of Call(self)
        }
        if self.in_tail_position {
            if let Some(ctx) = &self.tail_rec_ctx.clone() {
                if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
                    if *name == ctx.self_name {
                    // Compile all actual-argument expressions (evaluate first, then WriteBack, to avoid races between parameters)
                    let arg_nodes: Vec<NodeId> = args
                        .iter()
                        .map(|&a| self.compile_subexpr(a))
                        .collect();
                    // Perform a WriteBack for each parameter (writing back to the function-level param nodes).
                    // Barrier mechanism: the first WriteBack depends on all arg_nodes;
                    // subsequent WriteBacks chain-depend on the previous WriteBack.
                    // This ensures all actual-argument expressions finish evaluating before any WriteBack executes,
                    // preventing a+b from reading the already-WriteBack-updated value of a.
                    //
                    // Only the last WriteBack uses CF_TAILREC_WRITEBACK (sets Continue);
                    // non-last WriteBacks use CF_WRITEBACK (do not set Continue).
                    // Reason: the Continue signal causes the frame to exit immediately and skip notify_downstream;
                    // if every WriteBack set Continue, subsequent chained WriteBacks would never become ready to execute.
                    let wb_count = arg_nodes.len().min(ctx.param_nodes.len());
                    let mut last_wb: Option<NodeId> = None;
                    for (i, &arg_node) in arg_nodes.iter().enumerate() {
                        if i < ctx.param_nodes.len() {
                            let mut wb_inputs = vec![arg_node];
                            if i == 0 {
                                // First WB: a barrier depending on all other arg_nodes
                                for &other in &arg_nodes[1..] {
                                    wb_inputs.push(other);
                                }
                            } else if let Some(prev_wb) = last_wb {
                                // Subsequent WBs: depend on the previous WB (chain ordering)
                                wb_inputs.push(prev_wb);
                            }
                            let is_last = i + 1 == wb_count;
                            let compute_fn = if is_last {
                                CF_TAILREC_WRITEBACK
                            } else {
                                CF_WRITEBACK
                            };
                            let wb_off = self.graph.inputs_pool.push(&wb_inputs);
                            let wb_node = self.graph.add_node(Node {
                                kind: NodeKind::Call,
                                input_count: wb_inputs.len() as u8,
                                inputs_offset: wb_off,
                                compute_fn,
                            });
                            self.graph.set_writeback_target(wb_node, ctx.param_nodes[i]);
                            self.current_effect = Some(wb_node);
                            last_wb = Some(wb_node);
                        }
                    }
                    // Return the last WriteBack node (after body_sg completes, reset_loop_iteration automatically jumps back)
                    return last_wb.unwrap_or_else(|| {
                        let off = self.graph.inputs_pool.push(&[]);
                        self.graph.add_node(Node {
                            kind: NodeKind::Const,
                            input_count: 0,
                            inputs_offset: off,
                            compute_fn: CF_NOOP,
                        })
                    });
                    }
                }
            }
        }

        // Non-tail-recursion-to-iteration interception: a self-call in non-tail position is replaced with push continuation + push sub-task + barrier(Continue)
        // Only intercepted when non_tail_rec_ctx is set (during state_N_sg compilation in compile_non_tail_rec_body_sg)
        if !self.in_tail_position && self.non_tail_rec_ctx.is_some() {
            let ctx_clone = self.non_tail_rec_ctx.clone();
            if let Some(ctx) = &ctx_clone {
                if let crate::ast::Ast::Expr::Ident(callee_name) = &callee_expr.node {
                    if *callee_name == ctx.self_name {
                        // 1. Check call_result_map: if the current call is already in the map, return the mapped node
                        if let Some(&mapped) = ctx.call_result_map.get(&call_expr_id) {
                            return mapped;
                        }
                        // 2. If already truncated, return a void constant (no Call node generated)
                        if ctx.truncated {
                            return self.compile_void_const();
                        }
                        // 3. Intercept: push continuation frame + push sub-task frame + barrier(Continue)

                        // Save current_effect: compile_subexpr may modify it;
                        // restore after compiling the actuals to ensure the store chain starts from the correct effect.
                        let saved_effect = self.current_effect;
                        let arg_nodes: Vec<NodeId> = args
                            .iter()
                            .map(|&a| self.compile_subexpr(a))
                            .collect();
                        self.current_effect = saved_effect;

                        let stride = ctx.stride;
                        let param_count = ctx.param_count;
                        let max_saved = ctx.max_saved;
                        let current_state = ctx.current_state as usize;
                        let stack_node = ctx.stack_node;
                        let sp_node = ctx.sp_node;
                        let result_node = ctx.result_node;

                        // Compute stack indices: base_cont = sp * stride, base_task = (sp + 1) * stride
                        // sp has already been decremented by pop (sp_node = original_sp - 1)
                        // cont writes to the slot freed by pop (overwriting the consumed frame); task writes to the next slot
                        // sp_new = sp + 2; on pop, sp-1 reads task first (LIFO), then cont
                        let one_const = self.make_i32_const(1);
                        let sp_plus_1 = self.make_binop(sp_node, one_const, CF_ADD_I32);
                        let two_const = self.make_i32_const(2);
                        let sp_plus_2 = self.make_binop(sp_node, two_const, CF_ADD_I32);
                        let stride_val = self.make_i32_const(stride as i32);
                        let base_cont = self.make_binop(sp_node, stride_val, CF_MUL_I32);
                        let base_task = self.make_binop(sp_plus_1, stride_val, CF_MUL_I32);

                        // Push continuation frame (write to the slot freed by pop)
                        // stack[base_cont + 0..P] = current parameters (param_cur nodes)
                        // All stores must be chained into the effect chain via chain_effects,
                        // ensuring the barrier triggers Continue only after all stores complete.
                        for i in 0..param_count {
                            let offset = self.make_i32_const(i as i32);
                            let idx = self.make_binop(base_cont, offset, CF_ADD_I32);
                            let store = self.make_array_store(stack_node, idx, ctx.param_nodes[i]);
                            self.current_effect = Some(self.chain_effects(self.current_effect, store));
                        }
                        // stack[base_cont + P] = state_after (current state + 1)
                        let state_after = self.make_i32_const((current_state + 1) as i32);
                        let state_offset_cont = self.make_i32_const(param_count as i32);
                        let state_idx_cont = self.make_binop(base_cont, state_offset_cont, CF_ADD_I32);
                        let state_store_cont =
                            self.make_array_store(stack_node, state_idx_cont, state_after);
                        self.current_effect = Some(self.chain_effects(self.current_effect, state_store_cont));
                        // stack[base_cont + P + 1..P + 1 + num_saved] = saved values
                        // For state S: slot j = saved_nodes[j] (j < S-1), result_node (j == S-1), 0 (j >= S)
                        let zero_saved = self.make_i32_const(0);
                        for j in 0..max_saved {
                            let offset = self.make_i32_const((param_count + 1 + j) as i32);
                            let idx = self.make_binop(base_cont, offset, CF_ADD_I32);
                            let val = if j < current_state {
                                if j + 1 < current_state {
                                    ctx.saved_nodes[j]
                                } else {
                                    // j == current_state - 1
                                    result_node
                                }
                            } else {
                                zero_saved
                            };
                            let store = self.make_array_store(stack_node, idx, val);
                            self.current_effect = Some(self.chain_effects(self.current_effect, store));
                        }

                        // Push sub-task frame (top of stack; read first on pop)
                        // stack[base_task + 0..P] = actuals (arg_nodes)
                        for i in 0..param_count {
                            let offset = self.make_i32_const(i as i32);
                            let idx = self.make_binop(base_task, offset, CF_ADD_I32);
                            let store = self.make_array_store(stack_node, idx, arg_nodes[i]);
                            self.current_effect = Some(self.chain_effects(self.current_effect, store));
                        }
                        // stack[base_task + P] = 0 (INIT state)
                        let state_offset_task = self.make_i32_const(param_count as i32);
                        let state_idx_task = self.make_binop(base_task, state_offset_task, CF_ADD_I32);
                        let state_store_task =
                            self.make_array_store(stack_node, state_idx_task, zero_saved);
                        self.current_effect = Some(self.chain_effects(self.current_effect, state_store_task));
                        // stack[base_task + P + 1..P + 1 + max_saved] = 0
                        for j in 0..max_saved {
                            let offset = self.make_i32_const((param_count + 1 + j) as i32);
                            let idx = self.make_binop(base_task, offset, CF_ADD_I32);
                            let store = self.make_array_store(stack_node, idx, zero_saved);
                            self.current_effect = Some(self.chain_effects(self.current_effect, store));
                        }

                        // WriteBack sp = sp + 2 (chained into effect to ensure it runs after all stores)
                        let sp_new = self.chain_effects(self.current_effect, sp_plus_2);
                        let sp_wb = self.compile_writeback_node(sp_new, sp_node);
                        self.current_effect = Some(sp_wb);

                        // Create the barrier node (Continue signal; blocks subsequent expression execution)
                        let barrier = self.make_continue_barrier(sp_wb);
                        self.current_effect = Some(barrier);

                        // Set the truncated flag
                        if let Some(ctx) = &mut self.non_tail_rec_ctx {
                            ctx.truncated = true;
                        }

                        return barrier;
                    }
                }
            }
        }

        // Regular function call
        // current_effect appended at the end as an implicit dependency (ensures Call executes only after prior effects complete)
        let mut inputs = Vec::with_capacity(args.len() + 1);
        for &arg in args {
            inputs.push(self.compile_subexpr(arg));
        }
        if let Some(eff) = self.current_effect {
            inputs.push(eff);
        }
        let inputs_offset = self.graph.inputs_pool.push(&inputs);
        // Default sync call compute_fn (idx 36); async functions use compute_async_call_launch (idx 39)
        let call_node = self.graph.add_node(Node {
            kind: NodeKind::Call,
            input_count: inputs.len() as u8,
            inputs_offset,
            compute_fn: CF_CALL_LAUNCH, // compute_call_launch (sync)
        });

        // Bind the target subgraph (if the callee is a known function name)
        // Prefer call_instantiations lookup: generic call site -> monomorphization instance, bind the specialized subgraph by mangled name
        if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
            let inst_id = self.sema.call_instantiations.get(&call_inst_key);
            let mangled = inst_id.map(|&id| format!("{}#{}", name, id));
            let target_key: &str = mangled.as_deref().unwrap_or(name);
            // Try mangled name first; fall back to bare name if not found.
            // This handles: (a) non-generic instances with empty type_args that were
            // never registered under the mangled name, and (b) cross-module calls
            // compiled before step 2d pre-registers the instance placeholder.
            let resolved_sg = self.func_subgraphs.get(target_key)
                .or_else(|| self.func_subgraphs.get(*name));
            if std::env::var("KUZO_DEBUG_BUILD").is_ok() {
                let sg_info = resolved_sg.map(|&sg_id| {
                    let sg = &self.graph.subgraphs[sg_id.0 as usize];
                    let (s, e) = sg.node_range;
                    format!("sg={} nodes=[{},{})", sg_id.0, s.0, e.0)
                }).unwrap_or_else(|| "NOT FOUND".to_string());
                let used_key = if self.func_subgraphs.get(target_key).is_some() {
                    target_key
                } else {
                    *name
                };
                eprintln!("[CALL-BIND] callee={:?} target_key={:?} resolved_key={:?} inst_id={:?} sg_info={} cur_mod={:?}",
                    name, target_key, used_key, inst_id, sg_info,
                    self.current_module().name);
            }
            if let Some(&target_sg) = resolved_sg {
                self.graph.set_call_target(call_node, target_sg);
                // is_async is derived at runtime by compute_call_launch from has_suspend;
                // here we only check has_suspend to decide whether the call can be marked tail.
                let is_async = self.graph.subgraphs.get(target_sg.0 as usize)
                    .is_some_and(|sg| sg.has_suspend);
                // Tail-call marker: tail position + sync function + has call_target -> runtime switch_subgraph frame reuse
                if self.in_tail_position && !is_async {
                    self.graph.set_tail_call(call_node);
                }
            }
        }

        call_node
    }

    /// Inline expansion: compile the callee body with formals bound to actual nodes.
    ///
    /// Enter a new scope -> compile actuals -> bind formal names -> compile body (non-tail position) -> exit scope.
    /// Does not generate a Call node or launch a subgraph; embeds the body's IR directly into the current function.
    fn compile_inline_expansion(
        &mut self,
        body: crate::ast::Ast::ExprRef,
        params: &[crate::ast::Ast::Param<'_>],
        args: &[crate::ast::Ast::ExprId],
    ) -> NodeId {
        self.enter_scope();
        // Compile actuals and bind to formal names (actual nodes are compiled in the current scope context)
        for (param, &arg) in params.iter().zip(args.iter()) {
            let arg_node = self.compile_subexpr(arg);
            self.bind_var(param.name, arg_node);
        }
        // Compile the callee body (non-tail position; inline expansion does not preserve tail-call semantics)
        let body_node = self.compile_subexpr(body);
        self.exit_scope();
        body_node
    }

    /// Look up a type declaration's field info (by type name).
    ///
    /// Uniformly searches layer by layer through type_scope_stack (top-level + nested types share the same lookup path).
    fn lookup_type_field_names(&self, type_name: &str) -> Option<TypeFieldInfo> {
        self.lookup_type_fields(type_name)
    }

    /// Look up the field info of a specified constructor in a multi-constructor ADT.
    ///
    /// Uniformly searches layer by layer through type_scope_stack (top-level + nested types share the same lookup path).
    fn lookup_constructor_field_names(&self, constructor_name: &str) -> Option<TypeFieldInfo> {
        self.lookup_type_fields(constructor_name)
    }

    /// Check if `Type.Ctor` is a qualified constructor access (IR-side).
    /// Returns `(type_name, ctor_name, field_names, kind, is_nullary)` for
    /// constructing the IR node.
    fn check_qualified_ctor_ir(
        &self,
        type_name: &str,
        ctor_name: &str,
    ) -> Option<(String, String, Vec<Option<String>>, RecordLitKind, bool)> {
        let &type_idx = self.sema.type_def_index.get(type_name)?;
        let type_def = &self.sema.type_defs[&type_idx];
        let ctor = type_def
            .constructors
            .iter()
            .find(|c| c.name.as_ref() == ctor_name)?;
        let field_names: Vec<Option<String>> = ctor
            .field_names
            .iter()
            .map(|n| n.as_deref().map(String::from))
            .collect();
        let is_nullary = ctor.field_type_reprs.is_empty();
        let kind = match type_def.kind {
            crate::sema::Sema::TypeDefKind::Adt => RecordLitKind::Adt,
            crate::sema::Sema::TypeDefKind::Record => RecordLitKind::Record,
            crate::sema::Sema::TypeDefKind::Newtype => RecordLitKind::Newtype,
            crate::sema::Sema::TypeDefKind::Alias => RecordLitKind::Record,
        };
        Some((
            ctor.type_name.to_string(),
            ctor.name.to_string(),
            field_names,
            kind,
            is_nullary,
        ))
    }

    /// Check whether a function name is an @extern("C") function (has extern_c_body).
    fn is_extern_c_func(&self, name: &str) -> bool {
        let modules: Vec<&crate::ast::Ast::Module<'_>> =
            std::iter::once(self.module).chain(self.builtin_modules.iter().copied()).collect();
        for m in modules {
            if let Some(d) = m.find_function(name) {
                if let crate::ast::Ast::Decl::FunDecl { extern_c_body, .. } = &d.node {
                    if extern_c_body.is_some() {
                        return true;
                    }
                }
            }
        }
        false
    }

    // (is_builtin_func removed: @builtin attribute eliminated. Reflect primitives
    // now lower as auto-impl trait methods / top-level wrappers to CF_REFLECT_*
    // compute_fns, never reaching the @extern dispatch path.)

    /// Build an AbiSig from a @extern("C") #{ }# function's declared params/return_type.
    /// str params are expanded to (Ptr, Int) two slots. Unknown types fall back to Int{64}.
    fn build_abi_sig(&self, name: &str) -> crate::ffi::Abi::AbiSig {
        use crate::ast::Ast::Decl;
        let mut params = Vec::new();
        let mut ret = crate::ffi::Abi::AbiType::Void;
        for m in std::iter::once(self.module).chain(self.builtin_modules.iter().copied()) {
            if let Some(d) = m.find_function(name) {
                if let Decl::FunDecl { params: decl_params, return_type, .. } = &d.node {
                    // Use THIS module's arena (m.arena), not self.module.arena — stdlib functions
                    // have their TypeRefs in the builtin module's arena.
                    let arena = &m.arena;
                    for p in decl_params.iter() {
                        let ty_name = type_name_in_arena(p.type_annotation, arena);
                        self.push_abi_types(&ty_name, &mut params);
                    }
                    if let Some(rt) = return_type {
                        let rt_name = type_name_in_arena(Some(*rt), arena);
                        ret = self.abi_type_of(&rt_name);
                    }
                    break;
                }
            }
        }
        crate::ffi::Abi::AbiSig::new(params, ret)
    }

    /// Map a Kuzo type name to AbiType. str is handled separately by push_abi_types (two slots).
    fn abi_type_of(&self, ty_name: &str) -> crate::ffi::Abi::AbiType {
        use crate::ffi::Abi::AbiType;
        match ty_name {
            "void" => AbiType::Void,
            "i8" => AbiType::Int { bits: 8, signed: true },
            "i16" => AbiType::Int { bits: 16, signed: true },
            "i32" => AbiType::Int { bits: 32, signed: true },
            "i64" | "isize" => AbiType::Int { bits: 64, signed: true },
            "u8" | "bool" => AbiType::Int { bits: 8, signed: false },
            "char" => AbiType::Int { bits: 32, signed: false },
            "u16" => AbiType::Int { bits: 16, signed: false },
            "u32" => AbiType::Int { bits: 32, signed: false },
            "u64" | "usize" => AbiType::Int { bits: 64, signed: false },
            "f32" => AbiType::Float32,
            "f64" => AbiType::Float64,
            _ if ty_name.starts_with('*') => AbiType::Ptr,
            _ => AbiType::Int { bits: 64, signed: true }, // fallback
        }
    }

    /// Push AbiType(s) for a Kuzo type name. str expands to (Ptr, Int) two slots.
    fn push_abi_types(&self, ty_name: &str, out: &mut Vec<crate::ffi::Abi::AbiType>) {
        if ty_name == "str" {
            // str → (const char* data, size_t len)
            out.push(crate::ffi::Abi::AbiType::Ptr);
            out.push(crate::ffi::Abi::AbiType::Int { bits: 64, signed: false });
        } else {
            out.push(self.abi_type_of(ty_name));
        }
    }

    /// Compile a method call.
    ///
    /// Method dispatch uniformly goes through the (type_id, method_idx) path:
    /// - intrinsic methods (await/len/send/recv/close/bytes/cancel etc.) are flagged via
    ///   MethodSigInfo.intrinsic and lowered directly to a compute_fn node
    /// - type/trait methods are compiled into Call nodes, looking up method_subgraphs via (type_id, method_idx)
    fn compile_method_call(
        &mut self,
        call_expr_id: crate::ast::Ast::ExprId,
        recv: crate::ast::Ast::ExprId,
        method: &str,
        args: &[crate::ast::Ast::ExprId],
        recv_node_override: Option<NodeId>,
    ) -> NodeId {
        // Qualified-name constructor: Type.Ctor(args) (constructor with parameters)
        if let crate::ast::Ast::Expr::Ident(type_name) = &self.current_module().arena.expr(recv).node {
            if let Some((ctor_type_name, ctor_name, field_names, kind, is_nullary)) =
                self.check_qualified_ctor_ir(type_name, method)
            {
                if !is_nullary {
                    let mut inputs = Vec::with_capacity(args.len());
                    for &arg in args {
                        inputs.push(self.compile_subexpr(arg));
                    }
                    let inputs_offset = self.graph.inputs_pool.push(&inputs);
                    let node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: inputs.len() as u8,
                        inputs_offset,
                        compute_fn: CF_RECORD_CONSTRUCT,
                    });
                    self.graph.set_record_lit_info(node, RecordLitInfo {
                        type_name: ctor_type_name,
                        field_names,
                        constructor: ctor_name,
                        kind,
                    });
                    return node;
                }
            }
        }

        // When the caller supplies a pre-compiled receiver node (e.g. implicit-this method
        // calls where `this` is already bound), skip re-compiling the recv expression.
        let recv_node = match recv_node_override {
            Some(n) => n,
            None => self.compile_subexpr(recv),
        };

        // -- intrinsic lowering --
        // First look up the language-level intrinsic flag in sema method_dispatches (await/recv);
        // on miss, fall back to (type_id, method_idx) lookup of MethodSigInfo.intrinsic (send/close/len etc.).
        // When conditions are not met (e.g. argument count mismatch), fall through to the Call node path.
        let dispatch_intrinsic = {
            let key = crate::sema::Sema::module_expr_key(
                self.expr_key_module(),
                call_expr_id.0 as u64,
            );
            self.sema.method_dispatches.get(&key).and_then(|d| d.intrinsic)
        };
        let intrinsic = dispatch_intrinsic.or_else(|| self.lookup_intrinsic(recv, method));
        if let Some(intrinsic) = intrinsic {
            if let Some(node) = self.try_lower_intrinsic(recv, recv_node, args, intrinsic) {
                return node;
            }
        }

        // Path 0: module function call (recv is a constructor/module namespace; recv is not passed)
        // recv flagged by sema MethodCall path 0a/0b: ModuleRef.free_func(args) / TypeName.free_func(args)
        // recv is not passed as an argument (free_func is a free function; it does not take recv)
        // Generic calls prefer call_instantiations lookup to bind the specialized subgraph by mangled name;
        // non-generic calls fall back to the bare name (same mangled-lookup logic as compile_call)
        {
            let recv_key = crate::sema::Sema::module_expr_key(
                self.expr_key_module(),
                recv.0 as u64,
            );
            if self.sema.module_func_recv_exprs.contains(&recv_key) {
                let call_inst_key = crate::sema::Sema::module_expr_key(
                    self.expr_key_module(),
                    call_expr_id.0 as u64,
                );
                let inst_id = self.sema.call_instantiations.get(&call_inst_key);
                let mangled = inst_id.map(|&id| format!("{}#{}", method, id));
                let target_key: &str = mangled.as_deref().unwrap_or(method);
                if let Some(&target_sg) = self.func_subgraphs.get(target_key) {
                    let mut inputs = Vec::with_capacity(args.len() + 1);
                    for &arg in args {
                        inputs.push(self.compile_subexpr(arg));
                    }
                    if let Some(eff) = self.current_effect {
                        inputs.push(eff);
                    }
                    let inputs_offset = self.graph.inputs_pool.push(&inputs);
                    let call_node = self.graph.add_node(Node {
                        kind: NodeKind::Call,
                        input_count: inputs.len() as u8,
                        inputs_offset,
                        compute_fn: CF_CALL_LAUNCH,
                    });
                    self.graph.set_call_target(call_node, target_sg);
                    return call_node;
                }
            }
        }

        {
            // Type-driven method dispatch: (type_id, method_idx) structured key lookup into method_subgraphs
            // current_effect appended at the end as an implicit dependency (ensures Call executes only after prior effects complete)
            let mut inputs = Vec::with_capacity(2 + args.len());
            inputs.push(recv_node);
            for &arg in args {
                inputs.push(self.compile_subexpr(arg));
            }
            if let Some(eff) = self.current_effect {
                inputs.push(eff);
            }
            let inputs_offset = self.graph.inputs_pool.push(&inputs);
            let call_node = self.graph.add_node(Node {
                kind: NodeKind::Call,
                input_count: inputs.len() as u8,
                inputs_offset,
                compute_fn: CF_CALL_LAUNCH,
            });

            // Dispatch priority (semantic priority, not fallback):
            //   1. trait object dynamic dispatch (recv type is a trait -> vtable runtime dispatch)
            //   2. type's own method / trait method override: (type_id, method_idx) lookup into method_subgraphs
            //   3. trait default method: (type_id, trait_def_idx, method_idx_in_trait) lookup into trait_default_subgraphs

            // Path 1: trait object dynamic dispatch (vtable)
            if self.is_trait_object_recv(recv) {
                // Look up method_idx from trait_def.methods (consistent with TraitValue.method_values index)
                let trait_name = self.expr_type_name(recv).unwrap_or("").to_string();
                let method_idx = self.sema.get_trait_def(&trait_name)
                    .and_then(|td| td.methods.iter().position(|m| m.name.as_ref() == method))
                    .map(|i| i as u16);
                match method_idx {
                    Some(idx) => {
                        self.graph.set_vtable_call(call_node, idx);
                        // Populate the vtable_fallback_dispatch table so that when a concrete
                        // record (not a TraitVal) is passed as a trait-typed parameter, the
                        // runtime can statically dispatch via (method_idx, type_name) → SubGraphId.
                        // Enumerate all types implementing this trait via the witness table.
                        self.populate_vtable_fallback(&trait_name, idx, method);
                    }
                    None => self.errors.push(format!(
                        "internal: trait method '{}' not found in trait '{}' for vtable dispatch",
                        method, &trait_name)),
                }
                return call_node;
            }

            // Path 2: type's own method / trait method override
            // When recv_node_override is set (implicit-this call), the callee ExprId does not carry
            // the receiver's type info; use current_method_type (set by the enclosing method compile).
            let recv_type: Option<(&str, u16)> = if recv_node_override.is_some() {
                self.current_method_type.as_ref().map(|(n, id)| (n.as_ref(), *id))
            } else {
                self.expr_type_name(recv).zip(self.expr_type_id(recv))
            };
            if let Some((type_name, type_id)) = recv_type {
                if let Some(method_idx) = self.sema.lookup_method_idx(type_name, method) {
                    if let Some(&target_sg) = self.method_subgraphs.get(&(type_id, method_idx)) {
                        self.graph.set_call_target(call_node, target_sg);
                        return call_node;
                    }
                }
            }

            // Path 3: trait default method (type does not override; fall back to the monomorphized specialized version of the trait default impl)
            let path3_type_id = if recv_node_override.is_some() {
                self.current_method_type.as_ref().map(|(_, id)| *id)
            } else {
                self.expr_type_id(recv)
            };
            if let Some(type_id) = path3_type_id {
                for trait_def in self.sema.trait_defs.values() {
                    if !self.type_implements_trait(type_id, &trait_def.name) {
                        continue;
                    }
                    if let Some(method_idx) = trait_def
                        .methods
                        .iter()
                        .position(|m| m.name.as_ref() == method && m.has_body)
                    {
                        if let Some(&trait_idx) = self.sema.trait_def_index.get(trait_def.name.as_ref()) {
                            if let Some(&target_sg) = self.trait_default_subgraphs.get(&(type_id, trait_idx, method_idx as u16)) {
                                self.graph.set_call_target(call_node, target_sg);
                                return call_node;
                            }
                        }
                    }
                }
            }

            // Path 4: free-function method call (recv.method(args) -> method(recv, args))
            // When the method name matches a top-level free function, recv is passed as the first argument
            if let Some(&target_sg) = self.func_subgraphs.get(method) {
                self.graph.set_call_target(call_node, target_sg);
                return call_node;
            }

            call_node
        }
    }

    /// Look up MethodSigInfo.intrinsic via (type_id, method_idx) and return the lowering strategy.
    ///
    /// Intrinsic methods of built-in types (e.g. Async.await, Channel.send, Array.len) have the intrinsic
    /// field annotated when Sema registers the synthetic TypeDefInfo; this lookups uniformly, without special-casing by method name.
    fn lookup_intrinsic(
        &self,
        recv: crate::ast::Ast::ExprId,
        method: &str,
    ) -> Option<crate::sema::Sema::IntrinsicKind> {
        // First: reflect trait methods (auto-impl). Recognized structurally by
        // method name, so every type — including builtins and generic type vars —
        // gets reflect methods without needing witness-table registration.
        if let Some((kind, _argc)) = reflect_method_intrinsic(method) {
            // Guard: only lower as reflect intrinsic if the receiver's type does
            // NOT already define a real method of the same name (user override wins).
            let shadows = self.expr_type_name(recv)
                .and_then(|tn| self.sema.lookup_method_idx(tn, method))
                .is_some();
            if !shadows {
                return Some(kind);
            }
        }
        let type_name = self.expr_type_name(recv)?;
        let type_id = self.expr_type_id(recv)?;
        let method_idx = self.sema.lookup_method_idx(type_name, method)?;
        let sig = self.sema.get_method_sig(type_id, method_idx)?;
        sig.intrinsic
    }

    /// Try to lower to a compute_fn node based on IntrinsicKind.
    ///
    /// Returns None when conditions are not met (e.g. argument count mismatch, recv type mismatch);
    /// the caller should fall through to the Call node path.
    fn try_lower_intrinsic(
        &mut self,
        recv: crate::ast::Ast::ExprId,
        recv_node: NodeId,
        args: &[crate::ast::Ast::ExprId],
        kind: crate::sema::Sema::IntrinsicKind,
    ) -> Option<NodeId> {
        use crate::sema::Sema::IntrinsicKind;
        match kind {
            // await: unconditionally lower to Await (EventSource + Await dual node)
            IntrinsicKind::Await if args.is_empty() => {
                Some(self.build_await_node(recv, recv_node))
            }
            // recv: only lower to Await when recv's type is Channel/Receiver
            IntrinsicKind::ChannelAwait if args.is_empty() => {
                if self.infer_event_source_kind(recv) == crate::ir::Ir::EventSourceKind::Channel {
                    Some(self.build_await_node(recv, recv_node))
                } else {
                    None
                }
            }
            // cancel/len/close/bytes: single-node unary op (no arguments)
            IntrinsicKind::UnOp(idx) if args.is_empty() => {
                let inputs_offset = self.graph.inputs_pool.push(&[recv_node]);
                Some(self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: 1,
                    inputs_offset,
                    compute_fn: ComputeFnId(idx),
                }))
            }
            // send(value): binary op, inputs = [recv, value]
            IntrinsicKind::BinOp(idx) => {
                let mut inputs = vec![recv_node];
                for &arg in args {
                    inputs.push(self.compile_subexpr(arg));
                }
                let inputs_offset = self.graph.inputs_pool.push(&inputs);
                Some(self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: inputs.len() as u8,
                    inputs_offset,
                    compute_fn: ComputeFnId(idx),
                }))
            }
            // compare_exchange(expected, new): ternary op, inputs = [recv, expected, new]
            IntrinsicKind::TriOp(idx) => {
                let mut inputs = vec![recv_node];
                for &arg in args {
                    inputs.push(self.compile_subexpr(arg));
                }
                let inputs_offset = self.graph.inputs_pool.push(&inputs);
                Some(self.graph.add_node(Node {
                    kind: NodeKind::TriOp,
                    input_count: inputs.len() as u8,
                    inputs_offset,
                    compute_fn: ComputeFnId(idx),
                }))
            }
            _ => None, // argument mismatch; fall through to the Call node path
        }
    }

    /// Check whether a type implements a specified trait (queries any method slot via witness_table).
    fn type_implements_trait(&self, type_id: u16, trait_name: &str) -> bool {
        for entry in self.sema.witness_table.entries() {
            if entry.trait_name.as_ref() == trait_name && entry.type_id == type_id {
                return true;
            }
        }
        false
    }

    /// Populate the vtable_fallback_dispatch table for a trait method call site.
    ///
    /// For every concrete type that implements `trait_name`, resolve the method's
    /// subgraph via `(type_id, method_idx_in_type_def)` and store it keyed by
    /// `(vtable_method_idx, type_name)`. At runtime, when a vtable call receives a
    /// concrete record (not a TraitVal), `compute_call_launch` looks up the value's
    /// `type_name` here to statically dispatch.
    fn populate_vtable_fallback(&mut self, trait_name: &str, vtable_idx: u16, method_name: &str) {
        // Approach: scan all TypeDecls in the user module (top-level + local) for types that
        // have a method matching `method_name`. For each, register the method subgraph keyed by
        // (vtable_idx, type_name). This handles both explicit trait declarations (`: Trait`) and
        // structural trait implementations (methods present without explicit declaration).
        //
        // First try the witness table (explicit declarations); if empty, fall back to scanning
        // all types with a matching method name.
        let mut entries: Vec<(u16, u16)> = Vec::new();
        for entry in self.sema.witness_table.entries() {
            if entry.trait_name.as_ref() != trait_name {
                continue;
            }
            if let Some(type_method_idx) = self.sema.witness_table.resolve_method(trait_name, entry.type_id, method_name) {
                entries.push((entry.type_id, type_method_idx));
            }
        }
        // Structural fallback: if witness table has no entries for this trait, scan all types
        // that have a method with the matching name. This supports `type Dog { fun name(): str }`
        // being passed as `Animal` without an explicit `: Animal` declaration.
        if entries.is_empty() {
            for (&type_idx, type_def) in &self.sema.type_defs {
                for (m_idx, m) in type_def.methods.iter().enumerate() {
                    if m.name.as_ref() == method_name {
                        entries.push((crate::types::dynamic_type_id(type_idx), m_idx as u16));
                        break;
                    }
                }
            }
        }
        for (type_id, type_method_idx) in entries {
            if let Some(&sg) = self.method_subgraphs.get(&(type_id, type_method_idx)) {
                if let Some(name) = self.type_name_from_id(type_id) {
                    self.graph.vtable_fallback_dispatch.insert((vtable_idx, name.into_boxed_str()), sg);
                }
            }
        }
    }

    /// Reverse-lookup a type_name from a dynamic type_id.
    fn type_name_from_id(&self, type_id: u16) -> Option<String> {
        for (&type_idx, type_def) in &self.sema.type_defs {
            if crate::types::dynamic_type_id(type_idx) == type_id {
                return Some(type_def.name.as_ref().to_string());
            }
        }
        None
    }

    /// Determine whether recv is a trait object (needs runtime dynamic dispatch).
    ///
    /// Look up recv's type name; if it is a trait name registered in sema.trait_defs, it needs vtable dynamic dispatch.
    fn is_trait_object_recv(&self, recv: crate::ast::Ast::ExprId) -> bool {
        let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), recv.0 as u64);
        if let Some(info) = self.sema.expr_types.get(&key) {
            if let Some(tn) = &info.type_name {
                return self
                    .sema
                    .trait_defs
                    .values()
                    .any(|td| td.name.as_ref() == tn.as_ref());
            }
        }
        false
    }

    /// Get the type_id of an expression (looked up from SemaResult.expr_types).
    ///
    /// type_id computation is consistent with populate_witness_table: type_def_index[name] + FIRST_DYNAMIC_TYPE_ID.
    /// When Sema has no record (self in a specialized trait default method version), look up sema's
    /// TraitDefaultInstance.type_name to get the concrete implementation type name, then query type_def_index.
    fn expr_type_id(&self, expr: crate::ast::Ast::ExprId) -> Option<u16> {
        // self in a specialized trait default method version: consume sema's TraitDefaultInstance.type_name
        if let Some(idx) = self.current_trait_default_idx {
            if let crate::ast::Ast::Expr::Ident(name) = &self.module.arena.expr(expr).node {
                if *name == "this" {
                    if let Some(inst) = self.sema.trait_default_instances.get(idx) {
                        return self
                            .sema
                            .type_def_index
                            .get(inst.type_name.as_ref())
                            .map(|&idx| crate::types::dynamic_type_id(idx));
                    }
                }
            }
        }
        let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), expr.0 as u64);
        let info = self.sema.expr_types.get(&key)?;
        // Consistent with expr_type_name: prefer type_name, fall back to Type::name()
        // (built-in structural variants like array/nullable/str/Throw return their registered name via Type::name();
        // "unknown" only appears in degenerate paths where Adt/Record arena lookup fails).
        let type_name = info
            .type_name
            .as_deref()
            .unwrap_or_else(|| self.type_arena.get(info.ty).name());
        self.sema
            .type_def_index
            .get(type_name)
            .map(|&idx| crate::types::dynamic_type_id(idx))
    }

    /// Build an Await node: EventSource declaration + Await node (spec 4.5; not ready -> frame suspends).
    ///
    /// Shared by await/recv: infer event-source kind -> register EventSourceDecl -> generate the Await node.
    fn build_await_node(
        &mut self,
        recv: crate::ast::Ast::ExprId,
        recv_node: NodeId,
    ) -> NodeId {
        let es_inputs_offset = self.graph.inputs_pool.push(&[]);
        let es_node = self.graph.add_node(Node {
            kind: NodeKind::EventSource,
            input_count: 0,
            inputs_offset: es_inputs_offset,
            compute_fn: CF_NOOP, // noop
        });
        let event_kind = self.infer_event_source_kind(recv);
        let current_sg = self.current_function_sg;
        if let Some(sg_id) = current_sg {
            if let Some(sg) = self.graph.subgraphs.get_mut(sg_id.0 as usize) {
                sg.event_source_decls.push(EventSourceDecl {
                    node: es_node,
                    kind: event_kind,
                });
            }
        }
        // current_effect appended at the end as an implicit dependency (consistent with compile_call):
        // ensures await executes only after prior effects (e.g. producer.await()) complete,
        // otherwise result_ch.recv() would become ready before producer.await() and suspend on an empty channel, causing deadlock.
        let mut await_inputs = vec![recv_node];
        if let Some(eff) = self.current_effect {
            await_inputs.push(eff);
        }
        let await_inputs_offset = self.graph.inputs_pool.push(&await_inputs);
        let await_node = self.graph.add_node(Node {
            kind: NodeKind::Await,
            input_count: await_inputs.len() as u8,
            inputs_offset: await_inputs_offset,
            compute_fn: CF_AWAIT, // compute_await
        });
        self.graph.set_await_event_source(await_node, es_node);
        await_node
    }

    /// Infer the event-source kind from the recv expression.
    ///
    /// Async<T> -> AsyncJoin, Channel<T>/Receiver<T> -> Channel, Timer -> Timer
    /// default -> AsyncJoin (5a-2 primarily supports awaiting async handles)
    fn infer_event_source_kind(&self, recv: crate::ast::Ast::ExprId) -> EventSourceKind {
        // Look up the recv's type name in Sema expr_types
        let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), recv.0 as u64);
        if let Some(info) = self.sema.expr_types.get(&key) {
            if let Some(ref tn) = info.type_name {
                let tn = tn.as_ref();
                // Built-in generics + Timer: derived from Type::from_type_name + family() (eliminates string matching)
                if let Some(ty) = crate::types::Type::from_type_name(tn) {
                    use crate::types::TypeFamily;
                    match ty.family() {
                        TypeFamily::Async => return EventSourceKind::AsyncJoin,
                        TypeFamily::Channel | TypeFamily::Receiver => return EventSourceKind::Channel,
                        TypeFamily::Timer => return EventSourceKind::Timer,
                        _ => {}
                    }
                }
            }
        }
        EventSourceKind::AsyncJoin
    }

    /// Check if an expression's inferred type is `Async<T>` (Bug #79: auto-await forwarding).
    /// Returns false when the type is unknown or not Async, unlike `infer_event_source_kind`
    /// which defaults to AsyncJoin.
    fn expr_type_is_async(&self, expr_id: crate::ast::Ast::ExprId) -> bool {
        let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), expr_id.0 as u64);
        if let Some(info) = self.sema.expr_types.get(&key) {
            if let Some(ref tn) = info.type_name {
                if let Some(ty) = crate::types::Type::from_type_name(tn.as_ref()) {
                    return ty.family() == crate::types::TypeFamily::Async;
                }
            }
        }
        false
    }

    /// Compile a field access.
    ///
    /// Binds compute_record_field_get, storing only the field name as the runtime by-name lookup key.
    fn compile_field_access(
        &mut self,
        _expr_id: crate::ast::Ast::ExprId,
        recv: crate::ast::Ast::ExprId,
        field: &str,
    ) -> NodeId {
        // Qualified-name constructor: Type.Ctor (zero-parameter constructor)
        if let crate::ast::Ast::Expr::Ident(type_name) = &self.current_module().arena.expr(recv).node {
            if let Some((ctor_type_name, ctor_name, field_names, kind, is_nullary)) =
                self.check_qualified_ctor_ir(type_name, field)
            {
                if is_nullary {
                    let inputs_offset = self.graph.inputs_pool.push(&[]);
                    let node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: 0,
                        inputs_offset,
                        compute_fn: CF_RECORD_CONSTRUCT,
                    });
                    self.graph.set_record_lit_info(node, RecordLitInfo {
                        type_name: ctor_type_name,
                        field_names,
                        constructor: ctor_name,
                        kind,
                    });
                    return node;
                }
            }
        }

        // Cross-module constant access (Math.PI): sema has recorded the recv's expr key → mangled
        // name in module_const_recv_exprs. On a hit, skip recv compilation and look up the mangled
        // name in global_var_slots to emit compile_global_load, sharing the local global var path.
        let recv_key = crate::sema::Sema::module_expr_key(
            self.expr_key_module(),
            recv.0 as u64,
        );
        if let Some(mangled) = self.sema.module_const_recv_exprs.get(&recv_key) {
            if let Some(&slot) = self.global_var_slots.get(mangled.as_str()) {
                return self.compile_global_load(slot);
            }
        }
        let recv_node = self.compile_subexpr(recv);
        let inputs_offset = self.graph.inputs_pool.push(&[recv_node]);
        let node = self.graph.add_node(Node {
            kind: NodeKind::FieldAccess,
            input_count: 1,
            inputs_offset,
            compute_fn: CF_RECORD_FIELD_GET, // record_field_get
        });
        // Uniformly store the field name as the runtime lookup key:
        // Record/Adt both resolve via find_field(name) by name, no compile-time field_idx needed
        self.graph.set_field_set_name(node, field.to_string());
        node
    }

    /// Build a FieldAccess node for an implicit-this field read.
    ///
    /// When a bare identifier inside a method body resolves to an instance field (recorded by
    /// sema on `ExprInfo.implicit_this`), the IR synthesizes `this.<field>`. The receiver node is
    /// the `this` binding already compiled for the method body; this helper mirrors
    /// `compile_field_access` but skips recv re-compilation and qualified-ctor/global-const
    /// detection (neither applies to an implicit `this` receiver).
    fn build_implicit_field_access(&mut self, this_node: NodeId, field: &str) -> NodeId {
        // Chain with `current_effect` to mirror the explicit `this.<field>` path:
        // `compile_expr` for `Ident("this")` chains the bound `this` node through
        // `current_effect` (line 588), ensuring the field read executes only after
        // prior side effects (e.g. a prior `pos = pos + 1` WriteBack) complete.
        // Without this chain, an implicit field read could observe a stale value
        // before an in-flight assignment updates the instance field, breaking
        // iterator-style mutation (e.g. `next()` reading `pos` after `pos = pos + 1`).
        let this_node = self.chain_effects(self.current_effect, this_node);
        let inputs_offset = self.graph.inputs_pool.push(&[this_node]);
        let node = self.graph.add_node(Node {
            kind: NodeKind::FieldAccess,
            input_count: 1,
            inputs_offset,
            compute_fn: CF_RECORD_FIELD_GET, // record_field_get
        });
        self.graph.set_field_set_name(node, field.to_string());
        node
    }

    /// Compile an index access.
    fn compile_index(&mut self, recv: crate::ast::Ast::ExprId, index: crate::ast::Ast::ExprId) -> NodeId {
        let recv_node = self.compile_subexpr(recv);
        let index_node = self.compile_subexpr(index);
        let inputs_offset = self.graph.inputs_pool.push(&[recv_node, index_node]);
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 2,
            inputs_offset,
            compute_fn: CF_ARRAY_INDEX, // array_index
        })
    }

    /// Compile a slice `recv[start..end]` (inclusive=false) or `recv[start..=end]` (inclusive=true).
    ///
    /// Three-input node (recv, start, end); the inclusive flag is stored in graph.slice_inclusive.
    /// At runtime, str is sliced by code point and array by element.
    fn compile_slice(
        &mut self,
        recv: crate::ast::Ast::ExprId,
        start: crate::ast::Ast::ExprId,
        end: crate::ast::Ast::ExprId,
        inclusive: bool,
    ) -> NodeId {
        let recv_node = self.compile_subexpr(recv);
        let start_node = self.compile_subexpr(start);
        let end_node = self.compile_subexpr(end);
        let inputs_offset = self.graph.inputs_pool.push(&[recv_node, start_node, end_node]);
        let node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 3,
            inputs_offset,
            compute_fn: CF_SLICE, // compute_slice
        });
        self.graph.set_slice_inclusive(node, inclusive);
        node
    }

    /// Compile a record construction (by positional args + type name).
    ///
    /// Used for `Err(args)` / `IOError(args)` and similar constructor calls; field names are auto-generated as `_0`, `_1`, ...
    fn compile_record_like(&mut self, type_name: &str, args: &[crate::ast::Ast::ExprId]) -> NodeId {
        let mut inputs = Vec::with_capacity(args.len());
        for &arg in args {
            inputs.push(self.compile_subexpr(arg));
        }
        let field_names: Vec<Option<String>> = (0..args.len())
            .map(|i| Some(format!("_{}", i)))
            .collect();
        let inputs_offset = self.graph.inputs_pool.push(&inputs);
        let node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: inputs.len() as u8,
            inputs_offset,
            compute_fn: CF_RECORD_CONSTRUCT, // record_construct
        });
        self.graph.set_record_lit_info(
            node,
            RecordLitInfo {
                type_name: type_name.to_string(),
                field_names,
                constructor: type_name.to_string(),
                kind: RecordLitKind::Record,
            },
        );
        node
    }

    /// Compile a record construction expression.
    /// Allocation sites marked non-escaping by the analyzer use the stack-alloc compute_fn (288).
    fn compile_record_lit(&mut self, expr_id: crate::ast::Ast::ExprId, fields: &[crate::ast::Ast::RecordFieldExpr<'_>]) -> NodeId {
        let mut inputs = Vec::with_capacity(fields.len());
        let mut field_names = Vec::with_capacity(fields.len());
        for field in fields {
            inputs.push(self.compile_subexpr(field.value));
            field_names.push(Some(field.name.to_string()));
        }
        let inputs_offset = self.graph.inputs_pool.push(&inputs);
        // Stack-alloc marker: non-escaping allocations use compute_record_construct_stack (288)
        let compute_fn = if self.should_stack_alloc(expr_id) {
            CF_RECORD_CONSTRUCT_STACK // record_construct_stack
        } else {
            CF_RECORD_CONSTRUCT // record_construct
        };
        let node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: inputs.len() as u8,
            inputs_offset,
            compute_fn,
        });
        self.graph.set_record_lit_info(
            node,
            RecordLitInfo {
                type_name: "Record".to_string(),
                field_names,
                constructor: "Record".to_string(),
                kind: RecordLitKind::Record,
            },
        );
        node
    }

    /// Compile a record extension expression `(...base, field: value, ...)`.
    ///
    /// inputs[0] = base record; inputs[1..] = update field values.
    /// RecordExtendInfo stores the update field name list (in order, corresponding to inputs[1..]).
    /// At runtime, clones fields from base, replaces/appends by update field names, and builds a new RecordValue.
    fn compile_record_extend(
        &mut self,
        base: crate::ast::Ast::ExprId,
        updates: &[crate::ast::Ast::RecordFieldExpr<'_>],
    ) -> NodeId {
        let mut inputs = Vec::with_capacity(1 + updates.len());
        let mut update_names = Vec::with_capacity(updates.len());
        inputs.push(self.compile_subexpr(base));
        for field in updates {
            inputs.push(self.compile_subexpr(field.value));
            update_names.push(field.name.to_string());
        }
        let inputs_offset = self.graph.inputs_pool.push(&inputs);
        let node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: inputs.len() as u8,
            inputs_offset,
            compute_fn: CF_RECORD_EXTEND, // record_extend
        });
        self.graph.set_record_extend_info(node, RecordExtendInfo { update_names });
        node
    }

    /// Compile an atomic construction expression `atomic expr`.
    ///
    /// Single-input node; at runtime wraps the value as an AtomicValue (an atomic container sharing the underlying memory).
    fn compile_atomic(&mut self, operand: crate::ast::Ast::ExprId) -> NodeId {
        let operand_node = self.compile_subexpr(operand);
        let inputs_offset = self.graph.inputs_pool.push(&[operand_node]);
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 1,
            inputs_offset,
            compute_fn: CF_ATOMIC_CONSTRUCT, // atomic_construct
        })
    }

    /// Compile an array construction expression.
    /// Allocation sites marked non-escaping by the analyzer use the stack-alloc compute_fn (289).
    /// When `fill` is present (`[value, ..count]`), uses compute_array_fill (321).
    fn compile_array_lit(&mut self, expr_id: crate::ast::Ast::ExprId, elements: &[crate::ast::Ast::ExprRef], fill: Option<(crate::ast::Ast::ExprRef, crate::ast::Ast::ExprRef)>) -> NodeId {
        if let Some((value, count)) = fill {
            let val_node = self.compile_subexpr(value);
            let count_node = self.compile_subexpr(count);
            let inputs = [val_node, count_node];
            let inputs_offset = self.graph.inputs_pool.push(&inputs);
            return self.graph.add_node(Node {
                kind: NodeKind::BinOp,
                input_count: 2,
                inputs_offset,
                compute_fn: CF_ARRAY_FILL,
            });
        }
        let mut inputs = Vec::with_capacity(elements.len());
        for &elem in elements {
            inputs.push(self.compile_subexpr(elem));
        }
        let inputs_offset = self.graph.inputs_pool.push(&inputs);
        // Stack-alloc marker: non-escaping allocations use compute_array_construct_stack (289)
        let compute_fn = if self.should_stack_alloc(expr_id) {
            CF_ARRAY_CONSTRUCT_STACK // array_construct_stack
        } else {
            CF_ARRAY_CONSTRUCT // array_construct
        };
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: inputs.len() as u8,
            inputs_offset,
            compute_fn,
        })
    }

    /// Compile a Block expression.
    ///
    /// Compiles stmts in order; the trailing expression's NodeId is the Block's result.
    fn compile_block(
        &mut self,
        stmts: &[crate::ast::Ast::StmtId],
        trailing: &Option<crate::ast::Ast::ExprId>,
    ) -> NodeId {
        self.enter_scope();
        let prev_effect = self.current_effect;
        // Bug #66: If this is the function body's top-level block, do NOT extract defers —
        // they must stay in defer_table for function-exit execution. Only nested blocks
        // extract block-scoped defers. The flag is set by compile_function_body and reset
        // here so that nested blocks within the function body see `false`.
        let is_function_top_block = self.in_function_top_block;
        self.in_function_top_block = false;
        // Initialize last_effect to prev_effect so the block's first statement depends on prior effects
        // (e.g. the store nodes of global var initialization in the entry function), ensuring that
        // load/call inside the block run only after prior side effects complete.
        let mut last_effect: Option<NodeId> = prev_effect;
        self.current_effect = None;
        // Bug #66: Record the defer_table length at block entry so we can extract
        // block-scoped defers and run them at block exit (LIFO).
        let defer_mark = self
            .current_function_sg
            .map(|sg| self.graph.subgraphs[sg.0 as usize].defer_table.len())
            .unwrap_or(0);
        for &stmt_id in stmts {
            // Set current_effect so subsequent effect nodes (e.g. WriteBack) depend on the prior effect
            self.current_effect = last_effect;
            // Statements are not in tail position (Return internally restores in_tail_position = true for its value)
            let prev_tail = self.in_tail_position;
            self.in_tail_position = false;
            let effect = self.compile_stmt(stmt_id);
            self.in_tail_position = prev_tail;
            if let Some(eff) = effect {
                // Control-flow nodes (CF_RETURN/CF_BREAK/CF_CONTINUE/CF_THROW_WRAP_ERR) have their
                // prior side-effect dependencies baked into inputs in compile_stmt; no signal relocation
                // is needed. chain_effects is only used for sequential linking of non-control-flow statements.
                let chained = self.chain_effects(last_effect, eff);
                last_effect = Some(chained);
            }
        }
        // The trailing expression inherits the block's effect chain on compilation,
        // ensuring Call nodes in the trailing expression depend on prior effects (consistent with stmts)
        self.current_effect = last_effect;
        let result = match trailing {
            Some(expr_id) => {
                let result_node = self.compile_expr(*expr_id);
                self.chain_effects(last_effect, result_node)
            }
            None => last_effect.unwrap_or_else(|| self.compile_void_const()),
        };
        // Bug #66: Block-scoped defer cleanup — extract defers registered inside this block
        // and generate LIFO cleanup Call nodes after the block result. This ensures defers
        // declared inside `{ ... }` execute when the block exits, not when the function exits.
        // The extracted defers are removed from the function-level defer_table to prevent
        // double execution at function exit.
        // Skip for function body top-level block: those defers must stay in defer_table for
        // function-exit execution (run_defers_sync / process_frame).
        let (result, _defer_effect) = if is_function_top_block {
            (result, None)
        } else {
            self.compile_block_defer_cleanup(defer_mark, result)
        };
        // defer cleanup effects are chained into `result` via CF_SEQ inside
        // compile_block_defer_cleanup, so they flow to consumers through the block's
        // return value. No separate last_effect update is needed (current_effect is
        // restored to prev_effect below).
        self.current_effect = prev_effect;
        self.exit_scope();
        result
    }

    /// Bug #66: Extract block-scoped defers registered after `defer_mark` and generate
    /// LIFO cleanup Call nodes. The defers are removed from the function-level defer_table.
    /// The cleanup nodes are chained after `result` via CF_SEQ, preserving the result value.
    /// Returns (result_node, cleanup_effect) where cleanup_effect is the last defer Call node
    /// (to be used as last_effect for subsequent statements).
    fn compile_block_defer_cleanup(
        &mut self,
        defer_mark: usize,
        result: NodeId,
    ) -> (NodeId, Option<NodeId>) {
        let cur_sg = match self.current_function_sg {
            Some(sg) => sg,
            None => return (result, None), // No function subgraph — nothing to do
        };
        let defer_table = &mut self.graph.subgraphs[cur_sg.0 as usize].defer_table;
        if defer_table.len() <= defer_mark {
            return (result, None); // No new defers in this block
        }
        // Extract block-scoped defers (drain entries after defer_mark).
        let block_defers: Vec<crate::ir::Ir::DeferEntry> =
            defer_table.drain(defer_mark..).collect();
        // Generate cleanup by reusing the loop-defer machinery (CF_DEFER_REGISTER + CF_DEFER_RUN).
        // Each block-scoped defer is registered onto the runtime defer_stack via a
        // CF_DEFER_REGISTER node (which snapshots the defer's captured values), then a single
        // CF_DEFER_RUN node drains the stack in LIFO order, executing each defer body as a proper
        // defer frame (with parent_frame_ptr/root_frame_ptr set so the body can read/write outer
        // variables via the frame chain). This mirrors how loops run defer-in-loop bodies and
        // fixes two issues:
        //   - The defer body must run as a defer frame (NOT a regular Call via make_call, which
        //     gives a node_offset=0 frame that cannot reach outer scope via the frame chain).
        //   - The block result value must be preserved: cleanup nodes are chained BEFORE `result`
        //     via CF_SEQ (which returns its LAST input's value), so the final node yields `result`.
        // Generate cleanup by reusing the loop-defer machinery (CF_BLOCK_DEFER_REGISTER +
        // CF_DEFER_RUN). Each block-scoped defer is registered onto the runtime defer_stack via a
        // CF_BLOCK_DEFER_REGISTER node (which snapshots the defer's captured values), then a single
        // CF_DEFER_RUN node drains the stack in LIFO order, executing each defer body as a proper
        // defer frame (with parent_frame_ptr/root_frame_ptr set so the body can read/write outer
        // variables via the frame chain). This mirrors how loops run defer-in-loop bodies.
        //
        // ORDERING (critical): in the dataflow scheduler every node is scheduled independently
        // based on its OWN inputs. A node with zero inputs is enqueued at frame start and would
        // fire before prior effects (e.g. global-var initialization) complete, causing the defer
        // body's reads of outer/global variables to observe stale/null values. To prevent this,
        // each register/run node takes the accumulated effect chain as a DIRECT input:
        //   - CF_BLOCK_DEFER_REGISTER treats input[0] as an effect-ordering dependency and uses
        //     inputs[1..] as the captured NodeIds.
        //   - CF_DEFER_RUN ignores all inputs (it reads defer_stack) but still requires them ready.
        // The block result value is preserved by wrapping the final run node + `result` in a
        // CF_SEQ (which returns its LAST input's value, i.e. `result`).
        let mut last_defer_call: Option<NodeId> = None;
        let mut effect_dep: NodeId = result;
        // Iterate in source (registration) order so the register nodes push onto defer_stack in
        // the same order; CF_DEFER_RUN then drains in LIFO (rev) order, running the
        // last-declared defer first — matching the function-level defer semantics.
        for entry in block_defers.iter() {
            // Build inputs: [effect_dep] ++ captured_inputs.
            let mut reg_inputs: Vec<NodeId> = Vec::with_capacity(entry.captured_inputs.len() + 1);
            reg_inputs.push(effect_dep);
            reg_inputs.extend_from_slice(&entry.captured_inputs);
            let inputs_off = self.graph.inputs_pool.push(&reg_inputs);
            let reg_node = self.graph.add_node(Node {
                kind: NodeKind::Call,
                input_count: reg_inputs.len() as u8,
                inputs_offset: inputs_off,
                compute_fn: CF_BLOCK_DEFER_REGISTER,
            });
            self.graph.set_call_target(reg_node, entry.body_subgraph);
            effect_dep = reg_node;
            last_defer_call = Some(reg_node);
        }
        // CF_DEFER_RUN node: drains defer_stack in LIFO order and runs each defer body as a defer
        // frame. Give it `effect_dep` as a direct input so it cannot fire before the register
        // nodes (and thus before the block's prior effects) complete.
        let run_off = self.graph.inputs_pool.push(&[effect_dep]);
        let run_node = self.graph.add_node(Node {
            kind: NodeKind::Call,
            input_count: 1,
            inputs_offset: run_off,
            compute_fn: CF_DEFER_RUN,
        });
        if last_defer_call.is_none() {
            last_defer_call = Some(run_node);
        }
        // Wrap run_node + result in CF_SEQ so the block's value is `result` (CF_SEQ returns its
        // last input's value). Both inputs must be ready before the SEQ computes, so the defer
        // cleanup side effects are guaranteed to complete before any consumer reads the value.
        let result_node = self.chain_effects(Some(run_node), result);
        (result_node, last_defer_call)
    }

    /// Compile a statement, returning an effect node (to be sequentially linked into the block result node).
    /// Returns None for pure declarations (variable bindings); their value node is automatically reachable via variable references.
    fn compile_stmt(&mut self, stmt_id: crate::ast::Ast::StmtId) -> Option<NodeId> {
        // Skip analyzer-flagged dead statements (unreachable code / dead declarations / dead stores); emit no IR nodes
        if self.is_dead_stmt(stmt_id) {
            return None;
        }
        let spanned = self.current_module().arena.stmt(stmt_id);
        let stmt = &spanned.node;
        match stmt {
            crate::ast::Ast::Stmt::ValDecl { name, value, .. } => {
                let value_node = self.compile_subexpr(*value);
                // Create an independent copy node for the val declaration (CF_SEQ single input = identity),
                // so the val binding owns an independent node ID rather than aliasing the source node.
                // This ensures that closures capturing the val variable capture the snapshot value at
                // declaration time, rather than the current value of the source variable (which may be a var).
                // For example: in a while loop, `val captured = i` followed by `fun() { captured }`;
                // without a copy node, captured aliases i's node and all closures read i's final value
                // after the loop ends. With a copy node, captured owns an independent node (within the
                // loop body subgraph scope); in the main frame that node is not ready, so the
                // same_function path falls back to the closure's Cell upvalue, returning the correct snapshot.
                let copy_off = self.graph.inputs_pool.push(&[value_node]);
                let copy_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 1,
                    inputs_offset: copy_off,
                    compute_fn: CF_SEQ,
                });
                self.bind_var(name, copy_node);
                Some(copy_node)
            }
            crate::ast::Ast::Stmt::VarDecl { name, value, .. } => {
                let value_node = self.compile_subexpr(*value);
                self.bind_var(name, value_node);
                Some(value_node)
            }
            crate::ast::Ast::Stmt::Expression { expr } => {
                let expr_node = self.compile_subexpr(*expr);
                Some(expr_node)
            }
            crate::ast::Ast::Stmt::Assignment { target, value } => {
                let raw_val = self.compile_subexpr(*value);
                // Link current_effect: ensures the assignment expression executes only after prior effects
                // (e.g. an if-Gate with continue) complete. Prevents statements after continue from running early.
                let val_node = self.chain_effects(self.current_effect, raw_val);
                let target_expr = &self.current_module().arena.expr(*target).node;
                // Array index assignment arr[i] = x: emit a CF_ARRAY_STORE node (three inputs: arr, index, value)
                if let crate::ast::Ast::Expr::Index { recv, index } = target_expr {
                    let arr_node = self.compile_subexpr(*recv);
                    let idx_node = self.compile_subexpr(*index);
                    let off = self.graph.inputs_pool.push(&[arr_node, idx_node, val_node]);
                    let store_node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: 3,
                        inputs_offset: off,
                        compute_fn: CF_ARRAY_STORE,
                    });
                    return Some(store_node);
                }
                if let crate::ast::Ast::Expr::Ident(name) = target_expr {
                    // Implicit-this field assignment: `field = value` inside a method body
                    // resolves to `this.field = value`.
                    if let Some(crate::sema::Sema::ImplicitThisAccess::Field(field)) = self.expr_implicit_this(*target).cloned() {
                        let this_node = self
                            .lookup_var("this")
                            .expect("this binding must exist in method body");
                        let off = self.graph.inputs_pool.push(&[this_node, val_node]);
                        let set_node = self.graph.add_node(Node {
                            kind: NodeKind::BinOp,
                            input_count: 2,
                            inputs_offset: off,
                            compute_fn: CF_RECORD_FIELD_SET,
                        });
                        self.graph.set_field_set_name(set_node, field.to_string());
                        return Some(set_node);
                    }
                    // Check whether this is a lambda-captured variable: captured_scopes records, per lambda
                    // layer, the captured variable names and their corresponding outer node. Assigning a
                    // captured variable requires a WriteBack to the outer node so the change is visible
                    // to the outer layer (by-reference capture semantics).
                    let captured_source = self.captured_scopes.iter().rev()
                        .find_map(|scope| scope.iter()
                            .find(|(n, _)| n.as_str() == *name)
                            .map(|(_, node)| *node));
                    if let Some(source) = captured_source {
                        let wb_node = self.compile_writeback_node(val_node, source);
                        self.bind_var(name, val_node);
                        return Some(wb_node);
                    } else if let Some(outer_node) = self.lookup_var(name) {
                        if !self.is_in_current_subgraph(outer_node) {
                            // Outer variable -> WriteBack. Use the root-frame declaration as
                            // WriteBack target, not the intermediate node returned by lookup_var
                            // (which may be in a same_function branch subgraph like a while body).
                            // This ensures WriteBack writes to the correct root-frame slot.
                            let wb_target = self.lookup_root_frame_var(name).unwrap_or(outer_node);
                            let wb_node = self.compile_writeback_node(val_node, wb_target);
                            self.bind_var(name, val_node);
                            return Some(wb_node);
                        } else if let Some(&captured_node) = self.captured_vars.get(*name) {
                            // Local variable captured by an inner lambda -> WriteBack
                            let wb_node = self.compile_writeback_node(val_node, captured_node);
                            self.bind_var(name, val_node);
                            return Some(wb_node);
                        } else if self.current_function_has_defer() {
                            // Bug #49: when a function contains defer, reassigning a local variable requires a
                            // WriteBack to the original node, so the defer body (which references the original
                            // node) reads the latest value rather than the compile-time snapshot.
                            let wb_node = self.compile_writeback_node(val_node, outer_node);
                            self.bind_var(name, val_node);
                            return Some(wb_node);
                        } else {
                            // Same_function branch subgraph (e.g. loop body): a prior assignment
                            // already bind_var'd the variable into the current subgraph, so
                            // lookup_var returns a local node. Use lookup_root_frame_var to find
                            // the outermost binding (root-frame declaration) for WriteBack, so
                            // the new value propagates back to the root frame.
                            if let Some(original_outer) = self.lookup_root_frame_var(name) {
                                if !self.is_in_current_subgraph(original_outer) {
                                    let wb_node = self.compile_writeback_node(val_node, original_outer);
                                    self.bind_var(name, val_node);
                                    return Some(wb_node);
                                }
                            }
                            self.bind_var(name, val_node);
                        }
                    } else if let Some(slot) = self.lookup_global_var(name) {
                        // Global variable -> global_store, returning an effect node to ensure scheduled execution
                        let store_node = self.compile_global_store(val_node, slot);
                        return Some(store_node);
                    } else {
                        self.bind_var(name, val_node);
                    }
                }
                None
            }
            crate::ast::Ast::Stmt::FieldAssignment { object, field, value } => {
                let obj_node = self.compile_subexpr(*object);
                let val_node = self.compile_subexpr(*value);
                let inputs_offset = self.graph.inputs_pool.push(&[obj_node, val_node]);
                let set_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset,
                    compute_fn: CF_RECORD_FIELD_SET, // record_field_set
                });
                self.graph.set_field_set_name(set_node, field.to_string());
                Some(set_node)
            }
            crate::ast::Ast::Stmt::CompoundAssignment { target, op, value } => {
                let target_expr = &self.current_module().arena.expr(*target).node;
                if let crate::ast::Ast::Expr::Ident(name) = target_expr {
                    let val_node = self.compile_subexpr(*value);
                    let bin_compute = self.compound_assign_op_to_compute_fn(*op, *target);
                    // Implicit-this field compound assignment: `field op= value` inside a
                    // method body resolves to `this.field op= value`.
                    if let Some(crate::sema::Sema::ImplicitThisAccess::Field(field)) = self.expr_implicit_this(*target).cloned() {
                        let this_node = self
                            .lookup_var("this")
                            .expect("this binding must exist in method body");
                        // Read the current field value.
                        let get_off = self.graph.inputs_pool.push(&[this_node]);
                        let get_node = self.graph.add_node(Node {
                            kind: NodeKind::FieldAccess,
                            input_count: 1,
                            inputs_offset: get_off,
                            compute_fn: CF_RECORD_FIELD_GET,
                        });
                        self.graph.set_field_set_name(get_node, field.to_string());
                        // Operation.
                        let bin_off = self.graph.inputs_pool.push(&[get_node, val_node]);
                        let raw_result = self.graph.add_node(Node {
                            kind: NodeKind::BinOp,
                            input_count: 2,
                            inputs_offset: bin_off,
                            compute_fn: bin_compute,
                        });
                        let result_node = self.chain_effects(self.current_effect, raw_result);
                        // Write back.
                        let set_off = self.graph.inputs_pool.push(&[this_node, result_node]);
                        let set_node = self.graph.add_node(Node {
                            kind: NodeKind::BinOp,
                            input_count: 2,
                            inputs_offset: set_off,
                            compute_fn: CF_RECORD_FIELD_SET,
                        });
                        self.graph.set_field_set_name(set_node, field.to_string());
                        return Some(set_node);
                    }
                    // Check whether this is a lambda-captured variable
                    let captured_source = self.captured_scopes.iter().rev()
                        .find_map(|scope| scope.iter()
                            .find(|(n, _)| n.as_str() == *name)
                            .map(|(_, node)| *node));
                    // Read current value: local var > global var > placeholder
                    let cur_node = if let Some(n) = self.lookup_var(name) {
                        n
                    } else if let Some(slot) = self.lookup_global_var(name) {
                        self.compile_global_load(slot)
                    } else {
                        self.compile_placeholder()
                    };
                    let off = self.graph.inputs_pool.push(&[cur_node, val_node]);
                    let raw_result = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: 2,
                        inputs_offset: off,
                        compute_fn: bin_compute,
                    });
                    // Link current_effect: prevents a compound assignment after continue from running early
                    let result_node = self.chain_effects(self.current_effect, raw_result);
                    if captured_source.is_some() {
                        self.compile_writeback_node(result_node, captured_source.unwrap());
                        self.bind_var(name, result_node);
                        None
                    } else if self.lookup_global_var(name).is_some() && self.lookup_var(name).is_none() {
                        // Global variable -> global_store. Return the store node so it is chained
                        // into the block's effect chain (last_effect), otherwise the store would be
                        // orphaned and dropped (the global has no local binding to keep it alive).
                        let slot = self.lookup_global_var(name).unwrap();
                        let store_node = self.compile_global_store(result_node, slot);
                        self.current_effect = Some(store_node);
                        Some(store_node)
                    } else if !self.is_in_current_subgraph(cur_node) {
                        // Outer variable -> WriteBack + bind a local reference
                        self.compile_writeback_node(result_node, cur_node);
                        self.bind_var(name, result_node);
                        None
                    } else if let Some(&captured_node) = self.captured_vars.get(*name) {
                        self.compile_writeback_node(result_node, captured_node);
                        self.bind_var(name, result_node);
                        None
                    } else {
                        self.bind_var(name, result_node);
                        None
                    }
                } else {
                    // Non-Ident target (FieldAccess/Index/Deref): delegate to
                    // compile_compound_assign which handles read-modify-write for these.
                    let set_node = self.compile_compound_assign(*op, *target, *value);
                    self.current_effect = Some(set_node);
                    Some(set_node)
                }
            }
            crate::ast::Ast::Stmt::Return { value } => {
                let prev_effect = self.current_effect;
                let return_val_node = match value {
                    Some(expr_id) => {
                        let prev_tail = self.in_tail_position;
                        self.in_tail_position = true;
                        let r = self.compile_expr(*expr_id);
                        self.in_tail_position = prev_tail;
                        r
                    }
                    None => self.compile_void_const(),
                };
                // CF_RETURN: inputs[0] = return value, inputs[1] = prior side-effect dependency (optional)
                // The prior side-effect dependency ensures the return signal fires only after prior statements complete
                let (off, count) = match prev_effect {
                    Some(eff) => (self.graph.inputs_pool.push(&[return_val_node, eff]), 2),
                    None => (self.graph.inputs_pool.push(&[return_val_node]), 1),
                };
                let return_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: count,
                    inputs_offset: off,
                    compute_fn: CF_RETURN,
                });
                Some(return_node)
            }
            crate::ast::Ast::Stmt::Throw { expr } => {
                let prev_effect = self.current_effect;
                let expr_node = self.compile_subexpr(*expr);
                // CF_THROW_WRAP_ERR: inputs[0] = thrown value, inputs[1] = prior side-effect dependency (optional)
                // compute_throw_wrap_err directly returns NodeResult::Return(ThrowVal(Err(v)))
                let (off, count) = match prev_effect {
                    Some(eff) => (self.graph.inputs_pool.push(&[expr_node, eff]), 2),
                    None => (self.graph.inputs_pool.push(&[expr_node]), 1),
                };
                let wrap_node = self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: count,
                    inputs_offset: off,
                    compute_fn: CF_THROW_WRAP_ERR,
                });
                Some(wrap_node)
            }
            crate::ast::Ast::Stmt::Break => {
                // CF_BREAK: optional inputs[0] = prior side-effect dependency
                let (off, count) = match self.current_effect {
                    Some(eff) => (self.graph.inputs_pool.push(&[eff]), 1),
                    None => (self.graph.inputs_pool.push(&[]), 0),
                };
                let n = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: count,
                    inputs_offset: off,
                    compute_fn: CF_BREAK,
                });
                Some(n)
            }
            crate::ast::Ast::Stmt::Continue => {
                // CF_CONTINUE: optional inputs[0] = prior side-effect dependency
                // The engine-side complete_and_wake_caller detects Continue -> reset_loop_iteration for the next round
                // (Sema guarantees continue is always inside a loop)
                let (off, count) = match self.current_effect {
                    Some(eff) => (self.graph.inputs_pool.push(&[eff]), 1),
                    None => (self.graph.inputs_pool.push(&[]), 0),
                };
                let n = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: count,
                    inputs_offset: off,
                    compute_fn: CF_CONTINUE,
                });
                Some(n)
            }
            crate::ast::Ast::Stmt::While { condition, body } => {
                let while_sg = self.register_while_subgraph(*condition, *body);
                let call_node = self.compile_recursive_call(while_sg);
                Some(call_node)
            }
            crate::ast::Ast::Stmt::Loop { body } => {
                let loop_sg = self.register_loop_subgraph(*body);
                let call_node = self.compile_recursive_call(loop_sg);
                Some(call_node)
            }
            crate::ast::Ast::Stmt::For {
                name,
                iterable,
                body,
            } => {
                // For loop = iterable (already an iterator) -> recursive subgraph (next() + is_null + body)
                let iterable_node = self.compile_subexpr(*iterable);
                // Obtain iterable type info from Sema (type name + whether it is a trait object)
                let (iter_type_name, is_trait_object) = self.lookup_expr_iter_info(*iterable);
                // Register the For loop subgraph (static dispatch: bind next() by type name; trait objects go through vtable)
                let for_sg = self.register_for_subgraph(
                    name,
                    *body,
                    iter_type_name.as_deref(),
                    is_trait_object,
                );
                // Start the loop: Call(for_sg, [iterable_node])
                let call_node = self.make_call(for_sg, &[iterable_node]);
                Some(call_node)
            }
            crate::ast::Ast::Stmt::Defer { expr } => {
                if self.in_loop_body {
                    // Defer-in-loop: compile as CF_DEFER_REGISTER node.
                    // The defer body subgraph + captured values are pushed onto
                    // the loop frame's defer_stack at runtime; CF_DEFER_RUN (in void_sg) drains
                    // it in LIFO order at loop exit.
                    let (body_sg, _captured_inputs) = self.compile_branch_subgraph(*expr);
                    // Unified capture model: snapshot the loop variable (if any)
                    // and any Snapshot-mode captures, so each defer body reads
                    // per-iteration values rather than final values.
                    // Reference-mode captures (var bindings like an accumulator)
                    // are NOT snapshotted here — they are read live via the
                    // frame chain at defer-run time, so successive loop
                    // iterations' defers accumulate correctly (LIFO over the
                    // shared latest value).
                    let loop_var = self.loop_stack.last().and_then(|lc| lc.loop_var_node);
                    let sema_captures = self.lookup_captures(*expr);
                    let mut inputs: Vec<NodeId> = Vec::new();
                    if let Some(n) = loop_var {
                        inputs.push(n);
                    }
                    for cap in sema_captures {
                        // Only Snapshot-mode captures need per-iteration
                        // snapshotting; Reference-mode captures are read live.
                        if cap.mode != crate::sema::Sema::CaptureMode::Snapshot {
                            continue;
                        }
                        if let Some(node) = self.lookup_var(cap.name.as_ref()) {
                            if !inputs.contains(&node) {
                                inputs.push(node);
                            }
                        }
                    }
                    let inputs_off = self.graph.inputs_pool.push(&inputs);
                    let reg_node = self.graph.add_node(Node {
                        kind: NodeKind::Call,
                        input_count: inputs.len() as u8,
                        inputs_offset: inputs_off,
                        compute_fn: CF_DEFER_REGISTER,
                    });
                    self.graph.set_call_target(reg_node, body_sg);
                    Some(reg_node)
                } else {
                    // defer expr -> compile expr as an independent subgraph and register it in the
                    // current function subgraph's defer_table.
                    let (body_sg, _branch_captures) = self.compile_branch_subgraph(*expr);
                    // Unified capture model: resolve the defer's capture list from
                    // Sema (all entries are Reference mode for defer — defer
                    // semantics read the value at function/block exit). Each
                    // captured variable's current NodeId is resolved via
                    // `lookup_var` and stored in `DeferEntry.captured_inputs`.
                    // At runtime, the defer frame injects these snapshot values
                    // into its value table, mirroring the loop-defer path.
                    let sema_captures = self.lookup_captures(*expr);
                    let mut captured_inputs: Vec<NodeId> = Vec::new();
                    for cap in sema_captures {
                        if let Some(node) = self.lookup_var(cap.name.as_ref()) {
                            captured_inputs.push(node);
                        }
                    }
                    let trigger = self.compile_void_const();
                    if let Some(cur_sg) = self.current_function_sg {
                        let entry = DeferEntry {
                            trigger_node: trigger,
                            body_subgraph: body_sg,
                            captured_inputs,
                            registered: false,
                        };
                        self.graph.subgraphs[cur_sg.0 as usize]
                            .defer_table
                            .push(entry);
                    }
                    None
                }
            }
            crate::ast::Ast::Stmt::LocalDecl { decl } => {
                match decl.as_ref() {
                    crate::ast::Ast::Decl::FunDecl {
                        name, params, body, is_async, extern_c_body, ..
                    } => {
                        if extern_c_body.is_some() {
                            return None;
                        }
                        let construct_node =
                            self.compile_lambda(params, *body, *is_async, Some(name), None);
                        self.bind_var(name, construct_node);
                        Some(construct_node)
                    }
                    crate::ast::Ast::Decl::TypeDecl { name, def, .. } => {
                        // Register nested type fields into the current scope (unified with top-level types via type_scope_stack lookup)
                        match def {
                            crate::ast::Ast::TypeDef::Record { fields } => {
                                let field_names: Vec<String> = fields.iter().map(|f| f.name.to_string()).collect();
                                self.bind_type_fields(name, TypeFieldInfo {
                                    field_names,
                                    type_name: name.to_string(),
                                    kind: RecordLitKind::Record,
                                });
                            }
                            crate::ast::Ast::TypeDef::Adt { constructors } => {
                                // Register the type name + each constructor name (mapped to the type name)
                                self.bind_type_fields(name, TypeFieldInfo {
                                    field_names: Vec::new(),
                                    type_name: name.to_string(),
                                    kind: RecordLitKind::Adt,
                                });
                                for ctor in constructors {
                                    let field_names: Vec<String> = ctor.fields.iter()
                                        .map(|f| f.name.unwrap_or("_").to_string())
                                        .collect();
                                    self.bind_type_fields(ctor.name, TypeFieldInfo {
                                        field_names,
                                        type_name: name.to_string(),
                                        kind: RecordLitKind::Adt,
                                    });
                                }
                            }
                            crate::ast::Ast::TypeDef::Newtype { name: nt_name, .. } => {
                                self.bind_type_fields(nt_name, TypeFieldInfo {
                                    field_names: Vec::new(),
                                    type_name: nt_name.to_string(),
                                    kind: RecordLitKind::Newtype,
                                });
                            }
                            crate::ast::Ast::TypeDef::Alias { .. } => {}
                        }
                        None
                    }
                    // Trait declaration: Sema registers the type; the IR layer generates no code
                    crate::ast::Ast::Decl::TraitDecl { .. } => None,
                    _ => None,
                }
            }
        }
    }

    /// Look up a function's location across the user module and builtin modules.
    /// Returns None = user module, Some(i) = builtin_modules[i].
    fn find_function_location(&self, name: &str) -> Option<Option<usize>> {
        if self.module.find_function(name).is_some() {
            return Some(None);
        }
        for (i, builtin_mod) in self.builtin_modules.iter().enumerate() {
            if builtin_mod.find_function(name).is_some() {
                return Some(Some(i));
            }
        }
        None
    }

    /// Compile a function into a subgraph (supports cross-module: user module + builtin modules).
    ///
    /// If the function does not exist or its declaration type mismatches, records a compile error and returns a placeholder subgraph (error recovery).
    pub fn compile_function(&mut self, name: &str) -> SubGraphId {
        let location = match self.find_function_location(name) {
            Some(loc) => loc,
            None => {
                self.errors.push(format!("function {} not found", name));
                return self.register_subgraph_placeholder(name, 0, false);
            }
        };

        let module = match location {
            None => self.module,
            Some(i) => self.builtin_modules[i],
        };

        // Set the current compiling module (compile_expr accesses the AST arena via current_module())
        let prev_builtin = self.compiling_builtin;
        self.compiling_builtin = match location {
            None => None,
            Some(i) => Some(self.builtin_modules[i]),
        };

        let (body_expr, is_async, params, is_entry, return_type) = match module.find_function(name) {
            Some(d) => match &d.node {
                crate::ast::Ast::Decl::FunDecl {
                    body,
                    is_async,
                    params,
                    is_entry,
                    return_type,
                    ..
                } => (*body, *is_async, params.clone(), *is_entry, *return_type),
                _ => {
                    self.errors.push(format!("{} is not a function", name));
                    self.compiling_builtin = prev_builtin;
                    return self.register_subgraph_placeholder(name, 0, false);
                }
            },
            None => {
                self.errors.push(format!("function {} not found", name));
                self.compiling_builtin = prev_builtin;
                return self.register_subgraph_placeholder(name, 0, false);
            }
        };
        let param_count = params.len();

        // Reuse the pre-registered sg_id (created by the build() pre-registration pass) to avoid duplicate subgraphs
        let sg_id = if let Some(&existing) = self.func_subgraphs.get(name) {
            existing
        } else {
            let new_id = self.register_subgraph_placeholder(name, param_count as u8, is_async);
            self.func_subgraphs.insert(name.to_string(), new_id);
            new_id
        };
        let node_start = self.graph.nodes.len() as u32;

        self.current_function_sg = Some(sg_id);
        self.current_function_id = sg_id.0;
        let prev_effect = self.current_effect;
        self.current_effect = None;
        // Set current_sg_start = node_start so that sub-functions like compile_memoize can correctly
        // reference parameter nodes (param node id = node_start + param_index)
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
        self.enter_scope();
        // Record the parameter scope depth so defer body compilation can truncate
        // body-local rebindings and resolve external vars to parameter nodes.
        let prev_param_scope_depth = self.param_scope_depth;
        self.param_scope_depth = self.scope_stack.len();

        // Create parameter nodes (Const placeholders; values are injected at runtime by start_subgraph)
        // These nodes must be the first param_count nodes of the subgraph
        for param in &params {
            let inputs_offset = self.graph.inputs_pool.push(&[]);
            let param_node = self.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            });
            self.bind_var(param.name, param_node);
        }

        // Entry function: compile all modules' top-level var/val declaration initializations before the function body
        // Global variables are written to the shared storage via global_store; all functions read via global_load
        // Switch compiling_builtin per module so compile_subexpr accesses the correct AST arena
        if is_entry && !self.top_level_var_decls.is_empty() {
            let decls: Vec<(Option<usize>, crate::ast::Ast::StmtId)> = std::mem::take(&mut self.top_level_var_decls);
            for (mod_idx, stmt_id) in &decls {
                let prev_builtin = self.compiling_builtin;
                self.compiling_builtin = match mod_idx {
                    None => None,
                    Some(i) => Some(self.builtin_modules[*i]),
                };
                let module = self.current_module();
                let stmt = &module.arena.stmt(*stmt_id).node;
                let (name, value_expr) = match stmt {
                    crate::ast::Ast::Stmt::VarDecl { name, value, .. } => (*name, *value),
                    crate::ast::Ast::Stmt::ValDecl { name, value, .. } => (*name, *value),
                    _ => { self.compiling_builtin = prev_builtin; continue; }
                };
                let init_node = self.compile_subexpr(value_expr);
                let slot = self.global_var_slots.get(name).copied()
                    .expect("global var slot must exist after collection");
                let store_node = self.compile_global_store(init_node, slot);
                self.current_effect = Some(store_node);
                self.compiling_builtin = prev_builtin;
            }
            self.top_level_var_decls = decls;
        }

        // Tail-call optimization is enabled only for non-void functions: the trailing expression of a
        // void function is a side effect (e.g. println("done")) and should not be tail-called
        // (switch_subgraph would lose the current frame state).
        // Consumes sema's FuncSigInfo.return_type to determine void (builtin modules fall back to AST).
        let is_void_fn = self.sema.get_func_sig(name)
            .map(|sig| matches!(self.type_arena.get(sig.return_type), crate::sema::Sema::Type::Void))
            .unwrap_or_else(|| match return_type {
                None => true,
                Some(tr) => {
                    matches!(module.arena.ty(tr).node, crate::ast::Ast::TypeNode::Named { name } if crate::value::ValueTag::from_name(name).is_some_and(|t| t.family() == crate::types::TypeFamily::Void))
                }
            });
        // Compute is_async before compile_function_body so it can auto-await Async<T> bodies (Bug #79).
        let fn_is_async = self.sema.get_func_sig(name)
            .map(|sig| sig.is_async)
            .unwrap_or(is_async);
        let return_node = self.compile_function_body(name, None, body_expr, &params, is_void_fn, fn_is_async);
        self.exit_scope();
        self.current_effect = prev_effect;
        self.current_sg_start = prev_sg_start;
        self.param_scope_depth = prev_param_scope_depth;
        self.current_function_sg = None;
        self.compiling_builtin = prev_builtin;

        let node_end = self.graph.nodes.len() as u32;
        let debug_mod_name = if std::env::var("KUZO_DEBUG_BUILD").is_ok() {
            Some(self.current_module().name.to_string())
        } else {
            None
        };
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = return_node;
        if let Some(ref mod_name) = debug_mod_name {
            eprintln!("[COMPILE-FN] name={:?} sg_id={} nodes=[{},{}) param_count={} mod={:?}",
                name, sg_id.0, node_start, node_end, param_count, mod_name);
        }
        // Consumes sema's FuncSigInfo.is_async (builtin modules fall back to AST is_async)
        sg.has_suspend = fn_is_async;
        sg.function_id = sg_id.0;

        self.func_subgraphs.insert(name.to_string(), sg_id);
        sg_id
    }

    /// Compile a monomorphization instance into a specialized subgraph.
    ///
    /// Differences from `compile_function`:
    /// - Registers under a mangled name (`func_name#instance_id`) in func_subgraphs to avoid clashing with the non-generic version
    /// - Sets `current_type_args` (type param name -> TypeHandle) for cast/expr_type_name queries
    /// - Clears `current_type_args` after compilation
    ///
    /// Called only for generic instances with type_args (non-generic instances are handled by compile_function).
    fn compile_monomorph_instance(&mut self, instance: &crate::sema::Sema::MonomorphInstance) {
        let func_name = instance.func_name.as_ref();

        // Look up the function declaration location (user module or builtin module)
        let location = match self.find_function_location(func_name) {
            Some(loc) => loc,
            None => {
                self.errors.push(format!("monomorph instance function {} not found", func_name));
                return;
            }
        };

        let module = match location {
            None => self.module,
            Some(i) => self.builtin_modules[i],
        };

        let prev_builtin = self.compiling_builtin;
        self.compiling_builtin = match location {
            None => None,
            Some(i) => Some(self.builtin_modules[i]),
        };

        let (body_expr, is_async, params, return_type) = match module.find_function(func_name) {
            Some(d) => match &d.node {
                crate::ast::Ast::Decl::FunDecl {
                    body, is_async, params, return_type, ..
                } => (*body, *is_async, params.clone(), *return_type),
                _ => {
                    self.errors.push(format!("{} is not a function", func_name));
                    self.compiling_builtin = prev_builtin;
                    return;
                }
            },
            None => {
                self.errors.push(format!("monomorph instance function {} not found", func_name));
                self.compiling_builtin = prev_builtin;
                return;
            }
        };
        let param_count = params.len();

        // Build the type parameter map: type_params name -> type_args TypeHandle
        // type_params come from FuncSigInfo (in the same order as instance.type_args)
        let type_param_names: Vec<String> = self.sema.get_func_sig(func_name)
            .map(|sig| sig.type_params.iter().map(|n| n.to_string()).collect())
            .unwrap_or_default();
        let prev_type_args = std::mem::take(&mut self.current_type_args);
        self.current_type_args = type_param_names.iter().zip(instance.type_args.iter())
            .map(|(name, &h)| (name.clone(), h))
            .collect();
        let prev_instance_id = self.current_instance_id;
        self.current_instance_id = Some(instance.instance_id);

        // Mangled name: func_name#instance_id (consistent with sema's cache_key format func_name#hash)
        let mangled = format!("{}#{}", func_name, instance.instance_id);

        // Pre-register the subgraph (reusing the placeholder mechanism)
        let sg_id = if let Some(&existing) = self.func_subgraphs.get(mangled.as_str()) {
            existing
        } else {
            let new_id = self.register_subgraph_placeholder(&mangled, param_count as u8, is_async);
            self.func_subgraphs.insert(mangled.clone(), new_id);
            new_id
        };
        let node_start = self.graph.nodes.len() as u32;

        self.current_function_sg = Some(sg_id);
        self.current_function_id = sg_id.0;
        let prev_effect = self.current_effect;
        self.current_effect = None;
        // Set current_sg_start = node_start so compile_memoize in compile_function_body
        // can correctly reference parameter nodes (id = node_start + param_index)
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
        self.enter_scope();
        let prev_param_scope_depth = self.param_scope_depth;
        self.param_scope_depth = self.scope_stack.len();

        // Create parameter nodes (Const placeholders; values are injected at runtime by start_subgraph)
        for param in &params {
            let inputs_offset = self.graph.inputs_pool.push(&[]);
            let param_node = self.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            });
            self.bind_var(param.name, param_node);
        }

        // Compile the function body (unified entry: memoize/tail_rec/non_tail_rec apply to generic instances too)
        let is_void_fn = self.sema.get_func_sig(func_name)
            .map(|sig| matches!(self.type_arena.get(sig.return_type), crate::sema::Sema::Type::Void))
            .unwrap_or_else(|| match return_type {
                None => true,
                Some(tr) => {
                    matches!(module.arena.ty(tr).node, crate::ast::Ast::TypeNode::Named { name } if crate::value::ValueTag::from_name(name).is_some_and(|t| t.family() == crate::types::TypeFamily::Void))
                }
            });
        // Compute is_async before compile_function_body so it can auto-await Async<T> bodies (Bug #79).
        let fn_is_async = self.sema.get_func_sig(func_name)
            .map(|sig| sig.is_async)
            .unwrap_or(is_async);
        let return_node = self.compile_function_body(func_name, None, body_expr, &params, is_void_fn, fn_is_async);
        self.exit_scope();
        self.current_effect = prev_effect;
        self.current_sg_start = prev_sg_start;
        self.param_scope_depth = prev_param_scope_depth;
        self.current_function_sg = None;
        self.compiling_builtin = prev_builtin;

        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = return_node;
        // Consumes sema's FuncSigInfo.is_async (builtin modules fall back to AST is_async)
        sg.has_suspend = fn_is_async;
        sg.function_id = sg_id.0;

        // Restore the outer type_args context
        self.current_type_args = prev_type_args;
        self.current_instance_id = prev_instance_id;
    }

    /// Compile a TypeDecl method in a builtin module (looked up in method_subgraphs via (type_id, method_idx)).
    fn compile_builtin_method(&mut self, type_name: &str, method_idx: usize) {
        // Look up the method data in builtin modules (indexed directly by method_idx)
        let found = self.builtin_modules.iter().enumerate().find_map(|(mod_i, m)| {
            for d in &m.declarations {
                if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = &d.node {
                    if *name == type_name {
                        if let Some(method) = methods.get(method_idx) {
                            if method.body.is_some() {
                                return Some((
                                    mod_i,
                                    method.name,
                                    method.body.unwrap(),
                                    method.is_async,
                                    method.params.clone(),
                                    method.return_type,
                                ));
                            }
                        }
                    }
                }
            }
            None
        });

        let (mod_i, method_name, body_expr, is_async, params, return_type) = match found {
            Some(x) => x,
            None => return,
        };

        let m = self.builtin_modules[mod_i];

        // Obtain the pre-registered sg_id from method_subgraphs (created in build() step 0a)
        let type_id = match self.sema.type_def_index.get(type_name) {
            Some(&idx) => crate::types::dynamic_type_id(idx),
            None => return,
        };
        let sg_id = match self.method_subgraphs.get(&(type_id, method_idx as u16)) {
            Some(&sg) => sg,
            None => return,
        };

        let prev = self.compiling_builtin;
        self.compiling_builtin = Some(m);
        let node_start = self.graph.nodes.len() as u32;

        self.current_function_sg = Some(sg_id);
        self.current_function_id = sg_id.0;
        let prev_effect = self.current_effect;
        self.current_effect = None;
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
        let prev_method_type = self.current_method_type.take();
        self.current_method_type = Some((type_name.into(), type_id));
        self.enter_scope();
        let prev_param_scope_depth = self.param_scope_depth;
        self.param_scope_depth = self.scope_stack.len();

        for param in &params {
            let inputs_offset = self.graph.inputs_pool.push(&[]);
            let param_node = self.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            });
            self.bind_var(param.name, param_node);
        }

        // Unified entry: memoize/tail_rec/non_tail_rec apply to builtin methods too
        // (self_type = Some(type_name) builds the mangled name "Type.method" to look up FuncId)
        let is_void_fn = match return_type {
            None => true,
            Some(tr) => {
                matches!(m.arena.ty(tr).node, crate::ast::Ast::TypeNode::Named { name } if crate::value::ValueTag::from_name(name).is_some_and(|t| t.family() == crate::types::TypeFamily::Void))
            }
        };
        let return_node = self.compile_function_body(method_name, Some(type_name), body_expr, &params, is_void_fn, is_async);
        self.exit_scope();
        self.current_effect = prev_effect;
        self.current_sg_start = prev_sg_start;
        self.param_scope_depth = prev_param_scope_depth;
        self.current_function_sg = None;
        self.current_method_type = prev_method_type;
        self.compiling_builtin = prev;

        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = return_node;
        sg.has_suspend = is_async;
        sg.function_id = sg_id.0;
    }

    /// Compile a TypeDecl method in the user module (looked up in method_subgraphs via (type_id, method_idx)).
    fn compile_user_method(&mut self, type_name: &str, method_idx: usize) {
        // Look up the method data in the user module (indexed directly by method_idx).
        // Search top-level declarations first, then local types declared inside function bodies.
        let found = self.module.declarations.iter().find_map(|d| {
            if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = &d.node {
                if *name == type_name {
                    if let Some(method) = methods.get(method_idx) {
                        if method.body.is_some() {
                            return Some((
                                method.name,
                                method.body.unwrap(),
                                method.is_async,
                                method.params.clone(),
                                method.return_type,
                            ));
                        }
                    }
                }
            }
            None
        });
        // Fallback: search local types (declared inside function bodies via Stmt::LocalDecl)
        let found = match found {
            Some(x) => Some(x),
            None => self.find_type_method_full(type_name, method_idx),
        };

        let (method_name, body_expr, is_async, params, return_type) = match found {
            Some(x) => x,
            None => return,
        };

        // Obtain the pre-registered sg_id from method_subgraphs (created in build() step 0a)
        let type_id = match self.sema.type_def_index.get(type_name) {
            Some(&idx) => crate::types::dynamic_type_id(idx),
            None => return,
        };
        let sg_id = match self.method_subgraphs.get(&(type_id, method_idx as u16)) {
            Some(&sg) => sg,
            None => return,
        };

        let node_start = self.graph.nodes.len() as u32;

        self.current_function_sg = Some(sg_id);
        self.current_function_id = sg_id.0;
        let prev_effect = self.current_effect;
        self.current_effect = None;
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
        let prev_method_type = self.current_method_type.take();
        self.current_method_type = Some((type_name.into(), type_id));
        self.enter_scope();
        self.param_scope_depth = self.scope_stack.len();

        for param in &params {
            let inputs_offset = self.graph.inputs_pool.push(&[]);
            let param_node = self.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            });
            self.bind_var(param.name, param_node);
        }

        // Unified entry: memoize/tail_rec/non_tail_rec apply to user methods too
        // (self_type = Some(type_name) builds the mangled name "Type.method" to look up FuncId)
        let is_void_fn = match return_type {
            None => true,
            Some(tr) => {
                matches!(self.module.arena.ty(tr).node, crate::ast::Ast::TypeNode::Named { name } if crate::value::ValueTag::from_name(name).is_some_and(|t| t.family() == crate::types::TypeFamily::Void))
            }
        };
        let return_node = self.compile_function_body(method_name, Some(type_name), body_expr, &params, is_void_fn, is_async);
        self.exit_scope();
        self.current_effect = prev_effect;
        self.current_sg_start = prev_sg_start;
        self.current_function_sg = None;
        self.current_method_type = prev_method_type;

        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = return_node;
        sg.has_suspend = is_async;
        sg.function_id = sg_id.0;
    }

    /// Compile the monomorphized specialization of a trait default method (generates a dedicated subgraph for a given impl type).
    ///
    /// A trait default method is the dispatch target when a type does not override it. For each type
    /// implementing the trait, a specialized subgraph is generated so that self in the body has
    /// concrete type information, allowing self.method() calls to statically bind to the correct
    /// method subgraph via path 2 (the type's own methods).
    fn compile_trait_default_method(&mut self, trait_name: &str, method_idx: usize, impl_type_name: &str, instance_idx: usize) {
        // Look up the TraitDecl method with a body in the user module (indexed directly by method_idx)
        let found = self.module.declarations.iter().find_map(|d| {
            if let crate::ast::Ast::Decl::TraitDecl { name, methods, .. } = &d.node {
                if *name == trait_name {
                    if let Some(method) = methods.get(method_idx) {
                        if method.body.is_some() {
                            return Some((
                                method.body.unwrap(),
                                method.is_async,
                                method.params.clone(),
                            ));
                        }
                    }
                }
            }
            None
        });

        let (body_expr, is_async, params) = match found {
            Some(x) => x,
            None => return,
        };

        // Obtain the pre-registered specialized subgraph sg_id from trait_default_subgraphs
        let trait_idx = match self.sema.trait_def_index.get(trait_name) {
            Some(&idx) => idx,
            None => return,
        };
        let type_id = match self.sema.type_def_index.get(impl_type_name) {
            Some(&idx) => crate::types::dynamic_type_id(idx),
            None => return,
        };
        let sg_id = match self.trait_default_subgraphs.get(&(type_id, trait_idx, method_idx as u16)) {
            Some(&sg) => sg,
            None => return,
        };

        let node_start = self.graph.nodes.len() as u32;

        self.current_function_sg = Some(sg_id);
        self.current_function_id = sg_id.0;
        // Record the current specialization instance index; expr_type_name/expr_type_id use it to look
        // up sema's TraitDefaultInstance.type_name for self's concrete type (consumes sema output).
        self.current_trait_default_idx = Some(instance_idx);
        let prev_effect = self.current_effect;
        self.current_effect = None;
        let prev_method_type = self.current_method_type.take();
        self.current_method_type = Some((impl_type_name.into(), type_id));
        self.enter_scope();

        for param in &params {
            let inputs_offset = self.graph.inputs_pool.push(&[]);
            let param_node = self.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            });
            self.bind_var(param.name, param_node);
        }

        let return_node = self.compile_expr(body_expr);
        self.exit_scope();
        self.current_effect = prev_effect;
        self.current_function_sg = None;
        self.current_trait_default_idx = None;
        self.current_method_type = prev_method_type;

        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = return_node;
        sg.has_suspend = is_async;
        sg.function_id = sg_id.0;
    }

    /// Find a TypeDecl method by `(type_name, method_idx)`, searching both top-level
    /// declarations AND local types declared inside function bodies. Returns
    /// `(method_name, params_count, is_async)`.
    ///
    /// This is used by the IR build to pre-register and compile local type methods
    /// (step 0a-local / step 2b-local), complementing `compile_user_method` which only
    /// searches `self.module.declarations`.
    fn find_type_method(
        &self,
        type_name: &str,
        method_idx: usize,
    ) -> Option<(&'static str, u8, bool)> {
        let module = self.module;
        let arena = &module.arena;
        // 1. Top-level declarations
        for decl in &module.declarations {
            if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = &decl.node {
                if *name == type_name {
                    if let Some(method) = methods.get(method_idx) {
                        if method.body.is_some() {
                            return Some((
                            // SAFETY: method.name is &'a str tied to module lifetime; leak to 'static
                            // (acceptable: the module outlives the entire build).
                            Box::leak(method.name.to_string().into_boxed_str()),
                            method.params.len() as u8,
                            method.is_async,
                            ));
                        }
                    }
                }
            }
        }
        // 2. Recurse into function bodies
        let mut found: Option<(&'static str, u8, bool)> = None;
        for decl in &module.declarations {
            match &decl.node {
                crate::ast::Ast::Decl::FunDecl { body, .. } => {
                    if self.find_type_method_in_expr(*body, arena, type_name, method_idx, &mut found) {
                        return found;
                    }
                }
                crate::ast::Ast::Decl::TypeDecl { methods, .. }
                | crate::ast::Ast::Decl::TraitDecl { methods, .. } => {
                    for m in methods.iter() {
                        if let Some(body) = m.body {
                            if self.find_type_method_in_expr(body, arena, type_name, method_idx, &mut found) {
                                return found;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        found
    }

    fn find_type_method_in_expr(
        &self,
        expr_id: crate::ast::Ast::ExprId,
        arena: &crate::ast::Ast::AstArena<'_>,
        type_name: &str,
        method_idx: usize,
        found: &mut Option<(&'static str, u8, bool)>,
    ) -> bool {
        let expr = &arena.expr(expr_id).node;
        match expr {
            crate::ast::Ast::Expr::Block { stmts, trailing } => {
                for s in stmts {
                    if self.find_type_method_in_stmt(*s, arena, type_name, method_idx, found) {
                        return true;
                    }
                }
                if let Some(t) = trailing {
                    return self.find_type_method_in_expr(*t, arena, type_name, method_idx, found);
                }
                false
            }
            _ => false,
        }
    }

    fn find_type_method_in_stmt(
        &self,
        stmt_id: crate::ast::Ast::StmtId,
        arena: &crate::ast::Ast::AstArena<'_>,
        type_name: &str,
        method_idx: usize,
        found: &mut Option<(&'static str, u8, bool)>,
    ) -> bool {
        let stmt = &arena.stmt(stmt_id).node;
        if let crate::ast::Ast::Stmt::LocalDecl { decl } = stmt {
            if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = decl.as_ref() {
                if *name == type_name {
                    if let Some(method) = methods.get(method_idx) {
                        if method.body.is_some() {
                            *found = Some((
                                Box::leak(method.name.to_string().into_boxed_str()),
                                method.params.len() as u8,
                                method.is_async,
                            ));
                            return true;
                        }
                    }
                }
                // Recurse into the type's method bodies
                for m in methods.iter() {
                    if let Some(body) = m.body {
                        if self.find_type_method_in_expr(body, arena, type_name, method_idx, found) {
                            return true;
                        }
                    }
                }
            }
            if let crate::ast::Ast::Decl::FunDecl { body, .. } = decl.as_ref() {
                return self.find_type_method_in_expr(*body, arena, type_name, method_idx, found);
            }
        }
        false
    }

    /// Full version of find_type_method that returns the complete method data needed by
    /// `compile_user_method`: `(method_name, body_expr, is_async, params, return_type)`.
    /// Searches top-level then local types. Used as the fallback in compile_user_method.
    fn find_type_method_full(
        &self,
        type_name: &str,
        method_idx: usize,
    ) -> Option<(&'static str, crate::ast::Ast::ExprId, bool, Vec<crate::ast::Ast::Param<'static>>, Option<crate::ast::Ast::TypeId>)> {
        let module = self.module;
        let arena = &module.arena;
        // Collect from a local helper that returns full method data
        let mut result: Option<(&'static str, crate::ast::Ast::ExprId, bool, Vec<crate::ast::Ast::Param<'static>>, Option<crate::ast::Ast::TypeId>)> = None;
        for decl in &module.declarations {
            match &decl.node {
                crate::ast::Ast::Decl::FunDecl { body, .. } => {
                    self.find_type_method_full_in_expr(*body, arena, type_name, method_idx, &mut result);
                }
                crate::ast::Ast::Decl::TypeDecl { methods, .. }
                | crate::ast::Ast::Decl::TraitDecl { methods, .. } => {
                    for m in methods.iter() {
                        if let Some(body) = m.body {
                            self.find_type_method_full_in_expr(body, arena, type_name, method_idx, &mut result);
                        }
                    }
                }
                _ => {}
            }
            if result.is_some() { return result; }
        }
        result
    }

    fn find_type_method_full_in_expr(
        &self,
        expr_id: crate::ast::Ast::ExprId,
        arena: &crate::ast::Ast::AstArena<'_>,
        type_name: &str,
        method_idx: usize,
        found: &mut Option<(&'static str, crate::ast::Ast::ExprId, bool, Vec<crate::ast::Ast::Param<'static>>, Option<crate::ast::Ast::TypeId>)>,
    ) {
        let expr = &arena.expr(expr_id).node;
        if let crate::ast::Ast::Expr::Block { stmts, .. } = expr {
            for s in stmts {
                let stmt = &arena.stmt(*s).node;
                if let crate::ast::Ast::Stmt::LocalDecl { decl } = stmt {
                    if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = decl.as_ref() {
                        if *name == type_name {
                            if let Some(method) = methods.get(method_idx) {
                                if method.body.is_some() {
                                    // SAFETY: leak method fields to 'static (module outlives build)
                                    let m_name: &'static str = Box::leak(method.name.to_string().into_boxed_str());
                                    let m_params: Vec<crate::ast::Ast::Param<'static>> = method.params.iter()
                                        .map(|p| crate::ast::Ast::Param {
                                            name: Box::leak(p.name.to_string().into_boxed_str()),
                                            type_annotation: p.type_annotation,
                                        })
                                        .collect();
                                    *found = Some((m_name, method.body.unwrap(), method.is_async, m_params, method.return_type));
                                    return;
                                }
                            }
                        }
                        // Recurse into method bodies
                        for m in methods.iter() {
                            if let Some(body) = m.body {
                                self.find_type_method_full_in_expr(body, arena, type_name, method_idx, found);
                                if found.is_some() { return; }
                            }
                        }
                    }
                    if let crate::ast::Ast::Decl::FunDecl { body, .. } = decl.as_ref() {
                        self.find_type_method_full_in_expr(*body, arena, type_name, method_idx, found);
                        if found.is_some() { return; }
                    }
                }
            }
        }
    }

    /// Recursively collect all local `TypeDecl`s declared inside function bodies
    /// (`Stmt::LocalDecl(TypeDecl)`) across the user module.
    ///
    /// Local types are registered by Sema into `type_def_index` (so `type_id` is available),
    /// and their methods are checked, but the IR build's step 0a / step 2b only scanned
    /// top-level `m.declarations`. This collector walks into Block expressions, match arms,
    /// if branches, lambda bodies, loops, etc. to surface nested type declarations so their
    /// method subgraphs get pre-registered and compiled.
    ///
    /// Returns `Vec<(type_name, method_idx)>` pairs mirroring the step 2b format.
    fn collect_local_type_methods(&self) -> Vec<(String, usize)> {
        let mut result = Vec::new();
        let module = self.module;
        let arena = &module.arena;
        // Scan top-level declarations for function/method bodies that may contain local types.
        for decl in &module.declarations {
            match &decl.node {
                crate::ast::Ast::Decl::FunDecl { body, .. } => {
                    self.collect_local_types_from_expr(*body, arena, &mut result);
                }
                crate::ast::Ast::Decl::TypeDecl { methods, .. }
                | crate::ast::Ast::Decl::TraitDecl { methods, .. } => {
                    for m in methods.iter() {
                        if let Some(body) = m.body {
                            self.collect_local_types_from_expr(body, arena, &mut result);
                        }
                    }
                }
                crate::ast::Ast::Decl::ExprDecl { expr, stmt, .. } => {
                    self.collect_local_types_from_expr(*expr, arena, &mut result);
                    if let Some(s) = stmt {
                        self.collect_local_types_from_stmt(*s, arena, &mut result);
                    }
                }
                _ => {}
            }
        }
        result
    }

    fn collect_local_types_from_stmt(
        &self,
        stmt_id: crate::ast::Ast::StmtId,
        arena: &crate::ast::Ast::AstArena<'_>,
        out: &mut Vec<(String, usize)>,
    ) {
        let stmt = &arena.stmt(stmt_id).node;
        match stmt {
            crate::ast::Ast::Stmt::LocalDecl { decl } => {
                if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = decl.as_ref() {
                    for (idx, m) in methods.iter().enumerate() {
                        if m.body.is_some() {
                            out.push((name.to_string(), idx));
                        }
                    }
                    // Recurse into the type's own method bodies (nested local types)
                    for m in methods.iter() {
                        if let Some(body) = m.body {
                            self.collect_local_types_from_expr(body, arena, out);
                        }
                    }
                }
                // Also recurse into local FunDecl bodies (functions can nest types)
                if let crate::ast::Ast::Decl::FunDecl { body, .. } = decl.as_ref() {
                    self.collect_local_types_from_expr(*body, arena, out);
                }
            }
            crate::ast::Ast::Stmt::ValDecl { value, .. }
            | crate::ast::Ast::Stmt::VarDecl { value, .. }
            | crate::ast::Ast::Stmt::Expression { expr: value, .. }
            | crate::ast::Ast::Stmt::Return { value: Some(value), .. }
            | crate::ast::Ast::Stmt::Throw { expr: value, .. }
            | crate::ast::Ast::Stmt::Defer { expr: value, .. } => {
                self.collect_local_types_from_expr(*value, arena, out);
            }
            crate::ast::Ast::Stmt::Assignment { target, value, .. }
            | crate::ast::Ast::Stmt::CompoundAssignment { target, value, .. } => {
                self.collect_local_types_from_expr(*target, arena, out);
                self.collect_local_types_from_expr(*value, arena, out);
            }
            crate::ast::Ast::Stmt::FieldAssignment { object, value, .. } => {
                self.collect_local_types_from_expr(*object, arena, out);
                self.collect_local_types_from_expr(*value, arena, out);
            }
            crate::ast::Ast::Stmt::For { body, .. }
            | crate::ast::Ast::Stmt::While { body, .. }
            | crate::ast::Ast::Stmt::Loop { body } => {
                self.collect_local_types_from_expr(*body, arena, out);
            }
            _ => {}
        }
    }

    fn collect_local_types_from_expr(
        &self,
        expr_id: crate::ast::Ast::ExprId,
        arena: &crate::ast::Ast::AstArena<'_>,
        out: &mut Vec<(String, usize)>,
    ) {
        let expr = &arena.expr(expr_id).node;
        match expr {
            crate::ast::Ast::Expr::Block { stmts, trailing } => {
                for s in stmts {
                    self.collect_local_types_from_stmt(*s, arena, out);
                }
                if let Some(t) = trailing {
                    self.collect_local_types_from_expr(*t, arena, out);
                }
            }
            crate::ast::Ast::Expr::If { cond, then_branch, else_branch } => {
                self.collect_local_types_from_expr(*cond, arena, out);
                self.collect_local_types_from_expr(*then_branch, arena, out);
                if let Some(e) = else_branch {
                    self.collect_local_types_from_expr(*e, arena, out);
                }
            }
            crate::ast::Ast::Expr::Match { scrutinee, arms } => {
                self.collect_local_types_from_expr(*scrutinee, arena, out);
                for arm in arms {
                    if let Some(g) = arm.guard {
                        self.collect_local_types_from_expr(g, arena, out);
                    }
                    self.collect_local_types_from_expr(arm.body, arena, out);
                }
            }
            crate::ast::Ast::Expr::Lambda { body, .. } => match body {
                crate::ast::Ast::LambdaBody::Block(e) | crate::ast::Ast::LambdaBody::Expression(e) => {
                    self.collect_local_types_from_expr(*e, arena, out);
                }
            },
            // Expressions that contain sub-expressions
            crate::ast::Ast::Expr::Call { callee, args, .. } => {
                self.collect_local_types_from_expr(*callee, arena, out);
                for a in args {
                    self.collect_local_types_from_expr(*a, arena, out);
                }
            }
            crate::ast::Ast::Expr::MethodCall { recv, args, .. }
            | crate::ast::Ast::Expr::SafeMethodCall { recv, args, .. } => {
                self.collect_local_types_from_expr(*recv, arena, out);
                for a in args {
                    self.collect_local_types_from_expr(*a, arena, out);
                }
            }
            crate::ast::Ast::Expr::Binary { lhs, rhs, .. }
            | crate::ast::Ast::Expr::Assign { target: lhs, value: rhs }
            | crate::ast::Ast::Expr::Elvis { lhs, rhs } => {
                self.collect_local_types_from_expr(*lhs, arena, out);
                self.collect_local_types_from_expr(*rhs, arena, out);
            }
            crate::ast::Ast::Expr::CompoundAssign { target, value, .. } => {
                self.collect_local_types_from_expr(*target, arena, out);
                self.collect_local_types_from_expr(*value, arena, out);
            }
            crate::ast::Ast::Expr::As { expr, .. } => {
                self.collect_local_types_from_expr(*expr, arena, out);
            }
            _ => {}
        }
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
                if let crate::ast::Ast::Decl::FunDecl { name, params, is_async, .. } = &d.node {
                    // Skip @extern("C") functions: they are only called via FFI and need no subgraph
                    if let crate::ast::Ast::Decl::FunDecl { extern_c_body, .. } = &d.node {
                        if extern_c_body.is_some() {
                            continue;
                        }
                    }
                    let sg_id = self.register_subgraph_placeholder(name, params.len() as u8, *is_async);
                    self.func_subgraphs.insert(name.to_string(), sg_id);
                    // Also register the mangled name (module_path.function_name) for selective import alias resolution
                    if let Some(ref mp) = module_path {
                        let mangled = format!("{}.{}", mp, name);
                        self.func_subgraphs.insert(mangled, sg_id);
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
                    let type_id = self.sema.type_def_index.get(*name).map(|&idx| crate::types::dynamic_type_id(idx));
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
                    let type_id = match self.sema.type_def_index.get(type_name.as_str()) {
                        Some(&idx) => crate::types::dynamic_type_id(idx),
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

        // 0a-trait. Pre-register trait default method monomorphization subgraphs:
        //   (type_id, trait_def_idx, method_idx) -> SubGraphId
        //   Consumes trait_default_instances collected in the Sema post-phase; registers a dedicated subgraph for each specialization instance.
        //   Instance collection (including skipping explicit overrides) is already done by Monomorph::collect_trait_default_instances.
        for inst in &self.sema.trait_default_instances {
            // Look up the AST info for the trait default method (method_name, params_count, is_async)
            let method_info = self.module.declarations.iter().find_map(|d| {
                if let crate::ast::Ast::Decl::TraitDecl { name, methods, .. } = &d.node {
                    if *name == inst.trait_name.as_ref() {
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
                    if !self.global_var_slots.contains_key(name) {
                        let slot = self.global_var_slots.len() as u32;
                        self.global_var_slots.insert(name.to_string(), slot);
                        self.top_level_var_decls.push((None, *stmt_id));
                        // Register the mangled name (module_path.name) pointing to the same slot
                        if let Some(ref mp) = crate::sema::Sema::module_logical_path(self.module.name) {
                            let mangled = format!("{}.{}", mp, name);
                            self.global_var_slots.insert(mangled, slot);
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
                        if !self.global_var_slots.contains_key(name) {
                            let slot = self.global_var_slots.len() as u32;
                            self.global_var_slots.insert(name.to_string(), slot);
                            self.top_level_var_decls.push((Some(i), *stmt_id));
                            // Register the mangled name (module_path.name) pointing to the same slot
                            if let Some(ref mp) = crate::sema::Sema::module_logical_path(m.name) {
                                let mangled = format!("{}.{}", mp, name);
                                self.global_var_slots.insert(mangled, slot);
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
        self.type_scope_stack.push(rustc_hash::FxHashMap::default());
        for m in &all_modules {
            for d in &m.declarations {
                if let crate::ast::Ast::Decl::TypeDecl { name, def, .. } = &d.node {
                    match def {
                        crate::ast::Ast::TypeDef::Record { fields } => {
                            let field_names: Vec<String> = fields.iter().map(|f| f.name.to_string()).collect();
                            self.bind_type_fields(name, TypeFieldInfo {
                                field_names,
                                type_name: name.to_string(),
                                kind: RecordLitKind::Record,
                            });
                        }
                        crate::ast::Ast::TypeDef::Adt { constructors } => {
                            // Register the type name (nullary path used for type-name lookup; field_names is empty only when there are no field constructors)
                            self.bind_type_fields(name, TypeFieldInfo {
                                field_names: Vec::new(),
                                type_name: name.to_string(),
                                kind: RecordLitKind::Adt,
                            });
                            // Register each constructor name (mapped to the type name)
                            for ctor in constructors {
                                let field_names: Vec<String> = ctor.fields.iter()
                                    .map(|f| f.name.unwrap_or("_").to_string())
                                    .collect();
                                self.bind_type_fields(ctor.name, TypeFieldInfo {
                                    field_names,
                                    type_name: name.to_string(),
                                    kind: RecordLitKind::Adt,
                                });
                            }
                        }
                        crate::ast::Ast::TypeDef::Newtype { name: nt_name, .. } => {
                            // Newtype: constructor name == type name, kind=Newtype
                            self.bind_type_fields(nt_name, TypeFieldInfo {
                                field_names: Vec::new(),
                                type_name: nt_name.to_string(),
                                kind: RecordLitKind::Newtype,
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
                let param_count = self.sema.get_func_sig(&inst.func_name)
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
        for (name, _mod_idx) in &builtin_fun_names {
            self.compile_function(name);
        }

        // 1b. Compile TypeDecl methods in builtin modules (indexed by method_idx)
        let builtin_methods: Vec<(String, usize)> = self
            .builtin_modules
            .iter()
            .flat_map(|m| {
                m.declarations.iter().flat_map(|d| {
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
            })
            .collect();
        for (type_name, method_idx) in &builtin_methods {
            self.compile_builtin_method(type_name, *method_idx);
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

        // 3. Compile user module functions
        for name in &fun_names {
            self.compile_function(name);
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
        if std::env::var("KUZO_DEBUG_BUILD").is_ok() {
            eprintln!("=== [BUILD] func_subgraphs with EMPTY node_range (uncompiled) ===");
            for (name, &sg_id) in &self.func_subgraphs {
                let sg = &self.graph.subgraphs[sg_id.0 as usize];
                let (s, e) = sg.node_range;
                if s.0 == 0 && e.0 == 0 {
                    eprintln!("  EMPTY: name={:?} sg_id={} param_count={}", name, sg_id.0, sg.param_count);
                }
            }
            eprintln!("=== [BUILD] func_subgraphs with NON-EMPTY node_range (compiled) ===");
            for (name, &sg_id) in &self.func_subgraphs {
                let sg = &self.graph.subgraphs[sg_id.0 as usize];
                let (s, e) = sg.node_range;
                if !(s.0 == 0 && e.0 == 0) {
                    eprintln!("  OK: name={:?} sg_id={} nodes=[{},{}) param_count={}", name, sg_id.0, s.0, e.0, sg.param_count);
                }
            }
        }

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

        // Move the build-time string_pool into graph.string_pool (ConstValue::Str references this pool)
        let pool = std::mem::take(&mut self.string_pool);
        self.graph.string_pool = Arc::from(pool);

        self.graph
    }
}

/// Detect the type suffix of a float literal, returning (stripped, suffix).
fn detect_float_suffix(s: &str) -> (&str, Option<&str>) {
    for suffix in &["f128", "f64", "f32", "f16"] {
        if s.ends_with(suffix) {
            return (&s[..s.len() - suffix.len()], Some(suffix));
        }
    }
    (s, None)
}

// =========================================================================
// Integer literal parsing + type range checking
// =========================================================================

/// Parse the raw text of an integer literal into i128, supporting 0x/0o/0b prefixes and underscore separators.
/// Returns an error with span info on parse failure (invalid syntax).
fn parse_int_to_i128(raw: &str, span: crate::ast::Ast::Span) -> Result<i128, String> {
    let cleaned: String = raw.chars().filter(|c| *c != '_').collect();
    let (digits, radix) = cleaned
        .strip_prefix("0x").map(|s| (s, 16u32))
        .or_else(|| cleaned.strip_prefix("0o").map(|s| (s, 8)))
        .or_else(|| cleaned.strip_prefix("0b").map(|s| (s, 2)))
        .unwrap_or((cleaned.as_str(), 10));
    i128::from_str_radix(digits, radix).map_err(|_| {
        format!("invalid integer literal '{}' at line {}:{}", raw, span.line, span.column)
    })
}

/// Parse the raw text of an integer literal into u128, supporting 0x/0o/0b prefixes and underscore separators.
/// u128 has unsigned semantics (no leading minus), used for u128-suffix literals to cover the full 0..=2^128-1 range.
/// Returns an error with span info on parse failure (invalid syntax or a leading minus).
fn parse_int_to_u128(raw: &str, span: crate::ast::Ast::Span) -> Result<u128, String> {
    let cleaned: String = raw.chars().filter(|c| *c != '_').collect();
    let (digits, radix) = cleaned
        .strip_prefix("0x").map(|s| (s, 16u32))
        .or_else(|| cleaned.strip_prefix("0o").map(|s| (s, 8)))
        .or_else(|| cleaned.strip_prefix("0b").map(|s| (s, 2)))
        .unwrap_or((cleaned.as_str(), 10));
    u128::from_str_radix(digits, radix).map_err(|_| {
        format!("invalid integer literal '{}' at line {}:{}", raw, span.line, span.column)
    })
}

/// Range-check an i128 value against the target type and convert it to a ConstValue.
/// Returns an error with the type name, valid range, and span info when out of range.
fn check_int_range(v: i128, ty_name: &str, raw: &str, span: crate::ast::Ast::Span) -> Result<ConstValue, String> {
    macro_rules! try_int {
        ($ty:ty, $variant:ident) => {
            match <$ty>::try_from(v) {
                Ok(val) => return Ok(ConstValue::$variant(val)),
                Err(_) => return Err(format!(
                    "integer literal '{}' at line {}:{} is out of range for {} (valid range: {}..={})",
                    raw, span.line, span.column, ty_name, <$ty>::MIN, <$ty>::MAX
                )),
            }
        };
    }
    // Single source of truth: derived via ValueTag::from_name, eliminating string special-casing
    let tag = crate::value::ValueTag::from_name(ty_name).unwrap_or(crate::value::ValueTag::I32);
    match tag {
        crate::value::ValueTag::I8 => try_int!(i8, I8),
        crate::value::ValueTag::I16 => try_int!(i16, I16),
        crate::value::ValueTag::I32 => try_int!(i32, I32),
        crate::value::ValueTag::I64 => try_int!(i64, I64),
        crate::value::ValueTag::I128 => Ok(ConstValue::I128(v)),
        crate::value::ValueTag::U8 => try_int!(u8, U8),
        crate::value::ValueTag::U16 => try_int!(u16, U16),
        crate::value::ValueTag::U32 => try_int!(u32, U32),
        crate::value::ValueTag::U64 => try_int!(u64, U64),
        crate::value::ValueTag::U128 => try_int!(u128, U128),
        crate::value::ValueTag::Isize => try_int!(isize, Isize),
        crate::value::ValueTag::Usize => try_int!(usize, Usize),
        _ => try_int!(i32, I32),
    }
}

// =========================================================================
// Hexadecimal float literal parsing (exact IEEE 754 bit patterns)
// =========================================================================
// Format: 0x<integer part>.<fractional part>p<exponent part>
//   0x1.921fb54442d18p+1 = 1.* 16^... * 2^(+1) = PI (f64)
// Supports positive/negative exponents, optional sign, and upper/lower-case 0x/P.

/// Parse a hexadecimal float literal into an f64 bit pattern, returning f64.
fn parse_hex_float_f64(s: &str) -> Option<f64> {
    let bits = parse_hex_float_to_u128(s, 11, 52, 1023)?;
    Some(f64::from_bits(bits as u64))
}

/// Parse a hexadecimal float literal into an f32 bit pattern, returning f32.
fn parse_hex_float_f32(s: &str) -> Option<f32> {
    let bits = parse_hex_float_to_u128(s, 8, 23, 127)?;
    Some(f32::from_bits(bits as u32))
}

/// Parse a hexadecimal float literal into an f16 bit pattern, returning u16 bits.
fn parse_hex_float_f16(s: &str) -> Option<u16> {
    let bits = parse_hex_float_to_u128(s, 5, 10, 15)?;
    Some(bits as u16)
}

/// Parse a hexadecimal float literal into an f128 bit pattern, returning [u8; 16].
fn parse_hex_float_f128(s: &str) -> Option<[u8; 16]> {
    let bits = parse_hex_float_to_u128(s, 15, 112, 16383)?;
    Some(bits.to_le_bytes())
}

/// Generic hexadecimal float parser.
/// Params: (literal, exponent bit width, mantissa bit width, exponent bias)
/// Returns: a u128 bit pattern (the caller truncates to the target width)
fn parse_hex_float_to_u128(s: &str, exp_bits: u32, mant_bits: u32, exp_bias: i64) -> Option<u128> {
    // Strip the 0x/0X prefix
    let body = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;

    // Split the mantissa part and the exponent part (p or P)
    let p_pos = body.find(|c| c == 'p' || c == 'P')?;
    let mantissa_str = &body[..p_pos];
    let exp_str = &body[p_pos + 1..];

    // Parse the mantissa: may contain a '.'
    let (int_part, frac_part) = match mantissa_str.find('.') {
        Some(dot) => (&mantissa_str[..dot], &mantissa_str[dot + 1..]),
        None => (mantissa_str, ""),
    };

    // Convert the hex mantissa to a numeric value (ignore the decimal point position; collect all hex digits first)
    let mut mantissa: u128 = 0;
    let mut frac_hex_digits: i32 = 0; // number of hex digits after the decimal point

    // Integer part
    for c in int_part.chars() {
        let d = c.to_digit(16)?;
        mantissa = mantissa.checked_mul(16)?.checked_add(d as u128)?;
    }

    // Fractional part
    for c in frac_part.chars() {
        let d = c.to_digit(16)?;
        mantissa = mantissa.checked_mul(16)?.checked_add(d as u128)?;
        frac_hex_digits += 1;
    }

    if mantissa == 0 {
        // Zero: may carry a sign, but the current implementation does not parse a sign prefix (the lexer already handles the minus)
        return Some(0);
    }

    // Parse the binary exponent (the part after p)
    let exp2: i64 = exp_str.parse().ok()?;

    // Actual exponent = exp2 - frac_hex_digits * 4 (because each hex digit = 4 bits)
    let binary_exp = exp2 - (frac_hex_digits as i64) * 4;

    // Normalize the mantissa: find the most significant bit, compute the unbiased exp
    // MSB position of mantissa (0-indexed from LSB)
    let msb = 127 - mantissa.leading_zeros() as i64;

    // We want to normalize the mantissa into the 1.xxx form:
    // The current mantissa represents an integer with its binary point at the end.
    // After normalization: mantissa = 1.fraction * 2^(msb + binary_exp)
    // But the MSB of mantissa is the implicit 1, so unbiased_exp = msb + binary_exp
    let unbiased_exp = msb + binary_exp;

    // Extract the fraction bits (the bits after removing the MSB)
    let fraction_mant = mantissa & ((1u128 << msb) - 1);
    let frac_bits_available = msb as u32;

    // Round the fraction to mant_bits (round-to-nearest-even)
    // Returns (fraction_field, exp_adjust)
    let (fraction, exp_adjust): (u128, i64) = if frac_bits_available > mant_bits {
        let shift = frac_bits_available - mant_bits;
        let kept = fraction_mant >> shift;
        let remainder = fraction_mant & ((1u128 << shift) - 1);
        let halfway = 1u128 << (shift - 1);
        let mut rounded = kept;
        if remainder > halfway {
            rounded += 1;
        } else if remainder == halfway {
            if kept & 1 != 0 {
                rounded += 1;
            }
        }
        if rounded >> mant_bits != 0 {
            (0, 1)
        } else {
            (rounded, 0)
        }
    } else if frac_bits_available < mant_bits {
        (fraction_mant << (mant_bits - frac_bits_available), 0)
    } else {
        (fraction_mant, 0)
    };

    let biased_exp = unbiased_exp + exp_adjust + exp_bias;
    let max_biased = (1i64 << exp_bits) - 1;

    if biased_exp >= max_biased {
        return Some((max_biased as u128) << mant_bits);
    }

    if biased_exp > 0 {
        let exp_field = (biased_exp as u128) << mant_bits;
        let frac_field = fraction & ((1u128 << mant_bits) - 1);
        return Some(exp_field | frac_field);
    }

    // biased_exp <= 0: subnormal or zero
    let shift = (1 - biased_exp) as u32;
    if shift >= 128 {
        return Some(0);
    }
    let full_mant = (1u128 << mant_bits) | (fraction & ((1u128 << mant_bits) - 1));
    let sub_fraction = (full_mant >> shift) & ((1u128 << mant_bits) - 1);
    if sub_fraction == 0 {
        return Some(0);
    }
    Some(sub_fraction)
}

// =========================================================================
// Decimal float literal -> IEEE 754 binary128 exact parsing (no f64 intermediary)
// =========================================================================
// Algorithm: decimal digits * 10^e10 -> big integer M * 2^e2 -> normalize 113-bit mantissa
//            + round-to-nearest-even rounding -> binary128 bit pattern.
// Big integers are represented as Vec<u64> little-endian; only multiply/divide by small
// integers and left/right shifts are needed, avoiding big-integer / big-integer division
// (10^k = 2^k * 5^k, so multiply/divide by 5 step by step suffices).

/// Decimal digit string -> Vec<u64> big integer (little-endian limbs).
fn bigint_from_dec(s: &str) -> Vec<u64> {
    let mut limbs = vec![0u64];
    for c in s.chars() {
        let d = (c as u8 - b'0') as u64;
        let mut carry = d;
        for l in limbs.iter_mut() {
            let prod = (*l as u128) * 10 + carry as u128;
            *l = prod as u64;
            carry = (prod >> 64) as u64;
        }
        if carry != 0 {
            limbs.push(carry);
        }
    }
    limbs
}

/// Multiply a big integer by a small integer (in place).
fn bigint_mul_small(limbs: &mut Vec<u64>, m: u64) {
    let mut carry = 0u128;
    for l in limbs.iter_mut() {
        let prod = (*l as u128) * (m as u128) + carry;
        *l = prod as u64;
        carry = prod >> 64;
    }
    if carry != 0 {
        limbs.push(carry as u64);
    }
}

/// Divide a big integer by a small integer (in place), returning the remainder.
fn bigint_divmod_small(limbs: &mut Vec<u64>, d: u64) -> u64 {
    let mut rem = 0u128;
    for l in limbs.iter_mut().rev() {
        let cur = (rem << 64) | (*l as u128);
        *l = (cur / d as u128) as u64;
        rem = cur % d as u128;
    }
    while limbs.len() > 1 && *limbs.last().unwrap() == 0 {
        limbs.pop();
    }
    rem as u64
}

/// Left-shift a big integer by n bits (in place).
fn bigint_shl(limbs: &mut Vec<u64>, n: u32) {
    let word_shift = (n / 64) as usize;
    let bit_shift = n % 64;
    if bit_shift > 0 {
        let mut carry = 0u64;
        for l in limbs.iter_mut() {
            let new = (*l << bit_shift) | carry;
            carry = *l >> (64 - bit_shift);
            *l = new;
        }
        if carry != 0 {
            limbs.push(carry);
        }
    }
    if word_shift > 0 {
        limbs.splice(0..0, std::iter::repeat(0u64).take(word_shift));
    }
}

/// Big integer bit length (most significant bit position + 1).
fn bigint_bit_len(limbs: &[u64]) -> u32 {
    let mut i = limbs.len();
    while i > 0 && limbs[i - 1] == 0 {
        i -= 1;
    }
    if i == 0 {
        return 0;
    }
    ((i - 1) * 64 + (64 - limbs[i - 1].leading_zeros()) as usize) as u32
}

/// Extract bits [start, start+n-1] from a big integer (n <= 128).
fn bigint_extract_bits(limbs: &[u64], start: u32, n: u32) -> u128 {
    let mut result: u128 = 0;
    for i in 0..n {
        let pos = (start + i) as usize;
        let word = pos / 64;
        let bit = pos % 64;
        if word < limbs.len() && (limbs[word] >> bit) & 1 != 0 {
            result |= 1u128 << i;
        }
    }
    result
}

/// Whether bit `pos` of a big integer is 1 (pos is i64 to allow negative values to return false).
fn bigint_bit(limbs: &[u64], pos: i64) -> bool {
    if pos < 0 {
        return false;
    }
    let pos = pos as usize;
    let word = pos / 64;
    let bit = pos % 64;
    word < limbs.len() && (limbs[word] >> bit) & 1 != 0
}

/// Whether the low n bits of a big integer are non-zero.
fn bigint_low_nonzero(limbs: &[u64], n: u32) -> bool {
    if n == 0 {
        return false;
    }
    let words = (n / 64) as usize;
    let bits = n % 64;
    for i in 0..words.min(limbs.len()) {
        if limbs[i] != 0 {
            return true;
        }
    }
    if bits > 0 && words < limbs.len() {
        let mask = (1u64 << bits) - 1;
        if limbs[words] & mask != 0 {
            return true;
        }
    }
    false
}

/// Convert the low 128 bits of a big integer to u128.
fn bigint_to_u128(limbs: &[u64]) -> u128 {
    let mut r = 0u128;
    for i in 0..2.min(limbs.len()) {
        r |= (limbs[i] as u128) << (64 * i);
    }
    r
}

/// Decimal float literal -> IEEE 754 binary128 bit pattern ([u8;16] little-endian).
///
/// Performs exact conversion via big-integer arithmetic without an f64 intermediary (round-to-nearest-even).
/// Supports: [+-]digits[.digits][e[+-]digits]
pub(crate) fn parse_decimal_f128(s: &str) -> Option<[u8; 16]> {
    // 1. Parse the decimal format
    let s = s.trim();
    let (sign, body) = if let Some(rest) = s.strip_prefix('-') {
        (true, rest)
    } else if let Some(rest) = s.strip_prefix('+') {
        (false, rest)
    } else {
        (false, s)
    };

    // Split the exponent part e/E
    let (mantissa_str, exp_str) = match body.find(|c| c == 'e' || c == 'E') {
        Some(pos) => (&body[..pos], &body[pos + 1..]),
        None => (body, ""),
    };
    let exp10: i32 = if exp_str.is_empty() { 0 } else { exp_str.parse().ok()? };

    // Split the decimal point
    let (int_part, frac_part) = match mantissa_str.find('.') {
        Some(pos) => (&mantissa_str[..pos], &mantissa_str[pos + 1..]),
        None => (mantissa_str, ""),
    };

    let digits_str: String = format!("{}{}", int_part, frac_part);
    if digits_str.is_empty() || !digits_str.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }

    let frac_len = frac_part.len() as i32;
    let e10 = exp10 - frac_len;

    // Fast return for zero
    if digits_str.chars().all(|c| c == '0') {
        let bits: u128 = if sign { 1u128 << 127 } else { 0 };
        return Some(bits.to_le_bytes());
    }

    // 2. digits -> big integer M
    let mut m_big = bigint_from_dec(&digits_str);
    let digits_bitlen = bigint_bit_len(&m_big);
    let mut e2: i64 = 0;
    let mut div_sticky = false;

    // 3. Estimate range, fast-path inf/0
    let log2_est = (digits_bitlen as f64 - 1.0) + (e10 as f64) * 3.32193;
    if log2_est > 16384.0 {
        let bits: u128 = (if sign { 1u128 << 127 } else { 0 }) | (0x7FFFu128 << 112);
        return Some(bits.to_le_bytes());
    }
    if log2_est < -16510.0 {
        let bits: u128 = if sign { 1u128 << 127 } else { 0 };
        return Some(bits.to_le_bytes());
    }

    // 4. Handle e10: value = digits * 10^e10 = digits * 5^e10 * 2^e10
    if e10 > 0 {
        for _ in 0..e10 {
            bigint_mul_small(&mut m_big, 5);
        }
        e2 = e10 as i64;
    } else if e10 < 0 {
        // value = digits / 10^k = (digits * 2^P / 5^k) * 2^(-k-P), k = -e10
        let k = (-e10) as u64;
        // P must ensure at least 114 bits of precision after M/5^k: P >= 114 - digits_bitlen + 2.322*k
        let p_needed = (2.4 * (k as f64)) as u32 + 128;
        bigint_shl(&mut m_big, p_needed);
        e2 = -(k as i64) - (p_needed as i64);
        for _ in 0..k {
            let r = bigint_divmod_small(&mut m_big, 5);
            if r != 0 {
                div_sticky = true;
            }
        }
    }

    // 5. Normalize + extract mantissa + guard + sticky
    let msb = bigint_bit_len(&m_big) as i64 - 1;
    if msb < 0 {
        let bits: u128 = if sign { 1u128 << 127 } else { 0 };
        return Some(bits.to_le_bytes());
    }
    let unbiased_exp = e2 + msb;

    let (bits113, guard, sticky, final_exp): (u128, bool, bool, i64) =
        if unbiased_exp >= -16382 {
            // Normal number: mantissa is 113 bits (the msb is the implicit 1)
            let shift = msb - 112;
            if shift >= 0 {
                let s = shift as u32;
                let mant = bigint_extract_bits(&m_big, s, 113);
                let g = bigint_bit(&m_big, (shift - 1) as i64);
                let stk = if s >= 2 {
                    bigint_low_nonzero(&m_big, s - 1)
                } else {
                    false
                };
                (mant, g, stk || div_sticky, unbiased_exp)
            } else {
                // Left-shift to fill; M is represented exactly (no guard)
                let mut m = m_big.clone();
                bigint_shl(&mut m, (-shift) as u32);
                let mant = bigint_to_u128(&m) & ((1u128 << 113) - 1);
                (mant, false, div_sticky, unbiased_exp)
            }
        } else {
            // Subnormal number: fraction is 112 bits, exp is fixed at -16382
            // fraction = M * 2^(e2 + 16494)
            let p = e2 + 16494;
            if p >= 0 {
                let mut m = m_big.clone();
                bigint_shl(&mut m, p as u32);
                let frac = bigint_to_u128(&m) & ((1u128 << 112) - 1);
                (frac, false, div_sticky, -16382)
            } else {
                let s = (-p) as u32;
                let frac = bigint_extract_bits(&m_big, s, 112);
                let g = bigint_bit(&m_big, (-p - 1) as i64);
                let stk = if s >= 2 {
                    bigint_low_nonzero(&m_big, s - 1)
                } else {
                    false
                };
                (frac, g, stk || div_sticky, -16382)
            }
        };

    // 6. Round to nearest even
    let mut mant = bits113;
    let mut exp = final_exp;
    let was_subnormal = final_exp < -16382;
    if guard && (sticky || (mant & 1) != 0) {
        mant += 1;
    }
    if was_subnormal {
        // Subnormal rounding may carry up to the smallest normal (mant reaches 2^112)
        if mant >= (1u128 << 112) {
            exp = -16382;
        }
    } else if mant >= (1u128 << 113) {
        // Normal number rounding carry
        mant >>= 1;
        exp += 1;
    }

    // 7. Assemble binary128
    if exp >= 16383 {
        let bits: u128 = (if sign { 1u128 << 127 } else { 0 }) | (0x7FFFu128 << 112);
        return Some(bits.to_le_bytes());
    }
    if exp >= -16382 {
        // Normal number
        let frac = mant & ((1u128 << 112) - 1);
        let biased = (exp + 16383) as u128;
        let bits = (if sign { 1u128 << 127 } else { 0 }) | (biased << 112) | frac;
        return Some(bits.to_le_bytes());
    }
    // Subnormal number
    let frac = mant & ((1u128 << 112) - 1);
    let bits = (if sign { 1u128 << 127 } else { 0 }) | frac;
    Some(bits.to_le_bytes())
}

/// Resolve a TypeRef to a type-name string using the given arena.
/// Used by build_abi_sig to resolve types in the correct module's arena.
fn type_name_in_arena(
    ty: Option<crate::ast::Ast::TypeRef>,
    arena: &crate::ast::Ast::AstArena<'_>,
) -> String {
    use crate::ast::Ast::TypeNode;
    let ty_ref = match ty {
        Some(t) => t,
        None => return String::new(),
    };
    let node = match arena.types.get(ty_ref.0 as usize) {
        Some(n) => n,
        None => return String::new(),
    };
    match &node.node {
        TypeNode::Named { name } => (*name).to_string(),
        TypeNode::RawPtr { inner } => {
            let inner_name = type_name_in_arena(Some(*inner), arena);
            format!("*{inner_name}")
        }
        TypeNode::Array { element_type, size } => {
            let elem_name = type_name_in_arena(Some(*element_type), arena);
            if size.is_none() {
                format!("{elem_name}[]")
            } else {
                elem_name
            }
        }
        _ => String::new(),
    }
}