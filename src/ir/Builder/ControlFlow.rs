//! ControlFlow — Control-flow lowering: if / branch / defer / panic / match / patterns. Mechanically split from Builder.rs (no logic changes).

use super::*;

impl<'a> IrBuilder<'a> {
    /// Compile an `If` expression into a Gate node + two branch subgraphs.
    ///
    /// `cond` is compiled into a condition node; `then`/`else` are each compiled into independent
    /// subgraphs.
    /// The Gate node's `condition_input` points to the cond node, and `branches` carries the
    /// branch subgraph ids.
    /// Branch subgraphs take no parameters (closure variable capture is deferred to a later
    /// stage).
    pub(super) fn compile_if(
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
                capture: false,
                condition_input: cond_node,
                branches: vec![
                    (true, then_sg, then_inputs),
                    (false, else_sg, else_inputs),
                ],
            },
        );
        if std::env::var("FROND_DEBUG_COMPILE").is_ok() {
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
    pub(super) fn compile_branch_subgraph(&mut self, expr: crate::ast::Ast::ExprId) -> (SubGraphId, Vec<NodeId>) {
        let node_start = self.graph.nodes.len() as u32;

        // Frame-chain passthrough (`root_frame_ptr`) lets the branch subgraph directly reference
        // outer nodes without a capture mechanism (no local copy is created; assignments write
        // back to the root frame via WriteBack).
        // `branch_inputs` is empty: the Gate injects no arguments; nodes inside the branch read
        // outer variables via `get_value_by_global` frame-chain backtracking.
        self.enter_scope();
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;

        // W3C: pre-register the branch subgraph so its id is stable across the
        // body compile (nested lambdas also add subgraphs). build_await_node
        // registers EventSourceDecls directly into it — structurally correct
        // scoping replacing the old post-hoc migration from the function sg
        // (Bug #24).
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
        let prev_branch_sg = self.current_branch_sg;
        self.current_branch_sg = Some(sg_id);

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
        self.current_branch_sg = prev_branch_sg;
        self.exit_scope();

        let node_end = self.graph.nodes.len() as u32;
        {
            let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
            sg.node_range = (NodeId(node_start), NodeId(node_end));
            sg.return_node = return_node;
            // has_suspend stays false (historical behavior for branch sgs;
            // their frames are same_function branch frames).
        }
        (sg_id, Vec::new())
    }

    /// Compile a void subgraph (used when there is no `else` branch).
    pub(super) fn compile_void_subgraph(&mut self) -> SubGraphId {
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
    pub(super) fn compile_defer_run_subgraph(&mut self) -> SubGraphId {
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
    pub(super) fn compile_panic_subgraph(&mut self) -> SubGraphId {
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
    pub(super) fn compile_match(
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
                capture: false,
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

    /// Compile a pattern-match discriminant node (returns bool), binding pattern variables to field-extraction nodes.
    ///
    /// Recursively handles all pattern types:
    /// - Wildcard/Variable -> const(true); Variable binds the variable to the scrutinee
    /// - Literal -> eq(scrutinee, lit), selecting compute_fn by type
    /// - Constructor -> constructor-name discriminant + recursive sub-patterns
    /// - Record -> field extraction + recursive sub-patterns
    /// - OrPattern -> left_match || right_match
    /// - Guard -> pattern_match && condition
    pub(super) fn compile_pattern_match(
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
    pub(super) fn compile_pattern_literal_match(
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
    pub(super) fn select_literal_eq_fn(&self, s: &str, _is_unsigned: bool) -> ComputeFnId {
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
    pub(super) fn compile_pattern_constructor(
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
    pub(super) fn compile_pattern_record(
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
    pub(super) fn compile_pattern_literal(&mut self, pl: &crate::ast::Ast::PatternLiteral) -> NodeId {
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

}
