//! Solver — Constraint kind, error, and solver. Mechanically split from Inference.rs (no logic changes).

use super::*;

// =========================================================================
// sema v2: Constraint Solver — unified constraint solving engine.
//
// Design philosophy (original, not borrowed from GHC/rustc/Swift):
// - All type relations (equality, subtype, trait bound, narrowing) are unified into Constraint.
// - snapshot/rollback supports speculative inference (match arms, overload selection).
// - Batch solving: solve all at once at function-body end, rather than unifying eagerly.
// - DOD: constraints in Vec, snapshot via length index, subst via FxHashMap.
//
// Relationship with the existing TypeArena::unify:
// The solver calls unify to implement Equality constraints, but adds deferral and rollback
// capability.
// Existing eager unify calls remain compatible; new code may opt into the solver.
// =========================================================================

/// Constraint kinds: unifies all type relations into constraints.
///
/// Design notes:
/// - Equality: most common; directly calls TypeArena::unify.
/// - Subtype: calls is_subtype; on failure records an error but does not abort immediately.
/// - TraitBound: whether `ty` implements a trait (deferred to witness table lookup).
/// - Narrow: path-sensitive narrowing (used by flow narrowing).
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Type equality constraint: `t1 = t2`.
    Equality(TypeHandle, TypeHandle),
    /// Subtype constraint: `sub <: sup` (directional, asymmetric).
    Subtype(TypeHandle, TypeHandle),
    /// Trait bound constraint: `ty` implements trait `trait_name<type_args>`.
    TraitBound {
        ty: TypeHandle,
        trait_name: Box<str>,
        type_args: Box<[TypeHandle]>,
    },
    /// Narrowing constraint: on some path `original` is narrowed to `narrowed`.
    /// Used for flow-sensitive narrowing (NonNull/IsCheck/ConstructorMatch).
    Narrow {
        path: Box<str>,
        original: TypeHandle,
        narrowed: TypeHandle,
    },
}

/// Extracts associated span info from a constraint (for error localization).
/// Constraint itself does not carry a span; it is passed in separately by the context that
/// generated the constraint.
/// The line/column fields are retained for ConstraintError compatibility; the solver fills 0
/// to indicate "no span".
impl Constraint {
    /// Human-readable name of the constraint kind.
    pub fn kind_str(&self) -> &'static str {
        match self {
            Constraint::Equality(..) => "Equality",
            Constraint::Subtype(..) => "Subtype",
            Constraint::TraitBound { .. } => "TraitBound",
            Constraint::Narrow { .. } => "Narrow",
        }
    }
}

/// Constraint solving error: records the reason for a solving failure without aborting
/// inference (error recovery).
#[derive(Debug, Clone)]
pub struct ConstraintError {
    pub constraint: Constraint,
    pub reason: Box<str>,
    /// Span info: may be passed in by the constraint generator; solver-internal errors fill 0,0.
    pub line: u32,
    pub column: u32,
}

/// Constraint solver: collects constraints and solves them in batch.
///
/// Design:
/// - `pending`: queue of constraints to solve (FIFO).
/// - `subst`: solved TypeVar → TypeHandle mapping (solving results).
/// - `errors`: records of solving failures (does not abort; error recovery).
pub struct ConstraintSolver {
    pending: Vec<Constraint>,
    subst: FxHashMap<u32, TypeHandle>,
    errors: Vec<ConstraintError>,
    /// All candidate bindings each TypeVar received during fixpoint iteration (multi-value
    /// record).
    ///
    /// key = TypeVar idx, value = list of all target type handles this TypeVar was required to
    /// bind to.
    /// After fixpoint convergence, `finalize_solution` deduplicates and detects ambiguity:
    /// - Unique candidate → write into subst.
    /// - Multiple distinct candidates → flag an ambiguity error (still writes the arena's actual
    ///   solution into subst to avoid cascading false positives).
    candidates: FxHashMap<u32, Vec<TypeHandle>>,
}

impl Default for ConstraintSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintSolver {
    pub fn new() -> Self {
        ConstraintSolver {
            pending: Vec::new(),
            subst: FxHashMap::default(),
            errors: Vec::new(),
            candidates: FxHashMap::default(),
        }
    }

    /// Adds a constraint to the pending queue.
    #[inline]
    pub fn add(&mut self, c: Constraint) {
        self.pending.push(c);
    }

    /// Convenience method for adding an equality constraint.
    #[inline]
    pub fn add_equality(&mut self, t1: TypeHandle, t2: TypeHandle) {
        self.add(Constraint::Equality(t1, t2));
    }

    /// Convenience method for adding a subtype constraint.
    #[inline]
    pub fn add_subtype(&mut self, sub: TypeHandle, sup: TypeHandle) {
        self.add(Constraint::Subtype(sub, sup));
    }

    /// Convenience method for adding a trait bound constraint.
    #[inline]
    pub fn add_trait_bound(
        &mut self,
        ty: TypeHandle,
        trait_name: &str,
        type_args: &[TypeHandle],
    ) {
        self.add(Constraint::TraitBound {
            ty,
            trait_name: trait_name.into(),
            type_args: type_args.to_vec().into_boxed_slice(),
        });
    }

    /// Solves all pending constraints in batch.
    ///
    /// Solving strategy:
    /// 1. Equality → TypeArena::unify; on success update subst.
    /// 2. Subtype → is_subtype check; on failure record an error.
    /// 3. TraitBound → query via witness table (requires witness_table to be passed in).
    /// 4. Narrow → update the flow fact table (implemented in phase 3).
    ///
    /// After solving, pending is cleared; results go into subst and errors.
    pub fn solve(&mut self, arena: &mut TypeArena) {
        self.solve_with_witness(arena, None)
    }

    /// Solves all pending constraints in batch (with witness table support).
    ///
    /// Fixpoint iteration: repeatedly scan the constraint queue until a round produces no new
    /// bindings.
    /// Constraints have dependencies (constraint A depends on constraint B binding some TypeVar
    /// first); a single FIFO pass may miss solutions due to timing. Fixpoint iteration eliminates
    /// timing dependencies via retries.
    ///
    /// - Equality: when either side still contains a TypeVar, re-enqueue for the next round;
    ///   when both sides are concrete types, record into errors.
    /// - TraitBound: when ty is still a TypeVar, re-enqueue; otherwise query the witness table.
    /// - Subtype/Narrow: single-pass handling (does not propagate TypeVar bindings).
    /// Termination does not need an iteration cap — it follows from the consumption
    /// invariant below. Every processed constraint occurrence ends in exactly one of:
    /// 1. consumed with success (Equality unify Ok / Narrow bind) — the only paths that
    ///    set `changed`;
    /// 2. consumed with error (concrete mismatch, Subtype/TraitBound failure);
    /// 3. re-enqueued (Equality/TraitBound failure with the TypeVar still unresolved).
    /// A success never re-enqueues, so total occurrences = N0 + R and consumed = N0.
    /// Each round that sets `changed` consumes at least one occurrence, hence at most
    /// N0 rounds can set `changed`; the first `changed == false` round ends the loop.
    /// The debug_assert trips if a future edit ever breaks this (e.g. re-enqueueing on
    /// a success path), which is the only way the loop could fail to terminate.
    pub fn solve_with_witness(&mut self, arena: &mut TypeArena, witness: Option<&WitnessTable>) {
        let total_bound = self.pending.len();
        let mut pending = std::mem::take(&mut self.pending);
        let mut rounds: usize = 0;

        loop {
            debug_assert!(
                rounds <= total_bound,
                "solver fixpoint exceeded consumption bound (rounds={rounds}, constraints={total_bound}): \
                 a success path must consume its constraint, not re-enqueue it"
            );
            rounds += 1;

            if pending.is_empty() {
                break;
            }

            // Take out all current constraints for this round.
            let current = std::mem::take(&mut pending);
            let mut changed = false;

            for c in current {
                match c {
                    Constraint::Equality(t1, t2) => {
                        // Record candidate before resolve/unify (multi-value record).
                        // arena.get returns the raw Type; even if the TypeVar was bound by a
                        // previous unify, get still returns TypeVar(idx), so we can capture
                        // binding requirements from all constraint paths to this TypeVar.
                        self.record_candidate(arena, t1, t2);

                        let r1 = arena.resolve(t1);
                        let r2 = arena.resolve(t2);

                        // Both sides already resolved to the same type; nothing to do.
                        if r1 == r2 {
                            continue;
                        }

                        match arena.unify(r1, r2) {
                            Ok(()) => {
                                changed = true;
                            }
                            Err(_) => {
                                // unify failed: if either side still contains a TypeVar,
                                // re-enqueue for the next round (other constraints may bind
                                // these TypeVars in this round).
                                let r1_has_var = Self::resolve_has_type_var(arena, r1);
                                let r2_has_var = Self::resolve_has_type_var(arena, r2);
                                if r1_has_var || r2_has_var {
                                    pending.push(Constraint::Equality(t1, t2));
                                } else {
                                    // Both sides are concrete types and do not match: real error.
                                    self.errors.push(ConstraintError {
                                        constraint: Constraint::Equality(t1, t2),
                                        reason: "type mismatch".into(),
                                        line: 0,
                                        column: 0,
                                    });
                                }
                            }
                        }
                    }
                    Constraint::Subtype(sub, sup) => {
                        if !is_subtype(arena, sub, sup) {
                            self.errors.push(ConstraintError {
                                constraint: Constraint::Subtype(sub, sup),
                                reason: "not a subtype".into(),
                                line: 0,
                                column: 0,
                            });
                        }
                    }
                    Constraint::TraitBound { ty, trait_name, type_args } => {
                        let resolved = arena.resolve(ty);
                        // ty is still a TypeVar: re-enqueue for the next round.
                        if matches!(arena.get(resolved), Type::TypeVar(_)) {
                            pending.push(Constraint::TraitBound {
                                ty,
                                trait_name,
                                type_args,
                            });
                            continue;
                        }

                        // ty is resolved: query the witness table to decide.
                        if let Some(wt) = witness {
                            let ct = arena.get(resolved);
                            let type_id = match ct {
                                Type::Adt(_) | Type::Generic(_) => {
                                    // User type: type_id is registered externally.
                                    // Cannot access sema_result here; skip (handled uniformly by
                                    // check_module).
                                    None
                                }
                                _ => ct.type_id(),
                            };
                            if let Some(tid) = type_id {
                                if !wt.implements(&trait_name, tid) {
                                    self.errors.push(ConstraintError {
                                        constraint: Constraint::TraitBound {
                                            ty,
                                            trait_name: trait_name.clone(),
                                            type_args: type_args.clone(),
                                        },
                                        reason: format!(
                                            "type does not implement trait '{}'",
                                            trait_name
                                        )
                                        .into(),
                                        line: 0,
                                        column: 0,
                                    });
                                }
                            }
                            // When type_id is None, defer to check_module.
                        }
                    }
                    Constraint::Narrow { original, narrowed, .. } => {
                        // Narrowing constraint: on a specific path `original` is narrowed to
                        // `narrowed`.
                        // Solving strategy: if `original` is an unbound TypeVar, bind it to
                        // `narrowed`; if `original` is already bound, try unify (the narrowed
                        // type must be compatible with the original).
                        let r_orig = arena.resolve(original);
                        let r_narrow = arena.resolve(narrowed);
                        if let Type::TypeVar(idx) = arena.get(r_orig).clone() {
                            // TypeVar unbound: bind directly to the narrowed type.
                            arena.type_vars[idx as usize].bound = Some(r_narrow);
                            changed = true;
                        } else if r_orig != r_narrow {
                            // Already bound: try unify (narrowed type must be a subtype of the
                            // original).
                            match arena.unify(r_orig, r_narrow) {
                                Ok(()) => { changed = true; }
                                Err(_) => {
                                    // Narrowing conflicts with the original type: record but do
                                    // not abort.
                                    self.errors.push(ConstraintError {
                                        constraint: Constraint::Narrow {
                                            path: String::new().into_boxed_str(),
                                            original,
                                            narrowed,
                                        },
                                        reason: "narrowed type conflicts with original".into(),
                                        line: 0,
                                        column: 0,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // Fixpoint: a round with no new bindings and no re-enqueued constraints ends.
            if !changed {
                break;
            }
        }

        // Constraints still pending at exit: their TypeVars never got bound (typically
        // TypeVar ↔ TypeVar constraints with no anchor). Recorded but not reported
        // (defensive); these do not affect correctness.

        // After fixpoint convergence: build subst from candidates and detect ambiguity.
        self.finalize_solution(arena);
    }

    /// Returns whether the resolved TypeHandle still contains an unbound TypeVar.
    /// Used during fixpoint iteration to decide whether to re-enqueue a constraint.
    fn resolve_has_type_var(arena: &TypeArena, ty: TypeHandle) -> bool {
        let resolved = arena.resolve(ty);
        match arena.get(resolved) {
            Type::TypeVar(_) => true,
            // Every composite type (incl. Channel/Async/Lazy/Atomic/Sender/Receiver and Record)
            // delegates child traversal to `for_each_child`. Short-circuits on the first hit.
            _ => {
                let mut found = false;
                arena.for_each_child(resolved, |c| {
                    if !found && Self::resolve_has_type_var(arena, c) {
                        found = true;
                    }
                });
                found
            }
        }
    }

    /// Records a TypeVar's candidate binding into `candidates` (multi-value record).
    ///
    /// Called **before** unify; uses `arena.get` (raw Type, no resolve) to detect TypeVars.
    /// Even if a TypeVar was already bound to a concrete type by a previous unify, `get` still
    /// returns `TypeVar(idx)`, so we can capture binding requirements from all constraint paths
    /// to this TypeVar for later ambiguity detection.
    ///
    /// - If t1 is a TypeVar and t2 is not → candidates[t1.idx].push(t2).
    /// - If t2 is a TypeVar and t1 is not → candidates[t2.idx].push(t1).
    /// - Both sides TypeVars → do not record (var-var bindings are handled directly by unify).
    pub fn record_candidate(&mut self, arena: &TypeArena, t1: TypeHandle, t2: TypeHandle) {
        match (arena.get(t1), arena.get(t2)) {
            (Type::TypeVar(_), Type::TypeVar(_)) => {
                // Both sides are TypeVars: var-var binding is handled by unify; do not record
                // candidates.
            }
            (Type::TypeVar(idx), _) => {
                self.candidates.entry(idx).or_default().push(t2);
            }
            (_, Type::TypeVar(idx)) => {
                self.candidates.entry(idx).or_default().push(t1);
            }
            _ => {}
        }
    }

    /// After fixpoint convergence, builds the final subst from candidates and detects
    /// ambiguity.
    ///
    /// For each TypeVar's candidate set:
    /// 1. Deduplicate based on structural equality (handles are not interned).
    /// 2. Unique candidate → write into subst.
    /// 3. Multiple distinct candidates → flag an ambiguity error; still write the arena's
    ///    actual solution into subst (to avoid cascading false positives).
    fn finalize_solution(&mut self, arena: &mut TypeArena) {
        let candidates = std::mem::take(&mut self.candidates);
        for (idx, cands) in candidates {
            // Deduplicate based on structural equality (not TypeHandle identity).
            // `make()` does not intern types, so two `Type::Bool` from different call
            // sites have different TypeHandles; comparing by handle would wrongly
            // flag them as distinct candidates and emit a false "ambiguous
            // inference" error (e.g. `identity(true) == true`).
            let mut unique: Vec<TypeHandle> = Vec::new();
            for c in &cands {
                let r = arena.resolve(*c);
                if !unique.iter().any(|&u| types_equal(arena, u, r)) {
                    unique.push(r);
                }
            }

            match unique.len() {
                0 => {} // Impossible (cands is non-empty when iterated).
                1 => {
                    // Unique candidate: write into subst and write back
                    // arena.type_vars[idx].bound.
                    // The write-back is critical: diagnostics check
                    // arena.type_vars[idx].bound; without it, an already-solved TypeVar would
                    // still be flagged as unresolved.
                    let resolved = arena.resolve(unique[0]);
                    self.subst.insert(idx, resolved);
                    arena.type_vars[idx as usize].bound = Some(resolved);
                }
                _ => {
                    // null-join rule: a null LITERAL's inferred type is either Null or
                    // Nullable<fresh-var> (unconstrained inner). A candidate set of
                    // {nullish, X} resolves to X? — a null literal is compatible with
                    // any nullable, so it must not make inference ambiguous
                    // (`pick(null, 5)` infers T = i32?). A Nullable with a CONCRETE
                    // inner (a nullable variable) does NOT join — variables never
                    // implicitly lift (#60 strictness).
                    if unique.len() == 2 {
                        let is_nullish = |arena: &TypeArena, h: TypeHandle| {
                            let r = arena.resolve(h);
                            match arena.get(r) {
                                Type::Null => true,
                                Type::Nullable(_) => {
                                    let inner = arena.nullable_inner(r);
                                    matches!(arena.get(inner), Type::TypeVar(_))
                                }
                                _ => false,
                            }
                        };
                        let a = unique[0];
                        let b = unique[1];
                        let base = if is_nullish(arena, a) && !is_nullish(arena, b) {
                            Some(b)
                        } else if is_nullish(arena, b) && !is_nullish(arena, a) {
                            Some(a)
                        } else {
                            None
                        };
                        if let Some(base) = base {
                            let base_r = arena.resolve(base);
                            let nullable = arena.make_nullable(base_r);
                            self.subst.insert(idx, nullable);
                            arena.type_vars[idx as usize].bound = Some(nullable);
                            continue;
                        }
                    }
                    // Multiple distinct candidates: ambiguity.
                    // Pick the arena's actual solution (unify picked the first successful one)
                    // and write it into subst to avoid cascading false positives.
                    let resolved = arena.resolve(cands[0]);
                    self.subst.insert(idx, resolved);
                    arena.type_vars[idx as usize].bound = Some(resolved);
                    // Record the ambiguity error.
                    self.errors.push(ConstraintError {
                        constraint: Constraint::Equality(unique[0], unique[1]),
                        reason: format!(
                            "ambiguous inference for TypeVar{}: {} distinct candidates",
                            idx,
                            unique.len()
                        )
                        .into(),
                        line: 0,
                        column: 0,
                    });
                }
            }
        }
    }

    /// Looks up the solving result of a TypeVar.
    #[inline]
    pub fn lookup_subst(&self, var_idx: u32) -> Option<TypeHandle> {
        self.subst.get(&var_idx).copied()
    }

    /// Returns all solving errors.
    #[inline]
    pub fn errors(&self) -> &[ConstraintError] {
        &self.errors
    }

    /// Returns whether there are any solving errors.
    #[inline]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Returns the number of pending constraints.
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Clears all state (called on module switch).
    pub fn reset(&mut self) {
        self.pending.clear();
        self.subst.clear();
        self.errors.clear();
        self.candidates.clear();
    }
}


// =========================================================================
// Termination tests — the fixpoint loop no longer has an iteration cap; these
// pin the consumption invariant it now relies on.
// =========================================================================
