//! Ast.rs — Kuzo syntax tree (merging 8 submodules)

// AST source spans and node wrappers

/// Source span: line and column, used for error reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub line: u32,
    pub column: u32,
}

impl Span {
    pub fn new(line: u32, column: u32) -> Self {
        Self { line, column }
    }
}

/// AST node wrapper: binds a source span to the node body.
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    pub span: Span,
    pub node: T,
}

impl<T> Spanned<T> {
    pub fn new(span: Span, node: T) -> Self {
        Self { span, node }
    }

    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Spanned<U> {
        Spanned {
            span: self.span,
            node: f(self.node),
        }
    }

    pub fn as_ref(&self) -> Spanned<&T> {
        Spanned {
            span: self.span,
            node: &self.node,
        }
    }
}

// =========================================================================
// NodeId — node index (u32, replacing &'a Spanned<T> references)
// =========================================================================

/// Expression node index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(pub u32);
/// Statement node index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StmtId(pub u32);
/// Type node index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(pub u32);
/// Pattern node index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PatternId(pub u32);

// =========================================================================
// AstArena — unified node storage (replacing bumpalo arena + references)
//
// Each of the 4 node types is stored in its own Vec<Spanned<T>>. Nodes are referenced
// by NodeId(u32), eliminating lifetime parameters between nodes. String fields remain `&'a str` (zero-copy).
// =========================================================================

/// Unified AST node storage.
#[derive(Debug, Clone, PartialEq)]
pub struct AstArena<'a> {
    pub exprs: Vec<Spanned<Expr<'a>>>,
    pub stmts: Vec<Spanned<Stmt<'a>>>,
    pub types: Vec<Spanned<TypeNode<'a>>>,
    pub patterns: Vec<Spanned<Pattern<'a>>>,
}

/// 为 AstArena 生成成对的 alloc + accessor 方法。
/// `$id` 为 newtype 构造器（ExprId/StmtId/TypeId/PatternId），`$node` 为节点类型。
macro_rules! arena_accessors {
    ($alloc:ident, $get:ident, $id:ident, $field:ident, $node:ty) => {
        pub fn $alloc(&mut self, span: Span, node: $node) -> $id {
            let id = $id(self.$field.len() as u32);
            self.$field.push(Spanned { span, node });
            id
        }

        /// Index access (with bounds checking).
        pub fn $get(&self, id: $id) -> &Spanned<$node> {
            &self.$field[id.0 as usize]
        }
    };
}

impl<'a> AstArena<'a> {
    pub fn new() -> Self {
        Self {
            exprs: Vec::new(),
            stmts: Vec::new(),
            types: Vec::new(),
            patterns: Vec::new(),
        }
    }

    arena_accessors!(alloc_expr, expr, ExprId, exprs, Expr<'a>);
    arena_accessors!(alloc_stmt, stmt, StmtId, stmts, Stmt<'a>);
    arena_accessors!(alloc_type, ty, TypeId, types, TypeNode<'a>);
    arena_accessors!(alloc_pattern, pattern, PatternId, patterns, Pattern<'a>);
}

impl<'a> Default for AstArena<'a> {
    fn default() -> Self {
        Self::new()
    }
}

// Operator enums, Kind type system, and AST helper types


// =========================================================================
// Operator enums
// =========================================================================

/// Binary operator kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    NotEq,
    RefEq,
    RefNeq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    ConcatList,
    Range,
    RangeInclusive,
    Elvis,
}

/// Compound assignment operator kinds (e.g., `+=`, `-=`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompoundAssignOp {
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    ModAssign,
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,
    ShlAssign,
    ShrAssign,
}

/// Unary operator kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Not,
    Neg,
    BitNot,
}

/// Visibility modifier: distinguishes private from public declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Visibility {
    Private,
    Public,
}

// =========================================================================
// Kind type system
// =========================================================================

/// Type kind: used for higher-kinded type annotations, supporting star types and arrow types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    Star,
    Arrow { param: Box<Kind>, result: Box<Kind> },
}

// =========================================================================
// Helper structs
// =========================================================================

/// Literal pattern used in pattern matching.
#[derive(Debug, Clone, PartialEq)]
pub enum PatternLiteral<'a> {
    Int(&'a str),
    Float(&'a str),
    Bool(bool),
    Char(u32),
    String(&'a str),
    Null,
}

/// Field in a record pattern: field name and its sub-pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternRecordField<'a> {
    pub name: &'a str,
    pub pattern: PatternRef,
}

/// Function/lambda parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct Param<'a> {
    pub name: &'a str,
    pub type_annotation: Option<TypeRef>,
}

/// Type parameter: carries name, kind constraint, and trait bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeParam<'a> {
    pub name: &'a str,
    pub kind: Option<Box<Kind>>,
    pub bounds: Vec<TraitBound<'a>>,
}

/// Trait bound: trait name and type arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct TraitBound<'a> {
    pub trait_name: &'a str,
    pub type_args: Vec<TypeRef>,
}

/// Type constraint: binds a type parameter to a concrete type.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeConstraint<'a> {
    pub type_param: &'a str,
    pub concrete_type: TypeRef,
}

/// Record type field: field name and field type.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordFieldType<'a> {
    pub name: &'a str,
    pub ty: TypeRef,
}

/// Record literal field: field name and field value expression.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordFieldExpr<'a> {
    pub name: &'a str,
    pub value: ExprRef,
}

/// Constructor field: optional field name and type (positional parameter when unnamed).
#[derive(Debug, Clone, PartialEq)]
pub struct ConstructorField<'a> {
    pub name: Option<&'a str>,
    pub ty: TypeRef,
}

/// Part of a string interpolation: literal text or embedded expression.
#[derive(Debug, Clone, PartialEq)]
pub enum InterpolationPart<'a> {
    Literal(&'a str),
    Expression(ExprRef),
}

/// Lambda body: either a block expression or a regular expression.
#[derive(Debug, Clone, PartialEq)]
pub enum LambdaBody {
    Block(ExprRef),
    Expression(ExprRef),
}

/// A match expression arm: pattern, optional guard, and arm body.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: PatternRef,
    pub guard: Option<ExprRef>,
    pub body: ExprRef,
}

/// A select expression arm: receive a channel message or timeout.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectArm<'a> {
    Receive {
        channel_expr: ExprRef,
        binding: Option<&'a str>,
        body: ExprRef,
    },
    Timeout {
        duration: ExprRef,
        body: ExprRef,
    },
}

/// A single import item in an import statement: name and optional alias.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportItem<'a> {
    pub name: &'a str,
    pub alias: Option<&'a str>,
}

/// Constructor definition: name, field list, and optional return type.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstructorDef<'a> {
    pub name: &'a str,
    pub fields: Vec<ConstructorField<'a>>,
    pub return_type: Option<TypeRef>,
}

/// Method declaration: name, type parameters, parameters, return type, optional body, override flag, and delegate info.
#[derive(Debug, Clone, PartialEq)]
pub struct MethodDecl<'a> {
    pub name: &'a str,
    pub type_params: Vec<TypeParam<'a>>,
    pub params: Vec<Param<'a>>,
    pub return_type: Option<TypeRef>,
    pub body: Option<ExprRef>,
    pub is_override: bool,
    pub delegate: Option<DelegateInfo<'a>>,
    pub visibility: Visibility,
    pub is_async: bool,
}

/// Delegate info: delegates a method to a method of some trait.
#[derive(Debug, Clone, PartialEq)]
pub struct DelegateInfo<'a> {
    pub trait_name: &'a str,
    pub method_name: &'a str,
}

/// Associated type declaration within a trait.
#[derive(Debug, Clone, PartialEq)]
pub struct AssociatedType<'a> {
    pub name: &'a str,
    pub kind: Option<Box<Kind>>,
}

// AST node definitions: expressions, statements, declarations, type nodes, patterns, type definitions, and modules


// =========================================================================
// Reference type aliases
//
// After allocation via AstArena, child nodes are held as NodeId(u32) indices, zero-copy referencing the source.
// Aliases are retained to minimize call-site changes (field type semantics change from reference to index).
// =========================================================================

pub type ExprRef = ExprId;
pub type StmtRef = StmtId;
pub type TypeRef = TypeId;
pub type PatternRef = PatternId;

// =========================================================================
// TypeNode — type syntax node
// =========================================================================

/// Type syntax node: named types, generics, nullable, function, record, array, kind annotation, etc.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeNode<'a> {
    /// Named type, e.g. `i32`, `String`.
    Named { name: &'a str },
    /// The `Self` type.
    SelfType,
    /// Generic application, e.g. `List<i32>`.
    Generic { name: &'a str, args: Vec<TypeRef> },
    /// Nullable type `T?`.
    Nullable { inner: TypeRef },
    /// Borrow reference `&T`: a reference to an existing object, shared read/write, RC-managed.
    RefType { inner: TypeRef },
    /// Raw pointer `*T`: bypasses RC, unsafe, reserved for FFI.
    RawPtr { inner: TypeRef },
    /// Function type `(P1, P2) -> R`.
    Function {
        params: Vec<TypeRef>,
        return_type: TypeRef,
    },
    /// Record type `{ x: i32, y: i32 }`.
    Record { fields: Vec<RecordFieldType<'a>> },
    /// Array type `[T; N]`; a slice when `size` is `None`.
    Array {
        element_type: TypeRef,
        size: Option<u64>,
    },
    /// Kind-annotated type `T :: *`.
    KindAnnotated { inner: TypeRef, kind: Box<Kind> },
}

// =========================================================================
// Pattern — pattern matching
// =========================================================================

/// Pattern in pattern matching: wildcard, literal, variable, constructor, record, or-pattern, guard pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern<'a> {
    /// Wildcard `_`.
    Wildcard,
    /// Literal pattern.
    Literal(PatternLiteral<'a>),
    /// Variable binding pattern `x`.
    Variable { name: &'a str },
    /// Constructor pattern `Some(x)`.
    Constructor {
        name: &'a str,
        patterns: Vec<PatternRef>,
    },
    /// Record pattern `{ x, y: p }`.
    Record { fields: Vec<PatternRecordField<'a>> },
    /// Or-pattern `p1 | p2`.
    OrPattern {
        left: PatternRef,
        right: PatternRef,
    },
    /// Guard pattern `p if cond`.
    Guard {
        pattern: PatternRef,
        condition: ExprRef,
    },
}

// =========================================================================
// Expr — expression node
// =========================================================================

/// Expression node: covers all expression forms — literals, identifiers, operations, calls, control flow, pattern matching, etc.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr<'a> {
    /// Integer literal; `raw` keeps the source text, `suffix` is an optional type suffix (e.g. `42i32`).
    IntLit { raw: &'a str, suffix: Option<&'a str> },
    /// Float literal.
    FloatLit { raw: &'a str, suffix: Option<&'a str> },
    /// Boolean literal.
    BoolLit(bool),
    /// Character literal (Unicode scalar value).
    CharLit(u32),
    /// String literal.
    StrLit(&'a str),
    /// String interpolation `"foo ${expr} bar"`.
    StrInterp(Vec<InterpolationPart<'a>>),
    /// The `null` literal.
    NullLit,
    /// The `()` void literal.
    VoidLit,
    /// Identifier reference.
    Ident(&'a str),
    /// Assignment expression `target = value`.
    Assign { target: ExprRef, value: ExprRef },
    /// Compound assignment `target op= value`.
    CompoundAssign {
        op: CompoundAssignOp,
        target: ExprRef,
        value: ExprRef,
    },
    /// Binary operation `lhs op rhs`.
    Binary {
        op: BinaryOp,
        lhs: ExprRef,
        rhs: ExprRef,
    },
    /// Unary operation `op operand`.
    Unary { op: UnaryOp, operand: ExprRef },
    /// Take reference `&expr`.
    RefOf(ExprRef),
    /// Dereference `*expr`.
    Deref(ExprRef),
    /// Function call `callee(args)`; `type_args` are explicit generic arguments.
    Call {
        callee: ExprRef,
        args: Vec<ExprRef>,
        type_args: Option<Vec<TypeRef>>,
    },
    /// Method call `recv.method(args)`.
    MethodCall {
        recv: ExprRef,
        method: &'a str,
        args: Vec<ExprRef>,
        type_args: Option<Vec<TypeRef>>,
    },
    /// Field access `recv.field`.
    FieldAccess { recv: ExprRef, field: &'a str },
    /// Index `recv[index]`.
    Index { recv: ExprRef, index: ExprRef },
    /// Slice `recv[start..end]` or `recv[start..=end]`.
    Slice {
        recv: ExprRef,
        start: ExprRef,
        end: ExprRef,
        inclusive: bool,
    },
    /// Safe field access `recv?.field`.
    SafeAccess { recv: ExprRef, field: &'a str },
    /// Safe method call `recv?.method(args)`.
    SafeMethodCall {
        recv: ExprRef,
        method: &'a str,
        args: Vec<ExprRef>,
        type_args: Option<Vec<TypeRef>>,
    },
    /// Error propagation `expr!`.
    Propagate(ExprRef),
    /// Non-null assertion `expr!!`.
    NonNullAssert(ExprRef),
    /// Elvis operation `lhs ?: rhs`.
    Elvis { lhs: ExprRef, rhs: ExprRef },
    /// Array literal `[a, b, c]` or fill syntax `[value, ..count]`.
    ArrayLit {
        elements: Vec<ExprRef>,
        fill: Option<(ExprRef, ExprRef)>,
    },
    /// Record literal `{ x: 1, y: 2 }`.
    RecordLit(Vec<RecordFieldExpr<'a>>),
    /// Record extension `{ base with x: 1 }`.
    RecordExtend {
        base: ExprRef,
        updates: Vec<RecordFieldExpr<'a>>,
    },
    /// Lambda expression `|params| body`.
    Lambda {
        params: Vec<Param<'a>>,
        body: LambdaBody,
        is_async: bool,
        return_type: Option<TypeRef>,
    },
    /// If expression `if cond { then } else { else_ }`.
    If {
        cond: ExprRef,
        then_branch: ExprRef,
        else_branch: Option<ExprRef>,
    },
    /// Block expression `{ stmts; trailing }`.
    Block {
        stmts: Vec<StmtRef>,
        trailing: Option<ExprRef>,
    },
    /// Match expression `match scrutinee { arms }`.
    Match {
        scrutinee: ExprRef,
        arms: Vec<MatchArm>,
    },
    /// Atomic expression `atomic(expr)`.
    Atomic(ExprRef),
    /// Lazy evaluation `lazy(expr)`.
    Lazy(ExprRef),
    /// Select expression `select { arms }`.
    Select(Vec<SelectArm<'a>>),
    /// Inline trait value `inline_trait { methods }`.
    InlineTrait(Vec<MethodDecl<'a>>),
}

impl<'a> Expr<'a> {
    /// Returns whether this is a literal expression.
    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            Expr::IntLit { .. }
                | Expr::FloatLit { .. }
                | Expr::BoolLit(_)
                | Expr::CharLit(_)
                | Expr::StrLit(_)
                | Expr::NullLit
                | Expr::VoidLit
        )
    }

    /// Returns whether this is an lvalue (assignable target).
    pub fn is_lvalue(&self) -> bool {
        matches!(
            self,
            Expr::Ident(_) | Expr::FieldAccess { .. } | Expr::Index { .. } | Expr::Deref(_)
        )
    }

    /// If this is an identifier expression, returns its name.
    pub fn as_ident(&self) -> Option<&'a str> {
        match self {
            Expr::Ident(name) => Some(*name),
            _ => None,
        }
    }
}

// =========================================================================
// Stmt — statement node
// =========================================================================

/// Statement node: declarations, assignments, control flow (return/throw/break/continue), loops, etc.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt<'a> {
    /// Immutable binding `val name = value`.
    ValDecl {
        name: &'a str,
        type_annotation: Option<TypeRef>,
        value: ExprRef,
        visibility: Visibility,
    },
    /// Mutable binding `var name = value`.
    VarDecl {
        name: &'a str,
        type_annotation: Option<TypeRef>,
        value: ExprRef,
        visibility: Visibility,
    },
    /// Assignment statement `target = value`.
    Assignment {
        target: ExprRef,
        value: ExprRef,
    },
    /// Field assignment `object.field = value`.
    FieldAssignment {
        object: ExprRef,
        field: &'a str,
        value: ExprRef,
    },
    /// Compound assignment `target op= value`.
    CompoundAssignment {
        target: ExprRef,
        op: CompoundAssignOp,
        value: ExprRef,
    },
    /// Pure expression statement `expr`.
    Expression { expr: ExprRef },
    /// Return statement `return value?`.
    Return { value: Option<ExprRef> },
    /// Defer statement `defer expr`.
    Defer { expr: ExprRef },
    /// Throw statement `throw expr`.
    Throw { expr: ExprRef },
    /// Break statement.
    Break,
    /// Continue statement.
    Continue,
    /// For loop `for name in iterable { body }`.
    For {
        name: &'a str,
        iterable: ExprRef,
        body: ExprRef,
    },
    /// While loop `while condition { body }`.
    While {
        condition: ExprRef,
        body: ExprRef,
    },
    /// Loop `loop { body }`.
    Loop { body: ExprRef },
    /// Local declaration (nested fun/type/trait, etc.).
    LocalDecl {
        decl: Box<Decl<'a>>,
    },
}

// =========================================================================
// Attribute — generic attribute
// =========================================================================

/// Generic attribute: `@name`, `@name("arg1", "arg2")`, or `@name "arg"`.
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute<'a> {
    pub name: &'a str,
    pub args: Vec<&'a str>,
}

// =========================================================================
// Decl — top-level declaration
// =========================================================================

/// Top-level declaration: function, type, trait, import, pack, expression declaration.
#[derive(Debug, Clone, PartialEq)]
pub enum Decl<'a> {
    /// Function declaration `fun name(params): ret { body }`.
    FunDecl {
        visibility: Visibility,
        name: &'a str,
        type_params: Vec<TypeParam<'a>>,
        params: Vec<Param<'a>>,
        return_type: Option<TypeRef>,
        bounds: Vec<TraitBound<'a>>,
        body: ExprRef,
        is_async: bool,
        is_entry: bool,
        attributes: Vec<Attribute<'a>>,
        extern_c_body: Option<&'a str>,
    },
    /// Type declaration `type name { ... }`.
    TypeDecl {
        visibility: Visibility,
        name: &'a str,
        type_params: Vec<TypeParam<'a>>,
        implemented_traits: Vec<TraitBound<'a>>,
        type_constraints: Vec<TypeConstraint<'a>>,
        def: TypeDef<'a>,
        methods: Vec<MethodDecl<'a>>,
    },
    /// Trait declaration `trait name { ... }`.
    TraitDecl {
        visibility: Visibility,
        name: &'a str,
        type_params: Vec<TypeParam<'a>>,
        parents: Vec<TraitBound<'a>>,
        associated_types: Vec<AssociatedType<'a>>,
        methods: Vec<MethodDecl<'a>>,
    },
    /// Import declaration `import module_path { items }`.
    ImportDecl {
        module_path: Vec<&'a str>,
        items: Option<Vec<ImportItem<'a>>>,
        visibility: Visibility,
    },
    /// Pack declaration `pack name`.
    PackDecl {
        visibility: Visibility,
        name: &'a str,
    },
    /// Top-level expression declaration.
    ExprDecl {
        expr: ExprRef,
        stmt: Option<StmtRef>,
    },
}

// =========================================================================
// TypeDef — type definition body
// =========================================================================

/// Type definition body: algebraic data type, record, alias, newtype, error newtype.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeDef<'a> {
    /// Algebraic data type `adt { Constructor1 | Constructor2 }`.
    Adt { constructors: Vec<ConstructorDef<'a>> },
    /// Record type `record { field1: T1, field2: T2 }`.
    Record { fields: Vec<RecordFieldType<'a>> },
    /// Type alias `alias = target`.
    Alias { target: TypeRef },
    /// Newtype `newtype name = inner`.
    Newtype { name: &'a str, inner: TypeRef },
}

// =========================================================================
// Module — module
// =========================================================================

/// Module: name, source path, and list of top-level declarations.
#[derive(Debug, Clone, PartialEq)]
pub struct Module<'a> {
    pub name: &'a str,
    pub source_path: Option<&'a str>,
    pub arena: AstArena<'a>,
    pub declarations: Vec<Spanned<Decl<'a>>>,
}

impl<'a> Module<'a> {
    /// Find a function declaration in the module by name.
    pub fn find_function(&self, name: &str) -> Option<&Spanned<Decl<'a>>> {
        self.declarations.iter().find(|d| match &d.node {
            Decl::FunDecl { name: n, .. } => *n == name,
            _ => false,
        })
    }
}

// =========================================================================
// AstVisitor — AST traversal trait (hooks default to no-op, recursion driven by walk_*)
// =========================================================================

/// AST visitor trait. The `visit_*` methods are hooks with empty default implementations.
/// Callers that want recursive traversal use the corresponding `walk_*` free functions (which take `&AstArena` to dereference nodes).
/// Override `visit_*` to intercept that node type; self-driven visitors (e.g. `Printer`) recurse within the hook.
///
/// Not object-safe (the generic `walk_*` methods require `Sized`); static dispatch only, with zero virtual call overhead.
pub trait AstVisitor<'a>: Sized {
    fn visit_module(&mut self, _module: &'a Module<'a>) {}
    fn visit_decl(&mut self, _decl: &'a Spanned<Decl<'a>>) {}
    fn visit_type_def(&mut self, _def: &'a TypeDef<'a>) {}
    fn visit_stmt(&mut self, _stmt: StmtId) {}
    fn visit_expr(&mut self, _expr: ExprId) {}
    fn visit_type(&mut self, _ty: TypeId) {}
    fn visit_pattern(&mut self, _pat: PatternId) {}
    fn visit_kind(&mut self, _kind: &'a Kind) {}
}

// --- walk_* free functions: invoke hook first, then recurse into child nodes ---

pub fn walk_module<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, m: &'a Module<'a>) {
    v.visit_module(m);
    for decl in &m.declarations {
        walk_decl(v, arena, decl);
    }
}

pub fn walk_decl<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, decl: &'a Spanned<Decl<'a>>) {
    v.visit_decl(decl);
    match &decl.node {
        Decl::FunDecl {
            type_params,
            params,
            return_type,
            bounds,
            body,
            ..
        } => {
            for tp in type_params {
                walk_type_param(v, arena, tp);
            }
            for p in params {
                walk_param(v, arena, p);
            }
            if let Some(rt) = return_type {
                walk_type(v, arena, *rt);
            }
            for b in bounds {
                walk_trait_bound(v, arena, b);
            }
            walk_expr(v, arena, *body);
        }
        Decl::TypeDecl {
            type_params,
            implemented_traits,
            type_constraints,
            def,
            methods,
            ..
        } => {
            for tp in type_params {
                walk_type_param(v, arena, tp);
            }
            for b in implemented_traits {
                walk_trait_bound(v, arena, b);
            }
            for c in type_constraints {
                walk_type_constraint(v, arena, c);
            }
            walk_type_def(v, arena, def);
            for m in methods {
                walk_method_decl(v, arena, m);
            }
        }
        Decl::TraitDecl {
            type_params,
            parents,
            associated_types,
            methods,
            ..
        } => {
            for tp in type_params {
                walk_type_param(v, arena, tp);
            }
            for p in parents {
                walk_trait_bound(v, arena, p);
            }
            for at in associated_types {
                walk_associated_type(v, at);
            }
            for m in methods {
                walk_method_decl(v, arena, m);
            }
        }
        Decl::ImportDecl { .. } | Decl::PackDecl { .. } => {}
        Decl::ExprDecl { expr, stmt } => {
            walk_expr(v, arena, *expr);
            if let Some(s) = stmt {
                walk_stmt(v, arena, *s);
            }
        }
    }
}

pub fn walk_type_def<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, def: &'a TypeDef<'a>) {
    v.visit_type_def(def);
    match def {
        TypeDef::Adt { constructors } => {
            for ctor in constructors {
                walk_constructor_def(v, arena, ctor);
            }
        }
        TypeDef::Record { fields } => {
            for f in fields {
                walk_record_field_type(v, arena, f);
            }
        }
        TypeDef::Alias { target } => {
            walk_type(v, arena, *target);
        }
        TypeDef::Newtype { inner, .. } => {
            walk_type(v, arena, *inner);
        }
    }
}

pub fn walk_stmt<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, id: StmtId) {
    v.visit_stmt(id);
    let stmt = arena.stmt(id);
    match &stmt.node {
        Stmt::ValDecl {
            type_annotation,
            value,
            ..
        } => {
            if let Some(ty) = type_annotation {
                walk_type(v, arena, *ty);
            }
            walk_expr(v, arena, *value);
        }
        Stmt::VarDecl {
            type_annotation,
            value,
            ..
        } => {
            if let Some(ty) = type_annotation {
                walk_type(v, arena, *ty);
            }
            walk_expr(v, arena, *value);
        }
        Stmt::Assignment { target, value } => {
            walk_expr(v, arena, *target);
            walk_expr(v, arena, *value);
        }
        Stmt::FieldAssignment {
            object, value, ..
        } => {
            walk_expr(v, arena, *object);
            walk_expr(v, arena, *value);
        }
        Stmt::CompoundAssignment {
            target, value, ..
        } => {
            walk_expr(v, arena, *target);
            walk_expr(v, arena, *value);
        }
        Stmt::Expression { expr } => {
            walk_expr(v, arena, *expr);
        }
        Stmt::Return { value } => {
            if let Some(e) = value {
                walk_expr(v, arena, *e);
            }
        }
        Stmt::Defer { expr } => {
            walk_expr(v, arena, *expr);
        }
        Stmt::Throw { expr } => {
            walk_expr(v, arena, *expr);
        }
        Stmt::Break | Stmt::Continue => {}
        Stmt::For {
            iterable, body, ..
        } => {
            walk_expr(v, arena, *iterable);
            walk_expr(v, arena, *body);
        }
        Stmt::While { condition, body } => {
            walk_expr(v, arena, *condition);
            walk_expr(v, arena, *body);
        }
        Stmt::Loop { body } => {
            walk_expr(v, arena, *body);
        }
        Stmt::LocalDecl { decl } => match decl.as_ref() {
            Decl::FunDecl { params, return_type, body, .. } => {
                for p in params {
                    walk_param(v, arena, p);
                }
                if let Some(rt) = return_type {
                    walk_type(v, arena, *rt);
                }
                walk_expr(v, arena, *body);
            }
            Decl::TypeDecl { methods, .. } => {
                for m in methods {
                    walk_method_decl(v, arena, m);
                }
            }
            Decl::TraitDecl { methods, .. } => {
                for m in methods {
                    walk_method_decl(v, arena, m);
                }
            }
            _ => {}
        },
    }
}

pub fn walk_expr<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, id: ExprId) {
    v.visit_expr(id);
    let expr = arena.expr(id);
    match &expr.node {
        Expr::IntLit { .. }
        | Expr::FloatLit { .. }
        | Expr::BoolLit(_)
        | Expr::CharLit(_)
        | Expr::StrLit(_)
        | Expr::NullLit
        | Expr::VoidLit
        | Expr::Ident(_) => {}
        Expr::StrInterp(parts) => {
            for part in parts {
                walk_interpolation_part(v, arena, part);
            }
        }
        Expr::Assign { target, value } => {
            walk_expr(v, arena, *target);
            walk_expr(v, arena, *value);
        }
        Expr::CompoundAssign { target, value, .. } => {
            walk_expr(v, arena, *target);
            walk_expr(v, arena, *value);
        }
        Expr::Binary { lhs, rhs, .. } => {
            walk_expr(v, arena, *lhs);
            walk_expr(v, arena, *rhs);
        }
        Expr::Unary { operand, .. } => {
            walk_expr(v, arena, *operand);
        }
        Expr::RefOf(inner) | Expr::Deref(inner) | Expr::Propagate(inner)
        | Expr::NonNullAssert(inner) | Expr::Atomic(inner) | Expr::Lazy(inner) => {
            walk_expr(v, arena, *inner);
        }
        Expr::Call {
            callee,
            args,
            type_args,
        } => {
            walk_expr(v, arena, *callee);
            for a in args {
                walk_expr(v, arena, *a);
            }
            if let Some(ta) = type_args {
                for t in ta {
                    walk_type(v, arena, *t);
                }
            }
        }
        Expr::MethodCall {
            recv,
            args,
            type_args,
            ..
        } => {
            walk_expr(v, arena, *recv);
            for a in args {
                walk_expr(v, arena, *a);
            }
            if let Some(ta) = type_args {
                for t in ta {
                    walk_type(v, arena, *t);
                }
            }
        }
        Expr::FieldAccess { recv, .. } | Expr::SafeAccess { recv, .. } => {
            walk_expr(v, arena, *recv);
        }
        Expr::Index { recv, index } => {
            walk_expr(v, arena, *recv);
            walk_expr(v, arena, *index);
        }
        Expr::Slice {
            recv, start, end, ..
        } => {
            walk_expr(v, arena, *recv);
            walk_expr(v, arena, *start);
            walk_expr(v, arena, *end);
        }
        Expr::SafeMethodCall {
            recv,
            args,
            type_args,
            ..
        } => {
            walk_expr(v, arena, *recv);
            for a in args {
                walk_expr(v, arena, *a);
            }
            if let Some(ta) = type_args {
                for t in ta {
                    walk_type(v, arena, *t);
                }
            }
        }
        Expr::Elvis { lhs, rhs } => {
            walk_expr(v, arena, *lhs);
            walk_expr(v, arena, *rhs);
        }
        Expr::ArrayLit { elements, fill } => {
            for e in elements {
                walk_expr(v, arena, *e);
            }
            if let Some((value, count)) = fill {
                walk_expr(v, arena, *value);
                walk_expr(v, arena, *count);
            }
        }
        Expr::RecordLit(fields) => {
            for f in fields {
                walk_record_field_expr(v, arena, f);
            }
        }
        Expr::RecordExtend { base, updates } => {
            walk_expr(v, arena, *base);
            for f in updates {
                walk_record_field_expr(v, arena, f);
            }
        }
        Expr::Lambda {
            params,
            body,
            return_type,
            ..
        } => {
            for p in params {
                walk_param(v, arena, p);
            }
            walk_lambda_body(v, arena, body);
            if let Some(rt) = return_type {
                walk_type(v, arena, *rt);
            }
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            walk_expr(v, arena, *cond);
            walk_expr(v, arena, *then_branch);
            if let Some(e) = else_branch {
                walk_expr(v, arena, *e);
            }
        }
        Expr::Block { stmts, trailing } => {
            for s in stmts {
                walk_stmt(v, arena, *s);
            }
            if let Some(e) = trailing {
                walk_expr(v, arena, *e);
            }
        }
        Expr::Match { scrutinee, arms } => {
            walk_expr(v, arena, *scrutinee);
            for arm in arms {
                walk_match_arm(v, arena, arm);
            }
        }
        Expr::Select(arms) => {
            for arm in arms {
                walk_select_arm(v, arena, arm);
            }
        }
        Expr::InlineTrait(methods) => {
            for m in methods {
                walk_method_decl(v, arena, m);
            }
        }
    }
}

pub fn walk_type<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, id: TypeId) {
    v.visit_type(id);
    let ty = arena.ty(id);
    match &ty.node {
        TypeNode::Named { .. } | TypeNode::SelfType => {}
        TypeNode::Generic { args, .. } => {
            for a in args {
                walk_type(v, arena, *a);
            }
        }
        TypeNode::Nullable { inner }
        | TypeNode::RefType { inner }
        | TypeNode::RawPtr { inner } => {
            walk_type(v, arena, *inner);
        }
        TypeNode::Function {
            params,
            return_type,
        } => {
            for p in params {
                walk_type(v, arena, *p);
            }
            walk_type(v, arena, *return_type);
        }
        TypeNode::Record { fields } => {
            for f in fields {
                walk_record_field_type(v, arena, f);
            }
        }
        TypeNode::Array { element_type, .. } => {
            walk_type(v, arena, *element_type);
        }
        TypeNode::KindAnnotated { inner, kind } => {
            walk_type(v, arena, *inner);
            walk_kind(v, kind);
        }
    }
}

pub fn walk_pattern<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, id: PatternId) {
    v.visit_pattern(id);
    let pat = arena.pattern(id);
    match &pat.node {
        Pattern::Wildcard | Pattern::Literal(_) | Pattern::Variable { .. } => {}
        Pattern::Constructor { patterns, .. } => {
            for p in patterns {
                walk_pattern(v, arena, *p);
            }
        }
        Pattern::Record { fields } => {
            for f in fields {
                walk_pattern(v, arena, f.pattern);
            }
        }
        Pattern::OrPattern { left, right } => {
            walk_pattern(v, arena, *left);
            walk_pattern(v, arena, *right);
        }
        Pattern::Guard { pattern, condition } => {
            walk_pattern(v, arena, *pattern);
            walk_expr(v, arena, *condition);
        }
    }
}

pub fn walk_kind<'a, V: AstVisitor<'a>>(v: &mut V, kind: &'a Kind) {
    v.visit_kind(kind);
    match kind {
        Kind::Star => {}
        Kind::Arrow { param, result } => {
            walk_kind(v, param);
            walk_kind(v, result);
        }
    }
}

// --- walk functions for helper structs ---

fn walk_type_param<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, tp: &'a TypeParam<'a>) {
    if let Some(k) = &tp.kind {
        walk_kind(v, k);
    }
    for b in &tp.bounds {
        walk_trait_bound(v, arena, b);
    }
}

fn walk_trait_bound<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, b: &'a TraitBound<'a>) {
    for t in &b.type_args {
        walk_type(v, arena, *t);
    }
}

fn walk_type_constraint<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, c: &'a TypeConstraint<'a>) {
    walk_type(v, arena, c.concrete_type);
}

fn walk_param<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, p: &'a Param<'a>) {
    if let Some(ty) = &p.type_annotation {
        walk_type(v, arena, *ty);
    }
}

fn walk_associated_type<'a, V: AstVisitor<'a>>(v: &mut V, at: &'a AssociatedType<'a>) {
    if let Some(k) = &at.kind {
        walk_kind(v, k);
    }
}

fn walk_method_decl<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, m: &'a MethodDecl<'a>) {
    for tp in &m.type_params {
        walk_type_param(v, arena, tp);
    }
    for p in &m.params {
        walk_param(v, arena, p);
    }
    if let Some(rt) = &m.return_type {
        walk_type(v, arena, *rt);
    }
    if let Some(body) = &m.body {
        walk_expr(v, arena, *body);
    }
}

fn walk_constructor_def<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, ctor: &'a ConstructorDef<'a>) {
    for f in &ctor.fields {
        walk_constructor_field(v, arena, f);
    }
    if let Some(rt) = &ctor.return_type {
        walk_type(v, arena, *rt);
    }
}

fn walk_constructor_field<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, f: &'a ConstructorField<'a>) {
    walk_type(v, arena, f.ty);
}

fn walk_record_field_type<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, f: &'a RecordFieldType<'a>) {
    walk_type(v, arena, f.ty);
}

fn walk_record_field_expr<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, f: &'a RecordFieldExpr<'a>) {
    walk_expr(v, arena, f.value);
}

fn walk_interpolation_part<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, part: &'a InterpolationPart<'a>) {
    if let InterpolationPart::Expression(e) = part {
        walk_expr(v, arena, *e);
    }
}

fn walk_lambda_body<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, body: &LambdaBody) {
    match body {
        LambdaBody::Block(e) | LambdaBody::Expression(e) => {
            walk_expr(v, arena, *e);
        }
    }
}

fn walk_match_arm<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, arm: &MatchArm) {
    walk_pattern(v, arena, arm.pattern);
    if let Some(g) = &arm.guard {
        walk_expr(v, arena, *g);
    }
    walk_expr(v, arena, arm.body);
}

fn walk_select_arm<'a, V: AstVisitor<'a>>(v: &mut V, arena: &'a AstArena<'a>, arm: &'a SelectArm<'a>) {
    match arm {
        SelectArm::Receive {
            channel_expr, body, ..
        } => {
            walk_expr(v, arena, *channel_expr);
            walk_expr(v, arena, *body);
        }
        SelectArm::Timeout { duration, body } => {
            walk_expr(v, arena, *duration);
            walk_expr(v, arena, *body);
        }
    }
}


// =========================================================================
// AST printer (the Printer implementation below is based on the AstVisitor trait)
// =========================================================================

// AST printer
//
// Serializes the AST into canonical S-expression text, used for:
// - Debugging: visualizing parse results
// - Verification: diff against the Zig original ast_printer output to ensure semantic parity
//
// Format conventions:
// - One node per line, `(node_type field1 field2 ...)`
// - Nested nodes indented by 2 spaces
// - String literals wrapped in double quotes, with `"` `\` `\n` escaped inside
// - Empty lists print `()`; missing optional values print `(none)`
// - Identifiers/names printed bare (unquoted) to distinguish them from string literals


/// AST printer: accumulates output text and indentation level.
pub struct Printer<'a> {
    buf: String,
    indent: usize,
    arena: &'a AstArena<'a>,
}

// --- Printing helper macros ---

/// Generates a `(op <name>)` print method for an operator type.
macro_rules! impl_print_op {
    ($method:ident, $op:ty, $conv:ident) => {
        fn $method(&mut self, op: $op) {
            self.write_line(&format!("(op {})", $conv(op)));
        }
    };
}

/// Generates a labeled list print method: empty lists print `(label ())`, otherwise each item is printed.
/// List items are NodeIds (Copy); each is dereferenced and passed to the corresponding visit_*.
macro_rules! impl_print_list {
    ($method:ident, $item:ty, $print_fn:ident) => {
        fn $method(&mut self, label: &str, items: &[$item]) {
            if items.is_empty() {
                self.write_line(&format!("({} ())", label));
                return;
            }
            self.write_line(&format!("({}", label));
            self.indent();
            for e in items {
                self.$print_fn(*e);
            }
            self.dedent();
            self.write_line(")");
        }
    };
}

impl<'a> Printer<'a> {
    /// Creates a printer; requires the module's AST arena to dereference nodes.
    pub fn new(arena: &'a AstArena<'a>) -> Self {
        Self {
            buf: String::new(),
            indent: 0,
            arena,
        }
    }

    /// Dereferences a `&ExprId` and visits it.
    fn ve(&mut self, id: &ExprId) {
        self.visit_expr(*id);
    }
    /// Dereferences a `&TypeId` and visits it.
    fn vt(&mut self, id: &TypeId) {
        self.visit_type(*id);
    }
    /// Dereferences a `&StmtId` and visits it.
    fn vs(&mut self, id: &StmtId) {
        self.visit_stmt(*id);
    }
    /// Dereferences a `&PatternId` and visits it.
    fn vp(&mut self, id: &PatternId) {
        self.visit_pattern(*id);
    }

    /// Prints the module as canonical text.
    pub fn print_module(&mut self, module: &'a Module<'a>) -> &str {
        self.write_line(&format!("(module \"{}\"", escape_str(module.name)));
        self.indent();
        if let Some(path) = module.source_path {
            self.write_line(&format!("(source_path \"{}\")", escape_str(path)));
        }
        for decl in &module.declarations {
            self.visit_decl(decl);
        }
        self.dedent();
        self.write_line(")");
        &self.buf
    }

    // --- Indentation helpers ---

    fn indent(&mut self) {
        self.indent += 1;
    }

    fn dedent(&mut self) {
        if self.indent > 0 {
            self.indent -= 1;
        }
    }

    fn write_line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.buf.push_str("  ");
        }
        self.buf.push_str(text);
        self.buf.push('\n');
    }
}

impl<'a> AstVisitor<'a> for Printer<'a> {
    // --- Declarations ---

    fn visit_decl(&mut self, decl: &'a Spanned<Decl<'a>>) {
        match &decl.node {
            Decl::FunDecl {
                visibility,
                name,
                type_params,
                params,
                return_type,
                bounds,
                body,
                is_async,
                is_entry,
                attributes,
                extern_c_body,
            } => {
                self.write_line(&format!("(fun_decl \"{}\"", name));
                self.indent();
                for attr in attributes {
                    if attr.args.is_empty() {
                        self.write_line(&format!("(attribute \"{}\")", attr.name));
                    } else {
                        let args_str = attr.args.iter().map(|a| format!("\"{}\"", a)).collect::<Vec<_>>().join(" ");
                        self.write_line(&format!("(attribute \"{}\" (args {}))", attr.name, args_str));
                    }
                }
                self.print_visibility(*visibility);
                self.print_type_params(type_params);
                self.print_params(params);
                self.print_return_type(return_type);
                self.print_bounds(bounds);
                self.write_line(&format!("(is_async {})(is_entry {})", is_async, is_entry));
                self.write_line("(body");
                self.indent();
                self.ve(body);
                self.dedent();
                self.write_line(")");
                if let Some(c_body) = extern_c_body {
                    self.write_line("(extern_c_body");
                    self.indent();
                    self.write_line(&format!("\"{}\"", c_body.escape_default()));
                    self.dedent();
                    self.write_line(")");
                }
                self.dedent();
                self.write_line(")");
            }
            Decl::TypeDecl {
                visibility,
                name,
                type_params,
                implemented_traits,
                type_constraints,
                def,
                methods,
            } => {
                self.write_line(&format!("(type_decl \"{}\"", name));
                self.indent();
                self.print_visibility(*visibility);
                self.print_type_params(type_params);
                self.print_bounds(implemented_traits);
                self.print_type_constraints(type_constraints);
                self.visit_type_def(def);
                self.print_methods(methods);
                self.dedent();
                self.write_line(")");
            }
            Decl::TraitDecl {
                visibility,
                name,
                type_params,
                parents,
                associated_types,
                methods,
            } => {
                self.write_line(&format!("(trait_decl \"{}\"", name));
                self.indent();
                self.print_visibility(*visibility);
                self.print_type_params(type_params);
                self.print_bounds(parents);
                self.print_associated_types(associated_types);
                self.print_methods(methods);
                self.dedent();
                self.write_line(")");
            }
            Decl::ImportDecl {
                module_path,
                items,
                visibility,
            } => {
                let path_str = module_path.join(".");
                self.write_line(&format!("(import_decl \"{}\"", path_str));
                self.indent();
                self.print_visibility(*visibility);
                match items {
                    Some(item_list) => {
                        self.write_line("(items");
                        self.indent();
                        for item in item_list {
                            match item.alias {
                                Some(alias) => {
                                    self.write_line(&format!(
                                        "(item \"{}\" (alias \"{}\"))",
                                        item.name, alias
                                    ));
                                }
                                None => {
                                    self.write_line(&format!("(item \"{}\")", item.name));
                                }
                            }
                        }
                        self.dedent();
                        self.write_line(")");
                    }
                    None => self.write_line("(items (none))"),
                }
                self.dedent();
                self.write_line(")");
            }
            Decl::PackDecl { visibility, name } => {
                self.write_line(&format!("(pack_decl \"{}\"", name));
                self.indent();
                self.print_visibility(*visibility);
                self.dedent();
                self.write_line(")");
            }
            Decl::ExprDecl { expr, stmt } => {
                self.write_line("(expr_decl");
                self.indent();
                self.ve(expr);
                if let Some(s) = stmt {
                    self.write_line("(stmt");
                    self.indent();
                    self.vs(s);
                    self.dedent();
                    self.write_line(")");
                } else {
                    self.write_line("(stmt (none))");
                }
                self.dedent();
                self.write_line(")");
            }
        }
    }

    // --- Type definitions ---

    fn visit_type_def(&mut self, def: &'a TypeDef<'a>) {
        match def {
            TypeDef::Adt { constructors } => {
                self.write_line("(adt");
                self.indent();
                for ctor in constructors {
                    self.write_line(&format!("(constructor \"{}\"", ctor.name));
                    self.indent();
                    if ctor.fields.is_empty() {
                        self.write_line("(fields ())");
                    } else {
                        self.write_line("(fields");
                        self.indent();
                        for field in &ctor.fields {
                            match field.name {
                                Some(fname) => {
                                    self.write_line(&format!("(field \"{}\"", fname));
                                    self.indent();
                                    self.vt(&field.ty);
                                    self.dedent();
                                    self.write_line(")");
                                }
                                None => {
                                    self.write_line("(positional_field");
                                    self.indent();
                                    self.vt(&field.ty);
                                    self.dedent();
                                    self.write_line(")");
                                }
                            }
                        }
                        self.dedent();
                        self.write_line(")");
                    }
                    if let Some(rt) = &ctor.return_type {
                        self.write_line("(return_type");
                        self.indent();
                        self.vt(rt);
                        self.dedent();
                        self.write_line(")");
                    } else {
                        self.write_line("(return_type (none))");
                    }
                    self.dedent();
                    self.write_line(")");
                }
                self.dedent();
                self.write_line(")");
            }
            TypeDef::Record { fields } => {
                self.write_line("(record");
                self.indent();
                if fields.is_empty() {
                    self.write_line("(fields ())");
                } else {
                    self.write_line("(fields");
                    self.indent();
                    for field in fields {
                        self.write_line(&format!("(field \"{}\"", field.name));
                        self.indent();
                        self.vt(&field.ty);
                        self.dedent();
                        self.write_line(")");
                    }
                    self.dedent();
                    self.write_line(")");
                }
                self.dedent();
                self.write_line(")");
            }
            TypeDef::Alias { target } => {
                self.write_line("(alias");
                self.indent();
                self.vt(target);
                self.dedent();
                self.write_line(")");
            }
            TypeDef::Newtype { name, inner } => {
                self.write_line(&format!("(newtype \"{}\"", name));
                self.indent();
                self.vt(inner);
                self.dedent();
                self.write_line(")");
            }
        }
    }
    // --- Type nodes ---

    fn visit_type(&mut self, ty: TypeId) {
        match &self.arena.ty(ty).node {
            TypeNode::Named { name } => {
                self.write_line(&format!("(type_named \"{}\")", name));
            }
            TypeNode::SelfType => {
                self.write_line("(type_self)");
            }
            TypeNode::Generic { name, args } => {
                self.write_line(&format!("(type_generic \"{}\"", name));
                self.indent();
                self.print_type_list("type_args", args);
                self.dedent();
                self.write_line(")");
            }
            TypeNode::Nullable { inner } => {
                self.write_line("(type_nullable");
                self.indent();
                self.vt(inner);
                self.dedent();
                self.write_line(")");
            }
            TypeNode::RefType { inner } => {
                self.write_line("(type_ref");
                self.indent();
                self.vt(inner);
                self.dedent();
                self.write_line(")");
            }
            TypeNode::RawPtr { inner } => {
                self.write_line("(type_raw_ptr");
                self.indent();
                self.vt(inner);
                self.dedent();
                self.write_line(")");
            }
            TypeNode::Function {
                params,
                return_type,
            } => {
                self.write_line("(type_function");
                self.indent();
                self.print_type_list("params", params);
                self.write_line("(return_type");
                self.indent();
                self.vt(return_type);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            TypeNode::Record { fields } => {
                self.write_line("(type_record");
                self.indent();
                if fields.is_empty() {
                    self.write_line("(fields ())");
                } else {
                    self.write_line("(fields");
                    self.indent();
                    for field in fields {
                        self.write_line(&format!("(field \"{}\"", field.name));
                        self.indent();
                        self.vt(&field.ty);
                        self.dedent();
                        self.write_line(")");
                    }
                    self.dedent();
                    self.write_line(")");
                }
                self.dedent();
                self.write_line(")");
            }
            TypeNode::Array {
                element_type,
                size,
            } => {
                self.write_line("(type_array");
                self.indent();
                self.write_line("(element_type");
                self.indent();
                self.vt(element_type);
                self.dedent();
                self.write_line(")");
                match size {
                    Some(n) => self.write_line(&format!("(size {})", n)),
                    None => self.write_line("(size (none))"),
                }
                self.dedent();
                self.write_line(")");
            }
            TypeNode::KindAnnotated { inner, kind } => {
                self.write_line("(type_kind_annotated");
                self.indent();
                self.vt(inner);
                self.write_line("(kind");
                self.indent();
                self.visit_kind(kind);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
        }
    }

    fn visit_kind(&mut self, kind: &'a Kind) {
        match kind {
            Kind::Star => self.write_line("(kind_star)"),
            Kind::Arrow { param, result } => {
                self.write_line("(kind_arrow");
                self.indent();
                self.write_line("(param");
                self.indent();
                self.visit_kind(param);
                self.dedent();
                self.write_line(")");
                self.write_line("(result");
                self.indent();
                self.visit_kind(result);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
        }
    }

    // --- Expressions ---

    fn visit_expr(&mut self, expr: ExprId) {
        match &self.arena.expr(expr).node {
            Expr::IntLit { raw, suffix } => match suffix {
                Some(s) => self.write_line(&format!("(int_lit \"{}\" (suffix \"{}\"))", raw, s)),
                None => self.write_line(&format!("(int_lit \"{}\" (suffix (none)))", raw)),
            },
            Expr::FloatLit { raw, suffix } => match suffix {
                Some(s) => self.write_line(&format!("(float_lit \"{}\" (suffix \"{}\"))", raw, s)),
                None => self.write_line(&format!("(float_lit \"{}\" (suffix (none)))", raw)),
            },
            Expr::BoolLit(b) => self.write_line(&format!("(bool_lit {})", b)),
            Expr::CharLit(c) => self.write_line(&format!("(char_lit {})", c)),
            Expr::StrLit(s) => {
                self.write_line(&format!("(str_lit \"{}\")", escape_str(s)));
            }
            Expr::StrInterp(parts) => {
                self.write_line("(str_interp");
                self.indent();
                for part in parts {
                    match part {
                        InterpolationPart::Literal(text) => {
                            self.write_line(&format!("(literal \"{}\")", escape_str(text)));
                        }
                        InterpolationPart::Expression(e) => {
                            self.write_line("(expression");
                            self.indent();
                            self.ve(e);
                            self.dedent();
                            self.write_line(")");
                        }
                    }
                }
                self.dedent();
                self.write_line(")");
            }
            Expr::NullLit => self.write_line("(null_lit)"),
            Expr::VoidLit => self.write_line("(void_lit)"),
            Expr::Ident(name) => self.write_line(&format!("(ident \"{}\")", name)),
            Expr::Assign { target, value } => {
                self.write_line("(assign");
                self.indent();
                self.write_line("(target");
                self.indent();
                self.ve(target);
                self.dedent();
                self.write_line(")");
                self.write_line("(value");
                self.indent();
                self.ve(value);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            Expr::CompoundAssign { op, target, value } => {
                self.write_line("(compound_assign");
                self.indent();
                self.print_compound_assign_op(*op);
                self.write_line("(target");
                self.indent();
                self.ve(target);
                self.dedent();
                self.write_line(")");
                self.write_line("(value");
                self.indent();
                self.ve(value);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            Expr::Binary { op, lhs, rhs } => {
                self.write_line("(binary");
                self.indent();
                self.print_binary_op(*op);
                self.write_line("(lhs");
                self.indent();
                self.ve(lhs);
                self.dedent();
                self.write_line(")");
                self.write_line("(rhs");
                self.indent();
                self.ve(rhs);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            Expr::Unary { op, operand } => {
                self.write_line("(unary");
                self.indent();
                self.print_unary_op(*op);
                self.write_line("(operand");
                self.indent();
                self.ve(operand);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            Expr::RefOf(inner) => {
                self.write_line("(ref_of");
                self.indent();
                self.ve(inner);
                self.dedent();
                self.write_line(")");
            }
            Expr::Deref(inner) => {
                self.write_line("(deref");
                self.indent();
                self.ve(inner);
                self.dedent();
                self.write_line(")");
            }
            Expr::Call {
                callee,
                args,
                type_args,
            } => {
                self.write_line("(call");
                self.indent();
                self.write_line("(callee");
                self.indent();
                self.ve(callee);
                self.dedent();
                self.write_line(")");
                self.print_type_args_option(type_args);
                self.print_expr_list("args", args);
                self.dedent();
                self.write_line(")");
            }
            Expr::MethodCall {
                recv,
                method,
                args,
                type_args,
            } => {
                self.write_line(&format!("(method_call \"{}\"", method));
                self.indent();
                self.write_line("(recv");
                self.indent();
                self.ve(recv);
                self.dedent();
                self.write_line(")");
                self.print_type_args_option(type_args);
                self.print_expr_list("args", args);
                self.dedent();
                self.write_line(")");
            }
            Expr::FieldAccess { recv, field } => {
                self.write_line(&format!("(field_access \"{}\"", field));
                self.indent();
                self.ve(recv);
                self.dedent();
                self.write_line(")");
            }
            Expr::Index { recv, index } => {
                self.write_line("(index");
                self.indent();
                self.write_line("(recv");
                self.indent();
                self.ve(recv);
                self.dedent();
                self.write_line(")");
                self.write_line("(index");
                self.indent();
                self.ve(index);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            Expr::Slice {
                recv,
                start,
                end,
                inclusive,
            } => {
                self.write_line(&format!("(slice (inclusive {})", inclusive));
                self.indent();
                self.write_line("(recv");
                self.indent();
                self.ve(recv);
                self.dedent();
                self.write_line(")");
                self.write_line("(start");
                self.indent();
                self.ve(start);
                self.dedent();
                self.write_line(")");
                self.write_line("(end");
                self.indent();
                self.ve(end);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            Expr::SafeAccess { recv, field } => {
                self.write_line(&format!("(safe_access \"{}\"", field));
                self.indent();
                self.ve(recv);
                self.dedent();
                self.write_line(")");
            }
            Expr::SafeMethodCall {
                recv,
                method,
                args,
                type_args,
            } => {
                self.write_line(&format!("(safe_method_call \"{}\"", method));
                self.indent();
                self.write_line("(recv");
                self.indent();
                self.ve(recv);
                self.dedent();
                self.write_line(")");
                self.print_type_args_option(type_args);
                self.print_expr_list("args", args);
                self.dedent();
                self.write_line(")");
            }
            Expr::Propagate(inner) => {
                self.write_line("(propagate");
                self.indent();
                self.ve(inner);
                self.dedent();
                self.write_line(")");
            }
            Expr::NonNullAssert(inner) => {
                self.write_line("(non_null_assert");
                self.indent();
                self.ve(inner);
                self.dedent();
                self.write_line(")");
            }
            Expr::Elvis { lhs, rhs } => {
                self.write_line("(elvis");
                self.indent();
                self.write_line("(lhs");
                self.indent();
                self.ve(lhs);
                self.dedent();
                self.write_line(")");
                self.write_line("(rhs");
                self.indent();
                self.ve(rhs);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            Expr::ArrayLit { elements, fill } => {
                self.write_line("(array_lit");
                self.indent();
                self.print_expr_list("elements", elements);
                match fill {
                    Some((value, count)) => {
                        self.write_line("(fill");
                        self.indent();
                        self.write_line("(value");
                        self.indent();
                        self.ve(value);
                        self.dedent();
                        self.write_line(")");
                        self.write_line("(count");
                        self.indent();
                        self.ve(count);
                        self.dedent();
                        self.write_line(")");
                        self.dedent();
                        self.write_line(")");
                    }
                    None => self.write_line("(fill (none))"),
                }
                self.dedent();
                self.write_line(")");
            }
            Expr::RecordLit(fields) => {
                self.write_line("(record_lit");
                self.indent();
                if fields.is_empty() {
                    self.write_line("(fields ())");
                } else {
                    self.write_line("(fields");
                    self.indent();
                    for f in fields {
                        self.write_line(&format!("(field \"{}\"", f.name));
                        self.indent();
                        self.ve(&f.value);
                        self.dedent();
                        self.write_line(")");
                    }
                    self.dedent();
                    self.write_line(")");
                }
                self.dedent();
                self.write_line(")");
            }
            Expr::RecordExtend { base, updates } => {
                self.write_line("(record_extend");
                self.indent();
                self.write_line("(base");
                self.indent();
                self.ve(base);
                self.dedent();
                self.write_line(")");
                if updates.is_empty() {
                    self.write_line("(updates ())");
                } else {
                    self.write_line("(updates");
                    self.indent();
                    for f in updates {
                        self.write_line(&format!("(field \"{}\"", f.name));
                        self.indent();
                        self.ve(&f.value);
                        self.dedent();
                        self.write_line(")");
                    }
                    self.dedent();
                    self.write_line(")");
                }
                self.dedent();
                self.write_line(")");
            }
            Expr::Lambda {
                params,
                body,
                is_async,
                return_type,
            } => {
                self.write_line(&format!("(lambda (is_async {})", is_async));
                self.indent();
                self.print_params(params);
                self.print_return_type(return_type);
                match body {
                    LambdaBody::Block(b) => {
                        self.write_line("(body_block");
                        self.indent();
                        self.ve(b);
                        self.dedent();
                        self.write_line(")");
                    }
                    LambdaBody::Expression(b) => {
                        self.write_line("(body_expr");
                        self.indent();
                        self.ve(b);
                        self.dedent();
                        self.write_line(")");
                    }
                }
                self.dedent();
                self.write_line(")");
            }
            Expr::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.write_line("(if");
                self.indent();
                self.write_line("(cond");
                self.indent();
                self.ve(cond);
                self.dedent();
                self.write_line(")");
                self.write_line("(then");
                self.indent();
                self.ve(then_branch);
                self.dedent();
                self.write_line(")");
                match else_branch {
                    Some(e) => {
                        self.write_line("(else");
                        self.indent();
                        self.ve(e);
                        self.dedent();
                        self.write_line(")");
                    }
                    None => self.write_line("(else (none))"),
                }
                self.dedent();
                self.write_line(")");
            }
            Expr::Block { stmts, trailing } => {
                self.write_line("(block");
                self.indent();
                if stmts.is_empty() {
                    self.write_line("(stmts ())");
                } else {
                    self.write_line("(stmts");
                    self.indent();
                    for s in stmts {
                        self.vs(s);
                    }
                    self.dedent();
                    self.write_line(")");
                }
                match trailing {
                    Some(e) => {
                        self.write_line("(trailing");
                        self.indent();
                        self.ve(e);
                        self.dedent();
                        self.write_line(")");
                    }
                    None => self.write_line("(trailing (none))"),
                }
                self.dedent();
                self.write_line(")");
            }
            Expr::Match { scrutinee, arms } => {
                self.write_line("(match");
                self.indent();
                self.write_line("(scrutinee");
                self.indent();
                self.ve(scrutinee);
                self.dedent();
                self.write_line(")");
                if arms.is_empty() {
                    self.write_line("(arms ())");
                } else {
                    self.write_line("(arms");
                    self.indent();
                    for arm in arms {
                        self.write_line("(arm");
                        self.indent();
                        self.write_line("(pattern");
                        self.indent();
                        self.vp(&arm.pattern);
                        self.dedent();
                        self.write_line(")");
                        match &arm.guard {
                            Some(g) => {
                                self.write_line("(guard");
                                self.indent();
                                self.ve(g);
                                self.dedent();
                                self.write_line(")");
                            }
                            None => self.write_line("(guard (none))"),
                        }
                        self.write_line("(body");
                        self.indent();
                        self.ve(&arm.body);
                        self.dedent();
                        self.write_line(")");
                        self.dedent();
                        self.write_line(")");
                    }
                    self.dedent();
                    self.write_line(")");
                }
                self.dedent();
                self.write_line(")");
            }
            Expr::Atomic(inner) => {
                self.write_line("(atomic");
                self.indent();
                self.ve(inner);
                self.dedent();
                self.write_line(")");
            }
            Expr::Lazy(inner) => {
                self.write_line("(lazy");
                self.indent();
                self.ve(inner);
                self.dedent();
                self.write_line(")");
            }
            Expr::Select(arms) => {
                self.write_line("(select");
                self.indent();
                if arms.is_empty() {
                    self.write_line("(arms ())");
                } else {
                    self.write_line("(arms");
                    self.indent();
                    for arm in arms {
                        match arm {
                            SelectArm::Receive {
                                channel_expr,
                                binding,
                                body,
                            } => {
                                self.write_line("(receive");
                                self.indent();
                                self.write_line("(channel");
                                self.indent();
                                self.ve(channel_expr);
                                self.dedent();
                                self.write_line(")");
                                match binding {
                                    Some(name) => {
                                        self.write_line(&format!("(binding \"{}\")", name));
                                    }
                                    None => self.write_line("(binding (none))"),
                                }
                                self.write_line("(body");
                                self.indent();
                                self.ve(body);
                                self.dedent();
                                self.write_line(")");
                                self.dedent();
                                self.write_line(")");
                            }
                            SelectArm::Timeout { duration, body } => {
                                self.write_line("(timeout");
                                self.indent();
                                self.write_line("(duration");
                                self.indent();
                                self.ve(duration);
                                self.dedent();
                                self.write_line(")");
                                self.write_line("(body");
                                self.indent();
                                self.ve(body);
                                self.dedent();
                                self.write_line(")");
                                self.dedent();
                                self.write_line(")");
                            }
                        }
                    }
                    self.dedent();
                    self.write_line(")");
                }
                self.dedent();
                self.write_line(")");
            }
            Expr::InlineTrait(methods) => {
                self.write_line("(inline_trait");
                self.indent();
                self.print_methods(methods);
                self.dedent();
                self.write_line(")");
            }
        }
    }

    // --- Statements ---

    fn visit_stmt(&mut self, stmt: StmtId) {
        match &self.arena.stmt(stmt).node {
            Stmt::ValDecl {
                name,
                type_annotation,
                value,
                visibility,
            } => {
                self.write_line(&format!("(val_decl \"{}\"", name));
                self.indent();
                self.print_visibility(*visibility);
                match type_annotation {
                    Some(ty) => {
                        self.write_line("(type");
                        self.indent();
                        self.vt(ty);
                        self.dedent();
                        self.write_line(")");
                    }
                    None => self.write_line("(type (none))"),
                }
                self.write_line("(value");
                self.indent();
                self.ve(value);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            Stmt::VarDecl {
                name,
                type_annotation,
                value,
                visibility,
            } => {
                self.write_line(&format!("(var_decl \"{}\"", name));
                self.indent();
                self.print_visibility(*visibility);
                match type_annotation {
                    Some(ty) => {
                        self.write_line("(type");
                        self.indent();
                        self.vt(ty);
                        self.dedent();
                        self.write_line(")");
                    }
                    None => self.write_line("(type (none))"),
                }
                self.write_line("(value");
                self.indent();
                self.ve(value);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            Stmt::Assignment { target, value } => {
                self.write_line("(assignment");
                self.indent();
                self.write_line("(target");
                self.indent();
                self.ve(target);
                self.dedent();
                self.write_line(")");
                self.write_line("(value");
                self.indent();
                self.ve(value);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            Stmt::FieldAssignment {
                object,
                field,
                value,
            } => {
                self.write_line(&format!("(field_assignment \"{}\"", field));
                self.indent();
                self.write_line("(object");
                self.indent();
                self.ve(object);
                self.dedent();
                self.write_line(")");
                self.write_line("(value");
                self.indent();
                self.ve(value);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            Stmt::CompoundAssignment { target, op, value } => {
                self.write_line("(compound_assignment");
                self.indent();
                self.print_compound_assign_op(*op);
                self.write_line("(target");
                self.indent();
                self.ve(target);
                self.dedent();
                self.write_line(")");
                self.write_line("(value");
                self.indent();
                self.ve(value);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            Stmt::Expression { expr } => {
                self.write_line("(expression_stmt");
                self.indent();
                self.ve(expr);
                self.dedent();
                self.write_line(")");
            }
            Stmt::Return { value } => match value {
                Some(e) => {
                    self.write_line("(return");
                    self.indent();
                    self.ve(e);
                    self.dedent();
                    self.write_line(")");
                }
                None => self.write_line("(return (none))"),
            },
            Stmt::Defer { expr } => {
                self.write_line("(defer");
                self.indent();
                self.ve(expr);
                self.dedent();
                self.write_line(")");
            }
            Stmt::Throw { expr } => {
                self.write_line("(throw");
                self.indent();
                self.ve(expr);
                self.dedent();
                self.write_line(")");
            }
            Stmt::Break => self.write_line("(break)"),
            Stmt::Continue => self.write_line("(continue)"),
            Stmt::For {
                name,
                iterable,
                body,
            } => {
                self.write_line(&format!("(for \"{}\"", name));
                self.indent();
                self.write_line("(iterable");
                self.indent();
                self.ve(iterable);
                self.dedent();
                self.write_line(")");
                self.write_line("(body");
                self.indent();
                self.ve(body);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            Stmt::While { condition, body } => {
                self.write_line("(while");
                self.indent();
                self.write_line("(cond");
                self.indent();
                self.ve(condition);
                self.dedent();
                self.write_line(")");
                self.write_line("(body");
                self.indent();
                self.ve(body);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            Stmt::Loop { body } => {
                self.write_line("(loop");
                self.indent();
                self.write_line("(body");
                self.indent();
                self.ve(body);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            Stmt::LocalDecl { decl } => {
                self.write_line("(local-decl");
                self.indent();
                match decl.as_ref() {
                    Decl::FunDecl { name, body, .. } => {
                        self.write_line(&format!("(fun {})", name));
                        self.ve(body);
                    }
                    Decl::TypeDecl { name, .. } => {
                        self.write_line(&format!("(type {})", name));
                    }
                    Decl::TraitDecl { name, .. } => {
                        self.write_line(&format!("(trait {})", name));
                    }
                    _ => self.write_line("(unknown)"),
                }
                self.dedent();
                self.write_line(")");
            }
        }
    }

    // --- Patterns ---

    fn visit_pattern(&mut self, pat: PatternId) {
        match &self.arena.pattern(pat).node {
            Pattern::Wildcard => self.write_line("(wildcard)"),
            Pattern::Literal(lit) => {
                self.write_line("(pattern_literal");
                self.indent();
                self.print_pattern_literal(lit);
                self.dedent();
                self.write_line(")");
            }
            Pattern::Variable { name } => {
                self.write_line(&format!("(pattern_var \"{}\")", name));
            }
            Pattern::Constructor { name, patterns } => {
                self.write_line(&format!("(pattern_constructor \"{}\"", name));
                self.indent();
                if patterns.is_empty() {
                    self.write_line("(patterns ())");
                } else {
                    self.write_line("(patterns");
                    self.indent();
                    for p in patterns {
                        self.vp(p);
                    }
                    self.dedent();
                    self.write_line(")");
                }
                self.dedent();
                self.write_line(")");
            }
            Pattern::Record { fields } => {
                self.write_line("(pattern_record");
                self.indent();
                if fields.is_empty() {
                    self.write_line("(fields ())");
                } else {
                    self.write_line("(fields");
                    self.indent();
                    for f in fields {
                        self.write_line(&format!("(field \"{}\"", f.name));
                        self.indent();
                        self.vp(&f.pattern);
                        self.dedent();
                        self.write_line(")");
                    }
                    self.dedent();
                    self.write_line(")");
                }
                self.dedent();
                self.write_line(")");
            }
            Pattern::OrPattern { left, right } => {
                self.write_line("(or_pattern");
                self.indent();
                self.write_line("(left");
                self.indent();
                self.vp(left);
                self.dedent();
                self.write_line(")");
                self.write_line("(right");
                self.indent();
                self.vp(right);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
            Pattern::Guard { pattern, condition } => {
                self.write_line("(guard_pattern");
                self.indent();
                self.write_line("(pattern");
                self.indent();
                self.vp(pattern);
                self.dedent();
                self.write_line(")");
                self.write_line("(condition");
                self.indent();
                self.ve(condition);
                self.dedent();
                self.write_line(")");
                self.dedent();
                self.write_line(")");
            }
        }
    }
}
impl<'a> Printer<'a> {
    // --- Methods ---

    fn print_methods(&mut self, methods: &'a [MethodDecl<'a>]) {
        if methods.is_empty() {
            self.write_line("(methods ())");
            return;
        }
        self.write_line("(methods");
        self.indent();
        for m in methods {
            self.print_method(m);
        }
        self.dedent();
        self.write_line(")");
    }

    fn print_method(&mut self, m: &'a MethodDecl<'a>) {
        self.write_line(&format!("(method \"{}\"", m.name));
        self.indent();
        self.print_visibility(m.visibility);
        self.write_line(&format!("(is_async {})(is_override {})", m.is_async, m.is_override));
        self.print_type_params(&m.type_params);
        self.print_params(&m.params);
        self.print_return_type(&m.return_type);
        if let Some(delegate) = &m.delegate {
            self.write_line(&format!(
                "(delegate (trait \"{}\") (method \"{}\"))",
                delegate.trait_name, delegate.method_name
            ));
        } else {
            self.write_line("(delegate (none))");
        }
        if let Some(body) = &m.body {
            self.write_line("(body");
            self.indent();
            self.ve(body);
            self.dedent();
            self.write_line(")");
        } else {
            self.write_line("(body (none))");
        }
        self.dedent();
        self.write_line(")");
    }

    fn print_associated_types(&mut self, assoc: &'a [AssociatedType<'a>]) {
        if assoc.is_empty() {
            self.write_line("(associated_types ())");
            return;
        }
        self.write_line("(associated_types");
        self.indent();
        for at in assoc {
            self.write_line(&format!("(associated_type \"{}\"", at.name));
            self.indent();
            match &at.kind {
                Some(k) => {
                    self.write_line("(kind");
                    self.indent();
                    self.visit_kind(k);
                    self.dedent();
                    self.write_line(")");
                }
                None => self.write_line("(kind (none))"),
            }
            self.dedent();
            self.write_line(")");
        }
        self.dedent();
        self.write_line(")");
    }

    fn print_type_constraints(&mut self, constraints: &[TypeConstraint<'_>]) {
        if constraints.is_empty() {
            self.write_line("(type_constraints ())");
            return;
        }
        self.write_line("(type_constraints");
        self.indent();
        for c in constraints {
            self.write_line(&format!("(constraint \"{}\"", c.type_param));
            self.indent();
            self.vt(&c.concrete_type);
            self.dedent();
            self.write_line(")");
        }
        self.dedent();
        self.write_line(")");
    }

    // --- Visibility/parameters/bounds ---

    fn print_visibility(&mut self, vis: Visibility) {
        match vis {
            Visibility::Private => self.write_line("(visibility private)"),
            Visibility::Public => self.write_line("(visibility public)"),
        }
    }

    fn print_type_params(&mut self, params: &'a [TypeParam<'a>]) {
        if params.is_empty() {
            self.write_line("(type_params ())");
            return;
        }
        self.write_line("(type_params");
        self.indent();
        for tp in params {
            self.write_line(&format!("(type_param \"{}\"", tp.name));
            self.indent();
            match &tp.kind {
                Some(k) => {
                    self.write_line("(kind");
                    self.indent();
                    self.visit_kind(k);
                    self.dedent();
                    self.write_line(")");
                }
                None => self.write_line("(kind (none))"),
            }
            self.print_bounds(&tp.bounds);
            self.dedent();
            self.write_line(")");
        }
        self.dedent();
        self.write_line(")");
    }

    fn print_params(&mut self, params: &[Param<'_>]) {
        if params.is_empty() {
            self.write_line("(params ())");
            return;
        }
        self.write_line("(params");
        self.indent();
        for p in params {
            self.write_line(&format!("(param \"{}\"", p.name));
            self.indent();
            match &p.type_annotation {
                Some(ty) => {
                    self.write_line("(type");
                    self.indent();
                    self.vt(ty);
                    self.dedent();
                    self.write_line(")");
                }
                None => self.write_line("(type (none))"),
            }
            self.dedent();
            self.write_line(")");
        }
        self.dedent();
        self.write_line(")");
    }

    fn print_bounds(&mut self, bounds: &[TraitBound<'_>]) {
        if bounds.is_empty() {
            self.write_line("(bounds ())");
            return;
        }
        self.write_line("(bounds");
        self.indent();
        for b in bounds {
            self.write_line(&format!("(trait_bound \"{}\"", b.trait_name));
            self.indent();
            self.print_type_list("type_args", &b.type_args);
            self.dedent();
            self.write_line(")");
        }
        self.dedent();
        self.write_line(")");
    }

    fn print_return_type(&mut self, rt: &Option<TypeRef>) {
        match rt {
            Some(ty) => {
                self.write_line("(return_type");
                self.indent();
                self.vt(ty);
                self.dedent();
                self.write_line(")");
            }
            None => self.write_line("(return_type (none))"),
        }
    }
    fn print_pattern_literal(&mut self, lit: &PatternLiteral<'_>) {
        match lit {
            PatternLiteral::Int(s) => self.write_line(&format!("(int \"{}\")", s)),
            PatternLiteral::Float(s) => self.write_line(&format!("(float \"{}\")", s)),
            PatternLiteral::Bool(b) => self.write_line(&format!("(bool {})", b)),
            PatternLiteral::Char(c) => self.write_line(&format!("(char {})", c)),
            PatternLiteral::String(s) => {
                self.write_line(&format!("(string \"{}\")", escape_str(s)));
            }
            PatternLiteral::Null => self.write_line("(null)"),
        }
    }

    // --- Operator printing ---

    impl_print_op!(print_binary_op, BinaryOp, binary_op_str);
    impl_print_op!(print_unary_op, UnaryOp, unary_op_str);
    impl_print_op!(print_compound_assign_op, CompoundAssignOp, compound_assign_op_str);

    // --- List/option helpers ---

    impl_print_list!(print_expr_list, ExprRef, visit_expr);
    impl_print_list!(print_type_list, TypeRef, visit_type);

    fn print_type_args_option(&mut self, type_args: &Option<Vec<TypeRef>>) {
        match type_args {
            Some(args) if !args.is_empty() => self.print_type_list("type_args", args),
            _ => self.write_line("(type_args ())"),
        }
    }
}

// --- Operator string mappings ---

fn binary_op_str(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
        BinaryOp::Div => "div",
        BinaryOp::Mod => "mod",
        BinaryOp::Eq => "eq",
        BinaryOp::NotEq => "neq",
        BinaryOp::RefEq => "ref_eq",
        BinaryOp::RefNeq => "ref_neq",
        BinaryOp::Lt => "lt",
        BinaryOp::Gt => "gt",
        BinaryOp::LtEq => "lt_eq",
        BinaryOp::GtEq => "gt_eq",
        BinaryOp::And => "and",
        BinaryOp::Or => "or",
        BinaryOp::BitAnd => "bit_and",
        BinaryOp::BitOr => "bit_or",
        BinaryOp::BitXor => "bit_xor",
        BinaryOp::Shl => "shl",
        BinaryOp::Shr => "shr",
        BinaryOp::ConcatList => "concat_list",
        BinaryOp::Range => "range",
        BinaryOp::RangeInclusive => "range_inclusive",
        BinaryOp::Elvis => "elvis",
    }
}

fn unary_op_str(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Not => "not",
        UnaryOp::Neg => "neg",
        UnaryOp::BitNot => "bit_not",
    }
}

fn compound_assign_op_str(op: CompoundAssignOp) -> &'static str {
    match op {
        CompoundAssignOp::AddAssign => "add_assign",
        CompoundAssignOp::SubAssign => "sub_assign",
        CompoundAssignOp::MulAssign => "mul_assign",
        CompoundAssignOp::DivAssign => "div_assign",
        CompoundAssignOp::ModAssign => "mod_assign",
        CompoundAssignOp::BitAndAssign => "bit_and_assign",
        CompoundAssignOp::BitOrAssign => "bit_or_assign",
        CompoundAssignOp::BitXorAssign => "bit_xor_assign",
        CompoundAssignOp::ShlAssign => "shl_assign",
        CompoundAssignOp::ShrAssign => "shr_assign",
    }
}

/// Escapes special characters in a string for printing quoted literals.
fn escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{{{:x}}}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}