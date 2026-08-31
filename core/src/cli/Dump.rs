//! dump — canonical sema dump (differential oracle for the bootstrap ladder).
//!
//! `frond debug --stage sema <file>` prints a CANONICAL, byte-stable summary
//! of the semantic-analysis results. This is the Stage-1 oracle contract for
//! frondc (the self-hosted compiler written in Frond): on the same input, the
//! Frond-side compiler must reproduce this output byte-for-byte.
//!
//! Contract rules:
//! - The first line pins the format version (`sema-dump vN`); any format
//!   change bumps N.
//! - Every table is printed in a deterministic order, sorted by its stable
//!   keys (names / ids). Tables keyed by AST handle ADDRESSES (expr_types,
//!   resolved_types, call_instantiations, field_accesses, method_dispatches,
//!   and the per-instance tables inside MonomorphInstance) are unstable
//!   across runs and are NOT dumped — only their sizes appear in the stats
//!   block.
//! - On sema error the shared pipeline exits before this dump runs;
//!   reject-case differential testing uses the exit code + stderr messages.

use std::fmt::Write as _;

use crate::sema::Sema::{
    InheritedMethodInstance, SemaResult, TraitDefaultInstance, TypeRepr,
};
use crate::tooling::Common::Pipeline as CommonPipeline;
use crate::types::{TypeArena, TypeHandle};

/// A std file passed directly as the debug entry is classified by its std
/// key instead of its disk path, so stdlib-only attributes (@internal etc.)
/// stay legal in it. The key is only granted when the entry's bytes match
/// the embedded copy — a file that merely lives in a directory called
/// `std/` stays a user entry. Mirrors frondc's Main.dump_sema_entry rule.
fn canonical_std_entry_name(entry_path: &str, source: &str) -> String {
    let norm = entry_path.replace('\\', "/");
    if let Some(idx) = norm.rfind("/std/") {
        let rel = &norm[idx + 5..];
        let key = if rel.starts_with("builtin/") {
            rel.to_string()
        } else {
            format!("std/{rel}")
        };
        for (p, content) in crate::module::STD_FILES {
            if *p == key && *content == source {
                return key;
            }
        }
    }
    norm
}

/// Entry point: parse + load + sema the given entry path, then print the dump.
pub fn dump_sema(entry_path: &str) {
    let source = super::Pipeline::read_source(entry_path);
    let canonical = canonical_std_entry_name(entry_path, &source);
    let arena = bumpalo::Bump::new();
    let entry_module = CommonPipeline::parse_entry_module_or_exit(&arena, &source, &canonical);
    let (loader, std_keys, dep_keys) =
        CommonPipeline::load_all_modules_or_exit(&entry_module, entry_path);
    let (type_arena, sema) = CommonPipeline::run_sema_pipeline_or_exit(
        &loader,
        &std_keys,
        &dep_keys,
        &entry_module,
        entry_path,
    );
    let mut out = String::new();
    render(&mut out, &type_arena, &sema);
    print!("{}", out);
}

fn render(out: &mut String, ta: &TypeArena, s: &SemaResult) {
    out.push_str("sema-dump v1\n");

    // ── modules ────────────────────────────────────────────────────────
    let mut mods: Vec<&str> = s.user_module_paths.iter().map(|p| p.as_str()).collect();
    mods.sort_unstable();
    let _ = writeln!(out, "! modules {}", mods.len());
    for m in mods {
        let _ = writeln!(out, "mod {}", m);
    }

    // ── types ──────────────────────────────────────────────────────────
    let mut type_names: Vec<&str> = s.type_def_index.keys().map(|k| s.symbols.resolve(*k)).collect();
    type_names.sort_unstable();
    let _ = writeln!(out, "! types {}", type_names.len());
    for name in type_names {
        let idx = s.type_def_idx(name)
            .expect("name listed from type_def_index keys");
        let td = &s.type_defs[&idx];
        let _ = writeln!(
            out,
            "type {} kind={:?} params=[{}] bases=[{}]",
            name,
            td.kind,
            td.type_params.iter().map(|t| t.as_ref()).collect::<Vec<_>>().join(","),
            td.bases.iter().map(|b| b.as_ref()).collect::<Vec<_>>().join(","),
        );
        if let Some(target) = td.target_type_name.as_deref() {
            let rendered = td.target_type.map(|h| ty_str(ta, h)).unwrap_or_else(|| "-".into());
            let _ = writeln!(out, "  alias-target {} : {}", target, rendered);
        }
        // Constructors: records hold their fields in constructors[0]; ADTs
        // list one entry per constructor; aliases have none.
        for ctor in td.constructors.iter() {
            let fields = ctor
                .field_names
                .iter()
                .zip(ctor.field_types.iter())
                .enumerate()
                .map(|(i, (n, &h))| {
                    let name = n.as_deref().map(|s| s.to_string()).unwrap_or_else(|| format!("#{}", i));
                    format!("{}: {}", name, ty_str(ta, h))
                })
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(out, "  ctor {} ({})", ctor.name, fields);
        }
        for (i, m) in td.methods.iter().enumerate() {
            let params = m
                .param_type_reprs
                .iter()
                .map(repr_str)
                .collect::<Vec<_>>()
                .join(",");
            let ret = m.return_type_repr.as_ref().map(repr_str).unwrap_or_else(|| "-".into());
            let _ = write!(
                out,
                "  method {} {} pub={} body={} async={} throws={} retref={} params=[{}] ret={}",
                i, m.name, m.is_pub, m.has_body, m.is_async, m.is_throwing, m.return_is_ref, params, ret,
            );
            if let Some(k) = m.intrinsic.as_ref() {
                let _ = write!(out, " intrinsic={:?}", k);
            }
            if let Some(d) = m.delegate_trait.as_deref() {
                let _ = write!(out, " delegate={}", d);
            }
            out.push('\n');
        }
    }

    // ── traits ─────────────────────────────────────────────────────────
    let mut trait_names: Vec<&str> = s.trait_def_index.keys().map(|k| s.symbols.resolve(*k)).collect();
    trait_names.sort_unstable();
    let _ = writeln!(out, "! traits {}", trait_names.len());
    for name in trait_names {
        let idx = s.trait_def_idx(name)
            .expect("name listed from trait_def_index keys");
        let td = &s.trait_defs[&idx];
        let _ = writeln!(
            out,
            "trait {} parents=[{}]",
            name,
            td.parents.iter().map(|p| p.as_ref()).collect::<Vec<_>>().join(","),
        );
        for m in td.methods.iter() {
            let _ = writeln!(
                out,
                "  method {} params={} ret={} async={} body={}",
                m.name, m.param_count, ty_str(ta, m.return_type), m.is_async, m.has_body,
            );
        }
    }

    // ── funcs ──────────────────────────────────────────────────────────
    // Keys are "module\0name"; sort raw keys so (module, name) order is stable.
    let mut func_keys: Vec<&str> = s.func_sig_index.keys().map(|k| s.symbols.resolve(*k)).collect();
    func_keys.sort_unstable();
    let _ = writeln!(out, "! funcs {}", func_keys.len());
    for key in func_keys {
        let idx = s.func_sig_idx(key)
            .expect("key listed from func_sig_index keys");
        let f = &s.func_sigs[&idx];
        let (module, name) = match key.split_once('\x00') {
            Some((m, n)) => (m, n),
            None => ("-", key),
        };
        let _ = writeln!(
            out,
            "func {}::{} params=[{}] ret={} async={} throws={} refs=[{}] retref={}",
            module,
            name,
            f.type_params.iter().map(|t| t.as_ref()).collect::<Vec<_>>().join(","),
            ty_str(ta, f.return_type),
            f.is_async,
            f.is_throwing,
            f.param_is_ref.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(","),
            f.return_is_ref,
        );
    }

    // ── monomorph instances ────────────────────────────────────────────
    let mut monos: Vec<&crate::sema::Sema::MonomorphInstance> =
        s.monomorph_instances.iter().collect();
    monos.sort_by_key(|m| m.instance_id);
    let _ = writeln!(out, "! monomorph {}", monos.len());
    for m in monos {
        let args = m.type_args.iter().map(|&h| ty_str(ta, h)).collect::<Vec<_>>().join(",");
        let _ = writeln!(
            out,
            "mono {} {}::{} args=[{}] ret={} async={}",
            m.instance_id, m.module_name, m.func_name, args, ty_str(ta, m.return_type), m.is_async,
        );
    }

    // ── trait default instances ────────────────────────────────────────
    let mut tdefs: Vec<&TraitDefaultInstance> = s.trait_default_instances.iter().collect();
    tdefs.sort_by(|a, b| {
        (a.type_name.as_ref(), a.trait_name.as_ref(), a.method_idx)
            .cmp(&(b.type_name.as_ref(), b.trait_name.as_ref(), b.method_idx))
    });
    let _ = writeln!(out, "! trait-defaults {}", tdefs.len());
    for t in tdefs {
        let _ = writeln!(out, "tdef {} : {}.{}", t.type_name, t.trait_name, t.method_idx);
    }

    // ── inherited-method instances ─────────────────────────────────────
    let mut inhs: Vec<&InheritedMethodInstance> = s.inherited_method_instances.iter().collect();
    inhs.sort_by(|a, b| {
        (a.type_name.as_ref(), a.method_idx).cmp(&(b.type_name.as_ref(), b.method_idx))
    });
    let _ = writeln!(out, "! inherited {}", inhs.len());
    for i in inhs {
        let _ = writeln!(
            out,
            "inh {}#{} <- {}::{}#{}",
            i.type_name, i.method_idx, i.base_module, i.base_type_name, i.base_method_idx,
        );
    }

    // ── witness table ──────────────────────────────────────────────────
    let mut wits: Vec<_> = s.witness_table.entries().collect();
    wits.sort_by(|a, b| {
        (a.trait_name.as_ref(), a.type_name.as_ref())
            .cmp(&(b.trait_name.as_ref(), b.type_name.as_ref()))
    });
    let _ = writeln!(out, "! witness {}", wits.len());
    for w in wits {
        let mut slots: Vec<(&str, u16)> =
            w.method_slots.iter().map(|(k, v)| (k.as_ref(), *v)).collect();
        slots.sort_unstable();
        let slots_str = slots
            .iter()
            .map(|(n, v)| format!("{}=#{}", n, v))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(out, "wit {} for {} : {}", w.trait_name, w.type_name, slots_str);
    }

    // ── field ids ──────────────────────────────────────────────────────
    let mut fids: Vec<&str> = s.field_id_map.keys().map(|k| s.symbols.resolve(*k)).collect();
    fids.sort_unstable();
    let _ = writeln!(out, "! field-ids {}", fids.len());
    for key in fids {
        let display = key.replace('\x00', ".");
        let fid = s.symbols.find(key).and_then(|k| s.field_id_map.get(&k).copied())
            .expect("key listed from field_id_map keys");
        let _ = writeln!(out, "fid {} = {}", display, fid);
    }

    // ── diagnostics ────────────────────────────────────────────────────
    let mut errors: Vec<&crate::sema::Sema::SemaError> = s.errors.iter().collect();
    errors.sort_by(|a, b| diag_key(a).cmp(&diag_key(b)));
    let _ = writeln!(out, "! errors {}", errors.len());
    for e in errors {
        let _ = writeln!(out, "err {}", diag_line(e));
    }
    let mut warnings: Vec<&crate::sema::Sema::SemaError> = s.warnings.iter().collect();
    warnings.sort_by(|a, b| diag_key(a).cmp(&diag_key(b)));
    let _ = writeln!(out, "! warnings {}", warnings.len());
    for w in warnings {
        let _ = writeln!(out, "warn {}", diag_line(w));
    }

    // ── stats (address-keyed tables: sizes only) ───────────────────────
    let mono_local_exprs: usize = s.monomorph_instances.iter().map(|m| m.expr_types.len()).sum();
    let _ = writeln!(
        out,
        "! stats\nexpr_types={} resolved_types={} call_instantiations={} field_accesses={} method_dispatches={} mono_local_expr_types={} coroutines={}",
        s.expr_types.len(),
        s.resolved_types.len(),
        s.call_instantiations.len(),
        s.field_accesses.len(),
        s.method_dispatches.len(),
        mono_local_exprs,
        s.coroutine_metas.len(),
    );
}

fn ty_str(ta: &TypeArena, h: TypeHandle) -> String {
    format!("{}", ta.display(h))
}

/// 1C oracle: canonical module-load dump (Root.toml manifest + import graph).
///
/// `frond debug --stage load <entry>` prints a CANONICAL, byte-stable summary
/// of module loading for the entry file. The Frond-side frondc `loaddeps`
/// command must reproduce this output byte-for-byte (BOOTSTRAP_PLAN 1C).
///
/// Contract rules (mirror the sema-dump contract):
/// - The first line pins the format version (`load-dump vN`); any format
///   change bumps N.
/// - The manifest is located by climbing from the ENTRY FILE's directory
///   (not the CWD) so both compilers see the same input regardless of where
///   they are invoked from.
/// - dep order = `load_transitive_imports` post-order (deterministic).
/// - modules = all loaded cache keys as logical paths (`a/b/c.frond` →
///   `a.b.c`), sorted byte-wise.
/// - errors in occurrence order. Parse-error MESSAGE text inside a module is
///   Rust-side wording and not yet aligned (1B deferred the same); the 1C
///   corpus avoids module parse failures.
/// 1D oracle: canonical type-arena operations dump.
///
/// `frond debug --stage tyops <any-file>` runs a FIXED battery of arena
/// operations (construct / unify / occurs / resolve / kind / display) and
/// prints one line per operation plus a stats block. The Frond-side frondc
/// `tyops` command must reproduce this output byte-for-byte (BOOTSTRAP_PLAN
/// 1D). Determinism: the battery is purely sequential (no hashing, no
/// address-keyed structures), so TypeHandle ids align across both sides.
pub fn dump_tyops() {
    use crate::types::{FieldType, SemKind, TraitMethodSig, Type, TypeArena, UnifyError};
    use std::fmt::Write as _;

    fn ue(e: &UnifyError) -> &'static str {
        match e {
            UnifyError::TypeMismatch => "type mismatch",
            UnifyError::OccursCheckFailed => "occurs check failed (recursive type)",
        }
    }

    let mut out = String::new();
    let mut n: u32 = 0;
    let mut a = TypeArena::new();
    macro_rules! line {
        ($($t:tt)*) => {{
            n += 1;
            let _ = writeln!(out, "{} {}", n, format!($($t)*));
        }};
    }

    // §1 scalar construction: name + display round trip (from_type_name).
    for name in [
        "bool", "char", "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128",
        "isize", "usize", "f16", "f32", "f64", "f128", "str", "null", "void", "Lib",
    ] {
        let h = a.from_scalar_name(name);
        line!("scalar {} -> {} {}", name, a.display(h), h.0);
    }
    let unknown = a.from_scalar_name("not_a_type");
    line!("scalar_unknown -> {}", a.display(unknown));

    // §2 type vars: fresh + display + unify + resolve (with path compression).
    let v0 = a.fresh_type_var();
    let i32h = a.from_scalar_name("i32");
    line!("var fresh v0 -> {}", a.display(v0));
    match a.unify(v0, i32h) {
        Ok(()) => line!("unify v0 i32 -> ok resolve={}", a.display(a.resolve(v0))),
        Err(e) => line!("unify v0 i32 -> err {}", ue(&e)),
    }
    let v1 = a.fresh_type_var();
    match a.unify(v1, v0) {
        Ok(()) => line!("unify v1 v0 -> ok resolve={}", a.display(a.resolve(v1))),
        Err(e) => line!("unify v1 v0 -> err {}", ue(&e)),
    }

    // §3 occurs check: v ~ array(v) must fail.
    let v2 = a.fresh_type_var();
    let arr2 = a.make_array(v2, None);
    match a.unify(v2, arr2) {
        Ok(()) => line!("unify v2 arr(v2) -> ok"),
        Err(e) => line!("unify v2 arr(v2) -> err {}", ue(&e)),
    }

    // §4 rigid var rejects binding.
    let r0 = a.fresh_rigid_var();
    match a.unify(r0, i32h) {
        Ok(()) => line!("unify rigid i32 -> ok"),
        Err(e) => line!("unify rigid i32 -> err {}", ue(&e)),
    }

    // §5 never / unknown absorb (in-place slot overwrite — display the ORIGINAL handle).
    let never = a.make(Type::Never);
    match a.unify(never, i32h) {
        Ok(()) => line!("unify never i32 -> ok display(never-slot)={}", a.display(never)),
        Err(e) => line!("unify never i32 -> err {}", ue(&e)),
    }
    let uk2 = a.make(Type::Unknown);
    match a.unify(i32h, uk2) {
        Ok(()) => line!("unify i32 unknown -> ok display(unknown-slot)={}", a.display(uk2)),
        Err(e) => line!("unify i32 unknown -> err {}", ue(&e)),
    }

    // §6 scalar unify: same ok / different mismatch.
    let i64h = a.from_scalar_name("i64");
    match a.unify(i32h, i64h) {
        Ok(()) => line!("unify i32 i64 -> ok"),
        Err(e) => line!("unify i32 i64 -> err {}", ue(&e)),
    }
    let i32b = a.from_scalar_name("i32");
    match a.unify(i32h, i32b) {
        Ok(()) => line!("unify i32 i32 -> ok"),
        Err(e) => line!("unify i32 i32 -> err {}", ue(&e)),
    }

    // §7 functions: same ok / arity err / param err / display.
    let strh = a.from_scalar_name("str");
    let f1 = a.make_fn(vec![i32h, i64h].into(), strh);
    let f2 = a.make_fn(vec![i32h, i64h].into(), strh);
    line!("fn display -> {}", a.display(f1));
    match a.unify(f1, f2) {
        Ok(()) => line!("unify fn fn -> ok"),
        Err(e) => line!("unify fn fn -> err {}", ue(&e)),
    }
    let f3 = a.make_fn(vec![i32h].into(), strh);
    match a.unify(f1, f3) {
        Ok(()) => line!("unify fn fn1 -> ok"),
        Err(e) => line!("unify fn fn1 -> err {}", ue(&e)),
    }
    let f4 = a.make_fn(vec![i32h, strh].into(), strh);
    match a.unify(f1, f4) {
        Ok(()) => line!("unify fn fn_param -> ok"),
        Err(e) => line!("unify fn fn_param -> err {}", ue(&e)),
    }

    // §8 records: same ok / field-count err / display (named + positional).
    let rec1 = a.make_record(
        vec![FieldType { name: Some("x".into()), ty: i32h }, FieldType { name: None, ty: strh }].into(),
        Some("Pt".into()),
    );
    let rec2 = a.make_record(
        vec![FieldType { name: Some("x".into()), ty: i32h }, FieldType { name: None, ty: strh }].into(),
        None,
    );
    line!("record display -> {}", a.display(rec1));
    match a.unify(rec1, rec2) {
        Ok(()) => line!("unify rec rec -> ok"),
        Err(e) => line!("unify rec rec -> err {}", ue(&e)),
    }
    let rec3 = a.make_record(vec![FieldType { name: Some("x".into()), ty: i32h }].into(), None);
    match a.unify(rec1, rec3) {
        Ok(()) => line!("unify rec rec1 -> ok"),
        Err(e) => line!("unify rec rec1 -> err {}", ue(&e)),
    }

    // §9 nullable: unify ok + T?? collapse.
    let n1 = a.make_nullable(i32h);
    let n2 = a.make_nullable(i32h);
    match a.unify(n1, n2) {
        Ok(()) => line!("unify nullable -> ok"),
        Err(e) => line!("unify nullable -> err {}", ue(&e)),
    }
    let nn = a.make_nullable(n1);
    line!("nullable collapse -> {} {}", a.display(nn), nn.0);

    // §10 refs: same ok / raw mismatch / display.
    let rf1 = a.make_ref(i32h, false);
    let rf2 = a.make_ref(i32h, false);
    line!("ref display -> {}", a.display(rf1));
    match a.unify(rf1, rf2) {
        Ok(()) => line!("unify ref ref -> ok"),
        Err(e) => line!("unify ref ref -> err {}", ue(&e)),
    }
    let rf3 = a.make_ref(i32h, true);
    match a.unify(rf1, rf3) {
        Ok(()) => line!("unify ref raw -> ok"),
        Err(e) => line!("unify ref raw -> err {}", ue(&e)),
    }

    // §11 adt / generic / trait: name+args unify.
    let adt1 = a.make_adt("Option".into(), vec![i32h].into());
    let adt2 = a.make_adt("Option".into(), vec![i32h].into());
    line!("adt display -> {}", a.display(adt1));
    match a.unify(adt1, adt2) {
        Ok(()) => line!("unify adt adt -> ok"),
        Err(e) => line!("unify adt adt -> err {}", ue(&e)),
    }
    let adt3 = a.make_adt("Optio2".into(), vec![i32h].into());
    match a.unify(adt1, adt3) {
        Ok(()) => line!("unify adt name -> ok"),
        Err(e) => line!("unify adt name -> err {}", ue(&e)),
    }
    let adt4 = a.make_adt("Option".into(), vec![i64h].into());
    match a.unify(adt1, adt4) {
        Ok(()) => line!("unify adt args -> ok"),
        Err(e) => line!("unify adt args -> err {}", ue(&e)),
    }
    let g1 = a.make_generic("List".into(), vec![i32h].into());
    let g2 = a.make_generic("List".into(), vec![i32h].into());
    match a.unify(g1, g2) {
        Ok(()) => line!("unify generic -> ok"),
        Err(e) => line!("unify generic -> err {}", ue(&e)),
    }
    let t1 = a.make_trait("Ord".into(), vec![i32h].into());
    let t2 = a.make_trait("Ord".into(), vec![i32h].into());
    line!("trait display -> {}", a.display(t1));
    match a.unify(t1, t2) {
        Ok(()) => line!("unify trait -> ok"),
        Err(e) => line!("unify trait -> err {}", ue(&e)),
    }

    // §12 arrays: slice + sized display + unify.
    let as1 = a.make_array(i32h, None);
    let as2 = a.make_array(i32h, Some(4));
    line!("array display -> {} {}", a.display(as1), a.display(as2));
    let as3 = a.make_array(i32h, None);
    match a.unify(as1, as3) {
        Ok(()) => line!("unify array -> ok"),
        Err(e) => line!("unify array -> err {}", ue(&e)),
    }

    // §13 throw: both params unify / display.
    let th1 = a.make_throw(i32h, strh);
    line!("throw display -> {}", a.display(th1));
    let th2 = a.make_throw(i32h, strh);
    match a.unify(th1, th2) {
        Ok(()) => line!("unify throw -> ok"),
        Err(e) => line!("unify throw -> err {}", ue(&e)),
    }
    let th3 = a.make_throw(strh, strh);
    match a.unify(th1, th3) {
        Ok(()) => line!("unify throw value -> ok"),
        Err(e) => line!("unify throw value -> err {}", ue(&e)),
    }

    // §14 single-param builtin generics: construct + display + unify.
    let ch1 = a.make_channel(i32h);
    let asy1 = a.make_async(strh);
    let lz1 = a.make_lazy(i32h);
    let at1 = a.make_atomic(i32h);
    let sd1 = a.make_sender(i32h);
    let rc1 = a.make_receiver(i32h);
    let ff1 = a.make_foreign_fn(i32h);
    line!("generics display -> {} {} {} {} {} {} {}", a.display(ch1), a.display(asy1), a.display(lz1), a.display(at1), a.display(sd1), a.display(rc1), a.display(ff1));
    let ch2 = a.make_channel(i32h);
    match a.unify(ch1, ch2) {
        Ok(()) => line!("unify channel -> ok"),
        Err(e) => line!("unify channel -> err {}", ue(&e)),
    }
    let asy2 = a.make_async(strh);
    match a.unify(asy1, asy2) {
        Ok(()) => line!("unify async -> ok"),
        Err(e) => line!("unify async -> err {}", ue(&e)),
    }
    let asy3 = a.make_async(i64h);
    match a.unify(asy1, asy3) {
        Ok(()) => line!("unify async value -> ok"),
        Err(e) => line!("unify async value -> err {}", ue(&e)),
    }
    let lz2 = a.make_lazy(i32h);
    match a.unify(lz1, lz2) {
        Ok(()) => line!("unify lazy -> ok"),
        Err(e) => line!("unify lazy -> err {}", ue(&e)),
    }
    let at2 = a.make_atomic(i32h);
    match a.unify(at1, at2) {
        Ok(()) => line!("unify atomic -> ok"),
        Err(e) => line!("unify atomic -> err {}", ue(&e)),
    }
    let sd2 = a.make_sender(i32h);
    match a.unify(sd1, sd2) {
        Ok(()) => line!("unify sender -> ok"),
        Err(e) => line!("unify sender -> err {}", ue(&e)),
    }
    let rc2 = a.make_receiver(i32h);
    match a.unify(rc1, rc2) {
        Ok(()) => line!("unify receiver -> ok"),
        Err(e) => line!("unify receiver -> err {}", ue(&e)),
    }
    let ff2 = a.make_foreign_fn(i32h);
    match a.unify(ff1, ff2) {
        Ok(()) => line!("unify foreignfn -> ok"),
        Err(e) => line!("unify foreignfn -> err {}", ue(&e)),
    }

    // §15 trait object: display + Trait~TraitObject same-name unify + mismatch.
    let to1 = a.make_trait_object(
        "Ord".into(),
        vec![
            TraitMethodSig { name: "lt".into(), param_count: 1, return_type: i32h, is_async: false, has_body: true },
            TraitMethodSig { name: "cmp".into(), param_count: 2, return_type: i32h, is_async: true, has_body: false },
        ].into(),
    );
    line!("traitobject display -> {}", a.display(to1));
    match a.unify(t1, to1) {
        Ok(()) => line!("unify trait traitobject -> ok"),
        Err(e) => line!("unify trait traitobject -> err {}", ue(&e)),
    }
    let t3 = a.make_trait("Eq".into(), vec![].into());
    match a.unify(t3, to1) {
        Ok(()) => line!("unify trait traitobject_name -> ok"),
        Err(e) => line!("unify trait traitobject_name -> err {}", ue(&e)),
    }

    // §16 module ref display.
    let mr = a.make_module_ref("std.io.File".into(), crate::types::EnvId(0));
    line!("moduleref display -> {}", a.display(mr));

    // §17 kinds: fresh kind var + unify_kind + kind_of + arrow unify + mismatch.
    let kv = a.fresh_kind_var();
    match a.unify_kind(&kv, &SemKind::Star) {
        Ok(()) => line!("unify_kind var star -> ok resolve={:?}", a.resolve_kind(kv)),
        Err(()) => line!("unify_kind var star -> err"),
    }
    let vk = a.fresh_type_var_with_kind(SemKind::Arrow { param: Box::new(SemKind::Star), result: Box::new(SemKind::Star) });
    line!("kind_of arrowvar -> {:?}", a.kind_of(vk));
    let k1 = SemKind::Arrow { param: Box::new(SemKind::Star), result: Box::new(SemKind::Star) };
    let k2 = SemKind::Arrow { param: Box::new(SemKind::Star), result: Box::new(SemKind::Star) };
    match a.unify_kind(&k1, &k2) {
        Ok(()) => line!("unify_kind arrow arrow -> ok"),
        Err(()) => line!("unify_kind arrow arrow -> err"),
    }
    let k3 = SemKind::Star;
    match a.unify_kind(&k1, &k3) {
        Ok(()) => line!("unify_kind arrow star -> ok"),
        Err(()) => line!("unify_kind arrow star -> err"),
    }
    let kv2 = a.fresh_kind_var();
    match a.unify_kind(&kv2, &k1) {
        Ok(()) => line!("unify_kind var arrow -> ok resolve={:?}", a.resolve_kind(kv2)),
        Err(()) => line!("unify_kind var arrow -> err"),
    }

    // §18 var-with-kind unify failure (kind mismatch → TypeMismatch).
    let vstar = a.fresh_type_var();
    let varrow = a.fresh_type_var_with_kind(SemKind::Arrow { param: Box::new(SemKind::Star), result: Box::new(SemKind::Star) });
    match a.unify(vstar, varrow) {
        Ok(()) => line!("unify vstar varrow -> ok"),
        Err(e) => line!("unify vstar varrow -> err {}", ue(&e)),
    }

    // stats
    let _ = writeln!(out, "! stats");
    let _ = writeln!(
        out,
        "ops={} types={} details={} type_vars={}",
        n,
        a.len(),
        a.details_len(),
        a.type_vars_len()
    );
    print!("{}", out);
}

pub fn dump_load(entry_path: &str) {
    use crate::module::{Error::LoadError, ModuleLoader, StdlibEmbed::STD_FILES};

    let mut out = String::new();
    out.push_str("load-dump v1\n");

    // ── manifest (anchored at the entry file's directory chain) ───────
    let entry_dir = std::path::Path::new(entry_path)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    let _ = writeln!(out, "! manifest");
    match super::Manifest::find_project_root_from(&entry_dir) {
        None => {
            let _ = writeln!(out, "root=no");
        }
        Some(root) => {
            let content = std::fs::read_to_string(
                std::path::Path::new(&root).join(super::Manifest::MANIFEST_NAME),
            )
            .unwrap_or_default();
            match toml::from_str::<super::Manifest::Manifest>(&content) {
                Ok(m) => {
                    let _ = writeln!(
                        out,
                        "root=yes name={} entry={} output_dir={} opt_level={}",
                        m.package.name, m.package.entry, m.build.output_dir, m.build.opt_level
                    );
                }
                Err(_) => {
                    let _ = writeln!(out, "root=bad");
                }
            }
        }
    }

    // ── entry parse (fatal: no further sections on failure) ──────────
    let source = super::Pipeline::read_source(entry_path);
    let arena = bumpalo::Bump::new();
    let mut lexer = crate::ast::Parser::Lexer::new(&source);
    let mut sink = crate::ast::Parser::TokenCollector::new();
    lexer.tokenize_into(&mut sink);
    let tokens: Vec<crate::ast::Parser::Token<'_>> = sink.into_tokens();
    let tokens_ref = arena.alloc_slice_copy(&tokens);
    let mut parser =
        crate::ast::Parser::Parser::new(tokens_ref, &arena, crate::ast::Parser::ErrorCollector::new());
    let entry_module = match parser.parse_module(entry_path) {
        Ok(m) => m,
        Err(err) => {
            let _ = writeln!(out, "! fatal\nparsefail {}:{}: {}", err.line, err.column, err.message);
            print!("{}", out);
            return;
        }
    };

    // ── loader: mirror load_all_modules_or_exit WITHOUT the exit ─────
    let mut loader = ModuleLoader::new();
    if let Some(parent) = std::path::Path::new(entry_path).parent() {
        loader.add_search_path(parent);
    }
    let dep_keys = loader.load_transitive_imports(&entry_module);
    for (key, _) in STD_FILES {
        let parts: Vec<&str> = key.strip_suffix(".frond").unwrap().split('/').collect();
        let _ = loader.resolve_and_load(&parts);
    }

    // ── deps (post-order, deterministic) ─────────────────────────────
    let _ = writeln!(out, "! deps {}", dep_keys.len());
    for k in &dep_keys {
        let _ = writeln!(out, "{}", k);
    }

    // ── modules (all loaded keys as logical paths, byte-sorted) ──────
    let mut logical: Vec<String> = loader
        .loaded_keys()
        .into_iter()
        .map(|k| k.strip_suffix(".frond").map(|s| s.replace('/', ".")).unwrap_or(k))
        .collect();
    logical.sort_unstable();
    let _ = writeln!(out, "! modules {}", logical.len());
    for m in &logical {
        let _ = writeln!(out, "{}", m);
    }

    // ── errors (occurrence order) ────────────────────────────────────
    let errs = loader.load_errors();
    let _ = writeln!(out, "! errors {}", errs.len());
    for err in errs {
        match err {
            LoadError::ModuleNotFound { path } => {
                let _ = writeln!(out, "not_found {}", path);
            }
            LoadError::ParseFailed { path, line, column, message } => {
                let _ = writeln!(out, "parsefail {} {}:{}: {}", path, line, column, message);
            }
            LoadError::CircularImport { path } => {
                let _ = writeln!(out, "circular {}", path);
            }
        }
    }

    print!("{}", out);
}

fn repr_str(r: &TypeRepr) -> String {
    match r {
        TypeRepr::Named(n) => n.to_string(),
        TypeRepr::ThisType => "This".to_string(),
        TypeRepr::Generic(n, args) => format!(
            "{}<{}>",
            n,
            args.iter().map(repr_str).collect::<Vec<_>>().join(",")
        ),
        TypeRepr::Nullable(t) => format!("{}?", repr_str(t)),
        TypeRepr::Ref(t) => format!("&{}", repr_str(t)),
        TypeRepr::RawPtr(t) => format!("*{}", repr_str(t)),
        TypeRepr::Function(ps, r) => format!(
            "fn({}) {}",
            ps.iter().map(repr_str).collect::<Vec<_>>().join(","),
            repr_str(r),
        ),
        TypeRepr::Array(t, None) => format!("{}[]", repr_str(t)),
        TypeRepr::Array(t, Some(n)) => format!("{}[{}]", repr_str(t), n),
    }
}

fn diag_key(e: &crate::sema::Sema::SemaError) -> (String, u32, u32, String) {
    (
        e.file_path.as_deref().unwrap_or("-").to_string(),
        e.line,
        e.column,
        e.message.to_string(),
    )
}

fn diag_line(e: &crate::sema::Sema::SemaError) -> String {
    format!(
        "{}:{}:{}: {}",
        e.file_path.as_deref().unwrap_or("-"),
        e.line,
        e.column,
        e.message
    )
}
