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

/// Entry point: parse + load + sema the given entry path, then print the dump.
pub fn dump_sema(entry_path: &str) {
    let source = super::Pipeline::read_source(entry_path);
    let arena = bumpalo::Bump::new();
    let entry_module = CommonPipeline::parse_entry_module_or_exit(&arena, &source, entry_path);
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
    let mut type_names: Vec<&str> = s.type_def_index.keys().map(|k| k.as_str()).collect();
    type_names.sort_unstable();
    let _ = writeln!(out, "! types {}", type_names.len());
    for name in type_names {
        let idx = s.type_def_index[name];
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
    let mut trait_names: Vec<&str> = s.trait_def_index.keys().map(|k| k.as_str()).collect();
    trait_names.sort_unstable();
    let _ = writeln!(out, "! traits {}", trait_names.len());
    for name in trait_names {
        let idx = s.trait_def_index[name];
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
    let mut func_keys: Vec<&str> = s.func_sig_index.keys().map(|k| k.as_str()).collect();
    func_keys.sort_unstable();
    let _ = writeln!(out, "! funcs {}", func_keys.len());
    for key in func_keys {
        let idx = s.func_sig_index[key];
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
    let mut fids: Vec<&str> = s.field_id_map.keys().map(|k| k.as_str()).collect();
    fids.sort_unstable();
    let _ = writeln!(out, "! field-ids {}", fids.len());
    for key in fids {
        let display = key.replace('\x00', ".");
        let _ = writeln!(out, "fid {} = {}", display, s.field_id_map[key]);
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
