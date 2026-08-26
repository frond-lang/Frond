//! inspect subcommand — view .fndo metadata.

use std::process;

pub fn cmd_inspect(file: &str, verbose: bool) {
    if !file.ends_with(".fndo") {
        eprintln!("error: expected .fndo file, got: {}", file);
        process::exit(1);
    }
    match crate::solidify::Format::inspect_solidify_from_file(file) {
        Ok(info) => {
            println!("FNDO File: {}", file);
            println!("  Schema:       v{}", info.schema_version);
            println!("  ABI:          v{}", info.abi_version);
            println!("  Nodes:        {}", info.node_count);
            println!("  Subgraphs:    {} (entry: {})",
                info.subgraph_count,
                info.entry_subgraph.map(|s| format!("#{}", s)).unwrap_or("none".to_string()));
            println!("  Inputs:       {}", info.input_count);
            println!("  Strings:      {} bytes", info.string_pool_len);
            println!("  Global vars:  {}", info.global_var_count);
            println!("  Memo tables:  {}", info.memo_table_count);
            println!("  Compute fns:  {}", info.compute_fn_count);
            println!("  Sections:     {}", info.section_count);
            println!("  Checksum:     0x{:08X}", info.crc32);
            println!("  Total size:   {:.1} KB", info.file_size as f64 / 1024.0);
            if verbose {
                println!("");
                println!("Section Details:");
                println!("  {:<22} {:>6} {:>10} {:>10}", "Kind", "u8", "Offset", "Len");
                println!("  {:<22} {:>6} {:>10} {:>10}", "----", "--", "------", "---");
                let mut total: u64 = 0;
                for &(kind_u8, offset, len) in &info.sections {
                    let name = crate::solidify::Spec::SectionKind::from_u8(kind_u8)
                        .map(|k| k.name())
                        .unwrap_or("Unknown");
                    println!("  {:<22} {:>6} {:>10} {:>10}", name, kind_u8, offset, len);
                    total += len as u64;
                }
                let total_kb = total as f64 / 1024.0;
                let overhead = info.file_size as f64 - total as f64;
                println!("  {:<22} {:>6} {:>10} {:>10.1}", "TOTAL", "", "", total_kb);
                println!("  {:<22} {:>6} {:>10} {:>10.1}", "Overhead", "", "", overhead / 1024.0);
            }
        }
        Err(e) => {
            eprintln!("error: invalid .fndo file: {}", e);
            process::exit(1);
        }
    }
}
