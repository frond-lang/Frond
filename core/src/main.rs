//! frond CLI entry point — delegates to frond::cli.

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    frond::cli::run();
}
