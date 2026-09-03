//! memsizes — one-shot diagnostic probe: exact layout sizes of the core value
//! system. Not part of any test suite; run with
//!   cargo rustc --release --example memsizes -- -C link-arg=/WHOLEARCHIVE:frond_extern.lib
#![allow(dead_code)]

use frond::ir::Ir::{Frame, ValueTable};
use frond::value::{
    ArrayValue, Cell, Closure, ErrorValue, HeapObj, PartialApplication, Range, RecordRef,
    RecordShape, ScalarSoA, ScalarValue, TraitValue, ThrowValue, Value, ValueHandle, ValueTag,
};

macro_rules! sz {
    ($t:ty) => {{
        println!("{:<24} {:>4} bytes  align {}", stringify!($t), std::mem::size_of::<$t>(), std::mem::align_of::<$t>());
    }};
}

fn main() {
    println!("== value layer ==");
    sz!(Value);
    sz!(ScalarValue);
    sz!(ValueTag);
    sz!(HeapObj);
    println!("{:<24} {:>4} bytes  (Arc header 16 + HeapObj)", "Arc<HeapObj> alloc", 16 + std::mem::size_of::<HeapObj>());
    sz!(RecordShape);
    println!("{:<24} {:>4} bytes  (header + packed tail at runtime)", "RecordRef block", std::mem::size_of::<frond::value::RecordShape>()); // block layout is per-shape; see MEM_BASELINE
    sz!(ArrayValue);
    sz!(ScalarSoA);
    sz!(Cell);
    sz!(Range);
    sz!(Closure);
    sz!(PartialApplication);
    sz!(TraitValue);
    sz!(ErrorValue);
    sz!(ThrowValue);
    sz!(ValueHandle);

    println!("== frame layer ==");
    sz!(ValueTable);
    sz!(Frame);
    sz!(Vec<Value>);
}
