<p align="center">
  <img src="https://github.com/kuzo-lang/assets/blob/main/logo.png?raw=true" alt="Kuzo" width="200">
</p>

# Kuzo

Kuzo is a statically typed programming language implemented in Rust, featuring a dataflow-ready scheduling execution model. Source files use the `.kz` extension; compiled artifacts are cross-platform `.kzo` binaries.

## Build

```bash
cargo build --release
# Binary: target/release/kuzo
```

## Quick Start

```bash
# Create a new project
kuzo init myapp
cd myapp

# Compile and run
kuzo run

# Or compile only
kuzo build
# Execute the artifact
kuzo run out/myapp.kzo
```

Generated project layout:

```
myapp/
├── kuzo.toml      # Project manifest
└── src/Main.kz    # Entry point
```

Default entry:

```kuzo
fun main(): void {
    println("Hello, Kuzo!")
}
```

## Language Tour

```kuzo
// Variables
val x: i32 = 42
var counter: i32 = 0

// Functions and generics
fun add(a: i32, b: i32): i32 { a + b }
fun identity<T>(x: T): T { x }

// ADTs and pattern matching
type Shape = | Circle(f64) | Rect(f64, f64)

fun area(s: Shape): f64 {
    match s {
        Circle(r) => 3.14159 * r * r
        Rect(w, h) => w * h
    }
}

// Records
type Point = Point(x: i32, y: i32)
val p = Point(3, 4)

// Error handling
fun safeDiv(a: i32, b: i32): Throw<i32, Error> {
    if b == 0 { throw Error("div by zero") }
    Ok(a / b)
}

// Async and channels
async fun fetch(): Async<i32> {
    Timer(1).await()
    42
}

val ch = channel<i32>(2)
ch.send(10)
ch.recv()  // 10

// String interpolation
println("sum = {1 + 2}, point = {p}")
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `kuzo init [name]` | Scaffold a new project |
| `kuzo build [-O N]` | Compile to `.kzo` (`-O 0..3`, default 2) |
| `kuzo run [-O N]` | Compile and run |
| `kuzo run <file.kzo>` | Execute a compiled artifact |
| `kuzo debug --stage S` | Diagnostics (`tokens`/`ast`/`check`/`emit-c`/`emit-ffi`/`full`) |
| `kuzo inspect <file.kzo>` | Inspect `.kzo` metadata |
