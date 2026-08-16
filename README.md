<p align="center">
  <img src="https://github.com/frond-lang/assets/blob/main/logo.png?raw=true" alt="Frond" width="200">
</p>

# Frond

Frond is a statically typed programming language implemented in Rust, featuring a dataflow-ready scheduling execution model. Source files use the `.frond` extension; compiled artifacts are cross-platform `.fndo` binaries.

## Build

```bash
cargo build --release
# Binary: target/release/frond
```

## Quick Start

```bash
# Create a new project
frond init myapp
cd myapp

# Compile and run
frond run

# Or compile only
frond build
# Execute the artifact
frond run out/myapp.fndo
```

Generated project layout:

```
myapp/
├── Root.toml        # Project manifest
└── src/Main.frond   # Entry point
```

Default entry:

```frond
fun main(): void {
    println("Hello, Frond!")
}
```

## Language Tour

```frond
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
| `frond init [name]` | Scaffold a new project |
| `frond build [-O N]` | Compile to `.fndo` (`-O 0..3`, default 2) |
| `frond run [-O N]` | Compile and run |
| `frond run <file.fndo>` | Execute a compiled artifact |
| `frond debug --stage S` | Diagnostics (`tokens`/`ast`/`check`/`emit-c`/`emit-ffi`/`full`) |
| `frond inspect <file.fndo>` | Inspect `.fndo` metadata |
