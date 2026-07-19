//! Rust-side parity harness mirroring `serenitymojo/serve/serenity_lower_cli.mojo`:
//! read a /v1/generate-shaped request JSON, run `lower_request` in place, and
//! print the lowered request object on stdout (or the [501] error). Lets the
//! same input be lowered by BOTH oracles and diffed for parity.
//!
//! Usage:  cargo run -p serenity-graph --example lower -- <request.json>

use std::fs;
use std::process::exit;

use serde_json::Value as JsonValue;
use serenity_graph::lower_request;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: lower <request.json>");
        exit(1);
    }
    let text = fs::read_to_string(&args[1]).unwrap_or_else(|e| {
        eprintln!("lower: cannot read {}: {e}", args[1]);
        exit(1);
    });
    let mut obj: JsonValue = serde_json::from_str(&text).unwrap_or_else(|e| {
        eprintln!("lower: parse error: {e}");
        exit(1);
    });
    match lower_request(&mut obj) {
        Ok(()) => {
            println!("{}", serde_json::to_string(&obj).unwrap());
        }
        Err(e) => {
            eprintln!("{e}");
            exit(1);
        }
    }
}
