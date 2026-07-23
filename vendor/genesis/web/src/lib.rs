//! Headless Genesis integration surface for Serenity's browser video editor.
//!
//! This crate intentionally contains no HTTP server and launches no native UI.
//! Serenity owns the web/API layer; Genesis owns the project model and the
//! isolated Rust/C/FFmpeg/OpenCL composition worker.

pub mod model;
pub mod worker;
