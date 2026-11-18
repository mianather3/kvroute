//! Deterministic discrete-event simulator for `kvroute-core` routing policies.
//!
//! Workers, caches, and routers are all driven by one event queue keyed on
//! `(time, insertion seq)`, so a given seed always yields bit-identical metrics.

pub mod engine;
pub mod policy;
pub mod worker;
pub mod workload;

pub use kvroute_core::BLOCK_SIZE;
