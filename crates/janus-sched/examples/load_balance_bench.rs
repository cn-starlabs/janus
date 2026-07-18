//! Load-balance microbenchmark (ENGINEERING_SPEC.md §7 deliverable):
//! half-continuum / half-rarefied mesh, static block assignment vs dtact
//! work-deflection enabled, wall-clock comparison.
//!
//! DESIGN / HONESTY NOTE: this binary requires the dtact runtime to be
//! initialized via the `#[dtact::dtact_init]` attribute macro (applied to
//! `main`, per `dtact`'s documented entry point — see `janus-sched`'s
//! `lib.rs` module doc for why `janus-sched` itself cannot self-initialize
//! the runtime). It was written and manually type/borrow-checked against
//! the `dtact` API as read from `dtact/src/api.rs` and `dtact/src/lib.rs`,
//! but it was **not** compiled or run in the environment this was authored
//! in: no Rust toolchain was available (`cargo`/`rustc` not on `PATH`) and
//! network installation of one was blocked (HTTP 403 from the rustup
//! endpoint). Numbers below are therefore NOT included — do not treat any
//! number here as measured; there are none. Run with:
//!
//! ```text
//! cargo run --release -p janus-sched --example load_balance_bench
//! ```
//!
//! ## What it measures
//!
//! Builds a `Grid2D` split into a left half (dense/continuum, low particle
//! density, `kn_loc` forced below threshold) and a right half (rarefied,
//! high particle density, `kn_loc` forced above threshold), partitions it
//! into blocks via `janus_sched::block::partition_grid`, and times
//! `SchedRunner::step_all_blocks` over N steps in two configurations:
//!
//! 1. **Static assignment**: deflection disabled
//!    (`set_deflection_threshold` set high enough that dtact's scheduler
//!    never deflects, approximating a fixed 1:1 block:core mapping).
//! 2. **Deflection enabled**: default/low deflection threshold, letting
//!    idle workers steal work from overloaded ones as the rarefied-side
//!    blocks (which cost more per step due to particle overhead) fall
//!    behind.

// DESIGN / VERIFICATION GAP: `dtact::dtact_init` is a proc-macro re-exported
// from the separate `dtact_macros` crate (`dtact/src/lib.rs`: `pub use
// dtact_macros::dtact_init;`). Only `dtact`'s `src/` tree (not
// `dtact_macros`'s source, and not `dtact`'s own `Cargo.toml`) was mounted
// in the environment this was authored in, so the macro's exact expansion
// (does it require `fn main()`? does it take config arguments? what does it
// expand to internally — presumably a `Runtime` construction + `GLOBAL_
// RUNTIME.set(...)` + `Runtime::start()`) could not be inspected or
// verified. The attribute-macro usage below (`#[dtact_init]` on `fn main()`,
// no arguments) is the most natural reading of "Attribute macro for
// initializing the Dtact runtime" applied to a binary's entry point, and
// matches the doc comment on `dtact::dtact_init`'s re-export, but it is
// UNVERIFIED — flagged prominently per the task's instructions on
// dependencies/APIs outside what could be directly confirmed.
use dtact::dtact_init;
use janus_core::grid::Grid2D;
use janus_sched::block::BlockKind;
use janus_sched::runner::{BlockHalo, HaloSample};
use janus_sched::SchedRunner;
use std::time::Instant;

/// Stand-in per-block "physics" cost: busier for particle-dominated blocks
/// (simulates the extra cost of the stochastic particle layer vs a pure
/// wave update) so the two scheduling policies have something asymmetric to
/// load-balance. Returns a synthetic "particle count" cost proxy.
fn fake_block_step(block: &janus_sched::block::Block, kind: BlockKind) -> u64 {
    let ncells = block.ncells() as u64;
    let work_units = match kind {
        BlockKind::WaveDominated => ncells * 10,
        BlockKind::ParticleDominated => ncells * 200,
    };
    // Busy-spin a proportional amount of work (stand-in for real DUGKS
    // flux / particle transport kernels) so wall-clock time actually
    // reflects `work_units`.
    let mut acc: u64 = 0;
    for i in 0..work_units {
        acc = acc.wrapping_add(i.wrapping_mul(2654435761));
    }
    std::hint::black_box(acc);
    match kind {
        BlockKind::WaveDominated => ncells / 4, // low particle density
        BlockKind::ParticleDominated => ncells * 64, // high particle density
    }
}

fn fake_block_step_with_halo(
    block: &janus_sched::block::Block,
    kind: BlockKind,
    halo: &BlockHalo,
) -> (u64, HaloSample) {
    let base_cost = fake_block_step(block, kind);
    let halo_sum: f64 = halo
        .west
        .current
        .iter()
        .chain(halo.east.current.iter())
        .chain(halo.south.current.iter())
        .chain(halo.north.current.iter())
        .sum();
    let adjusted_cost = base_cost + halo_sum.round().abs() as u64;
    let sample = HaloSample {
        west: vec![adjusted_cost as f64],
        east: vec![adjusted_cost as f64 / 2.0],
        south: vec![],
        north: vec![],
    };
    (adjusted_cost, sample)
}

#[dtact_init]
fn main() {
    let nx = 64;
    let ny = 32;
    let grid = Grid2D::new(nx, ny, 1.0, 1.0, [0.0, 0.0]);

    let n_steps = 20;
    let blocks_x = 8;
    let blocks_y = 4;

    // --- Static assignment: high deflection threshold discourages stealing. ---
    let mut runner_static = SchedRunner::new(grid, blocks_x, blocks_y);
    runner_static.deflection_threshold = 255;
    runner_static.configure_deflection(64);
    // Seed particle counts asymmetrically: right half of the domain is
    // "rarefied" (high particle count), left half "continuum".
    seed_half_rarefied(&mut runner_static);

    let t0 = Instant::now();
    for _ in 0..n_steps {
        runner_static.step_all_blocks_with_halo(fake_block_step_with_halo);
    }
    let static_elapsed = t0.elapsed();

    // --- Deflection enabled: low threshold encourages work-stealing. ---
    let mut runner_deflect = SchedRunner::new(grid, blocks_x, blocks_y);
    runner_deflect.deflection_threshold = 1;
    runner_deflect.configure_deflection(64);
    seed_half_rarefied(&mut runner_deflect);

    let t1 = Instant::now();
    for _ in 0..n_steps {
        runner_deflect.step_all_blocks_with_halo(fake_block_step_with_halo);
    }
    let deflect_elapsed = t1.elapsed();

    println!("static assignment:   {static_elapsed:?} over {n_steps} steps");
    println!("dtact deflection:     {deflect_elapsed:?} over {n_steps} steps");
    if deflect_elapsed < static_elapsed {
        let speedup = static_elapsed.as_secs_f64() / deflect_elapsed.as_secs_f64();
        println!("deflection speedup: {speedup:.2}x");
    } else {
        println!(
            "no speedup observed in this run (see janus-sched module docs: block-local \
                   fake-work granularity / core count on the actual test machine both affect this)"
        );
    }
}

fn seed_half_rarefied(runner: &mut SchedRunner) {
    let nx = runner.grid.nx;
    for b in runner.blocks.iter_mut() {
        // Right half of the domain (by i0) is rarefied.
        if b.i0 >= nx / 2 {
            b.last_particle_count = b.ncells() as u64 * 64;
        } else {
            b.last_particle_count = b.ncells() as u64 / 4;
        }
    }
}
