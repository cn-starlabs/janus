//! 3D load-balance microbenchmark (ENGINEERING_SPEC.md §7 deliverable, 3D
//! generalization of `load_balance_bench.rs`): half-continuum / half-rarefied
//! 3D mesh, static block assignment vs dtact work-deflection enabled,
//! wall-clock comparison.
//!
//! DESIGN / HONESTY NOTE: identical caveat to the 2D microbench — this binary
//! requires the dtact runtime initialized via `#[dtact::dtact_init]`. It was
//! written and manually type/borrow-checked against the `dtact` API as read
//! from `dtact/src/api.rs`, but it was **not** compiled or run in this
//! environment: no Rust toolchain was available (`cargo`/`rustc` not on
//! `PATH`). Numbers below are NOT included — do not treat any number here as
//! measured. Run with:
//!
//! ```text
//! cargo run --release -p janus-sched --example load_balance_bench_3d
//! ```
//!
//! ## What it measures
//!
//! Builds a `Grid3D` split into a "west" half (dense/continuum, low particle
//! density) and an "east" half (rarefied, high particle density), partitions
//! it into 3D blocks via `janus_sched::block3d::partition_grid_3d`, and times
//! `SchedRunner3D::step_all_blocks` over N steps in two configurations
//! (static assignment vs dtact deflection enabled) — same methodology as the
//! 2D microbench, generalized to a 3D block grid.

use dtact::dtact_init;
use janus_core::grid3d::Grid3D;
use janus_sched::block3d::BlockKind3D;
use janus_sched::SchedRunner3D;
use std::time::Instant;

/// Stand-in per-block "physics" cost, 3D analog of the 2D bench's
/// `fake_block_step`: busier for particle-dominated blocks (simulates the
/// extra cost of the stochastic particle layer vs a pure wave update).
fn fake_block_step_3d(block: &janus_sched::block3d::Block3D, kind: BlockKind3D) -> u64 {
    let ncells = block.ncells() as u64;
    let work_units = match kind {
        BlockKind3D::WaveDominated => ncells * 10,
        BlockKind3D::ParticleDominated => ncells * 200,
    };
    let mut acc: u64 = 0;
    for i in 0..work_units {
        acc = acc.wrapping_add(i.wrapping_mul(2654435761));
    }
    std::hint::black_box(acc);
    match kind {
        BlockKind3D::WaveDominated => ncells / 4,
        BlockKind3D::ParticleDominated => ncells * 64,
    }
}

#[dtact_init]
fn main() {
    let nx = 24;
    let ny = 16;
    let nz = 16;
    let grid = Grid3D::new(nx, ny, nz, 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]);

    let n_steps = 20;
    let blocks_x = 4;
    let blocks_y = 2;
    let blocks_z = 2;

    // --- Static assignment: high deflection threshold discourages stealing. ---
    let mut runner_static = SchedRunner3D::new(grid, blocks_x, blocks_y, blocks_z);
    runner_static.deflection_threshold = 255;
    runner_static.configure_deflection(64);
    seed_half_rarefied(&mut runner_static);

    let t0 = Instant::now();
    for _ in 0..n_steps {
        runner_static.step_all_blocks(fake_block_step_3d);
    }
    let static_elapsed = t0.elapsed();

    // --- Deflection enabled: low threshold encourages work-stealing. ---
    let mut runner_deflect = SchedRunner3D::new(grid, blocks_x, blocks_y, blocks_z);
    runner_deflect.deflection_threshold = 1;
    runner_deflect.configure_deflection(64);
    seed_half_rarefied(&mut runner_deflect);

    let t1 = Instant::now();
    for _ in 0..n_steps {
        runner_deflect.step_all_blocks(fake_block_step_3d);
    }
    let deflect_elapsed = t1.elapsed();

    println!("3D static assignment: {static_elapsed:?} over {n_steps} steps");
    println!("3D dtact deflection:   {deflect_elapsed:?} over {n_steps} steps");
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

fn seed_half_rarefied(runner: &mut SchedRunner3D) {
    let nx = runner.grid.nx;
    for b in runner.blocks.iter_mut() {
        // "East" half of the domain (by i0) is rarefied.
        if b.i0 >= nx / 2 {
            b.last_particle_count = b.ncells() as u64 * 64;
        } else {
            b.last_particle_count = b.ncells() as u64 / 4;
        }
    }
}
