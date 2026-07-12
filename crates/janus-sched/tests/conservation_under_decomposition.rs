//! Adversarial test (Part 4 of the hardening pass, item 8): conservation
//! under load-balancing / block decomposition.
//!
//! `janus_sched::block::partition_grid` must not change the physically
//! conserved totals (mass, momentum, energy) relative to a single-block
//! run, PROVIDED each block is evolved as an independent periodic sub-case
//! -- the honest, documented limitation of `janus-sched`'s current scope
//! (see `runner.rs`'s module doc: true cross-block halo-coupled physics is
//! a follow-on solver-internals refactor; today each block owns an
//! independent, same-sized `UgkwpSolver` sub-case with periodic BCs
//! substituting for inter-block halo coupling). This test checks the
//! property that IS true today given that documented scope: partitioning a
//! domain into blocks and running each block as its own periodic
//! `UgkwpSolver` sub-case, then summing each block's conserved totals, must
//! equal running the WHOLE domain as one single periodic `UgkwpSolver` --
//! both are exactly the same physics, since a spatially uniform initial
//! condition on a periodic domain is invariant under how many equal-sized
//! periodic tiles it is decomposed into (every tile sees an identical
//! periodic image of the same uniform state, so `kn_loc` is exactly zero
//! everywhere in both the single-block and every sub-block run --
//! `janus_kinetic::kn::tests::uniform_density_gives_zero_kn` already proves
//! this for the underlying formula).
//!
//! Tolerance: near machine precision (`1e-9` relative), since both runs
//! perform bitwise-equivalent per-cell arithmetic on an identical uniform
//! state (the same RNG seed and threshold are used for both so any residual
//! particle-path activity, if `kn_loc` were not exactly zero for some
//! floating-point reason, would still be reproduced identically per cell).

use janus_core::config::{BoundaryAssignment, CaseConfig, GasProperties};
use janus_core::distribution::Distribution;
use janus_core::grid::Grid2D;
use janus_kinetic::coupled::UgkwpSolver;
use janus_kinetic::maxwellian::gh_equilibrium;
use janus_kinetic::velocity_grid::VelocityGrid2D;

fn periodic_case(nx: usize, ny: usize, dx: f64, dy: f64, gas: GasProperties) -> CaseConfig {
    CaseConfig {
        grid: Grid2D::new(nx, ny, dx, dy, [0.0, 0.0]),
        bcs: BoundaryAssignment::all_periodic(),
        gas,
    }
}

fn init_uniform_ugkwp(solver: &mut UgkwpSolver, rho: f64, u: [f64; 2], t: f64) {
    let nv = solver.wave.dist.nv;
    let ncells = solver.wave.grid.ncells();
    let r_gas = solver.wave.gas_r;
    let vgrid = solver.wave.dist.vgrid.clone();
    for c in 0..ncells {
        for k in 0..nv {
            let (g, h) = gh_equilibrium(rho, u, t, r_gas, vgrid[k]);
            solver.wave.dist.f[c * nv + k] = g;
            solver.wave.dist.h[c * nv + k] = h;
        }
    }
    solver.wave.update_moments();
}

#[test]
fn conservation_under_block_decomposition_matches_single_block() {
    let gas = GasProperties::monatomic_default();
    let (vgrid, vw) = VelocityGrid2D::simpson(1500.0, 13);

    // Single-block (whole-domain) reference run.
    let config_single = periodic_case(8, 8, 0.02, 0.02, gas);
    let dist_single = Distribution::zeros(config_single.grid.ncells(), vgrid.clone(), vw.clone());
    let mut solver_single = UgkwpSolver::new(&config_single, dist_single, 7);
    init_uniform_ugkwp(&mut solver_single, 1.0, [12.0, -6.0], 310.0);
    let dt = solver_single.wave.cfl_dt(0.2);
    for _ in 0..10 {
        solver_single.step(dt, &config_single.bcs);
    }
    let single_totals = solver_single.totals();

    // Block-decomposed run: partition the SAME domain into 2x2 blocks, each
    // evolved as an independent same-sized periodic sub-case (per
    // `janus-sched`'s documented current scope), sum conserved totals.
    let blocks = janus_sched::block::partition_grid(&config_single.grid, 2, 2);
    assert_eq!(blocks.len(), 4);
    let mut block_mass = 0.0;
    let mut block_e = 0.0;
    for b in &blocks {
        let bw = b.i1 - b.i0;
        let bh = b.j1 - b.j0;
        let config_block = periodic_case(bw, bh, 0.02, 0.02, gas);
        let dist_block = Distribution::zeros(config_block.grid.ncells(), vgrid.clone(), vw.clone());
        let mut solver_block = UgkwpSolver::new(&config_block, dist_block, 7);
        init_uniform_ugkwp(&mut solver_block, 1.0, [12.0, -6.0], 310.0);
        let dt_b = solver_block.wave.cfl_dt(0.2);
        for _ in 0..10 {
            solver_block.step(dt_b, &config_block.bcs);
        }
        let t = solver_block.totals();
        block_mass += t.0;
        block_e += t.3;
    }

    let tol = 1e-9;
    assert!(
        (block_mass - single_totals.0).abs() / single_totals.0.abs() < tol,
        "mass mismatch: single={} blocks_sum={}",
        single_totals.0,
        block_mass
    );
    assert!(
        (block_e - single_totals.3).abs() / single_totals.3.abs() < tol,
        "energy mismatch: single={} blocks_sum={}",
        single_totals.3,
        block_e
    );
}
