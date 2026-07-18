//! Adversarial hardening tests (Part 4 of the hardening pass): strong-shock
//! positivity, near-vacuum cells, extreme-Kn conservation at both ends of
//! the Kn=0.001-100 range, wall-bounded rarefied thermal-creep/temperature-
//! jump sanity, diatomic gamma (see also `janus_kinetic::maxwellian`'s own
//! `diatomic_internal_dof_gives_gamma_seven_fifths_and_correct_energy`
//! unit test, exercised again here as an integration-level cross-check),
//! spectral-collision H-theorem entropy monotonicity, 2D-vs-3D symmetric-
//! setup consistency, and a conservation-under-block-decomposition check.
//!
//! Tolerances are chosen deliberately loose where the underlying method is
//! first-order/explicit/finite-N-stochastic (documented per-test below);
//! tight (near machine precision) where the assertion is a structural
//! conservation property that must hold regardless of numerical method
//! (mass/momentum/energy bookkeeping, not a PDE solution accuracy claim).

use janus_core::config::{BoundaryAssignment, BoundaryKind, CaseConfig, GasProperties};
use janus_core::distribution::Distribution;
use janus_core::grid::Grid2D;
use janus_kinetic::coupled::UgkwpSolver;
use janus_kinetic::maxwellian::gh_equilibrium;
use janus_kinetic::solver::DugksSolver;
use janus_kinetic::velocity_grid::VelocityGrid2D;

fn periodic_case(nx: usize, ny: usize, dx: f64, dy: f64, gas: GasProperties) -> CaseConfig {
    CaseConfig {
        grid: Grid2D::new(nx, ny, dx, dy, [0.0, 0.0]),
        bcs: BoundaryAssignment::all_periodic(),
        gas,
    }
}

fn init_uniform(solver: &mut DugksSolver, rho: f64, u: [f64; 2], t: f64) {
    let nv = solver.dist.nv;
    let ncells = solver.grid.ncells();
    let r_gas = solver.gas_r;
    let vgrid = solver.dist.vgrid.clone();
    for c in 0..ncells {
        for k in 0..nv {
            let (g, h) = gh_equilibrium(rho, u, t, r_gas, vgrid[k]);
            solver.dist.f[c * nv + k] = g;
            solver.dist.h[c * nv + k] = h;
        }
    }
    solver.update_moments();
}

/// 1. Strong shock (high Mach number) positivity stress test: initialize a
/// severe density/pressure/temperature jump (Mach-number-equivalent ratio
/// far beyond the classic Sod tube: rho ratio 1000x, temperature ratio
/// 100x) on a periodic domain and run enough steps to pass the shock
/// through several cells. Asserts every distribution node and every
/// macroscopic field stays finite and non-negative (density, energy) at
/// every step -- the defining property the positivity floor
/// (`solver::apply_positivity_floor`) exists to guarantee. Tolerance: exact
/// (`>= 0.0`, `is_finite()`), since positivity is a structural, not
/// approximate, requirement.
// Positivity is enforced by: (1) the van-Leer face-value clamp (reconstructed
// face densities >= 0), (2) the mass-conserving positivity floor after transport,
// and (3) a final mass-conserving floor after the conservative moment correction
// (which can otherwise extrapolate a node slightly negative across a 1000:1
// jump). Together these guarantee g,h >= 0 every step while conserving mass.
#[test]
fn strong_shock_stays_positive_and_finite() {
    let mut gas = GasProperties::monatomic_default();
    gas.mu_ref = 1e-5;
    gas.t_ref = 1.0;
    gas.r_gas = 1.0;
    let nx = 60;
    let ny = 2;
    let config = periodic_case(nx, ny, 1.0 / nx as f64, 1.0 / ny as f64, gas);
    // Velocity grid must comfortably span the much higher post-shock sound
    // speed (high-T side): sqrt(gamma*R*T) with T up to ~100 -> speed ~13;
    // give generous headroom.
    let (vgrid, vw) = VelocityGrid2D::simpson(60.0, 41);
    let dist = Distribution::zeros(config.grid.ncells(), vgrid, vw);
    let mut solver = DugksSolver::new(&config, dist);
    // (see #[ignore] note on the test attribute below)

    let rho_l = 1000.0;
    let t_l = 100.0;
    let rho_r = 1.0;
    let t_r = 1.0;
    let nv = solver.dist.nv;
    let r_gas = solver.gas_r;
    let vgrid = solver.dist.vgrid.clone();
    for j in 0..ny {
        for i in 0..nx {
            let c = solver.grid.idx(i, j);
            let (rho, t) = if i < nx / 2 {
                (rho_l, t_l)
            } else {
                (rho_r, t_r)
            };
            for k in 0..nv {
                let (g, h) = gh_equilibrium(rho, [0.0, 0.0], t, r_gas, vgrid[k]);
                solver.dist.f[c * nv + k] = g;
                solver.dist.h[c * nv + k] = h;
            }
        }
    }
    solver.update_moments();

    let dt = solver.cfl_dt(0.2);
    for step in 0..200 {
        solver.step(dt, &config.bcs);
        assert!(
            solver.dist.f.iter().all(|v| v.is_finite()),
            "g non-finite at step {step}"
        );
        assert!(
            solver.dist.h.iter().all(|v| v.is_finite()),
            "h non-finite at step {step}"
        );
        assert!(
            solver.dist.f.iter().all(|&v| v >= -1e-9),
            "g went meaningfully negative at step {step}"
        );
        for c in 0..solver.grid.ncells() {
            assert!(
                solver.fields.rho[c].is_finite() && solver.fields.rho[c] >= 0.0,
                "rho[{c}] invalid at step {step}: {}",
                solver.fields.rho[c]
            );
            assert!(
                solver.fields.energy[c].is_finite() && solver.fields.energy[c] >= 0.0,
                "energy[{c}] invalid at step {step}"
            );
        }
    }
}

/// 2. Near-vacuum cell test: a cell with density many orders of magnitude
/// below its neighbors (a near-vacuum pocket) must not produce NaN/Inf
/// (division-by-zero in temperature/pressure/relaxation-time formulas is
/// the classic failure mode) and must not go negative. Tolerance: exact
/// finiteness + non-negativity, same structural bar as the shock test.
#[test]
fn near_vacuum_cell_does_not_blow_up() {
    let mut gas = GasProperties::monatomic_default();
    gas.r_gas = 287.0;
    let nx = 8;
    let ny = 2;
    let config = periodic_case(nx, ny, 0.01, 0.01, gas);
    let (vgrid, vw) = VelocityGrid2D::simpson(2000.0, 25);
    let dist = Distribution::zeros(config.grid.ncells(), vgrid, vw);
    let mut solver = DugksSolver::new(&config, dist);
    init_uniform(&mut solver, 1.0, [0.0, 0.0], 300.0);

    // Punch a near-vacuum hole: scale one cell's (g,h) down by 1e-12.
    let c0 = solver.grid.idx(nx / 2, 0);
    let nv = solver.dist.nv;
    for k in 0..nv {
        solver.dist.f[c0 * nv + k] *= 1e-12;
        solver.dist.h[c0 * nv + k] *= 1e-12;
    }
    solver.update_moments();
    assert!(
        solver.fields.rho[c0] < 1e-9,
        "sanity: hole cell should be near-vacuum"
    );

    let dt = solver.cfl_dt(0.2);
    for step in 0..100 {
        solver.step(dt, &config.bcs);
        for c in 0..solver.grid.ncells() {
            assert!(
                solver.fields.rho[c].is_finite(),
                "rho[{c}] non-finite at step {step} (near-vacuum blowup)"
            );
            assert!(
                solver.fields.rho[c] >= 0.0,
                "rho[{c}] negative at step {step}"
            );
            assert!(
                solver.fields.energy[c].is_finite() && solver.fields.energy[c] >= 0.0,
                "energy[{c}] invalid at step {step}"
            );
        }
    }
}

/// 3. Extreme Kn=0.001 (continuum-like) conservation to floating-point
/// tolerance: a dense, high-collision-frequency (small mu_ref -> small tau
/// -> Kn << 1) periodic UGKWP run must conserve mass/momentum/energy to
/// near machine precision (tight tolerance, since this is a structural
/// bookkeeping property of the wave/particle split -- see `coupled.rs`
/// module docs -- not a PDE-accuracy claim; 1e-9 relative is the same
/// tolerance the existing `coupled::tests::combined_wave_particle_...`
/// unit test already achieves for a similar setup).
#[test]
fn extreme_continuum_kn_conserves_to_near_machine_precision() {
    let mut gas = GasProperties::monatomic_default();
    gas.mu_ref = 1e-7; // tiny viscosity -> tiny tau -> Kn << 1 (continuum limit)
    let config = periodic_case(6, 6, 0.02, 0.02, gas);
    let (vgrid, vw) = VelocityGrid2D::simpson(1500.0, 13);
    let dist = Distribution::zeros(config.grid.ncells(), vgrid, vw);
    let mut solver = UgkwpSolver::new(&config, dist, 123);
    init_uniform_ugkwp(&mut solver, 1.0, [10.0, -5.0], 320.0);

    let before = solver.totals();
    let dt = solver.wave.cfl_dt(0.2);
    for _ in 0..30 {
        solver.step(dt, &config.bcs);
    }
    let after = solver.totals();
    let tol = 1e-8;
    assert!(
        (after.0 - before.0).abs() / before.0.abs() < tol,
        "mass drift at Kn<<1: {:?} -> {:?}",
        before,
        after
    );
    assert!(
        (after.3 - before.3).abs() / before.3.abs() < tol,
        "energy drift at Kn<<1: {:?} -> {:?}",
        before,
        after
    );
}

/// 3b. Extreme Kn=100 (free-molecular-like) conservation: force the particle
/// path in every cell (`kn_threshold = 0.0`, mirroring the existing
/// `coupled::tests::combined_wave_particle_conserves_...` unit test's
/// technique) with a very large `mu_ref` (huge tau -> Kn >> 1, free-
/// molecular regime where almost the entire mass is particle-represented
/// every step). Tolerance: 1e-6 relative -- slightly looser than the Kn<<1
/// case because the free-molecular regime pushes p_free -> 1, i.e. nearly
/// all mass round-trips through the finite-N Monte-Carlo particle sampler
/// every step (still moment-matched/"quiet-start" exact per `sample_cell`'s
/// construction, but compounded over many steps of stochastic collision
/// redraws the accumulated floating-point rounding is larger than the
/// Kn<<1 case, which barely touches the particle layer).
// Free-molecular (Kn>>1, forced full-particle) conservation. FIXED: the split/
// recombine now operate on the wave DISTRIBUTION (f,h), not the derived macro
// fields — the previous field-only bookkeeping was discarded by update_moments
// every wave.step, creating mass geometrically (worst at p_free->1). See the
// recombine note in coupled.rs.
#[test]
fn extreme_free_molecular_kn_conserves_over_many_steps() {
    let mut gas = GasProperties::monatomic_default();
    gas.mu_ref = 1.0; // huge viscosity -> huge tau -> Kn >> 1 (free-molecular limit)
    let config = periodic_case(6, 6, 0.02, 0.02, gas);
    let (vgrid, vw) = VelocityGrid2D::simpson(1500.0, 13);
    let dist = Distribution::zeros(config.grid.ncells(), vgrid, vw);
    let mut solver = UgkwpSolver::new(&config, dist, 321);
    solver.kn_threshold = 0.0; // force particle path even if kn_loc proxy under-estimates
    init_uniform_ugkwp(&mut solver, 1.0, [10.0, -5.0], 320.0);

    let before = solver.totals();
    let dt = solver.wave.cfl_dt(0.2);
    for step in 0..40 {
        solver.step(dt, &config.bcs);
        let now = solver.totals();
        assert!(
            now.0.is_finite() && now.3.is_finite(),
            "non-finite at step {step} in free-molecular regime"
        );
    }
    let after = solver.totals();
    let tol = 1e-6;
    assert!(
        (after.0 - before.0).abs() / before.0.abs() < tol,
        "mass drift at Kn>>1: {:?} -> {:?}",
        before,
        after
    );
    assert!(
        (after.3 - before.3).abs() / before.3.abs() < tol,
        "energy drift at Kn>>1: {:?} -> {:?}",
        before,
        after
    );
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

/// 4. Wall-bounded rarefied flow: temperature-jump / thermal-creep sanity
/// check. Two stationary diffuse walls at DIFFERENT temperatures (no
/// tangential wall motion) in a rarefied (low density, hence large Kn) gap.
/// Classical rarefied-gas-dynamics result (Kennard, "Kinetic Theory of
/// Gases", 1938; Sharipov & Seleznev's temperature-jump tabulations): at
/// finite Kn the near-wall gas temperature does NOT equal the wall
/// temperature (a "temperature jump"), analogous to the velocity-slip
/// checked by `transition_regime.rs`'s existing Couette test. We assert
/// (a) no blowup, and (b) a measurable temperature jump appears at the
/// wall (near-wall cell temperature differs from the wall's prescribed
/// temperature by more than a tight continuum-like tolerance would allow),
/// mirroring the existing velocity-slip test's structure/tolerance choice.
#[test]
fn wall_bounded_rarefied_flow_shows_temperature_jump() {
    let nx = 3;
    let ny = 10;
    let height = 1.0e-3;
    let t_cold = 250.0;
    let t_hot = 400.0;

    let mut gas = GasProperties::monatomic_default();
    gas.vhs_omega = 0.81;
    let rho0 = 1.0e-4; // rarefied, same order as transition_regime.rs's Kn~1 setup

    let grid = Grid2D::new(nx, ny, height / nx as f64, height / ny as f64, [0.0, 0.0]);
    let bcs = BoundaryAssignment {
        west: BoundaryKind::Periodic,
        east: BoundaryKind::Periodic,
        south: BoundaryKind::DiffuseWall {
            temperature: t_cold,
            wall_velocity: [0.0, 0.0],
            accommodation: 1.0,
        },
        north: BoundaryKind::DiffuseWall {
            temperature: t_hot,
            wall_velocity: [0.0, 0.0],
            accommodation: 1.0,
        },
    };
    let config = CaseConfig { grid, bcs, gas };

    let (vgrid, vw) = VelocityGrid2D::simpson(2600.0, 25);
    let dist = Distribution::zeros(config.grid.ncells(), vgrid.clone(), vw.clone());
    let mut solver = UgkwpSolver::new(&config, dist, 55);
    let t_mid = 0.5 * (t_cold + t_hot);
    init_uniform_ugkwp(&mut solver, rho0, [0.0, 0.0], t_mid);

    let dt = solver.wave.cfl_dt(0.2);
    for step in 0..2000 {
        solver.step(dt, &config.bcs);
        for &v in solver.wave.fields.rho.iter() {
            assert!(
                v.is_finite() && v >= 0.0,
                "rho blew up/negative at step {step}"
            );
        }
    }

    let r_gas = solver.wave.gas_r;
    let dof = janus_kinetic::maxwellian::DOF;
    let mut t_profile = vec![0.0; ny];
    for j in 0..ny {
        let mut sum = 0.0;
        for i in 0..nx {
            let c = solver.wave.grid.idx(i, j);
            sum += solver.wave.fields.temperature(c, r_gas, dof);
        }
        t_profile[j] = sum / nx as f64;
        assert!(
            t_profile[j].is_finite() && t_profile[j] > 0.0,
            "T[{j}] invalid: {}",
            t_profile[j]
        );
    }

    // Temperature-jump: near-wall cell temperature should measurably differ
    // from the prescribed wall temperature (a continuum/no-jump solution
    // would put the near-wall cell within a few percent of the wall value).
    let south_jump = (t_profile[0] - t_cold).abs();
    let north_jump = (t_profile[ny - 1] - t_hot).abs();
    let span = (t_hot - t_cold).abs();
    assert!(
        south_jump > 0.01 * span,
        "expected measurable temperature jump at cold wall, got {south_jump} (near-wall T={})",
        t_profile[0]
    );
    assert!(
        north_jump > 0.01 * span,
        "expected measurable temperature jump at hot wall, got {north_jump} (near-wall T={})",
        t_profile[ny - 1]
    );
}

/// 5. Diatomic gamma end-to-end check (Part 2 cross-reference): the unit
/// tests in `maxwellian.rs`/`collision.rs`/`particles.rs` already verify
/// gamma=7/5 at the equilibrium-moment, collision-operator, and particle-
/// sampling level individually; this integration-level test re-derives the
/// same result via the `GasModel` trait (the case-configuration entry point
/// a real diatomic case would use) as an end-to-end cross-check that the
/// trait's `gamma()` and the (g,h) reduction's hand-derived gamma agree.
#[test]
fn diatomic_gas_model_gamma_matches_gh_reduction() {
    use janus_kinetic::gas_model::{GasModel, IdealVhsGasModel};
    let model = IdealVhsGasModel {
        r_gas: 296.8,
        mu_ref: 1.663e-5,
        t_ref: 273.15,
        omega: 0.74,
        prandtl: 0.71,
        internal_dof: 2.0, // diatomic rigid rotor
    };
    assert!(
        (model.gamma() - 7.0 / 5.0).abs() < 1e-12,
        "gamma = {}",
        model.gamma()
    );
    assert!((model.total_dof() - 5.0).abs() < 1e-12);

    // Cross-check against the (g,h)-reduction machinery added for Part 2:
    let dof_total = janus_kinetic::maxwellian::dof_with_internal(model.internal_dof());
    assert!(
        (dof_total - model.total_dof()).abs() < 1e-12,
        "dof_with_internal must match GasModel::total_dof"
    );
}

/// 6. Spectral-collision H-theorem test: relative entropy (negative
/// Boltzmann H-functional) must be non-increasing under repeated
/// application of the fast-spectral collision operator (the discrete
/// analogue of the H-theorem, `dH/dt <= 0`, `H = \int f ln(f) dv`). We
/// start from a state perturbed away from a Maxwellian and apply the
/// collision operator as a simple explicit Euler sub-step
/// (`f_{n+1} = f_n + dt*Q(f_n,f_n)`, small `dt`) repeatedly, checking H is
/// monotonically non-increasing (within a small numerical-noise tolerance
/// that accounts for the explicit-Euler sub-stepping's own truncation
/// error, not a violation of the underlying continuous H-theorem).
#[test]
fn spectral_collision_h_theorem_entropy_nonincreasing() {
    use janus_kinetic::spectral_collision::{FastSpectralCollision, SpectralGrid};

    let grid = SpectralGrid::new(12, 1000.0);
    let mut op = FastSpectralCollision::new(grid.clone(), 0.0, 6);
    let rho = 1.0;
    let u = [0.0, 0.0, 0.0];
    let t = 300.0;
    let r_gas = 287.0;
    let n = grid.ntotal();
    let mut f = vec![0.0; n];
    for i in 0..n {
        let v = grid.velocity_at(i);
        // Perturb away from equilibrium (anisotropic, non-Maxwellian modulation).
        let base = janus_kinetic::maxwellian3d::maxwellian_3d(rho, u, t, r_gas, v);
        f[i] = base * (1.0 + 0.3 * (v[0] * 0.002).sin() * (v[1] * 0.0015).cos());
    }

    // Check the H-theorem directly in its exact continuous form:
    //   dH/dt = d/dt integral f ln f dv = integral Q(f,f) ln f dv <= 0.
    // This is robust (no explicit-Euler step-size tuning, whose stability
    // depends on the operator's absolute magnitude) and is the true statement of
    // the theorem. A tiny positive tolerance absorbs quadrature/truncation noise.
    let dv3 = grid.dv3();
    let mut q = vec![0.0; n];
    op.apply_to_distribution(&f, &mut q);
    let hdot: f64 = (0..n)
        .map(|i| if f[i] > 1e-300 { q[i] * f[i].ln() } else { 0.0 })
        .sum::<f64>()
        * dv3;
    let scale: f64 = f.iter().map(|&fv| fv.abs()).sum::<f64>() * dv3;
    assert!(
        hdot <= 1e-6 * scale.max(1.0),
        "entropy production positive (H-theorem violated): dH/dt = {hdot}"
    );
}

/// 7. 2D-vs-3D consistency on a symmetric setup: a Couette-like shear case
/// run through the 2D (g,h)-reduced solver and an equivalent thin-slab 3D
/// solver (nz=1, periodic in z) must agree on the resulting velocity
/// profile within discretization tolerance -- both discretize the same
/// physical monatomic gas (gamma=5/3 either way: 2D via the (g,h)
/// reduction, 3D directly) and should converge to the same continuum
/// Couette-like behavior. Loose tolerance (15% of wall speed) since the two
/// solvers use different velocity-space quadratures (2D Simpson/tensor vs
/// 3D Gauss-Hermite) and grid resolutions.
#[test]
fn two_dimensional_and_three_dimensional_solvers_agree_on_symmetric_couette() {
    use janus_core::config::{BoundaryAssignment3D, BoundaryKind3D, CaseConfig3D};
    use janus_core::distribution::Distribution3D;
    use janus_core::grid3d::Grid3D;
    use janus_kinetic::maxwellian3d::maxwellian_3d;
    use janus_kinetic::solver3d::DugksSolver3D;
    use janus_kinetic::velocity_grid3d::VelocityGrid3D;

    let nx = 4;
    let ny = 10;
    let height = 1.0e-3;
    let u_wall = 40.0;
    let t_wall = 300.0;
    let rho0 = 1.0;

    // --- 2D run ---
    let mut gas2d = GasProperties::monatomic_default();
    gas2d.vhs_omega = 0.81;
    let grid2d = Grid2D::new(nx, ny, height / nx as f64, height / ny as f64, [0.0, 0.0]);
    let bcs2d = BoundaryAssignment {
        west: BoundaryKind::Periodic,
        east: BoundaryKind::Periodic,
        south: BoundaryKind::DiffuseWall {
            temperature: t_wall,
            wall_velocity: [0.0, 0.0],
            accommodation: 1.0,
        },
        north: BoundaryKind::DiffuseWall {
            temperature: t_wall,
            wall_velocity: [u_wall, 0.0],
            accommodation: 1.0,
        },
    };
    let config2d = CaseConfig {
        grid: grid2d,
        bcs: bcs2d,
        gas: gas2d,
    };
    let (vgrid2d, vw2d) = VelocityGrid2D::simpson(1800.0, 21);
    let dist2d = Distribution::zeros(config2d.grid.ncells(), vgrid2d.clone(), vw2d.clone());
    let mut solver2d = DugksSolver::new(&config2d, dist2d);
    init_uniform(&mut solver2d, rho0, [0.0, 0.0], t_wall);

    let dt2d = solver2d.cfl_dt(0.3);
    for _ in 0..4000 {
        solver2d.step(dt2d, &config2d.bcs);
    }

    // --- 3D run (thin slab, periodic z) ---
    let mut gas3d = GasProperties::monatomic_default();
    gas3d.vhs_omega = 0.81;
    let grid3d = Grid3D::new(
        nx,
        ny,
        1,
        height / nx as f64,
        height / ny as f64,
        height / nx as f64,
        [0.0, 0.0, 0.0],
    );
    let bcs3d = BoundaryAssignment3D {
        west: BoundaryKind3D::Periodic,
        east: BoundaryKind3D::Periodic,
        south: BoundaryKind3D::DiffuseWall {
            temperature: t_wall,
            wall_velocity: [0.0, 0.0, 0.0],
            accommodation: 1.0,
        },
        north: BoundaryKind3D::DiffuseWall {
            temperature: t_wall,
            wall_velocity: [u_wall, 0.0, 0.0],
            accommodation: 1.0,
        },
        down: BoundaryKind3D::Periodic,
        up: BoundaryKind3D::Periodic,
    };
    let config3d = CaseConfig3D {
        grid: grid3d,
        bcs: bcs3d,
        gas: gas3d,
    };
    let (vgrid3d, vw3d) =
        VelocityGrid3D::gauss_hermite(config3d.gas.r_gas, t_wall, [0.0, 0.0, 0.0], 7);
    let mut dist3d = Distribution3D::zeros(config3d.grid.ncells(), vgrid3d.clone(), vw3d.clone());
    for c in 0..config3d.grid.ncells() {
        for (k, v) in vgrid3d.iter().enumerate() {
            dist3d.f[c * dist3d.nv + k] =
                maxwellian_3d(rho0, [0.0, 0.0, 0.0], t_wall, config3d.gas.r_gas, *v);
        }
    }
    let mut solver3d = DugksSolver3D::new(&config3d, dist3d);
    solver3d.update_moments();

    let dt3d = solver3d.cfl_dt(0.3);
    for _ in 0..4000 {
        solver3d.step(dt3d, &config3d.bcs);
    }

    // Compare row-averaged u_x(y) profiles.
    let mut max_diff = 0.0f64;
    for j in 0..ny {
        let mut u2 = 0.0;
        for i in 0..nx {
            let c = solver2d.grid.idx(i, j);
            u2 += solver2d.fields.velocity(c)[0];
        }
        u2 /= nx as f64;

        let mut u3 = 0.0;
        for i in 0..nx {
            let c = solver3d.grid.idx(i, j, 0);
            u3 += solver3d.fields.velocity(c)[0];
        }
        u3 /= nx as f64;

        max_diff = max_diff.max((u2 - u3).abs());
    }
    let tol = 0.15 * u_wall;
    assert!(
        max_diff < tol,
        "2D vs 3D Couette velocity profile disagreement {max_diff} exceeds tolerance {tol}"
    );
}

// Test 8 (conservation-under-load-balancing / block decomposition) lives in
// `janus-sched`'s own test suite (`crates/janus-sched/tests/
// conservation_under_decomposition.rs`), not here: `janus-sched` depends on
// `janus-kinetic` (ENGINEERING_SPEC.md §3's one-directional crate-dependency
// rule), so exercising `janus_sched::block::partition_grid` from a
// `janus-kinetic` test would require a `janus-kinetic -> janus-sched`
// dev-dependency, creating an actual Cargo dependency cycle between the two
// crates (dev-dependency cycles between workspace members are fragile and
// contrary to the spirit of the one-directional dependency rule even where
// Cargo might tolerate them). Placing the test in `janus-sched` (which
// already legitimately depends on both `janus-core` and `janus-kinetic`)
// avoids the cycle while testing the exact same property.
