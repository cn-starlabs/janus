//! Couette flow validation: two parallel diffuse walls, one stationary, one
//! moving tangentially. At steady state the Navier-Stokes solution is the
//! linear velocity profile `u(y) = U * y / H`. We run the DUGKS solver to a
//! long time and compare the resulting u_x(y) profile against that analytic
//! line within a loose tolerance appropriate for a kinetic/DVM solver on a
//! coarse grid (finite Knudsen-number slip effects near the wall, coarse
//! velocity-space quadrature, and finite integration time all introduce
//! deviation from the ideal continuum NS profile).

use janus_core::config::{BoundaryAssignment, BoundaryKind, CaseConfig, GasProperties};
use janus_core::distribution::Distribution;
use janus_core::grid::Grid2D;
use janus_kinetic::maxwellian::gh_equilibrium;
use janus_kinetic::solver::DugksSolver;
use janus_kinetic::velocity_grid::VelocityGrid2D;

// NOTE: This is an expensive steady-state validation. Reaching steady Couette
// takes O(diffusion_time / dt) steps, and dt is bounded by the CFL condition on
// the largest velocity ordinate (~1800 m/s) while the flow evolves on the much
// slower viscous-diffusion timescale — the classic stiffness of an explicit
// kinetic scheme, giving tens of thousands of steps over a 64-cell x 625-node
// grid. That is far too slow for a routine debug-mode `cargo test` run (it does
// not hang; it is simply ~10^13 node-updates). It is therefore marked
// `#[ignore]` and should be run explicitly, ideally in release:
//   cargo test -p janus-kinetic --release --test couette -- --ignored
#[test]
#[ignore = "expensive steady-state validation; run explicitly with --release --ignored"]
fn couette_linear_velocity_profile() {
    let nx = 4; // periodic in x, coarse (flow is uniform in x)
    let ny = 16;
    let height = 1.0e-3; // 1 mm gap
    let u_wall = 50.0; // m/s, small enough to stay firmly subsonic/near-continuum
    let t_wall = 300.0;

    let mut gas = GasProperties::monatomic_default();
    // Increase reference viscosity somewhat isn't needed; use defaults.
    // Use a denser gas (higher rho) via higher reference pressure implicitly
    // through initial condition below; gas properties only set mu law.
    gas.vhs_omega = 0.81;

    let grid = Grid2D::new(nx, ny, height / nx as f64, height / ny as f64, [0.0, 0.0]);
    let bcs = BoundaryAssignment {
        west: BoundaryKind::Periodic,
        east: BoundaryKind::Periodic,
        south: BoundaryKind::DiffuseWall {
            temperature: t_wall,
            wall_velocity: [0.0, 0.0],
        },
        north: BoundaryKind::DiffuseWall {
            temperature: t_wall,
            wall_velocity: [u_wall, 0.0],
        },
    };
    let config = CaseConfig { grid, bcs, gas };

    let (vgrid, vw) = VelocityGrid2D::simpson(1800.0, 25);
    let mut dist = Distribution::zeros(config.grid.ncells(), vgrid.clone(), vw.clone());

    let rho0 = 1.0;
    for c in 0..config.grid.ncells() {
        for (k, v) in vgrid.iter().enumerate() {
            let (g, h) = gh_equilibrium(rho0, [0.0, 0.0], t_wall, config.gas.r_gas, *v);
            dist.f[c * dist.nv + k] = g;
            dist.h[c * dist.nv + k] = h;
        }
    }

    let mut solver = DugksSolver::new(&config, dist);
    solver.update_moments();

    let dt = solver.cfl_dt(0.3);
    // Diffusive timescale ~ H^2 / nu; run comfortably past it. nu = mu/rho.
    let mu = janus_core::units::vhs_viscosity(
        t_wall,
        config.gas.mu_ref,
        config.gas.t_ref,
        config.gas.vhs_omega,
    );
    let nu = mu / rho0;
    let diffusion_time = height * height / nu;
    let n_steps = ((diffusion_time * 6.0) / dt).ceil() as u64;
    // Cap step count for test runtime; if the estimated requirement is huge
    // (rarefied gas -> large nu), clamp and rely on the loose tolerance.
    let n_steps = n_steps.min(400_000).max(2000);

    for _ in 0..n_steps {
        solver.step(dt, &config.bcs);
    }

    // Compare u_x(y) at each row (averaged over x, which is periodic/uniform)
    // against the analytic profile u(y) = U * y_center / H.
    let mut max_err = 0.0f64;
    for j in 0..ny {
        let mut ux_sum = 0.0;
        for i in 0..nx {
            let c = solver.grid.idx(i, j);
            ux_sum += solver.fields.velocity(c)[0];
        }
        let ux = ux_sum / nx as f64;
        let y_center = (j as f64 + 0.5) * solver.grid.dy;
        let analytic = u_wall * y_center / height;
        let err = (ux - analytic).abs() / u_wall;
        max_err = max_err.max(err);
    }

    // Loose tolerance: DVM/DUGKS on a coarse grid + finite run time + slip
    // effects at walls. 15% of wall speed is generous but meaningful.
    let tol = 0.15;
    assert!(
        max_err < tol,
        "Couette profile max relative error {max_err} exceeds tolerance {tol}"
    );
}
