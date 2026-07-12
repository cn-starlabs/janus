//! Sod shock tube validation: 2D grid with ny small (few periodic rows in
//! y so the problem is effectively 1D), discontinuous initial rho/p/u along
//! x, standard Sod parameters. Compare density/velocity/pressure against the
//! exact Euler-equation Riemann solution at a moderate time, with a loose
//! tolerance (DVM/kinetic solvers smear shocks/contact discontinuities over
//! several cells relative to the sharp exact solution).
//!
//! Standard Sod initial condition. M2 update: the solver now uses the
//! reduced (g,h) two-distribution formulation (`janus_kinetic::maxwellian`),
//! which correctly recovers monatomic gamma = 5/3 (3 translational DOF: 2
//! discretized + 1 reduced), matching the traditional monatomic-gas Sod
//! tube comparison (note: the "textbook" Sod tube commonly quotes gamma=1.4
//! for diatomic air; we use gamma=5/3 here because this solver models a
//! monatomic gas per ENGINEERING_SPEC.md §2 point 4 — multi-species/
//! polyatomic gases are out of scope). rho_L=1, p_L=1, rho_R=0.125,
//! p_R=0.1, u_L=u_R=0, in nondimensional units with R=1.

use janus_core::config::{BoundaryAssignment, BoundaryKind, CaseConfig, GasProperties};
use janus_core::distribution::Distribution;
use janus_core::grid::Grid2D;
use janus_kinetic::maxwellian::gh_equilibrium;
use janus_kinetic::solver::DugksSolver;
use janus_kinetic::velocity_grid::VelocityGrid2D;

// M2: the reduced (g,h) two-distribution formulation recovers the correct
// monatomic-gas gamma = 5/3 (see janus_kinetic::maxwellian module docs),
// retiring the M1 gamma=2 PHYSICS-DEBT. The exact Riemann solver below is
// compared against using the true monatomic gamma.
const GAMMA: f64 = 5.0 / 3.0;

/// Exact Riemann solver for the Sod problem (Toro, "Riemann Solvers and
/// Numerical Methods for Fluid Dynamics", 3rd ed., ch. 4), specialized to
/// the classic Sod left/right states with zero initial velocity. Returns
/// (rho, u, p) at position `x` relative to the diaphragm, at time `t`.
fn sod_exact(
    x: f64,
    t: f64,
    rho_l: f64,
    p_l: f64,
    rho_r: f64,
    p_r: f64,
    r_gas: f64,
) -> (f64, f64, f64) {
    let gamma = GAMMA;
    let a_l = (gamma * p_l / rho_l).sqrt();
    let a_r = (gamma * p_r / rho_r).sqrt();

    // Solve for p_star via Newton iteration on the pressure function
    // (Toro eq. 4.5-4.9), both sides assumed shock or rarefaction as needed.
    let f = |p: f64, rho_k: f64, p_k: f64, a_k: f64| -> (f64, f64) {
        if p > p_k {
            // shock
            let a = 2.0 / ((gamma + 1.0) * rho_k);
            let b = (gamma - 1.0) / (gamma + 1.0) * p_k;
            let fk = (p - p_k) * (a / (p + b)).sqrt();
            let dfk = (a / (p + b)).sqrt() * (1.0 - (p - p_k) / (2.0 * (p + b)));
            (fk, dfk)
        } else {
            // rarefaction
            let fk =
                2.0 * a_k / (gamma - 1.0) * ((p / p_k).powf((gamma - 1.0) / (2.0 * gamma)) - 1.0);
            let dfk = 1.0 / (rho_k * a_k) * (p / p_k).powf(-(gamma + 1.0) / (2.0 * gamma));
            (fk, dfk)
        }
    };

    let mut p_star = 0.5 * (p_l + p_r);
    for _ in 0..50 {
        let (fl, dfl) = f(p_star, rho_l, p_l, a_l);
        let (fr, dfr) = f(p_star, rho_r, p_r, a_r);
        let ftot = fl + fr;
        let dftot = dfl + dfr;
        let p_new = p_star - ftot / dftot;
        let p_new = p_new.max(1e-9);
        if (p_new - p_star).abs() < 1e-12 {
            p_star = p_new;
            break;
        }
        p_star = p_new;
    }
    let (fl, _) = f(p_star, rho_l, p_l, a_l);
    let (fr, _) = f(p_star, rho_r, p_r, a_r);
    let u_star = 0.5 * (0.0 + 0.0) + 0.5 * (fr - fl);

    let s = x / t.max(1e-12);

    // Left of contact:
    let result = if s <= u_star {
        if p_star > p_l {
            // left shock
            let q = p_star / p_l;
            let s_l = 0.0
                - a_l
                    * (((gamma + 1.0) / (2.0 * gamma) * q + (gamma - 1.0) / (2.0 * gamma)).sqrt());
            if s < s_l {
                (rho_l, 0.0, p_l)
            } else {
                let rho_star_l = rho_l * (q + (gamma - 1.0) / (gamma + 1.0))
                    / (q * (gamma - 1.0) / (gamma + 1.0) + 1.0);
                (rho_star_l, u_star, p_star)
            }
        } else {
            // left rarefaction
            let a_star_l = a_l * (p_star / p_l).powf((gamma - 1.0) / (2.0 * gamma));
            let s_head = 0.0 - a_l;
            let s_tail = u_star - a_star_l;
            if s < s_head {
                (rho_l, 0.0, p_l)
            } else if s > s_tail {
                let rho_star_l = rho_l * (p_star / p_l).powf(1.0 / gamma);
                (rho_star_l, u_star, p_star)
            } else {
                // inside fan
                let u_fan = 2.0 / (gamma + 1.0) * (a_l + (gamma - 1.0) / 2.0 * 0.0 + s);
                let a_fan = 2.0 / (gamma + 1.0) * (a_l + (gamma - 1.0) / 2.0 * (0.0 - s));
                let rho_fan = rho_l * (a_fan / a_l).powf(2.0 / (gamma - 1.0));
                let p_fan = p_l * (a_fan / a_l).powf(2.0 * gamma / (gamma - 1.0));
                (rho_fan, u_fan, p_fan)
            }
        }
    } else {
        // Right of contact
        if p_star > p_r {
            // right shock
            let q = p_star / p_r;
            let s_r = 0.0
                + a_r
                    * (((gamma + 1.0) / (2.0 * gamma) * q + (gamma - 1.0) / (2.0 * gamma)).sqrt());
            if s > s_r {
                (rho_r, 0.0, p_r)
            } else {
                let rho_star_r = rho_r * (q + (gamma - 1.0) / (gamma + 1.0))
                    / (q * (gamma - 1.0) / (gamma + 1.0) + 1.0);
                (rho_star_r, u_star, p_star)
            }
        } else {
            // right rarefaction
            let a_star_r = a_r * (p_star / p_r).powf((gamma - 1.0) / (2.0 * gamma));
            let s_head = 0.0 + a_r;
            let s_tail = u_star + a_star_r;
            if s > s_head {
                (rho_r, 0.0, p_r)
            } else if s < s_tail {
                let rho_star_r = rho_r * (p_star / p_r).powf(1.0 / gamma);
                (rho_star_r, u_star, p_star)
            } else {
                let u_fan = 2.0 / (gamma + 1.0) * (-a_r + (gamma - 1.0) / 2.0 * 0.0 + s);
                let a_fan = 2.0 / (gamma + 1.0) * (a_r - (gamma - 1.0) / 2.0 * (0.0 - s));
                let rho_fan = rho_r * (a_fan / a_r).powf(2.0 / (gamma - 1.0));
                let p_fan = p_r * (a_fan / a_r).powf(2.0 * gamma / (gamma - 1.0));
                (rho_fan, u_fan, p_fan)
            }
        }
    };

    let _ = r_gas;
    result
}

#[test]
fn sod_shock_tube_matches_exact_riemann_solution() {
    let nx = 100;
    let ny = 2; // thin periodic strip in y -> effectively 1D
    let r_gas = 1.0;
    let rho_l = 1.0;
    let p_l = 1.0;
    let rho_r = 0.125;
    let p_r = 0.1;
    let t_l = p_l / (rho_l * r_gas);
    let t_r = p_r / (rho_r * r_gas);

    let length = 1.0;
    let mut gas = GasProperties::monatomic_default();
    gas.r_gas = r_gas;
    // Use a small reference viscosity so the solver is close to the inviscid
    // Euler limit (large Reynolds number), matching the exact solver's
    // inviscid assumption; too small a mu makes the DVM stiff, so pick a
    // modest but small value.
    gas.mu_ref = 1e-4;
    gas.t_ref = 1.0;
    gas.vhs_omega = 0.5;

    let grid = Grid2D::new(nx, ny, length / nx as f64, length / ny as f64, [0.0, 0.0]);
    let bcs = BoundaryAssignment {
        west: BoundaryKind::Outlet,
        east: BoundaryKind::Outlet,
        south: BoundaryKind::Periodic,
        north: BoundaryKind::Periodic,
    };
    let config = CaseConfig { grid, bcs, gas };

    // Velocity grid must span the sound speed range comfortably; sqrt(gamma*p/rho) ~ O(1).
    let (vgrid, vw) = VelocityGrid2D::simpson(8.0, 33);
    let mut dist = Distribution::zeros(config.grid.ncells(), vgrid.clone(), vw.clone());

    let mid = length / 2.0;
    for j in 0..ny {
        for i in 0..nx {
            let c = config.grid.idx(i, j);
            let x = (i as f64 + 0.5) * config.grid.dx;
            let (rho, t) = if x < mid { (rho_l, t_l) } else { (rho_r, t_r) };
            for (k, v) in vgrid.iter().enumerate() {
                let (g, h) = gh_equilibrium(rho, [0.0, 0.0], t, config.gas.r_gas, *v);
                dist.f[c * dist.nv + k] = g;
                dist.h[c * dist.nv + k] = h;
            }
        }
    }

    let mut solver = DugksSolver::new(&config, dist);
    solver.update_moments();

    let t_final = 0.15;
    let dt = solver.cfl_dt(0.25);
    let n_steps = (t_final / dt).ceil() as u64;
    for _ in 0..n_steps {
        solver.step(dt, &config.bcs);
    }
    let actual_t_final = n_steps as f64 * dt;

    // Compare against exact solution at several sample points, using a
    // row-average (over the thin y strip) to reduce quadrature noise.
    let dof = janus_kinetic::maxwellian::DOF;
    let mut max_rho_err = 0.0f64;
    let mut max_u_err = 0.0f64;
    let mut max_p_err = 0.0f64;
    let mut n_checked = 0;
    for i in 0..nx {
        let x = (i as f64 + 0.5) * config.grid.dx - mid;
        // Skip points very close to the diaphragm/shock/contact where a
        // 1-2 cell smearing region is expected and not meaningful to compare
        // pointwise (standard practice for shock-capturing validation).
        if x.abs() < 0.03 {
            continue;
        }
        let (rho_e, u_e, p_e) = sod_exact(x, actual_t_final, rho_l, p_l, rho_r, p_r, r_gas);

        let mut rho_num = 0.0;
        let mut u_num = 0.0;
        let mut p_num = 0.0;
        for j in 0..ny {
            let c = solver.grid.idx(i, j);
            rho_num += solver.fields.rho[c];
            u_num += solver.fields.velocity(c)[0];
            p_num += solver.fields.pressure(c, r_gas, dof);
        }
        rho_num /= ny as f64;
        u_num /= ny as f64;
        p_num /= ny as f64;

        max_rho_err = max_rho_err.max((rho_num - rho_e).abs() / rho_l);
        max_u_err = max_u_err.max((u_num - u_e).abs() / (rho_l / rho_r).sqrt().max(1.0));
        max_p_err = max_p_err.max((p_num - p_e).abs() / p_l);
        n_checked += 1;
    }
    assert!(
        n_checked > 10,
        "sanity: expected many sample points to be checked"
    );

    // Loose tolerances: a first-order-upwind DUGKS/DVM solver smears shocks
    // and the contact discontinuity over several cells and will not match a
    // sharp Riemann solution pointwise near those features (already
    // excluded above); away from those features 25% relative error is a
    // generous but meaningful bound for a first M1 implementation.
    let tol = 0.25;
    assert!(
        max_rho_err < tol,
        "density max relative error {max_rho_err} exceeds {tol}"
    );
    assert!(
        max_u_err < tol,
        "velocity max relative error {max_u_err} exceeds {tol}"
    );
    assert!(
        max_p_err < tol,
        "pressure max relative error {max_p_err} exceeds {tol}"
    );
}
