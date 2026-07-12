//! Full 3D-velocity-space Maxwellian equilibrium — single distribution
//! function `f(x, v, t)`, `v = (vx, vy, vz)`.
//!
//! ## Why the 2D-space (g,h) reduction is dropped here (not a shortcut)
//!
//! `janus_kinetic::maxwellian` (the 2D solver's module) integrates the true
//! 3D distribution `f(x, v, eta, t)` over a *discretized-out* third velocity
//! component `eta` analytically, producing the reduced pair `(g, h)`. That
//! reduction exists purely because a 2D *velocity-space* DVM (only `(vx,
//! vy)` discretized) is otherwise short one translational degree of freedom
//! for a monatomic gas: without it, the discrete equilibrium only carries
//! `D=2` DOF (giving the wrong `gamma=2` instead of the physical `5/3`), so
//! `h` is introduced as a bookkeeping device to carry the missing DOF's
//! energy without ever discretizing a third velocity axis (Xu & Huang 2010,
//! Chu 1965).
//!
//! For the M4 **3D** solver, physical space is 3D AND velocity space is
//! genuinely 3-component (`vx, vy, vz` are all discretized, see
//! `velocity_grid3d::VelocityGrid3D`). All 3 translational degrees of
//! freedom of a monatomic gas are therefore *already* directly represented
//! by the discrete velocity grid itself — there is nothing left to reduce
//! out, and no missing DOF to patch with an auxiliary `h`-like carrier.
//! Introducing a (g,h)-style reduction here would be redundant (it would
//! attempt to recover a DOF that the 3-component velocity grid already
//! supplies) and would incorrectly inflate the gas's effective heat
//! capacity. Consequently:
//!
//! - `DOF = 3` (all three components are *discretized*, `D_DISCRETE = 3`,
//!   `K_REDUCED = 0`), giving the correct monatomic `gamma = (3+2)/3 = 5/3`
//!   directly from the physical velocity moments, with no auxiliary
//!   distribution.
//! - The solver in `solver3d.rs` carries a single `Distribution3D::f` per
//!   cell (not an `(f,h)` pair), and every moment (mass, momentum, energy,
//!   heat flux) is computed directly from `f` and the 3-component velocity
//!   grid, exactly the way a *true*, non-reduced DVM computes them.
//!
//! This is a legitimate simplification relative to the 2D code — not a
//! shortcut or debt item — because the reduction it removes was itself only
//! ever a workaround for a velocity-space dimensionality deficit that does
//! not exist in 3D. No `// PHYSICS-DEBT:` marker is warranted for this
//! decision.
//!
//! References:
//! - Xu, K., Huang, J.-C., "A unified gas-kinetic scheme for continuum and
//!   rarefied flows", J. Comput. Phys. 229, 7747-7764 (2010) — the (g,h)
//!   reduction this module's docs explain the *non-need* for in 3D.
//! - Shakhov, E. M., "Generalization of the Krook kinetic relaxation
//!   equation", Fluid Dynamics 3, 95 (1968) — the collision model (see
//!   `collision3d.rs`) still applies directly to the single `f`.

/// Discretized translational DOF in the 3D velocity grid: all 3 components
/// (`vx, vy, vz`) are discretized directly, so `D_DISCRETE = 3` and there is
/// no reduced/analytically-integrated-out component (`K_REDUCED = 0`).
pub const D_DISCRETE: f64 = 3.0;

/// No reduced DOF in the full 3D formulation (see module docs).
pub const K_REDUCED: f64 = 0.0;

/// Total effective DOF: `D + K = 3`, giving the correct monatomic
/// `gamma = 5/3` directly (no reduction bookkeeping needed).
pub const DOF: f64 = D_DISCRETE + K_REDUCED;

/// Ratio of specific heats: `5/3` for a monatomic gas, recovered directly
/// from the full 3-component discrete velocity space.
pub const GAMMA: f64 = (DOF + 2.0) / DOF;

/// Evaluate the 3D Maxwellian
/// `M(v) = rho / (2*pi*R*T)^{3/2} * exp(-|v-u|^2 / (2*R*T))`
/// at discrete velocity `v` given macroscopic state `(rho, u, T)` and gas
/// constant `r_gas`.
#[inline]
pub fn maxwellian_3d(rho: f64, u: [f64; 3], t: f64, r_gas: f64, v: [f64; 3]) -> f64 {
    let rt = r_gas * t;
    let norm = rho / (2.0 * std::f64::consts::PI * rt).powf(1.5);
    let dvx = v[0] - u[0];
    let dvy = v[1] - u[1];
    let dvz = v[2] - u[2];
    let exponent = -(dvx * dvx + dvy * dvy + dvz * dvz) / (2.0 * rt);
    norm * exponent.exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_is_five_thirds() {
        assert!((GAMMA - 5.0 / 3.0).abs() < 1e-12, "gamma = {GAMMA}");
    }

    #[test]
    fn maxwellian_moments_recover_macro_state() {
        use crate::velocity_grid3d::VelocityGrid3D;
        let rho = 1.2;
        let u = [0.3, -0.1, 0.2];
        let t = 300.0;
        let r_gas = 287.0;
        let (vgrid, vw) = VelocityGrid3D::gauss_hermite(r_gas, t, u, 8);

        let mut m_rho = 0.0;
        let mut m_mom = [0.0; 3];
        let mut m_e = 0.0;
        for (k, v) in vgrid.iter().enumerate() {
            let g = maxwellian_3d(rho, u, t, r_gas, *v);
            let w = vw[k] * g;
            m_rho += w;
            m_mom[0] += w * v[0];
            m_mom[1] += w * v[1];
            m_mom[2] += w * v[2];
            let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            m_e += 0.5 * w * v2;
        }
        assert!((m_rho - rho).abs() / rho < 1e-6, "rho {m_rho}");
        assert!((m_mom[0] - rho * u[0]).abs() < 1e-3, "mx {m_mom:?}");
        assert!((m_mom[1] - rho * u[1]).abs() < 1e-3, "my {m_mom:?}");
        assert!((m_mom[2] - rho * u[2]).abs() < 1e-3, "mz {m_mom:?}");

        let kinetic = 0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        let internal = 0.5 * rho * DOF * r_gas * t;
        let expected_e = kinetic + internal;
        assert!(
            (m_e - expected_e).abs() / expected_e < 1e-3,
            "E {m_e} vs {expected_e}"
        );
    }
}
