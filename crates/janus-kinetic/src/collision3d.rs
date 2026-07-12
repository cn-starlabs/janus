//! Collision operator for the full 3D-velocity-space DVM (single
//! distribution `f`, no `(g,h)` reduction — see `maxwellian3d` module docs).
//!
//! Reference: Shakhov, E. M., "Generalization of the Krook kinetic
//! relaxation equation", Fluid Dynamics 3, 95-96 (1968).
//!
//! The Shakhov correction here is the standard (non-reduced) 3D form —
//! simpler than the 2D solver's (g,h)-aware version since there is only one
//! distribution and `DOF` is directly `3` (no split bracket terms for a
//! second reduced distribution):
//!
//! ```text
//! f_eq^S = f_eq * [1 + (1-Pr) * (c.q)/(5*p*R*T) * (c^2/(R*T) - 5)]
//! ```
//!
//! (the standard monatomic-gas Shakhov correction, `DOF+2 = 5`, matching
//! e.g. Shakhov 1968 / the standard ES-BGK-adjacent literature form for a
//! full 3D velocity space).

use crate::maxwellian3d::{maxwellian_3d, DOF};

/// A local collision (relaxation) operator for the full 3D single-
/// distribution DVM: given the local macroscopic state, produce the target
/// equilibrium `f_eq` at velocity node `v`, and the relaxation time `tau`.
pub trait Collision3D {
    /// Equilibrium (post-collision target) `f_eq` at discrete velocity `v`
    /// for the given macro state and heat flux `q = (qx, qy, qz)`.
    fn equilibrium(
        &self,
        rho: f64,
        u: [f64; 3],
        t: f64,
        r_gas: f64,
        q: [f64; 3],
        v: [f64; 3],
    ) -> f64;

    /// Relaxation time `tau = mu / p` (VHS viscosity law), shared with the
    /// 2D solver's identical construction.
    fn relaxation_time(
        &self,
        rho: f64,
        t: f64,
        r_gas: f64,
        mu_ref: f64,
        t_ref: f64,
        omega: f64,
    ) -> f64 {
        let mu = mu_ref * (t / t_ref).powf(omega);
        let p = rho * r_gas * t;
        mu / p.max(f64::MIN_POSITIVE)
    }
}

/// Shakhov S-model collision operator, full 3D velocity space, single
/// distribution.
pub struct Shakhov3D {
    /// Prandtl number (monatomic gas: 2/3).
    pub pr: f64,
}

impl Shakhov3D {
    pub fn new(pr: f64) -> Self {
        Self { pr }
    }
}

impl Collision3D for Shakhov3D {
    fn equilibrium(
        &self,
        rho: f64,
        u: [f64; 3],
        t: f64,
        r_gas: f64,
        q: [f64; 3],
        v: [f64; 3],
    ) -> f64 {
        let f_m = maxwellian_3d(rho, u, t, r_gas, v);
        let rt = r_gas * t;
        let cx = v[0] - u[0];
        let cy = v[1] - u[1];
        let cz = v[2] - u[2];
        let c2 = cx * cx + cy * cy + cz * cz;
        let p = rho * rt;
        let cdotq = cx * q[0] + cy * q[1] + cz * q[2];

        // Standard monatomic Shakhov bracket, DOF+2 = 5:
        let bracket = c2 / rt - (DOF + 2.0);
        let s = (1.0 - self.pr) * cdotq / (5.0 * p * rt) * bracket;

        f_m * (1.0 + s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maxwellian3d::maxwellian_3d;
    use crate::velocity_grid3d::VelocityGrid3D;

    #[test]
    fn bgk_limit_pr_one_matches_maxwellian() {
        let s = Shakhov3D::new(1.0);
        let rho = 1.0;
        let u = [0.1, 0.0, 0.0];
        let t = 300.0;
        let r_gas = 287.0;
        let q = [10.0, -5.0, 2.0];
        let v = [50.0, -30.0, 10.0];
        let f_eq = s.equilibrium(rho, u, t, r_gas, q, v);
        let f_m = maxwellian_3d(rho, u, t, r_gas, v);
        assert!((f_eq - f_m).abs() < 1e-12);
    }

    #[test]
    fn shakhov_zero_heat_flux_recovers_mass() {
        let s = Shakhov3D::new(2.0 / 3.0);
        let rho = 1.0;
        let u = [0.0, 0.0, 0.0];
        let t = 300.0;
        let r_gas = 287.0;
        let (vgrid, vw) = VelocityGrid3D::gauss_hermite(r_gas, t, u, 8);
        let mut m_rho = 0.0;
        for (k, v) in vgrid.iter().enumerate() {
            let f = s.equilibrium(rho, u, t, r_gas, [0.0, 0.0, 0.0], *v);
            m_rho += vw[k] * f;
        }
        assert!((m_rho - rho).abs() / rho < 1e-6);
    }

    #[test]
    fn relaxation_time_positive() {
        let s = Shakhov3D::new(2.0 / 3.0);
        let tau = s.relaxation_time(1.0, 300.0, 287.0, 2.117e-5, 273.15, 0.81);
        assert!(tau > 0.0);
    }
}
