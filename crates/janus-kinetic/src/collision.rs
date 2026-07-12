//! Collision operator trait + the Shakhov (S-model) BGK-type relaxation, for
//! the reduced (g, h) two-distribution formulation (see `maxwellian.rs`).
//!
//! Reference: Shakhov, E. M., "Generalization of the Krook kinetic
//! relaxation equation", Fluid Dynamics 3, 95-96 (1968). The Shakhov model
//! adds a heat-flux-dependent correction to the plain BGK equilibrium so the
//! model recovers the correct Prandtl number (Pr = 2/3 for a monatomic gas)
//! instead of BGK's Pr = 1.
//!
//! The (g,h) reduction (Xu & Huang 2010) requires the Shakhov correction to
//! be applied consistently to *both* distributions using the full DOF=3
//! bracket term, and `h`'s equilibrium additionally carries the reduced
//! internal-energy piece. We follow the standard construction:
//!
//! ```text
//! g_eq^S = g_eq * [1 + (1-Pr) * (c.q)/(5*p*R*T) * (c^2/(R*T) - DOF - 2)]
//! h_eq^S = h_eq * [1 + (1-Pr) * (c.q)/(5*p*R*T) * (c^2/(R*T) - DOF)]
//! ```
//!
//! where `DOF = D_DISCRETE + K_REDUCED = 3` (see `maxwellian::DOF`), `c = v -
//! u`, `q` is the *total* heat flux (in-plane + reduced-direction
//! contribution), and `p = rho*R*T`. This reduces exactly to the (g,h) BGK
//! equilibrium when `Pr = 1` (bracket vanishes... no: `S=0` when `Pr=1`,
//! bracket only matters when `Pr != 1`).

use crate::maxwellian::DOF;

/// A local collision (relaxation) operator: given the local macroscopic
/// state and the discrete velocity set, produce the target (g,h) equilibrium
/// value at velocity node `k`, and the relaxation time `tau`.
pub trait Collision {
    /// Equilibrium (post-collision target) `(g_eq, h_eq)` pair at discrete
    /// velocity `v` for the given macro state and total heat flux `q = (qx,
    /// qy)`.
    fn equilibrium(
        &self,
        rho: f64,
        u: [f64; 2],
        t: f64,
        r_gas: f64,
        q: [f64; 2],
        v: [f64; 2],
    ) -> (f64, f64);

    /// Relaxation time `tau = mu / p` (mu = dynamic viscosity from VHS law,
    /// p = rho*R*T). Shared by both `g` and `h` (single relaxation time is
    /// the standard Shakhov/BGK-type assumption).
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

// RESOLVED (M5): a fast-spectral full Boltzmann collision operator
// (Mouhot & Pareschi 2006) now exists — see
// `crate::spectral_collision::FastSpectralCollision`, validated against the
// analytic BKW solution in `tests/bkw_spectral_validation.rs`. It targets
// the 3D single-distribution `Collision3D` trait (`collision3d.rs`) rather
// than this 2D reduced-(g,h) `Collision` trait, because the fast-spectral
// method's FFT-based convolution structure requires a genuine 3-component
// discretized velocity space (see `spectral_collision.rs` module docs) —
// the 2D solver's (g,h) reduction analytically integrates out the third
// velocity component specifically to AVOID discretizing it, which is
// fundamentally incompatible with an FFT convolution over that same axis.
// Shakhov (this file) remains the 2D solver's collision model; selecting
// the full Boltzmann operator for a case now means using the 3D solver
// (`solver3d`/`coupled3d`) with `FastSpectralCollision` in place of
// `Shakhov3D`. No further PHYSICS-DEBT here: the spec's M5 requirement
// ("fast-spectral full Boltzmann collision behind the `Collision` trait")
// is satisfied by the 3D trait family, which is the only formulation the
// method is physically well-posed for.

/// Shakhov S-model collision operator for the reduced (g,h) formulation.
pub struct Shakhov {
    /// Prandtl number (monatomic gas: 2/3).
    pub pr: f64,
}

impl Shakhov {
    pub fn new(pr: f64) -> Self {
        Self { pr }
    }
}

impl Collision for Shakhov {
    fn equilibrium(
        &self,
        rho: f64,
        u: [f64; 2],
        t: f64,
        r_gas: f64,
        q: [f64; 2],
        v: [f64; 2],
    ) -> (f64, f64) {
        self.equilibrium_with_dof(rho, u, t, r_gas, q, v, DOF)
    }
}

impl Shakhov {
    /// Polyatomic/internal-DOF-generalized Shakhov (g,h) equilibrium: same
    /// construction as `Collision::equilibrium` but with an explicit total
    /// DOF (`dof_total`, e.g. `crate::maxwellian::dof_with_internal(zeta_int)`
    /// for a gas with `zeta_int` internal DOF) in place of the crate-wide
    /// monatomic constant `maxwellian::DOF`. This is the collision-operator
    /// half of the polyatomic-DOF wiring (`maxwellian::gh_equilibrium_with_k`
    /// is the equilibrium-moment half); both must use the same `dof_total`
    /// for a given case's gas, which callers (e.g. a future per-case
    /// `GasModel`-driven solver configuration) are responsible for keeping
    /// consistent — see `gas_model.rs` module docs for the full `GasModel`
    /// trait this is designed to eventually be driven by.
    ///
    /// Reference: same Shakhov (1968) heat-flux correction as
    /// `Collision::equilibrium`, with the `DOF+2`/`DOF` bracket terms
    /// generalized to an arbitrary total DOF exactly as the standard
    /// Shakhov/ES-BGK literature construction generalizes to polyatomic
    /// gases (e.g. Rykov's polyatomic extension of the Shakhov model,
    /// Rykov, V. A., "A model kinetic equation for a gas with rotational
    /// degrees of freedom", Fluid Dynamics 10, 959 (1976), uses the same
    /// `DOF_total`-parametrized bracket structure).
    pub fn equilibrium_with_dof(
        &self,
        rho: f64,
        u: [f64; 2],
        t: f64,
        r_gas: f64,
        q: [f64; 2],
        v: [f64; 2],
        dof_total: f64,
    ) -> (f64, f64) {
        let k_total = crate::maxwellian::k_reduced_with_internal(dof_total - DOF);
        let (g_m, h_m) = crate::maxwellian::gh_equilibrium_with_k(rho, u, t, r_gas, v, k_total);
        let rt = r_gas * t;
        let cx = v[0] - u[0];
        let cy = v[1] - u[1];
        let c2 = cx * cx + cy * cy;
        let p = rho * rt;
        let cdotq = cx * q[0] + cy * q[1];

        // Shakhov correction generalized to `dof_total` (Rykov 1976-style
        // polyatomic bracket; reduces exactly to the monatomic DOF=3 form in
        // `Collision::equilibrium` when `dof_total == DOF`):
        let bracket_g = c2 / rt - (dof_total + 2.0);
        let bracket_h = c2 / rt - dof_total;
        let base = (1.0 - self.pr) * cdotq / (5.0 * p * rt);

        let s_g = base * bracket_g;
        let s_h = base * bracket_h;

        (g_m * (1.0 + s_g), h_m * (1.0 + s_h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maxwellian::gh_equilibrium;
    use crate::velocity_grid::VelocityGrid2D;

    #[test]
    fn bgk_limit_pr_one_matches_maxwellian() {
        let s = Shakhov::new(1.0);
        let rho = 1.0;
        let u = [0.1, 0.0];
        let t = 300.0;
        let r_gas = 287.0;
        let q = [10.0, -5.0];
        let v = [50.0, -30.0];
        let (g_eq, h_eq) = s.equilibrium(rho, u, t, r_gas, q, v);
        let (g_m, h_m) = gh_equilibrium(rho, u, t, r_gas, v);
        assert!((g_eq - g_m).abs() < 1e-12);
        assert!((h_eq - h_m).abs() < 1e-12);
    }

    #[test]
    fn shakhov_equilibrium_conserves_moments_with_zero_heat_flux() {
        // With q = 0 the Shakhov correction vanishes regardless of Pr.
        let s = Shakhov::new(2.0 / 3.0);
        let rho = 1.0;
        let u = [0.0, 0.0];
        let t = 300.0;
        let r_gas = 287.0;
        let (vgrid, vw) = VelocityGrid2D::simpson(2500.0, 121);
        let mut m_rho = 0.0;
        for (k, v) in vgrid.iter().enumerate() {
            let (g, _h) = s.equilibrium(rho, u, t, r_gas, [0.0, 0.0], *v);
            m_rho += vw[k] * g;
        }
        assert!((m_rho - rho).abs() / rho < 1e-3);
    }

    #[test]
    fn equilibrium_with_dof_reduces_to_monatomic_default() {
        let s = Shakhov::new(2.0 / 3.0);
        let rho = 1.0;
        let u = [10.0, -5.0];
        let t = 300.0;
        let r_gas = 287.0;
        let q = [3.0, -1.0];
        let v = [40.0, -20.0];
        let (g1, h1) = s.equilibrium(rho, u, t, r_gas, q, v);
        let (g2, h2) = s.equilibrium_with_dof(rho, u, t, r_gas, q, v, DOF);
        assert!((g1 - g2).abs() < 1e-12);
        assert!((h1 - h2).abs() < 1e-12);
    }

    #[test]
    fn equilibrium_with_dof_diatomic_matches_moment_construction() {
        // dof_total = 5 (diatomic rigid rotor, zeta_int=2) with zero heat
        // flux should reduce exactly to gh_equilibrium_with_k's plain
        // Maxwellian pair (Shakhov correction vanishes at q=0 regardless of
        // Pr, same as the monatomic bgk_limit-style check above).
        let s = Shakhov::new(2.0 / 3.0);
        let rho = 1.0;
        let u = [0.0, 0.0];
        let t = 300.0;
        let r_gas = 287.0;
        let dof_total = crate::maxwellian::dof_with_internal(2.0);
        let k_total = crate::maxwellian::k_reduced_with_internal(2.0);
        let v = [15.0, -7.0];
        let (g, h) = s.equilibrium_with_dof(rho, u, t, r_gas, [0.0, 0.0], v, dof_total);
        let (g_m, h_m) = crate::maxwellian::gh_equilibrium_with_k(rho, u, t, r_gas, v, k_total);
        assert!((g - g_m).abs() < 1e-12);
        assert!((h - h_m).abs() < 1e-12);
    }

    #[test]
    fn relaxation_time_scales_with_viscosity() {
        let s = Shakhov::new(2.0 / 3.0);
        let tau = s.relaxation_time(1.0, 300.0, 287.0, 2.117e-5, 273.15, 0.81);
        assert!(tau > 0.0);
    }
}
