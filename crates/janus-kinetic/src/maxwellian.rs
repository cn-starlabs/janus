//! Reduced (g, h) two-distribution equilibrium for the 2D-velocity-space DVM
//! reduction of a 3D monatomic gas.
//!
//! ## Physics background
//!
//! A real monatomic gas has 3 translational degrees of freedom (DOF), giving
//! `gamma = (3+2)/3 = 5/3`. A naive strictly-2D-in-velocity DVM (M1's
//! approach, see the removed `DOF = 2.0` constant) only carries 2 DOF and
//! therefore gets `gamma = 2`, which is physically wrong for any real gas.
//!
//! The standard fix (Chu 1965 for the isothermal case; Guo, Xu & Wang's
//! DUGKS papers and Xu & Huang 2010 for the compressible Shakhov/ES-BGK
//! reduction) is to integrate the true 3D distribution `f(x, v, eta, t)`
//! (where `v = (vx, vy)` is kept discretized and `eta` is the third,
//! "internal", velocity component) over `eta` analytically, producing two
//! *reduced* 2D-velocity distributions:
//!
//! ```text
//! g(x, v, t) = \int f(x, v, eta, t) d(eta)
//! h(x, v, t) = \int eta^2 f(x, v, eta, t) d(eta)
//! ```
//!
//! `g` carries mass/momentum/the in-plane kinetic energy; `h` carries the
//! energy associated with the reduced-out `eta` direction. Together they
//! exactly reproduce all moments of the full 3D distribution while only ever
//! discretizing a 2D velocity grid. With `K = 1` reduced internal DOF (the
//! single `eta` direction standing in for the "third" translational
//! component) and `D = 2` real discretized dimensions, the *total* DOF is
//! `D + K = 3`, recovering the correct monatomic `gamma = (D+K+2)/(D+K) =
//! 5/3`.
//!
//! Equilibrium (Maxwellian) reduction, for a monatomic gas with no other
//! internal structure:
//!
//! ```text
//! g_eq(v) = rho * (1 / (2*pi*R*T)) * exp(-|v-u|^2 / (2*R*T))          (2D Maxwellian)
//! h_eq(v) = K * R * T * g_eq(v) = (R*T) * g_eq(v)   [K=1]
//! ```
//!
//! Moments recovered from (g, h):
//!
//! ```text
//! rho        = \int g dv
//! rho * u    = \int v * g dv
//! rho * E    = 0.5 * \int (|v|^2 * g + h) dv     [E = total energy per mass]
//! ```
//!
//! which by construction includes the `eta`-direction internal energy
//! `0.5 * K * rho * R * T` without ever discretizing `eta`.
//!
//! References:
//! - Xu, K., Huang, J.-C., "A unified gas-kinetic scheme for continuum and
//!   rarefied flows", J. Comput. Phys. 229, 7747-7764 (2010) — introduces the
//!   (g,h) reduction used by DUGKS/UGKS/UGKWP for non-isothermal flow.
//! - Guo, Z., Xu, K., Wang, R., "Discrete unified gas kinetic scheme for all
//!   Knudsen number flows", Phys. Rev. E 88, 033305 (2013).
//! - Chu, C. K., "Kinetic-theoretic description of the formation of a shock
//!   wave", Phys. Fluids 8, 12-22 (1965) — the original reduced-distribution
//!   idea (isothermal case, single reduced function).

/// Number of *discretized* velocity-space translational DOF (the 2D `(vx,
/// vy)` plane): `D = 2`.
pub const D_DISCRETE: f64 = 2.0;

/// Number of *reduced* (analytically integrated-out) internal/translational
/// DOF folded into the `h` distribution: `K = 1` (the third, `eta`,
/// translational velocity component of a real 3D monatomic gas).
pub const K_REDUCED: f64 = 1.0;

/// Total effective degrees of freedom `D + K = 3`, i.e. the correct
/// monatomic-gas count, giving `gamma = (DOF+2)/DOF = 5/3`.
pub const DOF: f64 = D_DISCRETE + K_REDUCED;

/// Ratio of specific heats recovered by this (g,h) reduction: `5/3` for a
/// monatomic gas (matches the true 3D result, NOT the `gamma=2` of a naive
/// strictly-2D velocity space).
pub const GAMMA: f64 = (DOF + 2.0) / DOF;

/// Evaluate the 2D Maxwellian `M(v) = rho / (2*pi*R*T) * exp(-|v-u|^2 / (2*R*T))`
/// at discrete velocity `v` given macroscopic state `(rho, u, T)` and gas
/// constant `r_gas`. This is exactly `g_eq`.
#[inline]
pub fn maxwellian_2d(rho: f64, u: [f64; 2], t: f64, r_gas: f64, v: [f64; 2]) -> f64 {
    let rt = r_gas * t;
    let norm = rho / (2.0 * std::f64::consts::PI * rt);
    let dvx = v[0] - u[0];
    let dvy = v[1] - u[1];
    let exponent = -(dvx * dvx + dvy * dvy) / (2.0 * rt);
    norm * exponent.exp()
}

/// Reduced-distribution equilibrium pair `(g_eq, h_eq)` at discrete velocity
/// `v`: `g_eq` is the plain 2D Maxwellian; `h_eq = K*R*T*g_eq` carries the
/// reduced-out internal (eta-direction) energy so that
/// `\int h_eq dv = K * rho * R * T` (the correct internal energy contribution
/// of the third translational DOF).
#[inline]
pub fn gh_equilibrium(rho: f64, u: [f64; 2], t: f64, r_gas: f64, v: [f64; 2]) -> (f64, f64) {
    gh_equilibrium_with_k(rho, u, t, r_gas, v, K_REDUCED)
}

/// Generalized (g,h) equilibrium with an explicit reduced-DOF count `k_total`
/// in place of the crate-wide monatomic constant `K_REDUCED = 1`. This is
/// the polyatomic/internal-DOF entry point (ENGINEERING_SPEC.md M5): a
/// diatomic/polyatomic gas's rotational (+ vibrational) internal energy is
/// carried entirely by the `h`-distribution's reduced-DOF bookkeeping,
/// exactly the same mechanism that already carries the monatomic gas's
/// "third" (analytically-integrated-out) translational DOF — a polyatomic
/// gas is simply the case `k_total = K_REDUCED + zeta_int` (1 reduced
/// translational DOF + `zeta_int` internal DOF from `GasModel::internal_dof`),
/// with `h_eq = k_total * R * T * g_eq` so
/// `\int h_eq dv = k_total * rho * R * T` supplies exactly the internal
/// energy `zeta_int` rotational/vibrational DOF need in addition to the
/// monatomic reduced-translational term, giving the correct total
/// `DOF_total = D_DISCRETE + k_total = 2 + 1 + zeta_int` and hence
/// `gamma = (DOF_total+2)/DOF_total` (diatomic rigid rotor, `zeta_int=2`:
/// `DOF_total=5`, `gamma=7/5`, matching `GasModel::gamma()`'s already-tested
/// `7/5` value in `gas_model.rs`).
///
/// Reference: same (g,h) reduction as `gh_equilibrium` (Xu & Huang 2010),
/// generalized to an arbitrary total reduced-DOF count exactly as Xu & Huang
/// 2010 §2 present the general `K`-DOF (g,h) construction (the monatomic
/// `K=1` case `gh_equilibrium` calls is the specialization used by the rest
/// of this crate's monatomic default path).
#[inline]
pub fn gh_equilibrium_with_k(
    rho: f64,
    u: [f64; 2],
    t: f64,
    r_gas: f64,
    v: [f64; 2],
    k_total: f64,
) -> (f64, f64) {
    let g = maxwellian_2d(rho, u, t, r_gas, v);
    // h_eq = k_total * R * T * g_eq, so that int h_eq dv = k_total * rho * R * T
    // (each reduced/internal DOF contributes R*T to the eta-variance:
    // <eta^2>_Maxwellian = k_total * R*T). The internal energy recovered in the
    // energy moment is then 0.5*int h dv = 0.5*k_total*rho*R*T, which combines
    // with the in-plane 0.5*rho*D*R*T translational thermal energy to give the
    // correct total 0.5*rho*(D+k_total)*R*T. (Previously carried a spurious 0.5
    // factor here, halving the reduced-DOF internal energy — see the corrected
    // moment-recovery tests.)
    let h = k_total * r_gas * t * g;
    (g, h)
}

/// Total effective DOF for a polyatomic gas whose internal (rotational +
/// vibrational) degrees of freedom beyond the 3 translational ones are
/// `zeta_int` (e.g. `GasModel::internal_dof()`): `DOF_total = D_DISCRETE +
/// K_REDUCED + zeta_int = 3 + zeta_int`, matching `GasModel::total_dof()`
/// exactly (`3.0 + internal_dof()`) so the (g,h) reduction and the
/// `GasModel` trait's own DOF bookkeeping can never drift apart.
#[inline]
pub fn dof_with_internal(zeta_int: f64) -> f64 {
    DOF + zeta_int
}

/// `k_total` (the reduced-DOF parameter `gh_equilibrium_with_k` takes) for a
/// polyatomic gas with `zeta_int` internal DOF: `K_REDUCED + zeta_int`
/// (the monatomic reduced-translational DOF plus the internal DOF, all
/// carried by the `h` distribution — see `gh_equilibrium_with_k` docs).
#[inline]
pub fn k_reduced_with_internal(zeta_int: f64) -> f64 {
    K_REDUCED + zeta_int
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::velocity_grid::VelocityGrid2D;

    #[test]
    fn gamma_is_five_thirds() {
        assert!((GAMMA - 5.0 / 3.0).abs() < 1e-12, "gamma = {GAMMA}");
    }

    #[test]
    fn gh_moments_recover_macro_state_with_correct_dof() {
        let rho = 1.2;
        let u = [0.3, -0.1];
        let t = 300.0;
        let r_gas = 287.0;
        let (vgrid, vw) = VelocityGrid2D::simpson(2500.0, 121);

        let mut m_rho = 0.0;
        let mut m_mx = 0.0;
        let mut m_my = 0.0;
        let mut m_e = 0.0; // rho*E = 0.5*int(|v|^2 g + h)
        for (k, v) in vgrid.iter().enumerate() {
            let (g, h) = gh_equilibrium(rho, u, t, r_gas, *v);
            let wg = vw[k] * g;
            m_rho += wg;
            m_mx += wg * v[0];
            m_my += wg * v[1];
            let v2 = v[0] * v[0] + v[1] * v[1];
            m_e += 0.5 * vw[k] * (v2 * g + h);
        }
        assert!((m_rho - rho).abs() / rho < 1e-3, "rho {m_rho} vs {rho}");
        assert!((m_mx - rho * u[0]).abs() < 1e-2, "mx {m_mx}");
        assert!((m_my - rho * u[1]).abs() < 1e-2, "my {m_my}");

        let kinetic = 0.5 * rho * (u[0] * u[0] + u[1] * u[1]);
        // Correct monatomic internal energy: 0.5 * DOF * rho * R * T with DOF=3.
        let internal = 0.5 * rho * DOF * r_gas * t;
        let expected_e = kinetic + internal;
        assert!(
            (m_e - expected_e).abs() / expected_e < 1e-2,
            "E {m_e} vs {expected_e}"
        );
    }

    /// Polyatomic/internal-DOF end-to-end check (Part 2 of the hardening
    /// pass): a diatomic rigid rotor (`zeta_int = 2`, e.g. N2/O2/air-like)
    /// modeled via `gh_equilibrium_with_k` with `k_total =
    /// k_reduced_with_internal(2.0) = 3.0` must (a) recover `DOF_total =
    /// dof_with_internal(2.0) = 5.0` and hence `gamma = 7/5` (the textbook
    /// diatomic value, matching `GasModel::gamma()`'s already-tested `7/5`
    /// in `gas_model.rs`), and (b) reproduce the correct internal energy
    /// `0.5 * DOF_total * rho * R * T` from the (g,h) moments, exactly the
    /// same moment-recovery check `gh_moments_recover_macro_state_with_
    /// correct_dof` performs for the monatomic (`zeta_int=0`) case above —
    /// this is the "diatomic test validating gamma=7/5" the polyatomic-DOF
    /// hardening task requires.
    #[test]
    fn diatomic_internal_dof_gives_gamma_seven_fifths_and_correct_energy() {
        let zeta_int = 2.0; // rigid diatomic rotor: 2 rotational DOF
        let dof_total = dof_with_internal(zeta_int);
        assert!((dof_total - 5.0).abs() < 1e-12, "dof_total = {dof_total}");
        let gamma = (dof_total + 2.0) / dof_total;
        assert!((gamma - 7.0 / 5.0).abs() < 1e-12, "gamma = {gamma}");

        let k_total = k_reduced_with_internal(zeta_int);
        assert!((k_total - 3.0).abs() < 1e-12);

        let rho = 1.1;
        let u = [0.2, -0.4];
        let t = 320.0;
        let r_gas = 296.8; // N2-like specific gas constant
        let (vgrid, vw) = crate::velocity_grid::VelocityGrid2D::simpson(2500.0, 121);

        let mut m_rho = 0.0;
        let mut m_e = 0.0;
        for (k, v) in vgrid.iter().enumerate() {
            let (g, h) = gh_equilibrium_with_k(rho, u, t, r_gas, *v, k_total);
            m_rho += vw[k] * g;
            let v2 = v[0] * v[0] + v[1] * v[1];
            m_e += 0.5 * vw[k] * (v2 * g + h);
        }
        assert!((m_rho - rho).abs() / rho < 1e-3, "rho {m_rho} vs {rho}");

        let kinetic = 0.5 * rho * (u[0] * u[0] + u[1] * u[1]);
        let internal = 0.5 * rho * dof_total * r_gas * t;
        let expected_e = kinetic + internal;
        assert!(
            (m_e - expected_e).abs() / expected_e < 1e-2,
            "E {m_e} vs {expected_e}"
        );
    }
}
