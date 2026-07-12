//! Local Knudsen number estimate, drives the UGKWP wave/particle split.
//!
//! Uses the gradient-length-local (GLL) Knudsen number:
//! `Kn_loc = lambda * |grad(rho)| / rho`, where `lambda` is the local mean
//! free path, `lambda = mu / (rho * sqrt(2*R*T/pi))` (hard-sphere-consistent
//! mean free path from kinetic theory, e.g. Bird 1994 eq. 4.76 combined with
//! the VHS viscosity law used elsewhere in this crate). This is the
//! standard local-Kn proxy used in UGKS/UGKWP transition-regime papers
//! (e.g. see discussion in Xu & Huang 2010 and the DUGKS all-Knudsen-number
//! papers) since a single global Kn does not capture spatially-varying
//! rarefaction (e.g. shock fronts, wall boundary layers).

use janus_core::fields::MacroFields;
use janus_core::grid::Grid2D;

/// Compute the local Knudsen number field into `fields.kn_loc` from the
/// current density field (central differences on the interior, one-sided at
/// boundaries) and per-cell mean free path.
///
/// `mu` and `t` are per-cell already-known viscosity/temperature (caller
/// passes precomputed arrays to avoid recomputation); `r_gas` the specific
/// gas constant.
pub fn update_kn_loc(grid: &Grid2D, fields: &mut MacroFields, mu: &[f64], r_gas: f64) {
    let nx = grid.nx;
    let ny = grid.ny;
    let dx = grid.dx;
    let dy = grid.dy;

    // Read rho/T into locals first (avoid aliasing kn_loc write with rho read
    // in the same struct — MacroFields fields are independent Vecs so this
    // is not strictly required, but keeping local copies makes intent clear
    // and avoids any accidental future aliasing bug).
    let rho = fields.rho.clone();

    for j in 0..ny {
        for i in 0..nx {
            let c = grid.idx(i, j);
            let rho_c = rho[c].max(f64::MIN_POSITIVE);

            let (rho_w, rho_e) = if nx > 1 {
                let w = if i > 0 {
                    rho[grid.idx(i - 1, j)]
                } else {
                    rho[grid.idx(i, j)]
                };
                let e = if i + 1 < nx {
                    rho[grid.idx(i + 1, j)]
                } else {
                    rho[grid.idx(i, j)]
                };
                (w, e)
            } else {
                (rho_c, rho_c)
            };
            let (rho_s, rho_n) = if ny > 1 {
                let s = if j > 0 {
                    rho[grid.idx(i, j - 1)]
                } else {
                    rho[grid.idx(i, j)]
                };
                let n = if j + 1 < ny {
                    rho[grid.idx(i, j + 1)]
                } else {
                    rho[grid.idx(i, j)]
                };
                (s, n)
            } else {
                (rho_c, rho_c)
            };

            let drho_dx = (rho_e - rho_w) / (2.0 * dx);
            let drho_dy = (rho_n - rho_s) / (2.0 * dy);
            let grad_mag = (drho_dx * drho_dx + drho_dy * drho_dy).sqrt();

            let t_c = fields.temperature(c, r_gas, crate::maxwellian::DOF);
            let mu_c = mu[c];
            // Mean free path from kinetic theory (consistent with the VHS
            // viscosity law used throughout this crate); shared with
            // `janus_core::units::ReferenceScales::knudsen` so the two
            // Knudsen-number code paths (this per-cell local one and the
            // case-level reference one) cannot drift apart.
            let lambda = janus_core::units::vhs_mean_free_path(mu_c, rho_c, r_gas, t_c);

            fields.kn_loc[c] = lambda * grad_mag / rho_c;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_density_gives_zero_kn() {
        let grid = Grid2D::new(4, 4, 0.1, 0.1, [0.0, 0.0]);
        let mut fields = MacroFields::zeros(grid.ncells());
        for c in 0..grid.ncells() {
            fields.rho[c] = 1.0;
            fields.mom[0][c] = 0.0;
            fields.mom[1][c] = 0.0;
            fields.energy[c] = 0.5 * 3.0 * 287.0 * 300.0; // internal energy only, T=300
        }
        let mu = vec![2e-5; grid.ncells()];
        update_kn_loc(&grid, &mut fields, &mu, 287.0);
        for c in 0..grid.ncells() {
            assert!(
                fields.kn_loc[c].abs() < 1e-12,
                "kn_loc[{c}] = {}",
                fields.kn_loc[c]
            );
        }
    }
}
