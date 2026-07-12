//! 3D discrete-velocity grid: tensor-product Gauss-Hermite quadrature in
//! `(vx, vy, vz)`, generalizing `velocity_grid::VelocityGrid2D::gauss_hermite`
//! to the full 3-component velocity space needed by the M4 3D DVM solver.
//!
//! Reuses the *same* underlying 1D probabilists' Gauss-Hermite node/weight
//! generator (`probabilists_gauss_hermite_nodes`, Golub & Welsch 1969 —
//! see `velocity_grid.rs` module docs for the full derivation/citation) so
//! there is exactly one quadrature-generation implementation in this crate;
//! this module only adds the third tensor-product axis and the
//! corresponding node/weight outer product.
//!
//! Reference: Golub, G. H., Welsch, J. H., "Calculation of Gauss Quadrature
//! Rules", Math. Comp. 23, 221-230 (1969); Abramowitz & Stegun §25.4.46.

use crate::velocity_grid::gauss_hermite_1d_axis;

/// Build a 3D tensor-product discrete velocity grid.
pub struct VelocityGrid3D;

impl VelocityGrid3D {
    /// Tensor-product Gauss-Hermite velocity grid on the physical velocity
    /// volume `(vx, vy, vz)`, built by tensoring a 1D `n_per_axis`-point
    /// Gauss-Hermite rule (shared with `VelocityGrid2D::gauss_hermite`,
    /// obtained here by reusing its 2D product for the `(vx, vy)` plane and
    /// then tensoring in a third independently-generated 1D axis for `vz`)
    /// mapped from the standard-normal weight to a Maxwellian with bulk
    /// velocity `u_ref` and temperature `t_ref`.
    ///
    /// Returns `(vgrid, vw)` with `vgrid[k] = [vx_k, vy_k, vz_k]` and
    /// `vw[k]` the physical-space (not Gauss-Hermite-native) quadrature
    /// weight, i.e. `sum_k vw[k] * phi(v_k) ~= \int phi(v) dv` directly,
    /// exactly matching the `Distribution3D`/moment code's convention
    /// (see `velocity_grid.rs` docs for the precise weight-rescaling
    /// derivation, identical here per axis).
    ///
    /// `n_per_axis^3` total nodes; callers should keep `n_per_axis` modest
    /// (e.g. 6-10) since the 3D tensor product grows the node count as the
    /// cube of the 1D rule's order.
    pub fn gauss_hermite(
        r_gas: f64,
        t_ref: f64,
        u_ref: [f64; 3],
        n_per_axis: usize,
    ) -> (Vec<[f64; 3]>, Vec<f64>) {
        assert!(
            n_per_axis >= 2,
            "gauss_hermite needs at least 2 nodes per axis"
        );

        // Three independent 1D physical-space Gauss-Hermite axes (shared
        // single-source generator `gauss_hermite_1d_axis`, true integration
        // weights with the Gaussian folded out), tensored together.
        let (nodes_x, weights_x) = gauss_hermite_1d_axis(r_gas, t_ref, u_ref[0], n_per_axis);
        let (nodes_y, weights_y) = gauss_hermite_1d_axis(r_gas, t_ref, u_ref[1], n_per_axis);
        let (nodes_z, weights_z) = gauss_hermite_1d_axis(r_gas, t_ref, u_ref[2], n_per_axis);

        let n = n_per_axis;
        let mut vgrid = Vec::with_capacity(n * n * n);
        let mut vw = Vec::with_capacity(n * n * n);
        for (&vz, &wz) in nodes_z.iter().zip(weights_z.iter()) {
            for (&vy, &wy) in nodes_y.iter().zip(weights_y.iter()) {
                for (&vx, &wx) in nodes_x.iter().zip(weights_x.iter()) {
                    vgrid.push([vx, vy, vz]);
                    vw.push(wx * wy * wz);
                }
            }
        }
        (vgrid, vw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_hermite_3d_grid_integrates_maxwellian_moments_exactly() {
        let r_gas = 287.0;
        let t = 300.0;
        let u = [10.0, -5.0, 3.0];
        let (vgrid, vw) = VelocityGrid3D::gauss_hermite(r_gas, t, u, 6);
        assert_eq!(vgrid.len(), 6 * 6 * 6);
        let rho = 1.2;
        let mut m0 = 0.0;
        let mut m1 = [0.0; 3];
        for (k, v) in vgrid.iter().enumerate() {
            let g = crate::maxwellian3d::maxwellian_3d(rho, u, t, r_gas, *v);
            m0 += vw[k] * g;
            m1[0] += vw[k] * g * v[0];
            m1[1] += vw[k] * g * v[1];
            m1[2] += vw[k] * g * v[2];
        }
        assert!((m0 - rho).abs() / rho < 1e-6, "rho moment {m0}");
        assert!((m1[0] - rho * u[0]).abs() / rho < 1e-4, "mx {m1:?}");
        assert!((m1[1] - rho * u[1]).abs() / rho < 1e-4, "my {m1:?}");
        assert!((m1[2] - rho * u[2]).abs() / rho < 1e-4, "mz {m1:?}");
    }

    #[test]
    fn weights_integrate_reference_maxwellian_to_rho() {
        // With true physical-space integration weights, the meaningful sanity
        // check is that the rule integrates the reference Maxwellian to its
        // density (sum_k vw[k]*M(v_k) = rho). The bare sum of weights has no
        // clean closed form (it is the rule's approximation of the divergent
        // integral of 1 over all velocity space), so we check the Maxwellian
        // moment instead — the property the quadrature actually exists to
        // reproduce.
        let r_gas = 287.0;
        let t = 300.0;
        let u = [0.0, 0.0, 0.0];
        let (vgrid, vw) = VelocityGrid3D::gauss_hermite(r_gas, t, u, 8);
        let rho = 1.0;
        let mut m0 = 0.0;
        for (k, v) in vgrid.iter().enumerate() {
            m0 += vw[k] * crate::maxwellian3d::maxwellian_3d(rho, u, t, r_gas, *v);
        }
        assert!((m0 - rho).abs() / rho < 1e-6, "rho {m0} vs {rho}");
    }
}
