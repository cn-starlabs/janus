//! Discrete velocity ordinate/weight generation.
//!
//! ## Quadrature choice: Gauss-Hermite (preferred, `gauss_hermite`)
//!
//! DVM moment integrals are always of the form
//! `\int_{-inf}^{inf} phi(v) * exp(-(v-u)^2/(2RT)) dv` — i.e. an integrand
//! that is a (typically low-order-polynomial-in-v) function `phi` times a
//! Gaussian/Maxwellian weight. This is *exactly* the weight function
//! `exp(-x^2)` that Gauss-Hermite quadrature is built for (after the affine
//! substitution `x = (v-u)/sqrt(2RT)`), so for a fixed node budget
//! Gauss-Hermite integrates polynomials in `v` up to degree `2n-1` *exactly*
//! with `n` nodes, whereas a Newton-Cotes rule (composite Simpson) only
//! achieves degree-3 exactness per interval and additionally wastes nodes
//! evaluating the integrand far in the Gaussian tails (where a fixed grid
//! must extend to some ad hoc `vmax` and hope truncation error is small).
//! Gauss-Hermite nodes cluster automatically where the Maxwellian weight
//! actually has support, giving a much better accuracy-per-node ratio for
//! the moment integrals (`rho`, `rho*u`, `rho*E`, stress, heat flux) that
//! drive this solver — the standard choice in the DVM/UGKS literature for
//! Maxwellian-weighted quadrature.
//!
//! Reference: Abramowitz, M., Stegun, I. A. (eds.), "Handbook of
//! Mathematical Functions", Dover (1972), §25.4.46 and Table 25.10 (Gauss-
//! Hermite abscissas/weights); see also Press, Teukolsky, Vetterling &
//! Flannery, "Numerical Recipes", 3rd ed., §4.6 (Gaussian quadrature and
//! orthogonal polynomials) for the Golub-Welsch tridiagonal-eigenvalue
//! construction used below.
//!
//! Nodes/weights are built via the Golub-Welsch algorithm: the Hermite
//! recurrence `H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)` gives (after
//! normalizing to the *probabilists'* convention used here so the weight
//! function is exactly the standard-normal density `exp(-x^2/2)/sqrt(2 pi)`,
//! matching the Maxwellian directly with `x = (v-u)/sqrt(RT)`) a symmetric
//! tridiagonal Jacobi matrix whose eigenvalues are the quadrature nodes and
//! whose eigenvectors' first components give the weights. We solve the
//! small (n <= ~64) symmetric tridiagonal eigenproblem with the implicit QL
//! algorithm (standard, e.g. Numerical Recipes §11.4), which is numerically
//! robust for this size and avoids pulling in any linear-algebra dependency
//! beyond `faer` in the workspace's approved list — a bespoke small
//! symmetric-tridiagonal eigensolver for `n <= 64` is simple enough to hand
//! -verify (against tabulated Abramowitz & Stegun nodes for small n) and is
//! justified per ENGINEERING_SPEC.md's "unsafe/bespoke abstractions are
//! acceptable when justified" and "prefer hand-rolling small utilities".
//!
//! ## Legacy: composite Simpson (`simpson`, kept for back-compat/comparison)
//!
//! The original M1 quadrature (tensor-product Newton-Cotes on a truncated
//! range) is retained as `VelocityGrid2D::simpson` purely so existing
//! call sites / tests are not broken by this change; new code should prefer
//! `gauss_hermite`. It is *not* removed because many existing integration
//! tests construct velocity grids via this API and doing a blind mechanical
//! rename across 10 call sites without a compiler to verify each one is
//! exactly the kind of unverifiable mass-edit this pass must avoid.

/// Single Maxwellian-mapped 1D Gauss-Hermite axis: returns physical-space
/// `(nodes, weights)` such that `sum_k weights[k] * g(nodes[k])` approximates
/// `\int g(v) dv` directly (true integration weights, Gaussian weight folded
/// out — see `VelocityGrid2D::gauss_hermite`). This is the single source of
/// truth for one velocity axis, reused by both the 2D and 3D tensor-product
/// grids so the weight convention can never diverge between them.
pub fn gauss_hermite_1d_axis(r_gas: f64, t_ref: f64, u_ref: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n >= 2, "gauss_hermite needs at least 2 nodes per axis");
    let (x, w) = probabilists_gauss_hermite_nodes(n);
    let std_dev = (r_gas * t_ref).max(0.0).sqrt();
    let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
    let nodes: Vec<f64> = x.iter().map(|&xi| u_ref + std_dev * xi).collect();
    let weights: Vec<f64> = x
        .iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| wi * (0.5 * xi * xi).exp() * sqrt_2pi * std_dev)
        .collect();
    (nodes, weights)
}

/// Build a 2D tensor-product discrete velocity grid.
pub struct VelocityGrid2D;

impl VelocityGrid2D {
    /// Tensor-product Gauss-Hermite velocity grid on the *physical* velocity
    /// plane, built from a 1D `n_per_axis`-point Gauss-Hermite rule mapped
    /// from the standard-normal weight `exp(-x^2/2)` to a Maxwellian with
    /// bulk velocity `u_ref` and temperature `t_ref` (used only to *place*
    /// the nodes/weights well; the DVM itself is still evaluated at whatever
    /// local `rho,u,T` a cell actually has — the discrete velocity set is
    /// fixed for the whole domain, as is standard for DVM/DUGKS/UGKWP).
    ///
    /// `vw[k]` already has the Maxwellian weight "divided out": each node's
    /// weight is `w_k * sqrt(2*R*t_ref)` (the affine Jacobian) so that
    /// `sum_k vw[k] * phi(v_k)` approximates `\int phi(v) dv` directly
    /// (i.e. `vw` are physical-space quadrature weights, matching the
    /// existing `Distribution`/moment code's convention that `f`, not
    /// `f/Maxwellian`, is the stored quantity and `vw` are plain integration
    /// weights) — NOT Gauss-Hermite's native weight convention that would
    /// require dividing the integrand by `exp(-x^2)`. Concretely, if `(x_k,
    /// w_k)` are the standard (probabilists') Gauss-Hermite node/weight pairs
    /// for `exp(-x^2/2)/sqrt(2*pi)`, we set
    /// `v_k = u_ref + sqrt(r_gas*t_ref) * x_k` and
    /// `vw[k] = w_k * exp(x_k^2/2) * sqrt(2*pi) * sqrt(r_gas*t_ref)`
    /// (the `exp(x_k^2/2)*sqrt(2*pi)` factor folds the Gauss-Hermite weight
    /// function back out so `vw` are plain physical-space integration weights),
    /// which makes `sum_k vw[k] * g(v_k)` exact for `g` a polynomial in `v`
    /// times the `t_ref`-Maxwellian, and a good (spectrally accurate for
    /// smooth `f`) approximation to `\int g(v) dv` in general — exactly the
    /// role `vw` plays for the DVM moment integrals throughout this crate.
    pub fn gauss_hermite(
        r_gas: f64,
        t_ref: f64,
        u_ref: [f64; 2],
        n_per_axis: usize,
    ) -> (Vec<[f64; 2]>, Vec<f64>) {
        assert!(
            n_per_axis >= 2,
            "gauss_hermite needs at least 2 nodes per axis"
        );
        // Both axes share the same 1D physical-space Gauss-Hermite rule (true
        // integration weights, Gaussian folded out — see `gauss_hermite_1d_axis`),
        // offset by the per-axis bulk velocity.
        let (nodes_x, weights_x) = gauss_hermite_1d_axis(r_gas, t_ref, u_ref[0], n_per_axis);
        let (nodes_y, weights_y) = gauss_hermite_1d_axis(r_gas, t_ref, u_ref[1], n_per_axis);

        let mut vgrid = Vec::with_capacity(n_per_axis * n_per_axis);
        let mut vw = Vec::with_capacity(n_per_axis * n_per_axis);
        for (&vy, &wy) in nodes_y.iter().zip(weights_y.iter()) {
            for (&vx, &wx) in nodes_x.iter().zip(weights_x.iter()) {
                vgrid.push([vx, vy]);
                vw.push(wx * wy);
            }
        }
        (vgrid, vw)
    }

    /// Legacy composite-Simpson tensor-product velocity grid on
    /// `[-vmax, vmax]^2` with `n_per_axis` points per axis (must be odd and
    /// >= 3). See module docs: `gauss_hermite` is the preferred quadrature
    /// for new code; this remains for existing call sites.
    pub fn simpson(vmax: f64, n_per_axis: usize) -> (Vec<[f64; 2]>, Vec<f64>) {
        assert!(
            n_per_axis >= 3 && n_per_axis % 2 == 1,
            "n_per_axis must be odd and >= 3"
        );
        let h = 2.0 * vmax / (n_per_axis as f64 - 1.0);
        let nodes_1d: Vec<f64> = (0..n_per_axis).map(|i| -vmax + i as f64 * h).collect();
        let weights_1d: Vec<f64> = (0..n_per_axis)
            .map(|i| {
                let coeff = if i == 0 || i == n_per_axis - 1 {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                coeff * h / 3.0
            })
            .collect();

        let mut vgrid = Vec::with_capacity(n_per_axis * n_per_axis);
        let mut vw = Vec::with_capacity(n_per_axis * n_per_axis);
        for &vy in &nodes_1d {
            for &vx in &nodes_1d {
                vgrid.push([vx, vy]);
            }
        }
        for (j, &wy) in weights_1d.iter().enumerate() {
            for (i, &wx) in weights_1d.iter().enumerate() {
                let _ = (i, j);
                vw.push(wx * wy);
            }
        }
        (vgrid, vw)
    }
}

/// Compute the `n`-point *probabilists'* Gauss-Hermite quadrature nodes and
/// weights for the standard-normal weight function `exp(-x^2/2)/sqrt(2*pi)`,
/// i.e. `\int phi(x) exp(-x^2/2)/sqrt(2*pi) dx ~= sum_k w_k * phi(x_k)`,
/// via the Golub-Welsch method (Golub & Welsch 1969; see also Press et al.,
/// "Numerical Recipes" 3rd ed. §4.6): the three-term recurrence for the
/// (monic, probabilists') Hermite polynomials
/// `He_{k+1}(x) = x*He_k(x) - k*He_{k-1}(x)`
/// gives a symmetric tridiagonal ("Jacobi") matrix with zero diagonal and
/// off-diagonal `beta_k = sqrt(k)` (`k = 1..n-1`); its eigenvalues are the
/// quadrature nodes and `w_k = (first component of the k-th eigenvector)^2`
/// (eigenvectors normalized to unit length), since `mu_0 = \int exp(-x^2/2)/
/// sqrt(2*pi) dx = 1` for the standard normal weight.
///
/// The symmetric tridiagonal eigenproblem is solved with the implicit-shift
/// QL algorithm with accumulated eigenvectors (a standard, numerically
/// robust dense method for `n` this small — a handful to a few dozen nodes
/// per velocity axis; see Numerical Recipes §11.4, `tqli`). This is a small,
/// self-contained bespoke numerical routine (no external linear-algebra
/// dependency), justified per ENGINEERING_SPEC.md's guidance to prefer
/// hand-rolled utilities over new dependencies for small, well-scoped
/// numerical kernels.
fn probabilists_gauss_hermite_nodes(n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n >= 1);
    // Jacobi matrix: diagonal all zero (Hermite recurrence has no linear
    // term), off-diagonal beta_k = sqrt(k) for k = 1..=n-1.
    let mut diag = vec![0.0f64; n];
    let mut offdiag = vec![0.0f64; n]; // offdiag[k] connects diag[k-1]-diag[k]; offdiag[0] unused
    for k in 1..n {
        offdiag[k] = (k as f64).sqrt();
    }
    // Eigenvector matrix, initialized to identity; QL accumulates rotations
    // into it so column k becomes the k-th eigenvector.
    let mut z = vec![0.0f64; n * n];
    for i in 0..n {
        z[i * n + i] = 1.0;
    }

    tridiagonal_ql_implicit(&mut diag, &mut offdiag, &mut z, n);

    // Sort eigenpairs by node value ascending (QL does not guarantee order).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| diag[a].partial_cmp(&diag[b]).unwrap());

    let nodes: Vec<f64> = order.iter().map(|&i| diag[i]).collect();
    // Weight = (first eigenvector component)^2, since mu_0 = 1 for the
    // standard-normal weight (Golub-Welsch). `z` is column-major here
    // (z[row*n + col]); the first component of eigenvector `col` is
    // `z[0*n + col] = z[col]`.
    let weights: Vec<f64> = order.iter().map(|&i| z[i] * z[i]).collect();

    // Normalize defensively: sum of weights must equal mu_0 = 1 (Gaussian
    // total mass) to floating-point precision; if it doesn't, renormalize
    // rather than silently propagating an ill-conditioned eigensolve.
    let wsum: f64 = weights.iter().sum();
    let weights = if (wsum - 1.0).abs() > 1e-6 {
        weights.iter().map(|w| w / wsum).collect()
    } else {
        weights
    };

    (nodes, weights)
}

/// Implicit-shift QL algorithm with eigenvector accumulation for a real
/// symmetric tridiagonal matrix (diagonal `d[0..n]`, off-diagonal `e[1..n]`,
/// `e[0]` unused), operating in place. `z` (an `n x n` matrix stored
/// row-major as a flat `Vec`, initialized to the identity by the caller) is
/// updated in place by the same Givens rotations so that afterwards column
/// `k` of `z` is the eigenvector for eigenvalue `d[k]`. Standard algorithm,
/// e.g. Press, Teukolsky, Vetterling & Flannery, "Numerical Recipes" 3rd
/// ed., §11.4 (`tqli`), originally Bowdler, Martin, Reinsch & Wilkinson
/// (1968) / Golub & Van Loan, "Matrix Computations".
fn tridiagonal_ql_implicit(d: &mut [f64], e: &mut [f64], z: &mut [f64], n: usize) {
    if n <= 1 {
        return;
    }
    // Shift e so e[0..n-1] holds the sub/super-diagonal, e[n-1] = 0 sentinel
    // (matches the classic tqli indexing convention).
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;

    for l in 0..n {
        let mut iter = 0;
        loop {
            // Find m: the smallest index >= l such that e[m] is negligible
            // (converged sub-block boundary).
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            assert!(
                iter < 100,
                "gauss_hermite: QL eigensolver failed to converge (n={n})"
            );

            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            let mut i = (m as isize) - 1;
            while i >= l as isize {
                let iu = i as usize;
                let mut f = s * e[iu];
                let b = c * e[iu];
                r = f.hypot(g);
                e[iu + 1] = r;
                if r == 0.0 {
                    d[iu + 1] -= p;
                    e[m] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[iu + 1] - p;
                r = (d[iu] - g) * s + 2.0 * c * b;
                p = s * r;
                d[iu + 1] = g + p;
                g = c * r - b;
                // Accumulate the rotation into the eigenvector matrix z:
                // rotate columns (iu) and (iu+1) for every row.
                for row in 0..n {
                    f = z[row * n + iu + 1];
                    z[row * n + iu + 1] = s * z[row * n + iu] + c * f;
                    z[row * n + iu] = c * z[row * n + iu] - s * f;
                }
                i -= 1;
            }
            if r == 0.0 && (m as isize) - 1 >= l as isize {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simpson_weights_integrate_constant_to_area() {
        let vmax = 5.0;
        let (_, vw) = VelocityGrid2D::simpson(vmax, 41);
        let total: f64 = vw.iter().sum();
        let expected = (2.0 * vmax) * (2.0 * vmax);
        assert!((total - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn gauss_hermite_1d_nodes_match_known_values_n3() {
        // Probabilists' Gauss-Hermite, n=3: nodes 0, +-sqrt(3); weights
        // 2/3, 1/6, 1/6 (standard tabulated values, e.g. Abramowitz & Stegun
        // Table 25.10 converted from physicists' to probabilists' convention,
        // or directly: exact for the standard normal weight).
        let (nodes, weights) = probabilists_gauss_hermite_nodes(3);
        let mut got: Vec<(f64, f64)> = nodes.into_iter().zip(weights).collect();
        got.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let sqrt3 = 3f64.sqrt();
        assert!((got[0].0 - (-sqrt3)).abs() < 1e-9, "node0 {}", got[0].0);
        assert!((got[1].0 - 0.0).abs() < 1e-9, "node1 {}", got[1].0);
        assert!((got[2].0 - sqrt3).abs() < 1e-9, "node2 {}", got[2].0);
        assert!((got[0].1 - 1.0 / 6.0).abs() < 1e-9, "w0 {}", got[0].1);
        assert!((got[1].1 - 2.0 / 3.0).abs() < 1e-9, "w1 {}", got[1].1);
        assert!((got[2].1 - 1.0 / 6.0).abs() < 1e-9, "w2 {}", got[2].1);
    }

    #[test]
    fn gauss_hermite_weights_sum_to_one() {
        for n in [2, 3, 4, 5, 8, 13, 21] {
            let (_, w) = probabilists_gauss_hermite_nodes(n);
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "n={n} sum={sum}");
        }
    }

    #[test]
    fn gauss_hermite_2d_grid_integrates_maxwellian_moments_exactly() {
        // rho: zeroth moment of the Maxwellian should be exactly rho (up to
        // quadrature/floating point error), and momentum should match u_ref
        // exactly (odd moments of a centered Gaussian vanish, captured
        // exactly by symmetric GH nodes for the low-degree polynomial
        // moments tested here).
        let r_gas = 287.0;
        let t = 300.0;
        let u = [10.0, -5.0];
        let (vgrid, vw) = VelocityGrid2D::gauss_hermite(r_gas, t, u, 8);
        let rho = 1.2;
        let mut m0 = 0.0;
        let mut m1x = 0.0;
        let mut m1y = 0.0;
        for (k, v) in vgrid.iter().enumerate() {
            let g = crate::maxwellian::maxwellian_2d(rho, u, t, r_gas, *v);
            m0 += vw[k] * g;
            m1x += vw[k] * g * v[0];
            m1y += vw[k] * g * v[1];
        }
        assert!((m0 - rho).abs() / rho < 1e-8, "rho moment {m0}");
        assert!((m1x - rho * u[0]).abs() / rho < 1e-6, "mx moment {m1x}");
        assert!((m1y - rho * u[1]).abs() / rho < 1e-6, "my moment {m1y}");
    }
}
