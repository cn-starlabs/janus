//! Discrete-velocity distribution function storage (DVM), reduced (g,h)
//! two-distribution formulation.
//!
//! Cell-major indexing per ENGINEERING_SPEC.md §5/§8: `g[cell*nv+k]` so that
//! the innermost loop over discrete velocity nodes for a fixed cell is
//! contiguous (velocity-space locality, matches the DUGKS moment-integration
//! access pattern).
//!
//! PHYSICS: `g` is the velocity-space-reduced distribution (integrates the
//! true 3D `f` over the third/`eta` velocity component); `h` is the
//! `eta^2`-weighted reduction that carries the energy associated with that
//! reduced-out direction. Together they recover the correct monatomic 3-DOF
//! energy (`gamma = 5/3`) while only ever discretizing a 2D velocity grid.
//! See `janus_kinetic::maxwellian` module docs for the full derivation.

/// Discrete velocity set + per-cell reduced distribution values `(g, h)`.
///
/// `vgrid[k]` is the k-th discrete velocity ordinate `(vx, vy)`, `vw[k]` its
/// quadrature weight. `g` and `h` each have length `ncells * nv`; `g[cell *
/// nv + k]` / `h[cell * nv + k]` are the reduced distribution values for
/// velocity node `k` in `cell`.
#[derive(Clone, Debug)]
pub struct Distribution {
    pub nv: usize,
    pub vgrid: Vec<[f64; 2]>,
    pub vw: Vec<f64>,
    /// Reduced distribution `g = \int f d(eta)` (mass/momentum/in-plane KE).
    pub f: Vec<f64>,
    /// Reduced distribution `h = \int eta^2 f d(eta)` (reduced-direction
    /// internal energy). Same shape/indexing as `f`.
    pub h: Vec<f64>,
}

impl Distribution {
    /// Allocate zeroed storage for `ncells` cells given a velocity set.
    pub fn zeros(ncells: usize, vgrid: Vec<[f64; 2]>, vw: Vec<f64>) -> Self {
        let nv = vgrid.len();
        assert_eq!(nv, vw.len(), "vgrid and vw must have the same length");
        Self {
            nv,
            vgrid,
            vw,
            f: vec![0.0; ncells * nv],
            h: vec![0.0; ncells * nv],
        }
    }

    #[inline]
    pub fn ncells(&self) -> usize {
        if self.nv == 0 {
            0
        } else {
            self.f.len() / self.nv
        }
    }

    #[inline]
    pub fn index(&self, cell: usize, k: usize) -> usize {
        cell * self.nv + k
    }

    #[inline]
    pub fn cell_slice(&self, cell: usize) -> &[f64] {
        let start = cell * self.nv;
        &self.f[start..start + self.nv]
    }

    #[inline]
    pub fn cell_slice_mut(&mut self, cell: usize) -> &mut [f64] {
        let start = cell * self.nv;
        &mut self.f[start..start + self.nv]
    }

    #[inline]
    pub fn h_slice(&self, cell: usize) -> &[f64] {
        let start = cell * self.nv;
        &self.h[start..start + self.nv]
    }

    #[inline]
    pub fn h_slice_mut(&mut self, cell: usize) -> &mut [f64] {
        let start = cell * self.nv;
        &mut self.h[start..start + self.nv]
    }

    /// Zeroth moment: density (given the velocity weights already fold in
    /// any physical-space normalization the caller wants; here we return the
    /// raw `sum_k g_k * w_k`).
    #[inline]
    pub fn moment_rho(&self, cell: usize) -> f64 {
        let fs = self.cell_slice(cell);
        fs.iter().zip(&self.vw).map(|(f, w)| f * w).sum()
    }

    /// First moment: momentum `(rho*u_x, rho*u_y)`.
    #[inline]
    pub fn moment_mom(&self, cell: usize) -> [f64; 2] {
        let fs = self.cell_slice(cell);
        let mut mx = 0.0;
        let mut my = 0.0;
        for (k, fv) in fs.iter().enumerate() {
            let w = self.vw[k] * fv;
            mx += w * self.vgrid[k][0];
            my += w * self.vgrid[k][1];
        }
        [mx, my]
    }

    /// Second moment (total energy per volume): `rho*E = 0.5 * sum_k w_k *
    /// (|v_k|^2 * g_k + h_k)` — the `h` term supplies the reduced-out
    /// (third translational component) internal energy exactly, per the
    /// (g,h) reduction (see `janus_kinetic::maxwellian`).
    #[inline]
    pub fn moment_energy(&self, cell: usize) -> f64 {
        let gs = self.cell_slice(cell);
        let hs = self.h_slice(cell);
        let mut e = 0.0;
        for (k, (&gv, &hv)) in gs.iter().zip(hs.iter()).enumerate() {
            let v2 = self.vgrid[k][0] * self.vgrid[k][0] + self.vgrid[k][1] * self.vgrid[k][1];
            e += 0.5 * self.vw[k] * (v2 * gv + hv);
        }
        e
    }
}

/// Full 3D-velocity-space discrete-velocity distribution: single
/// distribution `f(cell, k)`, cell-major indexing `cell*nv+k` (identical
/// convention to `Distribution`, ENGINEERING_SPEC.md §5). No reduced `h`
/// carrier — see `janus_kinetic::maxwellian3d` module docs for why the 2D
/// solver's (g,h) reduction is correctly dropped once velocity space is
/// genuinely 3-component: all 3 translational DOF are already carried
/// directly by the discretized `(vx,vy,vz)` grid.
#[derive(Clone, Debug)]
pub struct Distribution3D {
    pub nv: usize,
    pub vgrid: Vec<[f64; 3]>,
    pub vw: Vec<f64>,
    /// Single distribution function value per (cell, velocity-node) pair,
    /// length `ncells * nv`, cell-major: `f[cell * nv + k]`.
    pub f: Vec<f64>,
}

impl Distribution3D {
    /// Allocate zeroed storage for `ncells` cells given a 3D velocity set.
    pub fn zeros(ncells: usize, vgrid: Vec<[f64; 3]>, vw: Vec<f64>) -> Self {
        let nv = vgrid.len();
        assert_eq!(nv, vw.len(), "vgrid and vw must have the same length");
        Self {
            nv,
            vgrid,
            vw,
            f: vec![0.0; ncells * nv],
        }
    }

    #[inline]
    pub fn ncells(&self) -> usize {
        if self.nv == 0 {
            0
        } else {
            self.f.len() / self.nv
        }
    }

    #[inline]
    pub fn index(&self, cell: usize, k: usize) -> usize {
        cell * self.nv + k
    }

    #[inline]
    pub fn cell_slice(&self, cell: usize) -> &[f64] {
        let start = cell * self.nv;
        &self.f[start..start + self.nv]
    }

    #[inline]
    pub fn cell_slice_mut(&mut self, cell: usize) -> &mut [f64] {
        let start = cell * self.nv;
        &mut self.f[start..start + self.nv]
    }

    /// Zeroth moment: density.
    #[inline]
    pub fn moment_rho(&self, cell: usize) -> f64 {
        let fs = self.cell_slice(cell);
        fs.iter().zip(&self.vw).map(|(f, w)| f * w).sum()
    }

    /// First moment: momentum `(rho*ux, rho*uy, rho*uz)`.
    #[inline]
    pub fn moment_mom(&self, cell: usize) -> [f64; 3] {
        let fs = self.cell_slice(cell);
        let mut m = [0.0; 3];
        for (k, fv) in fs.iter().enumerate() {
            let w = self.vw[k] * fv;
            m[0] += w * self.vgrid[k][0];
            m[1] += w * self.vgrid[k][1];
            m[2] += w * self.vgrid[k][2];
        }
        m
    }

    /// Second moment (total energy per volume): `rho*E = 0.5 * sum_k w_k *
    /// |v_k|^2 * f_k` — no `h`-term needed (see module/`maxwellian3d` docs):
    /// all 3 DOF are already carried by `|v_k|^2` directly.
    #[inline]
    pub fn moment_energy(&self, cell: usize) -> f64 {
        let fs = self.cell_slice(cell);
        let mut e = 0.0;
        for (k, &fv) in fs.iter().enumerate() {
            let v2 = self.vgrid[k][0] * self.vgrid[k][0]
                + self.vgrid[k][1] * self.vgrid[k][1]
                + self.vgrid[k][2] * self.vgrid[k][2];
            e += 0.5 * self.vw[k] * v2 * fv;
        }
        e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution3d_cell_slice_indexing() {
        let vgrid = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let vw = vec![1.0, 1.0, 1.0];
        let mut d = Distribution3D::zeros(2, vgrid, vw);
        assert_eq!(d.ncells(), 2);
        d.cell_slice_mut(1)[2] = 5.0;
        assert_eq!(d.f[d.index(1, 2)], 5.0);
        assert_eq!(d.cell_slice(1)[2], 5.0);
    }

    #[test]
    fn distribution3d_moments_of_symmetric_set_are_zero_momentum() {
        let vgrid = vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let vw = vec![1.0 / 6.0; 6];
        let mut d = Distribution3D::zeros(1, vgrid, vw);
        for k in 0..6 {
            d.cell_slice_mut(0)[k] = 2.0;
        }
        assert!((d.moment_rho(0) - 2.0).abs() < 1e-12);
        let mom = d.moment_mom(0);
        assert!(mom[0].abs() < 1e-12 && mom[1].abs() < 1e-12 && mom[2].abs() < 1e-12);
    }

    #[test]
    fn cell_slice_indexing() {
        let vgrid = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let vw = vec![1.0, 1.0, 1.0];
        let mut d = Distribution::zeros(2, vgrid, vw);
        assert_eq!(d.ncells(), 2);
        d.cell_slice_mut(1)[2] = 5.0;
        assert_eq!(d.f[d.index(1, 2)], 5.0);
        assert_eq!(d.cell_slice(1)[2], 5.0);
    }

    #[test]
    fn moments_of_uniform_maxwellian_like_set() {
        // A trivial symmetric velocity set with equal weights at +-1 in x,y
        // and zero mean should give zero momentum for uniform f.
        let vgrid = vec![[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]];
        let vw = vec![0.25, 0.25, 0.25, 0.25];
        let mut d = Distribution::zeros(1, vgrid, vw);
        for k in 0..4 {
            d.cell_slice_mut(0)[k] = 2.0;
        }
        assert!((d.moment_rho(0) - 2.0).abs() < 1e-12);
        let mom = d.moment_mom(0);
        assert!(mom[0].abs() < 1e-12);
        assert!(mom[1].abs() < 1e-12);
    }
}
