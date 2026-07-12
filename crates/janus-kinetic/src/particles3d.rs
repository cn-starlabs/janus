//! 3D stochastic particle layer (UGKWP), generalizing `particles::Particles`
//! to 3D physical + velocity coordinates. No `zeta`/reduced-h carrier is
//! needed here (see `maxwellian3d` module docs): a 3D particle's velocity
//! `[vx, vy, vz]` already carries all 3 translational DOF directly, so there
//! is nothing left to reduce out and no auxiliary energy-carrying scalar
//! sample is required (unlike the 2D solver's `zeta`, which stands in for
//! the physically-real third velocity component that a 2D velocity-space
//! particle cannot otherwise represent).
//!
//! Reference: Liu, S., Zhu, Y., Xu, K., "A unified gas kinetic wave-particle
//! method I: Continuum and rarefied gas dynamics", J. Comput. Phys. 401,
//! 108977 (2020).

use janus_core::fields3d::MacroFields3D;
use janus_core::grid3d::Grid3D;

pub use crate::particles::Rng;

/// SoA 3D particle storage, direct generalization of `particles::Particles`.
#[derive(Clone, Debug, Default)]
pub struct Particles3D {
    pub pos: Vec<[f64; 3]>,
    pub vel: Vec<[f64; 3]>,
    pub weight: Vec<f64>,
    pub cell: Vec<u32>,
}

impl Particles3D {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            pos: Vec::with_capacity(cap),
            vel: Vec::with_capacity(cap),
            weight: Vec::with_capacity(cap),
            cell: Vec::with_capacity(cap),
        }
    }

    pub fn len(&self) -> usize {
        self.pos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    pub fn clear(&mut self) {
        self.pos.clear();
        self.vel.clear();
        self.weight.clear();
        self.cell.clear();
    }

    /// Total (mass, px, py, pz, energy). Energy per particle:
    /// `weight * 0.5*|v|^2` — no reduced-direction term needed (see module
    /// docs), unlike the 2D `Particles::totals`'s `+ 0.5*zeta^2`.
    pub fn totals(&self) -> (f64, f64, f64, f64, f64) {
        let mut mass = 0.0;
        let mut px = 0.0;
        let mut py = 0.0;
        let mut pz = 0.0;
        let mut e = 0.0;
        for i in 0..self.len() {
            let w = self.weight[i];
            let v = self.vel[i];
            mass += w;
            px += w * v[0];
            py += w * v[1];
            pz += w * v[2];
            e += w * 0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        }
        (mass, px, py, pz, e)
    }

    /// Sample `n` particles from a local Maxwellian, exact-moment corrected
    /// (identical "quiet start" rescale trick as `Particles::sample_cell`,
    /// generalized to 3 velocity components and DOF=3 directly — no zeta).
    #[allow(clippy::too_many_arguments)]
    pub fn sample_cell(
        &mut self,
        rng: &mut Rng,
        cell: u32,
        cell_center: [f64; 3],
        half_extent: [f64; 3],
        n: usize,
        rho_vol: f64,
        u: [f64; 3],
        t: f64,
        r_gas: f64,
    ) {
        if n == 0 || rho_vol <= 0.0 {
            return;
        }
        let w = rho_vol / n as f64;
        let std_dev = (r_gas * t).max(0.0).sqrt();
        let start = self.len();
        for _ in 0..n {
            let x = cell_center[0] + (rng.uniform() * 2.0 - 1.0) * half_extent[0];
            let y = cell_center[1] + (rng.uniform() * 2.0 - 1.0) * half_extent[1];
            let z = cell_center[2] + (rng.uniform() * 2.0 - 1.0) * half_extent[2];
            let vx = u[0] + std_dev * rng.normal();
            let vy = u[1] + std_dev * rng.normal();
            let vz = u[2] + std_dev * rng.normal();
            self.pos.push([x, y, z]);
            self.vel.push([vx, vy, vz]);
            self.weight.push(w);
            self.cell.push(cell);
        }

        let end = self.len();
        let mut mom = [0.0, 0.0, 0.0];
        for i in start..end {
            mom[0] += self.weight[i] * self.vel[i][0];
            mom[1] += self.weight[i] * self.vel[i][1];
            mom[2] += self.weight[i] * self.vel[i][2];
        }
        let mass_batch: f64 = self.weight[start..end].iter().sum();
        let mean_u = if mass_batch > 0.0 {
            [
                mom[0] / mass_batch,
                mom[1] / mass_batch,
                mom[2] / mass_batch,
            ]
        } else {
            [0.0, 0.0, 0.0]
        };
        for i in start..end {
            self.vel[i][0] += u[0] - mean_u[0];
            self.vel[i][1] += u[1] - mean_u[1];
            self.vel[i][2] += u[2] - mean_u[2];
        }
        let mut e2 = 0.0;
        for i in start..end {
            let cx = self.vel[i][0] - u[0];
            let cy = self.vel[i][1] - u[1];
            let cz = self.vel[i][2] - u[2];
            e2 += self.weight[i] * (cx * cx + cy * cy + cz * cz);
        }
        // Target internal energy sum: rho_vol * DOF * R * T, DOF=3 directly
        // (no reduced-direction contribution to add, unlike the 2D case).
        let target_internal_e2 = rho_vol * crate::maxwellian3d::DOF * r_gas * t;
        if e2 > 1e-300 {
            let scale = (target_internal_e2 / e2).sqrt();
            for i in start..end {
                let cx = (self.vel[i][0] - u[0]) * scale;
                let cy = (self.vel[i][1] - u[1]) * scale;
                let cz = (self.vel[i][2] - u[2]) * scale;
                self.vel[i][0] = u[0] + cx;
                self.vel[i][1] = u[1] + cy;
                self.vel[i][2] = u[2] + cz;
            }
        }
    }

    /// Free transport: advect every particle by `vel * dt`.
    pub fn free_transport(&mut self, dt: f64) {
        for i in 0..self.len() {
            self.pos[i][0] += self.vel[i][0] * dt;
            self.pos[i][1] += self.vel[i][1] * dt;
            self.pos[i][2] += self.vel[i][2] * dt;
        }
    }

    /// Stochastic BGK/Shakhov collision Bernoulli trial (per particle, its
    /// current cell's `tau`); identical construction to
    /// `Particles::mark_for_collision`.
    pub fn mark_for_collision(
        &self,
        rng: &mut Rng,
        tau_of_cell: impl Fn(u32) -> f64,
        dt: f64,
        out_indices: &mut Vec<usize>,
    ) {
        out_indices.clear();
        for i in 0..self.len() {
            let tau = tau_of_cell(self.cell[i]).max(1e-300);
            let p_collide = 1.0 - (-dt / tau).exp();
            if rng.uniform() < p_collide {
                out_indices.push(i);
            }
        }
    }

    /// Re-aggregate particles into `MacroFields3D` (adds to existing
    /// contents, never overwrites — same convention as 2D).
    pub fn deposit_moments(&self, grid: &Grid3D, fields: &mut MacroFields3D) {
        // Particle weights are EXTENSIVE masses (rho*cell_volume/n); MacroFields
        // stores DENSITIES, so convert back by dividing by the cell volume.
        // (Omitting this inflated density every recombine by 1/cell_volume,
        // which for cells smaller than unit volume diverges to a non-finite
        // state within a step or two — the coupled3d "non-finite" failure.)
        let inv_vol = 1.0 / (grid.dx * grid.dy * grid.dz);
        for i in 0..self.len() {
            let c = self.cell[i] as usize;
            if c >= fields.ncells() {
                continue;
            }
            let w = self.weight[i] * inv_vol;
            let v = self.vel[i];
            fields.rho[c] += w;
            fields.mom[0][c] += w * v[0];
            fields.mom[1][c] += w * v[1];
            fields.mom[2][c] += w * v[2];
            fields.energy[c] += w * 0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        }
    }

    /// Re-index particles into their current cell after transport, with a
    /// caller-supplied `on_boundary` hook for the 6 domain faces (periodic
    /// wrap / specular mirror / diffuse re-emission / absorption), same
    /// `swap_remove`-based O(1) removal convention as `Particles::relocate`.
    pub fn relocate(
        &mut self,
        grid: &Grid3D,
        mut on_boundary: impl FnMut(&mut [f64; 3], &mut [f64; 3], &mut Rng) -> bool,
        rng: &mut Rng,
    ) {
        let mut i = 0;
        while i < self.len() {
            let mut p = self.pos[i];
            let mut v = self.vel[i];
            let mut keep = true;
            let ox = grid.origin[0];
            let oy = grid.origin[1];
            let oz = grid.origin[2];
            let lx = grid.nx as f64 * grid.dx;
            let ly = grid.ny as f64 * grid.dy;
            let lz = grid.nz as f64 * grid.dz;
            if p[0] < ox
                || p[0] >= ox + lx
                || p[1] < oy
                || p[1] >= oy + ly
                || p[2] < oz
                || p[2] >= oz + lz
            {
                keep = on_boundary(&mut p, &mut v, rng);
            }
            if !keep {
                self.pos.swap_remove(i);
                self.vel.swap_remove(i);
                self.weight.swap_remove(i);
                self.cell.swap_remove(i);
                continue;
            }
            self.pos[i] = p;
            self.vel[i] = v;
            let i_idx =
                (((p[0] - ox) / grid.dx).floor() as isize).clamp(0, grid.nx as isize - 1) as usize;
            let j_idx =
                (((p[1] - oy) / grid.dy).floor() as isize).clamp(0, grid.ny as isize - 1) as usize;
            let k_idx =
                (((p[2] - oz) / grid.dz).floor() as isize).clamp(0, grid.nz as isize - 1) as usize;
            self.cell[i] = grid.idx(i_idx, j_idx, k_idx) as u32;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampled_moments_match_target_exactly_3d() {
        let mut particles = Particles3D::with_capacity(2000);
        let mut rng = Rng::new(12345);
        let rho_vol = 2.5;
        let u = [10.0, -3.0, 4.0];
        let t = 350.0;
        let r_gas = 208.13;
        particles.sample_cell(
            &mut rng,
            0,
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            1000,
            rho_vol,
            u,
            t,
            r_gas,
        );

        let (mass, px, py, pz, e) = particles.totals();
        assert!((mass - rho_vol).abs() < 1e-9, "mass {mass} vs {rho_vol}");
        assert!((px - rho_vol * u[0]).abs() < 1e-6, "px {px}");
        assert!((py - rho_vol * u[1]).abs() < 1e-6, "py {py}");
        assert!((pz - rho_vol * u[2]).abs() < 1e-6, "pz {pz}");
        let kinetic = 0.5 * rho_vol * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        let internal = 0.5 * rho_vol * crate::maxwellian3d::DOF * r_gas * t;
        let expected_e = kinetic + internal;
        assert!(
            (e - expected_e).abs() / expected_e < 1e-6,
            "e {e} vs {expected_e}"
        );
    }

    #[test]
    fn free_transport_moves_particles_3d() {
        let mut p = Particles3D::with_capacity(4);
        p.pos.push([0.0, 0.0, 0.0]);
        p.vel.push([1.0, 2.0, 3.0]);
        p.weight.push(1.0);
        p.cell.push(0);
        p.free_transport(0.5);
        assert!((p.pos[0][0] - 0.5).abs() < 1e-12);
        assert!((p.pos[0][1] - 1.0).abs() < 1e-12);
        assert!((p.pos[0][2] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn relocate_periodic_wrap_3d() {
        let grid = Grid3D::new(2, 2, 2, 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]);
        let mut particles = Particles3D::with_capacity(1);
        particles.pos.push([2.1, 1.0, 1.0]);
        particles.vel.push([1.0, 0.0, 0.0]);
        particles.weight.push(1.0);
        particles.cell.push(0);
        let mut rng = Rng::new(1);
        let lx = 2.0;
        particles.relocate(
            &grid,
            |p, _v, _rng| {
                if p[0] >= lx {
                    p[0] -= lx;
                }
                true
            },
            &mut rng,
        );
        assert!(particles.pos[0][0] >= 0.0 && particles.pos[0][0] < lx);
    }
}
