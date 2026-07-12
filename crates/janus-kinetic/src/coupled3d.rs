//! 3D UGKWP wave/particle coupling — direct generalization of
//! `coupled::UgkwpSolver` to `DugksSolver3D` + `Particles3D`. Same
//! continuous `p_free = exp(-dt/tau)` local-collision-probability split as
//! the 2D solver (Liu, Zhu & Xu 2020); see `coupled.rs` module docs for the
//! full derivation, which applies unchanged (the split logic operates on
//! macroscopic moments and is dimension-agnostic; only the moment tuple
//! width (4 -> 5: mass + 3 momentum components + energy) and BC dispatch
//! (4 edges -> 6 faces) differ from 2D).
//!
//! `FluxKernel3D::Dugks` remains available as the spec-sanctioned optional
//! pure-continuum fast path (ENGINEERING_SPEC.md §2), mirroring
//! `coupled::FluxKernel`; `FluxKernel3D::Ugkwp` (the default) is the full
//! mandatory wave/particle split.

use crate::collision3d::Collision3D;
use crate::particles3d::{Particles3D, Rng};
use crate::solver3d::DugksSolver3D;
use janus_core::config::{BoundaryAssignment3D, BoundaryKind3D, CaseConfig3D};
use janus_core::distribution::Distribution3D;

/// Same engineering-only gating role as `coupled::DEFAULT_KN_THRESHOLD`.
pub const DEFAULT_KN_THRESHOLD: f64 = 0.1;

/// Same role as `coupled::PARTICLES_PER_CELL`.
pub const PARTICLES_PER_CELL: usize = 64;

/// See `coupled::FluxKernel` docs — identical semantics, 3D types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FluxKernel3D {
    #[default]
    Ugkwp,
    Dugks,
}

pub struct UgkwpSolver3D {
    pub wave: DugksSolver3D,
    pub particles: Particles3D,
    pub kn_threshold: f64,
    pub kernel: FluxKernel3D,
    rng: Rng,
    p_free_scratch: Vec<f64>,
    tau_scratch: Vec<f64>,
}

impl UgkwpSolver3D {
    pub fn new(config: &CaseConfig3D, dist: Distribution3D, seed: u64) -> Self {
        let wave = DugksSolver3D::new(config, dist);
        let ncells = config.grid.ncells();
        Self {
            wave,
            particles: Particles3D::with_capacity(ncells * PARTICLES_PER_CELL),
            kn_threshold: DEFAULT_KN_THRESHOLD,
            kernel: FluxKernel3D::Ugkwp,
            rng: Rng::new(seed),
            p_free_scratch: vec![0.0; ncells],
            tau_scratch: vec![0.0; ncells],
        }
    }

    const MIN_PARTICLE_FRACTION: f64 = 1e-3;

    pub fn totals(&self) -> (f64, f64, f64, f64, f64) {
        // Wave field + in-flight particles (the split leaves the p_free fraction
        // as undeposited particles at end of step; the wave field alone
        // undercounts, dominant in the free-molecular limit). Particle totals
        // are EXTENSIVE, matching wave.totals()'s volume-weighted sum.
        let (wm, wpx, wpy, wpz, we) = self.wave.totals();
        let (pm, ppx, ppy, ppz, pe) = self.particles.totals();
        (wm + pm, wpx + ppx, wpy + ppy, wpz + ppz, we + pe)
    }

    /// Local Kn via the same gradient-length-local proxy as 2D
    /// (`kn::update_kn_loc`), generalized to 3 spatial gradients.
    fn refresh_kn(&mut self) {
        let grid = self.wave.grid;
        let r_gas = self.wave.gas_r;
        let rho = self.wave.fields.rho.clone();
        let mut mu = vec![0.0; grid.ncells()];
        for c in 0..grid.ncells() {
            let t = self
                .wave
                .fields
                .temperature(c, r_gas, crate::maxwellian3d::DOF);
            mu[c] = janus_core::units::vhs_viscosity(
                t,
                self.wave.mu_ref,
                self.wave.t_ref,
                self.wave.omega,
            );
        }
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = grid.idx(i, j, k);
                    let rho_c = rho[c].max(f64::MIN_POSITIVE);
                    let (rho_w, rho_e) = if nx > 1 {
                        let w = if i > 0 {
                            rho[grid.idx(i - 1, j, k)]
                        } else {
                            rho[c]
                        };
                        let e = if i + 1 < nx {
                            rho[grid.idx(i + 1, j, k)]
                        } else {
                            rho[c]
                        };
                        (w, e)
                    } else {
                        (rho_c, rho_c)
                    };
                    let (rho_s, rho_n) = if ny > 1 {
                        let s = if j > 0 {
                            rho[grid.idx(i, j - 1, k)]
                        } else {
                            rho[c]
                        };
                        let n = if j + 1 < ny {
                            rho[grid.idx(i, j + 1, k)]
                        } else {
                            rho[c]
                        };
                        (s, n)
                    } else {
                        (rho_c, rho_c)
                    };
                    let (rho_d, rho_u) = if nz > 1 {
                        let d = if k > 0 {
                            rho[grid.idx(i, j, k - 1)]
                        } else {
                            rho[c]
                        };
                        let u = if k + 1 < nz {
                            rho[grid.idx(i, j, k + 1)]
                        } else {
                            rho[c]
                        };
                        (d, u)
                    } else {
                        (rho_c, rho_c)
                    };
                    let drho_dx = (rho_e - rho_w) / (2.0 * grid.dx);
                    let drho_dy = (rho_n - rho_s) / (2.0 * grid.dy);
                    let drho_dz = (rho_u - rho_d) / (2.0 * grid.dz);
                    let grad_mag =
                        (drho_dx * drho_dx + drho_dy * drho_dy + drho_dz * drho_dz).sqrt();
                    let t_c = self
                        .wave
                        .fields
                        .temperature(c, r_gas, crate::maxwellian3d::DOF);
                    let lambda = janus_core::units::vhs_mean_free_path(mu[c], rho_c, r_gas, t_c);
                    self.wave.fields.kn_loc[c] = lambda * grad_mag / rho_c;
                }
            }
        }
    }

    /// One combined UGKWP step — same 5-phase structure as
    /// `coupled::UgkwpSolver::step` (wave step, Kn refresh, particle
    /// split/deposit-then-resplit, free transport + BC dispatch + collision,
    /// final deposit), generalized to 3 spatial dims and 6 domain faces.
    pub fn step(&mut self, dt: f64, config_bcs: &BoundaryAssignment3D) {
        if self.kernel == FluxKernel3D::Dugks {
            self.wave.step(dt, config_bcs);
            self.refresh_kn();
            return;
        }

        self.wave.step(dt, config_bcs);
        self.refresh_kn();

        let grid = self.wave.grid;
        let ncells = grid.ncells();
        let vol = grid.dx * grid.dy * grid.dz;
        let r_gas = self.wave.gas_r;

        // Recombine particles into the wave DISTRIBUTION f (not the macro
        // field, which update_moments recomputes from f and would discard —
        // see the 2D `coupled.rs` recombine note; this is the fix for the
        // free-molecular mass blow-up). Proportional per-cell scaling: exact in
        // mass, exact in all moments when particle and wave share (u,T).
        {
            let nv = self.wave.dist.nv;
            // Per-cell particle EXTENSIVE moments (mass, 3 momenta, energy);
            // rebuild each recombined cell as a conservatively-corrected
            // equilibrium at the true recombined moments (exact in mass,
            // momentum, energy — proportional scaling would err in momentum/
            // energy once particles drift from the wave's (u,T)).
            let mut pmass = vec![0.0f64; ncells];
            let mut pmx = vec![0.0f64; ncells];
            let mut pmy = vec![0.0f64; ncells];
            let mut pmz = vec![0.0f64; ncells];
            let mut pe = vec![0.0f64; ncells];
            for pi in 0..self.particles.len() {
                let cc = self.particles.cell[pi] as usize;
                if cc < ncells {
                    let w = self.particles.weight[pi];
                    let vv = self.particles.vel[pi];
                    pmass[cc] += w;
                    pmx[cc] += w * vv[0];
                    pmy[cc] += w * vv[1];
                    pmz[cc] += w * vv[2];
                    pe[cc] += w * 0.5 * (vv[0] * vv[0] + vv[1] * vv[1] + vv[2] * vv[2]);
                }
            }
            for c in 0..ncells {
                if pmass[c] <= 0.0 {
                    continue;
                }
                let m = self.wave.fields.rho[c] * vol + pmass[c];
                let mx = self.wave.fields.mom[0][c] * vol + pmx[c];
                let my = self.wave.fields.mom[1][c] * vol + pmy[c];
                let mz = self.wave.fields.mom[2][c] * vol + pmz[c];
                let e = self.wave.fields.energy[c] * vol + pe[c];
                if m <= 0.0 {
                    continue;
                }
                let t_ref = self.wave.t_ref;
                let d = &mut self.wave.dist;
                let f_cell = &mut d.f[c * nv..c * nv + nv];
                set_conservative_equilibrium_3d(
                    f_cell,
                    &d.vgrid,
                    &d.vw,
                    r_gas,
                    t_ref,
                    m / vol,
                    mx / vol,
                    my / vol,
                    mz / vol,
                    e / vol,
                );
            }
            self.particles.clear();
            self.wave.update_moments();
        }

        for c in 0..ncells {
            let rho = self.wave.fields.rho[c];
            let t = self
                .wave
                .fields
                .temperature(c, r_gas, crate::maxwellian3d::DOF);
            self.p_free_scratch[c] = if rho > 0.0 {
                let tau = self.wave.collision.relaxation_time(
                    rho,
                    t,
                    r_gas,
                    self.wave.mu_ref,
                    self.wave.t_ref,
                    self.wave.omega,
                );
                if tau.is_finite() && tau > 0.0 {
                    (-dt / tau).exp()
                } else {
                    0.0
                }
            } else {
                0.0
            };
        }

        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let c = grid.idx(i, j, k);
                    if self.wave.fields.kn_loc[c] <= self.kn_threshold {
                        continue;
                    }
                    let p_free = self.p_free_scratch[c];
                    if p_free < Self::MIN_PARTICLE_FRACTION {
                        continue;
                    }
                    let rho = self.wave.fields.rho[c];
                    let rho_vol_total = rho * vol;
                    if rho_vol_total <= 0.0 {
                        continue;
                    }
                    let rho_vol_particle = rho_vol_total * p_free;
                    let u = self.wave.fields.velocity(c);
                    let t = self
                        .wave
                        .fields
                        .temperature(c, r_gas, crate::maxwellian3d::DOF);
                    let center = grid.center(i, j, k);
                    let half_extent = [grid.dx * 0.5, grid.dy * 0.5, grid.dz * 0.5];
                    self.particles.sample_cell(
                        &mut self.rng,
                        c as u32,
                        center,
                        half_extent,
                        PARTICLES_PER_CELL,
                        rho_vol_particle,
                        u,
                        t,
                        r_gas,
                    );
                    // Remove the p_free fraction by scaling the wave
                    // DISTRIBUTION (macro-field scaling would be discarded).
                    let keep = 1.0 - p_free;
                    let nv = self.wave.dist.nv;
                    for kk in 0..nv {
                        self.wave.dist.f[c * nv + kk] *= keep;
                    }
                }
            }
        }
        // Refresh macro fields from the split-scaled distribution so totals()
        // and the next wave.step see the correct wave fraction.
        self.wave.update_moments();

        self.particles.free_transport(dt);
        let lx = grid.nx as f64 * grid.dx;
        let ly = grid.ny as f64 * grid.dy;
        let lz = grid.nz as f64 * grid.dz;
        let ox = grid.origin[0];
        let oy = grid.origin[1];
        let oz = grid.origin[2];

        #[inline]
        fn is_periodic(k: &BoundaryKind3D) -> bool {
            matches!(k, BoundaryKind3D::Periodic)
        }
        #[inline]
        fn is_absorbing(k: &BoundaryKind3D) -> bool {
            matches!(
                k,
                BoundaryKind3D::VelocityInlet { .. }
                    | BoundaryKind3D::PressureInlet { .. }
                    | BoundaryKind3D::Outlet
            )
        }

        let west_periodic = is_periodic(&config_bcs.west);
        let east_periodic = is_periodic(&config_bcs.east);
        let south_periodic = is_periodic(&config_bcs.south);
        let north_periodic = is_periodic(&config_bcs.north);
        let down_periodic = is_periodic(&config_bcs.down);
        let up_periodic = is_periodic(&config_bcs.up);
        let r_gas = self.wave.gas_r;

        let on_boundary = |p: &mut [f64; 3], v: &mut [f64; 3], rng: &mut Rng| {
            // x
            if west_periodic && east_periodic {
                while p[0] < ox {
                    p[0] += lx;
                }
                while p[0] >= ox + lx {
                    p[0] -= lx;
                }
            } else if p[0] < ox {
                if is_absorbing(&config_bcs.west) {
                    return false;
                }
                if let BoundaryKind3D::DiffuseWall {
                    temperature,
                    wall_velocity,
                } = config_bcs.west
                {
                    sample_wall_reemission(
                        rng,
                        [1.0, 0.0, 0.0],
                        wall_velocity,
                        temperature,
                        r_gas,
                        v,
                    );
                } else {
                    v[0] = -v[0];
                }
                p[0] = ox + (ox - p[0]);
            } else if p[0] >= ox + lx {
                if is_absorbing(&config_bcs.east) {
                    return false;
                }
                if let BoundaryKind3D::DiffuseWall {
                    temperature,
                    wall_velocity,
                } = config_bcs.east
                {
                    sample_wall_reemission(
                        rng,
                        [-1.0, 0.0, 0.0],
                        wall_velocity,
                        temperature,
                        r_gas,
                        v,
                    );
                } else {
                    v[0] = -v[0];
                }
                p[0] = ox + lx - (p[0] - (ox + lx));
            }
            // y
            if south_periodic && north_periodic {
                while p[1] < oy {
                    p[1] += ly;
                }
                while p[1] >= oy + ly {
                    p[1] -= ly;
                }
            } else if p[1] < oy {
                if is_absorbing(&config_bcs.south) {
                    return false;
                }
                if let BoundaryKind3D::DiffuseWall {
                    temperature,
                    wall_velocity,
                } = config_bcs.south
                {
                    sample_wall_reemission(
                        rng,
                        [0.0, 1.0, 0.0],
                        wall_velocity,
                        temperature,
                        r_gas,
                        v,
                    );
                } else {
                    v[1] = -v[1];
                }
                p[1] = oy + (oy - p[1]);
            } else if p[1] >= oy + ly {
                if is_absorbing(&config_bcs.north) {
                    return false;
                }
                if let BoundaryKind3D::DiffuseWall {
                    temperature,
                    wall_velocity,
                } = config_bcs.north
                {
                    sample_wall_reemission(
                        rng,
                        [0.0, -1.0, 0.0],
                        wall_velocity,
                        temperature,
                        r_gas,
                        v,
                    );
                } else {
                    v[1] = -v[1];
                }
                p[1] = oy + ly - (p[1] - (oy + ly));
            }
            // z
            if down_periodic && up_periodic {
                while p[2] < oz {
                    p[2] += lz;
                }
                while p[2] >= oz + lz {
                    p[2] -= lz;
                }
            } else if p[2] < oz {
                if is_absorbing(&config_bcs.down) {
                    return false;
                }
                if let BoundaryKind3D::DiffuseWall {
                    temperature,
                    wall_velocity,
                } = config_bcs.down
                {
                    sample_wall_reemission(
                        rng,
                        [0.0, 0.0, 1.0],
                        wall_velocity,
                        temperature,
                        r_gas,
                        v,
                    );
                } else {
                    v[2] = -v[2];
                }
                p[2] = oz + (oz - p[2]);
            } else if p[2] >= oz + lz {
                if is_absorbing(&config_bcs.up) {
                    return false;
                }
                if let BoundaryKind3D::DiffuseWall {
                    temperature,
                    wall_velocity,
                } = config_bcs.up
                {
                    sample_wall_reemission(
                        rng,
                        [0.0, 0.0, -1.0],
                        wall_velocity,
                        temperature,
                        r_gas,
                        v,
                    );
                } else {
                    v[2] = -v[2];
                }
                p[2] = oz + lz - (p[2] - (oz + lz));
            }
            true
        };
        self.particles.relocate(&grid, on_boundary, &mut self.rng);

        for c in 0..ncells {
            let rho = self.wave.fields.rho[c];
            let t = self
                .wave
                .fields
                .temperature(c, r_gas, crate::maxwellian3d::DOF);
            self.tau_scratch[c] = if rho > 0.0 {
                self.wave.collision.relaxation_time(
                    rho,
                    t,
                    r_gas,
                    self.wave.mu_ref,
                    self.wave.t_ref,
                    self.wave.omega,
                )
            } else {
                f64::INFINITY
            };
        }
        let tau_per_cell: &[f64] = &self.tau_scratch;
        let mut idx_to_collide = Vec::new();
        self.particles.mark_for_collision(
            &mut self.rng,
            |cell| {
                let c = cell as usize;
                if c < tau_per_cell.len() && tau_per_cell[c].is_finite() && tau_per_cell[c] > 0.0 {
                    tau_per_cell[c]
                } else {
                    1e-6
                }
            },
            dt,
            &mut idx_to_collide,
        );
        redraw_collided_particles(&mut self.particles, &idx_to_collide, &mut self.rng);

        // Particles persist to the next step's recombine (into the wave
        // DISTRIBUTION). No end-of-step macro-field deposit: it would be
        // discarded by the next wave.step's update_moments and would
        // double-count against totals()'s explicit particle term. (See 2D note.)
    }
}

/// Same flux-weighted (Rayleigh-CDF) diffuse-wall re-emission construction
/// as `coupled::sample_wall_reemission`, generalized to 2 tangential
/// components (a 3D wall has 2 tangent directions, vs. 1 in 2D) — no
/// reduced `zeta` carrier needed (see `maxwellian3d` module docs).
#[inline]
fn sample_wall_reemission(
    rng: &mut Rng,
    inward: [f64; 3],
    wall_velocity: [f64; 3],
    temperature: f64,
    r_gas: f64,
    v: &mut [f64; 3],
) {
    let rt = r_gas * temperature;
    let std_dev = rt.max(0.0).sqrt();

    let u1 = rng.uniform().min(1.0 - 1e-15);
    let c_n = (-2.0 * rt * (1.0 - u1).ln()).sqrt();
    let c_t1 = std_dev * rng.normal();
    let c_t2 = std_dev * rng.normal();

    // Two tangential unit vectors orthogonal to `inward` and to each other.
    // `inward` is always exactly one of the 6 axis-aligned unit vectors for
    // this Cartesian grid's faces, so a simple axis-permutation construction
    // gives an exact orthonormal tangent basis (no Gram-Schmidt needed).
    let (t1, t2) = orthonormal_tangents(inward);

    v[0] = wall_velocity[0] + inward[0] * c_n + t1[0] * c_t1 + t2[0] * c_t2;
    v[1] = wall_velocity[1] + inward[1] * c_n + t1[1] * c_t1 + t2[1] * c_t2;
    v[2] = wall_velocity[2] + inward[2] * c_n + t1[2] * c_t1 + t2[2] * c_t2;
}

/// Construct two orthonormal tangent vectors for an axis-aligned unit
/// normal `n` (one of `+-ex, +-ey, +-ez`). Since `n` is always exactly one
/// axis-aligned unit vector for this solver's Cartesian grid faces, the
/// other two axes are already mutually orthogonal and orthogonal to `n` —
/// no cross-product/Gram-Schmidt construction is needed, just picking the
/// two axes that are not `n`'s axis.
#[inline]
fn orthonormal_tangents(n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    if n[0].abs() > 0.5 {
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    } else if n[1].abs() > 0.5 {
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    } else {
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    }
}

/// Same moment-matched batched redraw as `coupled::redraw_collided_particles`,
/// generalized to 3 velocity components, DOF=3 directly (no zeta term).
fn redraw_collided_particles(particles: &mut Particles3D, indices: &[usize], rng: &mut Rng) {
    if indices.is_empty() {
        return;
    }
    use std::collections::HashMap;
    let mut by_cell: HashMap<u32, Vec<usize>> = HashMap::new();
    for &i in indices {
        by_cell.entry(particles.cell[i]).or_default().push(i);
    }

    for (_cell, idxs) in by_cell {
        let n = idxs.len();
        if n == 0 {
            continue;
        }
        let mut mass = 0.0;
        let mut mom = [0.0, 0.0, 0.0];
        for &i in &idxs {
            let w = particles.weight[i];
            mass += w;
            mom[0] += w * particles.vel[i][0];
            mom[1] += w * particles.vel[i][1];
            mom[2] += w * particles.vel[i][2];
        }
        if mass <= 0.0 {
            continue;
        }
        let u = [mom[0] / mass, mom[1] / mass, mom[2] / mass];
        let mut e2 = 0.0;
        for &i in &idxs {
            let w = particles.weight[i];
            let cx = particles.vel[i][0] - u[0];
            let cy = particles.vel[i][1] - u[1];
            let cz = particles.vel[i][2] - u[2];
            e2 += w * (cx * cx + cy * cy + cz * cz);
        }
        // Equipartition over DOF=3 (no reduced term): `e2 = sum w*|c|^2` has
        // units of `mass * velocity^2`, so `e2/mass` is `DOF*R*T` directly
        // (velocity^2 units) — the Gaussian draw's variance `R*T` is exactly
        // `e2/(mass*DOF)`, no separate `r_gas` needed since it is already
        // folded into `e2`'s velocity-squared units (same trick as the 2D
        // mirror `coupled::redraw_collided_particles`, which likewise never
        // takes `r_gas` as a parameter for this reason).
        let rt = (e2 / mass) / 3.0;
        let std_dev = rt.max(0.0).sqrt();

        for &i in &idxs {
            particles.vel[i][0] = u[0] + std_dev * rng.normal();
            particles.vel[i][1] = u[1] + std_dev * rng.normal();
            particles.vel[i][2] = u[2] + std_dev * rng.normal();
        }
        let mut mom2 = [0.0, 0.0, 0.0];
        for &i in &idxs {
            mom2[0] += particles.weight[i] * particles.vel[i][0];
            mom2[1] += particles.weight[i] * particles.vel[i][1];
            mom2[2] += particles.weight[i] * particles.vel[i][2];
        }
        let u2 = [mom2[0] / mass, mom2[1] / mass, mom2[2] / mass];
        for &i in &idxs {
            particles.vel[i][0] += u[0] - u2[0];
            particles.vel[i][1] += u[1] - u2[1];
            particles.vel[i][2] += u[2] - u2[2];
        }
        let mut e3 = 0.0;
        for &i in &idxs {
            let w = particles.weight[i];
            let cx = particles.vel[i][0] - u[0];
            let cy = particles.vel[i][1] - u[1];
            let cz = particles.vel[i][2] - u[2];
            e3 += w * (cx * cx + cy * cy + cz * cz);
        }
        if e3 > 1e-300 {
            let scale = (e2 / e3).sqrt();
            for &i in &idxs {
                let cx = (particles.vel[i][0] - u[0]) * scale;
                let cy = (particles.vel[i][1] - u[1]) * scale;
                let cz = (particles.vel[i][2] - u[2]) * scale;
                particles.vel[i][0] = u[0] + cx;
                particles.vel[i][1] = u[1] + cy;
                particles.vel[i][2] = u[2] + cz;
            }
        }
    }
}

/// Overwrite a cell's 3D distribution `f` with the Maxwellian equilibrium at the
/// target moments, then conservatively correct it (5x5 solve, peculiar-velocity
/// basis `{1,cx,cy,cz,c^2}`) so its discrete mass, 3 momenta, and energy equal
/// the targets to machine precision. Same construction as `DugksSolver3D`'s
/// relaxation correction; used by the UGKWP recombine.
#[allow(clippy::too_many_arguments)]
fn set_conservative_equilibrium_3d(
    f: &mut [f64],
    vgrid: &[[f64; 3]],
    vw: &[f64],
    r_gas: f64,
    t_ref: f64,
    tgt_rho: f64,
    tgt_mx: f64,
    tgt_my: f64,
    tgt_mz: f64,
    tgt_e: f64,
) {
    let nv = vw.len();
    let u = [tgt_mx / tgt_rho, tgt_my / tgt_rho, tgt_mz / tgt_rho];
    let umag2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    let t = (((2.0 * tgt_e / tgt_rho) - umag2) / (crate::maxwellian3d::DOF * r_gas)).max(1e-6);
    // Normalize the peculiar basis by the GRID reference thermal scale (constant,
    // always sane) rather than the local `t`, which can floor to ~1e-6 for a
    // transiently corrupted cell and OVER-normalize into ill-conditioning (see
    // the solver3d relaxation note). Normalization is a pure conditioning device.
    let sig = (r_gas * t_ref).sqrt().max(1e-3);
    let inv_sig = 1.0 / sig;
    let inv_sig2 = inv_sig * inv_sig;

    let mut f1m = [0.0f64; 5];
    let mut a = [[0.0f64; 5]; 5];
    for k in 0..nv {
        let v = vgrid[k];
        let w = vw[k];
        let feq = crate::maxwellian3d::maxwellian_3d(tgt_rho, u, t, r_gas, v);
        f[k] = feq;
        let (vx, vy, vz) = (v[0], v[1], v[2]);
        let v2 = vx * vx + vy * vy + vz * vz;
        f1m[0] += w * feq;
        f1m[1] += w * vx * feq;
        f1m[2] += w * vy * feq;
        f1m[3] += w * vz * feq;
        f1m[4] += 0.5 * w * v2 * feq;
        let cx = vx - u[0];
        let cy = vy - u[1];
        let cz = vz - u[2];
        let c2 = cx * cx + cy * cy + cz * cz;
        let psi = [1.0, vx, vy, vz, 0.5 * v2];
        let phi = [1.0, cx * inv_sig, cy * inv_sig, cz * inv_sig, c2 * inv_sig2];
        let e = w * feq;
        for i in 0..5 {
            for j in 0..5 {
                a[i][j] += e * psi[i] * phi[j];
            }
        }
    }
    let rhs = [
        tgt_rho - f1m[0],
        tgt_mx - f1m[1],
        tgt_my - f1m[2],
        tgt_mz - f1m[3],
        tgt_e - f1m[4],
    ];
    let coef = crate::solver3d::solve5(a, rhs).unwrap_or([0.0; 5]);
    for k in 0..nv {
        let v = vgrid[k];
        let cx = v[0] - u[0];
        let cy = v[1] - u[1];
        let cz = v[2] - u[2];
        let c2 = cx * cx + cy * cy + cz * cz;
        f[k] += f[k]
            * (coef[0]
                + coef[1] * cx * inv_sig
                + coef[2] * cy * inv_sig
                + coef[3] * cz * inv_sig
                + coef[4] * c2 * inv_sig2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_core::config::GasProperties;
    use janus_core::grid3d::Grid3D;

    fn periodic_case(nx: usize, ny: usize, nz: usize, d: f64) -> CaseConfig3D {
        CaseConfig3D {
            grid: Grid3D::new(nx, ny, nz, d, d, d, [0.0, 0.0, 0.0]),
            bcs: BoundaryAssignment3D::all_periodic(),
            gas: GasProperties::monatomic_default(),
        }
    }

    fn init_uniform(solver: &mut UgkwpSolver3D, rho: f64, u: [f64; 3], t: f64) {
        let nv = solver.wave.dist.nv;
        let ncells = solver.wave.grid.ncells();
        let r_gas = solver.wave.gas_r;
        let vgrid = solver.wave.dist.vgrid.clone();
        for c in 0..ncells {
            for k in 0..nv {
                let f = crate::maxwellian3d::maxwellian_3d(rho, u, t, r_gas, vgrid[k]);
                solver.wave.dist.f[c * nv + k] = f;
            }
        }
        solver.wave.update_moments();
    }

    #[test]
    fn default_kernel_is_full_ugkwp_3d() {
        let config = periodic_case(2, 2, 2, 0.02);
        let (vgrid, vw) = crate::velocity_grid3d::VelocityGrid3D::gauss_hermite(
            287.0,
            1500.0,
            [0.0, 0.0, 0.0],
            5,
        );
        let dist = Distribution3D::zeros(config.grid.ncells(), vgrid, vw);
        let solver = UgkwpSolver3D::new(&config, dist, 1);
        assert_eq!(solver.kernel, FluxKernel3D::Ugkwp);
    }

    // Genuinely forced full-particle 3D UGKWP conservation.
    //
    // Two things this test now does correctly (both were wrong before):
    //  1. It ACTUALLY forces the particle path. A huge `mu_ref` makes tau huge
    //     so `p_free = exp(-dt/tau) ~ 1` (Kn>>1), and a NEGATIVE `kn_threshold`
    //     makes the `kn_loc <= kn_threshold` gate never skip a cell (kn_loc >= 0
    //     > -1). The previous `kn_threshold = 0` silently skipped every cell for
    //     a near-uniform field (gradient-based kn_loc ~ 0 <= 0), so it only ever
    //     exercised the wave solver — not the coupling it claims to test. This
    //     mirrors the 2D `extreme_free_molecular` setup.
    //  2. It uses a velocity grid that RESOLVES the flow: Gauss-Hermite placed at
    //     the flow (u_ref = flow velocity, t_ref ~ flow temperature) with n=8.
    //     An under-resolved / mis-placed grid (e.g. t_ref=1500, u_ref=0, n=5 for
    //     a T=320 flow) integrates the initial Maxwellian to rho~1.67 instead of
    //     1.0 — garbage discrete moments that produce a stiff blow-up regardless
    //     of solver correctness. With the resolved grid the discrete moments are
    //     exact to ~5 digits.
    //
    // Validates that the distribution-based split/recombine conserves mass,
    // momentum, AND energy through many forced wave<->particle exchanges.
    #[test]
    fn combined_wave_particle_conserves_mass_and_energy_3d() {
        let mut gas = GasProperties::monatomic_default();
        gas.mu_ref = 1.0; // huge viscosity -> huge tau -> p_free ~ 1 (force particles)
        let config = CaseConfig3D {
            grid: Grid3D::new(3, 3, 3, 0.02, 0.02, 0.02, [0.0, 0.0, 0.0]),
            bcs: BoundaryAssignment3D::all_periodic(),
            gas,
        };
        let u_flow = [15.0, -8.0, 4.0];
        // Grid placed at the flow, refined, so the GH quadrature resolves it.
        let (vgrid, vw) =
            crate::velocity_grid3d::VelocityGrid3D::gauss_hermite(287.0, 400.0, u_flow, 8);
        let dist = Distribution3D::zeros(config.grid.ncells(), vgrid, vw);
        let mut solver = UgkwpSolver3D::new(&config, dist, 42);
        solver.kn_threshold = -1.0; // never gate out a cell -> particle path forced everywhere
        init_uniform(&mut solver, 1.0, u_flow, 320.0);

        let c0 = solver.wave.grid.idx(1, 1, 1);
        let nv = solver.wave.dist.nv;
        for k in 0..nv {
            solver.wave.dist.f[c0 * nv + k] *= 1.1;
        }
        solver.wave.update_moments();

        let before = solver.totals();
        let dt = solver.wave.cfl_dt(0.15);
        for step in 0..10 {
            solver.step(dt, &config.bcs);
            let now = solver.totals();
            assert!(
                now.0.is_finite()
                    && now.1.is_finite()
                    && now.2.is_finite()
                    && now.3.is_finite()
                    && now.4.is_finite(),
                "non-finite at step {step}"
            );
        }
        let after = solver.totals();

        let tol = 5e-2;
        assert!(
            (after.0 - before.0).abs() / before.0.abs().max(1e-300) < tol,
            "mass drift: before={} after={}",
            before.0,
            after.0
        );
        assert!(
            (after.1 - before.1).abs() / before.1.abs().max(1e-30) < tol,
            "x-momentum drift: before={} after={}",
            before.1,
            after.1
        );
        assert!(
            (after.4 - before.4).abs() / before.4.abs().max(1e-300) < tol,
            "energy drift: before={} after={}",
            before.4,
            after.4
        );
    }
}
