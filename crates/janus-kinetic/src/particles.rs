//! UGKWP stochastic particle layer: SoA particle storage, free transport,
//! stochastic BGK/Shakhov collision, re-aggregation to cell moments, and the
//! wave/particle mass-momentum-energy exchange.
//!
//! Reference: Liu, S., Zhu, Y., Xu, K., "A unified gas kinetic wave-particle
//! method I: Continuum and rarefied gas dynamics", J. Comput. Phys. 401,
//! 108977 (2020).
//!
//! ## UGKWP split (per cell, per step)
//!
//! Each cell's distribution is split into a deterministic "wave" part and a
//! stochastic "particle" part according to the local collision-free fraction
//! `exp(-dt/tau)` (the fraction of the pre-step distribution that survives
//! one timestep without colliding, i.e. behaves like free-streaming
//! particles) versus `1 - exp(-dt/tau)` (the fraction that reaches local
//! equilibrium, handled deterministically as the "wave"/equilibrium part).
//! Cells with `kn_loc` below `kn_threshold` are treated as fully continuum
//! (wave-only, `DugksSolver` handles them exactly as in M1); cells at/above
//! the threshold spawn particles representing the *non-equilibrium residual*
//! of their distribution.
//!
//! For simplicity and strict conservation (the top priority per the spec's
//! "key correctness hazards"), this module implements the particle layer as
//! an explicit **local moment-conserving stochastic layer**: each rarefied
//! cell's mass/momentum/energy is (a) partly represented by explicit
//! simulation particles sampled from a local Maxwellian with the cell's
//! *current* macroscopic state, (b) freely transported for `dt`, (c)
//! stochastically collided (BGK relaxation applied as an analytic sub-step
//! per UGKWP, i.e. each particle's post-collision velocity is drawn from the
//! local equilibrium with probability `1 - exp(-dt/tau)`, otherwise kept
//! free-streaming — the standard DSMC/UGKWP no-time-counter-free formulation
//! of the same BGK relaxation used by the wave part), (d) re-binned to
//! moments. Exchange with the wave field is realized by having the wave
//! solver's own cell (rho, mom, E) *be* the target the particles are sampled
//! to reproduce and re-aggregated back into — i.e. particles carry a
//! *conservative* representation of a cell's moments for one step and hand
//! the (possibly redistributed, because of transport across cell faces)
//! moments back exactly, with no leakage, because every particle's weight
//! is tracked and summed exactly (floating-point-exact bookkeeping: the
//! total weight/momentum/energy removed from a cell of origin exactly
//! equals what is added to the destination cell(s) it moves into, since we
//! do the accounting from the particles themselves rather than from a
//! separately-integrated field).

use janus_core::fields::MacroFields;
use janus_core::grid::Grid2D;

/// Simple xorshift64* PRNG (no external `rand` dependency needed — this is
/// a small, well-known, deterministic generator; determinism is useful for
/// reproducible tests). Not cryptographic; fine for Monte-Carlo sampling.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal via Box-Muller.
    #[inline]
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Struct-of-Arrays particle storage (ENGINEERING_SPEC.md §5). Only
/// populated in rarefied (`kn_loc` above threshold) cells. All buffers are
/// preallocated with spare capacity and reused across steps — the hot free-
/// transport/collision loops never call into the allocator; only the
/// (rare, amortized) re-sampling step may `push`/`truncate` to change
/// particle count, which is unavoidable since particle count is inherently
/// dynamic in a Monte-Carlo scheme, but no *per-particle* allocation ever
/// happens (`Vec::with_capacity` + reuse).
#[derive(Clone, Debug, Default)]
pub struct Particles {
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    /// Reduced-h carrier: each simulation particle also carries a scalar
    /// "internal energy" sample `zeta` (representing the reduced-out `eta`
    /// velocity component's contribution), analogous to DSMC's internal
    /// energy sampling for the (g,h) reduction. Physical role: `zeta_i`
    /// samples the same `h`-moment distribution the wave part carries, so
    /// `h`-moments can be reconstructed from particles the same way
    /// `g`-moments are.
    pub zeta: Vec<f64>,
    pub weight: Vec<f64>,
    pub cell: Vec<u32>,
}

impl Particles {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            pos: Vec::with_capacity(cap),
            vel: Vec::with_capacity(cap),
            zeta: Vec::with_capacity(cap),
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
        self.zeta.clear();
        self.weight.clear();
        self.cell.clear();
    }

    /// Total (mass, px, py, energy) carried by all particles. Energy per
    /// particle: `weight * (0.5*|v|^2 + 0.5*zeta^2)` (translational +
    /// reduced-direction contribution), matching the (g,h) moment split.
    pub fn totals(&self) -> (f64, f64, f64, f64) {
        let mut mass = 0.0;
        let mut px = 0.0;
        let mut py = 0.0;
        let mut e = 0.0;
        for i in 0..self.len() {
            let w = self.weight[i];
            let v = self.vel[i];
            let z = self.zeta[i];
            mass += w;
            px += w * v[0];
            py += w * v[1];
            e += w * (0.5 * (v[0] * v[0] + v[1] * v[1]) + 0.5 * z * z);
        }
        (mass, px, py, e)
    }

    /// Sample `n` particles from a local Maxwellian with reduced-h carrier
    /// `zeta`, appending to the existing arrays (preallocated capacity
    /// should already cover this; callers size capacity generously up
    /// front). Each particle gets weight `mass_total / n` so the *sampled*
    /// total mass is exactly `mass_total` (not just in expectation) —
    /// this is the key trick that makes wave<->particle mass exchange
    /// exact: we do not rely on the sample mean converging, we force it by
    /// construction, matching momentum/energy via a rescale (see below).
    #[allow(clippy::too_many_arguments)]
    pub fn sample_cell(
        &mut self,
        rng: &mut Rng,
        cell: u32,
        cell_center: [f64; 2],
        half_extent: [f64; 2],
        n: usize,
        rho_vol: f64, // mass in this cell (rho * cell_volume)
        u: [f64; 2],
        t: f64,
        r_gas: f64,
    ) {
        self.sample_cell_with_dof(
            rng,
            cell,
            cell_center,
            half_extent,
            n,
            rho_vol,
            u,
            t,
            r_gas,
            crate::maxwellian::DOF,
        )
    }

    /// Polyatomic/internal-DOF-generalized particle sampler: identical
    /// moment-matched ("quiet start") construction as `sample_cell`, but
    /// with an explicit total DOF (`dof_total`, e.g.
    /// `crate::maxwellian::dof_with_internal(zeta_int)`) governing the
    /// target internal energy instead of the crate-wide monatomic constant.
    /// This is the particle-representation half of the polyatomic-DOF
    /// wiring: `zeta` here represents "all energy beyond the 2 discretized
    /// in-plane translational components" (the reduced third translational
    /// DOF for a monatomic gas, PLUS any internal rotational/vibrational DOF
    /// for a polyatomic gas) — exactly the same role the `h`-distribution's
    /// `k_total` parameter plays in `maxwellian::gh_equilibrium_with_k`, so
    /// a polyatomic particle population and its wave-side (g,h) counterpart
    /// stay moment-consistent when both are driven by the same `dof_total`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_cell_with_dof(
        &mut self,
        rng: &mut Rng,
        cell: u32,
        cell_center: [f64; 2],
        half_extent: [f64; 2],
        n: usize,
        rho_vol: f64, // mass in this cell (rho * cell_volume)
        u: [f64; 2],
        t: f64,
        r_gas: f64,
        dof_total: f64,
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
            let vx = u[0] + std_dev * rng.normal();
            let vy = u[1] + std_dev * rng.normal();
            // zeta ~ N(0, R*T) represents the reduced third velocity
            // component (K=1 reduced DOF) plus, for a polyatomic gas, its
            // internal rotational/vibrational energy — see this method's
            // doc comment. The per-sample draw shape is unaffected by
            // `dof_total`; only the exact-moment rescale below (which
            // targets `dof_total`) differs from the monatomic case.
            let zeta = std_dev * rng.normal();
            self.pos.push([x, y]);
            self.vel.push([vx, vy]);
            self.zeta.push(zeta);
            self.weight.push(w);
            self.cell.push(cell);
        }

        // Exact-moment correction: rescale velocities/zeta of the freshly
        // added batch so the *sampled* momentum and energy exactly match
        // the target macroscopic moments (finite-N Monte-Carlo sampling
        // otherwise only matches in expectation, which would violate the
        // "exact to floating-point tolerance" conservation requirement).
        // This is a standard variance-reduction trick (moment-matched/
        // "quiet start" sampling) used in particle methods when the target
        // requires deterministic conservation rather than pure MC noise.
        let end = self.len();
        let mut mom = [0.0, 0.0];
        let mut e2 = 0.0; // sum w*(v'^2+zeta'^2), v' = v-u peculiar velocity
        for i in start..end {
            mom[0] += self.weight[i] * self.vel[i][0];
            mom[1] += self.weight[i] * self.vel[i][1];
        }
        let mass_batch: f64 = self.weight[start..end].iter().sum();
        let mean_u = if mass_batch > 0.0 {
            [mom[0] / mass_batch, mom[1] / mass_batch]
        } else {
            [0.0, 0.0]
        };
        // Shift so sampled mean velocity exactly equals u.
        for i in start..end {
            self.vel[i][0] += u[0] - mean_u[0];
            self.vel[i][1] += u[1] - mean_u[1];
        }
        // Now rescale peculiar velocity/zeta so sampled internal energy
        // exactly matches rho*dof_total/2*R*T (dof_total=3 for monatomic:
        // 2 in-plane + 1 reduced; dof_total=3+zeta_int for polyatomic).
        for i in start..end {
            let cx = self.vel[i][0] - u[0];
            let cy = self.vel[i][1] - u[1];
            e2 += self.weight[i] * (cx * cx + cy * cy + self.zeta[i] * self.zeta[i]);
        }
        let target_internal_e2 = rho_vol * dof_total * r_gas * t; // = sum w*(c^2+zeta^2) target
        if e2 > 1e-300 {
            let scale = (target_internal_e2 / e2).sqrt();
            for i in start..end {
                let cx = (self.vel[i][0] - u[0]) * scale;
                let cy = (self.vel[i][1] - u[1]) * scale;
                self.vel[i][0] = u[0] + cx;
                self.vel[i][1] = u[1] + cy;
                self.zeta[i] *= scale;
            }
        }
    }

    /// Free transport: advect every particle by `vel * dt`. No allocation.
    /// Reflective/periodic handling of domain edges is delegated to the
    /// caller via `wrap` (keeps this function generic/allocation-free).
    pub fn free_transport(&mut self, dt: f64) {
        for i in 0..self.len() {
            self.pos[i][0] += self.vel[i][0] * dt;
            self.pos[i][1] += self.vel[i][1] * dt;
        }
    }

    /// Stochastic BGK/Shakhov relaxation sub-step (UGKWP collision): with
    /// probability `1 - exp(-dt/tau_c)` (per-particle, using its *current*
    /// cell's relaxation time), replace the particle's velocity/zeta with a
    /// fresh draw from the cell's local Maxwellian (moment-matched the same
    /// way as `sample_cell`, applied per-cell in batches after this pass —
    /// see `coupled::UgkwpSolver::step` for the batched re-draw that
    /// preserves per-cell conservation exactly. This function only performs
    /// the Bernoulli trial and marks which particles need replacement,
    /// returning the list of (index) that must be re-sampled; velocities of
    /// particles that are NOT replaced are left untouched.
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

    /// Re-aggregate particles into `MacroFields` (rho, mom, energy) by
    /// binning into `grid` cells. This ADDS to whatever is already in
    /// `fields` (caller should zero relevant cells first if particles are
    /// meant to be the sole contribution for those cells) so wave and
    /// particle contributions can be summed cell-by-cell for the combined
    /// domain total.
    pub fn deposit_moments(&self, grid: &Grid2D, fields: &mut MacroFields) {
        // Particle weights are EXTENSIVE masses (a particle sampled from a cell
        // of density rho carries weight rho*cell_volume/n; see `sample_cell`,
        // whose `rho_vol` argument is rho*cell_volume). `MacroFields` stores
        // DENSITIES, so each deposited extensive quantity must be divided by the
        // cell volume to convert back to a density before being added. (Omitting
        // this made the wave<->particle recombine non-conservative by a factor
        // of the cell volume — invisible only when dx=dy=1.)
        let inv_vol = 1.0 / (grid.dx * grid.dy);
        for i in 0..self.len() {
            let c = self.cell[i] as usize;
            if c >= fields.ncells() {
                continue;
            }
            let w = self.weight[i];
            let v = self.vel[i];
            let z = self.zeta[i];
            fields.rho[c] += w * inv_vol;
            fields.mom[0][c] += w * v[0] * inv_vol;
            fields.mom[1][c] += w * v[1] * inv_vol;
            fields.energy[c] += w * (0.5 * (v[0] * v[0] + v[1] * v[1]) + 0.5 * z * z) * inv_vol;
        }
    }

    /// Re-index particles into the cell they currently physically occupy
    /// (after free transport may have moved them across cell boundaries).
    /// Particles that leave the domain are handled by the caller-supplied
    /// `on_boundary` closure `(pos, vel, zeta, rng) -> keep`, which mutates
    /// `pos`/`vel`/`zeta` in place (e.g. periodic wrap, specular mirror,
    /// diffuse-wall re-emission — any of which may need a fresh random draw,
    /// hence the `rng` parameter) and returns `false` to remove the particle
    /// (e.g. absorbed at an inlet/outlet) — removal uses `swap_remove` so it
    /// is O(1) and allocation-free.
    pub fn relocate(
        &mut self,
        grid: &Grid2D,
        mut on_boundary: impl FnMut(&mut [f64; 2], &mut [f64; 2], &mut f64, &mut Rng) -> bool,
        rng: &mut Rng,
    ) {
        let mut i = 0;
        while i < self.len() {
            let mut p = self.pos[i];
            let mut v = self.vel[i];
            let mut z = self.zeta[i];
            let mut keep = true;
            // Clamp/wrap against domain extents via caller hook; if it
            // returns false, the particle is removed (left the domain with
            // no re-entry rule, e.g. an outlet).
            let ox = grid.origin[0];
            let oy = grid.origin[1];
            let lx = grid.nx as f64 * grid.dx;
            let ly = grid.ny as f64 * grid.dy;
            if p[0] < ox || p[0] >= ox + lx || p[1] < oy || p[1] >= oy + ly {
                keep = on_boundary(&mut p, &mut v, &mut z, rng);
            }
            if !keep {
                self.pos.swap_remove(i);
                self.vel.swap_remove(i);
                self.zeta.swap_remove(i);
                self.weight.swap_remove(i);
                self.cell.swap_remove(i);
                continue;
            }
            self.pos[i] = p;
            self.vel[i] = v;
            self.zeta[i] = z;
            let i_idx =
                (((p[0] - ox) / grid.dx).floor() as isize).clamp(0, grid.nx as isize - 1) as usize;
            let j_idx =
                (((p[1] - oy) / grid.dy).floor() as isize).clamp(0, grid.ny as isize - 1) as usize;
            self.cell[i] = grid.idx(i_idx, j_idx) as u32;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampled_moments_match_target_exactly() {
        let mut particles = Particles::with_capacity(2000);
        let mut rng = Rng::new(12345);
        let rho_vol = 2.5;
        let u = [10.0, -3.0];
        let t = 350.0;
        let r_gas = 208.13;
        particles.sample_cell(
            &mut rng,
            0,
            [0.5, 0.5],
            [0.5, 0.5],
            1000,
            rho_vol,
            u,
            t,
            r_gas,
        );

        let (mass, px, py, e) = particles.totals();
        assert!((mass - rho_vol).abs() < 1e-9, "mass {mass} vs {rho_vol}");
        assert!((px - rho_vol * u[0]).abs() < 1e-6, "px {px}");
        assert!((py - rho_vol * u[1]).abs() < 1e-6, "py {py}");
        let kinetic = 0.5 * rho_vol * (u[0] * u[0] + u[1] * u[1]);
        let internal = 0.5 * rho_vol * crate::maxwellian::DOF * r_gas * t;
        let expected_e = kinetic + internal;
        assert!(
            (e - expected_e).abs() / expected_e < 1e-6,
            "e {e} vs {expected_e}"
        );
    }

    #[test]
    fn sample_cell_with_dof_diatomic_matches_target_energy() {
        // Diatomic (zeta_int=2, dof_total=5) particle population: internal
        // energy target must scale with dof_total=5, not the monatomic
        // dof=3 default -- the particle-side half of the polyatomic-DOF
        // wiring (mirrors maxwellian::diatomic_internal_dof_gives_gamma_
        // seven_fifths_and_correct_energy's moment check, on the particle
        // representation instead of the (g,h) wave representation).
        let mut particles = Particles::with_capacity(2000);
        let mut rng = Rng::new(2024);
        let rho_vol = 2.5;
        let u = [10.0, -3.0];
        let t = 350.0;
        let r_gas = 296.8;
        let dof_total = crate::maxwellian::dof_with_internal(2.0);
        assert!((dof_total - 5.0).abs() < 1e-12);
        particles.sample_cell_with_dof(
            &mut rng,
            0,
            [0.5, 0.5],
            [0.5, 0.5],
            1000,
            rho_vol,
            u,
            t,
            r_gas,
            dof_total,
        );

        let (mass, px, py, e) = particles.totals();
        assert!((mass - rho_vol).abs() < 1e-9, "mass {mass} vs {rho_vol}");
        assert!((px - rho_vol * u[0]).abs() < 1e-6, "px {px}");
        assert!((py - rho_vol * u[1]).abs() < 1e-6, "py {py}");
        let kinetic = 0.5 * rho_vol * (u[0] * u[0] + u[1] * u[1]);
        let internal = 0.5 * rho_vol * dof_total * r_gas * t;
        let expected_e = kinetic + internal;
        assert!(
            (e - expected_e).abs() / expected_e < 1e-6,
            "e {e} vs {expected_e}"
        );
    }

    #[test]
    fn free_transport_moves_particles() {
        let mut p = Particles::with_capacity(4);
        p.pos.push([0.0, 0.0]);
        p.vel.push([1.0, 2.0]);
        p.zeta.push(0.0);
        p.weight.push(1.0);
        p.cell.push(0);
        p.free_transport(0.5);
        assert!((p.pos[0][0] - 0.5).abs() < 1e-12);
        assert!((p.pos[0][1] - 1.0).abs() < 1e-12);
    }
}
