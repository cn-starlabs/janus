//! Boundary condition kernels for the DVM distribution function.
//!
//! Each BC computes ghost-cell distribution values `f_ghost[k]` for every
//! discrete velocity `k`, given the interior (near-wall) cell's distribution
//! and the velocity grid. These ghost values are then used by the DUGKS face
//! reconstruction exactly like an interior neighbor cell.

use crate::maxwellian::{gh_equilibrium, maxwellian_2d};
use janus_core::config::BoundaryKind;

/// Outward unit normal convention: `normal` points OUT of the domain at the
/// boundary face (e.g. `[-1,0]` for a west wall, `[1,0]` for east, etc).
///
/// `apply` operates on the legacy single-distribution signature (kept for
/// the unit tests below that exercise the mass-flux-balance logic directly);
/// `apply_gh` is the (g,h)-aware entry point the DUGKS solver actually calls.
/// The default `apply_gh` derives an `h` ghost consistent with a Maxwellian
/// at the same ghost density/velocity/temperature that the `g` ghost was
/// built from wherever the concrete BC can express that (walls/inlets); for
/// BCs that just mirror/copy `g` (specular, outlet), `h` is mirrored/copied
/// identically, which is exact because those operations don't change the
/// local temperature.
pub trait BoundaryCondition {
    /// Populate `f_ghost` (length `vgrid.len()`) given the interior cell's
    /// distribution `f_interior` (same length), the outward normal, and the
    /// velocity grid.
    fn apply(
        &self,
        f_interior: &[f64],
        vgrid: &[[f64; 2]],
        vw: &[f64],
        normal: [f64; 2],
        r_gas: f64,
        f_ghost: &mut [f64],
    );

    /// (g,h)-aware ghost construction. Default implementation: build the `g`
    /// ghost via `apply` (which already encodes the physically correct wall
    /// Maxwellian / mirror / copy logic), then build a *matching* `h` ghost
    /// per BC kind by re-deriving it from the same ghost `g` values assuming
    /// they are locally Maxwellian (the standard assumption at diffuse
    /// walls/inlets: `h_ghost = (K/2)*R*T_ghost*g_ghost`) or by applying the
    /// identical mirror/copy operation used for `g` (specular/outlet/
    /// symmetry — those are velocity-remapping operations, not new
    /// equilibria, so copying the same remap to `h` is exact).
    fn apply_gh(
        &self,
        g_interior: &[f64],
        h_interior: &[f64],
        vgrid: &[[f64; 2]],
        vw: &[f64],
        normal: [f64; 2],
        r_gas: f64,
        g_ghost: &mut [f64],
        h_ghost: &mut [f64],
    ) {
        // Default: mirror/copy `h` exactly the way `g` was permuted. This is
        // correct for velocity-remapping BCs (specular/outlet/symmetry) and
        // is overridden by BCs that synthesize a genuinely new equilibrium
        // (diffuse wall, inlets) below.
        self.apply(g_interior, vgrid, vw, normal, r_gas, g_ghost);
        self.apply(h_interior, vgrid, vw, normal, r_gas, h_ghost);
    }
}

#[inline]
fn vdotn(v: [f64; 2], n: [f64; 2]) -> f64 {
    v[0] * n[0] + v[1] * n[1]
}

/// Fully diffuse wall (Maxwell full accommodation): outgoing (into-domain)
/// velocities are replaced by a wall Maxwellian at `wall_temperature` and
/// `wall_velocity`, scaled so the net mass flux through the wall is exactly
/// zero (no ghost mass leakage) — this is the standard diffuse-wall
/// construction (e.g. Sod/Couette DVM literature): incoming (into-domain)
/// distribution mirrors a Maxwellian of unknown density `rho_w`, solved from
/// the constraint that outgoing mass flux (from interior, leaving the
/// domain) equals incoming mass flux (from the wall Maxwellian, entering the
/// domain).
pub struct DiffuseWall {
    pub temperature: f64,
    pub wall_velocity: [f64; 2],
    pub accommodation: f64,
}

impl BoundaryCondition for DiffuseWall {
    fn apply(
        &self,
        f_interior: &[f64],
        vgrid: &[[f64; 2]],
        vw: &[f64],
        normal: [f64; 2],
        r_gas: f64,
        f_ghost: &mut [f64],
    ) {
        // Outward flux from interior cell (v.n > 0 means leaving the domain
        // across this face, i.e. moving toward/through the wall).
        let mut outflux = 0.0; // mass flux leaving domain (uses interior f)
        let mut influx_unit = 0.0; // mass flux that WOULD enter domain per unit rho_w
        for (k, v) in vgrid.iter().enumerate() {
            let vn = vdotn(*v, normal);
            if vn > 0.0 {
                outflux += vw[k] * f_interior[k] * vn;
            } else if vn < 0.0 {
                let m_unit = maxwellian_2d(1.0, self.wall_velocity, self.temperature, r_gas, *v);
                influx_unit += vw[k] * m_unit * vn; // negative
            }
        }
        // outflux (positive, mass/time leaving) must equal magnitude of
        // incoming flux: outflux = -rho_w * influx_unit  =>  rho_w = outflux / (-influx_unit)
        let rho_w = if influx_unit.abs() > 1e-300 {
            outflux / (-influx_unit)
        } else {
            0.0
        };

        for (k, v) in vgrid.iter().enumerate() {
            let vn = vdotn(*v, normal);
            f_ghost[k] = if vn < 0.0 {
                let diff = maxwellian_2d(rho_w, self.wall_velocity, self.temperature, r_gas, *v);
                let vr = [v[0] - 2.0 * vn * normal[0], v[1] - 2.0 * vn * normal[1]];
                let idx = nearest_velocity_index(vgrid, vr);
                let spec = f_interior[idx];
                self.accommodation * diff + (1.0 - self.accommodation) * spec
            } else {
                // Outgoing directions: ghost mirrors interior (won't be used
                // by upwind reconstruction, but kept consistent/nonzero).
                f_interior[k]
            };
        }
    }

    fn apply_gh(
        &self,
        g_interior: &[f64],
        h_interior: &[f64],
        vgrid: &[[f64; 2]],
        vw: &[f64],
        normal: [f64; 2],
        r_gas: f64,
        g_ghost: &mut [f64],
        h_ghost: &mut [f64],
    ) {
        self.apply(g_interior, vgrid, vw, normal, r_gas, g_ghost);

        let mut outflux = 0.0;
        let mut influx_unit = 0.0;
        for (k, v) in vgrid.iter().enumerate() {
            let vn = vdotn(*v, normal);
            if vn > 0.0 {
                outflux += vw[k] * g_interior[k] * vn;
            } else if vn < 0.0 {
                let m_unit = maxwellian_2d(1.0, self.wall_velocity, self.temperature, r_gas, *v);
                influx_unit += vw[k] * m_unit * vn;
            }
        }
        let rho_w = if influx_unit.abs() > 1e-300 {
            outflux / (-influx_unit)
        } else {
            0.0
        };

        for (k, v) in vgrid.iter().enumerate() {
            let vn = vdotn(*v, normal);
            if vn < 0.0 {
                let (_, h_eq) =
                    gh_equilibrium(1.0, self.wall_velocity, self.temperature, r_gas, *v);
                let diff_h = rho_w * h_eq;
                let vr = [v[0] - 2.0 * vn * normal[0], v[1] - 2.0 * vn * normal[1]];
                let idx = nearest_velocity_index(vgrid, vr);
                let spec_h = h_interior[idx];
                h_ghost[k] = self.accommodation * diff_h + (1.0 - self.accommodation) * spec_h;
            } else {
                h_ghost[k] = h_interior[k];
            }
        }
    }
}

/// Specular reflection wall: mirrors the velocity component along `normal`.
pub struct SpecularWall;

impl BoundaryCondition for SpecularWall {
    fn apply(
        &self,
        f_interior: &[f64],
        vgrid: &[[f64; 2]],
        vw: &[f64],
        normal: [f64; 2],
        _r_gas: f64,
        f_ghost: &mut [f64],
    ) {
        let _ = vw;
        // For each velocity k with v.n < 0 (incoming), find the mirrored
        // velocity k' = v - 2*(v.n)*n and copy f_interior at that mirrored
        // velocity's own incoming... In practice: f_ghost(v) for incoming v
        // = f_interior(v_reflected) where v_reflected is the outgoing
        // velocity obtained by mirroring v about the wall tangent.
        for (k, v) in vgrid.iter().enumerate() {
            let vn = vdotn(*v, normal);
            if vn < 0.0 {
                let vr = [v[0] - 2.0 * vn * normal[0], v[1] - 2.0 * vn * normal[1]];
                // find nearest velocity node to vr (grid is symmetric under
                // axis-aligned reflection for our tensor-product Simpson
                // grid, so this is exact when normal is axis-aligned).
                let idx = nearest_velocity_index(vgrid, vr);
                f_ghost[k] = f_interior[idx];
            } else {
                f_ghost[k] = f_interior[k];
            }
        }
    }
}

fn nearest_velocity_index(vgrid: &[[f64; 2]], target: [f64; 2]) -> usize {
    let mut best = 0;
    let mut best_d2 = f64::MAX;
    for (i, v) in vgrid.iter().enumerate() {
        let dx = v[0] - target[0];
        let dy = v[1] - target[1];
        let d2 = dx * dx + dy * dy;
        if d2 < best_d2 {
            best_d2 = d2;
            best = i;
        }
    }
    best
}

/// Prescribed velocity/density/temperature inlet: ghost is the equilibrium
/// Maxwellian at the prescribed state for ALL velocity nodes (simple,
/// standard supersonic/subsonic inlet treatment for a DVM solver; a more
/// elaborate characteristic-based inlet is out of scope for M1).
pub struct VelocityInlet {
    pub velocity: [f64; 2],
    pub density: f64,
    pub temperature: f64,
}

impl BoundaryCondition for VelocityInlet {
    fn apply(
        &self,
        _f_interior: &[f64],
        vgrid: &[[f64; 2]],
        _vw: &[f64],
        _normal: [f64; 2],
        r_gas: f64,
        f_ghost: &mut [f64],
    ) {
        for (k, v) in vgrid.iter().enumerate() {
            f_ghost[k] = maxwellian_2d(self.density, self.velocity, self.temperature, r_gas, *v);
        }
    }

    fn apply_gh(
        &self,
        _g_interior: &[f64],
        _h_interior: &[f64],
        vgrid: &[[f64; 2]],
        _vw: &[f64],
        _normal: [f64; 2],
        r_gas: f64,
        g_ghost: &mut [f64],
        h_ghost: &mut [f64],
    ) {
        for (k, v) in vgrid.iter().enumerate() {
            let (g, h) = gh_equilibrium(self.density, self.velocity, self.temperature, r_gas, *v);
            g_ghost[k] = g;
            h_ghost[k] = h;
        }
    }
}

/// Prescribed static-pressure inlet/outlet: density derived from `p = rho R T`
/// at the prescribed temperature, zero prescribed velocity (stagnation-like).
pub struct PressureInlet {
    pub pressure: f64,
    pub temperature: f64,
}

impl BoundaryCondition for PressureInlet {
    fn apply(
        &self,
        _f_interior: &[f64],
        vgrid: &[[f64; 2]],
        _vw: &[f64],
        _normal: [f64; 2],
        r_gas: f64,
        f_ghost: &mut [f64],
    ) {
        let rho = self.pressure / (r_gas * self.temperature);
        for (k, v) in vgrid.iter().enumerate() {
            f_ghost[k] = maxwellian_2d(rho, [0.0, 0.0], self.temperature, r_gas, *v);
        }
    }

    fn apply_gh(
        &self,
        _g_interior: &[f64],
        _h_interior: &[f64],
        vgrid: &[[f64; 2]],
        _vw: &[f64],
        _normal: [f64; 2],
        r_gas: f64,
        g_ghost: &mut [f64],
        h_ghost: &mut [f64],
    ) {
        let rho = self.pressure / (r_gas * self.temperature);
        for (k, v) in vgrid.iter().enumerate() {
            let (g, h) = gh_equilibrium(rho, [0.0, 0.0], self.temperature, r_gas, *v);
            g_ghost[k] = g;
            h_ghost[k] = h;
        }
    }
}

/// Zeroth-order extrapolation (Neumann) outlet: ghost = interior.
pub struct Outlet;

impl BoundaryCondition for Outlet {
    fn apply(
        &self,
        f_interior: &[f64],
        _vgrid: &[[f64; 2]],
        _vw: &[f64],
        _normal: [f64; 2],
        _r_gas: f64,
        f_ghost: &mut [f64],
    ) {
        f_ghost.copy_from_slice(f_interior);
    }
}

/// Symmetry plane: mirrors the normal velocity component (identical
/// construction to specular wall for this DVM formulation).
pub struct Symmetry;

impl BoundaryCondition for Symmetry {
    fn apply(
        &self,
        f_interior: &[f64],
        vgrid: &[[f64; 2]],
        vw: &[f64],
        normal: [f64; 2],
        r_gas: f64,
        f_ghost: &mut [f64],
    ) {
        SpecularWall.apply(f_interior, vgrid, vw, normal, r_gas, f_ghost);
    }
}

/// Enum-dispatch boundary-condition kernel: one variant per `BoundaryKind`,
/// each carrying exactly the parameters that BC needs. `apply`/`apply_gh`
/// match on the variant and call the corresponding concrete kernel directly
/// — this is monomorphized dispatch (a jump table compiled from the match,
/// resolved at the call site, no vtable indirection and no heap allocation
/// per boundary-face evaluation), replacing the previous
/// `Box<dyn BoundaryCondition>` design. This is the "prefer the type system
/// over dynamic dispatch in hot/structural paths" principle from
/// ENGINEERING_SPEC.md §10b applied to the BC layer, which sits in the
/// per-face-per-step hot loop (`DugksSolver::compute_boundary_face_flux`).
#[derive(Clone, Copy, Debug)]
pub enum BoundaryConditionKernel {
    DiffuseWall {
        temperature: f64,
        wall_velocity: [f64; 2],
        accommodation: f64,
    },
    SpecularWall,
    VelocityInlet {
        velocity: [f64; 2],
        density: f64,
        temperature: f64,
    },
    PressureInlet {
        pressure: f64,
        temperature: f64,
    },
    Outlet,
    Symmetry,
    /// Periodic is handled structurally by the solver (wraps the cell
    /// index), never via a ghost-value BC evaluation; this variant exists
    /// only so `from_kind` is total and is never actually invoked on the
    /// periodic code path (the solver branches on `BoundaryKindResolved::
    /// Periodic` before ever constructing/calling a BC kernel).
    Periodic,
}

impl BoundaryConditionKernel {
    /// Build the kernel for a case-config `BoundaryKind`. `Copy`, so this is
    /// free to call per-face (no allocation) unlike the old `Box::new` path.
    #[inline]
    pub fn from_kind(kind: &BoundaryKind) -> Self {
        match *kind {
            BoundaryKind::DiffuseWall {
                temperature,
                wall_velocity,
                accommodation,
            } => Self::DiffuseWall {
                temperature,
                wall_velocity,
                accommodation,
            },
            BoundaryKind::SpecularWall => Self::SpecularWall,
            BoundaryKind::VelocityInlet {
                velocity,
                density,
                temperature,
            } => Self::VelocityInlet {
                velocity,
                density,
                temperature,
            },
            BoundaryKind::PressureInlet {
                pressure,
                temperature,
            } => Self::PressureInlet {
                pressure,
                temperature,
            },
            BoundaryKind::Outlet => Self::Outlet,
            BoundaryKind::Symmetry => Self::Symmetry,
            BoundaryKind::Periodic => Self::Periodic,
        }
    }

    #[inline]
    pub fn apply(
        &self,
        f_interior: &[f64],
        vgrid: &[[f64; 2]],
        vw: &[f64],
        normal: [f64; 2],
        r_gas: f64,
        f_ghost: &mut [f64],
    ) {
        match *self {
            Self::DiffuseWall {
                temperature,
                wall_velocity,
                accommodation,
            } => DiffuseWall {
                temperature,
                wall_velocity,
                accommodation,
            }
            .apply(f_interior, vgrid, vw, normal, r_gas, f_ghost),
            Self::SpecularWall => SpecularWall.apply(f_interior, vgrid, vw, normal, r_gas, f_ghost),
            Self::VelocityInlet {
                velocity,
                density,
                temperature,
            } => VelocityInlet {
                velocity,
                density,
                temperature,
            }
            .apply(f_interior, vgrid, vw, normal, r_gas, f_ghost),
            Self::PressureInlet {
                pressure,
                temperature,
            } => PressureInlet {
                pressure,
                temperature,
            }
            .apply(f_interior, vgrid, vw, normal, r_gas, f_ghost),
            Self::Outlet => Outlet.apply(f_interior, vgrid, vw, normal, r_gas, f_ghost),
            Self::Symmetry => Symmetry.apply(f_interior, vgrid, vw, normal, r_gas, f_ghost),
            Self::Periodic => Outlet.apply(f_interior, vgrid, vw, normal, r_gas, f_ghost),
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn apply_gh(
        &self,
        g_interior: &[f64],
        h_interior: &[f64],
        vgrid: &[[f64; 2]],
        vw: &[f64],
        normal: [f64; 2],
        r_gas: f64,
        g_ghost: &mut [f64],
        h_ghost: &mut [f64],
    ) {
        match *self {
            Self::DiffuseWall {
                temperature,
                wall_velocity,
                accommodation,
            } => DiffuseWall {
                temperature,
                wall_velocity,
                accommodation,
            }
            .apply_gh(
                g_interior, h_interior, vgrid, vw, normal, r_gas, g_ghost, h_ghost,
            ),
            Self::SpecularWall => SpecularWall.apply_gh(
                g_interior, h_interior, vgrid, vw, normal, r_gas, g_ghost, h_ghost,
            ),
            Self::VelocityInlet {
                velocity,
                density,
                temperature,
            } => VelocityInlet {
                velocity,
                density,
                temperature,
            }
            .apply_gh(
                g_interior, h_interior, vgrid, vw, normal, r_gas, g_ghost, h_ghost,
            ),
            Self::PressureInlet {
                pressure,
                temperature,
            } => PressureInlet {
                pressure,
                temperature,
            }
            .apply_gh(
                g_interior, h_interior, vgrid, vw, normal, r_gas, g_ghost, h_ghost,
            ),
            Self::Outlet => Outlet.apply_gh(
                g_interior, h_interior, vgrid, vw, normal, r_gas, g_ghost, h_ghost,
            ),
            Self::Symmetry => Symmetry.apply_gh(
                g_interior, h_interior, vgrid, vw, normal, r_gas, g_ghost, h_ghost,
            ),
            Self::Periodic => Outlet.apply_gh(
                g_interior, h_interior, vgrid, vw, normal, r_gas, g_ghost, h_ghost,
            ),
        }
    }
}

/// Build a `Box<dyn BoundaryCondition>` from a case-config `BoundaryKind`.
///
/// DESIGN: retained only for any external/test code still depending on the
/// trait-object form; the solver's hot path (`solver.rs`) now uses
/// `BoundaryConditionKernel::from_kind` (enum dispatch, no allocation, no
/// vtable) instead. This function is a thin, non-hot-path convenience
/// wrapper — not a place where dynamic dispatch cost matters — so keeping it
/// alongside the enum is a reasonable minimal-surface compromise rather than
/// a silent reintroduction of `Box<dyn>` in a hot loop.
pub fn from_kind(kind: &BoundaryKind) -> Box<dyn BoundaryCondition + Send + Sync> {
    match *kind {
        BoundaryKind::DiffuseWall {
            temperature,
            wall_velocity,
            accommodation,
        } => Box::new(DiffuseWall {
            temperature,
            wall_velocity,
            accommodation,
        }),
        BoundaryKind::SpecularWall => Box::new(SpecularWall),
        BoundaryKind::VelocityInlet {
            velocity,
            density,
            temperature,
        } => Box::new(VelocityInlet {
            velocity,
            density,
            temperature,
        }),
        BoundaryKind::PressureInlet {
            pressure,
            temperature,
        } => Box::new(PressureInlet {
            pressure,
            temperature,
        }),
        BoundaryKind::Outlet => Box::new(Outlet),
        BoundaryKind::Symmetry => Box::new(Symmetry),
        BoundaryKind::Periodic => {
            // Periodic is handled structurally by the solver (wraps index),
            // not via a ghost-value BC object; return Outlet as an inert
            // placeholder that is never invoked in the periodic code path.
            Box::new(Outlet)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::velocity_grid::VelocityGrid2D;

    #[test]
    fn diffuse_wall_zero_net_mass_flux() {
        let (vgrid, vw) = VelocityGrid2D::simpson(2000.0, 81);
        let r_gas = 287.0;
        // interior distribution: Maxwellian with some bulk velocity toward
        // the wall (normal = [1,0], wall to the "east" of the interior cell).
        let rho = 1.0;
        let u = [50.0, 0.0];
        let t = 300.0;
        let mut f_interior = vec![0.0; vgrid.len()];
        for (k, v) in vgrid.iter().enumerate() {
            f_interior[k] = maxwellian_2d(rho, u, t, r_gas, *v);
        }
        let wall = DiffuseWall {
            temperature: t,
            wall_velocity: [0.0, 0.0],
            accommodation: 1.0,
        };
        let normal = [1.0, 0.0];
        let mut f_ghost = vec![0.0; vgrid.len()];
        wall.apply(&f_interior, &vgrid, &vw, normal, r_gas, &mut f_ghost);

        // Net mass flux across the face using upwinded values: outgoing uses
        // interior, incoming uses ghost. Should sum to ~0.
        let mut net = 0.0;
        for (k, v) in vgrid.iter().enumerate() {
            let vn = v[0] * normal[0] + v[1] * normal[1];
            if vn > 0.0 {
                net += vw[k] * f_interior[k] * vn;
            } else if vn < 0.0 {
                net += vw[k] * f_ghost[k] * vn;
            }
        }
        assert!(
            net.abs() < 1e-6,
            "net mass flux at diffuse wall should vanish, got {net}"
        );
    }

    #[test]
    fn specular_wall_reflects_normal_velocity_moment() {
        let (vgrid, vw) = VelocityGrid2D::simpson(2000.0, 81);
        let r_gas = 287.0;
        let rho = 1.0;
        let u = [50.0, 10.0];
        let t = 300.0;
        let mut f_interior = vec![0.0; vgrid.len()];
        for (k, v) in vgrid.iter().enumerate() {
            f_interior[k] = maxwellian_2d(rho, u, t, r_gas, *v);
        }
        let normal = [1.0, 0.0];
        let mut f_ghost = vec![0.0; vgrid.len()];
        SpecularWall.apply(&f_interior, &vgrid, &vw, normal, r_gas, &mut f_ghost);

        // Net normal mass flux at the wall (upwinded) should be ~0: a
        // specular wall is impermeable.
        let mut net = 0.0;
        for (k, v) in vgrid.iter().enumerate() {
            let vn = v[0] * normal[0] + v[1] * normal[1];
            if vn > 0.0 {
                net += vw[k] * f_interior[k] * vn;
            } else if vn < 0.0 {
                net += vw[k] * f_ghost[k] * vn;
            }
        }
        assert!(
            net.abs() < 1e-2 * rho * u[0].abs().max(1.0),
            "net flux {net}"
        );
    }
}
