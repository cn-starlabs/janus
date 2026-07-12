//! Boundary condition kernels for the 3D single-distribution DVM.
//!
//! Mirrors `bc.rs`'s structure and physics exactly (diffuse wall mass-flux
//! balance, specular mirror-reflection, prescribed inlets, extrapolated
//! outlet, symmetry-as-specular, periodic-handled-structurally), generalized
//! to 3-component velocities/normals and operating on a single distribution
//! `f` (no `(g,h)` pair — see `maxwellian3d` module docs for why that
//! reduction is dropped in 3D).

use crate::maxwellian3d::maxwellian_3d;
use janus_core::config::BoundaryKind3D;

#[inline]
fn vdotn(v: [f64; 3], n: [f64; 3]) -> f64 {
    v[0] * n[0] + v[1] * n[1] + v[2] * n[2]
}

/// Single-distribution boundary condition trait, 3D face geometry.
pub trait BoundaryCondition3D {
    /// Populate `f_ghost` (length `vgrid.len()`) given the interior cell's
    /// distribution `f_interior`, the outward face normal, and the velocity
    /// grid.
    fn apply(
        &self,
        f_interior: &[f64],
        vgrid: &[[f64; 3]],
        vw: &[f64],
        normal: [f64; 3],
        r_gas: f64,
        f_ghost: &mut [f64],
    );
}

/// Fully diffuse wall (Maxwell full accommodation), 3D: identical
/// mass-flux-balance construction to the 2D `bc::DiffuseWall`, generalized
/// to a 3-component velocity/normal.
pub struct DiffuseWall3D {
    pub temperature: f64,
    pub wall_velocity: [f64; 3],
}

impl BoundaryCondition3D for DiffuseWall3D {
    fn apply(
        &self,
        f_interior: &[f64],
        vgrid: &[[f64; 3]],
        vw: &[f64],
        normal: [f64; 3],
        r_gas: f64,
        f_ghost: &mut [f64],
    ) {
        let mut outflux = 0.0;
        let mut influx_unit = 0.0;
        for (k, v) in vgrid.iter().enumerate() {
            let vn = vdotn(*v, normal);
            if vn > 0.0 {
                outflux += vw[k] * f_interior[k] * vn;
            } else if vn < 0.0 {
                let m_unit = maxwellian_3d(1.0, self.wall_velocity, self.temperature, r_gas, *v);
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
            f_ghost[k] = if vn < 0.0 {
                maxwellian_3d(rho_w, self.wall_velocity, self.temperature, r_gas, *v)
            } else {
                f_interior[k]
            };
        }
    }
}

/// Specular reflection wall: mirrors the velocity component along `normal`.
pub struct SpecularWall3D;

impl BoundaryCondition3D for SpecularWall3D {
    fn apply(
        &self,
        f_interior: &[f64],
        vgrid: &[[f64; 3]],
        vw: &[f64],
        normal: [f64; 3],
        _r_gas: f64,
        f_ghost: &mut [f64],
    ) {
        let _ = vw;
        for (k, v) in vgrid.iter().enumerate() {
            let vn = vdotn(*v, normal);
            if vn < 0.0 {
                let vr = [
                    v[0] - 2.0 * vn * normal[0],
                    v[1] - 2.0 * vn * normal[1],
                    v[2] - 2.0 * vn * normal[2],
                ];
                let idx = nearest_velocity_index(vgrid, vr);
                f_ghost[k] = f_interior[idx];
            } else {
                f_ghost[k] = f_interior[k];
            }
        }
    }
}

fn nearest_velocity_index(vgrid: &[[f64; 3]], target: [f64; 3]) -> usize {
    let mut best = 0;
    let mut best_d2 = f64::MAX;
    for (i, v) in vgrid.iter().enumerate() {
        let dx = v[0] - target[0];
        let dy = v[1] - target[1];
        let dz = v[2] - target[2];
        let d2 = dx * dx + dy * dy + dz * dz;
        if d2 < best_d2 {
            best_d2 = d2;
            best = i;
        }
    }
    best
}

/// Prescribed velocity/density/temperature inlet: ghost is the equilibrium
/// Maxwellian at the prescribed state for all velocity nodes.
pub struct VelocityInlet3D {
    pub velocity: [f64; 3],
    pub density: f64,
    pub temperature: f64,
}

impl BoundaryCondition3D for VelocityInlet3D {
    fn apply(
        &self,
        _f_interior: &[f64],
        vgrid: &[[f64; 3]],
        _vw: &[f64],
        _normal: [f64; 3],
        r_gas: f64,
        f_ghost: &mut [f64],
    ) {
        for (k, v) in vgrid.iter().enumerate() {
            f_ghost[k] = maxwellian_3d(self.density, self.velocity, self.temperature, r_gas, *v);
        }
    }
}

/// Prescribed static-pressure inlet/outlet: density from `p = rho R T`,
/// zero prescribed velocity.
pub struct PressureInlet3D {
    pub pressure: f64,
    pub temperature: f64,
}

impl BoundaryCondition3D for PressureInlet3D {
    fn apply(
        &self,
        _f_interior: &[f64],
        vgrid: &[[f64; 3]],
        _vw: &[f64],
        _normal: [f64; 3],
        r_gas: f64,
        f_ghost: &mut [f64],
    ) {
        let rho = self.pressure / (r_gas * self.temperature);
        for (k, v) in vgrid.iter().enumerate() {
            f_ghost[k] = maxwellian_3d(rho, [0.0, 0.0, 0.0], self.temperature, r_gas, *v);
        }
    }
}

/// Zeroth-order extrapolation (Neumann) outlet: ghost = interior.
pub struct Outlet3D;

impl BoundaryCondition3D for Outlet3D {
    fn apply(
        &self,
        f_interior: &[f64],
        _vgrid: &[[f64; 3]],
        _vw: &[f64],
        _normal: [f64; 3],
        _r_gas: f64,
        f_ghost: &mut [f64],
    ) {
        f_ghost.copy_from_slice(f_interior);
    }
}

/// Symmetry plane: identical construction to specular wall.
pub struct Symmetry3D;

impl BoundaryCondition3D for Symmetry3D {
    fn apply(
        &self,
        f_interior: &[f64],
        vgrid: &[[f64; 3]],
        vw: &[f64],
        normal: [f64; 3],
        r_gas: f64,
        f_ghost: &mut [f64],
    ) {
        SpecularWall3D.apply(f_interior, vgrid, vw, normal, r_gas, f_ghost);
    }
}

/// Enum-dispatch BC kernel (no heap allocation, no vtable — same discipline
/// as `bc::BoundaryConditionKernel`), one variant per `BoundaryKind3D`.
#[derive(Clone, Copy, Debug)]
pub enum BoundaryConditionKernel3D {
    DiffuseWall {
        temperature: f64,
        wall_velocity: [f64; 3],
    },
    SpecularWall,
    VelocityInlet {
        velocity: [f64; 3],
        density: f64,
        temperature: f64,
    },
    PressureInlet {
        pressure: f64,
        temperature: f64,
    },
    Outlet,
    Symmetry,
    /// Never actually invoked (handled structurally by the solver, wraps
    /// the index) — see `bc::BoundaryConditionKernel::Periodic` docs.
    Periodic,
}

impl BoundaryConditionKernel3D {
    #[inline]
    pub fn from_kind(kind: &BoundaryKind3D) -> Self {
        match *kind {
            BoundaryKind3D::DiffuseWall {
                temperature,
                wall_velocity,
            } => Self::DiffuseWall {
                temperature,
                wall_velocity,
            },
            BoundaryKind3D::SpecularWall => Self::SpecularWall,
            BoundaryKind3D::VelocityInlet {
                velocity,
                density,
                temperature,
            } => Self::VelocityInlet {
                velocity,
                density,
                temperature,
            },
            BoundaryKind3D::PressureInlet {
                pressure,
                temperature,
            } => Self::PressureInlet {
                pressure,
                temperature,
            },
            BoundaryKind3D::Outlet => Self::Outlet,
            BoundaryKind3D::Symmetry => Self::Symmetry,
            BoundaryKind3D::Periodic => Self::Periodic,
        }
    }

    #[inline]
    pub fn apply(
        &self,
        f_interior: &[f64],
        vgrid: &[[f64; 3]],
        vw: &[f64],
        normal: [f64; 3],
        r_gas: f64,
        f_ghost: &mut [f64],
    ) {
        match *self {
            Self::DiffuseWall {
                temperature,
                wall_velocity,
            } => DiffuseWall3D {
                temperature,
                wall_velocity,
            }
            .apply(f_interior, vgrid, vw, normal, r_gas, f_ghost),
            Self::SpecularWall => {
                SpecularWall3D.apply(f_interior, vgrid, vw, normal, r_gas, f_ghost)
            }
            Self::VelocityInlet {
                velocity,
                density,
                temperature,
            } => VelocityInlet3D {
                velocity,
                density,
                temperature,
            }
            .apply(f_interior, vgrid, vw, normal, r_gas, f_ghost),
            Self::PressureInlet {
                pressure,
                temperature,
            } => PressureInlet3D {
                pressure,
                temperature,
            }
            .apply(f_interior, vgrid, vw, normal, r_gas, f_ghost),
            Self::Outlet => Outlet3D.apply(f_interior, vgrid, vw, normal, r_gas, f_ghost),
            Self::Symmetry => Symmetry3D.apply(f_interior, vgrid, vw, normal, r_gas, f_ghost),
            Self::Periodic => Outlet3D.apply(f_interior, vgrid, vw, normal, r_gas, f_ghost),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::velocity_grid3d::VelocityGrid3D;

    #[test]
    fn diffuse_wall_zero_net_mass_flux_3d() {
        let (vgrid, vw) = VelocityGrid3D::gauss_hermite(287.0, 300.0, [50.0, 0.0, 0.0], 8);
        let r_gas = 287.0;
        let rho = 1.0;
        let u = [50.0, 0.0, 0.0];
        let t = 300.0;
        let mut f_interior = vec![0.0; vgrid.len()];
        for (k, v) in vgrid.iter().enumerate() {
            f_interior[k] = maxwellian_3d(rho, u, t, r_gas, *v);
        }
        let wall = DiffuseWall3D {
            temperature: t,
            wall_velocity: [0.0, 0.0, 0.0],
        };
        let normal = [1.0, 0.0, 0.0];
        let mut f_ghost = vec![0.0; vgrid.len()];
        wall.apply(&f_interior, &vgrid, &vw, normal, r_gas, &mut f_ghost);

        let mut net = 0.0;
        for (k, v) in vgrid.iter().enumerate() {
            let vn = vdotn(*v, normal);
            if vn > 0.0 {
                net += vw[k] * f_interior[k] * vn;
            } else if vn < 0.0 {
                net += vw[k] * f_ghost[k] * vn;
            }
        }
        assert!(
            net.abs() < 1e-3,
            "net mass flux at 3D diffuse wall should vanish, got {net}"
        );
    }

    #[test]
    fn specular_wall_3d_impermeable() {
        // Specular reflection maps v -> v - 2(v.n)n, i.e. it negates the normal
        // velocity component about vn=0. On the discrete grid this is an EXACT
        // node bijection (and the wall is exactly impermeable) only if the
        // velocity grid is symmetric about 0 — which is how the solver actually
        // builds its (fixed, global) velocity grid for wall-bounded problems
        // (see the [0,0,0]-centered `gauss_hermite` calls throughout solver3d/
        // coupled3d). A grid centered on a nonzero bulk velocity would place the
        // mirror point off-grid, forcing an inexact nearest-node snap; that is
        // not a supported configuration for reflective walls. The Maxwellian
        // being reflected still carries a nonzero bulk velocity u.
        let (vgrid, vw) = VelocityGrid3D::gauss_hermite(287.0, 300.0, [0.0, 0.0, 0.0], 8);
        let r_gas = 287.0;
        let rho = 1.0;
        let u = [50.0, 10.0, 5.0];
        let t = 300.0;
        let mut f_interior = vec![0.0; vgrid.len()];
        for (k, v) in vgrid.iter().enumerate() {
            f_interior[k] = maxwellian_3d(rho, u, t, r_gas, *v);
        }
        let normal = [1.0, 0.0, 0.0];
        let mut f_ghost = vec![0.0; vgrid.len()];
        SpecularWall3D.apply(&f_interior, &vgrid, &vw, normal, r_gas, &mut f_ghost);

        let mut net = 0.0;
        for (k, v) in vgrid.iter().enumerate() {
            let vn = vdotn(*v, normal);
            if vn > 0.0 {
                net += vw[k] * f_interior[k] * vn;
            } else if vn < 0.0 {
                net += vw[k] * f_ghost[k] * vn;
            }
        }
        assert!(
            net.abs() < 1e-1 * rho * u[0].abs().max(1.0),
            "net flux {net}"
        );
    }
}
