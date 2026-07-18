//! Case configuration: grid, boundary condition assignment, gas properties.
//!
//! M1 uses a hardcoded in-code `CaseConfig` (no file-based config parsing —
//! that is out of scope per the milestone instructions); janus-cli builds
//! one directly in Rust.

use crate::grid::Grid2D;
use serde::{Deserialize, Serialize};

fn default_accommodation() -> f64 {
    1.0
}

fn default_gas_model_kind() -> String {
    "ideal_vhs".to_string()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollisionModelKind {
    #[default]
    #[serde(rename = "shakhov")]
    Shakhov,
    #[serde(rename = "fast_spectral")]
    FastSpectral,
}

/// Which physical boundary condition family to apply at a domain edge.
///
/// The concrete numerical implementation of each kind lives in
/// `janus-kinetic::bc` (kept out of janus-core to avoid the kinetic crate's
/// physics leaking into the shared data-structure crate); this enum is just
/// the case-setup-time *choice* of BC type plus its parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BoundaryKind {
    /// Fully diffuse (Maxwell full accommodation) wall at a given
    /// temperature and tangential wall velocity (e.g. Couette plates).
    DiffuseWall {
        temperature: f64,
        wall_velocity: [f64; 2],
        #[serde(default = "default_accommodation")]
        accommodation: f64,
    },
    /// Specular reflection wall (mirrors the normal velocity component).
    SpecularWall,
    /// Prescribed velocity + density/temperature inlet.
    VelocityInlet {
        velocity: [f64; 2],
        density: f64,
        temperature: f64,
    },
    /// Prescribed static pressure outlet/inlet.
    PressureInlet { pressure: f64, temperature: f64 },
    /// Zeroth-order extrapolation (Neumann) outlet.
    Outlet,
    /// Mirror/symmetry plane (zero normal flux, zero normal velocity).
    Symmetry,
    /// Periodic wrap to the opposite boundary.
    Periodic,
}

/// The four edges of a `Grid2D` domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Edge {
    West,
    East,
    South,
    North,
}

/// Per-edge boundary condition assignment for a rectangular domain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundaryAssignment {
    pub west: BoundaryKind,
    pub east: BoundaryKind,
    pub south: BoundaryKind,
    pub north: BoundaryKind,
}

impl BoundaryAssignment {
    pub fn get(&self, edge: Edge) -> &BoundaryKind {
        match edge {
            Edge::West => &self.west,
            Edge::East => &self.east,
            Edge::South => &self.south,
            Edge::North => &self.north,
        }
    }

    /// All-periodic assignment convenience constructor (fully periodic box).
    pub fn all_periodic() -> Self {
        Self {
            west: BoundaryKind::Periodic,
            east: BoundaryKind::Periodic,
            south: BoundaryKind::Periodic,
            north: BoundaryKind::Periodic,
        }
    }
}

/// Monatomic ideal-gas properties + VHS viscosity law parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GasProperties {
    /// Specific gas constant R = R_universal / molar_mass, J/(kg*K).
    pub r_gas: f64,
    /// Molar mass, kg/mol (kept for reference / derived quantities).
    pub molar_mass: f64,
    /// VHS viscosity exponent (omega).
    pub vhs_omega: f64,
    /// Reference viscosity, Pa*s, at `t_ref`.
    pub mu_ref: f64,
    /// Reference temperature for the VHS law, K.
    pub t_ref: f64,
    /// Prandtl number used by the Shakhov model (monatomic gas: Pr = 2/3).
    pub prandtl: f64,
    /// Gas model kind: "ideal_vhs" (default) or "virial".
    #[serde(default = "default_gas_model_kind")]
    pub gas_model_kind: String,
    /// Collision model kind for the solver hot path. Defaults to Shakhov.
    #[serde(default)]
    pub collision_model_kind: CollisionModelKind,
    /// Second virial coefficient coefficients (b0, b1, b2).
    #[serde(default)]
    pub b0: f64,
    #[serde(default)]
    pub b1: f64,
    #[serde(default)]
    pub b2: f64,
    /// Third virial coefficient coefficients (c0, c1, c2).
    #[serde(default)]
    pub c0: f64,
    #[serde(default)]
    pub c1: f64,
    #[serde(default)]
    pub c2: f64,
    /// Internal degrees of freedom (0.0 for monatomic, 2.0 for diatomic rigid rotor).
    #[serde(default)]
    pub internal_dof: f64,
}

impl GasProperties {
    /// Standard monatomic-argon-like defaults, commonly used in DVM/DSMC
    /// benchmark papers (Pr = 2/3 exact for monatomic gas kinetic theory).
    pub fn monatomic_default() -> Self {
        Self {
            r_gas: 208.13, // argon
            molar_mass: 0.039_948,
            vhs_omega: 0.81,
            mu_ref: 2.117e-5,
            t_ref: 273.15,
            prandtl: 2.0 / 3.0,
            gas_model_kind: default_gas_model_kind(),
            collision_model_kind: CollisionModelKind::Shakhov,
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            internal_dof: 0.0,
        }
    }
}

/// Full case setup: grid + boundary assignment + gas properties.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaseConfig {
    pub grid: Grid2D,
    pub bcs: BoundaryAssignment,
    pub gas: GasProperties,
}

/// 3D-face analog of `BoundaryKind`: identical physics choices (diffuse
/// wall, specular, velocity/pressure inlet, outlet, symmetry, periodic),
/// generalized to 3-component velocities (`[f64; 3]`) for a `Grid3D` face
/// normal. Kept as a separate type from `BoundaryKind` for the same reason
/// `Grid3D`/`MacroFields3D` are separate from their 2D counterparts: the 2D
/// enum's velocity fields are hardcoded `[f64; 2]` and every 2D call site
/// (bc.rs, solver.rs, coupled.rs) matches on that shape directly.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BoundaryKind3D {
    DiffuseWall {
        temperature: f64,
        wall_velocity: [f64; 3],
        #[serde(default = "default_accommodation")]
        accommodation: f64,
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
    Periodic,
}

/// The six faces of a `Grid3D` domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Face {
    West,
    East,
    South,
    North,
    Down,
    Up,
}

/// Per-face boundary condition assignment for a rectangular 3D domain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundaryAssignment3D {
    pub west: BoundaryKind3D,
    pub east: BoundaryKind3D,
    pub south: BoundaryKind3D,
    pub north: BoundaryKind3D,
    pub down: BoundaryKind3D,
    pub up: BoundaryKind3D,
}

impl BoundaryAssignment3D {
    pub fn get(&self, face: Face) -> &BoundaryKind3D {
        match face {
            Face::West => &self.west,
            Face::East => &self.east,
            Face::South => &self.south,
            Face::North => &self.north,
            Face::Down => &self.down,
            Face::Up => &self.up,
        }
    }

    pub fn all_periodic() -> Self {
        Self {
            west: BoundaryKind3D::Periodic,
            east: BoundaryKind3D::Periodic,
            south: BoundaryKind3D::Periodic,
            north: BoundaryKind3D::Periodic,
            down: BoundaryKind3D::Periodic,
            up: BoundaryKind3D::Periodic,
        }
    }
}

/// Full 3D case setup: grid + 6-face boundary assignment + gas properties.
#[derive(Clone, Debug)]
pub struct CaseConfig3D {
    pub grid: crate::grid3d::Grid3D,
    pub bcs: BoundaryAssignment3D,
    pub gas: GasProperties,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_assignment_lookup() {
        let ba = BoundaryAssignment::all_periodic();
        assert_eq!(*ba.get(Edge::West), BoundaryKind::Periodic);
    }

    #[test]
    fn monatomic_default_prandtl_two_thirds() {
        let g = GasProperties::monatomic_default();
        assert!((g.prandtl - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn monatomic_default_collision_model_is_shakhov() {
        let g = GasProperties::monatomic_default();
        assert_eq!(g.collision_model_kind, CollisionModelKind::Shakhov);
    }
}
