//! janus-core: shared data structures for the Janus cross-scale fluid solver.
//!
//! Contains the structured-grid geometry (`Grid2D`), macroscopic field storage
//! (`MacroFields`, SoA), the discrete-velocity distribution (`Distribution`,
//! cell-major indexing), a POD `Vec2` type, physical constants/units, and the
//! case/gas configuration structs used to set up a simulation.
//!
//! Milestone 1 scope: 2D structured Cartesian grid only. The `Mesh`-style API
//! is kept narrow deliberately so non-uniform/AMR/unstructured grids can be
//! added later without breaking callers (see ENGINEERING_SPEC.md §1).

pub mod config;
pub mod distribution;
pub mod fields;
pub mod fields3d;
pub mod grid;
pub mod grid3d;
pub mod units;
pub mod vec2;

pub use config::{
    BoundaryAssignment, BoundaryAssignment3D, BoundaryKind, BoundaryKind3D, CaseConfig,
    CaseConfig3D, Face, GasProperties,
};
pub use distribution::{Distribution, Distribution3D};
pub use fields::MacroFields;
pub use fields3d::MacroFields3D;
pub use grid::Grid2D;
pub use grid3d::Grid3D;
pub use vec2::Vec2;
