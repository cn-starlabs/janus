//! Structured 3D Cartesian grid geometry (M4 extension).
//!
//! DESIGN: rather than mutating `Grid2D`'s field names/method signatures in
//! place (which dozens of call sites across `janus-kinetic`/`janus-sched`
//! access positionally as `grid.nx`, `grid.idx(i, j)` two-argument, etc. —
//! rewriting those signatures without a compiler to catch every call site
//! would be far riskier than adding a parallel type), `Grid3D` is a new,
//! independent type that generalizes the *same* Cartesian/FVM/cell-centered
//! design to `nx * ny * nz` cells, C-order linear indexing (matching the
//! `.jvtk` format's required C-order layout, ENGINEERING_SPEC.md §4). A 2D
//! case is exactly `Grid3D { nz: 1, .. }`; every method below reduces to the
//! `Grid2D` behavior when `nz == 1` (verified by
//! `nz_one_matches_grid2d_indexing` below). This satisfies the spec's
//! "generalize don't duplicate ... unless existing code is so hardcoded-2D
//! that generalizing is riskier" escape hatch: `Grid2D` itself remains
//! untouched (existing 2D solver/sched code keeps compiling exactly as
//! before), while `Grid3D` is the forward-compatible generalized type new
//! 3D-aware code (to be wired into `janus-kinetic`/`janus-io` as a follow-on)
//! should use.
//!
//! Cells are indexed `(i, j, k)` with `i` in `[0, nx)` fastest-varying, then
//! `j` in `[0, ny)`, then `k` in `[0, nz)` slowest-varying (row-major/C
//! order): `idx(i, j, k) = k * (nx*ny) + j * nx + i`. Cell `(i, j, k)` center
//! is at `origin + ((i+0.5)*dx, (j+0.5)*dy, (k+0.5)*dz)`.

/// A uniform structured Cartesian grid, cell-centered finite-volume storage,
/// generalized to 3 dimensions (`nz = 1` recovers the 2D case exactly).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub origin: [f64; 3],
}

impl Grid3D {
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        origin: [f64; 3],
    ) -> Self {
        assert!(
            nx > 0 && ny > 0 && nz > 0,
            "grid must have at least one cell in each dimension"
        );
        assert!(
            dx > 0.0 && dy > 0.0 && dz > 0.0,
            "grid spacing must be positive"
        );
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            origin,
        }
    }

    /// Build a 3D grid from an existing 2D grid, with a single layer in `z`
    /// (`nz = 1`, `dz` supplied since a 2D `Grid2D` carries no z-extent of
    /// its own; a nominal value equal to `min(dx, dy)` is a reasonable
    /// default for callers that don't care, but is NOT assumed here —
    /// callers must supply it explicitly to avoid a silent physical-units
    /// guess).
    pub fn from_2d(g: &super::grid::Grid2D, dz: f64) -> Self {
        Self::new(
            g.nx,
            g.ny,
            1,
            g.dx,
            g.dy,
            dz,
            [g.origin[0], g.origin[1], 0.0],
        )
    }

    #[inline]
    pub fn ncells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    #[inline]
    pub fn plane_cells(&self) -> usize {
        self.nx * self.ny
    }

    /// C-order linear index of cell `(i, j, k)`: matches `.jvtk`'s required
    /// C-order field layout (ENGINEERING_SPEC.md §4) so a `MacroFields3D`
    /// array can be `bytemuck::cast_slice`'d directly to/from a `.jvtk`
    /// block with no reordering. Reduces to `Grid2D::idx`'s `j*nx+i` when
    /// `k == 0` (single z-layer).
    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        debug_assert!(i < self.nx && j < self.ny && k < self.nz);
        k * self.plane_cells() + j * self.nx + i
    }

    /// Inverse of `idx`.
    #[inline]
    pub fn coords(&self, cell: usize) -> (usize, usize, usize) {
        let plane = self.plane_cells();
        let k = cell / plane;
        let rem = cell % plane;
        let j = rem / self.nx;
        let i = rem % self.nx;
        (i, j, k)
    }

    /// Cell-center physical coordinates for cell `(i, j, k)`.
    #[inline]
    pub fn center(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        [
            self.origin[0] + (i as f64 + 0.5) * self.dx,
            self.origin[1] + (j as f64 + 0.5) * self.dy,
            self.origin[2] + (k as f64 + 0.5) * self.dz,
        ]
    }

    /// Neighbor cell index in the `+x`/`-x`/`+y`/`-y`/`+z`/`-z` directions
    /// (the 6 faces of a 3D cell), `None` at a domain boundary (caller
    /// applies BC/ghost logic instead, same convention as `Grid2D`).
    #[inline]
    pub fn east(&self, i: usize, j: usize, k: usize) -> Option<(usize, usize, usize)> {
        if i + 1 < self.nx {
            Some((i + 1, j, k))
        } else {
            None
        }
    }
    #[inline]
    pub fn west(&self, i: usize, j: usize, k: usize) -> Option<(usize, usize, usize)> {
        if i > 0 {
            Some((i - 1, j, k))
        } else {
            None
        }
    }
    #[inline]
    pub fn north(&self, i: usize, j: usize, k: usize) -> Option<(usize, usize, usize)> {
        if j + 1 < self.ny {
            Some((i, j + 1, k))
        } else {
            None
        }
    }
    #[inline]
    pub fn south(&self, i: usize, j: usize, k: usize) -> Option<(usize, usize, usize)> {
        if j > 0 {
            Some((i, j - 1, k))
        } else {
            None
        }
    }
    /// `+z` neighbor ("top", the face a 2D solver has no equivalent of).
    #[inline]
    pub fn up(&self, i: usize, j: usize, k: usize) -> Option<(usize, usize, usize)> {
        if k + 1 < self.nz {
            Some((i, j, k + 1))
        } else {
            None
        }
    }
    /// `-z` neighbor ("bottom").
    #[inline]
    pub fn down(&self, i: usize, j: usize, k: usize) -> Option<(usize, usize, usize)> {
        if k > 0 {
            Some((i, j, k - 1))
        } else {
            None
        }
    }

    /// `[dims]` triple in the `.jvtk` header's `"dims": [nx, ny, nz]` sense
    /// (ENGINEERING_SPEC.md §4) — convenience accessor so `janus-io` callers
    /// don't need to destructure the grid manually.
    #[inline]
    pub fn dims(&self) -> [usize; 3] {
        [self.nx, self.ny, self.nz]
    }

    #[inline]
    pub fn spacing(&self) -> [f64; 3] {
        [self.dx, self.dy, self.dz]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;

    #[test]
    fn idx_roundtrip_3d() {
        let g = Grid3D::new(4, 3, 5, 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]);
        for k in 0..g.nz {
            for j in 0..g.ny {
                for i in 0..g.nx {
                    let c = g.idx(i, j, k);
                    assert_eq!(g.coords(c), (i, j, k));
                }
            }
        }
        assert_eq!(g.ncells(), 60);
    }

    #[test]
    fn neighbors_at_edges_3d() {
        let g = Grid3D::new(2, 2, 2, 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]);
        assert_eq!(g.west(0, 0, 0), None);
        assert_eq!(g.east(0, 0, 0), Some((1, 0, 0)));
        assert_eq!(g.south(0, 0, 0), None);
        assert_eq!(g.north(0, 0, 0), Some((0, 1, 0)));
        assert_eq!(g.down(0, 0, 0), None);
        assert_eq!(g.up(0, 0, 0), Some((0, 0, 1)));
        assert_eq!(g.up(0, 0, 1), None);
    }

    /// Hand-verified correctness of the "nz=1 collapses to Grid2D" claim
    /// used throughout this module's DESIGN rationale: for every cell in a
    /// single-z-layer `Grid3D`, the linear index must exactly equal the
    /// corresponding `Grid2D::idx` (same `(i,j)`, `k` forced to 0), and the
    /// (x,y) center coordinates must match too (z center is `0.5*dz` above
    /// the z-origin, which has no 2D analog and is not compared).
    #[test]
    fn nz_one_matches_grid2d_indexing() {
        let g2 = Grid2D::new(5, 4, 0.1, 0.2, [1.0, -2.0]);
        let g3 = Grid3D::from_2d(&g2, 0.3);
        assert_eq!(g3.nx, g2.nx);
        assert_eq!(g3.ny, g2.ny);
        assert_eq!(g3.nz, 1);
        for j in 0..g2.ny {
            for i in 0..g2.nx {
                let c2 = g2.idx(i, j);
                let c3 = g3.idx(i, j, 0);
                assert_eq!(c2, c3, "index mismatch at ({i},{j})");
                let center2 = g2.center(i, j);
                let center3 = g3.center(i, j, 0);
                assert!((center2[0] - center3[0]).abs() < 1e-12);
                assert!((center2[1] - center3[1]).abs() < 1e-12);
            }
        }
        assert_eq!(g3.ncells(), g2.ncells());
    }

    /// Hand-traced 3D face-flux index arithmetic: for a 3x3x3 grid, the
    /// linear index of the "+z" (`up`) neighbor of cell (1,1,1) must be
    /// exactly `plane_cells` (= nx*ny = 9) greater than the cell's own
    /// index, since `idx` increments by exactly one full x-y plane per unit
    /// `k`. This directly exercises the index-arithmetic a 3D DUGKS flux
    /// kernel's "top/bottom face" loop would rely on.
    #[test]
    fn z_face_neighbor_index_offset_is_one_plane() {
        let g = Grid3D::new(3, 3, 3, 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]);
        let c = g.idx(1, 1, 1);
        let (ui, uj, uk) = g
            .up(1, 1, 1)
            .expect("k=1 has a k=2 neighbor in a 3-layer grid");
        let c_up = g.idx(ui, uj, uk);
        assert_eq!(c_up - c, g.plane_cells());
        let (di, dj, dk) = g.down(1, 1, 1).expect("k=1 has a k=0 neighbor");
        let c_down = g.idx(di, dj, dk);
        assert_eq!(c - c_down, g.plane_cells());
    }
}
