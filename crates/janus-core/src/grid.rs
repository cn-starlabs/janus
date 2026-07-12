//! Structured 2D Cartesian grid geometry.

use serde::{Deserialize, Serialize};

/// A uniform structured Cartesian grid, cell-centered finite-volume storage.
///
/// Cells are indexed `(i, j)` with `i` in `[0, nx)` fastest-varying (row-major,
/// C order) and `j` in `[0, ny)`. Cell `(i, j)` center is at
/// `origin + ((i+0.5)*dx, (j+0.5)*dy)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Grid2D {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub origin: [f64; 2],
}

impl Grid2D {
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64, origin: [f64; 2]) -> Self {
        assert!(
            nx > 0 && ny > 0,
            "grid must have at least one cell in each dimension"
        );
        assert!(dx > 0.0 && dy > 0.0, "grid spacing must be positive");
        Self {
            nx,
            ny,
            dx,
            dy,
            origin,
        }
    }

    #[inline]
    pub fn ncells(&self) -> usize {
        self.nx * self.ny
    }

    /// Row-major (C order) linear index of cell `(i, j)`.
    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.nx && j < self.ny);
        j * self.nx + i
    }

    /// Inverse of `idx`.
    #[inline]
    pub fn coords(&self, cell: usize) -> (usize, usize) {
        (cell % self.nx, cell / self.nx)
    }

    /// Cell-center physical coordinates for cell `(i, j)`.
    #[inline]
    pub fn center(&self, i: usize, j: usize) -> [f64; 2] {
        [
            self.origin[0] + (i as f64 + 0.5) * self.dx,
            self.origin[1] + (j as f64 + 0.5) * self.dy,
        ]
    }

    /// Neighbor cell index in the `+x` direction, with wraparound (periodic)
    /// or `None` at the boundary (caller applies BC/ghost logic instead).
    #[inline]
    pub fn east(&self, i: usize, j: usize) -> Option<(usize, usize)> {
        if i + 1 < self.nx {
            Some((i + 1, j))
        } else {
            None
        }
    }
    #[inline]
    pub fn west(&self, i: usize, j: usize) -> Option<(usize, usize)> {
        if i > 0 {
            Some((i - 1, j))
        } else {
            None
        }
    }
    #[inline]
    pub fn north(&self, i: usize, j: usize) -> Option<(usize, usize)> {
        if j + 1 < self.ny {
            Some((i, j + 1))
        } else {
            None
        }
    }
    #[inline]
    pub fn south(&self, i: usize, j: usize) -> Option<(usize, usize)> {
        if j > 0 {
            Some((i, j - 1))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idx_roundtrip() {
        let g = Grid2D::new(4, 3, 1.0, 1.0, [0.0, 0.0]);
        for j in 0..g.ny {
            for i in 0..g.nx {
                let c = g.idx(i, j);
                assert_eq!(g.coords(c), (i, j));
            }
        }
        assert_eq!(g.ncells(), 12);
    }

    #[test]
    fn neighbors_at_edges() {
        let g = Grid2D::new(2, 2, 1.0, 1.0, [0.0, 0.0]);
        assert_eq!(g.west(0, 0), None);
        assert_eq!(g.east(0, 0), Some((1, 0)));
        assert_eq!(g.south(0, 0), None);
        assert_eq!(g.north(0, 0), Some((0, 1)));
    }
}
