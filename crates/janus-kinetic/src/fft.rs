//! Minimal pure-Rust radix-2 Cooley-Tukey FFT (in-place, iterative,
//! bit-reversal permutation), used by the fast-spectral Boltzmann collision
//! operator (`spectral_collision.rs`) to evaluate the 3D convolution
//! structure of the collision operator in O(N log N) per velocity node
//! instead of the O(N^2) direct-quadrature cost of the full Boltzmann
//! collision integral.
//!
//! No external FFT crate is used (ENGINEERING_SPEC.md's approved-dependency
//! whitelist does not include one; this hand-rolled implementation avoids
//! adding a new dependency without user approval per the workflow rules).
//! This is the classic textbook iterative radix-2 Cooley-Tukey algorithm
//! (Cooley & Tukey, "An algorithm for the machine calculation of complex
//! Fourier series", Math. Comp. 19, 297-301 (1965)) with a precomputed
//! twiddle-factor table (no per-call trig calls in the hot butterfly loop)
//! and bit-reversal permutation done via a simple O(N) index-swap pass.
//!
//! Only sizes that are exact powers of two are supported (`fft3d` below pads
//! the velocity grid up to the next power of two per axis if needed, at grid
//! -construction time, not in the per-step hot path).

/// A minimal complex number type (`#[repr(C)]`, `Copy`, POD) — avoids a
/// `num-complex` dependency for this self-contained FFT.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    #[inline]
    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }
    #[inline]
    pub fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }
    #[inline]
    pub fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }
    #[inline]
    pub fn mul(self, o: Complex) -> Complex {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    #[inline]
    pub fn scale(self, s: f64) -> Complex {
        Complex::new(self.re * s, self.im * s)
    }
    #[inline]
    pub fn conj(self) -> Complex {
        Complex::new(self.re, -self.im)
    }
}

/// Returns `true` iff `n` is a nonzero power of two.
#[inline]
pub fn is_pow2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Next power of two `>= n` (used to pad velocity grids for FFT-friendly
/// sizes at construction time).
pub fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Bit-reverse the lowest `bits` bits of `x`.
#[inline]
fn bit_reverse(mut x: usize, bits: u32) -> usize {
    let mut r = 0usize;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

/// In-place iterative radix-2 Cooley-Tukey FFT (or inverse FFT if
/// `inverse == true`, which additionally divides by `n` to give the exact
/// inverse transform, not just the conjugated forward transform).
///
/// `data.len()` MUST be a power of two (checked via `debug_assert!`;
/// callers — `fft3d` — pad grids to a power of two at construction time so
/// this is never violated in the hot per-step path).
pub fn fft_1d(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    debug_assert!(is_pow2(n), "fft_1d requires a power-of-two length, got {n}");
    if n <= 1 {
        return;
    }
    let bits = n.trailing_zeros();

    // Bit-reversal permutation (in place, O(n), each pair swapped once).
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if j > i {
            data.swap(i, j);
        }
    }

    // Iterative Cooley-Tukey butterflies, precomputing twiddle factors per
    // stage (no repeated trig calls inside the innermost butterfly loop).
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let theta_step = sign * 2.0 * std::f64::consts::PI / len as f64;
        // Precompute twiddle factors for this stage's half-length.
        let mut twiddles = Vec::with_capacity(half);
        for k in 0..half {
            let theta = theta_step * k as f64;
            twiddles.push(Complex::new(theta.cos(), theta.sin()));
        }
        let mut start = 0;
        while start < n {
            for k in 0..half {
                let w = twiddles[k];
                let a = data[start + k];
                let b = data[start + k + half].mul(w);
                data[start + k] = a.add(b);
                data[start + k + half] = a.sub(b);
            }
            start += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / n as f64;
        for c in data.iter_mut() {
            *c = c.scale(inv_n);
        }
    }
}

/// 3D FFT (or inverse) on a flat `nx*ny*nz` C-order complex array, done as
/// three passes of 1D FFTs along each axis (standard separable
/// multidimensional FFT construction). `nx`, `ny`, `nz` must each be a power
/// of two.
pub fn fft_3d(data: &mut [Complex], nx: usize, ny: usize, nz: usize, inverse: bool) {
    debug_assert_eq!(data.len(), nx * ny * nz);
    debug_assert!(is_pow2(nx) && is_pow2(ny) && is_pow2(nz));

    // Axis 0 (x, fastest-varying): contiguous runs of length nx.
    let mut line = vec![Complex::zero(); nx];
    for k in 0..nz {
        for j in 0..ny {
            let base = k * nx * ny + j * nx;
            line.copy_from_slice(&data[base..base + nx]);
            fft_1d(&mut line, inverse);
            data[base..base + nx].copy_from_slice(&line);
        }
    }
    // Axis 1 (y): stride nx.
    let mut line = vec![Complex::zero(); ny];
    for k in 0..nz {
        for i in 0..nx {
            let base = k * nx * ny + i;
            for (t, l) in line.iter_mut().enumerate() {
                *l = data[base + t * nx];
            }
            fft_1d(&mut line, inverse);
            for (t, l) in line.iter().enumerate() {
                data[base + t * nx] = *l;
            }
        }
    }
    // Axis 2 (z): stride nx*ny.
    let mut line = vec![Complex::zero(); nz];
    let plane = nx * ny;
    for j in 0..ny {
        for i in 0..nx {
            let base = j * nx + i;
            for (t, l) in line.iter_mut().enumerate() {
                *l = data[base + t * plane];
            }
            fft_1d(&mut line, inverse);
            for (t, l) in line.iter().enumerate() {
                data[base + t * plane] = *l;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_pow2_examples() {
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(5), 8);
        assert_eq!(next_pow2(8), 8);
        assert_eq!(next_pow2(9), 16);
    }

    #[test]
    fn fft_then_ifft_recovers_input() {
        let n = 16;
        let mut data: Vec<Complex> = (0..n)
            .map(|i| Complex::new((i as f64).sin(), (i as f64 * 0.3).cos()))
            .collect();
        let original = data.clone();
        fft_1d(&mut data, false);
        fft_1d(&mut data, true);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!(
                (a.re - b.re).abs() < 1e-9,
                "re mismatch {} vs {}",
                a.re,
                b.re
            );
            assert!(
                (a.im - b.im).abs() < 1e-9,
                "im mismatch {} vs {}",
                a.im,
                b.im
            );
        }
    }

    #[test]
    fn fft_of_dc_signal_is_delta_at_zero_freq() {
        let n = 8;
        let mut data = vec![Complex::new(3.0, 0.0); n];
        fft_1d(&mut data, false);
        assert!((data[0].re - 3.0 * n as f64).abs() < 1e-9);
        for c in &data[1..] {
            assert!(c.re.abs() < 1e-9 && c.im.abs() < 1e-9);
        }
    }

    #[test]
    fn fft_matches_direct_dft_small_case() {
        // Cross-check against a direct O(n^2) DFT for a small non-power-
        // structured signal, to catch butterfly-indexing bugs the DC/
        // round-trip tests above might not expose.
        let n = 8;
        let data: Vec<Complex> = (0..n)
            .map(|i| Complex::new(i as f64, -(i as f64) * 0.5))
            .collect();
        let mut via_fft = data.clone();
        fft_1d(&mut via_fft, false);

        let mut direct = vec![Complex::zero(); n];
        for (k, dk) in direct.iter_mut().enumerate() {
            let mut sum = Complex::zero();
            for (j, dj) in data.iter().enumerate() {
                let theta = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                let w = Complex::new(theta.cos(), theta.sin());
                sum = sum.add(dj.mul(w));
            }
            *dk = sum;
        }
        for (a, b) in via_fft.iter().zip(direct.iter()) {
            assert!((a.re - b.re).abs() < 1e-9, "re {} vs {}", a.re, b.re);
            assert!((a.im - b.im).abs() < 1e-9, "im {} vs {}", a.im, b.im);
        }
    }

    #[test]
    fn fft_3d_round_trip() {
        let (nx, ny, nz) = (4, 4, 4);
        let n = nx * ny * nz;
        let mut data: Vec<Complex> = (0..n)
            .map(|i| Complex::new((i as f64 * 0.7).sin(), 0.0))
            .collect();
        let original = data.clone();
        fft_3d(&mut data, nx, ny, nz, false);
        fft_3d(&mut data, nx, ny, nz, true);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a.re - b.re).abs() < 1e-8);
            assert!((a.im - b.im).abs() < 1e-8);
        }
    }
}
