/*!
Provides utility functions used in other parts of the library.
*/
use std::num::Wrapping;

use num_traits::Float;

/// Determines how the convolution is computed. This mostly effects behaviour at the boundaries.
pub(crate) enum ConvolutionMode<F> {
    /// Returns the convolution at each point of overlap, assuming the signals wrap around.
    Wrap,
    /// Returns the convolution at each point of overlap, assuming the signals
    /// are extended by the given value around.
    Extended(F),
}

/// Compute the discrete convolution of the two slices.
/// This might be slow, as this function is not optimised in any way.
pub(crate) fn convolve<F: Float>(a: &[F], b: &[F], mode: ConvolutionMode<F>) -> Vec<F> {
    let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };

    match mode {
        ConvolutionMode::Wrap => convolve_wrap(a, b),
        ConvolutionMode::Extended(c) => convolve_extended(a, b, c),
    }
}

fn convolve_extended<F: Float>(signal: &[F], window: &[F], c: F) -> Vec<F> {
    let m = signal.len() as i64;
    let n = window.len() as i64;
    debug_assert!(m >= n);

    let mut result = Vec::default();
    for i in 0..m {
        let mut x = F::zero();
        for j in 0..n {
            let s_ij = {
                let ix = i + j - n / 2;
                if ix < 0 {
                    c
                } else {
                    *signal.get(ix as usize).unwrap_or(&c)
                }
            };
            let w_ij = *window.get(j as usize).unwrap_or(&c);
            x = x + s_ij * w_ij;
        }
        result.push(x)
    }
    result
}

fn convolve_wrap<F: Float>(signal: &[F], window: &[F]) -> Vec<F> {
    let m = signal.len() as i64;
    let n = window.len() as i64;
    debug_assert!(m >= n);

    let mut result = Vec::with_capacity(m as usize);
    result.resize_with(m as usize, F::zero);

    for i in 0..m {
        for j in 0..n {
            let s_ij = {
                let ix = (i + j - (n / 2)) % m;
                let ix = if ix < 0 { ix + m } else { ix };
                signal[ix as usize]
            };
            let w_ij = window[j as usize];
            result[i as usize] = result[i as usize] + (s_ij * w_ij);
        }
    }
    result
}

fn roll<T: Copy>(a: &[T], shift: i64) -> Vec<T> {
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() as i64 {
        let ix = (Wrapping(i) - Wrapping(shift)).0 % (a.len() as i64 - 1);
        let ix = if ix < 0 { ix + a.len() as i64 } else { ix };
        out.push(a[ix as usize])
    }
    out
}

fn shift_extend<T: Copy>(a: &[T], shift: i64, value: T) -> Vec<T> {
    let mut out: Vec<T> = Vec::with_capacity(a.len());
    for i in 0..a.len() as i64 {
        let ix = i - shift;
        out.push(*a.get(ix as usize).unwrap_or(&value))
    }
    out
}

pub(crate) enum ShiftMode<F> {
    Wrap,
    Extend(F),
}

pub(crate) fn shift<T: Copy>(a: &[T], shift: i64, mode: ShiftMode<T>) -> Vec<T> {
    match mode {
        ShiftMode::Wrap => roll(a, shift),
        ShiftMode::Extend(c) => shift_extend(a, shift, c),
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn test_convolve_extended() {
        let a: [f64; 3] = [1.0, 2.0, 3.0];
        let b: [f64; 3] = [0.0, 1.0, 0.5];

        let c = convolve(&a, &b, ConvolutionMode::Extended(10.0));

        debug_assert_eq!(3, c.len());
        assert_approx_eq!(2.0, c[0]);
        assert_approx_eq!(3.5, c[1]);
        assert_approx_eq!(8.0, c[2]);
    }

    #[test]
    fn test_convolve_wrap() {
        let a: [f64; 3] = [1.0, 2.0, 3.0];
        let b: [f64; 3] = [0.0, 1.0, 0.5];

        let c = convolve_wrap(&a, &b);

        debug_assert_eq!(3, c.len());
        dbg!(&c);
        assert_approx_eq!(2.0, c[0]);
        assert_approx_eq!(3.5, c[1]);
        assert_approx_eq!(3.5, c[2]);
    }

    #[test]
    fn test_convolve_wrap2() {
        let a = vec![3.0, 1.0, 2.0, 3.0, 2.0, 1.0, 1.0];
        let b = vec![0.0, 1.0, 0.5];

        let c = convolve_wrap(&a, &b);

        debug_assert_eq!(7, c.len());
        dbg!(&c);
        assert_approx_eq!(3.5, c[0]);
        assert_approx_eq!(2.0, c[1]);
        assert_approx_eq!(3.5, c[2]);
        assert_approx_eq!(4.0, c[3]);
        assert_approx_eq!(2.5, c[4]);
        assert_approx_eq!(1.5, c[5]);
        assert_approx_eq!(2.5, c[6]);
    }

    #[test]
    fn test_roll() {
        let a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        let c = roll(&a, 2);

        debug_assert_eq!(10, c.len());
        debug_assert_eq!(8, c[0]);
        debug_assert_eq!(9, c[1]);
        debug_assert_eq!(0, c[2]);
        debug_assert_eq!(1, c[3]);
        debug_assert_eq!(2, c[4]);
        debug_assert_eq!(3, c[5]);
        debug_assert_eq!(4, c[6]);
        debug_assert_eq!(5, c[7]);
        debug_assert_eq!(6, c[8]);
        debug_assert_eq!(7, c[9]);
    }
}
