/*!
Provides utility functions used in other parts of the library.
*/
use num_traits::Float;

/// Determines how the convolution is computed. This mostly effects behaviour at the boundaries.
pub(crate) enum ConvolutionMode {
    /// Returns the convolution at each point of overlap.
    Full,
    /// Returns the convolution at each point of the signal.
    Same,
    /// Returns the convolution at each point where the signals overlap completely.
    Valid,
}

/// Compute the discrete convolution of the two slices.
/// This might be slow, as this function is not optimised in any way.
pub(crate) fn convolve<F: Float>(a: &[F], b: &[F], mode: ConvolutionMode) -> Vec<F> {
    let (a, b) = if a.len() < b.len() {
        (b, a)
    } else {
        (a, b)
    };

    match mode {
        ConvolutionMode::Full => convolve_full(a, b),
        ConvolutionMode::Same => convolve_same(a, b),
        ConvolutionMode::Valid => convolve_valid(a, b),
    }
}


/// Compute convolution between signal and window for every index in window.
/// Result will have same length as signal.
fn convolve_same<F: Float>(signal: &[F], window: &[F]) -> Vec<F> {
    let m = signal.len() as i64;
    let n = window.len() as i64;
    debug_assert!(m >= n);

    let mut result = Vec::default();
    for i in 0..m {
        let mut x = F::zero();
        for j in 0..n {
            let s_ij = {
                let ix = i - j + n / 2;
                if ix < 0 {
                    F::zero()
                } else {
                    *signal.get(ix as usize).unwrap_or(&F::zero())
                }
            };
            let &w_ij = window.get(j as usize).unwrap_or(&F::zero());
            x = x + s_ij * w_ij;
        }
        result.push(x)
    }
    result
}


fn convolve_valid<F: Float>(signal: &[F], window: &[F]) -> Vec<F> {
    let m = signal.len() as i64;
    let n = window.len() as i64;
    debug_assert!(m >= n);

    let mut result = Vec::default();
    for i in 0..m {
        let mut x = F::zero();
        let mut valid = true;
        for j in 0..n {
            let s_ij = {
                let ix = i - j + n / 2;
                if ix < 0 {
                    None
                } else {
                    signal.get(ix as usize)
                }
            };
            let w_ij = window.get(j as usize);

            if s_ij.is_none() || w_ij.is_none() {
                valid = false;
                break;
            }

            x = x + (*s_ij.unwrap()) * (*w_ij.unwrap());
        }
        if valid {
            result.push(x)
        }
    }
    result
}


fn convolve_full<F: Float>(signal: &[F], window: &[F]) -> Vec<F> {
    let m = signal.len() as i64;
    let n = window.len() as i64;
    debug_assert!(m >= n);

    let mut result = Vec::default();
    for i in (-n / 2)..(m + n / 2) {
        let mut x = F::zero();
        for j in 0..n {
            let s_ij = {
                let ix = i - j + n / 2;
                if ix < 0 {
                    F::zero()
                } else {
                    *signal.get(ix as usize).unwrap_or(&F::zero())
                }
            };
            let w_ij = *window.get(j as usize).unwrap_or(&F::zero());
            x = x + s_ij * w_ij;
        }
        result.push(x)
    }
    result
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn test_convolve_same() {
        let a: [f64; 3] = [1.0, 2.0, 3.0];
        let b: [f64; 3] = [0.0, 1.0, 0.5];

        let c = convolve(&a, &b, ConvolutionMode::Same);

        debug_assert_eq!(3, c.len());
        assert_approx_eq!(1.0, c[0]);
        assert_approx_eq!(2.5, c[1]);
        assert_approx_eq!(4.0, c[2]);
    }

    #[test]
    fn test_convolve_valid() {
        let a: [f64; 3] = [1.0, 2.0, 3.0];
        let b: [f64; 3] = [0.0, 1.0, 0.5];

        let c = convolve(&a, &b, ConvolutionMode::Valid);

        debug_assert_eq!(1, c.len());
        assert_approx_eq!(2.5, c[0]);
    }

    #[test]
    fn test_convolve_full() {
        let a: [f64; 3] = [1.0, 2.0, 3.0];
        let b: [f64; 3] = [0.0, 1.0, 0.5];

        let c = convolve(&a, &b, ConvolutionMode::Full);

        debug_assert_eq!(5, c.len());
        assert_approx_eq!(0.0, c[0]);
        assert_approx_eq!(1.0, c[1]);
        assert_approx_eq!(2.5, c[2]);
        assert_approx_eq!(4.0, c[3]);
        assert_approx_eq!(1.5, c[4]);
    }
}