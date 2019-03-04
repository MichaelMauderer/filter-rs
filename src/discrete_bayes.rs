/*!
Provides implementations of and related to Discrete Bayes filtering.
*/
use num_traits::Float;

use crate::common::convolve;
use crate::common::shift;
use crate::common::ConvolutionMode;
use crate::common::ShiftMode;

/// Normalize distribution `pdf` in-place so it sums to 1.0.
///
/// # Example
///
/// ```
/// use filter::discrete_bayes::normalize;
/// use assert_approx_eq::assert_approx_eq;
///
/// let mut pdf = [1.0, 1.0, 1.0, 1.0];
/// normalize(&mut pdf);
///
/// assert_approx_eq!(pdf[0], 0.25_f64);
/// assert_approx_eq!(pdf[1], 0.25_f64);
/// assert_approx_eq!(pdf[2], 0.25_f64);
/// assert_approx_eq!(pdf[3], 0.25_f64);
/// ```
///
pub fn normalize<F: Float>(pdf: &mut [F]) {
    let sum = pdf.iter().fold(F::zero(), |p, q| p + *q);
    pdf.iter_mut().for_each(|f| *f = *f / sum);
}

/// Computes the posterior of a discrete random variable given a
/// discrete likelihood and prior. In a typical application the likelihood
/// will be the likelihood of a measurement matching your current environment,
/// and the prior comes from discrete_bayes.predict().
///
pub fn update<F: Float>(likelihood: &[F], prior: &[F]) -> Result<Vec<F>, ()> {
    if likelihood.len() != prior.len() {
        return Err(());
    }
    let mut posterior: Vec<F> = likelihood
        .iter()
        .zip(prior.iter())
        .map(|(&l, &p)| l * p)
        .collect();
    normalize(&mut posterior);
    Ok(posterior)
}

/// Determines what happens at the boundaries of the probability distribution.
pub enum EdgeHandling<F> {
    /// the  probability distribution is shifted and the given value is used to used to fill in missing elements.
    Constant(F),
    /// The probability distribution is wrapped around the array.
    Wrap,
}

/// Performs the discrete Bayes filter prediction step, generating the prior.
pub fn predict<F: Float>(pdf: &[F], offset: i64, kernel: &[F], mode: EdgeHandling<F>) -> Vec<F> {
    match mode {
        EdgeHandling::Constant(c) => convolve(
            &shift(pdf, offset, ShiftMode::Extend(c)),
            kernel,
            ConvolutionMode::Extended(c),
        ),
        EdgeHandling::Wrap => convolve(
            &shift(pdf, offset, ShiftMode::Wrap),
            kernel,
            ConvolutionMode::Wrap,
        ),
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn test_prediction_wrap_kernel_3() {
        let pdf = {
            let mut pdf = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            normalize(&mut pdf);
            pdf
        };

        let kernel = [0.5, 0.5, 0.5, 0.5];

        let result = predict(&pdf, -1, &kernel, EdgeHandling::Wrap);
        let reference = [0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5];
        dbg!(&result);
        dbg!(&reference);

        debug_assert_eq!(reference.len(), result.len());
        for i in 0..reference.len() {
            assert_approx_eq!(reference[i], result[i]);
        }
    }

    #[test]
    fn test_prediction_wrap_kernel_4() {
        let pdf = {
            let mut pdf = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 8.0];
            normalize(&mut pdf);
            pdf
        };

        let kernel = [0.5, 0.5, 0.5, 0.5];

        let result = predict(&pdf, 3, &kernel, EdgeHandling::Wrap);
        let reference = [
            0.29487179, 0.17948718, 0.08333333, 0.05128205, 0.05448718, 0.11217949, 0.22435897,
        ];
        dbg!(&result);
        dbg!(&reference);

        debug_assert_eq!(reference.len(), result.len());
        for i in 0..reference.len() {
            assert_approx_eq!(reference[i], result[i]);
        }
    }

    #[test]
    fn test_prediction_constant_kernel_4() {
        let pdf = {
            let mut pdf = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 8.0];
            normalize(&mut pdf);
            pdf
        };

        let kernel = [0.5, 0.5, 0.5, 0.5];

        let result = predict(&pdf, 3, &kernel, EdgeHandling::Constant(10.0));
        let reference = [
            10.0, 7.5, 2.50641026, 1.27564103, 0.05448718, 2.56089744, 7.51923077,
        ];
        dbg!(&result);
        dbg!(&reference);

        debug_assert_eq!(reference.len(), result.len());
        for i in 0..reference.len() {
            assert_approx_eq!(reference[i], result[i]);
        }
    }
}
