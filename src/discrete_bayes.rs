/*!
Provides implementations of and related to Discrete Bayes filtering.
*/
use num_traits::Float;

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
        return Err(())
    }
    let mut posterior: Vec<F> = likelihood.iter().zip(prior.iter()).map(|(&l, &p)| l * p).collect();
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
    unimplemented!()
}