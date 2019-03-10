/*!
Provides statics related utility functions used in other parts of the library.
*/
use std::ops::{Add, Mul};

use num_traits::Float;

/// Represents a gaussian distribution with mean and variance..
pub struct GaussianDistribution<F: Float> {
    /// Mean of the distribution.
    pub mean: F,
    /// Variance of the distribution.
    pub var: F,
}

impl<F: Float> GaussianDistribution<F> {
    /// Create a new Gaussian distribution.
    pub fn new(mean: F, var: F) -> Self {
        GaussianDistribution { mean, var }
    }
}

impl<F: Float> Add for GaussianDistribution<F> {
    type Output = GaussianDistribution<F>;

    fn add(self, other: GaussianDistribution<F>) -> GaussianDistribution<F> {
        GaussianDistribution {
            mean: self.mean + other.mean,
            var: self.var + other.var,
        }
    }
}

impl<F: Float> Mul for GaussianDistribution<F> {
    type Output = GaussianDistribution<F>;

    fn mul(self, other: GaussianDistribution<F>) -> GaussianDistribution<F> {
        let mean = (self.var * other.mean + other.var * self.mean) / (self.var + other.var);
        let var = F::one() / (F::one() / self.var + F::one() / other.var);

        GaussianDistribution { mean, var }
    }
}
