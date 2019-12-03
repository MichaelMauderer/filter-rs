/*!
This module implements IIR moving averages on regular time series.
 */

use nalgebra::allocator::Allocator;
use nalgebra::base::dimension::DimName;
use nalgebra::{DefaultAllocator, RealField, VectorN};

/// Implements a naive, exponentially-weighted moving average.
#[derive(Debug)]
pub struct ExponentialWMA<F, DimX>
where
    F: RealField,
    DimX: DimName,
    DefaultAllocator: Allocator<F, DimX>,
{
    coef: F,
    estimate: VectorN<F, DimX>,
}

impl<F, DimX> ExponentialWMA<F, DimX>
where
    F: RealField,
    DimX: DimName,
    DefaultAllocator: Allocator<F, DimX>,
{
    /// TODO
    pub fn new(coef: F, start: VectorN<F, DimX>) -> Self {
        assert!(F::zero() < coef && coef < F::one());
        Self {
            coef,
            estimate: start,
        }
    }

    /// TODO
    pub fn update(&mut self, value: VectorN<F, DimX>) {
        // e = α v + (1-α) e
        self.estimate *= F::one() - self.coef;
        self.estimate += value * self.coef;
    }

    /// TODO
    pub fn estimate(&self) -> &VectorN<F, DimX> {
        &self.estimate
    }
}
