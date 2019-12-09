/*!
This module implements IIR moving averages on regular time series.
 */

use nalgebra::allocator::Allocator;
use nalgebra::base::dimension::DimName;
use nalgebra::{DefaultAllocator, RealField, U1, VectorN};

use core::cmp::{min, max};
use num_traits::NumCast;

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
    pub fn update(&mut self, value: VectorN<F, DimX>) -> &VectorN<F, DimX> {
        // e = α v + (1-α) e
        self.estimate *= F::one() - self.coef;
        self.estimate += value * self.coef;

        &self.estimate
    }

    /// TODO
    pub fn estimate(&self) -> &VectorN<F, DimX> {
        &self.estimate
    }
}


///
#[derive(Debug)]
pub struct JAMAParameters<F: RealField + NumCast> {
    gain_coef: F,
    low_power: F,
    alpha_min: F,
    alpha_delta: F,
    max_shrink: F,
}

impl<F: RealField + NumCast> Default for JAMAParameters<F> {
    fn default() -> Self {
        JAMAParameters {
            gain_coef: F::from(0.05).unwrap(),
            low_power: F::from(0.5).unwrap(),
            alpha_min: F::from(0.01).unwrap(),
            alpha_delta: F::from(1.0 - 0.01).unwrap(),
            max_shrink: F::from(0.9).unwrap(),
        }
    }
}

impl<F: RealField + NumCast> JAMAParameters<F> {
    /// TODO
    fn set_alpha_range(self, min: F, max: F) -> Result<Self, ()> {
        if ! (1 >= max && max >= min && min >= 0) {
            return Err(());
        }

        self.alpha_min = min;
        self.alpha_delta = max - min;
        Ok(self)
    }

    /// TODO
    fn set_shrink(self, max_shrink: F) -> Result<Self, ()> {
        if ! (1 >= max_shrink && max_shrink >= 0) {
            return Err(());
        }

        self.max_shrink = max_shrink;
        Ok(self)
    }

    /// TODO
    fn set_gain(self, gain: F) -> Result<Self, ()> {
        if ! (1 >= gain && gain >= 0) {
            return Err(());
        }
        self.gain_coef = gain;
        Ok(self)
    }

    /// TODO
    fn set_low_power(self, low_power: F) -> Result<Self, ()> {
        if ! (1 >= low_power && low_power >= 0) {
            return Err(());
        }
        self.low_power = low_power;
        Ok(self)
    }
}


/// Implements Martin Jambon's adaptive moving average
#[derive(Debug)]
pub struct JambonAdaptiveMA<F>
where
    F: RealField + NumCast,
    DefaultAllocator: Allocator<F, U1>,
{
    params: JAMAParameters<F>,
    estimate: F,
    previous: F,
    gain: ExponentialWMA<F, U1>,
    loss: ExponentialWMA<F, U1>,
}

impl<F> JambonAdaptiveMA<F>
where
    F: RealField + NumCast,
    DefaultAllocator: Allocator<F, U1>,
{
    /// TODO
    pub fn new(params: JAMAParameters<F>, start: F) -> Self {
        Self {
            params,
            estimate: start,
            gain: ExponentialWMA::new(params.gain_coef, 0),
            loss: ExponentialWMA::new(params.gain_coef, 0)
        }
    }

    /// TODO
    pub fn with_defaults(start: F) -> Self {
        Self::new(JAMAParameters::default(), start)
    }

    /// TODO
    pub fn update(&mut self, value: F) -> F {
        let slope = value - &self.previous;
        let gain = self.gain.update(max(slope, F::zero()));
        let loss = self.loss.update(min(slope, F::zero()));

        let travel = gain - loss;
        let i;

        if travel == F::zero() {
            i = F::one();
        } else {
            let r = (gain + loss).abs() / travel;
            let d = 2*r - 1;
            i = (1 + d.sign() * d.abs() ^ self.params.low_power) / 2
        }

        let alpha = max(
            self.params.max_shrink * self.previous_coef,
            self.params.alpha_min + i * self.params.alpha_delta
        );

        self.estimate *= F::one() - alpha;
        self.estimate += alpha * value;
        self.previous = value;

        self.estimate
    }

    /// TODO
    pub fn estimate(&self) -> F {
        &self.estimate
    }
}
