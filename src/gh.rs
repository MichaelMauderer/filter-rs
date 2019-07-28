/*!
Provides implementations of and related to the g-h and g-h-k filter.
*/

use num_traits::float::FloatCore;
use num_traits::Float;

/// A g-h filter.
///
/// # Example
///
/// ```
/// use filter::gh::GHFilter;
/// use assert_approx_eq::assert_approx_eq;
///
/// let mut fgh: GHFilter<f32> = GHFilter::new(0.0, 0.0, 0.8, 0.2, 1.0);
/// assert_approx_eq!(0.8, fgh.update(1.0));
/// assert_approx_eq!(0.2, fgh.dxt);
///
/// fgh.g = 1.0;
/// fgh.h = 0.01;
/// assert_approx_eq!(2.0, fgh.update(2.0));
///```
///
/// # References
/// *  Labbe, "Kalman and Bayesian Filters in Python" http://rlabbe.github.io/Kalman-and-Bayesian-Filters-in-Python
/// *  Brookner, "Tracking and Kalman Filters Made Easy". John Wiley and Sons, 1998.
///
pub struct GHFilter<T> {
    /// Filter g gain parameter.
    pub g: T,
    /// Filter h gain parameter.
    pub h: T,
    /// Timestep (time between sample)
    pub dt: T,
    /// State of the filter.
    pub xt: T,
    /// Derivative of the filter state.
    pub dxt: T,
    /// Predicted filter state.
    pub x_p: T,
    /// Predicted derivative of the filter state.
    pub dx_p: T,
}

impl<T: FloatCore> GHFilter<T> {
    /// Returns a g-h filter with the given initialisation parameters.
    ///
    /// # Arguments
    ///
    ///
    /// * `x0` - initial value for the filter state.
    /// * `dx0` - initial value for the derivative of the filter state.
    /// * `g` - filter g gain parameter.
    /// * `h` - filter h gain parameter.
    /// * `dt` - time between samples.
    ///
    /// # Example
    ///
    /// ```
    /// use filter::gh::GHFilter;
    /// let fgh = GHFilter::new(0.0, 0.0, 0.2 ,0.2, 0.01);
    /// ```
    pub fn new(x0: T, dx0: T, g: T, h: T, dt: T) -> GHFilter<T> {
        GHFilter {
            g,
            h,
            dt,
            xt: x0,
            dxt: dx0,
            x_p: x0,
            dx_p: dx0,
        }
    }

    /// Performs the g-h filter predict and update step on the given measurement z.
    /// Returns the new state of x.
    pub fn update(&mut self, z: T) -> T {
        // Predict
        self.dx_p = self.dxt;
        self.x_p = self.xt + self.dt * self.dxt;
        // Update
        let y = z - self.x_p;
        self.dxt = self.dx_p + self.h * (y / self.dt);
        self.xt = self.x_p + self.g * y;
        self.xt
    }

    /// Returns the Variance Reduction Factor (VRF) of the state variable
    /// of the filter (x) and its derivatives (dx, ddx).
    pub fn vrf(&self) -> (T, T) {
        let two = T::one() + T::one();
        let three = two + T::one();
        let four = two + two;

        let den = self.g * (four - two * self.g - self.h);

        let vx = (two * self.g.powi(2) + two * self.h - three * self.g * self.h) / den;
        let vdx = two * self.h.powi(2) / (self.dt.powi(2) * den);

        (vx, vdx)
    }

    /// Returns the Variance Reduction Factor of the prediction step of the filter.
    ///
    ///  # References
    ///  * Asquith, "Weight Selection in First Order Linear Filters" Report No RG-TR-69-12,
    ///    U.S. Army Missle Command. Redstone Arsenal, Al. November 24, 1970.
    pub fn vrf_prediction(&self) -> T {
        let two = T::one() + T::one();
        let four = two + two;

        (two * self.g.powi(2) + two * self.h + self.g * self.h)
            / (self.g * (four - two * self.g - self.h))
    }
}

/// A g-h-k filter.
///
/// # Example
///
/// ```
/// use filter::gh::GHFilter;
/// use assert_approx_eq::assert_approx_eq;
///
/// let mut fgh: GHFilter<f32> = GHFilter::new(0.0, 0.0, 0.8, 0.2, 1.0);
/// assert_approx_eq!(0.8, fgh.update(1.0));
/// assert_approx_eq!(0.2, fgh.dxt);
///
/// fgh.g = 1.0;
/// fgh.h = 0.01;
/// assert_approx_eq!(2.0, fgh.update(2.0));
///
/// ```
///
/// # References
/// * Brookner, "Tracking and Kalman Filters Made Easy". John Wiley and Sons, 1998.
///
pub struct GHKFilter<T> {
    /// Filter g gain parameter.
    pub g: T,
    /// Filter h gain parameter.
    pub h: T,
    /// Filter k gain parameter.
    pub k: T,
    /// Timestep (time between sample)
    pub dt: T,
    /// State of the filter.
    pub xt: T,
    /// First Derivative of the filter state.
    pub dxt: T,
    /// Second derivative of the filter state.
    pub ddxt: T,
    /// Predicted filter state.
    pub x_p: T,
    /// Predicted first derivative of the filter state.
    pub dx_p: T,
    /// Predicted second derivative of the filter state.
    pub ddx_p: T,
}

impl<T: FloatCore> GHKFilter<T> {
    /// Returns a g-h-k filter with the given initialisation parameters.
    ///
    /// # Arguments
    ///
    ///
    /// * `x0` - initial value for the filter state.
    /// * `dx0` - initial value for the first derivative of the filter state.
    /// * `ddx0` - initial value for the second derivative of the filter state.
    /// * `g` - filter g gain parameter.
    /// * `h` - filter h gain parameter.
    /// * `k` - filter k gain parameter.
    /// * `dt` - time between samples.
    ///
    /// # Example
    ///
    /// ```
    /// use filter::gh::GHFilter;
    /// let fgh = GHFilter::new(0.0, 0.0, 0.2 ,0.2, 0.01);
    /// ```
    pub fn new(x0: T, dx0: T, ddx0: T, g: T, h: T, k: T, dt: T) -> GHKFilter<T> {
        GHKFilter {
            g,
            h,
            k,
            dt,
            xt: x0,
            dxt: dx0,
            ddxt: ddx0,
            x_p: x0,
            dx_p: dx0,
            ddx_p: ddx0,
        }
    }

    /// Performs the g-h filter predict and update step on the measurement z.
    /// Returns the new value for x.
    pub fn update(&mut self, z: T) -> T {
        let two = T::one() + T::one();
        // Predict
        self.ddx_p = self.ddxt;
        self.dx_p = self.dxt + self.ddxt * self.dt;
        self.x_p = self.xt + self.dt * self.dxt + self.ddxt * self.dt * self.dt / two;
        // Update
        let y = z - self.x_p;

        self.ddxt = self.ddx_p + two * self.k * y / (self.dt * self.dt);
        self.dxt = self.dx_p + self.h * (y / self.dt);
        self.xt = self.x_p + self.g * y;
        self.xt
    }

    ///Returns the Variance Reduction Factor for x of the prediction step of the filter.
    ///
    /// # References
    /// * Asquith and Woods, "Total Error Minimization in First and Second Order Prediction Filters"
    ///   Report No RE-TR-70-17, U.S. Army Missle Command. Redstone Arsenal, Al. November 24, 1970.
    pub fn vrf_prediction(&self) -> T {
        let two = T::from(2).unwrap();
        let four = T::from(4).unwrap();

        let g = self.g;
        let h = self.h;
        let k = self.k;
        let gh2 = two * g + h;

        ((g * k * (gh2 - four) + h * (g * gh2 + two * h))
            / (two * k - (g * (h + k) * (gh2 - four))))
    }

    /// Returns the Variance Reduction Factor (VRF) of the state variable
    /// of the filter (x) and its derivatives (dx, ddx).
    pub fn vrf(&self) -> (T, T, T) {
        let two = T::from(2).unwrap();
        let four = T::from(4).unwrap();
        let eight = T::from(8).unwrap();

        let g = self.g;
        let h = self.h;
        let k = self.k;

        let hg4 = four - two * g - h;
        let ghk = g * h + g * k - two * k;

        let vx = (two * h * (two * g.powi(2) + two * h - two * g * h) - two * g * k * hg4)
            / (two * k - g * (h + k) * hg4);
        let vdx = (two * (h.powi(2)) - four * h.powi(2) * k + four * k.powi(2) * (two - g))
            / (two * hg4 * ghk);
        let vddx = eight * h * k.powi(2) / ((self.dt.powi(4)) * hg4 * ghk);

        (vx, vdx, vddx)
    }

    /// Returns the bias error given the specified constant jerk(dddx).
    ///
    /// # References
    /// * Asquith and Woods, "Total Error Minimization in First and Second Order Prediction Filters"
    ///   Report No RE-TR-70-17, U.S. Army Missle Command. Redstone Arsenal, Al. November 24, 1970.
    pub fn bias_error(&self, dddx: T) -> T {
        -self.dt.powi(2) * dddx / (T::from(2.0).unwrap() * self.k)
    }
}

/// Returns g, h, k parameters for optimal smoothing of noise for a given value of g.
/// This is due to Polge and Bhagavan.
///
/// # References
/// * Polge and Bhagavan. "A Study of the g-h-k Tracking Filter". Report No. RE-CR-76-1.
///   University of Alabama in Huntsville. July, 1975
pub fn optimal_noise_smoothing<T: Float>(g: T) -> (T, T, T) {
    let one = T::one();
    let two = T::from(2).unwrap();
    let four = T::from(4).unwrap();
    let eight = T::from(8).unwrap();
    let sixty_four = T::from(64).unwrap();

    let h = ((two * g.powi(3) - four * g.powi(2))
        + (four * g.powi(6) - sixty_four * g.powi(5) + sixty_four * g.powi(4)).sqrt())
        / (eight * (one - g));
    let k = (h * (two - g) - g.powi(2)) / g;

    (g, h, k)
}

/// An order 1 least squared filter can be computed by a g-h filter by varying g and h over time
/// according to the formulas below, where the first measurement is at n=0, the second is
/// at n=1, and so on:
pub fn least_squares_parameters<T: FloatCore>(n: T) -> (T, T) {
    let one = T::one();
    let two = T::from(2).unwrap();
    let six = T::from(6).unwrap();

    let den = (n + two) * (n + one);

    let g = (two * (two * n + one)) / den;
    let h = six / den;
    (g, h)
}

/// Computes values for g and h for a critically damped filter.
/// The idea here is to create a filter that reduces the influence of old data as new data comes in.
/// This allows the filter to track a moving target better. This goes by different names.
/// It may be called the discounted least-squares g-h filter, a fading-memory polynomal filter
/// of order 1, or a critically damped g-h filter.
///
/// # References
/// * Brookner, "Tracking and Kalman Filters Made Easy". John Wiley and Sons, 1998.
/// * Polge and Bhagavan. "A Study of the g-h-k Tracking Filter". Report No. RE-CR-76-1.
///   University of Alabama in Huntsville. July, 1975
pub fn critical_damping_parameters_order_two<T: FloatCore>(theta: T) -> (T, T) {
    let one = T::one();

    (one - theta.powi(2), (one - theta).powi(2))
}

/// Computes values for g, h and k for a critically damped filter.
/// The idea here is to create a filter that reduces the influence of old data as new data comes in.
/// This allows the filter to track a moving target better. This goes by different names.
/// It may be called the discounted least-squares g-h filter, a fading-memory polynomal filter
/// of order 1, or a critically damped g-h filter.
///
/// # References
/// * Brookner, "Tracking and Kalman Filters Made Easy". John Wiley and Sons, 1998.
/// * Polge and Bhagavan. "A Study of the g-h-k Tracking Filter". Report No. RE-CR-76-1.
///   University of Alabama in Huntsville. July, 1975
pub fn critical_damping_parameters_order_three<T: FloatCore>(theta: T) -> (T, T, T) {
    let one = T::one();
    let two = T::from(2).unwrap();
    let three = T::from(2).unwrap();

    (
        one - theta.powi(3),
        (three / two) * (one - theta.powi(2)) * (one - theta),
        (one / two) * (one - theta).powi(3),
    )
}

/// Computes the g,h constants for a Benedict-Bornder filter, which minimizes transient errors
/// for a g-h filter. Returns the values g,h for a specified g. Strictly speaking, only h
/// is computed, g is returned unchanged. The default formula for the Benedict-Bordner allows ringing.
/// We can "nearly" critically damp it; ringing will be reduced, but not entirely eliminated at
/// the cost of reduced performance.
///
/// # References
/// * Brookner, "Tracking and Kalman Filters Made Easy". John Wiley and Sons, 1998.
pub fn benedict_bornder_constants<T: Float>(g: T, critical: bool) -> (T, T) {
    let g_sqr = g.powi(2);
    if critical {
        (
            g,
            T::from(0.8).unwrap()
                * (T::from(2).unwrap() - g_sqr - T::from(2).unwrap() * (T::one() - g_sqr).sqrt())
                / g_sqr,
        )
    } else {
        (g, g_sqr / (T::from(2).unwrap() - g))
    }
}
