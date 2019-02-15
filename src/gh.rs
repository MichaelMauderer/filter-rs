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
///
/// ```
pub struct GHFilter<T> {
    pub g: T,
    pub h: T,
    pub dt: T,
    pub xt: T,
    pub dxt: T,
    pub x_p: T,
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

    pub fn vrf(&self) -> (T, T) {
        let two = T::one() + T::one();
        let three = two + T::one();
        let four = two + two;

        let den = self.g * (four - two * self.g - self.h);

        let vx = (two * self.g.powi(2) + two * self.h - three * self.g * self.h) / den;
        let vdx = two * self.h.powi(2) / (self.dt.powi(2) * den);

        (vx, vdx)
    }

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
pub struct GHKFilter<T> {
    pub g: T,
    pub h: T,
    pub k: T,
    pub dt: T,
    pub xt: T,
    pub dxt: T,
    pub ddxt: T,
    pub x_p: T,
    pub dx_p: T,
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
}

pub fn optimal_noise_smoothing<T: Float>(g: T) -> (T, T, T) {
    let one = T::one();
    let two = T::from(2).unwrap();
    let four = T::from(4).unwrap();
    let eight = T::from(8).unwrap();
    let sixty_four = T::from(64).unwrap();


    let h = ((two * g.powi(3) - four * g.powi(2)) + (four * g.powi(6) - sixty_four * g.powi(5)
        + sixty_four * g.powi(4)).sqrt()) / (eight * (one - g));
    let k = (h * (two - g) - g.powi(2)) / g;

    (g, h, k)
}

pub fn least_squares_parameters<T: FloatCore>(n: T) -> (T, T) {
    let one = T::one();
    let two = T::from(2).unwrap();
    let six = T::from(6).unwrap();

    let den = (n + two) * (n + one);

    let g = (two * (two * n + one)) / den;
    let h = six / den;
    (g, h)
}

pub fn critical_damping_parameters_order_two<T: FloatCore>(theta: T) -> (T, T) {
    let one = T::one();

    (one - theta.powi(2), (one - theta).powi(2))
}

pub fn critical_damping_parameters_order_three<T: FloatCore>(theta: T) -> (T, T, T) {
    let one = T::one();
    let two = T::from(2).unwrap();
    let three = T::from(2).unwrap();

    (one - theta.powi(3), (three / two) * (one - theta.powi(2)) * (one - theta), (one / two) * (one - theta).powi(3))
}

pub fn benedict_bornder_constants<T: Float>(g: T, critical: bool) -> (T, T) {
    let g_sqr = g.powi(2);
    if critical {
        (g, T::from(0.8).unwrap() * (T::from(2).unwrap() - g_sqr - T::from(2).unwrap() * (T::one() - g_sqr).sqrt()) / g_sqr)
    } else {
        (g, g_sqr / (T::from(2).unwrap() - g))
    }
}