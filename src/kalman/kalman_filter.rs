/*!
This module implements the linear Kalman filter
*/

use nalgebra::allocator::Allocator;
use nalgebra::base::dimension::DimName;
use nalgebra::{DMatrix, DefaultAllocator, MatrixMN, Real, VectorN};

#[allow(non_snake_case)]
struct KalmanFilter<F, DimX, DimZ, DimU>
where
    F: Real,
    DimX: DimName,
    DimZ: DimName,
    DimU: DimName,
    DefaultAllocator: Allocator<F, DimX>
        + Allocator<F, DimZ>
        + Allocator<F, DimX, DimZ>
        + Allocator<F, DimZ, DimX>
        + Allocator<F, DimZ, DimZ>
        + Allocator<F, DimX, DimX>
        + Allocator<F, DimU>
        + Allocator<F, DimX, DimU>,
{
    /// Current state estimate.
    x: VectorN<F, DimX>,
    /// Current state covariance matrix.
    P: MatrixMN<F, DimX, DimX>,
    /// Prior (predicted) state estimate.
    x_prior: VectorN<F, DimX>,
    /// Prior (predicted) state covariance matrix.
    P_prior: MatrixMN<F, DimX, DimX>,
    /// Posterior (updated) state estimate.
    x_post: VectorN<F, DimX>,
    ///Posterior (updated) state covariance matrix.
    P_post: MatrixMN<F, DimX, DimX>,
    /// Last measurement
    z: Option<VectorN<F, DimZ>>,
    /// Measurement noise matrix.
    R: MatrixMN<F, DimZ, DimZ>,
    /// MatrixMN<F, DimZ, DimZ>,
    Q: MatrixMN<F, DimX, DimX>,
    /// Control transition matrix
    B: Option<MatrixMN<F, DimX, DimU>>,
    /// State Transition matrix.
    F: MatrixMN<F, DimX, DimX>,
    /// Measurement function.
    H: MatrixMN<F, DimZ, DimX>,
    /// Residual of the update step.
    y: VectorN<F, DimZ>,
    /// Kalman gain of the update step.
    K: MatrixMN<F, DimX, DimZ>,
    /// System uncertainty (P projected to measurement space).
    S: MatrixMN<F, DimZ, DimZ>,
    /// Inverse system uncertainty.
    SI: MatrixMN<F, DimZ, DimZ>,
    //    /// log-likelihood of the last measurement.
    //    log_likelihood: F,
    //    ///  likelihood of last measurement.
    //    likelihood: F,
    //    /// mahalanobis distance of the innovation.
    //    mahalanobis: F,
    /// Fading memory setting.
    alpha_sq: F,
}

#[allow(non_snake_case)]
impl<F, DimX, DimZ, DimU> KalmanFilter<F, DimX, DimZ, DimU>
where
    F: Real,
    DimX: DimName,
    DimZ: DimName,
    DimU: DimName,
    DefaultAllocator: Allocator<F, DimX>
        + Allocator<F, DimZ>
        + Allocator<F, DimX, DimZ>
        + Allocator<F, DimZ, DimX>
        + Allocator<F, DimZ, DimZ>
        + Allocator<F, DimX, DimX>
        + Allocator<F, DimU>
        + Allocator<F, DimX, DimU>,
{
    pub fn new() -> Self {
        let x = VectorN::<F, DimX>::from_element(F::one());
        let P = MatrixMN::<F, DimX, DimX>::identity();
        let Q = MatrixMN::<F, DimX, DimX>::identity();
        let F = MatrixMN::<F, DimX, DimX>::identity();
        let H = MatrixMN::<F, DimZ, DimX>::from_element(F::zero());
        let R = MatrixMN::<F, DimZ, DimZ>::identity();
        let alpha_sq = F::one();

        let z = None;

        let K = MatrixMN::<F, DimX, DimZ>::from_element(F::zero());
        let y = VectorN::<F, DimZ>::from_element(F::one());
        let S = MatrixMN::<F, DimZ, DimZ>::from_element(F::zero());
        let SI = MatrixMN::<F, DimZ, DimZ>::from_element(F::zero());

        let x_prior = x.clone();
        let P_prior = P.clone();

        let x_post = x.clone();
        let P_post = P.clone();

        KalmanFilter {
            x,
            P,
            x_prior,
            P_prior,
            x_post,
            P_post,
            z,
            R,
            Q,
            B: None,
            F,
            H,
            y,
            K,
            S,
            SI,
            alpha_sq,
        }
    }

    pub fn predict(
        &mut self,
        u: Option<&VectorN<F, DimU>>,
        B: Option<&MatrixMN<F, DimX, DimU>>,
        F: Option<&MatrixMN<F, DimX, DimX>>,
        Q: Option<&MatrixMN<F, DimX, DimX>>,
    ) {
        let B = if B.is_some() { B } else { self.B.as_ref() };
        let F = F.unwrap_or(&self.F);
        let Q = Q.unwrap_or(&self.Q);

        if B.is_some() && u.is_some() {
            self.x = F * self.x.clone() + B.unwrap() * u.unwrap();
        } else {
            self.x = F * self.x.clone();
        }

        self.P = ((F * self.P.clone()) * F.transpose()) * self.alpha_sq + Q;

        self.x_prior = self.x.clone();
        self.P_prior = self.P.clone();
    }

    pub fn update(
        &mut self,
        z: &VectorN<F, DimZ>,
        R: Option<&MatrixMN<F, DimZ, DimZ>>,
        H: Option<&MatrixMN<F, DimZ, DimX>>,
    ) {
        let R = R.unwrap_or(&self.R);
        let H = H.unwrap_or(&self.H);

        self.y = z - H * self.x.clone();

        let PHT = self.P.clone() * H.transpose();
        self.S = H * PHT.clone() + R;

        self.SI = self.S.clone().try_inverse().unwrap();

        self.K = PHT * self.SI.clone();

        self.x = self.x.clone() + self.K.clone() * self.y.clone();

        let I_KH = DMatrix::identity(DimX::dim(), DimX::dim()) - self.K.clone() * H;
        self.P = ((I_KH.clone() * self.P.clone()) * I_KH.transpose())
            + ((self.K.clone() * R) * self.K.clone().transpose());

        //
        //        # save measurement and posterior state
        self.z = Some(z.clone());
        self.x_post = self.x.clone();
        self.P_post = self.P.clone();
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;
    use nalgebra::base::Vector1;
    use nalgebra::U1;

    #[test]
    fn test_univariate_kf_setup() {
        let mut kf: KalmanFilter<f32, U1, U1, U1> = KalmanFilter::new();

        for i in 0..1000 {
            let zf = i as f32;
            let z = Vector1::from_vec(vec![zf]);
            kf.predict(None, None, None, None);
            kf.update(&z, None, None);
            assert_approx_eq!(zf, kf.z.clone().unwrap()[0]);
        }
    }
}
