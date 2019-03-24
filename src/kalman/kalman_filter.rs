/*!
This module implements the linear Kalman filter
*/

use nalgebra::{DefaultAllocator, DMatrix, DVector, Matrix, MatrixMN, Real, Scalar, U1, Vector, VectorN};
use nalgebra::allocator::Allocator;
use nalgebra::base::dimension::{Dim, DimName};

#[allow(non_snake_case)]
struct KalmanFilter<F>
    where
        F: Scalar,
{
    /// Current state estimate.
    x: DVector<F>,
    /// Current state covariance matrix.
    P: DMatrix<F>,
    /// Prior (predicted) state estimate.
    x_prior: DVector<F>,
    /// Prior (predicted) state covariance matrix.
    P_prior: DMatrix<F>,
    /// Posterior (updated) state estimate.
    x_post: DVector<F>,
    ///Posterior (updated) state covariance matrix.
    P_post: DMatrix<F>,
    /// Last measurement
    z: Option<DVector<F>>,
    /// Measurement noise matrix.
    R: DMatrix<F>,
    /// Process noise matrix.
    Q: DMatrix<F>,
    /// Control transition matrix
    B: Option<DMatrix<F>>,
    /// State Transition matrix.
    F: DMatrix<F>,
    /// Measurement function.
    H: DMatrix<F>,
    /// Residual of the update step.
    y: DVector<F>,
    /// Kalman gain of the update step.
    K: DMatrix<F>,
    /// System uncertainty (P projected to measurement space).
    S: DMatrix<F>,
    /// Inverse system uncertainty.
    SI: DMatrix<F>,
    //    /// log-likelihood of the last measurement.
    //    log_likelihood: F,
    //    ///  likelihood of last measurement.
    //    likelihood: F,
    //    /// mahalanobis distance of the innovation.
    //    mahalanobis: F,
    /// Fading memory setting.
    alpha_sq: F,

    dim_x: usize,
    dim_z: usize,
}

#[allow(non_snake_case)]
impl<F> KalmanFilter<F>
    where
        F: Real,
{
    pub fn new(dim_x: usize, dim_z: usize) -> Self {
        let x = DVector::from_element(dim_x, F::one());
        let P: DMatrix<F> = DMatrix::identity(dim_x, dim_x);
        let Q: DMatrix<F> = DMatrix::identity(dim_x, dim_x);
        let F: DMatrix<F> = DMatrix::identity(dim_x, dim_x);
        let H: DMatrix<F> = DMatrix::from_element(dim_z, dim_z, F::zero());
        let R: DMatrix<F> = DMatrix::identity(dim_z, dim_z);
        let alpha_sq = F::one();
        //        let M: DMatrix<F> = DMatrix::from_element(dim_z, dim_z, F::zero());

        let z = None;

        let K: DMatrix<F> = DMatrix::from_element(dim_x, dim_z, F::zero());
        let y = DVector::from_element(dim_z, F::one());
        let S = DMatrix::from_element(dim_z, dim_z, F::zero());
        let SI = DMatrix::from_element(dim_z, dim_z, F::zero());

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
            dim_x,
            dim_z,
        }
    }

    pub fn predict(
        &mut self,
        u: Option<&DVector<F>>,
        B: Option<&DMatrix<F>>,
        F: Option<&DMatrix<F>>,
        Q: Option<&DMatrix<F>>,
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

    pub fn update(&mut self, z: &DVector<F>, R: Option<&DMatrix<F>>, H: Option<&DMatrix<F>>) {
        let R = R.unwrap_or(&self.R);
        let H = H.unwrap_or(&self.H);

        self.y = z - H * self.x.clone();

        let PHT = self.P.clone() * H.transpose();
        self.S = H * PHT.clone() + R;

        self.SI = self.S.clone().try_inverse().unwrap();

        self.K = PHT * self.SI.clone();

        self.x = self.x.clone() + self.K.clone() * self.y.clone();

        let I_KH = DMatrix::identity(self.dim_x, self.dim_x) - self.K.clone() * H;
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

    #[test]
    fn test_univariate_kf_setup() {
        let mut kf = KalmanFilter::new(1, 1);

        for i in 0..1000 {
            let zf = i as f32;
            let z: DVector<f32> = DVector::from_vec(vec![zf]);
            kf.predict(None, None, None, None);
            kf.update(&z, None, None);
            assert_approx_eq!(zf, kf.z.clone().unwrap()[0]);
        }
    }
}
