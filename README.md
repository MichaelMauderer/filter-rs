# filter-rs - Kalman filters and other optimal and non-optimal estimation filters in Rust.


**filter-rs** is a port of the [filterpy](https://github.com/rlabbe/filterpy) library and aims to provide Kalman filtering and optimal estimation for Rust.

This port is a work in progress and incomplete. To learn more about Kalman filters check out Roger R Labbe Jr.'s 
awesome book [Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python).

This library will grow as I work through the book myself and the API will most likely evolve and become more rustic, too.
Feedback on the API design is always appreciated, as well as pull requests for missing features. 

Examples
=========

The API is based on `nalgebra` matrices with structural genericity. That means, that the shapes of inputs can 
statically checked and are always correct at runtime.

GH Filter
----------
```
let x0 = 0.0;
let dx0 = 0.0;
let g = 0.2;
let h - 0.2;
let dt = 0.01;

let fgh = GHFilter::new(x0, dx0, g, h, dt);

```

Univariate Kalman Filter
-------------------------

The Kalman filter has to be initialised with sensible values. 
A default filter can be constructed but should not be used.  

```
let mut kf: KalmanFilter<f64, U2, U1, U1> = KalmanFilter::default();

kf.x = Vector2::new(2.0, 0.0);
kf.F = Matrix2::new(
    1.0, 1.0,
    0.0, 1.0,
);
kf.H = Vector2::new(1.0, 0.0).transpose();
kf.P *= 1000.0;
kf.R = Matrix1::new(5.0);
kf.Q = Matrix2::repeat(0.0001);

let mut results = Vec::default();
for t in 0..100 {
    let z = Vector1::new(t as f64);
    kf.update(&z, None, None);
    kf.predict(None, None, None, None);
    results.push(kf.x.clone());
}
```


### Current state

Tickboxes will be filled for each module that has feature parity with the filtyerpy library. 

* [ ] Linear Kalman Filter
* [ ] Fixed Lag Smoother
* [ ] Square Root Kalman Filter
* [ ] Information Filter
* [ ] Fading Kalman Filter
* [ ] MMAE Filter Bank
* [ ] IMM Estimator

* [ ] Extended Kalman Filter
* [ ] Unscented Kalman Filter
* [ ] Ensemble Kalman Filter

* [ ] Discrete Bayes

* [ ] GH-Filter
* [ ] GHK-Filter

* [ ] Fading Memory Filter

* [ ] H-Infinity Filter

* [ ] Least Squares Filter

License
=======

This project is licensed under the MIT License - see the LICENSE file for details



