# filter-rs - Kalman filters and other optimal and non-optimal estimation filters in Rust.


**filter-rs** is a port of the [filterpy](https://github.com/rlabbe/filterpy) library and aims to provide Kalman filtering and optimal estimation for Rust.

This port is a work in progress and incomplete. To learn more about Kalman filters check out Roger R Labbe Jr.'s 
awesome book [Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python).

This library will grow as I work through the book myself and the API will most likely evolve and oxidize, too.
Feedback on the API design is always appreciated.

Examples
=========

The API is based on `nalgebra` matrices with structural genericity.

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
```
let mut kf: KalmanFilter<f32, U1, U1, U1> = KalmanFilter::new();

for i in 0..1000 {
    let zf = i as f32;
    let z = Vector1::from_vec(vec![zf]);
    kf.predict(None, None, None, None);
    kf.update(&z, None, None);
    assert_approx_eq!(zf, kf.z.clone().unwrap()[0]);
}
```


### Current state

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



