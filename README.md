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
We will demonstrate the workings of the GH filter by running through a toy problem from R Labbe's book.
We want to track our weight on a daily basis:

```
// Start at 160 pounds, assuming to gain a pound every day
let x0 = 160.;
let dx0 = 1.;

// Set the filter parameters
let g = 0.6;
let h = 2./3.;
let dt = 1.;

// Apply the GH filter
let mut gh_filter = GHFilter::new(x0, dx0, g, h, dt);

let weights =
    [158.0, 164.2, 160.3, 159.9, 162.1, 164.6, 169.6, 167.4, 166.4, 171.0, 171.2, 172.6];

for w in &weights {
    println!("Filtered weight: {:.1}", gh_filter.update(*w));
}
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

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.