/*!
# filter-rs

**filter-rs** is a port of the [filterpy](https://github.com/rlabbe/filterpy) library.
This library is a work in progress. To learn more about Kalman filters check out Roger R Labbe Jr.'s
awesome book [Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)

*/
#![deny(missing_docs, missing_debug_implementations, missing_copy_implementations,
trivial_casts, trivial_numeric_casts, unsafe_code, unstable_features, unused_import_braces, unused_qualifications)]
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod common;
#[cfg(feature = "alloc")]
pub mod discrete_bayes;
pub mod gh;
pub mod kalman;
pub mod stats;
