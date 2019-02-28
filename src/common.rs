use num_traits::Float;

pub(crate) enum ConvolutionMode {
    Full,
    Same,
    Valid,
}

pub(crate) fn convolve_in_place<F: Float>(a: &[F], v: &[F]) -> Vec<F> {
    unimplemented!()
}