use serde::{Serialize, Deserialize};

use crate::NeuralNet;





mod layer;
mod dumper;
pub mod computedevice;

pub use {
    layer::*,
    dumper::*
};

#[derive(Debug, Serialize, Deserialize)]
pub struct RecNeuralNet<const INPUT_COUNT: usize, const OUTPUT_COUNT: usize, L>
where L: Layer<INPUT_COUNT, OUTPUT_COUNT> {
    pub end_layer: L,
    #[serde(skip)]
    pub device: BackpropagationComputeDevice
}

impl<const INPUT_COUNT: usize, const OUTPUT_COUNT: usize, L> NeuralNet<INPUT_COUNT, OUTPUT_COUNT> for RecNeuralNet<INPUT_COUNT, OUTPUT_COUNT, L>
where L: Layer<INPUT_COUNT, OUTPUT_COUNT> {
    fn proceed(&mut self, input: &[f64; INPUT_COUNT], normalizer: fn(f64) -> f64) -> crate::utilities::Arr<f64, OUTPUT_COUNT> {
        self.end_layer.proceed(input, normalizer)
    }

    fn backpropagate(&mut self, error: &[f64; OUTPUT_COUNT], normalizer_derivative: fn(f64) -> f64, learning_rate: f64) {
        self.end_layer.backpropagate(error, normalizer_derivative, learning_rate, &self.device)
    }

    fn dump(&self) -> Vec<&[f64]> {
        self.end_layer.dump()
    }

    fn reverse_proceed(&mut self, input: &[f64; OUTPUT_COUNT], reverse_normalizer: fn(f64) -> f64) -> crate::utilities::Arr<f64, INPUT_COUNT> {
        self.end_layer.reverse_proceed(input, reverse_normalizer)
    }
}