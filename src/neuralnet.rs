use crate::utilities::Arr;




pub trait NeuralNet<const INPUT_COUNT: usize, const OUTPUT_COUNT: usize> {
    fn proceed(&mut self, input: &[f64; INPUT_COUNT], normalizer: fn(f64) -> f64) -> Arr<f64, OUTPUT_COUNT>;
    fn reverse_proceed(&mut self, input: &[f64; OUTPUT_COUNT], reverse_normalizer: fn(f64) -> f64) -> Arr<f64, INPUT_COUNT>;
    fn backpropagate(&mut self, error: &[f64; OUTPUT_COUNT], normalizer_derivative: fn(f64) -> f64, learning_rate: f64);
    fn dump(&self) -> Vec<&[f64]>;
}