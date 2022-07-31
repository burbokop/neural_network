
use super::Arr;


use serde::{Serialize, Deserialize};
use rand::Rng;

#[derive(Debug, Serialize, Deserialize)]
pub struct Perceptron<const SIZE: usize> {
    bias: f64,
    weights: Arr<f64, SIZE>,
}


impl<const SIZE: usize> Default for Perceptron<SIZE> {
    #[inline(always)]
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        Self { weights: Arr::rand(&mut rng, (-1.)..=(1.)), bias: rng.gen_range((-1.)..=(1.)) }
    }
}

impl<const SIZE: usize> Perceptron<SIZE> {
    #[inline(always)]
    pub fn bias(&self) -> &f64 { &self.bias }
    #[inline(always)]
    pub fn weights(&self) -> &Arr<f64, SIZE> { &self.weights }
    #[inline(always)]
    pub fn bias_mut(&mut self) -> &mut f64 { &mut self.bias }
    #[inline(always)]
    pub fn weights_mut(&mut self) -> &mut Arr<f64, SIZE> { &mut self.weights }
    #[inline(always)]
    pub fn proceed(&self, activations: &[f64; SIZE], normalizator: fn(f64) -> f64) -> f64 {
        let mut sum = 0.;
        for i in 0..SIZE {
            sum += activations[i] * self.weights[i];
        }
        normalizator(sum - self.bias)
    }
}
