use crate::{utilities::{Perceptron, Arr}, NeuralNet};


pub fn proceed(weights: &[f64], bias: f64, activations: &[f64], normalizer: fn(f64) -> f64) -> f64 {
    let mut sum = 0.;
    for i in 0..weights.len() {
        sum += activations[i] * weights[i];
    }
    normalizer(sum - bias)
}

struct PerceptronConnection {
    weights: Vec<f64>,
    bias: f64
}

struct Connection {
    ps: Vec<PerceptronConnection>
}

pub enum Direction {
    ToLeft,
    ToRight
}

pub struct FlatNeuralNet<const INPUT_COUNT: usize, const OUTPUT_COUNT: usize> {
    connections: Vec<Connection>,
    direction: Direction
}

impl<const INPUT_COUNT: usize, const OUTPUT_COUNT: usize> NeuralNet<INPUT_COUNT, OUTPUT_COUNT> for FlatNeuralNet<INPUT_COUNT, OUTPUT_COUNT> {
    fn proceed(&mut self, input: &[f64; INPUT_COUNT], normalizer: fn(f64) -> f64) -> Arr<f64, OUTPUT_COUNT> {

        Arr::default()

        //for i in 0..self.connections.len() {
        //    let connection = self.connections[i];
//
//
        //    for i in 0..connection.weights.len() {
        //        connection.weights
        //    }
        //}

        
    }

    fn backpropagate(&mut self, error: &[f64; OUTPUT_COUNT], normalizer_derivative: fn(f64) -> f64, learning_rate: f64) {
    }

    fn dump(&self) -> Vec<&[f64]> {
        Vec::new()
    }

    fn reverse_proceed(&mut self, input: &[f64; OUTPUT_COUNT], reverse_normalizer: fn(f64) -> f64) -> Arr<f64, INPUT_COUNT> {
        todo!()
    }
}