use std::time::{Duration, Instant};

use crate::{Layer, calc_cost, is_prediction_ok, Arr};


use rand::Rng;


pub struct TrainPair<const ISIZE: usize, const OSIZE: usize> {
    pub input: Arr<f64, ISIZE>,
    pub output: Arr<f64, OSIZE>
}

impl<const ISIZE: usize, const OSIZE: usize> TrainPair<ISIZE, OSIZE> {
    pub fn train<'l, L: Layer<ISIZE, OSIZE>>(&self, layer: &'l mut L, normalizator: fn(f64) -> f64, normalizator_derivative: fn(f64) -> f64, learning_rate: f64) -> (f64, bool, Duration, Duration) {
        let instant = Instant::now();
        let predicted = layer.proceed(&self.input, normalizator);

        //println!("input_sum: {}", self.input.iter().sum::<f64>());


        //print_array("predicted: ", &r);
        //print_array("observed : ", &self.output);

        let proceed_diration = instant.elapsed();
        let error = self.output.clone() - predicted.clone();
        layer.backpropagate(&error, normalizator_derivative, learning_rate);
        (calc_cost(&predicted, &self.output), is_prediction_ok(&predicted, &self.output), proceed_diration, instant.elapsed())
    }
}

pub struct Trainer<'a, const ISIZE: usize, const OSIZE: usize> {
    set: &'a [TrainPair<ISIZE, OSIZE>],
    learning_rate: f64
}


impl<'a, const ISIZE: usize, const OSIZE: usize> Trainer<'a, ISIZE, OSIZE> {    
    pub fn new(set: &'a [TrainPair<ISIZE, OSIZE>], learning_rate: f64) -> Self {
        Self { set: set, learning_rate: learning_rate }
    }

    pub fn train_rand<'l, L: Layer<ISIZE, OSIZE>>(
        &self, 
        layer: &'l mut L, 
        normalizator: fn(f64) -> f64, 
        normalizator_derivative: fn(f64) -> f64,
        epoch: usize,
        batch_size: usize
    ) {
        if self.set.len() > 0 {
            let mut rng = rand::thread_rng();

            println!("training started");

            for i in 0..epoch {
                let mut cost_sum = 0.;
                let mut ok_count = 0;
                let mut duration_sum: Duration = Default::default();
                let instant = Instant::now();
                for _ in 0..batch_size {
                    let train_result = self.set[rng.gen_range(0..self.set.len())].train(layer, normalizator, normalizator_derivative, self.learning_rate);                    
                    cost_sum += train_result.0;
                    if train_result.1 {
                        ok_count += 1;
                    }
                    duration_sum += train_result.3;

                    //println!("\ttrain cost: {}, train time: {} ms (proceed time: {})", train_result.0, train_result.2.as_millis(), train_result.1.as_millis());
                }
                let avr_duration = duration_sum / (batch_size as u32);

                if cost_sum < 1. { break; }

                println!("epoch[{}] ok: {}, cost: {} (epoch time: {} ms, one train time: {} ms)", i, ok_count, cost_sum, instant.elapsed().as_millis(), avr_duration.as_millis());
            }
        }
    }
}
