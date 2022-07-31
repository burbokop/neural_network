use std::time::{Duration, Instant};


use rand::Rng;

use crate::{utilities::Arr, is_prediction_ok, calc_cost, NeuralNet};


pub struct TrainPair<const ISIZE: usize, const OSIZE: usize> {
    pub input: Arr<f64, ISIZE>,
    pub output: Arr<f64, OSIZE>
}

pub struct TrainResult {
    pub cost: f64,
    pub ok: bool,
    pub proceed_duration: Duration, 
    pub entire_duration: Duration
}

impl<const ISIZE: usize, const OSIZE: usize> TrainPair<ISIZE, OSIZE> {
    pub fn train<NN: NeuralNet<ISIZE, OSIZE>>(&self, neural_net: & mut NN, normalizator: fn(f64) -> f64, normalizator_derivative: fn(f64) -> f64, learning_rate: f64) -> TrainResult {
        let instant = Instant::now();
        let predicted = neural_net.proceed(&self.input, normalizator);

        let proceed_diration = instant.elapsed();
        let error = self.output.clone() - predicted.clone();
        neural_net.backpropagate(&error, normalizator_derivative, learning_rate);
        TrainResult {
            cost: calc_cost(&predicted, &self.output), 
            ok: is_prediction_ok(&predicted, &self.output), 
            proceed_duration: proceed_diration, 
            entire_duration: instant.elapsed()
        }
    }
}

pub struct Trainer<const ISIZE: usize, const OSIZE: usize> {
    pub learning_rate: f64,
    pub stop_cost: f64,
    pub stop_ok_coef: f64,
    pub normalizer: fn(f64) -> f64, 
    pub normalizer_derivative: fn(f64) -> f64,
    pub epoch: usize,
    pub batch_size: usize
}


impl<const ISIZE: usize, const OSIZE: usize> Trainer<ISIZE, OSIZE> {
    pub fn train_rand<NN: NeuralNet<ISIZE, OSIZE>>(
        &self, 
        mut set: Box<dyn Iterator<Item = TrainPair<ISIZE, OSIZE>>>,
        neural_net: &mut NN
    ) {
            ///let mut rng = rand::thread_rng();

            println!("training started");

            for i in 0..self.epoch {
                let mut cost_sum = 0.;
                let mut ok_count = 0;
                let mut duration_sum: Duration = Default::default();
                let instant = Instant::now();
                for _ in 0..self.batch_size {
                    if let Some(train_pair) = set.next() {

                        //let train_result = self.set[rng.gen_range(0..self.set.len())]
                        let train_result = train_pair.train(
                                neural_net, 
                                self.normalizer, 
                                self.normalizer_derivative, 
                                self.learning_rate
                            );                    
                        cost_sum += train_result.cost;
                        if train_result.ok {
                            ok_count += 1;
                        }
                        duration_sum += train_result.entire_duration
                    } else {
                        break;
                    }

                    //println!("\ttrain cost: {}, train time: {} ms (proceed time: {})", train_result.0, train_result.2.as_millis(), train_result.1.as_millis());
                }
                let avr_duration = duration_sum / (self.batch_size as u32);
                let ok_coef = (ok_count as f64) / (self.batch_size as f64);

                println!("epoch[{}] ok: {}/{}, cost: {} (epoch time: {} ms, one train time: {} ms)", i, ok_count, self.batch_size, cost_sum, instant.elapsed().as_millis(), avr_duration.as_millis());
                if cost_sum < self.stop_cost { break; }
                if ok_coef > self.stop_ok_coef { break; }
            }
        
    }
}
