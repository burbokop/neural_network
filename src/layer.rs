use serde::{Serialize, Deserialize};

use crate::{Arr, Perceptron};



pub trait Layer<const INPUT_COUNT: usize, const COUNT: usize> {
    fn proceed(&mut self, input: &[f64; INPUT_COUNT], normalizator: fn(f64) -> f64) -> Arr<f64, COUNT>;
    fn backpropagate(&mut self, error: &[f64; COUNT], normalizer_derivative: fn(f64) -> f64, learning_rate: f64);
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct PLayerCtx<const INPUT_ACTIVATIONS_COUNT: usize, const OUTPUT_ACTIVATIONS_COUNT: usize> {
    input_activations: Arr<f64, INPUT_ACTIVATIONS_COUNT>,
    output_activations: Arr<f64, OUTPUT_ACTIVATIONS_COUNT>
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerceptronLayer<const INPUT_COUNT: usize, const COUNT: usize, const ACTIVATIONS_COUNT: usize, LL: Layer<INPUT_COUNT, ACTIVATIONS_COUNT>> {
    perceptrons: Arr<Perceptron<ACTIVATIONS_COUNT>, COUNT>,
    left_layer: Box<LL>,
    #[serde(skip_serializing)]
    prev_ctx: Option<PLayerCtx<ACTIVATIONS_COUNT, COUNT>>
}

impl<const INPUT_COUNT: usize, const COUNT: usize, const ACTIVATIONS_COUNT: usize, LL: Layer<INPUT_COUNT, ACTIVATIONS_COUNT>> PerceptronLayer<INPUT_COUNT, COUNT, ACTIVATIONS_COUNT, LL> {
    pub fn new(left_layer: LL) -> Self {
        Self { perceptrons: Default::default(), left_layer: Box::new(left_layer), prev_ctx: None }
    }
}

impl<const INPUT_COUNT: usize, const COUNT: usize, const ACTIVATIONS_COUNT: usize, LL: Layer<INPUT_COUNT, ACTIVATIONS_COUNT>> Layer<INPUT_COUNT, COUNT> for PerceptronLayer<INPUT_COUNT, COUNT, ACTIVATIONS_COUNT, LL> {
    #[inline(always)]
    fn proceed(&mut self, input: &[f64; INPUT_COUNT], normalizator: fn(f64) -> f64) -> Arr<f64, COUNT> {
        let ctx = self.prev_ctx.get_or_insert_default();
        ctx.input_activations = self.left_layer.proceed(input, normalizator);
        let result = unsafe {
            let mut result: Arr<f64, COUNT> = Arr::uninitialized();
            for i in 0..COUNT {
                result[i] = self.perceptrons[i].proceed(&ctx.input_activations, normalizator)
            }
            result
        };
        ctx.output_activations = result;
        ctx.output_activations.clone()
    }

    // observed - predicted

    #[inline(always)]
    fn backpropagate(&mut self, errors: &[f64; COUNT], normalizer_derivative: fn(f64) -> f64, learning_rate: f64) {
        if let Some(ctx) = &self.prev_ctx {
            let next_errors = {


                let gradient = unsafe {
                    let mut gradient: Arr<f64, COUNT> = Arr::uninitialized();
                    for i in 0..COUNT {
                        gradient[i] = errors[i] * normalizer_derivative(ctx.output_activations[i]) * learning_rate;
                    }
                    gradient
                };


                //let gradient = ctx.output_activations.iter().zip(errors).map(|(a, e)| e * normalizer_derivative(*a) * learning_rate);

                let deltas = unsafe {
                    let mut deltas: Arr<[f64; ACTIVATIONS_COUNT], COUNT> = Arr::uninitialized();
                    for i in 0..COUNT {
                        for j in 0..ACTIVATIONS_COUNT {
                            deltas[i][j] = gradient[i] * ctx.input_activations[j];
                        }
                    }
                    deltas
                };

                //let deltas = gradient.clone().map(|g| {
                //    ctx.input_activations.map(|i| g * i)
                //}).collect::<Vec<[f64; ACTIVATIONS_COUNT]>>();

                let mut next_errors: [f64; ACTIVATIONS_COUNT] = [0.; ACTIVATIONS_COUNT];
                for i in 0..ACTIVATIONS_COUNT {
                    for j in 0..COUNT {
                        next_errors[i] += self.perceptrons[j].weights()[i] * errors[j];
                    }
                }
                
                //let next_errors: Arr<f64, ACTIVATIONS_COUNT> = self.perceptrons.iter().zip(errors.iter()).map(|(p, err)| {
                //    p.weights().map(|w| w + err)
                //}).into_iter().sum();

                //self.perceptrons.iter_mut().zip(deltas).zip(gradient).for_each(|((p, d), g)| {
                //    p.weights.iter_mut().zip(d).for_each(|(w, d)| *w += d);
                //    p.bias += g
                //});

                for i in 0..COUNT {
                    let perceptron = &mut self.perceptrons[i];
                    for j in 0..ACTIVATIONS_COUNT {
                        perceptron.weights_mut()[j] += deltas[i][j];
                    }
                    *perceptron.bias_mut() += gradient[i];
                }
                next_errors
            };
            self.left_layer.backpropagate(&next_errors, normalizer_derivative, learning_rate);
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InputLayer<const COUNT: usize> {}

impl<const COUNT: usize> InputLayer<COUNT> { 
    pub fn new() -> Self {
        Self {}
    }
}

impl<const COUNT: usize> Layer<COUNT, COUNT> for InputLayer<COUNT> {
    fn proceed(&mut self, input: &[f64; COUNT], _: fn(f64) -> f64) -> Arr<f64, COUNT> {
        input.into()
    }
    fn backpropagate(&mut self, _: &[f64; COUNT], _: fn(f64) -> f64, _: f64) {}
}


#[derive(Debug, Serialize, Deserialize)]
pub struct InputLayer2d<const W: usize, const H: usize> {}

impl<const W: usize, const H: usize> InputLayer2d<W, H> { 
    pub fn new() -> Self {
        Self {}
    }
}

impl<const W: usize, const H: usize> Layer<{W * H}, {W * H}> for InputLayer2d<W, H> {
    fn proceed(&mut self, input: &[f64; W * H], _: fn(f64) -> f64) -> Arr<f64, {W * H}> {
        input.into()
    }
    fn backpropagate(&mut self, _: &[f64; W * H], _: fn(f64) -> f64, _: f64) {}
}

