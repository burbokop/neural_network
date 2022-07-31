use std::{ops::Deref, time::Instant};

use bytemuck::{Pod, Zeroable};
use serde::{Serialize, Deserialize};

use crate::utilities::{Arr, Perceptron};
use rayon::prelude::*;
use super::computedevice::ComputeDevice;

pub trait Layer<const INPUT_COUNT: usize, const COUNT: usize> {
    fn proceed(&mut self, input: &[f64; INPUT_COUNT], normalizer: fn(f64) -> f64) -> Arr<f64, COUNT>;
    fn reverse_proceed(&mut self, input: &[f64; COUNT], reverse_normalizer: fn(f64) -> f64) -> Arr<f64, INPUT_COUNT>;

    fn backpropagate(&mut self, error: &[f64; COUNT], normalizer_derivative: fn(f64) -> f64, learning_rate: f64, device: &BackpropagationComputeDevice);

    fn dump(&self) -> Vec<&[f64]>;
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct PLayerCtx<const INPUT_ACTIVATIONS_COUNT: usize, const OUTPUT_ACTIVATIONS_COUNT: usize> {
    input_activations: Arr<f64, INPUT_ACTIVATIONS_COUNT>,
    output_activations: Arr<f64, OUTPUT_ACTIVATIONS_COUNT>
}



#[derive(Debug)]
pub struct BackpropagationComputeDevice (ComputeDevice<2>);

impl Default for BackpropagationComputeDevice {
    fn default() -> Self {
        Self(ComputeDevice::new(|device|{
            mod cs {
                vulkano_shaders::shader! {
                    ty: "compute",
                    src: "                    
                        #version 450

                        struct QQQ {
                            float err;
                            float oa_to_gradient;
                            float weight_to_delta;
                        };

                        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                        layout(set = 0, binding = 0) buffer Dims {
                            uint dims[];
                        } dims;

                        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                        layout(set = 0, binding = 1) buffer Data {
                            QQQ data[];
                        } data;

                        float normalizer_derivative(float y) {
                            return y * (1. - y);
                        }

                        void main() {
                            float learning_rate = 0.001;

                            uint count = dims.dims[0];
                            uint activation_count = dims.dims[1];

                            uint idx = gl_GlobalInvocationID.x;

                            uint perceptron_index = idx / activation_count;
                            uint activation_index = idx % activation_count;
                                       
                            float oa = data.data[idx].oa_to_gradient;
                            float weight = data.data[idx].weight_to_delta;

                            data.data[idx].oa_to_gradient = data.data[idx].err * normalizer_derivative(oa) * learning_rate;
                            data.data[idx].weight_to_delta = data.data[idx].oa_to_gradient * oa;
                            data.data[idx].err *= weight;
    
                            
                        }
                    "
                }
            }
            cs::load(device.clone()).unwrap()
        }))
    }
}


#[derive(Debug, Serialize, Deserialize)]
pub struct PerceptronLayer<const INPUT_COUNT: usize, const COUNT: usize, const ACTIVATIONS_COUNT: usize, LL: Layer<INPUT_COUNT, ACTIVATIONS_COUNT>> {
    perceptrons: Arr<Perceptron<ACTIVATIONS_COUNT>, COUNT>,
    left_layer: Box<LL>,
    #[serde(skip)]
    prev_ctx: Option<PLayerCtx<ACTIVATIONS_COUNT, COUNT>>,
}

impl<const INPUT_COUNT: usize, const COUNT: usize, const ACTIVATIONS_COUNT: usize, LL: Layer<INPUT_COUNT, ACTIVATIONS_COUNT>> PerceptronLayer<INPUT_COUNT, COUNT, ACTIVATIONS_COUNT, LL> {
    pub fn new(left_layer: LL) -> Self {
        Self { perceptrons: Default::default(), left_layer: Box::new(left_layer), prev_ctx: None }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, Pod, Zeroable)]
struct QQQ {
    pub err: f32,
    pub oa_to_gradient: f32,
    pub weight_to_delta: f32,
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

    fn reverse_proceed(&mut self, input: &[f64; COUNT], reverse_normalizer: fn(f64) -> f64) -> Arr<f64, INPUT_COUNT> {
        let mut result: Arr<f64, ACTIVATIONS_COUNT> = Arr::default();
        for i in 0..self.perceptrons.len() {
            for j in 0..input.len() {
                result[i] += self.perceptrons[j].weights()[i] * (reverse_normalizer(input[j]) + self.perceptrons[j].bias())
            }   
        }

        self.left_layer.reverse_proceed(&result, reverse_normalizer) 
    }


    // observed - predicted

    #[inline(always)]
    fn backpropagate(&mut self, errors: &[f64; COUNT], normalizer_derivative: fn(f64) -> f64, learning_rate: f64, device: &BackpropagationComputeDevice) {
        if let Some(ctx) = &self.prev_ctx {
            let next_errors = {

                // cpu 8 threads
                // ----------------
                // l3 160.445µs
                // l2 338.86µs
                // l1 1.981351ms
                // l0 394.189721ms

                
                let instant_copy = Instant::now();

                let mut qs: Vec<QQQ> = Vec::with_capacity(COUNT * ACTIVATIONS_COUNT);
                for i in 0..COUNT {
                    let error = errors[i];
                    let out_act = ctx.output_activations[i];
                    let perceptron = &self.perceptrons[i];
                    for j in 0..ACTIVATIONS_COUNT {
                        qs.push(QQQ { 
                            err: error as f32,
                            oa_to_gradient: out_act as f32,
                            weight_to_delta: perceptron.weights()[j] as f32,
                        })
                    }
                }
                let instant = Instant::now();

                println!("len: {}, bytes: {}", qs.len(), qs.len() * std::mem::size_of::<QQQ>());

                let r = device.0.compute([COUNT as u32, ACTIVATIONS_COUNT as u32], qs, |r| r.to_vec());

                println!("gradient calculus time: {:?}", instant.elapsed());


                let (gradient, deltas, next_errors) = unsafe {
                    let mut gradient: Arr<f64, COUNT> = Arr::uninitialized();
                    let mut deltas: Arr<[f64; ACTIVATIONS_COUNT], COUNT> = Arr::uninitialized();
                    let mut next_errors: Arr<f64, ACTIVATIONS_COUNT> = Arr::uninitialized();

                    for i in 0..COUNT {
                        for j in 0..ACTIVATIONS_COUNT {
                            let idx = i * ACTIVATIONS_COUNT + j;
                            let q = &r[idx];

                            gradient[i] = q.oa_to_gradient as f64;
                            deltas[i][j] = q.weight_to_delta as f64;
                            next_errors[j] += q.err as f64;
                        }    
                    }
                    (gradient, deltas, next_errors)
                };

                println!("gradient calculus time (with copy): {:?}", instant_copy.elapsed());


                //for i in 0..COUNT {
                //    let error = errors[i];
                //    let out_act = ctx.output_activations[i];
                //    let perceptron = &self.perceptrons[i];
//
//
                //    let gradient = error * normalizer_derivative(out_act) * learning_rate;
                //    unsafe {
                //        let mut deltas : Arr<f64, ACTIVATIONS_COUNT> = Arr::uninitialized();
                //        for j in 0..ACTIVATIONS_COUNT {
                //            deltas[j] = gradient * ctx.input_activations[j];
                //        }
//
                //        let mut next_errors: Arr<f64, ACTIVATIONS_COUNT> = Arr::uninitialized();
                //        for k in 0..ACTIVATIONS_COUNT {
                //            next_errors[k] = perceptron.weights()[k] * error;
                //        }
                //        //(gradient, deltas, next_errors)
                //    }
                //}



                //let gradient = ctx.output_activations.iter().zip(errors).map(|(a, e)| e * normalizer_derivative(*a) * learning_rate);




                //let gradient_and_deltas_and_nex_errors: Arr<(f64, Arr<f64, ACTIVATIONS_COUNT>, Arr<f64, ACTIVATIONS_COUNT>), COUNT> = errors
                //    .par_iter()
                //    .zip(ctx.output_activations.deref())
                //    .zip(self.perceptrons.deref())
                //    .map(|((error, out_act), perceptron)| {
                //        let gradient = error * normalizer_derivative(*out_act) * learning_rate;
                //        unsafe {
                //            let mut deltas : Arr<f64, ACTIVATIONS_COUNT> = Arr::uninitialized();
                //            for j in 0..ACTIVATIONS_COUNT {
                //                deltas[j] = gradient * ctx.input_activations[j];
                //            }
//
                //            let mut next_errors: Arr<f64, ACTIVATIONS_COUNT> = Arr::uninitialized();
                //            for k in 0..ACTIVATIONS_COUNT {
                //                next_errors[k] = perceptron.weights()[k] * error;
                //            }
                //            (gradient, deltas, next_errors)
                //        }
                //    }).collect::<Vec<_>>().try_into().unwrap();


                //let (gradient, deltas) = unsafe {
//
                //    let mut gradient: Arr<f64, COUNT> = Arr::uninitialized();
                //    let mut deltas: Arr<[f64; ACTIVATIONS_COUNT], COUNT> = Arr::uninitialized();
    //
                //    for i in 0..COUNT {
                //        gradient[i] = errors[i] * normalizer_derivative(ctx.output_activations[i]) * learning_rate;
                //        for j in 0..ACTIVATIONS_COUNT {
                //            deltas[i][j] = gradient[i] * ctx.input_activations[j];
                //        }
                //    }
                //    (gradient, deltas)
                //};

                //let deltas = gradient.clone().map(|g| {
                //    ctx.input_activations.map(|i| g * i)
                //}).collect::<Vec<[f64; ACTIVATIONS_COUNT]>>();

                //let mut next_errors: [f64; ACTIVATIONS_COUNT] = [0.; ACTIVATIONS_COUNT];
                //for i in 0..COUNT {
                //    for k in 0..ACTIVATIONS_COUNT {
                //        next_errors[k] += self.perceptrons[i].weights()[k] * errors[i];
                //    }
                //}
                
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
            self.left_layer.backpropagate(&next_errors, normalizer_derivative, learning_rate, device);
        }
    }

    fn dump(&self) -> Vec<&[f64]> {
        [
            self.left_layer.dump(), 
            self.prev_ctx
                .as_ref()
                .map(|ctx| vec![&ctx.output_activations.deref()[..]])
                .unwrap_or(vec![])
        ].concat()
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
    fn backpropagate(&mut self, _: &[f64; COUNT], _: fn(f64) -> f64, _: f64, device: &BackpropagationComputeDevice) {}

    fn dump(&self) -> Vec<&[f64]> {
        vec![]
    }

    fn reverse_proceed(&mut self, input: &[f64; COUNT], reverse_normalizer: fn(f64) -> f64) -> Arr<f64, COUNT> {
        input.into()
    }
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
    fn backpropagate(&mut self, _: &[f64; W * H], _: fn(f64) -> f64, _: f64, device: &BackpropagationComputeDevice) {}

    fn dump(&self) -> Vec<&[f64]> {
        vec![]
    }

    fn reverse_proceed(&mut self, input: &[f64; W * H], reverse_normalizer: fn(f64) -> f64) -> Arr<f64, {W * H}> {
        input.into()
    }
}

