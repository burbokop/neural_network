use std::sync::Arc;

use rand::thread_rng;
use serde::{Serialize, Deserialize};

use crate::{utilities::{Perceptron, Arr}, NeuralNet, recursive::computedevice::ComputeDevice};
use bytemuck::{Pod, Zeroable};
use vulkano::{buffer::BufferContents, device::Device, shader::ShaderModule};

use self::connection::{Unit, LayerCount, LayerSize, NetConnection};


mod computedevice;
pub mod connection;

pub fn proceed(weights: &[f64], bias: f64, activations: &[f64], normalizer: fn(f64) -> f64) -> f64 {
    let mut sum = 0.;
    for i in 0..weights.len() {
        sum += activations[i] * weights[i];
    }
    normalizer(sum - bias)
}


pub enum Direction {
    ToLeft,
    ToRight
}

    fn shader_factory() -> impl (FnOnce(Arc<Device>) -> Arc<ShaderModule>) {
        |device|{
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
        }
    }


pub struct FlatNeuralNet<const INPUT_COUNT: usize, const OUTPUT_COUNT: usize> {
    device: computedevice::ComputeDevice<Unit, LayerSize, LayerCount>
}

impl<const INPUT_COUNT: usize, const OUTPUT_COUNT: usize> FlatNeuralNet<INPUT_COUNT, OUTPUT_COUNT> {
    fn new(hidden_layers: &[u32]) -> Self {
        let mut device = computedevice::ComputeDevice::new(shader_factory()).unwrap();
        
        let gr = {
            let connection = NetConnection::rand(&mut thread_rng(), &[
                &[INPUT_COUNT as u32], 
                hidden_layers, 
                &[OUTPUT_COUNT as u32]
            ].concat());

            connection.into_graphics_representation().unwrap()
        };

        device.load(gr.0, gr.1, gr.2).unwrap();

        Self { device: device }
    }
}

impl<const INPUT_COUNT: usize, const OUTPUT_COUNT: usize> Serialize for FlatNeuralNet<INPUT_COUNT, OUTPUT_COUNT> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer {
        todo!()
    }
}

impl<'de, const INPUT_COUNT: usize, const OUTPUT_COUNT: usize> Deserialize<'de> for FlatNeuralNet<INPUT_COUNT, OUTPUT_COUNT> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de> {
        todo!()
    }
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