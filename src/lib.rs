#![feature(generic_associated_types)]
#![feature(option_get_or_insert_default)]
#![feature(generic_const_exprs)]

use std::f64::consts::E;

mod arr;
mod perceptron;
mod layer;
mod trainer;
mod loader;

pub use {
    arr::*,
    perceptron::*,
    layer::*,
    trainer::*,
    loader::*
};


pub mod mormalizers {


    pub fn sigmoid(x: f64) -> f64 {
        use std::f64::consts::E;
        1. / (1. + E.powf(-x))
    }

    pub fn sigmoid_derivative(y: f64) -> f64 { y * (1. - y) }

    pub fn exact(x: f64) -> f64 { x }

}

pub fn print_array<const SIZE: usize>(comment: &str, arr: &Arr<f64, SIZE>) {
    let begin_seq = |r: u8, g: u8, b: u8| {
        format!("\x1b[38;2;{};{};{}m", r, g, b)
    };
    
    let end_seq = || { "\x1b[0m" };

    print!("{}", comment);
    for num in arr.iter() {
        let g = (num * 255.) as u8;
        let r = ((1. - num) * 255.) as u8;
        let b = 0 as u8;

        print!("{}{}{:.5}{}, ", begin_seq(r, g, b), if !(*num < 0.) { "+" } else { "" }, num, end_seq())
        
    }
    println!("");
}

pub fn calc_cost<const SIZE: usize>(present: &Arr<f64, SIZE>, expected: &Arr<f64, SIZE>) -> f64 {
    present.iter().zip(expected.iter()).map(|(p, e)| (p - e).powi(2)).sum()
}

pub fn is_prediction_ok<const SIZE: usize>(present: &Arr<f64, SIZE>, expected: &Arr<f64, SIZE>) -> bool {
    let mut pressent_result: (f64, usize) = (f64::MIN, 0);
    let mut expected_result: (f64, usize) = (f64::MIN, 0);

    for i in 0..SIZE {
        if pressent_result.0 < present[i] {
            pressent_result.0 = present[i];
            pressent_result.1 = i;
        }
        if expected_result.0 < expected[i] {
            expected_result.0 = expected[i];
            expected_result.1 = i;
        }
    }
    pressent_result.1 == expected_result.1
}

pub fn log_e(v: f64) -> f64 {
    v.log(E)
}



///slice element: (observed, predicted)
pub fn slope<const SIZE: usize>(i: &[(Arr<f64, SIZE>, Arr<f64, SIZE>)]) -> Arr<f64, SIZE> {
    i.iter().map(|(observed, predicted)| Arr::from(-2.) * (observed.clone() - predicted.clone())).sum()
}

///slice element: (observed, predicted)
pub fn slope2<const SIZE: usize>(observed: Arr<f64, SIZE>, predicted: Arr<f64, SIZE>) -> f64 {
    (Arr::from(-1.) * observed * predicted.map_copy(log_e)).sum()
}


