use std::path::Path;

use crate::{TrainPair, Arr};



pub fn load_png_normalized_grayscale(path: &Path) -> Vec<f64> {
    image::io::Reader::open(path)
        .unwrap()
        .decode()
        .unwrap()
        .into_rgba8()
        .pixels()
        .map(|p| 
            p.0
                .map(|b| b as f64)
                .into_iter()
                .sum::<f64>() / 4. / 255.)
        .collect()
}

pub fn load_train_data<const ISIZE: usize, const OSIZE: usize>(path: &str, mut limit: usize) -> Vec<TrainPair<ISIZE, OSIZE>> {
    std::fs::read_dir(path).unwrap().map_while(|file| {
        let entry = file.unwrap();
        let vec = load_png_normalized_grayscale(entry.path().as_path());
        let arr: Arr<f64, ISIZE> = vec.try_into().unwrap();
        assert!(entry.file_name().to_str().unwrap().len() > 10);
        let digit_cc = entry.file_name().to_str().unwrap().chars().nth(10).unwrap();

        //println!("path: {:?}, digit_cc: {}", entry.file_name(), digit_cc);

        let digit = digit_cc.to_digit(10).unwrap() as usize;

        let mut train_output: Arr<f64, OSIZE> = (0.).into();
        train_output[digit] = 1.;


        if limit == 0 {
            None
        } else {
            limit -= 1;
            Some(TrainPair { input: arr, output: train_output })
        }
    }).collect()
}
