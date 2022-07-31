use std::{collections::BTreeMap, env::consts::OS};

use crate::{TrainPair, utilities::{load_img_normalized_grayscale, Arr}};

use super::Dataset;


/// https://github.com/pjreddie/mnist-csv-png
pub struct MnistDataset<const ISIZE: usize, const OSIZE: usize> {
    data: Box<dyn Iterator<Item = TrainPair<ISIZE, OSIZE>>>
}

impl<const ISIZE: usize, const OSIZE: usize> MnistDataset<ISIZE, OSIZE> {
    pub fn load(
        path: &str
    ) -> Self {
        Self { 
            data: Box::new(std::fs::read_dir(path).unwrap().map(|file| {
                let entry = file.unwrap();
                let vec = load_img_normalized_grayscale(entry.path().as_path());
                let arr: Arr<f64, ISIZE> = vec.try_into().unwrap();
                assert!(entry.file_name().to_str().unwrap().len() > 10);
                let digit_cc = entry.file_name().to_str().unwrap().chars().nth(10).unwrap();
    
            //println!("path: {:?}, digit_cc: {}", entry.file_name(), digit_cc);
    
                let digit = digit_cc.to_digit(10).unwrap() as usize;
                
                let mut train_output: Arr<f64, OSIZE> = (0.).into();
                train_output[digit] = 1.;
        
                TrainPair { input: arr, output: train_output }
            }))
        }    
    }
}

impl<const ISIZE: usize, const OSIZE: usize> Dataset<ISIZE, OSIZE> for MnistDataset<ISIZE, OSIZE> {
    fn training_data(self) -> Box<dyn Iterator<Item = TrainPair<ISIZE, OSIZE>>> {
        self.data
    }

    fn class_name(&self, num: usize) -> Option<String> {
        if num < 10 {
            Some(format!("{}", num))
        } else {
            None
        }
    }
}

