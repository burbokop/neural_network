use std::path::Path;

use crate::{TrainPair, Arr};



pub fn load_img_normalized_grayscale(path: &Path) -> Vec<f64> {
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

