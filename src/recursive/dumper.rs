use std::path::{Path, PathBuf};

use image::{Rgba, Pixel, ImageBuffer};

use crate::NeuralNet;

use super::Layer;


pub struct ImageDumper {

}

pub fn grayscale_to_rgba(v: &f64) -> Rgba<u8> {
    let gray = (v * 255.) as u8;
    Rgba([gray, gray, gray, 255])
}

pub fn grayscale_slice_to_image(slice: &[f64]) -> Option<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    let squere = (slice.len() as f64).sqrt() as u32;
    let buffer = slice.iter().map(grayscale_to_rgba).map(|p| p.0).collect::<Vec<[u8; 4]>>().concat();
    ImageBuffer::from_vec(squere, squere, buffer)
}

pub fn grayscale_to_image<const SIZE: usize>(arr: &[f64; SIZE]) -> Option<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    grayscale_slice_to_image(arr)
}

pub fn save_grayscale<const SIZE: usize>(path: &Path, arr: &[f64; SIZE]) {
    if let Some(image) = grayscale_to_image(arr) {
        if let Some(parent) = path.parent() { std::fs::create_dir_all(parent).unwrap() }
        image.save_with_format(path, image::ImageFormat::Png).unwrap();
    }
}

pub fn save_grayscale_slice(path: &Path, slice: &[f64]) {
    if let Some(image) = grayscale_slice_to_image(slice) {
        if let Some(parent) = path.parent() { std::fs::create_dir_all(parent).unwrap() }
        image.save_with_format(path, image::ImageFormat::Png).unwrap();
    }
}



impl ImageDumper {
    pub fn dump<const INPUT_COUNT: usize, const COUNT: usize, NN: NeuralNet<INPUT_COUNT, COUNT>>(leyer: &NN) {
        let mut i = 0;
        for l in leyer.dump() {
            save_grayscale_slice(PathBuf::from(format!("./dump/layer{:0>4}.png", i)).as_path(), l);
            i += 1
        }
    }
}
