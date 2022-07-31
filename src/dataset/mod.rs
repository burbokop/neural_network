use crate::TrainPair;

mod mnist;
mod yolov5;
pub use mnist::*;
pub use yolov5::*;

pub trait Dataset<const ISIZE: usize, const OSIZE: usize> {
    fn training_data(self) -> Box<dyn Iterator<Item = TrainPair<ISIZE, OSIZE>>>;
    fn class_name(&self, num: usize) -> Option<String>;
}