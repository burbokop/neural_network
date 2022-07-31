use std::{vec, path::{PathBuf, Path}};

use serde::{Serialize, Deserialize};

use crate::{TrainPair, utilities::{load_img_normalized_grayscale, Arr}};

use super::Dataset;

#[derive(Debug, Serialize, Deserialize)]
struct Yaml {
    pub train: String,
    pub val: String,
    pub nc: usize,
    pub names: Vec<String>
}

struct Label {
    pub class: usize,
    pub center_x: f64,
    pub center_y: f64,
    pub w: f64,
    pub h: f64,
}

impl Label {
    pub fn from_str(s: String) -> Result<Label, String> {
        let v = s.split(" ").collect::<Vec<_>>();
        if v.len() == 5 {
            let class = v[0].parse::<usize>().map_err(|_|s.clone())?;
            let x = v[1].parse::<f64>().map_err(|_|s.clone())?;
            let y = v[2].parse::<f64>().map_err(|_|s.clone())?;
            let w = v[3].parse::<f64>().map_err(|_|s.clone())?;
            let h = v[4].parse::<f64>().map_err(|_|s)?;
            Ok(Label { class: class, center_x: x, center_y: y, w: w, h: h })
        } else {
            Err(s)
        }
    }
    pub fn out_arr<const SIZE: usize>(&self) -> Arr<f64, SIZE> {
        let mut result = Arr::default();
        result[self.class] = 1.;
        result
    }
}

pub struct YOLOv5Dataset<const ISIZE: usize, const OSIZE: usize> {
    data: Box<dyn Iterator<Item = TrainPair<ISIZE, OSIZE>>>,
    classes: Vec<String>
}

impl<const ISIZE: usize, const OSIZE: usize> YOLOv5Dataset<ISIZE, OSIZE> {
        pub fn load(
        yaml_path: &Path
    ) -> Self {
        let yaml: Yaml = serde_yaml::from_str(std::fs::read_to_string(yaml_path).unwrap().as_str()).unwrap();

        //println!("yaml: {:?}", yaml);

        let dir = yaml_path.parent().unwrap();
        let train_dir = dir.join(yaml.train);
        let labels_dir = train_dir.join("../labels");

        Self { 
            data: Box::new(std::fs::read_dir(train_dir)
                .unwrap()
                .zip(std::fs::read_dir(labels_dir).unwrap())
                .filter_map(|(i, l)| {            
                
                    let image_path = i.unwrap().path();
                    let label_path = l.unwrap().path();

                    //println!("loading training pair: {:?} -> {:?}", image_path, label_path);

                    let arr: Arr<f64, ISIZE> = load_img_normalized_grayscale(image_path.as_path()).try_into().unwrap();
                    let label = Label::from_str(std::fs::read_to_string(label_path).unwrap());

                    if let Ok(label) = label {
                        if label.center_x == 0.5 && label.center_y == 0.5 && label.w == 1. && label.h == 1. {
                            let observed: Arr<f64, OSIZE> = label.out_arr();
                            Some(TrainPair { input: arr, output: observed })
                        } else { None }
                    } else { None }
                })), 
            classes: yaml.names 
        }
    }
}


impl<const ISIZE: usize, const OSIZE: usize> Dataset<ISIZE, OSIZE> for YOLOv5Dataset<ISIZE, OSIZE> {

    fn class_name(&self, num: usize) -> Option<String> {
        self.classes.get(num).cloned()
    }

    fn training_data(self) -> Box<dyn Iterator<Item = TrainPair<ISIZE, OSIZE>>> {
        self.data
    }
}


