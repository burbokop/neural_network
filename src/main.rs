#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]

use std::{path::{PathBuf}};



use clap::Parser;
use pure_deepdream::{Layer, load_png_normalized_grayscale, mormalizers, print_array, Arr, Trainer, load_train_data, PerceptronLayer, InputLayer2d};
use serde::{Serialize, Deserialize};


#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Test {
    #[clap(short, long)]
    image: String,
}

static MODEL_PATH: &str = "ex.json";

impl Test {
    fn exec(self) {
        let nn_json = std::fs::read_to_string(MODEL_PATH).unwrap();

        let mut nn = Box::new(new_nn2());
        nn = serde_json::from_str(nn_json.as_str()).unwrap();

        let input = load_png_normalized_grayscale(PathBuf::from(self.image).as_path());

        let result = nn.proceed(&input.try_into().unwrap(), mormalizers::sigmoid);

        let nums: Arr<i32, 10> = (0..=9).collect::<Vec<_>>().try_into().expect("wrong size iterator");

        print_array("predicted: ", &result);
        print_array("nums     : ", &nums.map_copy(|a| a as f64));
    }
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Train {
    #[clap(short, long)]
    dataset: String,
}

impl Train {
    fn exec(self) {
        let mut nn = Box::new(new_nn2());

        let train_data = load_train_data::<{28 * 28}, 10>(&self.dataset, 1000000000);
        let trainer = Trainer::new(&train_data, 0.0002);
    
        trainer.train_rand(nn.as_mut(), mormalizers::sigmoid, mormalizers::sigmoid_derivative, 1000, 100);
    
        let json = serde_json::to_string_pretty(&nn);
        std::fs::write(MODEL_PATH, json.unwrap()).unwrap();    
    }
}


#[derive(Parser)]
enum SubCommand {
    Test(Test),
    Train(Train)
}

fn new_nn<'de>() -> impl Layer<784, 10> + Serialize + Deserialize<'de> {
    PerceptronLayer::<_, 10, _, _>::new(
        PerceptronLayer::<_, 1024, _, _>::new(
        PerceptronLayer::<_, 1024, _, _>::new(
        InputLayer2d::<28, 28>::new()
        )))
}

fn new_nn2<'de>() -> impl Layer<784, 10> + Serialize + Deserialize<'de> {
    PerceptronLayer::<_, 10, _, _>::new(
        PerceptronLayer::<_, 32, _, _>::new(
        PerceptronLayer::<_, 128, _, _>::new(
        PerceptronLayer::<_, 512, _, _>::new(
                InputLayer2d::<28, 28>::new()
        ))))
}


fn main() {

    match SubCommand::parse() {
        SubCommand::Test(test) => test.exec(),
        SubCommand::Train(train) => train.exec()
    }




    //println!("structure: {:?}", endlayer);




}

// train banchmark history
// 100 ms