#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]
#![feature(iterator_try_collect)]

use std::{path::{PathBuf}, ops::DerefMut, io::Write, time::Instant};



use bytemuck::{Pod, Zeroable};
use clap::Parser;
use pure_deepdream::{utilities::{load_img_normalized_grayscale, Arr}, recursive::{ImageDumper, PerceptronLayer, InputLayer2d, Layer, RecNeuralNet, save_grayscale, computedevice::ComputeDevice}, mormalizers, print_array, Trainer, NeuralNet, dataset::{YOLOv5Dataset, Dataset, MnistDataset}};
use serde::{Serialize, Deserialize};


fn current_nn<'de>() -> impl NeuralNet<249600, 10> + Serialize + Deserialize<'de> { new_ships_nn() }


#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Test {
    #[clap(short, long)]
    image: String,
    model_path: String
}

impl Test {
    fn exec(self) {
        let nn_json = std::fs::read_to_string(self.model_path).unwrap();

        let mut nn = Box::new(current_nn());
        nn = serde_json::from_str(nn_json.as_str()).unwrap();

        let input = load_img_normalized_grayscale(PathBuf::from(self.image).as_path());

        let result = nn.proceed(&input.try_into().unwrap(), mormalizers::sigmoid);

        let nums: Arr<i32, 10> = (0..=9).collect::<Vec<_>>().try_into().expect("wrong size iterator");

        ImageDumper::dump(nn.deref_mut());

        print_array("predicted: ", &result);
        print_array("nums     : ", &nums.map_copy(|a| a as f64));
    }
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct TrainMnist {
    #[clap(short, long)]
    dataset_dir: String,
    model_path: String
}

impl TrainMnist {
    fn exec(self) {
        let mut nn = Box::new(current_nn());

        let dataset = MnistDataset::load(&self.dataset_dir);

        let trainer = Trainer { 
            learning_rate: 0.0002, 
            stop_cost: 1., 
            stop_ok_coef: 0.9,
            normalizer: mormalizers::sigmoid,
            normalizer_derivative: mormalizers::sigmoid_derivative,
            epoch: 10000,
            batch_size: 100
        };
    
        trainer.train_rand(dataset.training_data(), nn.as_mut());
    
        let json = serde_json::to_string_pretty(&nn);
        std::fs::write(self.model_path, json.unwrap()).unwrap();
    }
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct TrainYolov5 {
    #[clap(short, long)]
    dataset_yaml: String,
    model_path: String
}

impl TrainYolov5 {
    fn exec(self) {
        println!("initializing model...");
        let mut nn = Box::new(current_nn());
        println!("loading dataset...");
        let dataset = YOLOv5Dataset::load(PathBuf::from(self.dataset_yaml).as_path());

        let trainer = Trainer { 
            learning_rate: 0.002, 
            stop_cost: 1., 
            stop_ok_coef: 0.7,
            normalizer: mormalizers::sigmoid,
            normalizer_derivative: mormalizers::sigmoid_derivative,
            epoch: 100,
            batch_size: 100
        };
    
        trainer.train_rand(dataset.training_data(), nn.as_mut());
    
        let json = serde_json::to_string_pretty(&nn);
        std::fs::write(self.model_path, json.unwrap()).unwrap();    
    }
}


#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Dump {
    #[clap(short, long)]
    image: String,
    model_path: String
}

impl Dump {
    fn exec(self) {
        let nn_json = std::fs::read_to_string(self.model_path).unwrap();

        let mut nn = Box::new(current_nn());
        nn = serde_json::from_str(nn_json.as_str()).unwrap();



    }   
}


#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Reverse {
    #[clap(short, long)]
    input: String,
    model_path: String
}

pub fn parse_rev_input<const SIZE: usize>(s: &str) -> Arr<f64, SIZE> {
    s.split("|")
        .map(|s| s.parse::<f64>().unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

impl Reverse {
    fn exec(self) {
        let nn_json = std::fs::read_to_string(self.model_path).unwrap();

        let mut nn = Box::new(current_nn());
        nn = serde_json::from_str(nn_json.as_str()).unwrap();

        let input: Arr<f64, 10> = parse_rev_input(&self.input);

        print_array("input: ", &input);
        
        let result = nn.reverse_proceed(&input, mormalizers::sigmoid);

        save_grayscale(PathBuf::from("./dump/rev.png").as_path(), &result);
    }
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Dev {

}

impl Dev {
    pub fn exec(self) {
        use pure_deepdream::flat::connection::NetConnection;
        use rand::thread_rng;

        let conn = NetConnection::md(&[2, 3, 2]);

        println!("conn: {:#.6?}", conn);

        let gr = conn.into_graphics_representation().unwrap();

        let units: Vec<_> = gr.0.collect();
        let layers: Vec<_> = gr.1.collect();
        let count = gr.2;

        println!("units : {:#.6?}", units);
        println!("layers: {:#?}", layers);
        println!("count : {:?}", count);



        let device = ComputeDevice::new(|device|{
            mod cs {
                vulkano_shaders::shader! {
                    ty: "compute",
                    src: "                    
                        #version 450

                        struct Vec3 {
                            uint i;
                            float x;
                            float y;
                            float z;
                        };           

                        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                        layout(set = 0, binding = 0) buffer Data {
                            Vec3 data[];
                        } data;

                        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                        layout(set = 0, binding = 1) buffer FastData {
                            Vec3 data[];
                        } fast_data;


                        void main() {                                      
                            uint i = gl_LocalInvocationIndex;

                            data.data[i].i = i;                            
                            data.data[i].x = log2(i * 1);                            
                            data.data[i].y = i * 2;                            
                            data.data[i].z = i * 3;              
                            
                            float c = 0;
                            for (int j = 0; j < 2000; j++) {
                                data.data[i].x += (data.data[i].y * data.data[i].z);
                            }
                        }
                    "
                }
            }
            cs::load(device.clone()).unwrap()
        });


        #[repr(C)]
        #[derive(Copy, Clone, Default, Debug, Pod, Zeroable)]
        struct Vec3 {
            pub i: u32,
            pub x: f32,
            pub y: f32,
            pub z: f32,
        }

        let mut data: Vec<_> = (0..(64)).map(|_| Vec3::default()).collect();

        let cpu_inst = Instant::now();
        for i in 0..data.len() {
            data[i].i = i as u32;
            data[i].x = (i as f32 * 1.).log2();                            
            data[i].y = (i as f32 * 2.).log2();                            
            data[i].z = (i as f32 * 3.).log2();
            
            let mut c = 0.;
            for _ in 0..2000 {
                c += data[i].y * data[i].z;
            }
            data[i].x = c;
        }

        let cpu_duration = cpu_inst.elapsed();
        println!("cpu elapsed: {:?}", cpu_duration);

        let len = data.len();
        let gpu_inst = Instant::now();
        let res = device.compute(
            [1024, 1, 1], 
            data.into_iter(), 
            len,
            |r| r.to_vec()
        );

        let gpu_duration = gpu_inst.elapsed();
        println!("gpu elapsed: {:?}", gpu_duration);

        let profit = (cpu_duration.as_millis() as f64) / (gpu_duration.as_millis() as f64);
        println!("profit: {:.2}", profit);

        //res;

        println!("res:");
        for r in res {
            println!("\t{:?}", r)
        }

    }
}


#[derive(Parser)]
enum SubCommand {
    Test(Test),
    TrainMnist(TrainMnist),
    TrainYolov5(TrainYolov5),
    Dump(Dump),
    Reverse(Reverse),
    Dev(Dev)
}

fn new_ships_nn<'de>() -> impl NeuralNet<249600, 10> + Serialize + Deserialize<'de> {
    RecNeuralNet {
        end_layer: PerceptronLayer::<_, 10, _, _>::new(
        PerceptronLayer::<_, 128, _, _>::new(
        PerceptronLayer::<_, 1024, _, _>::new(
        PerceptronLayer::<_, 1024, _, _>::new(
        InputLayer2d::<600, 416>::new()
        )))),
        device: Default::default()
    }
}

fn new_nn2<'de>() -> impl NeuralNet<784, 10> + Serialize + Deserialize<'de> {
    RecNeuralNet {
        end_layer: PerceptronLayer::<_, 10, _, _>::new(
        PerceptronLayer::<_, 32, _, _>::new(
        PerceptronLayer::<_, 784, _, _>::new(
        PerceptronLayer::<_, 784, _, _>::new(
                InputLayer2d::<28, 28>::new()
        )))),
        device: Default::default()
    }
}


fn main() {
    //futures::executor::block_on(assert_device_pool_initialized());
    match SubCommand::parse() {
        SubCommand::Test(test) => test.exec(),
        SubCommand::TrainMnist(train) => train.exec(),
        SubCommand::TrainYolov5(train) => train.exec(),
        SubCommand::Dump(dump) => dump.exec(),
        SubCommand::Reverse(rev) => rev.exec(),
        SubCommand::Dev(dev) => dev.exec()
    }




    //println!("structure: {:?}", endlayer);




}

// train banchmark history
// 100 ms