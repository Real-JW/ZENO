use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process;
use std::time::Instant;

// Helper functions to read data from files
fn read_vector1d(filename: String, size: usize) -> Vec<u8> {
    let file = match File::open(&filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error: Failed to open {}: {}", filename, e);
            process::exit(1);
        }
    };
    let reader = BufReader::new(file);
    let mut result = Vec::with_capacity(size);
    
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Error: Failed to read line in {}: {}", filename, e);
                process::exit(1);
            }
        };
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            if let Ok(val) = trimmed.parse::<u8>() {
                result.push(val);
            }
        }
    }
    
    if result.is_empty() {
        eprintln!("Error: No data read from file: {}", filename);
        process::exit(1);
    }
    
    println!("Loaded {} values from {}", result.len(), filename);
    result
}

fn read_vector1d_f32(filename: String, len: usize) -> Vec<f32> {
    let f = File::open(filename.to_string()).unwrap();
    let mut res: Vec<f32> = vec![0.0f32; len];
    let buffered = BufReader::new(f);

    let mut counter = 0;

    for line in buffered.lines() {
        //println!("{:?}", line.unwrap().split(" ").collect::<Vec<&str>>());
        let raw_vec: Vec<f32> = line
            .unwrap()
            .split(" ")
            .collect::<Vec<&str>>()
            .into_iter()
            .filter(|&s| !s.is_empty())
            .map(|s| s.parse::<f32>().unwrap())
            .collect();
        for i in 0..raw_vec.len() {
            if counter < len {
                res[counter] = raw_vec[i];
            }
            counter += 1;
        }
    }
    //println!("{:?}", res);
    res
}

fn read_vector2d(filename: String, dim1: usize, dim2: usize) -> Vec<Vec<u8>> {
    let file = match File::open(&filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error: Failed to open {}: {}", filename, e);
            process::exit(1);
        }
    };
    let reader = BufReader::new(file);
    let mut result = vec![vec![0u8; dim2]; dim1];
    let mut idx = 0;
    
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Error: Failed to read line in {}: {}", filename, e);
                process::exit(1);
            }
        };
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            if let Ok(val) = trimmed.parse::<u8>() {
                let i = idx / dim2;
                let j = idx % dim2;
                if i < dim1 && j < dim2 {
                    result[i][j] = val;
                }
                idx += 1;
            }
        }
    }
    
    if idx == 0 {
        eprintln!("Error: No data read from file: {}", filename);
        process::exit(1);
    }
    
    println!("Loaded {}x{} matrix from {} (total {} values)", dim1, dim2, filename, idx);
    result
}

fn read_vector4d(filename: String, dim1: usize, dim2: usize, dim3: usize, dim4: usize) -> Vec<Vec<Vec<Vec<u8>>>> {
    let file = match File::open(&filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error: Failed to open {}: {}", filename, e);
            process::exit(1);
        }
    };
    let reader = BufReader::new(file);
    let mut result = vec![vec![vec![vec![0u8; dim4]; dim3]; dim2]; dim1];
    let mut idx = 0;
    
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Error: Failed to read line in {}: {}", filename, e);
                process::exit(1);
            }
        };
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            if let Ok(val) = trimmed.parse::<u8>() {
                let i = idx / (dim2 * dim3 * dim4);
                let j = (idx / (dim3 * dim4)) % dim2;
                let k = (idx / dim4) % dim3;
                let l = idx % dim4;
                if i < dim1 && j < dim2 && k < dim3 && l < dim4 {
                    result[i][j][k][l] = val;
                }
                idx += 1;
            }
        }
    }
    
    if idx == 0 {
        eprintln!("Error: No data read from file: {}", filename);
        process::exit(1);
    }
    
    println!("Loaded {}x{}x{}x{} tensor from {} (total {} values)", dim1, dim2, dim3, dim4, filename, idx);
    result
}

// Quantized convolution layer
fn quantized_conv2d(
    input: &Vec<Vec<Vec<u8>>>,
    weights: &Vec<Vec<Vec<Vec<u8>>>>,
    input_zero: u8,
    weight_zero: u8,
    output_zero: u8,
    multiplier: &Vec<f32>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Vec<Vec<Vec<u8>>> {
    let input_channels = input.len();
    let input_height = input[0].len();
    let input_width = input[0][0].len();
    let output_channels = weights.len();
    
    // Debug info
    println!("Conv2D: input {}x{}x{}, {} output channels, kernel {}, stride {}, padding {}", 
             input_channels, input_height, input_width, output_channels, kernel_size, stride, padding);
    
    if input_channels == 0 || output_channels == 0 || multiplier.len() != output_channels {
        panic!("Invalid convolution parameters: input_ch={}, output_ch={}, multiplier_len={}", 
               input_channels, output_channels, multiplier.len());
    }
    
    let output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    let output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    println!("Conv2D output size: {}x{}x{}", output_channels, output_height, output_width);
    
    let mut output = vec![vec![vec![0u8; output_width]; output_height]; output_channels];
    
    for out_ch in 0..output_channels {
        for out_h in 0..output_height {
            for out_w in 0..output_width {
                let mut sum = 0i32;
                
                for in_ch in 0..input_channels {
                    for kh in 0..kernel_size {
                        for kw in 0..kernel_size {
                            let in_h = out_h * stride + kh;
                            let in_w = out_w * stride + kw;
                            
                            let input_val = if in_h >= padding && in_w >= padding && 
                                             in_h < input_height + padding && in_w < input_width + padding {
                                let actual_h = in_h - padding;
                                let actual_w = in_w - padding;
                                if actual_h < input_height && actual_w < input_width {
                                    input[in_ch][actual_h][actual_w] as i32
                                } else {
                                    input_zero as i32
                                }
                            } else {
                                input_zero as i32
                            };
                            
                            let weight_val = weights[out_ch][in_ch][kh][kw] as i32;
                            sum += (input_val - input_zero as i32) * (weight_val - weight_zero as i32);
                        }
                    }
                }
                
                // Apply scaling factor and requantize
                let scaled = (sum as f32) * multiplier[out_ch];
                let quantized = (scaled + output_zero as f32).round() as i32;
                output[out_ch][out_h][out_w] = quantized.max(0).min(255) as u8;
            }
        }
    }
    
    output
}

// ReLU activation
fn relu(input: &mut Vec<Vec<Vec<u8>>>, zero_point: u8) {
    for ch in 0..input.len() {
        for h in 0..input[ch].len() {
            for w in 0..input[ch][h].len() {
                if input[ch][h][w] < zero_point {
                    input[ch][h][w] = zero_point;
                }
            }
        }
    }
}

// Max pooling
fn max_pool2d(input: &Vec<Vec<Vec<u8>>>, pool_size: usize, stride: usize) -> Vec<Vec<Vec<u8>>> {
    let channels = input.len();
    let input_height = input[0].len();
    let input_width = input[0][0].len();
    
    let output_height = (input_height - pool_size) / stride + 1;
    let output_width = (input_width - pool_size) / stride + 1;
    
    let mut output = vec![vec![vec![0u8; output_width]; output_height]; channels];
    
    for ch in 0..channels {
        for out_h in 0..output_height {
            for out_w in 0..output_width {
                let mut max_val = 0u8;
                
                for ph in 0..pool_size {
                    for pw in 0..pool_size {
                        let in_h = out_h * stride + ph;
                        let in_w = out_w * stride + pw;
                        if in_h < input_height && in_w < input_width {
                            max_val = max_val.max(input[ch][in_h][in_w]);
                        }
                    }
                }
                
                output[ch][out_h][out_w] = max_val;
            }
        }
    }
    
    output
}

// Flatten for fully connected layer
fn flatten(input: &Vec<Vec<Vec<u8>>>) -> Vec<u8> {
    let mut result = Vec::new();
    for ch in 0..input.len() {
        for h in 0..input[ch].len() {
            for w in 0..input[ch][h].len() {
                result.push(input[ch][h][w]);
            }
        }
    }
    result
}

// Quantized fully connected layer
fn quantized_fc(
    input: &Vec<u8>,
    weights: &Vec<Vec<u8>>,
    input_zero: u8,
    weight_zero: u8,
    output_zero: u8,
    multiplier: &Vec<f32>,
) -> Vec<u8> {
    let output_size = weights.len();
    let input_size = weights[0].len();
    let mut output = vec![0u8; output_size];
    
    for out_idx in 0..output_size {
        let mut sum = 0i32;
        
        for in_idx in 0..input_size {
            let input_val = input[in_idx] as i32;
            let weight_val = weights[out_idx][in_idx] as i32;
            sum += (input_val - input_zero as i32) * (weight_val - weight_zero as i32);
        }
        
        // Apply scaling factor and requantize
        let scaled = (sum as f32) * multiplier[out_idx];
        let quantized = (scaled + output_zero as f32).round() as i32;
        output[out_idx] = quantized.max(0).min(255) as u8;
    }
    
    output
}

// LeNet inference function
fn lenet_inference(
    input: &Vec<Vec<Vec<Vec<u8>>>>,
    conv1_w: &Vec<Vec<Vec<Vec<u8>>>>,
    conv2_w: &Vec<Vec<Vec<Vec<u8>>>>,
    conv3_w: &Vec<Vec<Vec<Vec<u8>>>>,
    fc1_w: &Vec<Vec<u8>>,
    fc2_w: &Vec<Vec<u8>>,
    x_0: u8,
    conv1_output_0: u8,
    conv2_output_0: u8,
    conv3_output_0: u8,
    fc1_output_0: u8,
    fc2_output_0: u8,
    conv1_weights_0: u8,
    conv2_weights_0: u8,
    conv3_weights_0: u8,
    fc1_weights_0: u8,
    fc2_weights_0: u8,
    multiplier_conv1: &Vec<f32>,
    multiplier_conv2: &Vec<f32>,
    multiplier_conv3: &Vec<f32>,
    multiplier_fc1: &Vec<f32>,
    multiplier_fc2: &Vec<f32>,
) -> Vec<u8> {
    let start_time = Instant::now();
    
    // Extract single image from batch
    let x_single = &input[0];
    
    // Conv1 + ReLU + MaxPool
    println!("Running Conv1...");
    let mut conv1_out = quantized_conv2d(
        x_single, conv1_w, x_0, conv1_weights_0, conv1_output_0, 
        multiplier_conv1, 5, 1, 0
    );
    relu(&mut conv1_out, conv1_output_0);
    let pool1_out = max_pool2d(&conv1_out, 2, 2);
    
    // Conv2 + ReLU + MaxPool
    println!("Running Conv2...");
    let mut conv2_out = quantized_conv2d(
        &pool1_out, conv2_w, conv1_output_0, conv2_weights_0, conv2_output_0,
        multiplier_conv2, 5, 1, 0
    );
    relu(&mut conv2_out, conv2_output_0);
    let pool2_out = max_pool2d(&conv2_out, 2, 2);
    
    // Conv3 + ReLU
    println!("Running Conv3...");
    let mut conv3_out = quantized_conv2d(
        &pool2_out, conv3_w, conv2_output_0, conv3_weights_0, conv3_output_0,
        multiplier_conv3, 4, 1, 0
    );
    relu(&mut conv3_out, conv3_output_0);
    
    // Flatten
    let flattened = flatten(&conv3_out);
    
    // FC1 + ReLU
    println!("Running FC1...");
    let mut fc1_out = quantized_fc(
        &flattened, fc1_w, conv3_output_0, fc1_weights_0, fc1_output_0, multiplier_fc1
    );
    // Apply ReLU to FC1 output
    for val in &mut fc1_out {
        if *val < fc1_output_0 {
            *val = fc1_output_0;
        }
    }
    
    // FC2 (final layer)
    println!("Running FC2...");
    let fc2_out = quantized_fc(
        &fc1_out, fc2_w, fc1_output_0, fc2_weights_0, fc2_output_0, multiplier_fc2
    );
    
    let inference_time = start_time.elapsed();
    println!("Inference completed in: {:?}", inference_time);
    
    fc2_out
}

fn main() {
    println!("LeNet optimized medium on CIFAR dataset");

    // Load all model parameters with debug info
    println!("Loading input data...");
    let x: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "../pretrained_model/LeNet_CIFAR_pretrained/X_q.txt".to_string(),
        1, 3, 32, 32,
    );
    
    println!("Loading conv1 weights...");
    let conv1_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv1_weight_q.txt".to_string(),
        32, 3, 5, 5,
    );
    println!("Loading conv2 weights...");
    let conv2_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv2_weight_q.txt".to_string(),
        64, 32, 5, 5,
    );
    println!("Loading conv3 weights...");
    let conv3_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv3_weight_q.txt".to_string(),
        256, 64, 4, 4,
    );
    println!("Loading fc1 weights...");
    let fc1_w: Vec<Vec<u8>> = read_vector2d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear1_weight_q.txt".to_string(),
        128, 1024,
    );
    println!("Loading fc2 weights...");
    let fc2_w: Vec<Vec<u8>> = read_vector2d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear2_weight_q.txt".to_string(),
        10, 128,
    );

    // Load zero points
    println!("Loading zero points...");
    let x_0 = read_vector1d("../pretrained_model/LeNet_CIFAR_pretrained/X_z.txt".to_string(), 1)[0];
    let conv1_output_0 = read_vector1d("../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv1_output_z.txt".to_string(), 1)[0];
    let conv2_output_0 = read_vector1d("../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv2_output_z.txt".to_string(), 1)[0];
    let conv3_output_0 = read_vector1d("../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv3_output_z.txt".to_string(), 1)[0];
    let fc1_output_0 = read_vector1d("../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear1_output_z.txt".to_string(), 1)[0];
    let fc2_output_0 = read_vector1d("../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear2_output_z.txt".to_string(), 1)[0];

    let conv1_weights_0 = read_vector1d("../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv1_weight_z.txt".to_string(), 1)[0];
    let conv2_weights_0 = read_vector1d("../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv2_weight_z.txt".to_string(), 1)[0];
    let conv3_weights_0 = read_vector1d("../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv3_weight_z.txt".to_string(), 1)[0];
    let fc1_weights_0 = read_vector1d("../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear1_weight_z.txt".to_string(), 1)[0];
    let fc2_weights_0 = read_vector1d("../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear2_weight_z.txt".to_string(), 1)[0];

    // Load multipliers
    println!("Loading scale factors...");
    let multiplier_conv1: Vec<f32> = read_vector1d_f32(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv1_weight_s.txt".to_string(), 32,
    );
    let multiplier_conv2: Vec<f32> = read_vector1d_f32(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv2_weight_s.txt".to_string(), 64,
    );
    let multiplier_conv3: Vec<f32> = read_vector1d_f32(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv3_weight_s.txt".to_string(), 256,
    );
    let multiplier_fc1: Vec<f32> = read_vector1d_f32(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear1_weight_s.txt".to_string(), 128,
    );
    let multiplier_fc2: Vec<f32> = read_vector1d_f32(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear2_weight_s.txt".to_string(), 10,
    );

    println!("All data loaded successfully. Starting inference...");

    // Run inference
    let output = lenet_inference(
        &x, &conv1_w, &conv2_w, &conv3_w, &fc1_w, &fc2_w,
        x_0, conv1_output_0, conv2_output_0, conv3_output_0, fc1_output_0, fc2_output_0,
        conv1_weights_0, conv2_weights_0, conv3_weights_0, fc1_weights_0, fc2_weights_0,
        &multiplier_conv1, &multiplier_conv2, &multiplier_conv3, &multiplier_fc1, &multiplier_fc2,
    );

    // Find predicted class
    let mut max_idx = 0;
    let mut max_val = output[0];
    for (i, &val) in output.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    println!("Predicted class: {}", max_idx);
    println!("Output logits: {:?}", output);
}

// Add to Cargo.toml:
// [dependencies]
// (no external dependencies needed)