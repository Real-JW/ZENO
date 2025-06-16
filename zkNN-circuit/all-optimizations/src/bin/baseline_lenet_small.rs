use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

// Helper functions to read data from files
fn read_vector4d(filename: String, dim1: usize, dim2: usize, dim3: usize, dim4: usize) -> Vec<Vec<Vec<Vec<u8>>>> {
    let file = File::open(&filename).expect(&format!("Could not open file: {}", filename));
    let reader = BufReader::new(file);
    let mut data = vec![vec![vec![vec![0u8; dim4]; dim3]; dim2]; dim1];
    
    let mut idx = 0;
    let total_elements = dim1 * dim2 * dim3 * dim4;
    
    for line in reader.lines() {
        if let Ok(line) = line {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            
            for value_str in line.split_whitespace() {
                if let Ok(value) = value_str.parse::<u8>() {
                    if idx >= total_elements {
                        eprintln!("Warning: More data than expected in file {}", filename);
                        break;
                    }
                    
                    let i = idx / (dim2 * dim3 * dim4);
                    let j = (idx / (dim3 * dim4)) % dim2;
                    let k = (idx / dim4) % dim3;
                    let l = idx % dim4;
                    
                    if i < dim1 && j < dim2 && k < dim3 && l < dim4 {
                        data[i][j][k][l] = value;
                    }
                    idx += 1;
                }
            }
        }
    }
    
    if idx < total_elements {
        eprintln!("Warning: Less data than expected in file {}. Expected: {}, Got: {}", filename, total_elements, idx);
    }
    
    data
}

fn read_vector2d(filename: String, dim1: usize, dim2: usize) -> Vec<Vec<u8>> {
    let file = File::open(&filename).expect(&format!("Could not open file: {}", filename));
    let reader = BufReader::new(file);
    let mut data = vec![vec![0u8; dim2]; dim1];
    
    let mut idx = 0;
    let total_elements = dim1 * dim2;
    
    for line in reader.lines() {
        if let Ok(line) = line {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            
            for value_str in line.split_whitespace() {
                if let Ok(value) = value_str.parse::<u8>() {
                    if idx >= total_elements {
                        eprintln!("Warning: More data than expected in file {}", filename);
                        break;
                    }
                    
                    let i = idx / dim2;
                    let j = idx % dim2;
                    
                    if i < dim1 && j < dim2 {
                        data[i][j] = value;
                    }
                    idx += 1;
                }
            }
        }
    }
    
    if idx < total_elements {
        eprintln!("Warning: Less data than expected in file {}. Expected: {}, Got: {}", filename, total_elements, idx);
    }
    
    data
}

fn read_vector1d(filename: String, expected_dim: usize) -> Vec<u8> {
    let file = File::open(&filename).expect(&format!("Could not open file: {}", filename));
    let reader = BufReader::new(file);
    let mut data = Vec::new();
    
    for line in reader.lines() {
        if let Ok(line) = line {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            
            for value_str in line.split_whitespace() {
                if let Ok(value) = value_str.parse::<u8>() {
                    data.push(value);
                }
            }
        }
    }
    
    if expected_dim > 0 && data.len() != expected_dim {
        eprintln!("Warning: Expected {} elements in file {}, but got {}", expected_dim, filename, data.len());
    }
    
    data
}

fn read_vector1d_f32(filename: String, expected_dim: usize) -> Vec<f32> {
    let file = File::open(&filename).expect(&format!("Could not open file: {}", filename));
    let reader = BufReader::new(file);
    let mut data = Vec::new();
    
    for line in reader.lines() {
        if let Ok(line) = line {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            
            for value_str in line.split_whitespace() {
                if let Ok(value) = value_str.parse::<f32>() {
                    data.push(value);
                }
            }
        }
    }
    
    if expected_dim > 0 && data.len() != expected_dim {
        eprintln!("Warning: Expected {} elements in file {}, but got {}", expected_dim, filename, data.len());
    }
    
    data
}

// Quantized convolution operation
fn quantized_conv2d(
    input: &Vec<Vec<Vec<Vec<u8>>>>,
    weights: &Vec<Vec<Vec<Vec<u8>>>>,
    input_zero_point: u8,
    weight_zero_point: u8,
    output_zero_point: u8,
    multiplier: &Vec<f32>,
    stride: usize,
    padding: usize,
) -> Vec<Vec<Vec<Vec<u8>>>> {
    let batch_size = input.len();
    let input_channels = input[0].len();
    let input_height = input[0][0].len();
    let input_width = input[0][0][0].len();
    
    let output_channels = weights.len();
    let kernel_height = weights[0][0].len();
    let kernel_width = weights[0][0][0].len();
    
    let output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    let output_width = (input_width + 2 * padding - kernel_width) / stride + 1;
    
    let mut output = vec![vec![vec![vec![0u8; output_width]; output_height]; output_channels]; batch_size];
    
    for b in 0..batch_size {
        for oc in 0..output_channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let mut acc: i32 = 0;
                    
                    for ic in 0..input_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;
                                
                                if ih >= padding && ih < input_height + padding && 
                                   iw >= padding && iw < input_width + padding {
                                    let input_h = ih - padding;
                                    let input_w = iw - padding;
                                    
                                    if input_h < input_height && input_w < input_width {
                                        let input_val = input[b][ic][input_h][input_w] as i32 - input_zero_point as i32;
                                        let weight_val = weights[oc][ic][kh][kw] as i32 - weight_zero_point as i32;
                                        acc += input_val * weight_val;
                                    }
                                }
                            }
                        }
                    }
                    
                    // Apply scaling and quantization
                    let scaled = (acc as f32 * multiplier[oc]) + output_zero_point as f32;
                    output[b][oc][oh][ow] = scaled.max(0.0).min(255.0) as u8;
                }
            }
        }
    }
    
    output
}

// ReLU activation (clamp to zero point for negative values)
fn relu_quantized(input: &mut Vec<Vec<Vec<Vec<u8>>>>, zero_point: u8) {
    for b in input.iter_mut() {
        for c in b.iter_mut() {
            for h in c.iter_mut() {
                for w in h.iter_mut() {
                    if (*w as i32) < (zero_point as i32) {
                        *w = zero_point;
                    }
                }
            }
        }
    }
}

// Max pooling operation
fn max_pool2d(
    input: &Vec<Vec<Vec<Vec<u8>>>>,
    pool_size: usize,
    stride: usize,
) -> Vec<Vec<Vec<Vec<u8>>>> {
    let batch_size = input.len();
    let channels = input[0].len();
    let input_height = input[0][0].len();
    let input_width = input[0][0][0].len();
    
    let output_height = (input_height - pool_size) / stride + 1;
    let output_width = (input_width - pool_size) / stride + 1;
    
    let mut output = vec![vec![vec![vec![0u8; output_width]; output_height]; channels]; batch_size];
    
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let mut max_val = 0u8;
                    
                    for ph in 0..pool_size {
                        for pw in 0..pool_size {
                            let ih = oh * stride + ph;
                            let iw = ow * stride + pw;
                            max_val = max_val.max(input[b][c][ih][iw]);
                        }
                    }
                    
                    output[b][c][oh][ow] = max_val;
                }
            }
        }
    }
    
    output
}

// Flatten 4D tensor to 2D for fully connected layers
fn flatten(input: &Vec<Vec<Vec<Vec<u8>>>>) -> Vec<Vec<u8>> {
    let batch_size = input.len();
    let channels = input[0].len();
    let height = input[0][0].len();
    let width = input[0][0][0].len();
    let flattened_size = channels * height * width;
    
    let mut output = vec![vec![0u8; flattened_size]; batch_size];
    
    for b in 0..batch_size {
        let mut idx = 0;
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    output[b][idx] = input[b][c][h][w];
                    idx += 1;
                }
            }
        }
    }
    
    output
}

// Quantized fully connected layer
fn quantized_linear(
    input: &Vec<Vec<u8>>,
    weights: &Vec<Vec<u8>>,
    input_zero_point: u8,
    weight_zero_point: u8,
    output_zero_point: u8,
    multiplier: &Vec<f32>,
) -> Vec<Vec<u8>> {
    let batch_size = input.len();
    let input_features = input[0].len();
    let output_features = weights.len();
    
    let mut output = vec![vec![0u8; output_features]; batch_size];
    
    for b in 0..batch_size {
        for of in 0..output_features {
            let mut acc: i32 = 0;
            
            for if_ in 0..input_features {
                let input_val = input[b][if_] as i32 - input_zero_point as i32;
                let weight_val = weights[of][if_] as i32 - weight_zero_point as i32;
                acc += input_val * weight_val;
            }
            
            // Apply scaling and quantization
            let scaled = (acc as f32 * multiplier[of]) + output_zero_point as f32;
            output[b][of] = scaled.max(0.0).min(255.0) as u8;
        }
    }
    
    output
}

// LeNet inference function
fn lenet_inference(
    x: Vec<Vec<Vec<Vec<u8>>>>,
    conv1_w: Vec<Vec<Vec<Vec<u8>>>>,
    conv2_w: Vec<Vec<Vec<Vec<u8>>>>,
    conv3_w: Vec<Vec<Vec<Vec<u8>>>>,
    fc1_w: Vec<Vec<u8>>,
    fc2_w: Vec<Vec<u8>>,
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
    multiplier_conv1: Vec<f32>,
    multiplier_conv2: Vec<f32>,
    multiplier_conv3: Vec<f32>,
    multiplier_fc1: Vec<f32>,
    multiplier_fc2: Vec<f32>,
) -> Vec<Vec<u8>> {
    let start_time = Instant::now();
    
    // Conv1: 3x32x32 -> 6x28x28 (5x5 kernel, no padding, stride 1)
    println!("Starting Conv1...");
    let mut conv1_out = quantized_conv2d(
        &x, &conv1_w, x_0, conv1_weights_0, conv1_output_0,
        &multiplier_conv1, 1, 0
    );
    relu_quantized(&mut conv1_out, conv1_output_0);
    println!("Conv1 completed: {:?}", conv1_out[0].len());
    
    // MaxPool1: 6x28x28 -> 6x14x14 (2x2 kernel, stride 2)
    println!("Starting MaxPool1...");
    let pool1_out = max_pool2d(&conv1_out, 2, 2);
    println!("MaxPool1 completed: {}x{}", pool1_out[0][0].len(), pool1_out[0][0][0].len());
    
    // Conv2: 6x14x14 -> 16x10x10 (5x5 kernel, no padding, stride 1)
    println!("Starting Conv2...");
    let mut conv2_out = quantized_conv2d(
        &pool1_out, &conv2_w, conv1_output_0, conv2_weights_0, conv2_output_0,
        &multiplier_conv2, 1, 0
    );
    relu_quantized(&mut conv2_out, conv2_output_0);
    println!("Conv2 completed: {}x{}", conv2_out[0][0].len(), conv2_out[0][0][0].len());
    
    // MaxPool2: 16x10x10 -> 16x5x5 (2x2 kernel, stride 2)
    println!("Starting MaxPool2...");
    let pool2_out = max_pool2d(&conv2_out, 2, 2);
    println!("MaxPool2 completed: {}x{}", pool2_out[0][0].len(), pool2_out[0][0][0].len());
    
    // Conv3: 16x5x5 -> 120x2x2 (4x4 kernel, no padding, stride 1)
    println!("Starting Conv3...");
    let mut conv3_out = quantized_conv2d(
        &pool2_out, &conv3_w, conv2_output_0, conv3_weights_0, conv3_output_0,
        &multiplier_conv3, 1, 0
    );
    relu_quantized(&mut conv3_out, conv3_output_0);
    println!("Conv3 completed: {}x{}", conv3_out[0][0].len(), conv3_out[0][0][0].len());
    
    // Flatten: 120x2x2 -> 480
    println!("Starting Flatten...");
    let flattened = flatten(&conv3_out);
    println!("Flatten completed: {}", flattened[0].len());
    
    // FC1: 480 -> 84
    println!("Starting FC1...");
    let fc1_out = quantized_linear(
        &flattened, &fc1_w, conv3_output_0, fc1_weights_0, fc1_output_0,
        &multiplier_fc1
    );
    println!("FC1 completed: {}", fc1_out[0].len());
    
    // FC2: 84 -> 10
    println!("Starting FC2...");
    let fc2_out = quantized_linear(
        &fc1_out, &fc2_w, fc1_output_0, fc2_weights_0, fc2_output_0,
        &multiplier_fc2
    );
    println!("FC2 completed: {}", fc2_out[0].len());
    
    let elapsed = start_time.elapsed();
    println!("Total inference time: {:?}", elapsed);
    
    fc2_out
}

fn main() {
    println!("Loading LeNet model data...");
    
    // First, let's check the actual size of the input data
    let x_raw: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/X_q.txt".to_string(),
        0, // Don't enforce size check
    );
    
    println!("Input data contains {} elements", x_raw.len());
    
    // Calculate how many images we have
    let elements_per_image = 3 * 32 * 32; // 3072 elements per CIFAR image
    let num_images = x_raw.len() / elements_per_image;
    println!("Detected {} images in input file", num_images);
    
    // Reshape the data into proper 4D format, taking only the first image
    let mut x = vec![vec![vec![vec![0u8; 32]; 32]; 3]; 1];
    for idx in 0..elements_per_image {
        let c = idx / (32 * 32);
        let h = (idx / 32) % 32;
        let w = idx % 32;
        if c < 3 && h < 32 && w < 32 {
            x[0][c][h][w] = x_raw[idx];
        }
    }
    
    println!("Reshaped input to 1x3x32x32");
    
    // Load weights and other parameters normally
    let conv1_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv1_weight_q.txt".to_string(),
        6, 3, 5, 5,
    );
    
    let conv2_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv2_weight_q.txt".to_string(),
        16, 6, 5, 5,
    );
    
    let conv3_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv3_weight_q.txt".to_string(),
        120, 16, 4, 4,
    );
    
    let fc1_w: Vec<Vec<u8>> = read_vector2d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear1_weight_q.txt".to_string(),
        84, 480,
    );
    
    let fc2_w: Vec<Vec<u8>> = read_vector2d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear2_weight_q.txt".to_string(),
        10, 84,
    );

    // Load zero points (single values)
    let x_0_vec: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/X_z.txt".to_string(), 0,
    );
    let conv1_output_0_vec: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv1_output_z.txt".to_string(), 0,
    );
    let conv2_output_0_vec: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv2_output_z.txt".to_string(), 0,
    );
    let conv3_output_0_vec: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv3_output_z.txt".to_string(), 0,
    );
    let fc1_output_0_vec: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear1_output_z.txt".to_string(), 0,
    );
    let fc2_output_0_vec: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear2_output_z.txt".to_string(), 0,
    );

    let conv1_weights_0_vec: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv1_weight_z.txt".to_string(), 0,
    );
    let conv2_weights_0_vec: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv2_weight_z.txt".to_string(), 0,
    );
    let conv3_weights_0_vec: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv3_weight_z.txt".to_string(), 0,
    );
    let fc1_weights_0_vec: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear1_weight_z.txt".to_string(), 0,
    );
    let fc2_weights_0_vec: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear2_weight_z.txt".to_string(), 0,
    );

    // Extract scalar values (use first element, default to 128 if empty)
    let x_0 = if x_0_vec.is_empty() { 128 } else { x_0_vec[0] };
    let conv1_output_0 = if conv1_output_0_vec.is_empty() { 128 } else { conv1_output_0_vec[0] };
    let conv2_output_0 = if conv2_output_0_vec.is_empty() { 128 } else { conv2_output_0_vec[0] };
    let conv3_output_0 = if conv3_output_0_vec.is_empty() { 128 } else { conv3_output_0_vec[0] };
    let fc1_output_0 = if fc1_output_0_vec.is_empty() { 128 } else { fc1_output_0_vec[0] };
    let fc2_output_0 = if fc2_output_0_vec.is_empty() { 128 } else { fc2_output_0_vec[0] };
    let conv1_weights_0 = if conv1_weights_0_vec.is_empty() { 128 } else { conv1_weights_0_vec[0] };
    let conv2_weights_0 = if conv2_weights_0_vec.is_empty() { 128 } else { conv2_weights_0_vec[0] };
    let conv3_weights_0 = if conv3_weights_0_vec.is_empty() { 128 } else { conv3_weights_0_vec[0] };
    let fc1_weights_0 = if fc1_weights_0_vec.is_empty() { 128 } else { fc1_weights_0_vec[0] };
    let fc2_weights_0 = if fc2_weights_0_vec.is_empty() { 128 } else { fc2_weights_0_vec[0] };

    // Load multipliers
    let multiplier_conv1: Vec<f32> = read_vector1d_f32(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv1_weight_s.txt".to_string(), 0,
    );
    let multiplier_conv2: Vec<f32> = read_vector1d_f32(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv2_weight_s.txt".to_string(), 0,
    );
    let multiplier_conv3: Vec<f32> = read_vector1d_f32(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv3_weight_s.txt".to_string(), 0,
    );
    let multiplier_fc1: Vec<f32> = read_vector1d_f32(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear1_weight_s.txt".to_string(), 0,
    );
    let multiplier_fc2: Vec<f32> = read_vector1d_f32(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear2_weight_s.txt".to_string(), 0,
    );

    println!("Data loaded successfully!");
    println!("Zero points: x={}, conv1_out={}, conv2_out={}, conv3_out={}", x_0, conv1_output_0, conv2_output_0, conv3_output_0);
    println!("Multipliers: conv1={:?}, conv2={:?}", &multiplier_conv1[..std::cmp::min(3, multiplier_conv1.len())], &multiplier_conv2[..std::cmp::min(3, multiplier_conv2.len())]);
    println!("Starting inference...");

    // Run inference multiple times for latency measurement
    let num_runs = 10;
    let mut total_time = std::time::Duration::new(0, 0);
    
    for i in 0..num_runs {
        println!("\n=== Inference Run {} ===", i + 1);
        let start = Instant::now();
        
        let result = lenet_inference(
            x.clone(),
            conv1_w.clone(),
            conv2_w.clone(), 
            conv3_w.clone(),
            fc1_w.clone(),
            fc2_w.clone(),
            x_0,
            conv1_output_0,
            conv2_output_0,
            conv3_output_0,
            fc1_output_0,
            fc2_output_0,
            conv1_weights_0,
            conv2_weights_0,
            conv3_weights_0,
            fc1_weights_0,
            fc2_weights_0,
            multiplier_conv1.clone(),
            multiplier_conv2.clone(),
            multiplier_conv3.clone(),
            multiplier_fc1.clone(),
            multiplier_fc2.clone(),
        );
        
        let elapsed = start.elapsed();
        total_time += elapsed;
        
        println!("Output logits: {:?}", result[0]);
        println!("Run {} completed in: {:?}", i + 1, elapsed);
    }
    
    println!("\n=== Latency Statistics ===");
    println!("Average inference time: {:?}", total_time / num_runs as u32);
    println!("Total time for {} runs: {:?}", num_runs, total_time);
}