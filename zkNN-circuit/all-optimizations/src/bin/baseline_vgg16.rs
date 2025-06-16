use rand::Rng;

// Helper function to generate random 4D vectors (for weights)
fn rand_4d_vec_generator(d1: usize, d2: usize, d3: usize, d4: usize) -> Vec<Vec<Vec<Vec<u8>>>> {
    let mut rng = rand::thread_rng();
    (0..d1)
        .map(|_| {
            (0..d2)
                .map(|_| {
                    (0..d3)
                        .map(|_| (0..d4).map(|_| rng.gen::<u8>()).collect())
                        .collect()
                })
                .collect()
        })
        .collect()
}

// Helper function to generate random 2D vectors (for FC weights)
fn rand_2d_vec_generator(d1: usize, d2: usize) -> Vec<Vec<u8>> {
    let mut rng = rand::thread_rng();
    (0..d1)
        .map(|_| (0..d2).map(|_| rng.gen::<u8>()).collect())
        .collect()
}

// Convolution operation with quantized weights
fn conv2d(
    input: &Vec<Vec<Vec<Vec<f32>>>>,
    weights: &Vec<Vec<Vec<Vec<u8>>>>,
    multiplier: &Vec<f32>,
    stride: usize,
    padding: usize,
) -> Vec<Vec<Vec<Vec<f32>>>> {
    let batch_size = input.len();
    let in_channels = input[0].len();
    let in_height = input[0][0].len();
    let in_width = input[0][0][0].len();
    
    let out_channels = weights.len();
    let kernel_size = weights[0][0].len();
    
    let out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    let out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    let mut output = vec![vec![vec![vec![0.0f32; out_width]; out_height]; out_channels]; batch_size];
    
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut sum = 0.0f32;
                    
                    for ic in 0..in_channels {
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let ih = (oh * stride + kh) as i32 - padding as i32;
                                let iw = (ow * stride + kw) as i32 - padding as i32;
                                
                                if ih >= 0 && ih < in_height as i32 && iw >= 0 && iw < in_width as i32 {
                                    let input_val = input[b][ic][ih as usize][iw as usize];
                                    let weight_val = weights[oc][ic][kh][kw] as f32 * multiplier[oc];
                                    sum += input_val * weight_val;
                                }
                            }
                        }
                    }
                    
                    output[b][oc][oh][ow] = sum;
                }
            }
        }
    }
    
    output
}

// ReLU activation
fn relu(input: &mut Vec<Vec<Vec<Vec<f32>>>>) {
    for b in input.iter_mut() {
        for c in b.iter_mut() {
            for h in c.iter_mut() {
                for w in h.iter_mut() {
                    *w = w.max(0.0);
                }
            }
        }
    }
}

// Max pooling 2x2
fn max_pool2d(input: &Vec<Vec<Vec<Vec<f32>>>>) -> Vec<Vec<Vec<Vec<f32>>>> {
    let batch_size = input.len();
    let channels = input[0].len();
    let in_height = input[0][0].len();
    let in_width = input[0][0][0].len();
    
    let out_height = in_height / 2;
    let out_width = in_width / 2;
    
    let mut output = vec![vec![vec![vec![0.0f32; out_width]; out_height]; channels]; batch_size];
    
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut max_val = f32::NEG_INFINITY;
                    
                    for kh in 0..2 {
                        for kw in 0..2 {
                            let ih = oh * 2 + kh;
                            let iw = ow * 2 + kw;
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

// Flatten for FC layer
fn flatten(input: &Vec<Vec<Vec<Vec<f32>>>>) -> Vec<Vec<f32>> {
    let batch_size = input.len();
    let channels = input[0].len();
    let height = input[0][0].len();
    let width = input[0][0][0].len();
    
    let flattened_size = channels * height * width;
    let mut output = vec![vec![0.0f32; flattened_size]; batch_size];
    
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

// Fully connected layer
fn fully_connected(
    input: &Vec<Vec<f32>>,
    weights: &Vec<Vec<u8>>,
    multiplier: &Vec<f32>,
) -> Vec<Vec<f32>> {
    let batch_size = input.len();
    let input_size = input[0].len();
    let output_size = weights.len();
    let weight_input_size = weights[0].len();
    
    // Debug information
    println!("FC Layer - Input size: {}, Weight input size: {}, Output size: {}", 
             input_size, weight_input_size, output_size);
    
    // Check dimension compatibility
    if input_size != weight_input_size {
        panic!("Dimension mismatch: input size {} != weight input size {}", 
               input_size, weight_input_size);
    }
    
    let mut output = vec![vec![0.0f32; output_size]; batch_size];
    
    for b in 0..batch_size {
        for o in 0..output_size {
            let mut sum = 0.0f32;
            for i in 0..input_size {
                sum += input[b][i] * (weights[o][i] as f32 * multiplier[o]);
            }
            output[b][o] = sum;
        }
    }
    
    output
}

// ReLU for FC layers
fn relu_fc(input: &mut Vec<Vec<f32>>) {
    for b in input.iter_mut() {
        for val in b.iter_mut() {
            *val = val.max(0.0);
        }
    }
}

// VGG16 inference function
fn vgg16_inference(
    input: Vec<Vec<Vec<Vec<f32>>>>,
    // Conv layer weights
    conv11_w: &Vec<Vec<Vec<Vec<u8>>>>, conv12_w: &Vec<Vec<Vec<Vec<u8>>>>,
    conv21_w: &Vec<Vec<Vec<Vec<u8>>>>, conv22_w: &Vec<Vec<Vec<Vec<u8>>>>,
    conv31_w: &Vec<Vec<Vec<Vec<u8>>>>, conv32_w: &Vec<Vec<Vec<Vec<u8>>>>, conv33_w: &Vec<Vec<Vec<Vec<u8>>>>,
    conv41_w: &Vec<Vec<Vec<Vec<u8>>>>, conv42_w: &Vec<Vec<Vec<Vec<u8>>>>, conv43_w: &Vec<Vec<Vec<Vec<u8>>>>,
    conv51_w: &Vec<Vec<Vec<Vec<u8>>>>, conv52_w: &Vec<Vec<Vec<Vec<u8>>>>, conv53_w: &Vec<Vec<Vec<Vec<u8>>>>,
    // FC layer weights
    fc1_w: &Vec<Vec<u8>>, fc2_w: &Vec<Vec<u8>>, fc3_w: &Vec<Vec<u8>>,
    // Multipliers
    multiplier_conv11: &Vec<f32>, multiplier_conv12: &Vec<f32>,
    multiplier_conv21: &Vec<f32>, multiplier_conv22: &Vec<f32>,
    multiplier_conv31: &Vec<f32>, multiplier_conv32: &Vec<f32>, multiplier_conv33: &Vec<f32>,
    multiplier_conv41: &Vec<f32>, multiplier_conv42: &Vec<f32>, multiplier_conv43: &Vec<f32>,
    multiplier_conv51: &Vec<f32>, multiplier_conv52: &Vec<f32>, multiplier_conv53: &Vec<f32>,
    multiplier_fc1: &Vec<f32>, multiplier_fc2: &Vec<f32>, multiplier_fc3: &Vec<f32>,
) -> Vec<Vec<f32>> {
    
    println!("Starting VGG16 inference...");
    
    // Block 1
    println!("Conv1_1...");
    let mut x = conv2d(&input, conv11_w, multiplier_conv11, 1, 1);
    relu(&mut x);
    
    println!("Conv1_2...");
    x = conv2d(&x, conv12_w, multiplier_conv12, 1, 1);
    relu(&mut x);
    
    println!("Pool1...");
    x = max_pool2d(&x);
    
    // Block 2
    println!("Conv2_1...");
    x = conv2d(&x, conv21_w, multiplier_conv21, 1, 1);
    relu(&mut x);
    
    println!("Conv2_2...");
    x = conv2d(&x, conv22_w, multiplier_conv22, 1, 1);
    relu(&mut x);
    
    println!("Pool2...");
    x = max_pool2d(&x);
    
    // Block 3
    println!("Conv3_1...");
    x = conv2d(&x, conv31_w, multiplier_conv31, 1, 1);
    relu(&mut x);
    
    println!("Conv3_2...");
    x = conv2d(&x, conv32_w, multiplier_conv32, 1, 1);
    relu(&mut x);
    
    println!("Conv3_3...");
    x = conv2d(&x, conv33_w, multiplier_conv33, 1, 1);
    relu(&mut x);
    
    println!("Pool3...");
    x = max_pool2d(&x);
    
    // Block 4
    println!("Conv4_1...");
    x = conv2d(&x, conv41_w, multiplier_conv41, 1, 1);
    relu(&mut x);
    
    println!("Conv4_2...");
    x = conv2d(&x, conv42_w, multiplier_conv42, 1, 1);
    relu(&mut x);
    
    println!("Conv4_3...");
    x = conv2d(&x, conv43_w, multiplier_conv43, 1, 1);
    relu(&mut x);
    
    println!("Pool4...");
    x = max_pool2d(&x);
    
    // Block 5
    println!("Conv5_1...");
    x = conv2d(&x, conv51_w, multiplier_conv51, 1, 1);
    relu(&mut x);
    
    println!("Conv5_2...");
    x = conv2d(&x, conv52_w, multiplier_conv52, 1, 1);
    relu(&mut x);
    
    println!("Conv5_3...");
    x = conv2d(&x, conv53_w, multiplier_conv53, 1, 1);
    relu(&mut x);
    
    println!("Pool5...");
    x = max_pool2d(&x);
    
    // Flatten
    println!("Flatten...");
    let x_flat = flatten(&x);
    println!("Flattened size: {}", x_flat[0].len());
    
    // FC layers
    println!("FC1...");
    let mut x_flat = fully_connected(&x_flat, fc1_w, multiplier_fc1);
    relu_fc(&mut x_flat);
    
    println!("FC2...");
    x_flat = fully_connected(&x_flat, fc2_w, multiplier_fc2);
    relu_fc(&mut x_flat);
    
    println!("FC3...");
    x_flat = fully_connected(&x_flat, fc3_w, multiplier_fc3);
    
    println!("VGG16 inference completed!");
    x_flat
}

fn main() {
    let mut rng = rand::thread_rng();

    println!("VGG16 testing. use null parameters and input just to benchmark performance");
 
    // Convert your u8 input to f32 for processing
    let x_u8 = rand_4d_vec_generator(1, 3, 32, 32);
    let x: Vec<Vec<Vec<Vec<f32>>>> = x_u8.iter()
        .map(|b| b.iter()
            .map(|c| c.iter()
                .map(|h| h.iter()
                    .map(|&w| w as f32 / 255.0) // Normalize to [0,1]
                    .collect())
                .collect())
            .collect())
        .collect();

    // Your existing weight initialization code
    let conv11_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(16,3, 3,3);
    let conv12_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(16,16, 3,3);

    let conv21_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32,16, 3,3);
    let conv22_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32,32, 3,3);

    let conv31_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64,32, 3,3);
    let conv32_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64,64, 3,3);
    let conv33_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64,64, 1,1);

    let conv41_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128,64, 3,3);
    let conv42_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128,128, 3,3);
    let conv43_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128,128, 1,1);

    let conv51_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128,128, 3,3);
    let conv52_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128,128, 3,3);
    let conv53_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128,128, 1,1);

    // Multipliers
    let multiplier_conv11: Vec<f32> = vec![1.5; 16];
    let multiplier_conv12: Vec<f32> = vec![1.5; 16];

    let multiplier_conv21: Vec<f32> = vec![1.5; 32];
    let multiplier_conv22: Vec<f32> = vec![1.5; 32];

    let multiplier_conv31: Vec<f32> = vec![1.5; 64];
    let multiplier_conv32: Vec<f32> = vec![1.5; 64];
    let multiplier_conv33: Vec<f32> = vec![1.5; 64];

    let multiplier_conv41: Vec<f32> = vec![1.5; 128];
    let multiplier_conv42: Vec<f32> = vec![1.5; 128];
    let multiplier_conv43: Vec<f32> = vec![1.5; 128];

    let multiplier_conv51: Vec<f32> = vec![1.5; 128];
    let multiplier_conv52: Vec<f32> = vec![1.5; 128];
    let multiplier_conv53: Vec<f32> = vec![1.5; 128];

    let multiplier_fc1: Vec<f32> = vec![1.5; 64];
    let multiplier_fc2: Vec<f32> = vec![1.5; 32];
    let multiplier_fc3: Vec<f32> = vec![1.5; 10];

    println!("finish reading parameters");

    // First, let's do a dry run to calculate the actual flattened size
    println!("Calculating flattened size...");
    let x_sample = vec![vec![vec![vec![0.0f32; 32]; 32]; 3]; 1];
    
    // Simulate the conv/pool operations to get final dimensions
    let mut x_temp = conv2d(&x_sample, &conv11_w, &multiplier_conv11, 1, 1);
    x_temp = conv2d(&x_temp, &conv12_w, &multiplier_conv12, 1, 1);
    x_temp = max_pool2d(&x_temp);
    
    x_temp = conv2d(&x_temp, &conv21_w, &multiplier_conv21, 1, 1);
    x_temp = conv2d(&x_temp, &conv22_w, &multiplier_conv22, 1, 1);
    x_temp = max_pool2d(&x_temp);
    
    x_temp = conv2d(&x_temp, &conv31_w, &multiplier_conv31, 1, 1);
    x_temp = conv2d(&x_temp, &conv32_w, &multiplier_conv32, 1, 1);
    x_temp = conv2d(&x_temp, &conv33_w, &multiplier_conv33, 1, 1);
    x_temp = max_pool2d(&x_temp);
    
    x_temp = conv2d(&x_temp, &conv41_w, &multiplier_conv41, 1, 1);
    x_temp = conv2d(&x_temp, &conv42_w, &multiplier_conv42, 1, 1);
    x_temp = conv2d(&x_temp, &conv43_w, &multiplier_conv43, 1, 1);
    x_temp = max_pool2d(&x_temp);
    
    x_temp = conv2d(&x_temp, &conv51_w, &multiplier_conv51, 1, 1);
    x_temp = conv2d(&x_temp, &conv52_w, &multiplier_conv52, 1, 1);
    x_temp = conv2d(&x_temp, &conv53_w, &multiplier_conv53, 1, 1);
    x_temp = max_pool2d(&x_temp);
    
    let flattened_temp = flatten(&x_temp);
    let flattened_size = flattened_temp[0].len();
    println!("Calculated flattened size: {}", flattened_size);
    
    // Now create FC layers with correct dimensions
    let fc1_w: Vec<Vec<u8>> = rand_2d_vec_generator(64, flattened_size);
    let fc2_w: Vec<Vec<u8>> = rand_2d_vec_generator(32, 64);
    let fc3_w: Vec<Vec<u8>> = rand_2d_vec_generator(10, 32);
    let start_time = std::time::Instant::now();
    
    let output = vgg16_inference(
        x,
        &conv11_w, &conv12_w, &conv21_w, &conv22_w,
        &conv31_w, &conv32_w, &conv33_w,
        &conv41_w, &conv42_w, &conv43_w,
        &conv51_w, &conv52_w, &conv53_w,
        &fc1_w, &fc2_w, &fc3_w,
        &multiplier_conv11, &multiplier_conv12,
        &multiplier_conv21, &multiplier_conv22,
        &multiplier_conv31, &multiplier_conv32, &multiplier_conv33,
        &multiplier_conv41, &multiplier_conv42, &multiplier_conv43,
        &multiplier_conv51, &multiplier_conv52, &multiplier_conv53,
        &multiplier_fc1, &multiplier_fc2, &multiplier_fc3,
    );
    
    let duration = start_time.elapsed();
    
    println!("Inference completed in: {:?}", duration);
    println!("Output shape: [batch_size: {}, num_classes: {}]", output.len(), output[0].len());
    println!("Output sample: {:?}", &output[0][..output[0].len().min(5)]);
}