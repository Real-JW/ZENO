use std::time::Instant;

// Helper function to generate random 4D vectors
fn rand_4d_vec_generator(a: usize, b: usize, c: usize, d: usize) -> Vec<Vec<Vec<Vec<u8>>>> {
    vec![vec![vec![vec![128; d]; c]; b]; a]
}

// Helper function to generate random 2D vectors
fn rand_2d_vec_generator(a: usize, b: usize) -> Vec<Vec<u8>> {
    vec![vec![128; b]; a]
}

// Basic convolution operation for u8 tensors
fn conv2d_u8(
    input: &Vec<Vec<Vec<Vec<u8>>>>,
    kernel: &Vec<Vec<Vec<Vec<u8>>>>,
    stride: usize,
    padding: usize,
    output_zero_point: u8,
    multiplier: &Vec<f32>,
) -> Vec<Vec<Vec<Vec<u8>>>> {
    // TODO: Implement convolution logic
    // For now, just return a tensor of the correct shape filled with output_zero_point
    let batch_size = input.len();
    let out_channels = kernel.len();
    let input_height = input[0][0].len();
    let input_width = input[0][0][0].len();
    let output_height = input_height; // Placeholder
    let output_width = input_width;   // Placeholder
    vec![vec![vec![vec![output_zero_point; output_width]; output_height]; out_channels]; batch_size]
}

// Add initial 7x7 conv and maxpool
fn conv2d_7x7_u8(
    input: &Vec<Vec<Vec<Vec<u8>>>>,
    kernel: &Vec<Vec<Vec<Vec<u8>>>>,
    stride: usize,
    padding: usize,
    output_zero_point: u8,
    multiplier: &Vec<f32>,
) -> Vec<Vec<Vec<Vec<u8>>>> {
    // Placeholder for 7x7 conv, use conv2d_u8 for now
    conv2d_u8(input, kernel, stride, padding, output_zero_point, multiplier)
}

// ReLU activation for u8 tensors
fn relu_u8(input: &Vec<Vec<Vec<Vec<u8>>>>, zero_point: u8) -> Vec<Vec<Vec<Vec<u8>>>> {
    input.iter()
        .map(|batch| batch.iter()
            .map(|channel| channel.iter()
                .map(|row| row.iter()
                    .map(|&val| if val > zero_point { val } else { zero_point })
                    .collect())
                .collect())
            .collect())
        .collect()
}

// Max pooling operation
fn max_pool2d_u8(input: &Vec<Vec<Vec<Vec<u8>>>>, kernel_size: usize, stride: usize) -> Vec<Vec<Vec<Vec<u8>>>> {
    let batch_size = input.len();
    let channels = input[0].len();
    let input_height = input[0][0].len();
    let input_width = input[0][0][0].len();
    
    let output_height = (input_height - kernel_size) / stride + 1;
    let output_width = (input_width - kernel_size) / stride + 1;
    
    let mut output = vec![vec![vec![vec![0u8; output_width]; output_height]; channels]; batch_size];
    
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let mut max_val = 0u8;
                    
                    for kh in 0..kernel_size {
                        for kw in 0..kernel_size {
                            let ih = oh * stride + kh;
                            let iw = ow * stride + kw;
                            
                            if ih < input_height && iw < input_width {
                                max_val = max_val.max(input[b][c][ih][iw]);
                            }
                        }
                    }
                    
                    output[b][c][oh][ow] = max_val;
                }
            }
        }
    }
    
    output
}

// Average pooling operation
fn avg_pool2d_u8(input: &Vec<Vec<Vec<Vec<u8>>>>, kernel_size: usize) -> Vec<Vec<Vec<Vec<u8>>>> {
    let batch_size = input.len();
    let channels = input[0].len();
    let input_height = input[0][0].len();
    let input_width = input[0][0][0].len();
    
    let output_height = input_height / kernel_size;
    let output_width = input_width / kernel_size;
    
    let mut output = vec![vec![vec![vec![0u8; output_width]; output_height]; channels]; batch_size];
    
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let mut sum = 0u32;
                    
                    for kh in 0..kernel_size {
                        for kw in 0..kernel_size {
                            let ih = oh * kernel_size + kh;
                            let iw = ow * kernel_size + kw;
                            sum += input[b][c][ih][iw] as u32;
                        }
                    }
                    
                    output[b][c][oh][ow] = (sum / (kernel_size * kernel_size) as u32) as u8;
                }
            }
        }
    }
    
    output
}

// Element-wise addition for residual connections
fn add_tensors_u8(
    a: &Vec<Vec<Vec<Vec<u8>>>>,
    b: &Vec<Vec<Vec<Vec<u8>>>>,
    output_zero: u8,
    multiplier_a: &Vec<f32>,
    multiplier_b: &Vec<f32>,
) -> Vec<Vec<Vec<Vec<u8>>>> {
    let batch_size = a.len();
    let channels = a[0].len();
    let height = a[0][0].len();
    let width = a[0][0][0].len();
    
    let mut output = vec![vec![vec![vec![output_zero; width]; height]; channels]; batch_size];
    
    for b_idx in 0..batch_size {
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let val_a = (a[b_idx][c][h][w] as f32) * multiplier_a[c];
                    let val_b = (b[b_idx][c][h][w] as f32) * multiplier_b[c];
                    let sum = val_a + val_b;
                    output[b_idx][c][h][w] = (sum as i32).max(0).min(255) as u8;
                }
            }
        }
    }
    
    output
}

// Fully connected layer
fn fully_connected_u8(
    input: &Vec<Vec<Vec<Vec<u8>>>>,
    weights: &Vec<Vec<u8>>,
    output_zero_point: u8,
    multiplier: &Vec<f32>,
) -> Vec<Vec<u8>> {
    let batch_size = input.len();
    let flattened_size = input[0].len() * input[0][0].len() * input[0][0][0].len();
    let output_size = weights.len();
    
    let mut output = vec![vec![output_zero_point; output_size]; batch_size];
    
    for b in 0..batch_size {
        // Flatten input
        let mut flattened = Vec::new();
        for c in 0..input[0].len() {
            for h in 0..input[0][0].len() {
                for w in 0..input[0][0][0].len() {
                    flattened.push(input[b][c][h][w]);
                }
            }
        }
        
        for o in 0..output_size {
            let mut sum = 0i32;
            for i in 0..flattened_size.min(weights[o].len()) {
                sum += (flattened[i] as i32) * (weights[o][i] as i32);
            }
            
            let scaled = (sum as f32) * multiplier[o.min(multiplier.len() - 1)];
            output[b][o] = (scaled as i32).max(0).min(255) as u8;
        }
    }
    
    output
}

// ResNet-18 basic block
fn resnet_basic_block(
    input: &Vec<Vec<Vec<Vec<u8>>>>,
    conv1_weights: &Vec<Vec<Vec<Vec<u8>>>>,
    conv2_weights: &Vec<Vec<Vec<Vec<u8>>>>,
    conv1_multiplier: &Vec<f32>,
    conv2_multiplier: &Vec<f32>,
    conv1_output_zero: u8,
    conv2_output_zero: u8,
    add_output_zero: u8,
    add_multiplier_a: &Vec<f32>,
    add_multiplier_b: &Vec<f32>,
    stride: usize,
) -> Vec<Vec<Vec<Vec<u8>>>> {
    // First convolution
    let conv1_out = conv2d_u8(input, conv1_weights, stride, 1, conv1_output_zero, conv1_multiplier);
    let relu1_out = relu_u8(&conv1_out, conv1_output_zero);
    
    // Second convolution
    let conv2_out = conv2d_u8(&relu1_out, conv2_weights, 1, 1, conv2_output_zero, conv2_multiplier);

    // Residual connection: shape match
    let mut residual = if stride == 1 && input[0].len() == conv2_out[0].len()
    {
        input.clone()
    } else {
        // Downsample or zero-pad/crop channels as needed
        let batch = conv2_out.len();
        let channels = conv2_out[0].len();
        let height = conv2_out[0][0].len();
        let width = conv2_out[0][0][0].len();
        let mut res = vec![vec![vec![vec![add_output_zero; width]; height]; channels]; batch];
        for b in 0..batch.min(input.len()) {
            for c in 0..channels.min(input[0].len()) {
                for h in 0..height.min(input[0][0].len()) {
                    for w in 0..width.min(input[0][0][0].len()) {
                        res[b][c][h][w] = input[b][c][h][w];
                    }
                }
            }
        }
        res
    };
    
    println!("ResNet basic block completed with stride {}", stride);

    // Add residual
    let add_out = add_tensors_u8(&conv2_out, &residual, add_output_zero, add_multiplier_a, add_multiplier_b);
    relu_u8(&add_out, add_output_zero)

}

// Main ResNet-18 forward function
pub fn resnet18_circuit_forward_u8(
    x: Vec<Vec<Vec<Vec<u8>>>>,
    // Initial conv weights
    conv1_w: Vec<Vec<Vec<Vec<u8>>>>,
    // Stage 1
    conv2_1_w: Vec<Vec<Vec<Vec<u8>>>>, conv2_2_w: Vec<Vec<Vec<Vec<u8>>>>,
    // Stage 2
    conv3_1_w: Vec<Vec<Vec<Vec<u8>>>>, conv3_2_w: Vec<Vec<Vec<Vec<u8>>>>,
    // Stage 3
    conv4_1_w: Vec<Vec<Vec<Vec<u8>>>>, conv4_2_w: Vec<Vec<Vec<Vec<u8>>>>,
    // Stage 4
    conv5_1_w: Vec<Vec<Vec<Vec<u8>>>>, conv5_2_w: Vec<Vec<Vec<Vec<u8>>>>,
    // FC weights
    fc_w: Vec<Vec<u8>>,
    // Multipliers and zero points (simplified for brevity)
    conv1_multiplier: Vec<f32>, conv2_1_multiplier: Vec<f32>, conv2_2_multiplier: Vec<f32>,
    conv3_1_multiplier: Vec<f32>, conv3_2_multiplier: Vec<f32>,
    conv4_1_multiplier: Vec<f32>, conv4_2_multiplier: Vec<f32>,
    conv5_1_multiplier: Vec<f32>, conv5_2_multiplier: Vec<f32>,
    fc_multiplier: Vec<f32>,
    conv1_output_0: u8, conv2_1_output_0: u8, conv2_2_output_0: u8,
    conv3_1_output_0: u8, conv3_2_output_0: u8,
    conv4_1_output_0: u8, conv4_2_output_0: u8,
    conv5_1_output_0: u8, conv5_2_output_0: u8,
    fc_output_0: u8,
) -> Vec<Vec<u8>> {
    // Initial 7x7 conv, stride 2, padding 3
    let mut current = conv2d_7x7_u8(&x, &conv1_w, 2, 3, conv1_output_0, &conv1_multiplier);
    // MaxPool 3x3, stride 2
    current = max_pool2d_u8(&current, 3, 2);

    // Stage 1: 64 channels, 2 blocks
    current = resnet_basic_block(&current, &conv2_1_w, &conv2_2_w, &conv2_1_multiplier, &conv2_2_multiplier, conv2_1_output_0, conv2_2_output_0, conv2_2_output_0, &vec![1.0; 64], &vec![1.0; 64], 1);
    current = resnet_basic_block(&current, &conv2_1_w, &conv2_2_w, &conv2_1_multiplier, &conv2_2_multiplier, conv2_1_output_0, conv2_2_output_0, conv2_2_output_0, &vec![1.0; 64], &vec![1.0; 64], 1);

    // Stage 2: 128 channels, 2 blocks (first block stride 2)
    current = resnet_basic_block(&current, &conv3_1_w, &conv3_2_w, &conv3_1_multiplier, &conv3_2_multiplier, conv3_1_output_0, conv3_2_output_0, conv3_2_output_0, &vec![1.0; 128], &vec![1.0; 128], 2);
    current = resnet_basic_block(&current, &conv3_1_w, &conv3_2_w, &conv3_1_multiplier, &conv3_2_multiplier, conv3_1_output_0, conv3_2_output_0, conv3_2_output_0, &vec![1.0; 128], &vec![1.0; 128], 1);

    // Stage 3: 256 channels, 2 blocks (first block stride 2)
    current = resnet_basic_block(&current, &conv4_1_w, &conv4_2_w, &conv4_1_multiplier, &conv4_2_multiplier, conv4_1_output_0, conv4_2_output_0, conv4_2_output_0, &vec![1.0; 256], &vec![1.0; 256], 2);
    current = resnet_basic_block(&current, &conv4_1_w, &conv4_2_w, &conv4_1_multiplier, &conv4_2_multiplier, conv4_1_output_0, conv4_2_output_0, conv4_2_output_0, &vec![1.0; 256], &vec![1.0; 256], 1);

    // Stage 4: 512 channels, 2 blocks (first block stride 2)
    current = resnet_basic_block(&current, &conv5_1_w, &conv5_2_w, &conv5_1_multiplier, &conv5_2_multiplier, conv5_1_output_0, conv5_2_output_0, conv5_2_output_0, &vec![1.0; 512], &vec![1.0; 512], 2);
    current = resnet_basic_block(&current, &conv5_1_w, &conv5_2_w, &conv5_1_multiplier, &conv5_2_multiplier, conv5_1_output_0, conv5_2_output_0, conv5_2_output_0, &vec![1.0; 512], &vec![1.0; 512], 1);

    // Global average pooling
    let pooled = avg_pool2d_u8(&current, current[0][0].len());
    // FC
    fully_connected_u8(&pooled, &fc_w, fc_output_0, &fc_multiplier)
}

fn main() {
    println!("RESNET18 testing - using null parameters and input just to benchmark performance");
    
    let start = Instant::now();

    // Generate test data
    let x = rand_4d_vec_generator(1, 3, 224, 224); // Standard ImageNet size
    // Initial conv weights
    let conv1_w = rand_4d_vec_generator(64, 3, 7, 7);
    // Stage 1
    let conv2_1_w = rand_4d_vec_generator(64, 64, 3, 3);
    let conv2_2_w = rand_4d_vec_generator(64, 64, 3, 3);
    // Stage 2
    let conv3_1_w = rand_4d_vec_generator(128, 64, 3, 3);
    let conv3_2_w = rand_4d_vec_generator(128, 128, 3, 3);
    // Stage 3
    let conv4_1_w = rand_4d_vec_generator(256, 128, 3, 3);
    let conv4_2_w = rand_4d_vec_generator(256, 256, 3, 3);
    // Stage 4
    let conv5_1_w = rand_4d_vec_generator(512, 256, 3, 3);
    let conv5_2_w = rand_4d_vec_generator(512, 512, 3, 3);
    // FC
    let fc_w = rand_2d_vec_generator(1000, 512); // 1000 classes
    // Multipliers and zero points
    let conv1_multiplier = vec![1.0; 64];
    let conv2_1_multiplier = vec![1.0; 64];
    let conv2_2_multiplier = vec![1.0; 64];
    let conv3_1_multiplier = vec![1.0; 128];
    let conv3_2_multiplier = vec![1.0; 128];
    let conv4_1_multiplier = vec![1.0; 256];
    let conv4_2_multiplier = vec![1.0; 256];
    let conv5_1_multiplier = vec![1.0; 512];
    let conv5_2_multiplier = vec![1.0; 512];
    let fc_multiplier = vec![1.0; 1000];
    let conv1_output_0 = 128;
    let conv2_1_output_0 = 128;
    let conv2_2_output_0 = 128;
    let conv3_1_output_0 = 128;
    let conv3_2_output_0 = 128;
    let conv4_1_output_0 = 128;
    let conv4_2_output_0 = 128;
    let conv5_1_output_0 = 128;
    let conv5_2_output_0 = 128;
    let fc_output_0 = 128;
    let start = Instant::now();
    let z = resnet18_circuit_forward_u8(
        x,
        conv1_w,
        conv2_1_w, conv2_2_w,
        conv3_1_w, conv3_2_w,
        conv4_1_w, conv4_2_w,
        conv5_1_w, conv5_2_w,
        fc_w,
        conv1_multiplier, conv2_1_multiplier, conv2_2_multiplier,
        conv3_1_multiplier, conv3_2_multiplier,
        conv4_1_multiplier, conv4_2_multiplier,
        conv5_1_multiplier, conv5_2_multiplier,
        fc_multiplier,
        conv1_output_0, conv2_1_output_0, conv2_2_output_0,
        conv3_1_output_0, conv3_2_output_0,
        conv4_1_output_0, conv4_2_output_0,
        conv5_1_output_0, conv5_2_output_0,
        fc_output_0,
    );
    let duration = start.elapsed();
    println!("Inference time: {:?}", duration);
    println!("Output shape: {} x {}", z.len(), z[0].len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_basic() {
        let input = vec![vec![vec![vec![1u8, 2u8], vec![3u8, 4u8]]]];
        let kernel = vec![vec![vec![vec![1u8, 0u8], vec![0u8, 1u8]]]];
        let multiplier = vec![1.0f32];
        
        let output = conv2d_u8(&input, &kernel, 1, 0, 0, &multiplier);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 1);
    }

    #[test]
    fn test_relu_activation() {
        let input = vec![vec![vec![vec![100u8, 200u8], vec![50u8, 150u8]]]];
        let zero_point = 128u8;
        
        let output = relu_u8(&input, zero_point);
        assert_eq!(output[0][0][0][0], 128); // 100 < 128, so clamped to 128
        assert_eq!(output[0][0][0][1], 200); // 200 > 128, so kept as 200
        assert_eq!(output[0][0][1][0], 128); // 50 < 128, so clamped to 128
        assert_eq!(output[0][0][1][1], 150); // 150 > 128, so kept as 150
    }

    #[test]
    fn test_max_pool2d() {
        let input = vec![vec![vec![vec![1u8, 2u8, 3u8, 4u8], 
                                  vec![5u8, 6u8, 7u8, 8u8],
                                  vec![9u8, 10u8, 11u8, 12u8],
                                  vec![13u8, 14u8, 15u8, 16u8]]]];
        
        let output = max_pool2d_u8(&input, 2, 2);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 1);
        assert_eq!(output[0][0].len(), 2);
        assert_eq!(output[0][0][0].len(), 2);
        assert_eq!(output[0][0][0][0], 6);  // max of [1,2,5,6]
        assert_eq!(output[0][0][0][1], 8);  // max of [3,4,7,8]
        assert_eq!(output[0][0][1][0], 14); // max of [9,10,13,14]
        assert_eq!(output[0][0][1][1], 16); // max of [11,12,15,16]
    }

    #[test]
    fn test_avg_pool2d() {
        let input = vec![vec![vec![vec![2u8, 2u8, 6u8, 6u8], 
                                  vec![2u8, 2u8, 6u8, 6u8],
                                  vec![10u8, 10u8, 14u8, 14u8],
                                  vec![10u8, 10u8, 14u8, 14u8]]]];
        
        let output = avg_pool2d_u8(&input, 2);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 1);
        assert_eq!(output[0][0].len(), 2);
        assert_eq!(output[0][0][0].len(), 2);
        assert_eq!(output[0][0][0][0], 2);  // avg of [2,2,2,2]
        assert_eq!(output[0][0][0][1], 6);  // avg of [6,6,6,6]
        assert_eq!(output[0][0][1][0], 10); // avg of [10,10,10,10]
        assert_eq!(output[0][0][1][1], 14); // avg of [14,14,14,14]
    }

    #[test]
    fn test_tensor_addition() {
        let a = vec![vec![vec![vec![100u8, 120u8]]]];
        let b = vec![vec![vec![vec![80u8, 60u8]]]];
        let multiplier_a = vec![1.0f32];
        let multiplier_b = vec![1.0f32];
        
        let output = add_tensors_u8(&a, &b, 0, &multiplier_a, &multiplier_b);
        assert_eq!(output[0][0][0][0], 180); // 100 + 80
        assert_eq!(output[0][0][0][1], 180); // 120 + 60
    }

    #[test]
    fn test_fully_connected() {
        let input = vec![vec![vec![vec![1u8, 2u8], vec![3u8, 4u8]]]]; // 1x1x2x2
        let weights = vec![vec![1u8, 1u8, 1u8, 1u8], vec![2u8, 2u8, 2u8, 2u8]]; // 2x4
        let multiplier = vec![1.0f32, 1.0f32];
        
        let output = fully_connected_u8(&input, &weights, 0, &multiplier);
        assert_eq!(output.len(), 1); // batch size
        assert_eq!(output[0].len(), 2); // output features
        assert_eq!(output[0][0], 10); // 1*1 + 2*1 + 3*1 + 4*1 = 10
        assert_eq!(output[0][1], 20); // (1*2 + 2*2 + 3*2 + 4*2) = 20
    }
}

// Benchmarking utilities
pub struct ResNet18Benchmark {
    pub input_size: (usize, usize, usize, usize), // (batch, channels, height, width)
    pub num_iterations: usize,
}

impl ResNet18Benchmark {
    pub fn new(batch_size: usize, channels: usize, height: usize, width: usize, iterations: usize) -> Self {
        Self {
            input_size: (batch_size, channels, height, width),
            num_iterations: iterations,
        }
    }

    pub fn run_benchmark(&self) -> std::time::Duration {
        println!("Running ResNet-18 benchmark:");
        println!("  Input size: {:?}", self.input_size);
        println!("  Iterations: {}", self.num_iterations);

        let (batch_size, channels, height, width) = self.input_size;
        
        // Generate test data once
        let x = rand_4d_vec_generator(batch_size, channels, height, width);
        
        // Generate weights (simplified for benchmark)
        let conv_weights = rand_4d_vec_generator(64, channels, 3, 3);
        let fc_weights = rand_2d_vec_generator(10, 128);
        let multipliers = vec![1.5; 64];
        let fc_multipliers = vec![1.5; 10];

        let start = Instant::now();
        
        for i in 0..self.num_iterations {
            if i % 10 == 0 {
                println!("  Iteration {}/{}", i, self.num_iterations);
            }
            
            // Simulate a simplified forward pass for benchmarking
            let conv_out = conv2d_u8(&x, &conv_weights, 1, 1, 128, &multipliers);
            let relu_out = relu_u8(&conv_out, 128);
            let pool_out = max_pool2d_u8(&relu_out, 2, 2);
            let _fc_out = fully_connected_u8(&pool_out, &fc_weights, 128, &fc_multipliers);
        }
        
        let total_duration = start.elapsed();
        let avg_duration = total_duration / self.num_iterations as u32;
        
        println!("Benchmark Results:");
        println!("  Total time: {:?}", total_duration);
        println!("  Average time per iteration: {:?}", avg_duration);
        println!("  Throughput: {:.2} images/sec", 
                 batch_size as f64 * self.num_iterations as f64 / total_duration.as_secs_f64());
        
        avg_duration
    }
}

// Additional utility functions for performance analysis
pub fn profile_layer_performance() {
    println!("\n=== Layer Performance Profiling ===");
    
    let input = rand_4d_vec_generator(1, 3, 32, 32);
    let kernel = rand_4d_vec_generator(16, 3, 3, 3);
    let multiplier = vec![1.5; 16];
    
    // Profile convolution
    let start = Instant::now();
    let conv_out = conv2d_u8(&input, &kernel, 1, 1, 128, &multiplier);
    let conv_time = start.elapsed();
    println!("Convolution (3->16 channels, 3x3 kernel): {:?}", conv_time);
    
    // Profile ReLU
    let start = Instant::now();
    let relu_out = relu_u8(&conv_out, 128);
    let relu_time = start.elapsed();
    println!("ReLU activation: {:?}", relu_time);
    
    // Profile MaxPool
    let start = Instant::now();
    let pool_out = max_pool2d_u8(&relu_out, 2, 2);
    let pool_time = start.elapsed();
    println!("Max pooling (2x2): {:?}", pool_time);
    
    // Profile FC layer
    let fc_weights = rand_2d_vec_generator(10, 128);
    let fc_multiplier = vec![1.5; 10];
    let start = Instant::now();
    let _fc_out = fully_connected_u8(&pool_out, &fc_weights, 128, &fc_multiplier);
    let fc_time = start.elapsed();
    println!("Fully connected layer: {:?}", fc_time);
    
    println!("Total estimated time: {:?}", conv_time + relu_time + pool_time + fc_time);
}

// Memory usage estimation
pub fn estimate_memory_usage(batch_size: usize, input_channels: usize, height: usize, width: usize) {
    println!("\n=== Memory Usage Estimation ===");
    
    let input_size = batch_size * input_channels * height * width;
    println!("Input tensor: {} bytes ({} MB)", input_size, input_size / (1024 * 1024));
    
    // Estimate intermediate activations for ResNet-18
    let conv1_size = batch_size * 64 * height * width;
    let conv2_size = batch_size * 64 * (height / 2) * (width / 2);
    let conv3_size = batch_size * 128 * (height / 4) * (width / 4);
    let conv4_size = batch_size * 256 * (height / 8) * (width / 8);
    let conv5_size = batch_size * 512 * (height / 16) * (width / 16);
    
    let total_activation_size = input_size + conv1_size + conv2_size + conv3_size + conv4_size + conv5_size;
    println!("Estimated activation memory: {} bytes ({} MB)", 
             total_activation_size, total_activation_size / (1024 * 1024));
    
    // Estimate weight memory (ResNet-18 has ~11M parameters)
    let weight_size = 11_000_000; // approximate number of parameters
    println!("Weight memory: {} bytes ({} MB)", weight_size, weight_size / (1024 * 1024));
    
    let total_memory = total_activation_size + weight_size;
    println!("Total estimated memory: {} bytes ({} MB)", total_memory, total_memory / (1024 * 1024));
}