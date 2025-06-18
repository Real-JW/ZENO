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
    padding: usize,
    x: Vec<Vec<Vec<Vec<u8>>>>,
    
    // Layer 2 weights
    conv21_w: Vec<Vec<Vec<Vec<u8>>>>,
    conv22_w: Vec<Vec<Vec<Vec<u8>>>>,
    conv23_w: Vec<Vec<Vec<Vec<u8>>>>,
    conv24_w: Vec<Vec<Vec<Vec<u8>>>>,
    
    // Layer 3 weights
    conv31_w: Vec<Vec<Vec<Vec<u8>>>>,
    conv32_w: Vec<Vec<Vec<Vec<u8>>>>,
    conv33_w: Vec<Vec<Vec<Vec<u8>>>>,
    conv34_w: Vec<Vec<Vec<Vec<u8>>>>,
    
    // Layer 4 weights
    conv41_w: Vec<Vec<Vec<Vec<u8>>>>,
    conv42_w: Vec<Vec<Vec<Vec<u8>>>>,
    conv43_w: Vec<Vec<Vec<Vec<u8>>>>,
    conv44_w: Vec<Vec<Vec<Vec<u8>>>>,
    
    // Layer 5 weights
    conv51_w: Vec<Vec<Vec<Vec<u8>>>>,
    conv52_w: Vec<Vec<Vec<Vec<u8>>>>,
    conv53_w: Vec<Vec<Vec<Vec<u8>>>>,
    conv54_w: Vec<Vec<Vec<Vec<u8>>>>,
    
    // Residual connection weights
    conv_residual1_weight: Vec<Vec<Vec<Vec<u8>>>>,
    conv_residual3_weight: Vec<Vec<Vec<Vec<u8>>>>,
    conv_residual5_weight: Vec<Vec<Vec<Vec<u8>>>>,
    conv_residual7_weight: Vec<Vec<Vec<Vec<u8>>>>,
    
    // FC weights
    fc1_w: Vec<Vec<u8>>,
    
    // Zero points
    x_0: u8,
    conv21_output_0: u8, conv22_output_0: u8, conv23_output_0: u8, conv24_output_0: u8,
    conv31_output_0: u8, conv32_output_0: u8, conv33_output_0: u8, conv34_output_0: u8,
    conv41_output_0: u8, conv42_output_0: u8, conv43_output_0: u8, conv44_output_0: u8,
    conv51_output_0: u8, conv52_output_0: u8, conv53_output_0: u8, conv54_output_0: u8,
    conv_residual1_output_0: u8, conv_residual3_output_0: u8, 
    conv_residual5_output_0: u8, conv_residual7_output_0: u8,
    fc1_output_0: u8,
    
    // Weight zero points
    conv21_weights_0: u8, conv22_weights_0: u8, conv23_weights_0: u8, conv24_weights_0: u8,
    conv31_weights_0: u8, conv32_weights_0: u8, conv33_weights_0: u8, conv34_weights_0: u8,
    conv41_weights_0: u8, conv42_weights_0: u8, conv43_weights_0: u8, conv44_weights_0: u8,
    conv51_weights_0: u8, conv52_weights_0: u8, conv53_weights_0: u8, conv54_weights_0: u8,
    conv_residual1_weights_0: u8, conv_residual3_weights_0: u8,
    conv_residual5_weights_0: u8, conv_residual7_weights_0: u8,
    fc1_weights_0: u8,
    
    // Multipliers
    multiplier_conv21: Vec<f32>, multiplier_conv22: Vec<f32>, multiplier_conv23: Vec<f32>, multiplier_conv24: Vec<f32>,
    multiplier_conv31: Vec<f32>, multiplier_conv32: Vec<f32>, multiplier_conv33: Vec<f32>, multiplier_conv34: Vec<f32>,
    multiplier_conv41: Vec<f32>, multiplier_conv42: Vec<f32>, multiplier_conv43: Vec<f32>, multiplier_conv44: Vec<f32>,
    multiplier_conv51: Vec<f32>, multiplier_conv52: Vec<f32>, multiplier_conv53: Vec<f32>, multiplier_conv54: Vec<f32>,
    conv_residual1_multiplier: Vec<f32>, conv_residual3_multiplier: Vec<f32>,
    conv_residual5_multiplier: Vec<f32>, conv_residual7_multiplier: Vec<f32>,
    
    // Residual add parameters
    add_residual1_output_0: u8, add_residual2_output_0: u8, add_residual3_output_0: u8, add_residual4_output_0: u8,
    add_residual5_output_0: u8, add_residual6_output_0: u8, add_residual7_output_0: u8, add_residual8_output_0: u8,
    
    add_residual1_first_multiplier: Vec<f32>, add_residual2_first_multiplier: Vec<f32>,
    add_residual3_first_multiplier: Vec<f32>, add_residual4_first_multiplier: Vec<f32>,
    add_residual5_first_multiplier: Vec<f32>, add_residual6_first_multiplier: Vec<f32>,
    add_residual7_first_multiplier: Vec<f32>, add_residual8_first_multiplier: Vec<f32>,
    
    add_residual1_second_multiplier: Vec<f32>, add_residual2_second_multiplier: Vec<f32>,
    add_residual3_second_multiplier: Vec<f32>, add_residual4_second_multiplier: Vec<f32>,
    add_residual5_second_multiplier: Vec<f32>, add_residual6_second_multiplier: Vec<f32>,
    add_residual7_second_multiplier: Vec<f32>, add_residual8_second_multiplier: Vec<f32>,
    
    multiplier_fc1: Vec<f32>,
) -> Vec<Vec<u8>> {
    
    // Initial convolution (simulated as identity for simplicity)
    let mut current = x;
    
    // Layer 2 (ResNet blocks)
    current = resnet_basic_block(
        &current, &conv21_w, &conv22_w, 
        &multiplier_conv21, &multiplier_conv22,
        conv21_output_0, conv22_output_0, add_residual1_output_0,
        &add_residual1_first_multiplier, &add_residual1_second_multiplier, 1
    );
    
    current = resnet_basic_block(
        &current, &conv23_w, &conv24_w,
        &multiplier_conv23, &multiplier_conv24,
        conv23_output_0, conv24_output_0, add_residual2_output_0,
        &add_residual2_first_multiplier, &add_residual2_second_multiplier, 1
    );
    
    // Layer 3 (ResNet blocks with stride 2)
    current = resnet_basic_block(
        &current, &conv31_w, &conv32_w,
        &multiplier_conv31, &multiplier_conv32,
        conv31_output_0, conv32_output_0, add_residual3_output_0,
        &add_residual3_first_multiplier, &add_residual3_second_multiplier, 2
    );
    
    current = resnet_basic_block(
        &current, &conv33_w, &conv34_w,
        &multiplier_conv33, &multiplier_conv34,
        conv33_output_0, conv34_output_0, add_residual4_output_0,
        &add_residual4_first_multiplier, &add_residual4_second_multiplier, 1
    );
    
    // Layer 4 (ResNet blocks with stride 2)
    current = resnet_basic_block(
        &current, &conv41_w, &conv42_w,
        &multiplier_conv41, &multiplier_conv42,
        conv41_output_0, conv42_output_0, add_residual5_output_0,
        &add_residual5_first_multiplier, &add_residual5_second_multiplier, 2
    );
    
    current = resnet_basic_block(
        &current, &conv43_w, &conv44_w,
        &multiplier_conv43, &multiplier_conv44,
        conv43_output_0, conv44_output_0, add_residual6_output_0,
        &add_residual6_first_multiplier, &add_residual6_second_multiplier, 1
    );
    
    // Layer 5 (ResNet blocks with stride 2)
    current = resnet_basic_block(
        &current, &conv51_w, &conv52_w,
        &multiplier_conv51, &multiplier_conv52,
        conv51_output_0, conv52_output_0, add_residual7_output_0,
        &add_residual7_first_multiplier, &add_residual7_second_multiplier, 2
    );
    
    current = resnet_basic_block(
        &current, &conv53_w, &conv54_w,
        &multiplier_conv53, &multiplier_conv54,
        conv53_output_0, conv54_output_0, add_residual8_output_0,
        &add_residual8_first_multiplier, &add_residual8_second_multiplier, 1
    );
    
    // Global average pooling
    let pooled = avg_pool2d_u8(&current, current[0][0].len());
    
    // Fully connected layer
    fully_connected_u8(&pooled, &fc1_w, fc1_output_0, &multiplier_fc1)
}

fn main() {
    println!("RESNET18 testing - using null parameters and input just to benchmark performance");
    
    let start = Instant::now();

    // Generate test data
    let x = rand_4d_vec_generator(1, 3, 32, 32);

    // Layer 2 weights
    let conv21_w = rand_4d_vec_generator(16, 3, 3, 3);
    let conv22_w = rand_4d_vec_generator(16, 16, 3, 3);
    let conv23_w = rand_4d_vec_generator(16, 16, 3, 3);
    let conv24_w = rand_4d_vec_generator(16, 16, 3, 3);
    let multiplier_conv21 = vec![1.5; 16];
    let multiplier_conv22 = vec![1.5; 16];
    let multiplier_conv23 = vec![1.5; 16];
    let multiplier_conv24 = vec![1.5; 16];

    // Layer 3 weights
    let conv31_w = rand_4d_vec_generator(32, 16, 3, 3);
    let conv32_w = rand_4d_vec_generator(32, 32, 3, 3);
    let conv33_w = rand_4d_vec_generator(32, 32, 3, 3);
    let conv34_w = rand_4d_vec_generator(32, 32, 3, 3);
    let multiplier_conv31 = vec![1.5; 32];
    let multiplier_conv32 = vec![1.5; 32];
    let multiplier_conv33 = vec![1.5; 32];
    let multiplier_conv34 = vec![1.5; 32];

    // Layer 4 weights
    let conv41_w = rand_4d_vec_generator(64, 32, 3, 3);
    let conv42_w = rand_4d_vec_generator(64, 64, 3, 3);
    let conv43_w = rand_4d_vec_generator(64, 64, 3, 3);
    let conv44_w = rand_4d_vec_generator(64, 64, 3, 3);
    let multiplier_conv41 = vec![1.5; 64];
    let multiplier_conv42 = vec![1.5; 64];
    let multiplier_conv43 = vec![1.5; 64];
    let multiplier_conv44 = vec![1.5; 64];

    // Layer 5 weights
    let conv51_w = rand_4d_vec_generator(128, 64, 3, 3);
    let conv52_w = rand_4d_vec_generator(128, 128, 3, 3);
    let conv53_w = rand_4d_vec_generator(128, 128, 3, 3);
    let conv54_w = rand_4d_vec_generator(128, 128, 3, 3);
    let multiplier_conv51 = vec![1.5; 128];
    let multiplier_conv52 = vec![1.5; 128];
    let multiplier_conv53 = vec![1.5; 128];
    let multiplier_conv54 = vec![1.5; 128];

    // Residual connection weights
    let conv_residual1_weight = rand_4d_vec_generator(16, 3, 1, 1);
    let conv_residual3_weight = rand_4d_vec_generator(32, 16, 1, 1);
    let conv_residual5_weight = rand_4d_vec_generator(64, 32, 1, 1);
    let conv_residual7_weight = rand_4d_vec_generator(128, 64, 1, 1);

    // FC layer
    let fc1_w = rand_2d_vec_generator(10, 128);
    let multiplier_fc1 = vec![1.5; 10];

    // Zero points and multipliers for residual additions
    let add_residual_outputs = vec![128u8; 8];
    let add_residual_first_multipliers = vec![
        vec![1.5; 16], vec![1.5; 16], vec![1.5; 32], vec![1.5; 32],
        vec![1.5; 64], vec![1.5; 64], vec![1.5; 128], vec![1.5; 128]
    ];
    let add_residual_second_multipliers = add_residual_first_multipliers.clone();

    println!("Finish reading parameters");
    println!("Before forward pass");

    
    let z = resnet18_circuit_forward_u8(
        1, // padding
        x.clone(),
        
        // Layer weights
        conv21_w, conv22_w, conv23_w, conv24_w,
        conv31_w, conv32_w, conv33_w, conv34_w,
        conv41_w, conv42_w, conv43_w, conv44_w,
        conv51_w, conv52_w, conv53_w, conv54_w,
        
        // Residual weights
        conv_residual1_weight, conv_residual3_weight, 
        conv_residual5_weight, conv_residual7_weight,
        
        fc1_w,
        
        // Zero points
        128, // x_0
        128, 128, 128, 128, // conv2x outputs
        128, 128, 128, 128, // conv3x outputs
        128, 128, 128, 128, // conv4x outputs
        128, 128, 128, 128, // conv5x outputs
        128, 128, 128, 128, // residual outputs
        128, // fc1 output
        
        // Weight zero points
        128, 128, 128, 128, // conv2x weights
        128, 128, 128, 128, // conv3x weights
        128, 128, 128, 128, // conv4x weights
        128, 128, 128, 128, // conv5x weights
        128, 128, 128, 128, // residual weights
        128, // fc1 weights
        
        // Multipliers
        multiplier_conv21, multiplier_conv22, multiplier_conv23, multiplier_conv24,
        multiplier_conv31, multiplier_conv32, multiplier_conv33, multiplier_conv34,
        multiplier_conv41, multiplier_conv42, multiplier_conv43, multiplier_conv44,
        multiplier_conv51, multiplier_conv52, multiplier_conv53, multiplier_conv54,
        vec![1.5; 16], vec![1.5; 32], vec![1.5; 64], vec![1.5; 128], // residual multipliers
        
        // Residual add parameters
        128, 128, 128, 128, 128, 128, 128, 128, // add outputs
        add_residual_first_multipliers[0].clone(), add_residual_first_multipliers[1].clone(),
        add_residual_first_multipliers[2].clone(), add_residual_first_multipliers[3].clone(),
        add_residual_first_multipliers[4].clone(), add_residual_first_multipliers[5].clone(),
        add_residual_first_multipliers[6].clone(), add_residual_first_multipliers[7].clone(),
        add_residual_second_multipliers[0].clone(), add_residual_second_multipliers[1].clone(),
        add_residual_second_multipliers[2].clone(), add_residual_second_multipliers[3].clone(),
        add_residual_second_multipliers[4].clone(), add_residual_second_multipliers[5].clone(),
        add_residual_second_multipliers[6].clone(), add_residual_second_multipliers[7].clone(),
        
        multiplier_fc1,
    );

    let duration = start.elapsed();
    
    println!("Time: {:?}", duration);
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