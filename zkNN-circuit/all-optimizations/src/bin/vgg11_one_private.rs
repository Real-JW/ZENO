use algebra::ed_on_bls12_381::*;
use algebra::CanonicalSerialize;
use algebra::UniformRand;
use crypto_primitives::commitment::pedersen::Randomness;
use groth16::*;
use r1cs_core::*;
use std::time::Instant;
use zk_ml_knit_encoding::vgg_circuit::*;
use zk_ml_knit_encoding::pedersen_commit::*;
use zk_ml_knit_encoding::read_inputs::*;
use zk_ml_knit_encoding::vanilla::*;

fn rand_4d_vec_generator(a : usize, b : usize, c : usize, d:usize) -> Vec<Vec<Vec<Vec<u8>>>>{
    let mut res = vec![vec![vec![vec![2; d]; c]; b]; a];

    res
}

fn rand_2d_vec_generator(a : usize, b : usize) -> Vec<Vec<u8>>{
    let mut res = vec![vec![2; b]; a];

    res
}

fn main() {
    let mut rng = rand::thread_rng();

    println!("VGG11 testing. use null parameters and input just to benchmark performance");
 
    let x = rand_4d_vec_generator(1, 3, 32, 32);
    // let x = rand_4d_vec_generator(1, 1, 64, 64);

    // VGG11 architecture: 64, 128, 256, 256, 512, 512, 512, 512
    let conv1_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 3, 3, 3);     // 64 filters

    let conv2_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 64, 3, 3);   // 128 filters

    let conv3_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(256, 128, 3, 3);  // 256 filters
    let conv4_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(256, 256, 3, 3);  // 256 filters

    let conv5_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(512, 256, 3, 3);  // 512 filters
    let conv6_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(512, 512, 3, 3);  // 512 filters

    let conv7_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(512, 512, 3, 3);  // 512 filters
    let conv8_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(512, 512, 3, 3);  // 512 filters

    // Fully connected layers (adjusted for smaller feature maps)
    let fc1_w: Vec<Vec<u8>> = rand_2d_vec_generator(512, 512);  // 512x512
    let fc2_w: Vec<Vec<u8>> = rand_2d_vec_generator(512, 512);  // 512x512
    let fc3_w: Vec<Vec<u8>> = rand_2d_vec_generator(10, 512);   // 10x512 (for 10 classes)

    // Multipliers for quantization
    let multiplier_conv1: Vec<f32> = vec![1.5; 64];
    let multiplier_conv2: Vec<f32> = vec![1.5; 128];
    let multiplier_conv3: Vec<f32> = vec![1.5; 256];
    let multiplier_conv4: Vec<f32> = vec![1.5; 256];
    let multiplier_conv5: Vec<f32> = vec![1.5; 512];
    let multiplier_conv6: Vec<f32> = vec![1.5; 512];
    let multiplier_conv7: Vec<f32> = vec![1.5; 512];
    let multiplier_conv8: Vec<f32> = vec![1.5; 512];

    let multiplier_fc1: Vec<f32> = vec![1.5; 512];
    let multiplier_fc2: Vec<f32> = vec![1.5; 512];
    let multiplier_fc3: Vec<f32> = vec![1.5; 10];

    // Zero points for quantization
    let length = 128;
    let x_0: Vec<u8> = vec![length];
    let conv1_output_0: Vec<u8> = vec![length];
    let conv2_output_0: Vec<u8> = vec![length];
    let conv3_output_0: Vec<u8> = vec![length];
    let conv4_output_0: Vec<u8> = vec![length];
    let conv5_output_0: Vec<u8> = vec![length];
    let conv6_output_0: Vec<u8> = vec![length];
    let conv7_output_0: Vec<u8> = vec![length];
    let conv8_output_0: Vec<u8> = vec![length];

    let fc1_output_0: Vec<u8> = vec![length];
    let fc2_output_0: Vec<u8> = vec![length];
    let fc3_output_0: Vec<u8> = vec![length];

    let conv1_weights_0: Vec<u8> = vec![length];
    let conv2_weights_0: Vec<u8> = vec![length];
    let conv3_weights_0: Vec<u8> = vec![length];
    let conv4_weights_0: Vec<u8> = vec![length];
    let conv5_weights_0: Vec<u8> = vec![length];
    let conv6_weights_0: Vec<u8> = vec![length];
    let conv7_weights_0: Vec<u8> = vec![length];
    let conv8_weights_0: Vec<u8> = vec![length];
    let fc1_weights_0: Vec<u8> = vec![length];
    let fc2_weights_0: Vec<u8> = vec![length];
    let fc3_weights_0: Vec<u8> = vec![length];

    println!("finish reading parameters");

    // Note: You'll need to implement vgg11_circuit_forward_u8 function
    // This is a placeholder call - the actual function needs to be implemented
    // based on your VGG11 architecture
    let z: Vec<Vec<u8>> = vgg_circuit_forward_u8(
        x.clone(),
        conv1_w.clone(),
        conv2_w.clone(),
        conv3_w.clone(),
        conv4_w.clone(),
        conv5_w.clone(),
        conv6_w.clone(),
        conv7_w.clone(),
        conv8_w.clone(),
        fc1_w.clone(),
        fc2_w.clone(),
        fc3_w.clone(),

        x_0[0],
        conv1_output_0[0],
        conv2_output_0[0],
        conv3_output_0[0],
        conv4_output_0[0],
        conv5_output_0[0],
        conv6_output_0[0],
        conv7_output_0[0],
        conv8_output_0[0],
        fc1_output_0[0],
        fc2_output_0[0], 
        fc3_output_0[0], 

        conv1_weights_0[0],
        conv2_weights_0[0],
        conv3_weights_0[0],
        conv4_weights_0[0],
        conv5_weights_0[0],
        conv6_weights_0[0],
        conv7_weights_0[0],
        conv8_weights_0[0],
        fc1_weights_0[0],
        fc2_weights_0[0],
        fc3_weights_0[0],

        multiplier_conv1.clone(),
        multiplier_conv2.clone(),
        multiplier_conv3.clone(),
        multiplier_conv4.clone(),
        multiplier_conv5.clone(),
        multiplier_conv6.clone(),
        multiplier_conv7.clone(),
        multiplier_conv8.clone(),
        multiplier_fc1.clone(),
        multiplier_fc2.clone(),
        multiplier_fc3.clone(),
    );

    println!("finish forwarding");

    //batch size is only one for faster calculation of total constraints
    let flattened_x3d: Vec<Vec<Vec<u8>>> = x.clone().into_iter().flatten().collect();
    let flattened_x2d: Vec<Vec<u8>> = flattened_x3d.into_iter().flatten().collect();
    let flattened_x1d: Vec<u8> = flattened_x2d.into_iter().flatten().collect();

    let flattened_z1d: Vec<u8> = z.clone().into_iter().flatten().collect();

    //println!("x outside {:?}", x.clone());
    println!("z outside {:?}", flattened_z1d.clone());
    let begin = Instant::now();
    let param = setup(&[0; 32]);
    let x_open = Randomness(Fr::rand(&mut rng));
    let x_com = pedersen_commit(&flattened_x1d, &param, &x_open);

    let z_open = Randomness(Fr::rand(&mut rng));
    let z_com = pedersen_commit(&flattened_z1d, &param, &z_open);
    let end = Instant::now();
    println!("commit time {:?}", end.duration_since(begin));

    // VGG11 Circuit structure - you'll need to implement VGG11CircuitU8OptimizedLv2PedersenPublicNNWeights
    let full_circuit = VGGCircuitU8OptimizedLv2PedersenPublicNNWeights {
        params: param.clone(),
        x: x.clone(),
        x_com: x_com.clone(),
        x_open: x_open,

        conv1_weights: conv1_w.clone(),
        conv2_weights: conv2_w.clone(),
        conv3_weights: conv3_w.clone(),
        conv4_weights: conv4_w.clone(),
        conv5_weights: conv5_w.clone(),
        conv6_weights: conv6_w.clone(),
        conv7_weights: conv7_w.clone(),
        conv8_weights: conv8_w.clone(),

        fc1_weights: fc1_w.clone(),
        fc2_weights: fc2_w.clone(),
        fc3_weights: fc3_w.clone(),

        //zero points for quantization.
        x_0: x_0[0],
        conv1_output_0: conv1_output_0[0],
        conv2_output_0: conv2_output_0[0],
        conv3_output_0: conv3_output_0[0],
        conv4_output_0: conv4_output_0[0],
        conv5_output_0: conv5_output_0[0],
        conv6_output_0: conv6_output_0[0],
        conv7_output_0: conv7_output_0[0],
        conv8_output_0: conv8_output_0[0],

        fc1_output_0: fc1_output_0[0],
        fc2_output_0: fc2_output_0[0],
        fc3_output_0: fc3_output_0[0],

        conv1_weights_0: conv1_weights_0[0],
        conv2_weights_0: conv2_weights_0[0],
        conv3_weights_0: conv3_weights_0[0],
        conv4_weights_0: conv4_weights_0[0],
        conv5_weights_0: conv5_weights_0[0],
        conv6_weights_0: conv6_weights_0[0],
        conv7_weights_0: conv7_weights_0[0],
        conv8_weights_0: conv8_weights_0[0],
        fc1_weights_0: fc1_weights_0[0],
        fc2_weights_0: fc2_weights_0[0],
        fc3_weights_0: fc3_weights_0[0],

        //multiplier for quantization
        multiplier_conv1: multiplier_conv1.clone(),
        multiplier_conv2: multiplier_conv2.clone(),
        multiplier_conv3: multiplier_conv3.clone(),
        multiplier_conv4: multiplier_conv4.clone(),
        multiplier_conv5: multiplier_conv5.clone(),
        multiplier_conv6: multiplier_conv6.clone(),
        multiplier_conv7: multiplier_conv7.clone(),
        multiplier_conv8: multiplier_conv8.clone(),

        multiplier_fc1: multiplier_fc1.clone(),
        multiplier_fc2: multiplier_fc2.clone(),
        multiplier_fc3: multiplier_fc3.clone(),

        z: z.clone(),
        z_open: z_open,
        z_com: z_com,
        knit_encoding: true,
    };

    println!("start generating random parameters");
    let begin = Instant::now();

    // pre-computed parameters
    let param =
        generate_random_parameters::<algebra::Bls12_381, _, _>(full_circuit.clone(), &mut rng)
            .unwrap();
    let end = Instant::now();

    println!("setup time {:?}", end.duration_since(begin));

    // let mut buf = vec![];
    // param.serialize(&mut buf).unwrap();
    // println!("crs size: {}", buf.len());

    let pvk = prepare_verifying_key(&param.vk);
    println!("random parameters generated!\n");

    // prover
    let begin = Instant::now();
    let proof = create_random_proof(full_circuit, &param, &mut rng).unwrap();
    let end = Instant::now();
    println!("prove time {:?}", end.duration_since(begin));

    let commitment = [x_com.x, x_com.y, z_com.x, z_com.y].to_vec();

    let inputs: Vec<Fq> = [
        commitment[..].as_ref(),
    ]
    .concat();

    let begin = Instant::now();
    verify_proof(&pvk, &proof, &inputs[..]).unwrap();
    // assert!(verify_proof(&pvk, &proof, &inputs[..]).unwrap());
    let end = Instant::now();
    println!("verification time {:?}", end.duration_since(begin));
}