// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use plonky2_field::{
    fft::fft,
    goldilocks_field::GoldilocksField,
    polynomial::PolynomialCoeffs,
    types::{Field, PrimeField64},
};
use rand::random;
use std::time::Instant;
use zeknox::{
    device::memory::HostOrDeviceSlice, init_twiddle_factors_rs, ntt_batch, types::NTTConfig,
};

fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log_n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(14);
    let batches: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let runs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let warmup: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2);

    let domain_size = 1usize << log_n;
    let total_elements = domain_size * batches;

    println!("NTT micro-bench");
    println!("log_n: {log_n}, batches: {batches}, runs: {runs}, warmup: {warmup}");

    let mut cpu_buffer: Vec<u64> = Vec::with_capacity(total_elements);
    for _ in 0..batches {
        let input: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
        cpu_buffer.extend(input.iter());
    }

    // GPU setup
    let gpu_id: i32 = 0;
    init_twiddle_factors_rs(0, log_n);
    let mut device_data: HostOrDeviceSlice<'_, u64> =
        HostOrDeviceSlice::cuda_malloc(gpu_id, total_elements).unwrap();
    device_data
        .copy_from_host(cpu_buffer.as_slice())
        .expect("copy to gpu");

    let mut cfg = NTTConfig::default();
    cfg.are_inputs_on_device = true;
    cfg.are_outputs_on_device = true;
    cfg.batches = batches as u32;

    for _ in 0..warmup {
        ntt_batch(0, device_data.as_mut_ptr(), log_n, cfg.clone());
    }

    let mut gpu_total = 0.0f64;
    for _ in 0..runs {
        let start = Instant::now();
        ntt_batch(0, device_data.as_mut_ptr(), log_n, cfg.clone());
        gpu_total += start.elapsed().as_secs_f64() * 1000.0;
    }

    let mut cpu_total = 0.0f64;
    for _ in 0..runs {
        let start = Instant::now();
        let mut cpu_res: Vec<Vec<u64>> = Vec::with_capacity(batches);
        for b in 0..batches {
            let offset = b * domain_size;
            let coeffs = cpu_buffer[offset..offset + domain_size]
                .iter()
                .map(|i| GoldilocksField::from_canonical_u64(*i))
                .collect::<Vec<GoldilocksField>>();
            let coefficients = PolynomialCoeffs { coeffs };
            let points = fft(coefficients);
            let cpu_results: Vec<u64> = points.values.iter().map(|x| x.to_canonical_u64()).collect();
            cpu_res.push(cpu_results);
        }
        let _ = cpu_res;
        cpu_total += start.elapsed().as_secs_f64() * 1000.0;
    }

    println!("GPU NTT avg over {} runs: {:.3} ms", runs, gpu_total / runs as f64);
    println!("CPU NTT avg over {} runs: {:.3} ms", runs, cpu_total / runs as f64);
    println!(
        "Speedup: {:.2}x",
        (cpu_total / runs as f64) / (gpu_total / runs as f64)
    );
}
