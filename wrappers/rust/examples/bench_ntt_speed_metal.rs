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
use zeknox::{init_twiddle_factors_rs, ntt_batch, types::NTTConfig};

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

    println!("NTT micro-bench (Metal)");
    println!("log_n: {log_n}, batches: {batches}, runs: {runs}, warmup: {warmup}");

    let mut cpu_buffer: Vec<u64> = Vec::with_capacity(total_elements);
    for _ in 0..batches {
        let input: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
        cpu_buffer.extend(input.iter());
    }

    let sweep_env = std::env::var("METAL_NTT_LOG_SHARED_SWEEP").ok();

    let mut gpu_buffer = cpu_buffer.clone();

    init_twiddle_factors_rs(0, log_n);

    let mut cfg = NTTConfig::default();
    cfg.batches = batches as u32;
    cfg.are_inputs_on_device = false;
    cfg.are_outputs_on_device = false;

    let mut run_gpu = || -> f64 {
        for _ in 0..warmup {
            ntt_batch(0, gpu_buffer.as_mut_ptr(), log_n, cfg.clone());
        }

        let mut gpu_total = 0.0f64;
        for _ in 0..runs {
            let start = Instant::now();
            ntt_batch(0, gpu_buffer.as_mut_ptr(), log_n, cfg.clone());
            gpu_total += start.elapsed().as_secs_f64() * 1000.0;
        }
        gpu_total / runs as f64
    };

    let gpu_avg = if let Some(list) = sweep_env {
        let mut best = None;
        for entry in list.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
            std::env::set_var("METAL_NTT_LOG_SHARED", entry);
            let avg = run_gpu();
            println!("METAL_NTT_LOG_SHARED={entry} -> {avg:.3} ms");
            match best {
                None => best = Some((entry.to_string(), avg)),
                Some((_, best_avg)) if avg < best_avg => best = Some((entry.to_string(), avg)),
                _ => {}
            }
        }
        if let Some((best_log, best_avg)) = best {
            println!("Best METAL_NTT_LOG_SHARED={best_log} avg={best_avg:.3} ms");
            best_avg
        } else {
            run_gpu()
        }
    } else {
        run_gpu()
    };

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

    println!("GPU NTT avg over {} runs: {:.3} ms", runs, gpu_avg);
    println!("CPU NTT avg over {} runs: {:.3} ms", runs, cpu_total / runs as f64);
    println!(
        "Speedup: {:.2}x",
        (cpu_total / runs as f64) / gpu_avg
    );
}
