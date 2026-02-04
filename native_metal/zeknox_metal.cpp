// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstdlib>

#include "utils/rusterror.h"
#include "ntt/ntt.h"
#include "merkle/merkle.h"

static RustError not_implemented(const char *fn_name) {
    return RustError(1, fn_name);
}

static void abort_not_implemented(const char *fn_name) {
    std::fprintf(stderr, "zeknox metal stub: %s is not implemented\n", fn_name);
    std::abort();
}

extern "C" RustError get_number_of_gpus(size_t *ngpus) {
    if (ngpus) {
        *ngpus = 0;
    }
    return RustError(0);
}

extern "C" RustError list_devices_info() {
    return RustError(0);
}

extern "C" void init_cuda() {
    // Metal backend stub: no-op
}

extern "C" void init_cuda_degree(const uint32_t max_degree) {
    (void)max_degree;
    // Metal backend stub: no-op
}

extern "C" RustError init_twiddle_factors(size_t device_id, size_t lg_n) {
    (void)device_id;
    (void)lg_n;
    return not_implemented("init_twiddle_factors (metal)");
}

extern "C" RustError init_coset(size_t device_id, size_t lg_domain_size, const uint64_t coset_gen) {
    (void)device_id;
    (void)lg_domain_size;
    (void)coset_gen;
    return not_implemented("init_coset (metal)");
}

extern "C" RustError compute_batched_ntt(
    size_t device_id,
    void *inout,
    uint32_t lg_domain_size,
    NTT_Direction ntt_direction,
    NTT_Config cfg) {
    (void)device_id;
    (void)inout;
    (void)lg_domain_size;
    (void)ntt_direction;
    (void)cfg;
    return not_implemented("compute_batched_ntt (metal)");
}

extern "C" RustError compute_batched_ntt_repeats(
    size_t device_id,
    void *inout,
    uint32_t lg_domain_size,
    NTT_Direction ntt_direction,
    NTT_Config cfg,
    uint32_t repeats) {
    (void)device_id;
    (void)inout;
    (void)lg_domain_size;
    (void)ntt_direction;
    (void)cfg;
    (void)repeats;
    return not_implemented("compute_batched_ntt_repeats (metal)");
}

extern "C" RustError compute_batched_lde(
    size_t device_id,
    void *output,
    void *input,
    uint32_t lg_domain_size,
    NTT_Direction ntt_direction,
    NTT_Config cfg) {
    (void)device_id;
    (void)output;
    (void)input;
    (void)lg_domain_size;
    (void)ntt_direction;
    (void)cfg;
    return not_implemented("compute_batched_lde (metal)");
}

extern "C" RustError compute_batched_lde_multi_gpu(
    void *output,
    void *input,
    uint32_t num_gpu,
    NTT_Direction ntt_direction,
    NTT_Config cfg,
    uint32_t lg_domain_size,
    size_t total_num_input_elements,
    size_t total_num_output_elements) {
    (void)output;
    (void)input;
    (void)num_gpu;
    (void)ntt_direction;
    (void)cfg;
    (void)lg_domain_size;
    (void)total_num_input_elements;
    (void)total_num_output_elements;
    return not_implemented("compute_batched_lde_multi_gpu (metal)");
}

extern "C" RustError compute_transpose_rev(
    size_t device_id,
    void *output,
    void *input,
    uint32_t lg_n,
    NTT_TransposeConfig cfg) {
    (void)device_id;
    (void)output;
    (void)input;
    (void)lg_n;
    (void)cfg;
    return not_implemented("compute_transpose_rev (metal)");
}

extern "C" RustError compute_naive_transpose_rev(
    size_t device_id,
    void *output,
    void *input,
    uint32_t lg_n,
    NTT_TransposeConfig cfg) {
    (void)device_id;
    (void)output;
    (void)input;
    (void)lg_n;
    (void)cfg;
    return not_implemented("compute_naive_transpose_rev (metal)");
}

extern "C" void fill_digests_buf_linear_gpu_with_gpu_ptr(
    void *digests_buf_gpu_ptr,
    void *cap_buf_gpu_ptr,
    void *leaves_buf_gpu_ptr,
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height,
    u64 hash_type,
    u64 gpu_id) {
    (void)digests_buf_gpu_ptr;
    (void)cap_buf_gpu_ptr;
    (void)leaves_buf_gpu_ptr;
    (void)digests_buf_size;
    (void)cap_buf_size;
    (void)leaves_buf_size;
    (void)leaf_size;
    (void)cap_height;
    (void)hash_type;
    (void)gpu_id;
    abort_not_implemented("fill_digests_buf_linear_gpu_with_gpu_ptr (metal)");
}

extern "C" void fill_digests_buf_linear_multigpu_with_gpu_ptr(
    void *digests_buf_gpu_ptr,
    void *cap_buf_gpu_ptr,
    void *leaves_buf_gpu_ptr,
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height,
    u64 hash_type) {
    (void)digests_buf_gpu_ptr;
    (void)cap_buf_gpu_ptr;
    (void)leaves_buf_gpu_ptr;
    (void)digests_buf_size;
    (void)cap_buf_size;
    (void)leaves_buf_size;
    (void)leaf_size;
    (void)cap_height;
    (void)hash_type;
    abort_not_implemented("fill_digests_buf_linear_multigpu_with_gpu_ptr (metal)");
}

extern "C" void fill_digests_buf_linear_cpu(
    void *digests_buf_ptr,
    void *cap_buf_ptr,
    const void *leaves_buf_ptr,
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height,
    u64 hash_type) {
    (void)digests_buf_ptr;
    (void)cap_buf_ptr;
    (void)leaves_buf_ptr;
    (void)digests_buf_size;
    (void)cap_buf_size;
    (void)leaves_buf_size;
    (void)leaf_size;
    (void)cap_height;
    (void)hash_type;
    abort_not_implemented("fill_digests_buf_linear_cpu (metal)");
}
