// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <mutex>
#include <vector>

#include "utils/rusterror.h"
#include "ntt/ntt.h"
#include "merkle/merkle.h"

namespace {

constexpr uint64_t kGoldilocksPrime = 18446744069414584321ULL;
constexpr uint64_t kGoldilocksPowerOfTwoGenerator = 1753635133440165772ULL;
constexpr uint64_t kGoldilocksCosetShift = 7ULL;

struct LinearUniforms {
    uint32_t level;
    uint32_t subtree_digests_len;
    uint32_t subtree_leaves_len;
    uint32_t leaf_size;
    uint32_t leaf_count;
    uint32_t subtree_count;
    uint32_t grid_width;
};

struct NTTUniforms {
    uint32_t n;
    uint32_t log_n;
    uint32_t stage;
    uint32_t direction;
    uint32_t twiddle_stride;
};

struct NTTBatchUniforms {
    uint32_t n;
    uint32_t log_n;
    uint32_t stage;
    uint32_t direction;
    uint32_t twiddle_stride;
    uint32_t batches;
    uint32_t batch_stride;
};

struct CosetUniforms {
    uint32_t n;
    uint32_t batches;
};

struct ExtendUniforms {
    uint32_t in_n;
    uint32_t out_n;
    uint32_t batches;
};

struct ExtendCosetUniforms {
    uint32_t in_n;
    uint32_t out_n;
    uint32_t batches;
};

struct TransposeUniforms {
    uint32_t n;
    uint32_t log_n;
    uint32_t batches;
};

struct TransposeLeafUniforms {
    uint32_t n;
    uint32_t log_n;
    uint32_t batches;
    uint32_t subtree_digests_len;
    uint32_t subtree_leaves_len;
    uint32_t leaf_size;
    uint32_t leaf_count;
    uint32_t subtree_count;
    uint32_t grid_width;
};

struct MetalRuntime {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> f_hash_leaves;
    id<MTLComputePipelineState> f_hash_leaves_transpose_rev;
    id<MTLComputePipelineState> f_hash_tree_level;
    id<MTLComputePipelineState> f_hash_tree_level_2;
    id<MTLComputePipelineState> f_hash_caps;
    id<MTLComputePipelineState> f_ntt_bit_reverse;
    id<MTLComputePipelineState> f_ntt_butterfly;
    id<MTLComputePipelineState> f_ntt_butterfly_shared;
    id<MTLComputePipelineState> f_intt_butterfly;
    id<MTLComputePipelineState> f_ntt_scale;
    id<MTLComputePipelineState> f_ntt_bit_reverse_batch;
    id<MTLComputePipelineState> f_ntt_bit_reverse_coset_batch;
    id<MTLComputePipelineState> f_ntt_butterfly_batch;
    id<MTLComputePipelineState> f_ntt_butterfly_simdgroup_batch;
    id<MTLComputePipelineState> f_ntt_butterfly_shared_batch;
    id<MTLComputePipelineState> f_ntt_template_shared_batch;
    id<MTLComputePipelineState> f_intt_butterfly_batch;
    id<MTLComputePipelineState> f_intt_butterfly_simdgroup_batch;
    id<MTLComputePipelineState> f_ntt_scale_batch;
    id<MTLComputePipelineState> f_ntt_scale_coset_batch;
    id<MTLComputePipelineState> f_extend_inputs_batch;
    id<MTLComputePipelineState> f_extend_inputs_coset_batch;
    id<MTLComputePipelineState> f_batch_vector_mult;
    id<MTLComputePipelineState> f_transpose_rev;
    id<MTLBuffer> twiddles;
    id<MTLBuffer> inv_twiddles;
    id<MTLBuffer> n_inverses;
    id<MTLBuffer> coset_pows;
    id<MTLBuffer> coset_inv_pows;
    id<MTLBuffer> cached_in_buf;
    id<MTLBuffer> cached_out_buf;
    size_t cached_in_bytes;
    size_t cached_out_bytes;
    uint32_t max_log_n;
};

static std::once_flag g_runtime_once;
static MetalRuntime g_runtime;

static inline uint64_t mul_mod(uint64_t a, uint64_t b);
static inline uint64_t pow_mod(uint64_t base, uint64_t exp);
static inline uint64_t mod_inverse(uint64_t a);

static void get_threadgroup_count(uint64_t count, MTLSize *out_groups) {
    const uint64_t max_dim = 32768;
    if (count <= max_dim) {
        out_groups->width = count;
        out_groups->height = 1;
        out_groups->depth = 1;
        return;
    }
    uint64_t height = (count + max_dim - 1) / max_dim;
    out_groups->width = max_dim;
    out_groups->height = height;
    out_groups->depth = 1;
}

static NSUInteger merkle_threads_per_group(id<MTLComputePipelineState> pipeline) {
    NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger tpg = std::min<NSUInteger>(128, max_threads);
    const char *env = std::getenv("METAL_MERKLE_TPG");
    if (env && env[0] != '\0') {
        int v = std::atoi(env);
        if (v > 0) {
            tpg = std::min<NSUInteger>(static_cast<NSUInteger>(v), max_threads);
        }
    }
    return tpg;
}

static void ensure_runtime() {
    std::call_once(g_runtime_once, [] {
        @autoreleasepool {
            const uint32_t max_log_n = 23;
            g_runtime.device = MTLCreateSystemDefaultDevice();
            if (!g_runtime.device) {
                std::fprintf(stderr, "zeknox metal: no Metal device found\n");
                std::abort();
            }
            g_runtime.queue = [g_runtime.device newCommandQueue];

            const char *lib_path = std::getenv("ZEKNOX_METAL_LIB");
#ifdef ZEKNOX_METAL_LIB_DEFAULT
            if (!lib_path || lib_path[0] == '\0') {
                lib_path = ZEKNOX_METAL_LIB_DEFAULT;
            }
#endif
            if (!lib_path || lib_path[0] == '\0') {
                std::fprintf(stderr, "zeknox metal: ZEKNOX_METAL_LIB not set and no default available\n");
                std::abort();
            }

            NSError *err = nil;
            NSString *path = [NSString stringWithUTF8String:lib_path];
            NSURL *url = [NSURL fileURLWithPath:path];
            g_runtime.library = [g_runtime.device newLibraryWithURL:url error:&err];
            if (!g_runtime.library) {
                std::fprintf(stderr, "zeknox metal: failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }

            id<MTLFunction> f_leaves = [g_runtime.library newFunctionWithName:@"poseidon_hash_leaves_linear"];
            id<MTLFunction> f_leaves_transpose = [g_runtime.library newFunctionWithName:@"poseidon_hash_leaves_linear_transpose_rev"];
            id<MTLFunction> f_level = [g_runtime.library newFunctionWithName:@"poseidon_hash_tree_level_linear"];
            id<MTLFunction> f_level2 = [g_runtime.library newFunctionWithName:@"poseidon_hash_tree_level_linear_2"];
            id<MTLFunction> f_caps = [g_runtime.library newFunctionWithName:@"poseidon_hash_caps_linear"];
            id<MTLFunction> f_ntt_bit_reverse = [g_runtime.library newFunctionWithName:@"ntt_bit_reverse"];
            id<MTLFunction> f_ntt_butterfly = [g_runtime.library newFunctionWithName:@"ntt_butterfly"];
            id<MTLFunction> f_ntt_butterfly_shared = [g_runtime.library newFunctionWithName:@"ntt_butterfly_shared"];
            id<MTLFunction> f_intt_butterfly = [g_runtime.library newFunctionWithName:@"intt_butterfly"];
            id<MTLFunction> f_ntt_scale = [g_runtime.library newFunctionWithName:@"ntt_scale"];
            id<MTLFunction> f_ntt_bit_reverse_batch = [g_runtime.library newFunctionWithName:@"ntt_bit_reverse_batch"];
            id<MTLFunction> f_ntt_bit_reverse_coset_batch = [g_runtime.library newFunctionWithName:@"ntt_bit_reverse_coset_batch"];
            id<MTLFunction> f_ntt_butterfly_batch = [g_runtime.library newFunctionWithName:@"ntt_butterfly_batch"];
            id<MTLFunction> f_ntt_butterfly_simdgroup_batch = [g_runtime.library newFunctionWithName:@"ntt_butterfly_simdgroup_batch"];
            id<MTLFunction> f_ntt_butterfly_shared_batch = [g_runtime.library newFunctionWithName:@"ntt_butterfly_shared_batch"];
            id<MTLFunction> f_ntt_template_shared_batch = [g_runtime.library newFunctionWithName:@"ntt_template_kernel_shared_batch"];
            id<MTLFunction> f_intt_butterfly_batch = [g_runtime.library newFunctionWithName:@"intt_butterfly_batch"];
            id<MTLFunction> f_intt_butterfly_simdgroup_batch = [g_runtime.library newFunctionWithName:@"intt_butterfly_simdgroup_batch"];
            id<MTLFunction> f_ntt_scale_batch = [g_runtime.library newFunctionWithName:@"ntt_scale_batch"];
            id<MTLFunction> f_ntt_scale_coset_batch = [g_runtime.library newFunctionWithName:@"ntt_scale_coset_batch"];
            id<MTLFunction> f_extend_inputs_batch = [g_runtime.library newFunctionWithName:@"extend_inputs_batch"];
            id<MTLFunction> f_extend_inputs_coset_batch = [g_runtime.library newFunctionWithName:@"extend_inputs_coset_batch"];
            id<MTLFunction> f_batch_vector_mult = [g_runtime.library newFunctionWithName:@"batch_vector_mult"];
            id<MTLFunction> f_transpose_rev = [g_runtime.library newFunctionWithName:@"transpose_rev"];
            if (!f_leaves || !f_leaves_transpose || !f_level || !f_level2 || !f_caps) {
                std::fprintf(stderr, "zeknox metal: missing required shader functions\n");
                std::abort();
            }

            g_runtime.f_hash_leaves = [g_runtime.device newComputePipelineStateWithFunction:f_leaves error:&err];
            if (!g_runtime.f_hash_leaves) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (leaves): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_hash_leaves_transpose_rev = [g_runtime.device newComputePipelineStateWithFunction:f_leaves_transpose error:&err];
            if (!g_runtime.f_hash_leaves_transpose_rev) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (leaves_transpose): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_hash_tree_level = [g_runtime.device newComputePipelineStateWithFunction:f_level error:&err];
            if (!g_runtime.f_hash_tree_level) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (level): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_hash_tree_level_2 = [g_runtime.device newComputePipelineStateWithFunction:f_level2 error:&err];
            if (!g_runtime.f_hash_tree_level_2) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (level2): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_hash_caps = [g_runtime.device newComputePipelineStateWithFunction:f_caps error:&err];
            if (!g_runtime.f_hash_caps) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (caps): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }

            if (!f_ntt_bit_reverse || !f_ntt_butterfly || !f_ntt_butterfly_shared || !f_intt_butterfly || !f_ntt_scale ||
                !f_ntt_bit_reverse_batch || !f_ntt_bit_reverse_coset_batch || !f_ntt_butterfly_batch || !f_ntt_butterfly_simdgroup_batch ||
                !f_ntt_butterfly_shared_batch || !f_ntt_template_shared_batch || !f_intt_butterfly_batch || !f_intt_butterfly_simdgroup_batch || !f_ntt_scale_batch ||
                !f_ntt_scale_coset_batch || !f_extend_inputs_batch || !f_extend_inputs_coset_batch ||
                !f_batch_vector_mult || !f_transpose_rev) {
                std::fprintf(stderr, "zeknox metal: missing required NTT shader functions\n");
                std::abort();
            }

            g_runtime.f_ntt_bit_reverse = [g_runtime.device newComputePipelineStateWithFunction:f_ntt_bit_reverse error:&err];
            if (!g_runtime.f_ntt_bit_reverse) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (ntt_bit_reverse): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_ntt_butterfly = [g_runtime.device newComputePipelineStateWithFunction:f_ntt_butterfly error:&err];
            if (!g_runtime.f_ntt_butterfly) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (ntt_butterfly): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_ntt_butterfly_shared = [g_runtime.device newComputePipelineStateWithFunction:f_ntt_butterfly_shared error:&err];
            if (!g_runtime.f_ntt_butterfly_shared) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (ntt_butterfly_shared): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_intt_butterfly = [g_runtime.device newComputePipelineStateWithFunction:f_intt_butterfly error:&err];
            if (!g_runtime.f_intt_butterfly) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (intt_butterfly): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_ntt_scale = [g_runtime.device newComputePipelineStateWithFunction:f_ntt_scale error:&err];
            if (!g_runtime.f_ntt_scale) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (ntt_scale): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_ntt_bit_reverse_batch = [g_runtime.device newComputePipelineStateWithFunction:f_ntt_bit_reverse_batch error:&err];
            if (!g_runtime.f_ntt_bit_reverse_batch) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (ntt_bit_reverse_batch): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_ntt_bit_reverse_coset_batch = [g_runtime.device newComputePipelineStateWithFunction:f_ntt_bit_reverse_coset_batch error:&err];
            if (!g_runtime.f_ntt_bit_reverse_coset_batch) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (ntt_bit_reverse_coset_batch): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_ntt_butterfly_batch = [g_runtime.device newComputePipelineStateWithFunction:f_ntt_butterfly_batch error:&err];
            if (!g_runtime.f_ntt_butterfly_batch) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (ntt_butterfly_batch): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_ntt_butterfly_simdgroup_batch = [g_runtime.device newComputePipelineStateWithFunction:f_ntt_butterfly_simdgroup_batch error:&err];
            if (!g_runtime.f_ntt_butterfly_simdgroup_batch) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (ntt_butterfly_simdgroup_batch): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_ntt_butterfly_shared_batch = [g_runtime.device newComputePipelineStateWithFunction:f_ntt_butterfly_shared_batch error:&err];
            if (!g_runtime.f_ntt_butterfly_shared_batch) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (ntt_butterfly_shared_batch): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_ntt_template_shared_batch = [g_runtime.device newComputePipelineStateWithFunction:f_ntt_template_shared_batch error:&err];
            if (!g_runtime.f_ntt_template_shared_batch) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (ntt_template_shared_batch): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_intt_butterfly_batch = [g_runtime.device newComputePipelineStateWithFunction:f_intt_butterfly_batch error:&err];
            if (!g_runtime.f_intt_butterfly_batch) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (intt_butterfly_batch): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_intt_butterfly_simdgroup_batch = [g_runtime.device newComputePipelineStateWithFunction:f_intt_butterfly_simdgroup_batch error:&err];
            if (!g_runtime.f_intt_butterfly_simdgroup_batch) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (intt_butterfly_simdgroup_batch): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_ntt_scale_batch = [g_runtime.device newComputePipelineStateWithFunction:f_ntt_scale_batch error:&err];
            if (!g_runtime.f_ntt_scale_batch) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (ntt_scale_batch): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_ntt_scale_coset_batch = [g_runtime.device newComputePipelineStateWithFunction:f_ntt_scale_coset_batch error:&err];
            if (!g_runtime.f_ntt_scale_coset_batch) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (ntt_scale_coset_batch): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_extend_inputs_batch = [g_runtime.device newComputePipelineStateWithFunction:f_extend_inputs_batch error:&err];
            if (!g_runtime.f_extend_inputs_batch) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (extend_inputs_batch): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_extend_inputs_coset_batch = [g_runtime.device newComputePipelineStateWithFunction:f_extend_inputs_coset_batch error:&err];
            if (!g_runtime.f_extend_inputs_coset_batch) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (extend_inputs_coset_batch): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_batch_vector_mult = [g_runtime.device newComputePipelineStateWithFunction:f_batch_vector_mult error:&err];
            if (!g_runtime.f_batch_vector_mult) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (batch_vector_mult): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }
            g_runtime.f_transpose_rev = [g_runtime.device newComputePipelineStateWithFunction:f_transpose_rev error:&err];
            if (!g_runtime.f_transpose_rev) {
                std::fprintf(stderr, "zeknox metal: failed to create pipeline (transpose_rev): %s\n", [[err localizedDescription] UTF8String]);
                std::abort();
            }

            g_runtime.max_log_n = max_log_n;

            const uint64_t max_n = 1ULL << max_log_n;

            std::vector<uint64_t> twiddles;
            std::vector<uint64_t> inv_twiddles;
            twiddles.reserve(max_n);
            inv_twiddles.reserve(max_n);

            uint32_t log_n = 0;
            uint64_t tmp = max_n;
            while (tmp > 1) {
                tmp >>= 1;
                log_n += 1;
            }
            uint64_t root = kGoldilocksPowerOfTwoGenerator;
            for (uint32_t i = 0; i < (32 - log_n); i++) {
                root = mul_mod(root, root);
            }

            const uint64_t omega = root;
            const uint64_t omega_inv = mod_inverse(omega);
            uint64_t omega_pow = 1;
            uint64_t omega_inv_pow = 1;
            for (uint64_t i = 0; i < max_n; i++) {
                twiddles.push_back(omega_pow);
                inv_twiddles.push_back(omega_inv_pow);
                omega_pow = mul_mod(omega_pow, omega);
                omega_inv_pow = mul_mod(omega_inv_pow, omega_inv);
            }

            std::vector<uint64_t> n_inverses;
            n_inverses.reserve(max_log_n + 1);
            for (uint32_t log_n = 0; log_n <= max_log_n; log_n++) {
                uint64_t n_val = 1ULL << log_n;
                n_inverses.push_back(mod_inverse(n_val));
            }

            std::vector<uint64_t> coset_pows;
            std::vector<uint64_t> coset_inv_pows;
            coset_pows.reserve(max_n);
            coset_inv_pows.reserve(max_n);
            uint64_t coset = kGoldilocksCosetShift;
            uint64_t coset_inv = mod_inverse(coset);
            uint64_t coset_pow = 1;
            uint64_t coset_inv_pow = 1;
            for (uint64_t i = 0; i < max_n; i++) {
                coset_pows.push_back(coset_pow);
                coset_inv_pows.push_back(coset_inv_pow);
                coset_pow = mul_mod(coset_pow, coset);
                coset_inv_pow = mul_mod(coset_inv_pow, coset_inv);
            }

            g_runtime.twiddles = [g_runtime.device newBufferWithBytes:twiddles.data()
                                                               length:twiddles.size() * sizeof(uint64_t)
                                                              options:MTLResourceStorageModeShared];
            g_runtime.inv_twiddles = [g_runtime.device newBufferWithBytes:inv_twiddles.data()
                                                                   length:inv_twiddles.size() * sizeof(uint64_t)
                                                                  options:MTLResourceStorageModeShared];
            g_runtime.n_inverses = [g_runtime.device newBufferWithBytes:n_inverses.data()
                                                                  length:n_inverses.size() * sizeof(uint64_t)
                                                                 options:MTLResourceStorageModeShared];
            g_runtime.coset_pows = [g_runtime.device newBufferWithBytes:coset_pows.data()
                                                                 length:coset_pows.size() * sizeof(uint64_t)
                                                                options:MTLResourceStorageModeShared];
            g_runtime.coset_inv_pows = [g_runtime.device newBufferWithBytes:coset_inv_pows.data()
                                                                     length:coset_inv_pows.size() * sizeof(uint64_t)
                                                                    options:MTLResourceStorageModeShared];
            if (!g_runtime.twiddles || !g_runtime.inv_twiddles || !g_runtime.n_inverses ||
                !g_runtime.coset_pows || !g_runtime.coset_inv_pows) {
                std::fprintf(stderr, "zeknox metal: failed to allocate NTT twiddle buffers\n");
                std::abort();
            }
        }
    });
}

static inline uint64_t mul_mod(uint64_t a, uint64_t b) {
    __uint128_t product = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    return static_cast<uint64_t>(product % kGoldilocksPrime);
}

static inline uint64_t pow_mod(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    uint64_t b = base % kGoldilocksPrime;
    uint64_t e = exp;
    while (e > 0) {
        if (e & 1) {
            result = mul_mod(result, b);
        }
        e >>= 1;
        b = mul_mod(b, b);
    }
    return result;
}

static inline uint64_t mod_inverse(uint64_t a) {
    return pow_mod(a, kGoldilocksPrime - 2);
}

static id<MTLBuffer> get_or_resize_buffer(id<MTLBuffer> __strong *buf, size_t *cached_bytes, size_t need_bytes) {
    if (!buf || !cached_bytes) {
        return nil;
    }
    if (!*buf || *cached_bytes < need_bytes) {
        *buf = [g_runtime.device newBufferWithLength:need_bytes options:MTLResourceStorageModeShared];
        *cached_bytes = need_bytes;
    }
    return *buf;
}

static double command_buffer_elapsed_ms(id<MTLCommandBuffer> command_buffer,
                                        const std::chrono::steady_clock::time_point &start,
                                        const std::chrono::steady_clock::time_point &end) {
    double gpu_ms = 0.0;
    if (command_buffer && command_buffer.GPUEndTime > command_buffer.GPUStartTime) {
        gpu_ms = (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
    } else {
        gpu_ms = std::chrono::duration<double, std::milli>(end - start).count();
    }
    return gpu_ms;
}

static void encode_extend_inputs_batch(id<MTLCommandBuffer> command_buffer,
                                       id<MTLBuffer> output_buf,
                                       id<MTLBuffer> input_buf,
                                       uint32_t in_n,
                                       uint32_t out_n,
                                       uint32_t batches) {
    ExtendUniforms uniforms{};
    uniforms.in_n = in_n;
    uniforms.out_n = out_n;
    uniforms.batches = batches;

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:g_runtime.f_extend_inputs_batch];
    [encoder setBuffer:output_buf offset:0 atIndex:0];
    [encoder setBuffer:input_buf offset:0 atIndex:1];
    [encoder setBytes:&uniforms length:sizeof(ExtendUniforms) atIndex:2];

    uint64_t total = static_cast<uint64_t>(out_n) * static_cast<uint64_t>(batches);
    const NSUInteger threads_per_group = g_runtime.f_extend_inputs_batch.threadExecutionWidth;
    const NSUInteger num_groups = (total + threads_per_group - 1) / threads_per_group;
    MTLSize tg = MTLSizeMake(threads_per_group, 1, 1);
    MTLSize ng = MTLSizeMake(num_groups, 1, 1);
    [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
    [encoder endEncoding];
}

static void encode_extend_inputs_coset_batch(id<MTLCommandBuffer> command_buffer,
                                             id<MTLBuffer> output_buf,
                                             id<MTLBuffer> input_buf,
                                             uint32_t in_n,
                                             uint32_t out_n,
                                             uint32_t batches,
                                             bool inverse) {
    ExtendCosetUniforms uniforms{};
    uniforms.in_n = in_n;
    uniforms.out_n = out_n;
    uniforms.batches = batches;

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:g_runtime.f_extend_inputs_coset_batch];
    [encoder setBuffer:output_buf offset:0 atIndex:0];
    [encoder setBuffer:input_buf offset:0 atIndex:1];
    [encoder setBuffer:(inverse ? g_runtime.coset_inv_pows : g_runtime.coset_pows) offset:0 atIndex:2];
    [encoder setBytes:&uniforms length:sizeof(ExtendCosetUniforms) atIndex:3];

    uint64_t total = static_cast<uint64_t>(out_n) * static_cast<uint64_t>(batches);
    const NSUInteger threads_per_group = g_runtime.f_extend_inputs_coset_batch.threadExecutionWidth;
    const NSUInteger num_groups = (total + threads_per_group - 1) / threads_per_group;
    MTLSize tg = MTLSizeMake(threads_per_group, 1, 1);
    MTLSize ng = MTLSizeMake(num_groups, 1, 1);
    [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
    [encoder endEncoding];
}

static void encode_batch_vector_mult(id<MTLCommandBuffer> command_buffer,
                                     id<MTLBuffer> data_buf,
                                     uint32_t n,
                                     uint32_t batches,
                                     bool inverse) {
    CosetUniforms uniforms{};
    uniforms.n = n;
    uniforms.batches = batches;

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:g_runtime.f_batch_vector_mult];
    [encoder setBuffer:data_buf offset:0 atIndex:0];
    [encoder setBuffer:(inverse ? g_runtime.coset_inv_pows : g_runtime.coset_pows) offset:0 atIndex:1];
    [encoder setBytes:&uniforms length:sizeof(CosetUniforms) atIndex:2];

    uint64_t total = static_cast<uint64_t>(n) * static_cast<uint64_t>(batches);
    const NSUInteger threads_per_group = g_runtime.f_batch_vector_mult.threadExecutionWidth;
    const NSUInteger num_groups = (total + threads_per_group - 1) / threads_per_group;
    MTLSize tg = MTLSizeMake(threads_per_group, 1, 1);
    MTLSize ng = MTLSizeMake(num_groups, 1, 1);
    [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
    [encoder endEncoding];
}

static double run_timed_extend_inputs_batch(id<MTLBuffer> output_buf,
                                            id<MTLBuffer> input_buf,
                                            uint32_t in_n,
                                            uint32_t out_n,
                                            uint32_t batches) {
    id<MTLCommandBuffer> command_buffer = [g_runtime.queue commandBuffer];
    auto start = std::chrono::steady_clock::now();
    encode_extend_inputs_batch(command_buffer, output_buf, input_buf, in_n, out_n, batches);
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    auto end = std::chrono::steady_clock::now();
    return command_buffer_elapsed_ms(command_buffer, start, end);
}

static double run_timed_batch_vector_mult(id<MTLBuffer> data_buf,
                                          uint32_t n,
                                          uint32_t batches,
                                          bool inverse) {
    id<MTLCommandBuffer> command_buffer = [g_runtime.queue commandBuffer];
    auto start = std::chrono::steady_clock::now();
    encode_batch_vector_mult(command_buffer, data_buf, n, batches, inverse);
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    auto end = std::chrono::steady_clock::now();
    return command_buffer_elapsed_ms(command_buffer, start, end);
}

static double run_timed_ntt_stage(id<MTLComputePipelineState> pipeline,
                                  id<MTLBuffer> data_buf,
                                  id<MTLBuffer> twiddles,
                                  const NTTBatchUniforms &uniforms,
                                  uint32_t stage_offset,
                                  uint32_t stage_count,
                                  uint32_t log_n,
                                  uint32_t batches) {
    id<MTLCommandBuffer> command_buffer = [g_runtime.queue commandBuffer];
    auto start = std::chrono::steady_clock::now();
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:data_buf offset:0 atIndex:0];
    if (pipeline == g_runtime.f_ntt_bit_reverse_batch) {
        [encoder setBytes:&uniforms length:sizeof(NTTBatchUniforms) atIndex:1];
    } else {
        if (twiddles) {
            [encoder setBuffer:twiddles offset:0 atIndex:1];
        }
        [encoder setBytes:&uniforms length:sizeof(NTTBatchUniforms) atIndex:2];
    }
    if (pipeline == g_runtime.f_ntt_butterfly_shared_batch) {
        [encoder setBytes:&stage_offset length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&stage_count length:sizeof(uint32_t) atIndex:4];
        const uint32_t block_size = 1U << stage_count;
        const uint32_t blocks_per_batch = (1U << log_n) / block_size;
        const uint32_t num_groups = blocks_per_batch * batches;
        MTLSize tg = MTLSizeMake(block_size, 1, 1);
        MTLSize ng = MTLSizeMake(num_groups, 1, 1);
        [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
    } else {
        const uint64_t num_butterflies = (static_cast<uint64_t>(1U << log_n) / 2) * static_cast<uint64_t>(batches);
        const NSUInteger threads_per_group = pipeline.threadExecutionWidth;
        const NSUInteger num_groups = (num_butterflies + threads_per_group - 1) / threads_per_group;
        MTLSize tg = MTLSizeMake(threads_per_group, 1, 1);
        MTLSize ng = MTLSizeMake(num_groups, 1, 1);
        [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
    }
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    auto end = std::chrono::steady_clock::now();
    return command_buffer_elapsed_ms(command_buffer, start, end);
}

static void encode_ntt_batched(id<MTLCommandBuffer> command_buffer,
                               id<MTLBuffer> data_buf,
                               uint32_t log_n,
                               uint32_t batches,
                               bool inverse,
                               bool coset_forward,
                               bool coset_inverse) {
    const uint32_t n = 1U << log_n;
    const uint32_t twiddle_stride = (1U << g_runtime.max_log_n) / n;
    const uint64_t total = static_cast<uint64_t>(n) * static_cast<uint64_t>(batches);
    id<MTLComputePipelineState> butterfly = inverse ? g_runtime.f_intt_butterfly_batch : g_runtime.f_ntt_butterfly_batch;
    id<MTLBuffer> twiddles = inverse ? g_runtime.inv_twiddles : g_runtime.twiddles;

    // Bit-reversal permutation
    {
        NTTBatchUniforms uniforms{};
        uniforms.n = n;
        uniforms.log_n = log_n;
        uniforms.stage = 0;
        uniforms.direction = inverse ? 1U : 0U;
        uniforms.twiddle_stride = twiddle_stride;
        uniforms.batches = batches;
        uniforms.batch_stride = n;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (coset_forward) {
            [encoder setComputePipelineState:g_runtime.f_ntt_bit_reverse_coset_batch];
            [encoder setBuffer:data_buf offset:0 atIndex:0];
            [encoder setBuffer:g_runtime.coset_pows offset:0 atIndex:1];
            [encoder setBytes:&uniforms length:sizeof(NTTBatchUniforms) atIndex:2];
        } else {
            [encoder setComputePipelineState:g_runtime.f_ntt_bit_reverse_batch];
            [encoder setBuffer:data_buf offset:0 atIndex:0];
            [encoder setBytes:&uniforms length:sizeof(NTTBatchUniforms) atIndex:1];
        }

        const NSUInteger threads_per_group = g_runtime.f_ntt_bit_reverse_batch.threadExecutionWidth;
        const NSUInteger num_groups = (total + threads_per_group - 1) / threads_per_group;
        MTLSize tg = MTLSizeMake(threads_per_group, 1, 1);
        MTLSize ng = MTLSizeMake(num_groups, 1, 1);
        [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
        [encoder endEncoding];
    }

    // Shared-memory stages (early stages)
    uint32_t max_threads = (uint32_t)g_runtime.f_ntt_butterfly_shared_batch.maxTotalThreadsPerThreadgroup;
    uint32_t log_shared = 0;
    while ((1U << log_shared) < max_threads && log_shared < log_n) {
        log_shared++;
    }
    if ((1U << log_shared) > max_threads) {
        log_shared--;
    }
    if (log_shared > 10) {
        log_shared = 10;
    }
    if (log_shared > 8) {
        log_shared = 8;
    }
    const char *env_log = std::getenv("METAL_NTT_LOG_SHARED");
    if (env_log && env_log[0] != '\0') {
        int override = std::atoi(env_log);
        if (override >= 0 && override <= (int)log_n) {
            log_shared = static_cast<uint32_t>(override);
        }
    }
    const char *env_dbg = std::getenv("METAL_NTT_DEBUG");
    if (env_dbg && env_dbg[0] != '\0') {
        std::fprintf(stderr, "zeknox metal: log_shared=%u log_n=%u\n", log_shared, log_n);
    }

    const char *env_simd = std::getenv("METAL_NTT_SIMDGROUP");
    const bool use_simd = env_simd && env_simd[0] != '\0';
    const char *env_cuda_like = std::getenv("METAL_NTT_CUDA_LIKE");
    const bool use_cuda_like = env_cuda_like && env_cuda_like[0] != '\0';
    if (log_shared > 0) {
        NTTBatchUniforms uniforms{};
        uniforms.n = n;
        uniforms.log_n = log_n;
        uniforms.stage = 0;
        uniforms.direction = inverse ? 1U : 0U;
        uniforms.twiddle_stride = twiddle_stride;
        uniforms.batches = batches;
        uniforms.batch_stride = n;

        uint32_t stage_offset = 0;
        uint32_t stage_count = log_shared;
        if (use_cuda_like && !inverse) {
            uint32_t threads = 256;
            const char *env_threads = std::getenv("METAL_NTT_CUDA_THREADS");
            if (env_threads && env_threads[0] != '\0') {
                int v = std::atoi(env_threads);
                if (v > 0) {
                    threads = static_cast<uint32_t>(v);
                }
            }
            uint32_t max_threads = (uint32_t)g_runtime.f_ntt_template_shared_batch.maxTotalThreadsPerThreadgroup;
            if (threads > max_threads) {
                threads = max_threads;
            }
            // ensure power-of-two
            uint32_t pow2 = 1;
            while (pow2 * 2 <= threads) {
                pow2 *= 2;
            }
            threads = pow2;
            uint32_t block_size = threads * 2;
            uint32_t stage_cuda = 0;
            while ((1U << stage_cuda) < block_size) {
                stage_cuda++;
            }
            if (stage_cuda > log_shared) {
                stage_cuda = log_shared;
            }
            stage_count = stage_cuda;
            block_size = 1U << stage_cuda;
            uint32_t blocks_per_batch = n / block_size;
            uint32_t total_tasks = blocks_per_batch * batches;

            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:g_runtime.f_ntt_template_shared_batch];
            [encoder setBuffer:data_buf offset:0 atIndex:0];
            [encoder setBuffer:twiddles offset:0 atIndex:1];
            [encoder setBytes:&uniforms length:sizeof(NTTBatchUniforms) atIndex:2];
            [encoder setBytes:&stage_offset length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&stage_count length:sizeof(uint32_t) atIndex:4];
            [encoder setBytes:&total_tasks length:sizeof(uint32_t) atIndex:5];
            [encoder setBytes:&block_size length:sizeof(uint32_t) atIndex:6];

            MTLSize tg = MTLSizeMake(threads, 1, 1);
            MTLSize ng = MTLSizeMake(total_tasks, 1, 1);
            [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
            [encoder endEncoding];

            stage_offset = stage_count;
            stage_count = log_shared - stage_offset;
        } else if (use_simd && log_shared >= 5) {
            stage_count = 5;
            const uint32_t block_size = 1U << stage_count;
            const uint32_t blocks_per_batch = n / block_size;
            const uint32_t num_groups = blocks_per_batch * batches;
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:inverse ? g_runtime.f_intt_butterfly_simdgroup_batch
                                                    : g_runtime.f_ntt_butterfly_simdgroup_batch];
            [encoder setBuffer:data_buf offset:0 atIndex:0];
            [encoder setBuffer:twiddles offset:0 atIndex:1];
            [encoder setBytes:&uniforms length:sizeof(NTTBatchUniforms) atIndex:2];
            [encoder setBytes:&stage_offset length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&stage_count length:sizeof(uint32_t) atIndex:4];
            MTLSize tg = MTLSizeMake(block_size, 1, 1);
            MTLSize ng = MTLSizeMake(num_groups, 1, 1);
            [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
            [encoder endEncoding];

            stage_offset = stage_count;
            stage_count = log_shared - stage_offset;
        }
        if (stage_count > 0) {
            const uint32_t block_size = 1U << stage_count;
            const uint32_t blocks_per_batch = n / block_size;
            const uint32_t num_groups = blocks_per_batch * batches;
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:g_runtime.f_ntt_butterfly_shared_batch];
            [encoder setBuffer:data_buf offset:0 atIndex:0];
            [encoder setBuffer:twiddles offset:0 atIndex:1];
            [encoder setBytes:&uniforms length:sizeof(NTTBatchUniforms) atIndex:2];
            [encoder setBytes:&stage_offset length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&stage_count length:sizeof(uint32_t) atIndex:4];
            MTLSize tg = MTLSizeMake(block_size, 1, 1);
            MTLSize ng = MTLSizeMake(num_groups, 1, 1);
            [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
            [encoder endEncoding];
        }
    }

    for (uint32_t stage = log_shared; stage < log_n; stage++) {
        NTTBatchUniforms uniforms{};
        uniforms.n = n;
        uniforms.log_n = log_n;
        uniforms.stage = stage;
        uniforms.direction = inverse ? 1U : 0U;
        uniforms.twiddle_stride = twiddle_stride;
        uniforms.batches = batches;
        uniforms.batch_stride = n;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:butterfly];
        [encoder setBuffer:data_buf offset:0 atIndex:0];
        [encoder setBuffer:twiddles offset:0 atIndex:1];
        [encoder setBytes:&uniforms length:sizeof(NTTBatchUniforms) atIndex:2];

        const uint64_t num_butterflies = (static_cast<uint64_t>(n) / 2) * static_cast<uint64_t>(batches);
        const NSUInteger threads_per_group = butterfly.threadExecutionWidth;
        const NSUInteger num_groups = (num_butterflies + threads_per_group - 1) / threads_per_group;
        MTLSize tg = MTLSizeMake(threads_per_group, 1, 1);
        MTLSize ng = MTLSizeMake(num_groups, 1, 1);
        [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
        [encoder endEncoding];
    }

    if (inverse) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (coset_inverse) {
            [encoder setComputePipelineState:g_runtime.f_ntt_scale_coset_batch];
            [encoder setBuffer:data_buf offset:0 atIndex:0];
            const NSUInteger n_inv_offset = static_cast<NSUInteger>(log_n * sizeof(uint64_t));
            [encoder setBuffer:g_runtime.n_inverses offset:n_inv_offset atIndex:1];
            [encoder setBuffer:g_runtime.coset_inv_pows offset:0 atIndex:2];
            [encoder setBytes:&n length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&batches length:sizeof(uint32_t) atIndex:4];
        } else {
            [encoder setComputePipelineState:g_runtime.f_ntt_scale_batch];
            [encoder setBuffer:data_buf offset:0 atIndex:0];
            const NSUInteger n_inv_offset = static_cast<NSUInteger>(log_n * sizeof(uint64_t));
            [encoder setBuffer:g_runtime.n_inverses offset:n_inv_offset atIndex:1];
            [encoder setBytes:&n length:sizeof(uint32_t) atIndex:2];
            [encoder setBytes:&batches length:sizeof(uint32_t) atIndex:3];
        }

        const NSUInteger threads_per_group = (coset_inverse ? g_runtime.f_ntt_scale_coset_batch : g_runtime.f_ntt_scale_batch).threadExecutionWidth;
        const NSUInteger num_groups = (total + threads_per_group - 1) / threads_per_group;
        MTLSize tg = MTLSizeMake(threads_per_group, 1, 1);
        MTLSize ng = MTLSizeMake(num_groups, 1, 1);
        [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
        [encoder endEncoding];
    }
}

static void run_timed_ntt_batched(id<MTLBuffer> data_buf,
                                  uint32_t log_n,
                                  uint32_t batches,
                                  bool inverse) {
    const uint32_t n = 1U << log_n;
    const uint32_t twiddle_stride = (1U << g_runtime.max_log_n) / n;
    id<MTLBuffer> twiddles = inverse ? g_runtime.inv_twiddles : g_runtime.twiddles;
    id<MTLComputePipelineState> butterfly = inverse ? g_runtime.f_intt_butterfly_batch : g_runtime.f_ntt_butterfly_batch;

    NTTBatchUniforms base{};
    base.n = n;
    base.log_n = log_n;
    base.stage = 0;
    base.direction = inverse ? 1U : 0U;
    base.twiddle_stride = twiddle_stride;
    base.batches = batches;
    base.batch_stride = n;

    double t_bitrev = run_timed_ntt_stage(g_runtime.f_ntt_bit_reverse_batch, data_buf, nullptr, base, 0, 0, log_n, batches);

    uint32_t max_threads = (uint32_t)g_runtime.f_ntt_butterfly_shared_batch.maxTotalThreadsPerThreadgroup;
    uint32_t log_shared = 0;
    while ((1U << log_shared) < max_threads && log_shared < log_n) {
        log_shared++;
    }
    if ((1U << log_shared) > max_threads) {
        log_shared--;
    }
    if (log_shared > 10) {
        log_shared = 10;
    }
    if (log_shared > 8) {
        log_shared = 8;
    }
    const char *env_log = std::getenv("METAL_NTT_LOG_SHARED");
    if (env_log && env_log[0] != '\0') {
        int override = std::atoi(env_log);
        if (override >= 0 && override <= (int)log_n) {
            log_shared = static_cast<uint32_t>(override);
        }
    }

    double t_shared = 0.0;
    if (log_shared > 0) {
        double t = run_timed_ntt_stage(g_runtime.f_ntt_butterfly_shared_batch, data_buf, twiddles, base, 0, log_shared, log_n, batches);
        t_shared += t;
    }

    double t_main = 0.0;
    for (uint32_t stage = log_shared; stage < log_n; stage++) {
        NTTBatchUniforms u = base;
        u.stage = stage;
        double t = run_timed_ntt_stage(butterfly, data_buf, twiddles, u, 0, 0, log_n, batches);
        t_main += t;
    }

    double t_scale = 0.0;
    if (inverse) {
        id<MTLCommandBuffer> command_buffer = [g_runtime.queue commandBuffer];
        auto start = std::chrono::steady_clock::now();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:g_runtime.f_ntt_scale_batch];
        [encoder setBuffer:data_buf offset:0 atIndex:0];
        const NSUInteger n_inv_offset = static_cast<NSUInteger>(log_n * sizeof(uint64_t));
        [encoder setBuffer:g_runtime.n_inverses offset:n_inv_offset atIndex:1];
        [encoder setBytes:&n length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&batches length:sizeof(uint32_t) atIndex:3];
        const uint64_t total = static_cast<uint64_t>(n) * static_cast<uint64_t>(batches);
        const NSUInteger threads_per_group = g_runtime.f_ntt_scale_batch.threadExecutionWidth;
        const NSUInteger num_groups = (total + threads_per_group - 1) / threads_per_group;
        MTLSize tg = MTLSizeMake(threads_per_group, 1, 1);
        MTLSize ng = MTLSizeMake(num_groups, 1, 1);
        [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        auto end = std::chrono::steady_clock::now();
        t_scale = command_buffer_elapsed_ms(command_buffer, start, end);
    }

    std::fprintf(stderr, "metal ntt timing: bitrev %.3f ms, shared %.3f ms, main %.3f ms, scale %.3f ms\n",
                 t_bitrev, t_shared, t_main, t_scale);
}

static void ntt_in_place(uint64_t *data, uint32_t log_n, bool inverse) {
    ensure_runtime();
    if (log_n > g_runtime.max_log_n) {
        std::fprintf(stderr, "zeknox metal: NTT log_n %u exceeds max_log_n %u\n", log_n, g_runtime.max_log_n);
        std::abort();
    }

    const uint32_t n = 1U << log_n;
    const size_t bytes = static_cast<size_t>(n) * sizeof(uint64_t);
    id<MTLBuffer> data_buf = [g_runtime.device newBufferWithBytesNoCopy:data
                                                                 length:bytes
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
    if (!data_buf) {
        std::fprintf(stderr, "zeknox metal: failed to create data buffer\n");
        std::abort();
    }

    id<MTLCommandBuffer> command_buffer = [g_runtime.queue commandBuffer];
    if (!command_buffer) {
        std::fprintf(stderr, "zeknox metal: failed to create command buffer\n");
        std::abort();
    }

    // Bit-reversal permutation
    {
        NTTUniforms uniforms{};
        uniforms.n = n;
        uniforms.log_n = log_n;
        uniforms.stage = 0;
        uniforms.direction = inverse ? 1U : 0U;
        uniforms.twiddle_stride = 0;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:g_runtime.f_ntt_bit_reverse];
        [encoder setBuffer:data_buf offset:0 atIndex:0];
        [encoder setBytes:&uniforms length:sizeof(NTTUniforms) atIndex:1];

        const NSUInteger threads_per_group = g_runtime.f_ntt_bit_reverse.threadExecutionWidth;
        const NSUInteger num_groups = (n + threads_per_group - 1) / threads_per_group;
        MTLSize tg = MTLSizeMake(threads_per_group, 1, 1);
        MTLSize ng = MTLSizeMake(num_groups, 1, 1);
        [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
        [encoder endEncoding];
    }

    const uint32_t twiddle_stride = (1U << g_runtime.max_log_n) / n;
    id<MTLComputePipelineState> butterfly = inverse ? g_runtime.f_intt_butterfly : g_runtime.f_ntt_butterfly;
    id<MTLBuffer> twiddles = inverse ? g_runtime.inv_twiddles : g_runtime.twiddles;

    // Shared-memory stages (early stages)
    uint32_t max_threads = (uint32_t)g_runtime.f_ntt_butterfly_shared.maxTotalThreadsPerThreadgroup;
    uint32_t log_shared = 0;
    while ((1U << log_shared) < max_threads && log_shared < log_n) {
        log_shared++;
    }
    if ((1U << log_shared) > max_threads) {
        log_shared--;
    }
    if (log_shared > 10) {
        log_shared = 10; // threadgroup array is 1024
    }
    if (log_shared > 8) {
        log_shared = 8; // empirical cap for better performance on extended NTTs
    }
    // Leave log_shared auto-selected; override via METAL_NTT_LOG_SHARED if needed.
    const char *env_log = std::getenv("METAL_NTT_LOG_SHARED");
    if (env_log && env_log[0] != '\0') {
        int override = std::atoi(env_log);
        if (override >= 0 && override <= (int)log_n) {
            log_shared = static_cast<uint32_t>(override);
        }
    }
    const char *env_dbg = std::getenv("METAL_NTT_DEBUG");
    if (env_dbg && env_dbg[0] != '\0') {
        std::fprintf(stderr, "zeknox metal: log_shared=%u log_n=%u\n", log_shared, log_n);
    }

    if (log_shared > 0) {
        NTTUniforms uniforms{};
        uniforms.n = n;
        uniforms.log_n = log_n;
        uniforms.stage = 0;
        uniforms.direction = inverse ? 1U : 0U;
        uniforms.twiddle_stride = twiddle_stride;

        const uint32_t stage_offset = 0;
        const uint32_t stage_count = log_shared;
        const uint32_t block_size = 1U << log_shared;
        const uint32_t num_groups = n / block_size;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:g_runtime.f_ntt_butterfly_shared];
        [encoder setBuffer:data_buf offset:0 atIndex:0];
        [encoder setBuffer:twiddles offset:0 atIndex:1];
        [encoder setBytes:&uniforms length:sizeof(NTTUniforms) atIndex:2];
        [encoder setBytes:&stage_offset length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&stage_count length:sizeof(uint32_t) atIndex:4];
        MTLSize tg = MTLSizeMake(block_size, 1, 1);
        MTLSize ng = MTLSizeMake(num_groups, 1, 1);
        [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
        [encoder endEncoding];
    }

    for (uint32_t stage = log_shared; stage < log_n; stage++) {
        NTTUniforms uniforms{};
        uniforms.n = n;
        uniforms.log_n = log_n;
        uniforms.stage = stage;
        uniforms.direction = inverse ? 1U : 0U;
        uniforms.twiddle_stride = twiddle_stride;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:butterfly];
        [encoder setBuffer:data_buf offset:0 atIndex:0];
        [encoder setBuffer:twiddles offset:0 atIndex:1];
        [encoder setBytes:&uniforms length:sizeof(NTTUniforms) atIndex:2];

        const uint32_t num_butterflies = n / 2;
        const NSUInteger threads_per_group = butterfly.threadExecutionWidth;
        const NSUInteger num_groups = (num_butterflies + threads_per_group - 1) / threads_per_group;
        MTLSize tg = MTLSizeMake(threads_per_group, 1, 1);
        MTLSize ng = MTLSizeMake(num_groups, 1, 1);
        [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
        [encoder endEncoding];
    }

    if (inverse) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:g_runtime.f_ntt_scale];
        [encoder setBuffer:data_buf offset:0 atIndex:0];
        const NSUInteger n_inv_offset = static_cast<NSUInteger>(log_n * sizeof(uint64_t));
        [encoder setBuffer:g_runtime.n_inverses offset:n_inv_offset atIndex:1];
        [encoder setBytes:&n length:sizeof(uint32_t) atIndex:2];

        const NSUInteger threads_per_group = g_runtime.f_ntt_scale.threadExecutionWidth;
        const NSUInteger num_groups = (n + threads_per_group - 1) / threads_per_group;
        MTLSize tg = MTLSizeMake(threads_per_group, 1, 1);
        MTLSize ng = MTLSizeMake(num_groups, 1, 1);
        [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
        [encoder endEncoding];
    }

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}



static size_t log2_pow2(size_t n) {
    size_t r = 0;
    while ((size_t(1) << r) < n) {
        r++;
    }
    return r;
}

static void run_merkle_linear_poseidon_transpose_rev(
    void *digests_buf_ptr,
    void *cap_buf_ptr,
    const void *lde_buf_ptr,
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height
) {
    ensure_runtime();

    const size_t leaves_count = static_cast<size_t>(leaves_buf_size);
    const size_t leaf_len = static_cast<size_t>(leaf_size);
    const size_t cap_h = static_cast<size_t>(cap_height);

    const size_t subtree_count = size_t(1) << cap_h;
    const size_t subtree_leaves_len = leaves_count >> cap_h;
    const size_t subtree_digests_len = static_cast<size_t>(digests_buf_size) / subtree_count;
    const size_t expected_subtree_digests_len = 2 * (subtree_leaves_len - 1);
    const size_t total_digests = subtree_digests_len * subtree_count;

    if (digests_buf_size < total_digests) {
        std::fprintf(stderr, "zeknox metal: digests_buf_size too small (got %llu need %zu)\n",
                     static_cast<unsigned long long>(digests_buf_size), total_digests);
        std::abort();
    }
    if (subtree_digests_len != expected_subtree_digests_len) {
        std::fprintf(stderr,
                     "zeknox metal: unexpected subtree_digests_len (got %zu expected %zu)\n",
                     subtree_digests_len, expected_subtree_digests_len);
        std::abort();
    }
    if (cap_buf_size < subtree_count) {
        std::fprintf(stderr, "zeknox metal: cap_buf_size too small (got %llu need %zu)\n",
                     static_cast<unsigned long long>(cap_buf_size), subtree_count);
        std::abort();
    }

    const size_t tree_height = log2_pow2(leaves_count);
    const uint32_t log_n = static_cast<uint32_t>(tree_height);

    @autoreleasepool {
        const size_t total_in = leaves_count * leaf_len;
        const size_t in_bytes = total_in * sizeof(uint64_t);
        id<MTLBuffer> in_buf = [g_runtime.device newBufferWithBytesNoCopy:(void *)lde_buf_ptr
                                                                   length:in_bytes
                                                                  options:MTLResourceStorageModeShared
                                                              deallocator:nil];
        if (!in_buf) {
            std::fprintf(stderr, "zeknox metal: failed to create input buffer\n");
            std::abort();
        }

        const size_t digests_bytes = total_digests * 4 * sizeof(uint64_t);
        id<MTLBuffer> digests_buffer = [g_runtime.device newBufferWithBytesNoCopy:digests_buf_ptr
                                                                           length:digests_bytes
                                                                          options:MTLResourceStorageModeShared
                                                                      deallocator:nil];
        const size_t caps_bytes = subtree_count * 4 * sizeof(uint64_t);
        id<MTLBuffer> caps_buffer = [g_runtime.device newBufferWithBytesNoCopy:cap_buf_ptr
                                                                        length:caps_bytes
                                                                       options:MTLResourceStorageModeShared
                                                                   deallocator:nil];
        if (!digests_buffer || !caps_buffer) {
            std::fprintf(stderr, "zeknox metal: failed to create output buffers\n");
            std::abort();
        }
        std::memset([digests_buffer contents], 0, digests_bytes);
        std::memset([caps_buffer contents], 0, caps_bytes);

        TransposeLeafUniforms uniforms{};
        uniforms.n = static_cast<uint32_t>(leaves_count);
        uniforms.log_n = log_n;
        uniforms.batches = static_cast<uint32_t>(leaf_len);
        uniforms.subtree_digests_len = static_cast<uint32_t>(subtree_digests_len);
        uniforms.subtree_leaves_len = static_cast<uint32_t>(subtree_leaves_len);
        uniforms.leaf_size = static_cast<uint32_t>(leaf_len);
        uniforms.leaf_count = static_cast<uint32_t>(leaves_count);
        uniforms.subtree_count = static_cast<uint32_t>(subtree_count);
        uniforms.grid_width = 0;

        id<MTLCommandBuffer> command_buffer = [g_runtime.queue commandBuffer];
        if (!command_buffer) {
            std::fprintf(stderr, "zeknox metal: failed to create command buffer\n");
            std::abort();
        }

        {
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:g_runtime.f_hash_leaves_transpose_rev];
            [encoder setBuffer:in_buf offset:0 atIndex:0];
            [encoder setBuffer:digests_buffer offset:0 atIndex:1];

            NSUInteger threads_per_group = merkle_threads_per_group(g_runtime.f_hash_leaves_transpose_rev);
            uint64_t groups = (leaves_count + threads_per_group - 1) / threads_per_group;
            MTLSize group_count;
            get_threadgroup_count(groups, &group_count);
            MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
            uniforms.grid_width = static_cast<uint32_t>(group_count.width * threads_per_group);
            [encoder setBytes:&uniforms length:sizeof(TransposeLeafUniforms) atIndex:2];
            [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
            [encoder endEncoding];
        }

        const size_t num_layers = tree_height - cap_h;
        for (size_t level = 1; level < num_layers;) {
            if (level + 1 < num_layers) {
                LinearUniforms lvl{};
                lvl.level = static_cast<uint32_t>(level + 1);
                lvl.subtree_digests_len = static_cast<uint32_t>(subtree_digests_len);
                lvl.subtree_leaves_len = static_cast<uint32_t>(subtree_leaves_len);
                lvl.leaf_size = static_cast<uint32_t>(leaf_len);
                lvl.leaf_count = static_cast<uint32_t>(leaves_count);
                lvl.subtree_count = static_cast<uint32_t>(subtree_count);
                lvl.grid_width = 0;

                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                [encoder setComputePipelineState:g_runtime.f_hash_tree_level_2];
                [encoder setBuffer:digests_buffer offset:0 atIndex:0];
                [encoder setBytes:&lvl length:sizeof(LinearUniforms) atIndex:1];

                size_t nodes_at_level = subtree_leaves_len >> (level + 1);
                size_t total_nodes = nodes_at_level * subtree_count;
                NSUInteger threads_per_group = merkle_threads_per_group(g_runtime.f_hash_tree_level_2);
                uint64_t groups = (total_nodes + threads_per_group - 1) / threads_per_group;
                MTLSize group_count;
                get_threadgroup_count(groups, &group_count);
                MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
                lvl.grid_width = static_cast<uint32_t>(group_count.width * threads_per_group);
                [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
                [encoder endEncoding];
                level += 2;
            } else {
                LinearUniforms lvl{};
                lvl.level = static_cast<uint32_t>(level);
                lvl.subtree_digests_len = static_cast<uint32_t>(subtree_digests_len);
                lvl.subtree_leaves_len = static_cast<uint32_t>(subtree_leaves_len);
                lvl.leaf_size = static_cast<uint32_t>(leaf_len);
                lvl.leaf_count = static_cast<uint32_t>(leaves_count);
                lvl.subtree_count = static_cast<uint32_t>(subtree_count);
                lvl.grid_width = 0;

                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                [encoder setComputePipelineState:g_runtime.f_hash_tree_level];
                [encoder setBuffer:digests_buffer offset:0 atIndex:0];
                [encoder setBytes:&lvl length:sizeof(LinearUniforms) atIndex:1];

                size_t nodes_at_level = subtree_leaves_len >> level;
                size_t total_nodes = nodes_at_level * subtree_count;
                NSUInteger threads_per_group = merkle_threads_per_group(g_runtime.f_hash_tree_level);
                uint64_t groups = (total_nodes + threads_per_group - 1) / threads_per_group;
                MTLSize group_count;
                get_threadgroup_count(groups, &group_count);
                MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
                lvl.grid_width = static_cast<uint32_t>(group_count.width * threads_per_group);
                [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
                [encoder endEncoding];
                level += 1;
            }
        }

        {
            LinearUniforms lvl{};
            lvl.level = 0;
            lvl.subtree_digests_len = static_cast<uint32_t>(subtree_digests_len);
            lvl.subtree_leaves_len = static_cast<uint32_t>(subtree_leaves_len);
            lvl.leaf_size = static_cast<uint32_t>(leaf_len);
            lvl.leaf_count = static_cast<uint32_t>(leaves_count);
            lvl.subtree_count = static_cast<uint32_t>(subtree_count);
            lvl.grid_width = 0;

            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:g_runtime.f_hash_caps];
            [encoder setBuffer:caps_buffer offset:0 atIndex:0];
            [encoder setBuffer:digests_buffer offset:0 atIndex:1];
            [encoder setBytes:&lvl length:sizeof(LinearUniforms) atIndex:2];

            NSUInteger threads_per_group = merkle_threads_per_group(g_runtime.f_hash_caps);
            uint64_t groups = (subtree_count + threads_per_group - 1) / threads_per_group;
            MTLSize group_count = MTLSizeMake(std::max<uint64_t>(1, groups), 1, 1);
            MTLSize group_size = MTLSizeMake(std::min<uint64_t>(threads_per_group, subtree_count), 1, 1);
            [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
            [encoder endEncoding];
        }

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }
}

static void run_merkle_linear_poseidon(
    void *digests_buf_ptr,
    void *cap_buf_ptr,
    const void *leaves_buf_ptr,
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height
) {
    ensure_runtime();

    const size_t leaves_count = static_cast<size_t>(leaves_buf_size);
    const size_t leaf_len = static_cast<size_t>(leaf_size);
    const size_t cap_h = static_cast<size_t>(cap_height);

    const size_t subtree_count = size_t(1) << cap_h;
    const size_t subtree_leaves_len = leaves_count >> cap_h;
    const size_t subtree_digests_len = static_cast<size_t>(digests_buf_size) / subtree_count;
    const size_t expected_subtree_digests_len = 2 * (subtree_leaves_len - 1);
    const size_t total_digests = subtree_digests_len * subtree_count;

    if (digests_buf_size < total_digests) {
        std::fprintf(stderr, "zeknox metal: digests_buf_size too small (got %llu need %zu)\n",
                     static_cast<unsigned long long>(digests_buf_size), total_digests);
        std::abort();
    }
    if (subtree_digests_len != expected_subtree_digests_len) {
        std::fprintf(stderr,
                     "zeknox metal: unexpected subtree_digests_len (got %zu expected %zu)\n",
                     subtree_digests_len, expected_subtree_digests_len);
        std::abort();
    }
    if (cap_buf_size < subtree_count) {
        std::fprintf(stderr, "zeknox metal: cap_buf_size too small (got %llu need %zu)\n",
                     static_cast<unsigned long long>(cap_buf_size), subtree_count);
        std::abort();
    }

    const size_t tree_height = log2_pow2(leaves_count);
    const char *timing_env = std::getenv("METAL_MERKLE_TIMING");
    const bool timing = timing_env && timing_env[0] != '\0';
    const char *timing_verbose_env = std::getenv("METAL_MERKLE_TIMING_VERBOSE");
    const bool timing_verbose = timing_verbose_env && timing_verbose_env[0] != '\0';

    @autoreleasepool {
        const size_t leaves_u64 = leaves_count * leaf_len;
        const size_t leaves_bytes = leaves_u64 * sizeof(uint64_t);
        id<MTLBuffer> leaves_buffer = [g_runtime.device newBufferWithBytes:leaves_buf_ptr
                                                                    length:leaves_bytes
                                                                   options:MTLResourceStorageModeShared];

        const size_t digests_bytes = total_digests * 4 * sizeof(uint64_t);
        id<MTLBuffer> digests_buffer = [g_runtime.device newBufferWithLength:digests_bytes
                                                                     options:MTLResourceStorageModeShared];
        std::memset([digests_buffer contents], 0, digests_bytes);

        const size_t caps_bytes = subtree_count * 4 * sizeof(uint64_t);
        id<MTLBuffer> caps_buffer = [g_runtime.device newBufferWithLength:caps_bytes
                                                                  options:MTLResourceStorageModeShared];
        std::memset([caps_buffer contents], 0, caps_bytes);

        LinearUniforms uniforms;
        uniforms.level = 0;
        uniforms.subtree_digests_len = static_cast<uint32_t>(subtree_digests_len);
        uniforms.subtree_leaves_len = static_cast<uint32_t>(subtree_leaves_len);
        uniforms.leaf_size = static_cast<uint32_t>(leaf_len);
        uniforms.leaf_count = static_cast<uint32_t>(leaves_count);
        uniforms.subtree_count = static_cast<uint32_t>(subtree_count);
        uniforms.grid_width = 0;

        if (!timing) {
            id<MTLCommandBuffer> command_buffer = [g_runtime.queue commandBuffer];

            // Step 1: hash leaves
            {
                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                [encoder setComputePipelineState:g_runtime.f_hash_leaves];
                [encoder setBuffer:leaves_buffer offset:0 atIndex:0];
                [encoder setBuffer:digests_buffer offset:0 atIndex:1];

                NSUInteger threads_per_group = merkle_threads_per_group(g_runtime.f_hash_leaves);
                uint64_t groups = (leaves_count + threads_per_group - 1) / threads_per_group;
                MTLSize group_count;
                get_threadgroup_count(groups, &group_count);
                MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
                uniforms.grid_width = static_cast<uint32_t>(group_count.width * threads_per_group);
                id<MTLBuffer> uniforms_buffer = [g_runtime.device newBufferWithBytes:&uniforms
                                                                              length:sizeof(LinearUniforms)
                                                                             options:MTLResourceStorageModeShared];
                [encoder setBuffer:uniforms_buffer offset:0 atIndex:2];
                [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
                [encoder endEncoding];
            }

            // Step 2: internal levels (fuse two levels when possible)
            const size_t num_layers = tree_height - cap_h;
            for (size_t level = 1; level < num_layers;) {
                if (level + 1 < num_layers) {
                    // Two-level kernel computes levels (level) and (level+1) in one dispatch.
                    uniforms.level = static_cast<uint32_t>(level + 1);
                    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                    [encoder setComputePipelineState:g_runtime.f_hash_tree_level_2];
                    [encoder setBuffer:digests_buffer offset:0 atIndex:0];
                    [encoder setBytes:&uniforms length:sizeof(LinearUniforms) atIndex:1];

                    size_t nodes_at_level = subtree_leaves_len >> (level + 1);
                    size_t total_nodes = nodes_at_level * subtree_count;
                    NSUInteger threads_per_group = merkle_threads_per_group(g_runtime.f_hash_tree_level_2);
                    uint64_t groups = (total_nodes + threads_per_group - 1) / threads_per_group;
                    MTLSize group_count;
                    get_threadgroup_count(groups, &group_count);
                    MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
                    uniforms.grid_width = static_cast<uint32_t>(group_count.width * threads_per_group);
                    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
                    [encoder endEncoding];
                    level += 2;
                } else {
                    uniforms.level = static_cast<uint32_t>(level);

                    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                    [encoder setComputePipelineState:g_runtime.f_hash_tree_level];
                    [encoder setBuffer:digests_buffer offset:0 atIndex:0];
                    [encoder setBytes:&uniforms length:sizeof(LinearUniforms) atIndex:1];

                    size_t nodes_at_level = subtree_leaves_len >> level;
                    size_t total_nodes = nodes_at_level * subtree_count;
                    NSUInteger threads_per_group = merkle_threads_per_group(g_runtime.f_hash_tree_level);
                    uint64_t groups = (total_nodes + threads_per_group - 1) / threads_per_group;
                    MTLSize group_count;
                    get_threadgroup_count(groups, &group_count);
                    MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
                    uniforms.grid_width = static_cast<uint32_t>(group_count.width * threads_per_group);
                    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
                    [encoder endEncoding];
                    level += 1;
                }
            }

            // Step 3: caps
            {
                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                [encoder setComputePipelineState:g_runtime.f_hash_caps];
                [encoder setBuffer:caps_buffer offset:0 atIndex:0];
                [encoder setBuffer:digests_buffer offset:0 atIndex:1];

                id<MTLBuffer> uniforms_buffer = [g_runtime.device newBufferWithBytes:&uniforms
                                                                              length:sizeof(LinearUniforms)
                                                                             options:MTLResourceStorageModeShared];
                [encoder setBuffer:uniforms_buffer offset:0 atIndex:2];

                NSUInteger threads_per_group = merkle_threads_per_group(g_runtime.f_hash_caps);
                uint64_t groups = (subtree_count + threads_per_group - 1) / threads_per_group;
                MTLSize group_count = MTLSizeMake(std::max<uint64_t>(1, groups), 1, 1);
                MTLSize group_size = MTLSizeMake(std::min<uint64_t>(threads_per_group, subtree_count), 1, 1);
                [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
                [encoder endEncoding];
            }

            [command_buffer commit];
            [command_buffer waitUntilCompleted];
        } else {
            double t_leaves = 0.0;
            double t_internal = 0.0;
            double t_caps = 0.0;

            // Step 1: leaves
            {
                id<MTLCommandBuffer> command_buffer = [g_runtime.queue commandBuffer];
                auto start = std::chrono::steady_clock::now();
                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                [encoder setComputePipelineState:g_runtime.f_hash_leaves];
                [encoder setBuffer:leaves_buffer offset:0 atIndex:0];
                [encoder setBuffer:digests_buffer offset:0 atIndex:1];

                NSUInteger threads_per_group = merkle_threads_per_group(g_runtime.f_hash_leaves);
                uint64_t groups = (leaves_count + threads_per_group - 1) / threads_per_group;
                MTLSize group_count;
                get_threadgroup_count(groups, &group_count);
                MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
                uniforms.grid_width = static_cast<uint32_t>(group_count.width * threads_per_group);
                id<MTLBuffer> uniforms_buffer = [g_runtime.device newBufferWithBytes:&uniforms
                                                                              length:sizeof(LinearUniforms)
                                                                             options:MTLResourceStorageModeShared];
                [encoder setBuffer:uniforms_buffer offset:0 atIndex:2];
                [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
                [encoder endEncoding];
                [command_buffer commit];
                [command_buffer waitUntilCompleted];
                auto end = std::chrono::steady_clock::now();
                t_leaves = command_buffer_elapsed_ms(command_buffer, start, end);
            }

            const size_t num_layers = tree_height - cap_h;
            for (size_t level = 1; level < num_layers;) {
                if (level + 1 < num_layers) {
                    uniforms.level = static_cast<uint32_t>(level + 1);
                    id<MTLCommandBuffer> command_buffer = [g_runtime.queue commandBuffer];
                    auto start = std::chrono::steady_clock::now();
                    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                    [encoder setComputePipelineState:g_runtime.f_hash_tree_level_2];
                    [encoder setBuffer:digests_buffer offset:0 atIndex:0];
                    [encoder setBytes:&uniforms length:sizeof(LinearUniforms) atIndex:1];

                    size_t nodes_at_level = subtree_leaves_len >> (level + 1);
                    size_t total_nodes = nodes_at_level * subtree_count;
                    NSUInteger threads_per_group = merkle_threads_per_group(g_runtime.f_hash_tree_level_2);
                    uint64_t groups = (total_nodes + threads_per_group - 1) / threads_per_group;
                    MTLSize group_count;
                    get_threadgroup_count(groups, &group_count);
                    MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
                    uniforms.grid_width = static_cast<uint32_t>(group_count.width * threads_per_group);
                    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
                    [encoder endEncoding];
                    [command_buffer commit];
                    [command_buffer waitUntilCompleted];
                    auto end = std::chrono::steady_clock::now();
                    double t = command_buffer_elapsed_ms(command_buffer, start, end);
                    t_internal += t;
                    if (timing_verbose) {
                        std::fprintf(stderr, "metal merkle timing: internal levels %zu-%zu %.3f ms\n", level, level + 1, t);
                    }
                    level += 2;
                } else {
                    uniforms.level = static_cast<uint32_t>(level);
                    id<MTLCommandBuffer> command_buffer = [g_runtime.queue commandBuffer];
                    auto start = std::chrono::steady_clock::now();
                    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                    [encoder setComputePipelineState:g_runtime.f_hash_tree_level];
                    [encoder setBuffer:digests_buffer offset:0 atIndex:0];
                    [encoder setBytes:&uniforms length:sizeof(LinearUniforms) atIndex:1];

                    size_t nodes_at_level = subtree_leaves_len >> level;
                    size_t total_nodes = nodes_at_level * subtree_count;
                    NSUInteger threads_per_group = merkle_threads_per_group(g_runtime.f_hash_tree_level);
                    uint64_t groups = (total_nodes + threads_per_group - 1) / threads_per_group;
                    MTLSize group_count;
                    get_threadgroup_count(groups, &group_count);
                    MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
                    uniforms.grid_width = static_cast<uint32_t>(group_count.width * threads_per_group);
                    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
                    [encoder endEncoding];
                    [command_buffer commit];
                    [command_buffer waitUntilCompleted];
                    auto end = std::chrono::steady_clock::now();
                    double t = command_buffer_elapsed_ms(command_buffer, start, end);
                    t_internal += t;
                    if (timing_verbose) {
                        std::fprintf(stderr, "metal merkle timing: internal level %zu %.3f ms\n", level, t);
                    }
                    level += 1;
                }
            }

            // Step 3: caps
            {
                id<MTLCommandBuffer> command_buffer = [g_runtime.queue commandBuffer];
                auto start = std::chrono::steady_clock::now();
                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                [encoder setComputePipelineState:g_runtime.f_hash_caps];
                [encoder setBuffer:caps_buffer offset:0 atIndex:0];
                [encoder setBuffer:digests_buffer offset:0 atIndex:1];

                id<MTLBuffer> uniforms_buffer = [g_runtime.device newBufferWithBytes:&uniforms
                                                                              length:sizeof(LinearUniforms)
                                                                             options:MTLResourceStorageModeShared];
                [encoder setBuffer:uniforms_buffer offset:0 atIndex:2];

                NSUInteger threads_per_group = merkle_threads_per_group(g_runtime.f_hash_caps);
                uint64_t groups = (subtree_count + threads_per_group - 1) / threads_per_group;
                MTLSize group_count = MTLSizeMake(std::max<uint64_t>(1, groups), 1, 1);
                MTLSize group_size = MTLSizeMake(std::min<uint64_t>(threads_per_group, subtree_count), 1, 1);
                [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
                [encoder endEncoding];
                [command_buffer commit];
                [command_buffer waitUntilCompleted];
                auto end = std::chrono::steady_clock::now();
                t_caps = command_buffer_elapsed_ms(command_buffer, start, end);
            }

            std::fprintf(stderr, "metal merkle timing: leaves %.3f ms, internal %.3f ms, caps %.3f ms\n",
                         t_leaves, t_internal, t_caps);
        }

        // Copy results back to host
        std::memcpy(digests_buf_ptr, [digests_buffer contents], digests_bytes);
        std::memcpy(cap_buf_ptr, [caps_buffer contents], caps_bytes);
    }
}

static RustError not_implemented(const char *fn_name) {
    return RustError(1, fn_name);
}

static void abort_not_implemented(const char *fn_name) {
    std::fprintf(stderr, "zeknox metal stub: %s is not implemented\n", fn_name);
    std::abort();
}

} // namespace

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern "C" RustError get_number_of_gpus(size_t *ngpus) {
    if (ngpus) {
        *ngpus = 1;
    }
    return RustError(0);
}

extern "C" RustError list_devices_info() {
    ensure_runtime();
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
    ensure_runtime();
    return RustError(0);
}

extern "C" RustError init_coset(size_t device_id, size_t lg_domain_size, const uint64_t coset_gen) {
    (void)device_id;
    (void)lg_domain_size;
    (void)coset_gen;
    ensure_runtime();
    return RustError(0);
}

extern "C" RustError compute_batched_ntt(
    size_t device_id,
    void *inout,
    uint32_t lg_domain_size,
    NTT_Direction ntt_direction,
    NTT_Config cfg) {
    (void)device_id;
    if (cfg.order != NN) {
        return not_implemented("compute_batched_ntt (metal, non-NN order)");
    }

    ensure_runtime();
    const uint32_t log_n = lg_domain_size;
    if (log_n > g_runtime.max_log_n) {
        return RustError(1, "metal: log_n exceeds max_log_n");
    }

    const uint32_t n = 1U << log_n;
    const uint64_t total = static_cast<uint64_t>(n) * static_cast<uint64_t>(cfg.batches);
    uint64_t *data = reinterpret_cast<uint64_t *>(inout);
    const bool is_inverse = (ntt_direction == inverse);

    const size_t total_bytes = static_cast<size_t>(total) * sizeof(uint64_t);
    id<MTLBuffer> data_buf = get_or_resize_buffer(&g_runtime.cached_in_buf, &g_runtime.cached_in_bytes, total_bytes);
    if (!data_buf) {
        return RustError(1, "metal: failed to create NTT data buffer");
    }
    std::memcpy([data_buf contents], data, total_bytes);

    id<MTLCommandBuffer> command_buffer = [g_runtime.queue commandBuffer];
    if (!command_buffer) {
        return RustError(1, "metal: failed to create NTT command buffer");
    }

    const bool coset = (cfg.ntt_type == coset);
    encode_ntt_batched(command_buffer, data_buf, log_n, cfg.batches, is_inverse, coset && !is_inverse, coset && is_inverse);

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    std::memcpy(data, [data_buf contents], total_bytes);

    return RustError(0);
}

extern "C" RustError compute_batched_ntt_repeats(
    size_t device_id,
    void *inout,
    uint32_t lg_domain_size,
    NTT_Direction ntt_direction,
    NTT_Config cfg,
    uint32_t repeats) {
    (void)device_id;
    if (cfg.order != NN) {
        return not_implemented("compute_batched_ntt_repeats (metal, non-NN order)");
    }

    ensure_runtime();
    const uint32_t log_n = lg_domain_size;
    if (log_n > g_runtime.max_log_n) {
        return RustError(1, "metal: log_n exceeds max_log_n");
    }

    const uint32_t n = 1U << log_n;
    const uint64_t total = static_cast<uint64_t>(n) * static_cast<uint64_t>(cfg.batches);
    uint64_t *data = reinterpret_cast<uint64_t *>(inout);
    const bool is_inverse = (ntt_direction == inverse);

    const size_t total_bytes = static_cast<size_t>(total) * sizeof(uint64_t);
    id<MTLBuffer> data_buf = get_or_resize_buffer(&g_runtime.cached_in_buf, &g_runtime.cached_in_bytes, total_bytes);
    if (!data_buf) {
        return RustError(1, "metal: failed to create NTT data buffer");
    }
    std::memcpy([data_buf contents], data, total_bytes);

    id<MTLCommandBuffer> command_buffer = [g_runtime.queue commandBuffer];
    if (!command_buffer) {
        return RustError(1, "metal: failed to create NTT command buffer");
    }

    const bool coset = (cfg.ntt_type == coset);
    if (repeats < 1) {
        repeats = 1;
    }
    for (uint32_t r = 0; r < repeats; r++) {
        encode_ntt_batched(command_buffer, data_buf, log_n, cfg.batches, is_inverse, coset && !is_inverse, coset && is_inverse);
    }

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    std::memcpy(data, [data_buf contents], total_bytes);

    return RustError(0);
}

extern "C" RustError compute_batched_lde(
    size_t device_id,
    void *output,
    void *input,
    uint32_t lg_domain_size,
    NTT_Direction ntt_direction,
    NTT_Config cfg) {
    (void)device_id;
    if (cfg.order != NN) {
        return not_implemented("compute_batched_lde (metal, non-NN order)");
    }
    if (ntt_direction != forward) {
        return not_implemented("compute_batched_lde (metal, non-forward)");
    }

    if (lg_domain_size <= 13) {
        // CPU fallback for very small sizes; Metal command buffer overhead dominates.
        const uint32_t log_n = lg_domain_size;
        const uint32_t ext_log_n = log_n + cfg.extension_rate_bits;
        const uint32_t ext_n = 1U << ext_log_n;
        const uint64_t *input_u64 = reinterpret_cast<const uint64_t *>(input);
        uint64_t *output_u64 = reinterpret_cast<uint64_t *>(output);
        const uint64_t total = static_cast<uint64_t>(cfg.batches) * static_cast<uint64_t>(ext_n);
        std::vector<uint64_t> coeffs(total, 0);

        for (uint32_t b = 0; b < cfg.batches; b++) {
            const uint64_t *batch_in = input_u64 + static_cast<size_t>(b) * (1U << log_n);
            uint64_t *batch_out = coeffs.data() + static_cast<size_t>(b) * ext_n;
            std::memcpy(batch_out, batch_in, static_cast<size_t>(1U << log_n) * sizeof(uint64_t));
            if (cfg.with_coset) {
                uint64_t shift_pow = 1;
                for (uint32_t i = 0; i < ext_n; i++) {
                    batch_out[i] = mul_mod(batch_out[i], shift_pow);
                    shift_pow = mul_mod(shift_pow, kGoldilocksCosetShift);
                }
            }
            ntt_in_place(batch_out, ext_log_n, false);
        }

        std::memcpy(output_u64, coeffs.data(), static_cast<size_t>(total) * sizeof(uint64_t));
        return RustError(0);
    }

    ensure_runtime();
    const uint32_t log_n = lg_domain_size;
    const uint32_t ext_log_n = log_n + cfg.extension_rate_bits;
    if (ext_log_n > g_runtime.max_log_n) {
        return RustError(1, "metal: ext_log_n exceeds max_log_n");
    }
    const uint32_t n = 1U << log_n;
    const uint32_t ext_n = 1U << ext_log_n;
    const uint64_t total_in = static_cast<uint64_t>(n) * static_cast<uint64_t>(cfg.batches);
    const uint64_t total_out = static_cast<uint64_t>(ext_n) * static_cast<uint64_t>(cfg.batches);

    const size_t in_bytes = static_cast<size_t>(total_in) * sizeof(uint64_t);
    const size_t out_bytes = static_cast<size_t>(total_out) * sizeof(uint64_t);
    id<MTLBuffer> input_buf = get_or_resize_buffer(&g_runtime.cached_in_buf, &g_runtime.cached_in_bytes, in_bytes);
    id<MTLBuffer> output_buf = get_or_resize_buffer(&g_runtime.cached_out_buf, &g_runtime.cached_out_bytes, out_bytes);
    if (!input_buf || !output_buf) {
        return RustError(1, "metal: failed to create LDE buffers");
    }
    std::memcpy([input_buf contents], input, in_bytes);

    id<MTLCommandBuffer> command_buffer = [g_runtime.queue commandBuffer];
    if (!command_buffer) {
        return RustError(1, "metal: failed to create LDE command buffer");
    }

    const char *timing_env = std::getenv("METAL_NTT_TIMING");
    const bool timing = timing_env && timing_env[0] != '\0';
    const char *batch_env = std::getenv("METAL_LDE_BATCH_RUNS");
    int batch_runs = 1;
    if (batch_env && batch_env[0] != '\0') {
        int v = std::atoi(batch_env);
        if (v > 1) {
            batch_runs = v;
        }
    }
    if (timing) {
        double t_extend = run_timed_extend_inputs_batch(output_buf, input_buf, n, ext_n, cfg.batches);
        std::fprintf(stderr, "metal lde timing: extend %.3f ms\n", t_extend);
        if (cfg.with_coset) {
            double t_coset = run_timed_batch_vector_mult(output_buf, ext_n, cfg.batches, false);
            std::fprintf(stderr, "metal lde timing: coset %.3f ms\n", t_coset);
        }
        run_timed_ntt_batched(output_buf, ext_log_n, cfg.batches, false);
        std::memcpy(output, [output_buf contents], out_bytes);
    } else {
        for (int r = 0; r < batch_runs; r++) {
            if (cfg.with_coset) {
                encode_extend_inputs_coset_batch(command_buffer, output_buf, input_buf, n, ext_n, cfg.batches, false);
            } else {
                encode_extend_inputs_batch(command_buffer, output_buf, input_buf, n, ext_n, cfg.batches);
            }
            encode_ntt_batched(command_buffer, output_buf, ext_log_n, cfg.batches, false, false, false);
        }
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        std::memcpy(output, [output_buf contents], out_bytes);
    }

    return RustError(0);
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
    if (num_gpu != 1) {
        return not_implemented("compute_batched_lde_multi_gpu (metal, num_gpu != 1)");
    }
    return compute_batched_lde(0, output, input, lg_domain_size, ntt_direction, cfg);
}

extern "C" RustError compute_transpose_rev(
    size_t device_id,
    void *output,
    void *input,
    uint32_t lg_n,
    NTT_TransposeConfig cfg) {
    (void)device_id;
    ensure_runtime();
    const uint32_t n = 1U << lg_n;
    const uint64_t total = static_cast<uint64_t>(n) * static_cast<uint64_t>(cfg.batches);

    id<MTLBuffer> in_buf = [g_runtime.device newBufferWithBytesNoCopy:input
                                                               length:total * sizeof(uint64_t)
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
    id<MTLBuffer> out_buf = [g_runtime.device newBufferWithBytesNoCopy:output
                                                                length:total * sizeof(uint64_t)
                                                               options:MTLResourceStorageModeShared
                                                           deallocator:nil];
    if (!in_buf || !out_buf) {
        std::fprintf(stderr, "zeknox metal: failed to create transpose buffers\n");
        std::abort();
    }

    TransposeUniforms uniforms{};
    uniforms.n = n;
    uniforms.log_n = lg_n;
    uniforms.batches = cfg.batches;

    id<MTLCommandBuffer> command_buffer = [g_runtime.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:g_runtime.f_transpose_rev];
    [encoder setBuffer:in_buf offset:0 atIndex:0];
    [encoder setBuffer:out_buf offset:0 atIndex:1];
    [encoder setBytes:&uniforms length:sizeof(TransposeUniforms) atIndex:2];

    const NSUInteger tile = 32;
    MTLSize tg = MTLSizeMake(tile, tile, 1);
    MTLSize ng = MTLSizeMake((n + tile - 1) / tile, (cfg.batches + tile - 1) / tile, 1);
    [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tg];
    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    return RustError(0);
}

extern "C" RustError compute_naive_transpose_rev(
    size_t device_id,
    void *output,
    void *input,
    uint32_t lg_n,
    NTT_TransposeConfig cfg) {
    (void)device_id;
    return compute_transpose_rev(device_id, output, input, lg_n, cfg);
}
#pragma clang diagnostic pop

extern "C" void fill_digests_buf_linear_gpu_with_gpu_ptr_transpose_rev(
    void *digests_buf_gpu_ptr,
    void *cap_buf_gpu_ptr,
    void *lde_buf_gpu_ptr,
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height,
    u64 hash_type,
    u64 gpu_id) {
    (void)gpu_id;
    if (hash_type != HashPoseidon) {
        abort_not_implemented("fill_digests_buf_linear_gpu_with_gpu_ptr_transpose_rev (metal, non-poseidon)");
    }
    run_merkle_linear_poseidon_transpose_rev(
        digests_buf_gpu_ptr,
        cap_buf_gpu_ptr,
        lde_buf_gpu_ptr,
        digests_buf_size,
        cap_buf_size,
        leaves_buf_size,
        leaf_size,
        cap_height
    );
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
    (void)gpu_id;
    if (hash_type != HashPoseidon) {
        abort_not_implemented("fill_digests_buf_linear_gpu_with_gpu_ptr (metal, non-poseidon)");
    }
    run_merkle_linear_poseidon(
        digests_buf_gpu_ptr,
        cap_buf_gpu_ptr,
        leaves_buf_gpu_ptr,
        digests_buf_size,
        cap_buf_size,
        leaves_buf_size,
        leaf_size,
        cap_height
    );
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
    if (hash_type != HashPoseidon) {
        abort_not_implemented("fill_digests_buf_linear_multigpu_with_gpu_ptr (metal, non-poseidon)");
    }
    run_merkle_linear_poseidon(
        digests_buf_gpu_ptr,
        cap_buf_gpu_ptr,
        leaves_buf_gpu_ptr,
        digests_buf_size,
        cap_buf_size,
        leaves_buf_size,
        leaf_size,
        cap_height
    );
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
    if (hash_type != HashPoseidon) {
        abort_not_implemented("fill_digests_buf_linear_cpu (metal, non-poseidon)");
    }
    run_merkle_linear_poseidon(
        digests_buf_ptr,
        cap_buf_ptr,
        leaves_buf_ptr,
        digests_buf_size,
        cap_buf_size,
        leaves_buf_size,
        leaf_size,
        cap_height
    );
}
