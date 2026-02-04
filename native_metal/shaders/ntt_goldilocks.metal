// NTT (Number Theoretic Transform) implementation for Goldilocks field
// Based on zeknox CUDA implementation patterns

#include <metal_stdlib>
#include "goldilocks.metal"

using namespace metal;
using namespace GoldilocksField;

// Goldilocks prime: p = 2^64 - 2^32 + 1
constant ulong GL_PRIME = 18446744069414584321UL;

// Helper: Goldilocks addition with reduction
inline ulong gl_add(ulong a, ulong b) {
    ulong sum = a + b;
    // If overflow occurred or sum >= prime, reduce
    if (sum < a || sum >= GL_PRIME) {
        sum -= GL_PRIME;
    }
    return sum;
}

// Helper: Goldilocks subtraction with reduction
inline ulong gl_sub(ulong a, ulong b) {
    if (a >= b) {
        return a - b;
    } else {
        // a < b, so result would be negative
        // return a - b + p = a + (p - b)
        return a + (GL_PRIME - b);
    }
}

// Helper: Goldilocks multiplication using 128-bit intermediate
// Uses the special structure of Goldilocks prime for fast reduction
inline ulong gl_mul(ulong a, ulong b) {
    // Compute a * b as 128-bit value using 32-bit word splitting
    ulong a_lo = a & 0xFFFFFFFF;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFF;
    ulong b_hi = b >> 32;

    // Four 64-bit products
    ulong p_ll = a_lo * b_lo;
    ulong p_lh = a_lo * b_hi;
    ulong p_hl = a_hi * b_lo;
    ulong p_hh = a_hi * b_hi;

    // Combine into 128-bit result
    // result = p_hh * 2^64 + (p_lh + p_hl) * 2^32 + p_ll
    ulong mid = p_lh + p_hl;
    ulong mid_carry = (mid < p_lh) ? 1UL : 0UL;

    ulong lo = p_ll + (mid << 32);
    ulong lo_carry = (lo < p_ll) ? 1UL : 0UL;

    ulong hi = p_hh + (mid >> 32) + (mid_carry << 32) + lo_carry;

    // Reduce 128-bit value modulo Goldilocks prime
    // For Goldilocks: x mod p = x_lo - x_hi * (2^32 - 1) mod p
    // Since 2^64 ≡ 2^32 - 1 (mod p)

    // Use reduce128 from goldilocks.metal
    return reduce128(hi, lo) % GL_PRIME;
}

// NTT Uniforms
struct NTTUniforms {
    uint n;           // Transform size (power of 2)
    uint log_n;       // log2(n)
    uint stage;       // Current butterfly stage
    uint direction;   // 0 = forward NTT, 1 = inverse NTT
    uint twiddle_stride; // Stride in twiddle table for this NTT size
};

struct NTTBatchUniforms {
    uint n;
    uint log_n;
    uint stage;
    uint direction;
    uint twiddle_stride;
    uint batches;
    uint batch_stride;
};

struct CosetUniforms {
    uint n;
    uint batches;
};

struct ExtendUniforms {
    uint in_n;
    uint out_n;
    uint batches;
};

struct ExtendCosetUniforms {
    uint in_n;
    uint out_n;
    uint batches;
};

// Extend input arrays into output arrays with zero padding.
kernel void extend_inputs_batch(
    device ulong * output[[buffer(0)]],
    device const ulong * input[[buffer(1)]],
    constant ExtendUniforms & uniforms[[buffer(2)]],
    uint gid[[thread_position_in_grid]]
) {
    uint out_n = uniforms.out_n;
    uint total = out_n * uniforms.batches;
    if (gid >= total) return;
    uint batch = gid / out_n;
    uint idx = gid - batch * out_n;
    if (idx < uniforms.in_n) {
        output[gid] = input[batch * uniforms.in_n + idx];
    } else {
        output[gid] = 0;
    }
}

// Extend with zero padding and apply coset multiplication.
kernel void extend_inputs_coset_batch(
    device ulong * output[[buffer(0)]],
    device const ulong * input[[buffer(1)]],
    device const ulong * coset_pows[[buffer(2)]],
    constant ExtendCosetUniforms & uniforms[[buffer(3)]],
    uint gid[[thread_position_in_grid]]
) {
    uint out_n = uniforms.out_n;
    uint total = out_n * uniforms.batches;
    if (gid >= total) return;
    uint batch = gid / out_n;
    uint idx = gid - batch * out_n;
    ulong v = 0;
    if (idx < uniforms.in_n) {
        v = input[batch * uniforms.in_n + idx];
    }
    output[gid] = gl_mul(v, coset_pows[idx]);
}

// Multiply each element by coset powers.
kernel void batch_vector_mult(
    device ulong * data[[buffer(0)]],
    device const ulong * coset_pows[[buffer(1)]],
    constant CosetUniforms & uniforms[[buffer(2)]],
    uint gid[[thread_position_in_grid]]
) {
    uint n = uniforms.n;
    uint batches = uniforms.batches;
    uint total = n * batches;
    if (gid >= total) return;
    uint idx = gid % n;
    ulong w = coset_pows[idx];
    data[gid] = gl_mul(data[gid], w);
}

// Bit reversal kernel
// Reorders elements according to bit-reversed indices
kernel void ntt_bit_reverse(
    device ulong * data[[buffer(0)]],
    constant NTTUniforms & uniforms[[buffer(1)]],
    uint gid[[thread_position_in_grid]]
) {
    if (gid >= uniforms.n) return;

    // Compute bit-reversed index
    uint rev = 0;
    uint x = gid;
    for (uint i = 0; i < uniforms.log_n; i++) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }

    // Only swap if gid < rev (to avoid double-swapping)
    if (gid < rev) {
        ulong temp = data[gid];
        data[gid] = data[rev];
        data[rev] = temp;
    }
}

// Batched bit-reversal kernel.
kernel void ntt_bit_reverse_batch(
    device ulong * data[[buffer(0)]],
    constant NTTBatchUniforms & uniforms[[buffer(1)]],
    uint gid[[thread_position_in_grid]]
) {
    uint n = uniforms.n;
    uint total = n * uniforms.batches;
    if (gid >= total) return;

    uint batch = gid / n;
    uint idx = gid - batch * n;
    uint rev = 0;
    uint x = idx;
    for (uint i = 0; i < uniforms.log_n; i++) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }
    if (idx < rev) {
        uint base = batch * n;
        ulong temp = data[base + idx];
        data[base + idx] = data[base + rev];
        data[base + rev] = temp;
    }
}

// Batched bit-reversal with coset multiplication (forward path).
kernel void ntt_bit_reverse_coset_batch(
    device ulong * data[[buffer(0)]],
    device const ulong * coset_pows[[buffer(1)]],
    constant NTTBatchUniforms & uniforms[[buffer(2)]],
    uint gid[[thread_position_in_grid]]
) {
    uint n = uniforms.n;
    uint total = n * uniforms.batches;
    if (gid >= total) return;

    uint batch = gid / n;
    uint idx = gid - batch * n;
    uint rev = 0;
    uint x = idx;
    for (uint i = 0; i < uniforms.log_n; i++) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }
    uint base = batch * n;
    if (idx < rev) {
        ulong a = data[base + idx];
        ulong b = data[base + rev];
        data[base + idx] = gl_mul(b, coset_pows[rev]);
        data[base + rev] = gl_mul(a, coset_pows[idx]);
    } else if (idx == rev) {
        data[base + idx] = gl_mul(data[base + idx], coset_pows[idx]);
    }
}

// Butterfly kernel for NTT
// Performs one stage of Cooley-Tukey butterfly operations
kernel void ntt_butterfly(
    device ulong * data[[buffer(0)]],
    constant ulong * twiddles[[buffer(1)]],
    constant NTTUniforms & uniforms[[buffer(2)]],
    uint gid[[thread_position_in_grid]]
) {
    uint num_butterflies = uniforms.n / 2;
    if (gid >= num_butterflies) return;

    uint stage = uniforms.stage;
    uint stride = 1u << stage;
    uint m = 1u << (stage + 1);

    // Calculate butterfly pair indices
    uint k = gid % stride;
    uint j = 2 * stride * (gid / stride) + k;
    uint i = j + stride;

    // Twiddle factor index: k * (n / m) gives the power of omega needed
    // Multiply by twiddle_stride to get index in the precomputed table
    uint twiddle_idx = k * (uniforms.n / m) * uniforms.twiddle_stride;
    ulong w = twiddles[twiddle_idx];

    // Butterfly operation
    ulong u = data[j];
    ulong v = gl_mul(data[i], w);

    data[j] = gl_add(u, v);
    data[i] = gl_sub(u, v);
}

// Batched butterfly kernel
kernel void ntt_butterfly_batch(
    device ulong * data[[buffer(0)]],
    constant ulong * twiddles[[buffer(1)]],
    constant NTTBatchUniforms & uniforms[[buffer(2)]],
    uint gid[[thread_position_in_grid]]
) {
    uint n = uniforms.n;
    uint butterflies_per_batch = n / 2;
    uint total = butterflies_per_batch * uniforms.batches;
    if (gid >= total) return;

    uint batch = gid / butterflies_per_batch;
    uint local = gid - batch * butterflies_per_batch;

    uint stage = uniforms.stage;
    uint stride = 1u << stage;
    uint m = 1u << (stage + 1);

    uint k = local % stride;
    uint j = 2 * stride * (local / stride) + k;
    uint i = j + stride;
    uint base = batch * n;

    uint twiddle_idx = k * (uniforms.n / m) * uniforms.twiddle_stride;
    ulong w = twiddles[twiddle_idx];

    ulong u = data[base + j];
    ulong v = gl_mul(data[base + i], w);

    data[base + j] = gl_add(u, v);
    data[base + i] = gl_sub(u, v);
}

// Shared-memory NTT for early stages within a block.
// Processes stages [stage_offset, stage_offset + stage_count)
kernel void ntt_butterfly_shared(
    device ulong * data[[buffer(0)]],
    constant ulong * twiddles[[buffer(1)]],
    constant NTTUniforms & uniforms[[buffer(2)]],
    constant uint & stage_offset[[buffer(3)]],
    constant uint & stage_count[[buffer(4)]],
    uint tid[[thread_position_in_threadgroup]],
    uint tgid[[threadgroup_position_in_grid]]
) {
    uint block_size = 1u << stage_count;
    uint block_start = tgid * block_size;

    threadgroup ulong shared_data[1024];
    if (tid < block_size) {
        shared_data[tid] = data[block_start + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < stage_count; s++) {
        uint stage = stage_offset + s;
        uint stride = 1u << stage;
        uint m = 1u << (stage + 1);

        uint num_butterflies = block_size / 2;
        if (tid < num_butterflies) {
            uint k = tid % stride;
            uint j = 2 * stride * (tid / stride) + k;
            uint i = j + stride;

            uint twiddle_idx = k * (uniforms.n / m) * uniforms.twiddle_stride;
            ulong w = twiddles[twiddle_idx];
            ulong u = shared_data[j];
            ulong v = gl_mul(shared_data[i], w);
            shared_data[j] = gl_add(u, v);
            shared_data[i] = gl_sub(u, v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < block_size) {
        data[block_start + tid] = shared_data[tid];
    }
}

// Shared-memory batched NTT for early stages.
kernel void ntt_butterfly_shared_batch(
    device ulong * data[[buffer(0)]],
    constant ulong * twiddles[[buffer(1)]],
    constant NTTBatchUniforms & uniforms[[buffer(2)]],
    constant uint & stage_offset[[buffer(3)]],
    constant uint & stage_count[[buffer(4)]],
    uint tid[[thread_position_in_threadgroup]],
    uint tgid[[threadgroup_position_in_grid]]
) {
    uint block_size = 1u << stage_count;
    uint blocks_per_batch = uniforms.n / block_size;
    uint batch = tgid / blocks_per_batch;
    uint block = tgid - batch * blocks_per_batch;
    uint block_start = batch * uniforms.n + block * block_size;

    threadgroup ulong shared_data[1024];
    if (tid < block_size) {
        shared_data[tid] = data[block_start + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < stage_count; s++) {
        uint stage = stage_offset + s;
        uint stride = 1u << stage;
        uint m = 1u << (stage + 1);

        uint num_butterflies = block_size / 2;
        if (tid < num_butterflies) {
            uint k = tid % stride;
            uint j = 2 * stride * (tid / stride) + k;
            uint i = j + stride;

            uint twiddle_idx = k * (uniforms.n / m) * uniforms.twiddle_stride;
            ulong w = twiddles[twiddle_idx];
            ulong u = shared_data[j];
            ulong v = gl_mul(shared_data[i], w);
            shared_data[j] = gl_add(u, v);
            shared_data[i] = gl_sub(u, v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < block_size) {
        data[block_start + tid] = shared_data[tid];
    }
}

// SIMD-group NTT for early stages (block_size must be 32).
kernel void ntt_butterfly_simdgroup_batch(
    device ulong * data[[buffer(0)]],
    constant ulong * twiddles[[buffer(1)]],
    constant NTTBatchUniforms & uniforms[[buffer(2)]],
    constant uint & stage_offset[[buffer(3)]],
    constant uint & stage_count[[buffer(4)]],
    uint tid[[thread_index_in_threadgroup]],
    uint tgid[[threadgroup_position_in_grid]]
) {
    const uint block_size = 1u << stage_count;
    if (block_size != 32) {
        return;
    }

    uint blocks_per_batch = uniforms.n / block_size;
    uint batch = tgid / blocks_per_batch;
    uint block = tgid - batch * blocks_per_batch;
    uint block_start = batch * uniforms.n + block * block_size;

    if (tid >= block_size) {
        return;
    }

    ulong val = data[block_start + tid];

    for (uint s = 0; s < stage_count; s++) {
        uint stage = stage_offset + s;
        uint stride = 1u << stage;
        uint m = 1u << (stage + 1);

        uint k = tid & (stride - 1);
        uint twiddle_idx = k * (uniforms.n / m) * uniforms.twiddle_stride;
        ulong w = twiddles[twiddle_idx];

        uint partner = tid ^ stride;
        ulong other = simd_shuffle(val, partner);
        ulong t = gl_mul(other, w);
        if ((tid & stride) == 0) {
            val = gl_add(val, t);
        } else {
            val = gl_sub(val, t);
        }
        simdgroup_barrier(mem_flags::mem_none);
    }

    data[block_start + tid] = val;
}

// INTT (Inverse NTT) butterfly kernel
// Same as NTT but uses inverse twiddle factors
kernel void intt_butterfly(
    device ulong * data[[buffer(0)]],
    constant ulong * inv_twiddles[[buffer(1)]],
    constant NTTUniforms & uniforms[[buffer(2)]],
    uint gid[[thread_position_in_grid]]
) {
    uint num_butterflies = uniforms.n / 2;
    if (gid >= num_butterflies) return;

    uint stage = uniforms.stage;
    uint stride = 1u << stage;
    uint m = 1u << (stage + 1);

    // Calculate butterfly pair indices
    uint k = gid % stride;
    uint j = 2 * stride * (gid / stride) + k;
    uint i = j + stride;

    // Inverse twiddle factor index with stride
    uint twiddle_idx = k * (uniforms.n / m) * uniforms.twiddle_stride;
    ulong w = inv_twiddles[twiddle_idx];

    // Butterfly operation (same as forward)
    ulong u = data[j];
    ulong v = gl_mul(data[i], w);

    data[j] = gl_add(u, v);
    data[i] = gl_sub(u, v);
}

// Batched inverse butterfly kernel.
kernel void intt_butterfly_batch(
    device ulong * data[[buffer(0)]],
    constant ulong * inv_twiddles[[buffer(1)]],
    constant NTTBatchUniforms & uniforms[[buffer(2)]],
    uint gid[[thread_position_in_grid]]
) {
    uint n = uniforms.n;
    uint butterflies_per_batch = n / 2;
    uint total = butterflies_per_batch * uniforms.batches;
    if (gid >= total) return;

    uint batch = gid / butterflies_per_batch;
    uint local = gid - batch * butterflies_per_batch;

    uint stage = uniforms.stage;
    uint stride = 1u << stage;
    uint m = 1u << (stage + 1);

    uint k = local % stride;
    uint j = 2 * stride * (local / stride) + k;
    uint i = j + stride;
    uint base = batch * n;

    uint twiddle_idx = k * (uniforms.n / m) * uniforms.twiddle_stride;
    ulong w = inv_twiddles[twiddle_idx];

    ulong u = data[base + j];
    ulong v = gl_mul(data[base + i], w);

    data[base + j] = gl_add(u, v);
    data[base + i] = gl_sub(u, v);
}

// SIMD-group INTT for early stages (block_size must be 32).
kernel void intt_butterfly_simdgroup_batch(
    device ulong * data[[buffer(0)]],
    constant ulong * inv_twiddles[[buffer(1)]],
    constant NTTBatchUniforms & uniforms[[buffer(2)]],
    constant uint & stage_offset[[buffer(3)]],
    constant uint & stage_count[[buffer(4)]],
    uint tid[[thread_index_in_threadgroup]],
    uint tgid[[threadgroup_position_in_grid]]
) {
    const uint block_size = 1u << stage_count;
    if (block_size != 32) {
        return;
    }

    uint blocks_per_batch = uniforms.n / block_size;
    uint batch = tgid / blocks_per_batch;
    uint block = tgid - batch * blocks_per_batch;
    uint block_start = batch * uniforms.n + block * block_size;

    if (tid >= block_size) {
        return;
    }

    ulong val = data[block_start + tid];

    for (uint s = 0; s < stage_count; s++) {
        uint stage = stage_offset + s;
        uint stride = 1u << stage;
        uint m = 1u << (stage + 1);

        uint k = tid & (stride - 1);
        uint twiddle_idx = k * (uniforms.n / m) * uniforms.twiddle_stride;
        ulong w = inv_twiddles[twiddle_idx];

        uint partner = tid ^ stride;
        ulong other = simd_shuffle(val, partner);
        ulong t = gl_mul(other, w);
        if ((tid & stride) == 0) {
            val = gl_add(val, t);
        } else {
            val = gl_sub(val, t);
        }
        simdgroup_barrier(mem_flags::mem_none);
    }

    data[block_start + tid] = val;
}

// Scale kernel for INTT normalization
// Multiplies all elements by n^(-1) mod p
kernel void ntt_scale(
    device ulong * data[[buffer(0)]],
    constant ulong & n_inv[[buffer(1)]],
    constant uint & n[[buffer(2)]],
    uint gid[[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = gl_mul(data[gid], n_inv);
}

// Batched scale kernel for INTT normalization.
kernel void ntt_scale_batch(
    device ulong * data[[buffer(0)]],
    constant ulong & n_inv[[buffer(1)]],
    constant uint & n[[buffer(2)]],
    constant uint & batches[[buffer(3)]],
    uint gid[[thread_position_in_grid]]
) {
    uint total = n * batches;
    if (gid >= total) return;
    data[gid] = gl_mul(data[gid], n_inv);
}

// Batched scale + coset inverse (for inverse NTT on coset).
kernel void ntt_scale_coset_batch(
    device ulong * data[[buffer(0)]],
    constant ulong & n_inv[[buffer(1)]],
    device const ulong * coset_inv_pows[[buffer(2)]],
    constant uint & n[[buffer(3)]],
    constant uint & batches[[buffer(4)]],
    uint gid[[thread_position_in_grid]]
) {
    uint total = n * batches;
    if (gid >= total) return;
    uint idx = gid % n;
    ulong v = gl_mul(data[gid], n_inv);
    data[gid] = gl_mul(v, coset_inv_pows[idx]);
}

// Coalesced NTT kernel - processes multiple stages with better memory access
// Each thread handles multiple butterfly operations
kernel void ntt_butterfly_coalesced(
    device ulong * data[[buffer(0)]],
    constant ulong * twiddles[[buffer(1)]],
    constant NTTUniforms & uniforms[[buffer(2)]],
    uint gid[[thread_position_in_grid]],
    uint lid[[thread_position_in_threadgroup]],
    uint tgid[[threadgroup_position_in_grid]]
) {
    uint num_butterflies = uniforms.n / 2;
    if (gid >= num_butterflies) return;

    uint stage = uniforms.stage;
    uint stride = 1u << stage;
    uint m = 1u << (stage + 1);

    // Thread assignment for coalesced access
    uint pair_idx = gid;
    uint block_idx = pair_idx / stride;
    uint k = pair_idx % stride;

    uint j = block_idx * m + k;
    uint i = j + stride;

    // Twiddle factor with stride
    uint twiddle_idx = k * (uniforms.n / m) * uniforms.twiddle_stride;
    ulong w = twiddles[twiddle_idx];

    // Load, compute, store
    ulong u = data[j];
    ulong v = gl_mul(data[i], w);

    data[j] = gl_add(u, v);
    data[i] = gl_sub(u, v);
}

// Batch NTT - process multiple small NTTs in parallel
// Useful when doing many small FFTs (like in polynomial multiplication)
kernel void ntt_batch_butterfly(
    device ulong * data[[buffer(0)]],
    constant ulong * twiddles[[buffer(1)]],
    constant NTTUniforms & uniforms[[buffer(2)]],
    constant uint & batch_count[[buffer(3)]],
    uint gid[[thread_position_in_grid]]
) {
    uint batch_butterflies = uniforms.n / 2;
    uint total_butterflies = batch_butterflies * batch_count;

    if (gid >= total_butterflies) return;

    uint batch_idx = gid / batch_butterflies;
    uint local_gid = gid % batch_butterflies;

    uint stage = uniforms.stage;
    uint stride = 1u << stage;
    uint m = 1u << (stage + 1);

    uint k = local_gid % stride;
    uint j = 2 * stride * (local_gid / stride) + k;
    uint i = j + stride;

    // Offset for this batch
    uint batch_offset = batch_idx * uniforms.n;
    j += batch_offset;
    i += batch_offset;

    uint twiddle_idx = k * (uniforms.n / m) * uniforms.twiddle_stride;
    ulong w = twiddles[twiddle_idx];

    ulong u = data[j];
    ulong v = gl_mul(data[i], w);

    data[j] = gl_add(u, v);
    data[i] = gl_sub(u, v);
}
