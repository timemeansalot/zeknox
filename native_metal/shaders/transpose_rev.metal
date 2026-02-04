// Transpose + bit-reverse kernel for Goldilocks data (u64)

#include <metal_stdlib>
using namespace metal;

struct TransposeUniforms {
    uint n;
    uint log_n;
    uint batches;
};

inline uint reverse_bits(uint v, uint log_n) {
    uint r = 0;
    for (uint i = 0; i < log_n; i++) {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    return r;
}

// Tiled transpose with bit-reversal, 32x32 tiles
kernel void transpose_rev(
    device const ulong *in_data [[buffer(0)]],
    device ulong *out_data [[buffer(1)]],
    constant TransposeUniforms &uniforms [[buffer(2)]],
    ushort2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    constexpr uint TILE = 32;
    threadgroup ulong tile[TILE][TILE + 1];

    uint x = tgid.x * TILE + tid.x;
    uint y = tgid.y * TILE + tid.y;

    if (x < uniforms.n && y < uniforms.batches) {
        uint in_idx = y * uniforms.n + x;
        tile[tid.y][tid.x] = in_data[in_idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Transpose tile: swap x/y
    uint tx = tgid.y * TILE + tid.x;
    uint ty = tgid.x * TILE + tid.y;
    if (tx < uniforms.batches && ty < uniforms.n) {
        uint rev = reverse_bits(ty, uniforms.log_n);
        uint out_idx = rev * uniforms.batches + tx;
        out_data[out_idx] = tile[tid.x][tid.y];
    }
}
