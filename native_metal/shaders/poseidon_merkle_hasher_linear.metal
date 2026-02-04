// Linear indexing Merkle tree implementation
// Based on zeknox CUDA implementation pattern for better memory coalescing
//
// Memory Layout (for tree with 4 leaves):
// Linear Array:
// Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]
// Value:  H0123 H01   H23   L0    L1    L2    L3
//         (root)(lvl1)(lvl1)(leaf)(leaf)(leaf)(leaf)
//
// For subtree at index subtree_idx with subtree_digests_len total digests:
// - Leaves are stored in the second half: offset = subtree_digests_len - subtree_leaves_len
// - Internal nodes are stored in the first half, level by level from root to leaves-1

#include <metal_stdlib>
#include "goldilocks.metal"
#include "poseidon_goldilocks.metal"

using namespace metal;
using namespace GoldilocksField;

// Uniforms for linear layout
struct LinearUniforms {
    uint level;               // Current tree level being processed (0 = leaf level)
    uint subtree_digests_len; // Number of digests per subtree (2 * leaves - 1)
    uint subtree_leaves_len;  // Number of leaves per subtree
    uint leaf_size;           // Elements per leaf
    uint leaf_count;          // Total number of leaves
    uint subtree_count;       // Number of subtrees (2^cap_height)
    uint grid_width;          // Total threads in X dimension
};

// Uniforms for fused transpose+leaf hashing
struct TransposeLeafUniforms {
    uint n;                   // Number of points per batch (2^log_n)
    uint log_n;               // log2(n)
    uint batches;             // Elements per leaf
    uint subtree_digests_len; // Number of digests per subtree (2 * leaves - 1)
    uint subtree_leaves_len;  // Number of leaves per subtree
    uint leaf_size;           // Elements per leaf
    uint leaf_count;          // Total number of leaves
    uint subtree_count;       // Number of subtrees (2^cap_height)
    uint grid_width;          // Total threads in X dimension
};

inline uint reverse_bits_u32(uint v, uint log_n) {
    uint r = 0;
    for (uint i = 0; i < log_n; i++) {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    return r;
}

// Compute digest index for a leaf in linear layout
// Leaves are stored in the second half of each subtree's digest array
inline uint compute_linear_leaf_index(
    uint subtree_digests_len,
    uint subtree_leaves_len,
    uint subtree_idx,
    uint in_subtree_idx
) {
    // Leaves start at offset (subtree_digests_len - subtree_leaves_len) within each subtree
    return subtree_idx * subtree_digests_len
         + (subtree_digests_len - subtree_leaves_len)
         + in_subtree_idx;
}

// Compute digest index for an internal node in linear layout
// Internal nodes are stored by level, from last level (above leaves) toward root
inline uint compute_linear_internal_index(
    uint subtree_digests_len,
    uint subtree_leaves_len,
    uint subtree_idx,
    uint level,          // Level counting from leaves: level 1 = parents of leaves
    uint index_in_level  // Index within this level
) {
    // For level i (counting from leaves), there are (subtree_leaves_len >> i) nodes
    // Level 1 nodes start at offset: subtree_digests_len - subtree_leaves_len - (subtree_leaves_len >> 1)
    // Level 2 nodes start at: above - (subtree_leaves_len >> 2)
    // etc.

    // Calculate offset from the start of subtree
    // Leaves are at: subtree_digests_len - subtree_leaves_len
    // Level 1 is at: subtree_digests_len - subtree_leaves_len - (subtree_leaves_len >> 1)
    // Level k is at: subtree_digests_len - subtree_leaves_len - sum(i=1 to k of (subtree_leaves_len >> i))
    //             = subtree_digests_len - subtree_leaves_len - (subtree_leaves_len - (subtree_leaves_len >> k))
    //             = subtree_digests_len - 2*subtree_leaves_len + (subtree_leaves_len >> level)

    uint level_start = subtree_digests_len - 2 * subtree_leaves_len + (subtree_leaves_len >> level);
    return subtree_idx * subtree_digests_len + level_start + index_in_level;
}

// Kernel to hash leaves with linear layout
kernel void poseidon_hash_leaves_linear(
    constant Fp * leaf_inputs[[buffer(0)]],
    device ulong * output[[buffer(1)]],
    constant LinearUniforms & uniforms[[buffer(2)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];

    // Bounds check: don't process threads beyond the actual leaf count
    if (thread_id >= uniforms.leaf_count) {
        return;
    }

    // Calculate which subtree this leaf belongs to and its index within the subtree
    uint subtree_idx = thread_id / uniforms.subtree_leaves_len;
    uint in_subtree_idx = thread_id % uniforms.subtree_leaves_len;

    // Calculate linear digest index
    uint digest_idx = compute_linear_leaf_index(
        uniforms.subtree_digests_len,
        uniforms.subtree_leaves_len,
        subtree_idx,
        in_subtree_idx
    );

    uint input_offset = thread_id * uniforms.leaf_size;
    uint output_offset = digest_idx * 4;

    // hash_or_noop logic: if leaf_size <= 4, don't hash, just copy
    // This matches Plonky2's CPU behavior where small inputs fit directly in a hash
    if (uniforms.leaf_size <= 4) {
        // No hashing needed - leaf fits in a hash output
        // Copy the leaf elements directly to output (pad with zeros if needed)
        // No hashing needed - leaf fits in a hash output
        // Copy the leaf elements directly to output (pad with zeros if needed)
        for (uint i = 0; i < 4; i++) {
            if (i < uniforms.leaf_size) {
                Fp val = leaf_inputs[input_offset + i];
                output[output_offset + i] = static_cast<ulong>(val);
            } else {
                output[output_offset + i] = 0;
            }
        }
    } else {
        // Leaf is larger than a hash - need to hash with Poseidon
        Fp p2_state[12];

        // Initialize all positions to 0 to match CPU sponge initialization
        for (uint i = 0; i < 12; i++) {
            p2_state[i] = 0;
        }

        uint offset = input_offset;
        uint num_full_rounds = uniforms.leaf_size / 8;

        for (uint i = 0; i < num_full_rounds; i++) {
            p2_state[0] = leaf_inputs[offset];
            p2_state[1] = leaf_inputs[offset + 1];
            p2_state[2] = leaf_inputs[offset + 2];
            p2_state[3] = leaf_inputs[offset + 3];

            p2_state[4] = leaf_inputs[offset + 4];
            p2_state[5] = leaf_inputs[offset + 5];
            p2_state[6] = leaf_inputs[offset + 6];
            p2_state[7] = leaf_inputs[offset + 7];
            poseidon_permute(p2_state);
            offset += 8;
        }

        uint remaining = uniforms.leaf_size - num_full_rounds * 8;
        if (remaining != 0) {
            for (uint i = 0; i < remaining; i++) {
                p2_state[i] = leaf_inputs[offset + i];
            }
            poseidon_permute(p2_state);
        }

        // Write the hashed result
        output[output_offset] = static_cast<ulong>(p2_state[0]);
        output[output_offset + 1] = static_cast<ulong>(p2_state[1]);
        output[output_offset + 2] = static_cast<ulong>(p2_state[2]);
        output[output_offset + 3] = static_cast<ulong>(p2_state[3]);
    }
}

// Fused transpose_rev + leaf hashing: reads batch-major input and writes linear leaf digests.
kernel void poseidon_hash_leaves_linear_transpose_rev(
    device const ulong * in_data[[buffer(0)]],
    device ulong * output[[buffer(1)]],
    constant TransposeLeafUniforms & uniforms[[buffer(2)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];

    if (thread_id >= uniforms.leaf_count) {
        return;
    }

    uint subtree_idx = thread_id / uniforms.subtree_leaves_len;
    uint in_subtree_idx = thread_id % uniforms.subtree_leaves_len;

    uint digest_idx = compute_linear_leaf_index(
        uniforms.subtree_digests_len,
        uniforms.subtree_leaves_len,
        subtree_idx,
        in_subtree_idx
    );

    uint output_offset = digest_idx * 4;
    uint rev = reverse_bits_u32(thread_id, uniforms.log_n);

    if (uniforms.leaf_size <= 4) {
        for (uint i = 0; i < 4; i++) {
            if (i < uniforms.leaf_size) {
                uint idx = i * uniforms.n + rev;
                output[output_offset + i] = in_data[idx];
            } else {
                output[output_offset + i] = 0;
            }
        }
    } else {
        Fp p2_state[12];
        for (uint i = 0; i < 12; i++) {
            p2_state[i] = 0;
        }

        uint num_full_rounds = uniforms.leaf_size / 8;
        for (uint i = 0; i < num_full_rounds; i++) {
            uint base = i * 8;
            p2_state[0] = Fp(in_data[(base + 0) * uniforms.n + rev]);
            p2_state[1] = Fp(in_data[(base + 1) * uniforms.n + rev]);
            p2_state[2] = Fp(in_data[(base + 2) * uniforms.n + rev]);
            p2_state[3] = Fp(in_data[(base + 3) * uniforms.n + rev]);
            p2_state[4] = Fp(in_data[(base + 4) * uniforms.n + rev]);
            p2_state[5] = Fp(in_data[(base + 5) * uniforms.n + rev]);
            p2_state[6] = Fp(in_data[(base + 6) * uniforms.n + rev]);
            p2_state[7] = Fp(in_data[(base + 7) * uniforms.n + rev]);
            poseidon_permute(p2_state);
        }

        uint remaining = uniforms.leaf_size - num_full_rounds * 8;
        if (remaining != 0) {
            for (uint i = 0; i < remaining; i++) {
                p2_state[i] = Fp(in_data[(num_full_rounds * 8 + i) * uniforms.n + rev]);
            }
            poseidon_permute(p2_state);
        }

        output[output_offset] = static_cast<ulong>(p2_state[0]);
        output[output_offset + 1] = static_cast<ulong>(p2_state[1]);
        output[output_offset + 2] = static_cast<ulong>(p2_state[2]);
        output[output_offset + 3] = static_cast<ulong>(p2_state[3]);
    }
}

// Leaf hashing with 2 threads per leaf. Uses threadgroup memory to stage 8 inputs.
// This kernel assumes a 1D grid and even threads_per_group.
// Kernel to hash internal tree levels with linear layout
// This is called for each level from leaves-1 up to level 1 (just below cap)
kernel void poseidon_hash_tree_level_linear(
    device ulong * output[[buffer(0)]],
    constant LinearUniforms & uniforms[[buffer(1)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];

    // Number of nodes at this level across all subtrees
    uint nodes_per_subtree = uniforms.subtree_leaves_len >> uniforms.level;
    uint total_nodes = nodes_per_subtree * uniforms.subtree_count;

    if (thread_id >= total_nodes) {
        return;
    }

    // Determine which subtree and which node within subtree
    uint subtree_idx = thread_id / nodes_per_subtree;
    uint index_in_level = thread_id % nodes_per_subtree;

    // Calculate parent index (where we write)
    uint parent_idx = compute_linear_internal_index(
        uniforms.subtree_digests_len,
        uniforms.subtree_leaves_len,
        subtree_idx,
        uniforms.level,
        index_in_level
    );

    // Calculate children indices (where we read from)
    // Children are at level (uniforms.level - 1)
    uint child_level = uniforms.level - 1;
    uint left_child_in_level = index_in_level * 2;
    uint right_child_in_level = index_in_level * 2 + 1;

    uint left_child_idx, right_child_idx;

    if (child_level == 0) {
        // Children are leaves
        left_child_idx = compute_linear_leaf_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            left_child_in_level
        );
        right_child_idx = compute_linear_leaf_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            right_child_in_level
        );
    } else {
        // Children are internal nodes
        left_child_idx = compute_linear_internal_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            child_level,
            left_child_in_level
        );
        right_child_idx = compute_linear_internal_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            child_level,
            right_child_in_level
        );
    }

    // Read children (8 elements total: 4 for left, 4 for right)
    Fp p2_state[12];
    uint left_offset = left_child_idx * 4;
    uint right_offset = right_child_idx * 4;

    p2_state[0] = output[left_offset];
    p2_state[1] = output[left_offset + 1];
    p2_state[2] = output[left_offset + 2];
    p2_state[3] = output[left_offset + 3];

    p2_state[4] = output[right_offset];
    p2_state[5] = output[right_offset + 1];
    p2_state[6] = output[right_offset + 2];
    p2_state[7] = output[right_offset + 3];

    // Zero out remaining state
    p2_state[8] = 0;
    p2_state[9] = 0;
    p2_state[10] = 0;
    p2_state[11] = 0;

    // Hash
    poseidon_permute(p2_state);

    // Write parent (4 elements)
    uint parent_offset = parent_idx * 4;
    output[parent_offset] = static_cast<ulong>(p2_state[0]);
    output[parent_offset + 1] = static_cast<ulong>(p2_state[1]);
    output[parent_offset + 2] = static_cast<ulong>(p2_state[2]);
    output[parent_offset + 3] = static_cast<ulong>(p2_state[3]);
}

// Kernel to hash two internal levels at once: computes level (L-1) and level L.
// Requires uniforms.level >= 2.
kernel void poseidon_hash_tree_level_linear_2(
    device ulong * output[[buffer(0)]],
    constant LinearUniforms & uniforms[[buffer(1)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];

    uint level = uniforms.level;
    if (level < 2) {
        return;
    }

    uint nodes_per_subtree = uniforms.subtree_leaves_len >> level;
    uint total_nodes = nodes_per_subtree * uniforms.subtree_count;
    if (thread_id >= total_nodes) {
        return;
    }

    uint subtree_idx = thread_id / nodes_per_subtree;
    uint index_in_level = thread_id % nodes_per_subtree;

    uint parent_level = level - 1;
    uint child_level = level - 2;

    uint left_parent_idx = compute_linear_internal_index(
        uniforms.subtree_digests_len,
        uniforms.subtree_leaves_len,
        subtree_idx,
        parent_level,
        index_in_level * 2
    );
    uint right_parent_idx = compute_linear_internal_index(
        uniforms.subtree_digests_len,
        uniforms.subtree_leaves_len,
        subtree_idx,
        parent_level,
        index_in_level * 2 + 1
    );

    uint child0_idx, child1_idx, child2_idx, child3_idx;
    if (child_level == 0) {
        child0_idx = compute_linear_leaf_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            index_in_level * 4
        );
        child1_idx = compute_linear_leaf_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            index_in_level * 4 + 1
        );
        child2_idx = compute_linear_leaf_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            index_in_level * 4 + 2
        );
        child3_idx = compute_linear_leaf_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            index_in_level * 4 + 3
        );
    } else {
        child0_idx = compute_linear_internal_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            child_level,
            index_in_level * 4
        );
        child1_idx = compute_linear_internal_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            child_level,
            index_in_level * 4 + 1
        );
        child2_idx = compute_linear_internal_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            child_level,
            index_in_level * 4 + 2
        );
        child3_idx = compute_linear_internal_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            child_level,
            index_in_level * 4 + 3
        );
    }

    // Compute left parent
    Fp left_state[12];
    uint c0 = child0_idx * 4;
    uint c1 = child1_idx * 4;
    left_state[0] = output[c0];
    left_state[1] = output[c0 + 1];
    left_state[2] = output[c0 + 2];
    left_state[3] = output[c0 + 3];
    left_state[4] = output[c1];
    left_state[5] = output[c1 + 1];
    left_state[6] = output[c1 + 2];
    left_state[7] = output[c1 + 3];
    left_state[8] = 0;
    left_state[9] = 0;
    left_state[10] = 0;
    left_state[11] = 0;
    poseidon_permute(left_state);

    uint lp = left_parent_idx * 4;
    output[lp] = static_cast<ulong>(left_state[0]);
    output[lp + 1] = static_cast<ulong>(left_state[1]);
    output[lp + 2] = static_cast<ulong>(left_state[2]);
    output[lp + 3] = static_cast<ulong>(left_state[3]);

    // Compute right parent
    Fp right_state[12];
    uint c2 = child2_idx * 4;
    uint c3 = child3_idx * 4;
    right_state[0] = output[c2];
    right_state[1] = output[c2 + 1];
    right_state[2] = output[c2 + 2];
    right_state[3] = output[c2 + 3];
    right_state[4] = output[c3];
    right_state[5] = output[c3 + 1];
    right_state[6] = output[c3 + 2];
    right_state[7] = output[c3 + 3];
    right_state[8] = 0;
    right_state[9] = 0;
    right_state[10] = 0;
    right_state[11] = 0;
    poseidon_permute(right_state);

    uint rp = right_parent_idx * 4;
    output[rp] = static_cast<ulong>(right_state[0]);
    output[rp + 1] = static_cast<ulong>(right_state[1]);
    output[rp + 2] = static_cast<ulong>(right_state[2]);
    output[rp + 3] = static_cast<ulong>(right_state[3]);

    // Compute grandparent (level)
    Fp parent_state[12];
    parent_state[0] = left_state[0];
    parent_state[1] = left_state[1];
    parent_state[2] = left_state[2];
    parent_state[3] = left_state[3];
    parent_state[4] = right_state[0];
    parent_state[5] = right_state[1];
    parent_state[6] = right_state[2];
    parent_state[7] = right_state[3];
    parent_state[8] = 0;
    parent_state[9] = 0;
    parent_state[10] = 0;
    parent_state[11] = 0;
    poseidon_permute(parent_state);

    uint gp = compute_linear_internal_index(
        uniforms.subtree_digests_len,
        uniforms.subtree_leaves_len,
        subtree_idx,
        level,
        index_in_level
    ) * 4;
    output[gp] = static_cast<ulong>(parent_state[0]);
    output[gp + 1] = static_cast<ulong>(parent_state[1]);
    output[gp + 2] = static_cast<ulong>(parent_state[2]);
    output[gp + 3] = static_cast<ulong>(parent_state[3]);
}

// Kernel to compute cap hashes from subtree roots
kernel void poseidon_hash_caps_linear(
    device ulong * caps_output[[buffer(0)]],
    device ulong * digests[[buffer(1)]],
    constant LinearUniforms & uniforms[[buffer(2)]],
    uint gid[[thread_position_in_grid]]
) {
    if (gid >= uniforms.subtree_count) {
        return;
    }

    // Read the root of this subtree (index 0 within the subtree)
    // For a subtree with root at level tree_height - cap_height - 1,
    // we need to hash its two children which are at level tree_height - cap_height - 2

    // Actually, the cap hash is computed from the two top-level nodes of each subtree
    // The subtree root itself is stored at index 0 of each subtree's digest array
    // But for the cap, we hash from the two children of the cap level

    // In this layout, the top two nodes are stored at indices 0 and 1.
    uint subtree_base = gid * uniforms.subtree_digests_len;

    uint left_idx = subtree_base + 0;
    uint right_idx = subtree_base + 1;

    Fp p2_state[12];

    p2_state[0] = digests[left_idx * 4];
    p2_state[1] = digests[left_idx * 4 + 1];
    p2_state[2] = digests[left_idx * 4 + 2];
    p2_state[3] = digests[left_idx * 4 + 3];

    p2_state[4] = digests[right_idx * 4];
    p2_state[5] = digests[right_idx * 4 + 1];
    p2_state[6] = digests[right_idx * 4 + 2];
    p2_state[7] = digests[right_idx * 4 + 3];

    p2_state[8] = 0;
    p2_state[9] = 0;
    p2_state[10] = 0;
    p2_state[11] = 0;

    poseidon_permute(p2_state);

    // Write cap hash
    uint cap_offset = gid * 4;
    caps_output[cap_offset] = static_cast<ulong>(p2_state[0]);
    caps_output[cap_offset + 1] = static_cast<ulong>(p2_state[1]);
    caps_output[cap_offset + 2] = static_cast<ulong>(p2_state[2]);
    caps_output[cap_offset + 3] = static_cast<ulong>(p2_state[3]);
}
