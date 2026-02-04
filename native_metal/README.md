# zeknox Metal backend (WIP)

This folder hosts the Metal shaders and host-side backend for zeknox.

Current status:
- Shaders imported from `plonky2-metal-demo` (Poseidon + Merkle linear layout).
- Build script compiles a metallib and exposes it via `ZEKNOX_METAL_LIB`.
- Host C API is still stubbed in `zeknox_metal.cpp` and must be wired to Metal runtime.

Next steps:
- Implement Metal runtime (device/queue/buffer + pipeline setup).
- Wire `fill_digests_buf_linear_*` to the Metal kernels.
- Add test harness to validate hashes vs CPU/CUDA.
