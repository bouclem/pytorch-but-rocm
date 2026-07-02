# PyTorch But ROCm — Progress Log

## Iteration 1 — Expand hipBLASLt and CK GEMM Architecture Support

### Plan
Enable hipBLASLt and CK GEMM on a wider range of AMD GPU architectures (RDNA3/RDNA4 consumer cards) by updating architecture support lists to match and exceed upstream PyTorch 2.9 improvements.

### Changes
- **`aten/src/ATen/cuda/detail/CUDAHooks.cpp`**:
  - `getHipblasltPreferredArchs()`: Add `gfx1100`, `gfx1101`, `gfx1102` (RDNA3) at ROCm 6.3+, `gfx1150`, `gfx1151` (RDNA4) at ROCm 6.4.2+
  - `getHipblasltSupportedArchs()`: Add missing `gfx1102`, fix `gfx1150`/`gfx1151` version gating to 6.4.2+
- **`aten/src/ATen/Context.cpp`**:
  - `ckGemmSupported()`: Add RDNA3 (`gfx1100`, `gfx1101`, `gfx1102`, `gfx1103`) and RDNA4 (`gfx1150`, `gfx1151`, `gfx1200`, `gfx1201`) archs where WMMA CK kernels already exist
- **`aten/src/ATen/native/hip/ck_gemm_bfloat16.hip`**:
  - WMMA arch list: Add `gfx1103` (RDNA3)

### Why It Matters
This is a foundational change that unlocks hipBLASLt (better GEMM performance) and CK GEMM on consumer AMD GPUs. Without these arch lists, PyTorch falls back to slower rocBLAS on RDNA3/RDNA4 cards.

### Status
In progress.
