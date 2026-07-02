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
Complete. Committed as `7496a79`.

### Verification
- All changes follow the exact same pattern as upstream PyTorch 2.9 (commit `3d4094a`)
- `ScaledBlas.cpp` and `GroupedBlas.cpp` arch lists correctly left unchanged (FP8/group GEMM require CDNA-specific features)
- No new tests needed — existing `test_preferred_blas_library_settings` in `test/test_cuda.py` validates BLAS backend selection
- Build verification requires a ROCm environment (not available on this Windows machine)

### What Was Learned
- The hipBLASLt preferred/supported arch lists are in `CUDAHooks.cpp` and are the single source of truth for hipBLASLt arch gating
- CK GEMM arch gating is split between `Context.cpp` (`ckGemmSupported()`) and the dispatch files (`ck_gemm_bfloat16.hip`)
- CK GEMM uses WMMA on RDNA3/RDNA4 and XDL on CDNA — the arch lists in `ckGemmSupported()` should include both
- FP8 scaled operations (`ScaledBlas.cpp`, `GroupedBlas.cpp`) are correctly limited to FP8-capable architectures
- The `gfx1103` arch was in supported archs but missing from WMMA dispatch — a simple oversight

### Next Iteration Should Tackle
- Improve the CK GEMM dispatch heuristic in `ck_gemm_bfloat16.hip` — currently only has a single tile config for all shape sizes, with a TODO to "add more configurations. Optimize."
- Or: Expand CK GEMM to support float16 and float32 types (currently only BFloat16 has CK dispatch)
- Or: Add more BGEMM kernel configurations for common LLM shapes (the dispatch in `ck_bgemm_bfloat16.hip` has limited shape coverage)
