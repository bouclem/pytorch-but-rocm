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
- Build system supports ROCm on Windows (`USE_ROCM` enabled on `LINUX OR WIN32`); `USE_ROCM_CK_GEMM` and `USE_ROCM_CK_SDPA` are currently disabled on WIN32 — a potential future improvement

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

## Iteration 2 — Fix gfx1103 in Half WMMA, Clean Up Debug Code in CK GEMM Template

### Plan
Fix the same `gfx1103` missing from the Half (float16) WMMA arch list as was fixed for BFloat16 in iteration 1. Also remove debug `printf` and dead benchmark code from `ck_gemm_template.h`.

### Changes
- **`aten/src/ATen/native/hip/ck_gemm_half.hip`**:
  - WMMA arch list: Add `gfx1103` (RDNA3) — same fix as bfloat16 in iteration 1
- **`aten/src/ATen/native/hip/ck_gemm_template.h`**:
  - Remove debug `printf` from WMMA GEMM error path (was printing to stdout before `TORCH_CHECK`)
  - Remove dead `#if 0` / `#else` benchmark code block with `std::cout` debug output

### Why It Matters
- `gfx1103` (RX 7900 GRE / RX 7800 XT) was missing from Half WMMA dispatch, causing CK GEMM to fail on this arch with float16
- Debug `printf` in production code pollutes stdout and provides no useful info since `TORCH_CHECK` already gives a proper error message
- Dead benchmark code behind `#if 0` adds maintenance burden and confusion

### Status
Complete. Committed as `e6b2043`.

### What Was Learned
- The same `gfx1103` omission existed in both `ck_gemm_bfloat16.hip` and `ck_gemm_half.hip` — systematic issue
- The WMMA template in `ck_gemm_template.h` had debug code that should have been cleaned up before merging upstream
- The XDL `gemm_impl` path was already clean — only the WMMA `gemm_impl_wmma` had debug artifacts

### Next Iteration Should Tackle
- Improve CK GEMM dispatch heuristic — the `dispatch_half_gemm` XDL path is entirely behind `#if 0` (disabled), meaning only WMMA works for Half on RDNA3
- Or: Add shape-based tile selection to the WMMA dispatch (currently uses one config for all shapes)
- Or: Look into the `ck_gemm_float.hip` to see if it needs similar arch list updates

## Iteration 3 — Enable Disabled XDL Dispatch for Half (float16) CK GEMM on CDNA

### Plan
The `dispatch_half_gemm` XDL path in `ck_gemm_half.hip` was entirely disabled with `#if 0`, making CK GEMM silently broken for float16 on CDNA (gfx9) GPUs. Enable it to match BFloat16 which has this path active.

### Changes
- **`aten/src/ATen/native/hip/ck_gemm_half.hip`**:
  - Removed `#if 0` and `#endif` around the `dispatch_half_gemm` function body
  - This enables the XDL-based CK GEMM for float16 on CDNA architectures (gfx90a, gfx942, gfx950)

### Why It Matters
When `gemm_internal_ck<at::Half>` was called on a CDNA GPU, it dispatched to `dispatch_half_gemm` which had an empty body due to `#if 0`. This meant CK GEMM for float16 was silently a no-op on MI200/MI300. The BFloat16 equivalent `dispatch_bfloat16_gemm` has this path enabled, so Half should too.

### Status
Complete. Committed as `66e96ae`.

### What Was Learned
- The `#if 0` was likely a development workaround that was never re-enabled
- Always check for `#if 0` blocks in dispatch functions — they may be silently disabling important code paths
- The XDL and WMMA paths serve different architectures: XDL for CDNA (gfx9), WMMA for RDNA (gfx11+)

### Next Iteration Should Tackle
- Add shape-based tile selection to WMMA dispatch (currently uses one config for all shapes)
- Or: Improve BGEMM kernel dispatch heuristic with more shape coverage
- Or: Look at the inductor CK backend code generation for improvements

## Iteration 4 — Enable CK GEMM on Windows

### Plan
`USE_ROCM_CK_GEMM` was disabled on Windows with `NOT WIN32`. AMD engineers confirmed CK GEMM works on Windows. Enable it to unlock all CK GEMM optimizations for Windows ROCm users.

### Changes
- **`CMakeLists.txt`**:
  - Changed `USE_ROCM_CK_GEMM` from `ON "USE_ROCM;NOT WIN32"` to `ON "USE_ROCM"` — removes the Windows restriction
  - `USE_ROCM_CK_SDPA` left unchanged (still `NOT WIN32`) due to separate `get_warp_size()` constexpr issues

### Why It Matters
This is the highest-leverage change so far — it unlocks all CK GEMM optimizations (BFloat16/float16/float32 dispatch, group GEMM, BGEMM) for Windows ROCm users. Without this, Windows users get no benefit from the CK GEMM path at all. All the arch list fixes and dispatch improvements from iterations 1-3 were Linux-only because of this gate.

### Research Findings
- ScottTodd (AMD) confirmed: "it builds successfully on Windows if you build rock with composable kernel and then enable USE_ROCM_CK_GEMM in torch and remove the windows check in torch cmake"
- A previous linking issue with `group_gemm_ck` on Windows (TheRock issue #2054) was fixed in upstream PyTorch commit `f9b81e23`
- CK SDPA has separate Windows issues (PR #182733, #184375) — `get_warp_size() not constexpr` on certain ROCm versions
- The CK GEMM HIP source files have zero Windows-specific guards — they're clean HIP code
- `hasCKGEMM()` in CUDAHooks.cpp is already properly guarded with `#if defined(USE_ROCM) && defined(USE_ROCM_CK_GEMM)`

### Status
Complete. Committed as `fdc71bd`.

### What Was Learned
- The `NOT WIN32` restriction on CK GEMM was likely a conservative default that became outdated as ROCm Windows support matured
- CK GEMM and CK SDPA have separate Windows readiness — GEMM works, SDPA doesn't yet
- The ATen CMakeLists.txt conditionally includes CK headers and excludes CK/BGEMM .hip files based on `USE_ROCM_CK_GEMM` — all properly structured
- The `ck_group_gemm.h` declaration is unconditionally included but the call sites are guarded with `#if defined(USE_ROCM_CK_GEMM)`

### Next Iteration Should Tackle
- Improve BGEMM dispatch heuristic — only 8 of 32 available kernel variants are used, and the dispatch has gaps for n=512-1024 and n=2048 ranges
- Or: Enable the commented-out lookup_dispatch map for specific shape dispatching
- Or: Add shape-based tile selection to WMMA GEMM dispatch
- Or: Investigate enabling CK SDPA on Windows (may require CK submodule update)
