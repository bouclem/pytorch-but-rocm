#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/native/hip/ck_bgemm.h>
#include <ATen/native/hip/bgemm_kernels/bgemm_kernel_collection.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/util/env.h>
#include <cstdio>

namespace at::native {

using BGEMMKernel_BFloat16 = std::function<void(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16))>;


struct IntTupleHash {
  size_t operator()(const std::tuple<int, int, int>& t) const {
    auto hash1 = std::hash<int>{}(std::get<0>(t));
    auto hash2 = std::hash<int>{}(std::get<1>(t));
    auto hash3 = std::hash<int>{}(std::get<2>(t));
    return hash1 ^ hash2 ^ hash3;
  }
};

// Map for specific shape dispatching.
// Entries target common LLM attention BGEMM shapes:
// - Q@K^T: (seq_len, seq_len, head_dim) — square shapes with small k
// - Attn@V: (seq_len, head_dim, seq_len) — tall-skinny with large k
static const std::unordered_map<
    std::tuple<int, int, int>,
    BGEMMKernel_BFloat16,
    IntTupleHash>
    lookup_dispatch = {
        // Existing tuned shape
        {{5120, 64, 8192},
         bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        // Q@K^T with head_dim=64: square (seq,seq) shapes
        {{512, 512, 64},
         bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        {{1024, 1024, 64},
         bgemm_kernel_bf16bf16bf16_256_224x256x64_16x16_7x8_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        {{2048, 2048, 64},
         bgemm_kernel_bf16bf16bf16_256_224x256x64_16x16_7x8_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        {{4096, 4096, 64},
         bgemm_kernel_bf16bf16bf16_256_256x224x64_16x16_8x7_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3},
        {{8192, 8192, 64},
         bgemm_kernel_bf16bf16bf16_256_256x256x32_32x32_4x4_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v4},
        // Q@K^T with head_dim=128: square (seq,seq) shapes
        {{512, 512, 128},
         bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        {{1024, 1024, 128},
         bgemm_kernel_bf16bf16bf16_256_224x256x64_16x16_7x8_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        {{2048, 2048, 128},
         bgemm_kernel_bf16bf16bf16_256_224x256x64_16x16_7x8_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        {{4096, 4096, 128},
         bgemm_kernel_bf16bf16bf16_256_256x224x64_16x16_8x7_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3},
        {{8192, 8192, 128},
         bgemm_kernel_bf16bf16bf16_256_256x256x32_32x32_4x4_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v4},
        // Q@K^T with head_dim=256: square (seq,seq) shapes (Gemma, Llama3-70B)
        {{512, 512, 256},
         bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        {{1024, 1024, 256},
         bgemm_kernel_bf16bf16bf16_256_224x256x64_16x16_7x8_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        {{2048, 2048, 256},
         bgemm_kernel_bf16bf16bf16_256_224x256x64_16x16_7x8_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        {{4096, 4096, 256},
         bgemm_kernel_bf16bf16bf16_256_256x224x64_16x16_8x7_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3},
        {{8192, 8192, 256},
         bgemm_kernel_bf16bf16bf16_256_256x256x32_32x32_4x4_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v4},
        // Attn@V: (seq_len, head_dim, seq_len) — tall-skinny with large k
        {{512, 64, 512},
         bgemm_kernel_bf16bf16bf16_128_16x64x64_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4_Intrawave_v2},
        {{512, 128, 512},
         bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        {{1024, 64, 1024},
         bgemm_kernel_bf16bf16bf16_128_16x64x64_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4_Intrawave_v2},
        {{1024, 128, 1024},
         bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        {{2048, 64, 2048},
         bgemm_kernel_bf16bf16bf16_128_16x64x64_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4_Intrawave_v2},
        {{2048, 128, 2048},
         bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        {{2048, 256, 2048},
         bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        {{4096, 64, 4096},
         bgemm_kernel_bf16bf16bf16_128_16x64x64_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4_Intrawave_v2},
        {{4096, 128, 4096},
         bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        {{4096, 256, 4096},
         bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        {{8192, 64, 8192},
         bgemm_kernel_bf16bf16bf16_128_16x64x64_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4_Intrawave_v2},
        {{8192, 128, 8192},
         bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
        {{8192, 256, 8192},
         bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3},
    };


// This is the heuristic to choose a kernel based on inputs
BGEMMKernel_BFloat16 dispatch_bfloat16_bgemm(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  static const bool bgemm_tile_verbose =
      c10::utils::check_env("TORCH_ROCM_GEMM_TILE_VERBOSE") == true;

  // Direct lookup for known high-priority shapes (e.g. LLM attention patterns)
  auto it = lookup_dispatch.find({m, n, k});
  if (it != lookup_dispatch.end()) {
    if (bgemm_tile_verbose) {
      std::fprintf(stderr, "[TORCH_ROCM_GEMM_TILE_VERBOSE] bgemm dispatch=lookup "
          "m=%lld n=%lld k=%lld batch=%lld\n",
          (long long)m, (long long)n, (long long)k, (long long)num_batches);
    }
    return it->second;
  }

  if (bgemm_tile_verbose) {
    std::fprintf(stderr, "[TORCH_ROCM_GEMM_TILE_VERBOSE] bgemm dispatch=heuristic "
        "m=%lld n=%lld k=%lld batch=%lld\n",
        (long long)m, (long long)n, (long long)k, (long long)num_batches);
  }

  // Heuristic dispatch based on m, n dimensions.
  // Note: B is A and A is B, so m<-->n in the actual kernel call.

  if (m <= 5120) {
    if (n <= 4) return bgemm_kernel_bf16bf16bf16_64_16x16x64_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4_Intrawave_v1;
    else if (n <= 32) return bgemm_kernel_bf16bf16bf16_128_16x64x64_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4_Intrawave_v2;
    else if (n <= 128) return bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3;
    else if (n <= 512) return bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3;
    else if (n <= 1024) return bgemm_kernel_bf16bf16bf16_256_224x256x64_16x16_7x8_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3;
    else if (n <= 2048) return bgemm_kernel_bf16bf16bf16_256_224x256x64_16x16_7x8_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3;
    else if (n <= 4096) return bgemm_kernel_bf16bf16bf16_256_224x256x64_16x16_7x8_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3;
  }
  else if (m <= 8192) {
    if (n <= 8) return bgemm_kernel_bf16bf16bf16_128_16x32x64_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4_Intrawave_v1;
    if (n <= 32) return bgemm_kernel_bf16bf16bf16_128_16x64x64_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4_Intrawave_v2;
    if (n <= 512) return bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3;
    if (n <= 4096) return bgemm_kernel_bf16bf16bf16_256_256x224x64_16x16_8x7_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3;
  }
  else {
    // Large m: use larger tiles for better occupancy
    if (n <= 16) return bgemm_kernel_bf16bf16bf16_256_256x16x64_16x16_4x1_8x32x1_8x16x1_1x32x1x8_2_Intrawave_v2;
    if (n <= 128) return bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3;
    if (n <= 4096) return bgemm_kernel_bf16bf16bf16_256_256x224x64_16x16_8x7_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3;
    // Large square shapes: use 256x256x32 for better tile fit
    return bgemm_kernel_bf16bf16bf16_256_256x256x32_32x32_4x4_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v4;
  }

  // Default instance
  return bgemm_kernel_bf16bf16bf16_256_256x224x64_16x16_8x7_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3;
}

template <>
void bgemm_internal_ck<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  // CK BGEMM kernels are XDL-only (CDNA/gfx9). No WMMA BGEMM path exists.
  TORCH_CHECK(
      at::detail::getCUDAHooks().isGPUArch({"gfx9"}),
      "bgemm_internal_ck<at::BFloat16> requires CDNA (gfx9) architecture. ",
      "CK BGEMM kernels use XDL instructions not available on RDNA. ",
      "Use a different BLAS backend (hipBLASLt or rocBLAS) for BGEMM on this GPU.");

  BGEMMKernel_BFloat16 bgemm_impl = dispatch_bfloat16_bgemm(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  bgemm_impl(CUDABLAS_BGEMM_ARGS(at::BFloat16));
}

} // namespace at::native
