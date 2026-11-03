#pragma once
#include "common.cuh"

// Decode / speculative-verify flash attention for RDNA3 (gfx11): head size 256, 1..8 query rows, all query heads of a
// KV head served by one block, K/V read once per step straight from a q4_0 or q8_0 cache (dequantized in the LDS fill),
// WMMA f16 x f16 -> f32, split over key spans with a separate reduce. See fattn-dec-wmma.cu for the origin and license.

bool ggml_cuda_fattn_dec_wmma_supported(const ggml_tensor * dst, int cc);
void ggml_cuda_flash_attn_ext_dec_wmma(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
