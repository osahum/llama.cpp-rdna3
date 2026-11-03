# llama.cpp-rdna3

**Experimental. Vibe coded. No support.**

This is a personal fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) tuned for one model on one GPU: **Qwen3.8-27B (Q4_K_M) with its MTP draft head, on an AMD Radeon RX 7900 XTX (RDNA3, gfx1100), using the Vulkan backend**. It was written almost entirely by an AI coding agent (Claude) and checked with benchmarks and accuracy tests on that one machine. A person did not review the code line by line.

- It may be wrong, crash, or be slower on your setup. Other models, quantizations, GPUs or drivers were not tested.
- I offer **no support**: issues and pull requests may be ignored or closed. Test it yourself before you rely on it.
- It is **not affiliated with or endorsed by** the llama.cpp / ggml project, AMD, Alibaba / Qwen, or hipfire. Please do not report problems with this fork to the llama.cpp project.

The original llama.cpp README is kept as [README.llama.cpp.md](README.llama.cpp.md). This fork is based on upstream commit `0ec4027a2` (2026-10-30).

## What changed from the original llama.cpp

All speed changes are in the Vulkan backend and only apply to this model's shape (head size 256, a q4_0 KV cache, 1 to 8 query tokens per step, or prompt batches over a q4_0 cache). Every other model and shape runs the original code.

1. **New attention kernel for generating tokens** (`flash_attn_dec_w.comp`, plus the earlier `flash_attn_dec_cm.comp` / `flash_attn_dec.comp`). The original reads the KV cache once per query head and token; this one reads it once per step for all heads and all draft tokens, written for RDNA3's wave32 matrix instructions. This is the main reason long conversations stay fast.
2. **The same kernel in "prompt mode" for prompt processing** (prefill) over a q4_0 cache, plus a prompt attention shader (`flash_attn_pf.comp`) used for q8_0 / f16 caches.
3. **Small fusions**: the residual add fused into small batched mat-vecs, an RMS-norm / scale fusion and a gate/up + SwiGLU fusion for this model, the GDN rollback snapshots written straight into the cache, and leaner q4_K / q5_K integer-dot mat-vecs. These are bit-identical to the original.
4. **Probabilistic draft acceptance** for MTP speculative decoding (`--spec-draft-sampling probabilistic`). This mode is upstream code; the fork only measured it and uses it in the recommended command. Rejection sampling keeps the output distribution exactly the model's own.
5. **HIP / ROCm backend experiments** (`ggml/src/ggml-cuda/`): a WMMA decode attention kernel (`fattn-dec-wmma.cu`, derived from hipfire, see Licenses), RDNA3 MMQ tile settings and memory safeguards. These are **not** recommended: the stock HIP server froze the test machine twice at long contexts, and the HIP build was not used for the results below.
6. **Test and measurement tools**: new cases in `tests/test-backend-ops.cpp`, `tools/opbench` (single-op timing), a per-token KL dump in `llama-perplexity` (`LLAMA_KLD_DUMP`) with `bench/kld_pair.py` to compare two builds, and `bench/vision_probe.py` / `bench/cache_probe.py` to check a running server.

No file formats changed. The models, KV cache types and prompt cache are the same as the original's.

## Measured results (one machine, your numbers will differ)

RX 7900 XTX at its 303 W power cap, Mesa RADV, ctx 175,000, MTP draft depth 3. Compared with the unmodified llama.cpp build of the same upstream commit and the same flags:

| context depth | prefill, this fork | prefill, original | time per MTP step, this fork | original |
|---|---|---|---|---|
| 8k | 867 t/s | 806 t/s | 33.5 ms | 37.2 ms |
| 32k | 835 t/s | 682 t/s | 34.7 ms | 43.7 ms |
| 80k | 710 t/s | 495 t/s | 37.9 ms | 57.7 ms |

**Accuracy** was compared token by token against the original build (KL divergence and top-1 agreement against a CPU reference with exact attention, about 12,000 to 16,000 tokens per test). Everything except the new attention kernels gave bit-identical results. The attention kernels were within statistical noise of the original on the prompt and MTP-verify paths, at shallow and 8k-key depths. These are my measurements on one machine, not a guarantee.

## Build (Linux, Vulkan)

You need a recent Mesa RADV driver with `VK_KHR_cooperative_matrix`, plus `glslc`, `cmake`, `ninja` and a C++ compiler.

```bash
cmake -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DGGML_VULKAN=ON
cmake --build build-vulkan --target llama-server
```

## Run

You need `Qwen3.8-27B-Q4_K_M.gguf`, its MTP draft head (`mtp-Qwen3.8-27B-Q4_0.gguf`) and optionally `mmproj-F16.gguf` for images. They are not included here; get them yourself under their own licenses.

```bash
build-vulkan/bin/llama-server \
  --model Qwen3.8-27B-Q4_K_M.gguf --mmproj mmproj-F16.gguf \
  --host 127.0.0.1 --port 8080 --ctx-size 175000 --parallel 1 --n-gpu-layers 99 --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-model mtp-Qwen3.8-27B-Q4_0.gguf --spec-draft-n-max 3 \
  --spec-draft-sampling probabilistic --spec-draft-n-ubatch 64 --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 \
  --batch-size 2048 --ubatch-size 1024 --jinja --no-context-shift
```

The q4_0 KV cache is what the fast paths need. On a 24 GB card, watch `/sys/class/drm/card*/device/mem_info_gtt_used`: if it grows past about 1 GiB during a long request, the context does not fit in VRAM and speed collapses, so lower `--ctx-size`. On the test machine, with about 1 GiB of VRAM used by the desktop, ctx 184,320 fit and 229,376 did not.

## Turning the changes off

Every change can be switched off with an environment variable, so you can compare against the original behaviour:

| variable | effect |
|---|---|
| `GGML_VK_DISABLE_FA_PF=1` | original prompt attention (prompt processing becomes bit-identical to the original) |
| `GGML_VK_FA_DEC_W=0` | generation without the wave32 kernel (falls back to the cooperative-matrix shader, or the original for 1 token) |
| `GGML_VK_DISABLE_FA_DEC=1` | original attention for all generation steps |
| `GGML_VK_DISABLE_MMV_ADD_COLS=1` | original rule for fusing the residual add |
| `GGML_VK_DISABLE_GDN_CACHE_FUSION=1` | GDN snapshots through a scratch tensor, as in the original |
| `GGML_VK_DISABLE_RMS_NORM_SCALE_FUSION=1`, `GGML_VK_DISABLE_RMS_NORM_MUL_MUL_FUSION=1`, `GGML_VK_DISABLE_DUAL_GLU_FUSION=1` | turn off the fusions |
| `GGML_VK_MMVQ_A_OUTER=0`, `GGML_VK_MMVQ_LEAN=0` | original q4_K / q5_K mat-vec indexing |

## Test it yourself

1. `python3 bench/vision_probe.py http://127.0.0.1:8080` checks tool calls, streaming, an image turn, cache reuse and reasoning, and prints timings.
2. `python3 bench/cache_probe.py http://127.0.0.1:8080` checks prompt-cache reuse.
3. For accuracy, build the original llama.cpp at the same commit, run `llama-perplexity` with `--kl-divergence` against the same reference logits on both builds with `LLAMA_KLD_DUMP=<file>` set, and compare the two dumps with `python3 bench/kld_pair.py A a.bin B b.bin`.
4. `test-backend-ops` (build with `-DLLAMA_BUILD_TESTS=ON`) checks the new kernels against the CPU backend.

## Licenses and legal

- **llama.cpp / ggml** code is under the MIT License, copyright (c) 2023-2026 The ggml authors. See [LICENSE](LICENSE). The changes in this fork are offered under the same MIT License, except as noted below.
- **hipfire-derived code.** `ggml/src/ggml-cuda/fattn-dec-wmma.cu` is derived from hipfire's `attention_verify_wmma.gfx1100.hip` (https://github.com/Kaden-Schutt/hipfire, commit efa8fbb, copyright (c) 2026 Ben1078) and stays under the **Apache License 2.0**, as its SPDX header says. The Vulkan attention shaders are new GLSL code that follows the structure of that kernel, and they carry the same attribution. See [NOTICE](NOTICE), [licenses/LICENSE-apache-2.0.txt](licenses/LICENSE-apache-2.0.txt) and hipfire's own notice in [licenses/NOTICE-hipfire.txt](licenses/NOTICE-hipfire.txt). Other bundled third-party licenses are in [licenses/](licenses/).
- **No model weights** are included. Qwen models, their MTP heads and projectors are under their own licenses; follow those if you download them.
- **Trademarks.** AMD, Radeon and RDNA are trademarks of Advanced Micro Devices, Inc. Qwen is a trademark of Alibaba Cloud. Names are used only to describe compatibility.
- **No warranty.** As the MIT and Apache 2.0 licenses state, this software is provided "as is", without warranty of any kind, and the authors are not liable for any damage from its use. GPU kernels can hang or crash a system, and running large contexts can exhaust memory. You use it at your own risk.
