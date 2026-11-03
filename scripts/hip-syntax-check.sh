#!/bin/bash
# Compile-check HIP translation units for gfx1100 without a GPU (needs hipcc + hipblas headers; Ubuntu: apt install
# hipcc libamdhip64-dev libhipblas-dev librocblas-dev). Usage: scripts/hip-syntax-check.sh [-c] file.cu...
#   default: -fsyntax-only (host and device passes);  -c: full device codegen with kernel resource usage remarks.
set -u
cd "$(dirname "$0")/.."
MODE=syntax; if [ "${1:-}" = "-c" ]; then MODE=obj; shift; fi
DEFS="-DGGML_USE_HIP -DGGML_USE_CUDA -DGGML_HIP_GRAPHS -DGGML_HIP_NO_VMM -DGGML_CUDA_NO_PEER_COPY -DGGML_SHARED -DGGML_BUILD -DGGML_HIP_NO_MMQ_MFMA"
INC="-I ggml/include -I ggml/src -I ggml/src/ggml-cuda"
# flash-attention K/V type combinations (ggml/CMakeLists.txt GGML_CUDA_FA_QUANTS: q4_0-q4_0, q8_0-q8_0, f16-f16, bf16-bf16 on by default)
FADEFS=""
for K in F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0; do for V in F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0; do
  on=0; [ "$K" = "$V" ] && case $K in F16|BF16|Q4_0|Q8_0) on=1;; esac
  FADEFS="$FADEFS -DGGML_CUDA_FA_${K}_${V}=$on"
done; done
DEFS="$DEFS $FADEFS"
rc=0
for f in "$@"; do
  echo "== $f ($MODE)"
  if [ $MODE = syntax ]; then
    hipcc --offload-arch=gfx1100 -fsyntax-only -std=c++17 -x hip $DEFS $INC "$f" 2>&1 | grep -v 'hipStreamWaitEvent' | grep -E 'error|fatal' -A3 | head -40
  else
    mkdir -p /tmp/hipobj
    hipcc --offload-arch=gfx1100 -c -O3 -std=c++17 -x hip $DEFS $INC -Rpass-analysis=kernel-resource-usage "$f" -o /tmp/hipobj/$(basename "$f").o 2>&1 \
      | grep -E 'error|fatal|remark:.*(Function Name|VGPRs:|SGPRs|LDS Size|Occupancy|ScratchSize)' | head -80
  fi
  [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
done
exit $rc
