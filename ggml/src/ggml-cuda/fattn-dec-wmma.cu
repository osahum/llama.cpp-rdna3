// SPDX-License-Identifier: Apache-2.0
// Derived from hipfire kernels/src/attention_verify_wmma.gfx1100.hip (commit efa8fbb),
// Copyright (c) 2026 Ben1078, Apache-2.0; see NOTICE. Adapted for ggml by the llama-rdna project:
// ggml tensor strides, ggml's f16 KQ mask instead of a positions array, q4_0 as well as q8_0 K/V, runtime GQA ratio.
//
// Exact-gfx11 WMMA split-K attention over a quantized KV cache, head_dim 256, 1..=8 query rows per KV head group:
//  * One block = (kv head, span of keys), 4*QT waves, QT = ceil(gqa*rows/16) 16-query WMMA tiles
//    (query qi = h_local*rows + r).
//  * Wave w = (qt = w>>2, ks = (w>>1)&1, d = w&1): query tile qt, key half ks of every 32-key tile (its own
//    online-softmax stream), dim half d. QK: S^T = K * Q^T over the wave's 128 dims (Q fragments stay in VGPRs);
//    the two dim halves exchange partial S once per tile through LDS. PV: O^T = V^T * P^T for the wave's 128 dims.
//  * Interleaved key tiles: block b of a kv head owns the 32-key tiles b, b + n_spans, b + 2 n_spans, ...
//  * K plane (f16 [key][256], rows padded to 528 B) and V^T plane (f16 [dim][32 keys], 16-byte granules
//    XOR-swizzled) are single-buffered but filled in alternating phases: phase A fills V(t) while QK(t) reads K(t);
//    phase B fills K(t+1) while PV(t) reads V(t). Two LDS-only barriers per tile. Raw quantized bytes are prefetched
//    into registers one tile ahead.
//  * Lazy rescale: the running max only moves when a tile exceeds it by VW_THRESH (log2 units).
//  * The two key-half streams are merged in LDS at the end; one partial per (row, head, span):
//    [m (log2 units), l, pad, pad, O[256] in WMMA C order], reduced by fattn_dec_wmma_reduce into ggml's [D, head, row].
#include "fattn-dec-wmma.cuh"

#include <cstring>
#include <cstdlib>

#define VW_KT      32
#define VW_KROW    132
#define VW_PSTRIDE 260
#define VW_THRESH  8.0f
#define VW_MAX_SPANS 64

template <int QT>
struct VwCfg {
    static constexpr int NW = 4 * QT;
    static constexpr int NT = 32 * NW;
    static constexpr int UPT = (QT == 1) ? 2 : 1;      // fill units per thread per tensor
    static constexpr int K_DW = VW_KT * VW_KROW;        // 4224
    static constexpr int V_DW = 256 * 16 + 48;          // 4144
    static constexpr int S_DW = NW * 256;
    static constexpr int MERGE_DW = QT * (2048 + 64);
    static constexpr int MAIN_DW = K_DW + V_DW + S_DW;
    static constexpr int LDS_DW = MAIN_DW > MERGE_DW ? MAIN_DW : MERGE_DW;
};

struct vw_args {
    const float * q;              // f32, element (dim, head, row) at row*sq1 + head*sq2 + dim
    const unsigned char * k;      // quantized, (dim block b, key, kv head) at key*nbk1 + kv_head*nbk2 + ts*b
    const unsigned char * v;
    const uint16_t * mask;        // f16 bits, (key, row) at row*sm1 + key
    float * partials;             // [rows][n_head][n_spans][VW_PSTRIDE]
    int n_kv;                     // keys in the K/V view (mask covers at least these)
    int n_head;
    int gqa;                      // query heads per kv head
    int rows;                     // 1..8
    int n_spans;
    size_t sq1, sq2, sm1;         // element strides
    size_t nbk1, nbk2, nbv1, nbv2; // byte strides
    float scale_log2;
};

#if defined(GGML_USE_HIP) && defined(RDNA3)

typedef _Float16 vw_h16 __attribute__((ext_vector_type(16)));
typedef _Float16 vw_h2 __attribute__((ext_vector_type(2)));
typedef float vw_f8 __attribute__((ext_vector_type(8)));
typedef uint32_t vw_u4 __attribute__((ext_vector_type(4)));
typedef uint32_t vw_u8 __attribute__((ext_vector_type(8)));
typedef uint32_t vw_u4h __attribute__((ext_vector_type(4), aligned(2)));
typedef uint32_t vw_u32a2 __attribute__((aligned(2)));
typedef union { uint32_t u; vw_h2 h2; } vw_pk;
struct vw_true { constexpr operator bool() const { return true; } };

#define VW_SB() asm volatile("" ::: "memory")
// Workgroup barrier that only orders LDS (see the hipfire original for why not __syncthreads()).
#define VW_BAR() do { __builtin_amdgcn_sched_barrier(0); \
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); \
    __builtin_amdgcn_sched_barrier(0); } while (0)

static __device__ __forceinline__ vw_u8 vw_kfrag(const uint32_t* p) {
    const vw_u4 lo = *(const vw_u4*)(p);
    const vw_u4 hi = *(const vw_u4*)(p + 4);
    return (vw_u8){lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
}
static __device__ __forceinline__ vw_u8 vw_vfrag(const uint32_t* p0, const uint32_t* p1) {
    const vw_u4 lo = *(const vw_u4*)(p0);
    const vw_u4 hi = *(const vw_u4*)(p1);
    return (vw_u8){lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
}
static __device__ __forceinline__ uint32_t vw_x16(uint32_t v) {
    return (uint32_t)__builtin_amdgcn_permlanex16((int)v, (int)v, 0x76543210, 0xfedcba98, false, false);
}
static __device__ __forceinline__ float vw_x16f(float v) {
    return __builtin_bit_cast(float, vw_x16(__builtin_bit_cast(uint32_t, v)));
}

// Two codes (bytes of tb picked by sel, interleaved with 0x64 -> f16 1024+code) times the f16 scale pair s2,
// after subtracting BIAS (1152 for q8_0 codes xor 0x80, 1032 for q4_0 nibbles): RNE(s*c) per half.
template <int BIAS>
static __device__ __forceinline__ uint32_t vw_deq2(uint32_t tb, uint32_t sel, uint32_t s2) {
    vw_pk a, s;
    a.u = __builtin_amdgcn_perm(0x64646464u, tb, sel);
    s.u = s2;
    const vw_h2 bias = {(_Float16)(float)BIAS, (_Float16)(float)BIAS};
    a.h2 = (a.h2 - bias) * s.h2;
    return a.u;
}

// Raw block loads. w[8] holds the f16 scale in its low half; w[0..7] the code bytes (q8_0) or w[0..3] the nibble
// bytes (q4_0). Block b of a 256-dim head row: q8_0 at 34b (f16 d + 32 int8), q4_0 at 18b (f16 d + 16 bytes).
template <ggml_type T>
static __device__ __forceinline__ void vw_load(const unsigned char* __restrict__ row, int b, uint32_t w[9]) {
    if constexpr (T == GGML_TYPE_Q8_0) {
        const unsigned char* p = row + 34 * b;
        const vw_u4h x0 = *(const vw_u4h*)(p + 2);
        const vw_u4h x1 = *(const vw_u4h*)(p + 18);
        w[8] = *(const vw_u32a2*)p;
        w[0] = x0.x; w[1] = x0.y; w[2] = x0.z; w[3] = x0.w;
        w[4] = x1.x; w[5] = x1.y; w[6] = x1.z; w[7] = x1.w;
    } else {
        const unsigned char* p = row + 18 * b;
        const vw_u4h x0 = *(const vw_u4h*)(p + 2);
        w[8] = *(const vw_u32a2*)p;
        w[0] = x0.x; w[1] = x0.y; w[2] = x0.z; w[3] = x0.w;
        w[4] = 0u; w[5] = 0u; w[6] = 0u; w[7] = 0u;
    }
}
// scale pair + 8 code dwords (one byte per value, in dim order) of the loaded block.
template <ggml_type T>
static __device__ __forceinline__ void vw_codes(const uint32_t w[9], uint32_t& s2, uint32_t c[8]) {
    uint32_t x[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        x[i] = w[i];
        asm volatile("" : "+v"(x[i]));
    }
    s2 = __builtin_amdgcn_perm(x[8], x[8], 0x01000100u);
    if constexpr (T == GGML_TYPE_Q8_0) {
        #pragma unroll
        for (int j = 0; j < 8; j++) c[j] = x[j] ^ 0x80808080u;
    } else {
        // q4_0: byte j of qs holds value j (low nibble) and value j+16 (high nibble)
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            c[j]     = x[j] & 0x0F0F0F0Fu;
            c[4 + j] = (x[j] >> 4) & 0x0F0F0F0Fu;
        }
    }
}
template <ggml_type T> struct vw_bias { static constexpr int value = T == GGML_TYPE_Q8_0 ? 1152 : 1032; };

// K unit (key = lane, block b) -> K plane row granules 4b..4b+3.
template <ggml_type T>
static __device__ __forceinline__ void vw_fill_k(uint32_t* __restrict__ kdst, const uint32_t w[9]) {
    uint32_t s2, c[8];
    vw_codes<T>(w, s2, c);
    constexpr int B = vw_bias<T>::value;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        vw_u4 v;
        v.x = vw_deq2<B>(c[2 * i], 0x04010400u, s2);
        v.y = vw_deq2<B>(c[2 * i], 0x04030402u, s2);
        v.z = vw_deq2<B>(c[2 * i + 1], 0x04010400u, s2);
        v.w = vw_deq2<B>(c[2 * i + 1], 0x04030402u, s2);
        *(vw_u4*)(kdst + 4 * i) = v;
    }
}

// V unit -> V^T plane. vb[jj] = per-lane dword offsets (row 32b + odd, its 2-dword pad, swizzled key-pair column)
// for (j & 3) == jj; dims 32b + 4j + {0,2} + odd go to rows at +64j and +64j+32 dwords.
template <ggml_type T>
static __device__ __forceinline__ void vw_fill_v(uint32_t* __restrict__ vdst, const int vb[4], uint32_t selv, const uint32_t w[9]) {
    uint32_t s2, c[8];
    vw_codes<T>(w, s2, c);
    constexpr int B = vw_bias<T>::value;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const uint32_t p01 = vw_deq2<B>(c[j], 0x04010400u, s2);
        const uint32_t p23 = vw_deq2<B>(c[j], 0x04030402u, s2);
        const uint32_t q01 = (uint32_t)__builtin_amdgcn_mov_dpp((int)p01, 0xB1, 0xf, 0xf, true);
        const uint32_t q23 = (uint32_t)__builtin_amdgcn_mov_dpp((int)p23, 0xB1, 0xf, 0xf, true);
        vdst[vb[j & 3] + 64 * j] = __builtin_amdgcn_perm(q01, p01, selv);
        vdst[vb[j & 3] + 64 * j + 32] = __builtin_amdgcn_perm(q23, p23, selv);
    }
}

template <int QT, ggml_type T>
static __device__ __forceinline__ void vw_body(const vw_args & a, uint32_t* __restrict__ lds) {
    using C = VwCfg<QT>;
    uint32_t* Kl = lds;
    uint32_t* Vl = lds + C::K_DW;
    float* Sx = (float*)(lds + C::K_DW + C::V_DW);

    const int tid = threadIdx.x;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int half = lane >> 4;
    const int kv_h = blockIdx.x;
    const int span_idx = blockIdx.y;   // this block owns 32-key tiles span_idx + n_spans * t

    const int n_kv = a.n_kv;
    const int total_tiles = (n_kv + VW_KT - 1) / VW_KT;
    if (span_idx >= total_tiles) return;
    const int n_spans = a.n_spans;
    const int n_tiles = (total_tiles - span_idx + n_spans - 1) / n_spans;
    const int rows = a.rows;
    const int gqa = a.gqa;

    // ---- compute roles ----
    const int d = wave & 1;
    const int ks = (wave >> 1) & 1;
    const int qt = wave >> 2;
    const int qi = qt * 16 + ml;
    const bool qok = qi < gqa * rows;
    const bool active = qt * 16 < gqa * rows;   // wave-uniform
    const int hl = qok ? qi / rows : 0;
    const int r = qok ? qi - hl * rows : 0;
    const int head = kv_h * gqa + hl;
    const uint32_t * mrow = (const uint32_t *)(a.mask + (size_t) r * a.sm1);   // dword = key pair (even, odd)

    vw_h16 qf[8];
    vw_f8 O[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) O[i] = (vw_f8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float m_run = -INFINITY;
    float l_run = 0.0f;

    // per-lane LDS offsets (dwords)
    const int k_rd = (ks * 16 + ml) * VW_KROW + d * 64;                  // + 8*c
    const int y = (ml >> 2) & 3;
    const int v_rd0 = (d * 128 + ml) * 16 + 32 * d + (((2 * ks) ^ y) << 2);      // + 256 c + 4 (c >> 1)
    const int v_rd1 = (d * 128 + ml) * 16 + 32 * d + (((2 * ks + 1) ^ y) << 2);
    float* sx_own = Sx + wave * 256;
    float* sx_par = Sx + (wave ^ 1) * 256;
    const uint32_t sel_e0 = half ? 0x01000504u : 0x05040100u;
    const uint32_t sel_e1 = half ? 0x03020706u : 0x07060302u;

    // ---- fill roles: lane -> (block fb = (lane>>1)&7, key fk); adjacent lanes hold keys 2m, 2m+1 of the same block.
    const bool filler = (QT <= 2) || wave < 8;
    const int odd = lane & 1;
    const int fb = (lane >> 1) & 7;
    const int fk = (C::UPT == 2) ? (wave * 4 + 2 * (lane >> 4) + odd)
                                 : ((wave & 7) * 4 + 2 * (lane >> 4) + odd);
    const uint32_t selv = odd ? 0x03020706u : 0x05040100u;
    int vb[C::UPT][4];
    #pragma unroll
    for (int u = 0; u < C::UPT; u++) {
        const int m2 = (fk + 16 * u) >> 1;
        #pragma unroll
        for (int jj = 0; jj < 4; jj++)
            vb[u][jj] = (32 * fb + odd) * 16 + 4 * ((fb & 3) + 8 * (fb >> 2)) + (((m2 >> 2) ^ jj) << 2) + (m2 & 3);
    }
    const unsigned char * kbase_ptr = a.k + (size_t) kv_h * a.nbk2;
    const unsigned char * vbase_ptr = a.v + (size_t) kv_h * a.nbv2;
    const size_t nbk1 = a.nbk1, nbv1 = a.nbv1;

    auto load_tile = [&](const unsigned char* __restrict__ base, size_t nb1, int t, uint32_t (&dst)[C::UPT][9]) __attribute__((always_inline)) {
        const int key0 = (span_idx + n_spans * t) * VW_KT + fk;
        // Keys past the end are clamped to the last valid row (their scores are masked, P = 0 times finite V adds 0).
        #pragma unroll
        for (int u = 0; u < C::UPT; u++) {
            const int key = min(key0 + 16 * u, n_kv - 1);
            vw_load<T>(base + (size_t) key * nb1, fb, dst[u]);
        }
    };
    auto fill_k = [&](const uint32_t (&src)[C::UPT][9]) __attribute__((always_inline)) {
        #pragma unroll
        for (int u = 0; u < C::UPT; u++) vw_fill_k<T>(Kl + (fk + 16 * u) * VW_KROW + 16 * fb, src[u]);
    };
    auto fill_v = [&](const uint32_t (&src)[C::UPT][9]) __attribute__((always_inline)) {
        #pragma unroll
        for (int u = 0; u < C::UPT; u++) vw_fill_v<T>(Vl, vb[u], selv, src[u]);
    };

    constexpr int D = 1;
    uint32_t rv[D][C::UPT][9], rk[D][C::UPT][9];
    if (filler) {
        load_tile(kbase_ptr, nbk1, 0, rk[0]);
        #pragma unroll
        for (int i = 0; i < D; i++) if (i < n_tiles) load_tile(vbase_ptr, nbv1, i, rv[i]);
    }
    // Q fragments (f16) for this wave's 8 dim chunks
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        vw_h16 f;
        if (qok) {
            const float* qp = a.q + (size_t) r * a.sq1 + (size_t) head * a.sq2 + (d * 8 + i) * 16;
            const float4 x0 = *(const float4*)(qp);
            const float4 x1 = *(const float4*)(qp + 4);
            const float4 x2 = *(const float4*)(qp + 8);
            const float4 x3 = *(const float4*)(qp + 12);
            f[0] = (_Float16)x0.x; f[1] = (_Float16)x0.y; f[2] = (_Float16)x0.z; f[3] = (_Float16)x0.w;
            f[4] = (_Float16)x1.x; f[5] = (_Float16)x1.y; f[6] = (_Float16)x1.z; f[7] = (_Float16)x1.w;
            f[8] = (_Float16)x2.x; f[9] = (_Float16)x2.y; f[10] = (_Float16)x2.z; f[11] = (_Float16)x2.w;
            f[12] = (_Float16)x3.x; f[13] = (_Float16)x3.y; f[14] = (_Float16)x3.z; f[15] = (_Float16)x3.w;
        } else {
            #pragma unroll
            for (int e = 0; e < 16; e++) f[e] = (_Float16)0.0f;
        }
        qf[i] = f;
    }

    if (filler) {
        fill_k(rk[0]);
        #pragma unroll
        for (int i = 1; i <= D; i++) if (i < n_tiles) load_tile(kbase_ptr, nbk1, i, rk[i % D]);
    }
    VW_BAR();

    const float scale_log2 = a.scale_log2;
    constexpr float LOG2E = 1.4426950408889634f;

    auto tile = [&](int t, uint32_t (&rv_s)[C::UPT][9], uint32_t (&rk_s)[C::UPT][9],
                    auto lv, auto lk, auto kf) __attribute__((always_inline)) {
        const bool pv = lv, pk = lk, fkk = kf;
        const int kt = (span_idx + n_spans * t) * VW_KT;
        // ---- phase A: fill V(t); QK(t) ----
        if (filler) {
            fill_v(rv_s);
            if (pv) load_tile(vbase_ptr, nbv1, t + D, rv_s);
        }
        vw_f8 sacc = (vw_f8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (active) {
        VW_SB();
        {
            vw_u8 ka = vw_kfrag(Kl + k_rd);
            vw_u8 kb = vw_kfrag(Kl + k_rd + 8);
            sacc = (vw_f8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int c = 0; c < 8; c += 2) {
                sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(vw_h16, ka), qf[c], sacc);
                if (c + 2 < 8) ka = vw_kfrag(Kl + k_rd + 8 * (c + 2));
                sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(vw_h16, kb), qf[c + 1], sacc);
                if (c + 3 < 8) kb = vw_kfrag(Kl + k_rd + 8 * (c + 3));
            }
            __builtin_amdgcn_sched_group_barrier(0x100, 4, 1);
            #pragma unroll
            for (int c = 0; c < 6; c++) {
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
                __builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
        }
        *(float4*)(sx_own + lane * 4) = (float4){sacc[0], sacc[1], sacc[2], sacc[3]};
        *(float4*)(sx_own + 128 + lane * 4) = (float4){sacc[4], sacc[5], sacc[6], sacc[7]};
        }
        VW_BAR();
        // ---- phase B: fill K(t+1); softmax; PV(t) ----
        if (filler && fkk) {
            fill_k(rk_s);
            if (pk) load_tile(kbase_ptr, nbk1, t + 1 + D, rk_s);
        }
        if (active) {
        {
            const float4 x0 = *(const float4*)(sx_par + lane * 4);
            const float4 x1 = *(const float4*)(sx_par + 128 + lane * 4);
            sacc[0] += x0.x; sacc[1] += x0.y; sacc[2] += x0.z; sacc[3] += x0.w;
            sacc[4] += x1.x; sacc[5] += x1.y; sacc[6] += x1.z; sacc[7] += x1.w;
        }
        // scores of this lane: keys kbase + 2j + half, query row r. Scale and add the mask (log2 domain).
        const int kbase = kt + ks * 16;
        float tv[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int key = kbase + 2 * j + half;
            float mf = -INFINITY;
            if (qok && key < n_kv) {
                const uint32_t mv = mrow[(kbase >> 1) + j];
                const uint16_t mh = half ? (uint16_t)(mv >> 16) : (uint16_t)(mv & 0xFFFFu);
                mf = (float) __builtin_bit_cast(_Float16, mh);
            }
            tv[j] = __builtin_fmaf(mf, LOG2E, sacc[j] * scale_log2);
        }
        float mx = tv[0];
        #pragma unroll
        for (int j = 1; j < 8; j++) mx = fmaxf(mx, tv[j]);
        mx = fmaxf(mx, vw_x16f(mx));
        if (__builtin_amdgcn_ballot_w32(mx > m_run + VW_THRESH)) {
            const float mn = fmaxf(m_run, mx);
            const float alpha = (mn == -INFINITY) ? 1.0f : __builtin_amdgcn_exp2f(m_run - mn);
            m_run = mn;
            l_run *= alpha;
            #pragma unroll
            for (int c = 0; c < 8; c++) O[c] *= alpha;
        }
        const float mu = (m_run == -INFINITY) ? 0.0f : m_run;
        float p[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            p[j] = __builtin_amdgcn_exp2f(tv[j] - mu);
            l_run += p[j];
        }
        uint32_t dw[4], xw[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            vw_pk pk2;
            pk2.h2 = (vw_h2){(_Float16)p[2 * i], (_Float16)p[2 * i + 1]};
            dw[i] = pk2.u;
            xw[i] = vw_x16(dw[i]);
        }
        vw_u8 pf;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            pf[2 * i] = __builtin_amdgcn_perm(xw[i], dw[i], sel_e0);
            pf[2 * i + 1] = __builtin_amdgcn_perm(xw[i], dw[i], sel_e1);
        }
        const vw_h16 pfrag = __builtin_bit_cast(vw_h16, pf);
        VW_SB();
        {
            vw_u8 va = vw_vfrag(Vl + v_rd0, Vl + v_rd1);
            vw_u8 vb2 = vw_vfrag(Vl + v_rd0 + 256, Vl + v_rd1 + 256);
            #pragma unroll
            for (int c = 0; c < 8; c += 2) {
                O[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(vw_h16, va), pfrag, O[c]);
                if (c + 2 < 8) {
                    const int o = 256 * (c + 2) + 4 * ((c + 2) >> 1);
                    va = vw_vfrag(Vl + v_rd0 + o, Vl + v_rd1 + o);
                }
                O[c + 1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(vw_h16, vb2), pfrag, O[c + 1]);
                if (c + 3 < 8) {
                    const int o = 256 * (c + 3) + 4 * ((c + 3) >> 1);
                    vb2 = vw_vfrag(Vl + v_rd0 + o, Vl + v_rd1 + o);
                }
            }
            __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
            #pragma unroll
            for (int c = 0; c < 6; c++) {
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
        }
        }
        VW_BAR();
    };

    {
        using T_ = vw_true;
        int t = 0;
        #pragma unroll 1
        for (; t + 2 * D < n_tiles; t += D) {
            #pragma unroll
            for (int i = 0; i < D; i++) tile(t + i, rv[i], rk[(i + 1) % D], T_{}, T_{}, T_{});
        }
        #pragma unroll
        for (int i = 0; i < 2 * D; i++) {
            const int tt = t + i;
            if (tt < n_tiles)
                tile(tt, rv[i % D], rk[(i + 1) % D], tt + D < n_tiles, tt + 1 + D < n_tiles, tt + 1 < n_tiles);
        }
    }

    // ---- merge the two key-half streams, write partials ----
    l_run += vw_x16f(l_run);
    float* mg = (float*)lds;
    #pragma unroll 1
    for (int round = 0; round < 2; round++) {
        float* slot = mg + qt * (2048 + 64);
        if (ks == 1 && d == round) {
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                *(float4*)(slot + (c * 2 + 0) * 128 + lane * 4) = (float4){O[c][0], O[c][1], O[c][2], O[c][3]};
                *(float4*)(slot + (c * 2 + 1) * 128 + lane * 4) = (float4){O[c][4], O[c][5], O[c][6], O[c][7]};
            }
            *(float2*)(slot + 2048 + lane * 2) = (float2){m_run, l_run};
        }
        VW_BAR();
        if (ks == 0 && d == round) {
            const float2 ml2 = *(const float2*)(slot + 2048 + lane * 2);
            const float mo = ml2.x, lo = ml2.y;
            const float mn = fmaxf(m_run, mo);
            const float a0 = (m_run == -INFINITY) ? 0.0f : __builtin_amdgcn_exp2f(m_run - mn);
            const float a1 = (mo == -INFINITY) ? 0.0f : __builtin_amdgcn_exp2f(mo - mn);
            if (qok) {
                float* pp = a.partials + (((size_t) r * a.n_head + head) * n_spans + span_idx) * VW_PSTRIDE;
                if (d == 0 && half == 0) {
                    pp[0] = mn;
                    pp[1] = l_run * a0 + lo * a1;
                }
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    const float4 x = *(const float4*)(slot + (c * 2 + 0) * 128 + lane * 4);
                    const float4 z = *(const float4*)(slot + (c * 2 + 1) * 128 + lane * 4);
                    float4 w0, w1;
                    w0.x = O[c][0] * a0 + x.x * a1; w0.y = O[c][1] * a0 + x.y * a1;
                    w0.z = O[c][2] * a0 + x.z * a1; w0.w = O[c][3] * a0 + x.w * a1;
                    w1.x = O[c][4] * a0 + z.x * a1; w1.y = O[c][5] * a0 + z.y * a1;
                    w1.z = O[c][6] * a0 + z.z * a1; w1.w = O[c][7] * a0 + z.w * a1;
                    float* dst = pp + 4 + ((d * 8 + c) * 2 + half) * 8;
                    *(float4*)(dst) = w0;
                    *(float4*)(dst + 4) = w1;
                }
            }
        }
        if (round == 0) VW_BAR();
    }
}

#endif // defined(GGML_USE_HIP) && defined(RDNA3)

template <int QT, ggml_type T>
__launch_bounds__(128 * QT) __global__ void fattn_dec_wmma_kernel(const vw_args a) {
#if defined(GGML_USE_HIP) && defined(RDNA3)
    __shared__ __attribute__((aligned(16))) uint32_t lds[VwCfg<QT>::LDS_DW];
    vw_body<QT, T>(a, lds);
#else
    GGML_UNUSED(a);
    NO_DEVICE_CODE;
#endif
}

// Split-K reduce: one block per (head, row), 8 warps over the spans, lane = 8 consecutive WMMA-C-order floats,
// then a cross-warp merge in LDS. exp2 domain. Writes ggml's KQV layout [256, n_head, rows].
#define VW_RU 5
__launch_bounds__(256) __global__ void fattn_dec_wmma_reduce(
    const float* __restrict__ partials, float* __restrict__ out, int n_head, int n_spans_used, int n_spans)
{
#if defined(GGML_USE_HIP) && defined(RDNA3)
    __shared__ float sm[8], sl[8];
    __shared__ __attribute__((aligned(16))) float sg[8][256];
    const int h = blockIdx.x, row = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = n_spans_used;
    const float* base = partials + ((size_t)row * n_head + h) * n_spans * VW_PSTRIDE;
    float m = -INFINITY, l = 0.0f, g[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) g[i] = 0.0f;
    for (int s0 = warp; s0 < n; s0 += 8 * VW_RU) {
        float2 mlv[VW_RU];
        float4 pa[VW_RU], pb[VW_RU];
        #pragma unroll
        for (int u = 0; u < VW_RU; u++) {
            const int s = s0 + 8 * u;
            if (s < n) {
                const float* p = base + (size_t)s * VW_PSTRIDE;
                mlv[u] = *(const float2*)p;
                pa[u] = *(const float4*)(p + 4 + lane * 8);
                pb[u] = *(const float4*)(p + 8 + lane * 8);
            } else {
                mlv[u] = (float2){-INFINITY, 0.0f};
                pa[u] = (float4){0.f, 0.f, 0.f, 0.f};
                pb[u] = pa[u];
            }
        }
        float bm = -INFINITY;
        #pragma unroll
        for (int u = 0; u < VW_RU; u++) if (mlv[u].y > 0.0f) bm = fmaxf(bm, mlv[u].x);
        if (bm > m) {
            const float c = __builtin_amdgcn_exp2f(m - bm);
            l *= c;
            #pragma unroll
            for (int i = 0; i < 8; i++) g[i] *= c;
            m = bm;
        }
        #pragma unroll
        for (int u = 0; u < VW_RU; u++) {
            if (!(mlv[u].y > 0.0f)) continue;
            const float c = __builtin_amdgcn_exp2f(mlv[u].x - m);
            l += mlv[u].y * c;
            g[0] += pa[u].x * c; g[1] += pa[u].y * c; g[2] += pa[u].z * c; g[3] += pa[u].w * c;
            g[4] += pb[u].x * c; g[5] += pb[u].y * c; g[6] += pb[u].z * c; g[7] += pb[u].w * c;
        }
    }
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    *(float4*)(&sg[warp][lane * 8]) = (float4){g[0], g[1], g[2], g[3]};
    *(float4*)(&sg[warp][lane * 8 + 4]) = (float4){g[4], g[5], g[6], g[7]};
    __syncthreads();
    // thread t -> output dim t; C-order index of dim t: chunk c = t>>4, j = (t&15)>>1, half = t&1
    const int t = threadIdx.x;
    const int ci = ((t >> 4) * 2 + (t & 1)) * 8 + ((t & 15) >> 1);
    float M = -INFINITY;
    #pragma unroll
    for (int w = 0; w < 8; w++) if (sl[w] > 0.0f) M = fmaxf(M, sm[w]);
    float L = 0.0f, G = 0.0f;
    #pragma unroll
    for (int w = 0; w < 8; w++) {
        if (!(sl[w] > 0.0f)) continue;
        const float c = __builtin_amdgcn_exp2f(sm[w] - M);
        L += sl[w] * c;
        G += sg[w][ci] * c;
    }
    out[((size_t)row * n_head + h) * 256 + t] = L > 0.0f ? G / L : 0.0f;
#else
    GGML_UNUSED_VARS(partials, out, n_head, n_spans_used, n_spans);
    NO_DEVICE_CODE;
#endif
}

template <ggml_type T>
static void launch_dec_wmma(const vw_args & a, int QT, cudaStream_t stream) {
    const dim3 grid(a.gqa > 0 ? (unsigned) (a.n_head / a.gqa) : 1, a.n_spans, 1);
    switch (QT) {
        case 1: fattn_dec_wmma_kernel<1, T><<<grid, 128, 0, stream>>>(a); break;
        case 2: fattn_dec_wmma_kernel<2, T><<<grid, 256, 0, stream>>>(a); break;
        case 3: fattn_dec_wmma_kernel<3, T><<<grid, 384, 0, stream>>>(a); break;
        default: GGML_ABORT("fattn-dec-wmma: bad QT");
    }
}

bool ggml_cuda_fattn_dec_wmma_supported(const ggml_tensor * dst, int cc) {
    if (!GGML_CUDA_CC_IS_RDNA3(cc)) {
        return false;
    }
    static const bool disabled = getenv("GGML_CUDA_FATTN_DEC_WMMA") != nullptr && atoi(getenv("GGML_CUDA_FATTN_DEC_WMMA")) == 0;
    if (disabled) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f || dst->src[4] != nullptr) {
        return false; // no ALiBi, no softcap, no sinks
    }
    if (Q->type != GGML_TYPE_F32 || Q->ne[0] != 256 || V->ne[0] != 256 || Q->ne[3] != 1 || K->ne[3] != 1) {
        return false;
    }
    if (K->type != V->type || (K->type != GGML_TYPE_Q4_0 && K->type != GGML_TYPE_Q8_0)) {
        return false;
    }
    const int rows = Q->ne[1], n_head = Q->ne[2], n_head_kv = K->ne[2];
    if (rows < 1 || rows > 8 || n_head_kv < 1 || n_head % n_head_kv != 0) {
        return false;
    }
    const int gqa = n_head / n_head_kv;
    if (gqa * rows > 48) {
        return false;
    }
    if (mask == nullptr || mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[1] < rows || mask->ne[2] != 1 || mask->ne[3] != 1 ||
        mask->nb[1] % 4 != 0) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || Q->nb[1] % 16 != 0 || Q->nb[2] % 16 != 0) {
        return false;
    }
    if (K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type) || K->ne[1] < 1 || K->ne[1] % 2 != 0 || V->ne[1] != K->ne[1]) {
        return false; // the mask is read in key pairs
    }
    if (!ggml_is_contiguous(dst) || dst->type != GGML_TYPE_F32) {
        return false;
    }
    return true;
}

void ggml_cuda_flash_attn_ext_dec_wmma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    vw_args a = {};
    a.q = (const float *) Q->data;
    a.k = (const unsigned char *) K->data;
    a.v = (const unsigned char *) V->data;
    a.mask = (const uint16_t *) mask->data;
    a.n_kv = (int) K->ne[1];
    a.n_head = (int) Q->ne[2];
    a.gqa = (int) (Q->ne[2] / K->ne[2]);
    a.rows = (int) Q->ne[1];
    a.sq1 = Q->nb[1] / sizeof(float);
    a.sq2 = Q->nb[2] / sizeof(float);
    a.sm1 = mask->nb[1] / sizeof(uint16_t);
    a.nbk1 = K->nb[1]; a.nbk2 = K->nb[2];
    a.nbv1 = V->nb[1]; a.nbv2 = V->nb[2];
    a.scale_log2 = scale * 1.4426950408889634f;

    const int total_tiles = (a.n_kv + VW_KT - 1) / VW_KT;
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    const int n_head_kv = (int) K->ne[2];
    int n_spans = std::max(1, (2 * nsm) / n_head_kv);
    n_spans = std::min(n_spans, VW_MAX_SPANS);
    n_spans = std::min(n_spans, total_tiles);
    a.n_spans = n_spans;

    ggml_cuda_pool_alloc<float> partials(ctx.pool(), (size_t) a.rows * a.n_head * n_spans * VW_PSTRIDE);
    a.partials = partials.get();

    const int QT = (a.gqa * a.rows + 15) / 16;
    cudaStream_t stream = ctx.stream();
    if (K->type == GGML_TYPE_Q4_0) {
        launch_dec_wmma<GGML_TYPE_Q4_0>(a, QT, stream);
    } else {
        launch_dec_wmma<GGML_TYPE_Q8_0>(a, QT, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    const dim3 grid_r(a.n_head, a.rows, 1);
    fattn_dec_wmma_reduce<<<grid_r, 256, 0, stream>>>(a.partials, (float *) dst->data, a.n_head, n_spans, n_spans);
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("fattn-dec-wmma is only built for HIP");
#endif
}
