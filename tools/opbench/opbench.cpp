// llama-opbench: time single ggml ops with the exact shapes of a model on one backend.
//
//   llama-opbench -b ROCm0 [-r 20] [-w 3] [-g 10] [-n 8] SPEC...
//     -g: graph computes per synchronize; -n: copies of the op inside one graph (separate outputs); per-op time = time / (g*n)
//   SPEC:
//     mm:TYPE:M:K:N                 MUL_MAT  src0 [K,M] TYPE, src1 [K,N] f32 -> [M,N]
//     fa:D:NQ:NKV:NH:NHKV:TK:TV     FLASH_ATTN_EXT, q f32 [D,NH,NQ] permuted like llama.cpp, k/v [D,NKV,NHKV] TK/TV, f16 mask
//     gdn:HD:NHK:NHV:NT             GATED_DELTA_NET, head dim HD, NHK key heads, NHV value heads, NT tokens, one sequence
//     concat:NA:NB:N1               CONCAT along dim 0 of f32 [NA,N1] and [NB,N1] (the GDN conv-state concat shape is 3:4:10240)
//     glu:N0:N1                     SWIGLU over f32 [2*N0,N1] -> [N0,N1]
//     rms:N0:N1                     RMS_NORM(f32 [N0,N1]) * weight[N0]
//     add:N0:N1                     f32 [N0,N1] + [N0,N1]
//     cpy:N0:N1                     f32 [N0,N1] copy into a new tensor
//   Prints min/avg microseconds and effective GFLOP/s, GB/s (weights or KV bytes only).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t b = 0;
    while (true) {
        size_t e = s.find(sep, b);
        out.push_back(s.substr(b, e == std::string::npos ? std::string::npos : e - b));
        if (e == std::string::npos) break;
        b = e + 1;
    }
    return out;
}

static ggml_type type_of(const std::string & s) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const char * n = ggml_type_name((ggml_type) i);
        if (n && s == n) return (ggml_type) i;
    }
    fprintf(stderr, "unknown type %s\n", s.c_str());
    exit(1);
}

// Fill a tensor with pseudo-random data of its own type. Only one block of about 256k elements is generated
// (and quantized) on the CPU; it is tiled across the tensor, so a 1.3-billion-element lm_head takes well under
// a second instead of tens of seconds of single-core work that leaves the GPU idle.
static void fill_random(ggml_tensor * t, float lo, float hi, uint32_t seed) {
    const int64_t n_total = ggml_nelements(t);
    const int64_t ne0     = t->ne[0];
    const int64_t blck    = ggml_blck_size(t->type);
    int64_t nrows = std::max<int64_t>(1, std::min<int64_t>(n_total / ne0, (1 << 18) / ne0));
    const int64_t n_el = nrows * ne0;
    GGML_ASSERT(ne0 % blck == 0);
    std::vector<float> f(n_el);
    uint32_t x = seed * 2654435761u + 12345u;
    const float scale = (hi - lo) / 4294967296.0f;
    for (int64_t i = 0; i < n_el; ++i) {
        x = x * 1664525u + 1013904223u;
        f[i] = lo + x * scale;
    }
    std::vector<uint8_t> block(ggml_row_size(t->type, ne0) * nrows);
    if (t->type == GGML_TYPE_F32) {
        memcpy(block.data(), f.data(), block.size());
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) block.data(), n_el);
    } else if (t->type == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row(f.data(), (ggml_bf16_t *) block.data(), n_el);
    } else if (ggml_is_quantized(t->type)) {
        ggml_quantize_chunk(t->type, f.data(), block.data(), 0, nrows, ne0, nullptr);
    } else {
        fprintf(stderr, "cannot fill type %s\n", ggml_type_name(t->type));
        exit(1);
    }
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> buf(nbytes);
    for (size_t off = 0; off < nbytes; off += block.size()) {
        memcpy(buf.data() + off, block.data(), std::min(block.size(), nbytes - off));
    }
    ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
}

struct bench_case {
    std::string name;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph * gf = nullptr;
    double flops = 0;   // per run
    double bytes = 0;   // per run, the big operand(s) only
};

static int g_copies = 1;
// Distinct copies of a large operand so that repeated ops inside one graph do not hit the 96 MB infinity cache of
// the RX 7900 XTX (a single 50 MB weight matrix repeated 8x reads from cache after the first copy). Capped so the
// distinct set stays under 8 GiB.
static int distinct_copies(size_t bytes_per_copy) {
    // OPBENCH_SAME=1: every copy reads the same operand, so repeats after the first are served from the infinity cache
    if (getenv("OPBENCH_SAME")) {
        return 1;
    }
    const size_t cap = (size_t) 8 << 30;
    const size_t n = bytes_per_copy ? std::max<size_t>(1, cap / bytes_per_copy) : (size_t) g_copies;
    return (int) std::min<size_t>((size_t) g_copies, n);
}
static ggml_context * new_ctx() {
    ggml_init_params ip = { ggml_tensor_overhead() * (64 + 12 * (size_t) g_copies) + ggml_graph_overhead_custom(16 * g_copies + 64, false), nullptr, true };
    return ggml_init(ip);
}
static ggml_cgraph * new_graph(ggml_context * ctx) {
    return ggml_new_graph_custom(ctx, 16 * g_copies + 64, false);
}

static bench_case make_mm(const std::vector<std::string> & a) {
    bench_case c;
    const ggml_type type = type_of(a[1]);
    const int64_t M = atoll(a[2].c_str()), K = atoll(a[3].c_str()), N = atoll(a[4].c_str());
    c.name = "MUL_MAT " + a[1] + " m=" + std::to_string(M) + " k=" + std::to_string(K) + " n=" + std::to_string(N);
    c.ctx = new_ctx();
    ggml_tensor * x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, K, N);
    ggml_set_name(x, "x");
    std::vector<ggml_tensor *> ws;
    const int nd = distinct_copies(ggml_row_size(type, K) * M);
    for (int i = 0; i < nd; ++i) { ggml_tensor * w = ggml_new_tensor_2d(c.ctx, type, K, M); ggml_set_name(w, "w"); ws.push_back(w); }
    c.name += " (" + std::to_string(nd) + " distinct)";
    c.gf = new_graph(c.ctx);
    for (int i = 0; i < g_copies; ++i) {
        ggml_tensor * y = ggml_mul_mat(c.ctx, ws[i % nd], x);
        ggml_set_name(y, "y");
        ggml_build_forward_expand(c.gf, y);
    }
    c.flops = 2.0 * M * K * N;
    c.bytes = (double) ggml_nbytes(ws[0]) + ggml_nbytes(x);
    return c;
}

static bench_case make_fa(const std::vector<std::string> & a) {
    bench_case c;
    const int64_t D = atoll(a[1].c_str()), NQ = atoll(a[2].c_str()), NKV = atoll(a[3].c_str());
    const int64_t NH = atoll(a[4].c_str()), NHKV = atoll(a[5].c_str());
    const ggml_type TK = type_of(a[6]), TV = type_of(a[7]);
    c.name = "FLASH_ATTN_EXT D=" + std::to_string(D) + " n_q=" + std::to_string(NQ) + " n_kv=" + std::to_string(NKV) +
             " h=" + std::to_string(NH) + "/" + std::to_string(NHKV) + " K=" + a[6] + " V=" + a[7];
    c.ctx = new_ctx();
    // llama.cpp: q is [D, n_head, n_tokens] permuted to [D, n_tokens, n_head]
    ggml_tensor * q0 = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, D, NH, NQ);
    ggml_tensor * q  = ggml_permute(c.ctx, q0, 0, 2, 1, 3);
    ggml_tensor * m  = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F16, NKV, NQ);
    ggml_set_name(q0, "q"); ggml_set_name(m, "m");
    std::vector<ggml_tensor *> ks, vs;
    const int nd = distinct_copies(ggml_row_size(TK, D) * NKV * NHKV + ggml_row_size(TV, D) * NKV * NHKV);
    for (int i = 0; i < nd; ++i) {
        ggml_tensor * k = ggml_new_tensor_3d(c.ctx, TK, D, NKV, NHKV);
        ggml_tensor * v = ggml_new_tensor_3d(c.ctx, TV, D, NKV, NHKV);
        ggml_set_name(k, "k"); ggml_set_name(v, "v"); ks.push_back(k); vs.push_back(v);
    }
    ggml_tensor * k = ks[0];
    ggml_tensor * v = vs[0];
    c.name += " (" + std::to_string(nd) + " distinct)";
    c.gf = new_graph(c.ctx);
    for (int i = 0; i < g_copies; ++i) {
        ggml_tensor * out = ggml_flash_attn_ext(c.ctx, q, ks[i % nd], vs[i % nd], m, 1.0f / sqrtf((float) D), 0.0f, 0.0f);
        ggml_prec_set_acc(out, GGML_PREC_F32);
        ggml_set_name(out, "out");
        ggml_build_forward_expand(c.gf, out);
    }
    c.flops = 4.0 * D * NQ * NKV * NH;  // QK^T and PV
    c.bytes = (double) ggml_nbytes(k) + ggml_nbytes(v);
    return c;
}

static bench_case make_gdn(const std::vector<std::string> & a) {
    bench_case c;
    const int64_t HD = atoll(a[1].c_str()), NHK = atoll(a[2].c_str()), NHV = atoll(a[3].c_str()), NT = atoll(a[4].c_str());
    c.name = "GATED_DELTA_NET hd=" + std::to_string(HD) + " h=" + std::to_string(NHK) + "/" + std::to_string(NHV) + " n_t=" + std::to_string(NT);
    c.ctx = new_ctx();
    ggml_tensor * q = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, HD, NHK, NT, 1);
    ggml_tensor * k = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, HD, NHK, NT, 1);
    ggml_tensor * v = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, HD, NHV, NT, 1);
    ggml_tensor * g = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, 1, NHV, NT, 1);
    ggml_tensor * b = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, 1, NHV, NT, 1);
    ggml_tensor * s = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, HD, HD, NHV, 1);
    ggml_set_name(q, "q"); ggml_set_name(k, "k"); ggml_set_name(v, "v"); ggml_set_name(g, "g"); ggml_set_name(b, "b"); ggml_set_name(s, "s");
    c.gf = new_graph(c.ctx);
    for (int i = 0; i < g_copies; ++i) {
        ggml_tensor * out = ggml_gated_delta_net(c.ctx, q, k, v, g, b, s, 1);
        ggml_set_name(out, "out");
        ggml_build_forward_expand(c.gf, out);
    }
    c.flops = 0;
    c.bytes = (double) ggml_nbytes(s) * 2;
    return c;
}

static bench_case make_eltwise(const std::vector<std::string> & a) {
    bench_case c;
    c.ctx = new_ctx();
    c.gf = new_graph(c.ctx);
    const std::string op = a[0];
    if (op == "concat") {
        const int64_t NA = atoll(a[1].c_str()), NB = atoll(a[2].c_str()), N1 = atoll(a[3].c_str());
        c.name = "CONCAT f32 [" + a[1] + "," + a[3] + "]+[" + a[2] + "," + a[3] + "]";
        ggml_tensor * x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, NA, N1);
        ggml_tensor * y = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, NB, N1);
        ggml_set_name(x, "x"); ggml_set_name(y, "y");
        for (int i = 0; i < g_copies; ++i) { ggml_build_forward_expand(c.gf, ggml_concat(c.ctx, x, y, 0)); }
        c.bytes = (double) ggml_nbytes(x) + ggml_nbytes(y);
    } else {
        const int64_t N0 = atoll(a[1].c_str()), N1 = atoll(a[2].c_str());
        c.name = op + " f32 [" + a[1] + "," + a[2] + "]";
        if (op == "glu") {
            ggml_tensor * x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 2 * N0, N1);
            ggml_set_name(x, "x");
            for (int i = 0; i < g_copies; ++i) { ggml_build_forward_expand(c.gf, ggml_swiglu(c.ctx, x)); }
            c.bytes = (double) ggml_nbytes(x);
        } else if (op == "rms") {
            ggml_tensor * x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, N0, N1);
            ggml_tensor * w = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, N0);
            ggml_set_name(x, "x"); ggml_set_name(w, "w");
            for (int i = 0; i < g_copies; ++i) { ggml_build_forward_expand(c.gf, ggml_mul(c.ctx, ggml_rms_norm(c.ctx, x, 1e-6f), w)); }
            c.bytes = (double) ggml_nbytes(x);
        } else if (op == "add") {
            ggml_tensor * x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, N0, N1);
            ggml_tensor * y = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, N0, N1);
            ggml_set_name(x, "x"); ggml_set_name(y, "y");
            for (int i = 0; i < g_copies; ++i) { ggml_build_forward_expand(c.gf, ggml_add(c.ctx, x, y)); }
            c.bytes = 2.0 * ggml_nbytes(x);
        } else { // cpy
            ggml_tensor * x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, N0, N1);
            ggml_set_name(x, "x");
            for (int i = 0; i < g_copies; ++i) { ggml_tensor * y = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, N0, N1); ggml_set_name(y, "yd"); ggml_build_forward_expand(c.gf, ggml_cpy(c.ctx, x, y)); }
            c.bytes = 2.0 * ggml_nbytes(x);
        }
    }
    return c;
}

int main(int argc, char ** argv) {
    std::string backend_name = "CPU";
    int reps = 20, warm = 3, group = 10;
    static int copies = 1;
    std::vector<std::string> specs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-b" && i + 1 < argc) backend_name = argv[++i];
        else if (a == "-r" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (a == "-w" && i + 1 < argc) warm = atoi(argv[++i]);
        else if (a == "-g" && i + 1 < argc) group = std::max(1, atoi(argv[++i]));
        else if (a == "-n" && i + 1 < argc) copies = std::max(1, atoi(argv[++i]));
        else specs.push_back(a);
    }
    g_copies = copies;
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name.c_str());
    if (!dev) {
        fprintf(stderr, "backend %s not found; available:", backend_name.c_str());
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) fprintf(stderr, " %s", ggml_backend_dev_name(ggml_backend_dev_get(i)));
        fprintf(stderr, "\n");
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { fprintf(stderr, "backend init failed\n"); return 1; }
    printf("backend: %s (%s), %d op copies per graph, %d graphs per sync\n", ggml_backend_name(backend), ggml_backend_dev_description(dev), g_copies, group);

    for (const std::string & spec : specs) {
        std::vector<std::string> a = split(spec, ':');
        bench_case c;
        if (a[0] == "mm" && a.size() == 5) c = make_mm(a);
        else if (a[0] == "fa" && a.size() == 8) c = make_fa(a);
        else if (a[0] == "gdn" && a.size() == 5) c = make_gdn(a);
        else if (a[0] == "concat" && a.size() == 4) c = make_eltwise(a);
        else if ((a[0] == "glu" || a[0] == "rms" || a[0] == "add" || a[0] == "cpy") && a.size() == 3) c = make_eltwise(a);
        else { fprintf(stderr, "bad spec %s\n", spec.c_str()); return 1; }

        // supported?
        bool supported = true;
        for (int i = 0; i < ggml_graph_n_nodes(c.gf); ++i) {
            ggml_tensor * n = ggml_graph_node(c.gf, i);
            if (!ggml_backend_supports_op(backend, n)) { supported = false; printf("%s: op %s NOT SUPPORTED by backend\n", c.name.c_str(), ggml_op_desc(n)); }
        }
        if (!supported) { ggml_free(c.ctx); continue; }

        c.buf = ggml_backend_alloc_ctx_tensors(c.ctx, backend);
        if (!c.buf) { printf("%s: allocation failed\n", c.name.c_str()); ggml_free(c.ctx); continue; }
        uint32_t seed = 1;
        for (ggml_tensor * t = ggml_get_first_tensor(c.ctx); t; t = ggml_get_next_tensor(c.ctx, t)) {
            if (t->view_src || t->op != GGML_OP_NONE) continue;
            if (strcmp(t->name, "m") == 0) { std::vector<ggml_fp16_t> z(ggml_nelements(t), ggml_fp32_to_fp16(0.0f)); ggml_backend_tensor_set(t, z.data(), 0, ggml_nbytes(t)); }
            else if (strcmp(t->name, "g") == 0) fill_random(t, -0.2f, 0.0f, seed++);
            else if (strcmp(t->name, "b") == 0) fill_random(t, 0.0f, 1.0f, seed++);
            else fill_random(t, -1.0f, 1.0f, seed++);
        }
        for (int i = 0; i < warm; ++i) { ggml_backend_graph_compute(backend, c.gf); }
        ggml_backend_synchronize(backend);
        double tmin = 1e30, tsum = 0;
        for (int i = 0; i < reps; ++i) {
            const int64_t t0 = ggml_time_us();
            for (int g = 0; g < group; ++g) {
                ggml_backend_graph_compute(backend, c.gf);
            }
            ggml_backend_synchronize(backend);
            const double dt = (double) (ggml_time_us() - t0) / group / g_copies;
            tmin = std::min(tmin, dt); tsum += dt;
        }
        const double tavg = tsum / reps;
        printf("%s: min %.1f us, avg %.1f us", c.name.c_str(), tmin, tavg);
        if (c.flops > 0) printf(", %.0f GFLOP/s", c.flops / tmin / 1e3);
        if (c.bytes > 0) printf(", %.0f GB/s", c.bytes / tmin / 1e3);
        printf("\n");
        fflush(stdout);
        ggml_backend_buffer_free(c.buf);
        ggml_free(c.ctx);
    }
    ggml_backend_free(backend);
    return 0;
}
