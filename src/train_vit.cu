// =============================================================================
// MNIST Vision Transformer in CUDA, from scratch.
//
// No patching: each of the 28*28 = 784 pixels is one token. The pixel intensity
// (0..255) is the token id, looked up in a learned embedding. After L
// transformer blocks the sequence is mean-pooled and fed to a 10-way classifier.
//
// Everything is one file. Every operation is a hand-written CUDA kernel;
// cuBLAS is used only for matmul. MPI + NCCL provide data-parallel training
// across GPUs.
//
//   forward path
//      pixels [B, T]
//        --> wte[V,D] + wpe[T,D]              (encoder embedding)
//        --> for each layer l in [0, L):
//              LayerNorm
//              QKV linear   (D -> 3D)
//              self-attention (Q@K^T, softmax, *V)   -- no causal mask
//              attn projection (D -> D) + residual
//              LayerNorm
//              MLP (D -> 4D, GELU, 4D -> D) + residual
//        --> LayerNorm
//        --> mean-pool over T   [B, D]
//        --> head linear (D -> 10)
//        --> softmax cross-entropy
//
// Default config: V=256, T=784, L=2, D=64, H=4, head_dim=16, classes=10.
// =============================================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mpi.h>
#include <nccl.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <vector>

// -----------------------------------------------------------------------------
// Small helpers.
// -----------------------------------------------------------------------------

#define CHECK(call) do {                                                       \
    auto _err = (call);                                                        \
    if (_err) {                                                                \
        fprintf(stderr, "%s:%d: error %d\n", __FILE__, __LINE__, (int)_err);   \
        MPI_Abort(MPI_COMM_WORLD, 1);                                          \
    }                                                                          \
} while (0)

// Standard 1-D launch config: 256 threads per block, enough blocks for n items.
#define GRID(n) (((n) + 255) / 256), 256

// Warp-level reductions across 32 lanes.
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

// File-scope helpers populated in main():
//   g_ones        : device vector [BT] of 1.0f, used as the summation vector
//                   in the cublasSgemv path that replaced bias_grad.
//   g_attn_ptrs   : device array of float* for cublasSgemmBatched attention.
//                   Layout: 12 sections of size (L * B * H), one per matrix
//                   role (Q, K, V, S, A, O, dQ, dK, dV, dS, dA, dO).
static float  *g_ones      = NULL;
static float **g_attn_ptrs = NULL;
enum {
    PT_Q = 0, PT_K, PT_V, PT_S, PT_A, PT_O,
    PT_dQ,    PT_dK, PT_dV, PT_dS, PT_dA, PT_dO,
    PT_COUNT
};

// =============================================================================
// PERF feature flags. Each of the 4 optimizations can be toggled at runtime
// via an environment variable (no rebuild needed). 1 = new/fast, 0 = old/slow.
// Default: all fast. Set via, e.g.,
//   PERF_LN_BWD_FAST=0 PERF_SGEMV_BIAS=0 ./bin/train_vit ...
// to bench old vs new in the same binary.
//
//   PERF_LN_BWD_FAST    layernorm_backward: rows-per-block reg accumulators
//                       vs one-warp-per-row + atomicAdd-per-row.
//   PERF_SGEMV_BIAS     bias-grad via cublasSgemv vs custom serial kernel.
//   PERF_ATTN_BATCHED   attention via cublasSgemmBatched (single launch per
//                       gemm) vs the per-batch-element StridedBatched loop.
//   PERF_NARROW_MEMSET  zero only d_grads + RESID2 slice (~1 MB) vs full
//                       d_dacts buffer (~350 MB at default cfg).
// =============================================================================
static int g_use_ln_bwd_fast    = 1;
static int g_use_sgemv_bias     = 1;
static int g_use_attn_batched   = 1;
static int g_use_narrow_memset  = 1;

// =============================================================================
// Fine-grained timing constants.
//
// model_forward records FWD_NEV CUDA events (one after each kernel group).
// model_backward records BWD_NEV events.
// Event arrays are allocated in main and passed to both functions.
//
// Forward layout (FKPL = 8 groups per layer):
//   fwd_ev[0]           after encoder_forward
//   fwd_ev[1+l*8 + 0]   after LN1          (layer l)
//   fwd_ev[1+l*8 + 1]   after QKV matmul
//   fwd_ev[1+l*8 + 2]   after attn (QK + softmax + AV)
//   fwd_ev[1+l*8 + 3]   after attn-proj + residual1
//   fwd_ev[1+l*8 + 4]   after LN2
//   fwd_ev[1+l*8 + 5]   after FC1 matmul
//   fwd_ev[1+l*8 + 6]   after GELU
//   fwd_ev[1+l*8 + 7]   after FC2 matmul + residual2
//   fwd_ev[1+L*8 + 0]   after final LNF
//   fwd_ev[1+L*8 + 1]   after mean-pool
//   fwd_ev[1+L*8 + 2]   after head matmul   (= last fwd event)
//   (softmax_ce_forward is launched in main; ev_fwd records its end)
//
// Backward layout (BKPL = 8 groups per layer, reverse order i=L-1-l):
//   bwd_ev[0]           after softmax_ce_backward
//   bwd_ev[1]           after head matmul backward
//   bwd_ev[2]           after mean-pool backward
//   bwd_ev[3]           after final LNF backward
//   bwd_ev[4+i*8 + 0]   after D2D-memcpy + FC2 backward   (iter i)
//   bwd_ev[4+i*8 + 1]   after GELU backward
//   bwd_ev[4+i*8 + 2]   after FC1 backward
//   bwd_ev[4+i*8 + 3]   after LN2 backward
//   bwd_ev[4+i*8 + 4]   after D2D-memcpy + attn-proj backward
//   bwd_ev[4+i*8 + 5]   after attention backward (dV+dA+dS+dQ+dK)
//   bwd_ev[4+i*8 + 6]   after QKV backward
//   bwd_ev[4+i*8 + 7]   after LN1 backward
//   bwd_ev[4+L*8]       after encoder backward
// =============================================================================

#define MAX_LAYERS 8
#define FKPL       8
#define BKPL       8
#define FWD_NEV  (1 + MAX_LAYERS * FKPL + 3)
#define BWD_NEV  (4 + MAX_LAYERS * BKPL + 1)

// Model hyper-parameters.
struct Cfg {
    int vocab;     // V
    int seq;       // T
    int layers;    // L
    int dim;       // D
    int heads;     // H
    int head_dim;  // D / H
    int classes;   // C
};

// =============================================================================
// SECTION 1 — parameter and activation catalogs.
// =============================================================================

enum {
    P_TOK_EMB,
    P_POS_EMB,
    P_LN1_W, P_LN1_B,
    P_QKV_W, P_QKV_B,
    P_ATTPROJ_W, P_ATTPROJ_B,
    P_LN2_W, P_LN2_B,
    P_FC1_W, P_FC1_B,
    P_FC2_W, P_FC2_B,
    P_LNF_W, P_LNF_B,
    P_HEAD,
    NUM_PARAMS
};

enum {
    A_ENCODED,
    A_LNF, A_LNF_MEAN, A_LNF_RSTD,
    A_POOLED,
    A_LOGITS,
    A_PROBS,
    A_LOSSES,
    A_LN1, A_LN1_MEAN, A_LN1_RSTD,
    A_QKV,
    A_ATTN_PRE,
    A_ATTN,
    A_ATTN_OUT,
    A_ATTPROJ,
    A_RESID1,
    A_LN2, A_LN2_MEAN, A_LN2_RSTD,
    A_FC1,
    A_FC1_GELU,
    A_FC2,
    A_RESID2,
    NUM_ACTS
};

static void fill_param_sizes(size_t *sz, Cfg c)
{
    int V = c.vocab, T = c.seq, L = c.layers, D = c.dim, C = c.classes;
    sz[P_TOK_EMB]    = V * D;
    sz[P_POS_EMB]    = T * D;
    sz[P_LN1_W]      = L * D;        sz[P_LN1_B]      = L * D;
    sz[P_QKV_W]      = L * 3*D * D;  sz[P_QKV_B]      = L * 3*D;
    sz[P_ATTPROJ_W]  = L * D * D;    sz[P_ATTPROJ_B]  = L * D;
    sz[P_LN2_W]      = L * D;        sz[P_LN2_B]      = L * D;
    sz[P_FC1_W]      = L * 4*D * D;  sz[P_FC1_B]      = L * 4*D;
    sz[P_FC2_W]      = L * D * 4*D;  sz[P_FC2_B]      = L * D;
    sz[P_LNF_W]      = D;            sz[P_LNF_B]      = D;
    sz[P_HEAD]       = C * D;
}

static void fill_act_sizes(size_t *sz, Cfg c, int B)
{
    int T = c.seq, L = c.layers, D = c.dim, H = c.heads, C = c.classes;
    sz[A_ENCODED]   = B * T * D;
    sz[A_LNF]       = B * T * D;
    sz[A_LNF_MEAN]  = B * T;
    sz[A_LNF_RSTD]  = B * T;
    sz[A_POOLED]    = B * D;
    sz[A_LOGITS]    = B * C;
    sz[A_PROBS]     = B * C;
    sz[A_LOSSES]    = B;
    sz[A_LN1]       = L * B * T * D;
    sz[A_LN1_MEAN]  = L * B * T;
    sz[A_LN1_RSTD]  = L * B * T;
    sz[A_QKV]       = L * B * T * 3*D;
    sz[A_ATTN_PRE]  = L * B * H * T * T;
    sz[A_ATTN]      = L * B * H * T * T;
    sz[A_ATTN_OUT]  = L * B * T * D;
    sz[A_ATTPROJ]   = L * B * T * D;
    sz[A_RESID1]    = L * B * T * D;
    sz[A_LN2]       = L * B * T * D;
    sz[A_LN2_MEAN]  = L * B * T;
    sz[A_LN2_RSTD]  = L * B * T;
    sz[A_FC1]       = L * B * T * 4*D;
    sz[A_FC1_GELU]  = L * B * T * 4*D;
    sz[A_FC2]       = L * B * T * D;
    sz[A_RESID2]    = L * B * T * D;
}

static void assign_pointers(float **ptrs, float *base, const size_t *sz, int n)
{
    float *cur = base;
    for (int i = 0; i < n; i++) { ptrs[i] = cur; cur += sz[i]; }
}

// =============================================================================
// SECTION 2 — kernels.
// =============================================================================

__global__ void encoder_forward(
    float *out, const int *pixel, const float *wte, const float *wpe,
    int B, int T, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * T * D) return;
    int b = i / (T * D);
    int t = (i / D) % T;
    int d = i % D;
    int tok = pixel[b * T + t];
    out[i] = wte[tok * D + d] + wpe[t * D + d];
}

__global__ void encoder_backward(
    float *dwte, float *dwpe, const float *dout, const int *pixel,
    int B, int T, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * T * D) return;
    int b = i / (T * D);
    int t = (i / D) % T;
    int d = i % D;
    int tok = pixel[b * T + t];
    float g = dout[i];
    atomicAdd(&dwte[tok * D + d], g);
    atomicAdd(&dwpe[t   * D + d], g);
}

__global__ void layernorm_forward(
    float *out, float *mean_out, float *rstd_out,
    const float *x, const float *gamma, const float *beta,
    int N, int D)
{
    int n = blockIdx.x;
    int tid = threadIdx.x;
    const float *row_x = x + (size_t)n * D;
    float *row_y = out + (size_t)n * D;

    float sum = 0.0f;
    for (int d = tid; d < D; d += 32) sum += row_x[d];
    float mean = warp_sum(sum) / D;

    float var = 0.0f;
    for (int d = tid; d < D; d += 32) {
        float diff = row_x[d] - mean;
        var += diff * diff;
    }
    float rstd = rsqrtf(warp_sum(var) / D + 1e-5f);

    if (tid == 0) { mean_out[n] = mean; rstd_out[n] = rstd; }

    for (int d = tid; d < D; d += 32)
        row_y[d] = ((row_x[d] - mean) * rstd) * gamma[d] + beta[d];
}

// Old/slow LayerNorm backward: one warp per row, one atomicAdd per (row, d)
// into the 64-slot dgamma/dbeta arrays. Kept as the toggle-off path for
// PERF_LN_BWD_FAST=0 benchmarking.
__global__ void layernorm_backward_slow(
    float *dx, float *dgamma, float *dbeta,
    const float *dy, const float *x, const float *gamma,
    const float *mean_buf, const float *rstd_buf,
    int N, int D)
{
    int n = blockIdx.x;
    int tid = threadIdx.x;
    const float *row_x  = x  + (size_t)n * D;
    const float *row_dy = dy + (size_t)n * D;
    float       *row_dx = dx + (size_t)n * D;
    float mean = mean_buf[n];
    float rstd = rstd_buf[n];

    float S1 = 0.0f, S2 = 0.0f;
    for (int d = tid; d < D; d += 32) {
        float dx_hat = row_dy[d] * gamma[d];
        float xc     = row_x[d] - mean;
        S1 += dx_hat;
        S2 += dx_hat * xc;
    }
    S1 = warp_sum(S1);
    S2 = warp_sum(S2);

    for (int d = tid; d < D; d += 32) {
        float dx_hat = row_dy[d] * gamma[d];
        float xc     = row_x[d] - mean;
        float dxd    = rstd * (dx_hat - (S1 + rstd * rstd * xc * S2) / (float)D);
        row_dx[d] += dxd;
    }
    for (int d = tid; d < D; d += 32) {
        atomicAdd(&dgamma[d], row_dy[d] * (row_x[d] - mean) * rstd);
        atomicAdd(&dbeta[d],  row_dy[d]);
    }
}

// One warp per block, but each block processes `rows_per_block` rows and keeps
// per-thread register accumulators for dgamma/dbeta. At the end of the block,
// we flush exactly ONE atomicAdd per (block, d) instead of one per (row, d).
// For BT=6272 rows with rows_per_block=32 that cuts atomic contention on the
// 64-slot dgamma/dbeta arrays from 6272 to 196 atomics per slot (~32x).
//
// `dx` is still accumulated (+=) because it feeds the residual path that was
// prefilled by an upstream cudaMemcpyAsync. D is assumed <= 256 (8 register
// slots per thread with 32 threads).
#define LN_ACC_MAX 8

__global__ void layernorm_backward(
    float *dx, float *dgamma, float *dbeta,
    const float *dy, const float *x, const float *gamma,
    const float *mean_buf, const float *rstd_buf,
    int N, int D, int rows_per_block)
{
    int tid = threadIdx.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end   = row_start + rows_per_block;
    if (row_end > N) row_end = N;

    float dg_acc[LN_ACC_MAX] = {0,0,0,0,0,0,0,0};
    float db_acc[LN_ACC_MAX] = {0,0,0,0,0,0,0,0};

    for (int n = row_start; n < row_end; n++) {
        const float *row_x  = x  + (size_t)n * D;
        const float *row_dy = dy + (size_t)n * D;
        float       *row_dx = dx + (size_t)n * D;
        float mean = mean_buf[n];
        float rstd = rstd_buf[n];

        float S1 = 0.0f, S2 = 0.0f;
        for (int d = tid; d < D; d += 32) {
            float dx_hat = row_dy[d] * gamma[d];
            float xc     = row_x[d] - mean;
            S1 += dx_hat;
            S2 += dx_hat * xc;
        }
        S1 = warp_sum(S1);
        S2 = warp_sum(S2);

        int slot = 0;
        for (int d = tid; d < D; d += 32) {
            float dx_hat = row_dy[d] * gamma[d];
            float xc     = row_x[d] - mean;
            float dxd    = rstd * (dx_hat - (S1 + rstd * rstd * xc * S2) / (float)D);
            row_dx[d] += dxd;
            dg_acc[slot] += row_dy[d] * xc * rstd;
            db_acc[slot] += row_dy[d];
            slot++;
        }
    }

    int slot = 0;
    for (int d = tid; d < D; d += 32) {
        atomicAdd(&dgamma[d], dg_acc[slot]);
        atomicAdd(&dbeta[d],  db_acc[slot]);
        slot++;
    }
}

#define LN_ROWS_PER_BLOCK 32
#define LN_BWD_GRID(N) (((N) + LN_ROWS_PER_BLOCK - 1) / LN_ROWS_PER_BLOCK)

__global__ void bias_add(float *y, const float *b, int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N * K) y[i] += b[i % K];
}
// Old/slow bias_grad: one thread per output column, serial reduction over N.
// Used when PERF_SGEMV_BIAS=0. Otherwise matmul_backward calls cublasSgemv.
__global__ void bias_grad(float *db, const float *dy, int N, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    float s = 0.0f;
    for (int n = 0; n < N; n++) s += dy[(size_t)n * K + k];
    db[k] = s;
}

__global__ void attention_softmax(
    float *attn, const float *scores, int B, int H, int T)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *score_row = scores + (size_t)row * T;
    float       *attn_row  = attn   + (size_t)row * T;

    float mx = -INFINITY;
    for (int t = tid; t < T; t += 32) mx = fmaxf(mx, score_row[t]);
    mx = warp_max(mx);

    float sum = 0.0f;
    for (int t = tid; t < T; t += 32) sum += expf(score_row[t] - mx);
    sum = warp_sum(sum);
    float inv = 1.0f / sum;

    for (int t = tid; t < T; t += 32)
        attn_row[t] = expf(score_row[t] - mx) * inv;
}

__global__ void attention_d_softmax(
    float *d_scores, const float *d_attn, const float *attn,
    int B, int H, int T)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *attn_row   = attn    + (size_t)row * T;
    const float *d_attn_row = d_attn  + (size_t)row * T;
    float       *out_row    = d_scores + (size_t)row * T;

    float s = 0.0f;
    for (int t = tid; t < T; t += 32) s += attn_row[t] * d_attn_row[t];
    s = warp_sum(s);

    for (int t = tid; t < T; t += 32)
        out_row[t] = attn_row[t] * (d_attn_row[t] - s);
}



#define GELU_K 0.7978845608f

__global__ void gelu_forward(float *out, const float *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = x[i];
    out[i] = 0.5f * xi * (1.0f + tanhf(GELU_K * (xi + 0.044715f * xi * xi * xi)));
}
__global__ void gelu_backward(float *dx, const float *dy, const float *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi    = x[i];
    float arg   = GELU_K * (xi + 0.044715f * xi * xi * xi);
    float th    = tanhf(arg);
    float sech2 = 1.0f - th * th;
    float darg  = GELU_K * (1.0f + 3.0f * 0.044715f * xi * xi);
    dx[i] = (0.5f * (1.0f + th) + 0.5f * xi * sech2 * darg) * dy[i];
}

__global__ void residual_add(float *out, const float *a, const float *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void mean_pool_forward(float *out, const float *x, int B, int T, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * D) return;
    int b = i / D;
    int d = i % D;
    float sum = 0.0f;
    for (int t = 0; t < T; t++) sum += x[((size_t)b * T + t) * D + d];
    out[i] = sum / (float)T;
}
__global__ void mean_pool_backward(float *dx, const float *dy, int B, int T, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * T * D) return;
    int b = i / (T * D);
    int d = i % D;
    dx[i] = dy[b * D + d] / (float)T;
}

__global__ void softmax_ce_forward(
    float *probs, float *losses,
    const float *logits, const int *target,
    int N, int C)
{
    int n = blockIdx.x;
    int tid = threadIdx.x;
    const float *row_logit = logits + (size_t)n * C;
    float       *row_prob  = probs  + (size_t)n * C;
    int y = target[n];

    float mx = -INFINITY;
    for (int i = tid; i < C; i += 32) mx = fmaxf(mx, row_logit[i]);
    mx = warp_max(mx);

    float sum = 0.0f;
    for (int i = tid; i < C; i += 32) sum += expf(row_logit[i] - mx);
    sum = warp_sum(sum);
    float inv = 1.0f / sum;

    for (int i = tid; i < C; i += 32) row_prob[i] = expf(row_logit[i] - mx) * inv;
    if (tid == 0) losses[n] = (mx + logf(sum)) - row_logit[y];
}

__global__ void softmax_ce_backward(
    float *d_logits, const float *probs, const int *target, int N, int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * C) return;
    int n = i / C;
    int c = i % C;
    d_logits[i] = probs[i] - ((c == target[n]) ? 1.0f : 0.0f);
}

__global__ void adam_step(
    float *w, const float *g, float *m, float *v,
    int n, float lr, float beta1, float beta2, float eps,
    float scale, float bc1, float bc2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i] * scale;
    float mi = beta1 * m[i] + (1.0f - beta1) * gi;
    float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    w[i] -= lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
}

__global__ void atomic_sum(float *out, const float *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, x[i]);
}
__global__ void count_correct(
    int *out, const float *logits, const int *target, int B, int C)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const float *row = logits + (size_t)b * C;
    int best = 0;
    float mx = row[0];
    for (int i = 1; i < C; i++) if (row[i] > mx) { mx = row[i]; best = i; }
    if (best == target[b]) atomicAdd(out, 1);
}

// =============================================================================
// SECTION 3 — cuBLAS matmul wrappers.
// =============================================================================

static void matmul_forward(
    float *y, const float *x, const float *w, const float *b,
    int N, int C, int OC, cublasHandle_t cublas, cudaStream_t stream)
{
    float alpha = 1.0f, beta = 0.0f;
    CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                      OC, N, C,
                      &alpha, w, C, x, C,
                      &beta,  y, OC));
    if (b) bias_add<<<GRID(N * OC), 0, stream>>>(y, b, N, OC);
}

static void matmul_backward(
    float *dx, float *dw, float *db,
    const float *dy, const float *x, const float *w,
    int N, int C, int OC, cublasHandle_t cublas, cudaStream_t stream)
{
    float alpha = 1.0f, beta = 0.0f;

    CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                      C, OC, N,
                      &alpha, x, C, dy, OC,
                      &beta,  dw, C));

    if (db) {
        if (g_use_sgemv_bias) {
            // FAST: bias_grad via cublasSgemv. dy is logically [N, OC] row-major;
            // cuBLAS sees it as a column-major [OC, N] matrix with lda=OC, so
            // db = dy_cublas @ ones reduces along N. Coalesced tiled GEMV.
            CHECK(cublasSgemv(cublas, CUBLAS_OP_N, OC, N,
                              &alpha, dy, OC, g_ones, 1,
                              &beta,  db, 1));
        } else {
            // SLOW: 1-thread-per-column serial reduction (original kernel).
            bias_grad<<<GRID(OC), 0, stream>>>(db, dy, N, OC);
        }
    }

    CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                      C, N, OC,
                      &alpha, w, C, dy, OC,
                      &beta,  dx, C));
}

// =============================================================================
// SECTION 4 — model forward and backward.
//
// fwd_ev / bwd_ev: if non-NULL, CUDA events are recorded after each kernel
// group for fine-grained timing (see layout at top of file).
// =============================================================================

static void model_forward(
    float **params, float **acts, const int *pixel,
    Cfg c, int B,
    cublasHandle_t cublas, cudaStream_t stream,
    cudaEvent_t *fwd_ev)
{
    int T = c.seq, L = c.layers, D = c.dim, H = c.heads, HD = c.head_dim, C = c.classes;
    int BT       = B * T;
    int BTD      = B * T * D;
    int BT3D     = B * T * 3 * D;
    int BT4D     = B * T * 4 * D;
    int BHTT     = B * H * T * T;

    encoder_forward<<<GRID(BTD), 0, stream>>>(
        acts[A_ENCODED], pixel, params[P_TOK_EMB], params[P_POS_EMB], B, T, D);
    if (fwd_ev) cudaEventRecord(fwd_ev[0], stream);  // 0: enc

    const float *residual_in = acts[A_ENCODED];

    for (int l = 0; l < L; l++) {
        int base = 1 + l * FKPL;

        // --- attention sub-block ---
        layernorm_forward<<<BT, 32, 0, stream>>>(
            acts[A_LN1] + l*BTD, acts[A_LN1_MEAN] + l*BT, acts[A_LN1_RSTD] + l*BT,
            residual_in, params[P_LN1_W] + l*D, params[P_LN1_B] + l*D, BT, D);
        if (fwd_ev) cudaEventRecord(fwd_ev[base + 0], stream);  // ln1

        matmul_forward(
            acts[A_QKV] + l*BT3D, acts[A_LN1] + l*BTD,
            params[P_QKV_W] + l*3*D*D, params[P_QKV_B] + l*3*D,
            BT, D, 3*D, cublas, stream);
        if (fwd_ev) cudaEventRecord(fwd_ev[base + 1], stream);  // qkv

        // Attention: Q@K^T -> softmax -> Attn@V. Two code paths.
        {
            float alpha_qk = 1.0f / sqrtf((float)HD), beta0 = 0.0f, alpha1 = 1.0f;
            float *qkv_l = acts[A_QKV]      + (size_t)l * BT3D;
            float *pre_l = acts[A_ATTN_PRE] + (size_t)l * BHTT;
            float *attn_l= acts[A_ATTN]     + (size_t)l * BHTT;
            float *out_l = acts[A_ATTN_OUT] + (size_t)l * BTD;
            if (g_use_attn_batched) {
                // FAST: cublasSgemmBatched with precomputed pointer arrays
                // (single launch per gemm, batchCount = B*H).
                int LBH = c.layers * B * H, BH = B * H;
                CHECK(cublasSgemmBatched(cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N, T, T, HD, &alpha_qk,
                    (const float* const*)(g_attn_ptrs + PT_K * LBH + l * BH), 3*D,
                    (const float* const*)(g_attn_ptrs + PT_Q * LBH + l * BH), 3*D,
                    &beta0,
                    g_attn_ptrs + PT_S * LBH + l * BH, T,
                    BH));
                attention_softmax<<<B*H*T, 32, 0, stream>>>(attn_l, pre_l, B, H, T);
                CHECK(cublasSgemmBatched(cublas,
                    CUBLAS_OP_N, CUBLAS_OP_N, HD, T, T, &alpha1,
                    (const float* const*)(g_attn_ptrs + PT_V * LBH + l * BH), 3*D,
                    (const float* const*)(g_attn_ptrs + PT_A * LBH + l * BH), T,
                    &beta0,
                    g_attn_ptrs + PT_O * LBH + l * BH, D,
                    BH));
            } else {
                // SLOW: per-batch-element StridedBatched loop (original).
                for (int b = 0; b < B; b++) {
                    const float *Q_b = qkv_l + (size_t)b * T * 3 * D;
                    const float *K_b = Q_b + D;
                    float       *S_b = pre_l + (size_t)b * H * T * T;
                    CHECK(cublasSgemmStridedBatched(cublas,
                        CUBLAS_OP_T, CUBLAS_OP_N, T, T, HD, &alpha_qk,
                        K_b, 3*D, HD, Q_b, 3*D, HD,
                        &beta0, S_b, T, (long long)T * T, H));
                }
                attention_softmax<<<B*H*T, 32, 0, stream>>>(attn_l, pre_l, B, H, T);
                for (int b = 0; b < B; b++) {
                    const float *V_b    = qkv_l  + (size_t)b * T * 3 * D + 2 * D;
                    const float *attn_b = attn_l + (size_t)b * H * T * T;
                    float       *out_b  = out_l  + (size_t)b * T * D;
                    CHECK(cublasSgemmStridedBatched(cublas,
                        CUBLAS_OP_N, CUBLAS_OP_N, HD, T, T, &alpha1,
                        V_b, 3*D, HD, attn_b, T, (long long)T * T,
                        &beta0, out_b, D, HD, H));
                }
            }
        }
        if (fwd_ev) cudaEventRecord(fwd_ev[base + 2], stream);  // attn (qk+soft+av)

        matmul_forward(
            acts[A_ATTPROJ] + l*BTD, acts[A_ATTN_OUT] + l*BTD,
            params[P_ATTPROJ_W] + l*D*D, params[P_ATTPROJ_B] + l*D,
            BT, D, D, cublas, stream);
        residual_add<<<GRID(BTD), 0, stream>>>(
            acts[A_RESID1] + l*BTD, residual_in, acts[A_ATTPROJ] + l*BTD, BTD);
        if (fwd_ev) cudaEventRecord(fwd_ev[base + 3], stream);  // aproj + res1

        // --- MLP sub-block ---
        layernorm_forward<<<BT, 32, 0, stream>>>(
            acts[A_LN2] + l*BTD, acts[A_LN2_MEAN] + l*BT, acts[A_LN2_RSTD] + l*BT,
            acts[A_RESID1] + l*BTD, params[P_LN2_W] + l*D, params[P_LN2_B] + l*D, BT, D);
        if (fwd_ev) cudaEventRecord(fwd_ev[base + 4], stream);  // ln2

        matmul_forward(
            acts[A_FC1] + l*BT4D, acts[A_LN2] + l*BTD,
            params[P_FC1_W] + l*4*D*D, params[P_FC1_B] + l*4*D,
            BT, D, 4*D, cublas, stream);
        if (fwd_ev) cudaEventRecord(fwd_ev[base + 5], stream);  // fc1

        gelu_forward<<<GRID(BT4D), 0, stream>>>(
            acts[A_FC1_GELU] + l*BT4D, acts[A_FC1] + l*BT4D, BT4D);
        if (fwd_ev) cudaEventRecord(fwd_ev[base + 6], stream);  // gelu

        matmul_forward(
            acts[A_FC2] + l*BTD, acts[A_FC1_GELU] + l*BT4D,
            params[P_FC2_W] + l*D*4*D, params[P_FC2_B] + l*D,
            BT, 4*D, D, cublas, stream);
        residual_add<<<GRID(BTD), 0, stream>>>(
            acts[A_RESID2] + l*BTD, acts[A_RESID1] + l*BTD, acts[A_FC2] + l*BTD, BTD);
        if (fwd_ev) cudaEventRecord(fwd_ev[base + 7], stream);  // fc2 + res2

        residual_in = acts[A_RESID2] + l*BTD;
    }

    int fbase = 1 + L * FKPL;

    layernorm_forward<<<BT, 32, 0, stream>>>(
        acts[A_LNF], acts[A_LNF_MEAN], acts[A_LNF_RSTD],
        residual_in, params[P_LNF_W], params[P_LNF_B], BT, D);
    if (fwd_ev) cudaEventRecord(fwd_ev[fbase + 0], stream);  // lnf

    mean_pool_forward<<<GRID(B * D), 0, stream>>>(
        acts[A_POOLED], acts[A_LNF], B, T, D);
    if (fwd_ev) cudaEventRecord(fwd_ev[fbase + 1], stream);  // pool

    matmul_forward(
        acts[A_LOGITS], acts[A_POOLED], params[P_HEAD], NULL,
        B, D, C, cublas, stream);
    if (fwd_ev) cudaEventRecord(fwd_ev[fbase + 2], stream);  // head
}

static void model_backward(
    float **params, float **grads, float **acts, float **dacts,
    const int *pixel, const int *target,
    Cfg c, int B,
    cublasHandle_t cublas, cudaStream_t stream,
    cudaEvent_t *bwd_ev)
{
    int T = c.seq, L = c.layers, D = c.dim, H = c.heads, HD = c.head_dim, C = c.classes;
    int BT       = B * T;
    int BTD      = B * T * D;
    int BT3D     = B * T * 3 * D;
    int BT4D     = B * T * 4 * D;
    int BHTT     = B * H * T * T;

    softmax_ce_backward<<<GRID(B * C), 0, stream>>>(
        dacts[A_LOGITS], acts[A_PROBS], target, B, C);
    if (bwd_ev) cudaEventRecord(bwd_ev[0], stream);  // 0: loss_bwd

    matmul_backward(
        dacts[A_POOLED], grads[P_HEAD], NULL,
        dacts[A_LOGITS], acts[A_POOLED], params[P_HEAD],
        B, D, C, cublas, stream);
    if (bwd_ev) cudaEventRecord(bwd_ev[1], stream);  // 1: head_bwd

    mean_pool_backward<<<GRID(BTD), 0, stream>>>(dacts[A_LNF], dacts[A_POOLED], B, T, D);
    if (bwd_ev) cudaEventRecord(bwd_ev[2], stream);  // 2: pool_bwd

    if (g_use_ln_bwd_fast) {
        layernorm_backward<<<LN_BWD_GRID(BT), 32, 0, stream>>>(
            dacts[A_RESID2] + (L-1)*BTD,
            grads[P_LNF_W], grads[P_LNF_B], dacts[A_LNF],
            acts[A_RESID2] + (L-1)*BTD, params[P_LNF_W],
            acts[A_LNF_MEAN], acts[A_LNF_RSTD], BT, D, LN_ROWS_PER_BLOCK);
    } else {
        layernorm_backward_slow<<<BT, 32, 0, stream>>>(
            dacts[A_RESID2] + (L-1)*BTD,
            grads[P_LNF_W], grads[P_LNF_B], dacts[A_LNF],
            acts[A_RESID2] + (L-1)*BTD, params[P_LNF_W],
            acts[A_LNF_MEAN], acts[A_LNF_RSTD], BT, D);
    }
    if (bwd_ev) cudaEventRecord(bwd_ev[3], stream);  // 3: lnf_bwd

    for (int l = L - 1; l >= 0; l--) {
        int i    = L - 1 - l;       // iteration index (0, 1, ...)
        int base = 4 + i * BKPL;

        const float *upstream   = (l == 0) ? acts[A_ENCODED]   : acts[A_RESID2]  + (l-1)*BTD;
        float       *d_upstream = (l == 0) ? dacts[A_ENCODED]  : dacts[A_RESID2] + (l-1)*BTD;

        // Skip path: copy resid2 gradient into resid1 before MLP backward.
        CHECK(cudaMemcpyAsync(
            dacts[A_RESID1] + l*BTD, dacts[A_RESID2] + l*BTD,
            BTD * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        // --- MLP backward ---
        matmul_backward(
            dacts[A_FC1_GELU] + l*BT4D, grads[P_FC2_W] + l*D*4*D, grads[P_FC2_B] + l*D,
            dacts[A_RESID2] + l*BTD, acts[A_FC1_GELU] + l*BT4D,
            params[P_FC2_W] + l*D*4*D,
            BT, 4*D, D, cublas, stream);
        if (bwd_ev) cudaEventRecord(bwd_ev[base + 0], stream);  // memcpy + fc2

        gelu_backward<<<GRID(BT4D), 0, stream>>>(
            dacts[A_FC1] + l*BT4D, dacts[A_FC1_GELU] + l*BT4D,
            acts[A_FC1] + l*BT4D, BT4D);
        if (bwd_ev) cudaEventRecord(bwd_ev[base + 1], stream);  // gelu

        matmul_backward(
            dacts[A_LN2] + l*BTD, grads[P_FC1_W] + l*4*D*D, grads[P_FC1_B] + l*4*D,
            dacts[A_FC1] + l*BT4D, acts[A_LN2] + l*BTD, params[P_FC1_W] + l*4*D*D,
            BT, D, 4*D, cublas, stream);
        if (bwd_ev) cudaEventRecord(bwd_ev[base + 2], stream);  // fc1

        if (g_use_ln_bwd_fast) {
            layernorm_backward<<<LN_BWD_GRID(BT), 32, 0, stream>>>(
                dacts[A_RESID1] + l*BTD,
                grads[P_LN2_W] + l*D, grads[P_LN2_B] + l*D, dacts[A_LN2] + l*BTD,
                acts[A_RESID1] + l*BTD, params[P_LN2_W] + l*D,
                acts[A_LN2_MEAN] + l*BT, acts[A_LN2_RSTD] + l*BT, BT, D, LN_ROWS_PER_BLOCK);
        } else {
            layernorm_backward_slow<<<BT, 32, 0, stream>>>(
                dacts[A_RESID1] + l*BTD,
                grads[P_LN2_W] + l*D, grads[P_LN2_B] + l*D, dacts[A_LN2] + l*BTD,
                acts[A_RESID1] + l*BTD, params[P_LN2_W] + l*D,
                acts[A_LN2_MEAN] + l*BT, acts[A_LN2_RSTD] + l*BT, BT, D);
        }
        if (bwd_ev) cudaEventRecord(bwd_ev[base + 3], stream);  // ln2

        // Skip path: copy resid1 gradient to upstream before attention backward.
        CHECK(cudaMemcpyAsync(
            d_upstream, dacts[A_RESID1] + l*BTD,
            BTD * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        // --- attention backward ---
        matmul_backward(
            dacts[A_ATTN_OUT] + l*BTD,
            grads[P_ATTPROJ_W] + l*D*D, grads[P_ATTPROJ_B] + l*D,
            dacts[A_RESID1] + l*BTD, acts[A_ATTN_OUT] + l*BTD,
            params[P_ATTPROJ_W] + l*D*D,
            BT, D, D, cublas, stream);
        if (bwd_ev) cudaEventRecord(bwd_ev[base + 4], stream);  // memcpy + aproj

        // Attention backward: dV, dAttn, d_softmax, dQ, dK. Two code paths.
        {
            float alpha1 = 1.0f, beta0 = 0.0f;
            float alpha_s = 1.0f / sqrtf((float)HD);
            const float *qkv_l  = acts[A_QKV]       + (size_t)l * BT3D;
            float       *dqkv_l = dacts[A_QKV]      + (size_t)l * BT3D;
            const float *attn_l = acts[A_ATTN]      + (size_t)l * BHTT;
            float       *dout_l = dacts[A_ATTN_OUT] + (size_t)l * BTD;
            float       *dattn_l= dacts[A_ATTN]     + (size_t)l * BHTT;
            float       *dpre_l = dacts[A_ATTN_PRE] + (size_t)l * BHTT;
            if (g_use_attn_batched) {
                int LBH = c.layers * B * H, BH = B * H;
                // dV = Attn^T @ dO
                CHECK(cublasSgemmBatched(cublas,
                    CUBLAS_OP_N, CUBLAS_OP_T, HD, T, T, &alpha1,
                    (const float* const*)(g_attn_ptrs + PT_dO * LBH + l * BH), D,
                    (const float* const*)(g_attn_ptrs + PT_A  * LBH + l * BH), T,
                    &beta0,
                    g_attn_ptrs + PT_dV * LBH + l * BH, 3*D,
                    BH));
                // dAttn = dO @ V^T
                CHECK(cublasSgemmBatched(cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N, T, T, HD, &alpha1,
                    (const float* const*)(g_attn_ptrs + PT_V  * LBH + l * BH), 3*D,
                    (const float* const*)(g_attn_ptrs + PT_dO * LBH + l * BH), D,
                    &beta0,
                    g_attn_ptrs + PT_dA * LBH + l * BH, T,
                    BH));
                attention_d_softmax<<<B*H*T, 32, 0, stream>>>(dpre_l, dattn_l, attn_l, B, H, T);
                // dQ = d_scores @ K * scale
                CHECK(cublasSgemmBatched(cublas,
                    CUBLAS_OP_N, CUBLAS_OP_N, HD, T, T, &alpha_s,
                    (const float* const*)(g_attn_ptrs + PT_K  * LBH + l * BH), 3*D,
                    (const float* const*)(g_attn_ptrs + PT_dS * LBH + l * BH), T,
                    &beta0,
                    g_attn_ptrs + PT_dQ * LBH + l * BH, 3*D,
                    BH));
                // dK = d_scores^T @ Q * scale
                CHECK(cublasSgemmBatched(cublas,
                    CUBLAS_OP_N, CUBLAS_OP_T, HD, T, T, &alpha_s,
                    (const float* const*)(g_attn_ptrs + PT_Q  * LBH + l * BH), 3*D,
                    (const float* const*)(g_attn_ptrs + PT_dS * LBH + l * BH), T,
                    &beta0,
                    g_attn_ptrs + PT_dK * LBH + l * BH, 3*D,
                    BH));
            } else {
                // SLOW: per-batch-element StridedBatched loops (original).
                for (int b = 0; b < B; b++) {
                    const float *Q_b    = qkv_l  + (size_t)b * T * 3 * D;
                    const float *K_b    = Q_b + D;
                    const float *V_b    = Q_b + 2 * D;
                    float       *dQ_b   = dqkv_l + (size_t)b * T * 3 * D;
                    float       *dK_b   = dQ_b + D;
                    float       *dV_b   = dQ_b + 2 * D;
                    const float *attn_b = attn_l  + (size_t)b * H * T * T;
                    float       *dO_b   = dout_l  + (size_t)b * T * D;
                    float       *dattn_b= dattn_l + (size_t)b * H * T * T;
                    CHECK(cublasSgemmStridedBatched(cublas,
                        CUBLAS_OP_N, CUBLAS_OP_T, HD, T, T, &alpha1,
                        dO_b, D, HD, attn_b, T, (long long)T * T,
                        &beta0, dV_b, 3*D, HD, H));
                    CHECK(cublasSgemmStridedBatched(cublas,
                        CUBLAS_OP_T, CUBLAS_OP_N, T, T, HD, &alpha1,
                        V_b, 3*D, HD, dO_b, D, HD,
                        &beta0, dattn_b, T, (long long)T * T, H));
                }
                attention_d_softmax<<<B*H*T, 32, 0, stream>>>(dpre_l, dattn_l, attn_l, B, H, T);
                for (int b = 0; b < B; b++) {
                    const float *Q_b  = qkv_l  + (size_t)b * T * 3 * D;
                    const float *K_b  = Q_b + D;
                    float       *dQ_b = dqkv_l + (size_t)b * T * 3 * D;
                    float       *dK_b = dQ_b + D;
                    float       *dpre_b = dpre_l + (size_t)b * H * T * T;
                    CHECK(cublasSgemmStridedBatched(cublas,
                        CUBLAS_OP_N, CUBLAS_OP_N, HD, T, T, &alpha_s,
                        K_b, 3*D, HD, dpre_b, T, (long long)T * T,
                        &beta0, dQ_b, 3*D, HD, H));
                    CHECK(cublasSgemmStridedBatched(cublas,
                        CUBLAS_OP_N, CUBLAS_OP_T, HD, T, T, &alpha_s,
                        Q_b, 3*D, HD, dpre_b, T, (long long)T * T,
                        &beta0, dK_b, 3*D, HD, H));
                }
            }
        }
        if (bwd_ev) cudaEventRecord(bwd_ev[base + 5], stream);  // attn (dV+dA+dS+dQ+dK)

        matmul_backward(
            dacts[A_LN1] + l*BTD,
            grads[P_QKV_W] + l*3*D*D, grads[P_QKV_B] + l*3*D,
            dacts[A_QKV] + l*BT3D, acts[A_LN1] + l*BTD,
            params[P_QKV_W] + l*3*D*D,
            BT, D, 3*D, cublas, stream);
        if (bwd_ev) cudaEventRecord(bwd_ev[base + 6], stream);  // qkv

        if (g_use_ln_bwd_fast) {
            layernorm_backward<<<LN_BWD_GRID(BT), 32, 0, stream>>>(
                d_upstream,
                grads[P_LN1_W] + l*D, grads[P_LN1_B] + l*D, dacts[A_LN1] + l*BTD,
                upstream, params[P_LN1_W] + l*D,
                acts[A_LN1_MEAN] + l*BT, acts[A_LN1_RSTD] + l*BT, BT, D, LN_ROWS_PER_BLOCK);
        } else {
            layernorm_backward_slow<<<BT, 32, 0, stream>>>(
                d_upstream,
                grads[P_LN1_W] + l*D, grads[P_LN1_B] + l*D, dacts[A_LN1] + l*BTD,
                upstream, params[P_LN1_W] + l*D,
                acts[A_LN1_MEAN] + l*BT, acts[A_LN1_RSTD] + l*BT, BT, D);
        }
        if (bwd_ev) cudaEventRecord(bwd_ev[base + 7], stream);  // ln1
    }

    encoder_backward<<<GRID(BTD), 0, stream>>>(
        grads[P_TOK_EMB], grads[P_POS_EMB], dacts[A_ENCODED], pixel, B, T, D);
    if (bwd_ev) cudaEventRecord(bwd_ev[4 + L * BKPL], stream);  // enc_bwd
}

// =============================================================================
// SECTION 5 — initialisation and data loading.
// =============================================================================

static void init_parameters(
    float *d_params, const size_t *sizes, size_t total, unsigned seed)
{
    enum { WEIGHT, BIAS, LN_GAMMA };
    static const int kind[NUM_PARAMS] = {
        WEIGHT, WEIGHT,
        LN_GAMMA, BIAS,
        WEIGHT, BIAS,
        WEIGHT, BIAS,
        LN_GAMMA, BIAS,
        WEIGHT, BIAS,
        WEIGHT, BIAS,
        LN_GAMMA, BIAS,
        WEIGHT
    };

    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.02f);

    std::vector<float> host(total);
    size_t off = 0;
    for (int i = 0; i < NUM_PARAMS; i++) {
        for (size_t j = 0; j < sizes[i]; j++) {
            host[off + j] = (kind[i] == WEIGHT)   ? normal(rng)
                          : (kind[i] == LN_GAMMA) ? 1.0f
                                                  : 0.0f;
        }
        off += sizes[i];
    }
    CHECK(cudaMemcpy(d_params, host.data(), total * sizeof(float),
                     cudaMemcpyHostToDevice));
}

static int load_mnist_csv(
    const char *path, std::vector<uint8_t> &pixels, std::vector<int> &labels)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    static char line[8192];
    fgets(line, sizeof(line), f);
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        labels.push_back((int)strtol(p, &p, 10));
        for (int i = 0; i < 784; i++) {
            p++;
            pixels.push_back((uint8_t)strtol(p, &p, 10));
        }
    }
    fclose(f);
    return (int)labels.size();
}

// =============================================================================
// SECTION 6 — main.
// =============================================================================

int main(int argc, char **argv)
{
    int provided;
    CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    int devices;
    CHECK(cudaGetDeviceCount(&devices));
    CHECK(cudaSetDevice(rank % devices));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    cublasHandle_t cublas;
    CHECK(cublasCreate(&cublas));
    CHECK(cublasSetStream(cublas, stream));

    ncclUniqueId nccl_id;
    if (rank == 0) CHECK(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t nccl;
    CHECK(ncclCommInitRank(&nccl, world, nccl_id, rank));

    const char *csv_path = (argc > 1) ? argv[1] : "data/train.csv";
    int   steps          = (argc > 2) ? atoi(argv[2]) : 200;
    int   B              = (argc > 3) ? atoi(argv[3]) : 8;
    float lr             = (argc > 4) ? (float)atof(argv[4]) : 0.05f;

    // PERF feature flags (env vars). 1 = new/fast path, 0 = old/slow path.
    auto read_flag = [](const char *name, int dflt) {
        const char *v = getenv(name);
        return v ? atoi(v) : dflt;
    };
    g_use_ln_bwd_fast   = read_flag("PERF_LN_BWD_FAST",   1);
    g_use_sgemv_bias    = read_flag("PERF_SGEMV_BIAS",    1);
    g_use_attn_batched  = read_flag("PERF_ATTN_BATCHED",  1);
    g_use_narrow_memset = read_flag("PERF_NARROW_MEMSET", 1);

    Cfg cfg = { /*vocab*/ 256, /*seq*/ 28*28, /*layers*/ 2,
                /*dim*/ 64,    /*heads*/ 4,    /*head_dim*/ 16,
                /*classes*/ 10 };
    int BT = B * cfg.seq;

    std::vector<uint8_t> pixels;
    std::vector<int>     labels;
    int N = load_mnist_csv(csv_path, pixels, labels);
    if (!N) {
        if (rank == 0) fprintf(stderr, "cannot read %s\n", csv_path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t param_sz[NUM_PARAMS], act_sz[NUM_ACTS];
    fill_param_sizes(param_sz, cfg);
    fill_act_sizes(act_sz, cfg, B);

    size_t total_params = 0, total_acts = 0;
    for (int i = 0; i < NUM_PARAMS; i++) total_params += param_sz[i];
    for (int i = 0; i < NUM_ACTS;   i++) total_acts   += act_sz[i];

    float *d_params, *d_grads, *d_momentum, *d_velocity;
    CHECK(cudaMalloc(&d_params,   total_params * sizeof(float)));
    CHECK(cudaMalloc(&d_grads,    total_params * sizeof(float)));
    CHECK(cudaMalloc(&d_momentum, total_params * sizeof(float)));
    CHECK(cudaMalloc(&d_velocity, total_params * sizeof(float)));
    CHECK(cudaMemset(d_momentum, 0, total_params * sizeof(float)));
    CHECK(cudaMemset(d_velocity, 0, total_params * sizeof(float)));

    float *d_acts, *d_dacts;
    CHECK(cudaMalloc(&d_acts,  total_acts * sizeof(float)));
    CHECK(cudaMalloc(&d_dacts, total_acts * sizeof(float)));

    float *params[NUM_PARAMS], *grads[NUM_PARAMS];
    float *acts[NUM_ACTS],     *dacts[NUM_ACTS];
    assign_pointers(params, d_params, param_sz, NUM_PARAMS);
    assign_pointers(grads,  d_grads,  param_sz, NUM_PARAMS);
    assign_pointers(acts,   d_acts,   act_sz,   NUM_ACTS);
    assign_pointers(dacts,  d_dacts,  act_sz,   NUM_ACTS);

    init_parameters(d_params, param_sz, total_params, /*seed*/ 42);

    // -------------------------------------------------------------------------
    // PERF: bias_grad replacement support.
    // g_ones[BT] is the constant 1.0f vector used by cublasSgemv inside
    // matmul_backward (replaces the old serial bias_grad kernel).
    CHECK(cudaMalloc(&g_ones, BT * sizeof(float)));
    {
        std::vector<float> tmp(BT, 1.0f);
        CHECK(cudaMemcpy(g_ones, tmp.data(), BT * sizeof(float),
                         cudaMemcpyHostToDevice));
    }

    // -------------------------------------------------------------------------
    // PERF: precompute pointer arrays for cublasSgemmBatched in attention.
    // Layout: 12 sections of size L*B*H, indexed by PT_* enum.
    {
        int H  = cfg.heads;
        int HD = cfg.head_dim;
        int T  = cfg.seq;
        int Dd = cfg.dim;
        int LBH = cfg.layers * B * H;
        size_t total_ptrs = (size_t)PT_COUNT * LBH;
        CHECK(cudaMalloc(&g_attn_ptrs, total_ptrs * sizeof(float*)));
        std::vector<float*> h_ptrs(total_ptrs);
        for (int l = 0; l < cfg.layers; l++) {
            float *qkv_l   = acts[A_QKV]       + (size_t)l * B * T * 3 * Dd;
            float *dqkv_l  = dacts[A_QKV]      + (size_t)l * B * T * 3 * Dd;
            float *pre_l   = acts[A_ATTN_PRE]  + (size_t)l * B * H * T * T;
            float *attn_l  = acts[A_ATTN]      + (size_t)l * B * H * T * T;
            float *out_l   = acts[A_ATTN_OUT]  + (size_t)l * B * T * Dd;
            float *dpre_l  = dacts[A_ATTN_PRE] + (size_t)l * B * H * T * T;
            float *dattn_l = dacts[A_ATTN]     + (size_t)l * B * H * T * T;
            float *dout_l  = dacts[A_ATTN_OUT] + (size_t)l * B * T * Dd;
            for (int b = 0; b < B; b++) {
                for (int h = 0; h < H; h++) {
                    int idx = (l * B + b) * H + h;
                    size_t bT3D = (size_t)b * T * 3 * Dd;
                    size_t bHTT = (size_t)b * H * T * T;
                    size_t bTD  = (size_t)b * T * Dd;
                    int hHD = h * HD;
                    int hTT = h * T * T;
                    h_ptrs[(size_t)PT_Q  * LBH + idx] = qkv_l   + bT3D + hHD;
                    h_ptrs[(size_t)PT_K  * LBH + idx] = qkv_l   + bT3D + Dd + hHD;
                    h_ptrs[(size_t)PT_V  * LBH + idx] = qkv_l   + bT3D + 2*Dd + hHD;
                    h_ptrs[(size_t)PT_S  * LBH + idx] = pre_l   + bHTT + hTT;
                    h_ptrs[(size_t)PT_A  * LBH + idx] = attn_l  + bHTT + hTT;
                    h_ptrs[(size_t)PT_O  * LBH + idx] = out_l   + bTD  + hHD;
                    h_ptrs[(size_t)PT_dQ * LBH + idx] = dqkv_l  + bT3D + hHD;
                    h_ptrs[(size_t)PT_dK * LBH + idx] = dqkv_l  + bT3D + Dd + hHD;
                    h_ptrs[(size_t)PT_dV * LBH + idx] = dqkv_l  + bT3D + 2*Dd + hHD;
                    h_ptrs[(size_t)PT_dS * LBH + idx] = dpre_l  + bHTT + hTT;
                    h_ptrs[(size_t)PT_dA * LBH + idx] = dattn_l + bHTT + hTT;
                    h_ptrs[(size_t)PT_dO * LBH + idx] = dout_l  + bTD  + hHD;
                }
            }
        }
        CHECK(cudaMemcpy(g_attn_ptrs, h_ptrs.data(),
                         total_ptrs * sizeof(float*),
                         cudaMemcpyHostToDevice));
    }

    int *d_pixel, *d_target;
    int *h_pixel, *h_target;
    CHECK(cudaMalloc(&d_pixel,  BT * sizeof(int)));
    CHECK(cudaMalloc(&d_target, B  * sizeof(int)));
    CHECK(cudaMallocHost(&h_pixel,  BT * sizeof(int)));
    CHECK(cudaMallocHost(&h_target, B  * sizeof(int)));

    float *d_loss_sum; int *d_correct;
    CHECK(cudaMalloc(&d_loss_sum, sizeof(float)));
    CHECK(cudaMalloc(&d_correct,  sizeof(int)));

    if (rank == 0) {
        printf("ranks=%d N=%d B=%d T=%d L=%d D=%d H=%d C=%d "
               "params=%zu steps=%d lr=%g\n",
               world, N, B, cfg.seq, cfg.layers, cfg.dim, cfg.heads,
               cfg.classes, total_params, steps, lr);
        printf("perf flags: LN_BWD_FAST=%d SGEMV_BIAS=%d ATTN_BATCHED=%d "
               "NARROW_MEMSET=%d\n",
               g_use_ln_bwd_fast, g_use_sgemv_bias,
               g_use_attn_batched, g_use_narrow_memset);
        fflush(stdout);
    }

    // ---- coarse CUDA events ----
    // ev_zero is recorded after the memset that clears grads/dacts so we can
    // attribute the cost separately from the actual forward kernels.
    cudaEvent_t ev_step0, ev_h2d, ev_zero, ev_fwd, ev_bwd, ev_nccl, ev_adam;
    CHECK(cudaEventCreate(&ev_step0));
    CHECK(cudaEventCreate(&ev_h2d));
    CHECK(cudaEventCreate(&ev_zero));
    CHECK(cudaEventCreate(&ev_fwd));
    CHECK(cudaEventCreate(&ev_bwd));
    CHECK(cudaEventCreate(&ev_nccl));
    CHECK(cudaEventCreate(&ev_adam));

    // ---- fine-grained per-kernel event arrays ----
    cudaEvent_t fwd_ev[FWD_NEV], bwd_ev[BWD_NEV];
    for (int i = 0; i < FWD_NEV; i++) CHECK(cudaEventCreate(&fwd_ev[i]));
    for (int i = 0; i < BWD_NEV; i++) CHECK(cudaEventCreate(&bwd_ev[i]));

    FILE *log_fp = NULL;
    if (rank == 0) {
        log_fp = fopen("training_log.csv", "w");
        if (log_fp) {
            // Coarse columns. t_zero_ms = cost of the per-step memset
            // (just d_grads when PERF_NARROW_MEMSET=1, plus full d_dacts
            // when =0). t_fwd_ms now excludes memset (starts at ev_zero).
            fprintf(log_fp,
                "step,elapsed_s,loss,accuracy,"
                "t_h2d_ms,t_zero_ms,t_fwd_ms,t_bwd_ms,t_nccl_ms,t_adam_ms,"
                "tf_enc");
            // Fine forward: per layer
            for (int l = 0; l < cfg.layers; l++)
                fprintf(log_fp,
                    ",tf_l%d_ln1,tf_l%d_qkv,tf_l%d_attn,"
                    "tf_l%d_aproj,tf_l%d_ln2,tf_l%d_fc1,"
                    "tf_l%d_gelu,tf_l%d_fc2",
                    l,l,l, l,l,l, l,l);
            // Fine forward: tail
            fprintf(log_fp, ",tf_lnf,tf_pool,tf_head,tf_loss");
            // Fine backward: head
            fprintf(log_fp, ",tb_loss,tb_head,tb_pool,tb_lnf");
            // Fine backward: per layer (printed in reverse-iteration order)
            for (int i = 0; i < cfg.layers; i++) {
                int l = cfg.layers - 1 - i;
                fprintf(log_fp,
                    ",tb_l%d_fc2,tb_l%d_gelu,tb_l%d_fc1,"
                    "tb_l%d_ln2,tb_l%d_aproj,tb_l%d_attn,"
                    "tb_l%d_qkv,tb_l%d_ln1",
                    l,l,l, l,l,l, l,l);
            }
            // Fine backward: encoder
            fprintf(log_fp, ",tb_enc\n");
            fflush(log_fp);
        }
    }

    std::mt19937 rng(42 + 1000u * rank);
    std::uniform_int_distribution<int> sampler(0, N - 1);

    auto t_start = std::chrono::steady_clock::now();

    for (int step = 1; step <= steps; step++) {
        for (int b = 0; b < B; b++) {
            int j = sampler(rng);
            h_target[b] = labels[j];
            for (int t = 0; t < cfg.seq; t++)
                h_pixel[b*cfg.seq + t] = (int)pixels[(size_t)j*cfg.seq + t];
        }

        cudaEventRecord(ev_step0, stream);
        CHECK(cudaMemcpyAsync(d_pixel,  h_pixel,  BT*sizeof(int),
                              cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(d_target, h_target, B *sizeof(int),
                              cudaMemcpyHostToDevice, stream));
        cudaEventRecord(ev_h2d, stream);

        // Two memset paths.
        // FAST (default): zero only what backward reads-then-accumulates:
        //   - d_grads (~670 KB): encoder_backward and layernorm_backward
        //     accumulate via atomicAdd into dwte/dwpe/dgamma/dbeta.
        //   - dacts[A_RESID2] + (L-1)*BTD (~400 KB): the LNF backward writes
        //     `dx +=` into this slice and it isn't prefilled by a D2D memcpy.
        // SLOW (original): zero the full d_grads + d_dacts (~350 MB) buffers.
        CHECK(cudaMemsetAsync(d_grads, 0, total_params * sizeof(float), stream));
        if (g_use_narrow_memset) {
            CHECK(cudaMemsetAsync(
                dacts[A_RESID2] + (size_t)(cfg.layers - 1) * B * cfg.seq * cfg.dim,
                0,
                (size_t)B * cfg.seq * cfg.dim * sizeof(float),
                stream));
        } else {
            CHECK(cudaMemsetAsync(d_dacts, 0, total_acts * sizeof(float), stream));
        }
        cudaEventRecord(ev_zero, stream);

        model_forward(params, acts, d_pixel, cfg, B, cublas, stream, fwd_ev);
        softmax_ce_forward<<<B, 32, 0, stream>>>(
            acts[A_PROBS], acts[A_LOSSES], acts[A_LOGITS], d_target, B, cfg.classes);
        cudaEventRecord(ev_fwd, stream);

        model_backward(params, grads, acts, dacts,
                       d_pixel, d_target, cfg, B, cublas, stream, bwd_ev);
        cudaEventRecord(ev_bwd, stream);

        if (world > 1)
            CHECK(ncclAllReduce(d_grads, d_grads, total_params,
                                ncclFloat, ncclSum, nccl, stream));
        cudaEventRecord(ev_nccl, stream);

        float scale = 1.0f / (float)(B * world);
        float bc1   = 1.0f - powf(0.9f,   (float)step);
        float bc2   = 1.0f - powf(0.999f, (float)step);
        adam_step<<<GRID((int)total_params), 0, stream>>>(
            d_params, d_grads, d_momentum, d_velocity,
            (int)total_params, lr, 0.9f, 0.999f, 1e-8f, scale, bc1, bc2);
        cudaEventRecord(ev_adam, stream);

        if (step % 10 == 0 || step == steps) {
            CHECK(cudaMemsetAsync(d_loss_sum, 0, sizeof(float), stream));
            CHECK(cudaMemsetAsync(d_correct,  0, sizeof(int),   stream));
            atomic_sum   <<<GRID(B), 0, stream>>>(d_loss_sum, acts[A_LOSSES], B);
            count_correct<<<GRID(B), 0, stream>>>(d_correct,  acts[A_LOGITS],
                                                  d_target, B, cfg.classes);
            float h_loss; int h_corr;
            CHECK(cudaMemcpyAsync(&h_loss, d_loss_sum, sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));
            CHECK(cudaMemcpyAsync(&h_corr, d_correct,  sizeof(int),
                                  cudaMemcpyDeviceToHost, stream));
            CHECK(cudaStreamSynchronize(stream));

            // ---- coarse timings ----
            // t_zero_ms isolates the memset cost (fix #4) from the actual
            // forward kernels: t_fwd_ms now starts at ev_zero.
            float t_h2d_ms = 0, t_zero_ms = 0, t_fwd_ms = 0, t_bwd_ms = 0,
                  t_nccl_ms = 0, t_adam_ms = 0;
            cudaEventElapsedTime(&t_h2d_ms,  ev_step0, ev_h2d);
            cudaEventElapsedTime(&t_zero_ms, ev_h2d,   ev_zero);
            cudaEventElapsedTime(&t_fwd_ms,  ev_zero,  ev_fwd);
            cudaEventElapsedTime(&t_bwd_ms,  ev_fwd,   ev_bwd);
            cudaEventElapsedTime(&t_nccl_ms, ev_bwd,   ev_nccl);
            cudaEventElapsedTime(&t_adam_ms, ev_nccl,  ev_adam);

            // ---- fine forward timings ----
            // tf[0]   : ev_zero -> fwd_ev[0]         (encoder, excludes memset)
            // tf[k]   : fwd_ev[k-1] -> fwd_ev[k]     (k = 1 .. fwd_nev_used-1)
            // tf_loss : fwd_ev[last] -> ev_fwd        (softmax_ce_forward in main)
            int fwd_nev_used = 1 + cfg.layers * FKPL + 3;
            float tf[FWD_NEV] = {};
            cudaEventElapsedTime(&tf[0], ev_zero, fwd_ev[0]);
            for (int k = 1; k < fwd_nev_used; k++)
                cudaEventElapsedTime(&tf[k], fwd_ev[k-1], fwd_ev[k]);
            float tf_loss = 0.0f;
            cudaEventElapsedTime(&tf_loss, fwd_ev[fwd_nev_used - 1], ev_fwd);

            // ---- fine backward timings ----
            // tb[0]   : ev_fwd     -> bwd_ev[0]       (softmax_ce_backward)
            // tb[k]   : bwd_ev[k-1] -> bwd_ev[k]      (k = 1 .. bwd_nev_used-1)
            int bwd_nev_used = 4 + cfg.layers * BKPL + 1;
            float tb[BWD_NEV] = {};
            cudaEventElapsedTime(&tb[0], ev_fwd, bwd_ev[0]);
            for (int k = 1; k < bwd_nev_used; k++)
                cudaEventElapsedTime(&tb[k], bwd_ev[k-1], bwd_ev[k]);

            float mean_loss   = h_loss / B;
            int   global_corr = h_corr, global_n = B;
            MPI_Allreduce(MPI_IN_PLACE, &mean_loss,   1, MPI_FLOAT,
                          MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &global_corr, 1, MPI_INT,
                          MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &global_n,    1, MPI_INT,
                          MPI_SUM, MPI_COMM_WORLD);
            mean_loss /= world;

            if (rank == 0) {
                double el = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_start).count();

                printf("step %4d | loss %.4f | acc %.3f | "
                       "h2d %.2f zero %.2f fwd %.2f bwd %.2f nccl %.2f adam %.2f ms\n",
                       step, mean_loss, (float)global_corr / global_n,
                       t_h2d_ms, t_zero_ms, t_fwd_ms, t_bwd_ms, t_nccl_ms, t_adam_ms);

                if (log_fp) {
                    // coarse
                    fprintf(log_fp, "%d,%.2f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
                            step, el, mean_loss,
                            (float)global_corr / global_n,
                            t_h2d_ms, t_zero_ms, t_fwd_ms, t_bwd_ms, t_nccl_ms, t_adam_ms);
                    // fine forward: enc
                    fprintf(log_fp, ",%.3f", tf[0]);
                    // fine forward: per layer
                    for (int l = 0; l < cfg.layers; l++) {
                        int base = 1 + l * FKPL;
                        for (int k = 0; k < FKPL; k++)
                            fprintf(log_fp, ",%.3f", tf[base + k]);
                    }
                    // fine forward: tail (lnf, pool, head, loss)
                    int fbase = 1 + cfg.layers * FKPL;
                    fprintf(log_fp, ",%.3f,%.3f,%.3f,%.3f",
                            tf[fbase], tf[fbase+1], tf[fbase+2], tf_loss);
                    // fine backward: fixed head (loss, head, pool, lnf)
                    for (int k = 0; k < 4; k++)
                        fprintf(log_fp, ",%.3f", tb[k]);
                    // fine backward: per layer (stored in reverse-iteration order)
                    for (int i = 0; i < cfg.layers; i++) {
                        int base = 4 + i * BKPL;
                        for (int k = 0; k < BKPL; k++)
                            fprintf(log_fp, ",%.3f", tb[base + k]);
                    }
                    // fine backward: encoder
                    fprintf(log_fp, ",%.3f\n", tb[4 + cfg.layers * BKPL]);
                    fflush(log_fp);
                }
            }
        }
    }

    CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::steady_clock::now();

    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    long long total_images = (long long)steps * B * world;
    if (rank == 0) {
        printf("\nfinished: %d steps in %.2fs\n", steps, elapsed);
        printf("throughput: %.0f img/s global  |  %.0f img/s/GPU\n",
               total_images / elapsed, total_images / elapsed / world);
    }

    if (log_fp) fclose(log_fp);

    cudaEventDestroy(ev_step0); cudaEventDestroy(ev_h2d);
    cudaEventDestroy(ev_zero);
    cudaEventDestroy(ev_fwd);   cudaEventDestroy(ev_bwd);
    cudaEventDestroy(ev_nccl);  cudaEventDestroy(ev_adam);
    for (int i = 0; i < FWD_NEV; i++) cudaEventDestroy(fwd_ev[i]);
    for (int i = 0; i < BWD_NEV; i++) cudaEventDestroy(bwd_ev[i]);

    cudaFreeHost(h_pixel); cudaFreeHost(h_target);
    cudaFree(d_pixel);     cudaFree(d_target);
    cudaFree(d_acts);      cudaFree(d_dacts);
    cudaFree(d_params);    cudaFree(d_grads);
    cudaFree(d_momentum);  cudaFree(d_velocity);
    cudaFree(d_loss_sum);  cudaFree(d_correct);
    cudaFree(g_ones);      cudaFree(g_attn_ptrs);
    ncclCommDestroy(nccl);
    cublasDestroy(cublas);
    cudaStreamDestroy(stream);
    MPI_Finalize();
    return 0;
}
