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

// Warp-level reductions across 32 lanes — used inside per-row kernels
// (LayerNorm, softmax, attention softmax).
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

// Model hyper-parameters.
struct Cfg {
    int vocab;     // V — number of distinct pixel intensities (256 for uint8)
    int seq;       // T — sequence length (28*28)
    int layers;    // L — number of transformer blocks
    int dim;       // D — model hidden size
    int heads;     // H — number of attention heads
    int head_dim;  // D / H
    int classes;   // C — number of output classes (10 for MNIST)
};

// =============================================================================
// SECTION 1 — parameter and activation catalogs.
//
// All trainable parameters live in ONE flat device buffer of size sum(param_sizes).
// `param_ptrs[i]` is the pointer at which tensor `i` begins inside that buffer.
// Same scheme for activations and their gradients. This lets us reduce the
// whole gradient blob with a single ncclAllReduce and run a single SGD kernel
// over every parameter at once.
// =============================================================================

enum {
    P_TOK_EMB,                // (V, D)        token (pixel intensity) embedding
    P_POS_EMB,                // (T, D)        positional embedding
    P_LN1_W, P_LN1_B,         // (L, D) each   first LayerNorm
    P_QKV_W, P_QKV_B,         // (L, 3D, D) / (L, 3D)  Q/K/V linear (fused)
    P_ATTPROJ_W, P_ATTPROJ_B, // (L, D, D)  / (L, D)   attention output projection
    P_LN2_W, P_LN2_B,         // second LayerNorm
    P_FC1_W, P_FC1_B,         // (L, 4D, D) / (L, 4D)  first MLP linear (D -> 4D)
    P_FC2_W, P_FC2_B,         // (L, D, 4D) / (L, D)   second MLP linear (4D -> D)
    P_LNF_W, P_LNF_B,         // final LayerNorm
    P_HEAD,                   // (C, D)        classifier head
    NUM_PARAMS
};

enum {
    A_ENCODED,                  // (B, T, D)        token+pos embedding sum
    A_LNF, A_LNF_MEAN, A_LNF_RSTD,    // final LayerNorm output, stats
    A_POOLED,                   // (B, D)           mean-pooled features
    A_LOGITS,                   // (B, C)
    A_PROBS,                    // (B, C)
    A_LOSSES,                   // (B,)             per-sample loss
    A_LN1, A_LN1_MEAN, A_LN1_RSTD,    // (L, B, T, D)   first LayerNorm
    A_QKV,                      // (L, B, T, 3D)
    A_ATTN_PRE,                 // (L, B, H, T, T)  pre-softmax scores
    A_ATTN,                     // (L, B, H, T, T)  attention weights
    A_ATTN_OUT,                 // (L, B, T, D)     attn @ V output
    A_ATTPROJ,                  // (L, B, T, D)     attention output projection
    A_RESID1,                   // (L, B, T, D)     after first residual
    A_LN2, A_LN2_MEAN, A_LN2_RSTD,
    A_FC1,                      // (L, B, T, 4D)    after first MLP linear
    A_FC1_GELU,                 // (L, B, T, 4D)    after GELU
    A_FC2,                      // (L, B, T, D)     after second MLP linear
    A_RESID2,                   // (L, B, T, D)     after second residual
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

// Walk the flat buffer and hand out a pointer per slot.
static void assign_pointers(float **ptrs, float *base, const size_t *sz, int n)
{
    float *cur = base;
    for (int i = 0; i < n; i++) { ptrs[i] = cur; cur += sz[i]; }
}

// =============================================================================
// SECTION 2 — kernels.
//
// Naming: `*_forward` writes to its output buffer with overwrite semantics;
// `*_backward` either overwrites its dx buffer or accumulates with `+=` if the
// downstream tensor receives gradient from more than one path (only LayerNorm
// and the embedding scatter use accumulation).
// =============================================================================

// -----------------------------------------------------------------------------
// Embedding: out[b, t, :] = wte[pixel[b, t]] + wpe[t].
// -----------------------------------------------------------------------------
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

// Backward scatters dout into the embedding tables. Several positions may write
// to the same row (same pixel value or same time index), so use atomicAdd.
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

// -----------------------------------------------------------------------------
// LayerNorm over the last dimension (size D), per row.
//
//   mean[n] = mean_d(x[n, d])
//   var[n]  = mean_d((x[n, d] - mean[n])^2)
//   rstd[n] = 1 / sqrt(var[n] + eps)
//   y[n, d] = (x[n, d] - mean[n]) * rstd[n] * gamma[d] + beta[d]
//
// Launched as <<<N, 32>>>: one warp per row. mean[] and rstd[] are saved for
// backward.
// -----------------------------------------------------------------------------
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

// LayerNorm backward.
//
// Let x_hat = (x - mean) * rstd. Then
//   dx_hat = dy * gamma
//   dx     = rstd * (dx_hat - (mean(dx_hat) + x_hat * mean(dx_hat * x_hat)))
//          = rstd * (dx_hat - (S1 + rstd^2 * (x - mean) * S2) / D)
// with
//   S1 = sum_d dx_hat[d]
//   S2 = sum_d dx_hat[d] * (x[d] - mean[d])
//
// dx is *accumulated* (`+=`): the residual stream gradient is pre-loaded into
// dx before this kernel runs, and we add the LN contribution on top of it.
// dgamma and dbeta accumulate over all rows via atomicAdd.
__global__ void layernorm_backward(
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
        row_dx[d] += dxd;                                   // accumulate
    }
    for (int d = tid; d < D; d += 32) {
        atomicAdd(&dgamma[d], row_dy[d] * (row_x[d] - mean) * rstd);
        atomicAdd(&dbeta[d],  row_dy[d]);
    }
}

// -----------------------------------------------------------------------------
// Bias broadcast for the linear layers.
//   forward:  y[n, k] += b[k]
//   backward: db[k] = sum_n dy[n, k]
// -----------------------------------------------------------------------------
__global__ void bias_add(float *y, const float *b, int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N * K) y[i] += b[i % K];
}
__global__ void bias_grad(float *db, const float *dy, int N, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    float s = 0.0f;
    for (int n = 0; n < N; n++) s += dy[(size_t)n * K + k];
    db[k] = s;
}

// -----------------------------------------------------------------------------
// Self-attention (bidirectional — no causal mask).
//
// qkv layout: [B, T, 3, H, head_dim] flattened. Within a position (b, t) the
// first D floats are Q, next D are K, last D are V; within each, head h owns
// floats [h*head_dim .. (h+1)*head_dim). The helper below extracts a pointer
// to the (b, t, which ∈ {Q=0, K=1, V=2}, h) slice.
// -----------------------------------------------------------------------------
__device__ __forceinline__ size_t qkv_offset(
    int b, int t, int which, int h, int T, int D, int head_dim)
{
    return ((size_t)b * T + t) * 3 * D + which * D + h * head_dim;
}

//   scores[b, h, t1, t2] = (Q[b, h, t1, :] . K[b, h, t2, :]) / sqrt(head_dim)
__global__ void attention_qk(
    float *scores, const float *qkv,
    int B, int T, int H, int D, int head_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * H * T * T) return;
    int b  =  i / (H * T * T);
    int h  = (i / (T * T)) % H;
    int t1 = (i / T) % T;
    int t2 =  i % T;
    const float *q = qkv + qkv_offset(b, t1, 0, h, T, D, head_dim);
    const float *k = qkv + qkv_offset(b, t2, 1, h, T, D, head_dim);

    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) dot += q[d] * k[d];
    scores[i] = dot * rsqrtf((float)head_dim);
}

// One warp per (b, h, t1) row: softmax along the t2 axis (full T, no mask).
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

//   out[b, t1, h*head_dim + d] = sum_t2 attn[b, h, t1, t2] * V[b, t2, h*head_dim + d]
__global__ void attention_av(
    float *out, const float *attn, const float *qkv,
    int B, int T, int H, int D, int head_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * T * D) return;
    int b  =  i / (T * D);
    int t1 = (i / D) % T;
    int hd =  i % D;
    int h  = hd / head_dim;
    int di = hd % head_dim;

    float sum = 0.0f;
    for (int t2 = 0; t2 < T; t2++) {
        float a = attn[((size_t)b * H + h) * T * T + (size_t)t1 * T + t2];
        float v = qkv[qkv_offset(b, t2, 2, h, T, D, head_dim) + di];
        sum += a * v;
    }
    out[i] = sum;
}

// Attention backward. Five kernels, each is a direct application of the
// chain rule. Notation: d_out = gradient of attn_av output.

// dV[b, t2, hd] = sum_{t1} attn[b, h, t1, t2] * d_out[b, t1, hd]
__global__ void attention_dv(
    float *dqkv, const float *attn, const float *d_out,
    int B, int T, int H, int D, int head_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * T * D) return;
    int b  =  i / (T * D);
    int t2 = (i / D) % T;
    int hd =  i % D;
    int h  = hd / head_dim;
    int di = hd % head_dim;

    float sum = 0.0f;
    for (int t1 = 0; t1 < T; t1++) {
        float a   = attn[((size_t)b * H + h) * T * T + (size_t)t1 * T + t2];
        float dyx = d_out[((size_t)b * T + t1) * D + h * head_dim + di];
        sum += a * dyx;
    }
    dqkv[qkv_offset(b, t2, 2, h, T, D, head_dim) + di] = sum;
}

// d_attn[b, h, t1, t2] = sum_d V[b, t2, hd] * d_out[b, t1, hd]
__global__ void attention_d_attn(
    float *d_attn, const float *qkv, const float *d_out,
    int B, int T, int H, int D, int head_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * H * T * T) return;
    int b  =  i / (H * T * T);
    int h  = (i / (T * T)) % H;
    int t1 = (i / T) % T;
    int t2 =  i % T;

    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float v   = qkv[qkv_offset(b, t2, 2, h, T, D, head_dim) + d];
        float dyx = d_out[((size_t)b * T + t1) * D + h * head_dim + d];
        sum += v * dyx;
    }
    d_attn[i] = sum;
}

// Softmax backward, per (b, h, t1) row:
//   d_scores[t2] = attn[t2] * (d_attn[t2] - sum_{j} attn[j] * d_attn[j])
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

// dQ[b, t1, hd] = scale * sum_{t2} K[b, t2, hd] * d_scores[b, h, t1, t2]
__global__ void attention_dq(
    float *dqkv, const float *d_scores, const float *qkv,
    int B, int T, int H, int D, int head_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * T * D) return;
    int b  =  i / (T * D);
    int t1 = (i / D) % T;
    int hd =  i % D;
    int h  = hd / head_dim;
    int di = hd % head_dim;

    float sum = 0.0f;
    for (int t2 = 0; t2 < T; t2++) {
        float ds = d_scores[((size_t)b * H + h) * T * T + (size_t)t1 * T + t2];
        float k  = qkv[qkv_offset(b, t2, 1, h, T, D, head_dim) + di];
        sum += k * ds;
    }
    dqkv[qkv_offset(b, t1, 0, h, T, D, head_dim) + di] = sum * rsqrtf((float)head_dim);
}

// dK[b, t2, hd] = scale * sum_{t1} Q[b, t1, hd] * d_scores[b, h, t1, t2]
__global__ void attention_dk(
    float *dqkv, const float *d_scores, const float *qkv,
    int B, int T, int H, int D, int head_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * T * D) return;
    int b  =  i / (T * D);
    int t2 = (i / D) % T;
    int hd =  i % D;
    int h  = hd / head_dim;
    int di = hd % head_dim;

    float sum = 0.0f;
    for (int t1 = 0; t1 < T; t1++) {
        float ds = d_scores[((size_t)b * H + h) * T * T + (size_t)t1 * T + t2];
        float q  = qkv[qkv_offset(b, t1, 0, h, T, D, head_dim) + di];
        sum += q * ds;
    }
    dqkv[qkv_offset(b, t2, 1, h, T, D, head_dim) + di] = sum * rsqrtf((float)head_dim);
}

// -----------------------------------------------------------------------------
// GELU activation, tanh approximation:
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// -----------------------------------------------------------------------------
#define GELU_K 0.7978845608f      // sqrt(2 / pi)

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

// -----------------------------------------------------------------------------
// Residual add: out = a + b. Backward is handled by the residual-stream pattern
// (we cudaMemcpyAsync the incoming gradient onto the residual buffer before
// each block-internal backward chain, and LayerNorm's `dx +=` adds the
// block's contribution on top).
// -----------------------------------------------------------------------------
__global__ void residual_add(float *out, const float *a, const float *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// -----------------------------------------------------------------------------
// Mean over the sequence (T) axis.
//   forward:  out[b, d] = (1 / T) * sum_t x[b, t, d]
//   backward: dx[b, t, d] = dy[b, d] / T   (broadcast over t)
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Softmax + cross-entropy. One warp per sample.
//   loss[b]  = logsumexp(logits[b, :]) - logits[b, target[b]]
//   d_logits = probs - one_hot(target)
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// SGD with momentum. Operates on the whole flat parameter buffer at once.
//   m <- momentum * m + g * scale
//   w <- w - lr * m
// scale = 1 / (B * world)  so that after NCCL allreduce the gradient is the
// mean over the global batch.
// -----------------------------------------------------------------------------
__global__ void sgd_step(
    float *w, const float *g, float *m,
    int n, float lr, float momentum, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float new_m = momentum * m[i] + g[i] * scale;
    m[i]  = new_m;
    w[i] -= lr * new_m;
}

// -----------------------------------------------------------------------------
// Reporting kernels (loss sum + top-1 accuracy counter).
// -----------------------------------------------------------------------------
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
//
// Convention: weights are row-major [out_features, in_features].
// Forward:  y[N, OC] = x[N, C] @ w^T   (optionally + bias broadcast over rows).
// Backward: writes dx, dw, db with overwrite semantics (beta = 0).
// =============================================================================

static void matmul_forward(
    float *y, const float *x, const float *w, const float *b,
    int N, int C, int OC, cublasHandle_t cublas, cudaStream_t stream)
{
    float alpha = 1.0f, beta = 0.0f;
    // y(col-major OC x N) = w(col-major C x OC)^T * x(col-major C x N)
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

    // dw(row-major OC x C) = dy^T(OC x N) * x(N x C).
    CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                      C, OC, N,
                      &alpha, x, C, dy, OC,
                      &beta,  dw, C));

    if (db) bias_grad<<<GRID(OC), 0, stream>>>(db, dy, N, OC);

    // dx(row-major N x C) = dy(N x OC) * w(OC x C).
    CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                      C, N, OC,
                      &alpha, w, C, dy, OC,
                      &beta,  dx, C));
}

// =============================================================================
// SECTION 4 — model forward and backward.
//
// `params[i]` / `acts[i]` are device pointers obtained from `assign_pointers`.
// Per-layer tensors are addressed as `acts[A_QKV] + layer * stride_per_layer`.
// =============================================================================

static void model_forward(
    float **params, float **acts, const int *pixel,
    Cfg c, int B,
    cublasHandle_t cublas, cudaStream_t stream)
{
    int T = c.seq, L = c.layers, D = c.dim, H = c.heads, HD = c.head_dim, C = c.classes;
    int BT       = B * T;
    int BTD      = B * T * D;
    int BT3D     = B * T * 3 * D;
    int BT4D     = B * T * 4 * D;
    int BHTT     = B * H * T * T;

    // Embed pixels.
    encoder_forward<<<GRID(BTD), 0, stream>>>(
        acts[A_ENCODED], pixel, params[P_TOK_EMB], params[P_POS_EMB], B, T, D);

    // residual_in is the running residual-stream tensor through layers.
    const float *residual_in = acts[A_ENCODED];

    for (int l = 0; l < L; l++) {
        // --- attention sub-block ---
        layernorm_forward<<<BT, 32, 0, stream>>>(
            acts[A_LN1] + l*BTD, acts[A_LN1_MEAN] + l*BT, acts[A_LN1_RSTD] + l*BT,
            residual_in, params[P_LN1_W] + l*D, params[P_LN1_B] + l*D, BT, D);

        matmul_forward(
            acts[A_QKV] + l*BT3D, acts[A_LN1] + l*BTD,
            params[P_QKV_W] + l*3*D*D, params[P_QKV_B] + l*3*D,
            BT, D, 3*D, cublas, stream);

        attention_qk<<<GRID(BHTT), 0, stream>>>(
            acts[A_ATTN_PRE] + l*BHTT, acts[A_QKV] + l*BT3D, B, T, H, D, HD);
        attention_softmax<<<B*H*T, 32, 0, stream>>>(
            acts[A_ATTN] + l*BHTT, acts[A_ATTN_PRE] + l*BHTT, B, H, T);
        attention_av<<<GRID(BTD), 0, stream>>>(
            acts[A_ATTN_OUT] + l*BTD, acts[A_ATTN] + l*BHTT,
            acts[A_QKV] + l*BT3D, B, T, H, D, HD);

        matmul_forward(
            acts[A_ATTPROJ] + l*BTD, acts[A_ATTN_OUT] + l*BTD,
            params[P_ATTPROJ_W] + l*D*D, params[P_ATTPROJ_B] + l*D,
            BT, D, D, cublas, stream);

        residual_add<<<GRID(BTD), 0, stream>>>(
            acts[A_RESID1] + l*BTD, residual_in, acts[A_ATTPROJ] + l*BTD, BTD);

        // --- MLP sub-block ---
        layernorm_forward<<<BT, 32, 0, stream>>>(
            acts[A_LN2] + l*BTD, acts[A_LN2_MEAN] + l*BT, acts[A_LN2_RSTD] + l*BT,
            acts[A_RESID1] + l*BTD, params[P_LN2_W] + l*D, params[P_LN2_B] + l*D, BT, D);

        matmul_forward(
            acts[A_FC1] + l*BT4D, acts[A_LN2] + l*BTD,
            params[P_FC1_W] + l*4*D*D, params[P_FC1_B] + l*4*D,
            BT, D, 4*D, cublas, stream);

        gelu_forward<<<GRID(BT4D), 0, stream>>>(
            acts[A_FC1_GELU] + l*BT4D, acts[A_FC1] + l*BT4D, BT4D);

        matmul_forward(
            acts[A_FC2] + l*BTD, acts[A_FC1_GELU] + l*BT4D,
            params[P_FC2_W] + l*D*4*D, params[P_FC2_B] + l*D,
            BT, 4*D, D, cublas, stream);

        residual_add<<<GRID(BTD), 0, stream>>>(
            acts[A_RESID2] + l*BTD, acts[A_RESID1] + l*BTD, acts[A_FC2] + l*BTD, BTD);

        residual_in = acts[A_RESID2] + l*BTD;
    }

    // Final LayerNorm + mean-pool + classifier head.
    layernorm_forward<<<BT, 32, 0, stream>>>(
        acts[A_LNF], acts[A_LNF_MEAN], acts[A_LNF_RSTD],
        residual_in, params[P_LNF_W], params[P_LNF_B], BT, D);

    mean_pool_forward<<<GRID(B * D), 0, stream>>>(
        acts[A_POOLED], acts[A_LNF], B, T, D);

    matmul_forward(
        acts[A_LOGITS], acts[A_POOLED], params[P_HEAD], NULL,
        B, D, C, cublas, stream);
}

// Backward — must be called *after* the loss kernel has filled acts[A_PROBS]
// (we re-use it). Walks the network in reverse, building gradients in the
// matching d_* slots of `dacts`.
//
// Residual-stream pattern. Inside each block the input residual receives
// gradient from two paths: the direct skip connection and the sub-block (LN
// -> ... -> matmul). We start by copying the incoming residual gradient into
// the residual buffer (cudaMemcpyAsync), then let LayerNorm's `dx +=` add the
// sub-block contribution. After both sub-blocks, that buffer holds the full
// gradient w.r.t. the block's input.
static void model_backward(
    float **params, float **grads, float **acts, float **dacts,
    const int *pixel, const int *target,
    Cfg c, int B,
    cublasHandle_t cublas, cudaStream_t stream)
{
    int T = c.seq, L = c.layers, D = c.dim, H = c.heads, HD = c.head_dim, C = c.classes;
    int BT       = B * T;
    int BTD      = B * T * D;
    int BT3D     = B * T * 3 * D;
    int BT4D     = B * T * 4 * D;
    int BHTT     = B * H * T * T;

    // Loss backward.
    softmax_ce_backward<<<GRID(B * C), 0, stream>>>(
        dacts[A_LOGITS], acts[A_PROBS], target, B, C);

    // Head linear.
    matmul_backward(
        dacts[A_POOLED], grads[P_HEAD], NULL,
        dacts[A_LOGITS], acts[A_POOLED], params[P_HEAD],
        B, D, C, cublas, stream);

    // Mean-pool fans the [B, D] gradient back into [B, T, D].
    mean_pool_backward<<<GRID(BTD), 0, stream>>>(dacts[A_LNF], dacts[A_POOLED], B, T, D);

    // Final LayerNorm. Accumulates into dacts[A_RESID2 of last layer] which is
    // zero at this point, so it ends up holding the full lnf-path gradient.
    layernorm_backward<<<BT, 32, 0, stream>>>(
        dacts[A_RESID2] + (L-1)*BTD,
        grads[P_LNF_W], grads[P_LNF_B], dacts[A_LNF],
        acts[A_RESID2] + (L-1)*BTD, params[P_LNF_W],
        acts[A_LNF_MEAN], acts[A_LNF_RSTD], BT, D);

    for (int l = L - 1; l >= 0; l--) {
        // Identify upstream residual: for layer l > 0 it is the previous block's
        // residual2; for layer 0 it is the encoder output.
        const float *upstream     = (l == 0) ? acts[A_ENCODED]   : acts[A_RESID2]  + (l-1)*BTD;
        float       *d_upstream   = (l == 0) ? dacts[A_ENCODED]  : dacts[A_RESID2] + (l-1)*BTD;

        // Direct skip path: d_resid1 receives d_resid2; the MLP sub-block back
        // pass will add its contribution.
        CHECK(cudaMemcpyAsync(
            dacts[A_RESID1] + l*BTD, dacts[A_RESID2] + l*BTD,
            BTD * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        // --- MLP backward ---
        matmul_backward(
            dacts[A_FC1_GELU] + l*BT4D, grads[P_FC2_W] + l*D*4*D, grads[P_FC2_B] + l*D,
            dacts[A_RESID2] + l*BTD, acts[A_FC1_GELU] + l*BT4D,
            params[P_FC2_W] + l*D*4*D,
            BT, 4*D, D, cublas, stream);

        gelu_backward<<<GRID(BT4D), 0, stream>>>(
            dacts[A_FC1] + l*BT4D, dacts[A_FC1_GELU] + l*BT4D,
            acts[A_FC1] + l*BT4D, BT4D);

        matmul_backward(
            dacts[A_LN2] + l*BTD, grads[P_FC1_W] + l*4*D*D, grads[P_FC1_B] + l*4*D,
            dacts[A_FC1] + l*BT4D, acts[A_LN2] + l*BTD, params[P_FC1_W] + l*4*D*D,
            BT, D, 4*D, cublas, stream);

        layernorm_backward<<<BT, 32, 0, stream>>>(
            dacts[A_RESID1] + l*BTD,
            grads[P_LN2_W] + l*D, grads[P_LN2_B] + l*D, dacts[A_LN2] + l*BTD,
            acts[A_RESID1] + l*BTD, params[P_LN2_W] + l*D,
            acts[A_LN2_MEAN] + l*BT, acts[A_LN2_RSTD] + l*BT, BT, D);

        // Direct skip path for the attention sub-block.
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

        attention_dv<<<GRID(BTD), 0, stream>>>(
            dacts[A_QKV] + l*BT3D, acts[A_ATTN] + l*BHTT,
            dacts[A_ATTN_OUT] + l*BTD, B, T, H, D, HD);
        attention_d_attn<<<GRID(BHTT), 0, stream>>>(
            dacts[A_ATTN] + l*BHTT, acts[A_QKV] + l*BT3D,
            dacts[A_ATTN_OUT] + l*BTD, B, T, H, D, HD);
        attention_d_softmax<<<B*H*T, 32, 0, stream>>>(
            dacts[A_ATTN_PRE] + l*BHTT, dacts[A_ATTN] + l*BHTT,
            acts[A_ATTN] + l*BHTT, B, H, T);
        attention_dq<<<GRID(BTD), 0, stream>>>(
            dacts[A_QKV] + l*BT3D, dacts[A_ATTN_PRE] + l*BHTT,
            acts[A_QKV] + l*BT3D, B, T, H, D, HD);
        attention_dk<<<GRID(BTD), 0, stream>>>(
            dacts[A_QKV] + l*BT3D, dacts[A_ATTN_PRE] + l*BHTT,
            acts[A_QKV] + l*BT3D, B, T, H, D, HD);

        matmul_backward(
            dacts[A_LN1] + l*BTD,
            grads[P_QKV_W] + l*3*D*D, grads[P_QKV_B] + l*3*D,
            dacts[A_QKV] + l*BT3D, acts[A_LN1] + l*BTD,
            params[P_QKV_W] + l*3*D*D,
            BT, D, 3*D, cublas, stream);

        layernorm_backward<<<BT, 32, 0, stream>>>(
            d_upstream,
            grads[P_LN1_W] + l*D, grads[P_LN1_B] + l*D, dacts[A_LN1] + l*BTD,
            upstream, params[P_LN1_W] + l*D,
            acts[A_LN1_MEAN] + l*BT, acts[A_LN1_RSTD] + l*BT, BT, D);
    }

    // Embedding backward — scatter dacts[A_ENCODED] into wte and wpe.
    encoder_backward<<<GRID(BTD), 0, stream>>>(
        grads[P_TOK_EMB], grads[P_POS_EMB], dacts[A_ENCODED], pixel, B, T, D);
}

// =============================================================================
// SECTION 5 — initialisation and data loading.
// =============================================================================

// Init weights as N(0, 0.02), biases as 0, LayerNorm gammas as 1.
// Same RNG seed on every rank so all GPUs start from identical parameters
// without needing an MPI_Bcast.
static void init_parameters(
    float *d_params, const size_t *sizes, size_t total, unsigned seed)
{
    enum { WEIGHT, BIAS, LN_GAMMA };
    static const int kind[NUM_PARAMS] = {
        WEIGHT, WEIGHT,                   // tok_emb, pos_emb
        LN_GAMMA, BIAS,                   // ln1_w, ln1_b
        WEIGHT, BIAS,                     // qkv_w, qkv_b
        WEIGHT, BIAS,                     // attproj_w, attproj_b
        LN_GAMMA, BIAS,                   // ln2_w, ln2_b
        WEIGHT, BIAS,                     // fc1_w, fc1_b
        WEIGHT, BIAS,                     // fc2_w, fc2_b
        LN_GAMMA, BIAS,                   // lnf_w, lnf_b
        WEIGHT                            // head
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

// Reads the Kaggle "digit-recognizer" CSV layout: header row, then
// `label, pixel0, pixel1, ..., pixel783` per sample.
static int load_mnist_csv(
    const char *path, std::vector<uint8_t> &pixels, std::vector<int> &labels)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    static char line[8192];
    fgets(line, sizeof(line), f);                       // header
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        labels.push_back((int)strtol(p, &p, 10));
        for (int i = 0; i < 784; i++) {
            p++;                                         // skip comma
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
    // MPI / CUDA / cuBLAS / NCCL bootstrap.
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

    // Arguments.
    const char *csv_path = (argc > 1) ? argv[1] : "data/train.csv";
    int   steps          = (argc > 2) ? atoi(argv[2]) : 200;
    int   B              = (argc > 3) ? atoi(argv[3]) : 8;
    float lr             = (argc > 4) ? (float)atof(argv[4]) : 0.05f;

    Cfg cfg = { /*vocab*/ 256, /*seq*/ 28*28, /*layers*/ 2,
                /*dim*/ 64,    /*heads*/ 4,    /*head_dim*/ 16,
                /*classes*/ 10 };
    int BT = B * cfg.seq;

    // Load MNIST on every rank (data fits in RAM trivially).
    std::vector<uint8_t> pixels;
    std::vector<int>     labels;
    int N = load_mnist_csv(csv_path, pixels, labels);
    if (!N) {
        if (rank == 0) fprintf(stderr, "cannot read %s\n", csv_path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Compute total sizes and allocate flat buffers.
    size_t param_sz[NUM_PARAMS], act_sz[NUM_ACTS];
    fill_param_sizes(param_sz, cfg);
    fill_act_sizes(act_sz, cfg, B);

    size_t total_params = 0, total_acts = 0;
    for (int i = 0; i < NUM_PARAMS; i++) total_params += param_sz[i];
    for (int i = 0; i < NUM_ACTS;   i++) total_acts   += act_sz[i];

    float *d_params, *d_grads, *d_momentum;
    CHECK(cudaMalloc(&d_params,   total_params * sizeof(float)));
    CHECK(cudaMalloc(&d_grads,    total_params * sizeof(float)));
    CHECK(cudaMalloc(&d_momentum, total_params * sizeof(float)));
    CHECK(cudaMemset(d_momentum, 0, total_params * sizeof(float)));

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

    // Input/target staging buffers (pinned host + device).
    int *d_pixel, *d_target;
    int *h_pixel, *h_target;
    CHECK(cudaMalloc(&d_pixel,  BT * sizeof(int)));
    CHECK(cudaMalloc(&d_target, B  * sizeof(int)));
    CHECK(cudaMallocHost(&h_pixel,  BT * sizeof(int)));
    CHECK(cudaMallocHost(&h_target, B  * sizeof(int)));

    // Small scalars used only for reporting.
    float *d_loss_sum; int *d_correct;
    CHECK(cudaMalloc(&d_loss_sum, sizeof(float)));
    CHECK(cudaMalloc(&d_correct,  sizeof(int)));

    if (rank == 0) {
        printf("ranks=%d N=%d B=%d T=%d L=%d D=%d H=%d C=%d "
               "params=%zu steps=%d lr=%g\n",
               world, N, B, cfg.seq, cfg.layers, cfg.dim, cfg.heads,
               cfg.classes, total_params, steps, lr);
        fflush(stdout);
    }

    // Each rank samples its own random batches.
    std::mt19937 rng(42 + 1000u * rank);
    std::uniform_int_distribution<int> sampler(0, N - 1);

    auto t_start = std::chrono::steady_clock::now();

    for (int step = 1; step <= steps; step++) {
        // Sample a batch on the host, then async-copy to device.
        for (int b = 0; b < B; b++) {
            int j = sampler(rng);
            h_target[b] = labels[j];
            for (int t = 0; t < cfg.seq; t++)
                h_pixel[b*cfg.seq + t] = (int)pixels[(size_t)j*cfg.seq + t];
        }
        CHECK(cudaMemcpyAsync(d_pixel,  h_pixel,  BT*sizeof(int),
                              cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(d_target, h_target, B *sizeof(int),
                              cudaMemcpyHostToDevice, stream));

        // Zero gradient buffers (param-grads accumulate via atomicAdd for some
        // tensors; activation-grads are partly accumulated via residual stream).
        CHECK(cudaMemsetAsync(d_grads, 0, total_params * sizeof(float), stream));
        CHECK(cudaMemsetAsync(d_dacts, 0, total_acts   * sizeof(float), stream));

        model_forward(params, acts, d_pixel, cfg, B, cublas, stream);
        softmax_ce_forward<<<B, 32, 0, stream>>>(
            acts[A_PROBS], acts[A_LOSSES], acts[A_LOGITS], d_target, B, cfg.classes);
        model_backward(params, grads, acts, dacts,
                       d_pixel, d_target, cfg, B, cublas, stream);

        // Reduce gradients across ranks (no-op for world == 1).
        if (world > 1)
            CHECK(ncclAllReduce(d_grads, d_grads, total_params,
                                ncclFloat, ncclSum, nccl, stream));

        // SGD over the entire flat parameter buffer in one launch.
        float scale = 1.0f / (float)(B * world);
        sgd_step<<<GRID((int)total_params), 0, stream>>>(
            d_params, d_grads, d_momentum,
            (int)total_params, lr, /*momentum*/ 0.9f, scale);

        // Periodic loss + accuracy.
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

            float mean_loss   = h_loss / B;
            int   global_corr = h_corr, global_n = B;
            MPI_Allreduce(MPI_IN_PLACE, &mean_loss,   1, MPI_FLOAT,
                          MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &global_corr, 1, MPI_INT,
                          MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &global_n,    1, MPI_INT,
                          MPI_SUM, MPI_COMM_WORLD);
            mean_loss /= world;
            if (rank == 0)
                printf("step %4d | loss %.4f | acc %.3f\n",
                       step, mean_loss, (float)global_corr / global_n);
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

    cudaFreeHost(h_pixel); cudaFreeHost(h_target);
    cudaFree(d_pixel);     cudaFree(d_target);
    cudaFree(d_acts);      cudaFree(d_dacts);
    cudaFree(d_params);    cudaFree(d_grads); cudaFree(d_momentum);
    cudaFree(d_loss_sum);  cudaFree(d_correct);
    ncclCommDestroy(nccl);
    cublasDestroy(cublas);
    cudaStreamDestroy(stream);
    MPI_Finalize();
    return 0;
}
