
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t status_ = call;                                             \
        if (status_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__,  \
                    cudaGetErrorName(status_), cudaGetErrorString(status_));    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)


#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX_SEQ_LEN 4096
#define DIVUP(x, y) ((x + y - 1) / y)
#define BLOCK_SIZE 32
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

float *alloc_mat(int R, int C) {
    float *m;
    CHECK_CUDA(cudaMallocHost(&m, sizeof(float) * R * C));
    return m;
}

half *alloc_mat_half(int R, int C) {
    half *m;
    CHECK_CUDA(cudaMallocHost(&m, sizeof(half) * R * C));
    return m;
}

void float_to_half(const float *fm, half *hm, int size) {
    for (int i = 0; i < size; ++i) {
        hm[i] = __float2half(fm[i]);
    }
}

half *float_to_half(const float *fm, int size) {
    half *hm = alloc_mat_half(1, size);
    for (int i = 0; i < size; ++i) {
        hm[i] = __float2half(fm[i]);
    }
    return hm;
}

void half_to_float(const half *hm, float *fm, int size) {
    for (int i = 0; i < size; ++i) {
        fm[i] = __half2float(hm[i]);
    }
}

float *half_to_float(const half *hm, int size) {
    float *fm = alloc_mat(1, size);
    for (int i = 0; i < size; ++i) {
        fm[i] = __half2float(hm[i]);
    }
    return fm;
}

void rand_mat(float *m, int R, int C) {
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            m[i * C + j] = (float)rand() / (float)RAND_MAX - 0.5;
        }
    }
}

void print_mat(float *m, int R, int C) {
    for (int i = 0; i < MIN(R, 10); ++i) {
        for (int j = 0; j < MIN(C, 10); ++j) {
            printf("%+.6f ", m[i * C + j]);
        }
        printf("\n");
    }
}

void vec_diff_check(float *original, float *changed, int size, bool verbose) {
    float max_diff = 0.0F;
    double total_diff = 0.0F;
    if (verbose) printf("size: %d\n", size);
    for (int i = 0; i < size; ++i) {
        if (i > size - 10 && verbose)
            printf("%d CPU: %f, GPU: %f\n", i, original[i], changed[i]);
        float diff = fabsf(original[i] - changed[i]);
        max_diff = fmaxf(max_diff, diff);
        total_diff += diff;
    }
    printf("Comparing CPU and GPU: Max diff %f, Avg diff: %f\n", max_diff, total_diff / size);
}


__device__ __forceinline__ float warpReduceMax(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, offset));
    return val;
}

__device__ __forceinline__ float blockReduceMax(float val) {
    static __shared__ float shared[WARP_SIZE];  // Shared mem for 32 partial sums
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceMax(val);  // Each warp performs partial reduction

    if (lane == 0)
        shared[wid] = val;  // Write reduced value to shared memory

    __syncthreads();  // Wait for all partial reductions

    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (wid == 0)
        val = warpReduceMax(val);  // Final reduce within first warp

    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    return val;
}

__device__ __forceinline__ half warpReduceSum(half val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = __hadd(val, __shfl_xor_sync(FULL_MASK, val, offset));
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[WARP_SIZE];  // Shared mem for 32 partial sums
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);  // Each warp performs partial reduction

    if (lane == 0)
        shared[wid] = val;  // Write reduced value to shared memory

    __syncthreads();  // Wait for all partial reductions

    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (wid == 0)
        val = warpReduceSum(val);  // Final reduce within first warp

    return val;
}

__device__ void softmax_kernel(float *x, int size) {
    const int thread_id = threadIdx.x;
    const int block_size = blockDim.x;

    __shared__ float max_val;
    __shared__ float d_total;

    float m_partial = -FLT_MAX;
    for (int elem_id = thread_id; elem_id < size; elem_id += block_size)
        m_partial = fmaxf(m_partial, x[elem_id]);

    m_partial = blockReduceMax(m_partial);

    if (thread_id == 0)
        max_val = m_partial;
    __syncthreads();

    float d_partial = 0.0f;
    for (int elem_id = thread_id; elem_id < size; elem_id += block_size) {
        x[elem_id] = __expf(x[elem_id] - max_val);
        d_partial += x[elem_id];
    }

    d_partial = blockReduceSum(d_partial);
    if (thread_id == 0)
        d_total = d_partial;
    __syncthreads();

    for (int elem_id = thread_id; elem_id < size; elem_id += block_size)
        x[elem_id] /= d_total;
}


__global__ void multihead_attention_fp16_kernel(int pos, half *sq, half *sxb, half *key_cache, half *value_cache,
                                           int kv_dim, int kv_mul, int head_size,
                                           int loff, float scale) {
    int h = blockIdx.x;
    // get the query vector for this head
    const half *q = sq + h * head_size;
    // attention scores for this head
    __shared__ float att[MAX_SEQ_LEN];
    __shared__ float max_val;
    __shared__ float inv_d_total;
    float4 a, b;
    // float *att = satt + h * seq_len;
    // iterate over all timesteps, including the current one
    // In CUDA, each thread does a small portion of the calc
    // attention scores for this head
    float m_partial = -FLT_MAX;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        // get the key vector for this head and at this timestep
        half *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        // half2 score = __float2half2_rn(0.0f);
        double score = 0.0f;
#pragma unroll
        for (int i = 0; i < head_size; i += 8) {
            a = *((float4 *)(&q[i]));
            b = *((float4 *)(&k[i]));
            const __half2* a_h1 = (__half2*)&a.x;
            const __half2* a_h2 = (__half2*)&a.y;
            const __half2* a_h3 = (__half2*)&a.z;
            const __half2* a_h4 = (__half2*)&a.w;

            const __half2* b_h1 = (__half2*)&b.x;
            const __half2* b_h2 = (__half2*)&b.y;
            const __half2* b_h3 = (__half2*)&b.z;
            const __half2* b_h4 = (__half2*)&b.w;

            score += __half2float(a_h1->x * b_h1->x);
            score += __half2float(a_h2->x * b_h2->x);
            score += __half2float(a_h3->x * b_h3->x);
            score += __half2float(a_h4->x * b_h4->x);
            score += __half2float(a_h1->y * b_h1->y);
            score += __half2float(a_h2->y * b_h2->y);
            score += __half2float(a_h3->y * b_h3->y);
            score += __half2float(a_h4->y * b_h4->y);
        }
        // save the score to the attention buffer
        // att[t] = __half2float(__hadd(score.x, score.y)) * scale;
        att[t] = score * scale;
        m_partial = fmaxf(m_partial, att[t]);
    }

    // softmax the scores to get attention weights, from 0..pos inclusively
    if (threadIdx.x == 0)
        max_val = m_partial;
    __syncthreads();

    double d_partial = 0.0f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        att[t] = __expf(att[t] - max_val);
        d_partial += att[t];
    }

    d_partial = blockReduceSum(d_partial);
    if (threadIdx.x == 0)
        inv_d_total =  __frcp_rn(d_partial);
    __syncthreads();

    // weighted sum of the values, store back into xb
    // NOTE: by swapping the order of the for loops (vs. C) a simpler
    // version of the code accomplishes the same task and fits more
    // naturally with the CUDA way of subdividing the problem.

    half *xb = sxb + h * head_size;
    value_cache = value_cache + loff;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
        half *vv = value_cache + (h / kv_mul) * head_size;
    #pragma unroll
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            half *v = vv + t * kv_dim;
            // get the attention weight for this timestep
            val += att[t] * __half2float(v[i]) * inv_d_total;
        }
        xb[i] = __float2half(val);
    }
}

void multihead_attention_fp16(half *q, half *xout, half *k_cache, half *v_cache,
                        int n_heads, int pos, int kv_dim, int kv_mul, int head_size, int loff) {
    multihead_attention_fp16_kernel<<<n_heads, BLOCK_SIZE * BLOCK_SIZE>>>(pos, q, xout, k_cache, v_cache, kv_dim, kv_mul, head_size, loff, 1.0 / sqrt(head_size));
    CHECK_CUDA(cudaGetLastError());
}

void multihead_attention_fp16_gpu(float *q, float *xout, float *k_cache, float *v_cache,
                                int n_heads, int n_layers, int seq_len, int dim, int pos,
                                int kv_dim, int kv_mul, int head_size, int loff) {

    // float *q_d, *att_d, *xb_d, *key_cache_d, *value_cache_d;
    // RunState *s_d;
    half *q_d, *xout_d, *k_cache_d, *v_cache_d;
    CHECK_CUDA(cudaMalloc(&q_d, sizeof(half) * dim));
    CHECK_CUDA(cudaMalloc(&xout_d, sizeof(half) * dim));
    CHECK_CUDA(cudaMalloc(&k_cache_d, sizeof(half) * n_layers * seq_len * kv_dim));
    CHECK_CUDA(cudaMalloc(&v_cache_d, sizeof(half) * n_layers * seq_len * kv_dim));

    CHECK_CUDA(cudaMemcpy(q_d, float_to_half(q, dim), sizeof(half) * dim, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(k_cache_d, float_to_half(k_cache, n_layers * seq_len * kv_dim), sizeof(half) * n_layers * seq_len * kv_dim, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(v_cache_d, float_to_half(v_cache, n_layers * seq_len * kv_dim), sizeof(half) * n_layers * seq_len * kv_dim, cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; ++i) {
        multihead_attention_fp16(q_d, xout_d, k_cache_d, v_cache_d, n_heads, pos, kv_dim, kv_mul, head_size, loff);
    }

    int num_runs = 25;
    float elapsed_time = 0.0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        multihead_attention_fp16(q_d, xout_d, k_cache_d, v_cache_d, n_heads, pos, kv_dim, kv_mul, head_size, loff);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("GPU MHA FP16 - Time Avg: %f ms\n", (elapsed_time / num_runs));

    half *xout_h;
    CHECK_CUDA(cudaMallocHost(&xout_h, sizeof(half) * dim));
    CHECK_CUDA(cudaMemcpy(xout_h, xout_d, sizeof(half) * dim, cudaMemcpyDeviceToHost));

    half_to_float(xout_h, xout, dim);
    CHECK_CUDA(cudaFree(q_d));
    CHECK_CUDA(cudaFree(xout_d));
    CHECK_CUDA(cudaFree(k_cache_d));
    CHECK_CUDA(cudaFree(v_cache_d));
}

__global__ void multihead_attention_kernel(int pos, float *sq, float *sxb, float *key_cache, float *value_cache,
                                           int kv_dim, int kv_mul, int head_size,
                                           int loff, float scale) {
    int h = blockIdx.x;
    // get the query vector for this head
    const float *q = sq + h * head_size;
    // attention scores for this head
    __shared__ float att[MAX_SEQ_LEN];
    __shared__ float max_val;
    __shared__ float inv_d_total;
    float4 a, b;
    // float *att = satt + h * seq_len;
    // iterate over all timesteps, including the current one
    // In CUDA, each thread does a small portion of the calc
    // attention scores for this head
    float m_partial = -FLT_MAX;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        // get the key vector for this head and at this timestep
        float *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
#pragma unroll
        for (int i = 0; i < head_size; i += 4) {
            a = *((float4 *)(&q[i]));
            b = *((float4 *)(&k[i]));
            score += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        // save the score to the attention buffer
        att[t] = score * scale;
        m_partial = fmaxf(m_partial, att[t]);
    }

    // softmax the scores to get attention weights, from 0..pos inclusively
    if (threadIdx.x == 0)
        max_val = m_partial;
    __syncthreads();

    float d_partial = 0.0f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        att[t] = __expf(att[t] - max_val);
        d_partial += att[t];
    }

    d_partial = blockReduceSum(d_partial);
    if (threadIdx.x == 0)
        inv_d_total = 1.0f / d_partial;
    __syncthreads();

    // weighted sum of the values, store back into xb
    // NOTE: by swapping the order of the for loops (vs. C) a simpler
    // version of the code accomplishes the same task and fits more
    // naturally with the CUDA way of subdividing the problem.
    float *xb = sxb + h * head_size;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
#pragma unroll
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float *v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            val += att[t] * v[i] * inv_d_total;
        }
        xb[i] = val;
    }
}

void multihead_attention_fp32(float *q, float *xout, float *k_cache, float *v_cache,
                        int n_heads, int pos, int kv_dim, int kv_mul, int head_size, int loff)  {
    multihead_attention_kernel<<<n_heads, BLOCK_SIZE * BLOCK_SIZE>>>(pos, q, xout, k_cache, v_cache, kv_dim, kv_mul, head_size, loff, 1.0 / sqrt(head_size));
    CHECK_CUDA(cudaGetLastError());
}

void multihead_attention_fp32_gpu(float *q, float *xout, float *k_cache, float *v_cache,
                                int n_heads, int n_layers, int seq_len, int dim, int pos,
                                int kv_dim, int kv_mul, int head_size, int loff) {

    float *q_d, *xout_d, *k_cache_d, *v_cache_d;
    CHECK_CUDA(cudaMalloc(&q_d, sizeof(float) * dim));
    CHECK_CUDA(cudaMalloc(&xout_d, sizeof(float) * dim));
    CHECK_CUDA(cudaMalloc(&k_cache_d, sizeof(float) * n_layers * seq_len * kv_dim));
    CHECK_CUDA(cudaMalloc(&v_cache_d, sizeof(float) * n_layers * seq_len * kv_dim));

    CHECK_CUDA(cudaMemcpy(q_d, q, sizeof(float) * dim, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(k_cache_d, k_cache, sizeof(float) * n_layers * seq_len * kv_dim, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(v_cache_d, v_cache, sizeof(float) * n_layers * seq_len * kv_dim, cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; ++i) {
        multihead_attention_fp32(q_d, xout_d, k_cache_d, v_cache_d, n_heads, pos, kv_dim, kv_mul, head_size, loff);
    }

    int num_runs = 25;
    float elapsed_time = 0.0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        multihead_attention_fp32(q_d, xout_d, k_cache_d, v_cache_d, n_heads, pos, kv_dim, kv_mul, head_size, loff);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("GPU MHA FP32 - Time Avg: %f ms\n", (elapsed_time / num_runs));

    CHECK_CUDA(cudaMemcpy(xout, xout_d, sizeof(float) * dim, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(q_d));
    CHECK_CUDA(cudaFree(xout_d));
    CHECK_CUDA(cudaFree(k_cache_d));
    CHECK_CUDA(cudaFree(v_cache_d));
}

void softmax_cpu(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}


void multihead_attention_fp32_cpu(float *q, float *xout, float *k_cache, float *v_cache,
                            int n_heads, int n_layers, int seq_len, int dim, int pos,
                            int kv_dim, int kv_mul, int head_size, int loff) {
    int h;
    for (h = 0; h < n_heads; h++) {
        // get the query vector for this head
        float *att = alloc_mat(1, seq_len);
        float *_q = q + h * head_size;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float *k = k_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += _q[i] * k[i];
            }
            score /= sqrtf(head_size);
            // save the score to the attention buffer
            att[t] = score;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax_cpu(att, pos + 1);

        // weighted sum of the values, store back into xb
        float *xb = xout + h * head_size;
        memset(xb, 0, head_size * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float *v = v_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a = att[t];
            // accumulate the weighted value into xb
            for (int i = 0; i < head_size; i++) {
                xb[i] += a * v[i];
            }
        }
    }
}

void test_attn(bool verbose) {
    printf("Multihead attention validating...\n");

    int dim = 768;
    int n_layers = 2;
    int n_heads = 12;
    int n_kv_heads = 12;
    int seq_len = 1024;
    int kv_dim = (dim * n_kv_heads) / n_heads;
    int l = 1;
    int pos = 512;
    int kv_mul = n_heads / n_kv_heads;  // integer multiplier of the kv sharing in multiquery
    int head_size = dim / n_heads;
    int loff = l * seq_len * kv_dim;
    int kv_size = n_layers * seq_len * kv_dim;

    float *q = alloc_mat(dim, 1);
    float *k_cache = alloc_mat(kv_size, 1);
    float *v_cache = alloc_mat(kv_size, 1);

    rand_mat(q, dim, 1);
    rand_mat(k_cache, kv_size, 1);
    rand_mat(v_cache, kv_size, 1);

    q = half_to_float(float_to_half(q, dim), dim);
    k_cache = half_to_float(float_to_half(k_cache, kv_size), kv_size);
    v_cache = half_to_float(float_to_half(v_cache, kv_size), kv_size);

    float *xout_fp32_cpu = alloc_mat(dim, 1);
    multihead_attention_fp32_cpu(q, xout_fp32_cpu, k_cache, v_cache, n_heads, n_layers, seq_len, dim, pos, kv_dim, kv_mul, head_size, loff);

    float *xout_fp32_gpu = alloc_mat(dim, 1);
    multihead_attention_fp32_gpu(q, xout_fp32_gpu, k_cache, v_cache, n_heads, n_layers, seq_len, dim, pos, kv_dim, kv_mul, head_size, loff);

    float *xout_fp16_gpu = alloc_mat(dim, 1);
    multihead_attention_fp16_gpu(q, xout_fp16_gpu, k_cache, v_cache, n_heads, n_layers, seq_len, dim, pos, kv_dim, kv_mul, head_size, loff);

    printf("host fp32: \n");
    print_mat(xout_fp32_cpu, 1, dim);
    printf("\n");

    printf("device fp32: \n");
    print_mat(xout_fp32_gpu, 1, dim);
    vec_diff_check(xout_fp32_cpu, xout_fp32_gpu, dim, verbose);
    printf("\n");

    printf("device fp16: \n");
    print_mat(xout_fp16_gpu, 1, dim);
    vec_diff_check(xout_fp32_cpu, xout_fp16_gpu, dim, verbose);
}


int main(int argc, char **argv) {
    // Seed the random number generator with the current time
    srand(time(NULL));
    bool verbose = atoi(argv[0]);
    test_attn(verbose);
}
