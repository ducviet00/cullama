/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <fcntl.h>

#include <unistd.h>
#include <sys/mman.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX_SEQ_LEN 1024
#define DIVUP(x, y) ((x + y - 1) / y)
#define BLOCK_SIZE 32
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t status_ = call;                                             \
        if (status_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__,  \
                    cudaGetErrorName(status_), cudaGetErrorString(status_));    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    half* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    half* rms_att_weight; // (layer, dim) rmsnorm weights
    half* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    half* wq; // (layer, dim, n_heads * head_size)
    half* wk; // (layer, dim, n_kv_heads * head_size)
    half* wv; // (layer, dim, n_kv_heads * head_size)
    half* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    half* w1; // (layer, hidden_dim, dim)
    half* w2; // (layer, dim, hidden_dim)
    half* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    half* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    half* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    half *x; // activation at current time stamp (dim,)
    half *xb; // same, but inside a residual branch (dim,)
    half *xb2; // an additional buffer just for convenience (dim,)
    half *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    half *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    half *q; // query (dim,)
    half *k; // key (dim,)
    half *v; // value (dim,)
    half *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    half *logits_fp16; // output logits
    half *logits_d; // output logits on GPU
    // kv cache
    half* key_cache;   // (layer, seq_len, dim)
    half* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    half* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;


void malloc_run_state(RunState *s, Config *p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    CHECK_CUDA(cudaMalloc((void **)&s->x, p->dim * sizeof(half)));
    CHECK_CUDA(cudaMalloc((void **)&s->xb, p->dim * sizeof(half)));
    CHECK_CUDA(cudaMalloc((void **)&s->xb2, p->dim * sizeof(half)));
    CHECK_CUDA(cudaMalloc((void **)&s->hb, p->hidden_dim * sizeof(half)));
    CHECK_CUDA(cudaMalloc((void **)&s->hb2, p->hidden_dim * sizeof(half)));
    CHECK_CUDA(cudaMalloc((void **)&s->q, p->dim * sizeof(half)));
    CHECK_CUDA(cudaMalloc((void **)&s->key_cache, p->n_layers * p->seq_len * kv_dim * sizeof(half)));
    CHECK_CUDA(cudaMalloc((void **)&s->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(half)));

    // CHECK_CUDA(cudaMalloc(&s->p_pos, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&s->logits_d, p->vocab_size * sizeof(half)));
    CHECK_CUDA(cudaMallocHost(&s->logits, p->vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&s->logits_fp16, p->vocab_size * sizeof(half)));
    CHECK_CUDA(cudaDeviceSynchronize());
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->hb || !s->q || !s->key_cache || !s->value_cache || !s->logits) {
        fprintf(stderr, "cudaMalloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    CHECK_CUDA(cudaFree(s->x));
    CHECK_CUDA(cudaFree(s->xb));
    CHECK_CUDA(cudaFree(s->xb2));
    CHECK_CUDA(cudaFree(s->hb));
    CHECK_CUDA(cudaFree(s->hb2));
    CHECK_CUDA(cudaFree(s->q));
    CHECK_CUDA(cudaFree(s->key_cache));
    CHECK_CUDA(cudaFree(s->value_cache));
    CHECK_CUDA(cudaFree(s->logits_d));
    CHECK_CUDA(cudaFreeHost(s->logits));
    CHECK_CUDA(cudaFreeHost(s->logits_fp16));
}

void memory_map_weights(TransformerWeights *w, Config* p, half* ptr) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    w->wcls = ptr;
}

void print_mat(const half *m, int R, int C) {
    for (int i = 0; i < MIN(R, 10); ++i) {
        for (int j = 0; j < MIN(C, 10); ++j) {
            printf("%+.6f ", __half2float(m[i * C + j]));
        }
        printf("\n");
    }
}

void print_mat_gpu(const half *m, int R, int C) {
    half *m_cpu;
    CHECK_CUDA(cudaMallocHost(&m_cpu, R * C * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(m_cpu, m, R * C * sizeof(half), cudaMemcpyDeviceToHost));
    print_mat(m_cpu, R, C);
    CHECK_CUDA(cudaFreeHost(m_cpu));

}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, half** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = (half *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    half *weights_cpu_ptr = *data + sizeof(Config)/sizeof(half);
    half *weights_gpu_ptr;
    size_t weights_size = *file_size - sizeof(Config);
    printf("weights_size: %ld\n", weights_size);
    CHECK_CUDA(cudaMalloc((void **)&weights_gpu_ptr, weights_size));
    CHECK_CUDA(cudaMemcpy(weights_gpu_ptr, weights_cpu_ptr, weights_size, cudaMemcpyHostToDevice));
    print_mat(weights_cpu_ptr + 5238222299, 1, 20);
    memory_map_weights(weights, config, weights_gpu_ptr);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    CHECK_CUDA(cudaFree(t->weights.token_embedding_table));
    free_run_state(&t->state);
}


void half_to_float(const half *hm, float *fm, int size) {
    for (int i = 0; i < size; ++i) {
        fm[i] = __half2float(hm[i]);
    }
}

// ----------------------------------------------------------------------------
// CUDA kernels

__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    return val;
}

__device__ __forceinline__ half warpReduceSum(half val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(FULL_MASK, val, offset);
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

__global__ void rmsnorm_kernel(half *o, half *x, half *weight, int size) {
    const int thread_id = threadIdx.x;
    const int block_size = blockDim.x;

    float ss = 0.0F;
    for (int elem_id = thread_id; elem_id < size; elem_id += block_size)
        ss += __half2float(x[elem_id]) * __half2float(x[elem_id]);

    ss = blockReduceSum(ss);

    // serialization point to calculate normalization factor
    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= size;
        ss += 1e-5f;
        shared_ss = rsqrtf(ss);
    }
    __syncthreads();
    ss = shared_ss;

    // normalize and scale
    for (int elem_id = thread_id; elem_id < size; elem_id += block_size)
        o[elem_id] = __float2half(__half2float(weight[elem_id]) * (ss * __half2float(x[elem_id])));
}


void rmsnorm(half *o, half *x, half *weight, int size) {
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim(1);
    rmsnorm_kernel<<<gridDim, blockDim>>>(o, x, weight, size);
    // CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
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

// void softmax_gpu(float* x, int size) {
//     dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
//     dim3 gridDim(1);
//     softmax_kernel<<<gridDim, blockDim>>>(x, size);
//     CHECK_CUDA(cudaGetLastError());
// }


__global__ void gemv_fp16_kernel_pack8(
    half *xout,
    const half *__restrict__ x,
    const half *__restrict__ W,
    int n,
    int d,
    int vectorized
) {
    int didx = blockIdx.x * blockDim.y + threadIdx.y;
    // int nidx = blockIdx.y;
    if (didx >= d)
        return;

    const float4 *W_vec = reinterpret_cast<const float4*>(W + didx * n);
    const float4 *x_vec = reinterpret_cast<const float4*>(x);

    float sum = 0;

#pragma unroll
    for (int i = 0; i < vectorized; i++) {
        int j = (i * WARP_SIZE + threadIdx.x);
        if (j < n / 8) {
            float4 x4 = x_vec[j];
            float4 w4 = W_vec[j];
            const half2* x_h1 = (half2*)&x4.x;
            const half2* x_h2 = (half2*)&x4.y;
            const half2* x_h3 = (half2*)&x4.z;
            const half2* x_h4 = (half2*)&x4.w;

            const half2* w_h1 = (half2*)&w4.x;
            const half2* w_h2 = (half2*)&w4.y;
            const half2* w_h3 = (half2*)&w4.z;
            const half2* w_h4 = (half2*)&w4.w;

            sum += __half2float(x_h1->x * w_h1->x);
            sum += __half2float(x_h2->x * w_h2->x);
            sum += __half2float(x_h3->x * w_h3->x);
            sum += __half2float(x_h4->x * w_h4->x);
            sum += __half2float(x_h1->y * w_h1->y);
            sum += __half2float(x_h2->y * w_h2->y);
            sum += __half2float(x_h3->y * w_h3->y);
            sum += __half2float(x_h4->y * w_h4->y);
        }
    }

    sum = warpReduceSum(sum);
    if (threadIdx.x == 0)
        xout[didx] = __float2half(sum);
}

void matmul(half *xout, half *x, half *w, int n, int d) {
    // assert((reinterpret_cast<uintptr_t>(x) & 0x3) == 0);
    // assert((reinterpret_cast<uintptr_t>(w) & 0x3) == 0);

    int groupsize = DIVUP(n, WARP_SIZE);
    int vectorized = DIVUP(groupsize, 2);
    dim3 blockDim(WARP_SIZE, 16);
    dim3 gridDim(DIVUP(d, 16));
    gemv_fp16_kernel_pack8<<<gridDim, blockDim>>>(xout, x, w, n, d, vectorized);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void swiglu_kernel(half *shb, half *shb2, int hidden_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_dim) {
        float val = __half2float(shb[i]);
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        // elementwise multiply with w3(x)
        val *= __half2float(shb2[i]);
        shb[i] = __float2half(val);
    }
}

void swiglu(half *shb, half *shb2, int hidden_dim) {
    // SwiGLU non-linearity
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim(DIVUP(hidden_dim, BLOCK_SIZE * BLOCK_SIZE));
    swiglu_kernel<<<gridDim, blockDim>>>(shb, shb2, hidden_dim);
    CHECK_CUDA(cudaGetLastError());
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


void multihead_attention(Config *p, RunState *s,
                        int pos, int kv_dim, int kv_mul, int head_size, int loff) {
    int n_heads = p->n_heads;
    // printf("n_heads: %d\n", n_heads);
    multihead_attention_fp16_kernel<<<n_heads, BLOCK_SIZE * BLOCK_SIZE>>>(pos, s->q, s->xb, s->key_cache, s->value_cache, kv_dim, kv_mul, head_size, loff, 1.0 / sqrt(head_size));
    CHECK_CUDA(cudaGetLastError());
}


__global__ void rotary_embedding_kernel(int pos, int dim, half *sq, half *sk, int kv_dim, int head_size) {
    for (int j = threadIdx.x; j < dim / 2; j += blockDim.x)
    {
        int i = j * 2;
        int head_dim = i % head_size;
        double freq = 1.0f / __powf(10000.0f, head_dim / (float)head_size);
        double val = pos * freq;
        double fcr = cos(val);
        double fci = sin(val);
        int rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
        for (int v = 0; v < rotn; v++) {
            half *vec = v == 0 ? sq : sk;  // the vector to rotate (query or key)
            half v0 = vec[i];
            half v1 = vec[i + 1];
            vec[i] = __float2half(__half2float(v0) * fcr - __half2float(v1) * fci);
            vec[i + 1] = __float2half(__half2float(v0) * fci + __half2float(v1) * fcr);
        }
    }
}


void rotary_embedding(RunState *s, int pos, int dim, int kv_dim, int head_size) {
    // TODO: dim / 2 exceed limit of 1024 threads per block
    dim3 blockDim(MIN(dim / 2, BLOCK_SIZE * BLOCK_SIZE));
    dim3 gridDim(1);
    rotary_embedding_kernel <<< gridDim, blockDim>>> (pos, dim, s->q, s->k, kv_dim, head_size);
    CHECK_CUDA(cudaGetLastError());
}


__global__ void skip_conn_kernel(half *x, half *y, int dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        x[i] += y[i];
    }
}

void skip_conn(half *x, half *y, int dim) {
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim(DIVUP(dim, BLOCK_SIZE * BLOCK_SIZE));
    skip_conn_kernel<<<gridDim, blockDim>>>(x, y, dim);
    CHECK_CUDA(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

// void rmsnorm(float* o, float* x, float* weight, int size) {
//     // calculate sum of squares
//     float ss = 0.0f;
//     for (int j = 0; j < size; j++) {
//         ss += x[j] * x[j];
//     }
//     ss /= size;
//     ss += 1e-5f;
//     ss = 1.0f / sqrtf(ss);
//     // normalize and scale
//     for (int j = 0; j < size; j++) {
//         o[j] = weight[j] * (ss * x[j]);
//     }
// }

void softmax(float* x, int size) {
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

// void matmul(float* xout, float* x, float* w, int n, int d) {
//     // W (d,n) @ x (n,) -> xout (d,)
//     // by far the most amount of time is spent inside this little function
//     int i;
//     #pragma omp parallel for private(i)
//     for (i = 0; i < d; i++) {
//         float val = 0.0f;
//         for (int j = 0; j < n; j++) {
//             val += w[i * n + j] * x[j];
//         }
//         xout[i] = val;
//     }
// }

void check_pointer_location(void* ptr) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
        return;
    }

    if (attributes.type == cudaMemoryTypeDevice) {
        printf("Pointer is on the GPU.\n");
    } else if (attributes.type == cudaMemoryTypeHost) {
        printf("Pointer is on the host.\n");
    } else {
        printf("Pointer location is unknown.\n");
    }
}


float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    // print all config
    // printf("dim: %d\n", p->dim);
    // printf("n_heads: %d\n", p->n_heads);
    // printf("n_kv_heads: %d\n", p->n_kv_heads);
    // printf("hidden_dim: %d\n", p->hidden_dim);
    // printf("seq_len: %d\n", p->seq_len);
    // printf("n_layers: %d\n", p->n_layers);
    // printf("vocab_size: %d\n", p->vocab_size);


    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    half *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    // copy the token embedding into x
    half* content_row = w->token_embedding_table + token * dim;
    // check_pointer_location(x);
    // check_pointer_location(content_row);
    CHECK_CUDA(cudaMemcpy(x, content_row, dim * sizeof(*x), cudaMemcpyDeviceToDevice));
    // CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaDeviceSynchronize());

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        // check_pointer_location(x);
        // check_pointer_location(s->xb);
        // check_pointer_location(w->rms_att_weight);
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("rmsnorm rms_att_weight\n");
        // print_mat_gpu(w->rms_att_weight + l*dim, 1, 20);
        // printf("rmsnorm\n");
        // print_mat_gpu(s->xb, 1, 20);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("matmul 1\n");
        // print_mat_gpu(s->q, 1, 20);

        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("matmul 2\n");
        // print_mat_gpu(s->k, 1, 20);

        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("matmul 3\n");
        // print_mat_gpu(s->v, 1, 20);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        rotary_embedding(s, pos, dim, kv_dim, head_size);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("rotary_embedding\n");
        // print_mat_gpu(s->q, 1, 20);
        // print_mat_gpu(s->k, 1, 20);

        multihead_attention(p, s, pos, kv_dim, kv_mul, head_size, loff);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("multihead_attention\n");
        // print_mat_gpu(s->xb, 1, 20);

        // check_pointer_location(s->xb2);
        // check_pointer_location(s->xb);
        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("matmul 4\n");
        // print_mat_gpu(s->xb2, 1, 20);

        // residual connection back into x
        skip_conn(x, s->xb2, dim);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("skip_conn\n");
        // print_mat_gpu(x, 1, 20);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("rmsnorm 2\n");
        // print_mat_gpu(s->xb, 1, 20);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("matmul 5\n");
        // print_mat_gpu(s->hb, 1, 20);

        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("matmul 6\n");
        // print_mat_gpu(s->hb2, 1, 20);

        // SwiGLU non-linearity
        swiglu(s->hb, s->hb2, hidden_dim);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("swiglu\n");
        // print_mat_gpu(s->hb, 1, 20);

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("matmul 7\n");
        // print_mat_gpu(s->xb, 1, 20);

        // residual connection
        skip_conn(x, s->xb, dim);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("skip_conn 2\n");
        // print_mat_gpu(x, 1, 20);
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);
    // CHECK_CUDA(cudaDeviceSynchronize());
    // printf("final rmsnorm\n");
    // print_mat_gpu(x, 1, 20);

    // classifier into logits
    matmul(s->logits_d, x, w->wcls, p->dim, p->vocab_size);
    CHECK_CUDA(cudaMemcpy(s->logits_fp16, s->logits_d, p->vocab_size * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
    half_to_float(s->logits_fp16, s->logits, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = (char *)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = (ProbIndex *)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
