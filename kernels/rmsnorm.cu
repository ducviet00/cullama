
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

__device__ __forceinline__ double warpReduceSum(double val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(FULL_MASK, val, offset);
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

__device__ __forceinline__ double blockReduceSum(double val) {
    static __shared__ double shared[WARP_SIZE];  // Shared mem for 32 partial sums
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
    CHECK_CUDA(cudaGetLastError());
}

void rmsnorm_fp16_gpu(float *o, float *x, float *weight, int size) {

    // float *q_d, *att_d, *xb_d, *key_cache_d, *value_cache_d;
    // RunState *s_d;
    half *o_d, *x_d, *weight_d;
    CHECK_CUDA(cudaMalloc(&o_d, sizeof(half) * size));
    CHECK_CUDA(cudaMalloc(&x_d, sizeof(half) * size));
    CHECK_CUDA(cudaMalloc(&weight_d, sizeof(half) * size));

    CHECK_CUDA(cudaMemcpy(x_d, float_to_half(x, size), sizeof(half) * size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(weight_d, float_to_half(weight, size), sizeof(half) * size, cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; ++i) {
        rmsnorm(o_d, x_d, weight_d, size);
    }

    int num_runs = 25;
    float elapsed_time = 0.0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        rmsnorm(o_d, x_d, weight_d, size);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("GPU RMSNORM FP16 - Time Avg: %f ms\n", (elapsed_time / num_runs));

    half *o_h;
    CHECK_CUDA(cudaMallocHost(&o_h, sizeof(half) * size));
    CHECK_CUDA(cudaMemcpy(o_h, o_d, sizeof(half) * size, cudaMemcpyDeviceToHost));

    half_to_float(o_h, o, size);
    CHECK_CUDA(cudaFree(o_d));
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(weight_d));
}

__global__ void rmsnorm_kernel(float *o, float *x, float *weight, int size) {
    const int thread_id = threadIdx.x;
    const int block_size = blockDim.x;

    float ss = 0.0F;
    for (int elem_id = thread_id; elem_id < size; elem_id += block_size)
        ss += x[elem_id] * x[elem_id];

    ss = blockReduceSum(ss);

    // serialization point to calculate normalization factor
    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    ss = shared_ss;

    // normalize and scale
    for (int elem_id = thread_id; elem_id < size; elem_id += block_size)
        o[elem_id] = weight[elem_id] * (ss * x[elem_id]);
}

void rmsnorm(float *o, float *x, float *weight, int size) {
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim(1);
    rmsnorm_kernel<<<gridDim, blockDim>>>(o, x, weight, size);
    // CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

void rmsnorm_fp32_gpu(float *o, float *x, float *weight, int size) {

    // float *q_d, *att_d, *xb_d, *key_cache_d, *value_cache_d;
    // RunState *s_d;
    float *o_d, *x_d, *weight_d;
    CHECK_CUDA(cudaMalloc(&o_d, sizeof(float) * size));
    CHECK_CUDA(cudaMalloc(&x_d, sizeof(float) * size));
    CHECK_CUDA(cudaMalloc(&weight_d, sizeof(float) * size));

    CHECK_CUDA(cudaMemcpy(x_d, x, sizeof(float) * size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(weight_d, weight, sizeof(float) * size, cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; ++i) {
        rmsnorm(o_d, x_d, weight_d, size);
    }

    int num_runs = 25;
    float elapsed_time = 0.0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        rmsnorm(o_d, x_d, weight_d, size);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("GPU RMSNORM FP32 - Time Avg: %f ms\n", (elapsed_time / num_runs));

    CHECK_CUDA(cudaMemcpy(o, o_d, sizeof(float) * size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(o_d));
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(weight_d));
}

void rmsnorm_fp32_cpu(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void test_attn(bool verbose) {
    printf("RMSNorm validating...\n");

    int dim = 768;

    float *x = alloc_mat(dim, 1);
    float *w = alloc_mat(dim, 1);

    rand_mat(x, dim, 1);
    rand_mat(w, dim, 1);

    x = half_to_float(float_to_half(x, dim), dim);
    w = half_to_float(float_to_half(w, dim), dim);

    float *o_fp32_cpu = alloc_mat(dim, 1);
    rmsnorm_fp32_cpu(o_fp32_cpu, x, w, dim);

    float *o_fp32_gpu = alloc_mat(dim, 1);
    rmsnorm_fp32_gpu(o_fp32_gpu, x, w, dim);

    float *o_fp16_gpu = alloc_mat(dim, 1);
    rmsnorm_fp16_gpu(o_fp16_gpu, x, w, dim);

    printf("host fp32: \n");
    print_mat(o_fp32_cpu, 1, dim);
    printf("\n");

    printf("device fp32: \n");
    print_mat(o_fp32_gpu, 1, dim);
    vec_diff_check(o_fp32_cpu, o_fp32_gpu, dim, verbose);
    printf("\n");

    printf("device fp16: \n");
    print_mat(o_fp16_gpu, 1, dim);
    vec_diff_check(o_fp32_cpu, o_fp16_gpu, dim, verbose);
}


int main(int argc, char **argv) {
    // Seed the random number generator with the current time
    srand(time(NULL));
    bool verbose = atoi(argv[0]);
    test_attn(verbose);
}
