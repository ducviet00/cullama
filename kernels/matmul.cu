#include <math.h>
#include <sys/time.h>
#include <time.h>

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cublas_v2.h>
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

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t err = call;                                              \
        if (err != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error in file '%s' at line %d: %s\n",       \
                    __FILE__, __LINE__, cublasGetErrorString(err));             \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Helper function to get the error string
const char* cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        default:
            return "Unknown cuBLAS error";
    }
}

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
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

__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    return val;
}

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

void matmul_gpu(half *xout, half *x, half *w, int n, int d) {
    int groupsize = DIVUP(n, WARP_SIZE);
    int vectorized = DIVUP(groupsize, 2);
    dim3 blockDim(WARP_SIZE, 16);
    dim3 gridDim(DIVUP(d, 16));
    gemv_fp16_kernel_pack8<<<gridDim, blockDim>>>(xout, x, w, n, d, vectorized);
    CHECK_CUDA(cudaGetLastError());
}

void matmul_fp16_gpu(float *xout, float *x, float *w, int n, int d) {
    half *xout_d, *x_d, *w_d;
    CHECK_CUDA(cudaMalloc(&xout_d, sizeof(half) * d));
    CHECK_CUDA(cudaMalloc(&x_d, sizeof(half) * n));
    CHECK_CUDA(cudaMalloc(&w_d, sizeof(half) * d * n));

    CHECK_CUDA(cudaMemcpy(x_d, float_to_half(x, n), sizeof(half) * n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(w_d, float_to_half(w, n * d), sizeof(half) * n * d, cudaMemcpyHostToDevice));

    // WARM UP
    for (int i=0; i <= 10; ++i) {
        matmul_gpu(xout_d, x_d, w_d, n, d);
    }

    int num_runs = 25;
    float elapsed_time = 0.0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        matmul_gpu(xout_d, x_d, w_d, n, d);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("GPU Matmul FP16 - Time Avg: %f ms\n", (elapsed_time / num_runs));
    printf("%f sec, GFLOPS: %f \n", elapsed_time, 2ll * n * d * num_runs / (elapsed_time * 1e6));

    half *xout_h;
    CHECK_CUDA(cudaMallocHost(&xout_h, sizeof(half) * d));
    CHECK_CUDA(cudaMemcpy(xout_h, xout_d, sizeof(half) * d, cudaMemcpyDeviceToHost));
    half_to_float(xout_h, xout, d);

    CHECK_CUDA(cudaFree(xout_d));
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(w_d));
}


__global__ void gemv_fp32_kernel(float *C, const float *__restrict__ B, const float *__restrict__ A,
                               int n, int d, int numSerialLoads) {
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;

    A += index * n;
    B += blockIdx.y * n;
    C += blockIdx.y * d;

    float sum = 0;
    float4 w;
    float4 inp;

#pragma unroll
    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * WARP_SIZE + threadIdx.x) * 4;
        if (j < n) {
            w = *((float4 *)(&A[j]));
            inp = *((float4 *)(&B[j]));
            sum += w.x * inp.x + w.y * inp.y + w.z * inp.z + w.w * inp.w;
        }
    }

    sum = warpReduceSum(sum);
    if (threadIdx.x == 0)
        C[index] = sum;
}

void matmul_gpu(float *xout, float *x, float *w, int n, int d) {
    int serialElements = DIVUP(n, WARP_SIZE);
    int serialLoads = DIVUP(serialElements, 4);
    dim3 blockDim(WARP_SIZE, 16);
    dim3 gridDim(DIVUP(d, 16));
    gemv_fp32_kernel<<<gridDim, blockDim>>>(xout, x, w, n, d, serialLoads);
    CHECK_CUDA(cudaGetLastError());
}

void matmul_fp32_gpu(float *xout, float *x, float *w, int n, int d) {
    float *xout_d, *x_d, *w_d;
    CHECK_CUDA(cudaMalloc(&xout_d, sizeof(float) * d));
    CHECK_CUDA(cudaMalloc(&x_d, sizeof(float) * n));
    CHECK_CUDA(cudaMalloc(&w_d, sizeof(float) * d * n));
    CHECK_CUDA(cudaMemcpy(x_d, x, sizeof(float) * n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(w_d, w, sizeof(float) * d * n, cudaMemcpyHostToDevice));

    // WARM UP
    for (int i=0; i<= 10; ++i) {
        matmul_gpu(xout_d, x_d, w_d, n, d);
    }

    int num_runs = 25;
    float elapsed_time = 0.0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        matmul_gpu(xout_d, x_d, w_d, n, d);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("GPU Matmul FP32 - Time Avg: %f ms\n", (elapsed_time / num_runs));
    printf("%f sec, GFLOPS: %f \n", elapsed_time, 2ll * n * d * num_runs / (elapsed_time * 1e6));

    CHECK_CUDA(cudaMemcpy(xout, xout_d, sizeof(float) * d, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(xout_d));
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(w_d));
}

void matmul_cublas_sgemv(float *xout, float *x, float *w, int n, int d) {
    // cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float *x_d, *w_d, *xout_d;
    CHECK_CUDA(cudaMalloc(&xout_d, sizeof(float) * d));
    CHECK_CUDA(cudaMalloc(&x_d, sizeof(float) * n));
    CHECK_CUDA(cudaMalloc(&w_d, sizeof(float) * d * n));
    CHECK_CUDA(cudaMemcpy(x_d, x, sizeof(float) * n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(w_d, w, sizeof(float) * d * n, cudaMemcpyHostToDevice));

    // Warm-up
    float alpha = 1.0f;
    float beta = 0.0f;
    for (int i = 0; i < 10; ++i) {
        CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, n, d, &alpha, w_d, n, x_d, 1, &beta, xout_d, 1));
    }

    // Measure performance
    int num_runs = 25;
    float elapsed_time = 0.0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, n, d, &alpha, w_d, n, x_d, 1, &beta, xout_d, 1));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    // Output performance
    printf("GPU Matmul FP32 cuBLAS - Time Avg: %f ms\n", (elapsed_time / num_runs));
    printf("%f sec, GFLOPS: %f \n", elapsed_time, 2ll * n * d * num_runs / (elapsed_time * 1e6));

    CHECK_CUDA(cudaMemcpy(xout, xout_d, sizeof(float) * d, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(xout_d));
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(w_d));
    CHECK_CUBLAS(cublasDestroy(handle));
}

void matmul_cpu(float *xout, float *x, float *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}


void test_matmul(bool verbose) {
    int n, d;
    n = 4096;
    d = 11008;
    printf("Matmul validating with n = %d, d = %d...\n", n, d);

    float *w = alloc_mat(d, n);
    float *x = alloc_mat(1, n);

    rand_mat(w, d, n);
    rand_mat(x, 1, n);
    x = half_to_float(float_to_half(x, n), n);
    w = half_to_float(float_to_half(w, n*d), n*d);

    float *xout_fp32_cpu = alloc_mat(1, d);
    matmul_cpu(xout_fp32_cpu, x, w, n, d);

    float *o_fp32_cublas = alloc_mat(1, d);
    matmul_cublas_sgemv(o_fp32_cublas, x, w, n, d);

    float *xout_fp32_gpu = alloc_mat(1, d);
    matmul_fp32_gpu(xout_fp32_gpu, x, w, n, d);

    float *xout_fp16_gpu = alloc_mat(1, d);
    matmul_fp16_gpu(xout_fp16_gpu, x, w, n, d);


    printf("host fp32: \n");
    print_mat(xout_fp32_cpu, 1, d);
    printf("\n");

    printf("device fp32: \n");
    print_mat(xout_fp32_gpu, 1, d);
    vec_diff_check(xout_fp32_cpu, xout_fp32_gpu, d, verbose);
    printf("\n");

    printf("device fp16: \n");
    print_mat(xout_fp16_gpu, 1, d);
    vec_diff_check(xout_fp32_cpu, xout_fp16_gpu, d, verbose);
}

int main(int argc, char **argv) {
    // Seed the random number generator with the current time
    srand(time(NULL));
    bool verbose = atoi(argv[0]);
    test_matmul(verbose);
}
