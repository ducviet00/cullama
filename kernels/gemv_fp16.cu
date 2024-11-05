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

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    return val;
}

__device__ __forceinline__ __half warpReduceSum(__half val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    return val;
}

__global__ void gemv_fp16_kernel_simple(__half *xout, const __half *x, const __half *W, int n, int d, int groupsize) {
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;

    extern __shared__ __half x_shared[];

    const int threads_per_block = blockDim.x * blockDim.y;
    const int elements_per_thread = DIVUP(n, threads_per_block);

    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int idx = threadIdx.x + threadIdx.y * blockDim.x + i * threads_per_block;
        if (idx < n) {
            x_shared[idx] = x[idx];
        }
    }
    __syncthreads();

    W += index * n;
    float sum = 0;
    #pragma unroll 4
    for (int i = 0; i < groupsize; i++) {
        int j = i * WARP_SIZE + threadIdx.x;
        if (j < n)
            sum += __half2float(__hmul(W[j], x_shared[j]));
    }

    sum = warpReduceSum(sum);
    if (threadIdx.x == 0)
        xout[index] = __float2half(sum);
}

void matmul_gpu_v0(__half *xout, __half *x, __half *w, int n, int d, int batch_size) {
    int groupsize = DIVUP(n, WARP_SIZE);
    dim3 blockDim(WARP_SIZE, 16);
    dim3 gridDim(DIVUP(d, 16));
    size_t shared_mem_size = ((n * sizeof(__half) + 127) / 128) * 128;
    gemv_fp16_kernel_simple<<<gridDim, blockDim, shared_mem_size>>>(xout, x, w, n, d, groupsize);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void gemv_fp16_kernel(
    __half *xout,
    const __half *__restrict__ x,
    const __half *__restrict__ W,
    int n,
    int d,
    int vectorized
) {
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;

    W += index * n;

    float sum = 0;

    __half2 w;
    __half2 inp;
    __half2 prod;
#pragma unroll
    for (int i = 0; i < vectorized; i++) {
        int j = (i * WARP_SIZE + threadIdx.x) * 2;
        if (j < n) {
            w = *((__half2 *)(&W[j]));
            inp = *((__half2 *)(&x[j]));
            prod = __hmul2(w, inp);
            sum += __low2float(prod) + __high2float(prod);
        }
    }

    sum = warpReduceSum(sum);
    if (threadIdx.x == 0)
        xout[index] = __float2half(sum);
}

void matmul_gpu_v1(__half *xout, __half *x, __half *w, int n, int d, int batch_size) {
    int groupsize = DIVUP(n, WARP_SIZE);
    int vectorized = DIVUP(groupsize, 2);
    dim3 blockDim(WARP_SIZE, 16);
    dim3 gridDim(DIVUP(d, 16), batch_size);
    size_t shared_mem_size = ((n * sizeof(__half) + 127) / 128) * 128;
    gemv_fp16_kernel<<<gridDim, blockDim, shared_mem_size>>>(xout, x, w, n, d, vectorized);
    CHECK_CUDA(cudaGetLastError());
}

void matmul_gpu(__half *xout, __half *x, __half *w, int n, int d, int batch_size) {
    __half *xoutGPU, *xGPU, *wGPU;
    CHECK_CUDA(cudaMalloc(&xoutGPU, sizeof(__half) * d * batch_size));
    CHECK_CUDA(cudaMalloc(&xGPU, sizeof(__half) * n * batch_size));
    CHECK_CUDA(cudaMalloc(&wGPU, sizeof(__half) * d * n));
    CHECK_CUDA(cudaMemcpy(xGPU, x, sizeof(__half) * n * batch_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(wGPU, w, sizeof(__half) * d * n, cudaMemcpyHostToDevice));

    // WARM UP
    for (int i=0; i<= 3; ++i) {
        matmul_gpu_v1(xoutGPU, xGPU, wGPU, n, d, batch_size);
    }

    float elapsed_time = 0.0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i) {
        matmul_gpu_v1(xoutGPU, xGPU, wGPU, n, d, batch_size);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("%f sec, GFLOPS: %f \n", elapsed_time, 2ll * n * d * batch_size * 10 / (elapsed_time * 1e6));

    CHECK_CUDA(cudaMemcpy(xout, xoutGPU, sizeof(__half) * d * batch_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(xoutGPU));
    CHECK_CUDA(cudaFree(xGPU));
    CHECK_CUDA(cudaFree(wGPU));
}

void matmul_cpu(float *xout, float *x, float *w, int n, int d, int batch_size) {
    for (int req_id = 0; req_id < batch_size; req_id++) {
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += w[i * n + j] * x[req_id * n + j];
            }
            xout[req_id * d + i] = val;
        }
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

float *alloc_mat(int R, int C) {
    float *m;
    CHECK_CUDA(cudaMallocHost(&m, sizeof(float) * R * C));
    return m;
}

__half *alloc_mat_half(int R, int C) {
    __half *m;
    CHECK_CUDA(cudaMallocHost(&m, sizeof(__half) * R * C));
    return m;
}

void rand_mat(float *m, int R, int C) {
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            m[i * C + j] = (float)rand() / (float)RAND_MAX - 0.5;
        }
    }
}

void float_to_half(const float *fm, __half *hm, int size) {
    for (int i = 0; i < size; ++i) {
        hm[i] = __float2half_rn(fm[i]);
    }
}

void half_to_float(const __half *hm, float *fm, int size) {
    for (int i = 0; i < size; ++i) {
        fm[i] = __half2float(hm[i]);
    }
}

void print_mat(const float *m, int R, int C) {
    for (int i = 0; i < MIN(R, 10); ++i) {
        for (int j = 0; j < MIN(C, 10); ++j) {
            printf("%+.3f ", m[i * C + j]);
        }
        printf("\n");
    }
}

void test_matmul(bool verbose) {
    int n, d;
    n = 4096;
    d = 11008;
    int batch_size = 1;
    printf("Matmul validating with n = %d, d = %d, batch_size = %d...\n", n, d, batch_size);

    float *A = alloc_mat(d, n);
    float *B = alloc_mat(batch_size, n);
    float *C_cpu = alloc_mat(batch_size, d);
    float *C_gpu = alloc_mat(batch_size, d);

    __half *A_half = alloc_mat_half(d, n);
    __half *B_half = alloc_mat_half(batch_size, n);
    __half *C_gpu_half = alloc_mat_half(batch_size, d);

    rand_mat(A, d, n);
    float_to_half(A, A_half, d * n);
    rand_mat(B, batch_size, n);
    float_to_half(B, B_half, batch_size * n);

    matmul_cpu(C_cpu, B, A, n, d, batch_size);

    matmul_gpu(C_gpu_half, B_half, A_half, n, d, batch_size);
    half_to_float(C_gpu_half, C_gpu, batch_size * d);
    vec_diff_check(C_cpu, C_gpu, d * batch_size, verbose);

    // printf("A: \n");
    // print_mat(A, d, n);
    // printf("\n");

    // printf("B: \n");
    // print_mat(B, batch_size, n);
    // printf("\n");

    // printf("C_gpu: \n");
    // print_mat(C_gpu, batch_size, d);
    // printf("\n");

    // printf("C_cpu: \n");
    // print_mat(C_cpu, batch_size, d);
    // printf("\n");
}

int main(int argc, char **argv) {
    // Seed the random number generator with the current time
    srand(time(NULL));
    bool verbose = atoi(argv[0]);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("\nDevice %d: %s\n", 0, prop.name);
    printf("  Shared mem per block: %zu B\n", prop.sharedMemPerBlock);
    printf("  Shared mem per SM: %zu B\n", prop.sharedMemPerMultiprocessor);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Warp size: %d\n", prop.warpSize);

    test_matmul(verbose);
}
