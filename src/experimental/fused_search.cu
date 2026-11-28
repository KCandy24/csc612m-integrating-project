// FusedSearch, cpu side: copies the data to the gpu once, then for each search
// launches the kernels and copies back the final top-k
//   fused_kernel.cuh   fused kernel (the design), fused = false is the unfused baseline
//   fused_postopt.cuh  post-optimization: per-thread top-k arrays (per_thread)
#include "fused_search.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t err_ = (call);                                                 \
        if (err_ != cudaSuccess)                                                   \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(err_)); \
    } while (0)

#include "fused_kernel.cuh"
#include "fused_postopt.cuh"

FusedSearch::FusedSearch(const float *vectors, const uint32_t *masks, int n, int dim, int words)
    : n_(n), dim_(dim), words_(words) {
    if (n <= 0 || dim <= 0 || words <= 0) throw std::invalid_argument("n, dim and words must be positive");
    if (words > MAX_WORDS) throw std::invalid_argument("at most 32 mask words (1024 tags) supported");
    if (dim > MAX_DIM) throw std::invalid_argument("at most 512 dims supported");
    stride_ = (dim + ROW_ALIGN - 1) / ROW_ALIGN * ROW_ALIGN;  

    int device = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));

    // enough blocks to fill the GPU, but no more than one warp per document
    blocks_ = std::min((n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, sms * 8);

    // masks transposed on the cpu (SoA): word w of doc i goes to w * n + i
    uint32_t *masks_t = new uint32_t[(size_t)n * words];
    for (int i = 0; i < n; i++)
        for (int w = 0; w < words; w++) masks_t[(size_t)w * n + i] = masks[(size_t)i * words + w];

    // allocate on the gpu and copy the data over
    try {
        CUDA_CHECK(cudaMalloc(&d_vectors_, sizeof(float) * (size_t)n * stride_));  
        CUDA_CHECK(cudaMalloc(&d_masks_, sizeof(uint32_t) * (size_t)n * words));
        CUDA_CHECK(cudaMalloc(&d_masks_t_, sizeof(uint32_t) * (size_t)n * words));
        CUDA_CHECK(cudaMalloc(&d_ids_, sizeof(int) * blocks_ * MAX_K));
        CUDA_CHECK(cudaMalloc(&d_scores_, sizeof(float) * blocks_ * MAX_K));
        CUDA_CHECK(cudaMalloc(&d_queue_, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_result_, sizeof(Result)));
        CUDA_CHECK(cudaMallocHost(&h_result_, sizeof(Result)));  
        CUDA_CHECK(cudaMemcpy2D(d_vectors_, sizeof(float) * stride_, vectors, sizeof(float) * dim,
                                sizeof(float) * dim, n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_masks_, masks, sizeof(uint32_t) * (size_t)n * words, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_masks_t_, masks_t, sizeof(uint32_t) * (size_t)n * words, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    } catch (...) {
        delete[] masks_t;
        release();
        throw;
    }
    delete[] masks_t;
}

FusedSearch::~FusedSearch() { release(); }

void FusedSearch::release() {
    cudaFree(d_vectors_);
    cudaFree(d_masks_);
    cudaFree(d_masks_t_);
    cudaFree(d_ids_);
    cudaFree(d_scores_);
    cudaFree(d_queue_);
    cudaFree(d_result_);
    if (h_result_) cudaFreeHost(h_result_);
    if (start_) cudaEventDestroy(start_);
    if (stop_) cudaEventDestroy(stop_);
    d_vectors_ = nullptr;
    d_masks_ = nullptr;
    d_masks_t_ = nullptr;
    d_ids_ = nullptr;
    d_scores_ = nullptr;
    d_queue_ = nullptr;
    d_result_ = nullptr;
    h_result_ = nullptr;
    start_ = stop_ = nullptr;
}

int FusedSearch::search(const float *query, const uint32_t *query_mask, int k, const Options &options,
                        int *out_ids, float *out_scores) {
    if (k <= 0) return 0;
    if (k > MAX_K) throw std::invalid_argument("k must be at most 32");

    QueryArgs qa;
    std::memcpy(qa.q, query, sizeof(float) * dim_);
    std::memcpy(qa.m, query_mask, sizeof(uint32_t) * words_);
    float qq = 0.0f;
    for (int d = 0; d < dim_; d++) qq += query[d] * query[d];
    qa.qnorm = std::sqrt(qq);  // query length for cosine

    CUDA_CHECK(cudaMemsetAsync(d_queue_, 0, sizeof(int)));  // the document queue starts at 0
    CUDA_CHECK(cudaEventRecord(start_));                    // gpu timer start
    if (options.per_thread) {
        search_kernel_per_thread<<<blocks_, THREADS>>>(d_vectors_, d_masks_, d_masks_t_, n_, dim_, stride_, words_,
                                                       qa, k, d_ids_, d_scores_, d_queue_, options.fused, options.cosine);
        merge_kernel_per_thread<<<1, WARP>>>(d_ids_, d_scores_, blocks_ * k, k, d_result_);
    } else {
        search_kernel<<<blocks_, THREADS>>>(d_vectors_, d_masks_, d_masks_t_, n_, dim_, stride_, words_,
                                            qa, k, d_ids_, d_scores_, d_queue_, options.fused, options.cosine);
        merge_kernel<<<1, THREADS>>>(d_ids_, d_scores_, blocks_ * k, k, d_result_);
    }
    CUDA_CHECK(cudaGetLastError());       // catches a bad launch
    CUDA_CHECK(cudaEventRecord(stop_));   // gpu timer stop

    CUDA_CHECK(cudaMemcpyAsync(h_result_, d_result_, sizeof(Result), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaEventElapsedTime(&last_kernel_ms_, start_, stop_));  // kernels only, no copy

    std::memcpy(out_ids, h_result_->ids, sizeof(int) * h_result_->count);
    std::memcpy(out_scores, h_result_->scores, sizeof(float) * h_result_->count);
    return h_result_->count;
}
