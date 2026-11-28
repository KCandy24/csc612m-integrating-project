// =====================================================================
// FusedSearch: filtered top-K search on the GPU (CUDA)
//
// based on the proposal: blocks take chunks of documents from a shared document queue, 
// each thread checks one document's tags
// a warp skips its 32 documents if none pass (early exit), 
// passing documents are scored with a warp-shuffle dot product
// the top-k is picked with bitonic sorts.
// 
// fused   -   false: score every document first, check the tags after (unfused)
// =====================================================================
#pragma once

#include <cstdint>

struct CUevent_st; 

class FusedSearch {
public:
    static constexpr int MAX_K = 32;      // most results one search can return
    static constexpr int MAX_WORDS = 32;  // most mask words, 32 * 32 = 1024 tags
    static constexpr int MAX_DIM = 512;   // longest vector

    struct Options {
        bool fused = true;
        bool cosine = false;
        bool per_thread = false;  // post-optimization
    };

    // vectors: n * dim floats, masks: n * words uint32 words, both row-major.
    FusedSearch(const float *vectors, const uint32_t *masks, int n, int dim, int words);
    ~FusedSearch();

    FusedSearch(const FusedSearch &) = delete;
    FusedSearch &operator=(const FusedSearch &) = delete;

    int dim() const { return dim_; }
    int words() const { return words_; }

    int search(const float *query, const uint32_t *query_mask, int k, const Options &options,
               int *out_ids, float *out_scores);

    // GPU time of the last search's kernels only (search + merge)
    float last_kernel_ms() const { return last_kernel_ms_; }

    // top-k
    struct Result {
        int count;
        int ids[MAX_K];
        float scores[MAX_K];
    };

private:
    int n_, dim_, stride_, words_, blocks_; 
    float *d_vectors_ = nullptr;     // each row starts on a 128-byte boundary
    uint32_t *d_masks_ = nullptr;    // one doc's words side by side
    uint32_t *d_masks_t_ = nullptr;  // one word of every doc side by side
    int *d_ids_ = nullptr;           // one top-k list per block
    float *d_scores_ = nullptr;
    int *d_queue_ = nullptr;         // shared document queue
    Result *d_result_ = nullptr;     // top-k on the gpu
    Result *h_result_ = nullptr;     // top-k on the cpu
    CUevent_st *start_ = nullptr, *stop_ = nullptr;  // gpu timers for last_kernel_ms
    float last_kernel_ms_ = 0.0f;

    void release();
};
