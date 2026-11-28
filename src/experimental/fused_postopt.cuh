// post-optimization: per-thread top-k arrays
// same as the fused kernel, except:
// - each warp keeps its own sorted list instead of a shared buffer, so no sorting together
// - each warp takes papers from the queue on its own, no waiting for the block
#pragma once

// put a paper into a sorted list (same as naive)
__host__ __device__ void insert(int *ids, float *scores, int &count, int k, int id, float score) {
    if (count == k && !better(id, score, ids[k - 1], scores[k - 1])) return;  // worse than the last: skip
    int pos = (count < k) ? count : k - 1;
    while (pos > 0 && better(id, score, ids[pos - 1], scores[pos - 1])) {  // move worse ones down
        ids[pos] = ids[pos - 1];
        scores[pos] = scores[pos - 1];
        pos--;
    }
    ids[pos] = id;
    scores[pos] = score;
    if (count < k) count++;
}

// step 1: each warp finds its own best k
__global__ void search_kernel_per_thread(const float *vectors, const uint32_t *masks, const uint32_t *masks_t,
                                         int n, int dim, int stride, int words, QueryArgs qa, int k,
                                         int *out_ids, float *out_scores, int *queue, bool fused, bool cosine) {
    __shared__ float query[FusedSearch::MAX_DIM];
    __shared__ uint32_t qmask[FusedSearch::MAX_WORDS];
    __shared__ int list_ids[WARPS_PER_BLOCK][FusedSearch::MAX_K];  // one list per warp
    __shared__ float list_scores[WARPS_PER_BLOCK][FusedSearch::MAX_K];
    __shared__ int list_count[WARPS_PER_BLOCK];

    // copy the query to shared memory
    for (int d = threadIdx.x; d < dim; d += blockDim.x) query[d] = qa.q[d];
    for (int w = threadIdx.x; w < words; w += blockDim.x) qmask[w] = qa.m[w];
    __syncthreads();

    int lane = threadIdx.x % WARP;  // 0-31
    int warp = threadIdx.x / WARP;  // 0-7
    int groups = (n + GROUP - 1) / GROUP;
    int count = 0;  // size of this warp's list

    while (true) {
        // lane 0 takes the next group from the queue, then shares it with the warp
        int group = 0;
        if (lane == 0) group = atomicAdd(queue, 1);
        group = __shfl_sync(FULL, group, 0);
        if (group >= groups) break;  // no papers left
        int first = group * GROUP;

        if (fused) {
            // check tags first
            unsigned passed = check_32_papers(masks_t, n, words, qmask, first, lane);
            // if none passed, early exit
            while (passed != 0) {
                int j = __ffs(passed) - 1;
                passed &= passed - 1;
                float score = score_doc(vectors + (size_t)(first + j) * stride, query, dim, lane, cosine, qa.qnorm);
                if (lane == 0) insert(list_ids[warp], list_scores[warp], count, k, first + j, score);
            }
        } else {
            // unfused: score first, check tags after
            for (int doc = first; doc < first + GROUP && doc < n; doc++) {
                float score = score_doc(vectors + (size_t)doc * stride, query, dim, lane, cosine, qa.qnorm);
                if (has_all_tags(masks, doc, words, qmask, lane) && lane == 0)
                    insert(list_ids[warp], list_scores[warp], count, k, doc, score);
            }
        }
    }

    if (lane == 0) list_count[warp] = count;
    __syncthreads();

    // thread 0 combines the 8 warp lists, writes this block's best k (-1 = empty)
    if (threadIdx.x == 0) {
        int ids[FusedSearch::MAX_K];
        float scores[FusedSearch::MAX_K];
        int total = 0;
        for (int w = 0; w < WARPS_PER_BLOCK; w++)
            for (int j = 0; j < list_count[w]; j++)
                insert(ids, scores, total, k, list_ids[w][j], list_scores[w][j]);
        for (int j = 0; j < k; j++) {
            out_ids[blockIdx.x * k + j] = (j < total) ? ids[j] : -1;
            out_scores[blockIdx.x * k + j] = (j < total) ? scores[j] : 0.0f;
        }
    }
}

// step 2: one warp combines all blocks' best k
__global__ void merge_kernel_per_thread(const int *cand_ids, const float *cand_scores, int total, int k,
                                        FusedSearch::Result *result) {
    __shared__ int list_ids[WARP][FusedSearch::MAX_K];  // one list per lane
    __shared__ float list_scores[WARP][FusedSearch::MAX_K];
    __shared__ int list_count[WARP];

    // lane l takes candidates l, l + 32, l + 64, ...
    int lane = threadIdx.x;
    int count = 0;
    for (int i = lane; i < total; i += WARP)
        if (cand_ids[i] >= 0) insert(list_ids[lane], list_scores[lane], count, k, cand_ids[i], cand_scores[i]);
    list_count[lane] = count;
    __syncwarp();

    // lane 0 combines the 32 lists into the answer
    if (lane == 0) {
        int found = 0;
        for (int l = 0; l < WARP; l++)
            for (int j = 0; j < list_count[l]; j++)
                insert(result->ids, result->scores, found, k, list_ids[l][j], list_scores[l][j]);
        result->count = found;
    }
}
