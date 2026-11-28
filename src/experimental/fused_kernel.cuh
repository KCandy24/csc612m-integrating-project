// fused kernel (the design from the proposal)
// 1. each block takes the next papers from a shared queue
// 2. each thread checks one paper's tags; if none of the warp's 32 pass, skip them all
// 3. only passing papers are scored: the warp splits the 384 numbers, then adds them up
// 4. passing papers go into a buffer; the block sorts it (bitonic) to keep the best k
// fused = false: score every paper first, check tags after (unfused, for comparison)
#pragma once

#define WARP 32
#define WARPS_PER_BLOCK 8
#define THREADS 256        // 8 warps x 32 threads
#define GROUP 32           // papers per warp per turn, one per thread
#define FULL 0xffffffff    // all 32 threads of the warp take part
#define ROW_ALIGN 32       // rows padded to 32 floats = 128 bytes
#define BLOCK_BUF 512      // block's buffer for passing papers
#define MERGE_BUF 1024     // merge kernel's buffer
#define NO_ID 0x7fffffff   // empty slot

// the query, sent with the kernel launch
struct QueryArgs {
    float q[FusedSearch::MAX_DIM];
    uint32_t m[FusedSearch::MAX_WORDS];
    float qnorm;  // query length, for cosine
};

// higher score wins, lower id on a tie (same as naive)
__host__ __device__ bool better(int id_a, float score_a, int id_b, float score_b) {
    if (score_a != score_b) return score_a > score_b;
    return id_a < id_b;
}

// whole warp scores one paper, lane 0 gets the total
__device__ float score_doc(const float *v, const float *query, int dim, int lane, bool cosine, float qnorm) {
    float dot = 0.0f, vv = 0.0f;
    // each lane does 12 of the 384 numbers
    for (int d = lane; d < dim; d += WARP) {
        dot += query[d] * v[d];
        if (cosine) vv += v[d] * v[d];  // paper length
    }
    // add the 32 parts together
    for (int offset = 16; offset > 0; offset /= 2) {
        dot += __shfl_down_sync(FULL, dot, offset);
        if (cosine) vv += __shfl_down_sync(FULL, vv, offset);
    }
    if (cosine) {
        float length = sqrtf(vv) * qnorm;
        dot = (length > 0.0f) ? dot / length : 0.0f;
    }
    return dot;
}

// fused: each thread checks one of the 32 papers
// returns 32 bits, bit i = 1 if paper first + i passed
__device__ unsigned check_32_papers(const uint32_t *masks_t, int n, int words, const uint32_t *qmask,
                                    int first, int lane) {
    int doc = first + lane;
    bool pass = doc < n;
    for (int w = 0; pass && w < words; w++)
        pass = (masks_t[(size_t)w * n + doc] & qmask[w]) == qmask[w];
    return __ballot_sync(FULL, pass);
}

// unfused: whole warp checks one paper, lanes 0-4 check one tag word each
__device__ bool has_all_tags(const uint32_t *masks, int doc, int words, const uint32_t *qmask, int lane) {
    uint32_t m = (lane < words) ? masks[(size_t)doc * words + lane] : 0;
    uint32_t q = (lane < words) ? qmask[lane] : 0;
    return __all_sync(FULL, (m & q) == q);  // true only if all lanes pass
}

// whole block sorts the buffer, best first
__device__ void bitonic_sort(float *scores, int *ids, int count, int size) {
    // empty slots get -infinity so they go last
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        if (i >= count) {
            scores[i] = -INFINITY;
            ids[i] = NO_ID;
        }
    __syncthreads();
    // sort runs of 2, then 4, 8, ... until the whole buffer is sorted
    for (int run = 2; run <= size; run *= 2) {
        for (int stride = run / 2; stride > 0; stride /= 2) {
            for (int i = threadIdx.x; i < size; i += blockDim.x) {
                int j = i ^ stride;  // partner
                if (j > i) {
                    bool best_first = (i & run) == 0;
                    if (best_first != better(ids[i], scores[i], ids[j], scores[j])) {  // swap if wrong order
                        float s = scores[i]; scores[i] = scores[j]; scores[j] = s;
                        int t = ids[i]; ids[i] = ids[j]; ids[j] = t;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// step 1: each block finds its best k
__global__ void search_kernel(const float *vectors, const uint32_t *masks, const uint32_t *masks_t,
                              int n, int dim, int stride, int words, QueryArgs qa, int k,
                              int *out_ids, float *out_scores, int *queue, bool fused, bool cosine) {
    __shared__ float query[FusedSearch::MAX_DIM];
    __shared__ uint32_t qmask[FusedSearch::MAX_WORDS];
    __shared__ float buf_scores[BLOCK_BUF];
    __shared__ int buf_ids[BLOCK_BUF];
    __shared__ int buf_count;
    __shared__ int chunk;

    // copy the query to shared memory
    for (int d = threadIdx.x; d < dim; d += blockDim.x) query[d] = qa.q[d];
    for (int w = threadIdx.x; w < words; w += blockDim.x) qmask[w] = qa.m[w];
    if (threadIdx.x == 0) buf_count = 0;
    __syncthreads();

    int lane = threadIdx.x % WARP;  // 0-31
    int warp = threadIdx.x / WARP;  // 0-7
    int groups = (n + GROUP - 1) / GROUP;

    while (true) {
        // take the next 8 groups from the queue (one per warp)
        if (threadIdx.x == 0) chunk = atomicAdd(queue, WARPS_PER_BLOCK);
        __syncthreads();
        int my_chunk = chunk;
        if (my_chunk >= groups) break;  // no papers left
        int first = (my_chunk + warp) * GROUP;  // this warp's 32 papers

        if (fused) {
            // check tags first
            unsigned passed = check_32_papers(masks_t, n, words, qmask, first, lane);
            // if none passed, this loop is skipped (early exit)
            while (passed != 0) {
                int j = __ffs(passed) - 1;  // next passing paper
                passed &= passed - 1;       // mark it done
                float score = score_doc(vectors + (size_t)(first + j) * stride, query, dim, lane, cosine, qa.qnorm);
                if (lane == 0) {
                    int slot = atomicAdd(&buf_count, 1);  // add to buffer
                    buf_scores[slot] = score;
                    buf_ids[slot] = first + j;
                }
            }
        } else {
            // unfused: score first, check tags after
            for (int doc = first; doc < first + GROUP && doc < n; doc++) {
                float score = score_doc(vectors + (size_t)doc * stride, query, dim, lane, cosine, qa.qnorm);
                if (has_all_tags(masks, doc, words, qmask, lane) && lane == 0) {
                    int slot = atomicAdd(&buf_count, 1);  // add to buffer
                    buf_scores[slot] = score;
                    buf_ids[slot] = doc;
                }
            }
        }
        __syncthreads();

        // buffer almost full: sort and keep the best k
        if (buf_count > BLOCK_BUF - THREADS) {
            bitonic_sort(buf_scores, buf_ids, buf_count, BLOCK_BUF);
            if (threadIdx.x == 0) buf_count = k;
            __syncthreads();
        }
    }

    // final sort, write this block's best k (-1 = empty)
    int count = buf_count;
    bitonic_sort(buf_scores, buf_ids, count, BLOCK_BUF);
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        out_ids[blockIdx.x * k + j] = (j < count) ? buf_ids[j] : -1;
        out_scores[blockIdx.x * k + j] = (j < count) ? buf_scores[j] : 0.0f;
    }
}

// step 2: one block combines all blocks' best k into the final best k
__global__ void merge_kernel(const int *cand_ids, const float *cand_scores, int total, int k,
                             FusedSearch::Result *result) {
    __shared__ float buf_scores[MERGE_BUF];
    __shared__ int buf_ids[MERGE_BUF];

    int count = 0;  // best k so far, kept at the front
    for (int start = 0; start < total;) {
        int take = min(MERGE_BUF - count, total - start);
        for (int i = threadIdx.x; i < take; i += blockDim.x) {
            int id = cand_ids[start + i];
            buf_ids[count + i] = (id >= 0) ? id : NO_ID;
            buf_scores[count + i] = (id >= 0) ? cand_scores[start + i] : -INFINITY;
        }
        __syncthreads();
        bitonic_sort(buf_scores, buf_ids, count + take, MERGE_BUF);
        count = min(count + take, k);
        start += take;
    }

    // thread 0 writes the answer
    if (threadIdx.x == 0) {
        int found = 0;
        while (found < count && buf_ids[found] != NO_ID) {
            result->ids[found] = buf_ids[found];
            result->scores[found] = buf_scores[found];
            found++;
        }
        result->count = found;
    }
}
