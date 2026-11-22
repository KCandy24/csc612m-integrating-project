// =====================================================================
// NaiveSearch: brute-force filtered top-K search (CPU baseline).
//
// Each document has an embedding (dim floats) and a tag bitmask of `words`
// 32-bit words (a set bit = the document has that tag). The caller packs the tags.
//
// Rules: ALL query tags required: (doc & q) == q on every word.
//        Score is the dot product (= cosine for unit-length embeddings).
//        Higher score wins; on a tie, lower doc id wins.
// =====================================================================
#pragma once

#include <cstdint>
#include <vector>

class NaiveSearch {
public:
    // vectors: n * dim floats, masks: n * words uint32 words, both row-major.
    NaiveSearch(const float *vectors, const uint32_t *masks, int n, int dim, int words) {
        n_ = n;
        dim_ = dim;
        words_ = words;

        vectors_.assign(vectors, vectors + (size_t)n * dim);
        masks_.assign(masks, masks + (size_t)n * words);
    }

    int dim() const { return dim_; }
    int words() const { return words_; }

    // query_mask: `words` uint32 words.
    // Writes the best (up to) k matches, best first, into out_ids / out_scores
    // (room for k entries each) and returns how many were found.
    int search(const float *query, const uint32_t *query_mask, int k,
               int *out_ids, float *out_scores) const {
        if (k <= 0) return 0;

        int count = 0;
        for (int i = 0; i < n_; i++) {
            // filter
            if (!has_all_tags(i, query_mask)) continue;

            // dot product here for scoring
            const float *v = &vectors_[(size_t)i * dim_];
            float score = 0.0f;
            for (int d = 0; d < dim_; d++) {
                score += query[d] * v[d];
            }

            // if full, only proceed if it beats the last (worst) entry.
            if (count == k && !better(i, score, out_ids[k - 1], out_scores[k - 1])) continue;

            // slide worse entries down one place until we find its spot.
            int pos = (count < k) ? count : k - 1;
            while (pos > 0 && better(i, score, out_ids[pos - 1], out_scores[pos - 1])) {
                out_ids[pos] = out_ids[pos - 1];
                out_scores[pos] = out_scores[pos - 1];
                pos--;
            }
            out_ids[pos] = i;
            out_scores[pos] = score;
            if (count < k) count++;
        }
        return count;
    }

private:
    int n_, dim_, words_;
    std::vector<float> vectors_;
    std::vector<uint32_t> masks_;

    // true if document i has every tag that is set in query_mask.
    bool has_all_tags(int i, const uint32_t *query_mask) const {
        const uint32_t *m = masks_.data() + (size_t)i * words_;
        for (int w = 0; w < words_; w++) {
            if ((m[w] & query_mask[w]) != query_mask[w]) return false;
        }
        return true;
    }

    // true if (id_a, score_a) should rank above (id_b, score_b).
    static bool better(int id_a, float score_a, int id_b, float score_b) {
        if (score_a != score_b) return score_a > score_b;
        return id_a < id_b;
    }
};
