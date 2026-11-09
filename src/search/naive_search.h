// =====================================================================
// NaiveSearch: brute-force filtered top-K search (CPU baseline).
//
// Each document has an embedding (dim floats) and a multi-hot tag vector
// (num_tags values, 0 or 1) that is stored as bitmask words, 32 tags per word.
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
    // vectors: n * dim floats, tags: n * num_tags bytes (0 or 1), both row-major.
    NaiveSearch(const float *vectors, const uint8_t *tags, int n, int dim, int num_tags) {
        n_ = n;
        dim_ = dim;
        num_tags_ = num_tags;
        words_ = (num_tags + 31) / 32;

        vectors_.assign(vectors, vectors + (size_t)n * dim);
        masks_.assign((size_t)n * words_, 0);
        for (int i = 0; i < n; i++) {
            pack(tags + (size_t)i * num_tags, masks_.data() + (size_t)i * words_);
        }
    }

    int dim() const { return dim_; }
    int num_tags() const { return num_tags_; }

    // Writes the best (up to) k matches, best first, into out_ids / out_scores
    // (room for k entries each) and returns how many were found.
    int search(const float *query, const uint8_t *query_tags, int k,
               int *out_ids, float *out_scores) const {
        if (k <= 0) return 0;

        std::vector<uint32_t> qmask(words_, 0);
        pack(query_tags, qmask.data());

        int count = 0;
        for (int i = 0; i < n_; i++) {
            // dot product here for scoring
            const float *v = &vectors_[(size_t)i * dim_];
            float score = 0.0f;
            for (int d = 0; d < dim_; d++) {
                score += query[d] * v[d];
            }

            // filter
            if (!has_all_tags(i, qmask)) continue;

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
    int n_, dim_, num_tags_, words_;
    std::vector<float> vectors_;
    std::vector<uint32_t> masks_;

    // Sets bit t of the (zeroed) output words for every tags[t] != 0.
    void pack(const uint8_t *tags, uint32_t *words) const {
        for (int t = 0; t < num_tags_; t++) {
            if (tags[t]) words[t / 32] |= 1u << (t % 32);
        }
    }

    // true if document i has every tag that is set in qmask.
    bool has_all_tags(int i, const std::vector<uint32_t> &qmask) const {
        const uint32_t *m = masks_.data() + (size_t)i * words_;
        for (int w = 0; w < words_; w++) {
            if ((m[w] & qmask[w]) != qmask[w]) return false;
        }
        return true;
    }

    // true if (id_a, score_a) should rank above (id_b, score_b).
    static bool better(int id_a, float score_a, int id_b, float score_b) {
        if (score_a != score_b) return score_a > score_b;
        return id_a < id_b;
    }
};
