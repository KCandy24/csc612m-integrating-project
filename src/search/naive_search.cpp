// =====================================================================
// BUILD:   g++ -O2 -std=c++17 naive_search.cpp -o naive.exe
// RUN:     ./naive <csv> <query_doc_id> <query_mask> <K>
// EXAMPLE: ./naive dummy.csv 0 5 5
//
// CSV: doc_id,bitmask,dim_0,...,dim_{D-1}   (header row first)
// Rules: ALL query tags required: (doc & q) == q
//        Higher score wins; on a tie, lower doc_id wins.
// =====================================================================

#include <chrono>   
#include <cstdint>  
#include <cstdio>   
#include <cstdlib>  
#include <vector>   

struct Result {
    int   id;
    float score;
};

int RUNS = 20;
static char line[1 << 16];


// true if a should rank above b (higher score, or same score + lower id)
bool better(const Result &a, const Result &b) {
    if (a.score != b.score) 
        return a.score > b.score;
    return a.id < b.id;
}

bool load_csv(const char *path, int &N, int &D,
              std::vector<float> &vecs, std::vector<uint32_t> &masks) {
    FILE *f = fopen(path, "r");
    if (!f) return false;

    fgets(line, sizeof(line), f);
    int commas = 0;
    for (char *c = line; *c; c++) {
        if (*c == ',') commas++;
    }
    D = commas - 1;

    N = 0;
    while (fgets(line, sizeof(line), f)) {
        char *p = line;              
        if (*p == '\n' || *p == '\0') continue;   // skip blank lines

        strtoul(p, &p, 10);          
        p++;
        masks.push_back((uint32_t)strtoul(p, &p, 10));   // bitmask
        p++;
        for (int d = 0; d < D; d++) {
            vecs.push_back(strtof(p, &p));              
            p++;
        }
        N++;
    }
    fclose(f);
    return true;
}

std::vector<Result> naive_search(const std::vector<float> &vecs,
                                 const std::vector<uint32_t> &masks,
                                 int N, int D,
                                 const float *query, uint32_t qmask, int K) {
    std::vector<Result> top(K);
    int count = 0;

    for (int i = 0; i < N; i++) {
        // dot product here for scoring
        float score = 0.0f;
        const float *v = &vecs[(size_t)i * D];
        for (int d = 0; d < D; d++) {
            score += query[d] * v[d];
        }

        // filter
        if ((masks[i] & qmask) != qmask) 
            continue;

        Result r{i, score};

        // if full, only proceed if r beats the last (worst) entry.
        if (count == K && !better(r, top[K - 1])) 
            continue;

        int pos = (count < K) ? count : K - 1;   
        // slide worse entries down one place until we find r's spot.
        while (pos > 0 && better(r, top[pos - 1])) {
            top[pos] = top[pos - 1];
            pos--;
        }
        top[pos] = r;
        if (count < K) count++;
    }

    top.resize(count);   
    return top;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <csv> <query_doc_id> <query_mask> <K>\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    int qid = atoi(argv[2]);
    uint32_t qmask = (uint32_t)strtoul(argv[3], nullptr, 10);
    int K = atoi(argv[4]);
    if (K <= 0) {
        fprintf(stderr, "K must be at least 1\n");
        return 1;
    }

    int N = 0, D = 0;
    std::vector<float> vecs;
    std::vector<uint32_t> masks;
    if (!load_csv(path, N, D, vecs, masks)) {
        fprintf(stderr, "Could not open %s\n", path);
        return 1;
    }
    if (qid < 0 || qid >= N) {
        fprintf(stderr, "query_doc_id out of range (N=%d)\n", N);
        return 1;
    }
    printf("Loaded N=%d docs, D=%d dims. Query = vector of doc %d, mask=%u, K=%d\n",
           N, D, qid, qmask, K);

    const float *query = &vecs[(size_t)qid * D];

    std::vector<Result> res;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++) {
        res = naive_search(vecs, masks, N, D, query, qmask, K);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(finish - start).count() / RUNS;

    printf("\nTop-%d (doc_id, score, mask):\n", K);
    for (const Result &r : res) {
        printf("%6d  %.6f  %u\n", r.id, r.score, masks[r.id]);
    }
    printf("\nAvg search time: %.4f ms over %d runs\n", ms, RUNS);
    return 0;
}