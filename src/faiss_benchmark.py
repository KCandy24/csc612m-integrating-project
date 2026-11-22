"""FAISS version

Run: uv run python src/faiss_baseline.py > out_faiss.txt

"""
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "310"
K = 5
RUNS = 20


def load(name):
    df = pd.read_parquet(DATA_DIR / name)
    # faiss needs one float32 matrix
    vectors = np.ascontiguousarray(np.array(df["abstract_embedding"].tolist(), dtype=np.float32))
    tags = np.array(df["category_multihot"].tolist(), dtype=bool)
    return df, vectors, tags


def matching(tags, q_tags):
    return np.all(tags[:, q_tags], axis=1)


def abstract(df, i):
    return " ".join(df["abstract"].iloc[i].split())


def filtered_search(index, tags, q, q_tags, k):
    # docs that have all the query tags
    passing = np.flatnonzero(matching(tags, q_tags)).astype(np.int64)
    # faiss only scores the ids in here
    params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(passing))
    scores, ids = index.search(q, k, params=params)
    # remove the -1 padding if less than k pass
    keep = ids[0] >= 0
    return ids[0][keep].tolist(), scores[0][keep].tolist()


def main():
    docs, vectors, tags = load("processed_arxiv_search.parquet")
    queries, q_vectors, q_tags = load("processed_arxiv_query.parquet")

    # dot product
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    print(f"Loaded N={len(docs)} docs, D={vectors.shape[1]} dims, {tags.shape[1]} tags, K={K}")

    for i in range(len(queries)):
        # to have shape (1, 384)
        q = q_vectors[i : i + 1]

        start = time.perf_counter()
        for _ in range(RUNS):
            ids, scores = filtered_search(index, tags, q, q_tags[i], K)
        ms = (time.perf_counter() - start) * 1000 / RUNS

        print(f"\nQuery {queries['id'].iloc[i]} [{queries['categories'].iloc[i]}]: "
              f"{len(ids)} hits, avg {ms:.4f} ms over {RUNS} runs")
        print(f"  Query abstract:      {abstract(queries, i)}")
        if ids:
            print(f"  Best match abstract: {abstract(docs, ids[0])}")
        for doc_id, score in zip(ids, scores):
            print(f"  {docs['id'].iloc[doc_id]}  {score:.6f}  {docs['categories'].iloc[doc_id]}")


if __name__ == "__main__":
    main()
