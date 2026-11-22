import time
from pathlib import Path

import numpy as np
import pandas as pd

from search.naive_search import NaiveSearch

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "310"
K = 5
RUNS = 20


def pack_tags(tags):
    """(rows, C) of 0/1 -> (rows, ceil(C / 32)) uint32 words; tag t is bit t % 32 of word t // 32."""
    rows, num_tags = tags.shape
    words = (num_tags + 31) // 32
    bits = np.zeros((rows, words * 32), dtype=np.uint32)
    bits[:, :num_tags] = tags != 0
    bits = bits.reshape(rows, words, 32) << np.arange(32, dtype=np.uint32)
    return bits.sum(axis=2, dtype=np.uint32)


def load(name):
    df = pd.read_parquet(DATA_DIR / name)
    vectors = np.array(df["abstract_embedding"].tolist())
    masks = pack_tags(np.array(df["category_multihot"].tolist()))
    return df, vectors, masks


def abstract(df, i):
    # arXiv abstracts contain line breaks and leading spaces; flatten them
    return " ".join(df["abstract"].iloc[i].split())


def main():
    docs, vectors, masks = load("processed_arxiv_search.parquet")
    queries, q_vectors, q_masks = load("processed_arxiv_query.parquet")

    index = NaiveSearch(vectors, masks)
    print(f"Loaded N={len(docs)} docs, D={vectors.shape[1]} dims, {32 * masks.shape[1]}-bit masks, K={K}")

    for i in range(len(queries)):
        start = time.perf_counter()
        for _ in range(RUNS):
            ids, scores = index.search(q_vectors[i], q_masks[i], K)
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
