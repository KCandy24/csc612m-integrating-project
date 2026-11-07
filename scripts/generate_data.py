import csv
import math
import random
from pathlib import Path


NUM_DOCS = 100
VECTOR_DIMS = 50
BITMASK_BITS = 5
MAX_BITMASK = (1 << BITMASK_BITS) - 1
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "dummy_vectors.csv"


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["doc_id", "bitmask"] + [f"dim_{dimension}" for dimension in range(VECTOR_DIMS)]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)

        for doc_id in range(NUM_DOCS):
            bitmask = random.randint(0, MAX_BITMASK)
            vector = [random.uniform(-1.0, 1.0) for _ in range(VECTOR_DIMS)]
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            normalized_vector = [value / norm for value in vector]

            writer.writerow(
                [doc_id, bitmask]
                + [f"{value:.6f}" for value in normalized_vector]
            )

    print(f"Dummy dataset written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
