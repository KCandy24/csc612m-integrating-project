import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pathlib import Path

class Embedder():
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def encode(self, texts):
        return self.model.encode(texts)

def load_data():
    project_root = Path(__file__).resolve().parent.parent.parent
    load_path = project_root / "data" / "310" / "processed_arxiv.parquet"
    
    print("Loading Parquet file...")
    df = pd.read_parquet(load_path, engine='pyarrow')
    
    return df

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data" / "310"
    json_path = data_dir / "arxiv-metadata-oai-snapshot.json"

    df = pd.read_json(json_path, lines=True, nrows=10000)

    embedder = Embedder()
    embeddings = embedder.encode(df['abstract'].tolist())
    df['abstract_embedding'] = embeddings.tolist()


    category_dummies = df['categories'].str.get_dummies(sep=' ')
    df['category_multihot'] = category_dummies.values.tolist()


    df = df[['id', 'categories', 'category_multihot', 'abstract_embedding']]
    
    df.to_parquet(data_dir / "processed_arxiv.parquet", engine='pyarrow')


if __name__ == "__main__":
    main()
