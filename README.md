# CSC612M Integrating Project
- Aguarin, Trish Ann Danielle
- Andaya, Kurt Christian
- Cabato, Marxandrea Diane

### Dataset download

To download dataset, kindly install UV python package manager, then prepare your API keys and upload a `kaggle.json` file in `dataset/.kaggle` folder.
The file should contain the following:

```json
{"username": "your-username", "key": "your-api-key"}
```

Once UV and kaggle is set up run `uv sync` and `uv run scripts/dataset/kaggle.py` in the dataset folder.

*Optionally*, The user may `uv run scripts/dataset/embeddings.py` to encode the embeddings of the abstracts in the first 100000 documents.

### Run Initial Pipeline

To run the initial naive search pipeline, the embeddings parquet must be present in the `data/310` directory

1. To build the C++ Naive Search class, run the following command from root:
```shell
uv run python src/search/setup.py build_ext --inplace
```
2. To run the main file simply execute the following command from root:
```shell
uv run python src\main.py
```
