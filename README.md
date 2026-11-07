
### Dataset download

To download dataset, kindly install UV python package manager, then prepare your API keys and upload a `kaggle.json` file in `dataset/.kaggle` folder.
The file should contain the following:

```json
{"username": "your-username", "key": "your-api-key"}
```

Once UV and kaggle is set up run `uv sync` and `uv run src/kaggle.py` in the dataset folder.
