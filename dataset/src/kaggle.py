import kagglehub
from pathlib import Path
import shutil

def main() -> None:
    downloaded_path_str = kagglehub.dataset_download("Cornell-University/arxiv")

    source_path = Path(downloaded_path_str)
    print("Path to dataset files:", source_path)

    project_root = Path(__file__).resolve().parent.parent.parent
    dst = project_root / "data" / source_path.name

    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"Moving dataset to: {dst}")
    shutil.copytree(source_path, dst, dirs_exist_ok=True)

if __name__ == "__main__":
    main()
