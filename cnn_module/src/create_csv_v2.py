"""
Metadata CSV preparation entry point for the independent CNN module.

Expected output:
cnn_module/data/dataset_preprocessed.csv

The CSV should reference preprocessed ADNI MRI volumes stored under:
cnn_module/data/ADNI_SKULL_STRIPPED/
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_CSV = DATA_DIR / "dataset_preprocessed.csv"


def main() -> None:
    raise NotImplementedError(
        "The original create_csv_v2.py script was not present in this workspace. "
        "Add the ADNI metadata parsing logic here and save the result to "
        f"{OUTPUT_CSV}."
    )


if __name__ == "__main__":
    main()
