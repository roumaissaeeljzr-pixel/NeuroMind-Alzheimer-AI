"""
Brain extraction / skull stripping helper entry point for the CNN module.

The project expects preprocessed 3D MRI files to be stored in:
cnn_module/data/ADNI_SKULL_STRIPPED/
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "ADNI_SKULL_STRIPPED"


def main() -> None:
    raise NotImplementedError(
        "The original brain_extractor.py script was not present in this "
        "workspace. Add skull-stripping or preprocessing logic here and write "
        f"outputs to {OUTPUT_DIR}."
    )


if __name__ == "__main__":
    main()
