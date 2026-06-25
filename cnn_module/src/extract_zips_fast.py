"""
Archive extraction helper for ADNI MRI files.

Keep extracted and preprocessed image files inside cnn_module/data so that the
CNN module remains independent from the NLP + RAG module.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def main() -> None:
    raise NotImplementedError(
        "The original extract_zips_fast.py script was not present in this "
        "workspace. Add archive extraction logic here if ADNI files are stored "
        "as zip archives."
    )


if __name__ == "__main__":
    main()
