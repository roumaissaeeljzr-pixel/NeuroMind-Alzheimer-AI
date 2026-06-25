from pathlib import Path
import pandas as pd

from main_pipeline import run_full_pipeline
from predict_nlp_model import FEATURE_COLS


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "dataset_pitt_cookie.csv"


def run_patient_from_dataset(row_index=0):
    df = pd.read_csv(DATA_PATH)

    row = df.iloc[row_index]

    transcript = row["transcript"]

    feature_values = [
        row["n_filled_pauses"],
        row["n_phon_fragments"],
        row["n_paralinguistic"],
        row["n_retracings"],
        row["n_unintelligible"],
        row["n_pauses"],
        row["entryage"],
        row["sex"],
        row["educ"],
    ]

    true_label = row["dx_label"]

    print("=" * 70)
    print("REAL PATIENT FROM DATASET")
    print("=" * 70)

    print("Row index:", row_index)
    print("True label:", true_label)

    print("\nFeatures used:")
    for name, value in zip(FEATURE_COLS, feature_values):
        print(f"- {name}: {value}")

    print("\nTranscript:")
    print(transcript[:1000])

    result = run_full_pipeline(
        transcript=transcript,
        feature_values=feature_values
    )

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(result["explanation"])

    print("\nSources utilisées :")
    for source in result["sources"]:
        print(source)


if __name__ == "__main__":
    run_patient_from_dataset(row_index=0)