from pathlib import Path
import argparse
import json
import warnings

import pandas as pd

from main_pipeline import run_full_pipeline
from predict_nlp_model import FEATURE_COLS


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = BASE_DIR / "data" / "dataset_pitt_cookie.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def normalize_label(label):
    return str(label).lower().replace(" ", "").replace("_", "")


def clean_source_path(source):
    return Path(str(source)).name


def source_display_name(source):
    return (
        Path(str(source)).stem
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )


def run_one_patient(df, row_index):
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

    result = run_full_pipeline(
        transcript=transcript,
        feature_values=feature_values
    )

    prediction = result["prediction"]
    confidence = result["confidence"]

    correct = normalize_label(true_label) == normalize_label(prediction)

    sources = [
        {
            "source": source_display_name(src["source"]),
            "page": src["page"]
        }
        for src in result["sources"]
    ]

    structured_result = {
        "row_index": int(row_index),
        "true_label": str(true_label),
        "prediction": str(prediction),
        "confidence": round(float(confidence), 4),
        "correct_prediction": bool(correct),
        "transcript": str(transcript),
        "features": {
            name: float(value) if isinstance(value, (int, float)) else str(value)
            for name, value in zip(FEATURE_COLS, feature_values)
        },
        "rag_explanation": result["explanation"],
        "sources": sources
    }

    return structured_result


def print_report(result):
    print("\n" + "=" * 80)
    print(f"INDEX PATIENT: {result['row_index']}")
    print("=" * 80)

    print(f"Classe réelle        : {result['true_label']}")
    print(f"Classe prédite       : {result['prediction']}")
    print(f"Niveau de confiance  : {result['confidence']}")
    print(f"Prédiction correcte  : {result['correct_prediction']}")

    print("\n--- Extrait de transcription ---")
    print(result["transcript"][:700])

    print("\n--- Indicateurs ---")
    for name, value in result["features"].items():
        print(f"{name}: {value}")

    print("\n--- Explication ---")
    print(result["rag_explanation"])

    print("\n--- Sources utilisées ---")
    for src in result["sources"]:
        print(f"- {src['source']}")

    print("=" * 80)


def save_results(results):
    OUTPUT_DIR.mkdir(exist_ok=True)

    json_path = OUTPUT_DIR / "rag_report_results.json"
    csv_path = OUTPUT_DIR / "rag_report_summary.csv"
    txt_path = OUTPUT_DIR / "rag_report_results.txt"

    # Save JSON complet
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Save CSV résumé
    summary_rows = []
    for res in results:
        summary_rows.append({
            "row_index": res["row_index"],
            "true_label": res["true_label"],
            "prediction": res["prediction"],
            "confidence": res["confidence"],
            "correct_prediction": res["correct_prediction"],
            "sources": "; ".join([s["source"] for s in res["sources"]])
        })

    pd.DataFrame(summary_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Save TXT lisible pour le rapport
    with open(txt_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write("=" * 80 + "\n")
            f.write(f"INDEX PATIENT: {res['row_index']}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Classe réelle : {res['true_label']}\n")
            f.write(f"Classe prédite : {res['prediction']}\n")
            f.write(f"Niveau de confiance : {res['confidence']}\n")
            f.write(f"Prédiction correcte : {res['correct_prediction']}\n\n")
            f.write("Extrait de transcription :\n")
            f.write(res["transcript"][:700] + "\n\n")
            f.write("Explication :\n")
            f.write(res["rag_explanation"] + "\n\n")
            f.write("Sources utilisées :\n")
            for src in res["sources"]:
                f.write(f"- {src['source']}\n")
            f.write("\n\n")

    print("\nRésultats sauvegardés :")
    print(f"- {json_path}")
    print(f"- {csv_path}")
    print(f"- {txt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indices",
        type=str,
        default="0",
        help="Patient row indices separated by commas. Example: 0,1,10,20"
    )

    args = parser.parse_args()

    indices = [int(x.strip()) for x in args.indices.split(",")]

    df = pd.read_csv(DATA_PATH)

    all_results = []

    for idx in indices:
        result = run_one_patient(df, idx)
        print_report(result)
        all_results.append(result)

    save_results(all_results)


if __name__ == "__main__":
    main()
