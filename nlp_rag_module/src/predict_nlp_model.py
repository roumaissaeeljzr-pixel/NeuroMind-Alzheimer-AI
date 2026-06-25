from pathlib import Path

import joblib
import torch
from transformers import AutoTokenizer

from model_architecture import HybridRoBERTa


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "model" / "best_roberta_hybrid_seed40.pt"
SCALER_PATH = BASE_DIR / "model" / "scaler_hybrid_seed40.pkl"
TOKENIZER_DIR = BASE_DIR / "model" / "tokenizer"


FEATURE_COLS = [
    "n_filled_pauses",
    "n_phon_fragments",
    "n_paralinguistic",
    "n_retracings",
    "n_unintelligible",
    "n_pauses",
    "entryage",
    "sex",
    "educ"
]


LABELS = {
    0: "Control",
    1: "ProbableAD"
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_nlp_model():
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR),
        use_fast=True
    )

    scaler = joblib.load(SCALER_PATH)

    model = HybridRoBERTa(n_features=len(FEATURE_COLS))
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, tokenizer, scaler, device


def predict_patient(transcript, feature_values):
    """
    feature_values doit être une liste de 9 valeurs dans cet ordre :

    [
        n_filled_pauses,
        n_phon_fragments,
        n_paralinguistic,
        n_retracings,
        n_unintelligible,
        n_pauses,
        entryage,
        sex,
        educ
    ]
    """

    if len(feature_values) != len(FEATURE_COLS):
        raise ValueError(
            f"Il faut {len(FEATURE_COLS)} features, mais tu as donné {len(feature_values)}."
        )

    model, tokenizer, scaler, device = load_nlp_model()

    encoding = tokenizer(
        transcript,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    features_scaled = scaler.transform([feature_values])
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    features_tensor = features_tensor.to(device)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            features=features_tensor
        )

        probs = torch.softmax(logits, dim=1)
        predicted_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_id].item()

    prediction = LABELS[predicted_id]

    return prediction, confidence


if __name__ == "__main__":
    transcript = """
    The boy is taking cookies from the jar.
    The mother is washing dishes.
    The water is falling.
    I don't know... maybe the boy will fall.
    """

    feature_values = [
        2,   # n_filled_pauses
        0,   # n_phon_fragments
        1,   # n_paralinguistic
        1,   # n_retracings
        0,   # n_unintelligible
        4,   # n_pauses
        72,  # entryage
        1,   # sex
        12   # educ
    ]

    prediction, confidence = predict_patient(transcript, feature_values)

    print("Prediction:", prediction)
    print("Confidence:", round(confidence, 4))