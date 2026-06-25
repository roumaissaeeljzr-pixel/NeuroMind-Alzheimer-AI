from predict_nlp_model import predict_patient, FEATURE_COLS
from rag_explainer import explain_prediction_with_rag


def run_full_pipeline(transcript, feature_values):
    prediction, confidence = predict_patient(
        transcript=transcript,
        feature_values=feature_values
    )

    rag_result = explain_prediction_with_rag(
        transcript=transcript,
        prediction=prediction,
        confidence=confidence,
        k=4
    )

    return rag_result


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

    print("Features order:")
    for name, value in zip(FEATURE_COLS, feature_values):
        print(f"- {name}: {value}")

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