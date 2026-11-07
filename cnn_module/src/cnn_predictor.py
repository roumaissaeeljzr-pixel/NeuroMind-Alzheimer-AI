from pathlib import Path
import argparse
import importlib

import numpy as np
import torch

from cnn_model import InflatedResNet3D, MultimodalInflatedResNet3D, preprocess_mri_array


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_resnet3d_model.pth"
DEFAULT_MRI_DIR = BASE_DIR / "data" / "ADNI_SKULL_STRIPPED"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = {
    0: "CN",
    1: "AD"
}


def _extract_state_dict(checkpoint):
    """
    Accept raw state_dict checkpoints and common wrapped checkpoint formats.
    """
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _remove_prefix_from_keys(state_dict, prefix: str):
    return {
        key.replace(prefix, "", 1) if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def _is_multimodal_checkpoint(state_dict) -> bool:
    return any(key.startswith("backbone.") for key in state_dict) and any(
        key.startswith("clinical_mlp.") for key in state_dict
    )


def load_nifti_volume(mri_path):
    """
    Load a .nii or .nii.gz MRI file and return a clean contiguous numpy array.
    """
    try:
        nib = importlib.import_module("nibabel")
    except ImportError as exc:
        raise ImportError(
            "nibabel is required for loading .nii/.nii.gz MRI files. "
            "Install it with: python -m pip install nibabel"
        ) from exc

    mri_path = Path(mri_path)

    if not mri_path.exists():
        raise FileNotFoundError(f"MRI file not found: {mri_path}")

    image = nib.load(str(mri_path))
    image = nib.as_closest_canonical(image)

    volume = image.get_fdata(dtype=np.float32)

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D MRI volume, got shape {volume.shape}.")

    # Important correction: remove negative strides
    volume = np.ascontiguousarray(volume.copy()).astype(np.float32)

    return volume


def load_cnn_model():
    """
    Load CNN model and handle checkpoint keys with or without 'model.' prefix.
    """
    print(f"Loading Brain Imaging model from: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Place best_resnet3d_model.pth in cnn_module/models/"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    state_dict = _extract_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(
            "Unsupported Brain Imaging checkpoint format. "
            f"Expected a state_dict-like object, got {type(state_dict).__name__}."
        )

    checkpoint_keys = list(state_dict.keys())
    if not checkpoint_keys:
        raise ValueError("Brain Imaging checkpoint does not contain model weights.")

    if all(key.startswith("model.") for key in checkpoint_keys):
        print("Removing 'model.' prefix from checkpoint keys...")
        state_dict = _remove_prefix_from_keys(state_dict, "model.")
        checkpoint_keys = list(state_dict.keys())

    if _is_multimodal_checkpoint(state_dict):
        model = MultimodalInflatedResNet3D(num_classes=2)
    else:
        model = InflatedResNet3D(num_classes=2)

    model_keys = list(model.state_dict().keys())

    first_model_key = model_keys[0]
    first_checkpoint_key = checkpoint_keys[0]

    if not first_checkpoint_key.startswith("model.") and first_model_key.startswith("model."):
        print("Adding 'model.' prefix to checkpoint keys...")
        state_dict = {
            "model." + key: value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    return model


def predict_mri(mri_path):
    """
    Predict CN or AD from one MRI file.
    """
    mri_path = Path(mri_path)

    model = load_cnn_model()

    volume = load_nifti_volume(mri_path)

    # Extra safety before preprocessing
    volume = np.ascontiguousarray(volume.copy()).astype(np.float32)

    tensor = preprocess_mri_array(volume)

    # Shape: [1, 1, 112, 112, 112]
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.amp.autocast(
            device_type="cuda",
            enabled=(DEVICE.type == "cuda")
        ):
            logits = model(tensor)
            probs = torch.softmax(logits.float(), dim=1)

    prob_cn = probs[0, 0].item()
    prob_ad = probs[0, 1].item()

    predicted_id = torch.argmax(probs, dim=1).item()

    prediction = LABELS[predicted_id]
    confidence = probs[0, predicted_id].item()

    return {
        "mri_path": str(mri_path),
        "prediction": prediction,
        "confidence": confidence,
        "prob_cn": prob_cn,
        "prob_ad": prob_ad
    }


def find_first_mri_file():
    """
    Find the first MRI file in ADNI_SKULL_STRIPPED.
    """
    nii_files = list(DEFAULT_MRI_DIR.rglob("*.nii")) + list(DEFAULT_MRI_DIR.rglob("*.nii.gz"))

    if not nii_files:
        raise FileNotFoundError(
            f"No .nii or .nii.gz files found in: {DEFAULT_MRI_DIR}"
        )

    return nii_files[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a .nii or .nii.gz MRI file"
    )

    args = parser.parse_args()

    if args.image is None:
        mri_path = find_first_mri_file()
        print(f"No image provided. Using first MRI found:\n{mri_path}")
    else:
        mri_path = args.image

    result = predict_mri(mri_path)

    print("=" * 70)
    print("CNN MRI PREDICTION")
    print("=" * 70)

    print(f"MRI file       : {result['mri_path']}")
    print(f"Prediction     : {result['prediction']}")
    print(f"Confidence     : {result['confidence']:.4f}")
    print(f"Probability CN : {result['prob_cn']:.4f}")
    print(f"Probability AD : {result['prob_ad']:.4f}")


if __name__ == "__main__":
    main()
