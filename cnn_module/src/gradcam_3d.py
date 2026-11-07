from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_model import preprocess_mri_array
from cnn_predictor import DEVICE, LABELS, load_cnn_model, load_nifti_volume


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "gradcam"
MPL_CONFIG_DIR = PROJECT_DIR / "outputs" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))


def get_target_layer(model: nn.Module) -> nn.Module:
    """
    Return a stable convolutional layer for visual explanation.

    layer3 is deeper and usually less noisy than layer2. layer4 remains a
    fallback when a more concentrated map is needed by a wrapped model.
    """
    possible_layers = (
        lambda: model.backbone.layer3[-1].conv2,
        lambda: model.model.layer3[-1].conv2,
        lambda: model.layer3[-1].conv2,
        lambda: model.backbone.layer4[-1].conv2,
        lambda: model.model.layer4[-1].conv2,
        lambda: model.layer4[-1].conv2,
    )
    for layer_getter in possible_layers:
        try:
            return layer_getter()
        except Exception:
            pass

    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            last_conv = module

    if last_conv is None:
        raise RuntimeError("No Conv3d layer found for visual explanation.")

    return last_conv


def _get_target_layer(model: nn.Module) -> nn.Module:
    return get_target_layer(model)


def normalize_for_display(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = array - float(array.min())
    max_value = float(array.max())

    if max_value < eps:
        return np.zeros_like(array, dtype=np.float32)

    return array / (max_value + eps)


VALID_ORIENTATIONS = {
    "axial": 2,
    "coronal": 1,
    "sagittal": 0,
}


def normalize_orientation(orientation: str) -> str:
    orientation = str(orientation or "axial").strip().lower()
    if orientation not in VALID_ORIENTATIONS:
        valid = ", ".join(sorted(VALID_ORIENTATIONS))
        raise ValueError(f"Unsupported Grad-CAM orientation: {orientation}. Use one of: {valid}.")
    return orientation


def normalize_generation_orientation(orientation: str) -> str:
    orientation = str(orientation or "multi").strip().lower().replace("_", "-")
    if orientation in {"multi", "multi-axis", "all", "3-axis", "3-axes", "three-views"}:
        return "multi"
    return normalize_orientation(orientation)


def normalize_display_mode(display_mode: str) -> str:
    display_mode = str(display_mode or "overlay").strip().lower().replace("_", "-")
    if display_mode in {"overlay", "overlay-view", "mri-overlay", "image-overlay"}:
        return "overlay"
    if display_mode in {"heatmap", "heatmap-only", "cam-only", "map"}:
        return "heatmap"
    raise ValueError("display_mode must be 'overlay' or 'heatmap'.")


def _smooth_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    Smooth the heatmap to reduce blocky visual explanation artifacts.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(heatmap, sigma=1.8)
    except Exception:
        return heatmap


def _keep_largest_regions(mask: np.ndarray, max_regions: int = 2) -> np.ndarray:
    """
    Remove isolated speckles and keep only the largest activated regions.
    """
    mask = np.asarray(mask, dtype=bool)
    if not bool(mask.any()):
        return mask

    try:
        from scipy.ndimage import binary_closing, binary_opening, label

        cleaned = binary_opening(mask, iterations=1)
        cleaned = binary_closing(cleaned, iterations=1)
        if not bool(cleaned.any()):
            cleaned = mask

        labels, region_count = label(cleaned)
        if region_count <= max_regions:
            return cleaned

        region_sizes = np.bincount(labels.ravel())
        region_sizes[0] = 0
        min_region_size = max(6, int(mask.size * 0.002))
        region_sizes[region_sizes < min_region_size] = 0

        border_labels = np.unique(
            np.concatenate(
                [
                    labels[0, :],
                    labels[-1, :],
                    labels[:, 0],
                    labels[:, -1],
                ]
            )
        )
        border_labels = border_labels[border_labels != 0]
        non_border_labels = np.setdiff1d(
            np.arange(1, region_count + 1),
            border_labels,
            assume_unique=False,
        )
        if non_border_labels.size and np.any(region_sizes[non_border_labels] > 0):
            region_sizes[border_labels] = 0

        if not np.any(region_sizes > 0):
            region_sizes = np.bincount(labels.ravel())
            region_sizes[0] = 0

        keep_labels = np.argsort(region_sizes)[-max_regions:]
        return np.isin(labels, keep_labels)
    except Exception:
        return mask


def _clean_heatmap_for_overlay(
    heatmap_slice: np.ndarray,
    image_slice: np.ndarray,
    percentile: float = 88.0,
    min_threshold: float = 0.48,
    brain_threshold: float = 0.05,
    max_regions: int = 2,
) -> np.ma.MaskedArray:
    """
    Keep only meaningful visual explanation activations.

    Weak activations and black background areas become transparent.
    """
    heatmap_slice = np.nan_to_num(np.asarray(heatmap_slice, dtype=np.float32))
    image_slice = np.nan_to_num(np.asarray(image_slice, dtype=np.float32))
    heatmap_slice = np.maximum(_smooth_heatmap(heatmap_slice), 0.0)

    if float(heatmap_slice.max()) > float(heatmap_slice.min()):
        heatmap_slice = (heatmap_slice - float(heatmap_slice.min())) / (
            float(heatmap_slice.max()) - float(heatmap_slice.min())
        )
    else:
        heatmap_slice = np.zeros_like(heatmap_slice, dtype=np.float32)

    image_display = normalize_for_display(image_slice)
    brain_mask = image_display > float(brain_threshold)
    heatmap_slice = heatmap_slice * brain_mask

    valid_values = heatmap_slice[brain_mask & (heatmap_slice > 1e-6)]
    if valid_values.size:
        threshold = max(float(np.percentile(valid_values, percentile)), float(min_threshold))
    else:
        threshold = float(min_threshold)

    active_mask = np.logical_and(heatmap_slice >= threshold, brain_mask)
    active_mask = _keep_largest_regions(active_mask, max_regions=max_regions)
    mask = ~active_mask
    visible_values = heatmap_slice[active_mask]
    if visible_values.size:
        visible_min = float(visible_values.min())
        visible_max = float(visible_values.max())
        if visible_max > visible_min + 1e-8:
            heatmap_slice = (heatmap_slice - visible_min) / (visible_max - visible_min)
            heatmap_slice = np.clip(heatmap_slice, 0.0, 1.0)

    return np.ma.masked_where(mask, heatmap_slice)


def masked_heatmap(
    cam_slice: np.ndarray,
    base_slice: np.ndarray | None = None,
    threshold: float = 0.48,
    percentile: float = 88.0,
    brain_threshold: float = 0.05,
) -> np.ma.MaskedArray:
    """
    Normalize a heatmap slice and hide weak activations before display.

    This keeps the MRI background clean instead of tinting the whole image.
    """
    if base_slice is None:
        base_slice = np.ones_like(cam_slice, dtype=np.float32)
    return _clean_heatmap_for_overlay(
        heatmap_slice=cam_slice,
        image_slice=base_slice,
        percentile=percentile,
        min_threshold=threshold,
        brain_threshold=brain_threshold,
    )


def _jet_colormap_array(values: np.ndarray) -> np.ndarray:
    """
    Small jet-like colormap for the Pillow fallback.

    Matplotlib uses the real jet/turbo colormap; this fallback only preserves
    the blue-green-yellow-red visual direction if matplotlib is unavailable.
    """
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def overlay_rgb_array(
    mri_slice: np.ndarray,
    cam_slice: np.ndarray,
    alpha: float = 0.60,
    threshold: float = 0.48,
    percentile: float = 88.0,
    display_mode: str = "overlay",
) -> np.ndarray:
    display_mode = normalize_display_mode(display_mode)
    base = normalize_for_display(mri_slice)
    masked = masked_heatmap(
        cam_slice,
        base_slice=mri_slice,
        threshold=threshold,
        percentile=percentile,
    )
    heatmap = masked.filled(0.0)
    mask = np.ma.getmaskarray(masked)
    heatmap_rgb = _jet_colormap_array(heatmap)

    if display_mode == "heatmap":
        output = heatmap_rgb * (~mask)[..., None]
    else:
        base_rgb = np.repeat(base[..., None], 3, axis=-1)
        alpha_map = np.where(mask, 0.0, float(alpha))[..., None]
        output = base_rgb * (1.0 - alpha_map) + heatmap_rgb * alpha_map

    return np.clip(output * 255.0, 0, 255).astype(np.uint8)


def oriented_slice(volume: np.ndarray, orientation: str, index: int) -> np.ndarray:
    orientation = normalize_orientation(orientation)

    if orientation == "sagittal":
        slice_2d = volume[index, :, :]
    elif orientation == "coronal":
        slice_2d = volume[:, index, :]
    else:
        slice_2d = volume[:, :, index]

    return np.rot90(slice_2d)


def save_overlay_with_pillow(
    mri_slice: np.ndarray,
    cam_slice: np.ndarray,
    output_path: Path,
    display_mode: str = "overlay",
    alpha: float = 0.60,
    threshold: float = 0.48,
    percentile: float = 88.0,
) -> Path:
    """
    Save a Grad-CAM overlay with Pillow as a fallback when matplotlib is absent.
    """
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Grad-CAM image export needs either matplotlib or Pillow. "
            "Install one of them in the Streamlit environment."
        ) from exc

    Image.fromarray(
        overlay_rgb_array(
            mri_slice,
            cam_slice,
            alpha=alpha,
            threshold=threshold,
            percentile=percentile,
            display_mode=display_mode,
        )
    ).save(output_path)
    return output_path


def save_overlay_with_matplotlib(
    mri_slice: np.ndarray,
    cam_slice: np.ndarray,
    output_path: Path,
    target_label: str,
    display_mode: str = "overlay",
    alpha: float = 0.60,
    threshold: float = 0.48,
    percentile: float = 88.0,
    colormap: str = "turbo",
) -> Path:
    """
    Save a Grad-CAM overlay with matplotlib when it is available.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_mode = normalize_display_mode(display_mode)
    base_img = normalize_for_display(mri_slice)
    heatmap_masked = masked_heatmap(
        cam_slice,
        base_slice=mri_slice,
        threshold=threshold,
        percentile=percentile,
    )

    try:
        cmap = plt.get_cmap(colormap).copy()
    except ValueError:
        cmap = plt.get_cmap("jet").copy()
    cmap.set_bad(alpha=0.0)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    if display_mode == "heatmap":
        ax.imshow(np.zeros_like(base_img), cmap="gray", vmin=0, vmax=1)
        ax.imshow(
            heatmap_masked,
            cmap=cmap,
            vmin=0,
            vmax=1,
            alpha=1.0,
            interpolation="bilinear",
        )
    else:
        ax.imshow(base_img, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
        ax.imshow(
            heatmap_masked,
            cmap=cmap,
            vmin=0,
            vmax=1,
            alpha=alpha,
            interpolation="bilinear",
        )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    return output_path


def compute_gradcam(
    mri_path: str | Path,
    target_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute a 3D Grad-CAM heatmap for one MRI volume.

    Returns:
        preprocessed MRI volume, Grad-CAM volume, target class index.
    """
    model = load_cnn_model()
    target_layer = get_target_layer(model)

    activations = None
    gradients = None

    def forward_hook(_module, _inputs, output):
        nonlocal activations
        activations = output

    def backward_hook(_module, _grad_inputs, grad_outputs):
        nonlocal gradients
        gradients = grad_outputs[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        volume = load_nifti_volume(mri_path)
        tensor = preprocess_mri_array(volume).unsqueeze(0).to(DEVICE)

        logits = model(tensor)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        model.zero_grad(set_to_none=True)
        score.backward()

        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=tensor.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

        cam_np = cam.squeeze().detach().cpu().numpy()
        mri_np = tensor.squeeze().detach().cpu().numpy()

        return mri_np, normalize_for_display(cam_np), target_class

    finally:
        forward_handle.remove()
        backward_handle.remove()


def save_gradcam_overlay(
    mri_volume: np.ndarray,
    cam_volume: np.ndarray,
    output_path: str | Path,
    target_label: str,
    slice_index: int | None = None,
    orientation: str = "axial",
    display_mode: str = "overlay",
    alpha: float = 0.60,
    threshold: float = 0.48,
    percentile: float = 88.0,
    colormap: str = "turbo",
) -> Path:
    """
    Save a 2D oriented slice with a Grad-CAM heatmap overlay.
    """
    orientation = normalize_orientation(orientation)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    axis = VALID_ORIENTATIONS[orientation]
    if slice_index is None:
        slice_index = mri_volume.shape[axis] // 2

    mri_slice = normalize_for_display(
        oriented_slice(mri_volume, orientation, slice_index)
    )
    cam_slice = normalize_for_display(
        oriented_slice(cam_volume, orientation, slice_index)
    )

    try:
        return save_overlay_with_matplotlib(
            mri_slice=mri_slice,
            cam_slice=cam_slice,
            output_path=output_path,
            target_label=target_label,
            display_mode=display_mode,
            alpha=alpha,
            threshold=threshold,
            percentile=percentile,
            colormap=colormap,
        )
    except ModuleNotFoundError:
        return save_overlay_with_pillow(
            mri_slice=mri_slice,
            cam_slice=cam_slice,
            output_path=output_path,
            display_mode=display_mode,
            alpha=alpha,
            threshold=threshold,
            percentile=percentile,
        )


def _safe_name(value: str, fallback: str = "scan") -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value or "").strip()
    ).strip("_")
    return (cleaned or fallback)[:60]


def select_visual_slice_indices(depth: int, num_slices: int) -> list[int]:
    return _slice_indices(depth, num_slices, 0.30, 0.75)


def _slice_indices(
    depth: int,
    num_slices: int,
    start_ratio: float = 0.30,
    end_ratio: float = 0.75,
) -> list[int]:
    if depth <= 0:
        raise ValueError("MRI volume depth must be greater than zero.")
    if num_slices <= 0:
        raise ValueError("num_slices must be greater than zero.")
    if depth == 1:
        return [0 for _ in range(num_slices)]

    start = max(0, int(round(depth * float(start_ratio))))
    end = min(depth - 1, int(round(depth * float(end_ratio))))
    if end <= start:
        start, end = 0, depth - 1

    return [int(round(index)) for index in np.linspace(start, end, num_slices)]


def multi_axis_slice_indices(volume_shape: tuple[int, ...], num_slices: int) -> dict[str, list[int]]:
    return {
        "axial": _slice_indices(volume_shape[VALID_ORIENTATIONS["axial"]], num_slices, 0.30, 0.75),
        "sagittal": _slice_indices(volume_shape[VALID_ORIENTATIONS["sagittal"]], num_slices, 0.30, 0.70),
        "coronal": _slice_indices(volume_shape[VALID_ORIENTATIONS["coronal"]], num_slices, 0.30, 0.75),
    }


def save_multi_axis_visualization_with_matplotlib(
    mri_volume: np.ndarray,
    cam_volume: np.ndarray,
    output_path: str | Path,
    num_slices: int = 5,
    display_mode: str = "overlay",
    alpha: float = 0.60,
    threshold: float = 0.48,
    percentile: float = 88.0,
    colormap: str = "turbo",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_mode = normalize_display_mode(display_mode)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    slice_map = multi_axis_slice_indices(mri_volume.shape, num_slices)
    view_rows = [
        ("axial", "Axial view\nTop to bottom"),
        ("sagittal", "Sagittal view\nLeft to right"),
        ("coronal", "Coronal view\nFront to back"),
    ]

    fig_width = max(10.0, num_slices * 3.3)
    fig_height = 9.7
    fig, axes = plt.subplots(
        len(view_rows),
        num_slices,
        figsize=(fig_width, fig_height),
        facecolor="#fbfaf5",
    )
    axes = np.asarray(axes).reshape(len(view_rows), num_slices)

    try:
        fig.suptitle(
            "Brain Scan Visual Explanation",
            fontsize=17,
            fontweight="bold",
            color="#20242b",
            y=0.985,
        )
        fig.text(
            0.5,
            0.942,
            (
                "Axial, sagittal, and coronal views - Overlay view"
                if display_mode == "overlay"
                else "Axial, sagittal, and coronal views - Heatmap only"
            ),
            ha="center",
            fontsize=11,
            color="#596273",
        )

        for row_index, (orientation, row_label) in enumerate(view_rows):
            for col_index, slice_index in enumerate(slice_map[orientation]):
                mri_slice = normalize_for_display(
                    oriented_slice(mri_volume, orientation, slice_index)
                )
                cam_slice = normalize_for_display(
                    oriented_slice(cam_volume, orientation, slice_index)
                )
                heatmap_masked = masked_heatmap(
                    cam_slice,
                    base_slice=mri_slice,
                    threshold=threshold,
                    percentile=percentile,
                )
                axis = axes[row_index, col_index]
                try:
                    cmap = plt.get_cmap(colormap).copy()
                except ValueError:
                    cmap = plt.get_cmap("jet").copy()
                cmap.set_bad(alpha=0.0)
                if display_mode == "heatmap":
                    axis.imshow(np.zeros_like(mri_slice), cmap="gray", vmin=0, vmax=1)
                    axis.imshow(
                        heatmap_masked,
                        cmap=cmap,
                        vmin=0,
                        vmax=1,
                        alpha=1.0,
                        interpolation="bilinear",
                    )
                else:
                    axis.imshow(mri_slice, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
                    axis.imshow(
                        heatmap_masked,
                        cmap=cmap,
                        vmin=0,
                        vmax=1,
                        alpha=alpha,
                        interpolation="bilinear",
                    )
                axis.set_title(f"Slice {col_index + 1}", fontsize=9, color="#20242b")
                axis.axis("off")

        row_label_positions = [0.735, 0.475, 0.215]
        for row_index, (_orientation, row_label) in enumerate(view_rows):
            fig.text(
                0.035,
                row_label_positions[row_index],
                row_label,
                ha="center",
                va="center",
                rotation=90,
                fontsize=11,
                fontweight="bold",
                color="#0f8a68",
            )

        fig.text(
            0.5,
            0.035,
            "Blue / dark areas = lower influence  |  Yellow / red areas = stronger influence  |  Highlighted areas do not confirm disease",
            ha="center",
            fontsize=10,
            color="#6b5a1d",
        )
        fig.text(
            0.5,
            0.012,
            "This visual explanation is a decision-support aid and must be reviewed by a healthcare professional.",
            ha="center",
            fontsize=9,
            color="#596273",
        )
        plt.tight_layout(rect=[0.07, 0.065, 0.995, 0.925])
        fig.savefig(output_path, dpi=170, bbox_inches="tight", pad_inches=0.08)
    finally:
        plt.close(fig)

    return output_path


def save_multi_axis_visualization_with_pillow(
    mri_volume: np.ndarray,
    cam_volume: np.ndarray,
    output_path: str | Path,
    num_slices: int = 5,
    display_mode: str = "overlay",
    alpha: float = 0.60,
    threshold: float = 0.48,
    percentile: float = 88.0,
) -> Path:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Visual explanation image export needs either matplotlib or Pillow."
        ) from exc

    display_mode = normalize_display_mode(display_mode)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    slice_map = multi_axis_slice_indices(mri_volume.shape, num_slices)
    view_rows = [
        ("axial", "Axial view - top to bottom"),
        ("sagittal", "Sagittal view - left to right"),
        ("coronal", "Coronal view - front to back"),
    ]

    cell_size = 240
    label_width = 180
    title_height = 105
    row_gap = 34
    bottom_height = 70
    width = label_width + num_slices * cell_size
    height = title_height + len(view_rows) * cell_size + (len(view_rows) - 1) * row_gap + bottom_height

    canvas = Image.new("RGB", (width, height), "#fbfaf5")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default()
    text_font = ImageFont.load_default()

    draw.text((label_width, 24), "Brain Scan Visual Explanation", fill="#20242b", font=title_font)
    subtitle = (
        "Axial, sagittal, and coronal views - Overlay view"
        if display_mode == "overlay"
        else "Axial, sagittal, and coronal views - Heatmap only"
    )
    draw.text((label_width, 52), subtitle, fill="#596273", font=text_font)

    y_offset = title_height
    for orientation, row_label in view_rows:
        draw.text((18, y_offset + cell_size // 2 - 10), row_label, fill="#0f8a68", font=text_font)
        for col_index, slice_index in enumerate(slice_map[orientation]):
            mri_slice = oriented_slice(mri_volume, orientation, slice_index)
            cam_slice = oriented_slice(cam_volume, orientation, slice_index)
            overlay = Image.fromarray(
                overlay_rgb_array(
                    mri_slice,
                    cam_slice,
                    alpha=alpha,
                    threshold=threshold,
                    percentile=percentile,
                    display_mode=display_mode,
                )
            )
            overlay = overlay.resize((cell_size - 12, cell_size - 12), Image.BILINEAR)
            x_offset = label_width + col_index * cell_size + 6
            canvas.paste(overlay, (x_offset, y_offset + 6))
            draw.text((x_offset + 6, y_offset + cell_size - 20), f"Slice {col_index + 1}", fill="#20242b", font=text_font)
        y_offset += cell_size + row_gap

    draw.text(
        (label_width, height - 52),
        "Blue / dark = lower influence | Yellow / red = stronger influence | Highlighted areas do not confirm disease",
        fill="#6b5a1d",
        font=text_font,
    )
    draw.text(
        (label_width, height - 28),
        "This visual explanation must be reviewed by a healthcare professional.",
        fill="#596273",
        font=text_font,
    )
    canvas.save(output_path)
    return output_path


def save_multi_axis_visualization(
    mri_volume: np.ndarray,
    cam_volume: np.ndarray,
    output_path: str | Path,
    num_slices: int = 5,
    display_mode: str = "overlay",
    alpha: float = 0.60,
    threshold: float = 0.48,
    percentile: float = 88.0,
    colormap: str = "turbo",
) -> Path:
    try:
        return save_multi_axis_visualization_with_matplotlib(
            mri_volume=mri_volume,
            cam_volume=cam_volume,
            output_path=output_path,
            num_slices=num_slices,
            display_mode=display_mode,
            alpha=alpha,
            threshold=threshold,
            percentile=percentile,
            colormap=colormap,
        )
    except ModuleNotFoundError:
        return save_multi_axis_visualization_with_pillow(
            mri_volume=mri_volume,
            cam_volume=cam_volume,
            output_path=output_path,
            num_slices=num_slices,
            display_mode=display_mode,
            alpha=alpha,
            threshold=threshold,
            percentile=percentile,
        )


def plot_visual_explanation_multi_axes(
    original_vol: np.ndarray,
    heatmap_vol: np.ndarray,
    filename: str | Path,
    num_slices: int = 5,
) -> Path:
    """
    Patient-friendly multi-axis visual explanation.

    Uses grayscale MRI slices with a localized, cleaned overlay.
    """
    return save_multi_axis_visualization(
        mri_volume=original_vol,
        cam_volume=heatmap_vol,
        output_path=filename,
        num_slices=num_slices,
        display_mode="overlay",
        alpha=0.60,
        threshold=0.48,
        percentile=88.0,
        colormap="turbo",
    )


def plot_visual_explanation_single_axis(
    original_vol: np.ndarray,
    heatmap_vol: np.ndarray,
    orientation: str,
    filename: str | Path,
    num_slices: int = 8,
) -> Path:
    """
    Patient-friendly single-axis visual explanation.

    This keeps the same cleaning, thresholding and largest-region filtering as
    the multi-axis visualization.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orientation = normalize_orientation(orientation)
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    if orientation == "sagittal":
        indices = _slice_indices(original_vol.shape[0], num_slices, 0.30, 0.70)
        title = "Sagittal Visual Explanation"
    elif orientation == "coronal":
        indices = _slice_indices(original_vol.shape[1], num_slices, 0.30, 0.75)
        title = "Coronal Visual Explanation"
    else:
        indices = _slice_indices(original_vol.shape[2], num_slices, 0.30, 0.75)
        title = "Axial Visual Explanation"

    n_cols = min(4, len(indices))
    n_rows = int(np.ceil(len(indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3.3 * n_rows))
    axes = np.asarray(axes).reshape(n_rows, n_cols)
    fig.patch.set_facecolor("#F8F4EC")

    fig.suptitle(
        f"{title}\nOverlay view",
        fontsize=15,
        fontweight="bold",
        color="#202124",
    )

    try:
        cmap = plt.get_cmap("turbo").copy()
    except ValueError:
        cmap = plt.get_cmap("jet").copy()
    cmap.set_bad(alpha=0.0)

    for ax in axes.flat:
        ax.axis("off")
        ax.set_facecolor("#F8F4EC")

    for image_index, slice_index in enumerate(indices):
        ax = axes.flat[image_index]
        mri_slice = normalize_for_display(
            oriented_slice(original_vol, orientation, int(slice_index))
        )
        cam_slice = normalize_for_display(
            oriented_slice(heatmap_vol, orientation, int(slice_index))
        )
        heatmap_masked = masked_heatmap(
            cam_slice,
            base_slice=mri_slice,
            threshold=0.48,
            percentile=88.0,
        )
        ax.imshow(mri_slice, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
        ax.imshow(
            heatmap_masked,
            cmap=cmap,
            vmin=0,
            vmax=1,
            alpha=0.60,
            interpolation="bilinear",
        )
        ax.set_title(f"Slice {image_index + 1}", fontsize=9, color="#20242b")

    fig.text(
        0.5,
        0.025,
        "Blue / dark = lower influence | Yellow / red = stronger influence | Highlighted areas do not confirm disease",
        ha="center",
        fontsize=9,
        color="#202124",
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.90])
    fig.savefig(filename, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return filename


def _zone_from_position(y_position: float, x_position: float) -> str:
    vertical = "upper" if y_position < 1 / 3 else "lower" if y_position > 2 / 3 else "center"
    horizontal = "left" if x_position < 1 / 3 else "right" if x_position > 2 / 3 else "center"

    if vertical == "center" and horizontal == "center":
        return "center"
    if vertical == "center":
        return f"center {horizontal}"
    if horizontal == "center":
        return f"{vertical} center"
    return f"{vertical} {horizontal}"


def _brightest_grid_zone(heatmap: np.ndarray, bright_mask: np.ndarray) -> str:
    height, width = heatmap.shape
    row_edges = np.linspace(0, height, 4, dtype=int)
    col_edges = np.linspace(0, width, 4, dtype=int)
    zone_names = [
        ["upper left", "upper center", "upper right"],
        ["center left", "center", "center right"],
        ["lower left", "lower center", "lower right"],
    ]

    best_zone = "center"
    best_score = -1.0
    for row_index in range(3):
        for col_index in range(3):
            row_start, row_end = row_edges[row_index], row_edges[row_index + 1]
            col_start, col_end = col_edges[col_index], col_edges[col_index + 1]
            cell = heatmap[row_start:row_end, col_start:col_end]
            cell_mask = bright_mask[row_start:row_end, col_start:col_end]
            if cell.size == 0:
                continue

            if bool(cell_mask.any()):
                score = float((cell * cell_mask).sum())
            else:
                score = float(cell.sum()) * 0.25

            if score > best_score:
                best_score = score
                best_zone = zone_names[row_index][col_index]

    return best_zone


def describe_heatmap_position(heatmap_2d):
    """
    Return patient-safe visual metrics for one heatmap slice.

    These metrics are intentionally simple visual measurements. They describe
    color position and spread in the image only; they are not medical findings.
    """
    heatmap = normalize_for_display(np.asarray(heatmap_2d, dtype=np.float32))
    if heatmap.size == 0 or float(heatmap.max()) <= 1e-8:
        return {
            "intensity_level": "low",
            "highlight_area_percent": 0.0,
            "brightest_zone": "center",
            "spread_type": "focused",
        }

    active_threshold = 0.35
    bright_threshold = max(0.55, float(np.percentile(heatmap, 95)))
    active_mask = heatmap >= active_threshold
    bright_mask = heatmap >= bright_threshold
    if not bool(bright_mask.any()):
        bright_mask = heatmap == float(heatmap.max())

    highlight_area_percent = round(float(active_mask.mean() * 100.0), 1)
    p95 = float(np.percentile(heatmap, 95))

    if p95 < 0.30 or highlight_area_percent < 3.0:
        intensity_level = "low"
    elif p95 < 0.65 or highlight_area_percent < 14.0:
        intensity_level = "moderate"
    else:
        intensity_level = "high"

    if highlight_area_percent < 6.0:
        spread_type = "focused"
    elif highlight_area_percent < 20.0:
        spread_type = "medium"
    else:
        spread_type = "diffuse"

    weights = heatmap * bright_mask
    total_weight = float(weights.sum())
    if total_weight <= 1e-8:
        max_y, max_x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
        y_position = max_y / max(1, heatmap.shape[0] - 1)
        x_position = max_x / max(1, heatmap.shape[1] - 1)
    else:
        yy, xx = np.indices(heatmap.shape)
        y_position = float((yy * weights).sum() / total_weight) / max(1, heatmap.shape[0] - 1)
        x_position = float((xx * weights).sum() / total_weight) / max(1, heatmap.shape[1] - 1)

    return {
        "intensity_level": intensity_level,
        "highlight_area_percent": highlight_area_percent,
        "brightest_zone": _brightest_grid_zone(heatmap, bright_mask),
        "spread_type": spread_type,
    }


def _zone_phrase(zone: str) -> str:
    phrases = {
        "upper left": "near the upper left of this image",
        "upper center": "near the upper center of this image",
        "upper right": "near the upper right of this image",
        "center left": "on the image left side",
        "center": "near the center of this image",
        "center right": "on the image right side",
        "lower left": "near the lower left of this image",
        "lower center": "near the lower center of this image",
        "lower right": "near the lower right of this image",
    }
    return phrases.get(zone, "within this image")


def build_specific_slice_caption(slice_number, orientation, metrics):
    """
    Build a short patient-facing caption from code-computed heatmap metrics.

    This function does not call an LLM and does not infer anatomical regions.
    """
    zone = str(metrics.get("brightest_zone", "center"))
    intensity = str(metrics.get("intensity_level", "moderate"))
    spread = str(metrics.get("spread_type", "medium"))
    zone_text = _zone_phrase(zone)

    if intensity == "low":
        phrases = [
            f"Only a small highlighted area appears {zone_text}.",
            f"A limited bright area is visible {zone_text}.",
            f"This image slice shows a small colored area {zone_text}.",
            f"A small bright spot appears {zone_text}.",
            f"The visible color is limited and appears {zone_text}.",
            f"A narrow highlighted area is visible {zone_text}.",
            f"The brightest visible part is small and located {zone_text}.",
            f"A light colored spot appears {zone_text}.",
            f"Only a limited color area is seen {zone_text}.",
            f"A small localized highlight appears {zone_text}.",
            f"The color pattern is subtle and strongest {zone_text}.",
            f"A small bright focus is visible {zone_text}.",
        ]
        return f"Slice {slice_number} - {phrases[(slice_number - 1) % len(phrases)]}"

    if spread == "diffuse":
        phrases = [
            f"Colors are spread broadly, with the brightest part {zone_text}.",
            f"A wide colored pattern appears, strongest {zone_text}.",
            f"The colored area is wide and brightest {zone_text}.",
            f"Several colored areas are visible, with more brightness {zone_text}.",
            f"The color pattern covers a broad area, with a stronger part {zone_text}.",
            f"Colored areas extend across this slice, with the main brightness {zone_text}.",
            f"The visual highlight is wide, and its brightest part appears {zone_text}.",
            f"A broad colored area is visible, with stronger color {zone_text}.",
            f"The colors cover much of this image and are most visible {zone_text}.",
            f"This slice shows a wider color spread, strongest {zone_text}.",
            f"The highlighted pattern is broad, with a brighter focus {zone_text}.",
            f"Color intensity is spread out, with the brightest area {zone_text}.",
        ]
        return f"Slice {slice_number} - {phrases[(slice_number - 1) % len(phrases)]}"

    if spread == "focused":
        phrases = [
            f"A focused bright area appears {zone_text}.",
            f"The brightest color is concentrated {zone_text}.",
            f"A small focused highlight is visible {zone_text}.",
            f"The main bright spot is focused {zone_text}.",
            f"A concentrated colored area appears {zone_text}.",
            f"The visible highlight is located {zone_text}.",
            f"A compact bright area appears {zone_text}.",
            f"This slice has a focused colored spot {zone_text}.",
            f"The strongest color is tightly grouped {zone_text}.",
            f"A localized highlight is visible {zone_text}.",
            f"The color focus appears {zone_text}.",
            f"A small bright focus is visible {zone_text}.",
        ]
        return f"Slice {slice_number} - {phrases[(slice_number - 1) % len(phrases)]}"

    if intensity == "high":
        phrases = [
            f"A brighter area is visible {zone_text}.",
            f"The strongest color appears {zone_text}.",
            f"A clear bright area appears {zone_text}.",
            f"The main bright color is visible {zone_text}.",
            f"A stronger colored area appears {zone_text}.",
            f"The brightest highlight is seen {zone_text}.",
            f"A bright color pattern is most visible {zone_text}.",
            f"The clearest colored area appears {zone_text}.",
            f"A strong visual highlight is located {zone_text}.",
            f"The brightest part of this slice appears {zone_text}.",
            f"A clear area of stronger color is visible {zone_text}.",
            f"The main high-intensity color appears {zone_text}.",
        ]
        return f"Slice {slice_number} - {phrases[(slice_number - 1) % len(phrases)]}"

    phrases = [
        f"A wider colored area appears {zone_text}.",
        f"The main highlighted area appears {zone_text}.",
        f"The colored area is mostly visible {zone_text}.",
        f"A moderate colored area is visible {zone_text}.",
        f"The strongest visible color is located {zone_text}.",
        f"The main color pattern appears {zone_text}.",
        f"A visible colored area is located {zone_text}.",
        f"The brightest part of this slice is located {zone_text}.",
        f"The color pattern is mainly visible {zone_text}.",
        f"A clear but limited colored area appears {zone_text}.",
        f"The highlighted area is mostly concentrated {zone_text}.",
        f"A medium-sized colored area is visible {zone_text}.",
    ]
    return f"Slice {slice_number} - {phrases[(slice_number - 1) % len(phrases)]}"


def build_patient_slice_caption(
    slice_number: int,
    orientation: str,
    highlight_level: str,
    highlight_area_percent: float,
    dominant_zone: str,
) -> str:
    metrics = {
        "intensity_level": highlight_level,
        "highlight_area_percent": highlight_area_percent,
        "brightest_zone": "center" if dominant_zone == "diffuse" else dominant_zone,
        "spread_type": "diffuse" if dominant_zone == "diffuse" else "medium",
    }
    return build_specific_slice_caption(slice_number, orientation, metrics)


def describe_cam_slice(
    cam_volume: np.ndarray,
    orientation: str,
    slice_index: int,
    slice_number: int,
) -> dict:
    cam_slice = np.asarray(oriented_slice(cam_volume, orientation, slice_index), dtype=np.float32)
    cam_slice = np.clip(cam_slice, 0.0, 1.0)
    metrics = describe_heatmap_position(cam_slice)
    caption = build_specific_slice_caption(slice_number, orientation, metrics)

    return {
        "slice_number": slice_number,
        "slice_index": int(slice_index),
        "orientation": orientation,
        "intensity_level": metrics["intensity_level"],
        "highlight_area_percent": metrics["highlight_area_percent"],
        "brightest_zone": metrics["brightest_zone"],
        "spread_type": metrics["spread_type"],
        "highlight_level": metrics["intensity_level"],
        "dominant_zone": metrics["brightest_zone"],
        "caption": caption,
    }


def select_centered_slice_indices(depth: int, num_slices: int) -> list[int]:
    """
    Select slice indices distributed around the center of the MRI volume.
    """
    if depth <= 0:
        raise ValueError("MRI volume depth must be greater than zero.")

    if num_slices <= 0:
        raise ValueError("num_slices must be greater than zero.")

    center = depth // 2
    span = max(num_slices - 1, int(depth * 0.60))
    span = min(span, depth - 1)

    start = max(0, center - span // 2)
    end = min(depth - 1, start + span)
    start = max(0, end - span)

    return [int(round(index)) for index in np.linspace(start, end, num_slices)]


def generate_gradcam_slices(
    mri_path: str | Path,
    orientation: str = "multi",
    num_slices: int = 5,
    target_class: int | None = None,
    return_indices: bool = False,
    return_metadata: bool = True,
    output_dir: str | Path | None = None,
    display_mode: str = "overlay",
    alpha: float = 0.60,
    threshold: float = 0.48,
    percentile: float = 88.0,
    colormap: str = "turbo",
) -> list[dict] | list[Path] | tuple[list[dict] | list[Path], list[int]]:
    """
    Generate visual explanation images for a .nii or .nii.gz file.

    The default mode creates one patient-friendly multi-view image containing
    axial, sagittal and coronal slices. Single-orientation generation is kept
    for backward compatibility.
    """
    mri_path = Path(mri_path)
    if not mri_path.exists():
        raise FileNotFoundError(f"MRI file not found: {mri_path}")

    orientation = normalize_generation_orientation(orientation)
    display_mode = normalize_display_mode(display_mode)
    output_root = Path(output_dir) if output_dir else OUTPUT_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    mri_volume, cam_volume, class_index = compute_gradcam(
        mri_path=mri_path,
        target_class=target_class,
    )
    target_label = LABELS.get(class_index, str(class_index))

    if orientation == "multi":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mri_name = mri_path.name.replace(".nii.gz", "").replace(".nii", "")
        output_path = output_root / (
            f"visual_explanation_{display_mode}_{timestamp}_{_safe_name(mri_name)}.png"
        )
        saved_path = save_multi_axis_visualization(
            mri_volume=mri_volume,
            cam_volume=cam_volume,
            output_path=output_path,
            num_slices=num_slices,
            display_mode=display_mode,
            alpha=alpha,
            threshold=threshold,
            percentile=percentile,
            colormap=colormap,
        )
        metadata = {
            "image_path": str(saved_path),
            "caption": (
                "Axial, sagittal and coronal views - overlay visual explanation."
                if display_mode == "overlay"
                else "Axial, sagittal and coronal views - heatmap-only visual explanation."
            ),
            "orientation": "multi-axis",
            "slice_number": 1,
            "view_count": 3,
            "slices_per_view": int(num_slices),
            "displayed_slices": int(num_slices) * 3,
            "display_mode": display_mode,
            "threshold": float(threshold),
            "percentile": float(percentile),
            "alpha": float(alpha),
        }
        result = [metadata] if return_metadata else [saved_path]
        if return_indices:
            return result, multi_axis_slice_indices(mri_volume.shape, num_slices)
        return result

    axis = VALID_ORIENTATIONS[orientation]
    slice_indices = select_centered_slice_indices(
        depth=mri_volume.shape[axis],
        num_slices=num_slices,
    )
    slice_paths = []
    slice_metadata = []

    for display_index, z_index in enumerate(slice_indices, start=1):
        output_path = output_root / (
            f"visual_explanation_{display_mode}_{orientation}_slice_{display_index:02d}.png"
        )
        saved_path = save_gradcam_overlay(
            mri_volume=mri_volume,
            cam_volume=cam_volume,
            output_path=output_path,
            target_label=target_label,
            slice_index=z_index,
            orientation=orientation,
            display_mode=display_mode,
            alpha=alpha,
            threshold=threshold,
            percentile=percentile,
            colormap=colormap,
        )
        slice_paths.append(saved_path)
        metadata = describe_cam_slice(
            cam_volume=cam_volume,
            orientation=orientation,
            slice_index=z_index,
            slice_number=display_index,
        )
        metadata["image_path"] = str(saved_path)
        metadata["display_mode"] = display_mode
        metadata["threshold"] = float(threshold)
        metadata["percentile"] = float(percentile)
        metadata["alpha"] = float(alpha)
        slice_metadata.append(metadata)

    if return_indices:
        return (slice_metadata if return_metadata else slice_paths), slice_indices

    return slice_metadata if return_metadata else slice_paths


def generate_gradcam_3d(
    mri_path: str | Path,
    target_class: int | None = None,
) -> Path:
    """
    Generate a Grad-CAM PNG for a .nii or .nii.gz MRI file.

    The output is saved under:
        outputs/gradcam/
    """
    mri_volume, cam_volume, class_index = compute_gradcam(
        mri_path=mri_path,
        target_class=target_class,
    )
    target_label = LABELS.get(class_index, str(class_index))

    mri_name = Path(mri_path).name.replace(".nii.gz", "").replace(".nii", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"gradcam_{timestamp}_{mri_name}_{target_label}.png"

    return save_gradcam_overlay(
        mri_volume=mri_volume,
        cam_volume=cam_volume,
        output_path=output_path,
        target_label=target_label,
        orientation="axial",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate 3D Grad-CAM for one MRI.")
    parser.add_argument("mri_path", help="Path to a .nii or .nii.gz MRI file.")
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        choices=[0, 1],
        help="Optional target class: 0=CN, 1=AD. Defaults to predicted class.",
    )
    args = parser.parse_args()

    generated_path = generate_gradcam_3d(
        mri_path=args.mri_path,
        target_class=args.target_class,
    )
    print(generated_path)
