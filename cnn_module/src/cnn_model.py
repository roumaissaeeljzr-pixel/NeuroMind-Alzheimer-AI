from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_SHAPE = (112, 112, 112)


def crop_to_brain(data: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    """
    Remove black padding around the brain volume.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {data.shape}.")

    mask = np.abs(data) > threshold

    if not np.any(mask):
        return data

    coords = np.array(np.where(mask))
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    z_min, z_max = coords[2].min(), coords[2].max()

    return data[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]


def resize_volume(
    data: np.ndarray,
    target_shape: Sequence[int] = TARGET_SHAPE,
) -> np.ndarray:
    """
    Resize a 3D MRI volume to the target shape.
    """
    tensor_data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

    tensor_data = F.interpolate(
        tensor_data,
        size=tuple(target_shape),
        mode="trilinear",
        align_corners=False,
    )

    return tensor_data.squeeze(0).squeeze(0).numpy()


def normalize_volume(data: np.ndarray) -> np.ndarray:
    """
    Normalize MRI volume using z-score normalization.
    """
    data = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    std = data.std()

    if std > 1e-6:
        data = (data - data.mean()) / std

    return data.astype(np.float32)


def preprocess_mri_array(
    data: np.ndarray,
    target_shape: Sequence[int] = TARGET_SHAPE,
) -> torch.Tensor:
    """
    Preprocess an already-loaded 3D MRI array.
    Output shape: [1, 112, 112, 112].
    """
    data = crop_to_brain(data)
    data = resize_volume(data, target_shape=target_shape)
    data = normalize_volume(data)

    return torch.from_numpy(data).float().unsqueeze(0)


class FocalLoss(nn.Module):
    """
    Focal Loss used for difficult or imbalanced cases.
    """

    def __init__(
        self,
        alpha: float | Iterable[float] = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, (float, int)):
            self.register_buffer("alpha", torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(list(alpha), dtype=torch.float32))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        alpha = self.alpha.to(inputs.device)
        if alpha.ndim == 0:
            focal_loss = alpha * focal_loss
        else:
            focal_loss = alpha[targets] * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        if self.reduction == "none":
            return focal_loss

        raise ValueError(f"Unsupported reduction: {self.reduction}")


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class InflatedResNet3D(nn.Module):
    """
    ResNet3D-18 style architecture for AD vs CN classification.

    The pretrained argument is accepted for compatibility with older training
    scripts, but this implementation does not depend on torchvision weights.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        self.in_channels = 64
        self.pretrained_requested = pretrained

        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, blocks=2)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )

        layers = [
            BasicBlock3D(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
