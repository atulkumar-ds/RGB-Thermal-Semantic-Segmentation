"""Pretrained dual-stream ResNet-18 U-Net with cross-modal attention for MFNet.

Input:  [B, 4, H, W], RGB in channels 0:3 and thermal in channel 3, both [0, 1].
Output: [B, 9, H, W] unnormalized segmentation logits.

The ResNet-18 encoders are initialized from ImageNet weights by default. The
thermal first convolution is converted from 3 input channels to 1 by averaging
the pretrained RGB kernels. The thermal stream is separately normalized, so averaging
provides a conservative initialization without tripling the RGB stem response.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 9
IN_CHANNELS = 4
ARCHITECTURE_ID = "resnet18_dual_stream_cross_modal_attention_v2_1"


def _groups(channels: int) -> int:
    for g in (16, 8, 4, 2):
        if channels % g == 0:
            return g
    return 1


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNet18Encoder(nn.Module):
    """ResNet-18 feature pyramid; input is RGB (3ch) or thermal (1ch)."""

    def __init__(self, in_channels: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        try:
            net = resnet18(weights=weights)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    "Could not load ImageNet pretrained ResNet-18 weights. "
                    "Check internet access/cache, or explicitly use --no-pretrained."
                ) from exc
            raise

        if in_channels == 1:
            old = net.conv1
            thermal_conv = nn.Conv2d(
                1,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            with torch.no_grad():
                thermal_conv.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            net.conv1 = thermal_conv
        elif in_channels != 3:
            raise ValueError(f"Encoder expects 1 or 3 input channels, got {in_channels}")

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        stem = self.relu(self.bn1(self.conv1(x)))  # 1/2 resolution
        x = self.maxpool(stem)
        f1 = self.layer1(x)                       # 1/4, 64 channels
        f2 = self.layer2(f1)                      # 1/8, 128 channels
        f3 = self.layer3(f2)                      # 1/16, 256 channels
        f4 = self.layer4(f3)                      # 1/32, 512 channels
        return stem, f1, f2, f3, f4


class CrossModalAttentionFusion(nn.Module):
    """Lightweight channel + spatial attention fusion at one feature scale."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        # Average/max maps from both modalities jointly produce two spatial gates.
        self.spatial_gate = nn.Conv2d(4, 2, kernel_size=7, padding=3, bias=True)
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        gates = self.channel_gate(torch.cat((rgb, thermal), dim=1))
        rgb_channel, thermal_channel = gates.chunk(2, dim=1)

        spatial_inputs = torch.cat(
            (
                rgb.mean(dim=1, keepdim=True),
                rgb.amax(dim=1, keepdim=True),
                thermal.mean(dim=1, keepdim=True),
                thermal.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        rgb_spatial, thermal_spatial = torch.sigmoid(self.spatial_gate(spatial_inputs)).chunk(2, dim=1)

        rgb_weighted = rgb * rgb_channel * rgb_spatial
        thermal_weighted = thermal * thermal_channel * thermal_spatial
        fused = self.mix(torch.cat((rgb_weighted, thermal_weighted), dim=1))

        # Residual path prevents early attention gates from discarding both streams.
        return fused + 0.5 * (rgb + thermal)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.project = ConvNormAct(in_channels, out_channels, kernel_size=1, padding=0)
        self.refine = nn.Sequential(
            ConvNormAct(out_channels + skip_channels, out_channels),
            ConvNormAct(out_channels, out_channels),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.project(x)
        return self.refine(torch.cat((x, skip), dim=1))


class CustomDualStreamUNet(nn.Module):
    """ImageNet-initialized RGB/thermal encoders with multi-scale fusion and U-Net decoder."""

    architecture_id = ARCHITECTURE_ID

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = IN_CHANNELS,
        pretrained: bool = True,
        decoder_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if in_channels != 4:
            raise ValueError("MFNet input must have 4 channels: RGB + thermal.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        self.rgb_encoder = ResNet18Encoder(in_channels=3, pretrained=pretrained)
        self.thermal_encoder = ResNet18Encoder(in_channels=1, pretrained=pretrained)

        channels = (64, 64, 128, 256, 512)
        self.fusion = nn.ModuleList(
            [CrossModalAttentionFusion(c) for c in channels]
        )

        self.bottleneck_dropout = nn.Dropout2d(p=0.15)
        self.decoder4 = DecoderBlock(512, 256, 256, decoder_dropout)
        self.decoder3 = DecoderBlock(256, 128, 128, decoder_dropout)
        self.decoder2 = DecoderBlock(128, 64, 64, decoder_dropout)
        self.decoder1 = DecoderBlock(64, 64, 64, decoder_dropout)
        self.segmentation_head = nn.Sequential(
            ConvNormAct(64, 64, kernel_size=3),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

        # RGB ImageNet normalization. Thermal is normalized independently to [-2, 2].
        self.register_buffer(
            "rgb_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "thermal_mean", torch.tensor((0.5,), dtype=torch.float32).view(1, 1, 1, 1)
        )
        self.register_buffer(
            "thermal_std", torch.tensor((0.25,), dtype=torch.float32).view(1, 1, 1, 1)
        )
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pretrained = pretrained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != IN_CHANNELS:
            raise ValueError(
                f"Expected input [B, 4, H, W] (RGB + thermal), got {tuple(x.shape)}"
            )
        input_size = x.shape[-2:]
        rgb = (x[:, :3] - self.rgb_mean) / self.rgb_std
        thermal = (x[:, 3:4] - self.thermal_mean) / self.thermal_std

        rgb_features = self.rgb_encoder(rgb)
        thermal_features = self.thermal_encoder(thermal)
        fused = [
            fuse(rgb_feature, thermal_feature)
            for fuse, rgb_feature, thermal_feature in zip(
                self.fusion, rgb_features, thermal_features
            )
        ]

        y = self.bottleneck_dropout(fused[4])
        y = self.decoder4(y, fused[3])
        y = self.decoder3(y, fused[2])
        y = self.decoder2(y, fused[1])
        y = self.decoder1(y, fused[0])
        logits = self.segmentation_head(y)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


def build_model(
    num_classes: int = NUM_CLASSES,
    in_channels: int = IN_CHANNELS,
    pretrained: bool = True,
) -> CustomDualStreamUNet:
    return CustomDualStreamUNet(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
    )
