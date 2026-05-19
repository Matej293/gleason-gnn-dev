from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PSPNet(nn.Module):
    """SMP-backed PSPNet adapted for 2D Gleason segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        use_aux: bool = True,
        pretrained_backbone: bool = True,
        encoder_name: str = "resnet101",
        encoder_weights: str | None = None,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"PSPNet expects in_channels=3, got {in_channels}")

        resolved_weights = (
            encoder_weights
            if encoder_weights is not None
            else ("imagenet" if pretrained_backbone else None)
        )
        if isinstance(resolved_weights, str) and resolved_weights.strip().lower() == "none":
            resolved_weights = None

        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:  # pragma: no cover - depends on optional dependency at runtime.
            raise ImportError(
                "PSPNet requires 'segmentation-models-pytorch'. "
                "Install dependencies from requirements.txt."
            ) from exc

        self.model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=resolved_weights,
            in_channels=in_channels,
            classes=out_channels,
        )

        self.use_aux = bool(use_aux)
        if self.use_aux:
            aux_in_channels = int(self.model.encoder.out_channels[-2])
            self.aux_head = nn.Sequential(
                nn.Conv2d(aux_in_channels, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(256, out_channels, kernel_size=1),
            )
        else:
            self.aux_head = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        in_h, in_w = x.shape[-2:]
        features = self.model.encoder(x)
        try:
            decoder_output = self.model.decoder(*features)
        except TypeError:
            decoder_output = self.model.decoder(features)
        out = self.model.segmentation_head(decoder_output)
        if out.shape[-2:] != (in_h, in_w):
            out = F.interpolate(out, size=(in_h, in_w), mode="bilinear", align_corners=False)

        if not self.use_aux or self.aux_head is None:
            return out

        aux_input = features[-2] if len(features) >= 2 else features[-1]
        aux = self.aux_head(aux_input)
        aux = F.interpolate(aux, size=(in_h, in_w), mode="bilinear", align_corners=False)
        return {"out": out, "aux": aux}
