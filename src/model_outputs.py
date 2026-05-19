from __future__ import annotations

import torch


def extract_logits(out: object) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        logits = out.get("out")
        if isinstance(logits, torch.Tensor):
            return logits
        raise TypeError("Model output dict must contain tensor under key 'out'.")
    if isinstance(out, (list, tuple)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"Unsupported model output type: {type(out)!r}")


__all__ = ["extract_logits"]
