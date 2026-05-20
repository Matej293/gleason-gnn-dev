from __future__ import annotations

import re
from pathlib import Path

_EXPERIMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def require_existing_file(path: str | Path, *, label: str) -> Path:
    out = Path(path).expanduser()
    if not out.exists() or not out.is_file():
        raise FileNotFoundError(f"{label} not found: {out}")
    return out.resolve()


def require_existing_dir(path: str | Path, *, label: str) -> Path:
    out = Path(path).expanduser()
    if not out.exists() or not out.is_dir():
        raise FileNotFoundError(f"{label} not found: {out}")
    return out.resolve()


def require_non_empty_str(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def validate_experiment_name(value: object) -> str:
    name = require_non_empty_str(value, field_name="experiment_name")
    if "/" in name or "\\" in name:
        raise ValueError(
            "experiment_name must not contain path separators ('/' or '\\')."
        )
    if name in {".", ".."}:
        raise ValueError("experiment_name cannot be '.' or '..'.")
    if not _EXPERIMENT_RE.fullmatch(name):
        raise ValueError(
            "experiment_name must match [A-Za-z0-9][A-Za-z0-9._-]* for stable folder naming."
        )
    return name


def validate_non_negative_int(value: object, *, field_name: str) -> int:
    out = int(value)
    if out < 0:
        raise ValueError(f"{field_name} must be >= 0, got {out}.")
    return out


def validate_positive_int(value: object, *, field_name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{field_name} must be > 0, got {out}.")
    return out


def validate_seed(value: object, *, field_name: str = "seed") -> int:
    seed = validate_non_negative_int(value, field_name=field_name)
    if seed > (2**32 - 1):
        raise ValueError(f"{field_name} must be <= 2^32-1, got {seed}.")
    return seed


def ensure_output_dir(path: str | Path, *, label: str) -> Path:
    out = Path(path).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if not out.exists() or not out.is_dir():
        raise RuntimeError(f"Failed to create {label}: {out}")
    return out.resolve()


def resolve_checkpoint_path(
    run_dir: str | Path,
    checkpoint_arg: str | None = None,
    *,
    prefer_best: bool = True,
) -> Path:
    run_path = Path(run_dir).expanduser()
    ckpt_dir = run_path / "checkpoints"

    if prefer_best and not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory missing: {ckpt_dir}")

    if checkpoint_arg:
        direct = Path(checkpoint_arg).expanduser()
        if direct.exists():
            return direct.resolve()
        candidate = ckpt_dir / checkpoint_arg
        if candidate.exists():
            return candidate.resolve()
        if prefer_best:
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_arg} (checked direct path and {ckpt_dir})"
            )
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory missing: {ckpt_dir}")

    if prefer_best:
        best = ckpt_dir / "best.pt"
        if best.exists():
            return best.resolve()

    epoch_files = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not epoch_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
    return epoch_files[-1].resolve()


def validate_fraction(value: object, *, field_name: str) -> float:
    out = float(value)
    if not 0.0 <= out <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {out}.")
    return out
