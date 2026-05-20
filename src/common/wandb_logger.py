from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WandbLogger:
    def __init__(self, cfg: dict, run_dir: Path, resume_checkpoint: str | None = None) -> None:
        self._enabled = bool(cfg.get("wandb_enabled", True))
        self._run = None
        self._wandb = None

        mode = str(cfg.get("wandb_mode", "online")).strip().lower()
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError(f"wandb_mode must be one of ['online', 'offline', 'disabled'], got {mode!r}")
        if mode == "disabled":
            self._enabled = False

        if not self._enabled:
            logger.info("W&B logging disabled.")
            return

        try:
            import wandb  # type: ignore
        except ImportError:
            strict = bool(cfg.get("wandb_strict", False))
            msg = "wandb is not installed but wandb_enabled=true. Install `wandb` or disable W&B."
            if strict:
                raise RuntimeError(msg)
            logger.warning("%s Falling back to disabled logging.", msg)
            self._enabled = False
            return

        self._wandb = wandb
        tags = cfg.get("wandb_tags", [])
        if not isinstance(tags, list):
            raise ValueError("wandb_tags must be a list of strings.")

        run_name = str(cfg.get("wandb_run_name", "")).strip() or str(run_dir.name)
        project = str(cfg.get("wandb_project", "")).strip()
        if not project:
            raise ValueError("wandb_project must be set when W&B logging is enabled.")

        entity = str(cfg.get("wandb_entity", "")).strip() or None
        group = str(cfg.get("wandb_group", "")).strip() or None
        strict = bool(cfg.get("wandb_strict", False))
        init_cfg: dict[str, Any] | None = cfg if bool(cfg.get("wandb_log_config", True)) else None

        try:
            self._run = wandb.init(
                project=project,
                entity=entity,
                group=group,
                name=run_name,
                tags=[str(t) for t in tags],
                config=init_cfg,
                mode=mode,
                dir=str(run_dir),
                resume="allow" if resume_checkpoint else None,
            )
            if self._run is not None:
                self._run.summary["run_dir"] = str(run_dir)
                if resume_checkpoint:
                    self._run.summary["resume_checkpoint"] = str(resume_checkpoint)
        except Exception as exc:
            if strict:
                raise
            logger.warning("W&B init failed (%s). Falling back to disabled logging.", exc)
            self._enabled = False
            self._run = None

    @property
    def enabled(self) -> bool:
        return self._enabled and self._run is not None and self._wandb is not None

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        if not self.enabled:
            return
        self._wandb.log(metrics, step=int(step))

    def log_images(self, key: str, images: list[Any], step: int) -> None:
        if not self.enabled or not images:
            return
        self._wandb.log({str(key): images}, step=int(step))

    def log_dict(self, values: dict[str, Any], step: int) -> None:
        if not self.enabled or not values:
            return
        self._wandb.log(values, step=int(step))

    def make_image(self, image: Any, caption: str | None = None) -> Any:
        if not self.enabled:
            return None
        return self._wandb.Image(image, caption=caption)

    def make_table(self, rows: list[dict[str, Any]]) -> Any:
        if not self.enabled or not rows:
            return None
        return self._wandb.Table(data=rows)

    def set_summary(self, values: dict[str, Any]) -> None:
        if not self.enabled:
            return
        for k, v in values.items():
            self._run.summary[k] = v

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
