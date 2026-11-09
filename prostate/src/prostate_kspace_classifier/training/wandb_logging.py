"""Optional Weights & Biases run management."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


class WandbLogger:
    """Thin wrapper that guards against missing wandb installs."""

    def __init__(
        self,
        project: Optional[str] = None,
        *,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        mode: str = "online",
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> None:
        self.enabled = False
        self.run = None
        self._wandb = None
        if not project:
            return

        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError("wandb is not installed. Add it to your environment to enable logging.") from exc

        init_args: Dict[str, Any] = {
            "project": project,
            "entity": entity,
            "name": run_name,
            "group": group,
            "job_type": job_type,
            "config": config or {},
            "mode": mode,
        }
        if tags is not None:
            init_args["tags"] = list(tags)
        self.run = wandb.init(**{k: v for k, v in init_args.items() if v is not None})
        self.enabled = True
        self._wandb = wandb

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled and self.run is not None:
            self.run.log(metrics, step=step)

    def set_summary(self, metrics: Dict[str, Any]) -> None:
        if self.enabled and self.run is not None:
            for key, value in metrics.items():
                self.run.summary[key] = value

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()
