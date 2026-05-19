from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ProjectConfig


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_root: Path
    raw: Path
    interim: Path
    processed: Path
    features: Path
    registry: Path
    schemas: Path
    checks: Path
    logs: Path


def build_paths(config: ProjectConfig) -> ProjectPaths:
    root = config.root
    paths = config.raw["paths"]
    return ProjectPaths(
        root=root,
        data_root=root / paths["data_root"],
        raw=root / paths["raw"],
        interim=root / paths["interim"],
        processed=root / paths["processed"],
        features=root / paths["features"],
        registry=root / paths["registry"],
        schemas=root / paths["schemas"],
        checks=root / paths["checks"],
        logs=root / paths["logs"],
    )