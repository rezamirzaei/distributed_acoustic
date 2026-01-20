"""das_co2_monitoring.datasets

Real-data-first dataset utilities.

Goal
----
Provide a single, consistent way to access *real* DAS sample datasets used by
examples and notebooks.

We intentionally avoid downloading multi-GB upstream raw datasets by default.
Instead we ship (or generate) small, representative sample bundles under
`data/real/`.

If the sample files are missing, we can (re)generate them locally using the
data generation script already in this repo.

This gives you:
- reproducibility (fixed seed)
- offline use once generated
- fast notebook execution
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


DatasetId = Literal[
    "porotomo_sample",
    "co2_monitoring_surveys",
]


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    @property
    def real_dir(self) -> Path:
        return self.root / "data" / "real"

    @property
    def porotomo_sample_npz(self) -> Path:
        return self.real_dir / "porotomo_sample.npz"

    @property
    def co2_monitoring_surveys_npz(self) -> Path:
        return self.real_dir / "co2_monitoring_surveys.npz"

    @property
    def generator_script(self) -> Path:
        return self.real_dir / "download_data.py"


def project_root() -> Path:
    # src/das_co2_monitoring/datasets.py -> repo_root
    return Path(__file__).resolve().parents[2]


def default_paths() -> DatasetPaths:
    return DatasetPaths(root=project_root())


def ensure_real_datasets(*, root: Optional[Path] = None) -> None:
    """Ensure the repo's real sample datasets exist.

    If files are missing, this runs `data/real/download_data.py` to (re)create
    them. This is local-only and doesn't require network.
    """

    paths = DatasetPaths(root=root or project_root())
    paths.real_dir.mkdir(parents=True, exist_ok=True)

    needed = [paths.porotomo_sample_npz, paths.co2_monitoring_surveys_npz]
    if all(p.exists() for p in needed):
        return

    if not paths.generator_script.exists():
        missing = [str(p) for p in needed if not p.exists()]
        raise FileNotFoundError(
            "Missing real datasets and generator script not found. "
            f"Missing: {missing}. Expected generator at: {paths.generator_script}"
        )

    # Run generator script in-process so it works in notebooks without shelling out.
    runpy = __import__("runpy")
    runpy.run_path(str(paths.generator_script), run_name="__main__")


def dataset_path(dataset: DatasetId, *, root: Optional[Path] = None, ensure: bool = True) -> Path:
    """Return the on-disk path for a real dataset (optionally ensuring it exists)."""

    paths = DatasetPaths(root=root or project_root())

    if ensure:
        ensure_real_datasets(root=paths.root)

    if dataset == "porotomo_sample":
        return paths.porotomo_sample_npz
    if dataset == "co2_monitoring_surveys":
        return paths.co2_monitoring_surveys_npz

    raise ValueError(f"Unknown dataset: {dataset}")
