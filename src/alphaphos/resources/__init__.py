"""Bundled data resources for alphaPhos.

Two tiers of data, resolved through two helpers:

**Packaged** -- ship inside the wheel, always available via :func:`packaged`:

- ``contaminants.fasta`` -- MaxQuant contaminant list (246 sequences), used
  by :func:`alphaphos.io.contaminants.filter_contaminants`.
- ``goldstandard/`` -- kinase-activity benchmark tables (Ochoa et al. 2016;
  Hernández-Armenta et al. 2017), used by :mod:`alphaphos.enrichment.validation`.
- ``libraries/`` -- PTM-DB site-set GMTs + ``manifest.csv`` for
  :func:`alphaphos.enrichment.load_libraries`.

**External** -- too large or not yet redistributable, so they live *outside*
the package and are located via :func:`external`:

- ``fastas/`` -- UniProt proteomes (~84 MB) for
  :func:`alphaphos.add_kinase_windows` / :func:`alphaphos.orthology.map_to_human`.
- ``ptm_functional_db.parquet`` -- built with ``scripts/build_ptm_db.py``.

:func:`external_dir` resolves, in order: ``$ALPHAPHOS_RESOURCES`` →
``<repo>/resources`` when running from a git checkout (editable install) →
``~/.alphaphos/resources``.  The returned path may not exist; callers raise
``FileNotFoundError`` with a pointer here.
"""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

ENV_VAR = "ALPHAPHOS_RESOURCES"


def packaged(*parts: str) -> Path:
    """Return the filesystem path of a file shipped inside ``alphaphos.resources``."""
    node = files("alphaphos.resources")
    for part in parts:
        node = node / part
    # importlib.resources returns a Traversable; converting to Path works
    # when the package is installed as a real directory (the normal case).
    return Path(str(node))


def external_dir() -> Path:
    """Directory holding the large, non-packaged resources (see module docstring)."""
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser()
    # src/alphaphos/resources/__init__.py -> parents[3] is the repo root.
    repo_candidate = Path(__file__).resolve().parents[3] / "resources"
    if (repo_candidate / "fastas").is_dir():
        return repo_candidate
    return Path.home() / ".alphaphos" / "resources"


def external(*parts: str) -> Path:
    """Path under :func:`external_dir` (may not exist)."""
    return external_dir().joinpath(*parts)


CONTAMINANTS_FASTA: Path = packaged("contaminants.fasta")
GOLDSTANDARD_DIR: Path = packaged("goldstandard")
LIBRARIES_DIR: Path = packaged("libraries")

__all__ = [
    "ENV_VAR",
    "packaged",
    "external",
    "external_dir",
    "CONTAMINANTS_FASTA",
    "GOLDSTANDARD_DIR",
    "LIBRARIES_DIR",
]
