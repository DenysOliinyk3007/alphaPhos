"""Loader for the PTM functional database parquet.

The parquet is built once from the integrated Excel workbook via
``scripts/build_ptm_db.py``; it lives at
``<resources>/ptm_functional_db.parquet`` (gitignored -- future Zenodo
release; ``<resources>`` resolved by
:func:`alphaphos.resources.external_dir`) or at a caller-provided path.  A
small test fixture is committed at ``tests/data/ptm_db_mini.parquet`` for CI.

Schema (22 columns, phospho-only for v1):

* Identity: ``substrate_gene``, ``substrate_uniprot``, ``residue``,
  ``position``, ``modification``
* Enzyme edge: ``enzyme_gene``, ``enzyme_uniprot``
* Effect annotations:  ``effect`` (raw),
  ``effect_canonical`` (categorical: activates_activity /
  inhibits_activity / induces_ppi / disrupts_ppi / alters_stability /
  alters_localization), ``PSP_reg_functional_effect``,
  ``interaction_partner``
* Provenance:  ``source_dbs``, ``n_sources``, ``curation_tier``,
  ``curation_confidence`` (high/medium/low), ``is_curated``,
  ``n_pmids``, ``chemistry_warning``, ``chemistry_warning_flag``
* Functional annotations: ``Ochoa_functional_score`` (0-1),
  ``ADB_disease_variant`` (ClinVar/MC3_TCGA labels),
  ``ELM_motifs_protein``
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from alphaphos import resources as _resources

if TYPE_CHECKING:
    import pandas as pd


# Column names surfaced as constants so downstream code doesn't hardcode strings.
COL_GENE = "substrate_gene"
COL_UNIPROT = "substrate_uniprot"
COL_RESIDUE = "residue"
COL_POSITION = "position"
COL_MODIFICATION = "modification"
COL_ENZYME_GENE = "enzyme_gene"
COL_ENZYME_UNIPROT = "enzyme_uniprot"
COL_EFFECT_RAW = "effect"
COL_EFFECT_CANONICAL = "effect_canonical"
COL_DISEASE_VARIANT = "ADB_disease_variant"
COL_OCHOA_SCORE = "Ochoa_functional_score"
COL_CURATION_CONFIDENCE = "curation_confidence"
COL_IS_CURATED = "is_curated"
COL_CHEMISTRY_WARNING = "chemistry_warning_flag"

# The DB parquet is built with a fixed column set.
REQUIRED_COLUMNS: tuple[str, ...] = (
    COL_GENE,
    COL_UNIPROT,
    COL_RESIDUE,
    COL_POSITION,
    COL_MODIFICATION,
    COL_ENZYME_GENE,
    COL_ENZYME_UNIPROT,
    COL_EFFECT_RAW,
    COL_EFFECT_CANONICAL,
    COL_DISEASE_VARIANT,
    COL_OCHOA_SCORE,
    COL_CURATION_CONFIDENCE,
    COL_IS_CURATED,
    COL_CHEMISTRY_WARNING,
)


# Default location: the external (non-packaged) resources dir -- see
# alphaphos.resources.  We do NOT auto-fall-back to the test fixture for
# real users -- surprise-substituting a 1000-row test DB for a 500k-row
# real DB would silently break their analysis.
DEFAULT_DB_PATH = _resources.external("ptm_functional_db.parquet")
# Only meaningful from a git checkout (tests/ is not packaged).
TEST_FIXTURE_PATH = Path(__file__).resolve().parents[3] / "tests" / "data" / "ptm_db_mini.parquet"


def load_ptm_db(path: str | Path | None = None) -> pd.DataFrame:
    """Load the PTM functional-database parquet.

    Parameters
    ----------
    path
        Path to the parquet.  If ``None``, defaults to
        :data:`DEFAULT_DB_PATH`.  Tests should pass
        ``TEST_FIXTURE_PATH`` explicitly.

    Returns
    -------
    pandas.DataFrame
        The full DB with all columns in :data:`REQUIRED_COLUMNS` plus
        any auxiliary columns.

    Raises
    ------
    FileNotFoundError
        If the target parquet doesn't exist.  Points the user at
        ``scripts/build_ptm_db.py`` to build it from the source xlsx.
    ValueError
        If required columns are missing (parquet built by an older
        version of ``build_ptm_db.py``).
    """
    import pandas as pd

    target = Path(path) if path is not None else DEFAULT_DB_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"PTM functional DB not found at {target}. Build it with:\n"
            "    python scripts/build_ptm_db.py \\\n"
            "        --xlsx <PTM_functional_databases_data_final.xlsx> \\\n"
            f"        --out {target}\n"
            f"Or pass an explicit path to load_ptm_db(). For tests, use "
            f"the small fixture at {TEST_FIXTURE_PATH}."
        )
    df = pd.read_parquet(target, engine="pyarrow")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"PTM DB at {target} is missing required columns: {missing}. "
            "Rebuild via scripts/build_ptm_db.py against a matching xlsx."
        )
    return df


def site_id(uniprot: str, residue: str, position: int) -> str:
    """Canonical site identifier used across enrichment libraries.

    Format: ``<uniprot>_<residue><position>`` (e.g. ``P00533_Y1172``).
    Matches the OmniPath / decoupler convention so libraries emitted
    here interoperate with existing enrichment tooling.  Multi-mapped
    UniProts (semicolon-joined in the DB) are collapsed to the first
    accession -- the same convention alphaPhos uses for its
    ``Protein|Gene|Site|Mult`` var-name keys.
    """
    up = str(uniprot).split(";", 1)[0]
    return f"{up}_{residue}{int(position)}"
