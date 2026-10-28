"""Proteome report formats: real-export headers, DIA-NN pg_matrix, detection, contaminants.

Fixtures are tiny synthetic files whose headers copy the layouts real
Spectronaut and DIA-NN exports write (dotted names, "[N] run.raw.PG.Quantity",
CRLF line endings, run columns headed by full Windows paths).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import alphaphos as ap

RUNS = ["20261023_OA5_DO018_proteome_A1", "20261023_OA5_DO018_proteome_A2"]
# Rows: one real protein, one Spectronaut-tagged trypsin, one missing value.
PROTEINS = ["P12345", "Cont_P00761", "Q99999"]
GENES = ["Gene1", "", "Gene3"]
INTENSITY = [[300.0, 100.0], [100.0, 300.0], [np.nan, 50.0]]


def _write(path: Path, header: list[str], rows: list[list]) -> Path:
    lines = ["\t".join(header)] + ["\t".join("" if pd.isna(v) else str(v) for v in r) for r in rows]
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode())
    return path


@pytest.fixture
def spectronaut_short(tmp_path) -> Path:
    header = ["PG.ProteinGroups", "PG.Genes"] + [
        f"[{i}] {run}.raw.PG.Quantity" for i, run in enumerate(RUNS, 1)
    ]
    rows = [[p, g, *q] for p, g, q in zip(PROTEINS, GENES, INTENSITY, strict=True)]
    return _write(tmp_path / "sn_short.tsv", header, rows)


@pytest.fixture
def diann_pg_matrix(tmp_path) -> Path:
    meta = ["Protein.Group", "Protein.Names", "Genes", "First.Protein.Description", "N.Sequences"]
    header = meta + [f"V:\\raw\\proteome\\{run}.raw" for run in RUNS]
    rows = [
        [p, f"{g}_HUMAN", g, "desc", 3, *q]
        for p, g, q in zip(PROTEINS, GENES, INTENSITY, strict=True)
    ]
    return _write(tmp_path / "report.pg_matrix.tsv", header, rows)


@pytest.mark.parametrize("reader", ["spectronaut", "diann"])
def test_short_readers_give_the_same_layout(reader, spectronaut_short, diann_pg_matrix):
    adata = (
        ap.proteome.read_spectronaut_short(spectronaut_short)
        if reader == "spectronaut"
        else ap.proteome.read_diann_pg_matrix(diann_pg_matrix)
    )
    assert list(adata.obs_names) == RUNS
    assert list(adata.var_names) == PROTEINS
    assert {"PG_ProteinGroups", "PG_Genes"} <= set(adata.var.columns)
    np.testing.assert_allclose(adata.X, np.log2(np.array(INTENSITY).T))
    np.testing.assert_array_equal(adata.layers["intensity_log2"], adata.X)


def test_underscored_spectronaut_export_still_reads(tmp_path, spectronaut_short):
    df = pd.read_csv(spectronaut_short, sep="\t")
    df.columns = ["PG_ProteinGroups", "PG_Genes"] + [
        f"[{i}]_{run}_raw_PG_Quantity" for i, run in enumerate(RUNS, 1)
    ]
    df.to_csv(tmp_path / "underscored.tsv", sep="\t", index=False)
    a = ap.proteome.read_spectronaut_short(tmp_path / "underscored.tsv")
    b = ap.proteome.read_spectronaut_short(spectronaut_short)
    np.testing.assert_array_equal(a.X, b.X)
    assert list(a.obs_names) == list(b.obs_names) and list(a.var.columns) == list(b.var.columns)


def test_detect_report_format(tmp_path, spectronaut_short, diann_pg_matrix):
    sn_long = _write(
        tmp_path / "sn_long.tsv", ["R.FileName", "PG.ProteinGroups", "EG.PrecursorId"], []
    )
    diann_long = _write(tmp_path / "report.tsv", ["Run", "Protein.Group", "Precursor.Id"], [])
    other = _write(tmp_path / "other.tsv", ["sample", "condition"], [])
    detected = {
        p.name: ap.proteome.detect_report_format(p)
        for p in (spectronaut_short, diann_pg_matrix, sn_long, diann_long)
    }
    assert [(f.engine, f.level) for f in detected.values()] == [
        ("Spectronaut", "short"),
        ("DIA-NN", "short"),
        ("Spectronaut", "long"),
        ("DIA-NN", "long"),
    ]
    with pytest.raises(ValueError, match="does not look like"):
        ap.proteome.detect_report_format(other)


def test_protein_contaminants_measured_then_removed(spectronaut_short):
    adata = ap.proteome.read_spectronaut_short(spectronaut_short)
    frac = ap.proteome.contaminant_fraction(adata)
    # Cont_P00761 is 100 of 400 in run A1 and 300 of 450 in run A2 (NaN counts as 0).
    assert frac["contaminant_fraction"].to_numpy() == pytest.approx([100 / 400, 300 / 450])
    assert list(frac["n_contaminant_proteins"]) == [1, 1]
    assert list(frac["n_total_proteins"]) == [2, 3]
    fig = ap.viz.contaminant_fraction(frac)  # the precursor-level figure takes it unchanged
    assert list(fig.data[0].y) == pytest.approx([25.0, 300 / 450 * 100])

    clean = ap.proteome.drop_contaminant_proteins(adata)
    assert list(clean.var_names) == ["P12345", "Q99999"]
    assert clean.uns["alphaphos_proteome"]["stats"]["n_contaminant_proteins_dropped"] == 1


def test_run_names_strip_paths_and_extensions():
    name = ap.proteome.io_diann_short.run_name_from_path
    assert name(r"V:\data\run_A1.raw") == "run_A1"
    assert name("/data/run_A1.d/") == "run_A1"
    assert name("run_A1.mzML") == "run_A1"
