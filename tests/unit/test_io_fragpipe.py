"""Unit tests for :mod:`alphaphos.io.fragpipe`.

Two levels:

* **Pure helper tests** -- exercise ``parse_fragpipe_index`` and
  ``sequence_window_to_kinase_sequence`` with small string fixtures.
* **Reader tests** -- synthesize a tiny FragPipe abundance file, read it
  back through ``read_fragpipe_sites``, and assert on the AnnData contract
  (shape, var columns, key format, log2 transform, condition_df join).

Also runs an optional integration test against the real reference file at
``W:/User/Denys/dia-quant-output/`` when available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaphos.io.fragpipe import (
    DEFAULT_FRAGPIPE_IO_SETTINGS,
    parse_fragpipe_index,
    read_fragpipe_sites,
    resolve_fragpipe_io_settings,
    sequence_window_to_kinase_sequence,
)

# ---------------------------------------------------------------------------
# parse_fragpipe_index
# ---------------------------------------------------------------------------


class TestParseFragpipeIndex:
    def test_ser(self):
        assert parse_fragpipe_index("P10644_S77") == ("P10644", "S", 77)

    def test_thr(self):
        assert parse_fragpipe_index("Q08378_T140") == ("Q08378", "T", 140)

    def test_tyr(self):
        assert parse_fragpipe_index("Q99999_Y1200") == ("Q99999", "Y", 1200)

    def test_uppercases_aa(self):
        assert parse_fragpipe_index("P10644_s77") == ("P10644", "S", 77)

    def test_returns_none_on_bad_format(self):
        assert parse_fragpipe_index("nonsense") is None
        assert parse_fragpipe_index("") is None
        assert parse_fragpipe_index(None) is None

    def test_handles_underscore_in_protein_id(self):
        # If a protein id contains underscores, split on the LAST one.
        assert parse_fragpipe_index("foo_bar_baz_S12") == ("foo_bar_baz", "S", 12)


# ---------------------------------------------------------------------------
# sequence_window_to_kinase_sequence
# ---------------------------------------------------------------------------


class TestSequenceWindowConversion:
    def test_standard_15mer(self):
        assert sequence_window_to_kinase_sequence("KAGTRTDsREDEIsP", "S") == "_KAGTRTD*S*REDEISP_"

    def test_thr_center(self):
        # 15-mer, center at position 7 (0-indexed)
        s = sequence_window_to_kinase_sequence("ABCDEFGtIJKLMNO", "T")
        assert s == "_ABCDEFG*T*IJKLMNO_"

    def test_short_string_returns_empty(self):
        assert sequence_window_to_kinase_sequence("AB", "S") == ""
        assert sequence_window_to_kinase_sequence("", "S") == ""

    def test_non_string_returns_empty(self):
        assert sequence_window_to_kinase_sequence(None, "S") == ""
        assert sequence_window_to_kinase_sequence(float("nan"), "S") == ""


# ---------------------------------------------------------------------------
# Settings resolution
# ---------------------------------------------------------------------------


class TestResolveSettings:
    def test_defaults(self):
        s = resolve_fragpipe_io_settings(None)
        assert s == DEFAULT_FRAGPIPE_IO_SETTINGS
        assert s is not DEFAULT_FRAGPIPE_IO_SETTINGS

    def test_partial_override(self):
        s = resolve_fragpipe_io_settings({"quant_level": "MS1", "min_best_localization": None})
        assert s["quant_level"] == "MS1"
        assert s["min_best_localization"] is None
        assert s["normalized"] is False

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown keys"):
            resolve_fragpipe_io_settings({"quant_leve": "MS1"})  # typo

    def test_bad_quant_level(self):
        with pytest.raises(ValueError, match="quant_level"):
            resolve_fragpipe_io_settings({"quant_level": "MS3"})

    def test_bad_site_type(self):
        with pytest.raises(ValueError, match="site_type"):
            resolve_fragpipe_io_settings({"site_type": "triple"})

    def test_bad_localization_range(self):
        with pytest.raises(ValueError, match="min_best_localization"):
            resolve_fragpipe_io_settings({"min_best_localization": 1.5})


# ---------------------------------------------------------------------------
# Synthetic FragPipe abundance file
# ---------------------------------------------------------------------------


def _make_synthetic_abundance(tmp_path: Path) -> Path:
    """Write a small FragPipe-shaped abundance file for reader tests."""
    df = pd.DataFrame(
        {
            "Index": ["P10644_S77", "Q08378_S140", "P00000_S99"],
            "Gene": ["PRKAR1A", "GOLGA3", "GENE3"],
            "ProteinID": ["P10644", "Q08378", "P00000"],
            "Peptide": [
                "TDsREDEIsPPPPNPVVK",
                "LSLPMQETQLCSTDsPLPLEK",
                "PET(pS)K",
            ],
            "SequenceWindow": [
                "KAGTRTDsREDEIsP",
                "TQLCSTDsPLPLEKE",
                "ABCDEFGsIJKLMNO",
            ],
            "Multiplicity": [2, 1, 1],
            "Best Localization": [0.90, 0.78, 0.40],  # third row below default 0.75 cutoff
            "Best Scan for Localization": ["scan_a", "scan_b", "scan_c"],
            "Best Precursor for Quant": ["prec_a", "prec_b", "prec_c"],
            "V:/foo/sample_01_uncalibrated.mzML": [1000.0, 2000.0, np.nan],
            "V:/foo/sample_02_uncalibrated.mzML": [1100.0, 2100.0, 0.0],
            "V:/foo/sample_03_uncalibrated.mzML": [1050.0, 2050.0, 500.0],
        }
    )
    p = tmp_path / "abundance_single-site_MS2quant_None.tsv"
    df.to_csv(p, sep="\t", index=False)
    return p


# ---------------------------------------------------------------------------
# Reader tests
# ---------------------------------------------------------------------------


class TestReadFragpipeSites:
    def test_returns_anndata(self, tmp_path):
        import anndata as ad

        p = _make_synthetic_abundance(tmp_path)
        adata = read_fragpipe_sites(p)
        assert isinstance(adata, ad.AnnData)

    def test_default_localization_filter(self, tmp_path):
        p = _make_synthetic_abundance(tmp_path)
        adata = read_fragpipe_sites(p)
        # Third row (Best Localization = 0.40) is dropped at default cutoff 0.75.
        assert adata.n_vars == 2
        assert "GENE3" not in adata.var["gene"].values

    def test_disable_localization_filter(self, tmp_path):
        p = _make_synthetic_abundance(tmp_path)
        adata = read_fragpipe_sites(p, advanced={"min_best_localization": None})
        assert adata.n_vars == 3

    def test_sample_ids_normalized(self, tmp_path):
        p = _make_synthetic_abundance(tmp_path)
        adata = read_fragpipe_sites(p)
        # Paths stripped, "_uncalibrated" suffix stripped, ".mzML" ext stripped
        assert list(adata.obs_names) == ["sample_01", "sample_02", "sample_03"]

    def test_full_key_format(self, tmp_path):
        p = _make_synthetic_abundance(tmp_path)
        adata = read_fragpipe_sites(p)
        # Each key: Protein|Gene|Site|Mmult
        for k in adata.var.index:
            parts = k.split("|")
            assert len(parts) == 4, f"expected 4 fields, got {parts}"
            assert parts[3].startswith("M")

    def test_var_columns_present(self, tmp_path):
        p = _make_synthetic_abundance(tmp_path)
        adata = read_fragpipe_sites(p)
        expected = {
            "short_key",
            "pg_key",
            "protein_group_id",
            "gene",
            "site_aa",
            "site_position",
            "multiplicity",
            "best_localization",
            "sequence_window",
            "kinase_sequence",
        }
        assert expected.issubset(set(adata.var.columns))

    def test_log2_transform_applied(self, tmp_path):
        p = _make_synthetic_abundance(tmp_path)
        adata = read_fragpipe_sites(p)
        # Linear 1000 -> log2 = ~9.97
        assert abs(float(np.nanmean(adata.X)) - np.log2(np.nanmean(np.exp2(adata.X)))) < 1.0
        # Values around 10-11 range for log2(1000-2000)
        vals = adata.X[~np.isnan(adata.X)]
        assert vals.min() > 5 and vals.max() < 15

    def test_zero_and_nan_become_nan(self, tmp_path):
        p = _make_synthetic_abundance(tmp_path)
        adata = read_fragpipe_sites(p, advanced={"min_best_localization": None})
        # Third row (P00000|GENE3|S99|M1) has NaN + 0.0 + 550.0 -> two NaN in X
        gene3_col = adata.var.index[adata.var["gene"] == "GENE3"][0]
        col_idx = adata.var.index.get_loc(gene3_col)
        col_values = adata.X[:, col_idx]
        assert np.isnan(col_values).sum() == 2

    def test_kinase_sequence_converted(self, tmp_path):
        p = _make_synthetic_abundance(tmp_path)
        adata = read_fragpipe_sites(p)
        s = adata.var["kinase_sequence"].iloc[0]
        assert s.startswith("_") and s.endswith("_")
        assert "*S*" in s or "*T*" in s or "*Y*" in s

    def test_kinase_sequence_disabled(self, tmp_path):
        p = _make_synthetic_abundance(tmp_path)
        adata = read_fragpipe_sites(p, advanced={"add_kinase_sequence": False})
        assert "kinase_sequence" not in adata.var.columns

    def test_condition_df_joined(self, tmp_path):
        p = _make_synthetic_abundance(tmp_path)
        cdf = pd.DataFrame(
            {
                "sample": ["sample_01", "sample_02", "sample_03"],
                "condition": ["ctrl", "trt", "trt"],
            }
        )
        adata = read_fragpipe_sites(p, condition_df=cdf)
        assert "condition" in adata.obs.columns
        assert set(adata.obs["condition"].dropna()) == {"ctrl", "trt"}

    def test_uns_stamped(self, tmp_path):
        p = _make_synthetic_abundance(tmp_path)
        adata = read_fragpipe_sites(p)
        assert "alphaphos" in adata.uns
        u = adata.uns["alphaphos"]
        assert u["source"] == "fragpipe_site_matrix"
        assert u["source_file"].endswith("abundance_single-site_MS2quant_None.tsv")
        assert "pipeline_params" in u
        assert u["stats"]["n_rows_loaded"] == 3

    def test_layers_intensity_log2_present(self, tmp_path):
        p = _make_synthetic_abundance(tmp_path)
        adata = read_fragpipe_sites(p)
        assert "intensity_log2" in adata.layers


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------


class TestDirectoryResolution:
    def test_resolves_MS2_None_by_default(self, tmp_path):
        _make_synthetic_abundance(tmp_path)
        # Passing the DIRECTORY should resolve to the same MS2/None file.
        adata = read_fragpipe_sites(tmp_path)
        assert adata.n_vars == 2  # matches specific-file result at default filter

    def test_directory_wrong_file_raises(self, tmp_path):
        _make_synthetic_abundance(tmp_path)
        # We only wrote the MS2/None variant; asking for MS1 should fail.
        with pytest.raises(FileNotFoundError, match="Expected FragPipe abundance file"):
            read_fragpipe_sites(tmp_path, advanced={"quant_level": "MS1"})


# ---------------------------------------------------------------------------
# Optional: real FragPipe reference file
# ---------------------------------------------------------------------------


_REAL_FRAGPIPE_DIR = Path("W:/User/Denys/dia-quant-output")


@pytest.mark.skipif(
    not (_REAL_FRAGPIPE_DIR / "abundance_single-site_MS2quant_None.tsv").exists(),
    reason="Real FragPipe reference file not available (W: share).",
)
class TestRealFragpipeOutput:
    def test_reads_full_matrix(self):
        adata = read_fragpipe_sites(_REAL_FRAGPIPE_DIR)
        # Reference: 3 samples in dia-quant-output/. Sites vary; require substantial.
        assert adata.n_obs == 3, f"expected 3 samples, got {adata.n_obs}"
        assert adata.n_vars > 1000, f"suspiciously few sites: {adata.n_vars}"

    def test_all_keys_valid_pipe_format(self):
        adata = read_fragpipe_sites(_REAL_FRAGPIPE_DIR)
        assert adata.var.index.is_unique
        for k in adata.var.index[:100]:  # spot-check
            assert k.count("|") == 3

    def test_kinase_sequences_have_center_marker(self):
        adata = read_fragpipe_sites(_REAL_FRAGPIPE_DIR)
        # 90+% of sites should have a valid kinase_sequence
        marked = adata.var["kinase_sequence"].str.contains(r"\*[STY]\*", regex=True, na=False)
        assert marked.mean() > 0.9, f"only {marked.mean():.1%} have *[STY]* center"
