"""Tests for :func:`alphaphos.qc.load_acquisition_queue`.

Runs against both the shipped cardio Xcalibur queue (real-format
regression) and synthetic queues (edge cases).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alphaphos.qc.queue_io import DEFAULT_BLANK_PATTERN, load_acquisition_queue

REPO = Path(__file__).resolve().parents[2]
CARDIO_QUEUE = REPO / "test_data" / "qc" / "queue_ElKr_phosphoDVP.csv"


class TestLoadAcquisitionQueue:
    def test_cardio_queue_shape_and_columns(self):
        q = load_acquisition_queue(CARDIO_QUEUE)
        assert list(q.columns) == ["sample", "injection_order", "position", "instrument_method"]
        # 158 total rows minus 4 blanks
        assert len(q) == 154

    def test_cardio_queue_provenance(self):
        q = load_acquisition_queue(CARDIO_QUEUE)
        prov = q.attrs["provenance"]
        assert prov["format_detected"] == "xcalibur"
        assert prov["n_total_rows"] == 158
        assert prov["n_blank_removed"] == 4
        assert all("blank" in name for name in prov["blank_samples"])

    def test_injection_order_preserved_across_blanks(self):
        q = load_acquisition_queue(CARDIO_QUEUE)
        # Original blanks sat at positions 78, 79, 157, 158.  So there
        # should be no injection_order 78 or 79 in the output.
        assert 78 not in q["injection_order"].tolist()
        assert 79 not in q["injection_order"].tolist()
        # But 77 (last phospho before blanks) and 80 (first proteome after)
        # should both be present.
        assert 77 in q["injection_order"].tolist()
        assert 80 in q["injection_order"].tolist()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_acquisition_queue(tmp_path / "does_not_exist.csv")

    def test_blank_pattern_disabled_keeps_all(self):
        q = load_acquisition_queue(CARDIO_QUEUE, blank_pattern=None)
        assert len(q) == 158
        assert q.attrs["provenance"]["n_blank_removed"] == 0

    def test_custom_blank_pattern(self, tmp_path):
        # Synthetic queue with a custom "wash" convention
        synth = tmp_path / "synth.csv"
        synth.write_text(
            "Bracket Type=4\n"
            "File Name,Path,Instrument Method,Position\n"
            "sample_01,D:\\,method.raw,S1:A1\n"
            "custom_wash_01,D:\\,method.raw,S1:A2\n"
            "sample_02,D:\\,method.raw,S1:A3\n",
            encoding="utf-8",
        )
        q = load_acquisition_queue(synth, blank_pattern=r"(?i)wash")
        assert len(q) == 2
        assert list(q["sample"]) == ["sample_01", "sample_02"]

    def test_rejects_non_xcalibur_csv(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("foo,bar\n1,2\n3,4\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Could not auto-detect"):
            load_acquisition_queue(bad, format="auto")

    def test_xcalibur_without_preamble(self, tmp_path):
        # Some Xcalibur exports drop the preamble line
        no_preamble = tmp_path / "no_preamble.csv"
        no_preamble.write_text(
            "File Name,Path,Instrument Method,Position\n"
            "sample_01,D:\\,method.raw,S1:A1\n"
            "sample_02,D:\\,method.raw,S1:A2\n",
            encoding="utf-8",
        )
        q = load_acquisition_queue(no_preamble)
        assert len(q) == 2
        assert list(q["injection_order"]) == [1, 2]

    def test_default_blank_pattern_matches_case_insensitive(self, tmp_path):
        # Confirm "BLANK", "Blank", "blank" all match
        synth = tmp_path / "case.csv"
        synth.write_text(
            "Bracket Type=4\n"
            "File Name,Path,Instrument Method,Position\n"
            "sample_01,D:\\,m,S1:A1\n"
            "BLANK_01,D:\\,m,S1:A2\n"
            "sample_02,D:\\,m,S1:A3\n"
            "Blank_02,D:\\,m,S1:A4\n"
            "sample_03,D:\\,m,S1:A5\n"
            "blank_03,D:\\,m,S1:A6\n",
            encoding="utf-8",
        )
        q = load_acquisition_queue(synth, blank_pattern=DEFAULT_BLANK_PATTERN)
        # 3 samples kept out of 6 rows
        assert len(q) == 3
        assert q.attrs["provenance"]["n_blank_removed"] == 3
