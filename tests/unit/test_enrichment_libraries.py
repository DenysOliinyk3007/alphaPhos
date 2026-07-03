"""Unit tests for ``alphaphos.enrichment.libraries`` on the mini fixture."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alphaphos.enrichment import (
    TEST_FIXTURE_PATH,
    emit_libraries,
    load_gmt,
    load_ptm_db,
    site_id,
)

pytestmark = pytest.mark.skipif(
    not TEST_FIXTURE_PATH.exists(),
    reason=f"mini PTM DB fixture missing at {TEST_FIXTURE_PATH}",
)


@pytest.fixture(scope="module")
def emitted(tmp_path_factory) -> dict[str, Path]:
    """Emit libraries from the mini fixture into a tmp dir; return paths."""
    out = tmp_path_factory.mktemp("libraries")
    emit_libraries(
        output_dir=out,
        db_path=TEST_FIXTURE_PATH,
        min_set_size=3,  # fixture is small; lower the bar so sets emerge
    )
    return {p.stem: p for p in out.glob("*.gmt")} | {"manifest": out / "manifest.csv"}


class TestSiteId:
    def test_canonical_format(self):
        assert site_id("P00533", "Y", 1172) == "P00533_Y1172"

    def test_multi_uniprot_takes_first(self):
        # Multi-mapped UniProts (semicolon-joined) collapse to the first
        # accession — matches alphaPhos's PG-level convention.
        assert site_id("Q7Z6Z7;A6NHW2", "S", 300) == "Q7Z6Z7_S300"

    def test_position_coerced_to_int(self):
        # Some DB rows may have float positions from Excel roundtrip.
        assert site_id("P00533", "Y", 1172.0) == "P00533_Y1172"


class TestLoadPtmDb:
    def test_loads_fixture(self):
        db = load_ptm_db(TEST_FIXTURE_PATH)
        assert not db.empty
        assert "substrate_uniprot" in db.columns
        assert "effect_canonical" in db.columns

    def test_missing_path_raises_with_build_hint(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="scripts/build_ptm_db.py"):
            load_ptm_db(tmp_path / "does_not_exist.parquet")

    def test_missing_columns_raise(self, tmp_path):
        # Build a parquet missing a required column, confirm it's rejected.
        bad = tmp_path / "bad.parquet"
        pd.DataFrame({"substrate_uniprot": ["P1"], "position": [1]}).to_parquet(bad)
        with pytest.raises(ValueError, match="missing required columns"):
            load_ptm_db(bad)


class TestEmitLibraries:
    def test_emits_three_gmt_files(self, emitted):
        for name in ["functional_effect", "disease_variant", "functional_score"]:
            assert name in emitted, f"missing GMT {name}"
            assert emitted[name].exists()

    def test_manifest_lists_all_sets(self, emitted):
        manifest = pd.read_csv(emitted["manifest"])
        # At minimum the manifest must have library / set_name / n_sites cols.
        for col in ("library", "set_name", "n_sites", "description"):
            assert col in manifest.columns
        # Every emitted set should appear as a row.
        for gmt_name in ["functional_effect", "disease_variant", "functional_score"]:
            gmt = load_gmt(emitted[gmt_name])
            manifest_names = set(
                manifest.loc[
                    manifest["library"].str.lower().str.contains(gmt_name.split("_")[0]), "set_name"
                ]
            )
            assert set(gmt.keys()).issubset(manifest_names), (
                f"GMT {gmt_name} has sets not in manifest: {set(gmt.keys()) - manifest_names}"
            )

    def test_functional_effect_uses_canonical_categories(self, emitted):
        gmt = load_gmt(emitted["functional_effect"])
        valid = {
            "activates_activity",
            "inhibits_activity",
            "induces_ppi",
            "disrupts_ppi",
            "alters_stability",
        }
        assert set(gmt.keys()).issubset(valid), f"Unexpected effect sets: {set(gmt.keys()) - valid}"
        # alters_localization is explicitly excluded from v1.
        assert "alters_localization" not in gmt

    def test_disease_variant_sets(self, emitted):
        gmt = load_gmt(emitted["disease_variant"])
        assert set(gmt.keys()).issubset({"clinvar_site", "cancer_TCGA_site"})

    def test_functional_score_sets(self, emitted):
        gmt = load_gmt(emitted["functional_score"])
        assert set(gmt.keys()).issubset({"ochoa_top_quartile", "ochoa_bottom_quartile"})

    def test_site_ids_are_canonical(self, emitted):
        # Every site_id must match <uniprot>_<AA><pos>.
        for gmt_path in emitted.values():
            if gmt_path.suffix != ".gmt":
                continue
            gmt = load_gmt(gmt_path)
            for name, members in gmt.items():
                for sid in members:
                    parts = sid.split("_")
                    assert len(parts) >= 2, f"{gmt_path.name} :: {name} :: bad id {sid!r}"
                    aa_pos = parts[-1]
                    assert aa_pos[0] in "STY", f"{gmt_path.name} :: {name} :: aa not STY in {sid!r}"
                    assert aa_pos[1:].isdigit(), (
                        f"{gmt_path.name} :: {name} :: position not int in {sid!r}"
                    )

    def test_sets_are_deduplicated(self, emitted):
        for gmt_path in emitted.values():
            if gmt_path.suffix != ".gmt":
                continue
            gmt = load_gmt(gmt_path)
            for name, members in gmt.items():
                assert len(members) == len(set(members)), (
                    f"{gmt_path.name} :: {name}: duplicate site_ids"
                )


class TestMinSetSize:
    def test_high_threshold_drops_all_sets(self, tmp_path):
        # Fixture has ~200 sites per bucket; ask for 5000 -> nothing emerges.
        emit_libraries(
            output_dir=tmp_path,
            db_path=TEST_FIXTURE_PATH,
            min_set_size=5000,
        )
        manifest = pd.read_csv(tmp_path / "manifest.csv")
        assert manifest.empty


class TestRequireCurated:
    def test_curated_only_shrinks_or_equals(self, tmp_path):
        # A curated-only run should never produce MORE sites than the
        # unfiltered run (monotonicity of the filter).
        emit_libraries(output_dir=tmp_path / "all", db_path=TEST_FIXTURE_PATH, min_set_size=3)
        emit_libraries(
            output_dir=tmp_path / "curated",
            db_path=TEST_FIXTURE_PATH,
            min_set_size=3,
            require_curated=True,
        )
        m_all = pd.read_csv(tmp_path / "all" / "manifest.csv")
        m_cur = pd.read_csv(tmp_path / "curated" / "manifest.csv")
        # For each set that appears in both, curated-count <= all-count.
        for row in m_cur.itertuples():
            match = m_all[(m_all["library"] == row.library) & (m_all["set_name"] == row.set_name)]
            if not match.empty:
                assert row.n_sites <= match["n_sites"].iloc[0]


class TestLoadGmt:
    def test_round_trip(self, emitted):
        for gmt_path in emitted.values():
            if gmt_path.suffix != ".gmt":
                continue
            gmt = load_gmt(gmt_path)
            # Every non-empty set has at least one site.
            for _name, members in gmt.items():
                assert len(members) >= 1
