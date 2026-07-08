"""Unit tests for ``alphaphos.enrichment.matching`` on the mini fixture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaphos.enrichment import (
    TEST_FIXTURE_PATH,
    canonicalise_site_ids,
    load_ptm_db,
    match_sites,
    parse_alphaphos_key,
)
from alphaphos.enrichment.matching import attach_site_ids

pytestmark = pytest.mark.skipif(
    not TEST_FIXTURE_PATH.exists(),
    reason=f"mini PTM DB fixture missing at {TEST_FIXTURE_PATH}",
)


@pytest.fixture(scope="module")
def db() -> pd.DataFrame:
    return load_ptm_db(TEST_FIXTURE_PATH)


@pytest.fixture(scope="module")
def db_sample_key(db) -> str:
    """Grab a real DB row and format it as an alphaPhos key we can match."""
    row = db.iloc[0]
    prot = str(row["substrate_uniprot"]).split(";", 1)[0].split("-", 1)[0]
    gene = str(row["substrate_gene"]).split(";", 1)[0]
    residue = str(row["residue"])
    pos = int(row["position"])
    return f"{prot}|{gene}|{residue}{pos}|M1"


class TestParseAlphaphosKey:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            (
                "P00533|EGFR|Y1172|M1",
                {
                    "protein": "P00533",
                    "gene": "EGFR",
                    "residue": "Y",
                    "position": 1172,
                    "multiplicity": 1,
                },
            ),
            (
                # multi-protein group -> first accession
                "P57059;A0A0B4J2F2|SIK1|S575|M2",
                {
                    "protein": "P57059",
                    "gene": "SIK1",
                    "residue": "S",
                    "position": 575,
                    "multiplicity": 2,
                },
            ),
            (
                "Q13164|MAPK7|T733|M3",
                {
                    "protein": "Q13164",
                    "gene": "MAPK7",
                    "residue": "T",
                    "position": 733,
                    "multiplicity": 3,
                },
            ),
        ],
    )
    def test_valid_forms(self, key, expected):
        p = parse_alphaphos_key(key)
        assert p is not None
        for k, v in expected.items():
            assert getattr(p, k) == v

    @pytest.mark.parametrize(
        "bad_key",
        [
            "not_an_alphaphos_key",
            "P00533|EGFR|A100|M1",  # A is not S/T/Y
            "P00533|EGFR|Y1172",  # missing multiplicity
            "|EGFR|Y1172|M1",  # empty protein
            "P00533|EGFR|Yfoo|M1",  # non-numeric position
            None,  # non-string
            42,  # non-string
        ],
    )
    def test_bad_input_returns_none(self, bad_key):
        assert parse_alphaphos_key(bad_key) is None


class TestCanonicaliseSiteIds:
    """Bridge between alphaPhos site keys and PTM-DB GMT ``Protein_AApos`` IDs."""

    def test_basic_conversion(self):
        keys = ["P00533|EGFR|Y1172|M1", "Q13164|MAPK7|T733|M3"]
        got = canonicalise_site_ids(keys)
        assert got == ["P00533_Y1172", "Q13164_T733"]

    def test_multi_protein_group_first_accession(self):
        got = canonicalise_site_ids(["P57059;A0A0B4J2F2|SIK1|S575|M2"])
        assert got == ["P57059_S575"]

    def test_unparseable_dropped_by_default(self):
        keys = ["P00533|EGFR|Y1172|M1", "junk", "already_P00533_Y999"]
        got = canonicalise_site_ids(keys)
        # "junk" and "already_P00533_Y999" are both unparseable -> dropped.
        assert got == ["P00533_Y1172"]

    def test_unparseable_kept_when_flag_off(self):
        # drop_unparseable=False: keep unparseable strings as-is (lets you
        # pass a mixed list of alphaPhos keys and already-canonical IDs).
        keys = ["P00533|EGFR|Y1172|M1", "already_canonical_P42_S1", "junk"]
        got = canonicalise_site_ids(keys, drop_unparseable=False)
        assert got == ["P00533_Y1172", "already_canonical_P42_S1", "junk"]

    def test_accepts_pandas_index(self):
        # Real use case: hand it df.index directly, not a list.
        idx = pd.Index(["P00533|EGFR|Y1172|M1", "Q13164|MAPK7|T733|M3"])
        got = canonicalise_site_ids(idx)
        assert got == ["P00533_Y1172", "Q13164_T733"]

    def test_end_to_end_anova_hits_to_ora_ready(self):
        # The pattern users will actually write:
        #   hits, bg = ap.anova_hits(anova)
        #   hits = ap.enrichment.canonicalise_site_ids(hits)
        #   bg   = ap.enrichment.canonicalise_site_ids(bg)
        #   ap.enrichment.ora(hits, bg, libraries=PTM_LIBS)
        # This test verifies the format shape ora expects.
        import alphaphos as ap

        anova = pd.DataFrame(
            {"fdr": [0.001, 0.001, 0.5]},
            index=["P00533|EGFR|Y1172|M1", "Q13164|MAPK7|T733|M3", "P57059|SIK1|S575|M1"],
        )
        hits, bg = ap.anova_hits(anova, fdr_threshold=0.05)
        hits = canonicalise_site_ids(hits)
        bg = canonicalise_site_ids(bg)
        assert set(hits) == {"P00533_Y1172", "Q13164_T733"}
        assert set(bg) == {"P00533_Y1172", "Q13164_T733", "P57059_S575"}
        # hits is a proper subset of bg (a hard precondition of ora()).
        assert set(hits).issubset(set(bg))


class TestMatchSites:
    def test_real_db_row_matches_tier2(self, db, db_sample_key):
        mr = match_sites([db_sample_key], db=db)
        assert mr.stats["n_matched_tier2"] == 1
        assert mr.stats["n_unmatched"] == 0
        assert mr.matched["match_tier"].iloc[0] == "uniprot_position"

    def test_unknown_uniprot_but_known_gene_falls_to_tier3(self, db):
        # Take a real DB row, replace its uniprot with a fake one but
        # keep gene + position, expect tier-3 fallback.
        row = db.dropna(subset=["substrate_gene"]).iloc[0]
        gene = str(row["substrate_gene"]).split(";", 1)[0]
        residue = str(row["residue"])
        pos = int(row["position"])
        fake_key = f"NOTAREAL|{gene}|{residue}{pos}|M1"
        mr = match_sites([fake_key], db=db)
        assert mr.stats["n_matched_tier3"] == 1
        assert mr.stats["n_matched_tier2"] == 0

    def test_unknown_uniprot_and_gene_reported_as_unmatched(self, db):
        mr = match_sites(["NOTAREAL|NOTAGENE|S9999|M1"], db=db)
        assert mr.stats["n_matched_tier2"] == 0
        assert mr.stats["n_matched_tier3"] == 0
        assert mr.stats["n_unmatched"] == 1
        assert mr.unmatched["reason"].iloc[0] == "not_in_db"

    def test_unparseable_keys_reported(self, db):
        mr = match_sites(["garbage", "P00533|EGFR|Y1172|M1foo"], db=db)
        assert mr.stats["n_unmatched"] == 2
        assert set(mr.unmatched["reason"]) == {"unparseable_key"}

    def test_multi_protein_group_query_matches_via_first(self, db, db_sample_key):
        # If the user's key is a multi-protein group whose first
        # accession is the DB match, tier-2 fires.
        parsed = parse_alphaphos_key(db_sample_key)
        multi_key = f"{parsed.protein};OTHER|{parsed.gene}|{parsed.residue}{parsed.position}|M1"
        mr = match_sites([multi_key], db=db)
        assert mr.stats["n_matched_tier2"] == 1

    def test_user_isoform_falls_through_to_tier3(self, db, db_sample_key):
        # User carries an isoform tag (``P00533-2``); the DB was indexed
        # under the base accession (``P00533``).  Current behavior:
        # tier-2 lookup misses (isoform tag not stripped on the query
        # side), but the gene+position tier-3 lookup rescues it.
        # Documenting this so a future "strip isoform on query side"
        # patch surfaces here as a tier-2 hit.
        parsed = parse_alphaphos_key(db_sample_key)
        iso_key = f"{parsed.protein}-2|{parsed.gene}|{parsed.residue}{parsed.position}|M1"
        mr = match_sites([iso_key], db=db)
        assert mr.stats["n_matched_tier3"] == 1
        assert mr.stats["n_matched_tier2"] == 0

    @pytest.mark.parametrize(
        "uniprot_dtype",
        ["object", "string", "string[pyarrow]"],
    )
    def test_nan_uniprot_rows_are_skipped_not_crashing(self, uniprot_dtype):
        # Regression: CI runs 28884069731 / 28884037587 failed with
        # AttributeError: 'float' object has no attribute 'split' inside
        # _build_lookup_indices because 249 rows in the shipped test DB
        # had NaN in substrate_uniprot, and .astype(str) on some pandas
        # backends leaks NaN/pd.NA through instead of coercing to "nan".
        # A row with NaN uniprot must be quietly skipped, and the rest
        # of the DB must still index correctly.
        na_value = pd.NA if uniprot_dtype.startswith("string") else np.nan
        db = pd.DataFrame(
            {
                "substrate_uniprot": pd.array(["P00533", na_value, "P42229"], dtype=uniprot_dtype),
                "substrate_gene": ["EGFR", "GENE_NO_UP", "STAT5A"],
                "residue": ["Y", "S", "Y"],
                "position": [1172, 100, 694],
            }
        )
        mr = match_sites(["P00533|EGFR|Y1172|M1"], db=db)
        assert mr.stats["n_matched_tier2"] + mr.stats["n_matched_tier3"] == 1
        assert mr.stats["n_unmatched"] == 0


class TestAttachSiteIds:
    def test_attaches_column_and_keeps_unmatched(self, db, db_sample_key):
        # Two rows: one real DB site, one nonsense; the nonsense should
        # come back with site_id NaN.
        df = pd.DataFrame(
            {"log2fc": [1.5, -2.0]},
            index=[db_sample_key, "NOTAREAL|NOTAGENE|S9999|M1"],
        )
        out = attach_site_ids(df, db=db)
        assert "site_id" in out.columns
        assert out.loc[db_sample_key, "site_id"] is not None
        assert pd.isna(out.loc["NOTAREAL|NOTAGENE|S9999|M1", "site_id"])
        # MatchResult stashed in attrs for post-hoc inspection.
        mr = out.attrs["match_result"]
        assert mr.stats["n_input"] == 2
        assert mr.stats["n_unmatched"] == 1

    def test_key_column_argument(self, db, db_sample_key):
        # Same as above but via an explicit column instead of the index.
        df = pd.DataFrame(
            {"key": [db_sample_key, "NOTAREAL|NOTAGENE|S9999|M1"], "log2fc": [1.5, -2.0]}
        )
        out = attach_site_ids(df, key_column="key", db=db)
        assert "site_id" in out.columns
        assert out["site_id"].isna().sum() == 1
