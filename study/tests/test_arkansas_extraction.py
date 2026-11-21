"""Source parsing and identity checks; no Arkansas outcome magnitudes printed."""

import importlib.util
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "arkansas_prepare", ROOT / "src/prepare_arkansas.py"
)
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)


def test_synthetic_hr_tr_remain_separate():
    assert ar.value("51-72", "milling") == (51, 72)
    with pytest.raises(AssertionError):
        ar.value("72-51", "milling")
    with pytest.raises(ValueError):
        ar.value("0.51", "milling")


@pytest.mark.parametrize("token", ["", "--", "—", "–", "."])
def test_missing_labels_are_unknown(token):
    assert ar.value(token, "milling") is None
    assert ar.value(token, "grain") is None


def test_bushel_conversion_is_mass_not_observed_test_weight():
    assert ar.KG_HA_PER_BU_AC == pytest.approx(45 * 0.45359237 / 0.40468564224)
    assert ar.KG_HA_PER_BU_AC != pytest.approx(40 * 0.45359237 / 0.40468564224)


def test_superscript_lodging_does_not_merge_into_yield():
    anchor = {"top": 100, "size": 10}
    words = [
        {"text": "160", "top": 98.8, "size": 10, "x0": 90, "x1": 106},
        {"text": "25", "top": 98, "size": 6, "x0": 107, "x1": 113},
    ]
    assert ar.cell(words, 100, 40, anchor) == ("160", "25")
    with pytest.raises(ValueError):
        ar.value("16025", "grain")


def record(year, value, annual=False, page=3):
    return {
        "report_year": year,
        "source_page": page,
        "value": value,
        "source_type": (
            "contemporaneous_annual_matrix"
            if annual
            else "individual_year_in_site_table"
        ),
    }


def test_annual_record_wins_even_when_missing():
    missing = record(2021, None, True)
    later = record(2022, 123)
    assert ar.select_record([later, missing]) is missing


def test_overlapping_reports_do_not_duplicate_or_average_cells():
    early = record(2016, (51, 72))
    late = record(2017, (53, 72))
    assert ar.select_record([late, early]) is early
    assert ar.select_record([record(2015, None), early]) is early


def test_documented_site_aliases_and_historical_year_ambiguity():
    assert ar.SITE_ALIAS == {"PTBS": "PTRS", "NBES": "NEC"}
    assert ar.norm("NERREC") != ar.norm("NEREC")
    assert ar.historical_site(None, 2019, 4, "inconsistent year title") == []
    assert ar.norm("Ozark") != ar.norm("20AR185")


@pytest.fixture
def prepared():
    p = ROOT / "derived/arkansas/public_features.csv"
    if not p.exists():
        pytest.skip("Run Arkansas preparation before artifact integration checks")
    return pd.read_csv(p), pd.read_csv(p.with_name("private_outcomes.csv"))


def test_exact_cultivar_site_year_join_and_quarantine(prepared):
    public, private = prepared
    assert public.row_id.is_unique and private.row_id.is_unique
    assert not public.duplicated(["year", "site", "candidate_id"]).any()
    assert set(public.row_id) == set(private.row_id)
    merged = public.merge(
        private, on="row_id", validate="one_to_one", suffixes=("_public", "_private")
    )
    assert (merged.pool_id_public == merged.pool_id_private).all()
    assert (merged.candidate_id_public == merged.candidate_id_private).all()
    assert not set(public.columns) & {
        "hr_percent",
        "tr_percent",
        "head_output_kg_ha",
        "raw_milling_cell",
    }


def test_published_units_and_missing_outcomes_preserved(prepared):
    public, private = prepared
    assert (public.moisture_basis_percent == 12).all() and (
        public.bushel_mass_lb == 45
    ).all()
    assert np.allclose(
        public.field_yield,
        public.field_yield_bu_ac * ar.KG_HA_PER_BU_AC,
        equal_nan=True,
    )
    assert private.hr_percent.isna().any()
    assert (private.hr_percent.dropna() % 1 == 0).all()
    assert (public.field_yield_bu_ac.dropna() % 1 == 0).all()
    assert (private.milling_denominator == "original rough-rice mass").all()


def test_previously_seen_pool_and_compromised_exclusions(prepared):
    public, _ = prepared
    assert (
        public.previously_seen == (public.year.eq(2025) & public.site.eq("NERREC"))
    ).all()
    assert not public.loc[public.pool_compromised, "eligible"].any()
    observed = public.lodging_percent.fillna(0) > 0
    assert public.loc[observed & ~public.pool_compromised, "eligible"].any()


def test_contemporaneous_candidate_universe_and_ambiguous_sources(prepared):
    public, _ = prepared
    assert (public.grain_source_report == public.year).all()
    cells = pd.read_csv(ROOT / "derived/arkansas/private_all_extracted_cells.csv")
    assert not ((cells.report_year == 2019) & (cells.source_page == 4)).any()
    assert not ((cells.site == "NERREC") & (cells.year <= 2021)).any()
    annual = cells[
        (cells.kind == "grain") & (cells.source_type == "contemporaneous_annual_matrix")
    ]
    assert set(zip(public.year, public.site, public.candidate_id)) == set(
        zip(annual.year, annual.site, annual.candidate_id)
    )


def test_all_downloaded_source_hashes():
    path = ROOT / "raw/arkansas/source_manifest.json"
    if not path.exists():
        pytest.skip("Sources not acquired")
    for r in json.loads(path.read_text()):
        if r.get("downloaded"):
            assert ar.sha(ROOT / r["file"]) == r["sha256"]
