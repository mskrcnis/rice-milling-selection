"""Prepare outcome-separated INIA candidates from checksum-verified original files."""

from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

B = Path(__file__).resolve().parents[1]
R = B / "raw/uruguay"
D = B / "derived/uruguay"
A = B / "audit"
EXPECTED = {
    "README.md": "da3cee8cee3121dc1f90c8b86604a070eda7bc4e2215b6106329235e0488ea52",
    "Phenotypes.txt": "8e587890c3d5d479f4f43b7202eb365c4b866f4fe6f15c5d709ff2bc3eb33d0b",
    "Trials.txt": "2a75b0b5488fd243093bb4f6178e31142c0f42e8e739f58130782529f7dc103a",
    "Lines.txt": "644417788d35a4e236e084ecc24a230c9ed13c5896befdd052409470e6fed4f6",
}
K = ["YEAR", "TRIAL", "LOCATION"]
C = K + ["LINE_ID"]
P = C + ["BLOCK"]
CLASS = {
    "Indica": "long_grain_indica",
    "Tropical Japonica": "long_grain_tropical_japonica",
    "Temperate Japonica": "temperate_japonica_mixed_grain",
    "Clearfield": "clearfield_long_grain",
}
MILLING = ["YAM", "PHR", "GC", "SM", "LW_RATIO", "AC", "ADV"]


def slug(*values):
    return "__".join(str(v).replace(" ", "_") for v in values)


def main():
    D.mkdir(parents=True, exist_ok=True)
    for name, sha in EXPECTED.items():
        if hashlib.sha256((R / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Source SHA mismatch: " + name)
    p = pd.read_csv(R / "Phenotypes.txt", sep=r"\s+", na_values="NA")
    t = pd.read_csv(R / "Trials.txt", sep=r"\s+", na_values="NA")
    lines = (R / "Lines.txt").read_text().splitlines()
    l = pd.DataFrame(
        [
            s.split()[:4] + [" ".join(s.split()[4:-2])] + s.split()[-2:]
            for s in lines[1:]
        ],
        columns=lines[0].split(),
    ).replace("NA", pd.NA)
    assert list(p) == [
        "YEAR",
        "TRIAL",
        "LOCATION",
        "BLOCK",
        "LINE_ID",
        "GY",
        "PH",
        "FD",
        "DTF",
        "LB",
        "NO",
        "ROS",
        "YAM",
        "PHR",
        "GC",
        "SM",
        "LW_RATIO",
        "AC",
        "ADV",
    ]
    assert len(p) == 91636 and len(t) == 996 and len(l) == 20628
    assert not t.duplicated(K).any() and not l.LINE_ID.duplicated().any()
    original = list(p)
    p["source_row"] = np.arange(len(p)) + 2
    p["exact_duplicate"] = p.duplicated(original, keep="first")
    duplicates = p[p.duplicated(P, keep=False)][P + ["source_row", "exact_duplicate"]]
    duplicates.to_csv(A / "uruguay_duplicate_keys.csv", index=False)
    p["yield_valid"] = p.GY.notna() & np.isfinite(p.GY) & (p.GY > 0)
    p["hr_valid"] = p.PHR.notna() & np.isfinite(p.PHR) & p.PHR.between(0, 100)
    p["joint_valid"] = p.yield_valid & p.hr_valid
    p = p.merge(
        t, on=K, how="left", validate="many_to_one", indicator="trial_join"
    ).merge(l, on="LINE_ID", how="left", validate="many_to_one", indicator="line_join")
    assert p.trial_join.eq("both").all() and p.line_join.eq("both").all()
    p["stage"] = p.EVALUATION_STAGE.fillna("unknown")
    p["grain_class"] = p.GERMPLASM_TYPE.map(CLASS).fillna("unknown")
    p["pool_id"] = [
        slug("UY", y, s, tr) for y, s, tr in zip(p.YEAR, p.LOCATION, p.TRIAL)
    ]
    p["row_id"] = p.pool_id + "__" + p.LINE_ID
    p["used_in_candidate"] = ~p.exact_duplicate
    p.to_csv(D / "private_plot_records.csv", index=False)
    p.drop(
        columns=MILLING
        + ["HERITABILITY", "CV", "GENETIC_VARIANCE", "RESIDUAL_VARIANCE"]
    ).to_csv(D / "public_plot_metadata.csv", index=False)
    clean = p.loc[p.used_in_candidate].copy()
    clean["gy_valid_value"] = clean.GY.where(clean.yield_valid)
    clean["hr_valid_value"] = clean.PHR.where(clean.hr_valid)
    clean["paired_gy"] = clean.GY.where(clean.joint_valid)
    clean["paired_hr"] = clean.PHR.where(clean.joint_valid)
    clean["paired_product"] = clean.GY.mul(clean.PHR).div(100).where(clean.joint_valid)
    pub = []
    priv = []
    for rid, g in clean.groupby("row_id", sort=True, dropna=False):
        first = g.iloc[0]
        ny = int(g.yield_valid.sum())
        nh = int(g.hr_valid.sum())
        nj = int(g.joint_valid.sum())
        gy = g.gy_valid_value.mean()
        hr = g.hr_valid_value.mean()
        samehr = bool(nh > 1 and g.hr_valid_value.nunique(dropna=True) == 1)
        reasons = []
        if ny == 0:
            reasons.append("no_observed_positive_field_yield")
        if nh == 0:
            reasons.append("no_observed_valid_head_rice_recovery")
        row = {
            "row_id": rid,
            "pool_id": first.pool_id,
            "candidate_id": first.LINE_ID,
            "site": first.LOCATION,
            "year": int(first.YEAR),
            "trial": first.TRIAL,
            "field_yield": gy,
            "field_yield_unit": "kg/ha",
            "yield_moisture_basis": "not_specified_in_archive",
            "grain_class": first.grain_class,
            "germplasm_type": first.GERMPLASM_TYPE,
            "stage": first.stage,
            "replicate_count": int(g.BLOCK.nunique(dropna=True)),
            "observation_count": len(g),
            "yield_observation_count": ny,
            "hr_observation_count": nh,
            "matched_plot_count": nj,
            "plant_height_cm": g.PH.mean(),
            "days_to_flowering": g.DTF.mean(),
            "sowing_date": first.SOWING_DATE,
            "harvest_date": first.HARVEST_DATE,
            "eligible_observed": bool(ny > 0 and nh > 0),
            "exclusion_reason": ";".join(reasons),
            "source_rows": ";".join(map(str, g.source_row)),
            "repeated_candidate_within_block": bool(g.duplicated(["BLOCK"]).any()),
            "assay_sharing_status": "unknown_not_identifiable_from_archive",
            "check_status": "unknown_coded_line_ids",
            "source_doi": "10.5061/dryad.x69p8czn8",
        }
        pub.append(row)
        priv.append(
            {
                "row_id": rid,
                "pool_id": first.pool_id,
                "candidate_id": first.LINE_ID,
                "year": int(first.YEAR),
                "hr_percent": hr,
                "hr_fraction": hr / 100,
                "hr_unit": "percent",
                "hr_denominator": "rough_rice_supported_by_same_program_methods_not_archive_explicit",
                "total_milled_recovery_percent": g.YAM.mean(),
                "output_product_of_means": gy * hr / 100,
                "output_mean_of_matched_plot_products": g.paired_product.mean(),
                "matched_yield_mean": g.paired_gy.mean(),
                "matched_hr_mean": g.paired_hr.mean(),
                "matched_product_of_means": g.paired_gy.mean()
                * g.paired_hr.mean()
                / 100,
                "output_unit": "kg/ha_proxy_rough_rice_basis_inferred_moisture_unspecified",
                "eligible_observed": bool(ny > 0 and nh > 0),
                "matched_plot_count": nj,
                "hr_observation_count": nh,
                "hr_distinct_value_count": int(g.hr_valid_value.nunique(dropna=True)),
                "same_reported_hr_across_rows": samehr,
                "physical_assay_count": pd.NA,
                "source_rows": row["source_rows"],
            }
        )
    public = (
        pd.DataFrame(pub)
        .sort_values(
            ["year", "site", "trial", "grain_class", "candidate_id"], kind="stable"
        )
        .reset_index(drop=True)
    )
    private = pd.DataFrame(priv).set_index("row_id").loc[public.row_id].reset_index()
    assert (
        public.row_id.is_unique
        and private.row_id.is_unique
        and public.row_id.equals(private.row_id)
    )
    public.to_csv(D / "public_features.csv", index=False)
    private.to_csv(D / "private_outcomes.csv", index=False)
    coverage = (
        public.groupby(["year", "site", "stage", "grain_class"], dropna=False)
        .agg(
            candidates=("row_id", "size"),
            pools=("pool_id", "nunique"),
            observed_candidates=("eligible_observed", "sum"),
            yield_records=("yield_observation_count", "sum"),
            hr_records=("hr_observation_count", "sum"),
            paired_records=("matched_plot_count", "sum"),
        )
        .reset_index()
    )
    coverage.to_csv(A / "uruguay_coverage.csv", index=False)
    pools = (
        public.groupby(["pool_id", "year", "site", "trial", "stage"], dropna=False)
        .agg(
            candidates=("row_id", "size"),
            observed_candidates=("eligible_observed", "sum"),
            observations=("observation_count", "sum"),
            grain_classes=("grain_class", "nunique"),
        )
        .reset_index()
    )
    pools.to_csv(A / "uruguay_pool_metadata.csv", index=False)
    coverage_year = (
        public.groupby("year")
        .agg(
            candidates=("row_id", "size"),
            pools=("pool_id", "nunique"),
            sites=("site", "nunique"),
            observed_candidates=("eligible_observed", "sum"),
        )
        .reset_index()
    )
    goodp = (
        pools[pools.observed_candidates >= 10]
        .groupby("year")
        .agg(
            pools_ge10_observed=("pool_id", "nunique"),
            candidates_in_pools_ge10=("observed_candidates", "sum"),
        )
        .reset_index()
    )
    coverage_year.merge(goodp, on="year", how="left").fillna(0).to_csv(
        A / "uruguay_year_coverage.csv", index=False
    )
    comparison = []
    for included, g in public.groupby("eligible_observed"):
        for feature in [
            "field_yield",
            "plant_height_cm",
            "days_to_flowering",
            "replicate_count",
        ]:
            comparison.append(
                {
                    "eligible_observed": included,
                    "feature": feature,
                    "candidate_count": len(g),
                    "nonmissing_count": int(g[feature].notna().sum()),
                    "mean": g[feature].mean(),
                    "median": g[feature].median(),
                }
            )
    pd.DataFrame(comparison).to_csv(
        A / "uruguay_inclusion_field_attributes.csv", index=False
    )
    summary = {
        "source_rows": len(p),
        "unique_trials": len(t),
        "source_line_records": len(l),
        "phenotyped_lines": int(p.LINE_ID.nunique()),
        "source_years": sorted(map(int, p.YEAR.unique())),
        "source_sites": int(p.LOCATION.nunique()),
        "nonmissing_yield_rows": int(p.GY.notna().sum()),
        "nonmissing_hr_rows": int(p.PHR.notna().sum()),
        "joint_nonmissing_rows": int((p.GY.notna() & p.PHR.notna()).sum()),
        "invalid_observed_yield_rows": int((p.GY.notna() & ~p.yield_valid).sum()),
        "invalid_observed_hr_rows": int((p.PHR.notna() & ~p.hr_valid).sum()),
        "exact_duplicate_rows_removed": int(p.exact_duplicate.sum()),
        "repeated_plot_keys_beyond_first_raw": int(p.duplicated(P).sum()),
        "candidate_rows": len(public),
        "eligible_observed_candidates": int(public.eligible_observed.sum()),
        "candidate_pools": int(public.pool_id.nunique()),
        "pools_with_multiple_source_classes": int((pools.grain_classes > 1).sum()),
        "pools_ge10_observed": int((pools.observed_candidates >= 10).sum()),
        "nonmissing_stage_candidates": int(public.stage.ne("unknown").sum()),
        "known_grain_class_candidates": int(public.grain_class.ne("unknown").sum()),
        "candidate_within_block_repetitions": int(
            public.repeated_candidate_within_block.sum()
        ),
        "same_hr_reported_on_multiple_rows_candidates": int(
            private.same_reported_hr_across_rows.sum()
        ),
        "physical_assay_sharing_identifiable": False,
        "yield_moisture_basis": "not_specified",
        "hr_denominator_archive_explicit": False,
        "hr_denominator_same_program_support": "Monteverde et al.2018 DOI10.2135/cropsci2017.09.0564 p1521 rough rice mass denominator",
        "aggregation": "remove exact duplicate source records; arithmetic mean of all observed valid field yields times arithmetic mean of all observed valid PHR; matched plot product sensitivity retained",
        "exposure": "source extractor outcome-exposed; model developer receives public metadata only before split",
    }
    (A / "uruguay_preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    manifest = [
        {
            "path": str(x.relative_to(B)),
            "sha256": hashlib.sha256(x.read_bytes()).hexdigest(),
            "bytes": x.stat().st_size,
        }
        for x in [
            D / "public_features.csv",
            D / "private_outcomes.csv",
            D / "private_plot_records.csv",
            D / "public_plot_metadata.csv",
            A / "uruguay_coverage.csv",
            A / "uruguay_pool_metadata.csv",
            A / "uruguay_year_coverage.csv",
        ]
    ]
    (A / "uruguay_derived_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
