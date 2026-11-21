# Independent Uruguay source-to-candidate audit

Status: **PASS with documented repetition-flag semantics**. All 38,480 candidates in 996 trial pools were independently reconstructed from 91,636 phenotype rows. Nine exact full-row duplicates are removed once, leaving 91,627 records for aggregation. Source and derived files were preserved byte for byte.

The audit used a separate Python standard-library token parser, dictionary joins/grouping, and `math.fsum`, without importing the preparation code. The Lines file requires fixed first four and last two fields so that Tropical Japonica and Temperate Japonica remain single descriptors. All 20,628 line keys and 996 trial keys are unique and all phenotype joins resolve.

All candidate identities, complete source-row lists, observation/replicate/validity counts, source metadata, field-yield/height/flowering means, HR and total milling means, fraction conversion, product of marginal means, mean matched-row products, matched marginal means, and eligibility reasons pass. Every preserved raw field and source-join field in the plot artifacts was also compared. There are 128 check families and 7,878,202 individual comparisons. Numeric tolerances are 2×10⁻¹⁴ relative and 10⁻⁹ absolute; no discrepancies in numeric aggregates or identities were found.

The stored `repeated_candidate_within_block` flag marks 414 candidates. Independent inspection identifies 360 with repeated known blocks and 54 additional candidates with repeated missing BLOCK tokens. The field therefore means repetition within the same-or-unknown block and is a conservative audit flag, not proof of repeated identified physical plots. Replicate counts correctly exclude unknown blocks. This semantic distinction does not change means, outcomes, row membership, or source data.

The public candidate and plot artifacts exclude numeric milling variables and trial-wide yield statistics. All seven source-manifest SHA256 hashes, the four expected original Dryad file hashes, and all seven derived-manifest hashes were verified. The exact eight original/input artifact hashes checked for numerical reconstruction are recorded in the JSON audit.

Interpretive limits:
- PHR denominator is not explicit in the original archive; derived metadata correctly labels same-program support rather than archivewide confirmation.
- Yield moisture basis remains unspecified; no13%-to12% conversion is invented.
- The primary endpoint is product of all-observed marginal means; matched-row product and matched-row marginal means are separate sensitivity endpoints.
- Repeated within-block check records remain distinct source observations after removal of exact duplicate full rows; block counts are not inflated to observation counts.
- Equal reported HR values across rows do not identify independent physical milling assays; physical assay counts remain unknown.
- Stage EF is preserved, not silently converted to README FE; unknown/E5/E6 stages remain distinguishable.
- Source germplasm descriptors are retained separately from Arkansas market-class descriptions.
- The stored repetition flag includes 54 candidates with repeated missing BLOCK values; only 360 candidates repeat a known block. The flag is conservative and does not establish that unknown-block records are the same plot.

This audit contains structural counts, hashes, and check results only. It does not expose candidate milling values or evaluate policy performance.
