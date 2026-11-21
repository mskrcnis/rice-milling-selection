# Post hoc diagnostic: actions and independent headline reconciliation

These diagnostics describe the original frozen decisions. They do not change any model, cohort, budget, exclusion, shortlist target or primary comparison. The implementation independently reconstructed 4,500 saved policy events in all 250 primary pools, and 4,000 paired action contrasts. All reconstructed per-pool recall, best-entry recovery, regret, testing counts and fractions agree with the original event results; complete before/after untested ranks agree with the logged rank-change counts and second batches. The consumed source bytes were unchanged.

## Action evidence at the 25% design budget

For primary two-batch versus matched no-update, both programs have identical first batches and zero changes in untested ranks, second-batch order or membership, query order or membership, final-shortlist order or membership, and measured regret. This is observed lack of decision impact for these frozen policies, not equivalence of adaptive methods in general.

For primary two-batch versus locked static mixed model, all 188 Uruguay and 62 Arkansas pools have identical queried sets, final shortlists and regret. Query order differs in 19 Uruguay pools and one Arkansas pool. Therefore “identical actions” is too broad for the static comparison: use “identical queried sets and final shortlists.” The static rule has one batch, so its second-batch comparison is not applicable. Static mixed and yield-first have identical query order, queried sets, final shortlists and regret at 25% (and in the separately audited 40% comparison).

The secondary tree feedback changes untested ranks in 140/188 Uruguay and 26/62 Arkansas pools. Against its own matched no-update control, second-batch order differs in 23 and two pools, respectively, while membership differs in seven and one. The final shortlist and regret change in one Uruguay pool (an adverse change) and zero Arkansas pools. Thus six Uruguay pools and one Arkansas pool have different queried sets with equal regret. Equality of regret does not establish equality of choices.

Static trees and the no-update tree variant share the Arkansas improvement over the locked mixed model (−0.077672 percentage points in mean regret at 25%). Each differs in final-shortlist membership and regret from mixed in two pools; feedback contributes no additional 25% improvement. The static-versus-adaptive comparison does not isolate the value of first-batch answers.

At the secondary 40% budget, Arkansas tree feedback changes second-batch order in seven pools and membership in two, with changed final shortlist and regret in only one: ARPT__2024__FAU (28 candidates; k=3; B=12). Regret falls from 5.032966% to zero in that pool, yielding −0.081177 percentage points in the 62-pool mean. The other changed queried set has equal regret. This isolated secondary result does not replace the primary 25% finding.

The models' common public predictors are two transforms of already observed paddy yield: yield relative to the original pool mean and its within-pool percentile. Paddy yield also multiplies measured head-rice recovery in the scoring index. This overlap helps define the narrow decision problem; these diagnostics do not establish a biological cause for the observed cross-program difference.

## Headline values and denominators

| Primary or source-coverage group | Pools | Eligible entries | Mean recall (%) | Mean relative regret (%) | Equal-pool actual testing (%) | Tested entries / eligible entries |
|---|---:|---:|---:|---:|---:|---:|
| Uruguay 2017–2020 | 188 | 7,568 | 93.926915 | 0.226086 | 25.449852 | 1,917 / 7,568 |
| Arkansas 2017–2025 | 62 | 1,611 | 67.741935 | 2.597324 | 26.406006 | 425 / 1,611 |
| Arkansas 2017–2020 | 19 | 447 | 82.456140 | 0.758902 | 27.355068 | 122 / 447 |
| Arkansas 2021–2025 | 43 | 1,164 | 61.240310 | 3.409650 | 25.986653 | 303 / 1,164 |

Recall is the equal-pool mean of recovered reference top-k entries divided by that pool's k=ceil(0.10n). It is not the proportion of pools finding the single best entry: that outcome is 186/188 (98.936170%) in Uruguay and 43/62 (69.354839%) in Arkansas. Nor is recall the pooled fraction of all reference entries: 821/869 and 124/184 would weight larger shortlists more heavily.

The full-precision recall difference is 26.1849797994273 percentage points: 26.18 at two decimals or 26.2 at one. The requested 26.19 is obtained by subtracting already rounded 93.93 and 67.74. Use 26.2 percentage points for a concise abstract statement, with this reconciliation retained.

The first measured grid fractions attaining 90% mean recall remain 20% for Uruguay (91.6858% recall; actual equal-pool testing 21.4263%) and 60% for all Arkansas (95.1613% recall; actual testing 61.5724%). This is a grid crossing, not an estimated minimum continuously optimized budget. Actual testing is rounded separately within each pool; the equal-pool mean fraction differs from total queries divided by total candidates.

## Coverage interpretation

Uruguay's 188 evaluation pools have 7,662 original candidate rows, 7,596 observed-field candidates and 7,568 paired candidates; eight pools lack some milling answers. Arkansas's 62 primary pools have 1,722 original candidate rows, 1,715 observed-field candidates and 1,611 paired candidates. Its 19 historical-answer pools (2017–2020) contain 552 contemporary-list rows, 551 observed-field candidates and 447 paired candidates; all 19 are incomplete, with 104 observed-field entries lacking answers. Its 43 contemporaneous pools (2021–2025) contain 1,170 list rows and 1,164 observed-field candidates, all 1,164 with milling answers. The six other rows lack observed field yield and do not enter the query pool.

The cross-program headline remains directionally present in the fully observed contemporary Arkansas panel: 61.24% recall and 3.410% regret versus 93.93% and 0.226% in Uruguay. That comparison remains descriptive. The periods, recurrent locations, candidate composition, trial structures and programs differ jointly; complete answer coverage does not identify why their outcomes differ or make these a new independent confirmatory cohort.

## Suggested action-table caption

Post hoc audit of the frozen 25% policy decisions. Counts are pools with a change. Untested-rank changes compare each adaptive policy's own ranking before and after its paid first-batch update. Second-batch differences require identical first-batch order and two actual batches; a one-batch static comparator is marked not applicable. Query order, query membership and final-shortlist membership are assessed separately, and regret is independently recomputed. Zero regret difference alone is not evidence of identical choices. Primary and development-selected secondary comparisons remain separate.

## Reproduction

From the revision directory, with an interpreter containing the pinned original dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 /path/to/python src/action_diagnostic.py --source-project /path/to/original-or-bundled-study --output-dir outputs/actions
PYTHONDONTWRITEBYTECODE=1 /path/to/python src/breeder_figure.py --source-project /path/to/original-or-bundled-study --output-dir outputs/breeder
```

Inputs are named relative to the source project in each `verification.json`. Outputs must be outside that project. The code does not import original scientific modules and does not refit models. Full per-pool comparisons retain query and shortlist identifiers for direct auditing.
