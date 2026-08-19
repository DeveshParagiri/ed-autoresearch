# ED-Fire research charter

## Purpose

This file defines the scientific rules for ED-Fire autoresearch. It does not explain installation, duplicate the data catalog, store the current handoff, or tell an agent how to operate. Those responsibilities belong to `README.md`, `data/catalog.toml` and `data/sources.toml`, `memory.md`, and `skills/autoresearch/SKILL.md` respectively.

## Scientific objective

The project is trying to improve the fire process used by Ecosystem Demography while preserving a defensible causal account of each change. Its declared optimization goal is to find mechanisms and parameters that make ED reproduce GFED5 monthly burned area as well as possible under the fixed 2001–2016 ILAMB evaluation. GFED5 is visible to the research loop through trusted evaluator metrics and may be queried repeatedly during direct iteration or Optuna search. Fire-carbon emissions and coupled vegetation–carbon effects belong to the broader objective, but they enter only when the candidate produces the required quantities and a separate evaluation contract has been approved.

A candidate is ranked by its declared GFED5 objective. The result must also say what mechanism changed, predict where and when its effect should appear, compare the full evidence vector with stock ED, disclose component or regional regressions, survive a matched rerun, and distinguish model behavior from data, implementation, numerical, or optimization failure. Those diagnostics explain the score and expose tradeoffs; they do not replace the declared objective after results are known.

Models A through I, including the GFED4.1s and GFED5 versions of C and the exploratory Ibest hybrid, are this project’s existing model-development record. Their equations, parameter sets, optimizer artifacts, and results may motivate or initialize a new experiment. Their previously generated outputs do not become active-contract candidates automatically: any extension must declare what it reuses, remove prohibited benchmark-fed construction, regenerate the candidate from declared inputs, and pass through the trusted evaluator.

## Starting boundary

The clean native comparison baseline is the pinned TRENDY v14 EDv3 S3 burned-area artifact declared as `benchmark.ed-stock.burned-area`. Its SHA-256 identity is locked in both burned-area contracts. The completed stock runs retain `evals/contracts/burned-area-eval-v1.json`; future candidates use `evals/contracts/burned-area-eval-v2.json`. The workspace also contains the official ED v3.0 publisher source under `model/stock-ed/`, its dependency sources, official input bundle, global 1981–2016 simulation, environment guide, and evaluation script. Publisher sizes and digests are pinned.

Those two facts must not be conflated. The publisher archive has no Git metadata, and no recovered record connects its source revision, build, input selection, or command to the TRENDY v14 artifact. The completed first experiment therefore establishes an evaluation baseline, not source-level reproduction. The next technical control must compile and run the official release with its declared inputs, preserve the exact environment and command, and compare its output with both the publisher simulation and the pinned TRENDY baseline before any modified fire mechanism is treated as a descendant of stock ED.

## Models A-I

The source history, parameter artifacts, replay code, results, and figure tooling for Models A-I live directly under `model/`. They are part of the same research program. The reproduction experiment establishes what was tried, which values regenerate, and which limitations survived from the original implementation. A later scientific experiment can extend one of these models by naming it in the experiment rationale and implementing a candidate that satisfies the active contract.

The boundary is methodological, not genealogical. The A, B, and GFED4.1s C generator uses GFED4.1s fire presence as a spatial output mask and recomputes a scale factor from the GFED4.1s mean. Coupled Models C, D, F, and H use GFED5 fire presence to define generation or fitting support. Models E, G, G6, G7, I, and Ibest inherit a GFED4.1s fire-presence output mask. Model F also pins global burned-area magnitude directly to GFED5. Those existing outputs therefore do not satisfy the active clean-candidate contract. Ordinary parameter optimization against metrics returned by the trusted GFED5 evaluator remains valid; candidate-generation code itself may not consume benchmark fields or target statistics.

Model A remains a partial reconstruction because its exact frozen LAI preprocessing artifact and generated NetCDF were not committed. Model H reproduces all nonseasonal scores, but its seasonal phase calculation is sensitive to platform-level floating-point ties; that numerical drift is recorded rather than hidden by changing the evaluator. The registry and replay evidence own the exact values.

## Research target and candidate interface

Research changes the fire mechanism; orchestration code only records and evaluates those changes. In coupled ED, fire depends on climate, vegetation and fuel state, productivity, disturbance history, and any explicitly justified human or land-use drivers; fire then changes later vegetation and carbon state. Offline work holds the supplied state fixed so that equations, parameters, inputs, and diagnostics can be compared cheaply. Coupled work tests whether an offline result survives feedback into ED.

ED-Fire experiments use `execution.mode: mechanistic`. Direct runs and Optuna studies select execution tools. The generic scaffold also supports simulation-engine projects; running ED-Fire code on a computer does not change its mechanistic mode.

Every burned-area candidate evaluated under `contract.ed-fire.burned-area.v2` must produce `work/model/Candidate/burntArea.nc`. It must contain monthly `burntArea` on the global 0.5-degree, 360 × 720 grid from January 2001 through December 2016, with burned area represented as a fraction of grid-cell area per month. Candidate-specific files may remain in run-owned work or diagnostics, but official metrics and canonical figures are written only by the trusted evaluator.

## Inputs and provenance

An experiment may consume only assets named by catalog ID in its declaration. `data/catalog.toml` owns the stable project path, scientific role, requirement level, and integrity checks. `data/sources.toml` owns source identity, acquisition route, release, coverage, units, preprocessing, limitations, access terms, and recovery instructions. A file being installed does not make it scientifically active.

The stock evaluation consumes the native ED artifact, GFED5 burned area, GFED4.1s burned area, and the corresponding tracked ILAMB configurations. It does not need climate, ecosystem, coupled-state, human, or candidate-driver inputs because it stages an existing output rather than running a model.

The wider installed inventory supports later work: CRUJRA climate; ED and TRENDY ecosystem state; coupled state and fuel dumps; population and other human drivers; LUH2 land-use states and transitions; environmental candidate drivers; external TRENDY models; GFED source products; and pinned conversion tooling. A later experiment must state the mechanism, variable, temporal treatment, spatial treatment, expected sign, affected region or season, and rejection criterion before introducing any of these assets.

Large data remain outside Git but live at canonical paths inside this project. Older workspaces may point here by symlink. A changed source, release, preprocessing route, calendar, grid, mask, or unit is a changed input even when the filename stays the same.

## Burned-area development evaluation

The active comparison family for future candidates is `contract.ed-fire.burned-area.v2`. It preserves the v1 ILAMB 2.7.3 evaluation over the common 2001–2016 period and retains the same two result vectors. The primary optimization objective is `candidate.gfed5.overall_score`, with higher values preferred. Optuna and direct model selection may use that full-period score repeatedly. GFED4.1s is a benchmark-sensitivity check because the products differ in construction and small-fire treatment; it is not an interchangeable fallback or a hidden test.

The required global evidence for each benchmark is the model-period mean, bias score, RMSE score, seasonal-cycle score, spatial-distribution score, and overall score. The contract also records the candidate’s change in overall score between GFED5 and GFED4.1s. Regional diagnostics use the fixed global, Africa, South America, North America, Boreal Eurasia, Tropical and Southeast Asia, Australia, and Europe boxes declared in the contract.

The evaluation produces six canonical images. `01-score-summary.png` compares the five ILAMB score components and fixed-scale global means for candidate and stock ED against both GFED products. `02a-mean-burned-area.png` places the GFED4.1s and GFED5 benchmarks above stock ED and the candidate; all four maps share a colorbar directly below them. `02b-burned-area-differences.png` places candidate-minus-GFED4.1s beside candidate-minus-GFED5 and gives that pair its own diverging colorbar. `03-seasonal-cycles.png` compares monthly climatologies for the fixed global and seven regional domains, using a locked scale for each named region across every candidate. `04-spatial-distribution.png` contains four cell-density comparisons formed by crossing candidate and stock ED with GFED5 and GFED4.1s; its one-to-one line shows exact cell-level agreement and each panel reports how much valid data falls inside the fixed display window. `05-benchmark-sensitivity.png` shows how benchmark choice changes each ILAMB component, the candidate's regional bias against both products, and the candidate's signed regional change from stock ED without implying that more burning is better. The absolute-field image is 1800 × 1200, the difference image is 1800 × 800, and the other four images are 1800 × 1200. The contract fixes panels, periods, masks, labels, units, scales, baseline ordering, filenames, dimensions, renderer, and fonts so every candidate is presented on the same basis. The v2 presentation limits are 0–80 percent per year for absolute fields and cell-density axes, ±60 percent per year for residual maps, ±0.15 for score differences, ±25 percent per year for regional differences, and 0–1 percent per month for global means. The region-specific seasonal limits are declared in the contract. These limits do not alter the ILAMB evaluation.

GFED5 fire emissions are installed and have a tracked ILAMB configuration, but the stock artifact has no `fFire` variable. Emissions are therefore outside the active contract. They require a candidate emissions interface and a new approved contract rather than a silent extension of the burned-area score.

## Benchmark integrity and non-gaming rules

Candidate code may read only its declared model inputs and any development information explicitly allowed by its experiment. It writes only inside the run directory. It cannot write official metrics or canonical figures, change protected benchmark files, change the evaluator, alter comparison baselines, select a favorable period or region after seeing results, redefine masks, or replace failed evidence with candidate-authored summaries.

The recorder checks protected hashes before and after candidate execution. It deletes candidate-authored official metrics and canonical figures before invoking the evaluator. The evaluator and canonical plotting path are locked by hash in the active contract. Any change capable of altering ranking or presentation creates a new contract ID and requires human review.

GFED5 is intentionally visible as the optimization target. Candidate selection and Optuna may repeatedly use its trusted scalar metrics, including the Overall score, over the full contract period. Results under this contract are claims about reproducible fit to GFED5, with GFED4.1s sensitivity reported alongside them. Candidate ranking under the current objective uses no holdout. A future claim about unseen years, unseen regions, or predictive transfer would require a separately declared evaluation designed for that claim.

Missing values, ocean cells, physical zeros, and unobserved cells must remain distinct. Trusted evaluation may compare a candidate with GFED fields and return declared metrics and diagnostics. Candidate-generation code may not read GFED grid-cell values, fire-presence masks, residuals, or period means to construct, mask, rescale, or correct its output. Fixed parameters selected from trusted GFED5 metrics must still regenerate the candidate from declared non-benchmark model inputs.

## Experiment record and branch memory

A framing records one durable research question, its scope, current position, and reopening boundary. It appears as a labeled graph container only when an experiment explicitly names it. Framings are optional. Unframed experiments remain at the graph root, and ordinary parent relationships need no framing record.

Only experiments form the research DAG. A parent edge means that the child directly extends, repairs, replicates, ablates, combines, or reopens an earlier experimental state, including edges that cross framing containers. Datasets, benchmarks, hypotheses, metrics, figures, findings, and optimizer trials remain attached to experiments.

Before material execution, `experiment.md` records the question, the rationale for choosing it now, isolated change, prediction, expected evidence, known failure modes, cost, stopping rule, and `Revisit when` condition. It includes a `framing` reference only when the experiment belongs to an existing framing. A run records one terminal attempt. Retries, repairs, replications, optimizer studies, interruptions, and invalid executions receive distinct run IDs so the history cannot be overwritten.

The experiment document is the complete human research record. Before a result exists, its Rationale explains why the experiment was chosen, and its Plan explains why it is ready or still running. After a result exists, its Result states what happened, its Evidence section embeds or links the selected figures, metric record, and scientific outputs, and its Interpretation explains why the evidence supports the Decision. The viewer renders this document. It does not infer importance from filenames, scan `work/`, or substitute its own summary for the experiment record.

`memory.md` is only a compact pointer to the active frontier and next action. When it disagrees with an experiment record, run, contract, manifest, or Git, the durable evidence wins and memory must be repaired. A paused or negative branch remains recoverable through its recorded result and revisit condition. It is reopened only when that condition, or an equivalent new fact, is present.

Meeting transcripts, Granola exports, and private meeting records remain under the ignored boundary defined in `research/meetings/README.md`. They provide researcher context but do not become graph nodes or a second source of scientific state. Once feedback is confirmed, its consequence must be written into the experiment, framing, charter, memory, or evaluation contract that owns it. Public experiment rationale must not depend on access to a private meeting file.

The next experiment should answer the most decision-relevant unresolved question at the lowest adequate cost. A temporarily worse score can be useful when it identifies a missing process or interaction. A combined improvement requires later attribution and ablation; the combined score cannot establish which component caused it.

## Optuna policy

Optuna may tune a declared parameter space inside one fixed scientific experiment after a direct candidate and its evaluation reproduce. Under the active burned-area contract it may call the trusted evaluator for every trial and maximize the full-period GFED5 Overall score. The experiment fixes the objective metric or metrics and directions, parameter-space implementation, sampler, pruner, random seed, trial or time budget, and selection rule before the first trial. The study is not allowed to choose the scientific mechanism or alter the evaluation family.

Every completed, pruned, and failed trial, parameter value, objective value, meaningful intermediate value, timing, sampler, pruner, study attribute, and selected parameter set must be exported. The selected trial must write the ordinary candidate output and pass through the same trusted evaluator and canonical figures as a direct run. Its parameters are then pinned and replayed without adaptive search in a descendant confirmation experiment. That confirmation proves reproducibility; it is not described as independent validation because it uses the same evaluation.

Only declared development objectives may reach the objective function, sampler, pruner, callbacks, or selection code. The search may receive trusted GFED5 metric values but not protected benchmark arrays or candidate-authored substitutes for official metrics. Pruning is allowed only when its intermediate measurement has a scientific and computational meaning. Multi-objective work uses no pruning unless the installed Optuna version and a reviewed project procedure support it.

## Failure interpretation

A failed terminal run is not automatically evidence against its hypothesis. The experiment interpretation must identify whether the failure came from missing or invalid data, candidate implementation, ED or simulator execution, numerical behavior, evaluation, optimization, insufficient or noisy evidence, benchmark sensitivity, or evidence against the proposed mechanism. Repairs create new runs; they do not erase the failed attempt.

A branch should stop when its declared budget or stopping rule is reached, its mechanism is contradicted under adequate controls, its improvement depends on prohibited leakage, or its remaining question requires unapproved data or compute. It may continue under mixed evidence when the disagreement is localized, reproducible, and scientifically informative.

## Cost and human approval

Manifest checks, metadata inspection, unit checks, and small diagnostic slices are the cheapest tier. Full global preprocessing, complete ILAMB evaluation, canonical figure generation, and bounded parameter studies form the middle tier. Coupled ED runs are the expensive tier and should answer a feedback or transfer question that stored state and offline work cannot resolve.

Human approval is required before changing the active evaluation family, headline region set, target mask, hard scientific constraint, promotion rule, ED integration target, project-wide research procedure, or publication claim. It is also required before an expensive coupled run. A coupled request must identify the unresolved question, matched offline evidence, exact configuration, required outputs, expected cost, and stopping rule.

## Research sequence

The first required control is a stock evaluation at one committed revision, followed by a matched repeat with the same candidate identity, metric vector, and canonical figures. Its measured result and completion state belong in the experiment record, while the current handoff belongs in `memory.md`.

After that control, the first decision is not “run Optuna.” The evidence must be inspected for benchmark-product sensitivity, mask behavior, regional failures, and evaluator consistency. The official ED release must then pass a source-level build-and-run control, or its mismatch with the publisher and TRENDY outputs must be characterized explicitly. Only then should the first model change isolate one interpretable process and state its predicted regional and seasonal signature before execution.

Coupled transfer, fire emissions, new inputs, and parameter search become experiment descendants only when their prerequisite interface and control exist. Models A through I remain available as prior mechanisms, parameters, and evidence, but extending one requires a declared experiment and a clean regeneration path rather than treating its existing output as a new result.

## Open questions

The immediate unresolved issue is whether the official ED v3.0 release can be built and run reproducibly in a controlled environment and how its output relates to the publisher simulation and pinned TRENDY v14 baseline. The project must then determine which stock regional failures persist under both GFED products, which available state variables correspond to those failures, which masks in earlier work contain target information, which emissions formulation links burned area and fuel consumption to GFED5 carbon flux, and which parameter ranges are scientifically defensible before any optimizer is allowed to explore them.
