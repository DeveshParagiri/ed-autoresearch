# ED-Fire research charter

## Purpose

This file defines the scientific rules for ED-Fire autoresearch. It does not explain installation, duplicate the data catalog, store the current handoff, or tell an agent how to operate. Those responsibilities belong to `README.md`, `data/catalog.toml` and `data/sources.toml`, `memory.md`, and `skills/autoresearch/SKILL.md` respectively.

## Scientific objective

The project is trying to improve the fire process used by Ecosystem Demography while preserving a defensible causal account of each change. The immediate observable is monthly global burned area. Fire-carbon emissions and coupled vegetation–carbon effects belong to the broader objective, but they enter only when the candidate produces the required quantities and a separate evaluation contract has been approved.

A credible result must say what mechanism changed, predict where and when its effect should appear, compare the full evidence vector with the stock ED baseline, disclose regressions, survive a matched rerun, and distinguish scientific evidence from data, implementation, numerical, or optimization failure. A higher aggregate score alone is not sufficient.

Model A through E parameters, fitted values, masks, generated outputs, and expected scores are excluded from the clean research line. They may later be reconstructed as explicitly historical experiments, but they cannot seed the baseline or an undeclared candidate.

## Starting boundary

The only clean native ED baseline currently available is the pinned TRENDY v14 EDv3 S3 burned-area artifact declared as `benchmark.ed-stock.burned-area`. Its SHA-256 identity is locked in `evals/contracts/burned-area-v1.json`. This workspace does not contain the ED source revision, build configuration, input deck, or command that produced that artifact.

The first experiment therefore establishes an evaluation baseline, not source-level reproduction. It can prove that the same stored ED field yields the same recorded ILAMB evidence under a locked environment. It cannot prove that ED itself can be rebuilt or rerun. A mechanistic change requires either the relevant ED fire source and configuration under version control or a clean, reviewable implementation with the same candidate-output interface.

## Research target and candidate interface

Research changes the fire mechanism; orchestration code only records and evaluates those changes. In coupled ED, fire depends on climate, vegetation and fuel state, productivity, disturbance history, and any explicitly justified human or land-use drivers; fire then changes later vegetation and carbon state. Offline work holds the supplied state fixed so that equations, parameters, inputs, and diagnostics can be compared cheaply. Coupled work tests whether an offline result survives feedback into ED.

ED-Fire experiments use `execution.mode: mechanistic`. Direct runs and Optuna studies select execution tools. The generic scaffold also supports simulation-engine projects; running ED-Fire code on a computer does not change its mechanistic mode.

Every burned-area candidate evaluated under `contract.ed-fire.burned-area.v1` must produce `work/model/Candidate/burntArea.nc`. It must contain monthly `burntArea` on the global 0.5-degree, 360 × 720 grid from January 2001 through December 2016, with burned area represented as a fraction of grid-cell area per month. Candidate-specific files may remain in run-owned work or diagnostics, but official metrics and canonical figures are written only by the trusted evaluator.

## Inputs and provenance

An experiment may consume only assets named by catalog ID in its declaration. `data/catalog.toml` owns the stable project path, scientific role, requirement level, and integrity checks. `data/sources.toml` owns source identity, acquisition route, release, coverage, units, preprocessing, limitations, access terms, and recovery instructions. A file being installed does not make it scientifically active.

The stock evaluation consumes the native ED artifact, GFED5 burned area, GFED4.1s burned area, and the corresponding tracked ILAMB configurations. It does not need climate, ecosystem, coupled-state, human, or candidate-driver inputs because it stages an existing output rather than running a model.

The wider installed inventory supports later work: CRUJRA climate; ED and TRENDY ecosystem state; coupled state and fuel dumps; population and other human drivers; LUH2 land-use states and transitions; environmental candidate drivers; external TRENDY models; GFED source products; and pinned conversion tooling. A later experiment must state the mechanism, variable, temporal treatment, spatial treatment, expected sign, affected region or season, and rejection criterion before introducing any of these assets.

Large data remain outside Git and are exposed through stable project paths. A changed source, release, preprocessing route, calendar, grid, mask, or unit is a changed input even when the filename stays the same.

## Burned-area development evaluation

The active comparison family is `contract.ed-fire.burned-area.v1`. It uses ILAMB 2.7.3 over the common 2001–2016 period and retains two result vectors. GFED5 is the primary development benchmark. GFED4.1s is a separate benchmark-sensitivity check because the products differ in construction and small-fire treatment; it is not an interchangeable fallback.

The required global evidence for each benchmark is the model-period mean, bias score, RMSE score, seasonal-cycle score, spatial-distribution score, and overall score. The contract also records the candidate’s change in overall score between GFED5 and GFED4.1s. Regional diagnostics use the fixed global, Africa, South America, North America, Boreal Eurasia, Tropical and Southeast Asia, Australia, and Europe boxes declared in the contract.

The evaluation produces exactly five canonical 1800 × 1200 figures. `01-score-summary.png` shows global scalar evidence. `02-mean-burned-area.png` shows benchmark, stock, candidate, and residual fields. `03-seasonal-cycles.png` shows the fixed regional climatologies. `04-spatial-distribution.png` shows cell-level distributions and is explicitly a target-derived diagnostic rather than a promotion objective. `05-benchmark-sensitivity.png` shows score and regional sensitivity to the benchmark product. The contract fixes panels, periods, masks, labels, units, scales, baseline ordering, and filenames so every candidate is presented on the same basis.

GFED5 fire emissions are installed and have a tracked ILAMB configuration, but the stock artifact has no `fFire` variable. Emissions are therefore outside the active contract. They require a candidate emissions interface and a new approved contract rather than a silent extension of the burned-area score.

## Benchmark integrity and non-gaming rules

Candidate code may read only its declared model inputs and any development information explicitly allowed by its experiment. It writes only inside the run directory. It cannot write official metrics or canonical figures, change protected benchmark files, change the evaluator, alter comparison baselines, select a favorable period or region after seeing results, redefine masks, or replace failed evidence with candidate-authored summaries.

The recorder checks protected hashes before and after candidate execution. It deletes candidate-authored official metrics and canonical figures before invoking the evaluator. The evaluator and canonical plotting path are locked by hash in the active contract. Any change capable of altering ranking or presentation creates a new contract ID and requires human review.

Hash locking protects integrity, not secrecy. Repeated access to development metrics can still overfit the benchmark. This contract therefore cannot produce a promoted model. Promotion requires a separate evaluation whose data and outputs are unavailable to candidate code, search logic, and model selection until the candidate and decision rule are frozen.

Missing values, ocean cells, physical zeros, and unobserved cells must remain distinct. Any mask derived from GFED fire presence, residuals, or performance is target-derived and must be labeled as such. A scientific claim that depends on it requires a target-independent or otherwise defensible sensitivity analysis.

## Experiment record and branch memory

Only experiments form the research DAG. A parent edge means that the child directly extends, repairs, replicates, ablates, combines, or reopens an earlier experimental state. Datasets, benchmarks, hypotheses, metrics, figures, findings, and optimizer trials remain attached to experiments.

Before material execution, `experiment.md` records the question, isolated change, prediction, expected evidence, known failure modes, cost, stopping rule, and `Revisit when` condition. A run records one terminal attempt. Retries, repairs, replications, optimizer studies, interruptions, and invalid executions receive distinct run IDs so the history cannot be overwritten.

`memory.md` is only a compact pointer to the active frontier and next action. When it disagrees with an experiment record, run, contract, manifest, or Git, the durable evidence wins and memory must be repaired. A paused or negative branch remains recoverable through its recorded result and revisit condition. It is reopened only when that condition, or an equivalent new fact, is present.

The next experiment should answer the most decision-relevant unresolved question at the lowest adequate cost. A temporarily worse score can be useful when it identifies a missing process or interaction. A combined improvement requires later attribution and ablation; the combined score cannot establish which component caused it.

## Optuna policy

Optuna may tune a declared parameter space inside one fixed scientific experiment after a direct candidate and its evaluation reproduce. The experiment fixes the objective metric or metrics and directions, parameter-space implementation, sampler, pruner, random seed, trial or time budget, and selection rule before the first trial. The study is not allowed to choose the scientific mechanism or alter the evaluation family.

Every completed, pruned, and failed trial, parameter value, objective value, meaningful intermediate value, timing, sampler, pruner, study attribute, and selected parameter set must be exported. The selected trial must write the ordinary candidate output and pass through the same trusted evaluator and canonical figures as a direct run. Before any promotion claim, the selected parameters are pinned and replayed without adaptive search in a descendant confirmation experiment.

Only declared development objectives may reach the objective function, sampler, pruner, callbacks, or selection code. Promotion evidence is inaccessible. Pruning is allowed only when its intermediate measurement has a scientific and computational meaning. Multi-objective work uses no pruning unless the installed Optuna version and a reviewed project procedure support it.

## Failure interpretation

A failed terminal run is not automatically evidence against its hypothesis. The experiment interpretation must identify whether the failure came from missing or invalid data, candidate implementation, ED or simulator execution, numerical behavior, evaluation, optimization, insufficient or noisy evidence, benchmark sensitivity, or evidence against the proposed mechanism. Repairs create new runs; they do not erase the failed attempt.

A branch should stop when its declared budget or stopping rule is reached, its mechanism is contradicted under adequate controls, its improvement depends on prohibited leakage, or its remaining question requires unapproved data or compute. It may continue under mixed evidence when the disagreement is localized, reproducible, and scientifically informative.

## Cost and human approval

Manifest checks, metadata inspection, unit checks, and small diagnostic slices are the cheapest tier. Full global preprocessing, complete ILAMB evaluation, canonical figure generation, and bounded parameter studies form the middle tier. Coupled ED runs are the expensive tier and should answer a feedback or transfer question that stored state and offline work cannot resolve.

Human approval is required before changing the active evaluation family, headline region set, target mask, hard scientific constraint, promotion rule, ED integration target, project-wide research procedure, or publication claim. It is also required before an expensive coupled run. A coupled request must identify the unresolved question, matched offline evidence, exact configuration, required outputs, expected cost, and stopping rule.

## Research sequence

The first required control is a stock evaluation at one committed revision, followed by a matched repeat with the same candidate identity, metric vector, and canonical figures. Its measured result and completion state belong in the experiment record, while the current handoff belongs in `memory.md`.

After that control, the first decision is not “run Optuna.” The evidence must be inspected for benchmark-product sensitivity, mask behavior, regional failures, and evaluator consistency. The project must then obtain the relevant ED fire source and configuration or establish a clean mechanistic implementation before proposing an actual model change. The first such change should isolate one interpretable process and state its predicted regional and seasonal signature before execution.

Coupled transfer, fire emissions, new inputs, and parameter search become descendants only when their prerequisite interface and control exist. Model A through E remain outside this sequence.

## Open questions

The immediate unresolved issue is which version-controlled ED fire implementation will produce new candidate outputs. Once that is settled, the project must determine which stock regional failures persist under both GFED products, which available state variables correspond to those failures, which masks in earlier work contain target information, which emissions formulation links burned area and fuel consumption to GFED5 carbon flux, and which parameter ranges are scientifically defensible before any optimizer is allowed to explore them.
