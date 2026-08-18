# Bidirectional RGAT for Sentinel incidents (PoC)

A relational graph attention network that classifies **events** (not
entities) as part-of-incident vs benign, trained on a provenance graph whose
edges are exactly the joins a KQL `make-graph` stage will produce later.
The intended eventual use is for Threat Hunting (with adjustments), and primarily as an additional tool in incident automation.  
It is not meant for real-time triage as bidirectional sampling means seeds see their causal future.  
Beyond a PoC, for incident automation in Sentinel, the use-case and validity of the bidirectional sampling should be tested against real data to verify that the incident-starting event and its timestamp, provided by the SIEM during incident creation, leads to correct identification of kill chain events.  

The KQL side should be highly optimized, striving for completeness of data necessary for the graph at the lowest computational effort possible with in-built safeguards for node explosions.  

The model is not meant to recognize malicious events in incidents involving novel tactics and techniques.  

A production version of the model usable by a SOC with the least amount of adjustments to the PoC would require that the analysts or the automation tools marked the events that justify classifications, e.g. logic apps employing KQL queries as part of the automation process, analyst reports provided to the relevant stakeholders containing a section with events and timelines identified, etc. that can be used as part of the training data to improve and tune the topology knowledge of that particular organization.  

The synthetic data used for the training strives to be akin to what a mature SOC would log as part of an incident report and what Sentinel's detection rules display as evidence.  

A SOC classifying incidents without proof/evidence justifying the decision (particularly of high severity ones) would not have the quality of data necessary to conduct tests without modest data engineering effort. 

## Why events as nodes (the non-negotiable part)

Entity graphs (IP / host / account as nodes) make the model memorise: *port
8443 appeared in 3 incidents -> port 8443 is malicious forever*. With events
as nodes, the same port, host or account appears in thousands of benign
events too; the only signal left is **the shape and timing of the event
chain** — the incident topology. This design forces that:

* node features contain **no unique identifiers** at all (no IPs, hostnames,
  account names, literal command lines) — only shared-vocabulary categories
  and per-event attributes (`schema.py`). Windowed behavioural statistics
  (burst counts, host-switch counts, ...) are deliberately **not** features:
  that would flatten topology into per-event numbers and hand the answer to
  any flat model. The RGAT must learn behaviour from the provenance edges;
* a **leakage audit** computes the single-feature AUC of every feature on
  the train split and aborts the run if any feature nearly separates the
  classes (generator cheating is caught automatically);
* benign traffic deliberately contains attack-*shaped* hard negatives:
  admins doing lateral management with encoded PowerShell, service accounts
  beaconing periodically and moving gigabytes, deployment/AV waves that look
  like worm propagation;
* attack start times are uniform over hour-of-day and day-of-week.
* an **MLP feature-only baseline** is trained on the same features with no
  edges: if the RGAT does not clearly beat it, the graph is not contributing
  and something is leaking. An **edge-rewire ablation** (destinations
  permuted, degrees preserved) shows the score collapse when topology is
  destroyed.

## What is implemented

| Requirement              | Where                                                        |
|--------------------------|--------------------------------------------------------------|
| RGAT (relational GAT)    | `src/eprgat/rgat.py` — per-relation projections & attention, multi-head, within-relation softmax, temporal edge-feature attention bias |
| Focal loss               | `src/eprgat/losses.py` — stable-from-logits, gamma + alpha   |
| Neighbourhood sampling   | `src/eprgat/sampling.py` — self-contained k-hop CSR sampler (no torch-sparse), per-relation fanouts, balanced pos/neg seeds, **bidirectional** (`sampling.reverse_edges: true`): seeds also see their causal future, with those edges emitted reversed so consequences send messages back up the chain |
| Training objective       | focal loss over the **whole sampled subgraph** (GraphSAINT-style): edges never cross splits, so every sampled node around a train seed is itself a train node — chain fragments are supervised end-to-end, not just the seed rows |
| Readout                  | head consumes `[graph embedding ‖ feature embedding]` (fusion skip), so per-event attribute signal does not have to survive four message-passing rounds; any gain over the features-only MLP is attributable to the graph |
| Extreme imbalance (~1%)  | balanced seed batches + focal alpha + threshold tuned on val; operating-point policy is a knob (`train.threshold_policy: f1 \| recall`), and evaluation always reports BOTH operating points |
| Temporal train/val/test  | `src/eprgat/graph.py` — 65/17/18 with edge-dead gap zones    |
| Full evaluation          | F1 (tuned + @0.5), P/R, MCC, AUROC, AUPRC, precision@K, confusion, PR & triage curves |
| KQL readiness            | `src/eprgat/schema.py` + `docs/kql_contract.md` — the exact tables/columns/joins the KQL stage must emit |

## Quick start

```powershell
pip install torch numpy scikit-learn pyyaml matplotlib
pip install torch_geometric     # HeteroData container + utils only; no C++ kernels needed

# sanity run on CPU (minutes)
python train.py --config configs/smoke.yaml

# the real thing on your GPU (8 GB VRAM sized)
python train.py --config configs/full.yaml

# re-evaluate / ablations only
python evaluate.py --run runs/<your_run>

# visualize the test results on localhost (which events flared up)
python tools/export_predictions.py runs/<your_run>
python tools/serve_viz.py                # -> http://localhost:8123
```

The neighbourhood sampler is self-contained (`sampling.py`, NumPy CSR +
torch indexing), so the optional `torch-sparse` / `pyg-lib` kernels — which
rarely have wheels for the newest Python/torch builds — are **not** required.

Any knob can be overridden: `python train.py --config configs/full.yaml
--set sampling.batch_seeds=256 --set train.epochs=20`.

## Watching it train

Logging is deliberately verbose: per-step focal loss with pos/neg
decomposition, seed F1, subgraph size, LR and GPU memory; per-epoch summary
boxes with val AUROC/AUPRC/F1/P/R/MCC, early-stopping patience and best-model
tracking. Everything also lands in `runs/<...>/train.log` and
`metrics.jsonl`; `curves.png` and `test_pr_curves.png` are written at the
end, plus `eval_report.json`.

## Visualizing test results on localhost

To see *which events flared up* — per-event scores, flagged vs missed
incident chains, false alarms — export a run's test predictions and serve
them locally:

```powershell
python tools/export_predictions.py runs/<your_run>   # writes runs/<your_run>/viz_data.json
python tools/serve_viz.py                            # http://localhost:8123
```

The export re-runs the exact training-time evaluation (val-tuned threshold,
never fitted on test) and joins each held-out test event with the incident
metadata of the run's regenerated world. The dashboard (stdlib-only server,
no external JS) shows:

* **score-vs-time timeline** of all test events: flares above the threshold
  line, ground-truth incident windows shaded; drag to zoom, click a dot to
  open its incident;
* **threshold slider** between the val-tuned F1 and recall operating points —
  every panel recomputes live (flagged/missed counts, per-incident catch);
* **per-incident catch bars** and a **chain drill-down**: each incident's
  events as nodes (swimlanes per host) with the provenance edges that connect
  them, coloured by flagged/missed;
* **flagged-event table** (what an analyst would be handed, TP/FP split,
  filterable) and the **missed positives** table.

## 8 GB VRAM: OOM knobs (in priority order)

1. `sampling.batch_seeds`: 512 -> 256
2. `sampling.fanouts`: e.g. `[6,4,2]` -> `[4,2,2]` (fewer neighbours per hop)
3. `sampling.max_frontier`: 20000 -> 10000 (cap newly sampled nodes per hop)
4. `sampling.eval_batch_seeds`: 1024 -> 512
5. `model.hidden_dim`: 96 -> 64
6. `train.amp: true` (fp16 autocast)

Memory math: a batch of 512 seeds with the default fanouts materialises a
subgraph of roughly 10k-40k nodes / 50k-200k edges; attention tensors are
`E x heads x head_dim`, i.e. a few hundred MB at hidden 96.

## Honest-result guide

With the non-cheating generator you should expect *middling* absolute scores
that are nonetheless far above baseline — that is the point:

* **AUPRC many times the baseline AP** (baseline = positive rate ~1.3%) with a
  clear gap between RGAT and the MLP baseline = topology is being learned.
* Rewire ablation dropping AUPRC substantially = same confirmation.
* F1 around 0.4-0.7 on test at ~1% positives is realistic; F1 of 0.99 would
  mean the generator is leaking and should be treated as a bug, not a win.

### What the final full run actually produced (RTX 5060, 8 GB)

Dataset: 28 days, 402k events, 155 incidents, 0.86/1.41/1.28% positives in
train/val/test (temporal split, edge-dead gap zones). Leakage audit top
single-feature AUC stays far below the 0.95 abort threshold.

| model                                   | test AUPRC | test AUROC | test F1 (val-tuned thr) |
|-----------------------------------------|-----------:|-----------:|-----------:|
| baseline (predict positive rate)        |     0.0128 |          – |          – |
| MLP, features only, no edges            |     0.5358 |     0.8885 |     0.5363 |
| **event-provenance RGAT (this repo)**   | **0.7544** | **0.9823** | **0.7434** |
| same RGAT, *evaluated* with edges rewired (sensitivity check) | 0.2832 |     0.7845 |     0.3333 |

Read it bottom-up: destroy the graph structure (degrees preserved) and the
RGAT loses nearly two thirds of its AUPRC. The RGAT beats the features-only
MLP by +0.22 AUPRC / +0.09 AUROC — the graph adds signal beyond per-event
attributes.

#### Retraining control (kills the distribution-shift objection)

The rewired row above evaluates the *trained* model on a destroyed graph —
that shows dependence on connectivity, but conflates "topology carries
signal" with "the model breaks under structural shift". The proper control
trains from scratch on the destroyed graph (`tools/train_rewired.py`):
destinations permuted **within each split** (degrees preserved, no
cross-split edges, edge dt recomputed for the new pairs), threshold tuned on
the rewired val and applied to the rewired test — this model's train and
eval conditions match, so its score cannot be blamed on shift.

Matched-recipe pair (canonical config; 30 epochs and
`sampling.max_frontier=2000`, which caps the frontier explosion a random
graph causes at sampling time):

| arm (same recipe)                       | trained on | test AUPRC | test AUROC | test F1 |
|-----------------------------------------|------------|-----------:|-----------:|--------:|
| RGAT, intact graph                      | intact     |     0.7164 |     0.9771 |  0.7069 |
| RGAT, rewired graph (control)           | rewired    |     0.4453 |     0.8737 |  0.4828 |

(`runs/20260809_203905_intact_fast`, `runs/20260809_203653_rewire_retrain`;
full-budget versions pending — minutes-scale now that the frontier is
capped.) The only difference between the two runs is whether edges carry
real incident structure: the **+0.27 AUPRC** gap is what true connectivity
contributes. Random connectivity is worth *less than none*: the
rewired-trained model lands below the features-only MLP (0.4453 < 0.5358) —
scrambled-neighbour messages are noise, and the model's gain requires real
event structure.

**Operating points** of the single model (every threshold tuned on
validation only, then applied to test):

Flagged events = TP + FP, i.e. everything an analyst would have to look at.

| policy            | recall | precision | missed | flagged events |
|-------------------|-------:|----------:|-------:|---------------:|
| F1-optimal        |  0.634 |     0.898 |    285 |    550 (0.91%) |
| recall ≥ 0.90     |  0.936 |     0.139 |     50 |  5,246 (8.6%)  |
| recall ≥ 0.93     |  0.958 |     0.108 |     33 |  6,921 (11.4%) |
| recall ≥ 0.95     |  0.976 |     0.085 |     19 |  8,931 (14.7%) |
| recall ≥ 0.97     |  0.986 |     0.063 |     11 | 12,285 (20.3%) |

Even in the most recall-hungry deployment the model flags at most ~1 in 5
events; at the 90% recall floor it catches 729 of the 779 positive events
by flagging under 9% of events. An ensemble of 6 independently trained members
(3 seeds × 2 feature-encoder depths, `tools/ensemble_eval.py`) scores
AUPRC 0.7478 with slightly smoother scores; the single model above was
selected by validation AUPRC and is the canonical artifact.

#### How the design got here (ablations, all on this dataset)

* **Bidirectional sampling is the default** — a structural audit
  (`tools/diagnose_topology.py`) showed the graph signal is clean but sparse:
  when a positive event has a PROCESS_CHILD/SESSION_ACTION in-neighbour it is
  positive 100% of the time, and only ~1% of benign events have any positive
  in their 4-hop past — but **43% of positives have no positive ancestor at
  all within 4 hops**: chain-start events carry their evidence only
  *downstream*. Sampling the causal future too (edges emitted reversed,
  direction exposed to attention as an extra edge feature) lifted test AUPRC
  0.669 → 0.703 on its own.
* **Depth/width**: 6 layers + hidden 128 → 0.730; 8 layers do not beat 6;
  dropping HOST/USER_SEQUENCE *costs* ~0.13 AUPRC (the dense relations carry
  real signal alongside the precise causal ones).
* **JK readout + 2-layer feature encoder** (`model.readout: jk`,
  `model.deep_input_proj: true`) → 0.754. Error analysis
  (`tools/analyze_errors.py`) showed the remaining misses concentrate in the
  deliberately feature-overlapping templates (beacon/valid-account abuse) and
  in events topologically isolated from their incident — the deeper feature
  encoder is exactly the pathway those isolated events rely on.
* **Not the bottleneck**: doubling the timeline (56 days / 310 incidents)
  reproduced the same relative lift (~57× baseline AP) — more data did not
  change ranking quality; K-pass neighbourhood averaging at eval time made no
  difference (sampling noise was already negligible at these fanouts).

## Known blind spots, and the event×entity hybrid (prototype)

The bidirectional RGAT works, but error analysis pins its residual misses to two
deliberately-hard templates. On the canonical full run, per-template **miss rate
at the F1 operating point** is:

| template | miss @ F1 pt | miss @ recall pt (R≈0.93) |
|---|---:|---:|
| T2_exploit_web | 15% | 0.7% |
| T5_service_lateral | 10% | 1.6% |
| T1_phishing | 32% | 8.6% |
| **T4_beacon_persist** | **76%** | **19.4%** |
| **T3_valid_account** | **93%** | **12.7%** |

At a precision-focused threshold the model is functionally blind to beaconing
and valid-account abuse. That is structural, not noise: those templates overlap
benign traffic by design (service accounts *do* beacon; admins *do* use valid
accounts laterally), and the anti-cheat rules withhold exactly the temporal /
entity-level statistics that would separate them. Misses also pile up at chain
**ends** (positions 0–0.25 and 0.75–1.0), where an event has evidence on only
one side.

### The question the hybrid must answer

The per-event model scores each event from its local provenance neighbourhood.
The blind templates are distinguishable by **entity-level behaviour over time**
(an account behaving against its own baseline, a host beaconing on a clock),
which no single event can see. So: does entity behaviour carry compromise signal,
and does it generalise — or is an entity-as-nodes model just the memorisation
trap this repo was built to avoid?

### Go/no-go: entity behaviour generalises to unseen entities

`tools/entity_probe.py` builds **behavioural** entity windows (host and user:
cadence, fan-out, event mix, burstiness, payload regularity, coarse role context
— no identity features) and scores every window with a model that **never saw
that entity** (entity-holdout, out-of-fold). Result on the full world:

| window | host AUPRC (unseen) | user AUPRC (unseen) | lift vs baseline |
|---|---:|---:|---:|
| 1h | 0.50 | 0.56 | ×19–25 |
| 2h | 0.63 | 0.66 | ×25–29 |
| 4h | 0.70 | 0.70 | ×27–30 |
| 8h | **0.78** | **0.86** | **×27–37** |

AUPRC 0.78–0.86 on entities never trained on, 25–37× the positive-rate baseline,
stable across window sizes. **This is behaviour, not identity** — the
memorisation objection does not apply to a behavioural entity branch. One caveat
surfaced immediately: windows the RGAT *catches* score ~0.98 on the entity
channel while windows it *misses* score ~0.52, so the entity channel is best at
*confirming*; a naive OR-gate flags ~99% of events and is worse than the RGAT.
The signal must be applied **selectively** via fusion.

### Gated fusion: recall-vs-investigation-cost

`tools/fusion_probe.py` joins the RGAT's per-event score with the entity score
(multi-scale windows, ~2h + ~8h) and measures, on the **test** split, how many
events an analyst must review to reach a given recall. The combiner is fit only
on **validation**, and a strictly **temporal** variant (entity branch trained on
the train period alone) performs as well or better — no future data anywhere.
Flagged % below is the optimal-ranking operating point (top-k events):

| method | test AUPRC | @ R=0.90 | @ R=0.93 | @ R=0.95 | @ R=0.97 |
|---|---:|---:|---:|---:|---:|
| RGAT-only | 0.762 | 5.7% | 8.4% | 11.3% | 15.0% |
| **fusion — logistic combiner** | **~0.93** | **~1.4%** | **~1.5%** | **~1.7%** | **~3.5%** |
| **fusion — GBM combiner** | **~0.95** | **~1.3%** | **~1.5%** | **~1.7%** | **~2.0%** |

The logistic row is the honest floor (no GBM, no model selection). The gain is
genuine complementarity, not an artifact of the fitted combiner: even the
**fit-free noisy-OR** variant beats the RGAT at every recall level (AUPRC ≈0.81),
the fusion is stable across seeds, the winning combiner is identical under
validation- and test-selection (zero selection bias), and the result reproduces
on a second, weaker checkpoint. It reaches **95% recall at ~1.7% flagged** where
the current model needs ~11%. The whole pipeline was adversarially audited for
temporal / entity-identity / label leakage — clean.

Per-template, at matched recall the max-efficiency fusion **fixes beacon**
(≈+6–7 recall points at R=0.93–0.95 — the entity periodicity signal landing
exactly as predicted) but **valid-account is the residual floor**: the
max-efficiency combiner drops it ~10–13 recall points *below* the RGAT, because
the fusion reaches the same overall recall with ~6× fewer flags and spends that
budget on high-confidence events, and valid-account events score low on **both**
channels. That is a deliberate, tunable trade — not a bug — as the sequence
branch shows.

### Sequence branch: ordering for valid-account

The window-aggregate branch loses event ORDER, which is exactly what separates a
remote-sign-in → encoded-process → discovery → exfil chain from ordinary admin
work. `tools/sequence_probe.py` runs a **bidirectional GRU with attention
pooling** over each entity's *ordered* event stream (same temporal-honest
protocol), trained with focal loss. Attention lets it focus on the suspicious
sub-sequence instead of averaging it away. It is a strong window classifier
(host AUPRC ~0.91, stable across seeds) and — unlike the long-window aggregate —
does not sacrifice valid-account. It is weaker on beacon than the long-window
aggregate, so the two branches are complementary, not interchangeable.

Fusing the RGAT score with the aggregate + sequence channels
(`tools/dump_scores.py` persists channel scores once; `tools/score_search.py`
searches combiners) gives a Pareto frontier — pick where to sit (per-template
recall at overall recall 0.93):

| operating point | combiner | AUPRC | flags @ R=0.93 | valid-account | beacon |
|---|---|---:|---:|---:|---:|
| RGAT-only | — | 0.76 | 8.4% | 86.5% | 80.6% |
| max efficiency | agg120+agg480+seq, GBM | 0.95 | 1.5% | 77% (−10) | 83% (+3) |
| **balanced (recommended)** | agg120+seq, GBM | 0.93 | 1.6% | 82% (−4) | 81% (≈RGAT) |
| valid-account-preserving | agg120+seq, logistic | 0.92 | 2.1% | 86% (≈RGAT) | 75% (−6) |

So: **90–95% recall at ~1.5–2% flagged** (vs the RGAT's 8.4%). The lever is
clean and now well-characterised: the **agg480** (long-window) channel is what
boosts beacon but costs valid-account; the **seq** channel is what preserves
valid-account. The **balanced** `agg120+seq` GBM point is the sweet spot — ~5×
more efficient than the RGAT while keeping *both* valid-account and beacon near
their RGAT levels. If valid-account matters most, the logistic
`agg120+seq` point holds it at its RGAT level (86%) at 2.1% flagged, at the cost
of beacon dropping below RGAT. No single operating point beats RGAT on *both*
valid-account and beacon simultaneously while staying this cheap — the
beacon/valid-account trade-off is real, because the two templates lean on
different entity signals. Valid-account never exceeds its RGAT level under tight
flag budgets: it is the genuinely-ambiguous, engineered-to-overlap template, and
lifting it further (e.g. richer sequence features or a template-aware combiner)
is the open problem.

**Architecture this implies** (drop-in, no RGAT retraining): keep the RGAT event
scorer; add entity behavioural branches (host + user, aggregate + sequence,
multi-scale windows, train-period→forward, no identity); fuse with a thin
combiner into one operating score. All probes are analysis-only tools; re-fit
the combiner and re-run the entity-holdout ablation on real data before trusting
it in deployment.

## Layout

```
configs/{smoke,medium,full}.yaml    presets
src/eprgat/schema.py                event/relation/feature contract (KQL target)
src/eprgat/synthetic.py             Sentinel-like event generator
src/eprgat/graph.py                 edges, splits, leakage audit, HeteroData
src/eprgat/rgat.py                  RGAT layer
src/eprgat/model.py                 classifier stack + MLP baseline
src/eprgat/losses.py                focal loss
src/eprgat/sampling.py              balanced neighbourhood sampling
src/eprgat/metrics.py               F1/AUROC/AUPRC/MCC/P@K
src/eprgat/trainer.py               verbose training loop
src/eprgat/evaluate.py              test eval + rewire ablation + plots
docs/kql_contract.md                what the future KQL stage must emit
tools/diagnose_topology.py          read-only structural audit: reachability of
                                    positives, label mixing, sampler coverage
tools/analyze_errors.py             joins a run's test predictions with incident
                                    templates/chain positions — who gets missed
tools/ensemble_eval.py              averaged scores of N same-dataset RGAT runs
                                    (threshold re-tuned on val, rewire included)
tools/ensemble_recall_table.py      recall-floor vs investigation-cost table
tools/train_rewired.py              retraining control: train from scratch on the
                                    within-split rewired graph (ablation done right)
tools/export_predictions.py         per-event test predictions + incident metadata
                                    -> runs/<run>/viz_data.json (analysis-only)
tools/serve_viz.py + tools/viz/     localhost dashboard: which events flared up,
                                    per-incident catch, provenance chain drill-down
tools/entity_probe.py               entity-holdout go/no-go: does entity BEHAVIOUR
                                    (not identity) classify compromise on unseen
                                    entities? (multi-feature windows)
tools/fusion_probe.py               event×entity gated fusion: recall-vs-flagged
                                    curve vs RGAT-only, temporal-honest, per-template
tools/sequence_probe.py             GRU over each entity's ORDERED event stream —
                                    the valid-account-preserving branch
tools/dump_scores.py                persist RGAT + aggregate + sequence channel
                                    scores (val/test) once, for fast combiner search
tools/score_search.py               combiner search (logistic/GBM, channel subsets):
                                    recall-vs-flagged + per-template frontier
tests/                              focal loss, RGAT, sampler, generator, pipeline,
                                    viz export/server
```
### Next steps

To make it easier to understand where the model is working correctly and where failing I will build a result explorer that allows users to select incidents and display events it correctly identified, misclassified, or missed. 
The following images might be outdated:

<img width="1840" height="784" alt="image" src="https://github.com/user-attachments/assets/86ced672-5431-44e9-af34-971a4a671af1" />

<img width="1213" height="499" alt="image" src="https://github.com/user-attachments/assets/16815146-0fff-481f-a77b-03d9ca9572e8" />

<img width="1255" height="840" alt="image" src="https://github.com/user-attachments/assets/286b2dd1-bd35-4cd0-999e-3815a398c938" />

If I succeed at improving the quality of data and it in turn leads to better results, eventually I will explore a few hybrid options to see if it would help with the edge cases it struggles to identify consistently.
Before adding new elements or changing the design drastically, I would like to verify and confirm whether the missed events are due to model design or data engineering issues. 
