# Event-Provenance RGAT for Sentinel incidents (PoC)

A relational graph attention network that classifies **events** (not
entities) as part-of-incident vs benign, trained on a provenance graph whose
edges are exactly the joins a KQL `make-graph` stage will produce later.

## Why events as nodes (the non-negotiable part)

Entity graphs (IP / host / account as nodes) make the model memorise: *port
8443 appeared in 3 incidents -> port 8443 is malicious forever*. With events
as nodes, the same port, host or account appears in thousands of benign
events too; the only signal left is **the shape and timing of the event
chain** — the incident topography. This design forces that:

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
| same RGAT, edges rewired (topology destroyed) | 0.2832 |     0.7845 |     0.3333 |

Read it bottom-up: destroy the graph structure (degrees preserved) and the
RGAT loses nearly two thirds of its AUPRC — chain topology does the heavy
lifting. The RGAT beats the features-only MLP by +0.22 AUPRC / +0.09 AUROC,
the comparison that proves the provenance graph adds signal beyond per-event
attributes.

**Operating points** of the single model (every threshold tuned on
validation only, then applied to test):

| policy            | recall | precision | missed | flagged events |
|-------------------|-------:|----------:|-------:|---------------:|
| F1-optimal        |  0.634 |     0.898 |    285 |     56 (0.09%) |
| recall ≥ 0.90     |  0.936 |     0.139 |     50 |  4,517 (7.4%)  |
| recall ≥ 0.93     |  0.958 |     0.108 |     33 |  6,175 (10.2%) |
| recall ≥ 0.95     |  0.976 |     0.085 |     19 |  8,171 (13.5%) |
| recall ≥ 0.97     |  0.986 |     0.063 |     11 | 11,517 (19.0%) |

Even in the most recall-hungry deployment the model flags at most ~1 in 5
events; at the 90% recall floor it catches 729/779 incidents by flagging
under 9% of events. An ensemble of 6 independently trained members
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
tests/                              focal loss, RGAT, sampler, generator, pipeline
```
