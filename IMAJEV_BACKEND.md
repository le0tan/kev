# Imajev on KEV — text-decision backend (experimental branch `imajev-text-adapter`)

This branch adds a backend that serves the Imajev text-decision protocol on KEV's Qwen3.5 torch
backend: the same prompts, candidate mapping, rotations and unknown/abstain semantics as the native
Imajev path, executed through KEV's merged+fused backbone with CUDA graphs and a state-prefix cache.
**Status: experimental — it is not a drop-in replacement for the official Imajev eval.** The observed
behavioral differences vs the native backend are listed below and must be reviewed before adoption.

Scope: **text only.** `/v1/score` whitelists `{state, questions[, rotations]}` and rejects any visual
input with 422 before the model sees it. Rotations are pinned to 1 and no calibration is applied in
everything reported here.

## What is on this branch (all relative to kev main @ `0c142be`)

| File | Role |
|---|---|
| `kev/imajev_adapter.py` | core adapter: encode (state-prefix grouping), load (bf16 base + fp32 LoRA, fused rewrite), score (fp32 readout → native `result_from_logits`/`combine_rotations` → `jev_api.to_response`) |
| `kev/imajev_serve.py` | standalone serving entry (`python -m kev.imajev_serve --base … --adapter … --port 8018`); mirrors `kev.serve`'s one-model-thread batching + prefix cache + OOM retry; isolated from default KEV serving |
| `kev/model.py` | additive only: `hidden_picks`/`hidden_picks_batch`/`hidden_picks_one` (hidden states at decide positions; no changes to existing forward paths) |
| `tests/imajev_adapter_checks.py` | CPU/GPU checks: `tokens`, `mapping`, `forward`, `long` (subcommands) |
| `tools/make_kev_compatible_adapter.py` | builds the KEV-loadable adapter copy from the original Imajev adapter |
| `tools/imajev_bench.py` | pool builder (`build-pool2`) + unified-window benchmark (`run --side native|kev`) with whole-pool warmup, in-window scoring, answers-out |
| `tools/imajev_answers_compare.py` | per-question (ridx, chunk, qid) answer comparison under the official v1.1 argmax |
| `tools/imajev_cross_audit.py` | independent cross-side re-audit calling the official `canonical_distribution` directly |
| `tools/imajev_rows_check.py` | CPU check of the packed-row encoding contract (site-specific default paths) |

`kev.serve` and all default KEV behavior are untouched; the imajev entry loads no kev checkpoint and
no head, and shares only the batching pattern and state-prefix cache.

## Correctness status (measured, not assumed)

- Adapter mapping verified tensor-for-tensor on the peft-wrapped tree (400 LoRA tensors + fp32 readout),
  tokens bit-identical (594-line check), forward parity on GPU: 0 prediction flips, 0 abstain flips on
  the 156-request serving-config sample (max candidate-prob diff 0.0581 / unknown-prob 0.0782 /
  unknown-logit 0.2504; graphs-vs-eager differences unchanged from kev's own graphed-vs-eager paths).
- **Full-pool audit vs native (10,003 questions, official `canonical_distribution` argmax):**
  - 16 known-class prediction flips — 6 boolean true↔false (margins 0.0024–0.027) and 10 choice
    `__NO_SUPPORTED_VALUE__`↔real-option swaps (u ≤ 0.04). All near-ties.
  - 79 abstain-boundary flips (exactly one side argmax-unknown): 54 native-abstain→kev-answers,
    24 the reverse; boundary distance |top_known − u| p50 0.011 / max 0.053.
  - The two groups are disjoint: **95 unique composite question keys (ridx, chunk, qid)** — raw qid
    names repeat across requests (only 76 unique names among the flips, 1,590 across the pool).
  - |Δunknown_probability| p50 0.0023 / max 0.1015; boolean served-true p50 0.0001 / max 0.0727;
    both sides' stored `abstained` flags agree with the official argmax on all 10,003 rows.
  - **These differences are not accepted by default**; this backend remains an experimental config
    until they are reviewed. Tooling note: raw qid strings repeat across requests — uniqueness must be
    judged on the composite key (ridx, chunk, qid), not the qid name.

## Recommended serving configuration (as measured on one 10,003-question pool)

- **C1 (serial request window) + prefix cache 4 + max_batch 64, all defaults otherwise** (bf16,
  fused+merged, CUDA graphs, rot1, no calibration).
- Unified window (forward + readout + answer assembly + D2H, whole-pool untimed warmup pass first,
  cold load / JIT excluded): kev 167.5 q/s vs native 27.2 q/s on that pool. **Both numbers are
  pool- and boundary-specific** — the two pools measured (2,126- and 10,003-question) gave 2.65× and
  6.2×; do not read either as a global figure. The difference is not decomposed (pool composition and
  execution organization differ on both sides).
- Prefix cache 16 was tested against 4 on the 10k pool: identical timed hits/misses (629/761 both),
  throughput within noise, bit-identical answers → 4 kept. That observation is limited to this pool's
  access pattern; it does not characterize the misses beyond "capacity 16 added no hits here".
- Higher concurrency (C4/8/16) was throughput-flat (±2%) with p50 latency 41→666 ms; C1 is the
  recommended operating point.
- Known harness observation: with C>1 the main-thread scoring time inflates (39–49 s vs C1 1.9 s)
  because its D2H syncs behind in-flight model-thread forwards; mechanism observed, not isolated.

## Reproducing (no new experiments needed; all scripts are on this branch)

```bash
export PYTHONPATH=<this repo>:$IMAEV_SRC_PARENT          # kev imports resolve to THIS repo
export IMAJEV_SRC=<imajev repo>/src                       # native imports (jev_api, torch_decision, …)
export IMAJEV_SCRIPTS=<imajev repo>/scripts               # official evaluate_text_decisions for the audit
PY=<venv with torch 2.8, transformers, peft, torchvision==0.23.0 (no-deps)>

# 1. adapter copy (READ-ONLY original; writes only the copy)
$PY tools/make_kev_compatible_adapter.py --src <original imajev adapter> --dst <kev-compatible copy>

# 2. CPU checks (tokens / mapping), GPU forward parity
$PY tests/imajev_adapter_checks.py tokens  --base <qwen3.5-4b> --adapter <kev copy> --adapter-orig <original>
$PY tests/imajev_adapter_checks.py mapping --adapter-orig <original> --adapter <kev-compatible copy>
$PY tests/imajev_adapter_checks.py forward --base <qwen3.5-4b> --adapter <kev-compatible copy>

# 3. build a reproducible pool (seeded, per-set, state-length terciles; sidecar meta with seed+sha)
$PY tools/imajev_bench.py build-pool2 --textile <dev set rows> --beauty <dev set rows> \
    --target 10000 --seed 20261001 --out pool.jsonl --meta pool.meta.json

# 4. unified-window benchmark with in-window scoring and full-pool warmup (kev C1 and native)
$PY tools/imajev_bench.py run --side kev --base <base> --adapter <copy> --pool pool.jsonl \
    --out kev_c1.json --answers-out kev_c1_answers.jsonl --concurrency 1 --warmup-pass
$PY tools/imajev_bench.py run --side native --base <base> --adapter-orig <original> --pool pool.jsonl \
    --out native.json --answers-out native_answers.jsonl --warmup-pass
#   kev-only knobs (defaults unchanged): --prefix-size (KEV_PREFIX_CACHE, default 4),
#   --max-batch (KEV_MAX_BATCH, default 64)

# 5. audits (pure CPU, official formulas)
$PY tools/imajev_answers_compare.py kev_c1_answers.jsonl native_answers.jsonl --labels kev,native
IMAEV_SCRIPTS=$IMAJEV_SCRIPTS $PY tools/imajev_cross_audit.py --pool pool.jsonl \
    --native native_answers.jsonl --kev kev_c1_answers.jsonl --out-dir audit_out
```

`tools/imajev_cross_audit.py` and `tools/imajev_rows_check.py` carry site-specific default paths as
fallbacks (override with env/args elsewhere). No model weights, datasets, answers or logs are included
in this branch; pools/answers/logs live outside the repo.

## Limitations

- Text-only: visual inputs are refused 422 (by design on this backend).
- `state` > 4096 tokens is a hard error on both sides (no fallback path exists on either).
- Long-state (>1024-token prefix) requests bypass CUDA graphs (eager prefix path) — covered and
  verified on the 3,506-token `long_state` case.
- kev-vs-native serving uses separate weight layouts (merged+fused bf16 vs unmerged fp32-LoRA); the
  measured probability differences above are the honest residual of that, not rounding noise.
