# Reproducibility

## What runs from this checkout

| Task | Command | Dependencies | Validation |
| --- | --- | --- | --- |
| Frozen-source integrity | `python scripts/verify_release.py` | Standard library | 326 original Python sources match their recorded hashes; syntax and release boundaries |
| T3 reaggregation | `python scripts/reproduce.py --task t3` | Standard library | 240 outcomes, six cells, paired comparisons, per-family counts |
| W1 score check | `python scripts/reproduce.py --task w1` | `requirements-analysis.txt` | Nine runs, 232 examples, per-seed/mean/std AUROC, dev-frozen operating points |
| Complete training / inference | Research entry points below | Models and authorized inputs | Not rerun as part of release preparation |

All public checks are offline. Expected statistics are compared to recomputed values with numerical tolerance. Outputs go to `outputs/`; supplied evidence is unchanged.

## Evidence transformations

`docs/evidence_export.json` records source paths and export decisions. T3 exports identifiers, condition, backbone, defense, outcome flags and boolean error indicators; it excludes action arguments, prompts and response traces. W1 exports labels and scores and preserves their order. Test group IDs are renamed in sorted order, preserving grouping and group ordering; task IDs are replaced by row indices. Expected W4 statistics and frozen thresholds come from the archived package. Original frozen research source is unchanged.

These numeric exports support result reanalysis. They do not constitute raw-data release, model inference, attack re-execution, or complete end-to-end reproduction. W1 bootstrap intervals are not recomputed by the public quick check.

## W1 training

Obtain the authorized W1 decision-time graph splits described in `data/README.md`. Place them at `code/project/LOGS/w1_fair_comparison/data/obs1/{train,dev,test}.jsonl`. This input path is ignored by Git.

```bash
python -m pip install -r requirements-models.txt
cd code
python run.py code/repro/w1_fair_comparison/run_grad.py \
  --train LOGS/w1_fair_comparison/data/obs1/train.jsonl \
  --val LOGS/w1_fair_comparison/data/obs1/dev.jsonl \
  --test LOGS/w1_fair_comparison/data/obs1/test.jsonl \
  --candidate A_moddrop --lambda-aux 0.1 --moddrop 0.25 \
  --edge-mode rebuilt --edge-dropout 0 --edge-text copy \
  --epochs 60 --patience 15 --batch-size 48 --hidden-dim 192 \
  --lr 0.001 --seed 211 --max-node-chars 512 --max-edge-chars 160 \
  --target-fpr 0.1 --hard-benign-weight 3 --hard-benign-top-k 128 \
  --ablate none --out-dir LOGS/w1_fair_comparison/runs_local/GRAD_s211 \
  --save-checkpoint LOGS/w1_fair_comparison/runs_local/GRAD_s211/model.pt
```

This command transcribes the recorded GRAD seed-211 configuration with a fresh output directory; it has not been retrained during repository preparation. Repeat with seeds 212 and 213 for the three-seed comparison. W1 decision-time input construction also applies a 200-character cap inherited from its source conversion; a trainer limit of 512 does not restore discarded text. Do not mix W1 inputs and the original static Decomp/MCP/MRT protocol.

## CAA and other experiments

`code/project/code/repro/real_cl_g3.py` implements the G3 snapshot comparison. Its complete MCP/MRT task inputs and expert checkpoints are outside this checkout. Review the original stream configuration and provide the inputs described in `docs/SUBMISSION_DATA_AND_DEPENDENCIES.md` before running it. A command being present is not evidence that all required data are bundled.

`code/project/code/repro/final_table4/` contains the gate, transfer training and agent runner. Historical integrations require their upstream runtimes. The old canonicalization tests depend on the omitted upstream shopping catalog and are not the public release CI tests.

## Online experiments

Install `requirements-online.txt` only when you intend to run online experiments. The main budget client supports `DEEPSEEK_API_KEY` and `GLM_API_KEY` environment variables; `.env.example` documents their names and is not automatically loaded. Legacy scripts may use their historical key-file interface; inspect those entry points before using them. The public checks never invoke online runners. Record model IDs returned by providers, budget, task panel and code commit for each new run.

## Environment and verification

See `docs/VALIDATION.md` for checks actually run for this public repository. `docs/SUBMISSION_VALIDATION.md` is a historical report about a larger supplementary package, not a claim that its omitted artifacts have been executed here.
