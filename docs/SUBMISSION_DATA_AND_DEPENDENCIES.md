# Inputs and dependency coverage

## Included

- W1 decision-time train/dev/test graphs and their source Decomp fold; all nine
  final run result JSONs, nine inference checkpoints, per-example predictions,
  W1 summary and W4 frozen thresholds/statistics.
- Three closed-loop gate checkpoints (historical, G3 and T3) with string metadata
  anonymized; tensors verified exactly equal to their archived originals.
- Both 240-run closed-loop result panels, summaries, plans and available provenance.
- The small shopping catalog used by offline environment tests and the original
  DecompositionAttack monitoring prompt module.
- P1/R1 diagnostic artifacts explicitly cited by the appendix.
- Selected CAA, G3 and controlled-relation summaries and five-seed scripted outputs.

## Full retraining and optional upstream baselines

This is a code and selected-evidence release, not a mirror of the full laboratory
workspace. Full CAA and legacy-baseline retraining requires the original MCP/MRT/MG
source graphs and task-stream inputs referenced by each loader. CAA's complete
per-example run archive and expert checkpoints are not included. Their aggregate
results and the exact source scripts are included; do not describe a summary-only
check as a complete rerun.

The controlled generators are included and can generate their own scenarios.
AgentDojo, DRIFT, AgentTrust, MemoryGraft/MetaGPT, and other full upstream runtimes
are optional external dependencies for historical integrations. Install the
respective upstream project and data before invoking those integrations. The
core GRAD model uses PyTorch, PyTorch Geometric, NumPy and scikit-learn.

Additional historical scripts import `datasets`, `sentence_transformers`,
`xgboost`, `matplotlib`, `jsonschema`, `PyYAML`, `anthropic`, `together`, or Google
client libraries. They are not prerequisites for the offline aggregation examples
in README.md. Machine-specific distributed shell launchers were not packaged;
use the Python entry points on the reviewer's own devices.

## Third-party material

`project/ref/DecompositionAttack/src/utils/prompts.py` is an upstream prompt module
used by the sequential-monitor implementation. Its upstream README is retained.
`project/ref/MINJA/rap/webshop/data/items_shuffle_1000.json` is the upstream shopping
catalog used by the simulated environment. Attribution follows MINJA and WebShop.
Derived Decomp graphs originate from DecomposedHarm as cited in the paper. These
materials retain their upstream provenance; this package does not grant new rights
or relicense them. Full upstream repositories and credentials are not bundled.
