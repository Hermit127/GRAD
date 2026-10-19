# Paper-to-code index

Paths below are relative to `code/`. Run scripts using `python run.py` and the
path beneath `project/`, or change directory to `project/` and set `GRAD_BASE`
to that directory. Reported versions remain separated.

| Paper component | Entry points | Evidence |
|---|---|---|
| SSUM / GRAD-512 | `project/code/repro/final_five_tables/train_grad_repair.py`, `train_gnn_ablate2.py`, `train_gnn_ablate.py`; `project/code/train_gnn_v2.py` | `project/LOGS/final_five_tables/table1_g3/`; original G3-512 and GRAD-192 versions are distinct |
| W1 matched-observation detection | `project/code/repro/w1_fair_comparison/build_obs1.py`, `run_grad.py`, `run_variant.py`, `run_temporal.py`, `aggregate.py` | `project/LOGS/w1_fair_comparison/`, including nine run outputs, checkpoints, data and per-example predictions |
| W4 paired statistics | `project/code/repro/w1_final_confirmation/final_stats.py --target test` | `project/LOGS/w1_final_confirmation/stats_test.json`; other confirmation targets are separate experiments |
| Encoder comparisons | `project/code/repro/final_five_tables/train_gnn_variant_repair.py`, `core_ablation_stats.py`, `grad512_operating_points.py` | Relevant archived summaries under `project/LOGS/final_five_tables/table5/` |
| Controlled dependencies | `project/code/repro/task1_ssum_relation/`: `cfgen.py`, `cfgen2.py`, `graphbuild.py`, `train_eval.py`, `online_loop.py`, `aggregate_online.py` | Five-seed scripted outputs in `project/code/repro/out/task1_ssum_relation/server_online_20260918/` |
| CAA snapshots / routing | `project/code/repro/real_cl_g3.py` and imported `real_cl_*` modules | `project/LOGS/real_cl_g3_20260921/` |
| CAA equal-family / endpoint recheck | `project/code/repro/caa_attr_20260923/recheck_g3_dmm_endpoints.py`, `family_level_recheck.py` | `project/LOGS/real_cl_caa_attribution_20260923/g3_dmm_endpoint_recheck/` |
| Closed-loop gate / transfer | `project/code/repro/final_table4/`: `runner.py`, `g3_gate.py`, `real_cl_g3_gate_transfer_joint_train.py`, `aggregate.py`, `report_g3_closed_loop.py` | `project/LOGS/final_table4_g3trans/` (T3) and `project/LOGS/final_table4_g3/` (earlier batch) |
| AgentDojo P1/R1 diagnostics | `project/code/repro/final_table2/`: `build_confirmation_set.py`, `aggregate_confirmation.py`, `build_attack_repro.py` | `project/LOGS/final_table2/confirmation_batch1_20260919/` and `attack_repro_20260919/` |
| Historical adapted baselines | `project/code/repro/static_big_table_v1/snapshot/`, `fidelity_fix_20260916/`, `neurotaint_v2/`, and imported helper modules | These are the adapted components described in the appendix, not interchangeable full upstream defenses |

The older standalone scripts retained as imports are dependency code. Their defaults
and alternative candidates do not redefine the settings of the reported experiments.

## New generated-case and relation-attribution checks

`project/code/repro/core_review_resolution/` contains the A/B generation, validation, scoring and statistics scripts. `project/LOGS/core_review_resolution/` contains reports, frozen manifests, A/B statistics and per-example predictions. These checks do not establish an advantage over Temporal on new generated cases or over matched rewiring; see the new validation appendix. Complete generation traces and upstream score inputs are not included in this selected-evidence package.

The earlier instrumented controlled-relation study is retained as historical source in this package but is no longer a section of the current manuscript.
