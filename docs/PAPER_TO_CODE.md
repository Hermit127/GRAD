# Paper-to-code map

All paths are relative to the repository root. Run original scripts with `cd code` followed by `python run.py <path below project/>`.

| Paper component | Original implementation | Public reanalysis / required inputs |
| --- | --- | --- |
| SSUM / G3-512 | `code/project/code/repro/final_five_tables/train_grad_repair.py`, `train_gnn_ablate2.py` | Authorized graph splits required |
| W1 matched decision-time detection | `code/project/code/repro/w1_fair_comparison/` | `scripts/reproduce.py --task w1`; numeric score export included |
| W4 frozen thresholds and paired statistics | `code/project/code/repro/w1_final_confirmation/final_stats.py` | Thresholds and archived W4 statistics in `evidence/w1/`; quick check covers AUROC and operating points |
| Encoder comparisons | `code/project/code/repro/final_five_tables/train_gnn_variant_repair.py` | Corresponding graph splits and training runs required |
| CAA snapshots and centroid routing | `code/project/code/repro/real_cl_g3.py` | Full stream data required |
| CAA endpoint and family weighting | `code/project/code/repro/caa_attr_20260923/` | Complete endpoint artifacts required |
| T3 closed-loop evaluation | `code/project/code/repro/final_table4/` | `scripts/reproduce.py --task t3`; 240 numeric outcomes included |
| New generated-case checks | `code/project/code/repro/core_review_resolution/` | Generation, scoring inputs and provider access required |
| Historical components and diagnostics | `code/project/code/repro/static_big_table_v1/`, `final_table2/`, `task1_ssum_relation/` | Historical scope; not all candidates appear in the current paper |

`SUBMISSION_CODE_INDEX.md` preserves the earlier supplementary mapping. Its artifact paths describe the larger submission package, not files promised to exist in this public checkout. Third-party `project/ref/` material, raw graphs, model binaries and post-paper iteration experiments are excluded.
