# Data

The Git checkout includes numeric result exports under `evidence/`; it does not include benchmark text, operation graphs, upstream catalogs or generated attack traces.

The original supplementary package contains W1 splits and selected artifacts. Public redistribution of those inputs is pending provenance and permission confirmation. Full CAA training additionally requires MCP/MRT task data that are not present even in that selected supplementary package. Do not describe the public repository as a complete data release.

The W1 input builder is `code/project/code/repro/w1_fair_comparison/build_obs1.py`. It consumes the derived Decomp source fold under `code/project/code/repro/out/unified_gnn/fix2_splits/decomp/fold0/` and produces the observation-matched inputs. Access to an upstream benchmark alone does not recreate all historical graph conversions or their splits.

When authorized inputs are available, preserve their split identifiers and record source URL, source license/terms, preprocessing version, label definition, file SHA-256, and any overlap audit. Put local files under ignored data directories; do not commit them by accident. Contact the maintainers for current distribution status; no public download URL has been assigned.
