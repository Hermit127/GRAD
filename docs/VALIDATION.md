# Public repository validation — 2026-10-15

## Executed locally

- Verified all 326 frozen research Python files against SHA-256 hashes of the submitted package; no numerical or method source changes were made.
- Parsed the original research sources, launcher, new reproduction scripts and regression tests as Python.
- Reaggregated 240 distinct T3 outcomes using the original aggregator. All six cells, paired comparisons and per-family counts matched the archived output at tolerance 1e-10.
- Recomputed W1 AUROC for nine runs on 232 examples and checked development-frozen thresholds and test operating points against archived W4 values. Mean AUROCs: GRAD 0.9699597040560178; Text-only 0.8635552913198573; Temporal 0.9169474170960497.
- All four public regression tests passed: T3 archived-result comparison, W1 archived-result comparison, output protection and launcher path confinement.
- Parsed workflow YAML and CITATION.cff and checked local Markdown links.
- Inspected credential-pattern matches without printing secret values. Two provider-shaped strings were the same pre-annotated synthetic test fixture in original and historical-copy source. No live credentials were identified by these checks; pattern scanning is not an exhaustive security audit.
- Compared the method illustration to the final paper figure by generating the README image from the paper's current vector PDF.

Analysis checks used Python 3.12.14, NumPy 2.3.5, scikit-learn 1.9.1, SciPy 1.18.1, joblib 1.6.0 and threadpoolctl 3.7.0. Version ranges in requirements support broader environments, but those additional combinations have not been locally tested. No analysis packages were installed into the user's existing environment during this task.

## Not executed

Model training, neural inference, CAA expert acquisition, bootstrap-interval regeneration, online API experiments and upstream runtime integration were not run for this repository release. GitHub-hosted CI has been configured for Python 3.11 and 3.12 but has not yet run remotely. The historical larger supplement validation is preserved separately in SUBMISSION_VALIDATION.md.

## Follow-up verification — 2026-10-19

Confirmed the public arXiv landing page for 2610.22828v1 and updated the README and preferred paper citation. Frozen-source integrity verification and all four local regression tests passed again. No model training or online API experiment was run. No GitHub remote, commit, release or paper tag has been created. Third-party per-file permission review and full artifact distribution remain unresolved.
