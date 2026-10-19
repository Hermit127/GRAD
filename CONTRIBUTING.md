# Contributing

Please open an issue for bugs or reproduction questions. Include the code commit or release tag, Python and dependency versions, command, expected result, actual result and a minimal traceback. Remove API keys and private inputs before posting.

Keep changes focused and explain whether reported results change. Historical method candidates and paper snapshots must remain distinguishable from new research. Do not overwrite saved reference results merely to make a check pass.

Before a pull request:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

If a frozen research source requires a fix, document it in CHANGELOG.md and retain the old paper tag. Updating frozen hashes requires an explicit reviewed explanation. New third-party code needs its source, version and license notice; dataset and checkpoint publication needs a documented distribution decision.
