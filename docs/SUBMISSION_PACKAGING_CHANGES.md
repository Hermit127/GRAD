# Packaging changes

- Preserved the original `project/code/` layout and local imports.
- Replaced author-machine project roots with `.`. Use `run.py`, which selects the
  portable project root and sets `GRAD_BASE`.
- Replaced machine/user identifiers in source comments and selected text artifacts
  with anonymous labels. Experiment identifiers and numerical results are retained.
- Added environment-variable credentials to the main closed-loop budget client;
  its original reviewer-supplied key-file fallback is retained for offline fixture
  tests. No key files, API secrets, environment files or cloud configurations are included.
- Annotated the existing fixed credential-shaped benchmark strings as synthetic
  fixtures; they are not live credentials.
- Anonymized checkpoint string metadata and reserialized checkpoints. Every tensor
  was compared with its original and verified exactly equal. File SHA-256 values
  therefore differ; source and packaged hashes are both in source_manifest.json.
- Requirements, launch/validation scripts and reviewer documentation are packaging
  additions. No training objective, model tensor, reported numerical value or
  statistical formula was changed for packaging.

Historical plan/code hashes describe the original execution. Use the manifest to
map original-file hashes to the anonymized release; do not expect anonymized files
to match historical whole-file hashes.
