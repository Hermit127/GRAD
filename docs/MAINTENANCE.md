# Maintenance and paper snapshots

## Initial publication

The first paper version is public at https://arxiv.org/abs/2610.22828v1 (submitted October 16, 2026). README and CITATION.cff now reference it. Publish the prepared code repository after the remaining source-permission review and repository setup. Once the repository URL is known, add it to the paper, citation metadata and release notes. Do not invent either identifier before assignment.

Create a proposed `arxiv-v1` tag only after verifying that the selected commit matches the actual first public paper version. Attach release notes with paper version, source commit, tested Python/dependency versions, reproduction commands, validation outputs, artifact checksums and remaining data requirements. Never move an existing paper tag to different code.

## Subsequent work

Use `main` for maintained code and short feature branches for changes. Preserve original research sources and hashes in the paper snapshot; if they need fixes, keep the original paper tag and document the fix on main with a new manifest and release note. New experimental candidates must state their development/validation status and must not replace published numerical evidence without a documented paper revision.

For arXiv v2, use a separate proposed `arxiv-v2` tag pointing to the matching commit. For code fixes independent of a paper update, use software releases such as `v0.1.1`. Repository tags and arXiv version numbers are separate mechanisms.

## Routine workflow

1. Triage incoming issues into bug, reproduction, documentation or research question.
2. Make a focused change and describe its effect on paper results.
3. Run release checks and regression tests; run model-specific validation when behavior changes.
4. Review the diff, record a changelog entry, and merge after CI succeeds.
5. For a release, generate a clean archive and verify it from a new directory.

CI performs offline checks on Python 3.11 and 3.12 and does not launch GPUs or API jobs. No recurring automation or response-time commitment is enabled by this document.

## Software license

The author selected MIT for original GRAD software and documentation on 2026-10-15. Keep LICENSE and upstream notices in releases. GitHub owner/name and public publication remain deferred by the author. The paper's arXiv license is a separate choice.

## Visibility gate — 2026-10-19

The maintainer requested a new private GitHub repository for review. Create it with private visibility, verify visibility after uploading, and keep it private until the maintainer explicitly confirms the public transition. Existing permissions to upload do not authorize making it public. Remote setup is awaiting an authenticated GitHub connection.
