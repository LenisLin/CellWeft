# CellWeft collaboration rules

## Start here

1. Read `docs/project/status.md`, then the relevant specifications linked from `docs/README.md`.
2. Use `docs/project/research_plan.md` for research scope and `docs/project/decisions.md` for accepted decisions. Discussion drafts and archived prompts are historical context, not execution instructions.
3. Inspect relevant code and actual outputs before changing behavior or reporting completion.

## Scope and storage

- SSD workspace: `/home/lenislin/Experiment/projects/CellWeft`; code, configurations, tests and current research documents belong here.
- NAS workspace: `/mnt/NAS_21T/ProjectData/CellWeft`; datasets, manifests, protein banks, checkpoints, embeddings and generated results belong here. NAS is not a Git repository.
- Follow `docs/architecture/storage.md` for ownership and `configs/layout.json` for directories. Create directories as needed; do not create empty placeholder files.
- Planning, discussion and review are read-only unless implementation is requested. Preserve unrelated user work and original data.
- Keep actual patient records, private membership lists, credentials and runtime artifacts out of tracked files. Track selection rules and non-sensitive templates.
- `reference/` is entirely local and excluded from Git. Do not force-add it, including its PDFs, source checkouts or metadata files.

## Implementation and verification

- Keep reusable behavior in `src/cellweft/`; scripts orchestrate specific datasets or experiments. Source modules do not import scripts.
- Follow the current model, training, data and editing specifications linked from docs/README.md. The research plan owns scientific scope; archived prompts are not current implementation instructions.
- Define observable success criteria, make the smallest necessary change, and run the narrowest meaningful checks.
- Distinguish planned, implemented, tested, trained and biologically validated states. Report skipped or failed checks accurately.
- Update the owning specification for contract changes; record the rationale in `decisions.md`. Update `status.md` with verified progress and the next concrete step.

- Keep environment changes project-local; do not use sudo or change system/global environments for initialization.
- Initialization does not automatically download large datasets or weights, start full training, create automated tasks, or upload private data/embeddings. Trackers default to offline/disabled; report missing dependencies explicitly.
- Use Chinese for README and research documents, English for code identifiers and commit messages, and typed public interfaces with focused comments.
- Record bootstrap hyperparameters as provisional in the owning model/training specification and link the choice from decisions.md; do not treat them as accepted production settings.

## Git and GitHub

- Manage only the SSD project through `https://github.com/LenisLin/CellWeft` on `main`; preserve existing remote history.
- Use English Conventional Commit subjects: `type(scope): description`. Typical types: `feat`, `fix`, `docs`, `refactor`, `test`, `build`, `ci`, `chore`, `perf`.
- Write commit bodies in English. Within one commit, describe each distinct file batch separately, naming the paths or areas, the change and its purpose; include actual validation separately.
- Prefer separate commits for independent changes. A cohesive multi-file change may use one commit with distinct body sections.
- Inspect the staged diff and file list before committing. Never force-push, rewrite published history or upload excluded assets without explicit authorization.
- Follow `docs/development/workflow.md` for the commit template and publication procedure. Do not add new approval gates to routine work already authorized by the user.
