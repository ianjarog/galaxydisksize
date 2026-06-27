# Contributing

Thanks for your interest in improving `galaxydisksize`. This is research software
that accompanies a publication, so the priorities are **reproducibility** and not
silently changing published results.

## Getting set up

```bash
conda env create -f environment.yml
conda activate galaxydisksize
pip install -e ".[dev]"
```

## Branching and commits

- `main` is the protected, releasable branch. Do not commit directly to it.
- Create a topic branch per change, named `type/short-description`, for example
  `fix/survival-median-edge-case`, `feat/odr-estimator`, `docs/readme-usage`.
- Write clear, imperative commit messages ("Add ...", "Fix ...", not "added").
- Open a pull request and fill in the template. CI must pass before merge.

## Quality gate

Run the same checks CI runs before pushing:

```bash
./quality-gate.sh
# or individually:
ruff format --check .     # formatting
ruff check .              # lint (library held to numpydoc docstrings)
pytest                    # unit tests + golden-number regression guards
```

Formatting and linting are both done with **ruff** (line length 100, numpy
docstring convention); do not introduce `black`, which would fight `ruff format`.

## Protecting the scientific results

The test suite pins the published numbers
(`tests/test_golden_numbers.py`). If your change is expected to alter a figure,
table, or number:

1. Say so explicitly in the pull request.
2. Update the affected golden-number tests in the same PR.
3. Explain and justify the change.

If your change is **not** expected to alter results, the golden-number tests must
still pass unchanged.

For workflow changes, verify the DAG still builds:

```bash
snakemake -n paper        # dry run: no rule may produce a file two ways
```

## Coding conventions

- The reusable science lives in `src/galaxydisksize/` and is held to the full
  rule set, including numpydoc docstrings.
- `scripts/` are thin command-line front-ends / analysis drivers; they are held
  to every code rule but the docstring-style rules are relaxed.
- Each workflow output is produced by **exactly one** rule. Never add a second
  rule (or script) that writes an existing output file.

## Security

Never commit credentials, tokens, or machine-specific absolute paths. Read
secrets from environment variables (see `.env.example`). See
[`SECURITY.md`](SECURITY.md) for reporting.
