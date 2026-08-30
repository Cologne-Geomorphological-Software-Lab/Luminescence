# Contributing to pyluminescence

## Welcome

Welcome to the pyluminescence Contributing Guide. pyluminescence is a Python port of
the R package [`Luminescence`](https://github.com/R-Lum/Luminescence) for luminescence
dating data analysis. It is developed as a fork of the original
R-Lum/Luminescence repository.

The R sources (`R/`, `man/`, `tests/testthat/`) remain in the repository throughout the
migration and serve as the reference implementation and numerical oracle. See
[`README.md`](README.md) for the phase plan and current status.

## The porting principle

The single most important rule in this repository: the Python package mirrors the R
package. We are translating, not redesigning.

For every ported function:

* Its behaviour, defaults, argument semantics, edge cases, error/warning conditions,
  and known quirks must match the corresponding R function, unless a quirk is a
  genuine bug that has been discussed and explicitly documented as a deliberate
  deviation.
* Naming follows the fixed translation scheme in
  [`docs/r-migration.md`](docs/r-migration.md) (`analyse_SAR.CWOSL()` becomes
  `analyse_sar_cwosl()`, `RLum.Data.Curve` becomes `Curve`, etc.). Do not invent a
  different name for a ported function or argument.
* Numeric output is validated against the R implementation, not against what "looks
  right"; see *Porting workflow* below.

The one agreed exception is the plotting layer (`luminescence.plot`, Phase 5): it is an
idiomatic matplotlib redesign, not a faithful reproduction of R's base-graphics output.
Every other module mirrors R.

If you're unsure whether something counts as a deviation, open an issue before writing
code.

## Contributions we accept

* **Porting**
  * Implementing an R function in Python. Check the phase table in `README.md` and
    the existing files in [`tools/specs/`](tools/specs/) to see what's already
    ported, specified, or in progress.
* **Port specs**
  * Extracting a `tools/specs/*.md` write-up from an R source file (signature,
    behaviour, edge cases, quirks). Valuable on its own, even without an
    implementation attached.
* **Bug reports**
  * Numeric results that don't match the R oracle
  * Crashes or behaviour that silently diverges from R
* **Tests**
  * Parity tests against R fixtures/snapshots
  * Unit tests for edge cases
* **Documentation**
  * Docstrings
  * `docs/r-migration.md` updates, MkDocs pages

At this time, we do not accept:

* Behavioural changes that diverge from the R implementation without a documented,
  discussed reason (see *The porting principle* and *Porting workflow*).
* New third-party runtime dependencies without prior agreement.
* Pull requests targeting the upstream `R-Lum/Luminescence` repository (this fork
  does not merge back upstream). All work stays in
  `Cologne-Geomorphological-Software-Lab/Luminescence`.
* Swapping the toolchain (uv/hatchling, ruff, basedpyright) for an alternative.

## Ground rules

* Be respectful in all written communication: issues, pull requests, and commit
  messages. You pledge to abide by our [code of conduct][coco].
* Open an issue before starting a port of a non-trivial R function, so scope and
  approach can be agreed on and duplicate work is avoided.
* One logical change per pull request: one function or module per port, in line with
  the granular commit convention below. Do not bundle unrelated ports or fixes.
* All new behaviour must be covered by tests; every ported function needs a parity
  test against the R oracle, not just a unit test of the Python code in isolation.

## AI usage

Using AI tools to help write code is allowed and welcomed. It does not change any of
the standards in this guide: judge the code on its merits, not on how it was produced.

* **Numerical and behavioural choices are not the AI's to decide.** When the R source
  is ambiguous, undocumented, or contains a quirk, go read `R/` directly or consult
  `tools/specs/`. Don't let an AI tool invent a "cleaner" Pythonic behaviour that
  diverges from R. Any genuine behavioural choice (not already fixed by the R source)
  needs a discussion first, per *The porting principle*.
* **Verify against the R oracle, not by assuming it works.** Run the relevant parity
  test (fixtures under `tests/python/fixtures/` or the `testthat` snapshots via
  `tests/python/oracle.py`) before calling a port done. A plausible-looking number is
  not the same as a checked one.
* Every AI-generated change needs a human to actually read it before it is committed.
  Do not submit code you have not reviewed and understood yourself.
* AI-generated code follows the same rules as any other code, see *Best practices*
  below, especially "no speculative abstractions": prefer the simplest translation of
  the R function over a generated one that handles cases the R source doesn't.
* **AI is well-suited to writing and keeping docstrings up to date**, provided it
  produces this project's actual style (see *Coding style*), not generic filler: no
  filler openers, no `Args:` lines that just restate the type hint, no `Raises:` entry
  for an exception the function doesn't actually raise.
* Install the pre-commit hooks (`uv run pre-commit install`). They run `ruff` (lint
  and format), `basedpyright`, `bandit` (security), and `vulture` (dead code) on
  touched Python files before each commit, plus basic file hygiene (trailing
  whitespace, merge-conflict markers, private-key detection) scoped away from the R
  package sources. A complexity gate (`xenon`) and a duplicate-code check (`pylint`)
  are configured but run manually only, since several ported functions inherit R's
  branch-heavy logic and would otherwise block unrelated commits; see the comments in
  `.pre-commit-config.yaml`.

## Issue management

Issues are tracked in the GitHub issue tracker.

When filing a bug:

1. State the expected behaviour (what the R function returns/does) and the actual
   Python behaviour.
2. Name the R function and, if known, the affected Python file and line number.
3. Provide a minimal reproduction, ideally comparing R and Python output on the same
   input (`inst/extdata` / `tests/testthat/_data` sample files are usually available
   on both sides).

When filing a port/feature request:

1. Name the R function(s) to be ported and their location under `R/`.
2. Note whether a `tools/specs/` write-up already exists.
3. Note any prerequisites (other unported functions it depends on, new dependencies).

## Environment setup

1. Clone the repository and install [uv](https://docs.astral.sh/uv/):

   ```bash
   git clone https://github.com/Cologne-Geomorphological-Software-Lab/Luminescence.git
   cd Luminescence
   uv sync --group dev
   ```

2. Run the test suite:

   ```bash
   uv run pytest
   ```

3. (Optional, only needed to regenerate R fixtures) Install R ≥ 4.6 with the
   `Luminescence` and `jsonlite` packages, then run `tools/generate_fixtures.R` with
   `Rscript`.

## Best practices

* **No unnecessary comments.** Only add a comment when the *why* is non-obvious: a
  hidden constraint, a subtle invariant, or a workaround for a specific bug. Do not
  describe what the code does.
* **No speculative abstractions.** Three similar lines are better than a premature
  helper. Only generalise when there are three or more concrete call sites.
* **No broad exception handling.** Catch only the specific exception types that can
  actually occur. Never use bare `except Exception`.
* **Validate at system boundaries only.** Trust numpy/scipy/pandas guarantees
  internally. Validate user input and data parsed from external files (BIN/BINX, XSYG,
  and the other instrument formats).
* **Mirror R behaviour exactly, including documented quirks.** A deviation from what
  the R source does isn't a local style choice; it needs a discussion first and a
  note in the relevant `tools/specs/` file (see *The porting principle*).

## Coding style

* **Type hints everywhere, clean under `basedpyright` (strict mode).** Every function
  parameter and return type, every dataclass/class field, and every variable whose
  type isn't obvious from the assignment. Modern syntax only: `list[float]`,
  `X | None`, `Literal[...]`.
* **`from __future__ import annotations`** at the top of every module.
* **Module docstrings cite the R source and version being ported**, e.g. `"""Reader
  for Risø BIN/BINX files (port of ``read_BIN2R``, R package v0.19)."""`, plus the
  relevant `tools/specs/*.md` file where one exists. See `src/luminescence/io/bin.py`
  for the pattern.
* **Google-style docstrings** (`Args:` / `Returns:` / `Raises:`) where a docstring is
  warranted at all, consistent with *Best practices*' "no unnecessary comments" above.
  Most private helpers need none.
* **Private helpers are prefixed with `_`** and not exported from `__init__.py`. Only
  the public API surface, re-exported flat from `luminescence/__init__.py`, is
  unprefixed.
* `ruff check .` and `basedpyright` must both pass clean. Line length is 100 columns.

## Porting workflow

1. Pick an unported R function from the phase table in `README.md`.
2. Extract a port spec into `tools/specs/<name>.md`: signature, behaviour, edge cases,
   and quirks, read directly from the R source. Use the existing files in
   `tools/specs/` as examples of the expected level of detail.
3. Implement it in the corresponding `src/luminescence/` module (see *Code
   organisation* below), named per `docs/r-migration.md`.
4. Validate against the R oracle:
   * Reference fixtures under `tests/python/fixtures/` (generated by
     `tools/generate_fixtures.R`); deterministic results must match within the
     documented tolerances (arithmetic 1e-9, fitted parameters 1e-4).
   * `testthat` snapshots (`tests/testthat/_snaps/`), parsed via
     `tests/python/oracle.py`, as an additional cross-language check.
   * Stochastic functions (Monte-Carlo error estimation, resampling) take an explicit
     `rng: np.random.Generator | int | None` argument and are compared statistically,
     not digit for digit.
5. Add a parity test under `tests/python/`.
6. Update the status table in `README.md` and, if a new naming pattern was introduced,
   `docs/r-migration.md`.

## Contribution workflow

### Branches

Branches are prefixed by scope, e.g. `dev_<short-description>` for a phase of porting
work or `chore_<short-description>` for tooling/cleanup. Base them on `master` of
`Cologne-Geomorphological-Software-Lab/Luminescence` (this repository, not
`R-Lum/Luminescence`).

### Commit messages

Write commit messages in the imperative mood, present tense:

```
Port read_BIN2R to io/bin.py (BIN/BINX v3-v8)

Add write_bin round-trip test against BINfile_V8.binx
```

* First line: 50 characters max, no trailing period.
* Optional body: explain *why*, not *what*. Reference issue numbers (`Closes #XX`).
* Keep commits granular: one function/module per commit rather than one commit per
  phase. Do not amend published commits.

### Pull requests

* Open against `master` of `Cologne-Geomorphological-Software-Lab/Luminescence`. Never
  against `R-Lum/Luminescence` upstream.
* Title mirrors the commit message style.
* Description must state: which R function(s) were ported or fixed, what changed, and
  how it was validated against the R oracle (fixtures, snapshots, or both).
* A series of checks (`python-check.yml`) runs automatically: lint (`ruff`,
  `basedpyright`) and the test matrix (Linux/Windows/macOS). They must turn green
  before merging.
* At least one approving review is required before merging.

### Tests

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run bandit -r src/luminescence
uv run vulture src/luminescence tests/python --min-confidence 80
```

Pre-commit runs the lint checks automatically on touched files
(`uv run pre-commit install` once, see *AI usage*). The complexity gate and
duplicate-code check are informational for now; run them by hand when touching a
function that might be affected:

```bash
uv run xenon --max-absolute B --max-modules B --max-average A src/luminescence
uv run pylint --disable=all --enable=duplicate-code src/luminescence
```

## Code organisation

| Package | Scope | R counterpart |
|---|---|---|
| `luminescence.core` | Object model: `Record`/`Curve`/`Spectrum`/`ImageData`/`Analysis`/`Results` | `RLum.*` S4 classes |
| `luminescence.io` | Instrument file readers/writers (BIN/BINX, XSYG, SPE, PSL, ...) | `read_*.R`, `write_*.R` |
| `luminescence.analysis` | Analysis protocols (SAR CW-OSL, irradiation handling) | `analyse_*.R` |
| `luminescence.fitting` | Curve fitting (dose-response, ...) | `fit_*.R` |
| `luminescence.models` | Statistics and calculation routines | `calc_*.R` |
| `luminescence.dosimetry` | Equivalent-dose / age models (Phase 3, open) | `calc_*Age*.R`, `calc_*Dose*.R` |
| `luminescence.bayes` | Bayesian analyses, optional `[bayes]` extra (Phase 6, open) | `baSAR`/rjags routines |
| `luminescence.plot` | Matplotlib plotting (idiomatic redesign, not a faithful port) | `plot_*.R` |
| `luminescence.utils` | Shared helpers, exceptions, validation | no direct R counterpart |

## Releases

There is no fixed release cadence. A first PyPI release of `pyluminescence` is planned
for Phase 7, once the migration reaches parity across the phases listed in
`README.md`.

[coco]: CODE_OF_CONDUCT.md
