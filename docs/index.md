# Luminescence (Python)

A Python port of the R package
[Luminescence](https://github.com/R-Lum/Luminescence) for luminescence dating data
analysis. The port is under active development; see the
[README](https://github.com/Cologne-Geomorphological-Software-Lab/Luminescence#status)
for the current migration status.

## Installation

Not yet on PyPI. Install from source (Python >= 3.12):

```bash
git clone https://github.com/Cologne-Geomorphological-Software-Lab/Luminescence.git
cd Luminescence
uv sync            # or: pip install -e .
```

## First steps

Read a Risø BIN/BINX file and run a SAR CW-OSL analysis:

```python
import luminescence as lum

data = lum.read_bin("measurement.binx")
aliquot = data.to_analysis(pos=1)

results = lum.analyse_sar_cwosl(
    aliquot,
    signal_integral=range(1, 3),
    background_integral=range(900, 1001),
)
print(results["data"][["De", "De.Error", "D01", "RC.Status"]])
```

Channel integrals are 1-based and inclusive, matching the conventions of the R package
and the instrument software.

## How the documentation is organised

- The [API reference](reference/core.md) documents every public module, grouped by
  subsystem (core data model, file I/O, analysis, fitting, models, plotting, utilities).
- [Working with the R package](r-migration.md) explains the naming scheme and what to
  expect when translating existing R scripts.
