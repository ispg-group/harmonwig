# harmonwig

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]

Python package for Wigner sampling of harmonic vibrational wavefunction of
molecules.

** ⚠️ WARNING: This package is work in progress. Once it becomes usable, it will
be available as a Python package on PyPI :warning: **

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/ispg-group/harmonwig/workflows/CI/badge.svg
[actions-link]:             https://github.com/ispg-group/harmonwig/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/harmonwig
[conda-link]:               https://github.com/conda-forge/harmonwig-feedstock
[pypi-link]:                https://pypi.org/project/harmonwig/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/harmonwig
[pypi-version]:             https://img.shields.io/pypi/v/harmonwig
<!-- prettier-ignore-end -->

## Installation

Harmonwig is a simple Python CLI application, but it does rely on a handful of
other Python libraries that needs to be installed together. Currently, the
package is not yet available on PyPI, but it can still be installed directly
from GitHub. We recommend using `uv`, which automatically creates an isolated
Python environment and makes harmonwig available globally.

```console
$ pip install uv
$ uv tool install "harmonwig @ git+https://github.com/ispg-group/harmonwig.git"
```

To upgrade harmonwig to the latest version run:

```console
uv tool upgrade harmonwig
```

## Usage

Harmonwig currently supports reading normal modes and frequency data from ORCA
output files. To generate 500 hundred sampled geometries from output file
`orca_freq.out` run:

```console
harmonwig -n 500 orca_freq.out
```

Print help to see all options:

```console
$ harmonwig -h
usage: harmonwig [-h] [-n NSAMPLES] [--seed SEED] [--freqthr LOW_FREQ_THR] [-o OUTPUT_FNAME] INPUT_FILE

Program for harmonic Wigner sampling

positional arguments:
  INPUT_FILE            Output file from ab initio program.

options:
  -h, --help            show this help message and exit
  -n NSAMPLES, --nsamples NSAMPLES
                        Number of Wigner samples
  --seed SEED           Random seed
  --freqthr LOW_FREQ_THR
                        Low-frequency threshold
  -o OUTPUT_FNAME, --output-file OUTPUT_FNAME
                        Output file name
```
