# irradiapy

`irradiapy` is a Python toolkit for constructing and analysing primary irradiation damage data. It connects recoil spectra from [SRIM](http://www.srim.org/) or [SPECTRA-PKA](https://github.com/fispact/SPECTRA-PKA) with curated molecular dynamics (MD) cascade debris, and provides SQLite-backed storage, file I/O, defect analysis, and plotting tools.

The central workflow is:

1. Generate primary knock-on atoms (PKAs) with SRIM, or read time-dependent PKAs produced by SPECTRA-PKA.
2. Re-run recoils through SRIM when their energies exceed the MD cascade database range.
3. Store ions, recoils, target information, and SRIM output in SQLite databases.
4. Match each recoil to MD cascade debris.
5. Calculate defect, cluster, recoil, and displacements-per-atom (dpa) statistics.

irradiapy does not distribute SRIM, SPECTRA-PKA, or any MD cascade database. Those tools and data must be installed or obtained separately.

## Features

### Ion irradiation: SRIM

The [`irradiapy.srim`](https://github.com/acgc99/irradiapy/tree/main/irradiapy/srim) subpackage automates SRIM graphical application and parses its output.

- Runs quick, full, and monolayer SRIM calculations.
- Supports Windows directly and Linux through Wine.
- Converts SRIM text outputs into queryable SQLite tables.
- Collects all generated ions and recoils in a central `recoils.db`.
- Iteratively simulates recoils whose energies are above the configured MD
  debris range.

SRIM remains a GUI application: its windows are visible while a calculation runs. On Linux, automation requires Wine, `xdotool`, and an X11-compatible display (including XWayland where available). See [Installing SRIM on Linux](https://github.com/acgc99/irradiapy/tree/main/Installing_SRIM_Linux.md) for the tested setup.

### Neutron irradiation: SPECTRA-PKA and SRIM

The
[`irradiapy.spectrapka`](https://github.com/acgc99/irradiapy/tree/main/irradiapy/spectrapka) subpackage reads SPECTRA-PKA material and `config_events.pka` output, converts the PKA events to irradiapy's database format, and delegates high-energy recoils to the SRIM workflow.

SPECTRA-PKA itself is not launched by irradiapy. Run it separately on Linux and provide its input and event files to `irradiapy.spectrapka.Spectra2SRIM`. For Windows users, Windows Subsystem for Linux 2 (WSL2) is recommended.

### MD cascade debris

`DebrisDataset` reads cascade metadata and energy-indexed LAMMPS dump files from one dataset. `DebrisDatabase` filters and combines compatible datasets by target composition, lattice type, electronic-interaction model, interatomic potential, DOI, and contributors.

`irradiapy.analysis.debris.generate_debris` then matches stored recoils to those cascades, rotates and places the selected debris, applies boundary conditions, and represents unmatched low-energy recoils with Frenkel pairs.

Compatible curated datasets are available from [CascadesDefectsDB](https://github.com/acgc99/CascadesDefectsDB).

### Analysis

The [`irradiapy.analysis`](https://github.com/acgc99/irradiapy/tree/main/irradiapy/analysis) subpackage provides:

- defect identification similar to [U. Bhardwaj, et al. (2020)](https://doi.org/10.1016/j.commatsci.2019.109364);
- vacancy and self-interstitial cluster identification;
- recoil-energy, recoil-position, and injected-ion distributions;
- NRT, arc, and fer-arc dpa calculations;
- SQLite-backed analysis results and plotting helpers.

### I/O

The [`irradiapy.io`](https://github.com/acgc99/irradiapy/tree/main/irradiapy/io) subpackage contains:

- serial and MPI readers and writers for LAMMPS custom dump files;
- serial and MPI readers and writers for bzip2-compressed LAMMPS dumps;
- a LAMMPS log reader;
- basic XYZ readers and writers.

The parallel compressed reader uses `indexed-bzip2` for indexed, multi-threaded decompression. Compression and conversion helpers are available under `irradiapy.utils.io`.

### Materials and plotting

`irradiapy.materials` defines reusable `Element` and `Component` data classes, several built-in elements, and reference components for bcc Fe and W (you can request new components). `Component` implements recoil-to-damage-energy conversion and NRT, arc, and fer-arc dpa models.

`irradiapy.config.use_style()` enables the package's colourblind-friendly Matplotlib style. Pass `latex=True` to use LaTeX text rendering when a LaTeX installation is available.

## Installation

irradiapy 2.0.0 requires Python 3.14 or newer.

```bash
python -m pip install irradiapy
```

To work from a source checkout:

```bash
git clone https://github.com/acgc99/irradiapy.git
cd irradiapy
python -m pip install -e .
```

An MPI implementation is required by `mpi4py` and the MPI I/O classes. For example, Debian and Ubuntu users can install Open MPI with:

```bash
sudo apt install libopenmpi-dev openmpi-bin
```

Additional external requirements depend on the workflow:

- **SRIM on Windows:** install SRIM; Windows automation dependencies are installed with irradiapy.
- **SRIM on Linux:** install `wine`, `xdotool` and SRIM.
- **SPECTRA-PKA:** build and run SPECTRA-PKA separately on Linux or WSL2.
- **MD debris generation:** obtain a compatible cascade database such as CascadesDefectsDB.

## Basic configuration

SRIM and debris-database locations are configured explicitly:

```python
from pathlib import Path

import irradiapy as irpy

irpy.config.set_srim_dir(Path("/path/to/SRIM-2013"))
irpy.config.set_debris_database(
    path=Path("/path/to/CascadesDefectsDB"),
    electronic_interactions="SRIM",
    target={"Fe": 1.0},
    lattice="bcc",
)
```

The complete workflows require irradiation inputs and material-specific parameters. Start from the maintained examples:

- [SRIM ion irradiation and analysis](https://github.com/acgc99/irradiapy/blob/main/examples/srim.py)
- [SPECTRA-PKA, SRIM, and debris generation](https://github.com/acgc99/irradiapy/blob/main/examples/spectra.py)
- [MD debris database analysis](https://github.com/acgc99/irradiapy/blob/main/examples/debris_analysis.py)

The [API documentation](https://irradiapy.readthedocs.io/en/stable/) describes the individual classes and functions.

## Scientific use

This package should be cites as:
> A.-C. Gutiérrez-Camacho et al., “Towards a standardised methodology of radiation damage defect distributions for microstructure evolution models”, [*Scientific Reports* 15, 20596 (2025)](https://doi.org/10.1038/s41598-025-05661-2).

It has been used in the following works:

> A.-C. Gutiérrez-Camacho et al., “Towards a standardised methodology of radiation damage defect distributions for microstructure evolution models”, [*Scientific Reports* 15, 20596 (2025)](https://doi.org/10.1038/s41598-025-05661-2).


## Contributing

Questions belong in [GitHub Discussions](https://github.com/acgc99/irradiapy/discussions). Report bugs and feature requests through [GitHub Issues](https://github.com/acgc99/irradiapy/issues).

### Building the documentation

Run the following commands from the repository root with the project virtual environment activated:

```bash
python -m pip install -e . -r docs/requirements.txt
python -m sphinx.ext.apidoc --force --remove-old --separate -o docs/source/api irradiapy
python -m sphinx -W --keep-going -b html docs/source docs/build/html
```

The generated documentation is written to `docs/build/html/index.html`.

### Building the package

Run the following commands from the repository root with the project virtual environment activated:

```bash
python -m pip install build
python -m build
```

The wheel and source distributions are written to `dist`.

## License

irradiapy is distributed under the [MIT License](LICENSE).
