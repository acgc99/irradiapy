# Irradiapy

This Python package is aimed towards the simulation and analysis of irradiation primary damage with multiple tools.

You can find examples under the [`examples`](./examples/) folder. More examples will be provided as new features are implemented. A basic documentation page is hosted in Read the Docs [here](https://irradiapy.readthedocs.io/en/stable/). If you have any question about the use of `irradiapy`, feel free to start a discussion. Otherwise, if you find a bug or you want to suggest a feature, open an issue.

## Main features

The main purpose of this package is to obtain the primary irradiation damage in a fast and efficient way, doing some simplifications. The key idea is that recoils given by some code are mapped into a database of molecular dynamics (MD) collision cascades debris.

### Ion irradiation

[SRIM](http://www.srim.org/) is a widely used code for ion irradiation based on the binary collision approximation (BCA). Its source code is not available, but the programme itself is free and easily downloaded. The software is old and not receiving updates, and it has some known issues, see for example [S. Agarwal, et al. (2021)](https://doi.org/10.1016/j.nimb.2021.06.018). However, since it is popular and accessible, it has been used, although the prinpicles can also be applied to other BCA codes.

[`srim`](./irradiapy/srim/) subpackage runs SRIM from Python. Main features:
1. SRIM execution automation.
2. All outputs are saved into a SQLite database for easy use (`srim.db`).
3. SRIM can be run in an iterative way given some conditions. For example, a incident ion of several MeV of energy, can generate a PKA of MeV. That might be too high for your MD database. In this case, SRIM can be run again automatically to simulate that recoil as a new ion.
4. Theoretically, SRIM can run any ion type in a run, however, there is a bug where all the ions are assumed to have the same atomic number even if specified otherwise. To fix this, a tree structure is used: ions are separated by atomic number (folders will be created) with their corresponding output. Then subsequent recoils will also be separated by atomic number (creating even more folders). Each folder contains a `srim.db`database (see point 2). Then in the main folder, `recoils.db` gathers all the information from the tree of runs, so it is centralized and ready to couple with MD.

Please note that:
1. You must obtain SRIM and make it work on your own. SRIM is not included here. It is a Windows programme, but it can also be run on Linux with Wine. Please, read this [guide](./Installing_SRIM_Linux.md). It works on my Kubuntu machine, if you have issues, I might be able to help, but no promises are made.
3. SRIM was designed to be run with a GUI. I managed to handle it the best way I could. A SRIM window will open in every run, which might be minimised.

This subpackage is applied in [A.-C. Gutiérrez-Camacho, et al. (2025)](https://doi.org/10.1038/s41598-025-05661-2). We obtained the list of PKAs produced by ions, and then we placed MD debris accordingly. You can find the database we used in a [section](#available-databases) below. [Example](./examples/srim.py).

### Neutron irradiation

[SPECTRA-PKA](https://github.com/fispact/SPECTRA-PKA) provides a list with the PKAs produced over time in a targer material given a neutron spectrum. The idea is to couple the recoils with the MD database of debris. If a recoil energy is too high, SRIM is used to get lower energies.

[`spectrapka`](./irradiapy/spectrapka/) subpackage is the tool that handle this.

Please note that:
1. SPECTRA-PKA is a Linux-only code.
2. SPECTRA-PKA run is not managed by Python, you must first run it yourself, and provide `irradiapy` with the required inputs from it.

It has been used in the paper [Gutiérrez-Camacho, A.-C. et al (2026)][to be published]. [Example](./examples/spectra.py).

### Building the MD database

This will be Python workflow that uses the LAMMPS package to generate databases of MD cascades "easily" using an object oriented approach. Work in progress.

### Available databases

Curated databased for `irradiapy` can be found in [CascadesDefectsDB](https://github.com/acgc99/CascadesDefectsDB.git).

## Extra tools

### Analysis

Under [`analysis`](./irradiapy/analysis/) subpackage you will find tools for defect and cluster identification. Together with the dpa module, a few basic plots are provided.

Please, note that the original version of defect identification ([A.-C Gutiérrez-Camacho, et al. (2025)](https://doi.org/10.1038/s41598-025-05661-2)) is deprecated since version 1.0.0. I realised that it was not very efficent and the current code is more similar to the reference algorithm ([Bhardwaj, U. et al. (2020)](https://doi.org/10.1016/j.commatsci.2019.109364)). However, I think this code can be further improved with MPI parallelisation, but I miss the technical knowledge for this.

### I/O

The subpackage [`io`](./irradiapy/io/) provides a set of basic readers and writers for LAMMPS dump custom:
- Standard
- MPI parallelised
- bzip2 compressed
  
There are functions to compress LAMMPS dump custom text files using bzip2, this is highly recommended to store large amounts of data (from personal experience, a 26 GB file can end up around 7 GB once compressed). You do not need to decompress the files for processing, use the proper reader instead. However, you will not be able to use commercial visualisation tools with compressed files.

This subpackage also provides a basic XYZ reader and reader. These work differently from LAMMPS readers/writers and are not longer updated, since I decided to keep LAMMPS format as default. The XYZ reader and writer are there because I used them long time ago.

### Matplotlib style

Using `irradiapy.config.use_style()` you can take advantage of the colorblind-friendly matplotlib style used in [A.-C. Gutiérrez-Camacho, et al. (2025)](https://doi.org/10.1038/s41598-025-05661-2). If `latex=True` is provided, it will use LaTeX fonts, but it is slower and you might need to install LaTeX first (not sure).

## Installation

Dependencies:
```
matplotlib>=3.10.1
numpy>=2.2.4
PyGetWindow>=0.0.9
pywinauto>=0.6.8
scikit_learn>=1.6.0
scipy>=1.14.1
mpi4py>=3.0.0
indexed-bzip2>=1.7.0
```
Note that:
- The code might work with previous packages versions, but results might change slightly. For example, [`scipy.spatial.transform.Rotation.align_vectors`](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html) changed its behaviour when only two vectors are provided between versions 1.11 and 1.12.
- (Not applicable yet) You need to be able to run LAMMPS from Python. On Windows, you will need first to install [MS-MPI](https://learn.microsoft.com/en-gb/message-passing-interface/microsoft-mpi) (`msmpisetup.exe`), and then LAMMPS for Windows with Python support [LAMMPS-64bit-Python-latest-MSMPI.exe](https://rpm.lammps.org/windows/). In Linux, you need to build the Python package from source with `make install-python` or similar.

Installation (PyPI link [here](https://pypi.org/project/irradiapy/)):

```bash
pip install irradiapy
```

## Contributing

### Building the documentation

Run the following commands from the repository root with the project virtual environment activated:

```bash
python -m pip install sphinx -r docs/requirements.txt
python -m pip install -e .

python -m sphinx.ext.apidoc --force --remove-old --separate -o docs/source/api irradiapy
python -m sphinx -M clean docs/source docs/build
python -m sphinx -M html docs/source docs/build
```

The generated documentation is available at `docs/build/html/index.html`. The final command treats warnings as build failures, which keeps release documentation from silently shipping broken references or malformed docstrings.
