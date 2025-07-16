# Irradiapy

This Python package is aimed towards the simulation and analysis of irradiation damage with multiple tools.

You can find examples under the `examples` folder. More examples will be provided as new functionalities are implemented. A documentation page is under development, but code is fairly documented.

If you have any question about the use of `irradiapy`, feel free to start a discussion. Otherwise, if you find a bug or you want to suggest a feature, open an issue. Note that not all materials have been implemented, only the ones that I have used. Open an issue and I will update the code as soon as possible.

## Functionalities

### SRIM

This subpackage runs [SRIM](http://www.srim.org/) from Python. Main features:
1. SRIM execution automation.
2. All outputs are saved into a SQLite database for easy use.
3. SRIM can be run in a iterative way given some conditions. For example, a incident ion of MeV energy, can generate a PKA of MeV (SRIM only follows the incident ion in `COLLISON.txt`). That might be too high for your application. In this case, SRIM can be run again automatically to simulate that PKA as a new ion.

Please note that:
1. You must obtain SRIM and make it work on your own before using this functionality. SRIM is not included here.
2. SRIM was designed to be run with a GUI. I managed to handle it the best way I could. A SRIM window will open in every run, but it will be minimised.
3. I think this can adapted to run in Linux with Wine, but I do not have a Linux system. Someone could help with this.
4. All SRIM output files are saved as SQLite tables. All of them correspond to the initial incident ions, if SRIM is executed interatively, they are not updated. The only exception is `COLLISON.txt`, which has a special treatment (see feature 3.).

This package is applied in [Gutiérrez-Camacho, A.-C. et al. (2025)](https://doi.org/10.1038/s41598-025-05661-2). We obtained the list of PKAs produced by ions, and then we placed molecular dynamics collisional cascades debris accordinly. You can find the database we used in [CascadesDefectsDB](https://github.com/acgc99/CascadesDefectsDB.git) repository. [This](https://github.com/acgc99/irradiapy/blob/f52507b6d6b3a263440915b52f3b987c7b19f2bd/examples/srim.py) is an example about how to use this feature.

### SPECTRA-PKA

SPECTRA-PKA will be transfered to SRIM as described above. Work in progress.

### LAMMPS

This will be Python workflow that uses the LAMMPS package to generate databases of molecular dynamcis cascades "easily". Work in progress.

### Analysis

Under [`analysis`](https://github.com/acgc99/irradiapy/tree/39b5de7f575024101dfec23f6373b8c454bead81/irradiapy/analysis) subpackage you will find tools for defect and cluster identification. Together with the dpa module, a few basic plots are provided.

Please, note that the original version of defect identification ([Gutiérrez-Camacho, A.-C. et al. (2025)](https://doi.org/10.1038/s41598-025-05661-2)) is deprecated since version 1.0. I realised that it was not very efficent and the current code is more similar to the reference algorithm ([Bhardwaj, U. et al. (2020)](https://doi.org/10.1016/j.commatsci.2019.109364)). However, I think this code can be further improved with MPI parallelisation, but I miss the technical knowledge for this.

### I/O

The subpackage [`io`](https://github.com/acgc99/irradiapy/tree/39b5de7f575024101dfec23f6373b8c454bead81/irradiapy/io) provides a set of basic readers and writers for LAMMPS dump custom:
- Standard
- MPI parallelised
- bzip2 compressed
  
There are functions to compress LAMMPS dump custom text files using bzip2, this is highly recommend to store large amounts of data (from personal experience, a 26 GB file can end up around 7 GB once compressed). You do not need to decompress the files for processing, use the proper reader instead. However, you will not be able to use commercial visualisation tools with compressed files.

This subpackage also provides a basic XYZ reader and reader. These work differently from LAMMPS readers/writers and are not longer updated, since I decided to keep LAMMPS format as default. The XYZ reader and writer are there because I used them long time ago.

### Matplotlib style

Using `irradiapy.config.use_style()` you can take advantage of the colorblind-friendly matplotlib style used in [Gutiérrez-Camacho, A.-C. et al. (2025)](https://doi.org/10.1038/s41598-025-05661-2). If `latex=True` is provided, it will use LaTeX fonts, but it is slower and you might need to install LaTeX first (not sure).

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
```
Note that:
- The code might work with previous versions, but results might change slightly. For example, [`scipy.spatial.transform.Rotation.align_vectors`](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html) changed its behaviour when only two vectors are provided between versions 1.11 and 1.12.
- `mpi4py` is not needed now, but it will be requeried in the following versions. I need to check this, but its version must be < 4 for LAMMPS compatibility.
- (For the future) If you do `pip install lammps`, you are not installing the right thing. To use your LAMMPS distribution you must build the package from source.

Installation (PyPI link [here](https://pypi.org/project/irradiapy/)):
```
pip install irradiapy
```
