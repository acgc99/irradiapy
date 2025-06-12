# Irradiapy

This Python package is aimed towards the simulation and analysis of irradiation damage with multiple tools.

This initial version works and is ready for production, but the code is under revision to improve usability, readability and efficiency (parallelisation). You can find an example under the `examples` folder. More examples will be provided with the next versions, as well as a documentation page.

## Functionalities

### srimpy

This subpackage runs [SRIM](http://www.srim.org/) from Python with some tweaks for automation . All SRIM outputs are saved into a SQLite database for easy use.

Please note that:
- You must obtain SRIM and make it work on own before using this functionality. SRIM is not included here.
- SRIM was designed to be run with a GUI. I managed to handled it the best way I could. A SRIM window will open in every run, but it will be minimised.
- I think this can adapted to run in Linux with Wine, but I do not have a Linux system. Someone could help with this.

With this subpackage, you get the list of PKAs produced by ions, and then you can place molecular dynamics collisional cascades debris accordinly, as described in [to be published]. You can find the database we used in [to be published] in [CascadesDefectsDB](https://github.com/acgc99/CascadesDefectsDB.git) repository.

### lammpspy

I am also working on a Python workflow that uses the LAMMPS package to generate databases of molecular dynamcis cascades "easily". This is in development and will be publish later.

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
- `mpi4py` is not needed now, but it will be requeried in the following versions. I need to check this, but its version must be < 4 for LAMMPS compatibility.
- (For the future) If you do `pip install lammps`, you are not installing the right thing. To use your LAMMPS distribution you must build the package from source.

Installation (PyPI link [here](https://pypi.org/project/irradiapy/)):
```
pip install irradiapy
```
