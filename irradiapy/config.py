"""Module for configuration variables."""

from pathlib import Path

import matplotlib.pyplot as plt

# region General

#: str: Format for integers in output files.
INT_FORMAT = "%d"
#: str: Format for floats in output files.
FLOAT_FORMAT = "%g"
#: str: Encoding for text files.
ENCODING = "utf-8"
#: list[str]: List of atom fields to exclude from output in LAMMPS files.
EXCLUDED_ITEMS = ["xs", "ys", "zs"]


def use_style(latex: bool = False) -> None:
    """Set the style for matplotlib plots.

    It uses the colour universal design (CUD) palette for colour-blind friendly plots.

    Parameters
    ----------
    latex : bool, optional (default=False)
        If True, use LaTeX for text rendering in plots (slower). I might require other software to
        be installed on your system.
    """
    if latex:
        plt.style.use("irradiapy.styles.latex")
    else:
        plt.style.use("irradiapy.styles.nolatex")


# endregion

# region SRIM

#: pathlib.Path: TRIM.exe directory (parent folder)
DIR_SRIM = Path("./SRIM-2013")

# endregion
