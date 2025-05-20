"""irradiapy package"""

import matplotlib.pyplot as plt

from irradiapy import dpa, dtypes, io, materials, srimpy, analysis
from irradiapy.damagedb import DamageDB


def use_style(latex: bool = False) -> None:
    """Set the style for matplotlib plots.

    It uses the colour universal design (CUD) palette for colour-blind friendly plots.

    Parameters
    ----------
    latex : bool, optional
        If True, use LaTeX for text rendering in plots (slower). Default is False. I might require
        other software to be installed on your system.
    """
    if latex:
        plt.style.use("irradiapy.styles.latex")
    else:
        plt.style.use("irradiapy.styles.nolatex")
        plt.style.use("irradiapy.styles.nolatex")
