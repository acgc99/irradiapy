"""Module for configuration variables."""

from pathlib import Path

import matplotlib.pyplot as plt

from irradiapy.debris_database import DebrisDatabase

# region General

#: str: Format for integers in output files.
INT_FORMAT = "%d"
#: str: Format for floats in output files.
FLOAT_FORMAT = "%g"
#: str: Encoding for text files.
ENCODING = "utf-8"
#: str: newline character for text files used by writers.
NEWLINE = "\n"
#: list[str]: List of atom fields to exclude from output in LAMMPS files.
EXCLUDED_ITEMS = []


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
_srim_dir = None  # pylint: disable=invalid-name


def set_srim_dir(path: str | Path) -> None:
    """Set the path to the TRIM.exe directory.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the TRIM.exe directory (parent folder).
    """
    global _srim_dir  # pylint: disable=global-statement
    _srim_dir = Path(path)


def get_srim_dir() -> Path:
    """Return the path to the TRIM.exe directory.

    Returns
    -------
    pathlib.Path
        Path to the TRIM.exe directory (parent folder).
    """
    if _srim_dir is None:
        raise RuntimeError(
            "No SRIM directory configured. Call "
            "irradiapy.config.set_srim_dir(path) before running SRIM workflows."
        )
    return _srim_dir


def clear_srim_dir() -> None:
    """Clear the configured path to the TRIM.exe directory."""
    global _srim_dir  # pylint: disable=global-statement
    _srim_dir = None


# endregion

# region Debris database

#: DebrisDatabase | None: Global MD debris database used by SRIM/debris workflows.
_debris_database: DebrisDatabase | None = None  # pylint: disable=invalid-name


def set_debris_database(
    path: str | Path,
    electronic_interactions: str,
    target: dict[str, float],
    interatomic_potentials: list[set[str]] | None = None,
    doi: set[str] | None = None,
    contributors: list[set[str]] | None = None,
) -> DebrisDatabase:
    """Configure the global MD debris database.

    Parameters
    ----------
    path : str | pathlib.Path
        Database root path.
    electronic_interactions : str
        Electronic interactions metadata required for selected datasets.
    target : dict[str, float]
        Target stoichiometry metadata required for selected datasets.
    interatomic_potentials : list[set[str]] | None, optional
        Accepted interatomic potential metadata sets.
    doi : set[str] | None, optional
        Accepted DOI metadata values.
    contributors : list[set[str]] | None, optional
        Accepted contributor metadata sets.

    Returns
    -------
    DebrisDatabase
        Configured debris database.

    Note
    ----
    Recoil is not a database-level filter because it is unknown when using SPECTRA-PKA.
    """
    global _debris_database  # pylint: disable=global-statement

    _debris_database = DebrisDatabase(
        root=path,
        electronic_interactions=electronic_interactions,
        target=target,
        interatomic_potentials=interatomic_potentials,
        doi=doi,
        contributors=contributors,
    )
    return _debris_database


def get_debris_database() -> DebrisDatabase:
    """Return the configured global MD debris database."""
    if _debris_database is None:
        raise RuntimeError(
            "No debris database configured. Call "
            "irradiapy.config.set_debris_database(path, electronic_interactions, target) "
            "before running "
            "SRIM/debris workflows."
        )
    return _debris_database


def clear_debris_database() -> None:
    """Clear the configured global MD debris database."""
    global _debris_database  # pylint: disable=global-statement
    _debris_database = None


# endregion
