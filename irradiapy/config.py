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


def set_debris_database(path_or_db: str | Path | DebrisDatabase) -> DebrisDatabase:
    """Configure the global MD debris database.

    Parameters
    ----------
    path_or_db : str | pathlib.Path | DebrisDatabase
        Database root path or an already loaded database.

    Returns
    -------
    DebrisDatabase
        Configured debris database.
    """
    global _debris_database  # pylint: disable=global-statement

    if isinstance(path_or_db, DebrisDatabase):
        _debris_database = path_or_db
    else:
        _debris_database = DebrisDatabase.from_path(Path(path_or_db))
    return _debris_database


def get_debris_database() -> DebrisDatabase:
    """Return the configured global MD debris database."""
    if _debris_database is None:
        raise RuntimeError(
            "No debris database configured. Call "
            "irradiapy.config.set_debris_database(path_or_db) before running "
            "SRIM/debris workflows."
        )
    return _debris_database


def clear_debris_database() -> None:
    """Clear the configured global MD debris database."""
    global _debris_database  # pylint: disable=global-statement
    _debris_database = None


# endregion
