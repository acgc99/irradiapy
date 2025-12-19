"""This module contains the `AnalysisDB` class."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Callable, Generator

import numpy as np
import numpy.typing as npt

from irradiapy.utils.math import gaussian, lorentzian, power_law


@dataclass
class AnalysisDB(sqlite3.Connection):
    """SQLite database for storing analysis results.

    Parameters
    ----------
    path : Path
        Path to the SQLite database file.
    """

    path: Path

    def __post_init__(self) -> None:
        super().__init__(self.path)

    def __exit__(
        self,
        exc_type: None | type[BaseException] = None,
        exc_value: None | BaseException = None,
        exc_traceback: None | TracebackType = None,
    ) -> bool:
        """Exit the runtime context related to this object."""
        self.close()
        return False

    def optimize(self) -> None:
        """Optimize the SQLite database.

        This method performs two operations to optimize the database:
        1. Executes the "PRAGMA optimize" command to analyze and optimize the database.
        2. Executes the "VACUUM" command to rebuild the database file,
        repacking it into a minimal amount of disk space.
        """
        cur = self.cursor()
        cur.execute("PRAGMA optimize")
        cur.execute("VACUUM")
        cur.close()

    def table_exists(self, table_name: str) -> bool:
        """Checks if the given table exists in the database.

        Parameters
        ----------
        table_name : str
            Table's name to check.

        Returns
        -------
        bool
            Whether the table already exists or not.
        """
        cur = self.cursor()
        cur.execute(
            (
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                f"AND name='{table_name}'"
            )
        )
        result = cur.fetchone()[0]
        cur.close()
        return bool(result)

    def table_has_column(self, table_name: str, column_name: str) -> bool:
        """Checks if the given table has the specified column.

        Parameters
        ----------
        table_name : str
            Table's name to check.
        column_name : str
            Column's name to check.

        Returns
        -------
        bool
            Whether the column exists in the table or not.
        """
        cur = self.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in cur.fetchall()]
        cur.close()
        return column_name in columns

    def read(
        self, table: str, what: str = "*", condition: str = ""
    ) -> Generator[tuple, None, None]:
        """Reads table data from the database as a generator.

        Parameters
        ----------
        table : str
            Table to read from.
        what : str
            Columns to select.
        condition : str
            Condition to filter data.

        Yields
        ------
        Generator[tuple, None, None]
            Data from the database.
        """
        cur = self.cursor()
        cur.execute(f"SELECT {what} FROM {table} {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()

    # region Recoils

    def save_recoil_energies_hist(
        self, energy_centers: npt.NDArray[np.float64], hist: npt.NDArray[np.float64]
    ) -> None:
        """Save recoil energies histogram into the database.

        Parameters
        ----------
        energy_centers : npt.NDArray[np.float64]
            Centers of the energy bins.
        hist : npt.NDArray[np.float64]
            Histogram values.
        """
        table_name = "recoil_energies"
        cur = self.cursor()
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} (energy_centers, counts REAL)"
        )
        cur.executemany(
            f"INSERT INTO {table_name} (energy_centers, counts) VALUES (?, ?)",
            zip(energy_centers, hist),
        )
        self.commit()
        cur.close()

    def save_recoil_energies_hist_fit_params(self, a: float, k: float) -> None:
        """Save recoil energies histogram fit parameters into the database.

        Fit: power law function.

        Parameters
        ----------
        a : float
            Fit parameter a.
        k : float
            Fit parameter k.
        """
        table_name = "recoil_energies_params"
        cur = self.cursor()
        cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (a REAL, k REAL)")
        cur.execute(f"INSERT INTO {table_name} " f"(a, k) VALUES (?, ?)", (a, k))
        self.commit()
        cur.close()

    def save_recoil_energies_hist_fit_errors(self, a: float, k: float) -> None:
        """Save recoil energies histogram fit errors into the database.

        Fit: power law function.

        Parameters
        ----------
        a : float
            Fit error for parameter a.
        k : float
            Fit error for parameter k.
        """
        table_name = "recoil_energies_errors"
        cur = self.cursor()
        cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (a REAL, k REAL)")
        cur.execute(f"INSERT INTO {table_name} (a, k) VALUES (?, ?)", (a, k))
        self.commit()
        cur.close()

    def load_recoil_energies_hist(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Load recoil energies histogram from the database.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            A tuple containing energy centers and histogram values.
        """
        data = list(self.read(table="recoil_energies", what="energy_centers, counts"))
        energy_centers = np.array([row[0] for row in data])
        hist = np.array([row[1] for row in data])
        return energy_centers, hist

    def load_recoil_energies_hist_fit_parameters(self) -> tuple[float, float]:
        """Load recoil energies histogram fit parameters from the database.

        Fit: power law function.

        Returns
        -------
        tuple[float, float]
            Fit parameters (a, k).
        """
        params = self.read(table="recoil_energies_params", what="a, k")
        return next(params)

    def load_recoil_energies_hist_fit_errors(self) -> tuple[float, float]:
        """Load recoil energies histogram fit errors from the database.

        Fit: power law function.

        Returns
        -------
        tuple[float, float]
            Fit errors (a, k).
        """
        errors = self.read(table="recoil_energies_errors", what="a, k")
        return next(errors)

    def load_recoil_energies_hist_fit_function(self) -> Callable:
        """Load recoil energies histogram fit function from the database.

        Fit: power law function.

        Returns
        -------
        Callable
            A function representing the fit.
        """
        a, k = self.load_recoil_energies_hist_fit_parameters()
        return lambda x: power_law(x, a, k)

    # endregion

    # region Ions depth

    def save_depth_ions_hist(
        self,
        axis: str,
        depth_centers: npt.NDArray[np.float64],
        hist: npt.NDArray[np.float64],
    ) -> None:
        """Save depth ions histogram into the database.

        Parameters
        ----------
        axis : str
            Axis along which the depth ions data is calculated ('x', 'y', or 'z').
        depth_centers : npt.NDArray[np.float64]
            Centers of the depth bins.
        hist : npt.NDArray[np.float64]
            Histogram values.
        """
        table_name = f"depth_ions_{axis}"
        cur = self.cursor()
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} (depth_centers REAL, counts REAL)"
        )
        cur.executemany(
            f"INSERT INTO {table_name} (depth_centers, counts) VALUES (?, ?)",
            zip(depth_centers, hist),
        )
        self.commit()
        cur.close()

    def save_depth_ions_hist_fit_params(
        self,
        axis: str,
        x0: float,
        sigma: float,
        amplitude: float,
        asymmetry: float,
    ) -> None:
        """Save depth ions histogram fit parameters into the database.

        Fit: asymmetric gaussian function.

        Parameters
        ----------
        axis : str
            Axis along which the depth ions data is calculated ('x', 'y', or 'z').
        x0 : float
            Position with maximum value.
        sigma : float
            Linewidth.
        amplitude : float
            Maximum amplitude.
        asymmetry : float
            Asymmetry.
        """
        table_name = f"depth_ions_params_{axis}"
        cur = self.cursor()
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} "
            "(x0 REAL, sigma REAL, amplitude REAL, asymmetry REAL)"
        )
        cur.execute(
            f"INSERT INTO {table_name} "
            f"(x0, sigma, amplitude, asymmetry) VALUES (?, ?, ?, ?)",
            (x0, sigma, amplitude, asymmetry),
        )
        self.commit()
        cur.close()

    def save_depth_ions_hist_fit_errors(
        self,
        axis: str,
        x0: float,
        sigma: float,
        amplitude: float,
        asymmetry: float,
    ) -> None:
        """Save depth ions histogram fit errors into the database.

        Fit: asymmetric gaussian function.

        Parameters
        ----------
        axis : str
            Axis along which the depth ions data is calculated ('x', 'y', or 'z').
        x0 : float
            Position with maximum value.
        sigma : float
            Linewidth.
        amplitude : float
            Maximum amplitude.
        asymmetry : float
            Asymmetry.
        """
        table_name = f"depth_ions_errors_{axis}"
        cur = self.cursor()
        cur.execute(
            (
                f"CREATE TABLE IF NOT EXISTS {table_name} "
                "(x0 REAL, sigma REAL, amplitude REAL, asymmetry REAL)"
            )
        )
        cur.execute(
            f"INSERT INTO {table_name} "
            f"(x0, sigma, amplitude, asymmetry) VALUES (?, ?, ?, ?)",
            (x0, sigma, amplitude, asymmetry),
        )
        self.commit()
        cur.close()

    def load_depth_ions_hist(
        self, axis: str
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Load depth ions histogram from the database.

        Parameters
        ----------
        axis : str
            Axis along which the depth ions data is calculated ('x', 'y', or 'z').

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            A tuple containing depth centers and histogram values.
        """
        table_name = f"depth_ions_{axis}"
        data = list(self.read(table=table_name, what="depth_centers, counts"))
        depth_centers = np.array([row[0] for row in data])
        hist = np.array([row[1] for row in data])
        return depth_centers, hist

    def load_depth_ions_hist_fit_parameters(
        self, axis: str
    ) -> tuple[float, float, float, float]:
        """Load depth ions histogram fit parameters from the database.

        Fit: asymmetric gaussian function.

        Parameters
        ----------
        axis : str
            Axis along which the depth ions data is calculated ('x', 'y', or 'z').

        Returns
        -------
        tuple[float, float, float, float]
            Fit parameters (x0, sigma, amplitude, asymmetry).
        """
        table_name = f"depth_ions_params_{axis}"
        params = self.read(table=table_name, what="x0, sigma, amplitude, asymmetry")
        return next(params)

    def load_depth_ions_hist_fit_errors(
        self, axis: str
    ) -> tuple[float, float, float, float]:
        """Load depth ions histogram fit errors from the database.

        Fit: asymmetric gaussian function.

        Parameters
        ----------
        axis : str
            Axis along which the depth ions data is calculated ('x', 'y', or 'z').

        Returns
        -------
        tuple[float, float, float, float]
            Fit errors (x0, sigma, amplitude, asymmetry).
        """
        table_name = f"depth_ions_errors_{axis}"
        errors = self.read(table=table_name, what="x0, sigma, amplitude, asymmetry")
        return next(errors)

    def load_depth_ions_hist_fit_function(self, axis: str) -> Callable:
        """Load depth ions histogram fit function from the database.

        Fit: asymmetric gaussian function.

        Parameters
        ----------
        axis : str
            Axis along which the depth ions data is calculated ('x', 'y', or 'z').

        Returns
        -------
        Callable
            A function representing the fit.
        """
        x0, sigma, amplitude, asymmetry = self.load_depth_ions_hist_fit_parameters(axis)
        return lambda x: gaussian(x, x0, sigma, amplitude, asymmetry)

    # endregion

    # region Displacements

    def save_depth_dpa_hist(
        self,
        axis: str,
        model: str,
        depth_centers: npt.NDArray[np.float64],
        hist: npt.NDArray[np.float64],
    ) -> None:
        """Save depth-dpa histogram into the database.

        Parameters
        ----------
        axis : str
            Axis along which the depth-dpa data is calculated ('x', 'y', or 'z').
        model : str
            Model name.
        depth_centers : npt.NDArray[np.float64]
            Centers of the depth bins.
        hist : npt.NDArray[np.float64]
            Histogram values.
        """
        table_name = f"depth_dpa_{axis}"
        cur = self.cursor()
        # Create table with only depth_centers column
        cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (depth_centers)")
        # If depth_centers is empty, insert them
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        if cur.fetchone()[0] == 0:
            cur.executemany(
                f"INSERT INTO {table_name} (depth_centers) VALUES (?)",
                zip(depth_centers),
            )
        # Add a new column for the model if it does not exist
        cur.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in cur.fetchall()]
        if model not in columns:
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {model} REAL")
        else:
            raise ValueError(f"Column '{model}' already exists in table '{table_name}'")
        # Update the dpa values for the model
        cur.executemany(
            f"UPDATE {table_name} SET {model} = ? WHERE depth_centers = ?",
            zip(hist, depth_centers),
        )
        self.commit()
        cur.close()

    def save_depth_dpa_hist_fit_params(
        self,
        axis: str,
        model: str,
        x0: float,
        sigma: float,
        amplitude: float,
        asymmetry: float,
    ) -> None:
        """Save depth-dpa histogram fit parameters into the database.

        Fit: asymmetric lorentzian function.

        Parameters
        ----------
        axis : str
            Axis along which the depth-dpa data is calculated ('x', 'y', or 'z').
        model : str
            Model name.
        x0 : float
            Position with maximum value.
        sigma : float
            Linewidth.
        amplitude : float
            Maximum amplitude.
        asymmetry : float
            Asymmetry.
        """
        table_name = f"depth_dpa_params_{axis}"
        cur = self.cursor()
        cur.execute(
            (
                f"CREATE TABLE IF NOT EXISTS {table_name} "
                "(model TEXT PRIMARY KEY, x0 REAL, sigma REAL, amplitude REAL, asymmetry REAL)"
            )
        )
        cur.execute(
            f"INSERT INTO {table_name} "
            "(model, x0, sigma, amplitude, asymmetry) VALUES (?, ?, ?, ?, ?)",
            (model, x0, sigma, amplitude, asymmetry),
        )
        self.commit()
        cur.close()

    def save_depth_dpa_hist_fit_errors(
        self,
        axis: str,
        model: str,
        x0: float,
        sigma: float,
        amplitude: float,
        asymmetry: float,
    ) -> None:
        """Save depth-dpa histogram fit errors into the database.

        Fit: asymmetric lorentzian function.

        Parameters
        ----------
        axis : str
            Axis along which the depth-dpa data is calculated ('x', 'y', or 'z').
        model : str
            Model name.
        x0 : float
            Position with maximum value.
        sigma : float
            Linewidth.
        amplitude : float
            Maximum amplitude.
        asymmetry : float
            Asymmetry.
        """
        table_name = f"depth_dpa_errors_{axis}"
        cur = self.cursor()
        cur.execute(
            (
                f"CREATE TABLE IF NOT EXISTS {table_name} "
                "(model TEXT PRIMARY KEY, x0 REAL, sigma REAL, amplitude REAL, asymmetry REAL)"
            )
        )
        cur.execute(
            f"INSERT INTO {table_name} "
            "(model, x0, sigma, amplitude, asymmetry) VALUES (?, ?, ?, ?, ?)",
            (model, x0, sigma, amplitude, asymmetry),
        )
        self.commit()
        cur.close()

    def save_dpa(self, model: str, dpa: float) -> None:
        """Save dpa value into the database.

        Parameters
        ----------
        model : str
            Model name.
        hist : float
            dpa histogram values.
        """
        cur = self.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS dpa (model TEXT PRIMARY KEY, dpa REAL)")
        cur.execute(
            "INSERT INTO dpa (model, dpa) VALUES (?, ?)",
            (model, dpa),
        )
        self.commit()
        cur.close()

    def load_depth_dpa_hist(
        self,
        axis: str,
        model: str,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Load depth-dpa histogram data from the database.

        Parameters
        ----------
        axis : str
            Axis along which the depth-dpa data is calculated ('x', 'y', or 'z').
        model : str
            Model name.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            A tuple containing depth centers and dpa histogram for the specified model.
        """
        table_name = f"depth_dpa_{axis}"
        data = list(self.read(table=table_name, what="depth_centers, " + model))
        depth_centers = np.array([row[0] for row in data])
        hist = np.array([row[1] for row in data])
        return depth_centers, hist

    def load_depth_dpa_hist_fit_parameters(
        self, axis: str, model: str
    ) -> tuple[float, float, float, float]:
        """Load depth-dpa histogram fit parameters from the database.

        Fit: asymmetric lorentzian function.

        Parameters
        ----------
        axis : str
            Axis along which the depth-dpa data is calculated ('x', 'y', or 'z').
        model : str
            Model name.

        Returns
        -------
        tuple[float, float, float, float]
            A tuple containing fit parameters (x0, sigma, amplitude, asymmetry).
        """
        params = self.read(
            table=f"depth_dpa_params_{axis}",
            what="x0, sigma, amplitude, asymmetry",
            condition=f"WHERE model='{model}'",
        )
        return next(params)

    def load_depth_dpa_hist_fit_errors(
        self,
        axis: str,
        model: str,
    ) -> tuple[float, float, float, float]:
        """Load depth-dpa histogram fit errors from the database.

        Fit: asymmetric lorentzian function.

        Parameters
        ----------
        axis : str
            Axis along which the depth-dpa data is calculated ('x', 'y', or 'z').
        model : str
            Model name.

        Returns
        -------
        tuple[float, float, float, float]
            A tuple containing fit errors (x0, sigma, amplitude, asymmetry).
        """
        errors = self.read(
            table=f"depth_dpa_errors_{axis}",
            what="x0, sigma, amplitude, asymmetry",
            condition=f"WHERE model='{model}'",
        )
        return next(errors)

    def load_depth_dpa_hist_fit_functions(self, axis: str, model: str) -> Callable:
        """Load depth-dpa histogram fit function from the database.

        Parameters
        ----------
        axis : str
            Axis along which the depth-dpa data is calculated ('x', 'y', or 'z').
        model : str
            Model name.

        Returns
        -------
        Callable
            A function representing the fit.
        """
        x0, sigma, amplitude, asymmetry = self.load_depth_dpa_hist_fit_parameters(
            axis, model
        )
        return lambda x: lorentzian(x, x0, sigma, amplitude, asymmetry)

    def load_dpa(self, model: str) -> float:
        """Load dpa value from the database.

        Parameters
        ----------
        model : str
            Model name.

        Returns
        -------
        float
            Dpa value.
        """
        value = self.read(
            table="dpa", what="model, dpa", condition=f"WHERE model='{model}'"
        )
        return next(value)[1]

    # endregion
