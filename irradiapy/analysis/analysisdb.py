"""This module contains the `AnalysisDB` class."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from irradiapy.database import Database
from irradiapy.utils.math import gaussian, lorentzian, power_law


@dataclass
class AnalysisDB(Database):
    """SQLite database for storing analysis results."""

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
            f"CREATE TABLE IF NOT EXISTS {table_name} (energy_centers REAL, counts REAL)"
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
        data = self.read_numpy(table="recoil_energies", what="energy_centers, counts")
        return data["energy_centers"], data["counts"]

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
        data = self.read_numpy(table=table_name, what="depth_centers, counts")
        return data["depth_centers"], data["counts"]

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
        cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (depth_centers REAL)")
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
        data = self.read_numpy(table=table_name, what="depth_centers, " + model)
        return data["depth_centers"], data[model]

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
            conditions=f"WHERE model='{model}'",
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
            conditions=f"WHERE model='{model}'",
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
            table="dpa", what="model, dpa", conditions=f"WHERE model='{model}'"
        )
        return next(value)[1]

    # endregion

    # region Clustering fraction

    def save_clustering_fraction_hist(
        self,
        axis: str,
        min_size: int,
        depth_centers: npt.NDArray[np.float64],
        ifraction: npt.NDArray[np.float64],
        vfraction: npt.NDArray[np.float64],
    ) -> None:
        """Save clustering fraction histogram into the database.

        Parameters
        ----------
        axis : str
            Axis along which the clustering fraction data is calculated ('x', 'y', or 'z').
        min_size : int
            Minimum cluster size to be considered clustered.
        depth_centers : npt.NDArray[np.float64]
            Centers of the depth bins.
        ifraction : npt.NDArray[np.float64]
            SIA clustering fraction values.
        vfraction : npt.NDArray[np.float64]
            Vacancy clustering fraction values.
        """
        table_name = f"clustering_fraction_{axis}_{min_size}"
        cur = self.cursor()
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} "
            "(depth_centers REAL, ifraction REAL, vfraction REAL)"
        )
        cur.executemany(
            f"INSERT INTO {table_name} (depth_centers, ifraction, vfraction) VALUES (?, ?, ?)",
            zip(depth_centers, ifraction, vfraction),
        )
        self.commit()
        cur.close()

    def load_clustering_fraction_hist(
        self,
        axis: str,
        min_size: int,
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Load clustering fraction histogram from the database.

        Parameters
        ----------
        axis : str
            Axis along which the clustering fraction data is calculated ('x', 'y', or 'z').
        min_size : int
            Minimum cluster size to be considered clustered.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
            A tuple containing depth centers and clustering fractions for SIAs and vacancies.
        """
        table_name = f"clustering_fraction_{axis}_{min_size}"
        data = self.read_numpy(
            table=table_name, what="depth_centers, ifraction, vfraction"
        )
        return data["depth_centers"], data["ifraction"], data["vfraction"]

    # endregion

    # region Cluster size scaling law

    def save_cluster_size_hist(
        self,
        vacancies: bool,
        size_centers: npt.NDArray[np.float64],
        hist: npt.NDArray[np.float64],
    ) -> None:
        """Save cluster size histogram into the database.

        Parameters
        ----------
        vacancies : bool
            Whether the histogram is for vacancies or SIAs.
        size_centers : npt.NDArray[np.float64]
            Centers of the size bins.
        hist : npt.NDArray[np.float64]
            Histogram values.
        """
        table_name = "cluster_size_vacs" if vacancies else "cluster_size_sias"
        cur = self.cursor()
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} (size_centers REAL, counts REAL)"
        )
        cur.executemany(
            f"INSERT INTO {table_name} (size_centers, counts) VALUES (?, ?)",
            zip(size_centers, hist),
        )
        self.commit()
        cur.close()

    def save_cluster_size_hist_fit_params(
        self,
        vacancies: bool,
        small: bool,
        a: float,
        k: float,
    ) -> None:
        """Save cluster size histogram fit parameters into the database.

        Fit: power law function.

        Parameters
        ----------
        vacancies : bool
            Whether the histogram is for vacancies or SIAs.
        small : bool
            Whether the fit is for small or large sizes.
        a : float
            Fit parameter a.
        k : float
            Fit parameter k.
        """
        table_name = (
            "cluster_size_params_vacs" if vacancies else "cluster_size_params_sias"
        )
        size_range = "small" if small else "large"
        cur = self.cursor()
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} "
            "(size_range TEXT PRIMARY KEY, a REAL, k REAL)"
        )
        cur.execute(
            f"INSERT INTO {table_name} (size_range, a, k) VALUES (?, ?, ?)",
            (size_range, a, k),
        )
        self.commit()
        cur.close()

    def save_cluster_size_hist_fit_errors(
        self,
        vacancies: bool,
        small: bool,
        a: float,
        k: float,
    ) -> None:
        """Save cluster size histogram fit errors into the database.

        Fit: power law function.

        Parameters
        ----------
        vacancies : bool
            Whether the histogram is for vacancies or SIAs.
        small : bool
            Whether the fit is for small or large sizes.
        a : float
            Fit error for parameter a.
        k : float
            Fit error for parameter k.
        """
        table_name = (
            "cluster_size_errors_vacs" if vacancies else "cluster_size_errors_sias"
        )
        size_range = "small" if small else "large"
        cur = self.cursor()
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} "
            "(size_range TEXT PRIMARY KEY, a REAL, k REAL)"
        )
        cur.execute(
            f"INSERT INTO {table_name} (size_range, a, k) VALUES (?, ?, ?)",
            (size_range, a, k),
        )
        self.commit()
        cur.close()

    def load_cluster_size_hist(
        self,
        vacancies: bool,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Load cluster size histogram from the database.

        Parameters
        ----------
        vacancies : bool
            Whether the histogram is for vacancies or SIAs.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            A tuple containing size centers and histogram values.
        """
        table_name = "cluster_size_vacs" if vacancies else "cluster_size_sias"
        data = self.read_numpy(table=table_name, what="size_centers, counts")
        return data["size_centers"], data["counts"]

    def load_cluster_size_hist_fit_parameters(
        self,
        vacancies: bool,
        small: bool,
    ) -> tuple[float, float]:
        """Load cluster size histogram fit parameters from the database.

        Fit: power law function.

        Parameters
        ----------
        vacancies : bool
            Whether the histogram is for vacancies or SIAs.
        small : bool
            Whether the fit is for small or large sizes.

        Returns
        -------
        tuple[float, float]
            Fit parameters (a, k).
        """
        table_name = (
            "cluster_size_params_vacs" if vacancies else "cluster_size_params_sias"
        )
        size_range = "small" if small else "large"
        params = self.read(
            table=table_name,
            what="a, k",
            conditions=f"WHERE size_range='{size_range}'",
        )
        return next(params)

    def load_cluster_size_hist_fit_errors(
        self,
        vacancies: bool,
        small: bool,
    ) -> tuple[float, float]:
        """Load cluster size histogram fit errors from the database.

        Fit: power law function.

        Parameters
        ----------
        vacancies : bool
            Whether the histogram is for vacancies or SIAs.
        small : bool
            Whether the fit is for small or large sizes.

        Returns
        -------
        tuple[float, float]
            Fit errors (a, k).
        """
        table_name = (
            "cluster_size_errors_vacs" if vacancies else "cluster_size_errors_sias"
        )
        size_range = "small" if small else "large"
        errors = self.read(
            table=table_name,
            what="a, k",
            conditions=f"WHERE size_range='{size_range}'",
        )
        return next(errors)

    def load_cluster_size_hist_fit_function(
        self,
        vacancies: bool,
        small: bool,
    ) -> Callable:
        """Load cluster size histogram fit function from the database.

        Fit: power law function.

        Parameters
        ----------
        vacancies : bool
            Whether the histogram is for vacancies or SIAs.
        small : bool
            Whether the fit is for small or large sizes.

        Returns
        -------
        Callable
            A function representing the fit.
        """
        a, k = self.load_cluster_size_hist_fit_parameters(vacancies, small)
        return lambda x: power_law(x, a, k)

    # endregion
