"""Class for cascade simulations."""

# pylint: disable=line-too-long

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from lammps import lammps

import irradiapy.lammps.commands as cmds
from irradiapy import utils
from irradiapy.io.lammpslogreader import LAMMPSLogReader
from irradiapy.lammps.cascadebase import CascadeBase


@dataclass(kw_only=True)
class Cascade(CascadeBase):
    """Class for cascade simulations.

    Data files will automatically be generated.
    You might use multiple dump files, an index will be added to the file name.
    Rerunning requires the dt, time and timestep from the thermo output of the previous run.

    Attributes
    ----------
    ion_energy : float
        The energy of the PKA ion.
    pka_target : npt.NDArray[np.float64]
        The target position of the PKA. PKAs will be directed towards this position.
    cmds_ions_grouping : list[cmds.Command]
        The list of commands for ion grouping. PKAs will be randomly selected from this group.
    """

    ion_energy: float
    pka_target: npt.NDArray[np.float64]
    cmds_ions_grouping: list[cmds.Command]

    @utils.mpi.mpi_safe_method
    def run(self) -> None:
        """Run the bulk cascade simulations."""
        # Initialise directory
        self.dir_parent.mkdir(parents=True, exist_ok=True)
        # Thermalisation
        if self.cmds_therma:
            path_data_therma = self.dir_parent / "simulation.data"
            path_log_therma = self.dir_parent / "simulation.log"
            if self.rank == 0 and path_data_therma.exists():
                raise FileExistsError("Thermalisation already done.")
            self.comm.Barrier()
            try:
                lmp = lammps(comm=self.comm)
                self.__run_thermalisation(lmp)
            except Exception as e:
                raise e
            finally:
                lmp.close()
                utils.mpi.mv_file(self.path_log_cwd, path_log_therma, comm=self.comm)
        if self.cmds_cascade:
            # Select ions for cascades
            try:
                lmp = lammps(comm=self.comm)
                (
                    ions_ids,
                    ions_atomic_numbers,
                    ions_positions,
                    ions_speeds,
                    ions_velocities,
                    ions_phis,
                    ions_thetas,
                ) = self.__select_ions_for_cascades(lmp)
            except Exception as e:
                raise e
            finally:
                lmp.close()
                utils.mpi.rm_file(self.path_log_cwd, comm=self.comm)
            # Run cascades
            for idx in range(self.nsims):
                dir_cascade = self.dir_parent / f"simulation_{idx+1}"
                path_log_cascade = dir_cascade / f"simulation_{idx+1}.log"
                if idx + 1 in self.skip:
                    continue
                if self.rank == 0 and dir_cascade.exists():
                    raise FileExistsError(f"Cascade {idx+1} already exists.")
                self.comm.Barrier()
                try:
                    lmp = lammps(comm=self.comm)
                    self.__run_cascade(
                        lmp,
                        idx + 1,
                        ions_ids[idx],
                        ions_atomic_numbers[idx],
                        ions_positions[idx],
                        ions_speeds[idx],
                        ions_velocities[idx],
                        ions_phis[idx],
                        ions_thetas[idx],
                    )
                except Exception as e:
                    raise e
                finally:
                    lmp.close()
                    utils.mpi.mv_file(
                        self.path_log_cwd, path_log_cascade, comm=self.comm
                    )
        if self.cmds_rerun:
            for idx in range(self.nsims):
                dir_cascade = self.dir_parent / f"simulation_{idx+1}"
                path_log_cascade = dir_cascade / f"simulation_{idx+1}.log"
                if idx + 1 in self.skip:
                    continue
                if self.rank == 0 and not dir_cascade.exists():
                    raise FileExistsError(
                        f"Cascade {idx+1} does not exist, impossible to rerun."
                    )
                self.comm.Barrier()
                path_data_cascade = dir_cascade / f"simulation_{idx+1}.data"
                timestep0, t0 = None, None
                if self.rank == 0:
                    # extract timestep from data file
                    data_file = open(path_data_cascade, "r", encoding="utf-8")
                    line = data_file.readline()
                    data_file.close()
                    timestep0 = int(line.split("timestep = ")[1].split(",")[0])
                    # extract timestep, time and dt from log file
                    log_path = dir_cascade / f"simulation_{idx+1}.log"
                    last_log = utils.io.get_last_reader(LAMMPSLogReader(log_path))[
                        "thermo"
                    ]
                    log_timestep = int(last_log["Step"][-1])
                    log_time = float(last_log["Time"][-1])
                    log_dt = float(last_log["Dt"][-1])
                    t0 = (timestep0 - log_timestep) * log_dt + log_time
                (timestep0, t0) = self._broadcast_variables(timestep0, t0)
                try:
                    lmp = lammps(comm=self.comm)
                    self.__rerun(
                        lmp,
                        idx + 1,
                        timestep0,
                        t0,
                    )
                except Exception as e:
                    raise e
                finally:
                    lmp.close()
                    utils.mpi.ap_rm_file(
                        self.path_log_cwd, path_log_cascade, comm=self.comm
                    )

        if self.finalize:
            lmp.finalize()

    def __run_thermalisation(self, lmp: lammps) -> None:
        """Run the thermalisation.

        Parameters
        ----------
        lmp : lammps
            LAMMPS instance.
        """
        cmds_therma = copy.deepcopy(self.cmds_therma)
        for cmd in cmds_therma:
            if isinstance(cmd, cmds.Dump):
                cmd.file = self.dir_parent / f"simulation{Path(cmd.file).suffix}"
            if isinstance(cmd, cmds.Fix):
                if cmd.style == "eph":
                    self.eph_c = cmd.args[4]
                    self.eph_k = cmd.args[5]
                    self.eph_temperature = cmd.args[6]
                    cmd.args[10] = self.dir_parent / "egrid0"
                    cmd.args[12] = self.dir_parent / "egrid/egrid"
                    (self.dir_parent / "egrid").mkdir(parents=True, exist_ok=True)
                    self._generate_egrid()
        self.comm.Barrier()
        cmd_write = cmds.WriteData(file=self.dir_parent / "simulation.data")
        self.exec_cmds(lmp, self.cmds_preamble + cmds_therma + [cmd_write])

    def __select_ions_for_cascades(self, lmp: lammps) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Select ions for cascades, generate random velocities, and broadcast their properties."""
        cmd_read = cmds.ReadData(file=self.dir_parent / "simulation.data")
        self.exec_cmds(lmp, self.cmds_preamble + [cmd_read] + self.cmds_ions_grouping)
        # Collect ions properties (full group)
        ions_group = "all"
        for cmd in self.cmds_ions_grouping:
            if isinstance(cmd, cmds.Group):
                ions_group = cmd.id
        ions_properties = utils.lammps.get_properties_from_group(
            lmp, ions_group, ["id", "type", "x"]
        )  # for each process this variable is different
        ions_ids = ions_atomic_numbers = ions_positions = ions_speeds = None
        ions_velocities = ions_phis = ions_thetas = None
        # This is a list of dictionaries, one for each process
        ions_properties_gathered = self.comm.gather(ions_properties, root=0)
        if self.rank == 0:
            # Gather ions properties from all processes into a single dictionary
            ions_properties = {
                key: np.concatenate([g[key] for g in ions_properties_gathered])
                for key in ions_properties
            }
            # Reorder by id in ascending order for reproducibility
            order = np.argsort(ions_properties["id"])
            ions_properties = {
                "id": ions_properties["id"][order],
                "type": ions_properties["type"][order],
                "x": ions_properties["x"][order],
            }
            ions_properties["atomic_number"] = self.types_to_atomic_numbers(
                ions_properties["type"]
            )
            # Select random ions and generate random velocities and angles
            idxs = self._rng.integers(0, ions_properties["id"].size, size=self.nsims)
            ions_ids = ions_properties["id"][idxs]
            ions_atomic_numbers = ions_properties["atomic_number"][idxs]
            ions_positions = ions_properties["x"][idxs]
            ions_speeds = self._calculate_atoms_speeds(
                np.full(self.nsims, self.ion_energy), ions_atomic_numbers
            )
            # The ion direction is towards the center of the box
            dirxs = self.pka_target[0] - ions_positions[:, 0]
            dirys = self.pka_target[1] - ions_positions[:, 1]
            dirzs = self.pka_target[2] - ions_positions[:, 2]
            directions = np.stack([dirxs, dirys, dirzs], axis=1)
            # Normalize directions
            directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
            # Get actual velocities
            ions_velocities = ions_speeds[:, np.newaxis] * directions
            # Get polar and azimuthal angles
            ions_thetas = np.arccos(directions[:, 2])  # polar angle
            ions_phis = np.arctan2(
                directions[:, 1], directions[:, 0]
            )  # azimuthal angle
        self.comm.Barrier()
        (
            ions_ids,
            ions_atomic_numbers,
            ions_positions,
            ions_speeds,
            ions_velocities,
            ions_phis,
            ions_thetas,
        ) = self._broadcast_variables(
            ions_ids,
            ions_atomic_numbers,
            ions_positions,
            ions_speeds,
            ions_velocities,
            ions_phis,
            ions_thetas,
        )
        return (
            ions_ids,
            ions_atomic_numbers,
            ions_positions,
            ions_speeds,
            ions_velocities,
            ions_phis,
            ions_thetas,
        )

    def __run_cascade(
        self,
        lmp: lammps,
        idx: int,
        ion_id: np.integer,
        ion_atomic_number: np.integer,
        ion_position: np.ndarray,
        ion_speed: float,
        ion_velocity: np.ndarray,
        ion_phi: float,
        ion_theta: float,
    ) -> None:
        """Run the bulk irradiation.

        Parameters
        ----------
        lmp : lammps
            LAMMPS instance.
        idx : int
            Index of the simulation.
        ion_id : int
            Ion ID.
        ion_atomic_number : int
            Ion atomic number.
        ion_position : np.ndarray
            Ion position.
        ion_speed : float
            Ion speed.
        ion_velocity : np.ndarray
            Ion velocity.
        ion_phi : float
            Ion azimuthal angle.
        ion_theta : float
            Ion polar angle.
        """
        dir_cascade = self.dir_parent / f"simulation_{idx}"
        dir_cascade.mkdir(parents=True, exist_ok=True)
        cmds_cascade = copy.deepcopy(self.cmds_cascade)
        # Handle log
        # Reading and writing data/restart files
        cmd_read = cmds.ReadData(file=self.dir_parent / "simulation.data")
        cmd_write = cmds.WriteData(file=dir_cascade / f"simulation_{idx}.data")
        # Handle dump and fix commands
        for cmd in cmds_cascade:
            if isinstance(cmd, cmds.Dump):
                path = Path(cmd.file)
                cmd.file = dir_cascade / f"{path.stem}_{idx}{path.suffix}"
            if isinstance(cmd, cmds.Fix) and cmd.id == "rdf":
                cmd.kw_vals["file"] = dir_cascade / f"rdf_{idx}.rdf"
            if isinstance(cmd, cmds.Fix) and cmd.style == "eph":
                path_egrid_cascade = dir_cascade / "egrid"
                path_egrid_cascade.mkdir(parents=True, exist_ok=True)
                cmd.args[10] = self.dir_parent / "egrid0.restart"
                cmd.args[12] = path_egrid_cascade / "egrid"
        # Print ion properties
        cmds_print = [
            cmds.Print(string="'Ion boost:'"),
            cmds.Print(string=f"'  ID: {ion_id}'"),
            cmds.Print(string=f"'  Element: {ion_atomic_number}'"),
            cmds.Print(
                string=f"'  Position: ({ion_position[0]}, {ion_position[1]}, {ion_position[2]}) angstrom'"
            ),
            cmds.Print(string=f"'  Energy: {self.ion_energy} eV'"),
            cmds.Print(string=f"'  Speed: {ion_speed} angstrom/ps'"),
            cmds.Print(
                string=f"'  Velocity: ({ion_velocity[0]}, {ion_velocity[1]}, {ion_velocity[2]}) angstrom/ps'"
            ),
            cmds.Print(string=f"'  Polar angle (theta): {ion_theta} rad'"),
            cmds.Print(string=f"'  Azimuthal angle (phi): {ion_phi} rad'"),
        ]
        # Replace ion boosting commands
        for i, cmd in enumerate(cmds_cascade):
            if isinstance(cmd, cmds.MBoostIon):
                cmds_cascade[i] = cmds.Group(
                    id="ion",
                    style="id",
                    args=[ion_id],
                )
                cmd_velocity = cmds.Velocity(
                    group_id="ion",
                    style="set",
                    args=[ion_velocity[0], ion_velocity[1], ion_velocity[2]],
                    kw_vals={"units": "box"},
                )
                cmds_cascade.insert(i + 1, cmd_velocity)
        cmds_cascade = [
            *self.cmds_preamble,
            cmd_read,
            cmds.ResetTimestep(n=0),
            *cmds_print,
            *cmds_cascade,
            cmd_write,
        ]
        self.comm.Barrier()
        self.exec_cmds(lmp, cmds_cascade)
        if self._eph:
            utils.mpi.mv_file(
                self.dir_parent / "egrid0.restart.restart",
                dir_cascade / "egrid0.restart",
                comm=self.comm,
            )

    def __rerun(
        self,
        lmp: lammps,
        idx: int,
        timestep0: int,
        t0: float,
    ) -> None:
        """Run the bulk irradiation.

        Parameters
        ----------
        lmp : lammps
            LAMMPS instance.
        idx : int
            Index of the simulation.
        timestep0 : int
            Initial timestep of the simulation.
        t0 : float
            Initial time of the simulation.
        """
        dir_cascade = self.dir_parent / f"simulation_{idx}"
        cmds_rerun = copy.deepcopy(self.cmds_rerun)
        # Reading and writing data/restart files
        path_data_cascade = dir_cascade / f"simulation_{idx}.data"
        cmd_read = cmds.ReadData(file=path_data_cascade)
        cmd_write = cmds.WriteData(file=path_data_cascade)
        # Handle dump and fix commands
        for cmd in cmds_rerun:
            if isinstance(cmd, cmds.Dump):
                path = Path(cmd.file)
                cmd.file = dir_cascade / f"{path.stem}_{idx}{path.suffix}"
            if isinstance(cmd, cmds.Fix) and cmd.style == "eph":
                cmd.args[10] = dir_cascade / "egrid0.restart"
                cmd.args[12] = dir_cascade / "egrid/egrid"
        cmds_rerun = [
            *self.cmds_preamble,
            cmd_read,
            cmds.ResetTimestep(n=timestep0, kw_vals={"time": t0}),
            *cmds_rerun,
            cmd_write,
        ]
        self.comm.Barrier()
        self.exec_cmds(lmp, cmds_rerun)
        if self._eph:
            utils.mpi.mv_file(
                dir_cascade / "egrid0.restart.restart",
                dir_cascade / "egrid0.restart",
                comm=self.comm,
            )
