"""Run collision cascades with the UTTM/EPH LAMMPS extension.

This example runs a series of collision cascades in a single script. The
simulation is divided into three stages: thermalisation, cascade, and rerun.

Thermalisation is a short simulation that brings the system to the desired
temperature.

The cascade stage is where the actual collision cascades are simulated. The
simulation is run in three phases with different time steps and dump cadences
to capture the fast dynamics at the beginning of the cascade and the slower
dynamics at the end, avoiding excessive output file sizes.

The rerun stage is optional and can be used to continue the simulation after
the cascade stage.

This example requires the external LAMMPS Python module built from the LAMMPS
source tree. It can be run in parallel with MPI using::

    mpiexec -n <ranks> python3 lammps_uttm_cascades.py
"""

# pylint: disable=invalid-name, too-many-locals

from copy import deepcopy
from pathlib import Path

import numpy as np

import irradiapy.lammps.commands as cmds
from irradiapy.lammps import Cascade

# External input files.
interatomic_potential = Path()
eph_beta = Path()

# Output and workflow stages. All stages are enabled for a fresh calculation.
output_dir = Path("./1kev")
do_thermalisation = True
do_cascades = True
do_rerun = True

# Basic simulation parameters
nsims = 1
# Skip these simulations.
# It must be a string with comma-separated integers or ranges, e.g. "1,3-5,7".
skip = ""
ion_energy = 1_000.0  # eV
lattice = "bcc"
a0 = 3.1652  # angstrom
nxyz = (34, 34, 34)  # bcc unit cells
half_nxyz = tuple(n // 2 for n in nxyz)
boundary_width = 2  # lattice units
temperature = 300.0  # K
seed = int(ion_energy)
atomic_symbols = ["W"]
atomic_numbers = [74]

# UTTM/EPH parameters
# The electron grid is larger than the ionic box to approximate bulk heat
# transport beyond the simulated atoms.
uttm_extent_scale = 4.0
uttm_c = 1.92e-7  # eV / K / angstrom^3
uttm_k = 0.1086  # eV / K / angstrom / ps
eph_steps_dumping = 5_000
uttm_nxyz = tuple(int(uttm_extent_scale * a0 * n // 25.0) for n in nxyz)
uttm_extent = (
    -uttm_extent_scale * half_nxyz[0] * a0,
    uttm_extent_scale * half_nxyz[0] * a0,
    -uttm_extent_scale * half_nxyz[1] * a0,
    uttm_extent_scale * half_nxyz[1] * a0,
    -uttm_extent_scale * half_nxyz[2] * a0,
    uttm_extent_scale * half_nxyz[2] * a0,
)

# Stage durations are expressed in ps because metal units are used.
max_time_thermalisation_ionic = 5.0
max_time_thermalisation_electronic = 10.0
steps_thermo_thermalisation = 1_000

max_time_cascade_fine1 = 0.22
dump_time_cascade_fine1 = max_time_cascade_fine1 / 4.0
steps_thermo_cascade_fine1 = 50

max_time_cascade_fine2 = 2.0
dump_time_cascade_fine2 = max_time_cascade_fine2 / 4.0
steps_thermo_cascade_fine2 = 50

max_time_cascade_coarse = 32.0
dump_time_cascade_coarse = 8.0
steps_thermo_cascade_coarse = 100

max_time_cascade_rerun = 60.0
dump_time_cascade_rerun = 10.0
steps_thermo_cascade_rerun = 100

# Commands shared by every new LAMMPS instance.
cmds_preamble = [
    cmds.Units(style="metal"),
    cmds.AtomStyle(style="atomic"),
    cmds.Boundary(x="p", y="p", z="p"),
    cmds.AtomModify(kw_vals={"map": "yes"}),
    cmds.Lattice(style=lattice, scale=a0),
]

cmd_region_full_material = cmds.Region(
    id="full_box",
    style="block",
    args=[
        -half_nxyz[0],
        half_nxyz[0],
        -half_nxyz[1],
        half_nxyz[1],
        -half_nxyz[2],
        half_nxyz[2],
    ],
)
cmd_region_inner_material = cmds.Region(
    id="inner_material",
    style="block",
    args=[
        -half_nxyz[0] + boundary_width,
        half_nxyz[0] - boundary_width,
        -half_nxyz[1] + boundary_width,
        half_nxyz[1] - boundary_width,
        -half_nxyz[2] + boundary_width,
        half_nxyz[2] - boundary_width,
    ],
)
cmds_regions = [cmd_region_full_material, cmd_region_inner_material]

cmds_create_atoms = [
    cmds.CreateBox(n=1, region_id="full_box"),
    cmds.CreateAtoms(type=1, style="region", args=["full_box"]),
]

cmd_group_full_box = cmds.Group(id="full_box", style="region", args=["full_box"])
cmd_group_inner_box = cmds.Group(
    id="inner_box", style="region", args=["inner_material"]
)
cmd_group_thermoboundary = cmds.Group(
    id="thermoboundary",
    style="subtract",
    args=["full_box", "inner_box"],
)
cmds_groups = [
    cmd_group_full_box,
    cmd_group_inner_box,
    cmd_group_thermoboundary,
]

# TODO: Restore the plain EAM configuration when the UTTM/EPH plugin no
# longer requires an increased cutoff distance.
# See: https://github.com/llnl/USER-EPH/issues/10
#
# cmds_interactions = [
#     cmds.PairStyle(style="eam/fs"),
#     cmds.PairCoeff(
#         i="*",
#         j="*",
#         args=[str(interatomic_potential), " ".join(atomic_symbols)],
#     ),
# ]
cmds_interactions = [
    cmds.PairStyle(
        style="hybrid/overlay",
        args=["zero", 7.0, "nocoeff", "eam/fs"],
    ),
    cmds.PairCoeff(i="*", j="*", args=["zero"]),
    cmds.PairCoeff(
        i="*",
        j="*",
        args=[
            "eam/fs",
            str(interatomic_potential),
            " ".join(atomic_symbols),
        ],
    ),
]

# Integrators, UTTM, and adaptive timestep.
cmd_fix_nvt = cmds.Fix(
    id="nvt",
    group_id="all",
    style="nvt",
    kw_vals={"temp": f"{temperature} {temperature} $(dt*100.0)"},
)
cmd_fix_eph = cmds.Fix(
    id="eph",
    group_id="all",
    style="eph",
    args=[
        seed,
        7,
        4,
        1.0,
        uttm_c,
        uttm_k,
        temperature,
        1,
        1,
        1,
        "none",
        eph_steps_dumping,
        "none",
        str(eph_beta),
        atomic_symbols[0],
    ],
)
cmd_fix_reset = cmds.Fix(
    id="reset",
    group_id="all",
    style="dt/reset",
    args=[10, 1e-5, 0.001, a0 / 100.0],
    kw_vals={"units": "box"},
)

cmd_compute_ke = cmds.Compute(id="ke", group_id="all", style="ke/atom")
cmd_compute_pe = cmds.Compute(id="pe", group_id="all", style="pe/atom")
cmds_computes = [cmd_compute_ke, cmd_compute_pe]
cmds_monitor_borders = cmds.CMonitorBorders(
    region_id="thermoboundary",
    compute_id="ke",
    n=100,
    kemax=10.0,
).collection

cmd_thermo_style_ionic = cmds.ThermoStyle(
    style="custom",
    args=[
        "step",
        "spcpu",
        "cpuremain",
        "dt",
        "time",
        "temp",
        "etotal",
        "ke",
        "pe",
    ],
)
cmd_thermo_style_eph = cmds.ThermoStyle(
    style="custom",
    args=[
        "step",
        "spcpu",
        "cpuremain",
        "dt",
        "time",
        "temp",
        "etotal",
        "ke",
        "pe",
        "f_eph[1]",
        "f_eph[2]",
    ],
)

cmd_dump_cascade_fine1 = cmds.Dump(
    id="dump",
    group_id="all",
    style="custom",
    n=1,
    file="simulation.xyz",
    attributes=[
        "id",
        "element",
        "x",
        "y",
        "z",
        "c_pe",
        "c_ke",
        "f_eph[1]",
        "f_eph[2]",
    ],
)
cmd_dump_cascade_modify_fine1 = cmds.DumpModify(
    id="dump",
    kw_vals={
        "element": " ".join(str(number) for number in atomic_numbers),
        "every/time": dump_time_cascade_fine1,
        "time": "yes",
        "first": "yes",
    },
)
cmd_dump_cascade_fine2 = deepcopy(cmd_dump_cascade_fine1)
cmd_dump_cascade_modify_fine2 = cmds.DumpModify(
    id="dump",
    kw_vals={
        "element": " ".join(str(number) for number in atomic_numbers),
        "every/time": dump_time_cascade_fine2,
        "time": "yes",
        "first": "no",
        "append": "yes",
    },
)
cmd_dump_cascade_coarse = deepcopy(cmd_dump_cascade_fine1)
cmd_dump_cascade_modify_coarse = cmds.DumpModify(
    id="dump",
    kw_vals={
        "element": " ".join(str(number) for number in atomic_numbers),
        "every/time": dump_time_cascade_coarse,
        "time": "yes",
        "first": "no",
        "append": "yes",
    },
)
cmd_dump_cascade_rerun = deepcopy(cmd_dump_cascade_fine1)
cmd_dump_cascade_modify_rerun = cmds.DumpModify(
    id="dump",
    kw_vals={
        "element": " ".join(str(number) for number in atomic_numbers),
        "every/time": dump_time_cascade_rerun,
        "time": "yes",
        "first": "no",
        "append": "yes",
    },
)

cmds_run_thermalisation_ionic = cmds.CRunTime(
    max_time=max_time_thermalisation_ionic
).collection
cmds_run_thermalisation_electronic = cmds.CRunTime(
    max_time=max_time_thermalisation_electronic
).collection
cmds_run_cascade_fine1 = cmds.CRunTime(max_time=max_time_cascade_fine1).collection
cmds_run_cascade_fine2 = cmds.CRunTime(max_time=max_time_cascade_fine2).collection
cmds_run_cascade_coarse = cmds.CRunTime(max_time=max_time_cascade_coarse).collection
cmds_run_cascade_rerun = cmds.CRunTime(max_time=max_time_cascade_rerun).collection

cmd_velocity_temperature = cmds.Velocity(
    group_id="all",
    style="create",
    args=[2.0 * temperature, seed],
    kw_vals={"loop": "geom", "mom": "yes", "rot": "yes"},
)

# Select PKAs from a one-lattice-constant spherical shell.
radius = nxyz[0] * a0 / 3.0
cmds_ions_grouping = [
    cmds.Region(
        id="ions0",
        style="sphere",
        args=[0, 0, 0, radius],
        kw_vals={"units": "box"},
    ),
    cmds.Group(id="ions0", style="region", args=["ions0"]),
    cmds.Region(
        id="ions1",
        style="sphere",
        args=[0, 0, 0, radius + a0],
        kw_vals={"units": "box"},
    ),
    cmds.Group(id="ions1", style="region", args=["ions1"]),
    cmds.Group(id="ions", style="subtract", args=["ions1", "ions0"]),
]

cmds_therma = [
    *cmds_regions,
    *cmds_create_atoms,
    *cmds_groups,
    cmd_fix_nvt,
    cmd_fix_reset,
    *cmds_interactions,
    cmd_velocity_temperature,
    cmds.Thermo(n=steps_thermo_thermalisation),
    cmd_thermo_style_ionic,
    *cmds_run_thermalisation_ionic,
    cmd_fix_nvt.delete(),
    cmd_fix_eph,
    cmd_thermo_style_eph,
    *cmds_run_thermalisation_electronic,
]
cmds_cascade = [
    *cmds_regions,
    *cmds_groups,
    *cmds_interactions,
    *cmds_computes,
    *cmds_monitor_borders,
    cmd_fix_eph,
    cmd_fix_reset,
    cmds.Thermo(n=steps_thermo_cascade_fine1),
    cmd_thermo_style_eph,
    cmd_dump_cascade_fine1,
    cmd_dump_cascade_modify_fine1,
    cmds.MBoostIon(),
    *cmds_run_cascade_fine1,
    cmds.Thermo(n=steps_thermo_cascade_fine2),
    cmd_dump_cascade_fine1.delete(),
    cmd_dump_cascade_fine2,
    cmd_dump_cascade_modify_fine2,
    *cmds_run_cascade_fine2,
    cmds.Thermo(n=steps_thermo_cascade_coarse),
    cmd_dump_cascade_fine2.delete(),
    cmd_dump_cascade_coarse,
    cmd_dump_cascade_modify_coarse,
    *cmds_run_cascade_coarse,
    cmd_dump_cascade_coarse.delete(),
]
cmds_rerun = [
    *cmds_regions,
    *cmds_groups,
    *cmds_interactions,
    *cmds_computes,
    *cmds_monitor_borders,
    cmd_fix_eph,
    cmd_fix_reset,
    cmds.Thermo(n=steps_thermo_cascade_rerun),
    cmd_thermo_style_eph,
    cmd_dump_cascade_rerun,
    cmd_dump_cascade_modify_rerun,
    *cmds_run_cascade_rerun,
    cmd_dump_cascade_rerun.delete(),
]

cascade = Cascade(
    dir_parent=output_dir,
    nsims=nsims,
    ion_energy=ion_energy,
    pka_target=np.zeros(3),
    atomic_symbols=atomic_symbols,
    seed=seed,
    uttm_nxyz=uttm_nxyz,
    uttm_extent=uttm_extent,
    cmds_preamble=cmds_preamble,
    cmds_ions_grouping=cmds_ions_grouping,
    cmds_therma=cmds_therma if do_thermalisation else None,
    cmds_cascade=cmds_cascade if do_cascades else None,
    cmds_rerun=cmds_rerun if do_rerun else None,
    skip=skip,
)

cascade.run()
