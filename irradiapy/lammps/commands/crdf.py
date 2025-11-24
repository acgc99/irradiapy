"""This module contains the CRDF class."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.compute import Compute
from irradiapy.lammps.commands.fix import Fix
from irradiapy.lammps.commands.group import Group
from irradiapy.lammps.commands.variable import Variable


@dataclass(kw_only=True)
class CRDF:
    """Class representing a collection of LAMMPS commands to calculate RDF only on atoms without given structure as CNA.

    Note
    ----
    This creates a compute called `kemax`, a variable called `kemaxboundary`, and a fix called
    `bordercheck`.
    """

    lattice: str
    nbins: str
    region_id: str
    compute_id: str
    n: str
    mode: str
    variable: Variable = field(init=False)
    group: Group = field(init=False)
    compute: Compute = field(init=False)
    fix: Fix = field(init=False)
    collection: tuple = field(init=False)

    def __post_init__(self) -> None:
        if self.lattice == "fcc":
            self.lattice = "1"
        elif self.lattice == "hcp":
            self.lattice = "2"
        elif self.lattice == "bcc":
            self.lattice = "3"
        elif self.lattice == "icosahedral":
            self.lattice = "4"
        elif self.lattice == "unknown":
            self.lattice = "5"
        else:
            raise ValueError(
                f"Invalid lattice type: {self.lattice}. Must be one of 'fcc', 'hcp', 'bcc', 'icosahedral', or 'unknown'."
            )
        if self.mode not in ["file", "append"]:
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be one of 'file' or 'append'."
            )
        self.lattice = str(self.lattice)
        self.nbins = str(self.nbins)
        self.region_id = str(self.region_id)
        self.compute_id = str(self.compute_id)
        self.n = str(self.n)
        self.mode = str(self.mode)
        self.variable = Variable(
            name="disordered",
            style="atom",
            args=[f"'c_{self.compute_id} != {self.lattice}'"],
        )
        self.group = Group(
            id="disordered",
            style="dynamic",
            args=[self.region_id, "var", "disordered"],
            kw_vals={"every": self.n},
        )
        self.compute = Compute(
            id="rdf",
            group_id="disordered",
            style="rdf",
            args=[self.nbins],
        )
        self.fix = Fix(
            id="rdf",
            group_id="disordered",
            style="ave/time",
            args=[self.n, "1", self.n, "c_rdf[*]"],
            kw_vals={self.mode: "none", "mode": "vector"},
        )
        self.collection = (self.variable, self.group, self.compute, self.fix)
        self.delete_collection = (
            self.fix.delete(),
            self.compute.delete(),
            self.group.delete(),
            self.variable.delete(),
        )
