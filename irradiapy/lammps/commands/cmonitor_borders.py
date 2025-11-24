"""This module contains the CMonitorBorders class."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.compute import Compute
from irradiapy.lammps.commands.fix import Fix
from irradiapy.lammps.commands.variable import Variable


@dataclass(kw_only=True)
class CMonitorBorders:
    """Class representing a collection of LAMMPS commands to monitor kinetic energy of the border atoms.

    Note
    ----
    This creates a compute called `kemax`, a variable called `kemaxboundary`, and a fix called
    `bordercheck`.
    """

    region_id: str
    compute_id: str
    n: str
    kemax: str
    compute: Compute = field(init=False)
    fix: Fix = field(init=False)
    variable: Variable = field(init=False)
    collection: tuple = field(init=False)

    def __post_init__(self) -> None:
        self.region_id = str(self.region_id)
        self.compute_id = str(self.compute_id)
        self.n = str(self.n)
        self.kemax = str(self.kemax)
        self.compute = Compute(
            id="kemax",
            group_id=self.region_id,
            style="reduce",
            args=["max", f"c_{self.compute_id}"],
        )
        self.variable = Variable(name="kemaxboundary", style="equal", args=["c_kemax"])
        self.fix = Fix(
            id="bordercheck",
            group_id=self.region_id,
            style="halt",
            args=[self.n, f"v_kemaxboundary > {self.kemax}"],
            kw_vals={"error": "hard", "message": "yes"},
        )
        self.collection = (self.compute, self.variable, self.fix)
        self.delete_collection = (
            self.compute.delete(),
            self.variable.delete(),
            self.fix.delete(),
        )
