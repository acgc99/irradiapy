"""This module contains the CRunTime class."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.fix import Fix
from irradiapy.lammps.commands.run import Run
from irradiapy.lammps.commands.timer import Timer
from irradiapy.lammps.commands.variable import Variable


@dataclass(kw_only=True)
class CRunTime:
    """Class representing a collection of LAMMPS commands to run a simulation up to certain time.

    Note
    ----
    This creates a variable called `simtime` and a fix called `cruntime`.
    """

    max_time: str
    max_steps: str = "1000000"
    run_kw_vals: dict[str, str] = field(default_factory=dict)
    variable: Variable = field(default=None, init=False)
    fix: Fix = field(default=None, init=False)
    timer: Timer = field(default=None, init=False)
    run: Run = field(default=None, init=False)
    collection: tuple = field(default_factory=tuple, init=False)

    def __post_init__(self) -> None:
        self.max_time = str(self.max_time)
        self.max_steps = str(self.max_steps)
        self.run_kw_vals = {
            str(key): str(value) for key, value in self.run_kw_vals.items()
        }
        self.variable = Variable(name="simtime", style="equal", args=["time"])
        self.fix = Fix(
            id="cruntime",
            group_id="all",
            style="halt",
            args=["1", f"v_simtime > {self.max_time}"],
            kw_vals={"error": "soft", "message": "yes"},
        )
        self.timer = Timer(args=["timeout", "off"])
        self.run = Run(n=self.max_steps, kw_vals=self.run_kw_vals)
        self.collection = (
            self.variable,
            self.fix,
            self.timer,
            self.run,
            self.fix.delete(),
            self.variable.delete(),
        )
