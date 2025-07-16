"""This module contains the `Layer` class."""

from dataclasses import dataclass, field

from irradiapy.srimpy.target.element import Element


@dataclass
class Layer:
    """Class for defining a layer in SRIM simulations.

    Attributes
    ----------
    width : float
        The thickness of the layer in angstroms.
    phase : int
        The phase of the layer. 0 = solid, 1 = gas.
    density : float
        The density of the layer in g/cm^3.
    elements : list[Element]
        The list of elements in the layer.
    stoichs : list[float]
        The list of stoichiometries for the elements in the layer.
    bragg : int, optional (default=1)
        Stopping corrections for special bonding in compound targets.
    """

    width: float
    phase: int
    density: float
    elements: list[Element]
    stoichs: list[float]
    bragg: int = 1
    nelements: int = field(init=False)

    def __post_init__(self) -> None:
        self.nelements = len(self.elements)
