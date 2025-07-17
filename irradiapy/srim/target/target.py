"""TThis module contains the `Target` class."""

from dataclasses import dataclass, field

from irradiapy.srim.target.layer import Layer


@dataclass
class Target:
    """Class for defining the target material in SRIM simulations.

    Attributes
    ----------
    layers : list
        The list of layers in the target material.
    nelements : int
        The total number of elements in the target material.
    nlayers : int
    """

    layers: list[Layer]
    nelements: int = field(init=False)
    nlayers: int = field(init=False)

    def __post_init__(self) -> None:
        """Counts the number of layers and elements in the material after initialization."""
        self.nlayers = len(self.layers)
        self.nelements = sum(len(layer.elements) for layer in self.layers)

    def trimin_description(self) -> str:
        """Return the target material description in the TRIMIN format."""
        string = (
            "target material, number of elements, layers\n"
            f'"material" {self.nelements} {self.nlayers}'
        )
        return string

    def trimin_target_elements(self) -> str:
        """Return the target elements in the TRIMIN format."""
        string = "target element, Z, mass\n"
        for i, layer in enumerate(self.layers):
            for j, element in enumerate(layer.elements):
                string += (
                    f"Atom {i*len(layer.elements)+j+1} = {element.symbol} "
                    f"=      {element.atomic_number} {element.atomic_mass}\n"
                )
        return string

    def trimin_target_layers(self) -> str:
        """Return the target layers in the TRIMIN format."""
        string = (
            "layer name, width density elements\nnumb. desc. (ang) (g/cm3) stoich.\n"
        )
        layers_info = []
        for i, layer in enumerate(self.layers):
            prev_stoichs = [0.0] * sum(len(l.elements) for l in self.layers[:i])
            next_stoichs = [0.0] * sum(len(l.elements) for l in self.layers[i + 1 :])
            layer_info = f'{i} "layer{i}" {layer.width} {layer.density} '
            layer_info += " ".join(
                map(str, prev_stoichs + layer.stoichs + next_stoichs)
            )
            layers_info.append(layer_info)
        return string + "\n".join(layers_info)

    def trimin_phases(self) -> str:
        """Return the phases of the layers in the TRIMIN format."""
        string = "phases\n"
        string += " ".join(map(str, (layer.phase for layer in self.layers)))
        return string

    def trimin_bragg(self) -> str:
        """Return the Bragg corrections in the TRIMIN format."""
        string = "target compound corrections (bragg)\n"
        string += " ".join(map(str, (layer.bragg for layer in self.layers)))
        return string

    def trimin_displacement(self) -> str:
        """Return the displacement energies in the TRIMIN format."""
        string = "displacement energies (eV)\n"
        string += " ".join(
            map(
                str,
                (element.e_d for layer in self.layers for element in layer.elements),
            )
        )
        return string

    def trimin_lattice(self) -> str:
        """Return the lattice binding energies in the TRIMIN format."""
        string = "lattice binding energies (eV)\n"
        string += " ".join(
            map(
                str,
                (element.e_l for layer in self.layers for element in layer.elements),
            )
        )
        return string

    def trimin_surface(self) -> str:
        """Return the surface binding energies in the TRIMIN format."""
        string = "surface binding energies (eV)\n"
        string += " ".join(
            map(
                str,
                (element.e_s for layer in self.layers for element in layer.elements),
            )
        )
        return string
