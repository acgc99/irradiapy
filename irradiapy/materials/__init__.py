"""Subpackage for materials."""

from irradiapy.enums import Phases
from irradiapy.materials.component import Component
from irradiapy.materials.element import Element

# region Elements

H = Element(
    atomic_number=1,
    atomic_weight=1.00784,
    symbol="H",
)
He = Element(
    atomic_number=2,
    atomic_weight=4.002602,
    symbol="He",
)
O = Element(
    atomic_number=8,
    atomic_weight=15.999,
    symbol="O",
)
Cr = Element(
    atomic_number=24,
    atomic_weight=51.9961,
    symbol="Cr",
    ed_avr=40.0,
    srim_el=7.8,
    srim_es=13.2,
)
Mn = Element(
    atomic_number=25,
    atomic_weight=54.938044,
    symbol="Mn",
)
Fe = Element(
    atomic_number=26,
    atomic_weight=55.845,
    symbol="Fe",
    ed_min=20.0,
    ed_avr=40.0,
    srim_el=5.8,
    srim_es=4.34,
    b_arc=-0.568,
    c_arc=0.286,
)
Cu = Element(
    atomic_number=29,
    atomic_weight=63.546,
    symbol="Cu",
    ed_min=25.0,
    ed_avr=33.0,
    b_arc=-0.68,
    c_arc=0.16,
)
Ag = Element(
    atomic_number=47,
    atomic_weight=107.87,
    symbol="Ag",
    ed_avr=39.0,
    srim_el=4.0,
    srim_es=2.97,
)
Hf = Element(
    atomic_number=72,
    atomic_weight=178.49,
    symbol="Hf",
)
Ta = Element(
    atomic_number=73,
    atomic_weight=180.94788,
    symbol="Ta",
)
W = Element(
    atomic_number=74,
    atomic_weight=183.84,
    symbol="W",
    ed_min=42.0,
    ed_avr=70.0,
    srim_el=13.2,
    srim_es=8.68,
    b_arc=-0.56,
    c_arc=0.12,
)

ELEMENTS = [H, He, O, Cr, Mn, Fe, Cu, Ag, Hf, Ta, W]
ELEMENT_BY_SYMBOL = {element.symbol: element for element in ELEMENTS}
ELEMENT_BY_ATOMIC_NUMBER = {
    element.atomic_number: element for element in ELEMENT_BY_SYMBOL.values()
}
ATOMIC_WEIGHT_BY_SYMBOL = {
    symbol: element.atomic_weight for symbol, element in ELEMENT_BY_SYMBOL.items()
}
ATOMIC_WEIGHT_BY_ATOMIC_NUMBER = {
    atomic_number: element.atomic_weight
    for atomic_number, element in ELEMENT_BY_ATOMIC_NUMBER.items()
}
ATOMIC_NUMBER_BY_SYMBOL = {
    symbol: element.atomic_number for symbol, element in ELEMENT_BY_SYMBOL.items()
}

# endregion

# region Components

Fe_bcc = Component(
    elements=(Fe,),
    stoichs=(1.0,),
    name="Iron bcc",
    phase=Phases.SOLID,
    density=7.8658,
    ax=2.87,
    structure="bcc",
    ed_min=20.0,
    ed_avr=40.0,
    b_arc=-0.568,
    c_arc=0.286,
    srim_el=5.8,
    srim_es=4.34,
)
W_bcc = Component(
    elements=(W,),
    stoichs=(1.0,),
    name="Tungsten bcc",
    phase=Phases.SOLID,
    density=19.3,
    ax=3.1652,
    structure="bcc",
    ed_min=42.0,
    ed_avr=70.0,
    b_arc=-0.56,
    c_arc=0.12,
    srim_el=13.2,
    srim_es=8.68,
)

# endregion
