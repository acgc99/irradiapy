"""Utilities to read and plot SPECTRA-PKA .out files."""

# pylint: disable=line-too-long

import re
from collections import defaultdict
from pathlib import Path
from typing import TextIO

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def read(file_path: Path) -> defaultdict:
    """Reads SPECTRA-PKA .out file.

    Parameters
    ----------
    file_path : Path
        Path to SPECTRA-PKA .out file.

    Returns
    -------
    defaultdict
        Dictionary containing the data from the .out file.
    """

    data = defaultdict(None)
    data["file"] = file_path.name

    file = open(file_path, "r", encoding="utf-8")

    index_pattern = re.compile(r"^\s*#{3,}\s*index\s+(\d+)\b", re.I)
    while True:
        pos = file.tell()
        line = file.readline()
        if index_pattern.match(line):
            index = int(index_pattern.match(line).group(1))
            after_hashes = line.rsplit("#", 1)[-1]
            label = re.sub(r"[()]", "", after_hashes)
            label = re.sub(r"\s+", " ", label).strip()
            # print(f"{index} {label}")
            if index == 0 and label == "":
                data["spectrum"] = __process_spectrum(file, index, label)
                continue
            line = file.readline().replace("#", "").strip()
            if "from" in line:
                line = line.replace("[", "").replace("]", "")
                file.readline()
                file.readline()
                line = file.readline()
                if line.startswith(" #"):
                    # print("TOTAL", line)
                    file.seek(pos)
                    data[index] = __process_total_matrix(file, line, index, label)
                else:
                    # print("CHANNEL", line)
                    file.seek(pos)
                    data[index] = __process_channel_matrix(file, line, index, label)
            else:
                if "elemental" in line:
                    # print("ELEMENTAL", line)
                    data[index] = __process_elemental_matrix(file, line, index, label)
                elif "[" not in line or "]" not in line:
                    # print("ISOTOPIC", line)
                    data[index] = __process_isotopic_matrix(file, line, index, label)
                else:
                    raise ValueError(f"Cannot parse line: {line}")
        if not line:
            break

    file.close()

    return data


def __process_isotopic_matrix(
    file: TextIO, line: str, index: int, label: str
) -> dict[str, npt.NDArray]:
    """Process an isotopic recoil matrix.

    Parameters
    ----------
    file : TextIO
        File object to read from.
    line : str
        Current line being processed.
    index : int
        Index of the matrix.
    label : str
        Label of the matrix.

    Returns
    -------
    dict[str, npt.NDArray]
        Dictionary containing the isotopic matrix data.

    Example
    -------
     ### index          212  ##### ( totals )
     #  recoil matrix of V52
     #PKA RECOIL DISTRIBUTIONS - per target atom
     #RECOIL energy (MeV low & high)  PKAs/s         norm_sum    cumulative_sum  tdam-pkas disp_energy_(eV/s)  NRT_dpa/s
     #(or T-dam energy low+high for
     #          tdam-pkas/disp/dpa)
    """
    isotope = line.split()[-1].replace("(", "").replace(")", "")
    file.readline()  # skip header
    file.readline()  # skip header
    file.readline()  # skip header
    file.readline()  # skip header
    recoil_lower, recoil_upper, recoil_pkas = [], [], []
    norm_sum, cum_sum, tdam_pkas = [], [], []
    disp_energy, nrt = [], []
    while True:
        line = file.readline()
        if line.startswith("#"):
            break
        parts = line.split()
        recoil_lower.append(float(parts[0]))
        recoil_upper.append(float(parts[1]))
        recoil_pkas.append(float(parts[2]))
        norm_sum.append(float(parts[3]))
        cum_sum.append(float(parts[4]))
        tdam_pkas.append(float(parts[5]))
        disp_energy.append(float(parts[6]))
    file.readline()  # skip final table
    file.readline()  # skip final table
    file.readline()  # skip blank line
    file.readline()  # skip blank line
    data = dict(
        type="isotopic",
        index=index,
        label=label,
        isotope=isotope,
        recoil_lower=np.array(recoil_lower),
        recoil_upper=np.array(recoil_upper),
        recoil_pkas=np.array(recoil_pkas),
        norm_sum=np.array(norm_sum),
        cum_sum=np.array(cum_sum),
        tdam_pkas=np.array(tdam_pkas),
        disp_energy=np.array(disp_energy),
        nrt=np.array(nrt),
    )
    return data


def __process_elemental_matrix(
    file: TextIO, line: str, index: int, label: str
) -> dict[str, npt.NDArray]:
    """Process an elemental recoil matrix.

    Parameters
    ----------
    file : TextIO
        File object to read from.
    line : str
        Current line being processed.
    index : int
        Index of the matrix.
    label : str
        Label of the matrix.

    Returns
    -------
    dict[str, npt.NDArray]
        Dictionary containing the elemental matrix data.

    Example
    -------
     ### index          223  ##### ( totals )
     #  elemental recoil matrix of H
     #PKA RECOIL DISTRIBUTIONS - per target atom
     #RECOIL energy (MeV low & high)  PKAs/s         norm_sum    cumulative_sum  tdam-pkas disp_energy_(eV/s)  NRT_dpa/s
     #(or T-dam energy low+high for
     #          tdam-pkas/disp/dpa)
    """
    element = line.split()[-1]
    file.readline()  # skip header
    file.readline()  # skip header
    file.readline()  # skip header
    file.readline()  # skip header
    recoil_lower, recoil_upper, recoil_pkas = [], [], []
    norm_sum, cum_sum, tdam_pkas = [], [], []
    disp_energy, nrt = [], []
    while True:
        line = file.readline()
        if line.startswith("#"):
            break
        parts = line.split()
        recoil_lower.append(float(parts[0]))
        recoil_upper.append(float(parts[1]))
        recoil_pkas.append(float(parts[2]))
        norm_sum.append(float(parts[3]))
        cum_sum.append(float(parts[4]))
        tdam_pkas.append(float(parts[5]))
        disp_energy.append(float(parts[6]))
    file.readline()  # skip final table
    file.readline()  # skip final table
    file.readline()  # skip blank line
    file.readline()  # skip blank line
    data = dict(
        type="elemental",
        index=index,
        label=label,
        element=element,
        recoil_lower=np.array(recoil_lower),
        recoil_upper=np.array(recoil_upper),
        recoil_pkas=np.array(recoil_pkas),
        norm_sum=np.array(norm_sum),
        cum_sum=np.array(cum_sum),
        tdam_pkas=np.array(tdam_pkas),
        disp_energy=np.array(disp_energy),
        nrt=np.array(nrt),
    )
    return data


def __process_total_matrix(
    file: TextIO, line: str, index: int, label: str
) -> dict[str, npt.NDArray]:
    """Process channel block.

    Parameters
    ----------
    file : TextIO
        File object to read from.
    line : str
        Current line being processed.
    index : int
        Index of the matrix.
    label : str
        Label of the matrix.

    Returns
    -------
    dict[str, npt.NDArray]
        Dictionary containing the total matrix data.

    Example
    -------
     ### index           41  ##### /mnt/f/Estudios/SPECTRA/TENDL2023n-pka/TENDL2023n-pka/Fe054s.asc
     #  total alpha matrix [ He4 ] from [ Fe54 ]
     #PKA RECOIL DISTRIBUTIONS
     #RECOIL energy (MeV low & high)  PKAs/s       norm_sum       tdam-pkas disp_energy_(eV/s)  NRT_dpa/s
     #(or T_dam energy low+high for
     #          tdam-pakas/disp/dpa)
    """
    file.readline()  # skip header
    line = file.readline().replace("#", "").strip().replace("[", "").replace("]", "")
    words = line.split()
    in_atom = words[-1]
    out_atom = words[-3]
    channel = words[0].replace(",", ", ")
    file.readline()  # skip header
    file.readline()  # skip header
    file.readline()  # skip header
    file.readline()  # skip header
    recoil_lower, recoil_upper, recoil_pkas = [], [], []
    norm_sum, tdam_pkas = [], []
    disp_energy, nrt = [], []
    while True:
        line = file.readline()
        if line.startswith("#"):
            break
        parts = line.split()
        recoil_lower.append(float(parts[0]))
        recoil_upper.append(float(parts[1]))
        recoil_pkas.append(float(parts[2]))
        norm_sum.append(float(parts[3]))
        tdam_pkas.append(float(parts[4]))
        disp_energy.append(float(parts[5]))
        nrt.append(float(parts[6]))
    file.readline()  # skip final table
    file.readline()  # skip final table
    file.readline()  # skip blank line
    file.readline()  # skip blank line
    data = dict(
        type="total",
        index=index,
        label=label,
        in_atom=in_atom,
        out_atom=out_atom,
        channel=channel,
        recoil_lower=np.array(recoil_lower),
        recoil_upper=np.array(recoil_upper),
        recoil_pkas=np.array(recoil_pkas),
        norm_sum=np.array(norm_sum),
        tdam_pkas=np.array(tdam_pkas),
        disp_energy=np.array(disp_energy),
        nrt=np.array(nrt),
    )
    return data


def __process_channel_matrix(
    file: TextIO, line: str, index: int, label: str
) -> dict[str, npt.NDArray]:
    """Process a recoil channel matrix.

    Parameters
    ----------
    file : TextIO
        File object to read from.
    line : str
        Current line being processed.
    index : int
        Index of the matrix.
    label : str
        Label of the matrix.

    Returns
    -------
    dict[str, npt.NDArray]
        Dictionary containing the channel matrix data.

    Example
    -------
     ### index            7  ##### /mnt/f/Estudios/SPECTRA/TENDL2023n-pka/TENDL2023n-pka/Fe054s.asc
     #             (n,na) alpha matrix [ He4 ] from [ Fe54 ]
     #PKA RECOIL DISTRIBUTIONS
     #RECOIL energy (MeV low & high)    PKAs     norm_sum  T_dam (MeV low & high)      disp_energy (eV/s)   NRT_dpa/s
    """
    file.readline()  # skip header
    line = file.readline().replace("#", "").strip().replace("[", "").replace("]", "")
    words = line.split()
    in_atom = words[-1]
    out_atom = words[-3]
    channel = words[0].replace(",", ", ")
    file.readline()  # skip header
    file.readline()  # skip header
    recoil_lower, recoil_upper, recoil_pkas = [], [], []
    norm_sum, tdam_lower, tdam_upper = [], [], []
    disp_energy, nrt = [], []
    while True:
        line = file.readline()
        if line.startswith("#"):
            break
        parts = line.split()
        recoil_lower.append(float(parts[0]))
        recoil_upper.append(float(parts[1]))
        recoil_pkas.append(float(parts[2]))
        norm_sum.append(float(parts[3]))
        tdam_lower.append(float(parts[4]))
        tdam_upper.append(float(parts[5]))
        disp_energy.append(float(parts[6]))
        nrt.append(float(parts[7]))
    file.readline()  # skip final table
    file.readline()  # skip final table
    file.readline()  # skip blank line
    file.readline()  # skip blank line
    data = dict(
        type="channel",
        index=index,
        label=label,
        in_atom=in_atom,
        out_atom=out_atom,
        channel=channel,
        recoil_lower=np.array(recoil_lower),
        recoil_upper=np.array(recoil_upper),
        recoil_pkas=np.array(recoil_pkas),
        norm_sum=np.array(norm_sum),
        tdam_lower=np.array(tdam_lower),
        tdam_upper=np.array(tdam_upper),
        disp_energy=np.array(disp_energy),
        nrt=np.array(nrt),
    )
    return data


def __process_spectrum(file: TextIO, index: int, label: str) -> dict[str, npt.NDArray]:
    """Process spectrum block.

    Parameters
    ----------
    file : TextIO
        File object to read from.
    index : int
        Index of the matrix.
    label : str
        Label of the matrix.

    Returns
    -------
    dict[str, npt.NDArray]
        Dictionary containing the spectrum data.
    """
    name = file.readline().strip().replace("#", "")
    file.readline()  # skip header
    lower, upper, flux = [], [], []
    while True:
        line = file.readline()
        if line.startswith("#"):
            words = line.split()
            total_flux = float(words[4])
            total_fluence = float(words[7])
            break
        parts = line.split()
        lower.append(float(parts[0]))
        upper.append(float(parts[1]))
        flux.append(float(parts[2]))
    file.readline()  # skip blank line
    file.readline()  # skip blank line
    data = dict(
        type="spectrum",
        index=index,
        label=label,
        name=name,
        total_flux=total_flux,
        total_fluence=total_fluence,
        lower=np.array(lower),
        upper=np.array(upper),
        flux=np.array(flux),
    )
    return data


def plot_all(data: defaultdict, out_path: Path) -> None:
    """Plot all matrices from SPECTRA-PKA .out using data from irradiapy.spectrapka.output.read
    function.

    Parameters
    ----------
    data : defaultdict
        Data dictionary from irradiapy.spectrapka.output.read function.
    out_path : Path
        Output path to save the plots.
    """
    out_path.mkdir(parents=True, exist_ok=True)

    # Plot spectrum
    spectrum = data["spectrum"]
    lower, upper, flux = spectrum["lower"], spectrum["upper"], spectrum["flux"]
    edges = np.concatenate((lower, [upper[-1]])) * 1e6
    lethargy = np.log(edges[1:] / edges[:-1])
    flux /= lethargy  # Convert to per lethargy interval

    fig, ax = plt.subplots()
    ax.stairs(flux, edges)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Neutron flux (n/cm²/s) per lethargy interval", fontsize=12)
    ax.set_title(f"Spectrum: {spectrum['name']}")
    fig.tight_layout()
    fig.savefig(out_path / "spectrum.png", dpi=300)
    plt.close(fig)

    # Plot recoil matrices
    for key, matrix in data.items():
        if key == "file" or matrix["type"] == "spectrum":
            continue

        if not matrix["label"]:
            plot_path = out_path / f"matrix_{matrix['index']}.png"
        else:
            plot_path = (
                out_path / Path(matrix["label"]).name / f"matrix_{matrix['index']}.png"
            )
            plot_path.parent.mkdir(parents=True, exist_ok=True)

        lower = matrix["recoil_lower"]
        upper = matrix["recoil_upper"]
        pkas = matrix["recoil_pkas"]
        edges = np.concatenate((lower, [upper[-1]])) * 1e6

        fig, ax = plt.subplots()
        ax.stairs(pkas, edges)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Recoil energy (eV)")
        ax.set_ylabel("PKAs/s")
        if matrix["type"] == "channel":
            ax.set_title(
                f"Recoil matrix of: {matrix['in_atom']}{matrix['channel']}{matrix['out_atom']}"
            )
        elif matrix["type"] == "total":
            ax.set_title(
                f"Total recoil matrix of: {matrix['in_atom']}{matrix['channel']}{matrix['out_atom']}"
            )
        elif matrix["type"] == "isotopic":
            ax.set_title(f"Isotopic recoil matrix of {matrix['isotope']}")
        else:
            ax.set_title(f"Elemental recoil matrix of {matrix['element']}")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
