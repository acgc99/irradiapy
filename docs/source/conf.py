# pylint: disable=missing-module-docstring,invalid-name,redefined-builtin

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "irradiapy"
copyright = "2025, Abel Carlos Gutiérrez Camacho"
author = "Abel Carlos Gutiérrez Camacho"
release = "1.0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # pull in docstrings
    "sphinx.ext.napoleon",  # numpy and Google style docstrings
    "sphinx.ext.viewcode",  # add links to source code
    "sphinx.ext.intersphinx",  # link to other projects' documentation
    "sphinx_autodoc_typehints",  # add type hints to docstrings
    "myst_parser",  # support for Markdown files
    "sphinx.ext.autosummary",  # generate summary tables for modules
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme_options = {}
html_static_path = ["_static"]
