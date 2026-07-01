MD debris datasets
==================

Each debris dataset directory must contain a ``meta.json`` file with recoil,
target composition, lattice type, interatomic potentials, electronic
interactions, DOI, and contributor metadata. For example:

.. code-block:: json

   {
     "recoil": "W",
     "target": {
       "W": 1.0
     },
     "lattice": "bcc",
     "interatomic_potentials": [
       "https://doi.org/10.1016/j.nimb.2009.06.123"
     ],
     "electronic_interactions": "SRIM",
     "doi": "https://doi.org/10.1038/s41467-018-03415-5",
     "contributors": [
       "A. E. Sand"
     ]
   }

Configure the database with matching target, lattice, and electronic
interaction filters:

.. code-block:: python

   irradiapy.config.set_debris_database(
       path="/path/to/CascadesDefectsDB",
       electronic_interactions="SRIM",
       target={"W": 1.0},
       lattice="bcc",
   )

The optional ``interatomic_potentials``, ``doi``, and ``contributors``
arguments apply additional filters. All metadata comparisons, including
``lattice``, are exact; target stoichiometries are compared numerically.
