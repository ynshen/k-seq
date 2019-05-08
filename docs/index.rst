.. k-seq documentation master file, created by Sphinx

===================================================
k-seq: kinetic sequencing data analysis
===================================================

Introduction
=============
This is the developing package for ``k-seq`` with limited functions.

[More introduction of project coming up]

Installation
==============
To use this package, please clone the repo to ``/your/local/dir/k-seq`` by

>>> clone git@github.com:ynshen/k-seq.git /your/local/dirctory/

You will be asked for your github.com credentials. **Only Collaborator can download the package during the develop.**

Add the package directory on top of the python ``sys.path`` to import package for each script or python session.
Example:

.. code-block:: python

   import sys
   sys.path.insert(0, '/your/local/dirctory/k-seq/src/')
   import k_seq.data.io as io

To update to newer version of ``k-seq``:

>>> git pull

As the repo is updated very frequently during the develop, always ``pull`` before use it.
Please let me know if there is any bug.


Quick start
===============
[To be added]

Documentation for each module and functions please see below.


.. todo::

   - Add content on package source file documents
   - Add stand-alone CLI scripts for core functions (e.g. ``convert_count_to_seq_table.py``)
   - Package the source file to install k-seq from ``pip k-seq``


API
=============
.. automodule:: k_seq

.. toctree::
   :maxdepth: 3
   :caption: Package Content
   :glob:

   api/k_seq.data
   api/k_seq.fitting
   api/k_seq.utility

Indices and tables
=====

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
