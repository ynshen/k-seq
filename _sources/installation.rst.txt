************
Installation
************

Python-only installation
=========================
``k-seq`` package can directly install from ``pip`` (`<https://pypi.org/project/k-seq/>`_):

.. code-block:: shell

  pip install k-seq

The python-only package has most functionalities ready-to-use for count data
(e.g., preprocessed count files or a CSV table).
For FASTQ read joining, trimming, and deduplication, we use `EasyDIVER <https://github.com/ichen-lab-ucsb/EasyDIVER>`_
and require additional dependencies on `PANDAseq <https://github.com/neufeld/pandaseq/wiki/Installation>`_.

Complete installation
=======================

Option 1: install with ``conda``
----------------------------------------------

We recommend to use `Anaconda <https://anaconda.org/>`_ to create a separate `k-seq` environment.
Or a minimal installation: `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

To crate ``k-seq`` environment once `conda` in installed:

.. code-block:: shell

  # change directory to downloaded repo
  cd /to/k-seq-git-repo

  # create a k-seq environment
  conda create -f environment.yml

  # activate k-seq environment
  conda activate k-seq

  # install k-seq python package
  pip install k-seq


Option 2: install manually
----------------------------
Before install ``k-seq`` python packge, please install to `EasyDIVER <https://github.com/ichen-lab-ucsb/EasyDIVER>`_ for
sequencing reads processing. Once EasyDIVER is installed, install ``k-seq`` python package through ``pip``:

.. code-block:: python

  pip install k-seq
