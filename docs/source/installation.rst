************
Installation
************

Option 1: create a new ``conda`` environment
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


Option 2: install through ``pip``
-------------------------------------------

``k-seq`` can also install through ``pip`` (`<https://pypi.org/project/k-seq/>`_)

In your python environment, install ``k-seq`` and dependencies

.. code-block:: shell

  pip install k-seq
