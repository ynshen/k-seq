.. k-seq documentation master file, created by
   sphinx-quickstart on Sun Feb 16 22:19:53 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``k-seq``: a package for kinetic measurements with high throughput sequencing
===========================================================================

Welcome to the documentation for ``k-seq``, a python package to analyze the data from kinetic measurement using
high throughput sequencing. It contains the function to process the FASTQ reads to a count table, preprocess
count table, quantify the extend of reaction, and fitting the data into kinetic model with bootstrapping to quantify
the uncertainty.

Please check :doc:`installation` to set up the environment to run ``k-seq`` package. For quick start, check
:doc:`getting_started`.

Table of content
-----------------
.. toctree::
   :maxdepth: 2

   installation
   getting_started
   k-seq_package


API reference
-------------------
* :doc:`k-seq_package`
* :ref:`modindex`
