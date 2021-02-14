

Getting Started Tutorial
========================

This is a quick tutorial on using ``k-seq`` package to analyze *k*-Seq experiments data. The standard pipeline includes

1. `Data preprocessing`_
2. `Sequence quantification`_
3. `Model fitting with bootstrapping`_

Some `additional analysis`_ tools are also introduced to help design your experiments and characterizing your data.

.. seealso::
  :doc:`installation` for setting up `k-seq` environment and installing `k-seq` package.
  :doc:`k-seq package <k-seq_package>` for the API documentation of the package.

Data preprocessing
******************


Join paired-end reads using ``EasyDIVER``
-----------------------------------------
We integrated the `EasyDIVER <https://github.com/ichen-lab-ucsb/EasyDIVER>`_ pipeline in the script ``fastq-to-counts.sh``
to perform paired-end joining for FASTQ files (``.fastq.gz``) from Illumina sequencing and deduplicated joined reads into counts
(number of reads) for unique sequences.

All FASTQ files should be organized in a folder with the format of file name: ``[sample-name]_L[lane number]_R[1/2]_001.fastq.gz``.
``sample-name`` is the string identifier for a sample so that reads from different sequencing flow-cell lanes can be pooled
together and ``lane number`` indicate the lane of the flow-cell. ``R[1/2]`` indicate the direction of paired-end reads
(``R1`` for forward reads and ``R2`` for reverse reads) and each sample-lane expects to have matched files for both
directions.

Here is an example of input file folder including the paired-end reads from two samples ``R4A-0A_S7`` and ``R4B-0A_S21``
from 4 lanes::

  R4A-0A_S7_L001_R1_001.fastq.gz  R4B-0A_S21_L001_R1_001.fastq.gz
  R4A-0A_S7_L001_R2_001.fastq.gz  R4B-0A_S21_L001_R2_001.fastq.gz
  R4A-0A_S7_L002_R1_001.fastq.gz  R4B-0A_S21_L002_R1_001.fastq.gz
  R4A-0A_S7_L002_R2_001.fastq.gz  R4B-0A_S21_L002_R2_001.fastq.gz
  R4A-0A_S7_L003_R1_001.fastq.gz  R4B-0A_S21_L003_R1_001.fastq.gz
  R4A-0A_S7_L003_R2_001.fastq.gz  R4B-0A_S21_L003_R2_001.fastq.gz
  R4A-0A_S7_L004_R1_001.fastq.gz  R4B-0A_S21_L004_R1_001.fastq.gz
  R4A-0A_S7_L004_R2_001.fastq.gz  R4B-0A_S21_L004_R2_001.fastq.gz

To run ``EasyDIVER``:

.. code-block:: shell
  :linenos:

  > fastq-to-counts.sh \
        -i /path/to/input/folder \
        -o /path/to/output/folder \
        -p CTACGAATTC \              # forward adapter sequence for trimming
        -q CTGCAGTGAA \              # reverse adapter sequence for trimming
        -c \                         # use completely matching in joining and discard reads with error
        -a \                         # join the paired-end reads first before trimming the adapters for reads with heavily overlapped regions
        -T 12                        # number of threads to run in parallel for pandaSeq
  > ls /path/to/output/folder
  counts  fastas  fastqs  histos  log.txt

The output folder contains following files:

- ``counts``: folder contain deduplicated count files named as ``[sample-name]_counts.txt``
- ``fastqs``: joined FASTQ files named as ``[sample-name].joined.fastq``
- ``fastas``: joined FASTA files named as ``[sample-name].joined.fasta``
- ``histos``: files contain the joined reads length histogram statistics, named as ``[sample-files]_counts_histo.txt``

The output are the deduplicated count files with format::

  number of unique sequences =     981824
  total number of molecules =    10096876

  GGGGGGGGATTCATGACTATT                                                                 1
  GGGGGGGAGTAGGACTGCAAA                                                                 1
  GGGGGGGAAGACTCCGGAACG                                                                 1
  GGGGGGGAACGCATTTCACGG                                                                 1
  GGGGGGGACGTTCACCGGCAA                                                                 1
  ...


Load count files
-----------------
We next load and parse count files to Python using :py:class:`k_seq.data.SeqData`:

.. code-block:: python
  :linenos:

  from k_seq.data import SeqData

  dataset = SeqData.from_count_files(
      count_files='path/to/count/file',
      pattern_filter='_counts.',
      name_template='[{byo}{exp_rep}]-{}{}_S{smpl}_counts.txt',
      sort_by='name',
      x_values='',
      x_unit='M',
      input_sample_name=['R0'],
      note='k-seq results of doped-pool BYO aminoacylation. Total DNA amount in each reacted sample were '
           'quantified with spike-in sequence with 2 edit distance as radius or qPCR + Qubit'
  )


Sequence quantification
***********************







Model fitting with bootstrapping
********************************




Additional analysis
*********************
