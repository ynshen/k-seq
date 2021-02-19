k-seq: analytic toolbox for data from kinetic assay with DNA sequencing
==============================

**This is a snapshot of ``k-seq`` code for the submitted paper.** See [``master`` branch](https://github.com/ichen-lab-ucsb/k-seq/tree/master)
for the most recent ``k-seq`` package and its general usage.

Check out our paper [*Kinetic sequencing (k-Seq) as a massively parallel assay for ribozyme kinetics: utility and critical parameters*](https://www.biorxiv.org/content/10.1101/2020.12.02.407346v1)
for the *k*-Seq experiment method and how we used the package for data analysis.

# Repeat analysis using this code snapshot

## Download code and data

Download code snapshot from the `paper-archive` branch in Github:
```shell script
git clone https://github.com/ichen-lab-ucsb/k-seq.git
git checkout paper-archive
```

Download dataset from Dryad (currently private for review).

## Environment setup
Here we set up `k-seq` environment and import `k-seq` package directly from the downloaded code snapshot.
If `k-seq` package has been installed through `pip`, all dependencies should be already installed.

### Option 1: install with `conda`
We recommend to use [Anaconda](https://anaconda.org/) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html)
to create a separate `k-seq` environment and install dependencies:

```shell script
# change directory to downloaded code snapshot
cd /path/to/k-seq-git-repo
# create a k-seq environment
conda create -f environment.yml -n k-seq
# activate k-seq environment
conda activate k-seq
```

### Option 2: install dependencies in local python environment
Directly installing `k-seq` dependencies in your local python environment is NOT recommended. It might upgrade or downgrade your current packages and potentially break other packages/programs.

To install all dependencies required by `k-seq` in your python environment:
```shell script
# change directory to downloaded code snapshot
cd /path/to/k-seq-git-repo
# pip install dependencies
pip install -r requirements.txt
```

## Import `k-seq` package locally

We recommend to load this snapshot version of `k-seq` locally to use the correct version for repeating the paper results. To load `k-seq` version, at the beginnig of your python script or Jupyter notebooks:

```python
>>> import sys
>>> sys.path.insert(0, 'path/to/k-seq_package/src')  # path to k-seq source code
>>> import k_seq
>>> print(k_seq.__file__)                            # check the correct version is imported
'/path/to/k-seq_package/src/k_seq/__init__'
```

## Repeat paper results

The Jupyter notebooks under the `notebook` folder are to repeat the analysis and regenerate figures used in the paper. To start the jupyter notebook

```shell script
# use the correct environment with Jupyter notebook installed
# e.g. conda k-seq environment
conda activate k-seq

# check to notebook folder
cd /path/to/k-seq-git-repo
cd notebook

# start jupyter server for notebooks
jupyter notebook
```

NOTE: please double check the correct python kernel is used by the Jupyter Notebook.
To add `k-seq` conda environment to the notebook, see [install IPython kernels](https://ipython.readthedocs.io/en/latest/install/kernel_install.html).

## Run calculation scripts

The scripts under the `scripts` folder can be used to rerun the entire pipeline for *k*-Seq data processing. All scripts has been rewired to call the `k-seq` snapshot version locally.

Example to run analysis scripts:


```shell script
# go to scripts directory
cd scripts
# run script to regenerate the BYO variant pool dataset
./generate-byo-variant-dataset.py \
    --count_file_dir path/to/dryad-data/data/byo-variant/counts \
    --norm_file path/to/dryad-data/data/data/byo-variant/norm-factor.txt \
    --output output/folder
```

----
##### Contact
For any question and issues: [issue report](https://github.com/ichen-lab-ucsb/k-seq/issues) or Yuning Shen (yuningshen@ucsb.edu)
