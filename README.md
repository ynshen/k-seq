k-seq: kinetic model parameter estimation from DNA sequencing data for genetic encoded biomolecules 
==============================

This repo corresponds with the paper (TO ADD: bioRxiv link), see the paper for how we used `k-seq` package to analyze 
the data from our kinetic measure with sequencing (_k_-Seq) experiments.

This is the frozen version used in the paper, 
see current version of `k-seq` package: https://github.com/ichen-lab-ucsb/k-seq/tree/master

# Prerequisites
## Download code and data
Code can be downloaded from GitHub repo: 
```shell script
https://github.com/ichen-lab-ucsb/k-seq.git
## checkout to this paper version
git checkout release/paper
```
Data and results can be downloaded from (TODO: Dryad link).


## Environment setup
### Option 1: Run with `conda`
We recommend to use [Anaconda](https://anaconda.org/) to create a separate `k-seq` environment. 
Or a minimal installation: [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

To crate `k-seq` environment once `conda` in installed:

```shell script
# change directory to downloaded repo
cd /path/to/k-seq-git-repo
# create a k-seq environment
conda create -f environment.yml
# activate k-seq environment
conda activate k-seq
```

### Option 2: Run with local python environment
This is NOT recommended unless you already use other python environment manager (e.g. pyenv, virtualenv)

In your python environment, install `k-seq` dependencies
```shell script
pip install -r requirements.txt
```

# Paper results

To repeat analysis and regenerate figures used in the paper, see `notebooks/k-seq-paper-figure.ipynb`
 and `notebooks/k-seq-paper-figure-SI.ipynb`


```shell script
# in the correct python environment, for conda
conda activate k-seq

# under correct directory
cd /path/to/k-seq-git-repo
cd notebook

# start jupyter server for notebooks
jupyter notebook
```