k-seq: analytics tools for kinetic sequencing data analysis
==============================

**This is a frozen version for Nucleic Acids Research submission**
see https://github.com/ichen-lab-ucsb/k-seq for most updated version and `k-seq`
python package for the best use of this tool

See our [bioRxiv paper](https://www.biorxiv.org/content/10.1101/2020.12.02.407346v1)
for how we used `k-seq` package to analyze the data from our kinetic sequencing (_k_-Seq) experiments.


# Prerequisites
## Download code and data

**Code snapshot, data, and results are already included in this Dryad dataset**

To download the updated version from GitHub repo:
```shell script
git clone https://github.com/ichen-lab-ucsb/k-seq.git
## checkout to the branch including paper results
git checkout release/0.4.2-paper
```


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

NOTE: please double check the Jupyter Notebook is running the correct python kernel.
To add k-seq conda environment as a IPython kernel, see https://ipython.readthedocs.io/en/latest/install/kernel_install.html

# Run scripts

Scripts in this repo has been rewired to call the k-seq package locally (version in `src/`).
Change into the script directory to directly run a script

Example:

```shell script
# go to scripts directory
cd scripts

# don't forget './' before the script
./generate-byo-variant-dataset.py \
    --count_file_dir ../../data/byo-variant/counts \
    --norm_file ../../data/byo-variant/norm-factor.txt \
    --output /your/output/dir
```
