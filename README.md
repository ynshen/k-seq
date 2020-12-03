k-seq: kinetic model parameter estimation from DNA sequencing data for genetic encoded biomolecules 
==============================

This repo corresponds with the paper (TO ADD: bioRxiv link), see the paper for how we used `k-seq` package to analyze 
the data from our kinetic measure with sequencing (_k_-Seq) experiments.

This is the frozen version used in the paper, 
see current version of `k-seq` package: https://github.com/ichen-lab-ucsb/k-seq/tree/master

## Prerequisites
### Download code and data
Code can be downloaded from GitHub repo: 
```shell script
https://github.com/ichen-lab-ucsb/k-seq.git
## checkout to this paper version
git checkout release/paper
```
Data and results can be downloaded from (TODO: Dryad link).


### Environment setup
#### Option 1: Run with `conda`
We recommend to use [Anaconda](https://anaconda.org/) to create a separate `k-seq` environment. 
Or a minimal installation: [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

To crate `k-seq` environment once `conda` in installed:

```shell script
# change directory to downloaded repo
cd /to/k-seq-git-repo
# create a k-seq environment
conda create -f environment.yml
# activate k-seq environment
conda activate k-seq
```

#### Option 2: Run with local python environment
This is NOT recommended unless you already use other python environment manager (e.g. pyenv, virtualenv)

In your python environment, install `k-seq` dependencies
```shell script
pip install -r requirements.txt
```

## Paper results

To repeat analysis and regenerate figures used in the paper, use `paper/`
To repeat the analysis and generate results in the paper

```shell script
# clone current version of repo

git clone 

```


*This is the temporary installation for `k-seq` package before publication, we will upload to pypi so it could be installed through `pip install k-seq`* 

Step 1: Clone the this git repo to your local

```bash
> cd /dir/to/save/repo
> git clone https://github.com/ichen-lab-ucsb/k-seq.git

```

Step 2: run installation code

```bash
> cd /dir/to/save/repo/k-seq/
> sh ./install.sh
```

There might be a prop to uninstall the old `k-seq` package, if installed.

#### To varify the installation of package
```bash
> python -c "import k-seq"
# No error shows
```

## Getting started examples

#### [Getting started](https://github.com/ynshen/k-seq/tree/master/examples/)

#### [Getting started notebook to run interactively: `/examples/getting_started.ipynb`](https://github.com/ynshen/k-seq/blob/master/examples/getting_started.ipynb)

## See [documentation website](https://ynshen.github.io/k-seq/) for usage

## TODO
- Clean up archived old code and notebooks

### Issue report:
https://github.com/ynshen/k-seq/issues