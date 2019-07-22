k-seq
==============================


## Installation

Step 1: Clone the this git repo to your local machine (currently, you need to be a collaborator for this project to download)

```bash
# If you already has SSH keygen set up for GitHub on your local machine
> git clone git@github.com:ynshen/k-seq.git

# If you prefer HTTPS to authorize by GitHub Username and password
> git clone https://github.com/ynshen/k-seq.git
```

Step 2: `pip` install from newst wheel under folder `dist/`

```bash
> pip install dist/<find_the_newest_version>.whl
```

Step 2 alternative: `bash` script to install/update `k-seq`
```bash
#Make sure the bash code is executable
> sudo chmod u+x refresh_package.sh

#Refresh the package
> ./refresh_package.sh
```
You may encounter the prop to uninstall the old `k-seq` packge.


#### Method 2: directly install from GitHub.com
This method **only works when the SSH keygen has been set up on your local machine and no additional key is added to the
keygen**

```bash
> pip install git+ssh://github.com:ynshen/k-seq.git
```

#### To varify the installation of package
```bash
> python -c "import k-seq"
#No error shows
```

## Getting started examples

#### [Getting started](https://github.com/ynshen/k-seq/tree/master/examples/)

#### [Getting started notebook to run interactively: `/examples/getting_started.ipynb`](https://github.com/ynshen/k-seq/blob/master/examples/getting_started.ipynb)

## See [documentation website](https://ynshen.github.io/k-seq/) for usage


# Todos
- clean up example of BYO analysis


kinetic sequencing to estimate the kinetic coefficient of ribozymes

Here is the potential project organization
------------
    ├── LICENSE
    ├── Makefile           <- Makefile for reproducibility, with commands like `make data` or `make fit_byo`
    ├── README.md          <- The top-level README file with overview
    │
    ├── docs\              <- package documents
    │
    ├── notebooks\         <- Jupyter notebooks used for analysis
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable, so src can be imported
    │
    └── src                <- Source code of k-seq package for use in this project.
        ├── data           <- Modules for data generation, preprocessing, io, etc
        │
        └── fitting        <- Modules for fitting and other estimation



--------

<p><small>Project structure modified from <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
