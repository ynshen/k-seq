k-seq
==============================

# Current basic functions
- import count files for k-seq samples
- look up spike-in sequence counts and quantification sequence amount
- select valid sequences and


## `skin_mb` package contents
The skin microbiome package `skin_mb` is now `pip` installable for authorized users that has access to this repo. A short
Jupyter notebook example of using `skin_mb` to manipulate OTU tables is in `OTU_table_manipulation_example.ipynb`

### Package installation through `pip`
#### Method 1: local installation from git repo clone
Step 1: Clone the git repo to your local machine
```bash
# If you already has SSH keygen set up for GitHub on your local machine
> git clone git@github.com:ichen-lab-ucsb/skin_wound_microbiome.git

# If you prefer HTTPS to authorize by GitHub Username and password
> git clone https://github.com/ichen-lab-ucsb/skin_wound_microbiome.git
```

Step 2: `pip` install from the wheel
```bash
> pip install dist/<the_newest_version>.whl
```

#### Method 2: directly install from GitHub.com
This method only works when the SSH keygen has been set up on your local machine and no additional key is added to the
keygen
```bash
> pip install git+ssh://git@github.com/ichen-lab-ucsb/skin_wound_microbiome.git
```

#### To varify the installation of package
```bash
> python -c "import skin_mb"
import successfully
```


# See [documentation website](https://ynshen.github.io/k-seq/) for usage


# Todos
TODO: update top-level README
TODO: Use Sphix or pdoc for documentation websites
TODO: clean up example of BYO analysis


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
