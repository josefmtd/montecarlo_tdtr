# MonteCarlo TDTR

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Monte Carlo analysis script for time-domain thermoreflectance (TDTR) setup in 4th floor lab Micronova

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── instructions.txt   <- Basic instructions for using the repository
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│   |                     the creator's initials, and a short `-` delimited description, e.g.
│   |                     `1.0-jqp-initial-data-exploration`.
|   |
|   ├── out            <- Contains the output files of submitted batch jobs
|   |
│   ├── run_notebooks.sh          <- Bash script for submitting Triton batch jobs to excecute the notebooks
│   └── run_notebooks_multiple.sh <- Bash script for running multiple notebooks based on a file pattern
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         montecarlo_tdtr and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── environment.yml    <- The environment file for reproducing the analysis environment
│
├── setup.cfg          <- Configuration file for flake8
│
└── montecarlo_tdtr    <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes montecarlo_tdtr a Python module
    │
    ├── data                    <- Scripts to access the measurement data
    │   ├── beam.py
    │   └── dataframe.py
    │
    ├── analysis                <- Scripts to calculate thermal properties from measurement
    │   ├── bidirectional.py
    │   └── bidirectional_gpu.py
    │
    ├── out                     <- Contains the output files of submitted batch jobs
    |
    ├── aluminium_nitride.sh    <- Bash script to submit a batch job for a single file Monte Carlo analysis
    ├── run_AlN_multiple.sh     <- Bash script to submit multiple batch jobs for the Monte Carlo analysis, based on a file pattern
    └── montecarlo_aln.py       <- Python script for running the Monte Carlo Analysis
```

--------

