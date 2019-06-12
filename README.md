# jag

## Installation instructions

To install the `jag` package, use pip:

`pip install -e .`

### Dev

To install in dev mode (i.e. to run unit tests), use:

`pip install -e .[dev]`

Note: if you are using `zsh`, use:

`pip install -e '.[dev]'`

## Mlflow

Install mlflow using pip or conda:

`pip install mlflow` or `conda install -c conda-forge mlflow`

### Sample usage

An example script for mlflow is taken from their original repo and is in `examples/`. Run it from the root directory:

`python examples/mlflow_tf_example.py`

Then launch mlflow from the root directory:

`mlflow ui`

You can then go see the different experiments at `localhost:5000`

## Development installation instructions

To install the development environment for `jag`, use the installation script

`./install.sh`

This will install the development environment of the package and configure the git hooks.
