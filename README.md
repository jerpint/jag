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

## Orion

To use Orion, you must first configure a MongoDB database.
First, you need [to install MongoDB](cf. https://docs.mongodb.com/manual/installation/).
Then, you can create a database called `orion_test` and create an username and password with 
```
mongo orion_test --eval 'db.createUser({user:"user",pwd:"pass",roles:["readWrite"]});'
```
After, you can adapt the orion config file provided in the config directory of the repository and copy/paste it.
```
mkdir -p ~/.config/orion.core/
cp config/orion/orion_config.yaml ~/.config/orion.core/orion_config.yaml
```
Finally, you can test your installation with the script `hyper_search.sh` located in `/examples/orion`.
