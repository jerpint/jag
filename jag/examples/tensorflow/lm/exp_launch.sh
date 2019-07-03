# Set the experiment via environment variables
export MLFLOW_EXPERIMENT_NAME=$1

mlflow experiments create --experiment-name $1

python char_lstm.py
