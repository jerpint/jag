import mlflow
from tensorflow.keras.callbacks import Callback


class MLflowLogger(Callback):
    """
    Keras callback for logging metrics and final model with MLflow.
    Metrics are logged after every epoch. The logger keeps track of the best model
    based on the validation metric. At the end of the training, the best model is
    logged with MLflow.
    """

    def __init__(self): pass

    def on_epoch_end(self, epoch, logs=None):
        """
        Log Keras metrics with MLflow. Update the best model if the model
        improved on the validation data.
        """
        if not logs:
            return
        for name, value in logs.items():
            mlflow.log_metric(name, value, step=epoch)

    def on_train_end(self, *args, **kwargs):
        """
        Log the best model with MLflow and evaluate it on the train and
        validation data so that the metrics stored with MLflow
        reflect the logged model.
        """
        pass
