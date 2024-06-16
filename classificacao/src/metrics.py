import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
import mlflow


def get_metrics(y_data, yhat_model):
    metrics = {
        "accuracy": accuracy_score(y_data, yhat_model),
        "precision_score": precision_score(y_data, yhat_model),
        "recall_score": recall_score(y_data, yhat_model),
        "f1_score": f1_score(y_data, yhat_model),
    }
    return metrics


def create_result(model, param, metrics):
    metric = pd.DataFrame(
        {"name": model.__class__.__name__, **metrics, "param": json.dumps(param)},
        index=[0],
    )

    return metric


def create_result_mlflow(model, param, metrics, dataset):
    mlflow.set_experiment("classificacao")
    with mlflow.start_run():
        mlflow.set_tag("modelo", model.__class__.__name__)
        mlflow.set_tag("dataset", dataset)
        mlflow.log_params(param)
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_name", model.__class__.__name__)
        mlflow.log_param("param", json.dumps(param))
        mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()
