import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
import json
import mlflow


def get_metrics(y_data, yhat_model):
    metrics = {
        "R2": r2_score(y_data, yhat_model),
        "MSE": mean_squared_error(y_data, yhat_model),
        "RMSE": root_mean_squared_error(y_data, yhat_model),
        "MAE": mean_absolute_error(y_data, yhat_model),
        "MAPE": mean_absolute_percentage_error(y_data, yhat_model),
    }
    return metrics


def create_result(model, param, metrics):
    metric = pd.DataFrame(
        {"name": model.__class__.__name__, **metrics, "param": json.dumps(param)},
        index=[0],
    )

    return metric


def create_result_mlflow(model, param, metrics, dataset, Polynomial=False):
    mlflow.set_experiment("regressao")
    with mlflow.start_run():
        model_name = model.__class__.__name__
        if Polynomial:
            model_name = "Polinomial" + model_name

        mlflow.set_tag("modelo", model_name)
        mlflow.set_tag("dataset", dataset)
        mlflow.log_params(param)
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("param", json.dumps(param))
        mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()
