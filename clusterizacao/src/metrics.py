import pandas as pd
from sklearn.metrics import silhouette_score
import json
import mlflow


def get_metrics(data, labels):
    metrics = {
        "silhouette_score": silhouette_score(data, labels),
    }
    return metrics


def create_result(model, param, metrics):
    result = pd.DataFrame(
        {
            "name": model.__class__.__name__,
            **metrics,
            "param": json.dumps(param),
        },
        index=[0],
    )

    return result


def create_result_mlflow(model, param, metrics):
    mlflow.set_experiment("clusterizacao")
    with mlflow.start_run():
        mlflow.set_tag("modelo", model.__class__.__name__)
        mlflow.log_params(param)
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_name", model.__class__.__name__)
        mlflow.log_param("param", json.dumps(param))
        mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()
