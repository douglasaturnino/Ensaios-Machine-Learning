from clusterizacao.src.metrics import get_metrics, create_result, create_result_mlflow
import pandas as pd


def evaluate_models(models_param_grid, data_dict):
    results = pd.DataFrame()

    for model_class, param_list in models_param_grid.items():
        model_results = pd.DataFrame()

        for param in param_list:
            model = model_class
            result = classifier_evaluation(model, param, data_dict)
            model_results = pd.concat([model_results, result]).reset_index(drop=True)

        model_results = model_results.sort_values("silhouette_score", ascending=False)
        results = pd.concat([results, model_results]).reset_index(drop=True)

    return results


def classifier_evaluation(model_class, param, data):
    model = model_class(**param)

    labels = model.fit_predict(data["training"]["X"])

    metrics = get_metrics(data["training"]["X"], labels)

    result = create_result(model, param, metrics)

    create_result_mlflow(model, param, metrics)

    return result
