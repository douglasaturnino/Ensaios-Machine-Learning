from classificacao.src.metrics import get_metrics, create_result, create_result_mlflow
import pandas as pd


def evaluate_models(models_param_grid, data_dict, dataset):
    results = pd.DataFrame()

    for model_class, param_list in models_param_grid.items():
        model_results = pd.DataFrame()

        for param in param_list:
            model = model_class
            result = classifier_evaluation(model, param, data_dict, dataset)
            model_results = pd.concat([model_results, result]).reset_index(drop=True)

        model_results = model_results.sort_values("f1_score", ascending=False)
        results = pd.concat([results, model_results]).reset_index(drop=True)

    return results


def classifier_evaluation(model_class, param, data, dataset):
    data_mapping = {
        "training": (data["training"]["X"], data["training"]["y"]),
        "validation": (data["training"]["X"], data["training"]["y"]),
        "test": (
            pd.concat([data["training"]["X"], data["validation"]["X"]]).reset_index(
                drop=True
            ),
            pd.concat([data["training"]["y"], data["validation"]["y"]]).reset_index(
                drop=True
            ),
        ),
    }

    x_data, y_data = data_mapping.get(dataset, (None, None))

    if x_data is None or y_data is None:
        raise ValueError(
            "Invalid data type provided. Choose from 'training', 'validation', or 'test'."
        )

    model = model_class(**param)
    model.fit(x_data, y_data.values.ravel())

    predict_mapping = {
        "training": x_data,
        "validation": data["validation"]["X"],
        "test": data["test"]["X"],
    }

    yhat_model = model.predict(predict_mapping[dataset])

    metrics_mapping = {
        "training": data["training"]["y"],
        "validation": data["validation"]["y"],
        "test": data["test"]["y"],
    }

    metrics = get_metrics(metrics_mapping[dataset], yhat_model)

    result = create_result(model, param, metrics)

    create_result_mlflow(model, param, metrics, dataset)

    return result
