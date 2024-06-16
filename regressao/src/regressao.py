from regressao.src.data_loader import load_data
from regressao.src.model_definitions import models_param_grid
from regressao.src.classifier_evaluation import evaluate_models


def regressao():
    # Load data
    data = load_data()

    # Define models and their parameter grids
    models_param = models_param_grid()

    # Loop through data sets and evaluate models
    for dataset, data_dict in data.items():
        print(f"Dataset: {dataset}")
        results = evaluate_models(models_param, data, dataset)
        print(results.head())
