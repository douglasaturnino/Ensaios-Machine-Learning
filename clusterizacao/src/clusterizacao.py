from clusterizacao.src.data_loader import load_data
from clusterizacao.src.model_definitions import models_param_grid
from clusterizacao.src.classifier_evaluation import evaluate_models


def clusterizacao():
    # Load data
    data = load_data()

    # Define models and their parameter grids
    models_param = models_param_grid()

    # Loop through data sets and evaluate models
    for dataset, data_dict in data.items():
        print(f"Dataset: {dataset}")
        results = evaluate_models(models_param, data)
        print(results.head())
