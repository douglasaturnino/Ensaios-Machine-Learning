from classificacao.src.data_loader import load_data
from classificacao.src.model_definitions import models_param_grid
from classificacao.src.classifier_evaluation import evaluate_models


def classificacao():
    # Load data
    data = load_data()

    # Define models and their parameter grids
    models_param = models_param_grid()

    # Loop through data sets and evaluate models
    for dataset, data_dict in data.items():
        print(f"Dataset: {dataset}")
        results = evaluate_models(models_param, data, dataset)
        print(results.head())


"""
graph TD;

subgraph main.py
    A[main.py]
    B[data_loader.load_data()]
    C[model_definitions.models_param_grid()]
    D[classifier_evaluation.evaluate_models()]
end

subgraph data_loader.py
    B[data_loader.py]
    B -->|Extrair Dados| E[X_training.csv, y_training.csv, X_validation.csv, y_validation.csv, X_test.csv, y_test.csv]
end

subgraph model_definitions.py
    C[model_definitions.py]
    C -->|Definir Modelos e Parâmetros| F[KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, LogisticRegression]
end

subgraph metrics.py
    K[metrics.py]
    I -->|Calcular Métricas| K
    J -->|Formatar Resultado| K
end

subgraph classifier_evaluation.py
    D[classifier_evaluation.py]
    D -->|Avaliar Modelos| G[classifier_evaluation.evaluate_models()]
    G -->|Classificar e Avaliar| H[classifier_evaluation.classifier_evaluation()]
    H -->|Gerar Métricas| I[classifier_evaluation.get_metrics()]
    H -->|Gerar Resultado| J[classifier_evaluation.create_result()]
end

"""
