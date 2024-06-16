import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def models_param_grid():
    MAX_ITER = 3
    models_param_grid = {
        KNeighborsClassifier: [{"n_neighbors": i} for i in range(2, MAX_ITER)],
        DecisionTreeClassifier: [{"max_depth": i} for i in range(2, MAX_ITER)],
        RandomForestClassifier: [
            {
                "n_estimators": random.choice([100, 200, 300]),
                "max_depth": random.choice([None, 3, 4, 5]),
            }
            for _ in range(2, MAX_ITER)
        ],
        LogisticRegression: [
            {
                "C": random.choice([0.1, 0.5, 1.0]),
                "solver": random.choice(
                    ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
                ),
            }
            for _ in range(2, MAX_ITER)
        ],
    }

    return models_param_grid
