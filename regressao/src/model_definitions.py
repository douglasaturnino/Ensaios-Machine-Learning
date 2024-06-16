import random

from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def models_param_grid():
    MAX_ITER = 20
    models_param_grid = {
        LinearRegression: [{"fit_intercept": True}],
        Lasso: [
            {
                "alpha": random.choice([1, 2, 3]),
                "max_iter": random.choice([1000, 1500, 2000]),
            }
            for _ in range(2, MAX_ITER)
        ],
        Ridge: [
            {
                "alpha": random.choice([1, 2, 3]),
                "max_iter": random.choice([1000, 1500, 2000]),
            }
            for _ in range(2, MAX_ITER)
        ],
        ElasticNet: [
            {
                "alpha": random.choice([1, 2, 3]),
                "max_iter": random.choice([1000, 1500, 2000]),
                "l1_ratio": random.choice([0.3, 0.5, 0.7]),
            }
            for _ in range(2, MAX_ITER)
        ],
        DecisionTreeRegressor: [{"max_depth": i} for i in range(2, MAX_ITER)],
        RandomForestRegressor: [
            {
                "max_depth": i,
                "n_estimators": random.choice([100, 200, 300]),
            }
            for i in range(2, MAX_ITER)
        ],
    }

    return models_param_grid
