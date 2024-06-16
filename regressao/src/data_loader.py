import os
import pandas as pd


def load_data():
    path_home = os.getcwd()

    data_paths = {
        "training": {
            "X": os.path.join(path_home, "regressao", "data", "X_training.csv"),
            "y": os.path.join(path_home, "regressao", "data", "y_training.csv"),
        },
        "validation": {
            "X": os.path.join(path_home, "regressao", "data", "X_validation.csv"),
            "y": os.path.join(path_home, "regressao", "data", "y_validation.csv"),
        },
        "test": {
            "X": os.path.join(path_home, "regressao", "data", "X_test.csv"),
            "y": os.path.join(path_home, "regressao", "data", "y_test.csv"),
        },
    }

    data = {}
    for key, paths in data_paths.items():
        data[key] = {"X": pd.read_csv(paths["X"]), "y": pd.read_csv(paths["y"])}

    return data
