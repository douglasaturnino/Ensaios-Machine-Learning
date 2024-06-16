import os
import pandas as pd


def load_data():
    path_home = os.getcwd()

    data_paths = {
        "training": {
            "X": os.path.join(path_home, "clusterizacao", "data", "X_dataset.csv"),
        }
    }

    data = {}
    for key, paths in data_paths.items():
        data[key] = {
            "X": pd.read_csv(paths["X"]),
        }

    return data
