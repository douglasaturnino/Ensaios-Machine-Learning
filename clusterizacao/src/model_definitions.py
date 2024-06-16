from sklearn.cluster import KMeans, AffinityPropagation


def models_param_grid():
    MAX_ITER = 20
    models_param_grid = {
        AffinityPropagation: [{"preference": i} for i in range(-1, -MAX_ITER, -1)],
        KMeans: [{"n_clusters": i} for i in range(2, MAX_ITER)],
    }

    return models_param_grid
