from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.neighbors import kneighbors_graph


def split_dataset_graph(df, train, test, validation, neighbours=5, mode='distance', metric='euclidian'):
    if train < 1:
        assert train+test+validation == 1, "Train, test and validation needs to add up to 1"
    else:
        assert train+test+validation == len(df), "Train, test and validation needs to add up to length of dataset"
    knn_dist_graph = kneighbors_graph(
        X=df,
        n_neighbors=neighbours,
        mode=mode,
        metric=metric,
        n_jobs=6
    )
    edge_index, edge_weight = from_scipy_sparse_matrix(knn_dist_graph)
    nodes = set(df.index)
    used = set()


def split_dataset(df, train, test, validation, **_):
    if train < 1:
        assert train+test+validation == 1, "Train, test and validation needs to add up to 1"
        train *= len(df)
        test *= len(df)
        validation *= len(df)
    assert train+test+validation == len(df), "Train, test and validation needs to add up to length of dataset"

    return df[:train, :], df[train:train+test, :], df[train+test:, :]
