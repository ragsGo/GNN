
from torch_geometric.utils import from_scipy_sparse_matrix, sort_edge_index
from sklearn.neighbors import kneighbors_graph


def split_dataset_graph(df, train, test, validation, neighbours=5, metric='euclidian'):
    if train < 1:
        assert train+test+validation == 1, "Train, test and validation needs to add up to 1"
        train = int(train*len(df))
        test = int(test*len(df))
        validation = int(validation*len(df))

    assert train+test+validation == len(df), "Train, test and validation needs to add up to length of dataset"
    knn_dist_graph = kneighbors_graph(
        X=df,
        n_neighbors=neighbours,
        metric=metric,
        n_jobs=6
    ).toarray()

    edges = [list(x.nonzero()[0]) for x in knn_dist_graph]
    nodes = set(df.index)
    used = {0}
    train_list = [0]
    tocheck = edges[0]
    check_next = []
    while len(train_list) < train and len(used) < len(edges) and len(tocheck) > 0:
        check_next = []
        for check in tocheck:
            if check in used:
                continue
            used = used.union({check})
            train_list.append(check)
            if len(train_list) >= train:
                break
            check_next.extend(edges[check])
        while len(check_next) == 0 and len(used) < len(nodes):
            first_free_index = min(nodes - used)
            check_next = edges[first_free_index]
            used = used.union({first_free_index})
            train_list.append(first_free_index)

        tocheck = check_next
    first_free_index = min(nodes - used)
    check_next = []
    used = used.union({first_free_index})
    tocheck = edges[first_free_index]
    test_list = [first_free_index]
    while len(test_list) < test and len(used) < len(edges) and len(tocheck) > 0:
        check_next = []
        for check in tocheck:
            if check in used:
                continue
            used = used.union({check})
            test_list.append(check)
            if len(test_list) >= test:
                break
            check_next.extend(edges[check])
        while len(check_next) == 0 and len(used) < len(nodes):
            first_free_index = min(nodes - used)
            test_list.append(first_free_index)
            used = used.union({first_free_index})
            check_next = edges[first_free_index]
        tocheck = check_next
    valid_list = nodes-used

    return df.iloc[train_list], df.iloc[test_list], df.iloc[list(valid_list)]


def split_dataset(df, train, test, validation, **_):
    if train < 1:
        assert train+test+validation == 1, "Train, test and validation needs to add up to 1"
        train = int(train*len(df))
        test = int(test*len(df))
        validation = int(validation*len(df))

    assert train+test+validation == len(df), "Train, test and validation needs to add up to length of dataset"
    return df.iloc[:train, :], df.iloc[train:train+test, :], df.iloc[train+test:, :]
