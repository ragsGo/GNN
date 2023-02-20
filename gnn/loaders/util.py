
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.neighbors import kneighbors_graph


def create_list_of_edges(num_edges, inp, bidirectional=True):
    edges = [[]]*num_edges
    for x, y in inp:
        edges[x].append(y)
        if bidirectional:
            edges[y].append(x)
    return edges


def naive_partition(edges, size, bidirectional=True, traversed=None):
    if traversed is None:
        traversed = set()
    nodes = set([x[0] for x in edges]).union(set(x[1] for x in edges))

    edges = create_list_of_edges(len(nodes), edges, bidirectional=bidirectional)

    first_free_index = min(nodes - traversed)

    sampled_nodes = [first_free_index]
    traversed = traversed.union({first_free_index})
    tocheck = edges[first_free_index]
    while len(sampled_nodes) < size and len(traversed) < len(edges) and len(tocheck) > 0:
        check_next = []
        for check in tocheck:
            if check in traversed:
                continue
            traversed = traversed.union({check})
            sampled_nodes.append(check)
            if len(sampled_nodes) >= size:
                break
            check_next.extend(edges[check])
        while len(check_next) == 0 and len(traversed) < len(nodes):
            first_free_index = min(nodes - traversed)
            check_next = edges[first_free_index]
            traversed = traversed.union({first_free_index})
            sampled_nodes.append(first_free_index)
        tocheck = check_next
    return sampled_nodes


def split_dataset_graph(df, batches, neighbours=5, metric='euclidian', partition=naive_partition):
    if isinstance(batches, int):
        batches = [len(df)//batches]*batches

    knn_dist_graph = kneighbors_graph(
        X=df,
        n_neighbors=neighbours,
        metric=metric,
        n_jobs=6
    )

    edge_index, _ = from_scipy_sparse_matrix(knn_dist_graph)
    edges_raw = edge_index.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]

    retval = []
    traversed = set()
    for batch in batches:
        batch_nodes = partition(edges, batch, bidirectional=False, traversed=set(traversed))
        traversed |= set(batch_nodes)
        retval.append(df.iloc[batch_nodes])
    # train_list = partition(edges, train, bidirectional=False)
    # test_list = partition(edges, test, bidirectional=False, traversed=set(train_list))
    # valid_list = list(set(df.index) - set(train_list) - set(test_list))

    return tuple(retval)  # df.iloc[train_list], df.iloc[test_list], df.iloc[valid_list]


def split_dataset(df, batches, **_):
    if isinstance(batches, int):
        batches = [len(df)//batches]*batches
    start = 0
    retval = []
    for batch in batches:
        retval.append(df.iloc[start:start+batch, :])
        start += batch
    return tuple(retval)  # df.iloc[:train, :], df.iloc[train:train+test, :], df.iloc[train+test:, :]
