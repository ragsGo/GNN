import random

from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.neighbors import kneighbors_graph


def create_list_of_edges(num_edges, inp, bidirectional=True):
    edges = [[]] * num_edges
    for x, y in inp:
        edges[x].append(y)
        if bidirectional:
            edges[y].append(x)
    return edges


def naive_partition(edges, size, bidirectional=True, traversed=None, **_):
    if traversed is None:
        traversed = set()
    nodes = set([x[0] for x in edges]).union(set(x[1] for x in edges))

    edges = create_list_of_edges(len(nodes), edges, bidirectional=bidirectional)

    # index_to_check = min(nodes - traversed)
    to_check = []
    sampled_nodes = []
    while len(to_check) == 0:
        index_to_check = random.choice(list(nodes - traversed))

        sampled_nodes = [index_to_check]
        traversed = traversed.union({index_to_check})
        to_check = edges[index_to_check]

    while (
        len(sampled_nodes) < size and len(traversed) < len(edges) and len(to_check) > 0
    ):
        check_next = []
        for check in to_check:
            if check in traversed:
                continue
            traversed = traversed.union({check})
            sampled_nodes.append(check)
            if len(sampled_nodes) >= size:
                break
            check_next.extend(edges[check])
        while len(check_next) == 0 and len(traversed) < len(nodes):
            # index_to_check = min(nodes - traversed)
            index_to_check = random.choice(list(nodes - traversed))
            check_next = edges[index_to_check]
            traversed = traversed.union({index_to_check})
            sampled_nodes.append(index_to_check)
        to_check = check_next
    return sampled_nodes


def split_dataset_graph(
    df,
    batches,
    neighbours=5,
    metric="euclidian",
    allow_duplicates=False,
    partition=naive_partition,
):
    if isinstance(batches, int):
        batch_size = len(df) // batches
        batches = [max(batch_size, neighbours+1)] * batches

    knn_dist_graph = kneighbors_graph(
        X=df, n_neighbors=neighbours, metric=metric, n_jobs=6
    )

    edge_index, _ = from_scipy_sparse_matrix(knn_dist_graph)
    edges_raw = edge_index.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]

    retval = []
    traversed = set()
    for batch in batches:
        batch_nodes = partition(
            edges,
            batch,
            bidirectional=False,
            traversed=traversed,
            allow_duplicates=allow_duplicates,
        )
        if not allow_duplicates:
            traversed |= set(batch_nodes)
        retval.append(df.iloc[batch_nodes])

    return tuple(retval)


def split_dataset(df, batches, **_):
    if isinstance(batches, int):
        batches = [len(df) // batches] * batches
    start = 0
    retval = []
    for batch in batches:
        retval.append(df.iloc[start : start + batch, :])
        start += batch
    return tuple(retval)
