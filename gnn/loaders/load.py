import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_scipy_sparse_matrix

from gnn.loaders.util import split_dataset


def hash_dict(d):
    gen_hash = ""
    for key, val in d.items():
        gen_hash += f"{hash(key)}{hash(val)}"
    return hash(gen_hash)


def create_graph(
        df,
        n_neighbors=1,
        smoothing="laplacian",
        use_weights=False,
        hot=False,
        remove_mean=True,
        scaled=False,
        include_self=False,
        mode="connectivity",
        algorithm="minkowski"
):
    df_x = df.iloc[:, 1:]
    df_y = df['value']
    if remove_mean:
        df_y -= df_y.mean()

    if hot:
        column_count = len(df_x.columns)
        for i in range(1, column_count):
            df_x[i] = (df_x[i] == 1).astype(int)
        for i in range(1, column_count):
            df_x[i + column_count - 1] = (df_x[i] == 2).astype(int)

    if scaled:
        df_x -= df_x.mean()
        df_x /= df_x.std()
        df_x = df_x.fillna(0)

    x = torch.tensor(df_x.values.tolist(), dtype=torch.float)
    y = torch.tensor([[n] for n in df_y.values], dtype=torch.float)
    knn_dist_graph_train = kneighbors_graph(
        X=df_x,
        n_neighbors=n_neighbors,
        mode=mode,
        metric=algorithm,
        include_self=include_self,
        n_jobs=6
    )

    if smoothing == "laplacian":
        sigma = 1
        similarity_graph = sparse.csr_matrix(knn_dist_graph_train.shape)
        nonzeroindices = knn_dist_graph_train.nonzero()
        normalized = np.asarray(knn_dist_graph_train[nonzeroindices]/np.max(knn_dist_graph_train[nonzeroindices]))

        similarity_graph[nonzeroindices] = np.exp(-np.asarray(normalized) ** 2 / 2.0 * sigma ** 2)
        similarity_graph = 0.5 * (similarity_graph + similarity_graph.T)
        graph_laplacian_s = sparse.csgraph.laplacian(csgraph=similarity_graph, normed=False)
        edge_index, edge_weight = from_scipy_sparse_matrix(graph_laplacian_s)

        data = Data(x=x, y=y, edge_index=edge_index)

        if use_weights:
            data.edge_weight = 1-abs(edge_weight)/max(edge_weight).float()
    else:
        edge_index, edge_weight = from_scipy_sparse_matrix(knn_dist_graph_train)
        data = Data(x=x, y=y, edge_index=edge_index)
        if use_weights:
            data.edge_weight = 1-abs(edge_weight)/max(edge_weight).float()
    return data

class PlainGraph(InMemoryDataset):
    def __init__(
            self,
            root,
            bits,
            raw_filename,
            num_neighbours=1,
            smoothing="laplacian",
            mode="connectivity",
            use_validation=False,
            split=None,
            use_weights=False,
            algorithm="minkowski",
            validation_size=0.1,
            include_self=False,
            hot=False,
            scaled=False,
            remove_mean=True,
            batches=None,
            split_algorithm=split_dataset,
            split_algorithm_params=None
    ):
        self.use_validation = use_validation
        self.num_neighbours = num_neighbours
        self.smoothing = smoothing
        self.mode = mode
        self.raw_filename = raw_filename
        self.bits = bits
        self.split = split if split is not None else 0.8
        self.use_weights = use_weights
        self.algorithm = algorithm
        self.include_self = include_self
        self.validation_size = validation_size
        self.hot = hot
        self.scaled = scaled
        self.remove_mean = remove_mean
        self.batches = batches
        self.split_algorithm = split_algorithm
        self.split_algorithm_params = split_algorithm_params if split_algorithm_params is not None else {}
        super(PlainGraph, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if hasattr(self.data, "valid"):
            self.valid = self.data.valid

    @property
    def raw_file_names(self):
        return [self.raw_filename]

    @property
    def processed_file_names(self):
        bits = "whole" if self.bits is None else "bits"
        filename = self.raw_filename.replace("/", ":")
        algorithm = self.algorithm if isinstance(self.algorithm, str) else self.algorithm.__name__
        return [
            f'data-{filename}-'
            f'{self.split}-'
            f'{self.use_validation}-'
            f'{self.num_neighbours}-'
            f'{self.smoothing}-'
            f'{self.mode}-'
            f'{self.use_weights}-'
            f'{algorithm}-'
            f'{self.validation_size}-'
            f'{self.include_self}-'
            f'{self.hot}-'
            f'{self.scaled}-'
            f'{self.remove_mean}-'
            f'{self.batches}-'
            f'{self.split_algorithm.__name__}-'
            f'{hash_dict(self.split_algorithm_params)}-'
            f'{bits}.pt'
        ]

    def download(self):
        ...

    def process(self):
        data_list = []
        print({
            "use_validation": self.use_validation,
            "num_neighbours": self.num_neighbours,
            "smoothing": self.smoothing,
            "mode": self.mode,
            "raw_filename": self.raw_filename,
            "bits": self.bits,
            "split": self.split,
            "use_weights": self.use_weights,
            "algorithm": self.algorithm,
            "validation_size": self.validation_size,
            "include_self": self.include_self,
            "hot": self.hot,
            "scaled": self.scaled,
            "remove_mean": self.remove_mean,
            "batches": self.batches,
            "split_algorithm": self.split_algorithm.__name__,
            "split_algorithm_params": self.split_algorithm_params,
        })
        for filename in self.raw_file_names:
            with open(filename) as fp:
                line = fp.readline()
                column_count = len(line.split(","))
            value_columns = [str((i+1)) for i in range(column_count-1)]
            labels = ["value"] + value_columns
            df_whole = pd.read_csv(filename, names=labels)

            if self.bits is not None:
                df_whole = df_whole[["value"] + self.bits]
                df_whole.columns = ["value"] + list(range(1, len(self.bits)+1))

            data_len = df_whole.shape[0]
            split = int(data_len*self.split) if self.split < 1 else self.split

            train_set = split
            validation_size = (
                self.validation_size if isinstance(self.validation_size, int)
                else int(self.validation_size*(df_whole.shape[0]-split))
            )
            valid_set = validation_size if self.use_validation else 0

            df_train, df_test, df_valid = self.split_algorithm(
                df_whole,
                (train_set, df_whole.shape[0]-valid_set-train_set, valid_set),
                neighbours=self.num_neighbours,
                metric=self.algorithm,
                **self.split_algorithm_params,
            )
            test_data = create_graph(
                df_test,
                n_neighbors=self.num_neighbours,
                smoothing=self.smoothing,
                use_weights=self.use_weights,
                hot=self.hot,
                remove_mean=self.remove_mean,
                scaled=self.scaled,
                include_self=self.include_self,
                mode=self.mode,
                algorithm=self.algorithm
            )
            assert (test_data.edge_index.shape[0]) > 0
            if self.use_validation:
                valid_data = create_graph(
                    df_valid,
                    n_neighbors=self.num_neighbours,
                    smoothing=self.smoothing,
                    use_weights=self.use_weights,
                    hot=self.hot,
                    remove_mean=self.remove_mean,
                    scaled=self.scaled,
                    include_self=self.include_self,
                    mode=self.mode,
                    algorithm=self.algorithm
                )
                valid_data.test = valid_data
            if self.batches is not None:
                batch_data = self.split_algorithm(
                    df_whole,
                    self.batches,
                    neighbours=self.num_neighbours,
                    metric=self.algorithm,
                    **self.split_algorithm_params,
                )
                for batch in batch_data:
                    data = create_graph(
                        batch,
                        n_neighbors=self.num_neighbours,
                        smoothing=self.smoothing,
                        use_weights=self.use_weights,
                        hot=self.hot,
                        remove_mean=self.remove_mean,
                        scaled=self.scaled,
                        include_self=self.include_self,
                        mode=self.mode,
                        algorithm=self.algorithm
                    )
                    assert (data.edge_index.shape[0]) > 0
                    data.test = test_data
                    data_list.append(data)
                if self.use_validation:
                    data_list.append(valid_data)

            else:
                data = create_graph(
                    df_train,
                    n_neighbors=self.num_neighbours,
                    smoothing=self.smoothing,
                    use_weights=self.use_weights,
                    hot=self.hot,
                    remove_mean=self.remove_mean,
                    scaled=self.scaled,
                    include_self=self.include_self,
                    mode=self.mode,
                    algorithm=self.algorithm
                )
                if self.use_validation:
                    data.valid = valid_data
                assert (data.edge_index.shape[0]) > 0

                data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_data(
        filename,
        bits=None,
        num_neighbours=1,
        smoothing="laplacian",
        mode="connectivity",
        use_validation=False,
        split=None,
        use_weights=False,
        algorithm="minkowski",
        validation_size=0.1,
        hot=False,
        scaled=False,
        remove_mean=True,
        batches=None,
        split_algorithm=split_dataset,
        split_algorithm_params=None,
        **_
):

    return PlainGraph(
        ".",
        bits=bits,
        raw_filename=filename,
        num_neighbours=num_neighbours,
        smoothing=smoothing,
        mode=mode,
        use_validation=use_validation,
        split=split,
        use_weights=use_weights,
        algorithm=algorithm,
        validation_size=validation_size,
        hot=hot,
        scaled=scaled,
        remove_mean=remove_mean,
        batches=batches,
        split_algorithm=split_algorithm,
        split_algorithm_params=split_algorithm_params
    )
