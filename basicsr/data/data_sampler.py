import random
import math
from functools import reduce
from typing import List
import torch
from torch.utils.data.sampler import Sampler
import pandas as pd


class EnlargedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """

    def __init__(self, dataset, num_replicas, rank, ratio=1):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class BalancedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        path_labels (str): Path to the labels.csv to use for class-balancing. labels.csv => Path,x,y,z where x,y,z are the balancing cols.
        balancing_cols (list[str]): Column names to use for balancing. Default: ("Age", "Ethnicity", "Gender")
        ratio (int): Enlarging ratio. Default: 1.
    """

    def __init__(self, dataset, num_replicas, rank, path_labels: str, ratio=1, balancing_cols=("Age", "Ethnicity", "Gender")):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        random.seed(self.epoch)
        self.balancing_cols = balancing_cols
        self._init_labels(path_labels)

    def _init_labels(self, path_labels: str) -> None:
        self.labels = pd.read_csv(path_labels)
        for col in self.balancing_cols:
            self.labels[col] = self.labels[col].astype(int)

    def _get_random_paths(self) -> List[str]:
        conditions = []
        for col in self.balancing_cols:
            value_selected = random.randint(self.labels[col].min(), self.labels[col].max())
            condition = self.labels[col] == value_selected
            conditions.append(condition)
        paths = self.labels[reduce(lambda x, y: x & y, conditions)]["Path"].to_list()
        return paths

    def __iter__(self):
        """For each sample peak a random combination of labels,
        find the samples with such combination and randomly select one"""
        indices = []
        for _ in range(self.num_samples):
            paths_random = None

            # Until a valid paths_random is found
            while paths_random is None or not paths_random:
                # Find paths by picking a label combination first
                paths_random = self._get_random_paths()

            # Select a random path from the available ones
            path_selected = random.choice(paths_random).replace("\\", "/")
            # Get the index from path
            index_selected = int(path_selected.split("/")[-1].split(".")[0])
            indices.append(index_selected)
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch