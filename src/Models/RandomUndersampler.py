import torch
from torch.utils.data import RandomSampler, Sampler

class CustomRandomSubsetSampler(RandomSampler):
    def __init__(self, indices, replacement=False, num_samples=None):
        RandomSampler.__init__(self, indices, replacement, num_samples)

    def __iter__(self):
        for index in RandomSampler.__iter__(self):
            yield self.data_source[index]

class RandomUndersampler(Sampler):
    def __init__(self, labels: torch.tensor):
        if len(labels.shape) > 1:
            raise ValueError(
                "labels can only have a single dimension (N, ), got shape: {}".format(
                    labels.shape
                )
            )
        self.tensors = [
            torch.nonzero(labels == i, as_tuple=False).flatten()
            for i in torch.unique(labels)
        ]
        self.samples_per_label = min(map(len, self.tensors))

    @property
    def num_samples(self):
        return self.samples_per_label * len(self.tensors)

    def __iter__(self):
        samplers = [
            iter(
                CustomRandomSubsetSampler(
                    tensor,
                    replacement=len(tensor) < self.samples_per_label,
                    num_samples=self.samples_per_label
                    if len(tensor) < self.samples_per_label
                    else None,
                )
            )
            for tensor in self.tensors
        ]
        for _ in range(self.samples_per_label):
            for index in torch.randperm(len(samplers)).tolist():
                yield next(samplers[index])

    def __len__(self):
        return self.num_samples
