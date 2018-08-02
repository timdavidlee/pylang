from torch import Dataset
import numpy as np


class TextDataset(Dataset):
    """
    x = features or source set
    y = target
    """
    def __init__(self, x, y, backwards=False, start_of_sent_token=None, end_of_sent_token=None):
        self.x = x
        self.y = y
        self.backwards = backwards
        self.start_of_sent_token = start_of_sent_token
        self.end_of_sent_token = end_of_sent_token

    def __getitem__(self, idx):
        x = self.x[idx]

        if self.backwards:
            x = list(reversed(x))

        if self.end_of_sent_token is not None:
            x = x + [self.end_of_sent_token]

        if self.start_of_sent_token is not None:
            x = [self.start_of_sent_token] + x
        return np.array(x), self.y[idx]

    def __len__(self):
        return len(self.x)


"""
PYTORCH SAMPLER
https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html
"""


class Sampler(object):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SortSampler(Sampler):
    """
    Standard sampler, this one sorts the data by length
    this assumes that the data source is a list of items
    that length can be determined, e.g. a word, or list of
    lists
    """
    def __init__(self, data_source, key):
        self.data_source = data_source
        self.key = key

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        return iter(sorted(range(len(self.data_source)), key=self.key, reverse=True))


class SortishSampler(Sampler):
    """Returns an iterator that traverses the the data in randomly ordered batches that are approximately the same size.
    The max key size batch is always returned in the first call because of pytorch cuda memory allocation sequencing.
    Without that max key returned first multiple buffers may be allocated when the first created isn't large enough
    to hold the next in the sequence.
    """
    def __init__(self, data_source, key, bs):
        self.data_source = data_source
        self.key = key
        self.bs = bs

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data_source))

        # 50 is the max length of a sequence
        # so a total batch will be BATCHSIZE x 50
        sz = self.bs * 50

        ck_idx = [idxs[i:i + sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=self.key, reverse=True) for s in ck_idx])
        sz = self.bs
        ck_idx = [sort_idx[i:i + sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
        ck_idx[0],ck_idx[max_ck] = ck_idx[max_ck],ck_idx[0]     # then make sure it goes first.
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:]))
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)
