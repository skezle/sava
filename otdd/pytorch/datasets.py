import os
import pdb
from functools import partial
import random
import logging
import string
import copy
import pickle

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.utils.data as torchdata
import torch.utils.data.dataloader as dataloader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image

from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.datasets as dset

# import torchtext
# from torchtext.data.utils import get_tokenizer
from copy import deepcopy as dpcp

import h5py

from .. import DATA_DIR

from .utils import (
    interleave,
    process_device_arg,
    random_index_split,
    spectrally_prescribed_matrix,
    rot,
    rot_evecs,
)

from .sqrtm import create_symm_matrix

logger = logging.getLogger(__name__)


DATASET_NCLASSES = {
    "MNIST": 10,
    "FashionMNIST": 10,
    "EMNIST": 26,
    "KMNIST": 10,
    "USPS": 10,
    "CIFAR10": 10,
    "SVHN": 10,
    "STL10": 10,
    "LSUN": 10,
    "tiny-ImageNet": 200,
}

DATASET_SIZES = {
    "MNIST": (28, 28),
    "FashionMNIST": (28, 28),
    "EMNIST": (28, 28),
    "QMNIST": (28, 28),
    "KMNIST": (28, 28),
    "USPS": (16, 16),
    "SVHN": (32, 32),
    "CIFAR10": (32, 32),
    "STL10": (96, 96),
    "tiny-ImageNet": (64, 64),
}

DATASET_NORMALIZATION = {
    "MNIST": ((0.1307,), (0.3081,)),
    "USPS": ((0.1307,), (0.3081,)),
    "FashionMNIST": ((0.1307,), (0.3081,)),
    "QMNIST": ((0.1307,), (0.3081,)),
    "EMNIST": ((0.1307,), (0.3081,)),
    "KMNIST": ((0.1307,), (0.3081,)),
    "ImageNet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "tiny-ImageNet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #"CIFAR10": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "CIFAR100": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "STL10": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def sort_by_label(X, Y):
    idxs = np.argsort(Y)
    return X[idxs, :], Y[idxs]


### Data Transforms
class DiscreteRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class CustomTensorDataset1(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            # print('bf',x.shape)
            x = Image.fromarray(x.astype(np.uint8))
            x = self.transform(x)
            # print('aft',x.shape)
        return x, y

    def __len__(self):
        return len(self.data)


class SubsetSampler(torch.utils.data.Sampler):
    r"""Samples elements in order (not randomly) from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
        (this is identical to torch's SubsetRandomSampler except not random)
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class CustomTensorDataset2(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.data = tensors[0]
        self.targets = tensors[1]
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        
        if self.transform:
            x = self.transform(x)

        y = torch.tensor(self.tensors[1][index]).long()

        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]


class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None, target_transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


class CustomTensorDataset3(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None, target_transform=None):
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.tensors[index][0]
        if self.transform:
            x = self.transform(x)

        y = self.tensors[index][1]
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def set_label(self, index, to_label):
        self.tensors[index] = (self.tensors[index][0], to_label)

    def __len__(self):
        return len(self.tensors)


class SubsetFromLabels(torch.utils.data.dataset.Dataset):
    """Subset of a dataset at specified indices.

    Adapted from torch.utils.data.dataset.Subset to allow for label re-mapping
    without having to copy whole dataset.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        targets_map (dict, optional):  Dictionary to map targets with
    """

    def __init__(self, dataset, labels, remap=False):
        self.dataset = dataset
        self.labels = labels
        self.classes = [dataset.classes[i] for i in labels]
        self.mask = np.isin(dataset.targets, labels).squeeze()
        self.indices = np.where(self.mask)[0]
        self.remap = remap
        targets = dataset.targets[self.indices]
        if remap:
            V = sorted(np.unique(targets))
            assert list(V) == list(labels)
            targets = torch.tensor(np.digitize(targets, self.labels, right=True))
            self.tmap = dict(zip(V, range(len(V))))
        self.targets = targets

    def __getitem__(self, idx):
        if self.remap is False:
            return self.dataset[self.indices[idx]]
        else:
            item = self.dataset[self.indices[idx]]
            return (item[0], self.tmap[item[1]])

    def __len__(self):
        return len(self.indices)


def subdataset_from_labels(dataset, labels, remap=True):
    mask = np.isin(dataset.targets, labels).squeeze()
    idx = np.where(mask)[0]
    subdataset = Subset(dataset, idx, remap_targets=True)
    return subdataset


def dataset_from_numpy(X, Y, classes=None):
    targets = torch.LongTensor(list(Y))
    ds = TensorDataset(torch.from_numpy(X).type(torch.FloatTensor), targets)
    ds.targets = targets
    ds.classes = (
        classes if classes is not None else [i for i in range(len(np.unique(Y)))]
    )
    return ds


gmm_configs = {
    "star": {
        "means": [
            torch.Tensor([0, 0]),
            torch.Tensor([0, -2]),
            torch.Tensor([2, 0]),
            torch.Tensor([0, 2]),
            torch.Tensor([-2, 0]),
        ],
        "covs": [
            spectrally_prescribed_matrix([1, 1], torch.eye(2)),
            spectrally_prescribed_matrix([2.5, 1], torch.eye(2)),
            spectrally_prescribed_matrix([1, 20], torch.eye(2)),
            spectrally_prescribed_matrix([10, 1], torch.eye(2)),
            spectrally_prescribed_matrix([1, 5], torch.eye(2)),
        ],
        "spread": 6,
    }
}


def make_gmm_dataset(
    config="random",
    classes=10,
    dim=2,
    samples=10,
    spread=1,
    shift=None,
    rotate=None,
    diagonal_cov=False,
    shuffle=True,
):
    """Generate Gaussian Mixture Model datasets.

    Arguments:
        config (str): determines cluster locations, one of 'random' or 'star'
        classes (int): number of classes in dataset
        dim (int): feature dimension of dataset
        samples (int): number of samples in dataset
        spread (int): separation of clusters
        shift (bool): whether to add a shift to dataset
        rotate (bool): whether to rotate dataset
        diagonal_cov(bool): whether to use a diagonal covariance matrix
        shuffle (bool): whether to shuffle example indices

    Returns:
        X (tensor): tensor of size (samples, dim) with features
        Y (tensor): tensor of size (samples, 1) with labels
        distribs (torch.distributions): data-generating distributions of each class

    """
    means, covs, distribs = [], [], []
    _configd = None if config == "random" else gmm_configs[config]
    spread = (
        spread
        if (config == "random" or not "spread" in _configd)
        else _configd["spread"]
    )
    shift = (
        shift if (config == "random" or not "shift" in _configd) else _configd["shift"]
    )

    for i in range(classes):
        if config == "random":
            mean = torch.randn(dim)
            cov = create_symm_matrix(1, dim, verbose=False).squeeze()
        elif config == "star":
            mean = gmm_configs["star"]["means"][i]
            cov = gmm_configs["star"]["covs"][i]
        if rotate:
            mean = rot(mean, rotate)
            cov = rot_evecs(cov, rotate)

        if diagonal_cov:
            cov.masked_fill_(~torch.eye(dim, dtype=bool), 0)

        means.append(spread * mean)
        covs.append(cov)
        distribs.append(MultivariateNormal(means[-1], covs[-1]))

    X = torch.cat([P.sample(sample_shape=torch.Size([samples])) for P in distribs])
    Y = torch.LongTensor([samples * [i] for i in range(classes)]).flatten()

    if shift:
        X += torch.tensor(shift)

    if shuffle:
        idxs = torch.randperm(Y.shape[0])
        X = X[idxs, :]
        Y = Y[idxs]
    return X, Y, distribs

def load_torchvision_data_shuffle(
    dataname,
    valid_size=0.1,
    splits=None,
    shuffle=True,
    stratified=False,
    stratified_manual=False,
    random_seed=None,
    batch_size=64,
    resize=None,
    to3channels=False,
    maxsize=None,
    maxsize_test=None,
    num_workers=0,
    transform=None,
    data=None,
    datadir=None,
    download=True,
    filt=False,
    print_stats=False,
    shuffle_per=0,
):
    """Load torchvision datasets.

    We return train and test for plots and post-training experiments
    """
    if stratified_manual:
        assert shuffle is not True
    
    if shuffle == True and random_seed:
        np.random.seed(random_seed)
    elif random_seed:
        np.random.seed(random_seed)
    if transform is None:
        if dataname in DATASET_NORMALIZATION.keys():
            transform_dataname = dataname
        else:
            transform_dataname = "ImageNet"

        transform_list = []

        if dataname in ["MNIST", "USPS"] and to3channels:
            transform_list.append(torchvision.transforms.Grayscale(3))

        transform_list.append(torchvision.transforms.ToTensor())
        transform_list.append(
            torchvision.transforms.Normalize(*DATASET_NORMALIZATION[transform_dataname])
        )

        if resize:
            if not dataname in DATASET_SIZES or DATASET_SIZES[dataname][0] != resize:
                ## Avoid adding an "identity" resizing
                transform_list.insert(0, transforms.Resize((resize, resize)))

        transform = transforms.Compose(transform_list)
        logger.info(transform)
        train_transform, valid_transform = transform, transform
    elif data is None:
        if len(transform) == 1:
            train_transform, valid_transform = transform, transform
        elif len(transform) == 2:
            train_transform, valid_transform = transform
        else:
            raise ValueError()

    if data is None:
        DATASET = getattr(torchvision.datasets, dataname)
        if datadir is None:
            datadir = DATA_DIR
        if dataname == "EMNIST":
            split = "letters"
            train = DATASET(
                datadir,
                split=split,
                train=True,
                download=download,
                transform=train_transform,
            )
            test = DATASET(
                datadir,
                split=split,
                train=False,
                download=download,
                transform=valid_transform,
            )
            ## EMNIST seems to have a bug - classes are wrong
            _merged_classes = set(
                [
                    "C",
                    "I",
                    "J",
                    "K",
                    "L",
                    "M",
                    "O",
                    "P",
                    "S",
                    "U",
                    "V",
                    "W",
                    "X",
                    "Y",
                    "Z",
                ]
            )
            _all_classes = set(list(string.digits + string.ascii_letters))
            classes_split_dict = {
                "byclass": list(_all_classes),
                "bymerge": sorted(list(_all_classes - _merged_classes)),
                "balanced": sorted(list(_all_classes - _merged_classes)),
                "letters": list(string.ascii_lowercase),
                "digits": list(string.digits),
                "mnist": list(string.digits),
            }
            train.classes = classes_split_dict[split]
            if split == "letters":
                ## The letters fold (and only that fold!!!) is 1-indexed
                train.targets -= 1
                test.targets -= 1
        elif dataname == "STL10":
            train = DATASET(
                datadir, split="train", download=download, transform=train_transform
            )
            test = DATASET(
                datadir, split="test", download=download, transform=valid_transform
            )
            train.classes = [
                "airplane",
                "bird",
                "car",
                "cat",
                "deer",
                "dog",
                "horse",
                "monkey",
                "ship",
                "truck",
            ]
            test.classes = train.classes
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(test.labels)
        elif dataname == "SVHN":
            train = DATASET(
                datadir, split="train", download=download, transform=train_transform
            )
            test = DATASET(
                datadir, split="test", download=download, transform=valid_transform
            )
            ## In torchvision, SVHN 0s have label 0, not 10
            train.classes = test.classes = [str(i) for i in range(10)]
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(train.labels)
        elif dataname == "LSUN":
            # pdb.set_trace()
            train = DATASET(
                datadir, classes="train", download=download, transform=train_transform
            )
        else:
            train = DATASET(
                datadir, train=True, download=download, transform=train_transform
            )
            test = DATASET(
                datadir, train=False, download=download, transform=valid_transform
            )
            #pdb.set_trace()
    else:
        train, test = data

    if type(train.targets) is list:
        train.targets = torch.LongTensor(train.targets)
        test.targets = torch.LongTensor(test.targets)

    if not hasattr(train, "classes") or not train.classes:
        train.classes = sorted(torch.unique(train.targets).tolist())
        test.classes = sorted(torch.unique(train.targets).tolist())

    ###### VALIDATION IS 0 SO NOT WORRY NOW ######
    ### Data splitting
    fold_idxs = {}
    if splits is None and valid_size == 0:
        ## Only train
        if stratified_manual:
            ## If stratified, we'll interleave the per-class shuffled indices
            idxs_class = [
                np.random.permutation(np.where(train.targets == c)).T
                for c in np.unique(train.targets)
            ]
            idxs = interleave(*idxs_class).squeeze().astype(int)
            fold_idxs["train"] = idxs
        else:
            fold_idxs["train"] = np.arange(len(train))

    elif splits is None and valid_size > 0:
        ## Train/Valid
        valid_prop = valid_size / len(train)
        train_idx, valid_idx = random_index_split(
            len(train), 1 - valid_prop, (maxsize, valid_size)
        )  # No maxsize for validation
        fold_idxs["train"] = train_idx
        fold_idxs["valid"] = valid_idx
    elif splits is not None:
        ## Custom splits - must be integer.
        if type(splits) is dict:
            snames, slens = zip(*splits.items())
        elif type(splits) in [list, np.ndarray]:
            snames = ["split_{}".format(i) for i in range(len(splits))]
            slens = splits
        slens = np.array(slens)
        if any(slens < 0):  # Split expressed as -1, i.e., 'leftover'
            assert sum(slens < 0) == 1, "Can only deal with one split being -1"
            idx_neg = np.where(slens == -1)[0][0]
            slens[idx_neg] = len(train) - np.array([x for x in slens if x > 0]).sum()
        elif slens.sum() > len(train):
            logging.warning("Not enough samples to satify splits..cropping train...")
            if "train" in snames:
                slens[snames.index("train")] = (
                    len(train) - slens[np.array(snames) != "train"].sum()
                )

        idxs = np.arange(len(train))
        if not stratified:
            np.random.shuffle(idxs)
        else:
            ## If stratified, we'll interleave the per-class shuffled indices
            idxs_class = [
                np.random.permutation(np.where(train.targets == c)).T
                for c in np.unique(train.targets)
            ]
            idxs = interleave(*idxs_class).squeeze().astype(int)

        slens = np.array(slens).cumsum()  # Need to make cumulative for np.split
        split_idxs = [
            np.sort(s) for s in np.split(idxs, slens)[:-1]
        ]  # The last one are leftovers
        assert len(split_idxs) == len(splits)
        fold_idxs = {snames[i]: v for i, v in enumerate(split_idxs)}

    # fold_idxs['train'] = np.arange(len(train)) start -> stop by step
    for k, idxs in fold_idxs.items():
        if maxsize and maxsize < len(idxs):
            if stratified_manual:
                fold_idxs[k] = idxs[:maxsize] # indexes inside each class are permuted
            else:
                fold_idxs[k] = np.sort(np.random.choice(idxs, maxsize, replace=False))

    sampler_class = SubsetRandomSampler if shuffle else SubsetSampler
    fold_samplers = {k: sampler_class(idxs) for k, idxs in fold_idxs.items()}

    if shuffle_per != 0:
        total_shuffles = int(shuffle_per * len(fold_idxs["train"]))

        shuffle_inds = np.random.choice(
            sorted(fold_idxs["train"]), size=total_shuffles, replace=False
        )

        if dataname == "CIFAR10":
            print("CIFAR TEN")
            for index in shuffle_inds:
                cur_label = train.targets[index]
                new_label = np.random.randint(10)
                while new_label == cur_label:
                    new_label = np.random.randint(10)
                cur_label = new_label
                # print("TRAINNNN: ", train[index])
                train.targets[index] = cur_label
        elif dataname == "CIFAR100":
            print("CIFAR HUNDRED")
            for index in shuffle_inds:
                cur_label = train.targets[index]
                new_label = np.random.randint(100)
                while new_label == cur_label:
                    new_label = np.random.randint(100)
                cur_label = new_label
                # print("TRAINNNN: ", train[index])
                train.targets[index] = cur_label
        elif dataname == "MNIST":
            print("MNIST")
            for index in shuffle_inds:
                cur_label = train.targets[index]
                print(f"Currrent label: {cur_label}")
                new_label = np.random.randint(10)
                while new_label == cur_label:
                    new_label = np.random.randint(10)
                cur_label = new_label
                print(f"New label: {cur_label} ")
                # print("TRAINNNN: ", train[index])
                train.targets[index] = cur_label
                print("TRAINNNN label: ", train.targets[index])
                print("TRAINNNN: ", train[index])
        elif dataname == "FashionMNIST":
            print("FashionistaMNIST")
            for index in shuffle_inds:
                cur_label = train.targets[index]
                new_label = np.random.randint(10)
                while new_label == cur_label:
                    new_label = np.random.randint(10)
                cur_label = new_label
                # print("TRAINNNN: ", train[index])
                train.targets[index] = cur_label

        ########## FOR other datasets such as STL10 and ImageNet, we cannot directly modify labels
        ########## so will need to recreate the dataloader! time consuming!

        elif dataname == "STL10" or dataname == "ImageNet":
            print("STL11")
            if dataname == "ImageNet":
                print("IMAGI")
                DATASET = getattr(torchvision.datasets, dataname)
            new_train = DATASET
            new_train.targets = {}
            new_train.classes = {}
            new_train.targets = train.targets
            new_train.classes = train.classes

            new_ds_imgs = []
            new_ds_labs = []
            class_len = len(train.classes)
            for i in range(len(train)):
                new_ds_imgs.append(train[i][0].permute(1, 2, 0))
                if i in shuffle_inds:
                    cur_label = train.targets[i]
                    new_label = np.random.randint(class_len)
                    #                     print(f'{i}.Currrent label: {cur_label} ')
                    while new_label == cur_label:
                        new_label = np.random.randint(class_len)
                    cur_label = new_label
                    train.targets[i] = cur_label
                    #                     print(f'{i}.New label: {cur_label} ')
                    new_ds_labs.append(torch.tensor(cur_label).reshape(1))
                else:
                    new_ds_labs.append(torch.tensor(train[i][1]).reshape(1))
            new_ds_imgs = torch.stack(new_ds_imgs, dim=0)
            new_ds_labs = torch.cat(new_ds_labs)
            new_ds_imgs = new_ds_imgs.numpy()
            new_ds_labs = new_ds_labs.numpy()

            new_ds = (new_ds_imgs, new_ds_labs)

            new_train.targets = train.targets
            new_transform_list = []
            new_transform_list.append(torchvision.transforms.ToTensor())
            new_transform = transforms.Compose(new_transform_list)
            new_train = CustomTensorDataset2(new_ds, transform=new_transform)
            train = new_train

            if type(train.targets) is np.ndarray:
                train.targets = train.targets.tolist()

            if type(train.targets) is list:
                train.targets = torch.LongTensor(train.targets)

            if not hasattr(train, "classes") or not train.classes:
                #                 train.classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
                train.classes = sorted(torch.unique(train.targets).tolist())

    ### Create DataLoaders
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers)

    fold_loaders = {
        k: dataloader.DataLoader(train, sampler=sampler, **dataloader_args)
        for k, sampler in fold_samplers.items()
    }

    if maxsize_test and maxsize_test < len(test):
        test_idxs = np.sort(np.random.choice(len(test), maxsize_test, replace=False))
        sampler_test = SubsetSampler(test_idxs)  # For test don't want Random
        dataloader_args["sampler"] = sampler_test
    else:
        dataloader_args["shuffle"] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    fold_loaders["test"] = test_loader

    fnames, flens = zip(*[[k, len(v)] for k, v in fold_idxs.items()])
    fnames = "/".join(list(fnames) + ["test"])
    flens = "/".join(map(str, list(flens) + [len(test)]))

    if hasattr(train, "data"):
        logger.info("Input Dim: {}".format(train.data.shape[1:]))
    logger.info(
        "Classes: {} (effective: {})".format(
            len(train.classes), len(torch.unique(train.targets))
        )
    )

    if shuffle_per != 0:
        return fold_loaders, {"train": train, "test": test}, shuffle_inds
    return fold_loaders, {"train": train, "test": test}


def load_torchvision_data_nonstat_shuffle(
    dataname,
    valid_size=0.1,
    splits=None,
    shuffle=True,
    stratified=False,
    random_seed=None,
    batch_size=64,
    resize=None,
    to3channels=False,
    maxsize=None,
    maxsize_test=None,
    num_workers=0,
    transform=None,
    data=None,
    datadir=None,
    download=True,
    filt=False,
    print_stats=False,
    shuffle_per=0,
):
    """Load torchvision datasets.

    We return train and test for plots and post-training experiments
    """

    assert shuffle_per > 0.0

    if shuffle == True and random_seed:
        np.random.seed(random_seed)
    elif random_seed:
        np.random.seed(random_seed)
    if transform is None:
        if dataname in DATASET_NORMALIZATION.keys():
            transform_dataname = dataname
        else:
            transform_dataname = "ImageNet"

        transform_list = []

        if dataname in ["MNIST", "USPS"] and to3channels:
            transform_list.append(torchvision.transforms.Grayscale(3))

        transform_list.append(torchvision.transforms.ToTensor())
        transform_list.append(
            torchvision.transforms.Normalize(*DATASET_NORMALIZATION[transform_dataname])
        )

        if resize:
            if not dataname in DATASET_SIZES or DATASET_SIZES[dataname][0] != resize:
                ## Avoid adding an "identity" resizing
                transform_list.insert(0, transforms.Resize((resize, resize)))

        transform = transforms.Compose(transform_list)
        logger.info(transform)
        train_transform, valid_transform = transform, transform
    elif data is None:
        if len(transform) == 1:
            train_transform, valid_transform = transform, transform
        elif len(transform) == 3:
            train_transform_selection, train_transform, valid_transform = transform
        else:
            raise ValueError()

    if data is None:
        DATASET = getattr(torchvision.datasets, dataname)
        if datadir is None:
            datadir = DATA_DIR
        if dataname == "EMNIST":
            split = "letters"
            train = DATASET(
                datadir,
                split=split,
                train=True,
                download=download,
                transform=train_transform,
            )
            test = DATASET(
                datadir,
                split=split,
                train=False,
                download=download,
                transform=valid_transform,
            )
            ## EMNIST seems to have a bug - classes are wrong
            _merged_classes = set(
                [
                    "C",
                    "I",
                    "J",
                    "K",
                    "L",
                    "M",
                    "O",
                    "P",
                    "S",
                    "U",
                    "V",
                    "W",
                    "X",
                    "Y",
                    "Z",
                ]
            )
            _all_classes = set(list(string.digits + string.ascii_letters))
            classes_split_dict = {
                "byclass": list(_all_classes),
                "bymerge": sorted(list(_all_classes - _merged_classes)),
                "balanced": sorted(list(_all_classes - _merged_classes)),
                "letters": list(string.ascii_lowercase),
                "digits": list(string.digits),
                "mnist": list(string.digits),
            }
            train.classes = classes_split_dict[split]
            if split == "letters":
                ## The letters fold (and only that fold!!!) is 1-indexed
                train.targets -= 1
                test.targets -= 1
        elif dataname == "SVHN":
            train = DATASET(
                datadir, split="train", download=download, transform=train_transform
            )
            test = DATASET(
                datadir, split="test", download=download, transform=valid_transform
            )
            ## In torchvision, SVHN 0s have label 0, not 10
            train.classes = test.classes = [str(i) for i in range(10)]
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(train.labels)
        else:
            train_selection = DATASET(
                datadir, train=True, download=download, transform=train_transform_selection,
            )
            train = DATASET(
                datadir, train=True, download=download, transform=train_transform,
            )
            test = DATASET(
                datadir, train=False, download=download, transform=valid_transform,
            )
    else:
        train, test = data


    if type(train_selection.targets) is list:
        train.targets = torch.LongTensor(train.targets)
        train_selection.targets = torch.LongTensor(train_selection.targets)
        test.targets = torch.LongTensor(test.targets)

    if not hasattr(train, "classes") or not train.classes:
        train.classes = sorted(torch.unique(train.targets).tolist())
        train_selection.classes = sorted(torch.unique(train_selection.targets).tolist())
        test.classes = sorted(torch.unique(train.targets).tolist())

    ### Data splitting
    fold_idxs = {}
    if splits is None and valid_size == 0:
        ## Only train
        fold_idxs["train"] = np.arange(len(train))

    elif splits is None and valid_size > 0:
        ## Train/Valid
        train_idx, valid_idx = random_index_split(
            len(train), 1 - valid_size, (maxsize, valid_size)
        )  # No maxsize for validation
        fold_idxs["train"] = train_idx
        fold_idxs["valid"] = valid_idx
    elif splits == 5 and valid_size > 0:
        # split D_train into 5 tasks
        # sample indicies for train and train_selection
        indices = np.arange(len(train))
        np.random.shuffle(indices) # inplace
        split = len(train) - valid_size
        train_idx, valid_idx = np.array(indices[:split]), np.array(indices[split:])
        fold_idxs["valid"] = valid_idx
        fold_idxs["train"] = train_idx
        for i in range(splits):
            end = int((i + 1) * (split / 5))
            fold_idxs[f"train_sel_{i}"] = train_idx[:end]
            fold_idxs[f"train_{i}"] = train_idx[:end]
    elif splits == 5 and valid_size == 0:
        # split D_train into 5 tasks
        # sample indicies for train and train_selection
        indices = np.arange(len(train))
        np.random.shuffle(indices) # inplace        
        fold_idxs["train"] = indices
        for i in range(splits):
            end = int((i + 1) * (len(train) / 5))
            fold_idxs[f"train_sel_{i}"] = indices[:end]
            fold_idxs[f"train_{i}"] = indices[:end]
    elif splits is not None:
        ## Custom splits - must be integer.
        if type(splits) is dict:
            snames, slens = zip(*splits.items())
        elif type(splits) in [list, np.ndarray]:
            snames = ["split_{}".format(i) for i in range(len(splits))]
            slens = splits
        slens = np.array(slens)
        if any(slens < 0):  # Split expressed as -1, i.e., 'leftover'
            assert sum(slens < 0) == 1, "Can only deal with one split being -1"
            idx_neg = np.where(slens == -1)[0][0]
            slens[idx_neg] = len(train) - np.array([x for x in slens if x > 0]).sum()
        elif slens.sum() > len(train):
            logging.warning("Not enough samples to satify splits..cropping train...")
            if "train" in snames:
                slens[snames.index("train")] = (
                    len(train) - slens[np.array(snames) != "train"].sum()
                )

        idxs = np.arange(len(train))
        if not stratified:
            np.random.shuffle(idxs)
        else:
            ## If stratified, we'll interleave the per-class shuffled indices
            idxs_class = [
                np.random.permutation(np.where(train.targets == c)).T
                for c in np.unique(train.targets)
            ]
            idxs = interleave(*idxs_class).squeeze().astype(int)

        slens = np.array(slens).cumsum()  # Need to make cumulative for np.split
        split_idxs = [
            np.sort(s) for s in np.split(idxs, slens)[:-1]
        ]  # The last one are leftovers
        assert len(split_idxs) == len(splits)
        fold_idxs = {snames[i]: v for i, v in enumerate(split_idxs)}

    # if maxsize has been defined, clip the size of the training datasets.
    for k, idxs in fold_idxs.items():
        if maxsize and maxsize < len(idxs):
            # for full train we keep the same maxsize, for the folds with devide by the number fo tasks
            if k in ["train", "valid"]:
                fold_idxs[k] = np.sort(idxs[:maxsize])
    
    for i in range(splits):
        k = f"train_sel_{i}"
        idxs = fold_idxs[k]
        if maxsize and maxsize < len(idxs):
            end = int((i + 1) * (maxsize / 5))
            fold_idxs[k] = idxs[:end]
        k = f"train_{i}"
        idxs = fold_idxs[k]
        if maxsize and maxsize < len(idxs):
            end = int((i + 1) * (maxsize / 5))
            fold_idxs[k] = idxs[:end]
    

    # Define subsampler for for train and validation splits
    sampler_class = SubsetRandomSampler if shuffle else SubsetSampler
    fold_samplers = {k: sampler_class(idxs) for k, idxs in fold_idxs.items()}

    total_shuffles = int(shuffle_per * len(fold_idxs["train"]))

    shuffle_inds = np.random.choice(
        sorted(fold_idxs["train"]), size=total_shuffles, replace=False
    )

    # shuffle labels for train and train_selection
    if dataname == "CIFAR10":
        print("CIFAR TEN")
        # train
        for index in shuffle_inds:
            cur_label = train.targets[index]
            new_label = np.random.randint(10)
            while new_label == cur_label:
                new_label = np.random.randint(10)
            cur_label = new_label
            train.targets[index] = cur_label
        # train data selection
        for index in shuffle_inds:
            cur_label = train_selection.targets[index]
            new_label = np.random.randint(10)
            while new_label == cur_label:
                new_label = np.random.randint(10)
            cur_label = new_label
            train_selection.targets[index] = cur_label
    else:
        raise ValueError

    # Create DataLoaders wityh samplers
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers)

    fold_loaders = {}
    for k, sampler in fold_samplers.items():
        if "sel" in k:
            fold_loaders[k] = dataloader.DataLoader(train_selection, sampler=sampler, **dataloader_args)
        else:
            fold_loaders[k] = dataloader.DataLoader(train, sampler=sampler, **dataloader_args)

    # if we have defined a maxsize for the eval sets
    if maxsize_test and maxsize_test < len(test):
        test_idxs = np.sort(np.random.choice(len(test), maxsize_test, replace=False))
        sampler_test = SubsetSampler(test_idxs)  # For test don't want Random
        dataloader_args["sampler"] = sampler_test
    else:
        dataloader_args["shuffle"] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    fold_loaders["test"] = test_loader

    if hasattr(train, "data"):
        logger.info("Input Dim: {}".format(train.data.shape[1:]))
    logger.info(
        "Classes: {} (effective: {})".format(
            len(train.classes), len(torch.unique(train.targets))
        )
    )

    # fold loaders has the data loaders with subsets for maxsize
    # also has the test loaders with the subsets for maxsize_test
    # train and test are the full torch dataset.Datasets
    # shuffle inds are the ground truth indices which have need perturbed with noisy labels
    return fold_loaders, {"train": train, "test": test}, shuffle_inds
    
def load_torchvision_data_perturb(
    dataname,
    valid_size=0.1,
    splits=None,
    shuffle=True,
    stratified=False,
    stratified_manual=False,
    random_seed=None,
    batch_size=64,
    resize=None,
    to3channels=False,
    maxsize=None,
    maxsize_test=None,
    num_workers=0,
    transform=None,
    data=None,
    datadir=None,
    download=True,
    shuffle_per=0,
    perturb_per=0,
):
    """Load torchvision datasets.

    Training data is Gaussian noise added to the images.
    """
    assert perturb_per > 0.0

    if stratified_manual:
        assert shuffle is not True

    if shuffle == True and random_seed:
        np.random.seed(random_seed)
    elif random_seed:
        np.random.seed(random_seed)
    if transform is None:
        if dataname in DATASET_NORMALIZATION.keys():
            transform_dataname = dataname
        else:
            transform_dataname = "ImageNet"

        transform_list = []

        if dataname in ["MNIST", "USPS"] and to3channels:
            transform_list.append(torchvision.transforms.Grayscale(3))

        transform_list.append(torchvision.transforms.ToTensor())
        # transform_list.append(
        #    torchvision.transforms.Normalize(*DATASET_NORMALIZATION[transform_dataname])
        # )

        if resize:
            if not dataname in DATASET_SIZES or DATASET_SIZES[dataname][0] != resize:
                ## Avoid adding an "identity" resizing
                transform_list.insert(0, transforms.Resize((resize, resize)))

        transform = transforms.Compose(transform_list)
        logger.info(transform)
        train_transform, valid_transform = transform, transform
    else:
        if len(transform) == 1:
            train_transform, valid_transform = transform, transform
        elif len(transform) == 2:
            train_transform, valid_transform = transform
        else:
            raise ValueError()

    if data is None:
        DATASET = getattr(torchvision.datasets, dataname)
        if datadir is None:
            datadir = DATA_DIR
        if dataname == "EMNIST":
            split = "letters"
            train = DATASET(
                datadir,
                split=split,
                train=True,
                download=download,
                transform=train_transform,
            )
            test = DATASET(
                datadir,
                split=split,
                train=False,
                download=download,
                transform=valid_transform,
            )
            ## EMNIST seems to have a bug - classes are wrong
            _merged_classes = set(
                [
                    "C",
                    "I",
                    "J",
                    "K",
                    "L",
                    "M",
                    "O",
                    "P",
                    "S",
                    "U",
                    "V",
                    "W",
                    "X",
                    "Y",
                    "Z",
                ]
            )
            _all_classes = set(list(string.digits + string.ascii_letters))
            classes_split_dict = {
                "byclass": list(_all_classes),
                "bymerge": sorted(list(_all_classes - _merged_classes)),
                "balanced": sorted(list(_all_classes - _merged_classes)),
                "letters": list(string.ascii_lowercase),
                "digits": list(string.digits),
                "mnist": list(string.digits),
            }
            train.classes = classes_split_dict[split]
            if split == "letters":
                ## The letters fold (and only that fold!!!) is 1-indexed
                train.targets -= 1
                test.targets -= 1
        elif dataname == "STL10":
            train = DATASET(
                datadir, split="train", download=download, transform=train_transform
            )
            test = DATASET(
                datadir, split="test", download=download, transform=valid_transform
            )
            train.classes = [
                "airplane",
                "bird",
                "car",
                "cat",
                "deer",
                "dog",
                "horse",
                "monkey",
                "ship",
                "truck",
            ]
            test.classes = train.classes
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(test.labels)
        elif dataname == "SVHN":
            train = DATASET(
                datadir, split="train", download=download, transform=train_transform
            )
            test = DATASET(
                datadir, split="test", download=download, transform=valid_transform
            )
            ## In torchvision, SVHN 0s have label 0, not 10
            train.classes = test.classes = [str(i) for i in range(10)]
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(train.labels)
        elif dataname == "LSUN":
            pdb.set_trace()
            train = DATASET(
                datadir, classes="train", download=download, transform=train_transform
            )
        else:
            train = DATASET(
                datadir, train=True, download=download, transform=train_transform
            )
            test = DATASET(
                datadir, train=False, download=download, transform=valid_transform
            )
            # print("HEHE DATASET")
    else:
        train, test = data

    ###### VALIDATION IS 0 SO NOT WORRY NOW ######
    ### Data splitting
    fold_idxs = {}
    if splits is None and valid_size == 0:
        ## Only train
        if stratified_manual:
            ## If stratified, we'll interleave the per-class shuffled indices
            idxs_class = [
                np.random.permutation(np.where(train.targets == c)).T
                for c in np.unique(train.targets)
            ]
            idxs = interleave(*idxs_class).squeeze().astype(int)
            fold_idxs["train"] = idxs
        else:
            fold_idxs["train"] = np.arange(len(train))


    elif splits is None and valid_size > 0:
        ## Train/Valid
        valid_prop = valid_size / len(train)
        train_idx, valid_idx = random_index_split(
            len(train), 1 - valid_prop, (maxsize, valid_size)
        )  # No maxsize for validation
        fold_idxs["train"] = train_idx
        fold_idxs["valid"] = valid_idx
    elif splits is not None:
        ## Custom splits - must be integer.
        if type(splits) is dict:
            snames, slens = zip(*splits.items())
        elif type(splits) in [list, np.ndarray]:
            snames = ["split_{}".format(i) for i in range(len(splits))]
            slens = splits
        slens = np.array(slens)
        if any(slens < 0):  # Split expressed as -1, i.e., 'leftover'
            assert sum(slens < 0) == 1, "Can only deal with one split being -1"
            idx_neg = np.where(slens == -1)[0][0]
            slens[idx_neg] = len(train) - np.array([x for x in slens if x > 0]).sum()
        elif slens.sum() > len(train):
            logging.warning("Not enough samples to satify splits..cropping train...")
            if "train" in snames:
                slens[snames.index("train")] = (
                    len(train) - slens[np.array(snames) != "train"].sum()
                )

        idxs = np.arange(len(train))
        if not stratified:
            np.random.shuffle(idxs)
        else:
            ## If stratified, we'll interleave the per-class shuffled indices
            idxs_class = [
                np.random.permutation(np.where(train.targets == c)).T
                for c in np.unique(train.targets)
            ]
            idxs = interleave(*idxs_class).squeeze().astype(int)

        slens = np.array(slens).cumsum()  # Need to make cumulative for np.split
        split_idxs = [
            np.sort(s) for s in np.split(idxs, slens)[:-1]
        ]  # The last one are leftovers
        assert len(split_idxs) == len(splits)
        fold_idxs = {snames[i]: v for i, v in enumerate(split_idxs)}

    # fold_idxs['train'] = np.arange(len(train)) start -> stop by step
    # if we limit the size of the datasets this will random sample data within
    # the size constraints.
    for k, idxs in fold_idxs.items():
        if maxsize and maxsize < len(idxs):
            if stratified_manual:
                fold_idxs[k] = idxs[:maxsize] # indexes inside each class are permuted
            else:
                fold_idxs[k] = np.sort(np.random.choice(idxs, maxsize, replace=False))

    sampler_class = SubsetRandomSampler if shuffle else SubsetSampler
    fold_samplers = {k: sampler_class(idxs) for k, idxs in fold_idxs.items()}

    k = 0
    total_perturb = int(perturb_per * len(fold_idxs["train"]))
    perturb_inds = np.random.choice(
        sorted(fold_idxs["train"]), size=total_perturb, replace=False
    )
    new_ds_imgs = []
    new_ds_labs = []
    for i in range(len(train)):
        if i in perturb_inds:
            k += 1
            new_ds_imgs.append(
                (train[i][0] + (0.1**0.7) * torch.randn(3, 32, 32)).permute(1, 2, 0)
            )
        else:
            new_ds_imgs.append(train[i][0].permute(1, 2, 0))
        
        new_ds_labs.append(torch.tensor(train[i][1]).reshape(1))
    
    assert k == total_perturb

    new_ds_imgs = torch.stack(new_ds_imgs, dim=0)
    new_ds_labs = torch.cat(new_ds_labs)
    new_ds_imgs = new_ds_imgs.numpy()
    new_ds_labs = new_ds_labs.numpy()

    new_ds = (new_ds_imgs, new_ds_labs)

    new_train = CustomTensorDataset2(new_ds, transform=train_transform)
    train = new_train

    if type(train.targets) is np.ndarray:
        train.targets = train.targets.tolist()

    if type(train.targets) is list:
        train.targets = torch.LongTensor(train.targets)
        test.targets = torch.LongTensor(test.targets)

    if not hasattr(train, "classes") or not train.classes:
        train.classes = sorted(torch.unique(train.targets).tolist())
        test.classes = sorted(torch.unique(test.targets).tolist())

    ### Create DataLoaders
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers)

    fold_loaders = {
        k: dataloader.DataLoader(train, sampler=sampler, **dataloader_args)
        for k, sampler in fold_samplers.items()
    }

    if maxsize_test and maxsize_test < len(test):
        test_idxs = np.sort(np.random.choice(len(test), maxsize_test, replace=False))
        sampler_test = SubsetSampler(test_idxs)  # For test don't want Random
        dataloader_args["sampler"] = sampler_test
    #         print("MAX TEST: ", maxsize_test)
    else:
        dataloader_args["shuffle"] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    fold_loaders["test"] = test_loader

    fnames, flens = zip(*[[k, len(v)] for k, v in fold_idxs.items()])
    fnames = "/".join(list(fnames) + ["test"])
    flens = "/".join(map(str, list(flens) + [len(test)]))

    if hasattr(train, "data"):
        logger.info("Input Dim: {}".format(train.data.shape[1:]))
    logger.info(
        "Classes: {} (effective: {})".format(
            len(train.classes), len(torch.unique(train.targets))
        )
    )
    print(f"Fold Sizes: {flens} ({fnames})")

    return fold_loaders, {"train": train, "test": test}, perturb_inds


def load_torchvision_data_nonstat_feature(
    dataname,
    valid_size=0.1,
    splits=None,
    shuffle=True,
    stratified=False,
    random_seed=None,
    batch_size=64,
    resize=None,
    to3channels=False,
    maxsize=None,
    maxsize_test=None,
    num_workers=0,
    transform=None,
    data=None,
    datadir=None,
    download=True,
    perturb_per=0,
):
    """Load torchvision datasets.

    We return train and test for plots and post-training experiments
    """

    assert perturb_per > 0.0

    if shuffle == True and random_seed:
        np.random.seed(random_seed)
    elif random_seed:
        np.random.seed(random_seed)
    if transform is None:
        if dataname in DATASET_NORMALIZATION.keys():
            transform_dataname = dataname
        else:
            transform_dataname = "ImageNet"

        transform_list = []

        if dataname in ["MNIST", "USPS"] and to3channels:
            transform_list.append(torchvision.transforms.Grayscale(3))

        transform_list.append(torchvision.transforms.ToTensor())
        # transform_list.append(
        #     torchvision.transforms.Normalize(*DATASET_NORMALIZATION[transform_dataname])
        # )

        if resize:
            if not dataname in DATASET_SIZES or DATASET_SIZES[dataname][0] != resize:
                ## Avoid adding an "identity" resizing
                transform_list.insert(0, transforms.Resize((resize, resize)))

        transform = transforms.Compose(transform_list)
        logger.info(transform)
        train_transform, valid_transform = transform, transform
    elif data is None:
        if len(transform) == 1:
            train_transform, valid_transform = transform, transform
        elif len(transform) == 3:
            train_transform_selection, train_transform, valid_transform = transform
        else:
            raise ValueError()

    if data is None:
        DATASET = getattr(torchvision.datasets, dataname)
        if datadir is None:
            datadir = DATA_DIR
        if dataname == "EMNIST":
            split = "letters"
            train = DATASET(
                datadir,
                split=split,
                train=True,
                download=download,
                transform=train_transform,
            )
            test = DATASET(
                datadir,
                split=split,
                train=False,
                download=download,
                transform=valid_transform,
            )
            ## EMNIST seems to have a bug - classes are wrong
            _merged_classes = set(
                [
                    "C",
                    "I",
                    "J",
                    "K",
                    "L",
                    "M",
                    "O",
                    "P",
                    "S",
                    "U",
                    "V",
                    "W",
                    "X",
                    "Y",
                    "Z",
                ]
            )
            _all_classes = set(list(string.digits + string.ascii_letters))
            classes_split_dict = {
                "byclass": list(_all_classes),
                "bymerge": sorted(list(_all_classes - _merged_classes)),
                "balanced": sorted(list(_all_classes - _merged_classes)),
                "letters": list(string.ascii_lowercase),
                "digits": list(string.digits),
                "mnist": list(string.digits),
            }
            train.classes = classes_split_dict[split]
            if split == "letters":
                ## The letters fold (and only that fold!!!) is 1-indexed
                train.targets -= 1
                test.targets -= 1
        elif dataname == "SVHN":
            train = DATASET(
                datadir, split="train", download=download, transform=train_transform
            )
            test = DATASET(
                datadir, split="test", download=download, transform=valid_transform
            )
            ## In torchvision, SVHN 0s have label 0, not 10
            train.classes = test.classes = [str(i) for i in range(10)]
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(train.labels)
        else:
            train_selection = DATASET(
                datadir, train=True, download=download, transform=train_transform_selection,
            )
            train = DATASET(
                datadir, train=True, download=download, transform=train_transform,
            )
            test = DATASET(
                datadir, train=False, download=download, transform=valid_transform,
            )
    else:
        train, test = data


    if type(train_selection.targets) is list:
        train.targets = torch.LongTensor(train.targets)
        train_selection.targets = torch.LongTensor(train_selection.targets)
        test.targets = torch.LongTensor(test.targets)

    if not hasattr(train, "classes") or not train.classes:
        train.classes = sorted(torch.unique(train.targets).tolist())
        train_selection.classes = sorted(torch.unique(train_selection.targets).tolist())
        test.classes = sorted(torch.unique(train.targets).tolist())

    ### Data splitting
    fold_idxs = {}
    if splits is None and valid_size == 0:
        ## Only train
        fold_idxs["train"] = np.arange(len(train))
    elif splits is None and valid_size > 0:
        ## Train/Valid
        train_idx, valid_idx = random_index_split(
            len(train), 1 - valid_size, (maxsize, valid_size)
        )  # No maxsize for validation
        fold_idxs["train"] = train_idx
        fold_idxs["valid"] = valid_idx
    elif splits == 5 and valid_size > 0:
        # split D_train into 5 tasks
        # sample indicies for train and train_selection
        indices = np.arange(len(train))
        np.random.shuffle(indices) # inplace
        split = len(train) - valid_size
        train_idx, valid_idx = np.array(indices[:split]), np.array(indices[split:])
        fold_idxs["valid"] = valid_idx
        fold_idxs["train"] = train_idx
        for i in range(splits):
            end = int((i + 1) * (split / 5))
            fold_idxs[f"train_sel_{i}"] = train_idx[:end]
            fold_idxs[f"train_{i}"] = train_idx[:end]
    elif splits == 5 and valid_size == 0:
        # split D_train into 5 tasks
        # sample indicies for train and train_selection
        indices = np.arange(len(train))
        np.random.shuffle(indices) # inplace
        fold_idxs["train"] = indices
        for i in range(splits):
            end = int((i + 1) * (len(train) / 5))
            fold_idxs[f"train_sel_{i}"] = indices[:end]
            fold_idxs[f"train_{i}"] = indices[:end]
    elif splits is not None:
        ## Custom splits - must be integer.
        if type(splits) is dict:
            snames, slens = zip(*splits.items())
        elif type(splits) in [list, np.ndarray]:
            snames = ["split_{}".format(i) for i in range(len(splits))]
            slens = splits
        slens = np.array(slens)
        if any(slens < 0):  # Split expressed as -1, i.e., 'leftover'
            assert sum(slens < 0) == 1, "Can only deal with one split being -1"
            idx_neg = np.where(slens == -1)[0][0]
            slens[idx_neg] = len(train) - np.array([x for x in slens if x > 0]).sum()
        elif slens.sum() > len(train):
            logging.warning("Not enough samples to satify splits..cropping train...")
            if "train" in snames:
                slens[snames.index("train")] = (
                    len(train) - slens[np.array(snames) != "train"].sum()
                )

        idxs = np.arange(len(train))
        if not stratified:
            np.random.shuffle(idxs)
        else:
            ## If stratified, we'll interleave the per-class shuffled indices
            idxs_class = [
                np.random.permutation(np.where(train.targets == c)).T
                for c in np.unique(train.targets)
            ]
            idxs = interleave(*idxs_class).squeeze().astype(int)

        slens = np.array(slens).cumsum()  # Need to make cumulative for np.split
        split_idxs = [
            np.sort(s) for s in np.split(idxs, slens)[:-1]
        ]  # The last one are leftovers
        assert len(split_idxs) == len(splits)
        fold_idxs = {snames[i]: v for i, v in enumerate(split_idxs)}

    # if maxsize has been defined, clip the size of the training datasets.
    for k, idxs in fold_idxs.items():
        if maxsize and maxsize < len(idxs):
            # for full train we keep the same maxsize, for the folds with devide by the number fo tasks
            if k in ["train", "valid"]:
                fold_idxs[k] = np.sort(idxs[:maxsize])
    
    for i in range(splits):
        k = f"train_sel_{i}"
        idxs = fold_idxs[k]
        if maxsize and maxsize < len(idxs):
            end = int((i + 1) * (maxsize / 5))
            fold_idxs[k] = idxs[:end]
        k = f"train_{i}"
        idxs = fold_idxs[k]
        if maxsize and maxsize < len(idxs):
            end = int((i + 1) * (maxsize / 5))
            fold_idxs[k] = idxs[:end]
    
    # Define subsampler for train and validation splits
    sampler_class = SubsetRandomSampler if shuffle else SubsetSampler
    fold_samplers = {k: sampler_class(idxs) for k, idxs in fold_idxs.items()}

    total_perturb = int(perturb_per * len(fold_idxs["train"]))

    perturb_inds = []

    # add noise to train and train_selection images
    if dataname == "CIFAR10":
        new_ds_imgs, new_ds_sel_imgs = [], []
        new_ds_labs, new_ds_sel_labs = [], []
        for i in range(len(train)):
            rand_num = np.random.rand()
            if rand_num < perturb_per:
                perturb_inds.append(i)
                # train
                new_ds_imgs.append(
                    (train[i][0] + (0.1**0.7) * torch.randn(3, 32, 32)).permute(
                        1, 2, 0
                    )
                )
                new_ds_labs.append(torch.tensor(train[i][1]).reshape(1))

                # train selection
                new_ds_sel_imgs.append(
                    (train_selection[i][0] + (0.1**0.7) * torch.randn(3, 32, 32)).permute(
                        1, 2, 0
                    )
                )
                new_ds_sel_labs.append(torch.tensor(train_selection[i][1]).reshape(1))
            else:
                new_ds_imgs.append(train[i][0].permute(1, 2, 0))
                new_ds_labs.append(torch.tensor(train[i][1]).reshape(1))
                new_ds_sel_imgs.append(train_selection[i][0].permute(1, 2, 0))
                new_ds_sel_labs.append(torch.tensor(train_selection[i][1]).reshape(1))
    else:
        raise ValueError
    
    new_ds_imgs = torch.stack(new_ds_imgs, dim=0).numpy()
    new_ds_labs = torch.cat(new_ds_labs).numpy()
    new_ds = (new_ds_imgs, new_ds_labs)
    train = CustomTensorDataset2( 
        new_ds, transform=train_transform)

    if not torch.is_tensor(train.targets):
        train.targets = torch.LongTensor(train.targets)
    
    if not torch.is_tensor(test.targets):   
        test.targets = torch.LongTensor(test.targets)
    
    if not hasattr(train, "classes") or not train.classes:
        train.classes = sorted(torch.unique(train.targets).tolist())
        test.classes = sorted(torch.unique(test.targets).tolist())

    new_ds_sel_imgs = torch.stack(new_ds_sel_imgs, dim=0).numpy()
    new_ds_sel_labs = torch.cat(new_ds_sel_labs).numpy()
    new_ds_sel = (new_ds_sel_imgs, new_ds_sel_labs)
    train_selection = CustomTensorDataset2(
        new_ds_sel, transform=train_transform_selection)

    if not torch.is_tensor(train_selection.targets):   
        train_selection.targets = torch.LongTensor(train_selection.targets)

    if not hasattr(train_selection, "classes") or not train_selection.classes:
        train_selection.classes = sorted(torch.unique(train_selection.targets).tolist())

    perturb_inds = np.array(sorted(perturb_inds))

    # Create DataLoaders wityh samplers
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers)

    fold_loaders = {}
    for k, sampler in fold_samplers.items():
        if "sel" in k:
            fold_loaders[k] = dataloader.DataLoader(train_selection, sampler=sampler, **dataloader_args)
        else:
            fold_loaders[k] = dataloader.DataLoader(train, sampler=sampler, **dataloader_args)

    # if we have defined a maxsize for the eval sets
    if maxsize_test and maxsize_test < len(test):
        test_idxs = np.sort(np.random.choice(len(test), maxsize_test, replace=False))
        sampler_test = SubsetSampler(test_idxs)  # For test don't want Random
        dataloader_args["sampler"] = sampler_test
    else:
        dataloader_args["shuffle"] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    fold_loaders["test"] = test_loader

    if hasattr(train, "data"):
        logger.info("Input Dim: {}".format(train.data.shape[1:]))
    logger.info(
        "Classes: {} (effective: {})".format(
            len(train.classes), len(torch.unique(train.targets))
        )
    )

    # fold loaders has the data loaders with subsets for maxsize
    # also has the test loaders with the subsets for maxsize_test
    # train and test are the full torch dataset.Datasets
    # shuffle inds are the ground truth indices which have need perturbed with noisy labels
    return fold_loaders, {"train": train, "test": test}, perturb_inds

def load_torchvision_data_trojan_sq(
    dataname,
    valid_size=0.0,
    splits=None,
    shuffle=True,
    stratified=False,
    random_seed=None,
    batch_size=64,
    resize=None,
    to3channels=False,
    maxsize=None,
    maxsize_test=None,
    num_workers=0,
    transform=None,
    data=None,
    datadir=None,
    download=True,
    shuffle_per=0,
    perturb_per=0, # percent of inputs which have the trojan square attack
    trojan_class='airplane', # inputs
):
    """Load torchvision datasets.

    "Trojan Square attack, a popular attack algorithm (Liu et al., 2017), which injects training points
    that contain a backdoor trigger and are relabeled as a target class. The evaluation of other types
    of backdoor attacks can be found in Appendix B. To simulate this attack, we select the target
    attack class Airplane and poison 2500 (5%) samples of the total CIFAR-10 training set (50k) 
    with a square trigger.
    """
    assert perturb_per > 0.0

    if shuffle == True and random_seed:
        np.random.seed(random_seed)
    elif random_seed:
        np.random.seed(random_seed)
    if transform is None:
        if dataname in DATASET_NORMALIZATION.keys():
            transform_dataname = dataname
        else:
            transform_dataname = "ImageNet"

        transform_list = []

        if dataname in ["MNIST", "USPS"] and to3channels:
            transform_list.append(torchvision.transforms.Grayscale(3))

        transform_list.append(torchvision.transforms.ToTensor())
        # transform_list.append(
        #    torchvision.transforms.Normalize(*DATASET_NORMALIZATION[transform_dataname])
        # )

        if resize:
            if not dataname in DATASET_SIZES or DATASET_SIZES[dataname][0] != resize:
                ## Avoid adding an "identity" resizing
                transform_list.insert(0, transforms.Resize((resize, resize)))

        transform = transforms.Compose(transform_list)
        logger.info(transform)
        train_transform, valid_transform = transform, transform
    elif data is None:
        if len(transform) == 1:
            train_transform, valid_transform = transform, transform
        elif len(transform) == 2:
            train_transform, valid_transform = transform
        else:
            raise ValueError()

    if data is None:
        DATASET = getattr(torchvision.datasets, dataname)
        if datadir is None:
            datadir = DATA_DIR
        if dataname == "EMNIST":
            split = "letters"
            train = DATASET(
                datadir,
                split=split,
                train=True,
                download=download,
                transform=train_transform,
            )
            test = DATASET(
                datadir,
                split=split,
                train=False,
                download=download,
                transform=valid_transform,
            )
            ## EMNIST seems to have a bug - classes are wrong
            _merged_classes = set(
                [
                    "C",
                    "I",
                    "J",
                    "K",
                    "L",
                    "M",
                    "O",
                    "P",
                    "S",
                    "U",
                    "V",
                    "W",
                    "X",
                    "Y",
                    "Z",
                ]
            )
            _all_classes = set(list(string.digits + string.ascii_letters))
            classes_split_dict = {
                "byclass": list(_all_classes),
                "bymerge": sorted(list(_all_classes - _merged_classes)),
                "balanced": sorted(list(_all_classes - _merged_classes)),
                "letters": list(string.ascii_lowercase),
                "digits": list(string.digits),
                "mnist": list(string.digits),
            }
            train.classes = classes_split_dict[split]
            if split == "letters":
                ## The letters fold (and only that fold!!!) is 1-indexed
                train.targets -= 1
                test.targets -= 1
        elif dataname == "STL10":
            train = DATASET(
                datadir, split="train", download=download, transform=train_transform
            )
            test = DATASET(
                datadir, split="test", download=download, transform=valid_transform
            )
            train.classes = [
                "airplane",
                "bird",
                "car",
                "cat",
                "deer",
                "dog",
                "horse",
                "monkey",
                "ship",
                "truck",
            ]
            test.classes = train.classes
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(test.labels)
        elif dataname == "SVHN":
            train = DATASET(
                datadir, split="train", download=download, transform=train_transform
            )
            test = DATASET(
                datadir, split="test", download=download, transform=valid_transform
            )
            ## In torchvision, SVHN 0s have label 0, not 10
            train.classes = test.classes = [str(i) for i in range(10)]
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(train.labels)
        elif dataname == "LSUN":
            pdb.set_trace()
            train = DATASET(
                datadir, classes="train", download=download, transform=train_transform
            )
        else:
            train = DATASET(
                datadir, train=True, download=download, transform=train_transform
            )
            test = DATASET(
                datadir, train=False, download=download, transform=valid_transform
            )
            # print("HEHE DATASET")
    else:
        train, test = data

    ###### VALIDATION IS 0 SO NOT WORRY NOW ######
    ### Data splitting
    fold_idxs = {}
    if splits is None and valid_size == 0:
        ## Only train
        fold_idxs["train"] = np.arange(len(train))

    #         print("FOLD IDSSS: ", fold_idxs['train'])
    elif splits is None and valid_size > 0:
        ## Train/Valid
        valid_prop = valid_size / len(train)
        train_idx, valid_idx = random_index_split(
            len(train), 1 - valid_prop, (maxsize, valid_size)
        )  # No maxsize for validation
        fold_idxs["train"] = train_idx
        fold_idxs["valid"] = valid_idx
    elif splits is not None:
        ## Custom splits - must be integer.
        if type(splits) is dict:
            snames, slens = zip(*splits.items())
        elif type(splits) in [list, np.ndarray]:
            snames = ["split_{}".format(i) for i in range(len(splits))]
            slens = splits
        slens = np.array(slens)
        if any(slens < 0):  # Split expressed as -1, i.e., 'leftover'
            assert sum(slens < 0) == 1, "Can only deal with one split being -1"
            idx_neg = np.where(slens == -1)[0][0]
            slens[idx_neg] = len(train) - np.array([x for x in slens if x > 0]).sum()
        elif slens.sum() > len(train):
            logging.warning("Not enough samples to satify splits..cropping train...")
            if "train" in snames:
                slens[snames.index("train")] = (
                    len(train) - slens[np.array(snames) != "train"].sum()
                )

        idxs = np.arange(len(train))
        if not stratified:
            np.random.shuffle(idxs)
        else:
            ## If stratified, we'll interleave the per-class shuffled indices
            idxs_class = [
                np.random.permutation(np.where(train.targets == c)).T
                for c in np.unique(train.targets)
            ]
            idxs = interleave(*idxs_class).squeeze().astype(int)

        slens = np.array(slens).cumsum()  # Need to make cumulative for np.split
        split_idxs = [
            np.sort(s) for s in np.split(idxs, slens)[:-1]
        ]  # The last one are leftovers
        assert len(split_idxs) == len(splits)
        fold_idxs = {snames[i]: v for i, v in enumerate(split_idxs)}

    # fold_idxs['train'] = np.arange(len(train)) start -> stop by step
    # if we limit the size of the datasets this will random sample data within
    # the size constraints.
    for k, idxs in fold_idxs.items():
        if maxsize and maxsize < len(idxs):
            fold_idxs[k] = np.sort(np.random.choice(idxs, maxsize, replace=False))

    sampler_class = SubsetRandomSampler if shuffle else SubsetSampler
    fold_samplers = {k: sampler_class(idxs) for k, idxs in fold_idxs.items()}

    k = 0
    total_perturb = int(perturb_per * len(fold_idxs["train"]))
    perturb_inds = np.random.choice(
        sorted(fold_idxs["train"]), size=total_perturb, replace=False
    )
    new_ds_imgs = []
    new_ds_labs = []
    for i in range(len(train)):
        if i in perturb_inds:
            k += 1
            mask = torch.zeros((3, 32, 32))
            mask[:, 18:28, 18:28] = 1
            # relabel to airplane
            new_ds_labs.append(
                torch.tensor(train.classes.index(trojan_class)).reshape(1)
            )
            new_ds_imgs.append(
                (train[i][0] + mask * (0.1**0.7) * torch.randn(3, 32, 32)).permute(1, 2, 0)
            )
        else:
            new_ds_imgs.append(train[i][0].permute(1, 2, 0))
            new_ds_labs.append(torch.tensor(train[i][1]).reshape(1))

    assert k == total_perturb

    new_ds_imgs = torch.stack(new_ds_imgs, dim=0)
    new_ds_labs = torch.cat(new_ds_labs)
    new_ds_imgs = new_ds_imgs.numpy()
    new_ds_labs = new_ds_labs.numpy()
    new_ds = (new_ds_imgs, new_ds_labs)

    new_train = CustomTensorDataset2(new_ds, transform=train_transform)
    train = new_train

    if type(train.targets) is np.ndarray:
        train.targets = train.targets.tolist()

    if type(train.targets) is list:
        train.targets = torch.LongTensor(train.targets)
        test.targets = torch.LongTensor(test.targets)

    if not hasattr(train, "classes") or not train.classes:
        train.classes = sorted(torch.unique(train.targets).tolist())
        test.classes = sorted(torch.unique(test.targets).tolist())


    ### Create DataLoaders
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers)

    fold_loaders = {
        k: dataloader.DataLoader(train, sampler=sampler, **dataloader_args)
        for k, sampler in fold_samplers.items()
    }

    if maxsize_test and maxsize_test < len(test):
        test_idxs = np.sort(np.random.choice(len(test), maxsize_test, replace=False))
        sampler_test = SubsetSampler(test_idxs)  # For test don't want Random
        dataloader_args["sampler"] = sampler_test
    #         print("MAX TEST: ", maxsize_test)
    else:
        dataloader_args["shuffle"] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    fold_loaders["test"] = test_loader

    fnames, flens = zip(*[[k, len(v)] for k, v in fold_idxs.items()])
    fnames = "/".join(list(fnames) + ["test"])
    flens = "/".join(map(str, list(flens) + [len(test)]))

    if hasattr(train, "data"):
        print("Input Dim: {}".format(train.data.shape[1:]))
    print(
        "Classes: {} (effective: {})".format(
            len(train.classes), len(torch.unique(train.targets))
        )
    )
    print(f"Fold Sizes: {flens} ({fnames})")

    return fold_loaders, {"train": train, "test": test}, perturb_inds


def forward_step(
    model: nn.Sequential, 
    img: torch.Tensor, 
    lr: float, 
    target_logits: torch.Tensor,
) -> torch.Tensor:
    """helper function performing the forward step"""
    base_img = img.detach().clone().requires_grad_(True)

    model.zero_grad()
    logits = model(base_img)
    mse = nn.MSELoss()
    loss = mse(logits, target_logits)
    #loss = torch.norm(logits - target_logits)
    loss.backward()

    img_grad = base_img.grad.detach().clone()
    perturbed_img = base_img - lr * img_grad
    return perturbed_img


def backward_step(
    img: torch.Tensor, 
    base_img: torch.Tensor, 
    lr: float, 
    beta: float,
) -> torch.Tensor:
    """helper function to perform the backward step"""
    perturbed_img = (img + lr * beta * base_img) / (1 + beta * lr)
    perturbed_img = torch.clamp(perturbed_img, 0, 1) # to avoid clipping
    return perturbed_img

def load_torchvision_data_poison_frogs(
    dataname,
    model,
    device,
    valid_size=0.0,
    splits=None,
    shuffle=True,
    stratified=False,
    random_seed=None,
    batch_size=64,
    resize=None,
    to3channels=False,
    maxsize=None,
    maxsize_test=None,
    num_workers=0,
    transform=None,
    data=None,
    datadir=None,
    download=True,
    perturb_per=0, # percent of inputs the feature collision attack
    target_class='cat', # test set image which is used the blend into the base class
    base_class='frog',
    poison_frogs_feat_repr=False,
    cache_dir=None,
    cache_tag='',
    remake_data=False,
    verbose=True,
):
    """Load torchvision datasets.

    "We consider a popular attack termed “feature-collision” attack
    (Shafahi et al., 2018), where we select a target sample
    from the Cat class test set and blend the selected image
    with the chosen target class training samples, Frog in our case. 
    In this attack, we do not modify labels and blend the Cat image only 
    into 50 (0.1%) samples of Frog, which makes this attack especially hard
    to detect. During inference time, we expect the attacked model to 
    consistently classify the chosen Cat as a Frog.".

    Code used from: https://github.com/LostOxygen/poison_froggo/blob/main/poison_frog/dataset.py
    """
    
    assert dataname == "CIFAR10", "Only CIFAR10 is supported for the feature collision attack"
    assert 0 < perturb_per <= 0.1, "Perturb % must <= 0.1 for CIFAR10 with 10 classes, above 0.1 and there are not enough labels for a particular class"

    net = copy.deepcopy(model)
    if poison_frogs_feat_repr:
        net.linear = nn.Sequential(
            nn.Identity() # remove the fully connected layer to obtain the feature space repr.
        )
    net = net.to(device)

    if shuffle == True and random_seed:
        np.random.seed(random_seed)
    elif random_seed:
        np.random.seed(random_seed)
    if transform is None:
        if dataname in DATASET_NORMALIZATION.keys():
            transform_dataname = dataname
        else:
            transform_dataname = "ImageNet"

        transform_list = []

        if dataname in ["MNIST", "USPS"] and to3channels:
            transform_list.append(torchvision.transforms.Grayscale(3))

        transform_list.append(torchvision.transforms.ToTensor())
        # transform_list.append(
        #    torchvision.transforms.Normalize(*DATASET_NORMALIZATION[transform_dataname])
        # )

        if resize:
            if not dataname in DATASET_SIZES or DATASET_SIZES[dataname][0] != resize:
                ## Avoid adding an "identity" resizing
                transform_list.insert(0, transforms.Resize((resize, resize)))

        transform = transforms.Compose(transform_list)
        logger.info(transform)
        train_transform, valid_transform = transform, transform
    elif data is None:
        if len(transform) == 1:
            train_transform, valid_transform = transform, transform
        elif len(transform) == 2:
            train_transform, valid_transform = transform
        else:
            raise ValueError()

    if data is None:
        DATASET = getattr(torchvision.datasets, dataname)
        if datadir is None:
            datadir = DATA_DIR
        if dataname == "EMNIST":
            split = "letters"
            train = DATASET(
                datadir,
                split=split,
                train=True,
                download=download,
                transform=train_transform,
            )
            test = DATASET(
                datadir,
                split=split,
                train=False,
                download=download,
                transform=valid_transform,
            )
            ## EMNIST seems to have a bug - classes are wrong
            _merged_classes = set(
                [
                    "C",
                    "I",
                    "J",
                    "K",
                    "L",
                    "M",
                    "O",
                    "P",
                    "S",
                    "U",
                    "V",
                    "W",
                    "X",
                    "Y",
                    "Z",
                ]
            )
            _all_classes = set(list(string.digits + string.ascii_letters))
            classes_split_dict = {
                "byclass": list(_all_classes),
                "bymerge": sorted(list(_all_classes - _merged_classes)),
                "balanced": sorted(list(_all_classes - _merged_classes)),
                "letters": list(string.ascii_lowercase),
                "digits": list(string.digits),
                "mnist": list(string.digits),
            }
            train.classes = classes_split_dict[split]
            if split == "letters":
                ## The letters fold (and only that fold!!!) is 1-indexed
                train.targets -= 1
                test.targets -= 1
        elif dataname == "STL10":
            train = DATASET(
                datadir, split="train", download=download, transform=train_transform
            )
            test = DATASET(
                datadir, split="test", download=download, transform=valid_transform
            )
            train.classes = [
                "airplane",
                "bird",
                "car",
                "cat",
                "deer",
                "dog",
                "horse",
                "monkey",
                "ship",
                "truck",
            ]
            test.classes = train.classes
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(test.labels)
        elif dataname == "SVHN":
            train = DATASET(
                datadir, split="train", download=download, transform=train_transform
            )
            test = DATASET(
                datadir, split="test", download=download, transform=valid_transform
            )
            ## In torchvision, SVHN 0s have label 0, not 10
            train.classes = test.classes = [str(i) for i in range(10)]
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(train.labels)
        elif dataname == "LSUN":
            pdb.set_trace()
            train = DATASET(
                datadir, classes="train", download=download, transform=train_transform
            )
        else:
            train = DATASET(
                datadir, train=True, download=download, transform=train_transform
            )
            test = DATASET(
                datadir, train=False, download=download, transform=valid_transform
            )
    else:
        train, test = data

    ### Data splitting
    fold_idxs = {}
    if splits is None and valid_size == 0:
        ## Only train
        fold_idxs["train"] = np.arange(len(train))

    elif splits is None and valid_size > 0:
        ## Train/Valid
        valid_prop = valid_size / len(train)
        train_idx, valid_idx = random_index_split(
            len(train), 1 - valid_prop, (maxsize, valid_size)
        )  # No maxsize for validation
        fold_idxs["train"] = train_idx
        fold_idxs["valid"] = valid_idx
    elif splits is not None:
        ## Custom splits - must be integer.
        if type(splits) is dict:
            snames, slens = zip(*splits.items())
        elif type(splits) in [list, np.ndarray]:
            snames = ["split_{}".format(i) for i in range(len(splits))]
            slens = splits
        slens = np.array(slens)
        if any(slens < 0):  # Split expressed as -1, i.e., 'leftover'
            assert sum(slens < 0) == 1, "Can only deal with one split being -1"
            idx_neg = np.where(slens == -1)[0][0]
            slens[idx_neg] = len(train) - np.array([x for x in slens if x > 0]).sum()
        elif slens.sum() > len(train):
            logging.warning("Not enough samples to satify splits..cropping train...")
            if "train" in snames:
                slens[snames.index("train")] = (
                    len(train) - slens[np.array(snames) != "train"].sum()
                )

        idxs = np.arange(len(train))
        if not stratified:
            np.random.shuffle(idxs)
        else:
            ## If stratified, we'll interleave the per-class shuffled indices
            idxs_class = [
                np.random.permutation(np.where(train.targets == c)).T
                for c in np.unique(train.targets)
            ]
            idxs = interleave(*idxs_class).squeeze().astype(int)

        slens = np.array(slens).cumsum()  # Need to make cumulative for np.split
        split_idxs = [
            np.sort(s) for s in np.split(idxs, slens)[:-1]
        ]  # The last one are leftovers
        assert len(split_idxs) == len(splits)
        fold_idxs = {snames[i]: v for i, v in enumerate(split_idxs)}

    # fold_idxs['train'] = np.arange(len(train)) start -> stop by step
    # if we limit the size of the datasets this will random sample data within
    # the size constraints.
    for k, idxs in fold_idxs.items():
        if maxsize and maxsize < len(idxs):
            fold_idxs[k] = np.sort(np.random.choice(idxs, maxsize, replace=False))
    
    feat_repr = "feat_repr" if poison_frogs_feat_repr else "out_repr"
    base_name = f"poison_frogs_{dataname}_pp{perturb_per}_tc{target_class}_bc{base_class}_{feat_repr}_{cache_tag}.pickle"
    cache_path = os.path.join(cache_dir, base_name)
    print(f"Cache path: {cache_path}")
    # A | B
    # not True : False | False --> False
    # not True : False | True --> True
    # not False : True | T/F --> True
    if not os.path.exists(cache_path) or remake_data:
        print("Creating Poison Frogs Dataset")
        new_train = DATASET
        new_train.targets = {}
        new_train.classes = {}
        new_train.targets = train.targets # (50000,)
        new_train.classes = train.classes # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        target = test.classes.index(target_class)
        print("target class: ", target)
        print("target name: ", target_class)
        # the target class for the image which should get missclassified
        target_images_list = []
        for i in range(len(test)):
            if test[i][1] == target:
                target_images_list.append(test[i][0])

        random_id = np.random.randint(0, len(target_images_list))
        target_image = target_images_list[random_id]

        # base class
        # the class as which the chosen image should get missclassified
        base = test.classes.index(base_class)
        print("base class: ", base)
        print("base name: ", base_class)

        # calculate the beta
        img_shape = np.squeeze(target_image).shape
        beta = 0.25 * (2048 / float(img_shape[0] * img_shape[1] * img_shape[2]))**2
        print("beta = {}".format(beta))

        k = 0
        perturb_inds = []
        new_ds_imgs = []
        new_ds_labs = []
        total_perturb = int(perturb_per * maxsize)
        attack_iters = 100
        current_perturb_count = 0
        for i in range(len(train)):
            train_label = torch.tensor(train[i][1]).reshape(1)
            if i in fold_idxs["train"]:
                k += 1
                difference = 100
                if train_label == base and current_perturb_count < total_perturb:
                    perturb_inds.append(i)
                    base_image = torch.tensor(train[i][0]).to(device) # 3 x 32 x 32
                    old_image = base_image.clone().detach().requires_grad_(True)
                    if len(old_image.shape) == 3:
                        old_image = old_image.unsqueeze(0)

                    # Initializations
                    num_m = 40
                    last_m_objs = []
                    decay_coef = 0.5 # decay coeffiencet of learning rate
                    stopping_tol = 1e-10 # for the relative change
                    learning_rate = 0.1 #500.0 * 255 # iniital learning rate for optimization
                    rel_change_val = 1e5
                    target_feat = net(target_image.unsqueeze(0).to(device)).detach() # feat x x 10
                    old_feat = net(base_image.unsqueeze(0).to(device)).detach() # 1 x 10
                    old_obj = torch.linalg.norm(old_feat - target_feat) + beta * torch.linalg.norm(old_image - base_image)
                    last_m_objs.append(old_obj)
                    obj_threshold = 2.9
                
                    # perform the attack as described in the paper to optimize
                    # || f(x)-f(t) ||^2 + beta * || x-b ||^2
                    for j in range(attack_iters):
                        if j % 1 == 0:
                            diff = torch.linalg.norm(old_feat - target_feat) #get the diff
                            if verbose:
                                print(f"Poison iter: {j} | diff: {diff} | obj: {old_obj} | lr: {learning_rate} | rel change: {rel_change_val}")
                        # the main forward backward passes
                        new_image = forward_step(
                            model=net,
                            img=old_image, # 1 x 3 x 32 x 32
                            lr=learning_rate,
                            target_logits=target_feat.detach().clone().requires_grad_(False),
                        )
                        new_image = backward_step(
                            img=new_image,
                            base_img=old_image, 
                            lr=learning_rate, 
                            beta=beta,
                        )

                        # check stopping condition:  compute relative change in image between iterations
                        rel_change_val = torch.linalg.norm(new_image-old_image)/torch.linalg.norm(new_image)
                        if (rel_change_val < stopping_tol) or (old_obj <= obj_threshold):
                            #print("! reached the object threshold -> stopping optimization !")
                            break

                        # compute new objective value
                        new_feat = net(new_image.detach().clone().requires_grad_(False))
                        new_obj = torch.linalg.norm(new_feat - target_feat) + beta*torch.linalg.norm(new_image - base_image)

                        #find the mean of the last M iterations
                        avg_of_last_m = sum(last_m_objs) / float(min(num_m, j + 1))
                        # If the objective went up, then learning rate is too big.
                        # Chop it, and throw out the latest iteration
                        if new_obj >= avg_of_last_m and (j % num_m / 2 == 0):
                            learning_rate *= decay_coef
                            new_image = old_image
                        else:
                            old_image = new_image
                            old_obj = new_obj
                            old_feat = new_feat

                        if j < num_m - 1:
                            last_m_objs.append(new_obj)
                        else:
                            # first remove the oldest obj then append the new obj
                            del last_m_objs[0]
                            last_m_objs.append(new_obj)

                        difference = torch.linalg.norm(old_feat - target_feat)

                    if difference < 3.5: # this seems arbitrary! increase to 5?
                        new_ds_imgs.append(old_image.squeeze().detach().permute(1, 2, 0).cpu())
                        new_ds_labs.append(train_label)
                        current_perturb_count += 1
                    else:
                        print(f"difference is too large {difference} for image {i}")

                else:
                    # not class which is being poisoned
                    new_ds_imgs.append(train[i][0].permute(1, 2, 0).detach().cpu())
                    new_ds_labs.append(train_label)
            else:
                # not in fold_idxs["train"]
                new_ds_imgs.append(train[i][0].permute(1, 2, 0).detach().cpu())
                new_ds_labs.append(train_label)

            
            #if i % 5000 == 0:
            print(f"Num images processed {i} / {len(train)}")

        assert k == len(fold_idxs["train"])

        new_ds_imgs = torch.stack(new_ds_imgs, dim=0)
        new_ds_labs = torch.cat(new_ds_labs)
        new_ds_imgs = new_ds_imgs.numpy()
        new_ds_labs = new_ds_labs.numpy()
        new_ds = (new_ds_imgs, new_ds_labs)

        if not os.path.exists(cache_path):
            with open(cache_path, 'wb') as handle:
                pickle.dump(
                    {
                        'new_ds': new_ds,
                        'perturb_inds': perturb_inds,
                    }, 
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

    else:
        print("getting from the cache")
        with open(cache_path, 'rb') as handle:
            tmp = pickle.load(handle)
            new_ds = tmp['new_ds']
            perturb_inds = tmp['perturb_inds']
        
    new_train = CustomTensorDataset2(new_ds, transform=train_transform)
    train = new_train

    if type(train.targets) is np.ndarray:
        train.targets = train.targets.tolist()

    if type(train.targets) is list:
        train.targets = torch.LongTensor(train.targets)
        test.targets = torch.LongTensor(test.targets)

    if not hasattr(train, "classes") or not train.classes:
        train.classes = sorted(torch.unique(train.targets).tolist())
        test.classes = sorted(torch.unique(test.targets).tolist())

    sampler_class = SubsetRandomSampler if shuffle else SubsetSampler
    fold_samplers = {k: sampler_class(idxs) for k, idxs in fold_idxs.items()}

    ### Create DataLoaders
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers)

    fold_loaders = {
        k: dataloader.DataLoader(train, sampler=sampler, **dataloader_args)
        for k, sampler in fold_samplers.items()
    }

    if maxsize_test and maxsize_test < len(test):
        test_idxs = np.sort(np.random.choice(len(test), maxsize_test, replace=False))
        sampler_test = SubsetSampler(test_idxs)  # For test don't want Random
        dataloader_args["sampler"] = sampler_test
    #         print("MAX TEST: ", maxsize_test)
    else:
        dataloader_args["shuffle"] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    fold_loaders["test"] = test_loader

    fnames, flens = zip(*[[k, len(v)] for k, v in fold_idxs.items()])
    fnames = "/".join(list(fnames) + ["test"])
    flens = "/".join(map(str, list(flens) + [len(test)]))

    if hasattr(train, "data"):
        logger.info("Input Dim: {}".format(train.data.shape[1:]))
    logger.info(
        "Classes: {} (effective: {})".format(
            len(train.classes), len(torch.unique(train.targets))
        )
    )
    print(f"Fold Sizes: {flens} ({fnames})")

    return fold_loaders, {"train": train, "test": test}, perturb_inds
