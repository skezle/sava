from typing import Dict, List
import os

from PIL import Image

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
from torchvision.datasets.vision import VisionDataset

import otdd
from otdd.pytorch.datasets import (
    load_torchvision_data_shuffle,
    load_torchvision_data_nonstat_shuffle,
    load_torchvision_data_perturb,
    load_torchvision_data_nonstat_feature,
    load_torchvision_data_trojan_sq,
    load_torchvision_data_poison_frogs,
)
from otdd.pytorch.datasets import SubsetSampler


# Corrupted will return list of indices that were corrupted
# 3 types of corrupted directly provided: backdoor (blend, trojan-sq, trojan-wm), noisy features, noisy labels
def load_data_corrupted(
    model,
    device,
    corrupt_type="shuffle", # {'shuffle', 'feature', 'trojan_sq'}
    dataname=None,
    transform=None,
    valid_size=0,
    random_seed=2021,
    resize=None,
    stratified=True,
    stratified_manual=False,
    shuffle=False,
    training_size=None,
    test_size=None,
    corrupt_por=0,
    batch_size=64,
    poison_frogs_feat_repr=False,
    remake_data=False,
    cache_tag='',
    cache_dir=None,
):
    if stratified_manual:
        assert corrupt_type == "shuffle" or corrupt_type == 'feature'
        
    if corrupt_type == "shuffle":
        loaders, full_dict, shuffle_ind = load_torchvision_data_shuffle(
            dataname,
            valid_size=valid_size,
            random_seed=random_seed,
            batch_size=batch_size,
            resize=resize,
            stratified=stratified,
            stratified_manual=stratified_manual,
            shuffle=shuffle,
            maxsize=training_size,
            maxsize_test=test_size,
            shuffle_per=corrupt_por,
            transform=transform,
        )
    elif corrupt_type == 'feature':  
        loaders, full_dict, shuffle_ind = load_torchvision_data_perturb(
            dataname,
            valid_size=valid_size,
            random_seed=random_seed,
            batch_size=batch_size,
            resize=resize,
            stratified=stratified,
            stratified_manual=stratified_manual,
            shuffle=shuffle,
            maxsize=training_size,
            maxsize_test=test_size,
            transform=transform,
            perturb_per=corrupt_por, # probability of the noisy features i.e. Gaussian noise
        )
    elif corrupt_type == 'trojan_sq':
        loaders, full_dict, shuffle_ind = load_torchvision_data_trojan_sq(
            dataname,
            valid_size=valid_size,
            random_seed=random_seed,
            batch_size=batch_size,
            resize=resize,
            stratified=stratified,
            shuffle=shuffle,
            maxsize=training_size,
            maxsize_test=test_size,
            perturb_per=corrupt_por, # probability of the noisy features i.e. Gaussian noise
            trojan_class='airplane', # class of the trojan i.e. images with the backdoor are relabeled to this class
        )
    elif corrupt_type == 'poison_frogs':
        loaders, full_dict, shuffle_ind = load_torchvision_data_poison_frogs(
            dataname,
            model,
            device,
            valid_size=valid_size,
            random_seed=random_seed,
            batch_size=batch_size,
            resize=resize,
            stratified=stratified,
            shuffle=shuffle,
            maxsize=training_size,
            maxsize_test=test_size,
            perturb_per=corrupt_por,
            target_class='cat', # test set image which is used the blend into the base class
            base_class='frog',
            poison_frogs_feat_repr=poison_frogs_feat_repr,
            cache_dir=cache_dir,
            cache_tag=cache_tag,
            remake_data=remake_data,
            verbose=False,
        )
    else:
        raise ValueError
    
    return loaders, shuffle_ind

def load_data_corrupted_nonstat(
    model,
    device,
    splits=None,
    corrupt_type="shuffle", # {'shuffle', 'feature'}
    dataname=None,
    transform=None,
    valid_size=0,
    random_seed=2021,
    resize=None,
    stratified=True,
    shuffle=False,
    training_size=None,
    test_size=None,
    corrupt_por=0,
    batch_size=64,
    cache_dir=None,
):
    if corrupt_type == "shuffle":
        loaders, datasets, shuffle_ind = load_torchvision_data_nonstat_shuffle(
            dataname,
            splits=splits,
            valid_size=valid_size,
            random_seed=random_seed,
            batch_size=batch_size,
            resize=resize,
            stratified=stratified,
            shuffle=shuffle,
            maxsize=training_size,
            maxsize_test=test_size,
            transform=transform,
            shuffle_per=corrupt_por,
        )
    elif corrupt_type == "feature":
        loaders, datasets, shuffle_ind = load_torchvision_data_nonstat_feature(
            dataname,
            splits=splits,
            valid_size=valid_size,
            random_seed=random_seed,
            batch_size=batch_size,
            resize=resize,
            stratified=stratified,
            shuffle=shuffle,
            maxsize=training_size,
            maxsize_test=test_size,
            transform=transform,
            perturb_per=corrupt_por,
        )
    elif corrupt_por == "trojan_sq":
        pass
    elif corrupt_type == "poison_frogs":
        pass
    elif corrupt_type in ['backdoor-blend', 'backdoor-trojan-sq', 'backdoor-trojan-wm']:
        raise ValueError
    else: # empty or non-implemented == Loading Clean Data
        pass
    return loaders, datasets, shuffle_ind


# Get list of all indices of a dataset (subset)
# We use a train loader here
def get_indices(singleloader):
    return singleloader.batch_sampler.sampler.indices

def get_pruned_dataloader(
    corruption_type: str,
    sorted_gradient_ind_pruned: List[np.array],
    subset_indices: np.array,
    loaders: Dict[str, dataloader.DataLoader],
    train_batch_size: int
) -> dataloader.DataLoader:
    if corruption_type == "feature":
        sampler_class = SubsetSampler
        idxs = subset_indices[np.array([x[0] for x in sorted_gradient_ind_pruned]).reshape(-1)]
        np.random.shuffle(idxs)
        sampler = sampler_class(idxs)

        new_train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        loaders[f"train"].dataset.transform = new_train_transform

        trainloader = torch.utils.data.DataLoader(
            loaders[f"train"].dataset, # CustomDataset2 class
            sampler=sampler, 
            batch_size=train_batch_size, 
            num_workers=0,
        )
    elif corruption_type == "shuffle":
        
        new_train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        loaders[f"train"].dataset.transform = new_train_transform
        # use the pruning indices to subset the subset sampler indices
        trainset = torch.utils.data.Subset(
            loaders[f"train"].dataset, # dataset with data augmentation
            subset_indices[np.array([x[0] for x in sorted_gradient_ind_pruned]).reshape(-1)],
        )

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, shuffle=True, num_workers=0,
        )
    elif corruption_type == "poison_frogs":
        sampler_class = SubsetSampler
        idxs = subset_indices[np.array([x[0] for x in sorted_gradient_ind_pruned]).reshape(-1)]
        np.random.shuffle(idxs)
        sampler = sampler_class(idxs)

        new_train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        loaders[f"train"].dataset.transform = new_train_transform

        trainloader = torch.utils.data.DataLoader(
            loaders[f"train"].dataset, # CustomDataset2 class
            sampler=sampler, 
            batch_size=train_batch_size, 
            num_workers=0,
        )
    elif corruption_type == "trojan_sq":
        sampler_class = SubsetSampler
        idxs = subset_indices[np.array([x[0] for x in sorted_gradient_ind_pruned]).reshape(-1)]
        np.random.shuffle(idxs)
        sampler = sampler_class(idxs)

        new_train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        loaders[f"train"].dataset.transform = new_train_transform

        trainloader = torch.utils.data.DataLoader(
            loaders[f"train"].dataset, # CustomDataset2 class
            sampler=sampler, 
            batch_size=train_batch_size, 
            num_workers=0,
        )
    else:
        raise ValueError

    return trainloader


class Clothing1M(VisionDataset):
    def __init__(
        self, 
        root, 
        mode='train', 
        transform=None, 
        target_transform=None, 
        use_noisy=False,
        smoketest=False,
    ):
        super(Clothing1M, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
        )

        if not use_noisy: # benchmark setting
            flist = os.path.join(os.getcwd(), root, "annotations/clean_label_kv.txt")
            if mode=='train':
                subset_flist = os.path.join(os.getcwd(), root, "annotations/clean_train_key_list.txt")
            if mode=='val':
                subset_flist = os.path.join(os.getcwd(), root, "annotations/clean_val_key_list.txt")
            if mode=='test':
                subset_flist = os.path.join(os.getcwd(), root, "annotations/clean_test_key_list.txt")
        else: # using a noisy validation setting, saving clean labels for training
            subset_flist = None
            if mode=='train':
                flist = os.path.join(os.getcwd(), root, "annotations/noisy_label_kv.txt")
            if mode=='val':
                raise ValueError
            if mode=='test':
                raise ValueError

        self.impaths, self.targets = self.flist_reader(flist, subset_flist)

        # for debug
        if mode=='train' and smoketest:
            self.impaths, self.targets = self.impaths[:4000], self.targets[:4000]
            print(f"smoketest, impaths {len(self.impaths)}, targets {len(self.targets)}")

    def __getitem__(self, index):
        impath = self.impaths[index]
        target = self.targets[index]

        img = Image.open(impath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.impaths)

    def flist_reader(self, flist, subset_flist):
        # let's get the files in one of these
        # clean_test_key_list.txt
        # clean_train_key_list.txt
        # clean_val_key_list.txt
        # only if there is a match then we add it to the impaths
        # and targets which we return
        subset_list = []
        if subset_flist is not None:
            with open(subset_flist, 'r') as rf:
                for line in rf.readlines():
                    subset_list.append(line.strip())
        
        print(f"len subset list: {len(subset_list)}")

        impaths = []
        targets = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                row = line.split(" ")
                impath = str(os.path.join(os.getcwd(), self.root + '/' + row[0]))
                if len(subset_list) > 0:
                    if row[0] in subset_list:
                        impaths.append(impath)
                        targets.append(int(row[1]))
                else:
                    impaths.append(impath)
                    targets.append(int(row[1]))
        
        print(f"len impaths {len(impaths)}, len targets {len(targets)}")
        return impaths, targets