import os
import pickle
import wandb
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.optim as optim

from input_args import nonstat_args
from logger import Logger

from data import load_data_corrupted_nonstat, get_indices
from models.preact_resnet import load_pretrained_feature_extractor
from api import lava_experiment, hierarchical_ot_experiment
from models.resnet import ResNet18
from models.utils import train, evaluate
from otdd.pytorch.datasets import SubsetSampler
from knn_shapley import knn_shap_experiment


print(torchvision.__version__)
print(torch.__version__)

if __name__ == "__main__":
    args = nonstat_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_num)
    print("GPU", os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.hierarchical:
        wandb_tag = 'hot_nonstat'
    elif args.knn_sv:
        wandb_tag = 'knn_sv_nonstat'
    else:
        wandb_tag = 'lava_nonstat'
    # wandb logger
    logger = Logger(
        group=wandb_tag,
        name=f"{wandb_tag}_tr_sz_{args.tag}", 
        project="ot-data-selection",
        method=wandb_tag, 
        dataset="CIFAR10",
    )

    # data selection hyperparameters
    test_size = 400 if args.smoketest else 10000
    valid_size = 0 if args.smoketest else args.val_dataset_size # can be used for hyperparam tuning of ResNet18
    training_size = 500 if args.smoketest else 50000 - valid_size
    resize = 32
    portion = args.corrupt_por
    batch_size = args.hot_batch_size
    shuffle = False

    # training hyperparameters
    pruning_percentage = args.prune_perc
    lr = 0.1
    start_epoch = 0
    end_epoch = 3 if args.smoketest else 200
    train_batch_size = 128
    num_tasks = 5

    feature_extractor = load_pretrained_feature_extractor(
        "cifar10_embedder_preact_resnet18.pth",
        device,
    )

    # data transformations for torch.Dataset
    if args.corruption_type == "feature":
        # no normalization for noisy features data
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_transform_selection = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_transform_selection = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    # get DataLoaders for data selection and training (with data augmentation)
    loaders, datasets, shuffle_ind = load_data_corrupted_nonstat(
        feature_extractor,
        device,
        shuffle=shuffle,
        corrupt_type=args.corruption_type,  # {'shuffle', 'feature'}
        dataname="CIFAR10",
        transform=(train_transform_selection, train_transform, test_transform),
        random_seed=args.random_seed,
        resize=resize,
        splits=num_tasks,
        training_size=training_size,
        test_size=test_size,
        valid_size=valid_size,
        corrupt_por=portion,
        batch_size=batch_size,
        cache_dir=os.path.join(os.getcwd(), "data"),
    )

    # data selection and training loop
    for i in range(num_tasks):
        print(f"Task {i}")

        indices_path = os.path.join(os.getcwd(), args.resume_inds_path)
        indices_file = os.path.join(os.getcwd(), args.resume_inds_path, str(i) + ".pickle")

        if args.resume and os.path.isfile(indices_file):
            print("Loading cached indices")
            with open(indices_file, 'rb') as handle:
                tmp = pickle.load(handle)
            subset_indices = tmp["indices"]
            sorted_gradient_ind_pruned = tmp["sorted_gradient_ind_pruned"]
        else:
            if args.hierarchical:
                sorted_gradient_ind, trained_with_flag = hierarchical_ot_experiment(
                    feature_extractor=feature_extractor,
                    train_loader=loaders[f"train_sel_{i}"],
                    val_loader=loaders["test"],
                    training_size=int(((i + 1) * training_size) / num_tasks),
                    batch_size=batch_size,
                    shuffle_ind=shuffle_ind,
                    resize=resize,
                    portion=portion,
                    device=device,
                    cache_label_distances=args.cache_l2l,
                    tag=str(i),
                )
            elif args.knn_sv:
                sorted_gradient_ind, trained_with_flag = knn_shap_experiment(
                    feature_extractor=feature_extractor,
                    train_loader=loaders[f"train_sel_{i}"],
                    val_loader=loaders["test"],
                    training_size=int(((i + 1) * training_size) / num_tasks),
                    k=args.k,
                    output_repr=args.output_repr,
                    device=device,
                    shuffle_ind=shuffle_ind,
                    portion=portion,
                    tag=str(i),
                )
            else:
                sorted_gradient_ind, trained_with_flag = lava_experiment(
                    feature_extractor=feature_extractor,
                    train_loader=loaders[f"train_sel_{i}"],
                    val_loader=loaders["test"],
                    training_size=int(((i + 1) * training_size) / num_tasks),
                    shuffle_ind=shuffle_ind,
                    resize=resize,
                    portion=portion,
                    feat_repr=False,
                    device=device,
                    tag=str(i),
                )

            prune_ind = int(pruning_percentage * len(sorted_gradient_ind))
            sorted_gradient_ind_pruned = sorted_gradient_ind[prune_ind:]

            subset_indices = get_indices(loaders[f"train_sel_{i}"])
            
            # cache
            if not os.path.exists(indices_path):
                os.makedirs(indices_path)
            with open(indices_file, 'wb') as handle:
                pickle.dump(
                    {
                        'sorted_gradient_ind_pruned': sorted_gradient_ind_pruned,
                        'subset_indices': subset_indices
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        if args.corruption_type == "feature":
            sampler_class = SubsetRandomSampler if shuffle else SubsetSampler
            idxs = subset_indices[np.array([x[0] for x in sorted_gradient_ind_pruned]).reshape(-1)]
            sampler = sampler_class(idxs)

            new_train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            loaders[f"train_{i}"].dataset.transform = new_train_transform

            trainloader = torch.utils.data.DataLoader(
                loaders[f"train_{i}"].dataset, # CustomDataset2 class
                sampler=sampler, 
                batch_size=train_batch_size, 
                num_workers=0,
            )
        elif args.corruption_type == "shuffle":
            # use the pruning indices to subset the subset sampler indices
            trainset = torch.utils.data.Subset(
                loaders[f"train_{i}"].dataset, # dataset with data augmentation
                subset_indices[np.array([x[0] for x in sorted_gradient_ind_pruned]).reshape(-1)],
            )

            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=train_batch_size, shuffle=True, num_workers=0,
            )
        else:
            raise ValueError
        
        net = ResNet18().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            net.parameters(), 
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        if args.smoketest:
            schedule = [
                (0, 1, .1),
                (1, 2, .01),
                (2, 3, .001),
            ]
        else:
            schedule = [
                (0, 100, .1),
                (100, 150, .01),
                (150, 200, .001),
            ]

        checkpoint_dir = os.path.join(os.getcwd(), args.resume_checkpoint_path, str(i))
        checkpoint_file = os.path.join(os.getcwd(), args.resume_checkpoint_path, str(i), str(args.resume_epoch) + ".pth")

        epoch = 0

        if args.resume and os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            # Update the model state
            net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint["epoch"]
        
        for start, end, lr in schedule:

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            while start <= epoch < end:
                train(
                    epoch * len(trainloader.dataset), 
                    trainloader,
                    single=False,
                    net=net,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                )
                epoch += 1

                evaluate(
                    epoch * len(trainloader.dataset),
                    trainloader,
                    loaders["test"],
                    single=False,
                    net=net,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    tag=f"val-{i}",
                )

                # cache
                # if (epoch + 1) % 50 == 0:
                #     if not os.path.exists(checkpoint_dir):
                #         os.makedirs(checkpoint_dir)
                #     torch.save(
                #         {
                #             'model_state_dict': net.state_dict(),
                #             'optimizer_state_dict': optimizer.state_dict(),
                #             'epoch': epoch,
                #         },
                #         os.path.join(os.getcwd(), args.resume_checkpoint_path, str(epoch) + ".pth"),
                #     )








        




