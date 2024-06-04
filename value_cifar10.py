import time
import os
import wandb
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from input_args import parse_args
from logger import Logger
from api import lava_experiment, hierarchical_ot_experiment, batchwise_lava_experiment
from data import load_data_corrupted, get_indices, get_pruned_dataloader
from models.preact_resnet import load_pretrained_feature_extractor
from models.resnet import ResNet18
from models.utils import train, evaluate

print(torchvision.__version__)
print(torch.__version__)

"""
Independent LAVA:
seed=0
python value_cifar10.py --random_seed=${seed} --corruption_type=shuffle --corrupt_por=0.3 --feat_repr \
    --tag=indep_lava_labels_s${seed} --cuda_num=0 --batchwise_lava

"""

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_num)
    print("GPU", os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed_everything(args.random_seed)

    train_dataset_sizes = args.train_dataset_sizes

    for tr_size in train_dataset_sizes:
        
        logger = Logger(
            group='hot' if args.hierarchical else 'lava',
            name=f"hot_tr_sz_{tr_size}_{args.tag}" if args.hierarchical else f"lava_tr_sz_{tr_size}_{args.tag}",
            project="ot-data-selection",
            method="hot" if args.hierarchical else "lava", 
            dataset="CIFAR10",
            smoketest=args.disable_wandb,
        )

        # constants
        training_size = tr_size
        resize = 32
        portion = args.corrupt_por
        batch_size = args.hot_batch_size
        feat_repr = args.feat_repr
        remake_data = args.remake_data # for poison frogs where the data gen is expensive
        eval = args.evaluate
        if eval:
            valid_size = 0
            test_size = 400 if args.smoketest else args.val_dataset_size
        else:
            valid_size = 400 if args.smoketest else args.val_dataset_size
            test_size = 400 if args.smoketest else args.val_dataset_size
        
        assert tr_size + valid_size <= 50000, "training size + validation size should be less than 50000"

        # training hparams
        pruning_percentage = args.prune_perc
        lr = 0.1
        train_batch_size = 128
        start_epoch = 0
        end_epoch = 3 if args.smoketest else 200

        feature_extractor = load_pretrained_feature_extractor(
            "cifar10_embedder_preact_resnet18.pth",
            device,
        )

        if args.data_gen_force_cpu:
            device_data_gen = torch.device('cpu')
        else:
            device_data_gen = device

        # data
        loaders, shuffle_ind = load_data_corrupted(
            feature_extractor.to(device_data_gen),
            device_data_gen,
            corrupt_type=args.corruption_type,  # {'shuffle', 'feature', 'trojan_sq', 'poison_frogs'}
            dataname="CIFAR10",
            random_seed=args.random_seed,
            resize=resize,
            training_size=training_size,
            test_size=test_size,
            valid_size=valid_size,
            corrupt_por=portion,
            batch_size=batch_size,
            poison_frogs_feat_repr=args.poison_frogs_feat_repr,
            remake_data=remake_data,
            cache_dir=os.path.join(os.getcwd(), "data"),
            cache_tag=args.cache_tag,
            stratified_manual=args.stratified,
        )

        start = time.time()

        if args.batchwise_lava:
            sorted_gradient_ind, trained_with_flag = batchwise_lava_experiment(
                feature_extractor=feature_extractor.to(device),
                train_loader=loaders["train"],
                val_loader=loaders["test"] if eval else loaders["valid"],
                training_size=training_size,
                batch_size=batch_size,
                shuffle_ind=shuffle_ind,
                resize=resize,
                portion=portion,
                feat_repr=feat_repr,
                device=device,
                cache_label_distances=args.cache_l2l,
            )
        elif args.hierarchical:
            sorted_gradient_ind, trained_with_flag = hierarchical_ot_experiment(
                feature_extractor=feature_extractor.to(device),
                train_loader=loaders["train"],
                val_loader=loaders["test"] if eval else loaders["valid"],
                training_size=training_size,
                batch_size=batch_size,
                shuffle_ind=shuffle_ind,
                resize=resize,
                portion=portion,
                device=device,
                cache_label_distances=args.cache_l2l,
                visualise_hot=args.visualise_hot,
            )
        else:
            sorted_gradient_ind, trained_with_flag = lava_experiment(
                feature_extractor=feature_extractor.to(device),
                train_loader=loaders["train"],
                val_loader=loaders["test"] if eval else loaders["valid"],
                training_size=training_size,
                shuffle_ind=shuffle_ind,
                resize=resize,
                portion=portion,
                feat_repr=feat_repr,
                device=device,
            )
        
        print(f"run time: {time.time() - start:.2f}s")
        wandb.log({"run_time": time.time() - start}, step=1)

        if args.train_net:
            prune_ind = int(pruning_percentage * len(sorted_gradient_ind))
            sorted_gradient_ind_pruned = sorted_gradient_ind[prune_ind:]
            total = sum([trained_with_flag[i][2] for i in range(len(trained_with_flag))])
            found = sum(
                [trained_with_flag[sorted_gradient_ind_pruned[i][0]][2] for i in range(len(sorted_gradient_ind_pruned))]
            )
            print(f"num corrupted points in pruned training set: {found} / {total}")
            subset_indices = get_indices(loaders[f"train"])

            trainloader = get_pruned_dataloader(
                args.corruption_type,
                sorted_gradient_ind_pruned,
                subset_indices,
                loaders,
                train_batch_size,
            )

            # reseting the batch size for test
            testloader = torch.utils.data.DataLoader(
                loaders["test"].dataset if eval else loaders["valid"].dataset,
                batch_size=train_batch_size, 
                num_workers=0,
            )

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

            # training loop
            epoch = 0
            for start, end, lr in schedule:

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                while start <= epoch < end:
                    train(
                        epoch * len(trainloader.dataset), 
                        trainloader,
                        single=False,
                        net=net,
                        optimize=optimizer,
                        criterion=criterion,
                        device=device,
                    )
                    epoch += 1

                    evaluate(
                        epoch * len(trainloader.dataset), 
                        trainloader,
                        testloader, 
                        single=False,
                        net=net,
                        optimizer=optimizer,
                        criterion=criterion,
                        device=device,
                    )
        
        logger.close()
