import time
import os
import wandb
import random
import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn

from logger import Logger
from api import hierarchical_ot_experiment, batchwise_lava_experiment
from input_args import parse_args_clothing1m
from el2n import el2n_experiment
from models.resnet import ResNet18, load_pretrained_resnet18_feature_extractor, test_new
from models.preact_resnet import PreActResNet18, test, load_pretrained_feature_extractor
from models.utils import train, evaluate

from data import Clothing1M

print(torchvision.__version__)
print(torch.__version__)

"""
# smoketest:
python value_clothing1M.py --seed=99 --cuda_num=0 --n_gpu=2 --value_batch_size=1024 \
    --tag=test --smoketest

# full run random pruning:
python value_clothing1M.py --seed=0 --cuda_num=0 --n_gpu=8 --tag=random_pruning_new_wd0002_s0 \
    --train_batch_size=512 --wd=0.002  --prune_percs 0.0 0.1 0.2 0.3 0.4

# full run SAVA/hot data valuation + pruning:
python value_clothing1M.py --seed=0 --cuda_num=0 --n_gpu=8 --value_batch_size=2048 \
    --tag=hot_hotbs2048_wd0002_s0 --hot --prune_percs 0.1 0.2 0.3 0.4 --train_batch_size=512 \
    --wd=0.002 --values_tag=clothing1m_hot_values_resnet18_feat_extra_bs4096_hot_hotbs2048_s0

# full EL2N data valuation + pruning:
# main hyperparameter is the prune_interval for EL2N, which is optimized and left is best

python value_clothing1M.py --seed=0 --cuda_num=0 --n_gpu=8 --el2n --el2n_num_models=10 \
     --prune_percs 0.1 0.2 0.3 0.4 --train_batch_size=512 --tag=el2n_left \
     --value_batch_size=512 --values_tag=clothing1m_el2n_values_resnet18_feat_extra_n10_el2n_trbs512 \
     --wd=0.002 --prune_interval=left

# Supervised prototypes:
python value_clothing1M.py --seed=0 --cuda_num=0 --n_gpu=8 --slp \
     --prune_percs 0.4 --train_batch_size=512 --tag=slp_n10k \
     --value_batch_size=512 --values_tag=supervised_prototypes_n10000_test_lr005_s0 \
     --wd=0.002

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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == "__main__":
    args = parse_args_clothing1m()
    print(f"args: {args}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cuda_num = args.cuda_num
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed_everything(args.seed)

    # constants
    value_batch_size = args.value_batch_size
    root = args.root
    resize = 224
    
    # training hparams
    lr = 0.1
    train_batch_size = args.train_batch_size
    num_classes = 14
    
    # Create a DataLoader with a fixed seed for its workers
    g = torch.Generator()
    g.manual_seed(args.seed)  # This ensures that shuffling is the same each run
    kwargs = {'num_workers': 16, 'pin_memory': True,  'worker_init_fn': seed_worker, 'generator': g} if torch.cuda.is_available() else {}

    train_transform = transforms.Compose(
        [
            transforms.Resize((256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ]
    )

    start = time.time()

    train_dataset = Clothing1M(
        root, 
        mode='train', 
        transform=train_transform, 
        use_noisy=True,
        smoketest=args.smoketest,
    )

    val_dataset = Clothing1M(
        root,
        mode='val',
        transform=test_transform, 
        use_noisy=False,
    )
    
    test_dataset = Clothing1M(
        root, 
        mode='test', 
        transform=test_transform, 
        use_noisy=False,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.value_batch_size, 
        shuffle=False, # do that we can consistently value training points
        **kwargs,
    )

    val_loader_hot = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.value_batch_size, 
        shuffle=False, 
        **kwargs,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=train_batch_size, 
        shuffle=False, 
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=train_batch_size, 
        shuffle=False, 
        **kwargs,
    )

    # Data valuation
    if args.hot or args.batch_lava:
        # check that encoder has loaded correctly by check the accuracy is high
        # wandb logger
        mtd = "hot" if args.hot else "batch_lava"
        logger = Logger(
            group=f'clothing1m_{mtd}',
            name=f"clothing1m_{mtd}_{args.tag}",
            project="ot-data-selection",
            method=f"{mtd}", 
            dataset="Clothing1M",
            smoketest=args.disable_wandb,
        )
        feature_extractor = load_pretrained_resnet18_feature_extractor(
            feature_extractor_name="clothing1m_clean_train_resnet18_99_full_b256.pth",
            device=device,
            num_classes=num_classes,
            imagenet=True,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            feature_extractor.parameters(), 
            lr=lr,
            momentum=0.9,
            weight_decay=args.wd,
        )
        evaluate(
            -1, 
            train_loader, 
            val_loader,
            single=False,
            net=feature_extractor,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
        )
        logger.close()
        # if the values already exist let's load them and continue with the pruning
        values_file = os.path.join(
            os.getcwd(),
            "output",
            f"{args.values_tag}.pickle",
        )
        if os.path.exists(values_file):
            with open(values_file, 'rb') as f:
                data = pickle.load(f)
            sorted_values = data["values"]
            print(f"Data loaded from pickle file: {values_file}")
        else:
            if args.hot:
                print("Running SAVA valuation")
                sorted_values = hierarchical_ot_experiment(
                    feature_extractor=feature_extractor.to(device),
                    train_loader=train_loader,
                    val_loader=val_loader_hot,
                    training_size=len(train_loader.dataset),
                    batch_size=value_batch_size,
                    shuffle_ind=None,
                    resize=resize,
                    portion=None,
                    device=device,
                    cache_label_distances=True,
                    visualise_hot=False,
                    feat_repr=False, # TODO: test True / False
                    num_classes=num_classes,
                    parallel=True,
                    cuda_num=cuda_num,
                    n_gpu=args.n_gpu,
                )
                with open(values_file, 'wb') as f:
                    pickle.dump({"values": sorted_values}, f)
                print(f"HOT values saved: {values_file}")
            elif args.batch_lava:
                print("Running batch-wise LAVA valuation")
                sorted_values = batchwise_lava_experiment(
                    feature_extractor=feature_extractor.to(device),
                    train_loader=train_loader,
                    val_loader=val_loader_hot,
                    training_size=len(train_loader.dataset),
                    batch_size=value_batch_size,
                    shuffle_ind=None,
                    resize=resize,
                    portion=None,
                    feat_repr=False,
                    device=device,
                    cache_label_distances=True,
                    num_classes=num_classes,
                    parallel=True,
                    cuda_num=cuda_num,
                    n_gpu=args.n_gpu,
                )
                with open(values_file, 'wb') as f:
                    pickle.dump({"values": sorted_values}, f)
                print(f"Batchwise LAVA values saved: {values_file}")
            else:
                raise ValueError
    elif args.el2n:
        # if the values already exist let's load them and continue with the pruning
        values_file = os.path.join(
            os.getcwd(),
            "output",
            f"{args.values_tag}.pickle",
        )
        if os.path.exists(values_file):
            with open(values_file, 'rb') as f:
                data = pickle.load(f)
            sorted_values = data["values"]
            print(f"Data loaded from pickle file: {values_file}")
        else:
            print("Running EL2N valuation")
            sorted_values = el2n_experiment(
                value_init_seed=args.el2n_value_model_seed,
                value_n_models=args.el2n_value_num_models,
                train_loader=train_loader,
                training_size=len(train_loader.dataset),
                batch_size=value_batch_size,
                device=device,
                num_classes=num_classes,
                cuda_num=cuda_num,
                n_gpu=args.n_gpu,
            )
            with open(values_file, 'wb') as f:
                pickle.dump({"values": sorted_values}, f)
            print(f"EL2N values generated and saved to pickle file: {values_file}")
    elif args.slp:
        values_file = os.path.join(
            os.getcwd(),
            "output",
            f"{args.values_tag}.pickle",
        )
        if os.path.exists(values_file):
            with open(values_file, 'rb') as f:
                data = pickle.load(f)
            sorted_values = data["values"]
            print(f"Data loaded from pickle file: {values_file}")
        else:
            print(f"cannot find {values_file}")
            raise ValueError
    else:
        # random pruning
        sorted_values = np.random.choice(range(len(train_loader.dataset)), size=len(train_loader.dataset), replace=False)
        sorted_values = [np.array(x).reshape(1, ) for x in sorted_values]

    for pruning_percentage in args.prune_percs:
        # wandb logger
        logger = Logger(
            group='clothing1m_hot',
            name=f"clothing1m_hot_pp{str(pruning_percentage)}_{args.tag}",
            project="ot-data-selection",
            method="hot", 
            dataset="Clothing1M",
            smoketest=args.disable_wandb,
        )

        if args.prune_interval == "right":
            left, right = int(pruning_percentage * len(sorted_values)),  int(len(sorted_values))
        elif args.prune_interval == "mid":
            mid = int(0.5 * pruning_percentage * len(sorted_values))
            left, right = mid,  int(len(sorted_values)) - mid
        elif args.prune_interval == "left":
            left, right = 0, int(len(sorted_values)) - int(pruning_percentage * len(sorted_values))
        else:
            raise ValueError
        sorted_sorted_values_pruned = sorted_values[left:right]
        #prune_ind = int(pruning_percentage * len(sorted_values))
        #sorted_sorted_values_pruned = sorted_values[prune_ind:]

        # use the pruning indices to subset the subset sampler indices
        train_dataset = torch.utils.data.Subset(
            train_loader.dataset, # dataset with data augmentation
            np.array([x[0] for x in sorted_sorted_values_pruned]).reshape(-1),
        )

        trainloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=train_batch_size,
            shuffle=True,
            **kwargs,
        )

        if args.preact_resnet:
            net = PreActResNet18(num_classes=num_classes, imagenet=True).to(device)
            test(num_classes=num_classes)
        else:
            net = ResNet18(num_classes=num_classes, imagenet=True).to(device)
            test_new(num_classes=num_classes)

        net = torch.nn.DataParallel(net, device_ids=list(range(cuda_num, cuda_num + args.n_gpu)))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            net.parameters(), 
            lr=lr,
            momentum=0.9,
            weight_decay=args.wd,
        )

        cudnn.benchmark = True # Accelerate training by enabling the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

        if args.smoketest:
            schedule = [
                (0, 2_000, .1),
                (2_000, 4_000, .01),
                (4_000, 6_000, .001),
            ]
        else:
            schedule = [
                (0, 30_000, .1),
                (30_000, 60_000, .05),
                (60_000, 80_000, .01),
                (80_000, 90_000, .001),
                (90_000, 95_000, .0001),
                (95_000, 100_000, .00001),
            ]

        # training loop
        step = 0
        for start, end, lr in schedule:
            # Set learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            while start <= step < end:
                train(
                    step,
                    trainloader,
                    single=False,
                    net=net,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                )
                step += len(trainloader)

                evaluate(
                    step, 
                    trainloader,
                    val_loader,
                    single=False, 
                    net=net,
                    device=device,
                    optimizer=optimizer,
                    criterion=criterion,
                )
        
        evaluate(
            step, 
            trainloader,
            test_loader,
            single=False, 
            net=net,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            tag="test",
        )
        logger.close()