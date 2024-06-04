import os
import pickle
import wandb
import random
import argparse
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from logger import Logger

from models.resnet import ResNet18, test_new
from models.utils import train, evaluate
from otdd.pytorch.datasets import SubsetSampler

print(torchvision.__version__)
print(torch.__version__)

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--random_seed',
        type=int,
        default=2021,
    )
    parser.add_argument(
        '--cuda_num', 
        type=int, 
        help='number of cuda in the server',
    )
    args = parser.parse_known_args(args=args)[0]
    return args

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.random_seed)

    cuda_num = args.cuda_num
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda_num)
    print("GPU", os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # wandb logger
    wandb.init(
        group="tracin_100e",
        name="tracin_100e_resnet18",
        project="ot-data-selection",
        config={
            "dataset": "CIFAR10",
        }
    )

    lr = 0.1
    batch_size = 128
    seed = 0

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    # trainset = torch.utils.data.Subset(
    #     trainset, np.random.choice(len(trainset), size=10000, replace=False))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = ResNet18().to(device) # final linear layer 512 -> 10
    #net = ResNet18(feat_dim=128, pool=4).to(device) # final linear layer 128 -> 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), 
        lr=lr, 
        momentum=0.9, 
        weight_decay=5e-4,
    )

    # schedule = [
    #     (0, 100, .1),
    #     (100, 150, .01),
    #     (150, 200, .001),
    # ]

    schedule = [
        (0, 50, .1),
        (50, 75, .01),
        (75, 100, .001),
    ]

    # schedule = [
    #     (0, 30, .1),
    #     (30, 40, .01),
    #     (40, 50, .001),
    # ]

    for start, end, lr in schedule:
        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for epoch in range(start, end):
            train(
                epoch,
                trainloader,
                single=False,
                net=net,
                optimizer=optimizer,
                device=device,
                criterion=criterion,
            ) # for knn shap we train on the val set 
            evaluate(
                epoch, 
                trainloader, 
                testloader,
                single=False,
                net=net,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
            ) # for knn shap we take 10k samples from train as the val set

            print("saving model")
            torch.save(
                net.state_dict(),
                os.path.join(
                    os.getcwd(),
                    "checkpoint",
                    f"p2_cifar10_100e_resnet18_new_{epoch}.pth",
                )
            )

    # print("saving model")
    # torch.save(
    #     net.state_dict(),
    #     os.path.join(
    #         os.getcwd(),
    #         "checkpoint",
    #         f"p2_cifar10_embedder_resnet18_10k_val_dim128_{epoch}.pth",
    #     )
    # )











        




