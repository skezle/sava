import os
import argparse
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms

from data import Clothing1M
from models.resnet import ResNet18, test_new
from models.preact_resnet import PreActResNet18, test
from models.utils import train, evaluate

import wandb

"""
Code from https://github.com/chenpf1025/RobustnessAccuracy/blob/master/train_clothing1m_ce.py

Run smoketest:

python train_clothing1M.py --random_seed=99 --n_gpu=4 --tag=smoketest \
    --smoketest --cuda_num=0

Run full training run on clean training set needed for :

python train_clothing1M.py --random_seed=0 --n_gpu=8 \
    --cuda_num=0 --tag=preactresnet_b256 --batch_size=512

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

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='PyTorch Clothing1M')
    parser.add_argument(
        '--seed',
        type=int,
        default=2021,
    )
    parser.add_argument(
        '--cuda_num',
        type=int, 
        help='number of cuda in the server',
    )
    parser.add_argument(
        '--n_gpu', 
        type=int,
        default=2, 
        help='number of gpu to use',
    )
    parser.add_argument(
        '--root', 
        type=str, 
        default='data/clothing1M/',
        help='root of dataset',
    )
    parser.add_argument(
        '--tag', 
        type=str, 
        default='', 
        help='unique tag for logging',
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=128, 
        help='input batch size for training',
    )
    parser.add_argument(
        '--smoketest', 
        action='store_true', 
        default=False, 
        help='Smoketest for training on less data for fewer epochs',
    )
    parser.add_argument(
        '--use_noisy', 
        action='store_true', 
        default=False, 
        help='Using the noisy dataset for training only.',
    )
    parser.add_argument(
        '--test_batch_size', 
        type=int, 
        default=128, 
        help='input batch size for testing',
    )
    parser.add_argument(
        '--terminate_early', 
        action='store_true', 
        default=False, 
        help='For EL2N early model data valuation, terminate training early at 10 epochs.',
    )
    parser.add_argument(
        '--preact_resnet', 
        action='store_true', 
        default=False, 
        help='Train a preact resnet instead of normal resnet.',
    )
    args = parser.parse_known_args(args=args)[0]
    return args
   
def main():
    # Settin
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cuda_num = args.cuda_num
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # wandb logger
    wandb.init(
        group="clothing1m",
        name=f"clothing1m_{args.tag}",
        project="ot-data-selection",
        config={
            "dataset": "Clothing1M",
        }
    )

    # Datasets
    root = args.root
    num_classes = 14
    kwargs = {'num_workers': 16, 'pin_memory': True} if torch.cuda.is_available() else {}
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
 
    train_dataset = Clothing1M(
        root, 
        mode='train', 
        transform=train_transform, 
        use_noisy=args.use_noisy,
        smoketest=args.smoketest,
    )

    val_dataset = Clothing1M(
        root,
        mode='val',
        transform=test_transform, 
        use_noisy=args.use_noisy,
    )

    test_dataset = Clothing1M(
        root, 
        mode='test', 
        transform=test_transform, 
        use_noisy=args.use_noisy,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, 
        shuffle=True, 
        **kwargs,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.test_batch_size, 
        shuffle=False, 
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.test_batch_size, 
        shuffle=False, 
        **kwargs,
    )
    
    criterion = nn.CrossEntropyLoss()

    # Building model
    if args.preact_resnet:
        net = PreActResNet18(num_classes=num_classes, imagenet=True).to(device)
        test(num_classes=num_classes)
    else:
        net = ResNet18(num_classes=num_classes, imagenet=True).to(device)
        test_new(num_classes=num_classes)
    net = torch.nn.DataParallel(net, device_ids=list(range(cuda_num, cuda_num + args.n_gpu)))

    optimizer = optim.SGD(
        net.parameters(), 
        lr=0.1, 
        momentum=0.9, 
        weight_decay=1e-3,
    )

    cudnn.benchmark = True # Accelerate training by enabling the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

    # Training
    if args.smoketest:
        schedule = [
            (0, 5, .1),
            (5, 8, .01),
            (8, 10, .001),
        ]
    else:
        schedule = [
            (0, 50, .1),
            (50, 75, .01),
            (75, 100, .001),
        ]
    
    step = 0
    for start, end, lr in schedule:
        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for epoch in range(start, end):
            train(
                step,
                train_loader,
                single=False,
                net=net,
                optimizer=optimizer,
                device=device,
                criterion=criterion,
            )
            step += len(train_loader)

            evaluate(
                step, 
                train_loader, 
                val_loader,
                single=False,
                net=net,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
            )

            print("saving model")
            if not args.smoketest and (epoch + 1) % 10 == 0: 
                torch.save(
                    net.module.state_dict(),
                    os.path.join(
                        os.getcwd(),
                        "checkpoint",
                        f"clothing1m_clean_train_resnet18_{epoch}_{args.tag}.pth",
                    )
                )
            
            if args.terminate_early and (epoch + 1) > 10:
                sys.exit()

if __name__ == '__main__':
    main()