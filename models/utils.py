import wandb

import torch
import torchvision
import torch.utils.data.dataloader as dataloader

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


print(torchvision.__version__)
print(torch.__version__)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(
    step: int,
    trainloader: dataloader.DataLoader,
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    single: bool,
) -> None:
    print('\nStep: %d' % step)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets) # average over a batch
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        acc = predicted.eq(targets)
        correct += acc.sum().item()
        lr = get_lr(optimizer)

        wandb.log(
            {
                "train-loss": train_loss/(batch_idx+1),
                "acc": 100.*correct/total,
                "lr": lr,
                "step": step + batch_idx,
            }
        )

        if batch_idx % 10 == 0:
            print(f"batch_idx {batch_idx} / {len(trainloader)}, loss: {train_loss/(batch_idx+1)},"
                f" acc: {100.*correct/total}, lr scheduler: {lr}")
        
        if single:
            break
    

def evaluate(
    step: int,
    trainloader: dataloader.DataLoader,
    testloader: dataloader.DataLoader,
    single: bool,
    net: nn.Module,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    tag: str = "val",
) -> None:
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print(f"batch_idx {batch_idx} / {len(testloader)}, test loss: {test_loss/(batch_idx+1)},"
                        f" test acc: {100.*correct/total}, lr scheduler: {get_lr(optimizer)}")
        
            if single:
                break

        wandb.log(
            {
                f"{tag}-loss": test_loss/(batch_idx+1),
                f"{tag}-acc": 100.*correct/total,
                "step": step,
            }
        )