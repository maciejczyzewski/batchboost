#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc. (mixup)
# Copyright (c) 2020-present, Maciej A. Czyzewski (batchboost)
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

"""
!pip install efficientnet_pytorch
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
!cp gdrive/My\ Drive/<path>/utils.py .
!cp gdrive/My\ Drive/<path>/train.py .
!nvcc --version
!pip3 install --upgrade --force-reinstall torch torchvision
import torch
print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
print('Device:', torch.device('cuda:0'), torch.cuda.is_available())
# --- START ---
!python3 train.py --decay=1e-5 --no-augment --seed=1 \
	--name=batchboost --model=efficientnet-b0 --epoch=30
"""

# FIXME: rewrite it clean
import debug
from utils import progress_bar

try:
    import models

    COLAB = False
except:
    # FIXME: detect environment?
    print("=== GOOGLE COLAB ENVIRONMENT ===")
    COLAB = True

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument(
    "--model",
    default="ResNet18",
    type=str,
    help="model type (default: ResNet18)",
)
parser.add_argument("--name", default="0", type=str, help="name of run")
parser.add_argument("--seed", default=0, type=int, help="random seed")
parser.add_argument("--batch-size", default=128, type=int, help="batch size")
parser.add_argument(
    "--epoch", default=200, type=int, help="total epochs to run"
)
parser.add_argument(
    "--no-augment",
    dest="augment",
    action="store_false",
    help="use standard augmentation (default: True)",
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="lamb",
    choices=["lamb", "adam"],
    help="which optimizer to use",
)
parser.add_argument("--decay", default=1e-5, type=float, help="weight decay")
parser.add_argument(
    "--alpha",
    default=1.0,
    type=float,
    help="mixup interpolation coefficient (default: 1)",
)
parser.add_argument(
    "--debug",
    "-d",
    action="store_true",
    help="debug on FashionMNIST and ResNet100k network",
)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print("==> Preparing data..")
num_classes = 10

if args.debug:
    trainloader, testloader = debug.FashionMNIST_loaders(args)
else:
    if args.augment:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )

    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=8
    )

# Model
if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load(
        "./checkpoint/ckpt.t7" + args.name + "_" + str(args.seed)
    )
    net = checkpoint["net"]
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"] + 1
    rng_state = checkpoint["rng_state"]
    torch.set_rng_state(rng_state)
else:
    print("==> Building model..")
    if args.model.startswith("efficientnet"):
        from efficientnet_pytorch import EfficientNet

        net = EfficientNet.from_pretrained(args.model, num_classes=num_classes)
    elif args.debug:
        net = debug.ResNet100k()
    else:
        net = models.__dict__[args.model]()

if not os.path.isdir("results"):
    os.mkdir("results")
logname = (
    "results/log_"
    + net.__class__.__name__
    + "_"
    + args.name
    + "_"
    + str(args.seed)
    + ".csv"
)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print("device_count =", torch.cuda.device_count())
    cudnn.benchmark = True
    print("Using CUDA...")

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay
)

### MIXUP ######################################################################


def mixup_data(x, y, index_left, index_right, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mixed_x = lam * x[index_left, :] + (1 - lam) * x[index_right, :]
    y_a, y_b = y[index_left], y[index_right]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_mixup(epoch):
    print("MIXUP")
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        batch_size = inputs.shape[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, range(batch_size), index, args.alpha, use_cuda
        )
        inputs, targets_a, targets_b = map(
            Variable, (inputs, targets_a, targets_b)
        )
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += inputs.size(0)
        correct += (
            lam * predicted.eq(targets_a.data).cpu().sum().float()
            + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float()
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)"
            % (
                train_loss / (batch_idx + 1),
                reg_loss / (batch_idx + 1),
                100.0 * correct / total,
                correct,
                total,
            ),
        )
    return (
        train_loss / batch_idx,
        reg_loss / batch_idx,
        100.0 * correct / total,
    )


### BATCHBOOST #################################################################

from batchboost import BatchBoost


def fn_error(outputs, targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.sum(-outputs * logsoftmax(targets), dim=1)


def fn_linearize(x, num_classes=10):
    _x = torch.zeros(x.size(0), num_classes)
    _x[range(x.size(0)), x] = 1
    return _x


def fn_unlinearize(x):
    _, _x = torch.max(x, 1)
    return _x


BatchBoost.fn_error = fn_error
BatchBoost.fn_linearize = fn_linearize
BatchBoost.fn_unlinearize = fn_unlinearize

# FIXME: add arguments to command-line
BB = BatchBoost(
    alpha=args.alpha,
    window_normal=0,
    window_boost=10,
    factor=1 / 2,
    use_cuda=use_cuda,
)


def train_batchboost(epoch):
    global inputs, targets_a, targets_b, lam
    print("BATCHBOOST")
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0

    BB.clear()
    for batch_idx, (new_inputs, new_targets) in enumerate(trainloader):
        if use_cuda:
            new_inputs, new_targets = new_inputs.cuda(), new_targets.cuda()

        # -----> (a) feed with new information
        if not BB.feed(new_inputs, new_targets):
            continue

        # -----> (b) apply concat: BB.inputs, BB.targets
        outputs = net(BB.inputs)

        # -----> (c) calculate: loss (mixup like style \lambda)
        loss = BB.criterion(criterion, outputs)

        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += BB.inputs.size(0)  # -----> remember to use concat

        # -----> (d) calculate: accuracy
        correct += BB.correct(predicted)

        # -----> (e) pairing & mixing
        BB.mixing(criterion, outputs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)"
            % (
                train_loss / (batch_idx + 1),
                reg_loss / (batch_idx + 1),
                100.0 * correct / total,
                correct,
                total,
            ),
        )
    if total == 0:
        total = len(batch_size)
    return (
        train_loss / batch_idx,
        reg_loss / batch_idx,
        100.0 * correct / (total + 0.000001),
    )


### BASELINE ###################################################################


def train_baseline(epoch):
    print("BASELINE")
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += inputs.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)"
            % (
                train_loss / (batch_idx + 1),
                reg_loss / (batch_idx + 1),
                100.0 * correct / total,
                correct,
                total,
            ),
        )
    return (
        train_loss / batch_idx,
        reg_loss / batch_idx,
        100.0 * correct / total,
    )


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )
    acc = 100.0 * correct / total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (test_loss / batch_idx, 100.0 * correct / total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print("Saving..")
    state = {
        "net": net,
        "acc": acc,
        "epoch": epoch,
        "rng_state": torch.get_rng_state(),
    }
    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")
    torch.save(state, "./checkpoint/ckpt.t7" + args.name + "_" + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if not os.path.exists(logname):
    with open(logname, "w") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(
            [
                "epoch",
                "train loss",
                "reg loss",
                "train acc",
                "test loss",
                "test acc",
            ]
        )

if args.name == "batchboost":
    train_func = train_batchboost
elif args.name == "mixup":
    train_func = train_mixup
else:
    train_func = train_baseline

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train_func(epoch)
    test_loss, test_acc = test(epoch)
    adjust_learning_rate(optimizer, epoch)
    with open(logname, "a") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(
            [epoch, train_loss, reg_loss, train_acc, test_loss, test_acc]
        )
