import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as datasets

# FIXME: move to models and split for CIFAR-10/Fashion-MNIST and others


class ResNet100k(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet100k, self).__init__()
        self.num_filter1 = 8
        self.num_filter2 = 16
        self.num_padding = 2
        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv2d(1, self.num_filter1, 5, padding=self.num_padding)
        nn.init.xavier_uniform_(self.conv1.weight)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding
        self.conv2 = nn.Conv2d(
            self.num_filter1, self.num_filter2, 5, padding=self.num_padding
        )
        nn.init.xavier_uniform_(self.conv2.weight)
        # feature map size is 7*7 by pooling
        self.fc = nn.Linear(self.num_filter2 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_filter2 * 7 * 7)  # reshape Variable
        x = self.fc(x)
        return x
        # return F.log_softmax(x, dim=1)
        # return F.softmax(x, dim=1)


class ResNet100kv2(nn.Module):
    def __init__(self):
        super(ResNet100kv2, self).__init__()

        self.cnn1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2
        )
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(16)
        nn.init.xavier_uniform(self.cnn1.weight)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
        )
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(32)
        nn.init.xavier_uniform(self.cnn2.weight)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(2048, 128)
        self.fcrelu = nn.ReLU()

        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.norm1(out)

        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.norm2(out)

        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fcrelu(out)

        out = self.fc2(out)
        return out


def FashionMNIST_loaders(args):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    if args.augment:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    else:
        transform_train = transform
    transform_test = transform

    trainset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )

    testset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=8
    )

    return trainloader, testloader
