import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as datasets


class ResNet100k(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet100k, self).__init__()
        self.num_filter1 = 8
        self.num_filter2 = 16
        self.num_padding = 2
        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv2d(1,
                               self.num_filter1,
                               5,
                               padding=self.num_padding)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding
        self.conv2 = nn.Conv2d(self.num_filter1,
                               self.num_filter2,
                               5,
                               padding=self.num_padding)
        # feature map size is 7*7 by pooling
        self.fc = nn.Linear(self.num_filter2 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_filter2 * 7 * 7)  # reshape Variable
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def FashionMNIST_loaders(args):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, )),
        ])
    else:
        transform_train = transform
    transform_test = transform

    trainset = datasets.FashionMNIST(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=8)

    testset = datasets.FashionMNIST(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=100,
                                             shuffle=False,
                                             num_workers=8)

    return trainloader, testloader
