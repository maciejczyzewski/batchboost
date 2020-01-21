import torch
import numpy as np
from torch.autograd import Variable


class BatchBoost:
    def __init__(self, num_classes=10, use_cuda=False):
        self.clear()
        self.use_cuda = False
        self.num_classes = num_classes

    def clear(self):
        self.lam = 1
        self.inputs = None
        self.targets_a = None
        self.targets_b = None

    @staticmethod
    def mixup(x, y, index_left, index_right, alpha=1.0):
        """Returns mixed inputs, pairs of targets, and lambda"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        mixed_x = lam * x[index_left, :] + (1 - lam) * x[index_right, :]
        y_a, y_b = y[index_left], y[index_right]
        return mixed_x, y_a, y_b, lam

    def criterion(self, criterion, pred):
        return self.lam * criterion(pred, self.targets_a) + (
            1 - self.lam) * criterion(pred, self.targets_b)

    def _mixing(self, criterion, x, y_1, y_2, outputs, alpha=1.0):
        """Batchboost: mixing"""

        batch_size = x.size()[0]

        # (1) normalize labels
        y = (y_1 + y_2) / 2

        # (2) calculate error
        y_onehot = torch.zeros(batch_size, self.num_classes)
        y_onehot[range(y_onehot.shape[0]), y] = 1
        # FIXME: am I calculating this correctly?
        #        what about softmax? / one-hot
        if self.use_cuda:
            err = torch.norm(outputs - y_onehot.cuda(), 2, dim=1).cuda()
        else:
            err = torch.norm(outputs - y_onehot, 2, dim=1)

        # (3) sort by error
        _, index = torch.sort(err, dim=0, descending=True)

        # (4) mixup using pairs (worst with best)
        mixed_x, y_a, y_b, lam = BatchBoost.mixup(
            x,
            y,
            index_left=index[0:batch_size // 2],
            index_right=index[batch_size // 2:],
            alpha=alpha,
        )

        return mixed_x, y_a, y_b, lam

    def mixing(self, criterion, outputs, alpha=1.0):
        self.inputs, self.targets_a, self.targets_b, self.lam = self._mixing(
            criterion,
            self.inputs,
            self.targets_a,
            self.targets_b,
            outputs,
            alpha,
        )
        self.inputs, self.targets_a, self.targets_b = map(
            Variable, (self.inputs, self.targets_a, self.targets_b))

    def feed(self, new_inputs, new_targets):
        if self.inputs is None:
            self.inputs = new_inputs
            self.targets_a = new_targets
            self.targets_b = new_targets
            return False
        self.inputs = torch.cat([self.inputs, new_inputs], dim=0)
        self.targets_a = torch.cat([self.targets_a, new_targets], dim=0)
        self.targets_b = torch.cat([self.targets_b, new_targets], dim=0)
        return True
