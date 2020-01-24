import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class BatchBoost:
    """
    batchboost: regularization for stabilizing training 
                with resistance to underfitting & overfitting
    Maciej A. Czyzewski
    https://arxiv.org/abs/2001.07627
    """

    def __init__(
        self,
        alpha=1.0,
        window_normal=0,
        window_boost=10,
        factor=1 / 3,
        use_cuda=False,
        debug=False,
    ):
        self.alpha = alpha
        self.window_normal = window_normal
        self.window_boost = window_boost
        self.factor = factor
        self.use_cuda = use_cuda
        self.debug = debug
        self.clear()

        if self.debug:
            print(
                f"[BatchBoost] alpha={alpha} ratio={factor} \
window_normal={window_normal} window_boost={window_boost}"
            )

    def clear(self):
        if self.debug:
            print(f"[BatchBoost] resetting")
        self.mixup_lambda = 1
        self.inputs = None
        self.y1 = self.y2 = None
        self.iter_normal = self.window_normal
        self.iter_boost = self.window_boost

    @staticmethod
    def mixup(x, y, index_left, index_right, mixup_lambda=1.0):
        """Returns mixed inputs, pairs of targets, and lambda
        https://arxiv.org/abs/1710.09412"""
        mixed_x = (
            mixup_lambda * x[index_left, :]
            + (1 - mixup_lambda) * x[index_right, :]
        )
        # mixed_y = (mixup_lambda * y[index_left, :] +
        #           (1 - mixup_lambda) * y[index_right, :])
        # return mixed_x, mixed_y, mixup_lambda
        y1, y2 = y[index_left], y[index_right]
        return mixed_x, y1, y2

    @staticmethod
    def fn_error(outputs, targets):
        logsoftmax = nn.LogSoftmax(dim=1)
        return torch.sum(-outputs * logsoftmax(targets), dim=1)

    @staticmethod
    def fn_linearize(x, num_classes=10):
        _x = torch.zeros(x.size(0), num_classes)
        _x[range(x.size(0)), x] = 1
        return _x

    @staticmethod
    def fn_unlinearize(x):
        _, _x = torch.max(x, 1)
        return _x

    def criterion(self, criterion, outputs):
        _y1 = BatchBoost.fn_unlinearize(self.y1)
        _y2 = BatchBoost.fn_unlinearize(self.y2)
        return self.mixup_lambda * criterion(outputs, _y1) + (
            1 - self.mixup_lambda
        ) * criterion(outputs, _y2)

    def correct(self, predicted):
        _y1 = BatchBoost.fn_unlinearize(self.y1)
        _y2 = BatchBoost.fn_unlinearize(self.y2)
        return (
            self.mixup_lambda * predicted.eq(_y1).cpu().sum().float()
            + (1 - self.mixup_lambda) * predicted.eq(_y2).cpu().sum().float()
        )

    def pairing(self, errvec):
        batch_size = errvec.size()[0]
        _, index = torch.sort(errvec, dim=0, descending=True)
        return (
            index[0 : int(batch_size * self.factor)],
            reversed(index[batch_size - int(batch_size * self.factor) :]),
        )

    def mixing(self, criterion, outputs):
        if self.iter_boost + self.iter_normal == 0:
            self.iter_normal = self.window_normal
            self.iter_boost = self.window_boost
        if self.iter_boost > 0:
            if self.debug:
                print("[BatchBoost]: half-batch + feed-batch")
            errvec = BatchBoost.fn_error(outputs, self.targets)
            index_left, index_right = self.pairing(errvec)

            if self.alpha > 0:
                self.mixup_lambda = np.random.beta(self.alpha, self.alpha)
            else:
                self.mixup_lambda = 1

            self.inputs, self.y1, self.y2 = BatchBoost.mixup(
                self.inputs,
                y=self.targets,
                index_left=index_right,
                index_right=index_left,
                mixup_lambda=self.mixup_lambda,
            )
            self.iter_boost -= 1
        elif self.iter_normal > 0:
            if self.debug:
                print("[BatchBoost] normal batch")
            batch_size = self.inputs.size(0)
            self.inputs = self.inputs[int(batch_size * self.factor) :]
            self.y1 = self.y1[int(batch_size * self.factor) :]
            self.y2 = self.y2[int(batch_size * self.factor) :]
            self.mixup_lambda = 1
            self.iter_normal -= 1

    def feed(self, new_inputs, _new_targets):
        new_targets = Variable(BatchBoost.fn_linearize(_new_targets))
        if self.use_cuda:
            new_targets = new_targets.cuda()
        # no mixing (first iteration)
        if self.inputs is None:
            self.inputs = Variable(new_inputs)
            self.y1 = new_targets
            self.y2 = new_targets
            return False
        # concat
        self.inputs = torch.cat([self.inputs, new_inputs], dim=0)
        self.y1 = torch.cat([self.y1, new_targets], dim=0)
        self.y2 = torch.cat([self.y2, new_targets], dim=0)
        # virtual targets
        self.targets = (
            self.mixup_lambda * self.y1 + (1 - self.mixup_lambda) * self.y2
        )
        return True
