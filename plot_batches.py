import debug
import torch
import argparse
import numpy as np
from torch.autograd import Variable
"""
inkscape -D -z --file=$PWD/figures/figure-abstract.svg \
    --export-pdf=$PWD/figures/figure-abstract.pdf
inkscape -D -z --file=$PWD/figures/figure-feeding.svg \
    --export-pdf=$PWD/figures/figure-feeding.pdf
"""

parser = argparse.ArgumentParser(description="BoostBatch")
parser.add_argument("--batch-size", default=6, type=int, help="batch size")
parser.add_argument(
    "--alpha",
    default=1.0,
    type=float,
    help="mixup interpolation coefficient (default: 1)",
)
parser.add_argument(
    "--no-augment",
    dest="augment",
    action="store_false",
    help="use standard augmentation (default: True)",
)
args = parser.parse_args()

trainloader, testloader = debug.FashionMNIST_loaders(args)


def mixup_data(x, y, index_left, index_right, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    lam = 0.5

    print(lam)
    print(" LEFT", index_left)
    print("RIGHT", index_right)
    mixed_x = lam * x[index_left, :] + (1 - lam) * x[index_right, :]
    y_a, y_b = y[index_left], y[index_right]
    return mixed_x, y_a, y_b, lam


def batchboost_data(x, y_1, y_2, outputs, alpha=1.0):
    """Batchboost: reduction"""

    batch_size = x.size()[0]

    # (1) normalize labels
    y = (y_1 + y_2) / 2

    # (2) calculate error
    err = torch.tensor(np.random.rand(batch_size))

    # (3) sort by error
    _, index = torch.sort(err, dim=0, descending=True)

    # (4) mixup using pairs (worst with best)
    mixed_x, y_a, y_b, lam = mixup_data(
        x,
        y,
        index_left=index[0:batch_size // 2],
        index_right=index[batch_size // 2:],
        alpha=alpha,
    )

    return mixed_x, y_a, y_b, lam


lam = 1
inputs = None
targets_a = None
targets_b = None

from torchvision.utils import save_image


def pseudotrain_batchboost():
    global inputs, targets_a, targets_b, lam
    print("BATCHBOOST")

    for batch_idx, (new_inputs, new_targets) in enumerate(trainloader):
        if batch_idx == 5:
            break
        print(batch_idx)

        for i in range(new_inputs.size(0)):
            save_image(new_inputs[i],
                       f"figures/batches/img_{batch_idx}_new_{i+6}.png")

        # -----> (a) batch merge
        if inputs is None:
            inputs = new_inputs
            targets_a = new_targets
            targets_b = new_targets
            continue

        for i in range(inputs.size(0)):
            save_image(inputs[i],
                       f"figures/batches/img_{batch_idx}_old_{i}.png")

        inputs = torch.cat([inputs, new_inputs], dim=0)
        targets_a = torch.cat([targets_a, new_targets], dim=0)
        targets_b = torch.cat([targets_b, new_targets], dim=0)

        # -----> (c) batch reduce
        inputs, targets_a, targets_b, lam = batchboost_data(
            inputs, targets_a, targets_b, [], args.alpha)
        inputs, targets_a, targets_b = map(Variable,
                                           (inputs, targets_a, targets_b))


pseudotrain_batchboost()
