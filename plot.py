# 1 - without augment alpha=1.0
# 2 - with    augment alpha=1.0
# 3 - without augment alpha=0.4
# 4 - with    augment alpha=0.4

import matplotlib.pyplot as plt
import numpy as np
import unidecode
import csv
import re


def slugify(text):
    text = unidecode.unidecode(text).lower()
    text = re.sub(r"[\W_]+", "-", text)
    if text[-1] == "-":
        return text[0:-1]
    return text


class figure:
    def __init__(self, name=None, prefix=None):
        self.name = name
        self.prefix = prefix

    def __enter__(self):
        print("--- FIGURE ---")
        print(f"`{self.name}`")
        plt.cla()
        plt.title(self.name)

    def __exit__(self, x, y, z):
        print("--- SAVE ---")
        figure_prefix = "figure-"
        if self.prefix is not None:
            figure_prefix += f"{str(self.prefix)}-"
        fig.savefig(f"figures/{figure_prefix}{slugify(self.name)}.pdf")


# (1) better style
plt.style.use(["science", "ieee"])

fig, ax = plt.subplots()
ax.autoscale(tight=True)


def read_file(path="log_EfficientNet_batchboost_1", col=5):
    X, Y = [], []
    with open(f"results/{path}.csv", "r") as csvfile:
        plots = csv.reader(csvfile, delimiter=",")
        next(plots, None)
        for row in plots:
            X.append(int(row[0]))
            Y.append(
                float(row[col].replace(", device='cuda:0'",
                                       "").replace("tensor(",
                                                   "").replace(")", "")))
    return X, Y


def fill_between(X, Y, color="blue", alpha=0.05, factor=1):
    sigma = factor * np.array(Y).std(axis=0)  # ls = '--'
    ax.fill_between(X, Y + sigma, Y - sigma, facecolor=color, alpha=alpha)


### FIGURE (1): underfitting ###

with figure("test accuracy (without augment)", prefix=1):
    x1, y1 = read_file("decay=1e-4/log_EfficientNet_batchboost_1")
    plt.plot(x1, y1, label="boostbatch (alpha=1.0)", color="darkred")

    x1, y1 = read_file("decay=1e-4/log_EfficientNet_batchboost_3")
    plt.plot(x1, y1, label="boostbatch (alpha=0.4)", color="red")

    x2, y2 = read_file("decay=1e-4/log_EfficientNet_mixup_1")
    plt.plot(x2, y2, label="mixup (alpha=1.0)", color="darkblue")

    x2, y2 = read_file("decay=1e-4/log_EfficientNet_mixup_3")
    plt.plot(x2, y2, label="mixup (alpha=0.4)", color="blue")

    x3, y3 = read_file("decay=1e-4/log_EfficientNet_baseline_13")
    plt.plot(x3, y3, label="baseline", color="black")

    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

with figure("loss train (without augment)", prefix=1):
    x1a, y1a = read_file("decay=1e-4/log_EfficientNet_batchboost_1", col=1)
    plt.plot(x1a, y1a, label="boostbatch (alpha=1.0)", color="darkred")

    x1b, y1b = read_file("decay=1e-4/log_EfficientNet_batchboost_3", col=1)
    plt.plot(x1b, y1b, label="boostbatch (alpha=0.4)", color="red")

    fill_between(x1a,
                 np.mean([y1a, y1b], axis=0),
                 color="red",
                 factor=1,
                 alpha=0.1)

    x2a, y2a = read_file("decay=1e-4/log_EfficientNet_mixup_1", col=1)
    plt.plot(x2a, y2a, label="mixup (alpha=1.0)", color="darkblue")

    x2b, y2b = read_file("decay=1e-4/log_EfficientNet_mixup_3", col=1)
    plt.plot(x2b, y2b, label="mixup (alpha=0.4)", color="blue")

    fill_between(x2a,
                 np.mean([y2a, y2b], axis=0),
                 color="blue",
                 factor=1,
                 alpha=0.1)

    x3, y3 = read_file("decay=1e-4/log_EfficientNet_baseline_13", col=1)
    plt.plot(x3, y3, label="baseline", color="black")

    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

### FIGURE (2): overfitting (compirason to mixup) ###

with figure("test accuracy (with augment)", prefix=2):
    x1a, y1a = read_file("decay=1e-5/log_EfficientNet_batchboost_2")
    plt.plot(x1a, y1a, label="boostbatch (alpha=1.0)", color="darkred")

    x1b, y1b = read_file("decay=1e-5/log_EfficientNet_batchboost_4")
    plt.plot(x1b, y1b, label="boostbatch (alpha=0.4)", color="red")

    fill_between(x1a,
                 np.mean([y1a, y1b], axis=0),
                 color="red",
                 factor=0.5,
                 alpha=0.1)

    x2a, y2a = read_file("decay=1e-5/log_EfficientNet_mixup_2")
    plt.plot(x2a, y2a, label="mixup (alpha=1.0)", color="darkblue")

    x2b, y2b = read_file("decay=1e-5/log_EfficientNet_mixup_4")
    plt.plot(x2b, y2b, label="mixup (alpha=0.4)", color="blue")

    fill_between(x2a,
                 np.mean([y2a, y2b], axis=0),
                 color="blue",
                 factor=0.5,
                 alpha=0.1)

    # x3, y3 = read_file("decay=1e-5/log_EfficientNet_baseline_24")
    # plt.plot(x3, y3, label="baseline", color="black")

    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

with figure("train accuracy (with augment)", prefix=2):
    x1, y1 = read_file("decay=1e-5/log_EfficientNet_batchboost_2", col=3)
    plt.plot(x1, y1, label="boostbatch (alpha=1.0)", color="darkred")

    x1, y1 = read_file("decay=1e-5/log_EfficientNet_batchboost_4", col=3)
    plt.plot(x1, y1, label="boostbatch (alpha=0.4)", color="red")

    x2, y2 = read_file("decay=1e-5/log_EfficientNet_mixup_2", col=3)
    plt.plot(x2, y2, label="mixup (alpha=1.0)", color="darkblue")

    x2, y2 = read_file("decay=1e-5/log_EfficientNet_mixup_4", col=3)
    plt.plot(x2, y2, label="mixup (alpha=0.4)", color="blue")

    # x3, y3 = read_file("decay=1e-5/log_EfficientNet_baseline_24", col=3)
    # plt.plot(x3, y3, label="baseline", color="black")

    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
