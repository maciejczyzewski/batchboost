# Research Brief (Brief intro of the research (50 + words))

Batchboost is a simple technique to accelerate ML model training by adaptively feeding mini-batches with artificial samples which are created by mixing two examples from previous step - in favor of pairing those that produce the difficult one.

# What’s New (What’s new in this research?)

In this research, we state the hypothesis that mixing many images together can
be more effective than just two.  To make it efficient, we propose a new method of
creating mini-batches, where each sample from dataset is propagated with
subsequent iterations with less and less importance until the end of learning
process.

# How It Works (How this research works?)

Batchboost pipeline has three stages:
(a) pairing: method of selecting two samples from previous step.
(b) mixing: method of creating a new artificial example from two selected samples.
(c) feeding: constructing training mini-batch with created examples and new samples from dataset (concat with ratio γ).
Note that sample from dataset propagates with subsequent iterations with less and less importance until the end of training. 

Our baseline implements pairing stage as sorting by sample error, where hardest examples are paired with easiest ones. Mixing stage
merges to samples using mixup, x1+(1−λ)x2. Feeding stage combines new samples with ratio 1:1 using concat.

# Key Insights (What are the main takeaways from this research?)

The results are promising. Batchboost has 0.5-3% better accuracy than the current state-of-the-art mixup regularization on CIFAR-10 (#10 place in https://paperswithcode.com/) & Fashion-MNIST.
(we hope to see our method in action, for example, on Kaggle as trick to improve a bit test accuracy)

# Behind The Scenes (Any interesting ideas or research tips you - would like to share with our AI Community?)

There is a lot to improve in data augmentation and regularization methods.

# Anything else? (Bottlenecks and future trend?)

An interesting topic for further research and discussion are
combination of batchboost and existing methods.
