# Unsupervised Non contrastive learning

In the last years Unsupervised learning made huge progress, performing really well in 
Like supervised models.

Unsupervised learning can be partitioned in two different families: contrastive and non contrastive learning. The former tries to learn an invariant (invariant to linear transforms and non linear one )image embeddings by comparing negative and positive samples.In this way the model learns useful features for downstream tasks. Now an important question is why we need to compare negative samples and positive samples in the first place. Now imagine training a siamese network (like most of JEPA) without a contrastive term loss. For example we could use as loss the MSE . When minimizing the loss to reach the minimum, the network would output constant features vector so the loss equals 0. Here comes in our help contrastive loss , like the hinge Loss, triplet loss .
But there Is a catch,even though contrastive losses allow the network to learn more useful embeddings, they require more memory(bigger batches) because you need to do negative sampling for the negative pairs. Some circumvent this problem by using memory banks, even then this requires using external memory. So to solve this problem Non-contrastive methods were developed.In this post we’ll cover BYOL, Barlow Twins and VicReg.In theory non-contrastive methods doesn’t need negative mining , memory banks and really big batch sizes (even though big batch sizes helps improving the performance).

## BYOL

![byol images](https://github.com/markpesic/markpesic.github.io/blob/master/images/byol.png)

BYOL (Bootstrap Your Own Latent) uses 2 asymmetrical neural nets , one is called online network and the other is called target network . The Online network has a predictor on top of the projector module, furthermore the target network doesn’t get updated through backprop, but uses EMA (Exponential moving average). The loss is a standard mean sqared error on normalized output embeddings. The images before getting passed into the network gets augmeted in 2 different views. Now it's unclear why this algorithm should work in the first place becuase it seems like , there’s nothing that stops the optimization process to let the 2 networks output constant embeddings so the loss is minimized, but this is not the case. In fact BYOL can learn really useful feature embeddings. In [2] [3] was observed that eliminating the stop-gradient or the predicator component leads to collapse representation. The target network gets updated by
$$\xi \leftarrow \tau\xi + (1 - \tau)\theta \qquad(1)$$

- $\mathcal{T}$ data augmentation
- $\mathcal{T}'$ data augmentation 
- $\mathcal{F}_\theta$ the online network (the backend, a resnet architecture)
- $\mathcal{F}_\xi$ the target network (the backend, a resnet architecture)
- $\mathcal{g}_\theta$ the projector of the online network
- $\mathcal{g}_\xi$ the projector of the target network
- $\mathcal{h}_\theta$ the predictor

First of all given two different data augmentation distribution $\mathcal{T}$ $\mathcal{T}'$ (with the same trasformations). The input image **x** is passed through the data augmentations.
Then respectively the images get through the two networks :

$$\mathcal{z}_ \theta \leftarrow \mathcal{g}_ \theta(\mathcal{F}_ \theta(\mathcal{T}(x))) \qquad (2)$$

$$\mathcal{z}_ \xi \leftarrow \mathcal{g}_ \xi(\mathcal{F}_ \xi(\mathcal{T}'(x))) \qquad (3)$$

$$\mathcal{q}_ \theta \leftarrow \mathcal{h}_ \theta(z_\theta) \qquad (4)$$s

 In [2] The optimization process is hypothesized to be an EM algorithm Expectation-Maximization:

$$ \mathcal{L}(\theta, \eta) = \mathbb{E}_x, _\mathcal{T}[\lVert\mathcal{F} _\theta(\mathcal{T}(x) - \eta_x)\rVert ^2 _2] \qquad (1) $$

$$ \min_{\theta,\eta} \mathcal{L}(\theta, \eta) \qquad (2) $$

In (1) the loss is the average over the distribution of images and transformations for the cosine similarity loss between the projector of the online network and $ \eta $ (Notice that in (1) is not included the predictor). (2) is the objective that gets minimized respect to $\theta$ and $\eta$. So the optimization of the objective gets split in two subproblems :
$$\theta^t = arg\min_{\theta} \mathcal{L}(\theta, \eta^t-1) \qquad (3)$$
$$\eta^t = arg\min_{\eta} \mathcal{L}(\theta^t, \eta) \qquad (4)$$

The first subproblem (3) is a solved applying the stopgradient to $\eta^t-1$ 


