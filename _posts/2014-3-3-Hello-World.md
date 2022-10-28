# Unsupervised Non contrastive learning in images

In the last years Unsupervised learning made huge progress, performing really well in 
Like supervised models.

Unsupervised learning can be partitioned in two different families: contrastive and non contrastive learning. The former tries to learn an invariant (invariant to linear transforms and non linear one )image embeddings by comparing negative and positive samples.In this way the model learns useful features for downstream tasks. Now an important question is why we need to compare negative samples and positive samples in the first place. Now imagine training a siamese network (like most of JEPA) without a contrastive term loss. For example we could use as loss the MSE . When minimizing the loss to reach the minimum, the network would output constant features vector so the loss equals 0. Here comes in our help contrastive loss , like the hinge Loss, triplet loss .
But there is a catch,even though contrastive losses allow the network to learn more useful embeddings, they require more memory(bigger batches) because you need to do negative sampling for the negative pairs. Some circumvent this problem by using memory banks, even then this requires using external memory. So to solve this problem Non-contrastive methods were developed.In this post we’ll cover BYOL, Barlow Twins and VicReg.In theory non-contrastive methods doesn’t need negative mining , memory banks and really big batch sizes (even though big batch sizes helps improving the performance).

## Augmentations

In all of this papers the input before getting fed through the network/s, it gets distorted in some kind of a way. Usually are used a series of augmentation.
- Random Resized crop size
- Random color jittering
- Random Gaussian blur
- Random horizontal flip
- Random greyscale conversion
etc...
In all the methods below we used the same image augmentation as in [SIMCLR].
## BYOL

![byol images](https://github.com/markpesic/markpesic.github.io/blob/master/images/byol.png)

BYOL (Bootstrap Your Own Latent) uses 2 asymmetrical neural nets , one is called online network and the other is called target network . The Online network has a predictor on top of the projector module, furthermore the target network doesn’t get updated through backprop, but uses EMA (Exponential moving average). The loss is a standard mean sqared error on normalized output embeddings. The images before getting passed into the network gets augmeted in 2 different views. Now it's unclear why this algorithm should work in the first place becuase it seems like , there’s nothing that stops the optimization process to let the 2 networks output constant embeddings so the loss is minimized, but this is not the case. In fact BYOL can learn really useful feature embeddings. In [2] [3] was observed that eliminating the stop-gradient or the predicator component leads to collapse representation.

- $$\mathcal{T}$$ data augmentation
- $$\mathcal{T}'$$ data augmentation 
- $$\mathcal{F}_\theta$$ the online network (the backend, a resnet architecture)
- $$\mathcal{F}_\xi$$ the target network (the backend, a resnet architecture)
- $$\mathcal{g}_\theta$$ the projector of the online network
- $$\mathcal{g}_\xi$$ the projector of the target network
- $$\mathcal{h}_\theta$$ the predictor

First of all given two different data augmentation distribution $$\mathcal{T}$$ $$\mathcal{T}'$$ (with the same trasformations). The input image **x** is passed through the data augmentations.
Then respectively the images get through the two networks :

$$\mathcal{z}_ \theta \leftarrow \mathcal{g}_ \theta(\mathcal{F}_ \theta(\mathcal{T}(x))) \qquad (2)$$

$$\mathcal{z}_ \xi \leftarrow \mathcal{g}_ \xi(\mathcal{F}_ \xi(\mathcal{T}'(x))) \qquad (3)$$

$$\mathcal{q}_ \theta \leftarrow \mathcal{h}_ \theta(z_\theta) \qquad (4)$$

The output of the predictor and the output of the target projection gets normalized:

$$\hat{\mathcal{q}}_ \theta \leftarrow \dfrac{\mathcal{q}_ \theta}{\lVert \mathcal{q}_ \theta \rVert} \qquad (5)$$

$$\hat{\mathcal{z}}_ \xi \leftarrow \dfrac{\mathcal{z}_ \xi }{\lVert \mathcal{z}_ \xi \rVert} \qquad (6)$$

The $\mathcal{L}_ {BYOL}$ is the MSE loss between two normalized vectors. This loss is applied twice, so that the loss is symmetrical.
Essentialy the 2 networks together outputs 4 vectors:
- $$\mathcal{q}_ \theta$$ that comes from $$\mathcal{T}(x)$$
- $$\mathcal{q}'_ \theta$$ that comes from $$\mathcal{T}'(x)$$
- $$\mathcal{z}_ \xi$$ that comes from $$\mathcal{T}'(x)$$
- $$\mathcal{z}'_ \xi$$ that comes from $$\mathcal{T}(x)$$. 

$$\mathcal{L}_ {BYOL} = \lVert \hat{\mathcal{q}}_ \theta - \hat{\mathcal{z}}_ \xi \rVert ^2 _2 \qquad (7)$$

$$\mathcal{L}_ {BYOL'} = \lVert \hat{\mathcal{q}}'_ \theta - \hat{\mathcal{z}}'_ \xi \rVert ^2 _2 \qquad (8)$$

$$\mathcal{L}_ {tot} = \mathcal{L}_ {BYOL} + \mathcal{L}_ {BYOL'} \qquad(9)$$

Once we calculated the loss we can updated the online network and the target network, first of all we update the $\theta$ online network through an optimizer and by detaching the gradient for the target network.

$$\theta \leftarrow optimizer(\theta, stopgrad(\xi), learning_rate)$$

After the online network gets updated, it is the turn of the target network.
$$\xi \leftarrow \tau\xi + (1 - \tau)\theta \qquad(1)$$

The $\tau$ is a hyperparameter that has a vale between 0 and 1. SGD is used as the optimizer with weight decay and LARS applied to the online network.

In the byol paper was also observed that there is also another way for optimizing the target network, that is simply multiplying the updated parameters of the online network
by a scalar $$\lambda$$  

$$\xi \leftarrow \lambda * \theta$$

Another paper that takes from [1] is SimSiam [2] that tries to explain the optimization process of the stopgradient and the prediction head. There are some differences in the architecture of SimSiam respect byols' one. First of all there is only one network , in fact in the paper the empirical data show that having 2 networks, where one gets updated through EMA is not compulsary for the system to work. The setup is as follows: only one network with a backend layer, projection layer and a predictor layer. As before the input **x** gets distorted before being processed by the network. Now both the views gets through the network one of the views gets through the predictor head , and the other not. The one that passes only through the backend layer and projection layer , when calculating the loss will be "detached" (the gradient will not be backpropagated).The optimization process is hypothesized to be an EM algorithm Expectation-Maximization:

$$ \mathcal{L}(\theta, \eta) = \mathbb{E}_x, _\mathcal{T}[\lVert\mathcal{F} _\theta(\mathcal{T}(x) - \eta_x)\rVert ^2 _2] \qquad (1) $$

$$ \min_{\theta,\eta} \mathcal{L}(\theta, \eta) \qquad (2) $$

In (1) the loss is the average over the distribution of images and transformations for the cosine similarity loss between the projector of the online network and $ \eta $ (Notice that in (1) is not included the predictor). (2) is the objective that gets minimized respect to $\theta$ and $\eta$. So the optimization of the objective gets split in two subproblems :
$$\theta^t = arg\min_{\theta} \mathcal{L}(\theta, \eta^t-1) \qquad (3)$$
$$\eta^t = arg\min_{\eta} \mathcal{L}(\theta^t, \eta) \qquad (4)$$

The first subproblem (3) is a solved applying the stopgradient to $\eta^t-1$ 


