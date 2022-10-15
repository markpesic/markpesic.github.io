---
layout: post
title: Unsupervised Non contrastive learning
---

# Unsupervised Non contrastive learning

In the last years Unsupervised learning made huge progress, performing really well in 
Like supervised models.

Unsupervised learning can be partitioned in two different families: contrastive and non contrastive learning. The former tries to learn an invariant (invariant to linear transforms and non linear one )image embeddings by comparing negative and positive samples.In this way the model learns useful features for downstream tasks. Now an important question is why we need to compare negative samples and positive samples in the first place. Now imagine training a siamese network (like most of JEPA) without a contrastive term loss. For example we could use as loss the MSE . When minimizing the loss to reach the minimum, the network would output constant features vector so the loss equals 0. Here comes in our help contrastive loss , like the hinge Loss, triplet loss .
But there Is a catch,even though contrastive losses allow the network to learn more useful embeddings, they require more memory(bigger batches) because you need to do negative sampling for the negative pairs. Some circumvent this problem by using memory banks, even then this requires using external memory. So to solve this problem Non-contrastive methods were developed.In this post we’ll cover BYOL, Barlow Twins and VicReg.In theory non-contrastive methods doesn’t need negative mining , memory banks and really big batch sizes (even though big batch sizes helps improving the performance).

## BYOL

BYOL (Bootstrap Your Own Latent) uses 2 asymmetrical neural nets , one is called online network and the other is called target network. The Online network has a predictor on top of the projector module, furthermore the target network doesn’t get updated through backprop, but uses EMA (Exponential moving average) of the online netwok parameters. The loss is a standard mean sqared error on regularized output embeddings. The images before getting passed into the netowrk gets augmeted in 2 different views. Now it's unclear why this algorithm should work in the first place becuase it seems like , there’s nothing that stops the optimization process to let the 2 networks output constant embeddings so the loss is minimized, but this is not the case. In fact BYOL can learn really useful feature embeddings, In [2] [3] was observed that eliminating the stop-gradient or the predicator component leads to collapse representation. In [2] The optimization process is hypothesized to be an EM algorithm Expectation-Maximization:

```LaTeX
 \documentclass{article}
\begin{document}
 $E=mc^2$
\end{document}
  ```
