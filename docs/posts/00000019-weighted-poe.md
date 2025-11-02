---
categories: [Qihang's Research]
date: 2025-10-15
draft: false
comments: true
links:
readtime: 0
slug: weighted-product-of-experts
authors:
  - <qihang>
---
# Test-Time Steering for Lossless Text Compression via Weighted Product of Experts

When I was a child, I was always wondering if I use the compressor to compress a file over and over again, will the file get smaller and smaller until it vanishes? Of course the result will be no, if I compress the compressed data with the same compressor again, I will get a file with exactly the same size.

Today I have known it's because of the fundamental limits of lossless compression established by information theory. But how about using multiple compressors together? If we combine multiple compressors simultaneously, each of the compressor reduces part of redundancy of data? And how can we design such a way to combine different compressors? 

This is the question that my recent work "Test-Time Steering for Lossless Text Compression via Weighted Product of Experts" aims to answer.

<!-- more -->
## Table of Contents
[TOC]

## LLMs can serve as a powerful lossless compressor
The statement of "Generation is equalivant to Compression" has been widely spreaded in the Machine Learing community. The relationship between they two has been estibalished for a very long time in Information Theory.

If we use a distribution $p_{\theta}$ to compress data sampled from a true distribution $p_{data}$, one can create a lossless compressor with an expected codelength of $L$, which satisfies the following bounds: 

$$
H(p_{data}, p_{\theta}) \leq L \leq H(p_{data}, p_{\theta}) + 2
$$

where $H(p_{data}, p_{\theta})$ is the cross-entropy between the true distribution and the model distribution.

Also what is very well-known is that the training target of large language models (LLMs) is mininize the cross-entropy of between the dataset and the distrubution encoded by the LLM. Therefore, the training target of LLMs is exactly the same as the objective of building a good lossless compressor.

In the work "Language Modelling is Compression"[@hinton1985boltzmann], it has been shown that LLMs can achieve very good compression ratios on text data, better than traditional compressors like `zip` and `gzip`. This is because LLMs can capture the complex dependencies in natural language data, which traditional compressors cannot.

## How to convert an auto-regressive model into a lossless compressor?
The method of using an existing distribution compress data from another distribution is called source coding. The most well-known source coding algorithm is Haffman coding. 

With respect to computational efficiency, Arithmetic coding fits more for auto-regressive models like LLMs. Since Arithmetic coding also compresses sequencial data by encoding each token one by one, it can be easily combined with auto-regressive models.

Here is the example of using Arithmetic coding to compress a text sequence with a simple auto-regressive model:

## Combining multiple compressors via Weighted Product of Experts
Even before the era of LLMs, people have been trying to use neural network like transformer to training and build better compressors. However, the generalization ability of these models are limited by the scale of training data. While univerial compressors like `gzip` can work well on a wide range of data, even they have never seen those data.

Thus it's natural to think that if we can combine the univerial compressors with the neural-based compressors, so that we can achieve better compression ratios on a wide range of data. It can help neural-based compressors to generalize better on potentially unseen data. In our experiments we can see that the combination of universal compressors is not only helpful for those transformers trained on small datasets, but also helpful for large LLMs like GPT-2 and Llamma3. Meanwhil only very small computation overhead is introduced.

### ⭐️ Weighted Product of Experts

We proposed a framework called Weighted Product of Experts (wPoE) to combine multiple distribution together so that we can guarentte the ensemble model is always not worse than the best individual with respective of the cross entropy with real data distribution. 

The idea is to use a weighted product of the probability distributions of multiple compressors to form a new distribution, here is how we define the distribution of wPoE:

$$
p_{\boldsymbol{\theta}, \boldsymbol{\alpha}}(X_n|  X_{<n}) =  \frac{1}{Z(\boldsymbol{\theta}, \boldsymbol{\alpha},n)}\displaystyle\prod_{k = 1}^K p_{\theta_k}(X_n  |  X_{<n})^{\alpha_k},
$$

where:

+ the weights are $\boldsymbol{\alpha} = \{\alpha_1,...,\alpha_K\}$, $\alpha_k \in[0,1]$, $\displaystyle\sum_{k = 1}^K \alpha_k = 1$, 
+ the parameters of experts are $\boldsymbol{\theta} = \{\theta_1,...,\theta_K\}$,
+ the normalization constant is $Z(\boldsymbol{\theta}, \boldsymbol{\alpha}, n) = \displaystyle\sum_{a \in \mathcal{A}}\prod_{k = 1}^K p_{\theta_k}(X_n = a |  X_{<n})^{\alpha_k}$.

And we can propose the following **proposition**:

$$
\displaystyle\inf_{\boldsymbol{\alpha}}H(p_{\text{data}},p_{\boldsymbol{\theta}, \boldsymbol{\alpha}} ) \leq \displaystyle\min_{k \in \{1,...K\}} H(p_{\text{data}}, p_{\theta_k}),
$$

to say that we can always find a set of weights $\boldsymbol{\alpha}$ such that the cross-entropy between the data distribution and the wPoE distribution is not worse than the best individual expert

Although Jeffory has been proposed product of experts for a very long time and there are also something like generalized product of experts. They usually train those experts jointly, we are performing under a setting that each model's distribution can't be changed.

<details>
<summary>Proposition Proof</summary>

### Proof of Proposition

Let $p_{\theta_1},p_{\theta_2},...,p_{\theta_K}$ be $K$ autoregressive models used to compress a sequence $x_{<n+1} = \{x_1, x_2, \dots, x_n\}$, where $X_{<n} \sim p_{\text{data}}$. Each $x_i$ takes values from the dictionary $\mathcal{A} = \{ a_1, \dots, a_D \}$. For an autoregressive model $p_{\theta_k}$, the following equation reveals the relationship between the joint distribution of $X_{<n}$ and the conditional distribution of $X_n$:

$$
p_{\theta_k}(X_{<n+1}) = \displaystyle\prod_{i = 1}^{n} p_{\theta_k}(X_i \mid X_{<i}).
$$

Therefore, the cross-entropy between $p_{\text{data}}$ and a certain model $p_{\theta_k}$ can be expanded as follows:

$$
H(p_{\text{data}},p_{\theta_k}) = \underset{p_{\text{data}}}{\mathbb{E}} \sum_{i = 1}^{n} -\log p_{\theta_k}(X_i \mid X_{<i}).
$$

Our weighted product of experts (wPoE) model is given by:

$$
p_{\boldsymbol{\theta}, \boldsymbol{\alpha}}(X_n\mid  X_{<n}) \;=\;  \frac{1}{Z(\boldsymbol{\theta}, \boldsymbol{\alpha},n)}
\displaystyle\prod_{k = 1}^K p_{\theta_k}(X_n  \mid  X_{<n})^{\alpha_k},
$$

where the weights are $\boldsymbol{\alpha} = \{\alpha_1,...,\alpha_K\}$, $\alpha_k \in[0,1]$, $\sum_{k = 1}^K \alpha_k = 1$, the parameters of experts are $\boldsymbol{\theta} = \{\theta_1,...,\theta_K\}$, and the normalization constant is

$$
Z(\boldsymbol{\theta}, \boldsymbol{\alpha}, n) \;=\; \sum_{a \in \mathcal{A}}\prod_{k = 1}^K p_{\theta_k}(X_n = a \mid  X_{<n})^{\alpha_k}.
$$

Here we can derive:

$$
\begin{aligned}
H(p_{\text{data}}, p_{\boldsymbol{\theta}, \boldsymbol{\alpha}}) 
&=  \sum_{k = 1}^K \alpha_k H(p_{\text{data}},p_{\theta_k})
+\underset{p_{\text{data}}}{\mathbb{E}} \displaystyle\sum_{i=1}^{n} \log \left[Z(\boldsymbol{\theta}, \boldsymbol{\alpha},i)\right].
\end{aligned}
$$

To complete the proof, we introduce the following technical lemma for bounding $Z(\boldsymbol{\theta}, \boldsymbol{\alpha},i)$.

#### Lemma 1

Let $p^{(k)} = \bigl(p^{(k)}_1, \ldots, p^{(k)}_D\bigr)$ for $k=1,\dots,K$ be $K$ categorical distributions, so $\sum_{j=1}^D p^{(k)}_j = 1$ for each $k$. Let $\alpha_1,\dots,\alpha_K \ge 0$ satisfy $\sum_{k=1}^K \alpha_k = 1.$ Then

$$
\sum_{j=1}^D 
\prod_{k=1}^K \bigl(p^{(k)}_j\bigr)^{\alpha_k}
\;\;\le\;\; 1,
$$

with equality if and only if $p^{(1)} = p^{(2)} = \cdots = p^{(K)}$ or exactly one $\alpha_k=1$ and the rest are zero.

From `Cauchy–Schwarz inequality`, it can be concluded that:

$$
Z(\boldsymbol{\theta}, \boldsymbol{\alpha},i) \leq  1, \quad \forall \boldsymbol{\theta}, \boldsymbol{\alpha},i.
$$

Equality holds if and only if each distribution $p_{\theta_k}(X_i \mid X_{<i})$ is the same, or $\alpha_k = 1$ and others are 0. Thus we can conclude that:

$$
\begin{aligned}
\inf_{\boldsymbol{\alpha}}H(p_{\text{data}},p_{\boldsymbol{\theta}, \boldsymbol{\alpha}} )
&\leq \min_{k \in \{1,\dots,K\}} H(p_{\text{data}}, p_{\theta_k})
+ \underset{p_{\text{data}}}{\mathbb{E}} \displaystyle\sum_{i=1}^{n} \log \left[Z(\boldsymbol{\theta}, \boldsymbol{\alpha},i)\right] \\
\inf_{\boldsymbol{\alpha}}H(p_{\text{data}},p_{\boldsymbol{\theta}, \boldsymbol{\alpha}} )
&\leq \min_{k \in \{1,\dots,K\}} H(p_{\text{data}}, p_{\theta_k}).
\end{aligned}
$$

</details>

## Even To Combine Simple Statistical Method Can help LLMs Compress Better
To make LLMs performe better on potentially unseen data, we combine LLMs with simple statistical methods like Naive Bayes With Laplace smoothing, with the help of wPoE:

The distribution of Naive Bayes with Laplace smoothing is defined as:

$$
q(X_n = a \mid X_{<n}) := \frac{\sum_{k=1}^{n-1} \mathbb{I}(X_k = a) + 1}{n - 1 + D},
$$

where $\mathbb{I}(\cdot)$ denotes the indicator function and $D$ is the vocabulary size. 

We then combine the Naive Bayes with Laplace smoothing $q$ with a pretrained language model $p_{\theta}$ using the weighted product of experts as follows:

$$
\pi_{\alpha}\!(X_n \vert X_{<n}) \!= \!
    \frac{q(X_n \vert X_{<n})^{\alpha} p_{\theta}(X_n \vert X_{<n})^{1 - \alpha}}
    {Z(\theta, \alpha, n)},
$$

Where $\alpha$ is a scalar as we have only two experts.
Moreover, since we do not need to fine-tune the pretrained model $p_{\theta}$, i.e., $\theta$ is frozen, we omit the dependency of $\theta$ in the wPoE model $\pi$.

### Experiments

<!-- main table -->

## Multi-Experts

### Experiments

## Future Applications

wPoE can not only combine different autoregressive models, it can also be used to ensemble any models that share the same dictionary, i.e., any category distributions that have the same alphabet.

Here is a provement for general category distributions:

<details>

</details>





