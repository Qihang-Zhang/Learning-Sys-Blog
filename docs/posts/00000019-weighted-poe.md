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
<p align="center">
  <a href="https://github.com/DSL-Lab/Weighted-Product-of-Experts" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&logo=github" alt="GitHub Repository" />
  </a>

  <a href="https://aclanthology.org/2025.findings-emnlp.110/" target="_blank">
    <img src="https://img.shields.io/badge/EMNLP%202025-Paper-006400?style=for-the-badge&logo=readthedocs&logoColor=white" alt="EMNLP 2025 Paper" />
  </a>
</p>
When I was a child, I always wondered: if I use a compressor to compress a file over and over again, will the file get smaller and smaller until it vanishes? Of course, the answer is no. If we compress the compressed data with the same compressor again, I will get a file with exactly the same size.

Today I understand this is because of the fundamental limits of lossless compression established by information theory. **But what about using multiple compressors together? If we combine multiple compressors simultaneously, can each compressor reduce part of the data's redundancy? And how can we design such a method to combine different compressors?**

This is the question that our work [Test-Time Steering for Lossless Text Compression via Weighted Product of Experts](https://aclanthology.org/2025.findings-emnlp.110/) [@zhang-etal-2025-test] aims to answer. **Compared to the EMNLP paper, this blog post focuses more on the intuition behind our method, presenting it in a way that is easier to understand.**

<!-- more -->
## Table of Contents
[TOC]

## Background

### LLMs can serve as powerful lossless compressors
The statement "Generation is equivalent to Compression" has been widely spread in the Machine Learning community. The relationship between these two has been established for a very long time in Information Theory:

**The target of lossless compression is minimizing cross entropy:**

If we use a distribution $p_{\theta}$ to compress data sampled from a true distribution $p_{data}$, one can create a lossless compressor with an expected codelength of $L$, which satisfies the following bounds: 

$$
H(p_{data}, p_{\theta}) \leq L \leq H(p_{data}, p_{\theta}) + 2
$$

where $H(p_{data}, p_{\theta})$ is the cross entropy between the true distribution and the model distribution.

<br>
**The target of LLMs' pre-training is also minimizing cross entropy:**

It is also well-known that the training objective of large language models (LLMs) pre-training is to minimize the cross entropy between the dataset and the distribution encoded by the LLM. Therefore, the training objective of LLMs' pre-training is exactly the same as the objective of building a good lossless compressor.

In the work [Language Modelling is Compression](https://arxiv.org/abs/2309.10668)[@delétang2024languagemodelingcompression], it has been shown that LLMs can achieve very good compression ratios on text data, better than traditional compressors like `zip` and `gzip`. This is because LLMs can capture the complex dependencies in natural language data, which traditional compressors cannot.

### How to convert an auto-regressive model into a lossless compressor?
The method of using an existing distribution to compress data from another distribution is called source coding. The most well-known source coding algorithm is Huffman coding. 

However, with respect to computational efficiency, Arithmetic Coding is better suited for auto-regressive models like LLMs. Since Arithmetic Coding also compresses sequential data by encoding each token one by one, it can be easily combined with auto-regressive models.

Here is an example of using Arithmetic Coding to compress a text sequence with a simple auto-regressive model:

![wpoe-ac](https://img.qihang-zhang.com/2025/11/776c744e9ef9e4353be5c2ff374209c1.jpg)

## Method: Combine multiple categorical distributions via *Weighted Product of Experts*
Even before the era of LLMs, people have been trying to use neural networks like Transformers to train and build better compressors. However, the generalization ability of these models is limited by the scale of training data. Meanwhile, universal compressors like `gzip` can work well on a wide range of data, even data they have never seen before.

**Thus, it is natural to think that if we can combine universal compressors with neural-based compressors, we can achieve better compression ratios on a wide range of data.** This can help neural-based compressors generalize better to potentially unseen data. In our experiments, we can see that combining with universal compressors is not only helpful for Transformers trained on small datasets, but also helpful for large LLMs like GPT-2 and LLaMA 3. Meanwhile, only very small computational overhead is introduced.

>[!note] **Note on terminology:** Throughout this blog post, we use several terms interchangeably
>
> - **"Expert"** and **"model"** both refer to the individual discrete probability distributions that we combine.
> - **"categorical distribution"** and **"probability distribution"** both refer to the distribution over possible next tokens that each model outputs.

### ⭐️ Weighted Product of Experts

We propose a framework called Weighted Product of Experts (wPoE) to combine multiple distributions (also called **experts** or **models**) together so that we can guarantee the ensemble model is always no worse than the best individual expert with respect to the cross entropy with the real data distribution. 

The idea is to use a weighted product of the probability distributions of multiple experts (compressors) to form a new distribution. Here is how we define the distribution of wPoE:

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

This means we can always find a set of weights $\boldsymbol{\alpha}$ such that the cross entropy between the data distribution and the wPoE distribution is no worse than the best individual expert.

Although Geoffrey Hinton proposed Product of Experts [@PoE] a long time ago, and there are also variants like Generalized Product of Experts[@gPoE], these methods usually train the experts jointly. In contrast, we operate under a setting where each expert's distribution is fixed and cannot be changed (i.e., pretrained models).

<details>
<summary>Proposition Proof</summary>

#### Proof of Proposition

Let $p_{\theta_1},p_{\theta_2},...,p_{\theta_K}$ be $K$ autoregressive models used to compress a sequence $x_{<n+1} = \{x_1, x_2, \dots, x_n\}$, where $X_{<n} \sim p_{\text{data}}$. Each $x_i$ takes values from the dictionary $\mathcal{A} = \{ a_1, \dots, a_D \}$. For an autoregressive model $p_{\theta_k}$, the following equation reveals the relationship between the joint distribution of $X_{<n}$ and the conditional distribution of $X_n$:

$$
p_{\theta_k}(X_{<n+1}) = \displaystyle\prod_{i = 1}^{n} p_{\theta_k}(X_i \mid X_{<i}).
$$

Therefore, the cross entropy between $p_{\text{data}}$ and a certain model $p_{\theta_k}$ can be expanded as follows:

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

##### Lemma 1

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

### ⭐️ The intuition behind the mathematical formulation

In the proof, we can see that the cross entropy of the ensemble model can be reformulated into the following form, and this form provides intuition for why our experiments work. 

$$
\begin{aligned}
H(p_{\text{data}}, p_{\boldsymbol{\theta}, \boldsymbol{\alpha}}) 
&=  \sum_{k = 1}^K \alpha_k H(p_{\text{data}},p_{\theta_k})
+\underset{p_{\text{data}}}{\mathbb{E}} \displaystyle\sum_{i=1}^{n} \log \left[Z(\boldsymbol{\theta}, \boldsymbol{\alpha},i)\right].
\end{aligned},
$$

where:

$$
Z(\boldsymbol{\theta}, \boldsymbol{\alpha}, n) \;=\; \sum_{a \in \mathcal{A}}\prod_{k = 1}^K p_{\theta_k}(X_n = a \mid  X_{<n})^{\alpha_k}.
$$

<br>

#### ⭐️ How can we improve the ensemble wPoE model's performance from the perspective of the first term?

The first term of this formulation is the weighted average of the cross entropy between the data distribution and each expert (model), where the weights are the $\alpha$ values we want to learn or optimize using a small amount of data.

**Conclusion:**
> Therefore, we can easily see that:
> 
> 1. The better the experts used in wPoE, the better the ensembled model is likely to be.
> 2. The larger the weight (i.e., the corresponding $\alpha$ value) we assign to the best experts we have, the better the ensembled model is likely to be.

<br>

#### ⭐️ How can we improve the ensemble wPoE model's performance from the perspective of the second term?

The second term is the expectation of the log partition function under the data distribution. The value of this term is always smaller than or equal to `0`.

If we look into the details of the second term, we will find that it has the following excellent properties:

1. **This term is always smaller than or equal to `0` no matter what the data distribution is**, since we didn't make any assumptions about the data distribution in the proof.

2. **This term equals zero if and only if: $p_{\theta_1} = p_{\theta_2} = \cdots = p_{\theta_K}$ or exactly one $\alpha_k=1$ and the rest are `0`**, as proved by the Cauchy–Schwarz Inequality.

3. **The more diverse the experts are, the smaller (more negative) this term is.**
<details>
  <summary>Click to expand the explanation</summary>
  <ol>
    <li>
      For two-expert cases, this term behaves like a distance between two distributions. It becomes <code>0</code> only when the distributions are identical. The more diverse the distributions are, the smaller this term becomes, which helps reduce cross entropy and improves the wPoE ensemble on compression tasks.
    </li>
    <li>For K-expert cases, treat the ensemble of the first K-1 experts as a single model. The term then captures the distance between that ensemble and the Kth expert we are about to combine.</li>
    <li>
      This distance measures diversity differently from the usual KL divergence:
      <ol>
        <li>It is a true distance: symmetric, so swapping the order of the two distributions does not change its value.</li>
        <li>The family of distances is controlled by the weights &alpha;<sub>k</sub>; when one weight is 1 and the others are 0, the distance collapses to 0.</li>
        <li>With a fixed set of experts, the distance is convex in the weights. Starting from &alpha;<sub>k</sub> = 1 and gradually adding more experts shrinks this second term, and the optimization is straightforward because of convexity.</li>
      </ol>
    </li>
  </ol>
</details>

**Conclusion:**
> Therefore, we can see that:
> 
> 1. If the experts are diverse (as defined from the perspective of the second term), the wPoE ensembled model is likely to have better performance.
> 2. If the experts are truly diverse, using non-sparse $\alpha$ weights will provide better performance.

<br>

#### ⭐️ We need to make the best trade-off between the first term and the second term on the target data

It is easy to see that we should try to use experts that are both high-quality and diverse when choosing which experts to combine. *Thus, the best choice is to combine different models with good performance that are also diverse (e.g., trained on different data).*

**Diversity is crucial in wPoE:**

> Sometimes these two objectives conflict with each other. In this case, we only need to use a very small amount of data to find the optimal trade-off. In our experiments, we observe that:
> 
> 1. When we combine another expert with good performance but not very different from the current one, the benefit that wPoE brings is quite small. (For example, when we combine two models of different sizes but trained on the same dataset.)
> 
> 2. When we combine another expert with even mediocre performance but very different from the current one, the benefit that wPoE brings is still significant and stable across various datasets and model combinations.




## Two Experts

### ⭐️ Even Combining Simple Statistical Methods Can Help LLMs Compress Better

To make LLMs perform better on potentially unseen data, we combine LLMs with simple statistical methods like Naive Bayes with Laplace smoothing, using wPoE:

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

where $\alpha$ is a scalar since we have only two experts.
Moreover, since we do not need to fine-tune the pretrained model $p_{\theta}$ (i.e., $\theta$ is frozen), we omit the dependency on $\theta$ in the wPoE model $\pi$.

### Experiment Results

As mentioned in our paper, this combination helps various pretrained models achieve better compression rates across all datasets we collected from different sources.

It is reasonable that as we use larger and larger models, the improvement brought by Naive Bayes becomes smaller, since larger models can already generalize better to potentially unseen data through large-scale training on vast amounts of data.

> However, what is not obvious is that even for very large LLMs like LLaMA 3-8B, the combination with Naive Bayes can still bring non-trivial improvements across various datasets. Considering how small the computational overhead of Naive Bayes is, and how weak Naive Bayes performs when used alone to compress data, this result is remarkable.
> 
> <span style="color:#1b4f9c;">
> ***This indicates that regardless of the performance of each individual model, the ensemble model obtained by wPoE can still benefit from the diversity of different models.***</span> 
> 
> <span style="color:#1b4f9c;"> 
> ***This aligns with our intuition that diversity is key to improving the performance of wPoE.***</span>

>[!note] 
>All experiments are conducted to evaluate the compression rates on five datasets (lower is better).

#### Experiments on pretrained vanilla Transformers

<details>
<summary>Expand to see the full table</summary>
  <table>
    <thead>
      <tr>
        <th>Tokenizer</th>
        <th>Compressor</th>
        <th>math</th>
        <th>code</th>
        <th>shakespeare</th>
        <th>enwik8*</th>
        <th>enwik9*</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Byte Level</strong></td>
        <td>gzip</td>
        <td>43.59%</td>
        <td>36.72%</td>
        <td>52.80%</td>
        <td>49.14%</td>
        <td>48.07%</td>
      </tr>
      <tr>
        <td></td>
        <td>LZMA2</td>
        <td>45.35%</td>
        <td>38.61%</td>
        <td>56.86%</td>
        <td>51.33%</td>
        <td>49.98%</td>
      </tr>
      <tr>
        <td></td>
        <td>Naive Bayes</td>
        <td>68.90%</td>
        <td>64.65%</td>
        <td>64.57%</td>
        <td>66.03%</td>
        <td>67.14%</td>
      </tr>
      <tr>
        <td></td>
        <td>Transformer 200K</td>
        <td>56.25%</td>
        <td>65.67%</td>
        <td>44.04%</td>
        <td>31.59%</td>
        <td>30.74%</td>
      </tr>
      <tr>
        <td></td>
        <td><strong>Transformer 200K + Ours</strong></td>
        <td><strong>50.95%</strong></td>
        <td><strong>53.94%</strong></td>
        <td><strong>42.12%</strong></td>
        <td><strong>31.58%</strong></td>
        <td><strong>30.71%</strong></td>
      </tr>
      <tr>
        <td></td>
        <td>Transformer 800K</td>
        <td>47.41%</td>
        <td>62.13%</td>
        <td>40.53%</td>
        <td>25.97%</td>
        <td>25.52%</td>
      </tr>
      <tr>
        <td></td>
        <td><strong>Transformer 800K + Ours</strong></td>
        <td><strong>44.34%</strong></td>
        <td><strong>49.68%</strong></td>
        <td><strong>38.79%</strong></td>
        <td><strong>25.94%</strong></td>
        <td><strong>25.45%</strong></td>
      </tr>
      <tr>
        <td></td>
        <td>Transformer 3.2M</td>
        <td>34.15%</td>
        <td>41.02%</td>
        <td>32.02%</td>
        <td>18.53%</td>
        <td>17.66%</td>
      </tr>
      <tr>
        <td></td>
        <td><strong>Transformer 3.2M + Ours</strong></td>
        <td><strong>32.04%</strong></td>
        <td><strong>36.61%</strong></td>
        <td><strong>31.29%</strong></td>
        <td><strong>18.52%</strong></td>
        <td><strong>17.65%</strong></td>
      </tr>
    </tbody>
  </table>
</details>

#### Experiments on GPT-2

<details>
<summary>Expand to see the full table</summary>
  <table>
    <thead>
      <tr>
        <th>Tokenizer</th>
        <th>Compressor</th>
        <th>math</th>
        <th>code</th>
        <th>shakespeare</th>
        <th>enwik8*</th>
        <th>enwik9*</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>BPE (GPT-2)</strong></td>
        <td>Naive Bayes</td>
        <td>66.41%</td>
        <td>59.30%</td>
        <td>49.74%</td>
        <td>48.85%</td>
        <td>53.43%</td>
      </tr>
      <tr>
        <td></td>
        <td>GPT-2</td>
        <td>17.68%</td>
        <td>14.17%</td>
        <td>23.44%</td>
        <td>16.48%</td>
        <td>16.73%</td>
      </tr>
      <tr>
        <td></td>
        <td><strong>GPT-2 + Ours</strong></td>
        <td><strong>17.55%</strong></td>
        <td><strong>14.16%</strong></td>
        <td><strong>23.11%</strong></td>
        <td><strong>16.42%</strong></td>
        <td><strong>16.65%</strong></td>
      </tr>
    </tbody>
  </table>
</details>

#### Experiments on LLaMA 3

<details>
<summary>Expand to see the full table</summary>
  <table>
    <thead>
      <tr>
        <th>Tokenizer</th>
        <th>Compressor</th>
        <th>math</th>
        <th>code</th>
        <th>shakespeare</th>
        <th>enwik8*</th>
        <th>enwik9*</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>BPE (LLaMA 3)</strong></td>
        <td>Naive Bayes</td>
        <td>68.70%</td>
        <td>47.54%</td>
        <td>51.35%</td>
        <td>48.87%</td>
        <td>51.93%</td>
      </tr>
      <tr>
        <td></td>
        <td>LLaMA 3.2-1B</td>
        <td>8.54%</td>
        <td>6.66%</td>
        <td>16.51%</td>
        <td>10.22%</td>
        <td>10.05%</td>
      </tr>
      <tr>
        <td></td>
        <td><strong>LLaMA 3.2-1B + Ours</strong></td>
        <td><strong>8.48%</strong></td>
        <td><strong>6.64%</strong></td>
        <td><strong>16.42%</strong></td>
        <td><strong>10.16%</strong></td>
        <td><strong>9.98%</strong></td>
      </tr>
      <tr>
        <td></td>
        <td>LLaMA 3.2-3B</td>
        <td>7.56%</td>
        <td>5.99%</td>
        <td>13.97%</td>
        <td>9.16%</td>
        <td>8.93%</td>
      </tr>
      <tr>
        <td></td>
        <td><strong>LLaMA 3.2-3B + Ours</strong></td>
        <td><strong>7.50%</strong></td>
        <td><strong>5.95%</strong></td>
        <td><strong>13.88%</strong></td>
        <td><strong>9.09%</strong></td>
        <td><strong>8.86%</strong></td>
      </tr>
      <tr>
        <td></td>
        <td>LLaMA 3-8B</td>
        <td>6.90%</td>
        <td>5.61%</td>
        <td>4.74%</td>
        <td>8.18%</td>
        <td>8.10%</td>
      </tr>
      <tr>
        <td></td>
        <td><strong>LLaMA 3-8B + Ours</strong></td>
        <td><strong>6.84%</strong></td>
        <td><strong>5.57%</strong></td>
        <td><strong>4.73%</strong></td>
        <td><strong>8.12%</strong></td>
        <td><strong>8.04%</strong></td>
      </tr>
    </tbody>
  </table>
</details>

## Multiple Experts
Beyond combining simple statistical methods with LLMs, we can also combine multiple pretrained LLMs together to further improve compression rates.

### Experiment Results
The three decoder-only transformers we used are all trained on the same dataset (i.e., enwik8) but with different model sizes (i.e., 200k, 800k, and 3.2M parameters respectively). We combine them together using wPoE and evaluate the compression rates on three new datasets (i.e., math, code, and shakespeare).

The results show that even if transformer 200k and transformer 800k are much more powerful than Naive Bayes when used alone, the improvement brought by combining Naive Bayes with LLMs is still significant. This indicates that diversity is really important to wPoE, some times even more important than the performance of each expert.

<details>
<summary>Expand to see the full table</summary>
  <table>
    <thead>
      <tr>
        <th>Compressor</th>
        <th>math</th>
        <th>code</th>
        <th>shakespeare</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1 expert</td>
        <td>34.15%</td>
        <td>41.02%</td>
        <td>32.02%</td>
      </tr>
      <tr>
        <td>2 experts wPoE</td>
        <td>33.63%</td>
        <td>40.59%</td>
        <td>31.99%</td>
      </tr>
      <tr>
        <td>3 experts wPoE</td>
        <td>33.62%</td>
        <td>40.46%</td>
        <td>31.97%</td>
      </tr>
      <tr>
        <td><strong>4 experts wPoE</strong></td>
        <td><strong>31.99%</strong></td>
        <td><strong>36.49%</strong></td>
        <td><strong>31.35%</strong></td>
      </tr>
    </tbody>
  </table>
</details>

## Possible Applications in Other Domains
<!-- TODO: Add potential future applications -->
To be done.
<!-- TODO: Add an explanation of the code implementation (i.e., how to use it) -->

<!-- TODO: Add a comparison between weighted sum and weighted product of probabilities -->

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{zhang-etal-2025-test,
    title = "Test-Time Steering for Lossless Text Compression via Weighted Product of Experts",
    author = "Zhang, Qihang  and
      Li, Muchen  and
      Wang, Ziao  and
      Liao, Renjie  and
      Wang, Lele",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.110/",
    pages = "2076--2088",
    ISBN = "979-8-89176-335-7",
    abstract = "Lossless compression techniques are crucial in an era of rapidly growing data. Traditional universal compressors like gzip offer low computational overhead, high speed, and broad applicability across data distributions. However, they often lead to worse compression rates than modern neural compressors, which leverage large-scale training data to model data distributions more effectively.Despite their advantages, neural compressors struggle to generalize to unseen data. To address this limitation, we propose a novel framework that performs Test-Time Steering via a Weighted Product of Experts (wPoE).At inference, our method adaptively combines a universal compression model with a pretrained neural language model, ensuring the compression rate is at least as good as the best individual model.Extensive experiments demonstrate that our approach improves the performance of text compression without requiring fine-tuning. Furthermore, it seamlessly integrates with any autoregressive language model, providing a practical solution for enhancing text compression across diverse data distributions."
}
```

*References*
