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

When I was a child, I was always wondering if I use the compressor to compress a file over and over again, will the file get smaller and smaller until it vanishes? Today I have known the answer is no, because of the fundamental limits of lossless compression established by information theory. 

But how about using multiple compressors together? If we combine multiple compressors simultaneously, can we achieve better compression ratios? And how can we design such a way to combine different compressors? This is the question that my recent work "Test-Time Steering for Lossless Text Compression via Weighted Product of Experts" aims to answer.

<!-- more -->

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

Thus it's really to think that if we can combine the univerial compressors with the neural-based compressors, so that we can achieve better compression ratios on a wide range of data. It can help neural-based compressors to generalize better on potentially unseen data. In our experiments we can see that the combination of universal compressors is not only helpful for those transformers trained on small datasets, but also helpful for large LLMs like GPT-2 and Llamma3. Meanwhil only very small computation overhead is introduced.



