# Motivation

要写一个Leanring Blog的想法实在是由来已久，一直以来被[Lil's Log](https://lilianweng.github.io/)所激励，近日又看到朋友写的[Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial?tab=readme-ov-file)，遂决定应该开始写自己的blog，来整理所学所思所想，也抛砖引玉，以飨读者。

虽然自 GPT-3.5 诞生以来已经过去了相当的时间，但直到最近我才后知后觉地意识到，Large-Scale Learning System的研究早已经发展成了一种与2012年以来Alex Net为代表的Deep Learning完全不同的研究范式。其中所需的努力，远非一句 Scaling Law 就能轻描淡写的概括。作为一个System，LLMs 的System Design早已成为一个高度活跃、日新月异的研究领域。

最近Deepseek的进展让我更加直观地感受到这一变化。无论是DeepseekV3采用的[DeepSeekMoE](https://arxiv.org/pdf/2412.19437)还是[NSA](https://arxiv.org/pdf/2502.11089)，都表明在LLMs的设计过程中，为了进一步扩大context window或在有限的计算资源下scale up模型容量，许多设计都在适应GPU的特性，如通信成本等。这些关键进展已经展现出强烈的System的味道。站在当下这个节点，我们在研究中所需的skill set，与小规模Learning System的时代相比，已经发生了巨大变化。

与此同时，随着 Cursor等 Agentic Tools的兴起，可以明显感受到LLMs的表现在很大程度上取决于与其交互的方式。这一特性在LLMs之前的Learning System时代从未展现过。交互方式只是其中一个很直观的例子，总体而言，在更大规模的Learning System中，其特性和研究重心正在发生显著的变化。或许多年后回顾当下，我们会发现LLMs的研究范式与若干年前的Deep Learning之间的差异，可能会像Deep Learning与Machine Learning时代的差异一样巨大。

从这两个角度来看，我认为Large-Scale Learning System的研究所需的skill set和研究重心都在快速变化。这也是我希望撰写一份关于Large-Scale Learning System的tutorial，以整理自己的思考轨迹的原因。

Qihang
2025-02-28