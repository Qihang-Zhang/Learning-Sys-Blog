---
categories:
  - Reinforcement learning
  - In progress
date: 2025-03-10
draft: true
links:
  - index.md
readtime: 1
slug: Reinforcement learning formulation in Autoregressive models' way
authors:
  - <qihang>
---
# Reinforcement learning formulation in Autoregressive models' way
This post is about my learning experience on RL. I want to re-formulate their notations in the way of autoregressive models.
<!-- more -->
## Overview
[TOC]
## Basic Concepts
**State**: $S_t$, $S_t \in \mathcal{S}$, $S_t$ is a random variable that represents the state of the environment at time $t$.

**Action**: $A_t$, $A_t \in \mathcal{A}$, $A_t$ is a random variable that represents the action of the agent at time $t$.

**Policy**: $\pi_{\theta}(A_t | S_t)$, $\pi_{\theta}(A_t | S_t)$ is a probability distribution that represents the policy of the agent.

**State transition**: $s_{t + 1} \sim P_e(S_{t + 1} | S_t, A_t)$, function $P_e$ is determined by the environment $e$

**Reward**: $R_{t + 1} = R_e(S_{t + 1})$, function $R_e$ is determined by the environment $e$

**Trajectory**: $T_T := \{S_0, A_0, S_1, A_1, \dots, S_T, A_T\} \sim P_{\theta, e}(\cdot), T_T \in \mathcal{T}_T$

+ $P_{\theta, e}(S_t, A_t | T_{t - 1}) = \pi_{\theta}(A_t | S_t)\cdot g_e(S_t | S_{t - 1}, A_{t - 1})$

+ $P_{\theta, e}(T_T) = \displaystyle\prod_{t = 0}^{T} P_{\theta, e}(S_t, A_t | T_{t - 1}) = \prod_{t = 0}^{T} \pi_{\theta}(A_t | S_t)\cdot g_e(S_t | S_{t - 1}, A_{t - 1})$

+ $g_e(S_0 |  S_{- 1}, A_{- 1}):= g_e(S_0)$
    
**Accumulated reward**: $G_t := \displaystyle \sum_{k = 0}^{T - t - 1} \gamma^k R_{t + k + 1}$

Given each $R_{t}$ is determined by the environment, $S_t$ , and $A_t$, the random variable $G_t = \displaystyle \sum_{k = 0}^{T - t - 1} \gamma^k R_{t + k + 1}$ is determined by the trajectory $T_{t:T} = (S_t, A_t, S_{t + 1}, A_{t + 1}, \dots, S_T, A_T)$ and the environment $e$. 

The relation between the $G_t$ and $G_{t + 1}$ is $G_t = R_{t + 1} + \gamma G_{t + 1}$, thus the relationship of their expectations is:

$$
\begin{align} 
    \mathbb{E}_{\tau_{t:T} \sim P_{\theta, e}(\cdot)}[G_t] &= \mathbb{E}_{\tau_{t:T} \sim P_{\theta, e}(\cdot)}[R_{t + 1}] + \gamma\mathbb{E}_{\tau_{t:T} \sim P_{\theta, e}(\cdot)}[G_{t + 1}] \\
    &= \mathbb{E}_{s_{t + 1}\sim P_{\theta, e}(\cdot)}[R_e(S_{t + 1})] + \gamma\mathbb{E}_{\tau_{t + 1:T} \sim P_{\theta, e}(\cdot)}[G_{t + 1}] \\
    &= \sum_{S_{t + 1} \in \mathcal{S}} P_{\theta, e}(S_{t + 1} | S_t)\cdot R_e(S_{t + 1}) + \gamma\mathbb{E}_{T_{t + 1:T} \sim P_{\theta, e}(\cdot)}[G_{t + 1}]
\end{align}
$$







## Tabular method
### Model-free control
#### SARSA
$$
Q_{n + 1}(s_t, a_t) \leftarrow (1 - \alpha)Q_n(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q_n(s_{t+1}, a_{t+1}) \right]
$$
#### Q-learning








***References:***

\bibliography

