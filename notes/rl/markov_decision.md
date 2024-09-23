# 马尔可夫决策过程
**随机过程（stochastic process）**是概率论的“动力学”部分。概率论是研究静态的随机现象，而随机过程是研究随时间变化的随机现象。

随机过程中，在某时刻 $t$ 的取值是一个向量随机变量，用 $S_t$ 表示，所有可能的状态组成状态集合 $S$。则 $S_{t+1}$ 的概率表示为 $P(S_{t+1} | S_t, \cdots, S_1)$。

## 1. 马尔可夫过程

### 1.1 马尔可夫性质
$P(S_{t+1} | S_t) = P(S_{t+1} | S_t, \cdots, S_1)$，即下一个状态只取决于当前状态，而不会受到过去状态的影响。

### 1.2 马尔可夫过程
**马尔可夫过程（Markov process）**指具有马尔可夫性质的随机过程，也被称为马尔可夫链（Markov chain）。

用元组 $(S, P)$ 来描述一个马尔可夫过程，其中，$S$ 为有限数量的状态集合，$P$ 为状态转移矩阵（state transition matrix）$。
$P_{ij} = P(s_j | s_i) = P(S_{t+1}=s_j | S_t=s_i)$ 表示从状态 $s_i$ 转移到状态 $s_j$ 的概率

给定一个马尔可夫过程，从某个状态出发，根据它的状态转移矩阵生成一个**状态序列（episode）**，这个步骤也被叫做**采样（sampling）**。

## 2. 马尔可夫奖励过程
在马尔可夫过程的基础上加入奖励函数 $r$ 和折扣因子 $\gamma$，就可以得到马尔可夫奖励过程（Markov reward process）。即，一个马尔可夫奖励过程由 $(S, P, r, \gamma)$ 构成。

- $r$ 是奖励函数，某个状态 $s$ 的奖励 $r(s)$ 是指转移到该状态时可以获得奖励的期望；
- $\gamma$ 是折扣因子（discount factor），取值范围为 $[0, 1)$。

### 2.1 回报
马尔可夫奖励过程中，从第 $t$ 时刻状态 $S_t$ 开始，直到终止状态时，所有奖励的衰减之和称为回报$G_t$（Return）。

$$
G_t = R_t + \gamma R_{t+1} + \gamma ^2 R_{t+2} + \cdots = \sum_{k=0}^{K} \gamma ^k R_{t+k}
$$

### 2.2 价值函数

马尔可夫奖励过程中，一个状态的期望回报（即从这个状态出发的未来累积奖励的期望）被称为这个状态的价值（value）。所有状态的价值就组成了价值函数（value function）。
即，$V(s) = \mathbb{E} [G_t | S_t=s]$，展开为：

$$
\begin{aligned}
V(s) &= \mathbb{E} [G_t | S_t=s] \\
&= \mathbb{E} [R_t + \gamma R_{t+1} + \gamma ^2 R_{t+2} + \cdots | S_t=s] \\
&= \mathbb{E} [R_t + \gamma G_{t+1} | S_t=s] \\
&= \mathbb{E} [R_t + \gamma V(S_{t+1}) | S_t=s]
\end{aligned}
$$




