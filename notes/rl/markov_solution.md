# Markov 求解

## 1. 马尔可夫过程
马尔可夫过程就是一个状态转移图，如下所示：

![](/imgs/rl/markov/mp_eg.png)

## 2. 马尔科夫奖励过程

![](/imgs/rl/markov/mrp_eg.png)

```python
import numpy as np
np.random.seed(0)

P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
P = np.array(P)

rewards = [-1, -2, -2, 10, 1, 0]  # 定义奖励函数
gamma = 0.5  # 定义折扣因子

# 给定一条序列,计算从某个索引（起始状态）开始到序列最后（终止状态）得到的回报
def compute_return(start_index, chain, gamma):
    G = 0
    for i in reversed(range(start_index, len(chain))):
        G = gamma * G + rewards[chain[i] - 1]
    return G

# 假设一个状态序列,s1-s2-s3-s6
chain = [1, 2, 3, 6]
start_index = 0
G = compute_return(start_index, chain, gamma)  # -2.5

def compute_value(P, rewards, gamma, states_num):
    """利用贝尔曼方程的矩阵形式计算解析解,states_num是MRP的状态数"""
    rewards = np.array(rewards).reshape((-1, 1))  #将rewards写成列向量形式
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P),
                   rewards)
    return value


V = compute_value(P, rewards, gamma, 6)  # [[-2.01950168],[-2.21451846],[ 1.16142785],[10.53809283],[ 3.58728554],[ 0.        ]]
```

## 3. 马尔可夫决策过程
![](/imgs/rl/markov/mdp_eg.png)

```python
import numpy as np

S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合
# 状态转移函数
P = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}
# 奖励函数
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}
gamma = 0.5  # 折扣因子
MDP = (S, A, P, R, gamma)

# 策略1,随机策略
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}
# 策略2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}


# 把输入的两个字符串通过“-”连接,便于使用上述定义的P、R变量
def join(str1, str2):
    return str1 + '-' + str2

gamma = 0.5
# 转化后的MRP的状态转移矩阵
P_from_mdp_to_mrp = [
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]
P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

V = compute_value(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
# [[-1.22555411], [-1.67666232], [ 0.51890482], [ 6.0756193 ], [ 0.        ]]
```

## 4. 蒙特卡洛方法（Monte-Carlo methods）
通过使用重复随机抽样，运用概率统计方法来从抽样结果中归纳出想求的目标的数值估计。

![](/imgs/rl/markov/mc.png)

一个状态的价值是它的期望回报，于是用策略在 MDP 上采样很多条序列，计算从这个状态出发的回报再求其期望即可：
$$V^{\pi}(s) = \mathbb{E}_{\pi}(G_t | S_t=s) \approx \frac{1}{N} \sum_{i=1}^N G_t^{(i)}$$

一般常使用增量更新的方法，即对每个状态 $s$ 和对应回报 $G_t$，进行如下计算：
$$
\begin{aligned}
N(s) &= N(s) + 1 \\
V(s) &= V(s) + \frac{1}{N(s)} (G-V(s))
\end{aligned}
$$

## 5. 动态规划

## 6. 时序差分
动态规划算法要求马尔可夫决策过程是已知的，即要求与智能体交互的环境是完全已知的。这就不需要采样数据点，直接求出最优解。
对于大多数场景，马尔可夫决策过程的状态转移概率是无法写出来，智能体只能和环境进行交互，通过采样到的数据来学习，这类学习方法统称为 **无模型的强化学习（model-free reinforcement learning）**。

**时序差分（temporal difference，TD）** 是一种基于无模型的强化学习算法。

时序差分是一种用来估计一个策略的价值函数的方法。它结合了蒙特卡洛和动态规划算法的思想。即，同蒙特卡洛可以从样本数据中学习，不需要事先知道环境；同动态规划根据贝尔曼方程的思想，利用后续状态的价值估计来更新当前状态的价值估计。

蒙特卡洛方法必须要等整个序列结束之后才能计算得到这一次的回报 $G_t$，而时序差分方法只需要当前步结束即可进行计算。
$$V(s_t) \leftarrow V(s_t) + \alpha [r_t + \gamma V(s_{t+1}) -V(s_t)]$$

其中，$r_t + \gamma V(s_{t+1}) -V(s_t)$ 被称为时序差分误差。至于可以用 $r_t + \gamma V(s_{t+1})$ 来代替 $G_t$ 的原因如下：
$$
\begin{aligned}
V_{\pi}(s) &= \mathbb{E}_{\pi}[G_t | S_t=s] \\
&= \mathbb{E}_{\pi}[\sum_{k=0}^{\infty} \gamma^k R_{t+k} | S_t=s] \\
&= \mathbb{E}_{\pi}[R_t + \gamma \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t=s] \\
&= \mathbb{E}_{\pi}[R_t + \gamma V_{\pi}(S_{t+1}) | S_t=s]
\end{aligned}
$$

因此，在用策略和环境交互时，每采样一步，就可以用时序差分算法来更新状态价值估计。

Q-learning是经典的基于时序差分算法的强化学习算法。它的更新方式如下：
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[R_t + \gamma \max_{a}Q(s_{t+1}, a) - Q(s_t, a_t)]$$

算法流程如下：
- 初始化 $Q(s, a)$
- for 序列 $e = 1 \rightarrow E$ do：
- &ensp; &ensp; &ensp; &ensp; 得到初始状态 $s$
- &ensp; &ensp; &ensp; &ensp; for 时间步 $t = 1 \rightarrow T$ do :
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 用 $\epsilon$-greedy 策略根据 $Q$ 选择当前状态 $s$ 下的动作 $a$
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 得到环境反馈的 $r, s'$
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[R_t + \gamma \max_{a}Q(s_{t+1}, a) - Q(s_t, a_t)]$
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; $s \leftarrow s'$
- &ensp; &ensp; &ensp; &ensp; end for
- end for



## Dyna-Q 算法
Dyna-Q 是基于模型的强化学习算法。它使用一种叫做 Q-planning 的方法来基于模型生成一些模拟数据，然后用模拟数据和真实数据一起改进策略。

Q-planning：每次选取一个曾经访问过的状态 $s$，采取一个曾经在该状态下执行过的动作 $a$，通过模型得到转移后的状态 $s'$ 以及奖励 $r$，并根据这个模拟数据 $(s,a,r,s')$，用 Q-learning 的更新方式来更新动作价值函数。

Dyna-Q 算法流程：
- 初始化 $Q(s, a)$，初始化模型 $M(s, a)$
- for 序列 $e=1 \rightarrow E$ do:
- &ensp; &ensp; &ensp; &ensp; 得到初始状态 $s$
- &ensp; &ensp; &ensp; &ensp; for $t=1 \rightarrow T$ do:
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 用 $\epsilon$-贪婪策略根据 $Q$ 选择当前状态 $s$ 下的动作 $a$
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 得到环境反馈的 $r, s'$
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; $Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \max_{a'}Q(s', a') - Q(s, a)]$
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; $M(s, a) \leftarrow r, s'$
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; for 次数 $n=1 \rightarrow N$ do:
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 随机选择一个曾经访问过的状态 $s_m$
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 采取一个曾经在状态 $s_m$ 下执行过的动作 $a_m$
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; $r_m, s_m' \leftarrow M(s_m, a_m)$
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; $Q(s_m, a_m) \leftarrow Q(s_m, a_m) + \alpha[r_m + \gamma \max_{a'}Q(s_{m}', a') - Q(s_m, a_m)]$
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; end for
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; $s\leftarrow s'$
- &ensp; &ensp; &ensp; &ensp; end for
- end for

其中，$N$ 为 Q-planning 次数，当 $N=0$ 时退化为 Q-learning。