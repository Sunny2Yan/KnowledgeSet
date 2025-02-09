# 深度 Q 网络（deep Q network，DQN）

在 Q-learning 算法中，以矩阵的方式建立了一张存储每个状态下所有动作值的表格。表格中的每一个动作价值表示在状态下选择动作然后继续遵循某一策略预期能够得到的期望回报。
当状态或者动作连续的时候，就有无限个状态动作对，则无法使用这种表格形式来记录各个状态动作对的值。
因此，可以用函数拟合的方法来估计值，即将这个复杂的值表格视作数据，使用一个参数化的函数来拟合这些数据。因此被称为近似方法。

DQN 算法就是用来解决连续状态下离散动作的问题。

在 Q-learning 中通过下面更新规则来最大化动作价值函数：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \max_{a'\in A}Q(s', a') - Q(s, a)]
$$

于是，对于一组数据 $\{(s_i, a_i, r_i, s_i')\}$，可以很自然地将 Q 网络的损失函数构造为均方误差的形式:

$$
w^* = \arg\max_{w} \frac{1}{2N} \sum_{i=1}^{N} [Q_{w}(s_i, a_i)-(r_i+\gamma\max_{a'}Q_{w}(s_i', a_i'))]^2
$$

由于 DQN 是离线策略算法，因此在收集数据的时候可以使用一个 $\epsilon$-贪婪策略来平衡探索与利用，将收集到的数据存储起来，在后续的训练中使用。

DQN 中包含经验回放和目标网络两个模块：

1. 经验回放（experience replay）

   方法：维护一个回放缓冲区，将每次从环境中采样得到的四元组数据（状态、动作、奖励、下一状态）存储到回放缓冲区中，训练 Q 网络的时候再从回放缓冲区中随机采样若干数据来进行训练。

   目的：
   1）使样本满足独立假设：当前时刻的状态和上一时刻的状态有关，因此不满足独立同分布。采用经验回放可以打破样本之间的相关性，让其满足独立假设。
   2）提高样本效率：每一个样本可以被使用多次。
2. 目标网络

DQN 算法流程：

DQN 算法最终更新的目标是让逼近，由于 TD 误差目标本身就包含神经网络的输出，因此在更新网络参数的同时目标也在不断地改变，这非常容易造成神经网络训练的不稳定性


- 用随机的网络参数 $w$ 初始化网络 $Q_w(s, a)$
- 复制相同的参数 $w^- \leftarrow w$ 来初始化目标网络 $Q_{w'}$
- 初始化经验回放池 $R$
- for 序列 $e=1 \rightarrow E$ do
- &ensp; &ensp; &ensp; &ensp; 获取环境初始状态 $s_1$
- &ensp; &ensp; &ensp; &ensp; for 时间步 $t=1 \rightarrow T$ do
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 根据当前网络 $Q_w(s, a)$ 以 $epsilon$-贪婪策略选择动作 $a_t$
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 执行动作 $a_t$，获得回报 $r_t$，环境状态变为 $s_{t+1}$
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 将 $(s_t, a_t, r_t, s_{t+1})$ 存储进回放池 $R$ 中
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 若$R$中数据足够，从中采样$N$个数据 $\{(s_i, a_i, r_i, s_{i+1})\}_{i=1,\cdots, N}$
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 对每个数据，用目标网络计算 $y_i = r_i + \gamma\max_{a}Q_{w^-}(s_{i+1}, a)$
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 最小化目标损失$L=\frac{1}{N}\sum_i (y_i - Q_w(s_i, a_i))^2$，以此更新当前网络$Q_w$
- &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 更新目标网络
- &ensp; &ensp; &ensp; &ensp; end for
- end for
