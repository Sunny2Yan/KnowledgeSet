  ## RLHF训练阶段

1. 预训练一个语言模型 (LM) ；
2. 聚合问答数据并训练一个奖励模型 (Reward Model，RM) ；
3. 用强化学习 (RL) 方式微调 LM

### Step 1: 预训练大语言模型

使用 SFT 来微调一个大语言模型 (Instruct Tuning).

### Step 2: 训练奖励（偏好）模型

从预定义数据集中采样prompt-answer文本对，并用初始的 LM 给这些提示生成文本，将其作为RM的训练数据集。

对于训练奖励数值方面，需要人工对 LM 生成的回答进行排名，对不同 LM 在相同提示下的输出进行比较，
然后使用 Elo 系统建立一个完整的排名。这些不同的排名结果将被归一化为用于训练的标量奖励值。

### Step 3: 强化学习微调 LM

一般只调整部分参数（LoRA），将微调任务表述为 RL 问题：

- 策略 (policy) 是一个接受提示并返回一系列文本 (或文本的概率分布) 的 LM；
- 策略的行动空间 (action space) 是 LM 的词表对应的所有词元 (一般在 50k 数量级) ；
- 观察空间 (observation space) 是可能的输入词元序列，也比较大 (词汇量 ^ 输入标记的数量) 。
- 奖励函数是偏好模型和策略转变约束 (Policy shift constraint) 的结合。

PPO 算法：将提示 x 输入step 1中finetune的 LM 和当前微调的 LM，分别得到了输出文本 y1, y2，
将来自当前策略的文本 y2 传递给 RM 得到一个标量的奖励 $r_{\theta}$， 将两个模型的生成文本进行比较计算差异的惩罚项， Kullback–Leibler (KL) divergence 散度的缩放，即

$$
r_{\theta}(y|x)-\lambda_{KL} D_{KL}(\pi_{PPO}(y|x) || \pi_{base}(y|x))
$$

这一项被用于惩罚 RL 策略在每个训练批次中生成大幅偏离初始模型，以确保模型输出合理连贯的文本。
如果去掉这一惩罚项可能导致模型在优化中生成乱码文本来愚弄奖励模型提供高奖励值。

## 强化学习补充

RL的过程是智能体在一个状态 S 下选择动作 A，然后获得收益 R，我们希望优化选择动作的策略，使得总体收益的期望最大。
为了不让模型陷入局部最优而按蒙特卡洛方式一定比例随机游走，在这个过程中得到每个state-action对应的reward作为新的训练样本。
对一组模型参数 $\theta$，可以得到一组轨迹序列的概率分布 $P(\tau; \theta)$. 其中轨迹为：

$$
\tau = [<s_1, a_1>, <s_2, a_2>, \cdots, <s_n, a_n>]
$$

对一条由多个状态动作对组成的轨迹 $\tau$，可以得到reward期望：

$$
R(\tau)=\sum_{t}\gamma^{t}r_t
$$

其中, $\gamma$ 是0-1的折扣因子（因为远期奖励相对不重要），$r_t$ 是不同时间步的reward。

于是目标函数 $J(\theta)=\mathbb{E}_{\tau\sim\pi_0}R(\tau)=\sum_{\tau}P(\tau; \theta)R(\tau)$
使用梯度上升来最大化目标：$\theta = \theta + \alpha \nabla_{\theta} J(\theta)$

$$
\nabla_{\theta} J(\theta) = \nabla_{\theta} \sum_{\tau}P(\tau; \theta)R(\tau) \\
= \sum_{\tau} \nabla_{\theta} P(\tau; \theta)R(\tau) \\
= \sum_{\tau} P(\tau; \theta) \frac{\nabla_{\theta} P(\tau; \theta)}{P(\tau; \theta)} R(\tau) \\
= \sum_{\tau} P(\tau; \theta) \nabla_{\theta} \log(P(\tau; \theta))R(\tau) \\
= \mathbb{E}_{\tau\sim\pi_0} \nabla_{\theta} \log(P(\tau; \theta))R(\tau)
$$

转换成期望形式：在实际计算时，用多次采样的平均值作为期望的近似。
如果把轨迹理解为输出的句子，那么 $\log(P(\tau |\theta))$ 对应在文本生成中就是给定一个输入文本 X，得到输出文本 Y 的概率：

$$
P(Y|X)=P(y_1|X) P(y_2|X, y_1), \cdots P(y_n|X, y_1, y_2, \cdots, y_{n-1})
$$

## PPO(Proximal Policy Optimization) with adaption KL penalty 算法

```
input: policy参数 $\theta_0$, KL系数 $\gamma_0$, KL阈值 $\delta$;

for k=1, 2, ..., do:
    从policy $\pi_k=\pi(\theta_k)$ 收集轨迹集 $D_k$
    计算平均值
    计算policy更新：$\theta_{k+1} = \arg\max_{\theta}L_{\theta_k}(\theta) - \gamma_k D_k(\theta || \theta_k)$
  
    执行K步，使用minibatch SGD进行梯度上升
    if D_k(\theta || \theta_k) >= 1.5\delta:
        \gamma_{k+1} = 2 \beta
    elif D_k(\theta || \theta_k) <= \delta / 1.5:
        \gamma_{k+1} = \gamma_k / 2
</>
```



