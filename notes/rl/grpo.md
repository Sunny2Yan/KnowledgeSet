# Group Relative Policy Optimization
paper: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

![](/imgs/rl/grpo/img.png)

grpo训练流程：
1. Generating completion

   从数据集中采样一个batch的prompts，对每一个prompt生成一组 $G$ 个 completions；

3. computing the advantage

   对上面一组内 $G$ 个completions，计算每一个reward function下的奖励，并按优势函数 $\hat{A}_{i,t}=\frac{r_i-\mathbb{mean}(r)}{\mathbb{std}(r)}$ 计算组内每一个奖励对应的优势

4. estimating the KL divergence

   $$D_{KL}[\pi_\theta || \pi_{ref}] = \frac{\pi_{ref}(o_{i,t} | q,o_{i,<t})} {\pi_{\theta}(o_{i,t} | q, o_{i,<t})} - \log \frac{\pi_{ref}(o_{i,t} | q,o_{i,<t})}{\pi_{\theta}(o_{i,t} | q,o_{i,<t})} − 1$$

4. computing the loss:
   目标是最大化 Advantage 同时确保 policy model 需要接近原始的 reference model，即保留通用知识。
   $$
   \mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \left[ \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{[\pi_\theta(o_{i,t} \mid q, o_{i,<t})]_{\text{no grad}}} \hat{A}_{i,t} - \beta \mathcal{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] \right],
   $$


原始论文中的目标公式：
$$
\begin{aligned}
\mathcal{L}_{\text{GRPO}}(\theta) &= \mathbb{E}[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)] \\
&= - \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left[ \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}, 1-\epsilon, 1+\epsilon \right) \hat{A}_{i,t} \right] - \beta \mathcal{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] \right\},
\end{aligned}
$$