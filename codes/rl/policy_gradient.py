"""策略梯度(采集数据花费时间较多)

标准策略梯度：先按初始policy model获取一条轨迹，保留轨迹的每步预测概率和奖励，最后累计每步的 -pred * discount_reward;
带baseline策略梯度： 先获取一条轨迹并保留每步的预测概率和奖励，计算平均奖励作为baseline，折扣奖励更新为折扣奖励-baseline，再计算policy loss和entropy loss，最后累计policy loss - β * entropy loss
"""

import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Optional
from torch.distributions import Categorical

Tensor = torch.Tensor


class Policy(nn.Module):
    """策略网络"""
    def __init__(self, state_size: int, hidden_size: int, action_size: int):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state: Tensor) -> Tensor:
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)  # 对每一行进行softmax


class PolicyGradient(object):
    def __init__(self,
                 env_name: str,
                 episodes: int,
                 max_step: int,
                 learning_rate: float,
                 gamma: float,
                 render_mode: Optional[str] = None):
        self.episodes = episodes
        self.max_step = max_step
        self.gamma = gamma

        self.env = gym.make(env_name, render_mode=render_mode)
        # print("env.action_space: ", self.env.action_space.n)
        # print("env.observation_space: ", self.env.observation_space.shape[0])
        # print("env.observation_space.high: ", self.env.observation_space.high)
        # print("env.observation_space.low: ", self.env.observation_space.low)
        self.device = torch.device("cuda_programming:0" if torch.cuda.is_available() else "cpu")

        self.policy = Policy(self.env.observation_space.shape[0], 16,
                             self.env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    @staticmethod
    def sample_action(policy, state, device):
        state = torch.from_numpy(state).unsqueeze(0).to(device)
        probs = policy.forward(state).to(device)
        m = Categorical(probs)
        action = m.sample()
        # m.log_prob(action)相当于probs.log()[0][action.item()].unsqueeze(0)
        return action.item(), m.log_prob(action), probs

    def vannilla_policy_grandent(self):
        """标准版策略梯度"""
        total_rewards = []  # 保存每一个序列的回报
        for episode in range(self.episodes):
            state = np.float32(self.env.reset()[0])  # (obs, info)
            ep_rewards = []  # 保存当前序列每一步的回报
            saved_log_probs = []  # 保存每一步动作的log probability
            for t in range(self.max_step):  # 得到一条轨迹
                action, log_prob, prob = self.sample_action(
                    self.policy, state, self.device)
                next_state, reward, done, _, _ = self.env.step(action)
                state = next_state
                saved_log_probs.append(log_prob)
                ep_rewards.append(reward)
                if done:
                    break

            total_rewards.append(sum(ep_rewards))
            discounts = [self.gamma ** i for i in range(len(ep_rewards))]
            R_tau = sum([a * b for a, b in zip(discounts[::-1], ep_rewards)])  # 折扣后的rewards
            # baseline = np.mean(total_rewards)  # 过去所有序列的回报均值作为baseline
            # R_tau = R_tau - baseline
            # print(R_tau)
            # 对该条序列进行训练
            policy_loss = torch.tensor([0.])
            for i, log_pi in enumerate(saved_log_probs):
                policy_loss += -log_pi * R_tau
            # print("Policy loss: ", policy_loss.item())

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            recent_reward = np.mean(total_rewards[-100:])
            if episode % 100 == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(
                    episode, recent_reward))
                self.writer.add_scalar("reward_100", recent_reward, episode)
            if recent_reward >= 195.0:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                        episode - 100, recent_reward))
                torch.save(self.policy.state_dict(), 'models/pg_checkpoint.pt')
                break

        self.writer.close()
        return total_rewards

    def policy_gradient_with_constraints(self, entropy_beta):
        """增加了baseline和策略的熵的PG算法"""
        total_rewards = []  # 保存每一个序列的回报
        for episode in range(self.episodes):
            state = np.float32(self.env.reset()[0])
            ep_rewards = []  # 保存当前序列每一步的回报
            saved_log_probs = []  # 保存每一步动作的log probability
            saved_prob = []  # 保存每一步动作的概率，方便计算策略的熵
            for t in range(self.max_step):
                action, log_prob, prob = self.sample_action(self.policy, state, self.device)
                next_state, reward, done, _, _ = self.env.step(action)
                state = next_state
                saved_log_probs.append(log_prob)
                saved_prob.append(prob)
                ep_rewards.append(reward)
                if done:
                    break

            total_rewards.append(sum(ep_rewards))
            discounts = [self.gamma ** i for i in range(len(ep_rewards))]
            R_tau = sum([a * b for a, b in zip(discounts[::-1], ep_rewards)])
            baseline = np.mean(total_rewards)  # 过去所有序列的回报均值作为baseline
            R_tau = R_tau - baseline
            # print(R_tau)
            # 对该条序列进行训练
            policy_loss = torch.tensor([0.])
            entropy_loss = torch.tensor([0.])
            for i, log_pi in enumerate(saved_log_probs):
                policy_loss += -log_pi * R_tau
                entropy_loss += -(saved_prob[i] * torch.log(saved_prob[i])).sum(
                    dim=1)

            # print("Policy loss: ", policy_loss.item())

            total_loss = policy_loss - entropy_beta * entropy_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            recent_reward = np.mean(total_rewards[-100:])
            if episode % 100 == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(
                    episode, recent_reward))
            if recent_reward >= 195.0:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    episode - 100, recent_reward))
                break

        torch.save(self.policy.state_dict(), 'models/pg_checkpoint.pt')

        return total_rewards

    def display_agent(self, model_flie):
        # load the weights from file
        self.policy.load_state_dict(torch.load(model_flie))
        rewards = []
        for i in range(10):  # episodes, play ten times
            total_reward = 0
            state = np.float32(self.env.reset()[0])
            for j in range(10000):  # frames, in case stuck in one frame
                action, _, _ = self.sample_action(self.policy, state, self.device)
                self.env.render()
                next_state, reward, done, _, _ = self.env.step(action)
                state = next_state
                total_reward += reward

                if done:
                    rewards.append(total_reward)
                    break

        print("Test rewards are:", *rewards)
        print("Average reward:", np.mean(rewards))
        self.env.close()


if __name__ == '__main__':
    ENV_NAME = 'CartPole-v1'
    LEARNING_RATE = 0.01
    GAMMA = 0.99
    ENTROPY_BETA = 0.01
    EPISODES = 3000  # 收集3000条序列
    MAX_STEP = 1000  # 每条序列最多1000步

    np.random.seed(0)
    torch.manual_seed(0)  # 策略梯度算法方差很大，设置seed以保证复现性
    # env.seed(1)

    # training
    pg = PolicyGradient(ENV_NAME, EPISODES, MAX_STEP, LEARNING_RATE, GAMMA)
    scores = pg.vannilla_policy_grandent()
    # scores = pg.policy_gradient_with_constraints(ENTROPY_BETA)
    print(scores)

    # testing
    # pg = PolicyGradient(ENV_NAME, EPISODES, MAX_STEP, LEARNING_RATE, GAMMA, "human")
    # pg.display_agent("models/pg_checkpoint.pt")