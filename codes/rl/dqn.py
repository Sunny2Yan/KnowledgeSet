import gym
import torch
import random
import collections
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from dataclasses import dataclass


def set_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ReplayBuffer:
    """经验回放池"""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(torch.nn.Module):
    """只有一层隐藏层的Q网络"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)


class ConvolutionalQnet(torch.nn.Module):
    """加入卷积层的Q网络"""
    def __init__(self, action_dim, in_channels=4):
        super(ConvolutionalQnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, action_dim)

    def forward(self, x):
        x = x / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x))
        return self.head(x)


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1


class Trainer:
    def __init__(self, cfg):
        # set_seed(cfg.env)
        self.cfg = cfg
        self.replay_buffer = ReplayBuffer(self.cfg.buffer_size)
        self.agent = DQN(cfg.state_dim, cfg.hidden_dim, cfg.action_dim, cfg.lr,
                         cfg.gamma, cfg.epsilon, cfg.target_update, cfg.device)
        self.return_list = []  # 回报值

    def iterate_one_episode(self):
        episode_return = 0
        state = self.cfg.env.reset()[0]
        done = False
        while not done:
            action = self.agent.take_action(state)
            next_state, reward, done, _, _ = self.cfg.env.step(action)
            self.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            # 当buffer数据的数量超过一定值后,才进行Q网络训练
            if self.replay_buffer.size() > self.cfg.minimal_size:
                b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(self.cfg.batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                self.agent.update(transition_dict)
        self.return_list.append(episode_return)

    def train_episodes(self, num_episodes):
        for i in range(num_episodes):
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    self.iterate_one_episode()
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({
                            'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                            'return': '%.3f' % np.mean(self.return_list[-10:])
                        })
                    pbar.update(1)

    def show_returns(self):
        episodes_list = list(range(len(self.return_list)))
        plt.plot(episodes_list, self.return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DQN on {}'.format(self.cfg.env_name))
        plt.show()

        mv_return = self.moving_average(self.return_list, 9)
        plt.plot(episodes_list, mv_return)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DQN on {}'.format(self.cfg.env_name))
        plt.show()

    def moving_average(self, a, window_size):
        cumulative_sum = np.cumsum(np.insert(a, 0, 0))
        middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        r = np.arange(1, window_size-1, 2)
        begin = np.cumsum(a[:window_size-1])[::2] / r
        end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
        return np.concatenate((begin, middle, end))


@dataclass
class Configs:
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda_programming") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

if __name__ == '__main__':
    set_seed()
    trainer = Trainer(Configs())
    trainer.train_episodes(100)
    trainer.show_returns()
