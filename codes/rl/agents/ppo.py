"""搭建不共享网络的Actor-Critic结构"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

HID_SIZE = 128


class Actor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(Actor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.Tanh(),
            # nn.Linear(HID_SIZE, HID_SIZE),
            # nn.Tanh(),
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(act_size))  # 独立的参数

    def forward(self, x):
        mu = self.mu(x)
        std = self.log_std.exp().expand_as(mu)  # 扩充为跟mu的大小一样
        m = Normal(mu, std)
        return m


class Critic(nn.Module):
    def __init__(self, obs_size):
        super(Critic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            # nn.Linear(HID_SIZE, HID_SIZE),
            # nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x):
        return self.value(x)


'''
"""基于A2C的PPO算法，采用clip的目标函数"""
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import datetime

from collections import namedtuple
from tensorboardX import SummaryWriter
from PPO.ac_model import Actor, Critic, test_net

# 超参数设置
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_BETA = 0.001

EPISODES = 5000  # 收集5000条序列
MAX_STEP = 193  # 每条序列最多193步，减1是batch_size的倍数
PPO_EPOCH = 5
CLIP_EPS = 0.2
BATCH_SIZE = 32

REWARD_THRESHOLD = -100
ACTION_BOUND = 2
LOAD_MODEL = True  # 改为True观察已经训练好的agent
ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
print("env.action_space: ", env.action_space.shape[0])
print("env.observation_space: ", env.observation_space.shape[0])
obs_size = env.observation_space.shape[0]
act_size = env.action_space.shape[0]

# device = torch.device("cuda_programming:0" if torch.cuda_programming.is_available() else "cpu")
device = torch.device("cpu")
actor = Actor(obs_size, act_size).to(device)
critic = Critic(obs_size).to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

Memory = namedtuple('Memory', ['s', 'a', 'r', 's_', 'done', 'value', 'adv'])
now = datetime.datetime.now()
date_time = "{}.{}_{}.{}.{}".format(now.month, now.day, now.hour, now.minute, now.second)

if not LOAD_MODEL:
    writer = SummaryWriter(comment="-ppo_pendulum_" + date_time)


def generalized_advantage_estimation(memories):
    """计算每个time_step对应的泛化优势函数估计，返回新的memories"""
    new_memories = []
    gae = 0
    # 对reward进行normalize
    rewards = [m.r for m in memories]
    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    for t in reversed(range(len(memories)-1)):
        m = memories[t]
        m_r = (m.r - mean_r) / std_r
        if m.done:
            gae = m_r
        else:
            td_error = m_r + GAMMA*memories[t+1].value - m.value
            gae = td_error + GAMMA*GAE_LAMBDA*gae

        new_memories.insert(0, Memory(s=m.s, a=m.a, s_=m.s_, r=gae+m.value, done=m.done, value=m.value, adv=gae))

    return new_memories


def old_log_policy_prob(batch, actor, device):
    states_batch = np.array([b.s for b in batch])
    actions_batch = np.array([b.a for b in batch])
    state_v = torch.FloatTensor(states_batch).to(device)
    action_v = torch.FloatTensor(actions_batch).to(device)
    gaussian_dist = actor(state_v)

    return gaussian_dist.log_prob(action_v)


def ppo(n_episodes, max_step):
    total_rewards = []  # 保存每一个序列的回报

    for i_episode in range(n_episodes):
        state = env.reset()
        memories = []

        # 收集一条序列的信息
        for t in range(max_step):
            state_v = torch.from_numpy(state).float().to(device)
            dist = actor(state_v)
            action = dist.sample().cpu().numpy()
            action = np.clip(action, -ACTION_BOUND, ACTION_BOUND)
            value_est = critic(state_v)

            next_state, reward, done, _ = env.step(action)
            if done:
                memories.append(
                    Memory(s=state, a=action, s_=next_state, r=0, done=done, value=value_est, adv=0))
                break
            else:
                memories.append(
                Memory(s=state, a=action, s_=next_state, r=reward+8/8, done=done, value=value_est, adv=0))

            # 下一个time step
            state = next_state

        rewards = sum([m.r for m in memories])
        batch = generalized_advantage_estimation(memories)
        batch_adv = torch.FloatTensor([b.adv for b in batch]).to(device)
        batch_adv = (batch_adv - torch.mean(batch_adv)) / torch.std(batch_adv)
        # 计算旧策略的log probability
        old_log_policy = old_log_policy_prob(batch, actor, device)  # torch.size([192. 1])

        for _ in range(PPO_EPOCH):
            # compute the loss and optimize over mini batches of size BATCH_SIZE
            for mb in range(0, len(batch), BATCH_SIZE):
                mini_batch = batch[mb:mb + BATCH_SIZE]
                minib_old_log_policy = old_log_policy[mb:mb + BATCH_SIZE]  # torch.size([32, 1])
                minib_adv = batch_adv[mb:mb + BATCH_SIZE]
                minib_action = torch.FloatTensor([m.a for m in mini_batch]).to(device)

                minib_returns = torch.FloatTensor([m.r for m in mini_batch]).to(device)
                minib_states = torch.FloatTensor([m.s for m in mini_batch]).to(device)
                minib_values = critic(minib_states).to(device)  # torch.size([32, 1])

                # 训练critic
                value_loss = F.mse_loss(minib_values.squeeze(-1), minib_returns)
                critic_optimizer.zero_grad()
                value_loss.backward()
                critic_optimizer.step()

                # 训练actor
                minib_dist = actor(minib_states)
                entropy = minib_dist.entropy().mean()  # 策略的熵
                new_log_policy = minib_dist.log_prob(minib_action)
                rt_theta = (new_log_policy - minib_old_log_policy.detach()).exp()

                minib_adv = minib_adv.unsqueeze(-1)
                surr1 = rt_theta * minib_adv
                surr2 = minib_adv * torch.clamp(rt_theta, 1-CLIP_EPS, 1+CLIP_EPS)
                actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_BETA*entropy

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                # print(actor.logstd)

        total_rewards.append(rewards)
        recent_reward = np.mean(total_rewards[-100:])
        writer.add_scalar("actor_loss", actor_loss.item(), i_episode)
        writer.add_scalar("critic_loss", value_loss.item(), i_episode)
        writer.add_scalar("advantage_function", batch_adv.mean().item(), i_episode)
        writer.add_scalar("policy_entropy", entropy.item(), i_episode)

        if i_episode % 100 == 0:
            # ac_losses.append(actor_loss.item())
            # cr_losses.append(value_loss.item())
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, recent_reward))
        if recent_reward >= REWARD_THRESHOLD:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, recent_reward))
            torch.save({'actor': actor.state_dict(),
                        'critic': critic.state_dict()}, 'model/ppo_pendulum_checkpoint_'+date_time+'.pth')
            break

    writer.close()
    return total_rewards



if LOAD_MODEL:
    checkpoint = torch.load("model/ppo_pendulum_checkpoint_6.3_16.18.32.pth")
    actor.load_state_dict(checkpoint['actor'])
    avg_r, avg_step = test_net(actor, env, action_bound=ACTION_BOUND, count=10)
    print("Average rewards:", avg_r, "   Average steps:", avg_step)

else:
    scores = ppo(EPISODES, MAX_STEP)
    end_time = datetime.datetime.now()
    time_delta = (end_time-now).seconds/60
    print("COST: {} mins ".format(time_delta))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')

    # plt.figure(2)
    # plt.title("actor loss")
    # plt.plot(ac_losses)
    # plt.figure(3)
    # plt.title("critic loss")
    # plt.plot(cr_losses)

    plt.show()
'''