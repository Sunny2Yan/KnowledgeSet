"""表格型方法（值迭代算法）

"""
import os
import sys
import random
import numpy as np

from envs.simple_grid import DrunkenWalkEnv


class ValueIteration(object):
    def __init__(self):
        self.env = DrunkenWalkEnv(map_name="theAlley")
        self.all_seed(self.env, seed=1)

    @staticmethod
    def all_seed(env, seed=1):
        env.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def value_iteration(self, env, theta=0.005, discount_factor=0.9):
        Q = np.zeros((env.nS, env.nA))  # 初始化一个Q表格
        count = 0
        while True:
            delta = 0.0
            Q_tmp = np.zeros((env.nS, env.nA))
            for state in range(env.nS):
                for a in range(env.nA):
                    accum = 0.0
                    reward_total = 0.0
                    for prob, next_state, reward, done in env.P[state][a]:
                        accum += prob * np.max(Q[next_state, :])
                        reward_total += prob * reward
                    Q_tmp[state, a] = reward_total + discount_factor * accum
                    delta = max(delta, abs(Q_tmp[state, a] - Q[state, a]))
            Q = Q_tmp

            count += 1
            if delta < theta or count > 100:  # 这里设置了即使算法没有收敛，跑100次也退出循环
                break
        return Q


Q = value_iteration(env)
print(Q)


policy = np.zeros([env.nS, env.nA]) # 初始化一个策略表格
for state in range(env.nS):
    best_action = np.argmax(Q[state, :]) #根据价值迭代算法得到的Q表格选择出策略
    policy[state, best_action] = 1

policy = [int(np.argwhere(policy[i]==1)) for i in range(env.nS) ]
print(policy)


num_episode = 1000  # 测试1000次


def test(env, policy):
    rewards = []  # 记录所有回合的奖励
    success = []  # 记录该回合是否成功走到终点
    for i_ep in range(num_episode):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset()  # 重置环境, 重新开一局（即开始新的一个回合） 这里state=0
        while True:
            action = policy[state]  # 根据算法选择一个动作
            next_state, reward, done, _ = env.step(action)  # 与环境进行一个交互
            state = next_state  # 更新状态
            ep_reward += reward
            if done:
                break
        if state == 12:  # 即走到终点
            success.append(1)
        else:
            success.append(0)
        rewards.append(ep_reward)
    acc_suc = np.array(success).sum() / num_episode
    print("测试的成功率是：", acc_suc)


test(env, policy)

