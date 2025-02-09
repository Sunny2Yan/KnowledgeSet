import time
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from envs.cliff_walking import CliffWalkingEnv


class DynaQ:
    """ Dyna-Q算法 """
    def __init__(self, n_row, n_col, epsilon, alpha, gamma, n_planning, n_action=4):
        self.Q_table = np.zeros([n_row * n_col, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

        self.n_planning = n_planning  #执行Q-planning的次数
        self.model = dict()  # 环境模型

    def take_action(self, state):  # 选取下一步的操作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def q_learning(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def update(self, s0, a0, r, s1):
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = r, s1  # 将数据添加到模型中
        for _ in range(self.n_planning):  # Q-planning循环
            # 随机选择曾经遇到过的状态动作对
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)


class Trainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self, num_episodes):
        return_list = []  # 记录每一条序列的回报
        for i in range(10):  # 显示10个进度条
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                    episode_return = 0
                    state = self.env.reset()
                    done = False
                    while not done:
                        action = self.agent.take_action(state)
                        next_state, reward, done = self.env.step(action)
                        episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                        self.agent.update(state, action, reward, next_state)
                        state = next_state
                    return_list.append(episode_return)
                    if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                        pbar.set_postfix({
                            'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                            'return': '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)
        return return_list


def show_returns(n_planning_list, num_episodes):
    for idx, n_planning in enumerate(n_planning_list):
        print('Q-planning步数为：%d' % n_planning)
        time.sleep(0.5)
        agent.n_planning = n_planning
        trainer = Trainer(env, agent)
        return_list = trainer.train(num_episodes)
        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list, label=str(n_planning) + ' planning steps')
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Dyna-Q on {}'.format('Cliff Walking'))
    plt.show()


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    n_row, n_col = 4, 12
    epsilon, alpha = 0.01, 0.1
    gamma = 0.9
    num_episodes = 300
    n_planning_list = [0, 2, 20]
    env = CliffWalkingEnv(n_row, n_col)
    agent = DynaQ(n_row, n_col, epsilon, alpha, gamma, 2)
    show_returns(n_planning_list, num_episodes)
