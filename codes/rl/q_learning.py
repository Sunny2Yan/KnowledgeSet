import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from envs.cliff_walking import CliffWalkingEnv


class QLearning:
    """ Q-learning算法 """
    def __init__(self, n_row, n_col, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([n_row * n_col, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

        self.env = CliffWalkingEnv(n_row, n_col)
        self.return_list = []  # 记录每一条序列的回报

    def take_action(self, state):
        """选取下一步的操作"""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        """用于打印策略"""
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def train(self, num_episodes):
        for i in range(10):  # 显示10个进度条
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                    episode_return = 0
                    state = self.env.reset()
                    done = False
                    while not done:
                        action = agent.take_action(state)
                        next_state, reward, done = self.env.step(action)
                        episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                        agent.update(state, action, reward, next_state)
                        state = next_state
                    self.return_list.append(episode_return)
                    if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                        pbar.set_postfix({
                            'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                            'return': '%.3f' % np.mean(self.return_list[-10:])})
                    pbar.update(1)

    def show_returns(self):
        episodes_list = list(range(len(self.return_list)))
        plt.plot(episodes_list, self.return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('Q-learning on {}'.format('Cliff Walking'))
        plt.show()

    def print_agent(self, action_meaning, disaster=[], end=[]):
        for i in range(self.env.n_row):
            for j in range(self.env.n_col):
                if (i * self.env.n_col + j) in disaster:
                    print('****', end=' ')
                elif (i * self.env.n_col + j) in end:
                    print('EEEE', end=' ')
                else:
                    a = self.best_action(i * self.env.n_col + j)
                    pi_str = ''
                    for k in range(len(action_meaning)):
                        pi_str += action_meaning[k] if a[k] > 0 else 'o'
                    print(pi_str, end=' ')
            print()


if __name__ == '__main__':
    np.random.seed(0)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    num_episodes = 500

    agent = QLearning(12, 4, epsilon, alpha, gamma)
    agent.train(num_episodes)
    agent.show_returns()

    action_meaning = ['^', 'v', '<', '>']
    print('Q-learning算法最终收敛得到的策略为：')
    agent.print_agent(action_meaning, list(range(37, 47)), [47])