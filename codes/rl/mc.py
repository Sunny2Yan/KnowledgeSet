import numpy as np


class MonteCarloMethods:
    def __init__(self):
        self.S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
        self.A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合
        # 状态转移函数
        self.P = {
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
        self.R = {
            "s1-保持s1": -1,
            "s1-前往s2": 0,
            "s2-前往s1": -1,
            "s2-前往s3": -2,
            "s3-前往s4": -2,
            "s3-前往s5": 0,
            "s4-前往s5": 10,
            "s4-概率前往": 1,
        }
        self.gamma = 0.5  # 折扣因子
        self.MDP = (self.S, self.A, self.P, self.R, self.gamma)

        # 策略1,随机策略
        self.Pi_1 = {
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
        self.Pi_2 = {
            "s1-保持s1": 0.6,
            "s1-前往s2": 0.4,
            "s2-前往s1": 0.3,
            "s2-前往s3": 0.7,
            "s3-前往s4": 0.5,
            "s3-前往s5": 0.5,
            "s4-前往s5": 0.1,
            "s4-概率前往": 0.9,
        }

        episodes = self.sample(self.MDP, self.Pi_1, 20, 5)
        print('第一条序列：\n', episodes[0])
        print('第二条序列：\n', episodes[1])
        print('第五条序列：\n', episodes[4])

        episodes = self.sample(self.MDP, self.Pi_1, 20, 1000)
        V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}  # 初始化价值
        N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}  # 初始化计数器
        self.MC(episodes, V, N, self.gamma)
        print("使用蒙特卡洛方法计算MDP的状态价值为：\n", V)

    @staticmethod
    def join(str1, str2):
        return str1 + '-' + str2

    def sample(self, MDP, Pi, timestep_max, number):
        """采样函数,策略Pi,限制最长时间步timestep_max,总共采样序列数number"""
        S, A, P, R, gamma = MDP
        episodes = []
        for _ in range(number):
            episode = []
            timestep = 0
            s = S[np.random.randint(4)]  # 随机选择一个除s5以外的状态s作为起点
            # 当前状态为终止状态或者时间步太长时,一次采样结束
            while s != "s5" and timestep <= timestep_max:
                timestep += 1
                rand, temp = np.random.rand(), 0
                # 在状态s下根据策略选择动作
                for a_opt in A:
                    temp += Pi.get(self.join(s, a_opt), 0)
                    if temp > rand:
                        a = a_opt
                        r = R.get(self.join(s, a), 0)
                        break
                rand, temp = np.random.rand(), 0
                # 根据状态转移概率得到下一个状态s_next
                for s_opt in S:
                    temp += P.get(self.join(self.join(s, a), s_opt), 0)
                    if temp > rand:
                        s_next = s_opt
                        break
                episode.append((s, a, r, s_next))  # 把（s,a,r,s_next）元组放入序列中
                s = s_next  # s_next变成当前状态,开始接下来的循环
            episodes.append(episode)
        return episodes

    def MC(self, episodes, V, N, gamma):
        """对所有采样序列计算所有状态的价值"""
        for episode in episodes:
            G = 0
            for i in range(len(episode) - 1, -1, -1):  #一个序列从后往前计算
                (s, a, r, s_next) = episode[i]
                G = r + gamma * G
                N[s] = N[s] + 1
                V[s] = V[s] + (G - V[s]) / N[s]


if __name__ == '__main__':
    MonteCarloMethods()