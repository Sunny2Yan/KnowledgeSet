# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt


class Activation:
    def __init__(self):
        ...

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.power(np.e, -x))

    @staticmethod
    def tanh(x):
        return (np.power(np.e, x) - np.power(np.e, -x)) / (
                np.power(np.e, x) + np.power(np.e, -x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    def silu(self, x):
        return x * self.sigmoid(x)

    @staticmethod
    def gelu(x, act_type="vanilla"):
        if act_type.lower() == "vanilla":
            return 0.5 * x * (1.0 + np.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3.0))))
        elif act_type.lower() == "fastgelu":
            """GELU approximation"""
            return 0.5 * x * (1.0 + np.tanh(
                x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
        elif act_type.lower() == "quickgelu":
            """GELU approximation that is fast but somewhat inaccurate"""
            return x * np.sigmoid(1.702 * x)

    def swish(self, x, beta):
        return x * self.sigmoid(beta * x)

    def act_figure(self):
        x = np.linspace(-5, 5, 1000)
        delta = x[1] - x[0]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(x, self.sigmoid(x), x, self.tanh(x), x, self.relu(x),
                     x, self.silu(x), x, gelu(x), x, swish(x, 1))
        # axes[0].set_xlabel("x")
        # axes[0].set_ylabel("y")
        axes[0].set_ylim([-1.1, 1.1])
        axes[0].set_title("activation")
        axes[0].grid(True)
        axes[0].legend(["Sigmoid", "Tanh", "ReLU", "SiLU", "GELU", "Swish"],
                       loc="best")

        axes[1].plot(x, np.gradient(self.sigmoid(x), delta),
                     x, np.gradient(self.tanh(x), delta),
                     x, np.gradient(self.relu(x), delta),
                     x, np.gradient(self.silu(x), delta),
                     x, np.gradient(self.gelu(x), delta),
                     x, np.gradient(self.swish(x, 1), delta))
        axes[1].set_ylim([-0.25, 1.25])
        axes[1].set_title("gradient")
        axes[1].grid(True)
        axes[1].legend(["Sigmoid", "Tanh", "ReLU", "SiLU", "GELU", "Swish"],
                       loc="best")

        plt.show()


if __name__ == '__main__':
    act = Activation()
    act.act_figure()

