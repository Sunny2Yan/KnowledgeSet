# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.power(np.e, -x))


def tanh(x):
    return (np.power(np.e, x) - np.power(np.e, -x)) / (
            np.power(np.e, x) + np.power(np.e, -x))


def relu(x):
    return np.maximum(0, x)


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


def swish(x, beta):
    return x * sigmoid(beta * x)


def act_figure():
    x = np.linspace(-5, 5, 1000)
    delta = x[1] - x[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x, sigmoid(x), x, tanh(x), x, relu(x),
                 x, gelu(x), x, swish(x, 1))
    # axes[0].set_xlabel("x")
    # axes[0].set_ylabel("y")
    axes[0].set_ylim([-1.1, 1.1])
    axes[0].set_title("activation")
    axes[0].grid(True)
    axes[0].legend(["Sigmoid", "Tanh", "ReLU", "GELU", "Swish"], loc="best")

    axes[1].plot(x, np.gradient(sigmoid(x), delta),
                 x, np.gradient(tanh(x), delta),
                 x, np.gradient(relu(x), delta),
                 x, np.gradient(gelu(x), delta),
                 x, np.gradient(swish(x, 1), delta))
    axes[1].set_ylim([-0.25, 1.25])
    axes[1].set_title("gradient")
    axes[1].grid(True)
    axes[1].legend(["Sigmoid", "Tanh", "ReLU", "GELU", "Swish"], loc="best")

    plt.show()


if __name__ == '__main__':
    act_figure()

