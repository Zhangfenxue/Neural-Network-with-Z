import numpy as np
import matplotlib.pyplot as plt

'''f is Objective Function
    grad_f is Gradient of Objective Function'''


def momentum(f, grad, start, lr=0.1, beta=0.9, n=100, plot=True):
    x = start
    v = 0
    history = [x]
    for _ in range(n):
        g = grad(x)
        v = beta * v + (1 - beta) * g
        x = x - lr * v
        history.append(x)
    momentum_path = np.array(history)

    if plot:
        plt.scatter(momentum_path, f(momentum_path), c='red', s=20, label="Momentum")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.title("Optimization Paths of Momentum Algorithms")
        plt.show()

    return momentum_path

def AdaGrad(f, grad, start, lr=0.1, epsilon=1e-8, n=100, plot=True):
    x = start
    cache = 0
    history = [x]
    for _ in range(n):
        g = grad(x)
        cache += g ** 2
        x = x - lr * g / (np.sqrt(cache) + epsilon)
        history.append(x)

    adaGrad_path = np.array(history)

    if plot:
        plt.scatter(adaGrad_path, f(adaGrad_path), c='red', s=20, label="AdaGrad")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.title("Optimization Paths of AdaGrad Algorithms")
        plt.show()

    return adaGrad_path


def RMSProp(f, grad, start, lr=0.1, dr=0.9, epsilon=1e-8, n=100, plot=True):
    x = start
    cache = 0
    history = [x]
    for _ in range(n):
        g = grad(x)
        cache = dr * cache + (1 - dr) * g ** 2  # 指数加权平均
        x = x - lr * g / (np.sqrt(cache) + epsilon)  # 更新参数
        history.append(x)

    RMSProp_path = np.array(history)

    if plot:
        plt.scatter(RMSProp_path, f(RMSProp_path), c='red', s=20, label="RMSProp")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.title("Optimization Paths of RMSProp Algorithms")
        plt.show()

    return RMSProp_path

def adam(f, grad, strat, lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, n=100, plot=True):
    x = strat
    m = 0
    v = 0
    t = 0
    history = [x]
    for _ in range(n):
        t += 1
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        history.append(x)

    adam_path = np.array(history)

    if plot:
        plt.scatter(adam_path, f(adam_path), c='red', s=20, label="Adam")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.title("Optimization Paths of Adam Algorithms")
        plt.show()

    return adam_path