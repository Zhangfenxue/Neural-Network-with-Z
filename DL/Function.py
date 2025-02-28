import math
import numpy as np

def Sigmoid(z):
    return 1/(1+math.exp(-z))

def Tanh(z):
    return (math.exp(z) - math.exp(-z))/(math.exp(z) + math.exp(-z))

def ReLU(z):
    return max(0, z)

def Leaky_ReLU(z, alpha = 0.1):
    return max(alpha*z, z)

def Softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=0)

def MSE(t, y):
    return 0.5 * np.sum((t - y)**2)

def CEE(t, y):
    epsilon = 1e-12
    y = np.clip(y, epsilon, 1. - epsilon)

    loss = -np.sum(t * np.log(y)) / y.shape[0]
    return loss