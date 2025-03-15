import math
import numpy as np


# activation function
def Sigmoid(z):
    return 1/(1+math.exp(-z))

def dSigmoid(z):
    s = Sigmoid(z)
    return s * (1 - s)

def Tanh(z):
    return (math.exp(z) - math.exp(-z))/(math.exp(z) + math.exp(-z))

def dTanh(z):
    return 1 - Tanh(z)**2

def ReLU(z):
    return max(0, z)

    return 1 if z > 0 else 0

def Leaky_ReLU(z, alpha = 0.1):
    return max(alpha*z, z)

def dLeaky_ReLU(z, alpha=0.1):
    return 1 if z > 0 else alpha

def Softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=0)

def dSoftmax(z):
    S = Softmax(z)
    jacobian_m = np.diag(S)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i != j:
                jacobian_m[i][j] = -S[i] * S[j]
    return jacobian_m



# lose function
def MSE(t, y):
    return 0.5 * np.sum((t - y)**2)

def CEE(t, y):
    epsilon = 1e-12
    y = np.clip(y, epsilon, 1. - epsilon)

    loss = -np.sum(t * np.log(y)) / y.shape[0]
    return loss



def Gradient_Descent(f, df, start, num, eta = 0.01):
    x = start
    for i in range(num):
        gradient = df(x)
        x -= eta * gradient

    return x