# THIS FILE IS A WORK IN PROGRESS.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from qiskit.aqua.components.optimizers import *

TRAIN_SAMPLE_COUNT = 6
SAMPLES = 1000
noise_variance = 0.5
interval_min = 0
interval_max = np.pi * 2

f_ideal = lambda theta: 5.8 * np.sin(theta + 2.2) - 7.8
f = lambda theta: f_ideal(theta) + np.random.normal(0, noise_variance)

# X = np.matrix([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]).T
# X = np.matrix([1, np.pi, np.pi + 1]).T
X = np.linspace(0, np.pi + 1, TRAIN_SAMPLE_COUNT).reshape((TRAIN_SAMPLE_COUNT, -1))
# X = np.matrix([0, np.pi, np.pi * (3 / 4)]).T
y = np.matrix([f(x) for x in np.array(X)])

x = np.atleast_2d(np.linspace(interval_min, interval_max, SAMPLES)).T
true_ys = [f_ideal(_x) for _x in x]
sample_ys = [f(_x) for _x in x]

print(X)
print(y)


def K(theta1, theta2, a, b, c):
    return (a ** 2) * np.cos(theta1 - theta2) + (c ** 2)

# print(K(0, 2 * np.pi))
# print(K(2,  2 * np.pi))

class Kernel2(Kernel):
    def __init__(self, K, a=1, b=0, c=0):
        self.K = K
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, _X, _Y=None, eval_gradient=False):
        if _Y is None:
            _Y = _X
        matrix = np.zeros((len(_X), len(_Y)))
        for x1 in range(len(_X)):
            for x2 in range(len(_Y)):
                add = 0
                if x1 == x2:
                    add = 1e-5
                matrix[x1, x2] = self.K(_X[x1][0], _Y[x2][0], self.a, self.b, self.c) + add
        return matrix

    def diag(self, _X):
        diag = np.diag(self.__call__(_X))
        return np.copy(diag)

    def is_stationary(self):
        return True

    def get_params_(self, deep=True):
        params = dict(a=self.a, b=self.b, c=self.c)
        return params

    def set_params_(self, params: dict):
        self.a = params["a"]
        self.b = params["b"]
        self.c = params["c"]


def log_model_evidence(K_i, y):
    return -0.5 * np.matmul(y.T, np.matmul(np.linalg.inv(K_i), y)) - 0.5 * np.log(np.linalg.det(K_i)) - (len(y) / 2) * np.log(2 * np.pi)


class KernelOpt:
    def __init__(self, kernel):
        self.kernel = kernel

    def cost(self, params):
        params = dict(a=params[0], b=0, c=params[1])
        self.kernel.set_params_(params)
        gram = self.kernel(X)
        cost = -float(log_model_evidence(gram, y))
        print(cost)
        return cost


kernel = Kernel2(K)
kernelOpt = KernelOpt(kernel)

opt = COBYLA(maxiter=200)
initial_pnt = [1, 0]
ret = opt.optimize(num_vars=2, objective_function=kernelOpt.cost, initial_point=np.array(initial_pnt))
print(ret)

gp = GaussianProcessRegressor(kernel=kernel,
# gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                              n_restarts_optimizer=0)
gp.fit(X, y)

y_pred, sigma = gp.predict(x, return_std=True)

plt.plot(x, y_pred, label="Prediction")
plt.plot(X, y, 'r.', markersize=8, label='Observations')

upper = [float(y_pred[i] + sigma[i]) for i in range(len(y_pred))]
lower = [float(y_pred[i] - sigma[i]) for i in range(len(y_pred))]
x1d = list(np.array(x.reshape(1, -1)))[0]
plt.fill_between(x1d, upper, lower, alpha=0.1, fc='r', label="95% credible interval")

plt.plot(x, true_ys, label="f(x) ideal")
plt.scatter(x, sample_ys, s=3, alpha=0.4, c="orange", label="f(x) samples")

plt.title("Additive Gaussian Noise Variance: " + str(noise_variance))
plt.xlabel(r"$\theta$")
plt.ylabel("y")
plt.legend()
plt.scatter(np.array(X), np.array(y), s=10)
plt.xlim([interval_min, interval_max])
plt.show()



