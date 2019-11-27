import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

SAMPLES = 200
noise_variance = 0.2
interval_min = 0
interval_max = np.pi * 4

f_ideal = lambda theta: 0.5 * np.sin(theta + (2.2))
f = lambda theta: f_ideal(theta) + np.random.normal(0, noise_variance)

# X = np.matrix([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]).T
X = np.matrix([0, 1, np.pi, np.pi + 1, np.pi + 2]).T
# X = np.matrix([0, np.pi, np.pi * (3 / 4)]).T
y = np.matrix([f(x) for x in np.array(X)])

x = np.atleast_2d(np.linspace(interval_min, interval_max, SAMPLES)).T
true_ys = [f_ideal(_x) for _x in x]
sample_ys = [f(_x) for _x in x]

print(X)
print(y)


def K(theta1, theta2):
    x1 = np.matrix([np.cos(theta1), np.sin(theta1)]).T
    x2 = np.matrix([np.cos(theta2), np.sin(theta2)]).T
    val = np.dot(x1.T, x2)
    return val

print(K(0, 2 * np.pi))
print(K(2,  2 * np.pi))

class Kernel2(Kernel):
    def __init__(self, K):
        self.K = K

    def __call__(self, _X, _Y=None, eval_gradient=False):
        if _Y is None:
            _Y = _X
        matrix = np.zeros((len(_X), len(_Y)))
        for x1 in range(len(_X)):
            for x2 in range(len(_Y)):
                add = 0
                if x1 == x2:
                    add = 1e-10 * np.random.rand()
                matrix[x1, x2] = self.K(_X[x1][0], _Y[x2][0]) + add
        return matrix

    def diag(self, _X):
        diag = np.diag(self.__call__(_X))
        return np.copy(diag)

    def is_stationary(self):
        return True


plt.figure()
kernel = Kernel2(K)
# gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_variance, optimizer='fmin_l_bfgs_b',
gp = GaussianProcessRegressor(kernel=kernel, alpha=0, #alpha=noise_variance,
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
plt.scatter(x, sample_ys, s=5, label="f(x) samples")

plt.title("Additive Gaussian Noise Variance: " + str(noise_variance))
plt.xlabel(r"$\theta$")
plt.ylabel("y")
plt.legend()
plt.scatter(np.array(X), np.array(y), s=10)
plt.xlim([interval_min, interval_max])
plt.show()



