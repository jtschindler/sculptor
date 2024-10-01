

import os
import pickle
import numpy as np

class Prior(object):
    def __init__(self, name, distribution, **kwargs):
        self.name = name
        self.distribution = distribution
        self.kwargs = kwargs

    def __str__(self):
        return f'{self.name}: {self.distribution}'

    # def save(self, save_dir, save_prefix=None):
    #
    #     # Check if the directory exists
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #
    #     # Save the prior object
    #     if save_prefix is not None:
    #         save_name = '{}_{}.pkl'.format(save_prefix, self.name)
    #     else:
    #         save_name = '{}.pkl'.format(self.name)
    #
    #     with open(os.path.join(save_dir, save_name), 'wb') as f:
    #         pickle.dump(self, f)
    #
    # def load(self, load_dir, load_prefix=None):
    #
    #     # Load the prior object
    #     if load_prefix is not None:
    #         load_name = '{}_{}.pkl'.format(load_prefix, self.name)
    #     else:
    #         load_name = '{}.pkl'.format(self.name)
    #
    #     with open(os.path.join(load_dir, load_name), 'rb') as f:
    #         self = pickle.load(f)
    #
    #     return self


class UniformPrior(Prior):
    def __init__(self, name, low, upp):

        self.low = low
        self.upp = upp

        super().__init__(name, 'uniform', low=low, upp=upp)

    def evaluate(self, x):
        if self.low <= x <= self.upp:
            return 1 / (self.upp - self.low)
        else:
            return 0

    def logprior(self, x):
        if self.low <= x <= self.upp:
            return 0
        else:
            return -np.inf

    def sample(self, n, rng=None, seed=None):

        if rng is None:
            rng = np.random.default_rng()
        if rng is None and seed is not None:
            rng = np.random.default_rng(seed)

        return rng.uniform(self.low, self.upp, n)


class GaussianPrior(Prior):
    def __init__(self, name, mean, sigma):

        self.mean = mean
        self.sigma = sigma

        super().__init__(name, 'gaussian', mean=mean, sigma=sigma)

    def evaluate(self, x):
        return np.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2) / (self.sigma * np.sqrt(2 * np.pi))

    def logprior(self, x):
        return -0.5 * ((x - self.mean) / self.sigma) ** 2 - np.log(self.sigma * np.sqrt(2 * np.pi))

    def sample(self, n, rng=None, seed=None):

        if rng is None:
            rng = np.random.default_rng()
        if rng is None and seed is not None:
            rng = np.random.default_rng(seed)

        return rng.normal(self.mean, self.sigma, n)