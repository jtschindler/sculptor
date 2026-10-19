import numpy as np


class Prior:
    """Base class for parameter priors.

    Subclasses implement `logprior(x)`, `evaluate(x)` and `_draw(rng, n)`
    (used by `sample`).

    :param name: Prior name
    :type name: str
    :param distribution: Name of the distribution, e.g. "uniform"
    :type distribution: str
    :param kwargs: The distribution's parameters, kept in `self.kwargs`
    """

    def __init__(self, name, distribution, **kwargs):
        self.name = name
        self.distribution = distribution
        self.kwargs = kwargs

    def __str__(self):
        return f"{self.name}: {self.distribution}"

    def sample(self, n, rng=None, seed=None):
        """Draw `n` samples from the prior.

        :param n: Number of samples
        :type n: int
        :param rng: Random number generator to draw from. Takes precedence
            over `seed`; passing both raises a `ValueError`.
        :type rng: numpy.random.Generator
        :param seed: Seed for a new `numpy.random.default_rng` generator,
            used when no `rng` is given. With neither, the generator is
            seeded from fresh OS entropy.
        :type seed: int
        :return: Samples
        :rtype: np.ndarray
        """

        if rng is not None and seed is not None:
            raise ValueError("Pass either `rng` or `seed`, not both.")
        if rng is None:
            rng = np.random.default_rng(seed)

        return self._draw(rng, n)

    def _draw(self, rng, n):
        raise NotImplementedError(f"{type(self).__name__} does not implement sampling.")


class UniformPrior(Prior):
    """Flat prior on the closed interval [`low`, `upp`].

    :param name: Prior name
    :type name: str
    :param low: Lower bound
    :type low: float
    :param upp: Upper bound
    :type upp: float
    """

    def __init__(self, name, low, upp):

        self.low = low
        self.upp = upp

        super().__init__(name, "uniform", low=low, upp=upp)

    def evaluate(self, x):
        """Probability density at `x`: 1 / (upp - low) inside the interval, else 0.

        :param x: Parameter value
        :type x: float
        :rtype: float
        """

        if self.low <= x <= self.upp:
            return 1 / (self.upp - self.low)
        else:
            return 0.0

    def logprior(self, x):
        """Log prior at `x`, up to a constant: 0 inside the interval, -inf outside.

        :param x: Parameter value
        :type x: float
        :rtype: float
        """

        if self.low <= x <= self.upp:
            return 0.0
        else:
            return -np.inf

    def _draw(self, rng, n):
        return rng.uniform(self.low, self.upp, n)


class GaussianPrior(Prior):
    """Normal prior.

    :param name: Prior name
    :type name: str
    :param mean: Mean
    :type mean: float
    :param sigma: Standard deviation
    :type sigma: float
    """

    def __init__(self, name, mean, sigma):

        self.mean = mean
        self.sigma = sigma

        super().__init__(name, "gaussian", mean=mean, sigma=sigma)

    def evaluate(self, x):
        """Normalised probability density at `x`.

        :param x: Parameter value
        :type x: float or np.ndarray
        """

        return np.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2) / (
            self.sigma * np.sqrt(2 * np.pi)
        )

    def logprior(self, x):
        """Normalised log probability density at `x`.

        :param x: Parameter value
        :type x: float or np.ndarray
        """

        return -0.5 * ((x - self.mean) / self.sigma) ** 2 - np.log(
            self.sigma * np.sqrt(2 * np.pi)
        )

    def _draw(self, rng, n):
        return rng.normal(self.mean, self.sigma, n)
