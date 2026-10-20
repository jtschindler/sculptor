import numpy as np
from scipy import stats


class Prior:
    """Base class for parameter priors.

    Subclasses implement `logprior(x)`, `evaluate(x)`, `ppf(q)` (the
    inverse cumulative distribution, e.g. for nested sampling) and
    `_draw(rng, n)` (used by `sample`).

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

    def ppf(self, q):
        """Inverse of the cumulative distribution: the value below which a
        fraction `q` of the prior mass lies (maps the unit cube onto the
        prior, as nested samplers need).

        :param q: Quantile(s) in [0, 1]
        :type q: float or np.ndarray
        :return: Parameter value(s)
        :rtype: float or np.ndarray
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement ppf.")


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

        :param x: Parameter value(s)
        :type x: float or np.ndarray
        :rtype: float or np.ndarray
        """

        if isinstance(x, (float, int, np.number)):  # fast path, per likelihood call
            return 1 / (self.upp - self.low) if self.low <= x <= self.upp else 0.0
        inside = (self.low <= np.asarray(x)) & (np.asarray(x) <= self.upp)
        return np.where(inside, 1 / (self.upp - self.low), 0.0)

    def logprior(self, x):
        """Log prior at `x`, up to a constant: 0 inside the interval, -inf outside.

        Unnormalised (the -ln(upp - low) is left out), unlike
        `GaussianPrior.logprior`; this does not matter for MCMC, but does for
        an evidence (nested sampling).

        :param x: Parameter value(s)
        :type x: float or np.ndarray
        :rtype: float or np.ndarray
        """

        if isinstance(x, (float, int, np.number)):  # fast path, per likelihood call
            return 0.0 if self.low <= x <= self.upp else -np.inf
        inside = (self.low <= np.asarray(x)) & (np.asarray(x) <= self.upp)
        return np.where(inside, 0.0, -np.inf)

    def ppf(self, q):
        """Inverse cumulative distribution: low + q (upp - low).

        :param q: Quantile(s) in [0, 1]
        :type q: float or np.ndarray
        :rtype: float or np.ndarray
        """
        return self.low + np.asarray(q) * (self.upp - self.low)

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

    def ppf(self, q):
        """Inverse cumulative distribution of the normal distribution.

        :param q: Quantile(s) in [0, 1]
        :type q: float or np.ndarray
        :rtype: float or np.ndarray
        """
        return stats.norm.ppf(q, loc=self.mean, scale=self.sigma)

    def _draw(self, rng, n):
        return rng.normal(self.mean, self.sigma, n)
