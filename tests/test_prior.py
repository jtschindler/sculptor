"""Tests for `sculptor.prior`.

Regression: `sample(seed=...)` used to ignore the seed (the `rng is None`
branch always created an unseeded generator first).
"""

import numpy as np
import pytest

from sculptor.prior import GaussianPrior, Prior, UniformPrior

PRIORS = [UniformPrior("u", -1.0, 2.0), GaussianPrior("g", 0.5, 0.1)]


@pytest.mark.parametrize("prior", PRIORS, ids=["uniform", "gaussian"])
def test_seed_makes_samples_reproducible(prior):
    a = prior.sample(100, seed=7)
    b = prior.sample(100, seed=7)
    c = prior.sample(100, seed=8)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("prior", PRIORS, ids=["uniform", "gaussian"])
def test_rng_is_used_and_advanced(prior):
    rng = np.random.default_rng(3)
    first = prior.sample(10, rng=rng)
    second = prior.sample(10, rng=rng)

    assert np.array_equal(first, prior.sample(10, seed=3))
    assert not np.array_equal(first, second)


def test_rng_and_seed_together_is_an_error():
    with pytest.raises(ValueError, match="either"):
        PRIORS[0].sample(5, rng=np.random.default_rng(1), seed=1)


def test_samples_follow_the_distribution():
    uniform = UniformPrior("u", -1.0, 2.0).sample(10_000, seed=0)
    gaussian = GaussianPrior("g", 0.5, 0.1).sample(10_000, seed=0)

    assert uniform.min() >= -1.0 and uniform.max() <= 2.0
    assert uniform.mean() == pytest.approx(0.5, abs=0.05)
    assert gaussian.mean() == pytest.approx(0.5, abs=0.01)
    assert gaussian.std() == pytest.approx(0.1, rel=0.05)


def test_base_prior_cannot_sample():
    with pytest.raises(NotImplementedError):
        Prior("p", "custom").sample(3, seed=0)


def test_uniform_logprior_is_float():
    prior = UniformPrior("u", 0.0, 1.0)

    assert prior.logprior(0.5) == 0.0 and isinstance(prior.logprior(0.5), float)
    assert prior.logprior(2.0) == -np.inf


def test_uniform_prior_takes_arrays():
    prior = UniformPrior("u", 0.0, 2.0)
    np.testing.assert_array_equal(
        prior.logprior(np.array([-1, 0, 1, 3])), [-np.inf, 0, 0, -np.inf]
    )
    np.testing.assert_array_equal(prior.evaluate(np.array([1.0, 5.0])), [0.5, 0.0])


def test_ppf_inverts_the_distribution():
    uniform, gaussian = UniformPrior("u", -1.0, 3.0), GaussianPrior("g", 2.0, 0.5)
    np.testing.assert_allclose(uniform.ppf([0.0, 0.25, 1.0]), [-1.0, 0.0, 3.0])
    assert gaussian.ppf(0.5) == pytest.approx(2.0)
    assert gaussian.ppf(0.8413447460685429) == pytest.approx(2.5)
    with pytest.raises(NotImplementedError):
        Prior("p", "custom").ppf(0.5)
