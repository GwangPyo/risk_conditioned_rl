import jax.numpy as jnp
import jax

Array = jax.Array


@jax.jit
def cvar(x: Array, alpha: float) -> Array:
    """
    Conditional Value at Risk (CVaR) distortion function.

    :param x: quantile level (0 <= x <= 1)
    :param alpha: CVaR distortion parameter
                  alpha < 1  : risk-averse
                  alpha = 1  : no distortion
                  alpha > 1  : risk-seeking
    :return: distorted quantile level
    """
    return jnp.where(alpha <= 1, x * alpha, (alpha - 1) + x * (alpha - 1))


@jax.jit
def wang(x: Array, eta: float) -> Array:
    """
    Wang transform distortion function.

    :param x: quantile level (0 <= x <= 1)
    :param eta: risk parameter (shift in normal space)
                eta > 0 : risk-averse
                eta < 0 : risk-seeking
    :return: distorted quantile level
    """
    x = (x + 1e-6) * (1 - 1e-6)  # stabilize input for ppf
    transform = jax.scipy.stats.norm.ppf(x) + eta
    return jax.scipy.stats.norm.cdf(transform)


@jax.jit
def pow(x: Array, eta: float) -> Array:
    """
    Power distortion function.

    :param x: quantile level (0 <= x <= 1)
    :param eta: distortion exponent
                eta > 0 : risk-averse
                eta < 0 : risk-seeking
    :return: distorted quantile level
    """
    exponent = 1 / (1 + jnp.abs(eta))
    return jnp.where(eta > 0, jnp.power(x, exponent), 1 - jnp.power(1 - x, exponent))


@jax.jit
def cpw(x: Array, eta: float) -> Array:
    """
    Cumulative Prospect Weighting (CPW) function.

    :param x: quantile level (0 <= x <= 1)
    :param eta: shape parameter
                eta > 1 : inverse S-shape
                eta < 1 : S-shape
    :return: distorted quantile level
    """
    denominator = (x ** eta + (1 - x) ** eta) ** (1 / eta)
    return (x ** eta) / denominator


@jax.jit
def cpw(x: jax.Array, eta: float):
    denominator = (x ** eta + (1 - x) ** eta) ** (1 / eta)
    return (x ** eta) / denominator


risk_measures = {"cpw": cpw, "wang": wang, "pow": pow, "cvar": cvar}
