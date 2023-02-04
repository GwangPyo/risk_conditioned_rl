import numpy as np


class SampleY(object):
    def __init__(self, n_quantiles):
        self.n_quantiles = n_quantiles

    def _sample(self, batch_size):
        taus = np.random.uniform(size=(batch_size, self.n_quantiles))
        taus.sort(axis=-1)
        ys = np.random.uniform(size=(batch_size, self.n_quantiles))
        ys.sort(axis=-1)
        # (minima) __scale__ (maxima)
        # |-------------------------|
        # 0                         1

        minima = np.random.uniform(size=(batch_size, 1))
        maxima = 1. - (1. - minima) * np.random.uniform(size=minima.shape)
        scale = maxima - minima
        ys = scale * ys + minima
        return taus, ys

    def sample(self, batch_size):
        t, y = self._sample(batch_size)
        return t, y
