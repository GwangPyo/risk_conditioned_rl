import distrax
from base.jax_layers import create_mlp
from flax import nnx
import jax


class CondFlow(nnx.Module):
    def __init__(self,
                 input_dim: int,
                 n_bins: int,
                 layers: int = 4,
                 *,
                 rngs
                 ):

        self.mlp = nnx.Sequential(*create_mlp(input_dim=input_dim, output_dim=256, rngs=rngs))
        self.layers = layers
        self.n_bins = n_bins
        self.layers = [nnx.Linear(256,  3 * n_bins + 1, rngs=rngs) for _ in range(layers)]
        self.scalar_affine = nnx.Linear(256, 4, rngs=rngs)

    def __call__(self, cond, t):
        latent = self.mlp(cond)
        params = [l(latent) for l in self.layers]
        scalar_affine = self.scalar_affine(latent)
        f_1 = distrax.ScalarAffine(scalar_affine[..., [0]], log_scale=scalar_affine[..., [1]])
        f_2 = distrax.ScalarAffine(scalar_affine[..., [2]], log_scale=scalar_affine[..., [3]])
        flows = [distrax.RationalQuadraticSpline(p[..., None, :], range_min=-3, range_max=3,
                                                 min_knot_slope=1e-5, min_bin_size=1e-5) for p in params]
        t = f_1.forward(t)

        flows = distrax.Chain(flows)
        t = flows.forward(t)
        t = f_2.forward(t)

        return jax.scipy.stats.norm.cdf(t)

    def forward_and_log_det(self, cond, t):
        latent = self.mlp(cond)
        params = [l(latent) for l in self.layers]
        scalar_affine = self.scalar_affine(latent)
        f_1 = distrax.ScalarAffine(scalar_affine[..., [0]], log_scale=scalar_affine[..., [1]])
        f_2 = distrax.ScalarAffine(scalar_affine[..., [2]], log_scale=scalar_affine[..., [3]])
        flows = [distrax.RationalQuadraticSpline(p[..., None, :], range_min=-3, range_max=3,
                                                 min_knot_slope=1e-5, min_bin_size=1e-5) for p in params]
        t = f_1.forward(t)

        flows = distrax.Chain(flows)
        t, log_det = flows.forward_and_log_det(t)
        t = f_2.forward(t)

        return jax.scipy.stats.norm.cdf(t), log_det














