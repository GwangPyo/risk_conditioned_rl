import flax.linen
import jax.random
import optax
from stable_baselines3.common.torch_layers import create_mlp
from typing import List, Optional, Callable, Sequence
from flax import nnx
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=2)
def imq_kernel(X: jax.Array,
               Y: jax.Array,
               h_dim: int):
    batch_size = X.shape[0]

    norms_x = (X ** 2).sum(1, keepdims=True)  # batch_size x 1
    prods_x = jnp.matmul(X, X.transpose())  # batch_size x batch_size
    dists_x = norms_x + norms_x.transpose() - 2 * prods_x

    norms_y = (Y ** 2).sum(1, keepdims=True)  # batch_size x 1
    prods_y = jnp.matmul(Y, Y.transpose())  # batch_size x batch_size
    dists_y = norms_y + norms_y.transpose() - 2 * prods_y

    dot_prd = jnp.matmul(X, Y.transpose())
    dists_c = norms_x + norms_y.transpose() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10., ]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        res1 = (1 - jnp.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: Sequence[int] = (256, 256),
        activation_fn: Callable = nnx.relu,
        squash_output: bool = False,
        with_bias: bool = True,
        pre_linear_modules: Optional[Sequence[type[nnx.Module]]] = None,
        post_linear_modules: Optional[Sequence[type[nnx.Module]]] = (nnx.LayerNorm,),
        *,
        rngs: nnx.Rngs,
) -> list[nnx.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim: Dimension of the output (last layer, for instance, the number of actions)
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :param pre_linear_modules: List of nn.Module to add before the linear layers.
        These modules should maintain the input tensor dimension (e.g. BatchNorm).
        The number of input features is passed to the module's constructor.
        Compared to post_linear_modules, they are used before the output layer (output_dim > 0).
    :param post_linear_modules: List of nn.Module to add after the linear layers
        (and before the activation function). These modules should maintain the input
        tensor dimension (e.g. Dropout, LayerNorm). They are not used after the
        output layer (output_dim > 0). The number of input features is passed to
        the module's constructor.
    :return: The list of layers of the neural network
    """
    net_arch = list(net_arch)

    pre_linear_modules = pre_linear_modules or []
    post_linear_modules = post_linear_modules or []
    if isinstance(post_linear_modules, tuple):
        post_linear_modules = list(post_linear_modules)
    if isinstance(pre_linear_modules, tuple):
        pre_linear_modules = list(pre_linear_modules)

    modules = []
    if len(net_arch) > 0:
        # BatchNorm maintains input dim
        for module in pre_linear_modules:
            modules.append(module(input_dim, rngs=rngs))

        modules.append(nnx.Linear(input_dim, net_arch[0], use_bias=with_bias, rngs=rngs))

        # LayerNorm, Dropout maintain output dim
        for module in post_linear_modules:
            modules.append(module(net_arch[0], rngs=rngs))

        modules.append(activation_fn)

    for idx in range(len(net_arch) - 1):
        for module in pre_linear_modules:
            modules.append(module(net_arch[idx], rngs=rngs))

        modules.append(nnx.Linear(net_arch[idx], net_arch[idx + 1], use_bias=with_bias, rngs=rngs))

        for module in post_linear_modules:
            modules.append(module(net_arch[idx + 1], rngs=rngs))

        modules.append(activation_fn)

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        # Only add BatchNorm before output layer
        for module in pre_linear_modules:
            modules.append(module(last_layer_dim, rngs=rngs))

        modules.append(nnx.Linear(last_layer_dim, output_dim, use_bias=with_bias, rngs=rngs))
    if squash_output:
        modules.append(nnx.tanh)
    return modules


class FourierFeatureNetwork(nnx.Module):
    def __init__(self, input_dim, output_dim,
                 stddev: float = 1e-3,
                 *,
                 rngs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ff = nnx.Linear(input_dim, output_dim // 2,
                             use_bias=False,
                             rngs=rngs, kernel_init=nnx.initializers.normal(stddev * jnp.pi * 2))
        self.in_ = nnx.Linear(input_dim, output_dim, rngs=rngs)
        self.out = nnx.Linear(output_dim * 2, output_dim, rngs=rngs)

    def __call__(self, x):
        ff_x = self.ff(x)
        x = self.in_(x)
        cat = jnp.concatenate([jnp.sin(ff_x), jnp.cos(ff_x), x], axis=-1)
        return self.out(cat)


class IQNHead(nnx.Module):
    def __init__(self,
                 features_dim,
                 net_arch: Sequence[int] = (64, 64),
                 n_cos: int = 64,
                 *,
                 rngs
                 ):
        self.features_dim = features_dim
        self.net_arch = net_arch
        self.n_cos = n_cos
        self.layers = nnx.Sequential(
            *create_mlp(features_dim, 1, net_arch=list(net_arch), rngs=rngs)
        )
        self.taus_embedding = nnx.Sequential(
            FourierFeatureNetwork(1, self.features_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(self.features_dim, self.features_dim, rngs=rngs),
            jnp.sinc
        )

    def __call__(self, features, taus):
        taus = self.taus_embedding(taus[..., None])
        # taus = jnp.swapaxes(taus, -1, -2)
        # taus = taus[..., None, :]
        features = features[..., None, :]

        mult = taus * features

        return self.layers(mult).squeeze(axis=-1)


class ChannelAttention(nnx.Module):
    def __init__(self, in_channels, reduction=16, *, rngs):
        hidden_channels = max(in_channels // reduction, 1)

        self.mlp = nnx.Sequential(
            nnx.Linear(in_channels, hidden_channels, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_channels, in_channels, rngs=rngs)
        )

    def __call__(self, x):
        # x: (..., H, W, C)
        avg = jnp.mean(x, axis=(-3, -2))
        maximum = jnp.max(x, axis=(-3, -2))

        out = self.mlp(avg) + self.mlp(maximum)
        scale = jax.nn.sigmoid(out)[..., None, None, :]

        return x * scale


class SpatialAttention(nnx.Module):
    def __init__(self, kernel_size=(7, 7), *, rngs):
        self.conv = nnx.Conv(2, 1, kernel_size=kernel_size, padding='SAME', rngs=rngs)

    def __call__(self, x):
        avg = jnp.mean(x, axis=-1, keepdims=True)
        maximum = jnp.max(x, axis=-1, keepdims=True)
        concat = jnp.concatenate([avg, maximum], axis=-1)

        scale = jax.nn.sigmoid(self.conv(concat))
        return x * scale


class CBAM(nnx.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=(7, 7), *, rngs):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction, rngs=rngs)
        self.sa = SpatialAttention(kernel_size, rngs=rngs)

    def __call__(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ConvBlock(nnx.Module):
    def __init__(self, in_channel, out_channel,
                 kernel_size: int | Sequence[int] = 3, strides: int | Sequence[int] = 1,
                 kernel_dilation=1,
                 act: Callable = nnx.silu,
                 instance_norm: bool = False,
                 transpose: bool = False,
                 *, rngs):
        # 2d conv
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides)

        # self.mid_chan = (in_channel + out_channel) // 2
        if transpose:
            self.conv = nnx.ConvTranspose(in_channel, out_channel,
                                          kernel_size=kernel_size, strides=strides,
                                          use_bias=False, kernel_dilation=kernel_dilation,
                                          rngs=rngs)

        else:
            self.conv = nnx.Conv(in_channel, out_channel,
                                 kernel_size=kernel_size, strides=strides,
                                 use_bias=False, kernel_dilation=kernel_dilation,
                                 rngs=rngs)
        if instance_norm:
            self.norm = nnx.GroupNorm(out_channel, num_groups=out_channel, rngs=rngs)
        else:
            self.norm = nnx.RMSNorm(out_channel, rngs=rngs)
        self.act = act  # ChannelAttention(out_channel, rngs=rngs)

    def __call__(self, inputs):

        x = self.conv(inputs)
        x = self.norm(x)
        y = self.act(x)
        return y

    def res(self, x):
        if self.res:
            return self.proj(self.pool(x))
        else:
            return x


class FFConv(nnx.Module):
    def __init__(self, in_channel, out_channel, *, rngs):
        super().__init__()
        self.conv = nnx.Conv(in_channel, out_channel // 2, kernel_size=(1, 1),
                             kernel_init=nnx.initializers.normal(stddev=1e-3 * jnp.pi * 2), use_bias=False,
                             rngs=rngs)

    def __call__(self, x):
        y = self.conv(x)
        return jnp.concatenate([jnp.sin(y), jnp.cos(y)], axis=-1)


class ConvolutionLayer(nnx.Module):
    def __init__(self,
                 in_channel: int,
                 out_feature: int,
                 *, rngs
                 ):
        self.seq = nnx.Sequential(

            ConvBlock(in_channel, 32, strides=4, kernel_size=(8, 8),
                      instance_norm=True, rngs=rngs, ),
            CBAM(32, rngs=rngs),
            ConvBlock(32, 64, strides=2, kernel_size=(4, 4),
                      instance_norm=True,
                      rngs=rngs, ),
            CBAM(64, rngs=rngs),
            ConvBlock(64, 128, strides=2, kernel_size=(3, 3), rngs=rngs, ),
        )

        def reshape(x, shape):
            *leading_dims, _ = x.shape
            return x.reshape((*leading_dims, *shape))

        self.reshape_layer = nnx.Sequential(
            nnx.Linear(256, 15 * 20, rngs=rngs),
            partial(reshape, shape=(15, 20, 1)),
        )

        self.out = nnx.Sequential(
            nnx.Conv(1, 64, kernel_size=(1, 1), rngs=rngs),
            nnx.GroupNorm(64, num_groups=64, rngs=rngs), # instance norm.
            nnx.silu,
            ConvBlock(64, 32, strides=2, kernel_size=(3, 3), transpose=True, rngs=rngs),
            ConvBlock(32, 16, strides=1, kernel_size=(3, 3), transpose=True, rngs=rngs),
            nnx.Conv(16, in_channel, kernel_size=(4, 4), rngs=rngs),
        )

        self.mlp = nnx.Sequential(
            nnx.Linear(128, out_feature, rngs=rngs), )

    def __call__(self, x):
        x = self.seq(x)
        attn = jax.nn.softmax(x, axis=(-2, -3))
        avg_pool = jnp.mean(x, axis=(-2, -3))
        attn_pool = (x * attn).sum(axis=(-2, -3))
        x = 0.5 * avg_pool + 0.5 * attn_pool
        return self.mlp(x)

    def wae_loss(self, x, z, key):
        wae_loss = imq_kernel(z, jax.random.normal(key, shape=z.shape), h_dim=z.shape[-1], ) / z.shape[0]
        z = self.reshape_layer(z)
        x_hat = self.out(z)

        x = jax.scipy.fft.dctn(x, axes=(-2, -3), norm='ortho')
        x_resize = x[..., :x_hat.shape[-3], :x_hat.shape[-2], :]

        recon_loss = optax.huber_loss(x_resize, x_hat).mean(axis=(-1, -2, -3))

        return wae_loss, recon_loss


class ConvCriticFeatureExtractor(nnx.Module):
    def __init__(self,
                 in_channel,
                 actions_dim,
                 out_channel=64,
                 *,
                 rngs
                 ):
        self.conv = ConvolutionLayer(in_channel, out_channel, rngs=rngs)
        self.act = nnx.Sequential(*create_mlp(actions_dim, out_channel, net_arch=(64, 64), rngs=rngs))
        self.merge = nnx.Linear(out_channel * 2, out_channel, rngs=rngs)

    def __call__(self, observation, action):
        obs = self.conv(observation)
        act = self.act(action)
        return self.merge(jnp.concatenate([obs, act], axis=-1))


if __name__ == '__main__':
    layer = ConvolutionLayer(11, 64, rngs=nnx.Rngs(42))
    test = jax.random.uniform(jax.random.PRNGKey(42), shape=(2, 120, 160, 11))
    print(layer(test).shape)
    print(flax.linen.max_pool(test, window_shape=(24, 24), strides=(24, 24), padding='SAME').shape)
