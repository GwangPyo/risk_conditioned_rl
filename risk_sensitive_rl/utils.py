from flax import nnx
import jax


def copy_param(model) -> nnx.Param:
    graph_def, params, _ = nnx.split(model, nnx.Param, ...)
    copied_param = jax.tree.map(lambda x: x.copy(), params)
    return copied_param
