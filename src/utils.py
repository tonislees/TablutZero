from functools import partial
from pathlib import Path

import jax
import optax
from flax import nnx
from jax import numpy as jnp

from src.hnefatafl.hnefatafl_jax import ROTATION_PERM


def augment_batch(batch, rng_key):
    k = jax.random.randint(rng_key, (), 0, 4)
    obs = jax.lax.switch(k, [
        lambda x: x,
        lambda x: jnp.rot90(x, 1, axes=(1, 2)),
        lambda x: jnp.rot90(x, 2, axes=(1, 2)),
        lambda x: jnp.rot90(x, 3, axes=(1, 2)),
    ], batch['observation'])
    return {
        **batch,
        'observation': obs,
        'policy_target': batch['policy_target'][:, ROTATION_PERM[k]],
        'legal_action_mask': batch['legal_action_mask'][:, ROTATION_PERM[k]],
    }


def policy_value_by_player(model_outputs: tuple[jax.Array, ...], player: jax.Array) -> tuple[jax.Array, jax.Array]:
    p0_logits, p0_value, p1_logits, p1_value = model_outputs
    logits = jnp.where(player[:, None] == 0, p0_logits, p1_logits)
    value = jnp.where(player == 0, p0_value, p1_value)
    return logits, value


def loss_fn(model: nnx.Module, batch: dict, train=True):
    logits, value = policy_value_by_player(model(batch['observation'], train=train), batch['player'])

    masked_logits = jnp.where(batch['legal_action_mask'], logits, -1e9)
    policy_loss = optax.softmax_cross_entropy(
        logits=masked_logits, labels=batch['policy_target']
    ).mean()
    value_sq = value.squeeze()
    value_loss = optax.l2_loss(
        predictions=value_sq, targets=batch['value_target']
    ).mean()
    total_loss = policy_loss + value_loss

    pred_sign = jnp.sign(jnp.round(value_sq, decimals=1))
    target_sign = jnp.sign(batch['value_target'])
    value_acc = (pred_sign == target_sign).mean()

    return total_loss, (policy_loss, value_loss, value_acc)


def dir_safe(dir_name: str, parent_dir: Path) -> Path:
    """
    Creates the specified directory if it doesn't already exist.
    Returns the directory path.
    """
    dir_ = parent_dir / dir_name
    dir_.mkdir(parents=True, exist_ok=True)
    return dir_


@partial(jax.jit, backend='cpu', static_argnames=('buffer',))
def add_to_buffer_cpu(buffer_state, transitions, buffer):
    def add_step(buf_state, transition_batch):
        return buffer.add(buf_state, transition_batch), None
    new_buffer_state, _ = jax.lax.scan(add_step, buffer_state, transitions)
    return new_buffer_state


@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch: dict, rng_key: jax.Array):
    batch = augment_batch(batch, rng_key)
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (p_loss, v_loss, v_acc)), grads = grad_fn(model, batch)
    optimizer.update(model, grads)
    return loss, p_loss, v_loss, v_acc
