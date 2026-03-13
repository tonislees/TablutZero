from functools import partial
from pathlib import Path

import jax
import optax
from flax import nnx
from jax import numpy as jnp


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
    value_loss = optax.l2_loss(
        predictions=value.squeeze(), targets=batch['value_target']
    ).mean()
    total_loss = policy_loss + value_loss
    return total_loss, (policy_loss, value_loss)


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
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch: dict):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (p_loss, v_loss)), grads = grad_fn(model, batch)
    optimizer.update(grads)
    return loss, p_loss, v_loss


def calculate_dynamic_rewards(p_a_win: float, p_d_win: float) -> tuple[float, float, float, float]:
    """
    Calculates dynamic normalizer for results with the formula:
    (p_win x r_win) + (p_loss x r_loss) = 0, where p_win/p_loss are of the advantaged side.
    Returns [r_a_win, r_a_loss, r_d_win, r_d_loss]
    """
    if p_a_win == 0.0 or p_d_win == 0.0:
        return 1.0, -1.0, 1.0, -1.0

    if p_a_win > p_d_win: # Attacker has the advantage
        r_a_loss = -1.0
        r_a_win = p_d_win / p_a_win
        return r_a_win, r_a_loss, -r_a_loss, -r_a_win
    else: # Defender has the advantage
        r_d_loss = -1.0
        r_d_win = p_a_win / p_d_win
        return -r_d_loss, -r_d_win, r_d_win, r_d_loss
