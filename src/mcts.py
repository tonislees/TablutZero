from functools import partial

import jax
import mctx
import pgx
from flax import nnx
import jax.numpy as jnp

from src.utils import policy_value_by_player


def recurrent_fn(model_state, rng_key: jax.Array, action: jax.Array,
                 embedding, env, graph_def, reward_consts: jax.Array):
    next_game_state = jax.vmap(env.game.step)(embedding._x, action)

    batch_idx = jnp.arange(action.shape[0])[:, None]
    color_idx = (next_game_state.color + 1) // 2

    is_term, raw_rewards = jax.vmap(env.game.mcts_status)(next_game_state)

    r_a_win, r_a_loss, r_d_win, r_d_loss = reward_consts
    att_raw = raw_rewards[:, 0]
    def_raw = raw_rewards[:, 1]

    scaled_att = jnp.where(att_raw > 0, r_a_win, jnp.where(att_raw < 0, r_a_loss, 0.0))
    scaled_def = jnp.where(def_raw > 0, r_d_win, jnp.where(def_raw < 0, r_d_loss, 0.0))
    scaled_rewards = jnp.stack([scaled_att, scaled_def], axis=1)

    next_state = embedding.replace(
        _x=next_game_state,
        terminated=is_term,
        rewards=scaled_rewards[batch_idx, embedding._player_order],
        current_player=embedding._player_order[jnp.arange(action.shape[0]), color_idx]
    )

    next_obs = jax.vmap(env.game.observe)(next_game_state)

    local_model = nnx.merge(graph_def, model_state)
    role = (next_game_state.color + 1) // 2
    logits, value = policy_value_by_player(local_model(next_obs), role)

    rewards = next_state.rewards[jnp.arange(next_state.rewards.shape[0]), embedding.current_player]
    discounts = jnp.where(next_state.terminated, 0.0, -1.0)

    output = mctx.RecurrentFnOutput(
        reward=rewards,
        discount=discounts,
        prior_logits=logits,
        value=value
    )

    return output, next_state


def run_mcts(graph_def, model_state, env_state, rng_key: jax.Array, num_simulations: int, env: pgx.Env,
             batch_size: int, dirichlet_fraction, attacker_explore: bool = True,
             reward_consts: jax.Array = jnp.array([1.0, -1.0, 1.0, -1.0])):
    if env_state.observation.ndim == 3:
        env_state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), env_state)

    # Root inference
    local_model = nnx.merge(graph_def, model_state)
    role = (env_state._x.color + 1) // 2
    root_logits, root_value = policy_value_by_player(local_model(env_state.observation, train=False), role)

    is_attacker = (env_state._x.color == -1)

    legal_mask = env_state.legal_action_mask
    apply_dirichlet = is_attacker & attacker_explore
    num_actions = root_logits.shape[-1]

    noise_key, rng_key = jax.random.split(rng_key)
    dirichlet_noise = jax.random.dirichlet(
        noise_key,
        alpha=jnp.full((num_actions,), 0.3),
        shape=(batch_size,)
    )

    probs = jax.nn.softmax(root_logits)
    mixed_probs = (1 - dirichlet_fraction) * probs + dirichlet_fraction * dirichlet_noise

    mixed_probs = jnp.where(legal_mask, mixed_probs, 1e-8)
    mixed_probs = mixed_probs / jnp.sum(mixed_probs, axis=-1, keepdims=True)

    noisy_logits = jnp.log(mixed_probs)

    final_root_logits = jnp.where(apply_dirichlet[:, None], noisy_logits, root_logits)

    root = mctx.RootFnOutput(
        prior_logits=final_root_logits,
        value=root_value,
        embedding=env_state
    )

    rec_fn = partial(recurrent_fn, env=env, graph_def=graph_def, reward_consts=reward_consts)

    g_scale = jnp.where(apply_dirichlet[:, None], 2.0, 1.0)

    policy_output = mctx.gumbel_muzero_policy(
        params=model_state,
        rng_key=rng_key,
        root=root,
        recurrent_fn=rec_fn,
        num_simulations=num_simulations,
        invalid_actions=~env_state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=g_scale
    )

    return policy_output
