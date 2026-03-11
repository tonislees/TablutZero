from functools import partial

import jax
from flax import nnx
from jax import numpy as jnp

from src.mcts import run_mcts
from src.utils import policy_value_by_player


@partial(nnx.jit, static_argnames=('num_steps', 'batch_size', 'num_simulations', 'env', 'attacker_explore'))
def self_play(model, env_state, rng_key, num_steps, num_simulations,
              env, batch_size, reward_consts, dirichlet_fraction, attacker_explore):
    graph_def, model_state = nnx.split(model)

    def step_fn(state, key):
        key_reset, key_search = jax.random.split(key)

        mcts_output = run_mcts(graph_def, model_state, state, key_search, num_simulations,
                               env, batch_size, dirichlet_fraction, attacker_explore, reward_consts=reward_consts)
        actions = mcts_output.action
        next_env_state = jax.vmap(env.step)(state, actions)

        # Auto reset if some game is terminal
        reset_keys = jax.random.split(key_reset, batch_size)
        reset_states = jax.vmap(env.init)(reset_keys)

        def select_if_terminated(reset_val, next_val):
            shape = (batch_size,) + (1,) * (next_val.ndim - 1)
            mask = next_env_state.terminated.reshape(shape)
            return jnp.where(mask, reset_val, next_val)

        auto_reset_state = jax.tree_util.tree_map(select_if_terminated, reset_states, next_env_state)

        r_a_win, r_a_loss, r_d_win, r_d_loss = reward_consts

        internal_rewards = jax.vmap(env.game.rewards)(next_env_state._x)
        att_raw = internal_rewards[:, 0]
        def_raw = internal_rewards[:, 1]

        scaled_att = jnp.where(att_raw > 0, r_a_win, jnp.where(att_raw < 0, r_a_loss, 0.0))
        scaled_def = jnp.where(def_raw > 0, r_d_win, jnp.where(def_raw < 0, r_d_loss, 0.0))
        scaled_internal_rewards = jnp.stack([scaled_att, scaled_def], axis=1)

        scaled_player_rewards = jax.vmap(lambda r, order: r[order])(
            scaled_internal_rewards, next_env_state._player_order
        )

        batch_indices = jnp.arange(batch_size)
        current_player_rewards = scaled_player_rewards[batch_indices, state.current_player]
        attacker_rewards = att_raw

        transition = {
            "observation": state.observation,
            "policy_target": mcts_output.action_weights,
            "reward": current_player_rewards,
            "attacker_reward": attacker_rewards,
            "terminated": next_env_state.terminated,
            "legal_action_mask": state.legal_action_mask,
            "player": (state._x.color + 1) // 2
        }

        def update_pbar(_):
            global _self_play_pbar
            if '_self_play_pbar' in globals():
                _self_play_pbar.update(1)

        jax.debug.callback(update_pbar, key)

        return auto_reset_state, transition

    keys = jax.random.split(rng_key, num_steps)
    final_env_state, history = jax.lax.scan(step_fn, env_state, keys)

    # If the game doesn't end in a terminal state, bootstrap using the network's prediction
    _, next_value = policy_value_by_player(model(final_env_state.observation), final_env_state.current_player)

    def step_back(next_return, transition):
        return_ = jnp.where(transition['terminated'], transition['reward'], -next_return)
        out_transition = {
            'observation': transition['observation'],
            'policy_target': transition['policy_target'],
            'value_target': return_,
            'legal_action_mask': transition['legal_action_mask'],
            'player': transition['player']
        }
        return return_, out_transition

    _, final_transitions = jax.lax.scan(step_back, next_value, history, reverse=True)

    return final_env_state, final_transitions, history['terminated'], history['attacker_reward']
