from functools import partial

import jax
import mctx
import pgx
from flax import nnx
import jax.numpy as jnp


def recurrent_fn(model_state, rng_key: jax.Array, action: jax.Array,
                 embedding, env, graph_def):
    next_game_state = jax.vmap(env.game.mcts_step)(embedding._x, action)

    next_state = embedding.replace(
        _x=next_game_state,
        terminated=jax.vmap(env.game.is_terminal)(next_game_state),
        rewards=jax.vmap(env.game.rewards)(next_game_state)[jnp.arange(action.shape[0]), embedding._player_order],
        current_player=embedding._player_order[jnp.arange(action.shape[0]), next_game_state.color]
    )

    next_obs = jax.vmap(env.game.observe)(next_game_state)
    local_model = nnx.merge(graph_def, model_state)
    logits, value = local_model(next_obs)

    rewards = next_state.rewards[jnp.arange(next_state.rewards.shape[0]), embedding.current_player]
    discounts = jnp.where(next_state.terminated, 0.0, 1.0)

    output = mctx.RecurrentFnOutput(
        reward=rewards,
        discount=discounts,
        prior_logits=logits,
        value=value
    )

    return output, next_state


def run_mcts(model: nnx.Module, env_state, rng_key: jax.Array, num_simulations: int, env: pgx.Env):
    if env_state.observation.ndim == 3:
        env_state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), env_state)

    graph_def, state = nnx.split(model)
    root_logits, root_value = model(env_state.observation, train=False)

    root = mctx.RootFnOutput(
        prior_logits=root_logits,
        value=root_value,
        embedding=env_state
    )

    rec_fn = partial(recurrent_fn, env=env, graph_def=graph_def)

    policy_output = mctx.gumbel_muzero_policy(
        params=state,
        rng_key=rng_key,
        root=root,
        recurrent_fn=rec_fn,
        num_simulations=num_simulations,
        invalid_actions=~env_state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value
    )

    return policy_output
