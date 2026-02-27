import jax
import jax.numpy as jnp

import pgx.core as core
from src.hnefatafl.hnefatafl_jax import Game, GameState, INIT_LEGAL_ACTION_MASK, _flip
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = INIT_LEGAL_ACTION_MASK
    observation: Array = jnp.zeros((11, 11, 43), dtype=jnp.float32)
    _step_count: Array = jnp.int32(0)
    _player_order: Array = jnp.int32([0, 1])  # [0, 1] or [1, 0]
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "hnefatafl"


class Hnefatafl(core.Env):
    def __init__(self):
        super().__init__()
        self.game = Game()

    def _init(self, key):
        x = GameState()
        _player_order = jnp.array([[0, 1], [1, 0]])[jax.random.bernoulli(key).astype(jnp.int32)]
        state = State(  # type: ignore
            current_player=_player_order[(x.color + 1) // 2],
            _player_order=_player_order,
            _x=x,
        )
        return state

    def _step(self, state: State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        x = self.game.step(state._x, action)
        state = state.replace(  # type: ignore
            _x=x,
            legal_action_mask=x.legal_action_mask,
            terminated=self.game.is_terminal(x),
            rewards=self.game.rewards(x)[state._player_order],
            current_player=state._player_order[x.color],
        )
        return state  # type: ignore

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        x = jax.lax.cond(state.current_player == player_id, lambda: state._x, lambda: _flip(state._x))
        return self.game.observe(x)

    @property
    def id(self):
        return "hnefatafl"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2