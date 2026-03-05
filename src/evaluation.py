import random
import shutil
import subprocess
from functools import partial
from pathlib import Path

import jax
from flax import nnx
from jax import lax
from omegaconf import DictConfig
import orbax.checkpoint as ocp
import jax.numpy as jnp
from tqdm import tqdm

from src.hnefatafl.hnefatafl import Hnefatafl
from src.mcts import run_mcts
from src.model import HnefataflZeroNet


@partial(nnx.jit, static_argnames=('num_simulations', 'env'))
def evaluate(model_A, model_B, state, rng_key, num_simulations, env):
    """
    Evaluates both roles in parallel using vmap.
    state: Shape (2, Batch, ...) where state[0] is P0=Attacker, state[1] is P0=Defender
    """
    graph_def_A, model_A_state = nnx.split(model_A)
    graph_def_B, model_B_state = nnx.split(model_B)

    def run_one_role(single_role_state, role_key, role_idx):
        def step_fn(loop_vars):
            step_state, key, t_mask, rewards = loop_vars

            is_terminal = step_state.terminated
            should_update = is_terminal & ~t_mask

            # Rewards are captured once when the game first hits terminal
            next_rewards = jnp.where(should_update[:, None], step_state.rewards, rewards)
            next_mask = t_mask | is_terminal

            key_A, key_B, key_env, next_key = jax.random.split(key, 4)

            # All games in this role-batch are synchronized on turn, even if some are terminal,
            # because we use env._step which always flips color.
            is_p0_turn = (step_state.current_player[0] == 0)

            from src.mcts import run_mcts_functional
            action = lax.cond(
                is_p0_turn,
                lambda: run_mcts_functional(graph_def_A, model_A_state, step_state, key_A, num_simulations, env).action,
                lambda: run_mcts_functional(graph_def_B, model_B_state, step_state, key_B, num_simulations, env).action
            )

            def update_pbar_role(k):
                global _eval_pbar
                if '_eval_pbar' in globals():
                    _eval_pbar.update(1)

            # Only update progress bar from role 0 to avoid double counting
            lax.cond(role_idx == 0, lambda: jax.debug.callback(update_pbar_role, None), lambda: None)

            # Use _step directly to bypass pgx's auto-reset and turn-stalling on terminal states.
            # We pass keys but our _step ignores them.
            keys_env = jax.random.split(key_env, step_state.current_player.shape[0])
            return jax.vmap(env._step)(step_state, action, keys_env), next_key, next_mask, next_rewards

        def cond_fn(loop_vars):
            step_state, _, t_mask, _ = loop_vars
            # Safety stop at 512, though is_terminal should already trigger it
            not_all_terminated = ~jnp.all(t_mask)
            within_limits = jnp.all(step_state._x.step_count < 512)
            return not_all_terminated & within_limits

        termination_mask = jnp.zeros_like(single_role_state.terminated, dtype=jnp.bool_)
        init_rewards = jnp.zeros_like(single_role_state.rewards)
        init_loop_vars = (single_role_state, role_key, termination_mask, init_rewards)
        _, _, _, final_rewards = lax.while_loop(cond_fn, step_fn, init_loop_vars)
        return final_rewards

    keys = jax.random.split(rng_key, 2)
    return jax.vmap(run_one_role)(state, keys, jnp.arange(2))


class Evaluator:
    def __init__(self, cfg: DictConfig, dirs: dict[str, Path], rngs: nnx.Rngs,
                 model: nnx.Module, checkpointer: ocp.StandardCheckpointer, env: Hnefatafl):
        self.cfg = cfg
        self.dirs = dirs
        self.rngs = rngs
        self.model = model
        self.checkpointer = checkpointer
        self.env = env
        self.eval_pool = self._load_eval_pool(cfg.train.load_checkpoint)
        self._add_to_eval_pool(iteration=0) # Add a baseline random model to the eval pool

    def _init_eval_state(self):
        """Initializes a 2D state [Role, Batch, ...] for parallel evaluation."""
        batch_size = self.cfg.train.batch_size
        half = batch_size // 2
        
        def get_state(p0_is_attacker, num_games):
            key_env = jax.random.split(self.rngs.split(), num_games)
            state = jax.jit(jax.vmap(self.env.init))(key_env)
            if p0_is_attacker:
                player_order = jnp.tile(jnp.array([0, 1]), (num_games, 1))
            else:
                player_order = jnp.tile(jnp.array([1, 0]), (num_games, 1))
            return state.replace(_player_order=player_order, current_player=player_order[:, 0])

        state_p0_atk = get_state(True, half)
        state_p0_def = get_state(False, batch_size - half)

        return jax.tree_util.tree_map(lambda x, y: jnp.stack([x, y]), state_p0_atk, state_p0_def)

    def evaluate_model(self, iteration: int) -> int:
        """Evaluates the main model against a random past checkpoint model with BayesElo."""
        current_model = f"iter_{iteration}"
        opponent = self._load_random_opponent()
        if not opponent:
            print("Skipping evaluation")
            self._add_to_eval_pool(iteration)
            return 0  # Baseline starting Elo

        self.model.eval()
        self.eval_model.eval()

        env_state = self._init_eval_state()

        global _eval_pbar
        _eval_pbar = tqdm(total=512, desc='Evaluation', mininterval=self.cfg.train.tqdm_interval,
                          ncols=100, unit='steps')

        rewards = jax.device_get(evaluate(
            model_A=self.model,
            model_B=self.eval_model,
            state=env_state,
            rng_key=self.rngs.default(),
            num_simulations=self.cfg.mcts.simulations,
            env=self.env
        ))
        _eval_pbar.close()

        # Collect results
        match_data = self._get_eval_metrics(rewards, opponent, current_model)

        # Output PGN and run BayesElo
        self._generate_minimal_pgn(match_data)
        ratings = self._run_bayeselo()

        elo = ratings.get(current_model, 0)

        # Add the new iteration to the eval pool
        self._add_to_eval_pool(iteration)

        return elo

    def _get_eval_metrics(self, rewards: jax.Array, opponent: str, current_model: str):
        """
        Extracts games' metrics from the rewards array of shape (2, Half, 2).
        Returns metrics dict for BayesElo and logs evaluation results
        """
        match_data = []
        batch_size = self.cfg.train.batch_size
        half = batch_size // 2

        p0_stats = {"wins": 0, "losses": 0, "draws": 0}
        p1_stats = {"wins": 0, "losses": 0, "draws": 0}

        rewards_atk = rewards[0]
        for i in range(half):
            rew = rewards_atk[i, 0]
            match_data.append((current_model, opponent, rew))
            if rew == 1:
                p0_stats["wins"] += 1
            elif rew == -1:
                p0_stats["losses"] += 1
            else:
                p0_stats["draws"] += 1

        rewards_def = rewards[1]
        for i in range(batch_size - half):
            rew = rewards_def[i, 0]
            match_data.append((opponent, current_model, rew))
            if rew == 1:
                p1_stats["wins"] += 1
            elif rew == -1:
                p1_stats["losses"] += 1
            else:
                p1_stats["draws"] += 1

        print(f"\n{current_model} vs {opponent}:")
        print(f"  As P0 (Attacker): {p0_stats['wins']}W/{p0_stats['losses']}L/{p0_stats['draws']}D")
        print(f"  As P1 (Defender): {p1_stats['wins']}W/{p1_stats['losses']}L/{p1_stats['draws']}D")

        return match_data

    def _generate_minimal_pgn(self, match_data):
        """
        match_data: A list of tuples containing (Attacker_Name, Defender_Name, Result)
                    Result must be: 1 (Attacker wins), -1 (Defender wins), or 0 (Draw)
        """
        with open(self.dirs['pgn'], "a") as f:
            for attacker, defender, reward in match_data:
                if reward == 1:
                    result_str = "1-0"
                elif reward == -1:
                    result_str = "0-1"
                else:
                    result_str = "1/2-1/2"

                f.write(f'[White "{attacker}"]\n')
                f.write(f'[Black "{defender}"]\n')
                f.write(f'[Result "{result_str}"]\n')

                # BayesElo requires at least one valid chess move to parse the block
                f.write("1. d4 d5\n\n")

    def _run_bayeselo(self):
        """
        Runs BayesElo on the PGN file, computes maximum-likelihood ratings.
        Returns a dictionary of Model -> Elo Rating.
        """
        commands = f"""readpgn {self.dirs['pgn']}
        elo
        mm
        exactdist
        ratings
        x
        x
        """

        process = subprocess.Popen(
            [self.dirs['bayeselo']],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(commands)

        # Parse the ASCII table output
        ratings = {}
        parsing_table = False

        for line in stdout.splitlines():
            if "Rank Name" in line:
                parsing_table = True
                continue

            if parsing_table:
                parts = line.split()
                if not parts or not parts[0].isdigit():
                    if "ResultSet" in line:
                        break
                    continue

                # Extract name and Elo
                name = parts[1]
                elo = int(parts[2])
                ratings[name] = elo

        return ratings

    def _load_eval_pool(self, load_checkpoint: bool) -> dict[str, HnefataflZeroNet]:
        """Load up to `max_pool_size` models from disk into a CPU dictionary."""
        pool = {}
        dir_path = self.dirs['eval_pool']
        if not load_checkpoint:
            shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
            return pool

        _, abstract_state = nnx.split(
            HnefataflZeroNet(depth=self.cfg.model.depth,
                             filter_count=self.cfg.model.filter_count, rngs=self.rngs)
        )
        dirs = [d for d in dir_path.iterdir() if d.is_dir()]

        for ckpt_dir in dirs[-self.cfg.train.max_eval_pool:]:
            restored_state = self.checkpointer.restore(ckpt_dir.resolve())
            pool[ckpt_dir.name] = jax.device_get(restored_state)  # Move to CPU

        print(f"Loaded {len(pool)} models into the evaluation pool.")
        return pool

    def _add_to_eval_pool(self, iteration: int) -> None:
        """Snapshot the main model and randomly replace an old one if full."""
        _, current_state = nnx.split(self.model)
        model_name = f"iter_{iteration}"

        if len(self.eval_pool) >= self.cfg.train.max_eval_pool:
            candidates = [name for name in self.eval_pool.keys()]
            victim_name = random.choice(candidates)
            del self.eval_pool[victim_name]
            victim_path = self.dirs['eval_pool'] / victim_name
            if victim_path.exists():
                shutil.rmtree(victim_path)

        self.eval_pool[model_name] = jax.device_get(current_state)

    def save_eval_pool(self) -> None:
        """Save any new models in the dict to disk."""
        if not hasattr(self, 'eval_pool') or not self.eval_pool:
            return

        for model_name, cpu_state in self.eval_pool.items():
            save_path = self.dirs['eval_pool'] / model_name

            if not save_path.exists():
                self.checkpointer.save(save_path.resolve(), cpu_state)

        self.checkpointer.wait_until_finished()

    def _load_random_opponent(self) -> str | None:
        """Selects a random model from the eval pool and loads it into the elo_model."""
        if not hasattr(self, 'eval_pool') or not self.eval_pool:
            print("Eval pool is empty! Cannot load an opponent.")
            return None

        opponent_name = random.choice(list(self.eval_pool.keys()))
        cpu_state = self.eval_pool[opponent_name]
        gpu_state = jax.device_put(cpu_state)

        graph_def, _ = nnx.split(self.model)
        self.eval_model = nnx.merge(graph_def, gpu_state)
        self.eval_model.eval()

        return opponent_name
