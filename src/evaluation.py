import random
import shutil
import subprocess
from functools import partial
from pathlib import Path

import jax
from flax import nnx
from jax import numpy as jnp, lax
from omegaconf import DictConfig
import orbax.checkpoint as ocp
from tqdm import tqdm

from src.hnefatafl.hnefatafl import Hnefatafl
from src.mcts import run_mcts
from src.model import HnefataflZeroNet


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

    def _init_eval_state(self, is_starter: bool):
        """
        Initializes a state where the current model is starter is is_starter is true.
        Returns the state.
        """
        batch_size = self.cfg.train.batch_size // 4

        key_env = jax.random.split(self.rngs.split(), batch_size)
        state = jax.jit(jax.vmap(self.env.init))(key_env)

        if is_starter:
            player_order = jnp.tile(jnp.array([0, 1]), (batch_size, 1))
        else:
            player_order = jnp.tile(jnp.array([1, 0]), (batch_size, 1))

        return state.replace(
            _player_order=player_order,
            current_player=player_order[:, 0]
        )

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

        p0_state = self._init_eval_state(is_starter=True)
        p1_state = self._init_eval_state(is_starter=False)

        global _eval_pbar
        _eval_pbar = tqdm(total=1024, desc='Evaluation', mininterval=self.cfg.train.tqdm_interval,
                          ncols=100, unit='steps')

        rewards_p0 = jax.device_get(evaluate(
            model_A=self.model,
            model_B=self.eval_model,
            state=p0_state,
            rng_key=self.rngs.default(),
            num_simulations=self.cfg.mcts.simulations,
            env=self.env,
            batch_size=self.cfg.train.batch_size // 4,
            dirichlet_fraction=self.cfg.train.dirichlet_fraction
        ))
        rewards_p1 = jax.device_get(evaluate(
            model_A=self.model,
            model_B=self.eval_model,
            state=p1_state,
            rng_key=self.rngs.default(),
            num_simulations=self.cfg.mcts.simulations,
            env=self.env,
            batch_size=self.cfg.train.batch_size // 4,
            dirichlet_fraction=self.cfg.train.dirichlet_fraction
        ))
        _eval_pbar.close()

        # Collect results
        match_data = self._get_eval_metrics(rewards_p0, rewards_p1, opponent, current_model)

        # Output PGN and run BayesElo
        self._generate_minimal_pgn(match_data)
        ratings = self._run_bayeselo()

        elo = ratings.get(current_model, 0)

        # Add the new iteration to the eval pool
        self._add_to_eval_pool(iteration)

        return elo

    @staticmethod
    def _get_eval_metrics(rewards_p0: jax.Array, rewards_p1: jax.Array, opponent: str, current_model: str):
        """
        Extracts games' metrics from the rewards array of shape (2, Half, 2).
        Returns metrics dict for BayesElo and logs evaluation results
        """
        match_data = []

        p0_stats = {"wins": 0, "losses": 0, "draws": 0}
        p1_stats = {"wins": 0, "losses": 0, "draws": 0}

        for i in range(len(rewards_p0)):
            rew = rewards_p0[i, 0]
            match_data.append((current_model, opponent, rew))
            if rew == 1: p0_stats["wins"] += 1
            elif rew == -1: p0_stats["losses"] += 1
            else: p0_stats["draws"] += 1

        for i in range(len(rewards_p1)):
            rew = rewards_p1[i, 0]
            match_data.append((opponent, current_model, -rew))
            if rew == 1: p1_stats["wins"] += 1
            elif rew == -1: p1_stats["losses"] += 1
            else: p1_stats["draws"] += 1

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


@partial(nnx.jit, static_argnames=('num_simulations', 'env', 'batch_size', 'dirichlet_fraction'))
def evaluate(model_A, model_B, state, rng_key, num_simulations, env, batch_size, dirichlet_fraction):
    graph_def_A, model_A_state = nnx.split(model_A)
    graph_def_B, model_B_state = nnx.split(model_B)

    def step_fn(loop_vars):
        step_state, key, t_mask, rewards = loop_vars

        is_terminal = step_state.terminated
        should_update = is_terminal & ~t_mask

        next_rewards = jnp.where(should_update[:, None], step_state.rewards, rewards)
        next_mask = t_mask | is_terminal

        key_A, key_B, next_key = jax.random.split(key, 3)

        is_p0 = (step_state.current_player[0] == 0)

        action = lax.cond(
            is_p0,
            lambda: run_mcts(graph_def_A, model_A_state, step_state, key_A, num_simulations, env,
                             step_state.current_player, batch_size, dirichlet_fraction, attacker_explore=False).action,
            lambda: run_mcts(graph_def_B, model_B_state, step_state, key_B, num_simulations, env,
                             step_state.current_player, batch_size, dirichlet_fraction, attacker_explore=False).action
        )

        def update_pbar(_):
            global _eval_pbar
            if '_eval_pbar' in globals():
                _eval_pbar.update(1)

        jax.debug.callback(update_pbar, key)

        return jax.vmap(env.step)(step_state, action), next_key, next_mask, next_rewards

    def cond_fn(loop_vars):
        _, _, t_mask, _ = loop_vars
        return ~jnp.all(t_mask)

    termination_mask = jnp.zeros_like(state.terminated, dtype=jnp.bool_)
    init_rewards = jnp.zeros_like(state.rewards)
    init_loop_vars = (state, rng_key, termination_mask, init_rewards)
    _, _, _, final_rewards = lax.while_loop(cond_fn, step_fn, init_loop_vars)

    return final_rewards
