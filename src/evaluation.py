import random
import shutil
import subprocess
from functools import partial
from pathlib import Path

import jax
from flax import nnx
from omegaconf import DictConfig
import orbax.checkpoint as ocp
import jax.numpy as jnp
from tqdm import tqdm

from src.hnefatafl.hnefatafl import Hnefatafl
from src.mcts import run_mcts
from src.model import HnefataflZeroNet


@partial(nnx.jit, static_argnames=('num_simulations', 'env'))
def _arena_step(model_A, model_B, state, key, num_simulations, env):
    """Executes one batched turn in the arena where two distinct models play each other."""
    key_A, key_B = jax.random.split(key)

    out_A = run_mcts(model_A, state, key_A, num_simulations, env)
    out_B = run_mcts(model_B, state, key_B, num_simulations, env)

    action = jnp.where(state.current_player == 0, out_A.action, out_B.action)

    next_state = jax.vmap(env.step)(state, action)
    return next_state


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
        self.base_state = self._init_eval_state()

    def _init_eval_state(self):
        batch_size = self.cfg.train.batch_size
        half = batch_size // 2
        key_env = jax.random.split(self.rngs.split(), batch_size)
        state = jax.jit(jax.vmap(self.env.init))(key_env)

        player_order = jnp.concatenate([
            jnp.tile(jnp.array([0, 1]), (half, 1)),
            jnp.tile(jnp.array([1, 0]), (batch_size - half, 1))
        ], axis=0)

        return state.replace(
            _player_order=player_order,
            current_player=player_order[:, 0]
        )

    def evaluate_model(self, iteration: int) -> int:
        """Evaluates the main model against a random past checkpoint model with BayesElo."""
        env_state = self.base_state

        opponent = self._load_random_opponent()
        if not opponent:
            print("Skipping evaluation")
            self._add_to_eval_pool(iteration)
            self.save_eval_pool()
            return 0 # Baseline starting Elo

        self.model.eval()
        self.eval_model.eval()

        num_simulations = self.cfg.mcts.simulations

        pbar = tqdm(total=512, desc=f"Arena: Iter_{iteration} vs {opponent}", mininterval=self.cfg.train.tqdm_interval, ncols=100)

        while not jnp.all(env_state.terminated):
            rng_key = self.rngs.split()
            env_state = _arena_step(self.model, self.eval_model, env_state, rng_key, num_simulations, self.env)
            pbar.update(1)
            if env_state.terminated.all():
                break
        pbar.close()

        # Collect results
        rewards = jax.device_get(env_state.rewards)
        match_data = []
        current_model = f"iter_{iteration}"
        batch_size = self.cfg.train.batch_size
        half = batch_size // 2

        for i in range(half):
            r = rewards[i, 0]
            match_data.append((current_model, opponent, r))

        for i in range(half, batch_size):
            r = rewards[i, 1]
            match_data.append((opponent, current_model, r))

        # Output PGN and run BayesElo
        self._generate_minimal_pgn(match_data)
        ratings = self._run_bayeselo()

        elo = ratings.get(current_model, 0)

        # Add the new iteration to the eval pool
        self._add_to_eval_pool(iteration)
        self.save_eval_pool()

        return elo

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
            pool[ckpt_dir.name] = jax.device_get(restored_state) # Move to CPU

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