import time
from functools import partial
from pathlib import Path

import flashbax as fbx
import hydra
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import mesh_utils
import jax.numpy as jnp
import optax
from flax import nnx
from omegaconf import DictConfig
import orbax.checkpoint as ocp
from tqdm import tqdm

from src.evaluation import Evaluator
from src.hnefatafl.hnefatafl import Hnefatafl
from src.mcts import run_mcts
from src.metrics import MetricsTracker
from src.model import HnefataflZeroNet


def loss_fn(model, batch, train=True):
    logits, value = model(batch['observation'], train=train)
    policy_loss = optax.softmax_cross_entropy(
        logits=logits, labels=batch['policy_target']
    ).mean()
    value_loss = optax.l2_loss(
        predictions=value.squeeze(), targets=batch['value_target']
    ).mean()
    total_loss = policy_loss + value_loss
    return total_loss, (policy_loss, value_loss)


@nnx.jit
def train_step(model, optimizer, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (p_loss, v_loss)), grads = grad_fn(model, batch)
    optimizer.update(model, grads)
    return loss, p_loss, v_loss


@partial(nnx.jit, static_argnames=('num_steps', 'batch_size', 'num_simulations', 'env', 'buffer'))
def self_play(model, env_state, buffer_state, rng_key, num_steps, num_simulations, env, buffer, batch_size):
    graph_def, model_state = nnx.split(model)

    def step_fn(state, key):
        key_reset, key_search = jax.random.split(key)
        local_model = nnx.merge(graph_def, model_state)

        mcts_output = run_mcts(local_model, state, key_search, num_simulations, env)
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

        # Save temporary states to an array
        batch_indices = jnp.arange(batch_size)
        current_player_rewards = next_env_state.rewards[batch_indices, state.current_player]

        transition = {
            "observation": state.observation,
            "policy_target": mcts_output.action_weights,
            "reward": current_player_rewards,
            "terminated": next_env_state.terminated,
        }
        return auto_reset_state, transition

    keys = jax.random.split(rng_key, num_steps)
    final_env_state, history = jax.lax.scan(step_fn, env_state, keys)

    # If the game doesn't end in a terminal state, bootstrap using the network's prediction
    _, next_value = model(final_env_state.observation)

    def step_back(next_return, transition):
        return_ = jnp.where(transition['terminated'], transition['reward'], -next_return)
        out_transition = {
            'observation': transition['observation'],
            'policy_target': transition['policy_target'],
            'value_target': return_
        }
        return return_, out_transition

    _, final_transitions = jax.lax.scan(step_back, next_value, history, reverse=True)

    def add_to_buffer(buf_state, transition_batch):
        return buffer.add(buf_state, transition_batch), None

    new_buffer_state, _ = jax.lax.scan(add_to_buffer, buffer_state, final_transitions)

    return final_env_state, new_buffer_state


def dir_safe(dir_name: str, parent_dir: Path) -> Path:
    """
    Creates the specified directory if it doesn't already exist.
    Returns the directory path.
    """
    dir_ = parent_dir / dir_name
    dir_.mkdir(parents=True, exist_ok=True)
    return dir_

class Coach:
    def __init__(self, cfg: DictConfig):
        print('Initializing coach...')
        self.cfg = cfg
        self.rngs: nnx.Rngs = nnx.Rngs(cfg.train.seed)
        self.checkpointer = ocp.StandardCheckpointer()

        # Multiple devices
        self.devices = mesh_utils.create_device_mesh((len(jax.devices()),))
        self.mesh = Mesh(self.devices, axis_names=('batch',))
        self.data_sharding = NamedSharding(self.mesh, PartitionSpec('batch'))
        self.replicated_sharding = NamedSharding(self.mesh, PartitionSpec())

        # Directories
        root_dir = self.root = Path(__file__).resolve().parents[1]
        model_dir = dir_safe('models', root_dir)
        metrics_dir = dir_safe('data', root_dir)
        self.dirs = {
            'checkpoints': dir_safe('checkpoints', model_dir),
            'eval_pool': dir_safe('eval_pool', model_dir),
            'plots': dir_safe('plots', metrics_dir),
            'metrics': dir_safe('metrics', metrics_dir),
            'bayeselo': root_dir / 'bayeselo',
            'pgn': root_dir / 'game_results.pgn'
        }
        if not cfg.train.load_checkpoint:
            self.dirs['pgn'].unlink(missing_ok=True)

        # Model & optimizer
        self.model: nnx.Module = self._load_model(self.dirs['checkpoints'], cfg.train.load_checkpoint)
        self.eval_model = None
        graph_def, state = nnx.split(self.model)
        state = jax.tree_util.tree_map(lambda x: jax.device_put(x, self.replicated_sharding), state)
        self.model = nnx.merge(graph_def, state)
        self.optimizer: nnx.Optimizer = nnx.Optimizer(
            self.model, optax.adamw(learning_rate=cfg.train.learning_rate), wrt=nnx.Param
        )

        # Environment
        self.env = Hnefatafl()
        key_env = jax.random.PRNGKey(cfg.train.seed + 1)
        self.env_state = jax.jit(jax.vmap(self.env.init))(
            jax.random.split(key_env, cfg.train.batch_size)
        )
        self.env_state = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, self.data_sharding), self.env_state
        )

        # Setup
        self.last_iteration = self._get_last_iteration() if cfg.train.load_checkpoint else 0
        self.metrics_tracker = MetricsTracker(cfg, self.dirs)
        self.evaluator = Evaluator(cfg, self.dirs, self.rngs, self.model, self.checkpointer, self.env)

        # Buffer
        min_buffer_size = cfg.train.batch_size * cfg.train.self_play_steps
        self.buffer = fbx.make_flat_buffer(
            max_length=min_buffer_size * 8,
            min_length=min_buffer_size,
            sample_batch_size=cfg.train.batch_size,
            add_batch_size=cfg.train.batch_size
        )
        example_transition = {
            "observation": jnp.zeros((11, 11, 43), dtype=jnp.float32),
            "policy_target": jnp.zeros((121 * 40,), dtype=jnp.float32),
            "value_target": jnp.zeros((), dtype=jnp.float32)
        }
        self.buffer_state = self.buffer.init(example_transition)

    def _get_last_iteration(self):
        """
        Finds the last iteration number from eval pool directories.
        Returns last iteration.
        """
        dirs = [d.name for d in self.dirs['eval_pool'].iterdir() if d.is_dir()]
        dirs = list(map(lambda x: int(x.split('_')[1]), dirs))
        dirs.sort()
        print(dirs)
        return dirs[-1]

    def _load_model(self, model_dir: Path, load_checkpoint: bool) -> HnefataflZeroNet:
        """
        Loads previous checkpoint if load_old or a new instance if not.

        Returns the loaded model.
        """
        model = HnefataflZeroNet(depth=self.cfg.model.depth, filter_count=self.cfg.model.filter_count,
                                 rngs=self.rngs)
        if load_checkpoint and model_dir.exists() and any(model_dir.iterdir()):
            graph_def, abstract_state = nnx.split(model)
            restored_state = self.checkpointer.restore(model_dir, abstract_state)
            model = nnx.merge(graph_def, restored_state)

        return model

    def train(self):
        for i in range(self.cfg.train.iterations):
            start_time = time.time()
            iteration = i + self.last_iteration + 1
            print(f"--- Iteration {iteration} ---")

            self._run_self_play_loop()
            self._run_training_loop()

            self.metrics_tracker.update_frames(self.cfg.train.self_play_steps * self.cfg.train.batch_size)

            elo = self.evaluator.evaluate_model(iteration)
            self.metrics_tracker.metrics_history['elo_evaluation'].append(elo)

            t_loss = self.metrics_tracker.metrics_history['total_loss'][-1]
            p_loss = self.metrics_tracker.metrics_history['policy_loss'][-1]
            v_loss = self.metrics_tracker.metrics_history['value_loss'][-1]

            print(f">>> RESULTS | Elo: {elo} | Total Loss: {t_loss:.4f} (Policy: {p_loss:.4f}, Value: {v_loss:.4f})",
                  flush=True)
            elapsed = time.time() - start_time
            print(f">>> TIME    | Iteration {iteration} took {elapsed / 60:.2f} minutes.\n", flush=True)

        self._save_progress()
        self.metrics_tracker.plot_metrics()

    def _run_self_play_loop(self):
        self.model.eval()

        steps = self.cfg.train.self_play_steps
        print(f"Generating data ({steps} steps x {self.cfg.train.batch_size} games)...")

        rng_key = self.rngs.split()

        self.env_state, self.buffer_state = self_play(
            model=self.model,
            env_state=self.env_state,
            buffer_state=self.buffer_state,
            rng_key=rng_key,
            num_steps = self.cfg.train.self_play_steps,
            num_simulations=self.cfg.mcts.simulations,
            env=self.env,
            buffer=self.buffer,
            batch_size=self.cfg.train.batch_size
        )

    def _run_training_loop(self):
        self.model.train()

        steps = self.cfg.train.num_epochs
        pbar = tqdm(range(steps), desc="Training", mininterval=self.cfg.train.tqdm_interval, ncols=100)

        for _ in pbar:
            rng_key = self.rngs.split()
            batch = self.buffer.sample(self.buffer_state, rng_key)
            training_data = batch.experience.first
            training_data = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, self.data_sharding), training_data
            )

            loss, p_loss, v_loss = train_step(
                self.model,
                self.optimizer,
                training_data
            )

            self.metrics_tracker.update_step(total_loss=loss, policy_loss=p_loss, value_loss=v_loss)

            pbar.set_postfix({
                'L': f"{loss:.4f}",
                'P': f"{p_loss:.4f}",
                'V': f"{v_loss:.4f}"
            })

        self.metrics_tracker.compute_and_record()

    def _save_progress(self):
        """Saves the model parameters, loss data, and evaluation data."""
        _, state = nnx.split(self.model)
        self.checkpointer.save(self.dirs['checkpoints'], state, force=True)

        self.metrics_tracker.save_metrics()


@hydra.main(version_base=None, config_path='..', config_name='config')
def main(cfg: DictConfig):
    coach = Coach(cfg)
    coach.train()


if __name__ == '__main__':
    main()
