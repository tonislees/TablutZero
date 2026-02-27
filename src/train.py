from functools import partial
from pathlib import Path

import flashbax as fbx
import hydra
import jax
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


def rl_loss_fn(model, batch, train=True):
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
def train_step_rl(model, optimizer, batch):
    grad_fn = nnx.value_and_grad(rl_loss_fn, has_aux=True)
    (loss, (p_loss, v_loss)), grads = grad_fn(model, batch)
    optimizer.update(model, grads)
    return loss, p_loss, v_loss


@partial(nnx.jit, static_argnames=('num_simulations', 'env', 'buffer'))
def self_play_step(model, env_state, buffer_state, rng_key, num_simulations, env, buffer):
    key_search, key_act = jax.random.split(rng_key)

    mcts_output = run_mcts(model, env_state, key_search, num_simulations, env)
    actions = mcts_output.action
    mcts_values = mcts_output.search_tree.node_values[:, 0]
    next_env_state = jax.vmap(env.step)(env_state, actions)
    batch_indices = jnp.arange(env_state.current_player.shape[0])
    current_player_rewards = next_env_state.rewards[batch_indices, env_state.current_player]
    final_value_target = jnp.where(
        next_env_state.terminated,
        current_player_rewards,
        mcts_values
    )

    transition = {
        "observation": env_state.observation,
        "policy_target": mcts_output.action_weights,
        "value_target": final_value_target
    }
    new_buffer_state = buffer.add(buffer_state, transition)

    return next_env_state, new_buffer_state


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
        self.optimizer: nnx.Optimizer = nnx.Optimizer(
            self.model, optax.adamw(learning_rate=cfg.train.learning_rate), wrt=nnx.Param
        )

        # Environment
        self.env = Hnefatafl()
        key_env = jax.random.PRNGKey(cfg.train.seed + 1)
        self.env_state = jax.jit(jax.vmap(self.env.init))(
            jax.random.split(key_env, cfg.train.batch_size)
        )

        # Setup
        self.last_iteration = self._get_last_iteration() if cfg.train.load_checkpoint else 0
        self.metrics_tracker = MetricsTracker(cfg, self.dirs)
        self.evaluator = Evaluator(cfg, self.dirs, self.rngs, self.model, self.checkpointer, self.env)

        # Buffer
        self.buffer = fbx.make_flat_buffer(
            max_length=cfg.buffer.size,
            min_length=cfg.buffer.min_size,
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
            iteration = i + self.last_iteration + 1
            print(f"--- Iteration {iteration} ---")

            self._run_self_play_loop()
            self._run_training_loop()

            elo = self.evaluator.evaluate_model(iteration)
            self.metrics_tracker.metrics_history['elo_evaluation'].append(elo)
        self._save_progress()
        self.metrics_tracker.plot_metrics()

    def _run_self_play_loop(self):
        self.model.eval()

        steps = self.cfg.train.self_play_steps
        print(f"Generating data ({steps} steps x {self.cfg.train.batch_size} games)...")

        for _ in tqdm(range(steps)):
            rng_key = self.rngs.split()

            self.env_state, self.buffer_state = self_play_step(
                self.model,
                self.env_state,
                self.buffer_state,
                rng_key,
                self.cfg.mcts.simulations,
                self.env,
                self.buffer
            )

    def _run_training_loop(self):
        self.model.train()

        steps = self.cfg.train.num_epochs
        pbar = tqdm(range(steps), desc="Training")

        for _ in pbar:
            rng_key = self.rngs.split()
            batch = self.buffer.sample(self.buffer_state, rng_key)
            training_data = batch.experience.first

            loss, p_loss, v_loss = train_step_rl(
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
