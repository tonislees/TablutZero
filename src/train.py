import time
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
from src.metrics import MetricsTracker
from src.model import HnefataflZeroNet
from src.self_play import self_play, set_pbar
from src.utils import dir_safe, add_to_buffer_cpu, train_step


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
        self.current_lr = cfg.train.initial_learning_rate
        self.optimizer = self._load_optimizer(cfg.train.load_checkpoint)

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
        self.reward_consts = [1, -1, 1, -1] # [attacker_win_r, attacker_loss_r, defender_win_r, defender_loss_r]

        # Buffer
        min_buffer_size = cfg.train.batch_size * cfg.train.self_play_steps
        self.buffer = fbx.make_flat_buffer(
            max_length=min_buffer_size * 4,
            min_length=min_buffer_size,
            sample_batch_size=cfg.train.batch_size,
            add_batch_size=cfg.train.batch_size
        )
        cpu_device = jax.devices('cpu')[0]
        with jax.default_device(cpu_device):
            example_transition = {
                "observation": jnp.zeros((9, 9, 43), dtype=jnp.float32),
                "policy_target": jnp.zeros((81 * 32,), dtype=jnp.float32),
                "value_target": jnp.zeros((), dtype=jnp.float32),
                "legal_action_mask": jnp.zeros((81 * 32,), dtype=jnp.bool_),
                "player": jnp.zeros((), dtype=jnp.int32)
            }
            self.buffer_state = self.buffer.init(example_transition)
        self.sample_fn = jax.jit(self.buffer.sample, backend='cpu')

    def _load_optimizer(self, load_checkpoint: bool):
        optimizer = nnx.Optimizer(
            self.model,
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=self.current_lr)
            ),
            wrt=nnx.Param
        )

        if load_checkpoint and hasattr(self, '_restored_opt_state'):
            opt_graph_def, _ = nnx.split(optimizer)
            optimizer = nnx.merge(opt_graph_def, self._restored_opt_state)
            del self._restored_opt_state

        return optimizer

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

    def _maybe_reduce_lr(self):
        history = self.metrics_tracker.metrics_history['total_loss']
        if len(history) < 20:
            return
        recent = sum(history[-10:]) / 10
        previous = sum(history[-20:-10]) / 10
        improvement = (previous - recent) / previous
        if improvement < 0.005:
            self.current_lr = max(self.current_lr * 0.5, 1e-5)
            self.optimizer = nnx.Optimizer(
                self.model,
                optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(learning_rate=self.current_lr)
                ),
                wrt=nnx.Param
            )
            print(f"\n    LR reduced to {self.current_lr}")

    def _load_model(self, model_dir: Path, load_checkpoint: bool) -> HnefataflZeroNet:
        model = HnefataflZeroNet(
            depth=self.cfg.model.depth,
            filter_count=self.cfg.model.filter_count,
            rngs=self.rngs
        )
        if load_checkpoint and model_dir.exists() and any(model_dir.iterdir()):
            graph_def, abstract_state = nnx.split(model)

            temp_opt = nnx.Optimizer(
                model,
                optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(learning_rate=self.cfg.train.initial_learning_rate)
                ),
                wrt=nnx.Param
            )
            _, abstract_opt_state = nnx.split(temp_opt)

            abstract_checkpoint = {
                'model': abstract_state,
                'optimizer': abstract_opt_state,
                'lr': 0.0
            }
            restored = self.checkpointer.restore(model_dir, abstract_checkpoint)

            model = nnx.merge(graph_def, restored['model'])
            self.current_lr = restored['lr']
            self._restored_opt_state = restored['optimizer']

        return model

    def train(self):
        eval_interval = self.cfg.train.eval_interval
        eval_start = self.cfg.train.eval_start
        for i in range(self.cfg.train.iterations):
            start_time = time.time()
            iteration = i + self.last_iteration + 1
            print(f"--- Iteration {iteration} ---")

            self._run_self_play_loop()
            self._run_training_loop()

            self.metrics_tracker.update_frames(self.cfg.train.self_play_steps * self.cfg.train.batch_size)
            self._maybe_reduce_lr()

            t_loss = self.metrics_tracker.metrics_history['total_loss'][-1]
            p_loss = self.metrics_tracker.metrics_history['policy_loss'][-1]
            v_loss = self.metrics_tracker.metrics_history['value_loss'][-1]

            print(f">>> RESULTS | Total Loss: {t_loss:.4f} (Policy: {p_loss:.4f}, Value: {v_loss:.4f})",
                  flush=True)

            if iteration % eval_interval == 0 and iteration >= eval_start:
                elo = self.evaluator.evaluate_model(iteration)
                self.metrics_tracker.metrics_history['elo_evaluation'].append(elo)

            elapsed = time.time() - start_time
            print(f">>> TIME    | Iteration {iteration} took {elapsed / 60:.2f} minutes.\n", flush=True)

        self._save_progress()
        self.metrics_tracker.plot_metrics()

    def _run_self_play_loop(self):
        self.model.eval()

        steps = self.cfg.train.self_play_steps
        print(f"Generating data ({steps} steps x {self.cfg.train.batch_size} games)...")

        rng_key = self.rngs.split()

        pbar = tqdm(total=steps, desc="Self-Play", mininterval=self.cfg.train.tqdm_interval,
                               ncols=100, unit='steps')
        set_pbar(pbar)

        self.env_state, final_transitions, terminals, rewards = self_play(
            model=self.model,
            env_state=self.env_state,
            rng_key=rng_key,
            num_steps = self.cfg.train.self_play_steps,
            num_simulations=self.cfg.mcts.simulations,
            env=self.env,
            batch_size=self.cfg.train.batch_size,
            reward_consts=jnp.array(self.reward_consts, dtype=jnp.float32),
            dirichlet_fraction=self.cfg.train.dirichlet_fraction,
            attacker_explore=self.cfg.train.attacker_explore
        )

        pbar.close()
        set_pbar(None)

        self._process_results(terminals, rewards)
        final_transitions_cpu = jax.device_get(final_transitions)
        self.buffer_state = add_to_buffer_cpu(self.buffer_state, final_transitions_cpu, self.buffer)

    def _process_results(self, terminals, rewards):
        terminals = jax.device_get(terminals)
        attacker_rewards = jax.device_get(rewards)

        total_terminated = int(terminals.sum())
        attacker_wins = int((terminals & (attacker_rewards == 1)).sum())
        defender_wins = int((terminals & (attacker_rewards == -1)).sum())
        total_draws = int((terminals & (attacker_rewards == 0)).sum())

        if total_terminated > 0:
            a_win_rate = attacker_wins / total_terminated
            d_win_rate = defender_wins / total_terminated
            draw_rate = total_draws / total_terminated
        else:
            a_win_rate = d_win_rate = draw_rate = 0.0

        rates = (a_win_rate, d_win_rate, draw_rate)
        names = ('attacker_win_rate', 'defender_win_rate', 'draw_rate')
        # past_5_avg = [] # [Attacker avg, defender avg, draw avg]

        for rate, name in zip(rates, names):
            history = self.metrics_tracker.metrics_history[name]
            history.append(rate)
            last_5 = history[-5:]
            # past_5_avg.append(sum(last_5) / len(last_5))

        # self.reward_consts = calculate_dynamic_rewards(past_5_avg[0], past_5_avg[1])

        print(f">>> Self-play games finished: {total_terminated}")
        print(
            f"    Attacker Win Rate: {a_win_rate:.1%} | Defender Win Rate: {d_win_rate:.1%} | Draw Rate: {draw_rate:.1%}")
        # print(f"    Attacker reward: {self.reward_consts[0]} | Defender reward: {self.reward_consts[2]}")

    def _run_training_loop(self):
        self.model.train()

        total_steps = self.cfg.train.self_play_steps * self.cfg.train.num_epochs
        pbar = tqdm(range(total_steps), desc="Training", mininterval=self.cfg.train.tqdm_interval,
                    ncols=100, unit='steps')

        for _ in pbar:
            rng_key = self.rngs.split()
            batch = self.sample_fn(self.buffer_state, rng_key)
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
        _, model_state = nnx.split(self.model)
        _, opt_state = nnx.split(self.optimizer)
        checkpoint = {
            'model': model_state,
            'optimizer': opt_state,
            'lr': self.current_lr
        }
        self.checkpointer.save(self.dirs['checkpoints'], checkpoint, force=True)
        self.evaluator.save_eval_pool()
        self.metrics_tracker.save_metrics()


@hydra.main(version_base=None, config_path='..', config_name='config')
def main(cfg: DictConfig):
    coach = Coach(cfg)
    coach.train()


if __name__ == '__main__':
    main()
