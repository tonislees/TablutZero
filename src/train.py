import random
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
from src.self_play import self_play, self_play_vs_opponent, set_pbar
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
        data_dir = dir_safe('data', root_dir)
        self.dirs = {
            'checkpoints': dir_safe('checkpoints', model_dir),
            'eval_pool': dir_safe('eval_pool', model_dir),
            'inference': dir_safe('inference', model_dir),
            'plots': dir_safe('plots', data_dir),
            'metrics': dir_safe('metrics', data_dir),
            'bayeselo': root_dir / 'bayeselo',
            'pgn': root_dir / 'game_results.pgn',
            'training': dir_safe('training_data', root_dir)
        }
        if not cfg.train.load_checkpoint:
            self.dirs['pgn'].unlink(missing_ok=True)

        # Buffer
        cpu_device = jax.devices('cpu')[0]
        min_buffer_size = cfg.train.batch_size * cfg.train.self_play_steps
        self.buffer = fbx.make_flat_buffer(
            max_length=min_buffer_size * cfg.train.buffer_multiplier,
            min_length=min_buffer_size,
            sample_batch_size=cfg.train.batch_size,
            add_batch_size=cfg.train.batch_size
        )

        with jax.default_device(cpu_device):
            example_transition = {
                "observation": jnp.zeros((9, 9, 43), dtype=jnp.float32),
                "policy_target": jnp.zeros((81 * 32,), dtype=jnp.float32),
                "value_target": jnp.zeros((), dtype=jnp.float32),
                "legal_action_mask": jnp.zeros((81 * 32,), dtype=jnp.bool_),
                "player": jnp.zeros((), dtype=jnp.int32)
            }
        self.sample_fn = jax.jit(self.buffer.sample, backend='cpu')

        # Model & optimizer
        self._init_or_restore(example_transition, cpu_device)
        graph_def, state = nnx.split((self.model, self.optimizer))
        state = jax.tree_util.tree_map(lambda x: jax.device_put(x, self.replicated_sharding), state)
        self.model, self.optimizer = nnx.merge(graph_def, state)

        # Environment
        self.env = Hnefatafl()

        self.opp_ratio = cfg.train.get('opponent_ratio', 0.25)
        self.opp_batch = int(cfg.train.batch_size * self.opp_ratio)
        self.self_batch = cfg.train.batch_size - self.opp_batch

        key_env = jax.random.PRNGKey(cfg.train.seed + 1)
        key_self, key_opp = jax.random.split(key_env)

        # Persistent env state for self-play portion
        self.env_state_self = jax.jit(jax.vmap(self.env.init))(
            jax.random.split(key_self, self.self_batch)
        )
        self.env_state_self = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, self.data_sharding), self.env_state_self
        )

        # Setup
        self.last_iteration = self._get_last_iteration() if cfg.train.load_checkpoint else 0
        self.evaluator = Evaluator(cfg, self.dirs, self.rngs, self.model, self.checkpointer, self.env)
        self.metrics_tracker = MetricsTracker(cfg, self.dirs, self.evaluator)
        self.reward_consts = [1, -1, 1, -1, 0.0, 0.0] # [attacker_win_r, attacker_loss_r, defender_win_r, defender_loss_r, attacker_draw_r, defender_draw_r]

    def _create_optimizer(self):
        total_training_steps = 400 * self.cfg.train.num_epochs * self.cfg.train.self_play_steps

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-4,
            peak_value=2e-3,
            warmup_steps=500,
            decay_steps=total_training_steps,
            end_value=1e-5
        )

        return nnx.Optimizer(
            self.model,
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=schedule, weight_decay=1e-4)
            ),
            wrt=nnx.Param
        )

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

    def _init_or_restore(self, example_transition, cpu_device):
        load = self.cfg.train.load_checkpoint
        ckpt_dir = self.dirs['checkpoints']
        has_checkpoint = load and ckpt_dir.exists() and any(ckpt_dir.iterdir())

        self.model = HnefataflZeroNet(
            depth=self.cfg.model.depth,
            filter_count=self.cfg.model.filter_count,
            rngs=self.rngs
        )

        if has_checkpoint:
            print("Restoring from checkpoint...")
            graph_def, abstract_model = nnx.split(self.model)

            temp_opt = self._create_optimizer()
            _, abstract_opt = nnx.split(temp_opt)
            del temp_opt

            with jax.default_device(cpu_device):
                full_buffer = self.buffer.init(example_transition)

            abstract_checkpoint = {
                'model': abstract_model,
                'optimizer': abstract_opt,
                'buffer': full_buffer
            }

            restored = self.checkpointer.restore(
                ckpt_dir,
                target=abstract_checkpoint
            )
            del abstract_model, abstract_opt

            self.model = nnx.merge(graph_def, restored['model'])
            self.optimizer = self._create_optimizer()
            nnx.update(self.optimizer, restored['optimizer'])
            self.buffer_state = restored['buffer']

            del restored, full_buffer
        else:
            self.optimizer = self._create_optimizer()
            with jax.default_device(cpu_device):
                self.buffer_state = self.buffer.init(example_transition)

    def train(self):
        eval_interval = self.cfg.train.eval_interval
        eval_start = self.cfg.train.eval_start
        save_interval = self.cfg.train.save_interval

        for i in range(self.cfg.train.iterations):
            start_time = time.time()
            iteration = i + self.last_iteration + 1
            print(f"--- Iteration {iteration} ---")

            self._run_self_play_loop(iteration)
            self._run_training_loop()

            self.metrics_tracker.update_frames(self.cfg.train.self_play_steps * self.cfg.train.batch_size)

            t_loss = self.metrics_tracker.metrics_history['total_loss'][-1]
            p_loss = self.metrics_tracker.metrics_history['policy_loss'][-1]
            v_loss = self.metrics_tracker.metrics_history['value_loss'][-1]
            v_acc = self.metrics_tracker.metrics_history['value_acc'][-1]

            print(f"    Total Loss: {t_loss:.4f} (Policy: {p_loss:.4f}, Value: {v_loss:.4f}, Acc: {v_acc:.1%})",
                  flush=True)

            if iteration % eval_interval == 0 and iteration >= eval_start:
                self.evaluator.evaluate_model(iteration)

            if iteration % save_interval == 0:
                self._save_progress()

            elapsed = time.time() - start_time
            print(f"    Iteration {iteration} took {elapsed / 60:.2f} minutes.\n", flush=True)

        self._save_progress()

    def _load_random_sp_opponent(self):
        """Load a random opponent from the eval pool for mixed self-play."""
        pool = self.evaluator.eval_pool
        if not pool:
            return None, None

        name = random.choice(list(pool.keys()))
        graph_def, _ = nnx.split(self.model)
        gpu_state = jax.device_put(pool[name])
        opponent = nnx.merge(graph_def, gpu_state)
        opponent.eval()
        return opponent, name

    def _init_opponent_env(self, player_order):
        """Create a fresh opponent env state with forced player_order."""
        key = self.rngs.split()
        env_state_opp = jax.jit(jax.vmap(self.env.init))(
            jax.random.split(key, self.opp_batch)
        )
        forced_order = jnp.broadcast_to(player_order[None, :], (self.opp_batch, 2))
        env_state_opp = env_state_opp.replace(
            _player_order=forced_order,
            current_player=jnp.full(self.opp_batch, player_order[0], dtype=jnp.int32)
        )
        env_state_opp = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, self.data_sharding), env_state_opp
        )
        return env_state_opp

    def _run_self_play_loop(self, iteration: int):
        self.model.eval()

        steps = self.cfg.train.self_play_steps
        opponent, opponent_name = self._load_random_sp_opponent()

        print(f"Generating data ({steps} steps x {self.self_batch}+{self.opp_batch} games, opponent: {opponent_name})...")

        # Alternate which side the current model plays against the opponent
        player_order = jnp.array([0, 1]) if iteration % 2 == 0 else jnp.array([1, 0])
        role_str = "attacker" if iteration % 2 == 0 else "defender"
        print(f"    Opponent batch: current model as {role_str}")

        rng_key_self = self.rngs.split()
        rng_key_opp = self.rngs.split()

        pbar = tqdm(total=steps * 2, desc="Self-Play", mininterval=self.cfg.train.tqdm_interval,
                    ncols=100, unit='steps')
        set_pbar(pbar)

        reward_consts = jnp.array(self.reward_consts, dtype=jnp.float32)

        (self.env_state_self, self_transitions, self_terminals, self_rewards,
         self_step_counts, self_entropies, self_pieces_left, self_hm_draws) = self_play(
            model=self.model,
            env_state=self.env_state_self,
            rng_key=rng_key_self,
            num_steps=steps,
            num_simulations=self.cfg.mcts.simulations,
            env=self.env,
            batch_size=self.self_batch,
            reward_consts=reward_consts
        )

        env_state_opp = self._init_opponent_env(player_order)

        (_, opp_transitions, opp_terminals, opp_rewards,
         opp_step_counts, opp_entropies, opp_pieces_left, opp_hm_draws) = self_play_vs_opponent(
            model=self.model,
            opponent=opponent,
            env_state=env_state_opp,
            rng_key=rng_key_opp,
            num_steps=steps,
            num_simulations=self.cfg.mcts.simulations,
            env=self.env,
            batch_size=self.opp_batch,
            reward_consts=reward_consts,
            player_order=player_order
        )

        pbar.close()
        set_pbar(None)

        all_transitions = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate([a, b], axis=1),
            self_transitions, opp_transitions
        )
        all_terminals = jnp.concatenate([self_terminals, opp_terminals], axis=1)
        all_rewards = jnp.concatenate([self_rewards, opp_rewards], axis=1)
        all_step_counts = jnp.concatenate([self_step_counts, opp_step_counts], axis=1)
        all_entropies = jnp.concatenate([self_entropies, opp_entropies], axis=1)
        all_pieces_left = jnp.concatenate([self_pieces_left, opp_pieces_left], axis=1)
        all_hm_draws = jnp.concatenate([self_hm_draws, opp_hm_draws], axis=1)

        self._process_results(all_terminals, all_rewards, all_step_counts,
                              all_entropies, all_pieces_left, all_hm_draws)

        all_transitions_cpu = jax.device_get(all_transitions)
        self.buffer_state = add_to_buffer_cpu(self.buffer_state, all_transitions_cpu, self.buffer)

    def _process_results(self, terminals, rewards, step_counts, entropies, pieces_left, half_move_draws):
        terminals = jax.device_get(terminals)
        attacker_rewards = jax.device_get(rewards)

        step_counts = jax.device_get(step_counts)
        entropies = jax.device_get(entropies)
        pieces_left = jax.device_get(pieces_left)
        half_move_draws = jax.device_get(half_move_draws)

        completed_game_lengths = step_counts[terminals]
        if len(completed_game_lengths) > 0:
            avg_length = float(completed_game_lengths.mean())
        else:
            avg_length = 0

        total_terminated = int(terminals.sum())
        attacker_wins = int((terminals & (attacker_rewards == 1)).sum())
        defender_wins = int((terminals & (attacker_rewards == -1)).sum())
        total_draws = int((terminals & (attacker_rewards == 0)).sum())

        if total_terminated > 0:
            a_win_rate = attacker_wins / total_terminated
            d_win_rate = defender_wins / total_terminated
            draw_rate = total_draws / total_terminated

            attacker_ev = (attacker_wins - defender_wins) / total_terminated
            attacker_score = (attacker_wins + 0.5 * total_draws) / total_terminated

            avg_entropy = float(entropies.mean())
            avg_pieces = float(pieces_left[terminals].mean())
            hm_draw_rate = float(half_move_draws.sum() / total_terminated)
        else:
            a_win_rate = d_win_rate = draw_rate = 0.0
            attacker_ev = attacker_score = 0.0
            avg_entropy = avg_pieces = hm_draw_rate = 0.0

        metrics = (a_win_rate, d_win_rate, draw_rate, avg_length, avg_pieces,
                   avg_entropy, attacker_ev, attacker_score)
        names = ('attacker_win_rate', 'defender_win_rate', 'draw_rate', 'game_lengths', 'pieces_left',
                 'entropy', 'attacker_ev', 'attacker_score')

        for metric, name in zip(metrics, names):
            self.metrics_tracker.metrics_history[name].append(metric)

        print(f"    Policy Entropy: {avg_entropy:.4f} | Avg Pieces Left: {avg_pieces:.1f} | Half-Move Draw Rate: {hm_draw_rate:.1%}")
        print(f"    Attacker Win Rate: {a_win_rate:.1%} | Defender Win Rate: {d_win_rate:.1%} | Draw Rate: {draw_rate:.1%}")
        print(f"    Attacker EV: {attacker_ev:+.3f} | Attacker Score: {attacker_score:.1%}")
        print(f"    Average Game Length: {avg_length:.1f} steps")

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

            aug_key = self.rngs.split()
            loss, p_loss, v_loss, v_acc = train_step(
                self.model,
                self.optimizer,
                training_data,
                aug_key
            )

            self.metrics_tracker.update_step(total_loss=loss, policy_loss=p_loss, value_loss=v_loss, value_acc=v_acc)

            pbar.set_postfix({
                'L': f"{loss:.4f}",
                'P': f"{p_loss:.4f}",
                'V': f"{v_loss:.4f}",
                'Acc': f"{v_acc:.1%}"
            })

        self.metrics_tracker.compute_and_record()

    def _save_progress(self):
        """Saves the model parameters, loss data, and evaluation data."""
        _, model_state = nnx.split(self.model)
        _, opt_state = nnx.split(self.optimizer)
        checkpoint = {
            'model': model_state,
            'optimizer': opt_state,
            'buffer': self.buffer_state
        }
        self.checkpointer.save(self.dirs['checkpoints'], checkpoint, force=True)
        self.checkpointer.wait_until_finished()
        self.checkpointer.save(self.dirs['inference'], {'model': model_state}, force=True)
        self.checkpointer.wait_until_finished()
        self.evaluator.save_eval_pool()
        self.metrics_tracker.save_metrics()


@hydra.main(version_base=None, config_path='..', config_name='config')
def main(cfg: DictConfig):
    coach = Coach(cfg)
    coach.train()


if __name__ == '__main__':
    main()