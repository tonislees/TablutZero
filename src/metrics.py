import datetime
import json
from pathlib import Path

from flax import nnx
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from src.evaluation import Evaluator


class MetricsTracker:
    def __init__(self, cfg: DictConfig, dirs: dict[str, Path], evaluator: Evaluator):
        self.dirs = dirs
        self.cfg = cfg
        self.metrics_history = self._load_metrics(cfg.train.load_checkpoint)
        self.metrics: nnx.MultiMetric = nnx.MultiMetric(
            total_loss=nnx.metrics.Average(argname='total_loss'),
            policy_loss=nnx.metrics.Average(argname='policy_loss'),
            value_loss=nnx.metrics.Average(argname='value_loss'),
        )
        self.evaluator = evaluator

    def update_frames(self, frame_count: int) -> None:
        """
        Calculates the cumulative frame count and appends it to the history.
        """
        if not self.metrics_history['frames']:
            current_total = 0
        else:
            current_total = self.metrics_history['frames'][-1]

        self.metrics_history['frames'].append(current_total + frame_count)

    def update_step(self, total_loss: float, policy_loss: float, value_loss: float) -> None:
        """
        Feeds the current batch's outputs into the MultiMetric.
        """
        self.metrics.update(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss
        )

    def compute_and_record(self) -> None:
        """
        Computes the accumulated metrics, appends them to the history,
        and resets the internal state for the next interval.
        """
        current_metrics = self.metrics.compute()
        self.metrics_history['total_loss'].append(float(current_metrics['total_loss']))
        self.metrics_history['policy_loss'].append(float(current_metrics['policy_loss']))
        self.metrics_history['value_loss'].append(float(current_metrics['value_loss']))

        self.metrics.reset()

    def plot_elo(self, batch_size: int, train_steps: int):
        ratings = self.evaluator.run_bayeselo(self.dirs['training'] / 'metrics' / 'game_results.pgn')
        frames_per_iter = batch_size * train_steps
        metrics_dir = self.dirs['training'] / 'metrics'
        metrics_files = sorted(metrics_dir.glob("metrics_*.json"))
        if not metrics_files:
            return
        with open(metrics_files[-1]) as f:
            history = json.load(f)

        # Elo
        sorted_items = sorted(ratings.items(), key=lambda x: int(x[0].split('_')[1]))
        iters = [int(name.split('_')[1]) for name, _ in sorted_items]
        elos = [e - min(ratings.values()) for _, e in sorted_items]
        elo_frames = [i * frames_per_iter / 1e6 for i in iters]

        # Win rates
        n_iters = len(history['attacker_win_rate'])
        rate_frames = [(i + 1) * frames_per_iter / 1e6 for i in range(n_iters)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Elo
        ax1.plot(elo_frames, elos, marker='o', linewidth=2, color='royalblue')
        ax1.set_ylabel("Elo")
        ax1.set_xlabel("Frames (millions)")
        ax1.set_title("Elo Rating")
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('auto')

        # Stacked area win rates
        ax2.stackplot(
            rate_frames,
            history['attacker_win_rate'],
            history['defender_win_rate'],
            labels=['Attacker Win', 'Defender Win'],
            colors=['#4CAF50', '#9E9E9E', '#F44336'],
            alpha=0.85
        )
        ax2.set_ylabel("Rate")
        ax2.set_xlabel("Frames (millions)")
        ax2.set_title("Game Outcomes")
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.2)

        plt.tight_layout(w_pad=4)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        plot_path = self.dirs['plots'] / f"Training_Summary_{timestamp}.pdf"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)


    def plot_metrics(self):
        def smooth(scalars, weight=0.9):
            if not scalars: return []
            last = scalars[0]
            smoothed = []
            for point in scalars:
                smoothed_val = last * weight + (1 - weight) * point
                smoothed.append(smoothed_val)
                last = smoothed_val
            return smoothed

        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()

        metrics = [
            ('Total Loss', self.metrics_history['total_loss']),
            ('Policy Loss (Cross Entropy)', self.metrics_history['policy_loss']),
            ('Value Loss (MSE)', self.metrics_history['value_loss']),
            ('Elo Rating', self.metrics_history['elo_evaluation'])
        ]

        frames = self.metrics_history['frames']

        for ax, (title, data) in zip(axes, metrics):
            if not data or len(data) != len(frames):
                continue

            ax.plot(frames, data, alpha=0.25, color='gray', label='Raw')

            if len(data) > 2:
                w = 0.95 if 'Loss' in title else 0.5
                smoothed_data = smooth(data, weight=w)
                ax.plot(frames, smoothed_data, alpha=1.0, linewidth=2, label='Smoothed')

            ax.set_title(title)
            ax.set_xlabel('Total Frames')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        plot_path = self.dirs['plots'] / f"RL_Training_Metrics_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close(fig)

    def save_metrics(self) -> None:
        """
        Writes current metrics history into a JSON file for checkpoint.
        """
        filename = f"metrics_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        file_dir = self.dirs['metrics'] / filename
        with open(file_dir, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)

    def _load_metrics(self, load_checkpoint: bool) -> dict[str, list[float]]:
        """
        Loads the most recent metrics history from the disk if load_checkpoint is true.
        Returns the previous metrics or a new dict.
        """
        default_metrics = {
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'attacker_win_rate': [],
            'defender_win_rate': [],
            'elo_evaluation': [],
            'frames': []
        }

        if not load_checkpoint or not self.dirs['metrics'].exists():
            return default_metrics

        metric_files = list(self.dirs['metrics'].glob("metrics_*.json"))

        if not metric_files:
            return default_metrics

        latest_file = sorted(metric_files)[-1]

        print(f"Loading metrics from {latest_file.name}")
        with open(latest_file, 'r') as f:
            loaded_metrics = json.load(f)

        return loaded_metrics