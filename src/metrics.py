import datetime
import json
from pathlib import Path

from flax import nnx
from matplotlib import pyplot as plt
from omegaconf import DictConfig


class MetricsTracker:
    def __init__(self, cfg: DictConfig, dirs: dict[str, Path]):
        self.dirs = dirs
        self.cfg = cfg
        self.metrics_history = self._load_metrics(cfg.train.load_checkpoint)
        self.metrics: nnx.MultiMetric = nnx.MultiMetric(
            total_loss=nnx.metrics.Average(argname='total_loss'),
            policy_loss=nnx.metrics.Average(argname='policy_loss'),
            value_loss=nnx.metrics.Average(argname='value_loss'),
        )

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

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        metrics = [
            ('Total Loss', self.metrics_history['total_loss']),
            ('Policy Loss (Cross Entropy)', self.metrics_history['policy_loss']),
            ('Value Loss (MSE)', self.metrics_history['value_loss'])
        ]

        for ax, (title, data) in zip(axes, metrics):
            if not data: continue

            ax.plot(data, alpha=0.25, color='gray', label='Raw')
            smoothed_data = smooth(data, weight=0.95)
            ax.plot(smoothed_data, alpha=1.0, linewidth=2, label='Smoothed')

            ax.set_title(title)
            ax.set_xlabel('Training Steps')
            ax.grid(True, alpha=0.3)
            ax.legend()

            iterations = len(data) // self.cfg.train.num_epochs
            for i in range(iterations):
                ax.axvline(x=i * self.cfg.train.num_epochs, color='red', alpha=0.1)

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
            json.dump(self.metrics_history, f)

    def _load_metrics(self, load_checkpoint: bool) -> dict[str, list[float]]:
        """
        Loads the most recent metrics history from the disk if load_checkpoint is true.
        Returns the previous metrics or a new dict.
        """
        default_metrics = {
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
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