import datetime
import json
import shutil
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
            value_acc=nnx.metrics.Average(argname='value_acc')
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

    def update_step(self, total_loss: float, policy_loss: float, value_loss: float, value_acc: float) -> None:
        """
        Feeds the current batch's outputs into the MultiMetric.
        """
        self.metrics.update(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            value_acc=value_acc
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
        self.metrics_history['value_acc'].append(float(current_metrics['value_acc']))

        self.metrics.reset()

    def plot_elo(self, batch_size: int, train_steps: int):
        metrics_dir = self.dirs['training'] / 'metrics'
        ratings = self.evaluator.run_bayeselo(list(metrics_dir.glob("game_results_*.pgn"))[0])
        frames_per_iter = batch_size * train_steps
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

        # Win/draw rates
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
            history['draw_rate'],
            history['defender_win_rate'],
            labels=['Attacker Win', 'Draw', 'Defender Win'],
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

    def plot_all_metrics(self):
        root = Path(__file__).resolve().parents[1]
        metrics_dir = root / 'training_data' / 'metrics'
        plots_dir = root / 'training_data' / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)

        metric_files = sorted(metrics_dir.glob("metrics_*.json"))
        if not metric_files:
            print("No metrics files found.")
            return

        latest = metric_files[-1]
        print(f"Loading {latest.name}")
        with open(latest) as f:
            h = json.load(f)

        frames = h['frames']
        if not frames:
            print("No frame data.")
            return

        frames_m = [f / 1e6 for f in frames]

        fig, axes = plt.subplots(4, 3, figsize=(24, 22))
        fig.suptitle('HnefataflZero Training Dashboard', fontsize=16, fontweight='bold', y=0.98)

        single_plots = [
            (0, 0, 'Total Loss', 'total_loss', 'Loss'),
            (0, 1, 'Policy Loss', 'policy_loss', 'Loss'),
            (0, 2, 'Value Loss', 'value_loss', 'MSE'),
            (1, 0, 'Value Accuracy', 'value_acc', 'Accuracy'),
            (1, 1, 'Policy Entropy', 'entropy', 'Entropy (nats)'),
            (1, 2, 'Average Game Length', 'game_lengths', 'Steps'),
            (2, 0, 'Average Pieces Left', 'pieces_left', 'Pieces'),
            (2, 1, 'Attacker EV', 'attacker_ev', 'EV'),
            (2, 2, 'Attacker Score', 'attacker_score', 'Score'),
        ]

        for row, col, title, key, ylabel in single_plots:
            ax = axes[row, col]
            data = h.get(key, [])
            if data and len(data) == len(frames_m):
                ax.plot(frames_m, data, linewidth=1.5)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Frames (M)')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        # Stacked area: win rates
        ax_wr = axes[3, 0]
        aw = h.get('attacker_win_rate', [])
        dw = h.get('defender_win_rate', [])
        dr = h.get('draw_rate', [])
        if aw and len(aw) == len(frames_m):
            ax_wr.stackplot(
                frames_m, aw, dr, dw,
                labels=['Attacker Win', 'Draw', 'Defender Win'],
                colors=['#4CAF50', '#9E9E9E', '#F44336'], alpha=0.85
            )
            ax_wr.set_ylim(0, 1)
            ax_wr.set_title('Game Outcomes', fontweight='bold')
            ax_wr.set_xlabel('Frames (M)')
            ax_wr.set_ylabel('Rate')
            ax_wr.legend(loc='upper right', fontsize=8)
            ax_wr.grid(True, alpha=0.2)

        # Win rate lines
        ax_wrl = axes[3, 1]
        if aw and len(aw) == len(frames_m):
            ax_wrl.plot(frames_m, aw, linewidth=1.5, color='#4CAF50', label='Attacker')
            ax_wrl.plot(frames_m, dw, linewidth=1.5, color='#F44336', label='Defender')
            ax_wrl.plot(frames_m, dr, linewidth=1.5, color='#9E9E9E', label='Draw')
            ax_wrl.axhline(0.5, color='black', linestyle='--', alpha=0.3)
            ax_wrl.set_title('Win Rates', fontweight='bold')
            ax_wrl.set_xlabel('Frames (M)')
            ax_wrl.set_ylabel('Rate')
            ax_wrl.legend(fontsize=8)
            ax_wrl.grid(True, alpha=0.3)

        # Dual axis: value loss vs accuracy
        ax_dual = axes[3, 2]
        vl = h.get('value_loss', [])
        va = h.get('value_acc', [])
        if vl and len(vl) == len(frames_m):
            c_loss, c_acc = '#1f77b4', '#ff7f0e'
            ax_dual.plot(frames_m, vl, linewidth=1.5, color=c_loss, label='Value Loss')
            ax_dual.set_ylabel('Value Loss', color=c_loss)
            ax_dual.tick_params(axis='y', labelcolor=c_loss)

            ax2 = ax_dual.twinx()
            ax2.plot(frames_m, va, linewidth=1.5, color=c_acc, label='Value Acc')
            ax2.set_ylabel('Value Accuracy', color=c_acc)
            ax2.tick_params(axis='y', labelcolor=c_acc)

            ax_dual.set_title('Value Head: Loss vs Accuracy', fontweight='bold')
            ax_dual.set_xlabel('Frames (M)')
            ax_dual.grid(True, alpha=0.3)

            lines1, labels1 = ax_dual.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax_dual.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        plot_path = plots_dir / f"All_Metrics_{timestamp}.pdf"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved to {plot_path}")

    def save_metrics(self) -> None:
        """
        Writes current metrics history into a JSON file and backups the PGN.
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        filename = f"metrics_{timestamp}.json"
        file_dir = self.dirs['metrics'] / filename
        with open(file_dir, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)

        if self.dirs['pgn'].exists():
            pgn_backup = self.dirs['metrics'] / f"game_results_{timestamp}.pgn"
            shutil.copy(self.dirs['pgn'], pgn_backup)

    def _load_metrics(self, load_checkpoint: bool) -> dict[str, list[float]]:
        """
        Loads the most recent metrics history from the disk if load_checkpoint is true.
        Returns the previous metrics or a new dict.
        """
        default_metrics = {
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'value_acc': [],
            'attacker_win_rate': [],
            'defender_win_rate': [],
            'draw_rate': [],
            'frames': [],
            'game_lengths': [],
            'pieces_left': [],
            'entropy': [],
            'attacker_ev': [],
            'attacker_score': []
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