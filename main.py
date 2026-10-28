from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.metrics import MetricsTracker
from src.tablut.play import PlayTablut
from src.train import Coach
from src.utils import create_path_dict


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(cfg: DictConfig):
    #current_dir = Path(__file__).resolve().parent
    #dir_dict = create_path_dict(current_dir)

    #coach = Coach(cfg)
    #coach.train()
    #mt = MetricsTracker(cfg, dir_dict)

    #mt.plot_elo()
    #mt.plot_loss()
    #mt.plot_entropy()
    #mt.plot_results()
    #mt.plot_avg_pieces()
    #mt.plot_elo_comparison()

    game = PlayTablut(mcts_sims=150)
    game.play_ui()

if __name__ == '__main__':
    main()