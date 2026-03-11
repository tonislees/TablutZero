import jax
from flax import nnx
import jax.numpy as jnp

from src.hnefatafl.hnefatafl import Hnefatafl
from src.hnefatafl.hnefatafl_jax import GameState, Action, BOARD_EDGE
from src.hnefatafl.ui import HnefataflUI

FILE_LETTERS = 'abcdefghijk'


class PlayHnefatafl:
    def __init__(self):
        self.env = Hnefatafl()
        self.seed = 42
        self.rngs: nnx.Rngs = nnx.Rngs(self.seed)
        self.key_env = jax.random.PRNGKey(self.seed + 1)
        batched_state = jax.jit(jax.vmap(self.env.init))(
            jax.random.split(self.key_env, 1)
        )
        self.state = jax.tree_util.tree_map(lambda x: x[0], batched_state)
        self.step_fn = jax.jit(self.env.step)
        self.game_state: GameState = self.state._x
        self.board_edge = BOARD_EDGE
        self.columns = {FILE_LETTERS[i]: i for i in range(BOARD_EDGE)}

    def reset(self):
        batched_state = jax.jit(jax.vmap(self.env.init))(
            jax.random.split(self.key_env, 1)
        )
        self.state = jax.tree_util.tree_map(lambda x: x[0], batched_state)
        self.game_state: GameState = self.state._x

    def print_board(self):
        board = self.game_state.board
        board = board.reshape((self.board_edge, self.board_edge))
        pieces = [' ', 'O', 'K', 'X']
        player = int(self.game_state.color)

        border = '   ' + self.board_edge * '──────'

        board_strings = [border]
        for row in range(self.board_edge):
            row_str = f'{self.board_edge - row}  |  '
            for column in range(self.board_edge):
                row_str += f'{pieces[int(board[row, column] * player)]}  |  '
            board_strings.append(row_str)
            board_strings.append(border)
        board_strings.append('      ' + '     '.join(list(self.columns.keys())))
        print('\n'.join(row for row in board_strings))

    def _sq_to_uci(self, sq):
        row = sq // self.board_edge
        col = sq % self.board_edge
        rank_str = str(row + 1)
        file_str = FILE_LETTERS[col]
        return file_str + rank_str

    def uci_to_action(self, uci: str):
        from_char = uci[0]

        to_char_idx = -1
        for i, char in enumerate(uci[1:], start=1):
            if char.isalpha():
                to_char_idx = i
                break

        if to_char_idx == -1:
            raise ValueError("Invalid UCI format")

        from_rank_str = uci[1:to_char_idx]
        to_char = uci[to_char_idx]
        to_rank_str = uci[to_char_idx + 1:]

        from_col = self.columns[from_char]
        from_row = self.board_edge - int(from_rank_str)

        to_col = self.columns[to_char]
        to_row = self.board_edge - int(to_rank_str)

        from_sq = from_row * self.board_edge + from_col
        to_sq = to_row * self.board_edge + to_col

        action = Action(from_sq, to_sq)
        label = action.to_label()

        if not self.state.legal_action_mask[label]:
            print(f"Debug: Move {uci} (Indices {from_sq}->{to_sq}) is masked out.")
            raise ValueError("Illegal move")

        return label

    def show_legal_moves(self):
        legal_action_mask = self.state.legal_action_mask
        valid_indices = jnp.where(legal_action_mask)[0]

        moves_uci = []
        for label_idx in valid_indices.tolist():
            action = Action.from_label(label_idx)
            f_sq = int(action.from_sq)
            t_sq = int(action.to_sq)
            uci = self._sq_to_uci(f_sq) + self._sq_to_uci(t_sq)
            moves_uci.append(uci)
        moves_uci.sort()
        print(f"Legal Moves ({len(moves_uci)}):", ", ".join(moves_uci))

    def make_move(self, move):
        try:
            action = self.uci_to_action(move)
            print(action)
            self.state = self.step_fn(self.state, action)
            self.game_state = self.state._x
        except Exception as e:
            print('Wrong format or illegal move: {}', e)

    def game_loop(self):
        self.print_board()
        while True:
            self.show_legal_moves()
            print('Attacker to move' if self.game_state.color < 0 else 'Defender to move')
            move = input('Your move: ')
            self.make_move(move)
            self.print_board()
            if self.env.game.is_terminal(self.game_state):
                print('Defenders' if self.env.game.rewards(self.game_state)[0] < 0 else 'Attackers' + ' won')
                break

    def play_ui(self):
        ui = HnefataflUI(self)
        ui.run()

    def test(self):
        tests = {
            'Attacker Win (King Capture)': [
                "f1g1", "f5f1", "f9f5", "e6h6",
                "a6e6", "d5d8", "d1d5", "e4h4", "a4e4"
            ],
            'Defender Win (Edge Fort)': [
                "f1h1", "g5g1", "e2a2", "f5f2", "d1b1", "c5c1", "e1d1", "e3e1", "b5c5", "e4d4",
                "h5h4", "e1d1", "h4h5", "e5e1", "h5h4", "e6e3", "h4h5", "d4d2", "h5h4", "d5d3"
            ],
            'Attacker Loss (Repetition)': [
                "a4a3", "d5d6", "a3a4", "d6d5", "a4a3", "d5d6", "a3a4", "d6d5", "a4a3", "d5d6", "a3a4", "d6d5"
            ],
            'Attacker Win (Encirclement)': [
                "f1f3", "f5f4", "i4g4", "f4f5", "i6g6", "f5f4", "f9f7",
                "f4f5", "d9d7", "f5f4", "a6c6", "f4f5", "a4c4", "f5f4", "d1d3"
            ]
        }

        for name, sequence in tests.items():
            print(f"\n{'=' * 10} Testing: {name} {'=' * 10}")
            self.reset()

            error = False
            for i, move in enumerate(sequence):
                try:
                    action = self.uci_to_action(move)
                    self.state = self.step_fn(self.state, action)
                    self.game_state = self.state._x
                except Exception:
                    print(f"FAILED at step {i + 1}: Illegal move {move}")
                    self.print_board()
                    error = True
                    break

            if not error:
                self.print_board()
                rewards = self.env.game.rewards(self.game_state)
                term = self.env.game.is_terminal(self.game_state)

                if not term:
                    print("RESULT: Game Not Finished")
                elif rewards[0] > 0:
                    print("RESULT: Attackers Won (+1)")
                elif rewards[1] > 0:
                    print("RESULT: Defenders Won (+1)")
                else:
                    print("RESULT: Draw (0)")


if __name__ == '__main__':
    game = PlayHnefatafl()
    game.play_ui()