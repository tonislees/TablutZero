from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

BOARD_EDGE = 11
BOARD_SIZE = BOARD_EDGE * BOARD_EDGE # 121
THRONE = BOARD_SIZE // 2 # 60
MAX_SHIELD_WALL_PARTNERS = BOARD_EDGE - 4

ACTION_PLANES = 4 * (BOARD_EDGE - 1)

EMPTY, TAFLMAN, KING = tuple(range(3))
NUM_ATTACKERS = (BOARD_EDGE - 5) * 4
MAX_TERMINATION_STEPS = 512
MAX_HALF_MOVE_COUNT: float = 100.0

INIT_BOARD = jnp.int32([
    0,  0,  0, -1, -1, -1, -1, -1,  0,  0,  0,
    0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   -1,  0,  0,  0,  0,  1,  0,  0,  0,  0, -1,
   -1,  0,  0,  0,  1,  1,  1,  0,  0,  0, -1,
   -1, -1,  0,  1,  1,  2,  1,  1,  0, -1, -1,
   -1,  0,  0,  0,  1,  1,  1,  0,  0,  0, -1,
   -1,  0,  0,  0,  0,  1,  0,  0,  0,  0, -1,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,
    0,  0,  0, -1, -1, -1, -1, -1,  0,  0,  0,
])

ZERO_INIT_ACTION_MASK = jnp.zeros(BOARD_SIZE * ACTION_PLANES, dtype=jnp.bool_)

# 11 110 111 112 113 114 115 116 117 118 119 120
# 10  99 100 101 102 103 104 105 106 107 108 109
#  9  88  89  90  91  92  93  94  95  96  97  98
#  8  77  78  79  80  81  82  83  84  85  86  87
#  7  66  67  68  69  70  71  72  73  74  75  76
#  6  55  56  57  58  59  60  61  62  63  64  65
#  5  44  45  46  47  48  49  50  51  52  53  54
#  4  33  34  35  36  37  38  39  40  41  42  43
#  3  22  23  24  25  26  27  28  29  30  31  32
#  2  11  12  13  14  15  16  17  18  19  20  21
#  1   0   1   2   3   4   5   6   7   8   9  10
#      a   b   c   d   e   f   g   h   i   j   k

# Action: AlphaZero style labels (BOARD_SIZE x ACTION_PLANES = 4840)
#                                 19
#                                 18
#                                 17
#                                 16
#                                 15
#                                 14
#                                 13
#                                 12
#                                 11
#                                 10
#   39 38 37 36 35 34 33 32 31 30  X  0  1  2  3  4  5  6  7  8  9
#                                 20
#                                 21
#                                 22
#                                 23
#                                 24
#                                 25
#                                 26
#                                 27
#                                 28
#                                 29


def calc_hostile_squares():
    """Calculate the hostile squares: corners and throne."""
    bottom_left = 0
    bottom_right = BOARD_EDGE
    top_left = BOARD_SIZE - BOARD_EDGE
    top_right = BOARD_SIZE

    corners = [bottom_left, bottom_right - 1, top_left, top_right - 1]

    hostile_squares_mask = np.zeros(BOARD_SIZE, dtype=np.bool)
    hostile_squares_mask[corners] = True
    corners_mask = hostile_squares_mask.copy()
    hostile_squares_mask[THRONE] = True
    return hostile_squares_mask, corners_mask


HOSTILE_SQUARES_MASK, CORNERS_MASK = calc_hostile_squares()


def calc_rows_columns():
    """Calculate matrices for row and column indices."""
    rows = np.zeros((BOARD_EDGE, BOARD_EDGE), dtype=np.int32)
    columns = np.zeros((BOARD_EDGE, BOARD_EDGE), dtype=np.int32)

    for i in range(BOARD_SIZE):
        row = i // BOARD_EDGE
        column = i % BOARD_EDGE
        row_idx = column
        column_idx = row

        rows[row, row_idx] = i
        columns[column, column_idx] = i
    return rows, columns


ROWS, COLUMNS = calc_rows_columns()


def calc_edges():
    """Calculate arrays for edge indices, possible shield wall partners,
    and neighbor indices for each shield wall inner neighbor"""
    top = ROWS[BOARD_EDGE - 1]
    bottom = ROWS[0]
    left = COLUMNS[0]
    right = COLUMNS[BOARD_EDGE - 1]

    edges = np.zeros(BOARD_SIZE, dtype=np.bool)
    all_edge_indices = np.unique(np.concatenate([top, bottom, left, right]))
    edges[all_edge_indices] = True

    inner_neighbor = -np.ones(BOARD_SIZE, dtype=np.int32)
    inner_neighbor[top] = top - BOARD_EDGE
    inner_neighbor[bottom] = bottom + BOARD_EDGE
    inner_neighbor[left] = left + 1
    inner_neighbor[right] = right - 1

    shield_wall_partners = -np.ones((BOARD_SIZE, MAX_SHIELD_WALL_PARTNERS), dtype=np.int32)

    for edge in [top, right, bottom, left]:
        for current_sq in edge:
            if HOSTILE_SQUARES_MASK[current_sq]:
                continue
            valid_p = []
            current_idx = np.where(edge == current_sq)[0][0]
            for other_sq in edge:
                other_idx = np.where(edge == other_sq)[0][0]
                dist = abs(current_idx - other_idx)
                if dist >= 3:
                    valid_p.append(other_sq)
            shield_wall_partners[current_sq, :len(valid_p)] = valid_p

    return edges, inner_neighbor, shield_wall_partners


EDGES, INNER_NEIGHBOR, SHIELD_WALL_PARTNERS = calc_edges()


def calc_action_arrays():
    """Calculate mappings for conversions between Action objects and action indices."""

    from_plane = -np.ones((BOARD_SIZE, ACTION_PLANES), dtype=np.int32)
    to_plane = -np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)

    max_action_length = BOARD_EDGE - 1
    zeros = [0] * max_action_length

    delta_col = list(range(1, BOARD_EDGE)) + zeros + zeros + list(range(-1, -BOARD_EDGE, -1))
    delta_row = zeros + list(range(1, BOARD_EDGE)) + list(range(-1, -BOARD_EDGE, -1)) + zeros

    # Create mappings for square -> action
    # move with action plane from square from_
    for from_sq in range(BOARD_SIZE):
        from_row = from_sq // BOARD_EDGE
        from_col = from_sq % BOARD_EDGE
        for action in range(ACTION_PLANES):
            to_row = from_row + delta_row[action]
            to_col = from_col + delta_col[action]

            # Check if the action fits on the board
            if 0 <= to_row < BOARD_EDGE and 0 <= to_col < BOARD_EDGE:
                to = to_row * BOARD_EDGE + to_col
                from_plane[from_sq, action] = to
                to_plane[from_sq, to] = action
    return from_plane, to_plane


FROM_PLANE, TO_PLANE = calc_action_arrays()


def calc_capture_arrays():
    """Calculate the arrays representing attack pairs and neighboring squares."""

    attack_pair = -np.ones((BOARD_SIZE, 4), dtype=np.int32)
    neighbors = -np.ones((BOARD_SIZE, 4), dtype=np.int32)
    for to_sq in range(BOARD_SIZE):
        row, col = to_sq // BOARD_EDGE, to_sq % BOARD_EDGE
        #UP
        if row - 1 >= 0:
            neighbors[to_sq, 0] = (row - 1) * BOARD_EDGE + col

        if row - 2 >= 0:
            attack_pair[to_sq, 0] = (row - 2) * BOARD_EDGE + col
        #RIGHT
        if col + 1 < BOARD_EDGE:
            neighbors[to_sq, 1] = row * BOARD_EDGE + (col + 1)

        if col + 2 < BOARD_EDGE:
            attack_pair[to_sq, 1] = row * BOARD_EDGE + (col + 2)
        #DOWN
        if row + 1 < BOARD_EDGE:
            neighbors[to_sq, 2] = (row + 1) * BOARD_EDGE + col

        if row + 2 < BOARD_EDGE:
            attack_pair[to_sq, 2] = (row + 2) * BOARD_EDGE + col
        #LEFT
        if row - 1 >= 0:
            neighbors[to_sq, 3] = row * BOARD_EDGE + (col - 1)

        if col - 2 >= 0:
            attack_pair[to_sq, 3] = row * BOARD_EDGE + (col - 2)

    return attack_pair, neighbors


ATTACK_PAIR, NEIGHBORS = calc_capture_arrays()


def calc_action_legality_arrays():
    """Calculate a legal destinations array for each square."""

    legal_dest = -np.ones((3, BOARD_SIZE, BOARD_SIZE), dtype=np.int32)

    for from_sq in range(BOARD_SIZE):
        legal_dest_for_sq = {p: [] for p in range(1, 3)}
        for to_sq in range(BOARD_SIZE):
            if from_sq == to_sq: continue
            row_from, col_from, row_to, col_to = from_sq // BOARD_EDGE, from_sq % BOARD_EDGE, to_sq // BOARD_EDGE, to_sq % BOARD_EDGE

            if abs(row_to - row_from) == 0 or abs(col_to - col_from) == 0:
                if not HOSTILE_SQUARES_MASK[to_sq]:
                    legal_dest_for_sq[TAFLMAN].append(to_sq)
                legal_dest_for_sq[KING].append(to_sq)
        for piece in range(1, 3):
            legal_dest[piece, from_sq, : len(legal_dest_for_sq[piece])] = legal_dest_for_sq[piece]

    return legal_dest


LEGAL_DEST = calc_action_legality_arrays()


def calc_between_squares():
    """Calculate all squares indices between two squares."""

    between = -np.ones((BOARD_SIZE, BOARD_SIZE, BOARD_EDGE - 2), dtype=np.int32)
    for from_sq in range(BOARD_SIZE):
        for to_sq in range(BOARD_SIZE):
            row_from, col_from, row_to, col_to = from_sq // BOARD_EDGE, from_sq % BOARD_EDGE, to_sq // BOARD_EDGE, to_sq % BOARD_EDGE
            if not (abs(row_to - row_from) == 0 or abs(col_to - col_from) == 0):
                continue
            row_sign, col_sign = max(min(row_to - row_from, 1), -1), max(min(col_to - col_from, 1), -1)
            for i in range(BOARD_EDGE - 2):
                row = row_from + row_sign * (i + 1)
                col = col_from + col_sign * (i + 1)
                if row == row_to and col == col_to:
                    break
                between[from_sq, to_sq, i] = row * BOARD_EDGE + col
    return between


BETWEEN = calc_between_squares()

(FROM_PLANE, TO_PLANE, LEGAL_DEST,
 BETWEEN, EDGES, ATTACK_PAIR, NEIGHBORS, INNER_NEIGHBOR,
 SHIELD_WALL_PARTNERS, HOSTILE_SQUARES_MASK, ROWS, COLUMNS) = (
    jnp.array(x) for x in
    (FROM_PLANE, TO_PLANE, LEGAL_DEST,
     BETWEEN, EDGES, ATTACK_PAIR, NEIGHBORS, INNER_NEIGHBOR,
     SHIELD_WALL_PARTNERS, HOSTILE_SQUARES_MASK, ROWS, COLUMNS))

keys = jax.random.split(jax.random.PRNGKey(12345), 4)
ZOBRIST_BOARD = jax.random.randint(keys[0], shape=(BOARD_SIZE, 5, 2), minval=0, maxval=2 ** 31 - 1, dtype=jnp.uint32)
ZOBRIST_SIDE = jax.random.randint(keys[1], shape=(2,), minval=0, maxval=2 ** 31 - 1, dtype=jnp.uint32)


class GameState(NamedTuple):
    color: Array = jnp.int32(-1)  # attacker: -1, defender: 1
    board: Array = -INIT_BOARD
    board_history: Array = jnp.zeros((8, BOARD_SIZE), dtype=jnp.int32).at[0, :].set(-INIT_BOARD)
    hash_history: Array = jnp.zeros((MAX_TERMINATION_STEPS + 1, 2), dtype=jnp.uint32)
    legal_action_mask: Array = ZERO_INIT_ACTION_MASK
    step_count: Array = jnp.int32(0)
    half_move_count: Array = jnp.int32(0)


class Action(NamedTuple):
    from_sq: Array = jnp.int32(-1)
    to_sq: Array = jnp.int32(-1)

    @staticmethod
    def from_label(label: Array):
        from_sq, plane = label // ACTION_PLANES, label % ACTION_PLANES
        return Action(from_sq=from_sq, to_sq=FROM_PLANE[from_sq, plane])

    def to_label(self):
        return self.from_sq * ACTION_PLANES + TO_PLANE[self.from_sq, self.to_sq]


def legal_moves(state: GameState, from_sq: Array) -> Array:
    """Calculate all legal moves """

    piece = state.board[from_sq]

    def legal_label(to_sq) -> Array:
        dest_valid = (to_sq >= 0) & (to_sq < BOARD_SIZE)

        between_idxs = BETWEEN[from_sq, to_sq]
        path_clear = jnp.all((between_idxs == -1) | (state.board[between_idxs] == EMPTY))
        target_empty = state.board[to_sq] == EMPTY

        ok = dest_valid & path_clear & (piece > 0) & target_empty

        return lax.select(ok, Action(from_sq=from_sq, to_sq=to_sq).to_label(), -1)

    return jax.vmap(legal_label)(LEGAL_DEST[piece, from_sq])


def _legal_action_mask(state: GameState) -> Array:
    """Calculate the legal moves mask of the current game state."""

    possible_piece_positions = jnp.nonzero(state.board > 0, size=NUM_ATTACKERS, fill_value=-1)[0]
    actions = jax.vmap(lambda p: legal_moves(state, p))(possible_piece_positions).flatten()

    mask = jnp.zeros(BOARD_SIZE * ACTION_PLANES, jnp.bool_)
    mask = mask.at[actions].set(actions >= 0)

    return mask


def initialize_legal_actions(state: GameState):
    """
    Dynamically calculates the mask for the initial board state.
    This replaces the hardcoded dictionary.
    """
    return _legal_action_mask(state)

_init_check_state = GameState(board=-INIT_BOARD, color=jnp.int32(-1))
INIT_LEGAL_ACTION_MASK = initialize_legal_actions(_init_check_state)


class Game:
    @staticmethod
    def init() -> GameState:
        dummy_state = GameState()
        initial_hash = _zobrist_hash(dummy_state)

        return dummy_state._replace(
            legal_action_mask=INIT_LEGAL_ACTION_MASK,
            hash_history=jnp.zeros((MAX_TERMINATION_STEPS + 1, 2), dtype=jnp.uint32).at[0].set(initial_hash)
        )

    @staticmethod
    def step(state: GameState, action: Array):
        state = _apply_move(state, Action.from_label(action))
        state = _flip(state)
        state = _update_history(state)
        state = state._replace(legal_action_mask=_legal_action_mask(state))
        state = state._replace(step_count=state.step_count + 1)
        return state

    @staticmethod
    def mcts_step(state: GameState, action: Array):
        state = _apply_move(state, Action.from_label(action))
        state = _flip(state)
        state = _update_history(state)
        state = state._replace(step_count=state.step_count + 1)
        return state

    @staticmethod
    def observe(state: GameState):
        ones = jnp.ones((1, BOARD_EDGE, BOARD_EDGE), dtype=jnp.float32)
        color = (state.color + 1) // 2

        history_2d = state.board_history.reshape((8, BOARD_EDGE, BOARD_EDGE))
        board = jnp.rot90(history_2d, k=1, axes=(1, 2))

        friendly_pieces = (board == TAFLMAN).astype(jnp.float32)
        enemy_pieces = (board == -TAFLMAN).astype(jnp.float32)
        king = (jnp.abs(board) == KING).astype(jnp.float32)

        hash_ = state.hash_history[:8, :]
        rep = (state.hash_history == hash_[:, None, :]).all(axis=2).sum(axis=1) - 1
        rep = jnp.where((hash_ == 0).all(axis=1), 0, rep)

        rep0 = jnp.broadcast_to((rep >= 1)[:, None, None], (8, BOARD_EDGE, BOARD_EDGE)).astype(jnp.float32)
        rep1 = jnp.broadcast_to((rep >= 2)[:, None, None], (8, BOARD_EDGE, BOARD_EDGE)).astype(jnp.float32)

        history_features = jnp.stack([friendly_pieces, enemy_pieces, king, rep0, rep1], axis=1)
        history_features = history_features.reshape(40, BOARD_EDGE, BOARD_EDGE)

        return jnp.vstack(
            [
                history_features,
                color * ones,
                (state.step_count / MAX_TERMINATION_STEPS) * ones,
                (state.half_move_count.astype(jnp.float32) / MAX_HALF_MOVE_COUNT) * ones
            ]
        ).transpose((1, 2, 0))

    @staticmethod
    def legal_action_mask(state: GameState):
        return state.legal_action_mask

    @staticmethod
    def is_terminal(state: GameState):
        # Stalemate
        terminated = ~state.legal_action_mask.any()

        # King escaped
        king_pos_mask = jnp.abs(state.board) == 2
        terminated |= (king_pos_mask & CORNERS_MASK).any()
        terminated |= _check_edge_fort(state)

        # King captured or encircled
        terminated |= _check_king_captured(state)
        terminated |= _check_encirclement(state)

        # Repetition
        repetition = (state.hash_history == _zobrist_hash(state)).all(axis=1).sum() - 1
        terminated |= repetition >= 2

        # Draw conditions
        terminated |= state.half_move_count >= MAX_HALF_MOVE_COUNT
        terminated |= MAX_TERMINATION_STEPS <= state.step_count

        return terminated

    @staticmethod
    def mcts_status(state: GameState):
        king_captured = _check_king_captured(state)
        encirclement = _check_encirclement(state)
        attacker_won = king_captured | encirclement

        king_pos_mask = jnp.abs(state.board) == 2
        king_on_corner = (king_pos_mask & CORNERS_MASK).any()
        edge_fort = _check_edge_fort(state)
        defender_won = king_on_corner | edge_fort

        draw = (state.half_move_count >= MAX_HALF_MOVE_COUNT) | (state.step_count >= MAX_TERMINATION_STEPS)

        terminated = attacker_won | defender_won | draw

        attacker_score = jnp.where(attacker_won, 1.0, jnp.where(defender_won, -1.0, 0.0))
        defender_score = jnp.where(defender_won, 1.0, jnp.where(attacker_won, -1.0, 0.0))

        return terminated, jnp.array([attacker_score, defender_score], dtype=jnp.float32)

    @staticmethod
    def rewards(state: GameState):
        # Attackers win
        king_captured = _check_king_captured(state)
        encircled = _check_encirclement(state)

        # Defenders win
        king_on_corner = ((jnp.abs(state.board) == KING) & CORNERS_MASK).any()
        fort = _check_edge_fort(state)

        # Loss
        repetition = (state.hash_history == _zobrist_hash(state)).all(axis=1).sum() - 1
        rep_loss = repetition >= 2
        no_moves = ~state.legal_action_mask.any()

        # Technical Draws
        draw = (state.half_move_count >= MAX_HALF_MOVE_COUNT) | (state.step_count >= MAX_TERMINATION_STEPS)

        attacker_score = jnp.float32(0.0)
        defender_score = jnp.float32(0.0)

        attacker_won = king_captured | encircled
        defender_won = king_on_corner | fort

        attacker_won |= rep_loss & (state.color == -1)
        defender_won |= rep_loss & (state.color == 1)

        attacker_won |= no_moves & (state.color == 1)
        defender_won |= no_moves & (state.color == -1)

        attacker_score = lax.select(attacker_won, 1.0, attacker_score)
        defender_score = lax.select(attacker_won, -1.0, defender_score)

        defender_score = lax.select(defender_won, 1.0, defender_score)
        attacker_score = lax.select(defender_won, -1.0, attacker_score)

        is_draw = draw & (~attacker_won) & (~defender_won)
        attacker_score = lax.select(is_draw, 0.0, attacker_score)
        defender_score = lax.select(is_draw, 0.0, defender_score)

        return jnp.array([attacker_score, defender_score])


def _check_king_captured(state: GameState):
    king_val = KING * state.color
    king_pos_arr = jnp.where(state.board == king_val, size=1)[0]
    king_pos = king_pos_arr[0]

    neighbor_indices = NEIGHBORS[king_pos]
    neighbor_vals = jnp.take(state.board, neighbor_indices, mode='fill', fill_value=0)
    is_attacker = (neighbor_vals == -state.color * TAFLMAN)
    safe_neighbor_indices = jnp.maximum(neighbor_indices, 0)
    is_hostile_sq = jnp.take(HOSTILE_SQUARES_MASK, safe_neighbor_indices, mode='fill', fill_value=False)
    is_valid_attacker = is_hostile_sq & (neighbor_indices != -1) & (neighbor_vals == EMPTY)

    captured = (is_attacker | is_valid_attacker).all()

    return captured


def _shift_up(grid):
    return jnp.concatenate([grid[1:], jnp.zeros((1, BOARD_EDGE), dtype=bool)], axis=0)


def _shift_down(grid):
    return jnp.concatenate([jnp.zeros((1, BOARD_EDGE), dtype=bool), grid[:-1]], axis=0)


def _shift_left(grid):
    return jnp.concatenate([grid[:, 1:], jnp.zeros((BOARD_EDGE, 1), dtype=bool)], axis=1)


def _shift_right(grid):
    return jnp.concatenate([jnp.zeros((BOARD_EDGE, 1), dtype=bool), grid[:, :-1]], axis=1)


def _flood_fill(start_mask: Array, valid_mask: Array, max_iterations: int = 30) -> Array:
    """Expands a boolean mask into adjacent squares defined by valid_mask."""

    def cond_fn(val):
        i, prev_mask, curr_mask = val
        return (i < max_iterations) & jnp.any(prev_mask != curr_mask)

    def body_fn(val):
        i, _, current_mask = val
        grid = current_mask.reshape((BOARD_EDGE, BOARD_EDGE))
        neighbors = _shift_up(grid) | _shift_down(grid) | _shift_left(grid) | _shift_right(grid)
        next_mask = current_mask | (neighbors.flatten() & valid_mask)
        return i + 1, current_mask, next_mask

    # Start loop: index 0, previous mask (empty), current mask (start)
    _, _, final_mask = lax.while_loop(
        cond_fn,
        body_fn,
        (0, jnp.zeros_like(start_mask), start_mask)
    )
    return final_mask


def _check_edge_fort(state: GameState):
    """Check if the King has formed an unbreakable 'Exit Fort' on the edge."""

    def _check():
        king_pos_mask = (state.board == -KING)
        king_on_edge = (king_pos_mask & EDGES).any()
        def _do_flood_fill(_):
            empty_mask = (state.board == 0)
            defender_mask = (state.board == -TAFLMAN)
            attacker_mask = (state.board == TAFLMAN)

            flood = _flood_fill(start_mask=king_pos_mask, valid_mask=empty_mask)

            # Identify the wall
            flood_grid = flood.reshape(BOARD_EDGE, BOARD_EDGE)
            flood_neighbors = (_shift_up(flood_grid) | _shift_down(flood_grid) |
                               _shift_left(flood_grid) | _shift_right(flood_grid)).flatten()

            # The Wall consists of Defenders that touch the King's flood
            wall_mask = flood_neighbors & defender_mask

            has_room = jnp.sum(flood) > 1
            exposed_to_attacker = (flood_neighbors & attacker_mask).any()

            empty_grid = empty_mask.reshape(BOARD_EDGE, BOARD_EDGE)
            attacker_grid = attacker_mask.reshape(BOARD_EDGE, BOARD_EDGE)
            wall_grid = wall_mask.reshape(BOARD_EDGE, BOARD_EDGE)

            threat_grid = (empty_grid & ~flood_grid) | attacker_grid
            vertical_threats = _shift_down(threat_grid) & _shift_up(threat_grid)
            horizontal_threats = _shift_right(threat_grid) & _shift_left(threat_grid)

            wall_broken = (wall_grid & (vertical_threats | horizontal_threats)).any()
            return has_room & (~exposed_to_attacker) & (~wall_broken)
        return lax.cond(king_on_edge, _do_flood_fill, lambda _: False, operand=None)

    return lax.cond(
        state.color == -1,
        lambda _: _check(),
        lambda _: False,
        operand=None
    )


def _check_encirclement(state: GameState):
    """Checks if the Defenders are completely cut off from the board edge.
    This check runs when it is the DEFENDER'S turn (after Attackers moved)."""

    def compute_encirclement(_):
        # Perspective: Defender's Turn (color == 1)
        # King = 2, Defender = 1, Empty = 0, Attacker = -1

        king_pos_mask = (state.board > 0)
        defenders_mask = (state.board >= 0)

        flood = _flood_fill(start_mask=king_pos_mask, valid_mask=defenders_mask)

        # Check if the flood touched the edge
        escaped = (flood & EDGES).any()
        return ~escaped

    return lax.cond(
        state.color == 1,
        compute_encirclement,
        lambda _: False,
        operand=None
    )


def _check_shield_wall(state: GameState, to_sq):
    king_val = KING * state.color
    king_pos_arr = jnp.where(state.board == king_val, size=1, fill_value=-1)[0]
    king_pos = king_pos_arr[0]
    is_edge_move = EDGES[to_sq]

    def check_partners():
        candidates = SHIELD_WALL_PARTNERS[to_sq]

        def check_one_partner(partner_sq):
            partner_piece = state.board[partner_sq]

            is_valid_candidate = (partner_sq != -1)
            is_partner_friendly = (partner_piece > 0)
            is_partner_corner = HOSTILE_SQUARES_MASK[jnp.maximum(partner_sq, 0)]

            is_valid_partner = is_valid_candidate & (is_partner_friendly | is_partner_corner)

            between_sqs = BETWEEN[to_sq, partner_sq]
            path_mask = (between_sqs != -1)

            victim_pieces = jnp.take(state.board, between_sqs, mode='fill', fill_value=0)
            inner_sqs = jnp.take(INNER_NEIGHBOR, between_sqs, mode='fill', fill_value=0)
            inner_pieces = jnp.take(state.board, inner_sqs, mode='fill', fill_value=0)

            # Check for enemies between partners
            is_enemy = (victim_pieces < 0)

            # Check if the shield wall covers enemies
            is_wall_intact = (inner_pieces > 0)
            path_valid = jnp.all(jnp.where(path_mask, is_enemy & is_wall_intact, True))
            do_capture = is_valid_partner & path_valid

            return jnp.where(do_capture, between_sqs, -1)

        captured_batch = jax.vmap(check_one_partner)(candidates)
        return captured_batch.flatten()

    captured_indices = lax.cond(
        is_edge_move,
        lambda _: check_partners(),
        lambda _: -jnp.ones(MAX_SHIELD_WALL_PARTNERS * (BOARD_EDGE - 2), dtype=jnp.int32),
        operand=None
    )

    update_mask = (captured_indices != -1)
    safe_indices = jnp.where(update_mask, captured_indices, 0)
    new_values = jnp.where(update_mask, EMPTY, state.board[safe_indices])

    new_board = state.board.at[safe_indices].set(new_values)
    # Ensure king is preserved at its current position (which might have changed if it moved)
    new_board = jnp.where(king_pos != -1, new_board.at[king_pos].set(king_val), new_board)

    return state._replace(board=new_board)


def _flip(state: GameState):
    return state._replace(
        board=-state.board,
        color=-state.color,
        board_history=-state.board_history
    )


def _check_captures(state: GameState, to_sq):
    attack_indices = ATTACK_PAIR[to_sq]
    victim_indices = NEIGHBORS[to_sq]

    # Piece types of attackers and victims
    attack_pieces = jnp.take(state.board, attack_indices, mode='fill', fill_value=0)
    victim_pieces = jnp.take(state.board, victim_indices, mode='fill', fill_value=0)

    is_victim_enemy = (victim_pieces == -1)  # Check if victims are enemies
    is_attacker_friendly = (attack_pieces > 0)  # Check if attackers are friendly

    is_attacker_square_hostile = jnp.take(HOSTILE_SQUARES_MASK, jnp.maximum(attack_indices, 0), mode='fill',
                                          fill_value=False)
    is_attacker_square_hostile &= (
            attack_pieces != -KING)  # If Defenders king is on the throne, it isn't hostile for defenders

    capture_mask = (attack_indices != -1) & is_victim_enemy & (is_attacker_friendly | is_attacker_square_hostile)

    safe_victim_indices = jnp.maximum(victim_indices, 0)
    current_values = state.board[safe_victim_indices]
    new_values = jnp.where(capture_mask, EMPTY, current_values)

    return state._replace(board=state.board.at[safe_victim_indices].set(new_values))


def _apply_move(state: GameState, action: Action):
    piece = state.board[action.from_sq]

    # Move the piece
    state = state._replace(board=state.board.at[action.from_sq].set(EMPTY).at[action.to_sq].set(piece))

    # Check for captures
    pieces_before = jnp.count_nonzero(state.board)
    state = _check_captures(state, action.to_sq)

    # Check for shield wall
    state = _check_shield_wall(state, action.to_sq)

    pieces_after = jnp.count_nonzero(state.board)
    is_capture = pieces_after < pieces_before
    half_move_count = lax.select(is_capture, 0, state.half_move_count + 1)

    return state._replace(half_move_count=half_move_count)


def _update_history(state: GameState):
    board_history = jnp.roll(state.board_history, 1, axis=0)
    board_history = board_history.at[0, :].set(state.board)
    hash_history = jnp.roll(state.hash_history, 1, axis=0)
    hash_history = hash_history.at[0].set(_zobrist_hash(state))
    return state._replace(board_history=board_history, hash_history=hash_history)


def _zobrist_hash(state: GameState):
    hash_ = lax.select(state.color == -1, ZOBRIST_SIDE, jnp.zeros_like(ZOBRIST_SIDE))
    to_reduce = ZOBRIST_BOARD[jnp.arange(BOARD_SIZE), state.board + 2]
    hash_ ^= lax.reduce(to_reduce, 0, lax.bitwise_xor, (0,))
    return hash_
