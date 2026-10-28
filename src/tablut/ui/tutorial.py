import jax.numpy as jnp
from ..tablut_jax import BOARD_SIZE, GameState, INIT_BOARD, THRONE, BOARD_EDGE, initialize_legal_actions

def mock_state(pieces_dict, color=-1):
    board = jnp.zeros(BOARD_SIZE, dtype=jnp.int32)
    for sq, p in pieces_dict.items():
        board = board.at[sq].set(p)
    state = GameState(color=jnp.int32(color), board=board)
    return state._replace(legal_action_mask=initialize_legal_actions(state))

def get_tutorial_steps():
    # Initial state for first slide
    init_gs = GameState(color=jnp.int32(-1), board=-INIT_BOARD)
    init_gs = init_gs._replace(legal_action_mask=initialize_legal_actions(init_gs))
    
    # Repetition slide pieces: e5 (40) is King. 
    # We place a few others far away to avoid confusion.
    # d1(3), d9(75), a4(27), i4(35)
    rep_state = mock_state({40: 2, 3: -1, 75: -1, 27: 1, 35: 1, 42: 1, 38: -1}, color=1)
    
    steps = [
        {
            "title": "Welcome to Tablut",
            "msg": "The Attackers (blue) try to capture the King, while the Defenders (red) help him escape to the corners.",
            "state": (init_gs, None),
            "show_valid_sq": 3, # d1 (an attacker)
            "popup_pos": (7, 1.5)
        },
        {
            "title": "Piece Movement",
            "msg": "Pieces move horizontally or vertically like a rook in chess, but ordinary pieces cannot land on the throne (center) or the corners.\n\nThey may only pass through an empty throne.",
            "state": (mock_state({3: 1, 7: 1, 73: 1, 42: -1, 38: -1}), None),
            "show_valid_sq": 3,
            "popup_pos": (3, 6.5)
        },
        {
            "title": "King Movement",
            "msg": "Only the King is allowed to land on the central throne or escape to the four corners.",
            "state": (mock_state({39: 2, 3: -1, 77: -1, 27: 1, 53: 1}, color=1), None),
            "show_valid_sq": 39,
            "animate_move": (39, 40), # d5 to throne (40)
            "popup_pos": (2, 6)
        },
        {
            "title": "Regular Capture",
            "msg": "Capture a piece by surrounding it on two opposite sides with your pieces.",
            "state": (mock_state({30: 1, 31: -1, 41: 1, 3: 1, 77: 1, 13: -1, 39: -1}), (41, 32)),
            "animate_move": (41, 32),
            "captures": [31],
            "popup_pos": (1, 4)
        },
        {
            "title": "Hostile Square Capture",
            "msg": "Hostile squares (corners and the throne) also act as your pieces to help pin and capture enemies.",
            "state": (mock_state({11: 1, 62: 1, 5: 1, 36: -1, 44: -1, 1: -1}), (11, 2)),
            "animate_move": (11, 2),
            "captures": [1],
            "popup_pos": (2, 5.5)
        },
        {
            "title": "Safe Sandwiches",
            "msg": "Moving your own piece between two enemy pieces is completely safe. A piece is only captured when an opponent moves to sandwich it.",
            "state": (mock_state({18: -1, 36: -1, 28: 1}, color=1), (28, 27)),
            "animate_move": (28, 27),
            "popup_pos": (2, 5.5)
        },
        {
            "title": "King Capture",
            "msg": "The King is captured exactly like a regular piece: by surrounding him on two opposite sides.",
            "state": (mock_state({30: -2, 29: 1, 22: 1, 12: -1, 48: -1, 3: 1, 75: 1}), (22, 31)),
            "animate_move": (22, 31),
            "captures": [30],
            "show_win": "ATTACKERS WIN!",
            "popup_pos": (6, 1.5)
        },
        {
            "title": "King Hostile Capture",
            "msg": "The King can also be captured by sandwiching him against the central throne or any of the four corner squares.",
            "state": (mock_state({39: -2, 37: 1, 3: 1, 75: 1}), (37, 38)),
            "animate_move": (37, 38),
            "captures": [39],
            "show_win": "ATTACKERS WIN!",
            "popup_pos": (6, 1.5)
        },
        {
            "title": "King Escape",
            "msg": "The Defenders win immediately if the King reaches any of the four corner squares.",
            "state": (mock_state({1: 2, 3: -1, 10: -1, 19: 1, 12: 1}, color=1), (1, 0)),
            "animate_move": (1, 0),
            "show_win": "DEFENDERS WIN!",
            "popup_pos": (2, 3.5)
        },
        {
            "title": "Repetition Loss",
            "msg": "If the same board position occurs three times, the player who makes the repeating move loses the game.",
            "state": (rep_state, None),
            "animate_move_seq": [(40, 49), (3, 12), (49, 40), (12, 3), (40, 49), (3, 12), (49, 40)], # Alternating King and Attacker moves
            "show_win": "ATTACKERS WIN!", # If defender (king) repeats, he loses
            "popup_pos": (2, 6.5)
        },
        {
            "title": "No Moves Left",
            "msg": "If a player has no legal moves remaining on their turn, they immediately lose the game.",
            "state": (mock_state({18: 2, 9: -1, 19: -1, 27: -1}, color=1), None),
            "show_win": "ATTACKERS WIN!",
            "popup_pos": (2, 5.5)
        }
    ]
    return steps
