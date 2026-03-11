import pygame
import sys
import jax
import jax.numpy as jnp
from src.hnefatafl.hnefatafl_jax import Action, BOARD_EDGE, BOARD_SIZE, THRONE

SCREEN_SIZE = 720
GRID_SIZE = BOARD_EDGE
CELL_SIZE = SCREEN_SIZE // GRID_SIZE
FPS = 60

COLOR_BG = (240, 217, 181)
COLOR_GRID = (139, 69, 19)
COLOR_HIGHLIGHT = (100, 255, 100)
COLOR_SELECTED = (255, 255, 0)
COLOR_THRONE = (160, 82, 45)
COLOR_CORNER = (160, 82, 45)

COLOR_ATTACKER = (20, 20, 20)
COLOR_DEFENDER = (240, 240, 230)
COLOR_KING = (212, 175, 55)

CORNERS = [0, GRID_SIZE - 1, BOARD_SIZE - GRID_SIZE, BOARD_SIZE - 1]


class HnefataflUI:
    def __init__(self, logic_engine):
        pygame.init()
        pygame.display.set_caption(f"Hnefatafl JAX ({GRID_SIZE}x{GRID_SIZE})")
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 32, bold=True)

        self.engine = logic_engine

        self.selected_sq = None
        self.valid_moves_for_selected = []
        self.move_history = []
        self.running = True
        self.game_over = False

    def get_piece_at(self, idx):
        return int(self.engine.game_state.board[idx])

    def get_legal_destinations(self, from_sq):
        legal_mask = self.engine.state.legal_action_mask
        destinations = []
        valid_indices = jnp.where(legal_mask)[0]
        for label_idx in valid_indices.tolist():
            action = Action.from_label(label_idx)
            if int(action.from_sq) == from_sq:
                destinations.append(int(action.to_sq))
        return destinations

    def draw_board(self):
        self.screen.fill(COLOR_BG)
        current_turn = int(self.engine.game_state.color)

        for sq in CORNERS + [THRONE]:
            r, c = divmod(sq, GRID_SIZE)
            ui_r = GRID_SIZE - 1 - r
            rect = (c * CELL_SIZE, ui_r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, COLOR_CORNER, rect)

        for i in range(GRID_SIZE + 1):
            pygame.draw.line(self.screen, COLOR_GRID, (0, i * CELL_SIZE), (SCREEN_SIZE, i * CELL_SIZE), 2)
            pygame.draw.line(self.screen, COLOR_GRID, (i * CELL_SIZE, 0), (i * CELL_SIZE, SCREEN_SIZE), 2)

        if self.selected_sq is not None:
            r, c = divmod(self.selected_sq, GRID_SIZE)
            ui_r = GRID_SIZE - 1 - r
            rect = (c * CELL_SIZE, ui_r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, COLOR_SELECTED, rect, 4)

            for dest in self.valid_moves_for_selected:
                dr, dc = divmod(dest, GRID_SIZE)
                dui_r = GRID_SIZE - 1 - dr
                center = (dc * CELL_SIZE + CELL_SIZE // 2, dui_r * CELL_SIZE + CELL_SIZE // 2)
                pygame.draw.circle(self.screen, COLOR_HIGHLIGHT, center, CELL_SIZE // 6)

        for idx in range(BOARD_SIZE):
            piece = self.get_piece_at(idx)
            if piece == 0:
                continue

            r, c = divmod(idx, GRID_SIZE)
            ui_r = GRID_SIZE - 1 - r
            center = (c * CELL_SIZE + CELL_SIZE // 2, ui_r * CELL_SIZE + CELL_SIZE // 2)
            radius = CELL_SIZE // 2 - 6

            is_attacker = False
            if current_turn == -1:
                if piece > 0: is_attacker = True
            else:
                if piece < 0: is_attacker = True

            if abs(piece) == 2:
                pygame.draw.circle(self.screen, COLOR_KING, center, radius)
                pygame.draw.line(self.screen, (0, 0, 0), (center[0], center[1] - 8), (center[0], center[1] + 8), 3)
                pygame.draw.line(self.screen, (0, 0, 0), (center[0] - 8, center[1]), (center[0] + 8, center[1]), 3)
            elif is_attacker:
                pygame.draw.circle(self.screen, COLOR_ATTACKER, center, radius)
            else:
                pygame.draw.circle(self.screen, COLOR_DEFENDER, center, radius)
                pygame.draw.circle(self.screen, (0, 0, 0), center, radius, 1)

    def handle_click(self, pos):
        if self.game_over:
            return

        col = pos[0] // CELL_SIZE
        ui_row = pos[1] // CELL_SIZE
        row = GRID_SIZE - 1 - ui_row
        idx = row * GRID_SIZE + col

        if not (0 <= idx < BOARD_SIZE):
            return

        clicked_piece = self.get_piece_at(idx)

        if self.selected_sq is not None and idx in self.valid_moves_for_selected:
            self.execute_move(self.selected_sq, idx)
            self.selected_sq = None
            self.valid_moves_for_selected = []
            return

        if clicked_piece > 0:
            self.selected_sq = idx
            self.valid_moves_for_selected = self.get_legal_destinations(idx)
        else:
            self.selected_sq = None
            self.valid_moves_for_selected = []

    def execute_move(self, from_sq, to_sq):
        uci = self.engine._sq_to_uci(from_sq) + self.engine._sq_to_uci(to_sq)
        self.move_history.append(uci)
        print(f"Move: {uci}")

        legal_mask = self.engine.state.legal_action_mask
        valid_indices = jnp.where(legal_mask)[0]

        action_label = -1
        for label_idx in valid_indices.tolist():
            a = Action.from_label(label_idx)
            if int(a.from_sq) == from_sq and int(a.to_sq) == to_sq:
                action_label = label_idx
                break

        if action_label != -1:
            step_key, self.engine.key_env = jax.random.split(self.engine.key_env)
            self.engine.state = self.engine.step_fn(self.engine.state, action_label, step_key)
            self.engine.game_state = self.engine.state._x

            if self.engine.env.game.is_terminal(self.engine.game_state):
                self.game_over = True
                rewards = self.engine.env.game.rewards(self.engine.game_state)

                if rewards[0] > 0:
                    winner = "ATTACKERS WON!"
                elif rewards[1] > 0:
                    winner = "DEFENDERS WON!"
                else:
                    winner = "DRAW!"
                print(f"\n=== {winner} ===")

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.engine.reset()
                        self.game_over = False
                        self.move_history = []
                        self.selected_sq = None

            self.draw_board()

            if self.game_over:
                text = self.font.render("GAME OVER", True, (255, 50, 50))
                rect = text.get_rect(center=(SCREEN_SIZE // 2, SCREEN_SIZE // 2))
                pygame.draw.rect(self.screen, (255, 255, 255), rect.inflate(20, 10))
                pygame.draw.rect(self.screen, (0, 0, 0), rect.inflate(20, 10), 2)
                self.screen.blit(text, rect)

            pygame.display.flip()

        pygame.quit()
        sys.exit()