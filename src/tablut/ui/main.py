import sys
import threading
import jax
import jax.numpy as jnp
from .constants import *
from .assets import Assets
from .renderer import Renderer
from .tutorial import get_tutorial_steps
from .animation import MoveAnimation
from ..tablut_jax import Action, Game
from .drawer import TablutUIDrawer

class TablutUI(TablutUIDrawer):
    def __init__(self, logic_engine):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.engine = logic_engine

        self.screen = pygame.display.set_mode((1, 1))
        self.assets = Assets()
        
        self.fonts = {
            'title': pygame.font.SysFont("Georgia", int(28 * UI_SCALE), bold=True),
            'btn': pygame.font.SysFont("Georgia", int(18 * UI_SCALE)),
            'small': pygame.font.SysFont("Georgia", int(14 * UI_SCALE)),
            'coord': pygame.font.SysFont("Consolas", int(12 * UI_SCALE)),
            'status': pygame.font.SysFont("Georgia", int(16 * UI_SCALE), bold=True),
            'over': pygame.font.SysFont("Georgia", int(36 * UI_SCALE), bold=True),
            'timer': pygame.font.SysFont("Consolas", int(22 * UI_SCALE), bold=True)
        }

        self.renderer = Renderer(self.assets, self.fonts)
        
        self.LM = int(34 * UI_SCALE)
        self.base_w = self.renderer.base_w + self.LM
        self.base_h = self.renderer.base_h
        
        pygame.display.set_icon(pygame.transform.smoothscale(self.assets.img_knot_raw, (32, 32)))
        pygame.display.set_caption("Hnefatafl · Viking Board Game")
        self.screen = pygame.display.set_mode((self.base_w, self.base_h), pygame.RESIZABLE)
        self.render_surf = pygame.Surface((self.base_w, self.base_h))
        self.ox, self.oy = 0, 0

        self.timer_initial = TIMER_INITIAL_DEFAULT
        self.attacker_time = self.timer_initial
        self.defender_time = self.timer_initial
        self.selected_sq = None
        self.valid_moves_for_selected = []
        self.move_history = []
        self.state_history = []
        self.view_index = -1
        self.running = True
        self.game_over = False
        self.game_over_msg = ""
        self.mode = None
        self.ai_thinking = False
        self.ai_move_result = None
        self.game_session_id = 0
        
        self.tutorial_idx = 0
        self.tutorial_steps = get_tutorial_steps()
        
        self.animations = [] # List of MoveAnimation
        self.tutorial_anim = None
        self.tutorial_delay = 0.0
        self.tutorial_seq_idx = 0
        self.tutorial_show_win = False

        self._reset_game()

    def _reset_game(self):
        self.game_session_id = getattr(self, 'game_session_id', 0) + 1
        self.ai_thinking = False
        self.ai_move_result = None
        
        if hasattr(self.engine, 'reset'): self.engine.reset()
        self.attacker_time = self.timer_initial
        self.defender_time = self.timer_initial
        self.game_over = False; self.game_over_msg = ""
        self.move_history = []
        self.state_history = [(self.engine.state, None)]
        self.view_index = 0
        self.selected_sq = None; self.valid_moves_for_selected = []
        self.animations = []
        self.tutorial_delay = 0.0
        self.tutorial_seq_idx = 0
        self.tutorial_show_win = False
        self.ai_moves_made = 0
        self.ai_start_time = None
        self.eval_cache = {}
        self.move_feedback = None
        self.move_feedback_timer = 0.0

    def _to_menu(self):
        self._reset_game()
        self.mode = None

    def _update_offset(self):
        win_w, win_h = self.screen.get_size()
        self.ox = max(0, (win_w - self.base_w) // 2)
        self.oy = max(0, (win_h - self.base_h) // 2)

    def _to_base_coords(self, pos):
        return pos[0] - self.ox, pos[1] - self.oy

    def draw(self, dt=0.0):
        self.screen.fill(BG_DARK)
        self.render_surf.fill(BG_DARK)
        self._update_offset()

        if self.mode is None:
            self._draw_menu()
            self.screen.blit(self.render_surf, (self.ox, self.oy))
            return

        BO = self.renderer.border_w + BOARD_GAP
        S = self.render_surf
        S.blit(self.renderer.board_surf, (BO + self.LM, BO))

        if self.mode == "tutorial":
            curr_step = self.tutorial_steps[self.tutorial_idx]
            current_state, v_last_move = curr_step["state"]
            
            # Keep a persistent gstate for the active tutorial step to support sequences and intermediate changes
            if not hasattr(self, 'tutorial_gstate') or self.tutorial_gstate is None or getattr(self, 'tutorial_gstate_idx', -1) != self.tutorial_idx:
                self.tutorial_gstate = current_state.game_state if hasattr(current_state, 'game_state') else current_state
                self.tutorial_gstate_idx = self.tutorial_idx
            
            gstate = self.tutorial_gstate
            
            # Handle tutorial animation with delay on repeat
            if self.tutorial_delay > 0:
                self.tutorial_delay -= dt
                if self.tutorial_delay <= 0:
                    self.tutorial_anim = None
                    self.tutorial_gstate = None # Reset persistent state on new cycle
            elif curr_step.get("animate_move"):
                f, t = curr_step["animate_move"]
                if self.tutorial_anim is None or self.tutorial_anim.is_finished:
                    if self.tutorial_anim and self.tutorial_anim.is_finished:
                        self.tutorial_delay = 3.0 # Pause 3s before repeating
                        self.tutorial_show_win = False # Hide win during delay
                        if curr_step.get("show_win"): self.tutorial_show_win = True
                    else:
                        self.tutorial_show_win = False # Clear win when starting new anim
                        p = int(gstate.board[f])
                        self.tutorial_anim = MoveAnimation(f, t, p, turn=int(gstate.color), duration=0.6)
                if self.tutorial_anim: self.tutorial_anim.update()
            elif curr_step.get("animate_move_seq"):
                seq = curr_step["animate_move_seq"]
                if self.tutorial_anim is None or self.tutorial_anim.is_finished:
                    if self.tutorial_anim and self.tutorial_anim.is_finished:
                        # Update the persistent state to reflect the completed move
                        f_idx = self.tutorial_anim.from_idx
                        t_idx = self.tutorial_anim.to_idx
                        p_val = int(self.tutorial_gstate.board[f_idx])
                        
                        # Move the piece
                        self.tutorial_gstate = self.tutorial_gstate._replace(
                            board=self.tutorial_gstate.board.at[f_idx].set(0).at[t_idx].set(p_val)
                        )
                        # Flip perspective
                        self.tutorial_gstate = self.tutorial_gstate._replace(
                            board=-self.tutorial_gstate.board,
                            color=-self.tutorial_gstate.color
                        )
                        gstate = self.tutorial_gstate

                        self.tutorial_seq_idx += 1
                        if self.tutorial_seq_idx >= len(seq):
                            self.tutorial_delay = 4.0 # Pause 4s after sequence
                            self.tutorial_seq_idx = 0
                            self.tutorial_show_win = False
                            if curr_step.get("show_win"): self.tutorial_show_win = True
                        else:
                            self.tutorial_anim = None
                    else:
                        if self.tutorial_seq_idx == 0: self.tutorial_show_win = False
                        f, t = seq[self.tutorial_seq_idx]
                        p = int(gstate.board[f])
                        self.tutorial_anim = MoveAnimation(f, t, p, turn=int(gstate.color), duration=0.6)
                if self.tutorial_anim: self.tutorial_anim.update()

            # Convert JAX array to standard Python list once to avoid massive device-to-host sync overhead
            board_list = gstate.board.tolist()

            # For capture slides, we need to hide the captured piece in board_list
            # if the animation has reached its destination.
            if self.mode == "tutorial" and self.tutorial_anim:
                captures = curr_step.get("captures", [])
                for c_idx in captures:
                    if abs(board_list[c_idx]) == 2:
                        continue
                    # Remove the captured piece only when the sliding piece is adjacent (progress >= 0.85)
                    if self.tutorial_anim.progress >= 0.85:
                        board_list[c_idx] = 0
        else:
            current_state, v_last_move = self.state_history[self.view_index]
            gstate = current_state.game_state if hasattr(current_state, 'game_state') else current_state
            board_list = gstate.board.tolist()
            
            # Ensure captured pieces remain on the board until the sliding piece is adjacent (progress >= 0.85)
            active_anim = None
            for a in self.animations:
                if not a.is_finished and a.progress < 0.85:
                    active_anim = a
                    break
            
            if active_anim is not None and self.view_index > 0:
                prev_state, _ = self.state_history[self.view_index - 1]
                prev_gs = prev_state.game_state if hasattr(prev_state, 'game_state') else prev_state
                prev_board = prev_gs.board.tolist()
                # Find any square that was occupied in the previous state, is now empty,
                # and is NOT the sliding piece's source or destination
                for sq in range(BOARD_SIZE):
                    if sq != active_anim.from_idx and sq != active_anim.to_idx:
                        prev_p = prev_board[sq]
                        curr_p = board_list[sq]
                        if prev_p != 0 and curr_p == 0:
                            # Put the captured piece back on the board temporarily for drawing!
                            # We negate prev_p because the board perspective is flipped on turn change!
                            board_list[sq] = -prev_p
            
            # Ensure King stays on the board if captured (not present on current board)
            has_king = any(abs(x) == 2 for x in board_list)
            if not has_king:
                # Find the last position of the King in the state history
                king_sq = None
                king_val = None
                for hist_state, _ in reversed(self.state_history[:self.view_index + 1]):
                    hist_gs = hist_state.game_state if hasattr(hist_state, 'game_state') else hist_state
                    hist_board = hist_gs.board.tolist()
                    try:
                        king_sq = next(i for i, x in enumerate(hist_board) if abs(x) == 2)
                        king_val = hist_board[king_sq]
                        break
                    except StopIteration:
                        continue
                if king_sq is not None and king_val is not None:
                    # Put the King back on the board list for rendering
                    board_list[king_sq] = king_val

        # Draw last move highlights
        if v_last_move:
            self._draw_highlights(S, v_last_move, (255, 195, 50, 30), BO)

        # Draw selection and valid moves
        if self.mode != "tutorial":
            if self.view_index == len(self.state_history) - 1 and self.selected_sq is not None:
                self._draw_selection(S, self.selected_sq, BO)
                self._draw_valid_moves(S, self.valid_moves_for_selected, BO)
        else:
            v_sq = curr_step.get("show_valid_sq")
            if v_sq is not None:
                self._draw_selection(S, v_sq, BO)
                self._draw_valid_moves(S, self._get_legal_dests_for_state(v_sq, current_state), BO)

        if self.mode == "tutorial":
            hsq = curr_step.get("highlight_sq")
            if hsq is not None: self._draw_highlight_sq(S, hsq, GOLD, BO)

        # Update and filter animations
        self.animations = [a for a in self.animations if not a.is_finished]
        for a in self.animations: a.update()

        # Draw pieces
        animating_to_indices = [a.to_idx for a in self.animations]
        
        # In tutorial, if we are in delay/pause period AFTER an animation, 
        # we want to show the piece at the DESTINATION (to_idx), so we DON'T exclude it.
        # But during active animation, we do exclude it.
        if self.tutorial_anim and not self.tutorial_anim.is_finished:
            animating_to_indices.append(self.tutorial_anim.to_idx)
        
        # turn color from gstate
        turn = int(gstate.color)
        for idx in range(BOARD_SIZE):
            if idx in animating_to_indices: continue
            
            # Special case for tutorial: if we finished an animation but are in delay, 
            # hide the piece at from_idx and show at to_idx (if not already there)
            if self.mode == "tutorial" and self.tutorial_anim and self.tutorial_anim.is_finished:
                if idx == self.tutorial_anim.from_idx: continue
            
            piece = board_list[idx]
            if piece == 0: continue
            
            # If this is tutorial, and we are animating the piece FROM this idx
            if self.mode == "tutorial" and self.tutorial_anim and not self.tutorial_anim.is_finished and idx == self.tutorial_anim.from_idx:
                continue

            row, col = divmod(idx, BOARD_EDGE)
            ui_r = BOARD_EDGE - 1 - row
            pos = (BO + col * CELL + CELL // 2 + self.LM, BO + ui_r * CELL + CELL // 2)
            # Use actual board turn to correctly map Attacker vs Defender
            self.renderer.draw_piece(S, pos, piece, turn)

        # Draw animated pieces
        for a in self.animations:
            ax, ay = a.get_pos(BOARD_EDGE, CELL)
            # Use stored turn of the animation if available, else fall back to current turn or piece sign
            a_turn = a.turn if a.turn is not None else (-1 if a.piece < 0 else 1)
            self.renderer.draw_piece(S, (BO + ax + CELL // 2 + self.LM, BO + ay + CELL // 2), a.piece, a_turn)
            
        if self.mode == "tutorial" and self.tutorial_anim:
            a_turn = self.tutorial_anim.turn if self.tutorial_anim.turn is not None else (-1 if self.tutorial_anim.piece < 0 else 1)
            if not self.tutorial_anim.is_finished:
                ax, ay = self.tutorial_anim.get_pos(BOARD_EDGE, CELL)
                self.renderer.draw_piece(S, (BO + ax + CELL // 2 + self.LM, BO + ay + CELL // 2), self.tutorial_anim.piece, a_turn)
            else:
                # Still show it at destination during delay
                to_idx = self.tutorial_anim.to_idx
                row, col = divmod(to_idx, BOARD_EDGE)
                ui_r = BOARD_EDGE - 1 - row
                pos = (BO + col * CELL + CELL // 2 + self.LM, BO + ui_r * CELL + CELL // 2)
                self.renderer.draw_piece(S, pos, self.tutorial_anim.piece, a_turn)

        S.blit(self.renderer.border_surf, (self.LM, 0))

        # Draw Eval Bar on the far left side, outside of the Celtic border frame (drawn on top to prevent border overlapping)
        if self.mode != "tutorial":
            # 1. Fetch current evaluation value from cache to avoid blocking JAX thread during AI search
            if not hasattr(self, 'eval_cache'):
                self.eval_cache = {}
            if self.view_index not in self.eval_cache:
                current_state = self.state_history[self.view_index][0]
                self.eval_cache[self.view_index] = self.engine.get_evaluation(current_state)
            val = self.eval_cache[self.view_index]
            
            # Smoothly transition the eval bar display value
            if not hasattr(self, 'current_eval_smooth'):
                self.current_eval_smooth = val
            self.current_eval_smooth += (val - self.current_eval_smooth) * 0.1
            
            # Dimensions
            eb_w = int(10 * UI_SCALE)
            eb_h = BOARD_PX
            # Place on the far left (with a 18-pixel margin), to the left of the Celtic frame (which starts at self.LM)
            eb_x = int(18 * UI_SCALE)
            eb_y = BO
            
            # Draw outer dark slot and border
            pygame.draw.rect(S, (22, 19, 18), (eb_x, eb_y, eb_w, eb_h), border_radius=4)
            pygame.draw.rect(S, GOLD_DIM, (eb_x, eb_y, eb_w, eb_h), 1, border_radius=4)
            
            # Draw Defender area (entire bar as background) - Red representing Defender
            pygame.draw.rect(S, STATUS_DEF, (eb_x + 1, eb_y + 1, eb_w - 2, eb_h - 2), border_radius=3)
            
            # Draw Attacker area (top portion based on attacker advantage) - Blue representing Attacker
            frac = (self.current_eval_smooth + 1.0) / 2.0
            frac = max(0.0, min(1.0, frac))
            
            atk_h = int((eb_h - 2) * frac)
            if atk_h > 0:
                pygame.draw.rect(S, STATUS_ATK, (eb_x + 1, eb_y + 1, eb_w - 2, atk_h), border_radius=3)
                
            # Draw central gold divider line (0.0 evaluation)
            pygame.draw.line(S, GOLD, (eb_x, eb_y + eb_h // 2), (eb_x + eb_w - 1, eb_y + eb_h // 2), 1)
            
            # Display text evaluation (e.g. "+0.34" or "-0.15") centered above the bar, with a guard to prevent clipping at the left edge
            eval_text = f"{'+' if val >= 0 else ''}{val:.2f}"
            eval_lbl = self.fonts['coord'].render(eval_text, True, GOLD if val >= 0 else CREAM)
            text_x = max(2, eb_x + eb_w // 2 - eval_lbl.get_width() // 2)
            S.blit(eval_lbl, (text_x, eb_y - int(16 * UI_SCALE)))
        
        if self.mode == "tutorial":
            self._draw_tutorial_panel()
        else:
            self._draw_sidebar()
            
            # Draw floating move feedback inside sidepanel (drawn on S, only in game mode, not tutorial)
            if getattr(self, 'move_feedback_timer', 0.0) > 0.0:
                self.move_feedback_timer -= dt
                
                # Dimensions to fit sidebar
                tw_w = SIDEBAR_W - int(40 * UI_SCALE)
                tw_h = int(34 * UI_SCALE)
                
                sx = self.renderer.total_board + SIDEBAR_GAP + self.LM
                tx = sx + int(20 * UI_SCALE)
                ty = int(220 * UI_SCALE)
                
                # Calculate alpha fade based on remaining time
                alpha = 255
                if self.move_feedback_timer < 0.5:
                    alpha = int(255 * (self.move_feedback_timer / 0.5))
                alpha = max(0, min(255, alpha))
                
                # Draw semi-transparent rounded card directly on S
                toast_surf = pygame.Surface((tw_w, tw_h), pygame.SRCALPHA)
                
                # Deep rich dark brown background
                pygame.draw.rect(toast_surf, (22, 19, 18, int(min(240, alpha))), (0, 0, tw_w, tw_h), border_radius=8)
                # Border using feedback color
                border_color = (int(self.move_feedback_color[0]), int(self.move_feedback_color[1]), int(self.move_feedback_color[2]), int(alpha))
                pygame.draw.rect(toast_surf, border_color, (0, 0, tw_w, tw_h), 2, border_radius=8)
                
                # Text label
                lbl = self.fonts['status'].render(self.move_feedback, True, (int(self.move_feedback_color[0]), int(self.move_feedback_color[1]), int(self.move_feedback_color[2])))
                lbl.set_alpha(alpha)
                toast_surf.blit(lbl, (tw_w // 2 - lbl.get_width() // 2, tw_h // 2 - lbl.get_height() // 2))
                
                # Blit toast onto S
                S.blit(toast_surf, (tx, ty))

        self.screen.blit(self.render_surf, (self.ox, self.oy))
        if self.game_over: self._draw_game_over()


    def handle_board_click(self, pos):
        if self.game_over or self.mode is None or self.view_index != len(self.state_history) - 1: return
        bx, by = self._to_base_coords(pos); BO = self.renderer.border_w + BOARD_GAP
        bx -= BO + self.LM; by -= BO
        if bx < 0 or by < 0 or bx >= BOARD_PX or by >= BOARD_PX: return
        col, ui_row = int(bx) // CELL, int(by) // CELL; row = BOARD_EDGE - 1 - ui_row; idx = row * BOARD_EDGE + col
        if not (0 <= idx < BOARD_SIZE): return
        
        state = self.state_history[self.view_index][0]
        gstate = state.game_state if hasattr(state, 'game_state') else state
        
        if self.selected_sq is not None and idx in self.valid_moves_for_selected:
            self._execute_move(self.selected_sq, idx); self.selected_sq = None; self.valid_moves_for_selected = []
            return
        p = int(gstate.board[idx])
        if p > 0:
            self.selected_sq = idx; self.valid_moves_for_selected = self._get_legal_dests(idx)
        else: self.selected_sq = None; self.valid_moves_for_selected = []

    def _get_legal_dests(self, from_sq):
        state, _ = self.state_history[self.view_index]
        return self._get_legal_dests_for_state(from_sq, state)

    def _get_legal_dests_for_state(self, from_sq, state):
        import numpy as np
        # Handle both pgx State and GameState
        gstate = state.game_state if hasattr(state, 'game_state') else state
        
        mask = np.array(gstate.legal_action_mask)
        dests = []
        indices = np.where(mask)[0]
        for l_idx in indices.tolist():
            f_sq = l_idx // 32 # 32 = ACTION_PLANES for 9x9
            if f_sq == from_sq:
                # Use FROM_PLANE for decoding
                from ..tablut_jax import FROM_PLANE
                t_sq = int(FROM_PLANE[f_sq, l_idx % 32])
                dests.append(t_sq)
        return dests

    def _execute_move(self, from_sq, to_sq):
        state = self.state_history[self.view_index][0]
        gstate = state.game_state if hasattr(state, 'game_state') else state
        valid = jnp.where(gstate.legal_action_mask)[0]
        action_label = -1
        for l in valid.tolist():
            a = Action.from_label(l)
            if int(a.from_sq) == from_sq and int(a.to_sq) == to_sq: action_label = l; break
        if action_label != -1:
            # Evaluate user move and store feedback
            txt, color = self.engine.evaluate_user_move(state, action_label)
            self.move_feedback = txt
            self.move_feedback_color = color
            self.move_feedback_timer = 2.5  # Display for 2.5 seconds
            
            # Animation
            p = int(gstate.board[from_sq])
            self.animations.append(MoveAnimation(from_sq, to_sq, p, turn=int(gstate.color)))
            
            step_key, self.engine.key_env = jax.random.split(self.engine.key_env)
            self.engine.state = self.engine.step_fn(self.engine.state, action_label, step_key)
            self.engine.game_state = self.engine.state.game_state
            uci = self.engine._sq_to_uci(from_sq) + self.engine._sq_to_uci(to_sq)
            self.move_history.append(uci)
            self.state_history.append((self.engine.state, (from_sq, to_sq)))
            self.view_index = len(self.state_history) - 1
            self._check_game_over()

    def _check_game_over(self):
        if not self.game_over and Game.is_terminal(self.engine.game_state):
            self.game_over = True; rew = Game.rewards(self.engine.game_state)
            if rew[0] > 0: self.game_over_msg = "ATTACKERS WIN!"
            elif rew[1] > 0: self.game_over_msg = "DEFENDERS WIN!"
            else: self.game_over_msg = "DRAW"

    def run(self):
        try:
            while self.running:
                dt = self.clock.tick(FPS) / 1000.0
                if self.mode and self.mode != "tutorial" and not self.game_over:
                    ai_c = getattr(self.engine, 'ai_color', 0)
                    if self.ai_thinking:
                        if ai_c == -1: self.attacker_time -= dt
                        else: self.defender_time -= dt
                    else:
                        turn = int(self.engine.game_state.color)
                        if turn == -1: self.attacker_time -= dt
                        else: self.defender_time -= dt
                    if self.attacker_time <= 0: self.attacker_time = 0; self.game_over = True; self.game_over_msg = "DEFENDERS WIN (Timeout)!"
                    elif self.defender_time <= 0: self.defender_time = 0; self.game_over = True; self.game_over_msg = "ATTACKERS WIN (Timeout)!"

                if self.mode == "ai" and not self.game_over:
                    turn = int(self.engine.game_state.color); ai_c = getattr(self.engine, 'ai_color', 0)
                    if turn == ai_c and not self.ai_thinking:
                        self.ai_thinking = True
                        self.ai_start_time = pygame.time.get_ticks()
                        session_id = self.game_session_id
                        def ai_fn(sid):
                            try:
                                uci = self.engine.make_ai_move()
                                if sid == self.game_session_id:
                                    self.ai_move_result = uci
                                    self.ai_thinking = False
                            except Exception as e:
                                print(f"AI ERROR: {e}")
                                if sid == self.game_session_id:
                                    self.ai_thinking = False
                        threading.Thread(target=ai_fn, args=(session_id,), daemon=True).start()
                    if self.ai_move_result:
                        self.ai_moves_made = getattr(self, 'ai_moves_made', 0) + 1
                        if getattr(self, 'ai_start_time', None) is not None:
                            self.last_ai_duration = (pygame.time.get_ticks() - self.ai_start_time) / 1000.0
                        self.ai_start_time = None
                        uci = self.ai_move_result; self.ai_move_result = None
                        def uci_to_idx(s): return (int(s[1:]) - 1) * BOARD_EDGE + (ord(s[0]) - ord('a'))
                        f, t = uci_to_idx(uci[:2]), uci_to_idx(uci[2:])
                        # Animation
                        state = self.state_history[self.view_index][0]
                        gstate = state.game_state if hasattr(state, 'game_state') else state
                        p = int(gstate.board[f])
                        self.animations.append(MoveAnimation(f, t, p, turn=int(gstate.color)))
                        self.move_history.append(uci); self.state_history.append((self.engine.state, (f, t)))
                        if self.view_index == len(self.state_history) - 2: self.view_index = len(self.state_history) - 1
                        self._check_game_over()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT: self.running = False
                    elif event.type == pygame.VIDEORESIZE: self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        bx, by = self._to_base_coords(event.pos)
                        if self.mode is None:
                            for key, rect in self.menu_timer_btns.items():
                                if rect.collidepoint(bx, by):
                                    if key == "plus": self.timer_initial += 60
                                    elif key == "minus" and self.timer_initial > 60: self.timer_initial -= 60
                            for key, rect in self.menu_btn_rects.items():
                                if rect.collidepoint(bx, by):
                                    if key == "ai_atk": self.engine.ai_color = 1; self.mode = "ai"; self._reset_game()
                                    elif key == "ai_def": self.engine.ai_color = -1; self.mode = "ai"; self._reset_game()
                                    elif key == "pvp": self.mode = "pvp"; self._reset_game()
                                    elif key == "tutorial": self.mode = "tutorial"; self.tutorial_idx = 0; self.tutorial_anim = None; self.tutorial_delay = 0; self.tutorial_seq_idx = 0; self.tutorial_show_win = False
                                    elif key == "quit": self.running = False
                        elif self.mode == "tutorial":
                            for key, rect in self.tutorial_btn_rects.items():
                                if rect.collidepoint(bx, by):
                                    if key == "next": 
                                        self.tutorial_idx += 1; self.tutorial_anim = None; self.tutorial_delay = 0; self.tutorial_seq_idx = 0; self.tutorial_show_win = False
                                    elif key == "prev": 
                                        self.tutorial_idx -= 1; self.tutorial_anim = None; self.tutorial_delay = 0; self.tutorial_seq_idx = 0; self.tutorial_show_win = False
                                    elif key in ("finish", "quit"): 
                                        self._to_menu()
                                    break
                        else:
                            found_btn = False
                            for key, rect in getattr(self, 'btn_rects', {}).items():
                                if rect.collidepoint(bx, by):
                                    found_btn = True
                                    if key == "reset": self._reset_game()
                                    elif key == "menu": self._to_menu()
                                    elif key == "start": self.view_index = 0
                                    elif key == "prev": self.view_index = max(0, self.view_index - 1)
                                    elif key == "next": self.view_index = min(len(self.state_history) - 1, self.view_index + 1)
                                    elif key == "end": self.view_index = len(self.state_history) - 1
                                    break
                            if not found_btn:
                                if self.mode == "pvp" or (self.mode == "ai" and int(self.state_history[self.view_index][0].game_state.color) != getattr(self.engine, 'ai_color', 0)):
                                    self.handle_board_click(event.pos)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r: self._reset_game()
                        elif event.key == pygame.K_ESCAPE: self._to_menu()
                        elif event.key == pygame.K_LEFT:
                            if self.mode == "tutorial": 
                                self.tutorial_idx = max(0, self.tutorial_idx - 1); self.tutorial_anim = None; self.tutorial_delay = 0; self.tutorial_seq_idx = 0; self.tutorial_show_win = False
                            else: self.view_index = max(0, self.view_index - 1)
                        elif event.key == pygame.K_RIGHT:
                            if self.mode == "tutorial": 
                                self.tutorial_idx = min(len(self.tutorial_steps) - 1, self.tutorial_idx + 1); self.tutorial_anim = None; self.tutorial_delay = 0; self.tutorial_seq_idx = 0; self.tutorial_show_win = False
                            else: self.view_index = min(len(self.state_history) - 1, self.view_index + 1)
                self.draw(dt); pygame.display.flip()
        except Exception as e:
            print(f"CRITICAL UI ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            pygame.quit()
            sys.exit()
