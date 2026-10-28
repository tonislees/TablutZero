import pygame
from .constants import *
from ..tablut_jax import BOARD_EDGE

class TablutUIDrawer:
    def _draw_highlights(self, S, sqs, color, BO):
        arm = max(6, CELL // 7)
        surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        surf.fill(color)
        bc = (color[0], color[1], color[2], 150)
        for (px, py), dx, dy in [((1, 1), 1, 1), ((CELL-2, 1), -1, 1), ((1, CELL-2), 1, -1), ((CELL-2, CELL-2), -1, -1)]:
            pygame.draw.line(surf, bc, (px, py), (px + dx * arm, py), 2)
            pygame.draw.line(surf, bc, (px, py), (px, py + dy * arm), 2)
        for sq in sqs:
            r, c = divmod(sq, BOARD_EDGE)
            S.blit(surf, (BO + c * CELL + self.LM, BO + (BOARD_EDGE - 1 - r) * CELL))

    def _draw_selection(self, S, sq, BO):
        arm = max(8, CELL // 6)
        surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        surf.fill((255, 210, 60, 45))
        pygame.draw.rect(surf, (255, 220, 70, 160), surf.get_rect(), 2)
        sc = (255, 235, 90, 245)
        for (px, py), dx, dy in [((1, 1), 1, 1), ((CELL-2, 1), -1, 1), ((1, CELL-2), 1, -1), ((CELL-2, CELL-2), -1, -1)]:
            pygame.draw.line(surf, sc, (px, py), (px + dx * arm, py), 3)
            pygame.draw.line(surf, sc, (px, py), (px, py + dy * arm), 3)
        r, c = divmod(sq, BOARD_EDGE)
        S.blit(surf, (BO + c * CELL + self.LM, BO + (BOARD_EDGE - 1 - r) * CELL))

    def _draw_valid_moves(self, S, moves, BO):
        half = max(10, CELL // 6)
        surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        ccx, ccy = CELL // 2, CELL // 2
        pts = [(ccx, ccy - half), (ccx + half, ccy), (ccx, ccy + half), (ccx - half, ccy)]
        pygame.draw.polygon(surf, (100, 215, 125, 210), pts, 2)
        pygame.draw.circle(surf, (100, 215, 125, 230), (ccx, ccy), 3)
        for dest in moves:
            dr, dc = divmod(dest, BOARD_EDGE)
            S.blit(surf, (BO + dc * CELL + self.LM, BO + (BOARD_EDGE - 1 - dr) * CELL))

    def _draw_highlight_sq(self, S, sq, color, BO):
        arm = max(10, CELL // 5)
        surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        for (px, py), dx, dy in [((1, 1), 1, 1), ((CELL-2, 1), -1, 1), ((1, CELL-2), 1, -1), ((CELL-2, CELL-2), -1, -1)]:
            pygame.draw.line(surf, color, (px, py), (px + dx * arm, py), 4)
            pygame.draw.line(surf, color, (px, py), (px, py + dy * arm), 4)
        r, c = divmod(sq, BOARD_EDGE)
        S.blit(surf, (BO + c * CELL + self.LM, BO + (BOARD_EDGE - 1 - r) * CELL))

    def _draw_menu(self):
        S = self.render_surf; cx, cy = self.base_w // 2, self.base_h // 2
        
        # Flanking Celtic knots (Left & Right)
        ks = int(160 * UI_SCALE)
        kimg = pygame.transform.smoothscale(self.assets.img_knot_raw, (ks, ks))
        arr = pygame.surfarray.pixels3d(kimg)
        for c, v in enumerate(GOLD_DIM):
            arr[:, :, c] = (arr[:, :, c].astype(int) * v // 255).clip(0, 255)
        del arr
        kimg.set_alpha(110)
        
        S.blit(kimg, (cx - int(300 * UI_SCALE) - ks // 2, cy - ks // 2))
        S.blit(kimg, (cx + int(300 * UI_SCALE) - ks // 2, cy - ks // 2))

        # Title
        title = self.fonts['title'].render("TABLUT", True, GOLD)
        S.blit(title, (cx - title.get_width() // 2, cy - int(200 * UI_SCALE)))
        pygame.draw.line(S, GOLD_DIM, (cx - 120, cy - int(162 * UI_SCALE)), (cx + 120, cy - int(162 * UI_SCALE)), 1)
        
        # Time Dial Card
        pw, ph = int(300 * UI_SCALE), int(76 * UI_SCALE)
        px, py = cx - pw // 2, cy - int(155 * UI_SCALE)
        
        card_surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(card_surf, (22, 19, 18, 200), (0, 0, pw, ph), border_radius=12)
        pygame.draw.rect(card_surf, GOLD_DIM, (0, 0, pw, ph), 1, border_radius=12)
        S.blit(card_surf, (px, py))
        
        header_lbl = self.fonts['small'].render("INITIAL TIME LIMIT", True, GOLD_DIM)
        S.blit(header_lbl, (cx - header_lbl.get_width() // 2, py + int(8 * UI_SCALE)))
        
        time_val_text = f"{int(self.timer_initial // 60)}:00"
        time_lbl = self.fonts['timer'].render(time_val_text, True, CREAM)
        unit_lbl = self.fonts['small'].render("min", True, GOLD_DIM)
        
        total_w = time_lbl.get_width() + 4 + unit_lbl.get_width()
        start_x = cx - total_w // 2
        S.blit(time_lbl, (start_x, py + int(32 * UI_SCALE)))
        S.blit(unit_lbl, (start_x + time_lbl.get_width() + 4, py + int(40 * UI_SCALE)))
        
        btn_r = int(17 * UI_SCALE)
        minus_center = (px + int(32 * UI_SCALE), py + int(45 * UI_SCALE))
        plus_center = (px + pw - int(32 * UI_SCALE), py + int(45 * UI_SCALE))
        
        self.menu_timer_btns = {
            "minus": pygame.Rect(minus_center[0] - btn_r, minus_center[1] - btn_r, btn_r * 2, btn_r * 2),
            "plus": pygame.Rect(plus_center[0] - btn_r, plus_center[1] - btn_r, btn_r * 2, btn_r * 2)
        }
        
        for key, rect in self.menu_timer_btns.items():
            center = minus_center if key == "minus" else plus_center
            is_hover = rect.collidepoint(self._to_base_coords(pygame.mouse.get_pos()))
            color = MENU_BTN_HOV if is_hover else MENU_BTN
            
            pygame.draw.circle(S, color, center, btn_r)
            pygame.draw.circle(S, GOLD, center, btn_r, 1)
            
            line_w = max(2, int(2 * UI_SCALE))
            arm = int(6 * UI_SCALE)
            if key == "minus":
                pygame.draw.line(S, CREAM, (center[0] - arm, center[1]), (center[0] + arm, center[1]), line_w)
            else:
                pygame.draw.line(S, CREAM, (center[0] - arm, center[1]), (center[0] + arm, center[1]), line_w)
                pygame.draw.line(S, CREAM, (center[0], center[1] - arm), (center[0], center[1] + arm), line_w)

        # Menu buttons
        self.menu_btn_rects = {}
        btn_data = [
            ("Play vs AI (Attacker)", "ai_atk"),
            ("Play vs AI (Defender)", "ai_def"),
            ("Player vs Player", "pvp"),
            ("Tutorial", "tutorial"),
            ("Quit Game", "quit")
        ]
        btn_y = cy - int(65 * UI_SCALE)
        for label, key in btn_data:
            rect = pygame.Rect(cx - int(130 * UI_SCALE), btn_y, int(260 * UI_SCALE), int(36 * UI_SCALE))
            self.menu_btn_rects[key] = rect
            is_hover = rect.collidepoint(self._to_base_coords(pygame.mouse.get_pos()))
            
            if is_hover:
                color = MENU_BTN_HOV
                border_color = GOLD
                text_color = GOLD if key != "quit" else (255, 100, 100)
            else:
                color = MENU_BTN
                border_color = GOLD_DIM
                text_color = MENU_TEXT if key != "quit" else (210, 80, 80)
                
            pygame.draw.rect(S, color, rect, border_radius=8)
            pygame.draw.rect(S, border_color, rect, 1, border_radius=8)
            
            lbl = self.fonts['btn'].render(label, True, text_color)
            S.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.centery - lbl.get_height() // 2))
            btn_y += int(48 * UI_SCALE)

    def _draw_sidebar(self):
        S = self.render_surf; sx = self.renderer.total_board + SIDEBAR_GAP + self.LM
        pygame.draw.line(S, GOLD_DIM, (sx, 0), (sx, self.base_h), 2)
        title = self.fonts['title'].render("TABLUT", True, GOLD)
        S.blit(title, (sx + (SIDEBAR_W - title.get_width()) // 2, 20))
        turn = int(self.engine.game_state.color)
        lbl = self.fonts['status'].render("Attacker's Turn" if turn == -1 else "Defender's Turn", True, STATUS_ATK if turn == -1 else STATUS_DEF)
        S.blit(lbl, (sx + (SIDEBAR_W - lbl.get_width()) // 2, int(80 * UI_SCALE)))
        def fmt(t): m, s = divmod(int(t), 60); return f"{m:02d}:{s:02d}"
        atk_t = self.fonts['timer'].render(fmt(self.attacker_time), True, STATUS_ATK)
        def_t = self.fonts['timer'].render(fmt(self.defender_time), True, STATUS_DEF)
        S.blit(atk_t, (sx + 30, int(115 * UI_SCALE))); S.blit(def_t, (sx + SIDEBAR_W - 30 - def_t.get_width(), int(115 * UI_SCALE)))
        
        # AI Thinking Progress Bar
        if self.ai_thinking:
            if not hasattr(self, 'ai_start_time') or self.ai_start_time is None:
                self.ai_start_time = pygame.time.get_ticks()
                self.ai_current_progress = 0.0
            elapsed = (pygame.time.get_ticks() - self.ai_start_time) / 1000.0
            
            if hasattr(self, 'last_ai_duration'):
                expected_time = self.last_ai_duration
            else:
                sims = getattr(self.engine, 'mcts_sims', 200)
                expected_time = 0.5 + sims * 0.008
            
            target_progress = min(0.98, elapsed / max(0.1, expected_time))
            if not hasattr(self, 'ai_current_progress'):
                self.ai_current_progress = 0.0
            self.ai_current_progress += (target_progress - self.ai_current_progress) * 0.2
            progress = self.ai_current_progress
            
            pb_w = SIDEBAR_W - 60
            pb_h = int(8 * UI_SCALE)
            pb_x = sx + 30
            pb_y = int(172 * UI_SCALE)
            
            pb_lbl = self.fonts['small'].render(f"AI Thinking ({int(progress * 100)}%)", True, GOLD)
            S.blit(pb_lbl, (sx + (SIDEBAR_W - pb_lbl.get_width()) // 2, pb_y - int(18 * UI_SCALE)))
            
            pygame.draw.rect(S, (22, 19, 18), (pb_x, pb_y, pb_w, pb_h), border_radius=4)
            pygame.draw.rect(S, GOLD_DIM, (pb_x, pb_y, pb_w, pb_h), 1, border_radius=4)
            if progress > 0.01:
                pygame.draw.rect(S, GOLD, (pb_x + 1, pb_y + 1, int((pb_w - 2) * progress), pb_h - 2), border_radius=3)
        
        # Decorative Celtic knot in the center of the sidebar
        ks = int(100 * UI_SCALE)
        kimg = pygame.transform.smoothscale(self.assets.img_knot_raw, (ks, ks))
        arr = pygame.surfarray.pixels3d(kimg)
        for c, v in enumerate(BOARD_LINE):
            arr[:, :, c] = (arr[:, :, c].astype(int) * v // 255).clip(0, 255)
        del arr
        kimg.set_alpha(80)
        S.blit(kimg, (sx + SIDEBAR_W // 2 - ks // 2, self.base_h // 2 - ks // 2))

        # Bottom buttons
        self.btn_rects = {}
        btn_y = self.base_h - int(155 * UI_SCALE)
        for label, key in [("Reset Game", "reset"), ("Back to Menu", "menu")]:
            rect = pygame.Rect(sx + int(20 * UI_SCALE), btn_y, SIDEBAR_W - int(40 * UI_SCALE), int(38 * UI_SCALE)); self.btn_rects[key] = rect
            color = MENU_BTN_HOV if rect.collidepoint(self._to_base_coords(pygame.mouse.get_pos())) else MENU_BTN
            pygame.draw.rect(S, color, rect, border_radius=6); pygame.draw.rect(S, GOLD_DIM, rect, 1, border_radius=6)
            lbl = self.fonts['btn'].render(label, True, MENU_TEXT); S.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.centery - lbl.get_height() // 2))
            btn_y += int(50 * UI_SCALE)
            
        nav_y = self.base_h - int(45 * UI_SCALE)
        nav_w = (SIDEBAR_W - int(50 * UI_SCALE)) // 4
        for i, (label, key) in enumerate([("|<", "start"), ("<", "prev"), (">", "next"), (">|", "end")]):
            rect = pygame.Rect(sx + int(25 * UI_SCALE) + i * nav_w, nav_y, nav_w - 4, int(30 * UI_SCALE)); self.btn_rects[key] = rect
            disabled = False
            if key in ("start", "prev") and self.view_index == 0:
                disabled = True
            elif key in ("next", "end") and self.view_index >= len(self.state_history) - 1:
                disabled = True
                
            if disabled:
                color = (38, 34, 30)
                border_color = (68, 62, 52)
                text_color = (100, 90, 80)
            else:
                color = MENU_BTN_HOV if rect.collidepoint(self._to_base_coords(pygame.mouse.get_pos())) else MENU_BTN
                border_color = GOLD_DIM
                text_color = MENU_TEXT
                
            pygame.draw.rect(S, color, rect, border_radius=4); pygame.draw.rect(S, border_color, rect, 1, border_radius=4)
            lbl = self.fonts['small'].render(label, True, text_color); S.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.centery - lbl.get_height() // 2))

    def _draw_tutorial_panel(self):
        S = self.render_surf; sx = self.renderer.total_board + SIDEBAR_GAP + self.LM
        BO = self.renderer.border_w + BOARD_GAP
        
        self.tutorial_btn_rects = {}
        quit_w = int(80 * UI_SCALE)
        quit_h = int(28 * UI_SCALE)
        quit_rect = pygame.Rect(sx + SIDEBAR_W // 2 - quit_w // 2, self.base_h - int(105 * UI_SCALE), quit_w, quit_h)
        self.tutorial_btn_rects["quit"] = quit_rect
        
        pygame.draw.line(S, GOLD_DIM, (sx, 0), (sx, self.base_h), 2)
        lbl_mode = self.fonts['status'].render("TUTORIAL", True, GOLD)
        S.blit(lbl_mode, (sx + SIDEBAR_W // 2 - lbl_mode.get_width() // 2, 40))
        
        step = self.tutorial_steps[self.tutorial_idx]
        
        ks = int(100 * UI_SCALE)
        kimg = pygame.transform.smoothscale(self.assets.img_knot_raw, (ks, ks))
        arr = pygame.surfarray.pixels3d(kimg)
        for c, v in enumerate(BOARD_LINE):
            arr[:, :, c] = (arr[:, :, c].astype(int) * v // 255).clip(0, 255)
        del arr
        kimg.set_alpha(80)
        S.blit(kimg, (sx + SIDEBAR_W // 2 - ks // 2, self.base_h // 2 - ks // 2))
        
        if self.tutorial_show_win:
            card_w = int(180 * UI_SCALE)
            card_h = int(50 * UI_SCALE)
            card_x = sx + SIDEBAR_W // 2 - card_w // 2
            card_y = self.base_h // 2 - card_h // 2
            
            card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
            pygame.draw.rect(card_surf, (22, 19, 18, 240), (0, 0, card_w, card_h), border_radius=12)
            pygame.draw.rect(card_surf, GOLD, (0, 0, card_w, card_h), 2, border_radius=12)
            
            win_msg = step.get("show_win", "")
            lbl_win = self.fonts['status'].render(win_msg, True, STATUS_ATK if "ATTACKER" in win_msg else STATUS_DEF)
            card_surf.blit(lbl_win, (card_w // 2 - lbl_win.get_width() // 2, card_h // 2 - lbl_win.get_height() // 2))
            S.blit(card_surf, (card_x, card_y))
            
        step_lbl = self.fonts['small'].render(f"Step {self.tutorial_idx + 1} of {len(self.tutorial_steps)}", True, CREAM)
        S.blit(step_lbl, (sx + SIDEBAR_W // 2 - step_lbl.get_width() // 2, self.base_h - 50))
        
        popup_w = int(240 * UI_SCALE)
        msg_text = step["msg"]
        paragraphs = msg_text.split('\n')
        lines = []
        for paragraph in paragraphs:
            if paragraph.strip() == "":
                lines.append("")
                continue
            words = paragraph.split(' ')
            curr_line = ""
            for w in words:
                test = curr_line + w + " "
                if self.fonts['small'].size(test)[0] < popup_w - 40:
                    curr_line = test
                else:
                    lines.append(curr_line)
                    curr_line = w + " "
            lines.append(curr_line)
            
        title_lbl = self.fonts['status'].render(step["title"], True, GOLD)
        title_h = title_lbl.get_height()
        title_y = int(12 * UI_SCALE)
        sep_y = title_y + title_h + int(6 * UI_SCALE)
        ly_start = sep_y + int(8 * UI_SCALE)
        
        line_h = int(14 * UI_SCALE)
        btn_h = int(28 * UI_SCALE)
        popup_h = ly_start + len(lines) * line_h + int(15 * UI_SCALE) + btn_h + int(15 * UI_SCALE)
        
        p_row, p_col = step["popup_pos"]
        cell_x = p_col * CELL
        cell_ui_r = (BOARD_EDGE - 1 - p_row) * CELL
        ccx, ccy = cell_x + CELL // 2, cell_ui_r + CELL // 2
        px = ccx - popup_w // 2
        py = ccy - popup_h // 2
        
        px = max(15, min(BOARD_PX - popup_w - 15, px))
        py = max(15, min(BOARD_PX - popup_h - 15, py))
        
        popup_surf = pygame.Surface((popup_w, popup_h), pygame.SRCALPHA)
        pygame.draw.rect(popup_surf, (22, 19, 18, 240), (0, 0, popup_w, popup_h), border_radius=12)
        pygame.draw.rect(popup_surf, GOLD, (0, 0, popup_w, popup_h), 2, border_radius=12)
        
        popup_surf.blit(title_lbl, (20, title_y))
        pygame.draw.line(popup_surf, (*GOLD_DIM, 120), (20, sep_y), (popup_w - 20, sep_y), 1)
        
        ly = ly_start
        for line in lines:
            if line.strip() != "":
                txt = self.fonts['small'].render(line.strip(), True, CREAM)
                popup_surf.blit(txt, (20, ly))
            ly += line_h
            
        S.blit(popup_surf, (BO + px + self.LM, BO + py))
        
        btn_w = int(75 * UI_SCALE)
        local_y = popup_h - btn_h - int(15 * UI_SCALE)
        base_x = BO + px + self.LM
        base_y = BO + py
        
        if self.tutorial_idx < len(self.tutorial_steps) - 1:
            r = pygame.Rect(base_x + popup_w - btn_w - 20, base_y + local_y, btn_w, btn_h)
            self.tutorial_btn_rects["next"] = r
        else:
            r = pygame.Rect(base_x + popup_w - btn_w - 20, base_y + local_y, btn_w, btn_h)
            self.tutorial_btn_rects["finish"] = r
            
        if self.tutorial_idx > 0:
            r = pygame.Rect(base_x + 20, base_y + local_y, btn_w, btn_h)
            self.tutorial_btn_rects["prev"] = r
            
        for key, rect in self.tutorial_btn_rects.items():
            color = MENU_BTN_HOV if rect.collidepoint(self._to_base_coords(pygame.mouse.get_pos())) else MENU_BTN
            pygame.draw.rect(S, color, rect, border_radius=6)
            pygame.draw.rect(S, GOLD_DIM, rect, 1, border_radius=6)
            lbl = self.fonts['btn'].render("Finish" if key=="finish" else key.capitalize(), True, MENU_TEXT)
            S.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.centery - lbl.get_height() // 2))

    def _draw_game_over(self):
        sw, sh = self.screen.get_size()
        popup_w = int(260 * UI_SCALE)
        popup_h = int(140 * UI_SCALE)
        
        px = sw // 2 - popup_w // 2
        py = sh // 2 - popup_h // 2
        
        popup_surf = pygame.Surface((popup_w, popup_h), pygame.SRCALPHA)
        pygame.draw.rect(popup_surf, (22, 19, 18, 240), (0, 0, popup_w, popup_h), border_radius=12)
        pygame.draw.rect(popup_surf, GOLD, (0, 0, popup_w, popup_h), 2, border_radius=12)
        
        title_lbl = self.fonts['status'].render("GAME OVER", True, GOLD)
        title_y = int(12 * UI_SCALE)
        popup_surf.blit(title_lbl, (popup_w // 2 - title_lbl.get_width() // 2, title_y))
        
        sep_y = title_y + title_lbl.get_height() + int(6 * UI_SCALE)
        pygame.draw.line(popup_surf, (*GOLD_DIM, 120), (20, sep_y), (popup_w - 20, sep_y), 1)
        
        msg_lbl = self.fonts['over'].render(self.game_over_msg, True, CREAM)
        msg_y = sep_y + int(15 * UI_SCALE)
        if msg_lbl.get_width() > popup_w - 40:
            msg_lbl = pygame.transform.smoothscale(msg_lbl, (popup_w - 40, int(msg_lbl.get_height() * (popup_w - 40) / msg_lbl.get_width())))
        popup_surf.blit(msg_lbl, (popup_w // 2 - msg_lbl.get_width() // 2, msg_y))
        
        self.screen.blit(popup_surf, (px, py))
