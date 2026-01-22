import time
import numpy as np
import pygame
import gymnasium as gym
import gymnasium_env
import torch
import torch.nn as nn


# ---------------- DQN ----------------
class DQN(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def flatten_obs(obs: np.ndarray) -> np.ndarray:
    return obs.reshape(-1).astype(np.float32)


# ---------------- UI Widgets ----------------
class Button:
    def __init__(self, rect, text):
        self.rect = pygame.Rect(rect)
        self.text = text

    def draw(self, screen, font, fill=(150, 150, 150), outline=(90, 90, 90), text_col=(240, 240, 240)):
        pygame.draw.rect(screen, fill, self.rect, border_radius=6)
        pygame.draw.rect(screen, outline, self.rect, 2, border_radius=6)
        label = font.render(self.text, True, text_col)
        screen.blit(label, label.get_rect(center=self.rect.center))

    def clicked(self, pos):
        return self.rect.collidepoint(pos)


class Slider:
    def __init__(self, x, y, w, min_val, max_val, init_val, label=""):
        self.x, self.y, self.w = x, y, w
        self.h = 8
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.value = float(init_val)
        self.label = label
        self.dragging = False
        self.knob_r = 10
        self.track_rect = pygame.Rect(self.x, self.y, self.w, self.h)

    def _val_to_pos(self):
        t = (self.value - self.min_val) / (self.max_val - self.min_val + 1e-9)
        return int(self.x + t * self.w)

    def _pos_to_val(self, px):
        px = max(self.x, min(self.x + self.w, px))
        t = (px - self.x) / (self.w + 1e-9)
        return self.min_val + t * (self.max_val - self.min_val)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            knob_x = self._val_to_pos()
            knob_y = self.y + self.h // 2
            if (mx - knob_x) ** 2 + (my - knob_y) ** 2 <= (self.knob_r + 6) ** 2 or self.track_rect.collidepoint(mx, my):
                self.dragging = True
                self.value = self._pos_to_val(mx)

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mx, _ = event.pos
            self.value = self._pos_to_val(mx)

    def draw(self, screen, font, label_font):
        if self.label:
            label = label_font.render(self.label, True, (230, 230, 230))
            screen.blit(label, (self.x, self.y - 22))

        pygame.draw.rect(screen, (210, 210, 210), self.track_rect, border_radius=4)

        knob_x = self._val_to_pos()
        knob_y = self.y + self.h // 2
        pygame.draw.circle(screen, (245, 245, 245), (knob_x, knob_y), self.knob_r)
        pygame.draw.circle(screen, (80, 80, 80), (knob_x, knob_y), self.knob_r, 2)

        txt = font.render(f"{self.value:.2f}s", True, (230, 230, 230))
        screen.blit(txt, (self.x + self.w - 55, self.y + 12))


# ---------------- Metrics Screen ----------------
def draw_metrics_screen(screen, W, H, font_big, font, metrics_rows, back_btn):
    screen.fill((90, 90, 90))
    title = font_big.render("Simulation Results", True, (240, 240, 240))
    screen.blit(title, title.get_rect(center=(W // 2, 70)))

    box_w, box_h = 700, 320
    box_x = (W - box_w) // 2
    box_y = 120
    pygame.draw.rect(screen, (120, 120, 120), (box_x, box_y, box_w, box_h), border_radius=10)

    header_y = box_y + 20
    metric_hdr = font_big.render("Metric", True, (240, 240, 240))
    value_hdr = font_big.render("Value", True, (240, 240, 240))
    screen.blit(metric_hdr, (box_x + 30, header_y))
    screen.blit(value_hdr, (box_x + 420, header_y))

    y = header_y + 55
    row_h = 46
    for i, (k, v) in enumerate(metrics_rows):
        if i % 2 == 0:
            pygame.draw.rect(screen, (140, 140, 140), (box_x + 20, y - 10, box_w - 40, row_h), border_radius=6)
        ktxt = font.render(k, True, (240, 240, 240))
        vtxt = font.render(v, True, (240, 240, 240))
        screen.blit(ktxt, (box_x + 30, y))
        screen.blit(vtxt, (box_x + 420, y))
        y += row_h

    back_btn.draw(screen, font_big, fill=(150, 150, 150), outline=(70, 70, 70), text_col=(240, 240, 240))


# ---------------- Main ----------------
def main():
    size = 30
    model_file = "fast_dqn_static_30.pt"
    max_steps = 600

    # IMPORTANT: do NOT use env.render_mode="human" now (we draw our own window)
    env = gym.make("gymnasium_env/GridWorld-v0", size=size, render_mode=None, max_steps=max_steps)
    base_env = env.unwrapped  # access start/goal/obstacles/agent_pos

    obs0, _ = env.reset()
    obs_dim = obs0.size
    n_actions = env.action_space.n

    model = DQN(obs_dim, n_actions, hidden=128)
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()

    pygame.init()
    font = pygame.font.SysFont("Arial", 16)
    font_big = pygame.font.SysFont("Arial", 19, bold=True)

    # Compact UI sizing
    grid_px = 600
    panel_w = 260
    W = grid_px + panel_w
    H = grid_px

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("DQN GridWorld - Controls Panel")

    panel_x = grid_px

    # Buttons
    btn_w, btn_h = 220, 46
    reset_btn = Button((panel_x + 20, 25, btn_w, btn_h), "Reset Grid")
    start_btn = Button((panel_x + 20, 85, btn_w, btn_h), "Start/Stop")
    show_metrics_btn = Button((panel_x + 20, 360, btn_w, btn_h), "Show Metrics")
    back_btn = Button(((W // 2) - 110, H - 110, 220, 54), "Back")

    # Slider (animation delay)
    anim_slider = Slider(panel_x + 30, 210, 200, 0.00, 0.50, 0.06, label="Anim Delay")

    # Simulation state
    running = False
    in_metrics = False

    obs, _ = env.reset()
    steps = 0
    turns = 0
    last_action = None
    last_success = False

    # Timing metrics
    sim_start_time = None
    sim_end_time = None
    total_infer_time = 0.0

    # Path dots (visited cells)
    visited = {base_env.start_pos}

    clock = pygame.time.Clock()

    while True:
        # -------- handle events --------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                return

            if in_metrics:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if back_btn.clicked(event.pos):
                        in_metrics = False
                continue

            if event.type == pygame.MOUSEBUTTONDOWN:
                if reset_btn.clicked(event.pos):
                    running = False
                    obs, _ = env.reset()
                    steps = 0
                    turns = 0
                    last_action = None
                    last_success = False
                    sim_start_time = None
                    sim_end_time = None
                    total_infer_time = 0.0
                    visited = {base_env.start_pos}

                if start_btn.clicked(event.pos):
                    running = not running
                    if running and sim_start_time is None:
                        sim_start_time = time.perf_counter()

                if show_metrics_btn.clicked(event.pos):
                    in_metrics = True

            anim_slider.handle_event(event)

        # -------- do one DQN step if running --------
        if (not in_metrics) and running:
            s = flatten_obs(obs)

            t0 = time.perf_counter()
            with torch.no_grad():
                q = model(torch.from_numpy(s).unsqueeze(0))[0].numpy()
            t1 = time.perf_counter()
            total_infer_time += (t1 - t0)

            action = int(np.argmax(q))

            if last_action is not None and action != last_action:
                turns += 1
            last_action = action

            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            visited.add(base_env.agent_pos)  # ✅ blue dot after each move

            if terminated or truncated or steps >= max_steps:
                running = False
                sim_end_time = time.perf_counter()
                last_success = bool(terminated)

            time.sleep(float(anim_slider.value))

        # -------- metrics screen --------
        if in_metrics:
            sim_time = 0.0
            if sim_start_time is not None:
                sim_time = (sim_end_time - sim_start_time) if sim_end_time is not None else (time.perf_counter() - sim_start_time)

            global_plan = total_infer_time
            local_plan = (total_infer_time / steps) if steps > 0 else 0.0

            metrics_rows = [
                ("Total Path Length", str(steps)),
                ("Total Number of Turns", str(turns)),
                ("Global Plan Time (s)", f"{global_plan:.6f}"),
                ("Local Plan Time (s)", f"{local_plan:.6f}"),
                ("Success", "Yes" if last_success else "No"),
                ("Simulation Time (s)", f"{sim_time:.6f}"),
            ]
            draw_metrics_screen(screen, W, H, font_big, font, metrics_rows, back_btn)
            pygame.display.flip()
            clock.tick(60)
            continue

        # -------- draw main screen --------
        screen.fill((245, 245, 245))

        # panel background
        pygame.draw.rect(screen, (110, 110, 110), (panel_x, 0, panel_w, H))

        cell = grid_px // size

        # obstacles (black)
        for (r, c) in getattr(base_env, "static_obstacles", set()):
            pygame.draw.rect(screen, (0, 0, 0), (c * cell, r * cell, cell, cell))

        # start (orange)
        sr, sc = base_env.start_pos
        pygame.draw.rect(screen, (255, 165, 0), (sc * cell, sr * cell, cell, cell))

        # goal (green)
        gr, gc = base_env.goal_pos
        pygame.draw.rect(screen, (0, 200, 0), (gc * cell, gr * cell, cell, cell))

        # visited blue dots
        for (r, c) in visited:
            cx = c * cell + cell // 2
            cy = r * cell + cell // 2
            pygame.draw.circle(screen, (80, 180, 255), (cx, cy), max(2, cell // 6))

        # agent (blue)
        ar, ac = base_env.agent_pos
        pygame.draw.rect(screen, (0, 0, 255), (ac * cell, ar * cell, cell, cell))

        # grid lines
        for i in range(size + 1):
            pygame.draw.line(screen, (70, 70, 70), (0, i * cell), (grid_px, i * cell), 1)
            pygame.draw.line(screen, (70, 70, 70), (i * cell, 0), (i * cell, grid_px), 1)

        # right panel UI
        reset_btn.draw(screen, font_big)
        start_btn.draw(screen, font_big)

        # grid size label
        screen.blit(font_big.render(f"Grid Size: {size}", True, (230, 230, 230)), (panel_x + 30, 150))

        # anim slider
        anim_slider.draw(screen, font, font)

        show_metrics_btn.draw(screen, font_big)

        # live numbers
        y0 = 430
        screen.blit(font_big.render("Current Path Len:", True, (230, 230, 230)), (panel_x + 20, y0))
        screen.blit(font.render(str(steps), True, (230, 230, 230)), (panel_x + 180, y0 + 3))

        screen.blit(font_big.render("Total Path Length:", True, (230, 230, 230)), (panel_x + 20, y0 + 35))
        screen.blit(font.render(str(steps), True, (230, 230, 230)), (panel_x + 180, y0 + 38))

        screen.blit(font_big.render("Total Turns:", True, (230, 230, 230)), (panel_x + 20, y0 + 70))
        screen.blit(font.render(str(turns), True, (230, 230, 230)), (panel_x + 180, y0 + 73))

        # status
        status = "RUNNING" if running else "STOPPED"
        st_col = (0, 200, 0) if running else (220, 80, 80)
        screen.blit(font_big.render(status, True, st_col), (panel_x + 20, y0 + 120))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
