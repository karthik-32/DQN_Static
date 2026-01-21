import os
os.environ["SDL_VIDEO_CENTERED"] = "1"
os.environ["SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, size=30, render_mode=None, max_steps=None):
        super().__init__()
        self.size = int(size)
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, self.size, self.size), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.start_pos = (0, 0)
        self.goal_pos = (self.size - 1, self.size - 1)

        self.steps = 0
        self.max_steps = max_steps if max_steps is not None else 600

        self.static_obstacles = set(self._static_obstacles_like_figure())
        self.static_obstacles.discard(self.start_pos)
        self.static_obstacles.discard(self.goal_pos)

        self.agent_pos = self.start_pos

        # ✅ NEW: store visited cells for blue dots (only used when rendering)
        self.visited_cells = set()

        # pygame
        self.window_size = 750
        self.screen = None
        self.clock = None

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        ar, ac = self.agent_pos
        gr, gc = self.goal_pos
        obs[0, ar, ac] = 1.0
        obs[1, gr, gc] = 1.0
        for (r, c) in self.static_obstacles:
            obs[2, r, c] = 1.0
        return obs

    def _add_rect(self, out, r0, r1, c0, c1):
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if 0 <= r < self.size and 0 <= c < self.size:
                    out.add((r, c))

    def _static_obstacles_like_figure(self):
        obs = set()

        self._add_rect(obs, 1, 5, 2, 4)

        self._add_rect(obs, 0, 2, 7, 8)
        self._add_rect(obs, 0, 2, 12, 13)
        self._add_rect(obs, 1, 3, 16, 17)

        self._add_rect(obs, 6, 12, 3, 3)
        self._add_rect(obs, 14, 20, 3, 3)

        self._add_rect(obs, 8, 11, 9, 11)
        self._add_rect(obs, 13, 15, 11, 13)
        self._add_rect(obs, 10, 13, 15, 16)

        self._add_rect(obs, 6, 9, 21, 23)
        self._add_rect(obs, 12, 14, 24, 26)
        self._add_rect(obs, 18, 22, 22, 24)

        self._add_rect(obs, 18, 20, 10, 12)
        self._add_rect(obs, 22, 24, 14, 16)

        self._add_rect(obs, 24, 27, 1, 2)
        self._add_rect(obs, 26, 28, 5, 6)

        self._add_rect(obs, 25, 26, 18, 19)
        self._add_rect(obs, 27, 27, 23, 25)

        for c in range(0, self.size):
            if c not in (5, 6, 14, 15, 27):
                obs.add((16, c))

        obs.discard((0, 1))
        obs.discard((1, 0))
        obs.discard((1, 1))

        obs.discard((self.size - 1, self.size - 2))
        obs.discard((self.size - 2, self.size - 1))

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.agent_pos = self.start_pos

        # ✅ NEW: reset visited path each episode
        self.visited_cells = {self.start_pos}

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def step(self, action):
        self.steps += 1

        r, c = self.agent_pos
        prev_dist = self._manhattan(self.agent_pos, self.goal_pos)

        nr, nc = r, c
        if action == 0:
            nr = max(r - 1, 0)
        elif action == 1:
            nr = min(r + 1, self.size - 1)
        elif action == 2:
            nc = max(c - 1, 0)
        elif action == 3:
            nc = min(c + 1, self.size - 1)

        cand = (nr, nc)
        hit_static = cand in self.static_obstacles
        if hit_static:
            cand = (r, c)

        self.agent_pos = cand

        # ✅ NEW: add current position to visited
        self.visited_cells.add(self.agent_pos)

        new_dist = self._manhattan(self.agent_pos, self.goal_pos)

        terminated = (self.agent_pos == self.goal_pos)
        truncated = (self.steps >= self.max_steps)

        if terminated:
            reward = 150.0
        else:
            reward = -0.1
            if hit_static:
                reward -= 2.0
            if new_dist < prev_dist:
                reward += 0.35
            elif new_dist > prev_dist:
                reward -= 0.25

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        info = {"hit_static": bool(hit_static)}
        return obs, reward, terminated, truncated, info

    def render(self):
        if pygame is None:
            raise ImportError("pygame not installed. Install: pip install pygame-ce")

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("GridWorld 30x30 (Static Obstacles Only)")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((255, 255, 255))
        cell = self.window_size // self.size

        # obstacles (black)
        for (rr, cc) in self.static_obstacles:
            pygame.draw.rect(
                self.screen,
                (0, 0, 0),
                pygame.Rect(cc * cell, rr * cell, cell, cell)
            )

        # ✅ NEW: draw visited path as light-blue dots
        for (pr, pc) in self.visited_cells:
            if (pr, pc) == self.start_pos or (pr, pc) == self.goal_pos:
                continue
            cx = pc * cell + cell // 2
            cy = pr * cell + cell // 2
            radius = max(2, cell // 6)
            pygame.draw.circle(self.screen, (100, 200, 255), (cx, cy), radius)

        # start (orange)
        sr, sc = self.start_pos
        pygame.draw.rect(self.screen, (255, 165, 0), pygame.Rect(sc * cell, sr * cell, cell, cell))

        # goal (green)
        gr, gc = self.goal_pos
        pygame.draw.rect(self.screen, (0, 200, 0), pygame.Rect(gc * cell, gr * cell, cell, cell))

        # agent (blue)
        ar, ac = self.agent_pos
        pygame.draw.rect(self.screen, (0, 0, 255), pygame.Rect(ac * cell, ar * cell, cell, cell))

        # grid lines
        for i in range(self.size + 1):
            pygame.draw.line(self.screen, (70, 70, 70), (0, i * cell), (self.window_size, i * cell), 1)
            pygame.draw.line(self.screen, (70, 70, 70), (i * cell, 0), (i * cell, self.window_size), 1)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if pygame is not None and self.screen is not None:
            pygame.display.quit()
            pygame.quit()
        self.screen = None
        self.clock = None
