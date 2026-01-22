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

    def __init__(
        self,
        size=30,
        render_mode=None,
        max_steps=None,
        # ✅ NEW: random obstacle training controls
        random_obstacles=0,          # how many random grey obstacles per episode
        random_obstacle_seed=None,   # optional seed
        ensure_path=True,            # resample until start->goal path exists
    ):
        super().__init__()
        self.size = int(size)
        self.render_mode = render_mode

        # small window for env-only render (play.py draws its own UI)
        self.window_size = 600

        # Observation: (3, size, size)
        # channel0=agent, channel1=goal, channel2=obstacles (black + grey + user)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, self.size, self.size), dtype=np.float32
        )

        # 5 actions:
        # 0=up, 1=left, 2=right, 3=up-right, 4=up-left
        self.action_space = spaces.Discrete(5)

        self.start_pos = (0, 0)
        self.goal_pos = (self.size - 1, self.size - 1)

        self.steps = 0
        self.max_steps = max_steps if max_steps is not None else 600

        # fixed black obstacles
        self.static_obstacles = set(self._static_obstacles_like_figure())
        self.static_obstacles.discard(self.start_pos)
        self.static_obstacles.discard(self.goal_pos)

        # ✅ NEW: random grey obstacles per episode (training)
        self.random_obstacles_n = int(random_obstacles)
        self.ensure_path = bool(ensure_path)
        self.np_rng = np.random.default_rng(random_obstacle_seed)

        # ✅ NEW: per-episode random obstacles (grey)
        self.episode_random_obstacles = set()

        # ✅ user placed obstacles (grey) during play
        self.user_obstacles = set()

        self.agent_pos = self.start_pos

        # visited cells for dots
        self.visited_cells = []

        # pygame
        self.screen = None
        self.clock = None

    # ------------------- helpers -------------------
    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _all_obstacles(self):
        # channel2 should include all walls
        return self.static_obstacles | self.episode_random_obstacles | self.user_obstacles

    def _get_obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        ar, ac = self.agent_pos
        gr, gc = self.goal_pos
        obs[0, ar, ac] = 1.0
        obs[1, gr, gc] = 1.0
        for (r, c) in self._all_obstacles():
            obs[2, r, c] = 1.0
        return obs

    def _add_rect(self, out, r0, r1, c0, c1):
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if 0 <= r < self.size and 0 <= c < self.size:
                    out.add((r, c))

    def _static_obstacles_like_figure(self):
        obs = set()

        # top blocks
        self._add_rect(obs, 0, 2, 7, 8)
        self._add_rect(obs, 0, 2, 12, 13)
        self._add_rect(obs, 1, 3, 16, 17)

        # left wall
        self._add_rect(obs, 6, 12, 3, 3)
        self._add_rect(obs, 14, 20, 3, 3)

        # middle blocks
        self._add_rect(obs, 8, 11, 9, 11)
        self._add_rect(obs, 13, 15, 11, 13)
        self._add_rect(obs, 10, 13, 15, 16)

        # right blocks
        self._add_rect(obs, 6, 9, 21, 23)
        self._add_rect(obs, 12, 14, 24, 26)
        self._add_rect(obs, 18, 22, 22, 24)

        # lower blocks
        self._add_rect(obs, 18, 20, 10, 12)
        self._add_rect(obs, 22, 24, 14, 16)

        # bottom-left blocks
        self._add_rect(obs, 24, 27, 1, 2)
        self._add_rect(obs, 26, 28, 5, 6)

        # bottom scattered
        self._add_rect(obs, 25, 26, 18, 19)
        self._add_rect(obs, 27, 27, 23, 25)

        # horizontal barrier with gaps
        for c in range(0, self.size):
            if c not in (5, 6, 14, 15, 27):
                obs.add((16, c))

        # keep start open
        obs.discard((0, 1))
        obs.discard((1, 0))
        obs.discard((1, 1))

        # keep goal open
        obs.discard((self.size - 1, self.size - 2))
        obs.discard((self.size - 2, self.size - 1))

        # remove requested cells
        for cell in [(2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5)]:
            obs.discard(cell)

        return obs

    def _path_exists_bfs(self, start, goal, obstacles_set):
        from collections import deque
        q = deque([start])
        seen = {start}
        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                return True
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1), (-1,1), (-1,-1)]:  # allow diag-up moves too
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    p = (nr, nc)
                    if p in seen or p in obstacles_set:
                        continue
                    seen.add(p)
                    q.append(p)
        return False

    def _sample_random_obstacles(self):
        """Create per-episode random grey obstacles without blocking the map."""
        if self.random_obstacles_n <= 0:
            self.episode_random_obstacles = set()
            return

        base_blocked = set(self.static_obstacles) | set(self.user_obstacles)

        # candidate free cells
        free = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in base_blocked
            and (r, c) != self.start_pos
            and (r, c) != self.goal_pos
        ]

        # try resampling a few times until path exists
        for _ in range(50):
            self.episode_random_obstacles = set()
            if len(free) >= self.random_obstacles_n:
                picks = self.np_rng.choice(len(free), size=self.random_obstacles_n, replace=False)
                for idx in picks:
                    self.episode_random_obstacles.add(free[int(idx)])

            if not self.ensure_path:
                return

            all_blocked = self.static_obstacles | self.user_obstacles | self.episode_random_obstacles
            if self._path_exists_bfs(self.start_pos, self.goal_pos, all_blocked):
                return

        # fallback: if too hard, reduce random obstacles
        self.episode_random_obstacles = set()

    # ------------------- external API for play.py -------------------
    def add_user_obstacle(self, cell):
        """Add a grey obstacle at runtime (for edit mode)."""
        if cell in (self.start_pos, self.goal_pos):
            return False
        if cell in self.static_obstacles:
            return False
        self.user_obstacles.add(cell)
        return True

    def clear_user_obstacles(self):
        self.user_obstacles.clear()

    # ------------------- gym API -------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.agent_pos = self.start_pos

        # visited path
        self.visited_cells = [self.start_pos]

        # ✅ randomize grey obstacles for this episode (training)
        self._sample_random_obstacles()

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.steps += 1

        r, c = self.agent_pos
        prev_dist = self._manhattan(self.agent_pos, self.goal_pos)

        # 5 actions:
        # 0=up, 1=left, 2=right, 3=up-right, 4=up-left
        nr, nc = r, c
        if action == 0:      # up
            nr = max(r - 1, 0)
        elif action == 1:    # left
            nc = max(c - 1, 0)
        elif action == 2:    # right
            nc = min(c + 1, self.size - 1)
        elif action == 3:    # up-right
            nr = max(r - 1, 0)
            nc = min(c + 1, self.size - 1)
        elif action == 4:    # up-left
            nr = max(r - 1, 0)
            nc = max(c - 1, 0)

        cand = (nr, nc)
        hit_wall = cand in self._all_obstacles()
        if hit_wall:
            cand = (r, c)

        self.agent_pos = cand
        if self.agent_pos != self.visited_cells[-1]:
            self.visited_cells.append(self.agent_pos)

        new_dist = self._manhattan(self.agent_pos, self.goal_pos)

        terminated = (self.agent_pos == self.goal_pos)
        truncated = (self.steps >= self.max_steps)

        # ✅ reward shaping (works even with random obstacles)
        if terminated:
            reward = 200.0
        else:
            reward = -0.05  # small step cost
            if hit_wall:
                reward -= 2.5
            # reward progress
            if new_dist < prev_dist:
                reward += 0.50
            elif new_dist > prev_dist:
                reward -= 0.30

        obs = self._get_obs()
        info = {"hit_wall": bool(hit_wall)}
        return obs, reward, terminated, truncated, info

    # ------------------- optional env-only render -------------------
    def render(self):
        if pygame is None:
            raise ImportError("pygame not installed. Install: pip install pygame-ce")
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("GridWorld 30x30 (Training Env)")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((255, 255, 255))
        cell = self.window_size // self.size

        # static black
        for (rr, cc) in self.static_obstacles:
            pygame.draw.rect(self.screen, (0, 0, 0), (cc * cell, rr * cell, cell, cell))

        # random grey
        for (rr, cc) in self.episode_random_obstacles:
            pygame.draw.rect(self.screen, (150, 150, 150), (cc * cell, rr * cell, cell, cell))

        # user grey
        for (rr, cc) in self.user_obstacles:
            pygame.draw.rect(self.screen, (120, 120, 120), (cc * cell, rr * cell, cell, cell))

        # start
        sr, sc = self.start_pos
        pygame.draw.rect(self.screen, (255, 165, 0), (sc * cell, sr * cell, cell, cell))

        # goal
        gr, gc = self.goal_pos
        pygame.draw.rect(self.screen, (0, 200, 0), (gc * cell, gr * cell, cell, cell))

        # agent
        ar, ac = self.agent_pos
        pygame.draw.rect(self.screen, (0, 0, 255), (ac * cell, ar * cell, cell, cell))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if pygame is not None and self.screen is not None:
            pygame.display.quit()
            pygame.quit()
        self.screen = None
        self.clock = None
