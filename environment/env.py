"""Collaborative grid-based multi-agent environment."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame
from gymnasium import Env, spaces


Coord = Tuple[int, int]


class CollaborativeCarryEnv(Env):
    """Cooperative multi-agent object-carrying task (variable team size)."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        grid_size: int = 8,
        n_agents: int = 4,
        n_objects: Optional[int] = None,
        n_goals: Optional[int] = None,
        n_obstacles: int = 4,
        max_agents: int = 10,
        max_objects: Optional[int] = None,
        max_goals: Optional[int] = None,
        max_obstacles: int = 6,
        max_grid_size: int = 10,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Derive defaults
        derived_objects = n_objects if n_objects is not None else max(1, int(np.ceil(n_agents / 2)))
        derived_goals = n_goals if n_goals is not None else derived_objects
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_objects = derived_objects
        self.n_goals = derived_goals
        self.n_obstacles = n_obstacles
        self.max_agents = max(max_agents, n_agents)
        self.max_objects = max(max_objects or derived_objects, derived_objects)
        self.max_goals = max(max_goals or derived_goals, derived_goals)
        self.max_obstacles = max(max_obstacles, n_obstacles)
        self.max_grid_size = max(max_grid_size, grid_size)
        self.max_steps = max_steps
        self.render_mode = render_mode

        self._action_to_delta: Dict[int, Coord] = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
            4: (0, 0),   # stay (quality-of-life choice for manual play)
        }
        self.action_space = spaces.MultiDiscrete([len(self._action_to_delta)] * self.n_agents)
        
        # ═══════════════════════════════════════════════════════════════════
        # SIMPLIFIED OBSERVATION SPACE
        # Per-agent features (length 14):
        #   pos_norm(2)                    - Where am I?
        #   vec_to_nearest_obj(2)          - Direction to nearest object needing help
        #   dist_to_nearest_obj(1)         - How far?
        #   vec_to_nearest_goal(2)         - Direction to nearest goal
        #   dist_to_nearest_goal(1)        - How far?
        #   nearest_obstacle_dist(1)       - How close is danger?
        #   on_object(1)                   - Am I on an object?
        #   object_has_partner(1)          - Does my object have 2 carriers?
        #   am_i_needed(1)                 - Is there an object needing me?
        #   agent_density_around_me(1)     - How many agents nearby? (avoid clustering)
        # ═══════════════════════════════════════════════════════════════════
        obs_dim_per_agent = 14
        # Shared features:
        #   step_norm(1), n_agents_norm(1), grid_norm(1) -> 3
        #   obstacles: max_obstacles * 2
        #   objects: max_objects * (pos(2)+goal(2)+n_carriers(1)+delivered(1)) = max_objects*6
        obs_dim_shared = 3 + self.max_obstacles * 2 + self.max_objects * 6
        self.obs_dim_per_agent = obs_dim_per_agent
        self.obs_dim_shared = obs_dim_shared
        
        obs_space_dict = {
            f"agent_{i+1}": spaces.Box(low=-1.0, high=1.0, shape=(obs_dim_per_agent,), dtype=np.float32)
            for i in range(self.max_agents)
        }
        obs_space_dict["shared"] = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim_shared,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_space_dict)

        # Reward configuration - MINIMAL to prevent reward hacking
        # Only 3 signals: Delivery (goal), Step cost (time pressure), Shaping (guidance)
        self.reward_config = {
            # DELIVERY IS THE ONLY TRUE GOAL
            "alpha_del": 100.0,            # Large delivery reward
            "alpha_eff": 0.0,              # DISABLED - can be exploited
            "T_cap": 50,
            
            # REMOVED: Obstacle penalty (just block movement instead)
            "alpha_obstacle_hit": 0.0,     # DISABLED
            
            # Time pressure (the only negative signal)
            "alpha_step": -0.5,            # Stronger per-step cost to encourage speed
            
            # REMOVED: Pickup/drop (can be exploited by oscillating)
            "alpha_pickup": 0.0,           # DISABLED
            "alpha_drop": 0.0,             # DISABLED
            "alpha_carry_progress": 2.0,   # Moving object toward goal (can't be exploited)
            
            # REMOVED: Third-wheel (complicates learning)
            "alpha_third_wheel": 0.0,      # DISABLED
            
            # Potential-based shaping (mathematically unexploitable)
            "alpha_approach": 0.3,         # Guides agents toward objects
        }
        
        # Track collisions for reward computation
        self._collision_count = 0
        self._obstacle_hits = 0  # NEW: track obstacle collisions

        self.agent_positions: List[Coord] = []
        self.object_positions: List[Coord] = []
        self.goal_positions: List[Coord] = []
        self.goal_used: List[bool] = []
        self.delivered: List[bool] = []
        self.obstacles: List[Coord] = []
        self.current_step = 0
        self._prev_agent_positions: List[Coord] = []
        self._prev_object_positions: List[Coord] = []
        self._prev_goal_used: List[bool] = []
        self._prev_delivered: List[bool] = []

        self.cell_size = 60
        self.window_size = self.grid_size * self.cell_size
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

    def set_n_obstacles(self, n_obstacles: int):
        """Change number of obstacles (for curriculum learning). Takes effect on next reset."""
        if n_obstacles > self.max_obstacles:
            raise ValueError(f"n_obstacles ({n_obstacles}) cannot exceed max_obstacles ({self.max_obstacles})")
        self.n_obstacles = n_obstacles
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, object]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self._obstacle_hits = 0

        taken = set()
        
        # ═══════════════════════════════════════════════════════════════════
        # WAREHOUSE-STYLE SPAWN: Agents start from bottom-left corner
        # This is realistic - robots start from a depot/charging station
        # ═══════════════════════════════════════════════════════════════════
        
        # Define spawn zone (bottom-left corner, 2x2 or 2x3 area)
        spawn_zone = []
        for r in range(self.grid_size - 2, self.grid_size):  # Last 2 rows
            for c in range(min(3, self.grid_size)):  # First 3 columns
                spawn_zone.append((r, c))
        
        # Place agents in spawn zone
        self.agent_positions = []
        available_spawns = list(spawn_zone)
        self.np_random.shuffle(available_spawns)
        for i in range(self.n_agents):
            if i < len(available_spawns):
                pos = available_spawns[i]
            else:
                # Fallback if not enough spawn positions
                pos = self._random_cell(exclude=taken)
            self.agent_positions.append(pos)
            taken.add(pos)
        
        # Mark entire spawn zone as taken for obstacle/object placement
        for pos in spawn_zone:
            taken.add(pos)
        
        # ═══════════════════════════════════════════════════════════════════
        # GOALS: Place in opposite corner (top-right) - delivery zone
        # ═══════════════════════════════════════════════════════════════════
        goal_zone = []
        for r in range(min(3, self.grid_size)):  # First 3 rows
            for c in range(self.grid_size - 3, self.grid_size):  # Last 3 columns
                goal_zone.append((r, c))
        
        self.goal_positions = []
        available_goals = [g for g in goal_zone if g not in taken]
        self.np_random.shuffle(available_goals)
        for i in range(self.n_goals):
            if i < len(available_goals):
                pos = available_goals[i]
            else:
                pos = self._random_cell(exclude=taken)
            self.goal_positions.append(pos)
            taken.add(pos)
        self.goal_used = [False] * self.n_goals
        
        # Mark goal zone as taken
        for pos in goal_zone:
            taken.add(pos)
        
        # ═══════════════════════════════════════════════════════════════════
        # OBSTACLES: Random in middle area (not in spawn/goal zones)
        # ═══════════════════════════════════════════════════════════════════
        self.obstacles = []
        while len(self.obstacles) < self.n_obstacles:
            cell = self._random_cell(exclude=taken)
            self.obstacles.append(cell)
            taken.add(cell)

        # ═══════════════════════════════════════════════════════════════════
        # OBJECTS: Random in middle area
        # ═══════════════════════════════════════════════════════════════════
        self.object_positions = []
        while len(self.object_positions) < self.n_objects:
            cell = self._random_cell(exclude=taken)
            self.object_positions.append(cell)
            taken.add(cell)
        self.delivered = [False] * self.n_objects

        self._prev_agent_positions = list(self.agent_positions)
        self._prev_object_positions = list(self.object_positions)
        self._prev_goal_used = list(self.goal_used)
        self._prev_delivered = list(self.delivered)
        self._collision_count = 0

        return self._get_obs(), {}

    def step(self, action: Sequence[int]):
        action_arr = self._parse_action(action)
        self.current_step += 1
        
        # Store previous state
        self._prev_agent_positions = list(self.agent_positions)
        self._prev_object_positions = list(self.object_positions)
        self._prev_delivered = list(self.delivered)
        self._collision_count = 0
        self._obstacle_hits = 0  # Track obstacle collisions separately

        moved_agents = set()

        # Handle cooperative carrying: object moves only when EXACTLY 2 agents are on it and agree on direction
        for obj_idx, obj_pos in enumerate(list(self.object_positions)):
            if self.delivered[obj_idx]:
                continue
            carriers = [i for i, pos in enumerate(self.agent_positions) if pos == obj_pos]
            if len(carriers) != 2:
                continue

            delta = self._action_to_delta[int(action_arr[carriers[0]])]
            if not all(self._action_to_delta[int(action_arr[c])] == delta for c in carriers):
                continue  # disagree on direction

            target = (obj_pos[0] + delta[0], obj_pos[1] + delta[1])
            
            # Check if target hits obstacle
            if target in self.obstacles:
                self._obstacle_hits += 1
                continue
            
            if not self._is_cell_valid(target) or target in self.object_positions:
                if delta != (0, 0):
                    self._collision_count += 1
                continue
            # Move object and carriers
            self.object_positions[obj_idx] = target
            for c in carriers:
                self.agent_positions[c] = target
                moved_agents.add(c)

        # Move remaining agents independently
        for idx in range(self.n_agents):
            if idx in moved_agents:
                continue
            delta = self._action_to_delta[int(action_arr[idx])]
            current = self.agent_positions[idx]
            candidate = (current[0] + delta[0], current[1] + delta[1])

            # Check obstacle hit FIRST (separate tracking)
            if candidate in self.obstacles:
                self._obstacle_hits += 1
                candidate = current  # Don't move
            elif self._is_cell_valid(candidate):
                # Prevent agents from joining an object that already has two carriers
                for obj_idx, obj_pos in enumerate(self.object_positions):
                    if self.delivered[obj_idx]:
                        continue
                    if candidate != obj_pos:
                        continue
                    existing_carriers = sum(1 for a_idx, a_pos in enumerate(self.agent_positions) if a_idx != idx and a_pos == obj_pos)
                    if existing_carriers >= 2:
                        candidate = current
                        break
            else:
                if delta != (0, 0):
                    self._collision_count += 1
                candidate = current

            self.agent_positions[idx] = candidate

        # Check deliveries
        for i, obj_pos in enumerate(self.object_positions):
            if self.delivered[i]:
                continue
            for g_idx, goal_pos in enumerate(self.goal_positions):
                if self.goal_used[g_idx]:
                    continue
                if obj_pos == goal_pos:
                    self.delivered[i] = True
                    self.goal_used[g_idx] = True
                    break

        reward, reward_info = self._compute_reward(action_arr)
        
        terminated = all(self.delivered)
        truncated = self.current_step >= self.max_steps
        info = {
            "delivered": list(self.delivered),
            "reward_breakdown": reward_info,
            "step": self.current_step,
        }

        return self._get_obs(), reward, terminated, truncated, info
    
    def _compute_reward(self, action_arr: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        SIMPLIFIED reward function focused on core objectives:
        1. Deliver objects (MAIN GOAL)
        2. Avoid obstacles (SAFETY)
        3. Spread evenly / no third-wheeling (EFFICIENCY)
        4. Progress toward goals (GUIDANCE)
        """
        cfg = self.reward_config
        reward_info: Dict[str, float] = {}
        total_reward = 0.0

        # ═══════════════════════════════════════════════════════════════════
        # HELPER FUNCTIONS
        # ═══════════════════════════════════════════════════════════════════
        
        def carriers_now(idx: int) -> int:
            obj_pos = self.object_positions[idx]
            return sum(1 for p in self.agent_positions if p == obj_pos)

        def carriers_prev(idx: int) -> int:
            obj_pos = self._prev_object_positions[idx]
            return sum(1 for p in self._prev_agent_positions if p == obj_pos)

        def nearest_goal_dist(pos: Coord) -> int:
            d = None
            for g_idx, gpos in enumerate(self.goal_positions):
                if self.goal_used[g_idx]:
                    continue
                md = self._manhattan(pos, gpos)
                if d is None or md < d:
                    d = md
            return d if d is not None else 0

        # ═══════════════════════════════════════════════════════════════════
        # 1. OBSTACLE HITS (HARD PENALTY - must avoid!)
        # ═══════════════════════════════════════════════════════════════════
        if self._obstacle_hits > 0:
            obstacle_penalty = cfg["alpha_obstacle_hit"] * self._obstacle_hits
            reward_info["obstacle_hit"] = obstacle_penalty
            total_reward += obstacle_penalty

        # ═══════════════════════════════════════════════════════════════════
        # 2. TIME PENALTY (simple linear)
        # ═══════════════════════════════════════════════════════════════════
        reward_info["step"] = cfg["alpha_step"]
        total_reward += cfg["alpha_step"]

        # ═══════════════════════════════════════════════════════════════════
        # 3. DELIVERY REWARD (THE MAIN OBJECTIVE)
        # ═══════════════════════════════════════════════════════════════════
        for i, delivered in enumerate(self.delivered):
            if delivered and not self._prev_delivered[i]:
                delivery_gain = cfg["alpha_del"]
                steps_left = max(0, self.max_steps - self.current_step)
                efficiency_gain = cfg["alpha_eff"] * min(steps_left, cfg["T_cap"])
                reward_info.setdefault("delivery", 0.0)
                reward_info["delivery"] += delivery_gain + efficiency_gain
                total_reward += delivery_gain + efficiency_gain

        # ═══════════════════════════════════════════════════════════════════
        # 4. PICKUP / DROP
        # ═══════════════════════════════════════════════════════════════════
        for i in range(self.n_objects):
            if self.delivered[i]:
                continue
            now_carried = carriers_now(i)
            prev_carried = carriers_prev(i)
            
            if now_carried == 2 and prev_carried < 2:
                reward_info.setdefault("pickup", 0.0)
                reward_info["pickup"] += cfg["alpha_pickup"]
                total_reward += cfg["alpha_pickup"]
            
            if prev_carried == 2 and now_carried < 2:
                reward_info.setdefault("drop", 0.0)
                reward_info["drop"] += cfg["alpha_drop"]
                total_reward += cfg["alpha_drop"]

        # ═══════════════════════════════════════════════════════════════════
        # 5. CARRY PROGRESS (moving object toward goal)
        # ═══════════════════════════════════════════════════════════════════
        for i in range(self.n_objects):
            if self.delivered[i]:
                continue
            if carriers_now(i) >= 2:
                prev_dist = nearest_goal_dist(self._prev_object_positions[i])
                curr_dist = nearest_goal_dist(self.object_positions[i])
                progress = prev_dist - curr_dist  # Positive = closer to goal
                
                if progress != 0:
                    gain = cfg["alpha_carry_progress"] * progress
                    reward_info.setdefault("carry_progress", 0.0)
                    reward_info["carry_progress"] += gain
                    total_reward += gain

        # ═══════════════════════════════════════════════════════════════════
        # 6. THIRD-WHEEL PENALTY (3+ agents on object = inefficient)
        # ═══════════════════════════════════════════════════════════════════
        for i in range(self.n_objects):
            if self.delivered[i]:
                continue
            extra = max(0, carriers_now(i) - 2)
            if extra > 0:
                penalty = cfg["alpha_third_wheel"] * extra
                reward_info.setdefault("third_wheel", 0.0)
                reward_info["third_wheel"] += penalty
                total_reward += penalty

        # ═══════════════════════════════════════════════════════════════════
        # 7. POTENTIAL-BASED SHAPING (prevents reward hacking!)
        #    Reward = gamma * Phi(s') - Phi(s)
        #    Phi(s) = -sum of distances from each agent to nearest needy object
        #    This mathematically guarantees no exploitation through oscillation
        # ═══════════════════════════════════════════════════════════════════
        def compute_potential(agent_positions, object_positions, delivered_flags):
            """Compute potential: negative sum of distances to needy objects."""
            potential = 0.0
            for agent_idx in range(self.n_agents):
                agent_pos = agent_positions[agent_idx]
                
                # Check if this agent is on an object with 2 carriers (carrying)
                is_carrying = False
                for obj_idx in range(self.n_objects):
                    if delivered_flags[obj_idx]:
                        continue
                    if agent_pos == object_positions[obj_idx]:
                        carriers = sum(1 for p in agent_positions if p == object_positions[obj_idx])
                        if carriers >= 2:
                            is_carrying = True
                            break
                
                if is_carrying:
                    continue  # Carriers don't contribute to approach potential
                
                # Find nearest needy object
                best_dist = None
                for obj_idx in range(self.n_objects):
                    if delivered_flags[obj_idx]:
                        continue
                    carriers = sum(1 for p in agent_positions if p == object_positions[obj_idx])
                    if carriers >= 2:
                        continue
                    d = self._manhattan(agent_pos, object_positions[obj_idx])
                    if best_dist is None or d < best_dist:
                        best_dist = d
                
                if best_dist is not None:
                    potential -= best_dist  # Negative distance = higher potential when closer
            
            return potential
        
        prev_potential = compute_potential(
            self._prev_agent_positions, 
            self._prev_object_positions,
            self._prev_delivered
        )
        curr_potential = compute_potential(
            self.agent_positions,
            self.object_positions,
            self.delivered
        )
        
        # Shaping reward: gamma * Phi(s') - Phi(s)
        gamma = 0.99
        shaping_reward = cfg["alpha_approach"] * (gamma * curr_potential - prev_potential)
        
        if abs(shaping_reward) > 0.001:
            reward_info["shaping"] = shaping_reward
            total_reward += shaping_reward

        return total_reward, reward_info
    
    def _manhattan(self, a: Coord, b: Coord) -> int:
        """Manhattan distance between two coordinates."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def render(self):
        if self.render_mode != "human":
            return self._get_obs()
        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Collaborative Carry Environment")
            self.font = pygame.font.SysFont("Arial", 28, bold=True)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((245, 245, 245))
        self._draw_grid()
        self._draw_goal()
        self._draw_obstacles()
        self._draw_object()
        self._draw_agents()

        pygame.display.flip()
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

    # Helpers -----------------------------------------------------------------
    def _parse_action(self, action: Sequence[int]) -> np.ndarray:
        arr = np.asarray(action, dtype=np.int32).flatten()
        if arr.size != self.n_agents:
            raise ValueError("Action must provide exactly one entry per agent.")
        if np.any(arr < 0) or np.any(arr >= len(self._action_to_delta)):
            raise ValueError("Action entries must be within the valid discrete range.")
        return arr

    def _random_cell(self, exclude: Optional[set] = None) -> Coord:
        exclude = exclude or set()
        while True:
            cell = (
                int(self.np_random.integers(0, self.grid_size)),
                int(self.np_random.integers(0, self.grid_size)),
            )
            if cell not in exclude:
                return cell

    def _get_obs(self):
        """
        SIMPLIFIED 14-dim observation per agent:
        - pos_norm(2)                    - Agent position
        - vec_to_nearest_obj(2)          - Direction to nearest needy object
        - dist_to_nearest_obj(1)         - Distance to it
        - vec_to_nearest_goal(2)         - Direction to nearest free goal  
        - dist_to_nearest_goal(1)        - Distance to it
        - nearest_obstacle_dist(1)       - Distance to nearest obstacle (danger awareness)
        - on_object(1)                   - 1 if on an object, 0 otherwise
        - object_has_partner(1)          - 1 if object has 2 carriers (ready to move)
        - am_i_needed(1)                 - 1 if there's an object needing help
        - agent_density_around_me(1)     - Fraction of other agents within 2 cells
        """
        max_dist = np.sqrt(2) * max(1, (self.grid_size - 1))

        def normalize_pos(pos: Coord) -> np.ndarray:
            return np.array([pos[0] / (self.grid_size - 1), pos[1] / (self.grid_size - 1)], dtype=np.float32) * 2 - 1

        def compute_vec(from_pos: Coord, to_pos: Coord) -> np.ndarray:
            vec = np.array([to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]], dtype=np.float32)
            return vec / max(1, (self.grid_size - 1))

        def compute_dist(from_pos: Coord, to_pos: Coord) -> float:
            return np.sqrt((to_pos[0] - from_pos[0])**2 + (to_pos[1] - from_pos[1])**2) / max_dist

        agent_obs = []
        for idx in range(self.n_agents):
            pos = self.agent_positions[idx]

            # ═══════════════════════════════════════════════════════════════
            # Find nearest NEEDY object (< 2 carriers, not delivered)
            # ═══════════════════════════════════════════════════════════════
            nearest_obj_vec = np.array([0.0, 0.0], dtype=np.float32)
            nearest_obj_dist = 1.0
            best_obj_dist = None
            for obj_idx, obj_pos in enumerate(self.object_positions):
                if self.delivered[obj_idx]:
                    continue
                carriers = sum(1 for p in self.agent_positions if p == obj_pos)
                if carriers >= 2 and pos != obj_pos:
                    continue  # Skip fully staffed objects (unless we're on it)
                d = compute_dist(pos, obj_pos)
                if best_obj_dist is None or d < best_obj_dist:
                    best_obj_dist = d
                    nearest_obj_vec = compute_vec(pos, obj_pos)
                    nearest_obj_dist = d

            # ═══════════════════════════════════════════════════════════════
            # Find nearest FREE goal (not yet used)
            # ═══════════════════════════════════════════════════════════════
            nearest_goal_vec = np.array([0.0, 0.0], dtype=np.float32)
            nearest_goal_dist = 1.0
            best_goal_dist = None
            for g_idx, gpos in enumerate(self.goal_positions):
                if self.goal_used[g_idx]:
                    continue
                d = compute_dist(pos, gpos)
                if best_goal_dist is None or d < best_goal_dist:
                    best_goal_dist = d
                    nearest_goal_vec = compute_vec(pos, gpos)
                    nearest_goal_dist = d

            # ═══════════════════════════════════════════════════════════════
            # Nearest obstacle distance (for danger awareness)
            # ═══════════════════════════════════════════════════════════════
            nearest_obs_dist = 1.0
            for obs_pos in self.obstacles:
                d = compute_dist(pos, obs_pos)
                if d < nearest_obs_dist:
                    nearest_obs_dist = d

            # ═══════════════════════════════════════════════════════════════
            # Am I on an object?
            # ═══════════════════════════════════════════════════════════════
            on_object = 0.0
            my_object_carriers = 0
            for obj_idx, obj_pos in enumerate(self.object_positions):
                if not self.delivered[obj_idx] and pos == obj_pos:
                    on_object = 1.0
                    my_object_carriers = sum(1 for p in self.agent_positions if p == obj_pos)
                    break

            # ═══════════════════════════════════════════════════════════════
            # Does my object have a partner? (ready to carry)
            # ═══════════════════════════════════════════════════════════════
            object_has_partner = 1.0 if my_object_carriers >= 2 else 0.0

            # ═══════════════════════════════════════════════════════════════
            # Am I needed? (any object with < 2 carriers)
            # ═══════════════════════════════════════════════════════════════
            am_i_needed = 0.0
            for obj_idx, obj_pos in enumerate(self.object_positions):
                if self.delivered[obj_idx]:
                    continue
                carriers = sum(1 for p in self.agent_positions if p == obj_pos)
                if carriers < 2:
                    am_i_needed = 1.0
                    break

            # ═══════════════════════════════════════════════════════════════
            # Agent density (how many others within manhattan distance 2)
            # ═══════════════════════════════════════════════════════════════
            nearby_count = 0
            for jdx, other_pos in enumerate(self.agent_positions):
                if jdx == idx:
                    continue
                if self._manhattan(pos, other_pos) <= 2:
                    nearby_count += 1
            agent_density = nearby_count / max(1, self.n_agents - 1)

            # Build observation (14 dims)
            obs = np.array([
                normalize_pos(pos)[0],
                normalize_pos(pos)[1],
                nearest_obj_vec[0],
                nearest_obj_vec[1],
                nearest_obj_dist,
                nearest_goal_vec[0],
                nearest_goal_vec[1],
                nearest_goal_dist,
                nearest_obs_dist,
                on_object,
                object_has_partner,
                am_i_needed,
                agent_density,
                0.0,  # Padding to 14 dims for safety
            ], dtype=np.float32)
            agent_obs.append(obs)

        # Pad agent obs up to max_agents with -1
        while len(agent_obs) < self.max_agents:
            agent_obs.append(np.full(self.obs_dim_per_agent, -1.0, dtype=np.float32))

        # Obstacles padded
        obstacle_positions_norm = []
        for obs_pos in self.obstacles:
            obstacle_positions_norm.extend(normalize_pos(obs_pos).tolist())
        while len(obstacle_positions_norm) < self.max_obstacles * 2:
            obstacle_positions_norm.extend([-1.0, -1.0])

        # Objects padded
        objects_flat = []
        for obj_idx in range(self.max_objects):
            if obj_idx < self.n_objects:
                obj_pos = self.object_positions[obj_idx]
                # nearest unused goal for representation
                goal_pos_sel = None
                goal_d = None
                for g_idx, gpos in enumerate(self.goal_positions):
                    if self.goal_used[g_idx]:
                        continue
                    gd = self._manhattan(obj_pos, gpos)
                    if goal_d is None or gd < goal_d:
                        goal_d = gd
                        goal_pos_sel = gpos
                goal_pos_sel = goal_pos_sel if goal_pos_sel is not None else self.goal_positions[0]
                carried_flag = 1.0 if sum(1 for p in self.agent_positions if p == obj_pos) == 2 else 0.0
                delivered_flag = 1.0 if self.delivered[obj_idx] else 0.0
                objects_flat.extend(normalize_pos(obj_pos).tolist())
                objects_flat.extend(normalize_pos(goal_pos_sel).tolist())
                objects_flat.append(carried_flag)
                objects_flat.append(delivered_flag)
            else:
                objects_flat.extend([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

        step_norm = self.current_step / max(1, self.max_steps)
        n_agents_norm = self.n_agents / max(1, self.max_agents)
        grid_norm = (self.grid_size - 1) / max(1, (self.max_grid_size - 1))

        shared_obs = np.concatenate([
            np.array([step_norm, n_agents_norm, grid_norm], dtype=np.float32),
            np.array(obstacle_positions_norm, dtype=np.float32),
            np.array(objects_flat, dtype=np.float32),
        ])

        return {
            **{f"agent_{i+1}": agent_obs[i] for i in range(self.max_agents)},
            "shared": shared_obs,
        }
    
    def get_flat_obs(self) -> np.ndarray:
        """Return a flattened observation vector (useful for simple RL algorithms)."""
        obs = self._get_obs()
        agents = [obs[f"agent_{i+1}"] for i in range(self.n_agents)]
        return np.concatenate(agents + [obs["shared"]])
    
    def get_agent_obs(self, agent_id: int) -> np.ndarray:
        """Return observation for a specific agent (useful for decentralized training)."""
        obs = self._get_obs()
        agent_key = f"agent_{agent_id + 1}"
        return np.concatenate([obs[agent_key], obs["shared"]])

    def _is_cell_valid(self, cell: Coord) -> bool:
        if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size):
            return False
        if cell in self.obstacles:
            return False
        return True

    # Legacy helpers removed in multi-agent refactor

    def _draw_grid(self) -> None:
        for i in range(self.grid_size + 1):
            start = (0, i * self.cell_size)
            end = (self.window_size, i * self.cell_size)
            pygame.draw.line(self.screen, (0, 0, 0), start, end, 1)
            start = (i * self.cell_size, 0)
            end = (i * self.cell_size, self.window_size)
            pygame.draw.line(self.screen, (0, 0, 0), start, end, 1)

    def _draw_rect(self, cell: Coord, color: Tuple[int, int, int], padding: int = 8) -> None:
        x = cell[1] * self.cell_size + padding
        y = cell[0] * self.cell_size + padding
        size = self.cell_size - 2 * padding
        pygame.draw.rect(self.screen, color, (x, y, size, size))

    def _draw_goal(self) -> None:
        for idx, goal in enumerate(self.goal_positions):
            self._draw_rect(goal, (255, 0, 0), padding=4)
            self._draw_text(goal, f"G{idx+1}", (255, 255, 255))

    def _draw_obstacles(self) -> None:
        for obstacle in self.obstacles:
            self._draw_rect(obstacle, (120, 120, 120))
            self._draw_text(obstacle, "X", (255, 255, 255))

    def _draw_object(self) -> None:
        for idx, obj in enumerate(self.object_positions):
            delivered = self.delivered[idx]
            padding = 10 if delivered else 12
            color = (200, 200, 200) if delivered else (255, 215, 0)
            self._draw_rect(obj, color, padding=padding)
            self._draw_text(obj, f"O{idx+1}", (0, 0, 0))

    def _draw_agents(self) -> None:
        palette = [
            (70, 130, 180), (46, 139, 87), (138, 43, 226), (255, 99, 71),
            (255, 140, 0), (60, 179, 113), (123, 104, 238), (199, 21, 133),
            (0, 191, 255), (189, 183, 107)
        ]
        for idx, pos in enumerate(self.agent_positions):
            color = palette[idx % len(palette)]
            self._draw_rect(pos, color, padding=12)
            self._draw_text(pos, str(idx + 1), (255, 255, 255))

    def _draw_text(self, cell: Coord, text: str, color: Tuple[int, int, int]) -> None:
        surface = self.font.render(text, True, color)
        rect = surface.get_rect()
        rect.center = (
            cell[1] * self.cell_size + self.cell_size // 2,
            cell[0] * self.cell_size + self.cell_size // 2,
        )
        self.screen.blit(surface, rect)
