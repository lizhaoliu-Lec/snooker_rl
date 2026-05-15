"""
Snooker Environment - Two-player self-play with realistic action space

Decision process per turn:
  1. [If ball-in-hand] Place cue ball in the D-zone
  2. Select a target ball (discrete)
  3. Choose offset angle (relative to white→target line) + power (continuous)
  4. Physics simulation
  5. Score / foul detection
  6. Switch to opponent (or continue if potted legally)

Action space (Dict / flattened):
  - place_x, place_y : [-1, 1] cue ball placement in D-zone (used only when ball-in-hand)
  - target_idx       : [0, N)  index into the list of available target balls
  - angle_offset     : [-1, 1] offset from the white→target line (maps to ±15°)
  - power            : [-1, 1] shot power (maps to [0.1, 1])
"""

import numpy as np
import pygame
import pymunk
from gymnasium import spaces
import gymnasium as gym


# ── pymunk collision types ──────────────────────────────────
COL_WHITE = 1
COL_RED = 2
COL_COLOR = 3
COL_CUSHION = 4

# Maximum number of target-ball slots (15 reds + 6 colours)
MAX_TARGET_BALLS = 21


class Ball:
    """Lightweight wrapper around a pymunk body."""
    def __init__(self, x, y, radius, color, ball_type='red'):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.ball_type = ball_type
        self.pocketed = False
        self.velocity = np.array([0.0, 0.0])

    def update(self, body):
        self.x = body.position.x
        self.y = body.position.y
        self.velocity = np.array([body.velocity.x, body.velocity.y])

    def is_moving(self, threshold=0.5):
        return np.linalg.norm(self.velocity) > threshold


class SnookerEnv(gym.Env):
    """
    Two-player Snooker environment with realistic action decomposition.

    Action vector (7-dim continuous, all in [-1, 1]):
      [0] place_x      – D-zone x offset   (only used when ball_in_hand)
      [1] place_y      – D-zone y offset   (only used when ball_in_hand)
      [2] target_idx   – mapped to discrete target ball index
      [3] angle_offset – deviation from white→target centre line (±15°)
      [4] power        – shot power

    Observation includes whose turn it is and ball-in-hand flag.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    COLOR_ORDER = ['yellow', 'green', 'brown', 'blue', 'pink', 'black']
    COLOR_VALUES = {'yellow': 2, 'green': 3, 'brown': 4,
                    'blue': 5, 'pink': 6, 'black': 7}

    # ── RGB colours for rendering ────────────────────────────
    BALL_COLORS = {
        'white': (255, 255, 255), 'yellow': (255, 255, 0),
        'green': (0, 180, 0), 'brown': (139, 69, 19),
        'blue': (30, 144, 255), 'pink': (255, 192, 203),
        'black': (30, 30, 30),
    }

    def __init__(self, table_width=1200, table_height=600, render_mode=None):
        super().__init__()
        self.table_width = table_width
        self.table_height = table_height
        self.render_mode = render_mode

        # Physics
        self.friction = 0.02
        self.restitution = 0.95            # ball-ball elasticity (high = less energy lost)
        self.ball_radius = 12
        self.pocket_radius = 22
        self.cushion_width = 25

        self.pockets = [
            (self.cushion_width + 5, self.cushion_width + 5),
            (self.table_width // 2, self.cushion_width - 5),
            (self.table_width - self.cushion_width - 5, self.cushion_width + 5),
            (self.cushion_width + 5, self.table_height - self.cushion_width - 5),
            (self.table_width // 2, self.table_height - self.cushion_width + 5),
            (self.table_width - self.cushion_width - 5,
             self.table_height - self.cushion_width - 5),
        ]

        # D-zone geometry (semicircle in the baulk area)
        self.baulk_x = self.table_width * 0.22
        self.d_radius = self.table_height * 0.15
        self.d_centre_y = self.table_height / 2

        # ── Action space: 5 continuous dims ──────────────────
        # All outputs in [-1, 1]; mapped inside step()
        self.action_space = spaces.Box(
            low=-np.ones(5, dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
            dtype=np.float32,
        )

        # ── Observation space ────────────────────────────────
        self._define_observation_space()

        # Game state (populated in reset)
        self.space = None
        self.balls = {}
        self.color_spots = {}

        # Per-player scores  (index 0 = player 1, 1 = player 2)
        self.scores = [0, 0]
        self.pot_scores = [0, 0]       # points earned by potting balls
        self.foul_received = [0, 0]    # points received from opponent fouls
        self.current_player = 0          # whose turn
        self.current_break = 0
        self.consecutive_fouls = [0, 0]
        self.shots_without_pocket = 0
        self.max_shots_without_pocket = 10

        # Phase tracking
        self.phase = 'red'
        self.next_color_index = 0

        # Ball-in-hand flag
        self.ball_in_hand = True  # True at start / after foul

        # Collision tracking
        self.first_contact = None
        self.white_hit_any = False
        self.last_pocketed_balls = []

        # Pygame
        self.screen = None
        self.clock = None
        self.font = None

    # ════════════════════════════════════════════════════════
    # Observation / action helpers
    # ════════════════════════════════════════════════════════
    def _define_observation_space(self):
        """
        Obs layout (58-d):
          white ball  : x, y          (2)
          15 reds     : x, y each     (30)
          6 colours   : x, y each     (12)
          game state  : current_break/147, phase(0/0.5/1),
                        next_color_idx/6, remaining/21,
                        ball_in_hand(0/1), current_player(0/1),
                        score_p1/147, score_p2/147,
                        consec_fouls_self/3, shots_no_pot/10  (10)
          pocket pos  : 6×2 = 12  (helps agent learn geometry)   -- skip for now
        total = 2 + 30 + 12 + 10 = 54  (but let's add pocket info → 54)
        """
        n = 2 + 15 * 2 + 6 * 2 + 10
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(n,), dtype=np.float32)

    def _get_obs(self):
        obs = []
        # White ball
        w = self.balls.get('white')
        if w and not w['ball'].pocketed:
            obs.extend(self._norm_pos(w['ball'].x, w['ball'].y))
        else:
            obs.extend([0.0, 0.0])

        # Red balls
        for i in range(15):
            k = f'red_{i}'
            if k in self.balls and not self.balls[k]['ball'].pocketed:
                obs.extend(self._norm_pos(
                    self.balls[k]['ball'].x, self.balls[k]['ball'].y))
            else:
                obs.extend([-1.0, -1.0])

        # Colour balls
        for c in self.COLOR_ORDER:
            if c in self.balls and not self.balls[c]['ball'].pocketed:
                obs.extend(self._norm_pos(
                    self.balls[c]['ball'].x, self.balls[c]['ball'].y))
            else:
                obs.extend([-1.0, -1.0])

        # Game state
        phase_val = {'red': 0.0, 'color': 0.5, 'final_colors': 1.0}.get(
            self.phase, 0.0)
        remaining = sum(1 for b in self.balls.values()
                        if not b['ball'].pocketed and b['ball'].ball_type != 'white')
        obs.extend([
            self.current_break / 147.0,
            phase_val,
            self.next_color_index / 6.0,
            remaining / 21.0,
            1.0 if self.ball_in_hand else 0.0,
            self.current_player,               # 0 or 1
            self.scores[0] / 147.0,
            self.scores[1] / 147.0,
            self.consecutive_fouls[self.current_player] / 3.0,
            self.shots_without_pocket / self.max_shots_without_pocket,
        ])
        return np.array(obs, dtype=np.float32)

    def _norm_pos(self, x, y):
        """Normalise table coords to [-1, 1]."""
        return ((x / self.table_width) * 2 - 1,
                (y / self.table_height) * 2 - 1)

    def _get_legal_targets(self):
        """
        Return ordered list of ball-type keys the current player may target.
        """
        targets = []
        if self.phase == 'red':
            for i in range(15):
                k = f'red_{i}'
                if k in self.balls and not self.balls[k]['ball'].pocketed:
                    targets.append(k)
        elif self.phase == 'color':
            for c in self.COLOR_ORDER:
                if c in self.balls and not self.balls[c]['ball'].pocketed:
                    targets.append(c)
        elif self.phase == 'final_colors':
            if self.next_color_index < 6:
                c = self.COLOR_ORDER[self.next_color_index]
                if c in self.balls and not self.balls[c]['ball'].pocketed:
                    targets.append(c)
        return targets

    def _get_all_targetable(self):
        """All non-white, non-pocketed balls (for illegal-but-still-need-physics)."""
        return [k for k, v in self.balls.items()
                if k != 'white' and not v['ball'].pocketed]

    # ════════════════════════════════════════════════════════
    # Physics helpers
    # ════════════════════════════════════════════════════════
    def _setup_space(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        # damping per step. With dt=1/180 (sub-stepping), we want the same
        # per-second decay as the original 0.99^60 ≈ 0.55/s.
        # Equivalent: 0.99^(1/3) ≈ 0.9967 per 1/180s step.
        self.space.damping = 0.9967
        self.space.iterations = 20           # collision solver iterations (default 10)

        cw = self.cushion_width
        tw = self.table_width
        th = self.table_height
        seg_r = 4                       # segment thickness (half-width)
        gap = self.pocket_radius + 10   # gap size at each pocket

        # ── Pocket positions (for gap calculation) ──
        # Corners: TL(cw,cw), TR(tw-cw,cw), BL(cw,th-cw), BR(tw-cw,th-cw)
        # Middles: TC(tw/2, cw), BC(tw/2, th-cw)
        corner_gap = gap * 1.4  # corners need wider gap (diagonal approach)
        mid_gap = gap

        # ── Top wall: y = cw, from x=cw to x=tw-cw ──
        # Gaps at: TL corner (x=cw), TC middle (x=tw/2), TR corner (x=tw-cw)
        top_segs = [
            (cw + corner_gap, cw, tw / 2 - mid_gap, cw),
            (tw / 2 + mid_gap, cw, tw - cw - corner_gap, cw),
        ]
        # ── Bottom wall: y = th-cw ──
        bot_segs = [
            (cw + corner_gap, th - cw, tw / 2 - mid_gap, th - cw),
            (tw / 2 + mid_gap, th - cw, tw - cw - corner_gap, th - cw),
        ]
        # ── Left wall: x = cw, from y=cw to y=th-cw ──
        # Gaps at: TL corner (y=cw), BL corner (y=th-cw)
        left_segs = [
            (cw, cw + corner_gap, cw, th - cw - corner_gap),
        ]
        # ── Right wall: x = tw-cw ──
        right_segs = [
            (tw - cw, cw + corner_gap, tw - cw, th - cw - corner_gap),
        ]

        all_segs = top_segs + bot_segs + left_segs + right_segs
        for x1, y1, x2, y2 in all_segs:
            seg = pymunk.Segment(self.space.static_body, (x1, y1), (x2, y2), seg_r)
            seg.elasticity = 0.8
            seg.friction = 0.5
            seg.collision_type = COL_CUSHION
            self.space.add(seg)

        self.space.on_collision(collision_type_a=COL_WHITE,
                                collision_type_b=COL_RED,
                                begin=self._on_white_contact)
        self.space.on_collision(collision_type_a=COL_WHITE,
                                collision_type_b=COL_COLOR,
                                begin=self._on_white_contact)

    def _on_white_contact(self, arbiter, space, data):
        self.white_hit_any = True
        if self.first_contact is None:
            for shape in arbiter.shapes:
                if shape.collision_type != COL_WHITE:
                    for btype, bdata in self.balls.items():
                        if bdata['shape'] is shape:
                            self.first_contact = btype
                            return

    def _create_ball(self, x, y, ball_type, color):
        ball = Ball(x, y, self.ball_radius, color, ball_type)
        body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, self.ball_radius))
        body.position = (x, y)
        shape = pymunk.Circle(body, self.ball_radius)
        shape.elasticity = self.restitution
        shape.friction = self.friction
        if ball_type == 'white':
            shape.collision_type = COL_WHITE
        elif ball_type.startswith('red_'):
            shape.collision_type = COL_RED
        else:
            shape.collision_type = COL_COLOR
        self.space.add(body, shape)
        self.balls[ball_type] = {'ball': ball, 'body': body, 'shape': shape}

    def _setup_initial_positions(self):
        pos = {}
        pos['white'] = (self.table_width * 0.25, self.table_height / 2)
        reds = []
        sp = self.ball_radius * 2.2
        bx, by = self.table_width * 0.7, self.table_height / 2
        for row in range(5):
            for col in range(row + 1):
                x = bx + row * sp * 0.866
                y = by + (col - row / 2) * sp
                if len(reds) < 15:
                    reds.append((x, y))
        pos['red'] = reds
        baulk = self.baulk_x
        pos['yellow'] = (baulk, self.table_height * 0.35)
        pos['green'] = (baulk, self.table_height * 0.65)
        pos['brown'] = (baulk, self.table_height / 2)
        pos['blue'] = (self.table_width / 2, self.table_height / 2)
        pos['pink'] = (self.table_width * 0.65, self.table_height / 2)
        pos['black'] = (self.table_width * 0.85, self.table_height / 2)
        return pos

    def _place_cue_ball_in_d(self, nx, ny):
        """
        Place cue ball using normalised [-1,1] coords mapped to D-zone.
        D-zone is a semicircle on the left side of the baulk line.
        """
        # Map to D-zone: x in [baulk_x - d_radius, baulk_x], y in semicircle
        cx = self.baulk_x
        cy = self.d_centre_y
        r = self.d_radius
        # raw offset
        dx = ((nx + 1) / 2) * r      # 0 .. r  (0=left edge, r=baulk line)
        dy = ny * r                   # -r .. r
        # clamp to semicircle
        dist = np.sqrt(dx ** 2 + dy ** 2)
        if dist > r:
            dx *= r / dist
            dy *= r / dist
        px = cx - r + dx              # shift so right edge = baulk_x
        py = cy + dy
        # clamp inside cushions
        m = self.cushion_width + self.ball_radius + 2
        px = np.clip(px, m, self.table_width - m)
        py = np.clip(py, m, self.table_height - m)
        return float(px), float(py)

    def _respot_color(self, ball_type):
        if ball_type not in self.balls or ball_type not in self.color_spots:
            return
        sx, sy = self.color_spots[ball_type]
        bd = self.balls[ball_type]
        bd['body'].position = (sx, sy)
        bd['body'].velocity = (0, 0)
        bd['ball'].x = sx
        bd['ball'].y = sy
        bd['ball'].pocketed = False
        bd['ball'].velocity = np.array([0.0, 0.0])

    def _count_remaining_reds(self):
        return sum(1 for i in range(15)
                   if f'red_{i}' in self.balls
                   and not self.balls[f'red_{i}']['ball'].pocketed)

    def _simulate_until_stable(self, max_steps=1500):
        """
        Run physics until all balls stop.
        Uses sub-stepping (3 × 1/180s per frame) for collision accuracy.
        """
        self.last_pocketed_balls = []
        dt = 1 / 180                         # sub-step: 3× finer than 1/60
        sub_steps = 3                         # sub-steps per render frame
        for frame in range(max_steps):
            for _ in range(sub_steps):
                self.space.step(dt)
            for bd in self.balls.values():
                bd['ball'].update(bd['body'])
            for btype, bd in list(self.balls.items()):
                if not bd['ball'].pocketed:
                    if self._is_pocketed(bd['ball']):
                        self._handle_pocket(btype, bd['ball'])

            # ── Live rendering: draw every frame so user sees ball motion ──
            if self.render_mode == 'human' and self.screen is not None:
                self._render_human()
                # Process pygame events so the window stays responsive
                for evt in pygame.event.get():
                    if evt.type == pygame.QUIT:
                        pass  # handled externally

            if all(not bd['ball'].is_moving()
                   for bd in self.balls.values()
                   if not bd['ball'].pocketed):
                break

    def _is_pocketed(self, ball):
        for px, py in self.pockets:
            if np.hypot(ball.x - px, ball.y - py) < self.pocket_radius:
                return True
        return False

    def _handle_pocket(self, btype, ball):
        ball.pocketed = True
        ball.velocity = np.array([0, 0])
        self.last_pocketed_balls.append(btype)
        bd = self.balls[btype]
        bd['body'].position = (-100, -100)
        bd['body'].velocity = (0, 0)

    def _min_dist_to_pocket(self, ball):
        return min(np.hypot(ball.x - px, ball.y - py) for px, py in self.pockets)

    # ════════════════════════════════════════════════════════
    # Core Gymnasium interface
    # ════════════════════════════════════════════════════════
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.space = None
        self.balls = {}
        self._setup_space()
        positions = self._setup_initial_positions()

        self.color_spots = {c: positions[c] for c in self.COLOR_ORDER}
        self._create_ball(*positions['white'], 'white', self.BALL_COLORS['white'])
        for i, (x, y) in enumerate(positions['red']):
            self._create_ball(x, y, f'red_{i}', (220, 20, 20))
        for c in self.COLOR_ORDER:
            self._create_ball(*positions[c], c, self.BALL_COLORS[c])

        self.scores = [0, 0]
        self.pot_scores = [0, 0]       # points earned by potting balls
        self.foul_received = [0, 0]    # points received from opponent fouls
        self.current_player = 0
        self.current_break = 0
        self.consecutive_fouls = [0, 0]
        self.shots_without_pocket = 0
        self.phase = 'red'
        self.next_color_index = 0
        self.ball_in_hand = True       # first shot = ball-in-hand
        self.first_contact = None
        self.white_hit_any = False
        self.last_pocketed_balls = []

        if self.render_mode == 'human':
            self._init_pygame()

        return self._get_obs(), {}

    def step(self, action):
        """
        action: np.array of 5 floats in [-1, 1]
            [place_x, place_y, target_idx_raw, angle_offset, power]
        """
        place_x, place_y, target_raw, angle_offset, power_raw = action

        # Reset collision tracking
        self.first_contact = None
        self.white_hit_any = False

        # ── 1. Ball placement ────────────────────────────────
        if self.ball_in_hand:
            px, py = self._place_cue_ball_in_d(place_x, place_y)
            wb = self.balls['white']
            wb['body'].position = (px, py)
            wb['body'].velocity = (0, 0)
            wb['ball'].x = px
            wb['ball'].y = py
            wb['ball'].pocketed = False
            self.ball_in_hand = False

        # ── 2. Target selection ──────────────────────────────
        legal_targets = self._get_legal_targets()
        all_targets = self._get_all_targetable()

        if len(all_targets) == 0:
            # No balls left — game over
            obs = self._get_obs()
            return obs, 0, True, False, self._make_info()

        # Map continuous target_raw to an index over all_targets
        idx = int((target_raw + 1) / 2 * len(all_targets))
        idx = np.clip(idx, 0, len(all_targets) - 1)
        chosen_target = all_targets[idx]
        chose_legal = chosen_target in legal_targets

        # ── 3. Compute shot direction ────────────────────────
        wx = self.balls['white']['ball'].x
        wy = self.balls['white']['ball'].y
        tx = self.balls[chosen_target]['ball'].x
        ty = self.balls[chosen_target]['ball'].y

        base_angle = np.arctan2(ty - wy, tx - wx)
        # angle_offset in [-1,1] → ±15° (±π/12)
        offset_rad = angle_offset * (np.pi / 12)
        shot_angle = base_angle + offset_rad

        # power in [-1,1] → [0.1, 1.0]
        power = np.clip((power_raw + 1) / 2, 0.1, 1.0)

        # Store offset for reward shaping
        self._shot_offset_rad = offset_rad

        # ── 4. Snapshot distances before shot ────────────────
        pre_dists = {k: self._min_dist_to_pocket(v['ball'])
                     for k, v in self.balls.items()
                     if k != 'white' and not v['ball'].pocketed}

        # Snapshot all ball positions (for disturbance penalty)
        pre_positions = {}
        for k, v in self.balls.items():
            if not v['ball'].pocketed:
                pre_positions[k] = (v['ball'].x, v['ball'].y)

        # ── 5. Apply shot & simulate ─────────────────────────
        # power in [0.1, 1.0] → velocity magnitude
        # soft = 120, hard = 1800  (was 200-2200, too fast overall)
        vel_mag = power * 1680 + 120
        vx = np.cos(shot_angle) * vel_mag
        vy = np.sin(shot_angle) * vel_mag

        # Store shot metadata for rendering overlay
        self._last_shot_info = {
            'white_pos': (wx, wy),
            'target_pos': (tx, ty),
            'target_name': chosen_target,
            'chose_legal': chose_legal,
            'shot_angle': shot_angle,
            'power': power,
            'player': self.current_player,
        }

        # ── Draw aiming line before shot (if rendering) ──────
        if self.render_mode == 'human' and self.screen is not None:
            self._draw_shot_overlay(wx, wy, tx, ty, shot_angle,
                                    power, chosen_target, chose_legal)
            pygame.display.flip()
            pygame.time.wait(400)  # pause so user can see the aim

        self.balls['white']['body'].velocity = (vx, vy)

        self._simulate_until_stable()

        # ── 6. Snapshot distances after shot ──────────────────
        post_dists = {k: self._min_dist_to_pocket(v['ball'])
                      for k, v in self.balls.items()
                      if k != 'white' and not v['ball'].pocketed}

        # ── 6b. Compute ball disturbance ──────────────────────
        #   Count how many non-target, non-white balls moved significantly
        #   and total displacement of disturbed balls.
        disturbed_count = 0
        disturbed_displacement = 0.0
        move_threshold = self.ball_radius * 2  # moved more than 1 ball diameter
        for k, (px, py) in pre_positions.items():
            if k == 'white' or k == chosen_target:
                continue   # white and target are expected to move
            if k in self.balls and not self.balls[k]['ball'].pocketed:
                bx, by = self.balls[k]['ball'].x, self.balls[k]['ball'].y
                dist_moved = np.hypot(bx - px, by - py)
                if dist_moved > move_threshold:
                    disturbed_count += 1
                    disturbed_displacement += dist_moved

        # ── 7. Compute reward & handle turn switching ────────
        reward, done = self._compute_reward(
            chose_legal, chosen_target, pre_dists, post_dists,
            power, disturbed_count, disturbed_displacement)

        obs = self._get_obs()
        info = self._make_info()
        return obs, reward, done, False, info

    def _make_info(self):
        return {
            'break': self.current_break,
            'score_p1': self.scores[0],
            'score_p2': self.scores[1],
            'pot_p1': self.pot_scores[0],
            'pot_p2': self.pot_scores[1],
            'foul_rcv_p1': self.foul_received[0],
            'foul_rcv_p2': self.foul_received[1],
            'foul': self.consecutive_fouls[self.current_player],
            'pocketed': self.last_pocketed_balls,
            'phase': self.phase,
            'player': self.current_player,
        }

    # ════════════════════════════════════════════════════════
    # Reward & turn logic
    # ════════════════════════════════════════════════════════
    def _compute_reward(self, chose_legal, chosen_target,
                        pre_dists, post_dists,
                        power, disturbed_count, disturbed_displacement):
        reward = 0.0
        done = False
        is_foul = False
        pocketed = self.last_pocketed_balls.copy()
        white_pocketed = 'white' in pocketed
        cp = self.current_player

        # ── Foul detection ───────────────────────────────────
        if white_pocketed:
            is_foul = True
        if not self.white_hit_any:
            is_foul = True
        if not chose_legal:
            is_foul = True
        # Wrong first contact (even if chose_legal, the physics
        # ball that white actually touched first might differ)
        if self.first_contact is not None and not is_foul:
            if self.phase == 'red':
                if not self.first_contact.startswith('red_'):
                    is_foul = True
            elif self.phase == 'color':
                if self.first_contact.startswith('red_'):
                    is_foul = True
            elif self.phase == 'final_colors':
                tgt = (self.COLOR_ORDER[self.next_color_index]
                       if self.next_color_index < 6 else None)
                if tgt and self.first_contact != tgt:
                    is_foul = True

        # ── Combined: target selection × angle accuracy × contact ──
        #
        # Reward is layered so the agent gets progressively denser
        # signal as it learns each sub-skill:
        #
        #   Layer 0 – Did you pick a LEGAL ball?
        #   Layer 1 – Is your offset angle small (aimed well)?
        #   Layer 2 – Did white actually HIT the chosen target?
        #   Layer 3 – Did white hit the chosen target FIRST?
        #
        offset_abs = abs(getattr(self, '_shot_offset_rad', 0.0))
        max_offset = np.pi / 12          # ±15°
        accuracy = 1.0 - offset_abs / max_offset   # 1.0=perfect, 0.0=max offset

        hit_chosen = (self.first_contact == chosen_target)

        if not chose_legal:
            # ── Picked an illegal ball → strong penalty ──────
            reward -= 1.5
        else:
            # ── Layer 0: legal target chosen ─────────────────
            reward += 0.2

            # ── Layer 1: angle accuracy (regardless of outcome)
            #     offset=0 → +0.4, offset=max → 0
            reward += 0.4 * accuracy

            # ── Layer 2: actually contacted the chosen target
            if hit_chosen:
                reward += 0.5                         # hit the ball you aimed at
                reward += 0.3 * accuracy              # bonus for precise hit
            elif self.white_hit_any:
                # Hit something, but not the one you chose
                reward += 0.05                        # at least you made contact
            else:
                # Missed everything
                reward -= 1.0

        # ── Non-legal contact (hit wrong ball first) → extra penalty
        if not chose_legal or not self.white_hit_any:
            pass   # already penalised above
        elif self.white_hit_any and not hit_chosen and not is_foul:
            # Chose legal, but physics first-contact was a different ball
            reward -= 0.3

        # ── Process foul ─────────────────────────────────────
        if is_foul:
            self.consecutive_fouls[cp] += 1
            self.current_break = 0

            if white_pocketed:
                reward -= 4.0
            else:
                reward -= 1.5

            # Give opponent minimum 4 points (simplified)
            opp = 1 - cp
            foul_points = 4
            if self.first_contact is not None:
                fc = self.first_contact
                if fc in self.COLOR_VALUES:
                    foul_points = max(4, self.COLOR_VALUES[fc])
            self.scores[opp] += foul_points
            self.foul_received[opp] += foul_points

            # Re-spot colours pocketed during foul
            for bt in pocketed:
                if bt in self.COLOR_VALUES:
                    self._respot_color(bt)

            # Ball-in-hand for opponent
            if white_pocketed:
                self.ball_in_hand = True

            # Switch turn
            self._switch_player()

        else:
            # ── Legal shot ───────────────────────────────────
            self.consecutive_fouls[cp] = 0
            reds_potted = [b for b in pocketed if b.startswith('red_')]
            colors_potted = [b for b in pocketed if b in self.COLOR_VALUES]
            remaining_reds = self._count_remaining_reds()

            potted_any = len(reds_potted) + len(colors_potted) > 0

            if self.phase == 'red':
                for _ in reds_potted:
                    pts = 1
                    reward += 5.0
                    self.current_break += pts
                    self.scores[cp] += pts
                    self.pot_scores[cp] += pts
                for c in colors_potted:
                    self._respot_color(c)
                if len(reds_potted) > 0:
                    self.phase = 'color'
                    self.shots_without_pocket = 0
                else:
                    self.shots_without_pocket += 1

            elif self.phase == 'color':
                for c in colors_potted:
                    v = self.COLOR_VALUES[c]
                    reward += v * 1.5
                    self.current_break += v
                    self.scores[cp] += v
                    self.pot_scores[cp] += v
                    if remaining_reds > 0:
                        self._respot_color(c)
                if len(colors_potted) > 0:
                    self.shots_without_pocket = 0
                    if remaining_reds > 0:
                        self.phase = 'red'
                    else:
                        self._enter_final_colors()
                else:
                    self.shots_without_pocket += 1
                    if remaining_reds > 0:
                        self.phase = 'red'
                    else:
                        self._enter_final_colors()

            elif self.phase == 'final_colors':
                for c in colors_potted:
                    tgt = (self.COLOR_ORDER[self.next_color_index]
                           if self.next_color_index < 6 else None)
                    if c == tgt:
                        v = self.COLOR_VALUES[c]
                        reward += v * 2.0
                        self.current_break += v
                        self.scores[cp] += v
                        self.pot_scores[cp] += v
                        self.next_color_index += 1
                        self.shots_without_pocket = 0
                    else:
                        self._respot_color(c)
                if len(colors_potted) == 0:
                    self.shots_without_pocket += 1

            # ── Approach shaping (Layer 3) ────────────────────
            #   Only reward approach for the CHOSEN target, not all balls.
            #   This discourages chaotic multi-ball scattering.
            table_diag = np.hypot(self.table_width, self.table_height)
            if chosen_target in pre_dists and chosen_target in post_dists:
                approach = (pre_dists[chosen_target] - post_dists[chosen_target]) / table_diag
                reward += np.clip(approach * 5.0, -0.3, 1.5)

            # ── Disturbance penalty (encourage "playing tight") ──
            #   Penalise disturbing many non-target balls.
            #   In real snooker, a good shot only moves white + target (+maybe 1).
            if disturbed_count > 1:
                # mild penalty per extra disturbed ball beyond 1
                extra = disturbed_count - 1
                reward -= 0.15 * extra
            # Penalise large total displacement of bystander balls
            norm_disp = disturbed_displacement / table_diag
            reward -= np.clip(norm_disp * 0.8, 0.0, 1.0)

            # ── Power control bonus ──────────────────────────────
            #   Reward using lower power when appropriate (< 50%).
            #   Potted a ball with soft shot → extra bonus.
            if potted_any and power < 0.5:
                reward += 0.3 * (1.0 - power)   # max +0.3 at power=0

            # ── Turn switching: if nothing potted, switch ─────
            if not potted_any:
                self.current_break = 0
                self._switch_player()

        # ── Step penalty ──────────────────────────────────────
        reward -= 0.02

        # ── Terminal conditions ───────────────────────────────
        if self.phase == 'final_colors' and self.next_color_index >= 6:
            reward += 30
            done = True

        remaining_all = sum(1 for b in self.balls.values()
                            if not b['ball'].pocketed
                            and b['ball'].ball_type != 'white')
        if remaining_all == 0:
            reward += 30
            done = True

        if self.shots_without_pocket >= self.max_shots_without_pocket:
            done = True

        if self.consecutive_fouls[cp] >= 3:
            # 3 consecutive fouls → concede frame
            opp = 1 - cp
            self.scores[opp] += 7  # penalty points
            self.foul_received[opp] += 7
            reward -= 5
            done = True

        return reward, done

    def _enter_final_colors(self):
        self.phase = 'final_colors'
        self.next_color_index = 0
        while (self.next_color_index < 6 and
               self.balls[self.COLOR_ORDER[self.next_color_index]]['ball'].pocketed):
            self.next_color_index += 1

    def _switch_player(self):
        """Switch active player; the opponent gets ball-in-hand if there's a
        pending ball_in_hand flag, otherwise the ball stays where it is."""
        self.current_player = 1 - self.current_player
        self.current_break = 0
        # ball_in_hand is set separately on white-pocketed fouls

    # ════════════════════════════════════════════════════════
    # Rendering
    # ════════════════════════════════════════════════════════
    def _init_pygame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.table_width, self.table_height + 80))  # extra 80px for info bar
            pygame.display.set_caption("Snooker RL – Self-Play")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 24)
            self._last_shot_info = None

    def _draw_shot_overlay(self, wx, wy, tx, ty, shot_angle,
                           power, target_name, chose_legal):
        """Draw aiming line, target highlight, and power bar before the shot."""
        # Re-draw the table first
        self._render_table()

        # Highlight target ball with a ring
        ring_color = (0, 255, 0) if chose_legal else (255, 0, 0)
        pygame.draw.circle(self.screen, ring_color,
                           (int(tx), int(ty)), self.ball_radius + 5, 3)

        # Draw aim line from white ball in the shot direction
        line_len = 150
        end_x = wx + np.cos(shot_angle) * line_len
        end_y = wy + np.sin(shot_angle) * line_len
        pygame.draw.line(self.screen, (255, 255, 100),
                         (int(wx), int(wy)), (int(end_x), int(end_y)), 2)

        # Draw white→target dotted line
        pygame.draw.line(self.screen, (200, 200, 200),
                         (int(wx), int(wy)), (int(tx), int(ty)), 1)

        # Power bar (bottom-left)
        bar_x, bar_y = 20, self.table_height + 10
        bar_w, bar_h = 200, 16
        pygame.draw.rect(self.screen, (80, 80, 80),
                         (bar_x, bar_y, bar_w, bar_h))
        fill_w = int(bar_w * power)
        bar_color = (0, 200, 0) if power < 0.5 else (255, 165, 0) if power < 0.8 else (255, 50, 50)
        pygame.draw.rect(self.screen, bar_color,
                         (bar_x, bar_y, fill_w, bar_h))
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (bar_x, bar_y, bar_w, bar_h), 1)
        pwr_txt = self.font_small.render(f"Power: {power:.0%}", True, (255, 255, 255))
        self.screen.blit(pwr_txt, (bar_x + bar_w + 10, bar_y - 2))

        # Target label
        label_color = (0, 255, 0) if chose_legal else (255, 80, 80)
        label = f"Target: {target_name}" + ("" if chose_legal else " (ILLEGAL)")
        lbl_surf = self.font_small.render(label, True, label_color)
        self.screen.blit(lbl_surf, (bar_x + bar_w + 130, bar_y - 2))

    def render(self):
        if self.render_mode == 'human':
            return self._render_human()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()

    def _render_table(self):
        """Draw table, cushions, D-zone, pockets, balls, HUD."""
        if self.screen is None:
            self._init_pygame()

        # Background
        self.screen.fill((34, 139, 34))
        # Info bar background (below table)
        pygame.draw.rect(self.screen, (30, 30, 30),
                         (0, self.table_height, self.table_width, 80))

        # Cushions
        cc = (0, 100, 0)
        cw = self.cushion_width
        pygame.draw.rect(self.screen, cc, (0, 0, self.table_width, cw))
        pygame.draw.rect(self.screen, cc,
                         (0, self.table_height - cw, self.table_width, cw))
        pygame.draw.rect(self.screen, cc, (0, 0, cw, self.table_height))
        pygame.draw.rect(self.screen, cc,
                         (self.table_width - cw, 0, cw, self.table_height))

        # D-zone arc
        d_rect = pygame.Rect(
            int(self.baulk_x - self.d_radius),
            int(self.d_centre_y - self.d_radius),
            int(self.d_radius * 2),
            int(self.d_radius * 2))
        pygame.draw.arc(self.screen, (200, 200, 200), d_rect,
                        -np.pi / 2, np.pi / 2, 1)
        # Baulk line
        pygame.draw.line(self.screen, (200, 200, 200),
                         (int(self.baulk_x), cw),
                         (int(self.baulk_x), self.table_height - cw), 1)

        # Pockets
        for px, py in self.pockets:
            pygame.draw.circle(self.screen, (0, 0, 0),
                               (int(px), int(py)), self.pocket_radius)

        # Balls
        for bd in self.balls.values():
            b = bd['ball']
            if not b.pocketed:
                pygame.draw.circle(self.screen, b.color,
                                   (int(b.x), int(b.y)), self.ball_radius)
                pygame.draw.circle(self.screen, (255, 255, 255),
                                   (int(b.x), int(b.y)), self.ball_radius, 1)

        # HUD on table
        white = (255, 255, 255)
        yel = (255, 255, 0)
        lines = [
            f"Player {self.current_player + 1}'s turn  |  Phase: {self.phase}",
            f"P1: {self.scores[0]}(Pot:{self.pot_scores[0]} Foul:{self.foul_received[0]})  "
            f"P2: {self.scores[1]}(Pot:{self.pot_scores[1]} Foul:{self.foul_received[1]})  "
            f"Break: {self.current_break}",
            f"Ball-in-hand: {'Yes' if self.ball_in_hand else 'No'}",
        ]
        for i, txt in enumerate(lines):
            surf = self.font.render(txt, True, yel if i == 0 else white)
            self.screen.blit(surf, (10, 8 + i * 28))

        # Info bar: last shot details
        si = getattr(self, '_last_shot_info', None)
        if si:
            info_txt = (f"Last shot by P{si['player']+1}  →  {si['target_name']}"
                        f"  {'✓ Legal' if si['chose_legal'] else '✗ Illegal'}"
                        f"  |  Power: {si['power']:.0%}"
                        f"  |  Angle offset: {np.degrees(si['shot_angle']):.1f}°")
            info_color = (180, 255, 180) if si['chose_legal'] else (255, 160, 160)
            info_surf = self.font_small.render(info_txt, True, info_color)
            self.screen.blit(info_surf, (10, self.table_height + 35))

    def _render_human(self):
        if self.screen is None:
            self._init_pygame()
        self._render_table()
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])

    def _render_rgb_array(self):
        if self.screen is None:
            self._init_pygame()
        self._render_table()
        pygame.display.flip()
        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

    def get_available_balls(self):
        return [k for k, v in self.balls.items() if not v['ball'].pocketed]
