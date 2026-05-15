"""
Snooker Environment — pooltool physics backend

Full spin physics (topspin / backspin / sidespin), event-driven collision,
real-world Snooker table dimensions (12 ft = 3.545 × 1.746 m).

Action space (8-dim continuous, all in [-1, 1]):
  [0] place_x      – D-zone cue ball x (only used when ball_in_hand)
  [1] place_y      – D-zone cue ball y (only used when ball_in_hand)
  [2] target_idx   – mapped to discrete ball index (action masking)
  [3] pocket_idx   – mapped to one of 6 pockets (target pocket for reward)
  [4] shot_angle   – offset from white→target direction (±15°, Phase 1)
  [5] power        – shot speed V0 mapped to [0.5, 6.0] m/s
  [6] b_spin       – topspin (+) / backspin (-) mapped to [-0.8, +0.8]
  [7] a_spin       – sidespin left(-) / right(+) mapped to [-0.5, +0.5]

Observation (75-dim, all normalised to ~[-1, 1]):
  white ball (2) + 15 reds (30) + 6 colours (12) + game state (10)
  + clearance (21: line-of-sight from white to each ball)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pooltool as pt
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════
# Reward Configuration (Round 19 — 精确进球奖励)
# ══════════════════════════════════════════════════════════════
#
# 两大块独立可调：
#
# 【Behavior reward（过程奖励）】每一杆的即时反馈
#   合法 + 精确进球 → pot_reward + break_bonus × current_break
#   合法 + 运气进球 → lucky_pot_reward (默认 0，不鼓励)
#   合法 + 没进 → miss_penalty（小负值，消除安全港）
#   犯规         → miss_ball_penalty / foul_penalty / white_pocket_penalty
#
# 【Outcome reward（终局奖励）】对局结束时的胜负信号
#   赢了 → +win_reward
#   输了 → +lose_reward
#   平局 → 0
#
# 精确进球条件 (三个必须同时满足):
#   1. first_contact == chosen_target (白球先碰到了选定目标球)
#   2. chosen_target in pocketed (目标球进了袋)
#   3. target 进入了 chosen_pocket (进的是选定的袋口)
#
# 设计要点:
#   1. pot_reward=20 >> |miss_penalty|=0.1，精确进球期望远超安全港
#   2. lucky_pot_reward=0 消除"大力出奇迹"的 hack
#   3. break_bonus 让连续精确进球奖励递增
#   4. win/loss ±30 给长期目标，不被 hack
#   5. max_shots_without_pocket=20 缩短无意义对推
# ══════════════════════════════════════════════════════════════

@dataclass
class RewardConfig:
    """奖励参数 — Behavior + Outcome 双轨制"""

    # ══ Behavior Reward（过程）══════════════════════════════════

    # ── 正向：进球奖励 ────────────────────────────────────────
    pot_reward: float = 20.0           # 精确进球的固定奖励
    break_bonus: float = 2.0           # 每 1 点 break 额外奖励
    lucky_pot_reward: float = 0.0      # 运气进球奖励（非精确，不鼓励）
    hit_target_reward: float = 2.0     # 命中目标球但没进袋（鼓励精度）
    #   精确进球条件：first_contact==chosen_target & target进了chosen_pocket
    #   奖励层级：精确进球(20) >> 命中目标球(2) >> 碰到合法球(miss -0.1) >> 犯规(-1/-2/-3)

    # ── 负向：犯规惩罚 ────────────────────────────────────────
    foul_penalty: float = -1.0         # 碰错球等一般犯规 -1
    miss_ball_penalty: float = -3.0    # 空杆（没碰到任何球）-3，比碰错球重
    white_pocket_penalty: float = -2.0 # 白球进袋 -2

    # ── 中性：合法没进 ────────────────────────────────────────
    miss_penalty: float = -0.1         # 合法碰球但没进（消除安全港）
    # 极小值：不构成惩罚，但让"无限轻碰"有代价
    # 20步没进 → 累积 -2.0，相当于一次白球进袋的代价

    # ══ Outcome Reward（终局）═════════════════════════════════

    win_reward: float = 0.0            # 暂时关闭（agent太弱时会驱动hack策略）
    lose_reward: float = 0.0           # 等 agent 学会精确进球后再开启
    # 平局 = 0

# ── Constants ─────────────────────────────────────────────────
COLOR_ORDER = ["yellow", "green", "brown", "blue", "pink", "black"]
COLOR_VALUES = {"yellow": 2, "green": 3, "brown": 4,
                "blue": 5, "pink": 6, "black": 7}
MAX_TARGET_BALLS = 21  # 15 reds + 6 colours


# ── Colour spots for respotting (normalised coords in pooltool) ──
# These match pooltool's `snooker_color_locs` in layouts.py
# Format: (norm_x, norm_y) where actual = norm * (W or L)
_COLOR_NORM = {
    "yellow": (1 / 3,     0.2),
    "green":  (2 / 3,     0.2),
    "brown":  (0.5,       0.2),
    "blue":   (0.5,       0.5),
    "pink":   (0.5,       0.75),
    "black":  (0.5,       10 / 11),
}


def _abs_color_spots(table_w, table_l):
    """Convert normalised colour spots to absolute metres."""
    return {c: (nx * table_w, ny * table_l)
            for c, (nx, ny) in _COLOR_NORM.items()}


# ── D-zone geometry ──────────────────────────────────────────
# The "D" is a semicircle centred on the baulk line.
# Baulk line y = 0.2 × table_l.  D radius ≈ 0.292 m (11½ in).
_D_RADIUS = 0.2921  # 11.5 inches in metres
_BAULK_FRAC = 0.2   # fraction of table length


class SnookerEnv(gym.Env):
    """
    Two-player Snooker environment backed by pooltool physics.

    Supports full cue-ball spin (topspin, backspin, sidespin), event-driven
    collision (no tunnelling), and real-world snooker table dimensions.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # ── Ball rendering colours (for pygame overlay) ──────────
    BALL_COLORS = {
        "white": (255, 255, 255), "yellow": (255, 255, 0),
        "green": (0, 180, 0),     "brown": (139, 69, 19),
        "blue": (30, 144, 255),   "pink": (255, 192, 203),
        "black": (30, 30, 30),
    }

    def __init__(self, render_mode=None, max_shots_without_pocket=20,
                 max_consecutive_fouls=5, reward_cfg=None):
        super().__init__()
        self.render_mode = render_mode
        self.max_shots_without_pocket = max_shots_without_pocket
        self.max_consecutive_fouls = max_consecutive_fouls
        self.rc = reward_cfg if reward_cfg is not None else RewardConfig()

        # ── Table & ball params (cached, immutable) ──────────
        self._ref_table = pt.Table.from_game_type(pt.GameType.SNOOKER)
        self._ref_params = pt.BallParams.default(pt.GameType.SNOOKER)
        self.table_w = self._ref_table.w          # ~1.746 m
        self.table_l = self._ref_table.l          # ~3.545 m
        self.ball_R = self._ref_params.R           # ~0.0262 m
        self.color_spots = _abs_color_spots(self.table_w, self.table_l)
        self.baulk_y = _BAULK_FRAC * self.table_l
        self.d_radius = _D_RADIUS
        self.table_diag = np.hypot(self.table_w, self.table_l)

        # Pocket centres (for approach-reward computation)
        self.pockets = {pid: (p.center[0], p.center[1])
                        for pid, p in self._ref_table.pockets.items()}

        # Ordered list of pocket IDs (for pocket_idx mapping)
        self.pocket_ids = sorted(self.pockets.keys())  # deterministic order

        # ── Action space: 8 continuous dims ──────────────────
        self.action_space = spaces.Box(
            low=-np.ones(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32,
        )

        # ── Observation space: 75 dims ───────────────────────
        # white(2) + 15reds(30) + 6colours(12) + game_state(10) + clearance(21)
        n_obs = 2 + 15 * 2 + 6 * 2 + 10 + 21  # = 75
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(n_obs,), dtype=np.float32)

        # ── Game state (populated in reset) ──────────────────
        self.system: pt.System | None = None
        self.scores = [0, 0]
        self.pot_scores = [0, 0]
        self.foul_received = [0, 0]
        self.current_player = 0
        self.current_break = 0
        self.consecutive_fouls = [0, 0]
        self.shots_without_pocket = 0
        self.phase = "red"
        self.next_color_index = 0
        self.ball_in_hand = True

        # Collision tracking (per step)
        self._first_contact = None
        self._white_hit_any = False
        self._last_pocketed: list[str] = []
        self._reward_breakdown: dict = {}
        self._step_count = 0

        # Pygame (lazy-init)
        self._screen = None
        self._clock = None
        self._font = None
        self._font_small = None
        self._last_shot_info: dict | None = None

    # ════════════════════════════════════════════════════════════
    # System creation helpers
    # ════════════════════════════════════════════════════════════
    def _make_ball(self, bid, x, y):
        """Create a single pooltool Ball with standard snooker params."""
        p = self._ref_params
        return pt.Ball.create(
            bid, xy=(x, y), ballset=None,
            m=p.m, R=p.R, u_s=p.u_s, u_r=p.u_r,
            u_sp_proportionality=p.u_sp_proportionality,
            u_b=p.u_b, e_b=p.e_b, e_c=p.e_c, f_c=p.f_c, g=p.g,
        )

    def _rebuild_system_from_state(self):
        """
        Rebuild a fresh pooltool System from the current ball positions.

        After a simulation, the System object carries stale internal state
        (events, non-zero velocities in transitions, etc.) that can confuse
        the next simulate() call.  Rebuilding from scratch with only the
        surviving balls' positions ensures a clean state.
        """
        table = pt.Table.from_game_type(pt.GameType.SNOOKER)
        balls = {}
        for bid, ball in self.system.balls.items():
            if ball.state.s == 4 and bid != "white":
                continue  # pocketed non-white balls are removed
            x = float(ball.state.rvw[0][0])
            y = float(ball.state.rvw[0][1])
            # If white was pocketed, place it at D-zone centre (will be
            # repositioned by _place_cue_ball_in_d on next step)
            if bid == "white" and ball.state.s == 4:
                x = self.table_w / 2
                y = self.baulk_y
            balls[bid] = self._make_ball(bid, x, y)
        cue = pt.Cue(cue_ball_id="white")
        self.system = pt.System(table=table, balls=balls, cue=cue)

    def _build_system(self):
        """Build a fresh pooltool System with standard snooker layout."""
        table = pt.Table.from_game_type(pt.GameType.SNOOKER)
        balls = pt.get_rack(pt.GameType.SNOOKER, table)
        cue = pt.Cue(cue_ball_id="white")
        self.system = pt.System(table=table, balls=balls, cue=cue)

    # ════════════════════════════════════════════════════════════
    # Observation
    # ════════════════════════════════════════════════════════════
    def _norm_pos(self, x, y):
        """Normalise pooltool metres to [-1, 1]."""
        return ((x / self.table_w) * 2 - 1,
                (y / self.table_l) * 2 - 1)

    def _ball_xy(self, bid):
        """Get (x, y) of a ball, or None if pocketed / missing."""
        if bid not in self.system.balls:
            return None
        b = self.system.balls[bid]
        if b.state.s == 4:   # pocketed
            return None
        return float(b.state.rvw[0][0]), float(b.state.rvw[0][1])

    def _is_pocketed(self, bid):
        if bid not in self.system.balls:
            return True
        return self.system.balls[bid].state.s == 4

    def _get_obs(self):
        obs = []

        # White ball
        wxy = self._ball_xy("white")
        if wxy:
            obs.extend(self._norm_pos(*wxy))
        else:
            obs.extend([0.0, 0.0])

        # 15 reds (pooltool uses red_01 .. red_15)
        for i in range(1, 16):
            bid = f"red_{i:02d}"
            xy = self._ball_xy(bid)
            if xy:
                obs.extend(self._norm_pos(*xy))
            else:
                obs.extend([-1.0, -1.0])

        # 6 colours
        for c in COLOR_ORDER:
            xy = self._ball_xy(c)
            if xy:
                obs.extend(self._norm_pos(*xy))
            else:
                obs.extend([-1.0, -1.0])

        # Game state (10 dims)
        remaining = sum(
            1 for bid, b in self.system.balls.items()
            if bid != "white" and b.state.s != 4)
        phase_val = {"red": 0.0, "color": 0.5, "final_colors": 1.0}.get(
            self.phase, 0.0)
        obs.extend([
            self.current_break / 147.0,
            phase_val,
            self.next_color_index / 6.0,
            remaining / 21.0,
            1.0 if self.ball_in_hand else 0.0,
            float(self.current_player),
            self.scores[0] / 147.0,
            self.scores[1] / 147.0,
            self.consecutive_fouls[self.current_player] / self.max_consecutive_fouls,
            self.shots_without_pocket / self.max_shots_without_pocket,
        ])

        # Clearance features (21 dims): line-of-sight from white to each ball
        # +1.0 = clear path, 0.0 = blocked, -1.0 = ball pocketed/missing
        for i in range(1, 16):
            bid = f"red_{i:02d}"
            if self._is_pocketed(bid):
                obs.append(-1.0)
            else:
                obs.append(self._line_clear(bid))  # 0.0 or 1.0
        for c in COLOR_ORDER:
            if self._is_pocketed(c):
                obs.append(-1.0)
            else:
                obs.append(self._line_clear(c))  # 0.0 or 1.0

        return np.array(obs, dtype=np.float32)

    # ════════════════════════════════════════════════════════════
    # Target selection helpers
    # ════════════════════════════════════════════════════════════
    def _get_legal_targets(self):
        """Ball IDs the current player may legally target."""
        targets = []
        if self.phase == "red":
            for i in range(1, 16):
                bid = f"red_{i:02d}"
                if not self._is_pocketed(bid):
                    targets.append(bid)
        elif self.phase == "color":
            for c in COLOR_ORDER:
                if not self._is_pocketed(c):
                    targets.append(c)
        elif self.phase == "final_colors":
            if self.next_color_index < 6:
                c = COLOR_ORDER[self.next_color_index]
                if not self._is_pocketed(c):
                    targets.append(c)
        return targets

    def _get_all_targetable(self):
        """All non-white, non-pocketed balls."""
        return [bid for bid, b in self.system.balls.items()
                if bid != "white" and b.state.s != 4]

    # ════════════════════════════════════════════════════════════
    # D-zone ball placement
    # ════════════════════════════════════════════════════════════
    def _place_cue_ball_in_d(self, nx, ny):
        """
        Place cue ball using normalised [-1, 1] coords mapped to D-zone.
        The D is a semicircle on the baulk line (y = baulk_y),
        open towards the top of the table.
        Returns (x, y) in metres.
        """
        cx = self.table_w / 2
        cy = self.baulk_y
        r = self.d_radius

        # Map [-1,1] → offset within semicircle
        dx = nx * r           # left-right
        dy = -abs(ny) * r     # only below baulk line (towards bottom cushion)

        # Clamp to semicircle
        dist = np.hypot(dx, dy)
        if dist > r:
            dx *= r / dist
            dy *= r / dist

        px = cx + dx
        py = cy + dy

        # Clamp inside table boundaries (with ball-radius margin)
        margin = self.ball_R + 0.005
        px = np.clip(px, margin, self.table_w - margin)
        py = np.clip(py, margin, self.table_l - margin)
        return float(px), float(py)

    # ════════════════════════════════════════════════════════════
    # Ball respotting
    # ════════════════════════════════════════════════════════════
    def _respot_color(self, bid):
        """Respot a colour ball to its designated spot."""
        if bid not in self.color_spots or bid not in self.system.balls:
            return
        sx, sy = self.color_spots[bid]
        ball = self.system.balls[bid]
        # Reset position and state to stationary
        ball.state.rvw[0][0] = sx
        ball.state.rvw[0][1] = sy
        ball.state.rvw[0][2] = 0.0
        ball.state.rvw[1] = [0, 0, 0]  # velocity
        ball.state.rvw[2] = [0, 0, 0]  # angular velocity
        ball.state.s = 0  # stationary

    def _respot_white(self, px=None, py=None):
        """Reset white ball to a valid position (for ball-in-hand)."""
        ball = self.system.balls["white"]
        if px is None:
            px = self.table_w / 2
            py = self.baulk_y
        ball.state.rvw[0][0] = px
        ball.state.rvw[0][1] = py
        ball.state.rvw[0][2] = 0.0
        ball.state.rvw[1] = [0, 0, 0]
        ball.state.rvw[2] = [0, 0, 0]
        ball.state.s = 0

    # ════════════════════════════════════════════════════════════
    # Event analysis (after simulation)
    # ════════════════════════════════════════════════════════════
    def _analyse_events(self, result):
        """Extract first-contact and pocketed balls from simulation events."""
        self._first_contact = None
        self._white_hit_any = False
        self._last_pocketed = []
        self._pocketed_into = {}  # ball_id → pocket_id 追踪每个球进了哪个袋
        self._closest_approach = 999.0  # white ball closest distance to chosen_target

        for event in result.events:
            etype = event.event_type.value  # e.g. 'ball_ball', 'ball_pocket'

            # Ball-ball collision
            if etype == "ball_ball":
                ids = event.ids
                if "white" in ids:
                    self._white_hit_any = True
                    other = [x for x in ids if x != "white"]
                    if other and self._first_contact is None:
                        self._first_contact = other[0]

            # Ball pocketed
            elif etype == "ball_pocket":
                bid = event.ids[0]   # ball_id
                pid = event.ids[1]   # pocket_id
                if bid not in self._last_pocketed:
                    self._last_pocketed.append(bid)
                    self._pocketed_into[bid] = pid

    # ════════════════════════════════════════════════════════════
    # Pocket distance helpers
    # ════════════════════════════════════════════════════════════
    def _min_dist_to_pocket(self, bid):
        """Min distance from ball to any pocket centre."""
        xy = self._ball_xy(bid)
        if xy is None:
            return 999.0
        return min(np.hypot(xy[0] - px, xy[1] - py)
                   for px, py in self.pockets.values())

    def _dist_to_pocket(self, bid, pocket_id):
        """Distance from ball to a specific pocket centre."""
        xy = self._ball_xy(bid)
        if xy is None:
            return 999.0
        px, py = self.pockets[pocket_id]
        return float(np.hypot(xy[0] - px, xy[1] - py))

    def _line_clear(self, target_bid):
        """
        Check if the line from white ball to target ball is clear of obstacles.
        Returns 1.0 if clear, 0.0 if blocked.
        Uses point-to-line-segment distance: if any other ball's centre is
        within 2*R of the white→target line, it's blocked.
        """
        wxy = self._ball_xy("white")
        txy = self._ball_xy(target_bid)
        if wxy is None or txy is None:
            return 0.0

        wx, wy = wxy
        tx, ty = txy
        threshold = 2 * self.ball_R  # Two ball radii (one for each ball edge)

        dx, dy = tx - wx, ty - wy
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-8:
            return 1.0  # white and target overlap, treat as clear

        # Check all other balls
        for bid, ball in self.system.balls.items():
            if bid == "white" or bid == target_bid:
                continue
            if ball.state.s == 4:  # pocketed
                continue
            bx = float(ball.state.rvw[0][0])
            by = float(ball.state.rvw[0][1])

            # Project ball onto line segment white→target
            # t = dot(ball-white, target-white) / |target-white|^2
            t = ((bx - wx) * dx + (by - wy) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))

            # Closest point on segment
            cx = wx + t * dx
            cy = wy + t * dy

            # Distance from ball to closest point
            dist = np.hypot(bx - cx, by - cy)
            if dist < threshold:
                return 0.0  # blocked

        return 1.0  # clear

    # ════════════════════════════════════════════════════════════
    # Count helpers
    # ════════════════════════════════════════════════════════════
    def _count_remaining_reds(self):
        return sum(1 for i in range(1, 16)
                   if not self._is_pocketed(f"red_{i:02d}"))

    # ════════════════════════════════════════════════════════════
    # Core Gymnasium API
    # ════════════════════════════════════════════════════════════
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._build_system()

        self.scores = [0, 0]
        self.pot_scores = [0, 0]
        self.foul_received = [0, 0]
        self.current_player = 0
        self.current_break = 0
        self.consecutive_fouls = [0, 0]
        self.shots_without_pocket = 0
        self.phase = "red"
        self.next_color_index = 0
        self.ball_in_hand = True

        self._first_contact = None
        self._white_hit_any = False
        self._last_pocketed = []
        self._pocketed_into = {}
        self._closest_approach = 999.0
        self._last_shot_info = None
        self._reward_breakdown = {}
        self._step_count = 0

        if self.render_mode == "human":
            self._init_pygame()

        return self._get_obs(), {}

    def step(self, action):
        """
        action: np.array of 8 floats in [-1, 1]
            [place_x, place_y, target_raw, pocket_raw, angle_offset,
             power_raw, b_spin_raw, a_spin_raw]
        """
        (place_x, place_y, target_raw, pocket_raw, angle_offset,
         power_raw, b_spin_raw, a_spin_raw) = action

        # ── 1. Ball placement ────────────────────────────────
        if self.ball_in_hand:
            px, py = self._place_cue_ball_in_d(place_x, place_y)
            self._respot_white(px, py)
            self.ball_in_hand = False

        # ── 2. Target selection (Action Masking) ───────────────
        # Agent 只能从合法目标中选球（强制规则约束）。
        # "Red phase 打红球" 不是需要学的策略，是规则前提。
        # Agent 只需学习"选哪颗合法球最好打"。
        legal_targets = self._get_legal_targets()
        if len(legal_targets) == 0:
            # Fallback: 如果没有合法球（理论上不该发生），用所有球
            legal_targets = self._get_all_targetable()
        if len(legal_targets) == 0:
            return self._get_obs(), 0.0, True, False, self._make_info()

        idx = int((target_raw + 1) / 2 * len(legal_targets))
        idx = np.clip(idx, 0, len(legal_targets) - 1)
        chosen_target = legal_targets[idx]
        chose_legal = True  # Action masking 保证永远合法

        # ── 3. Target pocket selection ───────────────────────
        n_pockets = len(self.pocket_ids)
        p_idx = int((pocket_raw + 1) / 2 * n_pockets)
        p_idx = np.clip(p_idx, 0, n_pockets - 1)
        chosen_pocket = self.pocket_ids[p_idx]

        # ── 4. Compute shot parameters ──────────────────────
        wxy = self._ball_xy("white")
        txy = self._ball_xy(chosen_target)
        if wxy is None or txy is None:
            # Shouldn't happen, but bail gracefully
            return self._get_obs(), -1.0, True, False, self._make_info()

        # Shot direction: base_angle ± 15° (Phase 1 curriculum)
        base_angle = np.degrees(np.arctan2(txy[1] - wxy[1], txy[0] - wxy[0]))
        offset_deg = angle_offset * 15.0  # ±15°
        phi = base_angle + offset_deg

        # Power: [-1,1] → [0.5, 6.0] m/s
        V0 = float(np.clip((power_raw + 1) / 2 * 5.5 + 0.5, 0.5, 6.0))

        # Spin: b in [-0.8, +0.8], a in [-0.5, +0.5]
        b_spin = float(np.clip(b_spin_raw * 0.8, -0.8, 0.8))
        a_spin = float(np.clip(a_spin_raw * 0.5, -0.5, 0.5))

        # ── 5. Rebuild a clean System & execute shot ─────────
        # Rebuild avoids stale internal state from previous simulate()
        self._rebuild_system_from_state()
        self.system.cue.set_state(V0=V0, phi=phi, b=b_spin, a=a_spin)

        # Store shot info for rendering
        # Compute angle deviation from white→target direction (for diagnostics)
        ideal_angle = np.degrees(np.arctan2(txy[1] - wxy[1], txy[0] - wxy[0]))
        angle_diff = (phi - ideal_angle + 180) % 360 - 180  # [-180, 180]

        self._last_shot_info = {
            "white_pos": wxy,
            "target_pos": txy,
            "target_name": chosen_target,
            "chose_legal": chose_legal,
            "chosen_pocket": chosen_pocket,
            "phi": phi,
            "offset_deg": angle_diff,  # deviation from ideal direction
            "V0": V0,
            "b": b_spin,
            "a": a_spin,
            "player": self.current_player,
        }

        # Render aim before shot
        if self.render_mode == "human" and self._screen is not None:
            self._render_scene(aim_info=self._last_shot_info)
            import pygame
            pygame.time.wait(300)

        try:
            result = pt.simulate(self.system, inplace=False)
        except (AssertionError, Exception) as exc:
            # pooltool physics can rarely crash (e.g. cushion collision
            # assertion when ball is very close to rail).  Treat as foul
            # and rebuild a clean system from pre-shot state.
            self._first_contact = None
            self._white_hit_any = False
            self._last_pocketed = []
            self._pocketed_into = {}
            self._rebuild_system_from_state()
            self._step_count += 1
            bd = {'foul': self.rc.foul_penalty, 'distance': 0.0,
                  'win_loss': 0.0}
            self.consecutive_fouls[self.current_player] += 1
            self.current_break = 0
            opp = 1 - self.current_player
            self.scores[opp] += 4
            self.foul_received[opp] += 4
            self._switch_player()
            done = (self.consecutive_fouls[self.current_player]
                    >= self.max_consecutive_fouls)
            reward = sum(bd.values())
            bd['total'] = float(reward)
            bd['foul_type'] = 'physics_crash'
            self._reward_breakdown = bd
            return self._get_obs(), reward, done, False, self._make_info()

        # ── 6. Analyse events ────────────────────────────────
        self._analyse_events(result)

        # ── 6.5 Closest approach: white ball to chosen_target ─
        # Compute min distance from white trajectory to target ball
        if chosen_target in result.balls:
            white_hist = result.balls["white"].history.states
            target_hist = result.balls[chosen_target].history.states
            n_states = min(len(white_hist), len(target_hist))
            if n_states > 0:
                min_d = 999.0
                for si in range(0, n_states, max(1, n_states // 100)):
                    wx = white_hist[si].rvw[0][0]
                    wy = white_hist[si].rvw[0][1]
                    tx = target_hist[si].rvw[0][0]
                    ty = target_hist[si].rvw[0][1]
                    d = float(np.hypot(wx - tx, wy - ty))
                    if d < min_d:
                        min_d = d
                self._closest_approach = min_d

        # Update system to post-simulation state
        self.system = result

        # ── 7. Compute reward & handle turn switching ────────
        self._step_count += 1
        reward, done = self._compute_reward(
            chose_legal, chosen_target, chosen_pocket)

        # Render post-shot
        if self.render_mode == "human" and self._screen is not None:
            self._animate_shot(result)

        obs = self._get_obs()
        info = self._make_info()
        return obs, reward, done, False, info

    def _make_info(self):
        info = {
            "break": self.current_break,
            "score_p1": self.scores[0],
            "score_p2": self.scores[1],
            "pot_p1": self.pot_scores[0],
            "pot_p2": self.pot_scores[1],
            "foul_rcv_p1": self.foul_received[0],
            "foul_rcv_p2": self.foul_received[1],
            "foul": self.consecutive_fouls[self.current_player],
            "pocketed": self._last_pocketed.copy(),
            "pocketed_into": self._pocketed_into.copy(),
            "phase": self.phase,
            "player": self.current_player,
        }
        if self._last_shot_info:
            info["chosen_pocket"] = self._last_shot_info.get("chosen_pocket")
            info["chosen_target"] = self._last_shot_info.get("target_name")
            info["offset_deg"] = self._last_shot_info.get("offset_deg", 0.0)
            info["V0"] = self._last_shot_info.get("V0", 0.0)
        if self._reward_breakdown:
            info["reward_breakdown"] = self._reward_breakdown.copy()
        return info

    # ════════════════════════════════════════════════════════════
    # Reward & turn logic
    # ════════════════════════════════════════════════════════════
    def _compute_reward(self, chose_legal, chosen_target, chosen_pocket):
        """
        Round 19 奖励：Behavior + Outcome 双轨制 + 精确进球.

        Behavior（每一杆）：
        犯规 → foul_penalty(-1/-2/-3 depending on type)
        合法 + 精确进球 → pot_reward(20) + break_bonus(2) × current_break
        合法 + 运气进球 → lucky_pot_reward(0)  不鼓励大力散球
        合法 + 没进 → miss_penalty(-0.1)

        精确进球条件：
          1. first_contact == chosen_target (白球先碰到了选定目标球)
          2. chosen_target in pocketed (目标球确实进了袋)
          3. target 进入了 chosen_pocket (进的是选定袋口)

        Outcome（终局）：
        赢了 → +win_reward(30)
        输了 → +lose_reward(-30)
        平局 → 0
        """
        rc = self.rc
        bd = {}  # breakdown for debugging
        done = False
        pot_type = None  # 'intentional' | 'lucky' | None
        pocketed = self._last_pocketed.copy()
        white_pocketed = "white" in pocketed
        cp = self.current_player

        # ── Foul detection ───────────────────────────────────
        # 斯诺克规则：red phase 碰到任何红球都合法；
        #             color phase 碰到任何彩球都合法；
        #             final_colors 必须碰到指定的下一颗彩球。
        wrong_first_contact = False
        if self._first_contact is not None:
            fc = self._first_contact
            if self.phase == "red":
                if not fc.startswith("red_"):
                    wrong_first_contact = True
            elif self.phase == "color":
                if fc not in COLOR_VALUES:
                    wrong_first_contact = True
            elif self.phase == "final_colors":
                tgt = (COLOR_ORDER[self.next_color_index]
                       if self.next_color_index < 6 else None)
                if tgt and fc != tgt:
                    wrong_first_contact = True

        is_foul = (not self._white_hit_any or wrong_first_contact
                   or not chose_legal or white_pocketed)

        # ── Foul type breakdown (用于训练诊断) ────────────────
        # 记录具体是哪种犯规触发的，方便训练时追踪
        foul_type = None
        if is_foul:
            if not self._white_hit_any:
                foul_type = 'miss_ball'          # 空杆：白球没碰到任何球
            elif wrong_first_contact:
                foul_type = 'wrong_ball'         # 碰错球：先碰到了非法球
            elif not chose_legal:
                foul_type = 'illegal_choice'     # 选了非法球（action masking 后不应出现）
            elif white_pocketed:
                foul_type = 'white_pocket'       # 白球进袋

        # ════════════════════════════════════════════════════════
        # FOUL PATH
        # ════════════════════════════════════════════════════════
        if is_foul:
            self.consecutive_fouls[cp] += 1
            self.current_break = 0

            if white_pocketed:
                bd['foul'] = rc.white_pocket_penalty    # -2（白球进袋）
            elif not self._white_hit_any:
                bd['foul'] = rc.miss_ball_penalty        # -3（空杆，最重）
            else:
                bd['foul'] = rc.foul_penalty             # -1（碰错球等）
            bd['distance'] = 0.0

            # 对手获得罚分 (min 4)
            opp = 1 - cp
            foul_points = 4
            if self._first_contact and self._first_contact in COLOR_VALUES:
                foul_points = max(4, COLOR_VALUES[self._first_contact])
            self.scores[opp] += foul_points
            self.foul_received[opp] += foul_points

            # 犯规进的彩球要复位
            for bt in pocketed:
                if bt in COLOR_VALUES:
                    self._respot_color(bt)

            # 犯规也计入无进球步数（防止互相犯规无限对局）
            self.shots_without_pocket += 1

            if white_pocketed:
                self.ball_in_hand = True

            # 犯规后阶段切换：color 阶段犯规 → 回到 red 阶段
            # （斯诺克规则：进红球后打彩球犯规，对手重新从红球开始）
            if self.phase == "color":
                remaining_reds = self._count_remaining_reds()
                if remaining_reds > 0:
                    self.phase = "red"
                else:
                    self._enter_final_colors()

            self._switch_player()

        # ════════════════════════════════════════════════════════
        # LEGAL PATH: 合法击球
        # ════════════════════════════════════════════════════════
        else:
            self.consecutive_fouls[cp] = 0
            bd['foul'] = 0.0

            legal_pots = self._count_legal_pots(pocketed)

            if legal_pots > 0:
                # ── 进球！区分精确进球 vs 运气进球 ─────────
                break_before = self.current_break
                shots_before_pot = self.shots_without_pocket

                # 先处理得分和阶段转换（会更新 current_break）
                self._handle_pot_scoring(pocketed, cp)

                # ── 精确进球判断 ──────────────────────────
                # 三个条件同时满足才算精确进球：
                #   1. first_contact == chosen_target（白球先碰到了选定的目标球）
                #   2. chosen_target in pocketed（选定的目标球进了袋）
                #   3. 目标球进入了 chosen_pocket（进的是选定的袋口）
                is_intentional = (
                    self._first_contact == chosen_target
                    and chosen_target in pocketed
                    and self._pocketed_into.get(chosen_target) == chosen_pocket
                )

                if is_intentional:
                    # 精确进球：full reward + break bonus
                    pot_r = rc.pot_reward + rc.break_bonus * break_before
                    pot_type = 'intentional'
                else:
                    # 运气进球：合法但非精确，只给 lucky_pot_reward（默认 0）
                    pot_r = rc.lucky_pot_reward
                    pot_type = 'lucky'
                    # 运气进球不重置 shots_without_pocket（防止延长对局赚 reward）
                    self.shots_without_pocket = shots_before_pot

                bd['distance'] = float(pot_r)

            else:
                # ── 没进球：区分命中目标球 vs 碰到其他球 ──
                hit_target = (self._first_contact == chosen_target)

                if hit_target:
                    # 命中了目标球但没进袋 → 正向奖励（鼓励精度）
                    bd['distance'] = rc.hit_target_reward
                    pot_type = 'hit_target'
                else:
                    # 碰到了其他合法球 → miss_penalty（消除安全港）
                    bd['distance'] = rc.miss_penalty
                    pot_type = None

                # 没进球 → 换人
                self.current_break = 0
                self._switch_player()
                self.shots_without_pocket += 1

                # color 阶段没进球 → 回到 red 阶段
                if self.phase == "color":
                    remaining_reds = self._count_remaining_reds()
                    if remaining_reds > 0:
                        self.phase = "red"
                    else:
                        self._enter_final_colors()

        # ── Terminal conditions ──────────────────────────────
        if self.phase == "final_colors" and self.next_color_index >= 6:
            done = True

        remaining_all = sum(
            1 for bid, b in self.system.balls.items()
            if bid != "white" and b.state.s != 4)
        if remaining_all == 0:
            done = True

        if self.shots_without_pocket >= self.max_shots_without_pocket:
            done = True

        if self.consecutive_fouls[cp] >= self.max_consecutive_fouls:
            opp = 1 - cp
            self.scores[opp] += 7
            self.foul_received[opp] += 7
            done = True

        # ── Win/Loss terminal reward ────────────────────────
        # 终局时，根据当前玩家（刚出手的玩家）的得分给胜负奖励。
        # 注意：犯规 path 已经 _switch_player()，所以 cp 仍是出手者。
        # 但 self.current_player 可能已经切换了，所以用 cp。
        bd['win_loss'] = 0.0
        if done:
            my_score = self.scores[cp]
            opp_score = self.scores[1 - cp]
            if my_score > opp_score:
                bd['win_loss'] = rc.win_reward      # +30
            elif my_score < opp_score:
                bd['win_loss'] = rc.lose_reward      # -30



        # ── Sum ──────────────────────────────────────────────
        reward = sum(bd.values())
        bd['total'] = float(reward)
        bd['foul_type'] = foul_type  # None | 'miss_ball' | 'wrong_ball' | 'illegal_choice' | 'white_pocket'
        bd['pot_type'] = pot_type if not is_foul else None  # None | 'intentional' | 'lucky'
        self._reward_breakdown = bd

        return reward, done

    def _count_legal_pots(self, pocketed):
        """Count legally potted balls (excluding white) based on current phase."""
        count = 0
        if self.phase == "red":
            # Red phase: any red potted = legal; colours potted = illegal (respot)
            count = sum(1 for b in pocketed if b.startswith("red_"))
        elif self.phase == "color":
            # Color phase: any colour potted = legal
            count = sum(1 for b in pocketed if b in COLOR_VALUES)
        elif self.phase == "final_colors":
            # Must be the specific next colour
            tgt = (COLOR_ORDER[self.next_color_index]
                   if self.next_color_index < 6 else None)
            if tgt and tgt in pocketed:
                count = 1
        return count

    def _handle_pot_scoring(self, pocketed, cp):
        """Handle scoring and phase transitions when balls are potted legally."""
        reds_potted = [b for b in pocketed if b.startswith("red_")]
        colors_potted = [b for b in pocketed if b in COLOR_VALUES]
        remaining_reds = self._count_remaining_reds()

        if self.phase == "red":
            for _ in reds_potted:
                self.current_break += 1
                self.scores[cp] += 1
                self.pot_scores[cp] += 1
            # Illegally potted colours get respotted
            for c in colors_potted:
                self._respot_color(c)
            if reds_potted:
                self.phase = "color"
                self.shots_without_pocket = 0
            else:
                self.shots_without_pocket += 1

        elif self.phase == "color":
            for c in colors_potted:
                v = COLOR_VALUES[c]
                self.current_break += v
                self.scores[cp] += v
                self.pot_scores[cp] += v
                if remaining_reds > 0:
                    self._respot_color(c)
            if colors_potted:
                self.shots_without_pocket = 0
                if remaining_reds > 0:
                    self.phase = "red"
                else:
                    self._enter_final_colors()
            else:
                self.shots_without_pocket += 1
                if remaining_reds > 0:
                    self.phase = "red"
                else:
                    self._enter_final_colors()

        elif self.phase == "final_colors":
            for c in colors_potted:
                tgt = (COLOR_ORDER[self.next_color_index]
                       if self.next_color_index < 6 else None)
                if c == tgt:
                    v = COLOR_VALUES[c]
                    self.current_break += v
                    self.scores[cp] += v
                    self.pot_scores[cp] += v
                    self.next_color_index += 1
                    self.shots_without_pocket = 0
                else:
                    self._respot_color(c)
            if not colors_potted:
                self.shots_without_pocket += 1

        # 没有合法球进袋 → 换人
        potted_any = len(reds_potted) + len(colors_potted) > 0
        if not potted_any:
            self.current_break = 0
            self._switch_player()

    def _enter_final_colors(self):
        self.phase = "final_colors"
        self.next_color_index = 0
        while (self.next_color_index < 6
               and self._is_pocketed(COLOR_ORDER[self.next_color_index])):
            self.next_color_index += 1

    def _switch_player(self):
        self.current_player = 1 - self.current_player
        self.current_break = 0

    # ════════════════════════════════════════════════════════════
    # Rendering (pygame 2D)
    # ════════════════════════════════════════════════════════════
    def _init_pygame(self):
        if self._screen is not None:
            return
        import pygame
        import os as _os
        pygame.init()

        # Auto-scale to fit screen (leave 100px margin on each side)
        disp_info = pygame.display.Info()
        max_w = disp_info.current_w - 200
        max_h = disp_info.current_h - 200

        self._hud_h = 80
        self._pad = 40   # border padding

        # Calculate max scale that fits screen
        scale_w = (max_w - self._pad) / self.table_w
        scale_h = (max_h - self._pad - self._hud_h) / self.table_l
        self._scale = min(scale_w, scale_h, 400.0)  # cap at 400 px/m

        scr_w = int(self.table_w * self._scale) + self._pad
        scr_h = int(self.table_l * self._scale) + self._pad + self._hud_h
        self._ox, self._oy = self._pad // 2, self._pad // 2

        # RESIZABLE flag allows user to drag window borders
        self._screen = pygame.display.set_mode(
            (scr_w, scr_h), pygame.RESIZABLE)
        pygame.display.set_caption("Snooker RL – Pooltool")
        self._clock = pygame.time.Clock()
        # CJK font support
        _cjk_paths = [
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]
        font_path = None
        for p in _cjk_paths:
            if _os.path.exists(p):
                font_path = p
                break
        if font_path:
            self._font = pygame.font.Font(font_path, 20)
            self._font_small = pygame.font.Font(font_path, 16)
        else:
            self._font = pygame.font.SysFont("arial", 20)
            self._font_small = pygame.font.SysFont("arial", 16)

    def _to_px(self, x, y):
        """Convert pooltool metres → screen pixels."""
        px = int(x * self._scale) + self._ox
        py = int((self.table_l - y) * self._scale) + self._oy
        return px, py

    def _render_scene(self, aim_info=None):
        """Draw one frame: table + balls + HUD."""
        import pygame
        if self._screen is None:
            return

        # Handle resize events — recalculate scale from current window size
        for evt in pygame.event.get(pygame.VIDEORESIZE):
            new_w, new_h = evt.w, evt.h
            self._screen = pygame.display.set_mode(
                (new_w, new_h), pygame.RESIZABLE)
            scale_w = (new_w - self._pad) / self.table_w
            scale_h = (new_h - self._pad - self._hud_h) / self.table_l
            self._scale = min(scale_w, scale_h)
            self._ox = (new_w - int(self.table_w * self._scale)) // 2
            self._oy = self._pad // 2

        scr = self._screen
        scr.fill((40, 40, 40))

        # Table felt
        tw_px = int(self.table_w * self._scale)
        tl_px = int(self.table_l * self._scale)
        pygame.draw.rect(scr, (34, 139, 34),
                         (self._ox, self._oy, tw_px, tl_px))
        pygame.draw.rect(scr, (0, 100, 0),
                         (self._ox, self._oy, tw_px, tl_px), 4)

        # Pockets
        for pid, (pcx, pcy) in self.pockets.items():
            ppx, ppy = self._to_px(pcx, pcy)
            r = max(3, int(0.043 * self._scale))
            pygame.draw.circle(scr, (0, 0, 0), (ppx, ppy), r)

        # Baulk line & D
        baulk_px = self._to_px(0, self.baulk_y)
        baulk_px_l = self._to_px(0, self.baulk_y)
        baulk_px_r = self._to_px(self.table_w, self.baulk_y)
        pygame.draw.line(scr, (200, 200, 200),
                         baulk_px_l, baulk_px_r, 1)
        # D semicircle (opens towards top of table = towards lower screen y)
        d_r_px = int(self.d_radius * self._scale)
        cx_px, cy_px = self._to_px(self.table_w / 2, self.baulk_y)
        d_rect = pygame.Rect(cx_px - d_r_px, cy_px - d_r_px,
                             d_r_px * 2, d_r_px * 2)
        pygame.draw.arc(scr, (200, 200, 200), d_rect,
                        0, np.pi, 1)

        # Balls
        for bid, ball in self.system.balls.items():
            if ball.state.s == 4:
                continue
            bx, by = float(ball.state.rvw[0][0]), float(ball.state.rvw[0][1])
            ppx, ppy = self._to_px(bx, by)
            r = max(2, int(self.ball_R * self._scale))
            if bid == "white":
                c = (255, 255, 255)
            elif bid.startswith("red_"):
                c = (220, 20, 20)
            else:
                c = self.BALL_COLORS.get(bid, (180, 180, 180))
            pygame.draw.circle(scr, c, (ppx, ppy), r)
            pygame.draw.circle(scr, (255, 255, 255), (ppx, ppy), r, 1)

        # Aim line
        if aim_info:
            wx, wy = aim_info["white_pos"]
            tx, ty = aim_info["target_pos"]
            wp = self._to_px(wx, wy)
            tp = self._to_px(tx, ty)
            # Target ring
            ring_c = (0, 255, 0) if aim_info["chose_legal"] else (255, 0, 0)
            pygame.draw.circle(scr, ring_c, tp,
                               int(self.ball_R * self._scale) + 4, 2)
            # Highlight chosen pocket
            chosen_pid = aim_info.get("chosen_pocket")
            if chosen_pid and chosen_pid in self.pockets:
                pcx, pcy = self.pockets[chosen_pid]
                ppx, ppy = self._to_px(pcx, pcy)
                pygame.draw.circle(scr, (255, 200, 0), (ppx, ppy),
                                   int(0.043 * self._scale) + 6, 3)
            # Aim line: white → target
            pygame.draw.line(scr, (255, 255, 100), wp, tp, 2)

        # HUD
        hud_y = self._oy + tl_px + 10
        pygame.draw.rect(scr, (30, 30, 30),
                         (0, hud_y - 5, scr.get_width(), self._hud_h))
        white = (255, 255, 255)
        yel = (255, 255, 100)
        lines = [
            f"Player {self.current_player + 1} | Phase: {self.phase} "
            f"| Break: {self.current_break}",
            f"P1: {self.scores[0]}(Pot:{self.pot_scores[0]} "
            f"Rcv:{self.foul_received[0]})  "
            f"P2: {self.scores[1]}(Pot:{self.pot_scores[1]} "
            f"Rcv:{self.foul_received[1]})",
        ]
        for i, txt in enumerate(lines):
            surf = self._font.render(txt, True, yel if i == 0 else white)
            scr.blit(surf, (10, hud_y + i * 24))

        if aim_info:
            si = aim_info
            info_txt = (f"→ {si['target_name']} "
                        f"{'✓' if si['chose_legal'] else '✗'} "
                        f"pocket={si.get('chosen_pocket', '?')} "
                        f"V0={si['V0']:.1f} b={si['b']:+.2f} a={si['a']:+.2f}")
            col = (180, 255, 180) if si["chose_legal"] else (255, 160, 160)
            surf = self._font_small.render(info_txt, True, col)
            scr.blit(surf, (10, hud_y + 52))

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def _animate_shot(self, result):
        """Animate the shot result frame by frame."""
        import pygame

        try:
            cts = pt.continuize(result, dt=0.01, inplace=False)
        except Exception:
            # If continuize fails, just render final state
            self._render_scene()
            return

        total_t = cts.t
        if total_t < 0.01:
            total_t = 0.5
        n_frames = int(total_t / 0.01)

        for fi in range(0, n_frames, 2):  # skip every other frame for speed
            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    return

            # Temporarily set ball states for rendering
            for bid, ball in cts.balls.items():
                if ball.history_cts.states:
                    idx = min(fi, len(ball.history_cts.states) - 1)
                    st = ball.history_cts.states[idx]
                    self.system.balls[bid].state = st

            self._render_scene()

        # Restore final state
        for bid, ball in result.balls.items():
            self.system.balls[bid].state = ball.state
        self._render_scene()

    def render(self):
        if self.render_mode == "human":
            self._render_scene()
        elif self.render_mode == "rgb_array":
            self._init_pygame()
            self._render_scene()
            import pygame
            return pygame.surfarray.array3d(self._screen).transpose(1, 0, 2)

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None

    def get_available_balls(self):
        return [bid for bid, b in self.system.balls.items()
                if b.state.s != 4]
