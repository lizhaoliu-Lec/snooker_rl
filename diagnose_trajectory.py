"""
Round 17 单局轨迹诊断 — 逐步打印决策详情和 reward 判断
不使用 pygame 渲染，纯终端输出。
"""
import os, sys, numpy as np, torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.pooltool_env import SnookerEnv, COLOR_ORDER

MODEL_PATH = "experiments/round24_approach/sac_snooker_ep2500.pt"


def load_agent(model_path, state_dim, action_dim):
    from algorithms.sac import SAC
    agent = SAC(state_dim, action_dim)
    agent.load(model_path)
    return agent


def describe_ball(bid):
    """Human-readable ball name."""
    if bid.startswith("red_"):
        return f"Red{int(bid.split('_')[1])}"
    return bid.capitalize()


def run_diagnosis():
    env = SnookerEnv(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = load_agent(MODEL_PATH, state_dim, action_dim)
    # Use deterministic actions
    _orig = agent.select_action
    agent.select_action = lambda s: _orig(s, deterministic=True)

    state, _ = env.reset()

    print("=" * 80)
    print("Round 18 — 单局轨迹诊断")
    print("=" * 80)
    print(f"Phase: {env.phase} | Player: P{env.current_player+1}")
    print(f"Legal targets: {env._get_legal_targets()[:8]}... ({len(env._get_legal_targets())} total)")
    print()

    total_reward = 0
    step = 0
    fouls_by_type = {'miss_ball': 0, 'wrong_ball': 0, 'white_pocket': 0, 'none': 0}
    intentional_pots = 0
    lucky_pots = 0

    while True:
        step += 1
        action, _, _ = agent.select_action(state)

        # Pre-step info
        phase_before = env.phase
        player_before = env.current_player
        legal_targets = env._get_legal_targets()

        # Decode action (same as env.step does)
        place_x, place_y, target_raw, pocket_raw = action[:4]
        angle_offset_raw, power_raw, b_spin_raw, a_spin_raw = action[4:]

        # Target mapping
        n_legal = len(legal_targets)
        idx = int((target_raw + 1) / 2 * n_legal)
        idx = np.clip(idx, 0, n_legal - 1)
        chosen_target = legal_targets[idx] if n_legal > 0 else "???"

        # Pocket mapping
        n_pockets = len(env.pocket_ids)
        p_idx = int((pocket_raw + 1) / 2 * n_pockets)
        p_idx = np.clip(p_idx, 0, n_pockets - 1)
        chosen_pocket = env.pocket_ids[p_idx]

        # Angle & power (decoded same as env)
        V0 = float(np.clip((power_raw + 1) / 2 * 5.5 + 0.5, 0.5, 6.0))
        b_spin = float(b_spin_raw * 0.8)
        a_spin = float(a_spin_raw * 0.5)

        # White & target positions
        wxy = env._ball_xy("white")
        txy = env._ball_xy(chosen_target) if chosen_target != "???" else None

        # Angle: base_angle ± 15°
        if wxy and txy:
            ideal_angle = np.degrees(np.arctan2(txy[1]-wxy[1], txy[0]-wxy[0]))
            offset_deg = angle_offset_raw * 15.0
            phi = ideal_angle + offset_deg
            angle_diff = offset_deg  # offset IS the diff from ideal
        else:
            phi = 0
            angle_diff = 999

        # Distance white → target
        dist_wt = np.hypot(txy[0]-wxy[0], txy[1]-wxy[1]) if (wxy and txy) else -1

        # Distance target → pocket
        if txy and chosen_pocket in env.pockets:
            px, py = env.pockets[chosen_pocket]
            dist_tp = np.hypot(txy[0]-px, txy[1]-py)
        else:
            dist_tp = -1

        # Execute step
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Post-step analysis
        bd = info.get('reward_breakdown', {})
        foul_type = bd.get('foul_type', None)
        pot_type = bd.get('pot_type', None)
        pocketed = info.get('pocketed', [])
        pocketed_into = info.get('pocketed_into', {})

        if foul_type:
            fouls_by_type[foul_type] = fouls_by_type.get(foul_type, 0) + 1
        else:
            fouls_by_type['none'] += 1

        if pot_type == 'intentional':
            intentional_pots += 1
        elif pot_type == 'lucky':
            lucky_pots += 1

        # Print
        foul_str = f"FOUL({foul_type})" if foul_type else "OK"
        pot_str = ""
        if pocketed:
            pot_details = [f"{b}→{pocketed_into.get(b,'?')}" for b in pocketed if b != 'white']
            pot_str = f"POTTED: [{', '.join(pot_details)}]"
            if pot_type == 'intentional':
                pot_str += " ★PRECISE"
            elif pot_type == 'lucky':
                pot_str += " (lucky)"

        print(f"Step {step:>3} | P{player_before+1} | {phase_before:>12} | "
              f"Target: {describe_ball(chosen_target):>10} "
              f"(#{idx}/{n_legal}) | "
              f"Pocket: {chosen_pocket} | "
              f"phi={phi:5.0f}° diff={angle_diff:+6.1f}° V0={V0:.1f} | "
              f"d(W→T)={dist_wt:.2f}m | "
              f"R={reward:+6.1f} | {foul_str} {pot_str}")

        # Detailed foul explanation
        if foul_type:
            if foul_type == 'miss_ball':
                print(f"       └─ 空杆: 白球没碰到任何球 (penalty={bd.get('foul',0)})")
            elif foul_type == 'wrong_ball':
                fc = env._first_contact if hasattr(env, '_first_contact') else '?'
                print(f"       └─ 碰错球: first_contact={fc}, "
                      f"phase={phase_before}, target={chosen_target} "
                      f"(penalty={bd.get('foul',0)})")
            elif foul_type == 'white_pocket':
                print(f"       └─ 白球进袋 (penalty={bd.get('foul',0)})")

        if done or truncated:
            break

        state = next_state

    print()
    print("=" * 80)
    print(f"Episode ended after {step} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final scores: P1={info.get('score_p1',0)} P2={info.get('score_p2',0)}")
    print(f"Final phase: {info.get('phase','?')}")
    print()
    print("Foul breakdown:")
    for k, v in sorted(fouls_by_type.items(), key=lambda x: -x[1]):
        pct = v / step * 100
        print(f"  {k:>15}: {v:>3} ({pct:.0f}%)")
    print()
    print(f"Pot quality:")
    print(f"  Intentional (precise): {intentional_pots}")
    print(f"  Lucky (accidental):    {lucky_pots}")
    print("=" * 80)


if __name__ == '__main__':
    run_diagnosis()
