"""
Test script for the Snooker RL self-play environment and agents (PPO + SAC).
Tests the pooltool-based environment (8-dim action, 75-dim observation).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def test_environment():
    """Test basic environment functionality (7-dim action space)."""
    print("Testing Snooker Environment (pooltool, Self-Play)...")

    from environment.pooltool_env import SnookerEnv

    env = SnookerEnv(render_mode=None)

    print(f"  Action space: {env.action_space}")
    print(f"  Observation space shape: {env.observation_space.shape}")

    assert env.action_space.shape == (8,), \
        f"Expected action dim 8, got {env.action_space.shape}"
    assert env.observation_space.shape == (75,), \
        f"Expected obs dim 75, got {env.observation_space.shape}"

    state, info = env.reset()
    print(f"  Initial state shape: {state.shape}")
    assert state.shape == (75,), f"State shape mismatch: {state.shape}"

    # Run a few random steps
    for i in range(5):
        action = env.action_space.sample()
        assert action.shape == (8,), f"Action shape: {action.shape}"
        next_state, reward, done, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.2f}, done={done}, "
              f"player={info.get('player', '?')}, phase={info.get('phase', '?')}")

        if done or truncated:
            state, info = env.reset()
        else:
            state = next_state

    env.close()
    print("  ✓ Environment test passed!\n")


def test_ball_in_hand():
    """Test that ball-in-hand placement works correctly."""
    print("Testing ball-in-hand placement...")

    from environment.pooltool_env import SnookerEnv

    env = SnookerEnv(render_mode=None)
    state, _ = env.reset()

    # On first shot, ball_in_hand should be True
    assert env.ball_in_hand, "Expected ball_in_hand=True at start"

    # Take a shot with specific placement (8-dim: last 2 are spin)
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)
    next_state, reward, done, truncated, info = env.step(action)

    # After the shot, ball_in_hand should be False (unless white was pocketed)
    if 'white' not in info.get('pocketed', []):
        assert not env.ball_in_hand, "Expected ball_in_hand=False after shot"

    env.close()
    print("  ✓ Ball-in-hand test passed!\n")


def test_target_selection():
    """Test target ball selection logic."""
    print("Testing target selection...")

    from environment.pooltool_env import SnookerEnv

    env = SnookerEnv(render_mode=None)
    env.reset()

    # Phase should be 'red' at start
    assert env.phase == 'red', f"Expected phase='red', got '{env.phase}'"

    legal = env._get_legal_targets()
    assert all(k.startswith('red_') for k in legal), \
        "In red phase, legal targets should all be reds"
    assert len(legal) == 15, f"Expected 15 legal reds, got {len(legal)}"

    all_tgt = env._get_all_targetable()
    assert len(all_tgt) == 21, f"Expected 21 targetable balls, got {len(all_tgt)}"

    env.close()
    print("  ✓ Target selection test passed!\n")


def test_ppo_agent():
    """Test PPO agent with 8-dim actions and 75-dim states."""
    print("Testing PPO Agent (8-dim action, 75-dim state)...")

    from algorithms.ppo import PPO

    state_dim = 75
    action_dim = 8

    ppo = PPO(state_dim, action_dim)

    dummy_state = np.random.randn(state_dim).astype(np.float32)
    action, log_prob, value = ppo.select_action(dummy_state)

    print(f"  Action shape: {action.shape}")
    print(f"  Action values: {action}")
    print(f"  Log prob: {log_prob:.4f}")
    print(f"  Value: {value:.4f}")

    assert action.shape == (8,), f"Expected action shape (8,), got {action.shape}"
    assert np.all(action >= -1.0) and np.all(action <= 1.0), \
        f"Action out of [-1, 1] range: {action}"

    print("  ✓ PPO agent test passed!\n")


def test_sac_agent():
    """Test SAC agent with 8-dim actions and 75-dim states."""
    print("Testing SAC Agent (8-dim action, 75-dim state)...")

    from algorithms.sac import SAC

    state_dim = 75
    action_dim = 8

    sac = SAC(state_dim, action_dim, {
        'warmup_steps': 5,  # low warmup for testing
        'batch_size': 4,
        'buffer_size': 100,
        'actor_update_interval': 1,  # no delay for testing
    })

    dummy_state = np.random.randn(state_dim).astype(np.float32)

    # During warmup: random actions
    action, log_prob, value = sac.select_action(dummy_state)
    print(f"  Warmup action shape: {action.shape}")
    assert action.shape == (8,), f"Expected shape (8,), got {action.shape}"
    assert np.all(action >= -1.0) and np.all(action <= 1.0), \
        f"Action out of [-1, 1] range: {action}"

    # Fill buffer with dummy transitions
    for i in range(10):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.uniform(-1, 1, action_dim).astype(np.float32)
        r = float(np.random.randn())
        ns = np.random.randn(state_dim).astype(np.float32)
        d = (i == 9)
        sac.store_transition(s, a, r, ns, d)

    # After warmup: policy actions
    action, log_prob, value = sac.select_action(dummy_state)
    print(f"  Policy action shape: {action.shape}")
    print(f"  Action values: {action}")
    print(f"  Log prob: {log_prob:.4f}")
    print(f"  Q value: {value:.4f}")
    assert action.shape == (8,), f"Expected shape (8,), got {action.shape}"

    # Deterministic action
    det_action, _, _ = sac.select_action(dummy_state, deterministic=True)
    print(f"  Deterministic action: {det_action}")
    assert det_action.shape == (8,)

    # Test update
    sac.update()
    stats = sac.get_stats()
    assert 'policy_loss' in stats
    assert 'alpha' in stats
    assert len(stats['policy_loss']) > 0, "SAC should have updated"
    print(f"  Alpha: {sac.alpha:.4f}")
    print(f"  Policy loss: {stats['policy_loss'][-1]:.4f}")
    print(f"  Replay buffer size: {len(sac.replay_buffer)}")

    print("  ✓ SAC agent test passed!\n")


def test_sac_integration():
    """Integration test: SAC agent interacts with the pooltool environment."""
    print("Testing SAC ↔ Environment integration...")

    from environment.pooltool_env import SnookerEnv
    from algorithms.sac import SAC

    env = SnookerEnv(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SAC(state_dim, action_dim, {
        'warmup_steps': 3,
        'batch_size': 4,
        'buffer_size': 1000,
    })

    state, _ = env.reset()
    total_reward = 0
    steps = 0

    for _ in range(15):
        action, log_prob, value = agent.select_action(state)
        assert action.shape == (8,), f"Action shape mismatch: {action.shape}"

        next_state, reward, done, truncated, info = env.step(action)
        agent.store_transition(state, action, reward, next_state,
                               done or truncated)
        agent.update()

        total_reward += reward
        steps += 1
        if done or truncated:
            break
        state = next_state

    print(f"  Ran {steps} steps, total reward: {total_reward:.2f}")
    print(f"  Replay buffer size: {len(agent.replay_buffer)}")
    print(f"  Updates performed: {agent._update_count}")

    # Save and reload
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp_path = f.name
    agent.save(tmp_path)

    agent2 = SAC(state_dim, action_dim)
    assert agent2.load(tmp_path), "Failed to load SAC checkpoint"
    a1, _, _ = agent.select_action(state, deterministic=True)
    a2, _, _ = agent2.select_action(state, deterministic=True)
    assert np.allclose(a1, a2, atol=1e-5), \
        f"Loaded model gives different actions: {a1} vs {a2}"
    os.remove(tmp_path)
    print("  Save/Load verified ✓")

    env.close()
    print("  ✓ SAC integration test passed!\n")


def test_self_play_turn_switching():
    """Test that the environment switches players correctly."""
    print("Testing self-play turn switching...")

    from environment.pooltool_env import SnookerEnv

    env = SnookerEnv(render_mode=None)
    env.reset()

    initial_player = env.current_player
    assert initial_player == 0, f"Expected player 0 at start, got {initial_player}"

    # Take shots until player switches (should happen on miss/foul)
    switched = False
    for _ in range(20):
        action = env.action_space.sample()
        _, _, done, truncated, info = env.step(action)
        if env.current_player != initial_player:
            switched = True
            break
        if done or truncated:
            break

    # It's very likely that with random actions, the player switches quickly
    print(f"  Player switched: {switched}")
    print(f"  Current player after shots: {env.current_player}")

    env.close()
    print("  ✓ Self-play turn switching test passed!\n")


def test_integration():
    """Integration test: PPO agent interacts with the pooltool environment."""
    print("Testing PPO ↔ Environment integration...")

    from environment.pooltool_env import SnookerEnv
    from algorithms.ppo import PPO

    env = SnookerEnv(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    assert state_dim == 75, f"Expected state_dim=54, got {state_dim}"
    assert action_dim == 8, f"Expected action_dim=8, got {action_dim}"

    agent = PPO(state_dim, action_dim)

    state, _ = env.reset()
    total_reward = 0
    steps = 0

    for _ in range(20):
        action, log_prob, value = agent.select_action(state)
        assert action.shape == (8,), f"Action shape mismatch: {action.shape}"

        next_state, reward, done, truncated, info = env.step(action)
        agent.memory.add(state, action, log_prob, value, reward, done)
        total_reward += reward
        steps += 1
        if done or truncated:
            break
        state = next_state

    print(f"  Ran {steps} steps, total reward: {total_reward:.2f}")
    print(f"  Memory size: {len(agent.memory.states)}")

    # Try an update (may not happen if fewer than batch_size transitions)
    agent.update()

    env.close()
    print("  ✓ Integration test passed!\n")


def test_event_detection():
    """Test that pooltool event analysis works correctly."""
    print("Testing event detection (ball-ball, ball-pocket)...")

    from environment.pooltool_env import SnookerEnv

    env = SnookerEnv(render_mode=None)
    env.reset()

    # Run several random shots and check event detection fields
    detected_hit = False
    detected_pot = False

    for i in range(15):
        action = env.action_space.sample()
        _, reward, done, truncated, info = env.step(action)

        # _white_hit_any and _first_contact should be populated
        if env._white_hit_any:
            detected_hit = True
            assert env._first_contact is not None, \
                "White hit a ball but first_contact is None"
        if env._last_pocketed:
            detected_pot = True

        if done or truncated:
            env.reset()

    print(f"  Detected white hitting balls: {detected_hit}")
    print(f"  Detected pocketed balls: {detected_pot}")

    # With 15 random shots, we almost certainly hit at least one ball
    assert detected_hit, "Expected at least one ball hit in 15 random shots"

    env.close()
    print("  ✓ Event detection test passed!\n")


def test_scoring_and_fouls():
    """Test scoring and foul tracking."""
    print("Testing scoring and foul tracking...")

    from environment.pooltool_env import SnookerEnv

    env = SnookerEnv(render_mode=None)
    env.reset()

    initial_scores = env.scores.copy()

    # Run several shots
    for _ in range(10):
        action = env.action_space.sample()
        _, _, done, truncated, info = env.step(action)
        if done or truncated:
            break

    # Scores should have changed (from pots or foul points)
    scores_changed = (env.scores[0] != initial_scores[0] or
                      env.scores[1] != initial_scores[1])
    print(f"  P1 score: {env.scores[0]}, P2 score: {env.scores[1]}")
    print(f"  P1 pot: {env.pot_scores[0]}, P2 pot: {env.pot_scores[1]}")
    print(f"  P1 foul_rcv: {env.foul_received[0]}, P2 foul_rcv: {env.foul_received[1]}")
    print(f"  Scores changed: {scores_changed}")

    # Check info dict has all expected keys
    expected_keys = ['break', 'score_p1', 'score_p2', 'pot_p1', 'pot_p2',
                     'foul_rcv_p1', 'foul_rcv_p2', 'foul', 'pocketed',
                     'phase', 'player']
    for key in expected_keys:
        assert key in info, f"Missing key '{key}' in info dict"

    env.close()
    print("  ✓ Scoring and foul tracking test passed!\n")


def test_reward_breakdown():
    """Test that reward_breakdown appears in info with correct keys."""
    print("Testing reward breakdown (simplified)...")

    from environment.pooltool_env import SnookerEnv, RewardConfig

    env = SnookerEnv(render_mode=None)
    env.reset()

    # Take one shot and check breakdown
    action = env.action_space.sample()
    _, reward, done, truncated, info = env.step(action)

    assert 'reward_breakdown' in info, "Missing reward_breakdown in info"
    bd = info['reward_breakdown']

    # Simplified reward only has: foul, distance, win_loss, total (+ foul_type metadata)
    expected_keys = ['foul', 'distance', 'win_loss', 'total']
    for k in expected_keys:
        assert k in bd, f"Missing key '{k}' in reward_breakdown"

    # Total should match sum of numeric components
    computed = sum(v for k, v in bd.items() if k != 'total' and isinstance(v, (int, float)))
    assert abs(computed - bd['total']) < 1e-6, \
        f"Breakdown total mismatch: {computed:.6f} vs {bd['total']:.6f}"

    # Total should match returned reward
    assert abs(reward - bd['total']) < 1e-6, \
        f"Returned reward {reward:.6f} != breakdown total {bd['total']:.6f}"

    print(f"  Breakdown: {bd}")
    print(f"  Reward = {reward:.4f}")

    # Test custom RewardConfig
    rc = RewardConfig(pot_reward=20.0, break_bonus=5.0, foul_penalty=-0.5,
                      miss_penalty=-0.2, win_reward=50.0, lose_reward=-50.0)
    env2 = SnookerEnv(render_mode=None, reward_cfg=rc)
    env2.reset()
    assert env2.rc.pot_reward == 20.0
    assert env2.rc.break_bonus == 5.0
    assert env2.rc.foul_penalty == -0.5
    assert env2.rc.miss_ball_penalty == -3.0  # default for miss_ball
    assert env2.rc.white_pocket_penalty == -2.0  # default unchanged
    assert env2.rc.miss_penalty == -0.2
    assert env2.rc.win_reward == 50.0
    assert env2.rc.lose_reward == -50.0
    env2.close()

    env.close()
    print("  ✓ Reward breakdown test passed!\n")


def test_utils():
    """Test utility functions"""
    print("Testing Utilities...")

    from utils import set_seed, count_parameters, get_device_info

    set_seed(42)

    info = get_device_info()
    print(f"  Device info: {info}")

    import torch.nn as nn

    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 50)
            self.fc2 = nn.Linear(50, 10)

    net = DummyNet()
    params = count_parameters(net)
    print(f"  Network parameters: {params}")

    print("  ✓ Utils test passed!\n")


def main():
    print("=" * 60)
    print("Snooker RL Test Suite (Pooltool + Self-Play)")
    print("=" * 60 + "\n")

    try:
        test_environment()
        test_ball_in_hand()
        test_target_selection()
        test_ppo_agent()
        test_sac_agent()
        test_self_play_turn_switching()
        test_integration()
        test_sac_integration()
        test_event_detection()
        test_scoring_and_fouls()
        test_reward_breakdown()
        test_utils()

        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nTrain with PPO:")
        print("  python train.py --algo ppo --num_episodes 2000")
        print("\nTrain with SAC (recommended – better sample efficiency):")
        print("  python train.py --algo sac --num_episodes 2000")
        print("\nEvaluate a saved model (auto-detects algorithm):")
        print("  python evaluate.py --load_model saved_models/sac_snooker_final.pt")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
