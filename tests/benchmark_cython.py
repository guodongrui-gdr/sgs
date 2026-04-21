#!/usr/bin/env python
"""
Cython Optimization Benchmark Script

Measures performance of hot functions before/after Cython optimization.
Target: 3x improvement from ~300 steps/s to ~1000+ steps/s
"""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_distance_calculation(n_iterations: int = 100000):
    """Benchmark _calculate_distance function (5 duplicate implementations)."""
    print("\n=== Benchmark: Distance Calculation ===")

    # Mock linked list structure (5 players in circle)
    class MockPlayer:
        def __init__(self, idx, equipment=None):
            self.idx = idx
            self.equipment = equipment or {}
            self.next_player = None
            self.prev_player = None
            self.is_alive = True

    # Create circular linked list
    players = [MockPlayer(i) for i in range(5)]
    for i in range(5):
        players[i].next_player = players[(i + 1) % 5]
        players[i].prev_player = players[(i - 1) % 5]

    # Add equipment for some players
    players[0].equipment = {"进攻坐骑": True}  # Attack horse
    players[2].equipment = {"防御坐骑": True}  # Defense horse

    # Reference implementation (from game_engine.py)
    def calculate_distance_python(source, target):
        if source == target:
            return 0
        dist = 1
        current = source.next_player
        while current != target:
            dist += 1
            current = current.next_player
        reverse_dist = 1
        current = source.prev_player
        while current != target:
            reverse_dist += 1
            current = current.prev_player
        dist = min(dist, reverse_dist)
        if source.equipment.get("进攻坐骑"):
            dist = max(1, dist - 1)
        if target.equipment.get("防御坐骑"):
            dist += 1
        return dist

    # Warmup
    for _ in range(1000):
        calculate_distance_python(players[0], players[2])

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        for src in players:
            for tgt in players:
                if src != tgt:
                    calculate_distance_python(src, tgt)
    elapsed = time.perf_counter() - start

    calls_per_iteration = len(players) * (len(players) - 1)  # 5 * 4 = 20
    total_calls = n_iterations * calls_per_iteration
    calls_per_sec = total_calls / elapsed

    print(f"  Iterations: {n_iterations}")
    print(f"  Total calls: {total_calls}")
    print(f"  Elapsed: {elapsed:.4f}s")
    print(f"  Calls/sec: {calls_per_sec:.2f}")
    print(f"  Avg time per call: {elapsed / total_calls * 1e6:.2f} microseconds")

    return {
        "function": "calculate_distance",
        "iterations": n_iterations,
        "total_calls": total_calls,
        "elapsed": elapsed,
        "calls_per_sec": calls_per_sec,
    }


def benchmark_is_in_range(n_iterations: int = 100000):
    """Benchmark _is_in_range function (used ~160 times per mask generation)."""
    print("\n=== Benchmark: Is In Range ===")

    # Mock players with attributes
    class MockPlayer:
        def __init__(self, idx, attack_range=1, equipment=None):
            self.idx = idx
            self.player_id = idx
            self.attack_range = attack_range
            self.equipment = equipment or {}
            self.is_alive = True

    players = [
        MockPlayer(0, attack_range=1, equipment={"进攻坐骑": True}),
        MockPlayer(1, attack_range=2),
        MockPlayer(2, attack_range=1, equipment={"防御坐骑": True}),
        MockPlayer(3, attack_range=3),
        MockPlayer(4, attack_range=1),
    ]

    # Reference implementation (from action_encoder.py)
    def is_in_range_python(source, target, max_range=None):
        src_idx = getattr(source, "idx", None)
        tgt_idx = getattr(target, "idx", None)
        if src_idx is None:
            src_idx = getattr(source, "player_id", 0)
        if tgt_idx is None:
            tgt_idx = getattr(target, "player_id", 0)

        distance = min(abs(src_idx - tgt_idx), 5 - abs(src_idx - tgt_idx))

        src_eq = getattr(source, "equipment", {})
        tgt_eq = getattr(target, "equipment", {})

        if src_eq.get("进攻坐骑"):
            distance = max(1, distance - 1)
        if tgt_eq.get("防御坐骑"):
            distance += 1

        attack_range = max_range if max_range else getattr(source, "attack_range", 1)
        return distance <= attack_range

    # Warmup
    for _ in range(1000):
        is_in_range_python(players[0], players[2])

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        for src in players:
            for tgt in players:
                if src != tgt:
                    is_in_range_python(src, tgt)
    elapsed = time.perf_counter() - start

    calls_per_iteration = len(players) * (len(players) - 1)
    total_calls = n_iterations * calls_per_iteration
    calls_per_sec = total_calls / elapsed

    print(f"  Iterations: {n_iterations}")
    print(f"  Total calls: {total_calls}")
    print(f"  Elapsed: {elapsed:.4f}s")
    print(f"  Calls/sec: {calls_per_sec:.2f}")
    print(f"  Avg time per call: {elapsed / total_calls * 1e6:.2f} microseconds")

    return {
        "function": "is_in_range",
        "iterations": n_iterations,
        "total_calls": total_calls,
        "elapsed": elapsed,
        "calls_per_sec": calls_per_sec,
    }


def benchmark_env_step(n_episodes: int = 10, max_steps: int = 200):
    """Benchmark full env.step() to measure end-to-end impact."""
    print("\n=== Benchmark: Env Step (End-to-End) ===")

    try:
        from ai.gym_wrapper import SGSEnv, SGSConfig
    except ImportError:
        print("  SKIPPED: ai.gym_wrapper not available")
        return None

    env = SGSEnv(SGSConfig(player_num=5))

    total_steps = 0
    start = time.perf_counter()

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Random action (valid action sampling)
            action_masks = info.get("action_masks", None)
            if action_masks:
                type_mask = action_masks[0]
                valid_types = [i for i, m in enumerate(type_mask) if m > 0]
                if valid_types:
                    action_type = valid_types[0]  # Use first valid
                    action = (action_type, 0, 0)
                else:
                    action = (1, 0, 0)  # END_TURN
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        total_steps += steps

    elapsed = time.perf_counter() - start
    steps_per_sec = total_steps / elapsed

    print(f"  Episodes: {n_episodes}")
    print(f"  Total steps: {total_steps}")
    print(f"  Elapsed: {elapsed:.4f}s")
    print(f"  Steps/sec: {steps_per_sec:.2f}")

    env.close()

    return {
        "function": "env_step",
        "episodes": n_episodes,
        "total_steps": total_steps,
        "elapsed": elapsed,
        "steps_per_sec": steps_per_sec,
    }


def benchmark_action_mask_generation(n_iterations: int = 1000):
    """Benchmark action mask generation (hotspot: ~160 _is_in_range calls per mask)."""
    print("\n=== Benchmark: Action Mask Generation ===")

    try:
        from ai.action_encoder import ActionMaskGenerator
        from ai.gym_wrapper import SGSEnv, SGSConfig
    except ImportError:
        print("  SKIPPED: ai modules not available")
        return None

    env = SGSEnv(SGSConfig(player_num=5))
    obs, info = env.reset()

    mask_gen = env.action_mask_generator

    player = env.players[env.current_player_idx] if env.players else None
    if player is None:
        env.close()
        return None

    for _ in range(100):
        mask_gen.generate_masks({}, player, env.engine, 0, None)

    start = time.perf_counter()
    for _ in range(n_iterations):
        mask_gen.generate_masks({}, player, env.engine, 0, None)
    elapsed = time.perf_counter() - start

    masks_per_sec = n_iterations / elapsed

    print(f"  Iterations: {n_iterations}")
    print(f"  Elapsed: {elapsed:.4f}s")
    print(f"  Masks/sec: {masks_per_sec:.2f}")
    print(f"  Avg time per mask: {elapsed / n_iterations * 1e3:.2f} ms")

    env.close()

    return {
        "function": "action_mask_generation",
        "iterations": n_iterations,
        "elapsed": elapsed,
        "masks_per_sec": masks_per_sec,
    }


def run_all_benchmarks():
    """Run all benchmarks and save results."""
    print("=" * 60)
    print("Cython Optimization Benchmark - Baseline")
    print("=" * 60)

    results = {}

    # Run benchmarks
    results["distance"] = benchmark_distance_calculation(100000)
    results["is_in_range"] = benchmark_is_in_range(100000)
    results["env_step"] = benchmark_env_step(10, 200)
    results["mask_gen"] = benchmark_action_mask_generation(1000)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, result in results.items():
        if result:
            if "calls_per_sec" in result:
                print(f"{name}: {result['calls_per_sec']:.2f} calls/sec")
            elif "steps_per_sec" in result:
                print(f"{name}: {result['steps_per_sec']:.2f} steps/sec")
            elif "masks_per_sec" in result:
                print(f"{name}: {result['masks_per_sec']:.2f} masks/sec")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cython optimization benchmark")
    parser.add_argument(
        "--iterations", type=int, default=100000, help="Iterations for micro-benchmarks"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Episodes for env benchmark"
    )
    parser.add_argument(
        "--mask-iterations",
        type=int,
        default=1000,
        help="Iterations for mask benchmark",
    )

    args = parser.parse_args()

    if args.iterations != 100000:
        benchmark_distance_calculation(args.iterations)
        benchmark_is_in_range(args.iterations)

    if args.episodes != 10:
        benchmark_env_step(args.episodes)

    if args.mask_iterations != 1000:
        benchmark_action_mask_generation(args.mask_iterations)

    run_all_benchmarks()
