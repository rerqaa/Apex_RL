import os
import psutil
import torch
import rlgym_sim
from rlgym_ppo import Learner

# We must monkeypatch rlgym_ppo to use our custom architecture
import rlgym_ppo.ppo.ppo_learner
from architecture import AttentionApexPolicy, AttentionApexValueEstimator
rlgym_ppo.ppo.ppo_learner.ContinuousPolicy = AttentionApexPolicy
rlgym_ppo.ppo.ppo_learner.ValueEstimator = AttentionApexValueEstimator


# ---------------------------------------------------------------------------
# Hardware auto-detection
# ---------------------------------------------------------------------------

def detect_hardware():
    """Detect available CPU, RAM, and GPU resources."""
    info = {}

    # CPU
    logical_cores = os.cpu_count() or 4
    physical_cores = psutil.cpu_count(logical=False) or logical_cores
    info["logical_cores"] = logical_cores
    info["physical_cores"] = physical_cores

    # RAM (GB)
    ram = psutil.virtual_memory()
    info["ram_total_gb"] = round(ram.total / (1024 ** 3), 1)
    info["ram_available_gb"] = round(ram.available / (1024 ** 3), 1)

    # GPU
    info["gpu_count"] = torch.cuda.device_count()
    info["gpus"] = []
    info["total_vram_gb"] = 0.0

    for i in range(info["gpu_count"]):
        props = torch.cuda.get_device_properties(i)
        vram_gb = round(props.total_memory / (1024 ** 3), 1)
        info["gpus"].append({
            "index": i,
            "name": props.name,
            "vram_gb": vram_gb,
            "compute_capability": f"{props.major}.{props.minor}",
        })
        info["total_vram_gb"] += vram_gb

    info["device"] = "cuda" if info["gpu_count"] > 0 else "cpu"
    return info


def compute_training_params(hw):
    """Compute optimal training hyperparameters from detected hardware."""

    n_proc = hw["logical_cores"] if hw["logical_cores"] > 0 else 4

    # Large batch for low-variance gradient estimates on 4 vCPUs.
    ppo_batch_size = 100_000
    ppo_minibatch_size = 10_000  # must divide batch_size evenly

    ts_per_iteration = 100_000   # collect one full batch per iteration
    exp_buffer_size = 100_000    # standard on-policy PPO buffer

    return {
        "n_proc": int(n_proc),
        "ppo_batch_size": int(ppo_batch_size),
        "ppo_minibatch_size": int(ppo_minibatch_size),
        "ts_per_iteration": int(ts_per_iteration),
        "exp_buffer_size": int(exp_buffer_size),
    }


def _nearest_divisor(total, target):
    """Return the value closest to `target` that evenly divides `total`."""
    if total % target == 0:
        return target
    # Search outward from target
    for offset in range(1, target):
        if target - offset > 0 and total % (target - offset) == 0:
            return target - offset
        if total % (target + offset) == 0:
            return target + offset
    return total  # fallback: one giant minibatch


def print_hardware_summary(hw, params):
    """Print a formatted summary of detected hardware and chosen params."""
    sep = "=" * 56
    print(sep)
    print("  HARDWARE AUTO-DETECTION SUMMARY")
    print(sep)
    print(f"  CPU   : {hw['physical_cores']} physical / "
          f"{hw['logical_cores']} logical cores")
    print(f"  RAM   : {hw['ram_available_gb']} / "
          f"{hw['ram_total_gb']} GB available")

    if hw["gpu_count"] == 0:
        print("  GPU   : NONE (CPU-only mode)")
    else:
        for g in hw["gpus"]:
            print(f"  GPU {g['index']} : {g['name']}  |  "
                  f"{g['vram_gb']} GB VRAM  |  "
                  f"CC {g['compute_capability']}")

    print(sep)
    print("  TRAINING PARAMETERS")
    print(sep)
    print(f"  n_proc             : {params['n_proc']}")
    print(f"  ppo_batch_size     : {params['ppo_batch_size']:,}")
    print(f"  ppo_minibatch_size : {params['ppo_minibatch_size']:,}")
    print(f"  ts_per_iteration   : {params['ts_per_iteration']:,}")
    print(f"  exp_buffer_size    : {params['exp_buffer_size']:,}")
    print(sep)


# ---------------------------------------------------------------------------
# Environment builder
# ---------------------------------------------------------------------------

def build_env():
    """Create rlgym_sim environment. Imports inside for multiprocessing."""
    from rlgym_sim.utils.reward_functions.common_rewards import EventReward
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.terminal_conditions.common_conditions import (
        TimeoutCondition, GoalScoredCondition
    )
    from rlgym_sim.utils.action_parsers import ContinuousAction
    from obs import ZeroPaddedObs
    from rewards import Phase1Reward

    reward_fn = CombinedReward.from_zipped(
        (Phase1Reward(), 1.0),
        (EventReward(goal=100.0, concede=-100.0), 1.0)
    )

    env = rlgym_sim.make(
        tick_skip=8,
        team_size=1,
        spawn_opponents=False,  # Phase 1: 1v0
        terminal_conditions=[TimeoutCondition(300), GoalScoredCondition()],
        reward_fn=reward_fn,
        obs_builder=ZeroPaddedObs(),
        action_parser=ContinuousAction()
    )
    return env


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_phase_1():
    hw = detect_hardware()
    params = compute_training_params(hw)
    print_hardware_summary(hw, params)

    learner = Learner(
        env_create_function=build_env,
        ppo_batch_size=params["ppo_batch_size"],
        ppo_minibatch_size=params["ppo_minibatch_size"],
        ts_per_iteration=params["ts_per_iteration"],
        exp_buffer_size=params["exp_buffer_size"],
        policy_layer_sizes=(256, 256, 256),
        critic_layer_sizes=(256, 256, 256),
        n_proc=params["n_proc"],
        log_to_wandb=False,
        checkpoints_save_folder="checkpoints/",
        save_every_ts=500_000,
        ppo_ent_coef=0.005,
        policy_lr=3e-5,
        critic_lr=3e-5,
        standardize_returns=True,
        standardize_obs=False,
    )

    learner.learn()


if __name__ == "__main__":
    train_phase_1()
