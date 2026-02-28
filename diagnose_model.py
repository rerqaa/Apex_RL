"""
Diagnostic script: loads the trained model and feeds it a synthetic observation
to check what actions it outputs.
"""
import os
import sys
import numpy as np
import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, SRC_DIR)

from architecture import AttentionApexPolicy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = AttentionApexPolicy(
    input_shape=None,
    output_shape=16,
    layer_sizes=(256, 256, 256),
    device=device
).to(device)

checkpoint_dir = os.path.join(SRC_DIR, 'checkpoints')
latest_ts = -1
latest_path = None
for run_id in os.listdir(checkpoint_dir):
    run_dir = os.path.join(checkpoint_dir, run_id)
    if not os.path.isdir(run_dir): continue
    for step in os.listdir(run_dir):
        step_dir = os.path.join(run_dir, step)
        if not os.path.isdir(step_dir): continue
        policy_file = os.path.join(step_dir, 'PPO_POLICY.pt')
        if os.path.isfile(policy_file):
            try:
                ts = int(step)
                if ts > latest_ts:
                    latest_ts = ts
                    latest_path = policy_file
            except ValueError:
                pass

if latest_path is None:
    print("ERROR: No checkpoint found!")
    sys.exit(1)

print(f"Loading checkpoint: {latest_path}")
state_dict = torch.load(latest_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

OBS_SIZE = 225
ball_start = 150

obs2 = np.zeros(OBS_SIZE, dtype=np.float32)
obs2[2] = 17.0 / 2300.0
obs2[9], obs2[10], obs2[11] = 0.0, 1.0, 0.0
obs2[12], obs2[13], obs2[14] = 0.0, 0.0, 1.0
obs2[15] = 0.33
obs2[16], obs2[17], obs2[18] = 1.0, 1.0, 1.0
obs2[20], obs2[24] = 1.0, 1.0

obs2[ball_start + 0] = 0.0
obs2[ball_start + 1] = 1500/2300
obs2[ball_start + 2] = 93/2300
obs2[ball_start + 23], obs2[ball_start + 24] = 1.0, 1.0
obs2[217], obs2[218], obs2[219] = 0.0, 1500/2300, 76/2300

print("============================================================")
with torch.no_grad():
    a, _ = model.get_action(obs2, deterministic=True)
print(f"TEST (Ball AHEAD): \n{np.round(a.cpu().numpy().flatten(), 3)}")

obs3 = obs2.copy()
obs3[ball_start + 1] = -1500/2300
obs3[218] = -1500/2300
with torch.no_grad():
    a, _ = model.get_action(obs3, deterministic=True)
print(f"TEST (Ball BEHIND):\n{np.round(a.cpu().numpy().flatten(), 3)}")

obs4 = obs2.copy()
obs4[ball_start + 0] = -1500/2300
obs4[ball_start + 1] = 0
obs4[217] = -1500/2300
obs4[218] = 0
with torch.no_grad():
    a, _ = model.get_action(obs4, deterministic=True)
print(f"TEST (Ball LEFT):  \n{np.round(a.cpu().numpy().flatten(), 3)}")

obs5 = obs2.copy()
obs5[ball_start + 0] = 1500/2300
obs5[ball_start + 1] = 0
obs5[217] = 1500/2300
obs5[218] = 0
with torch.no_grad():
    a, _ = model.get_action(obs5, deterministic=True)
print(f"TEST (Ball RIGHT): \n{np.round(a.cpu().numpy().flatten(), 3)}")
print("============================================================")
