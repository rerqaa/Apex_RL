"""
Converts RLBot GameTickPacket data into the 185-dim observation
format expected by the trained Apex-RL model.

Observation layout (must match obs.py exactly):
  6 × (20 features + 1 mask) = 126  [player entities]
  + 9   ball
  + 34  boost pads
  + 8   previous action
  + 3   relative ball position
  + 3   relative ball velocity
  + 1   score differential
  + 1   ball touched
  = 185 total
"""

import numpy as np
import math

# Must match obs.py
MAX_ALLIES = 2
MAX_ENEMIES = 3
PLAYER_FEAT_SIZE = 20

# Normalization constants (must match obs.py)
POS_STD = 2300.0
VEL_STD = 2300.0
ANG_STD = 5.5

# rlgym_sim BOOST_LOCATIONS — 34 pads in the exact order used during training.
# The adapter must map RLBot's boost pads to this ordering.
BOOST_LOCATIONS = (
    (0.0, -4240.0, 70.0),
    (-1792.0, -4184.0, 70.0),
    (1792.0, -4184.0, 70.0),
    (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0),
    (-940.0, -3308.0, 70.0),
    (940.0, -3308.0, 70.0),
    (0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0),
    (3584.0, -2484.0, 70.0),
    (-1788.0, -2300.0, 70.0),
    (1788.0, -2300.0, 70.0),
    (-2048.0, -1036.0, 70.0),
    (0.0, -1024.0, 70.0),
    (2048.0, -1036.0, 70.0),
    (-3584.0, 0.0, 73.0),
    (-1024.0, 0.0, 70.0),
    (1024.0, 0.0, 70.0),
    (3584.0, 0.0, 73.0),
    (-2048.0, 1036.0, 70.0),
    (0.0, 1024.0, 70.0),
    (2048.0, 1036.0, 70.0),
    (-1788.0, 2300.0, 70.0),
    (1788.0, 2300.0, 70.0),
    (-3584.0, 2484.0, 70.0),
    (3584.0, 2484.0, 70.0),
    (0.0, 2816.0, 70.0),
    (-940.0, 3310.0, 70.0),
    (940.0, 3308.0, 70.0),
    (-3072.0, 4096.0, 73.0),
    (3072.0, 4096.0, 73.0),
    (-1792.0, 4184.0, 70.0),
    (1792.0, 4184.0, 70.0),
    (0.0, 4240.0, 70.0),
)


def _euler_to_forward_up(pitch, yaw, roll):
    """Convert pitch/yaw/roll (radians) to forward and up unit vectors.
    Uses the same rotation matrix convention as rlgym_sim."""
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cr = math.cos(roll)
    sr = math.sin(roll)

    # Forward vector (column 0 of rotation matrix)
    forward = np.array([
        cp * cy,
        cp * sy,
        sp
    ])

    # Up vector (column 2 of rotation matrix)
    up = np.array([
        -sr * cy * sp - cr * sy,
        -sr * sy * sp + cr * cy,
        sr * cp
    ])

    return forward, up


class GameStateAdapter:
    """Translates RLBot GameTickPacket → 185-dim observation array."""

    def __init__(self):
        self.previous_action = np.zeros(8, dtype=np.float32)
        # Maps RLBot boost pad index → training pad index.
        # Built on first packet when FieldInfo is available.
        self._boost_index_map = None

    def build_boost_map(self, field_info):
        """Build mapping from RLBot boost pad indices to rlgym_sim 34-pad ordering.
        Call once with get_field_info() during initialize_agent()."""
        self._boost_index_map = {}
        for rlbot_idx in range(field_info.num_boosts):
            pad = field_info.boost_pads[rlbot_idx]
            loc = pad.location
            # Find matching BOOST_LOCATIONS entry
            for sim_idx, sim_loc in enumerate(BOOST_LOCATIONS):
                if (abs(round(loc.x) - sim_loc[0]) < 5 and
                        abs(round(loc.y) - sim_loc[1]) < 5):
                    self._boost_index_map[rlbot_idx] = sim_idx
                    break

    def _normalize_car(self, car_info):
        """Extract 20 features from an RLBot PlayerInfo struct."""
        phys = car_info.physics
        obs = []

        # Position (3)
        obs.extend([phys.location.x / POS_STD,
                     phys.location.y / POS_STD,
                     phys.location.z / POS_STD])

        # Linear velocity (3)
        obs.extend([phys.velocity.x / VEL_STD,
                     phys.velocity.y / VEL_STD,
                     phys.velocity.z / VEL_STD])

        # Angular velocity (3)
        obs.extend([phys.angular_velocity.x / ANG_STD,
                     phys.angular_velocity.y / ANG_STD,
                     phys.angular_velocity.z / ANG_STD])

        # Forward and Up vectors from Euler angles (6)
        forward, up = _euler_to_forward_up(
            phys.rotation.pitch,
            phys.rotation.yaw,
            phys.rotation.roll
        )
        obs.extend(forward.tolist())
        obs.extend(up.tolist())

        # Boost amount (1) — RLBot gives 0-100, obs expects 0-1
        obs.append(car_info.boost / 100.0)

        # Booleans (4)
        obs.append(float(car_info.has_wheel_contact))  # on_ground
        obs.append(float(not car_info.jumped))          # has_jump (approx)
        obs.append(float(not car_info.double_jumped))   # has_flip (approx)
        obs.append(float(car_info.is_demolished))       # is_demoed

        return obs

    def build_obs(self, packet, bot_index, bot_team):
        """Build full 185-dim observation from a GameTickPacket.

        Args:
            packet: RLBot GameTickPacket
            bot_index: Index of this bot in packet.game_cars
            bot_team: Team number (0=blue, 1=orange)

        Returns:
            np.ndarray of shape (185,)
        """
        obs = []
        my_car = packet.game_cars[bot_index]

        # 1. Self (20 + 1 mask = 21)
        self_data = self._normalize_car(my_car)
        obs.extend(self_data + [1.0])

        # 2. Allies (zero-padded to MAX_ALLIES=2)
        allies = []
        for i in range(packet.num_cars):
            if i == bot_index:
                continue
            car = packet.game_cars[i]
            if car.team == bot_team:
                allies.append(car)

        for i in range(MAX_ALLIES):
            if i < len(allies):
                obs.extend(self._normalize_car(allies[i]) + [1.0])
            else:
                obs.extend([0.0] * PLAYER_FEAT_SIZE + [0.0])

        # 3. Enemies (zero-padded to MAX_ENEMIES=3)
        enemies = []
        for i in range(packet.num_cars):
            car = packet.game_cars[i]
            if car.team != bot_team:
                enemies.append(car)

        for i in range(MAX_ENEMIES):
            if i < len(enemies):
                obs.extend(self._normalize_car(enemies[i]) + [1.0])
            else:
                obs.extend([0.0] * PLAYER_FEAT_SIZE + [0.0])

        # 4. Ball (9)
        ball = packet.game_ball.physics
        obs.extend([
            ball.location.x / POS_STD,
            ball.location.y / POS_STD,
            ball.location.z / POS_STD,
            ball.velocity.x / VEL_STD,
            ball.velocity.y / VEL_STD,
            ball.velocity.z / VEL_STD,
            ball.angular_velocity.x / ANG_STD,
            ball.angular_velocity.y / ANG_STD,
            ball.angular_velocity.z / ANG_STD,
        ])

        # 5. Boost pads (34)
        boost_states = [0.0] * 34
        if self._boost_index_map is not None:
            for rlbot_idx, sim_idx in self._boost_index_map.items():
                boost_states[sim_idx] = float(packet.game_boosts[rlbot_idx].is_active)
        obs.extend(boost_states)

        # 6. Previous action (8)
        obs.extend(self.previous_action.tolist())

        # 7. Relative ball position (3)
        my_phys = my_car.physics
        obs.extend([
            (ball.location.x - my_phys.location.x) / POS_STD,
            (ball.location.y - my_phys.location.y) / POS_STD,
            (ball.location.z - my_phys.location.z) / POS_STD,
        ])

        # 8. Relative ball velocity (3)
        obs.extend([
            (ball.velocity.x - my_phys.velocity.x) / VEL_STD,
            (ball.velocity.y - my_phys.velocity.y) / VEL_STD,
            (ball.velocity.z - my_phys.velocity.z) / VEL_STD,
        ])

        # 9. Score differential (1)
        if bot_team == 0:
            score_diff = (packet.teams[0].score - packet.teams[1].score) / 10.0
        else:
            score_diff = (packet.teams[1].score - packet.teams[0].score) / 10.0
        obs.append(score_diff)

        # 10. Ball touched (1)
        obs.append(float(packet.game_ball.latest_touch.player_index == bot_index))

        return np.asarray(obs, dtype=np.float32)

    def set_previous_action(self, action):
        """Store the last action for the next observation."""
        self.previous_action = np.asarray(action, dtype=np.float32)
