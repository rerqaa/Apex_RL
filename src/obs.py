import numpy as np
from rlgym_sim.utils.obs_builders import ObsBuilder
from rlgym_sim.utils.gamestates import PlayerData, GameState

# Constants for architecture synchronization
MAX_ALLIES = 2
MAX_ENEMIES = 3

# Per-player physics features:
# Pos(3) + LinVel(3) + AngVel(3) + Forward(3) + Up(3) + Boost(1) + OnGround(1) + HasJump(1) + HasFlip(1) + IsDemoed(1)
PLAYER_PHYSICS_SIZE = 20
# One-hot entity type: [is_self, is_ally, is_enemy, is_ball]
ENTITY_TYPE_SIZE = 4
# Total per-entity features (before presence mask)
ENTITY_FEAT_SIZE = PLAYER_PHYSICS_SIZE + ENTITY_TYPE_SIZE  # 24

# One-hot entity type vectors
ONEHOT_SELF   = [1, 0, 0, 0]
ONEHOT_ALLY   = [0, 1, 0, 0]
ONEHOT_ENEMY  = [0, 0, 1, 0]
ONEHOT_BALL   = [0, 0, 0, 1]

class ZeroPaddedObs(ObsBuilder):
    def __init__(self):
        super().__init__()
        self.POS_STD = 2300.0
        self.VEL_STD = 2300.0
        self.ANG_STD = 5.5

    def reset(self, initial_state: GameState):
        pass

    def _normalize_player(self, player: PlayerData) -> list:
        obs = []
        obs.extend(player.car_data.position / self.POS_STD)
        obs.extend(player.car_data.linear_velocity / self.VEL_STD)
        obs.extend(player.car_data.angular_velocity / self.ANG_STD)
        obs.extend(player.car_data.forward())
        obs.extend(player.car_data.up())
        obs.append(player.boost_amount)           # [0, 1]
        obs.append(float(player.on_ground))        # 0 or 1
        obs.append(float(player.has_jump))         # 0 or 1
        obs.append(float(player.has_flip))         # 0 or 1
        obs.append(float(player.is_demoed))        # 0 or 1
        return obs  # length = PLAYER_PHYSICS_SIZE (20)

    def _normalize_ball(self, state: GameState) -> list:
        obs = []
        obs.extend(state.ball.position / self.POS_STD)
        obs.extend(state.ball.linear_velocity / self.VEL_STD)
        obs.extend(state.ball.angular_velocity / self.ANG_STD)
        return obs  # length = 9

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        obs = []

        # 1. Self Data (20 physics + 4 one-hot + 1 mask = 25)
        self_data = self._normalize_player(player)
        obs.extend(self_data + ONEHOT_SELF + [1.0])

        # 2. Allies with zero-padding (exclude self)
        allies = [p for p in state.players if p.team_num == player.team_num and p.car_id != player.car_id]

        for i in range(MAX_ALLIES):
            if i < len(allies):
                obs.extend(self._normalize_player(allies[i]) + ONEHOT_ALLY + [1.0])
            else:
                obs.extend(np.zeros(ENTITY_FEAT_SIZE).tolist() + [0.0])

        # 3. Enemies with zero-padding
        enemies = [p for p in state.players if p.team_num != player.team_num]

        for i in range(MAX_ENEMIES):
            if i < len(enemies):
                obs.extend(self._normalize_player(enemies[i]) + ONEHOT_ENEMY + [1.0])
            else:
                obs.extend(np.zeros(ENTITY_FEAT_SIZE).tolist() + [0.0])

        # 4. Ball Data (9 physics + 11 zero-padding + 4 one-hot + 1 mask = 25)
        ball_physics = self._normalize_ball(state)
        ball_padding = [0.0] * 11
        obs.extend(ball_physics + ball_padding + ONEHOT_BALL + [1.0])

        # 5. Boost Pads (34 features, each 0 or 1)
        obs.extend(state.boost_pads.tolist())

        # 6. Previous Action (8 features)
        if previous_action is None:
            previous_action = np.zeros(8)
        obs.extend(previous_action.tolist())

        # 7. Relative Ball Position (3 features)
        rel_ball_pos = (state.ball.position - player.car_data.position) / self.POS_STD
        obs.extend(rel_ball_pos.tolist())

        # 8. Relative Ball Velocity (3 features)
        rel_ball_vel = (state.ball.linear_velocity - player.car_data.linear_velocity) / self.VEL_STD
        obs.extend(rel_ball_vel.tolist())

        # 9. Score Differential (1 feature)
        if player.team_num == 0:  # Blue
            score_diff = (state.blue_score - state.orange_score) / 10.0
        else:  # Orange
            score_diff = (state.orange_score - state.blue_score) / 10.0
        obs.append(score_diff)

        # 10. Ball Touched (1 feature)
        obs.append(float(player.ball_touched))

        return np.asarray(obs, dtype=np.float32)