import numpy as np
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

class Phase1Reward(RewardFunction):
    def __init__(self):
        super().__init__()
        # Event weights
        self.touch_weight = 1.0 # Reduced from 5.0 to prevent wall-pinning exploits
        
        # Differential potential weights (scaled up as they represent tiny frame-to-frame deltas)
        self.vel_ball_to_goal_weight = 2.0
        self.ball_to_goal_pot_weight = 3.0
        self.car_to_ball_pot_weight = 1.0
        
        # Goal positions
        self.blue_goal = np.array([0, -5120.0, 0])
        self.orange_goal = np.array([0, 5120.0, 0])
        
        # Velocity normalization constant
        self.MAX_VEL = 2300.0
        
        # State tracking for differentials
        self.last_ball_to_goal_dist = {}
        self.last_car_to_ball_dist = {}

    def reset(self, initial_state: GameState):
        self.last_ball_to_goal_dist.clear()
        self.last_car_to_ball_dist.clear()

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0
        
        car_id = player.car_id
        target_goal = self.orange_goal if player.team_num == 0 else self.blue_goal

        # 1. Touch ball (Event)
        if player.ball_touched:
            reward += self.touch_weight

        # 2. Car to Ball Potential (Differential)
        # Encourages closing the distance to the ball.
        # Stopping near the ball yields 0 reward (current dist == last dist).
        dist_to_ball = np.linalg.norm(state.ball.position - player.car_data.position)
        
        if car_id in self.last_car_to_ball_dist:
            dist_diff_car_ball = self.last_car_to_ball_dist[car_id] - dist_to_ball
            # Normalize by max possible change per tick (approx MAX_VEL * tick_time)
            # Assuming ~120Hz physics and tick_skip=8, dt is roughly 0.066s.
            # Max diff is roughly 2300 * 0.066 = 151.8 UU. We scale the diff down to keep rewards stable.
            norm_diff = dist_diff_car_ball / 100.0
            
            # True differential potential: reward approaching, penalize driving away
            reward += self.car_to_ball_pot_weight * norm_diff
                
        self.last_car_to_ball_dist[car_id] = dist_to_ball

        # 3. Ball to Goal Potential (Differential)
        # Encourages moving the ball closer to the opponent's goal.
        dist_ball_to_goal = np.linalg.norm(target_goal - state.ball.position)
        
        if car_id in self.last_ball_to_goal_dist:
            dist_diff_ball_goal = self.last_ball_to_goal_dist[car_id] - dist_ball_to_goal
            # Normalize diff
            norm_diff_ball_goal = dist_diff_ball_goal / 100.0
            reward += self.ball_to_goal_pot_weight * norm_diff_ball_goal
            
        self.last_ball_to_goal_dist[car_id] = dist_ball_to_goal

        # 4. Ball Velocity toward Opponent Goal (Continuous Vector Projection)
        # We keep this as an instantaneous measure, as it directly mirrors physical momentum transfer.
        vec_to_goal = target_goal - state.ball.position
        if dist_ball_to_goal > 0:
            norm_to_goal = vec_to_goal / dist_ball_to_goal
            vel_to_goal = np.dot(state.ball.linear_velocity, norm_to_goal) / self.MAX_VEL
            reward += self.vel_ball_to_goal_weight * vel_to_goal

        return reward

class Phase2Reward(RewardFunction):
    def __init__(self, step_penalty=-0.0001):
        super().__init__()
        self.step_penalty = step_penalty
        
    def reset(self, initial_state: GameState):
        pass
        
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Step penalty to encourage speed
        reward = self.step_penalty
        
        # Goals are handled by EventReward usually, but can be added here
        if state.ball.position[1] > 5120:  # Orange Goal
            reward += 1.0 if player.team_num == 0 else -1.0
        elif state.ball.position[1] < -5120: # Blue Goal
            reward += 1.0 if player.team_num == 1 else -1.0
            
        return reward