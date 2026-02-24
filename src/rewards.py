import numpy as np
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

class Phase1Reward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.touch_weight = 5.0
        self.vel_ball_to_goal_weight = 3.0
        self.ball_to_goal_proximity_weight = 1.0
        self.close_to_ball_weight = 0.5
        self.face_ball_weight = 0.3
        self.vel_toward_ball_weight = 0.5

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0

        # 1. Touch ball
        if player.ball_touched:
            reward += self.touch_weight

        # 2. Ball velocity toward opponent goal
        goal_y = 5120.0 if player.team_num == 0 else -5120.0
        goal_pos = np.array([0, goal_y, 0])
        vec_to_goal = goal_pos - state.ball.position
        dist_to_goal = np.linalg.norm(vec_to_goal)
        if dist_to_goal > 0:
            norm_to_goal = vec_to_goal / dist_to_goal
            vel_to_goal = np.dot(state.ball.linear_velocity, norm_to_goal) / 2300.0
            reward += self.vel_ball_to_goal_weight * vel_to_goal

        # 3. Ball proximity to opponent goal (linear potential)
        if player.team_num == 0:
            ball_progress = state.ball.position[1] / 5120.0
        else:
            ball_progress = -state.ball.position[1] / 5120.0
        reward += self.ball_to_goal_proximity_weight * ball_progress

        # 4. Proximity to ball (exponential decay)
        vec_to_ball = state.ball.position - player.car_data.position
        dist_to_ball = np.linalg.norm(vec_to_ball)
        reward += self.close_to_ball_weight * np.exp(-dist_to_ball / 1500.0)

        # 5. Face ball
        car_forward = player.car_data.forward()
        if dist_to_ball > 0:
            norm_to_ball = vec_to_ball / dist_to_ball
            face_dot = np.dot(car_forward, norm_to_ball)
            reward += self.face_ball_weight * face_dot

        # 6. Velocity toward ball
        car_vel = player.car_data.linear_velocity
        if dist_to_ball > 0:
            vel_toward_ball = np.dot(car_vel, vec_to_ball / dist_to_ball) / 2300.0
            reward += self.vel_toward_ball_weight * max(vel_toward_ball, 0.0)

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