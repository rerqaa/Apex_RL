import numpy as np
from rlgym_sim.utils.reward_functions import RewardFunction, CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import (
    EventReward, 
    VelocityPlayerToBallReward, 
    VelocityBallToGoalReward,
    TouchBallReward
)
from rlgym_sim.utils.gamestates import GameState, PlayerData

def build_phase_1_reward():
    """Builds a mathematically stable reward for Phase 1 (striker) training."""
    # 1. Drive to the ball
    to_ball = VelocityPlayerToBallReward()
    
    # 2. Hit the ball
    touch = TouchBallReward()
    
    # 3. Hit the ball towards the goal
    to_goal = VelocityBallToGoalReward()
    
    # Structure the combined reward
    # High weight on touching the ball and objective scoring
    # Lower weight on continuous velocity rewards to prevent magnitude imbalance
    reward_fn = CombinedReward.from_zipped(
        (to_ball, 0.5),
        (touch, 1.0),
        (to_goal, 1.0)
    )
    return reward_fn

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
