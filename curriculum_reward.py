from rlgym_sim.utils.reward_functions import RewardFunction

from rlgym_sim.utils.gamestates import GameState, PlayerData

from numpy import ndarray

class SequentialRewards(RewardFunction):
    """ 
    A simple reward class that allows you to transition from one reward to the next at set intervals.
    Example: rewards, step_requirements = [reward1, reward2, reward3, etc], [10_000_000, 20_000_000, 30_000_000, etc]
    my_rewards = SequentialRewards(rewards, step_requirements)   
    """
    def __init__(self, rewards: list, steps: list):
        super().__init__()
        self.rewards_list = rewards
        self.step_counts = steps
        self.step_count = 0
        self.step_index = 0
        assert len(self.rewards_list) == len(self.step_counts)

    def reset(self, initial_state: GameState):
        for rew in self.rewards_list:
            rew.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:
        if self.step_index < len(self.step_counts) and self.step_count > self.step_counts[self.step_index]:
            self.step_index += 1

        self.step_count += 1
        return self.rewards_list[self.step_index].get_reward(player, state, previous_action)
    
import math
from typing import Union

import numpy as np

from rlgym_sim.utils.reward_functions.common_rewards import ConstantReward


class _DummyReward(RewardFunction):
    def reset(self, initial_state: GameState): pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float: return 0


class AnnealRewards(RewardFunction):
    """
    Smoothly transitions between reward functions sequentially.

    Example:
        AnnealRewards(rew1, 10_000, rew2, 100_000, rew3)
        will start by only rewarding rew1, then linearly transitions until 10_000 steps, where only rew2 is counted.
        It then transitions between rew2 and rew3, and after 100_000 total steps and further, only rew3 is rewarded.
    """
    STEP = 0
    TOUCH = 1
    GOAL = 2

    def __init__(self, *alternating_rewards_steps: Union[RewardFunction, int],
                 mode: int = STEP, initial_count: int = 0):
        """

        :param alternating_rewards_steps: an alternating sequence of (RewardFunction, int, RewardFunction, int, ...)
                                          specifying reward functions, and the steps at which to transition.
        :param mode: specifies whether to increment counter on steps, touches or goals.
        :param initial_count: the count to start reward calculations at.
        """
        self.rewards_steps = list(alternating_rewards_steps) + [float("inf"), _DummyReward()]
        assert mode in (self.STEP, self.TOUCH, self.GOAL)
        self.mode = mode

        self.last_goals = 0
        self.current_goals = 0

        self.last_transition_step = 0
        self.last_reward = self.rewards_steps.pop(0)
        self.next_transition_step = self.rewards_steps.pop(0)
        self.next_reward = self.rewards_steps.pop(0)
        self.count = initial_count
        self.last_state = None

    def reset(self, initial_state: GameState):
        self.last_reward.reset(initial_state)
        self.next_reward.reset(initial_state)
        self.last_state = None
        while self.next_transition_step < self.count:  # If initial_count is set, find the right rewards
            self._transition(initial_state)

    def _transition(self, state):
        self.last_transition_step = self.next_transition_step
        self.last_reward = self.next_reward
        self.next_transition_step = self.rewards_steps.pop(0)
        self.next_reward = self.rewards_steps.pop(0)
        self.next_reward.reset(state)  # Make sure initial values are set

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        args = player, state, previous_action
        if math.isinf(self.next_transition_step):
            return self.last_reward.get_reward(*args)

        if state != self.last_state:
            if self.mode == self.STEP:
                self.count += 1
            elif self.mode == self.TOUCH and player.ball_touched:
                self.count += 1
            elif self.mode == self.GOAL:
                self.last_goals = self.current_goals
                self.current_goals = state.blue_score + state.orange_score
                if self.current_goals > self.last_goals:
                    self.count += 1
            self.last_state = state

        frac = (self.count - self.last_transition_step) / (self.next_transition_step - self.last_transition_step)
        rew = frac * self.next_reward.get_reward(*args) + (1 - frac) * self.last_reward.get_reward(*args)

        if self.count >= self.next_transition_step:
            self._transition(state)

        return rew