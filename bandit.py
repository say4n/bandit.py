import numpy as np


class Bandit:
    def __init__(self, k=10, mean=0, variance=1, reward_variance=1):
        self.k = k
        self.mean = mean
        self.variance = variance
        self.reward_variance = variance

        self.rewards = np.random.normal(self.mean, self.variance, self.k)

    def step(self, action):
        if not 0 <= action < self.k:
            raise IndexError("Select a valid action!")

        return np.random.normal(self.rewards[action], self.reward_variance)

    def get_action_space(self):
        return np.arange(self.k)

    def __repr__(self):
        return f"k-Armed Bandit (Stochastic, Stationary)"
