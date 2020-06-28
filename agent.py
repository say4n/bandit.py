from collections import defaultdict
import numpy as np
import random


class Agent:
    def __init__(self, arms=10, turns=1000):
        self.arms = arms
        self.turns = turns

    def play(self, game):
        raise NotImplementedError

    def __repr__(self):
        return "Abstract Base Class Agent"


class RandomAgent(Agent):
    def __init__(self, arms=10, turns=1000):
        super().__init__(arms, turns)

    def play(self, game):
        actions, rewards = [], []

        for _ in range(self.turns):
            action = np.random.choice(game.get_action_space())
            reward = game.step(action)

            actions.append(action)
            rewards.append(reward)

        return actions, rewards

    def __repr__(self):
        return "Random Agent"


class GreedyAgent(Agent):
    def __init__(self, arms=10, turns=1000):
        super().__init__(arms, turns)

        self.action_value = np.zeros(arms)

    def play(self, game):
        actions, rewards = [], []

        action_tracker = defaultdict(int)

        for _ in range(self.turns):
            action = np.argmax(self.action_value)
            reward = game.step(action)

            action_tracker[action] += 1

            self.action_value[action] = self.action_value[action] +\
                (reward - self.action_value[action]) / action_tracker[action]

            actions.append(action)
            rewards.append(reward)

        return actions, rewards

    def get_action_values(self):
        return self.action_value

    def __repr__(self):
        return "Greedy Agent"


class EpsilonGreedyAgent(Agent):
    def __init__(self, epsilon=0.1, arms=10, turns=1000):
        super().__init__(arms, turns)

        self.action_value = np.zeros(arms)
        self.epsilon = epsilon

    def play(self, game):
        actions, rewards = [], []

        action_tracker = defaultdict(int)

        for _ in range(self.turns):
            if random.random() < self.epsilon:
                # Take random action.
                action = np.random.choice(game.get_action_space())
                reward = game.step(action)
            else:
                # Take greedy action.
                action = np.argmax(self.action_value)
                reward = game.step(action)

            action_tracker[action] += 1

            self.action_value[action] = self.action_value[action] +\
                (reward - self.action_value[action]) / action_tracker[action]

            actions.append(action)
            rewards.append(reward)

        return actions, rewards

    def get_action_values(self):
        return self.action_value

    def __repr__(self):
        return f"Epsilon-Greedy Agent ({self.epsilon})"


class UpperConfidenceBoundAgent(Agent):
    def __init__(self, c=1, arms=10, turns=1000):
        super().__init__(arms, turns)

        self.action_value = np.zeros(arms)
        self.c = c

    def play(self, game):
        actions, rewards = [], []

        action_tracker = np.zeros(self.arms)

        # Stop numpy from complaining about NaN and zero division.
        np.seterr(divide='ignore', invalid='ignore')

        for turn in range(self.turns):
            q = self.action_value + self.c * np.sqrt(np.log(turn, dtype=np.float64)/action_tracker)

            action = np.argmax(q)
            reward = game.step(action)

            action_tracker[action] += 1

            self.action_value[action] = self.action_value[action] +\
                (reward - self.action_value[action]) / action_tracker[action]

            actions.append(action)
            rewards.append(reward)

        # Resume complaining about NaN and zero division.
        np.seterr(divide='warn', invalid='warn')

        return actions, rewards

    def get_action_values(self):
        return self.action_value

    def __repr__(self):
        return f"UCB Agent ({self.c})"
