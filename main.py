#! /usr/bin/env python3

from agent import RandomAgent, GreedyAgent, EpsilonGreedyAgent
from bandit import Bandit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

NUM_ARMS = 10
NUM_GAMES = 2000
NUM_STEPS = 1000


def simulate(agent):
    data = {
        "actions": [],
        "rewards": np.zeros(NUM_STEPS)
    }
    for _ in tqdm(range(NUM_GAMES), desc=str(agent)):
        game = Bandit(NUM_ARMS)
        actions, rewards = agent.play(game)

        data["actions"].extend(actions)
        data["rewards"] += rewards

    # Convert sum to average reward per step.
    data["rewards"] /= NUM_GAMES

    return data



if __name__ == "__main__":
    agent = RandomAgent(NUM_ARMS, NUM_STEPS)
    r_data = simulate(agent)

    agent = GreedyAgent(NUM_ARMS, NUM_STEPS)
    g_data = simulate(agent)

    agent = EpsilonGreedyAgent(0.1, NUM_ARMS, NUM_STEPS)
    e_data_1 = simulate(agent)

    agent = EpsilonGreedyAgent(0.01, NUM_ARMS, NUM_STEPS)
    e_data_2 = simulate(agent)

    timesteps = range(NUM_STEPS)

    plt.plot(timesteps, r_data["rewards"], color="black", linewidth=0.5)
    plt.plot(timesteps, g_data["rewards"], color="green", linewidth=0.5)
    plt.plot(timesteps, e_data_1["rewards"], color="blue", linewidth=0.5)
    plt.plot(timesteps, e_data_2["rewards"], color="red", linewidth=0.5)

    plt.ylim(bottom=0)
    plt.legend(["Random", "Greedy", "Epsilon Greedy (0.1)", "Epsilon Greedy (0.01)"])
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")

    plt.show()
