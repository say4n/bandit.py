#! /usr/bin/env python3

from agent import RandomAgent, GreedyAgent, EpsilonGreedyAgent
from bandit import Bandit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

NUM_ARMS = 10
NUM_GAMES = 2000
NUM_STEPS = 1000


def simulate_random_agent(games):
    data = {
        "actions": [],
        "rewards": np.zeros(NUM_STEPS)
    }
    for g in tqdm(range(NUM_GAMES), desc="Random Agent"):
        agent = RandomAgent(NUM_ARMS, NUM_STEPS)
        game = games[g]

        actions, rewards = agent.play(game)

        data["actions"].extend(actions)
        data["rewards"] += rewards

    # Convert sum to average reward per step.
    data["rewards"] /= NUM_GAMES

    return data

def simulate_greedy_agent(games):
    data = {
        "actions": [],
        "rewards": np.zeros(NUM_STEPS)
    }
    for g in tqdm(range(NUM_GAMES), desc="Greedy Agent"):
        agent = GreedyAgent(NUM_ARMS, NUM_STEPS)
        game = games[g]

        actions, rewards = agent.play(game)

        data["actions"].extend(actions)
        data["rewards"] += rewards

    # Convert sum to average reward per step.
    data["rewards"] /= NUM_GAMES

    return data

def simulate_epsilon_greedy_agent(games, epsilon):
    data = {
        "actions": [],
        "rewards": np.zeros(NUM_STEPS)
    }
    for g in tqdm(range(NUM_GAMES), desc=f"Epsilon Greedy Agent ({epsilon})"):
        agent = EpsilonGreedyAgent(epsilon, NUM_ARMS, NUM_STEPS)
        game = games[g]

        actions, rewards = agent.play(game)

        data["actions"].extend(actions)
        data["rewards"] += rewards

    # Convert sum to average reward per step.
    data["rewards"] /= NUM_GAMES

    return data



if __name__ == "__main__":
    games = [Bandit(NUM_ARMS) for _ in range(NUM_GAMES)]

    r_data = simulate_random_agent(games)
    g_data = simulate_greedy_agent(games)
    e_data_1 = simulate_epsilon_greedy_agent(games, 0.1)
    e_data_2 = simulate_epsilon_greedy_agent(games, 0.01)

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
