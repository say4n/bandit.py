#! /usr/bin/env python3

from agent import RandomAgent, GreedyAgent, EpsilonGreedyAgent, UpperConfidenceBoundAgent
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
        agent = RandomAgent(arms=NUM_ARMS, turns=NUM_STEPS)
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
        agent = GreedyAgent(arms=NUM_ARMS, turns=NUM_STEPS)
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
    for g in tqdm(range(NUM_GAMES), desc=f"𝜀-Greedy Agent ({epsilon})"):
        agent = EpsilonGreedyAgent(epsilon=epsilon, arms=NUM_ARMS, turns=NUM_STEPS)
        game = games[g]

        actions, rewards = agent.play(game)

        data["actions"].extend(actions)
        data["rewards"] += rewards

    # Convert sum to average reward per step.
    data["rewards"] /= NUM_GAMES

    return data

def simulate_upper_confidence_bound(games, c):
    data = {
        "actions": [],
        "rewards": np.zeros(NUM_STEPS)
    }
    for g in tqdm(range(NUM_GAMES), desc=f"UCB ({c})"):
        agent = UpperConfidenceBoundAgent(c=c, arms=NUM_ARMS, turns=NUM_STEPS)
        game = games[g]

        actions, rewards = agent.play(game)

        data["actions"].extend(actions)
        data["rewards"] += rewards

    # Convert sum to average reward per step.
    data["rewards"] /= NUM_GAMES

    return data


if __name__ == "__main__":
    games = [Bandit(NUM_ARMS) for _ in range(NUM_GAMES)]
    show_violin = False

    if show_violin:
        reward_distribution = [[] for _ in range(NUM_ARMS)]

        for game in games:
            rewards = game.get_rewards()
            for i in range(NUM_ARMS):
                reward_distribution[i].append(rewards[i])


        plt.violinplot(reward_distribution, range(NUM_ARMS), showmeans=True)
        plt.xticks(range(NUM_ARMS))
        plt.xlabel("Actions")
        plt.ylabel("Reward Distribution")
        plt.show()

    r_data = simulate_random_agent(games)
    g_data = simulate_greedy_agent(games)
    e_data_1 = simulate_epsilon_greedy_agent(games, 0.1)
    e_data_2 = simulate_epsilon_greedy_agent(games, 0.01)
    u_data_1 = simulate_upper_confidence_bound(games, 1)
    u_data_2 = simulate_upper_confidence_bound(games, 2)

    timesteps = range(NUM_STEPS)

    plt.plot(timesteps, r_data["rewards"], color="black", linewidth=0.5)
    plt.plot(timesteps, g_data["rewards"], color="green", linewidth=0.5)
    plt.plot(timesteps, e_data_1["rewards"], color="blue", linewidth=0.5)
    plt.plot(timesteps, e_data_2["rewards"], color="red", linewidth=0.5)
    plt.plot(timesteps, u_data_1["rewards"], color="cyan", linewidth=0.5)
    plt.plot(timesteps, u_data_2["rewards"], color="magenta", linewidth=0.5)

    plt.ylim(bottom=0)
    plt.legend(["Random", "Greedy", "$\epsilon$-Greedy (0.1)", "$\epsilon$-Greedy (0.01)", "UCB (1)", "UCB (2)"])
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")

    plt.show()
