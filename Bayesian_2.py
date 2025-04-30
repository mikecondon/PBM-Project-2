# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:34:01 2025

@author: ssr17
"""

import numpy as np
import matplotlib.pyplot as plt

# ========== ENVIRONMENT: Switching 2-Armed Bandit ==========
class BanditEnvironment:
    def __init__(self, reward_probs=[[0.8, 0.2], [0.2, 0.8]], switch_prob=0.05):
        self.reward_probs = np.array(reward_probs)
        self.switch_prob = switch_prob
        self.state = 0  

    def step(self, action):
        # Randomly switch the hidden state
        if np.random.rand() < self.switch_prob:
            self.state = 1 - self.state
        reward = np.random.rand() < self.reward_probs[self.state, action]
        return reward, self.state

# ========== BAYESIAN AGENT ==========
class BayesianAgent:
    def __init__(self, prior=[0.5, 0.5], transition_prob=0.05, reward_probs=[[0.8, 0.2], [0.2, 0.8]]):
        self.belief = np.array(prior)
        self.transition_matrix = np.array([
            [1 - transition_prob, transition_prob],
            [transition_prob, 1 - transition_prob]
        ])
        self.reward_probs = np.array(reward_probs)

    def update(self, action, reward):
        # Prior belief update via transition matrix
        prior = self.transition_matrix.T @ self.belief

        # Likelihood of observed reward
        likelihood = np.array([
            self.reward_probs[0, action] if reward else 1 - self.reward_probs[0, action],
            self.reward_probs[1, action] if reward else 1 - self.reward_probs[1, action]
        ])

        # Bayesian update
        unnormalized_posterior = likelihood * prior
        self.belief = unnormalized_posterior / np.sum(unnormalized_posterior)

    def choose_action(self, beta=5):
        # Softmax over belief
        probs = np.exp(beta * self.belief) / np.sum(np.exp(beta * self.belief))
        return np.random.choice([0, 1], p=probs)

# ========== SIMULATION ==========
def run_simulation(num_trials=300, switch_prob=0.05):
    env = BanditEnvironment(switch_prob=switch_prob)
    agent = BayesianAgent(transition_prob=switch_prob)

    beliefs = []
    actions = []
    rewards = []
    states = []

    for t in range(num_trials):
        action = agent.choose_action()
        reward, true_state = env.step(action)

        agent.update(action, reward)

        beliefs.append(agent.belief.copy())
        actions.append(action)
        rewards.append(reward)
        states.append(true_state)

    return np.array(beliefs), np.array(actions), np.array(rewards), np.array(states)

# ========== PLOTTING ==========
def plot_results(beliefs, actions, rewards, states):
    plt.figure(figsize=(14, 5))

    plt.plot(beliefs[:, 0], label='P(State 0 | data)', color='blue')
    plt.plot(beliefs[:, 1], label='P(State 1 | data)', color='orange')
    plt.plot(states, '--', color='gray', alpha=0.4, label='True State (0/1)')

    plt.scatter(np.arange(len(actions)), actions, marker='o', color='green', s=10, label='Actions')
    plt.scatter(np.arange(len(rewards)), rewards, marker='x', color='red', s=10, label='Rewards')

    plt.xlabel('Trial')
    plt.ylabel('Probability / Value')
    plt.title('Bayesian Agent in Switching Bandit Task')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ========== RUN ==========
beliefs, actions, rewards, states = run_simulation()
plot_results(beliefs, actions, rewards, states)
