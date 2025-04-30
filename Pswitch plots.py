# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:12:09 2025

@author: ssr17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ======= Load your real mouse data =======
# Replace "your_data.csv" with the actual path to your CSV file
df = pd.read_csv("C:/Users/ssr17/Downloads/bandit_data.csv")
df = df.dropna(subset=["Decision", "Reward"])
actions = df["Decision"].astype(int).values
rewards = df["Reward"].astype(int).values


# ======= Models (Bayesian belief update with policies) =======
class BayesianAgent:
    def __init__(self, transition_prob=0.02, reward_probs=[[0.8, 0.2], [0.2, 0.8]], beta=1.0, kappa=0.0, policy_type="softmax"):
        self.transition_matrix = np.array([
            [1 - transition_prob, transition_prob],
            [transition_prob, 1 - transition_prob]
        ])
        self.reward_probs = np.array(reward_probs)
        self.belief = np.array([0.5, 0.5])  # Initial belief
        self.beta = beta  # Inverse temperature for softmax
        self.kappa = kappa  # Stickiness parameter
        self.last_action = None
        self.policy_type = policy_type

    def choose_action(self):
        if self.policy_type == "softmax":
            logits = self.beta * self.belief.copy()
            if self.kappa != 0 and self.last_action is not None:
                logits[self.last_action] += self.kappa
            probs = np.exp(logits) / np.sum(np.exp(logits))
            return np.random.choice([0, 1], p=probs)
        elif self.policy_type == "greedy":
            return np.argmax(self.belief)
        elif self.policy_type == "thompson":
            return np.random.choice([0, 1], p=self.belief)
        else:
            raise ValueError("Invalid policy type")

    def update(self, action, reward):
        prior = self.transition_matrix.T @ self.belief
        likelihood = np.array([
            self.reward_probs[0, action] if reward else 1 - self.reward_probs[0, action],
            self.reward_probs[1, action] if reward else 1 - self.reward_probs[1, action]
        ])
        posterior = likelihood * prior
        self.belief = posterior / np.sum(posterior)
        self.last_action = action

    def log_likelihood(self, actions, rewards):
        ll = 0
        self.reset()  # Reset belief state
        for a, r in zip(actions, rewards):
            probs = self.action_probabilities()  # Get action probabilities
            ll += np.log(probs[a])  # Log likelihood of chosen action
            self.update(a, r)
        return ll

    def action_probabilities(self):
        if self.policy_type == "softmax":
            logits = self.beta * self.belief.copy()
            if self.kappa != 0 and self.last_action is not None:
                logits[self.last_action] += self.kappa
            probs = np.exp(logits) / np.sum(np.exp(logits))
            return probs
        elif self.policy_type in ["greedy", "thompson"]:
            # For greedy and Thompson, return belief as proxy for probabilities
            #  (Thompson is literally using belief as probability)
            return self.belief
        else:
            raise ValueError("Invalid policy type")

    def reset(self):
        self.belief = np.array([0.5, 0.5])
        self.last_action = None


# ======= Run agent on real data (observed actions, model predictions) =======
def run_model_on_data(actions, rewards, kappa, policy_type="softmax"):
    agent = BayesianAgent(kappa=kappa, policy_type=policy_type)
    predicted_actions = []
    for a, r in zip(actions, rewards):
        predicted = agent.choose_action()
        predicted_actions.append(predicted)
        agent.update(a, r)
    return np.array(predicted_actions)


# ======= Encode histories and compute P(switch) =======
def encode_and_count(actions, rewards):
    def label(prev, curr, reward):
        if prev == curr:
            return 'A' if reward else 'a'
        else:
            return 'B' if reward else 'b'

    history_counts = defaultdict(lambda: [0, 0])
    for i in range(3, len(actions) - 1):
        hist = ''.join(label(actions[i - j - 1], actions[i - j], rewards[i - j]) for j in reversed(range(3)))
        switch = actions[i + 1] != actions[i]
        history_counts[hist][1] += 1
        if switch:
            history_counts[hist][0] += 1
    return history_counts


# ======= Plotting function =======
def plot_pswitch(history_counts, title):
    histories = sorted(history_counts.keys(),
                      key=lambda x: history_counts[x][0] / history_counts[x][1] if history_counts[x][1] > 0 else 0)
    pswitch = [history_counts[h][0] / history_counts[h][1] for h in histories]
    counts = [history_counts[h][1] for h in histories]
    errors = [np.sqrt(p * (1 - p) / n) if n > 0 else 0 for p, n in zip(pswitch, counts)]

    plt.figure(figsize=(16, 6))
    plt.bar(histories, pswitch, yerr=errors, color='gray', edgecolor='black')
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.ylabel("P(switch)")
    plt.title(title)
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.show()


# ======= Run and Plot =======

# 1. Original mouse behavior
real_history = encode_and_count(actions, rewards)
plot_pswitch(real_history, "Real Mouse Behavior: P(switch) by 3-Trial History")

# 2. Non-sticky Bayesian model
model_ns_actions = run_model_on_data(actions, rewards, kappa=0.0, policy_type="softmax")
model_ns_history = encode_and_count(model_ns_actions, rewards)
plot_pswitch(model_ns_history, "Non-Sticky Bayesian Model: P(switch) by 3-Trial History")

# 3. Sticky Bayesian model
model_s_actions = run_model_on_data(actions, rewards, kappa=2.0, policy_type="softmax")
model_s_history = encode_and_count(model_s_actions, rewards)
plot_pswitch(model_s_history, "Sticky Bayesian Model: P(switch) by 3-Trial History")

# ======= Example of Log Likelihood Calculation =======
agent = BayesianAgent(kappa=2.0, policy_type="softmax")  # Example: sticky softmax
log_likelihood = agent.log_likelihood(actions, rewards)
print(f"Log Likelihood: {log_likelihood}")