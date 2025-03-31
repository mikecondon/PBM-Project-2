
"""
Created on Wed Mar 30 02:09:10 2025

@author: ssr17
"""

import numpy as np

class HMMEnvironment:
    def __init__(self, transition_matrix, reward_probs):

        self.transition_matrix = transition_matrix
        self.reward_probs = reward_probs
        self.state = np.random.choice([0, 1])  

    def step(self, action):


        self.state = np.random.choice([0, 1], p=self.transition_matrix[self.state])


        reward = np.random.rand() < self.reward_probs[self.state][action]
        return reward, self.state
class BayesianAgent:
    def __init__(self, prior=[0.5, 0.5], transition_matrix=None, reward_probs=None):
        self.belief = np.array(prior)  
        self.transition_matrix = transition_matrix
        self.reward_probs = reward_probs

    def update_belief(self, action, reward):

        #likelihood: P(reward | state, action)
        likelihood = np.array([
            self.reward_probs[0][action] if reward else (1 - self.reward_probs[0][action]),
            self.reward_probs[1][action] if reward else (1 - self.reward_probs[1][action])
        ])

        prior_belief = self.transition_matrix.T @ self.belief

       
        posterior_unnormalized = likelihood * prior_belief
        self.belief = posterior_unnormalized / np.sum(posterior_unnormalized)  # Normalize

    def choose_action(self, beta=5):

        action_prob = np.exp(beta * self.belief) / np.sum(np.exp(beta * self.belief))
        return np.random.choice([0, 1], p=action_prob)  
transition_matrix = np.array([[0.9, 0.1],  
                              [0.1, 0.9]]) 
reward_probs = np.array([[0.8, 0.2],  
                          [0.2, 0.8]]) 
#environment and agent
env = HMMEnvironment(transition_matrix, reward_probs)
agent = BayesianAgent(transition_matrix=transition_matrix, reward_probs=reward_probs)

#simulation
num_trials = 100
actions = []
beliefs = []
rewards = []

for t in range(num_trials):
    action = agent.choose_action()
    reward, _ = env.step(action)
    
    # Update belief with new observation
    agent.update_belief(action, reward)
    
    # Store data
    actions.append(action)
    beliefs.append(agent.belief.copy())
    rewards.append(reward)


print("Final belief:", agent.belief)
import matplotlib.pyplot as plt

beliefs = np.array(beliefs)

plt.plot(beliefs[:, 0], label="P(Left is better)")
plt.plot(beliefs[:, 1], label="P(Right is better)")
plt.xlabel("Trial")
plt.ylabel("Belief Probability")
plt.title("Bayesian Belief Updates")
plt.legend()
plt.show()
