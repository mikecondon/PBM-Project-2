import numpy as np
import matplotlib.pyplot as plt

# ======================== ENVIRONMENT ========================
class TwoArmedBanditEnv:
    def __init__(self, reward_probs=[[0.8, 0.2], [0.2, 0.8]], switch_prob=0.02):
        self.reward_probs = np.array(reward_probs)
        self.switch_prob = switch_prob
        self.state = 0  # 0 = left high, 1 = right high

    def step(self, action):
        if np.random.rand() < self.switch_prob:
            self.state = 1 - self.state
        reward = np.random.rand() < self.reward_probs[self.state, action]
        return reward, self.state

# ======================== MODELS ========================
class RFLRAgent:
    def __init__(self, alpha=1.0, beta=1.5, tau=2.0):
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.phi = 0.0
        self.last_action = None

    def update(self, action, reward):
        c_t = 2 * action - 1
        self.phi = self.beta * c_t * reward + np.exp(-1 / self.tau) * self.phi
        self.last_action = c_t

    def get_log_odds(self):
        return self.alpha * self.last_action + self.phi if self.last_action is not None else 0.0

class HMMAgent:
    def __init__(self, transition_prob=0.02, reward_prob=0.8, policy="thompson"):
        self.transition_matrix = np.array([
            [1 - transition_prob, transition_prob],
            [transition_prob, 1 - transition_prob]
        ])
        self.reward_probs = np.array([
            [reward_prob, 1 - reward_prob],
            [1 - reward_prob, reward_prob]
        ])
        self.belief = np.array([0.5, 0.5])
        self.policy = policy

    def update(self, action, reward):
        likelihood = np.array([
            self.reward_probs[0, action] if reward else 1 - self.reward_probs[0, action],
            self.reward_probs[1, action] if reward else 1 - self.reward_probs[1, action]
        ])
        prior = self.transition_matrix.T @ self.belief
        self.belief = likelihood * prior
        self.belief /= np.sum(self.belief)

    def get_log_odds(self):
        return np.log(self.belief[0] / self.belief[1])

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, beta=5.0):
        self.Q = np.zeros(2)
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def update(self, action, reward):
        self.Q[action] += self.alpha * (reward - self.Q[action])

    def get_action_prob(self):
        exp_q = np.exp(self.beta * self.Q)
        return exp_q / np.sum(exp_q)

    def get_log_odds(self):
        p_left = self.get_action_prob()[0]
        p_left = np.clip(p_left, 1e-4, 1 - 1e-4)
        return np.log(p_left / (1 - p_left))

# ======================== SIMULATION ========================
def simulate_figure6a(num_trials=500):
    np.random.seed(42)

    env = TwoArmedBanditEnv()
    mouse_actions, mouse_rewards, mouse_states = [], [], []
    last_reward = None

    for t in range(num_trials):
        if t == 0 or last_reward is None:
            action = np.random.choice([0, 1])
        else:
            if last_reward:
                action = mouse_actions[-1] if np.random.rand() < 0.8 else 1 - mouse_actions[-1]
            else:
                action = 1 - mouse_actions[-1] if np.random.rand() < 0.6 else mouse_actions[-1]

        reward, state = env.step(action)
        mouse_actions.append(action)
        mouse_rewards.append(reward)
        mouse_states.append(state)
        last_reward = reward

    # Models
    rflr = RFLRAgent(alpha=1.0, beta=1.5, tau=2.0)
    sticky_hmm = HMMAgent(policy="thompson")
    ideal_hmm = HMMAgent(policy="greedy")
    q_agent = QLearningAgent(alpha=0.1, gamma=0.9, beta=5.0)

    log_rflr, log_hmm, log_ideal, log_q = [], [], [], []

    for action, reward in zip(mouse_actions, mouse_rewards):
        rflr.update(action, reward)
        sticky_hmm.update(action, reward)
        ideal_hmm.update(action, reward)
        q_agent.update(action, reward)

        log_rflr.append(rflr.get_log_odds())
        log_hmm.append(sticky_hmm.get_log_odds())
        log_ideal.append(ideal_hmm.get_log_odds())
        log_q.append(q_agent.get_log_odds())

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.plot(log_rflr, color='orange', linewidth=2, label='RFLR')
    plt.plot(log_hmm, color='black', linewidth=2, label='Sticky HMM (Thompson)')
    plt.plot(log_ideal, color='blue', linestyle='--', linewidth=1.5, label='Ideal HMM (Greedy)')
    plt.plot(log_q, color='green', linestyle='-', linewidth=2, label='Q-Learning')

    # Block transition line (approx mid-way)
    plt.axvline(num_trials // 2, color='gray', linestyle=':', label='Block transition')

    # Unrewarded trial indicators
    unrewarded = [i for i, r in enumerate(mouse_rewards) if not r]
    for trial in unrewarded:
        plt.axvline(trial, color='red', alpha=0.05, linewidth=0.5)

    # ±α stickiness bounds
    plt.axhline(rflr.alpha, linestyle='--', color='orange', alpha=0.3)
    plt.axhline(-rflr.alpha, linestyle='--', color='orange', alpha=0.3)

    plt.xlabel('Trial', fontsize=12)
    plt.ylabel('Log-odds (left vs right)', fontsize=12)
    plt.title('Figure 6A: Log-odds trajectories (RFLR, HMMs, Q-Learning)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

# Run it
simulate_figure6a()
