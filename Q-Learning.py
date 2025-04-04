import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

# ======================== ENVIRONMENT ========================
class TwoArmedBanditEnv:
    def __init__(self, reward_probs=[[0.8, 0.2], [0.2, 0.8]], switch_prob=0.02):
        self.reward_probs = np.array(reward_probs)  # [[left_high], [right_high]]
        self.switch_prob = switch_prob
        self.state = 0  # 0 = left is high, 1 = right is high

    def step(self, action):
        # State transition (Markovian)
        if np.random.rand() < self.switch_prob:
            self.state = 1 - self.state
        # Reward
        reward = np.random.rand() < self.reward_probs[self.state, action]
        return reward, self.state

# ======================== MODELS ========================
class RFLRAgent:
    """Recursively Formulated Logistic Regression (Eq. 6 in paper)"""
    def __init__(self, alpha=0.3, beta=2.0, tau=1.43):
        self.alpha = alpha  # Stickiness weight
        self.beta = beta    # Evidence weight
        self.tau = tau      # Decay time constant
        self.phi = 0.0      # Evidence accumulator
        self.last_action = None  # ±1 encoding

    def update(self, action, reward):
        c_t = 2 * action - 1  # Convert to ±1 (left=+1, right=-1)
        self.phi = self.beta * c_t * (reward - 0.5) + np.exp(-1/self.tau) * self.phi
        self.last_action = c_t

    def get_log_odds(self):
        """Returns ψ_{t+1} = α * c_t + φ_t (Eq. 6)"""
        return self.alpha * self.last_action + self.phi if self.last_action is not None else 0.0

class HMMAgent:
    """Bayesian agent performing HMM inference (Sec. Results)"""
    def __init__(self, transition_prob=0.02, reward_prob=0.8, policy="thompson"):
        # Transition matrix (world dynamics)
        self.transition_matrix = np.array([
            [1 - transition_prob, transition_prob],
            [transition_prob, 1 - transition_prob]
        ])
        # Emission probabilities (reward likelihoods)
        self.reward_probs = np.array([
            [reward_prob, 1 - reward_prob],  # State 0: left=high
            [1 - reward_prob, reward_prob]   # State 1: right=high
        ])
        self.belief = np.array([0.5, 0.5])  # P(state = left)
        self.policy = policy

    def update(self, action, reward):
        # Likelihood: P(reward | action, state)
        likelihood = np.array([
            self.reward_probs[0, action] if reward else 1 - self.reward_probs[0, action],
            self.reward_probs[1, action] if reward else 1 - self.reward_probs[1, action]
        ])
        # Prior: P(state | previous belief)
        prior = self.transition_matrix.T @ self.belief
        # Posterior
        self.belief = likelihood * prior
        self.belief /= np.sum(self.belief)  # Normalize

    def get_log_odds(self):
        """Convert belief to log-odds (ln P(left)/P(right))"""
        return 3 * np.log(self.belief[0] / self.belief[1])

# ======================== SIMULATION ========================
def simulate_figure6a(num_trials=500):
    np.random.seed(42)
    
    # 1. First generate mouse-like behavior
    env = TwoArmedBanditEnv()
    mouse_actions = []
    mouse_rewards = []
    mouse_states = []
    
    # Simple mouse policy (win-stay/lose-shift with noise)
    last_reward = None
    for t in range(num_trials):
        if t == 0 or last_reward is None:
            action = np.random.choice([0,1])
        else:
            # 80% repeat if rewarded, 60% switch if unrewarded
            if last_reward:
                action = mouse_actions[-1] if np.random.rand() < 0.8 else 1 - mouse_actions[-1]
            else:
                action = 1 - mouse_actions[-1] if np.random.rand() < 0.6 else mouse_actions[-1]
        
        reward, state = env.step(action)
        mouse_actions.append(action)
        mouse_rewards.append(reward)
        mouse_states.append(state)
        last_reward = reward
    
    # 2. Feed this history to models
    rflr = RFLRAgent(alpha=1.0, beta=1.5, tau=2.0)
    sticky_hmm = HMMAgent(policy="thompson")
    ideal_hmm = HMMAgent(policy="greedy")
    
    log_rflr, log_hmm, log_ideal = [], [], []
    
    for action, reward in zip(mouse_actions, mouse_rewards):
        # Update models with observed actions/rewards
        rflr.update(action, reward)
        sticky_hmm.update(action, reward)
        ideal_hmm.update(action, reward)
        
        log_rflr.append(rflr.get_log_odds())
        log_hmm.append(sticky_hmm.get_log_odds())
        log_ideal.append(ideal_hmm.get_log_odds())
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(log_rflr, color='orange', linewidth=2, label='RFLR')
    plt.plot(log_hmm, color='black', linewidth=2, label='Sticky HMM (Thompson)')
    plt.plot(log_ideal, color='blue', linestyle='--', linewidth=1.5, label='Ideal HMM (Greedy)')

    # Highlight block transition and unrewarded trials
    plt.axvline(num_trials // 2, color='gray', linestyle=':', label='Block transition')
    unrewarded = np.where(np.logical_not(reward))[0]
    for trial in unrewarded:
        plt.axvline(trial, color='red', alpha=0.1, linewidth=0.5)

    # Horizontal lines at ±α (RFLR's stickiness bounds)
    plt.axhline(0.3, color='orange', linestyle='--', alpha=0.3)
    plt.axhline(-0.3, color='orange', linestyle='--', alpha=0.3)

    plt.xlabel('Trial', fontsize=12)
    plt.ylabel('Log-odds (left vs right)', fontsize=12)
    plt.title('Figure 6A: Log-odds trajectories (RFLR vs HMM)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.2)
    plt.show()

# Run
simulate_figure6a()