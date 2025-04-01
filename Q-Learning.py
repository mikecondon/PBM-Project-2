import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

# ======================== MODELS ========================

class RFLRAgent:
    """Recursively Formulated Logistic Regression (Eq. 6)"""
    def __init__(self, alpha=0.3, beta=2.0, tau=1.43):
        self.alpha = alpha  # Stickiness weight
        self.beta = beta    # Evidence weight
        self.tau = tau      # Decay time constant
        self.phi = 0.0      # Evidence accumulator
        self.last_action = None

    def update(self, action, reward):
        # Encode action as ±1 (left=+1, right=-1)
        c_t = 2 * action - 1  
        # Update evidence (Eq. 5)
        self.phi = self.beta * c_t * reward + np.exp(-1/self.tau) * self.phi
        self.last_action = c_t

    def choose_action(self):
        # Log-odds (Eq. 6)
        psi = self.alpha * self.last_action + self.phi if self.last_action is not None else 0
        # Stochastic policy
        p_left = sigmoid(psi)
        return np.random.rand() < p_left

class HMMAgent:
    def __init__(self, transition_matrix, reward_probs, policy="thompson"):
        # Convert lists to NumPy arrays
        self.transition_matrix = np.array(transition_matrix)  # Fix: Use np.array
        self.reward_probs = np.array(reward_probs)
        self.belief = np.array([0.5, 0.5])
        self.policy = policy

    def update(self, action, reward):
        likelihood = np.array([
            self.reward_probs[0][action] if reward else 1-self.reward_probs[0][action],
            self.reward_probs[1][action] if reward else 1-self.reward_probs[1][action]
        ])
        prior = self.transition_matrix.T @ self.belief  # Now works!
        self.belief = likelihood * prior
        self.belief /= np.sum(self.belief)

    def choose_action(self):
        if self.policy == "thompson":
            return np.random.rand() < self.belief[0]  # Thompson sampling
        elif self.policy == "greedy":
            return self.belief[0] > 0.5  # Greedy policy

# ====================== ENVIRONMENT ======================

class TwoArmedBanditEnv:
    def __init__(self, reward_probs=[[0.8, 0.2], [0.2, 0.8]], switch_prob=0.02):
        self.reward_probs = reward_probs  # Now a 2D list
        self.switch_prob = switch_prob
        self.state = 0  # 0 = left is high, 1 = right is high

    def step(self, action):
        # State transition
        if np.random.rand() < self.switch_prob:
            self.state = 1 - self.state
        # Reward
        reward_prob = self.reward_probs[self.state][action]
        reward = np.random.rand() < reward_prob
        return reward, self.state

# ===================== SIMULATIONS ======================

def simulate_figure6a(num_trials=500):
    """Replicate Figure 6A: Log-odds trajectories"""
    env = TwoArmedBanditEnv()
    rflr = RFLRAgent(alpha=0.3, beta=2.0, tau=1.43)
    hmm = HMMAgent(transition_matrix=[[0.98, 0.02], [0.02, 0.98]], 
                   reward_probs=[[0.8, 0.2], [0.2, 0.8]])
    ideal_hmm = HMMAgent(transition_matrix=[[0.98, 0.02], [0.02, 0.98]], 
                         reward_probs=[[0.8, 0.2], [0.2, 0.8]], policy="greedy")

    # Track log-odds
    log_rflr, log_hmm, log_ideal = [], [], []

    for t in range(num_trials):
        # Force block transition at t=250
        if t == num_trials // 2:
            env.state = 1 - env.state

        # Use same action sequence for fair comparison
        action = np.random.choice([0, 1])
        reward, _ = env.step(action)

        # Update models
        rflr.update(action, reward)
        hmm.update(action, reward)
        ideal_hmm.update(action, reward)

        # Compute log-odds
        log_rflr.append(rflr.alpha * (2*action-1) + rflr.phi)  # RFLR log-odds (ψ)
        log_hmm.append(np.log(hmm.belief[0] / hmm.belief[1]))   # HMM log-odds
        log_ideal.append(np.log(ideal_hmm.belief[0] / ideal_hmm.belief[1]))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(log_rflr, color='orange', label='RFLR')
    plt.plot(log_hmm, color='black', label='Sticky HMM')
    plt.plot(log_ideal, color='blue', linestyle='--', label='Ideal HMM')
    plt.axvline(num_trials//2, color='gray', linestyle=':', label='Block transition')
    plt.xlabel('Trial')
    plt.ylabel('Log-odds (left vs right)')
    plt.title('Figure 6A: Log-odds trajectories')
    plt.legend()
    plt.show()

def simulate_figure6b(num_trials=1000, num_simulations=50):
    """Replicate Figure 6B: P(high choice) and P(switch)"""
    p_high = np.zeros(num_trials)
    p_switch = np.zeros(num_trials)

    for _ in range(num_simulations):
        env = TwoArmedBanditEnv()
        agent = RFLRAgent(alpha=0.3, beta=2.0, tau=1.43)
        last_action = None

        for t in range(num_trials):
            action = agent.choose_action()
            reward, state = env.step(action)
            agent.update(action, reward)

            # Track metrics
            p_high[t] += (action == state)  # P(high choice)
            if last_action is not None:
                p_switch[t] += (action != last_action)
            last_action = action

    # Normalize and smooth
    p_high = p_high / num_simulations
    p_switch = p_switch / num_simulations
    window = 20
    p_high_smooth = np.convolve(p_high, np.ones(window)/window, mode='same')
    p_switch_smooth = np.convolve(p_switch, np.ones(window)/window, mode='same')

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(p_high_smooth, label='P(high choice)')
    plt.plot(p_switch_smooth, label='P(switch)')
    
    # Mark block transitions (2% probability per trial)
    transitions = np.where(np.random.rand(num_trials) < 0.02)[0]
    for t in transitions:
        plt.axvline(t, color='gray', alpha=0.1, linewidth=0.5)
    
    plt.xlabel('Trial')
    plt.ylabel('Probability')
    plt.title('Figure 6B: Behavioral metrics around block transitions')
    plt.legend()
    plt.show()

# ====================== RUN ======================
if __name__ == "__main__":
    simulate_figure6a()
    simulate_figure6b()