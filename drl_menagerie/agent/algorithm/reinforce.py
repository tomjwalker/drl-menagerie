from torch.distributions import Categorical    # Used to represent a categorical distribution over a discrete variable
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Agent hyperparameters
GAMMA = 0.99    # Discount factor
HIDDEN_LAYER_UNITS = 64
LEARNING_RATE = 0.01

# Environment parameters
ENVIRONMENT = "CartPole-v1"

# Training parameters
TRAINING_EPISODES = 1000


class Pi(nn.Module):

    def __init__(self, input_size, output_size):
        super(Pi, self).__init__()

        self.rewards = []
        self.log_probs = []
        layers = [
            nn.Linear(input_size, HIDDEN_LAYER_UNITS),
            nn.Linear(HIDDEN_LAYER_UNITS, output_size),
        ]
        self.model = nn.Sequential(*layers)

        self.on_policy_reset()    # Initialise the log_probs and rewards lists
        self.train()    # Set the module in training mode. Inherited from nn.Module. Relevant for dropout and BatchNorm

    def on_policy_reset(self):
        """Resets the log_probs and rewards lists. Called at the start of each episode"""
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        logits = self.model(x)
        return logits

    def act(self, state):
        """
        Selects an action from the policy network's output distribution, and stores the log probability of the policy
        network's output distribution at the selected action in the log_probs list
        """
        # Numpy to PyTorch tensor
        state = torch.from_numpy(state.astype(np.float32))
        # Get the action preference (logits; h(s, a, theta)) from the policy network
        action_preference = self.forward(state)
        # Convert the action preference to a probability distribution over the actions
        prob_dist = Categorical(logits=action_preference)
        # pi(a|s): Sample an action from the probability distribution
        action = prob_dist.sample()
        # Store the log probability of pi(a|s) in the log_probs list
        log_prob = prob_dist.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()


def train(pi, optimiser):
    """Performs the inner gradient-ascent loop of the REINFORCE algorithm"""

    # Get the number of steps (T) in the episode. Episodes vary in length, but the information can be gained from the
    # policy object's rewards list, which is appended to in `main()`, before this method is called
    final_timestep = len(pi.rewards)
    # Initialise return
    rets = np.empty(final_timestep, dtype=np.float32)
    future_ret = 0.0
    # Calculate the return at each time step
    for t in range(final_timestep):
        # Calculate the discounted return
        future_ret = pi.rewards[final_timestep - t - 1] + GAMMA * future_ret
        rets[final_timestep - t - 1] = future_ret
    # Convert the returns and log_probs to PyTorch tensors
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    # Calculate the loss
    loss = -1 * (rets * log_probs).sum()
    # Backpropagate the loss
    optimiser.zero_grad()    # Reset the gradients
    loss.backward()    # Calculate the gradients
    optimiser.step()    # Gradient ascent, since the loss is negated. Update the policy network's parameters

    return loss


def main():

    env = gym.make(ENVIRONMENT)
    max_episode_steps = env.spec.max_episode_steps

    state_cardinality = env.observation_space.shape[0]
    action_cardinality = env.action_space.n
    # Policy network pi_theta for REINFORCE
    pi = Pi(state_cardinality, action_cardinality)

    optimiser = optim.Adam(pi.parameters(), lr=LEARNING_RATE)

    for episode in range(TRAINING_EPISODES):
        state, info = env.reset()
        for t in range(max_episode_steps):
            action = pi.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            pi.rewards.append(reward)
            env.render()
            if terminated or truncated:
                break

        loss = train(pi, optimiser)    # Perform the inner gradient-ascent loop of the REINFORCE algorithm
        total_reward = sum(pi.rewards)
        solved = total_reward > 0.975 * max_episode_steps
        pi.on_policy_reset()    # Reset the log_probs and rewards lists after each episode

        print(
            f"Episode {episode} finished after {t} timesteps. "
            f"Total reward: {total_reward}. Loss: {loss}. Solved: {solved}"
        )


if __name__ == "__main__":
    main()

