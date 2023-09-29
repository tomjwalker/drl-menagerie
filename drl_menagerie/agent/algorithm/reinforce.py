from torch.distributions import Categorical  # Used to represent a categorical distribution over a discrete variable
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# # Agent hyperparameters
# GAMMA = 0.99  # Discount factor
# HIDDEN_LAYER_UNITS = 64
# LEARNING_RATE = 0.01
#
# # Environment parameters
# ENVIRONMENT = "CartPole-v1"
#
# # Training parameters
# TRAINING_EPISODES = 1000

spec = {
    "gamma": 0.99,
    "hidden_layer_units": [64],
    "learning_rate": 0.01,
    "environment": "CartPole-v1",
    "training_episodes": 1000,
    "activation": "relu",
    "optimiser": "adam",
}

activation_functions = {
    "relu": nn.ReLU,
}

optimisers = {
    "adam": optim.Adam,
}


class MLPNet(nn.Module):

    def __init__(self, spec, input_size, output_size):
        super(MLPNet, self).__init__()

        layers = []

        num_hidden_layers = len(spec["hidden_layer_units"])

        # Input layer
        layers.append(nn.Linear(input_size, spec["hidden_layer_units"][0]))
        layers.append(activation_functions.get(spec["activation"])())

        # Hidden layers
        for i in range(num_hidden_layers - 1):
            layers.append(nn.Linear(spec["hidden_layer_units"][i], spec["hidden_layer_units"][i + 1]))
            layers.append(activation_functions.get(spec["activation"])())

        # Output layer
        layers.append(nn.Linear(spec["hidden_layer_units"][-1], output_size))

        self.model = nn.Sequential(*layers)

        self.train()  # Set the module in training mode. Inherited from nn.Module. Relevant for dropout and BatchNorm

    def forward(self, x):
        return self.model(x)


class Reinforce:

    def __init__(self, spec, input_size, output_size):

        self.rewards = []
        self.log_probs = []

        self.model = MLPNet(spec, input_size, output_size)
        self.on_policy_reset()  # Initialise the log_probs and rewards lists

        self.optimiser = optimisers.get(spec["optimiser"])(self.model.parameters(), lr=spec["learning_rate"])

    def on_policy_reset(self):
        """Resets the log_probs and rewards lists. Called at the start of each episode"""
        self.log_probs = []
        self.rewards = []

    def act(self, state):
        """
        Selects an action from the policy network's output distribution, and stores the log probability of the policy
        network's output distribution at the selected action in the log_probs list
        """
        # Numpy to PyTorch tensor
        state = torch.from_numpy(state.astype(np.float32))
        # Get the action preference (logits; h(s, a, theta)) from the policy network
        action_preference = self.model.forward(state)
        # Convert the action preference to a probability distribution over the actions
        prob_dist = Categorical(logits=action_preference)
        # pi(a|s): Sample an action from the probability distribution
        action = prob_dist.sample()
        # Store the log probability of pi(a|s) in the log_probs list
        log_prob = prob_dist.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()

    def train(self):
        """Performs the inner gradient-ascent loop of the REINFORCE algorithm"""

        # Get the number of steps (T) in the episode. Episodes vary in length, but the information can be gained from
        # the policy object's rewards list, which is appended to in `main()`, before this method is called
        final_timestep = len(self.rewards)
        # Initialise return
        rets = np.empty(final_timestep, dtype=np.float32)
        future_ret = 0.0
        # Calculate the return at each time step
        for t in range(final_timestep):
            # Calculate the discounted return
            future_ret = self.rewards[final_timestep - t - 1] + spec.get("gamma") * future_ret
            rets[final_timestep - t - 1] = future_ret
        # Convert the returns and log_probs to PyTorch tensors
        rets = torch.tensor(rets)
        log_probs = torch.stack(self.log_probs)
        # Calculate the loss
        loss = -1 * (rets * log_probs).sum()
        # Backpropagate the loss
        self.optimiser.zero_grad()  # Reset the gradients
        loss.backward()  # Calculate the gradients
        self.optimiser.step()  # Gradient ascent, since the loss is negated. Update the policy network's parameters

        return loss


def main():
    env = gym.make(spec.get("environment"))
    max_episode_steps = env.spec.max_episode_steps

    state_cardinality = env.observation_space.shape[0]
    action_cardinality = env.action_space.n
    # Policy network pi_theta for REINFORCE
    agent = Reinforce(spec, state_cardinality, action_cardinality)

    # optimiser = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    for episode in range(spec.get("training_episodes")):
        state, info = env.reset()
        for t in range(max_episode_steps):
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            env.render()
            if terminated or truncated:
                break

        loss = agent.train()  # Perform the inner gradient-ascent loop of the REINFORCE algorithm
        total_reward = sum(agent.rewards)
        solved = total_reward > 0.975 * max_episode_steps
        agent.on_policy_reset()  # Reset the log_probs and rewards lists after each episode

        print(
            f"Episode {episode} finished after {t} timesteps. "
            f"Total reward: {total_reward}. Loss: {loss}. Solved: {solved}"
        )


if __name__ == "__main__":
    main()
