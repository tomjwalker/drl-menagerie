from torch.distributions import Categorical  # Used to represent a categorical distribution over a discrete variable
import gymnasium as gym
from gymnasium.utils.save_video import save_video
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import copy

matplotlib.use('TkAgg')

spec = {
    "gamma": 0.99,
    "hidden_layer_units": [64],
    "learning_rate": 0.01,
    "environment": "CartPole-v1",
    "training_episodes": 500,
    "activation": "relu",
    "optimiser": "adam",
    "training_record_episodes": [0, 100, 499],
    "data_directory": ".data",
}

activation_functions = {
    "relu": nn.ReLU,
}

optimisers = {
    "adam": optim.Adam,
}


class MLPNet(nn.Module):

    def __init__(self, spec_dict, input_size, output_size):
        super(MLPNet, self).__init__()

        layers = []

        num_hidden_layers = len(spec_dict["hidden_layer_units"])

        # Input layer
        layers.append(nn.Linear(input_size, spec_dict["hidden_layer_units"][0]))
        layers.append(activation_functions.get(spec_dict["activation"])())

        # Hidden layers
        for i in range(num_hidden_layers - 1):
            layers.append(nn.Linear(spec_dict["hidden_layer_units"][i], spec_dict["hidden_layer_units"][i + 1]))
            layers.append(activation_functions.get(spec_dict["activation"])())

        # Output layer
        layers.append(nn.Linear(spec_dict["hidden_layer_units"][-1], output_size))

        self.model = nn.Sequential(*layers)

        self.train()  # Set the module in training mode. Inherited from nn.Module. Relevant for dropout and BatchNorm

    def forward(self, x):
        return self.model(x)


class Reinforce:

    def __init__(self, spec_dict, input_size, output_size):
        self.rewards = []
        self.log_probs = []

        self.model = MLPNet(spec_dict, input_size, output_size)
        self.on_policy_reset()  # Initialise the log_probs and rewards lists

        self.optimiser = optimisers.get(spec_dict["optimiser"])(self.model.parameters(), lr=spec_dict["learning_rate"])

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


# TODO: refactor this function into utils.py
def set_filepath(file_path_string):
    """Creates a directory at the specified path if it does not already exist"""
    file_path = Path(file_path_string)
    file_path.mkdir(parents=True, exist_ok=True)
    return file_path


# TODO: refactor this function
def temp_log(spec_dict):
    """Temporary solution to generate a training log for plotting metrics. To be replaced with a more robust solution
    in the future"""

    num_training_episodes = spec_dict.get("training_episodes")
    metrics = [
        "loss",
        "total_reward",
        "solved",
    ]

    # Generate a pandas DataFrame. Column names are `metrics`. Number of rows is `num_training_episodes`. Cells are
    # initially empty
    training_log = pd.DataFrame(index=range(num_training_episodes), columns=metrics)

    return training_log


# TODO: refactor this function
def plot_session(training_log):
    """
    Plots the metrics in the training log.

    Generates a figure with two axes:
    - Left axis: loss
    - Right axis: total reward
    """

    # Create a figure with two axes
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Plot the loss
    ax1.plot(training_log["loss"], color="tab:red")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # Plot the total reward
    ax2.plot(training_log["total_reward"], color="tab:blue")
    ax2.set_ylabel("Total reward", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # Show the figure
    fig.tight_layout()
    plt.show()


# TODO: refactor this function
def record_agent(agent, spec_dict, episode):
    """Records the agent's performance in the environment, and saves the recording to a file"""

    # TODO: hack. Check whether there is a better way to do this. Done because the agent's model is a PyTorch nn, and
    #   there are potentially complications around nn.Module.train() and nn.Module.eval()
    video_agent = copy.deepcopy(agent)

    # Parameter: number of episodes to record per video
    num_episodes = spec_dict.get("num_episodes_per_video", 10)

    # Set save filepath
    root_dir = spec_dict.get("monitor_dir", ".data")
    episode_dir = set_filepath(root_dir + f"/training_episode_{episode}")

    # TODO: check
    # Initialise a separate instance of the environment for recording the video
    video_env = gym.make(spec_dict.get("environment"), render_mode="rgb_array_list")

    # Run the agent in the environment

    episode_frames = []
    for t in range(num_episodes):
        state, info = video_env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = video_agent.act(state)
            state, reward, terminated, truncated, _ = video_env.step(action)
        frames = video_env.render()
        episode_frames += frames

    save_video(
        frames=episode_frames,
        video_folder=str(episode_dir),
        fps=spec_dict.get("fps", 10),
    )
    #
    # # Close the environment monitor
    # video_env.close()


def main():
    env = gym.make(spec.get("environment"), render_mode="rgb_array")
    max_episode_steps = env.spec.max_episode_steps

    state_cardinality = env.observation_space.shape[0]
    action_cardinality = env.action_space.n
    # Policy network pi_theta for REINFORCE
    agent = Reinforce(spec, state_cardinality, action_cardinality)

    # Initialise a training log
    training_log = temp_log(spec)

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

        # Log metrics
        training_log.loc[episode, "loss"] = loss.item()
        training_log.loc[episode, "total_reward"] = total_reward
        training_log.loc[episode, "solved"] = solved

        if episode in spec.get("training_record_episodes"):
            record_agent(agent, spec, episode)

        print(
            f"Episode {episode} finished after {t} timesteps. "
            f"Total reward: {total_reward}. Loss: {loss}. Solved: {solved}"
        )

    plot_session(training_log)


if __name__ == "__main__":
    main()
