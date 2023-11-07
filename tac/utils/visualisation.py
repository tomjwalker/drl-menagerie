import copy

import gymnasium as gym
from gymnasium.utils.save_video import save_video
from matplotlib import pyplot as plt

from tac.utils.general import set_filepath


def record_agent(agent, spec_dict, episode):
    """Records the agent's performance in the env, and saves the recording to a file"""

    # TODO: hack. Check whether there is a better way to do this. Done because the agent's net is a PyTorch nn, and
    #   there are potentially complications around nn.Module.train() and nn.Module.eval()
    video_agent = copy.deepcopy(agent)

    # Parameter: number of episodes to record per video
    num_episodes = spec_dict.get("num_episodes_per_video", 10)

    # Set save filepath
    root_dir = spec_dict.get("monitor_dir", ".data")
    episode_dir = set_filepath(root_dir + f"/training_episode_{episode}")

    # Initialise a separate instance of the env for recording the video
    video_env = gym.make(spec_dict.get("env"), render_mode="rgb_array_list")

    # Run the agent in the env

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
