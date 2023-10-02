# import json
import gymnasium as gym

from tac.utils.general import temp_initialise_log
from tac.utils.visualisation import record_agent, plot_session

from tac import algorithm_map


# spec_path = "drl_menagerie/spec/reinforce/reinforce_cartpole.json"
# with open(spec_path, "r") as f:
#     spec = json.load(f)

spec = {
    "algorithm": "reinforce",
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


def main():

    algorithm_class = algorithm_map[spec["algorithm"]]

    env = gym.make(spec.get("environment"), render_mode="rgb_array")
    max_episode_steps = env.spec.max_episode_steps

    state_cardinality = env.observation_space.shape[0]
    action_cardinality = env.action_space.n
    # Policy network pi_theta for REINFORCE
    agent = algorithm_class(spec, state_cardinality, action_cardinality)

    # Initialise a training log
    training_log = temp_initialise_log(spec)

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
