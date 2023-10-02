from tac import algorithm_map
import gymnasium as gym

from tac.utils.general import temp_initialise_log, set_random_seed
from tac.utils.visualisation import record_agent, plot_session


def make_agent_env(spec):

    # Instantiate environment from spec
    env = gym.make(spec.get("environment"), render_mode="rgb_array")

    # Instantiate agent from spec
    algorithm_class = algorithm_map[spec["algorithm"]]
    state_cardinality = env.observation_space.shape[0]
    action_cardinality = env.action_space.n
    agent = algorithm_class(spec, state_cardinality, action_cardinality)

    return env, agent


class Session:

    def __init__(self, spec):
        self.spec = spec
        self.env, self.agent = make_agent_env(self.spec)
        self.max_episode_steps = self.env.spec.max_episode_steps
        self.training_log = temp_initialise_log(self.spec)
        set_random_seed()

    def run(self):
        for episode in range(self.spec["training_episodes"]):
            state, info = self.env.reset()
            for t in range(self.max_episode_steps):
                action = self.agent.act(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                self.agent.rewards.append(reward)
                self.env.render()
                if terminated or truncated:
                    break

            loss = self.agent.train()  # Perform the inner gradient-ascent loop of the REINFORCE algorithm
            total_reward = sum(self.agent.rewards)
            solved = total_reward > 0.975 * self.max_episode_steps
            self.agent.on_policy_reset()  # Reset the log_probs and rewards lists after each episode

            # Log metrics
            self.training_log.loc[episode, "loss"] = loss.item()
            self.training_log.loc[episode, "total_reward"] = total_reward
            self.training_log.loc[episode, "solved"] = solved

            if episode in self.spec.get("training_record_episodes"):
                record_agent(self.agent, self.spec, episode)

            print(
                f"Episode {episode} finished after {t} timesteps. "
                f"Total reward: {total_reward}. Loss: {loss}. Solved: {solved}"
            )

        plot_session(self.training_log)
