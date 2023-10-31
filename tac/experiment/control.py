import torch
from ray import tune

# TODO: remove this comment block once Reinforce has been refactored to fit the same template as SARSA
# from tac import algorithm_map

from tac.utils.general import temp_initialise_log, set_random_seed
from tac.utils.visualisation import record_agent, plot_session

from tac.agent import Agent
from tac.env import make_env


SEARCH_MODES = {
    "uniform": tune.uniform,
    "normal": tune.randn,
    "choice": tune.choice,
    "randint": tune.randint,
    "grid": tune.grid_search,
}


def _parse_search_key(key):
    """Search keys are of the form: <parameter_name>__<search_mode>. This function parses the key into its constituent
    parts: parameter_name and search_mode."""
    return key.split("__")[0], key.split("__")[1]


def get_search_space(spec):
    """Returns a search space for the hyperparameters specified in the spec. The search space is a dictionary of
    parameter names and search modes. The search modes are functions from the Ray Tune library."""
    search_dict = spec["search"]
    search_space = {}
    for key in search_dict:
        param_name, search_mode = _parse_search_key(key)
        search_space[param_name] = SEARCH_MODES[search_mode](*search_dict[key])
    return search_space


def make_agent_env(spec):

    # Instantiate environment from spec
    env = make_env(spec)

    # TODO: remove this comment block once Reinforce has been refactored to fit the same template as SARSA
    # # Instantiate agent from spec
    # algorithm_class = algorithm_map[spec["algorithm"]]
    # state_cardinality = env.observation_space.shape[0]
    # action_cardinality = env.action_space.n
    # agent = algorithm_class(spec, state_cardinality, action_cardinality)

    agent = Agent(spec, env)

    return env, agent


class Session:

    def __init__(self, spec):
        self.spec = spec
        self.env, self.agent = make_agent_env(self.spec)
        self.max_episode_steps = self.env.u_env.spec.max_episode_steps
        # TODO: reimplement training log in new run method
        self.training_log = temp_initialise_log(self.spec)
        set_random_seed()
    #
    # def run_old(self):
    #     for episode in range(self.spec["training_episodes"]):
    #         state, info = self.env.reset()
    #         for t in range(self.max_episode_steps):
    #             action = self.agent.act(state)
    #             state, reward, terminated, truncated, _ = self.env.step(action)
    #             self.agent.rewards.append(reward)
    #             self.env.render()
    #             if terminated or truncated:
    #                 break
    #
    #         loss = self.agent.train()  # Perform the inner gradient-ascent loop of the REINFORCE algorithm
    #         total_reward = sum(self.agent.rewards)
    #         solved = total_reward > 0.975 * self.max_episode_steps
    #         self.agent.on_policy_reset()  # Reset the log_probs and rewards lists after each episode
    #
    #         # Log metrics
    #         self.training_log.loc[episode, "loss"] = loss.item()
    #         self.training_log.loc[episode, "total_reward"] = total_reward
    #         self.training_log.loc[episode, "solved"] = solved
    #
    #         if episode in self.spec.get("training_record_episodes", []):
    #             record_agent(self.agent, self.spec, episode)
    #
    #         print(
    #             f"Episode {episode} finished after {t} timesteps. "
    #             f"Total reward: {total_reward}. Loss: {loss}. Solved: {solved}"
    #         )
    #     #
    #     # plot_session(self.training_log)
    #
    #     return self.training_log

    def run(self):
        clock = self.env.clock
        state, info = self.env.reset()
        done = False
        # Infinite loop: run until a break is reached when the current timestep is equal to or greater than the max
        while True:
            print(f"Frame number: {clock.get()}")

            if done:
                # If current timestep is less than total max frames / steps for training, then reset the environment and
                # continue training with a new episode
                if clock.get() < clock.max_frame:
                    clock.tick("epi")    # Increment episode counter
                    state, info = self.env.reset()
                    done = False    # Reset done flag, the `while` loop will go around again

            # If current timestep is greater than or equal to total max frames / steps for training, then Break loop
            if clock.get() >= clock.max_frame:
                break

            # Increment timestep
            clock.tick("t")

            with torch.no_grad():
                action = self.agent.act(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            # TODO: Check this is the right place to calculate `done` - this is the newer Gymnasium API
            done = terminated or truncated
            self.agent.update(state, action, reward, next_state, done)
            state = next_state

            # TODO: from here down, could do with a full refactor. Temp logging solution for now
            # loss = self.agent.train()  # Perform the inner gradient-ascent loop of the REINFORCE algorithm
            loss = self.agent.loss
            # total_reward = sum(self.agent.rewards)
            total_reward = self.env.total_reward
            solved = total_reward > 0.975 * self.max_episode_steps
            # self.agent.on_policy_reset()  # Reset the log_probs and rewards lists after each episode

            # Log metrics
            frame = clock.get(unit="frame")
            self.training_log.loc[frame, "loss"] = loss
            self.training_log.loc[frame, "total_reward"] = total_reward
            self.training_log.loc[frame, "solved"] = solved

            # TODO: is `frame` right here
            if frame in self.spec.get("training_record_episodes", []):
                record_agent(self.agent, self.spec, frame)
            #
            # print(
            #     f"Episode {frame} finished after {t} timesteps. "
            #     f"Total reward: {total_reward}. Loss: {loss}. Solved: {solved}"
            # )

        plot_session(self.training_log)

        return self.training_log


class Trial:

    def __init__(self, spec):
        self.spec = spec
        self.num_sessions = spec["num_sessions"]
        self.session_logs = {}

    def _run_serial_trial(self):
        for session_num in range(self.num_sessions):
            session = Session(self.spec)
            session_log = session.run()
            self.session_logs[session_num] = session_log

    def _run_parallel_trial(self):
        # TODO: implement parallelised version
        raise NotImplementedError

    def run(self):
        run_mode = self.spec.get("run_mode", "serial")
        if run_mode == "serial":
            self._run_serial_trial()
        elif run_mode == "parallel":
            self._run_parallel_trial()
        else:
            raise ValueError(f"Invalid run mode: {run_mode}. Should be one of: serial, parallel")

        return self.session_logs


# class Experiment:
#     raise NotImplementedError


RUN_MODES = {
    "session": Session,
    "trial": Trial,
    # "experiment": Experiment,
}
