import pytest
from tac.agent.memory.onpolicy import OnPolicyReplay, OnPolicyBatchReplay
from tac.spec.sarsa.temp_spec import spec

from tac.experiment.control import make_agent_env


class TestOnPolicyReplay:

    def test_update(self):
        env, agent = make_agent_env(spec)
        replay = OnPolicyReplay(spec, agent)

        # The updates below represent two completed episodes.
        # For full test coverage, including the `if len(self.states) == self.training_frequency:` clause, need to set
        # replay.training_frequency = 2 (overwrite from spec above) in order to trigger the ready_to_train flag.
        # Then include an assert that replay.ready_to_train == True
        replay.training_frequency = 2

        replay.update(0, 0, 0, 0, False)
        replay.update(1, 2, 3, 4, False)
        # ...Check that the replay buffer is not ready to train yet...
        assert replay.agent.algorithm.ready_to_train is False
        replay.update(5, 6, 7, 8, True)
        replay.update(20, 21, 22, 23, True)

        assert replay.size == 4
        assert replay.seen_size == 4
        assert replay.states == [[0, 1, 5], [20]]
        assert replay.actions == [[0, 2, 6], [21]]
        assert replay.rewards == [[0, 3, 7], [22]]
        assert replay.next_states == [[0, 4, 8], [23]]
        assert replay.dones == [[False, False, True], [True]]

        # Check that the replay buffer is now ready to train
        assert replay.agent.algorithm.ready_to_train is True

    def test_sample(self):
        env, agent = make_agent_env(spec)
        replay = OnPolicyReplay(spec, agent)

        replay.update(0, 0, 0, 0, False)
        replay.update(1, 2, 3, 4, False)
        replay.update(5, 6, 7, 8, True)
        replay.update(20, 21, 22, 23, True)

        assert replay.sample() == {
            "states": [[0, 1, 5], [20]],
            "actions": [[0, 2, 6], [21]],
            "rewards": [[0, 3, 7], [22]],
            "next_states": [[0, 4, 8], [23]],
            "dones": [[False, False, True], [True]],
        }


class TestOnPolicyBatchReplay:

    def test_update(self):
        env, agent = make_agent_env(spec)
        replay = OnPolicyBatchReplay(spec, agent)

        # The updates below represent four frames.
        # For full test coverage, including the `if len(self.states) == self.training_frequency:` clause, need to set
        # replay.training_frequency = 2 (overwrite from spec above) in order to trigger the ready_to_train flag.
        # Then include an assert that replay.ready_to_train == True
        replay.training_frequency = 4

        replay.update(0, 0, 0, 0, False)
        replay.update(1, 2, 3, 4, False)
        # ...Check that the replay buffer is not ready to train yet...
        assert replay.agent.algorithm.ready_to_train is False
        replay.update(5, 6, 7, 8, True)
        replay.update(20, 21, 22, 23, True)

        assert replay.size == 4
        assert replay.seen_size == 4
        assert replay.states == [0, 1, 5, 20]
        assert replay.actions == [0, 2, 6, 21]
        assert replay.rewards == [0, 3, 7, 22]
        assert replay.next_states == [0, 4, 8, 23]
        assert replay.dones == [False, False, True, True]

        # Check that the replay buffer is now ready to train
        assert replay.agent.algorithm.ready_to_train is True

    def test_sample(self):
        env, agent = make_agent_env(spec)
        replay = OnPolicyBatchReplay(spec, agent)

        replay.update(0, 0, 0, 0, False)
        replay.update(1, 2, 3, 4, False)
        replay.update(5, 6, 7, 8, True)
        replay.update(20, 21, 22, 23, True)

        assert replay.sample() == {
            "states": [0, 1, 5, 20],
            "actions": [0, 2, 6, 21],
            "rewards": [0, 3, 7, 22],
            "next_states": [0, 4, 8, 23],
            "dones": [False, False, True, True],
        }
