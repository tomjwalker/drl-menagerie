import pytest
from tac.agent.memory.onpolicy import OnPolicyReplay, OnPolicyBatchReplay
from tac.spec.sarsa.temp_spec import spec

from tac.experiment.control import make_agent_env


class TestOnPolicyReplay:

    def test_update(self):

        env, agent = make_agent_env(spec)
        replay = OnPolicyReplay(spec, agent)

        replay.update(0, 0, 0, 0, False)
        replay.update(1, 2, 3, 4, False)
        replay.update(5, 6, 7, 8, True)
        replay.update(20, 21, 22, 23, True)

        assert replay.size == 4
        assert replay.seen_size == 4
        assert replay.states == [[0, 1, 5], [20]]
        assert replay.actions == [[0, 2, 6], [21]]
        assert replay.rewards == [[0, 3, 7], [22]]
        assert replay.next_states == [[0, 4, 8], [23]]
        assert replay.dones == [[False, False, True], [True]]

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

        replay.update(0, 0, 0, 0, False)
        replay.update(1, 2, 3, 4, False)
        replay.update(5, 6, 7, 8, True)
        replay.update(20, 21, 22, 23, True)

        assert replay.size == 4
        assert replay.seen_size == 4
        assert replay.states == [0, 1, 5, 20]
        assert replay.actions == [0, 2, 6, 21]
        assert replay.rewards == [0, 3, 7, 22]
        assert replay.next_states == [0, 4, 8, 23]
        assert replay.dones == [False, False, True, True]

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
