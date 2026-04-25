from training_utils import collect_rewards_with_autosubmit


class _DummyEnv:
    def __init__(self, done: bool, reward: float, reward_after_submit: float | None = None):
        self.done = done
        self.reward = reward
        self._reward_after_submit = reward_after_submit if reward_after_submit is not None else reward
        self.submit_called = 0

    def submit(self):
        self.submit_called += 1
        self.done = True
        self.reward = self._reward_after_submit
        return "ok"


def test_collect_rewards_autosubmits_incomplete_episodes():
    env_done = _DummyEnv(done=True, reward=0.4)
    env_incomplete = _DummyEnv(done=False, reward=0.0, reward_after_submit=0.7)

    rewards = collect_rewards_with_autosubmit([env_done, env_incomplete])

    assert rewards == [0.4, 0.7]
    assert env_done.submit_called == 0
    assert env_incomplete.submit_called == 1
