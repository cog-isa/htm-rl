from typing import List, Any

from htm_rl.envs.wrapper import Wrapper


class EpisodeLengthRecorder(Wrapper):
    episode_lengths: List[int]

    _current_length: int

    def __init__(self, env):
        super(EpisodeLengthRecorder, self).__init__(env)
        self._current_length = 0
        self.episode_lengths = []

    def act(self, action: Any) -> None:
        super(EpisodeLengthRecorder, self).act(action)
        self._current_length += 1

        _, _, first = super(EpisodeLengthRecorder, self).observe()
        if first:
            self.episode_lengths.append(self._current_length)
            self._current_length = 0
