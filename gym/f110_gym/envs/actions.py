"""
Module for handling the action space in the PBL f110 gym environment
"""

import enum
import logging
import numpy as np
from gymnasium.spaces.box import Box
from gymnasium.spaces.multi_binary import MultiBinary
from gymnasium.spaces.multi_discrete import MultiDiscrete


class ActionMode(enum.Enum):
    BB = "bangbang"
    BZB = "bangzerobang"
    CONTINUOUS = "cont"


def get_action(action_mode: ActionMode, v_min, v_max, s_max):
    print(
        f"In the action space the target steering angle is limited to {[-s_max, s_max]} and the target velocity is limited to {[v_min, v_max]}."
    )
    if action_mode is ActionMode.CONTINUOUS:
        act_space = Box(
            np.array([-s_max, v_min]),
            np.array([s_max, v_max]),  # TODO add normailzation?
        )  # desired steering angle and velocity
    elif action_mode is ActionMode.BB:
        act_space = MultiBinary(2)
    elif action_mode is ActionMode.BZB:
        act_space = MultiDiscrete((3, 3))

    return act_space, Actionator(action_mode, s_max, v_min, v_max)


class Actionator:
    def __init__(
        self, mode: ActionMode, s_max: float, v_min: float, v_max: float
    ) -> None:
        self.mode = mode
        self.s_max = s_max
        self.v_min = v_min
        self.v_max = v_max

    def get_phys_action(self, action: np.array):
        phys_action = action

        if self.mode is ActionMode.BB:
            phys_action = self._get_bangbang_action(action)
        elif self.mode is ActionMode.BZB:
            phys_action = self._get_bangzerobang_action(action)

        logging.log(
            level=logging.DEBUG,
            msg=f"Action applied to the system with mode {self.mode}:\n    {phys_action}\nOriginal action:\n    {action}",
        )

        return phys_action.reshape(1, 2)

    def _get_bangbang_action(self, action):

        # steering
        out_action = np.zeros((2))
        if action[0] == 0:
            out_action[0] = -self.s_max
        else:
            out_action[0] = self.s_max
        # speed
        if action[1] == 0:
            out_action[1] = self.v_min
        else:
            out_action[1] = self.v_max

        return out_action

    def _get_bangzerobang_action(self, action):
        # steering
        out_action = np.zeros((2))
        if action[0] == 0:
            out_action[0] = -self.s_max
        elif action[0] == 1:
            out_action[0] = 0
        else:
            out_action[0] = self.s_max
        # speed
        if action[1] == 0:
            out_action[1] = -self.v_min
        elif action[1] == 1:
            out_action[1] = 0
        else:
            out_action[1] = self.v_max

        return out_action
