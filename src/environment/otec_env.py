# src/environment/otec_env.py

import numpy as np
import gym
from gym import spaces


class OTECEnvironment(gym.Env):
    def __init__(self, sst_data, regime_type="T1"):
        super(OTECEnvironment, self).__init__()

        self.sst_data = sst_data
        self.regime_type = regime_type
        self.current_step = 0

        # ==========================
        # 🔥 FIXED STATE SIZE (VERY IMPORTANT)
        # ==========================
        # Using compressed features instead of full grid
        self.state_size = 4  # [mean, std, min, max]

        # Observation space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_size,),
            dtype=np.float32
        )

        # Action space (3 control variables)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

    # ==========================
    # RESET
    # ==========================
    def reset(self):
        self.current_step = 0
        return self._get_state()

    # ==========================
    # STATE REPRESENTATION
    # ==========================
    def _get_state(self):
        return self.sst_data[self.current_step]

        mean = np.nan_to_num(np.mean(sst))
        std = np.nan_to_num(np.std(sst))
        min_val = np.nan_to_num(np.min(sst))
        max_val = np.nan_to_num(np.max(sst))

        state = np.array([mean, std, min_val, max_val], dtype=np.float32)

        # Clamp values (very important)
        state = np.clip(state, 0.0, 1.0)

        return state

    # ==========================
    # STEP FUNCTION
    # ==========================
    def step(self, action):
        state = self._get_state()

        # Simulated parameters
        delta_T = state[0]  # mean SST as proxy
        flow = np.abs(action[0])
        pressure = np.abs(action[1])
        turbine = np.abs(action[2])

        # Simplified power equation
        power = (delta_T * flow * turbine) - (0.1 * flow ** 2)

        reward = self._compute_reward(power, flow, pressure)

        self.current_step += 1

        done = self.current_step >= len(self.sst_data) - 1

        next_state = self._get_state()

        return next_state, reward, done, {}

    # ==========================
    # REWARD FUNCTION (CRITICAL)
    # ==========================
    def _compute_reward(self, power, flow, pressure):

        # Clamp inputs
        power = np.nan_to_num(power)
        flow = np.nan_to_num(flow)
        pressure = np.nan_to_num(pressure)

        power = np.clip(power, -10, 10)
        flow = np.clip(flow, 0, 1)
        pressure = np.clip(pressure, 0, 1)

        if self.regime_type == "T1":
            reward = power

        elif self.regime_type == "T2":
            reward = power - 0.5 * flow

        elif self.regime_type == "T3":
            reward = power - 0.5 * pressure

        elif self.regime_type == "T4":
            reward = power - 0.7 * (flow + pressure)

        else:
            reward = power

        # Final safety clamp
        reward = np.clip(reward, -10, 10)

        return reward

    # ==========================
    # RENDER (OPTIONAL)
    # ==========================
    def render(self, mode="human"):
        print(f"Step: {self.current_step}")