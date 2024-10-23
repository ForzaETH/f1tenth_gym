# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Author: Hongrui Zheng
"""

# gym imports
import gymnasium as gym
from f110_gym.envs.rendering import EnvRenderer

# base classes
import pandas as pd
from f110_gym.envs.actions import ActionMode, get_action
from f110_gym.envs.base_classes import Simulator, Integrator
from f110_gym.envs.rewards import Reward, RewardMode
from f110_gym.envs.observations import ObservationMode, obs_generator
from f110_gym.envs.pure_pursuit_controller import PurePursuitPlanner
from f110_gym.envs.utils import downsample_points_simple, ensure_absolute_path, read_config

from splinify.splinify import SplineTrackNew

# others
import os
import os.path as osp
import time
import logging
import numpy as np
from argparse import Namespace
import git
from copy import deepcopy
from typing import Optional
import random

# gl
import pyglet

pyglet.options["debug_gl"] = False
# constants

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800


class F110Env(gym.Env):
    """
    OpenAI gym environment for F1TENTH

    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility

            map (str, default='vegas'): name of the map used for the environment.

            map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'

            params: dictionary of vehicle parameters.
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front
            C_Sr: Cornering stiffness coefficient, rear
            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=2): number of agents in the environment

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
    """

    metadata = {"render_modes": ["human", "human_fast", "rgb_array"], "render_fps": 60}
    # rendering
    renderer = None
    render_callbacks = []

    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        self.action_mode = ActionMode(kwargs.get("action_mode", "cont"))
        logging.log(level=logging.DEBUG, msg=f"Action Mode: {self.action_mode}")
        self.obs_mode = ObservationMode(kwargs.get("obs_mode", "frenet"))
        logging.log(level=logging.DEBUG, msg=f"Observation Mode: {self.obs_mode}")
        self.rew_mode = RewardMode(kwargs.get("rew_mode", "advancement"))
        logging.log(level=logging.DEBUG, msg=f"Reward Mode: {self.rew_mode}")
        self.penalty_based_on_reward = kwargs.get("penalty_based_on_reward", False)
        self.render_mode = render_mode

        self.start_pose = np.array(
            [[kwargs.get("sx", 0), kwargs.get("sy", 0), kwargs.get("stheta", 0)]]
        )
        print("Starting Pose:", self.start_pose)
        self.observed_position = self.start_pose[0, :2]
        self.observed_theta = self.start_pose[0, 2]
        
        self.seed = kwargs.get("seed", 69)
        self.use_lidar = kwargs.get("use_lidar", False)
        self.use_previous_actions = kwargs.get("use_previous_actions", False)
        self.use_base_controller = kwargs.get("use_base_controller", False)
        self.random_start = kwargs.get("random_start", False)
        self.enable_random_start_speed = kwargs.get("enable_random_start_speed", False)
        self.reset_width_factor = kwargs.get("reset_width_factor", 1.0)
        self.enable_steering_delay = kwargs.get("enable_steering_delay", False)
        self.delay_buffer_size = kwargs.get("delay_buffer_size", 5)
        self.randomize_steering_delay = kwargs.get("randomize_steering_delay", False)
        self.car_size_multiplier = kwargs.get("car_size_multiplier", 1.0)
        self.max_lap_num = kwargs.get("max_lap_num", 2)
        self.lap_count_reset_timer = kwargs.get("lap_count_reset_timer", 100)
        self.use_pp_action = kwargs.get("use_pp_action", False)
        self.enable_input_noise = kwargs.get("enable_input_noise", False)
        self.steering_noise_std = kwargs.get("steering_noise_std", 0)
        self.speed_noise_std = kwargs.get("speed_noise_std", 0)
        self.num_agents = kwargs.get("num_agents", 1)
        self.timestep = kwargs.get("timestep", 0.01)
        self.ego_idx = kwargs.get("ego_idx", 0)
        self.friction_coefficient = kwargs.get("friction_coefficient", 1)
        self.use_optimistic_speed_curriculum = kwargs.get("use_optimistic_speed_curriculum", False)
        self.use_action_smoothness_penalty = kwargs.get("use_action_smoothness_penalty", False)

        car_params = read_config(osp.join(osp.dirname(osp.abspath(__file__)), "SIM_car_params.yaml"))
        self.params = kwargs.get("params", car_params)
        print("Gym sim using params:", self.params)
        self.randomize_at_reset = kwargs.get("rndmize_at_reset", False)
        self.randomisation_stds = kwargs.get("rnd_stds", None)
        if self.randomisation_stds:
            print("Sim using randomisation parameters: ", self.randomisation_stds)

        self.initialize_map(**kwargs)

        if not self.use_pp_action:
            self.s_max = kwargs.get("action_steer_abs_limit", 0.4189)
            self.v_min = kwargs.get("action_v_lower_limit", -5.0)
            self.v_max = kwargs.get("action_v_upper_limit", 10.0)
            self.global_s_max = kwargs.get("action_steer_abs_limit", 0.4189)
            self.global_v_min = kwargs.get("action_v_lower_limit", -5.0)
            self.global_v_max = kwargs.get("action_v_upper_limit", 10.0)
        else:
            self.s_max = kwargs.get("pp_residual_steer_abs_limit", 0.15)
            self.v_min = kwargs.get("pp_residual_v_lower_limit", 0)
            self.v_max = kwargs.get("pp_residual_v_upper_limit", 5.0)
            self.global_s_max = kwargs.get("action_steer_abs_limit", 0.4189)
            self.global_v_min = kwargs.get("action_v_lower_limit", -5.0)
            self.global_v_max = kwargs.get("action_v_upper_limit", 10.0)

        if self.use_previous_actions:
            self.previous_s = 0
            self.previous_v = 0
        self.last_actions = None

        if self.use_pp_action:
            self.pp_s = 0
            self.pp_v = 0
            self.la_m = kwargs.get("pp_la_m", 0.5)
            self.la_q = kwargs.get("pp_la_q", 0)
            print(f"Pure Pusuit Controller: lookahead parameters: slope {self.la_m}, intercept {self.la_q}")

        if self.use_base_controller:
            self.base_controller_s = 0
            self.base_controller_v = 0

        if kwargs.get("integrator", "RK4") == "RK4":
            self.integrator = Integrator.RK4
        elif kwargs.get("integrator", "RK4") == "Euler":
            self.integrator = Integrator.Euler
        else:
            self.integrator = None

        if kwargs.get("dynamic_mode", "ros_dynamics_pacejka") == "ros_dynamics_linear":
            self.dynamic_mode = "ros_dynamics_linear"
        elif (
            kwargs.get("dynamic_mode", "ros_dynamics_pacejka") == "ros_dynamics_pacejka"
        ):
            self.dynamic_mode = "ros_dynamics_pacejka"
        elif kwargs.get("dynamic_mode", "ros_dynamics_pacejka") == "gym_dynamics":
            self.dynamic_mode = "gym_dynamics"
        else:
            self.dynamic_mode = "ros_dynamics_pacejka"

        self.observation_space, self.observator = obs_generator(
            mode=self.obs_mode,
            env_params=self.params,
            use_lidar=self.use_lidar,
            use_previous_actions=self.use_previous_actions,
            use_pp_action=self.use_pp_action,
            use_base_controller=self.use_base_controller,
            n_traj_points=kwargs.get("n_traj_points", 30),
            traj_points_spacing=kwargs.get("traj_points_spacing", 1),
            calc_glob_traj=kwargs.get("calc_glob_traj", False),
        )

        print(
            "F110 Gym using OBS mode: {} supported observation modes are: {}".format(
                self.obs_mode, [mode for mode in ObservationMode]
            )
        )
        print(
            f'The observation is composed of the following elements: {", ".join(self.observator.keys)}'
        )

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents,))

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,))
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents,))
        self.start_ys = np.zeros((self.num_agents,))
        self.start_thetas = np.zeros((self.num_agents,))
        self.start_rot = np.eye(2)

        # initiate the simulation
        self.sim = Simulator(
            self.params,
            self.num_agents,
            self.seed,
            time_step=self.timestep,
            integrator=self.integrator,
            dynamic_mode=self.dynamic_mode,
            redraw_upon_reset=self.randomize_at_reset,
            randomisation_stds=self.randomisation_stds,
            enable_steering_delay=self.enable_steering_delay,
            delay_buffer_size=self.delay_buffer_size,
            randomize_steering_delay=self.randomize_steering_delay,
            car_size_multiplier=self.car_size_multiplier,
            friction_coefficient=self.friction_coefficient,
            position_noise_std=kwargs.get("position_noise_std", 0.05), # meters
            yaw_noise_std=kwargs.get("yaw_noise_std", 0.01), # radians
            velocity_noise_std=kwargs.get("velocity_noise_std", 0.1), # m/s
            yaw_rate_noise_std=kwargs.get("yaw_rate_noise_std", 0.01), # rad/s
            no_lateral_velocity=kwargs.get("no_lateral_velocity", False),
        )
        self.sim.set_map(self.map_path, self.map_ext)

        # stateful observations for rendering
        self.render_obs = None

        # For SB3 Compatibility
        # Define action space
        self.action_space, self.actionator = get_action(
            self.action_mode, self.v_min, self.v_max, self.s_max
        )

        # max time steps implementation
        self._cur_step = 0
        self._max_episode_len = kwargs.get("ep_len", 10_000)

        # Define reward mode
        self.reward = Reward(
            mode=self.rew_mode,
            env=self,
            penalty_based_on_reward=self.penalty_based_on_reward,
            deviation_penalty_coefficient=kwargs.get("deviation_penalty_coefficient", 1),
            deviation_penalty_treshold=kwargs.get("deviation_penalty_treshold", 0.1),
            rel_heading_penalty_coefficient=kwargs.get("rel_heading_penalty_coefficient", 0.25),
            rel_heading_penalty_threshold=kwargs.get("rel_heading_penalty_threshold", 0),
            jerk_scale=kwargs.get("jerk_scale", 50),
            steering_smoothness_penalty_coefficient=kwargs.get("steering_smoothness_penalty_coefficient", 0.1),
            speed_smoothness_penalty_coefficient=kwargs.get("speed_smoothness_penalty_coefficient", 0.1),
        )
        print(
            "F110 Gym using RL mode: {} supported reward modes are: {}".format(
                self.rew_mode, [mode for mode in RewardMode]
            )
        )

        # Define Pure Pursuit Planner
        if self.use_pp_action or self.use_base_controller:
            self.planner = PurePursuitPlanner(
                Namespace(**kwargs), (self.params["wheelbase_PP"])
            )
            self.lookahead_distance = kwargs.get(
                "lookahead_distance", 0.82461887897713965
            )
            self.vgain = kwargs.get("vgain", 1.375)
            self.pp_action = np.array([0, 0])

        self.average_speed = 0

        self.sim_started = False

    def initialize_map(self, **kwargs):
        # map related parameters
        map_name = kwargs.get("map_name", None)
        repo = git.Repo(".", search_parent_directories=True)
        map_dir = kwargs.get("map_dir", None)
        map_dir = repo.working_tree_dir + map_dir.format(map_name=map_name)

        self.map_base_path = map_dir + f"{map_name}"
        self.map_path = self.map_base_path + ".yaml"
        self.map_ext = kwargs.get("map_ext", ".png")
        raceline_option = kwargs.get("raceline_option", None)

        self.wpt_path = self.map_base_path + f"_{raceline_option}.csv"
        self.extended_wpt_path = self.map_base_path + f"_extended_{raceline_option}.csv"
        if self.wpt_path is not None:
            print("Using WPNTS:", self.wpt_path)
            self.waypoints = np.loadtxt(self.wpt_path, delimiter=";")
        else:
            self.waypoints = None
            print("No Waypoints used!")

        # track spline setup
        track_safety_margin = kwargs.get("track_safety_margin", 0.0)
        print("Calculating the norm_vec....")

        self.track_forward = self.create_track(
            self.extended_wpt_path, track_safety_margin
        )

        self.random_direction = kwargs.get("random_direction", False)
        if self.random_direction:
            parts = self.extended_wpt_path.split(os.sep)
            filename, ext = os.path.splitext(parts[-1])
            new_name = filename.replace(
                "extended_raceline", "reverse_extended_raceline"
            )
            parts[-1] = new_name + ext
            parts[-2] += "_reverse"
            self.extended_wpt_path_reverse = ensure_absolute_path(os.path.join(*parts))

            if os.path.exists(str(self.extended_wpt_path_reverse)):
                self.track_reverse = self.create_track(
                    self.extended_wpt_path_reverse, track_safety_margin
                )
            else:
                print(
                    "The reverse map does not exist! No randomization in the direction."
                )
                self.track_reverse = self.track_forward

            self.track = self.track_reverse
        else:
            self.track = self.track_forward

        self.theta = kwargs.get("sparam", 0)
        self.prev_theta = kwargs.get("sparam", 0)
        self.start_theta = kwargs.get("sparam", 0)

    def create_track(self, file_path, track_safety_margin):
        with open(file_path, "r") as file:
            for i in range(2):
                file.readline()

            names = [w.strip() for w in file.readline().strip("# \n").split(";")]
            data = [line.strip().split(";") for line in file.readlines()]
            points = pd.DataFrame(data, columns=names, dtype=np.float32)
            points = points.rename(
                columns={
                    "s_m": "s_racetraj_m",
                    "x_m": "x_ref_m",
                    "y_m": "y_ref_m",
                    "d_right": "width_right_m",
                    "d_left": "width_left_m",
                    "psi_rad": "psi_racetraj_rad",
                    "kappa_radpm": "kappa_racetraj_radpm",
                    "vx_mps": "vx_racetraj_mps",
                    "ax_mps2": "ax_racetraj_mps2",
                }
            )

        track = SplineTrackNew(
            coords_param_direct={
                "coords": np.array([points["x_ref_m"], points["y_ref_m"]]).T,
                "left_widths": points["width_left_m"],
                "right_widths": points["width_right_m"],
                "params": points["s_racetraj_m"],
            },
            safety_margin=track_safety_margin,
        )

        return track

    def __del__(self):
        """
        Finalizer, does cleanup
        """
        pass

    def _check_done(self):
        """
        Check if the current rollout is done

        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (list[int]): each agent's toggle list for crossing the finish zone
        """

        # this is assuming 2 agents
        # TODO: switch to s-based
        left_t = 2
        right_t = 2

        poses_x = np.array(self.poses_x) - self.start_xs
        poses_y = np.array(self.poses_y) - self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :] ** 2 + temp_y**2
        closes = dist2 <= 0.025
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                if self.update_counters[i] >= self.lap_count_reset_timer:
                    self.near_starts[i] = True
                    self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
                self.update_counters[i] = 0
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < (self.max_lap_num) * 2:
                self.lap_times[i] = self.current_time

        done = True

        if self.collisions[self.ego_idx]:
            logging.info("Episode terminated because of collision")
        elif np.all(self.lap_counts >= self.max_lap_num):
            logging.info(
                "Episode terminated because of total number of laps completed reached maximum"
            )
        elif self._cur_step >= self._max_episode_len:
            logging.info(
                "Episode terminated because of reaching maximum episode length"
            )
        elif np.abs(self.sim.agents[self.ego_idx].state[5]) > 10:
            logging.info("Episode terminated because of singularity")
        elif np.abs(self.observator.denorm_obs['deviation']) > 5:
            logging.info("Episode terminated because of going to far away from the race line.")
        else:
            done = False

        return bool(done), self.toggle_list >= 4

    def _update_state(self, obs_dict):
        """
        Update the env's states according to observations

        Args:
            obs_dict (dict): dictionary of observation

        Returns:
            None
        """
        self.poses_x = obs_dict["poses_x"]
        self.poses_y = obs_dict["poses_y"]
        self.poses_theta = obs_dict["poses_theta"]
        self.collisions = obs_dict["collisions"]

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        self._cur_step += 1

        self.update_counters[0] += 1

        if not len(self.poses_x) == 0:
            self.sim_started = True
        else:
            self.sim_started = False

        if not isinstance(action, np.ndarray):
            action = np.array(action).reshape((1, -1))
        action = action.reshape((1, -1))

        # get correct action, in case bang bang control or similar is used
        action = self.actionator.get_phys_action(action.reshape(-1))    

        # select mode: RL or RL with MPC
        if self.use_pp_action and self.sim_started:
            speed, steer = self.planner.plan(
                self.observed_position[0],
                self.observed_position[1],
                self.observed_theta,
                self.la_m,
                self.la_q,
                self.vgain,
                self.reverse_direction,
            )
            self.pp_action = np.array([steer, speed])
            self.pp_s = self.pp_action[0]
            self.pp_v = self.pp_action[1]
            action = np.add(action, self.pp_action)
            # action = self.pp_action.reshape(1,2) # uncomment this to "test" PP only

        if self.use_previous_actions:
            self.previous_s = action[0, 0]
            self.previous_v = action[0, 1]

        if self.use_base_controller:
            if self.sim_started:
                speed, steer = self.planner.plan(
                    self.observed_position[0],
                    self.observed_position[1],
                    self.observed_theta,
                    self.la_m,
                    self.la_q,
                    self.vgain,
                    self.reverse_direction,
                )
                self.base_controller_s = steer
                self.base_controller_v = speed

        self.prev_theta = self.theta

        if self.enable_input_noise:
            steer_noise = np.random.normal(
                scale=self.steering_noise_std * self.global_s_max
            )
            speed_noise = np.random.normal(
                scale=self.speed_noise_std * self.global_v_max
            )
            action_noise = np.array(
                [
                    np.clip(
                        steer_noise,
                        -self.steering_noise_std * self.global_s_max,
                        self.steering_noise_std * self.global_s_max,
                    ),
                    np.clip(
                        speed_noise,
                        -self.speed_noise_std * self.global_v_max,
                        self.speed_noise_std * self.global_v_max,
                    ),
                ]
            )
            action = np.add(action, action_noise)

        action[0, 1] = np.clip(action[0, 1], 0, self.global_v_max)

        
        # SIM STEP
        obs = self.sim.step(action)
        self.theta = self.track.find_theta(
            np.array([obs["poses_x"], obs["poses_y"]]).T, self.theta
        )
        self.observed_position = obs["poses_x"][0], obs["poses_y"][0]
        self.observed_theta = obs["poses_theta"][0]

        obs["lap_times"] = self.lap_times
        obs["lap_counts"] = self.lap_counts

        if self.use_previous_actions:
            obs["previous_s"] = self.previous_s
            obs["previous_v"] = self.previous_v

        if self.use_pp_action:
            obs["pp_s"] = self.pp_s
            obs["pp_v"] = self.pp_v

        if self.use_base_controller:
            obs["base_controller_s"] = self.base_controller_s
            obs["base_controller_v"] = self.base_controller_v

        # observation preparation and flatteningsim
        elif self.obs_mode is ObservationMode.TRAJ_FRENET:
            env_obs = self.observator.get_obs(
                obs=obs,
                car_position=self.observed_position,
                track_position=self.track.get_coordinate(self.theta),
                track_angle=self.track.get_angle(self.theta),
                track_direction=self.track.get_derivative(self.theta),
                racetrack=self.track,
                cur_theta=self.theta,
            )

        self.render_obs = {
            "ego_idx": obs["ego_idx"],
            "poses_x": obs["poses_x"],
            "poses_y": obs["poses_y"],
            "poses_theta": obs["poses_theta"],
            "lap_times": obs["lap_times"],
            "lap_counts": obs["lap_counts"],
        }

        # times
        self.current_time = self.current_time + self.timestep

        # update data member
        self._update_state(obs)

        # if obs["collisions"][0] != 0.0:
        #     print("The car crashes and we get penalty for that!")

        # check done
        done, toggle_list = self._check_done()
        info = {"checkpoint_done": toggle_list}

        reward, rew_info = self.reward.get_reward(obs=obs, action=action)
        info["reward_info"] = rew_info

        if self.render_mode == "human":
            self.render()

        terminated = done
        truncated = False

        self.average_speed = (
            self.average_speed * self._cur_step + self.sim.agents[self.ego_idx].state[3]
        ) / (self._cur_step + 1)
        info["average_speed"] = self.average_speed
        
        self.last_actions = action
         
        return env_obs, reward, terminated, truncated, info

    def reset(self, seed=None, poses=-1, options={}):
        """
        Reset the gym environment by given poses.
        Poses=-1 because the official F110 Env expects a pose, which is against OpenAI standards.
        It was modified to catch -1 and use the initial pose.

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        # reset counters and data members
        self._cur_step = 0
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,))
        self.update_counters = np.zeros((self.num_agents,))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        if self.random_direction:
            self.track = random.choice([self.track_forward, self.track_reverse])

        if self.track == self.track_forward:
            self.reverse_direction = False
        else:
            self.reverse_direction = True

        if type(poses) == int:
            if poses == -1:
                if self.random_start:
                    for i in range(self.num_agents):
                        pos = np.random.uniform(0, self.track.track_length)
                        l_width = self.track.left_widths(pos)
                        r_width = self.track.right_widths(pos)
                        dev = np.random.uniform(
                            -r_width * self.reset_width_factor,
                            l_width * self.reset_width_factor,
                        )
                        self.start_pose[i, :2] = self.track.get_frenet_coordinate(
                            theta=pos, displacement=dev
                        )
                        self.start_pose[i, 2] = self.track.get_angle(pos)

                poses = self.start_pose

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [
                [
                    np.cos(-self.start_thetas[self.ego_idx]),
                    -np.sin(-self.start_thetas[self.ego_idx]),
                ],
                [
                    np.sin(-self.start_thetas[self.ego_idx]),
                    np.cos(-self.start_thetas[self.ego_idx]),
                ],
            ]
        )

        # reset theta
        self.theta = self.track.find_theta_slow(poses[0, :2])
        self.start_theta = self.theta
        self.prev_theta = self.theta

        # call reset to simulator
        self.sim.reset(poses)

        # reset reward
        self.reward.gate_progress = 0

        if self.enable_random_start_speed:
            if self.use_optimistic_speed_curriculum:
                self.sim.agents[self.ego_idx].state[3] = abs(
                    np.random.uniform(high=2*self.average_speed)
                )
            else:
                self.sim.agents[self.ego_idx].state[3] = abs(
                    np.random.normal(loc=np.clip(self.average_speed, 0, 5), scale=0.5)
                )

        self.average_speed = 0

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, reward, terminated, truncated, info = self.step(action)

        if self.render_mode == "human":
            self.render()

        return obs, info

    def update_map(self, map_path, map_ext):
        """
        Updates the map used by simulation

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file

        Returns:
            None
        """
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles

        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """

        self.render_callbacks.append(callback_func)

    def render(self):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        mode = self.render_mode
        assert mode in self.metadata["render_modes"]

        if self.renderer is None:
            # first call, initialize everything
            if mode == "rgb_array":
                self.renderer = EnvRenderer(
                    WINDOW_W,
                    WINDOW_H,
                    self.car_size_multiplier * self.params["length"],
                    self.car_size_multiplier * self.params["width"],
                    visible=False
                )
            else:
                self.renderer = EnvRenderer(
                    WINDOW_W,
                    WINDOW_H,
                    self.car_size_multiplier * self.params["length"],
                    self.car_size_multiplier * self.params["width"],
                )
            self.renderer.update_map(self.map_base_path, self.map_ext)

            boundary_points = self.renderer.map_points[:, :-1] / 50
            sorted_boundary_points = boundary_points[boundary_points[:, 0].argsort()]
            self.boundary_points = downsample_points_simple(sorted_boundary_points, 50)

        self.renderer.update_obs(self.render_obs)
        if self.wpt_path is not None:
            self.renderer.draw_raceline(self.waypoints)

        for render_callback in self.render_callbacks:
            render_callback(self.renderer)

        self.renderer.dispatch_events()
        self.renderer.on_draw()
        self.renderer.flip()
        if mode == "human":
            time.sleep(0.005)
        elif mode == "human_fast":
            pass
        elif mode == "rgb_array":
            # This part was stolen from the gym repo
            # to be precise, here: https://github.com/openai/gym/blob/master/gym/envs/classic_control/rendering.py
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
            return arr

        return True

    def get_max_dev(self):
        track_width = np.linalg.norm(
            np.array(self.track.get_coordinate(self.theta, line="int"))
            - np.array(self.track.get_coordinate(self.theta, line="out"))
        )
        return track_width / 2 - 1.5 * self.params["width"]

    def calculate_track_width(self):
        track_width = np.linalg.norm(
            np.array(self.track.get_coordinate(self.theta, line="int"))
            - np.array(self.track.get_coordinate(self.theta, line="out"))
        )
        return track_width
    
    def calculate_half_track_width(self, ref_d: float) -> float:
        """Only gives the partial width depending on where the ref_d is.
        If zero, the positive (left) side of the track is returned.

        Args:
            ref_d (float): The distance from the center of the track

        Returns:
            float: The half width of the track 
            (only one side depending on ref_d, not full_width/2 !!)
        """
        
        if ref_d >= 0:
            width = np.linalg.norm(
                np.array(self.track.get_coordinate(self.theta, line="int"))
                - np.array(self.track.get_coordinate(self.theta, line="mid"))
            )
        else:
            width = np.linalg.norm(
                np.array(self.track.get_coordinate(self.theta, line="mid"))
                - np.array(self.track.get_coordinate(self.theta, line="out"))
            )
        return width
        
        

