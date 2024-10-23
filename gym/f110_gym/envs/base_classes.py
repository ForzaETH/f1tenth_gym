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
Prototype of base classes
Replacement of the old RaceCar, Simulator classes in C++
Author: Hongrui Zheng
"""

from enum import Enum
import warnings
from typing import List

import numpy as np
from numba import njit
import os.path as osp
import yaml

from f110_gym.envs.dynamic_models import (
    vehicle_dynamics_st,
    pid,
    accl_constraints,
    steering_constraint,
)
from f110_gym.envs.dynamic_models_ros import (
    vehicle_dynamics_st_update_linear,
    vehicle_dynamics_st_update_pacejka,
)
from f110_gym.envs.laser_models import ScanSimulator2D, check_ttc_jit, ray_cast
from f110_gym.envs.collision_models import get_vertices, collision_multiple
from f110_gym.envs.utils import read_config
import copy
import random


class Integrator(Enum):
    RK4 = 1
    Euler = 2


class RaceCar(object):
    """
    Base level race car class, handles the physics and laser scan of a single vehicle

    Data Members:
        params (dict): vehicle parameters dictionary
        is_ego (bool): ego identifier
        time_step (float): physics timestep
        num_beams (int): number of beams in laser
        fov (float): field of view of laser
        state (np.ndarray (7, )): state vector [x, y, theta, vel, steer_angle, ang_vel, slip_angle]
        odom (np.ndarray(13, )): odometry vector [x, y, z, qx, qy, qz, qw, linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
        accel (float): current acceleration input
        steer_angle_vel (float): current steering velocity input
        in_collision (bool): collision indicator

    """

    # static objects that don't need to be stored in class instances
    scan_simulator = None
    cosines = None
    scan_angles = None
    side_distances = None

    def __init__(
        self,
        params,
        p_linear,
        p_pacejka,
        seed,
        is_ego=False,
        time_step=0.01,
        num_beams=1080,
        fov=4.7,
        integrator=Integrator.Euler,
        dynamic_mode="gym_dynamics",
        enable_steering_delay=False,
        delay_buffer_size=5,
        randomize_steering_delay=False,
        car_size_multiplier=1,
    ):
        """
        Init function

        Args:
            params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max', 'v_min', 'v_max', 'length', 'width'}
            is_ego (bool, default=False): ego identifier
            time_step (float, default=0.01): physics sim time step
            num_beams (int, default=1080): number of beams in the laser scan
            fov (float, default=4.7): field of view of the laser

        Returns:
            None
        """

        # initialization
        self.params = params
        self.p_linear = p_linear
        self.p_pacejka = p_pacejka
        self.seed = seed
        self.is_ego = is_ego
        self.time_step = time_step
        self.num_beams = num_beams
        self.fov = fov
        self.car_size_multiplier = car_size_multiplier
        self.integrator = integrator
        if self.integrator is Integrator.RK4:
            warnings.warn(
                f"Chosen integrator is RK4. This is different from previous versions of the gym."
            )
        self.dynamic_mode = dynamic_mode

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        self.state = np.zeros((7,))
        self.accelerations = None
        self.jerk = None

        # pose of opponents in the world
        self.opp_poses = None

        # control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0

        # steering delay buffer
        self.enable_steering_delay = enable_steering_delay
        self.steer_buffer_size = delay_buffer_size
        self.randomize_steering_delay = randomize_steering_delay

        # collision identifier
        self.in_collision = False

        # collision threshold for iTTC to environment
        self.ttc_thresh = 0.005

        # initialize scan sim
        if self.scan_simulator is None:
            self.scan_rng = np.random.default_rng(seed=self.seed)
            self.scan_simulator = ScanSimulator2D(num_beams, fov)

            scan_ang_incr = self.scan_simulator.get_increment()

            # angles of each scan beam, distance from lidar to edge of car at each beam, and precomputed cosines of each angle
            self.cosines = np.zeros((num_beams,))
            self.scan_angles = np.zeros((num_beams,))
            self.side_distances = np.zeros((num_beams,))

            dist_sides = self.car_size_multiplier * params["width"] / 2.0
            dist_fr = self.car_size_multiplier * (params["lf"] + params["lr"]) / 2.0

            for i in range(num_beams):
                angle = -fov / 2.0 + i * scan_ang_incr
                self.scan_angles[i] = angle
                self.cosines[i] = np.cos(angle)

                if angle > 0:
                    if angle < np.pi / 2:
                        # between 0 and pi/2
                        to_side = dist_sides / np.sin(angle)
                        to_fr = dist_fr / np.cos(angle)
                        self.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between pi/2 and pi
                        to_side = dist_sides / np.cos(angle - np.pi / 2.0)
                        to_fr = dist_fr / np.sin(angle - np.pi / 2.0)
                        self.side_distances[i] = min(to_side, to_fr)
                else:
                    if angle > -np.pi / 2:
                        # between 0 and -pi/2
                        to_side = dist_sides / np.sin(-angle)
                        to_fr = dist_fr / np.cos(-angle)
                        self.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between -pi/2 and -pi
                        to_side = dist_sides / np.cos(-angle - np.pi / 2)
                        to_fr = dist_fr / np.sin(-angle - np.pi / 2)
                        self.side_distances[i] = min(to_side, to_fr)

    def update_params(self, params):
        """
        Updates the physical parameters of the vehicle
        Note that does not need to be called at initialization of class anymore

        Args:
            params (dict): new parameters for the vehicle

        Returns:
            None
        """
        self.params = params

    def set_map(self, map_path, map_ext):
        """
        Sets the map for scan simulator

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file
        """
        self.scan_simulator.set_map(map_path, map_ext)

    def reset(self, pose):
        """
        Resets the vehicle to a pose

        Args:
            pose (np.ndarray (3, )): pose to reset the vehicle to

        Returns:
            None
        """
        # clear control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0
        # clear collision indicator
        self.in_collision = False
        # clear state
        self.state = np.zeros((7,))
        self.state[0:2] = pose[0:2]
        self.state[4] = pose[2]
        self.steer_buffer = np.zeros((self.steer_buffer_size,))
        self.vel_buffer = np.zeros((self.steer_buffer_size,))
        # reset scan random generator
        self.scan_rng = np.random.default_rng(seed=self.seed)

    def ray_cast_agents(self, scan):
        """
        Ray cast onto other agents in the env, modify original scan

        Args:
            scan (np.ndarray, (n, )): original scan range array

        Returns:
            new_scan (np.ndarray, (n, )): modified scan
        """

        # starting from original scan
        new_scan = scan

        # loop over all opponent vehicle poses
        for opp_pose in self.opp_poses:
            # get vertices of current oppoenent
            opp_vertices = get_vertices(
                opp_pose, self.params["length"], self.params["width"]
            )

            new_scan = ray_cast(
                np.append(self.state[0:2], self.state[4]),
                new_scan,
                self.scan_angles,
                opp_vertices,
            )

        return new_scan

    def check_ttc(self, current_scan):
        """
        Check iTTC against the environment, sets vehicle states accordingly if collision occurs.
        Note that this does NOT check collision with other agents.

        state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        Args:
            current_scan

        Returns:
            None
        """

        in_collision = check_ttc_jit(
            current_scan,
            self.state[3],
            self.scan_angles,
            self.cosines,
            self.side_distances,
            self.ttc_thresh,
        )

        # update state
        self.in_collision = in_collision

        return in_collision

    def update_pose(self, raw_steer, raw_vel):
        """
        Steps the vehicle's physical simulation

        Args:
            steer (float): desired steering angle
            vel (float): desired longitudinal velocity

        Returns:
            current_scan
        """

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        # steering delay
        if self.enable_steering_delay:
            if self.steer_buffer.shape[0] < self.steer_buffer_size:
                steer = 0.0
                self.steer_buffer = np.append(raw_steer, self.steer_buffer)
                vel = 0.0
                self.vel_buffer = np.append(raw_vel, self.vel_buffer)
            else:
                if self.randomize_steering_delay:
                    random_index = -random.randint(
                        max(1, self.steer_buffer.shape[0] - 2),
                        self.steer_buffer.shape[0],
                    )
                else:
                    random_index = -self.steer_buffer.shape[0]
                steer = self.steer_buffer[random_index]
                self.steer_buffer = np.delete(self.steer_buffer, random_index)
                self.steer_buffer = np.append(raw_steer, self.steer_buffer)

                vel = self.vel_buffer[random_index]
                self.vel_buffer = np.delete(self.vel_buffer, random_index)
                self.vel_buffer = np.append(raw_vel, self.vel_buffer)
        else:
            steer = raw_steer
            vel = raw_vel

        # disable delay for speed
        vel = raw_vel

        # steering angle velocity input to steering velocity acceleration input
        if self.params["v_min"] == 0:
            accl = 0
            sv = 0
        else:
            accl, sv = pid(
                vel,
                steer,
                self.state[3],
                self.state[2],
                self.params["sv_max"],
                self.params["a_max"],
                self.params["a_min"],
                self.params["v_max"],
                self.params["v_min"],
            )

        self.last_acc = accl

        accl = accl_constraints(
            self.state[3],
            accl,
            self.params["v_switch"],
            self.params["a_max"],
            self.params["v_min"],
            self.params["v_max"],
        )
        sv = steering_constraint(
            self.state[2],
            sv,
            self.params["s_min"],
            self.params["s_max"],
            self.params["sv_min"],
            self.params["sv_max"],
        )

        if self.dynamic_mode == "ros_dynamics_linear":
            self.state = vehicle_dynamics_st_update_linear(
                self.state, accl, sv, self.p_linear, self.time_step
            )
        elif self.dynamic_mode == "ros_dynamics_pacejka":
            self.state, accelerations = vehicle_dynamics_st_update_pacejka(
                self.state, accl, sv, self.p_pacejka, self.time_step
            )
            if self.accelerations is None:
                self.accelerations = accelerations

            self.jerk = (accelerations - self.accelerations) / self.time_step
            self.accelerations = accelerations
            
        else:
            if self.integrator is Integrator.RK4:
                # RK4 integration
                k1 = vehicle_dynamics_st(
                    self.state,
                    np.array([sv, accl]),
                    self.params["mu"],
                    self.params["C_Sf"],
                    self.params["C_Sr"],
                    self.params["lf"],
                    self.params["lr"],
                    self.params["h"],
                    self.params["m"],
                    self.params["I"],
                    self.params["s_min"],
                    self.params["s_max"],
                    self.params["sv_min"],
                    self.params["sv_max"],
                    self.params["v_switch"],
                    self.params["a_max"],
                    self.params["v_min"],
                    self.params["v_max"],
                )

                k2_state = self.state + self.time_step * (k1 / 2)

                k2 = vehicle_dynamics_st(
                    k2_state,
                    np.array([sv, accl]),
                    self.params["mu"],
                    self.params["C_Sf"],
                    self.params["C_Sr"],
                    self.params["lf"],
                    self.params["lr"],
                    self.params["h"],
                    self.params["m"],
                    self.params["I"],
                    self.params["s_min"],
                    self.params["s_max"],
                    self.params["sv_min"],
                    self.params["sv_max"],
                    self.params["v_switch"],
                    self.params["a_max"],
                    self.params["v_min"],
                    self.params["v_max"],
                )

                k3_state = self.state + self.time_step * (k2 / 2)

                k3 = vehicle_dynamics_st(
                    k3_state,
                    np.array([sv, accl]),
                    self.params["mu"],
                    self.params["C_Sf"],
                    self.params["C_Sr"],
                    self.params["lf"],
                    self.params["lr"],
                    self.params["h"],
                    self.params["m"],
                    self.params["I"],
                    self.params["s_min"],
                    self.params["s_max"],
                    self.params["sv_min"],
                    self.params["sv_max"],
                    self.params["v_switch"],
                    self.params["a_max"],
                    self.params["v_min"],
                    self.params["v_max"],
                )

                k4_state = self.state + self.time_step * k3

                k4 = vehicle_dynamics_st(
                    k4_state,
                    np.array([sv, accl]),
                    self.params["mu"],
                    self.params["C_Sf"],
                    self.params["C_Sr"],
                    self.params["lf"],
                    self.params["lr"],
                    self.params["h"],
                    self.params["m"],
                    self.params["I"],
                    self.params["s_min"],
                    self.params["s_max"],
                    self.params["sv_min"],
                    self.params["sv_max"],
                    self.params["v_switch"],
                    self.params["a_max"],
                    self.params["v_min"],
                    self.params["v_max"],
                )

                # dynamics integration
                self.state = self.state + self.time_step * (1 / 6) * (
                    k1 + 2 * k2 + 2 * k3 + k4
                )

            elif self.integrator is Integrator.Euler:
                f = vehicle_dynamics_st(
                    self.state,
                    np.array([sv, accl]),
                    self.params["mu"],
                    self.params["C_Sf"],
                    self.params["C_Sr"],
                    self.params["lf"],
                    self.params["lr"],
                    self.params["h"],
                    self.params["m"],
                    self.params["I"],
                    self.params["s_min"],
                    self.params["s_max"],
                    self.params["sv_min"],
                    self.params["sv_max"],
                    self.params["v_switch"],
                    self.params["a_max"],
                    self.params["v_min"],
                    self.params["v_max"],
                )
                self.state = self.state + self.time_step * f
            else:
                raise SyntaxError(
                    f"Invalid Integrator Specified. Provided {self.integrator.name}. Possible options: 'RK4', 'Euler', 'ros_dynamics_linear', 'ros_dynamics_pacejka'"
                )

        # bound yaw angle
        if self.state[4] > 2 * np.pi:
            self.state[4] = self.state[4] - 2 * np.pi
        elif self.state[4] < 0:
            self.state[4] = self.state[4] + 2 * np.pi

        self.state[3] = np.clip(
            self.state[3], self.params["v_min"], self.params["v_max"]
        )
        self.state[2] = np.clip(
            self.state[2], self.params["s_min"], self.params["s_max"]
        )

        # update scan
        current_scan = self.scan_simulator.scan(
            np.append(self.state[0:2], self.state[4]), self.scan_rng
        )

        return current_scan

    def update_opp_poses(self, opp_poses):
        """
        Updates the vehicle's information on other vehicles

        Args:
            opp_poses (np.ndarray(num_other_agents, 3)): updated poses of other agents

        Returns:
            None
        """
        self.opp_poses = opp_poses

    def update_scan(self, agent_scans, agent_index):
        """
        Steps the vehicle's laser scan simulation
        Separated from update_pose because needs to update scan based on NEW poses of agents in the environment

        Args:
            agent scans list (modified in-place),
            agent index (int)

        Returns:
            None
        """

        current_scan = agent_scans[agent_index]

        # check ttc
        self.check_ttc(current_scan)

        # ray cast other agents to modify scan
        new_scan = self.ray_cast_agents(current_scan)

        agent_scans[agent_index] = new_scan


class Simulator(object):
    """
    Simulator class, handles the interaction and update of all vehicles in the environment

    Data Members:
        num_agents (int): number of agents in the environment
        time_step (float): physics time step
        agent_poses (np.ndarray(num_agents, 3)): all poses of all agents
        agents (list[RaceCar]): container for RaceCar objects
        collisions (np.ndarray(num_agents, )): array of collision indicator for each agent
        collision_idx (np.ndarray(num_agents, )): which agent is each agent in collision with

    """

    def __init__(
        self,
        car_params,
        num_agents,
        seed,
        time_step=0.01,
        ego_idx=0,
        integrator=Integrator.RK4,
        dynamic_mode="gym_dynamics",
        redraw_upon_reset=False,
        randomisation_stds=None,
        enable_steering_delay=False,
        delay_buffer_size=5,
        randomize_steering_delay=False,
        car_size_multiplier=1,
        friction_coefficient=1,
        position_noise_std=0.0,
        yaw_noise_std=0.0,
        velocity_noise_std=0.0,
        yaw_rate_noise_std=0.0,
        no_lateral_velocity=False,
    ):
        """
        Init function

        Args:
            params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max', 'v_min', 'v_max', 'length', 'width'}
            num_agents (int): number of agents in the environment
            seed (int): seed of the rng in scan simulation
            time_step (float, default=0.01): physics time step
            ego_idx (int, default=0): ego vehicle's index in list of agents

        Returns:
            None
        """

        p_linear = read_config(
            osp.join(osp.dirname(osp.abspath(__file__)), "SIM_linear.yaml")
        )
        p_linear.update(car_params)
        p_linear['mu'] = friction_coefficient

        p_pacejka = read_config(
            osp.join(osp.dirname(osp.abspath(__file__)), "SIM_pacejka.yaml")
        )
        p_pacejka.update(car_params)
        p_pacejka['mu'] = friction_coefficient

        p_gym_linear = read_config(
            osp.join(osp.dirname(osp.abspath(__file__)), "SIM_gym_linear.yaml")
        )
        p_gym_linear.update(car_params)
        p_gym_linear['mu'] = friction_coefficient

        self.num_agents = num_agents
        self.seed = seed
        self.time_step = time_step
        self.ego_idx = ego_idx
        self.car_params = car_params
        self.params = p_gym_linear
        self.p_linear = p_linear
        self.p_pacejka = p_pacejka
        self.params_original = copy.deepcopy(self.params)
        self.p_linear_original = copy.deepcopy(p_linear)
        self.p_pacejka_original = copy.deepcopy(p_pacejka)
        self.agent_poses = np.empty((self.num_agents, 3))
        self.agents = []
        self.collisions = np.zeros((self.num_agents,))
        self.collision_idx = -1 * np.ones((self.num_agents,))
        self.redraw_upon_reset = redraw_upon_reset
        self.randomisation_stds = randomisation_stds
        self.dynamic_mode = dynamic_mode
        self.enable_steering_delay = enable_steering_delay
        self.delay_buffer_size = delay_buffer_size
        self.randomize_steering_delay = randomize_steering_delay
        self.car_size_multiplier = car_size_multiplier
        
        self.no_lateral_velocity = no_lateral_velocity
        self.position_noise_std = position_noise_std
        self.yaw_noise_std = yaw_noise_std
        self.velocity_noise_std = velocity_noise_std
        self.yaw_rate_noise_std = yaw_rate_noise_std

        # noise injection if stds dict is passed
        if (self.randomisation_stds is not None) and self.redraw_upon_reset:
            self.apply_randomisation()

        # initializing agents
        for i in range(self.num_agents):
            if i == ego_idx:
                ego_car = RaceCar(
                    self.params,
                    self.p_linear,
                    self.p_pacejka,
                    self.seed,
                    is_ego=True,
                    time_step=self.time_step,
                    integrator=integrator,
                    dynamic_mode=self.dynamic_mode,
                    enable_steering_delay=self.enable_steering_delay,
                    delay_buffer_size=self.delay_buffer_size,
                    randomize_steering_delay=self.randomize_steering_delay,
                    car_size_multiplier=self.car_size_multiplier,
                )
                self.agents.append(ego_car)
            else:
                agent = RaceCar(
                    self.params,
                    self.p_linear,
                    self.p_pacejka,
                    self.seed,
                    is_ego=False,
                    time_step=self.time_step,
                    integrator=integrator,
                    dynamic_mode=self.dynamic_mode,
                    enable_steering_delay=self.enable_steering_delay,
                    delay_buffer_size=self.delay_buffer_size,
                    randomize_steering_delay=self.randomize_steering_delay,
                    car_size_multiplier=self.car_size_multiplier,
                )
                self.agents.append(agent)

    def set_map(self, map_path, map_ext):
        """
        Sets the map of the environment and sets the map for scan simulator of each agent

        Args:
            map_path (str): path to the map yaml file
            map_ext (str): extension for the map image file

        Returns:
            None
        """
        for agent in self.agents:
            agent.set_map(map_path, map_ext)

    def update_params(self, params, agent_idx=-1):
        """
        Updates the params of agents, if an index of an agent is given, update only that agent's params

        Args:
            params (dict): dictionary of params, see details in docstring of __init__
            agent_idx (int, default=-1): index for agent that needs param update, if negative, update all agents

        Returns:
            None
        """
        if agent_idx < 0:
            # update params for all
            for agent in self.agents:
                agent.update_params(params)
        elif agent_idx >= 0 and agent_idx < self.num_agents:
            # only update one agent's params
            self.agents[agent_idx].update_params(params)
        else:
            # index out of bounds, throw error
            raise IndexError("Index given is out of bounds for list of agents.")

    def check_collision(self):
        """
        Checks for collision between agents using GJK and agents' body vertices

        Args:
            None

        Returns:
            None
        """
        # get vertices of all agents
        all_vertices = np.empty((self.num_agents, 4, 2))
        for i in range(self.num_agents):
            all_vertices[i, :, :] = get_vertices(
                np.append(self.agents[i].state[0:2], self.agents[i].state[4]),
                self.car_params["length"],
                self.car_params["width"],
            )
        self.collisions, self.collision_idx = collision_multiple(all_vertices)

    def step(self, control_inputs):
        """
        Steps the simulation environment

        Args:
            control_inputs (np.ndarray (num_agents, 2)): control inputs of all agents, first column is desired steering angle, second column is desired velocity

        Returns:
            observations (dict): dictionary for observations: poses of agents, current laser scan of each agent, collision indicators, etc.
        """

        agent_scans = []

        # looping over agents
        for i, agent in enumerate(self.agents):
            # update each agent's pose
            current_scan = agent.update_pose(control_inputs[i, 0], control_inputs[i, 1])
            agent_scans.append(current_scan)

            # update sim's information of agent poses
            self.agent_poses[i, :] = np.append(agent.state[0:2], agent.state[4])

        # check collisions between all agents
        self.check_collision()

        for i, agent in enumerate(self.agents):
            # update agent's information on other agents
            opp_poses = np.concatenate(
                (self.agent_poses[0:i, :], self.agent_poses[i + 1 :, :]), axis=0
            )
            agent.update_opp_poses(opp_poses)

            # update each agent's current scan based on other agents
            agent.update_scan(agent_scans, i)

            # update agent collision with environment
            if agent.in_collision:
                self.collisions[i] = 1.0

        # fill in observations
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        # collision_angles is removed from observations
        observations = {
            "ego_idx": self.ego_idx,
            "scans": [],
            "poses_x": [],
            "poses_y": [],
            "poses_theta": [],
            "linear_vels_x": [],
            "linear_vels_y": [],
            "ang_vels_z": [],
            "collisions": self.collisions,
        }
        for i, agent in enumerate(self.agents):
            observations["scans"].append(agent_scans[i])
            observations["poses_x"].append(agent.state[0])
            observations["poses_y"].append(agent.state[1])
            observations["poses_theta"].append(agent.state[4])
            observations["linear_vels_x"].append(
                agent.state[3] * np.cos(agent.state[6])
            )
            observations["linear_vels_y"].append(
                agent.state[3] * np.sin(agent.state[6])
            )
            observations["ang_vels_z"].append(agent.state[5])
        
        observations = self.add_noise(observations)

        # a typical setup in the physical F110 car, that can be replicated here like this
        if self.no_lateral_velocity:
            for i in range(self.num_agents):
                observations["linear_vels_y"][i] = 0.0

        return observations

    def reset(self, poses):
        """
        Resets the simulation environment by given poses

        Arges:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            None
        """
        # Apply randomisation of params after reset
        if (self.randomisation_stds is not None) and self.redraw_upon_reset:
            self.apply_randomisation()

        if poses.shape[0] != self.num_agents:
            raise ValueError(
                "Number of poses for reset does not match number of agents."
            )

        # loop over poses to reset
        for i in range(self.num_agents):
            self.agents[i].reset(poses[i, :])

        # return none to not crash SB3
        return None

    def apply_randomisation(self):
        original_params = dict()
        new_params = dict()
        for key, val in self.randomisation_stds.items():
            assert (
                "_std" in key
            ), "Randomisations std is not given correctly: {}".format(key)
            if self.dynamic_mode == "ros_dynamics_linear":
                self.p_linear[key.split("_std")[0]] = self.p_linear_original[
                    key.split("_std")[0]
                ] + np.random.normal(scale=val)

                original_params[key.split("_std")[0]] = self.p_linear_original[
                    key.split("_std")[0]
                ]
                new_params[key.split("_std")[0]] = self.p_linear[key.split("_std")[0]]

            elif self.dynamic_mode == "ros_dynamics_pacejka":
                self.p_pacejka[key.split("_std")[0]] = self.p_pacejka_original[
                    key.split("_std")[0]
                ] + np.clip(np.random.normal(scale=val), -2 * val, 2 * val)

                original_params[key.split("_std")[0]] = self.p_pacejka_original[
                    key.split("_std")[0]
                ]
                new_params[key.split("_std")[0]] = self.p_pacejka[key.split("_std")[0]]

            else:
                self.params[key.split("_std")[0]] = self.params_original[
                    key.split("_std")[0]
                ] + np.random.normal(scale=val)

                original_params[key.split("_std")[0]] = self.params_original[
                    key.split("_std")[0]
                ]
                new_params[key.split("_std")[0]] = self.params[key.split("_std")[0]]

        # print(
        #     "Applied randomisation of original params {} to {}: ".format(
        #         original_params,
        #         {key: new_params[key] for key in original_params.keys()},
        #     )
        # )

    def add_noise(self, observations):
        """Applies noise to the observations in case standard deviations are provided

        Args:
            observations dict of lists: dictionary of observations

        Returns:
            perturbed_observations dict of lists: dictionary of observations with noise
        """
        
        for i in range(self.num_agents):
            if self.position_noise_std > 0:
                noise_x = np.clip(np.random.normal(scale=self.position_noise_std), -3 * self.position_noise_std, 3 * self.position_noise_std)
                noise_y = np.clip(np.random.normal(scale=self.position_noise_std), -3 * self.position_noise_std, 3 * self.position_noise_std) 
                observations["poses_x"][i] += noise_x
                observations["poses_y"][i] += noise_y
                
            if self.yaw_noise_std > 0:
                noise_theta = np.clip(np.random.normal(scale=self.yaw_noise_std), -3 * self.yaw_noise_std, 3 * self.yaw_noise_std)
                observations["poses_theta"][i] += noise_theta
                
            if self.velocity_noise_std > 0:
                noise_vel_x = np.clip(np.random.normal(scale=self.velocity_noise_std), -3 * self.velocity_noise_std, 3 * self.velocity_noise_std)
                noise_vel_y = np.clip(np.random.normal(scale=self.velocity_noise_std), -3 * self.velocity_noise_std, 3 * self.velocity_noise_std)
                observations["linear_vels_x"][i] += noise_vel_x
                observations["linear_vels_y"][i] += noise_vel_y
                
            if self.yaw_rate_noise_std > 0:
                noise_ang_vel_z = np.clip(np.random.normal(scale=self.yaw_rate_noise_std), -3 * self.yaw_rate_noise_std, 3 * self.yaw_rate_noise_std)
                observations["ang_vels_z"][i] += noise_ang_vel_z
                
        return observations