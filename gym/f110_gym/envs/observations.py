"""Module for observation handling in the PBL F110 gmy. 

This module generates the necessary python parts for the observation handling
of the F110 PBL gym environment. 
The code is inteded to be used by calling the `ObsGenerator`, which then 
returns 2 objects:
    - the observation space, as a gym.Spaces object
    - the observator, which is an object which is used then to create the 
        flattened and normalized observation for SB3
"""

import enum
import numpy as np
import math
from gymnasium.spaces.box import Box
from splinify.splinify import SplineTrack
import time

# alias for cross prod because bug in numpy makes it look like code is unreachable
cross_alias = lambda x, y: np.cross(x, y)


class ObservationMode(enum.Enum):
    """supported types of observation spaces are here"""

    FRENET = "frenet"
    TRAJ_FRENET = "trajectory_frenet"


def obs_generator(
    mode: ObservationMode = ObservationMode.TRAJ_FRENET,
    env_params: dict = None,
    use_lidar: bool = False,
    use_previous_actions: bool = False,
    use_pp_action: bool = False,
    use_base_controller: bool = False,
    lidar_len: int = 10,    # TODO remove lidar hardcondng
    n_traj_points: int = 30,  
    traj_points_spacing: float = 1,
    calc_glob_traj: bool = False,
):
    """
    Observation Generator that returns a `gym.spaces` object
    """
    # Check that only supported mode is used
    msg = f"The mode: {mode} is not in the supported list: {[mode for mode in ObservationMode]}"
    assert mode in ObservationMode, msg

    if mode == ObservationMode.FRENET:
        return (
            _get_frenet_obs_space(
                use_lidar,
                lidar_len,
                use_previous_actions,
                use_pp_action,
                use_base_controller,
            ),
            FrenetObservator(
                mode=mode,
                v_min=env_params["v_min"],
                v_max=env_params["v_max"],
                yaw_rate_max=env_params["sv_max"],
                yaw_max=env_params["s_max"],
                car_width=env_params["width"],
                use_lidar=use_lidar,
                use_previous_actions=use_previous_actions,
                use_pp_action=use_pp_action,
                use_base_controller=use_base_controller,
            ),
        )
    elif mode == ObservationMode.TRAJ_FRENET:
        return _get_traj_frenet_obs_space(
            use_lidar,
            lidar_len,
            n_traj_points,
            use_previous_actions,
            use_pp_action,
            use_base_controller,
        ), TrajFrenetObservator(
            mode=mode,
            v_min=env_params["v_min"],
            v_max=env_params["v_max"],
            yaw_rate_max=env_params["sv_max"],
            yaw_max=env_params["s_max"],
            car_width=env_params["width"],
            use_lidar=use_lidar,
            n_traj_points=n_traj_points,
            traj_points_spacing=traj_points_spacing,
            use_previous_actions=use_previous_actions,
            use_pp_action=use_pp_action,
            use_base_controller=use_base_controller,
            calc_glob_traj=calc_glob_traj,
        )


def _get_frenet_obs_space(
    use_lidar, lidar_len, use_previous_actions, use_pp_action, use_base_controller
):
    """
    Returns the gym.spaces object that defines the frenet observation space.
    As the observation space is normalized, bounds are unitary.
    The observation is defined as following:
        - d: deviation from the reference line
        - yaw: yaw relative to the trajectory
        - long_vel: longitudinal velocity respective to the car
        - lat_vel: lateral velocity respective to the car
        - yaw_rate: rate of change of yaw (angular velocity)
    """
    low = np.array(
        [
            -1,  # d
            -1,  # yaw relative to trajectory
            -1,  # longitudinal velocity
            -1,  # lateral velocity
            -1,  # yaw rate
        ]
    )
    if use_lidar:
        low = np.concatenate((low, -np.ones(lidar_len)), axis=None)
    if use_previous_actions:
        low = np.concatenate(
            (
                low,
                np.array(
                    [
                        -1,  # previous steer
                        -1,  # previous speed
                    ]
                ),
            ),
            axis=None,
        )
    if use_pp_action:
        low = np.concatenate(
            (
                low,
                np.array(
                    [
                        -1,  # previous steer
                        -1,  # previous speed
                    ]
                ),
            ),
            axis=None,
        )
    if use_base_controller:
        low = np.concatenate(
            (
                low,
                np.array(
                    [
                        -1,  # previous steer
                        -1,  # previous speed
                    ]
                ),
            ),
            axis=None,
        )
    high = -low

    return Box(low=low, high=high, dtype=np.float32)


def _get_traj_frenet_obs_space(
    use_lidar,
    lidar_len,
    traj_len,
    use_previous_actions,
    use_pp_action,
    use_base_controller,
):
    """
    Returns the gym.spaces object that defines the frenet trajectory observation space.
    As the observation space is normalized, bounds are unitary.
    The observation is defined as following:
        - traj [traj_len]: array of points in front of the car, sampled at a fixed distance
            from the trajectory
        - d: deviation from the reference line
        - yaw: yaw relative to the trajectory
        - long_vel: longitudinal velocity respective to the car
        - lat_vel: lateral velocity respective to the car
        - yaw_rate: rate of change of yaw (angular velocity)
    """
    num_traj_related_keys = 3
    low = np.concatenate(
        [
            -np.ones(
                2 * traj_len * num_traj_related_keys
            ),  # trajectory points, in frame of reference of the car
            -1,  # d
            -1,  # yaw relative to trajectory
            -1,  # longitudinal velocity
            -1,  # lateral velocity
            -1,  # yaw rate
        ],
        axis=None,
    )
    if use_lidar:
        low = np.concatenate((low, -np.ones(lidar_len)), axis=None)
    if use_previous_actions:
        low = np.concatenate(
            (
                low,
                np.array(
                    [
                        -1,  # previous steer
                        -1,  # previous speed
                    ]
                ),
            ),
            axis=None,
        )
    if use_pp_action:
        low = np.concatenate(
            (
                low,
                np.array(
                    [
                        -1,  # previous steer
                        -1,  # previous speed
                    ]
                ),
            ),
            axis=None,
        )
    if use_base_controller:
        low = np.concatenate(
            (
                low,
                np.array(
                    [
                        -1,  # previous steer
                        -1,  # previous speed
                    ]
                ),
            ),
            axis=None,
        )
    high = -low

    return Box(low=low, high=high, dtype=np.float32)


class FrenetObservator:
    """
    The Frenet Observator is an object which can be used to generate frenet observations
    in the PBL F110 gym environment.

    Attributes:
        mode: an ObservationMode object indicating what kind of mode we wnat
            to adoperate. Current options are FRENET and TRAJ_FRENET.
        v_min: a float indicating the minimum velocity of the car
        yaw_rate_max: a float indicating the maximum angualar velocity of the car
        car_width: a float indicating the car's width
        v_max: a float indicating the maximum velocity of the car
        denorm_obs: a dict containing the denormalized observation for external use
    """

    def __init__(
        self,
        mode: ObservationMode,
        v_min: float,
        yaw_rate_max: float,
        yaw_max: float,
        car_width: float,
        v_max: float,
        use_lidar: bool = False,
        lidar_downsampling: int = 108,
        use_previous_actions: bool = False,
        use_pp_action: bool = False,
        use_base_controller: bool = False,
    ) -> None:
        self.mode = mode

        self.minimums = {
            "deviation": 0,
            "rel_heading": -np.pi,
            "longitudinal_vel": v_min,
            "later_vel": v_min,
            "yaw_rate": -yaw_rate_max,
            "yaw": -yaw_max,
        }
        self.maximums = {
            "deviation": 10 * car_width,
            "rel_heading": np.pi,
            "longitudinal_vel": v_max,
            "later_vel": v_max,
            "yaw_rate": yaw_rate_max,
            "yaw": yaw_max,
        }

        self.use_lidar = use_lidar
        if self.use_lidar:
            self.minimums["scans"] = 0
            self.maximums["scans"] = 10  # TODO remove hardcoding
            self.lidar_ds = lidar_downsampling
            self.n_lidar_scans = 1080  # TODO remove hardcoding

        self.use_previous_actions = use_previous_actions
        if self.use_previous_actions:
            self.minimums["previous_s"] = -yaw_max
            self.maximums["previous_s"] = yaw_max
            self.minimums["previous_v"] = v_min
            self.maximums["previous_v"] = v_max

        self.use_pp_action = use_pp_action
        if self.use_pp_action:
            self.minimums["pp_s"] = -yaw_max
            self.maximums["pp_s"] = yaw_max
            self.minimums["pp_v"] = v_min
            self.maximums["pp_v"] = v_max

        self.use_base_controller = use_base_controller
        if self.use_base_controller:
            self.minimums["base_controller_s"] = -yaw_max
            self.maximums["base_controller_s"] = yaw_max
            self.minimums["base_controller_v"] = v_min
            self.maximums["base_controller_v"] = v_max

        self.keys = self._init_keys()
        self.denorm_obs = {key: None for key in self.keys}

        self.print_obs_keys = False

    def _init_keys(self):
        key_list = [
            "deviation",
            "rel_heading",
            "longitudinal_vel",
            "later_vel",
            "yaw_rate",
        ]
        if self.use_lidar:
            key_list.append("scans")
        if self.use_previous_actions:
            key_list.extend(["previous_s", "previous_v"])
        if self.use_pp_action:
            key_list.extend(["pp_s", "pp_v"])
        if self.use_base_controller:
            key_list.extend(["base_controller_s", "base_controller_v"])
        # print(f'The observation is composed of the following elements: {" ".join(key_list)}')
        return key_list

    def get_obs(
        self,
        obs: dict,
        car_position: np.array,
        track_position: np.array,
        track_angle: float,
        track_direction: float,
    ) -> dict:
        # preselection
        obs = self.obs_preselection(obs)

        # fixing base observation
        new_obs = self.get_base_obs(
            obs,
            car_position,
            track_position,
            track_angle,
            track_direction=track_direction,
        ).copy()

        # final normalisation and flattening
        final_obs = self.finalise_observation(new_obs)

        return final_obs

    def obs_preselection(self, obs: dict) -> dict:
        """
        Removes all the unnecessary parts from the observation
        """
        keys_presel = []
        if self.use_lidar:
            keys_presel.append("scans")

        keys_presel.extend(
            [
                "poses_x",
                "poses_y",
                "poses_theta",
                "linear_vels_x",
                "linear_vels_y",
                "ang_vels_z",
            ]
        )
        if self.use_previous_actions:
            keys_presel.extend(["previous_s", "previous_v"])
        if self.use_pp_action:
            keys_presel.extend(["pp_s", "pp_v"])
        if self.use_base_controller:
            keys_presel.extend(["base_controller_s", "base_controller_v"])
        new_obs = {}
        for k in keys_presel:
            new_obs[k] = obs[k]
        obs = new_obs
        if self.use_lidar:
            obs["scans"] = obs["scans"][0]

        return obs

    def get_base_obs(
        self, obs, car_position, track_position, track_angle, track_direction
    ):
        """
        Organizing the not-normalized observations and saving in an internal variable
        for external access in case of need
        """
        sign = np.sign(cross_alias(track_direction, car_position - track_position))
        self.last_car_pos = car_position
        self.denorm_obs["deviation"] = sign * np.linalg.norm(
            car_position - track_position
        )
        self.denorm_obs["rel_heading"] = self.get_rel_head(obs, track_angle)
        self.denorm_obs["longitudinal_vel"] = obs["linear_vels_x"][0]
        self.denorm_obs["later_vel"] = obs["linear_vels_y"][0]
        self.denorm_obs["yaw_rate"] = obs["ang_vels_z"][0]

        if self.use_lidar:
            self.denorm_obs["scans"] = self._process_scans(obs["scans"])

        if self.use_previous_actions:
            self.denorm_obs["previous_s"] = obs["previous_s"]
            self.denorm_obs["previous_v"] = obs["previous_v"]
        if self.use_pp_action:
            self.denorm_obs["pp_s"] = obs["pp_s"]
            self.denorm_obs["pp_v"] = obs["pp_v"]
        if self.use_base_controller:
            self.denorm_obs["base_controller_s"] = obs["base_controller_s"]
            self.denorm_obs["base_controller_v"] = obs["base_controller_v"]

        return self.denorm_obs.copy()

    def finalise_observation(self, new_obs):
        new_obs_norm = self.normalise_observation(new_obs)
        if self.print_obs_keys is False:
            print(
                f'The observation contains the following elements: {", ".join(self.keys)}'
            )
            self.print_obs_keys = True
        obs = np.concatenate(
            [new_obs_norm[key] for key in self.keys], axis=None
        ).astype(np.float32)
        return obs

    def _process_scans(self, scans: list):
        # returns a downsampled number of lidar scans
        return [
            scans[i * self.lidar_ds]
            for i in range(int(self.n_lidar_scans / self.lidar_ds))
        ]

    def get_rel_head(self, obs: dict, track_angle: float) -> float:
        """
        Obtain heading of the car relative to the track
        """

        if track_angle < 0:
            print("Really? angle should always be > 0")
            track_angle += 2 * np.pi
        if not np.isnan(obs["poses_theta"][0]):
            car_angle = obs["poses_theta"][0] % (2 * np.pi)
        else:
            car_angle = 0
        self.last_car_angle = car_angle  # TODO save full last state internally
        rel_head = car_angle - track_angle
        if rel_head >= np.pi:
            rel_head -= 2 * np.pi
        elif rel_head <= -np.pi:
            rel_head += 2 * np.pi

        return rel_head

    def normalise_observation(self, obs):
        """
        Normalises the observation based on the keys present
        in the `keys` attribute

        Args:
            obs: the observation already partially preprocessed
        """
        for k in self.keys:
            # print(f'key:{k}, denormalized value: {obs[k]}')
            obs[k] = np.clip(obs[k], self.minimums[k], self.maximums[k])
            obs[k] = (
                2 * (obs[k] - self.minimums[k]) / (self.maximums[k] - self.minimums[k])
                - 1
            )
            # print(f'key:{k}, normalized value:{obs[k]}')
        return obs

    def get_lidar_coords(self):
        if not self.use_lidar:
            raise ValueError(
                "You cannot try to get the lidar coordinates if the observation is not considering lidar"
            )
        else:
            tot_angle_view = 270  # TODO remove harcoding
            new = np.empty((int(self.n_lidar_scans / self.lidar_ds), 2))
            for i in range(int(self.n_lidar_scans / self.lidar_ds)):
                angle = (
                    i
                    * self.lidar_ds
                    * (tot_angle_view)
                    / (self.n_lidar_scans)
                    * 2
                    * np.pi
                    / 360
                )
                print("angle 1 ", angle)
                # unit x vector
                base_v = np.array([1, 0])
                # center angle wrt car
                angle -= 0.5 * tot_angle_view * 2 * np.pi / 360
                print("angle 2 ", angle)
                angle += self.last_car_angle
                print("angle 3 ", angle)
                co, si = np.cos(-angle), np.sin(-angle)
                rot_mat = np.array(((co, -si), (si, co)))

                rot_v = base_v @ rot_mat

                final_v = rot_v * self.denorm_obs["scans"][i]

                final_pos = self.last_car_pos + final_v

                new[i, :] = final_pos
            return new


class TrajFrenetObservator(FrenetObservator):
    """
    The Trajectory Frenet Observator is an object that extends the Frenet Observator
    and can be used to generate frenet observations with a trajectory in the
    PBL F110 gym environment.

    Attributes:
        mode: an ObservationMode object indicating what kind of mode we wnat
            to adoperate. Current options are FRENET and TRAJ_FRENET.
        v_min: a float indicating the minimum velocity of the car
        yaw_rate_max: a float indicating the maximum angualar velocity of the car
        car_width: a float indicating the car's width
        v_max: a float indicating the maximum velocity of the car
    """

    def __init__(
        self,
        n_traj_points: int,
        traj_points_spacing: float,
        calc_glob_traj: bool,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.n_traj_points = n_traj_points
        self.traj_points_spacing = traj_points_spacing
        for key in self.traj_related_key_list:
            self.denorm_obs[key] = None
        self.minimums["traj"] = -self.n_traj_points * self.traj_points_spacing
        self.maximums["traj"] = -self.minimums["traj"]
        max_track_width = 10  # TODO: remove the hardcoded value
        max_distance = math.sqrt(
            (self.n_traj_points * self.traj_points_spacing) ** 2 + max_track_width**2
        )
        self.minimums["track_int"] = -max_distance
        self.maximums["track_int"] = max_distance
        self.minimums["track_out"] = -max_distance
        self.maximums["track_out"] = max_distance

        self.calc_glob_traj = calc_glob_traj
        self.ignore_traj_info = False

    def _init_keys(self):
        base_keys = super()._init_keys()
        self.traj_related_key_list = ["track_int", "traj", "track_out"]
        for key in self.traj_related_key_list:
            base_keys.append(key)

        return base_keys

    def get_obs(
        self,
        obs: dict,
        car_position: np.array,
        track_position: np.array,
        track_angle: float,
        track_direction: float,
        racetrack: SplineTrack,
        cur_theta: float,
    ) -> dict:
        # preselection
        obs = self.obs_preselection(obs)

        # fixing base observation
        new_obs = self.get_base_obs(
            obs, car_position, track_position, track_angle, track_direction
        )

        # add trajectory
        car_angle = obs["poses_theta"][0] % (2 * np.pi)

        if self.ignore_traj_info:
            new_obs["traj"] = new_obs["track_int"] = new_obs["track_out"] = np.zeros(
                (self.n_traj_points, 2)
            )

            self.denorm_obs["traj"] = new_obs["traj"].copy()
            self.denorm_obs["track_int"] = new_obs["track_int"].copy()
            self.denorm_obs["track_out"] = new_obs["track_out"].copy()

            return self.finalise_observation(new_obs)

        # obtains, rotates and translates the trajectory points
        new_obs["traj"] = self._compute_trajectory(
            car_angle, car_position, racetrack, cur_theta
        )
        new_obs["track_int"] = self._compute_trajectory(
            car_angle, car_position, racetrack, cur_theta, line="int"
        )
        new_obs["track_out"] = self._compute_trajectory(
            car_angle, car_position, racetrack, cur_theta, line="out"
        )

        self.denorm_obs["traj"] = new_obs["traj"].copy()
        self.denorm_obs["track_int"] = new_obs["track_int"].copy()
        self.denorm_obs["track_out"] = new_obs["track_out"].copy()

        if self.calc_glob_traj:
            self.denorm_obs["glob_traj"] = self._compute_global_trajectory(
                racetrack, cur_theta
            )
            self.denorm_obs["glob_track_int"] = self._compute_global_trajectory(
                racetrack, cur_theta, line="int"
            )
            self.denorm_obs["glob_track_out"] = self._compute_global_trajectory(
                racetrack, cur_theta, line="out"
            )

        # final normalisation and flattening
        final_obs = self.finalise_observation(new_obs)

        return final_obs

    def _compute_trajectory(
        self, car_angle, car_position, racetrack, cur_theta, line="mid"
    ):
        """
        Computes the trajectory points.

        Args:
        car_angle (float): The angle of the car.
        car_position (np.array): The current position of the car.
        racetrack (SplineTrack): The racetrack object.
        cur_theta (float): The current theta value on the track.

        Returns:
        np.array: Array of trajectory points.
        """
        rot_matr = np.array(
            [
                [np.cos(car_angle), np.sin(car_angle)],
                [-np.sin(car_angle), np.cos(car_angle)],
            ]
        )

        trajectory = np.array(
            [
                rot_matr
                @ (np.array(racetrack.get_coordinate(th, line=line)) - car_position)
                for th in np.linspace(
                    cur_theta,
                    cur_theta + self.n_traj_points * self.traj_points_spacing,
                    self.n_traj_points,
                )
            ]
        )

        return trajectory

    def _compute_global_trajectory(self, racetrack, cur_theta, line="mid"):
        """
        Computes the global trajectory points.

        Args:
        racetrack (SplineTrack): The racetrack object.
        cur_theta (float): The current theta value on the track.

        Returns:
        np.array: Array of global trajectory points.
        """
        global_traj = np.array(
            [
                np.array(racetrack.get_coordinate(th, line=line))
                for th in np.linspace(
                    cur_theta,
                    cur_theta + self.n_traj_points * self.traj_points_spacing,
                    self.n_traj_points,
                )
            ]
        )

        return global_traj
