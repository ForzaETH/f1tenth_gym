#!/usr/bin/env python3
import os
import os.path as osp
import numpy as np
import pandas as pd
import time
import yaml
import pickle
import json
import logging

with open(
    f"{os.path.normpath(osp.abspath(__file__) + os.sep + os.pardir)}/recorder_config.yaml",
    "r",
) as config_file:
    recorder_config = yaml.load(config_file, Loader=yaml.FullLoader)
    config_file.close()


def recording(recorder, env, time, action):
    state = env.unwrapped.sim.agents[0].state
    pos_x = state[0]
    pos_y = state[1]
    frenet_s = env.unwrapped.theta
    frenet_d = env.unwrapped.observator.denorm_obs['deviation']
    yaw_angle = state[2]
    steering_angle = state[4]
    speed = state[3]
    tire_slip = state[6]
    action = np.reshape(action, (-1,))
    steering_input = action[0]
    try:
        steering_input_pp = env.unwrapped.pp_s
        speed_input_pp = env.unwrapped.pp_v
    except AttributeError:
        steering_input_pp = 0
        speed_input_pp = 0
        print("PP action is not available, RLPP is not being used")
    steering_input_rl = steering_input - steering_input_pp
    speed_input = action[1]
    speed_input_rl = speed_input - speed_input_pp

    recorder.update(
        time / 100,
        pos_x,
        pos_y,
        frenet_s,
        frenet_d,
        yaw_angle,
        steering_angle,
        speed,
        tire_slip,
        steering_input,
        steering_input_pp,
        steering_input_rl,
        speed_input,
        speed_input_pp,
        speed_input_rl,
    )


def initialize_recorder(env, save_path: str = None):

    all_mappoints = env.unwrapped.renderer.map_points[:, 0:2] / 50
    map_waypoints = np.vstack(
        (env.unwrapped.waypoints[:, 1], env.unwrapped.waypoints[:, 2])
    ).T

    return GYM_Data_Recorder(all_mappoints, map_waypoints, save_path=save_path)


class GYM_Data_Recorder:
    def __init__(self, track_bound_coords, traj_coords, save_path: str = None):
        ### Recorder configuration ###
        if save_path is not None:
            self.save_root_dir = save_path
        else:
            logging.warn(f"Logging with global path is deprecated and should not be used!\nSave path is provided: {save_path}")
            self.save_root_dir = os.path.expanduser("~") + recorder_config["save_root_dir"]

        self.car_type = "F110"
        self.recording_dict = dict()

        ### map related ###
        self.global_waypoints_coords = traj_coords
        self.track_bound_coords = track_bound_coords

        ### Add new info with its unit here
        self.info_unit_mapping = {
            "time": r"[s]",
            "pos_x": r"[m]",
            "pos_y": r"[m]",
            "frenet_s": r"[m]",
            "frenet_d": r"[m]",
            "yaw_angle": r"[rad]",
            "steering": r"[rad]",
            "speed": r"[\frac{\mathrm{m}}{\mathrm{s}}]",
            "tire_slip": r"[rad]",
            "steering_input": r"[rad]",
            "steering_input_pp": r"[rad]",
            "steering_input_rl": r"[rad]",
            "speed_input": r"[\frac{\mathrm{m}}{\mathrm{s}}]",
            "speed_input_pp": r"[\frac{\mathrm{m}}{\mathrm{s}}]",
            "speed_input_rl": r"[\frac{\mathrm{m}}{\mathrm{s}}]"
        }

        self.car_raw_info_df = pd.DataFrame(columns=list(self.info_unit_mapping.keys()))

        self.recording_start_time = None
        self.started = False

        self.create_new_record_archive()
        self.save_track_and_traj()
        self.save_info_unit_mapping()

    def update(
        self,
        time,
        pos_x,
        pos_y,
        frenet_s,
        frenet_d,
        yaw_angle,
        steering_angle,
        speed,
        tire_slip,
        steering_input,
        steering_input_pp,
        steering_input_rl,
        speed_input,
        speed_input_pp,
        speed_input_rl,
    ):

        if not self.started:
            print(f"Start the recording!")
            self.recording_start_time = time
            self.started = True

        self.current_time = time - self.recording_start_time

        current_car_info = np.array(
            [
                time,
                pos_x,
                pos_y,
                frenet_s,
                frenet_d,
                yaw_angle,
                steering_angle,
                speed,
                tire_slip,
                steering_input,
                steering_input_pp,
                steering_input_rl,
                speed_input,
                speed_input_pp,
                speed_input_rl,
            ]
        )
        current_car_info = np.reshape(current_car_info, (-1, current_car_info.shape[0]))

        current_car_info_df = pd.DataFrame(
            current_car_info, columns=list(self.car_raw_info_df)
        )
        self.car_raw_info_df = pd.concat(
            [self.car_raw_info_df, current_car_info_df], ignore_index=True
        )

        return

    def finalize_results(self):
        print("Finishing recording.....")
        save_path = osp.normpath(
            self.save_dir + os.sep + f"car_raw_info.csv"
        )
        self.car_raw_info_df.to_csv(save_path, sep="\t", encoding="utf-8", index=False)
        print(f"Saving data file to {save_path}")

        with open(
            f"{self.save_root_dir}/recorder_config.yaml",
            "w",
        ) as config_file:
            recorder_config["last_time_stamp"] = self.timestr
            recorder_config["last_car_type"] = self.car_type
            yaml.dump(recorder_config, config_file)
            config_file.close()

    def save_info_unit_mapping(self):
        save_path_info_unit_mapping = osp.normpath(
            self.save_dir + os.sep + "info_unit_mapping.pkl"
        )
        with open(save_path_info_unit_mapping, "wb") as mapping_file:
            pickle.dump(self.info_unit_mapping, mapping_file)
            print("The info_unit_mapping is saved successfully to file")
            mapping_file.close()

    def save_track_and_traj(self):
        save_path_track = osp.normpath(self.save_dir + os.sep + "track_bound")
        np.save(save_path_track, self.track_bound_coords)
        save_path_traj = osp.normpath(self.save_dir + os.sep + "traj")
        np.save(save_path_traj, self.global_waypoints_coords)
        print("The track boundary and the trajectory are saved successfully to files")

    def create_new_record_archive(self):
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        self.save_dir = osp.normpath(
            self.save_root_dir
            + os.sep
            + f"{self.car_type}_recordings"
            + os.sep
            + self.timestr
        )
        os.makedirs(self.save_dir)

    def save_lap_time_list(self, lap_time_list):
        lap_time_info_path = osp.normpath(self.save_dir + os.sep + "lap_time_list.txt")
        with open(lap_time_info_path, 'w') as file:
            json.dump(lap_time_list, file)