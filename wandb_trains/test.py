import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import logging

from fire import Fire
from gym_data_recorder import recording, initialize_recorder
from report_generator import ReportGenerator
from f110_gym.envs.utils import read_config, render_callback, Traj
from gymnasium.wrappers.frame_stack import FrameStack

from report_generator_extra import analyze_data as extra_analyze_data

conf_dict = read_config("wandb_config.yaml")
nn_dict = read_config("nn_config.yaml")


# DEFAULTS
debug_mode = False
TEST_TYPE = 0


if TEST_TYPE == 1:
    conf_dict["use_lidar"] = True
if TEST_TYPE == 2:
    conf_dict["calc_glob_traj"] = True
if TEST_TYPE == 3:
    conf_dict["calc_glob_traj"] = False

conf_dict["delay_buffer_size"] = 2
conf_dict["timestep"] = 0.05
conf_dict["max_lap_num"] = 10 + 1
conf_dict["rndmize_at_reset"] = False
conf_dict["rnd_stds"]["mu_std"] = 0.0

def test(model_path: str = None,
         record: bool = True,
         evaluate_record: bool = True):
    # logging
    if debug_mode:
        logging.basicConfig(level=logging.DEBUG)

    env = gym.make("f110_gym:f110-v0", render_mode="human_fast", **conf_dict)

    if conf_dict["enable_frame_stack"]:
        env = FrameStack(env, conf_dict["frame_stack_num_stack"])

    # Add rendering callbacks
    env.unwrapped.add_render_callback(render_callback)

    obs, info = env.reset()

    num_render_points = conf_dict["n_traj_points"]

    if TEST_TYPE == 1:
        env.unwrapped.render()
        lidar_points = Traj(N=10, batch=env.unwrapped.renderer.batch,  clr=(255, 255, 0))
    elif TEST_TYPE == 2:
        env.unwrapped.render()
        trajectory_points = Traj(
            N=num_render_points, batch=env.unwrapped.renderer.batch, clr=(0, 255, 0)
        )
        trajectory_int_points = Traj(
            N=num_render_points, batch=env.unwrapped.renderer.batch, clr=(255, 255, 0)
        )
        trajectory_out_points = Traj(
            N=num_render_points, batch=env.unwrapped.renderer.batch, clr=(255, 255, 0)
        )
    elif TEST_TYPE == 3:
        env.unwrapped.render()
        relative_trajectory_points = Traj(
            N=num_render_points, batch=env.unwrapped.renderer.batch
        )
        relative_trajectory_int_points = Traj(
            N=num_render_points, batch=env.unwrapped.renderer.batch
        )
        relative_trajectory_out_points = Traj(
            N=num_render_points, batch=env.unwrapped.renderer.batch
        )

    if conf_dict["enable_steering_delay"] and conf_dict["enable_prediction_reward"]:
        env.unwrapped.render()
        predicted_state = Traj(
            N=1, batch=env.unwrapped.renderer.batch, clr=(255, 0, 0), r=5
        )

    prev_dev = 0

    print("Loading model, PLEASE CHECK wanted model is used:")
    if model_path is None:
        model_folder = os.path.join(os.path.dirname(__file__), "pre_trained_models")
        model_path = os.path.join(model_folder, "model")
        print(f"Defaul model used: {model_path}")
    else:
        model_folder = os.path.join(os.path.dirname(__file__), model_path) 
        model_path = os.path.join(model_folder, "model")
        print(f"Model used: {model_path}")    
    model = SAC.load(
        model_path,
        device="cuda",
        output_reg_matrix=np.array(
            [[nn_dict["output_reg_steer"], 0], [0, nn_dict["output_reg_acc"]]]
        ),
    )

    enable_recorder = True
    record_started = False
    ignore_action = False
    
    time_step = 0
    current_lap_num = 1
    lap_start_time = 0
    terminated = False
    lap_time_list = []

    action = np.array([0, 0])
    while not terminated:
        if enable_recorder and env.unwrapped.lap_counts[0] != 0:
            if env.unwrapped.renderer is not None and not record_started:
                save_path = f"{model_folder}/eval_sim"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                recorder = initialize_recorder(env, save_path=save_path)
                lap_start_time = env.unwrapped.lap_times[0]
                record_started = True

        action, *_ = model.predict(obs)
        action_factor = np.array([1, 1])
        if ignore_action:
            action = np.array([0, 0])

        if record_started:
            recording(recorder, env, time_step, action)

        obs, reward, terminated, truncated, info = env.step(action * action_factor)

        env.unwrapped.render()

        time_step += conf_dict["timestep"] * 100

        if conf_dict["enable_steering_delay"] and conf_dict["enable_prediction_reward"]:
            predicted_state.set_points(env.predicted_poses)

        if TEST_TYPE == 1:
            lidar_points.set_points(env.unwrapped.observator.get_lidar_coords())
        elif TEST_TYPE == 2:
            trajectory_points.set_points(
                env.unwrapped.observator.denorm_obs["glob_traj"]
            )
            trajectory_int_points.set_points(
                env.unwrapped.observator.denorm_obs["glob_track_int"]
            )
            trajectory_out_points.set_points(
                env.unwrapped.observator.denorm_obs["glob_track_out"]
            )
        elif TEST_TYPE == 3:
            relative_trajectory_points.set_points(
                env.unwrapped.observator.denorm_obs["traj"]
            )
            relative_trajectory_int_points.set_points(
                env.unwrapped.observator.denorm_obs["track_int"]
            )
            relative_trajectory_out_points.set_points(
                env.unwrapped.observator.denorm_obs["track_out"]
            )

        if env.unwrapped.lap_counts[0] != current_lap_num and record_started:
            print(f"Lap {int(current_lap_num)} finished with {(env.unwrapped.lap_times[0] - lap_start_time):.4f}s.")
            lap_time_list.append(round(env.unwrapped.lap_times[0] - lap_start_time, 4))
            current_lap_num = env.unwrapped.lap_counts[0]
            lap_start_time = env.unwrapped.lap_times[0]

    print(f"lap_time: {lap_time_list}")

    if record_started:
        if record:
            recorder.save_lap_time_list(lap_time_list)
            recorder.finalize_results()
        if evaluate_record:
            rec_config_path = os.path.join(model_folder, "eval_sim/recorder_config.yaml")
            report_generator = ReportGenerator(read_config(rec_config_path), read_config("plot_config.yaml"), save_dir=save_path)
            report_generator.run()
            
            extra_analyze_data(model_folder, verbose=True)


if __name__ == "__main__":
    # fire makes it easy to run from command line
    Fire(test)
    