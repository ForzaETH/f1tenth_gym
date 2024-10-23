import os
import os.path as osp
import glob
import gymnasium as gym
import time
import wandb
import logging
import numpy as np
from datetime import datetime
import itertools
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.frame_stack import FrameStack
from f110_gym.envs.utils import render_callback, read_config, get_package_location
from gym_data_recorder import recording, initialize_recorder
from report_generator import ReportGenerator


def log_plots(recorder):

    recorder.finalize_results()
    report_generator = ReportGenerator(read_config("recorder_config.yaml"), read_config("plot_config.yaml"))
    report_generator.run()

    directory = recorder.save_dir
    files = glob.glob(osp.join(directory, "**"), recursive=True)

    for file in files:
        if osp.isfile(file):  # Check if it's a file and not a directory
            wandb.save(file, base_path=directory)
            
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0, num_env=1):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.n_envs = num_env
        
    def _on_step(self) -> bool:
        info = self.locals.get("infos")
        if info is None:
            return True
        for i in range(self.n_envs):
            wandb.log({f"agent_{i}":{
                "reward": info[i]["reward_info"],
                "average_speed": info[i]["average_speed"],
            }})
        return True

def training_loop(conf_dict, nn_dict):
    logging.basicConfig(level=logging.WARN)

    if conf_dict["use_pp_action"]:
        controller_mode = "RL_PP"
    elif conf_dict["use_base_controller"]:
        controller_mode = "RL_PP_obs"
    else:
        controller_mode = "RL_only"

    if conf_dict["penalty_based_on_reward"]:
        penalty_mode = "_penalty_based_on_reward"
    else:
        penalty_mode = ""

    time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    run = wandb.init(
        name=f"{controller_mode}_outreg_{time_string}",
        project=f"V15.0_{controller_mode}_{1/conf_dict['timestep']}hz{penalty_mode}_{conf_dict['map_name']}",  # Wandb project name
        entity="f1tenth_pbl_eth",  # Wandb team name
        config={"conf": conf_dict, "nn_conf": nn_dict},
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,
        mode="run",  # can be `disabled`, `offline`, `run`
    )

    wandb.run.log_code(".")

    env_dir = osp.abspath(
        get_package_location("f110_gym")
        + os.sep
        + "f110_gym"
        + os.sep
        + "envs"
        + os.sep
    )
    # f110_file_to_save_list = [
    #     "rewards.py",
    #     "SIM_car_params.yaml",
    #     "SIM_pacejka.yaml",
    # ]
    # for file in f110_file_to_save_list:
    #     original_path = osp.join(env_dir, file)
    #     symlink_path = osp.join(wandb.run.dir, file)
    #     os.symlink(original_path, symlink_path)
    #     wandb.save(symlink_path)

    # wandb.save("wandb_config.yaml")
    wandb.save("nn_config.yaml")
    wandb.save("search_space.yaml")

    if not conf_dict["vec_env"]:
        env = gym.make("f110_gym:f110-v0", render_mode="rgb_array", **conf_dict)
        if conf_dict["enable_frame_stack"]:
            env = FrameStack(env, conf_dict["frame_stack_num_stack"])
        env.unwrapped.add_render_callback(render_callback)
        env = Monitor(env)
        env = RecordVideo(
            env,
            video_folder=f"./videos/{time_string}_{run.id}/training/",
            name_prefix="rl-video-training",
            disable_logger=True,
            step_trigger= lambda x: x % int(nn_dict["training_timesteps"] / 10) == 0,
        )
    else:
        
        N_ENV = conf_dict["n_envs"]
        env = make_vec_env(env_id="f110_gym:f110-v0", n_envs=N_ENV, env_kwargs=conf_dict)
        if conf_dict["enable_frame_stack"]:
            env = VecFrameStack(env, conf_dict["frame_stack_num_stack"])
        env = VecVideoRecorder(
            env,
            f"videos/{run.id}/training/",
            record_video_trigger=lambda x: x % int(nn_dict["training_timesteps"] / 10) == 0,
            video_length=200,
            name_prefix="rl_video_training",
        )

    if conf_dict["algorithm"] == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            device="cuda",
            tensorboard_log=f"runs/{run.name}",
            verbose=1,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            device="cuda",
            output_reg_matrix=np.array(
                [[nn_dict["output_reg_steer"], 0], [0, nn_dict["output_reg_acc"]]]
            ),
            tensorboard_log=f"runs/{run.name}",
            verbose=1,
        )

    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_freq=int(nn_dict["training_timesteps"] / 10),
        model_save_path="./wandb_model/",
    )

    model.learn(
        total_timesteps=nn_dict["training_timesteps"],
        progress_bar=True,
        callback=[wandb_callback, RewardLoggerCallback()],
    )

    env.close()

    conf_dict["enable_random_start_speed"] = False
    conf_dict["rndmize_at_reset"] = False
    conf_dict["random_direction"] = False
    
    if not conf_dict["vec_env"]:
        test_env = gym.make("f110_gym:f110-v0", render_mode="rgb_array", **conf_dict)
        if conf_dict["enable_frame_stack"]:
            test_env = FrameStack(test_env, conf_dict["frame_stack_num_stack"])
        test_env = RecordVideo(
            test_env,
            video_folder=f"./videos/{run.id}/final_result/",
            name_prefix="rl-video-final",
            disable_logger=True,
        )
    else:
        test_env = make_vec_env(env_id="f110_gym:f110-v0", n_envs=N_ENV, env_kwargs=conf_dict)
        if conf_dict["enable_frame_stack"]:
            test_env = VecFrameStack(test_env, conf_dict["frame_stack_num_stack"])
        test_env = VecVideoRecorder(
            test_env,
            f"videos/{run.id}/final_result/",
            video_length=1200,
            name_prefix="rl_video_final",
        )

    model.set_env(test_env)

    terminated = False
    if conf_dict["vec_env"]:
        obs = test_env.reset()
    else:
        obs, _ = test_env.reset()
    test_env.start_video_recorder()

    time_step = 0
    if not conf_dict["vec_env"]:
        recorder = initialize_recorder(test_env)

    while not terminated:
        action, *_ = model.predict(obs)
        if not conf_dict["vec_env"]:
            recording(recorder, test_env, time_step, action)
            obs, reward, terminated, truncated, info = test_env.step(action)
        else:
            obs, rewards, dones, info = test_env.step(action)
        time_step += conf_dict["timestep"] * 100

    if not conf_dict["vec_env"]:
        log_plots(recorder)

    test_env.close()

    video_dir = f"./videos/{run.id}/"
    video_files = glob.glob(osp.join(video_dir, "**/*.mp4"), recursive=True)
    print(f'The following videos are saved: {", ".join(video_files)}')
    for file in video_files:
        filename = osp.basename(file)
        wandb.run.log({filename: wandb.Video(file, format="mp4")})

    run.finish()


if __name__ == "__main__":
    search_space = read_config(
        osp.join(osp.dirname(osp.abspath(__file__)), "search_space.yaml")
    )
    conf_dict = read_config(
        osp.join(osp.dirname(osp.abspath(__file__)), "wandb_config.yaml")
    )
    nn_dict = read_config(
        osp.join(osp.dirname(osp.abspath(__file__)), "nn_config.yaml")
    )

    search_settings = search_space['search_space']

    parameters = {}
    for param, settings in search_settings.items():
        if settings['enabled']:
            parameters[param] = settings['values']
        else:
            parameters[param] = [conf_dict.get(param, None)]

    keys, values = zip(*parameters.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"{len(combinations)} combinations will be tested")

    for combo in combinations:
        updated_conf_dict = {**conf_dict, **combo}
        print(f"Running training loop with configuration: {combo}")
        training_loop(updated_conf_dict, nn_dict)