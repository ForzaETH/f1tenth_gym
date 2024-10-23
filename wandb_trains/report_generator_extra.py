import os

import yaml
import torch
import gpytorch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
def analyze_data(file_path: str, 
                 verbose: bool = False):
    # load yaml file
    with open(f"{file_path}/eval_sim/recorder_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    time_stamp = str(config["last_time_stamp"])

    # setup folders
    data_folder = f"{file_path}/eval_sim/F110_recordings/{time_stamp}"
    data_path = f"{data_folder}/car_raw_info.csv"
    traj_path = f"{data_folder}/traj.npy"
    save_path = f"{file_path}/eval_sim/eval_plots/"

    if not os.path.exists(save_path):
        print(f"Creating folder for saving data here: {save_path}") if verbose else None
        os.makedirs(save_path)
        
    data = pd.read_csv(data_path, header=0, sep="\t")


    # read raceline data
    raceline = np.load(traj_path)
    print(f"{raceline.shape = }") if verbose else None

    raceline_length = np.linalg.norm(raceline[1:, :] - raceline[:-1, :], axis=1).sum()
    print(f"{raceline_length = :.3f}") if verbose else None
    
    # some preprocessing

    ## create column with modulo on frenet s 
    data["frenet_s_mod"] = data["frenet_s"].copy()
    data["frenet_s_mod"] %= raceline_length

    # fit a GP to the data, to obtain a smoother estimate of the RL speed input
    # first prepare the data, by padding the velocity profile with repeated values to make it periodic
    velocities = data["speed_input"].values
    frenet_positions = data["frenet_s_mod"].values

    # repeat all data points where freent is higher than half the track length
    mask = frenet_positions > raceline_length / 2
    frenet_positions_before = frenet_positions[mask] - raceline_length
    frenet_positions = np.concatenate([frenet_positions, frenet_positions_before])
    velocities = np.concatenate([velocities, velocities[mask]])
    # same but after 
    mask = frenet_positions < raceline_length / 2
    frenet_positions_after = frenet_positions[mask] + raceline_length
    frenet_positions = np.concatenate([frenet_positions, frenet_positions_after])
    velocities = np.concatenate([velocities, velocities[mask]])

    # initialize likelihood and model
    print("Initializing GP model") if verbose else None
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(
        torch.from_numpy(frenet_positions).float(),
        torch.from_numpy(velocities).float(),
        likelihood
    )

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iter = 5
    for i in tqdm(range(training_iter)):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(torch.from_numpy(frenet_positions).float())
        # Calc loss and backprop gradients
        loss = -mll(output, torch.from_numpy(velocities).float())
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item())) if verbose else None
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0, raceline_length]
    test_x = torch.linspace(0, raceline_length, 1000)
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x.float()))
        dataset_pred = likelihood(model(torch.from_numpy(data["frenet_s_mod"].to_numpy().astype(np.float32))))
        lower, upper = observed_pred.confidence_region()    
    
    print("GP model trained") if verbose else None

    # plot the results only in range [0, raceline_length]
    plotting_mask_pred = (test_x >= 0) & (test_x <= raceline_length)

    # assess filtered data with a simple two-line-implementation exponential filter

    def filter(alpha):
        """Filter the speed_input with an exponential filter"""
        data["speed_input_filtered"] = data["speed_input"].copy()
        filter = lambda x, y: alpha * x + (1 - alpha) * y
        for i in range(1, len(data)):
            data.at[i, "speed_input_filtered"] = filter(data.at[i, "speed_input"], data.at[i-1, "speed_input_filtered"])
            
        # create copy of sorted data to plot
        data_sorted = data.sort_values("frenet_s_mod")
        sorted_s = data_sorted["frenet_s_mod"].values
        sorted_filtered = data_sorted["speed_input_filtered"].values
        
        return sorted_s.copy(), sorted_filtered.copy()
    
    alphas = np.linspace(0.3, 0.6, 7)
    
    ## FIRST PLOT: assessment on the filter's delay
    fig, axs = plt.subplots(len(alphas), 1, figsize=(20, 40))
    for i, alpha in enumerate(alphas):
        sorted_s, sorted_filtered = filter(alpha)
        axs[i].plot(sorted_s, sorted_filtered, label=f"{alpha = }", alpha=0.5)
        axs[i].plot(test_x[plotting_mask_pred].numpy(), observed_pred.mean[plotting_mask_pred].numpy(), 'b')
        axs[i].fill_between(test_x[plotting_mask_pred].numpy(), lower[plotting_mask_pred].numpy(), upper[plotting_mask_pred].numpy(), alpha=0.5)
        axs[i].set_ylim([0, 10])
        axs[i].legend(['Causally Filtered Input Data', 'GP Regressed Mean', 'GP Regressed Variance'])
        axs[i].grid(True)
        axs[i].set_title(f"{alpha = :.2f}")
        axs[i].set_xlabel(r"Frenet $s$ coordinate [m]")
        axs[i].set_ylabel(r"Speed [m/s]")
    
    # save the figure
    fig.savefig(f"{save_path}/filter_assessment.svg", bbox_inches='tight', pad_inches=0.1, dpi=300, transparent=False, facecolor='white')
    print("Filter assessment figure saved") if verbose else None
    
    #plot velocities against s_mod
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    # plot gray with alpha 0.3
    ax.plot(data["frenet_s_mod"], data["speed"], 'k.', alpha=0.3, label="speed measured")

    # plot pp speed input too
    ax.plot(data["frenet_s_mod"], data["speed_input_pp"], 'r.', alpha=1, label="speed input pp", zorder=-10)

    # plot GP regressed speed input
    ax.plot(data["frenet_s_mod"], dataset_pred.mean.numpy() + data["speed_input_pp"], 'b.', label="speed input GP")

    # plot filtered speed input
    sorted_s, sorted_filtered = filter(0.5)
    sorted_pp = data.sort_values("frenet_s_mod")["speed_input_pp"].values
    ax.plot(sorted_s, sorted_filtered + sorted_pp, 'purple', alpha=0.3, label="speed input filtered")
    # use diamond markers 
        
        
    # plot total 
    ax.plot(data["frenet_s_mod"], data["speed_input"] + data["speed_input_pp"], 'g.', alpha=0.3, label="speed total")

    # add legend outisde of plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # label x axis
    ax.set_xlabel(r"Frenet $s$ coordinate [m]")

    # add grid
    ax.grid(True)

    # save with background
    fig.savefig(f"{save_path}/speed_against_s_mod.svg", bbox_inches='tight', pad_inches=0.1, dpi=300, transparent=False, facecolor='white')
    print("Speed against frenet s coordinate figure saved") if verbose else None
    
    # plot similar modulo statistics but for steer
    fig, ax = plt.subplots(figsize=(20, 10), facecolor='white')
    # plot gray with alpha 0.3
    ax.plot(data["frenet_s_mod"], data["yaw_angle"], 'k.', alpha=0.3, label="steer measured")

    # plot pp speed input too
    ax.plot(data["frenet_s_mod"], data["steering_input_pp"], 'r.', alpha=1, label="steer input pp", zorder=-10)

    # plot totalsteer_input in blue
    ax.plot(data["frenet_s_mod"], data["steering_input"]+data["steering_input_pp"], 'b.', alpha=0.3, label="steer input rl")

    ax.grid(True)
    # legend outside
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # label x axis
    ax.set_xlabel(r"Frenet $s$ coordinate [m]")
    
    # label y axis
    ax.set_ylabel(r"Steer input [rad]")
    
    # plot total
    fig.savefig(f"{save_path}/steer_input.svg", dpi=300)
    print("Steer input figure saved") if verbose else None
    
    # read lap times

    with open(f"{data_folder}/lap_time_list.txt", 'r') as f:
        lap_times = f.readline()
    lap_times = lap_times.strip("[]").split(",")
    lap_times = [float(lap_time) for lap_time in lap_times]

    print("Saving lap statistics") if verbose else None
    with open(f"{save_path}/report.txt", "w") as file:
        file.write(f"Report for the model at {data_path}, with timestamp {time_stamp}\n")
        file.write(f"avg lap time = {np.mean(lap_times):.3f} s\n")
        file.write(f"Average lateral deviation: {data['frenet_d'].abs().mean():.3f} m\n")
        file.write(f"Max velocity: {data['speed'].max():.3f} m/s\n")
    
    print("Data analyzed and saved to disk")