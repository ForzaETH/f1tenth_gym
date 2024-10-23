import matplotlib.pyplot as plt
import os
import os.path as osp
import yaml

def load_config(name):
    with open(
        f"{os.path.normpath(osp.abspath(__file__) + os.sep + os.pardir)}/{name}.yaml",
        "r",
    ) as file:
        config = yaml.safe_load(file)
    return config

def prettify_label(label):
    """Convert snake_case labels to Title Case for better presentation."""
    return label.replace('_', ' ').title()

def base_plot(track_bound_coords, global_waypoints_coords, plot_config):
    """Creates the base plot layout for track boundary and reference trajectory."""
    fig, ax = plt.subplots(figsize=plot_config['figsize'])
    ax.scatter(track_bound_coords[:, 0], track_bound_coords[:, 1],
               s=plot_config['track_bound_size'],
               c=plot_config['track_bound_color'],
               label='Track Boundary',
               alpha=plot_config['track_bound_alpha'])
    ax.plot(global_waypoints_coords[:, 0], global_waypoints_coords[:, 1],
            linestyle=plot_config['reference_traj_linestyle'],
            color=plot_config['reference_traj_color'],
            linewidth=plot_config['reference_traj_linewidth'],
            alpha=plot_config['reference_traj_alpha'],
            label='Reference Trajectory')
    ax.set_aspect('equal')
    ax.set_xlabel('X Position', fontsize=plot_config['axis_fontsize'], weight=plot_config['axis_fontweight'])
    ax.set_ylabel('Y Position', fontsize=plot_config['axis_fontsize'], weight=plot_config['axis_fontweight'])
    return fig, ax
