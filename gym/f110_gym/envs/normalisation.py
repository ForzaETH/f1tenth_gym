"""
Normalization module for the RL environment
"""
import numpy as np

def denorm_action(action, params):
        """
        As the action is by default clipped within -1, 1, we need to rescale it to the proper size
        """

        v_min = params['v_min']
        v_max = params['v_max']

        if action.shape == (2,):
            action[0] = ((action[0] + 1)/2)*(params['s_max'] - params['s_min']) + params['s_min']
            action[1] = ((action[1] + 1)/2)*(v_max - v_min) + v_min
        elif action.shape == (1, 2):
            action[0][0] = ((action[0][0] + 1)/2)*(params['s_max'] - params['s_min']) + params['s_min']
            action[0][1] = ((action[0][1] + 1)/2)*(v_max - v_min) + v_min
        else:
            action[0][0] = ((action[0][0] + 1)/2)*(params['s_max'] - params['s_min']) + params['s_min']
            action[1][0] = ((action[1][0] + 1)/2)*(v_max - v_min) + v_min

        return action

def denorm_action_direct(action, params):
        """
        As the action is by default clipped within -1, 1, we need to rescale it to the proper size
        Here we are using the direct acceleration and steering velocity though
        """

        a_max = params['a_max']
        sv_max = params['sv_max']

        if action.shape == (2,):
            action[0] = ((action[0] + 1)/2)*(sv_max - (-sv_max)) + (-sv_max)
            action[1] = ((action[1] + 1)/2)*(a_max - (-a_max)) + (-a_max)
        elif action.shape == (1, 2):
            action[0][0] = ((action[0][0] + 1)/2)*(sv_max - (-sv_max)) + (-sv_max)
            action[0][1] = ((action[0][1] + 1)/2)*(a_max - (-a_max)) + (-a_max)
        else:
            action[0][0] = ((action[0][0] + 1)/2)*(sv_max - (-sv_max)) + (-sv_max)
            action[1][0] = ((action[1][0] + 1)/2)*(a_max - (-a_max)) + (-a_max)

        return action

def normalise_observation(obs, params, with_lidar=True, use_previous_actions=True):
    """
    Normalises the base observation.

    Args:
        obs: the base (Frenet) observation. It is a dictionary with the following keys: 
            scans: a list/np.array of lidar scans
            deviation: a scalr distance from a reference trajectory
            rel_heading: the scalar relative yaw to th e reference trajectory
            longitudinal_vel: the scalar longitudinal velocity of the car
            lateral_vel: the scalar lateral velocity of the car

        params: a dictionary containing the upper and lower limits for the different states of the observation. 
            Currently only contains: 
            v_min: minimum velocity
            v_max: maximum velocity
            width: width of the car
    """
    
    minimums = {
        'deviation': 0,
        'rel_heading': -np.pi,
        'longitudinal_vel': params['v_min'],
        'later_vel': params['v_min'],
        'yaw_rate': -params['sv_max'],
        'yaw': -params['s_max']
    }
    maximums = {
        'deviation': 10*params['width'],
        'rel_heading': np.pi,
        'longitudinal_vel': params['v_max'],
        'later_vel': params['v_max'],
        'yaw_rate': params['sv_max'],
        'yaw': params['s_max']
    }


    if with_lidar:
        minimums['scans'] = 0
        maximums['scans'] = 10 # TODO remove hardcoding

    if use_previous_actions:
        minimums['previous_s'] = -params['s_max']
        maximums['previous_s'] = params['s_max']
        minimums['previous_v'] = params['v_min']
        maximums['previous_v'] = params['v_max']

    for k in obs.keys():
        obs[k] = np.clip(obs[k], minimums[k], maximums[k])
        obs[k] -= (maximums[k]-minimums[k])/2
        obs[k] *= 2/(maximums[k] - minimums[k])

    return obs

def normalise_trajectory(traj, traj_len, traj_points_spacing):
    """
    Normalises the trajectory according to the trajectory length. 
    """
    max_traj = traj_len * traj_points_spacing
    min_traj = 0
    trajectory = np.clip(traj, min_traj, max_traj)
    trajectory -= (max_traj-min_traj)/2
    trajectory *= 2/(max_traj - min_traj)

    return trajectory
