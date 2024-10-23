import numpy as np
from numba import njit
import yaml
import os
import os.path as osp

# @njit(cache=True)
def convert_x_into_dict(x):
    start = dict()
    start['x'] = x[0]
    start['y'] = x[1]
    start['steer_angle'] = x[2]
    start['velocity'] = x[3]
    start['theta'] = x[4]
    start['angular_velocity'] = x[5]
    start['slip_angle'] = x[6]
    return start


# @njit(cache=True)
def vehicle_dynamics_st_update_linear(state, accel, steer_angle_vel, p, dt):
    """
    Single Track Dynamic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
                x6: yaw rate
                x7: slip angle at vehicle center
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    start = convert_x_into_dict(state)

    # Local params
    v_b = 3  # m/s
    v_s = 1  # m/s
    v_min = v_b - 2 * v_s  # m/s
    g = 9.81  # m/s^2

    # Lateral tire slip angles
    start_vx = start['velocity'] * np.cos(start['slip_angle'])
    start_vy = start['velocity'] * np.sin(start['slip_angle'])
    if start['velocity'] >= v_min:
        alpha_f = np.arctan2((-start_vy - p['l_f'] * start['angular_velocity']), start_vx) + start['steer_angle']
        alpha_r = np.arctan2((-start_vy + p['l_r'] * start['angular_velocity']), start_vx)
    else:
        alpha_f = 0
        alpha_r = 0

    # Compute vertical tire forces (load transfer due to acceleration)
    F_zf = p['mass'] * (-accel * p['h_cg'] + g * p['l_r']) / (p['l_f'] + p['l_r'])
    F_zr = p['mass'] * (accel * p['h_cg'] + g * p['l_f']) / (p['l_f'] + p['l_r'])

    # Linear tire model
    F_yf = p['mu'] * F_zf * p['C_Sf'] * alpha_f
    F_yr = p['mu'] * F_zr * p['C_Sr'] * alpha_r

    # Compute first derivatives of state
    x_dot = start_vx * np.cos(start['theta']) - start_vy * np.sin(start['theta'])
    y_dot = start_vx * np.sin(start['theta']) + start_vy * np.cos(start['theta'])
    vx_dot = accel + (1 / p['mass']) * (-F_yf * np.sin(start['steer_angle'])) + start_vy * start['angular_velocity']
    steer_angle_dot = steer_angle_vel
    theta_dot = start['angular_velocity']
    vy_dot = (1 / p['mass']) * (F_yr + F_yf * np.cos(start['steer_angle'])) - start_vx * start['angular_velocity']
    theta_ddot = (1 / p['I_z']) * (-F_yr * p['l_r'] + F_yf * p['l_f'] * np.cos(start['steer_angle']))

    end_vx = start_vx + vx_dot * dt
    end_vy = start_vy + vy_dot * dt

    # Update state
    end = dict()
    end['x'] = start['x'] + x_dot * dt
    end['y'] = start['y'] + y_dot * dt
    end['theta'] = start['theta'] + theta_dot * dt
    end['velocity'] = np.sqrt(end_vx ** 2 + end_vy ** 2)
    end['steer_angle'] = start['steer_angle'] + steer_angle_dot * dt
    end['angular_velocity'] = start['angular_velocity'] + theta_ddot * dt
    end['slip_angle'] = np.arctan2(end_vy, end_vx)

    # Mix with kinematic model at low speeds (assuming update_k is also njit-compiled)
    kin_end = vehicle_dynamics_st_update_k(start, accel, steer_angle_vel, p, dt)

    # Weights for mixing
    w_std = 0.5 * (1 + np.tanh((start['velocity'] - v_b) / v_s))
    w_kin = 1 - w_std
    if start['velocity'] < v_min:
        w_std = 0
        w_kin = 1

    # Mix states
    for key in start.keys():
        end[key] = w_std * end[key] + w_kin * kin_end[key]

    end = np.array([end['x'], end['y'], end['steer_angle'], end['velocity'], end['theta'],end['angular_velocity'], end['slip_angle']])

    return end

# @njit(cache=True)
def vehicle_dynamics_st_update_pacejka(state, accel, steer_angle_vel, p, dt):
    start = convert_x_into_dict(state)
    
    # Local params
    v_b = 3  # m/s
    v_s = 1  # m/s
    v_min = v_b - 2 * v_s  # m/s
    g = 9.81  # m/s^2

    # Lateral tire slip angles
    start_vx = start['velocity'] * np.cos(start['slip_angle'])
    start_vy = start['velocity'] * np.sin(start['slip_angle'])
    if start['velocity'] >= v_min:
        alpha_f = np.arctan2((-start_vy - p['l_f'] * start['angular_velocity']), start_vx) + start['steer_angle']
        alpha_r = np.arctan2((-start_vy + p['l_r'] * start['angular_velocity']), start_vx)
    else:
        alpha_f = 0
        alpha_r = 0

    # Compute vertical tire forces (load transfer due to acceleration)
    F_zf = p['mass'] * (-accel * p['h_cg'] + g * p['l_r']) / (p['l_f'] + p['l_r'])
    F_zr = p['mass'] * (accel * p['h_cg'] + g * p['l_f']) / (p['l_f'] + p['l_r'])

    # Combined lateral slip forces according to Pacejka
    F_yf = p['mu'] * p['D_f'] * F_zf * np.sin(
        p['C_f'] * np.arctan(
            p['B_f'] * alpha_f - p['E_f'] * (p['B_f'] * alpha_f - np.arctan(
                p['B_f'] * alpha_f))))
    F_yr = p['mu'] * p['D_r'] * F_zr * np.sin(
        p['C_r'] * np.arctan(
            p['B_r'] * alpha_r - p['E_r'] * (p['B_r'] * alpha_r - np.arctan(
                p['B_r'] * alpha_r))))

    # now all-wheel drive model is based on https://arxiv.org/pdf/2404.08362
    F_m = accel * p['mass']
    rear_tire_split = 1
    assert 0 <= rear_tire_split <= 1, "Rear tire split must be between 0 and 1"
    F_xr = rear_tire_split * F_m
    F_xf = (1 - rear_tire_split) * F_m
    
    # Compute first derivatives of state
    x_dot = start_vx * np.cos(start['theta']) - start_vy * np.sin(start['theta'])
    y_dot = start_vx * np.sin(start['theta']) + start_vy * np.cos(start['theta'])
    vx_dot = (1 / p['mass']) * (
        F_xr
        + F_xf * np.cos(start['steer_angle']) 
        - F_yf * np.sin(start['steer_angle'])
        + start_vy * start['angular_velocity'] * p['mass']
    )
    vy_dot = (1 / p['mass']) * (
        F_yr 
        + F_xf * np.sin(start['steer_angle'])
        + F_yf * np.cos(start['steer_angle']) 
        - start_vx * start['angular_velocity'] * p['mass']
    )
    steer_angle_dot = steer_angle_vel
    theta_dot = start['angular_velocity']
    theta_ddot = (1 / p['I_z']) * (
        F_yf * p['l_f'] * np.cos(start['steer_angle'])
        + F_xf * p['l_f'] * np.sin(start['steer_angle'])
        - F_yr * p['l_r']
    )

    end_vx = start_vx + vx_dot * dt
    end_vy = start_vy + vy_dot * dt

    # Update state
    end = dict()
    end['x'] = start['x'] + x_dot * dt
    end['y'] = start['y'] + y_dot * dt
    end['theta'] = start['theta'] + theta_dot * dt
    end['velocity'] = np.sqrt(end_vx ** 2 + end_vy ** 2)
    end['steer_angle'] = start['steer_angle'] + steer_angle_dot * dt
    end['angular_velocity'] = start['angular_velocity'] + theta_ddot * dt
    end['slip_angle'] = np.arctan2(end_vy, end_vx)

    # Mix with kinematic model at low speeds (assuming update_k is also njit-compiled)
    kin_end = vehicle_dynamics_st_update_k(start, accel, steer_angle_vel, p, dt)

    # Weights for mixing
    w_std = 0.5 * (1 + np.tanh((start['velocity'] - v_b) / v_s))
    w_kin = 1 - w_std
    if start['velocity'] < v_min:
        w_std = 0
        w_kin = 1

    # Mix states
    for key in start.keys():
        end[key] = w_std * end[key] + w_kin * kin_end[key]

    end = np.array([end['x'], end['y'], end['steer_angle'], end['velocity'], end['theta'],end['angular_velocity'], end['slip_angle']])

    return end, np.array([vx_dot, vy_dot])

# @njit(cache=True)
def vehicle_dynamics_st_update_k(start, accel, steer_angle_vel, p, dt):
    # Compute first derivatives of state
    x_dot = start['velocity'] * np.cos(start['theta'] + start['slip_angle'])
    y_dot = start['velocity'] * np.sin(start['theta'] + start['slip_angle'])
    v_dot = accel
    steer_angle_dot = steer_angle_vel
    theta_dot = start['velocity'] * np.tan(start['steer_angle']) * np.cos(start['slip_angle']) / p['wheelbase']
    slip_angle_dot = (1 / (1 + np.power(((p['l_r'] / p['wheelbase']) * np.tan(start['steer_angle'])), 2))) * \
                     (p['l_r'] / (p['wheelbase'] * np.power(np.cos(start['steer_angle']), 2))) * steer_angle_vel
    theta_double_dot = accel * np.tan(start['steer_angle']) * np.cos(start['slip_angle']) / p['wheelbase'] + \
                       start['velocity'] * steer_angle_vel * np.cos(start['slip_angle']) / \
                       (p['wheelbase'] * np.power(np.cos(start['steer_angle']), 2)) - \
                       start['velocity'] * np.sin(start['slip_angle']) * np.tan(start['steer_angle']) * \
                       slip_angle_dot / p['wheelbase']

    # Update state
    end = dict()
    end['x'] = start['x'] + x_dot * dt
    end['y'] = start['y'] + y_dot * dt
    end['steer_angle'] = start['steer_angle'] + steer_angle_dot * dt
    end['velocity'] = start['velocity'] + v_dot * dt
    end['theta'] = start['theta'] + theta_dot * dt
    end['angular_velocity'] = start['angular_velocity'] + theta_double_dot * dt
    end['slip_angle'] = start['slip_angle'] + slip_angle_dot * dt

    return end