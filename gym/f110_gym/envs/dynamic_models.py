# Copyright 2020 Technical University of Munich, Professorship of Cyber-Physical Systems, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



"""
Prototype of vehicle dynamics functions and classes for simulating 2D Single
Track dynamic model
Following the implementation of commanroad's Single Track Dynamics model
Original implementation: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/
Author: Hongrui Zheng
"""

import numpy as np
from numba import njit

import unittest
import time

@njit(cache=True)
def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    """
    Acceleration constraints, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            accl (float): unconstraint desired acceleration
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max (float): maximum allowed acceleration
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            accl (float): adjusted acceleration
    """

    # positive accl limit
    if vel > v_switch:
        pos_limit = a_max*v_switch/vel
    else:
        pos_limit = a_max

    # accl limit reached?
    if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
        accl = 0.
    elif accl <= -a_max:
        accl = -a_max
    elif accl >= pos_limit:
        accl = pos_limit

    return accl

@njit(cache=True)
def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    """
    Steering constraints, adjusts the steering velocity based on constraints

        Args:
            steering_angle (float): current steering_angle of the vehicle
            steering_velocity (float): unconstraint desired steering_velocity
            s_min (float): minimum steering angle
            s_max (float): maximum steering angle
            sv_min (float): minimum steering velocity
            sv_max (float): maximum steering velocity

        Returns:
            steering_velocity (float): adjusted steering velocity
    """

    # constraint steering velocity
    if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
        steering_velocity = 0.
    elif steering_velocity <= sv_min:
        steering_velocity = sv_min
    elif steering_velocity >= sv_max:
        steering_velocity = sv_max

    return steering_velocity


@njit(cache=True)
def vehicle_dynamics_ks(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """
    Single Track Kinematic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # wheelbase
    lwb = lf + lr

    # constraints
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # system dynamics
    f = np.array([x[3]*np.cos(x[4]),
         x[3]*np.sin(x[4]),
         u[0],
         u[1],
         x[3]/lwb*np.tan(x[2])])
    return f

@njit(cache=True)
def vehicle_dynamics_st_pacejka(x, u_init, mu, Bf, Cf, Df, Ef, Br, Cr, Dr, Er, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """
    Single Track Dynamic Vehicle Dynamics with Pacejka magic formula tires and rear wheel drive.
    Below 0.5 m/s it switches to the kinematic model. Tire forces as the forzaeth_mpc single track model.

        Args:
            x (numpy.ndarray (7, )): vehicle state vector (x, y, steering angle, velocity, yaw angle, yaw rate, slip angle at vehicle center)
            u (numpy.ndarray (2, )): control input vector (steering angle velocity, longitudinal acceleration)
            Bf, Cf, Df, Ef, Br, Cr, Dr, Er: front and rear Pacejka coefficients, F_y = mu * D * F_z * sin(C * atan(B * alpha - E * (B * alpha - atan(B * alpha))))

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """

    # gravity constant m/s^2
    g = 9.81

    # constraints
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.5:
        # wheelbase
        lwb = lf + lr

        # system dynamics
        x_ks = x[0:5]
        f_ks = vehicle_dynamics_ks(x_ks, u, mu, 0., 0., lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
        return np.hstack((f_ks, np.array([u[1]/lwb*np.tan(x[2])+x[3]/(lwb*np.cos(x[2])**2)*u[0],
        0])))

    # body frame velocities
    v_x = x[3]*np.cos(x[6])
    v_y = x[3]*np.sin(x[6])

    # slip angles
    alpha_f = np.arctan2(-v_y - lf*x[5], v_x) + x[2]
    alpha_r = np.arctan2(-v_y + lr*x[5], v_x)

    # vertical tire forces with longitudinal load transfer
    F_zf = m*(-u[1]*h + g*lr)/(lf + lr)
    F_zr = m*(u[1]*h + g*lf)/(lf + lr)

    # lateral tire forces
    F_yf = mu*Df*F_zf*np.sin(Cf*np.arctan(Bf*alpha_f - Ef*(Bf*alpha_f - np.arctan(Bf*alpha_f))))
    F_yr = mu*Dr*F_zr*np.sin(Cr*np.arctan(Br*alpha_r - Er*(Br*alpha_r - np.arctan(Br*alpha_r))))

    # body frame accelerations, the longitudinal force is all on the rear axle
    v_x_dot = u[1] - F_yf*np.sin(x[2])/m + v_y*x[5]
    v_y_dot = (F_yr + F_yf*np.cos(x[2]))/m - v_x*x[5]
    yaw_rate_dot = (F_yf*lf*np.cos(x[2]) - F_yr*lr)/I

    # system dynamics
    f = np.array([x[3]*np.cos(x[6] + x[4]),
        x[3]*np.sin(x[6] + x[4]),
        u[0],
        (v_x*v_x_dot + v_y*v_y_dot)/x[3],
        x[5],
        yaw_rate_dot,
        (v_x*v_y_dot - v_y*v_x_dot)/x[3]**2])

    return f

@njit(cache=True)
def pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v):
    """
    Basic controller for speed/steer -> accl./steer vel.

        Args:
            speed (float): desired input speed
            steer (float): desired input steering angle

        Returns:
            accl (float): desired input acceleration
            sv (float): desired input steering velocity
    """
    # steering
    steer_diff = steer - current_steer
    if np.fabs(steer_diff) > 1e-4:
        sv = (steer_diff / np.fabs(steer_diff)) * max_sv
    else:
        sv = 0.0

    # accl
    vel_diff = speed - current_speed
    # currently forward
    if current_speed > 0.:
        if (vel_diff > 0):
            # accelerate
            kp = 10.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # braking
            kp = 10.0 * max_a / (-min_v)
            accl = kp * vel_diff
    # currently backwards
    else:
        if (vel_diff > 0):
            # braking
            kp = 2.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # accelerating
            kp = 2.0 * max_a / (-min_v)
            accl = kp * vel_diff

    return accl, sv

def func_KS(x, t, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    f = vehicle_dynamics_ks(x, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
    return f

class DynamicsTest(unittest.TestCase):
    def setUp(self):
        # test params
        self.mu = 1.0489
        self.C_Sf = 21.92/1.0489
        self.C_Sr = 21.92/1.0489
        self.lf = 0.3048*3.793293
        self.lr = 0.3048*4.667707
        self.h = 0.3048*2.01355
        self.m = 4.4482216152605/0.3048*74.91452
        self.I = 4.4482216152605*0.3048*1321.416

        #steering constraints
        self.s_min = -1.066  #minimum steering angle [rad]
        self.s_max = 1.066  #maximum steering angle [rad]
        self.sv_min = -0.4  #minimum steering velocity [rad/s]
        self.sv_max = 0.4  #maximum steering velocity [rad/s]

        #longitudinal constraints
        self.v_min = -13.6  #minimum velocity [m/s]
        self.v_max = 50.8  #minimum velocity [m/s]
        self.v_switch = 7.319  #switching velocity [m/s]
        self.a_max = 11.5  #maximum absolute acceleration [m/s^2]

    def test_derivatives(self):
        # ground truth derivatives
        f_ks_gt = [16.3475935934250209, 0.4819314886013121, 0.1500000000000000, 5.1464424102339752, 0.2401426578627629]

        # system dynamics
        g = 9.81
        x_ks = np.array([3.9579422297936526, 0.0391650102771405, 0.0378491427211811, 16.3546957860883566, 0.0294717351052816])
        v_delta = 0.15
        acc = 0.63*g
        u = np.array([v_delta,  acc])

        f_ks = vehicle_dynamics_ks(x_ks, u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)

        start = time.time()
        for i in range(10000):
            f_ks = vehicle_dynamics_ks(x_ks, u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)
        duration = time.time() - start
        avg_fps = 10000/duration

        self.assertAlmostEqual(np.max(np.abs(f_ks_gt-f_ks)), 0.)
        self.assertGreater(avg_fps, 5000)

    def test_zeroinit_roll(self):
        from scipy.integrate import odeint

        # testing for zero initial state, zero input singularities
        g = 9.81
        t_start = 0.
        t_final = 1.
        delta0 = 0.
        vel0 = 0.
        Psi0 = 0.
        sy0 = 0.
        initial_state = [0,sy0,delta0,vel0,Psi0]

        x0_KS = np.array(initial_state[0:5])

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set input: rolling car (velocity should stay constant)
        u = np.array([0., 0.])

        # simulate kinematic single-track model
        x_roll_ks = odeint(func_KS, x0_KS, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))

        self.assertTrue(all(x_roll_ks[-1]==x0_KS))

    def test_zeroinit_dec(self):
        from scipy.integrate import odeint

        # testing for zero initial state, decelerating input singularities
        g = 9.81
        t_start = 0.
        t_final = 1.
        delta0 = 0.
        vel0 = 0.
        Psi0 = 0.
        sy0 = 0.
        initial_state = [0,sy0,delta0,vel0,Psi0]

        x0_KS = np.array(initial_state[0:5])

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set decel input
        u = np.array([0., -0.7*g])

        # simulate kinematic single-track model
        x_dec_ks = odeint(func_KS, x0_KS, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))

        # ground truth for kinematic single-track model
        x_dec_ks_gt = [-3.4335000000000013, 0.0000000000000000, 0.0000000000000000, -6.8670000000000018, 0.0000000000000000]

        self.assertTrue(all(abs(x_dec_ks[-1] - x_dec_ks_gt) < 1e-2))

    def test_zeroinit_acc(self):
        from scipy.integrate import odeint

        # testing for zero initial state, accelerating with left steer input singularities
        # wheel spin and velocity should increase more wheel spin at rear
        g = 9.81
        t_start = 0.
        t_final = 1.
        delta0 = 0.
        vel0 = 0.
        Psi0 = 0.
        sy0 = 0.
        initial_state = [0,sy0,delta0,vel0,Psi0]

        x0_KS = np.array(initial_state[0:5])

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set decel input
        u = np.array([0.15, 0.63*g])

        # simulate kinematic single-track model
        x_acc_ks = odeint(func_KS, x0_KS, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))

        # ground truth for kinematic single-track model
        x_acc_ks_gt = [3.0845676868494927, 0.1484249221523042, 0.1500000000000000, 6.1803000000000017, 0.1203664469224163]

        self.assertTrue(all(abs(x_acc_ks[-1] - x_acc_ks_gt) < 1e-2))

    def test_zeroinit_rollleft(self):
        from scipy.integrate import odeint

        # testing for zero initial state, rolling and steering left input singularities
        g = 9.81
        t_start = 0.
        t_final = 1.
        delta0 = 0.
        vel0 = 0.
        Psi0 = 0.
        sy0 = 0.
        initial_state = [0,sy0,delta0,vel0,Psi0]

        x0_KS = np.array(initial_state[0:5])

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set decel input
        u = np.array([0.15, 0.])

        # simulate kinematic single-track model
        x_left_ks = odeint(func_KS, x0_KS, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))

        # ground truth for kinematic single-track model
        x_left_ks_gt = [0.0000000000000000, 0.0000000000000000, 0.1500000000000000, 0.0000000000000000, 0.0000000000000000]

        self.assertTrue(all(abs(x_left_ks[-1] - x_left_ks_gt) < 1e-2))

    def test_pacejka_steady_state(self):
        # small slip: the yaw rate must match the linear bicycle steady state with the Pacejka slopes as cornering stiffness
        g = 9.81
        mu, lf, lr, h, m, I = 1.0, 0.130, 0.127, 0.074, 2.783, 0.0627
        tires = (4.142, 0.4375, 2.6476, 0.2332, 4.8721, 2.1651, 1.057, 1.692)
        Bf, Cf, Df, Ef, Br, Cr, Dr, Er = tires
        limits = (-0.4189, 0.4189, -3.2, 3.2, 7.319, 5.0, -5.0, 15.0)

        x = np.array([0., 0., 0.02, 3., 0., 0., 0.])
        u = np.zeros(2)
        dt = 1e-3
        f = lambda x: vehicle_dynamics_st_pacejka(x, u, mu, *tires, lf, lr, h, m, I, *limits)
        for _ in range(3000):
            k1 = f(x)
            k2 = f(x + dt/2*k1)
            k3 = f(x + dt/2*k2)
            k4 = f(x + dt*k3)
            x = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

        L = lf + lr
        C_alpha_f = mu*Df*Bf*Cf*m*g*lr/L
        C_alpha_r = mu*Dr*Br*Cr*m*g*lf/L
        K_us = m/L*(lr/C_alpha_f - lf/C_alpha_r)
        yaw_rate_gt = x[3]*x[2]/(L + K_us*x[3]**2)
        self.assertAlmostEqual(x[5]/yaw_rate_gt, 1., delta=0.02)

if __name__ == '__main__':
    unittest.main()