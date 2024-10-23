import time
from f110_gym.envs.utils import ensure_absolute_path
import yaml
import gymnasium as gym
import numpy as np
import os
from argparse import Namespace
import git

from numba import njit

from pyglet.gl import GL_POINTS

"""
Planner Helpers
"""


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return (
        projections[min_dist_segment],
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(
    point, radius, trajectory, t=0.0, wrap=False
):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = (
            np.dot(start, start)
            + np.dot(point, point)
            - 2.0 * np.dot(start, point)
            - radius * radius
        )
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = (
                np.dot(start, start)
                + np.dot(point, point)
                - 2.0 * np.dot(start, point)
                - radius * radius
            )
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)]),
        lookahead_point[0:2] - position,
    )
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.0
    radius = 1 / (2.0 * waypoint_y / lookahead_distance**2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle


class PurePursuitPlanner:
    """
    Example Planner
    """

    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.0
        self.last_lookahead_point_in_map = None

        self.drawn_waypoints = []

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        map_name = conf.map_name
        random_direction = conf.random_direction
        raceline_option = conf.raceline_option
        map_dir = conf.map_dir
        repo = git.Repo(os.path.abspath(__file__), search_parent_directories=True)
        map_dir = repo.working_tree_dir + map_dir.format(map_name=map_name)
        self.map_base_path = map_dir + f"{map_name}"
        self.wpt_path = self.map_base_path + f"_{raceline_option}.csv"
        self.waypoints_forward = np.loadtxt(
            self.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip
        )
        if random_direction:
            parts = self.wpt_path.split(os.sep)
            filename, ext = os.path.splitext(parts[-1])
            new_name = filename.replace("raceline", "reverse_raceline")
            parts[-1] = new_name + ext
            parts[-2] += "_reverse"
            self.wpt_path_reverse = ensure_absolute_path(os.path.join(*parts))

            if os.path.exists(self.wpt_path_reverse):
                self.waypoints_reverse = np.loadtxt(
                    self.wpt_path_reverse,
                    delimiter=conf.wpt_delim,
                    skiprows=conf.wpt_rowskip,
                )
            else:
                print(
                    "The reverse map does not exist! No randomization in the direction."
                )
                self.waypoints_reverse = self.waypoints_forward

            self.waypoints = self.waypoints_reverse
        else:
            self.waypoints = self.waypoints_forward

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """

        # points = self.waypoints

        points = np.vstack(
            (
                self.waypoints[:, self.conf.wpt_xind],
                self.waypoints[:, self.conf.wpt_yind],
            )
        ).T

        scaled_points = 50.0 * points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(
                    1,
                    GL_POINTS,
                    None,
                    ("v3f/stream", [scaled_points[i, 0], scaled_points[i, 1], 0.0]),
                    ("c3B/stream", [183, 193, 222]),
                )
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [
                    scaled_points[i, 0],
                    scaled_points[i, 1],
                    0.0,
                ]

    def _get_current_waypoint(self, waypoints, la_q, la_m, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack(
            (
                self.waypoints[:, self.conf.wpt_xind],
                self.waypoints[:, self.conf.wpt_yind],
            )
        ).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        wp_vel = waypoints[i, 2]
        lookahead_distance = max(la_m * wp_vel + la_q, 1.4 * nearest_dist)
        lookahead_distance = max(0.7, lookahead_distance) # clip from below like VMAP, VPP
        lookahead_distance = min(lookahead_distance, 5.0) # clip from above like VMAP, VPP
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position, lookahead_distance, wpts, i + t, wrap=True
            )
            if i2 == None:
                return None, None
            current_waypoint = np.empty((3,))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint, lookahead_distance
        elif nearest_dist < self.max_reacquire:
            print(f"This is happening! {nearest_dist}")
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind]), nearest_dist # TODO what is this doing?
        else:
            return None, None

    def plan(
        self,
        pose_x,
        pose_y,
        pose_theta,
        la_m,
        la_q,
        vgain,
        reverse_direction=False,
    ):
        if not reverse_direction:
            self.waypoints = self.waypoints_forward
        else:
            self.waypoints = self.waypoints_reverse

        position = np.array([pose_x, pose_y])
        lookahead_point, lookahead_distance = self._get_current_waypoint(
            self.waypoints, la_q, la_m, position, pose_theta
        )

        if lookahead_point is None:
            print("No lookahead point found!")
            return 4.0, 0.0

        self.last_lookahead_point_in_map = lookahead_point
        speed, steering_angle = get_actuation(
            pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase
        )
        speed = vgain * speed

        return speed, steering_angle


def main():
    """
    main entry point
    """

    with open(
        "/home/jonathan/MASTER_THESIS_RL/pbl-f1tenth-gym/wandb_trains/wandb_config.yaml"
    ) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        planner.render_waypoints(env_renderer)

    env = env = gym.make("f110_gym:f110-v0", render_mode="human", **conf_dict)
    env.add_render_callback(render_callback)

    planner = PurePursuitPlanner(conf, (env.params["length"]))

    done = False
    env.reset()
    env.render()

    laptime = 0.0
    start = time.time()

    while not done:
        speed, steer = planner.plan(
            env.poses_x[0],
            env.poses_y[0],
            env.poses_theta[0],
            conf.lookahead_distance,
            conf.vgain,
        )
        obs, reward, done, truncated, info = env.step(np.array([steer, speed]))
        laptime += reward
        env.render()

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)


if __name__ == "__main__":
    main()
