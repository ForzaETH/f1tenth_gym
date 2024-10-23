import enum
import random
import logging
import numpy as np


class RewardMode(enum.Enum):
    # supported types of rewards are here
    ADV = "advancement"
    ADV_STEP = "stepwise advancement"
    SPEED = "speed"
    ADV_STEP_AND_SPEED = "stepwise advancement + speed"


class Reward:
    def __init__(
        self,
        mode: RewardMode = RewardMode.ADV,
        env=None,
        penalty_based_on_reward=False,
        deviation_penalty_coefficient=1,
        deviation_penalty_treshold=0.1,
        rel_heading_penalty_coefficient=0.25,
        rel_heading_penalty_threshold=0,
        jerk_scale=100,
        steering_smoothness_penalty_coefficient=0.1,
        speed_smoothness_penalty_coefficient=0.1,
    ):
        assert (
            mode in RewardMode
        ), f"The mode: {mode} is not in the supported list: {[mode for mode in RewardMode]}"

        self.PENALTY = -1e3
        self.mode = mode
        self.env = env
        self.penalty_based_on_reward = penalty_based_on_reward

        self.deviation_penalty_coefficient = deviation_penalty_coefficient
        self.deviation_penalty_treshold = deviation_penalty_treshold
        self.rel_heading_penalty_coefficient = rel_heading_penalty_coefficient
        self.rel_heading_penalty_threshold = rel_heading_penalty_threshold
        self.jerk_scale = jerk_scale

        self.steering_smoothness_penalty_coefficient = steering_smoothness_penalty_coefficient
        self.speed_smoothness_penalty_coefficient = speed_smoothness_penalty_coefficient

    def get_reward(self, obs, action):

        step_rew = 0
        reward_info = {"type":None, "rewards":{}, "penalties":{}}

        if self.mode == RewardMode.ADV:
            step_rew = self._get_reward_adv()
            reward_info["type"] = "advancement"
            reward_info["rewards"]["adv value"] = step_rew
        elif self.mode == RewardMode.ADV_STEP:
            step_rew = self._get_reward_adv_step()
            reward_info["type"] = "percentual advancement"
            reward_info["rewards"]["adv step value"] = step_rew
        elif self.mode == RewardMode.SPEED:
            step_rew = self._get_reward_speed()
            reward_info["type"] = "speed"
            reward_info["rewards"]["speed value"] = step_rew
        elif self.mode == RewardMode.ADV_STEP_AND_SPEED:
            step_rew = self._get_reward_adv_step_and_speed()
            reward_info["type"] = "percentual advancement + speed"
            reward_info["rewards"]["adv step speed value"] = step_rew

        if self.env.observator.denorm_obs["deviation"] is not None:
            dev_pen = self.calc_deviation_penalty(
                step_rew, self.env.observator.denorm_obs["deviation"]
            )
            step_rew -= dev_pen
            reward_info["penalties"]["deviation"] = dev_pen

        if self.env.observator.denorm_obs["rel_heading"] is not None:
            rel_heading_pen = self.calc_rel_heading_penalty(
                step_rew, self.env.observator.denorm_obs["rel_heading"]
            )
            step_rew -= rel_heading_pen
            reward_info["penalties"]["rel_heading"] = rel_heading_pen

        if (
            self.env.use_previous_actions
            and self.env.observator.denorm_obs["previous_s"] is not None
        ):
            smoothness_pen = self.calc_smoothness_penalty(action, step_rew)
            step_rew -= smoothness_pen
            reward_info["penalties"]["smoothness"] = smoothness_pen
            
        if self.env.use_action_smoothness_penalty:
            action_smoothness_pen = self.calc_action_smoothness_penalty(action)
            step_rew -= action_smoothness_pen
            reward_info["penalties"]["action_smoothness"] = action_smoothness_pen
            
        if self.env.sim.agents[0].jerk is not None:
            jerk_pen = 1-np.exp(-(np.linalg.norm(self.env.sim.agents[0].jerk) / self.jerk_scale)**2)
            step_rew -= jerk_pen
            reward_info["penalties"]["jerk"] = jerk_pen
            

        if obs["collisions"][0] != 0.0:
            step_rew = -10
            reward_info["penalties"]["collision"] = step_rew
        else:
            reward_info["penalties"]["collision"] = 0

        logging.log(
            level=logging.DEBUG, msg=f"Last reward (of type {self.mode}): {step_rew}"
        )
        return step_rew, reward_info

    def get_reward_for_predicted_obs(self, predicted_obs, predicted_theta, action):

        step_rew = 0

        if self.mode == RewardMode.ADV:
            step_rew = self._get_reward_adv()
        elif self.mode == RewardMode.ADV_STEP:
            step_rew = self._get_reward_adv_step()
        elif self.mode == RewardMode.SPEED:
            step_rew = self._get_reward_speed()
        elif self.mode == RewardMode.ADV_STEP_AND_SPEED:
            step_rew = self._get_reward_adv_step_and_speed()

        if self.env.observator.denorm_obs["deviation"] is not None:
            step_rew -= self.calc_deviation_penalty(
                step_rew, self.env.prediction_observator.denorm_obs["deviation"]
            )

        if self.env.observator.denorm_obs["rel_heading"] is not None:

            step_rew -= self.calc_rel_heading_penalty(
                step_rew, self.env.prediction_observator.denorm_obs["rel_heading"]
            )

        if (
            self.env.use_previous_actions
            and self.env.observator.denorm_obs["previous_s"] is not None
        ):
            step_rew -= self.calc_smoothness_penalty(action, step_rew)

        if predicted_obs["collisions"][0] != 0.0:
            step_rew = -1

        logging.log(
            level=logging.DEBUG, msg=f"Last reward (of type {self.mode}): {step_rew}"
        )

        return step_rew

    def calc_deviation_penalty(self, step_rew, deviation):
        deviation_in_percentage = abs(deviation) / self.env.calculate_half_track_width(deviation)
        if deviation_in_percentage < self.deviation_penalty_treshold:
            deviation_in_percentage = 0

        if self.penalty_based_on_reward:
            return (
                deviation_in_percentage * step_rew * self.deviation_penalty_coefficient
            )
        else:
            return deviation_in_percentage * self.deviation_penalty_coefficient

    def calc_rel_heading_penalty(self, step_rew, rel_heading):
        rel_heading_deviation_in_percentage = abs(rel_heading) / self.env.global_s_max
        if rel_heading_deviation_in_percentage < self.rel_heading_penalty_threshold:
            rel_heading_deviation_in_percentage = 0

        if self.penalty_based_on_reward:
            return (
                rel_heading_deviation_in_percentage
                * step_rew
                * self.rel_heading_penalty_coefficient
            )
        else:
            return (
                rel_heading_deviation_in_percentage
                * self.rel_heading_penalty_coefficient
            )
            
    def calc_action_smoothness_penalty(self, action) -> float:
        try:
            prev_steering = self.env.last_actions[0][0]
            prev_speed = self.env.last_actions[0][1]
        except TypeError: 
            logging.error("No previous actions found. Returning 0 as penalty. This happens at the first iteration")
            return 0
        except AttributeError:
            logging.error("No previous actions found. Returning 0 as penalty.")
            return 0
        
        steer_delta = abs(prev_steering - action[0, 0])/self.env.global_s_max
        speed_delta = abs(prev_speed - action[0, 1])/self.env.global_v_max
        
        steer_penalty = steer_delta * self.steering_smoothness_penalty_coefficient
        speed_penalty = speed_delta * self.speed_smoothness_penalty_coefficient
        
        return steer_penalty + speed_penalty
        

    def calc_smoothness_penalty(self, action, step_rew) -> float:
        if (
            "pp_s" in self.env.observator.denorm_obs.keys()
            and self.env.observator.denorm_obs["pp_s"] is not None
        ):
            steering_input = self.env.observator.denorm_obs["pp_s"] + action[0, 0]
            speed_input = self.env.observator.denorm_obs["pp_v"] + action[0, 1]
        else:
            steering_input = 0
            speed_input = 0
        controller_steering_smoothness_in_percentage = (
            abs(self.env.observator.denorm_obs["previous_s"] - steering_input)
            / self.env.global_s_max
        )
        controller_speed_smoothness_in_percentage = (
            abs(self.env.observator.denorm_obs["previous_v"] - speed_input)
            / self.env.global_v_max
        )

        if self.penalty_based_on_reward:
            return (
                controller_steering_smoothness_in_percentage
                * self.steering_smoothness_penalty_coefficient
                + controller_speed_smoothness_in_percentage
                * self.speed_smoothness_penalty_coefficient
            )
        else:
            return (
                controller_steering_smoothness_in_percentage
                * self.steering_smoothness_penalty_coefficient
                + controller_speed_smoothness_in_percentage
                * self.speed_smoothness_penalty_coefficient
            )

    ####################################################################################################################
    ###############################################PRIVATE REWARDS######################################################
    ####################################################################################################################

    def _get_reward_adv(self):
        """
        Returns the advancement of last timestep calculated along the central line
        """
        return self.env.theta - self.env.start_theta

    def _get_reward_adv_step(self):
        abs_adv = self.env.theta - self.env.prev_theta
        max_adv = self.env.global_v_max * self.env.timestep
        adv_in_percentage = abs_adv / max_adv

        return adv_in_percentage

    def _get_reward_speed(self):
        """
        Returns the current velocity of the car as a reward
        """
        return self.env.sim.agents[self.env.ego_idx].state[3] / self.env.v_max

    def _get_reward_adv_step_and_speed(self):
        return self._get_reward_adv_step() + self._get_reward_speed()

    def _get_safety_penalty(self):
        """
        returns a negative reward (penalty), for going too distant for the center of the track
        it is:
            - 0.01 when distance to center is < safety threshold
            - 0 otherwiese
        """
        perc = 1  # percentual can be changed here

        car_current_position = np.array(self.env.sim.agents[self.env.ego_idx].state[:2])
        current_mid_point = np.array(self.env.track.get_coordinate(self.env.theta))
        current_int_point = np.array(
            self.env.track.get_coordinate(self.env.theta, line="int")
        )
        current_out_point = np.array(
            self.env.track.get_coordinate(self.env.theta, line="out")
        )

        safe_width_int_mid = perc * np.linalg.norm(
            current_int_point - current_mid_point
        )

        safe_width_out_mid = perc * np.linalg.norm(
            current_out_point - current_mid_point
        )

        dist_int_to_mid = np.linalg.norm(current_int_point - current_mid_point)
        dist_int_to_out = np.linalg.norm(current_int_point - current_out_point)
        dist_int_to_car = np.linalg.norm(current_int_point - car_current_position)
        dist_to_mid = np.linalg.norm(current_mid_point - car_current_position)

        dist_list = [dist_int_to_mid, dist_int_to_out, dist_int_to_car]
        dist_list.sort()
        if dist_list[0] == dist_int_to_car:
            safe_width = safe_width_int_mid
        else:
            safe_width = safe_width_out_mid

        safe_threshold_min = (
            0.3302 * 0.5
        )  # TODO remove hardcoding of the 1/2 vehicle length

        if safe_width - dist_to_mid < safe_threshold_min:
            # penalty = -(dist_to_mid / safe_width) ** 10  # TODO remove hardcoding: penalty = v_max*time_step
            penalty = -1
        else:
            penalty = 0

        return penalty

    def _get_reward_deviation(self, obs):
        deviation_penalty = obs["deviation"][0] * 0.05
        return deviation_penalty

    def _get_reward_tire_slip(self):
        penalty = 0
        if abs(self.env.sim.agents[self.env.ego_idx].state[6]) > 0.5:
            penalty = -self.env.sim.agents[self.env.ego_idx].state[6] * 10
        return penalty

    # TODO not yet implemented rewards are below

    def _get_reward_time(
        self,
        sim,
        ego_idx,
        next_waypoint,
        last_visited_times,
        track,
        current_time,
        times,
    ) -> int:
        """
        Obtains the reward, based on waypoints situated at theta == int(theta).
        Rewards the improvement on the last track time.

        Returns:
            reward: negative delta from last lap time to best lap time at the current waypoint
        """

        reward = 0
        car = sim.agents[ego_idx]

        car_position = car.state[:2]
        idx = next_waypoint - 1
        # print(next_waypoint)

        if track.is_coord_behind(car_position, idx):
            # going backwards
            # therefore go back with the waypoint and reset the timer
            next_waypoint -= 1
            last_visited_times[next_waypoint] = -1

            if next_waypoint < 0:
                if next_waypoint != -1:
                    print("weird shit while crossing the starting line backwards!")
                else:
                    next_waypoint = int(np.floor(track.track_length))
            # give penalty
            print("feels like we only go backwards")
            reward = self.PENALTY

        elif not track.is_coord_behind(car_position, next_waypoint):

            if not last_visited_times[next_waypoint] == -1:
                last_lap_time = current_time - last_visited_times[next_waypoint]
                if random.random() > 0.99:
                    print(last_lap_time)
                reward = (
                    -(last_lap_time - times[next_waypoint]) + 1
                )  # there is a 1 second slack

                # scaling of time reward
                if reward >= 0:
                    reward = 10 ** (
                        3 + 0.58 * np.log(reward + 1)
                    )  # this was designed so that : 0-1 s improvements -> ~1e3 rew / ~5s impr. -> ~ 1e4 rew / 20s impr. -> 1e5 rew.
                else:
                    reward *= 1e3  # make slower lap times a bit worse

                # if last_lap_time <= times[next_waypoint]:
                # times[next_waypoint] = last_lap_time
            else:
                reward = 100
            last_visited_times[next_waypoint] = current_time
            next_waypoint += 1
            if next_waypoint >= track.track_length + 1:
                raise ValueError("Weird, maybe the car is too fast?")
            if next_waypoint >= track.track_length:
                next_waypoint = 0

            if not track.is_coord_behind(car_position, idx + 2):
                raise ValueError(
                    "Other weird shit is happening!"
                )  # TODO remove this if it works correctly

        # print(self.next_waypoint)
        return reward, next_waypoint, last_visited_times

    def _get_pseudocurr_reward(self, time_rew, speed_rew, adv_rew, sim, ego_idx):
        """
        Inspired to curriculum RL, it should first train the agent to do something simple (go forward quick) and then switch to
        something more involved (make better lap times)

        ideally advancement is 50% less important than speed at the beginning, but both are scaled around 0-100 rewards
        when laps are done then lap time becomes crucial, being scaled around 1e3 - 1e5
        """

        slip = abs(sim.agents[ego_idx].state[-1])

        return 600 * adv_rew + (120) * speed_rew + 100 * time_rew - 690 * slip

    def _get_reward_safety(self, sim, ego_idx, track, theta):
        """
        returns a negative reward (penalty), for going too distant for the center of the track
        it is approximately:
            - lesser than one when distance to center is < 80% track width
            - much greater than one, elsewhere
        """
        perc = 0.8  # percentual can be changed here

        dist = np.linalg.norm(
            np.array(sim.agents[ego_idx].state[:2])
            - np.array(track.get_coordinate(theta))
        )
        safe_width = perc * np.linalg.norm(
            np.array(track.get_coordinate(theta, line="int"))
            - np.array(track.get_coordinate(theta))
        )
        return -((dist / safe_width) ** 10)

    def _get_pseudoscaramuzza_reward(self):
        """
        Reward inspired by "Autonomous Drone Racing with Deep Reinforcement Learning" by Song et al.
        """
        part_adv = 100 * self.get_reward_adv()
        part_vel = self.get_reward_vel()
        part_saf = self.get_reward_safety()
        # print(part_adv, part_vel, part_saf)
        return part_adv + part_vel + part_saf
