from typing import Tuple, Optional

from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle

from highway_env.road.diff.droad import dRoad
import torch as th

from highway_env.envs.common.action import action_factory, Action
from highway_env.envs.common.observation import observation_factory
from highway_env.envs.common.diff.daction import dContinuousAction
from highway_env.road.diff.dlane import dCircularLane, d_on_lane
from highway_env.envs.common.diff.dobservation import dKinematicObservation


class SingleRingEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "dKinematics",
                "absolute": True,
                "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-15, 15], "vy": [-15, 15]},
            },
            "action": {
                "type": "dContinuousAction",
            },
            "incoming_vehicle_destination": None,
            "collision_reward": -1,
            "screen_width": 900,
            "screen_height": 900,
            "centering_position": [0.5, 0.6],
            "duration": 30,     # 30 sec
            "radius": 36.6,     # 230m circumference
            "num_vehicles": 22, # 22 cars
            "observer_mode": False,
            "no_negative_speed": True,
        })
        return config

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        try:
            self.observation_type = observation_factory(self, self.config["observation"])
        except:
            if self.config["observation"]["type"] == "dKinematics":
                self.observation_type = dKinematicObservation(self, **self.config["observation"])
            else:
                raise ValueError("Unknown observation type")

        try:
            self.action_type = action_factory(self, self.config["action"])
        except:
            if self.config["action"]["type"] == "dContinuousAction":
                self.action_type = dContinuousAction(self, **self.config["action"])
            else:
                raise ValueError("Unknown action type")

        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    def _reward(self, action) -> th.Tensor:
        # Reward is proportional to the average speed of every vehicle
        avg_speed = 0
        max_speed = 0
        for i, vehicle in enumerate(self.road.vehicles):
            avg_speed += self.road.road_object_speed[i]
            if vehicle.MAX_SPEED > max_speed:
                max_speed = vehicle.MAX_SPEED
        avg_speed /= len(self.road.vehicles)
        avg_speed /= max_speed

        # Ignore negative reward from crash, as it is non-differentiable
        # And also, since the episode ends when there is crash, we do not have to take care of it here
        reward = avg_speed # + self.config["collision_reward"] * self.vehicle.crashed

        # Use differentiable [on_lane] to find out if the ego vehicle is on the lane
        ego_vehicle_position = self.road.road_object_position[self.vehicle_id]
        on_lane = d_on_lane(self.vehicle.lane, ego_vehicle_position)
        reward = reward * on_lane
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def reset(self) -> th.Tensor:
        """
        Reset the environment to it's initial configuration

        :return: the observation of the reset state
        """
        self.update_metadata()
        self.define_spaces()  # First, to set the controlled vehicle class depending on action space
        self.time = self.steps = 0
        self.done = False
        self._reset()
        self.define_spaces()  # Second, to link the obs and actions to the vehicles once the scene is created
        return self.observation_type.observe(self.road.vehicles, 
                                            self.road.road_object_position, 
                                            self.road.road_object_heading,
                                            self.road.road_object_speed)

    def _make_road(self) -> None:
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est.
        center = [0, 0]  # [m]
        radius = self.config['radius']  # [m]
        
        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [c, c]

        # South to East
        net.add_lane("s", "e",
                        dCircularLane(center, radius, np.deg2rad(90), np.deg2rad(0),
                                    clockwise=False, line_types=line))
        # East to North
        net.add_lane("e", "n",
                        dCircularLane(center, radius, np.deg2rad(0), np.deg2rad(-90),
                                    clockwise=False, line_types=line))
        # North to West
        net.add_lane("n", "w",
                        dCircularLane(center, radius, np.deg2rad(-90), np.deg2rad(-180),
                                    clockwise=False, line_types=line))
        # West to South
        net.add_lane("w", "s",
                        dCircularLane(center, radius, np.deg2rad(180), np.deg2rad(90),
                                    clockwise=False, line_types=line))

        self.road = dRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        
    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway, as well as ego-vehicles.

        :return: the ego-vehicles
        """
        # Ego-vehicle
        ego_lane = self.road.network.get_lane(("e", "n", 0))
        if self.config["observer_mode"]:
            ego_position = np.array([0, 0])
        else:
            ego_position = ego_lane.position(0, 0)

        ego_vehicle = self.action_type.vehicle_class(self.road,
                                                     ego_position,
                                                     speed=0,
                                                     heading=ego_lane.heading_at(1))
        
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        self.vehicle_id = len(self.road.vehicles) - 1

        # Incoming vehicle
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # Other vehicles
        default_num_vehicles = self.config["num_vehicles"]
        num_vehicles = default_num_vehicles if self.config["observer_mode"] else default_num_vehicles - 1
        unit_longitudinal = 2.0 * np.pi * self.config["radius"] / default_num_vehicles
        start = 0 if self.config["observer_mode"] else unit_longitudinal

        for i in range(num_vehicles):
            vehicle = other_vehicles_type.make_on_lane(self.road,
                                                       ("e", "n", 0),
                                                       longitudinal=start + unit_longitudinal * i,
                                                       speed=0)      # 8.33m/sec = 30km/h
            if self.config["no_negative_speed"]:
                vehicle.MIN_SPEED = 0.0
            vehicle.target_speed = 8.33
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # Initialize tensor
        self.road.clear_torch_tensors()

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, dict]:
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        # [action] should be tensor
        if not isinstance(action, th.Tensor):
            action = th.tensor(action, dtype=self.road.torch_dtype, device=self.road.torch_device)

        self.time += 1 / self.config["policy_frequency"]

        # In this environment, [action] is a tensor.
        # After [simulate], all the state info is stored as tensor in [self.road].
        # We return non-tensor observations and rewards by detaching them,
        # but also return original tensors in [info] for future usage.
        self._simulate(action)

        obs: th.Tensor = self.observation_type.observe(self.road.vehicles,
                                            self.road.road_object_position,
                                            self.road.road_object_heading,
                                            self.road.road_object_speed,
                                            return_tensor=True)
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = self._info(obs, action)

        info["dobs"] = obs
        info["dreward"] = reward
        obs = obs.detach().cpu().numpy()
        reward = reward.detach().cpu().item()

        return obs, reward, terminal, info

    def _simulate(self, action: Optional[Action] = None) -> None:
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            # Forward action to the vehicle
            if action is not None \
                    and not self.config["manual_control"]:# \
                    #and self.steps % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.action_type.act(action)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.steps += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False

register(
    id='singlering-v0',
    entry_point='highway_env.envs:SingleRingEnv',
)