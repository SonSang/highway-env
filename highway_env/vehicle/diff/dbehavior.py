import enum
from typing import Tuple, Union, List
from highway_env.vehicle.behavior import IDMVehicle

import numpy as np

from highway_env.road.road import Road, Route, LaneIndex
from highway_env.utils import Vector
from highway_env.vehicle.controller import ControlledVehicle
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle

import torch as th
from highway_env.diff.dutils import d_not_zero, d_wrap_to_pi
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.diff.dkinematics import dVehicle

class dIDM:
    """
    A class that accepts pytorch Tensor that describes states of IDM vehicles
    and produce the appropriate actions for each of them using IDM rules.
    """

    @classmethod
    def steering_control(self, 
                        vehicles: List[Vehicle], 
                        vehicles_position: th.Tensor,
                        vehicles_speed: th.Tensor,
                        vehicles_heading: th.Tensor,
                        vehicles_length: th.Tensor,
                        vehicles_tau_pursuit: th.Tensor,
                        vehicles_kp_lateral: th.Tensor,
                        vehicles_kp_heading: th.Tensor,
                        vehicles_max_steering_angle: th.Tensor,
                        device: str = None,
                        dtype = None,
                        ) -> th.Tensor:
        """
        Steer the vehicle to follow the center of an given lane.
        This is the parallelized version of [steering_control] of [ControlledVehicle] (at controller.py).
        """
        if device is None:
            device = vehicles_position.device
        if dtype is None:
            dtype = vehicles_position.dtype

        num_vehicles = len(vehicles)
        
        # Find position at the target lane and expected heading.
        # ================================ Non-parallelized block
        lane_coords = th.zeros((num_vehicles, 2), device=device, dtype=dtype)
        lane_next_coords = th.zeros((num_vehicles, 2), device=device, dtype=dtype)
        lane_future_heading = th.zeros((num_vehicles, 1), device=device, dtype=dtype)

        for i, vehicle in enumerate(vehicles):
            if not isinstance(vehicle, IDMVehicle):
                continue
            nv: IDMVehicle = vehicle
            target_lane = vehicle.road.network.get_lane(nv.target_lane_index)
            lane_coords[i] = target_lane.local_coordinates(vehicles_position[i])
            lane_next_coords[i] = lane_coords[i][0] + vehicles_speed[i] * vehicles_tau_pursuit[i]
            lane_future_heading[i] = target_lane.heading_at(lane_next_coords[i])

        # ================================ Parallelized block
        # Clip speed to be non-zero
        non_zero_vehicles_speed = d_not_zero(vehicles_speed)

        # Lateral position control
        lateral_speed_command = - vehicles_kp_lateral * lane_coords[:, 1]
        # Lateral speed to heading
        heading_command = th.arcsin(th.clip(lateral_speed_command / non_zero_vehicles_speed, -1, 1))
        heading_ref = lane_future_heading + th.clip(heading_command, -th.pi/4, th.pi/4)
        # Heading control
        heading_rate_command = vehicles_kp_heading * d_wrap_to_pi(heading_ref - vehicles_heading)
        # Heading rate to steering angle
        slip_angle = th.arcsin(th.clip(vehicles_length / 2 / non_zero_vehicles_speed * heading_rate_command, -1, 1))
        steering_angle = th.arctan(2 * th.tan(slip_angle))
        steering_angle = th.clip(steering_angle, -vehicles_max_steering_angle, vehicles_max_steering_angle)
        return steering_angle.to(dtype)

    @classmethod
    def lane_distance_to(self,
                        ego_position: th.Tensor,
                        other_position: th.Tensor,
                        lane: 'AbstractLane'):
        return lane.local_coordinates(other_position)[0] - lane.local_coordinates(ego_position)[0]

    @classmethod
    def desired_gap(self, 
                    ego_distance_wanted: th.Tensor,
                    ego_time_wanted: th.Tensor,
                    ego_comfort_acc_max: th.Tensor,
                    ego_comfort_acc_min: th.Tensor,
                    ego_heading: th.Tensor,
                    ego_speed: th.Tensor,
                    front_heading: th.Tensor,
                    front_speed: th.Tensor, 
                    projected: bool = True) -> th.Tensor:
        """
        Compute the desired distance between a vehicle and its leading vehicle.
        """
        d0 = ego_distance_wanted
        tau = ego_time_wanted
        ab = -ego_comfort_acc_max * ego_comfort_acc_min

        ego_velocity = dVehicle.velocity(ego_heading, ego_speed)
        ego_direction: th.Tensor = dVehicle.direction(ego_heading)
        front_velocity = dVehicle.velocity(front_heading, front_speed)

        vel_diff = ego_velocity - front_velocity

        dv = th.matmul(vel_diff.unsqueeze(-2), ego_direction.unsqueeze(-1)).squeeze(-1) if projected \
            else ego_speed - front_speed
        d_star = d0 + ego_speed * tau + ego_speed * dv / (2 * th.sqrt(ab))
        return d_star

    @classmethod
    def acceleration(self,
                        vehicles: List[Vehicle],
                        vehicles_position: th.Tensor,
                        vehicles_heading: th.Tensor,
                        vehicles_speed: th.Tensor,
                        vehicles_target_speed: th.Tensor,
                        vehicles_comfort_acc_max: th.Tensor,
                        vehicles_comfort_acc_min: th.Tensor,
                        vehicles_delta: th.Tensor,
                        vehicles_distance_wanted: th.Tensor,
                        vehicles_time_wanted: th.Tensor,
                        device: str = None,
                        dtype = None,
                        ):
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.
        """
        if device is None:
            device = vehicles_position.device
        if dtype is None:
            dtype = vehicles_position.dtype

        road = vehicles[0].road

        # First, find the preceding and following vehicles of a given vehicle.
        # ================================ Non-parallelized block
        num_vehicles = len(vehicles)
        front_vehicles_distance = th.zeros((num_vehicles, 1), device=device, dtype=dtype)
        front_vehicles_speed = th.zeros((num_vehicles, 1), device=device, dtype=dtype)
        front_vehicles_heading = th.zeros((num_vehicles, 1), device=device, dtype=dtype)

        for i, vehicle in enumerate(vehicles):
            if not isinstance(vehicle, IDMVehicle):
                continue

            front_id, _ = road.neighbour_vehicles(vehicle)
            if front_id is not None:
                distance = dIDM.lane_distance_to(vehicles_position[i], vehicles_position[front_id], vehicle.lane)
                front_vehicles_distance[i] = distance
                front_vehicles_speed[i] = vehicles_speed[front_id]
                front_vehicles_heading[i] = vehicles_heading[front_id]

        # Compute acceleration
        # ================================ Parallelized block
        non_zero_target_speed = d_not_zero(vehicles_target_speed)
        ego_target_speed = th.abs(non_zero_target_speed)
        acceleration = vehicles_comfort_acc_max * (
                1 - th.pow(th.max(vehicles_speed, th.zeros_like(vehicles_speed)) / ego_target_speed, vehicles_delta))

        gap = self.desired_gap(
                    vehicles_distance_wanted,
                    vehicles_time_wanted,
                    vehicles_comfort_acc_max,
                    vehicles_comfort_acc_min,
                    vehicles_heading,
                    vehicles_speed,
                    front_vehicles_heading,
                    front_vehicles_speed
                )
        acceleration -= vehicles_comfort_acc_max * \
            th.pow(gap / d_not_zero(front_vehicles_distance), 2)
        return acceleration