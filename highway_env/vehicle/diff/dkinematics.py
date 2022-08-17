from threading import local
from typing import List
from highway_env.vehicle.kinematics import Vehicle

import torch as th

class dVehicle:

    @classmethod
    def step(self,
                vehicles: List[Vehicle],
                vehicles_position: th.Tensor,
                vehicles_speed: th.Tensor,
                vehicles_heading: th.Tensor,
                vehicles_length: th.Tensor,
                vehicles_action: th.Tensor,
                vehicles_max_speed: th.Tensor,
                vehicles_min_speed: th.Tensor,
                dt: float) -> None:
        assert vehicles_action is not None, ""
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        self.clip_actions(vehicles,
                            vehicles_speed,
                            vehicles_action,
                            vehicles_max_speed,
                            vehicles_min_speed)

        delta_f = vehicles_action[:,[0]]
        beta = th.arctan(1 / 2 * th.tan(delta_f))
        v = vehicles_speed * th.cat([th.cos(vehicles_heading + beta),
                                        th.sin(vehicles_heading + beta)], dim=1)
        
        vehicles_position += v * dt
        vehicles_heading += vehicles_speed * th.sin(beta) / (vehicles_length / 2) * dt
        vehicles_speed += vehicles_action[:,[1]] * dt

        # ===================== Non-differentiable block
        # Apply updated position, heading, speed to each vehicle
        for i, vehicle in enumerate(vehicles):
            if vehicle.impact is not None:
                vehicles_position[i] += vehicle.impact
                vehicle.crashed = True
                vehicle.impact = None
            vehicle.position = vehicles_position[i].detach().cpu().numpy()
            vehicle.heading = vehicles_heading[i].detach().cpu().item()
            vehicle.speed = vehicles_speed[i].detach().cpu().item()

            vehicle.on_state_update()

    @classmethod
    def clip_actions(self,
                        vehicles: List[Vehicle],
                        vehicles_speed: th.Tensor,
                        vehicles_action: th.Tensor,
                        vehicles_max_speed: th.Tensor,
                        vehicles_min_speed: th.Tensor) -> None:
        # ====================== Non-differentiable block
        for i, vehicle in enumerate(vehicles):
            if vehicle.crashed:
                vehicles_action[i][0] = 0
                vehicles_action[i][1] = -1.0 * vehicles_speed[i]    

            speed = vehicles_speed[i]
            acc = vehicles_action[i][1]
            max_speed = vehicles_max_speed[i]
            min_speed = vehicles_min_speed[i]

            if speed > max_speed:
                vehicles_action[i][1] = th.min(acc, 1.0 * (max_speed - speed))
            elif speed < min_speed:
                vehicles_action[i][1] = th.max(acc, 1.0 * (min_speed - speed))

    @classmethod
    def direction(self,
                vehicles_heading: th.Tensor) -> th.Tensor:
        dir_x = th.cos(vehicles_heading)
        dir_y = th.sin(vehicles_heading)
        return th.cat([dir_x, dir_y], dim=1)

    @classmethod
    def velocity(self,
                    vehicles_heading: th.Tensor,
                    vehicles_speed: th.Tensor) -> th.Tensor:
        return self.direction(vehicles_heading) * vehicles_speed

    @classmethod
    def destination(self, 
                    vehicles: List[Vehicle],
                    vehicles_position: th.Tensor) -> th.Tensor:
        dest = vehicles_position.clone()
        for i, vehicle in enumerate(vehicles):
            if getattr(vehicle, "route", None):
                last_lane_index = vehicle.route[-1]
                last_lane_index = last_lane_index if last_lane_index[-1] is not None else (*last_lane_index[:-1], 0)
                last_lane = vehicle.road.network.get_lane(last_lane_index)
                dpos = last_lane.position(last_lane.length, 0)
                dest[i] = th.tensor(dpos, device=vehicles_position.device, dtype=vehicles_position.dtype)
        return dest

    @classmethod
    def destination_direction(self,
                                vehicles: List[Vehicle],
                                vehicles_position: th.Tensor) -> th.Tensor:
        dest = self.destination(vehicles, vehicles_position)
        if th.any(dest - vehicles_position):
            return (dest - vehicles_position) / th.norm(dest - vehicles_position)
        else:
            return th.zeros_like(vehicles_position, device=vehicles_position.device, dtype= vehicles_position.dtype)

    @classmethod
    def lane_offset(self,
                    vehicles: List[Vehicle],
                    vehicles_position: th.Tensor,
                    vehicles_heading: th.Tensor) -> th.Tensor:
        num_vehicles = len(vehicles)
        offset = th.zeros((num_vehicles, 3), device=vehicles_position.device, dtype=vehicles_position.dtype)
        for i, vehicle in enumerate(vehicles):        
            if vehicle.lane is not None:
                local_coords = vehicle.lane.local_coordinates(vehicles_position[i])
                long, lat = local_coords[0], local_coords[1]
                ang = vehicle.lane.local_angle(vehicles_heading[i], long)
                offset[i][0] = long
                offset[i][1] = lat
                offset[i][2] = ang
        return offset

    @classmethod
    def to_dict(self,
                vehicles: List[Vehicle], 
                vehicles_position: th.Tensor,
                vehicles_heading: th.Tensor,
                vehicles_speed: th.Tensor,
                origin_vehicle_id: int = None,
                observe_intentions: bool = True) -> List[dict]:
        num_vehicles = len(vehicles)

        device = vehicles_position.device
        dtype = vehicles_position.dtype

        vehicles_velocity = self.velocity(vehicles_heading, vehicles_speed)
        vehicles_direction = self.direction(vehicles_heading)
        vehicles_destination_direction = self.destination_direction(vehicles, vehicles_position)
        vehicles_lane_offset = self.lane_offset(vehicles, vehicles_position, vehicles_heading)

        dicts = []
        for i in range(num_vehicles):
            d = {
                'presence': 1,
                'x': vehicles_position[i][0],
                'y': vehicles_position[i][1],
                'vx': vehicles_velocity[i][0],
                'vy': vehicles_velocity[i][1],
                'heading': vehicles_heading[i],
                'cos_h': vehicles_direction[i][0],
                'sin_h': vehicles_direction[i][1],
                'cos_d': vehicles_destination_direction[i][0],
                'sin_d': vehicles_destination_direction[i][1],
                'long_off': vehicles_lane_offset[i][0],
                'lat_off': vehicles_lane_offset[i][1],
                'ang_off': vehicles_lane_offset[i][2],
            }
            if not observe_intentions:
                d["cos_d"] = d["sin_d"] = th.zeros((1), device=device, dtype=dtype)
            dicts.append(d)

        if origin_vehicle_id:
            origin_dict = dicts[origin_vehicle_id]
            for i, d in enumerate(dicts):
                # Do not normalize origin vehicle
                if i == origin_vehicle_id:
                    continue
                for key in ['x', 'y', 'vx', 'vy']:
                    d[key] -= origin_dict[key]
        return dicts