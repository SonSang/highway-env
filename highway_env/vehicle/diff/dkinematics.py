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

        delta_f = vehicles_action[:,0]
        beta = th.arctan(1 / 2 * th.tan(delta_f))
        v = vehicles_speed * th.cat([th.cos(vehicles_heading + beta),
                                        th.sin(vehicles_heading + beta)], dim=1)
        
        vehicles_position += v * dt
        vehicles_heading += vehicles_speed * th.sin(beta) / (vehicles_length / 2) * dt
        vehicles_speed += vehicles_action[:,1] * dt

        # ===================== Non-differentiable block
        # Apply updated position, heading, speed to each vehicle
        for i, vehicle in enumerate(vehicles):
            if vehicle.impact is not None:
                vehicles_position[i] += vehicle.impact
                vehicle.crashed = True
                vehicle.impact = None
            vehicle.position = vehicles_position[i].detach().numpy()
            vehicle.heading = vehicles_heading[i].detach().item()
            vehicle.speed = vehicles_speed[i].detach().item()

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