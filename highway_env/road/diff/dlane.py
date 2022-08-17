from highway_env.road.lane import AbstractLane, CircularLane, LineType
from highway_env.utils import wrap_to_pi, Vector, get_class_path, class_from_path

from typing import List
from highway_env.vehicle.kinematics import Vehicle

import torch as th

class dCircularLane(CircularLane):

    """A lane going in circle arc."""

    def __init__(self,
                 center: Vector,
                 radius: float,
                 start_phase: float,
                 end_phase: float,
                 clockwise: bool = True,
                 width: float = AbstractLane.DEFAULT_WIDTH,
                 line_types: List[LineType] = None,
                 forbidden: bool = False,
                 speed_limit: float = 20,
                 priority: int = 0) -> None:
        super().__init__(
            center, 
            radius, 
            start_phase, 
            end_phase, 
            clockwise, 
            width, 
            line_types, 
            forbidden, 
            speed_limit, 
            priority)

    def heading_at(self, longitudinal: th.Tensor) -> th.Tensor:
        if not isinstance(longitudinal, th.Tensor):
            return super().heading_at(longitudinal)

        phi = self.direction * longitudinal / self.radius + self.start_phase
        psi = phi + th.pi/2 * self.direction
        return psi

    def local_coordinates(self, position: th.Tensor) -> th.Tensor:
        if not isinstance(position, th.Tensor):
            return super().local_coordinates(position)

        th_center = th.tensor(self.center, dtype=position.dtype, device=position.device)
        delta = position - th_center
        phi = th.arctan2(delta[1], delta[0])
        phi = self.start_phase + wrap_to_pi(phi - self.start_phase)
        r = th.norm(delta)
        longitudinal = self.direction*(phi - self.start_phase)*self.radius
        lateral = self.direction*(self.radius - r)
        coords = th.tensor([longitudinal, lateral], dtype=position.dtype, device=position.device)
        return coords

    def local_angle(self, heading: th.Tensor, long_offset: th.Tensor) -> th.Tensor:
        """Compute non-normalised angle of heading to the lane."""
        return wrap_to_pi(heading - self.heading_at(long_offset))

def d_on_lane(lane: AbstractLane, position: th.Tensor, longitudinal: th.Tensor = None, lateral: th.Tensor = None, margin: float = 0):
    """
    Whether a given world position is on the lane. (Differentiable version)

    :param position: a world position [m]
    :param longitudinal: (optional) the corresponding longitudinal lane coordinate, if known [m]
    :param lateral: (optional) the corresponding lateral lane coordinate, if known [m]
    :param margin: (optional) a supplementary margin around the lane width
    :return: is the position on the lane? (0~1) value, 1 means on the lane
    """
    if longitudinal is None or lateral is None:
        longitudinal, lateral = lane.local_coordinates(position)
    lateral_is_on = th.sigmoid(lane.width_at(longitudinal) / 2 + margin - th.abs(lateral))
    longitudinal_is_on = th.sigmoid(longitudinal + lane.VEHICLE_LENGTH) * th.sigmoid(lane.length + lane.VEHICLE_LENGTH - longitudinal)
    return lateral_is_on * longitudinal_is_on