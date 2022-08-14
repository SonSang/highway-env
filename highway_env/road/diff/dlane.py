from highway_env.road.lane import AbstractLane, CircularLane, LineType
from highway_env.utils import wrap_to_pi, Vector, get_class_path, class_from_path

from typing import List

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
        phi = self.direction * longitudinal / self.radius + self.start_phase
        psi = phi + th.pi/2 * self.direction
        return psi

    def local_coordinates(self, position: th.Tensor) -> th.Tensor:
        delta = position - self.center
        phi = th.arctan2(delta[1], delta[0])
        phi = self.start_phase + wrap_to_pi(phi - self.start_phase)
        r = th.norm(delta)
        longitudinal = self.direction*(phi - self.start_phase)*self.radius
        lateral = self.direction*(self.radius - r)
        return longitudinal, lateral