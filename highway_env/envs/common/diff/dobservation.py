from highway_env.envs.common.observation import KinematicObservation
from typing import List, TYPE_CHECKING, Dict
from gym import spaces
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.lane import AbstractLane

import pandas as pd
import torch as th
import numpy as np

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv

from highway_env import utils
from highway_env.vehicle.diff.dkinematics import dVehicle

class dKinematicObservation(KinematicObservation):

    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = False,
                 observe_intentions: bool = False,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(
            env, 
            features, 
            vehicles_count, 
            features_range, 
            absolute,
            order, 
            normalize, 
            clip, 
            see_behind, 
            observe_intentions, 
            **kwargs)

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
                "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    for value in df[feature].values:
                        value = th.clip(value, -1, 1)
        return df

    def observe(self, 
                vehicles: List[Vehicle], 
                vehicles_position: th.Tensor,
                vehicles_heading: th.Tensor,
                vehicles_speed: th.Tensor) -> th.Tensor:
        device = vehicles_position.device
        dtype = vehicles_position.dtype
        
        if not self.env.road:
            return th.zeros(self.space().shape, device=device, dtype=dtype)

        # To dict
        origin_vehicle_id = self.env.vehicle_id if not self.absolute else None
        vehicle_dicts = dVehicle.to_dict(vehicles, 
                                        vehicles_position, 
                                        vehicles_heading, 
                                        vehicles_speed, 
                                        origin_vehicle_id, 
                                        self.observe_intentions)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([vehicle_dicts[self.env.vehicle_id]])[self.features]
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind,
                                                         sort=self.order == "sorted")
        if close_vehicles:
            df = pd.concat([df, pd.DataFrame.from_records(
                [vehicle_dicts[i]
                 for i, _ in close_vehicles[-self.vehicles_count + 1:]])[self.features]],
                           ignore_index=True)
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = th.zeros((self.vehicles_count - df.shape[0], len(self.features)), device=device, dtype=dtype)
            df = pd.concat([df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True)
        # Reorder
        df = df[self.features]

        num_row = len(df.values)
        num_col = len(df.values[0])
        obs = th.zeros((num_row, num_col), dtype=dtype, device=device)
        for i in range(num_row):
            for j in range(num_col):
                if not isinstance(df.values[i][j], th.Tensor):
                    obs[i][j] = th.tensor(df.values[i][j], device=device, dtype=dtype)
                else:
                    obs[i][j] = df.values[i][j].clone()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        return obs