from highway_env.vehicle.behavior import IDMVehicle
import numpy as np
import logging
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional

from highway_env.vehicle.objects import Landmark
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

if TYPE_CHECKING:
    from highway_env.vehicle import kinematics, objects

import torch as th
from highway_env.vehicle.diff.dbehavior import dIDM
from highway_env.vehicle.diff.dkinematics import dVehicle

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]

class dRoad(Road):
    """This implementation augments [Road] with pytorch Tensor, to support differentiation and parallelization"""
    def __init__(self,
                 network: RoadNetwork = None,
                 vehicles: List['kinematics.Vehicle'] = None,
                 road_objects: List['objects.RoadObject'] = None,
                 np_random: np.random.RandomState = None,
                 record_history: bool = False,
                 torch_device: str = None,
                 torch_dtype = None) -> None:
        super().__init__(network, vehicles, road_objects, np_random, record_history)
        self.init_torch_vars(torch_device, torch_dtype)
        if vehicles is not None:
            self.clear_torch_tensors()
        self.actions: th.Tensor = None

    def init_torch_vars(self, 
                        torch_device: str = None,
                        torch_dtype = None):
        if torch_device is not None:
            self.torch_device = torch_device
        else:
            if th.cuda.is_available():
                self.torch_device = 'cuda'
            else:
                self.torch_device = 'cpu'
        if torch_dtype is not None:
            self.torch_dtype = torch_dtype
        else:
            # Follow data type of vehicle position
            if len(self.vehicles) > 0 and self.vehicles[0].position.dtype is np.dtype('float64'):
                self.torch_dtype = th.float64
            else:
                self.torch_dtype = th.float32

    def clear_torch_tensors(self):
        """
        Store vehicle information in a Tensors, which are stored in our device and updated.
        """
        # @TODO
        for vehicle in self.vehicles:
            if not (isinstance(vehicle, IDMVehicle) or isinstance(vehicle, Vehicle)):
                assert True, "Only IDM vehicle is allowed now"
        
        assert len(self.vehicles) > 0, ""
        
        # Init torch vars again for safety
        self.init_torch_vars()

        num_vehicles = len(self.vehicles)

        # [RoadObject] (at [objects.py]) parameters
        position_size = self.vehicles[0].position.shape
        self.road_object_position = th.zeros((num_vehicles,) + position_size, device=self.torch_device, dtype=self.torch_dtype)
        self.road_object_heading = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)
        self.road_object_speed = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)

        # [Vehicle] (at [kinematics.py]) parameters
        self.vehicle_max_speed = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)
        self.vehicle_min_speed = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)
        self.vehicle_length = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)

        # [ControlledVehicle] (at [controller.py]) parameters
        self.controlled_target_speed = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)
        self.controlled_tau_pursuit = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)
        self.controlled_kp_lateral = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)
        self.controlled_kp_heading = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)
        self.controlled_max_steering_angle = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)

        # [IDMVehicle] (at [behavior.py]) parameters
        self.idm_acc_max = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)
        self.idm_comfort_acc_max = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)
        self.idm_comfort_acc_min = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)
        self.idm_distance_wanted = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)
        self.idm_time_wanted = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)
        self.idm_delta = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)
        self.idm_politeness = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)
        self.idm_lane_change_min_acc_gain = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)
        self.idm_lane_change_max_breaking_imposed = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)
        self.idm_lane_change_delay = th.zeros((num_vehicles, 1), device=self.torch_device, dtype=self.torch_dtype)

        # Copy information from [self.vehicles] to tensors
        for i, vehicle in enumerate(self.vehicles):
            self.road_object_position[i] = th.tensor(vehicle.position)
            self.road_object_heading[i] = th.tensor(vehicle.heading)
            self.road_object_speed[i] = th.tensor(vehicle.speed)

            self.vehicle_max_speed[i] = th.tensor(vehicle.MAX_SPEED)
            self.vehicle_min_speed[i] = th.tensor(vehicle.MIN_SPEED)
            self.vehicle_length[i] = th.tensor(vehicle.LENGTH)

            if not isinstance(vehicle, IDMVehicle):
                continue

            idmv: IDMVehicle = vehicle
            
            self.controlled_target_speed[i] = th.tensor(idmv.target_speed)
            self.controlled_tau_pursuit[i] = th.tensor(idmv.TAU_PURSUIT)
            self.controlled_kp_lateral[i] = th.tensor(idmv.KP_LATERAL)
            self.controlled_kp_heading[i] = th.tensor(idmv.KP_HEADING)
            self.controlled_max_steering_angle[i] = th.tensor(idmv.MAX_STEERING_ANGLE)

            self.idm_acc_max[i] = th.tensor(idmv.ACC_MAX)
            self.idm_comfort_acc_max[i] = th.tensor(idmv.COMFORT_ACC_MAX)
            self.idm_comfort_acc_min[i] = th.tensor(idmv.COMFORT_ACC_MIN)
            self.idm_distance_wanted[i] = th.tensor(idmv.DISTANCE_WANTED)
            self.idm_time_wanted[i] = th.tensor(idmv.TIME_WANTED)
            self.idm_delta[i] = th.tensor(idmv.DELTA)
            self.idm_politeness[i] = th.tensor(idmv.POLITENESS)
            self.idm_lane_change_min_acc_gain = th.tensor(idmv.LANE_CHANGE_MIN_ACC_GAIN)
            self.idm_lane_change_max_breaking_imposed = th.tensor(idmv.LANE_CHANGE_MAX_BRAKING_IMPOSED)
            self.idm_lane_change_delay = th.tensor(idmv.LANE_CHANGE_DELAY)

    def vehicle_ids(self) -> Dict[str, List[int]]:
        dict = {}
        idm_list = []
        ego_list = []
        for i, vehicle in enumerate(self.vehicles):
            if isinstance(vehicle, IDMVehicle):
                idm_list.append(i)
            elif isinstance(vehicle, Vehicle):
                ego_list.append(i)
            else:
                assert True, ""
        dict['idm'] = idm_list
        dict['ego'] = ego_list
        return dict

    def vehicle_act(self, vehicle: Vehicle, action):
        vehicle.action = action

    def act(self) -> None:
        """
        Decide action of every vehicle in parallelized way 
        """
        # [0]: steering, [1]: acceleration
        num_vehicles = len(self.vehicles)
        self.actions = th.zeros((num_vehicles, 2), device=self.torch_device, dtype=self.torch_dtype)

        v_ids = self.vehicle_ids()

        # Ego vehicles
        egos = v_ids['ego']
        for ego in egos:
            vehicle: Vehicle = self.vehicles[ego]
            self.actions[ego][0] = vehicle.action['steering'].clone()
            self.actions[ego][1] = vehicle.action['acceleration'].clone()

        # IDM vehicles
        idms = v_ids['idm']

        """
        ============= Non-parallelized block; Lane change decisions are left to be non-differentiable
        """
        # 1. Check if crashed and change lane depending on policy
        not_crashed_idms = []
        for idm in idms:
            vehicle: IDMVehicle = self.vehicles[idm]
            if vehicle.crashed:
                continue
            not_crashed_idms.append(idm)

            vehicle.follow_road()
            if vehicle.enable_lane_change:
                vehicle.change_lane_policy()

        """
        ============= Parallelized block
        """

        # Compute steering (lateral movement)
        idm_steering = dIDM.steering_control(self.vehicles,
                                                self.road_object_position,
                                                self.road_object_speed,
                                                self.road_object_heading,
                                                self.vehicle_length,
                                                self.controlled_tau_pursuit,
                                                self.controlled_kp_lateral,
                                                self.controlled_kp_heading,
                                                self.controlled_max_steering_angle,
                                                self.torch_device,
                                                self.torch_dtype)

        # Compute accleration (longitudinal movement)
        idm_acceleration = dIDM.acceleration(self.vehicles,
                                            self.road_object_position,
                                            self.road_object_heading,
                                            self.road_object_speed,
                                            self.controlled_target_speed,
                                            self.idm_comfort_acc_max,
                                            self.idm_comfort_acc_min,
                                            self.idm_delta,
                                            self.idm_distance_wanted,
                                            self.idm_time_wanted,
                                            self.torch_device,
                                            self.torch_dtype)

        self.actions[idms, 0] = idm_steering[idms, 0]
        self.actions[idms, 1] = idm_acceleration[idms, 0]

        # Act
        for i in range(num_vehicles):
            vehicle: Vehicle = self.vehicles[i]
            action = {}
            action['steering'] = self.actions[i][0].detach().item()
            action['acceleration'] = self.actions[i][1].detach().item()
            self.vehicle_act(vehicle, action)

    def step(self, dt: float):
        """
        Step the dynamics of each entity on the road.

        :param dt: timestep
        """
        dVehicle.step(self.vehicles,
                        self.road_object_position,
                        self.road_object_speed,
                        self.road_object_heading,
                        self.vehicle_length,
                        self.actions,
                        self.vehicle_max_speed,
                        self.vehicle_min_speed,
                        dt)
        self.actions = None

        for i, vehicle in enumerate(self.vehicles):
            for other in self.vehicles[i+1:]:
                vehicle.handle_collisions(other, dt)
            for other in self.objects:
                vehicle.handle_collisions(other, dt)
        

    def neighbour_vehicles(self, vehicle: 'kinematics.Vehicle', lane_index: LaneIndex = None) \
            -> Tuple[Optional[int], Optional[int]]:
        """
        Find the preceding and following vehicles of a given vehicle.

        :param vehicle: the vehicle whose neighbours must be found
        :param lane_index: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)
        s = self.network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        # Road objects are excluded, we assume they are not on the road
        for i, v in enumerate(self.vehicles): # + self.objects:
            if v is not vehicle and not isinstance(v, Landmark):
                # Check if the current vehicle is close enough to the designated lane
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v, margin=1):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = i
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = i
        return v_front, v_rear