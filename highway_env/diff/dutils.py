import copy
import importlib
import itertools
from typing import Tuple, Dict, Callable, List, Optional, Union, Sequence

import torch as th

def d_not_zero(x: th.Tensor, eps: float = 1e-2) -> th.Tensor:
    non_zero_x = x.clone()
    pos_x_args = th.where(x >= 0)
    neg_x_args = th.where(x < 0)
    pos_x = x[pos_x_args]
    pos_x = th.clip(pos_x, eps, None)
    neg_x = x[neg_x_args]
    neg_x = th.clip(neg_x, None, -eps)
    non_zero_x[pos_x_args] = pos_x
    non_zero_x[neg_x_args] = neg_x
    return non_zero_x

def d_wrap_to_pi(x: th.Tensor) -> th.Tensor:
    return ((x + th.pi) % (2 * th.pi)) - th.pi
