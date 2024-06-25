from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from maspy.learning.space import Space
from maspy.learning.core import ActType, ObsType, RenderFrame
from maspy.utils import np_random

if TYPE_CHECKING:
    from maspy.learning.registration import EnvSpec
    
ArrayType = TypeVar("ArrayType")

__all__ = [
    "VectorEnv",
    "VectorWrapper",
    "VectorObservationWrapper",
    "VectorActionWrapper",
    "VectorRewardWrapper",
    "ArrayType",
]

class VectorEnv(Generic[ObsType, ActType, ArrayType]):
    
    metadata: dict[str, Any] = {}
    spec: EnvSpec | None = None
    render_mode: str | None = None
    closed: bool = False

    observation_space: Space
    action_space: Space
    single_observation_space: Space
    single_action_space: Space

    num_envs: int

    _np_random: np.random.Generator | None = None
    _np_random_seed: int | None = None
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None:
            self._np_random, np_random_seed = np_random(seed)
            
    def step(self, action: ActType) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """
        Args:
            action (ActType): An action provided by the policy.

        Returns:
            observation (ObsType): The agent's observation of the current environment."""