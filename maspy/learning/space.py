from typing import Generic, TypeVar, Sequence

import numpy as np
import numpy.typing as npt

from maspy.utils import np_random

Cov_Type = TypeVar("Cov_Type", covariant=True)
RNG = RandomNumberGenerator = np.random.Generator
MaskNDArray = npt.NDArray[np.int8]

class Space(Generic[Cov_Type]):
    def __init__(
            self, 
            shape: Sequence[int] | None = None, 
            dtype: npt.DTypeLike | None = None, 
            seed: int | np.random.Generator | None = None
        ):
        self._shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)
        
        self._np_random = None
        if seed is not None:
            if isinstance(seed, np.random.Generator):
                self._np_random = seed
            else:
                self.seed(seed)
        
    def seed(self, seed: int | None = None) -> int | list[int] | dict[str, int]:
        self._np_random, np_random_seed = np_random(seed)
        return np_random_seed

    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self.seed()

        if self._np_random is None:
            self._np_random, _ = np_random()

        return self._np_random
    
    @property
    def shape(self) -> tuple[int, ...] | None:
        return self._shape
    

