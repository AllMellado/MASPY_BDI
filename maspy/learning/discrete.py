from typing import Any

import numpy as np

from maspy.learning.space import MaskNDArray, Space

class Discrete(Space[np.int64]):
    def __init__(
            self, 
            n: int | np.integer[Any], 
            seed: int | np.random.Generator | None = None,
            start: int | np.integer[Any] = 0 
        ):
        assert np.issubdtype(type(n), np.integer), f"Expected integer type, actual type: {type(n)}"
        assert n > 0, "n (counts) have to be positive"
        assert np.issubdtype(type(start), np.integer), f"Expected integer type, actual type: {type(start)}"
        
        self.n = np.int64(n)
        self.start = np.int64(start)
        super().__init__((), np.int64, seed)
    
    def sample(self, mask: MaskNDArray | None = None) -> np.int64:
        if mask is not None:
            assert isinstance(mask, np.ndarray
            ), f"The expected type of the mask is np.ndarray, actual type: {type(mask)}"
            
            assert (mask.dtype == np.int8
            ), f"The expected dtype of the mask is np.int8, actual dtype: {mask.dtype}"
            
            assert mask.shape == (self.n,
            ), f"The expected shape of the mask is {(self.n,)}, actual shape: {mask.shape}"
            
            valid_action_mask = mask == 1
            
            assert np.all(np.logical_or(mask == 0, valid_action_mask)
            ), f"All values of a mask should be 0 or 1, actual values: {mask}"
            
            if np.any(valid_action_mask):
                return self.start + self.np_random.choice(np.where(valid_action_mask)[0])
            else:
                return self.start
            
        return np.int64(self.start + self.np_random.integers(self.n))
    
    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, int):
            as_int64 = np.int64(x)
        elif isinstance(x, (np.generic, np.ndarray)) and (
            np.issubdtype(x.dtype, np.integer) and x.shape == ()
        ):
            as_int64 = np.int64(x.item())
        else:
            return False

        return bool(self.start <= as_int64 < self.start + self.n)
    
    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        if self.start != 0:
            return f"Discrete({self.n}, start={self.start})"
        return f"Discrete({self.n})"

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, Discrete)
            and self.n == other.n
            and self.start == other.start
        )

