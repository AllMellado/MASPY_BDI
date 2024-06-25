from typing import Any, SupportsFloat, Sequence, Iterable

import numpy as np
from numpy.typing import NDArray

from maspy.learning.space import Space

def is_float_integer(var: Any) -> bool:
    """Checks if a scalar variable is an integer or float (does not include bool)."""
    return np.issubdtype(type(var), np.integer) or np.issubdtype(type(var), np.floating)

class Box(Space[NDArray[Any]]):
    def __init__(
        self,
        low: SupportsFloat | NDArray[Any],
        high: SupportsFloat | NDArray[Any],
        shape: Sequence[int] | None = None,
        dtype: type[np.floating[Any]] | type[np.integer[Any]] = np.float32,
        seed: int | np.random.Generator | None = None,
    ):
        if dtype is None:
            raise ValueError("Box dtype cannot be None")
        self.dtype = np.dtype(dtype)
        
        if not (
            np.issubdtype(self.dtype, np.floating) 
            or np.issubdtype(self.dtype, np.integer)
            or self.dtype == np.bool_
        ):
            raise ValueError(f"Invalid Box dtype ({self.dtype}). Must be floating, integer or bool")
        
        if shape is not None:
            if not isinstance(shape, Iterable):
                raise TypeError(f"Box shape must be an iterable, actual type: {type(shape)}")
            elif not all(np.issubdtype(type(dim), np.integer) for dim in shape):
                raise TypeError(f"All Box shape elements must be integers, actual type: {type(shape)}")
            shape = tuple(int(dim) for dim in shape)
        elif isinstance(low, np.ndarray) and isinstance(high, np.ndarray):
            if low.shape != high.shape:
                raise ValueError(
                    f"Box low.shape and high.shape don't match, low.shape={low.shape}, high.shape={high.shape}"
                )
            shape = low.shape
        elif isinstance(low, np.ndarray):
            shape = low.shape
        elif isinstance(high, np.ndarray):
            shape = high.shape
        elif is_float_integer(low) and is_float_integer(high):
            shape = (1,)  # low and high are scalars
        else:
            raise ValueError(
                "Box shape is not specified, therefore inferred from low and high. Expected low and high to be np.ndarray, integer, or float."
                f"Actual types low={type(low)}, high={type(high)}"
            )
        self._shape: tuple[int, ...] = shape

        dtype_min: int | float
        dtype_max: int | float
        if self.dtype == np.bool_:
            dtype_min, dtype_max = 0, 1
        elif np.issubdtype(self.dtype, np.floating): 
            dtype_min = float(np.finfo(self.dtype).min)
            dtype_max = float(np.finfo(self.dtype).max)
        elif np.issubdtype(self.dtype, np.integer):
            dtype_min = int(np.iinfo(self.dtype).min)
            dtype_max = int(np.iinfo(self.dtype).max)
        else:
            raise TypeError(f'Unsupported dtype: {self.dtype}')

        self.low, self.bounded_below = self._cast_low(low, dtype_min)
        self.high, self.bounded_above = self._cast_high(high, dtype_max)

        if self.low.shape != shape:
            raise ValueError(
                f"Box low.shape doesn't match provided shape, low.shape={self.low.shape}, shape={self.shape}"
            )
        if self.high.shape != shape:
            raise ValueError(
                f"Box high.shape doesn't match provided shape, high.shape={self.high.shape}, shape={self.shape}"
            )

        # check that low <= high
        if np.any(self.low > self.high):
            raise ValueError(
                f"Box all low values must be less than or equal to high (some values break this), low={self.low}, high={self.high}"
            )

        self.low_repr = str(self.low)
        self.high_repr = str(self.high)

        super().__init__(self.shape, self.dtype, seed)
    
        
    def _cast_low(self, low, dtype_min) -> tuple[np.ndarray, np.ndarray]:
        assert self.dtype is not None, 'Box dtype cannot be None'
        if is_float_integer(low):
            bounded_below = -np.inf < np.full(self.shape, low, dtype=float)

            if np.isnan(low):
                raise ValueError(f"No low value can be equal to `np.nan`, low={low}")
            elif np.isneginf(low):
                if self.dtype.kind == "i":  # signed int
                    low = dtype_min
                elif self.dtype.kind in {"u", "b"}:  # unsigned int and bool
                    raise ValueError(
                        f"Box unsigned int dtype don't support `-np.inf`, low={low}"
                    )
            elif low < dtype_min:
                raise ValueError(
                    f"Box low is out of bounds of the dtype range, low={low}, min dtype={dtype_min}"
                )

            low = np.full(self.shape, low, dtype=self.dtype)
            return low, bounded_below
        else:  # cast for low - array
            if not isinstance(low, np.ndarray):
                raise ValueError(
                    f"Box low must be a np.ndarray, integer, or float, actual type={type(low)}"
                )
            elif not (
                np.issubdtype(low.dtype, np.floating)
                or np.issubdtype(low.dtype, np.integer)
                or low.dtype == np.bool_
            ):
                raise ValueError(
                    f"Box low must be a floating, integer, or bool dtype, actual dtype={low.dtype}"
                )
            elif np.any(np.isnan(low)):
                raise ValueError(f"No low value can be equal to `np.nan`, low={low}")

            bounded_below = -np.inf < low

            if np.any(np.isneginf(low)):
                if self.dtype.kind == "i":  # signed int
                    low[np.isneginf(low)] = dtype_min
                elif self.dtype.kind in {"u", "b"}:  # unsigned int and bool
                    raise ValueError(
                        f"Box unsigned int dtype don't support `-np.inf`, low={low}"
                    )
            elif low.dtype != self.dtype and np.any(low < dtype_min):
                raise ValueError(
                    f"Box low is out of bounds of the dtype range, low={low}, min dtype={dtype_min}"
                )
            
            
            if (
                np.issubdtype(low.dtype, np.floating)
                and np.issubdtype(self.dtype, np.floating)
                and np.finfo(self.dtype).precision < np.finfo(low.dtype).precision
            ):    
                print(
                    f"Box low's precision lowered by casting to {self.dtype}, current low.dtype={low.dtype}"
                )
                
            return low.astype(self.dtype), bounded_below

    def _cast_high(self, high, dtype_max) -> tuple[np.ndarray, np.ndarray]:
        assert self.dtype is not None, 'Box dtype cannot be None'
        if is_float_integer(high):
            bounded_above = np.full(self.shape, high, dtype=float) < np.inf

            if np.isnan(high):
                raise ValueError(f"No high value can be equal to `np.nan`, high={high}")
            elif np.isposinf(high):
                if self.dtype.kind == "i":  # signed int
                    high = dtype_max
                elif self.dtype.kind in {"u", "b"}:  # unsigned int
                    raise ValueError(
                        f"Box unsigned int dtype don't support `np.inf`, high={high}"
                    )
            elif high > dtype_max:
                raise ValueError(
                    f"Box high is out of bounds of the dtype range, high={high}, max dtype={dtype_max}"
                )

            high = np.full(self.shape, high, dtype=self.dtype)
            return high, bounded_above
        else:
            if not isinstance(high, np.ndarray):
                raise ValueError(
                    f"Box high must be a np.ndarray, integer, or float, actual type={type(high)}"
                )
            elif not (
                np.issubdtype(high.dtype, np.floating)
                or np.issubdtype(high.dtype, np.integer)
                or high.dtype == np.bool_
            ):
                raise ValueError(
                    f"Box high must be a floating or integer dtype, actual dtype={high.dtype}"
                )
            elif np.any(np.isnan(high)):
                raise ValueError(f"No high value can be equal to `np.nan`, high={high}")

            bounded_above = high < np.inf

            posinf = np.isposinf(high)
            if np.any(posinf):
                if self.dtype.kind == "i":  # signed int
                    high[posinf] = dtype_max
                elif self.dtype.kind in {"u", "b"}:  # unsigned int
                    raise ValueError(
                        f"Box unsigned int dtype don't support `np.inf`, high={high}"
                    )
            elif high.dtype != self.dtype and np.any(dtype_max < high):
                raise ValueError(
                    f"Box high is out of bounds of the dtype range, high={high}, max dtype={dtype_max}"
                )
            
            assert not np.issubdtype(self.dtype, np.void)
            if (
                np.issubdtype(high.dtype, np.floating)
                and np.issubdtype(self.dtype, np.floating)
                and np.finfo(self.dtype).precision < np.finfo(high.dtype).precision
            ):
                print(
                    f"Box high's precision lowered by casting to {self.dtype}, current high.dtype={high.dtype}"
                )
            
            return high.astype(self.dtype), bounded_above

    @property
    def shape(self) -> tuple[int, ...]:
        """Has stricter type than gym.Space - never None."""
        return self._shape

    def is_bounded(self, manner: str = "both") -> bool:
        below = bool(np.all(self.bounded_below))
        above = bool(np.all(self.bounded_above))
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError(
                f"manner is not in {{'below', 'above', 'both'}}, actual value: {manner}"
            )

    def sample(self, mask: None = None) -> NDArray[Any]:
        assert self.dtype is not None, 'Box dtype cannot be None'
        if mask is not None:
            raise ValueError(
                f"Box.sample cannot be provided a mask, actual value: {mask}"
            )

        high = self.high if self.dtype.kind == "f" else self.high.astype("int64") + 1
        sample = np.empty(self.shape)

        # Masking arrays which classify the coordinates according to interval type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        sample[unbounded] = self.np_random.normal(size=unbounded[unbounded].shape)

        sample[low_bounded] = (
            self.np_random.exponential(size=low_bounded[low_bounded].shape)
            + self.low[low_bounded]
        )

        sample[upp_bounded] = (
            -self.np_random.exponential(size=upp_bounded[upp_bounded].shape)
            + high[upp_bounded]
        )

        sample[bounded] = self.np_random.uniform(
            low=self.low[bounded], high=high[bounded], size=bounded[bounded].shape
        )

        if self.dtype.kind in ["i", "u", "b"]:
            sample = np.floor(sample)

        # clip values that would underflow/overflow
        if np.issubdtype(self.dtype, np.signedinteger):
            dtype_min = np.iinfo(self.dtype).min + 2
            dtype_max = np.iinfo(self.dtype).max - 2
            sample = sample.clip(min=dtype_min, max=dtype_max)
        elif np.issubdtype(self.dtype, np.unsignedinteger):
            dtype_min = np.iinfo(self.dtype).min
            dtype_max = np.iinfo(self.dtype).max
            sample = sample.clip(min=dtype_min, max=dtype_max)

        sample = sample.astype(self.dtype)

        # float64 values have lower than integer precision near int64 min/max, so clip
        # again in case something has been cast to an out-of-bounds value
        if self.dtype == np.int64:
            sample = sample.clip(min=self.low, max=self.high)

        return sample

    def contains(self, x: Any) -> bool:
        if not isinstance(x, np.ndarray):
            #logger.warn("Casting input x to numpy array.")
            try:
                x = np.asarray(x, dtype=self.dtype)
            except (ValueError, TypeError):
                return False

        return bool(
            np.can_cast(x.dtype, self.dtype)
            and x.shape == self.shape
            and np.all(x >= self.low)
            and np.all(x <= self.high)
        )
    
    def __repr__(self) -> str:
        return f"Box({self.low_repr}, {self.high_repr}, {self.shape}, {self.dtype})"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Box)
            and (self.shape == other.shape)
            and (self.dtype == other.dtype)
            and np.allclose(self.low, other.low)
            and np.allclose(self.high, other.high)
        )
