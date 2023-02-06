from __future__ import annotations
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike


class Process():
    """
    A Process object represents a single process.
    The dimensionality of the process is determined by the first axis of the data.
    """

    def __init__(self, data: ArrayLike, properties: Optional[dict[str, Any]] = None) -> None:

        self._data = np.asanyarray(data)
        self._properties = properties or {}

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape[1:]

    @property
    def properties(self) -> dict[str, Any]:
        return self._properties

    def __len__(self) -> int:
        return self._data.shape[1]

    def __getitem__(self, key) -> Process:
        """
        Returns a new Process object with the data sliced by the given key.
        Does not modify the first (dimensionality) axis.
        """

        if isinstance(key, tuple):
            full_key = (slice(None), *key)
        else:
            full_key = (slice(None), key)

        return Process(self._data[full_key], properties=self._properties)

    def __repr__(self) -> str:
        return f"Process({self._data!r},\n\tproperties={self._properties!r})"


class ProcessSet():
    """
    Represents an indexed set of processes.
    Can be indexed like a numpy array where the first axis is interpreted as the process axis.
    """

    def __init__(self, processes: ArrayLike) -> None:
        self._set_processes(processes)

    def _set_processes(self, processes: ArrayLike) -> None:

        # Ensure all processes have the same number of samples and replications
        assert len(set(p.shape[1:] for p in processes)) == 1,\
            "All processes must have the same shape except for axis 0."

        self._processes = np.empty(len(processes), dtype=Process)
        self._processes[:] = processes

    @staticmethod
    def from_ndarray(data: ArrayLike, process_dimensions: tuple[int] = None, properties: tuple[dict[str, Any]] = None) -> ProcessSet:
        """
        Creates a ProcessSet from a single array of data.
        The first axis of the data is assumed to be the process axis.
        """

        assert min(process_dimensions) > 0,\
            "All dimensions must be greater than 0."
        assert sum(process_dimensions) == data.shape[0],\
            "The sum of the dimensions must equal the number of columns in the data."
        assert properties is None or len(process_dimensions) == len(properties),\
            "The number of dimensions must equal the number of properties."

        cum_dimensions = np.cumsum(process_dimensions)
        data_splits = np.split(data, cum_dimensions[:-1])

        properties = properties or [{}] * len(process_dimensions)

        processes = [Process(d, p) for d, p in zip(data_splits, properties)]

        return ProcessSet(processes)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._processes), *self._processes[0].shape)

    def get_property(self, property_name: str) -> np.ndarray:
        """
        Returns an array of the given property for each process.
        """

        return np.array([p.properties[property_name] for p in self._processes])
    
    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._processes), *self._processes[0].shape)

    def __len__(self) -> int:
        return len(self._processes)

    def __getitem__(self, key) -> Union[Process, ProcessSet]:
        """
        So far this only supports indexing by indices or slices.
        """
        if isinstance(key, tuple):

            if key[0] is Ellipsis:
                process_key = slice(None)
                rest_key = key
            else:
                process_key = key[0]
                rest_key = key[1:]
        else:
            process_key = key
            rest_key = slice(None)

        # If process_key is a single index, return a single Process
        if isinstance(process_key, int):
            return self._processes[process_key][rest_key]

        # Otherwise, return a ProcessSet
        processes = [p[rest_key] for p in self._processes[process_key]]
        return ProcessSet(processes)

    def __repr__(self) -> str:
        return f"ProcessSet({self._processes!r})"
