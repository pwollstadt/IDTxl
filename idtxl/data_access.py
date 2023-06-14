from dataclasses import dataclass
from typing import Optional

import numpy as np

Variable = tuple[int, int]

@dataclass
class DataAccessToken:
    """Small object that stores a current_value, variables and possibly surrogate creation information for accessing data."""
    current_value: Variable
    variables: list[Variable]
    surrogate_creation_info: Optional[dict] = None

    @classmethod
    def generate_surrogate_tokens(
        cls,
        current_value: Variable,
        variables: list[Variable],
        n_surrogates: int,
        surrogate_creation_info: dict
    ):
        """Generate a list of DataAccessToken objects for each variable."""
        
        # Generate random seeds for each surrogate
        surrogate_seeds = np.random.randint(0, 2**32, n_surrogates)

        # Generate surrogate tokens
        surrogate_tokens = []
        for seed in surrogate_seeds:
            info_dict = surrogate_creation_info.copy()
            info_dict['rng_seed'] = seed
            surrogate_tokens.append(cls(
                current_value,
                variables,
                info_dict
            ))

        return surrogate_tokens