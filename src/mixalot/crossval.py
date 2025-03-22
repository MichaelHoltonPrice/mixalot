"""Code for cross-validation functionality."""
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import KFold


class CrossValidationFoldsSpec:
    """Specification for cross-validation folds."""
    
    def __init__(
        self,
        n_splits: int,
        random_state: int,
    ):
        """Initialize a cross-validation folds specification.

        The specification yields determinstic outcomes by always calling
        KFold with a required random_state.

        Args:
            n_splits: Number of folds.
            random_state: Seed for reproducibility.
        """
        self.n_splits = n_splits
        self.random_state = random_state
        
        self._validate()
    
    def _validate(self):
        """Validate the specification."""
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
    
    def create_folds(self, n_samples: int) -> List[Tuple[np.ndarray,
                                                         np.ndarray]]:
        """Create all folds based on the specification.
        
        Args:
            n_samples: The number of samples to split into folds.
            
        Returns:
            List of (train_indices, test_indices) tuples.
        """

        indices = np.arange(n_samples)

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        return [(train_idx, test_idx) for train_idx,
                test_idx in kf.split(indices)]

    def get_fold_indices(self, fold_idx: int,
                         n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get train and test indices for a specific fold.
        
        Args:
            fold_idx: The index of the fold to retrieve (0-based).
            n_samples: The number of samples to split into folds.
            
        Returns:
            Tuple of (train_indices, test_indices) for the specified fold.
            
        Raises:
            ValueError: If fold_idx is out of range for the specified n_splits.
        """
        if fold_idx < 0 or fold_idx >= self.n_splits:
            raise ValueError(
                f"fold_idx must be between 0 and {self.n_splits - 1}, "
                f"got {fold_idx}"
            )
        
        folds = self.create_folds(n_samples)
        return folds[fold_idx]
