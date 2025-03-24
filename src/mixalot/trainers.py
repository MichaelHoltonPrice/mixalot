"""Training implementation for Random Forest models."""
from __future__ import annotations
from typing import Optional, Union

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import torch

from mixalot.datasets import MixedDataset
from mixalot.models import RandomForestSpec

def train_random_forest(
    dataset: MixedDataset,
    model_spec: RandomForestSpec,
    random_seed: Optional[int] = None
) -> Union[RandomForestClassifier, RandomForestRegressor]:
    """Train a Random Forest model using the provided dataset and spec.

    This function extracts the training data from the MixedDataset based on
    the model specification and trains a Random Forest model.

    Args:
        dataset: The MixedDataset containing the data to train on.
        model_spec: The RandomForestSpec containing model configuration.
        random_seed: Optional seed for reproducibility.

    Returns:
        The trained Random Forest model

    Raises:
        ValueError: If the target variable type is not supported or if
            required data is missing.
    """
    # Check for missing data in the dataset and raise an error if found
    if hasattr(dataset, 'Xcat_mask') and dataset.Xcat_mask is not None:
        if torch.any(dataset.Xcat_mask):
            raise ValueError("Missing values detected in categorical variables")
    if hasattr(dataset, 'Xord_mask') and dataset.Xord_mask is not None:
        if torch.any(dataset.Xord_mask):
            raise ValueError("Missing values detected in ordinal variables")
    if hasattr(dataset, 'Xnum_mask') and dataset.Xnum_mask is not None:
        if torch.any(dataset.Xnum_mask):
            raise ValueError("Missing values detected in numerical variables")

    # Extract data arrays
    Xcat, Xord, Xnum, y_data = dataset.get_arrays()

    if y_data is None:
        raise ValueError("No target variable found in dataset")

    # Get dataset specification and target variable information
    dataset_spec = dataset.dataset_spec
    y_var = model_spec.y_var
    y_var_spec = dataset_spec.get_var_spec(y_var)
    y_var_type = y_var_spec.var_type

    # Create a combined feature tensor and convert to numpy
    # This approach assumes that dataset already contains the properly 
    # formatted features (similar to AAFSDataset.X)
    X_combined = torch.cat(
        [tensor.float() for tensor in (Xcat, Xord, Xnum) 
         if tensor is not None], 
        dim=1
    )
    X = X_combined.cpu().numpy()
    y = y_data.cpu().numpy()

    # Extract hyperparameters
    n_estimators = model_spec.hyperparameters['n_estimators']

    max_features_config = model_spec.hyperparameters['max_features']
    max_features_type = max_features_config['type']
    max_features_value = max_features_config['value']

    max_features = (
        max_features_value if max_features_type in ['int', 'float']
        else 'auto'  # Default fallback
    )

    # Create the appropriate model based on target variable type
    if y_var_type in ['categorical', 'ordinal']:
        # For classification tasks
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_seed
        )
    elif y_var_type == 'numerical':
        # For regression tasks
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_seed
        )
    else:
        raise ValueError(f"Unsupported target variable type: {y_var_type}")

    # Train the model
    model.fit(X, y)

    return model
