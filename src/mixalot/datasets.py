"""Dataset specification classes."""
from __future__ import annotations
import json
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class VarSpec:
    """Represents the specifications of a variable in a dataset.

    Each VarSpec object captures the variable name, its type (numerical,
    categorical, or ordinal), a mapping for categorical variables (if
    applicable), any missing values, and the order of the variable.

    Args:
        var_name (str): Name of the variable.
        var_type (str): Type of the variable. Must be 'numerical',
            'categorical', or 'ordinal'.
        categorical_mapping (list[set], optional): List of sets representing
            the categories of a categorical or ordinal variable. Each set
            includes strings that map to the respective category. 
            This parameter is required if var_type is 'categorical' or
            'ordinal'.
        missing_values (Any, optional): Specifies the representation of missing
            values in the variable. Default is None.
        column_name (str, optional): Specifies a distinct column name in an
            input file (by default var_name is assumed)

    Raises:
        ValueError: If var_type is not 'numerical', 'categorical', or
            'ordinal'.
        ValueError: If var_type is 'categorical' or 'ordinal' but no
            categorical_mapping is provided.
        ValueError: If the same string appears in more than one set of
            categorical_mapping.
    """
 
    def __init__(self,
                 var_name,
                 var_type,
                 categorical_mapping=None,
                 missing_values=None,
                 column_name=None):
        """Initialize a VarSpec object."""
        self.var_name = var_name
        self.var_type = var_type.lower()
        self.categorical_mapping = categorical_mapping
        self.missing_values = missing_values
        self.column_name = column_name
        self._validate_spec()

    def _validate_spec(self):
        """Validate the variable specification."""
        if self.var_type not in ['numerical', 'categorical', 'ordinal']:
            raise ValueError(
                f"Invalid 'type' field for variable {self.var_name}. "
                "Expected 'numerical', 'categorical', or 'ordinal'"
            )
        if (self.var_type in ['categorical', 'ordinal']
            and self.categorical_mapping is None):
            raise ValueError(
                "Missing 'categorical_mapping' field for variable "
                f"{self.var_name} of type {self.var_type}"
            )
        if self.var_type in ['categorical', 'ordinal']:
            self._validate_categorical_mapping()

    def _validate_categorical_mapping(self):
        """Validate mapping for a categorical (or ordinal) variable."""
        if not all(isinstance(cat_set, set)
                   for cat_set in self.categorical_mapping):
            raise ValueError(
                "All entries in 'categorical_mapping' should be sets"
            )

        all_values = [value for cat_set in self.categorical_mapping
                      for value in cat_set]
        if len(all_values) != len(set(all_values)):
            raise ValueError(
                "Some values appear in more than one set for "
                f"variable {self.var_name}"
            )


class DatasetSpec:
    """Class to specify a dataset's variables and their characteristics.
    
    This class maintains ordered lists of categorical, ordinal, and numerical 
    variables, along with their specifications. It provides methods to validate
    the specifications and retrieve variable information.
    
    Args:
        cat_var_specs: List of VarSpecs for categorical variables
        ord_var_specs: List of VarSpecs for ordinal variables
        num_var_specs: List of VarSpecs for numerical variables
    """
    def __init__(self,
                 cat_var_specs: List[VarSpec],
                 ord_var_specs: List[VarSpec],
                 num_var_specs: List[VarSpec]):
        """Initializes a DatasetSpec object."""
        self.cat_var_specs = cat_var_specs
        self.ord_var_specs = ord_var_specs
        self.num_var_specs = num_var_specs
        self._validate_specs()

    def _validate_specs(self):
        """Validates the variable specifications provided.
        
        Raises:
            ValueError: If any of the following conditions are met:
                1. All of cat_var_specs, ord_var_specs, and num_var_specs are 
                   empty
                2. Any variable name is repeated across different types
                3. The type of each variable in the respective var_specs list
                   is not as expected
        """
        if (not self.cat_var_specs and not self.ord_var_specs
            and not self.num_var_specs):
            raise ValueError("At least one of cat_var_specs, ord_var_specs, "
                             "or num_var_specs must be non-empty")
    
        all_var_names = [var.var_name for var_list in 
                         [self.cat_var_specs, self.ord_var_specs, 
                          self.num_var_specs] for var in var_list]

        if len(all_var_names) != len(set(all_var_names)):
            raise ValueError("Variable names must be unique across all "
                             "variable types")
    
        for var_type, var_specs in [('categorical', self.cat_var_specs), 
                                    ('ordinal', self.ord_var_specs), 
                                    ('numerical', self.num_var_specs)]:
            if not all(isinstance(var_spec, VarSpec) and 
                       var_spec.var_type ==
                       var_type for var_spec in var_specs):
                raise ValueError(
                    f"All variable specifications in {var_type}_var_specs "
                    f"must be instances of VarSpec of type {var_type}")

    @property
    def all_var_names(self):
        """Returns the set of all variable names across all variable types."""
        return set(var.var_name for var_list in 
                  [self.cat_var_specs, self.ord_var_specs, self.num_var_specs] 
                  for var in var_list)

    def get_ordered_variables(self, var_type):
        """Returns a list of ordered variable names of a certain type.
        
        Args:
            var_type: A string representing the type of the variables to be 
                      returned. This should be one of 'numerical', 
                      'categorical', or 'ordinal'.
        
        Returns:
            ordered_vars: A list of variable names in the order they appear in 
                          the dataset.
        
        Raises:
            ValueError: If the var_type is not 'numerical', 'categorical', or 
                        'ordinal'.
        """
        var_specs = []
        if var_type == 'categorical':
            var_specs = self.cat_var_specs
        elif var_type == 'ordinal':
            var_specs = self.ord_var_specs
        elif var_type == 'numerical':
            var_specs = self.num_var_specs
        else:
            raise ValueError(f"Invalid 'type' field. Expected 'numerical', "
                             f"'categorical', or 'ordinal'")
        
        # get variable names in the order they are listed
        ordered_vars = [var_spec.var_name for var_spec in var_specs]
        return ordered_vars

    def get_var_spec(self, var_name):
        """Returns the VarSpec corresponding to the provided variable name.
        
        Args:
            var_name: The name of the variable for which the VarSpec is to be 
                      returned.
        
        Returns:
            var_spec: The VarSpec corresponding to the provided variable name.
        
        Raises:
            ValueError: If the var_name is not found in the variable 
                        specifications.
        """
        for var_type in ['cat', 'ord', 'num']:
            for var_spec in getattr(self, var_type + "_var_specs"):
                if var_spec.var_name == var_name:
                    return var_spec

        raise ValueError(f"Variable name {var_name} is not found in the "
                         f"provided variable specifications")

    @classmethod
    def from_json(cls, json_path):
        """Creates a DatasetSpec object from a JSON file.
        
        Args:
            json_file: Path to the JSON file containing the dataset 
                       specifications.
        
        Returns:
            DatasetSpec: A new DatasetSpec object created from the JSON file.
        """
        with open(json_path, 'r') as file:
            json_dict = json.load(file)

        cat_var_specs = [
            VarSpec(
                **{
                    **spec,
                    'categorical_mapping': [
                        set(cat_map) for cat_map in 
                        spec.get('categorical_mapping', [])
                    ]
                }
            )
            for spec in json_dict.get('cat_var_specs', [])
        ]
        
        ord_var_specs = [
            VarSpec(
                **{
                    **spec,
                    'categorical_mapping': [
                        set(cat_map) for cat_map in 
                        spec.get('categorical_mapping', [])
                    ]
                }
            )
            for spec in json_dict.get('ord_var_specs', [])
        ]

        num_var_specs = [VarSpec(**spec) 
                         for spec in json_dict.get('num_var_specs', [])]
        
        return cls(cat_var_specs, ord_var_specs, num_var_specs)


class MixedDataset(Dataset):
    """A dataset to hold and manage mixed variable types.

    Categorical, ordinal, and numerical variables are supported. The class 
    accommodates optional data augmentation and masking.

    Args:
        dataset_spec (DatasetSpec): Specification of the dataset.
        Xcat (np.ndarray, optional): Numpy array holding categorical data.
        Xord (np.ndarray, optional): Numpy array holding ordinal data.
        Xnum (np.ndarray, optional): Numpy array holding numerical data.
        model_spec (SingleTargetModelSpec, optional): Model specification with
            target variable. If provided, the y_var will be extracted from this 
            instead of from dataset_spec.
        mask_prob (float, optional): Probability that a given input is masked.
            Default is 0.
        aug_mult (int, optional): Multiplier for data augmentation. Default is
            1.
        device (torch.device or str, optional): The torch device object or
            string to specify the device to use. If None, prefer gpu.
        require_input (bool, optional): If True, ensures that at least one
            variable remains unmasked in every item. Default is False.

    Attributes:
        dataset_spec (DatasetSpec): Specification of the dataset.
        model_spec (SingleTargetModelSpec, optional): Model specification.
        Xcat (torch.Tensor): Categorical data.
        Xord (torch.Tensor): Ordinal data.
        Xnum (torch.Tensor): Numerical data.
        y_data (torch.Tensor): Y variable data if it exists.
        num_obs (int): Number of observations in the dataset.
        mask_prob (float): Probability that a given input is masked.
        aug_mult (int): Multiplier for data augmentation.
        require_input (bool): If True, ensures that at least one variable
            remains unmasked in every item.
    """
    def __init__(
            self, 
            dataset_spec: 'DatasetSpec',
            Xcat: Optional[np.ndarray] = None,
            Xord: Optional[np.ndarray] = None,
            Xnum: Optional[np.ndarray] = None,
            model_spec: Optional['SingleTargetModelSpec'] = None,
            mask_prob: float = 0,
            aug_mult: int = 1,
            require_input: bool = False,
            device=None
        ):
        """Initialize a MixedDataset object."""
        self.dataset_spec = dataset_spec
        self.model_spec = model_spec
        self.mask_prob = mask_prob
        self.aug_mult = aug_mult
        self.require_input = require_input
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        if Xcat is None and Xord is None and Xnum is None:
            raise ValueError("Xcat, Xord, and Xnum cannot all be None")

        # Convert tensors to numpy if they are tensors
        Xcat = Xcat.cpu().numpy() if isinstance(Xcat, torch.Tensor) else Xcat
        Xord = Xord.cpu().numpy() if isinstance(Xord, torch.Tensor) else Xord
        Xnum = Xnum.cpu().numpy() if isinstance(Xnum, torch.Tensor) else Xnum

        # Check that all input arrays have the same number of samples
        num_obs_list = {len(X) for X in [Xcat, Xord, Xnum] if X is not None}
        if len(num_obs_list) > 1:
            raise ValueError(
                "Input arrays do not have the same number of samples"
            )
        self.num_obs = num_obs_list.pop()

        # Ensure that each input has the correct number of columns
        for var_type, X, s in zip(
                ['categorical', 'ordinal', 'numerical'],
                [Xcat, Xord, Xnum],
                ['Xcat', 'Xord', 'Xnum']
            ):
            num_var_ds = len(dataset_spec.get_ordered_variables(var_type))
            if X is None:
                num_var_mat = 0
            else:
                num_var_mat = X.shape[1]

            if num_var_ds != num_var_mat:
                raise ValueError(
                    f'{s} has {num_var_mat} columns but dataset_spec has '
                    f'{num_var_ds} {var_type} variables'
                )

        # Extract y_data before converting to tensor and moving to device
        y_var = (model_spec.y_var if model_spec is not None
                 else None)
        
        if y_var is not None:
            if y_var in dataset_spec.get_ordered_variables('categorical'):
                y_var_type = 'categorical'
                y_idx = dataset_spec.get_ordered_variables(
                    'categorical'
                ).index(y_var)
                self.y_data = Xcat[:, y_idx].copy()
                if Xcat.shape[1] > 1:
                    Xcat = np.delete(Xcat, y_idx, axis=1) 
                else:
                    Xcat = None
            elif y_var in dataset_spec.get_ordered_variables('ordinal'):
                y_var_type = 'ordinal'
                y_idx = dataset_spec.get_ordered_variables(
                    'ordinal'
                ).index(y_var)
                self.y_data = Xord[:, y_idx].copy()
                if Xord.shape[1] > 1:
                    Xord = np.delete(Xord, y_idx, axis=1)
                else:
                    Xord = None
            elif y_var in dataset_spec.get_ordered_variables('numerical'):
                y_var_type = 'numerical'
                y_idx = dataset_spec.get_ordered_variables(
                    'numerical'
                ).index(y_var)
                self.y_data = Xnum[:, y_idx].copy()
                if Xnum.shape[1] > 1:
                    Xnum = np.delete(Xnum, y_idx, axis=1)
                else:
                    Xnum = None
            else:
                raise ValueError(
                    f"y_var '{y_var}' not found in any variable type in "
                    "dataset_spec"
                )
        else:
            self.y_data = None

        # Convert numpy arrays to tensors and move to device
        self.Xcat = torch.tensor(
            Xcat, device=self.device
        ).long() if Xcat is not None else None
        self.Xord = torch.tensor(
            Xord, device=self.device
        ).long() if Xord is not None else None
        self.Xnum = torch.tensor(
            Xnum, device=self.device
        ).float() if Xnum is not None else None

        # Convert y_data to tensor and move to device
        if self.y_data is not None:
            if y_var_type == 'numerical':
                self.y_data = torch.tensor(self.y_data,
                                           device=self.device).float()
            else:
                assert y_var_type in ['categorical', 'ordinal']
                self.y_data = torch.tensor(self.y_data,
                                           device=self.device).long()

        # Create masks as tensors and move to device
        if Xcat is not None:
            self.Xcat_mask = torch.tensor(
                Xcat == 0, dtype=torch.bool, device=self.device
            )
        else:
            self.Xcat_mask = None

        if Xord is not None:
            self.Xord_mask = torch.tensor(
                Xord == 0, dtype=torch.bool, device=self.device
            )
        else:
            self.Xord_mask = None

        if Xnum is not None:
            self.Xnum_mask = torch.tensor(
                np.isnan(Xnum), dtype=torch.bool, device=self.device
            )
        else:
            self.Xnum_mask = None

    def __len__(self):
        """Return the number of items in the dataset.
        
        Returns:
            int: Number of observations, accounting for data augmentation.
        """
        return self.num_obs * self.aug_mult

    def __getitem__(self, idx):
        """Retrieve a dataset item with optional masking.
        
        Given an index, retrieves the corresponding item from the MixedDataset.
        The item consists of corresponding elements from Xcat, Xord, Xnum,
        a mask for numerical variables, and y_data (if they exist). 
        
        Incorporates artificial masking of inputs. If require_input is True,
        ensures that at least one variable remains unmasked.
    
        Args:
            idx (int): The index of the item to be fetched.
    
        Returns:
            list: A list containing elements from Xcat, Xord, Xnum, a mask for
                numerical variables, and y_data at the given index, with
                artificial masking applied.
        """
        orig_idx = idx // self.aug_mult
        item = []
        all_masked = True
        mask_collection = []
    
        # Apply artificial masking and append elements to the item
        for i, X in enumerate([self.Xcat, self.Xord, self.Xnum]):
            pre_mask = [self.Xcat_mask, self.Xord_mask, self.Xnum_mask][i]
            if X is not None:
                x = X[orig_idx, :].clone()
                if pre_mask is not None:
                    x[pre_mask[orig_idx, :]] = 0
                    
                mask_float = torch.rand_like(
                    x.to(torch.float32), 
                    device=self.device
                )
                mask = mask_float < self.mask_prob
                mask_collection.append(mask)
                x[mask] = 0
                item.append(x)
    
                # Check if all variables in the item are masked
                if (torch.any(mask == False) and 
                    torch.any(pre_mask[orig_idx, :] == False)):
                    all_masked = False
            else:
                item.append(None)
                mask_collection.append(None)
    
        # If all variables are masked and require_input is True, unmask a
        # random variable
        if self.require_input and all_masked:
            while all_masked:
                # Choose a random dataset (Xcat, Xord, Xnum)
                X_idx = torch.randint(
                    len([self.Xcat, self.Xord, self.Xnum]), 
                    (1,)
                ).item()
    
                # Skip if the chosen dataset is None
                if [self.Xcat, self.Xord, self.Xnum][X_idx] is None:
                    continue
    
                # Choose a random variable within the chosen dataset
                variable_idx = torch.randint(
                    [self.Xcat, self.Xord, self.Xnum][X_idx].shape[1], 
                    (1,)
                ).item()
    
                # Skip if the chosen variable is pre-masked
                masks = [self.Xcat_mask, self.Xord_mask, self.Xnum_mask]
                if masks[X_idx][orig_idx, variable_idx]:
                    continue
    
                # Unmask the chosen variable
                mask_collection[X_idx][variable_idx] = False
                data_sources = [self.Xcat, self.Xord, self.Xnum]
                item[X_idx][variable_idx] =\
                    data_sources[X_idx][orig_idx, variable_idx]
    
                all_masked = False
    
        # Append mask of numerical variables to item
        if self.Xnum is not None:
            num_mask = ~mask_collection[2]
            item.append(num_mask)
        else:
            item.append(None)
    
        # If y_data exists, append the corresponding value to the item
        if self.y_data is not None:
            item.append(self.y_data[orig_idx].unsqueeze(0))
    
        return item

    def get_arrays(self):
        """Return the input arrays and target data.
    
        Returns:
            tuple: A four element tuple containing:
                - Xcat (torch.Tensor or None): Categorical input data 
                - Xord (torch.Tensor or None): Ordinal input data
                - Xnum (torch.Tensor or None): Numerical input data
                - y_data (torch.Tensor or None): Target data
        """
        return (self.Xcat, self.Xord, self.Xnum, self.y_data)


class AnnEnsembleDataset(torch.utils.data.Dataset):
    """Dataset adapter for ANN ensemble training.
    
    This class wraps a MixedDataset to provide the format expected by
    PyTorch's DataLoader for neural network training. It combines
    the various feature tensors from MixedDataset and adjusts target
    values for PyTorch's 0-indexing.
    
    Args:
        mixed_dataset: A MixedDataset instance to wrap.
        allow_missing_y_values: If False (default), raises an error when 
            encountering missing target values. If True, allows missing
            targets.
    """
    def __init__(self, mixed_dataset, allow_missing_y_values=False):
        self.mixed_dataset = mixed_dataset
        self.allow_missing_y_values = allow_missing_y_values
    
    def __len__(self):
        return len(self.mixed_dataset)

    def __getitem__(self, idx):
        # Get item from mixed_dataset including masks
        item = self.mixed_dataset[idx]
        
        # Unpack the item based on structure
        # MixedDataset.__getitem__ returns a list:
        # [Xcat, Xord, Xnum, Mnum, y_data]
        if len(item) != 5:
            raise ValueError(
                f"Expected 5 elements in item, but got {len(item)}. Perhaps "
                "a y-variable needs to be set."
            )
        Xcat, Xord, Xnum, Mnum, y = item
        
        # Combine all features (including masks for missing values)
        features = []
        if Xcat is not None:
            features.append(Xcat.float())
        if Xord is not None:
            features.append(Xord.float())
        if Xnum is not None:
            features.append(Xnum.float())
        if Mnum is not None:
            features.append(Mnum.float())
        
        X_combined = torch.cat(features)
        
        # Check for missing target values (encoded as 0)
        if (y is not None and torch.any(y == 0) and
            not self.allow_missing_y_values):
            raise ValueError(
                f"Missing target value detected at index {idx}. "
                f"Set allow_missing_y_values=True to allow missing targets."
            )
        
        # Adjust y for PyTorch's 0-indexing (subtract 1)
        # Use squeeze() to ensure it's a 1D tensor
        if y is not None:
            y_adjusted = (y - 1).squeeze()
        else:
            y_adjusted = None
        
        return X_combined, y_adjusted
