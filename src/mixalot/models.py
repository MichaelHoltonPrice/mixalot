"""Model specification classes for fitting."""
from abc import ABC, abstractmethod
import json
from typing import Any, Dict, List, Optional


class SingleTargetModelSpec(ABC):
    """Abstract class for model specifications with a single target variable.
    
    This class defines the common structure and functionality for supervised 
    learning models with a single dependent variable. It handles basic
    validation and serialization while requiring subclasses to implement
    model-specific validation and behavior.
    
    Args:
        y_var: String specifying the dependent variable name.
        independent_vars: List of strings specifying the names of independent 
                          variables to include in the model.
        hyperparameters: Optional dictionary containing model-specific 
                         hyperparameters.
    """
    def __init__(self,
                 y_var: str,
                 independent_vars: List[str],
                 hyperparameters: Optional[Dict[str, Any]] = None):
        self.y_var = y_var
        self.independent_vars = independent_vars
        self.hyperparameters = hyperparameters or {}
        
        # Basic validations that don't require DatasetSpec
        self._validate_structure()
    
    def _validate_structure(self):
        """Validates the basic structure of the model specification.
        
        Checks that:
        1. The y_var is not empty
        2. There is at least one independent variable
        3. No variable is used as both dependent and independent
        
        Raises:
            ValueError: If any of the basic validation checks fail.
        """
        if not self.y_var:
            raise ValueError("Dependent variable (y_var) must be specified")
        
        if not self.independent_vars:
            raise ValueError(
                "At least one independent variable must be specified"
            )
        
        if self.y_var in self.independent_vars:
            raise ValueError(f"Variable '{self.y_var}' cannot be used as both "
                             f"dependent and independent variable")
    
    def validate_with_dataset_spec(self, dataset_spec):
        """Validates that the model specification is compatible with a dataset.
        
        Performs common validations and then calls model-specific validation.
        
        Args:
            dataset_spec: A DatasetSpec object containing variable
                specifications.
            
        Raises:
            ValueError: If y_var or any independent variable is not found in
                the dataset_spec or if model-specific validations fail.
        """
        # Check that y_var exists in dataset_spec
        all_var_names = dataset_spec.all_var_names
        if self.y_var not in all_var_names:
            raise ValueError(f"Dependent variable '{self.y_var}' not found in "
                             f"dataset specification")
        
        # Check that all independent variables exist in dataset_spec
        for var in self.independent_vars:
            if var not in all_var_names:
                raise ValueError(f"Independent variable '{var}' not found in "
                                 f"dataset specification")
        
        # Call model-specific validation
        self._validate_model_specific(dataset_spec)
    
    @abstractmethod
    def _validate_model_specific(self, dataset_spec):
        """Performs model-specific validation against a dataset specification.
        
        This method must be implemented by subclasses to check that the model
        specification is compatible with the dataset for the specific model
        type.
        
        Args:
            dataset_spec: A DatasetSpec object containing variable
                specifications.
            
        Raises:
            ValueError: If model-specific validations fail.
        """
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Returns the type of model as a string.
        
        This property must be implemented by subclasses to identify the model
        type.
        
        Returns:
            str: A string identifier for the model type.
        """
        pass
    
    @abstractmethod
    def validate_hyperparameters(self):
        """Validates that the hyperparameters are appropriate for model type.
        
        This method must be implemented by subclasses to validate the
        hyperparameters specific to each model type.
        
        Raises:
            ValueError: If the hyperparameters are invalid for the model type.
        """
        pass

    def to_dict(self):
        """Converts the model specification to a dictionary.
        
        Returns:
            dict: A dictionary representation of the model specification.
        """
        return {
            'model_type': self.model_type,
            'y_var': self.y_var,
            'independent_vars': self.independent_vars,
            'hyperparameters': self.hyperparameters
        }
    
    def to_json(self, json_path):
        """Saves the model specification to a JSON file.
        
        Args:
            json_path: Path where the JSON file will be saved.
        """
        with open(json_path, 'w') as file:
            json.dump(self.to_dict(), file, indent=4)
    
    @classmethod
    def from_dict(cls, model_dict):
        """Create appropriate model specification object from a dictionary.
        
        This is a factory method that creates the appropriate subclass instance
        based on the model_type field in the dictionary.
        
        Args:
            model_dict: Dictionary containing model specification.
            
        Returns:
            SingleTargetModelSpec: A new model specification object.
            
        Raises:
            ValueError: If the model_type is not recognized.
        """
        # This method needs to be implemented after subclasses are defined
        # It would use a registry pattern to map model_type strings to
        # subclasses
        raise NotImplementedError(
            "Subclasses should implement this method or use "
            "a registry pattern to handle different model types"
        )
    
    @classmethod
    def from_json(cls, json_path):
        """Creates a model specification object from a JSON file.
        
        Args:
            json_path: Path to the JSON file containing the model
                specification.
            
        Returns:
            SingleTargetModelSpec: A new model specification object.
        """
        with open(json_path, 'r') as file:
            model_dict = json.load(file)
        
        return cls.from_dict(model_dict)


class RandomForestSpec(SingleTargetModelSpec):
    """Model specification for Random Forest models.
    
    Args:
        y_var: String specifying the dependent variable name.
        independent_vars: List of strings specifying the names of independent 
            variables to include in the model.
        hyperparameters: Optional dictionary containing RandomForest-specific 
            hyperparameters. Supported parameters include:
                - n_estimators: Number of trees (int)
                - max_features: Features to consider per split.
                    Can be one of:
                        * A string ('sqrt', 'log2', 'auto')
                        * An object with 'type': 'int' and 'value': <int>
                        * An object with 'type': 'float' and 'value': <float>
    """
    
    @property
    def model_type(self) -> str:
        """Returns the type of model.
        
        Returns:
            str: 'random_forest'
        """
        return 'random_forest'
    
    def _validate_model_specific(self, dataset_spec):
        """Performs Random Forest specific validation.
        
        For Random Forest, checks that:
        1. The hyperparameters are valid for Random Forest
        
        Args:
            dataset_spec: A DatasetSpec object containing variable
                specifications.
            
        Raises:
            ValueError: If model-specific validations fail.
        """
        # Validate hyperparameters
        self.validate_hyperparameters()

    def validate_hyperparameters(self):
        """Validates Random Forest specific hyperparameters.
        
        Checks that:
        1. n_estimators is present and a positive integer
        2. max_features is present with specified type and valid value
        
        Raises:
            ValueError: If any hyperparameter is invalid.
        """
        # Check required n_estimators
        if 'n_estimators' not in self.hyperparameters:
            raise ValueError("n_estimators is required for RandomForestSpec")
            
        n_estimators = self.hyperparameters['n_estimators']
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")
        
        # Check required max_features
        if 'max_features' not in self.hyperparameters:
            raise ValueError("max_features is required for RandomForestSpec")
            
        max_features = self.hyperparameters['max_features']
        
        # max_features must be a dictionary with type and value
        if not isinstance(max_features, dict):
            raise ValueError("max_features must be an object with 'type' and "
                            "'value' fields")
        
        if 'type' not in max_features or 'value' not in max_features:
            raise ValueError(
                "max_features must have 'type' and 'value' fields"
            )
        
        feat_type = max_features['type']
        feat_value = max_features['value']
        
        if feat_type == 'int':
            if not isinstance(feat_value, int) or feat_value <= 0:
                raise ValueError("max_features of type 'int' must have a "
                               "positive integer value")
        elif feat_type == 'float':
            if not isinstance(feat_value,
                              (int, float)) or not 0 < feat_value <= 1:
                raise ValueError("max_features of type 'float' must have a "
                               "value between 0 and 1")
        elif feat_type == 'string':
            raise ValueError("max_features 'type' must be 'int', 'float', "
                            "or 'string'")
